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

#define EASYSIMD_TEST_X86_AVX512_INSN sub

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/sub.h>

static int
test_easysimd_mm_mask_sub_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t src[16];
    const uint16_t k;
    const int8_t a[16];
    const int8_t b[16];
    const int8_t r[16];
  } test_vec[] = {
    { { -INT8_C(  26), -INT8_C(   7),  INT8_C( 114),  INT8_C( 100), -INT8_C(  47),  INT8_C( 120), -INT8_C(  25), -INT8_C(  91),
        -INT8_C(  92), -INT8_C(  89),  INT8_C(  39),  INT8_C(  99),  INT8_C(  34), -INT8_C( 107), -INT8_C(  56),  INT8_C(  79) },
      UINT16_C(13513),
      {  INT8_C( 125), -INT8_C(  10), -INT8_C(  69),  INT8_C(  66),  INT8_C( 121),  INT8_C(   6),      INT8_MAX, -INT8_C(  77),
         INT8_C(  33), -INT8_C( 113),  INT8_C(  37),  INT8_C(  54), -INT8_C( 126),  INT8_C(  11),  INT8_C(  47), -INT8_C(  12) },
      {  INT8_C( 111),  INT8_C(   0),  INT8_C( 109),  INT8_C(  86), -INT8_C(  91),  INT8_C(  17), -INT8_C(   3), -INT8_C(  52),
         INT8_C( 116),  INT8_C(  31),  INT8_C(  97),  INT8_C(  61),  INT8_C( 110),  INT8_C(  42),  INT8_C( 113), -INT8_C(  21) },
      {  INT8_C(  14), -INT8_C(   7),  INT8_C( 114), -INT8_C(  20), -INT8_C(  47),  INT8_C( 120), -INT8_C( 126), -INT8_C(  25),
        -INT8_C(  92), -INT8_C(  89), -INT8_C(  60),  INT8_C(  99),  INT8_C(  20), -INT8_C(  31), -INT8_C(  56),  INT8_C(  79) } },
    { {  INT8_C(  32),  INT8_C(  45),  INT8_C(  46), -INT8_C( 102),  INT8_C(  51), -INT8_C(  83),  INT8_C(  77),  INT8_C(  84),
         INT8_C(  60),  INT8_C( 114), -INT8_C( 118), -INT8_C(  65),  INT8_C( 126), -INT8_C(  70), -INT8_C(  77), -INT8_C(  19) },
      UINT16_C( 8378),
      {  INT8_C(  68),  INT8_C(  96),  INT8_C(  49),  INT8_C(  65),  INT8_C(  44), -INT8_C(  90),  INT8_C(  97), -INT8_C( 114),
        -INT8_C(  29), -INT8_C(  49), -INT8_C(  72),  INT8_C(  84), -INT8_C(  69), -INT8_C(  39), -INT8_C( 127), -INT8_C(  23) },
      {  INT8_C( 115), -INT8_C(  76), -INT8_C( 106), -INT8_C(  64),  INT8_C(   9), -INT8_C(  46),  INT8_C(  50), -INT8_C( 109),
        -INT8_C( 111), -INT8_C(  80),  INT8_C(  77),  INT8_C(  69), -INT8_C(  98),  INT8_C(   8),  INT8_C( 101), -INT8_C(  30) },
      {  INT8_C(  32), -INT8_C(  84),  INT8_C(  46), -INT8_C( 127),  INT8_C(  35), -INT8_C(  44),  INT8_C(  77), -INT8_C(   5),
         INT8_C(  60),  INT8_C( 114), -INT8_C( 118), -INT8_C(  65),  INT8_C( 126), -INT8_C(  47), -INT8_C(  77), -INT8_C(  19) } },
    { {  INT8_C( 104), -INT8_C( 105),  INT8_C(  35), -INT8_C( 108),  INT8_C(  61), -INT8_C( 124),  INT8_C(  34),  INT8_C(  32),
         INT8_C(  84), -INT8_C(  37),  INT8_C( 116),  INT8_C(  15), -INT8_C(  76), -INT8_C(  10), -INT8_C(   8),  INT8_C(  39) },
      UINT16_C(36522),
      { -INT8_C(  25), -INT8_C(  77),  INT8_C(  96),  INT8_C(  25),  INT8_C(  71), -INT8_C(  14), -INT8_C(  54), -INT8_C( 108),
         INT8_C(  55),  INT8_C( 104), -INT8_C( 100), -INT8_C( 100),  INT8_C(  74),  INT8_C(   4),  INT8_C(  51),  INT8_C( 109) },
      { -INT8_C( 103),  INT8_C( 112), -INT8_C(  14), -INT8_C(  69), -INT8_C( 112),  INT8_C(  70), -INT8_C( 106),  INT8_C(   5),
         INT8_C(  85),  INT8_C(  74), -INT8_C(   5),  INT8_C(  77),  INT8_C( 113), -INT8_C(  91), -INT8_C(  37),  INT8_C(  88) },
      {  INT8_C( 104),  INT8_C(  67),  INT8_C(  35),  INT8_C(  94),  INT8_C(  61), -INT8_C(  84),  INT8_C(  34), -INT8_C( 113),
         INT8_C(  84),  INT8_C(  30), -INT8_C(  95),  INT8_C(  79), -INT8_C(  76), -INT8_C(  10), -INT8_C(   8),  INT8_C(  21) } },
    { {  INT8_C(  89),  INT8_C(  59),  INT8_C( 114), -INT8_C(  96),  INT8_C(  45),  INT8_C(  60),  INT8_C(  52),  INT8_C( 100),
        -INT8_C(  92), -INT8_C(  47),  INT8_C(   1), -INT8_C(  18), -INT8_C(  43),  INT8_C(  52),  INT8_C(  91),  INT8_C( 110) },
      UINT16_C(19877),
      {  INT8_C(  42),  INT8_C(  53), -INT8_C( 109), -INT8_C(  64),  INT8_C(  58), -INT8_C(  24),  INT8_C(  11),  INT8_C(  53),
         INT8_C(  53),  INT8_C( 124), -INT8_C(  37),  INT8_C(  16), -INT8_C(  43),  INT8_C(  52),  INT8_C(  76),  INT8_C(  71) },
      { -INT8_C(  44),  INT8_C( 121), -INT8_C( 125),  INT8_C(   8), -INT8_C(  34),  INT8_C(  39), -INT8_C(  39), -INT8_C(  33),
         INT8_C(  21), -INT8_C(  81),  INT8_C(  19),  INT8_C( 112),  INT8_C(  29), -INT8_C(  72), -INT8_C(  66),  INT8_C(  71) },
      {  INT8_C(  86),  INT8_C(  59),  INT8_C(  16), -INT8_C(  96),  INT8_C(  45), -INT8_C(  63),  INT8_C(  52),  INT8_C(  86),
         INT8_C(  32), -INT8_C(  47), -INT8_C(  56), -INT8_C(  96), -INT8_C(  43),  INT8_C(  52), -INT8_C( 114),  INT8_C( 110) } },
    { { -INT8_C(  18),  INT8_C(  81),  INT8_C(   8),  INT8_C(  40),  INT8_C(  58),  INT8_C(  19),  INT8_C(  94),  INT8_C( 111),
        -INT8_C( 113),  INT8_C(  57),      INT8_MIN,  INT8_C( 100),  INT8_C( 109), -INT8_C(  52), -INT8_C(  85),  INT8_C(  65) },
      UINT16_C(11845),
      {  INT8_C(  73),  INT8_C(  35),  INT8_C(  85),  INT8_C(  35),  INT8_C(   2),  INT8_C( 106), -INT8_C(  46),  INT8_C(  22),
        -INT8_C(  37), -INT8_C(  17), -INT8_C(  50), -INT8_C( 103),  INT8_C(  55), -INT8_C(  68), -INT8_C(  22),  INT8_C(  63) },
      { -INT8_C(  27),  INT8_C(  36),  INT8_C(  82),  INT8_C(  67), -INT8_C( 108), -INT8_C(  31),  INT8_C( 124),  INT8_C(  20),
         INT8_C(  70), -INT8_C(  23), -INT8_C(  32), -INT8_C(  15),  INT8_C(  42),  INT8_C(  37),  INT8_C(  32),  INT8_C( 115) },
      {  INT8_C( 100),  INT8_C(  81),  INT8_C(   3),  INT8_C(  40),  INT8_C(  58),  INT8_C(  19),  INT8_C(  86),  INT8_C( 111),
        -INT8_C( 113),  INT8_C(   6), -INT8_C(  18), -INT8_C(  88),  INT8_C( 109), -INT8_C( 105), -INT8_C(  85),  INT8_C(  65) } },
    { {  INT8_C(  73),  INT8_C( 117), -INT8_C( 106),  INT8_C(  75), -INT8_C(  32),  INT8_C( 104),  INT8_C(  97), -INT8_C(  69),
         INT8_C(  88),  INT8_C(  48),  INT8_C(  84), -INT8_C( 113), -INT8_C(  20),  INT8_C(  62), -INT8_C(  50), -INT8_C(  47) },
      UINT16_C( 8291),
      {  INT8_C(  20), -INT8_C(   9),  INT8_C(   1), -INT8_C( 112),  INT8_C(  11),  INT8_C(  71),  INT8_C( 121), -INT8_C(  21),
         INT8_C(  57), -INT8_C(  93),  INT8_C(  16),  INT8_C(  89),  INT8_C(  23),  INT8_C(  89), -INT8_C(  50), -INT8_C(  83) },
      { -INT8_C(  91), -INT8_C(  82),  INT8_C(  22),  INT8_C(   6),  INT8_C( 105),  INT8_C( 110),  INT8_C(  54), -INT8_C(  67),
        -INT8_C(   3),  INT8_C(  35), -INT8_C(   4), -INT8_C(  53), -INT8_C(  12),  INT8_C(  95), -INT8_C(  21),  INT8_C(   9) },
      {  INT8_C( 111),  INT8_C(  73), -INT8_C( 106),  INT8_C(  75), -INT8_C(  32), -INT8_C(  39),  INT8_C(  67), -INT8_C(  69),
         INT8_C(  88),  INT8_C(  48),  INT8_C(  84), -INT8_C( 113), -INT8_C(  20), -INT8_C(   6), -INT8_C(  50), -INT8_C(  47) } },
    { {  INT8_C(  86), -INT8_C(  20), -INT8_C( 103),  INT8_C(  97),  INT8_C(  52),  INT8_C(  19),  INT8_C(  76),  INT8_C( 109),
        -INT8_C(  74),  INT8_C(  92), -INT8_C(  58), -INT8_C(  51), -INT8_C(  74), -INT8_C( 108),  INT8_C( 123),  INT8_C(  91) },
      UINT16_C(37187),
      {  INT8_C(  97), -INT8_C(  84), -INT8_C(   1), -INT8_C( 104),  INT8_C( 106), -INT8_C(   4), -INT8_C(  69),  INT8_C( 102),
        -INT8_C(  57), -INT8_C(  81), -INT8_C(  59), -INT8_C(  78), -INT8_C(  72),  INT8_C(  27), -INT8_C(  98),  INT8_C(  82) },
      {  INT8_C( 124), -INT8_C(  46),  INT8_C( 101), -INT8_C(  56),  INT8_C(  63),  INT8_C(  27),  INT8_C(  36),  INT8_C(   5),
        -INT8_C(  23), -INT8_C(  38), -INT8_C( 102),  INT8_C( 100),  INT8_C(  53), -INT8_C(  35), -INT8_C(  11), -INT8_C( 105) },
      { -INT8_C(  27), -INT8_C(  38), -INT8_C( 103),  INT8_C(  97),  INT8_C(  52),  INT8_C(  19), -INT8_C( 105),  INT8_C( 109),
        -INT8_C(  34),  INT8_C(  92), -INT8_C(  58), -INT8_C(  51), -INT8_C( 125), -INT8_C( 108),  INT8_C( 123), -INT8_C(  69) } },
    { { -INT8_C( 119), -INT8_C(  12),  INT8_C(  47), -INT8_C(  13), -INT8_C(  16), -INT8_C(  22),  INT8_C(  89), -INT8_C(  73),
        -INT8_C( 103),  INT8_C(  30),  INT8_C( 105),  INT8_C(  82),  INT8_C(  57),  INT8_C(   7), -INT8_C(  92), -INT8_C(  75) },
      UINT16_C( 2522),
      {  INT8_C( 125),  INT8_C(  25),  INT8_C(  36), -INT8_C(  94),  INT8_C(  31),  INT8_C(  13),  INT8_C( 124), -INT8_C(  71),
         INT8_C( 113), -INT8_C(  78), -INT8_C( 106),  INT8_C( 102),  INT8_C(  73),  INT8_C(  31),  INT8_C(  90),  INT8_C( 120) },
      {  INT8_C(  19),  INT8_C(  74),  INT8_C(  98),  INT8_C( 108),  INT8_C(   1), -INT8_C(   5), -INT8_C( 117),  INT8_C( 106),
         INT8_C(  77), -INT8_C(  60),  INT8_C( 114), -INT8_C(  15),  INT8_C( 122),  INT8_C(  76), -INT8_C(   6), -INT8_C(   9) },
      { -INT8_C( 119), -INT8_C(  49),  INT8_C(  47),  INT8_C(  54),  INT8_C(  30), -INT8_C(  22), -INT8_C(  15),  INT8_C(  79),
         INT8_C(  36),  INT8_C(  30),  INT8_C( 105),  INT8_C( 117),  INT8_C(  57),  INT8_C(   7), -INT8_C(  92), -INT8_C(  75) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi8(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sub_epi8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_sub_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__m128i r = easysimd_mm_mask_sub_epi8(src, k, a, b);

    easysimd_test_x86_write_i8x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_sub_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const int8_t a[16];
    const int8_t b[16];
    const int8_t r[16];
  } test_vec[] = {
    { UINT16_C(19993),
      { -INT8_C(  59),  INT8_C(  16), -INT8_C( 100),  INT8_C(  39), -INT8_C( 107),  INT8_C(  11),  INT8_C( 116), -INT8_C(  10),
        -INT8_C(  71),  INT8_C(  29), -INT8_C(  97), -INT8_C(  98),  INT8_C( 103), -INT8_C(  82), -INT8_C(  85),  INT8_C(  87) },
      { -INT8_C(  78), -INT8_C(  99), -INT8_C( 121),  INT8_C(  78),  INT8_C( 104), -INT8_C(  78), -INT8_C(   9), -INT8_C(  89),
         INT8_C( 106),  INT8_C(  74), -INT8_C(  91), -INT8_C(  26),  INT8_C( 117), -INT8_C(  66),  INT8_C(  53),  INT8_C(  58) },
      {  INT8_C(  19),  INT8_C(   0),  INT8_C(   0), -INT8_C(  39),  INT8_C(  45),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  45), -INT8_C(   6), -INT8_C(  72),  INT8_C(   0),  INT8_C(   0),  INT8_C( 118),  INT8_C(   0) } },
    { UINT16_C(53710),
      {  INT8_C(  97),  INT8_C( 100), -INT8_C(  35), -INT8_C(  43),  INT8_C(  90), -INT8_C( 106), -INT8_C(  14), -INT8_C(   7),
         INT8_C(  52),  INT8_C(  89), -INT8_C(  89), -INT8_C(  33), -INT8_C(  79),  INT8_C(  89),  INT8_C( 125),  INT8_C(  56) },
      { -INT8_C(  88), -INT8_C(  27), -INT8_C(  21), -INT8_C(  97), -INT8_C( 116),  INT8_C(  85), -INT8_C(  23),  INT8_C(  49),
         INT8_C(  59),  INT8_C(  94), -INT8_C(  17),  INT8_C( 112), -INT8_C( 104), -INT8_C(  66),  INT8_C(  66), -INT8_C(   6) },
      {  INT8_C(   0),      INT8_MAX, -INT8_C(  14),  INT8_C(  54),  INT8_C(   0),  INT8_C(   0),  INT8_C(   9), -INT8_C(  56),
        -INT8_C(   7),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  25),  INT8_C(   0),  INT8_C(  59),  INT8_C(  62) } },
    { UINT16_C( 7970),
      { -INT8_C(  49),  INT8_C( 124), -INT8_C(  75), -INT8_C(  62),  INT8_C( 117), -INT8_C(  23),  INT8_C(  27),  INT8_C(  29),
        -INT8_C(  55), -INT8_C(  52),  INT8_C( 118),  INT8_C(  70),  INT8_C(   5),  INT8_C(  30),  INT8_C(  43), -INT8_C(  16) },
      { -INT8_C(  67), -INT8_C(  73),  INT8_C(  69), -INT8_C(  89), -INT8_C(  23),      INT8_MIN,  INT8_C(   5), -INT8_C(  40),
        -INT8_C(  15), -INT8_C(  98), -INT8_C( 106),  INT8_C(  51), -INT8_C( 104), -INT8_C(  72),  INT8_C(  82),  INT8_C( 103) },
      {  INT8_C(   0), -INT8_C(  59),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 105),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  40),  INT8_C(  46), -INT8_C(  32),  INT8_C(  19),  INT8_C( 109),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C( 1844),
      {  INT8_C(  41), -INT8_C(  86), -INT8_C(  16),  INT8_C(  69), -INT8_C(  57), -INT8_C(  71),  INT8_C(  17),  INT8_C(  61),
        -INT8_C(   1),  INT8_C(  22),  INT8_C(  92),  INT8_C(  42),  INT8_C(   6),  INT8_C(  25), -INT8_C(  30),  INT8_C(  75) },
      { -INT8_C(  64), -INT8_C(  53), -INT8_C(  52), -INT8_C(  58), -INT8_C(  93), -INT8_C(  67),  INT8_C( 100),  INT8_C(  58),
        -INT8_C(  16), -INT8_C(   4), -INT8_C(  14),  INT8_C(  66),  INT8_C(  99),  INT8_C(  39),  INT8_C(  73), -INT8_C( 115) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(  36),  INT8_C(   0),  INT8_C(  36), -INT8_C(   4),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  15),  INT8_C(  26),  INT8_C( 106),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(14801),
      { -INT8_C(  46), -INT8_C( 104), -INT8_C(  13), -INT8_C(  29), -INT8_C(  43), -INT8_C(  14), -INT8_C(   6),  INT8_C(  49),
         INT8_C(  29),  INT8_C(   0),  INT8_C(  75), -INT8_C(   1),  INT8_C(  76),  INT8_C(  11), -INT8_C(  54),  INT8_C(  24) },
      { -INT8_C(  47),  INT8_C( 109), -INT8_C(  43),  INT8_C(  53), -INT8_C(  89), -INT8_C(  59),  INT8_C(  49), -INT8_C( 102),
         INT8_C(   7), -INT8_C( 107), -INT8_C(  63),  INT8_C(  80),  INT8_C(  34), -INT8_C( 110), -INT8_C( 119), -INT8_C(  12) },
      {  INT8_C(   1),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  46),  INT8_C(   0), -INT8_C(  55), -INT8_C( 105),
         INT8_C(  22),  INT8_C(   0),  INT8_C(   0), -INT8_C(  81),  INT8_C(  42),  INT8_C( 121),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(31786),
      { -INT8_C(  41), -INT8_C(   1),  INT8_C( 111), -INT8_C(  47),  INT8_C(  49), -INT8_C( 116), -INT8_C(  46),  INT8_C( 124),
        -INT8_C( 117),  INT8_C(  30), -INT8_C( 121),  INT8_C(  85),  INT8_C(  54),  INT8_C(  89), -INT8_C(  62),  INT8_C(  11) },
      { -INT8_C( 114),  INT8_C( 106), -INT8_C(  48), -INT8_C(  64),  INT8_C(   4), -INT8_C(  41),  INT8_C(  85), -INT8_C(  59),
         INT8_C(  39),  INT8_C( 119),  INT8_C(  87), -INT8_C(  80),  INT8_C( 107), -INT8_C( 127),  INT8_C(  45),  INT8_C(  66) },
      {  INT8_C(   0), -INT8_C( 107),  INT8_C(   0),  INT8_C(  17),  INT8_C(   0), -INT8_C(  75),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  48), -INT8_C(  91), -INT8_C(  53), -INT8_C(  40), -INT8_C( 107),  INT8_C(   0) } },
    { UINT16_C(40064),
      {  INT8_C(  20), -INT8_C(  79),  INT8_C(  40), -INT8_C(  26),  INT8_C(  45), -INT8_C(  77),  INT8_C(   4), -INT8_C(  75),
         INT8_C(   8),  INT8_C(  58),  INT8_C(  14), -INT8_C(  54),  INT8_C(  69), -INT8_C( 100),  INT8_C(  52),  INT8_C(  21) },
      {  INT8_C(  92),  INT8_C(  56), -INT8_C(  20), -INT8_C(  79), -INT8_C(   3),  INT8_C(  19),  INT8_C(  40),  INT8_C(  84),
        -INT8_C(  61), -INT8_C( 109), -INT8_C(  43), -INT8_C(  16), -INT8_C(  42),  INT8_C(  86), -INT8_C( 116), -INT8_C(  22) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  97),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  57), -INT8_C(  38),  INT8_C( 111),  INT8_C(   0),  INT8_C(   0),  INT8_C(  43) } },
    { UINT16_C(46087),
      { -INT8_C(  48),  INT8_C(  53),  INT8_C( 103), -INT8_C(  44), -INT8_C(  22),  INT8_C( 111),  INT8_C(  14), -INT8_C(   8),
         INT8_C(  58),  INT8_C(  83), -INT8_C( 108),  INT8_C( 110),  INT8_C( 104), -INT8_C(  15), -INT8_C(  89),  INT8_C(  84) },
      { -INT8_C(  94), -INT8_C(  92),  INT8_C( 103), -INT8_C(  53), -INT8_C(   7),  INT8_C(  42),  INT8_C(  94), -INT8_C(  50),
         INT8_C(  27),  INT8_C(  52),  INT8_C(  36), -INT8_C(  89),  INT8_C(  30),  INT8_C(  44),  INT8_C(  92), -INT8_C(  18) },
      {  INT8_C(  46), -INT8_C( 111),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C( 112),  INT8_C(   0),  INT8_C(  74), -INT8_C(  59),  INT8_C(   0),  INT8_C( 102) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sub_epi8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_sub_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__m128i r = easysimd_mm_maskz_sub_epi8(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_sub_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { {  INT16_C(  8037), -INT16_C( 31591),  INT16_C(  5676), -INT16_C( 25027), -INT16_C( 11320),  INT16_C(  4356),  INT16_C( 24563),  INT16_C(  1673) },
      UINT8_C(169),
      {  INT16_C( 29419), -INT16_C(  6485),  INT16_C(  5629), -INT16_C( 15820),  INT16_C(  9607), -INT16_C( 11460),  INT16_C( 13088),  INT16_C( 16185) },
      { -INT16_C( 16947), -INT16_C(  7317),  INT16_C(  2555), -INT16_C( 12629), -INT16_C( 17394),  INT16_C( 28097), -INT16_C( 14523),  INT16_C( 12310) },
      { -INT16_C( 19170), -INT16_C( 31591),  INT16_C(  5676), -INT16_C(  3191), -INT16_C( 11320),  INT16_C( 25979),  INT16_C( 24563),  INT16_C(  3875) } },
    { { -INT16_C( 16070),  INT16_C( 14102),  INT16_C( 19159),  INT16_C( 24313),  INT16_C( 13680), -INT16_C( 28622),  INT16_C( 27497),  INT16_C( 14031) },
      UINT8_C( 40),
      {  INT16_C(  6458),  INT16_C( 17443), -INT16_C(  3388), -INT16_C( 32686), -INT16_C( 16461),  INT16_C( 31685), -INT16_C(  2603), -INT16_C( 26699) },
      { -INT16_C(  5109),  INT16_C( 22126), -INT16_C( 13082),  INT16_C(  7110),  INT16_C( 22270),  INT16_C( 27012), -INT16_C( 17883),  INT16_C( 24466) },
      { -INT16_C( 16070),  INT16_C( 14102),  INT16_C( 19159),  INT16_C( 25740),  INT16_C( 13680),  INT16_C(  4673),  INT16_C( 27497),  INT16_C( 14031) } },
    { { -INT16_C( 18989), -INT16_C( 26717), -INT16_C(  2649),  INT16_C( 23319), -INT16_C(  9036), -INT16_C( 29994), -INT16_C( 29743), -INT16_C(  8927) },
      UINT8_C(119),
      {  INT16_C( 13199),  INT16_C( 23389),  INT16_C( 31225),  INT16_C( 20314), -INT16_C( 15363), -INT16_C( 18316), -INT16_C( 11435),  INT16_C(  2955) },
      {  INT16_C(  9079),  INT16_C( 27826),  INT16_C(  3386),  INT16_C(  5921), -INT16_C( 21533),  INT16_C( 28392), -INT16_C( 14900),  INT16_C( 23526) },
      {  INT16_C(  4120), -INT16_C(  4437),  INT16_C( 27839),  INT16_C( 23319),  INT16_C(  6170),  INT16_C( 18828),  INT16_C(  3465), -INT16_C(  8927) } },
    { {  INT16_C( 17400), -INT16_C(  3658),  INT16_C(  4284), -INT16_C( 17856), -INT16_C( 19244),  INT16_C( 10610), -INT16_C(   632), -INT16_C(   204) },
      UINT8_C( 32),
      {  INT16_C( 27623), -INT16_C(  2981),  INT16_C( 29324),  INT16_C( 14296),  INT16_C( 18010),  INT16_C(  8195),  INT16_C( 24108),  INT16_C( 28696) },
      {  INT16_C(  2581),  INT16_C(  9516), -INT16_C(  6582), -INT16_C(     7),  INT16_C(  9048),  INT16_C( 22151), -INT16_C( 31145),  INT16_C( 15990) },
      {  INT16_C( 17400), -INT16_C(  3658),  INT16_C(  4284), -INT16_C( 17856), -INT16_C( 19244), -INT16_C( 13956), -INT16_C(   632), -INT16_C(   204) } },
    { { -INT16_C( 11791),  INT16_C( 32307),  INT16_C(  2883), -INT16_C( 24907), -INT16_C( 18095),  INT16_C( 32446), -INT16_C( 10729),  INT16_C( 11502) },
      UINT8_C(224),
      {  INT16_C( 21018),  INT16_C(   299),  INT16_C( 10827),  INT16_C( 28249), -INT16_C( 20559),  INT16_C( 14278),  INT16_C(  1062), -INT16_C(  2264) },
      { -INT16_C( 22985),  INT16_C( 16955), -INT16_C(  9892),  INT16_C(  5524),  INT16_C(  4759),  INT16_C( 27948),  INT16_C( 22784),  INT16_C(  6734) },
      { -INT16_C( 11791),  INT16_C( 32307),  INT16_C(  2883), -INT16_C( 24907), -INT16_C( 18095), -INT16_C( 13670), -INT16_C( 21722), -INT16_C(  8998) } },
    { {  INT16_C( 31147), -INT16_C(  2533),  INT16_C( 30115),  INT16_C( 21605),  INT16_C( 11044),  INT16_C( 19083), -INT16_C( 19665),  INT16_C( 26434) },
      UINT8_C( 90),
      { -INT16_C( 22147),  INT16_C( 22198), -INT16_C( 13507),  INT16_C( 20461),  INT16_C( 23287),  INT16_C( 20559),  INT16_C( 27304),  INT16_C(  8699) },
      { -INT16_C(  3451), -INT16_C(  1340),  INT16_C(  6231), -INT16_C( 32225),  INT16_C( 27043),  INT16_C( 22449),  INT16_C(  6315),  INT16_C( 10417) },
      {  INT16_C( 31147),  INT16_C( 23538),  INT16_C( 30115), -INT16_C( 12850), -INT16_C(  3756),  INT16_C( 19083),  INT16_C( 20989),  INT16_C( 26434) } },
    { {  INT16_C( 26562), -INT16_C(   130),  INT16_C( 27442),  INT16_C( 10575), -INT16_C( 24890),  INT16_C( 28282),  INT16_C( 29960), -INT16_C( 29040) },
      UINT8_C(103),
      { -INT16_C( 30636),  INT16_C( 28094),  INT16_C( 16551),  INT16_C(  4368),  INT16_C( 26610),  INT16_C(  2748), -INT16_C(  6888),  INT16_C( 32716) },
      { -INT16_C( 13213), -INT16_C( 12367), -INT16_C(  9445), -INT16_C( 18027),  INT16_C(   853), -INT16_C( 13630),  INT16_C( 20627), -INT16_C(  6094) },
      { -INT16_C( 17423), -INT16_C( 25075),  INT16_C( 25996),  INT16_C( 10575), -INT16_C( 24890),  INT16_C( 16378), -INT16_C( 27515), -INT16_C( 29040) } },
    { { -INT16_C(  3880), -INT16_C( 32683),  INT16_C( 25905),  INT16_C(  9105),  INT16_C( 19917), -INT16_C(  6867), -INT16_C(  1486), -INT16_C( 27035) },
      UINT8_C(198),
      {  INT16_C( 25878), -INT16_C(  3615), -INT16_C( 25862), -INT16_C(   698),  INT16_C(  4444), -INT16_C( 21359),  INT16_C( 31043),  INT16_C( 13189) },
      {  INT16_C(  1486),  INT16_C( 13156), -INT16_C( 30826), -INT16_C(  7424), -INT16_C(  6475), -INT16_C( 20714), -INT16_C( 21429),  INT16_C( 24949) },
      { -INT16_C(  3880), -INT16_C( 16771),  INT16_C(  4964),  INT16_C(  9105),  INT16_C( 19917), -INT16_C(  6867), -INT16_C( 13064), -INT16_C( 11760) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sub_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_sub_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_mask_sub_epi16(src, k, a, b);

    easysimd_test_x86_write_i16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_sub_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C( 97),
      { -INT16_C( 15677),  INT16_C( 13131),  INT16_C( 17360),  INT16_C(  9069), -INT16_C(  9257), -INT16_C( 14197), -INT16_C(  8318),  INT16_C( 10091) },
      {  INT16_C( 13894),  INT16_C( 28960), -INT16_C(  4460), -INT16_C( 13940),  INT16_C( 13075),  INT16_C( 16359), -INT16_C( 10609),  INT16_C( 21408) },
      { -INT16_C( 29571),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 30556),  INT16_C(  2291),  INT16_C(     0) } },
    { UINT8_C(152),
      { -INT16_C( 30997),  INT16_C( 11881), -INT16_C( 29453), -INT16_C( 12795), -INT16_C( 12776), -INT16_C(  2223),  INT16_C( 30777),  INT16_C( 28478) },
      { -INT16_C( 20584), -INT16_C( 31229), -INT16_C( 13253),  INT16_C( 28313), -INT16_C( 10060), -INT16_C( 29954),  INT16_C( 20856),  INT16_C( 25378) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 24428), -INT16_C(  2716),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3100) } },
    { UINT8_C(215),
      { -INT16_C( 28277),  INT16_C(  6346), -INT16_C( 26473),  INT16_C( 25904),  INT16_C( 10217),  INT16_C( 24990),  INT16_C(  3429),  INT16_C(  5369) },
      { -INT16_C( 32752), -INT16_C(  8881), -INT16_C( 16871), -INT16_C(  3439),  INT16_C(  7100),  INT16_C(  3434), -INT16_C( 12739), -INT16_C( 13852) },
      {  INT16_C(  4475),  INT16_C( 15227), -INT16_C(  9602),  INT16_C(     0),  INT16_C(  3117),  INT16_C(     0),  INT16_C( 16168),  INT16_C( 19221) } },
    { UINT8_C( 95),
      { -INT16_C(  7762),  INT16_C( 18166),  INT16_C( 23313),  INT16_C( 14384), -INT16_C( 28167),  INT16_C(  1694), -INT16_C( 19829),  INT16_C(  2839) },
      { -INT16_C(  3070), -INT16_C( 16348),  INT16_C(  5765), -INT16_C( 24452), -INT16_C( 30335),  INT16_C( 20445), -INT16_C( 22931),  INT16_C(  7086) },
      { -INT16_C(  4692), -INT16_C( 31022),  INT16_C( 17548), -INT16_C( 26700),  INT16_C(  2168),  INT16_C(     0),  INT16_C(  3102),  INT16_C(     0) } },
    { UINT8_C(135),
      {  INT16_C( 24997),  INT16_C(   152), -INT16_C( 11887),  INT16_C(  9210),  INT16_C(   111),  INT16_C(  8622), -INT16_C( 18153),  INT16_C(  2851) },
      { -INT16_C(  7203), -INT16_C(  2928),  INT16_C( 12383), -INT16_C(  6027), -INT16_C( 15346), -INT16_C( 19371),  INT16_C( 28786),  INT16_C(  5948) },
      {  INT16_C( 32200),  INT16_C(  3080), -INT16_C( 24270),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  3097) } },
    { UINT8_C(210),
      {  INT16_C(  6356), -INT16_C( 23197), -INT16_C( 31214),  INT16_C(  4628),  INT16_C( 13876), -INT16_C(  4822),  INT16_C( 13657),  INT16_C( 15819) },
      { -INT16_C( 16442), -INT16_C(  2404), -INT16_C( 31436), -INT16_C(  2044), -INT16_C( 17958),  INT16_C( 19306), -INT16_C( 32011), -INT16_C( 14051) },
      {  INT16_C(     0), -INT16_C( 20793),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31834),  INT16_C(     0), -INT16_C( 19868),  INT16_C( 29870) } },
    { UINT8_C(154),
      {  INT16_C( 28544),  INT16_C(  1964), -INT16_C( 16765), -INT16_C( 18117),  INT16_C( 10728),  INT16_C(  7699),  INT16_C( 20724), -INT16_C( 19484) },
      { -INT16_C(  9492),  INT16_C( 29159), -INT16_C(  8225), -INT16_C( 26548), -INT16_C( 26807), -INT16_C( 13427),  INT16_C( 22196),  INT16_C( 13413) },
      {  INT16_C(     0), -INT16_C( 27195),  INT16_C(     0),  INT16_C(  8431), -INT16_C( 28001),  INT16_C(     0),  INT16_C(     0),  INT16_C( 32639) } },
    { UINT8_C(197),
      {  INT16_C( 15121), -INT16_C( 12215),  INT16_C(   631), -INT16_C( 24392), -INT16_C( 10731),  INT16_C( 26004),  INT16_C( 18362), -INT16_C( 27310) },
      { -INT16_C( 15570),  INT16_C(  3444),  INT16_C(  3087), -INT16_C( 22954),  INT16_C(  8857), -INT16_C(  4262), -INT16_C( 28793), -INT16_C( 26187) },
      {  INT16_C( 30691),  INT16_C(     0), -INT16_C(  2456),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 18381), -INT16_C(  1123) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sub_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_sub_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_maskz_sub_epi16(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_sub_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(   190010897),  INT32_C(  1292409328), -INT32_C(   302409302), -INT32_C(   534675950) },
      UINT8_C(131),
      {  INT32_C(   219747461), -INT32_C(    87884524), -INT32_C(  1085968109),  INT32_C(  1020307430) },
      { -INT32_C(  1825776646),  INT32_C(  2101246435), -INT32_C(   242275469), -INT32_C(   764055476) },
      {  INT32_C(  2045524107),  INT32_C(  2105836337), -INT32_C(   302409302), -INT32_C(   534675950) } },
    { { -INT32_C(  1730179452), -INT32_C(  1634492021), -INT32_C(   128067566),  INT32_C(  2033462655) },
      UINT8_C(  8),
      { -INT32_C(   622064288),  INT32_C(  1984784715),  INT32_C(  1757626104),  INT32_C(  1122801075) },
      {  INT32_C(   365790580), -INT32_C(   265851880), -INT32_C(   143646775), -INT32_C(  1275074221) },
      { -INT32_C(  1730179452), -INT32_C(  1634492021), -INT32_C(   128067566), -INT32_C(  1897092000) } },
    { {  INT32_C(  1099820022),  INT32_C(  1303960148), -INT32_C(   860521703), -INT32_C(  2079415792) },
      UINT8_C( 39),
      {  INT32_C(  1212127708), -INT32_C(   552521792), -INT32_C(  1993209697), -INT32_C(   209722104) },
      {  INT32_C(  1296613747), -INT32_C(   194603655), -INT32_C(   335269046),  INT32_C(   487819329) },
      { -INT32_C(    84486039), -INT32_C(   357918137), -INT32_C(  1657940651), -INT32_C(  2079415792) } },
    { { -INT32_C(   513453535),  INT32_C(   566261634), -INT32_C(  2018839937),  INT32_C(  1249585879) },
      UINT8_C(235),
      {  INT32_C(  1482987715),  INT32_C(   832723198), -INT32_C(   462188708), -INT32_C(   184184670) },
      {  INT32_C(  1836574454), -INT32_C(  1729324890),  INT32_C(  1836020803), -INT32_C(  1302807825) },
      { -INT32_C(   353586739), -INT32_C(  1732919208), -INT32_C(  2018839937),  INT32_C(  1118623155) } },
    { {  INT32_C(  1342880850),  INT32_C(  1887611924), -INT32_C(   564857541),  INT32_C(  2077448581) },
      UINT8_C( 63),
      { -INT32_C(   488249270),  INT32_C(  1227193813), -INT32_C(  1506241812), -INT32_C(  1493636374) },
      { -INT32_C(  1598404108), -INT32_C(  1059312949), -INT32_C(   683296130), -INT32_C(   686374515) },
      {  INT32_C(  1110154838), -INT32_C(  2008460534), -INT32_C(   822945682), -INT32_C(   807261859) } },
    { {  INT32_C(  2126117801),  INT32_C(  1707597688),  INT32_C(  1544290161), -INT32_C(   587070231) },
      UINT8_C( 77),
      { -INT32_C(   417825091),  INT32_C(   342284378), -INT32_C(   559858403),  INT32_C(  1334343764) },
      {  INT32_C(   281544241), -INT32_C(   847106610),  INT32_C(  1018682936), -INT32_C(  1651927840) },
      { -INT32_C(   699369332),  INT32_C(  1707597688), -INT32_C(  1578541339), -INT32_C(  1308695692) } },
    { {  INT32_C(  1820696850), -INT32_C(  1753158791),  INT32_C(  2088050984), -INT32_C(   875758183) },
      UINT8_C(  4),
      { -INT32_C(  1043145836),  INT32_C(  1006215005), -INT32_C(   350472618),  INT32_C(  1644018111) },
      {  INT32_C(   702179902),  INT32_C(   206729706), -INT32_C(   458895641),  INT32_C(   786985114) },
      {  INT32_C(  1820696850), -INT32_C(  1753158791),  INT32_C(   108423023), -INT32_C(   875758183) } },
    { { -INT32_C(  1443906996), -INT32_C(  1327109798), -INT32_C(   560266977), -INT32_C(   113272390) },
      UINT8_C(  3),
      { -INT32_C(  1947393510),  INT32_C(  1131608436),  INT32_C(   266164127), -INT32_C(    94696385) },
      { -INT32_C(   464255493), -INT32_C(   352123670),  INT32_C(   967172768),  INT32_C(   993828385) },
      { -INT32_C(  1483138017),  INT32_C(  1483732106), -INT32_C(   560266977), -INT32_C(   113272390) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sub_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_sub_epi32");
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
    easysimd__m128i r = easysimd_mm_mask_sub_epi32(src, k, a, b);

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
test_easysimd_mm_maskz_sub_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(202),
      {  INT32_C(     4286974), -INT32_C(   132718303), -INT32_C(  1129153675), -INT32_C(  1846917171) },
      { -INT32_C(   945752133),  INT32_C(  1885357902),  INT32_C(   821514401),  INT32_C(    66818053) },
      {  INT32_C(           0), -INT32_C(  2018076205),  INT32_C(           0), -INT32_C(  1913735224) } },
    { UINT8_C(249),
      {  INT32_C(   505086780), -INT32_C(  1785523431),  INT32_C(   224547013),  INT32_C(   852030266) },
      { -INT32_C(   612331372),  INT32_C(  1081929968),  INT32_C(  2017832423), -INT32_C(   462337880) },
      {  INT32_C(  1117418152),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1314368146) } },
    { UINT8_C( 76),
      { -INT32_C(  1620770164), -INT32_C(   429524330),  INT32_C(  1344303709), -INT32_C(   891006406) },
      { -INT32_C(  1027883054), -INT32_C(   374670532), -INT32_C(  1986977215),  INT32_C(   550860179) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   963686372), -INT32_C(  1441866585) } },
    { UINT8_C(120),
      {  INT32_C(   890158907), -INT32_C(  1768754140),  INT32_C(  1741808404), -INT32_C(  2026267705) },
      {  INT32_C(  1388575830), -INT32_C(   929846106), -INT32_C(  1302651843),  INT32_C(   757758962) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1510940629) } },
    { UINT8_C( 59),
      {  INT32_C(   744448568), -INT32_C(   666765579),  INT32_C(  1654630599), -INT32_C(   575068447) },
      { -INT32_C(  1786573847), -INT32_C(  1160623202),  INT32_C(   581731751), -INT32_C(   396502609) },
      { -INT32_C(  1763944881),  INT32_C(   493857623),  INT32_C(           0), -INT32_C(   178565838) } },
    { UINT8_C( 60),
      { -INT32_C(  1288629059), -INT32_C(    42333867), -INT32_C(   807412568), -INT32_C(  1598505835) },
      { -INT32_C(  1958851008), -INT32_C(  1523386080),  INT32_C(  2136298917),  INT32_C(  1874542002) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1351255811),  INT32_C(   821919459) } },
    { UINT8_C( 81),
      { -INT32_C(   173595924),  INT32_C(  2040374428),  INT32_C(  1074687107),  INT32_C(  1937813285) },
      { -INT32_C(   426570771), -INT32_C(  1819526850), -INT32_C(   884667506), -INT32_C(  1306741306) },
      {  INT32_C(   252974847),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(215),
      {  INT32_C(  1752475587), -INT32_C(  1309938364), -INT32_C(  1428804613), -INT32_C(  1214756437) },
      {  INT32_C(   368410332),  INT32_C(   363038730), -INT32_C(  2099548467),  INT32_C(  1314557323) },
      {  INT32_C(  1384065255), -INT32_C(  1672977094),  INT32_C(   670743854),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sub_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_sub_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_sub_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_sub_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 4433731009757369919),  INT64_C( 6758264093256275600) },
      UINT8_C( 50),
      {  INT64_C( 2864860306467471902),  INT64_C( 6993109694420745132) },
      {  INT64_C( 1158910246035669323), -INT64_C( 3641494897007860457) },
      { -INT64_C( 4433731009757369919), -INT64_C( 7812139482280946027) } },
    { {  INT64_C( 5079693801395622417), -INT64_C( 7546171172018075730) },
      UINT8_C(222),
      {  INT64_C( 1668768594921004234),  INT64_C( 8501552428209368090) },
      { -INT64_C( 7429437653217488252), -INT64_C( 2638691491993302298) },
      {  INT64_C( 5079693801395622417), -INT64_C( 7306500153506881228) } },
    { { -INT64_C( 4764222766382777704), -INT64_C( 4406312375856682624) },
      UINT8_C(245),
      { -INT64_C( 6331112342540584472),  INT64_C(   96709777525679683) },
      { -INT64_C( 5974762658318769535),  INT64_C( 4224284446551603464) },
      { -INT64_C(  356349684221814937), -INT64_C( 4406312375856682624) } },
    { {  INT64_C(  516958226050672037), -INT64_C(  339707885862329024) },
      UINT8_C( 63),
      {  INT64_C( 2258256860241170669),  INT64_C(  373143050097066203) },
      { -INT64_C( 4396409489741729758),  INT64_C( 8787429474900394902) },
      {  INT64_C( 6654666349982900427), -INT64_C( 8414286424803328699) } },
    { {  INT64_C(  183835214991538120), -INT64_C( 3763962923877466594) },
      UINT8_C(103),
      { -INT64_C( 4038595804569237619), -INT64_C( 3295564434649143230) },
      {  INT64_C( 4454337381985333652), -INT64_C( 2403318764131611002) },
      { -INT64_C( 8492933186554571271), -INT64_C(  892245670517532228) } },
    { {  INT64_C( 3568030611014478138),  INT64_C( 4127249602420988120) },
      UINT8_C(160),
      {  INT64_C( 5315124675244714816), -INT64_C( 8146877973273051132) },
      {  INT64_C(  773967344111038133), -INT64_C( 5194082521655070373) },
      {  INT64_C( 3568030611014478138),  INT64_C( 4127249602420988120) } },
    { { -INT64_C( 1451928299164677420), -INT64_C( 6635048073546337097) },
      UINT8_C( 64),
      {  INT64_C( 8607079577853004136),  INT64_C( 8208555407882239020) },
      { -INT64_C( 4033462391171155303), -INT64_C( 5280935319511342108) },
      { -INT64_C( 1451928299164677420), -INT64_C( 6635048073546337097) } },
    { { -INT64_C( 3269542247060854302), -INT64_C( 3668795197876563971) },
      UINT8_C( 76),
      { -INT64_C( 8749254773089297043), -INT64_C( 1560062132851832035) },
      { -INT64_C( 5578162083900390823), -INT64_C( 4870163270127437237) },
      { -INT64_C( 3269542247060854302), -INT64_C( 3668795197876563971) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sub_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_sub_epi64");
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
    easysimd__m128i r = easysimd_mm_mask_sub_epi64(src, k, a, b);

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
test_easysimd_mm_maskz_sub_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C( 52),
      { -INT64_C( 3623661971754600754),  INT64_C( 8585322111889400064) },
      {  INT64_C( 5582941670728678181),  INT64_C( 2681091308287229299) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(221),
      { -INT64_C(  587635540610785106), -INT64_C( 9143881439439216187) },
      {  INT64_C( 4608498819856910314), -INT64_C( 2843817372742650491) },
      { -INT64_C( 5196134360467695420),  INT64_C(                   0) } },
    { UINT8_C(155),
      { -INT64_C( 6261928141377130489), -INT64_C( 1050621664973353191) },
      { -INT64_C( 3025448211852300847),  INT64_C( 4494267212168736264) },
      { -INT64_C( 3236479929524829642), -INT64_C( 5544888877142089455) } },
    { UINT8_C(136),
      { -INT64_C( 4912660092034182513), -INT64_C( 5986456262892301399) },
      {  INT64_C( 1910199609033845000),  INT64_C( 3730546897129129316) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 34),
      {  INT64_C( 1240893144595098582), -INT64_C( 6463654762610929042) },
      { -INT64_C( 8155406181828737527), -INT64_C( 6183048851957346434) },
      {  INT64_C(                   0), -INT64_C(  280605910653582608) } },
    { UINT8_C( 80),
      { -INT64_C( 4181839152204072193),  INT64_C( 5581272233004907247) },
      { -INT64_C( 4366480954297490757), -INT64_C( 6481732479813716650) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(124),
      {  INT64_C( 6928985003471548186), -INT64_C( 1649949107199437898) },
      { -INT64_C( 2358372817472519222),  INT64_C( 2789465450235691256) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 91),
      {  INT64_C( 1877686066081783716), -INT64_C( 8447051372408181304) },
      { -INT64_C(  431829918876912215), -INT64_C( 7574722582901474470) },
      {  INT64_C( 2309515984958695931), -INT64_C(  872328789506706834) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sub_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_sub_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_sub_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_sub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   422.32), EASYSIMD_FLOAT32_C(   687.73), EASYSIMD_FLOAT32_C(  -342.59), EASYSIMD_FLOAT32_C(  -405.79) },
      UINT8_C(139),
      { EASYSIMD_FLOAT32_C(    99.11), EASYSIMD_FLOAT32_C(    -3.68), EASYSIMD_FLOAT32_C(   526.59), EASYSIMD_FLOAT32_C(  -649.91) },
      { EASYSIMD_FLOAT32_C(   567.32), EASYSIMD_FLOAT32_C(   296.23), EASYSIMD_FLOAT32_C(   -38.52), EASYSIMD_FLOAT32_C(  -800.17) },
      { EASYSIMD_FLOAT32_C(  -468.21), EASYSIMD_FLOAT32_C(  -299.91), EASYSIMD_FLOAT32_C(  -342.59), EASYSIMD_FLOAT32_C(   150.26) } },
    { { EASYSIMD_FLOAT32_C(  -632.90), EASYSIMD_FLOAT32_C(   409.15), EASYSIMD_FLOAT32_C(  -733.60), EASYSIMD_FLOAT32_C(    15.47) },
      UINT8_C(217),
      { EASYSIMD_FLOAT32_C(   300.78), EASYSIMD_FLOAT32_C(  -652.21), EASYSIMD_FLOAT32_C(  -241.42), EASYSIMD_FLOAT32_C(    23.67) },
      { EASYSIMD_FLOAT32_C(   281.23), EASYSIMD_FLOAT32_C(   173.16), EASYSIMD_FLOAT32_C(   325.54), EASYSIMD_FLOAT32_C(  -658.98) },
      { EASYSIMD_FLOAT32_C(    19.55), EASYSIMD_FLOAT32_C(   409.15), EASYSIMD_FLOAT32_C(  -733.60), EASYSIMD_FLOAT32_C(   682.65) } },
    { { EASYSIMD_FLOAT32_C(   465.71), EASYSIMD_FLOAT32_C(   206.60), EASYSIMD_FLOAT32_C(   624.09), EASYSIMD_FLOAT32_C(  -350.67) },
      UINT8_C(211),
      { EASYSIMD_FLOAT32_C(    46.41), EASYSIMD_FLOAT32_C(  -662.94), EASYSIMD_FLOAT32_C(   404.81), EASYSIMD_FLOAT32_C(   640.62) },
      { EASYSIMD_FLOAT32_C(  -302.07), EASYSIMD_FLOAT32_C(  -496.08), EASYSIMD_FLOAT32_C(  -363.06), EASYSIMD_FLOAT32_C(  -775.48) },
      { EASYSIMD_FLOAT32_C(   348.48), EASYSIMD_FLOAT32_C(  -166.86), EASYSIMD_FLOAT32_C(   624.09), EASYSIMD_FLOAT32_C(  -350.67) } },
    { { EASYSIMD_FLOAT32_C(  -146.00), EASYSIMD_FLOAT32_C(  -795.74), EASYSIMD_FLOAT32_C(   520.75), EASYSIMD_FLOAT32_C(   815.49) },
      UINT8_C(127),
      { EASYSIMD_FLOAT32_C(   887.85), EASYSIMD_FLOAT32_C(   224.63), EASYSIMD_FLOAT32_C(  -329.51), EASYSIMD_FLOAT32_C(   -96.68) },
      { EASYSIMD_FLOAT32_C(  -870.18), EASYSIMD_FLOAT32_C(   971.27), EASYSIMD_FLOAT32_C(   251.11), EASYSIMD_FLOAT32_C(  -111.60) },
      { EASYSIMD_FLOAT32_C(  1758.03), EASYSIMD_FLOAT32_C(  -746.64), EASYSIMD_FLOAT32_C(  -580.62), EASYSIMD_FLOAT32_C(    14.92) } },
    { { EASYSIMD_FLOAT32_C(    -5.06), EASYSIMD_FLOAT32_C(  -467.67), EASYSIMD_FLOAT32_C(  -938.44), EASYSIMD_FLOAT32_C(  -679.51) },
      UINT8_C( 44),
      { EASYSIMD_FLOAT32_C(   527.27), EASYSIMD_FLOAT32_C(   527.09), EASYSIMD_FLOAT32_C(  -502.56), EASYSIMD_FLOAT32_C(  -823.40) },
      { EASYSIMD_FLOAT32_C(  -725.51), EASYSIMD_FLOAT32_C(   543.84), EASYSIMD_FLOAT32_C(  -486.34), EASYSIMD_FLOAT32_C(   679.29) },
      { EASYSIMD_FLOAT32_C(    -5.06), EASYSIMD_FLOAT32_C(  -467.67), EASYSIMD_FLOAT32_C(   -16.22), EASYSIMD_FLOAT32_C( -1502.69) } },
    { { EASYSIMD_FLOAT32_C(   184.46), EASYSIMD_FLOAT32_C(   211.59), EASYSIMD_FLOAT32_C(  -816.79), EASYSIMD_FLOAT32_C(   821.40) },
      UINT8_C( 77),
      { EASYSIMD_FLOAT32_C(    37.21), EASYSIMD_FLOAT32_C(  -974.34), EASYSIMD_FLOAT32_C(   -43.14), EASYSIMD_FLOAT32_C(  -147.30) },
      { EASYSIMD_FLOAT32_C(  -570.25), EASYSIMD_FLOAT32_C(  -155.29), EASYSIMD_FLOAT32_C(  -922.67), EASYSIMD_FLOAT32_C(   100.23) },
      { EASYSIMD_FLOAT32_C(   607.46), EASYSIMD_FLOAT32_C(   211.59), EASYSIMD_FLOAT32_C(   879.53), EASYSIMD_FLOAT32_C(  -247.53) } },
    { { EASYSIMD_FLOAT32_C(   748.03), EASYSIMD_FLOAT32_C(  -792.84), EASYSIMD_FLOAT32_C(    71.50), EASYSIMD_FLOAT32_C(    -0.86) },
      UINT8_C( 47),
      { EASYSIMD_FLOAT32_C(  -933.55), EASYSIMD_FLOAT32_C(   531.47), EASYSIMD_FLOAT32_C(   157.12), EASYSIMD_FLOAT32_C(  -613.07) },
      { EASYSIMD_FLOAT32_C(  -595.18), EASYSIMD_FLOAT32_C(  -315.61), EASYSIMD_FLOAT32_C(   914.02), EASYSIMD_FLOAT32_C(   -97.74) },
      { EASYSIMD_FLOAT32_C(  -338.37), EASYSIMD_FLOAT32_C(   847.08), EASYSIMD_FLOAT32_C(  -756.90), EASYSIMD_FLOAT32_C(  -515.33) } },
    { { EASYSIMD_FLOAT32_C(  -139.01), EASYSIMD_FLOAT32_C(  -811.49), EASYSIMD_FLOAT32_C(  -553.90), EASYSIMD_FLOAT32_C(   374.65) },
      UINT8_C(208),
      { EASYSIMD_FLOAT32_C(   630.57), EASYSIMD_FLOAT32_C(  -413.76), EASYSIMD_FLOAT32_C(  -948.99), EASYSIMD_FLOAT32_C(   451.97) },
      { EASYSIMD_FLOAT32_C(  -977.65), EASYSIMD_FLOAT32_C(    88.23), EASYSIMD_FLOAT32_C(   477.63), EASYSIMD_FLOAT32_C(   -20.78) },
      { EASYSIMD_FLOAT32_C(  -139.01), EASYSIMD_FLOAT32_C(  -811.49), EASYSIMD_FLOAT32_C(  -553.90), EASYSIMD_FLOAT32_C(   374.65) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sub_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_sub_ps");
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
    easysimd__m128 r = easysimd_mm_mask_sub_ps(src, k, a, b);

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
test_easysimd_mm_maskz_sub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(147),
      { EASYSIMD_FLOAT32_C(   570.95), EASYSIMD_FLOAT32_C(  -512.83), EASYSIMD_FLOAT32_C(   990.38), EASYSIMD_FLOAT32_C(   458.51) },
      { EASYSIMD_FLOAT32_C(   351.83), EASYSIMD_FLOAT32_C(   844.15), EASYSIMD_FLOAT32_C(   318.94), EASYSIMD_FLOAT32_C(  -990.79) },
      { EASYSIMD_FLOAT32_C(   219.12), EASYSIMD_FLOAT32_C( -1356.98), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(228),
      { EASYSIMD_FLOAT32_C(  -824.06), EASYSIMD_FLOAT32_C(   952.34), EASYSIMD_FLOAT32_C(   -35.69), EASYSIMD_FLOAT32_C(  -445.43) },
      { EASYSIMD_FLOAT32_C(  -369.74), EASYSIMD_FLOAT32_C(  -930.65), EASYSIMD_FLOAT32_C(   542.49), EASYSIMD_FLOAT32_C(   270.82) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -578.18), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(234),
      { EASYSIMD_FLOAT32_C(   217.16), EASYSIMD_FLOAT32_C(   377.69), EASYSIMD_FLOAT32_C(   351.62), EASYSIMD_FLOAT32_C(  -530.42) },
      { EASYSIMD_FLOAT32_C(  -138.14), EASYSIMD_FLOAT32_C(  -234.63), EASYSIMD_FLOAT32_C(  -166.50), EASYSIMD_FLOAT32_C(   -24.79) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   612.32), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -505.63) } },
    { UINT8_C( 73),
      { EASYSIMD_FLOAT32_C(   273.87), EASYSIMD_FLOAT32_C(   133.94), EASYSIMD_FLOAT32_C(   -61.63), EASYSIMD_FLOAT32_C(  -313.71) },
      { EASYSIMD_FLOAT32_C(  -295.11), EASYSIMD_FLOAT32_C(   425.55), EASYSIMD_FLOAT32_C(  -323.33), EASYSIMD_FLOAT32_C(  -836.60) },
      { EASYSIMD_FLOAT32_C(   568.98), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   522.89) } },
    { UINT8_C(112),
      { EASYSIMD_FLOAT32_C(  -479.18), EASYSIMD_FLOAT32_C(   482.34), EASYSIMD_FLOAT32_C(  -213.41), EASYSIMD_FLOAT32_C(  -641.19) },
      { EASYSIMD_FLOAT32_C(   658.28), EASYSIMD_FLOAT32_C(  -261.07), EASYSIMD_FLOAT32_C(   323.12), EASYSIMD_FLOAT32_C(  -787.15) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 47),
      { EASYSIMD_FLOAT32_C(   392.47), EASYSIMD_FLOAT32_C(   755.34), EASYSIMD_FLOAT32_C(  -359.99), EASYSIMD_FLOAT32_C(   755.33) },
      { EASYSIMD_FLOAT32_C(   -27.50), EASYSIMD_FLOAT32_C(  -982.30), EASYSIMD_FLOAT32_C(   106.95), EASYSIMD_FLOAT32_C(   442.08) },
      { EASYSIMD_FLOAT32_C(   419.97), EASYSIMD_FLOAT32_C(  1737.64), EASYSIMD_FLOAT32_C(  -466.94), EASYSIMD_FLOAT32_C(   313.25) } },
    { UINT8_C(134),
      { EASYSIMD_FLOAT32_C(   872.32), EASYSIMD_FLOAT32_C(  -724.42), EASYSIMD_FLOAT32_C(   854.78), EASYSIMD_FLOAT32_C(  -226.25) },
      { EASYSIMD_FLOAT32_C(   549.46), EASYSIMD_FLOAT32_C(   -11.29), EASYSIMD_FLOAT32_C(   712.13), EASYSIMD_FLOAT32_C(  -764.26) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -713.13), EASYSIMD_FLOAT32_C(   142.65), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 23),
      { EASYSIMD_FLOAT32_C(   137.68), EASYSIMD_FLOAT32_C(   -87.59), EASYSIMD_FLOAT32_C(   857.00), EASYSIMD_FLOAT32_C(   915.05) },
      { EASYSIMD_FLOAT32_C(   433.23), EASYSIMD_FLOAT32_C(   339.34), EASYSIMD_FLOAT32_C(  -298.36), EASYSIMD_FLOAT32_C(   792.04) },
      { EASYSIMD_FLOAT32_C(  -295.55), EASYSIMD_FLOAT32_C(  -426.93), EASYSIMD_FLOAT32_C(  1155.36), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sub_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_sub_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_sub_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_sub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -565.08), EASYSIMD_FLOAT64_C(  -791.01) },
      UINT8_C( 41),
      { EASYSIMD_FLOAT64_C(   590.93), EASYSIMD_FLOAT64_C(     5.59) },
      { EASYSIMD_FLOAT64_C(   -84.38), EASYSIMD_FLOAT64_C(  -410.99) },
      { EASYSIMD_FLOAT64_C(   675.31), EASYSIMD_FLOAT64_C(  -791.01) } },
    { { EASYSIMD_FLOAT64_C(  -875.24), EASYSIMD_FLOAT64_C(   251.86) },
      UINT8_C(152),
      { EASYSIMD_FLOAT64_C(   304.56), EASYSIMD_FLOAT64_C(   448.80) },
      { EASYSIMD_FLOAT64_C(   496.30), EASYSIMD_FLOAT64_C(  -472.92) },
      { EASYSIMD_FLOAT64_C(  -875.24), EASYSIMD_FLOAT64_C(   251.86) } },
    { { EASYSIMD_FLOAT64_C(   890.24), EASYSIMD_FLOAT64_C(   454.02) },
      UINT8_C(188),
      { EASYSIMD_FLOAT64_C(  -709.59), EASYSIMD_FLOAT64_C(  -384.74) },
      { EASYSIMD_FLOAT64_C(  -781.50), EASYSIMD_FLOAT64_C(   806.98) },
      { EASYSIMD_FLOAT64_C(   890.24), EASYSIMD_FLOAT64_C(   454.02) } },
    { { EASYSIMD_FLOAT64_C(  -870.62), EASYSIMD_FLOAT64_C(  -360.22) },
      UINT8_C(136),
      { EASYSIMD_FLOAT64_C(   730.85), EASYSIMD_FLOAT64_C(  -889.27) },
      { EASYSIMD_FLOAT64_C(  -145.03), EASYSIMD_FLOAT64_C(   334.78) },
      { EASYSIMD_FLOAT64_C(  -870.62), EASYSIMD_FLOAT64_C(  -360.22) } },
    { { EASYSIMD_FLOAT64_C(    73.93), EASYSIMD_FLOAT64_C(   153.37) },
      UINT8_C(113),
      { EASYSIMD_FLOAT64_C(   508.85), EASYSIMD_FLOAT64_C(   362.36) },
      { EASYSIMD_FLOAT64_C(   796.24), EASYSIMD_FLOAT64_C(    99.77) },
      { EASYSIMD_FLOAT64_C(  -287.39), EASYSIMD_FLOAT64_C(   153.37) } },
    { { EASYSIMD_FLOAT64_C(  -632.05), EASYSIMD_FLOAT64_C(  -288.13) },
      UINT8_C(236),
      { EASYSIMD_FLOAT64_C(  -507.29), EASYSIMD_FLOAT64_C(   963.72) },
      { EASYSIMD_FLOAT64_C(  -790.25), EASYSIMD_FLOAT64_C(   797.26) },
      { EASYSIMD_FLOAT64_C(  -632.05), EASYSIMD_FLOAT64_C(  -288.13) } },
    { { EASYSIMD_FLOAT64_C(   412.52), EASYSIMD_FLOAT64_C(   706.05) },
      UINT8_C( 22),
      { EASYSIMD_FLOAT64_C(   302.76), EASYSIMD_FLOAT64_C(   160.08) },
      { EASYSIMD_FLOAT64_C(   112.10), EASYSIMD_FLOAT64_C(   593.18) },
      { EASYSIMD_FLOAT64_C(   412.52), EASYSIMD_FLOAT64_C(  -433.10) } },
    { { EASYSIMD_FLOAT64_C(   775.33), EASYSIMD_FLOAT64_C(   330.60) },
      UINT8_C( 19),
      { EASYSIMD_FLOAT64_C(   904.72), EASYSIMD_FLOAT64_C(   970.38) },
      { EASYSIMD_FLOAT64_C(   947.75), EASYSIMD_FLOAT64_C(   635.57) },
      { EASYSIMD_FLOAT64_C(   -43.03), EASYSIMD_FLOAT64_C(   334.81) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_sub_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_sub_pd");
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
    easysimd__m128d r = easysimd_mm_mask_sub_pd(src, k, a, b);

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
test_easysimd_mm_maskz_sub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C( 51),
      { EASYSIMD_FLOAT64_C(   767.66), EASYSIMD_FLOAT64_C(   653.09) },
      { EASYSIMD_FLOAT64_C(   439.75), EASYSIMD_FLOAT64_C(  -877.91) },
      { EASYSIMD_FLOAT64_C(   327.91), EASYSIMD_FLOAT64_C(  1531.00) } },
    { UINT8_C(200),
      { EASYSIMD_FLOAT64_C(  -138.62), EASYSIMD_FLOAT64_C(   456.14) },
      { EASYSIMD_FLOAT64_C(  -696.40), EASYSIMD_FLOAT64_C(   921.03) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 20),
      { EASYSIMD_FLOAT64_C(  -456.80), EASYSIMD_FLOAT64_C(  -700.89) },
      { EASYSIMD_FLOAT64_C(  -814.76), EASYSIMD_FLOAT64_C(   327.59) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(  5),
      { EASYSIMD_FLOAT64_C(   836.53), EASYSIMD_FLOAT64_C(  -573.34) },
      { EASYSIMD_FLOAT64_C(   622.66), EASYSIMD_FLOAT64_C(    36.45) },
      { EASYSIMD_FLOAT64_C(   213.87), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(150),
      { EASYSIMD_FLOAT64_C(  -997.93), EASYSIMD_FLOAT64_C(  -260.38) },
      { EASYSIMD_FLOAT64_C(  -163.33), EASYSIMD_FLOAT64_C(  -853.33) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   592.95) } },
    { UINT8_C(148),
      { EASYSIMD_FLOAT64_C(   236.57), EASYSIMD_FLOAT64_C(  -568.81) },
      { EASYSIMD_FLOAT64_C(  -488.22), EASYSIMD_FLOAT64_C(  -665.69) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 78),
      { EASYSIMD_FLOAT64_C(   201.05), EASYSIMD_FLOAT64_C(  -898.03) },
      { EASYSIMD_FLOAT64_C(   435.19), EASYSIMD_FLOAT64_C(  -359.19) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -538.84) } },
    { UINT8_C( 22),
      { EASYSIMD_FLOAT64_C(    69.72), EASYSIMD_FLOAT64_C(   502.19) },
      { EASYSIMD_FLOAT64_C(   680.20), EASYSIMD_FLOAT64_C(   373.32) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   128.87) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_sub_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_sub_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_sub_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_sub_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t src[32];
    const uint32_t k;
    const int8_t a[32];
    const int8_t b[32];
    const int8_t r[32];
  } test_vec[] = {
    { { -INT8_C(  66),  INT8_C(   8), -INT8_C(  57),  INT8_C(  80),  INT8_C(   8),  INT8_C(  96),  INT8_C(  47), -INT8_C(  41),
         INT8_C(  59),  INT8_C(  36),  INT8_C(  61), -INT8_C(  79),  INT8_C(  34), -INT8_C(  91),  INT8_C( 125), -INT8_C(  82),
         INT8_C( 123), -INT8_C(  70), -INT8_C(  60), -INT8_C(  93), -INT8_C( 105),  INT8_C(  56),  INT8_C( 114),  INT8_C(  78),
        -INT8_C(  46), -INT8_C( 102), -INT8_C(  54), -INT8_C(  52), -INT8_C( 107), -INT8_C(  35), -INT8_C(  76),  INT8_C(  84) },
      UINT32_C(4003757030),
      { -INT8_C(  36), -INT8_C(  45), -INT8_C(  59),  INT8_C(  23), -INT8_C(   9),  INT8_C(   3), -INT8_C(  56),  INT8_C(  25),
        -INT8_C(  88),  INT8_C(  69), -INT8_C(  56),  INT8_C(  36),  INT8_C(   0), -INT8_C( 116), -INT8_C(  57), -INT8_C( 105),
        -INT8_C(  59),  INT8_C(  57), -INT8_C(  27), -INT8_C( 105), -INT8_C(  44), -INT8_C(  80),  INT8_C(  99),  INT8_C( 105),
        -INT8_C( 115),  INT8_C(  24), -INT8_C(  67),  INT8_C( 115), -INT8_C( 109),  INT8_C(  98),  INT8_C(  98),  INT8_C( 111) },
      {  INT8_C(  53),  INT8_C(  39), -INT8_C( 122),  INT8_C(  45),  INT8_C(  42),  INT8_C(  79),  INT8_C(  70), -INT8_C(  45),
        -INT8_C( 108),  INT8_C(  14), -INT8_C(   9), -INT8_C( 108), -INT8_C( 101), -INT8_C(  66),  INT8_C(  44),  INT8_C(  96),
        -INT8_C(   8),  INT8_C(  17), -INT8_C(   9), -INT8_C(  52), -INT8_C(  63),  INT8_C(  91),  INT8_C(  53),  INT8_C(  79),
         INT8_C( 115), -INT8_C(  13), -INT8_C(  62),  INT8_C(   6),  INT8_C(  85),  INT8_C(  36),  INT8_C( 118), -INT8_C( 118) },
      { -INT8_C(  66), -INT8_C(  84),  INT8_C(  63),  INT8_C(  80),  INT8_C(   8), -INT8_C(  76), -INT8_C( 126),  INT8_C(  70),
         INT8_C(  20),  INT8_C(  55),  INT8_C(  61), -INT8_C( 112),  INT8_C( 101), -INT8_C(  50), -INT8_C( 101), -INT8_C(  82),
         INT8_C( 123), -INT8_C(  70), -INT8_C(  18), -INT8_C(  93), -INT8_C( 105),  INT8_C(  85),  INT8_C( 114),  INT8_C(  26),
        -INT8_C(  46),  INT8_C(  37), -INT8_C(   5),  INT8_C( 109), -INT8_C( 107),  INT8_C(  62), -INT8_C(  20), -INT8_C(  27) } },
    { {  INT8_C(  76), -INT8_C(   4), -INT8_C(  73),  INT8_C( 118),  INT8_C(  75), -INT8_C(   2),  INT8_C(  73), -INT8_C(  32),
         INT8_C(  12),  INT8_C(  64),  INT8_C( 116), -INT8_C(  89), -INT8_C(   1), -INT8_C(  96),  INT8_C(   7), -INT8_C(   9),
        -INT8_C(  78), -INT8_C(   1), -INT8_C(  61),  INT8_C( 115),  INT8_C(  90), -INT8_C(   8), -INT8_C(  62), -INT8_C(  51),
        -INT8_C(  21), -INT8_C( 123), -INT8_C(  45),  INT8_C(  64), -INT8_C(  87),  INT8_C(  73), -INT8_C(  53), -INT8_C(  11) },
      UINT32_C(2439807558),
      {      INT8_MIN, -INT8_C(  75),  INT8_C( 113), -INT8_C( 115), -INT8_C(  10), -INT8_C(  26),  INT8_C(  52), -INT8_C(  11),
        -INT8_C( 122),  INT8_C(  60), -INT8_C(  20),  INT8_C(  56),  INT8_C(  59), -INT8_C(  81), -INT8_C(  84), -INT8_C( 107),
        -INT8_C(  89),  INT8_C( 110),  INT8_C(  98), -INT8_C( 109), -INT8_C(  13),  INT8_C(  53), -INT8_C(  45), -INT8_C(  99),
             INT8_MAX, -INT8_C(  98), -INT8_C( 110), -INT8_C(  59),  INT8_C(  33), -INT8_C(   2),  INT8_C(  86), -INT8_C(  95) },
      { -INT8_C(  76), -INT8_C(  56),  INT8_C(  46), -INT8_C(  86), -INT8_C(  82),  INT8_C(  99), -INT8_C(  97),  INT8_C(  52),
        -INT8_C(  97), -INT8_C( 117),  INT8_C( 109), -INT8_C(  38),  INT8_C(  58),  INT8_C(  25),  INT8_C( 111), -INT8_C(  31),
        -INT8_C( 121), -INT8_C(  47),  INT8_C( 116),  INT8_C( 123),  INT8_C(   6),  INT8_C(  72),  INT8_C(  24), -INT8_C( 123),
        -INT8_C(  26), -INT8_C(  86),  INT8_C(  74),  INT8_C(   7), -INT8_C(  87), -INT8_C(  95), -INT8_C(  87),  INT8_C(  93) },
      {  INT8_C(  76), -INT8_C(  19),  INT8_C(  67),  INT8_C( 118),  INT8_C(  75), -INT8_C(   2), -INT8_C( 107), -INT8_C(  32),
         INT8_C(  12), -INT8_C(  79),  INT8_C( 116), -INT8_C(  89), -INT8_C(   1), -INT8_C(  96),  INT8_C(   7), -INT8_C(  76),
        -INT8_C(  78), -INT8_C(   1), -INT8_C(  18),  INT8_C(  24),  INT8_C(  90), -INT8_C(  19), -INT8_C(  69), -INT8_C(  51),
        -INT8_C( 103), -INT8_C( 123), -INT8_C(  45),  INT8_C(  64),  INT8_C( 120),  INT8_C(  73), -INT8_C(  53),  INT8_C(  68) } },
    { {  INT8_C( 105), -INT8_C(  41),  INT8_C(   7),  INT8_C(  23),  INT8_C(  58), -INT8_C(  90),  INT8_C(  75), -INT8_C(  39),
         INT8_C(  49), -INT8_C(  72), -INT8_C(  77),  INT8_C( 107), -INT8_C(  47),  INT8_C(  34),  INT8_C(  76),  INT8_C(  89),
        -INT8_C(  13), -INT8_C(  63), -INT8_C(  44), -INT8_C(   6),  INT8_C(   9), -INT8_C(  20),      INT8_MAX, -INT8_C(  17),
        -INT8_C( 106), -INT8_C(  54), -INT8_C(   9),  INT8_C(  63),  INT8_C( 107), -INT8_C(  96), -INT8_C( 100), -INT8_C(  44) },
      UINT32_C(3001787255),
      {  INT8_C(  73),  INT8_C(  54), -INT8_C( 117),  INT8_C( 122), -INT8_C(  17),  INT8_C(  63), -INT8_C(  27), -INT8_C(  64),
         INT8_C(  97),  INT8_C(  50),  INT8_C(  25),  INT8_C(  85), -INT8_C(  13), -INT8_C(  19),  INT8_C(  79), -INT8_C(   4),
        -INT8_C(  39), -INT8_C(  50), -INT8_C(  21),  INT8_C( 112), -INT8_C( 104), -INT8_C(  30), -INT8_C(  81),  INT8_C(   3),
        -INT8_C( 126),  INT8_C(  76), -INT8_C(  41), -INT8_C(   6), -INT8_C(  17), -INT8_C(  62), -INT8_C(  84),  INT8_C(  57) },
      { -INT8_C(   7),  INT8_C(  55), -INT8_C(  77), -INT8_C(  24),  INT8_C( 118), -INT8_C( 103), -INT8_C(  88), -INT8_C(  40),
        -INT8_C(  53), -INT8_C(  62),  INT8_C(  45), -INT8_C(  66), -INT8_C(  81),  INT8_C( 124), -INT8_C(  70), -INT8_C( 119),
         INT8_C(  74), -INT8_C(  91), -INT8_C(   7), -INT8_C(  29), -INT8_C( 120), -INT8_C(  88), -INT8_C(  26),  INT8_C(  10),
        -INT8_C(  12), -INT8_C(  66),  INT8_C(   4), -INT8_C(  28),      INT8_MIN, -INT8_C(  80),  INT8_C(  29),  INT8_C( 121) },
      {  INT8_C(  80), -INT8_C(   1), -INT8_C(  40),  INT8_C(  23),  INT8_C( 121), -INT8_C(  90),  INT8_C(  61), -INT8_C(  39),
        -INT8_C( 106),  INT8_C( 112), -INT8_C(  77),  INT8_C( 107), -INT8_C(  47),  INT8_C( 113),  INT8_C(  76),  INT8_C( 115),
        -INT8_C( 113),  INT8_C(  41), -INT8_C(  44), -INT8_C( 115),  INT8_C(   9),  INT8_C(  58), -INT8_C(  55), -INT8_C(   7),
        -INT8_C( 106), -INT8_C( 114), -INT8_C(   9),  INT8_C(  63),  INT8_C( 111),  INT8_C(  18), -INT8_C( 100), -INT8_C(  64) } },
    { { -INT8_C(  24), -INT8_C(  48),  INT8_C(  97),  INT8_C(  94),  INT8_C( 105),  INT8_C(  10),  INT8_C(  54),  INT8_C(  52),
        -INT8_C(  52),  INT8_C(  99), -INT8_C(  14),  INT8_C( 123), -INT8_C(  33), -INT8_C(  84),  INT8_C(   4),  INT8_C(  42),
         INT8_C(  82), -INT8_C(   3),  INT8_C(  13), -INT8_C(  38), -INT8_C(  90), -INT8_C(  13), -INT8_C(  28), -INT8_C( 102),
        -INT8_C(  79), -INT8_C(  23),  INT8_C( 126),  INT8_C(  50), -INT8_C( 103), -INT8_C( 101), -INT8_C(  85), -INT8_C( 127) },
      UINT32_C(3588230508),
      {  INT8_C(  23),  INT8_C(  22),  INT8_C(  10), -INT8_C(  29),  INT8_C( 122), -INT8_C(   4),  INT8_C(  94),  INT8_C(  89),
        -INT8_C(  87),  INT8_C(  99), -INT8_C( 125), -INT8_C(   5),  INT8_C(  96), -INT8_C( 112), -INT8_C(  43),  INT8_C(   6),
        -INT8_C( 124), -INT8_C(  71), -INT8_C(  95),  INT8_C(  53), -INT8_C(  94),  INT8_C(  31),  INT8_C( 103),  INT8_C(  60),
        -INT8_C(  69),  INT8_C(  19), -INT8_C(  67),  INT8_C(  39),  INT8_C(  32), -INT8_C(  99), -INT8_C(   4),  INT8_C(  55) },
      { -INT8_C(  76),  INT8_C(   6),  INT8_C(  26),  INT8_C(  46),  INT8_C(   3),  INT8_C( 120), -INT8_C( 121), -INT8_C(  84),
        -INT8_C(  37),  INT8_C(  11), -INT8_C(  89),  INT8_C(  60), -INT8_C( 101),  INT8_C( 124),  INT8_C(  66),  INT8_C(  31),
         INT8_C(  53), -INT8_C(  29),  INT8_C(  85), -INT8_C(  40),  INT8_C(   3), -INT8_C(  68),  INT8_C(  20), -INT8_C(  66),
        -INT8_C(  49), -INT8_C(  47), -INT8_C(  27), -INT8_C(  17),  INT8_C( 111), -INT8_C(  31),  INT8_C(  38),  INT8_C(  35) },
      { -INT8_C(  24), -INT8_C(  48), -INT8_C(  16), -INT8_C(  75),  INT8_C( 105), -INT8_C( 124), -INT8_C(  41),  INT8_C(  52),
        -INT8_C(  50),  INT8_C(  99), -INT8_C(  36), -INT8_C(  65), -INT8_C(  33), -INT8_C(  84),  INT8_C(   4),  INT8_C(  42),
         INT8_C(  82), -INT8_C(   3),  INT8_C(  13), -INT8_C(  38), -INT8_C(  90),  INT8_C(  99),  INT8_C(  83),  INT8_C( 126),
        -INT8_C(  20), -INT8_C(  23), -INT8_C(  40),  INT8_C(  50), -INT8_C(  79), -INT8_C( 101), -INT8_C(  42),  INT8_C(  20) } },
    { { -INT8_C(  24),  INT8_C(  64),  INT8_C(  81), -INT8_C(  21), -INT8_C(  71), -INT8_C(  40), -INT8_C( 105), -INT8_C( 108),
        -INT8_C(  29),  INT8_C(  62), -INT8_C(  48),      INT8_MAX, -INT8_C(  70),  INT8_C(  19), -INT8_C(  98), -INT8_C(  17),
        -INT8_C(  10), -INT8_C(  13), -INT8_C(  57), -INT8_C(   7), -INT8_C(  80), -INT8_C(  37), -INT8_C(  73),      INT8_MAX,
        -INT8_C(  83), -INT8_C( 100),  INT8_C( 111),  INT8_C(  28),  INT8_C( 126), -INT8_C( 107),  INT8_C(  63),  INT8_C( 102) },
      UINT32_C(2404487382),
      {  INT8_C( 104), -INT8_C(  24),  INT8_C(  35),  INT8_C(  76),  INT8_C(  38), -INT8_C(  12), -INT8_C(  53), -INT8_C(  32),
         INT8_C(   7),  INT8_C( 105), -INT8_C(  49), -INT8_C(   3),  INT8_C(  93), -INT8_C( 105), -INT8_C(   9),  INT8_C(  13),
         INT8_C( 114), -INT8_C(  82), -INT8_C( 116),  INT8_C(  31),  INT8_C(  75), -INT8_C(   5),  INT8_C(  59), -INT8_C(  55),
        -INT8_C( 111),  INT8_C( 122),  INT8_C(  47),  INT8_C( 103),  INT8_C(  10),      INT8_MIN, -INT8_C(  10),  INT8_C( 115) },
      {  INT8_C( 104),  INT8_C(  25), -INT8_C(  65), -INT8_C( 114),  INT8_C(  13), -INT8_C( 118),  INT8_C( 110),  INT8_C(  20),
        -INT8_C(  13),  INT8_C(  61),  INT8_C(  18),  INT8_C(  80), -INT8_C(  44),  INT8_C(   9),  INT8_C(  93),  INT8_C(  71),
        -INT8_C(  73), -INT8_C(  22),  INT8_C( 102),  INT8_C(   2), -INT8_C(  27), -INT8_C(  94), -INT8_C(  53),  INT8_C( 118),
         INT8_C(  28), -INT8_C(   6), -INT8_C(  35),  INT8_C(  39),  INT8_C( 122), -INT8_C(  45), -INT8_C( 102), -INT8_C(  30) },
      { -INT8_C(  24), -INT8_C(  49),  INT8_C( 100), -INT8_C(  21),  INT8_C(  25), -INT8_C(  40),  INT8_C(  93), -INT8_C(  52),
        -INT8_C(  29),  INT8_C(  62), -INT8_C(  48),      INT8_MAX, -INT8_C( 119),  INT8_C(  19), -INT8_C(  98), -INT8_C(  58),
        -INT8_C(  69), -INT8_C(  13), -INT8_C(  57), -INT8_C(   7),  INT8_C( 102), -INT8_C(  37),  INT8_C( 112),      INT8_MAX,
         INT8_C( 117),      INT8_MIN,  INT8_C(  82),  INT8_C(  64),  INT8_C( 126), -INT8_C( 107),  INT8_C(  63), -INT8_C( 111) } },
    { { -INT8_C(  19),  INT8_C(  89),  INT8_C( 112), -INT8_C(   6), -INT8_C(  29), -INT8_C(  34),  INT8_C(  15), -INT8_C(  42),
         INT8_C(  28),  INT8_C(  33),  INT8_C(  39), -INT8_C(  16),  INT8_C(  42), -INT8_C( 124),  INT8_C(  55), -INT8_C(  31),
         INT8_C( 110), -INT8_C(  98), -INT8_C(  28),  INT8_C(  84),  INT8_C(  64), -INT8_C(  81), -INT8_C(  54),  INT8_C(  92),
        -INT8_C(  86), -INT8_C(  88), -INT8_C( 125),  INT8_C(  36),  INT8_C( 123),  INT8_C(  29),  INT8_C(   7),  INT8_C( 104) },
      UINT32_C(1499690870),
      {  INT8_C(  86),  INT8_C( 114),  INT8_C(  48),  INT8_C( 114), -INT8_C( 109),  INT8_C(  87),  INT8_C(  98), -INT8_C(  67),
        -INT8_C(  37), -INT8_C( 102), -INT8_C(  98),  INT8_C(  74),  INT8_C(  56), -INT8_C( 126), -INT8_C(  98),  INT8_C( 120),
         INT8_C(  50),  INT8_C( 104), -INT8_C(  44), -INT8_C(  36),  INT8_C(  16),  INT8_C(  88),  INT8_C(   0), -INT8_C( 116),
         INT8_C( 117),  INT8_C(   7), -INT8_C(  12), -INT8_C(  20),      INT8_MAX,  INT8_C(  87),  INT8_C(  69), -INT8_C(  43) },
      { -INT8_C(  55),  INT8_C( 117),  INT8_C(  71),  INT8_C(  92), -INT8_C(  52), -INT8_C(  87),  INT8_C(  25), -INT8_C(  88),
         INT8_C(  67), -INT8_C(  72), -INT8_C(  14),  INT8_C( 123),  INT8_C(  58), -INT8_C( 112), -INT8_C(  13),  INT8_C( 108),
        -INT8_C(   8), -INT8_C(  56),  INT8_C(  72),  INT8_C(   9),  INT8_C(  32),  INT8_C(  73), -INT8_C( 107), -INT8_C( 107),
         INT8_C(  80), -INT8_C( 119), -INT8_C( 127), -INT8_C(  49), -INT8_C(  31), -INT8_C(  57), -INT8_C(  92), -INT8_C(  86) },
      { -INT8_C(  19), -INT8_C(   3), -INT8_C(  23), -INT8_C(   6), -INT8_C(  57), -INT8_C(  82),  INT8_C(  73), -INT8_C(  42),
        -INT8_C( 104), -INT8_C(  30), -INT8_C(  84), -INT8_C(  16), -INT8_C(   2), -INT8_C(  14), -INT8_C(  85), -INT8_C(  31),
         INT8_C(  58), -INT8_C(  96), -INT8_C(  28),  INT8_C(  84),  INT8_C(  64),  INT8_C(  15),  INT8_C( 107),  INT8_C(  92),
         INT8_C(  37), -INT8_C(  88), -INT8_C( 125),  INT8_C(  29), -INT8_C(  98),  INT8_C(  29), -INT8_C(  95),  INT8_C( 104) } },
    { {  INT8_C(  60), -INT8_C(  21),  INT8_C(   7),  INT8_C(   9), -INT8_C( 107),  INT8_C(  32), -INT8_C(  79), -INT8_C(  40),
        -INT8_C(  40), -INT8_C(  93),  INT8_C(  84),  INT8_C(  19),  INT8_C(  51),  INT8_C(  71),      INT8_MAX,  INT8_C(  43),
         INT8_C(  15), -INT8_C(  56),  INT8_C(  52),  INT8_C(  47),  INT8_C(  17), -INT8_C(  55), -INT8_C(  59),  INT8_C(  97),
         INT8_C(  83),  INT8_C(  70),  INT8_C(  49),  INT8_C(  52),  INT8_C(  13), -INT8_C(  43), -INT8_C(  34),  INT8_C(  74) },
      UINT32_C(1448338881),
      {  INT8_C(   6),  INT8_C(   4),  INT8_C(  46), -INT8_C(  34), -INT8_C(  89), -INT8_C( 126), -INT8_C(  15), -INT8_C(  38),
        -INT8_C(  54),  INT8_C( 113),  INT8_C(   5), -INT8_C(  39),  INT8_C(  57),  INT8_C(  58),  INT8_C(   9),  INT8_C(  74),
         INT8_C(   3), -INT8_C(  50), -INT8_C(  85),  INT8_C(  86),  INT8_C(  20), -INT8_C(  36), -INT8_C( 118),  INT8_C(  34),
        -INT8_C(  78),  INT8_C( 105),  INT8_C( 108),  INT8_C( 115),  INT8_C(  78), -INT8_C(  65), -INT8_C(  55),  INT8_C(  84) },
      { -INT8_C(  61), -INT8_C(   9),  INT8_C(  51),  INT8_C( 106),  INT8_C( 122),  INT8_C(  36),  INT8_C(  68),  INT8_C(  68),
        -INT8_C( 107),  INT8_C(  73),  INT8_C(  29), -INT8_C(  50), -INT8_C( 125),  INT8_C(  38),  INT8_C(  24), -INT8_C( 121),
        -INT8_C(  12), -INT8_C(  60), -INT8_C(  35),  INT8_C(   9), -INT8_C(  96),  INT8_C( 104),  INT8_C(  43),  INT8_C(  82),
        -INT8_C(  47), -INT8_C( 105), -INT8_C(  59),  INT8_C(  31),  INT8_C(  86), -INT8_C( 114),  INT8_C( 116),  INT8_C(  25) },
      {  INT8_C(  67), -INT8_C(  21),  INT8_C(   7),  INT8_C(   9), -INT8_C( 107),  INT8_C(  32), -INT8_C(  83), -INT8_C( 106),
         INT8_C(  53), -INT8_C(  93), -INT8_C(  24),  INT8_C(  19),  INT8_C(  51),  INT8_C(  20), -INT8_C(  15), -INT8_C(  61),
         INT8_C(  15),  INT8_C(  10),  INT8_C(  52),  INT8_C(  47),  INT8_C( 116), -INT8_C(  55),  INT8_C(  95),  INT8_C(  97),
         INT8_C(  83), -INT8_C(  46), -INT8_C(  89),  INT8_C(  52), -INT8_C(   8), -INT8_C(  43),  INT8_C(  85),  INT8_C(  74) } },
    { { -INT8_C( 122), -INT8_C(  89), -INT8_C( 125),  INT8_C(   0), -INT8_C(  53), -INT8_C(  57),  INT8_C(  68),  INT8_C(  97),
         INT8_C(  16),  INT8_C(  97),  INT8_C(  47), -INT8_C( 108), -INT8_C( 120),  INT8_C(  72),  INT8_C(  27),  INT8_C( 124),
         INT8_C(  12), -INT8_C(   8), -INT8_C( 123), -INT8_C(  84),  INT8_C(  96), -INT8_C(  80), -INT8_C(   1),  INT8_C(  49),
         INT8_C(  71), -INT8_C(  60),  INT8_C(  81), -INT8_C(  99),  INT8_C(  83), -INT8_C(  59), -INT8_C(  74), -INT8_C(  39) },
      UINT32_C( 936982892),
      {  INT8_C(   0),  INT8_C(  29), -INT8_C( 104),  INT8_C(  17),  INT8_C( 126), -INT8_C(  56), -INT8_C(  91),  INT8_C(   6),
         INT8_C(  16), -INT8_C(  64), -INT8_C( 125),  INT8_C(  28), -INT8_C(  72),  INT8_C(   8), -INT8_C(  56),  INT8_C(  25),
        -INT8_C(  71), -INT8_C(  57),  INT8_C(  74),  INT8_C(   0), -INT8_C( 116), -INT8_C( 101), -INT8_C(  98), -INT8_C(  33),
         INT8_C(  96),  INT8_C(  84), -INT8_C(  72), -INT8_C(  52), -INT8_C( 114), -INT8_C( 111),  INT8_C(   4), -INT8_C( 114) },
      { -INT8_C(  82), -INT8_C( 100), -INT8_C(  97),  INT8_C(  44),  INT8_C( 100),  INT8_C(  68),  INT8_C(  51),  INT8_C( 116),
         INT8_C(   4), -INT8_C(  74), -INT8_C( 112), -INT8_C(  67), -INT8_C(  66),  INT8_C(  89), -INT8_C(  42),  INT8_C( 119),
         INT8_C(  32),  INT8_C(  32),  INT8_C( 120), -INT8_C(  84), -INT8_C(  68),  INT8_C(  22), -INT8_C( 117),  INT8_C(  28),
         INT8_C( 106),  INT8_C(  67), -INT8_C(  23), -INT8_C(   8), -INT8_C(  44), -INT8_C(  19), -INT8_C( 121), -INT8_C( 126) },
      { -INT8_C( 122), -INT8_C(  89), -INT8_C(   7), -INT8_C(  27), -INT8_C(  53), -INT8_C( 124),  INT8_C( 114),  INT8_C(  97),
         INT8_C(  12),  INT8_C(  97),  INT8_C(  47),  INT8_C(  95), -INT8_C(   6), -INT8_C(  81),  INT8_C(  27),  INT8_C( 124),
        -INT8_C( 103), -INT8_C(   8), -INT8_C( 123),  INT8_C(  84), -INT8_C(  48), -INT8_C(  80),  INT8_C(  19), -INT8_C(  61),
        -INT8_C(  10),  INT8_C(  17), -INT8_C(  49), -INT8_C(  99), -INT8_C(  70), -INT8_C(  92), -INT8_C(  74), -INT8_C(  39) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi8(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sub_epi8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_sub_epi8");
    easysimd_test_x86_assert_equal_i8x32(r, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = easysimd_mm256_mask_sub_epi8(src, k, a, b);

    easysimd_test_x86_write_i8x32(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_sub_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t k;
    const int8_t a[32];
    const int8_t b[32];
    const int8_t r[32];
  } test_vec[] = {
    { UINT32_C(1905911765),
      { -INT8_C(  64), -INT8_C(  84), -INT8_C(  14), -INT8_C(  53),  INT8_C(  43), -INT8_C( 101), -INT8_C(  15),  INT8_C(  58),
        -INT8_C(  79),  INT8_C(  46),  INT8_C( 120), -INT8_C( 108),  INT8_C(  96), -INT8_C(  39),  INT8_C(  60), -INT8_C( 105),
        -INT8_C(  47),  INT8_C(   0), -INT8_C( 109), -INT8_C(  80),  INT8_C(  70), -INT8_C(  91), -INT8_C(  94), -INT8_C(  98),
         INT8_C(  27),  INT8_C(  27),  INT8_C(  62), -INT8_C(  15),  INT8_C(   3), -INT8_C(  41),  INT8_C(  98), -INT8_C(  61) },
      { -INT8_C( 124),  INT8_C(  84), -INT8_C( 113), -INT8_C(  81), -INT8_C(  17),      INT8_MIN, -INT8_C(  23), -INT8_C(  95),
        -INT8_C(  82),  INT8_C(  98),  INT8_C(  53),  INT8_C(  14),  INT8_C(  59),  INT8_C( 113), -INT8_C(  90),  INT8_C(  12),
         INT8_C( 114),  INT8_C(  57), -INT8_C(  68), -INT8_C(  72), -INT8_C(  34),  INT8_C(  94),  INT8_C(  86), -INT8_C(   7),
         INT8_C( 122), -INT8_C( 107), -INT8_C(  22),  INT8_C( 125),  INT8_C( 108),  INT8_C(  76),  INT8_C(  64), -INT8_C(  16) },
      {  INT8_C(  60),  INT8_C(   0),  INT8_C(  99),  INT8_C(   0),  INT8_C(  60),  INT8_C(   0),  INT8_C(   8), -INT8_C( 103),
         INT8_C(   3), -INT8_C(  52),  INT8_C(  67),  INT8_C(   0),  INT8_C(   0),  INT8_C( 104), -INT8_C( 106), -INT8_C( 117),
         INT8_C(  95),  INT8_C(   0),  INT8_C(   0), -INT8_C(   8),  INT8_C( 104),  INT8_C(   0),  INT8_C(   0), -INT8_C(  91),
        -INT8_C(  95),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 105), -INT8_C( 117),  INT8_C(  34),  INT8_C(   0) } },
    { UINT32_C(2426458016),
      {  INT8_C(  79), -INT8_C( 119),  INT8_C(  49), -INT8_C(   2), -INT8_C(  21),  INT8_C( 102),  INT8_C(  12),  INT8_C(  38),
        -INT8_C(  41), -INT8_C(  78),  INT8_C(  51),  INT8_C(  73), -INT8_C(  21), -INT8_C(  17),  INT8_C(   1), -INT8_C(  55),
         INT8_C(  78),  INT8_C(  88), -INT8_C(  61), -INT8_C(  56), -INT8_C(  19), -INT8_C(  83),  INT8_C(  69),  INT8_C(  89),
        -INT8_C(   6), -INT8_C( 123),  INT8_C(  74), -INT8_C( 102),  INT8_C(  85), -INT8_C(  22),  INT8_C(  42), -INT8_C(  92) },
      {  INT8_C( 115),  INT8_C(  91), -INT8_C(  94),  INT8_C(  95), -INT8_C(  63), -INT8_C(  81), -INT8_C( 123), -INT8_C( 103),
         INT8_C(  97), -INT8_C(  72), -INT8_C(  30),  INT8_C(  77), -INT8_C(  88), -INT8_C(  28),  INT8_C(  22), -INT8_C(  10),
         INT8_C(  60), -INT8_C(  39), -INT8_C(  66),  INT8_C(  41), -INT8_C( 121),  INT8_C(   3), -INT8_C( 126), -INT8_C( 127),
        -INT8_C( 120), -INT8_C(  52),  INT8_C(  27), -INT8_C(  35), -INT8_C(  74),  INT8_C(  70), -INT8_C( 126),  INT8_C(  42) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  73),  INT8_C(   0), -INT8_C( 115),
         INT8_C( 118), -INT8_C(   6),  INT8_C(  81), -INT8_C(   4),  INT8_C(   0),  INT8_C(   0), -INT8_C(  21), -INT8_C(  45),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  86),  INT8_C(   0), -INT8_C(  40),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  97),  INT8_C(   0),  INT8_C(   0),  INT8_C( 122) } },
    { UINT32_C(1669932193),
      { -INT8_C(  45),  INT8_C(  14), -INT8_C(   4),  INT8_C(  53), -INT8_C(  57), -INT8_C(  34), -INT8_C( 126),  INT8_C( 111),
        -INT8_C(  62), -INT8_C( 104),  INT8_C( 101), -INT8_C(   2),  INT8_C( 114),  INT8_C(  35),  INT8_C(  39), -INT8_C(   7),
         INT8_C(  38), -INT8_C(  86),  INT8_C( 122), -INT8_C(  82),  INT8_C( 118), -INT8_C( 107), -INT8_C( 116),  INT8_C(  45),
        -INT8_C(  37),  INT8_C(  14),  INT8_C(  87),  INT8_C( 125),  INT8_C(  50), -INT8_C(  32), -INT8_C(  32),  INT8_C(   6) },
      { -INT8_C(  18), -INT8_C(  36),  INT8_C(  59), -INT8_C(  75), -INT8_C(  70), -INT8_C(  67),  INT8_C(  36),  INT8_C( 125),
         INT8_C(  85), -INT8_C( 119),  INT8_C( 123), -INT8_C(  57), -INT8_C(  84), -INT8_C(  93), -INT8_C(  64), -INT8_C(  46),
         INT8_C(  77),  INT8_C(  58), -INT8_C( 127), -INT8_C(  61), -INT8_C(  48),  INT8_C(  13), -INT8_C(  16), -INT8_C(  85),
         INT8_C(  27),  INT8_C(  71),  INT8_C(  40),  INT8_C(  77),  INT8_C(  39),  INT8_C(   8),  INT8_C(  83),  INT8_C(  22) },
      { -INT8_C(  27),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  33),  INT8_C(   0), -INT8_C(  14),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  22),  INT8_C(   0),  INT8_C(   0),      INT8_MIN,  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  39),  INT8_C(   0),  INT8_C(   0), -INT8_C(  21),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 126),
        -INT8_C(  64), -INT8_C(  57),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  40), -INT8_C( 115),  INT8_C(   0) } },
    { UINT32_C(2680917732),
      {  INT8_C(  75), -INT8_C(  16),  INT8_C(  28), -INT8_C(  95),  INT8_C( 121), -INT8_C( 105),  INT8_C( 104),  INT8_C(  38),
         INT8_C(  58),  INT8_C(  41), -INT8_C(   8), -INT8_C( 121),  INT8_C(  99),  INT8_C( 121),  INT8_C(  75),  INT8_C(  51),
        -INT8_C( 122),  INT8_C(  59), -INT8_C(  33), -INT8_C(  95), -INT8_C( 125),  INT8_C(   7), -INT8_C(  17), -INT8_C(  86),
         INT8_C(  16),  INT8_C(  66), -INT8_C(  64), -INT8_C(  12), -INT8_C(  47), -INT8_C( 116), -INT8_C( 109),  INT8_C(  28) },
      {  INT8_C( 124), -INT8_C(  81), -INT8_C(  67), -INT8_C(  11),  INT8_C(  71),  INT8_C(  38),  INT8_C(  27), -INT8_C( 127),
         INT8_C(  79),  INT8_C(  20),  INT8_C(   9), -INT8_C(  78), -INT8_C( 115),  INT8_C(  84), -INT8_C(  26),  INT8_C(  20),
        -INT8_C( 113), -INT8_C(  59), -INT8_C(  75),  INT8_C(  18), -INT8_C(  52), -INT8_C(  92), -INT8_C(  67), -INT8_C(  36),
        -INT8_C(  25),  INT8_C( 125), -INT8_C(  47), -INT8_C(  72),  INT8_C(   9),  INT8_C( 100), -INT8_C(  44), -INT8_C( 123) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(  95),  INT8_C(   0),  INT8_C(   0),  INT8_C( 113),  INT8_C(  77), -INT8_C(  91),
         INT8_C(   0),  INT8_C(  21), -INT8_C(  17), -INT8_C(  43),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  31),
        -INT8_C(   9),  INT8_C( 118),  INT8_C(   0), -INT8_C( 113),  INT8_C(   0),  INT8_C(   0),  INT8_C(  50), -INT8_C(  50),
         INT8_C(  41), -INT8_C(  59), -INT8_C(  17),  INT8_C(  60), -INT8_C(  56),  INT8_C(   0),  INT8_C(   0), -INT8_C( 105) } },
    { UINT32_C(1534824980),
      { -INT8_C(  72), -INT8_C( 106), -INT8_C(  36),  INT8_C(   7), -INT8_C(  86), -INT8_C(  27), -INT8_C(  71),  INT8_C(  56),
         INT8_C(  57), -INT8_C(  97),  INT8_C(  76), -INT8_C(  55),  INT8_C( 100),  INT8_C(   1), -INT8_C(  37),  INT8_C(  49),
        -INT8_C(  90), -INT8_C( 104),  INT8_C(  13), -INT8_C( 115),  INT8_C(  22), -INT8_C(  34),  INT8_C(  69),  INT8_C(  31),
         INT8_C(  67),  INT8_C(  25), -INT8_C(  91),  INT8_C(  87), -INT8_C(  85),  INT8_C(  32), -INT8_C(  78),  INT8_C(  99) },
      { -INT8_C(  74), -INT8_C( 114),  INT8_C( 106),  INT8_C(  97),  INT8_C( 116),  INT8_C(  36), -INT8_C( 103), -INT8_C(  83),
        -INT8_C(  61), -INT8_C(  27),  INT8_C( 118),  INT8_C(  40), -INT8_C(  26),  INT8_C(  82),  INT8_C(  89), -INT8_C( 116),
        -INT8_C(  22),  INT8_C( 102),  INT8_C(  25),  INT8_C(   0),  INT8_C(  69),  INT8_C(  94),  INT8_C(  32), -INT8_C( 120),
         INT8_C( 120), -INT8_C(  59), -INT8_C(  33),  INT8_C(  35), -INT8_C(  27), -INT8_C( 111), -INT8_C( 121), -INT8_C( 101) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C( 114),  INT8_C(   0),  INT8_C(  54),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  70),  INT8_C(   0),  INT8_C(   0),  INT8_C( 126),  INT8_C(   0),  INT8_C(   0), -INT8_C(  91),
        -INT8_C(  68),  INT8_C(  50),  INT8_C(   0), -INT8_C( 115), -INT8_C(  47),      INT8_MIN,  INT8_C(  37),  INT8_C(   0),
        -INT8_C(  53),  INT8_C(  84),  INT8_C(   0),  INT8_C(  52), -INT8_C(  58),  INT8_C(   0),  INT8_C(  43),  INT8_C(   0) } },
    { UINT32_C(2482827551),
      {  INT8_C(  21), -INT8_C( 107),  INT8_C(  65), -INT8_C(  39),  INT8_C( 122), -INT8_C(  73),  INT8_C(   1),  INT8_C(  97),
         INT8_C(   9),  INT8_C(  90), -INT8_C(  19), -INT8_C(  12), -INT8_C(  64),  INT8_C(   7), -INT8_C(  12),  INT8_C(   5),
         INT8_C( 101),  INT8_C(  20), -INT8_C( 115), -INT8_C(  35), -INT8_C(  39),  INT8_C( 108),  INT8_C(   1), -INT8_C(  66),
        -INT8_C(   3), -INT8_C( 120),  INT8_C(  90),  INT8_C(  29),  INT8_C( 121),  INT8_C(  86), -INT8_C(  80), -INT8_C( 113) },
      { -INT8_C(  20), -INT8_C(  15),  INT8_C( 104),  INT8_C( 102), -INT8_C(  87),  INT8_C( 105), -INT8_C(  57), -INT8_C(  78),
        -INT8_C(  61), -INT8_C(  75), -INT8_C(  90), -INT8_C( 125), -INT8_C(  68), -INT8_C( 101), -INT8_C( 119),  INT8_C(  33),
        -INT8_C(  81),  INT8_C(  22), -INT8_C(   1), -INT8_C( 119), -INT8_C( 125),  INT8_C(   0),  INT8_C(  71),      INT8_MIN,
        -INT8_C( 120), -INT8_C(  95), -INT8_C(  99),  INT8_C(   1), -INT8_C(   8),  INT8_C(  78), -INT8_C( 112), -INT8_C(  28) },
      {  INT8_C(  41), -INT8_C(  92), -INT8_C(  39),  INT8_C( 115), -INT8_C(  47),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  70),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   4),  INT8_C( 108),  INT8_C( 107), -INT8_C(  28),
         INT8_C(   0),  INT8_C(   0), -INT8_C( 114),  INT8_C(  84),  INT8_C(  86),  INT8_C( 108), -INT8_C(  70),  INT8_C(  62),
         INT8_C( 117), -INT8_C(  25),  INT8_C(   0),  INT8_C(   0), -INT8_C( 127),  INT8_C(   0),  INT8_C(   0), -INT8_C(  85) } },
    { UINT32_C(3897227327),
      {  INT8_C(  97),  INT8_C(  18), -INT8_C( 101),  INT8_C(  36), -INT8_C(  57),  INT8_C(  65), -INT8_C(  88), -INT8_C( 125),
        -INT8_C(  36),  INT8_C(  49), -INT8_C(  92), -INT8_C( 116),  INT8_C(  71), -INT8_C(  93),  INT8_C(  21), -INT8_C(  54),
        -INT8_C(  93),  INT8_C(  92),  INT8_C(  75),  INT8_C(  43), -INT8_C(   2), -INT8_C(  24),  INT8_C(  45), -INT8_C(  10),
         INT8_C(  54), -INT8_C(  67), -INT8_C(  38),  INT8_C( 118), -INT8_C(  74),  INT8_C(  36),  INT8_C(  94),  INT8_C(  23) },
      {  INT8_C(  54), -INT8_C(   7),  INT8_C(  60), -INT8_C(   3),  INT8_C(  59), -INT8_C(  28),      INT8_MIN,  INT8_C(  23),
         INT8_C(  21),  INT8_C(  37), -INT8_C(  93),  INT8_C(  92), -INT8_C(  56), -INT8_C(  72),  INT8_C(  39),  INT8_C( 108),
         INT8_C(  21),  INT8_C( 114), -INT8_C( 105),  INT8_C(  19),  INT8_C(  90), -INT8_C(  60),  INT8_C(   9), -INT8_C( 111),
        -INT8_C( 126), -INT8_C(  29),  INT8_C(   7),  INT8_C(  56),  INT8_C(   7),  INT8_C( 101),  INT8_C(  79),  INT8_C(  62) },
      {  INT8_C(  43),  INT8_C(  25),  INT8_C(  95),  INT8_C(  39), -INT8_C( 116),  INT8_C(  93),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  48),      INT8_MAX, -INT8_C(  21), -INT8_C(  18),  INT8_C(  94),
         INT8_C(   0), -INT8_C(  22),  INT8_C(   0),  INT8_C(  24),  INT8_C(   0),  INT8_C(   0),  INT8_C(  36),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  62),  INT8_C(   0), -INT8_C(  65),  INT8_C(  15), -INT8_C(  39) } },
    { UINT32_C(2587593567),
      {  INT8_C( 111), -INT8_C(  68), -INT8_C(  79), -INT8_C( 124), -INT8_C(  31),  INT8_C(  85), -INT8_C(  31), -INT8_C(  87),
         INT8_C(  13),  INT8_C(   8),  INT8_C(  21),  INT8_C(  34),  INT8_C( 122), -INT8_C(  83),  INT8_C(  53), -INT8_C(  44),
         INT8_C( 113),  INT8_C(  62),  INT8_C( 101), -INT8_C(  13),  INT8_C(  33),  INT8_C( 108),  INT8_C(  43),  INT8_C(  41),
        -INT8_C(  46),  INT8_C( 123),  INT8_C( 103),  INT8_C(  49),  INT8_C(   6), -INT8_C(  94), -INT8_C(  53),  INT8_C( 118) },
      {  INT8_C(  94),  INT8_C( 124), -INT8_C(   6),  INT8_C(  63), -INT8_C(  47), -INT8_C(  37), -INT8_C(  23), -INT8_C(  33),
        -INT8_C(  29), -INT8_C(   2),  INT8_C(   1),  INT8_C(  93), -INT8_C(  85),  INT8_C(  55),  INT8_C(  50),  INT8_C(  29),
         INT8_C( 117), -INT8_C( 105),  INT8_C(  16), -INT8_C( 105),  INT8_C(   4),  INT8_C(  60), -INT8_C(  64), -INT8_C(  42),
        -INT8_C(  73),  INT8_C(  39),  INT8_C(   7), -INT8_C(  67), -INT8_C(  55), -INT8_C(  46),  INT8_C(  51),  INT8_C(  40) },
      {  INT8_C(  17),  INT8_C(  64), -INT8_C(  73),  INT8_C(  69),  INT8_C(  16),  INT8_C(   0), -INT8_C(   8),  INT8_C(   0),
         INT8_C(  42),  INT8_C(  10),  INT8_C(   0), -INT8_C(  59),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  73),
        -INT8_C(   4), -INT8_C(  89),  INT8_C(   0),  INT8_C(  92),  INT8_C(  29),  INT8_C(  48),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  84),  INT8_C(   0),  INT8_C( 116),  INT8_C(  61),  INT8_C(   0),  INT8_C(   0),  INT8_C(  78) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sub_epi8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_sub_epi8");
    easysimd_test_x86_assert_equal_i8x32(r, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = easysimd_mm256_maskz_sub_epi8(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_sub_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[16];
    const uint16_t k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { {  INT16_C(  9865), -INT16_C(  4433), -INT16_C(  7573),  INT16_C( 28514), -INT16_C(  3176),  INT16_C( 22060),  INT16_C(   588),  INT16_C( 27854),
         INT16_C( 17955), -INT16_C(  8423), -INT16_C( 23460), -INT16_C( 14597), -INT16_C(  6936), -INT16_C( 17217),  INT16_C( 18129),  INT16_C( 23359) },
      UINT16_C(61036),
      { -INT16_C( 10423), -INT16_C( 21552),  INT16_C( 26695),  INT16_C( 29598), -INT16_C(  5442), -INT16_C( 29578), -INT16_C( 26281),  INT16_C( 28882),
         INT16_C( 11896),  INT16_C( 29460), -INT16_C(   779), -INT16_C( 19368),  INT16_C( 10681), -INT16_C(  1798),  INT16_C( 26244), -INT16_C( 12826) },
      { -INT16_C( 18882), -INT16_C( 31367),  INT16_C(  5918), -INT16_C(  8968),  INT16_C( 28162),  INT16_C( 22889),  INT16_C( 15111),  INT16_C( 32713),
        -INT16_C(  8854),  INT16_C( 24563),  INT16_C( 19418), -INT16_C( 27885),  INT16_C(  3444), -INT16_C(  1653),  INT16_C( 29043), -INT16_C( 20026) },
      {  INT16_C(  9865), -INT16_C(  4433),  INT16_C( 20777), -INT16_C( 26970), -INT16_C(  3176),  INT16_C( 13069),  INT16_C( 24144),  INT16_C( 27854),
         INT16_C( 17955),  INT16_C(  4897), -INT16_C( 20197),  INT16_C(  8517), -INT16_C(  6936), -INT16_C(   145), -INT16_C(  2799),  INT16_C(  7200) } },
    { {  INT16_C( 16167),  INT16_C( 17718),  INT16_C( 12119),  INT16_C( 22817), -INT16_C( 30051), -INT16_C( 23118),  INT16_C( 31686),  INT16_C( 12324),
         INT16_C(  5976),  INT16_C( 12943), -INT16_C( 23966), -INT16_C( 10299),  INT16_C( 20655),  INT16_C(  8912), -INT16_C( 26943), -INT16_C(  5932) },
      UINT16_C( 2774),
      {  INT16_C( 11565),  INT16_C( 20281), -INT16_C( 10362),  INT16_C( 14553), -INT16_C( 24708), -INT16_C( 24397),  INT16_C(  3023),  INT16_C( 24248),
         INT16_C(  6718),  INT16_C(   768), -INT16_C( 20495), -INT16_C( 16044),  INT16_C(  5586), -INT16_C( 22952),  INT16_C( 12030),  INT16_C( 11184) },
      { -INT16_C(  5541), -INT16_C(  7814),  INT16_C( 21697),  INT16_C( 15641), -INT16_C( 13069), -INT16_C( 15395), -INT16_C( 27177),  INT16_C(  5409),
         INT16_C(  8880), -INT16_C( 24295),  INT16_C( 28113), -INT16_C( 23709), -INT16_C( 17534), -INT16_C( 32695), -INT16_C(  1303),  INT16_C( 17580) },
      {  INT16_C( 16167),  INT16_C( 28095), -INT16_C( 32059),  INT16_C( 22817), -INT16_C( 11639), -INT16_C( 23118),  INT16_C( 30200),  INT16_C( 18839),
         INT16_C(  5976),  INT16_C( 25063), -INT16_C( 23966),  INT16_C(  7665),  INT16_C( 20655),  INT16_C(  8912), -INT16_C( 26943), -INT16_C(  5932) } },
    { {  INT16_C(  9956), -INT16_C( 23259),  INT16_C( 15994),  INT16_C( 28386), -INT16_C( 16630), -INT16_C(  7887),  INT16_C( 21077),  INT16_C(  1527),
         INT16_C(  4212),  INT16_C( 18086),  INT16_C(  2429), -INT16_C(    23),  INT16_C( 13252), -INT16_C( 21120),  INT16_C( 11309),  INT16_C(  4593) },
      UINT16_C( 5714),
      { -INT16_C( 12874), -INT16_C( 26540),  INT16_C( 24123),  INT16_C( 27735), -INT16_C( 21440),  INT16_C( 14270),  INT16_C( 13233),  INT16_C( 22599),
        -INT16_C( 15239),  INT16_C( 25185),  INT16_C(  9923),  INT16_C( 17301), -INT16_C( 15661), -INT16_C( 14993), -INT16_C( 15661), -INT16_C( 30245) },
      {  INT16_C( 12431), -INT16_C( 13791),  INT16_C( 31118), -INT16_C( 12746), -INT16_C(  3035), -INT16_C( 10491),  INT16_C( 19495), -INT16_C( 24529),
        -INT16_C( 28656), -INT16_C( 11261), -INT16_C( 26442), -INT16_C( 30185), -INT16_C( 30885),  INT16_C( 11855),  INT16_C( 10825), -INT16_C( 10056) },
      {  INT16_C(  9956), -INT16_C( 12749),  INT16_C( 15994),  INT16_C( 28386), -INT16_C( 18405), -INT16_C(  7887), -INT16_C(  6262),  INT16_C(  1527),
         INT16_C(  4212), -INT16_C( 29090), -INT16_C( 29171), -INT16_C(    23),  INT16_C( 15224), -INT16_C( 21120),  INT16_C( 11309),  INT16_C(  4593) } },
    { { -INT16_C(  9894), -INT16_C(  5726), -INT16_C( 10158),  INT16_C( 30903), -INT16_C( 16948), -INT16_C(  2993),  INT16_C( 32265),  INT16_C(  6804),
        -INT16_C( 26866), -INT16_C( 14866),  INT16_C(  1328), -INT16_C( 29873), -INT16_C( 24948), -INT16_C( 10823),  INT16_C( 29128),  INT16_C(  9133) },
      UINT16_C(20299),
      { -INT16_C( 25332), -INT16_C( 15577), -INT16_C(  3051),  INT16_C( 25728), -INT16_C( 29976),  INT16_C( 31970), -INT16_C(  3676), -INT16_C( 28140),
         INT16_C( 17590),  INT16_C(  1431),  INT16_C(  9423), -INT16_C( 30557),  INT16_C( 27641), -INT16_C( 22534),  INT16_C( 17806), -INT16_C( 25866) },
      {  INT16_C(  7906), -INT16_C(  1954), -INT16_C(  8686), -INT16_C(  1444),  INT16_C( 16232),  INT16_C(  3190), -INT16_C( 30160), -INT16_C(  6498),
         INT16_C( 14030), -INT16_C( 25109), -INT16_C( 29094),  INT16_C( 21286),  INT16_C(  8441), -INT16_C( 30470), -INT16_C(  3739),  INT16_C( 18210) },
      {  INT16_C( 32298), -INT16_C( 13623), -INT16_C( 10158),  INT16_C( 27172), -INT16_C( 16948), -INT16_C(  2993),  INT16_C( 26484),  INT16_C(  6804),
         INT16_C(  3560),  INT16_C( 26540), -INT16_C( 27019),  INT16_C( 13693), -INT16_C( 24948), -INT16_C( 10823),  INT16_C( 21545),  INT16_C(  9133) } },
    { { -INT16_C( 32753),  INT16_C(  8511), -INT16_C( 25505), -INT16_C( 14565), -INT16_C( 28197),  INT16_C(  3028),  INT16_C( 29212), -INT16_C(  5391),
        -INT16_C(  9048),  INT16_C(   648), -INT16_C( 20886),  INT16_C( 25430),  INT16_C( 20686),  INT16_C( 13291),  INT16_C(  3649),  INT16_C( 20602) },
      UINT16_C(47758),
      { -INT16_C(  4751), -INT16_C( 29610),  INT16_C( 12725), -INT16_C( 30434),  INT16_C( 14908),  INT16_C( 11771), -INT16_C( 23516), -INT16_C( 21495),
         INT16_C( 29606), -INT16_C(   934),  INT16_C( 10454), -INT16_C( 15795), -INT16_C( 29093), -INT16_C( 10544),  INT16_C( 24287),  INT16_C( 20624) },
      { -INT16_C(  6580),  INT16_C(   477), -INT16_C(  1257),  INT16_C( 21386), -INT16_C( 31435),  INT16_C( 22912), -INT16_C( 30423), -INT16_C( 12282),
         INT16_C( 24828), -INT16_C( 11572),  INT16_C(  6537), -INT16_C(  7020),  INT16_C( 25768), -INT16_C( 30790),  INT16_C( 19139),  INT16_C(  4055) },
      { -INT16_C( 32753), -INT16_C( 30087),  INT16_C( 13982),  INT16_C( 13716), -INT16_C( 28197),  INT16_C(  3028),  INT16_C( 29212), -INT16_C(  9213),
        -INT16_C(  9048),  INT16_C( 10638), -INT16_C( 20886), -INT16_C(  8775),  INT16_C( 10675),  INT16_C( 20246),  INT16_C(  3649),  INT16_C( 16569) } },
    { { -INT16_C( 19408),  INT16_C( 18192), -INT16_C( 25937), -INT16_C(  7014),  INT16_C(  6687),  INT16_C( 18750),  INT16_C( 17571), -INT16_C( 24807),
        -INT16_C(  6748),  INT16_C( 11634),  INT16_C(  1791), -INT16_C( 22766), -INT16_C( 13205),  INT16_C( 11822),  INT16_C(  1303),  INT16_C( 18237) },
      UINT16_C(19898),
      {  INT16_C( 27023),  INT16_C( 10727),  INT16_C(  1614), -INT16_C( 29628), -INT16_C(  6321),  INT16_C( 26832),  INT16_C( 29831), -INT16_C(  1714),
         INT16_C( 19874), -INT16_C( 19201),  INT16_C( 27380),  INT16_C(  8832), -INT16_C( 26728), -INT16_C( 10969), -INT16_C(  7713),  INT16_C( 28194) },
      {  INT16_C(  2379), -INT16_C( 26217), -INT16_C(  9456),  INT16_C( 24357), -INT16_C(  2621),  INT16_C( 19144),  INT16_C(  5737),  INT16_C(  2883),
         INT16_C( 16995),  INT16_C( 22463),  INT16_C( 16557),  INT16_C( 17785), -INT16_C( 24361), -INT16_C( 18917),  INT16_C( 15746), -INT16_C( 13020) },
      { -INT16_C( 19408), -INT16_C( 28592), -INT16_C( 25937),  INT16_C( 11551), -INT16_C(  3700),  INT16_C(  7688),  INT16_C( 17571), -INT16_C(  4597),
         INT16_C(  2879),  INT16_C( 11634),  INT16_C( 10823), -INT16_C(  8953), -INT16_C( 13205),  INT16_C( 11822), -INT16_C( 23459),  INT16_C( 18237) } },
    { { -INT16_C( 17337),  INT16_C( 22374), -INT16_C( 29801),  INT16_C( 23222),  INT16_C( 32384), -INT16_C(  5724), -INT16_C(  6252), -INT16_C(  2059),
        -INT16_C( 19414), -INT16_C( 10418), -INT16_C( 14348), -INT16_C( 13284),  INT16_C( 14184), -INT16_C(  5502), -INT16_C( 22667), -INT16_C( 17225) },
      UINT16_C( 7523),
      { -INT16_C(  1517), -INT16_C( 13912),  INT16_C( 10325), -INT16_C(  1720), -INT16_C(  9199),  INT16_C(  1761),  INT16_C(  3028),  INT16_C(  8891),
        -INT16_C( 20510), -INT16_C(   278),  INT16_C( 21115), -INT16_C(   458), -INT16_C( 21700), -INT16_C(  3163),  INT16_C(  2151),  INT16_C( 31248) },
      { -INT16_C( 18430),  INT16_C( 22339), -INT16_C( 29728), -INT16_C(  3759),  INT16_C( 12904),  INT16_C( 15608), -INT16_C( 19651),  INT16_C(  8030),
         INT16_C( 18530), -INT16_C(  8675),  INT16_C( 21402), -INT16_C( 10532), -INT16_C( 32258),  INT16_C( 26057), -INT16_C(  9847), -INT16_C( 29729) },
      {  INT16_C( 16913),  INT16_C( 29285), -INT16_C( 29801),  INT16_C( 23222),  INT16_C( 32384), -INT16_C( 13847),  INT16_C( 22679), -INT16_C(  2059),
         INT16_C( 26496), -INT16_C( 10418), -INT16_C(   287),  INT16_C( 10074),  INT16_C( 10558), -INT16_C(  5502), -INT16_C( 22667), -INT16_C( 17225) } },
    { {  INT16_C(  9105),  INT16_C( 29155),  INT16_C( 13486),  INT16_C(  5731),  INT16_C( 23398), -INT16_C( 23726), -INT16_C( 20210),  INT16_C( 28866),
        -INT16_C(  8199), -INT16_C( 27570),  INT16_C( 10803),  INT16_C( 12650),  INT16_C( 13483),  INT16_C( 13463),  INT16_C( 30221), -INT16_C( 24640) },
      UINT16_C(41881),
      {  INT16_C( 18448),  INT16_C( 29655),  INT16_C( 15710), -INT16_C( 20018), -INT16_C(  8992), -INT16_C( 23966),  INT16_C( 23373), -INT16_C( 25727),
        -INT16_C( 19217),  INT16_C( 23238),  INT16_C( 29158),  INT16_C( 32142), -INT16_C( 25690),  INT16_C( 26355), -INT16_C( 29382),  INT16_C( 19209) },
      { -INT16_C(  7979),  INT16_C( 13246), -INT16_C( 29411), -INT16_C(   540),  INT16_C( 18025), -INT16_C( 18785),  INT16_C(  8354), -INT16_C( 28334),
         INT16_C(  6357), -INT16_C( 17429),  INT16_C( 31113),  INT16_C( 12088),  INT16_C( 11029),  INT16_C( 20373), -INT16_C( 24904), -INT16_C( 29286) },
      {  INT16_C( 26427),  INT16_C( 29155),  INT16_C( 13486), -INT16_C( 19478), -INT16_C( 27017), -INT16_C( 23726), -INT16_C( 20210),  INT16_C(  2607),
        -INT16_C( 25574), -INT16_C( 24869),  INT16_C( 10803),  INT16_C( 12650),  INT16_C( 13483),  INT16_C(  5982),  INT16_C( 30221), -INT16_C( 17041) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sub_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_sub_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_mask_sub_epi16(src, k, a, b);

    easysimd_test_x86_write_i16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_sub_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C(11854),
      {  INT16_C(  8295),  INT16_C( 20489), -INT16_C(  4609),  INT16_C(    79), -INT16_C(  1462),  INT16_C( 31799), -INT16_C( 21225),  INT16_C( 10260),
         INT16_C(  6212),  INT16_C(  1124),  INT16_C(  7150), -INT16_C(  2773), -INT16_C(  2856),  INT16_C(  3271),  INT16_C(  5404), -INT16_C( 31686) },
      {  INT16_C( 17205),  INT16_C( 13524),  INT16_C(  9008),  INT16_C( 31541),  INT16_C( 27678),  INT16_C( 13815),  INT16_C(  2841),  INT16_C( 23901),
        -INT16_C( 16093),  INT16_C(  4449), -INT16_C( 29476), -INT16_C( 19194), -INT16_C( 12927), -INT16_C( 25151), -INT16_C(  1053),  INT16_C(  6177) },
      {  INT16_C(     0),  INT16_C(  6965), -INT16_C( 13617), -INT16_C( 31462),  INT16_C(     0),  INT16_C(     0), -INT16_C( 24066),  INT16_C(     0),
         INT16_C(     0), -INT16_C(  3325), -INT16_C( 28910),  INT16_C( 16421),  INT16_C(     0),  INT16_C( 28422),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(63038),
      {  INT16_C( 28493), -INT16_C( 32231),  INT16_C( 14314), -INT16_C(  7698),  INT16_C(  2157), -INT16_C( 13587),  INT16_C(  4197), -INT16_C( 14452),
         INT16_C( 26658),  INT16_C( 10323), -INT16_C( 11235), -INT16_C(  8458), -INT16_C(  9870), -INT16_C( 27687),  INT16_C(  6385),  INT16_C( 16009) },
      { -INT16_C( 23673),  INT16_C( 29120), -INT16_C( 20518),  INT16_C( 18258),  INT16_C( 16311),  INT16_C(  7186), -INT16_C( 25008),  INT16_C( 29411),
         INT16_C( 14086),  INT16_C(  9370), -INT16_C( 28661),  INT16_C( 32002), -INT16_C(  9111),  INT16_C( 23313), -INT16_C( 25868),  INT16_C( 31641) },
      {  INT16_C(     0),  INT16_C(  4185), -INT16_C( 30704), -INT16_C( 25956), -INT16_C( 14154), -INT16_C( 20773),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(   953),  INT16_C( 17426),  INT16_C(     0), -INT16_C(   759),  INT16_C( 14536),  INT16_C( 32253), -INT16_C( 15632) } },
    { UINT16_C(23101),
      {  INT16_C(  6380),  INT16_C( 15881), -INT16_C( 16289),  INT16_C( 29054), -INT16_C( 12580), -INT16_C( 16369),  INT16_C(  5696), -INT16_C(  9481),
         INT16_C(   570),  INT16_C( 15467), -INT16_C( 11136), -INT16_C( 28392),  INT16_C(  3119), -INT16_C( 14037),  INT16_C( 27015),  INT16_C( 29475) },
      {  INT16_C( 11393), -INT16_C(  8014),  INT16_C( 12524), -INT16_C( 14254),  INT16_C( 25086),  INT16_C( 16008),  INT16_C( 32631), -INT16_C( 20200),
        -INT16_C( 31870),  INT16_C(   750),  INT16_C(  1624), -INT16_C( 30829), -INT16_C( 16877), -INT16_C( 26032),  INT16_C( 29479), -INT16_C( 22514) },
      { -INT16_C(  5013),  INT16_C(     0), -INT16_C( 28813), -INT16_C( 22228),  INT16_C( 27870), -INT16_C( 32377),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 14717),  INT16_C(     0),  INT16_C(  2437),  INT16_C( 19996),  INT16_C(     0), -INT16_C(  2464),  INT16_C(     0) } },
    { UINT16_C(49311),
      { -INT16_C( 29815), -INT16_C(  9232), -INT16_C(  4524), -INT16_C(  9156), -INT16_C( 19412),  INT16_C( 17500), -INT16_C(  8603),  INT16_C( 21448),
         INT16_C(  8416),  INT16_C( 29530),  INT16_C( 28071), -INT16_C(  1999),  INT16_C( 22791),  INT16_C(  5483),  INT16_C(  2817), -INT16_C( 29995) },
      { -INT16_C( 14954), -INT16_C(  5531), -INT16_C( 23885), -INT16_C(  8249),  INT16_C(  9046), -INT16_C( 17628), -INT16_C(  5119), -INT16_C(  7921),
         INT16_C( 26892), -INT16_C( 19628), -INT16_C( 31274), -INT16_C(  8789),  INT16_C(  6110), -INT16_C(  7949), -INT16_C( 14302), -INT16_C( 18326) },
      { -INT16_C( 14861), -INT16_C(  3701),  INT16_C( 19361), -INT16_C(   907), -INT16_C( 28458),  INT16_C(     0),  INT16_C(     0),  INT16_C( 29369),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 17119), -INT16_C( 11669) } },
    { UINT16_C(53390),
      {  INT16_C( 16803),  INT16_C( 27250), -INT16_C( 14303),  INT16_C( 17805), -INT16_C( 29053), -INT16_C( 28111),  INT16_C( 15727), -INT16_C( 15365),
        -INT16_C( 11792), -INT16_C( 25528),  INT16_C( 10159), -INT16_C( 23885), -INT16_C( 11001),  INT16_C( 29034), -INT16_C(  1907),  INT16_C( 12353) },
      { -INT16_C( 19654),  INT16_C( 23450),  INT16_C( 10107), -INT16_C(    96), -INT16_C( 11851),  INT16_C(  9361), -INT16_C( 29426), -INT16_C(   281),
         INT16_C( 12382),  INT16_C(  3482),  INT16_C( 19799),  INT16_C( 24239),  INT16_C(  6690), -INT16_C( 20273),  INT16_C(  4370),  INT16_C( 19680) },
      {  INT16_C(     0),  INT16_C(  3800), -INT16_C( 24410),  INT16_C( 17901),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 15084),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 17691),  INT16_C(     0), -INT16_C(  6277), -INT16_C(  7327) } },
    { UINT16_C(31684),
      {  INT16_C( 16551),  INT16_C( 18338),  INT16_C( 22591), -INT16_C( 12264),  INT16_C(  9852),  INT16_C( 25693), -INT16_C( 17371), -INT16_C( 16492),
        -INT16_C(  5175),  INT16_C( 30989),  INT16_C( 12105),  INT16_C(  6291), -INT16_C( 23073), -INT16_C( 16343), -INT16_C(  4366), -INT16_C( 26309) },
      { -INT16_C(  8914),  INT16_C( 28129), -INT16_C(  1739), -INT16_C( 19907), -INT16_C( 25824),  INT16_C( 17686), -INT16_C( 21929),  INT16_C(  8196),
         INT16_C(  4501), -INT16_C(  8551),  INT16_C( 11329),  INT16_C(  8438),  INT16_C(  8402), -INT16_C( 15136),  INT16_C(  6926),  INT16_C( 15453) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 24330),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4558), -INT16_C( 24688),
        -INT16_C(  9676), -INT16_C( 25996),  INT16_C(     0), -INT16_C(  2147), -INT16_C( 31475), -INT16_C(  1207), -INT16_C( 11292),  INT16_C(     0) } },
    { UINT16_C(16121),
      {  INT16_C( 11945), -INT16_C(  6600),  INT16_C( 22752), -INT16_C(  2431), -INT16_C( 10083), -INT16_C( 24160),  INT16_C( 13817), -INT16_C( 27981),
        -INT16_C(  3053),  INT16_C(  2751), -INT16_C( 28396), -INT16_C(  2774),  INT16_C( 14421), -INT16_C( 19952),  INT16_C(  2420),  INT16_C(  7665) },
      {  INT16_C( 10552),  INT16_C(  6147), -INT16_C( 31359),  INT16_C(  7695), -INT16_C( 20643),  INT16_C( 22207),  INT16_C( 29413), -INT16_C(  1815),
        -INT16_C( 22426),  INT16_C( 31490),  INT16_C( 11321), -INT16_C( 29072), -INT16_C( 32668), -INT16_C( 10176),  INT16_C( 12682), -INT16_C( 15627) },
      {  INT16_C(  1393),  INT16_C(     0),  INT16_C(     0), -INT16_C( 10126),  INT16_C( 10560),  INT16_C( 19169), -INT16_C( 15596), -INT16_C( 26166),
         INT16_C(     0), -INT16_C( 28739),  INT16_C( 25819),  INT16_C( 26298), -INT16_C( 18447), -INT16_C(  9776),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(63834),
      { -INT16_C(  9254), -INT16_C(  5762), -INT16_C(  9223), -INT16_C( 18023),  INT16_C( 32306),  INT16_C(  6955), -INT16_C( 28042),  INT16_C( 31171),
        -INT16_C(  1011),  INT16_C( 32165),  INT16_C(  2698), -INT16_C( 13571), -INT16_C( 30750), -INT16_C(  9988),  INT16_C( 22089),  INT16_C(  9425) },
      {  INT16_C( 20274),  INT16_C( 11021), -INT16_C( 22998),  INT16_C( 23780),  INT16_C(  4132), -INT16_C( 25737),  INT16_C( 15010), -INT16_C( 20716),
        -INT16_C( 18122), -INT16_C( 16340),  INT16_C( 10691), -INT16_C( 22901), -INT16_C( 30799), -INT16_C(  1410),  INT16_C( 20445),  INT16_C(  3870) },
      {  INT16_C(     0), -INT16_C( 16783),  INT16_C(     0),  INT16_C( 23733),  INT16_C( 28174),  INT16_C(     0),  INT16_C( 22484),  INT16_C(     0),
         INT16_C( 17111),  INT16_C(     0),  INT16_C(     0),  INT16_C(  9330),  INT16_C(    49), -INT16_C(  8578),  INT16_C(  1644),  INT16_C(  5555) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sub_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_sub_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_maskz_sub_epi16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_sub_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1681827458),  INT32_C(  1335404006), -INT32_C(  1912195092),  INT32_C(   757028952), -INT32_C(   102233232), -INT32_C(  1725357948),  INT32_C(    82427467), -INT32_C(   611220643) },
      UINT8_C(220),
      { -INT32_C(   121473198),  INT32_C(  1206129167),  INT32_C(  1889497624), -INT32_C(  1662989167), -INT32_C(   735979084), -INT32_C(  1054885374),  INT32_C(   656286627),  INT32_C(   117701045) },
      { -INT32_C(  2130721168), -INT32_C(   255335464), -INT32_C(   413112747), -INT32_C(   427605966),  INT32_C(   481993753),  INT32_C(    48093535), -INT32_C(  1305871363),  INT32_C(  1706634740) },
      { -INT32_C(  1681827458),  INT32_C(  1335404006), -INT32_C(  1992356925), -INT32_C(  1235383201), -INT32_C(  1217972837), -INT32_C(  1725357948),  INT32_C(  1962157990), -INT32_C(  1588933695) } },
    { { -INT32_C(   874137101), -INT32_C(   222581604),  INT32_C(  1155078930),  INT32_C(  1948933211),  INT32_C(  1620108289), -INT32_C(  1167954499),  INT32_C(  1567394920), -INT32_C(  1396562247) },
      UINT8_C(223),
      {  INT32_C(  1400600487),  INT32_C(  1298492722), -INT32_C(  1549227706), -INT32_C(  1213981485),  INT32_C(   443810989), -INT32_C(   226283674), -INT32_C(  1045700453),  INT32_C(  1235244962) },
      {  INT32_C(    10230990), -INT32_C(   800259703),  INT32_C(  2104751530), -INT32_C(  1087105262), -INT32_C(  2116442085),  INT32_C(  1937005783), -INT32_C(   567009476),  INT32_C(  1160238455) },
      {  INT32_C(  1390369497),  INT32_C(  2098752425),  INT32_C(   640988060), -INT32_C(   126876223), -INT32_C(  1734714222), -INT32_C(  1167954499), -INT32_C(   478690977),  INT32_C(    75006507) } },
    { {  INT32_C(  2051458033),  INT32_C(  1850381252), -INT32_C(  1679049335), -INT32_C(   279306284), -INT32_C(  1619971128), -INT32_C(   871176816),  INT32_C(  2074822404),  INT32_C(   230806044) },
      UINT8_C(150),
      { -INT32_C(  1705343225), -INT32_C(  1893479982), -INT32_C(   731595084),  INT32_C(  1285313304), -INT32_C(  1445184572), -INT32_C(  1783715762),  INT32_C(   649144659), -INT32_C(   239288598) },
      {  INT32_C(   395056709), -INT32_C(  1800949793), -INT32_C(  2040001682),  INT32_C(   600966238), -INT32_C(  1915965889), -INT32_C(  1440581033), -INT32_C(  1898916956), -INT32_C(   679441263) },
      {  INT32_C(  2051458033), -INT32_C(    92530189),  INT32_C(  1308406598), -INT32_C(   279306284),  INT32_C(   470781317), -INT32_C(   871176816),  INT32_C(  2074822404),  INT32_C(   440152665) } },
    { { -INT32_C(  2098328413),  INT32_C(   689345979), -INT32_C(     5210464), -INT32_C(  1037925758), -INT32_C(  2008027599),  INT32_C(   221409897), -INT32_C(   677706939),  INT32_C(   850271119) },
      UINT8_C( 39),
      {  INT32_C(   853783964),  INT32_C(  1255279819),  INT32_C(  1070387644), -INT32_C(   495939853),  INT32_C(  1347156190),  INT32_C(   764762154),  INT32_C(   264072435), -INT32_C(  1221136614) },
      {  INT32_C(  1894324644), -INT32_C(   491078874), -INT32_C(  2145286515), -INT32_C(   211644139), -INT32_C(  1270633079), -INT32_C(   102639611),  INT32_C(  1594400325),  INT32_C(   840318606) },
      { -INT32_C(  1040540680),  INT32_C(  1746358693), -INT32_C(  1079293137), -INT32_C(  1037925758), -INT32_C(  2008027599),  INT32_C(   867401765), -INT32_C(   677706939),  INT32_C(   850271119) } },
    { {  INT32_C(  2124611416),  INT32_C(  1214274747), -INT32_C(   121077021), -INT32_C(  1645532397), -INT32_C(   565104936),  INT32_C(  1272394246),  INT32_C(  1605099473),  INT32_C(  1972486429) },
      UINT8_C(193),
      { -INT32_C(  1870859468), -INT32_C(   697056172), -INT32_C(  1209439348), -INT32_C(  2070903210),  INT32_C(   176910039),  INT32_C(   618387013),  INT32_C(  1128348289), -INT32_C(    16468021) },
      { -INT32_C(    24084310), -INT32_C(   791411900), -INT32_C(   980894097),  INT32_C(   474617924), -INT32_C(   886647418),  INT32_C(   770638251),  INT32_C(   108015675), -INT32_C(  1845070617) },
      { -INT32_C(  1846775158),  INT32_C(  1214274747), -INT32_C(   121077021), -INT32_C(  1645532397), -INT32_C(   565104936),  INT32_C(  1272394246),  INT32_C(  1020332614),  INT32_C(  1828602596) } },
    { {  INT32_C(   948999924),  INT32_C(   134768025),  INT32_C(  1741590563),  INT32_C(   780343464), -INT32_C(  1728468499), -INT32_C(   440014678),  INT32_C(    15480089), -INT32_C(  1651314007) },
      UINT8_C(136),
      { -INT32_C(  2011048669),  INT32_C(  1856711390),  INT32_C(   269947640),  INT32_C(  1073563030),  INT32_C(   669685055), -INT32_C(  1874800805), -INT32_C(  1388691013), -INT32_C(   164243501) },
      { -INT32_C(  1954654291),  INT32_C(  2046372225), -INT32_C(   762769348), -INT32_C(  1794013610),  INT32_C(  1992096539), -INT32_C(  2046296629),  INT32_C(   288637246), -INT32_C(   989304552) },
      {  INT32_C(   948999924),  INT32_C(   134768025),  INT32_C(  1741590563), -INT32_C(  1427390656), -INT32_C(  1728468499), -INT32_C(   440014678),  INT32_C(    15480089),  INT32_C(   825061051) } },
    { {  INT32_C(  1095796416), -INT32_C(   323335504), -INT32_C(  1329708198), -INT32_C(   448409655), -INT32_C(  1772420405),  INT32_C(  1025270527), -INT32_C(  1135718237),  INT32_C(  2055296698) },
      UINT8_C(221),
      {  INT32_C(   479050962), -INT32_C(  1166640778),  INT32_C(   109258551),  INT32_C(  1859217516),  INT32_C(   644704196), -INT32_C(   724915580),  INT32_C(  1334740729), -INT32_C(   651425529) },
      {  INT32_C(  1005959621),  INT32_C(  1777691698), -INT32_C(     9406061), -INT32_C(  1519500831),  INT32_C(   768400297),  INT32_C(  2130810502),  INT32_C(   600739868),  INT32_C(  1593703321) },
      { -INT32_C(   526908659), -INT32_C(   323335504),  INT32_C(   118664612), -INT32_C(   916248949), -INT32_C(   123696101),  INT32_C(  1025270527),  INT32_C(   734000861),  INT32_C(  2049838446) } },
    { { -INT32_C(   409341260), -INT32_C(   246378658), -INT32_C(   370032632), -INT32_C(  1416667390), -INT32_C(  1059562694),  INT32_C(   222288369),  INT32_C(    36703849), -INT32_C(  1117770487) },
      UINT8_C( 32),
      { -INT32_C(  1988188934), -INT32_C(  1248759563), -INT32_C(  1061717407),  INT32_C(  1694130697),  INT32_C(   341162810),  INT32_C(   142435066), -INT32_C(  1072594797), -INT32_C(   622800928) },
      {  INT32_C(  1751342963),  INT32_C(   807335119),  INT32_C(  2029049199),  INT32_C(  1927146040), -INT32_C(  1601817947),  INT32_C(   682099861),  INT32_C(  1692973700), -INT32_C(    62928503) },
      { -INT32_C(   409341260), -INT32_C(   246378658), -INT32_C(   370032632), -INT32_C(  1416667390), -INT32_C(  1059562694), -INT32_C(   539664795),  INT32_C(    36703849), -INT32_C(  1117770487) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sub_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_sub_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_mask_sub_epi32(src, k, a, b);

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
test_easysimd_mm256_maskz_sub_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(158),
      { -INT32_C(   758629588),  INT32_C(   804726047), -INT32_C(   674131300),  INT32_C(  1594720422), -INT32_C(   702296404), -INT32_C(   527972007), -INT32_C(  1765965497), -INT32_C(   868954720) },
      {  INT32_C(   664730632), -INT32_C(  1101556447), -INT32_C(   845862873),  INT32_C(  1429054121), -INT32_C(   919908240),  INT32_C(  1621733913), -INT32_C(   722049228),  INT32_C(  1017129524) },
      {  INT32_C(           0),  INT32_C(  1906282494),  INT32_C(   171731573),  INT32_C(   165666301),  INT32_C(   217611836),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1886084244) } },
    { UINT8_C( 38),
      { -INT32_C(   733453506), -INT32_C(   470088006),  INT32_C(  1032636827),  INT32_C(  1185866230), -INT32_C(  1084262643), -INT32_C(  1997291743), -INT32_C(   541276235), -INT32_C(  1509558169) },
      {  INT32_C(   377113947), -INT32_C(   285641389),  INT32_C(   875332926),  INT32_C(  1937431142),  INT32_C(  1916000593),  INT32_C(  1308239512),  INT32_C(  1428993774),  INT32_C(   184234670) },
      {  INT32_C(           0), -INT32_C(   184446617),  INT32_C(   157303901),  INT32_C(           0),  INT32_C(           0),  INT32_C(   989436041),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(127),
      { -INT32_C(   338485131), -INT32_C(  1641430759), -INT32_C(   956014867), -INT32_C(  1306953512), -INT32_C(   783643733),  INT32_C(  1019189382), -INT32_C(   152365628), -INT32_C(  2039024368) },
      {  INT32_C(   779176213), -INT32_C(   120808950), -INT32_C(   775958280), -INT32_C(   209463224), -INT32_C(   356135580),  INT32_C(   707167333), -INT32_C(  1440738919),  INT32_C(   456168966) },
      { -INT32_C(  1117661344), -INT32_C(  1520621809), -INT32_C(   180056587), -INT32_C(  1097490288), -INT32_C(   427508153),  INT32_C(   312022049),  INT32_C(  1288373291),  INT32_C(           0) } },
    { UINT8_C(223),
      {  INT32_C(  1005210017), -INT32_C(   432741867),  INT32_C(  2049836449),  INT32_C(  1440621192),  INT32_C(  1807468775),  INT32_C(      386542), -INT32_C(  1677283579), -INT32_C(  2139413793) },
      { -INT32_C(  2118425237), -INT32_C(   379064505),  INT32_C(  2086901236), -INT32_C(  1630387785), -INT32_C(   150303479),  INT32_C(  2012680050), -INT32_C(  1659633986), -INT32_C(  1960997088) },
      { -INT32_C(  1171332042), -INT32_C(    53677362), -INT32_C(    37064787), -INT32_C(  1223958319),  INT32_C(  1957772254),  INT32_C(           0), -INT32_C(    17649593), -INT32_C(   178416705) } },
    { UINT8_C(244),
      { -INT32_C(   935588648),  INT32_C(   163325299), -INT32_C(   910149240), -INT32_C(  1731043573),  INT32_C(  2013972841), -INT32_C(  1086946879),  INT32_C(   618648469), -INT32_C(   937923856) },
      { -INT32_C(   359639945),  INT32_C(    32722297), -INT32_C(  1848986490),  INT32_C(  2083101715),  INT32_C(   670315366),  INT32_C(  1239820980), -INT32_C(   311572995), -INT32_C(  1498053073) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   938837250),  INT32_C(           0),  INT32_C(  1343657475),  INT32_C(  1968199437),  INT32_C(   930221464),  INT32_C(   560129217) } },
    { UINT8_C(218),
      { -INT32_C(  1823174330),  INT32_C(   941184388), -INT32_C(  1135891936),  INT32_C(   102942675), -INT32_C(   440776261), -INT32_C(   186514385), -INT32_C(   148582543), -INT32_C(   875378043) },
      { -INT32_C(   530700709), -INT32_C(  1692895365), -INT32_C(   195533791), -INT32_C(   403015125), -INT32_C(   204688188),  INT32_C(   719892408),  INT32_C(    52497534),  INT32_C(   852423638) },
      {  INT32_C(           0), -INT32_C(  1660887543),  INT32_C(           0),  INT32_C(   505957800), -INT32_C(   236088073),  INT32_C(           0), -INT32_C(   201080077), -INT32_C(  1727801681) } },
    { UINT8_C( 25),
      { -INT32_C(  1550511572), -INT32_C(  1899745238),  INT32_C(    62568584),  INT32_C(  1741136306),  INT32_C(   471841389), -INT32_C(  1365554782),  INT32_C(  1585815147), -INT32_C(  1720141971) },
      { -INT32_C(   214102583), -INT32_C(   964558531),  INT32_C(  1825127610),  INT32_C(  1255379165), -INT32_C(   311954614), -INT32_C(  1483013572),  INT32_C(   218505376), -INT32_C(  1599635753) },
      { -INT32_C(  1336408989),  INT32_C(           0),  INT32_C(           0),  INT32_C(   485757141),  INT32_C(   783796003),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(139),
      { -INT32_C(   439839516),  INT32_C(  1386188310), -INT32_C(   416347049), -INT32_C(   768443937), -INT32_C(   502325280), -INT32_C(   612124998),  INT32_C(   984780988),  INT32_C(   465916727) },
      { -INT32_C(    50229529),  INT32_C(  1951375388), -INT32_C(  1940160852), -INT32_C(   664891913),  INT32_C(  1740271020), -INT32_C(   532529884),  INT32_C(    85718222),  INT32_C(   773972039) },
      { -INT32_C(   389609987), -INT32_C(   565187078),  INT32_C(           0), -INT32_C(   103552024),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   308055312) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sub_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_sub_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_sub_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_sub_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C(  443467777033216552), -INT64_C( 6412181382168176808),  INT64_C( 1246353111359392142),  INT64_C( 2607502804053799849) },
      UINT8_C( 87),
      {  INT64_C( 6444638156496968596), -INT64_C( 8332782995172032397), -INT64_C(   80453695814598839), -INT64_C( 2905350335524891487) },
      {  INT64_C(  348570000143850554),  INT64_C( 9001903127914492254), -INT64_C( 3136872368099227730), -INT64_C( 3763081222110350933) },
      {  INT64_C( 6096068156353118042),  INT64_C( 1112057950623026965),  INT64_C( 3056418672284628891),  INT64_C( 2607502804053799849) } },
    { {  INT64_C( 8145355943111603573), -INT64_C( 5043821522182808240), -INT64_C(  590347253547027001),  INT64_C( 5221833413314935068) },
      UINT8_C( 98),
      {  INT64_C( 3904311200635506367),  INT64_C( 7367603879564980664),  INT64_C(   69429794981427248), -INT64_C( 2693404252623887472) },
      { -INT64_C( 5550432576051080087),  INT64_C( 7471304893490936471), -INT64_C( 7158409980691783007),  INT64_C(   58399009455897563) },
      {  INT64_C( 8145355943111603573), -INT64_C(  103701013925955807), -INT64_C(  590347253547027001),  INT64_C( 5221833413314935068) } },
    { {  INT64_C(  377101072935847822),  INT64_C( 7967991223401997813), -INT64_C( 1726920392744446625), -INT64_C( 1598305782865206726) },
      UINT8_C( 52),
      {  INT64_C( 1316361353889477267), -INT64_C( 2787079436227350283), -INT64_C( 6386439875458581148), -INT64_C( 7814330292197183573) },
      { -INT64_C( 3810847008921342114), -INT64_C( 5804278677357331050), -INT64_C( 3226212137020728883), -INT64_C( 8578310377364046496) },
      {  INT64_C(  377101072935847822),  INT64_C( 7967991223401997813), -INT64_C( 3160227738437852265), -INT64_C( 1598305782865206726) } },
    { { -INT64_C( 7887859052563933721),  INT64_C( 2497422284490032814),  INT64_C(  933634545351490553), -INT64_C( 4621431160349909204) },
      UINT8_C( 53),
      {  INT64_C( 5463080774816333169), -INT64_C( 1330317322402805265), -INT64_C( 4786298779691537800),  INT64_C( 4010549778987739410) },
      { -INT64_C( 8420841124648068654),  INT64_C(   22980374520644850), -INT64_C( 6653510374603101618),  INT64_C( 7413696382811586532) },
      { -INT64_C( 4562822174245149793),  INT64_C( 2497422284490032814),  INT64_C( 1867211594911563818), -INT64_C( 4621431160349909204) } },
    { {  INT64_C( 6322596914625592214), -INT64_C( 3770625054107510492), -INT64_C(  996878751986412553), -INT64_C( 7739013759190089569) },
      UINT8_C(186),
      {  INT64_C( 6727383353128896215), -INT64_C( 4091901017835567054),  INT64_C( 2908420361317432668),  INT64_C(   44958110869459045) },
      {  INT64_C( 7521737283087154816),  INT64_C( 4474288718901361820),  INT64_C( 1993503325969706394),  INT64_C(  653760306966745320) },
      {  INT64_C( 6322596914625592214), -INT64_C( 8566189736736928874), -INT64_C(  996878751986412553), -INT64_C(  608802196097286275) } },
    { {  INT64_C( 8861800793033820765),  INT64_C( 1859390159146623660),  INT64_C(  621573854071670991), -INT64_C( 7086083886910103530) },
      UINT8_C(118),
      { -INT64_C( 3615037916132921161), -INT64_C(  556083299407338856),  INT64_C( 1035301721881496833),  INT64_C( 6907936511976472123) },
      { -INT64_C( 7560598197466877669), -INT64_C( 4114783982207344707),  INT64_C( 6459125169168252198), -INT64_C(  616256630446558869) },
      {  INT64_C( 8861800793033820765),  INT64_C( 3558700682800005851), -INT64_C( 5423823447286755365), -INT64_C( 7086083886910103530) } },
    { {  INT64_C( 1325272515504634318), -INT64_C(  848395479422110289),  INT64_C( 2258885839717746657), -INT64_C( 8723291368149870051) },
      UINT8_C(158),
      {  INT64_C( 3254981821115552557),  INT64_C( 2362683198711504643),  INT64_C( 4378047642900679032), -INT64_C( 2924953084009072091) },
      {  INT64_C( 7111887909318580941),  INT64_C( 1520267918427467040),  INT64_C(  521146392462946012),  INT64_C( 8554650893974318902) },
      {  INT64_C( 1325272515504634318),  INT64_C(  842415280284037603),  INT64_C( 3856901250437733020),  INT64_C( 6967140095726160623) } },
    { { -INT64_C( 5956663007437386605), -INT64_C( 8210414295236305060),  INT64_C( 8833466816413878445),  INT64_C( 3681150226499528694) },
      UINT8_C( 31),
      {  INT64_C( 1881477132382381586), -INT64_C( 1410653469407439361), -INT64_C(  530111313453450266),  INT64_C( 7797643392641942533) },
      {  INT64_C( 4217138620645827313), -INT64_C( 7822922236113104290),  INT64_C( 5758385708607840096),  INT64_C( 4535016301732987177) },
      { -INT64_C( 2335661488263445727),  INT64_C( 6412268766705664929), -INT64_C( 6288497022061290362),  INT64_C( 3262627090908955356) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sub_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_sub_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_mask_sub_epi64(src, k, a, b);

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
test_easysimd_mm256_maskz_sub_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(110),
      { -INT64_C(  544936127173547230), -INT64_C( 4065404275122963622),  INT64_C( 3643072614679444354), -INT64_C( 3011315281433095194) },
      {  INT64_C( 1893210311106347219), -INT64_C( 9141296799422466557), -INT64_C( 4611176004763709806), -INT64_C(  653658412482593950) },
      {  INT64_C(                   0),  INT64_C( 5075892524299502935),  INT64_C( 8254248619443154160), -INT64_C( 2357656868950501244) } },
    { UINT8_C(125),
      { -INT64_C( 4498718295867964282),  INT64_C( 2342224826502016601), -INT64_C( 3624348412377673342),  INT64_C( 4502728509217622594) },
      {  INT64_C( 7599927797067135275), -INT64_C(  324581908315293620),  INT64_C( 8380072123985328171),  INT64_C( 5674799565789936842) },
      {  INT64_C( 6348097980774452059),  INT64_C(                   0),  INT64_C( 6442323537346550103), -INT64_C( 1172071056572314248) } },
    { UINT8_C(169),
      { -INT64_C( 5388517930399300211), -INT64_C( 8322043994088499588), -INT64_C( 8629403432526086027),  INT64_C( 1027095050031186087) },
      { -INT64_C( 7336307586964456706), -INT64_C(  258265901710614839),  INT64_C(  308689391670339956), -INT64_C( 1441191540728007824) },
      {  INT64_C( 1947789656565156495),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 2468286590759193911) } },
    { UINT8_C(214),
      { -INT64_C(  651176316146259974),  INT64_C( 7523397592625828992),  INT64_C( 7280853237362928107), -INT64_C(  786372935604307100) },
      {  INT64_C( 1743483047868762788),  INT64_C( 1482938997004873709), -INT64_C( 8614056627014772066), -INT64_C( 6294380792453216587) },
      {  INT64_C(                   0),  INT64_C( 6040458595620955283), -INT64_C( 2551834209331851443),  INT64_C(                   0) } },
    { UINT8_C(232),
      { -INT64_C( 4121751235432749881), -INT64_C( 1749643451144603483),  INT64_C(  225622244553574579),  INT64_C( 4404713357849383258) },
      { -INT64_C( 2346118967688584812),  INT64_C( 1681727655314612450),  INT64_C( 5600971552763109498),  INT64_C(  300235329125138806) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 4104478028724244452) } },
    { UINT8_C( 71),
      {  INT64_C( 9156769253976813383),  INT64_C( 4073781668809507419), -INT64_C( 5000831800274370690), -INT64_C(  509698742796253441) },
      {  INT64_C( 3061670185685053150), -INT64_C( 2751769347705249602),  INT64_C( 8264832591509388031), -INT64_C( 1601868544216605668) },
      {  INT64_C( 6095099068291760233),  INT64_C( 6825551016514757021),  INT64_C( 5181079681925792895),  INT64_C(                   0) } },
    { UINT8_C(115),
      { -INT64_C( 4257470220735354897),  INT64_C( 5360682476360401061), -INT64_C( 1466431789647176460),  INT64_C( 2864535895438422856) },
      {  INT64_C( 1652607144535720903), -INT64_C(  263495732282257475), -INT64_C( 8343432587820788675), -INT64_C( 3098588634361025121) },
      { -INT64_C( 5910077365271075800),  INT64_C( 5624178208642658536),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(157),
      { -INT64_C(  194738977351205038), -INT64_C( 4197915945072240223),  INT64_C( 7877523902850206343),  INT64_C( 8838118727602547055) },
      {  INT64_C( 8879223635647771981), -INT64_C( 4168719999176982865),  INT64_C( 3779588555076611781), -INT64_C( 8459760953175707915) },
      { -INT64_C( 9073962612998977019),  INT64_C(                   0),  INT64_C( 4097935347773594562), -INT64_C( 1148864392931296646) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sub_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_sub_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_sub_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_sub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -777.58), EASYSIMD_FLOAT32_C(  -647.72), EASYSIMD_FLOAT32_C(  -439.30), EASYSIMD_FLOAT32_C(   381.55),
        EASYSIMD_FLOAT32_C(    50.90), EASYSIMD_FLOAT32_C(  -360.50), EASYSIMD_FLOAT32_C(   448.15), EASYSIMD_FLOAT32_C(   138.94) },
      UINT8_C(231),
      { EASYSIMD_FLOAT32_C(   621.74), EASYSIMD_FLOAT32_C(     3.99), EASYSIMD_FLOAT32_C(   311.19), EASYSIMD_FLOAT32_C(   855.96),
        EASYSIMD_FLOAT32_C(    16.64), EASYSIMD_FLOAT32_C(  -740.77), EASYSIMD_FLOAT32_C(  -799.67), EASYSIMD_FLOAT32_C(  -868.81) },
      { EASYSIMD_FLOAT32_C(  -711.56), EASYSIMD_FLOAT32_C(  -108.32), EASYSIMD_FLOAT32_C(  -882.41), EASYSIMD_FLOAT32_C(   378.98),
        EASYSIMD_FLOAT32_C(  -301.21), EASYSIMD_FLOAT32_C(   852.67), EASYSIMD_FLOAT32_C(   870.84), EASYSIMD_FLOAT32_C(   442.57) },
      { EASYSIMD_FLOAT32_C(  1333.30), EASYSIMD_FLOAT32_C(   112.31), EASYSIMD_FLOAT32_C(  1193.60), EASYSIMD_FLOAT32_C(   381.55),
        EASYSIMD_FLOAT32_C(    50.90), EASYSIMD_FLOAT32_C( -1593.44), EASYSIMD_FLOAT32_C( -1670.51), EASYSIMD_FLOAT32_C( -1311.38) } },
    { { EASYSIMD_FLOAT32_C(  -134.55), EASYSIMD_FLOAT32_C(   519.69), EASYSIMD_FLOAT32_C(  -209.54), EASYSIMD_FLOAT32_C(   -25.65),
        EASYSIMD_FLOAT32_C(  -134.79), EASYSIMD_FLOAT32_C(  -943.40), EASYSIMD_FLOAT32_C(   196.77), EASYSIMD_FLOAT32_C(   217.49) },
      UINT8_C(106),
      { EASYSIMD_FLOAT32_C(  -421.68), EASYSIMD_FLOAT32_C(  -731.61), EASYSIMD_FLOAT32_C(   256.80), EASYSIMD_FLOAT32_C(  -973.53),
        EASYSIMD_FLOAT32_C(   407.33), EASYSIMD_FLOAT32_C(  -302.14), EASYSIMD_FLOAT32_C(   648.21), EASYSIMD_FLOAT32_C(  -588.68) },
      { EASYSIMD_FLOAT32_C(  -990.95), EASYSIMD_FLOAT32_C(   504.17), EASYSIMD_FLOAT32_C(   427.96), EASYSIMD_FLOAT32_C(  -731.72),
        EASYSIMD_FLOAT32_C(   704.50), EASYSIMD_FLOAT32_C(   559.15), EASYSIMD_FLOAT32_C(  -443.28), EASYSIMD_FLOAT32_C(  -403.83) },
      { EASYSIMD_FLOAT32_C(  -134.55), EASYSIMD_FLOAT32_C( -1235.78), EASYSIMD_FLOAT32_C(  -209.54), EASYSIMD_FLOAT32_C(  -241.81),
        EASYSIMD_FLOAT32_C(  -134.79), EASYSIMD_FLOAT32_C(  -861.29), EASYSIMD_FLOAT32_C(  1091.49), EASYSIMD_FLOAT32_C(   217.49) } },
    { { EASYSIMD_FLOAT32_C(   676.74), EASYSIMD_FLOAT32_C(   935.70), EASYSIMD_FLOAT32_C(   294.97), EASYSIMD_FLOAT32_C(   529.40),
        EASYSIMD_FLOAT32_C(   806.54), EASYSIMD_FLOAT32_C(  -262.46), EASYSIMD_FLOAT32_C(  -605.15), EASYSIMD_FLOAT32_C(   326.24) },
      UINT8_C(180),
      { EASYSIMD_FLOAT32_C(   369.20), EASYSIMD_FLOAT32_C(  -808.55), EASYSIMD_FLOAT32_C(   584.59), EASYSIMD_FLOAT32_C(  -434.03),
        EASYSIMD_FLOAT32_C(   408.95), EASYSIMD_FLOAT32_C(  -798.12), EASYSIMD_FLOAT32_C(   144.30), EASYSIMD_FLOAT32_C(   677.34) },
      { EASYSIMD_FLOAT32_C(   458.68), EASYSIMD_FLOAT32_C(   170.77), EASYSIMD_FLOAT32_C(    84.67), EASYSIMD_FLOAT32_C(  -843.46),
        EASYSIMD_FLOAT32_C(  -181.02), EASYSIMD_FLOAT32_C(   496.00), EASYSIMD_FLOAT32_C(  -834.41), EASYSIMD_FLOAT32_C(  -676.85) },
      { EASYSIMD_FLOAT32_C(   676.74), EASYSIMD_FLOAT32_C(   935.70), EASYSIMD_FLOAT32_C(   499.92), EASYSIMD_FLOAT32_C(   529.40),
        EASYSIMD_FLOAT32_C(   589.97), EASYSIMD_FLOAT32_C( -1294.12), EASYSIMD_FLOAT32_C(  -605.15), EASYSIMD_FLOAT32_C(  1354.19) } },
    { { EASYSIMD_FLOAT32_C(   -76.04), EASYSIMD_FLOAT32_C(  -566.13), EASYSIMD_FLOAT32_C(  -972.36), EASYSIMD_FLOAT32_C(  -516.89),
        EASYSIMD_FLOAT32_C(    -9.40), EASYSIMD_FLOAT32_C(  -376.18), EASYSIMD_FLOAT32_C(  -840.15), EASYSIMD_FLOAT32_C(   -73.70) },
      UINT8_C(223),
      { EASYSIMD_FLOAT32_C(   689.25), EASYSIMD_FLOAT32_C(  -267.16), EASYSIMD_FLOAT32_C(  -343.68), EASYSIMD_FLOAT32_C(  -915.90),
        EASYSIMD_FLOAT32_C(  -940.92), EASYSIMD_FLOAT32_C(  -815.69), EASYSIMD_FLOAT32_C(   453.31), EASYSIMD_FLOAT32_C(  -749.47) },
      { EASYSIMD_FLOAT32_C(   768.90), EASYSIMD_FLOAT32_C(  -980.72), EASYSIMD_FLOAT32_C(   659.48), EASYSIMD_FLOAT32_C(   970.78),
        EASYSIMD_FLOAT32_C(   163.57), EASYSIMD_FLOAT32_C(   336.82), EASYSIMD_FLOAT32_C(   429.46), EASYSIMD_FLOAT32_C(  -665.66) },
      { EASYSIMD_FLOAT32_C(   -79.65), EASYSIMD_FLOAT32_C(   713.56), EASYSIMD_FLOAT32_C( -1003.16), EASYSIMD_FLOAT32_C( -1886.68),
        EASYSIMD_FLOAT32_C( -1104.49), EASYSIMD_FLOAT32_C(  -376.18), EASYSIMD_FLOAT32_C(    23.85), EASYSIMD_FLOAT32_C(   -83.81) } },
    { { EASYSIMD_FLOAT32_C(  -578.51), EASYSIMD_FLOAT32_C(   586.00), EASYSIMD_FLOAT32_C(   153.32), EASYSIMD_FLOAT32_C(   917.49),
        EASYSIMD_FLOAT32_C(   751.60), EASYSIMD_FLOAT32_C(   476.47), EASYSIMD_FLOAT32_C(  -158.56), EASYSIMD_FLOAT32_C(  -814.53) },
      UINT8_C(  9),
      { EASYSIMD_FLOAT32_C(   324.55), EASYSIMD_FLOAT32_C(   176.07), EASYSIMD_FLOAT32_C(  -872.07), EASYSIMD_FLOAT32_C(   484.40),
        EASYSIMD_FLOAT32_C(  -897.63), EASYSIMD_FLOAT32_C(  -953.29), EASYSIMD_FLOAT32_C(   173.65), EASYSIMD_FLOAT32_C(  -164.79) },
      { EASYSIMD_FLOAT32_C(  -296.97), EASYSIMD_FLOAT32_C(   257.75), EASYSIMD_FLOAT32_C(  -105.71), EASYSIMD_FLOAT32_C(  -112.66),
        EASYSIMD_FLOAT32_C(  -288.94), EASYSIMD_FLOAT32_C(   144.82), EASYSIMD_FLOAT32_C(  -343.76), EASYSIMD_FLOAT32_C(  -269.66) },
      { EASYSIMD_FLOAT32_C(   621.52), EASYSIMD_FLOAT32_C(   586.00), EASYSIMD_FLOAT32_C(   153.32), EASYSIMD_FLOAT32_C(   597.06),
        EASYSIMD_FLOAT32_C(   751.60), EASYSIMD_FLOAT32_C(   476.47), EASYSIMD_FLOAT32_C(  -158.56), EASYSIMD_FLOAT32_C(  -814.53) } },
    { { EASYSIMD_FLOAT32_C(  -195.70), EASYSIMD_FLOAT32_C(  -372.98), EASYSIMD_FLOAT32_C(   893.91), EASYSIMD_FLOAT32_C(  -858.88),
        EASYSIMD_FLOAT32_C(  -943.51), EASYSIMD_FLOAT32_C(  -771.74), EASYSIMD_FLOAT32_C(  -437.39), EASYSIMD_FLOAT32_C(   642.49) },
      UINT8_C( 28),
      { EASYSIMD_FLOAT32_C(  -519.90), EASYSIMD_FLOAT32_C(   394.09), EASYSIMD_FLOAT32_C(  -141.95), EASYSIMD_FLOAT32_C(   321.54),
        EASYSIMD_FLOAT32_C(   579.56), EASYSIMD_FLOAT32_C(  -637.84), EASYSIMD_FLOAT32_C(  -353.91), EASYSIMD_FLOAT32_C(  -244.37) },
      { EASYSIMD_FLOAT32_C(  -509.91), EASYSIMD_FLOAT32_C(  -869.51), EASYSIMD_FLOAT32_C(  -142.01), EASYSIMD_FLOAT32_C(  -463.20),
        EASYSIMD_FLOAT32_C(   304.14), EASYSIMD_FLOAT32_C(   693.20), EASYSIMD_FLOAT32_C(   239.83), EASYSIMD_FLOAT32_C(  -438.11) },
      { EASYSIMD_FLOAT32_C(  -195.70), EASYSIMD_FLOAT32_C(  -372.98), EASYSIMD_FLOAT32_C(     0.06), EASYSIMD_FLOAT32_C(   784.74),
        EASYSIMD_FLOAT32_C(   275.42), EASYSIMD_FLOAT32_C(  -771.74), EASYSIMD_FLOAT32_C(  -437.39), EASYSIMD_FLOAT32_C(   642.49) } },
    { { EASYSIMD_FLOAT32_C(  -412.51), EASYSIMD_FLOAT32_C(  -872.82), EASYSIMD_FLOAT32_C(   272.95), EASYSIMD_FLOAT32_C(   732.32),
        EASYSIMD_FLOAT32_C(  -216.58), EASYSIMD_FLOAT32_C(  -996.72), EASYSIMD_FLOAT32_C(  -463.38), EASYSIMD_FLOAT32_C(   410.44) },
      UINT8_C(121),
      { EASYSIMD_FLOAT32_C(  -322.26), EASYSIMD_FLOAT32_C(   466.93), EASYSIMD_FLOAT32_C(  -874.55), EASYSIMD_FLOAT32_C(   240.35),
        EASYSIMD_FLOAT32_C(   109.42), EASYSIMD_FLOAT32_C(   507.03), EASYSIMD_FLOAT32_C(   720.45), EASYSIMD_FLOAT32_C(  -496.49) },
      { EASYSIMD_FLOAT32_C(  -634.92), EASYSIMD_FLOAT32_C(    41.99), EASYSIMD_FLOAT32_C(  -916.94), EASYSIMD_FLOAT32_C(  -272.76),
        EASYSIMD_FLOAT32_C(   688.08), EASYSIMD_FLOAT32_C(  -161.31), EASYSIMD_FLOAT32_C(   217.33), EASYSIMD_FLOAT32_C(   818.57) },
      { EASYSIMD_FLOAT32_C(   312.66), EASYSIMD_FLOAT32_C(  -872.82), EASYSIMD_FLOAT32_C(   272.95), EASYSIMD_FLOAT32_C(   513.11),
        EASYSIMD_FLOAT32_C(  -578.66), EASYSIMD_FLOAT32_C(   668.34), EASYSIMD_FLOAT32_C(   503.12), EASYSIMD_FLOAT32_C(   410.44) } },
    { { EASYSIMD_FLOAT32_C(   696.68), EASYSIMD_FLOAT32_C(   754.13), EASYSIMD_FLOAT32_C(   122.71), EASYSIMD_FLOAT32_C(   389.88),
        EASYSIMD_FLOAT32_C(    -6.04), EASYSIMD_FLOAT32_C(   684.60), EASYSIMD_FLOAT32_C(   977.38), EASYSIMD_FLOAT32_C(   121.14) },
      UINT8_C(103),
      { EASYSIMD_FLOAT32_C(   709.70), EASYSIMD_FLOAT32_C(   904.56), EASYSIMD_FLOAT32_C(   -39.17), EASYSIMD_FLOAT32_C(  -753.68),
        EASYSIMD_FLOAT32_C(   315.00), EASYSIMD_FLOAT32_C(  -141.97), EASYSIMD_FLOAT32_C(   -75.94), EASYSIMD_FLOAT32_C(  -218.07) },
      { EASYSIMD_FLOAT32_C(   -16.52), EASYSIMD_FLOAT32_C(  -835.59), EASYSIMD_FLOAT32_C(   891.35), EASYSIMD_FLOAT32_C(  -509.48),
        EASYSIMD_FLOAT32_C(   884.86), EASYSIMD_FLOAT32_C(  -605.15), EASYSIMD_FLOAT32_C(  -144.40), EASYSIMD_FLOAT32_C(   -73.15) },
      { EASYSIMD_FLOAT32_C(   726.22), EASYSIMD_FLOAT32_C(  1740.15), EASYSIMD_FLOAT32_C(  -930.52), EASYSIMD_FLOAT32_C(   389.88),
        EASYSIMD_FLOAT32_C(    -6.04), EASYSIMD_FLOAT32_C(   463.18), EASYSIMD_FLOAT32_C(    68.46), EASYSIMD_FLOAT32_C(   121.14) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sub_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_sub_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_sub_ps(src, k, a, b);

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
test_easysimd_mm256_maskz_sub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(200),
      { EASYSIMD_FLOAT32_C(  -527.45), EASYSIMD_FLOAT32_C(   743.44), EASYSIMD_FLOAT32_C(  -243.28), EASYSIMD_FLOAT32_C(  -807.28),
        EASYSIMD_FLOAT32_C(  -219.43), EASYSIMD_FLOAT32_C(  -133.18), EASYSIMD_FLOAT32_C(   200.70), EASYSIMD_FLOAT32_C(   497.61) },
      { EASYSIMD_FLOAT32_C(   827.49), EASYSIMD_FLOAT32_C(   506.07), EASYSIMD_FLOAT32_C(   721.67), EASYSIMD_FLOAT32_C(  -176.35),
        EASYSIMD_FLOAT32_C(  -931.87), EASYSIMD_FLOAT32_C(   593.55), EASYSIMD_FLOAT32_C(  -816.67), EASYSIMD_FLOAT32_C(   172.20) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -630.93),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  1017.37), EASYSIMD_FLOAT32_C(   325.41) } },
    { UINT8_C( 39),
      { EASYSIMD_FLOAT32_C(   726.78), EASYSIMD_FLOAT32_C(   972.91), EASYSIMD_FLOAT32_C(  -570.51), EASYSIMD_FLOAT32_C(  -772.51),
        EASYSIMD_FLOAT32_C(   720.97), EASYSIMD_FLOAT32_C(   138.29), EASYSIMD_FLOAT32_C(  -621.96), EASYSIMD_FLOAT32_C(  -447.30) },
      { EASYSIMD_FLOAT32_C(   -24.69), EASYSIMD_FLOAT32_C(   624.32), EASYSIMD_FLOAT32_C(   570.16), EASYSIMD_FLOAT32_C(  -695.29),
        EASYSIMD_FLOAT32_C(  -335.61), EASYSIMD_FLOAT32_C(  -556.59), EASYSIMD_FLOAT32_C(  -222.74), EASYSIMD_FLOAT32_C(  -592.16) },
      { EASYSIMD_FLOAT32_C(   751.47), EASYSIMD_FLOAT32_C(   348.59), EASYSIMD_FLOAT32_C( -1140.67), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   694.88), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(156),
      { EASYSIMD_FLOAT32_C(   -30.02), EASYSIMD_FLOAT32_C(   188.41), EASYSIMD_FLOAT32_C(  -933.05), EASYSIMD_FLOAT32_C(  -829.33),
        EASYSIMD_FLOAT32_C(  -313.98), EASYSIMD_FLOAT32_C(   894.44), EASYSIMD_FLOAT32_C(   676.74), EASYSIMD_FLOAT32_C(  -592.31) },
      { EASYSIMD_FLOAT32_C(  -281.91), EASYSIMD_FLOAT32_C(   744.87), EASYSIMD_FLOAT32_C(  -998.75), EASYSIMD_FLOAT32_C(   -98.58),
        EASYSIMD_FLOAT32_C(   -82.93), EASYSIMD_FLOAT32_C(   884.57), EASYSIMD_FLOAT32_C(  -371.80), EASYSIMD_FLOAT32_C(  -110.02) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    65.70), EASYSIMD_FLOAT32_C(  -730.75),
        EASYSIMD_FLOAT32_C(  -231.05), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -482.29) } },
    { UINT8_C(163),
      { EASYSIMD_FLOAT32_C(  -144.31), EASYSIMD_FLOAT32_C(  -389.05), EASYSIMD_FLOAT32_C(   452.35), EASYSIMD_FLOAT32_C(   233.74),
        EASYSIMD_FLOAT32_C(   163.65), EASYSIMD_FLOAT32_C(  -572.34), EASYSIMD_FLOAT32_C(  -141.94), EASYSIMD_FLOAT32_C(  -266.20) },
      { EASYSIMD_FLOAT32_C(  -267.63), EASYSIMD_FLOAT32_C(   522.46), EASYSIMD_FLOAT32_C(   177.22), EASYSIMD_FLOAT32_C(   509.64),
        EASYSIMD_FLOAT32_C(   930.29), EASYSIMD_FLOAT32_C(  -622.66), EASYSIMD_FLOAT32_C(  -520.39), EASYSIMD_FLOAT32_C(   118.70) },
      { EASYSIMD_FLOAT32_C(   123.32), EASYSIMD_FLOAT32_C(  -911.51), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    50.32), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -384.90) } },
    { UINT8_C( 10),
      { EASYSIMD_FLOAT32_C(  -349.72), EASYSIMD_FLOAT32_C(   804.72), EASYSIMD_FLOAT32_C(  -661.26), EASYSIMD_FLOAT32_C(  -672.97),
        EASYSIMD_FLOAT32_C(  -787.58), EASYSIMD_FLOAT32_C(    56.83), EASYSIMD_FLOAT32_C(  -928.10), EASYSIMD_FLOAT32_C(  -786.34) },
      { EASYSIMD_FLOAT32_C(   958.25), EASYSIMD_FLOAT32_C(   -11.02), EASYSIMD_FLOAT32_C(  -901.77), EASYSIMD_FLOAT32_C(  -413.55),
        EASYSIMD_FLOAT32_C(   878.96), EASYSIMD_FLOAT32_C(  -587.70), EASYSIMD_FLOAT32_C(   442.15), EASYSIMD_FLOAT32_C(  -510.09) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   815.74), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -259.42),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 39),
      { EASYSIMD_FLOAT32_C(  -324.11), EASYSIMD_FLOAT32_C(   653.56), EASYSIMD_FLOAT32_C(  -707.69), EASYSIMD_FLOAT32_C(   533.95),
        EASYSIMD_FLOAT32_C(  -612.64), EASYSIMD_FLOAT32_C(    24.68), EASYSIMD_FLOAT32_C(    56.40), EASYSIMD_FLOAT32_C(   564.57) },
      { EASYSIMD_FLOAT32_C(  -465.68), EASYSIMD_FLOAT32_C(   -13.30), EASYSIMD_FLOAT32_C(   941.92), EASYSIMD_FLOAT32_C(    13.93),
        EASYSIMD_FLOAT32_C(  -894.60), EASYSIMD_FLOAT32_C(  -613.79), EASYSIMD_FLOAT32_C(   664.22), EASYSIMD_FLOAT32_C(   910.12) },
      { EASYSIMD_FLOAT32_C(   141.57), EASYSIMD_FLOAT32_C(   666.86), EASYSIMD_FLOAT32_C( -1649.61), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   638.47), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(136),
      { EASYSIMD_FLOAT32_C(   991.25), EASYSIMD_FLOAT32_C(  -877.46), EASYSIMD_FLOAT32_C(   781.78), EASYSIMD_FLOAT32_C(  -936.85),
        EASYSIMD_FLOAT32_C(  -663.80), EASYSIMD_FLOAT32_C(   740.03), EASYSIMD_FLOAT32_C(    52.12), EASYSIMD_FLOAT32_C(  -565.56) },
      { EASYSIMD_FLOAT32_C(  -673.51), EASYSIMD_FLOAT32_C(   -68.92), EASYSIMD_FLOAT32_C(  -153.27), EASYSIMD_FLOAT32_C(   768.64),
        EASYSIMD_FLOAT32_C(   420.99), EASYSIMD_FLOAT32_C(  -288.62), EASYSIMD_FLOAT32_C(  -555.48), EASYSIMD_FLOAT32_C(    74.55) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( -1705.49),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -640.11) } },
    { UINT8_C( 55),
      { EASYSIMD_FLOAT32_C(   978.47), EASYSIMD_FLOAT32_C(   461.90), EASYSIMD_FLOAT32_C(  -971.63), EASYSIMD_FLOAT32_C(    34.88),
        EASYSIMD_FLOAT32_C(    26.48), EASYSIMD_FLOAT32_C(  -437.31), EASYSIMD_FLOAT32_C(  -978.42), EASYSIMD_FLOAT32_C(   -31.60) },
      { EASYSIMD_FLOAT32_C(   576.62), EASYSIMD_FLOAT32_C(  -873.02), EASYSIMD_FLOAT32_C(   354.61), EASYSIMD_FLOAT32_C(   240.84),
        EASYSIMD_FLOAT32_C(  -962.90), EASYSIMD_FLOAT32_C(  -920.44), EASYSIMD_FLOAT32_C(   232.08), EASYSIMD_FLOAT32_C(  -840.36) },
      { EASYSIMD_FLOAT32_C(   401.85), EASYSIMD_FLOAT32_C(  1334.92), EASYSIMD_FLOAT32_C( -1326.24), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   989.38), EASYSIMD_FLOAT32_C(   483.13), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sub_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_sub_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_sub_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_sub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   559.93), EASYSIMD_FLOAT64_C(  -944.26), EASYSIMD_FLOAT64_C(   492.31), EASYSIMD_FLOAT64_C(   947.45) },
      UINT8_C(225),
      { EASYSIMD_FLOAT64_C(   202.24), EASYSIMD_FLOAT64_C(  -745.46), EASYSIMD_FLOAT64_C(   257.34), EASYSIMD_FLOAT64_C(   860.11) },
      { EASYSIMD_FLOAT64_C(  -819.90), EASYSIMD_FLOAT64_C(  -184.51), EASYSIMD_FLOAT64_C(   662.29), EASYSIMD_FLOAT64_C(  -913.93) },
      { EASYSIMD_FLOAT64_C(  1022.14), EASYSIMD_FLOAT64_C(  -944.26), EASYSIMD_FLOAT64_C(   492.31), EASYSIMD_FLOAT64_C(   947.45) } },
    { { EASYSIMD_FLOAT64_C(   386.88), EASYSIMD_FLOAT64_C(   -32.77), EASYSIMD_FLOAT64_C(   702.05), EASYSIMD_FLOAT64_C(    81.45) },
      UINT8_C(187),
      { EASYSIMD_FLOAT64_C(    62.44), EASYSIMD_FLOAT64_C(   372.70), EASYSIMD_FLOAT64_C(  -295.95), EASYSIMD_FLOAT64_C(   759.28) },
      { EASYSIMD_FLOAT64_C(   667.56), EASYSIMD_FLOAT64_C(   714.64), EASYSIMD_FLOAT64_C(  -611.99), EASYSIMD_FLOAT64_C(   813.11) },
      { EASYSIMD_FLOAT64_C(  -605.12), EASYSIMD_FLOAT64_C(  -341.94), EASYSIMD_FLOAT64_C(   702.05), EASYSIMD_FLOAT64_C(   -53.83) } },
    { { EASYSIMD_FLOAT64_C(  -674.88), EASYSIMD_FLOAT64_C(  -129.70), EASYSIMD_FLOAT64_C(   580.95), EASYSIMD_FLOAT64_C(   -77.97) },
      UINT8_C(155),
      { EASYSIMD_FLOAT64_C(   140.88), EASYSIMD_FLOAT64_C(   -22.23), EASYSIMD_FLOAT64_C(   322.84), EASYSIMD_FLOAT64_C(    88.33) },
      { EASYSIMD_FLOAT64_C(  -975.29), EASYSIMD_FLOAT64_C(  -474.93), EASYSIMD_FLOAT64_C(   342.87), EASYSIMD_FLOAT64_C(   282.05) },
      { EASYSIMD_FLOAT64_C(  1116.17), EASYSIMD_FLOAT64_C(   452.70), EASYSIMD_FLOAT64_C(   580.95), EASYSIMD_FLOAT64_C(  -193.72) } },
    { { EASYSIMD_FLOAT64_C(  -614.82), EASYSIMD_FLOAT64_C(   522.96), EASYSIMD_FLOAT64_C(  -902.46), EASYSIMD_FLOAT64_C(  -952.53) },
      UINT8_C(144),
      { EASYSIMD_FLOAT64_C(   484.42), EASYSIMD_FLOAT64_C(    14.70), EASYSIMD_FLOAT64_C(   311.08), EASYSIMD_FLOAT64_C(  -434.14) },
      { EASYSIMD_FLOAT64_C(   523.05), EASYSIMD_FLOAT64_C(  -626.47), EASYSIMD_FLOAT64_C(   938.57), EASYSIMD_FLOAT64_C(  -772.90) },
      { EASYSIMD_FLOAT64_C(  -614.82), EASYSIMD_FLOAT64_C(   522.96), EASYSIMD_FLOAT64_C(  -902.46), EASYSIMD_FLOAT64_C(  -952.53) } },
    { { EASYSIMD_FLOAT64_C(  -867.20), EASYSIMD_FLOAT64_C(   606.12), EASYSIMD_FLOAT64_C(   941.74), EASYSIMD_FLOAT64_C(  -479.19) },
      UINT8_C(245),
      { EASYSIMD_FLOAT64_C(  -733.14), EASYSIMD_FLOAT64_C(   391.11), EASYSIMD_FLOAT64_C(     0.19), EASYSIMD_FLOAT64_C(   188.89) },
      { EASYSIMD_FLOAT64_C(   221.64), EASYSIMD_FLOAT64_C(  -858.93), EASYSIMD_FLOAT64_C(  -833.34), EASYSIMD_FLOAT64_C(  -455.53) },
      { EASYSIMD_FLOAT64_C(  -954.78), EASYSIMD_FLOAT64_C(   606.12), EASYSIMD_FLOAT64_C(   833.53), EASYSIMD_FLOAT64_C(  -479.19) } },
    { { EASYSIMD_FLOAT64_C(   229.39), EASYSIMD_FLOAT64_C(  -808.64), EASYSIMD_FLOAT64_C(    69.55), EASYSIMD_FLOAT64_C(  -427.74) },
      UINT8_C(153),
      { EASYSIMD_FLOAT64_C(   454.73), EASYSIMD_FLOAT64_C(  -904.78), EASYSIMD_FLOAT64_C(   570.95), EASYSIMD_FLOAT64_C(   502.20) },
      { EASYSIMD_FLOAT64_C(   704.25), EASYSIMD_FLOAT64_C(    55.37), EASYSIMD_FLOAT64_C(  -483.10), EASYSIMD_FLOAT64_C(    15.33) },
      { EASYSIMD_FLOAT64_C(  -249.52), EASYSIMD_FLOAT64_C(  -808.64), EASYSIMD_FLOAT64_C(    69.55), EASYSIMD_FLOAT64_C(   486.87) } },
    { { EASYSIMD_FLOAT64_C(   621.23), EASYSIMD_FLOAT64_C(  -960.04), EASYSIMD_FLOAT64_C(   388.86), EASYSIMD_FLOAT64_C(   559.80) },
      UINT8_C( 37),
      { EASYSIMD_FLOAT64_C(   521.66), EASYSIMD_FLOAT64_C(   165.92), EASYSIMD_FLOAT64_C(  -791.21), EASYSIMD_FLOAT64_C(  -957.52) },
      { EASYSIMD_FLOAT64_C(  -414.84), EASYSIMD_FLOAT64_C(  -524.35), EASYSIMD_FLOAT64_C(   433.59), EASYSIMD_FLOAT64_C(   585.34) },
      { EASYSIMD_FLOAT64_C(   936.50), EASYSIMD_FLOAT64_C(  -960.04), EASYSIMD_FLOAT64_C( -1224.80), EASYSIMD_FLOAT64_C(   559.80) } },
    { { EASYSIMD_FLOAT64_C(   664.53), EASYSIMD_FLOAT64_C(  -344.77), EASYSIMD_FLOAT64_C(   726.41), EASYSIMD_FLOAT64_C(   831.19) },
      UINT8_C(205),
      { EASYSIMD_FLOAT64_C(   -44.19), EASYSIMD_FLOAT64_C(  -977.45), EASYSIMD_FLOAT64_C(  -730.75), EASYSIMD_FLOAT64_C(   528.07) },
      { EASYSIMD_FLOAT64_C(   495.97), EASYSIMD_FLOAT64_C(   723.98), EASYSIMD_FLOAT64_C(   623.29), EASYSIMD_FLOAT64_C(    66.92) },
      { EASYSIMD_FLOAT64_C(  -540.16), EASYSIMD_FLOAT64_C(  -344.77), EASYSIMD_FLOAT64_C( -1354.04), EASYSIMD_FLOAT64_C(   461.15) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_sub_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_sub_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_sub_pd(src, k, a, b);

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
test_easysimd_mm256_maskz_sub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(189),
      { EASYSIMD_FLOAT64_C(   -44.88), EASYSIMD_FLOAT64_C(  -405.94), EASYSIMD_FLOAT64_C(   386.97), EASYSIMD_FLOAT64_C(   634.48) },
      { EASYSIMD_FLOAT64_C(   890.02), EASYSIMD_FLOAT64_C(  -596.90), EASYSIMD_FLOAT64_C(  -407.95), EASYSIMD_FLOAT64_C(  -484.93) },
      { EASYSIMD_FLOAT64_C(  -934.90), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   794.92), EASYSIMD_FLOAT64_C(  1119.41) } },
    { UINT8_C( 33),
      { EASYSIMD_FLOAT64_C(  -740.95), EASYSIMD_FLOAT64_C(   546.86), EASYSIMD_FLOAT64_C(  -408.69), EASYSIMD_FLOAT64_C(   150.85) },
      { EASYSIMD_FLOAT64_C(   834.49), EASYSIMD_FLOAT64_C(   790.04), EASYSIMD_FLOAT64_C(   659.76), EASYSIMD_FLOAT64_C(   803.73) },
      { EASYSIMD_FLOAT64_C( -1575.44), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 51),
      { EASYSIMD_FLOAT64_C(    43.59), EASYSIMD_FLOAT64_C(   180.49), EASYSIMD_FLOAT64_C(  -228.64), EASYSIMD_FLOAT64_C(  -873.08) },
      { EASYSIMD_FLOAT64_C(  -516.64), EASYSIMD_FLOAT64_C(   501.28), EASYSIMD_FLOAT64_C(  -415.18), EASYSIMD_FLOAT64_C(  -223.04) },
      { EASYSIMD_FLOAT64_C(   560.23), EASYSIMD_FLOAT64_C(  -320.79), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 28),
      { EASYSIMD_FLOAT64_C(  -111.97), EASYSIMD_FLOAT64_C(  -840.66), EASYSIMD_FLOAT64_C(   191.54), EASYSIMD_FLOAT64_C(   507.38) },
      { EASYSIMD_FLOAT64_C(   114.46), EASYSIMD_FLOAT64_C(   785.61), EASYSIMD_FLOAT64_C(  -105.65), EASYSIMD_FLOAT64_C(  -251.06) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   297.19), EASYSIMD_FLOAT64_C(   758.44) } },
    { UINT8_C(101),
      { EASYSIMD_FLOAT64_C(   297.45), EASYSIMD_FLOAT64_C(   341.00), EASYSIMD_FLOAT64_C(  -809.30), EASYSIMD_FLOAT64_C(   114.44) },
      { EASYSIMD_FLOAT64_C(   600.05), EASYSIMD_FLOAT64_C(   737.56), EASYSIMD_FLOAT64_C(   705.75), EASYSIMD_FLOAT64_C(  -249.10) },
      { EASYSIMD_FLOAT64_C(  -302.60), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C( -1515.05), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 22),
      { EASYSIMD_FLOAT64_C(   495.79), EASYSIMD_FLOAT64_C(  -589.34), EASYSIMD_FLOAT64_C(   375.78), EASYSIMD_FLOAT64_C(  -965.23) },
      { EASYSIMD_FLOAT64_C(   454.24), EASYSIMD_FLOAT64_C(  -443.73), EASYSIMD_FLOAT64_C(  -193.87), EASYSIMD_FLOAT64_C(   581.16) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -145.61), EASYSIMD_FLOAT64_C(   569.65), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 72),
      { EASYSIMD_FLOAT64_C(  -692.59), EASYSIMD_FLOAT64_C(  -834.01), EASYSIMD_FLOAT64_C(   816.60), EASYSIMD_FLOAT64_C(  -572.66) },
      { EASYSIMD_FLOAT64_C(    54.02), EASYSIMD_FLOAT64_C(   975.94), EASYSIMD_FLOAT64_C(   618.88), EASYSIMD_FLOAT64_C(  -438.60) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -134.06) } },
    { UINT8_C(136),
      { EASYSIMD_FLOAT64_C(   404.49), EASYSIMD_FLOAT64_C(   455.74), EASYSIMD_FLOAT64_C(   839.35), EASYSIMD_FLOAT64_C(    80.11) },
      { EASYSIMD_FLOAT64_C(  -246.80), EASYSIMD_FLOAT64_C(   180.34), EASYSIMD_FLOAT64_C(   270.81), EASYSIMD_FLOAT64_C(   867.64) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -787.53) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_sub_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_sub_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_sub_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_sub_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C(  82), INT8_C(  83), INT8_C( 117), INT8_C(  65),
                           INT8_C( -47), INT8_C(-122), INT8_C( 116), INT8_C(  14),
                           INT8_C(  76), INT8_C(   1), INT8_C( -50), INT8_C(   4),
                           INT8_C(  83), INT8_C( -77), INT8_C( 112), INT8_C(-102),
                           INT8_C(  99), INT8_C(-118), INT8_C( -47), INT8_C( -67),
                           INT8_C(  60), INT8_C( -34), INT8_C(  78), INT8_C(-110),
                           INT8_C( -58), INT8_C(  87), INT8_C( -61), INT8_C(  26),
                           INT8_C( -17), INT8_C( -46), INT8_C( 116), INT8_C( -20),
                           INT8_C(-120), INT8_C(  48), INT8_C(  24), INT8_C(  46),
                           INT8_C( 103), INT8_C( -34), INT8_C(  42), INT8_C(  51),
                           INT8_C( -65), INT8_C(  86), INT8_C(  61), INT8_C( -56),
                           INT8_C(  58), INT8_C( 119), INT8_C(  93), INT8_C(  -1),
                           INT8_C( -58), INT8_C(-121), INT8_C( -32), INT8_C(  98),
                           INT8_C( -66), INT8_C(  79), INT8_C(  99), INT8_C( -93),
                           INT8_C(  77), INT8_C( -56), INT8_C( -78), INT8_C(  39),
                           INT8_C( -50), INT8_C( -60), INT8_C( -68), INT8_C(  -4)),
      easysimd_mm512_set_epi8(INT8_C( 106), INT8_C(  39), INT8_C(-106), INT8_C(  14),
                           INT8_C( -28), INT8_C(  -2), INT8_C(  39), INT8_C( -28),
                           INT8_C(  54), INT8_C(  70), INT8_C( -19), INT8_C( -12),
                           INT8_C( -42), INT8_C(  28), INT8_C( -13), INT8_C(  -6),
                           INT8_C( 116), INT8_C(   2), INT8_C(  23), INT8_C( 121),
                           INT8_C( 112), INT8_C( -35), INT8_C(-124), INT8_C(  10),
                           INT8_C( -16), INT8_C(-117), INT8_C(  26), INT8_C(-125),
                           INT8_C(  36), INT8_C( 109), INT8_C(  29), INT8_C( -35),
                           INT8_C(  -9), INT8_C( -85), INT8_C( -38), INT8_C(  95),
                           INT8_C( -88), INT8_C(   3), INT8_C(   4), INT8_C( 100),
                           INT8_C(  85), INT8_C(  21), INT8_C(  66), INT8_C( -33),
                           INT8_C( -77), INT8_C(  -5), INT8_C(-126), INT8_C( 122),
                           INT8_C( -30), INT8_C( -83), INT8_C(  74), INT8_C(-108),
                           INT8_C(  83), INT8_C( -96), INT8_C( -57), INT8_C(-108),
                           INT8_C(  89), INT8_C(  59), INT8_C( 111), INT8_C( -47),
                           INT8_C( -10), INT8_C( -18), INT8_C(  20), INT8_C(-125)),
      easysimd_mm512_set_epi8(INT8_C( -24), INT8_C(  44), INT8_C( -33), INT8_C(  51),
                           INT8_C( -19), INT8_C(-120), INT8_C(  77), INT8_C(  42),
                           INT8_C(  22), INT8_C( -69), INT8_C( -31), INT8_C(  16),
                           INT8_C( 125), INT8_C(-105), INT8_C( 125), INT8_C( -96),
                           INT8_C( -17), INT8_C(-120), INT8_C( -70), INT8_C(  68),
                           INT8_C( -52), INT8_C(   1), INT8_C( -54), INT8_C(-120),
                           INT8_C( -42), INT8_C( -52), INT8_C( -87), INT8_C(-105),
                           INT8_C( -53), INT8_C( 101), INT8_C(  87), INT8_C(  15),
                           INT8_C(-111), INT8_C(-123), INT8_C(  62), INT8_C( -49),
                           INT8_C( -65), INT8_C( -37), INT8_C(  38), INT8_C( -49),
                           INT8_C( 106), INT8_C(  65), INT8_C(  -5), INT8_C( -23),
                           INT8_C(-121), INT8_C( 124), INT8_C( -37), INT8_C(-123),
                           INT8_C( -28), INT8_C( -38), INT8_C(-106), INT8_C( -50),
                           INT8_C( 107), INT8_C( -81), INT8_C(-100), INT8_C(  15),
                           INT8_C( -12), INT8_C(-115), INT8_C(  67), INT8_C(  86),
                           INT8_C( -40), INT8_C( -42), INT8_C( -88), INT8_C( 121)) },
    { easysimd_mm512_set_epi8(INT8_C(-108), INT8_C(-116), INT8_C(  21), INT8_C(-123),
                           INT8_C( -53), INT8_C(  42), INT8_C(  66), INT8_C(  13),
                           INT8_C(   9), INT8_C( 115), INT8_C(  86), INT8_C( 126),
                           INT8_C( -24), INT8_C(  35), INT8_C(  -5), INT8_C( 103),
                           INT8_C(  38), INT8_C( 111), INT8_C(  24), INT8_C( -71),
                           INT8_C(  -1), INT8_C(  17), INT8_C( -63), INT8_C( -13),
                           INT8_C(  14), INT8_C(  82), INT8_C(  78), INT8_C(-102),
                           INT8_C(  -7), INT8_C(  93), INT8_C(  25), INT8_C( 103),
                           INT8_C( 113), INT8_C( -15), INT8_C( -19), INT8_C( -73),
                           INT8_C( -11), INT8_C( 103), INT8_C( -97), INT8_C( 123),
                           INT8_C(  28), INT8_C(  53), INT8_C( -15), INT8_C( 122),
                           INT8_C(   3), INT8_C( -54), INT8_C( -61), INT8_C(  58),
                           INT8_C( -44), INT8_C(  -3), INT8_C( -43), INT8_C( -35),
                           INT8_C(-118), INT8_C( -18), INT8_C(  15), INT8_C(  54),
                           INT8_C(-102), INT8_C( -58), INT8_C( -74), INT8_C( -70),
                           INT8_C(  46), INT8_C(  48), INT8_C( -35), INT8_C(  92)),
      easysimd_mm512_set_epi8(INT8_C(   6), INT8_C(  68), INT8_C(  77), INT8_C( -94),
                           INT8_C( -48), INT8_C(-101), INT8_C(  -8), INT8_C(  82),
                           INT8_C(  50), INT8_C( -15), INT8_C(   6), INT8_C(  30),
                           INT8_C( -47), INT8_C( -15), INT8_C( -14), INT8_C( -97),
                           INT8_C(  28), INT8_C( -47), INT8_C( -92), INT8_C( -84),
                           INT8_C( -37), INT8_C( -33), INT8_C(-123), INT8_C( -19),
                           INT8_C(  58), INT8_C(  29), INT8_C(  93), INT8_C( -55),
                           INT8_C(-127), INT8_C( -60), INT8_C(  32), INT8_C( 116),
                           INT8_C( -46), INT8_C(  51), INT8_C( -40), INT8_C(  10),
                           INT8_C(   4), INT8_C(  50), INT8_C(  48), INT8_C(  53),
                           INT8_C(  78), INT8_C(  21), INT8_C(  64), INT8_C( 107),
                           INT8_C(  16), INT8_C(  48), INT8_C( -46), INT8_C(  62),
                           INT8_C(  75), INT8_C(  85), INT8_C(-115), INT8_C( -14),
                           INT8_C( -99), INT8_C(  86), INT8_C(-116), INT8_C( -74),
                           INT8_C(  38), INT8_C(  27), INT8_C(-115), INT8_C(  55),
                           INT8_C( -91), INT8_C( -71), INT8_C( -14), INT8_C( -84)),
      easysimd_mm512_set_epi8(INT8_C(-114), INT8_C(  72), INT8_C( -56), INT8_C( -29),
                           INT8_C(  -5), INT8_C(-113), INT8_C(  74), INT8_C( -69),
                           INT8_C( -41), INT8_C(-126), INT8_C(  80), INT8_C(  96),
                           INT8_C(  23), INT8_C(  50), INT8_C(   9), INT8_C( -56),
                           INT8_C(  10), INT8_C( -98), INT8_C( 116), INT8_C(  13),
                           INT8_C(  36), INT8_C(  50), INT8_C(  60), INT8_C(   6),
                           INT8_C( -44), INT8_C(  53), INT8_C( -15), INT8_C( -47),
                           INT8_C( 120), INT8_C(-103), INT8_C(  -7), INT8_C( -13),
                           INT8_C( -97), INT8_C( -66), INT8_C(  21), INT8_C( -83),
                           INT8_C( -15), INT8_C(  53), INT8_C( 111), INT8_C(  70),
                           INT8_C( -50), INT8_C(  32), INT8_C( -79), INT8_C(  15),
                           INT8_C( -13), INT8_C(-102), INT8_C( -15), INT8_C(  -4),
                           INT8_C(-119), INT8_C( -88), INT8_C(  72), INT8_C( -21),
                           INT8_C( -19), INT8_C(-104), INT8_C(-125), INT8_C(-128),
                           INT8_C( 116), INT8_C( -85), INT8_C(  41), INT8_C(-125),
                           INT8_C(-119), INT8_C( 119), INT8_C( -21), INT8_C( -80)) },
    { easysimd_mm512_set_epi8(INT8_C(   2), INT8_C( -77), INT8_C( -19), INT8_C(  41),
                           INT8_C( -13), INT8_C(  75), INT8_C(-123), INT8_C(  96),
                           INT8_C( -86), INT8_C( -24), INT8_C( -27), INT8_C( -84),
                           INT8_C(  35), INT8_C( -86), INT8_C( -72), INT8_C( -97),
                           INT8_C(  44), INT8_C(  11), INT8_C(-106), INT8_C(  44),
                           INT8_C(   0), INT8_C(  90), INT8_C( -79), INT8_C(  91),
                           INT8_C( 119), INT8_C(  59), INT8_C( 105), INT8_C(-128),
                           INT8_C( 110), INT8_C( -29), INT8_C(  67), INT8_C( 114),
                           INT8_C( -39), INT8_C( -49), INT8_C( 105), INT8_C( -40),
                           INT8_C( -33), INT8_C( 120), INT8_C( -27), INT8_C( 100),
                           INT8_C( -90), INT8_C(  86), INT8_C( -18), INT8_C( -57),
                           INT8_C(  84), INT8_C( -26), INT8_C( -77), INT8_C(  17),
                           INT8_C( -47), INT8_C(  51), INT8_C( -83), INT8_C(  53),
                           INT8_C(  71), INT8_C(  96), INT8_C( 110), INT8_C( -89),
                           INT8_C(  27), INT8_C( -45), INT8_C(-126), INT8_C(  40),
                           INT8_C(  95), INT8_C( -87), INT8_C( -62), INT8_C( -52)),
      easysimd_mm512_set_epi8(INT8_C( -84), INT8_C( 127), INT8_C(  61), INT8_C( -16),
                           INT8_C(  30), INT8_C(   6), INT8_C(-112), INT8_C( 104),
                           INT8_C( -60), INT8_C( -88), INT8_C( -39), INT8_C( -19),
                           INT8_C(  44), INT8_C(  36), INT8_C( 105), INT8_C( 120),
                           INT8_C( -26), INT8_C(  21), INT8_C(  14), INT8_C(  42),
                           INT8_C(  49), INT8_C( -84), INT8_C(-120), INT8_C(-107),
                           INT8_C( 123), INT8_C( -47), INT8_C(  21), INT8_C( -10),
                           INT8_C(  95), INT8_C( 124), INT8_C( -33), INT8_C( -34),
                           INT8_C( -33), INT8_C( -71), INT8_C(  11), INT8_C(  74),
                           INT8_C( 104), INT8_C( 108), INT8_C( -35), INT8_C( -59),
                           INT8_C( -55), INT8_C(-126), INT8_C( 107), INT8_C(  23),
                           INT8_C(  29), INT8_C( -27), INT8_C( 123), INT8_C(  23),
                           INT8_C( -83), INT8_C( -90), INT8_C(   9), INT8_C(  94),
                           INT8_C(  91), INT8_C(  69), INT8_C( -51), INT8_C(-103),
                           INT8_C( -72), INT8_C( -45), INT8_C(  16), INT8_C( 108),
                           INT8_C( -80), INT8_C(  27), INT8_C(  58), INT8_C( -83)),
      easysimd_mm512_set_epi8(INT8_C(  86), INT8_C(  52), INT8_C( -80), INT8_C(  57),
                           INT8_C( -43), INT8_C(  69), INT8_C( -11), INT8_C(  -8),
                           INT8_C( -26), INT8_C(  64), INT8_C(  12), INT8_C( -65),
                           INT8_C(  -9), INT8_C(-122), INT8_C(  79), INT8_C(  39),
                           INT8_C(  70), INT8_C( -10), INT8_C(-120), INT8_C(   2),
                           INT8_C( -49), INT8_C( -82), INT8_C(  41), INT8_C( -58),
                           INT8_C(  -4), INT8_C( 106), INT8_C(  84), INT8_C(-118),
                           INT8_C(  15), INT8_C( 103), INT8_C( 100), INT8_C(-108),
                           INT8_C(  -6), INT8_C(  22), INT8_C(  94), INT8_C(-114),
                           INT8_C( 119), INT8_C(  12), INT8_C(   8), INT8_C( -97),
                           INT8_C( -35), INT8_C( -44), INT8_C(-125), INT8_C( -80),
                           INT8_C(  55), INT8_C(   1), INT8_C(  56), INT8_C(  -6),
                           INT8_C(  36), INT8_C(-115), INT8_C( -92), INT8_C( -41),
                           INT8_C( -20), INT8_C(  27), INT8_C( -95), INT8_C(  14),
                           INT8_C(  99), INT8_C(   0), INT8_C( 114), INT8_C( -68),
                           INT8_C( -81), INT8_C(-114), INT8_C(-120), INT8_C(  31)) },
    { easysimd_mm512_set_epi8(INT8_C(  17), INT8_C(  99), INT8_C( -13), INT8_C( -49),
                           INT8_C(  45), INT8_C(-128), INT8_C(  55), INT8_C( 105),
                           INT8_C( -34), INT8_C( -51), INT8_C( -97), INT8_C(-103),
                           INT8_C(-124), INT8_C( 111), INT8_C(  74), INT8_C(  75),
                           INT8_C( 102), INT8_C(  98), INT8_C(-117), INT8_C(   9),
                           INT8_C( -74), INT8_C(  61), INT8_C(  99), INT8_C( 124),
                           INT8_C(  79), INT8_C(-114), INT8_C(  19), INT8_C(  97),
                           INT8_C(-100), INT8_C(-124), INT8_C( -17), INT8_C( -62),
                           INT8_C(  25), INT8_C(  -3), INT8_C(  -7), INT8_C(  72),
                           INT8_C(-117), INT8_C( -27), INT8_C( -56), INT8_C(  92),
                           INT8_C( -20), INT8_C( -53), INT8_C(   2), INT8_C( -38),
                           INT8_C( -81), INT8_C(  59), INT8_C(  66), INT8_C(  90),
                           INT8_C(  36), INT8_C( 100), INT8_C( 112), INT8_C( 123),
                           INT8_C( -72), INT8_C( -97), INT8_C(-115), INT8_C(  17),
                           INT8_C( -93), INT8_C(-122), INT8_C(  31), INT8_C(  27),
                           INT8_C( 109), INT8_C( 115), INT8_C(  53), INT8_C( -96)),
      easysimd_mm512_set_epi8(INT8_C( -43), INT8_C( -18), INT8_C( 114), INT8_C( -29),
                           INT8_C( 118), INT8_C(  -1), INT8_C( -20), INT8_C( -38),
                           INT8_C( -80), INT8_C(  88), INT8_C(-111), INT8_C( -91),
                           INT8_C(  44), INT8_C( -72), INT8_C( 106), INT8_C(  19),
                           INT8_C( -46), INT8_C( 107), INT8_C(  46), INT8_C(  44),
                           INT8_C( -65), INT8_C(-128), INT8_C(  41), INT8_C(  44),
                           INT8_C(  68), INT8_C(  69), INT8_C( -78), INT8_C( -47),
                           INT8_C( 109), INT8_C( 120), INT8_C( -57), INT8_C( -95),
                           INT8_C(  95), INT8_C(  80), INT8_C( -30), INT8_C(  97),
                           INT8_C( -48), INT8_C( -97), INT8_C( 111), INT8_C( -80),
                           INT8_C(-122), INT8_C( -81), INT8_C( -71), INT8_C(  85),
                           INT8_C(  77), INT8_C( -42), INT8_C(-115), INT8_C( -77),
                           INT8_C(  29), INT8_C(  77), INT8_C(  64), INT8_C( -20),
                           INT8_C(  27), INT8_C(  41), INT8_C(  13), INT8_C( 109),
                           INT8_C(  22), INT8_C( -98), INT8_C(  20), INT8_C( -28),
                           INT8_C(  66), INT8_C(  -7), INT8_C(-113), INT8_C(-119)),
      easysimd_mm512_set_epi8(INT8_C(  60), INT8_C( 117), INT8_C(-127), INT8_C( -20),
                           INT8_C( -73), INT8_C(-127), INT8_C(  75), INT8_C(-113),
                           INT8_C(  46), INT8_C( 117), INT8_C(  14), INT8_C( -12),
                           INT8_C(  88), INT8_C( -73), INT8_C( -32), INT8_C(  56),
                           INT8_C(-108), INT8_C(  -9), INT8_C(  93), INT8_C( -35),
                           INT8_C(  -9), INT8_C( -67), INT8_C(  58), INT8_C(  80),
                           INT8_C(  11), INT8_C(  73), INT8_C(  97), INT8_C(-112),
                           INT8_C(  47), INT8_C(  12), INT8_C(  40), INT8_C(  33),
                           INT8_C( -70), INT8_C( -83), INT8_C(  23), INT8_C( -25),
                           INT8_C( -69), INT8_C(  70), INT8_C(  89), INT8_C( -84),
                           INT8_C( 102), INT8_C(  28), INT8_C(  73), INT8_C(-123),
                           INT8_C(  98), INT8_C( 101), INT8_C( -75), INT8_C( -89),
                           INT8_C(   7), INT8_C(  23), INT8_C(  48), INT8_C(-113),
                           INT8_C( -99), INT8_C( 118), INT8_C(-128), INT8_C( -92),
                           INT8_C(-115), INT8_C( -24), INT8_C(  11), INT8_C(  55),
                           INT8_C(  43), INT8_C( 122), INT8_C( -90), INT8_C(  23)) },
    { easysimd_mm512_set_epi8(INT8_C(-124), INT8_C( -73), INT8_C(  74), INT8_C(   5),
                           INT8_C(  -9), INT8_C(  17), INT8_C( -81), INT8_C( -54),
                           INT8_C(  -5), INT8_C( -33), INT8_C( -12), INT8_C(  26),
                           INT8_C(  86), INT8_C( 122), INT8_C( -44), INT8_C( -23),
                           INT8_C(   0), INT8_C(  43), INT8_C( -25), INT8_C(-122),
                           INT8_C( -79), INT8_C(-122), INT8_C( -88), INT8_C(-121),
                           INT8_C(-102), INT8_C(  66), INT8_C( -93), INT8_C( 105),
                           INT8_C( 109), INT8_C( -68), INT8_C(  24), INT8_C( -54),
                           INT8_C(  40), INT8_C(  68), INT8_C(   2), INT8_C(  60),
                           INT8_C(   0), INT8_C(   5), INT8_C(  59), INT8_C( -54),
                           INT8_C( -76), INT8_C(  27), INT8_C( -23), INT8_C(  77),
                           INT8_C(-108), INT8_C( -28), INT8_C(-114), INT8_C(  56),
                           INT8_C( -54), INT8_C(-108), INT8_C( -15), INT8_C( -89),
                           INT8_C(-103), INT8_C( -45), INT8_C(  74), INT8_C(  -3),
                           INT8_C(-108), INT8_C(  55), INT8_C( -79), INT8_C( -62),
                           INT8_C(  14), INT8_C( 106), INT8_C( -16), INT8_C( -10)),
      easysimd_mm512_set_epi8(INT8_C( -47), INT8_C( 124), INT8_C(  57), INT8_C( -74),
                           INT8_C(  20), INT8_C( 124), INT8_C(  70), INT8_C( -69),
                           INT8_C( -65), INT8_C( -12), INT8_C( 124), INT8_C( -90),
                           INT8_C(-113), INT8_C(  63), INT8_C( -79), INT8_C( -70),
                           INT8_C( -76), INT8_C( -34), INT8_C( -60), INT8_C(  -4),
                           INT8_C( -41), INT8_C(  60), INT8_C(  77), INT8_C( -57),
                           INT8_C(  13), INT8_C(   2), INT8_C( 111), INT8_C( -39),
                           INT8_C(  41), INT8_C(  54), INT8_C( -37), INT8_C( 114),
                           INT8_C(  92), INT8_C(-111), INT8_C(  77), INT8_C(  14),
                           INT8_C(-104), INT8_C( -39), INT8_C( -74), INT8_C(  66),
                           INT8_C(  16), INT8_C( -26), INT8_C( -89), INT8_C(-114),
                           INT8_C( -68), INT8_C(   6), INT8_C(  62), INT8_C( -93),
                           INT8_C(  55), INT8_C(-113), INT8_C( -60), INT8_C( -56),
                           INT8_C( -37), INT8_C(   2), INT8_C( -15), INT8_C(  88),
                           INT8_C(  26), INT8_C(  54), INT8_C(  82), INT8_C( 124),
                           INT8_C( -38), INT8_C(-107), INT8_C(  40), INT8_C(  13)),
      easysimd_mm512_set_epi8(INT8_C( -77), INT8_C(  59), INT8_C(  17), INT8_C(  79),
                           INT8_C( -29), INT8_C(-107), INT8_C( 105), INT8_C(  15),
                           INT8_C(  60), INT8_C( -21), INT8_C( 120), INT8_C( 116),
                           INT8_C( -57), INT8_C(  59), INT8_C(  35), INT8_C(  47),
                           INT8_C(  76), INT8_C(  77), INT8_C(  35), INT8_C(-118),
                           INT8_C( -38), INT8_C(  74), INT8_C(  91), INT8_C( -64),
                           INT8_C(-115), INT8_C(  64), INT8_C(  52), INT8_C(-112),
                           INT8_C(  68), INT8_C(-122), INT8_C(  61), INT8_C(  88),
                           INT8_C( -52), INT8_C( -77), INT8_C( -75), INT8_C(  46),
                           INT8_C( 104), INT8_C(  44), INT8_C(-123), INT8_C(-120),
                           INT8_C( -92), INT8_C(  53), INT8_C(  66), INT8_C( -65),
                           INT8_C( -40), INT8_C( -34), INT8_C(  80), INT8_C(-107),
                           INT8_C(-109), INT8_C(   5), INT8_C(  45), INT8_C( -33),
                           INT8_C( -66), INT8_C( -47), INT8_C(  89), INT8_C( -91),
                           INT8_C( 122), INT8_C(   1), INT8_C(  95), INT8_C(  70),
                           INT8_C(  52), INT8_C( -43), INT8_C( -56), INT8_C( -23)) },
    { easysimd_mm512_set_epi8(INT8_C(   5), INT8_C( -68), INT8_C( -18), INT8_C( -37),
                           INT8_C(   5), INT8_C(  16), INT8_C(-109), INT8_C( -67),
                           INT8_C( -62), INT8_C(  -4), INT8_C(  14), INT8_C(-109),
                           INT8_C( -29), INT8_C(-121), INT8_C(-109), INT8_C( -55),
                           INT8_C(   1), INT8_C( -38), INT8_C( 107), INT8_C(  55),
                           INT8_C( -36), INT8_C( -76), INT8_C(  35), INT8_C( -40),
                           INT8_C(  10), INT8_C( -90), INT8_C( -48), INT8_C(-112),
                           INT8_C(  -9), INT8_C( -53), INT8_C( 105), INT8_C(  27),
                           INT8_C( -97), INT8_C(-124), INT8_C(   4), INT8_C( -36),
                           INT8_C( -16), INT8_C( -87), INT8_C( -89), INT8_C(-104),
                           INT8_C( -30), INT8_C(-101), INT8_C(  69), INT8_C(  79),
                           INT8_C(  59), INT8_C( -97), INT8_C( -15), INT8_C(  17),
                           INT8_C( 106), INT8_C( -85), INT8_C( 126), INT8_C(-121),
                           INT8_C( -91), INT8_C(  26), INT8_C(-115), INT8_C(-117),
                           INT8_C(  91), INT8_C(  73), INT8_C( -60), INT8_C(  69),
                           INT8_C( -23), INT8_C(  48), INT8_C(  70), INT8_C(  -8)),
      easysimd_mm512_set_epi8(INT8_C(  91), INT8_C(-103), INT8_C(  69), INT8_C(  61),
                           INT8_C( -82), INT8_C(  73), INT8_C( 122), INT8_C( -22),
                           INT8_C( 122), INT8_C(  76), INT8_C(  -9), INT8_C( 121),
                           INT8_C(-123), INT8_C(-119), INT8_C(-127), INT8_C( 126),
                           INT8_C( 105), INT8_C(  10), INT8_C(-120), INT8_C(-127),
                           INT8_C( -50), INT8_C(  15), INT8_C( -93), INT8_C( -86),
                           INT8_C(-125), INT8_C(  45), INT8_C( -39), INT8_C(-119),
                           INT8_C(  74), INT8_C( -92), INT8_C( -78), INT8_C(  53),
                           INT8_C(  17), INT8_C( -21), INT8_C( 105), INT8_C(-102),
                           INT8_C(  -1), INT8_C( -19), INT8_C( 110), INT8_C( -84),
                           INT8_C( -93), INT8_C(  19), INT8_C( -98), INT8_C(-128),
                           INT8_C( -23), INT8_C(  49), INT8_C( 100), INT8_C( 122),
                           INT8_C( -96), INT8_C(-103), INT8_C(  60), INT8_C( -24),
                           INT8_C(  23), INT8_C( -52), INT8_C( -37), INT8_C( -56),
                           INT8_C( -50), INT8_C(   4), INT8_C( -69), INT8_C(   1),
                           INT8_C( -25), INT8_C( -10), INT8_C(  93), INT8_C(  51)),
      easysimd_mm512_set_epi8(INT8_C( -86), INT8_C(  35), INT8_C( -87), INT8_C( -98),
                           INT8_C(  87), INT8_C( -57), INT8_C(  25), INT8_C( -45),
                           INT8_C(  72), INT8_C( -80), INT8_C(  23), INT8_C(  26),
                           INT8_C(  94), INT8_C(  -2), INT8_C(  18), INT8_C(  75),
                           INT8_C(-104), INT8_C( -48), INT8_C( -29), INT8_C( -74),
                           INT8_C(  14), INT8_C( -91), INT8_C(-128), INT8_C(  46),
                           INT8_C(-121), INT8_C( 121), INT8_C(  -9), INT8_C(   7),
                           INT8_C( -83), INT8_C(  39), INT8_C( -73), INT8_C( -26),
                           INT8_C(-114), INT8_C(-103), INT8_C(-101), INT8_C(  66),
                           INT8_C( -15), INT8_C( -68), INT8_C(  57), INT8_C( -20),
                           INT8_C(  63), INT8_C(-120), INT8_C( -89), INT8_C( -49),
                           INT8_C(  82), INT8_C( 110), INT8_C(-115), INT8_C(-105),
                           INT8_C( -54), INT8_C(  18), INT8_C(  66), INT8_C( -97),
                           INT8_C(-114), INT8_C(  78), INT8_C( -78), INT8_C( -61),
                           INT8_C(-115), INT8_C(  69), INT8_C(   9), INT8_C(  68),
                           INT8_C(   2), INT8_C(  58), INT8_C( -23), INT8_C( -59)) },
    { easysimd_mm512_set_epi8(INT8_C( -89), INT8_C(  43), INT8_C(  52), INT8_C(  82),
                           INT8_C( -37), INT8_C(  55), INT8_C( 112), INT8_C( -22),
                           INT8_C( -75), INT8_C( -36), INT8_C( -34), INT8_C( -15),
                           INT8_C(  35), INT8_C( -42), INT8_C(-101), INT8_C(  -5),
                           INT8_C(   2), INT8_C(  35), INT8_C(  14), INT8_C( -73),
                           INT8_C( -50), INT8_C( -33), INT8_C( -65), INT8_C(  94),
                           INT8_C(  -6), INT8_C( -21), INT8_C( -28), INT8_C(  21),
                           INT8_C( 102), INT8_C( -87), INT8_C( 114), INT8_C( 125),
                           INT8_C( 113), INT8_C( 124), INT8_C(-121), INT8_C(-122),
                           INT8_C(  23), INT8_C( 107), INT8_C(  24), INT8_C( 126),
                           INT8_C(  80), INT8_C(  59), INT8_C(  39), INT8_C( -61),
                           INT8_C(-105), INT8_C(  32), INT8_C(  55), INT8_C(  -9),
                           INT8_C(  60), INT8_C(-125), INT8_C(  72), INT8_C( -36),
                           INT8_C(  77), INT8_C( -65), INT8_C( 117), INT8_C( -85),
                           INT8_C(  98), INT8_C( -83), INT8_C( -69), INT8_C( -52),
                           INT8_C(  41), INT8_C( -10), INT8_C( -18), INT8_C(  56)),
      easysimd_mm512_set_epi8(INT8_C(  22), INT8_C( 122), INT8_C( -90), INT8_C(   2),
                           INT8_C( -65), INT8_C(  51), INT8_C( -94), INT8_C( -50),
                           INT8_C( -15), INT8_C(  19), INT8_C( -19), INT8_C(  66),
                           INT8_C( 119), INT8_C(-118), INT8_C(-112), INT8_C(-116),
                           INT8_C(  44), INT8_C( -12), INT8_C(  31), INT8_C(  43),
                           INT8_C( -16), INT8_C( -37), INT8_C( -24), INT8_C( -32),
                           INT8_C( -95), INT8_C( -86), INT8_C( -96), INT8_C(  80),
                           INT8_C(  68), INT8_C(  13), INT8_C(  -8), INT8_C(  67),
                           INT8_C( 107), INT8_C(-125), INT8_C( 104), INT8_C( -80),
                           INT8_C(  97), INT8_C( -78), INT8_C( 106), INT8_C( -53),
                           INT8_C( -36), INT8_C( -90), INT8_C(  74), INT8_C( -72),
                           INT8_C(  59), INT8_C( -81), INT8_C(  -8), INT8_C( -25),
                           INT8_C( -55), INT8_C( -99), INT8_C(  20), INT8_C(   9),
                           INT8_C( -89), INT8_C( -90), INT8_C( 108), INT8_C(  56),
                           INT8_C( -19), INT8_C(  81), INT8_C( 122), INT8_C(   6),
                           INT8_C(-119), INT8_C( 122), INT8_C( -35), INT8_C( 106)),
      easysimd_mm512_set_epi8(INT8_C(-111), INT8_C( -79), INT8_C(-114), INT8_C(  80),
                           INT8_C(  28), INT8_C(   4), INT8_C( -50), INT8_C(  28),
                           INT8_C( -60), INT8_C( -55), INT8_C( -15), INT8_C( -81),
                           INT8_C( -84), INT8_C(  76), INT8_C(  11), INT8_C( 111),
                           INT8_C( -42), INT8_C(  47), INT8_C( -17), INT8_C(-116),
                           INT8_C( -34), INT8_C(   4), INT8_C( -41), INT8_C( 126),
                           INT8_C(  89), INT8_C(  65), INT8_C(  68), INT8_C( -59),
                           INT8_C(  34), INT8_C(-100), INT8_C( 122), INT8_C(  58),
                           INT8_C(   6), INT8_C(  -7), INT8_C(  31), INT8_C( -42),
                           INT8_C( -74), INT8_C( -71), INT8_C( -82), INT8_C( -77),
                           INT8_C( 116), INT8_C(-107), INT8_C( -35), INT8_C(  11),
                           INT8_C(  92), INT8_C( 113), INT8_C(  63), INT8_C(  16),
                           INT8_C( 115), INT8_C( -26), INT8_C(  52), INT8_C( -45),
                           INT8_C( -90), INT8_C(  25), INT8_C(   9), INT8_C( 115),
                           INT8_C( 117), INT8_C(  92), INT8_C(  65), INT8_C( -58),
                           INT8_C( -96), INT8_C( 124), INT8_C(  17), INT8_C( -50)) },
    { easysimd_mm512_set_epi8(INT8_C( 105), INT8_C(-115), INT8_C( 121), INT8_C(-101),
                           INT8_C(   0), INT8_C(  63), INT8_C( -42), INT8_C( -34),
                           INT8_C(  -5), INT8_C( -47), INT8_C(-123), INT8_C( -52),
                           INT8_C( -86), INT8_C( -28), INT8_C( -63), INT8_C(  20),
                           INT8_C( -60), INT8_C( -63), INT8_C(  99), INT8_C(  78),
                           INT8_C(  56), INT8_C( -72), INT8_C( -55), INT8_C( -72),
                           INT8_C(  79), INT8_C( -81), INT8_C( 124), INT8_C( -85),
                           INT8_C( -65), INT8_C( 122), INT8_C( -25), INT8_C( -58),
                           INT8_C( -64), INT8_C(  52), INT8_C( -12), INT8_C(   1),
                           INT8_C( -62), INT8_C( -28), INT8_C( -28), INT8_C(-104),
                           INT8_C(  54), INT8_C(-103), INT8_C( -55), INT8_C( -22),
                           INT8_C( -91), INT8_C(   6), INT8_C(  -9), INT8_C( -31),
                           INT8_C(  18), INT8_C(-111), INT8_C(  58), INT8_C(  71),
                           INT8_C( -73), INT8_C( -96), INT8_C(  28), INT8_C(  -4),
                           INT8_C(  47), INT8_C(  66), INT8_C( 121), INT8_C(  38),
                           INT8_C(  69), INT8_C(-107), INT8_C( -57), INT8_C(-120)),
      easysimd_mm512_set_epi8(INT8_C( -49), INT8_C(  15), INT8_C( -15), INT8_C( -59),
                           INT8_C(-113), INT8_C( 102), INT8_C( -48), INT8_C( -78),
                           INT8_C(  31), INT8_C(  94), INT8_C(  79), INT8_C(  92),
                           INT8_C( 106), INT8_C( -68), INT8_C(  96), INT8_C( -97),
                           INT8_C( -27), INT8_C(-118), INT8_C( -11), INT8_C( 112),
                           INT8_C(-125), INT8_C(  70), INT8_C(  26), INT8_C( -38),
                           INT8_C( -16), INT8_C(-112), INT8_C(  10), INT8_C(  98),
                           INT8_C(  -4), INT8_C( 120), INT8_C( -33), INT8_C(-127),
                           INT8_C( -65), INT8_C( -40), INT8_C(  88), INT8_C(  -6),
                           INT8_C(  74), INT8_C(  41), INT8_C(  39), INT8_C(  79),
                           INT8_C(-125), INT8_C(  -7), INT8_C(  62), INT8_C(-112),
                           INT8_C(-119), INT8_C(  -9), INT8_C(  71), INT8_C( -68),
                           INT8_C( -79), INT8_C(  48), INT8_C( -20), INT8_C( -97),
                           INT8_C(-116), INT8_C( 120), INT8_C( -65), INT8_C(   6),
                           INT8_C( -32), INT8_C( -75), INT8_C(-106), INT8_C(  26),
                           INT8_C( -96), INT8_C(  50), INT8_C( -45), INT8_C(  16)),
      easysimd_mm512_set_epi8(INT8_C(-102), INT8_C( 126), INT8_C(-120), INT8_C( -42),
                           INT8_C( 113), INT8_C( -39), INT8_C(   6), INT8_C(  44),
                           INT8_C( -36), INT8_C( 115), INT8_C(  54), INT8_C( 112),
                           INT8_C(  64), INT8_C(  40), INT8_C(  97), INT8_C( 117),
                           INT8_C( -33), INT8_C(  55), INT8_C( 110), INT8_C( -34),
                           INT8_C( -75), INT8_C( 114), INT8_C( -81), INT8_C( -34),
                           INT8_C(  95), INT8_C(  31), INT8_C( 114), INT8_C(  73),
                           INT8_C( -61), INT8_C(   2), INT8_C(   8), INT8_C(  69),
                           INT8_C(   1), INT8_C(  92), INT8_C(-100), INT8_C(   7),
                           INT8_C( 120), INT8_C( -69), INT8_C( -67), INT8_C(  73),
                           INT8_C( -77), INT8_C( -96), INT8_C(-117), INT8_C(  90),
                           INT8_C(  28), INT8_C(  15), INT8_C( -80), INT8_C(  37),
                           INT8_C(  97), INT8_C(  97), INT8_C(  78), INT8_C( -88),
                           INT8_C(  43), INT8_C(  40), INT8_C(  93), INT8_C( -10),
                           INT8_C(  79), INT8_C(-115), INT8_C( -29), INT8_C(  12),
                           INT8_C( -91), INT8_C(  99), INT8_C( -12), INT8_C( 120)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sub_epi8(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sub_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_sub_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask64 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C( 121), INT8_C(  -8), INT8_C(-121), INT8_C( -19),
                           INT8_C(  19), INT8_C(  -3), INT8_C(  10), INT8_C( -37),
                           INT8_C(  96), INT8_C(  15), INT8_C( -45), INT8_C( -44),
                           INT8_C( -83), INT8_C( -37), INT8_C(   8), INT8_C( 111),
                           INT8_C( -30), INT8_C( -99), INT8_C( 116), INT8_C( 112),
                           INT8_C(  67), INT8_C(-123), INT8_C(  59), INT8_C( -62),
                           INT8_C(  33), INT8_C(  51), INT8_C( -16), INT8_C( -35),
                           INT8_C( -53), INT8_C( 108), INT8_C( -37), INT8_C( -15),
                           INT8_C(  26), INT8_C(  83), INT8_C( -47), INT8_C( -23),
                           INT8_C(   6), INT8_C(  52), INT8_C( -19), INT8_C( 108),
                           INT8_C( -33), INT8_C( 120), INT8_C(  55), INT8_C(-128),
                           INT8_C( -46), INT8_C( 117), INT8_C(  41), INT8_C( -54),
                           INT8_C( -99), INT8_C( -39), INT8_C( 117), INT8_C(  57),
                           INT8_C(  78), INT8_C(-110), INT8_C(  -8), INT8_C(-114),
                           INT8_C( -54), INT8_C(  20), INT8_C( 112), INT8_C(  39),
                           INT8_C( -60), INT8_C(  36), INT8_C( -53), INT8_C(  53)),
      UINT64_C(           567574400),
      easysimd_mm512_set_epi8(INT8_C( 115), INT8_C(  65), INT8_C(-123), INT8_C( 116),
                           INT8_C( -46), INT8_C( -68), INT8_C(  -8), INT8_C(  96),
                           INT8_C( -90), INT8_C( -14), INT8_C(  27), INT8_C(  33),
                           INT8_C(  21), INT8_C(  58), INT8_C( -12), INT8_C( -76),
                           INT8_C(  70), INT8_C( -70), INT8_C(  -4), INT8_C(  64),
                           INT8_C(  35), INT8_C( -32), INT8_C(-115), INT8_C( 109),
                           INT8_C( -69), INT8_C( -61), INT8_C(-126), INT8_C(  45),
                           INT8_C(  93), INT8_C( 100), INT8_C( -53), INT8_C( 105),
                           INT8_C( -51), INT8_C(  43), INT8_C( -96), INT8_C( -95),
                           INT8_C(  86), INT8_C( -66), INT8_C( -47), INT8_C( 123),
                           INT8_C(  63), INT8_C(  16), INT8_C( -66), INT8_C(  12),
                           INT8_C( 115), INT8_C( -39), INT8_C(  10), INT8_C( 108),
                           INT8_C(-120), INT8_C( 110), INT8_C( -25), INT8_C(   7),
                           INT8_C( 105), INT8_C( -93), INT8_C(  68), INT8_C(   3),
                           INT8_C( 113), INT8_C( -50), INT8_C( -34), INT8_C(  22),
                           INT8_C( -61), INT8_C(  75), INT8_C(  28), INT8_C( 116)),
      easysimd_mm512_set_epi8(INT8_C(  26), INT8_C(  -6), INT8_C( -92), INT8_C(   7),
                           INT8_C(-105), INT8_C( -92), INT8_C(  38), INT8_C( -63),
                           INT8_C(  77), INT8_C(  86), INT8_C( 113), INT8_C( -48),
                           INT8_C( 108), INT8_C( -92), INT8_C(  69), INT8_C(  74),
                           INT8_C(  67), INT8_C(  96), INT8_C( -34), INT8_C(  78),
                           INT8_C( 124), INT8_C(   9), INT8_C(  -1), INT8_C( -86),
                           INT8_C( -35), INT8_C(  26), INT8_C(  67), INT8_C(  46),
                           INT8_C(  75), INT8_C(-119), INT8_C(  68), INT8_C(  31),
                           INT8_C( -52), INT8_C(-102), INT8_C(  -4), INT8_C( 118),
                           INT8_C(   0), INT8_C( -44), INT8_C( 123), INT8_C( -73),
                           INT8_C(  84), INT8_C(  30), INT8_C(  -8), INT8_C(  64),
                           INT8_C( -20), INT8_C( -79), INT8_C( -85), INT8_C( -23),
                           INT8_C( -34), INT8_C(  -4), INT8_C( -85), INT8_C( 107),
                           INT8_C(  -6), INT8_C(  16), INT8_C( -66), INT8_C(-113),
                           INT8_C(  60), INT8_C( 127), INT8_C( -54), INT8_C( -36),
                           INT8_C(  73), INT8_C( -97), INT8_C( -65), INT8_C(  63)),
      easysimd_mm512_set_epi8(INT8_C( 121), INT8_C(  -8), INT8_C(-121), INT8_C( -19),
                           INT8_C(  19), INT8_C(  -3), INT8_C(  10), INT8_C( -37),
                           INT8_C(  96), INT8_C(  15), INT8_C( -45), INT8_C( -44),
                           INT8_C( -83), INT8_C( -37), INT8_C(   8), INT8_C( 111),
                           INT8_C( -30), INT8_C( -99), INT8_C( 116), INT8_C( 112),
                           INT8_C(  67), INT8_C(-123), INT8_C(  59), INT8_C( -62),
                           INT8_C(  33), INT8_C(  51), INT8_C( -16), INT8_C( -35),
                           INT8_C( -53), INT8_C( 108), INT8_C( -37), INT8_C( -15),
                           INT8_C(  26), INT8_C(  83), INT8_C( -92), INT8_C( -23),
                           INT8_C(   6), INT8_C(  52), INT8_C( -19), INT8_C( -60),
                           INT8_C( -21), INT8_C( -14), INT8_C(  55), INT8_C( -52),
                           INT8_C( -46), INT8_C(  40), INT8_C(  41), INT8_C( -54),
                           INT8_C( -99), INT8_C( 114), INT8_C(  60), INT8_C(-100),
                           INT8_C( 111), INT8_C(-109), INT8_C(-122), INT8_C( 116),
                           INT8_C(  53), INT8_C(  20), INT8_C( 112), INT8_C(  39),
                           INT8_C( -60), INT8_C(  36), INT8_C( -53), INT8_C(  53)) },
    { easysimd_mm512_set_epi8(INT8_C(  44), INT8_C(  68), INT8_C(  96), INT8_C(  88),
                           INT8_C(   1), INT8_C(  68), INT8_C(  46), INT8_C(  19),
                           INT8_C(  31), INT8_C(  85), INT8_C(  35), INT8_C(  68),
                           INT8_C( -79), INT8_C(  41), INT8_C(  28), INT8_C(  92),
                           INT8_C( -26), INT8_C(  20), INT8_C( -16), INT8_C(  -7),
                           INT8_C(  41), INT8_C(  71), INT8_C(  88), INT8_C(  39),
                           INT8_C( -42), INT8_C(  76), INT8_C(  40), INT8_C( 108),
                           INT8_C( -87), INT8_C(-126), INT8_C(  42), INT8_C(  58),
                           INT8_C(   2), INT8_C( -23), INT8_C(   5), INT8_C(-116),
                           INT8_C(  34), INT8_C(  66), INT8_C(  28), INT8_C(  86),
                           INT8_C(  50), INT8_C( -67), INT8_C(  20), INT8_C(  73),
                           INT8_C(  27), INT8_C( -29), INT8_C(  84), INT8_C( 112),
                           INT8_C( 104), INT8_C(  53), INT8_C( -89), INT8_C(-113),
                           INT8_C(  -4), INT8_C(  94), INT8_C(  75), INT8_C(  21),
                           INT8_C(-120), INT8_C( -25), INT8_C( 111), INT8_C(-128),
                           INT8_C(  -4), INT8_C(  58), INT8_C(-115), INT8_C(   4)),
      UINT64_C(          2779079274),
      easysimd_mm512_set_epi8(INT8_C( -70), INT8_C( -58), INT8_C(  69), INT8_C(  -7),
                           INT8_C( 115), INT8_C( -10), INT8_C( -39), INT8_C(  78),
                           INT8_C(  56), INT8_C( 116), INT8_C( 104), INT8_C( -92),
                           INT8_C(   5), INT8_C( -47), INT8_C(  27), INT8_C(  94),
                           INT8_C(-126), INT8_C(  88), INT8_C(  80), INT8_C( 112),
                           INT8_C(  21), INT8_C(  62), INT8_C(  86), INT8_C(-103),
                           INT8_C(  66), INT8_C(  -9), INT8_C( -26), INT8_C(  47),
                           INT8_C( -50), INT8_C(-118), INT8_C( 115), INT8_C(  49),
                           INT8_C(  42), INT8_C(   6), INT8_C(  92), INT8_C(   2),
                           INT8_C(  63), INT8_C(  -6), INT8_C( -32), INT8_C(  15),
                           INT8_C(  66), INT8_C(  82), INT8_C(  -9), INT8_C( -79),
                           INT8_C(-123), INT8_C(  52), INT8_C( -90), INT8_C(-111),
                           INT8_C(  62), INT8_C( -43), INT8_C( -50), INT8_C(  62),
                           INT8_C(   4), INT8_C( -92), INT8_C(  86), INT8_C( -32),
                           INT8_C(  69), INT8_C( -15), INT8_C(  55), INT8_C(-127),
                           INT8_C( -36), INT8_C(  56), INT8_C(  46), INT8_C(-119)),
      easysimd_mm512_set_epi8(INT8_C(  18), INT8_C(  68), INT8_C(  -9), INT8_C(  64),
                           INT8_C(-111), INT8_C(  37), INT8_C( -82), INT8_C( -95),
                           INT8_C(  54), INT8_C(  75), INT8_C( -77), INT8_C( -34),
                           INT8_C(  52), INT8_C( -80), INT8_C( -94), INT8_C(  90),
                           INT8_C(   0), INT8_C(  -8), INT8_C( 123), INT8_C(-111),
                           INT8_C(  16), INT8_C( 125), INT8_C( -51), INT8_C(  99),
                           INT8_C( -22), INT8_C( 121), INT8_C(  63), INT8_C( -55),
                           INT8_C( 117), INT8_C( 109), INT8_C(-126), INT8_C(-111),
                           INT8_C(  47), INT8_C(-127), INT8_C( 109), INT8_C(  -9),
                           INT8_C( -42), INT8_C(  36), INT8_C( -32), INT8_C( 115),
                           INT8_C( -89), INT8_C(   7), INT8_C(  90), INT8_C(  46),
                           INT8_C( -83), INT8_C( -35), INT8_C(  30), INT8_C( -19),
                           INT8_C( -99), INT8_C( -56), INT8_C( -70), INT8_C(  73),
                           INT8_C( -61), INT8_C(  27), INT8_C( 117), INT8_C(  47),
                           INT8_C(   0), INT8_C( -45), INT8_C(  59), INT8_C(  51),
                           INT8_C( -35), INT8_C(  70), INT8_C(  73), INT8_C(  33)),
      easysimd_mm512_set_epi8(INT8_C(  44), INT8_C(  68), INT8_C(  96), INT8_C(  88),
                           INT8_C(   1), INT8_C(  68), INT8_C(  46), INT8_C(  19),
                           INT8_C(  31), INT8_C(  85), INT8_C(  35), INT8_C(  68),
                           INT8_C( -79), INT8_C(  41), INT8_C(  28), INT8_C(  92),
                           INT8_C( -26), INT8_C(  20), INT8_C( -16), INT8_C(  -7),
                           INT8_C(  41), INT8_C(  71), INT8_C(  88), INT8_C(  39),
                           INT8_C( -42), INT8_C(  76), INT8_C(  40), INT8_C( 108),
                           INT8_C( -87), INT8_C(-126), INT8_C(  42), INT8_C(  58),
                           INT8_C(  -5), INT8_C( -23), INT8_C( -17), INT8_C(-116),
                           INT8_C(  34), INT8_C( -42), INT8_C(  28), INT8_C(-100),
                           INT8_C(-101), INT8_C( -67), INT8_C( -99), INT8_C(  73),
                           INT8_C(  27), INT8_C(  87), INT8_C(  84), INT8_C( -92),
                           INT8_C( 104), INT8_C(  13), INT8_C(  20), INT8_C(-113),
                           INT8_C(  -4), INT8_C(  94), INT8_C( -31), INT8_C(  21),
                           INT8_C(-120), INT8_C(  30), INT8_C(  -4), INT8_C(-128),
                           INT8_C(  -1), INT8_C(  58), INT8_C( -27), INT8_C(   4)) },
    { easysimd_mm512_set_epi8(INT8_C( -35), INT8_C(  32), INT8_C( -43), INT8_C( 108),
                           INT8_C(  83), INT8_C( -59), INT8_C(  -4), INT8_C( 125),
                           INT8_C( -31), INT8_C( 118), INT8_C( -25), INT8_C( -91),
                           INT8_C(  50), INT8_C( -74), INT8_C(  78), INT8_C(  95),
                           INT8_C( -84), INT8_C( -63), INT8_C(  87), INT8_C(-108),
                           INT8_C(  28), INT8_C( -70), INT8_C(  77), INT8_C(-113),
                           INT8_C( -20), INT8_C(  50), INT8_C(  95), INT8_C(-108),
                           INT8_C( 105), INT8_C( 114), INT8_C(-109), INT8_C(  19),
                           INT8_C( -79), INT8_C( 106), INT8_C(  61), INT8_C( -12),
                           INT8_C( 126), INT8_C(-117), INT8_C( 126), INT8_C(-125),
                           INT8_C( -93), INT8_C(  69), INT8_C( 104), INT8_C( 119),
                           INT8_C(  63), INT8_C(  95), INT8_C(-106), INT8_C( -66),
                           INT8_C( -47), INT8_C( -45), INT8_C( -60), INT8_C( -54),
                           INT8_C(-109), INT8_C( -45), INT8_C( -86), INT8_C( 121),
                           INT8_C(  23), INT8_C( -12), INT8_C(  67), INT8_C(  -6),
                           INT8_C( -37), INT8_C(  92), INT8_C( -35), INT8_C(  99)),
      UINT64_C(          1100920337),
      easysimd_mm512_set_epi8(INT8_C(  13), INT8_C( 104), INT8_C(  50), INT8_C(  43),
                           INT8_C(  82), INT8_C(  -5), INT8_C( -23), INT8_C( -47),
                           INT8_C(  99), INT8_C(-116), INT8_C( 118), INT8_C(  73),
                           INT8_C( -10), INT8_C( -88), INT8_C( -42), INT8_C( -58),
                           INT8_C( -49), INT8_C(  65), INT8_C( -18), INT8_C(  54),
                           INT8_C( -68), INT8_C(   1), INT8_C(  -7), INT8_C( -96),
                           INT8_C(   4), INT8_C( 115), INT8_C(  42), INT8_C(-106),
                           INT8_C(  31), INT8_C(  94), INT8_C( -71), INT8_C( -41),
                           INT8_C(  33), INT8_C(-106), INT8_C( -65), INT8_C(-107),
                           INT8_C(  71), INT8_C( -10), INT8_C( -21), INT8_C(-128),
                           INT8_C( -23), INT8_C(  20), INT8_C(   2), INT8_C(  96),
                           INT8_C(-128), INT8_C( -51), INT8_C( -38), INT8_C(  47),
                           INT8_C( -56), INT8_C( 123), INT8_C( -20), INT8_C( -50),
                           INT8_C(  -2), INT8_C(  40), INT8_C(  24), INT8_C( -98),
                           INT8_C(   4), INT8_C( -62), INT8_C( -44), INT8_C(  49),
                           INT8_C(  83), INT8_C( 115), INT8_C(   5), INT8_C(  57)),
      easysimd_mm512_set_epi8(INT8_C(  76), INT8_C( -52), INT8_C( -96), INT8_C(  -6),
                           INT8_C(-119), INT8_C( -87), INT8_C( 102), INT8_C(   5),
                           INT8_C(  24), INT8_C( -44), INT8_C( 110), INT8_C(-113),
                           INT8_C(-116), INT8_C(  -3), INT8_C(  62), INT8_C( -87),
                           INT8_C(   7), INT8_C( -54), INT8_C( -57), INT8_C( -66),
                           INT8_C(  42), INT8_C( -82), INT8_C(  46), INT8_C( -16),
                           INT8_C(  91), INT8_C( -73), INT8_C( -20), INT8_C( -77),
                           INT8_C( -11), INT8_C(  25), INT8_C(  12), INT8_C(  76),
                           INT8_C( -58), INT8_C(   3), INT8_C(-125), INT8_C( -36),
                           INT8_C(  18), INT8_C( -40), INT8_C( 111), INT8_C( 107),
                           INT8_C(  88), INT8_C(  48), INT8_C( 113), INT8_C( -90),
                           INT8_C(-117), INT8_C( 116), INT8_C(  46), INT8_C( -70),
                           INT8_C(  51), INT8_C( -55), INT8_C( 127), INT8_C(  82),
                           INT8_C( -88), INT8_C(  60), INT8_C( -59), INT8_C(  80),
                           INT8_C( -51), INT8_C(  11), INT8_C( -44), INT8_C(  33),
                           INT8_C(  29), INT8_C(   8), INT8_C(   5), INT8_C(  70)),
      easysimd_mm512_set_epi8(INT8_C( -35), INT8_C(  32), INT8_C( -43), INT8_C( 108),
                           INT8_C(  83), INT8_C( -59), INT8_C(  -4), INT8_C( 125),
                           INT8_C( -31), INT8_C( 118), INT8_C( -25), INT8_C( -91),
                           INT8_C(  50), INT8_C( -74), INT8_C(  78), INT8_C(  95),
                           INT8_C( -84), INT8_C( -63), INT8_C(  87), INT8_C(-108),
                           INT8_C(  28), INT8_C( -70), INT8_C(  77), INT8_C(-113),
                           INT8_C( -20), INT8_C(  50), INT8_C(  95), INT8_C(-108),
                           INT8_C( 105), INT8_C( 114), INT8_C(-109), INT8_C(  19),
                           INT8_C( -79), INT8_C(-109), INT8_C(  61), INT8_C( -12),
                           INT8_C( 126), INT8_C(-117), INT8_C( 126), INT8_C(  21),
                           INT8_C(-111), INT8_C(  69), INT8_C( 104), INT8_C( -70),
                           INT8_C( -11), INT8_C(  89), INT8_C( -84), INT8_C( -66),
                           INT8_C(-107), INT8_C( -45), INT8_C( 109), INT8_C( 124),
                           INT8_C(-109), INT8_C( -20), INT8_C(  83), INT8_C( 121),
                           INT8_C(  23), INT8_C( -12), INT8_C(  67), INT8_C(  16),
                           INT8_C( -37), INT8_C(  92), INT8_C( -35), INT8_C( -13)) },
    { easysimd_mm512_set_epi8(INT8_C(  27), INT8_C(  45), INT8_C(  71), INT8_C( -63),
                           INT8_C(  96), INT8_C(-106), INT8_C( -43), INT8_C(  10),
                           INT8_C( 104), INT8_C( -19), INT8_C(-110), INT8_C( 126),
                           INT8_C( -52), INT8_C( -56), INT8_C( -96), INT8_C( -27),
                           INT8_C(-125), INT8_C(-116), INT8_C(  25), INT8_C(  78),
                           INT8_C( -76), INT8_C( -85), INT8_C( -23), INT8_C( -19),
                           INT8_C(-106), INT8_C( 126), INT8_C(  19), INT8_C( -41),
                           INT8_C(  40), INT8_C(  78), INT8_C( -69), INT8_C(  57),
                           INT8_C(  73), INT8_C( -58), INT8_C(   3), INT8_C(  65),
                           INT8_C( -87), INT8_C( -37), INT8_C(   5), INT8_C(-126),
                           INT8_C(  14), INT8_C( -36), INT8_C( -37), INT8_C(  11),
                           INT8_C(  94), INT8_C(  24), INT8_C(   8), INT8_C( -31),
                           INT8_C( -38), INT8_C(  -1), INT8_C(  48), INT8_C(  32),
                           INT8_C(  88), INT8_C( -18), INT8_C( 123), INT8_C(  27),
                           INT8_C( 111), INT8_C(  27), INT8_C(  -3), INT8_C(  52),
                           INT8_C( -31), INT8_C(   2), INT8_C( -47), INT8_C(  64)),
      UINT64_C(           361367503),
      easysimd_mm512_set_epi8(INT8_C( -20), INT8_C(-104), INT8_C( -27), INT8_C(  38),
                           INT8_C(  31), INT8_C( -21), INT8_C(  79), INT8_C( -62),
                           INT8_C(  36), INT8_C(  95), INT8_C(  42), INT8_C(-102),
                           INT8_C( -80), INT8_C( -69), INT8_C( 107), INT8_C(-114),
                           INT8_C(  76), INT8_C( 123), INT8_C(-126), INT8_C( 108),
                           INT8_C( -55), INT8_C(  89), INT8_C( -46), INT8_C(  18),
                           INT8_C( 117), INT8_C(  25), INT8_C(-120), INT8_C(  27),
                           INT8_C(  34), INT8_C(  64), INT8_C(  71), INT8_C(  64),
                           INT8_C( -13), INT8_C( -73), INT8_C( 112), INT8_C(  25),
                           INT8_C( -18), INT8_C( -63), INT8_C( 109), INT8_C(   9),
                           INT8_C(  14), INT8_C(-125), INT8_C( -89), INT8_C(  70),
                           INT8_C(  10), INT8_C(  15), INT8_C( 120), INT8_C( -59),
                           INT8_C(  55), INT8_C( 108), INT8_C(  41), INT8_C(  -5),
                           INT8_C( -91), INT8_C(-120), INT8_C( -46), INT8_C( 122),
                           INT8_C( 116), INT8_C(-120), INT8_C( -67), INT8_C( -86),
                           INT8_C(  48), INT8_C(   2), INT8_C(  37), INT8_C( -26)),
      easysimd_mm512_set_epi8(INT8_C( -70), INT8_C(   3), INT8_C( 118), INT8_C(  37),
                           INT8_C( 104), INT8_C( 111), INT8_C( -17), INT8_C( 110),
                           INT8_C( -58), INT8_C(  58), INT8_C( 102), INT8_C(  64),
                           INT8_C( -67), INT8_C( -76), INT8_C( -30), INT8_C( 108),
                           INT8_C(  79), INT8_C(  46), INT8_C( -40), INT8_C( 101),
                           INT8_C( -13), INT8_C( -25), INT8_C(  60), INT8_C(  25),
                           INT8_C(  32), INT8_C( -21), INT8_C( 114), INT8_C( -21),
                           INT8_C(  71), INT8_C( -85), INT8_C(  34), INT8_C(  82),
                           INT8_C(-114), INT8_C( -30), INT8_C( -58), INT8_C( 116),
                           INT8_C(  58), INT8_C(-105), INT8_C( 117), INT8_C(  11),
                           INT8_C( -91), INT8_C( 118), INT8_C( -50), INT8_C(  -8),
                           INT8_C( -22), INT8_C(  59), INT8_C( -29), INT8_C( -88),
                           INT8_C( -82), INT8_C( -24), INT8_C(  18), INT8_C( 115),
                           INT8_C( -15), INT8_C(  55), INT8_C(  78), INT8_C(  60),
                           INT8_C(  -8), INT8_C( -91), INT8_C( 126), INT8_C(  15),
                           INT8_C(  23), INT8_C(   6), INT8_C( -21), INT8_C( 120)),
      easysimd_mm512_set_epi8(INT8_C(  27), INT8_C(  45), INT8_C(  71), INT8_C( -63),
                           INT8_C(  96), INT8_C(-106), INT8_C( -43), INT8_C(  10),
                           INT8_C( 104), INT8_C( -19), INT8_C(-110), INT8_C( 126),
                           INT8_C( -52), INT8_C( -56), INT8_C( -96), INT8_C( -27),
                           INT8_C(-125), INT8_C(-116), INT8_C(  25), INT8_C(  78),
                           INT8_C( -76), INT8_C( -85), INT8_C( -23), INT8_C( -19),
                           INT8_C(-106), INT8_C( 126), INT8_C(  19), INT8_C( -41),
                           INT8_C(  40), INT8_C(  78), INT8_C( -69), INT8_C(  57),
                           INT8_C(  73), INT8_C( -58), INT8_C(   3), INT8_C( -91),
                           INT8_C( -87), INT8_C(  42), INT8_C(   5), INT8_C(  -2),
                           INT8_C( 105), INT8_C( -36), INT8_C( -37), INT8_C(  11),
                           INT8_C(  32), INT8_C(  24), INT8_C(-107), INT8_C( -31),
                           INT8_C( -38), INT8_C(  -1), INT8_C(  48), INT8_C(  32),
                           INT8_C(  88), INT8_C(  81), INT8_C(-124), INT8_C(  62),
                           INT8_C( 124), INT8_C( -29), INT8_C(  -3), INT8_C(  52),
                           INT8_C(  25), INT8_C(  -4), INT8_C(  58), INT8_C( 110)) },
    { easysimd_mm512_set_epi8(INT8_C(   4), INT8_C(  97), INT8_C(  53), INT8_C( -46),
                           INT8_C(  92), INT8_C(-100), INT8_C(  47), INT8_C( 107),
                           INT8_C( -52), INT8_C(  68), INT8_C(  11), INT8_C( -16),
                           INT8_C( -66), INT8_C( -79), INT8_C( -14), INT8_C(  27),
                           INT8_C(  14), INT8_C( 125), INT8_C(  22), INT8_C( -82),
                           INT8_C(  44), INT8_C( -12), INT8_C(  94), INT8_C( -30),
                           INT8_C(  98), INT8_C( 125), INT8_C(-107), INT8_C(  37),
                           INT8_C( -66), INT8_C(  90), INT8_C(  68), INT8_C(  10),
                           INT8_C( -72), INT8_C( -10), INT8_C(-119), INT8_C(  -9),
                           INT8_C(  49), INT8_C(-107), INT8_C(  10), INT8_C(  47),
                           INT8_C(  58), INT8_C(-125), INT8_C(   4), INT8_C(  68),
                           INT8_C( -24), INT8_C( -12), INT8_C(  44), INT8_C(-128),
                           INT8_C( -52), INT8_C( -61), INT8_C( -14), INT8_C( -38),
                           INT8_C( -93), INT8_C( -34), INT8_C(  64), INT8_C( -67),
                           INT8_C(-123), INT8_C( 123), INT8_C( -93), INT8_C(  41),
                           INT8_C(  97), INT8_C(  -8), INT8_C( -86), INT8_C( -16)),
      UINT64_C(           944667126),
      easysimd_mm512_set_epi8(INT8_C( -24), INT8_C( -47), INT8_C(-119), INT8_C(   5),
                           INT8_C(  95), INT8_C(  82), INT8_C(  -3), INT8_C( -62),
                           INT8_C(-116), INT8_C( -98), INT8_C( -29), INT8_C(  77),
                           INT8_C( -38), INT8_C(-118), INT8_C( -85), INT8_C( 121),
                           INT8_C( -72), INT8_C(-111), INT8_C(  28), INT8_C( -18),
                           INT8_C(  64), INT8_C(-126), INT8_C( 122), INT8_C( -54),
                           INT8_C(  87), INT8_C( -22), INT8_C(  17), INT8_C(  50),
                           INT8_C( -83), INT8_C( -39), INT8_C(  77), INT8_C( -13),
                           INT8_C(  17), INT8_C( -66), INT8_C(-128), INT8_C(  77),
                           INT8_C( 107), INT8_C(  47), INT8_C( -68), INT8_C( -44),
                           INT8_C( -30), INT8_C( -22), INT8_C(  14), INT8_C(  26),
                           INT8_C(  59), INT8_C( 103), INT8_C( -54), INT8_C( -39),
                           INT8_C(  16), INT8_C(   5), INT8_C(  18), INT8_C(-104),
                           INT8_C(-119), INT8_C( -46), INT8_C( -92), INT8_C(  37),
                           INT8_C( -84), INT8_C(   2), INT8_C( -49), INT8_C(  99),
                           INT8_C( -79), INT8_C(  48), INT8_C(-103), INT8_C(   3)),
      easysimd_mm512_set_epi8(INT8_C( -56), INT8_C( -56), INT8_C(  57), INT8_C( -25),
                           INT8_C(  -3), INT8_C(  99), INT8_C(  -6), INT8_C(  31),
                           INT8_C( -96), INT8_C(  49), INT8_C( 110), INT8_C( -10),
                           INT8_C( -82), INT8_C(  32), INT8_C( -27), INT8_C( 112),
                           INT8_C(  84), INT8_C(  37), INT8_C( -62), INT8_C(  38),
                           INT8_C( -53), INT8_C( -97), INT8_C(  76), INT8_C(  13),
                           INT8_C(-124), INT8_C(-120), INT8_C( -86), INT8_C(  98),
                           INT8_C(  96), INT8_C(   4), INT8_C(   4), INT8_C(  94),
                           INT8_C( -41), INT8_C( -81), INT8_C( -40), INT8_C( -28),
                           INT8_C( -23), INT8_C( -59), INT8_C( -15), INT8_C( -40),
                           INT8_C( 113), INT8_C( 116), INT8_C(  41), INT8_C( -96),
                           INT8_C( -83), INT8_C(   4), INT8_C(  93), INT8_C(  28),
                           INT8_C( 114), INT8_C(  29), INT8_C( -56), INT8_C( -61),
                           INT8_C(-124), INT8_C(-107), INT8_C( -23), INT8_C( -89),
                           INT8_C(  38), INT8_C( -97), INT8_C( 109), INT8_C(  53),
                           INT8_C(-117), INT8_C(  76), INT8_C( -82), INT8_C( -65)),
      easysimd_mm512_set_epi8(INT8_C(   4), INT8_C(  97), INT8_C(  53), INT8_C( -46),
                           INT8_C(  92), INT8_C(-100), INT8_C(  47), INT8_C( 107),
                           INT8_C( -52), INT8_C(  68), INT8_C(  11), INT8_C( -16),
                           INT8_C( -66), INT8_C( -79), INT8_C( -14), INT8_C(  27),
                           INT8_C(  14), INT8_C( 125), INT8_C(  22), INT8_C( -82),
                           INT8_C(  44), INT8_C( -12), INT8_C(  94), INT8_C( -30),
                           INT8_C(  98), INT8_C( 125), INT8_C(-107), INT8_C(  37),
                           INT8_C( -66), INT8_C(  90), INT8_C(  68), INT8_C(  10),
                           INT8_C( -72), INT8_C( -10), INT8_C( -88), INT8_C( 105),
                           INT8_C(-126), INT8_C(-107), INT8_C(  10), INT8_C(  47),
                           INT8_C(  58), INT8_C( 118), INT8_C(   4), INT8_C(  68),
                           INT8_C(-114), INT8_C(  99), INT8_C( 109), INT8_C(-128),
                           INT8_C( -52), INT8_C( -24), INT8_C(  74), INT8_C( -43),
                           INT8_C(   5), INT8_C( -34), INT8_C(  64), INT8_C( 126),
                           INT8_C(-122), INT8_C(  99), INT8_C(  98), INT8_C(  46),
                           INT8_C(  97), INT8_C( -28), INT8_C( -21), INT8_C( -16)) },
    { easysimd_mm512_set_epi8(INT8_C( -50), INT8_C(   0), INT8_C(  80), INT8_C(-123),
                           INT8_C(  19), INT8_C( 112), INT8_C(  30), INT8_C(  95),
                           INT8_C(  58), INT8_C(  21), INT8_C(  13), INT8_C(  32),
                           INT8_C( 113), INT8_C( 126), INT8_C(  27), INT8_C( 113),
                           INT8_C( 121), INT8_C(  97), INT8_C(  51), INT8_C( -16),
                           INT8_C( -77), INT8_C(  84), INT8_C(  16), INT8_C(-112),
                           INT8_C( -40), INT8_C( -69), INT8_C(-116), INT8_C( -97),
                           INT8_C(-120), INT8_C( 102), INT8_C( -82), INT8_C( -42),
                           INT8_C(  43), INT8_C( -70), INT8_C(  46), INT8_C(  17),
                           INT8_C( 108), INT8_C( -47), INT8_C(  53), INT8_C( -84),
                           INT8_C(  19), INT8_C( -37), INT8_C( -32), INT8_C( -59),
                           INT8_C(  33), INT8_C( 110), INT8_C(  17), INT8_C(  67),
                           INT8_C(  51), INT8_C( -19), INT8_C(  91), INT8_C(  26),
                           INT8_C(  33), INT8_C( -43), INT8_C( -14), INT8_C( -56),
                           INT8_C( 112), INT8_C( -72), INT8_C(  96), INT8_C( -62),
                           INT8_C( -21), INT8_C(  96), INT8_C( -25), INT8_C( 104)),
      UINT64_C(          1662672283),
      easysimd_mm512_set_epi8(INT8_C(  55), INT8_C(  43), INT8_C(-128), INT8_C(  23),
                           INT8_C( -59), INT8_C( -21), INT8_C( -11), INT8_C( -65),
                           INT8_C(-101), INT8_C( -89), INT8_C( -88), INT8_C( -71),
                           INT8_C( -70), INT8_C(  37), INT8_C( 122), INT8_C(  74),
                           INT8_C( 109), INT8_C( -13), INT8_C( -13), INT8_C(  72),
                           INT8_C(  -1), INT8_C( -35), INT8_C(  80), INT8_C( -20),
                           INT8_C(  14), INT8_C(-104), INT8_C( -76), INT8_C(-122),
                           INT8_C( -35), INT8_C( -33), INT8_C(  63), INT8_C(  74),
                           INT8_C(  98), INT8_C(  54), INT8_C( -12), INT8_C(  -1),
                           INT8_C( -30), INT8_C(  96), INT8_C(  95), INT8_C(  58),
                           INT8_C( -63), INT8_C(  -6), INT8_C(-113), INT8_C(  55),
                           INT8_C(-128), INT8_C( -43), INT8_C( -90), INT8_C( -63),
                           INT8_C(   3), INT8_C(  -6), INT8_C( -45), INT8_C( -75),
                           INT8_C( -83), INT8_C(-118), INT8_C(  74), INT8_C(  35),
                           INT8_C(  38), INT8_C(   4), INT8_C(  35), INT8_C(  15),
                           INT8_C( -42), INT8_C(  71), INT8_C(  -1), INT8_C(  27)),
      easysimd_mm512_set_epi8(INT8_C(  32), INT8_C( 122), INT8_C(  89), INT8_C(  21),
                           INT8_C( -83), INT8_C( -46), INT8_C( -78), INT8_C(  71),
                           INT8_C( -35), INT8_C(  54), INT8_C( -65), INT8_C(-111),
                           INT8_C(  45), INT8_C(  -5), INT8_C( 102), INT8_C(  32),
                           INT8_C(-110), INT8_C( 116), INT8_C( -61), INT8_C(  36),
                           INT8_C( -25), INT8_C( 106), INT8_C( -63), INT8_C(  23),
                           INT8_C( -59), INT8_C(  25), INT8_C(-108), INT8_C( -84),
                           INT8_C( -23), INT8_C( 118), INT8_C( -35), INT8_C(  92),
                           INT8_C( -29), INT8_C(-121), INT8_C( -87), INT8_C(  93),
                           INT8_C(   6), INT8_C( -12), INT8_C(-123), INT8_C(  42),
                           INT8_C( 121), INT8_C(   3), INT8_C(  69), INT8_C(  75),
                           INT8_C(  68), INT8_C(  -1), INT8_C( -25), INT8_C(  83),
                           INT8_C(  -4), INT8_C( -73), INT8_C( -63), INT8_C(  12),
                           INT8_C( -93), INT8_C( -22), INT8_C(  40), INT8_C( -24),
                           INT8_C( -60), INT8_C(  99), INT8_C( 122), INT8_C(  49),
                           INT8_C( -46), INT8_C( 127), INT8_C(  18), INT8_C( 124)),
      easysimd_mm512_set_epi8(INT8_C( -50), INT8_C(   0), INT8_C(  80), INT8_C(-123),
                           INT8_C(  19), INT8_C( 112), INT8_C(  30), INT8_C(  95),
                           INT8_C(  58), INT8_C(  21), INT8_C(  13), INT8_C(  32),
                           INT8_C( 113), INT8_C( 126), INT8_C(  27), INT8_C( 113),
                           INT8_C( 121), INT8_C(  97), INT8_C(  51), INT8_C( -16),
                           INT8_C( -77), INT8_C(  84), INT8_C(  16), INT8_C(-112),
                           INT8_C( -40), INT8_C( -69), INT8_C(-116), INT8_C( -97),
                           INT8_C(-120), INT8_C( 102), INT8_C( -82), INT8_C( -42),
                           INT8_C(  43), INT8_C( -81), INT8_C(  75), INT8_C(  17),
                           INT8_C( 108), INT8_C( -47), INT8_C( -38), INT8_C(  16),
                           INT8_C(  19), INT8_C( -37), INT8_C( -32), INT8_C( -20),
                           INT8_C(  60), INT8_C( 110), INT8_C( -65), INT8_C(  67),
                           INT8_C(  51), INT8_C(  67), INT8_C(  91), INT8_C( -87),
                           INT8_C(  10), INT8_C( -96), INT8_C( -14), INT8_C(  59),
                           INT8_C(  98), INT8_C( -72), INT8_C(  96), INT8_C( -34),
                           INT8_C(   4), INT8_C(  96), INT8_C( -19), INT8_C( -97)) },
    { easysimd_mm512_set_epi8(INT8_C( -82), INT8_C(  17), INT8_C( 105), INT8_C(   8),
                           INT8_C( -41), INT8_C( 122), INT8_C( -11), INT8_C( -52),
                           INT8_C( -81), INT8_C( -30), INT8_C( 109), INT8_C( 119),
                           INT8_C( -78), INT8_C(-123), INT8_C(   5), INT8_C( -23),
                           INT8_C(  44), INT8_C( -23), INT8_C(-122), INT8_C(-101),
                           INT8_C( -30), INT8_C( 103), INT8_C(  30), INT8_C(  -6),
                           INT8_C( 113), INT8_C( -64), INT8_C(  -3), INT8_C(-100),
                           INT8_C(  72), INT8_C( -30), INT8_C(  59), INT8_C(  -7),
                           INT8_C(-101), INT8_C(  48), INT8_C( -62), INT8_C(   5),
                           INT8_C( -52), INT8_C(  72), INT8_C(  56), INT8_C(   6),
                           INT8_C(  86), INT8_C( -78), INT8_C( -43), INT8_C(  91),
                           INT8_C( -63), INT8_C( -91), INT8_C(-105), INT8_C( -98),
                           INT8_C(  39), INT8_C(   5), INT8_C(  77), INT8_C(  91),
                           INT8_C( -82), INT8_C(  20), INT8_C(  41), INT8_C(  62),
                           INT8_C(  27), INT8_C(  82), INT8_C( -39), INT8_C(  57),
                           INT8_C(-116), INT8_C( -85), INT8_C(-107), INT8_C(  31)),
      UINT64_C(           782232724),
      easysimd_mm512_set_epi8(INT8_C(  11), INT8_C(  49), INT8_C( -30), INT8_C(-117),
                           INT8_C(  85), INT8_C(  19), INT8_C(  44), INT8_C(-110),
                           INT8_C(  61), INT8_C( -27), INT8_C(  26), INT8_C( -12),
                           INT8_C( 110), INT8_C(  11), INT8_C(  45), INT8_C( -32),
                           INT8_C(  -1), INT8_C(  86), INT8_C( 125), INT8_C(  95),
                           INT8_C( -41), INT8_C( -73), INT8_C(  -6), INT8_C( 122),
                           INT8_C(  65), INT8_C( -38), INT8_C(-116), INT8_C(  84),
                           INT8_C(-121), INT8_C( -15), INT8_C(  41), INT8_C(-102),
                           INT8_C( -31), INT8_C( -83), INT8_C( -68), INT8_C(  89),
                           INT8_C(  27), INT8_C(-107), INT8_C( -85), INT8_C(  74),
                           INT8_C(  95), INT8_C( -86), INT8_C(  94), INT8_C( -13),
                           INT8_C( -84), INT8_C(  38), INT8_C( 116), INT8_C(-101),
                           INT8_C(  72), INT8_C(  32), INT8_C( -98), INT8_C(  48),
                           INT8_C( -94), INT8_C( -55), INT8_C( -17), INT8_C(  28),
                           INT8_C(  42), INT8_C(  70), INT8_C(  89), INT8_C(-115),
                           INT8_C( -86), INT8_C( 126), INT8_C( -92), INT8_C(  91)),
      easysimd_mm512_set_epi8(INT8_C( -46), INT8_C( -24), INT8_C( -24), INT8_C( -26),
                           INT8_C(  89), INT8_C( 108), INT8_C(  49), INT8_C( 123),
                           INT8_C( -86), INT8_C( -61), INT8_C( -22), INT8_C( -47),
                           INT8_C(  21), INT8_C(  76), INT8_C(   6), INT8_C( -21),
                           INT8_C( -19), INT8_C(  38), INT8_C(-116), INT8_C( -22),
                           INT8_C( -75), INT8_C(  54), INT8_C( -81), INT8_C(   9),
                           INT8_C(  94), INT8_C( -15), INT8_C(  26), INT8_C(-110),
                           INT8_C(  18), INT8_C( -49), INT8_C( -21), INT8_C(  70),
                           INT8_C(  50), INT8_C(  20), INT8_C( -59), INT8_C(  63),
                           INT8_C( -20), INT8_C( -92), INT8_C( -44), INT8_C(  37),
                           INT8_C(-125), INT8_C(   4), INT8_C(  53), INT8_C( -49),
                           INT8_C( -10), INT8_C(  11), INT8_C(  91), INT8_C( -86),
                           INT8_C( -34), INT8_C(-108), INT8_C( -80), INT8_C( 122),
                           INT8_C(  31), INT8_C(  31), INT8_C( -29), INT8_C(  70),
                           INT8_C(  28), INT8_C(  33), INT8_C( 109), INT8_C(  55),
                           INT8_C( -79), INT8_C(  95), INT8_C( 100), INT8_C( -33)),
      easysimd_mm512_set_epi8(INT8_C( -82), INT8_C(  17), INT8_C( 105), INT8_C(   8),
                           INT8_C( -41), INT8_C( 122), INT8_C( -11), INT8_C( -52),
                           INT8_C( -81), INT8_C( -30), INT8_C( 109), INT8_C( 119),
                           INT8_C( -78), INT8_C(-123), INT8_C(   5), INT8_C( -23),
                           INT8_C(  44), INT8_C( -23), INT8_C(-122), INT8_C(-101),
                           INT8_C( -30), INT8_C( 103), INT8_C(  30), INT8_C(  -6),
                           INT8_C( 113), INT8_C( -64), INT8_C(  -3), INT8_C(-100),
                           INT8_C(  72), INT8_C( -30), INT8_C(  59), INT8_C(  -7),
                           INT8_C(-101), INT8_C(  48), INT8_C(  -9), INT8_C(   5),
                           INT8_C(  47), INT8_C( -15), INT8_C( -41), INT8_C(   6),
                           INT8_C( -36), INT8_C( -78), INT8_C( -43), INT8_C(  36),
                           INT8_C( -74), INT8_C(  27), INT8_C(  25), INT8_C( -15),
                           INT8_C( 106), INT8_C(-116), INT8_C( -18), INT8_C(  91),
                           INT8_C(-125), INT8_C( -86), INT8_C(  41), INT8_C(  62),
                           INT8_C(  14), INT8_C(  82), INT8_C( -39), INT8_C(  86),
                           INT8_C(-116), INT8_C(  31), INT8_C(-107), INT8_C(  31)) },
    { easysimd_mm512_set_epi8(INT8_C(  82), INT8_C( -55), INT8_C(  13), INT8_C(-104),
                           INT8_C(  62), INT8_C(  20), INT8_C( -36), INT8_C(  92),
                           INT8_C( -73), INT8_C( -79), INT8_C(  -7), INT8_C( -22),
                           INT8_C( -50), INT8_C(-119), INT8_C( -83), INT8_C( -71),
                           INT8_C( 125), INT8_C(  29), INT8_C( -61), INT8_C(-111),
                           INT8_C(  -9), INT8_C(  67), INT8_C( -39), INT8_C( -17),
                           INT8_C(  23), INT8_C( -11), INT8_C(-122), INT8_C( -24),
                           INT8_C(  37), INT8_C(-122), INT8_C( -16), INT8_C( -40),
                           INT8_C( -34), INT8_C( -17), INT8_C( 100), INT8_C( 120),
                           INT8_C( -51), INT8_C(   8), INT8_C(  82), INT8_C(  19),
                           INT8_C( -50), INT8_C( -24), INT8_C( -20), INT8_C( -32),
                           INT8_C(  74), INT8_C( -84), INT8_C(   9), INT8_C(  14),
                           INT8_C(-102), INT8_C(  -2), INT8_C( 106), INT8_C(  41),
                           INT8_C(  98), INT8_C( -87), INT8_C(-124), INT8_C(  -3),
                           INT8_C(  80), INT8_C( 110), INT8_C( -32), INT8_C(  20),
                           INT8_C( -15), INT8_C(  65), INT8_C( -54), INT8_C( -49)),
      UINT64_C(          1883474426),
      easysimd_mm512_set_epi8(INT8_C( -36), INT8_C(-125), INT8_C( -71), INT8_C(-101),
                           INT8_C( -95), INT8_C(  -1), INT8_C(  65), INT8_C(  67),
                           INT8_C(   4), INT8_C( 126), INT8_C(  -9), INT8_C(  50),
                           INT8_C(  46), INT8_C(  17), INT8_C(  12), INT8_C(   7),
                           INT8_C(  31), INT8_C( -83), INT8_C(  63), INT8_C(  21),
                           INT8_C(-105), INT8_C(  56), INT8_C(   6), INT8_C(  88),
                           INT8_C(  -4), INT8_C( -51), INT8_C( -16), INT8_C( -27),
                           INT8_C( -26), INT8_C(  50), INT8_C(   3), INT8_C( -65),
                           INT8_C(  -3), INT8_C( -86), INT8_C(  39), INT8_C(  48),
                           INT8_C(  65), INT8_C(  36), INT8_C( -65), INT8_C( -82),
                           INT8_C(-107), INT8_C(  14), INT8_C( 110), INT8_C(  56),
                           INT8_C( 111), INT8_C( -32), INT8_C( 109), INT8_C( -95),
                           INT8_C(  69), INT8_C(-111), INT8_C(  -7), INT8_C(   9),
                           INT8_C( 116), INT8_C(  77), INT8_C( 122), INT8_C(  26),
                           INT8_C(   9), INT8_C( -79), INT8_C(-127), INT8_C(  -5),
                           INT8_C( 117), INT8_C(   5), INT8_C(  80), INT8_C( -85)),
      easysimd_mm512_set_epi8(INT8_C(  82), INT8_C(-125), INT8_C( -33), INT8_C(  83),
                           INT8_C( -98), INT8_C(  50), INT8_C( -27), INT8_C( -16),
                           INT8_C( -63), INT8_C(-111), INT8_C( -65), INT8_C(   3),
                           INT8_C( -15), INT8_C(  37), INT8_C(  46), INT8_C(  58),
                           INT8_C( -41), INT8_C(  72), INT8_C( 108), INT8_C(-124),
                           INT8_C(   9), INT8_C(  40), INT8_C( 115), INT8_C(  12),
                           INT8_C(   1), INT8_C(  41), INT8_C( -71), INT8_C(  87),
                           INT8_C( -55), INT8_C(  52), INT8_C( -97), INT8_C(  49),
                           INT8_C(  32), INT8_C(-115), INT8_C(  71), INT8_C(  64),
                           INT8_C( -61), INT8_C(  43), INT8_C( -42), INT8_C(  57),
                           INT8_C( -56), INT8_C( 113), INT8_C(  60), INT8_C(  75),
                           INT8_C(   7), INT8_C( -47), INT8_C(   4), INT8_C( 115),
                           INT8_C(  67), INT8_C(  44), INT8_C(  -1), INT8_C( -85),
                           INT8_C( -95), INT8_C( 108), INT8_C(  37), INT8_C( -99),
                           INT8_C( -88), INT8_C( -11), INT8_C(  47), INT8_C( -69),
                           INT8_C(-123), INT8_C(  17), INT8_C( -30), INT8_C(  36)),
      easysimd_mm512_set_epi8(INT8_C(  82), INT8_C( -55), INT8_C(  13), INT8_C(-104),
                           INT8_C(  62), INT8_C(  20), INT8_C( -36), INT8_C(  92),
                           INT8_C( -73), INT8_C( -79), INT8_C(  -7), INT8_C( -22),
                           INT8_C( -50), INT8_C(-119), INT8_C( -83), INT8_C( -71),
                           INT8_C( 125), INT8_C(  29), INT8_C( -61), INT8_C(-111),
                           INT8_C(  -9), INT8_C(  67), INT8_C( -39), INT8_C( -17),
                           INT8_C(  23), INT8_C( -11), INT8_C(-122), INT8_C( -24),
                           INT8_C(  37), INT8_C(-122), INT8_C( -16), INT8_C( -40),
                           INT8_C( -34), INT8_C(  29), INT8_C( -32), INT8_C( -16),
                           INT8_C( -51), INT8_C(   8), INT8_C(  82), INT8_C(  19),
                           INT8_C( -50), INT8_C( -99), INT8_C( -20), INT8_C( -32),
                           INT8_C(  74), INT8_C( -84), INT8_C( 105), INT8_C(  46),
                           INT8_C(   2), INT8_C(  -2), INT8_C( 106), INT8_C(  41),
                           INT8_C( -45), INT8_C( -87), INT8_C(-124), INT8_C( 125),
                           INT8_C(  97), INT8_C( -68), INT8_C(  82), INT8_C(  64),
                           INT8_C( -16), INT8_C(  65), INT8_C( 110), INT8_C( -49)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sub_epi8(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sub_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_sub_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask64 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT64_C(          3290745653),
      easysimd_mm512_set_epi8(INT8_C(  68), INT8_C( -18), INT8_C(-120), INT8_C( -91),
                           INT8_C(  33), INT8_C( -44), INT8_C( 127), INT8_C(-128),
                           INT8_C( 121), INT8_C(  -8), INT8_C(-121), INT8_C( -19),
                           INT8_C(  19), INT8_C(  -3), INT8_C(  10), INT8_C( -37),
                           INT8_C(  96), INT8_C(  15), INT8_C( -45), INT8_C( -44),
                           INT8_C( -83), INT8_C( -37), INT8_C(   8), INT8_C( 111),
                           INT8_C( -30), INT8_C( -99), INT8_C( 116), INT8_C( 112),
                           INT8_C(  67), INT8_C(-123), INT8_C(  59), INT8_C( -62),
                           INT8_C(  33), INT8_C(  51), INT8_C( -16), INT8_C( -35),
                           INT8_C( -53), INT8_C( 108), INT8_C( -37), INT8_C( -15),
                           INT8_C(  26), INT8_C(  83), INT8_C( -47), INT8_C( -23),
                           INT8_C(   6), INT8_C(  52), INT8_C( -19), INT8_C( 108),
                           INT8_C( -33), INT8_C( 120), INT8_C(  55), INT8_C(-128),
                           INT8_C( -46), INT8_C( 117), INT8_C(  41), INT8_C( -54),
                           INT8_C( -99), INT8_C( -39), INT8_C( 117), INT8_C(  57),
                           INT8_C(  78), INT8_C(-110), INT8_C(  -8), INT8_C(-114)),
      easysimd_mm512_set_epi8(INT8_C( 115), INT8_C(  65), INT8_C(-123), INT8_C( 116),
                           INT8_C( -46), INT8_C( -68), INT8_C(  -8), INT8_C(  96),
                           INT8_C( -90), INT8_C( -14), INT8_C(  27), INT8_C(  33),
                           INT8_C(  21), INT8_C(  58), INT8_C( -12), INT8_C( -76),
                           INT8_C(  70), INT8_C( -70), INT8_C(  -4), INT8_C(  64),
                           INT8_C(  35), INT8_C( -32), INT8_C(-115), INT8_C( 109),
                           INT8_C( -69), INT8_C( -61), INT8_C(-126), INT8_C(  45),
                           INT8_C(  93), INT8_C( 100), INT8_C( -53), INT8_C( 105),
                           INT8_C( -51), INT8_C(  43), INT8_C( -96), INT8_C( -95),
                           INT8_C(  86), INT8_C( -66), INT8_C( -47), INT8_C( 123),
                           INT8_C(  63), INT8_C(  16), INT8_C( -66), INT8_C(  12),
                           INT8_C( 115), INT8_C( -39), INT8_C(  10), INT8_C( 108),
                           INT8_C(-120), INT8_C( 110), INT8_C( -25), INT8_C(   7),
                           INT8_C( 105), INT8_C( -93), INT8_C(  68), INT8_C(   3),
                           INT8_C( 113), INT8_C( -50), INT8_C( -34), INT8_C(  22),
                           INT8_C( -61), INT8_C(  75), INT8_C(  28), INT8_C( 116)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  84), INT8_C(   8), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -82), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  19), INT8_C(   0),
                           INT8_C(   0), INT8_C(  91), INT8_C(   0), INT8_C(   0),
                           INT8_C(  87), INT8_C(  10), INT8_C(   0), INT8_C(   0),
                           INT8_C( 105), INT8_C(   0), INT8_C( -27), INT8_C( -57),
                           INT8_C(   0), INT8_C(   0), INT8_C(-105), INT8_C(  35),
                           INT8_C(   0), INT8_C(  71), INT8_C(   0), INT8_C(  26)) },
    { UINT64_C(          1235205951),
      easysimd_mm512_set_epi8(INT8_C(-120), INT8_C( -25), INT8_C( 111), INT8_C(-128),
                           INT8_C(  -4), INT8_C(  58), INT8_C(-115), INT8_C(   4),
                           INT8_C(  26), INT8_C(  -6), INT8_C( -92), INT8_C(   7),
                           INT8_C(-105), INT8_C( -92), INT8_C(  38), INT8_C( -63),
                           INT8_C(  77), INT8_C(  86), INT8_C( 113), INT8_C( -48),
                           INT8_C( 108), INT8_C( -92), INT8_C(  69), INT8_C(  74),
                           INT8_C(  67), INT8_C(  96), INT8_C( -34), INT8_C(  78),
                           INT8_C( 124), INT8_C(   9), INT8_C(  -1), INT8_C( -86),
                           INT8_C( -35), INT8_C(  26), INT8_C(  67), INT8_C(  46),
                           INT8_C(  75), INT8_C(-119), INT8_C(  68), INT8_C(  31),
                           INT8_C( -52), INT8_C(-102), INT8_C(  -4), INT8_C( 118),
                           INT8_C(   0), INT8_C( -44), INT8_C( 123), INT8_C( -73),
                           INT8_C(  84), INT8_C(  30), INT8_C(  -8), INT8_C(  64),
                           INT8_C( -20), INT8_C( -79), INT8_C( -85), INT8_C( -23),
                           INT8_C( -34), INT8_C(  -4), INT8_C( -85), INT8_C( 107),
                           INT8_C(  -6), INT8_C(  16), INT8_C( -66), INT8_C(-113)),
      easysimd_mm512_set_epi8(INT8_C( -19), INT8_C( 117), INT8_C( 121), INT8_C(  67),
                           INT8_C( -91), INT8_C( -91), INT8_C(  98), INT8_C( 106),
                           INT8_C(  44), INT8_C(  68), INT8_C(  96), INT8_C(  88),
                           INT8_C(   1), INT8_C(  68), INT8_C(  46), INT8_C(  19),
                           INT8_C(  31), INT8_C(  85), INT8_C(  35), INT8_C(  68),
                           INT8_C( -79), INT8_C(  41), INT8_C(  28), INT8_C(  92),
                           INT8_C( -26), INT8_C(  20), INT8_C( -16), INT8_C(  -7),
                           INT8_C(  41), INT8_C(  71), INT8_C(  88), INT8_C(  39),
                           INT8_C( -42), INT8_C(  76), INT8_C(  40), INT8_C( 108),
                           INT8_C( -87), INT8_C(-126), INT8_C(  42), INT8_C(  58),
                           INT8_C(   2), INT8_C( -23), INT8_C(   5), INT8_C(-116),
                           INT8_C(  34), INT8_C(  66), INT8_C(  28), INT8_C(  86),
                           INT8_C(  50), INT8_C( -67), INT8_C(  20), INT8_C(  73),
                           INT8_C(  27), INT8_C( -29), INT8_C(  84), INT8_C( 112),
                           INT8_C( 104), INT8_C(  53), INT8_C( -89), INT8_C(-113),
                           INT8_C(  -4), INT8_C(  94), INT8_C(  75), INT8_C(  21)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -50), INT8_C(   0), INT8_C(   0),
                           INT8_C( -94), INT8_C(   0), INT8_C(   0), INT8_C( -27),
                           INT8_C( -54), INT8_C(   0), INT8_C(   0), INT8_C( -22),
                           INT8_C( -34), INT8_C(-110), INT8_C(  95), INT8_C(  97),
                           INT8_C(  34), INT8_C(   0), INT8_C( -28), INT8_C(  -9),
                           INT8_C( -47), INT8_C( -50), INT8_C(  87), INT8_C( 121),
                           INT8_C(   0), INT8_C(   0), INT8_C(   4), INT8_C( -36),
                           INT8_C(  -2), INT8_C( -78), INT8_C( 115), INT8_C( 122)) },
    { UINT64_C(          3694669449),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C( -45), INT8_C(  59), INT8_C(  51),
                           INT8_C( -35), INT8_C(  70), INT8_C(  73), INT8_C(  33),
                           INT8_C( -70), INT8_C( -58), INT8_C(  69), INT8_C(  -7),
                           INT8_C( 115), INT8_C( -10), INT8_C( -39), INT8_C(  78),
                           INT8_C(  56), INT8_C( 116), INT8_C( 104), INT8_C( -92),
                           INT8_C(   5), INT8_C( -47), INT8_C(  27), INT8_C(  94),
                           INT8_C(-126), INT8_C(  88), INT8_C(  80), INT8_C( 112),
                           INT8_C(  21), INT8_C(  62), INT8_C(  86), INT8_C(-103),
                           INT8_C(  66), INT8_C(  -9), INT8_C( -26), INT8_C(  47),
                           INT8_C( -50), INT8_C(-118), INT8_C( 115), INT8_C(  49),
                           INT8_C(  42), INT8_C(   6), INT8_C(  92), INT8_C(   2),
                           INT8_C(  63), INT8_C(  -6), INT8_C( -32), INT8_C(  15),
                           INT8_C(  66), INT8_C(  82), INT8_C(  -9), INT8_C( -79),
                           INT8_C(-123), INT8_C(  52), INT8_C( -90), INT8_C(-111),
                           INT8_C(  62), INT8_C( -43), INT8_C( -50), INT8_C(  62),
                           INT8_C(   4), INT8_C( -92), INT8_C(  86), INT8_C( -32)),
      easysimd_mm512_set_epi8(INT8_C(  23), INT8_C( -12), INT8_C(  67), INT8_C(  -6),
                           INT8_C( -37), INT8_C(  92), INT8_C( -35), INT8_C(  99),
                           INT8_C(  18), INT8_C(  68), INT8_C(  -9), INT8_C(  64),
                           INT8_C(-111), INT8_C(  37), INT8_C( -82), INT8_C( -95),
                           INT8_C(  54), INT8_C(  75), INT8_C( -77), INT8_C( -34),
                           INT8_C(  52), INT8_C( -80), INT8_C( -94), INT8_C(  90),
                           INT8_C(   0), INT8_C(  -8), INT8_C( 123), INT8_C(-111),
                           INT8_C(  16), INT8_C( 125), INT8_C( -51), INT8_C(  99),
                           INT8_C( -22), INT8_C( 121), INT8_C(  63), INT8_C( -55),
                           INT8_C( 117), INT8_C( 109), INT8_C(-126), INT8_C(-111),
                           INT8_C(  47), INT8_C(-127), INT8_C( 109), INT8_C(  -9),
                           INT8_C( -42), INT8_C(  36), INT8_C( -32), INT8_C( 115),
                           INT8_C( -89), INT8_C(   7), INT8_C(  90), INT8_C(  46),
                           INT8_C( -83), INT8_C( -35), INT8_C(  30), INT8_C( -19),
                           INT8_C( -99), INT8_C( -56), INT8_C( -70), INT8_C(  73),
                           INT8_C( -61), INT8_C(  27), INT8_C( 117), INT8_C(  47)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  88), INT8_C( 126), INT8_C(   0), INT8_C( 102),
                           INT8_C(  89), INT8_C(  29), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -17), INT8_C(  11),
                           INT8_C( 105), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -99), INT8_C(   0),
                           INT8_C( -40), INT8_C(  87), INT8_C(-120), INT8_C(   0),
                           INT8_C( -95), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  65), INT8_C(   0), INT8_C(   0), INT8_C( -79)) },
    { UINT64_C(          2480122489),
      easysimd_mm512_set_epi8(INT8_C(   4), INT8_C( -62), INT8_C( -44), INT8_C(  49),
                           INT8_C(  83), INT8_C( 115), INT8_C(   5), INT8_C(  57),
                           INT8_C( -76), INT8_C( -80), INT8_C(  40), INT8_C(  60),
                           INT8_C(  65), INT8_C( -98), INT8_C( -74), INT8_C(  17),
                           INT8_C( -35), INT8_C(  32), INT8_C( -43), INT8_C( 108),
                           INT8_C(  83), INT8_C( -59), INT8_C(  -4), INT8_C( 125),
                           INT8_C( -31), INT8_C( 118), INT8_C( -25), INT8_C( -91),
                           INT8_C(  50), INT8_C( -74), INT8_C(  78), INT8_C(  95),
                           INT8_C( -84), INT8_C( -63), INT8_C(  87), INT8_C(-108),
                           INT8_C(  28), INT8_C( -70), INT8_C(  77), INT8_C(-113),
                           INT8_C( -20), INT8_C(  50), INT8_C(  95), INT8_C(-108),
                           INT8_C( 105), INT8_C( 114), INT8_C(-109), INT8_C(  19),
                           INT8_C( -79), INT8_C( 106), INT8_C(  61), INT8_C( -12),
                           INT8_C( 126), INT8_C(-117), INT8_C( 126), INT8_C(-125),
                           INT8_C( -93), INT8_C(  69), INT8_C( 104), INT8_C( 119),
                           INT8_C(  63), INT8_C(  95), INT8_C(-106), INT8_C( -66)),
      easysimd_mm512_set_epi8(INT8_C( -51), INT8_C(  11), INT8_C( -44), INT8_C(  33),
                           INT8_C(  29), INT8_C(   8), INT8_C(   5), INT8_C(  70),
                           INT8_C(  13), INT8_C( 104), INT8_C(  50), INT8_C(  43),
                           INT8_C(  82), INT8_C(  -5), INT8_C( -23), INT8_C( -47),
                           INT8_C(  99), INT8_C(-116), INT8_C( 118), INT8_C(  73),
                           INT8_C( -10), INT8_C( -88), INT8_C( -42), INT8_C( -58),
                           INT8_C( -49), INT8_C(  65), INT8_C( -18), INT8_C(  54),
                           INT8_C( -68), INT8_C(   1), INT8_C(  -7), INT8_C( -96),
                           INT8_C(   4), INT8_C( 115), INT8_C(  42), INT8_C(-106),
                           INT8_C(  31), INT8_C(  94), INT8_C( -71), INT8_C( -41),
                           INT8_C(  33), INT8_C(-106), INT8_C( -65), INT8_C(-107),
                           INT8_C(  71), INT8_C( -10), INT8_C( -21), INT8_C(-128),
                           INT8_C( -23), INT8_C(  20), INT8_C(   2), INT8_C(  96),
                           INT8_C(-128), INT8_C( -51), INT8_C( -38), INT8_C(  47),
                           INT8_C( -56), INT8_C( 123), INT8_C( -20), INT8_C( -50),
                           INT8_C(  -2), INT8_C(  40), INT8_C(  24), INT8_C( -98)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -88), INT8_C(   0), INT8_C(   0), INT8_C(  -2),
                           INT8_C(   0), INT8_C(   0), INT8_C(-108), INT8_C( -72),
                           INT8_C( -53), INT8_C(-100), INT8_C(   0), INT8_C(  -1),
                           INT8_C(   0), INT8_C(   0), INT8_C( -88), INT8_C(-109),
                           INT8_C( -56), INT8_C(   0), INT8_C(  59), INT8_C(   0),
                           INT8_C(  -2), INT8_C(   0), INT8_C( -92), INT8_C(   0),
                           INT8_C(   0), INT8_C( -54), INT8_C( 124), INT8_C( -87),
                           INT8_C(  65), INT8_C(   0), INT8_C(   0), INT8_C(  32)) },
    { UINT64_C(          2822554960),
      easysimd_mm512_set_epi8(INT8_C( -38), INT8_C(  -1), INT8_C(  48), INT8_C(  32),
                           INT8_C(  88), INT8_C( -18), INT8_C( 123), INT8_C(  27),
                           INT8_C( 111), INT8_C(  27), INT8_C(  -3), INT8_C(  52),
                           INT8_C( -31), INT8_C(   2), INT8_C( -47), INT8_C(  64),
                           INT8_C(  76), INT8_C( -52), INT8_C( -96), INT8_C(  -6),
                           INT8_C(-119), INT8_C( -87), INT8_C( 102), INT8_C(   5),
                           INT8_C(  24), INT8_C( -44), INT8_C( 110), INT8_C(-113),
                           INT8_C(-116), INT8_C(  -3), INT8_C(  62), INT8_C( -87),
                           INT8_C(   7), INT8_C( -54), INT8_C( -57), INT8_C( -66),
                           INT8_C(  42), INT8_C( -82), INT8_C(  46), INT8_C( -16),
                           INT8_C(  91), INT8_C( -73), INT8_C( -20), INT8_C( -77),
                           INT8_C( -11), INT8_C(  25), INT8_C(  12), INT8_C(  76),
                           INT8_C( -58), INT8_C(   3), INT8_C(-125), INT8_C( -36),
                           INT8_C(  18), INT8_C( -40), INT8_C( 111), INT8_C( 107),
                           INT8_C(  88), INT8_C(  48), INT8_C( 113), INT8_C( -90),
                           INT8_C(-117), INT8_C( 116), INT8_C(  46), INT8_C( -70)),
      easysimd_mm512_set_epi8(INT8_C( 116), INT8_C(-120), INT8_C( -67), INT8_C( -86),
                           INT8_C(  48), INT8_C(   2), INT8_C(  37), INT8_C( -26),
                           INT8_C( -55), INT8_C(  66), INT8_C(  80), INT8_C(  -7),
                           INT8_C(  21), INT8_C(-118), INT8_C(   7), INT8_C( -49),
                           INT8_C(  27), INT8_C(  45), INT8_C(  71), INT8_C( -63),
                           INT8_C(  96), INT8_C(-106), INT8_C( -43), INT8_C(  10),
                           INT8_C( 104), INT8_C( -19), INT8_C(-110), INT8_C( 126),
                           INT8_C( -52), INT8_C( -56), INT8_C( -96), INT8_C( -27),
                           INT8_C(-125), INT8_C(-116), INT8_C(  25), INT8_C(  78),
                           INT8_C( -76), INT8_C( -85), INT8_C( -23), INT8_C( -19),
                           INT8_C(-106), INT8_C( 126), INT8_C(  19), INT8_C( -41),
                           INT8_C(  40), INT8_C(  78), INT8_C( -69), INT8_C(  57),
                           INT8_C(  73), INT8_C( -58), INT8_C(   3), INT8_C(  65),
                           INT8_C( -87), INT8_C( -37), INT8_C(   5), INT8_C(-126),
                           INT8_C(  14), INT8_C( -36), INT8_C( -37), INT8_C(  11),
                           INT8_C(  94), INT8_C(  24), INT8_C(   8), INT8_C( -31)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(-124), INT8_C(   0), INT8_C( -82), INT8_C(   0),
                           INT8_C( 118), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -39), INT8_C( -36),
                           INT8_C( -51), INT8_C( -53), INT8_C(   0), INT8_C(   0),
                           INT8_C( 125), INT8_C(  61), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  -3), INT8_C(   0), INT8_C( -23),
                           INT8_C(   0), INT8_C(  84), INT8_C(   0), INT8_C(-101),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C(          2777207418),
      easysimd_mm512_set_epi8(INT8_C( -82), INT8_C( -24), INT8_C(  18), INT8_C( 115),
                           INT8_C( -15), INT8_C(  55), INT8_C(  78), INT8_C(  60),
                           INT8_C(  -8), INT8_C( -91), INT8_C( 126), INT8_C(  15),
                           INT8_C(  23), INT8_C(   6), INT8_C( -21), INT8_C( 120),
                           INT8_C( -20), INT8_C(-104), INT8_C( -27), INT8_C(  38),
                           INT8_C(  31), INT8_C( -21), INT8_C(  79), INT8_C( -62),
                           INT8_C(  36), INT8_C(  95), INT8_C(  42), INT8_C(-102),
                           INT8_C( -80), INT8_C( -69), INT8_C( 107), INT8_C(-114),
                           INT8_C(  76), INT8_C( 123), INT8_C(-126), INT8_C( 108),
                           INT8_C( -55), INT8_C(  89), INT8_C( -46), INT8_C(  18),
                           INT8_C( 117), INT8_C(  25), INT8_C(-120), INT8_C(  27),
                           INT8_C(  34), INT8_C(  64), INT8_C(  71), INT8_C(  64),
                           INT8_C( -13), INT8_C( -73), INT8_C( 112), INT8_C(  25),
                           INT8_C( -18), INT8_C( -63), INT8_C( 109), INT8_C(   9),
                           INT8_C(  14), INT8_C(-125), INT8_C( -89), INT8_C(  70),
                           INT8_C(  10), INT8_C(  15), INT8_C( 120), INT8_C( -59)),
      easysimd_mm512_set_epi8(INT8_C( -52), INT8_C( -61), INT8_C( -14), INT8_C( -38),
                           INT8_C( -93), INT8_C( -34), INT8_C(  64), INT8_C( -67),
                           INT8_C(-123), INT8_C( 123), INT8_C( -93), INT8_C(  41),
                           INT8_C(  97), INT8_C(  -8), INT8_C( -86), INT8_C( -16),
                           INT8_C( -70), INT8_C(   3), INT8_C( 118), INT8_C(  37),
                           INT8_C( 104), INT8_C( 111), INT8_C( -17), INT8_C( 110),
                           INT8_C( -58), INT8_C(  58), INT8_C( 102), INT8_C(  64),
                           INT8_C( -67), INT8_C( -76), INT8_C( -30), INT8_C( 108),
                           INT8_C(  79), INT8_C(  46), INT8_C( -40), INT8_C( 101),
                           INT8_C( -13), INT8_C( -25), INT8_C(  60), INT8_C(  25),
                           INT8_C(  32), INT8_C( -21), INT8_C( 114), INT8_C( -21),
                           INT8_C(  71), INT8_C( -85), INT8_C(  34), INT8_C(  82),
                           INT8_C(-114), INT8_C( -30), INT8_C( -58), INT8_C( 116),
                           INT8_C(  58), INT8_C(-105), INT8_C( 117), INT8_C(  11),
                           INT8_C( -91), INT8_C( 118), INT8_C( -50), INT8_C(  -8),
                           INT8_C( -22), INT8_C(  59), INT8_C( -29), INT8_C( -88)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  -3), INT8_C(   0), INT8_C( -86), INT8_C(   0),
                           INT8_C(   0), INT8_C( 114), INT8_C(   0), INT8_C(  -7),
                           INT8_C(  85), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -37), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 101), INT8_C( -43), INT8_C(   0), INT8_C( -91),
                           INT8_C(   0), INT8_C(   0), INT8_C(  -8), INT8_C(   0),
                           INT8_C(   0), INT8_C(  13), INT8_C( -39), INT8_C(  78),
                           INT8_C(  32), INT8_C(   0), INT8_C(-107), INT8_C(   0)) },
    { UINT64_C(          3908316288),
      easysimd_mm512_set_epi8(INT8_C(  16), INT8_C(   5), INT8_C(  18), INT8_C(-104),
                           INT8_C(-119), INT8_C( -46), INT8_C( -92), INT8_C(  37),
                           INT8_C( -84), INT8_C(   2), INT8_C( -49), INT8_C(  99),
                           INT8_C( -79), INT8_C(  48), INT8_C(-103), INT8_C(   3),
                           INT8_C(  54), INT8_C( 118), INT8_C( -53), INT8_C(  24),
                           INT8_C(  56), INT8_C(  78), INT8_C( 121), INT8_C( -10),
                           INT8_C(   4), INT8_C(  97), INT8_C(  53), INT8_C( -46),
                           INT8_C(  92), INT8_C(-100), INT8_C(  47), INT8_C( 107),
                           INT8_C( -52), INT8_C(  68), INT8_C(  11), INT8_C( -16),
                           INT8_C( -66), INT8_C( -79), INT8_C( -14), INT8_C(  27),
                           INT8_C(  14), INT8_C( 125), INT8_C(  22), INT8_C( -82),
                           INT8_C(  44), INT8_C( -12), INT8_C(  94), INT8_C( -30),
                           INT8_C(  98), INT8_C( 125), INT8_C(-107), INT8_C(  37),
                           INT8_C( -66), INT8_C(  90), INT8_C(  68), INT8_C(  10),
                           INT8_C( -72), INT8_C( -10), INT8_C(-119), INT8_C(  -9),
                           INT8_C(  49), INT8_C(-107), INT8_C(  10), INT8_C(  47)),
      easysimd_mm512_set_epi8(INT8_C( 114), INT8_C(  29), INT8_C( -56), INT8_C( -61),
                           INT8_C(-124), INT8_C(-107), INT8_C( -23), INT8_C( -89),
                           INT8_C(  38), INT8_C( -97), INT8_C( 109), INT8_C(  53),
                           INT8_C(-117), INT8_C(  76), INT8_C( -82), INT8_C( -65),
                           INT8_C( -24), INT8_C( -47), INT8_C(-119), INT8_C(   5),
                           INT8_C(  95), INT8_C(  82), INT8_C(  -3), INT8_C( -62),
                           INT8_C(-116), INT8_C( -98), INT8_C( -29), INT8_C(  77),
                           INT8_C( -38), INT8_C(-118), INT8_C( -85), INT8_C( 121),
                           INT8_C( -72), INT8_C(-111), INT8_C(  28), INT8_C( -18),
                           INT8_C(  64), INT8_C(-126), INT8_C( 122), INT8_C( -54),
                           INT8_C(  87), INT8_C( -22), INT8_C(  17), INT8_C(  50),
                           INT8_C( -83), INT8_C( -39), INT8_C(  77), INT8_C( -13),
                           INT8_C(  17), INT8_C( -66), INT8_C(-128), INT8_C(  77),
                           INT8_C( 107), INT8_C(  47), INT8_C( -68), INT8_C( -44),
                           INT8_C( -30), INT8_C( -22), INT8_C(  14), INT8_C(  26),
                           INT8_C(  59), INT8_C( 103), INT8_C( -54), INT8_C( -39)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  20), INT8_C( -77), INT8_C( -17), INT8_C(   0),
                           INT8_C( 126), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -73), INT8_C(-109), INT8_C(   5), INT8_C( 124),
                           INT8_C(   0), INT8_C(  27), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  21), INT8_C(   0),
                           INT8_C(  83), INT8_C(  43), INT8_C(   0), INT8_C(   0),
                           INT8_C( -42), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C(          2902744348),
      easysimd_mm512_set_epi8(INT8_C(  19), INT8_C( -37), INT8_C( -32), INT8_C( -59),
                           INT8_C(  33), INT8_C( 110), INT8_C(  17), INT8_C(  67),
                           INT8_C(  51), INT8_C( -19), INT8_C(  91), INT8_C(  26),
                           INT8_C(  33), INT8_C( -43), INT8_C( -14), INT8_C( -56),
                           INT8_C( 112), INT8_C( -72), INT8_C(  96), INT8_C( -62),
                           INT8_C( -21), INT8_C(  96), INT8_C( -25), INT8_C( 104),
                           INT8_C( -56), INT8_C( -56), INT8_C(  57), INT8_C( -25),
                           INT8_C(  -3), INT8_C(  99), INT8_C(  -6), INT8_C(  31),
                           INT8_C( -96), INT8_C(  49), INT8_C( 110), INT8_C( -10),
                           INT8_C( -82), INT8_C(  32), INT8_C( -27), INT8_C( 112),
                           INT8_C(  84), INT8_C(  37), INT8_C( -62), INT8_C(  38),
                           INT8_C( -53), INT8_C( -97), INT8_C(  76), INT8_C(  13),
                           INT8_C(-124), INT8_C(-120), INT8_C( -86), INT8_C(  98),
                           INT8_C(  96), INT8_C(   4), INT8_C(   4), INT8_C(  94),
                           INT8_C( -41), INT8_C( -81), INT8_C( -40), INT8_C( -28),
                           INT8_C( -23), INT8_C( -59), INT8_C( -15), INT8_C( -40)),
      easysimd_mm512_set_epi8(INT8_C(   3), INT8_C(  -6), INT8_C( -45), INT8_C( -75),
                           INT8_C( -83), INT8_C(-118), INT8_C(  74), INT8_C(  35),
                           INT8_C(  38), INT8_C(   4), INT8_C(  35), INT8_C(  15),
                           INT8_C( -42), INT8_C(  71), INT8_C(  -1), INT8_C(  27),
                           INT8_C( -48), INT8_C( -77), INT8_C( 116), INT8_C(  52),
                           INT8_C(  99), INT8_C(  26), INT8_C(  93), INT8_C(-101),
                           INT8_C( -50), INT8_C(   0), INT8_C(  80), INT8_C(-123),
                           INT8_C(  19), INT8_C( 112), INT8_C(  30), INT8_C(  95),
                           INT8_C(  58), INT8_C(  21), INT8_C(  13), INT8_C(  32),
                           INT8_C( 113), INT8_C( 126), INT8_C(  27), INT8_C( 113),
                           INT8_C( 121), INT8_C(  97), INT8_C(  51), INT8_C( -16),
                           INT8_C( -77), INT8_C(  84), INT8_C(  16), INT8_C(-112),
                           INT8_C( -40), INT8_C( -69), INT8_C(-116), INT8_C( -97),
                           INT8_C(-120), INT8_C( 102), INT8_C( -82), INT8_C( -42),
                           INT8_C(  43), INT8_C( -70), INT8_C(  46), INT8_C(  17),
                           INT8_C( 108), INT8_C( -47), INT8_C(  53), INT8_C( -84)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 102), INT8_C(   0), INT8_C(  97), INT8_C(   0),
                           INT8_C(  61), INT8_C( -94), INT8_C(   0), INT8_C(  -1),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  75), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -51), INT8_C(   0), INT8_C( -61),
                           INT8_C( -40), INT8_C( -98), INT8_C(   0), INT8_C(-120),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -45),
                           INT8_C( 125), INT8_C( -12), INT8_C(   0), INT8_C(   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_sub_epi8(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_sub_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_sub_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi16(INT16_C( 21075), INT16_C( 30017), INT16_C(-11898), INT16_C( 29710),
                            INT16_C( 19457), INT16_C(-12796), INT16_C( 21427), INT16_C( 28826),
                            INT16_C( 25482), INT16_C(-11843), INT16_C( 15582), INT16_C( 20114),
                            INT16_C(-14761), INT16_C(-15590), INT16_C( -4142), INT16_C( 29932),
                            INT16_C(-30672), INT16_C(  6190), INT16_C( 26590), INT16_C( 10803),
                            INT16_C(-16554), INT16_C( 15816), INT16_C( 14967), INT16_C( 24063),
                            INT16_C(-14713), INT16_C( -8094), INT16_C(-16817), INT16_C( 25507),
                            INT16_C( 19912), INT16_C(-19929), INT16_C(-12604), INT16_C(-17156)),
      easysimd_mm512_set_epi16(INT16_C( 27175), INT16_C(-27122), INT16_C( -6914), INT16_C( 10212),
                            INT16_C( 13894), INT16_C( -4620), INT16_C(-10724), INT16_C( -3078),
                            INT16_C( 29698), INT16_C(  6009), INT16_C( 28893), INT16_C(-31734),
                            INT16_C( -3957), INT16_C(  6787), INT16_C(  9325), INT16_C(  7645),
                            INT16_C( -2133), INT16_C( -9633), INT16_C(-22525), INT16_C(  1124),
                            INT16_C( 21781), INT16_C( 17119), INT16_C(-19461), INT16_C(-32134),
                            INT16_C( -7507), INT16_C( 19092), INT16_C( 21408), INT16_C(-14444),
                            INT16_C( 22843), INT16_C( 28625), INT16_C( -2322), INT16_C(  5251)),
      easysimd_mm512_set_epi16(INT16_C( -6100), INT16_C( -8397), INT16_C( -4984), INT16_C( 19498),
                            INT16_C(  5563), INT16_C( -8176), INT16_C( 32151), INT16_C( 31904),
                            INT16_C( -4216), INT16_C(-17852), INT16_C(-13311), INT16_C(-13688),
                            INT16_C(-10804), INT16_C(-22377), INT16_C(-13467), INT16_C( 22287),
                            INT16_C(-28539), INT16_C( 15823), INT16_C(-16421), INT16_C(  9679),
                            INT16_C( 27201), INT16_C( -1303), INT16_C(-31108), INT16_C( -9339),
                            INT16_C( -7206), INT16_C(-27186), INT16_C( 27311), INT16_C(-25585),
                            INT16_C( -2931), INT16_C( 16982), INT16_C(-10282), INT16_C(-22407)) },
    { easysimd_mm512_set_epi16(INT16_C(-27508), INT16_C(  5509), INT16_C(-13526), INT16_C( 16909),
                            INT16_C(  2419), INT16_C( 22142), INT16_C( -6109), INT16_C( -1177),
                            INT16_C(  9839), INT16_C(  6329), INT16_C(  -239), INT16_C(-15885),
                            INT16_C(  3666), INT16_C( 20122), INT16_C( -1699), INT16_C(  6503),
                            INT16_C( 29169), INT16_C( -4681), INT16_C( -2713), INT16_C(-24709),
                            INT16_C(  7221), INT16_C( -3718), INT16_C(   970), INT16_C(-15558),
                            INT16_C(-11011), INT16_C(-10787), INT16_C(-29970), INT16_C(  3894),
                            INT16_C(-25914), INT16_C(-18758), INT16_C( 11824), INT16_C( -8868)),
      easysimd_mm512_set_epi16(INT16_C(  1604), INT16_C( 19874), INT16_C(-12133), INT16_C( -1966),
                            INT16_C( 13041), INT16_C(  1566), INT16_C(-11791), INT16_C( -3425),
                            INT16_C(  7377), INT16_C(-23380), INT16_C( -9249), INT16_C(-31251),
                            INT16_C( 14877), INT16_C( 24009), INT16_C(-32316), INT16_C(  8308),
                            INT16_C(-11725), INT16_C(-10230), INT16_C(  1074), INT16_C( 12341),
                            INT16_C( 19989), INT16_C( 16491), INT16_C(  4144), INT16_C(-11714),
                            INT16_C( 19285), INT16_C(-29198), INT16_C(-25258), INT16_C(-29514),
                            INT16_C(  9755), INT16_C(-29385), INT16_C(-23111), INT16_C( -3412)),
      easysimd_mm512_set_epi16(INT16_C(-29112), INT16_C(-14365), INT16_C( -1393), INT16_C( 18875),
                            INT16_C(-10622), INT16_C( 20576), INT16_C(  5682), INT16_C(  2248),
                            INT16_C(  2462), INT16_C( 29709), INT16_C(  9010), INT16_C( 15366),
                            INT16_C(-11211), INT16_C( -3887), INT16_C( 30617), INT16_C( -1805),
                            INT16_C(-24642), INT16_C(  5549), INT16_C( -3787), INT16_C( 28486),
                            INT16_C(-12768), INT16_C(-20209), INT16_C( -3174), INT16_C( -3844),
                            INT16_C(-30296), INT16_C( 18411), INT16_C( -4712), INT16_C(-32128),
                            INT16_C( 29867), INT16_C( 10627), INT16_C(-30601), INT16_C( -5456)) },
    { easysimd_mm512_set_epi16(INT16_C(   691), INT16_C( -4823), INT16_C( -3253), INT16_C(-31392),
                            INT16_C(-21784), INT16_C( -6740), INT16_C(  9130), INT16_C(-18273),
                            INT16_C( 11275), INT16_C(-27092), INT16_C(    90), INT16_C(-20133),
                            INT16_C( 30523), INT16_C( 27008), INT16_C( 28387), INT16_C( 17266),
                            INT16_C( -9777), INT16_C( 27096), INT16_C( -8328), INT16_C( -6812),
                            INT16_C(-22954), INT16_C( -4409), INT16_C( 21734), INT16_C(-19695),
                            INT16_C(-11981), INT16_C(-21195), INT16_C( 18272), INT16_C( 28327),
                            INT16_C(  7123), INT16_C(-32216), INT16_C( 24489), INT16_C(-15668)),
      easysimd_mm512_set_epi16(INT16_C(-21377), INT16_C( 15856), INT16_C(  7686), INT16_C(-28568),
                            INT16_C(-15192), INT16_C( -9747), INT16_C( 11300), INT16_C( 27000),
                            INT16_C( -6635), INT16_C(  3626), INT16_C( 12716), INT16_C(-30571),
                            INT16_C( 31697), INT16_C(  5622), INT16_C( 24444), INT16_C( -8226),
                            INT16_C( -8263), INT16_C(  2890), INT16_C( 26732), INT16_C( -8763),
                            INT16_C(-13950), INT16_C( 27415), INT16_C(  7653), INT16_C( 31511),
                            INT16_C(-21082), INT16_C(  2398), INT16_C( 23365), INT16_C(-12903),
                            INT16_C(-18221), INT16_C(  4204), INT16_C(-20453), INT16_C( 15021)),
      easysimd_mm512_set_epi16(INT16_C( 22068), INT16_C(-20679), INT16_C(-10939), INT16_C( -2824),
                            INT16_C( -6592), INT16_C(  3007), INT16_C( -2170), INT16_C( 20263),
                            INT16_C( 17910), INT16_C(-30718), INT16_C(-12626), INT16_C( 10438),
                            INT16_C( -1174), INT16_C( 21386), INT16_C(  3943), INT16_C( 25492),
                            INT16_C( -1514), INT16_C( 24206), INT16_C( 30476), INT16_C(  1951),
                            INT16_C( -9004), INT16_C(-31824), INT16_C( 14081), INT16_C( 14330),
                            INT16_C(  9101), INT16_C(-23593), INT16_C( -5093), INT16_C(-24306),
                            INT16_C( 25344), INT16_C( 29116), INT16_C(-20594), INT16_C(-30689)) },
    { easysimd_mm512_set_epi16(INT16_C(  4451), INT16_C( -3121), INT16_C( 11648), INT16_C( 14185),
                            INT16_C( -8499), INT16_C(-24679), INT16_C(-31633), INT16_C( 19019),
                            INT16_C( 26210), INT16_C(-29943), INT16_C(-18883), INT16_C( 25468),
                            INT16_C( 20366), INT16_C(  4961), INT16_C(-25468), INT16_C( -4158),
                            INT16_C(  6653), INT16_C( -1720), INT16_C(-29723), INT16_C(-14244),
                            INT16_C( -4917), INT16_C(   730), INT16_C(-20677), INT16_C( 16986),
                            INT16_C(  9316), INT16_C( 28795), INT16_C(-18273), INT16_C(-29423),
                            INT16_C(-23674), INT16_C(  7963), INT16_C( 28019), INT16_C( 13728)),
      easysimd_mm512_set_epi16(INT16_C(-10770), INT16_C( 29411), INT16_C( 30463), INT16_C( -4902),
                            INT16_C(-20392), INT16_C(-28251), INT16_C( 11448), INT16_C( 27155),
                            INT16_C(-11669), INT16_C( 11820), INT16_C(-16512), INT16_C( 10540),
                            INT16_C( 17477), INT16_C(-19759), INT16_C( 28024), INT16_C(-14431),
                            INT16_C( 24400), INT16_C( -7583), INT16_C(-12129), INT16_C( 28592),
                            INT16_C(-31057), INT16_C(-18091), INT16_C( 19926), INT16_C(-29261),
                            INT16_C(  7501), INT16_C( 16620), INT16_C(  6953), INT16_C(  3437),
                            INT16_C(  5790), INT16_C(  5348), INT16_C( 17145), INT16_C(-28791)),
      easysimd_mm512_set_epi16(INT16_C( 15221), INT16_C(-32532), INT16_C(-18815), INT16_C( 19087),
                            INT16_C( 11893), INT16_C(  3572), INT16_C( 22455), INT16_C( -8136),
                            INT16_C(-27657), INT16_C( 23773), INT16_C( -2371), INT16_C( 14928),
                            INT16_C(  2889), INT16_C( 24720), INT16_C( 12044), INT16_C( 10273),
                            INT16_C(-17747), INT16_C(  5863), INT16_C(-17594), INT16_C( 22700),
                            INT16_C( 26140), INT16_C( 18821), INT16_C( 24933), INT16_C(-19289),
                            INT16_C(  1815), INT16_C( 12175), INT16_C(-25226), INT16_C( 32676),
                            INT16_C(-29464), INT16_C(  2615), INT16_C( 10874), INT16_C(-23017)) },
    { easysimd_mm512_set_epi16(INT16_C(-31561), INT16_C( 18949), INT16_C( -2287), INT16_C(-20534),
                            INT16_C( -1057), INT16_C( -3046), INT16_C( 22138), INT16_C(-11031),
                            INT16_C(    43), INT16_C( -6266), INT16_C(-20090), INT16_C(-22393),
                            INT16_C(-26046), INT16_C(-23703), INT16_C( 28092), INT16_C(  6346),
                            INT16_C( 10308), INT16_C(   572), INT16_C(     5), INT16_C( 15306),
                            INT16_C(-19429), INT16_C( -5811), INT16_C(-27420), INT16_C(-29128),
                            INT16_C(-13676), INT16_C( -3673), INT16_C(-26157), INT16_C( 19197),
                            INT16_C(-27593), INT16_C(-20030), INT16_C(  3690), INT16_C( -3850)),
      easysimd_mm512_set_epi16(INT16_C(-11908), INT16_C( 14774), INT16_C(  5244), INT16_C( 18107),
                            INT16_C(-16396), INT16_C( 31910), INT16_C(-28865), INT16_C(-20038),
                            INT16_C(-19234), INT16_C(-15108), INT16_C(-10436), INT16_C( 19911),
                            INT16_C(  3330), INT16_C( 28633), INT16_C( 10550), INT16_C( -9358),
                            INT16_C( 23697), INT16_C( 19726), INT16_C(-26407), INT16_C(-18878),
                            INT16_C(  4326), INT16_C(-22642), INT16_C(-17402), INT16_C( 16035),
                            INT16_C( 14223), INT16_C(-15160), INT16_C( -9470), INT16_C( -3752),
                            INT16_C(  6710), INT16_C( 21116), INT16_C( -9579), INT16_C( 10253)),
      easysimd_mm512_set_epi16(INT16_C(-19653), INT16_C(  4175), INT16_C( -7531), INT16_C( 26895),
                            INT16_C( 15339), INT16_C( 30580), INT16_C(-14533), INT16_C(  9007),
                            INT16_C( 19277), INT16_C(  8842), INT16_C( -9654), INT16_C( 23232),
                            INT16_C(-29376), INT16_C( 13200), INT16_C( 17542), INT16_C( 15704),
                            INT16_C(-13389), INT16_C(-19154), INT16_C( 26412), INT16_C(-31352),
                            INT16_C(-23755), INT16_C( 16831), INT16_C(-10018), INT16_C( 20373),
                            INT16_C(-27899), INT16_C( 11487), INT16_C(-16687), INT16_C( 22949),
                            INT16_C( 31233), INT16_C( 24390), INT16_C( 13269), INT16_C(-14103)) },
    { easysimd_mm512_set_epi16(INT16_C(  1468), INT16_C( -4389), INT16_C(  1296), INT16_C(-27715),
                            INT16_C(-15620), INT16_C(  3731), INT16_C( -7289), INT16_C(-27703),
                            INT16_C(   474), INT16_C( 27447), INT16_C( -9036), INT16_C(  9176),
                            INT16_C(  2726), INT16_C(-12144), INT16_C( -2101), INT16_C( 26907),
                            INT16_C(-24700), INT16_C(  1244), INT16_C( -3927), INT16_C(-22632),
                            INT16_C( -7525), INT16_C( 17743), INT16_C( 15263), INT16_C( -3823),
                            INT16_C( 27307), INT16_C( 32391), INT16_C(-23270), INT16_C(-29301),
                            INT16_C( 23369), INT16_C(-15291), INT16_C( -5840), INT16_C( 18168)),
      easysimd_mm512_set_epi16(INT16_C( 23449), INT16_C( 17725), INT16_C(-20919), INT16_C( 31466),
                            INT16_C( 31308), INT16_C( -2183), INT16_C(-31351), INT16_C(-32386),
                            INT16_C( 26890), INT16_C(-30591), INT16_C(-12785), INT16_C(-23638),
                            INT16_C(-31955), INT16_C( -9847), INT16_C( 19108), INT16_C(-19915),
                            INT16_C(  4587), INT16_C( 27034), INT16_C(   -19), INT16_C( 28332),
                            INT16_C(-23789), INT16_C(-24960), INT16_C( -5839), INT16_C( 25722),
                            INT16_C(-24423), INT16_C( 15592), INT16_C(  6092), INT16_C( -9272),
                            INT16_C(-12796), INT16_C(-17663), INT16_C( -6154), INT16_C( 23859)),
      easysimd_mm512_set_epi16(INT16_C(-21981), INT16_C(-22114), INT16_C( 22215), INT16_C(  6355),
                            INT16_C( 18608), INT16_C(  5914), INT16_C( 24062), INT16_C(  4683),
                            INT16_C(-26416), INT16_C( -7498), INT16_C(  3749), INT16_C(-32722),
                            INT16_C(-30855), INT16_C( -2297), INT16_C(-21209), INT16_C(-18714),
                            INT16_C(-29287), INT16_C(-25790), INT16_C( -3908), INT16_C( 14572),
                            INT16_C( 16264), INT16_C(-22833), INT16_C( 21102), INT16_C(-29545),
                            INT16_C(-13806), INT16_C( 16799), INT16_C(-29362), INT16_C(-20029),
                            INT16_C(-29371), INT16_C(  2372), INT16_C(   314), INT16_C( -5691)) },
    { easysimd_mm512_set_epi16(INT16_C(-22741), INT16_C( 13394), INT16_C( -9417), INT16_C( 28906),
                            INT16_C(-18980), INT16_C( -8463), INT16_C(  9174), INT16_C(-25605),
                            INT16_C(   547), INT16_C(  3767), INT16_C(-12577), INT16_C(-16546),
                            INT16_C( -1301), INT16_C( -7147), INT16_C( 26281), INT16_C( 29309),
                            INT16_C( 29052), INT16_C(-30842), INT16_C(  5995), INT16_C(  6270),
                            INT16_C( 20539), INT16_C( 10179), INT16_C(-26848), INT16_C( 14327),
                            INT16_C( 15491), INT16_C( 18652), INT16_C( 19903), INT16_C( 30123),
                            INT16_C( 25261), INT16_C(-17460), INT16_C( 10742), INT16_C( -4552)),
      easysimd_mm512_set_epi16(INT16_C(  5754), INT16_C(-23038), INT16_C(-16589), INT16_C(-23858),
                            INT16_C( -3821), INT16_C( -4798), INT16_C( 30602), INT16_C(-28532),
                            INT16_C( 11508), INT16_C(  7979), INT16_C( -3877), INT16_C( -5920),
                            INT16_C(-24150), INT16_C(-24496), INT16_C( 17421), INT16_C( -1981),
                            INT16_C( 27523), INT16_C( 26800), INT16_C( 25010), INT16_C( 27339),
                            INT16_C( -9050), INT16_C( 19128), INT16_C( 15279), INT16_C( -1817),
                            INT16_C(-13923), INT16_C(  5129), INT16_C(-22618), INT16_C( 27704),
                            INT16_C( -4783), INT16_C( 31238), INT16_C(-30342), INT16_C( -8854)),
      easysimd_mm512_set_epi16(INT16_C(-28495), INT16_C(-29104), INT16_C(  7172), INT16_C(-12772),
                            INT16_C(-15159), INT16_C( -3665), INT16_C(-21428), INT16_C(  2927),
                            INT16_C(-10961), INT16_C( -4212), INT16_C( -8700), INT16_C(-10626),
                            INT16_C( 22849), INT16_C( 17349), INT16_C(  8860), INT16_C( 31290),
                            INT16_C(  1529), INT16_C(  7894), INT16_C(-19015), INT16_C(-21069),
                            INT16_C( 29589), INT16_C( -8949), INT16_C( 23409), INT16_C( 16144),
                            INT16_C( 29414), INT16_C( 13523), INT16_C(-23015), INT16_C(  2419),
                            INT16_C( 30044), INT16_C( 16838), INT16_C(-24452), INT16_C(  4302)) },
    { easysimd_mm512_set_epi16(INT16_C( 27021), INT16_C( 31131), INT16_C(    63), INT16_C(-10530),
                            INT16_C( -1071), INT16_C(-31284), INT16_C(-21788), INT16_C(-16108),
                            INT16_C(-15167), INT16_C( 25422), INT16_C( 14520), INT16_C(-13896),
                            INT16_C( 20399), INT16_C( 31915), INT16_C(-16518), INT16_C( -6202),
                            INT16_C(-16332), INT16_C( -3071), INT16_C(-15644), INT16_C( -7016),
                            INT16_C( 13977), INT16_C(-13846), INT16_C(-23290), INT16_C( -2079),
                            INT16_C(  4753), INT16_C( 14919), INT16_C(-18528), INT16_C(  7420),
                            INT16_C( 12098), INT16_C( 31014), INT16_C( 17813), INT16_C(-14456)),
      easysimd_mm512_set_epi16(INT16_C(-12529), INT16_C( -3643), INT16_C(-28826), INT16_C(-12110),
                            INT16_C(  8030), INT16_C( 20316), INT16_C( 27324), INT16_C( 24735),
                            INT16_C( -6774), INT16_C( -2704), INT16_C(-31930), INT16_C(  6874),
                            INT16_C( -3952), INT16_C(  2658), INT16_C(  -904), INT16_C( -8319),
                            INT16_C(-16424), INT16_C( 22778), INT16_C( 18985), INT16_C( 10063),
                            INT16_C(-31751), INT16_C( 16016), INT16_C(-30217), INT16_C( 18364),
                            INT16_C(-20176), INT16_C( -4961), INT16_C(-29576), INT16_C(-16634),
                            INT16_C( -8011), INT16_C(-27110), INT16_C(-24526), INT16_C(-11504)),
      easysimd_mm512_set_epi16(INT16_C(-25986), INT16_C(-30762), INT16_C( 28889), INT16_C(  1580),
                            INT16_C( -9101), INT16_C( 13936), INT16_C( 16424), INT16_C( 24693),
                            INT16_C( -8393), INT16_C( 28126), INT16_C(-19086), INT16_C(-20770),
                            INT16_C( 24351), INT16_C( 29257), INT16_C(-15614), INT16_C(  2117),
                            INT16_C(    92), INT16_C(-25849), INT16_C( 30907), INT16_C(-17079),
                            INT16_C(-19808), INT16_C(-29862), INT16_C(  6927), INT16_C(-20443),
                            INT16_C( 24929), INT16_C( 19880), INT16_C( 11048), INT16_C( 24054),
                            INT16_C( 20109), INT16_C( -7412), INT16_C(-23197), INT16_C( -2952)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sub_epi16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sub_epi16");
    easysimd_assert_m512i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_sub_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C(-2076524081), INT32_C( 1825078206), INT32_C(-1787857556), INT32_C(-1179707533),
                            INT32_C(  233802890), INT32_C( 1015107327), INT32_C(-1130135421), INT32_C(  769270921),
                            INT32_C(  970769619), INT32_C( -152032958), INT32_C(-1037455861), INT32_C( 1543352525),
                            INT32_C( 1997985923), INT32_C( 1878044503), INT32_C(   49641854), INT32_C(   78691943)),
      easysimd_mm512_set_epi32(INT32_C( 1273959589), INT32_C(  730948807), INT32_C(  152082522), INT32_C(  516109144),
                            INT32_C( -608654122), INT32_C(  326583665), INT32_C(-2143544685), INT32_C( 2015525957),
                            INT32_C(-1762782050), INT32_C( 1423018518), INT32_C(-1835751490), INT32_C(-1847524510),
                            INT32_C( 1152317453), INT32_C( -732966175), INT32_C(  916913335), INT32_C(-1961618071)),
      easysimd_mm512_set_epi32(INT32_C(  944483626), INT32_C( 1094129399), INT32_C(-1939940078), INT32_C(-1695816677),
                            INT32_C(  842457012), INT32_C(  688523662), INT32_C( 1013409264), INT32_C(-1246255036),
                            INT32_C(-1561415627), INT32_C(-1575051476), INT32_C(  798295629), INT32_C( -904090261),
                            INT32_C(  845668470), INT32_C(-1683956618), INT32_C( -867271481), INT32_C( 2040310014)) },
    { easysimd_mm512_set_epi32(INT32_C( 1516029066), INT32_C( 1696213023), INT32_C(  690963136), INT32_C( -395017807),
                            INT32_C(-1227102652), INT32_C( 1731549524), INT32_C( 1416885076), INT32_C( -891143280),
                            INT32_C(-1187279454), INT32_C(  699906112), INT32_C(  947982370), INT32_C(-1809113234),
                            INT32_C(  892884346), INT32_C(  173663466), INT32_C( -426903082), INT32_C(-1178201759)),
      easysimd_mm512_set_epi32(INT32_C(  565568434), INT32_C( 1477571639), INT32_C( -962268135), INT32_C(   93801511),
                            INT32_C( 1166085377), INT32_C(-1613873583), INT32_C( 1514751666), INT32_C(   -9777248),
                            INT32_C(  880861168), INT32_C(-1847118927), INT32_C( -454330268), INT32_C( -465889797),
                            INT32_C(  607148382), INT32_C( -892911578), INT32_C(-1830027716), INT32_C(  171620514)),
      easysimd_mm512_set_epi32(INT32_C(  950460632), INT32_C(  218641384), INT32_C( 1653231271), INT32_C( -488819318),
                            INT32_C( 1901779267), INT32_C( -949544189), INT32_C(  -97866590), INT32_C( -881366032),
                            INT32_C(-2068140622), INT32_C(-1747942257), INT32_C( 1402312638), INT32_C(-1343223437),
                            INT32_C(  285735964), INT32_C( 1066575044), INT32_C( 1403124634), INT32_C(-1349822273)) },
    { easysimd_mm512_set_epi32(INT32_C( -894737208), INT32_C( -894707310), INT32_C(-1734937643), INT32_C(-1821919338),
                            INT32_C(-1629473200), INT32_C( 1017176222), INT32_C(  555630880), INT32_C( 1893052174),
                            INT32_C( -395602197), INT32_C(  851153269), INT32_C( 1448617638), INT32_C( 1939202047),
                            INT32_C(-1165352739), INT32_C(  784136789), INT32_C(-1222569677), INT32_C(-1663359991)),
      easysimd_mm512_set_epi32(INT32_C(-2134962383), INT32_C(  711344265), INT32_C( -499544380), INT32_C(  658556967),
                            INT32_C(-1607446648), INT32_C(-2074003952), INT32_C(  449264495), INT32_C( -469125832),
                            INT32_C(-1465796532), INT32_C( -575249454), INT32_C( -236269065), INT32_C(  567769266),
                            INT32_C( -145854210), INT32_C(  502784491), INT32_C( -258238741), INT32_C( 1554234017)),
      easysimd_mm512_set_epi32(INT32_C( 1240225175), INT32_C(-1606051575), INT32_C(-1235393263), INT32_C( 1814490991),
                            INT32_C(  -22026552), INT32_C(-1203787122), INT32_C(  106366385), INT32_C(-1932789290),
                            INT32_C( 1070194335), INT32_C( 1426402723), INT32_C( 1684886703), INT32_C( 1371432781),
                            INT32_C(-1019498529), INT32_C(  281352298), INT32_C( -964330936), INT32_C( 1077373288)) },
    { easysimd_mm512_set_epi32(INT32_C( -658606825), INT32_C(-1465142546), INT32_C(-1613315081), INT32_C( 1981327993),
                            INT32_C( -540883338), INT32_C(  -52568431), INT32_C(  513288938), INT32_C(-1741957410),
                            INT32_C( -457290370), INT32_C(  949496535), INT32_C( -574503672), INT32_C( -516003313),
                            INT32_C( 1705152287), INT32_C(  268459282), INT32_C( -796672854), INT32_C(-2124069536)),
      easysimd_mm512_set_epi32(INT32_C(-1627464574), INT32_C(  688417349), INT32_C(-1204757032), INT32_C(-1541532775),
                            INT32_C( -489028243), INT32_C(  -14341503), INT32_C( 1546753292), INT32_C( -383774267),
                            INT32_C( 1479759913), INT32_C(-1792003336), INT32_C(  324281321), INT32_C(-1031805126),
                            INT32_C(-1668912025), INT32_C( -271675366), INT32_C(-1502890080), INT32_C( -582208760)),
      easysimd_mm512_set_epi32(INT32_C(  968857749), INT32_C( 2141407401), INT32_C( -408558049), INT32_C( -772106528),
                            INT32_C(  -51855095), INT32_C(  -38226928), INT32_C(-1033464354), INT32_C(-1358183143),
                            INT32_C(-1937050283), INT32_C(-1553467425), INT32_C( -898784993), INT32_C(  515801813),
                            INT32_C( -920902984), INT32_C(  540134648), INT32_C(  706217226), INT32_C(-1541860776)) },
    { easysimd_mm512_set_epi32(INT32_C( 1656401797), INT32_C(   50049750), INT32_C( -488722048), INT32_C( 1532620410),
                            INT32_C(  761833085), INT32_C(  -28253750), INT32_C( 1071891913), INT32_C( -578065038),
                            INT32_C( 2114869114), INT32_C( 1114386003), INT32_C( -192755303), INT32_C( -163390023),
                            INT32_C(-1012186074), INT32_C( -258665152), INT32_C(  548389384), INT32_C( -601025611)),
      easysimd_mm512_set_epi32(INT32_C( -962813354), INT32_C(-1563683363), INT32_C( 1476422960), INT32_C(-1996230234),
                            INT32_C(  594356694), INT32_C(  -37573818), INT32_C( 2109710080), INT32_C(-2049942476),
                            INT32_C(-1449482441), INT32_C(-1892730921), INT32_C( 1298337068), INT32_C(   30251788),
                            INT32_C( -250852108), INT32_C(-2130168940), INT32_C(  414197854), INT32_C( -971416192)),
      easysimd_mm512_set_epi32(INT32_C(-1675752145), INT32_C( 1613733113), INT32_C(-1965145008), INT32_C( -766116652),
                            INT32_C(  167476391), INT32_C(    9320068), INT32_C(-1037818167), INT32_C( 1471877438),
                            INT32_C( -730615741), INT32_C(-1287850372), INT32_C(-1491092371), INT32_C( -193641811),
                            INT32_C( -761333966), INT32_C( 1871503788), INT32_C(  134191530), INT32_C(  370390581)) },
    { easysimd_mm512_set_epi32(INT32_C(  841332080), INT32_C(  332746710), INT32_C( 1180202036), INT32_C(-1365461084),
                            INT32_C( -972107726), INT32_C( -919074620), INT32_C(  336794208), INT32_C(-2145769013),
                            INT32_C(-1090767268), INT32_C( 1447456701), INT32_C(-1878509449), INT32_C( 1479468832),
                            INT32_C(-2038652659), INT32_C( -428110707), INT32_C( -605535334), INT32_C( 1876977582)),
      easysimd_mm512_set_epi32(INT32_C(-1104919125), INT32_C(-1965384352), INT32_C( 1846340148), INT32_C( 1439724559),
                            INT32_C( 1174009148), INT32_C( -500908704), INT32_C( 2074430235), INT32_C(  746110301),
                            INT32_C( -229497465), INT32_C(  567264435), INT32_C(-1820479715), INT32_C( -409682629),
                            INT32_C(-1976550605), INT32_C(-1717329929), INT32_C(  392593328), INT32_C(  809330056)),
      easysimd_mm512_set_epi32(INT32_C( 1946251205), INT32_C(-1996836234), INT32_C( -666138112), INT32_C( 1489781653),
                            INT32_C(-2146116874), INT32_C( -418165916), INT32_C(-1737636027), INT32_C( 1403087982),
                            INT32_C( -861269803), INT32_C(  880192266), INT32_C(  -58029734), INT32_C( 1889151461),
                            INT32_C(  -62102054), INT32_C( 1289219222), INT32_C( -998128662), INT32_C( 1067647526)) },
    { easysimd_mm512_set_epi32(INT32_C(-1188475624), INT32_C(-1471681451), INT32_C( -219755555), INT32_C(-1657771963),
                            INT32_C( -257604504), INT32_C(  874981434), INT32_C(-1610485047), INT32_C(-1272947332),
                            INT32_C( 1561476022), INT32_C(  375243187), INT32_C( 1479356717), INT32_C( 1523794483),
                            INT32_C(-1698967593), INT32_C(  -80864233), INT32_C( 1644091986), INT32_C( -229623607)),
      easysimd_mm512_set_epi32(INT32_C(    9741774), INT32_C(  693305140), INT32_C(-1221395242), INT32_C(-1923328842),
                            INT32_C(   85084148), INT32_C( 1125599333), INT32_C( 2042080920), INT32_C( -456911551),
                            INT32_C( -399701639), INT32_C(-1860388051), INT32_C( -699039468), INT32_C(   84523143),
                            INT32_C(-1293034841), INT32_C(-1626054083), INT32_C(   96950550), INT32_C( 1663457642)),
      easysimd_mm512_set_epi32(INT32_C(-1198217398), INT32_C( 2129980705), INT32_C( 1001639687), INT32_C(  265556879),
                            INT32_C( -342688652), INT32_C( -250617899), INT32_C(  642401329), INT32_C( -816035781),
                            INT32_C( 1961177661), INT32_C(-2059336058), INT32_C(-2116571111), INT32_C( 1439271340),
                            INT32_C( -405932752), INT32_C( 1545189850), INT32_C( 1547141436), INT32_C(-1893081249)) },
    { easysimd_mm512_set_epi32(INT32_C(-1473946007), INT32_C(  121708864), INT32_C( 1020809582), INT32_C( 1669312470),
                            INT32_C( -682688365), INT32_C(  500732292), INT32_C( 1673154382), INT32_C(-1552445241),
                            INT32_C( 2068495467), INT32_C(-2039438173), INT32_C(  869593130), INT32_C( -471794528),
                            INT32_C(-1539319849), INT32_C( 1041904784), INT32_C( -120989465), INT32_C(-1180697219)),
      easysimd_mm512_set_epi32(INT32_C( 1388066655), INT32_C( 1341381019), INT32_C(-1738591736), INT32_C( -783428109),
                            INT32_C(-1884288937), INT32_C( 1340467391), INT32_C(-1349575878), INT32_C(-1411283384),
                            INT32_C( 1173507492), INT32_C( 1805408001), INT32_C( 1184512890), INT32_C(-1180223583),
                            INT32_C( -121255394), INT32_C(-2007254522), INT32_C(  970045213), INT32_C(-2132245994)),
      easysimd_mm512_set_epi32(INT32_C( 1432954634), INT32_C(-1219672155), INT32_C(-1535565978), INT32_C(-1842226717),
                            INT32_C( 1201600572), INT32_C( -839735099), INT32_C(-1272237036), INT32_C( -141161857),
                            INT32_C(  894987975), INT32_C(  450121122), INT32_C( -314919760), INT32_C(  708429055),
                            INT32_C(-1418064455), INT32_C(-1245807990), INT32_C(-1091034678), INT32_C(  951548775)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sub_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sub_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_sub_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C(-8918603015426376770), INT64_C(-7678789729811228813),
                            INT64_C( 1004175767275392767), INT64_C(-4853894672476920695),
                            INT64_C( 4169423769698314562), INT64_C(-4455838992495169331),
                            INT64_C( 8581284199031418711), INT64_C(  213210139521498727)),
      easysimd_mm512_set_epi64(INT64_C( 5471614771911550151), INT64_C(  653189458799309656),
                            INT64_C(-2614149548239010447), INT64_C(-9206454317574095803),
                            INT64_C(-7571091253302818282), INT64_C(-7884492610685828254),
                            INT64_C( 4949165778807018209), INT64_C( 3938112789424641385)),
      easysimd_mm512_set_epi64(INT64_C( 4056526286371624695), INT64_C(-8331979188610538469),
                            INT64_C( 3618325315514403214), INT64_C( 4352559645097175108),
                            INT64_C(-6706229050708418772), INT64_C( 3428653618190658923),
                            INT64_C( 3632118420224400502), INT64_C(-3724902649903142658)) },
    { easysimd_mm512_set_epi64(INT64_C( 6511295259951638559), INT64_C( 2967664075761549745),
                            INT64_C(-5270365757443319468), INT64_C( 6085475067014298512),
                            INT64_C(-5099326425442830272), INT64_C( 4071553278820425582),
                            INT64_C( 3834909065354011882), INT64_C(-1833534772634840735)),
      easysimd_mm512_set_epi64(INT64_C( 2429097929157506103), INT64_C(-4132910169714111449),
                            INT64_C( 5008298561239924305), INT64_C( 6505808871316705184),
                            INT64_C( 3783269911324210097), INT64_C(-1951333638813837829),
                            INT64_C( 2607682447911370790), INT64_C(-7859909190821955422)),
      easysimd_mm512_set_epi64(INT64_C( 4082197330794132456), INT64_C( 7100574245475661194),
                            INT64_C( 8168079755026307843), INT64_C( -420333804302406672),
                            INT64_C(-8882596336767040369), INT64_C( 6022886917634263411),
                            INT64_C( 1227226617442641092), INT64_C( 6026374418187114687)) },
    { easysimd_mm512_set_epi64(INT64_C(-3842867043474089582), INT64_C(-7451500434811275370),
                            INT64_C(-6998534102691290978), INT64_C( 2386416460140752654),
                            INT64_C(-1699098497489596043), INT64_C( 6221765381557968895),
                            INT64_C(-5005151901524886955), INT64_C(-5250896777164676087)),
      easysimd_mm512_set_epi64(INT64_C(-9169593612463882103), INT64_C(-2145526774342039513),
                            INT64_C(-6903930781003860464), INT64_C( 1929576317104796984),
                            INT64_C(-6295548163810499630), INT64_C(-1014767906663728974),
                            INT64_C( -626439061431131669), INT64_C(-1109126945600980319)),
      easysimd_mm512_set_epi64(INT64_C( 5326726568989792521), INT64_C(-5305973660469235857),
                            INT64_C(  -94603321687430514), INT64_C(  456840143035955670),
                            INT64_C( 4596449666320903587), INT64_C( 7236533288221697869),
                            INT64_C(-4378712840093755286), INT64_C(-4141769831563695768)) },
    { easysimd_mm512_set_epi64(INT64_C(-2828694771467570450), INT64_C(-6929135509057262983),
                            INT64_C(-2323076243418915183), INT64_C( 2204559204661581534),
                            INT64_C(-1964047182976242985), INT64_C(-2467474478892946929),
                            INT64_C( 7323573307633065234), INT64_C(-3421683851370085024)),
      easysimd_mm512_set_epi64(INT64_C(-6989907120040154555), INT64_C(-5174392049312590951),
                            INT64_C(-2100360306224715135), INT64_C( 6643254808031531461),
                            INT64_C( 6355520434769769208), INT64_C( 1392777671661840186),
                            INT64_C(-7167922563252842470), INT64_C(-6454863739370065144)),
      easysimd_mm512_set_epi64(INT64_C( 4161212348572584105), INT64_C(-1754743459744672032),
                            INT64_C( -222715937194200048), INT64_C(-4438695603369949927),
                            INT64_C(-8319567617746012193), INT64_C(-3860252150554787115),
                            INT64_C(-3955248202823643912), INT64_C( 3033179887999980120)) },
    { easysimd_mm512_set_epi64(INT64_C( 7114191547200680662), INT64_C(-2099045211461521798),
                            INT64_C( 3272048189352501706), INT64_C( 4603740714898779506),
                            INT64_C( 9083293681064881747), INT64_C( -827877718383993415),
                            INT64_C(-4347306081260333760), INT64_C( 2355314473447527349)),
      easysimd_mm512_set_epi64(INT64_C(-4135251864850786851), INT64_C( 6341188330562253222),
                            INT64_C( 2552742567146072902), INT64_C( 9061135799886568500),
                            INT64_C(-6225479677819013161), INT64_C( 5576315246274779916),
                            INT64_C(-1077401597827861612), INT64_C( 1778966240326933888)),
      easysimd_mm512_set_epi64(INT64_C(-7197300661658084103), INT64_C(-8440233542023775020),
                            INT64_C(  719305622206428804), INT64_C(-4457395084987788994),
                            INT64_C(-3137970714825656708), INT64_C(-6404192964658773331),
                            INT64_C(-3269904483432472148), INT64_C(  576348233120593461)) },
    { easysimd_mm512_set_epi64(INT64_C( 3613493769008402390), INT64_C( 5068929150222120868),
                            INT64_C(-4175170887983036220), INT64_C( 1446520110991419851),
                            INT64_C(-4684809742159810627), INT64_C(-8068136647202511072),
                            INT64_C(-8755946494441583475), INT64_C(-2600754454225459282)),
      easysimd_mm512_set_epi64(INT64_C(-4745591504270353056), INT64_C( 7929970554391524367),
                            INT64_C( 5042330899658882400), INT64_C( 8909610017904704861),
                            INT64_C( -985684106122640205), INT64_C(-7818900835071115973),
                            INT64_C(-8489220204786376713), INT64_C( 1686175505197131144)),
      easysimd_mm512_set_epi64(INT64_C( 8359085273278755446), INT64_C(-2861041404169403499),
                            INT64_C(-9217501787641918620), INT64_C(-7463089906913285010),
                            INT64_C(-3699125636037170422), INT64_C( -249235812131395099),
                            INT64_C( -266726289655206762), INT64_C(-4286929959422590426)) },
    { easysimd_mm512_set_epi64(INT64_C(-5104463934349906859), INT64_C( -943842919202133947),
                            INT64_C(-1106402919107319750), INT64_C(-6916980604540002948),
                            INT64_C( 6706488448353419699), INT64_C( 6353788720156721715),
                            INT64_C(-7297010244684735465), INT64_C( 7061321315551033545)),
      easysimd_mm512_set_epi64(INT64_C(   41840601428328244), INT64_C(-5245852617508367178),
                            INT64_C(  365433634193623141), INT64_C( 8770670771023648065),
                            INT64_C(-1716705465228018899), INT64_C(-3002351653588715385),
                            INT64_C(-5553542352014646723), INT64_C(  416399443242670442)),
      easysimd_mm512_set_epi64(INT64_C(-5146304535778235103), INT64_C( 4302009698306233231),
                            INT64_C(-1471836553300942891), INT64_C( 2759092698145900603),
                            INT64_C( 8423193913581438598), INT64_C(-9090603699964114516),
                            INT64_C(-1743467892670088742), INT64_C( 6644921872308363103)) },
    { easysimd_mm512_set_epi64(INT64_C(-6330549896013078208), INT64_C( 4384343771802742742),
                            INT64_C(-2932124200533978748), INT64_C( 7186143354591613127),
                            INT64_C( 8884120384944776355), INT64_C( 3734874057999449248),
                            INT64_C(-6611328408496753520), INT64_C( -519645792221266563)),
      easysimd_mm512_set_epi64(INT64_C( 5961700889234495899), INT64_C(-7467194643704326669),
                            INT64_C(-8092959359289136961), INT64_C(-5796384256596801976),
                            INT64_C( 5040176301556389633), INT64_C( 5087444127355189153),
                            INT64_C( -520787949405881850), INT64_C( 4166312467639075350)),
      easysimd_mm512_set_epi64(INT64_C( 6154493288461977509), INT64_C(-6595205658202482205),
                            INT64_C( 5160835158755158213), INT64_C(-5464216462521136513),
                            INT64_C( 3843944083388386722), INT64_C(-1352570069355739905),
                            INT64_C(-6090540459090871670), INT64_C(-4685958259860341913)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sub_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sub_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_sub_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -659.63), EASYSIMD_FLOAT32_C(  -759.67), EASYSIMD_FLOAT32_C(  -847.92), EASYSIMD_FLOAT32_C(   -61.45),
                         EASYSIMD_FLOAT32_C(  -337.36), EASYSIMD_FLOAT32_C(   139.68), EASYSIMD_FLOAT32_C(   658.69), EASYSIMD_FLOAT32_C(    86.55),
                         EASYSIMD_FLOAT32_C(  -150.13), EASYSIMD_FLOAT32_C(   450.66), EASYSIMD_FLOAT32_C(  -527.30), EASYSIMD_FLOAT32_C(  -641.78),
                         EASYSIMD_FLOAT32_C(   929.20), EASYSIMD_FLOAT32_C(  -281.32), EASYSIMD_FLOAT32_C(  -125.47), EASYSIMD_FLOAT32_C(  -963.36)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -406.77), EASYSIMD_FLOAT32_C(  -929.18), EASYSIMD_FLOAT32_C(   716.57), EASYSIMD_FLOAT32_C(     1.83),
                         EASYSIMD_FLOAT32_C(   179.14), EASYSIMD_FLOAT32_C(   145.16), EASYSIMD_FLOAT32_C(  -463.41), EASYSIMD_FLOAT32_C(  -573.03),
                         EASYSIMD_FLOAT32_C(    33.04), EASYSIMD_FLOAT32_C(   167.46), EASYSIMD_FLOAT32_C(  -891.13), EASYSIMD_FLOAT32_C(   473.74),
                         EASYSIMD_FLOAT32_C(  -547.95), EASYSIMD_FLOAT32_C(   516.90), EASYSIMD_FLOAT32_C(   -69.62), EASYSIMD_FLOAT32_C(  -976.88)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -252.86), EASYSIMD_FLOAT32_C(   169.51), EASYSIMD_FLOAT32_C( -1564.49), EASYSIMD_FLOAT32_C(   -63.28),
                         EASYSIMD_FLOAT32_C(  -516.50), EASYSIMD_FLOAT32_C(    -5.48), EASYSIMD_FLOAT32_C(  1122.10), EASYSIMD_FLOAT32_C(   659.58),
                         EASYSIMD_FLOAT32_C(  -183.17), EASYSIMD_FLOAT32_C(   283.20), EASYSIMD_FLOAT32_C(   363.83), EASYSIMD_FLOAT32_C( -1115.52),
                         EASYSIMD_FLOAT32_C(  1477.15), EASYSIMD_FLOAT32_C(  -798.22), EASYSIMD_FLOAT32_C(   -55.85), EASYSIMD_FLOAT32_C(    13.52)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -311.95), EASYSIMD_FLOAT32_C(  -956.32), EASYSIMD_FLOAT32_C(   248.48), EASYSIMD_FLOAT32_C(   995.45),
                         EASYSIMD_FLOAT32_C(   139.87), EASYSIMD_FLOAT32_C(   783.05), EASYSIMD_FLOAT32_C(   584.21), EASYSIMD_FLOAT32_C(  -920.08),
                         EASYSIMD_FLOAT32_C(  -210.14), EASYSIMD_FLOAT32_C(   816.06), EASYSIMD_FLOAT32_C(  -193.68), EASYSIMD_FLOAT32_C(   585.03),
                         EASYSIMD_FLOAT32_C(  -674.08), EASYSIMD_FLOAT32_C(   157.57), EASYSIMD_FLOAT32_C(  -919.13), EASYSIMD_FLOAT32_C(   451.36)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -736.64), EASYSIMD_FLOAT32_C(   551.91), EASYSIMD_FLOAT32_C(  -457.00), EASYSIMD_FLOAT32_C(  -294.64),
                         EASYSIMD_FLOAT32_C(  -589.82), EASYSIMD_FLOAT32_C(   788.44), EASYSIMD_FLOAT32_C(  -717.27), EASYSIMD_FLOAT32_C(   147.83),
                         EASYSIMD_FLOAT32_C(  -294.04), EASYSIMD_FLOAT32_C(  -678.25), EASYSIMD_FLOAT32_C(   428.59), EASYSIMD_FLOAT32_C(  -340.21),
                         EASYSIMD_FLOAT32_C(   447.13), EASYSIMD_FLOAT32_C(  -558.56), EASYSIMD_FLOAT32_C(  -584.22), EASYSIMD_FLOAT32_C(   801.21)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   424.69), EASYSIMD_FLOAT32_C( -1508.23), EASYSIMD_FLOAT32_C(   705.48), EASYSIMD_FLOAT32_C(  1290.09),
                         EASYSIMD_FLOAT32_C(   729.69), EASYSIMD_FLOAT32_C(    -5.39), EASYSIMD_FLOAT32_C(  1301.48), EASYSIMD_FLOAT32_C( -1067.91),
                         EASYSIMD_FLOAT32_C(    83.90), EASYSIMD_FLOAT32_C(  1494.31), EASYSIMD_FLOAT32_C(  -622.27), EASYSIMD_FLOAT32_C(   925.24),
                         EASYSIMD_FLOAT32_C( -1121.21), EASYSIMD_FLOAT32_C(   716.13), EASYSIMD_FLOAT32_C(  -334.91), EASYSIMD_FLOAT32_C(  -349.85)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -668.75), EASYSIMD_FLOAT32_C(  -693.34), EASYSIMD_FLOAT32_C(    34.22), EASYSIMD_FLOAT32_C(   781.55),
                         EASYSIMD_FLOAT32_C(   732.13), EASYSIMD_FLOAT32_C(  -735.61), EASYSIMD_FLOAT32_C(  -765.87), EASYSIMD_FLOAT32_C(  -276.25),
                         EASYSIMD_FLOAT32_C(   583.37), EASYSIMD_FLOAT32_C(   151.60), EASYSIMD_FLOAT32_C(  -526.34), EASYSIMD_FLOAT32_C(  -118.48),
                         EASYSIMD_FLOAT32_C(  -603.65), EASYSIMD_FLOAT32_C(   -96.99), EASYSIMD_FLOAT32_C(  -634.86), EASYSIMD_FLOAT32_C(   225.44)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     5.83), EASYSIMD_FLOAT32_C(   767.38), EASYSIMD_FLOAT32_C(   251.47), EASYSIMD_FLOAT32_C(  -790.79),
                         EASYSIMD_FLOAT32_C(   317.44), EASYSIMD_FLOAT32_C(   889.98), EASYSIMD_FLOAT32_C(   932.08), EASYSIMD_FLOAT32_C(   879.75),
                         EASYSIMD_FLOAT32_C(   583.36), EASYSIMD_FLOAT32_C(   192.11), EASYSIMD_FLOAT32_C(   241.22), EASYSIMD_FLOAT32_C(  -741.26),
                         EASYSIMD_FLOAT32_C(   815.78), EASYSIMD_FLOAT32_C(  -325.43), EASYSIMD_FLOAT32_C(   457.34), EASYSIMD_FLOAT32_C(   430.70)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -674.58), EASYSIMD_FLOAT32_C( -1460.72), EASYSIMD_FLOAT32_C(  -217.25), EASYSIMD_FLOAT32_C(  1572.34),
                         EASYSIMD_FLOAT32_C(   414.69), EASYSIMD_FLOAT32_C( -1625.59), EASYSIMD_FLOAT32_C( -1697.95), EASYSIMD_FLOAT32_C( -1156.00),
                         EASYSIMD_FLOAT32_C(     0.01), EASYSIMD_FLOAT32_C(   -40.51), EASYSIMD_FLOAT32_C(  -767.56), EASYSIMD_FLOAT32_C(   622.78),
                         EASYSIMD_FLOAT32_C( -1419.43), EASYSIMD_FLOAT32_C(   228.44), EASYSIMD_FLOAT32_C( -1092.20), EASYSIMD_FLOAT32_C(  -205.26)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -679.43), EASYSIMD_FLOAT32_C(   282.17), EASYSIMD_FLOAT32_C(   993.32), EASYSIMD_FLOAT32_C(   821.29),
                         EASYSIMD_FLOAT32_C(   165.53), EASYSIMD_FLOAT32_C(   519.53), EASYSIMD_FLOAT32_C(   873.49), EASYSIMD_FLOAT32_C(   728.89),
                         EASYSIMD_FLOAT32_C(   317.74), EASYSIMD_FLOAT32_C(   -77.37), EASYSIMD_FLOAT32_C(   975.52), EASYSIMD_FLOAT32_C(   188.84),
                         EASYSIMD_FLOAT32_C(  -557.86), EASYSIMD_FLOAT32_C(   759.72), EASYSIMD_FLOAT32_C(  -874.99), EASYSIMD_FLOAT32_C(    10.90)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   242.15), EASYSIMD_FLOAT32_C(   438.99), EASYSIMD_FLOAT32_C(   772.28), EASYSIMD_FLOAT32_C(  -279.74),
                         EASYSIMD_FLOAT32_C(  -310.93), EASYSIMD_FLOAT32_C(  -848.99), EASYSIMD_FLOAT32_C(   222.85), EASYSIMD_FLOAT32_C(   300.16),
                         EASYSIMD_FLOAT32_C(   693.31), EASYSIMD_FLOAT32_C(   248.74), EASYSIMD_FLOAT32_C(   748.13), EASYSIMD_FLOAT32_C(  -760.98),
                         EASYSIMD_FLOAT32_C(   787.06), EASYSIMD_FLOAT32_C(   732.48), EASYSIMD_FLOAT32_C(  -205.98), EASYSIMD_FLOAT32_C(   629.02)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -921.58), EASYSIMD_FLOAT32_C(  -156.82), EASYSIMD_FLOAT32_C(   221.04), EASYSIMD_FLOAT32_C(  1101.03),
                         EASYSIMD_FLOAT32_C(   476.46), EASYSIMD_FLOAT32_C(  1368.52), EASYSIMD_FLOAT32_C(   650.64), EASYSIMD_FLOAT32_C(   428.73),
                         EASYSIMD_FLOAT32_C(  -375.57), EASYSIMD_FLOAT32_C(  -326.11), EASYSIMD_FLOAT32_C(   227.39), EASYSIMD_FLOAT32_C(   949.82),
                         EASYSIMD_FLOAT32_C( -1344.92), EASYSIMD_FLOAT32_C(    27.24), EASYSIMD_FLOAT32_C(  -669.01), EASYSIMD_FLOAT32_C(  -618.12)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   271.85), EASYSIMD_FLOAT32_C(    70.43), EASYSIMD_FLOAT32_C(   982.50), EASYSIMD_FLOAT32_C(    45.42),
                         EASYSIMD_FLOAT32_C(   118.63), EASYSIMD_FLOAT32_C(  -985.91), EASYSIMD_FLOAT32_C(     8.06), EASYSIMD_FLOAT32_C(   547.65),
                         EASYSIMD_FLOAT32_C(  -976.69), EASYSIMD_FLOAT32_C(  -286.32), EASYSIMD_FLOAT32_C(   986.84), EASYSIMD_FLOAT32_C(   730.82),
                         EASYSIMD_FLOAT32_C(  -481.07), EASYSIMD_FLOAT32_C(   923.92), EASYSIMD_FLOAT32_C(   879.55), EASYSIMD_FLOAT32_C(   720.13)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   551.66), EASYSIMD_FLOAT32_C(  -312.49), EASYSIMD_FLOAT32_C(  -723.23), EASYSIMD_FLOAT32_C(   -17.59),
                         EASYSIMD_FLOAT32_C(   325.03), EASYSIMD_FLOAT32_C(  -395.41), EASYSIMD_FLOAT32_C(   883.19), EASYSIMD_FLOAT32_C(  -807.12),
                         EASYSIMD_FLOAT32_C(  -228.68), EASYSIMD_FLOAT32_C(   772.42), EASYSIMD_FLOAT32_C(  -645.24), EASYSIMD_FLOAT32_C(  -500.86),
                         EASYSIMD_FLOAT32_C(   -15.19), EASYSIMD_FLOAT32_C(   910.24), EASYSIMD_FLOAT32_C(   528.66), EASYSIMD_FLOAT32_C(  -744.64)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -279.81), EASYSIMD_FLOAT32_C(   382.92), EASYSIMD_FLOAT32_C(  1705.73), EASYSIMD_FLOAT32_C(    63.01),
                         EASYSIMD_FLOAT32_C(  -206.40), EASYSIMD_FLOAT32_C(  -590.50), EASYSIMD_FLOAT32_C(  -875.13), EASYSIMD_FLOAT32_C(  1354.77),
                         EASYSIMD_FLOAT32_C(  -748.01), EASYSIMD_FLOAT32_C( -1058.74), EASYSIMD_FLOAT32_C(  1632.08), EASYSIMD_FLOAT32_C(  1231.68),
                         EASYSIMD_FLOAT32_C(  -465.88), EASYSIMD_FLOAT32_C(    13.68), EASYSIMD_FLOAT32_C(   350.89), EASYSIMD_FLOAT32_C(  1464.77)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    84.80), EASYSIMD_FLOAT32_C(  -329.58), EASYSIMD_FLOAT32_C(   766.75), EASYSIMD_FLOAT32_C(  -652.57),
                         EASYSIMD_FLOAT32_C(  -735.85), EASYSIMD_FLOAT32_C(   809.23), EASYSIMD_FLOAT32_C(   200.31), EASYSIMD_FLOAT32_C(  -623.13),
                         EASYSIMD_FLOAT32_C(  -845.05), EASYSIMD_FLOAT32_C(   364.16), EASYSIMD_FLOAT32_C(   572.02), EASYSIMD_FLOAT32_C(     0.80),
                         EASYSIMD_FLOAT32_C(  -325.98), EASYSIMD_FLOAT32_C(  -311.07), EASYSIMD_FLOAT32_C(   800.65), EASYSIMD_FLOAT32_C(  -125.96)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   485.48), EASYSIMD_FLOAT32_C(  -140.23), EASYSIMD_FLOAT32_C(  -453.31), EASYSIMD_FLOAT32_C(   -34.02),
                         EASYSIMD_FLOAT32_C(   893.13), EASYSIMD_FLOAT32_C(   152.27), EASYSIMD_FLOAT32_C(    79.60), EASYSIMD_FLOAT32_C(  -817.18),
                         EASYSIMD_FLOAT32_C(  -608.22), EASYSIMD_FLOAT32_C(  -450.43), EASYSIMD_FLOAT32_C(   547.33), EASYSIMD_FLOAT32_C(  -843.17),
                         EASYSIMD_FLOAT32_C(   492.07), EASYSIMD_FLOAT32_C(   125.25), EASYSIMD_FLOAT32_C(    50.68), EASYSIMD_FLOAT32_C(   718.03)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -400.68), EASYSIMD_FLOAT32_C(  -189.35), EASYSIMD_FLOAT32_C(  1220.06), EASYSIMD_FLOAT32_C(  -618.55),
                         EASYSIMD_FLOAT32_C( -1628.98), EASYSIMD_FLOAT32_C(   656.96), EASYSIMD_FLOAT32_C(   120.71), EASYSIMD_FLOAT32_C(   194.05),
                         EASYSIMD_FLOAT32_C(  -236.83), EASYSIMD_FLOAT32_C(   814.59), EASYSIMD_FLOAT32_C(    24.69), EASYSIMD_FLOAT32_C(   843.97),
                         EASYSIMD_FLOAT32_C(  -818.05), EASYSIMD_FLOAT32_C(  -436.32), EASYSIMD_FLOAT32_C(   749.97), EASYSIMD_FLOAT32_C(  -843.99)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -677.15), EASYSIMD_FLOAT32_C(   104.38), EASYSIMD_FLOAT32_C(  -475.85), EASYSIMD_FLOAT32_C(   787.23),
                         EASYSIMD_FLOAT32_C(   133.69), EASYSIMD_FLOAT32_C(  -960.64), EASYSIMD_FLOAT32_C(   242.81), EASYSIMD_FLOAT32_C(  -225.39),
                         EASYSIMD_FLOAT32_C(   314.69), EASYSIMD_FLOAT32_C(   228.04), EASYSIMD_FLOAT32_C(  -592.56), EASYSIMD_FLOAT32_C(   407.24),
                         EASYSIMD_FLOAT32_C(  -825.26), EASYSIMD_FLOAT32_C(  -290.43), EASYSIMD_FLOAT32_C(   962.34), EASYSIMD_FLOAT32_C(   893.07)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -995.46), EASYSIMD_FLOAT32_C(   431.24), EASYSIMD_FLOAT32_C(  -960.38), EASYSIMD_FLOAT32_C(   -49.08),
                         EASYSIMD_FLOAT32_C(   813.87), EASYSIMD_FLOAT32_C(   674.48), EASYSIMD_FLOAT32_C(   397.88), EASYSIMD_FLOAT32_C(  -954.85),
                         EASYSIMD_FLOAT32_C(   446.57), EASYSIMD_FLOAT32_C(   897.67), EASYSIMD_FLOAT32_C(   880.04), EASYSIMD_FLOAT32_C(   250.06),
                         EASYSIMD_FLOAT32_C(  -272.88), EASYSIMD_FLOAT32_C(  -311.12), EASYSIMD_FLOAT32_C(   208.86), EASYSIMD_FLOAT32_C(  -234.41)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   318.31), EASYSIMD_FLOAT32_C(  -326.86), EASYSIMD_FLOAT32_C(   484.53), EASYSIMD_FLOAT32_C(   836.31),
                         EASYSIMD_FLOAT32_C(  -680.18), EASYSIMD_FLOAT32_C( -1635.12), EASYSIMD_FLOAT32_C(  -155.07), EASYSIMD_FLOAT32_C(   729.46),
                         EASYSIMD_FLOAT32_C(  -131.88), EASYSIMD_FLOAT32_C(  -669.63), EASYSIMD_FLOAT32_C( -1472.60), EASYSIMD_FLOAT32_C(   157.18),
                         EASYSIMD_FLOAT32_C(  -552.38), EASYSIMD_FLOAT32_C(    20.69), EASYSIMD_FLOAT32_C(   753.48), EASYSIMD_FLOAT32_C(  1127.48)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -375.37), EASYSIMD_FLOAT32_C(   635.19), EASYSIMD_FLOAT32_C(  -375.80), EASYSIMD_FLOAT32_C(   342.82),
                         EASYSIMD_FLOAT32_C(  -159.29), EASYSIMD_FLOAT32_C(   450.42), EASYSIMD_FLOAT32_C(    65.30), EASYSIMD_FLOAT32_C(     7.10),
                         EASYSIMD_FLOAT32_C(  -943.32), EASYSIMD_FLOAT32_C(  -222.67), EASYSIMD_FLOAT32_C(  -766.83), EASYSIMD_FLOAT32_C(   277.09),
                         EASYSIMD_FLOAT32_C(    50.31), EASYSIMD_FLOAT32_C(   780.30), EASYSIMD_FLOAT32_C(  -514.83), EASYSIMD_FLOAT32_C(   450.20)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -353.63), EASYSIMD_FLOAT32_C(   190.41), EASYSIMD_FLOAT32_C(   122.56), EASYSIMD_FLOAT32_C(   371.55),
                         EASYSIMD_FLOAT32_C(  -453.54), EASYSIMD_FLOAT32_C(  -448.42), EASYSIMD_FLOAT32_C(   943.54), EASYSIMD_FLOAT32_C(  -548.29),
                         EASYSIMD_FLOAT32_C(   313.64), EASYSIMD_FLOAT32_C(  -524.65), EASYSIMD_FLOAT32_C(   682.10), EASYSIMD_FLOAT32_C(  -220.88),
                         EASYSIMD_FLOAT32_C(   -36.78), EASYSIMD_FLOAT32_C(  -595.06), EASYSIMD_FLOAT32_C(   283.20), EASYSIMD_FLOAT32_C(   943.66)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -21.74), EASYSIMD_FLOAT32_C(   444.78), EASYSIMD_FLOAT32_C(  -498.36), EASYSIMD_FLOAT32_C(   -28.73),
                         EASYSIMD_FLOAT32_C(   294.25), EASYSIMD_FLOAT32_C(   898.84), EASYSIMD_FLOAT32_C(  -878.24), EASYSIMD_FLOAT32_C(   555.39),
                         EASYSIMD_FLOAT32_C( -1256.96), EASYSIMD_FLOAT32_C(   301.98), EASYSIMD_FLOAT32_C( -1448.93), EASYSIMD_FLOAT32_C(   497.97),
                         EASYSIMD_FLOAT32_C(    87.09), EASYSIMD_FLOAT32_C(  1375.36), EASYSIMD_FLOAT32_C(  -798.03), EASYSIMD_FLOAT32_C(  -493.46)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sub_ps(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sub_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_sub_round_ps(EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT32_C(   316.88), EASYSIMD_FLOAT32_C(  -494.81), EASYSIMD_FLOAT32_C(  -853.38), EASYSIMD_FLOAT32_C(  -621.77),
        EASYSIMD_FLOAT32_C(   576.36), EASYSIMD_FLOAT32_C(  -508.94), EASYSIMD_FLOAT32_C(  -986.88), EASYSIMD_FLOAT32_C(  -608.31),
        EASYSIMD_FLOAT32_C(   383.85), EASYSIMD_FLOAT32_C(  -837.18), EASYSIMD_FLOAT32_C(   492.68), EASYSIMD_FLOAT32_C(     0.60),
        EASYSIMD_FLOAT32_C(   -99.34), EASYSIMD_FLOAT32_C(   425.37), EASYSIMD_FLOAT32_C(  -826.99), EASYSIMD_FLOAT32_C(   831.89) },
      { EASYSIMD_FLOAT32_C(  -797.56), EASYSIMD_FLOAT32_C(   868.68), EASYSIMD_FLOAT32_C(  -410.29), EASYSIMD_FLOAT32_C(   976.71),
        EASYSIMD_FLOAT32_C(   271.39), EASYSIMD_FLOAT32_C(  -610.29), EASYSIMD_FLOAT32_C(   883.87), EASYSIMD_FLOAT32_C(  -113.47),
        EASYSIMD_FLOAT32_C(  -532.13), EASYSIMD_FLOAT32_C(  -623.43), EASYSIMD_FLOAT32_C(   419.38), EASYSIMD_FLOAT32_C(  -770.68),
        EASYSIMD_FLOAT32_C(   835.26), EASYSIMD_FLOAT32_C(  -682.21), EASYSIMD_FLOAT32_C(  -287.80), EASYSIMD_FLOAT32_C(   152.14) },
      { EASYSIMD_FLOAT32_C(  1114.00), EASYSIMD_FLOAT32_C( -1363.00), EASYSIMD_FLOAT32_C(  -443.00), EASYSIMD_FLOAT32_C( -1598.00),
        EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(   101.00), EASYSIMD_FLOAT32_C( -1871.00), EASYSIMD_FLOAT32_C(  -495.00),
        EASYSIMD_FLOAT32_C(   916.00), EASYSIMD_FLOAT32_C(  -214.00), EASYSIMD_FLOAT32_C(    73.00), EASYSIMD_FLOAT32_C(   771.00),
        EASYSIMD_FLOAT32_C(  -935.00), EASYSIMD_FLOAT32_C(  1108.00), EASYSIMD_FLOAT32_C(  -539.00), EASYSIMD_FLOAT32_C(   680.00) },
      { EASYSIMD_FLOAT32_C(  1114.00), EASYSIMD_FLOAT32_C( -1364.00), EASYSIMD_FLOAT32_C(  -444.00), EASYSIMD_FLOAT32_C( -1599.00),
        EASYSIMD_FLOAT32_C(   304.00), EASYSIMD_FLOAT32_C(   101.00), EASYSIMD_FLOAT32_C( -1871.00), EASYSIMD_FLOAT32_C(  -495.00),
        EASYSIMD_FLOAT32_C(   915.00), EASYSIMD_FLOAT32_C(  -214.00), EASYSIMD_FLOAT32_C(    73.00), EASYSIMD_FLOAT32_C(   771.00),
        EASYSIMD_FLOAT32_C(  -935.00), EASYSIMD_FLOAT32_C(  1107.00), EASYSIMD_FLOAT32_C(  -540.00), EASYSIMD_FLOAT32_C(   679.00) },
      { EASYSIMD_FLOAT32_C(  1115.00), EASYSIMD_FLOAT32_C( -1363.00), EASYSIMD_FLOAT32_C(  -443.00), EASYSIMD_FLOAT32_C( -1598.00),
        EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(   102.00), EASYSIMD_FLOAT32_C( -1870.00), EASYSIMD_FLOAT32_C(  -494.00),
        EASYSIMD_FLOAT32_C(   916.00), EASYSIMD_FLOAT32_C(  -213.00), EASYSIMD_FLOAT32_C(    74.00), EASYSIMD_FLOAT32_C(   772.00),
        EASYSIMD_FLOAT32_C(  -934.00), EASYSIMD_FLOAT32_C(  1108.00), EASYSIMD_FLOAT32_C(  -539.00), EASYSIMD_FLOAT32_C(   680.00) },
      { EASYSIMD_FLOAT32_C(  1114.00), EASYSIMD_FLOAT32_C( -1363.00), EASYSIMD_FLOAT32_C(  -443.00), EASYSIMD_FLOAT32_C( -1598.00),
        EASYSIMD_FLOAT32_C(   304.00), EASYSIMD_FLOAT32_C(   101.00), EASYSIMD_FLOAT32_C( -1870.00), EASYSIMD_FLOAT32_C(  -494.00),
        EASYSIMD_FLOAT32_C(   915.00), EASYSIMD_FLOAT32_C(  -213.00), EASYSIMD_FLOAT32_C(    73.00), EASYSIMD_FLOAT32_C(   771.00),
        EASYSIMD_FLOAT32_C(  -934.00), EASYSIMD_FLOAT32_C(  1107.00), EASYSIMD_FLOAT32_C(  -539.00), EASYSIMD_FLOAT32_C(   679.00) },
      { EASYSIMD_FLOAT32_C(  1114.00), EASYSIMD_FLOAT32_C( -1363.00), EASYSIMD_FLOAT32_C(  -443.00), EASYSIMD_FLOAT32_C( -1598.00),
        EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(   101.00), EASYSIMD_FLOAT32_C( -1871.00), EASYSIMD_FLOAT32_C(  -495.00),
        EASYSIMD_FLOAT32_C(   916.00), EASYSIMD_FLOAT32_C(  -214.00), EASYSIMD_FLOAT32_C(    73.00), EASYSIMD_FLOAT32_C(   771.00),
        EASYSIMD_FLOAT32_C(  -935.00), EASYSIMD_FLOAT32_C(  1108.00), EASYSIMD_FLOAT32_C(  -539.00), EASYSIMD_FLOAT32_C(   680.00) } },
    { { EASYSIMD_FLOAT32_C(  -177.01), EASYSIMD_FLOAT32_C(  -141.18), EASYSIMD_FLOAT32_C(   530.37), EASYSIMD_FLOAT32_C(  -600.65),
        EASYSIMD_FLOAT32_C(   349.88), EASYSIMD_FLOAT32_C(   543.49), EASYSIMD_FLOAT32_C(  -208.96), EASYSIMD_FLOAT32_C(  -266.27),
        EASYSIMD_FLOAT32_C(   706.31), EASYSIMD_FLOAT32_C(  -716.28), EASYSIMD_FLOAT32_C(   734.32), EASYSIMD_FLOAT32_C(  -393.03),
        EASYSIMD_FLOAT32_C(   709.09), EASYSIMD_FLOAT32_C(   907.33), EASYSIMD_FLOAT32_C(  -561.14), EASYSIMD_FLOAT32_C(   911.53) },
      { EASYSIMD_FLOAT32_C(   776.02), EASYSIMD_FLOAT32_C(    28.57), EASYSIMD_FLOAT32_C(   888.24), EASYSIMD_FLOAT32_C(    47.41),
        EASYSIMD_FLOAT32_C(   418.28), EASYSIMD_FLOAT32_C(   772.11), EASYSIMD_FLOAT32_C(   933.94), EASYSIMD_FLOAT32_C(   886.15),
        EASYSIMD_FLOAT32_C(  -851.32), EASYSIMD_FLOAT32_C(   353.32), EASYSIMD_FLOAT32_C(  -884.53), EASYSIMD_FLOAT32_C(   983.94),
        EASYSIMD_FLOAT32_C(   671.12), EASYSIMD_FLOAT32_C(  -172.34), EASYSIMD_FLOAT32_C(   136.08), EASYSIMD_FLOAT32_C(  -505.90) },
      { EASYSIMD_FLOAT32_C(  -953.00), EASYSIMD_FLOAT32_C(  -170.00), EASYSIMD_FLOAT32_C(  -358.00), EASYSIMD_FLOAT32_C(  -648.00),
        EASYSIMD_FLOAT32_C(   -68.00), EASYSIMD_FLOAT32_C(  -229.00), EASYSIMD_FLOAT32_C( -1143.00), EASYSIMD_FLOAT32_C( -1152.00),
        EASYSIMD_FLOAT32_C(  1558.00), EASYSIMD_FLOAT32_C( -1070.00), EASYSIMD_FLOAT32_C(  1619.00), EASYSIMD_FLOAT32_C( -1377.00),
        EASYSIMD_FLOAT32_C(    38.00), EASYSIMD_FLOAT32_C(  1080.00), EASYSIMD_FLOAT32_C(  -697.00), EASYSIMD_FLOAT32_C(  1417.00) },
      { EASYSIMD_FLOAT32_C(  -954.00), EASYSIMD_FLOAT32_C(  -170.00), EASYSIMD_FLOAT32_C(  -358.00), EASYSIMD_FLOAT32_C(  -649.00),
        EASYSIMD_FLOAT32_C(   -69.00), EASYSIMD_FLOAT32_C(  -229.00), EASYSIMD_FLOAT32_C( -1143.00), EASYSIMD_FLOAT32_C( -1153.00),
        EASYSIMD_FLOAT32_C(  1557.00), EASYSIMD_FLOAT32_C( -1070.00), EASYSIMD_FLOAT32_C(  1618.00), EASYSIMD_FLOAT32_C( -1377.00),
        EASYSIMD_FLOAT32_C(    37.00), EASYSIMD_FLOAT32_C(  1079.00), EASYSIMD_FLOAT32_C(  -698.00), EASYSIMD_FLOAT32_C(  1417.00) },
      { EASYSIMD_FLOAT32_C(  -953.00), EASYSIMD_FLOAT32_C(  -169.00), EASYSIMD_FLOAT32_C(  -357.00), EASYSIMD_FLOAT32_C(  -648.00),
        EASYSIMD_FLOAT32_C(   -68.00), EASYSIMD_FLOAT32_C(  -228.00), EASYSIMD_FLOAT32_C( -1142.00), EASYSIMD_FLOAT32_C( -1152.00),
        EASYSIMD_FLOAT32_C(  1558.00), EASYSIMD_FLOAT32_C( -1069.00), EASYSIMD_FLOAT32_C(  1619.00), EASYSIMD_FLOAT32_C( -1376.00),
        EASYSIMD_FLOAT32_C(    38.00), EASYSIMD_FLOAT32_C(  1080.00), EASYSIMD_FLOAT32_C(  -697.00), EASYSIMD_FLOAT32_C(  1418.00) },
      { EASYSIMD_FLOAT32_C(  -953.00), EASYSIMD_FLOAT32_C(  -169.00), EASYSIMD_FLOAT32_C(  -357.00), EASYSIMD_FLOAT32_C(  -648.00),
        EASYSIMD_FLOAT32_C(   -68.00), EASYSIMD_FLOAT32_C(  -228.00), EASYSIMD_FLOAT32_C( -1142.00), EASYSIMD_FLOAT32_C( -1152.00),
        EASYSIMD_FLOAT32_C(  1557.00), EASYSIMD_FLOAT32_C( -1069.00), EASYSIMD_FLOAT32_C(  1618.00), EASYSIMD_FLOAT32_C( -1376.00),
        EASYSIMD_FLOAT32_C(    37.00), EASYSIMD_FLOAT32_C(  1079.00), EASYSIMD_FLOAT32_C(  -697.00), EASYSIMD_FLOAT32_C(  1417.00) },
      { EASYSIMD_FLOAT32_C(  -953.00), EASYSIMD_FLOAT32_C(  -170.00), EASYSIMD_FLOAT32_C(  -358.00), EASYSIMD_FLOAT32_C(  -648.00),
        EASYSIMD_FLOAT32_C(   -68.00), EASYSIMD_FLOAT32_C(  -229.00), EASYSIMD_FLOAT32_C( -1143.00), EASYSIMD_FLOAT32_C( -1152.00),
        EASYSIMD_FLOAT32_C(  1558.00), EASYSIMD_FLOAT32_C( -1070.00), EASYSIMD_FLOAT32_C(  1619.00), EASYSIMD_FLOAT32_C( -1377.00),
        EASYSIMD_FLOAT32_C(    38.00), EASYSIMD_FLOAT32_C(  1080.00), EASYSIMD_FLOAT32_C(  -697.00), EASYSIMD_FLOAT32_C(  1417.00) } },
    { { EASYSIMD_FLOAT32_C(   686.48), EASYSIMD_FLOAT32_C(  -333.56), EASYSIMD_FLOAT32_C(  -106.55), EASYSIMD_FLOAT32_C(    36.35),
        EASYSIMD_FLOAT32_C(  -790.06), EASYSIMD_FLOAT32_C(   684.49), EASYSIMD_FLOAT32_C(   770.08), EASYSIMD_FLOAT32_C(   916.25),
        EASYSIMD_FLOAT32_C(   968.21), EASYSIMD_FLOAT32_C(   504.41), EASYSIMD_FLOAT32_C(  -476.78), EASYSIMD_FLOAT32_C(   677.31),
        EASYSIMD_FLOAT32_C(   411.74), EASYSIMD_FLOAT32_C(   -37.92), EASYSIMD_FLOAT32_C(   588.84), EASYSIMD_FLOAT32_C(   187.76) },
      { EASYSIMD_FLOAT32_C(   990.64), EASYSIMD_FLOAT32_C(   477.08), EASYSIMD_FLOAT32_C(  -764.83), EASYSIMD_FLOAT32_C(   408.92),
        EASYSIMD_FLOAT32_C(   249.20), EASYSIMD_FLOAT32_C(  -830.89), EASYSIMD_FLOAT32_C(   295.07), EASYSIMD_FLOAT32_C(   397.88),
        EASYSIMD_FLOAT32_C(   522.43), EASYSIMD_FLOAT32_C(   410.53), EASYSIMD_FLOAT32_C(   381.82), EASYSIMD_FLOAT32_C(   193.55),
        EASYSIMD_FLOAT32_C(  -761.80), EASYSIMD_FLOAT32_C(  -482.10), EASYSIMD_FLOAT32_C(   687.65), EASYSIMD_FLOAT32_C(   924.68) },
      { EASYSIMD_FLOAT32_C(  -304.00), EASYSIMD_FLOAT32_C(  -811.00), EASYSIMD_FLOAT32_C(   658.00), EASYSIMD_FLOAT32_C(  -373.00),
        EASYSIMD_FLOAT32_C( -1039.00), EASYSIMD_FLOAT32_C(  1515.00), EASYSIMD_FLOAT32_C(   475.00), EASYSIMD_FLOAT32_C(   518.00),
        EASYSIMD_FLOAT32_C(   446.00), EASYSIMD_FLOAT32_C(    94.00), EASYSIMD_FLOAT32_C(  -859.00), EASYSIMD_FLOAT32_C(   484.00),
        EASYSIMD_FLOAT32_C(  1174.00), EASYSIMD_FLOAT32_C(   444.00), EASYSIMD_FLOAT32_C(   -99.00), EASYSIMD_FLOAT32_C(  -737.00) },
      { EASYSIMD_FLOAT32_C(  -305.00), EASYSIMD_FLOAT32_C(  -811.00), EASYSIMD_FLOAT32_C(   658.00), EASYSIMD_FLOAT32_C(  -373.00),
        EASYSIMD_FLOAT32_C( -1040.00), EASYSIMD_FLOAT32_C(  1515.00), EASYSIMD_FLOAT32_C(   475.00), EASYSIMD_FLOAT32_C(   518.00),
        EASYSIMD_FLOAT32_C(   445.00), EASYSIMD_FLOAT32_C(    93.00), EASYSIMD_FLOAT32_C(  -859.00), EASYSIMD_FLOAT32_C(   483.00),
        EASYSIMD_FLOAT32_C(  1173.00), EASYSIMD_FLOAT32_C(   444.00), EASYSIMD_FLOAT32_C(   -99.00), EASYSIMD_FLOAT32_C(  -737.00) },
      { EASYSIMD_FLOAT32_C(  -304.00), EASYSIMD_FLOAT32_C(  -810.00), EASYSIMD_FLOAT32_C(   659.00), EASYSIMD_FLOAT32_C(  -372.00),
        EASYSIMD_FLOAT32_C( -1039.00), EASYSIMD_FLOAT32_C(  1516.00), EASYSIMD_FLOAT32_C(   476.00), EASYSIMD_FLOAT32_C(   519.00),
        EASYSIMD_FLOAT32_C(   446.00), EASYSIMD_FLOAT32_C(    94.00), EASYSIMD_FLOAT32_C(  -858.00), EASYSIMD_FLOAT32_C(   484.00),
        EASYSIMD_FLOAT32_C(  1174.00), EASYSIMD_FLOAT32_C(   445.00), EASYSIMD_FLOAT32_C(   -98.00), EASYSIMD_FLOAT32_C(  -736.00) },
      { EASYSIMD_FLOAT32_C(  -304.00), EASYSIMD_FLOAT32_C(  -810.00), EASYSIMD_FLOAT32_C(   658.00), EASYSIMD_FLOAT32_C(  -372.00),
        EASYSIMD_FLOAT32_C( -1039.00), EASYSIMD_FLOAT32_C(  1515.00), EASYSIMD_FLOAT32_C(   475.00), EASYSIMD_FLOAT32_C(   518.00),
        EASYSIMD_FLOAT32_C(   445.00), EASYSIMD_FLOAT32_C(    93.00), EASYSIMD_FLOAT32_C(  -858.00), EASYSIMD_FLOAT32_C(   483.00),
        EASYSIMD_FLOAT32_C(  1173.00), EASYSIMD_FLOAT32_C(   444.00), EASYSIMD_FLOAT32_C(   -98.00), EASYSIMD_FLOAT32_C(  -736.00) },
      { EASYSIMD_FLOAT32_C(  -304.00), EASYSIMD_FLOAT32_C(  -811.00), EASYSIMD_FLOAT32_C(   658.00), EASYSIMD_FLOAT32_C(  -373.00),
        EASYSIMD_FLOAT32_C( -1039.00), EASYSIMD_FLOAT32_C(  1515.00), EASYSIMD_FLOAT32_C(   475.00), EASYSIMD_FLOAT32_C(   518.00),
        EASYSIMD_FLOAT32_C(   446.00), EASYSIMD_FLOAT32_C(    94.00), EASYSIMD_FLOAT32_C(  -859.00), EASYSIMD_FLOAT32_C(   484.00),
        EASYSIMD_FLOAT32_C(  1174.00), EASYSIMD_FLOAT32_C(   444.00), EASYSIMD_FLOAT32_C(   -99.00), EASYSIMD_FLOAT32_C(  -737.00) } },
    { { EASYSIMD_FLOAT32_C(   184.34), EASYSIMD_FLOAT32_C(  -418.90), EASYSIMD_FLOAT32_C(   -38.97), EASYSIMD_FLOAT32_C(   394.28),
        EASYSIMD_FLOAT32_C(  -734.41), EASYSIMD_FLOAT32_C(  -268.89), EASYSIMD_FLOAT32_C(   310.53), EASYSIMD_FLOAT32_C(  -766.20),
        EASYSIMD_FLOAT32_C(  -764.48), EASYSIMD_FLOAT32_C(   833.74), EASYSIMD_FLOAT32_C(   911.11), EASYSIMD_FLOAT32_C(   647.26),
        EASYSIMD_FLOAT32_C(  -204.18), EASYSIMD_FLOAT32_C(   499.95), EASYSIMD_FLOAT32_C(  -164.98), EASYSIMD_FLOAT32_C(  -213.54) },
      { EASYSIMD_FLOAT32_C(   -22.97), EASYSIMD_FLOAT32_C(    70.19), EASYSIMD_FLOAT32_C(  -804.62), EASYSIMD_FLOAT32_C(  -773.77),
        EASYSIMD_FLOAT32_C(   239.29), EASYSIMD_FLOAT32_C(   490.45), EASYSIMD_FLOAT32_C(   624.11), EASYSIMD_FLOAT32_C(  -238.27),
        EASYSIMD_FLOAT32_C(   -99.01), EASYSIMD_FLOAT32_C(     5.92), EASYSIMD_FLOAT32_C(   955.28), EASYSIMD_FLOAT32_C(   139.18),
        EASYSIMD_FLOAT32_C(   523.82), EASYSIMD_FLOAT32_C(   642.93), EASYSIMD_FLOAT32_C(    63.86), EASYSIMD_FLOAT32_C(  -291.84) },
      { EASYSIMD_FLOAT32_C(   207.00), EASYSIMD_FLOAT32_C(  -489.00), EASYSIMD_FLOAT32_C(   766.00), EASYSIMD_FLOAT32_C(  1168.00),
        EASYSIMD_FLOAT32_C(  -974.00), EASYSIMD_FLOAT32_C(  -759.00), EASYSIMD_FLOAT32_C(  -314.00), EASYSIMD_FLOAT32_C(  -528.00),
        EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(   828.00), EASYSIMD_FLOAT32_C(   -44.00), EASYSIMD_FLOAT32_C(   508.00),
        EASYSIMD_FLOAT32_C(  -728.00), EASYSIMD_FLOAT32_C(  -143.00), EASYSIMD_FLOAT32_C(  -229.00), EASYSIMD_FLOAT32_C(    78.00) },
      { EASYSIMD_FLOAT32_C(   207.00), EASYSIMD_FLOAT32_C(  -490.00), EASYSIMD_FLOAT32_C(   765.00), EASYSIMD_FLOAT32_C(  1168.00),
        EASYSIMD_FLOAT32_C(  -974.00), EASYSIMD_FLOAT32_C(  -760.00), EASYSIMD_FLOAT32_C(  -314.00), EASYSIMD_FLOAT32_C(  -528.00),
        EASYSIMD_FLOAT32_C(  -666.00), EASYSIMD_FLOAT32_C(   827.00), EASYSIMD_FLOAT32_C(   -45.00), EASYSIMD_FLOAT32_C(   508.00),
        EASYSIMD_FLOAT32_C(  -728.00), EASYSIMD_FLOAT32_C(  -143.00), EASYSIMD_FLOAT32_C(  -229.00), EASYSIMD_FLOAT32_C(    78.00) },
      { EASYSIMD_FLOAT32_C(   208.00), EASYSIMD_FLOAT32_C(  -489.00), EASYSIMD_FLOAT32_C(   766.00), EASYSIMD_FLOAT32_C(  1169.00),
        EASYSIMD_FLOAT32_C(  -973.00), EASYSIMD_FLOAT32_C(  -759.00), EASYSIMD_FLOAT32_C(  -313.00), EASYSIMD_FLOAT32_C(  -527.00),
        EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(   828.00), EASYSIMD_FLOAT32_C(   -44.00), EASYSIMD_FLOAT32_C(   509.00),
        EASYSIMD_FLOAT32_C(  -728.00), EASYSIMD_FLOAT32_C(  -142.00), EASYSIMD_FLOAT32_C(  -228.00), EASYSIMD_FLOAT32_C(    79.00) },
      { EASYSIMD_FLOAT32_C(   207.00), EASYSIMD_FLOAT32_C(  -489.00), EASYSIMD_FLOAT32_C(   765.00), EASYSIMD_FLOAT32_C(  1168.00),
        EASYSIMD_FLOAT32_C(  -973.00), EASYSIMD_FLOAT32_C(  -759.00), EASYSIMD_FLOAT32_C(  -313.00), EASYSIMD_FLOAT32_C(  -527.00),
        EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(   827.00), EASYSIMD_FLOAT32_C(   -44.00), EASYSIMD_FLOAT32_C(   508.00),
        EASYSIMD_FLOAT32_C(  -728.00), EASYSIMD_FLOAT32_C(  -142.00), EASYSIMD_FLOAT32_C(  -228.00), EASYSIMD_FLOAT32_C(    78.00) },
      { EASYSIMD_FLOAT32_C(   207.00), EASYSIMD_FLOAT32_C(  -489.00), EASYSIMD_FLOAT32_C(   766.00), EASYSIMD_FLOAT32_C(  1168.00),
        EASYSIMD_FLOAT32_C(  -974.00), EASYSIMD_FLOAT32_C(  -759.00), EASYSIMD_FLOAT32_C(  -314.00), EASYSIMD_FLOAT32_C(  -528.00),
        EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(   828.00), EASYSIMD_FLOAT32_C(   -44.00), EASYSIMD_FLOAT32_C(   508.00),
        EASYSIMD_FLOAT32_C(  -728.00), EASYSIMD_FLOAT32_C(  -143.00), EASYSIMD_FLOAT32_C(  -229.00), EASYSIMD_FLOAT32_C(    78.00) } },
    { { EASYSIMD_FLOAT32_C(  -775.97), EASYSIMD_FLOAT32_C(  -975.11), EASYSIMD_FLOAT32_C(  -897.56), EASYSIMD_FLOAT32_C(  -510.37),
        EASYSIMD_FLOAT32_C(  -244.00), EASYSIMD_FLOAT32_C(   412.96), EASYSIMD_FLOAT32_C(  -276.57), EASYSIMD_FLOAT32_C(    -8.48),
        EASYSIMD_FLOAT32_C(   246.71), EASYSIMD_FLOAT32_C(  -365.46), EASYSIMD_FLOAT32_C(  -361.22), EASYSIMD_FLOAT32_C(  -957.47),
        EASYSIMD_FLOAT32_C(  -865.51), EASYSIMD_FLOAT32_C(   473.79), EASYSIMD_FLOAT32_C(  -171.01), EASYSIMD_FLOAT32_C(   111.52) },
      { EASYSIMD_FLOAT32_C(  -456.02), EASYSIMD_FLOAT32_C(    24.38), EASYSIMD_FLOAT32_C(   337.75), EASYSIMD_FLOAT32_C(   783.27),
        EASYSIMD_FLOAT32_C(  -485.17), EASYSIMD_FLOAT32_C(   -38.15), EASYSIMD_FLOAT32_C(  -455.00), EASYSIMD_FLOAT32_C(   415.81),
        EASYSIMD_FLOAT32_C(   967.78), EASYSIMD_FLOAT32_C(  -499.72), EASYSIMD_FLOAT32_C(  -445.00), EASYSIMD_FLOAT32_C(   491.60),
        EASYSIMD_FLOAT32_C(  -856.79), EASYSIMD_FLOAT32_C(   618.85), EASYSIMD_FLOAT32_C(  -800.24), EASYSIMD_FLOAT32_C(  -632.76) },
      { EASYSIMD_FLOAT32_C(  -320.00), EASYSIMD_FLOAT32_C(  -999.00), EASYSIMD_FLOAT32_C( -1235.00), EASYSIMD_FLOAT32_C( -1294.00),
        EASYSIMD_FLOAT32_C(   241.00), EASYSIMD_FLOAT32_C(   451.00), EASYSIMD_FLOAT32_C(   178.00), EASYSIMD_FLOAT32_C(  -424.00),
        EASYSIMD_FLOAT32_C(  -721.00), EASYSIMD_FLOAT32_C(   134.00), EASYSIMD_FLOAT32_C(    84.00), EASYSIMD_FLOAT32_C( -1449.00),
        EASYSIMD_FLOAT32_C(    -9.00), EASYSIMD_FLOAT32_C(  -145.00), EASYSIMD_FLOAT32_C(   629.00), EASYSIMD_FLOAT32_C(   744.00) },
      { EASYSIMD_FLOAT32_C(  -320.00), EASYSIMD_FLOAT32_C( -1000.00), EASYSIMD_FLOAT32_C( -1236.00), EASYSIMD_FLOAT32_C( -1294.00),
        EASYSIMD_FLOAT32_C(   241.00), EASYSIMD_FLOAT32_C(   451.00), EASYSIMD_FLOAT32_C(   178.00), EASYSIMD_FLOAT32_C(  -425.00),
        EASYSIMD_FLOAT32_C(  -722.00), EASYSIMD_FLOAT32_C(   134.00), EASYSIMD_FLOAT32_C(    83.00), EASYSIMD_FLOAT32_C( -1450.00),
        EASYSIMD_FLOAT32_C(    -9.00), EASYSIMD_FLOAT32_C(  -146.00), EASYSIMD_FLOAT32_C(   629.00), EASYSIMD_FLOAT32_C(   744.00) },
      { EASYSIMD_FLOAT32_C(  -319.00), EASYSIMD_FLOAT32_C(  -999.00), EASYSIMD_FLOAT32_C( -1235.00), EASYSIMD_FLOAT32_C( -1293.00),
        EASYSIMD_FLOAT32_C(   242.00), EASYSIMD_FLOAT32_C(   452.00), EASYSIMD_FLOAT32_C(   179.00), EASYSIMD_FLOAT32_C(  -424.00),
        EASYSIMD_FLOAT32_C(  -721.00), EASYSIMD_FLOAT32_C(   135.00), EASYSIMD_FLOAT32_C(    84.00), EASYSIMD_FLOAT32_C( -1449.00),
        EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(  -145.00), EASYSIMD_FLOAT32_C(   630.00), EASYSIMD_FLOAT32_C(   745.00) },
      { EASYSIMD_FLOAT32_C(  -319.00), EASYSIMD_FLOAT32_C(  -999.00), EASYSIMD_FLOAT32_C( -1235.00), EASYSIMD_FLOAT32_C( -1293.00),
        EASYSIMD_FLOAT32_C(   241.00), EASYSIMD_FLOAT32_C(   451.00), EASYSIMD_FLOAT32_C(   178.00), EASYSIMD_FLOAT32_C(  -424.00),
        EASYSIMD_FLOAT32_C(  -721.00), EASYSIMD_FLOAT32_C(   134.00), EASYSIMD_FLOAT32_C(    83.00), EASYSIMD_FLOAT32_C( -1449.00),
        EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(  -145.00), EASYSIMD_FLOAT32_C(   629.00), EASYSIMD_FLOAT32_C(   744.00) },
      { EASYSIMD_FLOAT32_C(  -320.00), EASYSIMD_FLOAT32_C(  -999.00), EASYSIMD_FLOAT32_C( -1235.00), EASYSIMD_FLOAT32_C( -1294.00),
        EASYSIMD_FLOAT32_C(   241.00), EASYSIMD_FLOAT32_C(   451.00), EASYSIMD_FLOAT32_C(   178.00), EASYSIMD_FLOAT32_C(  -424.00),
        EASYSIMD_FLOAT32_C(  -721.00), EASYSIMD_FLOAT32_C(   134.00), EASYSIMD_FLOAT32_C(    84.00), EASYSIMD_FLOAT32_C( -1449.00),
        EASYSIMD_FLOAT32_C(    -9.00), EASYSIMD_FLOAT32_C(  -145.00), EASYSIMD_FLOAT32_C(   629.00), EASYSIMD_FLOAT32_C(   744.00) } },
    { { EASYSIMD_FLOAT32_C(   643.74), EASYSIMD_FLOAT32_C(  -697.80), EASYSIMD_FLOAT32_C(  -143.13), EASYSIMD_FLOAT32_C(  -600.26),
        EASYSIMD_FLOAT32_C(   715.16), EASYSIMD_FLOAT32_C(   580.30), EASYSIMD_FLOAT32_C(   391.26), EASYSIMD_FLOAT32_C(   -38.13),
        EASYSIMD_FLOAT32_C(  -785.16), EASYSIMD_FLOAT32_C(  -969.97), EASYSIMD_FLOAT32_C(     4.39), EASYSIMD_FLOAT32_C(  -650.67),
        EASYSIMD_FLOAT32_C(   503.83), EASYSIMD_FLOAT32_C(   833.39), EASYSIMD_FLOAT32_C(   460.85), EASYSIMD_FLOAT32_C(  -952.19) },
      { EASYSIMD_FLOAT32_C(  -142.24), EASYSIMD_FLOAT32_C(  -201.40), EASYSIMD_FLOAT32_C(   831.08), EASYSIMD_FLOAT32_C(   372.59),
        EASYSIMD_FLOAT32_C(   760.45), EASYSIMD_FLOAT32_C(  -623.91), EASYSIMD_FLOAT32_C(  -211.59), EASYSIMD_FLOAT32_C(   728.23),
        EASYSIMD_FLOAT32_C(  -123.63), EASYSIMD_FLOAT32_C(   343.40), EASYSIMD_FLOAT32_C(   219.83), EASYSIMD_FLOAT32_C(    19.58),
        EASYSIMD_FLOAT32_C(   -37.75), EASYSIMD_FLOAT32_C(   419.59), EASYSIMD_FLOAT32_C(   386.82), EASYSIMD_FLOAT32_C(  -394.00) },
      { EASYSIMD_FLOAT32_C(   786.00), EASYSIMD_FLOAT32_C(  -496.00), EASYSIMD_FLOAT32_C(  -974.00), EASYSIMD_FLOAT32_C(  -973.00),
        EASYSIMD_FLOAT32_C(   -45.00), EASYSIMD_FLOAT32_C(  1204.00), EASYSIMD_FLOAT32_C(   603.00), EASYSIMD_FLOAT32_C(  -766.00),
        EASYSIMD_FLOAT32_C(  -662.00), EASYSIMD_FLOAT32_C( -1313.00), EASYSIMD_FLOAT32_C(  -215.00), EASYSIMD_FLOAT32_C(  -670.00),
        EASYSIMD_FLOAT32_C(   542.00), EASYSIMD_FLOAT32_C(   414.00), EASYSIMD_FLOAT32_C(    74.00), EASYSIMD_FLOAT32_C(  -558.00) },
      { EASYSIMD_FLOAT32_C(   785.00), EASYSIMD_FLOAT32_C(  -497.00), EASYSIMD_FLOAT32_C(  -975.00), EASYSIMD_FLOAT32_C(  -973.00),
        EASYSIMD_FLOAT32_C(   -46.00), EASYSIMD_FLOAT32_C(  1204.00), EASYSIMD_FLOAT32_C(   602.00), EASYSIMD_FLOAT32_C(  -767.00),
        EASYSIMD_FLOAT32_C(  -662.00), EASYSIMD_FLOAT32_C( -1314.00), EASYSIMD_FLOAT32_C(  -216.00), EASYSIMD_FLOAT32_C(  -671.00),
        EASYSIMD_FLOAT32_C(   541.00), EASYSIMD_FLOAT32_C(   413.00), EASYSIMD_FLOAT32_C(    74.00), EASYSIMD_FLOAT32_C(  -559.00) },
      { EASYSIMD_FLOAT32_C(   786.00), EASYSIMD_FLOAT32_C(  -496.00), EASYSIMD_FLOAT32_C(  -974.00), EASYSIMD_FLOAT32_C(  -972.00),
        EASYSIMD_FLOAT32_C(   -45.00), EASYSIMD_FLOAT32_C(  1205.00), EASYSIMD_FLOAT32_C(   603.00), EASYSIMD_FLOAT32_C(  -766.00),
        EASYSIMD_FLOAT32_C(  -661.00), EASYSIMD_FLOAT32_C( -1313.00), EASYSIMD_FLOAT32_C(  -215.00), EASYSIMD_FLOAT32_C(  -670.00),
        EASYSIMD_FLOAT32_C(   542.00), EASYSIMD_FLOAT32_C(   414.00), EASYSIMD_FLOAT32_C(    75.00), EASYSIMD_FLOAT32_C(  -558.00) },
      { EASYSIMD_FLOAT32_C(   785.00), EASYSIMD_FLOAT32_C(  -496.00), EASYSIMD_FLOAT32_C(  -974.00), EASYSIMD_FLOAT32_C(  -972.00),
        EASYSIMD_FLOAT32_C(   -45.00), EASYSIMD_FLOAT32_C(  1204.00), EASYSIMD_FLOAT32_C(   602.00), EASYSIMD_FLOAT32_C(  -766.00),
        EASYSIMD_FLOAT32_C(  -661.00), EASYSIMD_FLOAT32_C( -1313.00), EASYSIMD_FLOAT32_C(  -215.00), EASYSIMD_FLOAT32_C(  -670.00),
        EASYSIMD_FLOAT32_C(   541.00), EASYSIMD_FLOAT32_C(   413.00), EASYSIMD_FLOAT32_C(    74.00), EASYSIMD_FLOAT32_C(  -558.00) },
      { EASYSIMD_FLOAT32_C(   786.00), EASYSIMD_FLOAT32_C(  -496.00), EASYSIMD_FLOAT32_C(  -974.00), EASYSIMD_FLOAT32_C(  -973.00),
        EASYSIMD_FLOAT32_C(   -45.00), EASYSIMD_FLOAT32_C(  1204.00), EASYSIMD_FLOAT32_C(   603.00), EASYSIMD_FLOAT32_C(  -766.00),
        EASYSIMD_FLOAT32_C(  -662.00), EASYSIMD_FLOAT32_C( -1313.00), EASYSIMD_FLOAT32_C(  -215.00), EASYSIMD_FLOAT32_C(  -670.00),
        EASYSIMD_FLOAT32_C(   542.00), EASYSIMD_FLOAT32_C(   414.00), EASYSIMD_FLOAT32_C(    74.00), EASYSIMD_FLOAT32_C(  -558.00) } },
    { { EASYSIMD_FLOAT32_C(   721.78), EASYSIMD_FLOAT32_C(  -756.31), EASYSIMD_FLOAT32_C(     5.74), EASYSIMD_FLOAT32_C(   436.94),
        EASYSIMD_FLOAT32_C(   823.99), EASYSIMD_FLOAT32_C(  -603.01), EASYSIMD_FLOAT32_C(  -601.19), EASYSIMD_FLOAT32_C(  -961.17),
        EASYSIMD_FLOAT32_C(  -572.97), EASYSIMD_FLOAT32_C(   403.20), EASYSIMD_FLOAT32_C(  -611.84), EASYSIMD_FLOAT32_C(   930.86),
        EASYSIMD_FLOAT32_C(   236.59), EASYSIMD_FLOAT32_C(   849.01), EASYSIMD_FLOAT32_C(   978.67), EASYSIMD_FLOAT32_C(  -905.65) },
      { EASYSIMD_FLOAT32_C(  -352.39), EASYSIMD_FLOAT32_C(   809.75), EASYSIMD_FLOAT32_C(   466.94), EASYSIMD_FLOAT32_C(  -591.94),
        EASYSIMD_FLOAT32_C(  -814.16), EASYSIMD_FLOAT32_C(  -744.65), EASYSIMD_FLOAT32_C(  -863.71), EASYSIMD_FLOAT32_C(    62.21),
        EASYSIMD_FLOAT32_C(   598.75), EASYSIMD_FLOAT32_C(   356.12), EASYSIMD_FLOAT32_C(  -918.21), EASYSIMD_FLOAT32_C(  -439.00),
        EASYSIMD_FLOAT32_C(  -224.29), EASYSIMD_FLOAT32_C(   468.61), EASYSIMD_FLOAT32_C(   167.00), EASYSIMD_FLOAT32_C(  -502.51) },
      { EASYSIMD_FLOAT32_C(  1074.00), EASYSIMD_FLOAT32_C( -1566.00), EASYSIMD_FLOAT32_C(  -461.00), EASYSIMD_FLOAT32_C(  1029.00),
        EASYSIMD_FLOAT32_C(  1638.00), EASYSIMD_FLOAT32_C(   142.00), EASYSIMD_FLOAT32_C(   263.00), EASYSIMD_FLOAT32_C( -1023.00),
        EASYSIMD_FLOAT32_C( -1172.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(   306.00), EASYSIMD_FLOAT32_C(  1370.00),
        EASYSIMD_FLOAT32_C(   461.00), EASYSIMD_FLOAT32_C(   380.00), EASYSIMD_FLOAT32_C(   812.00), EASYSIMD_FLOAT32_C(  -403.00) },
      { EASYSIMD_FLOAT32_C(  1074.00), EASYSIMD_FLOAT32_C( -1567.00), EASYSIMD_FLOAT32_C(  -462.00), EASYSIMD_FLOAT32_C(  1028.00),
        EASYSIMD_FLOAT32_C(  1638.00), EASYSIMD_FLOAT32_C(   141.00), EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C( -1024.00),
        EASYSIMD_FLOAT32_C( -1172.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(   306.00), EASYSIMD_FLOAT32_C(  1369.00),
        EASYSIMD_FLOAT32_C(   460.00), EASYSIMD_FLOAT32_C(   380.00), EASYSIMD_FLOAT32_C(   811.00), EASYSIMD_FLOAT32_C(  -404.00) },
      { EASYSIMD_FLOAT32_C(  1075.00), EASYSIMD_FLOAT32_C( -1566.00), EASYSIMD_FLOAT32_C(  -461.00), EASYSIMD_FLOAT32_C(  1029.00),
        EASYSIMD_FLOAT32_C(  1639.00), EASYSIMD_FLOAT32_C(   142.00), EASYSIMD_FLOAT32_C(   263.00), EASYSIMD_FLOAT32_C( -1023.00),
        EASYSIMD_FLOAT32_C( -1171.00), EASYSIMD_FLOAT32_C(    48.00), EASYSIMD_FLOAT32_C(   307.00), EASYSIMD_FLOAT32_C(  1370.00),
        EASYSIMD_FLOAT32_C(   461.00), EASYSIMD_FLOAT32_C(   381.00), EASYSIMD_FLOAT32_C(   812.00), EASYSIMD_FLOAT32_C(  -403.00) },
      { EASYSIMD_FLOAT32_C(  1074.00), EASYSIMD_FLOAT32_C( -1566.00), EASYSIMD_FLOAT32_C(  -461.00), EASYSIMD_FLOAT32_C(  1028.00),
        EASYSIMD_FLOAT32_C(  1638.00), EASYSIMD_FLOAT32_C(   141.00), EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C( -1023.00),
        EASYSIMD_FLOAT32_C( -1171.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(   306.00), EASYSIMD_FLOAT32_C(  1369.00),
        EASYSIMD_FLOAT32_C(   460.00), EASYSIMD_FLOAT32_C(   380.00), EASYSIMD_FLOAT32_C(   811.00), EASYSIMD_FLOAT32_C(  -403.00) },
      { EASYSIMD_FLOAT32_C(  1074.00), EASYSIMD_FLOAT32_C( -1566.00), EASYSIMD_FLOAT32_C(  -461.00), EASYSIMD_FLOAT32_C(  1029.00),
        EASYSIMD_FLOAT32_C(  1638.00), EASYSIMD_FLOAT32_C(   142.00), EASYSIMD_FLOAT32_C(   263.00), EASYSIMD_FLOAT32_C( -1023.00),
        EASYSIMD_FLOAT32_C( -1172.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(   306.00), EASYSIMD_FLOAT32_C(  1370.00),
        EASYSIMD_FLOAT32_C(   461.00), EASYSIMD_FLOAT32_C(   380.00), EASYSIMD_FLOAT32_C(   812.00), EASYSIMD_FLOAT32_C(  -403.00) } },
    { { EASYSIMD_FLOAT32_C(   712.30), EASYSIMD_FLOAT32_C(  -827.27), EASYSIMD_FLOAT32_C(   934.43), EASYSIMD_FLOAT32_C(   536.30),
        EASYSIMD_FLOAT32_C(  -430.27), EASYSIMD_FLOAT32_C(  -666.77), EASYSIMD_FLOAT32_C(   575.13), EASYSIMD_FLOAT32_C(    -3.24),
        EASYSIMD_FLOAT32_C(   736.43), EASYSIMD_FLOAT32_C(   963.29), EASYSIMD_FLOAT32_C(   -72.39), EASYSIMD_FLOAT32_C(   -26.98),
        EASYSIMD_FLOAT32_C(   812.30), EASYSIMD_FLOAT32_C(   -93.72), EASYSIMD_FLOAT32_C(    67.37), EASYSIMD_FLOAT32_C(  -540.09) },
      { EASYSIMD_FLOAT32_C(  -283.97), EASYSIMD_FLOAT32_C(  -465.69), EASYSIMD_FLOAT32_C(  -132.03), EASYSIMD_FLOAT32_C(   -98.13),
        EASYSIMD_FLOAT32_C(  -210.34), EASYSIMD_FLOAT32_C(     4.27), EASYSIMD_FLOAT32_C(   964.08), EASYSIMD_FLOAT32_C(  -611.59),
        EASYSIMD_FLOAT32_C(  -639.61), EASYSIMD_FLOAT32_C(  -954.13), EASYSIMD_FLOAT32_C(   -50.58), EASYSIMD_FLOAT32_C(   136.09),
        EASYSIMD_FLOAT32_C(   514.48), EASYSIMD_FLOAT32_C(  -883.59), EASYSIMD_FLOAT32_C(   633.58), EASYSIMD_FLOAT32_C(   226.78) },
      { EASYSIMD_FLOAT32_C(   996.00), EASYSIMD_FLOAT32_C(  -362.00), EASYSIMD_FLOAT32_C(  1066.00), EASYSIMD_FLOAT32_C(   634.00),
        EASYSIMD_FLOAT32_C(  -220.00), EASYSIMD_FLOAT32_C(  -671.00), EASYSIMD_FLOAT32_C(  -389.00), EASYSIMD_FLOAT32_C(   608.00),
        EASYSIMD_FLOAT32_C(  1376.00), EASYSIMD_FLOAT32_C(  1917.00), EASYSIMD_FLOAT32_C(   -22.00), EASYSIMD_FLOAT32_C(  -163.00),
        EASYSIMD_FLOAT32_C(   298.00), EASYSIMD_FLOAT32_C(   790.00), EASYSIMD_FLOAT32_C(  -566.00), EASYSIMD_FLOAT32_C(  -767.00) },
      { EASYSIMD_FLOAT32_C(   996.00), EASYSIMD_FLOAT32_C(  -362.00), EASYSIMD_FLOAT32_C(  1066.00), EASYSIMD_FLOAT32_C(   634.00),
        EASYSIMD_FLOAT32_C(  -220.00), EASYSIMD_FLOAT32_C(  -672.00), EASYSIMD_FLOAT32_C(  -389.00), EASYSIMD_FLOAT32_C(   608.00),
        EASYSIMD_FLOAT32_C(  1376.00), EASYSIMD_FLOAT32_C(  1917.00), EASYSIMD_FLOAT32_C(   -22.00), EASYSIMD_FLOAT32_C(  -164.00),
        EASYSIMD_FLOAT32_C(   297.00), EASYSIMD_FLOAT32_C(   789.00), EASYSIMD_FLOAT32_C(  -567.00), EASYSIMD_FLOAT32_C(  -767.00) },
      { EASYSIMD_FLOAT32_C(   997.00), EASYSIMD_FLOAT32_C(  -361.00), EASYSIMD_FLOAT32_C(  1067.00), EASYSIMD_FLOAT32_C(   635.00),
        EASYSIMD_FLOAT32_C(  -219.00), EASYSIMD_FLOAT32_C(  -671.00), EASYSIMD_FLOAT32_C(  -388.00), EASYSIMD_FLOAT32_C(   609.00),
        EASYSIMD_FLOAT32_C(  1377.00), EASYSIMD_FLOAT32_C(  1918.00), EASYSIMD_FLOAT32_C(   -21.00), EASYSIMD_FLOAT32_C(  -163.00),
        EASYSIMD_FLOAT32_C(   298.00), EASYSIMD_FLOAT32_C(   790.00), EASYSIMD_FLOAT32_C(  -566.00), EASYSIMD_FLOAT32_C(  -766.00) },
      { EASYSIMD_FLOAT32_C(   996.00), EASYSIMD_FLOAT32_C(  -361.00), EASYSIMD_FLOAT32_C(  1066.00), EASYSIMD_FLOAT32_C(   634.00),
        EASYSIMD_FLOAT32_C(  -219.00), EASYSIMD_FLOAT32_C(  -671.00), EASYSIMD_FLOAT32_C(  -388.00), EASYSIMD_FLOAT32_C(   608.00),
        EASYSIMD_FLOAT32_C(  1376.00), EASYSIMD_FLOAT32_C(  1917.00), EASYSIMD_FLOAT32_C(   -21.00), EASYSIMD_FLOAT32_C(  -163.00),
        EASYSIMD_FLOAT32_C(   297.00), EASYSIMD_FLOAT32_C(   789.00), EASYSIMD_FLOAT32_C(  -566.00), EASYSIMD_FLOAT32_C(  -766.00) },
      { EASYSIMD_FLOAT32_C(   996.00), EASYSIMD_FLOAT32_C(  -362.00), EASYSIMD_FLOAT32_C(  1066.00), EASYSIMD_FLOAT32_C(   634.00),
        EASYSIMD_FLOAT32_C(  -220.00), EASYSIMD_FLOAT32_C(  -671.00), EASYSIMD_FLOAT32_C(  -389.00), EASYSIMD_FLOAT32_C(   608.00),
        EASYSIMD_FLOAT32_C(  1376.00), EASYSIMD_FLOAT32_C(  1917.00), EASYSIMD_FLOAT32_C(   -22.00), EASYSIMD_FLOAT32_C(  -163.00),
        EASYSIMD_FLOAT32_C(   298.00), EASYSIMD_FLOAT32_C(   790.00), EASYSIMD_FLOAT32_C(  -566.00), EASYSIMD_FLOAT32_C(  -767.00) } }
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

    r = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512_close(r, nearest_inf, 1);

    r = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512_close(r, neg_inf, 1);

    r = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512_close(r, pos_inf, 1);

    r = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sub_round_ps");
    easysimd_assert_m512_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512 nearest_inf = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512 neg_inf = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512 pos_inf = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512 zero = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512 direction = easysimd_mm512_sub_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

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
test_easysimd_mm512_sub_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -150.13), EASYSIMD_FLOAT64_C(  450.66),
                         EASYSIMD_FLOAT64_C( -527.30), EASYSIMD_FLOAT64_C( -641.78),
                         EASYSIMD_FLOAT64_C(  929.20), EASYSIMD_FLOAT64_C( -281.32),
                         EASYSIMD_FLOAT64_C( -125.47), EASYSIMD_FLOAT64_C( -963.36)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   33.04), EASYSIMD_FLOAT64_C(  167.46),
                         EASYSIMD_FLOAT64_C( -891.13), EASYSIMD_FLOAT64_C(  473.74),
                         EASYSIMD_FLOAT64_C( -547.95), EASYSIMD_FLOAT64_C(  516.90),
                         EASYSIMD_FLOAT64_C(  -69.62), EASYSIMD_FLOAT64_C( -976.88)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -183.17), EASYSIMD_FLOAT64_C(  283.20),
                         EASYSIMD_FLOAT64_C(  363.83), EASYSIMD_FLOAT64_C(-1115.52),
                         EASYSIMD_FLOAT64_C( 1477.15), EASYSIMD_FLOAT64_C( -798.22),
                         EASYSIMD_FLOAT64_C(  -55.85), EASYSIMD_FLOAT64_C(   13.52)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -659.63), EASYSIMD_FLOAT64_C( -759.67),
                         EASYSIMD_FLOAT64_C( -847.92), EASYSIMD_FLOAT64_C(  -61.45),
                         EASYSIMD_FLOAT64_C( -337.36), EASYSIMD_FLOAT64_C(  139.68),
                         EASYSIMD_FLOAT64_C(  658.69), EASYSIMD_FLOAT64_C(   86.55)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -406.77), EASYSIMD_FLOAT64_C( -929.18),
                         EASYSIMD_FLOAT64_C(  716.57), EASYSIMD_FLOAT64_C(    1.83),
                         EASYSIMD_FLOAT64_C(  179.14), EASYSIMD_FLOAT64_C(  145.16),
                         EASYSIMD_FLOAT64_C( -463.41), EASYSIMD_FLOAT64_C( -573.03)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -252.86), EASYSIMD_FLOAT64_C(  169.51),
                         EASYSIMD_FLOAT64_C(-1564.49), EASYSIMD_FLOAT64_C(  -63.28),
                         EASYSIMD_FLOAT64_C( -516.50), EASYSIMD_FLOAT64_C(   -5.48),
                         EASYSIMD_FLOAT64_C( 1122.10), EASYSIMD_FLOAT64_C(  659.58)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -210.14), EASYSIMD_FLOAT64_C(  816.06),
                         EASYSIMD_FLOAT64_C( -193.68), EASYSIMD_FLOAT64_C(  585.03),
                         EASYSIMD_FLOAT64_C( -674.08), EASYSIMD_FLOAT64_C(  157.57),
                         EASYSIMD_FLOAT64_C( -919.13), EASYSIMD_FLOAT64_C(  451.36)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -294.04), EASYSIMD_FLOAT64_C( -678.25),
                         EASYSIMD_FLOAT64_C(  428.59), EASYSIMD_FLOAT64_C( -340.21),
                         EASYSIMD_FLOAT64_C(  447.13), EASYSIMD_FLOAT64_C( -558.56),
                         EASYSIMD_FLOAT64_C( -584.22), EASYSIMD_FLOAT64_C(  801.21)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   83.90), EASYSIMD_FLOAT64_C( 1494.31),
                         EASYSIMD_FLOAT64_C( -622.27), EASYSIMD_FLOAT64_C(  925.24),
                         EASYSIMD_FLOAT64_C(-1121.21), EASYSIMD_FLOAT64_C(  716.13),
                         EASYSIMD_FLOAT64_C( -334.91), EASYSIMD_FLOAT64_C( -349.85)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -311.95), EASYSIMD_FLOAT64_C( -956.32),
                         EASYSIMD_FLOAT64_C(  248.48), EASYSIMD_FLOAT64_C(  995.45),
                         EASYSIMD_FLOAT64_C(  139.87), EASYSIMD_FLOAT64_C(  783.05),
                         EASYSIMD_FLOAT64_C(  584.21), EASYSIMD_FLOAT64_C( -920.08)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -736.64), EASYSIMD_FLOAT64_C(  551.91),
                         EASYSIMD_FLOAT64_C( -457.00), EASYSIMD_FLOAT64_C( -294.64),
                         EASYSIMD_FLOAT64_C( -589.82), EASYSIMD_FLOAT64_C(  788.44),
                         EASYSIMD_FLOAT64_C( -717.27), EASYSIMD_FLOAT64_C(  147.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  424.69), EASYSIMD_FLOAT64_C(-1508.23),
                         EASYSIMD_FLOAT64_C(  705.48), EASYSIMD_FLOAT64_C( 1290.09),
                         EASYSIMD_FLOAT64_C(  729.69), EASYSIMD_FLOAT64_C(   -5.39),
                         EASYSIMD_FLOAT64_C( 1301.48), EASYSIMD_FLOAT64_C(-1067.91)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  583.37), EASYSIMD_FLOAT64_C(  151.60),
                         EASYSIMD_FLOAT64_C( -526.34), EASYSIMD_FLOAT64_C( -118.48),
                         EASYSIMD_FLOAT64_C( -603.65), EASYSIMD_FLOAT64_C(  -96.99),
                         EASYSIMD_FLOAT64_C( -634.86), EASYSIMD_FLOAT64_C(  225.44)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  583.36), EASYSIMD_FLOAT64_C(  192.11),
                         EASYSIMD_FLOAT64_C(  241.22), EASYSIMD_FLOAT64_C( -741.26),
                         EASYSIMD_FLOAT64_C(  815.78), EASYSIMD_FLOAT64_C( -325.43),
                         EASYSIMD_FLOAT64_C(  457.34), EASYSIMD_FLOAT64_C(  430.70)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.01), EASYSIMD_FLOAT64_C(  -40.51),
                         EASYSIMD_FLOAT64_C( -767.56), EASYSIMD_FLOAT64_C(  622.78),
                         EASYSIMD_FLOAT64_C(-1419.43), EASYSIMD_FLOAT64_C(  228.44),
                         EASYSIMD_FLOAT64_C(-1092.20), EASYSIMD_FLOAT64_C( -205.26)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -668.75), EASYSIMD_FLOAT64_C( -693.34),
                         EASYSIMD_FLOAT64_C(   34.22), EASYSIMD_FLOAT64_C(  781.55),
                         EASYSIMD_FLOAT64_C(  732.13), EASYSIMD_FLOAT64_C( -735.61),
                         EASYSIMD_FLOAT64_C( -765.87), EASYSIMD_FLOAT64_C( -276.25)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    5.83), EASYSIMD_FLOAT64_C(  767.38),
                         EASYSIMD_FLOAT64_C(  251.47), EASYSIMD_FLOAT64_C( -790.79),
                         EASYSIMD_FLOAT64_C(  317.44), EASYSIMD_FLOAT64_C(  889.98),
                         EASYSIMD_FLOAT64_C(  932.08), EASYSIMD_FLOAT64_C(  879.75)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -674.58), EASYSIMD_FLOAT64_C(-1460.72),
                         EASYSIMD_FLOAT64_C( -217.25), EASYSIMD_FLOAT64_C( 1572.34),
                         EASYSIMD_FLOAT64_C(  414.69), EASYSIMD_FLOAT64_C(-1625.59),
                         EASYSIMD_FLOAT64_C(-1697.95), EASYSIMD_FLOAT64_C(-1156.00)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  317.74), EASYSIMD_FLOAT64_C(  -77.37),
                         EASYSIMD_FLOAT64_C(  975.52), EASYSIMD_FLOAT64_C(  188.84),
                         EASYSIMD_FLOAT64_C( -557.86), EASYSIMD_FLOAT64_C(  759.72),
                         EASYSIMD_FLOAT64_C( -874.99), EASYSIMD_FLOAT64_C(   10.90)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  693.31), EASYSIMD_FLOAT64_C(  248.74),
                         EASYSIMD_FLOAT64_C(  748.13), EASYSIMD_FLOAT64_C( -760.98),
                         EASYSIMD_FLOAT64_C(  787.06), EASYSIMD_FLOAT64_C(  732.48),
                         EASYSIMD_FLOAT64_C( -205.98), EASYSIMD_FLOAT64_C(  629.02)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -375.57), EASYSIMD_FLOAT64_C( -326.11),
                         EASYSIMD_FLOAT64_C(  227.39), EASYSIMD_FLOAT64_C(  949.82),
                         EASYSIMD_FLOAT64_C(-1344.92), EASYSIMD_FLOAT64_C(   27.24),
                         EASYSIMD_FLOAT64_C( -669.01), EASYSIMD_FLOAT64_C( -618.12)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -679.43), EASYSIMD_FLOAT64_C(  282.17),
                         EASYSIMD_FLOAT64_C(  993.32), EASYSIMD_FLOAT64_C(  821.29),
                         EASYSIMD_FLOAT64_C(  165.53), EASYSIMD_FLOAT64_C(  519.53),
                         EASYSIMD_FLOAT64_C(  873.49), EASYSIMD_FLOAT64_C(  728.89)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  242.15), EASYSIMD_FLOAT64_C(  438.99),
                         EASYSIMD_FLOAT64_C(  772.28), EASYSIMD_FLOAT64_C( -279.74),
                         EASYSIMD_FLOAT64_C( -310.93), EASYSIMD_FLOAT64_C( -848.99),
                         EASYSIMD_FLOAT64_C(  222.85), EASYSIMD_FLOAT64_C(  300.16)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -921.58), EASYSIMD_FLOAT64_C( -156.82),
                         EASYSIMD_FLOAT64_C(  221.04), EASYSIMD_FLOAT64_C( 1101.03),
                         EASYSIMD_FLOAT64_C(  476.46), EASYSIMD_FLOAT64_C( 1368.52),
                         EASYSIMD_FLOAT64_C(  650.64), EASYSIMD_FLOAT64_C(  428.73)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d b = test_vec[i].b;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sub_pd(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sub_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_sub_round_pd(EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT64_C(   160.87), EASYSIMD_FLOAT64_C(   222.89), EASYSIMD_FLOAT64_C(   365.63), EASYSIMD_FLOAT64_C(  -843.84),
        EASYSIMD_FLOAT64_C(   220.54), EASYSIMD_FLOAT64_C(   991.72), EASYSIMD_FLOAT64_C(  -422.31), EASYSIMD_FLOAT64_C(   949.77) },
      { EASYSIMD_FLOAT64_C(   570.98), EASYSIMD_FLOAT64_C(   136.32), EASYSIMD_FLOAT64_C(   547.23), EASYSIMD_FLOAT64_C(   572.74),
        EASYSIMD_FLOAT64_C(   -54.05), EASYSIMD_FLOAT64_C(   970.97), EASYSIMD_FLOAT64_C(  -495.16), EASYSIMD_FLOAT64_C(   884.37) },
      { EASYSIMD_FLOAT64_C(  -410.00), EASYSIMD_FLOAT64_C(    87.00), EASYSIMD_FLOAT64_C(  -182.00), EASYSIMD_FLOAT64_C( -1417.00),
        EASYSIMD_FLOAT64_C(   275.00), EASYSIMD_FLOAT64_C(    21.00), EASYSIMD_FLOAT64_C(    73.00), EASYSIMD_FLOAT64_C(    65.00) },
      { EASYSIMD_FLOAT64_C(  -411.00), EASYSIMD_FLOAT64_C(    86.00), EASYSIMD_FLOAT64_C(  -182.00), EASYSIMD_FLOAT64_C( -1417.00),
        EASYSIMD_FLOAT64_C(   274.00), EASYSIMD_FLOAT64_C(    20.00), EASYSIMD_FLOAT64_C(    72.00), EASYSIMD_FLOAT64_C(    65.00) },
      { EASYSIMD_FLOAT64_C(  -410.00), EASYSIMD_FLOAT64_C(    87.00), EASYSIMD_FLOAT64_C(  -181.00), EASYSIMD_FLOAT64_C( -1416.00),
        EASYSIMD_FLOAT64_C(   275.00), EASYSIMD_FLOAT64_C(    21.00), EASYSIMD_FLOAT64_C(    73.00), EASYSIMD_FLOAT64_C(    66.00) },
      { EASYSIMD_FLOAT64_C(  -410.00), EASYSIMD_FLOAT64_C(    86.00), EASYSIMD_FLOAT64_C(  -181.00), EASYSIMD_FLOAT64_C( -1416.00),
        EASYSIMD_FLOAT64_C(   274.00), EASYSIMD_FLOAT64_C(    20.00), EASYSIMD_FLOAT64_C(    72.00), EASYSIMD_FLOAT64_C(    65.00) },
      { EASYSIMD_FLOAT64_C(  -410.00), EASYSIMD_FLOAT64_C(    87.00), EASYSIMD_FLOAT64_C(  -182.00), EASYSIMD_FLOAT64_C( -1417.00),
        EASYSIMD_FLOAT64_C(   275.00), EASYSIMD_FLOAT64_C(    21.00), EASYSIMD_FLOAT64_C(    73.00), EASYSIMD_FLOAT64_C(    65.00) } },
    { { EASYSIMD_FLOAT64_C(   498.40), EASYSIMD_FLOAT64_C(   682.76), EASYSIMD_FLOAT64_C(  -196.80), EASYSIMD_FLOAT64_C(   406.39),
        EASYSIMD_FLOAT64_C(   -13.19), EASYSIMD_FLOAT64_C(   397.01), EASYSIMD_FLOAT64_C(   400.55), EASYSIMD_FLOAT64_C(   241.75) },
      { EASYSIMD_FLOAT64_C(   893.46), EASYSIMD_FLOAT64_C(   304.03), EASYSIMD_FLOAT64_C(  -724.61), EASYSIMD_FLOAT64_C(  -298.04),
        EASYSIMD_FLOAT64_C(  -699.71), EASYSIMD_FLOAT64_C(  -949.67), EASYSIMD_FLOAT64_C(   340.80), EASYSIMD_FLOAT64_C(   461.16) },
      { EASYSIMD_FLOAT64_C(  -395.00), EASYSIMD_FLOAT64_C(   379.00), EASYSIMD_FLOAT64_C(   528.00), EASYSIMD_FLOAT64_C(   704.00),
        EASYSIMD_FLOAT64_C(   687.00), EASYSIMD_FLOAT64_C(  1347.00), EASYSIMD_FLOAT64_C(    60.00), EASYSIMD_FLOAT64_C(  -219.00) },
      { EASYSIMD_FLOAT64_C(  -396.00), EASYSIMD_FLOAT64_C(   378.00), EASYSIMD_FLOAT64_C(   527.00), EASYSIMD_FLOAT64_C(   704.00),
        EASYSIMD_FLOAT64_C(   686.00), EASYSIMD_FLOAT64_C(  1346.00), EASYSIMD_FLOAT64_C(    59.00), EASYSIMD_FLOAT64_C(  -220.00) },
      { EASYSIMD_FLOAT64_C(  -395.00), EASYSIMD_FLOAT64_C(   379.00), EASYSIMD_FLOAT64_C(   528.00), EASYSIMD_FLOAT64_C(   705.00),
        EASYSIMD_FLOAT64_C(   687.00), EASYSIMD_FLOAT64_C(  1347.00), EASYSIMD_FLOAT64_C(    60.00), EASYSIMD_FLOAT64_C(  -219.00) },
      { EASYSIMD_FLOAT64_C(  -395.00), EASYSIMD_FLOAT64_C(   378.00), EASYSIMD_FLOAT64_C(   527.00), EASYSIMD_FLOAT64_C(   704.00),
        EASYSIMD_FLOAT64_C(   686.00), EASYSIMD_FLOAT64_C(  1346.00), EASYSIMD_FLOAT64_C(    59.00), EASYSIMD_FLOAT64_C(  -219.00) },
      { EASYSIMD_FLOAT64_C(  -395.00), EASYSIMD_FLOAT64_C(   379.00), EASYSIMD_FLOAT64_C(   528.00), EASYSIMD_FLOAT64_C(   704.00),
        EASYSIMD_FLOAT64_C(   687.00), EASYSIMD_FLOAT64_C(  1347.00), EASYSIMD_FLOAT64_C(    60.00), EASYSIMD_FLOAT64_C(  -219.00) } },
    { { EASYSIMD_FLOAT64_C(   273.22), EASYSIMD_FLOAT64_C(  -293.57), EASYSIMD_FLOAT64_C(   617.32), EASYSIMD_FLOAT64_C(  -506.23),
        EASYSIMD_FLOAT64_C(  -301.85), EASYSIMD_FLOAT64_C(  -804.98), EASYSIMD_FLOAT64_C(  -556.47), EASYSIMD_FLOAT64_C(  -730.87) },
      { EASYSIMD_FLOAT64_C(   331.33), EASYSIMD_FLOAT64_C(   990.76), EASYSIMD_FLOAT64_C(   841.87), EASYSIMD_FLOAT64_C(  -722.71),
        EASYSIMD_FLOAT64_C(   961.73), EASYSIMD_FLOAT64_C(  -653.29), EASYSIMD_FLOAT64_C(  -838.34), EASYSIMD_FLOAT64_C(   460.13) },
      { EASYSIMD_FLOAT64_C(   -58.00), EASYSIMD_FLOAT64_C( -1284.00), EASYSIMD_FLOAT64_C(  -225.00), EASYSIMD_FLOAT64_C(   216.00),
        EASYSIMD_FLOAT64_C( -1264.00), EASYSIMD_FLOAT64_C(  -152.00), EASYSIMD_FLOAT64_C(   282.00), EASYSIMD_FLOAT64_C( -1191.00) },
      { EASYSIMD_FLOAT64_C(   -59.00), EASYSIMD_FLOAT64_C( -1285.00), EASYSIMD_FLOAT64_C(  -225.00), EASYSIMD_FLOAT64_C(   216.00),
        EASYSIMD_FLOAT64_C( -1264.00), EASYSIMD_FLOAT64_C(  -152.00), EASYSIMD_FLOAT64_C(   281.00), EASYSIMD_FLOAT64_C( -1191.00) },
      { EASYSIMD_FLOAT64_C(   -58.00), EASYSIMD_FLOAT64_C( -1284.00), EASYSIMD_FLOAT64_C(  -224.00), EASYSIMD_FLOAT64_C(   217.00),
        EASYSIMD_FLOAT64_C( -1263.00), EASYSIMD_FLOAT64_C(  -151.00), EASYSIMD_FLOAT64_C(   282.00), EASYSIMD_FLOAT64_C( -1191.00) },
      { EASYSIMD_FLOAT64_C(   -58.00), EASYSIMD_FLOAT64_C( -1284.00), EASYSIMD_FLOAT64_C(  -224.00), EASYSIMD_FLOAT64_C(   216.00),
        EASYSIMD_FLOAT64_C( -1263.00), EASYSIMD_FLOAT64_C(  -151.00), EASYSIMD_FLOAT64_C(   281.00), EASYSIMD_FLOAT64_C( -1191.00) },
      { EASYSIMD_FLOAT64_C(   -58.00), EASYSIMD_FLOAT64_C( -1284.00), EASYSIMD_FLOAT64_C(  -225.00), EASYSIMD_FLOAT64_C(   216.00),
        EASYSIMD_FLOAT64_C( -1264.00), EASYSIMD_FLOAT64_C(  -152.00), EASYSIMD_FLOAT64_C(   282.00), EASYSIMD_FLOAT64_C( -1191.00) } },
    { { EASYSIMD_FLOAT64_C(  -970.53), EASYSIMD_FLOAT64_C(   -35.15), EASYSIMD_FLOAT64_C(  -133.48), EASYSIMD_FLOAT64_C(    16.28),
        EASYSIMD_FLOAT64_C(  -638.13), EASYSIMD_FLOAT64_C(  -732.93), EASYSIMD_FLOAT64_C(  -741.97), EASYSIMD_FLOAT64_C(  -744.67) },
      { EASYSIMD_FLOAT64_C(   571.10), EASYSIMD_FLOAT64_C(  -466.58), EASYSIMD_FLOAT64_C(   -42.71), EASYSIMD_FLOAT64_C(   871.39),
        EASYSIMD_FLOAT64_C(  -416.25), EASYSIMD_FLOAT64_C(  -701.91), EASYSIMD_FLOAT64_C(   332.55), EASYSIMD_FLOAT64_C(   856.98) },
      { EASYSIMD_FLOAT64_C( -1542.00), EASYSIMD_FLOAT64_C(   431.00), EASYSIMD_FLOAT64_C(   -91.00), EASYSIMD_FLOAT64_C(  -855.00),
        EASYSIMD_FLOAT64_C(  -222.00), EASYSIMD_FLOAT64_C(   -31.00), EASYSIMD_FLOAT64_C( -1075.00), EASYSIMD_FLOAT64_C( -1602.00) },
      { EASYSIMD_FLOAT64_C( -1542.00), EASYSIMD_FLOAT64_C(   431.00), EASYSIMD_FLOAT64_C(   -91.00), EASYSIMD_FLOAT64_C(  -856.00),
        EASYSIMD_FLOAT64_C(  -222.00), EASYSIMD_FLOAT64_C(   -32.00), EASYSIMD_FLOAT64_C( -1075.00), EASYSIMD_FLOAT64_C( -1602.00) },
      { EASYSIMD_FLOAT64_C( -1541.00), EASYSIMD_FLOAT64_C(   432.00), EASYSIMD_FLOAT64_C(   -90.00), EASYSIMD_FLOAT64_C(  -855.00),
        EASYSIMD_FLOAT64_C(  -221.00), EASYSIMD_FLOAT64_C(   -31.00), EASYSIMD_FLOAT64_C( -1074.00), EASYSIMD_FLOAT64_C( -1601.00) },
      { EASYSIMD_FLOAT64_C( -1541.00), EASYSIMD_FLOAT64_C(   431.00), EASYSIMD_FLOAT64_C(   -90.00), EASYSIMD_FLOAT64_C(  -855.00),
        EASYSIMD_FLOAT64_C(  -221.00), EASYSIMD_FLOAT64_C(   -31.00), EASYSIMD_FLOAT64_C( -1074.00), EASYSIMD_FLOAT64_C( -1601.00) },
      { EASYSIMD_FLOAT64_C( -1542.00), EASYSIMD_FLOAT64_C(   431.00), EASYSIMD_FLOAT64_C(   -91.00), EASYSIMD_FLOAT64_C(  -855.00),
        EASYSIMD_FLOAT64_C(  -222.00), EASYSIMD_FLOAT64_C(   -31.00), EASYSIMD_FLOAT64_C( -1075.00), EASYSIMD_FLOAT64_C( -1602.00) } },
    { { EASYSIMD_FLOAT64_C(     4.52), EASYSIMD_FLOAT64_C(   -50.12), EASYSIMD_FLOAT64_C(  -649.26), EASYSIMD_FLOAT64_C(   702.67),
        EASYSIMD_FLOAT64_C(   144.89), EASYSIMD_FLOAT64_C(  -205.72), EASYSIMD_FLOAT64_C(   971.80), EASYSIMD_FLOAT64_C(  -523.77) },
      { EASYSIMD_FLOAT64_C(  -214.96), EASYSIMD_FLOAT64_C(   813.67), EASYSIMD_FLOAT64_C(  -246.49), EASYSIMD_FLOAT64_C(  -253.24),
        EASYSIMD_FLOAT64_C(  -839.62), EASYSIMD_FLOAT64_C(   -84.83), EASYSIMD_FLOAT64_C(  -793.10), EASYSIMD_FLOAT64_C(  -810.15) },
      { EASYSIMD_FLOAT64_C(   219.00), EASYSIMD_FLOAT64_C(  -864.00), EASYSIMD_FLOAT64_C(  -403.00), EASYSIMD_FLOAT64_C(   956.00),
        EASYSIMD_FLOAT64_C(   985.00), EASYSIMD_FLOAT64_C(  -121.00), EASYSIMD_FLOAT64_C(  1765.00), EASYSIMD_FLOAT64_C(   286.00) },
      { EASYSIMD_FLOAT64_C(   219.00), EASYSIMD_FLOAT64_C(  -864.00), EASYSIMD_FLOAT64_C(  -403.00), EASYSIMD_FLOAT64_C(   955.00),
        EASYSIMD_FLOAT64_C(   984.00), EASYSIMD_FLOAT64_C(  -121.00), EASYSIMD_FLOAT64_C(  1764.00), EASYSIMD_FLOAT64_C(   286.00) },
      { EASYSIMD_FLOAT64_C(   220.00), EASYSIMD_FLOAT64_C(  -863.00), EASYSIMD_FLOAT64_C(  -402.00), EASYSIMD_FLOAT64_C(   956.00),
        EASYSIMD_FLOAT64_C(   985.00), EASYSIMD_FLOAT64_C(  -120.00), EASYSIMD_FLOAT64_C(  1765.00), EASYSIMD_FLOAT64_C(   287.00) },
      { EASYSIMD_FLOAT64_C(   219.00), EASYSIMD_FLOAT64_C(  -863.00), EASYSIMD_FLOAT64_C(  -402.00), EASYSIMD_FLOAT64_C(   955.00),
        EASYSIMD_FLOAT64_C(   984.00), EASYSIMD_FLOAT64_C(  -120.00), EASYSIMD_FLOAT64_C(  1764.00), EASYSIMD_FLOAT64_C(   286.00) },
      { EASYSIMD_FLOAT64_C(   219.00), EASYSIMD_FLOAT64_C(  -864.00), EASYSIMD_FLOAT64_C(  -403.00), EASYSIMD_FLOAT64_C(   956.00),
        EASYSIMD_FLOAT64_C(   985.00), EASYSIMD_FLOAT64_C(  -121.00), EASYSIMD_FLOAT64_C(  1765.00), EASYSIMD_FLOAT64_C(   286.00) } },
    { { EASYSIMD_FLOAT64_C(   880.02), EASYSIMD_FLOAT64_C(    73.42), EASYSIMD_FLOAT64_C(   206.13), EASYSIMD_FLOAT64_C(  -758.11),
        EASYSIMD_FLOAT64_C(   340.48), EASYSIMD_FLOAT64_C(   464.16), EASYSIMD_FLOAT64_C(  -502.78), EASYSIMD_FLOAT64_C(   -88.42) },
      { EASYSIMD_FLOAT64_C(   997.58), EASYSIMD_FLOAT64_C(   454.50), EASYSIMD_FLOAT64_C(  -217.02), EASYSIMD_FLOAT64_C(  -418.66),
        EASYSIMD_FLOAT64_C(   752.60), EASYSIMD_FLOAT64_C(  -884.47), EASYSIMD_FLOAT64_C(  -561.68), EASYSIMD_FLOAT64_C(  -242.88) },
      { EASYSIMD_FLOAT64_C(  -118.00), EASYSIMD_FLOAT64_C(  -381.00), EASYSIMD_FLOAT64_C(   423.00), EASYSIMD_FLOAT64_C(  -339.00),
        EASYSIMD_FLOAT64_C(  -412.00), EASYSIMD_FLOAT64_C(  1349.00), EASYSIMD_FLOAT64_C(    59.00), EASYSIMD_FLOAT64_C(   154.00) },
      { EASYSIMD_FLOAT64_C(  -118.00), EASYSIMD_FLOAT64_C(  -382.00), EASYSIMD_FLOAT64_C(   423.00), EASYSIMD_FLOAT64_C(  -340.00),
        EASYSIMD_FLOAT64_C(  -413.00), EASYSIMD_FLOAT64_C(  1348.00), EASYSIMD_FLOAT64_C(    58.00), EASYSIMD_FLOAT64_C(   154.00) },
      { EASYSIMD_FLOAT64_C(  -117.00), EASYSIMD_FLOAT64_C(  -381.00), EASYSIMD_FLOAT64_C(   424.00), EASYSIMD_FLOAT64_C(  -339.00),
        EASYSIMD_FLOAT64_C(  -412.00), EASYSIMD_FLOAT64_C(  1349.00), EASYSIMD_FLOAT64_C(    59.00), EASYSIMD_FLOAT64_C(   155.00) },
      { EASYSIMD_FLOAT64_C(  -117.00), EASYSIMD_FLOAT64_C(  -381.00), EASYSIMD_FLOAT64_C(   423.00), EASYSIMD_FLOAT64_C(  -339.00),
        EASYSIMD_FLOAT64_C(  -412.00), EASYSIMD_FLOAT64_C(  1348.00), EASYSIMD_FLOAT64_C(    58.00), EASYSIMD_FLOAT64_C(   154.00) },
      { EASYSIMD_FLOAT64_C(  -118.00), EASYSIMD_FLOAT64_C(  -381.00), EASYSIMD_FLOAT64_C(   423.00), EASYSIMD_FLOAT64_C(  -339.00),
        EASYSIMD_FLOAT64_C(  -412.00), EASYSIMD_FLOAT64_C(  1349.00), EASYSIMD_FLOAT64_C(    59.00), EASYSIMD_FLOAT64_C(   154.00) } },
    { { EASYSIMD_FLOAT64_C(    65.41), EASYSIMD_FLOAT64_C(  -210.94), EASYSIMD_FLOAT64_C(  -540.21), EASYSIMD_FLOAT64_C(  -789.70),
        EASYSIMD_FLOAT64_C(   583.34), EASYSIMD_FLOAT64_C(  -568.42), EASYSIMD_FLOAT64_C(  -313.47), EASYSIMD_FLOAT64_C(  -631.63) },
      { EASYSIMD_FLOAT64_C(  -754.75), EASYSIMD_FLOAT64_C(   440.04), EASYSIMD_FLOAT64_C(   115.14), EASYSIMD_FLOAT64_C(  -594.38),
        EASYSIMD_FLOAT64_C(  -644.79), EASYSIMD_FLOAT64_C(   322.03), EASYSIMD_FLOAT64_C(  -404.53), EASYSIMD_FLOAT64_C(  -764.77) },
      { EASYSIMD_FLOAT64_C(   820.00), EASYSIMD_FLOAT64_C(  -651.00), EASYSIMD_FLOAT64_C(  -655.00), EASYSIMD_FLOAT64_C(  -195.00),
        EASYSIMD_FLOAT64_C(  1228.00), EASYSIMD_FLOAT64_C(  -890.00), EASYSIMD_FLOAT64_C(    91.00), EASYSIMD_FLOAT64_C(   133.00) },
      { EASYSIMD_FLOAT64_C(   820.00), EASYSIMD_FLOAT64_C(  -651.00), EASYSIMD_FLOAT64_C(  -656.00), EASYSIMD_FLOAT64_C(  -196.00),
        EASYSIMD_FLOAT64_C(  1228.00), EASYSIMD_FLOAT64_C(  -891.00), EASYSIMD_FLOAT64_C(    91.00), EASYSIMD_FLOAT64_C(   133.00) },
      { EASYSIMD_FLOAT64_C(   821.00), EASYSIMD_FLOAT64_C(  -650.00), EASYSIMD_FLOAT64_C(  -655.00), EASYSIMD_FLOAT64_C(  -195.00),
        EASYSIMD_FLOAT64_C(  1229.00), EASYSIMD_FLOAT64_C(  -890.00), EASYSIMD_FLOAT64_C(    92.00), EASYSIMD_FLOAT64_C(   134.00) },
      { EASYSIMD_FLOAT64_C(   820.00), EASYSIMD_FLOAT64_C(  -650.00), EASYSIMD_FLOAT64_C(  -655.00), EASYSIMD_FLOAT64_C(  -195.00),
        EASYSIMD_FLOAT64_C(  1228.00), EASYSIMD_FLOAT64_C(  -890.00), EASYSIMD_FLOAT64_C(    91.00), EASYSIMD_FLOAT64_C(   133.00) },
      { EASYSIMD_FLOAT64_C(   820.00), EASYSIMD_FLOAT64_C(  -651.00), EASYSIMD_FLOAT64_C(  -655.00), EASYSIMD_FLOAT64_C(  -195.00),
        EASYSIMD_FLOAT64_C(  1228.00), EASYSIMD_FLOAT64_C(  -890.00), EASYSIMD_FLOAT64_C(    91.00), EASYSIMD_FLOAT64_C(   133.00) } },
    { { EASYSIMD_FLOAT64_C(  -604.55), EASYSIMD_FLOAT64_C(   801.61), EASYSIMD_FLOAT64_C(  -522.88), EASYSIMD_FLOAT64_C(   735.93),
        EASYSIMD_FLOAT64_C(   265.77), EASYSIMD_FLOAT64_C(   -25.66), EASYSIMD_FLOAT64_C(  -352.48), EASYSIMD_FLOAT64_C(   263.35) },
      { EASYSIMD_FLOAT64_C(  -571.16), EASYSIMD_FLOAT64_C(   430.49), EASYSIMD_FLOAT64_C(   844.69), EASYSIMD_FLOAT64_C(  -818.56),
        EASYSIMD_FLOAT64_C(   546.02), EASYSIMD_FLOAT64_C(  -717.00), EASYSIMD_FLOAT64_C(   -61.44), EASYSIMD_FLOAT64_C(  -388.57) },
      { EASYSIMD_FLOAT64_C(   -33.00), EASYSIMD_FLOAT64_C(   371.00), EASYSIMD_FLOAT64_C( -1368.00), EASYSIMD_FLOAT64_C(  1554.00),
        EASYSIMD_FLOAT64_C(  -280.00), EASYSIMD_FLOAT64_C(   691.00), EASYSIMD_FLOAT64_C(  -291.00), EASYSIMD_FLOAT64_C(   652.00) },
      { EASYSIMD_FLOAT64_C(   -34.00), EASYSIMD_FLOAT64_C(   371.00), EASYSIMD_FLOAT64_C( -1368.00), EASYSIMD_FLOAT64_C(  1554.00),
        EASYSIMD_FLOAT64_C(  -281.00), EASYSIMD_FLOAT64_C(   691.00), EASYSIMD_FLOAT64_C(  -292.00), EASYSIMD_FLOAT64_C(   651.00) },
      { EASYSIMD_FLOAT64_C(   -33.00), EASYSIMD_FLOAT64_C(   372.00), EASYSIMD_FLOAT64_C( -1367.00), EASYSIMD_FLOAT64_C(  1555.00),
        EASYSIMD_FLOAT64_C(  -280.00), EASYSIMD_FLOAT64_C(   692.00), EASYSIMD_FLOAT64_C(  -291.00), EASYSIMD_FLOAT64_C(   652.00) },
      { EASYSIMD_FLOAT64_C(   -33.00), EASYSIMD_FLOAT64_C(   371.00), EASYSIMD_FLOAT64_C( -1367.00), EASYSIMD_FLOAT64_C(  1554.00),
        EASYSIMD_FLOAT64_C(  -280.00), EASYSIMD_FLOAT64_C(   691.00), EASYSIMD_FLOAT64_C(  -291.00), EASYSIMD_FLOAT64_C(   651.00) },
      { EASYSIMD_FLOAT64_C(   -33.00), EASYSIMD_FLOAT64_C(   371.00), EASYSIMD_FLOAT64_C( -1368.00), EASYSIMD_FLOAT64_C(  1554.00),
        EASYSIMD_FLOAT64_C(  -280.00), EASYSIMD_FLOAT64_C(   691.00), EASYSIMD_FLOAT64_C(  -291.00), EASYSIMD_FLOAT64_C(   652.00) } }
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

    r = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512d_close(r, nearest_inf, 1);

    r = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512d_close(r, neg_inf, 1);

    r = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512d_close(r, pos_inf, 1);

    r = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512d_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_sub_round_pd");
    easysimd_assert_m512d_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512d nearest_inf = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512d neg_inf = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512d pos_inf = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512d zero = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512d direction = easysimd_mm512_sub_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

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
test_easysimd_mm512_mask_sub_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    int16_t arr_src[32];
    easysimd__mmask32 k;
    int16_t arr_a[32];
    int16_t arr_b[32];
    int16_t arr_r[32];
  } test_vec[] = {
    { { -INT16_C( 28514),  INT16_C( 29035), -INT16_C(   827), -INT16_C(  6146), -INT16_C( 23119), -INT16_C( 28860),  INT16_C(   683),  INT16_C( 10458),
        -INT16_C(  1052), -INT16_C(  2985),  INT16_C( 29089),  INT16_C( 24062), -INT16_C( 16058),  INT16_C(   290), -INT16_C(  2644),  INT16_C( 19167),
         INT16_C( 19078),  INT16_C( 19387), -INT16_C( 17849), -INT16_C(  1997),  INT16_C( 30559),  INT16_C(  2695),  INT16_C( 24953),  INT16_C( 24114),
        -INT16_C( 30372), -INT16_C(   686),  INT16_C( 20730),  INT16_C( 16474),  INT16_C( 31762), -INT16_C( 16830),  INT16_C(  8561), -INT16_C(  2295) },
      UINT32_C(3007562860),
      {  INT16_C( 30334), -INT16_C(  8789),  INT16_C( 13037),  INT16_C( 26599),  INT16_C(  6547), -INT16_C(  4155),  INT16_C(  6050), -INT16_C( 25108),
         INT16_C( 18024),  INT16_C( 31453),  INT16_C(  8130),  INT16_C( 13112),  INT16_C( 16705), -INT16_C( 21205),  INT16_C( 28166), -INT16_C( 31648),
         INT16_C(  3044), -INT16_C( 11934),  INT16_C( 18749), -INT16_C( 12232), -INT16_C(   669),  INT16_C(  1471), -INT16_C( 21739),  INT16_C( 32162),
        -INT16_C( 32527), -INT16_C( 19465),  INT16_C( 12191), -INT16_C(  7962),  INT16_C(  4465),  INT16_C( 30605), -INT16_C(  4737),  INT16_C( 25595) },
      {  INT16_C( 24056),  INT16_C( 13621),  INT16_C( 28071),  INT16_C(  2565), -INT16_C( 15253), -INT16_C( 32753), -INT16_C( 19857),  INT16_C( 24829),
        -INT16_C(  3022), -INT16_C( 12013), -INT16_C(  1501), -INT16_C( 27470),  INT16_C( 16139), -INT16_C( 29941),  INT16_C(  1837),  INT16_C(  9710),
         INT16_C(  9060),  INT16_C(  2907),  INT16_C( 24721), -INT16_C(  1003),  INT16_C(  9509), -INT16_C( 27524),  INT16_C( 31191),  INT16_C(  2549),
         INT16_C(  2157), -INT16_C( 28454), -INT16_C( 29694),  INT16_C(  3621),  INT16_C( 12492), -INT16_C(  1639), -INT16_C( 30921), -INT16_C( 25570) },
      { -INT16_C( 28514),  INT16_C( 29035), -INT16_C( 15034),  INT16_C( 24034), -INT16_C( 23119),  INT16_C( 28598),  INT16_C( 25907),  INT16_C( 10458),
        -INT16_C(  1052), -INT16_C(  2985),  INT16_C(  9631),  INT16_C( 24062), -INT16_C( 16058),  INT16_C(   290),  INT16_C( 26329),  INT16_C( 24178),
        -INT16_C(  6016), -INT16_C( 14841), -INT16_C( 17849), -INT16_C(  1997),  INT16_C( 30559),  INT16_C(  2695),  INT16_C( 12606),  INT16_C( 24114),
         INT16_C( 30852),  INT16_C(  8989),  INT16_C( 20730),  INT16_C( 16474), -INT16_C(  8027),  INT16_C( 32244),  INT16_C(  8561), -INT16_C( 14371) } },
    { {  INT16_C( 31147),  INT16_C( 15527), -INT16_C( 16934), -INT16_C(   200), -INT16_C( 19230), -INT16_C( 18029), -INT16_C( 30675), -INT16_C( 25918),
        -INT16_C( 25455), -INT16_C( 27862),  INT16_C( 20265), -INT16_C(  2655),  INT16_C( 14976), -INT16_C( 18450),  INT16_C(  3266),  INT16_C( 27987),
        -INT16_C(  1146),  INT16_C( 24745), -INT16_C(  7752), -INT16_C( 26017), -INT16_C(  3435), -INT16_C( 15789),  INT16_C(  5499),  INT16_C(  3164),
        -INT16_C( 31055), -INT16_C(  9569),  INT16_C( 16854),  INT16_C( 22223), -INT16_C( 17029),  INT16_C( 15629),  INT16_C( 25034),  INT16_C( 20650) },
      UINT32_C( 347099996),
      {  INT16_C(  3892), -INT16_C( 13906),  INT16_C(   257),  INT16_C( 31883), -INT16_C(  6378), -INT16_C( 14456),  INT16_C( 10350),  INT16_C( 17570),
         INT16_C( 29033), -INT16_C(  7014), -INT16_C( 22737), -INT16_C(  1758), -INT16_C( 13304),  INT16_C( 25673), -INT16_C(  1760),  INT16_C( 21624),
         INT16_C(  9736),  INT16_C(  2334), -INT16_C( 22233),  INT16_C( 15750),  INT16_C(  3729), -INT16_C(   251), -INT16_C( 22730), -INT16_C( 24765),
        -INT16_C(  8936),  INT16_C( 18308), -INT16_C( 22908), -INT16_C( 29376), -INT16_C( 30350), -INT16_C( 27919),  INT16_C( 27266), -INT16_C( 29977) },
      {  INT16_C(  1424), -INT16_C( 18284),  INT16_C(  6830),  INT16_C( 16373), -INT16_C(  1496),  INT16_C( 24382), -INT16_C( 32351), -INT16_C( 17666),
        -INT16_C( 32162), -INT16_C(  7423),  INT16_C( 16936), -INT16_C( 25744),  INT16_C( 25035),  INT16_C( 20013),  INT16_C(  5323),  INT16_C( 23768),
         INT16_C( 27673), -INT16_C( 14316),  INT16_C(  2438), -INT16_C( 20729),  INT16_C( 17924), -INT16_C( 23282),  INT16_C(  3271),  INT16_C(  9823),
         INT16_C( 24975), -INT16_C( 18679),  INT16_C( 31139),  INT16_C( 28242), -INT16_C( 32550), -INT16_C( 22852), -INT16_C( 27244), -INT16_C( 20990) },
      {  INT16_C( 31147),  INT16_C( 15527), -INT16_C(  6573),  INT16_C( 15510), -INT16_C(  4882), -INT16_C( 18029), -INT16_C( 22835), -INT16_C( 25918),
        -INT16_C(  4341),  INT16_C(   409),  INT16_C( 20265), -INT16_C(  2655),  INT16_C( 27197), -INT16_C( 18450), -INT16_C(  7083),  INT16_C( 27987),
        -INT16_C(  1146),  INT16_C( 24745), -INT16_C(  7752), -INT16_C( 26017), -INT16_C( 14195),  INT16_C( 23031),  INT16_C(  5499),  INT16_C( 30948),
        -INT16_C( 31055), -INT16_C(  9569),  INT16_C( 11489),  INT16_C( 22223),  INT16_C(  2200),  INT16_C( 15629),  INT16_C( 25034),  INT16_C( 20650) } },
    { {  INT16_C(  5633), -INT16_C( 30602),  INT16_C( 32031),  INT16_C(  9015),  INT16_C( 17859), -INT16_C( 29751),  INT16_C( 10321), -INT16_C(  8015),
        -INT16_C( 17783),  INT16_C( 11416), -INT16_C(  5581),  INT16_C(  3483),  INT16_C( 22378), -INT16_C(    77), -INT16_C( 18964), -INT16_C(  4435),
         INT16_C(  9163), -INT16_C(  5258), -INT16_C( 21088),  INT16_C( 25614), -INT16_C( 10254),  INT16_C( 17391), -INT16_C( 24576), -INT16_C( 30428),
        -INT16_C( 17318), -INT16_C( 29258),  INT16_C( 20902),  INT16_C(  4506),  INT16_C( 20136), -INT16_C( 27376), -INT16_C( 17149), -INT16_C( 12413) },
      UINT32_C(2159737312),
      { -INT16_C( 14170), -INT16_C( 26396), -INT16_C( 11360), -INT16_C( 24357), -INT16_C(   141), -INT16_C( 13015), -INT16_C(  8261),  INT16_C( 25178),
        -INT16_C(  2768), -INT16_C(  9869), -INT16_C( 31933),  INT16_C( 18030), -INT16_C(  3776),  INT16_C(  8213), -INT16_C( 12310), -INT16_C( 28512),
        -INT16_C( 31336),  INT16_C( 14376),  INT16_C(   856), -INT16_C( 13096),  INT16_C(   259), -INT16_C( 16743), -INT16_C(  2847),  INT16_C(  4384),
        -INT16_C( 27671),  INT16_C( 11498),  INT16_C( 22550),  INT16_C( 22130), -INT16_C( 30647),  INT16_C( 13174),  INT16_C(  5975), -INT16_C(  4157) },
      { -INT16_C(  5220), -INT16_C(  3033), -INT16_C(    17), -INT16_C(  3392),  INT16_C( 23041), -INT16_C(  7504), -INT16_C( 11954),  INT16_C( 14323),
        -INT16_C(  8604),  INT16_C( 31587), -INT16_C( 10954), -INT16_C( 32559),  INT16_C( 18525), -INT16_C( 19021),  INT16_C( 30559), -INT16_C(  1116),
        -INT16_C( 13214),  INT16_C( 20975), -INT16_C( 20277), -INT16_C( 13245), -INT16_C(  3062),  INT16_C( 22702), -INT16_C( 23867),  INT16_C( 10639),
        -INT16_C(  3456), -INT16_C( 18780),  INT16_C( 30407),  INT16_C(  9526), -INT16_C(  5442),  INT16_C(  7642),  INT16_C( 32353), -INT16_C( 15592) },
      {  INT16_C(  5633), -INT16_C( 30602),  INT16_C( 32031),  INT16_C(  9015),  INT16_C( 17859), -INT16_C(  5511),  INT16_C(  3693),  INT16_C( 10855),
         INT16_C(  5836),  INT16_C( 11416), -INT16_C(  5581), -INT16_C( 14947), -INT16_C( 22301),  INT16_C( 27234),  INT16_C( 22667), -INT16_C( 27396),
         INT16_C(  9163), -INT16_C(  6599), -INT16_C( 21088),  INT16_C(   149),  INT16_C(  3321),  INT16_C( 26091), -INT16_C( 24576), -INT16_C(  6255),
        -INT16_C( 17318), -INT16_C( 29258),  INT16_C( 20902),  INT16_C(  4506),  INT16_C( 20136), -INT16_C( 27376), -INT16_C( 17149),  INT16_C( 11435) } },
    { {  INT16_C(  1866),  INT16_C(  5653),  INT16_C( 22711), -INT16_C( 15902), -INT16_C( 28340),  INT16_C(  4377), -INT16_C( 22477), -INT16_C( 19653),
        -INT16_C(  8294),  INT16_C( 25193), -INT16_C( 24491),  INT16_C(  4999),  INT16_C( 24970), -INT16_C(  5328),  INT16_C( 18655),  INT16_C( 10926),
        -INT16_C( 15536),  INT16_C(  1856),  INT16_C(  8732),  INT16_C( 26825), -INT16_C(  7501), -INT16_C(  6534), -INT16_C( 19061),  INT16_C(  9625),
         INT16_C(   916), -INT16_C(  5497),  INT16_C(  3747),  INT16_C( 11773),  INT16_C( 11887),  INT16_C( 20248), -INT16_C( 14730), -INT16_C( 14727) },
      UINT32_C(2798565770),
      { -INT16_C( 26661), -INT16_C( 28914), -INT16_C( 30599),  INT16_C(  1141),  INT16_C(  3901), -INT16_C( 11734), -INT16_C( 20206), -INT16_C( 19012),
        -INT16_C( 17984),  INT16_C( 12258), -INT16_C(  1305),  INT16_C( 24190), -INT16_C(  2112),  INT16_C( 18980), -INT16_C(  3408), -INT16_C( 29456),
        -INT16_C(   119),  INT16_C(   795), -INT16_C( 28537), -INT16_C( 15097),  INT16_C( 12703), -INT16_C( 20073),  INT16_C( 21475), -INT16_C( 23706),
         INT16_C( 18444), -INT16_C(  2862),  INT16_C( 20802),  INT16_C(   850),  INT16_C( 30280), -INT16_C(  1715),  INT16_C( 15977), -INT16_C(  3451) },
      { -INT16_C( 24515), -INT16_C( 15115), -INT16_C(   720), -INT16_C( 12151),  INT16_C(  8238),  INT16_C(  4481), -INT16_C(  6029), -INT16_C( 32588),
        -INT16_C( 30928),  INT16_C( 29556), -INT16_C( 14632),  INT16_C(  8310), -INT16_C( 15556), -INT16_C( 23271), -INT16_C( 25087),  INT16_C( 16024),
        -INT16_C( 29378),  INT16_C( 28419), -INT16_C( 29558), -INT16_C( 18113), -INT16_C( 16211),  INT16_C(  8394),  INT16_C( 32680), -INT16_C(  9824),
         INT16_C(  5126), -INT16_C(  8628), -INT16_C( 15654),  INT16_C(  6142),  INT16_C(  6277), -INT16_C( 30788),  INT16_C( 21686), -INT16_C(  2619) },
      {  INT16_C(  1866), -INT16_C( 13799),  INT16_C( 22711),  INT16_C( 13292), -INT16_C( 28340),  INT16_C(  4377), -INT16_C( 22477),  INT16_C( 13576),
         INT16_C( 12944),  INT16_C( 25193), -INT16_C( 24491),  INT16_C( 15880),  INT16_C( 13444), -INT16_C( 23285),  INT16_C( 18655),  INT16_C( 20056),
        -INT16_C( 15536), -INT16_C( 27624),  INT16_C(  1021),  INT16_C(  3016), -INT16_C(  7501), -INT16_C(  6534), -INT16_C( 11205), -INT16_C( 13882),
         INT16_C(   916),  INT16_C(  5766), -INT16_C( 29080),  INT16_C( 11773),  INT16_C( 11887),  INT16_C( 29073), -INT16_C( 14730), -INT16_C(   832) } },
    { { -INT16_C( 14110),  INT16_C( 27748), -INT16_C( 23723),  INT16_C(   549), -INT16_C(  3997),  INT16_C(  3106), -INT16_C( 15505),  INT16_C( 30181),
         INT16_C( 12759), -INT16_C( 19885),  INT16_C( 20979),  INT16_C( 30921), -INT16_C( 31383),  INT16_C(  8447), -INT16_C( 14886), -INT16_C( 17387),
         INT16_C( 31117), -INT16_C(  7640),  INT16_C( 19996),  INT16_C( 32740),  INT16_C(  1854), -INT16_C( 21109),  INT16_C( 28874), -INT16_C( 24286),
         INT16_C( 30113), -INT16_C( 27565),  INT16_C(  7366),  INT16_C( 12301),  INT16_C(  3234),  INT16_C( 31824),  INT16_C( 26065),  INT16_C( 24376) },
      UINT32_C(4198588638),
      {  INT16_C(  9902), -INT16_C(  4999),  INT16_C(  1325), -INT16_C(  2151), -INT16_C( 17547),  INT16_C(  6040), -INT16_C(  5072), -INT16_C(  2133),
        -INT16_C( 18424), -INT16_C( 21977),  INT16_C( 30661), -INT16_C( 27098),  INT16_C( 24284), -INT16_C( 17675),  INT16_C( 14271),  INT16_C( 28084),
         INT16_C( 11613), -INT16_C( 30118), -INT16_C(  3278), -INT16_C( 22399),  INT16_C(  6575), -INT16_C(  8257),  INT16_C( 27141),  INT16_C(  3798),
        -INT16_C(   733), -INT16_C(  5960), -INT16_C(  8332),  INT16_C( 20606),  INT16_C( 29757), -INT16_C(  1014), -INT16_C( 16725),  INT16_C(  2154) },
      { -INT16_C( 15124),  INT16_C(  7826),  INT16_C(  5047),  INT16_C( 26310), -INT16_C( 31444),  INT16_C( 12870),  INT16_C(  7408),  INT16_C(  4928),
        -INT16_C(  2022), -INT16_C( 28933),  INT16_C( 31191),  INT16_C(  5599), -INT16_C(  5651), -INT16_C( 26607),  INT16_C( 31656), -INT16_C( 27488),
         INT16_C( 12863), -INT16_C(  2126),  INT16_C( 31045),  INT16_C( 29277), -INT16_C( 23554), -INT16_C(  4444), -INT16_C(  6976), -INT16_C(  9727),
        -INT16_C(   804), -INT16_C( 19352),  INT16_C( 18294),  INT16_C( 25545), -INT16_C(  9679), -INT16_C(  9732), -INT16_C( 25514), -INT16_C( 27283) },
      { -INT16_C( 14110), -INT16_C( 12825), -INT16_C(  3722), -INT16_C( 28461),  INT16_C( 13897),  INT16_C(  3106), -INT16_C( 12480), -INT16_C(  7061),
         INT16_C( 12759), -INT16_C( 19885),  INT16_C( 20979),  INT16_C( 30921), -INT16_C( 31383),  INT16_C(  8932), -INT16_C( 17385), -INT16_C( 17387),
        -INT16_C(  1250), -INT16_C(  7640),  INT16_C( 19996),  INT16_C( 32740),  INT16_C(  1854), -INT16_C( 21109), -INT16_C( 31419), -INT16_C( 24286),
         INT16_C( 30113),  INT16_C( 13392),  INT16_C(  7366), -INT16_C(  4939), -INT16_C( 26100),  INT16_C(  8718),  INT16_C(  8789),  INT16_C( 29437) } },
    { {  INT16_C(  8143),  INT16_C(  5260), -INT16_C(  5480), -INT16_C( 26746),  INT16_C( 10893),  INT16_C( 19845), -INT16_C( 30962), -INT16_C(  5337),
        -INT16_C( 28541), -INT16_C(  1633),  INT16_C( 26839),  INT16_C(  2141),  INT16_C( 22850), -INT16_C( 26399),  INT16_C( 20213), -INT16_C( 15314),
        -INT16_C( 17810),  INT16_C(  1753),  INT16_C( 24484),  INT16_C( 12957),  INT16_C(  9098), -INT16_C( 26497), -INT16_C( 22614),  INT16_C( 11651),
         INT16_C(  8759),  INT16_C(  3623), -INT16_C( 31606), -INT16_C( 13033), -INT16_C(  1827), -INT16_C( 11675), -INT16_C( 27833), -INT16_C( 19049) },
      UINT32_C(4072370254),
      {  INT16_C( 22991),  INT16_C( 22820), -INT16_C( 23428),  INT16_C(  9970),  INT16_C( 30027), -INT16_C( 32173),  INT16_C( 31384),  INT16_C(  8848),
        -INT16_C( 22530), -INT16_C(  9233),  INT16_C( 21920), -INT16_C(  6226),  INT16_C( 17896),  INT16_C( 13980),  INT16_C( 22453), -INT16_C( 31703),
         INT16_C( 19888),  INT16_C( 11486), -INT16_C( 12047),  INT16_C( 15442), -INT16_C( 22971), -INT16_C(  8770),  INT16_C( 20256),  INT16_C(  7936),
        -INT16_C(  4106), -INT16_C( 26886), -INT16_C( 22460),  INT16_C( 11645),  INT16_C(  6637), -INT16_C( 23965), -INT16_C( 29583),  INT16_C(  8487) },
      {  INT16_C(  1498), -INT16_C( 13490), -INT16_C( 24363),  INT16_C(  6664), -INT16_C( 14778),  INT16_C( 26616), -INT16_C(  2027),  INT16_C(  3206),
        -INT16_C( 32537),  INT16_C( 11426),  INT16_C(  8233),  INT16_C(  5721), -INT16_C( 17351), -INT16_C( 21831), -INT16_C(  8119),  INT16_C(  9164),
         INT16_C(  6885), -INT16_C( 17682), -INT16_C(  2374),  INT16_C(   468), -INT16_C( 13123), -INT16_C( 11672), -INT16_C(  4412), -INT16_C( 21282),
        -INT16_C( 32402), -INT16_C( 26664),  INT16_C( 12705), -INT16_C(  9554),  INT16_C( 26605),  INT16_C( 13957),  INT16_C( 20807),  INT16_C( 11353) },
      {  INT16_C(  8143), -INT16_C( 29226),  INT16_C(   935),  INT16_C(  3306),  INT16_C( 10893),  INT16_C( 19845), -INT16_C( 32125), -INT16_C(  5337),
        -INT16_C( 28541), -INT16_C(  1633),  INT16_C( 26839),  INT16_C(  2141), -INT16_C( 30289), -INT16_C( 29725),  INT16_C( 30572), -INT16_C( 15314),
         INT16_C( 13003),  INT16_C( 29168),  INT16_C( 24484),  INT16_C( 14974), -INT16_C(  9848),  INT16_C(  2902), -INT16_C( 22614),  INT16_C( 29218),
         INT16_C(  8759), -INT16_C(   222), -INT16_C( 31606), -INT16_C( 13033), -INT16_C( 19968),  INT16_C( 27614),  INT16_C( 15146), -INT16_C(  2866) } },
    { {  INT16_C( 18539),  INT16_C(  9702), -INT16_C( 17858), -INT16_C(  1242), -INT16_C( 29049),  INT16_C( 19406), -INT16_C( 21380), -INT16_C(  5129),
        -INT16_C( 12499), -INT16_C( 12670),  INT16_C( 12288), -INT16_C(  4439),  INT16_C( 11927), -INT16_C(  8668),  INT16_C( 32383), -INT16_C(  5622),
        -INT16_C(  3898),  INT16_C(  1039),  INT16_C( 13995),  INT16_C( 12800), -INT16_C( 12604),  INT16_C( 16765),  INT16_C( 30074), -INT16_C( 22484),
        -INT16_C( 20924),  INT16_C( 17782),  INT16_C(  8159),  INT16_C( 30259),  INT16_C( 22349), -INT16_C( 13227),  INT16_C( 24533), -INT16_C( 25674) },
      UINT32_C(4221617744),
      { -INT16_C( 24324), -INT16_C( 16339), -INT16_C( 21906), -INT16_C(  6143),  INT16_C( 11551),  INT16_C( 25744),  INT16_C(  2012), -INT16_C( 17495),
        -INT16_C(  9178),  INT16_C( 29745), -INT16_C( 31181),  INT16_C(  2368), -INT16_C(  2074),  INT16_C( 13988),  INT16_C( 17597), -INT16_C( 18127),
         INT16_C( 24292),  INT16_C( 21113),  INT16_C( 31496),  INT16_C( 10299), -INT16_C( 13400), -INT16_C( 31604),  INT16_C( 13778), -INT16_C(  1729),
         INT16_C( 28945),  INT16_C( 17517), -INT16_C( 21001), -INT16_C(  8883), -INT16_C(  3420),  INT16_C( 24851),  INT16_C( 17462),  INT16_C(  6938) },
      { -INT16_C( 27486), -INT16_C( 21651), -INT16_C( 22513), -INT16_C( 18477),  INT16_C( 24436),  INT16_C( 17980),  INT16_C( 31636), -INT16_C( 23233),
        -INT16_C( 21268), -INT16_C(  6935),  INT16_C( 14170), -INT16_C(   319), -INT16_C( 10967),  INT16_C( 24416),  INT16_C( 31257), -INT16_C( 17286),
        -INT16_C(  6130),  INT16_C(  7527),  INT16_C( 14992),  INT16_C(  1237),  INT16_C(  4505),  INT16_C( 11595), -INT16_C( 30068),  INT16_C( 31186),
        -INT16_C( 17609), -INT16_C( 28323),  INT16_C(  7922),  INT16_C(  7055), -INT16_C(  4109),  INT16_C(  3451), -INT16_C(  2710),  INT16_C( 30921) },
      {  INT16_C( 18539),  INT16_C(  9702), -INT16_C( 17858), -INT16_C(  1242), -INT16_C( 12885),  INT16_C( 19406), -INT16_C( 29624), -INT16_C(  5129),
        -INT16_C( 12499), -INT16_C( 28856),  INT16_C( 20185), -INT16_C(  4439),  INT16_C( 11927), -INT16_C(  8668), -INT16_C( 13660), -INT16_C(   841),
        -INT16_C(  3898),  INT16_C(  1039),  INT16_C( 13995),  INT16_C( 12800), -INT16_C( 12604),  INT16_C( 22337),  INT16_C( 30074),  INT16_C( 32621),
        -INT16_C( 18982), -INT16_C( 19696),  INT16_C(  8159), -INT16_C( 15938),  INT16_C(   689),  INT16_C( 21400),  INT16_C( 20172), -INT16_C( 23983) } },
    { {  INT16_C( 12509),  INT16_C( 28310),  INT16_C( 27498),  INT16_C(   882), -INT16_C( 17028),  INT16_C(  2096),  INT16_C(   584),  INT16_C( 32641),
        -INT16_C(  8515), -INT16_C( 20464), -INT16_C( 24579), -INT16_C(  3893),  INT16_C( 18063), -INT16_C(  1539), -INT16_C( 14788),  INT16_C(  6513),
         INT16_C(  2038),  INT16_C( 24711), -INT16_C(  1422), -INT16_C(  4509), -INT16_C( 27721), -INT16_C(     9),  INT16_C( 30869),  INT16_C( 21374),
        -INT16_C( 29097),  INT16_C( 21507), -INT16_C( 12754), -INT16_C( 17084),  INT16_C( 16917),  INT16_C( 20918),  INT16_C(  9992), -INT16_C(   150) },
      UINT32_C(2707419695),
      { -INT16_C( 15380), -INT16_C( 23664), -INT16_C( 30890), -INT16_C(  4957),  INT16_C(  8703),  INT16_C( 22079),  INT16_C( 17072), -INT16_C(  8534),
        -INT16_C(  4336),  INT16_C(  9627),  INT16_C( 20785),  INT16_C( 14710), -INT16_C(  7816), -INT16_C( 22728), -INT16_C( 26413), -INT16_C( 16567),
        -INT16_C(  9893), -INT16_C( 20126),  INT16_C(  1376),  INT16_C( 24477), -INT16_C(  9177), -INT16_C( 10314),  INT16_C( 24606),  INT16_C( 12213),
         INT16_C( 20559), -INT16_C( 32684), -INT16_C( 13407),  INT16_C(  6586), -INT16_C(  3412),  INT16_C( 32705),  INT16_C(  2698), -INT16_C(  6850) },
      { -INT16_C( 24349),  INT16_C( 17303),  INT16_C( 13478), -INT16_C( 12894),  INT16_C( 22545),  INT16_C( 12196),  INT16_C( 22969),  INT16_C(  2142),
        -INT16_C( 19543),  INT16_C( 19081),  INT16_C( 17278),  INT16_C( 10851),  INT16_C(  9269), -INT16_C( 16215), -INT16_C(  6354),  INT16_C(  4517),
         INT16_C( 15495),  INT16_C( 11604), -INT16_C(  2191), -INT16_C( 32006), -INT16_C( 25009),  INT16_C(  2225),  INT16_C(  4343), -INT16_C( 24559),
        -INT16_C( 25917),  INT16_C( 16874),  INT16_C( 20189),  INT16_C(  4715),  INT16_C(  5234), -INT16_C( 24110),  INT16_C( 30971), -INT16_C( 32078) },
      {  INT16_C(  8969),  INT16_C( 24569),  INT16_C( 21168),  INT16_C(  7937), -INT16_C( 17028),  INT16_C(  9883),  INT16_C(   584),  INT16_C( 32641),
        -INT16_C(  8515), -INT16_C(  9454), -INT16_C( 24579), -INT16_C(  3893), -INT16_C( 17085), -INT16_C(  6513), -INT16_C( 20059), -INT16_C( 21084),
        -INT16_C( 25388), -INT16_C( 31730),  INT16_C(  3567), -INT16_C(  9053),  INT16_C( 15832), -INT16_C(     9),  INT16_C( 20263),  INT16_C( 21374),
        -INT16_C( 19060),  INT16_C( 21507), -INT16_C( 12754), -INT16_C( 17084),  INT16_C( 16917), -INT16_C(  8721),  INT16_C(  9992),  INT16_C( 25228) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].arr_a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].arr_b);
    easysimd__m512i r = easysimd_mm512_loadu_epi16(test_vec[i].arr_r);
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].arr_src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_mask_sub_epi16(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sub_epi16");
    easysimd_assert_m512i_i32(ret, ==, r);
  }

#else 
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    easysimd__m512i r = easysimd_mm512_mask_sub_epi16(src, k, a, b);

    easysimd_test_x86_write_i16x32(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif

  return 0;
}

static int
test_easysimd_mm512_maskz_sub_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    easysimd__mmask32 k;
    int16_t arr_a[32];
    int16_t arr_b[32];
    int16_t arr_r[32];
  } test_vec[] = {
    { UINT32_C(1501289769),
      {  INT16_C( 22476),  INT16_C(  9639), -INT16_C( 21197),  INT16_C( 29938), -INT16_C( 25045), -INT16_C( 21785), -INT16_C( 29283),  INT16_C( 13668),
         INT16_C(  7552),  INT16_C(  1701),  INT16_C( 18097), -INT16_C(  2218), -INT16_C( 10819), -INT16_C(  6623), -INT16_C( 25421),  INT16_C( 32575),
        -INT16_C(  6413),  INT16_C(  9892), -INT16_C( 26732), -INT16_C( 16486), -INT16_C( 32459), -INT16_C( 11671), -INT16_C( 13041), -INT16_C( 28921),
        -INT16_C( 21013), -INT16_C( 25451), -INT16_C(  5133), -INT16_C( 20333), -INT16_C( 19007),  INT16_C( 29847), -INT16_C( 10671),  INT16_C( 17907) },
      { -INT16_C( 26435),  INT16_C( 20843),  INT16_C(  1583),  INT16_C( 25616),  INT16_C( 31111), -INT16_C( 27082),  INT16_C( 15687),  INT16_C( 12837),
        -INT16_C( 17430), -INT16_C(  8754),  INT16_C( 24998),  INT16_C( 26510),  INT16_C(  9494),  INT16_C( 26843), -INT16_C( 12293), -INT16_C( 18259),
         INT16_C(  6247), -INT16_C( 27127),  INT16_C(  6430), -INT16_C( 22790),  INT16_C( 12435), -INT16_C(  9668),  INT16_C( 25197),  INT16_C( 22540),
        -INT16_C(  9699), -INT16_C( 15563), -INT16_C( 15557),  INT16_C( 21035),  INT16_C(  1768), -INT16_C(  6982),  INT16_C( 26581),  INT16_C( 15516) },
      { -INT16_C( 16625),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4322),  INT16_C(     0),  INT16_C(  5297),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 24982),  INT16_C(     0), -INT16_C(  6901), -INT16_C( 28728), -INT16_C( 20313),  INT16_C(     0), -INT16_C( 13128), -INT16_C( 14702),
        -INT16_C( 12660), -INT16_C( 28517),  INT16_C(     0),  INT16_C(  6304),  INT16_C( 20642), -INT16_C(  2003),  INT16_C( 27298),  INT16_C(     0),
        -INT16_C( 11314),  INT16_C(     0),  INT16_C(     0),  INT16_C( 24168), -INT16_C( 20775),  INT16_C(     0),  INT16_C( 28284),  INT16_C(     0) } },
    { UINT32_C(2664605311),
      { -INT16_C( 13121),  INT16_C( 21060), -INT16_C( 32516),  INT16_C( 27180),  INT16_C( 14562), -INT16_C(    62), -INT16_C(  2286),  INT16_C( 20163),
        -INT16_C(  4421), -INT16_C( 23648),  INT16_C( 23284), -INT16_C( 13689),  INT16_C(  9409),  INT16_C( 16390), -INT16_C(  9782), -INT16_C( 30242),
         INT16_C(  8869), -INT16_C( 23844),  INT16_C(  2211), -INT16_C( 31476), -INT16_C( 12735),  INT16_C( 21381),  INT16_C( 18629), -INT16_C( 32607),
         INT16_C( 16694),  INT16_C( 10788), -INT16_C( 21605),  INT16_C( 23796), -INT16_C(  1073), -INT16_C( 26211),  INT16_C( 31700),  INT16_C( 31011) },
      { -INT16_C(    98),  INT16_C( 16667),  INT16_C(  9991),  INT16_C( 18630),  INT16_C( 19445), -INT16_C( 17508),  INT16_C( 15763), -INT16_C( 14021),
         INT16_C( 24447),  INT16_C(  6900), -INT16_C(  6133), -INT16_C(  9609),  INT16_C(  5347), -INT16_C( 18572), -INT16_C( 26737),  INT16_C( 11569),
         INT16_C( 19606), -INT16_C( 25234),  INT16_C( 13684),  INT16_C( 27110), -INT16_C( 32128),  INT16_C(  5156),  INT16_C( 24767),  INT16_C( 16093),
        -INT16_C( 11841), -INT16_C( 13735), -INT16_C( 12102), -INT16_C( 25179),  INT16_C(  6628),  INT16_C( 29525), -INT16_C( 31056),  INT16_C( 18081) },
      { -INT16_C( 13023),  INT16_C(  4393),  INT16_C( 23029),  INT16_C(  8550), -INT16_C(  4883),  INT16_C( 17446), -INT16_C( 18049),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 30548),  INT16_C( 29417),  INT16_C(     0),  INT16_C(     0), -INT16_C( 30574),  INT16_C(     0),  INT16_C( 23725),
         INT16_C(     0),  INT16_C(  1390),  INT16_C(     0),  INT16_C(     0),  INT16_C( 19393),  INT16_C(     0), -INT16_C(  6138),  INT16_C( 16836),
         INT16_C(     0),  INT16_C( 24523), -INT16_C(  9503), -INT16_C( 16561), -INT16_C(  7701),  INT16_C(     0),  INT16_C(     0),  INT16_C( 12930) } },
    { UINT32_C(1189285842),
      { -INT16_C( 14012), -INT16_C( 14928), -INT16_C( 11189),  INT16_C(  3033), -INT16_C( 18892), -INT16_C(  2999), -INT16_C( 23928),  INT16_C( 17086),
         INT16_C( 25458),  INT16_C( 22239),  INT16_C( 13436),  INT16_C( 11466),  INT16_C( 27578), -INT16_C( 29326),  INT16_C( 22138), -INT16_C( 16429),
        -INT16_C( 31969),  INT16_C( 27524),  INT16_C( 23896), -INT16_C( 29578), -INT16_C( 16621), -INT16_C( 25728),  INT16_C( 16226), -INT16_C( 11043),
        -INT16_C( 16990),  INT16_C(  7979), -INT16_C(  2575), -INT16_C( 21429), -INT16_C( 16800), -INT16_C(  9671),  INT16_C(  3092),  INT16_C( 13209) },
      {  INT16_C(  7568), -INT16_C(  5986),  INT16_C(  5242), -INT16_C( 29068), -INT16_C(  2604),  INT16_C( 13865),  INT16_C(  1844), -INT16_C( 10742),
         INT16_C( 13764), -INT16_C( 18955),  INT16_C( 16682), -INT16_C( 30111), -INT16_C( 25857),  INT16_C(  4965), -INT16_C(   345),  INT16_C( 14150),
        -INT16_C(  6884), -INT16_C( 27105), -INT16_C( 27655), -INT16_C( 13020),  INT16_C( 20104), -INT16_C( 17405),  INT16_C(  3669),  INT16_C(  6547),
        -INT16_C( 30653),  INT16_C( 28366),  INT16_C( 12489), -INT16_C( 14088),  INT16_C( 24010),  INT16_C( 29147),  INT16_C(  8796),  INT16_C( 30888) },
      {  INT16_C(     0), -INT16_C(  8942),  INT16_C(     0),  INT16_C(     0), -INT16_C( 16288),  INT16_C(     0), -INT16_C( 25772),  INT16_C( 27828),
         INT16_C( 11694), -INT16_C( 24342), -INT16_C(  3246), -INT16_C( 23959),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 25085), -INT16_C( 10907),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  8323),  INT16_C( 12557), -INT16_C( 17590),
         INT16_C(     0), -INT16_C( 20387), -INT16_C( 15064),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  5704),  INT16_C(     0) } },
    { UINT32_C(    968455),
      {  INT16_C( 13147), -INT16_C(  7218), -INT16_C( 11903), -INT16_C( 10592),  INT16_C( 13279),  INT16_C(  9199), -INT16_C( 16965), -INT16_C( 31343),
        -INT16_C( 30227), -INT16_C( 18355),  INT16_C( 10727),  INT16_C( 17193), -INT16_C( 11701),  INT16_C( 21179), -INT16_C( 13927), -INT16_C(  2990),
         INT16_C(  8444),  INT16_C( 32216),  INT16_C( 30962), -INT16_C( 11949),  INT16_C( 17067),  INT16_C( 26356), -INT16_C( 31488), -INT16_C(  4629),
         INT16_C( 14607), -INT16_C(  2395), -INT16_C( 12446), -INT16_C( 21191), -INT16_C(  2911),  INT16_C( 15103),  INT16_C( 20925), -INT16_C( 17873) },
      {  INT16_C(  1906),  INT16_C( 25655), -INT16_C( 29825),  INT16_C( 10805),  INT16_C( 10957), -INT16_C( 12912),  INT16_C( 31919), -INT16_C( 16709),
         INT16_C( 24757),  INT16_C(  6068), -INT16_C(  4817), -INT16_C( 12092), -INT16_C( 15391), -INT16_C( 24821),  INT16_C( 14868), -INT16_C( 31143),
        -INT16_C( 28607), -INT16_C( 16150),  INT16_C(  8219), -INT16_C(  5654),  INT16_C( 31306), -INT16_C(  1610),  INT16_C( 29174), -INT16_C( 21576),
         INT16_C( 27858),  INT16_C(   450), -INT16_C( 31142),  INT16_C( 15314), -INT16_C(  8887),  INT16_C( 24282),  INT16_C( 13079),  INT16_C( 22756) },
      {  INT16_C( 11241),  INT16_C( 32663),  INT16_C( 17922),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 10552), -INT16_C( 24423),  INT16_C( 15544),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28795),  INT16_C( 28153),
         INT16_C(     0), -INT16_C( 17170),  INT16_C( 22743), -INT16_C(  6295),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(3742945220),
      {  INT16_C(   751),  INT16_C( 14792),  INT16_C( 32636),  INT16_C( 29490), -INT16_C(  5392), -INT16_C( 15842), -INT16_C(  7849), -INT16_C( 20028),
        -INT16_C( 27033), -INT16_C( 19988), -INT16_C( 14477), -INT16_C( 30193), -INT16_C(  3078), -INT16_C( 16670), -INT16_C(  1342), -INT16_C( 20066),
         INT16_C( 26364),  INT16_C( 30954),  INT16_C(  7653), -INT16_C( 10517),  INT16_C(  2567),  INT16_C( 24216),  INT16_C( 23787),  INT16_C( 21007),
        -INT16_C(   782),  INT16_C( 25859),  INT16_C(  4803), -INT16_C( 16913), -INT16_C( 12026), -INT16_C( 14212),  INT16_C(  6859), -INT16_C( 14470) },
      {  INT16_C( 25728),  INT16_C( 26176),  INT16_C( 11137), -INT16_C( 30404), -INT16_C( 11211),  INT16_C(  8423), -INT16_C(  2255),  INT16_C(  9075),
         INT16_C( 30451), -INT16_C( 18807),  INT16_C( 30857), -INT16_C( 28813), -INT16_C(  4278),  INT16_C(  5463), -INT16_C( 12023), -INT16_C( 29987),
         INT16_C(  7478), -INT16_C( 18448),  INT16_C( 11336),  INT16_C( 32320),  INT16_C( 10240),  INT16_C( 12702),  INT16_C(  4383),  INT16_C(  4693),
        -INT16_C(  8568),  INT16_C(  4552),  INT16_C( 15190), -INT16_C( 24416), -INT16_C(  2261),  INT16_C( 13494), -INT16_C( 27703), -INT16_C(    66) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 21499),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  5594), -INT16_C( 29103),
         INT16_C(  8052), -INT16_C(  1181),  INT16_C( 20202), -INT16_C(  1380),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10681),  INT16_C(  9921),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 22699), -INT16_C(  7673),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(  7786),  INT16_C( 21307), -INT16_C( 10387),  INT16_C(  7503), -INT16_C(  9765),  INT16_C(     0), -INT16_C( 30974), -INT16_C( 14404) } },
    { UINT32_C(4172721840),
      { -INT16_C(  2086), -INT16_C(  9354),  INT16_C(  5407),  INT16_C( 15884),  INT16_C( 24870), -INT16_C( 20912),  INT16_C(  6207), -INT16_C( 26945),
         INT16_C( 24403),  INT16_C( 32310), -INT16_C(  5033),  INT16_C(  8371),  INT16_C( 29055),  INT16_C( 12063), -INT16_C( 10976), -INT16_C(  1496),
        -INT16_C( 24884), -INT16_C(  5163), -INT16_C(  7501), -INT16_C(  9687),  INT16_C( 31043), -INT16_C( 31864),  INT16_C( 18577), -INT16_C(  6887),
         INT16_C( 20391), -INT16_C(   413),  INT16_C(  5692), -INT16_C( 17634),  INT16_C( 15752), -INT16_C( 22293),  INT16_C(  4883), -INT16_C(  8286) },
      {  INT16_C( 30897),  INT16_C( 26059), -INT16_C(  2982), -INT16_C( 25281), -INT16_C( 14482), -INT16_C(   224),  INT16_C( 14607), -INT16_C( 18460),
         INT16_C( 18569), -INT16_C( 14923), -INT16_C( 11170), -INT16_C(  6528),  INT16_C( 27409),  INT16_C(  9358),  INT16_C( 12670),  INT16_C( 12292),
        -INT16_C( 12375),  INT16_C(   917), -INT16_C( 11069),  INT16_C( 12704), -INT16_C( 15973), -INT16_C( 21711),  INT16_C(  5626), -INT16_C( 31902),
         INT16_C(  5981), -INT16_C( 17336), -INT16_C( 13845), -INT16_C(   606),  INT16_C( 12596), -INT16_C( 19679),  INT16_C(  9570),  INT16_C(  3043) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 26184), -INT16_C( 20688),  INT16_C(     0), -INT16_C(  8485),
         INT16_C(     0), -INT16_C( 18303),  INT16_C(  6137),  INT16_C( 14899),  INT16_C(     0),  INT16_C(  2705),  INT16_C(     0), -INT16_C( 13788),
         INT16_C(     0), -INT16_C(  6080),  INT16_C(  3568),  INT16_C(     0), -INT16_C( 18520), -INT16_C( 10153),  INT16_C(     0),  INT16_C( 25015),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 17028),  INT16_C(  3156), -INT16_C(  2614), -INT16_C(  4687), -INT16_C( 11329) } },
    { UINT32_C(3087956212),
      { -INT16_C( 20916), -INT16_C(  6167),  INT16_C(  6767),  INT16_C( 27282), -INT16_C(  3024), -INT16_C( 29203),  INT16_C( 13836), -INT16_C(  2231),
        -INT16_C(  4865),  INT16_C( 13300),  INT16_C(  5661),  INT16_C( 32742), -INT16_C( 14021),  INT16_C( 12426), -INT16_C( 26559), -INT16_C( 29208),
        -INT16_C( 11962), -INT16_C( 18827),  INT16_C(  2028),  INT16_C(  7200),  INT16_C(  3580),  INT16_C(  2217), -INT16_C(  3261),  INT16_C( 17151),
        -INT16_C(  2849), -INT16_C(   906),  INT16_C( 23562),  INT16_C( 17787),  INT16_C(  1318),  INT16_C( 26485),  INT16_C( 23965), -INT16_C(  7179) },
      {  INT16_C( 27183),  INT16_C(  7065), -INT16_C( 18063),  INT16_C( 27959), -INT16_C(  7993),  INT16_C(  2677),  INT16_C( 30163), -INT16_C( 19891),
        -INT16_C( 15511),  INT16_C( 29614),  INT16_C( 10527),  INT16_C( 17848),  INT16_C( 11822), -INT16_C( 13395), -INT16_C( 23925), -INT16_C( 17745),
         INT16_C( 18444),  INT16_C( 32213),  INT16_C(  3074), -INT16_C( 13845),  INT16_C( 24813), -INT16_C( 16173),  INT16_C(  8405),  INT16_C( 15987),
         INT16_C(  8675),  INT16_C(   945),  INT16_C( 27211),  INT16_C( 31048), -INT16_C(  2664),  INT16_C(  9029), -INT16_C(  2921), -INT16_C( 23586) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 24830),  INT16_C(     0),  INT16_C(  4969), -INT16_C( 31880), -INT16_C( 16327),  INT16_C( 17660),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 14894), -INT16_C( 25843),  INT16_C( 25821), -INT16_C(  2634),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 14496), -INT16_C(  1046),  INT16_C( 21045),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 13261),  INT16_C(  3982),  INT16_C( 17456),  INT16_C(     0),  INT16_C( 16407) } },
    { UINT32_C(1042395964),
      {  INT16_C(  3264), -INT16_C( 21241), -INT16_C(  9364),  INT16_C( 17005), -INT16_C(  7941), -INT16_C(  8320),  INT16_C( 12802),  INT16_C( 19938),
         INT16_C( 10908),  INT16_C( 13510),  INT16_C(  2848), -INT16_C( 18601),  INT16_C( 13823),  INT16_C( 15451),  INT16_C( 31977), -INT16_C( 22150),
        -INT16_C( 32120), -INT16_C(  2986), -INT16_C( 15523),  INT16_C( 22582), -INT16_C( 18524), -INT16_C( 22985),  INT16_C(  6633), -INT16_C( 31245),
        -INT16_C( 18108),  INT16_C( 25785),  INT16_C(  4293), -INT16_C( 15333),  INT16_C( 30278),  INT16_C( 12032),  INT16_C( 31730),  INT16_C( 31448) },
      {  INT16_C( 12029),  INT16_C( 23151), -INT16_C( 23055), -INT16_C( 27214), -INT16_C(  5540),  INT16_C( 17723),  INT16_C( 11779),  INT16_C( 18378),
        -INT16_C( 31768), -INT16_C( 21077), -INT16_C( 14444), -INT16_C(  9615),  INT16_C( 29245),  INT16_C( 12297), -INT16_C(  7699), -INT16_C(  5462),
         INT16_C(  6415),  INT16_C(    68), -INT16_C(  2369),  INT16_C(  7062), -INT16_C( 11808), -INT16_C(  7071),  INT16_C( 11008), -INT16_C(  6101),
        -INT16_C( 10321),  INT16_C( 17301),  INT16_C(  1694), -INT16_C(  9443),  INT16_C(  9848),  INT16_C( 25867), -INT16_C( 18937),  INT16_C(  5711) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 13691), -INT16_C( 21317), -INT16_C(  2401), -INT16_C( 26043),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 22860), -INT16_C( 30949),  INT16_C(     0),  INT16_C(     0), -INT16_C( 15422),  INT16_C(  3154),  INT16_C(     0), -INT16_C( 16688),
         INT16_C( 27001),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 15914),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(  8484),  INT16_C(  2599), -INT16_C(  5890),  INT16_C( 20430), -INT16_C( 13835),  INT16_C(     0),  INT16_C(     0) } }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].arr_a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].arr_b);
    easysimd__m512i r = easysimd_mm512_loadu_epi16(test_vec[i].arr_r);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_maskz_sub_epi16(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_sub_epi16");
    easysimd_assert_m512i_i32(ret, ==, r);
  }

#else 
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    easysimd__m512i r = easysimd_mm512_maskz_sub_epi16(k, a, b);

    easysimd_test_codegen_write_u32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif

  return 0;
}

static int
test_easysimd_mm512_mask_sub_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( -957186609), INT32_C(-1524765283), INT32_C( 1290068568), INT32_C( 1887468775),
                            INT32_C( -904096999), INT32_C(-1189693212), INT32_C(  221355870), INT32_C(-1952779315),
                            INT32_C( 1347985035), INT32_C(-2063939133), INT32_C(-1602582649), INT32_C(-2096850611),
                            INT32_C(-2084994527), INT32_C(  -75386963), INT32_C( 1835417512), INT32_C(-2072964471)),
      UINT16_C(35396),
      easysimd_mm512_set_epi32(INT32_C(  136551409), INT32_C( 1192962314), INT32_C( 2058621765), INT32_C(-2039270859),
                            INT32_C(  -26254502), INT32_C(  733381108), INT32_C( -187934344), INT32_C(  989979336),
                            INT32_C(-1964919382), INT32_C(  126554293), INT32_C(  254011928), INT32_C( 1490517506),
                            INT32_C(-1065486850), INT32_C(   45941921), INT32_C(-1082899768), INT32_C( -219628031)),
      easysimd_mm512_set_epi32(INT32_C( -680185335), INT32_C(  111102276), INT32_C( 1222454066), INT32_C( -422241261),
                            INT32_C(  -78061198), INT32_C(-2084414007), INT32_C( 1367041146), INT32_C(-1471398421),
                            INT32_C( -348147705), INT32_C(  673564238), INT32_C(-1457376577), INT32_C(  613875036),
                            INT32_C( -859069431), INT32_C( -733638834), INT32_C(-1673403701), INT32_C(  842474288)),
      easysimd_mm512_set_epi32(INT32_C(  816736744), INT32_C(-1524765283), INT32_C( 1290068568), INT32_C( 1887468775),
                            INT32_C(   51806696), INT32_C(-1189693212), INT32_C(-1554975490), INT32_C(-1952779315),
                            INT32_C( 1347985035), INT32_C( -547009945), INT32_C(-1602582649), INT32_C(-2096850611),
                            INT32_C(-2084994527), INT32_C(  779580755), INT32_C( 1835417512), INT32_C(-2072964471)) },
    { easysimd_mm512_set_epi32(INT32_C( 2077489237), INT32_C(-2066152618), INT32_C( -825076901), INT32_C(-1372574642),
                            INT32_C( -889460158), INT32_C(  498921453), INT32_C(  943332338), INT32_C(-1383811831),
                            INT32_C( 1827152592), INT32_C( 1728034912), INT32_C( 1350913629), INT32_C(  868578809),
                            INT32_C( 1368636899), INT32_C( -389235219), INT32_C(  602990700), INT32_C( -400551366)),
      UINT16_C(47779),
      easysimd_mm512_set_epi32(INT32_C( 1704357216), INT32_C( -538157327), INT32_C( 1370875608), INT32_C( 1508504457),
                            INT32_C(  -68294915), INT32_C(-1479685367), INT32_C( -615597542), INT32_C(-1638887359),
                            INT32_C(-1417912572), INT32_C( 1479002949), INT32_C( -647118153), INT32_C( 1670566025),
                            INT32_C(-1880268561), INT32_C(-1083232065), INT32_C( 2092339698), INT32_C(-1021873283)),
      easysimd_mm512_set_epi32(INT32_C( -839277498), INT32_C(  551588590), INT32_C( 1834572496), INT32_C( 1613035598),
                            INT32_C(-1678404828), INT32_C(-1769391216), INT32_C(-1638931514), INT32_C(  156804649),
                            INT32_C( 1764158657), INT32_C( -132604621), INT32_C(  446542816), INT32_C( 2037189710),
                            INT32_C(  109296986), INT32_C(  257019297), INT32_C(  473079611), INT32_C( 1127076998)),
      easysimd_mm512_set_epi32(INT32_C(-1751332582), INT32_C(-2066152618), INT32_C( -463696888), INT32_C( -104531141),
                            INT32_C( 1610109913), INT32_C(  498921453), INT32_C( 1023333972), INT32_C(-1383811831),
                            INT32_C( 1112896067), INT32_C( 1728034912), INT32_C(-1093660969), INT32_C(  868578809),
                            INT32_C( 1368636899), INT32_C( -389235219), INT32_C( 1619260087), INT32_C( 2146017015)) },
    { easysimd_mm512_set_epi32(INT32_C(  307630641), INT32_C(-1560148595), INT32_C(  376284729), INT32_C(  278591183),
                            INT32_C( -277186219), INT32_C( 1940926671), INT32_C(  662058232), INT32_C( 1091202812),
                            INT32_C( -701136301), INT32_C( -504607320), INT32_C( -251380880), INT32_C( 1860616049),
                            INT32_C(-1752161866), INT32_C(-1199997313), INT32_C(-1668691262), INT32_C( 1717921298)),
      UINT16_C( 2459),
      easysimd_mm512_set_epi32(INT32_C( 2079917891), INT32_C(-1199015072), INT32_C(  -98602729), INT32_C( -930567988),
                            INT32_C(-1256209763), INT32_C( 1068967165), INT32_C( 1289079409), INT32_C( 1251085533),
                            INT32_C( -727360546), INT32_C(-1724797341), INT32_C( 2093813635), INT32_C( 1051617285),
                            INT32_C( 1264716001), INT32_C(  940727836), INT32_C( 1722577424), INT32_C(-1275657732)),
      easysimd_mm512_set_epi32(INT32_C(  671797033), INT32_C(-1012795446), INT32_C( 2106088193), INT32_C( -458612579),
                            INT32_C( -261772865), INT32_C( -550994046), INT32_C( 2105186719), INT32_C( 1074097751),
                            INT32_C(-1251411324), INT32_C(   65867416), INT32_C(-1495248139), INT32_C(  315553116),
                            INT32_C(-1869712369), INT32_C(-1246794510), INT32_C( 1218370652), INT32_C( -240388126)),
      easysimd_mm512_set_epi32(INT32_C(  307630641), INT32_C(-1560148595), INT32_C(  376284729), INT32_C(  278591183),
                            INT32_C( -994436898), INT32_C( 1940926671), INT32_C(  662058232), INT32_C(  176987782),
                            INT32_C(  524050778), INT32_C( -504607320), INT32_C( -251380880), INT32_C(  736064169),
                            INT32_C(-1160538926), INT32_C(-1199997313), INT32_C(  504206772), INT32_C(-1035269606)) },
    { easysimd_mm512_set_epi32(INT32_C( -789716549), INT32_C(-1932674309), INT32_C(  548470804), INT32_C( -318652401),
                            INT32_C(-2041118423), INT32_C(-2107945718), INT32_C( -715661009), INT32_C( 1609073505),
                            INT32_C( 1214609500), INT32_C(  283085327), INT32_C(-1633515677), INT32_C( 1697029857),
                            INT32_C( 1976447422), INT32_C(  904412076), INT32_C( 1198927422), INT32_C(-1498026761)),
      UINT16_C(54315),
      easysimd_mm512_set_epi32(INT32_C( 1385182319), INT32_C(  795273310), INT32_C( 1955628796), INT32_C( -526907127),
                            INT32_C(-2141025282), INT32_C( -931446405), INT32_C(-1422139726), INT32_C(-1101084337),
                            INT32_C( -254080461), INT32_C( -595291883), INT32_C( 1292692652), INT32_C(-1849951866),
                            INT32_C( -815091127), INT32_C(  370112774), INT32_C( -520479179), INT32_C( 1681391452)),
      easysimd_mm512_set_epi32(INT32_C(-1825216267), INT32_C( 1555513845), INT32_C(-2081576252), INT32_C(-1972081268),
                            INT32_C( -563427058), INT32_C( 1922040193), INT32_C(-2102270715), INT32_C(-1257264155),
                            INT32_C( -894851768), INT32_C( 1793334666), INT32_C( 1049305530), INT32_C(-1935379009),
                            INT32_C(   -8279361), INT32_C(-1567490719), INT32_C(-2014130513), INT32_C(-1826154506)),
      easysimd_mm512_set_epi32(INT32_C(-1084568710), INT32_C( -760240535), INT32_C(  548470804), INT32_C( 1445174141),
                            INT32_C(-2041118423), INT32_C( 1441480698), INT32_C( -715661009), INT32_C( 1609073505),
                            INT32_C( 1214609500), INT32_C(  283085327), INT32_C(  243387122), INT32_C( 1697029857),
                            INT32_C( -806811766), INT32_C(  904412076), INT32_C( 1493651334), INT32_C( -787421338)) },
    { easysimd_mm512_set_epi32(INT32_C(  997407681), INT32_C(  -83308341), INT32_C( 1430458288), INT32_C( -655910274),
                            INT32_C(   17159218), INT32_C(  197891822), INT32_C(  -82165524), INT32_C(   98130061),
                            INT32_C( -696255503), INT32_C(  616388941), INT32_C( 1383637516), INT32_C(  255219509),
                            INT32_C(-1280964183), INT32_C(-1753221031), INT32_C(  480974923), INT32_C(-1444611560)),
      UINT16_C(47568),
      easysimd_mm512_set_epi32(INT32_C(-1796791424), INT32_C(  919413682), INT32_C(  907613991), INT32_C(-1471064632),
                            INT32_C(-2017464794), INT32_C(  -67778959), INT32_C(-1033884668), INT32_C( -839095279),
                            INT32_C( -881742684), INT32_C( 1193890045), INT32_C( -817450648), INT32_C( -450889209),
                            INT32_C(-1829442769), INT32_C( -254239276), INT32_C( 1531184539), INT32_C(  204100550)),
      easysimd_mm512_set_epi32(INT32_C(-1574624316), INT32_C( 1965632168), INT32_C( -507137262), INT32_C(  868285762),
                            INT32_C( -287712967), INT32_C(-1275855491), INT32_C(-1948986373), INT32_C(  378189270),
                            INT32_C( 2028975029), INT32_C( -983819985), INT32_C(-1530834794), INT32_C( -267906659),
                            INT32_C( 2013371063), INT32_C( -972550977), INT32_C(-1345658151), INT32_C(-2001069348)),
      easysimd_mm512_set_epi32(INT32_C( -222167108), INT32_C(  -83308341), INT32_C( 1414751253), INT32_C( 1955616902),
                            INT32_C(-1729751827), INT32_C(  197891822), INT32_C(  -82165524), INT32_C(-1217284549),
                            INT32_C( 1384249583), INT32_C(-2117257266), INT32_C( 1383637516), INT32_C( -182982550),
                            INT32_C(-1280964183), INT32_C(-1753221031), INT32_C(  480974923), INT32_C(-1444611560)) },
    { easysimd_mm512_set_epi32(INT32_C( 1875288432), INT32_C( 1158027251), INT32_C( -303056299), INT32_C( -939396673),
                            INT32_C( 1585003262), INT32_C( 1365783459), INT32_C(  111845672), INT32_C(-1286713478),
                            INT32_C(  674624782), INT32_C( 2020528740), INT32_C(  497192398), INT32_C( 1112540789),
                            INT32_C(-1764167278), INT32_C(-1540772359), INT32_C(  395629026), INT32_C(  984304916)),
      UINT16_C(16877),
      easysimd_mm512_set_epi32(INT32_C( -344292944), INT32_C( 1968428151), INT32_C( 2086978939), INT32_C( 1501910543),
                            INT32_C(-1262393002), INT32_C( 2081469023), INT32_C( 2016768793), INT32_C( 1922434397),
                            INT32_C( -253304624), INT32_C(  515280842), INT32_C(-1708348294), INT32_C( 2107558843),
                            INT32_C( 1919035054), INT32_C( 1742835915), INT32_C(  989439209), INT32_C( 2080310116)),
      easysimd_mm512_set_epi32(INT32_C( 1560352883), INT32_C( -937050525), INT32_C(   15000953), INT32_C(  298895006),
                            INT32_C( -255287325), INT32_C( -851082971), INT32_C( -981170631), INT32_C(   30364523),
                            INT32_C( -626854551), INT32_C( 1776719697), INT32_C(-1286673883), INT32_C( 2134458392),
                            INT32_C(-1884377437), INT32_C(-2042525337), INT32_C( 2143156805), INT32_C(-1045267304)),
      easysimd_mm512_set_epi32(INT32_C( 1875288432), INT32_C(-1389488620), INT32_C( -303056299), INT32_C( -939396673),
                            INT32_C( 1585003262), INT32_C( 1365783459), INT32_C(  111845672), INT32_C( 1892069874),
                            INT32_C(  373549927), INT32_C(-1261438855), INT32_C( -421674411), INT32_C( 1112540789),
                            INT32_C( -491554805), INT32_C( -509606044), INT32_C(  395629026), INT32_C(-1169389876)) },
    { easysimd_mm512_set_epi32(INT32_C(  726531409), INT32_C( -606374582), INT32_C(-1057918709), INT32_C( -811736744),
                            INT32_C(-1460245574), INT32_C( -627872087), INT32_C( 1799586442), INT32_C(-1105519928),
                            INT32_C(-1288829692), INT32_C(-2144392739), INT32_C( 1110910857), INT32_C( -282270116),
                            INT32_C(-1420141426), INT32_C( 1682561587), INT32_C( 1308021682), INT32_C(  712875579)),
      UINT16_C(17567),
      easysimd_mm512_set_epi32(INT32_C(-1065890522), INT32_C( 1362887862), INT32_C(-1905482051), INT32_C(  174767211),
                            INT32_C( 1968089357), INT32_C(-1207243832), INT32_C( -701927204), INT32_C(-1701909648),
                            INT32_C(-1822821880), INT32_C(-1418686446), INT32_C( 2002979046), INT32_C( -531029674),
                            INT32_C( -233545704), INT32_C( 1270923539), INT32_C( -515398077), INT32_C(  870828526)),
      easysimd_mm512_set_epi32(INT32_C(-1161246521), INT32_C(-1263382687), INT32_C( -761171059), INT32_C( 1052537110),
                            INT32_C(-1225204820), INT32_C( 1299827393), INT32_C(  477328169), INT32_C( 2043159101),
                            INT32_C(  984199920), INT32_C( 1963689737), INT32_C(-1149812166), INT32_C( -500241318),
                            INT32_C( -953270640), INT32_C( 1180984926), INT32_C( -645305643), INT32_C( 1026486800)),
      easysimd_mm512_set_epi32(INT32_C(  726531409), INT32_C(-1668696747), INT32_C(-1057918709), INT32_C( -811736744),
                            INT32_C(-1460245574), INT32_C( 1787896071), INT32_C( 1799586442), INT32_C(-1105519928),
                            INT32_C( 1487945496), INT32_C(-2144392739), INT32_C( 1110910857), INT32_C(  -30788356),
                            INT32_C(  719724936), INT32_C(   89938613), INT32_C(  129907566), INT32_C( -155658274)) },
    { easysimd_mm512_set_epi32(INT32_C( 1723004290), INT32_C(  721161302), INT32_C( 1077400739), INT32_C(  861837752),
                            INT32_C(-1943224858), INT32_C( 2112602876), INT32_C(-1445821889), INT32_C(-2100432693),
                            INT32_C(-1175934343), INT32_C(  805502143), INT32_C( 1163969458), INT32_C(  873642413),
                            INT32_C( 2052720739), INT32_C(-1010971457), INT32_C(  199344228), INT32_C(  251460647)),
      UINT16_C(59134),
      easysimd_mm512_set_epi32(INT32_C(-1391704351), INT32_C( -847303025), INT32_C(-1711491580), INT32_C( -147993971),
                            INT32_C(-1140349230), INT32_C(  172650828), INT32_C(-2090294261), INT32_C( -216506888),
                            INT32_C(-1813744120), INT32_C( 1589656338), INT32_C( 1010967585), INT32_C(-2076714127),
                            INT32_C( 1156626662), INT32_C( -264321123), INT32_C(-1099385436), INT32_C( -148901794)),
      easysimd_mm512_set_epi32(INT32_C( 1003282629), INT32_C( 1250297288), INT32_C(   26548422), INT32_C(-1100962758),
                            INT32_C( 1934048830), INT32_C( -886200980), INT32_C( -228926178), INT32_C(   21722717),
                            INT32_C(-1321187708), INT32_C(  904822803), INT32_C( -875700432), INT32_C(-1302414558),
                            INT32_C(  962131440), INT32_C( -729214075), INT32_C(-1094266114), INT32_C( 1122895720)),
      easysimd_mm512_set_epi32(INT32_C( 1899980316), INT32_C(-2097600313), INT32_C(-1738040002), INT32_C(  861837752),
                            INT32_C(-1943224858), INT32_C( 1058851808), INT32_C(-1861368083), INT32_C(-2100432693),
                            INT32_C( -492556412), INT32_C(  684833535), INT32_C( 1886668017), INT32_C( -774299569),
                            INT32_C(  194495222), INT32_C(  464892952), INT32_C(   -5119322), INT32_C(  251460647)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sub_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sub_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_sub_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 8894478799917719473), INT64_C(-7614529333518044459),
                            INT64_C( 8458392650500739529), INT64_C( 7085639313865748967),
                            INT64_C(-7547504459018552290), INT64_C(-8310189466716392279),
                            INT64_C(-1750715323825344235), INT64_C(-2532781790488219528)),
      UINT8_C(106),
      easysimd_mm512_set_epi64(INT64_C(-7192427816606966254), INT64_C(-1619523557840103557),
                            INT64_C( 7616061596213068646), INT64_C( -560841280842371832),
                            INT64_C( -806373115982863580), INT64_C( -816793021936842074),
                            INT64_C( -317565234288882547), INT64_C(-7290553309909260368)),
      easysimd_mm512_set_epi64(INT64_C(-9084839040863053259), INT64_C(  332697972184433101),
                            INT64_C(-8959492887484217950), INT64_C( 7617292932467329680),
                            INT64_C(-2740045277871922718), INT64_C(-3634413508032825567),
                            INT64_C( -448440935066054877), INT64_C(-6805574594168851327)),
      easysimd_mm512_set_epi64(INT64_C( 8894478799917719473), INT64_C(-1952221530024536658),
                            INT64_C(-1871189590012265020), INT64_C( 7085639313865748967),
                            INT64_C( 1933672161889059138), INT64_C(-8310189466716392279),
                            INT64_C(  130875700777172330), INT64_C(-2532781790488219528)) },
    { easysimd_mm512_set_epi64(INT64_C(-3459089877760882917), INT64_C( 1753327656617706405),
                            INT64_C( 3932187030396497555), INT64_C(-4341921971190139713),
                            INT64_C(-7354864635860030437), INT64_C(-7512931671900842140),
                            INT64_C( 7677521206664265888), INT64_C(-8008068901606036732)),
      UINT8_C(  1),
      easysimd_mm512_set_epi64(INT64_C(-9084086707853197365), INT64_C( 5962789269656503800),
                            INT64_C( 6806616562165680967), INT64_C( 8724516399523474076),
                            INT64_C( -924171789017863248), INT64_C(-2255835938032964673),
                            INT64_C(-4560088794132063361), INT64_C(-5517329800302195238)),
      easysimd_mm512_set_epi64(INT64_C(-2849655299932577704), INT64_C( 2712991932590941674),
                            INT64_C( 2564329750539599066), INT64_C(-4536455326234991583),
                            INT64_C(-6477728239233614839), INT64_C(-5729565646249538826),
                            INT64_C( 3092410715614407585), INT64_C( 7984397770129184299)),
      easysimd_mm512_set_epi64(INT64_C(-3459089877760882917), INT64_C( 1753327656617706405),
                            INT64_C( 3932187030396497555), INT64_C(-4341921971190139713),
                            INT64_C(-7354864635860030437), INT64_C(-7512931671900842140),
                            INT64_C( 7677521206664265888), INT64_C( 4945016503278172079)) },
    { easysimd_mm512_set_epi64(INT64_C( -240340334077349403), INT64_C( 5647038489743797240),
                            INT64_C( 5171415873092064400), INT64_C(-1851380595205120917),
                            INT64_C( -836370148956202078), INT64_C( 8425549504970400810),
                            INT64_C( 2808549870315159479), INT64_C( 3545474415643732634)),
      UINT8_C(194),
      easysimd_mm512_set_epi64(INT64_C(-5877702108931305293), INT64_C(-5372639016544358566),
                            INT64_C(-4535660820549680684), INT64_C(-6747544612783901147),
                            INT64_C( 6705850594648382655), INT64_C(-1906321743942105225),
                            INT64_C( -281981608123407868), INT64_C(-5990711758326206044)),
      easysimd_mm512_set_epi64(INT64_C( 8110080903340414341), INT64_C(-3598578875674169061),
                            INT64_C( 4977285870543484474), INT64_C( 6776152673642620958),
                            INT64_C( 4245929756722282054), INT64_C( 3649495924615361625),
                            INT64_C( -638056186877872345), INT64_C(-8828385988165140326)),
      easysimd_mm512_set_epi64(INT64_C( 4458961061437831982), INT64_C(-1774060140870189505),
                            INT64_C( 5171415873092064400), INT64_C(-1851380595205120917),
                            INT64_C( -836370148956202078), INT64_C( 8425549504970400810),
                            INT64_C(  356074578754464477), INT64_C( 3545474415643732634)) },
    { easysimd_mm512_set_epi64(INT64_C(-6385979888474332285), INT64_C( 3716758445629922885),
                            INT64_C( 7861010731589253148), INT64_C(-6334773111204875550),
                            INT64_C(-5054960975820633825), INT64_C( 8639514840721539279),
                            INT64_C(-1027366943904624518), INT64_C(-4721195859159142702)),
      UINT8_C(222),
      easysimd_mm512_set_epi64(INT64_C(-7001132877809342173), INT64_C( 6512733899690414848),
                            INT64_C(  988878120815000883), INT64_C(-5994563704199492012),
                            INT64_C( 1587634372980811194), INT64_C( -914749563856678715),
                            INT64_C( 7495962388934953888), INT64_C(-7831181051188885332)),
      easysimd_mm512_set_epi64(INT64_C( 4229507402435677476), INT64_C( 2501842736425447642),
                            INT64_C( 8009397189160901283), INT64_C( 3833558633773719409),
                            INT64_C( 2852442819818074174), INT64_C(-8638015813272823849),
                            INT64_C( 8579593880416924807), INT64_C( 2713766728753976690)),
      easysimd_mm512_set_epi64(INT64_C( 7216103793464531967), INT64_C( 4010891163264967206),
                            INT64_C( 7861010731589253148), INT64_C( 8618621735736340195),
                            INT64_C(-1264808446837262980), INT64_C( 7723266249416145134),
                            INT64_C(-1083631491481970919), INT64_C(-4721195859159142702)) },
    { easysimd_mm512_set_epi64(INT64_C( 4051614369896270101), INT64_C( 6703896128856670897),
                            INT64_C(-5750389130785475983), INT64_C(-7878547924784098469),
                            INT64_C( 5491867996743881624), INT64_C(-2189602113514909499),
                            INT64_C( -887220462507309287), INT64_C(-5733898489940979010)),
      UINT8_C( 26),
      easysimd_mm512_set_epi64(INT64_C(  -99656633840764240), INT64_C(-3479731851565468885),
                            INT64_C(-7074577238264434881), INT64_C(-3836339826871533273),
                            INT64_C( 4198283975631841849), INT64_C(-3829622956767240841),
                            INT64_C( 5960966148924368684), INT64_C( -504125670847055963)),
      easysimd_mm512_set_epi64(INT64_C(-8344319212574510912), INT64_C(-3371415321000668561),
                            INT64_C(-8338525176508042897), INT64_C( 5173420397567361383),
                            INT64_C(-6751809518396836721), INT64_C(-8388491552134432960),
                            INT64_C(-9161028627110906680), INT64_C( 7472048750700349549)),
      easysimd_mm512_set_epi64(INT64_C( 4051614369896270101), INT64_C( 6703896128856670897),
                            INT64_C(-5750389130785475983), INT64_C(-9009760224438894656),
                            INT64_C(-7496650579680873046), INT64_C(-2189602113514909499),
                            INT64_C(-3324749297674276252), INT64_C(-5733898489940979010)) },
    { easysimd_mm512_set_epi64(INT64_C(-6378393891104748170), INT64_C(-8478287659785501826),
                            INT64_C(-2127236125072242134), INT64_C( 8702738982982040445),
                            INT64_C(  645844328650761785), INT64_C(-4561773442934600720),
                            INT64_C(-5793568656482259588), INT64_C( -379681413311801170)),
      UINT8_C(230),
      easysimd_mm512_set_epi64(INT64_C( -848706848545220792), INT64_C(-1124075123789220737),
                            INT64_C(-2005439629632543252), INT64_C( 8274388146286059619),
                            INT64_C( -261550962782015927), INT64_C(-8761037216848109215),
                            INT64_C(-3016365966836321630), INT64_C( 2543055264688040393)),
      easysimd_mm512_set_epi64(INT64_C( 1583638370136684317), INT64_C(-1184919915070849427),
                            INT64_C( 6948286910398693964), INT64_C( 2437457976149582578),
                            INT64_C( 3426542754873284897), INT64_C(-7983270512780038531),
                            INT64_C( 1779296328975282374), INT64_C(-5362999871220584978)),
      easysimd_mm512_set_epi64(INT64_C(-2432345218681905109), INT64_C(   60844791281628690),
                            INT64_C(-8953726540031237216), INT64_C( 8702738982982040445),
                            INT64_C(  645844328650761785), INT64_C( -777766704068070684),
                            INT64_C(-4795662295811604004), INT64_C( -379681413311801170)) },
    { easysimd_mm512_set_epi64(INT64_C(-2563692560784467599), INT64_C(-2764729313181954331),
                            INT64_C( 7449793955604076666), INT64_C(-6302011830015535814),
                            INT64_C(-5919077484698028869), INT64_C(-6127059769393124093),
                            INT64_C( 2958642729945465911), INT64_C( 2772140786646472311)),
      UINT8_C(198),
      easysimd_mm512_set_epi64(INT64_C(-3934991658845807023), INT64_C( 7561755153516237296),
                            INT64_C(-1521478373140770922), INT64_C( 6956443634033398294),
                            INT64_C(-5307063963483146371), INT64_C( 6556039892370535969),
                            INT64_C(-6645788521893978945), INT64_C(-6307512051127595595)),
      easysimd_mm512_set_epi64(INT64_C(-7270561721689602230), INT64_C( 8935792808270452615),
                            INT64_C( 1984489943341614372), INT64_C( 6860868624136070257),
                            INT64_C(-2243581398369652256), INT64_C(-6592818671779181804),
                            INT64_C( -308663241436655846), INT64_C(-8935526257161672911)),
      easysimd_mm512_set_epi64(INT64_C( 3335570062843795207), INT64_C(-1374037654754215319),
                            INT64_C( 7449793955604076666), INT64_C(-6302011830015535814),
                            INT64_C(-5919077484698028869), INT64_C(-5297885509559833843),
                            INT64_C(-6337125280457323099), INT64_C( 2772140786646472311)) },
    { easysimd_mm512_set_epi64(INT64_C(-7511866029206584895), INT64_C( 6685003933657692663),
                            INT64_C(  112057327023275278), INT64_C( 2785131907782223781),
                            INT64_C( -403719025987547254), INT64_C(-5974279397850363938),
                            INT64_C(-6601571580489345254), INT64_C( 1896379997419403836)),
      UINT8_C( 70),
      easysimd_mm512_set_epi64(INT64_C(-6334367433946281110), INT64_C(-5840485098030444461),
                            INT64_C(-6383956557021185117), INT64_C(-7600398675722821668),
                            INT64_C(-2279362749413199885), INT64_C(-8009539466982888201),
                            INT64_C(  340327559398526723), INT64_C(-2438629088141247826)),
      easysimd_mm512_set_epi64(INT64_C( 3758222621544461478), INT64_C( 8264387002851618510),
                            INT64_C( 5256515298231032169), INT64_C( 4555501816451377355),
                            INT64_C(-9184304616258229288), INT64_C( 5115688705834988612),
                            INT64_C(-3795492187184599084), INT64_C(-3221204559120447653)),
      easysimd_mm512_set_epi64(INT64_C(-7511866029206584895), INT64_C( 4341871972827488645),
                            INT64_C(  112057327023275278), INT64_C( 2785131907782223781),
                            INT64_C( -403719025987547254), INT64_C( 5321515900891674803),
                            INT64_C( 4135819746583125807), INT64_C( 1896379997419403836)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sub_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sub_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_sub_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 src;
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -417.79), EASYSIMD_FLOAT32_C(  -912.83), EASYSIMD_FLOAT32_C(   111.29), EASYSIMD_FLOAT32_C(  -470.87),
                         EASYSIMD_FLOAT32_C(   685.45), EASYSIMD_FLOAT32_C(   -92.85), EASYSIMD_FLOAT32_C(   704.55), EASYSIMD_FLOAT32_C(   450.79),
                         EASYSIMD_FLOAT32_C(  -761.01), EASYSIMD_FLOAT32_C(  -759.35), EASYSIMD_FLOAT32_C(   646.77), EASYSIMD_FLOAT32_C(   616.33),
                         EASYSIMD_FLOAT32_C(   922.76), EASYSIMD_FLOAT32_C(   721.94), EASYSIMD_FLOAT32_C(   721.78), EASYSIMD_FLOAT32_C(   651.66)),
      UINT16_C(55049),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   492.15), EASYSIMD_FLOAT32_C(   363.86), EASYSIMD_FLOAT32_C(  -906.93), EASYSIMD_FLOAT32_C(   -51.88),
                         EASYSIMD_FLOAT32_C(   976.36), EASYSIMD_FLOAT32_C(   844.84), EASYSIMD_FLOAT32_C(   525.57), EASYSIMD_FLOAT32_C(   575.43),
                         EASYSIMD_FLOAT32_C(  -719.61), EASYSIMD_FLOAT32_C(   570.91), EASYSIMD_FLOAT32_C(  -748.06), EASYSIMD_FLOAT32_C(   823.89),
                         EASYSIMD_FLOAT32_C(  -708.11), EASYSIMD_FLOAT32_C(  -805.87), EASYSIMD_FLOAT32_C(   626.28), EASYSIMD_FLOAT32_C(   344.43)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -814.48), EASYSIMD_FLOAT32_C(   843.19), EASYSIMD_FLOAT32_C(  -866.28), EASYSIMD_FLOAT32_C(  -230.51),
                         EASYSIMD_FLOAT32_C(  -264.51), EASYSIMD_FLOAT32_C(   935.39), EASYSIMD_FLOAT32_C(   479.68), EASYSIMD_FLOAT32_C(  -375.52),
                         EASYSIMD_FLOAT32_C(  -928.92), EASYSIMD_FLOAT32_C(  -243.75), EASYSIMD_FLOAT32_C(   771.60), EASYSIMD_FLOAT32_C(   150.31),
                         EASYSIMD_FLOAT32_C(  -627.83), EASYSIMD_FLOAT32_C(  -720.61), EASYSIMD_FLOAT32_C(   345.13), EASYSIMD_FLOAT32_C(   203.00)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  1306.63), EASYSIMD_FLOAT32_C(  -479.33), EASYSIMD_FLOAT32_C(   111.29), EASYSIMD_FLOAT32_C(   178.63),
                         EASYSIMD_FLOAT32_C(   685.45), EASYSIMD_FLOAT32_C(   -90.55), EASYSIMD_FLOAT32_C(    45.89), EASYSIMD_FLOAT32_C(   950.95),
                         EASYSIMD_FLOAT32_C(  -761.01), EASYSIMD_FLOAT32_C(  -759.35), EASYSIMD_FLOAT32_C(   646.77), EASYSIMD_FLOAT32_C(   616.33),
                         EASYSIMD_FLOAT32_C(   -80.28), EASYSIMD_FLOAT32_C(   721.94), EASYSIMD_FLOAT32_C(   721.78), EASYSIMD_FLOAT32_C(   141.43)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -594.79), EASYSIMD_FLOAT32_C(   -68.26), EASYSIMD_FLOAT32_C(   772.68), EASYSIMD_FLOAT32_C(  -615.12),
                         EASYSIMD_FLOAT32_C(   489.20), EASYSIMD_FLOAT32_C(  -609.74), EASYSIMD_FLOAT32_C(  -297.42), EASYSIMD_FLOAT32_C(  -701.58),
                         EASYSIMD_FLOAT32_C(    71.34), EASYSIMD_FLOAT32_C(  -811.20), EASYSIMD_FLOAT32_C(   -44.61), EASYSIMD_FLOAT32_C(   172.32),
                         EASYSIMD_FLOAT32_C(  -336.24), EASYSIMD_FLOAT32_C(  -959.77), EASYSIMD_FLOAT32_C(   896.40), EASYSIMD_FLOAT32_C(   321.28)),
      UINT16_C( 2266),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   136.73), EASYSIMD_FLOAT32_C(   408.70), EASYSIMD_FLOAT32_C(   907.04), EASYSIMD_FLOAT32_C(   175.32),
                         EASYSIMD_FLOAT32_C(   125.78), EASYSIMD_FLOAT32_C(  -176.42), EASYSIMD_FLOAT32_C(  -192.20), EASYSIMD_FLOAT32_C(   636.29),
                         EASYSIMD_FLOAT32_C(  -812.72), EASYSIMD_FLOAT32_C(  -295.02), EASYSIMD_FLOAT32_C(   426.00), EASYSIMD_FLOAT32_C(   348.29),
                         EASYSIMD_FLOAT32_C(   859.20), EASYSIMD_FLOAT32_C(   -28.95), EASYSIMD_FLOAT32_C(  -637.06), EASYSIMD_FLOAT32_C(  -450.15)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -929.87), EASYSIMD_FLOAT32_C(  -208.53), EASYSIMD_FLOAT32_C(   561.71), EASYSIMD_FLOAT32_C(   -74.05),
                         EASYSIMD_FLOAT32_C(   477.79), EASYSIMD_FLOAT32_C(   772.49), EASYSIMD_FLOAT32_C(   648.48), EASYSIMD_FLOAT32_C(   -58.61),
                         EASYSIMD_FLOAT32_C(   835.38), EASYSIMD_FLOAT32_C(  -689.00), EASYSIMD_FLOAT32_C(   607.03), EASYSIMD_FLOAT32_C(   421.78),
                         EASYSIMD_FLOAT32_C(  -574.15), EASYSIMD_FLOAT32_C(   302.76), EASYSIMD_FLOAT32_C(   178.11), EASYSIMD_FLOAT32_C(  -298.57)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -594.79), EASYSIMD_FLOAT32_C(   -68.26), EASYSIMD_FLOAT32_C(   772.68), EASYSIMD_FLOAT32_C(  -615.12),
                         EASYSIMD_FLOAT32_C(  -352.01), EASYSIMD_FLOAT32_C(  -609.74), EASYSIMD_FLOAT32_C(  -297.42), EASYSIMD_FLOAT32_C(  -701.58),
                         EASYSIMD_FLOAT32_C( -1648.10), EASYSIMD_FLOAT32_C(   393.98), EASYSIMD_FLOAT32_C(   -44.61), EASYSIMD_FLOAT32_C(   -73.49),
                         EASYSIMD_FLOAT32_C(  1433.35), EASYSIMD_FLOAT32_C(  -959.77), EASYSIMD_FLOAT32_C(  -815.17), EASYSIMD_FLOAT32_C(   321.28)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -914.76), EASYSIMD_FLOAT32_C(   285.68), EASYSIMD_FLOAT32_C(   695.03), EASYSIMD_FLOAT32_C(  -235.78),
                         EASYSIMD_FLOAT32_C(    90.17), EASYSIMD_FLOAT32_C(   891.02), EASYSIMD_FLOAT32_C(  -456.46), EASYSIMD_FLOAT32_C(   952.55),
                         EASYSIMD_FLOAT32_C(  -153.33), EASYSIMD_FLOAT32_C(  -533.35), EASYSIMD_FLOAT32_C(  -130.02), EASYSIMD_FLOAT32_C(  -580.21),
                         EASYSIMD_FLOAT32_C(  -857.73), EASYSIMD_FLOAT32_C(  -362.64), EASYSIMD_FLOAT32_C(   808.25), EASYSIMD_FLOAT32_C(   908.95)),
      UINT16_C(53407),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   415.38), EASYSIMD_FLOAT32_C(   622.33), EASYSIMD_FLOAT32_C(   849.49), EASYSIMD_FLOAT32_C(  -552.97),
                         EASYSIMD_FLOAT32_C(   837.01), EASYSIMD_FLOAT32_C(  -753.98), EASYSIMD_FLOAT32_C(   167.51), EASYSIMD_FLOAT32_C(   898.60),
                         EASYSIMD_FLOAT32_C(   -36.68), EASYSIMD_FLOAT32_C(  -931.19), EASYSIMD_FLOAT32_C(   230.22), EASYSIMD_FLOAT32_C(  -885.80),
                         EASYSIMD_FLOAT32_C(  -894.49), EASYSIMD_FLOAT32_C(  -402.23), EASYSIMD_FLOAT32_C(   -68.60), EASYSIMD_FLOAT32_C(  -153.88)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   247.18), EASYSIMD_FLOAT32_C(   507.40), EASYSIMD_FLOAT32_C(  -715.17), EASYSIMD_FLOAT32_C(   785.48),
                         EASYSIMD_FLOAT32_C(  -543.41), EASYSIMD_FLOAT32_C(   761.08), EASYSIMD_FLOAT32_C(   479.07), EASYSIMD_FLOAT32_C(  -938.93),
                         EASYSIMD_FLOAT32_C(  -655.56), EASYSIMD_FLOAT32_C(   618.55), EASYSIMD_FLOAT32_C(   224.83), EASYSIMD_FLOAT32_C(  -983.99),
                         EASYSIMD_FLOAT32_C(   -18.22), EASYSIMD_FLOAT32_C(  -142.62), EASYSIMD_FLOAT32_C(   120.01), EASYSIMD_FLOAT32_C(   186.92)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   168.20), EASYSIMD_FLOAT32_C(   114.93), EASYSIMD_FLOAT32_C(   695.03), EASYSIMD_FLOAT32_C( -1338.45),
                         EASYSIMD_FLOAT32_C(    90.17), EASYSIMD_FLOAT32_C(   891.02), EASYSIMD_FLOAT32_C(  -456.46), EASYSIMD_FLOAT32_C(   952.55),
                         EASYSIMD_FLOAT32_C(   618.88), EASYSIMD_FLOAT32_C(  -533.35), EASYSIMD_FLOAT32_C(  -130.02), EASYSIMD_FLOAT32_C(    98.19),
                         EASYSIMD_FLOAT32_C(  -876.27), EASYSIMD_FLOAT32_C(  -259.61), EASYSIMD_FLOAT32_C(  -188.61), EASYSIMD_FLOAT32_C(  -340.80)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -586.97), EASYSIMD_FLOAT32_C(  -706.71), EASYSIMD_FLOAT32_C(   862.31), EASYSIMD_FLOAT32_C(   901.76),
                         EASYSIMD_FLOAT32_C(  -777.23), EASYSIMD_FLOAT32_C(  -615.23), EASYSIMD_FLOAT32_C(   540.06), EASYSIMD_FLOAT32_C(  -837.05),
                         EASYSIMD_FLOAT32_C(   896.68), EASYSIMD_FLOAT32_C(  -818.79), EASYSIMD_FLOAT32_C(  -146.21), EASYSIMD_FLOAT32_C(  -751.20),
                         EASYSIMD_FLOAT32_C(  -724.86), EASYSIMD_FLOAT32_C(  -446.10), EASYSIMD_FLOAT32_C(   747.21), EASYSIMD_FLOAT32_C(  -830.22)),
      UINT16_C(24145),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   809.72), EASYSIMD_FLOAT32_C(  -191.45), EASYSIMD_FLOAT32_C(  -687.88), EASYSIMD_FLOAT32_C(  -561.69),
                         EASYSIMD_FLOAT32_C(   623.06), EASYSIMD_FLOAT32_C(  -685.16), EASYSIMD_FLOAT32_C(   155.59), EASYSIMD_FLOAT32_C(   -91.67),
                         EASYSIMD_FLOAT32_C(  -292.32), EASYSIMD_FLOAT32_C(   436.29), EASYSIMD_FLOAT32_C(   682.53), EASYSIMD_FLOAT32_C(  -427.71),
                         EASYSIMD_FLOAT32_C(  -252.26), EASYSIMD_FLOAT32_C(  -814.33), EASYSIMD_FLOAT32_C(  -116.78), EASYSIMD_FLOAT32_C(  -176.18)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -476.63), EASYSIMD_FLOAT32_C(  -403.49), EASYSIMD_FLOAT32_C(  -129.06), EASYSIMD_FLOAT32_C(  -540.32),
                         EASYSIMD_FLOAT32_C(  -296.84), EASYSIMD_FLOAT32_C(   354.93), EASYSIMD_FLOAT32_C(   301.70), EASYSIMD_FLOAT32_C(   818.26),
                         EASYSIMD_FLOAT32_C(   152.41), EASYSIMD_FLOAT32_C(    -7.33), EASYSIMD_FLOAT32_C(   901.12), EASYSIMD_FLOAT32_C(   276.49),
                         EASYSIMD_FLOAT32_C(  -421.45), EASYSIMD_FLOAT32_C(   -19.17), EASYSIMD_FLOAT32_C(   559.47), EASYSIMD_FLOAT32_C(   -62.60)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -586.97), EASYSIMD_FLOAT32_C(   212.04), EASYSIMD_FLOAT32_C(   862.31), EASYSIMD_FLOAT32_C(   -21.37),
                         EASYSIMD_FLOAT32_C(   919.90), EASYSIMD_FLOAT32_C( -1040.09), EASYSIMD_FLOAT32_C(  -146.11), EASYSIMD_FLOAT32_C(  -837.05),
                         EASYSIMD_FLOAT32_C(   896.68), EASYSIMD_FLOAT32_C(   443.62), EASYSIMD_FLOAT32_C(  -146.21), EASYSIMD_FLOAT32_C(  -704.20),
                         EASYSIMD_FLOAT32_C(  -724.86), EASYSIMD_FLOAT32_C(  -446.10), EASYSIMD_FLOAT32_C(   747.21), EASYSIMD_FLOAT32_C(  -113.58)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   853.44), EASYSIMD_FLOAT32_C(   804.93), EASYSIMD_FLOAT32_C(   753.54), EASYSIMD_FLOAT32_C(   129.42),
                         EASYSIMD_FLOAT32_C(  -911.24), EASYSIMD_FLOAT32_C(  -795.01), EASYSIMD_FLOAT32_C(  -264.21), EASYSIMD_FLOAT32_C(   110.23),
                         EASYSIMD_FLOAT32_C(   779.42), EASYSIMD_FLOAT32_C(   756.19), EASYSIMD_FLOAT32_C(   -61.94), EASYSIMD_FLOAT32_C(  -845.71),
                         EASYSIMD_FLOAT32_C(   522.75), EASYSIMD_FLOAT32_C(   703.06), EASYSIMD_FLOAT32_C(   989.80), EASYSIMD_FLOAT32_C(   594.14)),
      UINT16_C(58122),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   774.43), EASYSIMD_FLOAT32_C(   251.56), EASYSIMD_FLOAT32_C(  -915.66), EASYSIMD_FLOAT32_C(  -492.31),
                         EASYSIMD_FLOAT32_C(   722.32), EASYSIMD_FLOAT32_C(   853.19), EASYSIMD_FLOAT32_C(   466.28), EASYSIMD_FLOAT32_C(   573.97),
                         EASYSIMD_FLOAT32_C(  -516.73), EASYSIMD_FLOAT32_C(  -267.27), EASYSIMD_FLOAT32_C(   110.95), EASYSIMD_FLOAT32_C(   -68.16),
                         EASYSIMD_FLOAT32_C(  -400.30), EASYSIMD_FLOAT32_C(   327.53), EASYSIMD_FLOAT32_C(  -638.51), EASYSIMD_FLOAT32_C(   -96.92)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   101.96), EASYSIMD_FLOAT32_C(  -734.61), EASYSIMD_FLOAT32_C(   219.43), EASYSIMD_FLOAT32_C(  -507.66),
                         EASYSIMD_FLOAT32_C(  -747.54), EASYSIMD_FLOAT32_C(   794.68), EASYSIMD_FLOAT32_C(  -663.99), EASYSIMD_FLOAT32_C(  -123.94),
                         EASYSIMD_FLOAT32_C(  -793.12), EASYSIMD_FLOAT32_C(   673.57), EASYSIMD_FLOAT32_C(  -777.14), EASYSIMD_FLOAT32_C(   175.88),
                         EASYSIMD_FLOAT32_C(  -792.24), EASYSIMD_FLOAT32_C(  -246.51), EASYSIMD_FLOAT32_C(   848.21), EASYSIMD_FLOAT32_C(  -124.15)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   672.47), EASYSIMD_FLOAT32_C(   986.17), EASYSIMD_FLOAT32_C( -1135.09), EASYSIMD_FLOAT32_C(   129.42),
                         EASYSIMD_FLOAT32_C(  -911.24), EASYSIMD_FLOAT32_C(  -795.01), EASYSIMD_FLOAT32_C(  1130.27), EASYSIMD_FLOAT32_C(   697.91),
                         EASYSIMD_FLOAT32_C(   779.42), EASYSIMD_FLOAT32_C(   756.19), EASYSIMD_FLOAT32_C(   -61.94), EASYSIMD_FLOAT32_C(  -845.71),
                         EASYSIMD_FLOAT32_C(   391.94), EASYSIMD_FLOAT32_C(   703.06), EASYSIMD_FLOAT32_C( -1486.72), EASYSIMD_FLOAT32_C(   594.14)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -670.29), EASYSIMD_FLOAT32_C(   821.01), EASYSIMD_FLOAT32_C(  -293.06), EASYSIMD_FLOAT32_C(   -56.42),
                         EASYSIMD_FLOAT32_C(  -163.64), EASYSIMD_FLOAT32_C(  -919.47), EASYSIMD_FLOAT32_C(   636.75), EASYSIMD_FLOAT32_C(   555.64),
                         EASYSIMD_FLOAT32_C(   630.28), EASYSIMD_FLOAT32_C(   798.33), EASYSIMD_FLOAT32_C(  -536.88), EASYSIMD_FLOAT32_C(   256.29),
                         EASYSIMD_FLOAT32_C(   834.99), EASYSIMD_FLOAT32_C(  -678.50), EASYSIMD_FLOAT32_C(  -716.28), EASYSIMD_FLOAT32_C(  -235.17)),
      UINT16_C( 7968),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   575.18), EASYSIMD_FLOAT32_C(  -655.63), EASYSIMD_FLOAT32_C(   986.91), EASYSIMD_FLOAT32_C(   710.96),
                         EASYSIMD_FLOAT32_C(   921.30), EASYSIMD_FLOAT32_C(   -96.00), EASYSIMD_FLOAT32_C(   -68.75), EASYSIMD_FLOAT32_C(  -119.17),
                         EASYSIMD_FLOAT32_C(  -795.52), EASYSIMD_FLOAT32_C(  -851.06), EASYSIMD_FLOAT32_C(   982.58), EASYSIMD_FLOAT32_C(   432.45),
                         EASYSIMD_FLOAT32_C(   834.71), EASYSIMD_FLOAT32_C(  -931.48), EASYSIMD_FLOAT32_C(   421.86), EASYSIMD_FLOAT32_C(   549.54)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   699.42), EASYSIMD_FLOAT32_C(  -430.21), EASYSIMD_FLOAT32_C(  -842.83), EASYSIMD_FLOAT32_C(  -375.32),
                         EASYSIMD_FLOAT32_C(  -889.13), EASYSIMD_FLOAT32_C(    77.46), EASYSIMD_FLOAT32_C(  -426.32), EASYSIMD_FLOAT32_C(  -319.52),
                         EASYSIMD_FLOAT32_C(   633.46), EASYSIMD_FLOAT32_C(  -484.05), EASYSIMD_FLOAT32_C(   991.09), EASYSIMD_FLOAT32_C(   894.84),
                         EASYSIMD_FLOAT32_C(   148.17), EASYSIMD_FLOAT32_C(  -167.11), EASYSIMD_FLOAT32_C(  -811.87), EASYSIMD_FLOAT32_C(  -574.29)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -670.29), EASYSIMD_FLOAT32_C(   821.01), EASYSIMD_FLOAT32_C(  -293.06), EASYSIMD_FLOAT32_C(  1086.28),
                         EASYSIMD_FLOAT32_C(  1810.43), EASYSIMD_FLOAT32_C(  -173.46), EASYSIMD_FLOAT32_C(   357.57), EASYSIMD_FLOAT32_C(   200.35),
                         EASYSIMD_FLOAT32_C(   630.28), EASYSIMD_FLOAT32_C(   798.33), EASYSIMD_FLOAT32_C(    -8.51), EASYSIMD_FLOAT32_C(   256.29),
                         EASYSIMD_FLOAT32_C(   834.99), EASYSIMD_FLOAT32_C(  -678.50), EASYSIMD_FLOAT32_C(  -716.28), EASYSIMD_FLOAT32_C(  -235.17)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   640.00), EASYSIMD_FLOAT32_C(   440.55), EASYSIMD_FLOAT32_C(   793.44), EASYSIMD_FLOAT32_C(   554.05),
                         EASYSIMD_FLOAT32_C(   245.74), EASYSIMD_FLOAT32_C(  -388.16), EASYSIMD_FLOAT32_C(   -27.32), EASYSIMD_FLOAT32_C(  -923.44),
                         EASYSIMD_FLOAT32_C(   109.81), EASYSIMD_FLOAT32_C(   855.67), EASYSIMD_FLOAT32_C(  -513.53), EASYSIMD_FLOAT32_C(  -921.47),
                         EASYSIMD_FLOAT32_C(  -410.90), EASYSIMD_FLOAT32_C(  -404.15), EASYSIMD_FLOAT32_C(  -502.43), EASYSIMD_FLOAT32_C(  -674.13)),
      UINT16_C(34235),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   455.94), EASYSIMD_FLOAT32_C(   822.75), EASYSIMD_FLOAT32_C(   672.52), EASYSIMD_FLOAT32_C(   418.16),
                         EASYSIMD_FLOAT32_C(   993.17), EASYSIMD_FLOAT32_C(  -581.12), EASYSIMD_FLOAT32_C(   737.02), EASYSIMD_FLOAT32_C(   -48.12),
                         EASYSIMD_FLOAT32_C(   169.53), EASYSIMD_FLOAT32_C(   875.02), EASYSIMD_FLOAT32_C(   325.94), EASYSIMD_FLOAT32_C(  -197.05),
                         EASYSIMD_FLOAT32_C(   209.80), EASYSIMD_FLOAT32_C(   679.16), EASYSIMD_FLOAT32_C(  -743.34), EASYSIMD_FLOAT32_C(   192.93)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -400.32), EASYSIMD_FLOAT32_C(   747.89), EASYSIMD_FLOAT32_C(  -417.14), EASYSIMD_FLOAT32_C(  -149.76),
                         EASYSIMD_FLOAT32_C(  -769.13), EASYSIMD_FLOAT32_C(   952.70), EASYSIMD_FLOAT32_C(    55.59), EASYSIMD_FLOAT32_C(  -118.59),
                         EASYSIMD_FLOAT32_C(  -651.36), EASYSIMD_FLOAT32_C(   213.50), EASYSIMD_FLOAT32_C(   998.39), EASYSIMD_FLOAT32_C(   155.85),
                         EASYSIMD_FLOAT32_C(   985.22), EASYSIMD_FLOAT32_C(  -399.37), EASYSIMD_FLOAT32_C(  -660.54), EASYSIMD_FLOAT32_C(  -918.87)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   856.26), EASYSIMD_FLOAT32_C(   440.55), EASYSIMD_FLOAT32_C(   793.44), EASYSIMD_FLOAT32_C(   554.05),
                         EASYSIMD_FLOAT32_C(   245.74), EASYSIMD_FLOAT32_C( -1533.82), EASYSIMD_FLOAT32_C(   -27.32), EASYSIMD_FLOAT32_C(    70.47),
                         EASYSIMD_FLOAT32_C(   820.89), EASYSIMD_FLOAT32_C(   855.67), EASYSIMD_FLOAT32_C(  -672.45), EASYSIMD_FLOAT32_C(  -352.90),
                         EASYSIMD_FLOAT32_C(  -775.42), EASYSIMD_FLOAT32_C(  -404.15), EASYSIMD_FLOAT32_C(   -82.80), EASYSIMD_FLOAT32_C(  1111.80)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -717.43), EASYSIMD_FLOAT32_C(   307.65), EASYSIMD_FLOAT32_C(  -776.64), EASYSIMD_FLOAT32_C(   883.24),
                         EASYSIMD_FLOAT32_C(   462.38), EASYSIMD_FLOAT32_C(   941.52), EASYSIMD_FLOAT32_C(   465.21), EASYSIMD_FLOAT32_C(   772.92),
                         EASYSIMD_FLOAT32_C(  -448.96), EASYSIMD_FLOAT32_C(   167.95), EASYSIMD_FLOAT32_C(  -770.79), EASYSIMD_FLOAT32_C(   607.02),
                         EASYSIMD_FLOAT32_C(   588.25), EASYSIMD_FLOAT32_C(  -430.65), EASYSIMD_FLOAT32_C(  -379.22), EASYSIMD_FLOAT32_C(    62.66)),
      UINT16_C(21184),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   745.31), EASYSIMD_FLOAT32_C(   528.47), EASYSIMD_FLOAT32_C(   260.56), EASYSIMD_FLOAT32_C(   756.92),
                         EASYSIMD_FLOAT32_C(  -237.78), EASYSIMD_FLOAT32_C(   890.33), EASYSIMD_FLOAT32_C(  -276.66), EASYSIMD_FLOAT32_C(  -845.25),
                         EASYSIMD_FLOAT32_C(    73.01), EASYSIMD_FLOAT32_C(  -169.10), EASYSIMD_FLOAT32_C(  -390.26), EASYSIMD_FLOAT32_C(    55.87),
                         EASYSIMD_FLOAT32_C(   461.32), EASYSIMD_FLOAT32_C(  -911.03), EASYSIMD_FLOAT32_C(   362.01), EASYSIMD_FLOAT32_C(   998.06)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   177.96), EASYSIMD_FLOAT32_C(  -105.40), EASYSIMD_FLOAT32_C(  -516.55), EASYSIMD_FLOAT32_C(   -62.31),
                         EASYSIMD_FLOAT32_C(  -757.68), EASYSIMD_FLOAT32_C(   665.34), EASYSIMD_FLOAT32_C(   689.63), EASYSIMD_FLOAT32_C(   938.32),
                         EASYSIMD_FLOAT32_C(  -408.00), EASYSIMD_FLOAT32_C(   998.26), EASYSIMD_FLOAT32_C(  -263.70), EASYSIMD_FLOAT32_C(   807.54),
                         EASYSIMD_FLOAT32_C(   485.72), EASYSIMD_FLOAT32_C(   -74.68), EASYSIMD_FLOAT32_C(   725.36), EASYSIMD_FLOAT32_C(   301.00)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -717.43), EASYSIMD_FLOAT32_C(   633.87), EASYSIMD_FLOAT32_C(  -776.64), EASYSIMD_FLOAT32_C(   819.23),
                         EASYSIMD_FLOAT32_C(   462.38), EASYSIMD_FLOAT32_C(   941.52), EASYSIMD_FLOAT32_C(  -966.29), EASYSIMD_FLOAT32_C(   772.92),
                         EASYSIMD_FLOAT32_C(   481.01), EASYSIMD_FLOAT32_C( -1167.36), EASYSIMD_FLOAT32_C(  -770.79), EASYSIMD_FLOAT32_C(   607.02),
                         EASYSIMD_FLOAT32_C(   588.25), EASYSIMD_FLOAT32_C(  -430.65), EASYSIMD_FLOAT32_C(  -379.22), EASYSIMD_FLOAT32_C(    62.66)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sub_ps(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sub_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_sub_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d src;
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -621.09), EASYSIMD_FLOAT64_C(  350.18),
                         EASYSIMD_FLOAT64_C(  873.40), EASYSIMD_FLOAT64_C( -136.67),
                         EASYSIMD_FLOAT64_C( -484.90), EASYSIMD_FLOAT64_C(  672.37),
                         EASYSIMD_FLOAT64_C( -983.97), EASYSIMD_FLOAT64_C( -747.18)),
      UINT8_C(213),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -615.22), EASYSIMD_FLOAT64_C(  861.93),
                         EASYSIMD_FLOAT64_C(  -99.63), EASYSIMD_FLOAT64_C( -760.72),
                         EASYSIMD_FLOAT64_C(  803.54), EASYSIMD_FLOAT64_C( -811.65),
                         EASYSIMD_FLOAT64_C( -888.48), EASYSIMD_FLOAT64_C(  353.19)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  451.54), EASYSIMD_FLOAT64_C(  490.96),
                         EASYSIMD_FLOAT64_C( -563.07), EASYSIMD_FLOAT64_C( -968.95),
                         EASYSIMD_FLOAT64_C( -964.80), EASYSIMD_FLOAT64_C( -259.48),
                         EASYSIMD_FLOAT64_C(  -97.31), EASYSIMD_FLOAT64_C(  696.26)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(-1066.76), EASYSIMD_FLOAT64_C(  370.97),
                         EASYSIMD_FLOAT64_C(  873.40), EASYSIMD_FLOAT64_C(  208.23),
                         EASYSIMD_FLOAT64_C( -484.90), EASYSIMD_FLOAT64_C( -552.17),
                         EASYSIMD_FLOAT64_C( -983.97), EASYSIMD_FLOAT64_C( -343.07)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  956.74), EASYSIMD_FLOAT64_C(  507.70),
                         EASYSIMD_FLOAT64_C(  525.25), EASYSIMD_FLOAT64_C( -653.24),
                         EASYSIMD_FLOAT64_C( -748.66), EASYSIMD_FLOAT64_C(  738.72),
                         EASYSIMD_FLOAT64_C(  584.29), EASYSIMD_FLOAT64_C( -344.89)),
      UINT8_C(200),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -70.99), EASYSIMD_FLOAT64_C( -712.48),
                         EASYSIMD_FLOAT64_C(  721.37), EASYSIMD_FLOAT64_C(  290.11),
                         EASYSIMD_FLOAT64_C(  739.65), EASYSIMD_FLOAT64_C(  378.13),
                         EASYSIMD_FLOAT64_C(  523.23), EASYSIMD_FLOAT64_C(  338.41)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -243.21), EASYSIMD_FLOAT64_C(   71.87),
                         EASYSIMD_FLOAT64_C(   81.06), EASYSIMD_FLOAT64_C(  409.05),
                         EASYSIMD_FLOAT64_C( -595.58), EASYSIMD_FLOAT64_C(  278.33),
                         EASYSIMD_FLOAT64_C( -484.02), EASYSIMD_FLOAT64_C( -861.59)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  172.22), EASYSIMD_FLOAT64_C( -784.35),
                         EASYSIMD_FLOAT64_C(  525.25), EASYSIMD_FLOAT64_C( -653.24),
                         EASYSIMD_FLOAT64_C( 1335.23), EASYSIMD_FLOAT64_C(  738.72),
                         EASYSIMD_FLOAT64_C(  584.29), EASYSIMD_FLOAT64_C( -344.89)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  475.39), EASYSIMD_FLOAT64_C(  345.93),
                         EASYSIMD_FLOAT64_C(  233.76), EASYSIMD_FLOAT64_C( -401.11),
                         EASYSIMD_FLOAT64_C( -964.57), EASYSIMD_FLOAT64_C(  939.13),
                         EASYSIMD_FLOAT64_C( -392.63), EASYSIMD_FLOAT64_C( -585.02)),
      UINT8_C( 75),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  496.11), EASYSIMD_FLOAT64_C( -235.94),
                         EASYSIMD_FLOAT64_C( -715.35), EASYSIMD_FLOAT64_C(  338.71),
                         EASYSIMD_FLOAT64_C( -776.11), EASYSIMD_FLOAT64_C(  941.96),
                         EASYSIMD_FLOAT64_C(   76.10), EASYSIMD_FLOAT64_C( -188.31)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  824.70), EASYSIMD_FLOAT64_C( -886.45),
                         EASYSIMD_FLOAT64_C(  497.17), EASYSIMD_FLOAT64_C( -965.13),
                         EASYSIMD_FLOAT64_C( -601.99), EASYSIMD_FLOAT64_C( -657.07),
                         EASYSIMD_FLOAT64_C(  201.36), EASYSIMD_FLOAT64_C( -807.98)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  475.39), EASYSIMD_FLOAT64_C(  650.51),
                         EASYSIMD_FLOAT64_C(  233.76), EASYSIMD_FLOAT64_C( -401.11),
                         EASYSIMD_FLOAT64_C( -174.12), EASYSIMD_FLOAT64_C(  939.13),
                         EASYSIMD_FLOAT64_C( -125.26), EASYSIMD_FLOAT64_C(  619.67)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -246.72), EASYSIMD_FLOAT64_C( -493.17),
                         EASYSIMD_FLOAT64_C( -501.93), EASYSIMD_FLOAT64_C(  -95.50),
                         EASYSIMD_FLOAT64_C(  754.55), EASYSIMD_FLOAT64_C( -990.48),
                         EASYSIMD_FLOAT64_C( -396.36), EASYSIMD_FLOAT64_C( -466.97)),
      UINT8_C( 69),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  601.28), EASYSIMD_FLOAT64_C( -873.85),
                         EASYSIMD_FLOAT64_C( -689.96), EASYSIMD_FLOAT64_C(   31.77),
                         EASYSIMD_FLOAT64_C(  -97.11), EASYSIMD_FLOAT64_C(  971.94),
                         EASYSIMD_FLOAT64_C(  389.02), EASYSIMD_FLOAT64_C( -650.79)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  136.61), EASYSIMD_FLOAT64_C(  436.94),
                         EASYSIMD_FLOAT64_C( -777.02), EASYSIMD_FLOAT64_C(  166.29),
                         EASYSIMD_FLOAT64_C( -377.75), EASYSIMD_FLOAT64_C(   71.16),
                         EASYSIMD_FLOAT64_C(  481.01), EASYSIMD_FLOAT64_C( -926.81)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -246.72), EASYSIMD_FLOAT64_C(-1310.79),
                         EASYSIMD_FLOAT64_C( -501.93), EASYSIMD_FLOAT64_C(  -95.50),
                         EASYSIMD_FLOAT64_C(  754.55), EASYSIMD_FLOAT64_C(  900.78),
                         EASYSIMD_FLOAT64_C( -396.36), EASYSIMD_FLOAT64_C(  276.02)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -389.46), EASYSIMD_FLOAT64_C(   -8.03),
                         EASYSIMD_FLOAT64_C( -523.51), EASYSIMD_FLOAT64_C(  466.89),
                         EASYSIMD_FLOAT64_C(  698.90), EASYSIMD_FLOAT64_C( -346.04),
                         EASYSIMD_FLOAT64_C( -734.67), EASYSIMD_FLOAT64_C(  404.34)),
      UINT8_C(100),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  989.13), EASYSIMD_FLOAT64_C(  228.14),
                         EASYSIMD_FLOAT64_C(  840.94), EASYSIMD_FLOAT64_C( -718.83),
                         EASYSIMD_FLOAT64_C(  274.95), EASYSIMD_FLOAT64_C(  -99.21),
                         EASYSIMD_FLOAT64_C(   84.76), EASYSIMD_FLOAT64_C( -295.84)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -956.22), EASYSIMD_FLOAT64_C(  564.94),
                         EASYSIMD_FLOAT64_C(  -97.16), EASYSIMD_FLOAT64_C( -407.99),
                         EASYSIMD_FLOAT64_C(  352.62), EASYSIMD_FLOAT64_C(  244.25),
                         EASYSIMD_FLOAT64_C(   43.92), EASYSIMD_FLOAT64_C(  624.69)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -389.46), EASYSIMD_FLOAT64_C( -336.80),
                         EASYSIMD_FLOAT64_C(  938.10), EASYSIMD_FLOAT64_C(  466.89),
                         EASYSIMD_FLOAT64_C(  698.90), EASYSIMD_FLOAT64_C( -343.46),
                         EASYSIMD_FLOAT64_C( -734.67), EASYSIMD_FLOAT64_C(  404.34)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -571.96), EASYSIMD_FLOAT64_C(   40.27),
                         EASYSIMD_FLOAT64_C(  676.69), EASYSIMD_FLOAT64_C( -150.37),
                         EASYSIMD_FLOAT64_C(  945.34), EASYSIMD_FLOAT64_C(   75.83),
                         EASYSIMD_FLOAT64_C(   64.75), EASYSIMD_FLOAT64_C(  239.06)),
      UINT8_C(209),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  792.47), EASYSIMD_FLOAT64_C( -265.19),
                         EASYSIMD_FLOAT64_C( -768.95), EASYSIMD_FLOAT64_C(  515.15),
                         EASYSIMD_FLOAT64_C(  350.59), EASYSIMD_FLOAT64_C(  422.68),
                         EASYSIMD_FLOAT64_C(  582.99), EASYSIMD_FLOAT64_C( -985.50)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   12.38), EASYSIMD_FLOAT64_C(  -71.80),
                         EASYSIMD_FLOAT64_C(  363.01), EASYSIMD_FLOAT64_C( -195.65),
                         EASYSIMD_FLOAT64_C(  967.47), EASYSIMD_FLOAT64_C(   -4.13),
                         EASYSIMD_FLOAT64_C( -478.81), EASYSIMD_FLOAT64_C(  909.10)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  780.09), EASYSIMD_FLOAT64_C( -193.39),
                         EASYSIMD_FLOAT64_C(  676.69), EASYSIMD_FLOAT64_C(  710.80),
                         EASYSIMD_FLOAT64_C(  945.34), EASYSIMD_FLOAT64_C(   75.83),
                         EASYSIMD_FLOAT64_C(   64.75), EASYSIMD_FLOAT64_C(-1894.60)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -879.88), EASYSIMD_FLOAT64_C( -687.95),
                         EASYSIMD_FLOAT64_C( -892.89), EASYSIMD_FLOAT64_C( -642.85),
                         EASYSIMD_FLOAT64_C(  533.08), EASYSIMD_FLOAT64_C(  898.29),
                         EASYSIMD_FLOAT64_C(  -29.99), EASYSIMD_FLOAT64_C(    5.58)),
      UINT8_C(186),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  649.80), EASYSIMD_FLOAT64_C( -257.91),
                         EASYSIMD_FLOAT64_C(  356.56), EASYSIMD_FLOAT64_C(  567.70),
                         EASYSIMD_FLOAT64_C(  -80.43), EASYSIMD_FLOAT64_C( -499.15),
                         EASYSIMD_FLOAT64_C( -866.12), EASYSIMD_FLOAT64_C(  639.40)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  702.45), EASYSIMD_FLOAT64_C(  464.79),
                         EASYSIMD_FLOAT64_C(  387.80), EASYSIMD_FLOAT64_C( -528.10),
                         EASYSIMD_FLOAT64_C( -409.82), EASYSIMD_FLOAT64_C( -696.40),
                         EASYSIMD_FLOAT64_C(  455.43), EASYSIMD_FLOAT64_C(  856.81)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -52.65), EASYSIMD_FLOAT64_C( -687.95),
                         EASYSIMD_FLOAT64_C(  -31.24), EASYSIMD_FLOAT64_C( 1095.80),
                         EASYSIMD_FLOAT64_C(  329.39), EASYSIMD_FLOAT64_C(  898.29),
                         EASYSIMD_FLOAT64_C(-1321.55), EASYSIMD_FLOAT64_C(    5.58)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -750.95), EASYSIMD_FLOAT64_C(  203.46),
                         EASYSIMD_FLOAT64_C(  194.87), EASYSIMD_FLOAT64_C(  667.81),
                         EASYSIMD_FLOAT64_C( -258.76), EASYSIMD_FLOAT64_C(  897.89),
                         EASYSIMD_FLOAT64_C(  571.10), EASYSIMD_FLOAT64_C( -320.96)),
      UINT8_C( 56),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -938.69), EASYSIMD_FLOAT64_C(   74.05),
                         EASYSIMD_FLOAT64_C( -981.48), EASYSIMD_FLOAT64_C( -656.78),
                         EASYSIMD_FLOAT64_C( -794.37), EASYSIMD_FLOAT64_C(  177.36),
                         EASYSIMD_FLOAT64_C(  380.50), EASYSIMD_FLOAT64_C(  812.91)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -10.37), EASYSIMD_FLOAT64_C( -894.99),
                         EASYSIMD_FLOAT64_C( -148.09), EASYSIMD_FLOAT64_C(  314.75),
                         EASYSIMD_FLOAT64_C( -740.28), EASYSIMD_FLOAT64_C( -372.00),
                         EASYSIMD_FLOAT64_C( -357.36), EASYSIMD_FLOAT64_C( -791.79)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -750.95), EASYSIMD_FLOAT64_C(  203.46),
                         EASYSIMD_FLOAT64_C( -833.39), EASYSIMD_FLOAT64_C( -971.53),
                         EASYSIMD_FLOAT64_C(  -54.09), EASYSIMD_FLOAT64_C(  897.89),
                         EASYSIMD_FLOAT64_C(  571.10), EASYSIMD_FLOAT64_C( -320.96)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d b = test_vec[i].b;
    easysimd__m512d src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_sub_pd(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_sub_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_sub_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT16_C(42308),
      easysimd_mm512_set_epi32(INT32_C( 1724059665), INT32_C(-1181331137), INT32_C( -956878955), INT32_C( 1254662027),
                            INT32_C( -334196329), INT32_C( -462422656), INT32_C(  391895544), INT32_C( 1081692585),
                            INT32_C(-1420053828), INT32_C(-1016697350), INT32_C( 1995028549), INT32_C(-2003231670),
                            INT32_C( 1672190791), INT32_C(  255109958), INT32_C(-2019884289), INT32_C(-1398510440)),
      easysimd_mm512_set_epi32(INT32_C( 1758500044), INT32_C(  727344602), INT32_C(-1303831643), INT32_C( 1021495274),
                            INT32_C(-2113209677), INT32_C( 1628670789), INT32_C(  684532718), INT32_C( 1920084108),
                            INT32_C( -516238646), INT32_C( 1525557846), INT32_C( 1058541430), INT32_C(  232836803),
                            INT32_C( 1824295576), INT32_C(-1334166784), INT32_C(-1267999587), INT32_C( 1992895333)),
      easysimd_mm512_set_epi32(INT32_C(  -34440379), INT32_C(          0), INT32_C(  346952688), INT32_C(          0),
                            INT32_C(          0), INT32_C(-2091093445), INT32_C(          0), INT32_C( -838391523),
                            INT32_C(          0), INT32_C( 1752712100), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1589276742), INT32_C(          0), INT32_C(          0)) },
    { UINT16_C(57112),
      easysimd_mm512_set_epi32(INT32_C( 1427084198), INT32_C(  800677318), INT32_C(-1624555826), INT32_C( -189169736),
                            INT32_C( -500462405), INT32_C(  393027187), INT32_C( -215642095), INT32_C( 1795082661),
                            INT32_C(-1120274966), INT32_C( 1416315501), INT32_C( 2071781830), INT32_C( 1981287236),
                            INT32_C( 1895228887), INT32_C( -102536112), INT32_C(-1592734830), INT32_C(-1858725491)),
      easysimd_mm512_set_epi32(INT32_C( -450919787), INT32_C( 1299130560), INT32_C( 1762509692), INT32_C(  310818231),
                            INT32_C( -225659966), INT32_C(-1193662266), INT32_C(  959080993), INT32_C(  -80526553),
                            INT32_C( -695376176), INT32_C(  -26080833), INT32_C(  542712435), INT32_C( 1266358760),
                            INT32_C(  181254235), INT32_C(-2068678559), INT32_C( 1863289430), INT32_C( -269529302)),
      easysimd_mm512_set_epi32(INT32_C( 1878003985), INT32_C( -498453242), INT32_C(          0), INT32_C( -499987967),
                            INT32_C( -274802439), INT32_C( 1586689453), INT32_C(-1174723088), INT32_C( 1875609214),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(  714928476),
                            INT32_C( 1713974652), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
    { UINT16_C(45985),
      easysimd_mm512_set_epi32(INT32_C(-1997599226), INT32_C( 1542236612), INT32_C(  969579913), INT32_C(-1642088433),
                            INT32_C(  579114801), INT32_C(-1194258935), INT32_C(-1422143462), INT32_C( 1748279001),
                            INT32_C(-1953627340), INT32_C( 1674288033), INT32_C(  717963559), INT32_C(   34905906),
                            INT32_C( -149768860), INT32_C( 1400155142), INT32_C( 1757125654), INT32_C(-1787496119)),
      easysimd_mm512_set_epi32(INT32_C(   11674598), INT32_C( 1849959427), INT32_C(-1203439394), INT32_C( -261642074),
                            INT32_C(-2062167113), INT32_C( 1504166558), INT32_C( -111161554), INT32_C( -367200138),
                            INT32_C( 1040642836), INT32_C(  378025736), INT32_C( 1031970925), INT32_C(-1474878922),
                            INT32_C(-1560910320), INT32_C( 1296215099), INT32_C(-1595601438), INT32_C( -126839035)),
      easysimd_mm512_set_epi32(INT32_C(-2009273824), INT32_C(          0), INT32_C(-2121947989), INT32_C(-1380446359),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1310981908), INT32_C( 2115479139),
                            INT32_C( 1300697120), INT32_C(          0), INT32_C( -314007366), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(-1660657084)) },
    { UINT16_C(21153),
      easysimd_mm512_set_epi32(INT32_C( -788633826), INT32_C( 1642420282), INT32_C(  723895008), INT32_C(  207632598),
                            INT32_C(-2079938207), INT32_C( 1754477079), INT32_C( 1798135551), INT32_C(   23449555),
                            INT32_C( -151172429), INT32_C(  677778908), INT32_C(   90905464), INT32_C( 1354586615),
                            INT32_C(-1670436324), INT32_C( -505523122), INT32_C(-1519449460), INT32_C(-1685310582)),
      easysimd_mm512_set_epi32(INT32_C(  799456687), INT32_C(-1358763208), INT32_C(  737687311), INT32_C( 1515407453),
                            INT32_C(  439395016), INT32_C(  -78627541), INT32_C(-1674155016), INT32_C( 1063201251),
                            INT32_C( -686363587), INT32_C(  742525264), INT32_C(  701319512), INT32_C(   24989685),
                            INT32_C( -301118736), INT32_C( -785334161), INT32_C(-1489992316), INT32_C(  306022421)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(-1293783806), INT32_C(          0), INT32_C(-1307774855),
                            INT32_C(          0), INT32_C(          0), INT32_C( -822676729), INT32_C(          0),
                            INT32_C(  535191158), INT32_C(          0), INT32_C( -610414048), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(-1991333003)) },
    { UINT16_C(12143),
      easysimd_mm512_set_epi32(INT32_C( -246629264), INT32_C(  633039851), INT32_C( 1692158737), INT32_C( 1115946871),
                            INT32_C(  309808098), INT32_C( 1170830326), INT32_C( 1350105561), INT32_C(-1022199838),
                            INT32_C(  654046756), INT32_C( 1807741640), INT32_C(  224020334), INT32_C( 1191767429),
                            INT32_C( -990326759), INT32_C(   85294451), INT32_C( -252749112), INT32_C(-1788577569)),
      easysimd_mm512_set_epi32(INT32_C( 1174570840), INT32_C(  974062633), INT32_C(  983904988), INT32_C( 1803536893),
                            INT32_C( 1164598462), INT32_C( 1777437641), INT32_C(-1475760323), INT32_C( 1833217111),
                            INT32_C( 2013842885), INT32_C(  720911006), INT32_C(-1253744600), INT32_C( 1820529236),
                            INT32_C( -314819268), INT32_C(-1926268921), INT32_C( 2108913431), INT32_C( 1190393502)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(  708253749), INT32_C(          0),
                            INT32_C( -854790364), INT32_C( -606607315), INT32_C(-1469101412), INT32_C( 1439550347),
                            INT32_C(          0), INT32_C( 1086830634), INT32_C( 1477764934), INT32_C(          0),
                            INT32_C( -675507491), INT32_C( 2011563372), INT32_C( 1933304753), INT32_C( 1315996225)) },
    { UINT16_C(26005),
      easysimd_mm512_set_epi32(INT32_C( 1813548464), INT32_C( -757290941), INT32_C( 1295512986), INT32_C( 1291803276),
                            INT32_C( 2032260868), INT32_C(  316165049), INT32_C( 1037644878), INT32_C(-1728213057),
                            INT32_C(  231750243), INT32_C( 1220512969), INT32_C(-1711918828), INT32_C( 1618345779),
                            INT32_C( 1444876028), INT32_C( 1881924556), INT32_C(-1672732354), INT32_C(-1497726182)),
      easysimd_mm512_set_epi32(INT32_C(-2042300804), INT32_C( -199486597), INT32_C( -290224964), INT32_C(  -95049939),
                            INT32_C(  242789967), INT32_C(-2042388049), INT32_C(-1526333573), INT32_C( -943172088),
                            INT32_C(-1987449183), INT32_C( -802616226), INT32_C(  743071941), INT32_C(  -28537087),
                            INT32_C(-2054489846), INT32_C( 2118922267), INT32_C( 1876700525), INT32_C(  356823736)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( -557804344), INT32_C( 1585737950), INT32_C(          0),
                            INT32_C(          0), INT32_C(-1936414198), INT32_C(          0), INT32_C( -785040969),
                            INT32_C(-2075767870), INT32_C(          0), INT32_C(          0), INT32_C( 1646882866),
                            INT32_C(          0), INT32_C( -236997711), INT32_C(          0), INT32_C(-1854549918)) },
    { UINT16_C(22214),
      easysimd_mm512_set_epi32(INT32_C( 1255503250), INT32_C(  603134448), INT32_C( 1664652192), INT32_C( -343768171),
                            INT32_C(-1798248429), INT32_C(-1446513257), INT32_C(  127732840), INT32_C(-1651163018),
                            INT32_C(  741467989), INT32_C(  859412594), INT32_C(  472043835), INT32_C( 1771260096),
                            INT32_C(-1144930983), INT32_C(  236371534), INT32_C( 1323254991), INT32_C( 1564105257)),
      easysimd_mm512_set_epi32(INT32_C(  438781482), INT32_C( 1278794690), INT32_C(-1026818029), INT32_C( 2082034838),
                            INT32_C(  -20030271), INT32_C( -682181759), INT32_C( 1547951192), INT32_C(  690567023),
                            INT32_C( -270117367), INT32_C( -771535010), INT32_C(  916148853), INT32_C( 1687091511),
                            INT32_C( -535908173), INT32_C( -185822843), INT32_C( -711684672), INT32_C( -424619293)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C( -675660242), INT32_C(          0), INT32_C( 1869164287),
                            INT32_C(          0), INT32_C( -764331498), INT32_C(-1420218352), INT32_C(          0),
                            INT32_C( 1011585356), INT32_C( 1630947604), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(  422194377), INT32_C( 2034939663), INT32_C(          0)) },
    { UINT16_C(35591),
      easysimd_mm512_set_epi32(INT32_C( 1513047065), INT32_C( -104652818), INT32_C( 1564491564), INT32_C(  -98950215),
                            INT32_C(  631827200), INT32_C( 1322294700), INT32_C(  436005702), INT32_C( 1825722103),
                            INT32_C( 2013933934), INT32_C( -532774987), INT32_C( 1616518393), INT32_C(  803856137),
                            INT32_C(-1663534883), INT32_C(-2021437227), INT32_C(-1476004613), INT32_C( -899510926)),
      easysimd_mm512_set_epi32(INT32_C( -910624932), INT32_C(  209536966), INT32_C(-1923748050), INT32_C(-1520303619),
                            INT32_C( -387141989), INT32_C(  959069600), INT32_C( 1208361371), INT32_C(-1838273096),
                            INT32_C(-1330134815), INT32_C(  126713528), INT32_C( -150313435), INT32_C(-1972942202),
                            INT32_C( 1666269875), INT32_C(-1750237431), INT32_C(  950405946), INT32_C( -725753907)),
      easysimd_mm512_set_epi32(INT32_C(-1871295299), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1018969189), INT32_C(          0), INT32_C( -772355669), INT32_C( -630972097),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( -271199796), INT32_C( 1868556737), INT32_C( -173757019)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_sub_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_sub_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_sub_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i b;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT8_C( 68),
      easysimd_mm512_set_epi64(INT64_C(-5073778595823407211), INT64_C( 5388732377458839959),
                            INT64_C(-1986090184057562632), INT64_C( 4645834279775613628),
                            INT64_C(-4366681866184837051), INT64_C(-8603814507289273529),
                            INT64_C( 1095688928769016575), INT64_C(-6006556600469720682)),
      easysimd_mm512_set_epi64(INT64_C( 3123921281503271845), INT64_C( 4387288797030316723),
                            INT64_C( 6995087775390049262), INT64_C( 8246698453208060618),
                            INT64_C( 6552221057784745846), INT64_C( 1000026456014490264),
                            INT64_C(-5730202701662528355), INT64_C( 8559420281310089233)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C( 1001443580428523236),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 8842903110405787823),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(204),
      easysimd_mm512_set_epi64(INT64_C(-6977414139090468936), INT64_C(-2149469661959479693),
                            INT64_C( -926175743870842459), INT64_C(-4811544340081196435),
                            INT64_C( 8898235206278318916), INT64_C( 8139946092291910736),
                            INT64_C(-6840744003613877875), INT64_C( 6930156028979502872)),
      easysimd_mm512_set_epi64(INT64_C( 7569921486333851063), INT64_C( -969202170885166906),
                            INT64_C( 4119221503364645671), INT64_C(-2986617930068653633),
                            INT64_C( 2330932160723884520), INT64_C(  778481013812787297),
                            INT64_C( 8002767168857919274), INT64_C( 6129279959849065926)),
      easysimd_mm512_set_epi64(INT64_C( 3899408448285231617), INT64_C(-1180267491074312787),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 6567303045554434396), INT64_C( 7361465078479123439),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(192),
      easysimd_mm512_set_epi64(INT64_C(-7052716116295772367), INT64_C(-5129303065907965926),
                            INT64_C( 7508801135919891252), INT64_C( 7191012346537132327),
                            INT64_C(  149919728852448612), INT64_C( 6013620545973361686),
                            INT64_C(-7677237369544501225), INT64_C(-4708878852454120811)),
      easysimd_mm512_set_epi64(INT64_C(-1123744148854811721), INT64_C( 6460346178530692910),
                            INT64_C(-1577112582756044012), INT64_C( 1623608174198300781),
                            INT64_C(-6334556732815677936), INT64_C( 5567201461485768162),
                            INT64_C( -544769504883831290), INT64_C( 6623855812203421065)),
      easysimd_mm512_set_epi64(INT64_C(-5928971967440960646), INT64_C( 6857094829270892780),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(222),
      easysimd_mm512_set_epi64(INT64_C(-8933266575011401193), INT64_C( 7722933385343389651),
                            INT64_C( -649280637934103076), INT64_C(  390435996262291959),
                            INT64_C(-7174469377841015730), INT64_C(-6525985736015203446),
                            INT64_C( 8306044009918255777), INT64_C(   50142018453906435)),
      easysimd_mm512_set_epi64(INT64_C( 1887187227961736491), INT64_C(-7190441041091155485),
                            INT64_C(-2947909158587725488), INT64_C( 3012144368111669237),
                            INT64_C(-1293295119823224721), INT64_C(-6399468268205275115),
                            INT64_C(-3387156489546934214), INT64_C( 3109105385305290966)),
      easysimd_mm512_set_epi64(INT64_C( 7626290270736413932), INT64_C(-3533369647275006480),
                            INT64_C(                   0), INT64_C(-2621708371849377278),
                            INT64_C(-5881174258017791009), INT64_C( -126517467809928331),
                            INT64_C(-6753543574244361625), INT64_C(                   0)) },
    { UINT8_C( 93),
      easysimd_mm512_set_epi64(INT64_C( 5028677960685124057), INT64_C(-4390314873532451292),
                            INT64_C( 7764191223641425774), INT64_C( 5118602135297642521),
                            INT64_C(  366336881617492680), INT64_C(-7681882161808553379),
                            INT64_C( 1609244596442152367), INT64_C(-5835843540630358257)),
      easysimd_mm512_set_epi64(INT64_C( 7634036541593595709), INT64_C( 7873607540226444741),
                            INT64_C( 3096289197137682472), INT64_C( 7819113534012013884),
                            INT64_C(-8273262016887294185), INT64_C( 5112701164509248624),
                            INT64_C( 2718885458801871633), INT64_C( 4792955315328338914)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C( 6182821659950655583),
                            INT64_C(                   0), INT64_C(-2700511398714371363),
                            INT64_C( 8639598898504786865), INT64_C( 5652160747391749613),
                            INT64_C(                   0), INT64_C( 7817945217750854445)) },
    { UINT8_C(190),
      easysimd_mm512_set_epi64(INT64_C( 4456650818438664127), INT64_C(  995359715745565897),
                            INT64_C(-7352635378048303309), INT64_C( 6205695288916304844),
                            INT64_C(-7184330752593853670), INT64_C(  997194843856987541),
                            INT64_C( 5044743345609311273), INT64_C( 4225839747634809341)),
      easysimd_mm512_set_epi64(INT64_C(-6555552775470033400), INT64_C(-8536029239954568098),
                            INT64_C( 3191469689436671745), INT64_C(-8823966696415154149),
                            INT64_C( 8060367379617854136), INT64_C( 7789131346128709699),
                            INT64_C( 5564185907705109132), INT64_C( 8728493965316737977)),
      easysimd_mm512_set_epi64(INT64_C(-7434540479800854089), INT64_C(                   0),
                            INT64_C( 7902639006224576562), INT64_C(-3417082088378092623),
                            INT64_C( 3202045941497843810), INT64_C(-6791936502271722158),
                            INT64_C( -519442562095797859), INT64_C(                   0)) },
    { UINT8_C(175),
      easysimd_mm512_set_epi64(INT64_C(-7091691161933191339), INT64_C( 3691148985472569659),
                            INT64_C( 7607504188179856729), INT64_C( 1015208009558607055),
                            INT64_C( 6717780929629073882), INT64_C( 2140431133564008060),
                            INT64_C( -856788406100589380), INT64_C( -408236379249004977)),
      easysimd_mm512_set_epi64(INT64_C( 2965962783505929737), INT64_C(-3313717634752884107),
                            INT64_C( 7246002868863283379), INT64_C( -798103029951459904),
                            INT64_C(-1823725975430138478), INT64_C( 2590442730915664800),
                            INT64_C(-1476473049354016749), INT64_C(-6212727131917710232)),
      easysimd_mm512_set_epi64(INT64_C( 8389090128270430540), INT64_C(                   0),
                            INT64_C(  361501319316573350), INT64_C(                   0),
                            INT64_C( 8541506905059212360), INT64_C( -450011597351656740),
                            INT64_C(  619684643253427369), INT64_C( 5804490752668705255)) },
    { UINT8_C( 88),
      easysimd_mm512_set_epi64(INT64_C( 8649780386596814773), INT64_C( 6942893632121331465),
                            INT64_C(-7144827915966656299), INT64_C(-6339391538184680078),
                            INT64_C( 7515152281876400903), INT64_C( 1884552116559207362),
                            INT64_C(-4410149851416144746), INT64_C(  -86029355262231679)),
      easysimd_mm512_set_epi64(INT64_C(-5712885529569296712), INT64_C( -645591285152396666),
                            INT64_C( 7156574621979737865), INT64_C( 4081962459563155405),
                            INT64_C( 6498487665674100718), INT64_C( 6719440106443908025),
                            INT64_C( 2713677162045545900), INT64_C( 1872630232785243895)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C( 7588484917273728131),
                            INT64_C(                   0), INT64_C( 8025390075961716133),
                            INT64_C( 1016664616202300185), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i b = test_vec[i].b;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_sub_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_sub_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_sub_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 b;
    easysimd__m512 r;
  } test_vec[8] = {
    { UINT16_C(26074),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -524.33), EASYSIMD_FLOAT32_C(  -241.59), EASYSIMD_FLOAT32_C(  -105.89), EASYSIMD_FLOAT32_C(  -289.61),
                         EASYSIMD_FLOAT32_C(  -891.58), EASYSIMD_FLOAT32_C(   378.73), EASYSIMD_FLOAT32_C(   -71.99), EASYSIMD_FLOAT32_C(   449.90),
                         EASYSIMD_FLOAT32_C(  -415.75), EASYSIMD_FLOAT32_C(   784.67), EASYSIMD_FLOAT32_C(  -496.30), EASYSIMD_FLOAT32_C(   526.56),
                         EASYSIMD_FLOAT32_C(    67.17), EASYSIMD_FLOAT32_C(  -881.21), EASYSIMD_FLOAT32_C(   348.77), EASYSIMD_FLOAT32_C(   537.04)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   392.86), EASYSIMD_FLOAT32_C(    15.96), EASYSIMD_FLOAT32_C(  -681.24), EASYSIMD_FLOAT32_C(   759.61),
                         EASYSIMD_FLOAT32_C(  -507.08), EASYSIMD_FLOAT32_C(  -150.50), EASYSIMD_FLOAT32_C(   409.54), EASYSIMD_FLOAT32_C(  -197.17),
                         EASYSIMD_FLOAT32_C(   554.42), EASYSIMD_FLOAT32_C(   844.38), EASYSIMD_FLOAT32_C(  -817.51), EASYSIMD_FLOAT32_C(   338.74),
                         EASYSIMD_FLOAT32_C(   -70.99), EASYSIMD_FLOAT32_C(  -221.33), EASYSIMD_FLOAT32_C(    59.42), EASYSIMD_FLOAT32_C(   138.47)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -257.55), EASYSIMD_FLOAT32_C(   575.35), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   529.23), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   647.07),
                         EASYSIMD_FLOAT32_C(  -970.17), EASYSIMD_FLOAT32_C(   -59.71), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   187.82),
                         EASYSIMD_FLOAT32_C(   138.16), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   289.35), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(10432),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -855.26), EASYSIMD_FLOAT32_C(   444.16), EASYSIMD_FLOAT32_C(   962.50), EASYSIMD_FLOAT32_C(   987.86),
                         EASYSIMD_FLOAT32_C(  -410.31), EASYSIMD_FLOAT32_C(    36.70), EASYSIMD_FLOAT32_C(   874.49), EASYSIMD_FLOAT32_C(  -627.16),
                         EASYSIMD_FLOAT32_C(   911.91), EASYSIMD_FLOAT32_C(  -816.98), EASYSIMD_FLOAT32_C(  -164.10), EASYSIMD_FLOAT32_C(  -340.48),
                         EASYSIMD_FLOAT32_C(   -77.39), EASYSIMD_FLOAT32_C(   952.25), EASYSIMD_FLOAT32_C(   134.46), EASYSIMD_FLOAT32_C(   698.09)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -179.27), EASYSIMD_FLOAT32_C(   894.92), EASYSIMD_FLOAT32_C(  -553.39), EASYSIMD_FLOAT32_C(   676.19),
                         EASYSIMD_FLOAT32_C(  -747.28), EASYSIMD_FLOAT32_C(  -915.60), EASYSIMD_FLOAT32_C(  -132.34), EASYSIMD_FLOAT32_C(  -335.46),
                         EASYSIMD_FLOAT32_C(   243.51), EASYSIMD_FLOAT32_C(   766.95), EASYSIMD_FLOAT32_C(   899.58), EASYSIMD_FLOAT32_C(   478.33),
                         EASYSIMD_FLOAT32_C(   -35.25), EASYSIMD_FLOAT32_C(  -117.47), EASYSIMD_FLOAT32_C(   258.33), EASYSIMD_FLOAT32_C(  -248.63)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  1515.89), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   336.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   668.40), EASYSIMD_FLOAT32_C( -1583.93), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C( 9219),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   878.16), EASYSIMD_FLOAT32_C(  -299.57), EASYSIMD_FLOAT32_C(   829.01), EASYSIMD_FLOAT32_C(  -823.97),
                         EASYSIMD_FLOAT32_C(   313.21), EASYSIMD_FLOAT32_C(  -396.40), EASYSIMD_FLOAT32_C(   940.94), EASYSIMD_FLOAT32_C(  -281.84),
                         EASYSIMD_FLOAT32_C(   235.34), EASYSIMD_FLOAT32_C(   443.88), EASYSIMD_FLOAT32_C(  -185.89), EASYSIMD_FLOAT32_C(  -220.35),
                         EASYSIMD_FLOAT32_C(  -983.75), EASYSIMD_FLOAT32_C(  -348.00), EASYSIMD_FLOAT32_C(   167.63), EASYSIMD_FLOAT32_C(   489.46)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   439.60), EASYSIMD_FLOAT32_C(    39.73), EASYSIMD_FLOAT32_C(   948.24), EASYSIMD_FLOAT32_C(  -515.41),
                         EASYSIMD_FLOAT32_C(  -519.45), EASYSIMD_FLOAT32_C(   273.14), EASYSIMD_FLOAT32_C(   256.99), EASYSIMD_FLOAT32_C(    69.80),
                         EASYSIMD_FLOAT32_C(  -548.50), EASYSIMD_FLOAT32_C(  -730.33), EASYSIMD_FLOAT32_C(   337.76), EASYSIMD_FLOAT32_C(    90.27),
                         EASYSIMD_FLOAT32_C(  -665.67), EASYSIMD_FLOAT32_C(   930.26), EASYSIMD_FLOAT32_C(  -181.77), EASYSIMD_FLOAT32_C(   530.83)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -119.23), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -669.54), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   349.40), EASYSIMD_FLOAT32_C(   -41.37)) },
    { UINT16_C(60216),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -294.33), EASYSIMD_FLOAT32_C(   963.39), EASYSIMD_FLOAT32_C(  -504.91), EASYSIMD_FLOAT32_C(  -654.23),
                         EASYSIMD_FLOAT32_C(  -988.36), EASYSIMD_FLOAT32_C(   634.30), EASYSIMD_FLOAT32_C(  -857.50), EASYSIMD_FLOAT32_C(  -235.19),
                         EASYSIMD_FLOAT32_C(  -903.31), EASYSIMD_FLOAT32_C(  -183.01), EASYSIMD_FLOAT32_C(  -989.08), EASYSIMD_FLOAT32_C(  -684.38),
                         EASYSIMD_FLOAT32_C(  -369.22), EASYSIMD_FLOAT32_C(   764.60), EASYSIMD_FLOAT32_C(   215.22), EASYSIMD_FLOAT32_C(  -906.73)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -656.49), EASYSIMD_FLOAT32_C(  -795.39), EASYSIMD_FLOAT32_C(   220.41), EASYSIMD_FLOAT32_C(   680.39),
                         EASYSIMD_FLOAT32_C(  -673.42), EASYSIMD_FLOAT32_C(   859.78), EASYSIMD_FLOAT32_C(   306.17), EASYSIMD_FLOAT32_C(   632.76),
                         EASYSIMD_FLOAT32_C(  -662.91), EASYSIMD_FLOAT32_C(    31.45), EASYSIMD_FLOAT32_C(  -162.68), EASYSIMD_FLOAT32_C(   929.60),
                         EASYSIMD_FLOAT32_C(  -957.67), EASYSIMD_FLOAT32_C(   222.14), EASYSIMD_FLOAT32_C(   292.45), EASYSIMD_FLOAT32_C(   -99.46)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   362.16), EASYSIMD_FLOAT32_C(  1758.78), EASYSIMD_FLOAT32_C(  -725.32), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -314.94), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( -1163.67), EASYSIMD_FLOAT32_C(  -867.95),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -826.40), EASYSIMD_FLOAT32_C( -1613.98),
                         EASYSIMD_FLOAT32_C(   588.45), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C( 1065),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -160.16), EASYSIMD_FLOAT32_C(  -172.32), EASYSIMD_FLOAT32_C(  -146.34), EASYSIMD_FLOAT32_C(  -664.30),
                         EASYSIMD_FLOAT32_C(  -152.25), EASYSIMD_FLOAT32_C(   103.01), EASYSIMD_FLOAT32_C(  -445.68), EASYSIMD_FLOAT32_C(  -705.22),
                         EASYSIMD_FLOAT32_C(  -480.35), EASYSIMD_FLOAT32_C(  -454.79), EASYSIMD_FLOAT32_C(   524.00), EASYSIMD_FLOAT32_C(  -158.20),
                         EASYSIMD_FLOAT32_C(  -445.04), EASYSIMD_FLOAT32_C(  -960.28), EASYSIMD_FLOAT32_C(   167.13), EASYSIMD_FLOAT32_C(  -825.53)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -541.83), EASYSIMD_FLOAT32_C(  -457.69), EASYSIMD_FLOAT32_C(   312.80), EASYSIMD_FLOAT32_C(   -62.23),
                         EASYSIMD_FLOAT32_C(   416.18), EASYSIMD_FLOAT32_C(   853.40), EASYSIMD_FLOAT32_C(   -17.96), EASYSIMD_FLOAT32_C(   885.15),
                         EASYSIMD_FLOAT32_C(  -212.03), EASYSIMD_FLOAT32_C(  -855.73), EASYSIMD_FLOAT32_C(  -371.31), EASYSIMD_FLOAT32_C(  -695.44),
                         EASYSIMD_FLOAT32_C(  -895.68), EASYSIMD_FLOAT32_C(   538.84), EASYSIMD_FLOAT32_C(   882.30), EASYSIMD_FLOAT32_C(   585.87)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -750.39), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   895.31), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   450.64), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( -1411.40)) },
    { UINT16_C( 4987),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   955.74), EASYSIMD_FLOAT32_C(    48.94), EASYSIMD_FLOAT32_C(   560.80), EASYSIMD_FLOAT32_C(   626.25),
                         EASYSIMD_FLOAT32_C(   986.71), EASYSIMD_FLOAT32_C(   -13.30), EASYSIMD_FLOAT32_C(  -833.84), EASYSIMD_FLOAT32_C(   647.36),
                         EASYSIMD_FLOAT32_C(  -398.46), EASYSIMD_FLOAT32_C(  -852.77), EASYSIMD_FLOAT32_C(   195.24), EASYSIMD_FLOAT32_C(  -431.65),
                         EASYSIMD_FLOAT32_C(  -246.40), EASYSIMD_FLOAT32_C(  -123.66), EASYSIMD_FLOAT32_C(   302.57), EASYSIMD_FLOAT32_C(  -312.92)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   864.85), EASYSIMD_FLOAT32_C(  -886.94), EASYSIMD_FLOAT32_C(   289.25), EASYSIMD_FLOAT32_C(    74.52),
                         EASYSIMD_FLOAT32_C(  -653.98), EASYSIMD_FLOAT32_C(    43.30), EASYSIMD_FLOAT32_C(  -126.09), EASYSIMD_FLOAT32_C(  -155.50),
                         EASYSIMD_FLOAT32_C(  -396.73), EASYSIMD_FLOAT32_C(   -53.65), EASYSIMD_FLOAT32_C(  -516.81), EASYSIMD_FLOAT32_C(  -892.08),
                         EASYSIMD_FLOAT32_C(   202.83), EASYSIMD_FLOAT32_C(  -327.18), EASYSIMD_FLOAT32_C(   221.07), EASYSIMD_FLOAT32_C(  -891.88)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   551.73),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -707.75), EASYSIMD_FLOAT32_C(   802.86),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -799.12), EASYSIMD_FLOAT32_C(   712.05), EASYSIMD_FLOAT32_C(   460.43),
                         EASYSIMD_FLOAT32_C(  -449.23), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    81.50), EASYSIMD_FLOAT32_C(   578.96)) },
    { UINT16_C(56258),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -30.48), EASYSIMD_FLOAT32_C(   682.33), EASYSIMD_FLOAT32_C(  -678.43), EASYSIMD_FLOAT32_C(   640.73),
                         EASYSIMD_FLOAT32_C(  -214.39), EASYSIMD_FLOAT32_C(   913.47), EASYSIMD_FLOAT32_C(   802.27), EASYSIMD_FLOAT32_C(  -719.14),
                         EASYSIMD_FLOAT32_C(   839.92), EASYSIMD_FLOAT32_C(   326.41), EASYSIMD_FLOAT32_C(   231.12), EASYSIMD_FLOAT32_C(  -599.80),
                         EASYSIMD_FLOAT32_C(  -175.19), EASYSIMD_FLOAT32_C(  -889.93), EASYSIMD_FLOAT32_C(  -271.66), EASYSIMD_FLOAT32_C(  -767.93)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   521.85), EASYSIMD_FLOAT32_C(   990.67), EASYSIMD_FLOAT32_C(  -279.18), EASYSIMD_FLOAT32_C(   874.22),
                         EASYSIMD_FLOAT32_C(  -573.38), EASYSIMD_FLOAT32_C(   750.45), EASYSIMD_FLOAT32_C(   668.60), EASYSIMD_FLOAT32_C(  -415.36),
                         EASYSIMD_FLOAT32_C(  -224.84), EASYSIMD_FLOAT32_C(   162.63), EASYSIMD_FLOAT32_C(  -940.52), EASYSIMD_FLOAT32_C(  -654.73),
                         EASYSIMD_FLOAT32_C(  -780.19), EASYSIMD_FLOAT32_C(   466.85), EASYSIMD_FLOAT32_C(  -383.81), EASYSIMD_FLOAT32_C(   542.46)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -552.33), EASYSIMD_FLOAT32_C(  -308.34), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -233.49),
                         EASYSIMD_FLOAT32_C(   358.99), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   133.67), EASYSIMD_FLOAT32_C(  -303.78),
                         EASYSIMD_FLOAT32_C(  1064.76), EASYSIMD_FLOAT32_C(   163.78), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   112.15), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(18374),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   292.05), EASYSIMD_FLOAT32_C(  -553.40), EASYSIMD_FLOAT32_C(   143.99), EASYSIMD_FLOAT32_C(  -940.99),
                         EASYSIMD_FLOAT32_C(    81.28), EASYSIMD_FLOAT32_C(   184.98), EASYSIMD_FLOAT32_C(   662.04), EASYSIMD_FLOAT32_C(   951.27),
                         EASYSIMD_FLOAT32_C(   953.92), EASYSIMD_FLOAT32_C(  -384.26), EASYSIMD_FLOAT32_C(  -149.83), EASYSIMD_FLOAT32_C(   751.91),
                         EASYSIMD_FLOAT32_C(  -625.68), EASYSIMD_FLOAT32_C(    58.69), EASYSIMD_FLOAT32_C(   581.13), EASYSIMD_FLOAT32_C(   892.26)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   104.19), EASYSIMD_FLOAT32_C(   819.72), EASYSIMD_FLOAT32_C(  -437.31), EASYSIMD_FLOAT32_C(   380.61),
                         EASYSIMD_FLOAT32_C(   930.00), EASYSIMD_FLOAT32_C(  -224.08), EASYSIMD_FLOAT32_C(  -557.43), EASYSIMD_FLOAT32_C(  -295.43),
                         EASYSIMD_FLOAT32_C(  -271.48), EASYSIMD_FLOAT32_C(  -705.78), EASYSIMD_FLOAT32_C(  -796.97), EASYSIMD_FLOAT32_C(   -62.19),
                         EASYSIMD_FLOAT32_C(  -247.25), EASYSIMD_FLOAT32_C(   225.36), EASYSIMD_FLOAT32_C(   312.68), EASYSIMD_FLOAT32_C(  -185.21)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( -1373.12), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   409.06), EASYSIMD_FLOAT32_C(  1219.47), EASYSIMD_FLOAT32_C(  1246.70),
                         EASYSIMD_FLOAT32_C(  1225.40), EASYSIMD_FLOAT32_C(   321.52), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -166.67), EASYSIMD_FLOAT32_C(   268.45), EASYSIMD_FLOAT32_C(     0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 b = test_vec[i].b;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_sub_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_sub_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_sub_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d b;
    easysimd__m512d r;
  } test_vec[8] = {
    { UINT8_C( 63),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -415.75), EASYSIMD_FLOAT64_C(  784.67),
                         EASYSIMD_FLOAT64_C( -496.30), EASYSIMD_FLOAT64_C(  526.56),
                         EASYSIMD_FLOAT64_C(   67.17), EASYSIMD_FLOAT64_C( -881.21),
                         EASYSIMD_FLOAT64_C(  348.77), EASYSIMD_FLOAT64_C(  537.04)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  554.42), EASYSIMD_FLOAT64_C(  844.38),
                         EASYSIMD_FLOAT64_C( -817.51), EASYSIMD_FLOAT64_C(  338.74),
                         EASYSIMD_FLOAT64_C(  -70.99), EASYSIMD_FLOAT64_C( -221.33),
                         EASYSIMD_FLOAT64_C(   59.42), EASYSIMD_FLOAT64_C(  138.47)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  321.21), EASYSIMD_FLOAT64_C(  187.82),
                         EASYSIMD_FLOAT64_C(  138.16), EASYSIMD_FLOAT64_C( -659.88),
                         EASYSIMD_FLOAT64_C(  289.35), EASYSIMD_FLOAT64_C(  398.57)) },
    { UINT8_C(204),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  392.86), EASYSIMD_FLOAT64_C(   15.96),
                         EASYSIMD_FLOAT64_C( -681.24), EASYSIMD_FLOAT64_C(  759.61),
                         EASYSIMD_FLOAT64_C( -507.08), EASYSIMD_FLOAT64_C( -150.50),
                         EASYSIMD_FLOAT64_C(  409.54), EASYSIMD_FLOAT64_C( -197.17)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -661.30), EASYSIMD_FLOAT64_C( -524.33),
                         EASYSIMD_FLOAT64_C( -241.59), EASYSIMD_FLOAT64_C( -105.89),
                         EASYSIMD_FLOAT64_C( -289.61), EASYSIMD_FLOAT64_C( -891.58),
                         EASYSIMD_FLOAT64_C(  378.73), EASYSIMD_FLOAT64_C(  -71.99)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( 1054.16), EASYSIMD_FLOAT64_C(  540.29),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C( -217.47), EASYSIMD_FLOAT64_C(  741.08),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(198),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  911.91), EASYSIMD_FLOAT64_C( -816.98),
                         EASYSIMD_FLOAT64_C( -164.10), EASYSIMD_FLOAT64_C( -340.48),
                         EASYSIMD_FLOAT64_C(  -77.39), EASYSIMD_FLOAT64_C(  952.25),
                         EASYSIMD_FLOAT64_C(  134.46), EASYSIMD_FLOAT64_C(  698.09)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  243.51), EASYSIMD_FLOAT64_C(  766.95),
                         EASYSIMD_FLOAT64_C(  899.58), EASYSIMD_FLOAT64_C(  478.33),
                         EASYSIMD_FLOAT64_C(  -35.25), EASYSIMD_FLOAT64_C( -117.47),
                         EASYSIMD_FLOAT64_C(  258.33), EASYSIMD_FLOAT64_C( -248.63)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  668.40), EASYSIMD_FLOAT64_C(-1583.93),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( 1069.72),
                         EASYSIMD_FLOAT64_C( -123.87), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(149),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -179.27), EASYSIMD_FLOAT64_C(  894.92),
                         EASYSIMD_FLOAT64_C( -553.39), EASYSIMD_FLOAT64_C(  676.19),
                         EASYSIMD_FLOAT64_C( -747.28), EASYSIMD_FLOAT64_C( -915.60),
                         EASYSIMD_FLOAT64_C( -132.34), EASYSIMD_FLOAT64_C( -335.46)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -395.05), EASYSIMD_FLOAT64_C( -855.26),
                         EASYSIMD_FLOAT64_C(  444.16), EASYSIMD_FLOAT64_C(  962.50),
                         EASYSIMD_FLOAT64_C(  987.86), EASYSIMD_FLOAT64_C( -410.31),
                         EASYSIMD_FLOAT64_C(   36.70), EASYSIMD_FLOAT64_C(  874.49)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  215.78), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -286.31),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -505.29),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(-1209.95)) },
    { UINT8_C(196),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  235.34), EASYSIMD_FLOAT64_C(  443.88),
                         EASYSIMD_FLOAT64_C( -185.89), EASYSIMD_FLOAT64_C( -220.35),
                         EASYSIMD_FLOAT64_C( -983.75), EASYSIMD_FLOAT64_C( -348.00),
                         EASYSIMD_FLOAT64_C(  167.63), EASYSIMD_FLOAT64_C(  489.46)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -548.50), EASYSIMD_FLOAT64_C( -730.33),
                         EASYSIMD_FLOAT64_C(  337.76), EASYSIMD_FLOAT64_C(   90.27),
                         EASYSIMD_FLOAT64_C( -665.67), EASYSIMD_FLOAT64_C(  930.26),
                         EASYSIMD_FLOAT64_C( -181.77), EASYSIMD_FLOAT64_C(  530.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  783.84), EASYSIMD_FLOAT64_C( 1174.21),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(-1278.26),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(230),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  439.60), EASYSIMD_FLOAT64_C(   39.73),
                         EASYSIMD_FLOAT64_C(  948.24), EASYSIMD_FLOAT64_C( -515.41),
                         EASYSIMD_FLOAT64_C( -519.45), EASYSIMD_FLOAT64_C(  273.14),
                         EASYSIMD_FLOAT64_C(  256.99), EASYSIMD_FLOAT64_C(   69.80)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -138.55), EASYSIMD_FLOAT64_C(  878.16),
                         EASYSIMD_FLOAT64_C( -299.57), EASYSIMD_FLOAT64_C(  829.01),
                         EASYSIMD_FLOAT64_C( -823.97), EASYSIMD_FLOAT64_C(  313.21),
                         EASYSIMD_FLOAT64_C( -396.40), EASYSIMD_FLOAT64_C(  940.94)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  578.15), EASYSIMD_FLOAT64_C( -838.43),
                         EASYSIMD_FLOAT64_C( 1247.81), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  -40.07),
                         EASYSIMD_FLOAT64_C(  653.39), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C( 58),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -903.31), EASYSIMD_FLOAT64_C( -183.01),
                         EASYSIMD_FLOAT64_C( -989.08), EASYSIMD_FLOAT64_C( -684.38),
                         EASYSIMD_FLOAT64_C( -369.22), EASYSIMD_FLOAT64_C(  764.60),
                         EASYSIMD_FLOAT64_C(  215.22), EASYSIMD_FLOAT64_C( -906.73)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -662.91), EASYSIMD_FLOAT64_C(   31.45),
                         EASYSIMD_FLOAT64_C( -162.68), EASYSIMD_FLOAT64_C(  929.60),
                         EASYSIMD_FLOAT64_C( -957.67), EASYSIMD_FLOAT64_C(  222.14),
                         EASYSIMD_FLOAT64_C(  292.45), EASYSIMD_FLOAT64_C(  -99.46)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C( -826.40), EASYSIMD_FLOAT64_C(-1613.98),
                         EASYSIMD_FLOAT64_C(  588.45), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  -77.23), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(175),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -656.49), EASYSIMD_FLOAT64_C( -795.39),
                         EASYSIMD_FLOAT64_C(  220.41), EASYSIMD_FLOAT64_C(  680.39),
                         EASYSIMD_FLOAT64_C( -673.42), EASYSIMD_FLOAT64_C(  859.78),
                         EASYSIMD_FLOAT64_C(  306.17), EASYSIMD_FLOAT64_C(  632.76)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  367.28), EASYSIMD_FLOAT64_C( -294.33),
                         EASYSIMD_FLOAT64_C(  963.39), EASYSIMD_FLOAT64_C( -504.91),
                         EASYSIMD_FLOAT64_C( -654.23), EASYSIMD_FLOAT64_C( -988.36),
                         EASYSIMD_FLOAT64_C(  634.30), EASYSIMD_FLOAT64_C( -857.50)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(-1023.77), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C( -742.98), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  -19.19), EASYSIMD_FLOAT64_C( 1848.14),
                         EASYSIMD_FLOAT64_C( -328.13), EASYSIMD_FLOAT64_C( 1490.26)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d r = easysimd_mm512_maskz_sub_pd(test_vec[i].k, test_vec[i].a, test_vec[i].b);
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_subr_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1967746070), -INT32_C(   624328481),  INT32_C(  1867262811),  INT32_C(  1217365141), -INT32_C(   524695179), -INT32_C(  2063935476), -INT32_C(   946926535), -INT32_C(  2024613476),
         INT32_C(  1259407468), -INT32_C(   467281272), -INT32_C(  1689030139),  INT32_C(  1810097142), -INT32_C(  1202938708), -INT32_C(  1388558732), -INT32_C(   294270126),  INT32_C(   276154276) },
      {  INT32_C(  1465616079),  INT32_C(  1715175777), -INT32_C(   385773581),  INT32_C(   508945778), -INT32_C(   153706111),  INT32_C(   966988519), -INT32_C(  2111366946), -INT32_C(  1366057505),
        -INT32_C(  2079920605),  INT32_C(  1659584879),  INT32_C(  1112337616),  INT32_C(  1398841809),  INT32_C(   692663874),  INT32_C(   660794441), -INT32_C(   458651387),  INT32_C(  1251163174) },
      { -INT32_C(   861605147), -INT32_C(  1955463038),  INT32_C(  2041930904), -INT32_C(   708419363),  INT32_C(   370989068), -INT32_C(  1264043301), -INT32_C(  1164440411),  INT32_C(   658555971),
         INT32_C(   955639223),  INT32_C(  2126866151), -INT32_C(  1493599541), -INT32_C(   411255333),  INT32_C(  1895602582),  INT32_C(  2049353173), -INT32_C(   164381261),  INT32_C(   975008898) } },
    { { -INT32_C(  1697736405), -INT32_C(  1409435174),  INT32_C(  2012039590),  INT32_C(   751455978), -INT32_C(   850062460),  INT32_C(    99923712),  INT32_C(  1743363648),  INT32_C(    95517914),
        -INT32_C(   257917163), -INT32_C(   543449799), -INT32_C(   799635226),  INT32_C(  1543250390),  INT32_C(   875057460),  INT32_C(  1211702536), -INT32_C(  1783684165), -INT32_C(  1248108385) },
      {  INT32_C(   430259168), -INT32_C(  1091026728), -INT32_C(  1618063672), -INT32_C(  1527084433), -INT32_C(   472374565), -INT32_C(    97840577), -INT32_C(   728704203),  INT32_C(   461974331),
         INT32_C(  1043607142),  INT32_C(   939273326), -INT32_C(   355038597), -INT32_C(   275853292),  INT32_C(   852649970), -INT32_C(  1372783239),  INT32_C(   344112344),  INT32_C(  1294929127) },
      {  INT32_C(  2127995573),  INT32_C(   318408446),  INT32_C(   664864034),  INT32_C(  2016426885),  INT32_C(   377687895), -INT32_C(   197764289),  INT32_C(  1822899445),  INT32_C(   366456417),
         INT32_C(  1301524305),  INT32_C(  1482723125),  INT32_C(   444596629), -INT32_C(  1819103682), -INT32_C(    22407490),  INT32_C(  1710481521),  INT32_C(  2127796509), -INT32_C(  1751929784) } },
    { { -INT32_C(  1450482630),  INT32_C(   199264144),  INT32_C(   636925457),  INT32_C(  2014610566),  INT32_C(  1688921835), -INT32_C(  1139615772),  INT32_C(  2077267347), -INT32_C(   607649631),
        -INT32_C(   192654236), -INT32_C(   318741285), -INT32_C(  1609370086),  INT32_C(  1712924538), -INT32_C(   238370035),  INT32_C(   783211930),  INT32_C(   329875058), -INT32_C(   487689858) },
      { -INT32_C(  1579715643), -INT32_C(   225519657),  INT32_C(  1200791757), -INT32_C(   710038585),  INT32_C(   164001903), -INT32_C(   952667051),  INT32_C(  1910169843),  INT32_C(   391432274),
         INT32_C(   330836795), -INT32_C(   821737982), -INT32_C(  1374251033), -INT32_C(  1299987389), -INT32_C(  1849996740), -INT32_C(  1319570498),  INT32_C(   623063763),  INT32_C(   909998074) },
      { -INT32_C(   129233013), -INT32_C(   424783801),  INT32_C(   563866300),  INT32_C(  1570318145), -INT32_C(  1524919932),  INT32_C(   186948721), -INT32_C(   167097504),  INT32_C(   999081905),
         INT32_C(   523491031), -INT32_C(   502996697),  INT32_C(   235119053),  INT32_C(  1282055369), -INT32_C(  1611626705), -INT32_C(  2102782428),  INT32_C(   293188705),  INT32_C(  1397687932) } },
    { { -INT32_C(  1538656862),  INT32_C(   594759228),  INT32_C(   684820965), -INT32_C(  1982180019),  INT32_C(  1562023583),  INT32_C(  1544516233), -INT32_C(  1618857308),  INT32_C(  1272299433),
        -INT32_C(   252764492),  INT32_C(  1360224876),  INT32_C(   964355563), -INT32_C(   641575878),  INT32_C(  1932975594), -INT32_C(   187677361),  INT32_C(   546525815), -INT32_C(   982816751) },
      { -INT32_C(   222930298), -INT32_C(  1471952452), -INT32_C(   387859026), -INT32_C(    54418670), -INT32_C(   797902720), -INT32_C(  1245429699), -INT32_C(  1546299502),  INT32_C(  1164460479),
         INT32_C(  1480007323), -INT32_C(  1778353432),  INT32_C(  1249894712),  INT32_C(    88490116),  INT32_C(  1993717304), -INT32_C(  2010408458), -INT32_C(  1356136208), -INT32_C(   571173823) },
      {  INT32_C(  1315726564), -INT32_C(  2066711680), -INT32_C(  1072679991),  INT32_C(  1927761349),  INT32_C(  1935040993),  INT32_C(  1505021364),  INT32_C(    72557806), -INT32_C(   107838954),
         INT32_C(  1732771815),  INT32_C(  1156388988),  INT32_C(   285539149),  INT32_C(   730065994),  INT32_C(    60741710), -INT32_C(  1822731097), -INT32_C(  1902662023),  INT32_C(   411642928) } },
    { { -INT32_C(  1707791438), -INT32_C(   583977563), -INT32_C(  1708675050),  INT32_C(   698314480),  INT32_C(   463434788), -INT32_C(    39597555),  INT32_C(   212651978),  INT32_C(   367632483),
         INT32_C(  1907367627),  INT32_C(  1766777171), -INT32_C(  2130479471),  INT32_C(   145400804),  INT32_C(   623069463), -INT32_C(   568146157), -INT32_C(   102051946),  INT32_C(   990827375) },
      {  INT32_C(  1152171761),  INT32_C(   816708255),  INT32_C(  1420996720),  INT32_C(  1801280595), -INT32_C(  1181712218), -INT32_C(   577260985), -INT32_C(   237600383),  INT32_C(  1160570196),
         INT32_C(  1133107363),  INT32_C(  1131624146),  INT32_C(   982984167),  INT32_C(   681964674), -INT32_C(  1126091403),  INT32_C(  1771731432),  INT32_C(  1331327226), -INT32_C(   107706795) },
      { -INT32_C(  1435004097),  INT32_C(  1400685818), -INT32_C(  1165295526),  INT32_C(  1102966115), -INT32_C(  1645147006), -INT32_C(   537663430), -INT32_C(   450252361),  INT32_C(   792937713),
        -INT32_C(   774260264), -INT32_C(   635153025), -INT32_C(  1181503658),  INT32_C(   536563870), -INT32_C(  1749160866), -INT32_C(  1955089707),  INT32_C(  1433379172), -INT32_C(  1098534170) } },
    { {  INT32_C(   826023518),  INT32_C(   997502804),  INT32_C(  1467354069),  INT32_C(  1971264256),  INT32_C(   959537233), -INT32_C(   727528487), -INT32_C(  1859912388), -INT32_C(   494225533),
         INT32_C(   705939157),  INT32_C(  1264945014), -INT32_C(  1834820718),  INT32_C(  1208426999),  INT32_C(  1535195521),  INT32_C(  1076830980), -INT32_C(  1529720288), -INT32_C(   544842743) },
      { -INT32_C(  1727424221), -INT32_C(  1293652448),  INT32_C(  1095075402),  INT32_C(   680086695), -INT32_C(  1971123579),  INT32_C(  1321906733),  INT32_C(   250780676),  INT32_C(   485325049),
         INT32_C(   834008593), -INT32_C(  1360815772), -INT32_C(   957405153), -INT32_C(    85034891), -INT32_C(  1333497214),  INT32_C(   704532260), -INT32_C(   466095893),  INT32_C(  2030052456) },
      {  INT32_C(  1741519557),  INT32_C(  2003812044), -INT32_C(   372278667), -INT32_C(  1291177561),  INT32_C(  1364306484),  INT32_C(  2049435220),  INT32_C(  2110693064),  INT32_C(   979550582),
         INT32_C(   128069436),  INT32_C(  1669206510),  INT32_C(   877415565), -INT32_C(  1293461890),  INT32_C(  1426274561), -INT32_C(   372298720),  INT32_C(  1063624395), -INT32_C(  1720072097) } },
    { {  INT32_C(  2125116698),  INT32_C(  1831701838),  INT32_C(   724769974),  INT32_C(   388309653), -INT32_C(  1194874220), -INT32_C(   454965767),  INT32_C(   499718325),  INT32_C(  1452722492),
        -INT32_C(   841662337), -INT32_C(  2076507442), -INT32_C(  1280348642),  INT32_C(   617338000),  INT32_C(  2011009662),  INT32_C(   224181848),  INT32_C(   321594839),  INT32_C(  1835713006) },
      { -INT32_C(   801423614),  INT32_C(  1599370817),  INT32_C(  1964180452),  INT32_C(  1452924631), -INT32_C(   926058896),  INT32_C(   215361845),  INT32_C(  1025442126), -INT32_C(   995456574),
         INT32_C(   160753096),  INT32_C(  1080682587), -INT32_C(  1011516181), -INT32_C(   904311206), -INT32_C(    90970427),  INT32_C(  1577478416),  INT32_C(   748365162),  INT32_C(  2012300975) },
      {  INT32_C(  1368426984), -INT32_C(   232331021),  INT32_C(  1239410478),  INT32_C(  1064614978),  INT32_C(   268815324),  INT32_C(   670327612),  INT32_C(   525723801),  INT32_C(  1846788230),
         INT32_C(  1002415433), -INT32_C(  1137777267),  INT32_C(   268832461), -INT32_C(  1521649206), -INT32_C(  2101980089),  INT32_C(  1353296568),  INT32_C(   426770323),  INT32_C(   176587969) } },
    { { -INT32_C(  2021554901),  INT32_C(  1506273902), -INT32_C(  1071874970), -INT32_C(  1886767670),  INT32_C(   747183388), -INT32_C(   259354746),  INT32_C(  1679632053), -INT32_C(  1747251604),
         INT32_C(    18766995), -INT32_C(  1403263674),  INT32_C(   745305953), -INT32_C(   910428243), -INT32_C(  1678424812), -INT32_C(  1987346476),  INT32_C(   300787877),  INT32_C(  1252641206) },
      {  INT32_C(  1816905509),  INT32_C(   236496557), -INT32_C(   885357282), -INT32_C(  1869285764),  INT32_C(   254511419), -INT32_C(  1365657848),  INT32_C(   381650527),  INT32_C(  1969252431),
        -INT32_C(   572413136),  INT32_C(  1894512978), -INT32_C(    96786818),  INT32_C(  1468780316),  INT32_C(  1617344088), -INT32_C(   854655123), -INT32_C(   706490746),  INT32_C(  1716142902) },
      { -INT32_C(   456506886), -INT32_C(  1269777345),  INT32_C(   186517688),  INT32_C(    17481906), -INT32_C(   492671969), -INT32_C(  1106303102), -INT32_C(  1297981526), -INT32_C(   578463261),
        -INT32_C(   591180131), -INT32_C(   997190644), -INT32_C(   842092771), -INT32_C(  1915758737), -INT32_C(   999198396),  INT32_C(  1132691353), -INT32_C(  1007278623),  INT32_C(   463501696) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_subr_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_subr_epi32");
    easysimd_assert_m512i_i32(r, ==, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }
  
  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_subr_epi32(a, b);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_subr_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   956.48), EASYSIMD_FLOAT32_C(   743.49), EASYSIMD_FLOAT32_C(   126.40), EASYSIMD_FLOAT32_C(   607.10),
        EASYSIMD_FLOAT32_C(    27.30), EASYSIMD_FLOAT32_C(   277.75), EASYSIMD_FLOAT32_C(  -143.03), EASYSIMD_FLOAT32_C(  -845.80),
        EASYSIMD_FLOAT32_C(   121.11), EASYSIMD_FLOAT32_C(  -822.70), EASYSIMD_FLOAT32_C(   633.30), EASYSIMD_FLOAT32_C(   145.54),
        EASYSIMD_FLOAT32_C(   667.16), EASYSIMD_FLOAT32_C(  -850.91), EASYSIMD_FLOAT32_C(  -813.90), EASYSIMD_FLOAT32_C(  -734.39) },
      { EASYSIMD_FLOAT32_C(    32.74), EASYSIMD_FLOAT32_C(   912.36), EASYSIMD_FLOAT32_C(  -129.97), EASYSIMD_FLOAT32_C(  -106.51),
        EASYSIMD_FLOAT32_C(  -879.92), EASYSIMD_FLOAT32_C(   572.29), EASYSIMD_FLOAT32_C(   570.01), EASYSIMD_FLOAT32_C(  -727.35),
        EASYSIMD_FLOAT32_C(   198.49), EASYSIMD_FLOAT32_C(  -856.68), EASYSIMD_FLOAT32_C(  -646.47), EASYSIMD_FLOAT32_C(  -485.26),
        EASYSIMD_FLOAT32_C(   445.94), EASYSIMD_FLOAT32_C(  -781.63), EASYSIMD_FLOAT32_C(  -545.15), EASYSIMD_FLOAT32_C(   402.42) },
      { EASYSIMD_FLOAT32_C(  -923.74), EASYSIMD_FLOAT32_C(   168.87), EASYSIMD_FLOAT32_C(  -256.37), EASYSIMD_FLOAT32_C(  -713.61),
        EASYSIMD_FLOAT32_C(  -907.22), EASYSIMD_FLOAT32_C(   294.54), EASYSIMD_FLOAT32_C(   713.04), EASYSIMD_FLOAT32_C(   118.45),
        EASYSIMD_FLOAT32_C(    77.38), EASYSIMD_FLOAT32_C(   -33.98), EASYSIMD_FLOAT32_C( -1279.77), EASYSIMD_FLOAT32_C(  -630.80),
        EASYSIMD_FLOAT32_C(  -221.22), EASYSIMD_FLOAT32_C(    69.28), EASYSIMD_FLOAT32_C(   268.75), EASYSIMD_FLOAT32_C(  1136.81) } },
    { { EASYSIMD_FLOAT32_C(   961.87), EASYSIMD_FLOAT32_C(   581.25), EASYSIMD_FLOAT32_C(     9.52), EASYSIMD_FLOAT32_C(   -10.83),
        EASYSIMD_FLOAT32_C(  -141.00), EASYSIMD_FLOAT32_C(   866.49), EASYSIMD_FLOAT32_C(   143.36), EASYSIMD_FLOAT32_C(   980.11),
        EASYSIMD_FLOAT32_C(  -956.21), EASYSIMD_FLOAT32_C(  -223.34), EASYSIMD_FLOAT32_C(   125.65), EASYSIMD_FLOAT32_C(   710.95),
        EASYSIMD_FLOAT32_C(   -74.25), EASYSIMD_FLOAT32_C(   311.75), EASYSIMD_FLOAT32_C(   976.56), EASYSIMD_FLOAT32_C(   958.48) },
      { EASYSIMD_FLOAT32_C(   224.11), EASYSIMD_FLOAT32_C(  -153.41), EASYSIMD_FLOAT32_C(  -148.03), EASYSIMD_FLOAT32_C(   344.19),
        EASYSIMD_FLOAT32_C(  -581.12), EASYSIMD_FLOAT32_C(  -578.01), EASYSIMD_FLOAT32_C(   616.84), EASYSIMD_FLOAT32_C(   617.37),
        EASYSIMD_FLOAT32_C(  -434.70), EASYSIMD_FLOAT32_C(   970.37), EASYSIMD_FLOAT32_C(  -867.90), EASYSIMD_FLOAT32_C(  -988.76),
        EASYSIMD_FLOAT32_C(  -811.26), EASYSIMD_FLOAT32_C(  -413.05), EASYSIMD_FLOAT32_C(   413.66), EASYSIMD_FLOAT32_C(  -849.39) },
      { EASYSIMD_FLOAT32_C(  -737.76), EASYSIMD_FLOAT32_C(  -734.66), EASYSIMD_FLOAT32_C(  -157.55), EASYSIMD_FLOAT32_C(   355.02),
        EASYSIMD_FLOAT32_C(  -440.12), EASYSIMD_FLOAT32_C( -1444.50), EASYSIMD_FLOAT32_C(   473.48), EASYSIMD_FLOAT32_C(  -362.74),
        EASYSIMD_FLOAT32_C(   521.51), EASYSIMD_FLOAT32_C(  1193.71), EASYSIMD_FLOAT32_C(  -993.55), EASYSIMD_FLOAT32_C( -1699.71),
        EASYSIMD_FLOAT32_C(  -737.01), EASYSIMD_FLOAT32_C(  -724.80), EASYSIMD_FLOAT32_C(  -562.90), EASYSIMD_FLOAT32_C( -1807.87) } },
    { { EASYSIMD_FLOAT32_C(  -831.79), EASYSIMD_FLOAT32_C(  -576.82), EASYSIMD_FLOAT32_C(   139.77), EASYSIMD_FLOAT32_C(    27.21),
        EASYSIMD_FLOAT32_C(  -710.33), EASYSIMD_FLOAT32_C(  -716.86), EASYSIMD_FLOAT32_C(     7.32), EASYSIMD_FLOAT32_C(  -666.54),
        EASYSIMD_FLOAT32_C(    59.79), EASYSIMD_FLOAT32_C(  -867.03), EASYSIMD_FLOAT32_C(  -955.59), EASYSIMD_FLOAT32_C(   985.54),
        EASYSIMD_FLOAT32_C(   444.72), EASYSIMD_FLOAT32_C(  -979.03), EASYSIMD_FLOAT32_C(   944.02), EASYSIMD_FLOAT32_C(  -331.17) },
      { EASYSIMD_FLOAT32_C(  -132.44), EASYSIMD_FLOAT32_C(  -204.01), EASYSIMD_FLOAT32_C(  -986.98), EASYSIMD_FLOAT32_C(   286.44),
        EASYSIMD_FLOAT32_C(   217.98), EASYSIMD_FLOAT32_C(   629.86), EASYSIMD_FLOAT32_C(   -96.20), EASYSIMD_FLOAT32_C(   783.28),
        EASYSIMD_FLOAT32_C(   600.24), EASYSIMD_FLOAT32_C(    35.91), EASYSIMD_FLOAT32_C(   794.53), EASYSIMD_FLOAT32_C(   788.98),
        EASYSIMD_FLOAT32_C(   622.86), EASYSIMD_FLOAT32_C(   208.19), EASYSIMD_FLOAT32_C(   939.59), EASYSIMD_FLOAT32_C(   791.07) },
      { EASYSIMD_FLOAT32_C(   699.35), EASYSIMD_FLOAT32_C(   372.81), EASYSIMD_FLOAT32_C( -1126.75), EASYSIMD_FLOAT32_C(   259.23),
        EASYSIMD_FLOAT32_C(   928.31), EASYSIMD_FLOAT32_C(  1346.72), EASYSIMD_FLOAT32_C(  -103.52), EASYSIMD_FLOAT32_C(  1449.82),
        EASYSIMD_FLOAT32_C(   540.45), EASYSIMD_FLOAT32_C(   902.94), EASYSIMD_FLOAT32_C(  1750.12), EASYSIMD_FLOAT32_C(  -196.56),
        EASYSIMD_FLOAT32_C(   178.14), EASYSIMD_FLOAT32_C(  1187.22), EASYSIMD_FLOAT32_C(    -4.43), EASYSIMD_FLOAT32_C(  1122.24) } },
    { { EASYSIMD_FLOAT32_C(   631.37), EASYSIMD_FLOAT32_C(    79.36), EASYSIMD_FLOAT32_C(  -181.72), EASYSIMD_FLOAT32_C(   921.04),
        EASYSIMD_FLOAT32_C(   362.50), EASYSIMD_FLOAT32_C(   825.60), EASYSIMD_FLOAT32_C(  -745.50), EASYSIMD_FLOAT32_C(  -577.71),
        EASYSIMD_FLOAT32_C(   958.57), EASYSIMD_FLOAT32_C(  -701.10), EASYSIMD_FLOAT32_C(  -592.17), EASYSIMD_FLOAT32_C(   403.28),
        EASYSIMD_FLOAT32_C(  -680.13), EASYSIMD_FLOAT32_C(  -648.15), EASYSIMD_FLOAT32_C(  -927.89), EASYSIMD_FLOAT32_C(   187.43) },
      { EASYSIMD_FLOAT32_C(   147.85), EASYSIMD_FLOAT32_C(  -914.87), EASYSIMD_FLOAT32_C(  -526.13), EASYSIMD_FLOAT32_C(  -634.17),
        EASYSIMD_FLOAT32_C(   715.00), EASYSIMD_FLOAT32_C(   377.67), EASYSIMD_FLOAT32_C(  -850.89), EASYSIMD_FLOAT32_C(   315.23),
        EASYSIMD_FLOAT32_C(  -586.42), EASYSIMD_FLOAT32_C(   943.64), EASYSIMD_FLOAT32_C(   104.21), EASYSIMD_FLOAT32_C(  -963.56),
        EASYSIMD_FLOAT32_C(   151.83), EASYSIMD_FLOAT32_C(    43.80), EASYSIMD_FLOAT32_C(   827.52), EASYSIMD_FLOAT32_C(  -216.80) },
      { EASYSIMD_FLOAT32_C(  -483.52), EASYSIMD_FLOAT32_C(  -994.23), EASYSIMD_FLOAT32_C(  -344.41), EASYSIMD_FLOAT32_C( -1555.21),
        EASYSIMD_FLOAT32_C(   352.50), EASYSIMD_FLOAT32_C(  -447.93), EASYSIMD_FLOAT32_C(  -105.39), EASYSIMD_FLOAT32_C(   892.94),
        EASYSIMD_FLOAT32_C( -1544.99), EASYSIMD_FLOAT32_C(  1644.74), EASYSIMD_FLOAT32_C(   696.38), EASYSIMD_FLOAT32_C( -1366.84),
        EASYSIMD_FLOAT32_C(   831.96), EASYSIMD_FLOAT32_C(   691.95), EASYSIMD_FLOAT32_C(  1755.41), EASYSIMD_FLOAT32_C(  -404.23) } },
    { { EASYSIMD_FLOAT32_C(  -876.84), EASYSIMD_FLOAT32_C(  -354.21), EASYSIMD_FLOAT32_C(  -295.77), EASYSIMD_FLOAT32_C(   485.66),
        EASYSIMD_FLOAT32_C(  -528.61), EASYSIMD_FLOAT32_C(   -41.27), EASYSIMD_FLOAT32_C(   907.95), EASYSIMD_FLOAT32_C(  -570.04),
        EASYSIMD_FLOAT32_C(   257.63), EASYSIMD_FLOAT32_C(  -684.22), EASYSIMD_FLOAT32_C(   833.24), EASYSIMD_FLOAT32_C(   577.51),
        EASYSIMD_FLOAT32_C(  -332.36), EASYSIMD_FLOAT32_C(   905.35), EASYSIMD_FLOAT32_C(  -235.06), EASYSIMD_FLOAT32_C(   815.49) },
      { EASYSIMD_FLOAT32_C(   990.48), EASYSIMD_FLOAT32_C(   238.81), EASYSIMD_FLOAT32_C(  -818.68), EASYSIMD_FLOAT32_C(   705.48),
        EASYSIMD_FLOAT32_C(  -383.52), EASYSIMD_FLOAT32_C(  -669.57), EASYSIMD_FLOAT32_C(    20.71), EASYSIMD_FLOAT32_C(    30.06),
        EASYSIMD_FLOAT32_C(  -725.93), EASYSIMD_FLOAT32_C(  -875.08), EASYSIMD_FLOAT32_C(    66.50), EASYSIMD_FLOAT32_C(   425.90),
        EASYSIMD_FLOAT32_C(   168.72), EASYSIMD_FLOAT32_C(  -105.98), EASYSIMD_FLOAT32_C(  -790.90), EASYSIMD_FLOAT32_C(   291.88) },
      { EASYSIMD_FLOAT32_C(  1867.32), EASYSIMD_FLOAT32_C(   593.02), EASYSIMD_FLOAT32_C(  -522.91), EASYSIMD_FLOAT32_C(   219.82),
        EASYSIMD_FLOAT32_C(   145.09), EASYSIMD_FLOAT32_C(  -628.30), EASYSIMD_FLOAT32_C(  -887.24), EASYSIMD_FLOAT32_C(   600.10),
        EASYSIMD_FLOAT32_C(  -983.56), EASYSIMD_FLOAT32_C(  -190.86), EASYSIMD_FLOAT32_C(  -766.74), EASYSIMD_FLOAT32_C(  -151.61),
        EASYSIMD_FLOAT32_C(   501.08), EASYSIMD_FLOAT32_C( -1011.33), EASYSIMD_FLOAT32_C(  -555.84), EASYSIMD_FLOAT32_C(  -523.61) } },
    { { EASYSIMD_FLOAT32_C(   539.81), EASYSIMD_FLOAT32_C(   -86.67), EASYSIMD_FLOAT32_C(  -222.46), EASYSIMD_FLOAT32_C(  -988.79),
        EASYSIMD_FLOAT32_C(   872.06), EASYSIMD_FLOAT32_C(  -314.52), EASYSIMD_FLOAT32_C(  -558.84), EASYSIMD_FLOAT32_C(   129.69),
        EASYSIMD_FLOAT32_C(     1.27), EASYSIMD_FLOAT32_C(  -725.60), EASYSIMD_FLOAT32_C(  -292.81), EASYSIMD_FLOAT32_C(   668.90),
        EASYSIMD_FLOAT32_C(  -820.25), EASYSIMD_FLOAT32_C(   472.13), EASYSIMD_FLOAT32_C(   484.39), EASYSIMD_FLOAT32_C(  -829.77) },
      { EASYSIMD_FLOAT32_C(  -289.06), EASYSIMD_FLOAT32_C(   665.70), EASYSIMD_FLOAT32_C(   875.71), EASYSIMD_FLOAT32_C(   327.41),
        EASYSIMD_FLOAT32_C(   996.13), EASYSIMD_FLOAT32_C(  -103.58), EASYSIMD_FLOAT32_C(  -642.53), EASYSIMD_FLOAT32_C(  -729.79),
        EASYSIMD_FLOAT32_C(    21.34), EASYSIMD_FLOAT32_C(   423.98), EASYSIMD_FLOAT32_C(   696.11), EASYSIMD_FLOAT32_C(  -809.94),
        EASYSIMD_FLOAT32_C(  -682.00), EASYSIMD_FLOAT32_C(   905.20), EASYSIMD_FLOAT32_C(   481.94), EASYSIMD_FLOAT32_C(   857.81) },
      { EASYSIMD_FLOAT32_C(  -828.87), EASYSIMD_FLOAT32_C(   752.37), EASYSIMD_FLOAT32_C(  1098.17), EASYSIMD_FLOAT32_C(  1316.20),
        EASYSIMD_FLOAT32_C(   124.07), EASYSIMD_FLOAT32_C(   210.94), EASYSIMD_FLOAT32_C(   -83.69), EASYSIMD_FLOAT32_C(  -859.48),
        EASYSIMD_FLOAT32_C(    20.07), EASYSIMD_FLOAT32_C(  1149.58), EASYSIMD_FLOAT32_C(   988.92), EASYSIMD_FLOAT32_C( -1478.84),
        EASYSIMD_FLOAT32_C(   138.25), EASYSIMD_FLOAT32_C(   433.07), EASYSIMD_FLOAT32_C(    -2.45), EASYSIMD_FLOAT32_C(  1687.58) } },
    { { EASYSIMD_FLOAT32_C(  -181.47), EASYSIMD_FLOAT32_C(  -740.52), EASYSIMD_FLOAT32_C(   869.02), EASYSIMD_FLOAT32_C(  -309.41),
        EASYSIMD_FLOAT32_C(   -55.04), EASYSIMD_FLOAT32_C(  -689.82), EASYSIMD_FLOAT32_C(   820.28), EASYSIMD_FLOAT32_C(   946.23),
        EASYSIMD_FLOAT32_C(  -415.42), EASYSIMD_FLOAT32_C(  -472.53), EASYSIMD_FLOAT32_C(   615.13), EASYSIMD_FLOAT32_C(  -235.67),
        EASYSIMD_FLOAT32_C(   999.60), EASYSIMD_FLOAT32_C(    99.52), EASYSIMD_FLOAT32_C(   -65.44), EASYSIMD_FLOAT32_C(  -289.46) },
      { EASYSIMD_FLOAT32_C(  -234.78), EASYSIMD_FLOAT32_C(  -189.73), EASYSIMD_FLOAT32_C(  -962.05), EASYSIMD_FLOAT32_C(  -238.64),
        EASYSIMD_FLOAT32_C(   706.69), EASYSIMD_FLOAT32_C(  -604.57), EASYSIMD_FLOAT32_C(    31.56), EASYSIMD_FLOAT32_C(  -271.97),
        EASYSIMD_FLOAT32_C(   819.41), EASYSIMD_FLOAT32_C(  -272.33), EASYSIMD_FLOAT32_C(   -81.90), EASYSIMD_FLOAT32_C(  -862.59),
        EASYSIMD_FLOAT32_C(  -367.13), EASYSIMD_FLOAT32_C(  -599.96), EASYSIMD_FLOAT32_C(   995.22), EASYSIMD_FLOAT32_C(   451.41) },
      { EASYSIMD_FLOAT32_C(   -53.31), EASYSIMD_FLOAT32_C(   550.79), EASYSIMD_FLOAT32_C( -1831.07), EASYSIMD_FLOAT32_C(    70.77),
        EASYSIMD_FLOAT32_C(   761.73), EASYSIMD_FLOAT32_C(    85.25), EASYSIMD_FLOAT32_C(  -788.72), EASYSIMD_FLOAT32_C( -1218.20),
        EASYSIMD_FLOAT32_C(  1234.83), EASYSIMD_FLOAT32_C(   200.20), EASYSIMD_FLOAT32_C(  -697.03), EASYSIMD_FLOAT32_C(  -626.92),
        EASYSIMD_FLOAT32_C( -1366.73), EASYSIMD_FLOAT32_C(  -699.48), EASYSIMD_FLOAT32_C(  1060.66), EASYSIMD_FLOAT32_C(   740.87) } },
    { { EASYSIMD_FLOAT32_C(  -340.48), EASYSIMD_FLOAT32_C(   864.24), EASYSIMD_FLOAT32_C(  -858.00), EASYSIMD_FLOAT32_C(   604.48),
        EASYSIMD_FLOAT32_C(  -825.57), EASYSIMD_FLOAT32_C(   962.28), EASYSIMD_FLOAT32_C(   550.70), EASYSIMD_FLOAT32_C(  -240.99),
        EASYSIMD_FLOAT32_C(  -510.25), EASYSIMD_FLOAT32_C(   165.84), EASYSIMD_FLOAT32_C(   523.34), EASYSIMD_FLOAT32_C(  -510.65),
        EASYSIMD_FLOAT32_C(  -734.65), EASYSIMD_FLOAT32_C(  -542.10), EASYSIMD_FLOAT32_C(   199.89), EASYSIMD_FLOAT32_C(    30.58) },
      { EASYSIMD_FLOAT32_C(   268.17), EASYSIMD_FLOAT32_C(   237.85), EASYSIMD_FLOAT32_C(   791.93), EASYSIMD_FLOAT32_C(   -25.13),
        EASYSIMD_FLOAT32_C(   633.28), EASYSIMD_FLOAT32_C(  -176.50), EASYSIMD_FLOAT32_C(   702.90), EASYSIMD_FLOAT32_C(   452.69),
        EASYSIMD_FLOAT32_C(   551.17), EASYSIMD_FLOAT32_C(  -379.00), EASYSIMD_FLOAT32_C(   590.10), EASYSIMD_FLOAT32_C(  -815.96),
        EASYSIMD_FLOAT32_C(    21.04), EASYSIMD_FLOAT32_C(   585.32), EASYSIMD_FLOAT32_C(   635.45), EASYSIMD_FLOAT32_C(   680.55) },
      { EASYSIMD_FLOAT32_C(   608.65), EASYSIMD_FLOAT32_C(  -626.39), EASYSIMD_FLOAT32_C(  1649.93), EASYSIMD_FLOAT32_C(  -629.61),
        EASYSIMD_FLOAT32_C(  1458.85), EASYSIMD_FLOAT32_C( -1138.78), EASYSIMD_FLOAT32_C(   152.20), EASYSIMD_FLOAT32_C(   693.68),
        EASYSIMD_FLOAT32_C(  1061.42), EASYSIMD_FLOAT32_C(  -544.84), EASYSIMD_FLOAT32_C(    66.76), EASYSIMD_FLOAT32_C(  -305.31),
        EASYSIMD_FLOAT32_C(   755.69), EASYSIMD_FLOAT32_C(  1127.42), EASYSIMD_FLOAT32_C(   435.56), EASYSIMD_FLOAT32_C(   649.97) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_subr_ps(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_subr_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 r = easysimd_mm512_subr_ps(a, b);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_subr_round_ps(EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT32_C(   147.12), EASYSIMD_FLOAT32_C(   955.24), EASYSIMD_FLOAT32_C(   509.74), EASYSIMD_FLOAT32_C(  -653.57),
        EASYSIMD_FLOAT32_C(   263.10), EASYSIMD_FLOAT32_C(  -306.07), EASYSIMD_FLOAT32_C(  -926.92), EASYSIMD_FLOAT32_C(  -285.57),
        EASYSIMD_FLOAT32_C(  -763.45), EASYSIMD_FLOAT32_C(   298.79), EASYSIMD_FLOAT32_C(  -541.82), EASYSIMD_FLOAT32_C(  -524.09),
        EASYSIMD_FLOAT32_C(   207.67), EASYSIMD_FLOAT32_C(  -915.00), EASYSIMD_FLOAT32_C(    57.74), EASYSIMD_FLOAT32_C(   985.88) },
      { EASYSIMD_FLOAT32_C(   631.50), EASYSIMD_FLOAT32_C(  -100.71), EASYSIMD_FLOAT32_C(   210.04), EASYSIMD_FLOAT32_C(   940.64),
        EASYSIMD_FLOAT32_C(  -764.80), EASYSIMD_FLOAT32_C(  -203.36), EASYSIMD_FLOAT32_C(   797.80), EASYSIMD_FLOAT32_C(  -868.04),
        EASYSIMD_FLOAT32_C(   118.47), EASYSIMD_FLOAT32_C(   564.40), EASYSIMD_FLOAT32_C(   714.72), EASYSIMD_FLOAT32_C(   293.74),
        EASYSIMD_FLOAT32_C(  -254.10), EASYSIMD_FLOAT32_C(   -69.67), EASYSIMD_FLOAT32_C(  -955.01), EASYSIMD_FLOAT32_C(   893.02) },
      { EASYSIMD_FLOAT32_C(   484.00), EASYSIMD_FLOAT32_C( -1056.00), EASYSIMD_FLOAT32_C(  -300.00), EASYSIMD_FLOAT32_C(  1594.00),
        EASYSIMD_FLOAT32_C( -1028.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(  1725.00), EASYSIMD_FLOAT32_C(  -582.00),
        EASYSIMD_FLOAT32_C(   882.00), EASYSIMD_FLOAT32_C(   266.00), EASYSIMD_FLOAT32_C(  1257.00), EASYSIMD_FLOAT32_C(   818.00),
        EASYSIMD_FLOAT32_C(  -462.00), EASYSIMD_FLOAT32_C(   845.00), EASYSIMD_FLOAT32_C( -1013.00), EASYSIMD_FLOAT32_C(   -93.00) },
      { EASYSIMD_FLOAT32_C(   484.00), EASYSIMD_FLOAT32_C( -1056.00), EASYSIMD_FLOAT32_C(  -300.00), EASYSIMD_FLOAT32_C(  1594.00),
        EASYSIMD_FLOAT32_C( -1028.00), EASYSIMD_FLOAT32_C(   102.00), EASYSIMD_FLOAT32_C(  1724.00), EASYSIMD_FLOAT32_C(  -583.00),
        EASYSIMD_FLOAT32_C(   881.00), EASYSIMD_FLOAT32_C(   265.00), EASYSIMD_FLOAT32_C(  1256.00), EASYSIMD_FLOAT32_C(   817.00),
        EASYSIMD_FLOAT32_C(  -462.00), EASYSIMD_FLOAT32_C(   845.00), EASYSIMD_FLOAT32_C( -1013.00), EASYSIMD_FLOAT32_C(   -93.00) },
      { EASYSIMD_FLOAT32_C(   485.00), EASYSIMD_FLOAT32_C( -1055.00), EASYSIMD_FLOAT32_C(  -299.00), EASYSIMD_FLOAT32_C(  1595.00),
        EASYSIMD_FLOAT32_C( -1027.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(  1725.00), EASYSIMD_FLOAT32_C(  -582.00),
        EASYSIMD_FLOAT32_C(   882.00), EASYSIMD_FLOAT32_C(   266.00), EASYSIMD_FLOAT32_C(  1257.00), EASYSIMD_FLOAT32_C(   818.00),
        EASYSIMD_FLOAT32_C(  -461.00), EASYSIMD_FLOAT32_C(   846.00), EASYSIMD_FLOAT32_C( -1012.00), EASYSIMD_FLOAT32_C(   -92.00) },
      { EASYSIMD_FLOAT32_C(   484.00), EASYSIMD_FLOAT32_C( -1055.00), EASYSIMD_FLOAT32_C(  -299.00), EASYSIMD_FLOAT32_C(  1594.00),
        EASYSIMD_FLOAT32_C( -1027.00), EASYSIMD_FLOAT32_C(   102.00), EASYSIMD_FLOAT32_C(  1724.00), EASYSIMD_FLOAT32_C(  -582.00),
        EASYSIMD_FLOAT32_C(   881.00), EASYSIMD_FLOAT32_C(   265.00), EASYSIMD_FLOAT32_C(  1256.00), EASYSIMD_FLOAT32_C(   817.00),
        EASYSIMD_FLOAT32_C(  -461.00), EASYSIMD_FLOAT32_C(   845.00), EASYSIMD_FLOAT32_C( -1012.00), EASYSIMD_FLOAT32_C(   -92.00) },
      { EASYSIMD_FLOAT32_C(   484.00), EASYSIMD_FLOAT32_C( -1056.00), EASYSIMD_FLOAT32_C(  -300.00), EASYSIMD_FLOAT32_C(  1594.00),
        EASYSIMD_FLOAT32_C( -1028.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(  1725.00), EASYSIMD_FLOAT32_C(  -582.00),
        EASYSIMD_FLOAT32_C(   882.00), EASYSIMD_FLOAT32_C(   266.00), EASYSIMD_FLOAT32_C(  1257.00), EASYSIMD_FLOAT32_C(   818.00),
        EASYSIMD_FLOAT32_C(  -462.00), EASYSIMD_FLOAT32_C(   845.00), EASYSIMD_FLOAT32_C( -1013.00), EASYSIMD_FLOAT32_C(   -93.00) } },
    { { EASYSIMD_FLOAT32_C(  -114.43), EASYSIMD_FLOAT32_C(   554.74), EASYSIMD_FLOAT32_C(  -760.55), EASYSIMD_FLOAT32_C(  -851.33),
        EASYSIMD_FLOAT32_C(  -751.33), EASYSIMD_FLOAT32_C(  -687.47), EASYSIMD_FLOAT32_C(  -136.90), EASYSIMD_FLOAT32_C(  -514.78),
        EASYSIMD_FLOAT32_C(   611.32), EASYSIMD_FLOAT32_C(   321.28), EASYSIMD_FLOAT32_C(   -38.88), EASYSIMD_FLOAT32_C(  -181.01),
        EASYSIMD_FLOAT32_C(   406.28), EASYSIMD_FLOAT32_C(  -981.14), EASYSIMD_FLOAT32_C(  -195.13), EASYSIMD_FLOAT32_C(    37.78) },
      { EASYSIMD_FLOAT32_C(   -81.85), EASYSIMD_FLOAT32_C(  -985.09), EASYSIMD_FLOAT32_C(   -21.58), EASYSIMD_FLOAT32_C(   153.35),
        EASYSIMD_FLOAT32_C(  -188.44), EASYSIMD_FLOAT32_C(  -223.79), EASYSIMD_FLOAT32_C(   285.31), EASYSIMD_FLOAT32_C(   930.02),
        EASYSIMD_FLOAT32_C(  -659.38), EASYSIMD_FLOAT32_C(     0.03), EASYSIMD_FLOAT32_C(   223.76), EASYSIMD_FLOAT32_C(    86.52),
        EASYSIMD_FLOAT32_C(   930.36), EASYSIMD_FLOAT32_C(   268.75), EASYSIMD_FLOAT32_C(   -20.46), EASYSIMD_FLOAT32_C(  -184.07) },
      { EASYSIMD_FLOAT32_C(    33.00), EASYSIMD_FLOAT32_C( -1540.00), EASYSIMD_FLOAT32_C(   739.00), EASYSIMD_FLOAT32_C(  1005.00),
        EASYSIMD_FLOAT32_C(   563.00), EASYSIMD_FLOAT32_C(   464.00), EASYSIMD_FLOAT32_C(   422.00), EASYSIMD_FLOAT32_C(  1445.00),
        EASYSIMD_FLOAT32_C( -1271.00), EASYSIMD_FLOAT32_C(  -321.00), EASYSIMD_FLOAT32_C(   263.00), EASYSIMD_FLOAT32_C(   268.00),
        EASYSIMD_FLOAT32_C(   524.00), EASYSIMD_FLOAT32_C(  1250.00), EASYSIMD_FLOAT32_C(   175.00), EASYSIMD_FLOAT32_C(  -222.00) },
      { EASYSIMD_FLOAT32_C(    32.00), EASYSIMD_FLOAT32_C( -1540.00), EASYSIMD_FLOAT32_C(   738.00), EASYSIMD_FLOAT32_C(  1004.00),
        EASYSIMD_FLOAT32_C(   562.00), EASYSIMD_FLOAT32_C(   463.00), EASYSIMD_FLOAT32_C(   422.00), EASYSIMD_FLOAT32_C(  1444.00),
        EASYSIMD_FLOAT32_C( -1271.00), EASYSIMD_FLOAT32_C(  -322.00), EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C(   267.00),
        EASYSIMD_FLOAT32_C(   524.00), EASYSIMD_FLOAT32_C(  1249.00), EASYSIMD_FLOAT32_C(   174.00), EASYSIMD_FLOAT32_C(  -222.00) },
      { EASYSIMD_FLOAT32_C(    33.00), EASYSIMD_FLOAT32_C( -1539.00), EASYSIMD_FLOAT32_C(   739.00), EASYSIMD_FLOAT32_C(  1005.00),
        EASYSIMD_FLOAT32_C(   563.00), EASYSIMD_FLOAT32_C(   464.00), EASYSIMD_FLOAT32_C(   423.00), EASYSIMD_FLOAT32_C(  1445.00),
        EASYSIMD_FLOAT32_C( -1270.00), EASYSIMD_FLOAT32_C(  -321.00), EASYSIMD_FLOAT32_C(   263.00), EASYSIMD_FLOAT32_C(   268.00),
        EASYSIMD_FLOAT32_C(   525.00), EASYSIMD_FLOAT32_C(  1250.00), EASYSIMD_FLOAT32_C(   175.00), EASYSIMD_FLOAT32_C(  -221.00) },
      { EASYSIMD_FLOAT32_C(    32.00), EASYSIMD_FLOAT32_C( -1539.00), EASYSIMD_FLOAT32_C(   738.00), EASYSIMD_FLOAT32_C(  1004.00),
        EASYSIMD_FLOAT32_C(   562.00), EASYSIMD_FLOAT32_C(   463.00), EASYSIMD_FLOAT32_C(   422.00), EASYSIMD_FLOAT32_C(  1444.00),
        EASYSIMD_FLOAT32_C( -1270.00), EASYSIMD_FLOAT32_C(  -321.00), EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C(   267.00),
        EASYSIMD_FLOAT32_C(   524.00), EASYSIMD_FLOAT32_C(  1249.00), EASYSIMD_FLOAT32_C(   174.00), EASYSIMD_FLOAT32_C(  -221.00) },
      { EASYSIMD_FLOAT32_C(    33.00), EASYSIMD_FLOAT32_C( -1540.00), EASYSIMD_FLOAT32_C(   739.00), EASYSIMD_FLOAT32_C(  1005.00),
        EASYSIMD_FLOAT32_C(   563.00), EASYSIMD_FLOAT32_C(   464.00), EASYSIMD_FLOAT32_C(   422.00), EASYSIMD_FLOAT32_C(  1445.00),
        EASYSIMD_FLOAT32_C( -1271.00), EASYSIMD_FLOAT32_C(  -321.00), EASYSIMD_FLOAT32_C(   263.00), EASYSIMD_FLOAT32_C(   268.00),
        EASYSIMD_FLOAT32_C(   524.00), EASYSIMD_FLOAT32_C(  1250.00), EASYSIMD_FLOAT32_C(   175.00), EASYSIMD_FLOAT32_C(  -222.00) } },
    { { EASYSIMD_FLOAT32_C(  -176.51), EASYSIMD_FLOAT32_C(   218.99), EASYSIMD_FLOAT32_C(   -35.40), EASYSIMD_FLOAT32_C(    72.16),
        EASYSIMD_FLOAT32_C(   531.52), EASYSIMD_FLOAT32_C(   827.70), EASYSIMD_FLOAT32_C(   557.38), EASYSIMD_FLOAT32_C(   142.84),
        EASYSIMD_FLOAT32_C(   148.99), EASYSIMD_FLOAT32_C(  -481.50), EASYSIMD_FLOAT32_C(   961.82), EASYSIMD_FLOAT32_C(  -444.73),
        EASYSIMD_FLOAT32_C(  -462.64), EASYSIMD_FLOAT32_C(  -233.31), EASYSIMD_FLOAT32_C(   593.05), EASYSIMD_FLOAT32_C(   455.51) },
      { EASYSIMD_FLOAT32_C(  -218.39), EASYSIMD_FLOAT32_C(  -428.54), EASYSIMD_FLOAT32_C(  -391.15), EASYSIMD_FLOAT32_C(   593.16),
        EASYSIMD_FLOAT32_C(   347.68), EASYSIMD_FLOAT32_C(   894.16), EASYSIMD_FLOAT32_C(   523.19), EASYSIMD_FLOAT32_C(   688.30),
        EASYSIMD_FLOAT32_C(  -105.80), EASYSIMD_FLOAT32_C(  -253.05), EASYSIMD_FLOAT32_C(  -225.19), EASYSIMD_FLOAT32_C(  -175.45),
        EASYSIMD_FLOAT32_C(  -984.30), EASYSIMD_FLOAT32_C(   754.36), EASYSIMD_FLOAT32_C(   640.48), EASYSIMD_FLOAT32_C(  -160.81) },
      { EASYSIMD_FLOAT32_C(   -42.00), EASYSIMD_FLOAT32_C(  -648.00), EASYSIMD_FLOAT32_C(  -356.00), EASYSIMD_FLOAT32_C(   521.00),
        EASYSIMD_FLOAT32_C(  -184.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(   -34.00), EASYSIMD_FLOAT32_C(   545.00),
        EASYSIMD_FLOAT32_C(  -255.00), EASYSIMD_FLOAT32_C(   228.00), EASYSIMD_FLOAT32_C( -1187.00), EASYSIMD_FLOAT32_C(   269.00),
        EASYSIMD_FLOAT32_C(  -522.00), EASYSIMD_FLOAT32_C(   988.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(  -616.00) },
      { EASYSIMD_FLOAT32_C(   -42.00), EASYSIMD_FLOAT32_C(  -648.00), EASYSIMD_FLOAT32_C(  -356.00), EASYSIMD_FLOAT32_C(   521.00),
        EASYSIMD_FLOAT32_C(  -184.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(   -35.00), EASYSIMD_FLOAT32_C(   545.00),
        EASYSIMD_FLOAT32_C(  -255.00), EASYSIMD_FLOAT32_C(   228.00), EASYSIMD_FLOAT32_C( -1188.00), EASYSIMD_FLOAT32_C(   269.00),
        EASYSIMD_FLOAT32_C(  -522.00), EASYSIMD_FLOAT32_C(   987.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(  -617.00) },
      { EASYSIMD_FLOAT32_C(   -41.00), EASYSIMD_FLOAT32_C(  -647.00), EASYSIMD_FLOAT32_C(  -355.00), EASYSIMD_FLOAT32_C(   521.00),
        EASYSIMD_FLOAT32_C(  -183.00), EASYSIMD_FLOAT32_C(    67.00), EASYSIMD_FLOAT32_C(   -34.00), EASYSIMD_FLOAT32_C(   546.00),
        EASYSIMD_FLOAT32_C(  -254.00), EASYSIMD_FLOAT32_C(   229.00), EASYSIMD_FLOAT32_C( -1187.00), EASYSIMD_FLOAT32_C(   270.00),
        EASYSIMD_FLOAT32_C(  -521.00), EASYSIMD_FLOAT32_C(   988.00), EASYSIMD_FLOAT32_C(    48.00), EASYSIMD_FLOAT32_C(  -616.00) },
      { EASYSIMD_FLOAT32_C(   -41.00), EASYSIMD_FLOAT32_C(  -647.00), EASYSIMD_FLOAT32_C(  -355.00), EASYSIMD_FLOAT32_C(   521.00),
        EASYSIMD_FLOAT32_C(  -183.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(   -34.00), EASYSIMD_FLOAT32_C(   545.00),
        EASYSIMD_FLOAT32_C(  -254.00), EASYSIMD_FLOAT32_C(   228.00), EASYSIMD_FLOAT32_C( -1187.00), EASYSIMD_FLOAT32_C(   269.00),
        EASYSIMD_FLOAT32_C(  -521.00), EASYSIMD_FLOAT32_C(   987.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(  -616.00) },
      { EASYSIMD_FLOAT32_C(   -42.00), EASYSIMD_FLOAT32_C(  -648.00), EASYSIMD_FLOAT32_C(  -356.00), EASYSIMD_FLOAT32_C(   521.00),
        EASYSIMD_FLOAT32_C(  -184.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(   -34.00), EASYSIMD_FLOAT32_C(   545.00),
        EASYSIMD_FLOAT32_C(  -255.00), EASYSIMD_FLOAT32_C(   228.00), EASYSIMD_FLOAT32_C( -1187.00), EASYSIMD_FLOAT32_C(   269.00),
        EASYSIMD_FLOAT32_C(  -522.00), EASYSIMD_FLOAT32_C(   988.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(  -616.00) } },
    { { EASYSIMD_FLOAT32_C(   -26.65), EASYSIMD_FLOAT32_C(  -394.92), EASYSIMD_FLOAT32_C(   911.35), EASYSIMD_FLOAT32_C(  -495.13),
        EASYSIMD_FLOAT32_C(  -567.21), EASYSIMD_FLOAT32_C(   468.73), EASYSIMD_FLOAT32_C(   647.70), EASYSIMD_FLOAT32_C(   581.77),
        EASYSIMD_FLOAT32_C(   987.23), EASYSIMD_FLOAT32_C(   609.53), EASYSIMD_FLOAT32_C(  -862.96), EASYSIMD_FLOAT32_C(  -475.41),
        EASYSIMD_FLOAT32_C(  -623.78), EASYSIMD_FLOAT32_C(   730.09), EASYSIMD_FLOAT32_C(   980.09), EASYSIMD_FLOAT32_C(   157.82) },
      { EASYSIMD_FLOAT32_C(  -698.45), EASYSIMD_FLOAT32_C(  -411.05), EASYSIMD_FLOAT32_C(  -249.01), EASYSIMD_FLOAT32_C(   649.23),
        EASYSIMD_FLOAT32_C(  -516.89), EASYSIMD_FLOAT32_C(  -725.82), EASYSIMD_FLOAT32_C(   337.53), EASYSIMD_FLOAT32_C(   377.31),
        EASYSIMD_FLOAT32_C(    21.12), EASYSIMD_FLOAT32_C(  -887.66), EASYSIMD_FLOAT32_C(  -798.14), EASYSIMD_FLOAT32_C(    36.83),
        EASYSIMD_FLOAT32_C(   866.70), EASYSIMD_FLOAT32_C(   842.34), EASYSIMD_FLOAT32_C(   876.02), EASYSIMD_FLOAT32_C(  -159.96) },
      { EASYSIMD_FLOAT32_C(  -672.00), EASYSIMD_FLOAT32_C(   -16.00), EASYSIMD_FLOAT32_C( -1160.00), EASYSIMD_FLOAT32_C(  1144.00),
        EASYSIMD_FLOAT32_C(    50.00), EASYSIMD_FLOAT32_C( -1195.00), EASYSIMD_FLOAT32_C(  -310.00), EASYSIMD_FLOAT32_C(  -204.00),
        EASYSIMD_FLOAT32_C(  -966.00), EASYSIMD_FLOAT32_C( -1497.00), EASYSIMD_FLOAT32_C(    65.00), EASYSIMD_FLOAT32_C(   512.00),
        EASYSIMD_FLOAT32_C(  1490.00), EASYSIMD_FLOAT32_C(   112.00), EASYSIMD_FLOAT32_C(  -104.00), EASYSIMD_FLOAT32_C(  -318.00) },
      { EASYSIMD_FLOAT32_C(  -672.00), EASYSIMD_FLOAT32_C(   -17.00), EASYSIMD_FLOAT32_C( -1161.00), EASYSIMD_FLOAT32_C(  1144.00),
        EASYSIMD_FLOAT32_C(    50.00), EASYSIMD_FLOAT32_C( -1195.00), EASYSIMD_FLOAT32_C(  -311.00), EASYSIMD_FLOAT32_C(  -205.00),
        EASYSIMD_FLOAT32_C(  -967.00), EASYSIMD_FLOAT32_C( -1498.00), EASYSIMD_FLOAT32_C(    64.00), EASYSIMD_FLOAT32_C(   512.00),
        EASYSIMD_FLOAT32_C(  1490.00), EASYSIMD_FLOAT32_C(   112.00), EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C(  -318.00) },
      { EASYSIMD_FLOAT32_C(  -671.00), EASYSIMD_FLOAT32_C(   -16.00), EASYSIMD_FLOAT32_C( -1160.00), EASYSIMD_FLOAT32_C(  1145.00),
        EASYSIMD_FLOAT32_C(    51.00), EASYSIMD_FLOAT32_C( -1194.00), EASYSIMD_FLOAT32_C(  -310.00), EASYSIMD_FLOAT32_C(  -204.00),
        EASYSIMD_FLOAT32_C(  -966.00), EASYSIMD_FLOAT32_C( -1497.00), EASYSIMD_FLOAT32_C(    65.00), EASYSIMD_FLOAT32_C(   513.00),
        EASYSIMD_FLOAT32_C(  1491.00), EASYSIMD_FLOAT32_C(   113.00), EASYSIMD_FLOAT32_C(  -104.00), EASYSIMD_FLOAT32_C(  -317.00) },
      { EASYSIMD_FLOAT32_C(  -671.00), EASYSIMD_FLOAT32_C(   -16.00), EASYSIMD_FLOAT32_C( -1160.00), EASYSIMD_FLOAT32_C(  1144.00),
        EASYSIMD_FLOAT32_C(    50.00), EASYSIMD_FLOAT32_C( -1194.00), EASYSIMD_FLOAT32_C(  -310.00), EASYSIMD_FLOAT32_C(  -204.00),
        EASYSIMD_FLOAT32_C(  -966.00), EASYSIMD_FLOAT32_C( -1497.00), EASYSIMD_FLOAT32_C(    64.00), EASYSIMD_FLOAT32_C(   512.00),
        EASYSIMD_FLOAT32_C(  1490.00), EASYSIMD_FLOAT32_C(   112.00), EASYSIMD_FLOAT32_C(  -104.00), EASYSIMD_FLOAT32_C(  -317.00) },
      { EASYSIMD_FLOAT32_C(  -672.00), EASYSIMD_FLOAT32_C(   -16.00), EASYSIMD_FLOAT32_C( -1160.00), EASYSIMD_FLOAT32_C(  1144.00),
        EASYSIMD_FLOAT32_C(    50.00), EASYSIMD_FLOAT32_C( -1195.00), EASYSIMD_FLOAT32_C(  -310.00), EASYSIMD_FLOAT32_C(  -204.00),
        EASYSIMD_FLOAT32_C(  -966.00), EASYSIMD_FLOAT32_C( -1497.00), EASYSIMD_FLOAT32_C(    65.00), EASYSIMD_FLOAT32_C(   512.00),
        EASYSIMD_FLOAT32_C(  1490.00), EASYSIMD_FLOAT32_C(   112.00), EASYSIMD_FLOAT32_C(  -104.00), EASYSIMD_FLOAT32_C(  -318.00) } },
    { { EASYSIMD_FLOAT32_C(  -552.58), EASYSIMD_FLOAT32_C(   787.37), EASYSIMD_FLOAT32_C(   344.91), EASYSIMD_FLOAT32_C(  -119.79),
        EASYSIMD_FLOAT32_C(   256.09), EASYSIMD_FLOAT32_C(    -7.38), EASYSIMD_FLOAT32_C(  -538.01), EASYSIMD_FLOAT32_C(   243.32),
        EASYSIMD_FLOAT32_C(  -397.86), EASYSIMD_FLOAT32_C(  -400.97), EASYSIMD_FLOAT32_C(   767.91), EASYSIMD_FLOAT32_C(   -21.64),
        EASYSIMD_FLOAT32_C(  -670.89), EASYSIMD_FLOAT32_C(   748.00), EASYSIMD_FLOAT32_C(  -863.81), EASYSIMD_FLOAT32_C(  -369.34) },
      { EASYSIMD_FLOAT32_C(  -663.05), EASYSIMD_FLOAT32_C(  -112.83), EASYSIMD_FLOAT32_C(  -720.11), EASYSIMD_FLOAT32_C(  -179.94),
        EASYSIMD_FLOAT32_C(   161.35), EASYSIMD_FLOAT32_C(   617.42), EASYSIMD_FLOAT32_C(  -802.63), EASYSIMD_FLOAT32_C(  -817.53),
        EASYSIMD_FLOAT32_C(   729.76), EASYSIMD_FLOAT32_C(  -600.78), EASYSIMD_FLOAT32_C(   219.30), EASYSIMD_FLOAT32_C(   596.45),
        EASYSIMD_FLOAT32_C(  -758.43), EASYSIMD_FLOAT32_C(    95.32), EASYSIMD_FLOAT32_C(  -563.50), EASYSIMD_FLOAT32_C(  -311.01) },
      { EASYSIMD_FLOAT32_C(  -110.00), EASYSIMD_FLOAT32_C(  -900.00), EASYSIMD_FLOAT32_C( -1065.00), EASYSIMD_FLOAT32_C(   -60.00),
        EASYSIMD_FLOAT32_C(   -95.00), EASYSIMD_FLOAT32_C(   625.00), EASYSIMD_FLOAT32_C(  -265.00), EASYSIMD_FLOAT32_C( -1061.00),
        EASYSIMD_FLOAT32_C(  1128.00), EASYSIMD_FLOAT32_C(  -200.00), EASYSIMD_FLOAT32_C(  -549.00), EASYSIMD_FLOAT32_C(   618.00),
        EASYSIMD_FLOAT32_C(   -88.00), EASYSIMD_FLOAT32_C(  -653.00), EASYSIMD_FLOAT32_C(   300.00), EASYSIMD_FLOAT32_C(    58.00) },
      { EASYSIMD_FLOAT32_C(  -111.00), EASYSIMD_FLOAT32_C(  -901.00), EASYSIMD_FLOAT32_C( -1066.00), EASYSIMD_FLOAT32_C(   -61.00),
        EASYSIMD_FLOAT32_C(   -95.00), EASYSIMD_FLOAT32_C(   624.00), EASYSIMD_FLOAT32_C(  -265.00), EASYSIMD_FLOAT32_C( -1061.00),
        EASYSIMD_FLOAT32_C(  1127.00), EASYSIMD_FLOAT32_C(  -200.00), EASYSIMD_FLOAT32_C(  -549.00), EASYSIMD_FLOAT32_C(   618.00),
        EASYSIMD_FLOAT32_C(   -88.00), EASYSIMD_FLOAT32_C(  -653.00), EASYSIMD_FLOAT32_C(   300.00), EASYSIMD_FLOAT32_C(    58.00) },
      { EASYSIMD_FLOAT32_C(  -110.00), EASYSIMD_FLOAT32_C(  -900.00), EASYSIMD_FLOAT32_C( -1065.00), EASYSIMD_FLOAT32_C(   -60.00),
        EASYSIMD_FLOAT32_C(   -94.00), EASYSIMD_FLOAT32_C(   625.00), EASYSIMD_FLOAT32_C(  -264.00), EASYSIMD_FLOAT32_C( -1060.00),
        EASYSIMD_FLOAT32_C(  1128.00), EASYSIMD_FLOAT32_C(  -199.00), EASYSIMD_FLOAT32_C(  -548.00), EASYSIMD_FLOAT32_C(   619.00),
        EASYSIMD_FLOAT32_C(   -87.00), EASYSIMD_FLOAT32_C(  -652.00), EASYSIMD_FLOAT32_C(   301.00), EASYSIMD_FLOAT32_C(    59.00) },
      { EASYSIMD_FLOAT32_C(  -110.00), EASYSIMD_FLOAT32_C(  -900.00), EASYSIMD_FLOAT32_C( -1065.00), EASYSIMD_FLOAT32_C(   -60.00),
        EASYSIMD_FLOAT32_C(   -94.00), EASYSIMD_FLOAT32_C(   624.00), EASYSIMD_FLOAT32_C(  -264.00), EASYSIMD_FLOAT32_C( -1060.00),
        EASYSIMD_FLOAT32_C(  1127.00), EASYSIMD_FLOAT32_C(  -199.00), EASYSIMD_FLOAT32_C(  -548.00), EASYSIMD_FLOAT32_C(   618.00),
        EASYSIMD_FLOAT32_C(   -87.00), EASYSIMD_FLOAT32_C(  -652.00), EASYSIMD_FLOAT32_C(   300.00), EASYSIMD_FLOAT32_C(    58.00) },
      { EASYSIMD_FLOAT32_C(  -110.00), EASYSIMD_FLOAT32_C(  -900.00), EASYSIMD_FLOAT32_C( -1065.00), EASYSIMD_FLOAT32_C(   -60.00),
        EASYSIMD_FLOAT32_C(   -95.00), EASYSIMD_FLOAT32_C(   625.00), EASYSIMD_FLOAT32_C(  -265.00), EASYSIMD_FLOAT32_C( -1061.00),
        EASYSIMD_FLOAT32_C(  1128.00), EASYSIMD_FLOAT32_C(  -200.00), EASYSIMD_FLOAT32_C(  -549.00), EASYSIMD_FLOAT32_C(   618.00),
        EASYSIMD_FLOAT32_C(   -88.00), EASYSIMD_FLOAT32_C(  -653.00), EASYSIMD_FLOAT32_C(   300.00), EASYSIMD_FLOAT32_C(    58.00) } },
    { { EASYSIMD_FLOAT32_C(  -117.32), EASYSIMD_FLOAT32_C(   781.41), EASYSIMD_FLOAT32_C(   569.20), EASYSIMD_FLOAT32_C(  -861.22),
        EASYSIMD_FLOAT32_C(  -225.97), EASYSIMD_FLOAT32_C(  -968.81), EASYSIMD_FLOAT32_C(   382.10), EASYSIMD_FLOAT32_C(   376.17),
        EASYSIMD_FLOAT32_C(  -369.78), EASYSIMD_FLOAT32_C(   150.01), EASYSIMD_FLOAT32_C(  -645.47), EASYSIMD_FLOAT32_C(   -40.67),
        EASYSIMD_FLOAT32_C(  -101.99), EASYSIMD_FLOAT32_C(  -509.28), EASYSIMD_FLOAT32_C(   589.99), EASYSIMD_FLOAT32_C(   234.95) },
      { EASYSIMD_FLOAT32_C(   377.89), EASYSIMD_FLOAT32_C(   869.88), EASYSIMD_FLOAT32_C(  -944.99), EASYSIMD_FLOAT32_C(  -460.76),
        EASYSIMD_FLOAT32_C(   487.30), EASYSIMD_FLOAT32_C(  -747.62), EASYSIMD_FLOAT32_C(  -278.29), EASYSIMD_FLOAT32_C(   217.06),
        EASYSIMD_FLOAT32_C(  -348.40), EASYSIMD_FLOAT32_C(   941.01), EASYSIMD_FLOAT32_C(  -186.49), EASYSIMD_FLOAT32_C(  -106.83),
        EASYSIMD_FLOAT32_C(    36.33), EASYSIMD_FLOAT32_C(   250.01), EASYSIMD_FLOAT32_C(   582.16), EASYSIMD_FLOAT32_C(   919.01) },
      { EASYSIMD_FLOAT32_C(   495.00), EASYSIMD_FLOAT32_C(    88.00), EASYSIMD_FLOAT32_C( -1514.00), EASYSIMD_FLOAT32_C(   400.00),
        EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(   221.00), EASYSIMD_FLOAT32_C(  -660.00), EASYSIMD_FLOAT32_C(  -159.00),
        EASYSIMD_FLOAT32_C(    21.00), EASYSIMD_FLOAT32_C(   791.00), EASYSIMD_FLOAT32_C(   459.00), EASYSIMD_FLOAT32_C(   -66.00),
        EASYSIMD_FLOAT32_C(   138.00), EASYSIMD_FLOAT32_C(   759.00), EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(   684.00) },
      { EASYSIMD_FLOAT32_C(   495.00), EASYSIMD_FLOAT32_C(    88.00), EASYSIMD_FLOAT32_C( -1515.00), EASYSIMD_FLOAT32_C(   400.00),
        EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(   221.00), EASYSIMD_FLOAT32_C(  -661.00), EASYSIMD_FLOAT32_C(  -160.00),
        EASYSIMD_FLOAT32_C(    21.00), EASYSIMD_FLOAT32_C(   791.00), EASYSIMD_FLOAT32_C(   458.00), EASYSIMD_FLOAT32_C(   -67.00),
        EASYSIMD_FLOAT32_C(   138.00), EASYSIMD_FLOAT32_C(   759.00), EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(   684.00) },
      { EASYSIMD_FLOAT32_C(   496.00), EASYSIMD_FLOAT32_C(    89.00), EASYSIMD_FLOAT32_C( -1514.00), EASYSIMD_FLOAT32_C(   401.00),
        EASYSIMD_FLOAT32_C(   714.00), EASYSIMD_FLOAT32_C(   222.00), EASYSIMD_FLOAT32_C(  -660.00), EASYSIMD_FLOAT32_C(  -159.00),
        EASYSIMD_FLOAT32_C(    22.00), EASYSIMD_FLOAT32_C(   791.00), EASYSIMD_FLOAT32_C(   459.00), EASYSIMD_FLOAT32_C(   -66.00),
        EASYSIMD_FLOAT32_C(   139.00), EASYSIMD_FLOAT32_C(   760.00), EASYSIMD_FLOAT32_C(    -7.00), EASYSIMD_FLOAT32_C(   685.00) },
      { EASYSIMD_FLOAT32_C(   495.00), EASYSIMD_FLOAT32_C(    88.00), EASYSIMD_FLOAT32_C( -1514.00), EASYSIMD_FLOAT32_C(   400.00),
        EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(   221.00), EASYSIMD_FLOAT32_C(  -660.00), EASYSIMD_FLOAT32_C(  -159.00),
        EASYSIMD_FLOAT32_C(    21.00), EASYSIMD_FLOAT32_C(   791.00), EASYSIMD_FLOAT32_C(   458.00), EASYSIMD_FLOAT32_C(   -66.00),
        EASYSIMD_FLOAT32_C(   138.00), EASYSIMD_FLOAT32_C(   759.00), EASYSIMD_FLOAT32_C(    -7.00), EASYSIMD_FLOAT32_C(   684.00) },
      { EASYSIMD_FLOAT32_C(   495.00), EASYSIMD_FLOAT32_C(    88.00), EASYSIMD_FLOAT32_C( -1514.00), EASYSIMD_FLOAT32_C(   400.00),
        EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(   221.00), EASYSIMD_FLOAT32_C(  -660.00), EASYSIMD_FLOAT32_C(  -159.00),
        EASYSIMD_FLOAT32_C(    21.00), EASYSIMD_FLOAT32_C(   791.00), EASYSIMD_FLOAT32_C(   459.00), EASYSIMD_FLOAT32_C(   -66.00),
        EASYSIMD_FLOAT32_C(   138.00), EASYSIMD_FLOAT32_C(   759.00), EASYSIMD_FLOAT32_C(    -8.00), EASYSIMD_FLOAT32_C(   684.00) } },
    { { EASYSIMD_FLOAT32_C(    31.42), EASYSIMD_FLOAT32_C(   151.37), EASYSIMD_FLOAT32_C(  -942.21), EASYSIMD_FLOAT32_C(   805.45),
        EASYSIMD_FLOAT32_C(   182.56), EASYSIMD_FLOAT32_C(   439.89), EASYSIMD_FLOAT32_C(   181.62), EASYSIMD_FLOAT32_C(   812.77),
        EASYSIMD_FLOAT32_C(  -410.10), EASYSIMD_FLOAT32_C(   536.15), EASYSIMD_FLOAT32_C(  -227.90), EASYSIMD_FLOAT32_C(   487.90),
        EASYSIMD_FLOAT32_C(  -973.13), EASYSIMD_FLOAT32_C(  -637.91), EASYSIMD_FLOAT32_C(  -277.14), EASYSIMD_FLOAT32_C(   404.76) },
      { EASYSIMD_FLOAT32_C(  -768.03), EASYSIMD_FLOAT32_C(  -222.13), EASYSIMD_FLOAT32_C(   944.00), EASYSIMD_FLOAT32_C(   719.27),
        EASYSIMD_FLOAT32_C(    30.25), EASYSIMD_FLOAT32_C(  -334.29), EASYSIMD_FLOAT32_C(   -63.67), EASYSIMD_FLOAT32_C(   681.85),
        EASYSIMD_FLOAT32_C(  -393.28), EASYSIMD_FLOAT32_C(   749.84), EASYSIMD_FLOAT32_C(  -424.98), EASYSIMD_FLOAT32_C(   643.05),
        EASYSIMD_FLOAT32_C(    -0.16), EASYSIMD_FLOAT32_C(  -842.81), EASYSIMD_FLOAT32_C(   562.06), EASYSIMD_FLOAT32_C(  -968.74) },
      { EASYSIMD_FLOAT32_C(  -799.00), EASYSIMD_FLOAT32_C(  -374.00), EASYSIMD_FLOAT32_C(  1886.00), EASYSIMD_FLOAT32_C(   -86.00),
        EASYSIMD_FLOAT32_C(  -152.00), EASYSIMD_FLOAT32_C(  -774.00), EASYSIMD_FLOAT32_C(  -245.00), EASYSIMD_FLOAT32_C(  -131.00),
        EASYSIMD_FLOAT32_C(    17.00), EASYSIMD_FLOAT32_C(   214.00), EASYSIMD_FLOAT32_C(  -197.00), EASYSIMD_FLOAT32_C(   155.00),
        EASYSIMD_FLOAT32_C(   973.00), EASYSIMD_FLOAT32_C(  -205.00), EASYSIMD_FLOAT32_C(   839.00), EASYSIMD_FLOAT32_C( -1374.00) },
      { EASYSIMD_FLOAT32_C(  -800.00), EASYSIMD_FLOAT32_C(  -374.00), EASYSIMD_FLOAT32_C(  1886.00), EASYSIMD_FLOAT32_C(   -87.00),
        EASYSIMD_FLOAT32_C(  -153.00), EASYSIMD_FLOAT32_C(  -775.00), EASYSIMD_FLOAT32_C(  -246.00), EASYSIMD_FLOAT32_C(  -131.00),
        EASYSIMD_FLOAT32_C(    16.00), EASYSIMD_FLOAT32_C(   213.00), EASYSIMD_FLOAT32_C(  -198.00), EASYSIMD_FLOAT32_C(   155.00),
        EASYSIMD_FLOAT32_C(   972.00), EASYSIMD_FLOAT32_C(  -205.00), EASYSIMD_FLOAT32_C(   839.00), EASYSIMD_FLOAT32_C( -1374.00) },
      { EASYSIMD_FLOAT32_C(  -799.00), EASYSIMD_FLOAT32_C(  -373.00), EASYSIMD_FLOAT32_C(  1887.00), EASYSIMD_FLOAT32_C(   -86.00),
        EASYSIMD_FLOAT32_C(  -152.00), EASYSIMD_FLOAT32_C(  -774.00), EASYSIMD_FLOAT32_C(  -245.00), EASYSIMD_FLOAT32_C(  -130.00),
        EASYSIMD_FLOAT32_C(    17.00), EASYSIMD_FLOAT32_C(   214.00), EASYSIMD_FLOAT32_C(  -197.00), EASYSIMD_FLOAT32_C(   156.00),
        EASYSIMD_FLOAT32_C(   973.00), EASYSIMD_FLOAT32_C(  -204.00), EASYSIMD_FLOAT32_C(   840.00), EASYSIMD_FLOAT32_C( -1373.00) },
      { EASYSIMD_FLOAT32_C(  -799.00), EASYSIMD_FLOAT32_C(  -373.00), EASYSIMD_FLOAT32_C(  1886.00), EASYSIMD_FLOAT32_C(   -86.00),
        EASYSIMD_FLOAT32_C(  -152.00), EASYSIMD_FLOAT32_C(  -774.00), EASYSIMD_FLOAT32_C(  -245.00), EASYSIMD_FLOAT32_C(  -130.00),
        EASYSIMD_FLOAT32_C(    16.00), EASYSIMD_FLOAT32_C(   213.00), EASYSIMD_FLOAT32_C(  -197.00), EASYSIMD_FLOAT32_C(   155.00),
        EASYSIMD_FLOAT32_C(   972.00), EASYSIMD_FLOAT32_C(  -204.00), EASYSIMD_FLOAT32_C(   839.00), EASYSIMD_FLOAT32_C( -1373.00) },
      { EASYSIMD_FLOAT32_C(  -799.00), EASYSIMD_FLOAT32_C(  -374.00), EASYSIMD_FLOAT32_C(  1886.00), EASYSIMD_FLOAT32_C(   -86.00),
        EASYSIMD_FLOAT32_C(  -152.00), EASYSIMD_FLOAT32_C(  -774.00), EASYSIMD_FLOAT32_C(  -245.00), EASYSIMD_FLOAT32_C(  -131.00),
        EASYSIMD_FLOAT32_C(    17.00), EASYSIMD_FLOAT32_C(   214.00), EASYSIMD_FLOAT32_C(  -197.00), EASYSIMD_FLOAT32_C(   155.00),
        EASYSIMD_FLOAT32_C(   973.00), EASYSIMD_FLOAT32_C(  -205.00), EASYSIMD_FLOAT32_C(   839.00), EASYSIMD_FLOAT32_C( -1374.00) } },
    { { EASYSIMD_FLOAT32_C(   308.55), EASYSIMD_FLOAT32_C(   619.85), EASYSIMD_FLOAT32_C(   836.71), EASYSIMD_FLOAT32_C(  -508.89),
        EASYSIMD_FLOAT32_C(    59.74), EASYSIMD_FLOAT32_C(    18.32), EASYSIMD_FLOAT32_C(  -696.12), EASYSIMD_FLOAT32_C(   649.64),
        EASYSIMD_FLOAT32_C(  -445.53), EASYSIMD_FLOAT32_C(    75.98), EASYSIMD_FLOAT32_C(   137.54), EASYSIMD_FLOAT32_C(  -418.66),
        EASYSIMD_FLOAT32_C(   438.07), EASYSIMD_FLOAT32_C(   860.40), EASYSIMD_FLOAT32_C(   986.10), EASYSIMD_FLOAT32_C(   670.05) },
      { EASYSIMD_FLOAT32_C(  -361.73), EASYSIMD_FLOAT32_C(   930.10), EASYSIMD_FLOAT32_C(   389.32), EASYSIMD_FLOAT32_C(   668.53),
        EASYSIMD_FLOAT32_C(  -404.18), EASYSIMD_FLOAT32_C(  -674.35), EASYSIMD_FLOAT32_C(   350.38), EASYSIMD_FLOAT32_C(   202.54),
        EASYSIMD_FLOAT32_C(  -924.51), EASYSIMD_FLOAT32_C(   925.40), EASYSIMD_FLOAT32_C(  -154.41), EASYSIMD_FLOAT32_C(    75.33),
        EASYSIMD_FLOAT32_C(  -917.41), EASYSIMD_FLOAT32_C(  -592.35), EASYSIMD_FLOAT32_C(   106.59), EASYSIMD_FLOAT32_C(   391.14) },
      { EASYSIMD_FLOAT32_C(  -670.00), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -447.00), EASYSIMD_FLOAT32_C(  1177.00),
        EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(  -693.00), EASYSIMD_FLOAT32_C(  1046.00), EASYSIMD_FLOAT32_C(  -447.00),
        EASYSIMD_FLOAT32_C(  -479.00), EASYSIMD_FLOAT32_C(   849.00), EASYSIMD_FLOAT32_C(  -292.00), EASYSIMD_FLOAT32_C(   494.00),
        EASYSIMD_FLOAT32_C( -1355.00), EASYSIMD_FLOAT32_C( -1453.00), EASYSIMD_FLOAT32_C(  -880.00), EASYSIMD_FLOAT32_C(  -279.00) },
      { EASYSIMD_FLOAT32_C(  -671.00), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -448.00), EASYSIMD_FLOAT32_C(  1177.00),
        EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(  -693.00), EASYSIMD_FLOAT32_C(  1046.00), EASYSIMD_FLOAT32_C(  -448.00),
        EASYSIMD_FLOAT32_C(  -479.00), EASYSIMD_FLOAT32_C(   849.00), EASYSIMD_FLOAT32_C(  -292.00), EASYSIMD_FLOAT32_C(   493.00),
        EASYSIMD_FLOAT32_C( -1356.00), EASYSIMD_FLOAT32_C( -1453.00), EASYSIMD_FLOAT32_C(  -880.00), EASYSIMD_FLOAT32_C(  -279.00) },
      { EASYSIMD_FLOAT32_C(  -670.00), EASYSIMD_FLOAT32_C(   311.00), EASYSIMD_FLOAT32_C(  -447.00), EASYSIMD_FLOAT32_C(  1178.00),
        EASYSIMD_FLOAT32_C(  -463.00), EASYSIMD_FLOAT32_C(  -692.00), EASYSIMD_FLOAT32_C(  1047.00), EASYSIMD_FLOAT32_C(  -447.00),
        EASYSIMD_FLOAT32_C(  -478.00), EASYSIMD_FLOAT32_C(   850.00), EASYSIMD_FLOAT32_C(  -291.00), EASYSIMD_FLOAT32_C(   494.00),
        EASYSIMD_FLOAT32_C( -1355.00), EASYSIMD_FLOAT32_C( -1452.00), EASYSIMD_FLOAT32_C(  -879.00), EASYSIMD_FLOAT32_C(  -278.00) },
      { EASYSIMD_FLOAT32_C(  -670.00), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -447.00), EASYSIMD_FLOAT32_C(  1177.00),
        EASYSIMD_FLOAT32_C(  -463.00), EASYSIMD_FLOAT32_C(  -692.00), EASYSIMD_FLOAT32_C(  1046.00), EASYSIMD_FLOAT32_C(  -447.00),
        EASYSIMD_FLOAT32_C(  -478.00), EASYSIMD_FLOAT32_C(   849.00), EASYSIMD_FLOAT32_C(  -291.00), EASYSIMD_FLOAT32_C(   493.00),
        EASYSIMD_FLOAT32_C( -1355.00), EASYSIMD_FLOAT32_C( -1452.00), EASYSIMD_FLOAT32_C(  -879.00), EASYSIMD_FLOAT32_C(  -278.00) },
      { EASYSIMD_FLOAT32_C(  -670.00), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -447.00), EASYSIMD_FLOAT32_C(  1177.00),
        EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(  -693.00), EASYSIMD_FLOAT32_C(  1046.00), EASYSIMD_FLOAT32_C(  -447.00),
        EASYSIMD_FLOAT32_C(  -479.00), EASYSIMD_FLOAT32_C(   849.00), EASYSIMD_FLOAT32_C(  -292.00), EASYSIMD_FLOAT32_C(   494.00),
        EASYSIMD_FLOAT32_C( -1355.00), EASYSIMD_FLOAT32_C( -1453.00), EASYSIMD_FLOAT32_C(  -880.00), EASYSIMD_FLOAT32_C(  -279.00) } }
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

    r = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512_close(r, nearest_inf, 1);

    r = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512_close(r, neg_inf, 1);

    r = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512_close(r, pos_inf, 1);

    r = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_subr_round_ps");
    easysimd_assert_m512_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512 nearest_inf = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512 neg_inf = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512 pos_inf = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512 zero = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512 direction = easysimd_mm512_subr_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

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
test_easysimd_mm512_subr_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   449.56), EASYSIMD_FLOAT64_C(   777.45), EASYSIMD_FLOAT64_C(   285.03), EASYSIMD_FLOAT64_C(   623.99),
        EASYSIMD_FLOAT64_C(   739.72), EASYSIMD_FLOAT64_C(  -164.27), EASYSIMD_FLOAT64_C(  -617.00), EASYSIMD_FLOAT64_C(  -770.53) },
      { EASYSIMD_FLOAT64_C(  -998.43), EASYSIMD_FLOAT64_C(   906.34), EASYSIMD_FLOAT64_C(  -281.18), EASYSIMD_FLOAT64_C(  -733.08),
        EASYSIMD_FLOAT64_C(  -635.76), EASYSIMD_FLOAT64_C(   918.72), EASYSIMD_FLOAT64_C(   297.50), EASYSIMD_FLOAT64_C(   632.42) },
      { EASYSIMD_FLOAT64_C( -1447.99), EASYSIMD_FLOAT64_C(   128.89), EASYSIMD_FLOAT64_C(  -566.21), EASYSIMD_FLOAT64_C( -1357.07),
        EASYSIMD_FLOAT64_C( -1375.48), EASYSIMD_FLOAT64_C(  1082.99), EASYSIMD_FLOAT64_C(   914.50), EASYSIMD_FLOAT64_C(  1402.95) } },
    { { EASYSIMD_FLOAT64_C(   156.56), EASYSIMD_FLOAT64_C(    89.44), EASYSIMD_FLOAT64_C(  -392.72), EASYSIMD_FLOAT64_C(  -210.16),
        EASYSIMD_FLOAT64_C(   912.93), EASYSIMD_FLOAT64_C(  -689.82), EASYSIMD_FLOAT64_C(  -757.47), EASYSIMD_FLOAT64_C(   464.10) },
      { EASYSIMD_FLOAT64_C(   -68.82), EASYSIMD_FLOAT64_C(   832.62), EASYSIMD_FLOAT64_C(   648.14), EASYSIMD_FLOAT64_C(   952.22),
        EASYSIMD_FLOAT64_C(   417.94), EASYSIMD_FLOAT64_C(   283.59), EASYSIMD_FLOAT64_C(   632.77), EASYSIMD_FLOAT64_C(  -132.49) },
      { EASYSIMD_FLOAT64_C(  -225.38), EASYSIMD_FLOAT64_C(   743.18), EASYSIMD_FLOAT64_C(  1040.86), EASYSIMD_FLOAT64_C(  1162.38),
        EASYSIMD_FLOAT64_C(  -494.99), EASYSIMD_FLOAT64_C(   973.41), EASYSIMD_FLOAT64_C(  1390.24), EASYSIMD_FLOAT64_C(  -596.59) } },
    { { EASYSIMD_FLOAT64_C(    61.03), EASYSIMD_FLOAT64_C(   -82.20), EASYSIMD_FLOAT64_C(  -508.50), EASYSIMD_FLOAT64_C(  -199.24),
        EASYSIMD_FLOAT64_C(   753.54), EASYSIMD_FLOAT64_C(  -125.51), EASYSIMD_FLOAT64_C(    30.23), EASYSIMD_FLOAT64_C(   755.11) },
      { EASYSIMD_FLOAT64_C(  -219.17), EASYSIMD_FLOAT64_C(   749.05), EASYSIMD_FLOAT64_C(  -977.97), EASYSIMD_FLOAT64_C(   145.08),
        EASYSIMD_FLOAT64_C(   667.77), EASYSIMD_FLOAT64_C(   319.53), EASYSIMD_FLOAT64_C(  -222.51), EASYSIMD_FLOAT64_C(  -175.67) },
      { EASYSIMD_FLOAT64_C(  -280.20), EASYSIMD_FLOAT64_C(   831.25), EASYSIMD_FLOAT64_C(  -469.47), EASYSIMD_FLOAT64_C(   344.32),
        EASYSIMD_FLOAT64_C(   -85.77), EASYSIMD_FLOAT64_C(   445.04), EASYSIMD_FLOAT64_C(  -252.74), EASYSIMD_FLOAT64_C(  -930.78) } },
    { { EASYSIMD_FLOAT64_C(  -591.03), EASYSIMD_FLOAT64_C(   384.78), EASYSIMD_FLOAT64_C(   614.17), EASYSIMD_FLOAT64_C(  -678.10),
        EASYSIMD_FLOAT64_C(   694.96), EASYSIMD_FLOAT64_C(   856.70), EASYSIMD_FLOAT64_C(   786.00), EASYSIMD_FLOAT64_C(  -373.86) },
      { EASYSIMD_FLOAT64_C(   689.33), EASYSIMD_FLOAT64_C(   434.15), EASYSIMD_FLOAT64_C(  -421.64), EASYSIMD_FLOAT64_C(   107.27),
        EASYSIMD_FLOAT64_C(  -282.27), EASYSIMD_FLOAT64_C(  -788.86), EASYSIMD_FLOAT64_C(   974.78), EASYSIMD_FLOAT64_C(   778.77) },
      { EASYSIMD_FLOAT64_C(  1280.36), EASYSIMD_FLOAT64_C(    49.37), EASYSIMD_FLOAT64_C( -1035.81), EASYSIMD_FLOAT64_C(   785.37),
        EASYSIMD_FLOAT64_C(  -977.23), EASYSIMD_FLOAT64_C( -1645.56), EASYSIMD_FLOAT64_C(   188.78), EASYSIMD_FLOAT64_C(  1152.63) } },
    { { EASYSIMD_FLOAT64_C(   128.94), EASYSIMD_FLOAT64_C(  -533.73), EASYSIMD_FLOAT64_C(  -420.48), EASYSIMD_FLOAT64_C(  -117.52),
        EASYSIMD_FLOAT64_C(   340.77), EASYSIMD_FLOAT64_C(   609.75), EASYSIMD_FLOAT64_C(  -362.42), EASYSIMD_FLOAT64_C(  -878.40) },
      { EASYSIMD_FLOAT64_C(   358.80), EASYSIMD_FLOAT64_C(  -340.39), EASYSIMD_FLOAT64_C(   266.68), EASYSIMD_FLOAT64_C(    26.56),
        EASYSIMD_FLOAT64_C(   979.14), EASYSIMD_FLOAT64_C(  -955.83), EASYSIMD_FLOAT64_C(   850.90), EASYSIMD_FLOAT64_C(  -611.89) },
      { EASYSIMD_FLOAT64_C(   229.86), EASYSIMD_FLOAT64_C(   193.34), EASYSIMD_FLOAT64_C(   687.16), EASYSIMD_FLOAT64_C(   144.08),
        EASYSIMD_FLOAT64_C(   638.37), EASYSIMD_FLOAT64_C( -1565.58), EASYSIMD_FLOAT64_C(  1213.32), EASYSIMD_FLOAT64_C(   266.51) } },
    { { EASYSIMD_FLOAT64_C(   428.95), EASYSIMD_FLOAT64_C(   465.07), EASYSIMD_FLOAT64_C(  -289.98), EASYSIMD_FLOAT64_C(   123.91),
        EASYSIMD_FLOAT64_C(   321.77), EASYSIMD_FLOAT64_C(  -503.98), EASYSIMD_FLOAT64_C(   750.05), EASYSIMD_FLOAT64_C(    11.09) },
      { EASYSIMD_FLOAT64_C(   930.17), EASYSIMD_FLOAT64_C(  -671.58), EASYSIMD_FLOAT64_C(  -881.64), EASYSIMD_FLOAT64_C(  -352.10),
        EASYSIMD_FLOAT64_C(  -460.44), EASYSIMD_FLOAT64_C(  -906.86), EASYSIMD_FLOAT64_C(  -573.33), EASYSIMD_FLOAT64_C(   668.50) },
      { EASYSIMD_FLOAT64_C(   501.22), EASYSIMD_FLOAT64_C( -1136.65), EASYSIMD_FLOAT64_C(  -591.66), EASYSIMD_FLOAT64_C(  -476.01),
        EASYSIMD_FLOAT64_C(  -782.21), EASYSIMD_FLOAT64_C(  -402.88), EASYSIMD_FLOAT64_C( -1323.38), EASYSIMD_FLOAT64_C(   657.41) } },
    { { EASYSIMD_FLOAT64_C(  -440.58), EASYSIMD_FLOAT64_C(     6.19), EASYSIMD_FLOAT64_C(  -449.03), EASYSIMD_FLOAT64_C(   900.18),
        EASYSIMD_FLOAT64_C(  -384.06), EASYSIMD_FLOAT64_C(   188.55), EASYSIMD_FLOAT64_C(  -978.21), EASYSIMD_FLOAT64_C(   974.74) },
      { EASYSIMD_FLOAT64_C(   848.16), EASYSIMD_FLOAT64_C(   288.47), EASYSIMD_FLOAT64_C(     1.31), EASYSIMD_FLOAT64_C(   827.31),
        EASYSIMD_FLOAT64_C(   332.64), EASYSIMD_FLOAT64_C(  -147.80), EASYSIMD_FLOAT64_C(  -784.58), EASYSIMD_FLOAT64_C(  -238.41) },
      { EASYSIMD_FLOAT64_C(  1288.74), EASYSIMD_FLOAT64_C(   282.28), EASYSIMD_FLOAT64_C(   450.34), EASYSIMD_FLOAT64_C(   -72.87),
        EASYSIMD_FLOAT64_C(   716.70), EASYSIMD_FLOAT64_C(  -336.35), EASYSIMD_FLOAT64_C(   193.63), EASYSIMD_FLOAT64_C( -1213.15) } },
    { { EASYSIMD_FLOAT64_C(  -682.73), EASYSIMD_FLOAT64_C(   -74.56), EASYSIMD_FLOAT64_C(   885.50), EASYSIMD_FLOAT64_C(   639.04),
        EASYSIMD_FLOAT64_C(   421.46), EASYSIMD_FLOAT64_C(   635.55), EASYSIMD_FLOAT64_C(  -349.87), EASYSIMD_FLOAT64_C(   351.63) },
      { EASYSIMD_FLOAT64_C(   963.97), EASYSIMD_FLOAT64_C(  -231.51), EASYSIMD_FLOAT64_C(   999.53), EASYSIMD_FLOAT64_C(  -496.48),
        EASYSIMD_FLOAT64_C(  -138.36), EASYSIMD_FLOAT64_C(  -573.80), EASYSIMD_FLOAT64_C(  -827.98), EASYSIMD_FLOAT64_C(   421.05) },
      { EASYSIMD_FLOAT64_C(  1646.70), EASYSIMD_FLOAT64_C(  -156.95), EASYSIMD_FLOAT64_C(   114.03), EASYSIMD_FLOAT64_C( -1135.52),
        EASYSIMD_FLOAT64_C(  -559.82), EASYSIMD_FLOAT64_C( -1209.35), EASYSIMD_FLOAT64_C(  -478.11), EASYSIMD_FLOAT64_C(    69.42) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_subr_pd(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_subr_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d r = easysimd_mm512_subr_pd(a, b);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_subr_round_pd(EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT64_C(  -932.37), EASYSIMD_FLOAT64_C(  -267.61), EASYSIMD_FLOAT64_C(   820.46), EASYSIMD_FLOAT64_C(  -587.25),
        EASYSIMD_FLOAT64_C(  -187.76), EASYSIMD_FLOAT64_C(  -105.47), EASYSIMD_FLOAT64_C(  -688.53), EASYSIMD_FLOAT64_C(  -460.36) },
      { EASYSIMD_FLOAT64_C(  -875.27), EASYSIMD_FLOAT64_C(   333.70), EASYSIMD_FLOAT64_C(    55.46), EASYSIMD_FLOAT64_C(  -926.60),
        EASYSIMD_FLOAT64_C(  -186.82), EASYSIMD_FLOAT64_C(  -909.91), EASYSIMD_FLOAT64_C(   677.42), EASYSIMD_FLOAT64_C(   351.66) },
      { EASYSIMD_FLOAT64_C(    57.00), EASYSIMD_FLOAT64_C(   601.00), EASYSIMD_FLOAT64_C(  -765.00), EASYSIMD_FLOAT64_C(  -339.00),
        EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(  -804.00), EASYSIMD_FLOAT64_C(  1366.00), EASYSIMD_FLOAT64_C(   812.00) },
      { EASYSIMD_FLOAT64_C(    57.00), EASYSIMD_FLOAT64_C(   601.00), EASYSIMD_FLOAT64_C(  -765.00), EASYSIMD_FLOAT64_C(  -340.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -805.00), EASYSIMD_FLOAT64_C(  1365.00), EASYSIMD_FLOAT64_C(   812.00) },
      { EASYSIMD_FLOAT64_C(    58.00), EASYSIMD_FLOAT64_C(   602.00), EASYSIMD_FLOAT64_C(  -765.00), EASYSIMD_FLOAT64_C(  -339.00),
        EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(  -804.00), EASYSIMD_FLOAT64_C(  1366.00), EASYSIMD_FLOAT64_C(   813.00) },
      { EASYSIMD_FLOAT64_C(    57.00), EASYSIMD_FLOAT64_C(   601.00), EASYSIMD_FLOAT64_C(  -765.00), EASYSIMD_FLOAT64_C(  -339.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -804.00), EASYSIMD_FLOAT64_C(  1365.00), EASYSIMD_FLOAT64_C(   812.00) },
      { EASYSIMD_FLOAT64_C(    57.00), EASYSIMD_FLOAT64_C(   601.00), EASYSIMD_FLOAT64_C(  -765.00), EASYSIMD_FLOAT64_C(  -339.00),
        EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(  -804.00), EASYSIMD_FLOAT64_C(  1366.00), EASYSIMD_FLOAT64_C(   812.00) } },
    { { EASYSIMD_FLOAT64_C(    56.04), EASYSIMD_FLOAT64_C(  -506.23), EASYSIMD_FLOAT64_C(  -983.67), EASYSIMD_FLOAT64_C(  -209.29),
        EASYSIMD_FLOAT64_C(  -109.28), EASYSIMD_FLOAT64_C(   269.15), EASYSIMD_FLOAT64_C(   395.24), EASYSIMD_FLOAT64_C(  -906.35) },
      { EASYSIMD_FLOAT64_C(   587.40), EASYSIMD_FLOAT64_C(  -361.43), EASYSIMD_FLOAT64_C(   343.26), EASYSIMD_FLOAT64_C(  -478.22),
        EASYSIMD_FLOAT64_C(  -446.01), EASYSIMD_FLOAT64_C(   389.22), EASYSIMD_FLOAT64_C(   571.20), EASYSIMD_FLOAT64_C(  -378.38) },
      { EASYSIMD_FLOAT64_C(   531.00), EASYSIMD_FLOAT64_C(   145.00), EASYSIMD_FLOAT64_C(  1327.00), EASYSIMD_FLOAT64_C(  -269.00),
        EASYSIMD_FLOAT64_C(  -337.00), EASYSIMD_FLOAT64_C(   120.00), EASYSIMD_FLOAT64_C(   176.00), EASYSIMD_FLOAT64_C(   528.00) },
      { EASYSIMD_FLOAT64_C(   531.00), EASYSIMD_FLOAT64_C(   144.00), EASYSIMD_FLOAT64_C(  1326.00), EASYSIMD_FLOAT64_C(  -269.00),
        EASYSIMD_FLOAT64_C(  -337.00), EASYSIMD_FLOAT64_C(   120.00), EASYSIMD_FLOAT64_C(   175.00), EASYSIMD_FLOAT64_C(   527.00) },
      { EASYSIMD_FLOAT64_C(   532.00), EASYSIMD_FLOAT64_C(   145.00), EASYSIMD_FLOAT64_C(  1327.00), EASYSIMD_FLOAT64_C(  -268.00),
        EASYSIMD_FLOAT64_C(  -336.00), EASYSIMD_FLOAT64_C(   121.00), EASYSIMD_FLOAT64_C(   176.00), EASYSIMD_FLOAT64_C(   528.00) },
      { EASYSIMD_FLOAT64_C(   531.00), EASYSIMD_FLOAT64_C(   144.00), EASYSIMD_FLOAT64_C(  1326.00), EASYSIMD_FLOAT64_C(  -268.00),
        EASYSIMD_FLOAT64_C(  -336.00), EASYSIMD_FLOAT64_C(   120.00), EASYSIMD_FLOAT64_C(   175.00), EASYSIMD_FLOAT64_C(   527.00) },
      { EASYSIMD_FLOAT64_C(   531.00), EASYSIMD_FLOAT64_C(   145.00), EASYSIMD_FLOAT64_C(  1327.00), EASYSIMD_FLOAT64_C(  -269.00),
        EASYSIMD_FLOAT64_C(  -337.00), EASYSIMD_FLOAT64_C(   120.00), EASYSIMD_FLOAT64_C(   176.00), EASYSIMD_FLOAT64_C(   528.00) } },
    { { EASYSIMD_FLOAT64_C(  -878.39), EASYSIMD_FLOAT64_C(   391.66), EASYSIMD_FLOAT64_C(    34.37), EASYSIMD_FLOAT64_C(   -66.15),
        EASYSIMD_FLOAT64_C(  -713.81), EASYSIMD_FLOAT64_C(   345.84), EASYSIMD_FLOAT64_C(   473.49), EASYSIMD_FLOAT64_C(  -589.07) },
      { EASYSIMD_FLOAT64_C(  -320.46), EASYSIMD_FLOAT64_C(  -471.04), EASYSIMD_FLOAT64_C(  -515.67), EASYSIMD_FLOAT64_C(   492.72),
        EASYSIMD_FLOAT64_C(  -380.95), EASYSIMD_FLOAT64_C(  -838.26), EASYSIMD_FLOAT64_C(  -155.61), EASYSIMD_FLOAT64_C(   675.09) },
      { EASYSIMD_FLOAT64_C(   558.00), EASYSIMD_FLOAT64_C(  -863.00), EASYSIMD_FLOAT64_C(  -550.00), EASYSIMD_FLOAT64_C(   559.00),
        EASYSIMD_FLOAT64_C(   333.00), EASYSIMD_FLOAT64_C( -1184.00), EASYSIMD_FLOAT64_C(  -629.00), EASYSIMD_FLOAT64_C(  1264.00) },
      { EASYSIMD_FLOAT64_C(   557.00), EASYSIMD_FLOAT64_C(  -863.00), EASYSIMD_FLOAT64_C(  -551.00), EASYSIMD_FLOAT64_C(   558.00),
        EASYSIMD_FLOAT64_C(   332.00), EASYSIMD_FLOAT64_C( -1185.00), EASYSIMD_FLOAT64_C(  -630.00), EASYSIMD_FLOAT64_C(  1264.00) },
      { EASYSIMD_FLOAT64_C(   558.00), EASYSIMD_FLOAT64_C(  -862.00), EASYSIMD_FLOAT64_C(  -550.00), EASYSIMD_FLOAT64_C(   559.00),
        EASYSIMD_FLOAT64_C(   333.00), EASYSIMD_FLOAT64_C( -1184.00), EASYSIMD_FLOAT64_C(  -629.00), EASYSIMD_FLOAT64_C(  1265.00) },
      { EASYSIMD_FLOAT64_C(   557.00), EASYSIMD_FLOAT64_C(  -862.00), EASYSIMD_FLOAT64_C(  -550.00), EASYSIMD_FLOAT64_C(   558.00),
        EASYSIMD_FLOAT64_C(   332.00), EASYSIMD_FLOAT64_C( -1184.00), EASYSIMD_FLOAT64_C(  -629.00), EASYSIMD_FLOAT64_C(  1264.00) },
      { EASYSIMD_FLOAT64_C(   558.00), EASYSIMD_FLOAT64_C(  -863.00), EASYSIMD_FLOAT64_C(  -550.00), EASYSIMD_FLOAT64_C(   559.00),
        EASYSIMD_FLOAT64_C(   333.00), EASYSIMD_FLOAT64_C( -1184.00), EASYSIMD_FLOAT64_C(  -629.00), EASYSIMD_FLOAT64_C(  1264.00) } },
    { { EASYSIMD_FLOAT64_C(  -344.48), EASYSIMD_FLOAT64_C(  -139.29), EASYSIMD_FLOAT64_C(  -534.21), EASYSIMD_FLOAT64_C(   546.23),
        EASYSIMD_FLOAT64_C(  -870.14), EASYSIMD_FLOAT64_C(   861.03), EASYSIMD_FLOAT64_C(   639.89), EASYSIMD_FLOAT64_C(   717.26) },
      { EASYSIMD_FLOAT64_C(  -500.40), EASYSIMD_FLOAT64_C(   -16.85), EASYSIMD_FLOAT64_C(  -760.95), EASYSIMD_FLOAT64_C(    53.59),
        EASYSIMD_FLOAT64_C(  -627.63), EASYSIMD_FLOAT64_C(   810.25), EASYSIMD_FLOAT64_C(   675.21), EASYSIMD_FLOAT64_C(  -506.02) },
      { EASYSIMD_FLOAT64_C(  -156.00), EASYSIMD_FLOAT64_C(   122.00), EASYSIMD_FLOAT64_C(  -227.00), EASYSIMD_FLOAT64_C(  -493.00),
        EASYSIMD_FLOAT64_C(   243.00), EASYSIMD_FLOAT64_C(   -51.00), EASYSIMD_FLOAT64_C(    35.00), EASYSIMD_FLOAT64_C( -1223.00) },
      { EASYSIMD_FLOAT64_C(  -156.00), EASYSIMD_FLOAT64_C(   122.00), EASYSIMD_FLOAT64_C(  -227.00), EASYSIMD_FLOAT64_C(  -493.00),
        EASYSIMD_FLOAT64_C(   242.00), EASYSIMD_FLOAT64_C(   -51.00), EASYSIMD_FLOAT64_C(    35.00), EASYSIMD_FLOAT64_C( -1224.00) },
      { EASYSIMD_FLOAT64_C(  -155.00), EASYSIMD_FLOAT64_C(   123.00), EASYSIMD_FLOAT64_C(  -226.00), EASYSIMD_FLOAT64_C(  -492.00),
        EASYSIMD_FLOAT64_C(   243.00), EASYSIMD_FLOAT64_C(   -50.00), EASYSIMD_FLOAT64_C(    36.00), EASYSIMD_FLOAT64_C( -1223.00) },
      { EASYSIMD_FLOAT64_C(  -155.00), EASYSIMD_FLOAT64_C(   122.00), EASYSIMD_FLOAT64_C(  -226.00), EASYSIMD_FLOAT64_C(  -492.00),
        EASYSIMD_FLOAT64_C(   242.00), EASYSIMD_FLOAT64_C(   -50.00), EASYSIMD_FLOAT64_C(    35.00), EASYSIMD_FLOAT64_C( -1223.00) },
      { EASYSIMD_FLOAT64_C(  -156.00), EASYSIMD_FLOAT64_C(   122.00), EASYSIMD_FLOAT64_C(  -227.00), EASYSIMD_FLOAT64_C(  -493.00),
        EASYSIMD_FLOAT64_C(   243.00), EASYSIMD_FLOAT64_C(   -51.00), EASYSIMD_FLOAT64_C(    35.00), EASYSIMD_FLOAT64_C( -1223.00) } },
    { { EASYSIMD_FLOAT64_C(   201.91), EASYSIMD_FLOAT64_C(  -290.42), EASYSIMD_FLOAT64_C(   427.84), EASYSIMD_FLOAT64_C(   488.10),
        EASYSIMD_FLOAT64_C(  -944.58), EASYSIMD_FLOAT64_C(   -98.67), EASYSIMD_FLOAT64_C(   899.03), EASYSIMD_FLOAT64_C(  -265.04) },
      { EASYSIMD_FLOAT64_C(   430.29), EASYSIMD_FLOAT64_C(  -616.64), EASYSIMD_FLOAT64_C(  -772.32), EASYSIMD_FLOAT64_C(  -950.67),
        EASYSIMD_FLOAT64_C(  -454.90), EASYSIMD_FLOAT64_C(    72.07), EASYSIMD_FLOAT64_C(   724.42), EASYSIMD_FLOAT64_C(   200.62) },
      { EASYSIMD_FLOAT64_C(   228.00), EASYSIMD_FLOAT64_C(  -326.00), EASYSIMD_FLOAT64_C( -1200.00), EASYSIMD_FLOAT64_C( -1439.00),
        EASYSIMD_FLOAT64_C(   490.00), EASYSIMD_FLOAT64_C(   171.00), EASYSIMD_FLOAT64_C(  -175.00), EASYSIMD_FLOAT64_C(   466.00) },
      { EASYSIMD_FLOAT64_C(   228.00), EASYSIMD_FLOAT64_C(  -327.00), EASYSIMD_FLOAT64_C( -1201.00), EASYSIMD_FLOAT64_C( -1439.00),
        EASYSIMD_FLOAT64_C(   489.00), EASYSIMD_FLOAT64_C(   170.00), EASYSIMD_FLOAT64_C(  -175.00), EASYSIMD_FLOAT64_C(   465.00) },
      { EASYSIMD_FLOAT64_C(   229.00), EASYSIMD_FLOAT64_C(  -326.00), EASYSIMD_FLOAT64_C( -1200.00), EASYSIMD_FLOAT64_C( -1438.00),
        EASYSIMD_FLOAT64_C(   490.00), EASYSIMD_FLOAT64_C(   171.00), EASYSIMD_FLOAT64_C(  -174.00), EASYSIMD_FLOAT64_C(   466.00) },
      { EASYSIMD_FLOAT64_C(   228.00), EASYSIMD_FLOAT64_C(  -326.00), EASYSIMD_FLOAT64_C( -1200.00), EASYSIMD_FLOAT64_C( -1438.00),
        EASYSIMD_FLOAT64_C(   489.00), EASYSIMD_FLOAT64_C(   170.00), EASYSIMD_FLOAT64_C(  -174.00), EASYSIMD_FLOAT64_C(   465.00) },
      { EASYSIMD_FLOAT64_C(   228.00), EASYSIMD_FLOAT64_C(  -326.00), EASYSIMD_FLOAT64_C( -1200.00), EASYSIMD_FLOAT64_C( -1439.00),
        EASYSIMD_FLOAT64_C(   490.00), EASYSIMD_FLOAT64_C(   171.00), EASYSIMD_FLOAT64_C(  -175.00), EASYSIMD_FLOAT64_C(   466.00) } },
    { { EASYSIMD_FLOAT64_C(   932.78), EASYSIMD_FLOAT64_C(  -809.79), EASYSIMD_FLOAT64_C(  -253.15), EASYSIMD_FLOAT64_C(  -937.35),
        EASYSIMD_FLOAT64_C(  -948.76), EASYSIMD_FLOAT64_C(  -613.26), EASYSIMD_FLOAT64_C(   779.91), EASYSIMD_FLOAT64_C(  -449.16) },
      { EASYSIMD_FLOAT64_C(   369.88), EASYSIMD_FLOAT64_C(  -981.04), EASYSIMD_FLOAT64_C(   604.43), EASYSIMD_FLOAT64_C(   742.26),
        EASYSIMD_FLOAT64_C(   829.21), EASYSIMD_FLOAT64_C(   279.65), EASYSIMD_FLOAT64_C(  -763.76), EASYSIMD_FLOAT64_C(    31.12) },
      { EASYSIMD_FLOAT64_C(  -563.00), EASYSIMD_FLOAT64_C(  -171.00), EASYSIMD_FLOAT64_C(   858.00), EASYSIMD_FLOAT64_C(  1680.00),
        EASYSIMD_FLOAT64_C(  1778.00), EASYSIMD_FLOAT64_C(   893.00), EASYSIMD_FLOAT64_C( -1544.00), EASYSIMD_FLOAT64_C(   480.00) },
      { EASYSIMD_FLOAT64_C(  -563.00), EASYSIMD_FLOAT64_C(  -172.00), EASYSIMD_FLOAT64_C(   857.00), EASYSIMD_FLOAT64_C(  1679.00),
        EASYSIMD_FLOAT64_C(  1777.00), EASYSIMD_FLOAT64_C(   892.00), EASYSIMD_FLOAT64_C( -1544.00), EASYSIMD_FLOAT64_C(   480.00) },
      { EASYSIMD_FLOAT64_C(  -562.00), EASYSIMD_FLOAT64_C(  -171.00), EASYSIMD_FLOAT64_C(   858.00), EASYSIMD_FLOAT64_C(  1680.00),
        EASYSIMD_FLOAT64_C(  1778.00), EASYSIMD_FLOAT64_C(   893.00), EASYSIMD_FLOAT64_C( -1543.00), EASYSIMD_FLOAT64_C(   481.00) },
      { EASYSIMD_FLOAT64_C(  -562.00), EASYSIMD_FLOAT64_C(  -171.00), EASYSIMD_FLOAT64_C(   857.00), EASYSIMD_FLOAT64_C(  1679.00),
        EASYSIMD_FLOAT64_C(  1777.00), EASYSIMD_FLOAT64_C(   892.00), EASYSIMD_FLOAT64_C( -1543.00), EASYSIMD_FLOAT64_C(   480.00) },
      { EASYSIMD_FLOAT64_C(  -563.00), EASYSIMD_FLOAT64_C(  -171.00), EASYSIMD_FLOAT64_C(   858.00), EASYSIMD_FLOAT64_C(  1680.00),
        EASYSIMD_FLOAT64_C(  1778.00), EASYSIMD_FLOAT64_C(   893.00), EASYSIMD_FLOAT64_C( -1544.00), EASYSIMD_FLOAT64_C(   480.00) } },
    { { EASYSIMD_FLOAT64_C(   989.23), EASYSIMD_FLOAT64_C(   664.08), EASYSIMD_FLOAT64_C(  -480.78), EASYSIMD_FLOAT64_C(  -955.35),
        EASYSIMD_FLOAT64_C(  -434.59), EASYSIMD_FLOAT64_C(  -581.75), EASYSIMD_FLOAT64_C(  -220.39), EASYSIMD_FLOAT64_C(   995.70) },
      { EASYSIMD_FLOAT64_C(  -198.39), EASYSIMD_FLOAT64_C(     7.29), EASYSIMD_FLOAT64_C(  -954.97), EASYSIMD_FLOAT64_C(   346.71),
        EASYSIMD_FLOAT64_C(  -920.64), EASYSIMD_FLOAT64_C(   769.45), EASYSIMD_FLOAT64_C(  -452.68), EASYSIMD_FLOAT64_C(  -987.85) },
      { EASYSIMD_FLOAT64_C( -1188.00), EASYSIMD_FLOAT64_C(  -657.00), EASYSIMD_FLOAT64_C(  -474.00), EASYSIMD_FLOAT64_C(  1302.00),
        EASYSIMD_FLOAT64_C(  -486.00), EASYSIMD_FLOAT64_C(  1351.00), EASYSIMD_FLOAT64_C(  -232.00), EASYSIMD_FLOAT64_C( -1984.00) },
      { EASYSIMD_FLOAT64_C( -1188.00), EASYSIMD_FLOAT64_C(  -657.00), EASYSIMD_FLOAT64_C(  -475.00), EASYSIMD_FLOAT64_C(  1302.00),
        EASYSIMD_FLOAT64_C(  -487.00), EASYSIMD_FLOAT64_C(  1351.00), EASYSIMD_FLOAT64_C(  -233.00), EASYSIMD_FLOAT64_C( -1984.00) },
      { EASYSIMD_FLOAT64_C( -1187.00), EASYSIMD_FLOAT64_C(  -656.00), EASYSIMD_FLOAT64_C(  -474.00), EASYSIMD_FLOAT64_C(  1303.00),
        EASYSIMD_FLOAT64_C(  -486.00), EASYSIMD_FLOAT64_C(  1352.00), EASYSIMD_FLOAT64_C(  -232.00), EASYSIMD_FLOAT64_C( -1983.00) },
      { EASYSIMD_FLOAT64_C( -1187.00), EASYSIMD_FLOAT64_C(  -656.00), EASYSIMD_FLOAT64_C(  -474.00), EASYSIMD_FLOAT64_C(  1302.00),
        EASYSIMD_FLOAT64_C(  -486.00), EASYSIMD_FLOAT64_C(  1351.00), EASYSIMD_FLOAT64_C(  -232.00), EASYSIMD_FLOAT64_C( -1983.00) },
      { EASYSIMD_FLOAT64_C( -1188.00), EASYSIMD_FLOAT64_C(  -657.00), EASYSIMD_FLOAT64_C(  -474.00), EASYSIMD_FLOAT64_C(  1302.00),
        EASYSIMD_FLOAT64_C(  -486.00), EASYSIMD_FLOAT64_C(  1351.00), EASYSIMD_FLOAT64_C(  -232.00), EASYSIMD_FLOAT64_C( -1984.00) } },
    { { EASYSIMD_FLOAT64_C(   959.66), EASYSIMD_FLOAT64_C(   294.17), EASYSIMD_FLOAT64_C(  -925.21), EASYSIMD_FLOAT64_C(  -989.10),
        EASYSIMD_FLOAT64_C(   680.91), EASYSIMD_FLOAT64_C(   854.71), EASYSIMD_FLOAT64_C(  -438.26), EASYSIMD_FLOAT64_C(    50.79) },
      { EASYSIMD_FLOAT64_C(   873.66), EASYSIMD_FLOAT64_C(  -833.83), EASYSIMD_FLOAT64_C(  -206.95), EASYSIMD_FLOAT64_C(   702.87),
        EASYSIMD_FLOAT64_C(   445.82), EASYSIMD_FLOAT64_C(    29.29), EASYSIMD_FLOAT64_C(  -266.02), EASYSIMD_FLOAT64_C(   435.05) },
      { EASYSIMD_FLOAT64_C(   -86.00), EASYSIMD_FLOAT64_C( -1128.00), EASYSIMD_FLOAT64_C(   718.00), EASYSIMD_FLOAT64_C(  1692.00),
        EASYSIMD_FLOAT64_C(  -235.00), EASYSIMD_FLOAT64_C(  -825.00), EASYSIMD_FLOAT64_C(   172.00), EASYSIMD_FLOAT64_C(   384.00) },
      { EASYSIMD_FLOAT64_C(   -86.00), EASYSIMD_FLOAT64_C( -1128.00), EASYSIMD_FLOAT64_C(   718.00), EASYSIMD_FLOAT64_C(  1691.00),
        EASYSIMD_FLOAT64_C(  -236.00), EASYSIMD_FLOAT64_C(  -826.00), EASYSIMD_FLOAT64_C(   172.00), EASYSIMD_FLOAT64_C(   384.00) },
      { EASYSIMD_FLOAT64_C(   -86.00), EASYSIMD_FLOAT64_C( -1128.00), EASYSIMD_FLOAT64_C(   719.00), EASYSIMD_FLOAT64_C(  1692.00),
        EASYSIMD_FLOAT64_C(  -235.00), EASYSIMD_FLOAT64_C(  -825.00), EASYSIMD_FLOAT64_C(   173.00), EASYSIMD_FLOAT64_C(   385.00) },
      { EASYSIMD_FLOAT64_C(   -86.00), EASYSIMD_FLOAT64_C( -1128.00), EASYSIMD_FLOAT64_C(   718.00), EASYSIMD_FLOAT64_C(  1691.00),
        EASYSIMD_FLOAT64_C(  -235.00), EASYSIMD_FLOAT64_C(  -825.00), EASYSIMD_FLOAT64_C(   172.00), EASYSIMD_FLOAT64_C(   384.00) },
      { EASYSIMD_FLOAT64_C(   -86.00), EASYSIMD_FLOAT64_C( -1128.00), EASYSIMD_FLOAT64_C(   718.00), EASYSIMD_FLOAT64_C(  1692.00),
        EASYSIMD_FLOAT64_C(  -235.00), EASYSIMD_FLOAT64_C(  -825.00), EASYSIMD_FLOAT64_C(   172.00), EASYSIMD_FLOAT64_C(   384.00) } }
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

    r = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512d_close(r, nearest_inf, 1);

    r = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512d_close(r, neg_inf, 1);

    r = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512d_close(r, pos_inf, 1);

    r = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512d_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_subr_round_pd");
    easysimd_assert_m512d_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512d nearest_inf = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512d neg_inf = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512d pos_inf = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512d zero = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512d direction = easysimd_mm512_subr_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

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
test_easysimd_mm512_subsetb_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
    const easysimd__mmask16 borrow;
  } test_vec[] = {
    { { -INT32_C(   138256828), -INT32_C(   799180572), -INT32_C(  1908113826),  INT32_C(  2088658736),  INT32_C(  1249539352),  INT32_C(  1548337146),  INT32_C(   292762757), -INT32_C(   294187606),
         INT32_C(  1357265260),  INT32_C(   270549937), -INT32_C(   241277759), -INT32_C(   462545716), -INT32_C(  1943017326),  INT32_C(   770209960),  INT32_C(  1446927276), -INT32_C(   716851863) },
      { -INT32_C(  1608176657),  INT32_C(   800081262),  INT32_C(  1981894314), -INT32_C(    44396693),  INT32_C(   529107319), -INT32_C(  1387499007),  INT32_C(   906267341),  INT32_C(   789268800),
        -INT32_C(   489738124),  INT32_C(   538017910),  INT32_C(   966144718),  INT32_C(   959901889),  INT32_C(  2069414010), -INT32_C(    14047950),  INT32_C(  1865821487), -INT32_C(   358727306) },
      {  INT32_C(  1469919829), -INT32_C(  1599261834),  INT32_C(   404959156),  INT32_C(  2133055429),  INT32_C(   720432033), -INT32_C(  1359131143), -INT32_C(   613504584), -INT32_C(  1083456406),
         INT32_C(  1847003384), -INT32_C(   267467973), -INT32_C(  1207422477), -INT32_C(  1422447605),  INT32_C(   282535960),  INT32_C(   784257910), -INT32_C(   418894211), -INT32_C(   358124557) },
      UINT16_C(58216) },
    { { -INT32_C(   389255566), -INT32_C(  1140269586), -INT32_C(   772366832),  INT32_C(   134884494),  INT32_C(   511992812),  INT32_C(   924757256),  INT32_C(  1353143514),  INT32_C(   121259413),
        -INT32_C(  1578170701), -INT32_C(   211879965),  INT32_C(   616912021),  INT32_C(  1831653248),  INT32_C(   982233138),  INT32_C(   930261341), -INT32_C(  1819862531),  INT32_C(   312131934) },
      { -INT32_C(  1414296889),  INT32_C(   396235137), -INT32_C(   432315547),  INT32_C(  1699964723),  INT32_C(  1973476888), -INT32_C(  2052255096), -INT32_C(  1994902485), -INT32_C(  1113869322),
        -INT32_C(  1100460227), -INT32_C(   959117472), -INT32_C(  1649667990), -INT32_C(  1878786184),  INT32_C(  1694868445), -INT32_C(   521424203), -INT32_C(   580320281), -INT32_C(   208009802) },
      {  INT32_C(  1025041323), -INT32_C(  1536504723), -INT32_C(   340051285), -INT32_C(  1565080229), -INT32_C(  1461484076), -INT32_C(  1317954944), -INT32_C(   946921297),  INT32_C(  1235128735),
        -INT32_C(   477710474),  INT32_C(   747237507), -INT32_C(  2028387285), -INT32_C(   584527864), -INT32_C(   712635307),  INT32_C(  1451685544), -INT32_C(  1239542250),  INT32_C(   520141736) },
      UINT16_C(65020) },
    { { -INT32_C(  1263402156),  INT32_C(  1954187018),  INT32_C(   269624984),  INT32_C(    60822821),  INT32_C(  1835574712),  INT32_C(  1062032216),  INT32_C(   219985495),  INT32_C(   251770555),
        -INT32_C(  1010519111), -INT32_C(   768065990), -INT32_C(  1964881307),  INT32_C(   395149919), -INT32_C(  2138769880), -INT32_C(  1598041783),  INT32_C(  1135532935),  INT32_C(  1263710097) },
      { -INT32_C(  1676798366), -INT32_C(  1167112619), -INT32_C(   280735344), -INT32_C(    66661676),  INT32_C(   293374664), -INT32_C(   474858661), -INT32_C(  1490657258),  INT32_C(  1911715855),
        -INT32_C(   468909681), -INT32_C(   660702137), -INT32_C(  1563958578),  INT32_C(  2090782388), -INT32_C(  1265755560),  INT32_C(  1805074517), -INT32_C(  1357726304), -INT32_C(  1004534475) },
      {  INT32_C(   413396210), -INT32_C(  1173667659),  INT32_C(   550360328),  INT32_C(   127484497),  INT32_C(  1542200048),  INT32_C(  1536890877),  INT32_C(  1710642753), -INT32_C(  1659945300),
        -INT32_C(   541609430), -INT32_C(   107363853), -INT32_C(   400922729), -INT32_C(  1695632469), -INT32_C(   873014320),  INT32_C(   891850996), -INT32_C(  1801708057), -INT32_C(  2026722724) },
      UINT16_C(57326) },
    { {  INT32_C(  1302867206),  INT32_C(  2015708842), -INT32_C(   585437911),  INT32_C(   324712635),  INT32_C(   667412690), -INT32_C(   929931736),  INT32_C(  1366795291), -INT32_C(  1357539415),
         INT32_C(  1862123204),  INT32_C(   770056708), -INT32_C(   905248753), -INT32_C(  1965136456),  INT32_C(  1974576461),  INT32_C(   524108548), -INT32_C(  1854884632),  INT32_C(   255952459) },
      {  INT32_C(  1216233028),  INT32_C(  1886807136),  INT32_C(   490373477),  INT32_C(   866654438), -INT32_C(  1029154370), -INT32_C(  2048793187), -INT32_C(   468299111),  INT32_C(   485709784),
        -INT32_C(   161189483),  INT32_C(   996596438),  INT32_C(  1096327259),  INT32_C(  2004091065), -INT32_C(   147252134), -INT32_C(  1703142911),  INT32_C(  1149145708),  INT32_C(  2137027306) },
      {  INT32_C(    86634178),  INT32_C(   128901706), -INT32_C(  1075811388), -INT32_C(   541941803),  INT32_C(  1696567060),  INT32_C(  1118861451),  INT32_C(  1835094402), -INT32_C(  1843249199),
         INT32_C(  2023312687), -INT32_C(   226539730), -INT32_C(  2001576012),  INT32_C(   325739775),  INT32_C(  2121828595), -INT32_C(  2067715837),  INT32_C(  1290936956), -INT32_C(  1881074847) },
      UINT16_C(45912) },
    { { -INT32_C(  1183463965), -INT32_C(    67839073),  INT32_C(   893144444), -INT32_C(  1481854643), -INT32_C(   811670067),  INT32_C(  1818827519), -INT32_C(  1750013779),  INT32_C(  1024856410),
         INT32_C(  1979157718), -INT32_C(   479138969), -INT32_C(  2045203144),  INT32_C(   724419678), -INT32_C(  1443181399), -INT32_C(  1827314458), -INT32_C(  1507146420), -INT32_C(  1394392618) },
      {  INT32_C(   874699469), -INT32_C(    31943994), -INT32_C(  1652281281), -INT32_C(  1630948619),  INT32_C(  1682424702),  INT32_C(  1945656359), -INT32_C(   132570590),  INT32_C(   816184675),
        -INT32_C(  1654274089), -INT32_C(  1717797543), -INT32_C(  1573510995),  INT32_C(  1346502610), -INT32_C(   357201725),  INT32_C(   123645413),  INT32_C(   872380367),  INT32_C(  1281598580) },
      { -INT32_C(  2058163434), -INT32_C(    35895079), -INT32_C(  1749541571),  INT32_C(   149093976),  INT32_C(  1800872527), -INT32_C(   126828840), -INT32_C(  1617443189),  INT32_C(   208671735),
        -INT32_C(   661535489),  INT32_C(  1238658574), -INT32_C(   471692149), -INT32_C(   622082932), -INT32_C(  1085979674), -INT32_C(  1950959871),  INT32_C(  1915440509),  INT32_C(  1618976098) },
      UINT16_C( 7526) },
    { { -INT32_C(   974534549), -INT32_C(   211909307),  INT32_C(  2023068838),  INT32_C(  1472845460),  INT32_C(  1145142879), -INT32_C(    95707349), -INT32_C(  1959966185),  INT32_C(  1524077039),
        -INT32_C(  1625308839), -INT32_C(   325943994), -INT32_C(  1503320302),  INT32_C(  1576873726), -INT32_C(   677298516), -INT32_C(   170791714),  INT32_C(   629210934), -INT32_C(   377464688) },
      {  INT32_C(  1602789145),  INT32_C(   793516573),  INT32_C(  1087746370), -INT32_C(  1952591137), -INT32_C(   295551472),  INT32_C(  1625502506), -INT32_C(  1031379918), -INT32_C(   710146372),
        -INT32_C(  1036766043), -INT32_C(  1846443953),  INT32_C(   282183217), -INT32_C(  1449431400), -INT32_C(   694682196),  INT32_C(  1664514865), -INT32_C(  1691960097),  INT32_C(  1752224450) },
      {  INT32_C(  1717643602), -INT32_C(  1005425880),  INT32_C(   935322468), -INT32_C(   869530699),  INT32_C(  1440694351), -INT32_C(  1721209855), -INT32_C(   928586267), -INT32_C(  2060743885),
        -INT32_C(   588542796),  INT32_C(  1520499959), -INT32_C(  1785503519), -INT32_C(  1268662170),  INT32_C(    17383680), -INT32_C(  1835306579), -INT32_C(  1973796265), -INT32_C(  2129689138) },
      UINT16_C(18904) },
    { {  INT32_C(  1428857862),  INT32_C(  1457921061),  INT32_C(  2070394850), -INT32_C(   786169307),  INT32_C(   833075968),  INT32_C(   362077750),  INT32_C(  1571863194), -INT32_C(  1815797620),
        -INT32_C(   353833019), -INT32_C(   297742581), -INT32_C(  1402361978), -INT32_C(  1434612310),  INT32_C(  2145068360), -INT32_C(  1651216637), -INT32_C(  1225112278),  INT32_C(   709476197) },
      { -INT32_C(  1173081425), -INT32_C(  2019011327), -INT32_C(  1506602500), -INT32_C(   414142050), -INT32_C(   647615530), -INT32_C(   982058341), -INT32_C(  1518636737), -INT32_C(   540031696),
        -INT32_C(   124066569),  INT32_C(   897532473), -INT32_C(   220417196),  INT32_C(   987311204), -INT32_C(   216842408),  INT32_C(  2042136890),  INT32_C(   723399674), -INT32_C(   284496136) },
      { -INT32_C(  1693028009), -INT32_C(   818034908), -INT32_C(   717969946), -INT32_C(   372027257),  INT32_C(  1480691498),  INT32_C(  1344136091), -INT32_C(  1204467365), -INT32_C(  1275765924),
        -INT32_C(   229766450), -INT32_C(  1195275054), -INT32_C(  1181944782),  INT32_C(  1873043782), -INT32_C(  1933056528),  INT32_C(   601613769), -INT32_C(  1948511952),  INT32_C(   993972333) },
      UINT16_C(38399) },
    { {  INT32_C(   199730386),  INT32_C(   994076647),  INT32_C(  2116885530), -INT32_C(  1581775031), -INT32_C(  2137732282),  INT32_C(  1341803604),  INT32_C(  2021267583), -INT32_C(   664304634),
         INT32_C(   283332393), -INT32_C(   800381770), -INT32_C(  1991346112), -INT32_C(   970324353),  INT32_C(   625393361), -INT32_C(  1972092918),  INT32_C(  1594027609), -INT32_C(  1674024589) },
      {  INT32_C(  1856773048), -INT32_C(  2143357121), -INT32_C(   284586897),  INT32_C(  1672819858), -INT32_C(    41288718), -INT32_C(  1786249924),  INT32_C(  1609861612), -INT32_C(  1409602318),
        -INT32_C(  2028361912),  INT32_C(   218585246),  INT32_C(  2013008356),  INT32_C(   937079109), -INT32_C(   382442579),  INT32_C(  1300151137),  INT32_C(   934048324), -INT32_C(   404576353) },
      { -INT32_C(  1657042662), -INT32_C(  1157533528), -INT32_C(  1893494869),  INT32_C(  1040372407), -INT32_C(  2096443564), -INT32_C(  1166913768),  INT32_C(   411405971),  INT32_C(   745297684),
        -INT32_C(  1983272991), -INT32_C(  1018967016),  INT32_C(   290612828), -INT32_C(  1907403462),  INT32_C(  1007835940),  INT32_C(  1022723241),  INT32_C(   659979285), -INT32_C(  1269448236) },
      UINT16_C(37175) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 borrow = 0;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_subsetb_epi32(a, b, &borrow);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_subsetb_epi32");
    easysimd_assert_m512i_i32(r, ==, easysimd_mm512_loadu_epi32(test_vec[i].r));
    easysimd_assert_equal_mmask16(borrow, test_vec[i].borrow);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 borrow = 0;
    easysimd__m512i r = easysimd_mm512_subsetb_epi32(a, b, &borrow);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, borrow, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_subrsetb_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
    const easysimd__mmask16 borrow;
  } test_vec[] = {
    { { -INT32_C(   328271026),  INT32_C(   955872851), -INT32_C(   860883321),  INT32_C(  1409517991),  INT32_C(  1312635117),  INT32_C(   949730291), -INT32_C(   848345298),  INT32_C(  1018450414),
        -INT32_C(  1607982516),  INT32_C(   534258072), -INT32_C(  1075083497), -INT32_C(    49025264),  INT32_C(   441143591),  INT32_C(   978511372),  INT32_C(   453493037),  INT32_C(  1599585042) },
      {  INT32_C(  1996455902), -INT32_C(  1198073951),  INT32_C(  1853325662), -INT32_C(  1754494096), -INT32_C(   390940708), -INT32_C(   886897506), -INT32_C(   655938874), -INT32_C(  1003012378),
         INT32_C(  1597650622),  INT32_C(  1796722701), -INT32_C(  1025863854), -INT32_C(   161855974), -INT32_C(  1679946499), -INT32_C(   697892592),  INT32_C(   313478700),  INT32_C(  1255597708) },
      { -INT32_C(  1970240368),  INT32_C(  2141020494), -INT32_C(  1580758313),  INT32_C(  1130955209), -INT32_C(  1703575825), -INT32_C(  1836627797),  INT32_C(   192406424), -INT32_C(  2021462792),
        -INT32_C(  1089334158),  INT32_C(  1262464629),  INT32_C(    49219643), -INT32_C(   112830710), -INT32_C(  2121090090), -INT32_C(  1676403964), -INT32_C(   140014337), -INT32_C(   343987334) },
      UINT16_C(51461) },
    { {  INT32_C(   715722781),  INT32_C(   865518049),  INT32_C(  1794469968), -INT32_C(  1285533770),  INT32_C(  1817132635),  INT32_C(  1816311360), -INT32_C(  1870728956), -INT32_C(   170175272),
         INT32_C(  1159693412), -INT32_C(  1787251387), -INT32_C(   604017115),  INT32_C(   428826557), -INT32_C(   561652066), -INT32_C(  1739929708), -INT32_C(  1859598151), -INT32_C(  2138701028) },
      { -INT32_C(   842619512), -INT32_C(  2140979621),  INT32_C(  1784439468),  INT32_C(  1619258306),  INT32_C(  1564346569), -INT32_C(  1997174577),  INT32_C(  1813585232), -INT32_C(  1444110559),
        -INT32_C(  1602833851), -INT32_C(  1658791439), -INT32_C(    33063876),  INT32_C(   811502183),  INT32_C(  1653447826),  INT32_C(  1961525796), -INT32_C(  1042283360), -INT32_C(   395653981) },
      { -INT32_C(  1558342293),  INT32_C(  1288469626), -INT32_C(    10030500), -INT32_C(  1390175220), -INT32_C(   252786066),  INT32_C(   481481359), -INT32_C(   610653108), -INT32_C(  1273935287),
         INT32_C(  1532440033),  INT32_C(   128459948),  INT32_C(   570953239),  INT32_C(   382675626), -INT32_C(  2079867404), -INT32_C(   593511792),  INT32_C(   817314791),  INT32_C(  1743047047) },
      UINT16_C(12508) },
    { {  INT32_C(  1888018559), -INT32_C(   166876742), -INT32_C(  1913383643),  INT32_C(   851268255),  INT32_C(   311708654),  INT32_C(  1837530829),  INT32_C(   640575106), -INT32_C(  1324443598),
         INT32_C(   857839481),  INT32_C(  1697197888), -INT32_C(   470672060),  INT32_C(  1561702511), -INT32_C(   932206085), -INT32_C(  1439238872), -INT32_C(  1915722661),  INT32_C(  1983897597) },
      { -INT32_C(  1230413706), -INT32_C(   753151345),  INT32_C(  1589055215), -INT32_C(  1178874690), -INT32_C(  1635702154),  INT32_C(  2051585823),  INT32_C(   419895580),  INT32_C(  1854883576),
         INT32_C(   908343463), -INT32_C(   116769014),  INT32_C(   190300493),  INT32_C(    63181453),  INT32_C(  1537295932),  INT32_C(   433449725), -INT32_C(    80552957), -INT32_C(   899038941) },
      {  INT32_C(  1176535031), -INT32_C(   586274603), -INT32_C(   792528438), -INT32_C(  2030142945), -INT32_C(  1947410808),  INT32_C(   214054994), -INT32_C(   220679526), -INT32_C(  1115640122),
         INT32_C(    50503982), -INT32_C(  1813966902),  INT32_C(   660972553), -INT32_C(  1498521058), -INT32_C(  1825465279),  INT32_C(  1872688597),  INT32_C(  1835169704),  INT32_C(  1412030758) },
      UINT16_C(15558) },
    { {  INT32_C(    50367993),  INT32_C(   452725452),  INT32_C(  1495618507), -INT32_C(  1587746203),  INT32_C(   771554864), -INT32_C(   347614744), -INT32_C(   773424722),  INT32_C(   882593595),
        -INT32_C(  1472685092),  INT32_C(  1908552870), -INT32_C(   305469304),  INT32_C(    42936274),  INT32_C(   221219621),  INT32_C(   200832605),  INT32_C(   735895280),  INT32_C(   157251373) },
      { -INT32_C(  1196321006),  INT32_C(  1412068556),  INT32_C(   776074332),  INT32_C(  1076940827), -INT32_C(  1186111652), -INT32_C(   960215594),  INT32_C(  1358012451),  INT32_C(   693719063),
        -INT32_C(  1260254232), -INT32_C(   620229505),  INT32_C(   470370560),  INT32_C(  1985755418),  INT32_C(  1848617624),  INT32_C(   322237423), -INT32_C(  1436342893),  INT32_C(  1590934902) },
      { -INT32_C(  1246688999),  INT32_C(   959343104), -INT32_C(   719544175), -INT32_C(  1630280266), -INT32_C(  1957666516), -INT32_C(   612600850),  INT32_C(  2131437173), -INT32_C(   188874532),
         INT32_C(   212430860),  INT32_C(  1766184921),  INT32_C(   775839864),  INT32_C(  1942819144),  INT32_C(  1627398003),  INT32_C(   121404818),  INT32_C(  2122729123),  INT32_C(  1433683529) },
      UINT16_C( 1260) },
    { {  INT32_C(  1192408520), -INT32_C(  1037952319),  INT32_C(  2111712099), -INT32_C(    51168668), -INT32_C(   731176220), -INT32_C(  1461215467),  INT32_C(   995248837), -INT32_C(   812046841),
        -INT32_C(  1659458597),  INT32_C(   677329093), -INT32_C(   945406621),  INT32_C(  1556388215), -INT32_C(   785371204), -INT32_C(  1820715058),  INT32_C(  1775160417), -INT32_C(   835164174) },
      { -INT32_C(   680833262), -INT32_C(   352269689),  INT32_C(  2125637127), -INT32_C(    69568961),  INT32_C(  1959594661), -INT32_C(  2096675039),  INT32_C(   116184595),  INT32_C(  1356080189),
        -INT32_C(    81313933),  INT32_C(   283518729),  INT32_C(   227449037), -INT32_C(  1274517233), -INT32_C(  1792486029),  INT32_C(   806891548),  INT32_C(  1127613446), -INT32_C(  1668085208) },
      { -INT32_C(  1873241782),  INT32_C(   685682630),  INT32_C(    13925028), -INT32_C(    18400293), -INT32_C(  1604196415), -INT32_C(   635459572), -INT32_C(   879064242), -INT32_C(  2126840266),
         INT32_C(  1578144664), -INT32_C(   393810364),  INT32_C(  1172855658),  INT32_C(  1464061848), -INT32_C(  1007114825), -INT32_C(  1667360690), -INT32_C(   647546971), -INT32_C(   832921034) },
      UINT16_C(63224) },
    { {  INT32_C(  1385675593), -INT32_C(  1335722526),  INT32_C(   616427541), -INT32_C(   841366183), -INT32_C(  1218313829),  INT32_C(   937916977), -INT32_C(  1501880962),  INT32_C(  1883377191),
        -INT32_C(  1396516407),  INT32_C(  1817977942),  INT32_C(  1854937365),  INT32_C(  2033936862), -INT32_C(  1674470037), -INT32_C(  1781262313),  INT32_C(  1564233526),  INT32_C(   667844189) },
      { -INT32_C(  1361866664), -INT32_C(   904253515),  INT32_C(   641248072),  INT32_C(  2141221908),  INT32_C(   689754385),  INT32_C(   532607209), -INT32_C(  1669465537), -INT32_C(   775730311),
        -INT32_C(  1870686501),  INT32_C(   224041669),  INT32_C(  1496617797),  INT32_C(   416928775), -INT32_C(  1908279899),  INT32_C(   615383269),  INT32_C(  1941974010),  INT32_C(  1363444854) },
      {  INT32_C(  1547425039),  INT32_C(   431469011),  INT32_C(    24820531), -INT32_C(  1312379205),  INT32_C(  1908068214), -INT32_C(   405309768), -INT32_C(   167584575),  INT32_C(  1635859794),
        -INT32_C(   474170094), -INT32_C(  1593936273), -INT32_C(   358319568), -INT32_C(  1617008087), -INT32_C(   233809862), -INT32_C(  1898321714),  INT32_C(   377740484),  INT32_C(   695600665) },
      UINT16_C(16248) },
    { { -INT32_C(   522009574), -INT32_C(  1544733602), -INT32_C(   688119345), -INT32_C(  1695558155), -INT32_C(  1356255030),  INT32_C(   735303472),  INT32_C(  2023658498),  INT32_C(   852091672),
         INT32_C(    85109671), -INT32_C(  1213726488),  INT32_C(   395224097),  INT32_C(  1152482682), -INT32_C(   554378579), -INT32_C(  1291204687),  INT32_C(  1932240731),  INT32_C(   833025418) },
      { -INT32_C(  2009679712), -INT32_C(   633282888), -INT32_C(    34484605), -INT32_C(   129916341),  INT32_C(   785790333),  INT32_C(  1491263485),  INT32_C(   298585479), -INT32_C(  1555860990),
        -INT32_C(   483690198), -INT32_C(   608343208), -INT32_C(  2066174407), -INT32_C(   847439536),  INT32_C(  1291604815), -INT32_C(  1163600333), -INT32_C(   288657173),  INT32_C(   227610338) },
      { -INT32_C(  1487670138),  INT32_C(   911450714),  INT32_C(   653634740),  INT32_C(  1565641814),  INT32_C(  2142045363),  INT32_C(   755960013), -INT32_C(  1725073019),  INT32_C(  1887014634),
        -INT32_C(   568799869),  INT32_C(   605383280),  INT32_C(  1833568792), -INT32_C(  1999922218),  INT32_C(  1845983394),  INT32_C(   127604354),  INT32_C(  2074069392), -INT32_C(   605415080) },
      UINT16_C(36945) },
    { { -INT32_C(   521094008),  INT32_C(  1639689512), -INT32_C(  1410952357), -INT32_C(    75930708), -INT32_C(   381192778),  INT32_C(  1050930259),  INT32_C(  1059876700),  INT32_C(    88915325),
        -INT32_C(  1562035078),  INT32_C(  1141088489), -INT32_C(   521148109),  INT32_C(    64710732),  INT32_C(   820782045),  INT32_C(  1802473487),  INT32_C(  2091555839), -INT32_C(   746391975) },
      {  INT32_C(   460678962),  INT32_C(   996112392), -INT32_C(  1373941918), -INT32_C(  1783498825),  INT32_C(   700816922),  INT32_C(   764687406),  INT32_C(   698957776),  INT32_C(  1761356597),
        -INT32_C(  1685884525),  INT32_C(  1272374249), -INT32_C(   369429966),  INT32_C(    58633193),  INT32_C(  1999389769),  INT32_C(  1218756728),  INT32_C(   896618239),  INT32_C(   211643769) },
      {  INT32_C(   981772970), -INT32_C(   643577120),  INT32_C(    37010439), -INT32_C(  1707568117),  INT32_C(  1082009700), -INT32_C(   286242853), -INT32_C(   360918924),  INT32_C(  1672441272),
        -INT32_C(   123849447),  INT32_C(   131285760),  INT32_C(   151718143), -INT32_C(     6077539),  INT32_C(  1178607724), -INT32_C(   583716759), -INT32_C(  1194937600),  INT32_C(   958035744) },
      UINT16_C(59771) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 borrow = 0;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_subrsetb_epi32(a, b, &borrow);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_subrsetb_epi32");
    easysimd_assert_m512i_i32(r, ==, easysimd_mm512_loadu_epi32(test_vec[i].r));
    easysimd_assert_equal_mmask16(borrow, test_vec[i].borrow);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 borrow = 0;
    easysimd__m512i r = easysimd_mm512_subrsetb_epi32(a, b, &borrow);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, borrow, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sub_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sub_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sub_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sub_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sub_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sub_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sub_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sub_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_sub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_sub_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sub_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sub_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sub_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sub_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sub_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sub_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sub_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sub_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_sub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_sub_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sub_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sub_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_sub_epi8)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sub_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sub_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_sub_epi16)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sub_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sub_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_sub_epi32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sub_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sub_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_sub_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sub_round_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_sub_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_sub_round_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_sub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_sub_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_subr_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_subr_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_subr_round_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_subr_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_subr_round_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_subsetb_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_subrsetb_epi32)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
