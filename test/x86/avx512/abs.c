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

#define EASYSIMD_TEST_X86_AVX512_INSN abs

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/abs.h>

static int
test_easysimd_mm_mask_abs_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int8_t src[16];
    const easysimd__mmask16 k;
    const int8_t a[16];
    const int8_t r[16];
  } test_vec[] = {
    { {  INT8_C( 115), -INT8_C(  14), -INT8_C(  39), -INT8_C(  78),  INT8_C(  20), -INT8_C( 119),  INT8_C(  90), -INT8_C(  11),
         INT8_C(  38),  INT8_C(  60),  INT8_C(  35),  INT8_C(  25), -INT8_C(  81),  INT8_C(  45),  INT8_C(  59), -INT8_C(  89) },
      UINT16_C(61702),
      { -INT8_C( 102),  INT8_C(  16), -INT8_C( 119), -INT8_C(   7),  INT8_C(  26),  INT8_C( 102),  INT8_C(  48), -INT8_C( 119),
         INT8_C(  49), -INT8_C(  12), -INT8_C(  14),  INT8_C( 101), -INT8_C(  13), -INT8_C( 109),  INT8_C(  16), -INT8_C(  69) },
      {  INT8_C( 115),  INT8_C(  16),  INT8_C( 119), -INT8_C(  78),  INT8_C(  20), -INT8_C( 119),  INT8_C(  90), -INT8_C(  11),
         INT8_C(  49),  INT8_C(  60),  INT8_C(  35),  INT8_C(  25),  INT8_C(  13),  INT8_C( 109),  INT8_C(  16),  INT8_C(  69) } },
    { {  INT8_C( 105), -INT8_C( 125),  INT8_C(  57), -INT8_C(  42),  INT8_C( 112),  INT8_C(   9),  INT8_C(  90),  INT8_C(  60),
        -INT8_C(  44), -INT8_C(  15),  INT8_C( 105), -INT8_C(  88), -INT8_C(  21),  INT8_C(  36), -INT8_C(  54), -INT8_C(   5) },
      UINT16_C(16140),
      {  INT8_C(  85), -INT8_C(  80),  INT8_C(  35), -INT8_C( 120), -INT8_C(   2), -INT8_C( 126),  INT8_C(   4),  INT8_C(   2),
         INT8_C(  53), -INT8_C(  97), -INT8_C(  98), -INT8_C(  97), -INT8_C(   5),      INT8_MAX,  INT8_C(  32),  INT8_C(   2) },
      {  INT8_C( 105), -INT8_C( 125),  INT8_C(  35),  INT8_C( 120),  INT8_C( 112),  INT8_C(   9),  INT8_C(  90),  INT8_C(  60),
         INT8_C(  53),  INT8_C(  97),  INT8_C(  98),  INT8_C(  97),  INT8_C(   5),      INT8_MAX, -INT8_C(  54), -INT8_C(   5) } },
    { {  INT8_C(  57),  INT8_C(  51),  INT8_C(  59), -INT8_C(  24), -INT8_C(  59), -INT8_C( 127),  INT8_C( 102), -INT8_C(  27),
         INT8_C(  72),  INT8_C( 126),  INT8_C(  55),  INT8_C(   1),  INT8_C( 102),  INT8_C(  38), -INT8_C(   4),  INT8_C(  93) },
      UINT16_C(22560),
      { -INT8_C(  37), -INT8_C(  17), -INT8_C( 118),  INT8_C(  37), -INT8_C( 112), -INT8_C(  73), -INT8_C(  20), -INT8_C(   3),
         INT8_C(  65),  INT8_C(   0), -INT8_C(  19),  INT8_C(  33),  INT8_C(  99),  INT8_C(  38), -INT8_C(  14), -INT8_C(  68) },
      {  INT8_C(  57),  INT8_C(  51),  INT8_C(  59), -INT8_C(  24), -INT8_C(  59),  INT8_C(  73),  INT8_C( 102), -INT8_C(  27),
         INT8_C(  72),  INT8_C( 126),  INT8_C(  55),  INT8_C(  33),  INT8_C(  99),  INT8_C(  38),  INT8_C(  14),  INT8_C(  93) } },
    { { -INT8_C(  81),  INT8_C( 103), -INT8_C(  37),  INT8_C(  36), -INT8_C(  58), -INT8_C(  71),  INT8_C(  10), -INT8_C(   8),
        -INT8_C(  90), -INT8_C(  33), -INT8_C(  34),  INT8_C(  31),  INT8_C( 116), -INT8_C(   1), -INT8_C(  63), -INT8_C(   2) },
      UINT16_C(16195),
      {  INT8_C( 106),  INT8_C(  92), -INT8_C(  34), -INT8_C(  79), -INT8_C(  62),  INT8_C(  72), -INT8_C(   3),  INT8_C(  17),
         INT8_C( 107), -INT8_C(  70), -INT8_C(  97), -INT8_C( 103), -INT8_C(  45), -INT8_C( 123), -INT8_C(   5), -INT8_C(  87) },
      {  INT8_C( 106),  INT8_C(  92), -INT8_C(  37),  INT8_C(  36), -INT8_C(  58), -INT8_C(  71),  INT8_C(   3), -INT8_C(   8),
         INT8_C( 107),  INT8_C(  70),  INT8_C(  97),  INT8_C( 103),  INT8_C(  45),  INT8_C( 123), -INT8_C(  63), -INT8_C(   2) } },
    { { -INT8_C( 117), -INT8_C(  50), -INT8_C(   3),  INT8_C(  21), -INT8_C(  14), -INT8_C( 123),  INT8_C(  98),  INT8_C( 119),
        -INT8_C( 121), -INT8_C(  35),  INT8_C(  12), -INT8_C(  82), -INT8_C(  93),  INT8_C(  40),  INT8_C(  26), -INT8_C(  79) },
      UINT16_C(29568),
      { -INT8_C(  44),  INT8_C(  14),  INT8_C(  65), -INT8_C( 112),  INT8_C(  49),  INT8_C(  81),  INT8_C(  38),  INT8_C(  71),
         INT8_C(  11),  INT8_C(  26), -INT8_C(  44),  INT8_C(  39),  INT8_C( 116),  INT8_C(  41), -INT8_C( 105), -INT8_C(  71) },
      { -INT8_C( 117), -INT8_C(  50), -INT8_C(   3),  INT8_C(  21), -INT8_C(  14), -INT8_C( 123),  INT8_C(  98),  INT8_C(  71),
         INT8_C(  11),  INT8_C(  26),  INT8_C(  12), -INT8_C(  82),  INT8_C( 116),  INT8_C(  41),  INT8_C( 105), -INT8_C(  79) } },
    { { -INT8_C( 122), -INT8_C(  67),      INT8_MIN,  INT8_C( 104),  INT8_C(  85),      INT8_MIN, -INT8_C(  85), -INT8_C(  90),
        -INT8_C( 123),  INT8_C(  72),  INT8_C( 109),  INT8_C(  61), -INT8_C(   9),  INT8_C(  22),  INT8_C(  63),  INT8_C(   2) },
      UINT16_C(24034),
      { -INT8_C(  77),  INT8_C(  93),  INT8_C( 122),  INT8_C(  26),  INT8_C(  53), -INT8_C(   6), -INT8_C(  88), -INT8_C( 106),
        -INT8_C(  81),  INT8_C(  24),  INT8_C( 109), -INT8_C( 113),  INT8_C(  69), -INT8_C(  26), -INT8_C(  67),  INT8_C(  26) },
      { -INT8_C( 122),  INT8_C(  93),      INT8_MIN,  INT8_C( 104),  INT8_C(  85),  INT8_C(   6),  INT8_C(  88),  INT8_C( 106),
         INT8_C(  81),  INT8_C(  72),  INT8_C( 109),  INT8_C( 113),  INT8_C(  69),  INT8_C(  22),  INT8_C(  67),  INT8_C(   2) } },
    { {  INT8_C(  61), -INT8_C( 112), -INT8_C(  20),  INT8_C( 115),  INT8_C(  91), -INT8_C(  47), -INT8_C(  94), -INT8_C(  48),
         INT8_C(  13),  INT8_C(  17), -INT8_C( 121), -INT8_C(  53),  INT8_C(  98), -INT8_C(   9), -INT8_C(  39), -INT8_C(  82) },
      UINT16_C(34747),
      { -INT8_C( 105), -INT8_C(  92),  INT8_C(  17), -INT8_C(  55), -INT8_C( 127), -INT8_C(  14),  INT8_C(  79),  INT8_C( 123),
        -INT8_C(  78),  INT8_C(  70),  INT8_C(  37),  INT8_C(  87), -INT8_C(  52),  INT8_C(  75), -INT8_C(  98), -INT8_C(  26) },
      {  INT8_C( 105),  INT8_C(  92), -INT8_C(  20),  INT8_C(  55),      INT8_MAX,  INT8_C(  14), -INT8_C(  94),  INT8_C( 123),
         INT8_C(  78),  INT8_C(  70),  INT8_C(  37), -INT8_C(  53),  INT8_C(  98), -INT8_C(   9), -INT8_C(  39),  INT8_C(  26) } },
    { { -INT8_C(  89), -INT8_C(  81),  INT8_C(  27), -INT8_C(  66), -INT8_C( 104),  INT8_C(  27),  INT8_C(  70),  INT8_C( 113),
         INT8_C( 106), -INT8_C( 107), -INT8_C( 123),  INT8_C(  49),  INT8_C(  96), -INT8_C(  86),  INT8_C(  88), -INT8_C(  26) },
      UINT16_C(19488),
      { -INT8_C(  47),  INT8_C( 126),      INT8_MAX, -INT8_C(  96),  INT8_C(  81), -INT8_C(  29),  INT8_C(   8), -INT8_C(  35),
         INT8_C(  14),  INT8_C(  19), -INT8_C( 126),  INT8_C( 125), -INT8_C(  16),  INT8_C(  66), -INT8_C(  59), -INT8_C(  53) },
      { -INT8_C(  89), -INT8_C(  81),  INT8_C(  27), -INT8_C(  66), -INT8_C( 104),  INT8_C(  29),  INT8_C(  70),  INT8_C( 113),
         INT8_C( 106), -INT8_C( 107),  INT8_C( 126),  INT8_C( 125),  INT8_C(  96), -INT8_C(  86),  INT8_C(  59), -INT8_C(  26) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi8(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi8(test_vec[i].a);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_abs_epi8(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_abs_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_x_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_maskz_abs_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int8_t a[16];
    const int8_t r[16];
  } test_vec[] = {
    { UINT16_C(57432),
      {  INT8_C(  44),  INT8_C(  47), -INT8_C( 120),  INT8_C( 111), -INT8_C(  65),  INT8_C(  87),  INT8_C(  90),  INT8_C(  38),
         INT8_C(  10),  INT8_C(  24), -INT8_C(  56), -INT8_C(  43), -INT8_C( 119),  INT8_C( 102), -INT8_C(  58),  INT8_C(  89) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 111),  INT8_C(  65),  INT8_C(   0),  INT8_C(  90),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 102),  INT8_C(  58),  INT8_C(  89) } },
    { UINT16_C(21485),
      {  INT8_C(  28),  INT8_C(  88),  INT8_C(  30), -INT8_C(  28), -INT8_C(  93), -INT8_C(  61),  INT8_C(  19),  INT8_C(  71),
        -INT8_C(  40), -INT8_C(   4), -INT8_C( 116), -INT8_C(  23), -INT8_C(   3), -INT8_C(  67),  INT8_C( 102),  INT8_C(  23) },
      {  INT8_C(  28),  INT8_C(   0),  INT8_C(  30),  INT8_C(  28),  INT8_C(   0),  INT8_C(  61),  INT8_C(  19),  INT8_C(  71),
         INT8_C(  40),  INT8_C(   4),  INT8_C(   0),  INT8_C(   0),  INT8_C(   3),  INT8_C(   0),  INT8_C( 102),  INT8_C(   0) } },
    { UINT16_C(21564),
      {  INT8_C(   2), -INT8_C(  52),  INT8_C( 110), -INT8_C(  13),  INT8_C(  53), -INT8_C( 127),  INT8_C(  61), -INT8_C(  25),
         INT8_C( 101), -INT8_C(   7), -INT8_C( 107),  INT8_C(  67),  INT8_C(  59),  INT8_C( 123),  INT8_C(  35), -INT8_C( 120) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C( 110),  INT8_C(  13),  INT8_C(  53),      INT8_MAX,  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C( 107),  INT8_C(   0),  INT8_C(  59),  INT8_C(   0),  INT8_C(  35),  INT8_C(   0) } },
    { UINT16_C(51548),
      { -INT8_C(  64),  INT8_C(  83),  INT8_C( 100), -INT8_C(  30),  INT8_C(  81), -INT8_C(  69),  INT8_C( 123),  INT8_C(  41),
         INT8_C(  60),  INT8_C(  42),  INT8_C(  78),  INT8_C(  33),  INT8_C(  84), -INT8_C(   1),  INT8_C(  94),  INT8_C(  63) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C( 100),  INT8_C(  30),  INT8_C(  81),  INT8_C(   0),  INT8_C( 123),  INT8_C(   0),
         INT8_C(  60),  INT8_C(   0),  INT8_C(   0),  INT8_C(  33),  INT8_C(   0),  INT8_C(   0),  INT8_C(  94),  INT8_C(  63) } },
    { UINT16_C(13397),
      { -INT8_C( 105),  INT8_C(  53), -INT8_C(  18), -INT8_C( 116), -INT8_C( 108),  INT8_C(  36),  INT8_C(  83),  INT8_C( 110),
        -INT8_C(  23), -INT8_C(  15),  INT8_C(  70),  INT8_C(   2), -INT8_C( 110), -INT8_C( 126),  INT8_C( 110),  INT8_C( 117) },
      {  INT8_C( 105),  INT8_C(   0),  INT8_C(  18),  INT8_C(   0),  INT8_C( 108),  INT8_C(   0),  INT8_C(  83),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  70),  INT8_C(   0),  INT8_C( 110),  INT8_C( 126),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(25039),
      { -INT8_C( 113),  INT8_C(  73), -INT8_C(  72), -INT8_C(  54),  INT8_C(  79),  INT8_C(  50), -INT8_C(  17), -INT8_C(  75),
        -INT8_C( 121),  INT8_C( 107),  INT8_C(  94), -INT8_C( 105), -INT8_C(  33),  INT8_C( 111),  INT8_C(  57), -INT8_C(  24) },
      {  INT8_C( 113),  INT8_C(  73),  INT8_C(  72),  INT8_C(  54),  INT8_C(   0),  INT8_C(   0),  INT8_C(  17),  INT8_C(  75),
         INT8_C( 121),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 111),  INT8_C(  57),  INT8_C(   0) } },
    { UINT16_C(40507),
      { -INT8_C(  10), -INT8_C(  56),  INT8_C(   2), -INT8_C( 109),  INT8_C(  40), -INT8_C(  13),  INT8_C(  16),  INT8_C(  50),
        -INT8_C(  69),  INT8_C(  97), -INT8_C(  32),  INT8_C(  88), -INT8_C(  63),  INT8_C(  58),  INT8_C(  62), -INT8_C(  97) },
      {  INT8_C(  10),  INT8_C(  56),  INT8_C(   0),  INT8_C( 109),  INT8_C(  40),  INT8_C(  13),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  97),  INT8_C(  32),  INT8_C(  88),  INT8_C(  63),  INT8_C(   0),  INT8_C(   0),  INT8_C(  97) } },
    { UINT16_C(31988),
      { -INT8_C(  68),  INT8_C(   6),  INT8_C(  10),  INT8_C(  17),  INT8_C(  92),  INT8_C(  60), -INT8_C(  95), -INT8_C(  56),
         INT8_C( 106), -INT8_C( 113),  INT8_C(  12), -INT8_C(  97), -INT8_C(  56), -INT8_C(  73),  INT8_C(  21), -INT8_C(  95) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(  10),  INT8_C(   0),  INT8_C(  92),  INT8_C(  60),  INT8_C(  95),  INT8_C(  56),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  12),  INT8_C(  97),  INT8_C(  56),  INT8_C(  73),  INT8_C(  21),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_abs_epi8(test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_abs_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_x_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_mask_abs_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[8];
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int16_t r[8];
  } test_vec[] = {
    { {  INT16_C(  8153),  INT16_C( 16161),  INT16_C(  6855), -INT16_C( 17824), -INT16_C( 10464), -INT16_C( 18024), -INT16_C( 25775),  INT16_C( 32280) },
      UINT8_C( 66),
      {  INT16_C( 13385), -INT16_C( 15752),  INT16_C( 13235), -INT16_C( 19058),  INT16_C( 19462), -INT16_C(  3246), -INT16_C( 30050), -INT16_C(  6596) },
      {  INT16_C(  8153),  INT16_C( 15752),  INT16_C(  6855), -INT16_C( 17824), -INT16_C( 10464), -INT16_C( 18024),  INT16_C( 30050),  INT16_C( 32280) } },
    { {  INT16_C( 15536), -INT16_C( 28087), -INT16_C( 28835),  INT16_C( 17732),  INT16_C( 28497),  INT16_C(  9121),  INT16_C( 28628), -INT16_C( 30221) },
      UINT8_C( 56),
      {  INT16_C( 15263), -INT16_C( 25944), -INT16_C( 12189), -INT16_C(   659),  INT16_C( 23072),  INT16_C( 14027),  INT16_C( 26184),  INT16_C( 13567) },
      {  INT16_C( 15536), -INT16_C( 28087), -INT16_C( 28835),  INT16_C(   659),  INT16_C( 23072),  INT16_C( 14027),  INT16_C( 28628), -INT16_C( 30221) } },
    { { -INT16_C(  3774), -INT16_C(  7245),  INT16_C(  3049),  INT16_C(  7497), -INT16_C( 12617),  INT16_C( 28131),  INT16_C( 10527),  INT16_C(  6521) },
      UINT8_C(182),
      {  INT16_C( 23186), -INT16_C( 32513), -INT16_C( 16007),  INT16_C( 28358), -INT16_C(  7161), -INT16_C( 29226), -INT16_C( 31719),  INT16_C( 24598) },
      { -INT16_C(  3774),  INT16_C( 32513),  INT16_C( 16007),  INT16_C(  7497),  INT16_C(  7161),  INT16_C( 29226),  INT16_C( 10527),  INT16_C( 24598) } },
    { {  INT16_C( 28985), -INT16_C( 21137),  INT16_C(  9114), -INT16_C(   587),  INT16_C( 10616), -INT16_C( 12703), -INT16_C( 31567),  INT16_C( 22068) },
      UINT8_C( 76),
      {  INT16_C(  4627),  INT16_C( 12178), -INT16_C( 14639),  INT16_C( 24171), -INT16_C(  9608),  INT16_C( 30857), -INT16_C(   739), -INT16_C( 22827) },
      {  INT16_C( 28985), -INT16_C( 21137),  INT16_C( 14639),  INT16_C( 24171),  INT16_C( 10616), -INT16_C( 12703),  INT16_C(   739),  INT16_C( 22068) } },
    { { -INT16_C(  1171),  INT16_C(  7354), -INT16_C(  1877),  INT16_C( 16390), -INT16_C(  6177), -INT16_C(   784),  INT16_C( 22452),  INT16_C( 15509) },
      UINT8_C(230),
      {  INT16_C(  4801),  INT16_C(   872),  INT16_C( 16760), -INT16_C(  8622),  INT16_C( 24650), -INT16_C(  6092), -INT16_C( 25601), -INT16_C( 17682) },
      { -INT16_C(  1171),  INT16_C(   872),  INT16_C( 16760),  INT16_C( 16390), -INT16_C(  6177),  INT16_C(  6092),  INT16_C( 25601),  INT16_C( 17682) } },
    { { -INT16_C( 20314),  INT16_C( 14303), -INT16_C( 10837), -INT16_C( 17361), -INT16_C( 17869),  INT16_C(  9635), -INT16_C( 26066), -INT16_C( 16289) },
      UINT8_C(245),
      {  INT16_C(    14),  INT16_C(  8028), -INT16_C( 21476),  INT16_C( 17146), -INT16_C( 11337),  INT16_C( 20019), -INT16_C( 17783), -INT16_C(  6419) },
      {  INT16_C(    14),  INT16_C( 14303),  INT16_C( 21476), -INT16_C( 17361),  INT16_C( 11337),  INT16_C( 20019),  INT16_C( 17783),  INT16_C(  6419) } },
    { {  INT16_C(  4307),  INT16_C( 28142),  INT16_C( 27919),  INT16_C( 11490), -INT16_C(  8387), -INT16_C( 13172), -INT16_C( 10842),  INT16_C( 13655) },
      UINT8_C( 62),
      {  INT16_C( 26887),  INT16_C( 10704), -INT16_C(  3529), -INT16_C(  7720),  INT16_C( 27957), -INT16_C(  9436), -INT16_C(  7956), -INT16_C( 29431) },
      {  INT16_C(  4307),  INT16_C( 10704),  INT16_C(  3529),  INT16_C(  7720),  INT16_C( 27957),  INT16_C(  9436), -INT16_C( 10842),  INT16_C( 13655) } },
    { {  INT16_C( 29302), -INT16_C( 19446), -INT16_C( 25972),  INT16_C( 16877), -INT16_C(   505), -INT16_C(  4405),  INT16_C( 25296), -INT16_C( 23565) },
      UINT8_C(201),
      {  INT16_C(  9477),  INT16_C( 32174), -INT16_C( 29767),  INT16_C( 24616), -INT16_C(  3737), -INT16_C( 24240), -INT16_C( 27071),  INT16_C( 29624) },
      {  INT16_C(  9477), -INT16_C( 19446), -INT16_C( 25972),  INT16_C( 24616), -INT16_C(   505), -INT16_C(  4405),  INT16_C( 27071),  INT16_C( 29624) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi16(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi16(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_abs_epi16(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_abs_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_x_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_maskz_abs_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C(  9),
      { -INT16_C( 18170),  INT16_C(  3543), -INT16_C(  6732),  INT16_C(  2591),  INT16_C( 16089),  INT16_C( 25611), -INT16_C( 22208), -INT16_C( 26321) },
      {  INT16_C( 18170),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2591),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(246),
      {  INT16_C( 18289), -INT16_C( 24027), -INT16_C( 29991), -INT16_C(  1982),  INT16_C( 32714), -INT16_C(   354), -INT16_C( 22665),  INT16_C(  9312) },
      {  INT16_C(     0),  INT16_C( 24027),  INT16_C( 29991),  INT16_C(     0),  INT16_C( 32714),  INT16_C(   354),  INT16_C( 22665),  INT16_C(  9312) } },
    { UINT8_C(207),
      {  INT16_C( 11566),  INT16_C( 22926),  INT16_C( 14011), -INT16_C(  7358), -INT16_C( 18332),  INT16_C( 15086),  INT16_C( 13877),  INT16_C( 27416) },
      {  INT16_C( 11566),  INT16_C( 22926),  INT16_C( 14011),  INT16_C(  7358),  INT16_C(     0),  INT16_C(     0),  INT16_C( 13877),  INT16_C( 27416) } },
    { UINT8_C( 86),
      { -INT16_C(  2382), -INT16_C(  5907),  INT16_C( 22152),  INT16_C( 19099),  INT16_C( 30599),  INT16_C(  3850), -INT16_C(  4906),  INT16_C(  4835) },
      {  INT16_C(     0),  INT16_C(  5907),  INT16_C( 22152),  INT16_C(     0),  INT16_C( 30599),  INT16_C(     0),  INT16_C(  4906),  INT16_C(     0) } },
    { UINT8_C(119),
      { -INT16_C( 32551), -INT16_C( 10402), -INT16_C( 27415), -INT16_C( 17203), -INT16_C( 15129),  INT16_C( 10840), -INT16_C(  3485), -INT16_C( 22834) },
      {  INT16_C( 32551),  INT16_C( 10402),  INT16_C( 27415),  INT16_C(     0),  INT16_C( 15129),  INT16_C( 10840),  INT16_C(  3485),  INT16_C(     0) } },
    { UINT8_C(  9),
      { -INT16_C( 16078), -INT16_C(  4277),  INT16_C(  6446), -INT16_C( 24958), -INT16_C(  6732), -INT16_C(  8610), -INT16_C( 25530),  INT16_C( 24833) },
      {  INT16_C( 16078),  INT16_C(     0),  INT16_C(     0),  INT16_C( 24958),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(136),
      {  INT16_C(  2458),  INT16_C( 11322), -INT16_C( 10557),  INT16_C( 10157), -INT16_C( 13767),  INT16_C( 17938),  INT16_C( 29253), -INT16_C( 10355) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10157),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10355) } },
    { UINT8_C( 67),
      {  INT16_C(  3514), -INT16_C( 28361),  INT16_C( 31771),  INT16_C(  5728), -INT16_C(  9840),  INT16_C( 11002), -INT16_C(  7475), -INT16_C(  5133) },
      {  INT16_C(  3514),  INT16_C( 28361),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  7475),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_abs_epi16(test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_abs_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_x_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_mask_abs_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(   688398243),  INT32_C(  2117596500), -INT32_C(   750842275),  INT32_C(   366535198) },
      UINT8_C(131),
      { -INT32_C(  1004016930), -INT32_C(  1077141926), -INT32_C(  2083644661),  INT32_C(   399895044) },
      {  INT32_C(  1004016930),  INT32_C(  1077141926), -INT32_C(   750842275),  INT32_C(   366535198) } },
    { {  INT32_C(  1632121691),  INT32_C(   483536164), -INT32_C(   526963188), -INT32_C(  1230342708) },
      UINT8_C(  2),
      {  INT32_C(  2145654124), -INT32_C(  1724078204), -INT32_C(   190821781), -INT32_C(  1219539762) },
      {  INT32_C(  1632121691),  INT32_C(  1724078204), -INT32_C(   526963188), -INT32_C(  1230342708) } },
    { { -INT32_C(   780236771),  INT32_C(  1976716971),  INT32_C(  1074971562),  INT32_C(  1213854368) },
      UINT8_C( 82),
      { -INT32_C(   767166523),  INT32_C(  1085468303), -INT32_C(   295595563),  INT32_C(   669742458) },
      { -INT32_C(   780236771),  INT32_C(  1085468303),  INT32_C(  1074971562),  INT32_C(  1213854368) } },
    { { -INT32_C(  1066078121),  INT32_C(  1916170187), -INT32_C(  1589423098), -INT32_C(   746781550) },
      UINT8_C(136),
      { -INT32_C(   802933306), -INT32_C(   186975219), -INT32_C(  1081305950),  INT32_C(  1075243371) },
      { -INT32_C(  1066078121),  INT32_C(  1916170187), -INT32_C(  1589423098),  INT32_C(  1075243371) } },
    { {  INT32_C(   955441731), -INT32_C(  1927520383),  INT32_C(   841960739), -INT32_C(  1971983518) },
      UINT8_C( 63),
      { -INT32_C(  1129031646), -INT32_C(  1553699482), -INT32_C(  1621136138), -INT32_C(   791151103) },
      {  INT32_C(  1129031646),  INT32_C(  1553699482),  INT32_C(  1621136138),  INT32_C(   791151103) } },
    { {  INT32_C(  2072269077),  INT32_C(  1390338014), -INT32_C(   681233355), -INT32_C(   586259273) },
      UINT8_C( 73),
      {  INT32_C(   289225178), -INT32_C(  1951535354), -INT32_C(  1646281947), -INT32_C(   283269702) },
      {  INT32_C(   289225178),  INT32_C(  1390338014), -INT32_C(   681233355),  INT32_C(   283269702) } },
    { {  INT32_C(   352437480), -INT32_C(   669662064), -INT32_C(  1349420366),  INT32_C(  1478068007) },
      UINT8_C( 52),
      { -INT32_C(  1411603801), -INT32_C(  1980243425),  INT32_C(   161641122), -INT32_C(  1088019476) },
      {  INT32_C(   352437480), -INT32_C(   669662064),  INT32_C(   161641122),  INT32_C(  1478068007) } },
    { { -INT32_C(   968386477), -INT32_C(   888428856), -INT32_C(   552543373), -INT32_C(  1460967715) },
      UINT8_C(167),
      { -INT32_C(  1057832772), -INT32_C(  1469689236), -INT32_C(   300347505), -INT32_C(    52757827) },
      {  INT32_C(  1057832772),  INT32_C(  1469689236),  INT32_C(   300347505), -INT32_C(  1460967715) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_abs_epi32(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_abs_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_maskz_abs_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(145),
      {  INT32_C(    29805490), -INT32_C(  2083285805),  INT32_C(   753740199), -INT32_C(  1343338556) },
      {  INT32_C(    29805490),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 58),
      { -INT32_C(  2072344551), -INT32_C(   137560356), -INT32_C(   577438960), -INT32_C(  1224979635) },
      {  INT32_C(           0),  INT32_C(   137560356),  INT32_C(           0),  INT32_C(  1224979635) } },
    { UINT8_C(109),
      {  INT32_C(   815986804), -INT32_C(   520418861), -INT32_C(  1705291520), -INT32_C(  1422986918) },
      {  INT32_C(   815986804),  INT32_C(           0),  INT32_C(  1705291520),  INT32_C(  1422986918) } },
    { UINT8_C(145),
      { -INT32_C(  1602009068),  INT32_C(   676272594),  INT32_C(  1754227610),  INT32_C(   567182279) },
      {  INT32_C(  1602009068),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(195),
      {  INT32_C(  1694336367),  INT32_C(   738012218),  INT32_C(    87416787), -INT32_C(  2145881269) },
      {  INT32_C(  1694336367),  INT32_C(   738012218),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 97),
      { -INT32_C(   444185248), -INT32_C(   216805061),  INT32_C(   376077454),  INT32_C(   835265240) },
      {  INT32_C(   444185248),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    {    UINT8_MAX,
      { -INT32_C(  1160186485),  INT32_C(  1353662651),  INT32_C(  2034799586), -INT32_C(   705717215) },
      {  INT32_C(  1160186485),  INT32_C(  1353662651),  INT32_C(  2034799586),  INT32_C(   705717215) } },
    { UINT8_C(168),
      {  INT32_C(  1699267364),  INT32_C(   479861968), -INT32_C(   177248900), -INT32_C(  1180950087) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1180950087) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_abs_epi32(test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_abs_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_abs_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[2];
    const int64_t r[2];
  } test_vec[] = {
    { {                      INT64_MIN,  INT64_C( 5987331142896800384) },
      {                      INT64_MIN,  INT64_C( 5987331142896800384) } },
    { { -INT64_C( 6165271089019809896), -INT64_C( 1488269006246725939) },
      {  INT64_C( 6165271089019809896),  INT64_C( 1488269006246725939) } },
    { { -INT64_C(  287912670071654876),  INT64_C( 3376558256458965752) },
      {  INT64_C(  287912670071654876),  INT64_C( 3376558256458965752) } },
    { { -INT64_C( 1699690728377702014),  INT64_C( 2927647255755636771) },
      {  INT64_C( 1699690728377702014),  INT64_C( 2927647255755636771) } },
    { { -INT64_C( 8959542323819455163),  INT64_C( 3365246129411480893) },
      {  INT64_C( 8959542323819455163),  INT64_C( 3365246129411480893) } },
    { {  INT64_C( 4227824362795330185),  INT64_C( 6194577401110150880) },
      {  INT64_C( 4227824362795330185),  INT64_C( 6194577401110150880) } },
    { {  INT64_C( 6873617928876373866),  INT64_C( 1262142814710839683) },
      {  INT64_C( 6873617928876373866),  INT64_C( 1262142814710839683) } },
    { {  INT64_C(  722086948698055913), -INT64_C( 4941936896584979953) },
      {  INT64_C(  722086948698055913),  INT64_C( 4941936896584979953) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_abs_epi64(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_abs_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_x_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_mask_abs_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[2];
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 4845332346535929233),  INT64_C( 1700698435022438078) },
      UINT8_C(130),
      { -INT64_C( 6803045812735068648), -INT64_C( 7443449697644586270) },
      {  INT64_C( 4845332346535929233),  INT64_C( 7443449697644586270) } },
    { { -INT64_C( 6421298693609969513), -INT64_C( 6160319844260546176) },
      UINT8_C( 59),
      { -INT64_C( 4351657591194229711), -INT64_C( 8627740096035247728) },
      {  INT64_C( 4351657591194229711),  INT64_C( 8627740096035247728) } },
    { { -INT64_C( 8180037481821730213),  INT64_C( 7219960493591948494) },
      UINT8_C( 49),
      { -INT64_C( 6441345642108472215),  INT64_C( 4350044603238480648) },
      {  INT64_C( 6441345642108472215),  INT64_C( 7219960493591948494) } },
    { {  INT64_C( 4684076903763347163),  INT64_C( 6497802772857514833) },
      UINT8_C(205),
      { -INT64_C( 4870124432114791231),  INT64_C( 4454143856972221582) },
      {  INT64_C( 4870124432114791231),  INT64_C( 6497802772857514833) } },
    { { -INT64_C(  838855374297144746), -INT64_C( 2942560270663534524) },
      UINT8_C(120),
      {  INT64_C( 5641214509537388547),  INT64_C( 4712163805488714118) },
      { -INT64_C(  838855374297144746), -INT64_C( 2942560270663534524) } },
    { {  INT64_C( 7176515612344537603), -INT64_C( 8643734220088015145) },
      UINT8_C(168),
      { -INT64_C( 3490178188729363300),  INT64_C( 7993754077794638996) },
      {  INT64_C( 7176515612344537603), -INT64_C( 8643734220088015145) } },
    { { -INT64_C( 1529783215006713101), -INT64_C( 1978515024379923929) },
      UINT8_C(204),
      { -INT64_C( 8261273454123855187), -INT64_C(  408440238321563495) },
      { -INT64_C( 1529783215006713101), -INT64_C( 1978515024379923929) } },
    { { -INT64_C( 3158447172117950868), -INT64_C( 3303403632531072544) },
      UINT8_C( 84),
      { -INT64_C( 5291873217680795087),  INT64_C( 4801197429913235623) },
      { -INT64_C( 3158447172117950868), -INT64_C( 3303403632531072544) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi64(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi64(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_abs_epi64(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_abs_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_x_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_maskz_abs_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(158),
      { -INT64_C( 6298226099324920239), -INT64_C( 5000322151057574458) },
      {  INT64_C(                   0),  INT64_C( 5000322151057574458) } },
    { UINT8_C( 43),
      {  INT64_C( 2636038210509465369),  INT64_C( 7020928684628243752) },
      {  INT64_C( 2636038210509465369),  INT64_C( 7020928684628243752) } },
    { UINT8_C( 64),
      {  INT64_C( 7649134006013225985),  INT64_C( 2078749890811515096) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(207),
      {  INT64_C(  941032990317475364), -INT64_C( 6013459460053205151) },
      {  INT64_C(  941032990317475364),  INT64_C( 6013459460053205151) } },
    { UINT8_C(103),
      { -INT64_C( 1560295149959329567), -INT64_C( 3971587257135282239) },
      {  INT64_C( 1560295149959329567),  INT64_C( 3971587257135282239) } },
    { UINT8_C( 38),
      { -INT64_C( 5243445501069980794),  INT64_C( 4885633393584462144) },
      {  INT64_C(                   0),  INT64_C( 4885633393584462144) } },
    { UINT8_C( 94),
      {  INT64_C( 7243498660887455097),  INT64_C( 8890095449815425622) },
      {  INT64_C(                   0),  INT64_C( 8890095449815425622) } },
    { UINT8_C( 14),
      {  INT64_C( 7759806299451765498), -INT64_C( 6445959026453494579) },
      {  INT64_C(                   0),  INT64_C( 6445959026453494579) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_abs_epi64(test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_abs_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_x_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_mask_abs_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t src[32];
    uint32_t k;
    int8_t a[32];
    int8_t r[32];
  } test_vec[] = {
    { {  INT8_C(  10),  INT8_C(   0),  INT8_C(  35), -INT8_C(  12),  INT8_C( 116), -INT8_C(  89), -INT8_C( 119), -INT8_C(  53),
        -INT8_C(  45), -INT8_C(   8), -INT8_C(  84),  INT8_C( 102), -INT8_C(  56),  INT8_C( 109), -INT8_C(  78), -INT8_C( 127),
         INT8_C(   2), -INT8_C( 105),  INT8_C(  18),  INT8_C(  40),  INT8_C(  50), -INT8_C( 123),  INT8_C(  40),  INT8_C(  13),
         INT8_C(  37), -INT8_C(  59), -INT8_C(  82),  INT8_C(  94),  INT8_C(  12),      INT8_MAX,  INT8_C(  24),  INT8_C(  22) },
      UINT32_C(4077534079),
      { -INT8_C(  30), -INT8_C( 109), -INT8_C(  65), -INT8_C(  75), -INT8_C( 116),  INT8_C( 107),  INT8_C(  27),  INT8_C(  84),
        -INT8_C(  39), -INT8_C(  50), -INT8_C(  43), -INT8_C(  37),  INT8_C( 101), -INT8_C(  25),  INT8_C(   3), -INT8_C( 104),
         INT8_C( 108),  INT8_C(  44), -INT8_C(  91), -INT8_C( 111), -INT8_C(  15),  INT8_C(  83), -INT8_C(  17), -INT8_C(   3),
        -INT8_C(  46),  INT8_C(   7),  INT8_C(  19),  INT8_C(  81),  INT8_C(  67),  INT8_C(  29),  INT8_C(  69),  INT8_C(  37) },
      {  INT8_C(  30),  INT8_C( 109),  INT8_C(  65),  INT8_C(  75),  INT8_C( 116),  INT8_C( 107),  INT8_C(  27), -INT8_C(  53),
         INT8_C(  39),  INT8_C(  50), -INT8_C(  84),  INT8_C(  37),  INT8_C( 101),  INT8_C(  25), -INT8_C(  78), -INT8_C( 127),
         INT8_C(   2),  INT8_C(  44),  INT8_C(  18),  INT8_C( 111),  INT8_C(  50), -INT8_C( 123),  INT8_C(  40),  INT8_C(  13),
         INT8_C(  46),  INT8_C(   7), -INT8_C(  82),  INT8_C(  94),  INT8_C(  67),  INT8_C(  29),  INT8_C(  69),  INT8_C(  37) } },
    { { -INT8_C(  80),  INT8_C(   4), -INT8_C(  37),  INT8_C(  60),  INT8_C( 111), -INT8_C(  10), -INT8_C( 112),  INT8_C(  72),
        -INT8_C(  60),  INT8_C( 102),  INT8_C(  35),  INT8_C(  42),  INT8_C(  77),  INT8_C(  39), -INT8_C(  62), -INT8_C(  70),
         INT8_C(  83),  INT8_C( 103),  INT8_C(  75),  INT8_C(  68), -INT8_C(  70),  INT8_C(  59),  INT8_C(  65), -INT8_C( 116),
         INT8_C(  66),  INT8_C(  84), -INT8_C(  35), -INT8_C( 123),  INT8_C( 113),  INT8_C(  34), -INT8_C(  85),  INT8_C(  33) },
      UINT32_C(2522777126),
      {  INT8_C( 124), -INT8_C(  18), -INT8_C(  34),  INT8_C(  65),  INT8_C(  84),  INT8_C(   2),  INT8_C( 107), -INT8_C(  94),
         INT8_C(  41),  INT8_C(  45),  INT8_C(  92),  INT8_C( 124), -INT8_C( 108), -INT8_C(  89), -INT8_C(  64),  INT8_C(  78),
        -INT8_C(  30),  INT8_C(   1), -INT8_C(  38),  INT8_C(  37),  INT8_C(  85), -INT8_C(  73), -INT8_C(  86), -INT8_C(  58),
        -INT8_C(  38),  INT8_C(  85), -INT8_C(  25),  INT8_C(   0), -INT8_C(  37),  INT8_C(  69), -INT8_C( 106),  INT8_C(  88) },
      { -INT8_C(  80),  INT8_C(  18),  INT8_C(  34),  INT8_C(  60),  INT8_C( 111),  INT8_C(   2), -INT8_C( 112),  INT8_C(  72),
        -INT8_C(  60),  INT8_C(  45),  INT8_C(  92),  INT8_C(  42),  INT8_C(  77),  INT8_C(  39), -INT8_C(  62),  INT8_C(  78),
         INT8_C(  83),  INT8_C(   1),  INT8_C(  38),  INT8_C(  37),  INT8_C(  85),  INT8_C(  59),  INT8_C(  86), -INT8_C( 116),
         INT8_C(  66),  INT8_C(  85),  INT8_C(  25), -INT8_C( 123),  INT8_C(  37),  INT8_C(  34), -INT8_C(  85),  INT8_C(  88) } },
    { {  INT8_C(  52),  INT8_C( 117), -INT8_C( 103), -INT8_C( 120),  INT8_C( 119),  INT8_C(   4),  INT8_C(  42), -INT8_C(  96),
         INT8_C(  49), -INT8_C( 122),  INT8_C(  28), -INT8_C(  59),  INT8_C(  46), -INT8_C(  36),  INT8_C(  19),  INT8_C(  16),
        -INT8_C(  35), -INT8_C(  19),  INT8_C(  53),  INT8_C(  50), -INT8_C(  92), -INT8_C(  32), -INT8_C(   8),  INT8_C( 126),
         INT8_C(  53), -INT8_C(  33),      INT8_MAX,  INT8_C(  17),  INT8_C(  37),  INT8_C(  21),  INT8_C( 105),  INT8_C(  89) },
      UINT32_C(  31523466),
      {  INT8_C(   6),  INT8_C(  12), -INT8_C(  95),  INT8_C(  55), -INT8_C( 110), -INT8_C(  67), -INT8_C(   4), -INT8_C(  64),
        -INT8_C( 103),  INT8_C(  15), -INT8_C(  47),  INT8_C( 118), -INT8_C(   4),  INT8_C(   6), -INT8_C(  88), -INT8_C(  96),
        -INT8_C(  26), -INT8_C(  96),  INT8_C(  31),  INT8_C(  28),      INT8_MIN, -INT8_C(  98),  INT8_C(  45), -INT8_C(  91),
        -INT8_C(  77), -INT8_C( 106), -INT8_C(   2),  INT8_C(  62), -INT8_C( 104), -INT8_C(  33),  INT8_C(  63), -INT8_C(  98) },
      {  INT8_C(  52),  INT8_C(  12), -INT8_C( 103),  INT8_C(  55),  INT8_C( 119),  INT8_C(   4),  INT8_C(  42),  INT8_C(  64),
         INT8_C(  49),  INT8_C(  15),  INT8_C(  28), -INT8_C(  59),  INT8_C(  46), -INT8_C(  36),  INT8_C(  19),  INT8_C(  16),
         INT8_C(  26), -INT8_C(  19),  INT8_C(  53),  INT8_C(  50), -INT8_C(  92),  INT8_C(  98),  INT8_C(  45),  INT8_C(  91),
         INT8_C(  77), -INT8_C(  33),      INT8_MAX,  INT8_C(  17),  INT8_C(  37),  INT8_C(  21),  INT8_C( 105),  INT8_C(  89) } },
    { { -INT8_C(  21), -INT8_C(  31), -INT8_C(  43),  INT8_C( 126), -INT8_C(  98), -INT8_C(  47),  INT8_C(  62),  INT8_C(  56),
        -INT8_C(  32),  INT8_C(  15), -INT8_C(  82), -INT8_C(  36),  INT8_C(  22),  INT8_C(  87),  INT8_C( 124), -INT8_C(   4),
        -INT8_C(   9), -INT8_C( 101),  INT8_C(  24),  INT8_C( 119),  INT8_C(  57),  INT8_C(  69),  INT8_C(  28), -INT8_C(  19),
        -INT8_C(  37),  INT8_C(  26),  INT8_C(  43),  INT8_C( 115), -INT8_C(   6),  INT8_C( 106),  INT8_C(  17), -INT8_C(  27) },
      UINT32_C(3932415563),
      { -INT8_C(  73), -INT8_C(  94),  INT8_C(  34), -INT8_C( 105), -INT8_C(  79), -INT8_C(  48),  INT8_C( 115), -INT8_C(  57),
         INT8_C(  39), -INT8_C(  16), -INT8_C(  60),  INT8_C(  31), -INT8_C( 117), -INT8_C(  36), -INT8_C( 106), -INT8_C(  59),
         INT8_C(  34), -INT8_C(  77), -INT8_C(  78), -INT8_C(   3), -INT8_C(  51), -INT8_C(  35),  INT8_C( 113), -INT8_C(  57),
         INT8_C(  71), -INT8_C( 126), -INT8_C(  83), -INT8_C( 109),  INT8_C( 105),  INT8_C(  16),  INT8_C( 125),  INT8_C(  32) },
      {  INT8_C(  73),  INT8_C(  94), -INT8_C(  43),  INT8_C( 105), -INT8_C(  98), -INT8_C(  47),  INT8_C( 115),  INT8_C(  56),
        -INT8_C(  32),  INT8_C(  16),  INT8_C(  60), -INT8_C(  36),  INT8_C(  22),  INT8_C(  36),  INT8_C( 106),  INT8_C(  59),
         INT8_C(  34),  INT8_C(  77),  INT8_C(  24),  INT8_C( 119),  INT8_C(  57),  INT8_C(  35),  INT8_C( 113), -INT8_C(  19),
        -INT8_C(  37),  INT8_C( 126),  INT8_C(  43),  INT8_C( 109), -INT8_C(   6),  INT8_C(  16),  INT8_C( 125),  INT8_C(  32) } },
    { { -INT8_C(  78), -INT8_C(  97), -INT8_C(  72),  INT8_C( 100),  INT8_C( 111),  INT8_C(  43),  INT8_C(  43), -INT8_C( 105),
         INT8_C(  27), -INT8_C(  17), -INT8_C(  74), -INT8_C(  89), -INT8_C(  52),  INT8_C(  76),  INT8_C( 108), -INT8_C(  18),
        -INT8_C(   1),  INT8_C(  30), -INT8_C(  21), -INT8_C(  51), -INT8_C(   5),  INT8_C(  92), -INT8_C( 108),  INT8_C(  66),
        -INT8_C(  33),  INT8_C(  65), -INT8_C(  43),  INT8_C(  72),  INT8_C(  82),  INT8_C(  82),  INT8_C( 104),  INT8_C(   4) },
      UINT32_C(1634214129),
      {  INT8_C(  76), -INT8_C( 108), -INT8_C(   8),  INT8_C( 103), -INT8_C( 125), -INT8_C(  82),  INT8_C(  14),  INT8_C(  79),
        -INT8_C(   6),  INT8_C( 122),  INT8_C(  61), -INT8_C(   6), -INT8_C( 104),  INT8_C(  41), -INT8_C(  57), -INT8_C( 109),
        -INT8_C( 123),  INT8_C(  91), -INT8_C(  42),  INT8_C( 100), -INT8_C(  99), -INT8_C(  85), -INT8_C(  84), -INT8_C(  17),
        -INT8_C(   2),  INT8_C(  21), -INT8_C(  13), -INT8_C(  17),  INT8_C(  53),  INT8_C(  92),  INT8_C(  80), -INT8_C( 127) },
      {  INT8_C(  76), -INT8_C(  97), -INT8_C(  72),  INT8_C( 100),  INT8_C( 125),  INT8_C(  82),  INT8_C(  14),  INT8_C(  79),
         INT8_C(  27), -INT8_C(  17), -INT8_C(  74), -INT8_C(  89), -INT8_C(  52),  INT8_C(  41),  INT8_C( 108), -INT8_C(  18),
        -INT8_C(   1),  INT8_C(  30), -INT8_C(  21),  INT8_C( 100), -INT8_C(   5),  INT8_C(  85),  INT8_C(  84),  INT8_C(  66),
         INT8_C(   2),  INT8_C(  65), -INT8_C(  43),  INT8_C(  72),  INT8_C(  82),  INT8_C(  92),  INT8_C(  80),  INT8_C(   4) } },
    { { -INT8_C(  16),  INT8_C(  72), -INT8_C(  23),  INT8_C( 115), -INT8_C(  10), -INT8_C(   9), -INT8_C(  61), -INT8_C(  15),
         INT8_C( 114),  INT8_C(   0), -INT8_C(  21),  INT8_C(  10),  INT8_C(  41), -INT8_C(  78), -INT8_C(  98), -INT8_C(  81),
         INT8_C(  13),  INT8_C( 116),  INT8_C(  19), -INT8_C(  86),  INT8_C(  31), -INT8_C(  64), -INT8_C( 103),  INT8_C(  29),
        -INT8_C(  43), -INT8_C( 115),  INT8_C(  13),  INT8_C(  10), -INT8_C(  23),  INT8_C(  93), -INT8_C( 116), -INT8_C(  39) },
      UINT32_C(2622256550),
      {  INT8_C( 108),  INT8_C(  15), -INT8_C( 115), -INT8_C(  34),  INT8_C(  16),  INT8_C( 120), -INT8_C(  23),  INT8_C(  57),
         INT8_C(  42), -INT8_C( 121), -INT8_C(  24),  INT8_C(  56), -INT8_C(   5), -INT8_C(   4), -INT8_C(  30),  INT8_C(  26),
        -INT8_C(  68),  INT8_C( 124),  INT8_C(  56), -INT8_C( 111),  INT8_C(   9),  INT8_C(  69), -INT8_C( 101), -INT8_C(  14),
        -INT8_C(  94),  INT8_C(  39), -INT8_C(  53),  INT8_C(  72), -INT8_C( 100),  INT8_C(  23), -INT8_C(  27),  INT8_C(   9) },
      { -INT8_C(  16),  INT8_C(  15),  INT8_C( 115),  INT8_C( 115), -INT8_C(  10),  INT8_C( 120), -INT8_C(  61),  INT8_C(  57),
         INT8_C(  42),  INT8_C(   0),  INT8_C(  24),  INT8_C(  10),  INT8_C(   5),  INT8_C(   4),  INT8_C(  30), -INT8_C(  81),
         INT8_C(  13),  INT8_C( 116),  INT8_C(  56),  INT8_C( 111),  INT8_C(  31), -INT8_C(  64),  INT8_C( 101),  INT8_C(  29),
        -INT8_C(  43), -INT8_C( 115),  INT8_C(  53),  INT8_C(  72),  INT8_C( 100),  INT8_C(  93), -INT8_C( 116),  INT8_C(   9) } },
    { {  INT8_C(  39),  INT8_C( 114), -INT8_C(  25),  INT8_C(  55), -INT8_C(  21), -INT8_C(  48),  INT8_C( 112),  INT8_C(  21),
         INT8_C(  87),  INT8_C(  89),  INT8_C(  77),  INT8_C(  82),  INT8_C(  85),  INT8_C(  48),  INT8_C( 109),  INT8_C(  17),
        -INT8_C(  84), -INT8_C(  91), -INT8_C(  94), -INT8_C(  75), -INT8_C(  22),  INT8_C(  61), -INT8_C(  89), -INT8_C( 116),
         INT8_C( 101),  INT8_C( 114), -INT8_C(  43),  INT8_C(   1), -INT8_C( 119), -INT8_C(  70),  INT8_C(  10), -INT8_C(  80) },
      UINT32_C( 401076780),
      { -INT8_C(  62),  INT8_C(  88),  INT8_C(  45),  INT8_C(  26), -INT8_C(  79),  INT8_C( 122),  INT8_C( 108),  INT8_C(   6),
        -INT8_C(  86), -INT8_C(  39),  INT8_C(  23),  INT8_C(  86),  INT8_C( 126), -INT8_C(  71),  INT8_C(  11),  INT8_C( 104),
        -INT8_C(  10), -INT8_C(  78), -INT8_C(  11),  INT8_C(  91),  INT8_C(  36), -INT8_C(  54),  INT8_C(  93), -INT8_C(  82),
        -INT8_C( 124),  INT8_C( 103),  INT8_C(  94), -INT8_C(  80),  INT8_C(  89),  INT8_C(  70), -INT8_C(  56),  INT8_C(  28) },
      {  INT8_C(  39),  INT8_C( 114),  INT8_C(  45),  INT8_C(  26), -INT8_C(  21),  INT8_C( 122),  INT8_C( 112),  INT8_C(  21),
         INT8_C(  87),  INT8_C(  39),  INT8_C(  77),  INT8_C(  82),  INT8_C( 126),  INT8_C(  71),  INT8_C(  11),  INT8_C( 104),
         INT8_C(  10),  INT8_C(  78),  INT8_C(  11), -INT8_C(  75), -INT8_C(  22),  INT8_C(  54),  INT8_C(  93),  INT8_C(  82),
         INT8_C( 124),  INT8_C( 103),  INT8_C(  94),  INT8_C(   1),  INT8_C(  89), -INT8_C(  70),  INT8_C(  10), -INT8_C(  80) } },
    { { -INT8_C(  98), -INT8_C(  11),  INT8_C(  54),  INT8_C(  79),  INT8_C( 111), -INT8_C(  94),  INT8_C(  85),  INT8_C(  26),
         INT8_C( 124),  INT8_C( 108),  INT8_C( 112), -INT8_C(   6),  INT8_C(  37),  INT8_C( 124),  INT8_C(  99),  INT8_C(  27),
         INT8_C(  46),  INT8_C(  88),  INT8_C( 119),  INT8_C(  83),  INT8_C(  34), -INT8_C(  44),  INT8_C(   1), -INT8_C(  90),
         INT8_C(  59),  INT8_C(  95),  INT8_C(  86), -INT8_C( 107), -INT8_C(  91),  INT8_C(  30), -INT8_C(  79),  INT8_C(  67) },
      UINT32_C(2207442707),
      { -INT8_C( 119), -INT8_C(  25), -INT8_C(  99),  INT8_C(   5),  INT8_C(  83),  INT8_C(  13),  INT8_C(   0),  INT8_C( 120),
        -INT8_C( 119),  INT8_C(  99), -INT8_C( 108), -INT8_C(  72), -INT8_C(  69),  INT8_C(  11),  INT8_C(  11), -INT8_C(  35),
        -INT8_C(  33),  INT8_C(  12), -INT8_C( 125),  INT8_C(  26),  INT8_C( 107), -INT8_C(  39), -INT8_C(  81),  INT8_C(  17),
        -INT8_C(   8),  INT8_C(  96),  INT8_C(  84),  INT8_C(  11),  INT8_C(  71), -INT8_C(  25), -INT8_C( 114), -INT8_C(  47) },
      {  INT8_C( 119),  INT8_C(  25),  INT8_C(  54),  INT8_C(  79),  INT8_C(  83), -INT8_C(  94),  INT8_C(  85),  INT8_C(  26),
         INT8_C( 119),  INT8_C(  99),  INT8_C( 108), -INT8_C(   6),  INT8_C(  37),  INT8_C(  11),  INT8_C(  11),  INT8_C(  35),
         INT8_C(  46),  INT8_C(  12),  INT8_C( 119),  INT8_C(  83),  INT8_C( 107), -INT8_C(  44),  INT8_C(   1),  INT8_C(  17),
         INT8_C(   8),  INT8_C(  96),  INT8_C(  86), -INT8_C( 107), -INT8_C(  91),  INT8_C(  30), -INT8_C(  79),  INT8_C(  47) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi8(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_abs_epi8(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_abs_epi8");
    easysimd_test_x86_assert_equal_i8x32(easysimd_mm256_loadu_epi8(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = easysimd_mm256_mask_abs_epi8(src, k, a);

    easysimd_test_x86_write_i8x32(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_abs_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint32_t k;
    int8_t a[32];
    int8_t r[32];
  } test_vec[] = {
    { UINT32_C(  91363317),
      {  INT8_C(  37),  INT8_C(  93),  INT8_C(  35), -INT8_C( 104),  INT8_C(  26), -INT8_C(  69), -INT8_C( 112),  INT8_C(  83),
        -INT8_C(  95),  INT8_C( 112),  INT8_C(  94),  INT8_C(  44), -INT8_C(  24),  INT8_C(  42),  INT8_C( 115),  INT8_C(  22),
        -INT8_C(  15),  INT8_C(  82),  INT8_C(  24), -INT8_C(  88),  INT8_C( 108), -INT8_C( 115), -INT8_C(  39),  INT8_C( 124),
         INT8_C(  13),  INT8_C(   4),  INT8_C(  16),  INT8_C(   2),  INT8_C(  27), -INT8_C( 126),  INT8_C(   7),  INT8_C(  65) },
      {  INT8_C(  37),  INT8_C(   0),  INT8_C(  35),  INT8_C(   0),  INT8_C(  26),  INT8_C(  69),  INT8_C( 112),  INT8_C(  83),
         INT8_C(  95),  INT8_C( 112),  INT8_C(  94),  INT8_C(   0),  INT8_C(  24),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  82),  INT8_C(   0),  INT8_C(   0),  INT8_C( 108),  INT8_C( 115),  INT8_C(  39),  INT8_C(   0),
         INT8_C(  13),  INT8_C(   0),  INT8_C(  16),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C(4191759327),
      { -INT8_C(  26),  INT8_C( 105),  INT8_C(  77), -INT8_C( 121), -INT8_C(  39), -INT8_C(  85), -INT8_C(  77), -INT8_C(  63),
        -INT8_C(  42),  INT8_C(  38), -INT8_C(  41), -INT8_C(  57),  INT8_C( 121), -INT8_C(  17),  INT8_C( 111), -INT8_C(  27),
         INT8_C( 125),  INT8_C(  72),  INT8_C(  98), -INT8_C( 118),  INT8_C(  77),  INT8_C( 114), -INT8_C( 116),  INT8_C( 104),
        -INT8_C(  12), -INT8_C( 109), -INT8_C(  87), -INT8_C(  44), -INT8_C(  66), -INT8_C( 126), -INT8_C(  51), -INT8_C(  92) },
      {  INT8_C(  26),  INT8_C( 105),  INT8_C(  77),  INT8_C( 121),  INT8_C(  39),  INT8_C(   0),  INT8_C(  77),  INT8_C(  63),
         INT8_C(  42),  INT8_C(  38),  INT8_C(   0),  INT8_C(  57),  INT8_C(   0),  INT8_C(  17),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 125),  INT8_C(   0),  INT8_C(   0),  INT8_C( 118),  INT8_C(  77),  INT8_C(   0),  INT8_C( 116),  INT8_C( 104),
         INT8_C(  12),  INT8_C(   0),  INT8_C(   0),  INT8_C(  44),  INT8_C(  66),  INT8_C( 126),  INT8_C(  51),  INT8_C(  92) } },
    { UINT32_C(3291159275),
      { -INT8_C(  58), -INT8_C(  33), -INT8_C( 122), -INT8_C( 100),  INT8_C(   5),  INT8_C(  93),  INT8_C(  99),  INT8_C( 126),
         INT8_C(  77), -INT8_C(  46),  INT8_C( 100), -INT8_C(  54),  INT8_C(  26), -INT8_C(  58),  INT8_C(  84),  INT8_C( 103),
         INT8_C(  56), -INT8_C(  32), -INT8_C(  48),  INT8_C(  44),  INT8_C( 115),  INT8_C( 121),  INT8_C(   0),  INT8_C(  50),
        -INT8_C(   4), -INT8_C(  50), -INT8_C(  42), -INT8_C(  25), -INT8_C(  24),  INT8_C(   2), -INT8_C(  84), -INT8_C(  82) },
      {  INT8_C(  58),  INT8_C(  33),  INT8_C(   0),  INT8_C( 100),  INT8_C(   0),  INT8_C(  93),  INT8_C(  99),  INT8_C( 126),
         INT8_C(   0),  INT8_C(  46),  INT8_C(   0),  INT8_C(  54),  INT8_C(  26),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  56),  INT8_C(  32),  INT8_C(   0),  INT8_C(  44),  INT8_C(   0),  INT8_C( 121),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  42),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  84),  INT8_C(  82) } },
    { UINT32_C(3863622369),
      { -INT8_C( 113), -INT8_C(  83),  INT8_C( 101), -INT8_C(  36),      INT8_MAX, -INT8_C(  55), -INT8_C(  90), -INT8_C( 102),
        -INT8_C( 113), -INT8_C(   6),  INT8_C(   1), -INT8_C(  57), -INT8_C(  38), -INT8_C(  47), -INT8_C(  13),  INT8_C(  78),
         INT8_C(  75), -INT8_C(  12),      INT8_MIN,  INT8_C(  71), -INT8_C(  62),  INT8_C(  86),  INT8_C(  46), -INT8_C(  86),
         INT8_C(  88), -INT8_C(  38),  INT8_C(  89),  INT8_C(  57),  INT8_C(  12), -INT8_C(  93),  INT8_C(  32), -INT8_C( 100) },
      {  INT8_C( 113),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  55),  INT8_C(  90),  INT8_C( 102),
         INT8_C(   0),  INT8_C(   6),  INT8_C(   0),  INT8_C(   0),  INT8_C(  38),  INT8_C(  47),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  12),  INT8_C(   0),  INT8_C(  71),  INT8_C(   0),  INT8_C(   0),  INT8_C(  46),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  38),  INT8_C(  89),  INT8_C(   0),  INT8_C(   0),  INT8_C(  93),  INT8_C(  32),  INT8_C( 100) } },
    { UINT32_C(3497559377),
      {  INT8_C(  78),  INT8_C(  31),  INT8_C( 106), -INT8_C(  35),  INT8_C(  25),  INT8_C( 108), -INT8_C(  92), -INT8_C(  12),
         INT8_C(  61), -INT8_C( 105),  INT8_C(  66), -INT8_C( 120), -INT8_C( 117), -INT8_C(  62), -INT8_C(  49),  INT8_C(  77),
         INT8_C(  24), -INT8_C(   2), -INT8_C(   8),  INT8_C( 113), -INT8_C(  40),  INT8_C(  81), -INT8_C(  86), -INT8_C(  27),
        -INT8_C(  12), -INT8_C(  54), -INT8_C( 127),  INT8_C(  69),  INT8_C(  79), -INT8_C(   7),  INT8_C(  22), -INT8_C(  99) },
      {  INT8_C(  78),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  25),  INT8_C(   0),  INT8_C(  92),  INT8_C(   0),
         INT8_C(  61),  INT8_C(   0),  INT8_C(  66),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  77),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 113),  INT8_C(  40),  INT8_C(  81),  INT8_C(  86),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  79),  INT8_C(   0),  INT8_C(  22),  INT8_C(  99) } },
    { UINT32_C( 846888984),
      { -INT8_C(  20),  INT8_C(  30),  INT8_C(  38),  INT8_C(  42), -INT8_C(  74),  INT8_C( 104), -INT8_C(  78),  INT8_C(  65),
         INT8_C(  42), -INT8_C( 126), -INT8_C( 113),  INT8_C(  66),      INT8_MIN, -INT8_C( 121), -INT8_C(  77),  INT8_C(  88),
        -INT8_C(  40),  INT8_C(  94),  INT8_C(  61), -INT8_C(  52),  INT8_C(  40), -INT8_C(  66),  INT8_C(  18),  INT8_C( 120),
        -INT8_C(  72),  INT8_C(  40),  INT8_C(  21), -INT8_C(  48), -INT8_C(  88), -INT8_C( 112),  INT8_C(   2), -INT8_C( 107) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  42),  INT8_C(  74),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  88),
         INT8_C(   0),  INT8_C(  94),  INT8_C(   0),  INT8_C(  52),  INT8_C(  40),  INT8_C(  66),  INT8_C(  18),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),  INT8_C(  88),  INT8_C( 112),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C(1690249390),
      { -INT8_C( 112),  INT8_C( 113), -INT8_C(  90), -INT8_C(  70), -INT8_C(  13),  INT8_C(  53), -INT8_C(   3),  INT8_C( 115),
        -INT8_C(  68), -INT8_C(  80), -INT8_C(  52), -INT8_C( 108),  INT8_C(  14),  INT8_C(   9),  INT8_C(  96),  INT8_C(  55),
        -INT8_C(  56),  INT8_C( 114), -INT8_C(  81),      INT8_MIN, -INT8_C( 102), -INT8_C(  60),  INT8_C(  80),  INT8_C(  67),
         INT8_C(  84),  INT8_C(  83), -INT8_C(  40),  INT8_C(   3),  INT8_C( 123), -INT8_C( 105),  INT8_C( 103),  INT8_C(  12) },
      {  INT8_C(   0),  INT8_C( 113),  INT8_C(  90),  INT8_C(  70),  INT8_C(   0),  INT8_C(  53),  INT8_C(   0),  INT8_C( 115),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 108),  INT8_C(   0),  INT8_C(   9),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  56),  INT8_C( 114),  INT8_C(  81),      INT8_MIN,  INT8_C( 102),  INT8_C(  60),  INT8_C(   0),  INT8_C(  67),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),  INT8_C( 105),  INT8_C( 103),  INT8_C(   0) } },
    { UINT32_C(4240837896),
      {  INT8_C(  66), -INT8_C(  61),  INT8_C( 111), -INT8_C(   2),  INT8_C( 116),  INT8_C(  59), -INT8_C( 110), -INT8_C( 126),
         INT8_C(  69), -INT8_C(  13), -INT8_C(  71),  INT8_C(  13),  INT8_C( 101),  INT8_C( 104), -INT8_C( 115),  INT8_C(   0),
         INT8_C(  45), -INT8_C(  35),  INT8_C(  67), -INT8_C( 127),  INT8_C(  48),  INT8_C(  27), -INT8_C( 124), -INT8_C(  84),
        -INT8_C(  78), -INT8_C(  20), -INT8_C(  72), -INT8_C(  70), -INT8_C(   7),  INT8_C( 126), -INT8_C(  74),  INT8_C(  60) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   2),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  69),  INT8_C(   0),  INT8_C(  71),  INT8_C(  13),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  35),  INT8_C(  67),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 124),  INT8_C(  84),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  72),  INT8_C(  70),  INT8_C(   7),  INT8_C( 126),  INT8_C(  74),  INT8_C(  60) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_abs_epi8(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_abs_epi8");
    easysimd_test_x86_assert_equal_i8x32(easysimd_mm256_loadu_epi8(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = easysimd_mm256_maskz_abs_epi8(k, a);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_abs_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t src[16];
    uint16_t k;
    int16_t a[16];
    int16_t r[16];
  } test_vec[] = {
    { {  INT16_C( 11214),  INT16_C(  8918), -INT16_C( 10695), -INT16_C( 15718),  INT16_C( 11833), -INT16_C(  2950), -INT16_C( 31431),  INT16_C(  6353),
         INT16_C( 21649), -INT16_C(   717), -INT16_C(  7634),  INT16_C(  9742),  INT16_C( 25155), -INT16_C( 30159), -INT16_C( 16311),  INT16_C(  6235) },
      UINT16_C(13035),
      {  INT16_C(  9274), -INT16_C( 11256),  INT16_C( 17127),  INT16_C( 24835),  INT16_C( 15414),  INT16_C(  2279),  INT16_C( 30805), -INT16_C( 30628),
        -INT16_C( 30091), -INT16_C( 31894), -INT16_C( 21072), -INT16_C(  7450),  INT16_C( 12088), -INT16_C( 27742), -INT16_C( 29369), -INT16_C( 32315) },
      {  INT16_C(  9274),  INT16_C( 11256), -INT16_C( 10695),  INT16_C( 24835),  INT16_C( 11833),  INT16_C(  2279),  INT16_C( 30805),  INT16_C( 30628),
         INT16_C( 21649),  INT16_C( 31894), -INT16_C(  7634),  INT16_C(  9742),  INT16_C( 12088),  INT16_C( 27742), -INT16_C( 16311),  INT16_C(  6235) } },
    { { -INT16_C( 12622), -INT16_C( 26282),  INT16_C( 22800),  INT16_C( 18170), -INT16_C(  7787), -INT16_C(  5554), -INT16_C( 21670), -INT16_C( 12430),
        -INT16_C(  8907), -INT16_C(  6573),  INT16_C( 14730), -INT16_C( 15672),  INT16_C( 27240), -INT16_C( 20394),  INT16_C(  7159), -INT16_C( 22223) },
      UINT16_C(34793),
      { -INT16_C(  1726),  INT16_C( 15840),  INT16_C( 30272), -INT16_C( 29154),  INT16_C( 30816), -INT16_C( 11463),  INT16_C( 28488), -INT16_C( 25680),
         INT16_C( 14933),  INT16_C(  7636),  INT16_C( 15613),  INT16_C( 21383),  INT16_C( 32492),  INT16_C(  7790),  INT16_C( 22568),  INT16_C( 27301) },
      {  INT16_C(  1726), -INT16_C( 26282),  INT16_C( 22800),  INT16_C( 29154), -INT16_C(  7787),  INT16_C( 11463),  INT16_C( 28488),  INT16_C( 25680),
         INT16_C( 14933),  INT16_C(  7636),  INT16_C( 15613), -INT16_C( 15672),  INT16_C( 27240), -INT16_C( 20394),  INT16_C(  7159),  INT16_C( 27301) } },
    { { -INT16_C( 31151), -INT16_C( 28249), -INT16_C( 14596),  INT16_C( 23584),  INT16_C( 22846), -INT16_C( 31185), -INT16_C(  8248),  INT16_C(  7457),
        -INT16_C(  2790),  INT16_C(  5946), -INT16_C( 16078),  INT16_C(  7786), -INT16_C( 10176),  INT16_C( 26684), -INT16_C(  7632), -INT16_C( 32046) },
      UINT16_C(31336),
      {  INT16_C( 25619),  INT16_C( 13120),  INT16_C( 32448), -INT16_C(  3955),  INT16_C( 21765),  INT16_C(  9935), -INT16_C(  5773), -INT16_C( 21220),
         INT16_C( 19968),  INT16_C( 27247), -INT16_C( 20628), -INT16_C( 22205),  INT16_C( 29463), -INT16_C(  5749), -INT16_C(  3083),  INT16_C(  2403) },
      { -INT16_C( 31151), -INT16_C( 28249), -INT16_C( 14596),  INT16_C(  3955),  INT16_C( 22846),  INT16_C(  9935),  INT16_C(  5773),  INT16_C(  7457),
        -INT16_C(  2790),  INT16_C( 27247), -INT16_C( 16078),  INT16_C( 22205),  INT16_C( 29463),  INT16_C(  5749),  INT16_C(  3083), -INT16_C( 32046) } },
    { { -INT16_C( 23721),  INT16_C(  5948), -INT16_C( 14046),  INT16_C(  9991), -INT16_C( 10465), -INT16_C( 28083),  INT16_C( 27072), -INT16_C( 16065),
        -INT16_C( 20809),  INT16_C(  9259),  INT16_C( 28253),  INT16_C( 29901),  INT16_C( 22754), -INT16_C( 10402), -INT16_C( 16053), -INT16_C( 23840) },
      UINT16_C( 7525),
      { -INT16_C( 30791), -INT16_C( 15898),  INT16_C(  1454), -INT16_C(  1128),  INT16_C( 22679), -INT16_C( 10395),  INT16_C(  7193),  INT16_C( 17797),
        -INT16_C(  7360),  INT16_C(  3507), -INT16_C( 27305), -INT16_C( 19099), -INT16_C( 20371),  INT16_C( 19831), -INT16_C(  9134),  INT16_C(  3178) },
      {  INT16_C( 30791),  INT16_C(  5948),  INT16_C(  1454),  INT16_C(  9991), -INT16_C( 10465),  INT16_C( 10395),  INT16_C(  7193), -INT16_C( 16065),
         INT16_C(  7360),  INT16_C(  9259),  INT16_C( 27305),  INT16_C( 19099),  INT16_C( 20371), -INT16_C( 10402), -INT16_C( 16053), -INT16_C( 23840) } },
    { {  INT16_C( 20835),  INT16_C(  4557),  INT16_C( 25942), -INT16_C(  4596),  INT16_C( 29117), -INT16_C( 10299),  INT16_C( 19086), -INT16_C( 12772),
        -INT16_C( 12499), -INT16_C( 31268),  INT16_C( 16741), -INT16_C( 11718), -INT16_C( 19982),  INT16_C( 17439), -INT16_C( 30067), -INT16_C(  4016) },
      UINT16_C( 7643),
      {  INT16_C( 12545),  INT16_C(  3714),  INT16_C( 16415), -INT16_C(  7041),  INT16_C(  3351),  INT16_C( 13103),  INT16_C( 23772), -INT16_C( 18430),
         INT16_C( 26593),  INT16_C(  7417), -INT16_C(  5319),  INT16_C( 22989),  INT16_C( 23344), -INT16_C( 32541), -INT16_C( 16821),  INT16_C( 19870) },
      {  INT16_C( 12545),  INT16_C(  3714),  INT16_C( 25942),  INT16_C(  7041),  INT16_C(  3351), -INT16_C( 10299),  INT16_C( 23772),  INT16_C( 18430),
         INT16_C( 26593), -INT16_C( 31268),  INT16_C(  5319),  INT16_C( 22989),  INT16_C( 23344),  INT16_C( 17439), -INT16_C( 30067), -INT16_C(  4016) } },
    { {  INT16_C(  8431),  INT16_C(  3931), -INT16_C(  9632),  INT16_C( 30707),  INT16_C(  8936), -INT16_C( 15190), -INT16_C( 21121),  INT16_C( 24700),
         INT16_C( 29972),  INT16_C( 20092),  INT16_C( 19041), -INT16_C( 28249), -INT16_C( 30043), -INT16_C(  4079), -INT16_C( 20664),  INT16_C( 14141) },
      UINT16_C(39120),
      {  INT16_C( 12358),  INT16_C( 14963),  INT16_C( 23464),  INT16_C( 21084), -INT16_C(  9441), -INT16_C( 25601),  INT16_C(  5180), -INT16_C( 18416),
         INT16_C( 29026),  INT16_C(  2306), -INT16_C( 22782),  INT16_C(  5267), -INT16_C(  9320), -INT16_C( 10813), -INT16_C( 27886),  INT16_C( 22894) },
      {  INT16_C(  8431),  INT16_C(  3931), -INT16_C(  9632),  INT16_C( 30707),  INT16_C(  9441), -INT16_C( 15190),  INT16_C(  5180),  INT16_C( 18416),
         INT16_C( 29972),  INT16_C( 20092),  INT16_C( 19041),  INT16_C(  5267),  INT16_C(  9320), -INT16_C(  4079), -INT16_C( 20664),  INT16_C( 22894) } },
    { { -INT16_C(  7740),  INT16_C( 27795), -INT16_C(  4292),  INT16_C( 23486), -INT16_C( 16693),  INT16_C(  2038),  INT16_C(  1746),  INT16_C( 13503),
        -INT16_C( 15752),  INT16_C( 31293), -INT16_C( 12183),  INT16_C(   398),  INT16_C( 21163), -INT16_C( 16937),  INT16_C( 17893), -INT16_C( 22250) },
      UINT16_C(43302),
      {  INT16_C( 25109), -INT16_C( 11111),  INT16_C( 25789), -INT16_C( 19566),  INT16_C( 25707),  INT16_C( 10937),  INT16_C( 12696), -INT16_C( 10772),
         INT16_C( 22188),  INT16_C( 15013),  INT16_C( 20567),  INT16_C( 11916),  INT16_C( 29197),  INT16_C(  9331), -INT16_C( 26341),  INT16_C( 12749) },
      { -INT16_C(  7740),  INT16_C( 11111),  INT16_C( 25789),  INT16_C( 23486), -INT16_C( 16693),  INT16_C( 10937),  INT16_C(  1746),  INT16_C( 13503),
         INT16_C( 22188),  INT16_C( 31293), -INT16_C( 12183),  INT16_C( 11916),  INT16_C( 21163),  INT16_C(  9331),  INT16_C( 17893),  INT16_C( 12749) } },
    { {  INT16_C( 26363), -INT16_C( 18427), -INT16_C( 26678),  INT16_C( 13675),  INT16_C(  9723), -INT16_C( 27808),  INT16_C( 19542),  INT16_C(   616),
         INT16_C(  3490), -INT16_C(  1475), -INT16_C( 13987),  INT16_C( 27176), -INT16_C( 25541),  INT16_C( 22414),  INT16_C( 23605),  INT16_C( 12680) },
      UINT16_C(36290),
      { -INT16_C( 29207),  INT16_C( 21796),  INT16_C(  8130),  INT16_C(  8826), -INT16_C( 12110),  INT16_C(  6767),  INT16_C(  4563),  INT16_C(  4135),
        -INT16_C( 31733),  INT16_C( 13529),  INT16_C(  5614),  INT16_C( 32208),  INT16_C(  1388), -INT16_C(  2855), -INT16_C( 25802),  INT16_C(  8321) },
      {  INT16_C( 26363),  INT16_C( 21796), -INT16_C( 26678),  INT16_C( 13675),  INT16_C(  9723), -INT16_C( 27808),  INT16_C(  4563),  INT16_C(  4135),
         INT16_C( 31733), -INT16_C(  1475),  INT16_C(  5614),  INT16_C( 32208), -INT16_C( 25541),  INT16_C( 22414),  INT16_C( 23605),  INT16_C(  8321) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_abs_epi16(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_abs_epi16");
    easysimd_test_x86_assert_equal_i16x16(easysimd_mm256_loadu_epi16(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_mask_abs_epi16(src, k, a);

    easysimd_test_x86_write_i16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_abs_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t k;
    int16_t a[16];
    int16_t r[16];
  } test_vec[] = {
    { UINT16_C( 9794),
      { -INT16_C( 18886), -INT16_C( 12959), -INT16_C( 22984), -INT16_C(  3392),  INT16_C(  9651),  INT16_C( 16474), -INT16_C( 30939),  INT16_C( 26654),
         INT16_C( 19977), -INT16_C( 29309),  INT16_C( 13818), -INT16_C( 19847),  INT16_C( 29680), -INT16_C( 22991),  INT16_C( 29615), -INT16_C(  5684) },
      {  INT16_C(     0),  INT16_C( 12959),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 30939),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 29309),  INT16_C( 13818),  INT16_C(     0),  INT16_C(     0),  INT16_C( 22991),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(11817),
      {  INT16_C( 25014),  INT16_C( 30420), -INT16_C( 30637), -INT16_C( 20836), -INT16_C( 15928), -INT16_C(  6603),  INT16_C( 15914), -INT16_C( 21195),
         INT16_C( 12236),  INT16_C( 17891), -INT16_C( 11294),  INT16_C(  5048),  INT16_C( 26489),  INT16_C( 18054), -INT16_C( 20655),  INT16_C(  1908) },
      {  INT16_C( 25014),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20836),  INT16_C(     0),  INT16_C(  6603),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 17891),  INT16_C( 11294),  INT16_C(  5048),  INT16_C(     0),  INT16_C( 18054),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(18448),
      {  INT16_C( 25726),  INT16_C(  6864), -INT16_C( 26350),  INT16_C( 18395),  INT16_C(  1407), -INT16_C( 19322),  INT16_C( 21171), -INT16_C( 26908),
        -INT16_C( 14697),  INT16_C( 20585), -INT16_C(  7463),  INT16_C( 24503),  INT16_C(  2088), -INT16_C( 25586),  INT16_C(  7696), -INT16_C( 28955) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1407),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 24503),  INT16_C(     0),  INT16_C(     0),  INT16_C(  7696),  INT16_C(     0) } },
    { UINT16_C(46466),
      { -INT16_C( 27480), -INT16_C( 31922), -INT16_C( 12580),  INT16_C( 25225),  INT16_C( 15490),  INT16_C( 26292),  INT16_C( 19410),  INT16_C( 15148),
         INT16_C(  1435),  INT16_C( 21277),  INT16_C( 18020),  INT16_C( 29275),  INT16_C( 27618), -INT16_C( 14447),  INT16_C(  5113), -INT16_C( 24195) },
      {  INT16_C(     0),  INT16_C( 31922),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 15148),
         INT16_C(  1435),  INT16_C(     0),  INT16_C( 18020),  INT16_C(     0),  INT16_C( 27618),  INT16_C( 14447),  INT16_C(     0),  INT16_C( 24195) } },
    { UINT16_C(52136),
      { -INT16_C( 31707), -INT16_C( 20839),  INT16_C(  7398), -INT16_C( 25878), -INT16_C( 17278), -INT16_C( 20507), -INT16_C( 32265),  INT16_C(  5300),
         INT16_C(  6612),  INT16_C( 12122),  INT16_C( 15755),  INT16_C(  7323), -INT16_C( 27644), -INT16_C( 32464), -INT16_C( 10186),  INT16_C( 23373) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 25878),  INT16_C(     0),  INT16_C( 20507),  INT16_C(     0),  INT16_C(  5300),
         INT16_C(  6612),  INT16_C( 12122),  INT16_C(     0),  INT16_C(  7323),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10186),  INT16_C( 23373) } },
    { UINT16_C(58972),
      {  INT16_C( 16905), -INT16_C(  3326), -INT16_C( 31268), -INT16_C( 15953), -INT16_C( 22988), -INT16_C(  6078),  INT16_C(  5818),  INT16_C(  5377),
        -INT16_C( 29370), -INT16_C(  7854),  INT16_C( 22185), -INT16_C(  9867), -INT16_C( 21544),  INT16_C(  9649),  INT16_C(  3334),  INT16_C(  3851) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 31268),  INT16_C( 15953),  INT16_C( 22988),  INT16_C(     0),  INT16_C(  5818),  INT16_C(     0),
         INT16_C(     0),  INT16_C(  7854),  INT16_C( 22185),  INT16_C(     0),  INT16_C(     0),  INT16_C(  9649),  INT16_C(  3334),  INT16_C(  3851) } },
    { UINT16_C( 3663),
      {  INT16_C( 11010), -INT16_C( 20077), -INT16_C( 14355),  INT16_C( 12119),  INT16_C(  4783), -INT16_C( 20154), -INT16_C( 29657),  INT16_C( 31038),
        -INT16_C(  6291), -INT16_C(  7473), -INT16_C( 22591),  INT16_C( 29326), -INT16_C( 27444), -INT16_C( 10112), -INT16_C( 12380), -INT16_C( 22810) },
      {  INT16_C( 11010),  INT16_C( 20077),  INT16_C( 14355),  INT16_C( 12119),  INT16_C(     0),  INT16_C(     0),  INT16_C( 29657),  INT16_C(     0),
         INT16_C(     0),  INT16_C(  7473),  INT16_C( 22591),  INT16_C( 29326),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(31227),
      { -INT16_C(  6056), -INT16_C( 20672), -INT16_C(  4329),  INT16_C( 24001), -INT16_C(  5984), -INT16_C(  8471),  INT16_C( 22113),  INT16_C( 12742),
        -INT16_C( 30919), -INT16_C( 14376), -INT16_C( 23047),  INT16_C( 31067), -INT16_C(   131),  INT16_C( 25417),  INT16_C( 17574), -INT16_C(   292) },
      {  INT16_C(  6056),  INT16_C( 20672),  INT16_C(     0),  INT16_C( 24001),  INT16_C(  5984),  INT16_C(  8471),  INT16_C( 22113),  INT16_C( 12742),
         INT16_C( 30919),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31067),  INT16_C(   131),  INT16_C( 25417),  INT16_C( 17574),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_abs_epi16(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_abs_epi16");
    easysimd_test_x86_assert_equal_i16x16(easysimd_mm256_loadu_epi16(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_maskz_abs_epi16(k, a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_abs_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[8];
    uint8_t k;
    int32_t a[8];
    int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(   344611544),  INT32_C(  1980624836), -INT32_C(  1836024641), -INT32_C(  1717389426),  INT32_C(   701332539), -INT32_C(    39412335), -INT32_C(   638484573),  INT32_C(  1140421147) },
      UINT8_C( 23),
      {  INT32_C(  1574645358), -INT32_C(  1206038212), -INT32_C(  1740197919), -INT32_C(   824975278),  INT32_C(  1264581805),  INT32_C(   586046627), -INT32_C(  1086470323),  INT32_C(   802587073) },
      {  INT32_C(  1574645358),  INT32_C(  1206038212),  INT32_C(  1740197919), -INT32_C(  1717389426),  INT32_C(  1264581805), -INT32_C(    39412335), -INT32_C(   638484573),  INT32_C(  1140421147) } },
    { { -INT32_C(   343035473), -INT32_C(   475747838), -INT32_C(  1417942439),  INT32_C(  2021215946), -INT32_C(   305932214), -INT32_C(  2129612492),  INT32_C(   960515448),  INT32_C(  2120816334) },
      UINT8_C(199),
      { -INT32_C(  1597412874), -INT32_C(   117855219),  INT32_C(  1975690535), -INT32_C(   138397154), -INT32_C(  1372869123),  INT32_C(   187149757),  INT32_C(    81354989), -INT32_C(  1077192759) },
      {  INT32_C(  1597412874),  INT32_C(   117855219),  INT32_C(  1975690535),  INT32_C(  2021215946), -INT32_C(   305932214), -INT32_C(  2129612492),  INT32_C(    81354989),  INT32_C(  1077192759) } },
    { { -INT32_C(   832596543),  INT32_C(  1774606657),  INT32_C(   484346366), -INT32_C(  1055678781),  INT32_C(   158285644), -INT32_C(   652962068), -INT32_C(  1059197193),  INT32_C(   109095237) },
      UINT8_C( 62),
      {  INT32_C(   947901919),  INT32_C(   607578267),  INT32_C(  1709724615), -INT32_C(  1515083418), -INT32_C(  1349403880), -INT32_C(  1113166897), -INT32_C(   251435448), -INT32_C(   969995802) },
      { -INT32_C(   832596543),  INT32_C(   607578267),  INT32_C(  1709724615),  INT32_C(  1515083418),  INT32_C(  1349403880),  INT32_C(  1113166897), -INT32_C(  1059197193),  INT32_C(   109095237) } },
    { {  INT32_C(  2063511518),  INT32_C(  1587426711), -INT32_C(   289110392),  INT32_C(  1217688879),  INT32_C(    16196912), -INT32_C(   641884784), -INT32_C(   355811325), -INT32_C(  1464796470) },
      UINT8_C(169),
      { -INT32_C(   482336082),  INT32_C(  1198235585), -INT32_C(   663266717),  INT32_C(   319406062),  INT32_C(  1403259318), -INT32_C(  2024374842),  INT32_C(  1095843911), -INT32_C(  1611990544) },
      {  INT32_C(   482336082),  INT32_C(  1587426711), -INT32_C(   289110392),  INT32_C(   319406062),  INT32_C(    16196912),  INT32_C(  2024374842), -INT32_C(   355811325),  INT32_C(  1611990544) } },
    { { -INT32_C(   578671844),  INT32_C(   757395146),  INT32_C(   906337096),  INT32_C(   273223258), -INT32_C(   563876585), -INT32_C(  1301955990), -INT32_C(   336349446), -INT32_C(   880091473) },
      UINT8_C(  9),
      { -INT32_C(    86792180),  INT32_C(  1749156045),  INT32_C(   348354565), -INT32_C(  1356082238), -INT32_C(   249951945),  INT32_C(   619433070), -INT32_C(  1646995777),  INT32_C(  1839636320) },
      {  INT32_C(    86792180),  INT32_C(   757395146),  INT32_C(   906337096),  INT32_C(  1356082238), -INT32_C(   563876585), -INT32_C(  1301955990), -INT32_C(   336349446), -INT32_C(   880091473) } },
    { {  INT32_C(   359102792),  INT32_C(  2138942073), -INT32_C(   460111838),  INT32_C(  1268039188),  INT32_C(   909946568),  INT32_C(   962275194),  INT32_C(  1591160830),  INT32_C(   382434766) },
      UINT8_C(247),
      { -INT32_C(   579851469), -INT32_C(   369102935),  INT32_C(  1107158146),  INT32_C(   638142584), -INT32_C(  1398784124), -INT32_C(   911550054),  INT32_C(   781715632),  INT32_C(   119910100) },
      {  INT32_C(   579851469),  INT32_C(   369102935),  INT32_C(  1107158146),  INT32_C(  1268039188),  INT32_C(  1398784124),  INT32_C(   911550054),  INT32_C(   781715632),  INT32_C(   119910100) } },
    { { -INT32_C(  2082171430),  INT32_C(   124576645),  INT32_C(  1061710535),  INT32_C(   929386930),  INT32_C(   736298385), -INT32_C(  1879732769),  INT32_C(  1774030229),  INT32_C(   359719483) },
      UINT8_C(120),
      {  INT32_C(   956143700),  INT32_C(  1878983685), -INT32_C(  1641988275), -INT32_C(  1439737692), -INT32_C(   930522309), -INT32_C(   581035952), -INT32_C(  1206335530), -INT32_C(  1942999496) },
      { -INT32_C(  2082171430),  INT32_C(   124576645),  INT32_C(  1061710535),  INT32_C(  1439737692),  INT32_C(   930522309),  INT32_C(   581035952),  INT32_C(  1206335530),  INT32_C(   359719483) } },
    { { -INT32_C(   876335674),  INT32_C(  2134557746), -INT32_C(  1491248125), -INT32_C(   263107148), -INT32_C(   122103128), -INT32_C(   908781837),  INT32_C(   377613790), -INT32_C(   492654053) },
      UINT8_C(223),
      {  INT32_C(   722578791),  INT32_C(  1143902440), -INT32_C(    67578450),  INT32_C(    27519015), -INT32_C(  1208706143),  INT32_C(  1570094448), -INT32_C(   243684545), -INT32_C(  1244636338) },
      {  INT32_C(   722578791),  INT32_C(  1143902440),  INT32_C(    67578450),  INT32_C(    27519015),  INT32_C(  1208706143), -INT32_C(   908781837),  INT32_C(   243684545),  INT32_C(  1244636338) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_abs_epi32(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_abs_epi32");
    easysimd_test_x86_assert_equal_i32x8(easysimd_mm256_loadu_epi32(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_mask_abs_epi32(src, k, a);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_abs_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[8];
    int32_t r[8];
  } test_vec[] = {
    { UINT8_C( 44),
      {  INT32_C(   188984604),  INT32_C(  1470931311), -INT32_C(   507934070), -INT32_C(   686101936),  INT32_C(  1741808066),  INT32_C(  1021594172), -INT32_C(   673036397),  INT32_C(  1057218595) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   507934070),  INT32_C(   686101936),  INT32_C(           0),  INT32_C(  1021594172),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(141),
      { -INT32_C(   386118841), -INT32_C(  2123213577), -INT32_C(   137211123),  INT32_C(  1320790381), -INT32_C(   980737670),  INT32_C(  1297663749), -INT32_C(  1989136215),  INT32_C(  2048307251) },
      {  INT32_C(   386118841),  INT32_C(           0),  INT32_C(   137211123),  INT32_C(  1320790381),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2048307251) } },
    { UINT8_C(251),
      {  INT32_C(  1743938067),  INT32_C(   678720469), -INT32_C(   275354811),  INT32_C(  1164567588),  INT32_C(   910896751), -INT32_C(  1210083193), -INT32_C(  1192597496), -INT32_C(  1833736833) },
      {  INT32_C(  1743938067),  INT32_C(   678720469),  INT32_C(           0),  INT32_C(  1164567588),  INT32_C(   910896751),  INT32_C(  1210083193),  INT32_C(  1192597496),  INT32_C(  1833736833) } },
    { UINT8_C(199),
      {  INT32_C(   429717925), -INT32_C(   664877715),  INT32_C(  1073499483), -INT32_C(   441498953),  INT32_C(   627893645),  INT32_C(   757932997),  INT32_C(  1940710926),  INT32_C(  1060781721) },
      {  INT32_C(   429717925),  INT32_C(   664877715),  INT32_C(  1073499483),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1940710926),  INT32_C(  1060781721) } },
    { UINT8_C( 55),
      { -INT32_C(  1666950953),  INT32_C(    83328182), -INT32_C(  1145358727), -INT32_C(   884432667),  INT32_C(   814771469), -INT32_C(  2143371878), -INT32_C(  1474645654), -INT32_C(  1008707092) },
      {  INT32_C(  1666950953),  INT32_C(    83328182),  INT32_C(  1145358727),  INT32_C(           0),  INT32_C(   814771469),  INT32_C(  2143371878),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(177),
      {  INT32_C(    23551876), -INT32_C(  1938134186), -INT32_C(   948816602), -INT32_C(   355189379), -INT32_C(  1971059507), -INT32_C(   185334461),  INT32_C(  2027986207),  INT32_C(    19506045) },
      {  INT32_C(    23551876),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1971059507),  INT32_C(   185334461),  INT32_C(           0),  INT32_C(    19506045) } },
    { UINT8_C(  2),
      { -INT32_C(    61341040), -INT32_C(  1323113092), -INT32_C(  1808864937), -INT32_C(  1033824067), -INT32_C(  1576670307),  INT32_C(  2109864416),  INT32_C(  2113550810), -INT32_C(   226428062) },
      {  INT32_C(           0),  INT32_C(  1323113092),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(254),
      { -INT32_C(  1116016936), -INT32_C(    99341295),  INT32_C(  1941481562), -INT32_C(   183469559),  INT32_C(  2027270783), -INT32_C(  1403891085),  INT32_C(  1192153164),  INT32_C(   675610704) },
      {  INT32_C(           0),  INT32_C(    99341295),  INT32_C(  1941481562),  INT32_C(   183469559),  INT32_C(  2027270783),  INT32_C(  1403891085),  INT32_C(  1192153164),  INT32_C(   675610704) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_abs_epi32(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_abs_epi32");
    easysimd_test_x86_assert_equal_i32x8(easysimd_mm256_loadu_epi32(test_vec[i].r), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_abs_epi32(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_abs_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { easysimd_mm256_set_epi64x(INT64_C( 2298255581870375211), INT64_C(-3544843370875867424),
                             INT64_C( 3174188203889017774), INT64_C(-2855144460944446932)),
      easysimd_mm256_set_epi64x(INT64_C( 2298255581870375211), INT64_C( 3544843370875867424),
                             INT64_C( 3174188203889017774), INT64_C( 2855144460944446932)) },
    { easysimd_mm256_set_epi64x(INT64_C(-2343577668018514218), INT64_C( 6125961421606078258),
                             INT64_C(-3940514899539048661), INT64_C(-1443470135985810906)),
      easysimd_mm256_set_epi64x(INT64_C( 2343577668018514218), INT64_C( 6125961421606078258),
                             INT64_C( 3940514899539048661), INT64_C( 1443470135985810906)) },
    { easysimd_mm256_set_epi64x(INT64_C(-5113251846863269416), INT64_C( 4963302814062391174),
                             INT64_C(-8692429813673586920), INT64_C(-1299515304381535234)),
      easysimd_mm256_set_epi64x(INT64_C( 5113251846863269416), INT64_C( 4963302814062391174),
                             INT64_C( 8692429813673586920), INT64_C( 1299515304381535234)) },
    { easysimd_mm256_set_epi64x(INT64_C( 8282900993993562890), INT64_C( -871234380790935570),
                             INT64_C( 1016547295723275308), INT64_C( 2445109086053031177)),
      easysimd_mm256_set_epi64x(INT64_C( 8282900993993562890), INT64_C(  871234380790935570),
                             INT64_C( 1016547295723275308), INT64_C( 2445109086053031177)) },
    { easysimd_mm256_set_epi64x(INT64_C( 2885698025168517941), INT64_C( 4164132731831874360),
                             INT64_C( 5579124789695570138), INT64_C(-5071075354474953440)),
      easysimd_mm256_set_epi64x(INT64_C( 2885698025168517941), INT64_C( 4164132731831874360),
                             INT64_C( 5579124789695570138), INT64_C( 5071075354474953440)) },
    { easysimd_mm256_set_epi64x(INT64_C(-3829241843042224259), INT64_C(-5265306480458209716),
                             INT64_C( -199503262700073332), INT64_C(-3406476690611433698)),
      easysimd_mm256_set_epi64x(INT64_C( 3829241843042224259), INT64_C( 5265306480458209716),
                             INT64_C(  199503262700073332), INT64_C( 3406476690611433698)) },
    { easysimd_mm256_set_epi64x(INT64_C(-8511077884182051912), INT64_C(-2833485123520542356),
                             INT64_C(-8333607306604449051), INT64_C(-3068466298309072119)),
      easysimd_mm256_set_epi64x(INT64_C( 8511077884182051912), INT64_C( 2833485123520542356),
                             INT64_C( 8333607306604449051), INT64_C( 3068466298309072119)) },
    { easysimd_mm256_set_epi64x(INT64_C( 2822112346803664079), INT64_C(  298455952410199790),
                             INT64_C(  966686671017309845), INT64_C( 9214147743026689710)),
      easysimd_mm256_set_epi64x(INT64_C( 2822112346803664079), INT64_C(  298455952410199790),
                             INT64_C(  966686671017309845), INT64_C( 9214147743026689710)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_abs_epi64(test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_abs_epi64");
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_abs_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i src;
    easysimd__mmask8 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
    { easysimd_mm256_set_epi64x(INT64_C(  623879162816280883), INT64_C(-3225900025883395735),
                             INT64_C(  411040496809638529), INT64_C(-7584870799288762128)),
      UINT8_C( 62),
      easysimd_mm256_set_epi64x(INT64_C(-4625946001506527479), INT64_C(-2791937557159189467),
                             INT64_C(-5209880226959401821), INT64_C(-2130118807554140301)),
      easysimd_mm256_set_epi64x(INT64_C( 4625946001506527479), INT64_C( 2791937557159189467),
                             INT64_C( 5209880226959401821), INT64_C(-7584870799288762128)) },
    { easysimd_mm256_set_epi64x(INT64_C( 8448739575006176562), INT64_C( 3518346377803159044),
                             INT64_C(  844328342996800488), INT64_C( 8434264651311772530)),
      UINT8_C(156),
      easysimd_mm256_set_epi64x(INT64_C(-2671163103984174033), INT64_C( 4562965894666802973),
                             INT64_C(-8366536480676858800), INT64_C( 6120742655549907249)),
      easysimd_mm256_set_epi64x(INT64_C( 2671163103984174033), INT64_C( 4562965894666802973),
                             INT64_C(  844328342996800488), INT64_C( 8434264651311772530)) },
    { easysimd_mm256_set_epi64x(INT64_C(-7191173410794127611), INT64_C( 3688037766287492394),
                             INT64_C( 1547230041795852910), INT64_C( 3059339057736759292)),
      UINT8_C(119),
      easysimd_mm256_set_epi64x(INT64_C(-6542580348328468330), INT64_C(   44667239404533068),
                             INT64_C( 2360079993551421998), INT64_C(  219045572964647829)),
      easysimd_mm256_set_epi64x(INT64_C(-7191173410794127611), INT64_C(   44667239404533068),
                             INT64_C( 2360079993551421998), INT64_C(  219045572964647829)) },
    { easysimd_mm256_set_epi64x(INT64_C( 4128283011258120213), INT64_C( -108361944871310768),
                             INT64_C(-7759705295173963093), INT64_C(-2624902131704570248)),
      UINT8_C( 75),
      easysimd_mm256_set_epi64x(INT64_C(-5879975501041972673), INT64_C( 4967758226257621489),
                             INT64_C( 7728804239548221103), INT64_C( 8515647311939165123)),
      easysimd_mm256_set_epi64x(INT64_C( 5879975501041972673), INT64_C( -108361944871310768),
                             INT64_C( 7728804239548221103), INT64_C( 8515647311939165123)) },
    { easysimd_mm256_set_epi64x(INT64_C(-2790757822212524741), INT64_C( 4593245805939314417),
                             INT64_C(  507611866393274703), INT64_C( 3764810505633876098)),
      UINT8_C(205),
      easysimd_mm256_set_epi64x(INT64_C(-8403106197018531632), INT64_C( 4361313410194959167),
                             INT64_C(-3471819223171854464), INT64_C(-1064109494582275885)),
      easysimd_mm256_set_epi64x(INT64_C( 8403106197018531632), INT64_C( 4361313410194959167),
                             INT64_C(  507611866393274703), INT64_C( 1064109494582275885)) },
    { easysimd_mm256_set_epi64x(INT64_C(-7284244723237547041), INT64_C(-2704891057065522880),
                             INT64_C( 2088703461327613834), INT64_C(-6691637034812206656)),
      UINT8_C( 53),
      easysimd_mm256_set_epi64x(INT64_C( 7087054034507278743), INT64_C(-1904829140491124246),
                             INT64_C(-8979305972799046958), INT64_C(-9028640504948081950)),
      easysimd_mm256_set_epi64x(INT64_C(-7284244723237547041), INT64_C( 1904829140491124246),
                             INT64_C( 2088703461327613834), INT64_C( 9028640504948081950)) },
    { easysimd_mm256_set_epi64x(INT64_C(-6774164690615400180), INT64_C(  169354612478585762),
                             INT64_C(-2560732297798063552), INT64_C(-5440475278226442040)),
      UINT8_C(226),
      easysimd_mm256_set_epi64x(INT64_C( 4140219913643893074), INT64_C( 8233690702404220943),
                             INT64_C(-8119230973072356120), INT64_C( 5725416174942475460)),
      easysimd_mm256_set_epi64x(INT64_C(-6774164690615400180), INT64_C(  169354612478585762),
                             INT64_C( 8119230973072356120), INT64_C(-5440475278226442040)) },
    { easysimd_mm256_set_epi64x(INT64_C(-3618167506666580601), INT64_C(-3565111142066299914),
                             INT64_C( 4487949165835396675), INT64_C( 3493476883354981965)),
      UINT8_C(162),
      easysimd_mm256_set_epi64x(INT64_C(-4298605512042857739), INT64_C(-8701289307647237142),
                             INT64_C(-3191212805157153492), INT64_C( 6189308541761658990)),
      easysimd_mm256_set_epi64x(INT64_C(-3618167506666580601), INT64_C(-3565111142066299914),
                             INT64_C( 3191212805157153492), INT64_C( 3493476883354981965)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_abs_epi64(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_abs_epi64");
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_abs_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m256i a;
    easysimd__m256i r;
  } test_vec[8] = {
  {  UINT8_C( 51),
     easysimd_mm256_set_epi64x(INT64_C(-5558947899438156608), INT64_C(-5328111225624005045),
                             INT64_C(-5266448436194518899), INT64_C(-3023513724998191945)),
     easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(                   0),
                             INT64_C( 5266448436194518899), INT64_C( 3023513724998191945)) },
  {  UINT8_C(192),
     easysimd_mm256_set_epi64x(INT64_C( 1820775813457202726), INT64_C( 8407143534854112894),
                             INT64_C( 1164468631328972115), INT64_C( 3847858140267031773)),
     easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(                   0),
                             INT64_C(                   0), INT64_C(                   0)) },
  {  UINT8_C(150),
     easysimd_mm256_set_epi64x(INT64_C( 1329935347622458589), INT64_C(-6552239731915331500),
                             INT64_C(-5727672039115289046), INT64_C( 2814104926627850068)),
     easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 6552239731915331500),
                             INT64_C( 5727672039115289046), INT64_C(                   0)) },
  {  UINT8_C( 62),
     easysimd_mm256_set_epi64x(INT64_C(-5313485292314620515), INT64_C(-8562444952160280220),
                             INT64_C(-6743839490299418176), INT64_C(  -90311038632227591)),
     easysimd_mm256_set_epi64x(INT64_C( 5313485292314620515), INT64_C( 8562444952160280220),
                             INT64_C( 6743839490299418176), INT64_C(                   0)) },
  {  UINT8_C(146),
     easysimd_mm256_set_epi64x(INT64_C(  134169414195672899), INT64_C(-3653740064081149177),
                             INT64_C(-3907455768376978765), INT64_C(-2357591052420787867)),
     easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(                   0),
                             INT64_C( 3907455768376978765), INT64_C(                   0)) },
  {  UINT8_C( 80),
     easysimd_mm256_set_epi64x(INT64_C(-4112624575699262364), INT64_C( -503713654380207790),
                             INT64_C(-1026806857675583448), INT64_C( 3708988589081863948)),
      easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C(                   0),
                             INT64_C(                   0), INT64_C(                   0)) },
  {  UINT8_C( 70),
     easysimd_mm256_set_epi64x(INT64_C( 5155483861531614212), INT64_C(-1432515770334784350),
                             INT64_C( 5951616937413531378), INT64_C( 3407818380382978160)),
     easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 1432515770334784350),
                             INT64_C( 5951616937413531378), INT64_C(                   0)) },
  {  UINT8_C(215),
     easysimd_mm256_set_epi64x(INT64_C( 1187658108632559622), INT64_C( 3381325771936787939),
                             INT64_C(-4190080085529007037), INT64_C( 1815625056621359018)),
     easysimd_mm256_set_epi64x(INT64_C(                   0), INT64_C( 3381325771936787939),
                             INT64_C( 4190080085529007037), INT64_C( 1815625056621359018)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_abs_epi64(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_abs_epi64");
    easysimd_assert_m256i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_abs_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C( -97), INT8_C(  22), INT8_C(  -8), INT8_C(-101),
                           INT8_C( -18), INT8_C( 124), INT8_C( -73), INT8_C( -35),
                           INT8_C(-107), INT8_C( 125), INT8_C( -49), INT8_C( -14),
                           INT8_C( -55), INT8_C(  -2), INT8_C(   3), INT8_C( -86),
                           INT8_C( -70), INT8_C( -16), INT8_C(  -3), INT8_C( -98),
                           INT8_C( -20), INT8_C( -18), INT8_C( -58), INT8_C( -57),
                           INT8_C( 119), INT8_C(  17), INT8_C( -79), INT8_C(  80),
                           INT8_C(  82), INT8_C(   3), INT8_C( -18), INT8_C( -99),
                           INT8_C(  25), INT8_C(  25), INT8_C(  83), INT8_C(  88),
                           INT8_C( 117), INT8_C(-128), INT8_C(  16), INT8_C( -42),
                           INT8_C( 114), INT8_C(  -1), INT8_C(-110), INT8_C(  53),
                           INT8_C( 127), INT8_C( -61), INT8_C( -68), INT8_C(  74),
                           INT8_C( 103), INT8_C(  92), INT8_C(-115), INT8_C( -60),
                           INT8_C( -23), INT8_C(  82), INT8_C(-123), INT8_C(  21),
                           INT8_C( -37), INT8_C( 119), INT8_C( -39), INT8_C( -31),
                           INT8_C(  25), INT8_C( -69), INT8_C( -57), INT8_C(  30)),
      easysimd_mm512_set_epi8(INT8_C(  97), INT8_C(  22), INT8_C(   8), INT8_C( 101),
                           INT8_C(  18), INT8_C( 124), INT8_C(  73), INT8_C(  35),
                           INT8_C( 107), INT8_C( 125), INT8_C(  49), INT8_C(  14),
                           INT8_C(  55), INT8_C(   2), INT8_C(   3), INT8_C(  86),
                           INT8_C(  70), INT8_C(  16), INT8_C(   3), INT8_C(  98),
                           INT8_C(  20), INT8_C(  18), INT8_C(  58), INT8_C(  57),
                           INT8_C( 119), INT8_C(  17), INT8_C(  79), INT8_C(  80),
                           INT8_C(  82), INT8_C(   3), INT8_C(  18), INT8_C(  99),
                           INT8_C(  25), INT8_C(  25), INT8_C(  83), INT8_C(  88),
                           INT8_C( 117), INT8_C(-128), INT8_C(  16), INT8_C(  42),
                           INT8_C( 114), INT8_C(   1), INT8_C( 110), INT8_C(  53),
                           INT8_C( 127), INT8_C(  61), INT8_C(  68), INT8_C(  74),
                           INT8_C( 103), INT8_C(  92), INT8_C( 115), INT8_C(  60),
                           INT8_C(  23), INT8_C(  82), INT8_C( 123), INT8_C(  21),
                           INT8_C(  37), INT8_C( 119), INT8_C(  39), INT8_C(  31),
                           INT8_C(  25), INT8_C(  69), INT8_C(  57), INT8_C(  30)) },
    { easysimd_mm512_set_epi8(INT8_C( 122), INT8_C(  62), INT8_C( -43), INT8_C( -88),
                           INT8_C(  92), INT8_C(-116), INT8_C(  -6), INT8_C(  36),
                           INT8_C(  10), INT8_C(   2), INT8_C( -66), INT8_C( 108),
                           INT8_C( -38), INT8_C( 112), INT8_C( 123), INT8_C(  87),
                           INT8_C(  99), INT8_C( -46), INT8_C( -53), INT8_C(  41),
                           INT8_C(-105), INT8_C( -98), INT8_C(  18), INT8_C( -12),
                           INT8_C( -82), INT8_C( 126), INT8_C( -77), INT8_C( -19),
                           INT8_C(  18), INT8_C(  16), INT8_C(  35), INT8_C( -10),
                           INT8_C( -58), INT8_C(  48), INT8_C(-120), INT8_C(  38),
                           INT8_C(  62), INT8_C(  17), INT8_C(  33), INT8_C(-120),
                           INT8_C( 106), INT8_C(  25), INT8_C( -91), INT8_C(  15),
                           INT8_C( 101), INT8_C( 114), INT8_C( -46), INT8_C( -58),
                           INT8_C( 113), INT8_C(   4), INT8_C(  50), INT8_C(  42),
                           INT8_C(  -1), INT8_C( -29), INT8_C( -27), INT8_C( -23),
                           INT8_C( -69), INT8_C(  92), INT8_C( -67), INT8_C(  89),
                           INT8_C( -10), INT8_C( -42), INT8_C(  79), INT8_C( 112)),
      easysimd_mm512_set_epi8(INT8_C( 122), INT8_C(  62), INT8_C(  43), INT8_C(  88),
                           INT8_C(  92), INT8_C( 116), INT8_C(   6), INT8_C(  36),
                           INT8_C(  10), INT8_C(   2), INT8_C(  66), INT8_C( 108),
                           INT8_C(  38), INT8_C( 112), INT8_C( 123), INT8_C(  87),
                           INT8_C(  99), INT8_C(  46), INT8_C(  53), INT8_C(  41),
                           INT8_C( 105), INT8_C(  98), INT8_C(  18), INT8_C(  12),
                           INT8_C(  82), INT8_C( 126), INT8_C(  77), INT8_C(  19),
                           INT8_C(  18), INT8_C(  16), INT8_C(  35), INT8_C(  10),
                           INT8_C(  58), INT8_C(  48), INT8_C( 120), INT8_C(  38),
                           INT8_C(  62), INT8_C(  17), INT8_C(  33), INT8_C( 120),
                           INT8_C( 106), INT8_C(  25), INT8_C(  91), INT8_C(  15),
                           INT8_C( 101), INT8_C( 114), INT8_C(  46), INT8_C(  58),
                           INT8_C( 113), INT8_C(   4), INT8_C(  50), INT8_C(  42),
                           INT8_C(   1), INT8_C(  29), INT8_C(  27), INT8_C(  23),
                           INT8_C(  69), INT8_C(  92), INT8_C(  67), INT8_C(  89),
                           INT8_C(  10), INT8_C(  42), INT8_C(  79), INT8_C( 112)) },
    { easysimd_mm512_set_epi8(INT8_C(-115), INT8_C( 121), INT8_C( -28), INT8_C( -32),
                           INT8_C(  39), INT8_C(  97), INT8_C( 104), INT8_C( -44),
                           INT8_C( 120), INT8_C( -11), INT8_C( -74), INT8_C( -63),
                           INT8_C( -24), INT8_C( -35), INT8_C(-108), INT8_C(  -9),
                           INT8_C(  30), INT8_C( -94), INT8_C(  96), INT8_C(-119),
                           INT8_C( -14), INT8_C( -94), INT8_C(  34), INT8_C(-111),
                           INT8_C(  86), INT8_C(  -6), INT8_C(-116), INT8_C(  56),
                           INT8_C(  -2), INT8_C(  -8), INT8_C( -66), INT8_C(  73),
                           INT8_C(-111), INT8_C(  20), INT8_C( 114), INT8_C(  16),
                           INT8_C(  71), INT8_C(  17), INT8_C( -13), INT8_C(-101),
                           INT8_C(  32), INT8_C(  52), INT8_C(  -6), INT8_C( -16),
                           INT8_C(  78), INT8_C(  58), INT8_C(  14), INT8_C( -85),
                           INT8_C( -58), INT8_C( 120), INT8_C( 102), INT8_C(-125),
                           INT8_C(  73), INT8_C(-121), INT8_C(-118), INT8_C( -77),
                           INT8_C(  84), INT8_C(  62), INT8_C( 100), INT8_C(-122),
                           INT8_C( -17), INT8_C(  81), INT8_C( 105), INT8_C( -71)),
      easysimd_mm512_set_epi8(INT8_C( 115), INT8_C( 121), INT8_C(  28), INT8_C(  32),
                           INT8_C(  39), INT8_C(  97), INT8_C( 104), INT8_C(  44),
                           INT8_C( 120), INT8_C(  11), INT8_C(  74), INT8_C(  63),
                           INT8_C(  24), INT8_C(  35), INT8_C( 108), INT8_C(   9),
                           INT8_C(  30), INT8_C(  94), INT8_C(  96), INT8_C( 119),
                           INT8_C(  14), INT8_C(  94), INT8_C(  34), INT8_C( 111),
                           INT8_C(  86), INT8_C(   6), INT8_C( 116), INT8_C(  56),
                           INT8_C(   2), INT8_C(   8), INT8_C(  66), INT8_C(  73),
                           INT8_C( 111), INT8_C(  20), INT8_C( 114), INT8_C(  16),
                           INT8_C(  71), INT8_C(  17), INT8_C(  13), INT8_C( 101),
                           INT8_C(  32), INT8_C(  52), INT8_C(   6), INT8_C(  16),
                           INT8_C(  78), INT8_C(  58), INT8_C(  14), INT8_C(  85),
                           INT8_C(  58), INT8_C( 120), INT8_C( 102), INT8_C( 125),
                           INT8_C(  73), INT8_C( 121), INT8_C( 118), INT8_C(  77),
                           INT8_C(  84), INT8_C(  62), INT8_C( 100), INT8_C( 122),
                           INT8_C(  17), INT8_C(  81), INT8_C( 105), INT8_C(  71)) },
    { easysimd_mm512_set_epi8(INT8_C( 104), INT8_C(  89), INT8_C(  23), INT8_C( -69),
                           INT8_C( -81), INT8_C( -18), INT8_C(-115), INT8_C(  45),
                           INT8_C( 111), INT8_C(  97), INT8_C( -96), INT8_C( -52),
                           INT8_C( 117), INT8_C( -89), INT8_C(  83), INT8_C(  55),
                           INT8_C( -79), INT8_C( -41), INT8_C(  65), INT8_C( -18),
                           INT8_C( -14), INT8_C( -36), INT8_C(  -5), INT8_C(-118),
                           INT8_C( 102), INT8_C(  66), INT8_C(   6), INT8_C(  63),
                           INT8_C(   2), INT8_C(  71), INT8_C( -79), INT8_C( 103),
                           INT8_C(  99), INT8_C(  75), INT8_C(  18), INT8_C(-125),
                           INT8_C(  89), INT8_C(  97), INT8_C( -12), INT8_C( -68),
                           INT8_C( -29), INT8_C(  64), INT8_C(  90), INT8_C( 106),
                           INT8_C( -66), INT8_C(  46), INT8_C( -67), INT8_C(-122),
                           INT8_C(  35), INT8_C(  89), INT8_C(-123), INT8_C(  49),
                           INT8_C(  79), INT8_C(-111), INT8_C( 102), INT8_C(  13),
                           INT8_C(  18), INT8_C(   7), INT8_C(  11), INT8_C( -54),
                           INT8_C(  79), INT8_C( -18), INT8_C(  80), INT8_C(  58)),
      easysimd_mm512_set_epi8(INT8_C( 104), INT8_C(  89), INT8_C(  23), INT8_C(  69),
                           INT8_C(  81), INT8_C(  18), INT8_C( 115), INT8_C(  45),
                           INT8_C( 111), INT8_C(  97), INT8_C(  96), INT8_C(  52),
                           INT8_C( 117), INT8_C(  89), INT8_C(  83), INT8_C(  55),
                           INT8_C(  79), INT8_C(  41), INT8_C(  65), INT8_C(  18),
                           INT8_C(  14), INT8_C(  36), INT8_C(   5), INT8_C( 118),
                           INT8_C( 102), INT8_C(  66), INT8_C(   6), INT8_C(  63),
                           INT8_C(   2), INT8_C(  71), INT8_C(  79), INT8_C( 103),
                           INT8_C(  99), INT8_C(  75), INT8_C(  18), INT8_C( 125),
                           INT8_C(  89), INT8_C(  97), INT8_C(  12), INT8_C(  68),
                           INT8_C(  29), INT8_C(  64), INT8_C(  90), INT8_C( 106),
                           INT8_C(  66), INT8_C(  46), INT8_C(  67), INT8_C( 122),
                           INT8_C(  35), INT8_C(  89), INT8_C( 123), INT8_C(  49),
                           INT8_C(  79), INT8_C( 111), INT8_C( 102), INT8_C(  13),
                           INT8_C(  18), INT8_C(   7), INT8_C(  11), INT8_C(  54),
                           INT8_C(  79), INT8_C(  18), INT8_C(  80), INT8_C(  58)) },
    { easysimd_mm512_set_epi8(INT8_C( -69), INT8_C( -18), INT8_C( -24), INT8_C(  31),
                           INT8_C(-118), INT8_C(  28), INT8_C( 111), INT8_C(   9),
                           INT8_C( -62), INT8_C(   2), INT8_C(  24), INT8_C(  57),
                           INT8_C(  60), INT8_C(  85), INT8_C(-124), INT8_C(   4),
                           INT8_C( -47), INT8_C(  -2), INT8_C( -42), INT8_C(   4),
                           INT8_C(-111), INT8_C(   1), INT8_C(  -7), INT8_C(  49),
                           INT8_C(  87), INT8_C(-117), INT8_C(  70), INT8_C( -68),
                           INT8_C(  92), INT8_C(  73), INT8_C( 108), INT8_C(   6),
                           INT8_C( 108), INT8_C( -36), INT8_C(  61), INT8_C(  29),
                           INT8_C(  87), INT8_C(  64), INT8_C(-117), INT8_C(  17),
                           INT8_C( -12), INT8_C(  46), INT8_C( -75), INT8_C(  42),
                           INT8_C(  80), INT8_C( -38), INT8_C(  85), INT8_C(-124),
                           INT8_C(-126), INT8_C( -12), INT8_C(  41), INT8_C(  12),
                           INT8_C( -57), INT8_C( -47), INT8_C(  80), INT8_C( -60),
                           INT8_C(  24), INT8_C(  89), INT8_C( -45), INT8_C(-122),
                           INT8_C( -52), INT8_C(  21), INT8_C(  54), INT8_C( 124)),
      easysimd_mm512_set_epi8(INT8_C(  69), INT8_C(  18), INT8_C(  24), INT8_C(  31),
                           INT8_C( 118), INT8_C(  28), INT8_C( 111), INT8_C(   9),
                           INT8_C(  62), INT8_C(   2), INT8_C(  24), INT8_C(  57),
                           INT8_C(  60), INT8_C(  85), INT8_C( 124), INT8_C(   4),
                           INT8_C(  47), INT8_C(   2), INT8_C(  42), INT8_C(   4),
                           INT8_C( 111), INT8_C(   1), INT8_C(   7), INT8_C(  49),
                           INT8_C(  87), INT8_C( 117), INT8_C(  70), INT8_C(  68),
                           INT8_C(  92), INT8_C(  73), INT8_C( 108), INT8_C(   6),
                           INT8_C( 108), INT8_C(  36), INT8_C(  61), INT8_C(  29),
                           INT8_C(  87), INT8_C(  64), INT8_C( 117), INT8_C(  17),
                           INT8_C(  12), INT8_C(  46), INT8_C(  75), INT8_C(  42),
                           INT8_C(  80), INT8_C(  38), INT8_C(  85), INT8_C( 124),
                           INT8_C( 126), INT8_C(  12), INT8_C(  41), INT8_C(  12),
                           INT8_C(  57), INT8_C(  47), INT8_C(  80), INT8_C(  60),
                           INT8_C(  24), INT8_C(  89), INT8_C(  45), INT8_C( 122),
                           INT8_C(  52), INT8_C(  21), INT8_C(  54), INT8_C( 124)) },
    { easysimd_mm512_set_epi8(INT8_C(  23), INT8_C( -45), INT8_C( -87), INT8_C(-128),
                           INT8_C(  79), INT8_C(  64), INT8_C( -72), INT8_C( 109),
                           INT8_C(  -1), INT8_C( 120), INT8_C( -18), INT8_C(-122),
                           INT8_C( -56), INT8_C(   0), INT8_C( 100), INT8_C(  60),
                           INT8_C( -78), INT8_C( -63), INT8_C(  26), INT8_C(  35),
                           INT8_C( -65), INT8_C(  72), INT8_C(  38), INT8_C( -77),
                           INT8_C(-123), INT8_C( 106), INT8_C(   7), INT8_C(  83),
                           INT8_C(  87), INT8_C( 105), INT8_C( -86), INT8_C(  65),
                           INT8_C( -41), INT8_C( 111), INT8_C( -74), INT8_C( -72),
                           INT8_C(  30), INT8_C( -92), INT8_C(  62), INT8_C( -69),
                           INT8_C( -56), INT8_C( 120), INT8_C(  86), INT8_C(  20),
                           INT8_C( -82), INT8_C(  72), INT8_C(  45), INT8_C(  66),
                           INT8_C( -71), INT8_C(-128), INT8_C( -35), INT8_C(  10),
                           INT8_C( -92), INT8_C( -41), INT8_C( 102), INT8_C( -89),
                           INT8_C(  47), INT8_C(  44), INT8_C(  12), INT8_C(  18),
                           INT8_C( -29), INT8_C( 113), INT8_C( -21), INT8_C( 122)),
      easysimd_mm512_set_epi8(INT8_C(  23), INT8_C(  45), INT8_C(  87), INT8_C(-128),
                           INT8_C(  79), INT8_C(  64), INT8_C(  72), INT8_C( 109),
                           INT8_C(   1), INT8_C( 120), INT8_C(  18), INT8_C( 122),
                           INT8_C(  56), INT8_C(   0), INT8_C( 100), INT8_C(  60),
                           INT8_C(  78), INT8_C(  63), INT8_C(  26), INT8_C(  35),
                           INT8_C(  65), INT8_C(  72), INT8_C(  38), INT8_C(  77),
                           INT8_C( 123), INT8_C( 106), INT8_C(   7), INT8_C(  83),
                           INT8_C(  87), INT8_C( 105), INT8_C(  86), INT8_C(  65),
                           INT8_C(  41), INT8_C( 111), INT8_C(  74), INT8_C(  72),
                           INT8_C(  30), INT8_C(  92), INT8_C(  62), INT8_C(  69),
                           INT8_C(  56), INT8_C( 120), INT8_C(  86), INT8_C(  20),
                           INT8_C(  82), INT8_C(  72), INT8_C(  45), INT8_C(  66),
                           INT8_C(  71), INT8_C(-128), INT8_C(  35), INT8_C(  10),
                           INT8_C(  92), INT8_C(  41), INT8_C( 102), INT8_C(  89),
                           INT8_C(  47), INT8_C(  44), INT8_C(  12), INT8_C(  18),
                           INT8_C(  29), INT8_C( 113), INT8_C(  21), INT8_C( 122)) },
    { easysimd_mm512_set_epi8(INT8_C(   6), INT8_C( -58), INT8_C( -97), INT8_C(  99),
                           INT8_C(  24), INT8_C( 108), INT8_C( -42), INT8_C( 116),
                           INT8_C( -51), INT8_C(  37), INT8_C(  17), INT8_C(  87),
                           INT8_C( 119), INT8_C(  22), INT8_C(  38), INT8_C( -86),
                           INT8_C(  70), INT8_C( -19), INT8_C( 116), INT8_C(   4),
                           INT8_C( -77), INT8_C( -68), INT8_C(  19), INT8_C( -39),
                           INT8_C(  -4), INT8_C(-120), INT8_C(  84), INT8_C( -27),
                           INT8_C( -68), INT8_C( 120), INT8_C(-117), INT8_C( -33),
                           INT8_C(   3), INT8_C( 109), INT8_C(  85), INT8_C( -14),
                           INT8_C( 121), INT8_C(  30), INT8_C( 108), INT8_C(  -1),
                           INT8_C( 114), INT8_C( -61), INT8_C(  46), INT8_C(  93),
                           INT8_C(  48), INT8_C( -57), INT8_C( -97), INT8_C(-100),
                           INT8_C(  84), INT8_C(   0), INT8_C( -87), INT8_C( -47),
                           INT8_C(  85), INT8_C(   2), INT8_C( 125), INT8_C(  35),
                           INT8_C( -12), INT8_C(  -7), INT8_C(   3), INT8_C(   4),
                           INT8_C(  86), INT8_C( 111), INT8_C( -66), INT8_C(  29)),
      easysimd_mm512_set_epi8(INT8_C(   6), INT8_C(  58), INT8_C(  97), INT8_C(  99),
                           INT8_C(  24), INT8_C( 108), INT8_C(  42), INT8_C( 116),
                           INT8_C(  51), INT8_C(  37), INT8_C(  17), INT8_C(  87),
                           INT8_C( 119), INT8_C(  22), INT8_C(  38), INT8_C(  86),
                           INT8_C(  70), INT8_C(  19), INT8_C( 116), INT8_C(   4),
                           INT8_C(  77), INT8_C(  68), INT8_C(  19), INT8_C(  39),
                           INT8_C(   4), INT8_C( 120), INT8_C(  84), INT8_C(  27),
                           INT8_C(  68), INT8_C( 120), INT8_C( 117), INT8_C(  33),
                           INT8_C(   3), INT8_C( 109), INT8_C(  85), INT8_C(  14),
                           INT8_C( 121), INT8_C(  30), INT8_C( 108), INT8_C(   1),
                           INT8_C( 114), INT8_C(  61), INT8_C(  46), INT8_C(  93),
                           INT8_C(  48), INT8_C(  57), INT8_C(  97), INT8_C( 100),
                           INT8_C(  84), INT8_C(   0), INT8_C(  87), INT8_C(  47),
                           INT8_C(  85), INT8_C(   2), INT8_C( 125), INT8_C(  35),
                           INT8_C(  12), INT8_C(   7), INT8_C(   3), INT8_C(   4),
                           INT8_C(  86), INT8_C( 111), INT8_C(  66), INT8_C(  29)) },
    { easysimd_mm512_set_epi8(INT8_C(  48), INT8_C(  61), INT8_C( 127), INT8_C(  76),
                           INT8_C( -86), INT8_C( 122), INT8_C( -96), INT8_C(-118),
                           INT8_C( -38), INT8_C(  -8), INT8_C(  56), INT8_C(-108),
                           INT8_C(   1), INT8_C(   8), INT8_C(  22), INT8_C(-116),
                           INT8_C( -52), INT8_C(  92), INT8_C(  68), INT8_C( 112),
                           INT8_C( -94), INT8_C( -84), INT8_C(  98), INT8_C( -49),
                           INT8_C( -43), INT8_C( 105), INT8_C(  71), INT8_C(  34),
                           INT8_C(-126), INT8_C(  -5), INT8_C(   5), INT8_C( -61),
                           INT8_C(-125), INT8_C( -31), INT8_C(-128), INT8_C( -41),
                           INT8_C(  82), INT8_C(  17), INT8_C( -47), INT8_C(-121),
                           INT8_C(   0), INT8_C( 118), INT8_C( -18), INT8_C( -96),
                           INT8_C(  45), INT8_C(  28), INT8_C( 105), INT8_C(-104),
                           INT8_C( -15), INT8_C(  24), INT8_C(  94), INT8_C( 103),
                           INT8_C( -54), INT8_C(-112), INT8_C(  15), INT8_C( 123),
                           INT8_C( -27), INT8_C( 121), INT8_C(-118), INT8_C(-112),
                           INT8_C( -70), INT8_C(  97), INT8_C(  58), INT8_C( -42)),
      easysimd_mm512_set_epi8(INT8_C(  48), INT8_C(  61), INT8_C( 127), INT8_C(  76),
                           INT8_C(  86), INT8_C( 122), INT8_C(  96), INT8_C( 118),
                           INT8_C(  38), INT8_C(   8), INT8_C(  56), INT8_C( 108),
                           INT8_C(   1), INT8_C(   8), INT8_C(  22), INT8_C( 116),
                           INT8_C(  52), INT8_C(  92), INT8_C(  68), INT8_C( 112),
                           INT8_C(  94), INT8_C(  84), INT8_C(  98), INT8_C(  49),
                           INT8_C(  43), INT8_C( 105), INT8_C(  71), INT8_C(  34),
                           INT8_C( 126), INT8_C(   5), INT8_C(   5), INT8_C(  61),
                           INT8_C( 125), INT8_C(  31), INT8_C(-128), INT8_C(  41),
                           INT8_C(  82), INT8_C(  17), INT8_C(  47), INT8_C( 121),
                           INT8_C(   0), INT8_C( 118), INT8_C(  18), INT8_C(  96),
                           INT8_C(  45), INT8_C(  28), INT8_C( 105), INT8_C( 104),
                           INT8_C(  15), INT8_C(  24), INT8_C(  94), INT8_C( 103),
                           INT8_C(  54), INT8_C( 112), INT8_C(  15), INT8_C( 123),
                           INT8_C(  27), INT8_C( 121), INT8_C( 118), INT8_C( 112),
                           INT8_C(  70), INT8_C(  97), INT8_C(  58), INT8_C(  42)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_abs_epi8(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_abs_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_abs_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask64 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C( 117), INT8_C(-104), INT8_C( -35), INT8_C( -40),
                           INT8_C(  -1), INT8_C(  43), INT8_C(  10), INT8_C( -45),
                           INT8_C( -42), INT8_C(  80), INT8_C( -69), INT8_C( -15),
                           INT8_C( -14), INT8_C(-122), INT8_C(  60), INT8_C(  93),
                           INT8_C(  23), INT8_C( 122), INT8_C(  10), INT8_C( 108),
                           INT8_C( -24), INT8_C( -65), INT8_C( -39), INT8_C( -98),
                           INT8_C( -57), INT8_C(  -6), INT8_C(  81), INT8_C( -45),
                           INT8_C( -27), INT8_C(  28), INT8_C( -85), INT8_C(  15),
                           INT8_C(-118), INT8_C(  52), INT8_C(  10), INT8_C(-116),
                           INT8_C(  26), INT8_C( -43), INT8_C( -38), INT8_C( -27),
                           INT8_C(  66), INT8_C( -52), INT8_C(   5), INT8_C(  -1),
                           INT8_C( -28), INT8_C(   3), INT8_C( 123), INT8_C(-116),
                           INT8_C( -34), INT8_C( -32), INT8_C(  98), INT8_C( 103),
                           INT8_C( -19), INT8_C(-118), INT8_C( -77), INT8_C( -32),
                           INT8_C(  60), INT8_C( -80), INT8_C(  22), INT8_C( -26),
                           INT8_C(  60), INT8_C( -12), INT8_C( -65), INT8_C(  88)),
      UINT64_C(          2117573942),
      easysimd_mm512_set_epi8(INT8_C(  32), INT8_C(  22), INT8_C(  88), INT8_C( -34),
                           INT8_C(  12), INT8_C(  90), INT8_C(-101), INT8_C(  -4),
                           INT8_C( -14), INT8_C(  42), INT8_C( -87), INT8_C( 105),
                           INT8_C(  22), INT8_C(  34), INT8_C( 113), INT8_C( -72),
                           INT8_C( -40), INT8_C( -70), INT8_C( -24), INT8_C( -97),
                           INT8_C( -68), INT8_C(  -6), INT8_C(  98), INT8_C(-124),
                           INT8_C( -35), INT8_C(  11), INT8_C(-118), INT8_C( -49),
                           INT8_C( -42), INT8_C(  24), INT8_C( -34), INT8_C(  73),
                           INT8_C(  -3), INT8_C( -72), INT8_C(-103), INT8_C(  26),
                           INT8_C( -36), INT8_C(-109), INT8_C(  37), INT8_C(  50),
                           INT8_C(  26), INT8_C(  78), INT8_C(  33), INT8_C(  67),
                           INT8_C(  -8), INT8_C( -66), INT8_C(  29), INT8_C(  31),
                           INT8_C(  34), INT8_C(  40), INT8_C( -67), INT8_C(  86),
                           INT8_C(  38), INT8_C(-128), INT8_C(-106), INT8_C( -15),
                           INT8_C( 100), INT8_C(  53), INT8_C(  42), INT8_C(  55),
                           INT8_C(  87), INT8_C( -15), INT8_C(  -5), INT8_C( -85)),
      easysimd_mm512_set_epi8(INT8_C( 117), INT8_C(-104), INT8_C( -35), INT8_C( -40),
                           INT8_C(  -1), INT8_C(  43), INT8_C(  10), INT8_C( -45),
                           INT8_C( -42), INT8_C(  80), INT8_C( -69), INT8_C( -15),
                           INT8_C( -14), INT8_C(-122), INT8_C(  60), INT8_C(  93),
                           INT8_C(  23), INT8_C( 122), INT8_C(  10), INT8_C( 108),
                           INT8_C( -24), INT8_C( -65), INT8_C( -39), INT8_C( -98),
                           INT8_C( -57), INT8_C(  -6), INT8_C(  81), INT8_C( -45),
                           INT8_C( -27), INT8_C(  28), INT8_C( -85), INT8_C(  15),
                           INT8_C(-118), INT8_C(  72), INT8_C( 103), INT8_C(  26),
                           INT8_C(  36), INT8_C( 109), INT8_C(  37), INT8_C( -27),
                           INT8_C(  66), INT8_C( -52), INT8_C(  33), INT8_C(  67),
                           INT8_C( -28), INT8_C(  66), INT8_C(  29), INT8_C(  31),
                           INT8_C(  34), INT8_C( -32), INT8_C(  98), INT8_C(  86),
                           INT8_C(  38), INT8_C(-128), INT8_C( -77), INT8_C(  15),
                           INT8_C(  60), INT8_C( -80), INT8_C(  42), INT8_C(  55),
                           INT8_C(  60), INT8_C(  15), INT8_C(   5), INT8_C(  88)) },
    { easysimd_mm512_set_epi8(INT8_C( -27), INT8_C(-108), INT8_C(-117), INT8_C( -88),
                           INT8_C(-107), INT8_C(  53), INT8_C( -16), INT8_C(  -1),
                           INT8_C( -92), INT8_C(-119), INT8_C(  17), INT8_C(-122),
                           INT8_C(  22), INT8_C( -13), INT8_C(   7), INT8_C(-126),
                           INT8_C( -24), INT8_C( -51), INT8_C( -29), INT8_C(-114),
                           INT8_C( 100), INT8_C( -53), INT8_C(   0), INT8_C(-112),
                           INT8_C( -80), INT8_C(  89), INT8_C(  91), INT8_C(   1),
                           INT8_C( 102), INT8_C(  -2), INT8_C( -67), INT8_C( -88),
                           INT8_C(  -5), INT8_C( -85), INT8_C(  24), INT8_C(  13),
                           INT8_C(  67), INT8_C(  49), INT8_C(  20), INT8_C( -71),
                           INT8_C( -24), INT8_C(  19), INT8_C( -18), INT8_C(  58),
                           INT8_C( 109), INT8_C(-116), INT8_C(  95), INT8_C(  71),
                           INT8_C(  47), INT8_C( 118), INT8_C( -15), INT8_C( -31),
                           INT8_C( -70), INT8_C( -81), INT8_C(  45), INT8_C(  88),
                           INT8_C( -92), INT8_C(  95), INT8_C(  -3), INT8_C( -29),
                           INT8_C(  20), INT8_C( -86), INT8_C(  -5), INT8_C(  57)),
      UINT64_C(          3796566764),
      easysimd_mm512_set_epi8(INT8_C( -74), INT8_C(   1), INT8_C( -40), INT8_C(  93),
                           INT8_C(  28), INT8_C(  66), INT8_C(  14), INT8_C( 119),
                           INT8_C(  -8), INT8_C(-103), INT8_C( 124), INT8_C( -64),
                           INT8_C(  -5), INT8_C(  73), INT8_C(  83), INT8_C(-107),
                           INT8_C( -64), INT8_C( -31), INT8_C(  11), INT8_C(  45),
                           INT8_C( -14), INT8_C(-110), INT8_C( 100), INT8_C(  -6),
                           INT8_C( -50), INT8_C(-123), INT8_C( -94), INT8_C(  12),
                           INT8_C( -29), INT8_C(-100), INT8_C(  97), INT8_C(-115),
                           INT8_C( 103), INT8_C( -79), INT8_C( 102), INT8_C(  -6),
                           INT8_C( -20), INT8_C(-105), INT8_C(  -6), INT8_C(  69),
                           INT8_C(  19), INT8_C( 102), INT8_C(-126), INT8_C( -17),
                           INT8_C(  26), INT8_C(-105), INT8_C(  91), INT8_C( -38),
                           INT8_C( 106), INT8_C(   8), INT8_C(  85), INT8_C( -66),
                           INT8_C(  40), INT8_C( -49), INT8_C(  10), INT8_C(  15),
                           INT8_C(  30), INT8_C(  97), INT8_C( -48), INT8_C(  26),
                           INT8_C(  77), INT8_C( 104), INT8_C(-118), INT8_C(  49)),
      easysimd_mm512_set_epi8(INT8_C( -27), INT8_C(-108), INT8_C(-117), INT8_C( -88),
                           INT8_C(-107), INT8_C(  53), INT8_C( -16), INT8_C(  -1),
                           INT8_C( -92), INT8_C(-119), INT8_C(  17), INT8_C(-122),
                           INT8_C(  22), INT8_C( -13), INT8_C(   7), INT8_C(-126),
                           INT8_C( -24), INT8_C( -51), INT8_C( -29), INT8_C(-114),
                           INT8_C( 100), INT8_C( -53), INT8_C(   0), INT8_C(-112),
                           INT8_C( -80), INT8_C(  89), INT8_C(  91), INT8_C(   1),
                           INT8_C( 102), INT8_C(  -2), INT8_C( -67), INT8_C( -88),
                           INT8_C( 103), INT8_C(  79), INT8_C( 102), INT8_C(  13),
                           INT8_C(  67), INT8_C(  49), INT8_C(   6), INT8_C( -71),
                           INT8_C( -24), INT8_C( 102), INT8_C( -18), INT8_C(  58),
                           INT8_C(  26), INT8_C(-116), INT8_C(  91), INT8_C(  38),
                           INT8_C(  47), INT8_C( 118), INT8_C( -15), INT8_C( -31),
                           INT8_C( -70), INT8_C( -81), INT8_C(  10), INT8_C(  88),
                           INT8_C(  30), INT8_C(  97), INT8_C(  48), INT8_C( -29),
                           INT8_C(  77), INT8_C( 104), INT8_C(  -5), INT8_C(  57)) },
    { easysimd_mm512_set_epi8(INT8_C(  64), INT8_C(  45), INT8_C( -70), INT8_C(  94),
                           INT8_C( 127), INT8_C( -70), INT8_C( 127), INT8_C( -78),
                           INT8_C( -58), INT8_C(  92), INT8_C( -25), INT8_C(  -8),
                           INT8_C(  21), INT8_C(  89), INT8_C(   8), INT8_C(   1),
                           INT8_C(  85), INT8_C(   5), INT8_C( 111), INT8_C( 109),
                           INT8_C(   6), INT8_C( -27), INT8_C(  18), INT8_C(  62),
                           INT8_C(  -7), INT8_C( 126), INT8_C( -22), INT8_C( -36),
                           INT8_C( -10), INT8_C(  -1), INT8_C(   1), INT8_C( 115),
                           INT8_C(  87), INT8_C(  93), INT8_C( -71), INT8_C(-100),
                           INT8_C( -92), INT8_C( 103), INT8_C( -19), INT8_C(  -4),
                           INT8_C( 126), INT8_C( 112), INT8_C( -72), INT8_C(  45),
                           INT8_C(  61), INT8_C( -10), INT8_C(  68), INT8_C( -93),
                           INT8_C(   5), INT8_C( 127), INT8_C( 109), INT8_C( -62),
                           INT8_C( -89), INT8_C(-117), INT8_C(-126), INT8_C(  52),
                           INT8_C(  -8), INT8_C( -92), INT8_C( -23), INT8_C( -48),
                           INT8_C( 104), INT8_C(-120), INT8_C(  -2), INT8_C(-108)),
      UINT64_C(          2131497860),
      easysimd_mm512_set_epi8(INT8_C(  85), INT8_C( 118), INT8_C( 120), INT8_C( -48),
                           INT8_C( 112), INT8_C(  80), INT8_C( -83), INT8_C(  55),
                           INT8_C(  10), INT8_C(-104), INT8_C(  -7), INT8_C(-106),
                           INT8_C(  -6), INT8_C(   9), INT8_C( -88), INT8_C(  52),
                           INT8_C(  69), INT8_C(  91), INT8_C(-122), INT8_C(  83),
                           INT8_C(  54), INT8_C( -42), INT8_C(   9), INT8_C( 100),
                           INT8_C(  84), INT8_C(  66), INT8_C(  99), INT8_C( -57),
                           INT8_C(  20), INT8_C( -56), INT8_C( -41), INT8_C(  34),
                           INT8_C(  96), INT8_C( 125), INT8_C(  40), INT8_C( -10),
                           INT8_C(  37), INT8_C( -54), INT8_C( -41), INT8_C( 111),
                           INT8_C( -17), INT8_C(  73), INT8_C(  10), INT8_C(  78),
                           INT8_C( -64), INT8_C(  57), INT8_C(  95), INT8_C(  52),
                           INT8_C(-123), INT8_C( 102), INT8_C( -91), INT8_C( -25),
                           INT8_C( -74), INT8_C(  23), INT8_C(-127), INT8_C( -43),
                           INT8_C( 123), INT8_C( -21), INT8_C( -69), INT8_C(  72),
                           INT8_C( -86), INT8_C(  39), INT8_C( -52), INT8_C(  88)),
      easysimd_mm512_set_epi8(INT8_C(  64), INT8_C(  45), INT8_C( -70), INT8_C(  94),
                           INT8_C( 127), INT8_C( -70), INT8_C( 127), INT8_C( -78),
                           INT8_C( -58), INT8_C(  92), INT8_C( -25), INT8_C(  -8),
                           INT8_C(  21), INT8_C(  89), INT8_C(   8), INT8_C(   1),
                           INT8_C(  85), INT8_C(   5), INT8_C( 111), INT8_C( 109),
                           INT8_C(   6), INT8_C( -27), INT8_C(  18), INT8_C(  62),
                           INT8_C(  -7), INT8_C( 126), INT8_C( -22), INT8_C( -36),
                           INT8_C( -10), INT8_C(  -1), INT8_C(   1), INT8_C( 115),
                           INT8_C(  87), INT8_C( 125), INT8_C(  40), INT8_C(  10),
                           INT8_C(  37), INT8_C(  54), INT8_C(  41), INT8_C( 111),
                           INT8_C( 126), INT8_C( 112), INT8_C( -72), INT8_C(  45),
                           INT8_C(  64), INT8_C(  57), INT8_C(  68), INT8_C( -93),
                           INT8_C(   5), INT8_C( 127), INT8_C( 109), INT8_C(  25),
                           INT8_C( -89), INT8_C(-117), INT8_C( 127), INT8_C(  43),
                           INT8_C( 123), INT8_C( -92), INT8_C( -23), INT8_C( -48),
                           INT8_C( 104), INT8_C(  39), INT8_C(  -2), INT8_C(-108)) },
    { easysimd_mm512_set_epi8(INT8_C( -39), INT8_C(-117), INT8_C( -99), INT8_C( -55),
                           INT8_C(   3), INT8_C( -15), INT8_C( 113), INT8_C(  -3),
                           INT8_C( -35), INT8_C( 100), INT8_C( -74), INT8_C(-107),
                           INT8_C(  44), INT8_C( -58), INT8_C(  20), INT8_C(  23),
                           INT8_C( 105), INT8_C( -68), INT8_C( 118), INT8_C( -13),
                           INT8_C( -81), INT8_C(  41), INT8_C( -73), INT8_C(-115),
                           INT8_C(-111), INT8_C(  21), INT8_C(  99), INT8_C( 117),
                           INT8_C( -14), INT8_C(-112), INT8_C(  71), INT8_C(  21),
                           INT8_C(-114), INT8_C( -75), INT8_C(  66), INT8_C(-119),
                           INT8_C( -62), INT8_C( -30), INT8_C(  86), INT8_C(-128),
                           INT8_C( 109), INT8_C(  15), INT8_C( -69), INT8_C(  22),
                           INT8_C( -13), INT8_C(  38), INT8_C( -93), INT8_C( -41),
                           INT8_C(  96), INT8_C(  79), INT8_C( -24), INT8_C( -40),
                           INT8_C(  90), INT8_C(  31), INT8_C( -35), INT8_C(  22),
                           INT8_C(-112), INT8_C( -37), INT8_C(  29), INT8_C(  29),
                           INT8_C(   7), INT8_C(   8), INT8_C( 106), INT8_C( -46)),
      UINT64_C(           127712386),
      easysimd_mm512_set_epi8(INT8_C(  68), INT8_C( 120), INT8_C( -69), INT8_C( -50),
                           INT8_C( 102), INT8_C(-123), INT8_C(  95), INT8_C( 110),
                           INT8_C(  90), INT8_C( -66), INT8_C( -52), INT8_C(  44),
                           INT8_C(-111), INT8_C(  10), INT8_C(-111), INT8_C(  20),
                           INT8_C( -11), INT8_C(-128), INT8_C( -17), INT8_C( -40),
                           INT8_C( -41), INT8_C(   0), INT8_C( -15), INT8_C( 105),
                           INT8_C(  81), INT8_C(   3), INT8_C(  23), INT8_C( 107),
                           INT8_C( -18), INT8_C(  80), INT8_C(-106), INT8_C(  52),
                           INT8_C(  80), INT8_C( 120), INT8_C(  83), INT8_C(-117),
                           INT8_C(  84), INT8_C( -78), INT8_C(  47), INT8_C( -33),
                           INT8_C( 103), INT8_C(  66), INT8_C(  79), INT8_C(  53),
                           INT8_C( -45), INT8_C(  20), INT8_C( 111), INT8_C( -59),
                           INT8_C( -18), INT8_C(  30), INT8_C(  70), INT8_C( -25),
                           INT8_C( -57), INT8_C(  18), INT8_C(  -4), INT8_C( 101),
                           INT8_C(  75), INT8_C(  12), INT8_C(  85), INT8_C(  93),
                           INT8_C( -79), INT8_C( -13), INT8_C(  43), INT8_C(  45)),
      easysimd_mm512_set_epi8(INT8_C( -39), INT8_C(-117), INT8_C( -99), INT8_C( -55),
                           INT8_C(   3), INT8_C( -15), INT8_C( 113), INT8_C(  -3),
                           INT8_C( -35), INT8_C( 100), INT8_C( -74), INT8_C(-107),
                           INT8_C(  44), INT8_C( -58), INT8_C(  20), INT8_C(  23),
                           INT8_C( 105), INT8_C( -68), INT8_C( 118), INT8_C( -13),
                           INT8_C( -81), INT8_C(  41), INT8_C( -73), INT8_C(-115),
                           INT8_C(-111), INT8_C(  21), INT8_C(  99), INT8_C( 117),
                           INT8_C( -14), INT8_C(-112), INT8_C(  71), INT8_C(  21),
                           INT8_C(-114), INT8_C( -75), INT8_C(  66), INT8_C(-119),
                           INT8_C( -62), INT8_C(  78), INT8_C(  47), INT8_C(  33),
                           INT8_C( 103), INT8_C(  15), INT8_C( -69), INT8_C(  53),
                           INT8_C(  45), INT8_C(  20), INT8_C( -93), INT8_C( -41),
                           INT8_C(  18), INT8_C(  79), INT8_C(  70), INT8_C(  25),
                           INT8_C(  57), INT8_C(  18), INT8_C( -35), INT8_C(  22),
                           INT8_C(  75), INT8_C( -37), INT8_C(  29), INT8_C(  29),
                           INT8_C(   7), INT8_C(   8), INT8_C(  43), INT8_C( -46)) },
    { easysimd_mm512_set_epi8(INT8_C( -81), INT8_C(  98), INT8_C(  23), INT8_C(-108),
                           INT8_C(-126), INT8_C(  95), INT8_C( -44), INT8_C( -56),
                           INT8_C(  42), INT8_C(  32), INT8_C( -91), INT8_C(-126),
                           INT8_C( 119), INT8_C(  88), INT8_C( 110), INT8_C(  93),
                           INT8_C(  75), INT8_C( -49), INT8_C( -63), INT8_C( -42),
                           INT8_C(  54), INT8_C( -71), INT8_C(  87), INT8_C(  -1),
                           INT8_C( -25), INT8_C( -60), INT8_C( 102), INT8_C( -98),
                           INT8_C( -95), INT8_C( -34), INT8_C( -46), INT8_C(  94),
                           INT8_C( 118), INT8_C( 127), INT8_C( -62), INT8_C( -70),
                           INT8_C(  80), INT8_C( 125), INT8_C( -12), INT8_C(  33),
                           INT8_C( 110), INT8_C(  -9), INT8_C( -29), INT8_C(-115),
                           INT8_C(-117), INT8_C(  52), INT8_C(-126), INT8_C( -15),
                           INT8_C(-118), INT8_C(-123), INT8_C( -16), INT8_C(  72),
                           INT8_C(  84), INT8_C(  54), INT8_C(  76), INT8_C( -48),
                           INT8_C( -79), INT8_C( 100), INT8_C( -58), INT8_C(  30),
                           INT8_C(  35), INT8_C(  68), INT8_C( -40), INT8_C(   8)),
      UINT64_C(           522030218),
      easysimd_mm512_set_epi8(INT8_C(  -1), INT8_C( -56), INT8_C( -80), INT8_C(  17),
                           INT8_C( 127), INT8_C(  83), INT8_C(  -9), INT8_C(   0),
                           INT8_C(  -1), INT8_C( 117), INT8_C( -15), INT8_C(  26),
                           INT8_C(  30), INT8_C( -32), INT8_C(  47), INT8_C(  99),
                           INT8_C( -59), INT8_C( -81), INT8_C( -58), INT8_C(  71),
                           INT8_C(-119), INT8_C( -65), INT8_C( -78), INT8_C(-101),
                           INT8_C( -14), INT8_C(   4), INT8_C( -24), INT8_C( -95),
                           INT8_C( 106), INT8_C(  31), INT8_C( 104), INT8_C(  20),
                           INT8_C(  65), INT8_C(  -8), INT8_C( -75), INT8_C(-128),
                           INT8_C( -81), INT8_C(  68), INT8_C( -86), INT8_C(  98),
                           INT8_C( -55), INT8_C(  10), INT8_C(  75), INT8_C(  51),
                           INT8_C( -57), INT8_C(-111), INT8_C(  87), INT8_C(  47),
                           INT8_C( -21), INT8_C( 105), INT8_C(  17), INT8_C( 107),
                           INT8_C(-119), INT8_C( -18), INT8_C(-123), INT8_C(  81),
                           INT8_C(  54), INT8_C(-122), INT8_C( -83), INT8_C(  81),
                           INT8_C(  21), INT8_C(  13), INT8_C(   6), INT8_C( -56)),
      easysimd_mm512_set_epi8(INT8_C( -81), INT8_C(  98), INT8_C(  23), INT8_C(-108),
                           INT8_C(-126), INT8_C(  95), INT8_C( -44), INT8_C( -56),
                           INT8_C(  42), INT8_C(  32), INT8_C( -91), INT8_C(-126),
                           INT8_C( 119), INT8_C(  88), INT8_C( 110), INT8_C(  93),
                           INT8_C(  75), INT8_C( -49), INT8_C( -63), INT8_C( -42),
                           INT8_C(  54), INT8_C( -71), INT8_C(  87), INT8_C(  -1),
                           INT8_C( -25), INT8_C( -60), INT8_C( 102), INT8_C( -98),
                           INT8_C( -95), INT8_C( -34), INT8_C( -46), INT8_C(  94),
                           INT8_C( 118), INT8_C( 127), INT8_C( -62), INT8_C(-128),
                           INT8_C(  81), INT8_C(  68), INT8_C(  86), INT8_C(  98),
                           INT8_C( 110), INT8_C(  -9), INT8_C( -29), INT8_C(  51),
                           INT8_C(  57), INT8_C( 111), INT8_C(-126), INT8_C(  47),
                           INT8_C(  21), INT8_C(-123), INT8_C( -16), INT8_C(  72),
                           INT8_C( 119), INT8_C(  18), INT8_C(  76), INT8_C( -48),
                           INT8_C(  54), INT8_C( 100), INT8_C( -58), INT8_C(  30),
                           INT8_C(  21), INT8_C(  68), INT8_C(   6), INT8_C(   8)) },
    { easysimd_mm512_set_epi8(INT8_C(-112), INT8_C( -53), INT8_C(-107), INT8_C(  41),
                           INT8_C( -50), INT8_C( -58), INT8_C(  56), INT8_C(  54),
                           INT8_C(-101), INT8_C(-123), INT8_C(  64), INT8_C( -70),
                           INT8_C( -46), INT8_C(  -1), INT8_C(  70), INT8_C( -46),
                           INT8_C(  96), INT8_C(  45), INT8_C(  57), INT8_C(  -8),
                           INT8_C(  23), INT8_C(  34), INT8_C( -16), INT8_C( -48),
                           INT8_C(  74), INT8_C(  85), INT8_C(-106), INT8_C(  98),
                           INT8_C(  81), INT8_C(-107), INT8_C( -43), INT8_C(  64),
                           INT8_C(-110), INT8_C( 124), INT8_C(-122), INT8_C(-123),
                           INT8_C(  20), INT8_C( 122), INT8_C(  57), INT8_C( -15),
                           INT8_C(  58), INT8_C(  90), INT8_C(-103), INT8_C(  57),
                           INT8_C(  51), INT8_C(-118), INT8_C(  37), INT8_C( -79),
                           INT8_C(  13), INT8_C( 116), INT8_C( -79), INT8_C( -18),
                           INT8_C( -87), INT8_C( -79), INT8_C( -83), INT8_C( -25),
                           INT8_C( -30), INT8_C( -40), INT8_C( 126), INT8_C(  80),
                           INT8_C( -74), INT8_C(  71), INT8_C( -68), INT8_C(  53)),
      UINT64_C(          2821348422),
      easysimd_mm512_set_epi8(INT8_C(-126), INT8_C(  -8), INT8_C(  35), INT8_C( 112),
                           INT8_C( -78), INT8_C(  75), INT8_C( -25), INT8_C(   1),
                           INT8_C( -27), INT8_C( -67), INT8_C(  49), INT8_C(  75),
                           INT8_C( -39), INT8_C( -68), INT8_C( -51), INT8_C(  42),
                           INT8_C( -30), INT8_C(   1), INT8_C( -18), INT8_C(  -4),
                           INT8_C(  39), INT8_C(  85), INT8_C(  69), INT8_C(  68),
                           INT8_C(-113), INT8_C( -38), INT8_C(  28), INT8_C(  83),
                           INT8_C( -31), INT8_C(  61), INT8_C(  37), INT8_C(  67),
                           INT8_C(  46), INT8_C( -43), INT8_C(  32), INT8_C( -73),
                           INT8_C( -26), INT8_C(   2), INT8_C(  -6), INT8_C( 122),
                           INT8_C( -51), INT8_C( 118), INT8_C(   3), INT8_C(  17),
                           INT8_C(  32), INT8_C(  82), INT8_C(  40), INT8_C(   0),
                           INT8_C(  28), INT8_C(  37), INT8_C(  -3), INT8_C( -85),
                           INT8_C( -92), INT8_C(  45), INT8_C( -23), INT8_C( -58),
                           INT8_C(-108), INT8_C(  44), INT8_C(  28), INT8_C(  77),
                           INT8_C(  12), INT8_C(  81), INT8_C(-103), INT8_C(   7)),
      easysimd_mm512_set_epi8(INT8_C(-112), INT8_C( -53), INT8_C(-107), INT8_C(  41),
                           INT8_C( -50), INT8_C( -58), INT8_C(  56), INT8_C(  54),
                           INT8_C(-101), INT8_C(-123), INT8_C(  64), INT8_C( -70),
                           INT8_C( -46), INT8_C(  -1), INT8_C(  70), INT8_C( -46),
                           INT8_C(  96), INT8_C(  45), INT8_C(  57), INT8_C(  -8),
                           INT8_C(  23), INT8_C(  34), INT8_C( -16), INT8_C( -48),
                           INT8_C(  74), INT8_C(  85), INT8_C(-106), INT8_C(  98),
                           INT8_C(  81), INT8_C(-107), INT8_C( -43), INT8_C(  64),
                           INT8_C(  46), INT8_C( 124), INT8_C(  32), INT8_C(-123),
                           INT8_C(  26), INT8_C( 122), INT8_C(  57), INT8_C( -15),
                           INT8_C(  58), INT8_C(  90), INT8_C(   3), INT8_C(  57),
                           INT8_C(  32), INT8_C(-118), INT8_C(  40), INT8_C( -79),
                           INT8_C(  13), INT8_C(  37), INT8_C( -79), INT8_C(  85),
                           INT8_C(  92), INT8_C(  45), INT8_C( -83), INT8_C( -25),
                           INT8_C( -30), INT8_C(  44), INT8_C( 126), INT8_C(  80),
                           INT8_C( -74), INT8_C(  81), INT8_C( 103), INT8_C(  53)) },
    { easysimd_mm512_set_epi8(INT8_C( 115), INT8_C( -13), INT8_C( 104), INT8_C(  83),
                           INT8_C(  80), INT8_C(-118), INT8_C(  34), INT8_C(  48),
                           INT8_C(  50), INT8_C( -65), INT8_C(  88), INT8_C(  76),
                           INT8_C( -17), INT8_C( -86), INT8_C( -68), INT8_C(  75),
                           INT8_C( 121), INT8_C(   9), INT8_C( -63), INT8_C( 106),
                           INT8_C(  93), INT8_C(  44), INT8_C(   0), INT8_C( -33),
                           INT8_C( -53), INT8_C( 101), INT8_C(  76), INT8_C(  37),
                           INT8_C(  94), INT8_C( -32), INT8_C(-104), INT8_C( -20),
                           INT8_C( -48), INT8_C(  45), INT8_C(  88), INT8_C( -93),
                           INT8_C( 104), INT8_C(  42), INT8_C( -99), INT8_C(  59),
                           INT8_C(  90), INT8_C( -69), INT8_C( 107), INT8_C(  16),
                           INT8_C(-118), INT8_C(-119), INT8_C( -60), INT8_C(  51),
                           INT8_C( 126), INT8_C( -78), INT8_C( 114), INT8_C( -75),
                           INT8_C( -75), INT8_C(  19), INT8_C( 113), INT8_C(  84),
                           INT8_C(  47), INT8_C( -83), INT8_C( -26), INT8_C( -38),
                           INT8_C(  64), INT8_C(-106), INT8_C( 107), INT8_C(  56)),
      UINT64_C(          1977462364),
      easysimd_mm512_set_epi8(INT8_C(-106), INT8_C( -34), INT8_C( 105), INT8_C( -49),
                           INT8_C( -33), INT8_C( 121), INT8_C(   0), INT8_C( 127),
                           INT8_C( -65), INT8_C( -90), INT8_C(-123), INT8_C( 112),
                           INT8_C( -57), INT8_C(  77), INT8_C(  42), INT8_C(  34),
                           INT8_C( -12), INT8_C( -47), INT8_C( 117), INT8_C(  40),
                           INT8_C(  42), INT8_C(  16), INT8_C( -26), INT8_C( 122),
                           INT8_C( 122), INT8_C( -37), INT8_C( -98), INT8_C( -20),
                           INT8_C(  86), INT8_C( -87), INT8_C( -90), INT8_C(-112),
                           INT8_C(-115), INT8_C(  79), INT8_C( 123), INT8_C(  33),
                           INT8_C( -55), INT8_C(-125), INT8_C( 102), INT8_C(  59),
                           INT8_C( -57), INT8_C(  19), INT8_C(  -4), INT8_C( -55),
                           INT8_C( -86), INT8_C(  88), INT8_C( -47), INT8_C(  29),
                           INT8_C(-116), INT8_C( -58), INT8_C( 115), INT8_C( -63),
                           INT8_C( -15), INT8_C( -54), INT8_C(  84), INT8_C(  -1),
                           INT8_C(   5), INT8_C( -33), INT8_C( -96), INT8_C(  93),
                           INT8_C(  97), INT8_C( 124), INT8_C(  26), INT8_C( -34)),
      easysimd_mm512_set_epi8(INT8_C( 115), INT8_C( -13), INT8_C( 104), INT8_C(  83),
                           INT8_C(  80), INT8_C(-118), INT8_C(  34), INT8_C(  48),
                           INT8_C(  50), INT8_C( -65), INT8_C(  88), INT8_C(  76),
                           INT8_C( -17), INT8_C( -86), INT8_C( -68), INT8_C(  75),
                           INT8_C( 121), INT8_C(   9), INT8_C( -63), INT8_C( 106),
                           INT8_C(  93), INT8_C(  44), INT8_C(   0), INT8_C( -33),
                           INT8_C( -53), INT8_C( 101), INT8_C(  76), INT8_C(  37),
                           INT8_C(  94), INT8_C( -32), INT8_C(-104), INT8_C( -20),
                           INT8_C( -48), INT8_C(  79), INT8_C( 123), INT8_C(  33),
                           INT8_C( 104), INT8_C( 125), INT8_C( -99), INT8_C(  59),
                           INT8_C(  57), INT8_C(  19), INT8_C( 107), INT8_C(  55),
                           INT8_C(  86), INT8_C(  88), INT8_C( -60), INT8_C(  29),
                           INT8_C( 116), INT8_C( -78), INT8_C( 115), INT8_C( -75),
                           INT8_C(  15), INT8_C(  54), INT8_C(  84), INT8_C(  84),
                           INT8_C(  47), INT8_C(  33), INT8_C( -26), INT8_C(  93),
                           INT8_C(  97), INT8_C( 124), INT8_C( 107), INT8_C(  56)) },
    { easysimd_mm512_set_epi8(INT8_C(   2), INT8_C(  -4), INT8_C( 108), INT8_C(  27),
                           INT8_C( -49), INT8_C(  69), INT8_C( -84), INT8_C(  82),
                           INT8_C(   9), INT8_C(   0), INT8_C(  42), INT8_C( 118),
                           INT8_C(  -3), INT8_C( -67), INT8_C(   6), INT8_C(  30),
                           INT8_C( -88), INT8_C( -69), INT8_C( 118), INT8_C(  36),
                           INT8_C( 110), INT8_C(  81), INT8_C( -37), INT8_C(  36),
                           INT8_C( -74), INT8_C(-109), INT8_C(  47), INT8_C(  12),
                           INT8_C( -29), INT8_C( -81), INT8_C(  76), INT8_C( -22),
                           INT8_C(  91), INT8_C( 125), INT8_C(  98), INT8_C(  17),
                           INT8_C( 115), INT8_C(  58), INT8_C(-107), INT8_C(  90),
                           INT8_C( 115), INT8_C( -24), INT8_C(  83), INT8_C(  17),
                           INT8_C( -11), INT8_C(  20), INT8_C(  81), INT8_C(  54),
                           INT8_C( -59), INT8_C( 112), INT8_C(-102), INT8_C(  13),
                           INT8_C(   8), INT8_C(-105), INT8_C( -27), INT8_C(-127),
                           INT8_C(-112), INT8_C( 125), INT8_C(  21), INT8_C(  55),
                           INT8_C(  24), INT8_C(  58), INT8_C(   7), INT8_C( 127)),
      UINT64_C(           751965274),
      easysimd_mm512_set_epi8(INT8_C(  90), INT8_C(  75), INT8_C( -70), INT8_C(  89),
                           INT8_C(  25), INT8_C( -86), INT8_C( -40), INT8_C(  -9),
                           INT8_C(-119), INT8_C( -19), INT8_C( 110), INT8_C( -26),
                           INT8_C(-126), INT8_C( 124), INT8_C(   6), INT8_C( -11),
                           INT8_C( -92), INT8_C(  66), INT8_C( -68), INT8_C(  20),
                           INT8_C(  35), INT8_C(  35), INT8_C(  58), INT8_C(  98),
                           INT8_C(  84), INT8_C( -34), INT8_C(  36), INT8_C(-124),
                           INT8_C(  32), INT8_C( -74), INT8_C(  73), INT8_C( -74),
                           INT8_C(  77), INT8_C( 116), INT8_C(  50), INT8_C(  82),
                           INT8_C(  68), INT8_C(  72), INT8_C(  23), INT8_C(  32),
                           INT8_C( -54), INT8_C(  82), INT8_C(  53), INT8_C(  71),
                           INT8_C(  22), INT8_C(  92), INT8_C(  42), INT8_C(-123),
                           INT8_C( -41), INT8_C(  34), INT8_C(  75), INT8_C(  63),
                           INT8_C(-117), INT8_C(  23), INT8_C(-115), INT8_C(  66),
                           INT8_C( -90), INT8_C(  99), INT8_C( -73), INT8_C( -19),
                           INT8_C( -43), INT8_C( -64), INT8_C( -21), INT8_C(  20)),
      easysimd_mm512_set_epi8(INT8_C(   2), INT8_C(  -4), INT8_C( 108), INT8_C(  27),
                           INT8_C( -49), INT8_C(  69), INT8_C( -84), INT8_C(  82),
                           INT8_C(   9), INT8_C(   0), INT8_C(  42), INT8_C( 118),
                           INT8_C(  -3), INT8_C( -67), INT8_C(   6), INT8_C(  30),
                           INT8_C( -88), INT8_C( -69), INT8_C( 118), INT8_C(  36),
                           INT8_C( 110), INT8_C(  81), INT8_C( -37), INT8_C(  36),
                           INT8_C( -74), INT8_C(-109), INT8_C(  47), INT8_C(  12),
                           INT8_C( -29), INT8_C( -81), INT8_C(  76), INT8_C( -22),
                           INT8_C(  91), INT8_C( 125), INT8_C(  50), INT8_C(  17),
                           INT8_C(  68), INT8_C(  72), INT8_C(-107), INT8_C(  90),
                           INT8_C(  54), INT8_C(  82), INT8_C(  83), INT8_C(  71),
                           INT8_C( -11), INT8_C(  20), INT8_C(  42), INT8_C(  54),
                           INT8_C( -59), INT8_C( 112), INT8_C(-102), INT8_C(  63),
                           INT8_C(   8), INT8_C(  23), INT8_C( -27), INT8_C(-127),
                           INT8_C(-112), INT8_C(  99), INT8_C(  21), INT8_C(  19),
                           INT8_C(  43), INT8_C(  58), INT8_C(  21), INT8_C( 127)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i src = test_vec[i].src;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_abs_epi8(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_abs_epi8");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_abs_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask64 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT64_C(          1713497089),
      easysimd_mm512_set_epi8(INT8_C(-105), INT8_C(  80), INT8_C( -16), INT8_C(-124),
                           INT8_C( -48), INT8_C(  76), INT8_C( -91), INT8_C(-128),
                           INT8_C(  54), INT8_C(  63), INT8_C( 114), INT8_C( -73),
                           INT8_C( -26), INT8_C( -48), INT8_C( -24), INT8_C( -13),
                           INT8_C(   5), INT8_C( 123), INT8_C( -45), INT8_C( -57),
                           INT8_C(-107), INT8_C(  47), INT8_C(  90), INT8_C( -54),
                           INT8_C(   1), INT8_C( 118), INT8_C(  37), INT8_C(  -7),
                           INT8_C(  83), INT8_C(  31), INT8_C( -23), INT8_C( -20),
                           INT8_C(-104), INT8_C( 114), INT8_C(  63), INT8_C(  25),
                           INT8_C( -80), INT8_C(  17), INT8_C(  37), INT8_C( -44),
                           INT8_C(-112), INT8_C(  41), INT8_C( -18), INT8_C(  86),
                           INT8_C( 114), INT8_C( -23), INT8_C( -86), INT8_C( -99),
                           INT8_C( 114), INT8_C(  25), INT8_C(  94), INT8_C(  34),
                           INT8_C( -48), INT8_C(  -4), INT8_C(-123), INT8_C( -44),
                           INT8_C( -68), INT8_C(  19), INT8_C(  47), INT8_C(-122),
                           INT8_C( 117), INT8_C(  69), INT8_C(-121), INT8_C(  66)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( 114), INT8_C(  63), INT8_C(   0),
                           INT8_C(   0), INT8_C(  17), INT8_C(  37), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  18), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  99),
                           INT8_C( 114), INT8_C(  25), INT8_C(  94), INT8_C(   0),
                           INT8_C(   0), INT8_C(   4), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  66)) },
    { UINT64_C(           549841533),
      easysimd_mm512_set_epi8(INT8_C( -84), INT8_C(  24), INT8_C(  17), INT8_C( -28),
                           INT8_C(  -3), INT8_C(  88), INT8_C(  98), INT8_C( -52),
                           INT8_C( -76), INT8_C( -19), INT8_C( 100), INT8_C(  59),
                           INT8_C( -64), INT8_C( -60), INT8_C( -53), INT8_C(  16),
                           INT8_C(   0), INT8_C( -89), INT8_C(  13), INT8_C(  17),
                           INT8_C( 116), INT8_C(  41), INT8_C(  54), INT8_C(  -8),
                           INT8_C(-112), INT8_C( 109), INT8_C(  94), INT8_C(  19),
                           INT8_C(  46), INT8_C( -55), INT8_C( 103), INT8_C(   7),
                           INT8_C( -15), INT8_C( -12), INT8_C( -22), INT8_C( 127),
                           INT8_C( -48), INT8_C( -83), INT8_C(  -9), INT8_C( -85),
                           INT8_C( -79), INT8_C( -12), INT8_C(  76), INT8_C( -65),
                           INT8_C( -90), INT8_C(  19), INT8_C(  33), INT8_C( -50),
                           INT8_C(  89), INT8_C( -40), INT8_C(-117), INT8_C( 111),
                           INT8_C(  48), INT8_C( 119), INT8_C( -55), INT8_C(  66),
                           INT8_C( 113), INT8_C(  -2), INT8_C( -49), INT8_C(-110),
                           INT8_C( -55), INT8_C(  44), INT8_C( 125), INT8_C( -61)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  22), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  79), INT8_C(  12), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  19), INT8_C(   0), INT8_C(  50),
                           INT8_C(  89), INT8_C(  40), INT8_C( 117), INT8_C(   0),
                           INT8_C(  48), INT8_C(   0), INT8_C(  55), INT8_C(   0),
                           INT8_C(   0), INT8_C(   2), INT8_C(  49), INT8_C( 110),
                           INT8_C(  55), INT8_C(  44), INT8_C(   0), INT8_C(  61)) },
    { UINT64_C(          2304862624),
      easysimd_mm512_set_epi8(INT8_C(  71), INT8_C( -17), INT8_C(   0), INT8_C( -82),
                           INT8_C( -27), INT8_C( 124), INT8_C(  45), INT8_C(  57),
                           INT8_C( 107), INT8_C( -93), INT8_C( -77), INT8_C(  53),
                           INT8_C( 126), INT8_C(  10), INT8_C( 123), INT8_C(-113),
                           INT8_C( -41), INT8_C(-108), INT8_C( -59), INT8_C( -36),
                           INT8_C( -24), INT8_C( -51), INT8_C( -68), INT8_C( -38),
                           INT8_C(  19), INT8_C( 120), INT8_C(-118), INT8_C(  63),
                           INT8_C(  24), INT8_C(  72), INT8_C(  39), INT8_C(  31),
                           INT8_C( -92), INT8_C(  52), INT8_C(  81), INT8_C(  39),
                           INT8_C( -70), INT8_C(  73), INT8_C(  76), INT8_C( 114),
                           INT8_C(  -7), INT8_C(   4), INT8_C( -55), INT8_C( -68),
                           INT8_C( 120), INT8_C(  98), INT8_C(-115), INT8_C( -56),
                           INT8_C(  93), INT8_C(  -2), INT8_C(  78), INT8_C(  16),
                           INT8_C(  88), INT8_C(  71), INT8_C(-112), INT8_C(-118),
                           INT8_C(   4), INT8_C( -88), INT8_C(  76), INT8_C(  88),
                           INT8_C( -97), INT8_C( 107), INT8_C( -28), INT8_C( -59)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  92), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  70), INT8_C(   0), INT8_C(   0), INT8_C( 114),
                           INT8_C(   0), INT8_C(   4), INT8_C(  55), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  56),
                           INT8_C(   0), INT8_C(   2), INT8_C(  78), INT8_C(   0),
                           INT8_C(  88), INT8_C(   0), INT8_C(   0), INT8_C( 118),
                           INT8_C(   4), INT8_C(   0), INT8_C(  76), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C(          2156618221),
      easysimd_mm512_set_epi8(INT8_C( -88), INT8_C( -28), INT8_C( -75), INT8_C(  34),
                           INT8_C( -30), INT8_C(  -1), INT8_C(  52), INT8_C( -92),
                           INT8_C( -85), INT8_C(  43), INT8_C(   9), INT8_C(  24),
                           INT8_C( -64), INT8_C( 107), INT8_C( -57), INT8_C(  38),
                           INT8_C(  95), INT8_C( -18), INT8_C(  11), INT8_C(  96),
                           INT8_C(  -4), INT8_C( -94), INT8_C( 116), INT8_C( -31),
                           INT8_C(  52), INT8_C(  -2), INT8_C(  98), INT8_C(  10),
                           INT8_C(   5), INT8_C(  19), INT8_C( -65), INT8_C(  10),
                           INT8_C(-109), INT8_C(  52), INT8_C( -85), INT8_C( -32),
                           INT8_C(  38), INT8_C(  92), INT8_C(   6), INT8_C( -71),
                           INT8_C( -79), INT8_C(  79), INT8_C( -94), INT8_C( 113),
                           INT8_C(-117), INT8_C(  20), INT8_C( -82), INT8_C(  82),
                           INT8_C(-120), INT8_C( 114), INT8_C( -52), INT8_C( -68),
                           INT8_C( -20), INT8_C( -47), INT8_C( -90), INT8_C( -87),
                           INT8_C(  79), INT8_C( -37), INT8_C(  63), INT8_C( -89),
                           INT8_C( -40), INT8_C( -67), INT8_C( -69), INT8_C(-117)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 109), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  79), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 117), INT8_C(   0), INT8_C(  82), INT8_C(  82),
                           INT8_C(   0), INT8_C( 114), INT8_C(  52), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  87),
                           INT8_C(  79), INT8_C(  37), INT8_C(  63), INT8_C(   0),
                           INT8_C(  40), INT8_C(  67), INT8_C(   0), INT8_C( 117)) },
    { UINT64_C(          2985927056),
      easysimd_mm512_set_epi8(INT8_C(-128), INT8_C(  11), INT8_C( -31), INT8_C( 116),
                           INT8_C( -77), INT8_C(  97), INT8_C(  87), INT8_C(  53),
                           INT8_C( -33), INT8_C(  37), INT8_C(  28), INT8_C(  24),
                           INT8_C(-103), INT8_C(  99), INT8_C( -75), INT8_C(  41),
                           INT8_C(  83), INT8_C(  39), INT8_C( 120), INT8_C( 115),
                           INT8_C( -51), INT8_C( -28), INT8_C( 102), INT8_C( -98),
                           INT8_C( -77), INT8_C( 121), INT8_C(  42), INT8_C( 114),
                           INT8_C(  -1), INT8_C( 112), INT8_C(  17), INT8_C( -31),
                           INT8_C( 108), INT8_C( -27), INT8_C(  66), INT8_C(  23),
                           INT8_C(  69), INT8_C( -90), INT8_C( -46), INT8_C( -91),
                           INT8_C( -81), INT8_C( -87), INT8_C(   1), INT8_C( -11),
                           INT8_C(  84), INT8_C(-117), INT8_C(  79), INT8_C(-110),
                           INT8_C( -44), INT8_C( -30), INT8_C(  33), INT8_C(  53),
                           INT8_C(  64), INT8_C( -16), INT8_C(-111), INT8_C( -41),
                           INT8_C(-102), INT8_C(  13), INT8_C(  97), INT8_C( -55),
                           INT8_C(  19), INT8_C( -16), INT8_C( -68), INT8_C( -83)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( 108), INT8_C(   0), INT8_C(  66), INT8_C(  23),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  91),
                           INT8_C(  81), INT8_C(  87), INT8_C(   1), INT8_C(  11),
                           INT8_C(  84), INT8_C(   0), INT8_C(   0), INT8_C( 110),
                           INT8_C(  44), INT8_C(   0), INT8_C(  33), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  41),
                           INT8_C( 102), INT8_C(   0), INT8_C(   0), INT8_C(  55),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C(           390836854),
      easysimd_mm512_set_epi8(INT8_C(  -6), INT8_C( 127), INT8_C(-110), INT8_C(  -8),
                           INT8_C( 106), INT8_C(  95), INT8_C(-126), INT8_C(-127),
                           INT8_C(-103), INT8_C( -21), INT8_C( -20), INT8_C( -71),
                           INT8_C( 106), INT8_C(  23), INT8_C( -51), INT8_C( -47),
                           INT8_C(-107), INT8_C(  61), INT8_C( -93), INT8_C(  10),
                           INT8_C(   4), INT8_C( 110), INT8_C( -43), INT8_C(  40),
                           INT8_C(  60), INT8_C( -40), INT8_C(  36), INT8_C( -39),
                           INT8_C( -80), INT8_C(-110), INT8_C(  14), INT8_C( -61),
                           INT8_C( -39), INT8_C( -70), INT8_C(-116), INT8_C( -99),
                           INT8_C( -82), INT8_C(-113), INT8_C(-120), INT8_C(-116),
                           INT8_C( -58), INT8_C(  18), INT8_C(  72), INT8_C(  23),
                           INT8_C(-117), INT8_C(-105), INT8_C(  83), INT8_C(   3),
                           INT8_C(-104), INT8_C(  34), INT8_C(  72), INT8_C( -33),
                           INT8_C(  84), INT8_C( -90), INT8_C(-116), INT8_C( -46),
                           INT8_C( -18), INT8_C(  96), INT8_C( -46), INT8_C(-109),
                           INT8_C(-103), INT8_C( -18), INT8_C( -39), INT8_C(  67)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  99),
                           INT8_C(   0), INT8_C( 113), INT8_C( 120), INT8_C( 116),
                           INT8_C(   0), INT8_C(  18), INT8_C(   0), INT8_C(   0),
                           INT8_C( 117), INT8_C(   0), INT8_C(  83), INT8_C(   3),
                           INT8_C( 104), INT8_C(   0), INT8_C(  72), INT8_C(  33),
                           INT8_C(   0), INT8_C(   0), INT8_C( 116), INT8_C(   0),
                           INT8_C(   0), INT8_C(  96), INT8_C(  46), INT8_C( 109),
                           INT8_C(   0), INT8_C(  18), INT8_C(  39), INT8_C(   0)) },
    { UINT64_C(           189869641),
      easysimd_mm512_set_epi8(INT8_C(  28), INT8_C(-101), INT8_C(-104), INT8_C(-117),
                           INT8_C(  24), INT8_C( -55), INT8_C(  82), INT8_C(-100),
                           INT8_C( -42), INT8_C(  62), INT8_C(-113), INT8_C( 110),
                           INT8_C( -92), INT8_C( 127), INT8_C( -92), INT8_C(  20),
                           INT8_C( -35), INT8_C(  35), INT8_C(  30), INT8_C( -86),
                           INT8_C( 120), INT8_C(  91), INT8_C( -69), INT8_C( -49),
                           INT8_C(  19), INT8_C( -87), INT8_C(  42), INT8_C(-110),
                           INT8_C(  68), INT8_C(  97), INT8_C(-125), INT8_C(  75),
                           INT8_C(  30), INT8_C( -54), INT8_C( -38), INT8_C( -20),
                           INT8_C( -96), INT8_C(  84), INT8_C( 108), INT8_C(  24),
                           INT8_C( -54), INT8_C( -26), INT8_C(-125), INT8_C( -53),
                           INT8_C(  48), INT8_C( -78), INT8_C( -96), INT8_C(  82),
                           INT8_C( -16), INT8_C( -68), INT8_C( -65), INT8_C(  28),
                           INT8_C( -82), INT8_C(-116), INT8_C( 119), INT8_C(-113),
                           INT8_C( 102), INT8_C(  90), INT8_C(  86), INT8_C( -14),
                           INT8_C( -49), INT8_C(  71), INT8_C(   2), INT8_C(  28)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  96), INT8_C(   0), INT8_C( 108), INT8_C(  24),
                           INT8_C(   0), INT8_C(  26), INT8_C(   0), INT8_C(  53),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  82),
                           INT8_C(   0), INT8_C(   0), INT8_C(  65), INT8_C(   0),
                           INT8_C(  82), INT8_C( 116), INT8_C( 119), INT8_C(   0),
                           INT8_C(   0), INT8_C(  90), INT8_C(   0), INT8_C(   0),
                           INT8_C(  49), INT8_C(   0), INT8_C(   0), INT8_C(  28)) },
    { UINT64_C(          2755545546),
      easysimd_mm512_set_epi8(INT8_C( -71), INT8_C(  48), INT8_C(  -1), INT8_C( -17),
                           INT8_C( -90), INT8_C(   3), INT8_C( -34), INT8_C(  36),
                           INT8_C( -17), INT8_C( -38), INT8_C( 100), INT8_C( -30),
                           INT8_C( 118), INT8_C(  42), INT8_C( -25), INT8_C( -45),
                           INT8_C(   4), INT8_C(   8), INT8_C(  53), INT8_C(  84),
                           INT8_C(-120), INT8_C(  61), INT8_C(  90), INT8_C( -19),
                           INT8_C(  31), INT8_C(-108), INT8_C( -76), INT8_C(  95),
                           INT8_C( 101), INT8_C( -99), INT8_C( -14), INT8_C(  26),
                           INT8_C( -35), INT8_C( -61), INT8_C(  15), INT8_C(  71),
                           INT8_C( 113), INT8_C( 109), INT8_C(  91), INT8_C(-117),
                           INT8_C(   0), INT8_C( 121), INT8_C(  48), INT8_C( 109),
                           INT8_C(  55), INT8_C( 125), INT8_C(-112), INT8_C(  80),
                           INT8_C(  48), INT8_C(  40), INT8_C(  32), INT8_C( -98),
                           INT8_C(  64), INT8_C( -31), INT8_C( -10), INT8_C(  -6),
                           INT8_C( -40), INT8_C(  37), INT8_C(  76), INT8_C( -51),
                           INT8_C(  27), INT8_C(  -2), INT8_C(-101), INT8_C( -10)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  35), INT8_C(   0), INT8_C(  15), INT8_C(   0),
                           INT8_C(   0), INT8_C( 109), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  48), INT8_C( 109),
                           INT8_C(  55), INT8_C( 125), INT8_C( 112), INT8_C(   0),
                           INT8_C(   0), INT8_C(  40), INT8_C(   0), INT8_C(   0),
                           INT8_C(  64), INT8_C(   0), INT8_C(   0), INT8_C(   6),
                           INT8_C(  40), INT8_C(  37), INT8_C(   0), INT8_C(   0),
                           INT8_C(  27), INT8_C(   0), INT8_C( 101), INT8_C(   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_abs_epi8(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_abs_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_abs_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi16(INT16_C(-24810), INT16_C( -1893), INT16_C( -4484), INT16_C(-18467),
                            INT16_C(-27267), INT16_C(-12302), INT16_C(-13826), INT16_C(   938),
                            INT16_C(-17680), INT16_C(  -610), INT16_C( -4882), INT16_C(-14649),
                            INT16_C( 30481), INT16_C(-20144), INT16_C( 20995), INT16_C( -4451),
                            INT16_C(  6425), INT16_C( 21336), INT16_C( 30080), INT16_C(  4310),
                            INT16_C( 29439), INT16_C(-28107), INT16_C( 32707), INT16_C(-17334),
                            INT16_C( 26460), INT16_C(-29244), INT16_C( -5806), INT16_C(-31467),
                            INT16_C( -9353), INT16_C( -9759), INT16_C(  6587), INT16_C(-14562)),
      easysimd_mm512_set_epi16(INT16_C( 24810), INT16_C(  1893), INT16_C(  4484), INT16_C( 18467),
                            INT16_C( 27267), INT16_C( 12302), INT16_C( 13826), INT16_C(   938),
                            INT16_C( 17680), INT16_C(   610), INT16_C(  4882), INT16_C( 14649),
                            INT16_C( 30481), INT16_C( 20144), INT16_C( 20995), INT16_C(  4451),
                            INT16_C(  6425), INT16_C( 21336), INT16_C( 30080), INT16_C(  4310),
                            INT16_C( 29439), INT16_C( 28107), INT16_C( 32707), INT16_C( 17334),
                            INT16_C( 26460), INT16_C( 29244), INT16_C(  5806), INT16_C( 31467),
                            INT16_C(  9353), INT16_C(  9759), INT16_C(  6587), INT16_C( 14562)) },
    { easysimd_mm512_set_epi16(INT16_C( 31294), INT16_C(-10840), INT16_C( 23692), INT16_C( -1500),
                            INT16_C(  2562), INT16_C(-16788), INT16_C( -9616), INT16_C( 31575),
                            INT16_C( 25554), INT16_C(-13527), INT16_C(-26722), INT16_C(  4852),
                            INT16_C(-20866), INT16_C(-19475), INT16_C(  4624), INT16_C(  9206),
                            INT16_C(-14800), INT16_C(-30682), INT16_C( 15889), INT16_C(  8584),
                            INT16_C( 27161), INT16_C(-23281), INT16_C( 25970), INT16_C(-11578),
                            INT16_C( 28932), INT16_C( 12842), INT16_C(   -29), INT16_C( -6679),
                            INT16_C(-17572), INT16_C(-17063), INT16_C( -2346), INT16_C( 20336)),
      easysimd_mm512_set_epi16(INT16_C( 31294), INT16_C( 10840), INT16_C( 23692), INT16_C(  1500),
                            INT16_C(  2562), INT16_C( 16788), INT16_C(  9616), INT16_C( 31575),
                            INT16_C( 25554), INT16_C( 13527), INT16_C( 26722), INT16_C(  4852),
                            INT16_C( 20866), INT16_C( 19475), INT16_C(  4624), INT16_C(  9206),
                            INT16_C( 14800), INT16_C( 30682), INT16_C( 15889), INT16_C(  8584),
                            INT16_C( 27161), INT16_C( 23281), INT16_C( 25970), INT16_C( 11578),
                            INT16_C( 28932), INT16_C( 12842), INT16_C(    29), INT16_C(  6679),
                            INT16_C( 17572), INT16_C( 17063), INT16_C(  2346), INT16_C( 20336)) },
    { easysimd_mm512_set_epi16(INT16_C(-29319), INT16_C( -6944), INT16_C( 10081), INT16_C( 26836),
                            INT16_C( 30965), INT16_C(-18751), INT16_C( -5923), INT16_C(-27401),
                            INT16_C(  7842), INT16_C( 24713), INT16_C( -3422), INT16_C(  8849),
                            INT16_C( 22266), INT16_C(-29640), INT16_C(  -264), INT16_C(-16823),
                            INT16_C(-28396), INT16_C( 29200), INT16_C( 18193), INT16_C( -3173),
                            INT16_C(  8244), INT16_C( -1296), INT16_C( 20026), INT16_C(  3755),
                            INT16_C(-14728), INT16_C( 26243), INT16_C( 18823), INT16_C(-30029),
                            INT16_C( 21566), INT16_C( 25734), INT16_C( -4271), INT16_C( 27065)),
      easysimd_mm512_set_epi16(INT16_C( 29319), INT16_C(  6944), INT16_C( 10081), INT16_C( 26836),
                            INT16_C( 30965), INT16_C( 18751), INT16_C(  5923), INT16_C( 27401),
                            INT16_C(  7842), INT16_C( 24713), INT16_C(  3422), INT16_C(  8849),
                            INT16_C( 22266), INT16_C( 29640), INT16_C(   264), INT16_C( 16823),
                            INT16_C( 28396), INT16_C( 29200), INT16_C( 18193), INT16_C(  3173),
                            INT16_C(  8244), INT16_C(  1296), INT16_C( 20026), INT16_C(  3755),
                            INT16_C( 14728), INT16_C( 26243), INT16_C( 18823), INT16_C( 30029),
                            INT16_C( 21566), INT16_C( 25734), INT16_C(  4271), INT16_C( 27065)) },
    { easysimd_mm512_set_epi16(INT16_C( 26713), INT16_C(  6075), INT16_C(-20498), INT16_C(-29395),
                            INT16_C( 28513), INT16_C(-24372), INT16_C( 30119), INT16_C( 21303),
                            INT16_C(-20009), INT16_C( 16878), INT16_C( -3364), INT16_C( -1142),
                            INT16_C( 26178), INT16_C(  1599), INT16_C(   583), INT16_C(-20121),
                            INT16_C( 25419), INT16_C(  4739), INT16_C( 22881), INT16_C( -2884),
                            INT16_C( -7360), INT16_C( 23146), INT16_C(-16850), INT16_C(-17018),
                            INT16_C(  9049), INT16_C(-31439), INT16_C( 20369), INT16_C( 26125),
                            INT16_C(  4615), INT16_C(  3018), INT16_C( 20462), INT16_C( 20538)),
      easysimd_mm512_set_epi16(INT16_C( 26713), INT16_C(  6075), INT16_C( 20498), INT16_C( 29395),
                            INT16_C( 28513), INT16_C( 24372), INT16_C( 30119), INT16_C( 21303),
                            INT16_C( 20009), INT16_C( 16878), INT16_C(  3364), INT16_C(  1142),
                            INT16_C( 26178), INT16_C(  1599), INT16_C(   583), INT16_C( 20121),
                            INT16_C( 25419), INT16_C(  4739), INT16_C( 22881), INT16_C(  2884),
                            INT16_C(  7360), INT16_C( 23146), INT16_C( 16850), INT16_C( 17018),
                            INT16_C(  9049), INT16_C( 31439), INT16_C( 20369), INT16_C( 26125),
                            INT16_C(  4615), INT16_C(  3018), INT16_C( 20462), INT16_C( 20538)) },
    { easysimd_mm512_set_epi16(INT16_C(-17426), INT16_C( -6113), INT16_C(-30180), INT16_C( 28425),
                            INT16_C(-15870), INT16_C(  6201), INT16_C( 15445), INT16_C(-31740),
                            INT16_C(-11778), INT16_C(-10748), INT16_C(-28415), INT16_C( -1743),
                            INT16_C( 22411), INT16_C( 18108), INT16_C( 23625), INT16_C( 27654),
                            INT16_C( 27868), INT16_C( 15645), INT16_C( 22336), INT16_C(-29935),
                            INT16_C( -3026), INT16_C(-19158), INT16_C( 20698), INT16_C( 21892),
                            INT16_C(-32012), INT16_C( 10508), INT16_C(-14383), INT16_C( 20676),
                            INT16_C(  6233), INT16_C(-11386), INT16_C(-13291), INT16_C( 13948)),
      easysimd_mm512_set_epi16(INT16_C( 17426), INT16_C(  6113), INT16_C( 30180), INT16_C( 28425),
                            INT16_C( 15870), INT16_C(  6201), INT16_C( 15445), INT16_C( 31740),
                            INT16_C( 11778), INT16_C( 10748), INT16_C( 28415), INT16_C(  1743),
                            INT16_C( 22411), INT16_C( 18108), INT16_C( 23625), INT16_C( 27654),
                            INT16_C( 27868), INT16_C( 15645), INT16_C( 22336), INT16_C( 29935),
                            INT16_C(  3026), INT16_C( 19158), INT16_C( 20698), INT16_C( 21892),
                            INT16_C( 32012), INT16_C( 10508), INT16_C( 14383), INT16_C( 20676),
                            INT16_C(  6233), INT16_C( 11386), INT16_C( 13291), INT16_C( 13948)) },
    { easysimd_mm512_set_epi16(INT16_C(  6099), INT16_C(-22144), INT16_C( 20288), INT16_C(-18323),
                            INT16_C(  -136), INT16_C( -4474), INT16_C(-14336), INT16_C( 25660),
                            INT16_C(-19775), INT16_C(  6691), INT16_C(-16568), INT16_C(  9907),
                            INT16_C(-31382), INT16_C(  1875), INT16_C( 22377), INT16_C(-21951),
                            INT16_C(-10385), INT16_C(-18760), INT16_C(  7844), INT16_C( 16059),
                            INT16_C(-14216), INT16_C( 22036), INT16_C(-20920), INT16_C( 11586),
                            INT16_C(-18048), INT16_C( -8950), INT16_C(-23337), INT16_C( 26279),
                            INT16_C( 12076), INT16_C(  3090), INT16_C( -7311), INT16_C( -5254)),
      easysimd_mm512_set_epi16(INT16_C(  6099), INT16_C( 22144), INT16_C( 20288), INT16_C( 18323),
                            INT16_C(   136), INT16_C(  4474), INT16_C( 14336), INT16_C( 25660),
                            INT16_C( 19775), INT16_C(  6691), INT16_C( 16568), INT16_C(  9907),
                            INT16_C( 31382), INT16_C(  1875), INT16_C( 22377), INT16_C( 21951),
                            INT16_C( 10385), INT16_C( 18760), INT16_C(  7844), INT16_C( 16059),
                            INT16_C( 14216), INT16_C( 22036), INT16_C( 20920), INT16_C( 11586),
                            INT16_C( 18048), INT16_C(  8950), INT16_C( 23337), INT16_C( 26279),
                            INT16_C( 12076), INT16_C(  3090), INT16_C(  7311), INT16_C(  5254)) },
    { easysimd_mm512_set_epi16(INT16_C(  1734), INT16_C(-24733), INT16_C(  6252), INT16_C(-10636),
                            INT16_C(-13019), INT16_C(  4439), INT16_C( 30486), INT16_C(  9898),
                            INT16_C( 18157), INT16_C( 29700), INT16_C(-19524), INT16_C(  5081),
                            INT16_C(  -888), INT16_C( 21733), INT16_C(-17288), INT16_C(-29729),
                            INT16_C(   877), INT16_C( 22002), INT16_C( 31006), INT16_C( 27903),
                            INT16_C( 29379), INT16_C( 11869), INT16_C( 12487), INT16_C(-24676),
                            INT16_C( 21504), INT16_C(-22063), INT16_C( 21762), INT16_C( 32035),
                            INT16_C( -2823), INT16_C(   772), INT16_C( 22127), INT16_C(-16867)),
      easysimd_mm512_set_epi16(INT16_C(  1734), INT16_C( 24733), INT16_C(  6252), INT16_C( 10636),
                            INT16_C( 13019), INT16_C(  4439), INT16_C( 30486), INT16_C(  9898),
                            INT16_C( 18157), INT16_C( 29700), INT16_C( 19524), INT16_C(  5081),
                            INT16_C(   888), INT16_C( 21733), INT16_C( 17288), INT16_C( 29729),
                            INT16_C(   877), INT16_C( 22002), INT16_C( 31006), INT16_C( 27903),
                            INT16_C( 29379), INT16_C( 11869), INT16_C( 12487), INT16_C( 24676),
                            INT16_C( 21504), INT16_C( 22063), INT16_C( 21762), INT16_C( 32035),
                            INT16_C(  2823), INT16_C(   772), INT16_C( 22127), INT16_C( 16867)) },
    { easysimd_mm512_set_epi16(INT16_C( 12349), INT16_C( 32588), INT16_C(-21894), INT16_C(-24438),
                            INT16_C( -9480), INT16_C( 14484), INT16_C(   264), INT16_C(  5772),
                            INT16_C(-13220), INT16_C( 17520), INT16_C(-23892), INT16_C( 25295),
                            INT16_C(-10903), INT16_C( 18210), INT16_C(-32005), INT16_C(  1475),
                            INT16_C(-31775), INT16_C(-32553), INT16_C( 21009), INT16_C(-11897),
                            INT16_C(   118), INT16_C( -4448), INT16_C( 11548), INT16_C( 27032),
                            INT16_C( -3816), INT16_C( 24167), INT16_C(-13680), INT16_C(  3963),
                            INT16_C( -6791), INT16_C(-30064), INT16_C(-17823), INT16_C( 15062)),
      easysimd_mm512_set_epi16(INT16_C( 12349), INT16_C( 32588), INT16_C( 21894), INT16_C( 24438),
                            INT16_C(  9480), INT16_C( 14484), INT16_C(   264), INT16_C(  5772),
                            INT16_C( 13220), INT16_C( 17520), INT16_C( 23892), INT16_C( 25295),
                            INT16_C( 10903), INT16_C( 18210), INT16_C( 32005), INT16_C(  1475),
                            INT16_C( 31775), INT16_C( 32553), INT16_C( 21009), INT16_C( 11897),
                            INT16_C(   118), INT16_C(  4448), INT16_C( 11548), INT16_C( 27032),
                            INT16_C(  3816), INT16_C( 24167), INT16_C( 13680), INT16_C(  3963),
                            INT16_C(  6791), INT16_C( 30064), INT16_C( 17823), INT16_C( 15062)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_abs_epi16(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_abs_epi16");
    easysimd_assert_m512i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_abs_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C(  5316), -INT16_C( 21434),  INT16_C( 23082), -INT16_C( 11070), -INT16_C( 21411), -INT16_C(  1395),  INT16_C( 31798), -INT16_C( 18159),
        -INT16_C( 19036), -INT16_C( 29365),  INT16_C( 11378),  INT16_C( 16016), -INT16_C( 11452), -INT16_C( 20127),  INT16_C( 12785),  INT16_C( 11729),
         INT16_C(  5916),  INT16_C( 18393), -INT16_C( 25742), -INT16_C( 12517), -INT16_C( 22456),  INT16_C( 32458), -INT16_C(  9436), -INT16_C( 14025),
        -INT16_C( 32112),  INT16_C(   598), -INT16_C(  6225), -INT16_C(  3264), -INT16_C( 24134), -INT16_C( 21596),  INT16_C( 30418), -INT16_C(  4136) },
      UINT32_C(4281774477),
      {  INT16_C( 20812), -INT16_C( 27441), -INT16_C( 26119),  INT16_C(  7442),  INT16_C( 18804),  INT16_C(  1254),  INT16_C( 15820),  INT16_C( 31494),
         INT16_C( 17956), -INT16_C(  8594),  INT16_C(  4839), -INT16_C( 18039),  INT16_C( 24968),  INT16_C(  5800), -INT16_C(  8686),  INT16_C( 24085),
        -INT16_C(  7121),  INT16_C( 10483),  INT16_C(  1405), -INT16_C(  3770),  INT16_C( 11343),  INT16_C(  7157), -INT16_C(  1175), -INT16_C( 29290),
         INT16_C(  1089),  INT16_C( 10347), -INT16_C(  3050), -INT16_C( 24606), -INT16_C( 30123),  INT16_C( 26549), -INT16_C( 13719), -INT16_C( 26426) },
      {  INT16_C( 20812), -INT16_C( 21434),  INT16_C( 26119),  INT16_C(  7442), -INT16_C( 21411), -INT16_C(  1395),  INT16_C( 31798),  INT16_C( 31494),
         INT16_C( 17956), -INT16_C( 29365),  INT16_C( 11378),  INT16_C( 16016),  INT16_C( 24968),  INT16_C(  5800),  INT16_C( 12785),  INT16_C( 24085),
         INT16_C(  5916),  INT16_C( 10483),  INT16_C(  1405), -INT16_C( 12517),  INT16_C( 11343),  INT16_C(  7157), -INT16_C(  9436), -INT16_C( 14025),
         INT16_C(  1089),  INT16_C( 10347),  INT16_C(  3050),  INT16_C( 24606),  INT16_C( 30123),  INT16_C( 26549),  INT16_C( 13719),  INT16_C( 26426) } },
    { { -INT16_C( 18001),  INT16_C( 11457),  INT16_C(  1982),  INT16_C(  3358),  INT16_C(  4915), -INT16_C( 25304), -INT16_C( 16881),  INT16_C( 20522),
        -INT16_C( 26942), -INT16_C(  9863),  INT16_C( 23434), -INT16_C(  8072),  INT16_C( 11749),  INT16_C( 20039),  INT16_C(  3575), -INT16_C( 22809),
        -INT16_C( 22330), -INT16_C( 31277), -INT16_C(  3665), -INT16_C(  7534), -INT16_C( 17660),  INT16_C(  4991), -INT16_C( 21895),  INT16_C( 15460),
        -INT16_C(  8896), -INT16_C( 13803), -INT16_C( 29384),  INT16_C(  7594), -INT16_C(  3398), -INT16_C( 20116),  INT16_C( 21503), -INT16_C( 14760) },
      UINT32_C(2857053179),
      { -INT16_C(  8932),  INT16_C(  8332),  INT16_C(  3224),  INT16_C(  4660), -INT16_C( 26442), -INT16_C(  2482),  INT16_C( 25461), -INT16_C( 21056),
         INT16_C( 27632), -INT16_C( 21814),  INT16_C( 13917),  INT16_C( 23643), -INT16_C( 19575), -INT16_C( 31710),  INT16_C( 28126), -INT16_C(  1490),
        -INT16_C( 17589), -INT16_C(  7397),  INT16_C( 20423),  INT16_C( 32245),  INT16_C( 17383),  INT16_C( 23667),  INT16_C( 13222), -INT16_C( 27127),
        -INT16_C( 11362), -INT16_C(  1216), -INT16_C( 25590), -INT16_C( 27816),  INT16_C( 31311),  INT16_C( 11800),  INT16_C( 18152),  INT16_C( 13096) },
      {  INT16_C(  8932),  INT16_C(  8332),  INT16_C(  1982),  INT16_C(  4660),  INT16_C( 26442),  INT16_C(  2482),  INT16_C( 25461),  INT16_C( 21056),
         INT16_C( 27632),  INT16_C( 21814),  INT16_C( 23434),  INT16_C( 23643),  INT16_C( 11749),  INT16_C( 31710),  INT16_C(  3575), -INT16_C( 22809),
         INT16_C( 17589),  INT16_C(  7397), -INT16_C(  3665),  INT16_C( 32245), -INT16_C( 17660),  INT16_C(  4991),  INT16_C( 13222),  INT16_C( 15460),
        -INT16_C(  8896),  INT16_C(  1216), -INT16_C( 29384),  INT16_C( 27816), -INT16_C(  3398),  INT16_C( 11800),  INT16_C( 21503),  INT16_C( 13096) } },
    { {  INT16_C( 17153), -INT16_C( 14314),  INT16_C(  3218),  INT16_C( 31045), -INT16_C( 18353), -INT16_C(  2347), -INT16_C(  8468), -INT16_C( 30068),
        -INT16_C( 12878), -INT16_C( 17274), -INT16_C(  8599), -INT16_C( 18353),  INT16_C( 26456),  INT16_C( 16614),  INT16_C(  4014), -INT16_C( 20621),
        -INT16_C( 30126), -INT16_C(  6792), -INT16_C( 17002), -INT16_C(  6818),  INT16_C( 13430),  INT16_C( 25307),  INT16_C( 26642), -INT16_C( 15124),
         INT16_C( 29237), -INT16_C( 24960), -INT16_C( 12208), -INT16_C( 22186),  INT16_C( 15671), -INT16_C(  6679),  INT16_C( 23884), -INT16_C( 24939) },
      UINT32_C(2105740775),
      { -INT16_C(  7478),  INT16_C( 16482),  INT16_C( 15894),  INT16_C( 10402), -INT16_C( 28762), -INT16_C(  9235),  INT16_C( 27905),  INT16_C( 21113),
        -INT16_C( 12483),  INT16_C( 30203), -INT16_C(  7156),  INT16_C( 22618), -INT16_C(  4287),  INT16_C( 10487),  INT16_C( 31484), -INT16_C( 14427),
         INT16_C(  2140),  INT16_C( 29191), -INT16_C( 21946), -INT16_C(  4965), -INT16_C( 30663),  INT16_C( 15047),  INT16_C( 16629),  INT16_C( 13196),
        -INT16_C( 30961),  INT16_C(  7336),  INT16_C(   620), -INT16_C( 21132),  INT16_C( 27634), -INT16_C(  4394),  INT16_C( 31718),  INT16_C( 17077) },
      {  INT16_C(  7478),  INT16_C( 16482),  INT16_C( 15894),  INT16_C( 31045), -INT16_C( 18353),  INT16_C(  9235),  INT16_C( 27905),  INT16_C( 21113),
         INT16_C( 12483), -INT16_C( 17274),  INT16_C(  7156),  INT16_C( 22618),  INT16_C( 26456),  INT16_C( 16614),  INT16_C(  4014), -INT16_C( 20621),
         INT16_C(  2140),  INT16_C( 29191), -INT16_C( 17002), -INT16_C(  6818),  INT16_C( 13430),  INT16_C( 25307),  INT16_C( 26642),  INT16_C( 13196),
         INT16_C( 30961), -INT16_C( 24960),  INT16_C(   620),  INT16_C( 21132),  INT16_C( 27634),  INT16_C(  4394),  INT16_C( 31718), -INT16_C( 24939) } },
    { { -INT16_C( 17021), -INT16_C( 13899),  INT16_C( 20583), -INT16_C( 24395),  INT16_C( 31960), -INT16_C( 12838),  INT16_C( 26556), -INT16_C( 13312),
        -INT16_C( 22290),  INT16_C( 23272),  INT16_C( 23723), -INT16_C( 25336), -INT16_C(  8504), -INT16_C( 20853),  INT16_C( 16729), -INT16_C(  8720),
        -INT16_C( 23042),  INT16_C( 26022),  INT16_C( 23797), -INT16_C( 13051), -INT16_C(  8232), -INT16_C( 27237), -INT16_C( 25786),  INT16_C( 13665),
         INT16_C( 18756), -INT16_C(  4209), -INT16_C( 26715),  INT16_C( 28044),  INT16_C(  6005), -INT16_C( 12517),  INT16_C(  3160),  INT16_C( 22188) },
      UINT32_C(2814071473),
      { -INT16_C( 16210), -INT16_C( 30860),  INT16_C(  4000), -INT16_C(  6628),  INT16_C( 32171), -INT16_C(  4325), -INT16_C( 21562),  INT16_C( 27614),
         INT16_C( 27202), -INT16_C( 18215), -INT16_C(  2943), -INT16_C(  9593),  INT16_C( 13056), -INT16_C( 19920), -INT16_C(  4987),  INT16_C( 13401),
        -INT16_C( 12884),  INT16_C( 19643), -INT16_C( 10275), -INT16_C( 30669),  INT16_C( 20052),  INT16_C(  6775),  INT16_C( 22009),  INT16_C( 15493),
         INT16_C( 24255),  INT16_C( 16628),  INT16_C( 31571),  INT16_C( 21274),  INT16_C( 19374),  INT16_C( 13061),  INT16_C( 24119), -INT16_C(  7321) },
      {  INT16_C( 16210), -INT16_C( 13899),  INT16_C( 20583), -INT16_C( 24395),  INT16_C( 32171),  INT16_C(  4325),  INT16_C( 26556),  INT16_C( 27614),
        -INT16_C( 22290),  INT16_C( 18215),  INT16_C( 23723), -INT16_C( 25336),  INT16_C( 13056), -INT16_C( 20853),  INT16_C(  4987), -INT16_C(  8720),
         INT16_C( 12884),  INT16_C( 19643),  INT16_C( 23797),  INT16_C( 30669),  INT16_C( 20052),  INT16_C(  6775), -INT16_C( 25786),  INT16_C( 15493),
         INT16_C( 24255),  INT16_C( 16628),  INT16_C( 31571),  INT16_C( 28044),  INT16_C(  6005),  INT16_C( 13061),  INT16_C(  3160),  INT16_C(  7321) } },
    { {  INT16_C(  8748),  INT16_C(  2352),  INT16_C( 25593),  INT16_C( 19857),  INT16_C(  2225), -INT16_C( 21657), -INT16_C(  4771),  INT16_C(  7399),
        -INT16_C(  9397), -INT16_C( 24996),  INT16_C( 30550),  INT16_C(  1266), -INT16_C(  2110), -INT16_C(  1737), -INT16_C( 24746), -INT16_C( 32036),
         INT16_C(  3265), -INT16_C( 17525),  INT16_C(  7279),  INT16_C(  8456),  INT16_C( 28708), -INT16_C( 32308), -INT16_C( 19619), -INT16_C( 22371),
        -INT16_C(  1650), -INT16_C(  7097),  INT16_C( 14704),  INT16_C( 13032),  INT16_C(  7984), -INT16_C( 31189),  INT16_C(  2238), -INT16_C( 32760) },
      UINT32_C(2218496788),
      {  INT16_C( 17327), -INT16_C( 11355),  INT16_C( 29107),  INT16_C(  4180), -INT16_C(  3804), -INT16_C( 19783),  INT16_C(   235),  INT16_C( 23446),
         INT16_C( 32313),  INT16_C( 27022), -INT16_C( 18019),  INT16_C( 23792), -INT16_C(  1855), -INT16_C( 10532),  INT16_C(  6028),  INT16_C( 15194),
        -INT16_C(   166),  INT16_C(  3599),  INT16_C( 25456), -INT16_C( 27618), -INT16_C( 10411),  INT16_C( 16454), -INT16_C(  9001),  INT16_C(  4251),
         INT16_C( 10586), -INT16_C(  2182),  INT16_C( 27363), -INT16_C( 23469),  INT16_C( 12130), -INT16_C(  4486), -INT16_C( 11194), -INT16_C( 24278) },
      {  INT16_C(  8748),  INT16_C(  2352),  INT16_C( 29107),  INT16_C( 19857),  INT16_C(  3804), -INT16_C( 21657), -INT16_C(  4771),  INT16_C(  7399),
         INT16_C( 32313),  INT16_C( 27022),  INT16_C( 30550),  INT16_C(  1266),  INT16_C(  1855), -INT16_C(  1737), -INT16_C( 24746),  INT16_C( 15194),
         INT16_C(   166),  INT16_C(  3599),  INT16_C(  7279),  INT16_C( 27618),  INT16_C( 10411),  INT16_C( 16454), -INT16_C( 19619), -INT16_C( 22371),
        -INT16_C(  1650), -INT16_C(  7097),  INT16_C( 27363),  INT16_C( 13032),  INT16_C(  7984), -INT16_C( 31189),  INT16_C(  2238),  INT16_C( 24278) } },
    { {  INT16_C( 14803),  INT16_C( 17327), -INT16_C( 12900), -INT16_C(  3625),  INT16_C(  7589),  INT16_C( 31793), -INT16_C( 12807),  INT16_C( 21389),
         INT16_C(  2038), -INT16_C(  9909), -INT16_C( 24975), -INT16_C( 11394), -INT16_C(  1842),  INT16_C(  5314), -INT16_C(  4915), -INT16_C( 24395),
         INT16_C( 25637), -INT16_C( 15900), -INT16_C( 17614), -INT16_C( 10317), -INT16_C(  6951), -INT16_C( 11693), -INT16_C(  8015), -INT16_C( 22490),
         INT16_C( 29159),  INT16_C( 22657), -INT16_C(   241), -INT16_C(  8916), -INT16_C(  4360), -INT16_C( 14862), -INT16_C( 22566), -INT16_C(   155) },
      UINT32_C(1052789004),
      {  INT16_C( 29445), -INT16_C(  8683),  INT16_C( 26712),  INT16_C(  2480), -INT16_C( 10679),  INT16_C( 12465),  INT16_C( 13127),  INT16_C( 22409),
        -INT16_C( 19150),  INT16_C( 10804),  INT16_C(  9891),  INT16_C( 32239),  INT16_C( 21966), -INT16_C(  9604),  INT16_C( 15518), -INT16_C( 23784),
         INT16_C( 11696),  INT16_C(  2177),  INT16_C( 12949), -INT16_C(  8687), -INT16_C( 15608),  INT16_C( 20495), -INT16_C( 26378),  INT16_C( 10407),
        -INT16_C(  9395), -INT16_C(  4013),  INT16_C( 16898), -INT16_C( 12179), -INT16_C(  5737),  INT16_C( 13994), -INT16_C( 15835), -INT16_C( 10791) },
      {  INT16_C( 14803),  INT16_C( 17327),  INT16_C( 26712),  INT16_C(  2480),  INT16_C(  7589),  INT16_C( 31793), -INT16_C( 12807),  INT16_C( 21389),
         INT16_C( 19150), -INT16_C(  9909), -INT16_C( 24975),  INT16_C( 32239), -INT16_C(  1842),  INT16_C(  5314),  INT16_C( 15518), -INT16_C( 24395),
         INT16_C( 25637), -INT16_C( 15900), -INT16_C( 17614), -INT16_C( 10317), -INT16_C(  6951), -INT16_C( 11693),  INT16_C( 26378),  INT16_C( 10407),
         INT16_C( 29159),  INT16_C(  4013),  INT16_C( 16898),  INT16_C( 12179),  INT16_C(  5737),  INT16_C( 13994), -INT16_C( 22566), -INT16_C(   155) } },
    { {  INT16_C( 23535), -INT16_C( 31523), -INT16_C(  4211), -INT16_C( 27293),  INT16_C( 29362), -INT16_C( 22299), -INT16_C( 29686),  INT16_C( 22480),
         INT16_C(  9064),  INT16_C( 27207), -INT16_C( 19354), -INT16_C(   710), -INT16_C(  7011), -INT16_C( 15821),  INT16_C(  3494), -INT16_C( 27240),
         INT16_C( 30056), -INT16_C(  2791),  INT16_C( 31844),  INT16_C(  5770),  INT16_C( 28910), -INT16_C(  1858), -INT16_C( 28676),  INT16_C( 25679),
        -INT16_C( 26958),  INT16_C(  6350),  INT16_C(  2122), -INT16_C(  6378),  INT16_C( 18924), -INT16_C( 27990),  INT16_C( 16982), -INT16_C( 16857) },
      UINT32_C( 481509815),
      {  INT16_C( 16061), -INT16_C( 21454), -INT16_C(  3666), -INT16_C( 21852), -INT16_C(  2944),  INT16_C( 12815), -INT16_C(  8822), -INT16_C( 10933),
         INT16_C( 25062), -INT16_C( 11588),  INT16_C( 26282),  INT16_C(   357), -INT16_C( 29528),  INT16_C( 24767),  INT16_C( 29645), -INT16_C( 29828),
        -INT16_C( 20815),  INT16_C( 24375), -INT16_C(  9313),  INT16_C(  7945),  INT16_C(  6351),  INT16_C( 23122), -INT16_C( 25098), -INT16_C(  9169),
        -INT16_C(  5122), -INT16_C( 22354),  INT16_C(  4946), -INT16_C(  1367),  INT16_C( 27040),  INT16_C( 27994), -INT16_C( 10532), -INT16_C( 29192) },
      {  INT16_C( 16061),  INT16_C( 21454),  INT16_C(  3666), -INT16_C( 27293),  INT16_C(  2944),  INT16_C( 12815), -INT16_C( 29686),  INT16_C( 10933),
         INT16_C( 25062),  INT16_C( 27207), -INT16_C( 19354), -INT16_C(   710), -INT16_C(  7011), -INT16_C( 15821),  INT16_C( 29645), -INT16_C( 27240),
         INT16_C( 20815),  INT16_C( 24375),  INT16_C( 31844),  INT16_C(  5770),  INT16_C(  6351),  INT16_C( 23122), -INT16_C( 28676),  INT16_C(  9169),
        -INT16_C( 26958),  INT16_C(  6350),  INT16_C(  4946),  INT16_C(  1367),  INT16_C( 27040), -INT16_C( 27990),  INT16_C( 16982), -INT16_C( 16857) } },
    { {  INT16_C( 12165),  INT16_C(  9452), -INT16_C(  2805), -INT16_C(  9660), -INT16_C( 27122),  INT16_C(  1076),  INT16_C( 25395),  INT16_C( 12768),
        -INT16_C( 29105), -INT16_C( 24103), -INT16_C( 31838),  INT16_C( 17051), -INT16_C(  2324), -INT16_C( 14161), -INT16_C( 22324),  INT16_C( 20821),
         INT16_C( 16855), -INT16_C(  7562), -INT16_C( 17866),  INT16_C( 17597), -INT16_C(  3760), -INT16_C( 31928),  INT16_C( 10325), -INT16_C( 23372),
        -INT16_C( 29257),  INT16_C( 22853), -INT16_C(  8176), -INT16_C(   869),  INT16_C( 19158), -INT16_C( 23612),  INT16_C(  6642), -INT16_C( 13580) },
      UINT32_C(2443995738),
      {  INT16_C( 26916),  INT16_C( 29909),  INT16_C(  7771), -INT16_C( 20233), -INT16_C( 21690), -INT16_C(   684), -INT16_C( 26311),  INT16_C( 18774),
        -INT16_C(  3719),  INT16_C( 20550),  INT16_C(  2620),  INT16_C( 12019), -INT16_C(  6364),  INT16_C( 32504), -INT16_C( 23214),  INT16_C( 30223),
        -INT16_C(  6898),  INT16_C( 27115), -INT16_C(  7677),  INT16_C( 18713),  INT16_C( 28046), -INT16_C( 14521), -INT16_C( 25338), -INT16_C( 32752),
         INT16_C( 22159), -INT16_C( 13360), -INT16_C( 15519), -INT16_C( 31239), -INT16_C(  3414), -INT16_C(  1021),  INT16_C(  5015), -INT16_C( 23181) },
      {  INT16_C( 12165),  INT16_C( 29909), -INT16_C(  2805),  INT16_C( 20233),  INT16_C( 21690),  INT16_C(  1076),  INT16_C( 26311),  INT16_C( 12768),
        -INT16_C( 29105),  INT16_C( 20550), -INT16_C( 31838),  INT16_C( 12019), -INT16_C(  2324),  INT16_C( 32504),  INT16_C( 23214),  INT16_C( 20821),
         INT16_C( 16855), -INT16_C(  7562),  INT16_C(  7677),  INT16_C( 18713), -INT16_C(  3760),  INT16_C( 14521),  INT16_C( 10325),  INT16_C( 32752),
         INT16_C( 22159),  INT16_C( 22853), -INT16_C(  8176), -INT16_C(   869),  INT16_C(  3414), -INT16_C( 23612),  INT16_C(  6642),  INT16_C( 23181) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_abs_epi16(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_abs_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_abs_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int16_t r[32];
  } test_vec[] = {
    { UINT32_C( 426916685),
      {  INT16_C( 30267),  INT16_C( 27257), -INT16_C( 27872), -INT16_C( 31845), -INT16_C( 31872),  INT16_C( 14983), -INT16_C( 30143),  INT16_C(   666),
         INT16_C( 12540),  INT16_C(  6967), -INT16_C( 16243),  INT16_C(   692), -INT16_C(  5341),  INT16_C( 28883),  INT16_C( 17702),  INT16_C( 24969),
         INT16_C(   955), -INT16_C(  9013),  INT16_C( 26518),  INT16_C(  5727), -INT16_C(  6422),  INT16_C( 11089), -INT16_C(  5264),  INT16_C( 27693),
         INT16_C( 25627), -INT16_C( 22392),  INT16_C( 15396),  INT16_C( 18346),  INT16_C( 32295),  INT16_C( 19896),  INT16_C( 16835),  INT16_C( 32686) },
      {  INT16_C( 30267),  INT16_C(     0),  INT16_C( 27872),  INT16_C( 31845),  INT16_C(     0),  INT16_C(     0),  INT16_C( 30143),  INT16_C(     0),
         INT16_C( 12540),  INT16_C(  6967),  INT16_C(     0),  INT16_C(   692),  INT16_C(  5341),  INT16_C( 28883),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(  9013),  INT16_C(     0),  INT16_C(     0),  INT16_C(  6422),  INT16_C( 11089),  INT16_C(  5264),  INT16_C(     0),
         INT16_C( 25627),  INT16_C(     0),  INT16_C(     0),  INT16_C( 18346),  INT16_C( 32295),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(3680205380),
      { -INT16_C( 17695), -INT16_C( 13327),  INT16_C( 17057),  INT16_C(  4599),  INT16_C(  9261),  INT16_C( 18558),  INT16_C(  1673), -INT16_C( 21008),
        -INT16_C( 25790),  INT16_C( 27125), -INT16_C( 21223), -INT16_C(  9034),  INT16_C( 25838),  INT16_C( 13147), -INT16_C( 18722), -INT16_C( 16626),
        -INT16_C(   143),  INT16_C(  4747), -INT16_C( 32190),  INT16_C( 28451), -INT16_C( 24154),  INT16_C( 12216), -INT16_C( 22361), -INT16_C(  5667),
        -INT16_C( 11709),  INT16_C( 23634),  INT16_C(  2175),  INT16_C( 27961), -INT16_C( 27539),  INT16_C( 19360), -INT16_C( 20917), -INT16_C( 17397) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 17057),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1673),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 27125),  INT16_C(     0),  INT16_C(  9034),  INT16_C( 25838),  INT16_C( 13147),  INT16_C( 18722),  INT16_C(     0),
         INT16_C(   143),  INT16_C(  4747),  INT16_C(     0),  INT16_C( 28451),  INT16_C( 24154),  INT16_C(     0),  INT16_C( 22361),  INT16_C(     0),
         INT16_C( 11709),  INT16_C( 23634),  INT16_C(     0),  INT16_C( 27961),  INT16_C( 27539),  INT16_C(     0),  INT16_C( 20917),  INT16_C( 17397) } },
    { UINT32_C(4040070830),
      { -INT16_C(  3816), -INT16_C( 16801),  INT16_C(  6035),  INT16_C( 15086), -INT16_C( 13376),  INT16_C(   804),  INT16_C( 30365),  INT16_C(  7264),
        -INT16_C( 26241), -INT16_C(  4983),  INT16_C( 10797),  INT16_C( 30775),  INT16_C( 17112), -INT16_C( 31180),  INT16_C(   728), -INT16_C(  3978),
        -INT16_C( 10508), -INT16_C( 30801), -INT16_C( 25107), -INT16_C( 21055), -INT16_C(  6808),  INT16_C(  1457),  INT16_C(  4444), -INT16_C(  9439),
        -INT16_C( 21846), -INT16_C( 10297), -INT16_C(   300), -INT16_C( 21168), -INT16_C( 31679),  INT16_C(  6451), -INT16_C( 21881),  INT16_C( 31498) },
      {  INT16_C(     0),  INT16_C( 16801),  INT16_C(  6035),  INT16_C( 15086),  INT16_C(     0),  INT16_C(   804),  INT16_C(     0),  INT16_C(  7264),
         INT16_C(     0),  INT16_C(  4983),  INT16_C( 10797),  INT16_C(     0),  INT16_C( 17112),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3978),
         INT16_C(     0),  INT16_C( 30801),  INT16_C( 25107),  INT16_C( 21055),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4444),  INT16_C(  9439),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31679),  INT16_C(  6451),  INT16_C( 21881),  INT16_C( 31498) } },
    { UINT32_C(1828895104),
      { -INT16_C( 15530), -INT16_C( 16869), -INT16_C( 13143),  INT16_C(  1475), -INT16_C(  6947), -INT16_C( 30752), -INT16_C( 22642),  INT16_C( 25438),
        -INT16_C( 20827), -INT16_C(  6640),  INT16_C( 17203), -INT16_C( 17920),  INT16_C(  2797),  INT16_C( 27957),  INT16_C( 14275),  INT16_C(  6619),
        -INT16_C(  2310), -INT16_C( 23593), -INT16_C( 25918), -INT16_C( 24664), -INT16_C( 30594),  INT16_C(  3110), -INT16_C( 31697), -INT16_C( 10897),
         INT16_C( 32563),  INT16_C( 26299), -INT16_C( 17469), -INT16_C( 20448),  INT16_C( 21957), -INT16_C( 30690), -INT16_C(  1652), -INT16_C( 31071) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 25438),
         INT16_C( 20827),  INT16_C(     0),  INT16_C(     0),  INT16_C( 17920),  INT16_C(  2797),  INT16_C( 27957),  INT16_C(     0),  INT16_C(  6619),
         INT16_C(     0),  INT16_C( 23593),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 32563),  INT16_C(     0),  INT16_C( 17469),  INT16_C( 20448),  INT16_C(     0),  INT16_C( 30690),  INT16_C(  1652),  INT16_C(     0) } },
    { UINT32_C(2972350703),
      { -INT16_C( 11758), -INT16_C( 28592),  INT16_C( 30299), -INT16_C( 30051),  INT16_C(  3322),  INT16_C( 11615),  INT16_C(  7052),  INT16_C( 20371),
        -INT16_C( 19498), -INT16_C( 25345),  INT16_C(  7432), -INT16_C( 27612), -INT16_C( 14826),  INT16_C(  1307),  INT16_C( 17726),  INT16_C( 20918),
         INT16_C(  1559),  INT16_C( 29409),  INT16_C( 32380),  INT16_C( 30717),  INT16_C( 23691),  INT16_C(  6052),  INT16_C( 14455),  INT16_C( 20070),
         INT16_C( 26091), -INT16_C(  2838),  INT16_C(  3715), -INT16_C( 26232), -INT16_C( 23596),  INT16_C(  5023),  INT16_C( 21992),  INT16_C(   100) },
      {  INT16_C( 11758),  INT16_C( 28592),  INT16_C( 30299),  INT16_C( 30051),  INT16_C(     0),  INT16_C( 11615),  INT16_C(  7052),  INT16_C( 20371),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 27612),  INT16_C( 14826),  INT16_C(  1307),  INT16_C( 17726),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 29409),  INT16_C(     0),  INT16_C( 30717),  INT16_C(     0),  INT16_C(  6052),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 26091),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 23596),  INT16_C(  5023),  INT16_C(     0),  INT16_C(   100) } },
    { UINT32_C(3631367516),
      {  INT16_C( 28612),  INT16_C( 20303), -INT16_C(  2868),  INT16_C( 17254), -INT16_C( 13268),  INT16_C(  6033),  INT16_C( 31537), -INT16_C( 19445),
        -INT16_C( 27510),  INT16_C( 24142), -INT16_C(  4809),  INT16_C(  8305), -INT16_C( 10942), -INT16_C( 25056), -INT16_C( 28133), -INT16_C(  8329),
        -INT16_C( 14846), -INT16_C( 12754), -INT16_C( 27462), -INT16_C(  6639), -INT16_C( 23712), -INT16_C( 28162),  INT16_C(  2334), -INT16_C( 22458),
        -INT16_C( 27491), -INT16_C( 11001),  INT16_C( 30849), -INT16_C( 15371),  INT16_C(  5454),  INT16_C( 26978), -INT16_C(  9817), -INT16_C( 22200) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(  2868),  INT16_C( 17254),  INT16_C( 13268),  INT16_C(     0),  INT16_C( 31537),  INT16_C(     0),
         INT16_C( 27510),  INT16_C(     0),  INT16_C(  4809),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 28133),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 12754),  INT16_C(     0),  INT16_C(     0),  INT16_C( 23712),  INT16_C( 28162),  INT16_C(  2334),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 15371),  INT16_C(  5454),  INT16_C(     0),  INT16_C(  9817),  INT16_C( 22200) } },
    { UINT32_C(1517778591),
      { -INT16_C( 30454),  INT16_C( 27200),  INT16_C( 15916),  INT16_C( 19195),  INT16_C( 16712), -INT16_C(  6669), -INT16_C(  1323),  INT16_C( 22202),
        -INT16_C( 20622), -INT16_C( 16358),  INT16_C( 31940),  INT16_C( 27689),  INT16_C( 29013), -INT16_C(  3051), -INT16_C( 29209), -INT16_C(  3762),
        -INT16_C( 28906),  INT16_C( 16987),  INT16_C( 22477),  INT16_C(  5516),  INT16_C( 32664),  INT16_C( 28411), -INT16_C( 19079), -INT16_C(  4924),
        -INT16_C(  8603),  INT16_C( 10668), -INT16_C( 10662), -INT16_C( 20587), -INT16_C( 21689),  INT16_C( 12196), -INT16_C(  3528),  INT16_C( 20000) },
      {  INT16_C( 30454),  INT16_C( 27200),  INT16_C( 15916),  INT16_C( 19195),  INT16_C( 16712),  INT16_C(     0),  INT16_C(     0),  INT16_C( 22202),
         INT16_C(     0),  INT16_C( 16358),  INT16_C( 31940),  INT16_C(     0),  INT16_C( 29013),  INT16_C(  3051),  INT16_C( 29209),  INT16_C(     0),
         INT16_C( 28906),  INT16_C( 16987),  INT16_C( 22477),  INT16_C(     0),  INT16_C( 32664),  INT16_C( 28411),  INT16_C( 19079),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 10668),  INT16_C(     0),  INT16_C( 20587),  INT16_C( 21689),  INT16_C(     0),  INT16_C(  3528),  INT16_C(     0) } },
    { UINT32_C(1334869121),
      {  INT16_C(  7379),  INT16_C( 27492),  INT16_C( 24476),  INT16_C(  5593), -INT16_C( 25067),  INT16_C( 31233), -INT16_C( 20868), -INT16_C( 10333),
         INT16_C( 14724), -INT16_C( 13434),  INT16_C( 10980),  INT16_C(  7418),  INT16_C(  6941), -INT16_C( 24982), -INT16_C(  1385),  INT16_C( 27373),
         INT16_C( 21014), -INT16_C( 19755), -INT16_C( 20559), -INT16_C( 14648), -INT16_C( 14003), -INT16_C( 14016), -INT16_C(  7049), -INT16_C(  1120),
         INT16_C( 10013),  INT16_C(   455), -INT16_C( 16047),  INT16_C( 28189), -INT16_C( 30756),  INT16_C( 29453), -INT16_C(  1407), -INT16_C( 26659) },
      {  INT16_C(  7379),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10333),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 10980),  INT16_C(  7418),  INT16_C(  6941),  INT16_C( 24982),  INT16_C(  1385),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 14003),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1120),
         INT16_C( 10013),  INT16_C(   455),  INT16_C( 16047),  INT16_C( 28189),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1407),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_abs_epi16(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_abs_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_abs_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C(-1095158286), INT32_C( -133595553), INT32_C( -941949577), INT32_C(-1117722052),
                            INT32_C(-1053667317), INT32_C( -662420643), INT32_C( 2095193825), INT32_C( -799061081),
                            INT32_C(  347912513), INT32_C( -439299809), INT32_C( 2053030698), INT32_C( -277514113),
                            INT32_C( 1476262970), INT32_C( 1955038119), INT32_C(  -77085072), INT32_C( 1014069144)),
      easysimd_mm512_set_epi32(INT32_C( 1095158286), INT32_C(  133595553), INT32_C(  941949577), INT32_C( 1117722052),
                            INT32_C( 1053667317), INT32_C(  662420643), INT32_C( 2095193825), INT32_C(  799061081),
                            INT32_C(  347912513), INT32_C(  439299809), INT32_C( 2053030698), INT32_C(  277514113),
                            INT32_C( 1476262970), INT32_C( 1955038119), INT32_C(   77085072), INT32_C( 1014069144)) },
    { easysimd_mm512_set_epi32(INT32_C( 1865049709), INT32_C( -408997463), INT32_C( 1771073477), INT32_C( 1463780468),
                            INT32_C(  268019741), INT32_C(-1725054429), INT32_C( 1408597864), INT32_C( 1576985133),
                            INT32_C(  170783936), INT32_C(  836522882), INT32_C( 1364040350), INT32_C(  563663058),
                            INT32_C(-1491438903), INT32_C( -873504608), INT32_C( 1431273511), INT32_C( -164765086)),
      easysimd_mm512_set_epi32(INT32_C( 1865049709), INT32_C(  408997463), INT32_C( 1771073477), INT32_C( 1463780468),
                            INT32_C(  268019741), INT32_C( 1725054429), INT32_C( 1408597864), INT32_C( 1576985133),
                            INT32_C(  170783936), INT32_C(  836522882), INT32_C( 1364040350), INT32_C(  563663058),
                            INT32_C( 1491438903), INT32_C(  873504608), INT32_C( 1431273511), INT32_C(  164765086)) },
    { easysimd_mm512_set_epi32(INT32_C( 1505063340), INT32_C(  -79208486), INT32_C( -115790145), INT32_C( 1137793635),
                            INT32_C( -719063760), INT32_C( -465633360), INT32_C( 1417132608), INT32_C( 1715322300),
                            INT32_C( 1194443989), INT32_C( 1598244723), INT32_C( -360509626), INT32_C( -844528776),
                            INT32_C( -291907566), INT32_C( -980752736), INT32_C(  701363552), INT32_C( 1148036152)),
      easysimd_mm512_set_epi32(INT32_C( 1505063340), INT32_C(   79208486), INT32_C(  115790145), INT32_C( 1137793635),
                            INT32_C(  719063760), INT32_C(  465633360), INT32_C( 1417132608), INT32_C( 1715322300),
                            INT32_C( 1194443989), INT32_C( 1598244723), INT32_C(  360509626), INT32_C(  844528776),
                            INT32_C(  291907566), INT32_C(  980752736), INT32_C(  701363552), INT32_C( 1148036152)) },
    { easysimd_mm512_set_epi32(INT32_C(-1538804784), INT32_C(  -43683957), INT32_C(  -70380459), INT32_C(  259050545),
                            INT32_C(-1140217223), INT32_C(  -24242506), INT32_C(-1281378925), INT32_C( -426768587),
                            INT32_C(-1825251144), INT32_C( -975195895), INT32_C(  758020113), INT32_C(   -3401471),
                            INT32_C(  154668063), INT32_C( -827616009), INT32_C(  793625070), INT32_C( -735990247)),
      easysimd_mm512_set_epi32(INT32_C( 1538804784), INT32_C(   43683957), INT32_C(   70380459), INT32_C(  259050545),
                            INT32_C( 1140217223), INT32_C(   24242506), INT32_C( 1281378925), INT32_C(  426768587),
                            INT32_C( 1825251144), INT32_C(  975195895), INT32_C(  758020113), INT32_C(    3401471),
                            INT32_C(  154668063), INT32_C(  827616009), INT32_C(  793625070), INT32_C(  735990247)) },
    { easysimd_mm512_set_epi32(INT32_C( -919197120), INT32_C( 1902742720), INT32_C(  576001152), INT32_C(  772608991),
                            INT32_C( 1373611304), INT32_C(  156079462), INT32_C(  392030686), INT32_C( 1159450969),
                            INT32_C( 1376625025), INT32_C( -701917672), INT32_C( 1911493359), INT32_C( -115817480),
                            INT32_C( -875216623), INT32_C( 1333681477), INT32_C(-1067533891), INT32_C( 1671330781)),
      easysimd_mm512_set_epi32(INT32_C(  919197120), INT32_C( 1902742720), INT32_C(  576001152), INT32_C(  772608991),
                            INT32_C( 1373611304), INT32_C(  156079462), INT32_C(  392030686), INT32_C( 1159450969),
                            INT32_C( 1376625025), INT32_C(  701917672), INT32_C( 1911493359), INT32_C(  115817480),
                            INT32_C(  875216623), INT32_C( 1333681477), INT32_C( 1067533891), INT32_C( 1671330781)) },
    { easysimd_mm512_set_epi32(INT32_C(-1168385947), INT32_C(-1671882855), INT32_C(-1182456995), INT32_C(-1803534861),
                            INT32_C(  443878759), INT32_C(  702169153), INT32_C(-1879742181), INT32_C( 1627978919),
                            INT32_C(  583873330), INT32_C( -857098109), INT32_C(  710347808), INT32_C( 1707849385),
                            INT32_C( 1863512780), INT32_C( -371421167), INT32_C( 1902179408), INT32_C(-1189025654)),
      easysimd_mm512_set_epi32(INT32_C( 1168385947), INT32_C( 1671882855), INT32_C( 1182456995), INT32_C( 1803534861),
                            INT32_C(  443878759), INT32_C(  702169153), INT32_C( 1879742181), INT32_C( 1627978919),
                            INT32_C(  583873330), INT32_C(  857098109), INT32_C(  710347808), INT32_C( 1707849385),
                            INT32_C( 1863512780), INT32_C(  371421167), INT32_C( 1902179408), INT32_C( 1189025654)) },
    { easysimd_mm512_set_epi32(INT32_C(    7990856), INT32_C(-1991291137), INT32_C( 1404443548), INT32_C(-1023849862),
                            INT32_C( 2054941409), INT32_C(-1604088325), INT32_C(  721271909), INT32_C(-1622295089),
                            INT32_C( 1869222605), INT32_C(-1583998423), INT32_C( -801626928), INT32_C( -940395766),
                            INT32_C( 1108931720), INT32_C( -471669445), INT32_C( 1204289475), INT32_C( -752679106)),
      easysimd_mm512_set_epi32(INT32_C(    7990856), INT32_C( 1991291137), INT32_C( 1404443548), INT32_C( 1023849862),
                            INT32_C( 2054941409), INT32_C( 1604088325), INT32_C(  721271909), INT32_C( 1622295089),
                            INT32_C( 1869222605), INT32_C( 1583998423), INT32_C(  801626928), INT32_C(  940395766),
                            INT32_C( 1108931720), INT32_C(  471669445), INT32_C( 1204289475), INT32_C(  752679106)) },
    { easysimd_mm512_set_epi32(INT32_C( 1399806844), INT32_C( 1131841699), INT32_C( -346937782), INT32_C(  567816154),
                            INT32_C(-1589012616), INT32_C(-2005496894), INT32_C( 1401681986), INT32_C(  423760716),
                            INT32_C(  431684101), INT32_C(  852583616), INT32_C(-1369299290), INT32_C( -663899319),
                            INT32_C( 1580470265), INT32_C(  298083241), INT32_C( -630373638), INT32_C(-1937828661)),
      easysimd_mm512_set_epi32(INT32_C( 1399806844), INT32_C( 1131841699), INT32_C(  346937782), INT32_C(  567816154),
                            INT32_C( 1589012616), INT32_C( 2005496894), INT32_C( 1401681986), INT32_C(  423760716),
                            INT32_C(  431684101), INT32_C(  852583616), INT32_C( 1369299290), INT32_C(  663899319),
                            INT32_C( 1580470265), INT32_C(  298083241), INT32_C(  630373638), INT32_C( 1937828661)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_abs_epi32(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_abs_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_abs_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C(  114710097), INT32_C( 1837246098), INT32_C(-1399577225), INT32_C(-1388127606),
                            INT32_C( 1116027725), INT32_C( -871797325), INT32_C(-1979326643), INT32_C( 1477004857),
                            INT32_C( 1670723749), INT32_C(-1006052339), INT32_C( 1863789116), INT32_C( -690396684),
                            INT32_C( -629773535), INT32_C(  667046523), INT32_C( 1317445565), INT32_C( -732937024)),
      UINT16_C(28888),
      easysimd_mm512_set_epi32(INT32_C(-1877548571), INT32_C( -399920351), INT32_C(   15525797), INT32_C(   18165921),
                            INT32_C( 2085930596), INT32_C( 1662282658), INT32_C(-1842752263), INT32_C(  499820912),
                            INT32_C( 1419797765), INT32_C(  -44818966), INT32_C( 1761152620), INT32_C(-1288657930),
                            INT32_C(  894078020), INT32_C(-1369431563), INT32_C( -103362440), INT32_C(-1559726025)),
      easysimd_mm512_set_epi32(INT32_C(  114710097), INT32_C(  399920351), INT32_C(   15525797), INT32_C(   18165921),
                            INT32_C( 1116027725), INT32_C( -871797325), INT32_C(-1979326643), INT32_C( 1477004857),
                            INT32_C( 1419797765), INT32_C(   44818966), INT32_C( 1863789116), INT32_C( 1288657930),
                            INT32_C(  894078020), INT32_C(  667046523), INT32_C( 1317445565), INT32_C( -732937024)) },
    { easysimd_mm512_set_epi32(INT32_C( 1505436737), INT32_C(  342253548), INT32_C( 1435179252), INT32_C( 1326766533),
                            INT32_C(  156769011), INT32_C(  343760696), INT32_C(  611303965), INT32_C( 1457341409),
                            INT32_C(  165452421), INT32_C(-1824090116), INT32_C( -184738383), INT32_C( -191086464),
                            INT32_C( -117301127), INT32_C(-1526262537), INT32_C( -208138847), INT32_C(  807348777)),
      UINT16_C(61134),
      easysimd_mm512_set_epi32(INT32_C( 1427056174), INT32_C( 2097896620), INT32_C( 1813263538), INT32_C( 1909821993),
                            INT32_C( 1439822042), INT32_C(-1049213292), INT32_C( 1557133349), INT32_C( 1168931268),
                            INT32_C( -810546774), INT32_C(-1283013132), INT32_C(  654302587), INT32_C(  314275905),
                            INT32_C(-1091094079), INT32_C( -114174508), INT32_C(  407580338), INT32_C( 1906809805)),
      easysimd_mm512_set_epi32(INT32_C( 1427056174), INT32_C( 2097896620), INT32_C( 1813263538), INT32_C( 1326766533),
                            INT32_C( 1439822042), INT32_C( 1049213292), INT32_C( 1557133349), INT32_C( 1457341409),
                            INT32_C(  810546774), INT32_C( 1283013132), INT32_C( -184738383), INT32_C( -191086464),
                            INT32_C( 1091094079), INT32_C(  114174508), INT32_C(  407580338), INT32_C(  807348777)) },
    { easysimd_mm512_set_epi32(INT32_C(  905172649), INT32_C(-1044778809), INT32_C(-1938215986), INT32_C(-1138753169),
                            INT32_C(-1689961651), INT32_C(  890456168), INT32_C( 1382435241), INT32_C( -803845344),
                            INT32_C(  430838507), INT32_C( 1075259040), INT32_C(-1956785379), INT32_C(-1586468297),
                            INT32_C(  622055688), INT32_C(-1127740382), INT32_C(  466514910), INT32_C(-1745879628)),
      UINT16_C(30570),
      easysimd_mm512_set_epi32(INT32_C( -310045086), INT32_C(  560822999), INT32_C( -680371476), INT32_C( 1838395052),
                            INT32_C(-1152635838), INT32_C( -481448106), INT32_C(  871399876), INT32_C( -939960538),
                            INT32_C( -898000986), INT32_C( -641497176), INT32_C(  657638908), INT32_C(-1796735419),
                            INT32_C(-1032150818), INT32_C(  151713087), INT32_C( 1554707006), INT32_C( -318690470)),
      easysimd_mm512_set_epi32(INT32_C(  905172649), INT32_C(  560822999), INT32_C(  680371476), INT32_C( 1838395052),
                            INT32_C(-1689961651), INT32_C(  481448106), INT32_C(  871399876), INT32_C(  939960538),
                            INT32_C(  430838507), INT32_C(  641497176), INT32_C(  657638908), INT32_C(-1586468297),
                            INT32_C( 1032150818), INT32_C(-1127740382), INT32_C( 1554707006), INT32_C(-1745879628)) },
    { easysimd_mm512_set_epi32(INT32_C(-1675700291), INT32_C(  -85412591), INT32_C(-1865493216), INT32_C(-1122257925),
                            INT32_C(  955620837), INT32_C( -725693586), INT32_C( 1056307491), INT32_C( 1924019839),
                            INT32_C(-2012466116), INT32_C(-1808881746), INT32_C( -887453452), INT32_C(  160221724),
                            INT32_C( -886018282), INT32_C( 1222780200), INT32_C( 1877396684), INT32_C(  283360472)),
      UINT16_C(28339),
      easysimd_mm512_set_epi32(INT32_C(-1238615237), INT32_C(  583893938), INT32_C( -594441984), INT32_C( 1561597956),
                            INT32_C(  174377227), INT32_C(  319460903), INT32_C(-1295208114), INT32_C(  659707887),
                            INT32_C( 1117898731), INT32_C( -209622907), INT32_C(-1431480123), INT32_C(-2058827609),
                            INT32_C(-1519596795), INT32_C(   24332922), INT32_C( -338106630), INT32_C(-1565374776)),
      easysimd_mm512_set_epi32(INT32_C(-1675700291), INT32_C(  583893938), INT32_C(  594441984), INT32_C(-1122257925),
                            INT32_C(  174377227), INT32_C(  319460903), INT32_C( 1295208114), INT32_C( 1924019839),
                            INT32_C( 1117898731), INT32_C(-1808881746), INT32_C( 1431480123), INT32_C( 2058827609),
                            INT32_C( -886018282), INT32_C( 1222780200), INT32_C(  338106630), INT32_C( 1565374776)) },
    { easysimd_mm512_set_epi32(INT32_C(  178377352), INT32_C( -324510384), INT32_C(  446946466), INT32_C(-1323398690),
                            INT32_C( -720979875), INT32_C( -512216094), INT32_C( 1145272930), INT32_C( -706074883),
                            INT32_C(-1863795060), INT32_C( -525595897), INT32_C( 1357119557), INT32_C(  837734387),
                            INT32_C( -607392699), INT32_C( -498581669), INT32_C(-2108693629), INT32_C( -476969927)),
      UINT16_C(42507),
      easysimd_mm512_set_epi32(INT32_C(   -5472621), INT32_C( -263868960), INT32_C(-1867831731), INT32_C(  955254216),
                            INT32_C( 1990179011), INT32_C(-1729740457), INT32_C( 1711933869), INT32_C(-1566075058),
                            INT32_C( -550106516), INT32_C(-1087591249), INT32_C(  919917002), INT32_C(-1410389997),
                            INT32_C( -188117230), INT32_C( 1025569327), INT32_C(-1456210246), INT32_C( -254945819)),
      easysimd_mm512_set_epi32(INT32_C(    5472621), INT32_C( -324510384), INT32_C( 1867831731), INT32_C(-1323398690),
                            INT32_C( -720979875), INT32_C( 1729740457), INT32_C( 1711933869), INT32_C( -706074883),
                            INT32_C(-1863795060), INT32_C( -525595897), INT32_C( 1357119557), INT32_C(  837734387),
                            INT32_C(  188117230), INT32_C( -498581669), INT32_C( 1456210246), INT32_C(  254945819)) },
    { easysimd_mm512_set_epi32(INT32_C(-1007934437), INT32_C(  201253136), INT32_C( 2123754123), INT32_C( 1034305262),
                            INT32_C( 2139323878), INT32_C( -545410429), INT32_C(-1549231865), INT32_C( 1779895500),
                            INT32_C( 1932853973), INT32_C( 2135732954), INT32_C( 1232725518), INT32_C(  339564914),
                            INT32_C( -113030707), INT32_C(-1715459937), INT32_C( -492435091), INT32_C(-1720946495)),
      UINT16_C(49758),
      easysimd_mm512_set_epi32(INT32_C(  348473993), INT32_C(-1624874318), INT32_C(  361690252), INT32_C(  165927413),
                            INT32_C(-1864332117), INT32_C( -524477604), INT32_C(  481484649), INT32_C(-1499715490),
                            INT32_C(-1683117466), INT32_C(-2055457330), INT32_C( -850617531), INT32_C(-2081246973),
                            INT32_C( 1276057415), INT32_C( 1619064589), INT32_C(-1536816688), INT32_C( 2060578085)),
      easysimd_mm512_set_epi32(INT32_C(  348473993), INT32_C( 1624874318), INT32_C( 2123754123), INT32_C( 1034305262),
                            INT32_C( 2139323878), INT32_C( -545410429), INT32_C(  481484649), INT32_C( 1779895500),
                            INT32_C( 1932853973), INT32_C( 2055457330), INT32_C( 1232725518), INT32_C( 2081246973),
                            INT32_C( 1276057415), INT32_C( 1619064589), INT32_C( 1536816688), INT32_C(-1720946495)) },
    { easysimd_mm512_set_epi32(INT32_C(  860828042), INT32_C( 1459856596), INT32_C(-1901530659), INT32_C( 1296141157),
                            INT32_C(  778663095), INT32_C(-1872048536), INT32_C(-1115787645), INT32_C(-1142406643),
                            INT32_C( 1518955242), INT32_C( -174688543), INT32_C( 1537062129), INT32_C( -974095643),
                            INT32_C(  125816377), INT32_C(-1032428044), INT32_C( -374455538), INT32_C( -648832583)),
      UINT16_C(41340),
      easysimd_mm512_set_epi32(INT32_C( 1553986008), INT32_C( -808715903), INT32_C(-2114331727), INT32_C(  878797396),
                            INT32_C( 1547560130), INT32_C( -931453209), INT32_C(  639671594), INT32_C(  734358771),
                            INT32_C(-1802430748), INT32_C(   38083245), INT32_C(  636500349), INT32_C( 2020438947),
                            INT32_C(   89083218), INT32_C( 2041918986), INT32_C(-2068453500), INT32_C( 1772569863)),
      easysimd_mm512_set_epi32(INT32_C( 1553986008), INT32_C( 1459856596), INT32_C( 2114331727), INT32_C( 1296141157),
                            INT32_C(  778663095), INT32_C(-1872048536), INT32_C(-1115787645), INT32_C(  734358771),
                            INT32_C( 1518955242), INT32_C(   38083245), INT32_C(  636500349), INT32_C( 2020438947),
                            INT32_C(   89083218), INT32_C( 2041918986), INT32_C( -374455538), INT32_C( -648832583)) },
    { easysimd_mm512_set_epi32(INT32_C(-1208548961), INT32_C( 1705109710), INT32_C( -159097588), INT32_C( -879037423),
                            INT32_C( 2121552533), INT32_C(  595529007), INT32_C( -405863552), INT32_C( 1431630584),
                            INT32_C( -616000216), INT32_C(  444327364), INT32_C(  613413664), INT32_C(-2128463203),
                            INT32_C(  939927077), INT32_C(-1255659348), INT32_C(-1631544337), INT32_C(-1727626838)),
      UINT16_C(49163),
      easysimd_mm512_set_epi32(INT32_C(  895846723), INT32_C(  449272422), INT32_C( 1127330699), INT32_C(-1084895433),
                            INT32_C( -399265722), INT32_C(  697840482), INT32_C( -598276089), INT32_C(  -50403840),
                            INT32_C( 1970006978), INT32_C( 1602141812), INT32_C(-1773480652), INT32_C(  740913018),
                            INT32_C( 1668822994), INT32_C(  698152405), INT32_C( 1772335922), INT32_C(  847772835)),
      easysimd_mm512_set_epi32(INT32_C(  895846723), INT32_C(  449272422), INT32_C( -159097588), INT32_C( -879037423),
                            INT32_C( 2121552533), INT32_C(  595529007), INT32_C( -405863552), INT32_C( 1431630584),
                            INT32_C( -616000216), INT32_C(  444327364), INT32_C(  613413664), INT32_C(-2128463203),
                            INT32_C( 1668822994), INT32_C(-1255659348), INT32_C( 1772335922), INT32_C(  847772835)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i src = test_vec[i].src;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_abs_epi32(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_abs_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_abs_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT16_C(17600),
      easysimd_mm512_set_epi32(INT32_C(  393115914), INT32_C(   -9604904), INT32_C(  114710097), INT32_C( 1837246098),
                            INT32_C(-1399577225), INT32_C(-1388127606), INT32_C( 1116027725), INT32_C( -871797325),
                            INT32_C(-1979326643), INT32_C( 1477004857), INT32_C( 1670723749), INT32_C(-1006052339),
                            INT32_C( 1863789116), INT32_C( -690396684), INT32_C( -629773535), INT32_C(  667046523)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(    9604904), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1388127606), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1979326643), INT32_C( 1477004857), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
    { UINT16_C(30775),
      easysimd_mm512_set_epi32(INT32_C( -208138847), INT32_C(  807348777), INT32_C(-1877548571), INT32_C( -399920351),
                            INT32_C(   15525797), INT32_C(   18165921), INT32_C( 2085930596), INT32_C( 1662282658),
                            INT32_C(-1842752263), INT32_C(  499820912), INT32_C( 1419797765), INT32_C(  -44818966),
                            INT32_C( 1761152620), INT32_C(-1288657930), INT32_C(  894078020), INT32_C(-1369431563)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  807348777), INT32_C( 1877548571), INT32_C(  399920351),
                            INT32_C(   15525797), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( 1419797765), INT32_C(   44818966),
                            INT32_C(          0), INT32_C( 1288657930), INT32_C(  894078020), INT32_C( 1369431563)) },
    { UINT16_C( 5367),
      easysimd_mm512_set_epi32(INT32_C(  407580338), INT32_C( 1906809805), INT32_C( -849801752), INT32_C(-1965822258),
                            INT32_C( 1505436737), INT32_C(  342253548), INT32_C( 1435179252), INT32_C( 1326766533),
                            INT32_C(  156769011), INT32_C(  343760696), INT32_C(  611303965), INT32_C( 1457341409),
                            INT32_C(  165452421), INT32_C(-1824090116), INT32_C( -184738383), INT32_C( -191086464)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C( 1965822258),
                            INT32_C(          0), INT32_C(  342253548), INT32_C(          0), INT32_C(          0),
                            INT32_C(  156769011), INT32_C(  343760696), INT32_C(  611303965), INT32_C( 1457341409),
                            INT32_C(          0), INT32_C( 1824090116), INT32_C(  184738383), INT32_C(  191086464)) },
    { UINT16_C(54740),
      easysimd_mm512_set_epi32(INT32_C(  622055688), INT32_C(-1127740382), INT32_C(  466514910), INT32_C(-1745879628),
                            INT32_C( 1427056174), INT32_C( 2097896620), INT32_C( 1813263538), INT32_C( 1909821993),
                            INT32_C( 1439822042), INT32_C(-1049213292), INT32_C( 1557133349), INT32_C( 1168931268),
                            INT32_C( -810546774), INT32_C(-1283013132), INT32_C(  654302587), INT32_C(  314275905)),
      easysimd_mm512_set_epi32(INT32_C(  622055688), INT32_C( 1127740382), INT32_C(          0), INT32_C( 1745879628),
                            INT32_C(          0), INT32_C( 2097896620), INT32_C(          0), INT32_C( 1909821993),
                            INT32_C( 1439822042), INT32_C( 1049213292), INT32_C(          0), INT32_C( 1168931268),
                            INT32_C(          0), INT32_C( 1283013132), INT32_C(          0), INT32_C(          0)) },
    { UINT16_C(27191),
      easysimd_mm512_set_epi32(INT32_C(-1032150818), INT32_C(  151713087), INT32_C( 1554707006), INT32_C( -318690470),
                            INT32_C(  788893537), INT32_C( -230394006), INT32_C(  905172649), INT32_C(-1044778809),
                            INT32_C(-1938215986), INT32_C(-1138753169), INT32_C(-1689961651), INT32_C(  890456168),
                            INT32_C( 1382435241), INT32_C( -803845344), INT32_C(  430838507), INT32_C( 1075259040)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  151713087), INT32_C( 1554707006), INT32_C(          0),
                            INT32_C(  788893537), INT32_C(          0), INT32_C(  905172649), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( 1689961651), INT32_C(  890456168),
                            INT32_C(          0), INT32_C(  803845344), INT32_C(  430838507), INT32_C( 1075259040)) },
    { UINT16_C(65093),
      easysimd_mm512_set_epi32(INT32_C( -887453452), INT32_C(  160221724), INT32_C( -886018282), INT32_C( 1222780200),
                            INT32_C( 1877396684), INT32_C(  283360472), INT32_C( -310045086), INT32_C(  560822999),
                            INT32_C( -680371476), INT32_C( 1838395052), INT32_C(-1152635838), INT32_C( -481448106),
                            INT32_C(  871399876), INT32_C( -939960538), INT32_C( -898000986), INT32_C( -641497176)),
      easysimd_mm512_set_epi32(INT32_C(  887453452), INT32_C(  160221724), INT32_C(  886018282), INT32_C( 1222780200),
                            INT32_C( 1877396684), INT32_C(  283360472), INT32_C(  310045086), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1838395052), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(  939960538), INT32_C(          0), INT32_C(  641497176)) },
    { UINT16_C(42926),
      easysimd_mm512_set_epi32(INT32_C(-1431480123), INT32_C(-2058827609), INT32_C(-1519596795), INT32_C(   24332922),
                            INT32_C( -338106630), INT32_C(-1565374776), INT32_C(-1426452996), INT32_C( -680300877),
                            INT32_C(-1675700291), INT32_C(  -85412591), INT32_C(-1865493216), INT32_C(-1122257925),
                            INT32_C(  955620837), INT32_C( -725693586), INT32_C( 1056307491), INT32_C( 1924019839)),
      easysimd_mm512_set_epi32(INT32_C( 1431480123), INT32_C(          0), INT32_C( 1519596795), INT32_C(          0),
                            INT32_C(          0), INT32_C( 1565374776), INT32_C( 1426452996), INT32_C(  680300877),
                            INT32_C( 1675700291), INT32_C(          0), INT32_C( 1865493216), INT32_C(          0),
                            INT32_C(  955620837), INT32_C(  725693586), INT32_C( 1056307491), INT32_C(          0)) },
    { UINT16_C(26757),
      easysimd_mm512_set_epi32(INT32_C(-1863795060), INT32_C( -525595897), INT32_C( 1357119557), INT32_C(  837734387),
                            INT32_C( -607392699), INT32_C( -498581669), INT32_C(-2108693629), INT32_C( -476969927),
                            INT32_C(-1238615237), INT32_C(  583893938), INT32_C( -594441984), INT32_C( 1561597956),
                            INT32_C(  174377227), INT32_C(  319460903), INT32_C(-1295208114), INT32_C(  659707887)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  525595897), INT32_C( 1357119557), INT32_C(          0),
                            INT32_C(  607392699), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1238615237), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(  319460903), INT32_C(          0), INT32_C(  659707887)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_abs_epi32(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_abs_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_abs_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C(-4703669018152042913), INT64_C(-4045642624518788548),
                            INT64_C(-4525466663746518179), INT64_C( 8998788960652053415),
                            INT64_C( 1494272869059842335), INT64_C( 8817699709611505791),
                            INT64_C( 6340501178400867239), INT64_C( -331077862235736168)),
      easysimd_mm512_set_epi64(INT64_C( 4703669018152042913), INT64_C( 4045642624518788548),
                            INT64_C( 4525466663746518179), INT64_C( 8998788960652053415),
                            INT64_C( 1494272869059842335), INT64_C( 8817699709611505791),
                            INT64_C( 6340501178400867239), INT64_C(  331077862235736168)) },
    { easysimd_mm512_set_epi64(INT64_C( 8010327509455286697), INT64_C( 7606702663991788660),
                            INT64_C( 1151136024847303203), INT64_C( 6049881760672440877),
                            INT64_C(  733511420638679938), INT64_C( 5858508694238056658),
                            INT64_C(-6405681308945653600), INT64_C( 6147272925506298466)),
      easysimd_mm512_set_epi64(INT64_C( 8010327509455286697), INT64_C( 7606702663991788660),
                            INT64_C( 1151136024847303203), INT64_C( 6049881760672440877),
                            INT64_C(  733511420638679938), INT64_C( 5858508694238056658),
                            INT64_C( 6405681308945653600), INT64_C( 6147272925506298466)) },
    { easysimd_mm512_set_epi64(INT64_C( 6464197827924287450), INT64_C( -497314884836304285),
                            INT64_C(-3088355329109459024), INT64_C( 6086538207170510268),
                            INT64_C( 5130097871257028467), INT64_C(-1548377050112752776),
                            INT64_C(-1253733446110746976), INT64_C( 3012333519594431544)),
      easysimd_mm512_set_epi64(INT64_C( 6464197827924287450), INT64_C(  497314884836304285),
                            INT64_C( 3088355329109459024), INT64_C( 6086538207170510268),
                            INT64_C( 5130097871257028467), INT64_C( 1548377050112752776),
                            INT64_C( 1253733446110746976), INT64_C( 3012333519594431544)) },
    { easysimd_mm512_set_epi64(INT64_C(-6609116217957060725), INT64_C( -302281769423418319),
                            INT64_C(-4897195678850214218), INT64_C(-5503480572790438091),
                            INT64_C(-7839393967146815223), INT64_C( 3255671599336790273),
                            INT64_C(  664294275788018935), INT64_C( 3408593724494687769)),
      easysimd_mm512_set_epi64(INT64_C( 6609116217957060725), INT64_C(  302281769423418319),
                            INT64_C( 4897195678850214218), INT64_C( 5503480572790438091),
                            INT64_C( 7839393967146815223), INT64_C( 3255671599336790273),
                            INT64_C(  664294275788018935), INT64_C( 3408593724494687769)) },
    { easysimd_mm512_set_epi64(INT64_C(-3947921567074644800), INT64_C( 2473906111070933983),
                            INT64_C( 5899615628251993446), INT64_C( 1683758976557896025),
                            INT64_C( 5912559464823232024), INT64_C( 8209801467605337080),
                            INT64_C(-3759026771366879931), INT64_C(-4585023147545297955)),
      easysimd_mm512_set_epi64(INT64_C( 3947921567074644800), INT64_C( 2473906111070933983),
                            INT64_C( 5899615628251993446), INT64_C( 1683758976557896025),
                            INT64_C( 5912559464823232024), INT64_C( 8209801467605337080),
                            INT64_C( 3759026771366879931), INT64_C( 4585023147545297955)) },
    { easysimd_mm512_set_epi64(INT64_C(-5018179428847904871), INT64_C(-5078614119960003085),
                            INT64_C( 1906444753996234817), INT64_C(-8073431190678733657),
                            INT64_C( 2507716860794484867), INT64_C( 3050920605853136553),
                            INT64_C( 8003726449701589009), INT64_C( 8169798351590582410)),
      easysimd_mm512_set_epi64(INT64_C( 5018179428847904871), INT64_C( 5078614119960003085),
                            INT64_C( 1906444753996234817), INT64_C( 8073431190678733657),
                            INT64_C( 2507716860794484867), INT64_C( 3050920605853136553),
                            INT64_C( 8003726449701589009), INT64_C( 8169798351590582410)) },
    { easysimd_mm512_set_epi64(INT64_C(   34320467490721535), INT64_C( 6032039111009323642),
                            INT64_C( 8825906149542039035), INT64_C( 3097839263351160271),
                            INT64_C( 8028249960129894953), INT64_C(-3442961435998375158),
                            INT64_C( 4762825474720326971), INT64_C( 5172383913584297790)),
      easysimd_mm512_set_epi64(INT64_C(   34320467490721535), INT64_C( 6032039111009323642),
                            INT64_C( 8825906149542039035), INT64_C( 3097839263351160271),
                            INT64_C( 8028249960129894953), INT64_C( 3442961435998375158),
                            INT64_C( 4762825474720326971), INT64_C( 5172383913584297790)) },
    { easysimd_mm512_set_epi64(INT64_C( 6012124616828815523), INT64_C(-1490086426868961318),
                            INT64_C(-6824757216361935934), INT64_C( 6020178289686090572),
                            INT64_C( 1854069096850744512), INT64_C(-5881095665354951863),
                            INT64_C( 6788068100773536681), INT64_C(-2707434157113404213)),
      easysimd_mm512_set_epi64(INT64_C( 6012124616828815523), INT64_C( 1490086426868961318),
                            INT64_C( 6824757216361935934), INT64_C( 6020178289686090572),
                            INT64_C( 1854069096850744512), INT64_C( 5881095665354951863),
                            INT64_C( 6788068100773536681), INT64_C( 2707434157113404213)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_abs_epi64(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_abs_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_abs_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C(  492676116973233810), INT64_C(-6011138406694593910),
                            INT64_C( 4793302583727451571), INT64_C(-8501143198309462471),
                            INT64_C( 7175703865894427661), INT64_C( 8004913303465320948),
                            INT64_C(-2704856736044264837), INT64_C( 5658385619497272512)),
      UINT8_C(216),
      easysimd_mm512_set_epi64(INT64_C(-1717644828534315099), INT64_C(   78022038682650212),
                            INT64_C( 7139449655270167801), INT64_C( 2146714472316691717),
                            INT64_C( -192495991449383316), INT64_C(-5534743664186979260),
                            INT64_C(-5881663773003558792), INT64_C(-6698972267701962486)),
      easysimd_mm512_set_epi64(INT64_C( 1717644828534315099), INT64_C(   78022038682650212),
                            INT64_C( 4793302583727451571), INT64_C( 2146714472316691717),
                            INT64_C(  192495991449383316), INT64_C( 8004913303465320948),
                            INT64_C(-2704856736044264837), INT64_C( 5658385619497272512)) },
    { easysimd_mm512_set_epi64(INT64_C( 1469967797035145460), INT64_C( 5698418868819073779),
                            INT64_C( 1476440947581501981), INT64_C( 6259233690927012485),
                            INT64_C(-7834407389066617423), INT64_C( -820710109410615175),
                            INT64_C(-6555247677438161503), INT64_C( 3467536596098015717)),
      UINT8_C( 65),
      easysimd_mm512_set_epi64(INT64_C( 7787907596649075241), INT64_C( 6183988585695692436),
                            INT64_C( 6687836810634885572), INT64_C(-3481271883196348940),
                            INT64_C( 2810208213167470657), INT64_C(-4686213381983447596),
                            INT64_C( 1750544224109435853), INT64_C(-3649870730594357554)),
      easysimd_mm512_set_epi64(INT64_C( 1469967797035145460), INT64_C( 6183988585695692436),
                            INT64_C( 1476440947581501981), INT64_C( 6259233690927012485),
                            INT64_C(-7834407389066617423), INT64_C( -820710109410615175),
                            INT64_C(-6555247677438161503), INT64_C( 3649870730594357554)) },
    { easysimd_mm512_set_epi64(INT64_C(-8324574269298179729), INT64_C(-7258330021648709528),
                            INT64_C( 5937514152424000288), INT64_C( 1850437298497726112),
                            INT64_C(-8404329205387466185), INT64_C( 2671708839418006562),
                            INT64_C( 2003666284095471028), INT64_C( 6129159598982782124)),
      UINT8_C(199),
      easysimd_mm512_set_epi64(INT64_C( 7895846628610550850), INT64_C(-2067803869119741500),
                            INT64_C(-4037099766843598938), INT64_C(-2755209390738717188),
                            INT64_C(-7716919860907040546), INT64_C(  651602748594909758),
                            INT64_C(-1368765145407975583), INT64_C( -989534720059255127)),
      easysimd_mm512_set_epi64(INT64_C( 7895846628610550850), INT64_C( 2067803869119741500),
                            INT64_C( 5937514152424000288), INT64_C( 1850437298497726112),
                            INT64_C(-8404329205387466185), INT64_C(  651602748594909758),
                            INT64_C( 1368765145407975583), INT64_C(  989534720059255127)) },
    { easysimd_mm512_set_epi64(INT64_C(-4820061084596199963), INT64_C(-3116830217730655965),
                            INT64_C( 8263602287642686524), INT64_C(-7769087937993864972),
                            INT64_C(  688147068097687318), INT64_C( 5251800971073735884),
                            INT64_C( 1217023964204045922), INT64_C( 2408716443164236524)),
      UINT8_C( 32),
      easysimd_mm512_set_epi64(INT64_C(  748944487451629095), INT64_C(-5562876490484131857),
                            INT64_C( 4801338493970245765), INT64_C(-6148160310922917721),
                            INT64_C(-6526618537607083398), INT64_C(-1452156915681179960),
                            INT64_C(-6126568963486552397), INT64_C(-7197077943533128431)),
      easysimd_mm512_set_epi64(INT64_C(-4820061084596199963), INT64_C(-3116830217730655965),
                            INT64_C( 4801338493970245765), INT64_C(-7769087937993864972),
                            INT64_C(  688147068097687318), INT64_C( 5251800971073735884),
                            INT64_C( 1217023964204045922), INT64_C( 2408716443164236524)) },
    { easysimd_mm512_set_epi64(INT64_C(-3096584980416416798), INT64_C( 4918909782932989693),
                            INT64_C(-8004938825376986361), INT64_C( 5828784114914742259),
                            INT64_C(-2608731774237786277), INT64_C(-9056770170020559815),
                            INT64_C(-5319811934658395214), INT64_C(-2553108879087757308)),
      UINT8_C(222),
      easysimd_mm512_set_epi64(INT64_C(-7429178691671160403), INT64_C(-6726241153446442388),
                            INT64_C(-4671168844950875702), INT64_C(-6057578907613688046),
                            INT64_C( 4404786722084486842), INT64_C(-1094983952222664046),
                            INT64_C( 5287971478839612040), INT64_C(-1393761486045455198)),
      easysimd_mm512_set_epi64(INT64_C( 7429178691671160403), INT64_C( 6726241153446442388),
                            INT64_C(-8004938825376986361), INT64_C( 6057578907613688046),
                            INT64_C( 4404786722084486842), INT64_C( 1094983952222664046),
                            INT64_C( 5287971478839612040), INT64_C(-2553108879087757308)) },
    { easysimd_mm512_set_epi64(INT64_C(-2342519952706594553), INT64_C( 7644592964730421973),
                            INT64_C( 9172903191652197902), INT64_C( 1458420204680989133),
                            INT64_C(-7367844323210688147), INT64_C(-7391408909901332845),
                            INT64_C(-1133308551202396595), INT64_C( 4102785619076298947)),
      UINT8_C(230),
      easysimd_mm512_set_epi64(INT64_C( 2067960823776290910), INT64_C(-7228934469556881970),
                            INT64_C(-3653374474835545853), INT64_C( 5480624866862364429),
                            INT64_C(-6600577412846457563), INT64_C( 2529415530022027870),
                            INT64_C(-4329045443225919216), INT64_C( 9121454504064466670)),
      easysimd_mm512_set_epi64(INT64_C( 2067960823776290910), INT64_C( 7228934469556881970),
                            INT64_C( 3653374474835545853), INT64_C( 1458420204680989133),
                            INT64_C(-7367844323210688147), INT64_C( 2529415530022027870),
                            INT64_C( 4329045443225919216), INT64_C( 4102785619076298947)) },
    { easysimd_mm512_set_epi64(INT64_C(-4792271441403297267), INT64_C( 6523863092598044385),
                            INT64_C( 6601631579296004837), INT64_C(  540377227778745844),
                            INT64_C(-1608274285869950535), INT64_C( 1496684406111625906),
                            INT64_C( 1553447803787926005), INT64_C(-8007245467626955940)),
      UINT8_C(104),
      easysimd_mm512_set_epi64(INT64_C( 3154046907468289764), INT64_C(  163566292437055869),
                            INT64_C( 8677719201018760530), INT64_C( 8769975268177995652),
                            INT64_C( 7613129594859420923), INT64_C(-1998576254813523574),
                            INT64_C( 6270036339063321053), INT64_C( 5566883881093264567)),
      easysimd_mm512_set_epi64(INT64_C(-4792271441403297267), INT64_C(  163566292437055869),
                            INT64_C( 8677719201018760530), INT64_C(  540377227778745844),
                            INT64_C( 7613129594859420923), INT64_C( 1496684406111625906),
                            INT64_C( 1553447803787926005), INT64_C(-8007245467626955940)) },
    { easysimd_mm512_set_epi64(INT64_C( 6148806541912347944), INT64_C( 1908371497711301408),
                            INT64_C(-9141679846684482011), INT64_C(-5393015831913260049),
                            INT64_C(-7420100767347904040), INT64_C(-3473408352959472719),
                            INT64_C( 3774406077177521346), INT64_C(-4000561069769581270)),
      UINT8_C(128),
      easysimd_mm512_set_epi64(INT64_C( 8461115545003933300), INT64_C(-7617041399687843974),
                            INT64_C( 7167540182740956629), INT64_C( 7612124823363779747),
                            INT64_C(-1891905030773424117), INT64_C(-5190678261404669746),
                            INT64_C( -683318933916552175), INT64_C( 9111998746576489775)),
      easysimd_mm512_set_epi64(INT64_C( 8461115545003933300), INT64_C( 1908371497711301408),
                            INT64_C(-9141679846684482011), INT64_C(-5393015831913260049),
                            INT64_C(-7420100767347904040), INT64_C(-3473408352959472719),
                            INT64_C( 3774406077177521346), INT64_C(-4000561069769581270)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i src = test_vec[i].src;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_abs_epi64(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_abs_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_abs_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT8_C(192),
      easysimd_mm512_set_epi64(INT64_C(  -41252748446509487), INT64_C( 7890911908509001079),
                            INT64_C(-5961962669328745651), INT64_C(-3744340997299642547),
                            INT64_C( 6343687558518880421), INT64_C(-4320961892205516228),
                            INT64_C(-2965231175381652703), INT64_C( 2864943002512957373)),
      easysimd_mm512_set_epi64(INT64_C(   41252748446509487), INT64_C( 7890911908509001079),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C( 10),
      easysimd_mm512_set_epi64(INT64_C(-8064009705201487071), INT64_C(   66682790377500833),
                            INT64_C( 8959003693208071074), INT64_C(-7914560703715169936),
                            INT64_C( 6097984971859041770), INT64_C( 7564092909171024886),
                            INT64_C( 3840035858897969653), INT64_C( -443938296699520969)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 6097984971859041770), INT64_C(                   0),
                            INT64_C( 3840035858897969653), INT64_C(                   0)) },
    { UINT8_C( 41),
      easysimd_mm512_set_epi64(INT64_C(-8443142306353437631), INT64_C( 1469967797035145460),
                            INT64_C( 5698418868819073779), INT64_C( 1476440947581501981),
                            INT64_C( 6259233690927012485), INT64_C(-7834407389066617423),
                            INT64_C( -820710109410615175), INT64_C(-6555247677438161503)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 5698418868819073779), INT64_C(                   0),
                            INT64_C( 6259233690927012485), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 6555247677438161503)) },
    { UINT8_C(232),
      easysimd_mm512_set_epi64(INT64_C( 6129159598982782124), INT64_C( 7787907596649075241),
                            INT64_C( 6183988585695692436), INT64_C( 6687836810634885572),
                            INT64_C(-3481271883196348940), INT64_C( 2810208213167470657),
                            INT64_C(-4686213381983447596), INT64_C( 1750544224109435853)),
      easysimd_mm512_set_epi64(INT64_C( 6129159598982782124), INT64_C( 7787907596649075241),
                            INT64_C( 6183988585695692436), INT64_C(                   0),
                            INT64_C( 3481271883196348940), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(180),
      easysimd_mm512_set_epi64(INT64_C( -989534720059255127), INT64_C(-4487290813852079154),
                            INT64_C(-4890907616466355379), INT64_C( 3824480121463916969),
                            INT64_C(-3452489463091031317), INT64_C( 4618202413866537757),
                            INT64_C(-6813829451133759224), INT64_C(-4843608058602032162)),
      easysimd_mm512_set_epi64(INT64_C(  989534720059255127), INT64_C(                   0),
                            INT64_C( 4890907616466355379), INT64_C( 3824480121463916969),
                            INT64_C(                   0), INT64_C( 4618202413866537757),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C( 97),
      easysimd_mm512_set_epi64(INT64_C(-1331633504094684457), INT64_C(-2922173236712853844),
                            INT64_C(-4950533224594034858), INT64_C( 3742633972513462054),
                            INT64_C(-3856884862992283736), INT64_C( 2824537604935384645),
                            INT64_C(-4433054007697935041), INT64_C( 6677415749608352602)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C( 2922173236712853844),
                            INT64_C( 4950533224594034858), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 6677415749608352602)) },
    { UINT8_C(216),
      easysimd_mm512_set_epi64(INT64_C(-2921870015535851587), INT64_C( -366844282582149856),
                            INT64_C(-4820061084596199963), INT64_C(-3116830217730655965),
                            INT64_C( 8263602287642686524), INT64_C(-7769087937993864972),
                            INT64_C(  688147068097687318), INT64_C( 5251800971073735884)),
      easysimd_mm512_set_epi64(INT64_C( 2921870015535851587), INT64_C(  366844282582149856),
                            INT64_C(                   0), INT64_C( 3116830217730655965),
                            INT64_C( 8263602287642686524), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(252),
      easysimd_mm512_set_epi64(INT64_C(-5319811934658395214), INT64_C(-2553108879087757308),
                            INT64_C(  748944487451629095), INT64_C(-5562876490484131857),
                            INT64_C( 4801338493970245765), INT64_C(-6148160310922917721),
                            INT64_C(-6526618537607083398), INT64_C(-1452156915681179960)),
      easysimd_mm512_set_epi64(INT64_C( 5319811934658395214), INT64_C( 2553108879087757308),
                            INT64_C(  748944487451629095), INT64_C( 5562876490484131857),
                            INT64_C( 4801338493970245765), INT64_C( 6148160310922917721),
                            INT64_C(                   0), INT64_C(                   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_abs_epi64(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_abs_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_abs_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   747.74), EASYSIMD_FLOAT32_C(  -874.37), EASYSIMD_FLOAT32_C(   751.90), EASYSIMD_FLOAT32_C(  -592.77),
                         EASYSIMD_FLOAT32_C(  -708.81), EASYSIMD_FLOAT32_C(   252.42), EASYSIMD_FLOAT32_C(  -787.46), EASYSIMD_FLOAT32_C(  -882.47),
                         EASYSIMD_FLOAT32_C(  -140.56), EASYSIMD_FLOAT32_C(  -558.99), EASYSIMD_FLOAT32_C(   240.08), EASYSIMD_FLOAT32_C(  -481.72),
                         EASYSIMD_FLOAT32_C(   489.35), EASYSIMD_FLOAT32_C(   686.76), EASYSIMD_FLOAT32_C(  -206.54), EASYSIMD_FLOAT32_C(   728.61)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   747.74), EASYSIMD_FLOAT32_C(   874.37), EASYSIMD_FLOAT32_C(   751.90), EASYSIMD_FLOAT32_C(   592.77),
                         EASYSIMD_FLOAT32_C(   708.81), EASYSIMD_FLOAT32_C(   252.42), EASYSIMD_FLOAT32_C(   787.46), EASYSIMD_FLOAT32_C(   882.47),
                         EASYSIMD_FLOAT32_C(   140.56), EASYSIMD_FLOAT32_C(   558.99), EASYSIMD_FLOAT32_C(   240.08), EASYSIMD_FLOAT32_C(   481.72),
                         EASYSIMD_FLOAT32_C(   489.35), EASYSIMD_FLOAT32_C(   686.76), EASYSIMD_FLOAT32_C(   206.54), EASYSIMD_FLOAT32_C(   728.61)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    76.28), EASYSIMD_FLOAT32_C(  -319.23), EASYSIMD_FLOAT32_C(   655.09), EASYSIMD_FLOAT32_C(   773.21),
                         EASYSIMD_FLOAT32_C(  -928.32), EASYSIMD_FLOAT32_C(   -25.13), EASYSIMD_FLOAT32_C(  -847.53), EASYSIMD_FLOAT32_C(   859.40),
                         EASYSIMD_FLOAT32_C(   388.54), EASYSIMD_FLOAT32_C(  -184.67), EASYSIMD_FLOAT32_C(   102.38), EASYSIMD_FLOAT32_C(   833.56),
                         EASYSIMD_FLOAT32_C(  -722.29), EASYSIMD_FLOAT32_C(  -441.84), EASYSIMD_FLOAT32_C(  -821.42), EASYSIMD_FLOAT32_C(  -761.98)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    76.28), EASYSIMD_FLOAT32_C(   319.23), EASYSIMD_FLOAT32_C(   655.09), EASYSIMD_FLOAT32_C(   773.21),
                         EASYSIMD_FLOAT32_C(   928.32), EASYSIMD_FLOAT32_C(    25.13), EASYSIMD_FLOAT32_C(   847.53), EASYSIMD_FLOAT32_C(   859.40),
                         EASYSIMD_FLOAT32_C(   388.54), EASYSIMD_FLOAT32_C(   184.67), EASYSIMD_FLOAT32_C(   102.38), EASYSIMD_FLOAT32_C(   833.56),
                         EASYSIMD_FLOAT32_C(   722.29), EASYSIMD_FLOAT32_C(   441.84), EASYSIMD_FLOAT32_C(   821.42), EASYSIMD_FLOAT32_C(   761.98)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -645.61), EASYSIMD_FLOAT32_C(   594.77), EASYSIMD_FLOAT32_C(  -171.69), EASYSIMD_FLOAT32_C(   108.08),
                         EASYSIMD_FLOAT32_C(    -7.24), EASYSIMD_FLOAT32_C(   885.82), EASYSIMD_FLOAT32_C(   296.84), EASYSIMD_FLOAT32_C(  -408.70),
                         EASYSIMD_FLOAT32_C(   -40.31), EASYSIMD_FLOAT32_C(   866.84), EASYSIMD_FLOAT32_C(  -660.11), EASYSIMD_FLOAT32_C(   121.17),
                         EASYSIMD_FLOAT32_C(   988.31), EASYSIMD_FLOAT32_C(  -622.26), EASYSIMD_FLOAT32_C(   206.00), EASYSIMD_FLOAT32_C(   520.48)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   645.61), EASYSIMD_FLOAT32_C(   594.77), EASYSIMD_FLOAT32_C(   171.69), EASYSIMD_FLOAT32_C(   108.08),
                         EASYSIMD_FLOAT32_C(     7.24), EASYSIMD_FLOAT32_C(   885.82), EASYSIMD_FLOAT32_C(   296.84), EASYSIMD_FLOAT32_C(   408.70),
                         EASYSIMD_FLOAT32_C(    40.31), EASYSIMD_FLOAT32_C(   866.84), EASYSIMD_FLOAT32_C(   660.11), EASYSIMD_FLOAT32_C(   121.17),
                         EASYSIMD_FLOAT32_C(   988.31), EASYSIMD_FLOAT32_C(   622.26), EASYSIMD_FLOAT32_C(   206.00), EASYSIMD_FLOAT32_C(   520.48)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   530.77), EASYSIMD_FLOAT32_C(   966.97), EASYSIMD_FLOAT32_C(   -63.51), EASYSIMD_FLOAT32_C(   360.07),
                         EASYSIMD_FLOAT32_C(  -846.61), EASYSIMD_FLOAT32_C(  -749.79), EASYSIMD_FLOAT32_C(   510.77), EASYSIMD_FLOAT32_C(  -104.12),
                         EASYSIMD_FLOAT32_C(  -838.06), EASYSIMD_FLOAT32_C(  -901.25), EASYSIMD_FLOAT32_C(   -89.58), EASYSIMD_FLOAT32_C(   539.88),
                         EASYSIMD_FLOAT32_C(    88.35), EASYSIMD_FLOAT32_C(   773.77), EASYSIMD_FLOAT32_C(  -729.20), EASYSIMD_FLOAT32_C(  -254.72)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   530.77), EASYSIMD_FLOAT32_C(   966.97), EASYSIMD_FLOAT32_C(    63.51), EASYSIMD_FLOAT32_C(   360.07),
                         EASYSIMD_FLOAT32_C(   846.61), EASYSIMD_FLOAT32_C(   749.79), EASYSIMD_FLOAT32_C(   510.77), EASYSIMD_FLOAT32_C(   104.12),
                         EASYSIMD_FLOAT32_C(   838.06), EASYSIMD_FLOAT32_C(   901.25), EASYSIMD_FLOAT32_C(    89.58), EASYSIMD_FLOAT32_C(   539.88),
                         EASYSIMD_FLOAT32_C(    88.35), EASYSIMD_FLOAT32_C(   773.77), EASYSIMD_FLOAT32_C(   729.20), EASYSIMD_FLOAT32_C(   254.72)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -814.45), EASYSIMD_FLOAT32_C(  -377.80), EASYSIMD_FLOAT32_C(   640.68), EASYSIMD_FLOAT32_C(   778.00),
                         EASYSIMD_FLOAT32_C(   377.67), EASYSIMD_FLOAT32_C(  -489.06), EASYSIMD_FLOAT32_C(   933.74), EASYSIMD_FLOAT32_C(  -749.41),
                         EASYSIMD_FLOAT32_C(   193.12), EASYSIMD_FLOAT32_C(  -423.37), EASYSIMD_FLOAT32_C(  -194.06), EASYSIMD_FLOAT32_C(  -118.88),
                         EASYSIMD_FLOAT32_C(   -77.74), EASYSIMD_FLOAT32_C(  -506.16), EASYSIMD_FLOAT32_C(  -617.33), EASYSIMD_FLOAT32_C(  -947.60)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   814.45), EASYSIMD_FLOAT32_C(   377.80), EASYSIMD_FLOAT32_C(   640.68), EASYSIMD_FLOAT32_C(   778.00),
                         EASYSIMD_FLOAT32_C(   377.67), EASYSIMD_FLOAT32_C(   489.06), EASYSIMD_FLOAT32_C(   933.74), EASYSIMD_FLOAT32_C(   749.41),
                         EASYSIMD_FLOAT32_C(   193.12), EASYSIMD_FLOAT32_C(   423.37), EASYSIMD_FLOAT32_C(   194.06), EASYSIMD_FLOAT32_C(   118.88),
                         EASYSIMD_FLOAT32_C(    77.74), EASYSIMD_FLOAT32_C(   506.16), EASYSIMD_FLOAT32_C(   617.33), EASYSIMD_FLOAT32_C(   947.60)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   525.11), EASYSIMD_FLOAT32_C(   299.00), EASYSIMD_FLOAT32_C(   814.48), EASYSIMD_FLOAT32_C(   676.51),
                         EASYSIMD_FLOAT32_C(  -481.76), EASYSIMD_FLOAT32_C(   528.75), EASYSIMD_FLOAT32_C(  -375.20), EASYSIMD_FLOAT32_C(   146.55),
                         EASYSIMD_FLOAT32_C(   199.14), EASYSIMD_FLOAT32_C(  -505.05), EASYSIMD_FLOAT32_C(   833.96), EASYSIMD_FLOAT32_C(  -388.48),
                         EASYSIMD_FLOAT32_C(  -212.57), EASYSIMD_FLOAT32_C(   943.89), EASYSIMD_FLOAT32_C(   651.63), EASYSIMD_FLOAT32_C(   695.54)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   525.11), EASYSIMD_FLOAT32_C(   299.00), EASYSIMD_FLOAT32_C(   814.48), EASYSIMD_FLOAT32_C(   676.51),
                         EASYSIMD_FLOAT32_C(   481.76), EASYSIMD_FLOAT32_C(   528.75), EASYSIMD_FLOAT32_C(   375.20), EASYSIMD_FLOAT32_C(   146.55),
                         EASYSIMD_FLOAT32_C(   199.14), EASYSIMD_FLOAT32_C(   505.05), EASYSIMD_FLOAT32_C(   833.96), EASYSIMD_FLOAT32_C(   388.48),
                         EASYSIMD_FLOAT32_C(   212.57), EASYSIMD_FLOAT32_C(   943.89), EASYSIMD_FLOAT32_C(   651.63), EASYSIMD_FLOAT32_C(   695.54)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -13.73), EASYSIMD_FLOAT32_C(  -546.29), EASYSIMD_FLOAT32_C(  -787.44), EASYSIMD_FLOAT32_C(  -104.88),
                         EASYSIMD_FLOAT32_C(   979.47), EASYSIMD_FLOAT32_C(  -744.23), EASYSIMD_FLOAT32_C(   836.15), EASYSIMD_FLOAT32_C(   495.73),
                         EASYSIMD_FLOAT32_C(  -301.39), EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C(   466.22), EASYSIMD_FLOAT32_C(   536.10),
                         EASYSIMD_FLOAT32_C(  -613.16), EASYSIMD_FLOAT32_C(  -393.36), EASYSIMD_FLOAT32_C(   -56.94), EASYSIMD_FLOAT32_C(   670.22)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    13.73), EASYSIMD_FLOAT32_C(   546.29), EASYSIMD_FLOAT32_C(   787.44), EASYSIMD_FLOAT32_C(   104.88),
                         EASYSIMD_FLOAT32_C(   979.47), EASYSIMD_FLOAT32_C(   744.23), EASYSIMD_FLOAT32_C(   836.15), EASYSIMD_FLOAT32_C(   495.73),
                         EASYSIMD_FLOAT32_C(   301.39), EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C(   466.22), EASYSIMD_FLOAT32_C(   536.10),
                         EASYSIMD_FLOAT32_C(   613.16), EASYSIMD_FLOAT32_C(   393.36), EASYSIMD_FLOAT32_C(    56.94), EASYSIMD_FLOAT32_C(   670.22)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   493.79), EASYSIMD_FLOAT32_C(   -29.44), EASYSIMD_FLOAT32_C(  -941.83), EASYSIMD_FLOAT32_C(  -567.95),
                         EASYSIMD_FLOAT32_C(   535.05), EASYSIMD_FLOAT32_C(    43.85), EASYSIMD_FLOAT32_C(  -963.94), EASYSIMD_FLOAT32_C(   235.87),
                         EASYSIMD_FLOAT32_C(   143.93), EASYSIMD_FLOAT32_C(  -236.80), EASYSIMD_FLOAT32_C(   550.36), EASYSIMD_FLOAT32_C(    -8.58),
                         EASYSIMD_FLOAT32_C(   374.16), EASYSIMD_FLOAT32_C(   714.91), EASYSIMD_FLOAT32_C(  -355.51), EASYSIMD_FLOAT32_C(  -520.52)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   493.79), EASYSIMD_FLOAT32_C(    29.44), EASYSIMD_FLOAT32_C(   941.83), EASYSIMD_FLOAT32_C(   567.95),
                         EASYSIMD_FLOAT32_C(   535.05), EASYSIMD_FLOAT32_C(    43.85), EASYSIMD_FLOAT32_C(   963.94), EASYSIMD_FLOAT32_C(   235.87),
                         EASYSIMD_FLOAT32_C(   143.93), EASYSIMD_FLOAT32_C(   236.80), EASYSIMD_FLOAT32_C(   550.36), EASYSIMD_FLOAT32_C(     8.58),
                         EASYSIMD_FLOAT32_C(   374.16), EASYSIMD_FLOAT32_C(   714.91), EASYSIMD_FLOAT32_C(   355.51), EASYSIMD_FLOAT32_C(   520.52)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = test_vec[i].a;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_abs_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_abs_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_abs_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 src;
    easysimd__mmask16 k;
    easysimd__m512 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -319.23), EASYSIMD_FLOAT32_C(   773.21), EASYSIMD_FLOAT32_C(   -25.13), EASYSIMD_FLOAT32_C(   859.40),
                         EASYSIMD_FLOAT32_C(  -184.67), EASYSIMD_FLOAT32_C(   833.56), EASYSIMD_FLOAT32_C(  -441.84), EASYSIMD_FLOAT32_C(  -761.98),
                         EASYSIMD_FLOAT32_C(  -874.37), EASYSIMD_FLOAT32_C(  -592.77), EASYSIMD_FLOAT32_C(   252.42), EASYSIMD_FLOAT32_C(  -882.47),
                         EASYSIMD_FLOAT32_C(  -558.99), EASYSIMD_FLOAT32_C(  -481.72), EASYSIMD_FLOAT32_C(   686.76), EASYSIMD_FLOAT32_C(   728.61)),
      UINT16_C(15540),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    76.28), EASYSIMD_FLOAT32_C(   655.09), EASYSIMD_FLOAT32_C(  -928.32), EASYSIMD_FLOAT32_C(  -847.53),
                         EASYSIMD_FLOAT32_C(   388.54), EASYSIMD_FLOAT32_C(   102.38), EASYSIMD_FLOAT32_C(  -722.29), EASYSIMD_FLOAT32_C(  -821.42),
                         EASYSIMD_FLOAT32_C(   747.74), EASYSIMD_FLOAT32_C(   751.90), EASYSIMD_FLOAT32_C(  -708.81), EASYSIMD_FLOAT32_C(  -787.46),
                         EASYSIMD_FLOAT32_C(  -140.56), EASYSIMD_FLOAT32_C(   240.08), EASYSIMD_FLOAT32_C(   489.35), EASYSIMD_FLOAT32_C(  -206.54)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -319.23), EASYSIMD_FLOAT32_C(   773.21), EASYSIMD_FLOAT32_C(   928.32), EASYSIMD_FLOAT32_C(   847.53),
                         EASYSIMD_FLOAT32_C(   388.54), EASYSIMD_FLOAT32_C(   102.38), EASYSIMD_FLOAT32_C(  -441.84), EASYSIMD_FLOAT32_C(  -761.98),
                         EASYSIMD_FLOAT32_C(   747.74), EASYSIMD_FLOAT32_C(  -592.77), EASYSIMD_FLOAT32_C(   708.81), EASYSIMD_FLOAT32_C(   787.46),
                         EASYSIMD_FLOAT32_C(  -558.99), EASYSIMD_FLOAT32_C(   240.08), EASYSIMD_FLOAT32_C(   686.76), EASYSIMD_FLOAT32_C(   728.61)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -947.60), EASYSIMD_FLOAT32_C(   966.97), EASYSIMD_FLOAT32_C(   360.07), EASYSIMD_FLOAT32_C(  -749.79),
                         EASYSIMD_FLOAT32_C(  -104.12), EASYSIMD_FLOAT32_C(  -901.25), EASYSIMD_FLOAT32_C(   539.88), EASYSIMD_FLOAT32_C(   773.77),
                         EASYSIMD_FLOAT32_C(  -254.72), EASYSIMD_FLOAT32_C(   594.77), EASYSIMD_FLOAT32_C(   108.08), EASYSIMD_FLOAT32_C(   885.82),
                         EASYSIMD_FLOAT32_C(  -408.70), EASYSIMD_FLOAT32_C(   866.84), EASYSIMD_FLOAT32_C(   121.17), EASYSIMD_FLOAT32_C(  -622.26)),
      UINT16_C( 6415),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -617.33), EASYSIMD_FLOAT32_C(   530.77), EASYSIMD_FLOAT32_C(   -63.51), EASYSIMD_FLOAT32_C(  -846.61),
                         EASYSIMD_FLOAT32_C(   510.77), EASYSIMD_FLOAT32_C(  -838.06), EASYSIMD_FLOAT32_C(   -89.58), EASYSIMD_FLOAT32_C(    88.35),
                         EASYSIMD_FLOAT32_C(  -729.20), EASYSIMD_FLOAT32_C(  -645.61), EASYSIMD_FLOAT32_C(  -171.69), EASYSIMD_FLOAT32_C(    -7.24),
                         EASYSIMD_FLOAT32_C(   296.84), EASYSIMD_FLOAT32_C(   -40.31), EASYSIMD_FLOAT32_C(  -660.11), EASYSIMD_FLOAT32_C(   988.31)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -947.60), EASYSIMD_FLOAT32_C(   966.97), EASYSIMD_FLOAT32_C(   360.07), EASYSIMD_FLOAT32_C(   846.61),
                         EASYSIMD_FLOAT32_C(   510.77), EASYSIMD_FLOAT32_C(  -901.25), EASYSIMD_FLOAT32_C(   539.88), EASYSIMD_FLOAT32_C(    88.35),
                         EASYSIMD_FLOAT32_C(  -254.72), EASYSIMD_FLOAT32_C(   594.77), EASYSIMD_FLOAT32_C(   108.08), EASYSIMD_FLOAT32_C(   885.82),
                         EASYSIMD_FLOAT32_C(   296.84), EASYSIMD_FLOAT32_C(    40.31), EASYSIMD_FLOAT32_C(   660.11), EASYSIMD_FLOAT32_C(   988.31)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -393.36), EASYSIMD_FLOAT32_C(   670.22), EASYSIMD_FLOAT32_C(   299.00), EASYSIMD_FLOAT32_C(   676.51),
                         EASYSIMD_FLOAT32_C(   528.75), EASYSIMD_FLOAT32_C(   146.55), EASYSIMD_FLOAT32_C(  -505.05), EASYSIMD_FLOAT32_C(  -388.48),
                         EASYSIMD_FLOAT32_C(   943.89), EASYSIMD_FLOAT32_C(   695.54), EASYSIMD_FLOAT32_C(  -377.80), EASYSIMD_FLOAT32_C(   778.00),
                         EASYSIMD_FLOAT32_C(  -489.06), EASYSIMD_FLOAT32_C(  -749.41), EASYSIMD_FLOAT32_C(  -423.37), EASYSIMD_FLOAT32_C(  -118.88)),
      UINT16_C( 1525),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -613.16), EASYSIMD_FLOAT32_C(   -56.94), EASYSIMD_FLOAT32_C(   525.11), EASYSIMD_FLOAT32_C(   814.48),
                         EASYSIMD_FLOAT32_C(  -481.76), EASYSIMD_FLOAT32_C(  -375.20), EASYSIMD_FLOAT32_C(   199.14), EASYSIMD_FLOAT32_C(   833.96),
                         EASYSIMD_FLOAT32_C(  -212.57), EASYSIMD_FLOAT32_C(   651.63), EASYSIMD_FLOAT32_C(  -814.45), EASYSIMD_FLOAT32_C(   640.68),
                         EASYSIMD_FLOAT32_C(   377.67), EASYSIMD_FLOAT32_C(   933.74), EASYSIMD_FLOAT32_C(   193.12), EASYSIMD_FLOAT32_C(  -194.06)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -393.36), EASYSIMD_FLOAT32_C(   670.22), EASYSIMD_FLOAT32_C(   299.00), EASYSIMD_FLOAT32_C(   676.51),
                         EASYSIMD_FLOAT32_C(   528.75), EASYSIMD_FLOAT32_C(   375.20), EASYSIMD_FLOAT32_C(  -505.05), EASYSIMD_FLOAT32_C(   833.96),
                         EASYSIMD_FLOAT32_C(   212.57), EASYSIMD_FLOAT32_C(   651.63), EASYSIMD_FLOAT32_C(   814.45), EASYSIMD_FLOAT32_C(   640.68),
                         EASYSIMD_FLOAT32_C(  -489.06), EASYSIMD_FLOAT32_C(   933.74), EASYSIMD_FLOAT32_C(  -423.37), EASYSIMD_FLOAT32_C(   194.06)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    61.51), EASYSIMD_FLOAT32_C(  -643.69), EASYSIMD_FLOAT32_C(   -16.59), EASYSIMD_FLOAT32_C(   -29.44),
                         EASYSIMD_FLOAT32_C(  -567.95), EASYSIMD_FLOAT32_C(    43.85), EASYSIMD_FLOAT32_C(   235.87), EASYSIMD_FLOAT32_C(  -236.80),
                         EASYSIMD_FLOAT32_C(    -8.58), EASYSIMD_FLOAT32_C(   714.91), EASYSIMD_FLOAT32_C(  -520.52), EASYSIMD_FLOAT32_C(  -546.29),
                         EASYSIMD_FLOAT32_C(  -104.88), EASYSIMD_FLOAT32_C(  -744.23), EASYSIMD_FLOAT32_C(   495.73), EASYSIMD_FLOAT32_C(   262.00)),
      UINT16_C(29879),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   515.30), EASYSIMD_FLOAT32_C(   896.28), EASYSIMD_FLOAT32_C(   660.35), EASYSIMD_FLOAT32_C(   493.79),
                         EASYSIMD_FLOAT32_C(  -941.83), EASYSIMD_FLOAT32_C(   535.05), EASYSIMD_FLOAT32_C(  -963.94), EASYSIMD_FLOAT32_C(   143.93),
                         EASYSIMD_FLOAT32_C(   550.36), EASYSIMD_FLOAT32_C(   374.16), EASYSIMD_FLOAT32_C(  -355.51), EASYSIMD_FLOAT32_C(   -13.73),
                         EASYSIMD_FLOAT32_C(  -787.44), EASYSIMD_FLOAT32_C(   979.47), EASYSIMD_FLOAT32_C(   836.15), EASYSIMD_FLOAT32_C(  -301.39)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    61.51), EASYSIMD_FLOAT32_C(   896.28), EASYSIMD_FLOAT32_C(   660.35), EASYSIMD_FLOAT32_C(   493.79),
                         EASYSIMD_FLOAT32_C(  -567.95), EASYSIMD_FLOAT32_C(   535.05), EASYSIMD_FLOAT32_C(   235.87), EASYSIMD_FLOAT32_C(  -236.80),
                         EASYSIMD_FLOAT32_C(   550.36), EASYSIMD_FLOAT32_C(   714.91), EASYSIMD_FLOAT32_C(   355.51), EASYSIMD_FLOAT32_C(    13.73),
                         EASYSIMD_FLOAT32_C(  -104.88), EASYSIMD_FLOAT32_C(   979.47), EASYSIMD_FLOAT32_C(   836.15), EASYSIMD_FLOAT32_C(   301.39)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   148.99), EASYSIMD_FLOAT32_C(  -963.65), EASYSIMD_FLOAT32_C(   149.45), EASYSIMD_FLOAT32_C(  -850.34),
                         EASYSIMD_FLOAT32_C(  -524.37), EASYSIMD_FLOAT32_C(  -513.69), EASYSIMD_FLOAT32_C(    22.08), EASYSIMD_FLOAT32_C(   488.53),
                         EASYSIMD_FLOAT32_C(   770.65), EASYSIMD_FLOAT32_C(   491.66), EASYSIMD_FLOAT32_C(    89.59), EASYSIMD_FLOAT32_C(   924.64),
                         EASYSIMD_FLOAT32_C(  -763.40), EASYSIMD_FLOAT32_C(  -404.62), EASYSIMD_FLOAT32_C(  -957.75), EASYSIMD_FLOAT32_C(   281.78)),
      UINT16_C(44157),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -979.51), EASYSIMD_FLOAT32_C(  -129.70), EASYSIMD_FLOAT32_C(  -587.42), EASYSIMD_FLOAT32_C(    94.97),
                         EASYSIMD_FLOAT32_C(  -887.16), EASYSIMD_FLOAT32_C(  -189.75), EASYSIMD_FLOAT32_C(   881.78), EASYSIMD_FLOAT32_C(  -152.81),
                         EASYSIMD_FLOAT32_C(   943.19), EASYSIMD_FLOAT32_C(  -229.02), EASYSIMD_FLOAT32_C(  -577.41), EASYSIMD_FLOAT32_C(  -719.96),
                         EASYSIMD_FLOAT32_C(   770.58), EASYSIMD_FLOAT32_C(  -153.52), EASYSIMD_FLOAT32_C(  -991.64), EASYSIMD_FLOAT32_C(   -53.48)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   979.51), EASYSIMD_FLOAT32_C(  -963.65), EASYSIMD_FLOAT32_C(   587.42), EASYSIMD_FLOAT32_C(  -850.34),
                         EASYSIMD_FLOAT32_C(   887.16), EASYSIMD_FLOAT32_C(   189.75), EASYSIMD_FLOAT32_C(    22.08), EASYSIMD_FLOAT32_C(   488.53),
                         EASYSIMD_FLOAT32_C(   770.65), EASYSIMD_FLOAT32_C(   229.02), EASYSIMD_FLOAT32_C(   577.41), EASYSIMD_FLOAT32_C(   719.96),
                         EASYSIMD_FLOAT32_C(   770.58), EASYSIMD_FLOAT32_C(   153.52), EASYSIMD_FLOAT32_C(  -957.75), EASYSIMD_FLOAT32_C(    53.48)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   631.94), EASYSIMD_FLOAT32_C(  -409.79), EASYSIMD_FLOAT32_C(   668.07), EASYSIMD_FLOAT32_C(   542.88),
                         EASYSIMD_FLOAT32_C(  -896.06), EASYSIMD_FLOAT32_C(   248.80), EASYSIMD_FLOAT32_C(   200.01), EASYSIMD_FLOAT32_C(   669.33),
                         EASYSIMD_FLOAT32_C(  -642.07), EASYSIMD_FLOAT32_C(  -212.55), EASYSIMD_FLOAT32_C(  -356.51), EASYSIMD_FLOAT32_C(  -440.95),
                         EASYSIMD_FLOAT32_C(  -982.52), EASYSIMD_FLOAT32_C(  -842.67), EASYSIMD_FLOAT32_C(  -420.59), EASYSIMD_FLOAT32_C(  -949.02)),
      UINT16_C(15240),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   966.19), EASYSIMD_FLOAT32_C(   529.24), EASYSIMD_FLOAT32_C(  -544.06), EASYSIMD_FLOAT32_C(  -881.83),
                         EASYSIMD_FLOAT32_C(  -242.38), EASYSIMD_FLOAT32_C(  -380.44), EASYSIMD_FLOAT32_C(  -752.70), EASYSIMD_FLOAT32_C(  -160.45),
                         EASYSIMD_FLOAT32_C(   773.41), EASYSIMD_FLOAT32_C(  -474.98), EASYSIMD_FLOAT32_C(   573.78), EASYSIMD_FLOAT32_C(  -190.69),
                         EASYSIMD_FLOAT32_C(  -743.99), EASYSIMD_FLOAT32_C(  -698.61), EASYSIMD_FLOAT32_C(  -633.81), EASYSIMD_FLOAT32_C(   938.50)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   631.94), EASYSIMD_FLOAT32_C(  -409.79), EASYSIMD_FLOAT32_C(   544.06), EASYSIMD_FLOAT32_C(   881.83),
                         EASYSIMD_FLOAT32_C(   242.38), EASYSIMD_FLOAT32_C(   248.80), EASYSIMD_FLOAT32_C(   752.70), EASYSIMD_FLOAT32_C(   160.45),
                         EASYSIMD_FLOAT32_C(   773.41), EASYSIMD_FLOAT32_C(  -212.55), EASYSIMD_FLOAT32_C(  -356.51), EASYSIMD_FLOAT32_C(  -440.95),
                         EASYSIMD_FLOAT32_C(   743.99), EASYSIMD_FLOAT32_C(  -842.67), EASYSIMD_FLOAT32_C(  -420.59), EASYSIMD_FLOAT32_C(  -949.02)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   156.92), EASYSIMD_FLOAT32_C(  -736.34), EASYSIMD_FLOAT32_C(   166.92), EASYSIMD_FLOAT32_C(   300.41),
                         EASYSIMD_FLOAT32_C(  -295.98), EASYSIMD_FLOAT32_C(  -702.22), EASYSIMD_FLOAT32_C(  -740.49), EASYSIMD_FLOAT32_C(   -80.99),
                         EASYSIMD_FLOAT32_C(  -785.06), EASYSIMD_FLOAT32_C(    87.65), EASYSIMD_FLOAT32_C(  -482.52), EASYSIMD_FLOAT32_C(  -681.02),
                         EASYSIMD_FLOAT32_C(   764.25), EASYSIMD_FLOAT32_C(   305.46), EASYSIMD_FLOAT32_C(   526.44), EASYSIMD_FLOAT32_C(   369.20)),
      UINT16_C(49024),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   809.64), EASYSIMD_FLOAT32_C(  -790.72), EASYSIMD_FLOAT32_C(   295.53), EASYSIMD_FLOAT32_C(  -856.33),
                         EASYSIMD_FLOAT32_C(   237.04), EASYSIMD_FLOAT32_C(  -607.75), EASYSIMD_FLOAT32_C(  -732.96), EASYSIMD_FLOAT32_C(  -497.56),
                         EASYSIMD_FLOAT32_C(  -918.03), EASYSIMD_FLOAT32_C(   488.66), EASYSIMD_FLOAT32_C(  -523.80), EASYSIMD_FLOAT32_C(  -224.58),
                         EASYSIMD_FLOAT32_C(   298.04), EASYSIMD_FLOAT32_C(   606.61), EASYSIMD_FLOAT32_C(  -852.36), EASYSIMD_FLOAT32_C(  -314.42)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   809.64), EASYSIMD_FLOAT32_C(  -736.34), EASYSIMD_FLOAT32_C(   295.53), EASYSIMD_FLOAT32_C(   856.33),
                         EASYSIMD_FLOAT32_C(   237.04), EASYSIMD_FLOAT32_C(   607.75), EASYSIMD_FLOAT32_C(   732.96), EASYSIMD_FLOAT32_C(   497.56),
                         EASYSIMD_FLOAT32_C(   918.03), EASYSIMD_FLOAT32_C(    87.65), EASYSIMD_FLOAT32_C(  -482.52), EASYSIMD_FLOAT32_C(  -681.02),
                         EASYSIMD_FLOAT32_C(   764.25), EASYSIMD_FLOAT32_C(   305.46), EASYSIMD_FLOAT32_C(   526.44), EASYSIMD_FLOAT32_C(   369.20)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -897.44), EASYSIMD_FLOAT32_C(   113.95), EASYSIMD_FLOAT32_C(   654.01), EASYSIMD_FLOAT32_C(   620.23),
                         EASYSIMD_FLOAT32_C(   623.09), EASYSIMD_FLOAT32_C(  -407.46), EASYSIMD_FLOAT32_C(  -763.16), EASYSIMD_FLOAT32_C(  -768.89),
                         EASYSIMD_FLOAT32_C(   966.30), EASYSIMD_FLOAT32_C(   863.50), EASYSIMD_FLOAT32_C(   709.25), EASYSIMD_FLOAT32_C(   348.50),
                         EASYSIMD_FLOAT32_C(  -816.66), EASYSIMD_FLOAT32_C(  -662.92), EASYSIMD_FLOAT32_C(   913.50), EASYSIMD_FLOAT32_C(   301.72)),
      UINT16_C(64661),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -54.30), EASYSIMD_FLOAT32_C(  -771.33), EASYSIMD_FLOAT32_C(   -34.80), EASYSIMD_FLOAT32_C(   -55.97),
                         EASYSIMD_FLOAT32_C(  -654.29), EASYSIMD_FLOAT32_C(   768.64), EASYSIMD_FLOAT32_C(  -409.48), EASYSIMD_FLOAT32_C(   859.32),
                         EASYSIMD_FLOAT32_C(  -160.39), EASYSIMD_FLOAT32_C(  -988.34), EASYSIMD_FLOAT32_C(  -518.87), EASYSIMD_FLOAT32_C(  -778.28),
                         EASYSIMD_FLOAT32_C(   357.12), EASYSIMD_FLOAT32_C(   449.29), EASYSIMD_FLOAT32_C(   -46.50), EASYSIMD_FLOAT32_C(    93.99)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    54.30), EASYSIMD_FLOAT32_C(   771.33), EASYSIMD_FLOAT32_C(    34.80), EASYSIMD_FLOAT32_C(    55.97),
                         EASYSIMD_FLOAT32_C(   654.29), EASYSIMD_FLOAT32_C(   768.64), EASYSIMD_FLOAT32_C(  -763.16), EASYSIMD_FLOAT32_C(  -768.89),
                         EASYSIMD_FLOAT32_C(   160.39), EASYSIMD_FLOAT32_C(   863.50), EASYSIMD_FLOAT32_C(   709.25), EASYSIMD_FLOAT32_C(   778.28),
                         EASYSIMD_FLOAT32_C(  -816.66), EASYSIMD_FLOAT32_C(   449.29), EASYSIMD_FLOAT32_C(   913.50), EASYSIMD_FLOAT32_C(    93.99)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512  src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512    a = test_vec[i].a;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_abs_ps(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_abs_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_abs_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -140.56), EASYSIMD_FLOAT64_C( -558.99),
                         EASYSIMD_FLOAT64_C(  240.08), EASYSIMD_FLOAT64_C( -481.72),
                         EASYSIMD_FLOAT64_C(  489.35), EASYSIMD_FLOAT64_C(  686.76),
                         EASYSIMD_FLOAT64_C( -206.54), EASYSIMD_FLOAT64_C(  728.61)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  140.56), EASYSIMD_FLOAT64_C(  558.99),
                         EASYSIMD_FLOAT64_C(  240.08), EASYSIMD_FLOAT64_C(  481.72),
                         EASYSIMD_FLOAT64_C(  489.35), EASYSIMD_FLOAT64_C(  686.76),
                         EASYSIMD_FLOAT64_C(  206.54), EASYSIMD_FLOAT64_C(  728.61)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  747.74), EASYSIMD_FLOAT64_C( -874.37),
                         EASYSIMD_FLOAT64_C(  751.90), EASYSIMD_FLOAT64_C( -592.77),
                         EASYSIMD_FLOAT64_C( -708.81), EASYSIMD_FLOAT64_C(  252.42),
                         EASYSIMD_FLOAT64_C( -787.46), EASYSIMD_FLOAT64_C( -882.47)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  747.74), EASYSIMD_FLOAT64_C(  874.37),
                         EASYSIMD_FLOAT64_C(  751.90), EASYSIMD_FLOAT64_C(  592.77),
                         EASYSIMD_FLOAT64_C(  708.81), EASYSIMD_FLOAT64_C(  252.42),
                         EASYSIMD_FLOAT64_C(  787.46), EASYSIMD_FLOAT64_C(  882.47)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  388.54), EASYSIMD_FLOAT64_C( -184.67),
                         EASYSIMD_FLOAT64_C(  102.38), EASYSIMD_FLOAT64_C(  833.56),
                         EASYSIMD_FLOAT64_C( -722.29), EASYSIMD_FLOAT64_C( -441.84),
                         EASYSIMD_FLOAT64_C( -821.42), EASYSIMD_FLOAT64_C( -761.98)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  388.54), EASYSIMD_FLOAT64_C(  184.67),
                         EASYSIMD_FLOAT64_C(  102.38), EASYSIMD_FLOAT64_C(  833.56),
                         EASYSIMD_FLOAT64_C(  722.29), EASYSIMD_FLOAT64_C(  441.84),
                         EASYSIMD_FLOAT64_C(  821.42), EASYSIMD_FLOAT64_C(  761.98)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   76.28), EASYSIMD_FLOAT64_C( -319.23),
                         EASYSIMD_FLOAT64_C(  655.09), EASYSIMD_FLOAT64_C(  773.21),
                         EASYSIMD_FLOAT64_C( -928.32), EASYSIMD_FLOAT64_C(  -25.13),
                         EASYSIMD_FLOAT64_C( -847.53), EASYSIMD_FLOAT64_C(  859.40)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   76.28), EASYSIMD_FLOAT64_C(  319.23),
                         EASYSIMD_FLOAT64_C(  655.09), EASYSIMD_FLOAT64_C(  773.21),
                         EASYSIMD_FLOAT64_C(  928.32), EASYSIMD_FLOAT64_C(   25.13),
                         EASYSIMD_FLOAT64_C(  847.53), EASYSIMD_FLOAT64_C(  859.40)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -40.31), EASYSIMD_FLOAT64_C(  866.84),
                         EASYSIMD_FLOAT64_C( -660.11), EASYSIMD_FLOAT64_C(  121.17),
                         EASYSIMD_FLOAT64_C(  988.31), EASYSIMD_FLOAT64_C( -622.26),
                         EASYSIMD_FLOAT64_C(  206.00), EASYSIMD_FLOAT64_C(  520.48)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   40.31), EASYSIMD_FLOAT64_C(  866.84),
                         EASYSIMD_FLOAT64_C(  660.11), EASYSIMD_FLOAT64_C(  121.17),
                         EASYSIMD_FLOAT64_C(  988.31), EASYSIMD_FLOAT64_C(  622.26),
                         EASYSIMD_FLOAT64_C(  206.00), EASYSIMD_FLOAT64_C(  520.48)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -645.61), EASYSIMD_FLOAT64_C(  594.77),
                         EASYSIMD_FLOAT64_C( -171.69), EASYSIMD_FLOAT64_C(  108.08),
                         EASYSIMD_FLOAT64_C(   -7.24), EASYSIMD_FLOAT64_C(  885.82),
                         EASYSIMD_FLOAT64_C(  296.84), EASYSIMD_FLOAT64_C( -408.70)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  645.61), EASYSIMD_FLOAT64_C(  594.77),
                         EASYSIMD_FLOAT64_C(  171.69), EASYSIMD_FLOAT64_C(  108.08),
                         EASYSIMD_FLOAT64_C(    7.24), EASYSIMD_FLOAT64_C(  885.82),
                         EASYSIMD_FLOAT64_C(  296.84), EASYSIMD_FLOAT64_C(  408.70)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -838.06), EASYSIMD_FLOAT64_C( -901.25),
                         EASYSIMD_FLOAT64_C(  -89.58), EASYSIMD_FLOAT64_C(  539.88),
                         EASYSIMD_FLOAT64_C(   88.35), EASYSIMD_FLOAT64_C(  773.77),
                         EASYSIMD_FLOAT64_C( -729.20), EASYSIMD_FLOAT64_C( -254.72)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  838.06), EASYSIMD_FLOAT64_C(  901.25),
                         EASYSIMD_FLOAT64_C(   89.58), EASYSIMD_FLOAT64_C(  539.88),
                         EASYSIMD_FLOAT64_C(   88.35), EASYSIMD_FLOAT64_C(  773.77),
                         EASYSIMD_FLOAT64_C(  729.20), EASYSIMD_FLOAT64_C(  254.72)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  530.77), EASYSIMD_FLOAT64_C(  966.97),
                         EASYSIMD_FLOAT64_C(  -63.51), EASYSIMD_FLOAT64_C(  360.07),
                         EASYSIMD_FLOAT64_C( -846.61), EASYSIMD_FLOAT64_C( -749.79),
                         EASYSIMD_FLOAT64_C(  510.77), EASYSIMD_FLOAT64_C( -104.12)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  530.77), EASYSIMD_FLOAT64_C(  966.97),
                         EASYSIMD_FLOAT64_C(   63.51), EASYSIMD_FLOAT64_C(  360.07),
                         EASYSIMD_FLOAT64_C(  846.61), EASYSIMD_FLOAT64_C(  749.79),
                         EASYSIMD_FLOAT64_C(  510.77), EASYSIMD_FLOAT64_C(  104.12)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d a = test_vec[i].a;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_abs_pd(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_abs_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_abs_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d src;
    easysimd__mmask8 k;
    easysimd__m512d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -874.37), EASYSIMD_FLOAT64_C( -592.77),
                         EASYSIMD_FLOAT64_C(  252.42), EASYSIMD_FLOAT64_C( -882.47),
                         EASYSIMD_FLOAT64_C( -558.99), EASYSIMD_FLOAT64_C( -481.72),
                         EASYSIMD_FLOAT64_C(  686.76), EASYSIMD_FLOAT64_C(  728.61)),
      UINT8_C( 67),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  747.74), EASYSIMD_FLOAT64_C(  751.90),
                         EASYSIMD_FLOAT64_C( -708.81), EASYSIMD_FLOAT64_C( -787.46),
                         EASYSIMD_FLOAT64_C( -140.56), EASYSIMD_FLOAT64_C(  240.08),
                         EASYSIMD_FLOAT64_C(  489.35), EASYSIMD_FLOAT64_C( -206.54)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -874.37), EASYSIMD_FLOAT64_C(  751.90),
                         EASYSIMD_FLOAT64_C(  252.42), EASYSIMD_FLOAT64_C( -882.47),
                         EASYSIMD_FLOAT64_C( -558.99), EASYSIMD_FLOAT64_C( -481.72),
                         EASYSIMD_FLOAT64_C(  489.35), EASYSIMD_FLOAT64_C(  206.54)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   76.28), EASYSIMD_FLOAT64_C(  655.09),
                         EASYSIMD_FLOAT64_C( -928.32), EASYSIMD_FLOAT64_C( -847.53),
                         EASYSIMD_FLOAT64_C(  388.54), EASYSIMD_FLOAT64_C(  102.38),
                         EASYSIMD_FLOAT64_C( -722.29), EASYSIMD_FLOAT64_C( -821.42)),
      UINT8_C(153),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  520.48), EASYSIMD_FLOAT64_C( -319.23),
                         EASYSIMD_FLOAT64_C(  773.21), EASYSIMD_FLOAT64_C(  -25.13),
                         EASYSIMD_FLOAT64_C(  859.40), EASYSIMD_FLOAT64_C( -184.67),
                         EASYSIMD_FLOAT64_C(  833.56), EASYSIMD_FLOAT64_C( -441.84)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  520.48), EASYSIMD_FLOAT64_C(  655.09),
                         EASYSIMD_FLOAT64_C( -928.32), EASYSIMD_FLOAT64_C(   25.13),
                         EASYSIMD_FLOAT64_C(  859.40), EASYSIMD_FLOAT64_C(  102.38),
                         EASYSIMD_FLOAT64_C( -722.29), EASYSIMD_FLOAT64_C(  441.84)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -254.72), EASYSIMD_FLOAT64_C(  594.77),
                         EASYSIMD_FLOAT64_C(  108.08), EASYSIMD_FLOAT64_C(  885.82),
                         EASYSIMD_FLOAT64_C( -408.70), EASYSIMD_FLOAT64_C(  866.84),
                         EASYSIMD_FLOAT64_C(  121.17), EASYSIMD_FLOAT64_C( -622.26)),
      UINT8_C( 41),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -729.20), EASYSIMD_FLOAT64_C( -645.61),
                         EASYSIMD_FLOAT64_C( -171.69), EASYSIMD_FLOAT64_C(   -7.24),
                         EASYSIMD_FLOAT64_C(  296.84), EASYSIMD_FLOAT64_C(  -40.31),
                         EASYSIMD_FLOAT64_C( -660.11), EASYSIMD_FLOAT64_C(  988.31)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -254.72), EASYSIMD_FLOAT64_C(  594.77),
                         EASYSIMD_FLOAT64_C(  171.69), EASYSIMD_FLOAT64_C(  885.82),
                         EASYSIMD_FLOAT64_C(  296.84), EASYSIMD_FLOAT64_C(  866.84),
                         EASYSIMD_FLOAT64_C(  121.17), EASYSIMD_FLOAT64_C(  988.31)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -617.33), EASYSIMD_FLOAT64_C(  530.77),
                         EASYSIMD_FLOAT64_C(  -63.51), EASYSIMD_FLOAT64_C( -846.61),
                         EASYSIMD_FLOAT64_C(  510.77), EASYSIMD_FLOAT64_C( -838.06),
                         EASYSIMD_FLOAT64_C(  -89.58), EASYSIMD_FLOAT64_C(   88.35)),
      UINT8_C(208),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -506.16), EASYSIMD_FLOAT64_C( -947.60),
                         EASYSIMD_FLOAT64_C(  966.97), EASYSIMD_FLOAT64_C(  360.07),
                         EASYSIMD_FLOAT64_C( -749.79), EASYSIMD_FLOAT64_C( -104.12),
                         EASYSIMD_FLOAT64_C( -901.25), EASYSIMD_FLOAT64_C(  539.88)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  506.16), EASYSIMD_FLOAT64_C(  947.60),
                         EASYSIMD_FLOAT64_C(  -63.51), EASYSIMD_FLOAT64_C(  360.07),
                         EASYSIMD_FLOAT64_C(  510.77), EASYSIMD_FLOAT64_C( -838.06),
                         EASYSIMD_FLOAT64_C(  -89.58), EASYSIMD_FLOAT64_C(   88.35)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  943.89), EASYSIMD_FLOAT64_C(  695.54),
                         EASYSIMD_FLOAT64_C( -377.80), EASYSIMD_FLOAT64_C(  778.00),
                         EASYSIMD_FLOAT64_C( -489.06), EASYSIMD_FLOAT64_C( -749.41),
                         EASYSIMD_FLOAT64_C( -423.37), EASYSIMD_FLOAT64_C( -118.88)),
      UINT8_C( 52),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -212.57), EASYSIMD_FLOAT64_C(  651.63),
                         EASYSIMD_FLOAT64_C( -814.45), EASYSIMD_FLOAT64_C(  640.68),
                         EASYSIMD_FLOAT64_C(  377.67), EASYSIMD_FLOAT64_C(  933.74),
                         EASYSIMD_FLOAT64_C(  193.12), EASYSIMD_FLOAT64_C( -194.06)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  943.89), EASYSIMD_FLOAT64_C(  695.54),
                         EASYSIMD_FLOAT64_C(  814.45), EASYSIMD_FLOAT64_C(  640.68),
                         EASYSIMD_FLOAT64_C( -489.06), EASYSIMD_FLOAT64_C(  933.74),
                         EASYSIMD_FLOAT64_C( -423.37), EASYSIMD_FLOAT64_C( -118.88)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -613.16), EASYSIMD_FLOAT64_C(  -56.94),
                         EASYSIMD_FLOAT64_C(  525.11), EASYSIMD_FLOAT64_C(  814.48),
                         EASYSIMD_FLOAT64_C( -481.76), EASYSIMD_FLOAT64_C( -375.20),
                         EASYSIMD_FLOAT64_C(  199.14), EASYSIMD_FLOAT64_C(  833.96)),
      UINT8_C(108),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  536.10), EASYSIMD_FLOAT64_C( -393.36),
                         EASYSIMD_FLOAT64_C(  670.22), EASYSIMD_FLOAT64_C(  299.00),
                         EASYSIMD_FLOAT64_C(  676.51), EASYSIMD_FLOAT64_C(  528.75),
                         EASYSIMD_FLOAT64_C(  146.55), EASYSIMD_FLOAT64_C( -505.05)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -613.16), EASYSIMD_FLOAT64_C(  393.36),
                         EASYSIMD_FLOAT64_C(  670.22), EASYSIMD_FLOAT64_C(  814.48),
                         EASYSIMD_FLOAT64_C(  676.51), EASYSIMD_FLOAT64_C(  528.75),
                         EASYSIMD_FLOAT64_C(  199.14), EASYSIMD_FLOAT64_C(  833.96)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -8.58), EASYSIMD_FLOAT64_C(  714.91),
                         EASYSIMD_FLOAT64_C( -520.52), EASYSIMD_FLOAT64_C( -546.29),
                         EASYSIMD_FLOAT64_C( -104.88), EASYSIMD_FLOAT64_C( -744.23),
                         EASYSIMD_FLOAT64_C(  495.73), EASYSIMD_FLOAT64_C(  262.00)),
      UINT8_C(147),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  550.36), EASYSIMD_FLOAT64_C(  374.16),
                         EASYSIMD_FLOAT64_C( -355.51), EASYSIMD_FLOAT64_C(  -13.73),
                         EASYSIMD_FLOAT64_C( -787.44), EASYSIMD_FLOAT64_C(  979.47),
                         EASYSIMD_FLOAT64_C(  836.15), EASYSIMD_FLOAT64_C( -301.39)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  550.36), EASYSIMD_FLOAT64_C(  714.91),
                         EASYSIMD_FLOAT64_C( -520.52), EASYSIMD_FLOAT64_C(   13.73),
                         EASYSIMD_FLOAT64_C( -104.88), EASYSIMD_FLOAT64_C( -744.23),
                         EASYSIMD_FLOAT64_C(  836.15), EASYSIMD_FLOAT64_C(  301.39)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  515.30), EASYSIMD_FLOAT64_C(  896.28),
                         EASYSIMD_FLOAT64_C(  660.35), EASYSIMD_FLOAT64_C(  493.79),
                         EASYSIMD_FLOAT64_C( -941.83), EASYSIMD_FLOAT64_C(  535.05),
                         EASYSIMD_FLOAT64_C( -963.94), EASYSIMD_FLOAT64_C(  143.93)),
      UINT8_C( 75),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  520.16), EASYSIMD_FLOAT64_C(   61.51),
                         EASYSIMD_FLOAT64_C( -643.69), EASYSIMD_FLOAT64_C(  -16.59),
                         EASYSIMD_FLOAT64_C(  -29.44), EASYSIMD_FLOAT64_C( -567.95),
                         EASYSIMD_FLOAT64_C(   43.85), EASYSIMD_FLOAT64_C(  235.87)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  515.30), EASYSIMD_FLOAT64_C(   61.51),
                         EASYSIMD_FLOAT64_C(  660.35), EASYSIMD_FLOAT64_C(  493.79),
                         EASYSIMD_FLOAT64_C(   29.44), EASYSIMD_FLOAT64_C(  535.05),
                         EASYSIMD_FLOAT64_C(   43.85), EASYSIMD_FLOAT64_C(  235.87)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d  src = test_vec[i].src;
    easysimd__mmask8   k = test_vec[i].k;
    easysimd__m512d    a = test_vec[i].a;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_abs_pd(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_abs_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_abs_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_abs_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_abs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_abs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_abs_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_abs_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_abs_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_abs_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_abs_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_abs_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_abs_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_abs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_abs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_abs_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_abs_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_abs_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_abs_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_abs_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_abs_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_abs_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_abs_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_abs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_abs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_abs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_abs_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_abs_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_abs_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_abs_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_abs_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_abs_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_abs_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_abs_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_abs_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_abs_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
