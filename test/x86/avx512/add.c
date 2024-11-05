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

#define EASYSIMD_TEST_X86_AVX512_INSN add

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/add.h>

static int
test_easysimd_mm_mask_add_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int8_t src[16];
    const easysimd__mmask16 k;
    const int8_t a[16];
    const int8_t b[16];
    const int8_t r[16];
  } test_vec[] = {
           { { -INT8_C(  27), -INT8_C( 117), -INT8_C(   6),  INT8_C(  88), -INT8_C(  46),  INT8_C(  97), -INT8_C(  33),  INT8_C(  41),
        -INT8_C(  39), -INT8_C(   6), -INT8_C(  94),  INT8_C(  57), -INT8_C(  33),  INT8_C(  58),  INT8_C( 124), -INT8_C(  31) },
      UINT16_C( 5189),
      {  INT8_C( 116),  INT8_C(  54), -INT8_C(  13),  INT8_C(   7), -INT8_C(  49), -INT8_C(  33), -INT8_C( 119),  INT8_C(  78),
         INT8_C(  14), -INT8_C(  94),  INT8_C(  14), -INT8_C(  68), -INT8_C(  88),  INT8_C(  20),  INT8_C(  70), -INT8_C( 115) },
      {  INT8_C(   9), -INT8_C( 104), -INT8_C(  11), -INT8_C( 116), -INT8_C(  58), -INT8_C(  28),  INT8_C(  17), -INT8_C( 115),
        -INT8_C(  25), -INT8_C( 123), -INT8_C(  27),  INT8_C(  40),  INT8_C(  24), -INT8_C(  71), -INT8_C(  96), -INT8_C(  45) },
      {  INT8_C( 125), -INT8_C( 117), -INT8_C(  24),  INT8_C(  88), -INT8_C(  46),  INT8_C(  97), -INT8_C( 102),  INT8_C(  41),
        -INT8_C(  39), -INT8_C(   6), -INT8_C(  13),  INT8_C(  57), -INT8_C(  64),  INT8_C(  58),  INT8_C( 124), -INT8_C(  31) } },
    { { -INT8_C(  98), -INT8_C(  92), -INT8_C(  38), -INT8_C(  68), -INT8_C(  41),  INT8_C( 122), -INT8_C(  22), -INT8_C(  38),
         INT8_C(  27), -INT8_C(  33),  INT8_C( 109), -INT8_C(  84), -INT8_C(  65), -INT8_C(  97), -INT8_C(  87), -INT8_C(  26) },
      UINT16_C(48315),
      {  INT8_C(  33), -INT8_C(  21),  INT8_C( 102),  INT8_C( 107), -INT8_C(  24),  INT8_C( 118), -INT8_C(  19), -INT8_C( 100),
         INT8_C(  41),  INT8_C(  78), -INT8_C(  81),  INT8_C(  11),  INT8_C(  38), -INT8_C( 101),  INT8_C(  39), -INT8_C( 103) },
      {  INT8_C(  68), -INT8_C(  83),      INT8_MIN,  INT8_C( 118), -INT8_C( 117),  INT8_C( 112), -INT8_C(  32), -INT8_C( 114),
         INT8_C(  16), -INT8_C(  45),  INT8_C(  93),  INT8_C(  38), -INT8_C(  71), -INT8_C(  60),  INT8_C(  99), -INT8_C(  64) },
      {  INT8_C( 101), -INT8_C( 104), -INT8_C(  38), -INT8_C(  31),  INT8_C( 115), -INT8_C(  26), -INT8_C(  22),  INT8_C(  42),
         INT8_C(  27), -INT8_C(  33),  INT8_C(  12),  INT8_C(  49), -INT8_C(  33),  INT8_C(  95), -INT8_C(  87),  INT8_C(  89) } },
    { { -INT8_C(  32), -INT8_C(  57),  INT8_C(  78), -INT8_C(  38), -INT8_C(  31), -INT8_C(  49), -INT8_C(   2), -INT8_C(  91),
         INT8_C( 122), -INT8_C( 124),  INT8_C(  43),  INT8_C( 111),  INT8_C(  13), -INT8_C(  53), -INT8_C(  99), -INT8_C(   7) },
      UINT16_C(55519),
      { -INT8_C( 112),  INT8_C(  59),  INT8_C(  27), -INT8_C(  75), -INT8_C(  30),  INT8_C(  46),  INT8_C(  88), -INT8_C(  37),
         INT8_C(  68), -INT8_C(  86), -INT8_C(  84), -INT8_C(  81),  INT8_C(  73), -INT8_C(  81),  INT8_C( 111),  INT8_C(  71) },
      {  INT8_C(  75),  INT8_C( 105),  INT8_C(  40), -INT8_C(  50), -INT8_C(  66),  INT8_C( 107), -INT8_C(  47),  INT8_C( 108),
        -INT8_C(  44), -INT8_C(  70), -INT8_C( 126),  INT8_C(  70),  INT8_C(  82), -INT8_C(  48), -INT8_C(  16), -INT8_C(  11) },
      { -INT8_C(  37), -INT8_C(  92),  INT8_C(  67), -INT8_C( 125), -INT8_C(  96), -INT8_C(  49),  INT8_C(  41),  INT8_C(  71),
         INT8_C( 122), -INT8_C( 124),  INT8_C(  43), -INT8_C(  11), -INT8_C( 101), -INT8_C(  53),  INT8_C(  95),  INT8_C(  60) } },
    { { -INT8_C(  10),  INT8_C(  74),  INT8_C(  22),      INT8_MAX, -INT8_C(  86), -INT8_C(  38), -INT8_C(  55), -INT8_C(  34),
         INT8_C(  37), -INT8_C(  93),  INT8_C( 113),  INT8_C(  90), -INT8_C(  22),  INT8_C(  95), -INT8_C(  30),  INT8_C(  36) },
      UINT16_C(49528),
      { -INT8_C(  86), -INT8_C(  18),  INT8_C(   8), -INT8_C(  80),  INT8_C(  64), -INT8_C( 123), -INT8_C(  65),  INT8_C(  70),
         INT8_C(  76), -INT8_C(  38),  INT8_C(  79),  INT8_C(   7), -INT8_C( 112), -INT8_C(  51),  INT8_C(  59),  INT8_C(  55) },
      { -INT8_C( 115), -INT8_C(  53), -INT8_C( 116), -INT8_C( 104),  INT8_C(  80),  INT8_C(  21),  INT8_C(  61),  INT8_C(   7),
        -INT8_C(  99), -INT8_C(  92), -INT8_C(  35),  INT8_C(  32), -INT8_C(   2),  INT8_C(  80),  INT8_C( 120), -INT8_C(  12) },
      { -INT8_C(  10),  INT8_C(  74),  INT8_C(  22),  INT8_C(  72), -INT8_C( 112), -INT8_C( 102), -INT8_C(   4), -INT8_C(  34),
        -INT8_C(  23), -INT8_C(  93),  INT8_C( 113),  INT8_C(  90), -INT8_C(  22),  INT8_C(  95), -INT8_C(  77),  INT8_C(  43) } },
    { {  INT8_C(   4),      INT8_MIN, -INT8_C(  71), -INT8_C(  17), -INT8_C(  48), -INT8_C(   5),      INT8_MAX,  INT8_C( 111),
         INT8_C( 107), -INT8_C( 122), -INT8_C( 126), -INT8_C(  76), -INT8_C( 121), -INT8_C(  65), -INT8_C( 115), -INT8_C(  46) },
      UINT16_C(28165),
      {  INT8_C(  27), -INT8_C(  48), -INT8_C(  86),  INT8_C(  54), -INT8_C(  32),  INT8_C(  59),  INT8_C( 122), -INT8_C(  12),
        -INT8_C(   6),  INT8_C(  28),  INT8_C(  89),  INT8_C(  69),  INT8_C(  57), -INT8_C(  19), -INT8_C(  99), -INT8_C(  37) },
      {  INT8_C(  97), -INT8_C(  39),  INT8_C( 120),  INT8_C( 103), -INT8_C(  90),  INT8_C(  20),  INT8_C( 101),  INT8_C(  30),
         INT8_C(  57),  INT8_C(  57), -INT8_C(  13),  INT8_C(  10),  INT8_C(  41), -INT8_C(  13), -INT8_C(  53),  INT8_C(  21) },
      {  INT8_C( 124),      INT8_MIN,  INT8_C(  34), -INT8_C(  17), -INT8_C(  48), -INT8_C(   5),      INT8_MAX,  INT8_C( 111),
         INT8_C( 107),  INT8_C(  85),  INT8_C(  76),  INT8_C(  79), -INT8_C( 121), -INT8_C(  32),  INT8_C( 104), -INT8_C(  46) } },
    { { -INT8_C( 125),  INT8_C( 109),  INT8_C(  50),  INT8_C(  49),  INT8_C(  76),  INT8_C(  38),  INT8_C(  43),  INT8_C(  37),
        -INT8_C( 125), -INT8_C(  52),  INT8_C( 115),  INT8_C( 100),  INT8_C(  45),  INT8_C(  74), -INT8_C(   5), -INT8_C( 110) },
      UINT16_C(47574),
      { -INT8_C(  76),  INT8_C(  26), -INT8_C( 112), -INT8_C(  90), -INT8_C(  73), -INT8_C( 123),  INT8_C(  15), -INT8_C(  19),
        -INT8_C(  57), -INT8_C(  58), -INT8_C(  79),  INT8_C(   7), -INT8_C(  26),  INT8_C(  47), -INT8_C(  11),      INT8_MAX },
      { -INT8_C(  44),  INT8_C(  76), -INT8_C(   8),  INT8_C(  91),  INT8_C(  64),  INT8_C(  42), -INT8_C(  48),  INT8_C(  22),
        -INT8_C(  10),  INT8_C(   4),  INT8_C(  93), -INT8_C(  57), -INT8_C(  42), -INT8_C(  31),  INT8_C( 109),  INT8_C(  88) },
      { -INT8_C( 125),  INT8_C( 102), -INT8_C( 120),  INT8_C(  49), -INT8_C(   9),  INT8_C(  38), -INT8_C(  33),  INT8_C(   3),
        -INT8_C(  67), -INT8_C(  52),  INT8_C( 115), -INT8_C(  50), -INT8_C(  68),  INT8_C(  16), -INT8_C(   5), -INT8_C(  41) } },
    { {  INT8_C(  20),  INT8_C(  50),  INT8_C(   9),  INT8_C(  94), -INT8_C(  69),  INT8_C(  31),  INT8_C(  50),  INT8_C(  65),
        -INT8_C(  79),  INT8_C( 123), -INT8_C(  96),  INT8_C(  34), -INT8_C(  70), -INT8_C(  37), -INT8_C(   3),  INT8_C(  22) },
      UINT16_C(58436),
      {  INT8_C( 126), -INT8_C(  47), -INT8_C( 126), -INT8_C(  42),  INT8_C(  30),  INT8_C( 111), -INT8_C(  34),  INT8_C(  96),
        -INT8_C(  65),  INT8_C(  94), -INT8_C(  39), -INT8_C(  47), -INT8_C(  73),  INT8_C( 120),  INT8_C(  10),  INT8_C(  18) },
      {  INT8_C(  72),  INT8_C(  25),  INT8_C(  71),  INT8_C(  82),  INT8_C( 115), -INT8_C(  50),  INT8_C(  16),  INT8_C(  21),
        -INT8_C(  37), -INT8_C(  92),  INT8_C(  58),  INT8_C(  57),  INT8_C(  41), -INT8_C(  77), -INT8_C( 116),  INT8_C(  92) },
      {  INT8_C(  20),  INT8_C(  50), -INT8_C(  55),  INT8_C(  94), -INT8_C(  69),  INT8_C(  31), -INT8_C(  18),  INT8_C(  65),
        -INT8_C(  79),  INT8_C( 123),  INT8_C(  19),  INT8_C(  34), -INT8_C(  70),  INT8_C(  43), -INT8_C( 106),  INT8_C( 110) } },
    { {  INT8_C(  29), -INT8_C(  38), -INT8_C(  54), -INT8_C( 111), -INT8_C(  23), -INT8_C(  86), -INT8_C(   8),  INT8_C(  21),
         INT8_C(  98),  INT8_C(  54),  INT8_C( 102), -INT8_C(  84),  INT8_C(  14),  INT8_C(  23),  INT8_C(  88),  INT8_C(  42) },
      UINT16_C(45295),
      {  INT8_C(   4),  INT8_C(  61),  INT8_C(  20), -INT8_C(  59),  INT8_C(  94),  INT8_C(  39), -INT8_C(  42),  INT8_C(  68),
        -INT8_C( 116),  INT8_C(  24), -INT8_C(  43), -INT8_C(  18),  INT8_C(  50),  INT8_C(  74), -INT8_C(  67),  INT8_C( 123) },
      { -INT8_C(   6),  INT8_C(   9),  INT8_C(  16), -INT8_C(  80), -INT8_C(  95),  INT8_C( 115), -INT8_C(  62),  INT8_C(  63),
        -INT8_C(  62), -INT8_C( 105),  INT8_C( 107), -INT8_C( 107), -INT8_C(  91), -INT8_C(  56), -INT8_C(  62), -INT8_C(  15) },
      { -INT8_C(   2),  INT8_C(  70),  INT8_C(  36),  INT8_C( 117), -INT8_C(  23), -INT8_C( 102), -INT8_C( 104), -INT8_C( 125),
         INT8_C(  98),  INT8_C(  54),  INT8_C( 102), -INT8_C(  84), -INT8_C(  41),  INT8_C(  18),  INT8_C(  88),  INT8_C( 108) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi8(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_add_epi8(src, test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_add_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_x_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_maskz_add_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int8_t a[16];
    const int8_t b[16];
    const int8_t r[16];
  } test_vec[] = {
    { UINT16_C(19804),
      {  INT8_C(  27),  INT8_C(  33), -INT8_C(  11),  INT8_C(  13), -INT8_C(  52),  INT8_C(  94),  INT8_C(  66),  INT8_C(  86),
         INT8_C(  52),  INT8_C(  23),  INT8_C(  93), -INT8_C(  33),  INT8_C(  78), -INT8_C(  48), -INT8_C(  30),  INT8_C(  75) },
      { -INT8_C(  37), -INT8_C(  52), -INT8_C(  84),  INT8_C( 110), -INT8_C(  15), -INT8_C(  31), -INT8_C(  23), -INT8_C(  34),
        -INT8_C( 103), -INT8_C(  38),  INT8_C(  65),  INT8_C(  33), -INT8_C(  16), -INT8_C(  52),  INT8_C(  35),  INT8_C( 109) },
      {  INT8_C(   0),  INT8_C(   0), -INT8_C(  95),  INT8_C( 123), -INT8_C(  67),  INT8_C(   0),  INT8_C(  43),  INT8_C(   0),
        -INT8_C(  51),  INT8_C(   0), -INT8_C(  98),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   5),  INT8_C(   0) } },
    { UINT16_C(50023),
      { -INT8_C(  40),  INT8_C(  67), -INT8_C(  16),  INT8_C(  53),  INT8_C(  73), -INT8_C(  64), -INT8_C(  41),  INT8_C( 124),
         INT8_C(  97), -INT8_C( 126),  INT8_C(  77), -INT8_C(  78), -INT8_C(  92), -INT8_C( 116),  INT8_C(  16), -INT8_C( 126) },
      {  INT8_C(  76), -INT8_C( 111),      INT8_MIN, -INT8_C(  99), -INT8_C(  91),  INT8_C( 102),  INT8_C(  51), -INT8_C(  30),
         INT8_C(  45), -INT8_C(  47),  INT8_C(  25),  INT8_C(  16),  INT8_C( 104), -INT8_C(  95), -INT8_C( 111),  INT8_C(  93) },
      {  INT8_C(  36), -INT8_C(  44),  INT8_C( 112),  INT8_C(   0),  INT8_C(   0),  INT8_C(  38),  INT8_C(  10),  INT8_C(   0),
        -INT8_C( 114),  INT8_C(  83),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  95), -INT8_C(  33) } },
    { UINT16_C( 6986),
      {  INT8_C( 119), -INT8_C(  75), -INT8_C( 115),  INT8_C( 108), -INT8_C(  41), -INT8_C(  83),  INT8_C(  24),  INT8_C( 118),
         INT8_C( 117),  INT8_C(  98), -INT8_C(  80),  INT8_C( 105), -INT8_C(  62), -INT8_C( 104), -INT8_C(  75),  INT8_C(  22) },
      {      INT8_MAX,  INT8_C( 109), -INT8_C(  49),  INT8_C( 103), -INT8_C(  97), -INT8_C(  46), -INT8_C(  64),  INT8_C(  44),
        -INT8_C( 126), -INT8_C( 107), -INT8_C(  14),  INT8_C(   2), -INT8_C(  58),  INT8_C(  69), -INT8_C(  19), -INT8_C(  91) },
      {  INT8_C(   0),  INT8_C(  34),  INT8_C(   0), -INT8_C(  45),  INT8_C(   0),  INT8_C(   0), -INT8_C(  40),  INT8_C(   0),
        -INT8_C(   9), -INT8_C(   9),  INT8_C(   0),  INT8_C( 107), -INT8_C( 120),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(36448),
      { -INT8_C(   7),  INT8_C( 119),  INT8_C( 107),  INT8_C(  82),  INT8_C( 101),  INT8_C(  83), -INT8_C(  83),  INT8_C( 116),
         INT8_C(  14),  INT8_C(  72),  INT8_C( 114),  INT8_C(   3), -INT8_C(  28), -INT8_C(   7),  INT8_C( 124), -INT8_C(  77) },
      { -INT8_C(  10), -INT8_C( 106),  INT8_C(  24),  INT8_C( 124), -INT8_C(  82), -INT8_C(  56), -INT8_C(  50),  INT8_C( 105),
         INT8_C(  61), -INT8_C( 116),  INT8_C(  93),  INT8_C(   9),  INT8_C(  50),  INT8_C(  96),  INT8_C(  92),  INT8_C(   3) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  27),  INT8_C( 123),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  44), -INT8_C(  49),  INT8_C(  12),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  74) } },
    { UINT16_C(37475),
      {  INT8_C(  88),  INT8_C(   7), -INT8_C(  30),  INT8_C(  23),  INT8_C(  11), -INT8_C(  12),  INT8_C(  22), -INT8_C( 112),
         INT8_C(   7),  INT8_C( 103), -INT8_C(  62),  INT8_C(  79),  INT8_C(  13), -INT8_C(   6),  INT8_C(   3),  INT8_C( 110) },
      { -INT8_C( 109),  INT8_C(  83), -INT8_C( 125), -INT8_C(  26), -INT8_C(  64),  INT8_C( 118), -INT8_C( 108),  INT8_C( 120),
         INT8_C(  65),  INT8_C(  56),  INT8_C( 108), -INT8_C(  19), -INT8_C(  92), -INT8_C(  43),  INT8_C(  59), -INT8_C(  82) },
      { -INT8_C(  21),  INT8_C(  90),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 106), -INT8_C(  86),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  97),  INT8_C(   0),  INT8_C(   0), -INT8_C(  79),  INT8_C(   0),  INT8_C(   0),  INT8_C(  28) } },
    { UINT16_C(60868),
      { -INT8_C(  85),  INT8_C(  98), -INT8_C( 119),  INT8_C(  28), -INT8_C(  52),  INT8_C(  36), -INT8_C(  38),  INT8_C(  97),
        -INT8_C(  23),  INT8_C(  33),  INT8_C( 101),  INT8_C(  30), -INT8_C(  56),  INT8_C(  67), -INT8_C(  69), -INT8_C(  85) },
      { -INT8_C(  92),  INT8_C(  56),  INT8_C(   8),  INT8_C( 111),  INT8_C(  68), -INT8_C(  19), -INT8_C(  66),  INT8_C(  42),
         INT8_C(   1),  INT8_C(  32),  INT8_C( 112),  INT8_C(  10),  INT8_C(  11),  INT8_C( 114),  INT8_C( 126), -INT8_C( 127) },
      {  INT8_C(   0),  INT8_C(   0), -INT8_C( 111),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 104), -INT8_C( 117),
        -INT8_C(  22),  INT8_C(   0), -INT8_C(  43),  INT8_C(  40),  INT8_C(   0), -INT8_C(  75),  INT8_C(  57),  INT8_C(  44) } },
    { UINT16_C(25877),
      {  INT8_C(  40), -INT8_C(  65),  INT8_C(  46), -INT8_C(  42), -INT8_C(  23),  INT8_C( 101),  INT8_C(  37),  INT8_C(  87),
         INT8_C( 109), -INT8_C(  69),  INT8_C(  80), -INT8_C(  44), -INT8_C(  53),  INT8_C(  53),  INT8_C(  52),  INT8_C( 114) },
      { -INT8_C( 100),  INT8_C(   3), -INT8_C(  25), -INT8_C(  36), -INT8_C(  90), -INT8_C(  81), -INT8_C( 115), -INT8_C( 101),
         INT8_C(  18),  INT8_C( 125), -INT8_C(  12),  INT8_C( 115), -INT8_C(  22), -INT8_C(   9),  INT8_C( 117),  INT8_C(  27) },
      { -INT8_C(  60),  INT8_C(   0),  INT8_C(  21),  INT8_C(   0), -INT8_C( 113),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
             INT8_MAX,  INT8_C(   0),  INT8_C(  68),  INT8_C(   0),  INT8_C(   0),  INT8_C(  44), -INT8_C(  87),  INT8_C(   0) } },
    { UINT16_C( 8536),
      { -INT8_C(  44), -INT8_C(  95), -INT8_C(  28), -INT8_C(  36), -INT8_C( 103), -INT8_C(   7),  INT8_C(  72),  INT8_C(  25),
        -INT8_C( 117), -INT8_C( 114), -INT8_C(  95),  INT8_C(  17),  INT8_C(   7),  INT8_C(   1), -INT8_C(  44), -INT8_C(  70) },
      { -INT8_C(  68), -INT8_C(   8),  INT8_C(  12),  INT8_C(  65),  INT8_C( 102),  INT8_C(  62), -INT8_C(  99),  INT8_C( 118),
         INT8_C(   8), -INT8_C(  26),  INT8_C(  94),  INT8_C(  54),  INT8_C(   0), -INT8_C(  68),      INT8_MAX, -INT8_C( 126) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  29), -INT8_C(   1),  INT8_C(   0), -INT8_C(  27),  INT8_C(   0),
        -INT8_C( 109),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  67),  INT8_C(   0),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_add_epi8(test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_add_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_x_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_mask_add_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[8];
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { {  INT16_C( 17817), -INT16_C(  2243), -INT16_C( 28883), -INT16_C(  7874),  INT16_C(   338), -INT16_C( 12748), -INT16_C(  9412),  INT16_C(   228) },
      UINT8_C( 66),
      { -INT16_C( 13625), -INT16_C(   511), -INT16_C(  9176), -INT16_C( 32360), -INT16_C( 18674), -INT16_C( 29571), -INT16_C( 32208),  INT16_C(  4749) },
      { -INT16_C( 26006), -INT16_C( 12496),  INT16_C( 20326), -INT16_C(   234), -INT16_C( 21928), -INT16_C(  3464), -INT16_C( 18163), -INT16_C(  1715) },
      {  INT16_C( 17817), -INT16_C( 13007), -INT16_C( 28883), -INT16_C(  7874),  INT16_C(   338), -INT16_C( 12748),  INT16_C( 15165),  INT16_C(   228) } },
    { { -INT16_C( 31056), -INT16_C( 22914),  INT16_C( 26105), -INT16_C( 10738),  INT16_C(   397),  INT16_C( 25565), -INT16_C( 22215), -INT16_C( 22166) },
      UINT8_C( 95),
      { -INT16_C( 29933), -INT16_C(  7224),  INT16_C( 20985),  INT16_C(  6509), -INT16_C( 25426), -INT16_C( 14506), -INT16_C( 26474), -INT16_C( 16323) },
      { -INT16_C( 17906),  INT16_C(  4756), -INT16_C( 20982),  INT16_C(  9519), -INT16_C( 15356), -INT16_C( 11051),  INT16_C( 26863), -INT16_C( 25274) },
      {  INT16_C( 17697), -INT16_C(  2468),  INT16_C(     3),  INT16_C( 16028),  INT16_C( 24754),  INT16_C( 25565),  INT16_C(   389), -INT16_C( 22166) } },
    { { -INT16_C(  4880),  INT16_C(  8735),  INT16_C(  8130),  INT16_C( 26938), -INT16_C(  8020), -INT16_C( 31347),  INT16_C(  8055),  INT16_C(  9966) },
      UINT8_C(190),
      {  INT16_C(  1808),  INT16_C( 20344),  INT16_C(  2923), -INT16_C( 23307), -INT16_C( 27990), -INT16_C(  8377), -INT16_C( 32088), -INT16_C( 20615) },
      {  INT16_C( 25028), -INT16_C( 30164),  INT16_C( 13304),  INT16_C( 19564), -INT16_C( 24947),  INT16_C(  1894), -INT16_C(  6448), -INT16_C( 24485) },
      { -INT16_C(  4880), -INT16_C(  9820),  INT16_C( 16227), -INT16_C(  3743),  INT16_C( 12599), -INT16_C(  6483),  INT16_C(  8055),  INT16_C( 20436) } },
    { {  INT16_C( 13221), -INT16_C( 19991),  INT16_C( 21448),  INT16_C(  9585),  INT16_C( 18335),  INT16_C( 18834),  INT16_C( 31996), -INT16_C(  1809) },
      UINT8_C(131),
      {  INT16_C(  1798),  INT16_C( 26660), -INT16_C( 21916), -INT16_C( 14369),  INT16_C( 18610),  INT16_C( 19226), -INT16_C( 32140),  INT16_C(  6089) },
      { -INT16_C( 10658),  INT16_C( 26610),  INT16_C(  5435),  INT16_C( 22284),  INT16_C(  1294),  INT16_C( 25864),  INT16_C( 31360),  INT16_C( 29206) },
      { -INT16_C(  8860), -INT16_C( 12266),  INT16_C( 21448),  INT16_C(  9585),  INT16_C( 18335),  INT16_C( 18834),  INT16_C( 31996), -INT16_C( 30241) } },
    { {  INT16_C(  3819),  INT16_C(  7886), -INT16_C(  2308), -INT16_C( 23812), -INT16_C(    53),  INT16_C(  7661), -INT16_C(  6086), -INT16_C( 20132) },
      UINT8_C(126),
      {  INT16_C(  1025), -INT16_C(  1747),  INT16_C( 13483),  INT16_C( 15025),  INT16_C( 17328), -INT16_C( 14874),  INT16_C(  6565),  INT16_C( 14071) },
      { -INT16_C( 12931), -INT16_C( 25527), -INT16_C( 11161),  INT16_C( 18125), -INT16_C(  1194), -INT16_C( 29538), -INT16_C(  3219), -INT16_C( 14390) },
      {  INT16_C(  3819), -INT16_C( 27274),  INT16_C(  2322), -INT16_C( 32386),  INT16_C( 16134),  INT16_C( 21124),  INT16_C(  3346), -INT16_C( 20132) } },
    { {  INT16_C( 22971), -INT16_C( 32214), -INT16_C( 27903),  INT16_C( 14508),  INT16_C( 16600), -INT16_C( 22707), -INT16_C(  5461), -INT16_C( 11019) },
      UINT8_C( 35),
      {  INT16_C( 16262), -INT16_C( 24435),  INT16_C( 14751),  INT16_C( 27015), -INT16_C( 24225), -INT16_C(   541), -INT16_C(  9613),  INT16_C(  4216) },
      { -INT16_C(  1057),  INT16_C( 24640),  INT16_C( 27675),  INT16_C( 25767),  INT16_C(  8263),  INT16_C( 28018), -INT16_C( 30242),  INT16_C(  6202) },
      {  INT16_C( 15205),  INT16_C(   205), -INT16_C( 27903),  INT16_C( 14508),  INT16_C( 16600),  INT16_C( 27477), -INT16_C(  5461), -INT16_C( 11019) } },
    { {  INT16_C(  9473), -INT16_C( 12490),  INT16_C( 26373), -INT16_C( 31879), -INT16_C( 12795),  INT16_C( 27285),  INT16_C( 20597), -INT16_C(  2226) },
      UINT8_C( 62),
      { -INT16_C(  1230), -INT16_C( 11593),  INT16_C( 18697), -INT16_C(  8764), -INT16_C(  7130),  INT16_C( 25580), -INT16_C(  4090), -INT16_C( 25505) },
      {  INT16_C( 19431), -INT16_C( 14954), -INT16_C( 27621),  INT16_C( 15919), -INT16_C( 16076), -INT16_C( 32362), -INT16_C( 25322),  INT16_C( 25848) },
      {  INT16_C(  9473), -INT16_C( 26547), -INT16_C(  8924),  INT16_C(  7155), -INT16_C( 23206), -INT16_C(  6782),  INT16_C( 20597), -INT16_C(  2226) } },
    { { -INT16_C( 16798), -INT16_C( 16061),  INT16_C( 28632),  INT16_C( 12716),  INT16_C( 10145),  INT16_C( 23704),  INT16_C( 13844), -INT16_C(  1453) },
      UINT8_C( 10),
      {  INT16_C(  4456),  INT16_C(  1455), -INT16_C( 11882), -INT16_C( 21010), -INT16_C( 14517),  INT16_C(  9453),  INT16_C( 19422),  INT16_C( 22787) },
      {  INT16_C(  9815), -INT16_C(  4177), -INT16_C( 17752),  INT16_C(  8643), -INT16_C( 14132), -INT16_C( 28981),  INT16_C(  3734), -INT16_C(  2203) },
      { -INT16_C( 16798), -INT16_C(  2722),  INT16_C( 28632), -INT16_C( 12367),  INT16_C( 10145),  INT16_C( 23704),  INT16_C( 13844), -INT16_C(  1453) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi16(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_add_epi16(src, test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_add_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_x_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_maskz_add_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C(134),
      {  INT16_C( 12554), -INT16_C( 17057), -INT16_C( 28311), -INT16_C( 28428), -INT16_C( 15318), -INT16_C( 10100), -INT16_C( 18852), -INT16_C( 16229) },
      { -INT16_C( 31990),  INT16_C( 29908),  INT16_C( 15111), -INT16_C(  4862),  INT16_C( 23252), -INT16_C( 12060), -INT16_C( 25385), -INT16_C( 12982) },
      {  INT16_C(     0),  INT16_C( 12851), -INT16_C( 13200),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 29211) } },
    { UINT8_C(157),
      {  INT16_C(  1408),  INT16_C( 17385), -INT16_C( 25707), -INT16_C(  3037),  INT16_C(  2190), -INT16_C( 16188),  INT16_C(  9347), -INT16_C(  7492) },
      { -INT16_C( 31565),  INT16_C( 13950), -INT16_C( 23827),  INT16_C(  2179),  INT16_C(  5735),  INT16_C(  2100), -INT16_C( 26139), -INT16_C(  8740) },
      { -INT16_C( 30157),  INT16_C(     0),  INT16_C( 16002), -INT16_C(   858),  INT16_C(  7925),  INT16_C(     0),  INT16_C(     0), -INT16_C( 16232) } },
    { UINT8_C( 24),
      { -INT16_C( 18498), -INT16_C( 16851), -INT16_C(  8942),  INT16_C( 29587), -INT16_C( 30777), -INT16_C( 18973), -INT16_C( 17141),  INT16_C( 22434) },
      { -INT16_C(  7638),  INT16_C(   677), -INT16_C( 15883),  INT16_C( 16066),  INT16_C( 29995), -INT16_C( 27669),  INT16_C(  7858), -INT16_C( 16674) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 19883), -INT16_C(   782),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(157),
      { -INT16_C( 30517), -INT16_C(  8546),  INT16_C( 22395),  INT16_C( 25788), -INT16_C(  9860),  INT16_C( 13172),  INT16_C( 13481),  INT16_C(  7362) },
      {  INT16_C( 20929), -INT16_C(    87), -INT16_C( 17569), -INT16_C( 31496),  INT16_C( 15928), -INT16_C(  7297),  INT16_C( 24667),  INT16_C( 21390) },
      { -INT16_C(  9588),  INT16_C(     0),  INT16_C(  4826), -INT16_C(  5708),  INT16_C(  6068),  INT16_C(     0),  INT16_C(     0),  INT16_C( 28752) } },
    { UINT8_C(204),
      { -INT16_C(  7807), -INT16_C( 18593),  INT16_C(  5789),  INT16_C( 10188), -INT16_C(   320), -INT16_C( 16671), -INT16_C( 20439), -INT16_C( 18566) },
      { -INT16_C( 19073), -INT16_C( 12901),  INT16_C( 26791),  INT16_C( 16857), -INT16_C( 21045), -INT16_C(  9971),  INT16_C(  2843),  INT16_C( 25436) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 32580),  INT16_C( 27045),  INT16_C(     0),  INT16_C(     0), -INT16_C( 17596),  INT16_C(  6870) } },
    { UINT8_C(237),
      { -INT16_C( 16461), -INT16_C(  6930),  INT16_C( 16169), -INT16_C( 32323),  INT16_C(  2552),  INT16_C(  1679), -INT16_C( 26974),  INT16_C( 14036) },
      {  INT16_C(  2376), -INT16_C( 18722),  INT16_C(  3134),  INT16_C( 10788),  INT16_C(  9045),  INT16_C( 14703),  INT16_C( 12911),  INT16_C( 27034) },
      { -INT16_C( 14085),  INT16_C(     0),  INT16_C( 19303), -INT16_C( 21535),  INT16_C(     0),  INT16_C( 16382), -INT16_C( 14063), -INT16_C( 24466) } },
    { UINT8_C( 42),
      {  INT16_C( 26325),  INT16_C(  6988),  INT16_C( 23545), -INT16_C( 22647),  INT16_C( 31625), -INT16_C(  3414),  INT16_C( 10045), -INT16_C( 17949) },
      {  INT16_C(  1997), -INT16_C( 13645), -INT16_C(  8378),  INT16_C( 19517),  INT16_C( 12495), -INT16_C( 25023),  INT16_C( 24234), -INT16_C( 25731) },
      {  INT16_C(     0), -INT16_C(  6657),  INT16_C(     0), -INT16_C(  3130),  INT16_C(     0), -INT16_C( 28437),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(211),
      {  INT16_C( 12889),  INT16_C( 28312),  INT16_C( 22010),  INT16_C(  4663), -INT16_C( 15800), -INT16_C( 32484), -INT16_C( 11389),  INT16_C( 13519) },
      {  INT16_C( 15116), -INT16_C( 28247),  INT16_C(   679),  INT16_C( 25752), -INT16_C( 15591),  INT16_C( 22603), -INT16_C( 32111), -INT16_C( 14859) },
      {  INT16_C( 28005),  INT16_C(    65),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31391),  INT16_C(     0),  INT16_C( 22036), -INT16_C(  1340) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_add_epi16(test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_add_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_x_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_mask_add_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(  1958227227),  INT32_C(  1842333829), -INT32_C(  2080453323), -INT32_C(   249177196) },
      UINT8_C(155),
      {  INT32_C(   553106004), -INT32_C(  1372587746), -INT32_C(  1712075801),  INT32_C(   993452127) },
      {  INT32_C(   777515607), -INT32_C(   712684129),  INT32_C(   223497663), -INT32_C(   489447286) },
      {  INT32_C(  1330621611), -INT32_C(  2085271875), -INT32_C(  2080453323),  INT32_C(   504004841) } },
    { {  INT32_C(   697251714), -INT32_C(   951749739),  INT32_C(   836834350), -INT32_C(   346279314) },
      UINT8_C(111),
      { -INT32_C(   885142141), -INT32_C(   194598295), -INT32_C(  1261452693),  INT32_C(   375114831) },
      {  INT32_C(  1479180836), -INT32_C(  1604991616), -INT32_C(  1780092987),  INT32_C(   634429248) },
      {  INT32_C(   594038695), -INT32_C(  1799589911),  INT32_C(  1253421616),  INT32_C(  1009544079) } },
    { {  INT32_C(  1249061656),  INT32_C(   406248213),  INT32_C(  1546568796),  INT32_C(   345790387) },
      UINT8_C(173),
      {  INT32_C(  1116067984), -INT32_C(   306617666),  INT32_C(  1471337118),  INT32_C(  1742516687) },
      {  INT32_C(  1421198449),  INT32_C(  1861843318), -INT32_C(  2139880994),  INT32_C(   418138440) },
      { -INT32_C(  1757700863),  INT32_C(   406248213), -INT32_C(   668543876), -INT32_C(  2134312169) } },
    { { -INT32_C(   455875920), -INT32_C(  1805739296), -INT32_C(   834127167), -INT32_C(   193385963) },
      UINT8_C(243),
      {  INT32_C(   397937177),  INT32_C(   447724867), -INT32_C(  1604479719), -INT32_C(  1556316088) },
      {  INT32_C(   764118341),  INT32_C(    33168795), -INT32_C(   931609255),  INT32_C(  2093992876) },
      {  INT32_C(  1162055518),  INT32_C(   480893662), -INT32_C(   834127167), -INT32_C(   193385963) } },
    { {  INT32_C(   227843937), -INT32_C(  1816959923),  INT32_C(   110120824), -INT32_C(  1826017770) },
      UINT8_C(242),
      {  INT32_C(   598721326), -INT32_C(  1962044123), -INT32_C(  1919813583), -INT32_C(  1281349718) },
      { -INT32_C(  1464369420),  INT32_C(  1889351967), -INT32_C(   174840084), -INT32_C(  1849263339) },
      {  INT32_C(   227843937), -INT32_C(    72692156),  INT32_C(   110120824), -INT32_C(  1826017770) } },
    { {  INT32_C(   585486166), -INT32_C(  1881648464), -INT32_C(  1741597697),  INT32_C(  1501172127) },
      UINT8_C(177),
      { -INT32_C(  2072152845),  INT32_C(  1678312837),  INT32_C(   175231240),  INT32_C(   639313595) },
      {  INT32_C(  1844718395),  INT32_C(  1747844119), -INT32_C(  1642309052), -INT32_C(  1463847021) },
      { -INT32_C(   227434450), -INT32_C(  1881648464), -INT32_C(  1741597697),  INT32_C(  1501172127) } },
    { {  INT32_C(  1282734968), -INT32_C(  1805890056), -INT32_C(   170454139),  INT32_C(   939566096) },
      UINT8_C(  1),
      {  INT32_C(   373441333),  INT32_C(  1967739279),  INT32_C(   363886263), -INT32_C(  1478106109) },
      { -INT32_C(  1988739640),  INT32_C(   299055662),  INT32_C(   830616967),  INT32_C(   503576578) },
      { -INT32_C(  1615298307), -INT32_C(  1805890056), -INT32_C(   170454139),  INT32_C(   939566096) } },
    { {  INT32_C(  1100203671), -INT32_C(   234656697), -INT32_C(  2035991414),  INT32_C(  1938166869) },
      UINT8_C( 44),
      { -INT32_C(    77918946), -INT32_C(   927432354), -INT32_C(  2008458249),  INT32_C(  1379220591) },
      { -INT32_C(   261431271), -INT32_C(  1794574077),  INT32_C(  1874265007),  INT32_C(   695196668) },
      {  INT32_C(  1100203671), -INT32_C(   234656697), -INT32_C(   134193242),  INT32_C(  2074417259) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_add_epi32(src, test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_add_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_maskz_add_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C( 26),
      {  INT32_C(  1100397714),  INT32_C(  1720147146),  INT32_C(   782659498),  INT32_C(   164614002) },
      {  INT32_C(  1362429759), -INT32_C(   135924898),  INT32_C(  1277782591),  INT32_C(  1455600660) },
      {  INT32_C(           0),  INT32_C(  1584222248),  INT32_C(           0),  INT32_C(  1620214662) } },
    { UINT8_C(104),
      { -INT32_C(   511491329),  INT32_C(  2145361873), -INT32_C(   681927889), -INT32_C(  1760116045) },
      { -INT32_C(  1818488330), -INT32_C(  1227468567), -INT32_C(   914908373), -INT32_C(   761443622) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1773407629) } },
    { UINT8_C( 38),
      { -INT32_C(  1756543294), -INT32_C(  1924529315),  INT32_C(   753287628),  INT32_C(   821606796) },
      {  INT32_C(  1995981071),  INT32_C(  1941095443), -INT32_C(   997134313),  INT32_C(  1414962996) },
      {  INT32_C(           0),  INT32_C(    16566128), -INT32_C(   243846685),  INT32_C(           0) } },
    { UINT8_C( 11),
      {  INT32_C(  1936642854),  INT32_C(  1476527496), -INT32_C(  1916837668),  INT32_C(  1565927957) },
      {  INT32_C(   819891483),  INT32_C(    24098982),  INT32_C(  1427042923),  INT32_C(   967231402) },
      { -INT32_C(  1538432959),  INT32_C(  1500626478),  INT32_C(           0), -INT32_C(  1761807937) } },
    { UINT8_C(136),
      {  INT32_C(  2025931821), -INT32_C(  1602308853), -INT32_C(  1584066603),  INT32_C(  2144498786) },
      { -INT32_C(    32030623), -INT32_C(   522392968),  INT32_C(  2136840774),  INT32_C(   707776301) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1442692209) } },
    { UINT8_C( 16),
      {  INT32_C(   615979771),  INT32_C(    91372444), -INT32_C(  1715689431),  INT32_C(  1732560282) },
      {  INT32_C(  1582515072), -INT32_C(   566478811), -INT32_C(  1915644371), -INT32_C(   936530095) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(152),
      { -INT32_C(  1815019322),  INT32_C(  2115824139), -INT32_C(   541094950), -INT32_C(   308634405) },
      { -INT32_C(   923734690),  INT32_C(   926178071),  INT32_C(  1920079652), -INT32_C(  1332173880) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1640808285) } },
    { UINT8_C( 27),
      {  INT32_C(  1575440202),  INT32_C(  1941889206),  INT32_C(    41623433), -INT32_C(    28396641) },
      {  INT32_C(     7451246), -INT32_C(  1374405146),  INT32_C(  1597383244), -INT32_C(  1478059980) },
      {  INT32_C(  1582891448),  INT32_C(   567484060),  INT32_C(           0), -INT32_C(  1506456621) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_add_epi32(test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_add_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_mask_add_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[2];
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 8201918233897001447),  INT64_C(  363962166077692400) },
      UINT8_C(116),
      { -INT64_C( 2929372668825588728), -INT64_C( 3656510824753645306) },
      {  INT64_C( 2121463196383157106), -INT64_C( 2712448646012699723) },
      {  INT64_C( 8201918233897001447),  INT64_C(  363962166077692400) } },
    { { -INT64_C( 7660056023431247569), -INT64_C( 8800573589908441802) },
      UINT8_C( 26),
      { -INT64_C( 5155283151088348406), -INT64_C( 6777794566593036429) },
      {  INT64_C( 8217194832092526134), -INT64_C( 6988787196078811595) },
      { -INT64_C( 7660056023431247569),  INT64_C( 4680162311037703592) } },
    { { -INT64_C( 5177610656632927949), -INT64_C( 7835688890176823491) },
      UINT8_C( 38),
      {  INT64_C( 8694329990162064366), -INT64_C( 7553465337537219517) },
      {  INT64_C( 5861342603890327684),  INT64_C( 8969888432361565647) },
      { -INT64_C( 5177610656632927949),  INT64_C( 1416423094824346130) } },
    { {  INT64_C( 1563207507706856527), -INT64_C(  371485882345723171) },
      UINT8_C(246),
      { -INT64_C( 8155582183645986764), -INT64_C( 8042754456252808652) },
      { -INT64_C(  996858152082936078),  INT64_C(  999238294551887019) },
      {  INT64_C( 1563207507706856527), -INT64_C( 7043516161700921633) } },
    { {  INT64_C( 2269651972621910057),  INT64_C( 8122205111827084555) },
      UINT8_C(109),
      {  INT64_C( 5898016879431101179),  INT64_C( 8196109586946188276) },
      { -INT64_C( 3927996688380496977), -INT64_C( 8700540345223695011) },
      {  INT64_C( 1970020191050604202),  INT64_C( 8122205111827084555) } },
    { {  INT64_C( 8953512102049709771),  INT64_C( 4073568780934150804) },
      UINT8_C( 55),
      { -INT64_C( 8698567697690688449),  INT64_C( 2011128588034496860) },
      {  INT64_C( 4050759086289972052), -INT64_C( 4209687100771601707) },
      { -INT64_C( 4647808611400716397), -INT64_C( 2198558512737104847) } },
    { { -INT64_C( 2003310644913083277),  INT64_C( 3614518035412058723) },
      UINT8_C(247),
      {  INT64_C( 1202266463040515144),  INT64_C( 2203879785493297747) },
      { -INT64_C( 6903979043742968285),  INT64_C(  286701945599558971) },
      { -INT64_C( 5701712580702453141),  INT64_C( 2490581731092856718) } },
    { {  INT64_C( 9179970759417586743), -INT64_C(  419587667919506800) },
      UINT8_C( 88),
      { -INT64_C( 5871315755711329534), -INT64_C( 2058360122490679194) },
      {  INT64_C( 1804735659384354964),  INT64_C( 3266572330366650128) },
      {  INT64_C( 9179970759417586743), -INT64_C(  419587667919506800) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi64(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_add_epi64(src, test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_add_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_x_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_maskz_add_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(150),
      { -INT64_C(  993872781162934343),  INT64_C( 9134756112190127784) },
      { -INT64_C( 2891573308549781280), -INT64_C( 9143568260559188007) },
      {  INT64_C(                   0), -INT64_C(    8812148369060223) } },
    { UINT8_C(196),
      {  INT64_C( 6550556420904536135),  INT64_C( 8490360445651406694) },
      { -INT64_C( 1739704569367854626),  INT64_C( 4245131661435093091) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 52),
      { -INT64_C( 2760535867334763843),  INT64_C( 7115030050339329677) },
      {  INT64_C( 7188684116250616331), -INT64_C( 2471133335336396754) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(  5),
      {  INT64_C( 3814106451400715087),  INT64_C( 6178810664702908734) },
      {  INT64_C( 1820139726703884269), -INT64_C( 2876452185216044984) },
      {  INT64_C( 5634246178104599356),  INT64_C(                   0) } },
    { UINT8_C( 77),
      { -INT64_C( 7349641843913850521),  INT64_C( 3200744105211371253) },
      {  INT64_C(  228621185812703474),  INT64_C( 6391631982896984822) },
      { -INT64_C( 7121020658101147047),  INT64_C(                   0) } },
    { UINT8_C(254),
      {  INT64_C( 5885620942751936373),  INT64_C( 3334511588433406542) },
      { -INT64_C(  254487071634799123),  INT64_C( 3127732574282601076) },
      {  INT64_C(                   0),  INT64_C( 6462244162716007618) } },
    { UINT8_C(174),
      { -INT64_C( 2725183470148505474),  INT64_C( 8524564968923083055) },
      {  INT64_C( 3417677596325229905), -INT64_C( 1448789787674024211) },
      {  INT64_C(                   0),  INT64_C( 7075775181249058844) } },
    { UINT8_C( 60),
      {  INT64_C( 5475088381666832271),  INT64_C( 2253690732183149705) },
      {  INT64_C( 9160715340915633356), -INT64_C( 3092667301170657521) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_add_epi64(test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_add_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_x_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm_mask_add_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    easysimd_float32 src[4];
    uint8_t k;
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -704.19), EASYSIMD_FLOAT32_C(   802.66), EASYSIMD_FLOAT32_C(    64.33), EASYSIMD_FLOAT32_C(  -242.74) },
      UINT8_C( 65),
      { EASYSIMD_FLOAT32_C(   830.74), EASYSIMD_FLOAT32_C(   231.76), EASYSIMD_FLOAT32_C(   133.32), EASYSIMD_FLOAT32_C(  -535.73) },
      { EASYSIMD_FLOAT32_C(    53.11), EASYSIMD_FLOAT32_C(   886.44), EASYSIMD_FLOAT32_C(   239.60), EASYSIMD_FLOAT32_C(  -566.57) },
      { EASYSIMD_FLOAT32_C(   883.85), EASYSIMD_FLOAT32_C(   802.66), EASYSIMD_FLOAT32_C(    64.33), EASYSIMD_FLOAT32_C(  -242.74) } },
    { { EASYSIMD_FLOAT32_C(  -434.78), EASYSIMD_FLOAT32_C(  -245.57), EASYSIMD_FLOAT32_C(  -426.13), EASYSIMD_FLOAT32_C(  -451.85) },
      UINT8_C(190),
      { EASYSIMD_FLOAT32_C(   701.44), EASYSIMD_FLOAT32_C(  -415.03), EASYSIMD_FLOAT32_C(    14.91), EASYSIMD_FLOAT32_C(   511.98) },
      { EASYSIMD_FLOAT32_C(    73.69), EASYSIMD_FLOAT32_C(   -83.35), EASYSIMD_FLOAT32_C(  -501.71), EASYSIMD_FLOAT32_C(  -653.66) },
      { EASYSIMD_FLOAT32_C(  -434.78), EASYSIMD_FLOAT32_C(  -498.38), EASYSIMD_FLOAT32_C(  -486.80), EASYSIMD_FLOAT32_C(  -141.68) } },
    { { EASYSIMD_FLOAT32_C(   552.83), EASYSIMD_FLOAT32_C(  -769.05), EASYSIMD_FLOAT32_C(  -799.36), EASYSIMD_FLOAT32_C(  -621.51) },
      UINT8_C(214),
      { EASYSIMD_FLOAT32_C(  -503.56), EASYSIMD_FLOAT32_C(  -818.84), EASYSIMD_FLOAT32_C(   666.01), EASYSIMD_FLOAT32_C(   253.70) },
      { EASYSIMD_FLOAT32_C(   864.37), EASYSIMD_FLOAT32_C(   496.75), EASYSIMD_FLOAT32_C(  -514.54), EASYSIMD_FLOAT32_C(    -2.31) },
      { EASYSIMD_FLOAT32_C(   552.83), EASYSIMD_FLOAT32_C(  -322.09), EASYSIMD_FLOAT32_C(   151.47), EASYSIMD_FLOAT32_C(  -621.51) } },
    { { EASYSIMD_FLOAT32_C(   961.02), EASYSIMD_FLOAT32_C(   538.57), EASYSIMD_FLOAT32_C(  -115.87), EASYSIMD_FLOAT32_C(   200.62) },
      UINT8_C(252),
      { EASYSIMD_FLOAT32_C(   449.35), EASYSIMD_FLOAT32_C(   955.05), EASYSIMD_FLOAT32_C(  -454.12), EASYSIMD_FLOAT32_C(   997.50) },
      { EASYSIMD_FLOAT32_C(   692.41), EASYSIMD_FLOAT32_C(  -752.69), EASYSIMD_FLOAT32_C(  -417.53), EASYSIMD_FLOAT32_C(  -292.69) },
      { EASYSIMD_FLOAT32_C(   961.02), EASYSIMD_FLOAT32_C(   538.57), EASYSIMD_FLOAT32_C(  -871.65), EASYSIMD_FLOAT32_C(   704.81) } },
    { { EASYSIMD_FLOAT32_C(   759.29), EASYSIMD_FLOAT32_C(   656.16), EASYSIMD_FLOAT32_C(   623.96), EASYSIMD_FLOAT32_C(  -742.41) },
      UINT8_C( 67),
      { EASYSIMD_FLOAT32_C(   176.79), EASYSIMD_FLOAT32_C(  -511.46), EASYSIMD_FLOAT32_C(  -796.86), EASYSIMD_FLOAT32_C(   555.28) },
      { EASYSIMD_FLOAT32_C(    90.21), EASYSIMD_FLOAT32_C(  -300.42), EASYSIMD_FLOAT32_C(   736.44), EASYSIMD_FLOAT32_C(  -243.78) },
      { EASYSIMD_FLOAT32_C(   267.00), EASYSIMD_FLOAT32_C(  -811.88), EASYSIMD_FLOAT32_C(   623.96), EASYSIMD_FLOAT32_C(  -742.41) } },
    { { EASYSIMD_FLOAT32_C(   953.28), EASYSIMD_FLOAT32_C(   600.81), EASYSIMD_FLOAT32_C(  -747.03), EASYSIMD_FLOAT32_C(  -561.26) },
      UINT8_C( 29),
      { EASYSIMD_FLOAT32_C(  -786.01), EASYSIMD_FLOAT32_C(   977.31), EASYSIMD_FLOAT32_C(   482.63), EASYSIMD_FLOAT32_C(   414.61) },
      { EASYSIMD_FLOAT32_C(   949.32), EASYSIMD_FLOAT32_C(   -68.02), EASYSIMD_FLOAT32_C(   369.66), EASYSIMD_FLOAT32_C(  -504.80) },
      { EASYSIMD_FLOAT32_C(   163.31), EASYSIMD_FLOAT32_C(   600.81), EASYSIMD_FLOAT32_C(   852.29), EASYSIMD_FLOAT32_C(   -90.19) } },
    { { EASYSIMD_FLOAT32_C(   -70.52), EASYSIMD_FLOAT32_C(    62.07), EASYSIMD_FLOAT32_C(  -257.49), EASYSIMD_FLOAT32_C(   511.95) },
      UINT8_C(229),
      { EASYSIMD_FLOAT32_C(  -498.19), EASYSIMD_FLOAT32_C(   168.10), EASYSIMD_FLOAT32_C(   393.34), EASYSIMD_FLOAT32_C(  -240.61) },
      { EASYSIMD_FLOAT32_C(   170.60), EASYSIMD_FLOAT32_C(  -429.87), EASYSIMD_FLOAT32_C(   247.93), EASYSIMD_FLOAT32_C(   373.74) },
      { EASYSIMD_FLOAT32_C(  -327.59), EASYSIMD_FLOAT32_C(    62.07), EASYSIMD_FLOAT32_C(   641.27), EASYSIMD_FLOAT32_C(   511.95) } },
    { { EASYSIMD_FLOAT32_C(  -874.59), EASYSIMD_FLOAT32_C(  -661.85), EASYSIMD_FLOAT32_C(  -926.68), EASYSIMD_FLOAT32_C(   861.85) },
      UINT8_C(127),
      { EASYSIMD_FLOAT32_C(  -973.41), EASYSIMD_FLOAT32_C(   462.66), EASYSIMD_FLOAT32_C(   347.34), EASYSIMD_FLOAT32_C(  -534.67) },
      { EASYSIMD_FLOAT32_C(  -938.83), EASYSIMD_FLOAT32_C(   561.33), EASYSIMD_FLOAT32_C(  -557.36), EASYSIMD_FLOAT32_C(   543.79) },
      { EASYSIMD_FLOAT32_C( -1912.24), EASYSIMD_FLOAT32_C(  1023.99), EASYSIMD_FLOAT32_C(  -210.02), EASYSIMD_FLOAT32_C(     9.12) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_add_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_add_ps");
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
    easysimd__m128 r = easysimd_mm_mask_add_ps(src, k, a, b);

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
test_easysimd_mm_maskz_add_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    uint8_t k;
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(173),
      { EASYSIMD_FLOAT32_C(  -774.46), EASYSIMD_FLOAT32_C(  -838.55), EASYSIMD_FLOAT32_C(   506.16), EASYSIMD_FLOAT32_C(  -453.30) },
      { EASYSIMD_FLOAT32_C(   -77.24), EASYSIMD_FLOAT32_C(    80.42), EASYSIMD_FLOAT32_C(   602.28), EASYSIMD_FLOAT32_C(    77.40) },
      { EASYSIMD_FLOAT32_C(  -851.70), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  1108.44), EASYSIMD_FLOAT32_C(  -375.90) } },
    { UINT8_C(186),
      { EASYSIMD_FLOAT32_C(   232.09), EASYSIMD_FLOAT32_C(   536.66), EASYSIMD_FLOAT32_C(    70.39), EASYSIMD_FLOAT32_C(  -578.10) },
      { EASYSIMD_FLOAT32_C(   487.36), EASYSIMD_FLOAT32_C(   329.26), EASYSIMD_FLOAT32_C(  -266.13), EASYSIMD_FLOAT32_C(   216.56) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   865.92), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -361.54) } },
    { UINT8_C(220),
      { EASYSIMD_FLOAT32_C(     3.54), EASYSIMD_FLOAT32_C(   868.08), EASYSIMD_FLOAT32_C(  -329.84), EASYSIMD_FLOAT32_C(   999.14) },
      { EASYSIMD_FLOAT32_C(    93.19), EASYSIMD_FLOAT32_C(  -235.10), EASYSIMD_FLOAT32_C(  -808.06), EASYSIMD_FLOAT32_C(   470.82) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( -1137.90), EASYSIMD_FLOAT32_C(  1469.96) } },
    { UINT8_C( 94),
      { EASYSIMD_FLOAT32_C(  -342.75), EASYSIMD_FLOAT32_C(  -520.37), EASYSIMD_FLOAT32_C(   192.08), EASYSIMD_FLOAT32_C(   890.95) },
      { EASYSIMD_FLOAT32_C(  -294.83), EASYSIMD_FLOAT32_C(   353.53), EASYSIMD_FLOAT32_C(   397.11), EASYSIMD_FLOAT32_C(   251.87) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -166.84), EASYSIMD_FLOAT32_C(   589.19), EASYSIMD_FLOAT32_C(  1142.82) } },
    { UINT8_C(168),
      { EASYSIMD_FLOAT32_C(  -522.47), EASYSIMD_FLOAT32_C(  -145.85), EASYSIMD_FLOAT32_C(   353.70), EASYSIMD_FLOAT32_C(   -33.49) },
      { EASYSIMD_FLOAT32_C(  -913.76), EASYSIMD_FLOAT32_C(  -109.64), EASYSIMD_FLOAT32_C(  -963.10), EASYSIMD_FLOAT32_C(  -491.86) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -525.35) } },
    { UINT8_C(194),
      { EASYSIMD_FLOAT32_C(   366.16), EASYSIMD_FLOAT32_C(   242.01), EASYSIMD_FLOAT32_C(   594.27), EASYSIMD_FLOAT32_C(   627.24) },
      { EASYSIMD_FLOAT32_C(  -754.44), EASYSIMD_FLOAT32_C(   462.35), EASYSIMD_FLOAT32_C(  -702.60), EASYSIMD_FLOAT32_C(  -755.30) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   704.36), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 77),
      { EASYSIMD_FLOAT32_C(    62.30), EASYSIMD_FLOAT32_C(  -563.36), EASYSIMD_FLOAT32_C(  -973.64), EASYSIMD_FLOAT32_C(  -322.85) },
      { EASYSIMD_FLOAT32_C(    93.89), EASYSIMD_FLOAT32_C(  -494.01), EASYSIMD_FLOAT32_C(   869.23), EASYSIMD_FLOAT32_C(   -15.16) },
      { EASYSIMD_FLOAT32_C(   156.19), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -104.41), EASYSIMD_FLOAT32_C(  -338.01) } },
    { UINT8_C(138),
      { EASYSIMD_FLOAT32_C(   222.76), EASYSIMD_FLOAT32_C(  -618.05), EASYSIMD_FLOAT32_C(  -536.97), EASYSIMD_FLOAT32_C(   499.06) },
      { EASYSIMD_FLOAT32_C(  -140.51), EASYSIMD_FLOAT32_C(   317.18), EASYSIMD_FLOAT32_C(  -147.24), EASYSIMD_FLOAT32_C(   826.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -300.87), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  1325.06) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_add_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_add_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_add_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_add_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    easysimd_float64 src[2];
    uint8_t k;
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   590.73), EASYSIMD_FLOAT64_C(  -652.57) },
      UINT8_C(153),
      { EASYSIMD_FLOAT64_C(  -380.05), EASYSIMD_FLOAT64_C(  -971.75) },
      { EASYSIMD_FLOAT64_C(   810.79), EASYSIMD_FLOAT64_C(  -709.20) },
      { EASYSIMD_FLOAT64_C(   430.74), EASYSIMD_FLOAT64_C(  -652.57) } },
    { { EASYSIMD_FLOAT64_C(  -626.01), EASYSIMD_FLOAT64_C(    83.35) },
      UINT8_C(171),
      { EASYSIMD_FLOAT64_C(   839.74), EASYSIMD_FLOAT64_C(  -362.01) },
      { EASYSIMD_FLOAT64_C(    21.00), EASYSIMD_FLOAT64_C(   295.85) },
      { EASYSIMD_FLOAT64_C(   860.74), EASYSIMD_FLOAT64_C(   -66.16) } },
    { { EASYSIMD_FLOAT64_C(  -950.84), EASYSIMD_FLOAT64_C(    54.90) },
      UINT8_C( 68),
      { EASYSIMD_FLOAT64_C(  -940.46), EASYSIMD_FLOAT64_C(   556.14) },
      { EASYSIMD_FLOAT64_C(   820.53), EASYSIMD_FLOAT64_C(   751.62) },
      { EASYSIMD_FLOAT64_C(  -950.84), EASYSIMD_FLOAT64_C(    54.90) } },
    { { EASYSIMD_FLOAT64_C(  -869.08), EASYSIMD_FLOAT64_C(   171.39) },
      UINT8_C(105),
      { EASYSIMD_FLOAT64_C(   890.01), EASYSIMD_FLOAT64_C(  -912.04) },
      { EASYSIMD_FLOAT64_C(  -583.69), EASYSIMD_FLOAT64_C(  -920.37) },
      { EASYSIMD_FLOAT64_C(   306.32), EASYSIMD_FLOAT64_C(   171.39) } },
    { { EASYSIMD_FLOAT64_C(  -206.84), EASYSIMD_FLOAT64_C(   726.26) },
      UINT8_C( 83),
      { EASYSIMD_FLOAT64_C(  -616.11), EASYSIMD_FLOAT64_C(  -926.31) },
      { EASYSIMD_FLOAT64_C(  -561.73), EASYSIMD_FLOAT64_C(     3.84) },
      { EASYSIMD_FLOAT64_C( -1177.84), EASYSIMD_FLOAT64_C(  -922.47) } },
    { { EASYSIMD_FLOAT64_C(  -898.06), EASYSIMD_FLOAT64_C(  -750.94) },
      UINT8_C( 76),
      { EASYSIMD_FLOAT64_C(  -524.07), EASYSIMD_FLOAT64_C(   332.41) },
      { EASYSIMD_FLOAT64_C(   187.74), EASYSIMD_FLOAT64_C(  -684.33) },
      { EASYSIMD_FLOAT64_C(  -898.06), EASYSIMD_FLOAT64_C(  -750.94) } },
    { { EASYSIMD_FLOAT64_C(   970.41), EASYSIMD_FLOAT64_C(  -791.26) },
      UINT8_C( 98),
      { EASYSIMD_FLOAT64_C(  -980.43), EASYSIMD_FLOAT64_C(   263.64) },
      { EASYSIMD_FLOAT64_C(  -442.82), EASYSIMD_FLOAT64_C(  -920.89) },
      { EASYSIMD_FLOAT64_C(   970.41), EASYSIMD_FLOAT64_C(  -657.25) } },
    { { EASYSIMD_FLOAT64_C(  -180.22), EASYSIMD_FLOAT64_C(  -622.30) },
      UINT8_C( 97),
      { EASYSIMD_FLOAT64_C(   -49.30), EASYSIMD_FLOAT64_C(   549.09) },
      { EASYSIMD_FLOAT64_C(   796.13), EASYSIMD_FLOAT64_C(  -159.29) },
      { EASYSIMD_FLOAT64_C(   746.83), EASYSIMD_FLOAT64_C(  -622.30) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_add_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_add_pd");
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
    easysimd__m128d r = easysimd_mm_mask_add_pd(src, k, a, b);

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
test_easysimd_mm_maskz_add_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    uint8_t k;
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C( 71),
      { EASYSIMD_FLOAT64_C(   740.42), EASYSIMD_FLOAT64_C(  -750.20) },
      { EASYSIMD_FLOAT64_C(   135.19), EASYSIMD_FLOAT64_C(   863.84) },
      { EASYSIMD_FLOAT64_C(   875.61), EASYSIMD_FLOAT64_C(   113.64) } },
    { UINT8_C(125),
      { EASYSIMD_FLOAT64_C(  -687.43), EASYSIMD_FLOAT64_C(  -593.79) },
      { EASYSIMD_FLOAT64_C(  -737.40), EASYSIMD_FLOAT64_C(   381.92) },
      { EASYSIMD_FLOAT64_C( -1424.83), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(162),
      { EASYSIMD_FLOAT64_C(   362.57), EASYSIMD_FLOAT64_C(   579.86) },
      { EASYSIMD_FLOAT64_C(  -718.17), EASYSIMD_FLOAT64_C(  -379.85) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   200.01) } },
    { UINT8_C(  5),
      { EASYSIMD_FLOAT64_C(   524.73), EASYSIMD_FLOAT64_C(  -910.37) },
      { EASYSIMD_FLOAT64_C(   822.13), EASYSIMD_FLOAT64_C(   272.28) },
      { EASYSIMD_FLOAT64_C(  1346.86), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(202),
      { EASYSIMD_FLOAT64_C(  -593.74), EASYSIMD_FLOAT64_C(  -770.13) },
      { EASYSIMD_FLOAT64_C(   815.45), EASYSIMD_FLOAT64_C(   680.10) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   -90.03) } },
    { UINT8_C(105),
      { EASYSIMD_FLOAT64_C(  -655.08), EASYSIMD_FLOAT64_C(   287.76) },
      { EASYSIMD_FLOAT64_C(   615.35), EASYSIMD_FLOAT64_C(   450.47) },
      { EASYSIMD_FLOAT64_C(   -39.73), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(181),
      { EASYSIMD_FLOAT64_C(   995.53), EASYSIMD_FLOAT64_C(   190.89) },
      { EASYSIMD_FLOAT64_C(  -140.64), EASYSIMD_FLOAT64_C(   130.72) },
      { EASYSIMD_FLOAT64_C(   854.89), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 30),
      { EASYSIMD_FLOAT64_C(   778.36), EASYSIMD_FLOAT64_C(   443.29) },
      { EASYSIMD_FLOAT64_C(   460.94), EASYSIMD_FLOAT64_C(  -959.04) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -515.75) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_add_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_add_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_add_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_add_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -362.02), EASYSIMD_FLOAT32_C(  -753.03), EASYSIMD_FLOAT32_C(   336.60), EASYSIMD_FLOAT32_C(   433.83) },
      UINT8_C(125),
      { EASYSIMD_FLOAT32_C(   392.81), EASYSIMD_FLOAT32_C(  -464.44), EASYSIMD_FLOAT32_C(  -422.67), EASYSIMD_FLOAT32_C(  -899.86) },
      { EASYSIMD_FLOAT32_C(  -403.90), EASYSIMD_FLOAT32_C(   536.42), EASYSIMD_FLOAT32_C(  -805.91), EASYSIMD_FLOAT32_C(   663.43) },
      { EASYSIMD_FLOAT32_C(   -11.09), EASYSIMD_FLOAT32_C(  -464.44), EASYSIMD_FLOAT32_C(  -422.67), EASYSIMD_FLOAT32_C(  -899.86) } },
    { { EASYSIMD_FLOAT32_C(  -478.45), EASYSIMD_FLOAT32_C(  -735.39), EASYSIMD_FLOAT32_C(  -353.05), EASYSIMD_FLOAT32_C(   917.77) },
      UINT8_C( 83),
      { EASYSIMD_FLOAT32_C(  -443.57), EASYSIMD_FLOAT32_C(   820.62), EASYSIMD_FLOAT32_C(   677.29), EASYSIMD_FLOAT32_C(    81.36) },
      { EASYSIMD_FLOAT32_C(   246.07), EASYSIMD_FLOAT32_C(   669.97), EASYSIMD_FLOAT32_C(   862.97), EASYSIMD_FLOAT32_C(   545.49) },
      { EASYSIMD_FLOAT32_C(  -197.50), EASYSIMD_FLOAT32_C(   820.62), EASYSIMD_FLOAT32_C(   677.29), EASYSIMD_FLOAT32_C(    81.36) } },
    { { EASYSIMD_FLOAT32_C(   289.66), EASYSIMD_FLOAT32_C(   153.93), EASYSIMD_FLOAT32_C(  -971.51), EASYSIMD_FLOAT32_C(   876.28) },
      UINT8_C(104),
      { EASYSIMD_FLOAT32_C(  -333.53), EASYSIMD_FLOAT32_C(  -876.75), EASYSIMD_FLOAT32_C(  -699.81), EASYSIMD_FLOAT32_C(  -899.70) },
      { EASYSIMD_FLOAT32_C(  -343.74), EASYSIMD_FLOAT32_C(   692.99), EASYSIMD_FLOAT32_C(  -364.15), EASYSIMD_FLOAT32_C(   233.59) },
      { EASYSIMD_FLOAT32_C(   289.66), EASYSIMD_FLOAT32_C(  -876.75), EASYSIMD_FLOAT32_C(  -699.81), EASYSIMD_FLOAT32_C(  -899.70) } },
    { { EASYSIMD_FLOAT32_C(   793.13), EASYSIMD_FLOAT32_C(   231.95), EASYSIMD_FLOAT32_C(  -229.99), EASYSIMD_FLOAT32_C(   987.23) },
      UINT8_C(242),
      { EASYSIMD_FLOAT32_C(   291.56), EASYSIMD_FLOAT32_C(  -748.16), EASYSIMD_FLOAT32_C(   542.34), EASYSIMD_FLOAT32_C(   209.32) },
      { EASYSIMD_FLOAT32_C(   326.59), EASYSIMD_FLOAT32_C(  -901.23), EASYSIMD_FLOAT32_C(    29.95), EASYSIMD_FLOAT32_C(     3.89) },
      { EASYSIMD_FLOAT32_C(   793.13), EASYSIMD_FLOAT32_C(  -748.16), EASYSIMD_FLOAT32_C(   542.34), EASYSIMD_FLOAT32_C(   209.32) } },
    { { EASYSIMD_FLOAT32_C(   180.14), EASYSIMD_FLOAT32_C(  -723.98), EASYSIMD_FLOAT32_C(  -326.15), EASYSIMD_FLOAT32_C(    43.10) },
      UINT8_C(222),
      { EASYSIMD_FLOAT32_C(   963.51), EASYSIMD_FLOAT32_C(  -802.96), EASYSIMD_FLOAT32_C(   850.00), EASYSIMD_FLOAT32_C(   839.79) },
      { EASYSIMD_FLOAT32_C(   160.62), EASYSIMD_FLOAT32_C(  -483.52), EASYSIMD_FLOAT32_C(   963.04), EASYSIMD_FLOAT32_C(   460.80) },
      { EASYSIMD_FLOAT32_C(   180.14), EASYSIMD_FLOAT32_C(  -802.96), EASYSIMD_FLOAT32_C(   850.00), EASYSIMD_FLOAT32_C(   839.79) } },
    { { EASYSIMD_FLOAT32_C(  -383.22), EASYSIMD_FLOAT32_C(  -380.70), EASYSIMD_FLOAT32_C(   153.80), EASYSIMD_FLOAT32_C(   252.63) },
      UINT8_C( 59),
      { EASYSIMD_FLOAT32_C(   -53.07), EASYSIMD_FLOAT32_C(  -515.42), EASYSIMD_FLOAT32_C(  -377.10), EASYSIMD_FLOAT32_C(   -65.84) },
      { EASYSIMD_FLOAT32_C(   379.97), EASYSIMD_FLOAT32_C(   914.45), EASYSIMD_FLOAT32_C(   186.00), EASYSIMD_FLOAT32_C(   -77.69) },
      { EASYSIMD_FLOAT32_C(   326.90), EASYSIMD_FLOAT32_C(  -515.42), EASYSIMD_FLOAT32_C(  -377.10), EASYSIMD_FLOAT32_C(   -65.84) } },
    { { EASYSIMD_FLOAT32_C(   123.78), EASYSIMD_FLOAT32_C(  -487.41), EASYSIMD_FLOAT32_C(    21.08), EASYSIMD_FLOAT32_C(  -846.28) },
      UINT8_C(218),
      { EASYSIMD_FLOAT32_C(  -798.78), EASYSIMD_FLOAT32_C(  -570.26), EASYSIMD_FLOAT32_C(  -809.67), EASYSIMD_FLOAT32_C(   244.32) },
      { EASYSIMD_FLOAT32_C(  -748.74), EASYSIMD_FLOAT32_C(  -846.16), EASYSIMD_FLOAT32_C(   441.35), EASYSIMD_FLOAT32_C(  -898.74) },
      { EASYSIMD_FLOAT32_C(   123.78), EASYSIMD_FLOAT32_C(  -570.26), EASYSIMD_FLOAT32_C(  -809.67), EASYSIMD_FLOAT32_C(   244.32) } },
    { { EASYSIMD_FLOAT32_C(   993.63), EASYSIMD_FLOAT32_C(  -398.03), EASYSIMD_FLOAT32_C(  -382.26), EASYSIMD_FLOAT32_C(   956.67) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT32_C(   234.51), EASYSIMD_FLOAT32_C(  -424.03), EASYSIMD_FLOAT32_C(   216.57), EASYSIMD_FLOAT32_C(  -512.86) },
      { EASYSIMD_FLOAT32_C(  -571.15), EASYSIMD_FLOAT32_C(  -836.50), EASYSIMD_FLOAT32_C(   -28.28), EASYSIMD_FLOAT32_C(    51.75) },
      { EASYSIMD_FLOAT32_C(  -336.64), EASYSIMD_FLOAT32_C(  -424.03), EASYSIMD_FLOAT32_C(   216.57), EASYSIMD_FLOAT32_C(  -512.86) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r = easysimd_mm_mask_add_ss(src, test_vec[i].k, a, b);
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_add_ss(src, k, a, b);

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
test_easysimd_mm_maskz_add_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(229),
      { EASYSIMD_FLOAT32_C(   695.67), EASYSIMD_FLOAT32_C(  -467.38), EASYSIMD_FLOAT32_C(   303.41), EASYSIMD_FLOAT32_C(   444.32) },
      { EASYSIMD_FLOAT32_C(  -971.51), EASYSIMD_FLOAT32_C(   843.53), EASYSIMD_FLOAT32_C(  -243.67), EASYSIMD_FLOAT32_C(   463.81) },
      { EASYSIMD_FLOAT32_C(  -275.84), EASYSIMD_FLOAT32_C(  -467.38), EASYSIMD_FLOAT32_C(   303.41), EASYSIMD_FLOAT32_C(   444.32) } },
    { UINT8_C(226),
      { EASYSIMD_FLOAT32_C(   197.10), EASYSIMD_FLOAT32_C(   132.05), EASYSIMD_FLOAT32_C(  -305.37), EASYSIMD_FLOAT32_C(  -575.56) },
      { EASYSIMD_FLOAT32_C(   496.81), EASYSIMD_FLOAT32_C(  -398.18), EASYSIMD_FLOAT32_C(   186.52), EASYSIMD_FLOAT32_C(    10.74) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   132.05), EASYSIMD_FLOAT32_C(  -305.37), EASYSIMD_FLOAT32_C(  -575.56) } },
    { UINT8_C(101),
      { EASYSIMD_FLOAT32_C(  -659.66), EASYSIMD_FLOAT32_C(  -842.89), EASYSIMD_FLOAT32_C(  -218.71), EASYSIMD_FLOAT32_C(   619.24) },
      { EASYSIMD_FLOAT32_C(  -897.13), EASYSIMD_FLOAT32_C(  -873.47), EASYSIMD_FLOAT32_C(   228.22), EASYSIMD_FLOAT32_C(     5.28) },
      { EASYSIMD_FLOAT32_C( -1556.79), EASYSIMD_FLOAT32_C(  -842.89), EASYSIMD_FLOAT32_C(  -218.71), EASYSIMD_FLOAT32_C(   619.24) } },
    { UINT8_C(252),
      { EASYSIMD_FLOAT32_C(    87.18), EASYSIMD_FLOAT32_C(   911.77), EASYSIMD_FLOAT32_C(  -825.67), EASYSIMD_FLOAT32_C(   690.54) },
      { EASYSIMD_FLOAT32_C(   607.43), EASYSIMD_FLOAT32_C(  -293.05), EASYSIMD_FLOAT32_C(    -6.04), EASYSIMD_FLOAT32_C(    51.75) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   911.77), EASYSIMD_FLOAT32_C(  -825.67), EASYSIMD_FLOAT32_C(   690.54) } },
    { UINT8_C( 58),
      { EASYSIMD_FLOAT32_C(  -162.51), EASYSIMD_FLOAT32_C(   808.09), EASYSIMD_FLOAT32_C(  -800.75), EASYSIMD_FLOAT32_C(   733.18) },
      { EASYSIMD_FLOAT32_C(     5.19), EASYSIMD_FLOAT32_C(   331.30), EASYSIMD_FLOAT32_C(  -572.20), EASYSIMD_FLOAT32_C(   429.63) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   808.09), EASYSIMD_FLOAT32_C(  -800.75), EASYSIMD_FLOAT32_C(   733.18) } },
    { UINT8_C(165),
      { EASYSIMD_FLOAT32_C(    29.62), EASYSIMD_FLOAT32_C(  -383.85), EASYSIMD_FLOAT32_C(   838.85), EASYSIMD_FLOAT32_C(   -85.88) },
      { EASYSIMD_FLOAT32_C(   -43.51), EASYSIMD_FLOAT32_C(   995.96), EASYSIMD_FLOAT32_C(   695.41), EASYSIMD_FLOAT32_C(  -424.27) },
      { EASYSIMD_FLOAT32_C(   -13.89), EASYSIMD_FLOAT32_C(  -383.85), EASYSIMD_FLOAT32_C(   838.85), EASYSIMD_FLOAT32_C(   -85.88) } },
    { UINT8_C(151),
      { EASYSIMD_FLOAT32_C(   821.95), EASYSIMD_FLOAT32_C(   803.96), EASYSIMD_FLOAT32_C(   104.12), EASYSIMD_FLOAT32_C(   482.38) },
      { EASYSIMD_FLOAT32_C(  -108.86), EASYSIMD_FLOAT32_C(    15.89), EASYSIMD_FLOAT32_C(   656.71), EASYSIMD_FLOAT32_C(  -418.32) },
      { EASYSIMD_FLOAT32_C(   713.09), EASYSIMD_FLOAT32_C(   803.96), EASYSIMD_FLOAT32_C(   104.12), EASYSIMD_FLOAT32_C(   482.38) } },
    { UINT8_C(197),
      { EASYSIMD_FLOAT32_C(  -636.34), EASYSIMD_FLOAT32_C(   575.64), EASYSIMD_FLOAT32_C(   675.07), EASYSIMD_FLOAT32_C(    99.10) },
      { EASYSIMD_FLOAT32_C(  -586.87), EASYSIMD_FLOAT32_C(   483.16), EASYSIMD_FLOAT32_C(   298.35), EASYSIMD_FLOAT32_C(  -853.69) },
      { EASYSIMD_FLOAT32_C( -1223.21), EASYSIMD_FLOAT32_C(   575.64), EASYSIMD_FLOAT32_C(   675.07), EASYSIMD_FLOAT32_C(    99.10) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r = easysimd_mm_maskz_add_ss(test_vec[i].k, a, b);
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_add_ss(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_add_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t src[32];
    const uint32_t k;
    const int8_t a[32];
    const int8_t b[32];
    const int8_t r[32];
  } test_vec[] = {
    { {  INT8_C( 108), -INT8_C(  65), -INT8_C(  32), -INT8_C(  68), -INT8_C( 113), -INT8_C(  31), -INT8_C( 114),  INT8_C(  18),
         INT8_C(  82),  INT8_C(  50),  INT8_C(  59), -INT8_C(  48),  INT8_C( 103), -INT8_C( 117),  INT8_C( 122),  INT8_C(  99),
         INT8_C( 101), -INT8_C(  47), -INT8_C(  15),  INT8_C( 100),  INT8_C(  15),  INT8_C(  22), -INT8_C(  45), -INT8_C(  99),
        -INT8_C(  74), -INT8_C( 124),  INT8_C(  52), -INT8_C(  20),  INT8_C(  64), -INT8_C(  86),  INT8_C(  78), -INT8_C(  84) },
      UINT32_C(4184354665),
      {  INT8_C(  16), -INT8_C(  10),  INT8_C(  11),  INT8_C(  98),  INT8_C(  40),  INT8_C(  70),  INT8_C(  51), -INT8_C( 113),
        -INT8_C(  47), -INT8_C(  83), -INT8_C(  14),  INT8_C(  55),  INT8_C( 126), -INT8_C(  29), -INT8_C( 101), -INT8_C( 114),
        -INT8_C(   7),  INT8_C( 111),  INT8_C(  43), -INT8_C(  80), -INT8_C(  13),  INT8_C(  95), -INT8_C( 100),  INT8_C(  51),
         INT8_C(   9), -INT8_C(  22), -INT8_C(  33),  INT8_C( 115),  INT8_C(  25),  INT8_C(  71),  INT8_C( 108),  INT8_C(  42) },
      {  INT8_C(  61),  INT8_C( 119), -INT8_C( 116),  INT8_C( 102), -INT8_C(  67), -INT8_C(  65), -INT8_C(  11), -INT8_C( 114),
         INT8_C( 108), -INT8_C(  24), -INT8_C(  59), -INT8_C(  21), -INT8_C(  53),  INT8_C(  97),  INT8_C( 121), -INT8_C(  59),
        -INT8_C(  48), -INT8_C(  92),  INT8_C( 117), -INT8_C(  61),  INT8_C(   3),  INT8_C(  17), -INT8_C(  10),  INT8_C(  12),
        -INT8_C(   5), -INT8_C(  43),      INT8_MAX,  INT8_C(  21),  INT8_C(  28), -INT8_C(  21),  INT8_C(  63),  INT8_C(  89) },
      {  INT8_C(  77), -INT8_C(  65), -INT8_C(  32), -INT8_C(  56), -INT8_C( 113),  INT8_C(   5),  INT8_C(  40),  INT8_C(  18),
         INT8_C(  61), -INT8_C( 107), -INT8_C(  73),  INT8_C(  34),  INT8_C( 103),  INT8_C(  68),  INT8_C( 122),  INT8_C(  99),
         INT8_C( 101), -INT8_C(  47), -INT8_C(  15),  INT8_C( 115),  INT8_C(  15),  INT8_C( 112), -INT8_C( 110), -INT8_C(  99),
         INT8_C(   4), -INT8_C( 124),  INT8_C(  52), -INT8_C( 120),  INT8_C(  53),  INT8_C(  50), -INT8_C(  85), -INT8_C( 125) } },
    { {  INT8_C(  98), -INT8_C(  53), -INT8_C(  65),  INT8_C(  31), -INT8_C( 117), -INT8_C(  75), -INT8_C(  82), -INT8_C(   9),
        -INT8_C(  99),  INT8_C( 115), -INT8_C(  30),  INT8_C( 104), -INT8_C(  44),  INT8_C(  91),  INT8_C(  45), -INT8_C(  92),
        -INT8_C(   1), -INT8_C(  94),  INT8_C( 103),  INT8_C(   2), -INT8_C(  77),  INT8_C(  93),  INT8_C(  15), -INT8_C(  81),
         INT8_C(  50), -INT8_C( 114), -INT8_C(  60),  INT8_C(  78),  INT8_C( 122),  INT8_C(   3), -INT8_C(  88), -INT8_C(  36) },
      UINT32_C(1509713870),
      {  INT8_C(  28), -INT8_C(  86),  INT8_C(  81), -INT8_C(  71),  INT8_C(  29),  INT8_C(  51),  INT8_C(  34), -INT8_C(  14),
        -INT8_C( 113),  INT8_C(  79), -INT8_C( 106), -INT8_C( 114), -INT8_C(  14), -INT8_C(   2), -INT8_C( 111), -INT8_C(  91),
         INT8_C(  91), -INT8_C(  96),  INT8_C(  84), -INT8_C( 114),  INT8_C(  46),  INT8_C(  24), -INT8_C(  36), -INT8_C(  88),
         INT8_C(  27), -INT8_C( 124), -INT8_C( 123), -INT8_C(  22), -INT8_C(  20), -INT8_C( 127),  INT8_C(  67),  INT8_C(   8) },
      {  INT8_C(  43), -INT8_C( 108), -INT8_C(  62),  INT8_C(  72), -INT8_C(  56), -INT8_C(  28),  INT8_C(  58),  INT8_C(  87),
         INT8_C(  51), -INT8_C(  47), -INT8_C(  27),  INT8_C(  37), -INT8_C(  49),  INT8_C( 118), -INT8_C(  53),  INT8_C(  42),
         INT8_C(  22),  INT8_C(  31), -INT8_C(  72),  INT8_C(  69),  INT8_C(  56), -INT8_C( 107), -INT8_C(  19),  INT8_C(  83),
         INT8_C(  25),  INT8_C( 114),  INT8_C(  61),  INT8_C(   5), -INT8_C(  13), -INT8_C( 127),  INT8_C(  14),  INT8_C(  30) },
      {  INT8_C(  98),  INT8_C(  62),  INT8_C(  19),  INT8_C(   1), -INT8_C( 117), -INT8_C(  75),  INT8_C(  92),  INT8_C(  73),
        -INT8_C(  62),  INT8_C(  32),  INT8_C( 123),  INT8_C( 104), -INT8_C(  44),  INT8_C( 116),  INT8_C(  92), -INT8_C(  92),
        -INT8_C(   1), -INT8_C(  94),  INT8_C(  12), -INT8_C(  45),  INT8_C( 102), -INT8_C(  83), -INT8_C(  55), -INT8_C(   5),
         INT8_C(  52), -INT8_C( 114), -INT8_C(  60), -INT8_C(  17), -INT8_C(  33),  INT8_C(   3),  INT8_C(  81), -INT8_C(  36) } },
    { {  INT8_C(  21), -INT8_C(  48),  INT8_C( 103), -INT8_C(  35), -INT8_C(  76), -INT8_C(  95),  INT8_C(  52), -INT8_C(  25),
         INT8_C( 114),  INT8_C(  26),  INT8_C(  13),  INT8_C(  65), -INT8_C( 112), -INT8_C(  40),  INT8_C( 108), -INT8_C(  89),
        -INT8_C(   9),  INT8_C(  36), -INT8_C(  20),  INT8_C(  47), -INT8_C(  71), -INT8_C(  39), -INT8_C( 125), -INT8_C(  45),
         INT8_C(  76), -INT8_C(  64), -INT8_C(  40),  INT8_C(  63),  INT8_C(  65), -INT8_C(  26),  INT8_C(  94),  INT8_C(  87) },
      UINT32_C(1781843382),
      {  INT8_C( 102),  INT8_C( 105),  INT8_C(  82), -INT8_C(  39), -INT8_C( 125),  INT8_C(  95),  INT8_C(  26),  INT8_C(  19),
         INT8_C(  55), -INT8_C( 122), -INT8_C(  70),  INT8_C(  46), -INT8_C(  85), -INT8_C(  90),  INT8_C(  94),  INT8_C( 100),
             INT8_MIN, -INT8_C(  31),  INT8_C(  55), -INT8_C(  52), -INT8_C(  95),  INT8_C(  16),  INT8_C(  11), -INT8_C(  29),
        -INT8_C(  10),  INT8_C( 105),  INT8_C(  58), -INT8_C(  83),  INT8_C(  46),  INT8_C( 110),  INT8_C(  23), -INT8_C( 107) },
      { -INT8_C(  41),  INT8_C( 105),  INT8_C( 110),  INT8_C(  90), -INT8_C(  56), -INT8_C( 120),  INT8_C( 110), -INT8_C(   1),
         INT8_C(  15),  INT8_C(  40),  INT8_C(  46), -INT8_C(  70), -INT8_C(  49), -INT8_C( 116),  INT8_C(  30),  INT8_C(  79),
         INT8_C( 109),  INT8_C(  86),  INT8_C(  27),  INT8_C(  14),  INT8_C( 102),  INT8_C(  38), -INT8_C(  15),  INT8_C(  92),
        -INT8_C( 112),  INT8_C(  43),  INT8_C(   9), -INT8_C(  66), -INT8_C( 102),  INT8_C(  33),  INT8_C(  83),  INT8_C( 113) },
      {  INT8_C(  21), -INT8_C(  46), -INT8_C(  64), -INT8_C(  35),  INT8_C(  75), -INT8_C(  25),  INT8_C(  52),  INT8_C(  18),
         INT8_C(  70),  INT8_C(  26), -INT8_C(  24),  INT8_C(  65), -INT8_C( 112), -INT8_C(  40),  INT8_C( 124), -INT8_C(  77),
        -INT8_C(   9),  INT8_C(  36),  INT8_C(  82),  INT8_C(  47),  INT8_C(   7),  INT8_C(  54), -INT8_C( 125), -INT8_C(  45),
         INT8_C(  76), -INT8_C( 108), -INT8_C(  40),  INT8_C( 107),  INT8_C(  65), -INT8_C( 113),  INT8_C( 106),  INT8_C(  87) } },
    { { -INT8_C( 118), -INT8_C(  63), -INT8_C(  52),  INT8_C(  83),  INT8_C(  74),  INT8_C(  58),  INT8_C(  82),  INT8_C(  89),
         INT8_C(  98),      INT8_MIN,  INT8_C(  19),  INT8_C(  49),  INT8_C(  12),  INT8_C(  49),      INT8_MIN,  INT8_C( 121),
        -INT8_C( 121), -INT8_C( 101), -INT8_C( 120), -INT8_C(  19), -INT8_C(  62),  INT8_C( 121),  INT8_C(  74),  INT8_C(  82),
        -INT8_C(  91),  INT8_C(  83),  INT8_C(  16),  INT8_C(  63),  INT8_C( 116),  INT8_C( 100), -INT8_C(  80), -INT8_C(   1) },
      UINT32_C(1867676709),
      { -INT8_C(  74), -INT8_C(  92), -INT8_C(  56),  INT8_C(  25),  INT8_C(  37), -INT8_C(  37),  INT8_C(  74),  INT8_C(  49),
         INT8_C(  13), -INT8_C(  53), -INT8_C(  85), -INT8_C( 108),  INT8_C( 102),  INT8_C(  51), -INT8_C( 126),  INT8_C(  40),
        -INT8_C(  84), -INT8_C(  52),  INT8_C( 122),  INT8_C(  81),  INT8_C(  31), -INT8_C( 117), -INT8_C( 112), -INT8_C( 108),
        -INT8_C(  17),  INT8_C(  65), -INT8_C( 109),  INT8_C(  20), -INT8_C(  67), -INT8_C(  27), -INT8_C( 124),  INT8_C( 116) },
      { -INT8_C( 119),  INT8_C(  76), -INT8_C( 115), -INT8_C(  82),  INT8_C(  40), -INT8_C(  41), -INT8_C(  32),  INT8_C(  53),
        -INT8_C(  94), -INT8_C( 117), -INT8_C(  55),  INT8_C(   9), -INT8_C(  66),  INT8_C(  75),  INT8_C(  49),  INT8_C( 106),
         INT8_C(  23), -INT8_C(  84), -INT8_C(  68),  INT8_C(  55),  INT8_C(  55),  INT8_C(  76), -INT8_C(  53),  INT8_C(  38),
        -INT8_C( 115),  INT8_C(  94),  INT8_C(  58),  INT8_C(  75),  INT8_C(  67), -INT8_C(  66), -INT8_C(  65), -INT8_C(  52) },
      {  INT8_C(  63), -INT8_C(  63),  INT8_C(  85),  INT8_C(  83),  INT8_C(  74), -INT8_C(  78),  INT8_C(  82),  INT8_C(  89),
         INT8_C(  98),      INT8_MIN,  INT8_C( 116), -INT8_C(  99),  INT8_C(  36),  INT8_C( 126), -INT8_C(  77),  INT8_C( 121),
        -INT8_C( 121),  INT8_C( 120), -INT8_C( 120), -INT8_C(  19),  INT8_C(  86),  INT8_C( 121),  INT8_C(  91),  INT8_C(  82),
         INT8_C( 124), -INT8_C(  97), -INT8_C(  51),  INT8_C(  95),  INT8_C( 116), -INT8_C(  93),  INT8_C(  67), -INT8_C(   1) } },
    { {  INT8_C(  11),  INT8_C(  76),  INT8_C( 123),  INT8_C(  51),  INT8_C(  35),  INT8_C(  91),  INT8_C( 104), -INT8_C(  58),
        -INT8_C(  26),  INT8_C(  49), -INT8_C(  49), -INT8_C(  92),  INT8_C( 125),  INT8_C(   0),  INT8_C(  14), -INT8_C( 108),
        -INT8_C(  84), -INT8_C(  54), -INT8_C(  53), -INT8_C(  29),  INT8_C(  23), -INT8_C( 106),  INT8_C(   9), -INT8_C(  92),
        -INT8_C(  12),  INT8_C(  68), -INT8_C(  17),  INT8_C(  55),  INT8_C(   2), -INT8_C(  82),  INT8_C(   4),  INT8_C(  13) },
      UINT32_C( 507543546),
      { -INT8_C(  38), -INT8_C(  88), -INT8_C(  28), -INT8_C(  64), -INT8_C(  38), -INT8_C(  77),  INT8_C( 100),  INT8_C(  87),
        -INT8_C(  77),  INT8_C( 114), -INT8_C(  21),  INT8_C(  96),  INT8_C(  61), -INT8_C(  73),  INT8_C(  67),  INT8_C(  84),
         INT8_C(  77),  INT8_C(  77), -INT8_C(   8),  INT8_C(  66), -INT8_C( 111), -INT8_C(  24),  INT8_C( 121), -INT8_C( 109),
        -INT8_C( 106),  INT8_C( 125), -INT8_C(  95), -INT8_C( 111), -INT8_C(   4), -INT8_C(  31), -INT8_C(  81), -INT8_C(  42) },
      { -INT8_C( 118), -INT8_C( 109), -INT8_C( 106),  INT8_C( 100),  INT8_C(  70), -INT8_C(   6), -INT8_C(  69), -INT8_C(   7),
         INT8_C( 109), -INT8_C(  90),  INT8_C(  89), -INT8_C(  86),  INT8_C(  93), -INT8_C(  99), -INT8_C(   2), -INT8_C(  85),
        -INT8_C(  22), -INT8_C(  10), -INT8_C(  19),  INT8_C( 123), -INT8_C(  34),  INT8_C( 102),  INT8_C(  14),  INT8_C( 117),
        -INT8_C(  28), -INT8_C(  81),  INT8_C(   6), -INT8_C(  32), -INT8_C( 111), -INT8_C(  75), -INT8_C(  73),  INT8_C(  27) },
      {  INT8_C(  11),  INT8_C(  59),  INT8_C( 123),  INT8_C(  36),  INT8_C(  32), -INT8_C(  83),  INT8_C(  31),  INT8_C(  80),
         INT8_C(  32),  INT8_C(  24),  INT8_C(  68),  INT8_C(  10), -INT8_C( 102),  INT8_C(  84),  INT8_C(  65), -INT8_C( 108),
        -INT8_C(  84), -INT8_C(  54), -INT8_C(  53), -INT8_C(  29),  INT8_C(  23), -INT8_C( 106), -INT8_C( 121), -INT8_C(  92),
        -INT8_C(  12),  INT8_C(  44), -INT8_C(  89),  INT8_C( 113), -INT8_C( 115), -INT8_C(  82),  INT8_C(   4),  INT8_C(  13) } },
    { {  INT8_C(  72),  INT8_C(  77),      INT8_MAX, -INT8_C( 114),  INT8_C(  72),  INT8_C(  58), -INT8_C( 121), -INT8_C(  75),
        -INT8_C(  32), -INT8_C(  31),  INT8_C(  95),  INT8_C(  62),  INT8_C( 126),  INT8_C(  93), -INT8_C(  23),  INT8_C( 104),
         INT8_C(  83), -INT8_C(  42), -INT8_C(  29),  INT8_C(  50),  INT8_C(  60), -INT8_C(  15), -INT8_C(  89),  INT8_C(  32),
        -INT8_C(  95), -INT8_C(  83),  INT8_C(   1),  INT8_C(  50),  INT8_C(  98), -INT8_C(  72),  INT8_C(  77), -INT8_C(  86) },
      UINT32_C(1295567877),
      {  INT8_C(   6), -INT8_C(  65),  INT8_C(   2), -INT8_C(  26), -INT8_C(  96),  INT8_C(  97),  INT8_C(  36),  INT8_C(  30),
        -INT8_C(  66),  INT8_C(  13), -INT8_C( 122),  INT8_C(  18), -INT8_C(  29),  INT8_C( 105),  INT8_C(  68),  INT8_C(  32),
         INT8_C(  91), -INT8_C(  21),  INT8_C(  64), -INT8_C(   4), -INT8_C( 104),  INT8_C(  65),  INT8_C(  46), -INT8_C(   6),
        -INT8_C(   7),  INT8_C( 123), -INT8_C(  92), -INT8_C(   1),  INT8_C(  71), -INT8_C(  36),  INT8_C(  76),  INT8_C(  77) },
      { -INT8_C( 101),  INT8_C(  79),  INT8_C(  51),  INT8_C(  60), -INT8_C(  80),  INT8_C(  88),  INT8_C(  90),  INT8_C( 111),
         INT8_C( 101), -INT8_C(  31), -INT8_C( 127),  INT8_C(  73),  INT8_C(  74), -INT8_C(  59),  INT8_C( 105), -INT8_C(  91),
        -INT8_C(  80), -INT8_C(  87), -INT8_C(  95),  INT8_C(  72), -INT8_C(  21), -INT8_C(  49),  INT8_C(  66), -INT8_C(  28),
         INT8_C(  74), -INT8_C(  26), -INT8_C(  29), -INT8_C( 111), -INT8_C(  62),  INT8_C(  48), -INT8_C(  34),  INT8_C(  93) },
      { -INT8_C(  95),  INT8_C(  77),  INT8_C(  53), -INT8_C( 114),  INT8_C(  72),  INT8_C(  58), -INT8_C( 121), -INT8_C(  75),
        -INT8_C(  32), -INT8_C(  31),  INT8_C(   7),  INT8_C(  91),  INT8_C( 126),  INT8_C(  93), -INT8_C(  83), -INT8_C(  59),
         INT8_C(  83), -INT8_C(  42), -INT8_C(  29),  INT8_C(  68), -INT8_C( 125),  INT8_C(  16), -INT8_C(  89),  INT8_C(  32),
         INT8_C(  67), -INT8_C(  83), -INT8_C( 121), -INT8_C( 112),  INT8_C(  98), -INT8_C(  72),  INT8_C(  42), -INT8_C(  86) } },
    { {      INT8_MAX,  INT8_C(  18), -INT8_C( 103),  INT8_C(  47),  INT8_C( 106), -INT8_C(  12), -INT8_C(  98), -INT8_C(  49),
        -INT8_C(  43),  INT8_C(  31),  INT8_C(  24),  INT8_C(  31), -INT8_C(  28), -INT8_C( 127), -INT8_C(  59), -INT8_C( 108),
         INT8_C(  43),  INT8_C( 102), -INT8_C(  36),  INT8_C(  22),  INT8_C(  54),  INT8_C(  30), -INT8_C(   6),      INT8_MIN,
         INT8_C(   4), -INT8_C(  34),  INT8_C(  18), -INT8_C(  58),  INT8_C(  14), -INT8_C(  16),  INT8_C(  36), -INT8_C( 115) },
      UINT32_C(1824308482),
      { -INT8_C(  79),  INT8_C(  91),  INT8_C(  60), -INT8_C( 122),  INT8_C( 122),  INT8_C(  84), -INT8_C(  90),  INT8_C(  95),
        -INT8_C(  42),  INT8_C( 107), -INT8_C(  13),  INT8_C(   1), -INT8_C(  47), -INT8_C(  48),  INT8_C(  23),  INT8_C(   7),
        -INT8_C(  18),  INT8_C(  17), -INT8_C( 120), -INT8_C(  13), -INT8_C(  17), -INT8_C( 102), -INT8_C(  71), -INT8_C(   3),
        -INT8_C( 118), -INT8_C(  35), -INT8_C( 118), -INT8_C( 115), -INT8_C( 101),  INT8_C(  71), -INT8_C(   7),  INT8_C(  76) },
      { -INT8_C(  94),  INT8_C(  53), -INT8_C(  45),  INT8_C(  28), -INT8_C( 118),  INT8_C( 121),  INT8_C( 123),  INT8_C(  96),
        -INT8_C(  28),  INT8_C( 111),  INT8_C(  97), -INT8_C(  75),  INT8_C(  63),  INT8_C( 120), -INT8_C(  67),  INT8_C(  45),
        -INT8_C( 119),  INT8_C(  69),  INT8_C(  32),  INT8_C( 121), -INT8_C(  33), -INT8_C(  38),  INT8_C( 118),  INT8_C( 105),
        -INT8_C(  73),  INT8_C(   1), -INT8_C(  10),  INT8_C(  82),  INT8_C(  72), -INT8_C(  16), -INT8_C(  97), -INT8_C(  22) },
      {      INT8_MAX, -INT8_C( 112), -INT8_C( 103),  INT8_C(  47),  INT8_C( 106), -INT8_C(  12), -INT8_C(  98), -INT8_C(  49),
        -INT8_C(  70),  INT8_C(  31),  INT8_C(  84), -INT8_C(  74),  INT8_C(  16),  INT8_C(  72), -INT8_C(  59),  INT8_C(  52),
         INT8_C(  43),  INT8_C( 102), -INT8_C(  88),  INT8_C( 108), -INT8_C(  50),  INT8_C( 116), -INT8_C(   6),  INT8_C( 102),
         INT8_C(   4), -INT8_C(  34),      INT8_MIN, -INT8_C(  33),  INT8_C(  14),  INT8_C(  55), -INT8_C( 104), -INT8_C( 115) } },
    { {  INT8_C(  37),  INT8_C( 114),  INT8_C(   6), -INT8_C(  81), -INT8_C(  21), -INT8_C( 126),  INT8_C(  15), -INT8_C(  49),
        -INT8_C(  15),  INT8_C( 112), -INT8_C( 124),  INT8_C(  48), -INT8_C(  24),  INT8_C(  65),  INT8_C(  93),  INT8_C( 114),
        -INT8_C( 122),  INT8_C( 126), -INT8_C(  21),  INT8_C( 101),  INT8_C(  88),  INT8_C(  97), -INT8_C(  49),  INT8_C(  15),
         INT8_C(  98), -INT8_C(  59),  INT8_C(  98), -INT8_C(  86), -INT8_C(  75),  INT8_C(   1), -INT8_C( 108), -INT8_C(  37) },
      UINT32_C(1586142067),
      {  INT8_C(  29), -INT8_C( 102),  INT8_C(  45),  INT8_C(  14),  INT8_C(  10), -INT8_C(  79),  INT8_C(  62), -INT8_C(  13),
        -INT8_C(  13), -INT8_C( 101),  INT8_C( 101),  INT8_C( 121),  INT8_C(  25),  INT8_C(  80), -INT8_C(  33),  INT8_C( 113),
        -INT8_C(  79), -INT8_C(  82), -INT8_C( 127),  INT8_C(  20),  INT8_C( 115), -INT8_C(  29), -INT8_C(  66),  INT8_C(  41),
        -INT8_C(  28),  INT8_C(  83),  INT8_C(   4),  INT8_C(  87), -INT8_C(  18), -INT8_C( 114), -INT8_C(  75),  INT8_C(  11) },
      {  INT8_C(  40), -INT8_C(  30),  INT8_C(  25),  INT8_C(  51), -INT8_C( 109),  INT8_C(  87),  INT8_C(  38), -INT8_C( 122),
        -INT8_C(  14), -INT8_C( 117),  INT8_C(   0),  INT8_C(  12), -INT8_C(  37), -INT8_C(  33),  INT8_C( 125), -INT8_C( 116),
        -INT8_C( 115), -INT8_C(   2), -INT8_C(  96),  INT8_C(   0), -INT8_C(  31),  INT8_C(  95),  INT8_C(  41), -INT8_C(  59),
        -INT8_C(  78),  INT8_C(  45),  INT8_C(  28), -INT8_C(  96), -INT8_C(  68), -INT8_C(  47), -INT8_C(  85), -INT8_C(  28) },
      {  INT8_C(  69),  INT8_C( 124),  INT8_C(   6), -INT8_C(  81), -INT8_C(  99),  INT8_C(   8),  INT8_C( 100), -INT8_C(  49),
        -INT8_C(  27),  INT8_C(  38), -INT8_C( 124), -INT8_C( 123), -INT8_C(  12),  INT8_C(  65),  INT8_C(  93), -INT8_C(   3),
        -INT8_C( 122), -INT8_C(  84), -INT8_C(  21),  INT8_C(  20),  INT8_C(  88),  INT8_C(  97), -INT8_C(  49), -INT8_C(  18),
         INT8_C(  98),      INT8_MIN,  INT8_C(  32), -INT8_C(   9), -INT8_C(  86),  INT8_C(   1),  INT8_C(  96), -INT8_C(  37) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi8(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_add_epi8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_add_epi8");
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
    easysimd__m256i r = easysimd_mm256_mask_add_epi8(src, k, a, b);

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
test_easysimd_mm256_maskz_add_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t k;
    const int8_t a[32];
    const int8_t b[32];
    const int8_t r[32];
  } test_vec[] = {
    { UINT32_C(1969135549),
      { -INT8_C(  22),  INT8_C(  49),  INT8_C(  57), -INT8_C(  52),  INT8_C( 108), -INT8_C(  56), -INT8_C( 102),  INT8_C(  80),
         INT8_C(  62), -INT8_C(  45),  INT8_C(   5),  INT8_C( 119), -INT8_C(  18),  INT8_C(   3), -INT8_C(  24), -INT8_C(  34),
         INT8_C(  76),  INT8_C(  19),  INT8_C( 113),  INT8_C(  49), -INT8_C(  81),  INT8_C(  56), -INT8_C(   3), -INT8_C(  53),
         INT8_C(  34),  INT8_C(   9), -INT8_C(  11), -INT8_C(  33), -INT8_C(  88),  INT8_C(  83),  INT8_C(  84), -INT8_C( 110) },
      { -INT8_C( 124), -INT8_C( 115),  INT8_C(  94), -INT8_C(  15),  INT8_C(  85), -INT8_C(   8),  INT8_C(  65), -INT8_C( 109),
        -INT8_C(  52),  INT8_C(  71),  INT8_C(  10), -INT8_C(  70),  INT8_C(  74), -INT8_C(  13), -INT8_C( 104), -INT8_C( 106),
         INT8_C(   6),  INT8_C(  10), -INT8_C(  57), -INT8_C(  74),  INT8_C(  66), -INT8_C(  60), -INT8_C( 127),  INT8_C( 100),
        -INT8_C(  51),  INT8_C( 118),  INT8_C(  67),  INT8_C( 117), -INT8_C(  55), -INT8_C( 105),  INT8_C(   7),  INT8_C(  78) },
      {  INT8_C( 110),  INT8_C(   0), -INT8_C( 105), -INT8_C(  67), -INT8_C(  63), -INT8_C(  64),  INT8_C(   0), -INT8_C(  29),
         INT8_C(  10),  INT8_C(  26),  INT8_C(  15),  INT8_C(  49),  INT8_C(  56),  INT8_C(   0),  INT8_C(   0),  INT8_C( 116),
         INT8_C(   0),  INT8_C(  29),  INT8_C(  56), -INT8_C(  25), -INT8_C(  15),  INT8_C(   0),  INT8_C( 126),  INT8_C(   0),
        -INT8_C(  17),  INT8_C(   0),  INT8_C(  56),  INT8_C(   0),  INT8_C( 113), -INT8_C(  22),  INT8_C(  91),  INT8_C(   0) } },
    { UINT32_C(2050975269),
      {  INT8_C(  94),      INT8_MIN,  INT8_C(  14),  INT8_C(  42), -INT8_C(  57),  INT8_C(  24), -INT8_C(  28),  INT8_C(  17),
         INT8_C(  11),  INT8_C( 125), -INT8_C(  88),  INT8_C(  18), -INT8_C( 121),  INT8_C( 111), -INT8_C(  56), -INT8_C(  55),
         INT8_C(  52),  INT8_C(  73),  INT8_C(  46),  INT8_C(   1), -INT8_C(  64),  INT8_C( 113),  INT8_C( 119), -INT8_C( 119),
         INT8_C(   9),  INT8_C( 126), -INT8_C(  41),  INT8_C(  46), -INT8_C(  28),  INT8_C(  22), -INT8_C(  88),  INT8_C(  67) },
      { -INT8_C( 105), -INT8_C(  74),  INT8_C( 109),  INT8_C(  94), -INT8_C(  49),  INT8_C(  82),  INT8_C( 112), -INT8_C(  38),
        -INT8_C(  49),  INT8_C(  24), -INT8_C(  20),  INT8_C(  86), -INT8_C( 121), -INT8_C(  76),  INT8_C(  31), -INT8_C(  69),
        -INT8_C(   2),  INT8_C(  77), -INT8_C(  67), -INT8_C(  66), -INT8_C(  65),  INT8_C(  52),  INT8_C(  71), -INT8_C(  56),
        -INT8_C(  78),  INT8_C(  31), -INT8_C(  10), -INT8_C( 105),  INT8_C(  53), -INT8_C(  98), -INT8_C(  38), -INT8_C(  52) },
      { -INT8_C(  11),  INT8_C(   0),  INT8_C( 123),  INT8_C(   0),  INT8_C(   0),  INT8_C( 106),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C( 107), -INT8_C( 108),  INT8_C(   0),  INT8_C(   0),  INT8_C(  35), -INT8_C(  25),  INT8_C(   0),
         INT8_C(  50), -INT8_C( 106), -INT8_C(  21), -INT8_C(  65),      INT8_MAX, -INT8_C(  91),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  99),  INT8_C(   0), -INT8_C(  59),  INT8_C(  25), -INT8_C(  76), -INT8_C( 126),  INT8_C(   0) } },
    { UINT32_C( 606816085),
      { -INT8_C( 103), -INT8_C( 101), -INT8_C(   2),  INT8_C( 104), -INT8_C(  77), -INT8_C(  21), -INT8_C(  66),  INT8_C(  58),
        -INT8_C(  97), -INT8_C(  34), -INT8_C(  10), -INT8_C(  99),  INT8_C(  43), -INT8_C(  77),  INT8_C(  91), -INT8_C(  22),
        -INT8_C(  25), -INT8_C(  93), -INT8_C(  78), -INT8_C( 103), -INT8_C(  62), -INT8_C(  88),  INT8_C(  48), -INT8_C(   9),
         INT8_C(  71),  INT8_C(  10), -INT8_C(  60), -INT8_C( 100),  INT8_C(  82), -INT8_C(  17), -INT8_C(  64), -INT8_C(  21) },
      { -INT8_C( 118), -INT8_C(  66),  INT8_C(  84),  INT8_C(  61), -INT8_C(  87),  INT8_C(  18),  INT8_C( 119),  INT8_C(  73),
        -INT8_C(  16),  INT8_C( 109), -INT8_C(  26),  INT8_C(  28),  INT8_C(  32),  INT8_C(  66),  INT8_C(   6),  INT8_C(   7),
        -INT8_C(  27), -INT8_C(  71), -INT8_C(  95), -INT8_C(  89),  INT8_C(  97), -INT8_C(  47), -INT8_C(  98), -INT8_C(  88),
        -INT8_C(  36),  INT8_C(  98),  INT8_C(  68),  INT8_C(  46),  INT8_C(  81),  INT8_C(   4),  INT8_C(  25), -INT8_C(  37) },
      {  INT8_C(  35),  INT8_C(   0),  INT8_C(  82),  INT8_C(   0),  INT8_C(  92),  INT8_C(   0),  INT8_C(  53),  INT8_C(   0),
        -INT8_C( 113),  INT8_C(  75), -INT8_C(  36),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  97),  INT8_C(   0),
        -INT8_C(  52),  INT8_C(  92),  INT8_C(   0),  INT8_C(  64),  INT8_C(   0),  INT8_C( 121),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   8),  INT8_C(   0),  INT8_C(   0), -INT8_C(  13),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C(1813540291),
      {      INT8_MIN, -INT8_C( 112), -INT8_C(  75),  INT8_C( 112), -INT8_C(   3), -INT8_C( 100), -INT8_C( 116),  INT8_C(  30),
        -INT8_C(  34), -INT8_C( 109),  INT8_C(  37), -INT8_C(  61),  INT8_C(  76), -INT8_C(  58),  INT8_C( 106), -INT8_C(  83),
        -INT8_C( 104),  INT8_C(   8),  INT8_C(  86),  INT8_C( 116),  INT8_C( 107), -INT8_C( 102), -INT8_C(  94), -INT8_C(  68),
        -INT8_C(  97), -INT8_C(  69), -INT8_C( 104),  INT8_C(  98),  INT8_C(  41), -INT8_C(  80), -INT8_C(  50), -INT8_C(  87) },
      {  INT8_C(  64), -INT8_C( 124),  INT8_C(  25),  INT8_C(  62),  INT8_C(  32), -INT8_C(  90),  INT8_C(  92), -INT8_C(   2),
         INT8_C(  57), -INT8_C( 127), -INT8_C(  63), -INT8_C( 123),  INT8_C(  72),  INT8_C(  43),  INT8_C(  50), -INT8_C(  32),
         INT8_C(  51), -INT8_C( 120),  INT8_C(  84), -INT8_C(  98),  INT8_C(  35), -INT8_C(  10),  INT8_C(  91), -INT8_C(  62),
        -INT8_C(  79), -INT8_C(  13),  INT8_C(  36), -INT8_C(  38), -INT8_C(  93), -INT8_C(  14), -INT8_C( 125), -INT8_C(  28) },
      { -INT8_C(  64),  INT8_C(  20),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  24),  INT8_C(  28),
         INT8_C(  23),  INT8_C(   0), -INT8_C(  26),  INT8_C(  72),  INT8_C(   0), -INT8_C(  15), -INT8_C( 100),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  18), -INT8_C( 114),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  68),  INT8_C(  60),  INT8_C(   0), -INT8_C(  94),  INT8_C(  81),  INT8_C(   0) } },
    { UINT32_C(2518850934),
      {  INT8_C(  67),  INT8_C( 126), -INT8_C( 108),  INT8_C( 124), -INT8_C(   1),  INT8_C(  85),  INT8_C(   1),  INT8_C(  71),
             INT8_MIN,  INT8_C(  51),  INT8_C(  39), -INT8_C(  76), -INT8_C(  68),  INT8_C( 123),  INT8_C(  82), -INT8_C(  33),
         INT8_C( 113), -INT8_C(  83), -INT8_C(  95),  INT8_C(  35), -INT8_C(  96), -INT8_C(  59), -INT8_C(   3),  INT8_C(  68),
        -INT8_C(  73), -INT8_C( 127),  INT8_C(  40),  INT8_C(  46),  INT8_C(  30),  INT8_C(  74), -INT8_C(  60),  INT8_C(  97) },
      { -INT8_C(  56),  INT8_C(  89), -INT8_C(  35), -INT8_C(  57), -INT8_C(  82), -INT8_C(  34),  INT8_C(  15),  INT8_C(  47),
         INT8_C(  17),  INT8_C(  54), -INT8_C(  29), -INT8_C(  51), -INT8_C(  78),  INT8_C(  53), -INT8_C(  84),  INT8_C(  35),
        -INT8_C(  29),  INT8_C(  77),  INT8_C(  70), -INT8_C( 125),  INT8_C(  18),  INT8_C(  68), -INT8_C(  57), -INT8_C(  54),
        -INT8_C(  59), -INT8_C(  17), -INT8_C(   8), -INT8_C(  29),  INT8_C(  57), -INT8_C(  68),  INT8_C(  68),  INT8_C(   1) },
      {  INT8_C(   0), -INT8_C(  41),  INT8_C( 113),  INT8_C(   0), -INT8_C(  83),  INT8_C(  51),  INT8_C(  16),  INT8_C(   0),
        -INT8_C( 111),  INT8_C(   0),  INT8_C(  10), -INT8_C( 127),  INT8_C( 110),  INT8_C(   0),  INT8_C(   0),  INT8_C(   2),
         INT8_C(   0), -INT8_C(   6),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   9),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C( 112),  INT8_C(  32),  INT8_C(   0),  INT8_C(  87),  INT8_C(   0),  INT8_C(   0),  INT8_C(  98) } },
    { UINT32_C(3301515541),
      { -INT8_C(   1), -INT8_C(  40), -INT8_C(  13),  INT8_C(  16),  INT8_C(  14), -INT8_C(  42), -INT8_C(  34), -INT8_C(  64),
         INT8_C(  11), -INT8_C( 118), -INT8_C(  28), -INT8_C(  18), -INT8_C(  40),  INT8_C(  42),  INT8_C( 114), -INT8_C(  22),
         INT8_C( 110),  INT8_C(  57), -INT8_C(  76),  INT8_C(  51),  INT8_C(  41), -INT8_C(  84),  INT8_C(  22),  INT8_C(  98),
         INT8_C( 105),  INT8_C(  90),  INT8_C( 100),  INT8_C( 126),  INT8_C( 123),  INT8_C(  45),  INT8_C(  66),  INT8_C( 122) },
      {  INT8_C(   5),  INT8_C(  53), -INT8_C( 117),  INT8_C(  19),  INT8_C(  11),  INT8_C( 105), -INT8_C(  44),  INT8_C(  23),
        -INT8_C(  13), -INT8_C(  72),  INT8_C(   5), -INT8_C(  53), -INT8_C(  30),  INT8_C( 119), -INT8_C(  74),  INT8_C(  81),
        -INT8_C(  79),  INT8_C( 106), -INT8_C( 124), -INT8_C(  38),  INT8_C(  23), -INT8_C( 101),  INT8_C(  60),      INT8_MIN,
        -INT8_C(  11), -INT8_C(  96), -INT8_C(   2),  INT8_C( 113), -INT8_C(  51),  INT8_C(  65), -INT8_C(  21), -INT8_C(  46) },
      {  INT8_C(   4),  INT8_C(   0),  INT8_C( 126),  INT8_C(   0),  INT8_C(  25),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(   2),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  95),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  31),  INT8_C(   0),  INT8_C(   0),  INT8_C(  13),  INT8_C(   0),  INT8_C(   0),  INT8_C(  82), -INT8_C(  30),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  98),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  45),  INT8_C(  76) } },
    { UINT32_C(2196141686),
      { -INT8_C(  33), -INT8_C(  70), -INT8_C( 103), -INT8_C(  45),  INT8_C( 114), -INT8_C(  98), -INT8_C(  98),  INT8_C(  84),
         INT8_C(  22),  INT8_C(  84), -INT8_C(  91), -INT8_C(  57), -INT8_C(  65),  INT8_C(  42), -INT8_C(  95), -INT8_C(  42),
        -INT8_C(  59), -INT8_C(  35),  INT8_C(  86), -INT8_C(  70),  INT8_C( 126),  INT8_C(  84),  INT8_C(  43),  INT8_C(  75),
        -INT8_C( 107),  INT8_C(  23),  INT8_C(  30),  INT8_C(  12), -INT8_C( 115),  INT8_C(   4), -INT8_C( 114),  INT8_C( 109) },
      { -INT8_C(  66),  INT8_C(  39),  INT8_C(  64),  INT8_C(  48), -INT8_C(  59), -INT8_C(  34), -INT8_C( 124), -INT8_C(  37),
         INT8_C(  51),  INT8_C(  42), -INT8_C(  94), -INT8_C(  14),  INT8_C(  84),  INT8_C(  67), -INT8_C(  56),  INT8_C(  25),
         INT8_C(  33),  INT8_C(  30), -INT8_C(  45), -INT8_C(  97),  INT8_C( 114), -INT8_C(   1), -INT8_C(  22),  INT8_C(   8),
         INT8_C(  22),  INT8_C(   8),  INT8_C(  20), -INT8_C(  93),  INT8_C(  12), -INT8_C(  94),  INT8_C(  16), -INT8_C(  54) },
      {  INT8_C(   0), -INT8_C(  31), -INT8_C(  39),  INT8_C(   0),  INT8_C(  55),  INT8_C( 124),  INT8_C(  34),  INT8_C(   0),
         INT8_C(   0),  INT8_C( 126),  INT8_C(  71),  INT8_C(   0),  INT8_C(  19),  INT8_C( 109),  INT8_C( 105),  INT8_C(   0),
         INT8_C(   0), -INT8_C(   5),  INT8_C(  41),  INT8_C(   0),  INT8_C(   0),  INT8_C(  83),  INT8_C(  21),  INT8_C(  83),
         INT8_C(   0),  INT8_C(  31),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  55) } },
    { UINT32_C(2398769353),
      {  INT8_C(  47),      INT8_MAX,  INT8_C( 106),  INT8_C(  98), -INT8_C(  87),  INT8_C(  12),  INT8_C(  84), -INT8_C(   3),
         INT8_C(  80),  INT8_C(  28),  INT8_C(  22),  INT8_C( 113),  INT8_C(  58), -INT8_C(  23),  INT8_C(  16), -INT8_C(  84),
        -INT8_C(  24), -INT8_C(   6), -INT8_C(  76), -INT8_C(   2),  INT8_C(   3), -INT8_C(  56), -INT8_C(  94),  INT8_C(  15),
         INT8_C( 106), -INT8_C(  78), -INT8_C(  38),  INT8_C(  51),  INT8_C(   3), -INT8_C(  44), -INT8_C(  62),  INT8_C(  50) },
      {  INT8_C(  83),  INT8_C(  44), -INT8_C( 108), -INT8_C(   4),  INT8_C(  56), -INT8_C(  24), -INT8_C(   7), -INT8_C( 120),
         INT8_C(   4),  INT8_C(  15), -INT8_C(   7),  INT8_C(  62), -INT8_C(   7),  INT8_C(   9), -INT8_C(  22), -INT8_C(  31),
         INT8_C(   4), -INT8_C(  97), -INT8_C(  32),  INT8_C(   7),  INT8_C( 103), -INT8_C( 126),  INT8_C(  22), -INT8_C(  46),
         INT8_C(  52), -INT8_C(  16),  INT8_C(   5),  INT8_C(  55), -INT8_C(  59), -INT8_C(  57),  INT8_C( 105),  INT8_C(  24) },
      { -INT8_C( 126),  INT8_C(   0),  INT8_C(   0),  INT8_C(  94),  INT8_C(   0),  INT8_C(   0),  INT8_C(  77), -INT8_C( 123),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  51),  INT8_C(   0), -INT8_C(   6),  INT8_C(   0),
         INT8_C(   0), -INT8_C( 103),  INT8_C(   0),  INT8_C(   5),  INT8_C( 106),  INT8_C(  74), -INT8_C(  72), -INT8_C(  31),
         INT8_C(   0), -INT8_C(  94), -INT8_C(  33),  INT8_C( 106),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  74) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_add_epi8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_add_epi8");
    easysimd_test_x86_assert_equal_i8x32(r, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = easysimd_mm256_maskz_add_epi8(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_add_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[16];
    const easysimd__mmask16 k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { { -INT16_C( 18088),  INT16_C(  1606), -INT16_C(  7503),  INT16_C(  5992),  INT16_C( 28165),  INT16_C( 22274), -INT16_C( 26583), -INT16_C( 13076),
        -INT16_C( 12908),  INT16_C( 10372),  INT16_C( 29897),  INT16_C(   832),  INT16_C( 19766),  INT16_C( 21637), -INT16_C( 21677),  INT16_C( 16763) },
      UINT16_C(60084),
      {  INT16_C( 31188),  INT16_C( 10550), -INT16_C( 29740),  INT16_C( 24731), -INT16_C( 25844),  INT16_C( 20568), -INT16_C( 21197),  INT16_C( 13543),
        -INT16_C( 13074), -INT16_C(  5446),  INT16_C(  8264), -INT16_C( 22453), -INT16_C( 30162), -INT16_C(  8771),  INT16_C( 29554), -INT16_C(  6204) },
      {  INT16_C( 13652), -INT16_C(  1326), -INT16_C(  8634), -INT16_C( 16805), -INT16_C(  7376), -INT16_C(  1060), -INT16_C( 17004),  INT16_C(  3029),
        -INT16_C( 23256), -INT16_C( 24573), -INT16_C(  5392),  INT16_C(  9842),  INT16_C( 11581), -INT16_C( 25918), -INT16_C(  1929),  INT16_C(  6096) },
      { -INT16_C( 18088),  INT16_C(  1606),  INT16_C( 27162),  INT16_C(  5992),  INT16_C( 32316),  INT16_C( 19508), -INT16_C( 26583),  INT16_C( 16572),
        -INT16_C( 12908), -INT16_C( 30019),  INT16_C( 29897), -INT16_C( 12611),  INT16_C( 19766),  INT16_C( 30847),  INT16_C( 27625), -INT16_C(   108) } },
    { { -INT16_C( 28260),  INT16_C(  3355),  INT16_C( 30149),  INT16_C( 18841),  INT16_C(   897), -INT16_C( 18335), -INT16_C( 27856),  INT16_C( 31843),
         INT16_C( 24626), -INT16_C( 30048), -INT16_C(  8679),  INT16_C(  1167),  INT16_C(  9529),  INT16_C( 15283), -INT16_C( 18329),  INT16_C(  1611) },
      UINT16_C(36795),
      { -INT16_C( 20367),  INT16_C( 17977), -INT16_C( 10340),  INT16_C( 23796),  INT16_C(  4308), -INT16_C(  8548), -INT16_C( 31482), -INT16_C( 21319),
         INT16_C(    17), -INT16_C( 10987),  INT16_C( 12850),  INT16_C(  9166),  INT16_C(  5118), -INT16_C( 13916),  INT16_C( 27754), -INT16_C( 17049) },
      { -INT16_C( 29649),  INT16_C(  2250),  INT16_C( 31457),  INT16_C( 19428), -INT16_C( 17826),  INT16_C(  8517),  INT16_C(  3008), -INT16_C( 15739),
        -INT16_C(  3791), -INT16_C( 20340),  INT16_C( 28400),  INT16_C( 31847), -INT16_C( 20111),  INT16_C(  8560), -INT16_C(  3090), -INT16_C( 17728) },
      {  INT16_C( 15520),  INT16_C( 20227),  INT16_C( 30149), -INT16_C( 22312), -INT16_C( 13518), -INT16_C(    31), -INT16_C( 27856),  INT16_C( 28478),
        -INT16_C(  3774), -INT16_C( 31327), -INT16_C( 24286), -INT16_C( 24523),  INT16_C(  9529),  INT16_C( 15283), -INT16_C( 18329),  INT16_C( 30759) } },
    { {  INT16_C(  5027),  INT16_C( 31603), -INT16_C(  3038),  INT16_C( 25247),  INT16_C( 12894), -INT16_C(  2609),  INT16_C( 30080),  INT16_C( 22396),
        -INT16_C(  4776), -INT16_C(  7593), -INT16_C(  5922), -INT16_C( 20302),  INT16_C( 26389), -INT16_C(  2181),  INT16_C( 14645),  INT16_C(  6951) },
      UINT16_C(62936),
      { -INT16_C( 18607),  INT16_C( 18295), -INT16_C(  2304),  INT16_C( 28875), -INT16_C(  8411),  INT16_C(  2635),  INT16_C( 16083),  INT16_C(  5198),
        -INT16_C( 31435), -INT16_C(  4074),  INT16_C( 13887), -INT16_C(  4755),  INT16_C( 30885), -INT16_C(  1538), -INT16_C( 32035), -INT16_C(  4113) },
      {  INT16_C(  2642),  INT16_C(  4797), -INT16_C(  3564), -INT16_C( 27490),  INT16_C( 16589),  INT16_C( 10413),  INT16_C( 30700), -INT16_C(  8724),
         INT16_C( 25964),  INT16_C(  5287), -INT16_C( 19421),  INT16_C( 28120), -INT16_C( 30880), -INT16_C( 19119), -INT16_C( 19912), -INT16_C( 27324) },
      {  INT16_C(  5027),  INT16_C( 31603), -INT16_C(  3038),  INT16_C(  1385),  INT16_C(  8178), -INT16_C(  2609), -INT16_C( 18753), -INT16_C(  3526),
        -INT16_C(  5471), -INT16_C(  7593), -INT16_C(  5534), -INT16_C( 20302),  INT16_C(     5), -INT16_C( 20657),  INT16_C( 13589), -INT16_C( 31437) } },
    { {  INT16_C( 25041),  INT16_C( 25663),  INT16_C( 28102),  INT16_C( 28245),  INT16_C(  5599),  INT16_C( 24300), -INT16_C( 10886), -INT16_C(  9183),
         INT16_C(  3807),  INT16_C( 25125), -INT16_C( 25229), -INT16_C(  4659), -INT16_C( 15898),  INT16_C( 11743), -INT16_C( 28195),  INT16_C( 25686) },
      UINT16_C(55587),
      {  INT16_C(  2381), -INT16_C( 16078), -INT16_C( 27768),  INT16_C( 27601),  INT16_C( 16912),  INT16_C(  8673), -INT16_C( 29342), -INT16_C(  7031),
         INT16_C( 30710), -INT16_C(  7797), -INT16_C( 30252),  INT16_C( 23527), -INT16_C(  2137),  INT16_C( 31719),  INT16_C( 32572),  INT16_C( 32012) },
      { -INT16_C( 11761),  INT16_C( 10830), -INT16_C( 22217), -INT16_C( 26449), -INT16_C( 13353), -INT16_C(  6701), -INT16_C( 21906),  INT16_C( 26387),
         INT16_C( 28827), -INT16_C( 13374), -INT16_C(  1039), -INT16_C( 19312), -INT16_C(  2641),  INT16_C( 17473), -INT16_C( 21172), -INT16_C( 20086) },
      { -INT16_C(  9380), -INT16_C(  5248),  INT16_C( 28102),  INT16_C( 28245),  INT16_C(  5599),  INT16_C(  1972), -INT16_C( 10886), -INT16_C(  9183),
        -INT16_C(  5999),  INT16_C( 25125), -INT16_C( 25229),  INT16_C(  4215), -INT16_C(  4778),  INT16_C( 11743),  INT16_C( 11400),  INT16_C( 11926) } },
    { {  INT16_C( 12057),  INT16_C( 29150),  INT16_C( 23614),  INT16_C( 25130),  INT16_C( 10655), -INT16_C( 27568),  INT16_C( 28503), -INT16_C( 25443),
        -INT16_C( 10717), -INT16_C( 20322), -INT16_C( 29779), -INT16_C( 24431),  INT16_C( 27510), -INT16_C(  9356),  INT16_C(  4476),  INT16_C( 16174) },
      UINT16_C(36512),
      {  INT16_C( 19088),  INT16_C( 10128),  INT16_C( 13140),  INT16_C( 29098), -INT16_C(  3415),  INT16_C(  2524), -INT16_C( 20169), -INT16_C( 11551),
        -INT16_C( 26829),  INT16_C(  3366),  INT16_C( 12131),  INT16_C(  2463),  INT16_C( 30355),  INT16_C( 12022),  INT16_C( 19040),  INT16_C( 20815) },
      { -INT16_C( 24937),  INT16_C( 16359), -INT16_C(   602),  INT16_C( 32707), -INT16_C(  4799),  INT16_C( 16001),  INT16_C( 10569),  INT16_C(  2669),
         INT16_C(  8145), -INT16_C(   957),  INT16_C( 11518), -INT16_C( 18084), -INT16_C( 29708),  INT16_C(  9779),  INT16_C(   904), -INT16_C( 28159) },
      {  INT16_C( 12057),  INT16_C( 29150),  INT16_C( 23614),  INT16_C( 25130),  INT16_C( 10655),  INT16_C( 18525),  INT16_C( 28503), -INT16_C(  8882),
        -INT16_C( 10717),  INT16_C(  2409),  INT16_C( 23649), -INT16_C( 15621),  INT16_C( 27510), -INT16_C(  9356),  INT16_C(  4476), -INT16_C(  7344) } },
    { {  INT16_C( 13479),  INT16_C( 32082), -INT16_C(  1052), -INT16_C( 28178), -INT16_C(  5151),  INT16_C( 15355), -INT16_C( 21898), -INT16_C(  6248),
         INT16_C( 26798),  INT16_C( 24344), -INT16_C( 25169),  INT16_C( 15648), -INT16_C( 31017),  INT16_C(  1114), -INT16_C( 19793),  INT16_C( 27930) },
      UINT16_C( 8052),
      { -INT16_C( 26007),  INT16_C( 21812), -INT16_C( 28453), -INT16_C( 22252), -INT16_C(  2350),  INT16_C( 20554), -INT16_C(   112),  INT16_C( 28501),
         INT16_C( 20387),  INT16_C( 15898), -INT16_C(  4455),  INT16_C(  8743), -INT16_C(  3007),  INT16_C(  1364),  INT16_C( 26716), -INT16_C( 32024) },
      { -INT16_C( 26027), -INT16_C( 22945), -INT16_C( 11918), -INT16_C( 14268), -INT16_C( 11610), -INT16_C( 22539),  INT16_C(  6421), -INT16_C( 16498),
         INT16_C( 26152), -INT16_C( 22496), -INT16_C(  1902), -INT16_C( 10031),  INT16_C(  9340), -INT16_C( 16924), -INT16_C( 19351), -INT16_C( 25237) },
      {  INT16_C( 13479),  INT16_C( 32082),  INT16_C( 25165), -INT16_C( 28178), -INT16_C( 13960), -INT16_C(  1985),  INT16_C(  6309), -INT16_C(  6248),
        -INT16_C( 18997), -INT16_C(  6598), -INT16_C(  6357), -INT16_C(  1288),  INT16_C(  6333),  INT16_C(  1114), -INT16_C( 19793),  INT16_C( 27930) } },
    { { -INT16_C(  9402), -INT16_C(  4652),  INT16_C( 27455), -INT16_C( 17628),  INT16_C( 12568), -INT16_C(  6414), -INT16_C( 29207), -INT16_C(  1798),
        -INT16_C(  7113), -INT16_C( 19430),  INT16_C( 19790),  INT16_C( 17330), -INT16_C( 30097),  INT16_C( 28960),  INT16_C( 28059), -INT16_C( 12652) },
      UINT16_C(55494),
      {  INT16_C(  4668),  INT16_C( 21727),  INT16_C( 23842),  INT16_C(   859),  INT16_C( 24202), -INT16_C(  4733), -INT16_C( 18553), -INT16_C( 32032),
        -INT16_C( 22813),  INT16_C(   981), -INT16_C(  1521), -INT16_C( 13085), -INT16_C( 20947), -INT16_C(  1346), -INT16_C(  1102), -INT16_C( 30046) },
      {  INT16_C( 29305), -INT16_C( 30721),  INT16_C( 31389), -INT16_C(  1704),  INT16_C(  2370), -INT16_C( 31381),  INT16_C( 22976),  INT16_C(  6971),
         INT16_C(  2368),  INT16_C( 11549), -INT16_C(  7631), -INT16_C(  5582), -INT16_C( 17938), -INT16_C(  4477),  INT16_C( 28539), -INT16_C(  8234) },
      { -INT16_C(  9402), -INT16_C(  8994), -INT16_C( 10305), -INT16_C( 17628),  INT16_C( 12568), -INT16_C(  6414),  INT16_C(  4423), -INT16_C( 25061),
        -INT16_C(  7113), -INT16_C( 19430),  INT16_C( 19790), -INT16_C( 18667),  INT16_C( 26651),  INT16_C( 28960),  INT16_C( 27437),  INT16_C( 27256) } },
    { {  INT16_C( 17152), -INT16_C( 16650), -INT16_C( 14707),  INT16_C( 11632),  INT16_C( 19908), -INT16_C( 20810),  INT16_C(   377),  INT16_C( 15379),
        -INT16_C( 10283), -INT16_C( 26722), -INT16_C(   177), -INT16_C( 17760),  INT16_C(  8591),  INT16_C(  3646), -INT16_C( 16462),  INT16_C( 17616) },
      UINT16_C( 3229),
      {  INT16_C( 32634),  INT16_C( 32728),  INT16_C( 30084), -INT16_C( 23413), -INT16_C( 22910),  INT16_C( 20672),  INT16_C( 22735), -INT16_C( 29137),
        -INT16_C(  1705),  INT16_C(  5378),  INT16_C( 17391),  INT16_C( 15446), -INT16_C(  2220), -INT16_C(  3224), -INT16_C( 13014), -INT16_C( 19705) },
      {  INT16_C( 14637),  INT16_C( 15248), -INT16_C(  4954), -INT16_C( 14311), -INT16_C(   296), -INT16_C( 30474), -INT16_C(  5162), -INT16_C( 32073),
         INT16_C( 26895),  INT16_C( 18041), -INT16_C( 18825),  INT16_C(  9328),  INT16_C( 19262),  INT16_C(  3861), -INT16_C( 29853),  INT16_C( 17210) },
      { -INT16_C( 18265), -INT16_C( 16650),  INT16_C( 25130),  INT16_C( 27812), -INT16_C( 23206), -INT16_C( 20810),  INT16_C(   377),  INT16_C(  4326),
        -INT16_C( 10283), -INT16_C( 26722), -INT16_C(  1434),  INT16_C( 24774),  INT16_C(  8591),  INT16_C(  3646), -INT16_C( 16462),  INT16_C( 17616) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi16(test_vec[i].src);
    easysimd__m256i a = easysimd_x_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_add_epi16(src, test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_add_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_x_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_add_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C( 4262),
      {  INT16_C( 17131),  INT16_C( 28575),  INT16_C( 26454),  INT16_C( 26097),  INT16_C( 13425),  INT16_C(  2916),  INT16_C( 18413), -INT16_C( 10307),
        -INT16_C( 22620),  INT16_C( 19835),  INT16_C( 24739),  INT16_C( 18854), -INT16_C(  6815), -INT16_C(  4094),  INT16_C( 31637),  INT16_C( 24951) },
      { -INT16_C( 29452),  INT16_C( 29154), -INT16_C( 24158),  INT16_C( 20005), -INT16_C( 21148),  INT16_C( 28846),  INT16_C( 22284), -INT16_C( 15662),
        -INT16_C(  9581), -INT16_C( 17953), -INT16_C( 18145), -INT16_C( 27296), -INT16_C( 29945), -INT16_C( 14925), -INT16_C( 22585),  INT16_C( 11545) },
      {  INT16_C(     0), -INT16_C(  7807),  INT16_C(  2296),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31762),  INT16_C(     0), -INT16_C( 25969),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 28776),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(12070),
      { -INT16_C( 11840), -INT16_C( 15372),  INT16_C( 32629),  INT16_C( 21795), -INT16_C( 20660), -INT16_C( 29830), -INT16_C( 24024), -INT16_C( 29778),
        -INT16_C( 10477), -INT16_C( 11462),  INT16_C(  1337),  INT16_C(  5552),  INT16_C(  5961),  INT16_C( 23833), -INT16_C( 10057), -INT16_C( 19595) },
      { -INT16_C( 22405), -INT16_C( 25821), -INT16_C( 24191), -INT16_C(  3861), -INT16_C( 24777),  INT16_C( 24020),  INT16_C( 13500), -INT16_C( 20346),
        -INT16_C(   128),  INT16_C(  5177),  INT16_C( 12366), -INT16_C(  8346), -INT16_C(  7771), -INT16_C(  8032), -INT16_C( 11287), -INT16_C( 24811) },
      {  INT16_C(     0),  INT16_C( 24343),  INT16_C(  8438),  INT16_C(     0),  INT16_C(     0), -INT16_C(  5810),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 10605), -INT16_C(  6285),  INT16_C( 13703), -INT16_C(  2794),  INT16_C(     0),  INT16_C( 15801),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(58512),
      { -INT16_C( 21753), -INT16_C( 12602), -INT16_C(  7088), -INT16_C( 15287),  INT16_C(  2851), -INT16_C(  3784), -INT16_C( 19644), -INT16_C(  3081),
         INT16_C( 27159),  INT16_C( 23080), -INT16_C( 12718), -INT16_C( 24689), -INT16_C( 17761),  INT16_C(  9682), -INT16_C( 30808), -INT16_C( 21236) },
      { -INT16_C( 30812),  INT16_C(  6629),  INT16_C( 21297), -INT16_C( 17253),  INT16_C( 32442),  INT16_C(  2260), -INT16_C(  2679), -INT16_C(  7556),
        -INT16_C( 10834), -INT16_C( 26189),  INT16_C(  2827), -INT16_C( 12197), -INT16_C( 13163),  INT16_C( 26769),  INT16_C( 21450),  INT16_C(  4191) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 30243),  INT16_C(     0),  INT16_C(     0), -INT16_C( 10637),
         INT16_C(     0),  INT16_C(     0), -INT16_C(  9891),  INT16_C(     0),  INT16_C(     0), -INT16_C( 29085), -INT16_C(  9358), -INT16_C( 17045) } },
    { UINT16_C(26409),
      {  INT16_C( 23989), -INT16_C(  3893), -INT16_C( 10213), -INT16_C( 24561),  INT16_C( 24975), -INT16_C( 14143), -INT16_C( 26678), -INT16_C( 32272),
        -INT16_C( 25660),  INT16_C( 31804), -INT16_C( 10324),  INT16_C( 19769), -INT16_C( 22018), -INT16_C( 31803), -INT16_C( 22436),  INT16_C( 31880) },
      { -INT16_C( 13937),  INT16_C( 31205), -INT16_C( 12695),  INT16_C(  9893),  INT16_C( 19693), -INT16_C( 32334), -INT16_C( 25972),  INT16_C( 30977),
         INT16_C( 26854), -INT16_C( 10518), -INT16_C( 16582), -INT16_C( 26001), -INT16_C( 23041),  INT16_C(  5435),  INT16_C( 20892), -INT16_C(  8904) },
      {  INT16_C( 10052),  INT16_C(     0),  INT16_C(     0), -INT16_C( 14668),  INT16_C(     0),  INT16_C( 19059),  INT16_C(     0),  INT16_C(     0),
         INT16_C(  1194),  INT16_C( 21286), -INT16_C( 26906),  INT16_C(     0),  INT16_C(     0), -INT16_C( 26368), -INT16_C(  1544),  INT16_C(     0) } },
    { UINT16_C(33734),
      { -INT16_C( 22559), -INT16_C(  8699), -INT16_C(  3315),  INT16_C( 30631),  INT16_C( 26186),  INT16_C(   646),  INT16_C( 32221), -INT16_C( 21902),
         INT16_C(  8809), -INT16_C(  9404), -INT16_C( 30818),  INT16_C(  9706), -INT16_C( 24086),  INT16_C( 10730), -INT16_C( 14587), -INT16_C( 12657) },
      { -INT16_C( 32121), -INT16_C( 14411), -INT16_C( 12619),  INT16_C( 32766), -INT16_C(  5886), -INT16_C( 28837), -INT16_C( 21424),  INT16_C(  1152),
         INT16_C( 18809), -INT16_C(  4130),  INT16_C(  7614), -INT16_C( 25916), -INT16_C( 21360),  INT16_C( 31629), -INT16_C( 21160), -INT16_C( 31719) },
      {  INT16_C(     0), -INT16_C( 23110), -INT16_C( 15934),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10797), -INT16_C( 20750),
         INT16_C( 27618), -INT16_C( 13534),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 21160) } },
    { UINT16_C(12974),
      { -INT16_C(  9110), -INT16_C( 27434), -INT16_C(  6575), -INT16_C( 17103),  INT16_C( 10872), -INT16_C( 28881), -INT16_C(  1144),  INT16_C( 11533),
        -INT16_C( 28247),  INT16_C( 19770),  INT16_C( 10309), -INT16_C( 18656), -INT16_C(  7754),  INT16_C(  3438),  INT16_C(   758), -INT16_C( 19035) },
      { -INT16_C( 10315), -INT16_C(   614),  INT16_C(  9805),  INT16_C( 29426), -INT16_C( 17671),  INT16_C( 32637),  INT16_C( 32714),  INT16_C( 16551),
        -INT16_C( 22924), -INT16_C( 21545), -INT16_C(  9595), -INT16_C( 15050),  INT16_C(  9710),  INT16_C( 18424),  INT16_C(   172), -INT16_C(  1119) },
      {  INT16_C(     0), -INT16_C( 28048),  INT16_C(  3230),  INT16_C( 12323),  INT16_C(     0),  INT16_C(  3756),  INT16_C(     0),  INT16_C( 28084),
         INT16_C(     0), -INT16_C(  1775),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1956),  INT16_C( 21862),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(10927),
      {  INT16_C(  8380),  INT16_C( 25660),  INT16_C(  3043),  INT16_C( 26369), -INT16_C(  8173),  INT16_C( 10605),  INT16_C( 14118), -INT16_C(  4505),
        -INT16_C( 11066),  INT16_C( 28798), -INT16_C( 24766), -INT16_C( 11889), -INT16_C( 14630),  INT16_C( 31282), -INT16_C(  8140), -INT16_C(  9178) },
      {  INT16_C( 15994), -INT16_C( 30317), -INT16_C( 15077),  INT16_C( 17333), -INT16_C(  7916), -INT16_C( 24502),  INT16_C(  5769), -INT16_C( 29765),
        -INT16_C( 24127), -INT16_C( 29183), -INT16_C( 28892),  INT16_C( 13466),  INT16_C( 13695), -INT16_C(  4328),  INT16_C( 11614), -INT16_C( 16283) },
      {  INT16_C( 24374), -INT16_C(  4657), -INT16_C( 12034), -INT16_C( 21834),  INT16_C(     0), -INT16_C( 13897),  INT16_C(     0),  INT16_C( 31266),
         INT16_C(     0), -INT16_C(   385),  INT16_C(     0),  INT16_C(  1577),  INT16_C(     0),  INT16_C( 26954),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(24898),
      {  INT16_C( 31303), -INT16_C( 28888),  INT16_C( 32475), -INT16_C(  9009), -INT16_C( 21039),  INT16_C(  2463), -INT16_C(   235), -INT16_C( 27830),
         INT16_C( 12119), -INT16_C( 21538), -INT16_C( 26278), -INT16_C( 21134), -INT16_C( 12642), -INT16_C(  7297), -INT16_C( 26495),  INT16_C( 29429) },
      {  INT16_C( 31766),  INT16_C( 20949),  INT16_C(  1225),  INT16_C(  8709),  INT16_C( 22985), -INT16_C(  8523),  INT16_C( 31728), -INT16_C( 31368),
        -INT16_C( 17643), -INT16_C(  2286), -INT16_C( 26761),  INT16_C(  1099),  INT16_C( 26190), -INT16_C( 28221),  INT16_C( 25127), -INT16_C(  9460) },
      {  INT16_C(     0), -INT16_C(  7939),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31493),  INT16_C(     0),
        -INT16_C(  5524),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 30018), -INT16_C(  1368),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_x_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_add_epi16(test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_add_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_x_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_mask_add_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(   426361587), -INT32_C(   443903963),  INT32_C(  1108332220),  INT32_C(  1919687830),  INT32_C(  1848164895), -INT32_C(  1877796598),  INT32_C(   483830142), -INT32_C(  2033377471) },
      UINT8_C(237),
      {  INT32_C(  1091358368),  INT32_C(   454097987),  INT32_C(   794389166),  INT32_C(   249974803),  INT32_C(   152267150),  INT32_C(   907080727), -INT32_C(  1733770061),  INT32_C(   220923521) },
      {  INT32_C(  1746020713),  INT32_C(  1017500003), -INT32_C(   803273524),  INT32_C(  1210396863),  INT32_C(   579828512), -INT32_C(   646644006),  INT32_C(  1430804319),  INT32_C(   926985775) },
      { -INT32_C(  1457588215), -INT32_C(   443903963), -INT32_C(     8884358),  INT32_C(  1460371666),  INT32_C(  1848164895),  INT32_C(   260436721), -INT32_C(   302965742),  INT32_C(  1147909296) } },
    { {  INT32_C(  1684912389),  INT32_C(  1461764767),  INT32_C(   531585245), -INT32_C(  1449055219),  INT32_C(  1704880266),  INT32_C(    70446890),  INT32_C(  2071901944), -INT32_C(   833618578) },
      UINT8_C(215),
      { -INT32_C(  1260280170),  INT32_C(    60784922),  INT32_C(  1789074221),  INT32_C(   506242865), -INT32_C(   265290717),  INT32_C(   773465711), -INT32_C(   249802610),  INT32_C(  1884100310) },
      { -INT32_C(  1123572585),  INT32_C(   950443465),  INT32_C(  1139858930), -INT32_C(  1387438223), -INT32_C(   995912866), -INT32_C(   783462299),  INT32_C(   568247403), -INT32_C(   175356258) },
      {  INT32_C(  1911114541),  INT32_C(  1011228387), -INT32_C(  1366034145), -INT32_C(  1449055219), -INT32_C(  1261203583),  INT32_C(    70446890),  INT32_C(   318444793),  INT32_C(  1708744052) } },
    { {  INT32_C(   683801390), -INT32_C(  1524037838),  INT32_C(   971777976), -INT32_C(   179262957), -INT32_C(   662180444),  INT32_C(  1746109353), -INT32_C(   120687010), -INT32_C(  1478304295) },
      UINT8_C(142),
      { -INT32_C(  1699645306),  INT32_C(  1703095819), -INT32_C(  1247556922), -INT32_C(   676606248), -INT32_C(  2057411672), -INT32_C(  2133453849),  INT32_C(  1277225381),  INT32_C(   895840639) },
      { -INT32_C(   241606211), -INT32_C(   714501803), -INT32_C(   395132250), -INT32_C(  1428661824),  INT32_C(  1298870179),  INT32_C(  1006598770), -INT32_C(  1667424466),  INT32_C(  1226468081) },
      {  INT32_C(   683801390),  INT32_C(   988594016), -INT32_C(  1642689172), -INT32_C(  2105268072), -INT32_C(   662180444),  INT32_C(  1746109353), -INT32_C(   120687010),  INT32_C(  2122308720) } },
    { { -INT32_C(  1854580385),  INT32_C(  1115024973), -INT32_C(   902732038),  INT32_C(  1105570825), -INT32_C(  2019220757),  INT32_C(  1567591273), -INT32_C(  1045989337),  INT32_C(   483693948) },
      UINT8_C( 16),
      { -INT32_C(  1622664288), -INT32_C(  1596764141),  INT32_C(  1531617719),  INT32_C(  2136124051), -INT32_C(   833907649),  INT32_C(   442458548), -INT32_C(  1966743074),  INT32_C(  1510510672) },
      {  INT32_C(  1995792121),  INT32_C(  1901624212), -INT32_C(   724913828),  INT32_C(   574753287), -INT32_C(   939006740),  INT32_C(  1662860686), -INT32_C(   798097367), -INT32_C(   819579665) },
      { -INT32_C(  1854580385),  INT32_C(  1115024973), -INT32_C(   902732038),  INT32_C(  1105570825), -INT32_C(  1772914389),  INT32_C(  1567591273), -INT32_C(  1045989337),  INT32_C(   483693948) } },
    { { -INT32_C(  2128119628),  INT32_C(   286644021), -INT32_C(   144034294), -INT32_C(   755119821),  INT32_C(   737304527), -INT32_C(   795896062), -INT32_C(  1306877446),  INT32_C(  2017207584) },
      UINT8_C( 39),
      { -INT32_C(   889153325),  INT32_C(  1946641334), -INT32_C(   394827187), -INT32_C(  1838628604), -INT32_C(  1114607536), -INT32_C(   988331075),  INT32_C(   505604917), -INT32_C(   158258489) },
      { -INT32_C(    19176166), -INT32_C(   739831540),  INT32_C(   812816619),  INT32_C(  1345615940),  INT32_C(   485266830), -INT32_C(  1880526201), -INT32_C(   347473235), -INT32_C(   631923023) },
      { -INT32_C(   908329491),  INT32_C(  1206809794),  INT32_C(   417989432), -INT32_C(   755119821),  INT32_C(   737304527),  INT32_C(  1426110020), -INT32_C(  1306877446),  INT32_C(  2017207584) } },
    { { -INT32_C(    35579584), -INT32_C(  1740792689),  INT32_C(  1827803830), -INT32_C(  1820729369), -INT32_C(  2037363614),  INT32_C(  1301089229),  INT32_C(  2117227785),  INT32_C(  1313930535) },
      UINT8_C( 29),
      {  INT32_C(   773440826), -INT32_C(    38988069), -INT32_C(   324201942),  INT32_C(   299194520), -INT32_C(   582542291),  INT32_C(   984927229),  INT32_C(   579334463),  INT32_C(   166931855) },
      { -INT32_C(  1529113807),  INT32_C(   521813891), -INT32_C(  1447212676),  INT32_C(  1970271572), -INT32_C(  1806503392),  INT32_C(  1956442032), -INT32_C(   878298291),  INT32_C(   934511007) },
      { -INT32_C(   755672981), -INT32_C(  1740792689), -INT32_C(  1771414618), -INT32_C(  2025501204),  INT32_C(  1905921613),  INT32_C(  1301089229),  INT32_C(  2117227785),  INT32_C(  1313930535) } },
    { {  INT32_C(  1988271902), -INT32_C(   928422277),  INT32_C(   751124781), -INT32_C(  1615080274), -INT32_C(  1457437893),  INT32_C(  1967921693), -INT32_C(   719093602),  INT32_C(  1590290629) },
      UINT8_C(184),
      { -INT32_C(   200700131), -INT32_C(  1501991200),  INT32_C(  1863812719), -INT32_C(  2110506296), -INT32_C(   281811570), -INT32_C(  1957556859), -INT32_C(   274670855),  INT32_C(   588775262) },
      {  INT32_C(    48549005), -INT32_C(  1189837525), -INT32_C(   246645075),  INT32_C(   710580162),  INT32_C(  1547365021), -INT32_C(  1190799946), -INT32_C(   758766312), -INT32_C(  1139192475) },
      {  INT32_C(  1988271902), -INT32_C(   928422277),  INT32_C(   751124781), -INT32_C(  1399926134),  INT32_C(  1265553451),  INT32_C(  1146610491), -INT32_C(   719093602), -INT32_C(   550417213) } },
    { { -INT32_C(  1607524027), -INT32_C(  1435945627), -INT32_C(  1834704498), -INT32_C(  2007313332), -INT32_C(  1883122106), -INT32_C(  1075124600),  INT32_C(  1649535565), -INT32_C(   868112955) },
      UINT8_C( 77),
      {  INT32_C(   441881310),  INT32_C(  1966218471),  INT32_C(  1844649392), -INT32_C(   321863841),  INT32_C(    26145081),  INT32_C(  2066060518), -INT32_C(  1310174374), -INT32_C(  1468908845) },
      {  INT32_C(   900269734),  INT32_C(  1516547149),  INT32_C(  1563014998),  INT32_C(    47500850), -INT32_C(  2090628180),  INT32_C(   430978017), -INT32_C(  2091637067),  INT32_C(  1915750227) },
      {  INT32_C(  1342151044), -INT32_C(  1435945627), -INT32_C(   887302906), -INT32_C(   274362991), -INT32_C(  1883122106), -INT32_C(  1075124600),  INT32_C(   893155855), -INT32_C(   868112955) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m256i a = easysimd_x_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_add_epi32(src, test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_add_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_x_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_add_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(101),
      { -INT32_C(   628298337),  INT32_C(   792924887), -INT32_C(  1672768632), -INT32_C(  1074053347), -INT32_C(  1064007895),  INT32_C(  1455536926), -INT32_C(  1219047864),  INT32_C(  1560140371) },
      {  INT32_C(  1960233098),  INT32_C(  1892059781), -INT32_C(  1373718700), -INT32_C(   352286831),  INT32_C(  1327327802),  INT32_C(   454025751),  INT32_C(  2093702863), -INT32_C(  1771120467) },
      {  INT32_C(  1331934761),  INT32_C(           0),  INT32_C(  1248479964),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1909562677),  INT32_C(   874654999),  INT32_C(           0) } },
    { UINT8_C( 88),
      { -INT32_C(   643482639),  INT32_C(  1022430836),  INT32_C(  1708754555), -INT32_C(  1303850341),  INT32_C(  1960699394),  INT32_C(   694659735),  INT32_C(   142013690),  INT32_C(   821320707) },
      { -INT32_C(   242563209),  INT32_C(  1657605631), -INT32_C(  1229937754),  INT32_C(  1696216758), -INT32_C(  1213277610), -INT32_C(   353040792),  INT32_C(   392990364),  INT32_C(  1905216013) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   392366417),  INT32_C(   747421784),  INT32_C(           0),  INT32_C(   535004054),  INT32_C(           0) } },
    { UINT8_C(212),
      { -INT32_C(  1571159102),  INT32_C(  2016749220), -INT32_C(  1981095890), -INT32_C(  1870763484), -INT32_C(   836082282), -INT32_C(   660363674), -INT32_C(  1962323400), -INT32_C(   143449086) },
      { -INT32_C(   765186109),  INT32_C(  1029534834),  INT32_C(     6988289),  INT32_C(  1621597364), -INT32_C(   789981577),  INT32_C(  1804329931),  INT32_C(  1186001858),  INT32_C(   883248384) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1974107601),  INT32_C(           0), -INT32_C(  1626063859),  INT32_C(           0), -INT32_C(   776321542),  INT32_C(   739799298) } },
    { UINT8_C(109),
      { -INT32_C(  1770917884), -INT32_C(   530889520), -INT32_C(   504237856), -INT32_C(  1597380183), -INT32_C(  1049867779), -INT32_C(   468326631),  INT32_C(   559503757),  INT32_C(   564004527) },
      {  INT32_C(  1698045997), -INT32_C(   820697417), -INT32_C(    38633790),  INT32_C(   881187224),  INT32_C(  1921955153),  INT32_C(   941315295),  INT32_C(   759745706), -INT32_C(  2126521994) },
      { -INT32_C(    72871887),  INT32_C(           0), -INT32_C(   542871646), -INT32_C(   716192959),  INT32_C(           0),  INT32_C(   472988664),  INT32_C(  1319249463),  INT32_C(           0) } },
    { UINT8_C(137),
      { -INT32_C(  1858814058), -INT32_C(  1042377463), -INT32_C(   887474565), -INT32_C(  2122945850),  INT32_C(  1667461733), -INT32_C(   205579704), -INT32_C(   975714158),  INT32_C(  1635367554) },
      { -INT32_C(  1057794330),  INT32_C(  1052274960), -INT32_C(    41436220),  INT32_C(   143840279), -INT32_C(  1906448242), -INT32_C(  1788921821),  INT32_C(    10365593),  INT32_C(    35226418) },
      {  INT32_C(  1378358908),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1979105571),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1670593972) } },
    { UINT8_C( 14),
      {  INT32_C(   955121944), -INT32_C(  1858149961),  INT32_C(   989283609), -INT32_C(  1140579072), -INT32_C(   447254547), -INT32_C(   109214975), -INT32_C(   782653276), -INT32_C(   848887013) },
      { -INT32_C(  1458708646), -INT32_C(    83883126),  INT32_C(  1994371146),  INT32_C(  1884018050), -INT32_C(  1630751337), -INT32_C(   235406036),  INT32_C(   301613089),  INT32_C(  1337287860) },
      {  INT32_C(           0), -INT32_C(  1942033087), -INT32_C(  1311312541),  INT32_C(   743438978),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 68),
      { -INT32_C(    17343115), -INT32_C(  1057582083),  INT32_C(  1925676355),  INT32_C(   513737506),  INT32_C(  1449532125),  INT32_C(   651379898),  INT32_C(   265490234),  INT32_C(  1273345831) },
      {  INT32_C(  1163851905), -INT32_C(  1572780827),  INT32_C(  1722687548),  INT32_C(  1425094090),  INT32_C(    16744716),  INT32_C(  1308498419), -INT32_C(   505877045), -INT32_C(   219979393) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   646603393),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   240386811),  INT32_C(           0) } },
    { UINT8_C(143),
      { -INT32_C(  1104597870), -INT32_C(   724590379),  INT32_C(  1429138415),  INT32_C(  1193975544),  INT32_C(  1467334920), -INT32_C(  1899396965),  INT32_C(  2086966300),  INT32_C(   710186623) },
      { -INT32_C(  1112373334),  INT32_C(   406415980), -INT32_C(  1125556542), -INT32_C(   990567746), -INT32_C(   543146280), -INT32_C(  1976533491), -INT32_C(   337560786),  INT32_C(   640868710) },
      {  INT32_C(  2077996092), -INT32_C(   318174399),  INT32_C(   303581873),  INT32_C(   203407798),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1351055333) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_x_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_add_epi32(test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_add_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_x_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_mask_add_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[4];
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C(  360909485362196701),  INT64_C( 7148018188301179842),  INT64_C( 1649121149472017725),  INT64_C( 7018633285360450459) },
      UINT8_C(126),
      { -INT64_C( 2069953424375465822),  INT64_C( 2649419973831635918),  INT64_C( 6631995759656128229),  INT64_C( 9043749468674069294) },
      {  INT64_C( 9127572592675566832),  INT64_C( 3139630957469384257), -INT64_C( 5603674882213165495),  INT64_C( 5601563532454726799) },
      {  INT64_C(  360909485362196701),  INT64_C( 5789050931301020175),  INT64_C( 1028320877442962734), -INT64_C( 3801431072580755523) } },
    { {  INT64_C(  756376000083734984),  INT64_C( 9070926205802174906),  INT64_C( 8919263102054599581),  INT64_C( 7783409138876853393) },
      UINT8_C(148),
      {  INT64_C(  382287398418300828),  INT64_C( 1115930529724925645), -INT64_C( 2730316823403657925),  INT64_C(  770900165413777792) },
      {  INT64_C( 6713096438988588943),  INT64_C( 6664457870741226758),  INT64_C(  234662618570907527), -INT64_C(  313102228808514227) },
      {  INT64_C(  756376000083734984),  INT64_C( 9070926205802174906), -INT64_C( 2495654204832750398),  INT64_C( 7783409138876853393) } },
    { {  INT64_C( 1483073417317823758),  INT64_C( 8795153962326963887), -INT64_C( 4149002502104727210),  INT64_C( 8377900925383693364) },
      UINT8_C( 87),
      {  INT64_C( 9038467157306396791), -INT64_C( 9115228638979228790),  INT64_C( 3197974720961571240), -INT64_C( 4399071158149707528) },
      {  INT64_C( 4732531473964703663), -INT64_C( 5052728689924432136), -INT64_C( 7671229005429790575),  INT64_C( 8865457101011322896) },
      { -INT64_C( 4675745442438451162),  INT64_C( 4278786744805890690), -INT64_C( 4473254284468219335),  INT64_C( 8377900925383693364) } },
    { { -INT64_C( 1491568665462998214), -INT64_C( 3104012194786369954), -INT64_C( 7320413738587291346),  INT64_C( 2384471346820870965) },
      UINT8_C(225),
      {  INT64_C( 8948791811858071094),  INT64_C( 5692550040163038202),  INT64_C(  830789416759788308),  INT64_C( 8983297292243767262) },
      {  INT64_C( 1554136023168485625),  INT64_C( 4643374375046125828),  INT64_C( 1207394444856260247),  INT64_C( 1262475476216602754) },
      { -INT64_C( 7943816238682994897), -INT64_C( 3104012194786369954), -INT64_C( 7320413738587291346),  INT64_C( 2384471346820870965) } },
    { { -INT64_C( 2613596307693393920),  INT64_C( 3708150816535274390),  INT64_C( 8656585681334612292), -INT64_C( 4071969454335091598) },
      UINT8_C(205),
      {  INT64_C( 4385428620712312621), -INT64_C( 6056401822535469397), -INT64_C( 5800804598671026971), -INT64_C( 3092870165874030482) },
      { -INT64_C( 2397873348439446309),  INT64_C( 5910269085379428366), -INT64_C( 1513047638103079921),  INT64_C(   34238311786967204) },
      {  INT64_C( 1987555272272866312),  INT64_C( 3708150816535274390), -INT64_C( 7313852236774106892), -INT64_C( 3058631854087063278) } },
    { {  INT64_C( 8817443827781467208), -INT64_C( 6180758275292564870), -INT64_C( 4102290530891378202),  INT64_C( 1720054592936513257) },
      UINT8_C(101),
      {  INT64_C( 8678261917080766892),  INT64_C( 6939458579392878936), -INT64_C( 1907623884063940351), -INT64_C( 1780765326956861806) },
      {  INT64_C( 3690861860797904287),  INT64_C( 7163125489808023225), -INT64_C( 7713699132847182549), -INT64_C( 5838526841458117177) },
      { -INT64_C( 6077620295830880437), -INT64_C( 6180758275292564870),  INT64_C( 8825421056798428716),  INT64_C( 1720054592936513257) } },
    { { -INT64_C( 4226189128237357280), -INT64_C( 8943401384532428378),  INT64_C(  205521322776642791),  INT64_C( 8979119138226217421) },
      UINT8_C(100),
      {  INT64_C(   74791247571826988),  INT64_C( 6811658127022208365), -INT64_C( 2788904720803475790),  INT64_C( 2092431171636941532) },
      { -INT64_C( 4721001194988252981), -INT64_C( 5047893754993698744), -INT64_C( 9019989623988394261),  INT64_C( 6590525094843788764) },
      { -INT64_C( 4226189128237357280), -INT64_C( 8943401384532428378),  INT64_C( 6637849728917681565),  INT64_C( 8979119138226217421) } },
    { { -INT64_C( 5055154983797961474),  INT64_C(     708989412318452),  INT64_C( 3248858564412742665), -INT64_C( 1488234339305196330) },
      UINT8_C( 97),
      { -INT64_C( 1196281769919788973), -INT64_C( 5928234014316609444), -INT64_C( 3324535828431189494), -INT64_C( 5570270810707974551) },
      { -INT64_C( 1528923573651634989), -INT64_C( 8239253630832908272),  INT64_C(  446928026184942104), -INT64_C( 7452034225800619308) },
      { -INT64_C( 2725205343571423962),  INT64_C(     708989412318452),  INT64_C( 3248858564412742665), -INT64_C( 1488234339305196330) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi64(test_vec[i].src);
    easysimd__m256i a = easysimd_x_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_add_epi64(src, test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_add_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_x_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_add_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(211),
      {  INT64_C( 5396742621358824656), -INT64_C( 5784837964580948014),  INT64_C(  745905599035155894),  INT64_C( 4432459820370420263) },
      { -INT64_C( 2375627515263830277),  INT64_C( 7298802839973142414), -INT64_C( 4614916525088736096), -INT64_C( 9037591478603085962) },
      {  INT64_C( 3021115106094994379),  INT64_C( 1513964875392194400),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(176),
      { -INT64_C( 3141262892297976873), -INT64_C( 2562411322036282317),  INT64_C( 3833520914396613866),  INT64_C( 2194941788560940840) },
      {  INT64_C(  901766347898667771),  INT64_C( 7724355933804345671), -INT64_C( 7464625842775737952), -INT64_C( 2080691623985024546) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(195),
      { -INT64_C( 6193321100419901660),  INT64_C( 8380227380500028405),  INT64_C( 6069563355255056514),  INT64_C( 7567844964561032724) },
      { -INT64_C( 2028429546419720554),  INT64_C( 5900094282335619633), -INT64_C( 3089803978563711891),  INT64_C(  365280098963395815) },
      { -INT64_C( 8221750646839622214), -INT64_C( 4166422410873903578),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 97),
      { -INT64_C( 6385959693145891606),  INT64_C( 5329109541483815658), -INT64_C( 2688882311189433873),  INT64_C( 9217817113255199635) },
      {  INT64_C( 1459731060611346300), -INT64_C( 3886780218327649651),  INT64_C( 7977890084846895501),  INT64_C( 4735556178160298813) },
      { -INT64_C( 4926228632534545306),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(134),
      {  INT64_C( 7860706032630943107),  INT64_C( 2484009629061100715),  INT64_C( 7509120629608935666), -INT64_C( 2386733518501817427) },
      {  INT64_C( 8447004062159754569), -INT64_C( 1567962289685365340), -INT64_C( 8386187512053637187), -INT64_C( 1215046594740082176) },
      {  INT64_C(                   0),  INT64_C(  916047339375735375), -INT64_C(  877066882444701521),  INT64_C(                   0) } },
    { UINT8_C(199),
      { -INT64_C( 2180470979029174228),  INT64_C( 6443804020022187526), -INT64_C( 6418035538474219843), -INT64_C( 5507798642899854187) },
      {  INT64_C( 2466999869382999871),  INT64_C(  996995031899642676), -INT64_C(   73336982042601123),  INT64_C( 6989615507504111215) },
      {  INT64_C(  286528890353825643),  INT64_C( 7440799051921830202), -INT64_C( 6491372520516820966),  INT64_C(                   0) } },
    { UINT8_C(  0),
      { -INT64_C( 8767112038748657730), -INT64_C( 7052635034042790208), -INT64_C( 6603890804231331725), -INT64_C( 3131137631410272858) },
      { -INT64_C( 1591887663143957093),  INT64_C( 8922603379407992151), -INT64_C(  206750837837817474),  INT64_C( 1015076583748941132) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(231),
      {  INT64_C( 6842434063541616318), -INT64_C( 6075517079359538540),  INT64_C( 1270586480222899316),  INT64_C( 7380126505793994543) },
      {  INT64_C( 8872434914943821287), -INT64_C( 6662427551341978491), -INT64_C( 3648454390316234397),  INT64_C( 6544313852933680065) },
      { -INT64_C( 2731875095224114011),  INT64_C( 5708799443008034585), -INT64_C( 2377867910093335081),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_x_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_add_epi64(test_vec[i].k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_add_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_x_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm256_mask_add_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    easysimd_float32 src[8];
    uint8_t k;
    easysimd_float32 a[8];
    easysimd_float32 b[8];
    easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -499.47), EASYSIMD_FLOAT32_C(   579.31), EASYSIMD_FLOAT32_C(    63.05), EASYSIMD_FLOAT32_C(  -294.71),
        EASYSIMD_FLOAT32_C(  -471.55), EASYSIMD_FLOAT32_C(   169.36), EASYSIMD_FLOAT32_C(   436.63), EASYSIMD_FLOAT32_C(  -845.42) },
      UINT8_C(122),
      { EASYSIMD_FLOAT32_C(  -547.89), EASYSIMD_FLOAT32_C(  -657.95), EASYSIMD_FLOAT32_C(  -776.90), EASYSIMD_FLOAT32_C(   933.20),
        EASYSIMD_FLOAT32_C(   447.85), EASYSIMD_FLOAT32_C(   768.19), EASYSIMD_FLOAT32_C(   -40.75), EASYSIMD_FLOAT32_C(   198.36) },
      { EASYSIMD_FLOAT32_C(  -586.75), EASYSIMD_FLOAT32_C(  -525.93), EASYSIMD_FLOAT32_C(  -542.96), EASYSIMD_FLOAT32_C(   937.88),
        EASYSIMD_FLOAT32_C(   -53.92), EASYSIMD_FLOAT32_C(  -555.69), EASYSIMD_FLOAT32_C(  -739.30), EASYSIMD_FLOAT32_C(  -182.04) },
      { EASYSIMD_FLOAT32_C(  -499.47), EASYSIMD_FLOAT32_C( -1183.88), EASYSIMD_FLOAT32_C(    63.05), EASYSIMD_FLOAT32_C(  1871.08),
        EASYSIMD_FLOAT32_C(   393.93), EASYSIMD_FLOAT32_C(   212.50), EASYSIMD_FLOAT32_C(  -780.05), EASYSIMD_FLOAT32_C(  -845.42) } },
    { { EASYSIMD_FLOAT32_C(   -48.77), EASYSIMD_FLOAT32_C(   416.75), EASYSIMD_FLOAT32_C(   469.56), EASYSIMD_FLOAT32_C(   850.92),
        EASYSIMD_FLOAT32_C(  -583.09), EASYSIMD_FLOAT32_C(   556.98), EASYSIMD_FLOAT32_C(  -648.55), EASYSIMD_FLOAT32_C(   996.22) },
      UINT8_C(239),
      { EASYSIMD_FLOAT32_C(    56.74), EASYSIMD_FLOAT32_C(  -475.33), EASYSIMD_FLOAT32_C(   789.39), EASYSIMD_FLOAT32_C(  -506.63),
        EASYSIMD_FLOAT32_C(  -320.75), EASYSIMD_FLOAT32_C(  -778.91), EASYSIMD_FLOAT32_C(   -54.52), EASYSIMD_FLOAT32_C(    21.30) },
      { EASYSIMD_FLOAT32_C(  -555.81), EASYSIMD_FLOAT32_C(  -121.32), EASYSIMD_FLOAT32_C(  -530.85), EASYSIMD_FLOAT32_C(  -787.62),
        EASYSIMD_FLOAT32_C(   837.93), EASYSIMD_FLOAT32_C(   667.51), EASYSIMD_FLOAT32_C(  -374.37), EASYSIMD_FLOAT32_C(  -688.00) },
      { EASYSIMD_FLOAT32_C(  -499.07), EASYSIMD_FLOAT32_C(  -596.65), EASYSIMD_FLOAT32_C(   258.54), EASYSIMD_FLOAT32_C( -1294.25),
        EASYSIMD_FLOAT32_C(  -583.09), EASYSIMD_FLOAT32_C(  -111.40), EASYSIMD_FLOAT32_C(  -428.89), EASYSIMD_FLOAT32_C(  -666.70) } },
    { { EASYSIMD_FLOAT32_C(  -875.45), EASYSIMD_FLOAT32_C(  -436.50), EASYSIMD_FLOAT32_C(   258.08), EASYSIMD_FLOAT32_C(  -431.14),
        EASYSIMD_FLOAT32_C(  -175.80), EASYSIMD_FLOAT32_C(  -923.95), EASYSIMD_FLOAT32_C(   520.10), EASYSIMD_FLOAT32_C(  -759.05) },
      UINT8_C(146),
      { EASYSIMD_FLOAT32_C(   371.02), EASYSIMD_FLOAT32_C(  -342.15), EASYSIMD_FLOAT32_C(   102.59), EASYSIMD_FLOAT32_C(   722.47),
        EASYSIMD_FLOAT32_C(  -345.93), EASYSIMD_FLOAT32_C(   722.62), EASYSIMD_FLOAT32_C(  -220.79), EASYSIMD_FLOAT32_C(   178.74) },
      { EASYSIMD_FLOAT32_C(   512.02), EASYSIMD_FLOAT32_C(   272.57), EASYSIMD_FLOAT32_C(   858.00), EASYSIMD_FLOAT32_C(   733.11),
        EASYSIMD_FLOAT32_C(  -781.95), EASYSIMD_FLOAT32_C(  -120.71), EASYSIMD_FLOAT32_C(  -822.71), EASYSIMD_FLOAT32_C(    96.74) },
      { EASYSIMD_FLOAT32_C(  -875.45), EASYSIMD_FLOAT32_C(   -69.58), EASYSIMD_FLOAT32_C(   258.08), EASYSIMD_FLOAT32_C(  -431.14),
        EASYSIMD_FLOAT32_C( -1127.88), EASYSIMD_FLOAT32_C(  -923.95), EASYSIMD_FLOAT32_C(   520.10), EASYSIMD_FLOAT32_C(   275.48) } },
    { { EASYSIMD_FLOAT32_C(   348.44), EASYSIMD_FLOAT32_C(  -610.33), EASYSIMD_FLOAT32_C(   -65.33), EASYSIMD_FLOAT32_C(    15.95),
        EASYSIMD_FLOAT32_C(    15.30), EASYSIMD_FLOAT32_C(   246.67), EASYSIMD_FLOAT32_C(   140.50), EASYSIMD_FLOAT32_C(   578.80) },
      UINT8_C(  2),
      { EASYSIMD_FLOAT32_C(   709.37), EASYSIMD_FLOAT32_C(  -597.00), EASYSIMD_FLOAT32_C(  -419.20), EASYSIMD_FLOAT32_C(   229.46),
        EASYSIMD_FLOAT32_C(  -356.05), EASYSIMD_FLOAT32_C(  -873.60), EASYSIMD_FLOAT32_C(  -399.52), EASYSIMD_FLOAT32_C(   301.81) },
      { EASYSIMD_FLOAT32_C(   228.99), EASYSIMD_FLOAT32_C(  -677.05), EASYSIMD_FLOAT32_C(   955.88), EASYSIMD_FLOAT32_C(   -48.39),
        EASYSIMD_FLOAT32_C(   102.16), EASYSIMD_FLOAT32_C(   134.62), EASYSIMD_FLOAT32_C(  -536.37), EASYSIMD_FLOAT32_C(  -625.27) },
      { EASYSIMD_FLOAT32_C(   348.44), EASYSIMD_FLOAT32_C( -1274.05), EASYSIMD_FLOAT32_C(   -65.33), EASYSIMD_FLOAT32_C(    15.95),
        EASYSIMD_FLOAT32_C(    15.30), EASYSIMD_FLOAT32_C(   246.67), EASYSIMD_FLOAT32_C(   140.50), EASYSIMD_FLOAT32_C(   578.80) } },
    { { EASYSIMD_FLOAT32_C(    -7.38), EASYSIMD_FLOAT32_C(  -803.27), EASYSIMD_FLOAT32_C(  -407.21), EASYSIMD_FLOAT32_C(   871.91),
        EASYSIMD_FLOAT32_C(  -625.98), EASYSIMD_FLOAT32_C(   689.53), EASYSIMD_FLOAT32_C(   220.36), EASYSIMD_FLOAT32_C(  -236.30) },
      UINT8_C(157),
      { EASYSIMD_FLOAT32_C(  -763.69), EASYSIMD_FLOAT32_C(   779.00), EASYSIMD_FLOAT32_C(   870.87), EASYSIMD_FLOAT32_C(   376.81),
        EASYSIMD_FLOAT32_C(   357.80), EASYSIMD_FLOAT32_C(  -624.38), EASYSIMD_FLOAT32_C(    86.18), EASYSIMD_FLOAT32_C(   760.80) },
      { EASYSIMD_FLOAT32_C(   -43.58), EASYSIMD_FLOAT32_C(  -684.35), EASYSIMD_FLOAT32_C(  -595.25), EASYSIMD_FLOAT32_C(    82.82),
        EASYSIMD_FLOAT32_C(   -83.87), EASYSIMD_FLOAT32_C(   706.56), EASYSIMD_FLOAT32_C(  -688.19), EASYSIMD_FLOAT32_C(   239.08) },
      { EASYSIMD_FLOAT32_C(  -807.27), EASYSIMD_FLOAT32_C(  -803.27), EASYSIMD_FLOAT32_C(   275.62), EASYSIMD_FLOAT32_C(   459.63),
        EASYSIMD_FLOAT32_C(   273.93), EASYSIMD_FLOAT32_C(   689.53), EASYSIMD_FLOAT32_C(   220.36), EASYSIMD_FLOAT32_C(   999.88) } },
    { { EASYSIMD_FLOAT32_C(   662.43), EASYSIMD_FLOAT32_C(   263.42), EASYSIMD_FLOAT32_C(  -658.76), EASYSIMD_FLOAT32_C(  -202.94),
        EASYSIMD_FLOAT32_C(   727.04), EASYSIMD_FLOAT32_C(  -284.03), EASYSIMD_FLOAT32_C(   789.67), EASYSIMD_FLOAT32_C(   923.78) },
      UINT8_C(202),
      { EASYSIMD_FLOAT32_C(   661.59), EASYSIMD_FLOAT32_C(  -702.20), EASYSIMD_FLOAT32_C(    -1.71), EASYSIMD_FLOAT32_C(  -118.06),
        EASYSIMD_FLOAT32_C(    61.50), EASYSIMD_FLOAT32_C(   622.48), EASYSIMD_FLOAT32_C(   118.25), EASYSIMD_FLOAT32_C(  -159.51) },
      { EASYSIMD_FLOAT32_C(   493.35), EASYSIMD_FLOAT32_C(  -504.93), EASYSIMD_FLOAT32_C(  -801.71), EASYSIMD_FLOAT32_C(   868.97),
        EASYSIMD_FLOAT32_C(   581.25), EASYSIMD_FLOAT32_C(   959.09), EASYSIMD_FLOAT32_C(  -174.61), EASYSIMD_FLOAT32_C(   896.89) },
      { EASYSIMD_FLOAT32_C(   662.43), EASYSIMD_FLOAT32_C( -1207.13), EASYSIMD_FLOAT32_C(  -658.76), EASYSIMD_FLOAT32_C(   750.91),
        EASYSIMD_FLOAT32_C(   727.04), EASYSIMD_FLOAT32_C(  -284.03), EASYSIMD_FLOAT32_C(   -56.36), EASYSIMD_FLOAT32_C(   737.38) } },
    { { EASYSIMD_FLOAT32_C(  -636.15), EASYSIMD_FLOAT32_C(   908.21), EASYSIMD_FLOAT32_C(  -186.98), EASYSIMD_FLOAT32_C(  -929.60),
        EASYSIMD_FLOAT32_C(  -779.98), EASYSIMD_FLOAT32_C(  -947.90), EASYSIMD_FLOAT32_C(   732.84), EASYSIMD_FLOAT32_C(   483.44) },
      UINT8_C(231),
      { EASYSIMD_FLOAT32_C(  -470.10), EASYSIMD_FLOAT32_C(   210.49), EASYSIMD_FLOAT32_C(   109.31), EASYSIMD_FLOAT32_C(  -680.43),
        EASYSIMD_FLOAT32_C(   134.26), EASYSIMD_FLOAT32_C(  -581.93), EASYSIMD_FLOAT32_C(   981.16), EASYSIMD_FLOAT32_C(   432.06) },
      { EASYSIMD_FLOAT32_C(   416.36), EASYSIMD_FLOAT32_C(  -136.90), EASYSIMD_FLOAT32_C(  -506.44), EASYSIMD_FLOAT32_C(    38.84),
        EASYSIMD_FLOAT32_C(   981.35), EASYSIMD_FLOAT32_C(   334.06), EASYSIMD_FLOAT32_C(  -467.81), EASYSIMD_FLOAT32_C(  -523.58) },
      { EASYSIMD_FLOAT32_C(   -53.74), EASYSIMD_FLOAT32_C(    73.59), EASYSIMD_FLOAT32_C(  -397.13), EASYSIMD_FLOAT32_C(  -929.60),
        EASYSIMD_FLOAT32_C(  -779.98), EASYSIMD_FLOAT32_C(  -247.87), EASYSIMD_FLOAT32_C(   513.35), EASYSIMD_FLOAT32_C(   -91.52) } },
    { { EASYSIMD_FLOAT32_C(   532.35), EASYSIMD_FLOAT32_C(  -598.84), EASYSIMD_FLOAT32_C(  -942.33), EASYSIMD_FLOAT32_C(   491.44),
        EASYSIMD_FLOAT32_C(   226.55), EASYSIMD_FLOAT32_C(   954.56), EASYSIMD_FLOAT32_C(   855.29), EASYSIMD_FLOAT32_C(   134.76) },
      UINT8_C( 66),
      { EASYSIMD_FLOAT32_C(   925.70), EASYSIMD_FLOAT32_C(   354.79), EASYSIMD_FLOAT32_C(  -180.31), EASYSIMD_FLOAT32_C(   658.54),
        EASYSIMD_FLOAT32_C(  -161.77), EASYSIMD_FLOAT32_C(   213.02), EASYSIMD_FLOAT32_C(  -811.57), EASYSIMD_FLOAT32_C(  -951.28) },
      { EASYSIMD_FLOAT32_C(  -677.67), EASYSIMD_FLOAT32_C(  -492.00), EASYSIMD_FLOAT32_C(   182.98), EASYSIMD_FLOAT32_C(  -259.60),
        EASYSIMD_FLOAT32_C(  -510.84), EASYSIMD_FLOAT32_C(  -384.95), EASYSIMD_FLOAT32_C(  -843.24), EASYSIMD_FLOAT32_C(   352.26) },
      { EASYSIMD_FLOAT32_C(   532.35), EASYSIMD_FLOAT32_C(  -137.21), EASYSIMD_FLOAT32_C(  -942.33), EASYSIMD_FLOAT32_C(   491.44),
        EASYSIMD_FLOAT32_C(   226.55), EASYSIMD_FLOAT32_C(   954.56), EASYSIMD_FLOAT32_C( -1654.81), EASYSIMD_FLOAT32_C(   134.76) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_add_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_add_ps");
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
    easysimd__m256 r = easysimd_mm256_mask_add_ps(src, k, a, b);

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
test_easysimd_mm256_maskz_add_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    uint8_t k;
    easysimd_float32 a[8];
    easysimd_float32 b[8];
    easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(105),
      { EASYSIMD_FLOAT32_C(   -38.11), EASYSIMD_FLOAT32_C(   611.38), EASYSIMD_FLOAT32_C(   918.98), EASYSIMD_FLOAT32_C(   261.28),
        EASYSIMD_FLOAT32_C(  -422.51), EASYSIMD_FLOAT32_C(   299.95), EASYSIMD_FLOAT32_C(  -467.35), EASYSIMD_FLOAT32_C(     8.48) },
      { EASYSIMD_FLOAT32_C(   849.99), EASYSIMD_FLOAT32_C(  -825.80), EASYSIMD_FLOAT32_C(   513.27), EASYSIMD_FLOAT32_C(    38.79),
        EASYSIMD_FLOAT32_C(   170.09), EASYSIMD_FLOAT32_C(   573.35), EASYSIMD_FLOAT32_C(  -777.75), EASYSIMD_FLOAT32_C(  -382.16) },
      { EASYSIMD_FLOAT32_C(   811.88), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   300.07),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   873.30), EASYSIMD_FLOAT32_C( -1245.10), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 53),
      { EASYSIMD_FLOAT32_C(  -542.15), EASYSIMD_FLOAT32_C(  -266.66), EASYSIMD_FLOAT32_C(  -213.55), EASYSIMD_FLOAT32_C(  -716.82),
        EASYSIMD_FLOAT32_C(   696.54), EASYSIMD_FLOAT32_C(   526.54), EASYSIMD_FLOAT32_C(   496.52), EASYSIMD_FLOAT32_C(   735.34) },
      { EASYSIMD_FLOAT32_C(    78.93), EASYSIMD_FLOAT32_C(  -499.57), EASYSIMD_FLOAT32_C(  -756.22), EASYSIMD_FLOAT32_C(   221.93),
        EASYSIMD_FLOAT32_C(  -163.08), EASYSIMD_FLOAT32_C(  -814.89), EASYSIMD_FLOAT32_C(  -816.18), EASYSIMD_FLOAT32_C(  -551.70) },
      { EASYSIMD_FLOAT32_C(  -463.22), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -969.77), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   533.46), EASYSIMD_FLOAT32_C(  -288.35), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(163),
      { EASYSIMD_FLOAT32_C(   445.10), EASYSIMD_FLOAT32_C(    25.79), EASYSIMD_FLOAT32_C(   404.04), EASYSIMD_FLOAT32_C(   977.75),
        EASYSIMD_FLOAT32_C(  -965.74), EASYSIMD_FLOAT32_C(   254.03), EASYSIMD_FLOAT32_C(  -848.05), EASYSIMD_FLOAT32_C(   547.53) },
      { EASYSIMD_FLOAT32_C(  -707.18), EASYSIMD_FLOAT32_C(   322.04), EASYSIMD_FLOAT32_C(   120.88), EASYSIMD_FLOAT32_C(  -484.93),
        EASYSIMD_FLOAT32_C(   939.88), EASYSIMD_FLOAT32_C(  -484.26), EASYSIMD_FLOAT32_C(   -27.07), EASYSIMD_FLOAT32_C(  -326.78) },
      { EASYSIMD_FLOAT32_C(  -262.08), EASYSIMD_FLOAT32_C(   347.83), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -230.23), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   220.75) } },
    { UINT8_C(240),
      { EASYSIMD_FLOAT32_C(   256.11), EASYSIMD_FLOAT32_C(  -630.25), EASYSIMD_FLOAT32_C(  -171.26), EASYSIMD_FLOAT32_C(  -247.37),
        EASYSIMD_FLOAT32_C(  -894.91), EASYSIMD_FLOAT32_C(   907.67), EASYSIMD_FLOAT32_C(   253.06), EASYSIMD_FLOAT32_C(  -651.13) },
      { EASYSIMD_FLOAT32_C(   129.59), EASYSIMD_FLOAT32_C(  -910.02), EASYSIMD_FLOAT32_C(  -466.01), EASYSIMD_FLOAT32_C(   313.41),
        EASYSIMD_FLOAT32_C(  -461.72), EASYSIMD_FLOAT32_C(  -361.92), EASYSIMD_FLOAT32_C(  -241.49), EASYSIMD_FLOAT32_C(   564.07) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C( -1356.63), EASYSIMD_FLOAT32_C(   545.75), EASYSIMD_FLOAT32_C(    11.57), EASYSIMD_FLOAT32_C(   -87.06) } },
    { UINT8_C(190),
      { EASYSIMD_FLOAT32_C(  -263.74), EASYSIMD_FLOAT32_C(   598.33), EASYSIMD_FLOAT32_C(   296.15), EASYSIMD_FLOAT32_C(  -111.80),
        EASYSIMD_FLOAT32_C(   145.86), EASYSIMD_FLOAT32_C(   588.97), EASYSIMD_FLOAT32_C(  -789.76), EASYSIMD_FLOAT32_C(  -733.26) },
      { EASYSIMD_FLOAT32_C(  -895.96), EASYSIMD_FLOAT32_C(  -849.88), EASYSIMD_FLOAT32_C(  -217.51), EASYSIMD_FLOAT32_C(    76.97),
        EASYSIMD_FLOAT32_C(  -176.66), EASYSIMD_FLOAT32_C(  -915.32), EASYSIMD_FLOAT32_C(  -666.92), EASYSIMD_FLOAT32_C(   193.09) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -251.55), EASYSIMD_FLOAT32_C(    78.64), EASYSIMD_FLOAT32_C(   -34.83),
        EASYSIMD_FLOAT32_C(   -30.80), EASYSIMD_FLOAT32_C(  -326.35), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -540.17) } },
    { UINT8_C( 53),
      { EASYSIMD_FLOAT32_C(    85.71), EASYSIMD_FLOAT32_C(   298.18), EASYSIMD_FLOAT32_C(  -178.91), EASYSIMD_FLOAT32_C(  -661.23),
        EASYSIMD_FLOAT32_C(   647.06), EASYSIMD_FLOAT32_C(   950.68), EASYSIMD_FLOAT32_C(  -571.24), EASYSIMD_FLOAT32_C(  -818.96) },
      { EASYSIMD_FLOAT32_C(   264.09), EASYSIMD_FLOAT32_C(   -32.96), EASYSIMD_FLOAT32_C(  -180.88), EASYSIMD_FLOAT32_C(  -977.40),
        EASYSIMD_FLOAT32_C(  -468.89), EASYSIMD_FLOAT32_C(  -138.76), EASYSIMD_FLOAT32_C(  -241.14), EASYSIMD_FLOAT32_C(  -870.56) },
      { EASYSIMD_FLOAT32_C(   349.80), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -359.79), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   178.17), EASYSIMD_FLOAT32_C(   811.92), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(127),
      { EASYSIMD_FLOAT32_C(   647.06), EASYSIMD_FLOAT32_C(   275.30), EASYSIMD_FLOAT32_C(   746.36), EASYSIMD_FLOAT32_C(   857.30),
        EASYSIMD_FLOAT32_C(   542.04), EASYSIMD_FLOAT32_C(   850.40), EASYSIMD_FLOAT32_C(  -992.58), EASYSIMD_FLOAT32_C(  -675.47) },
      { EASYSIMD_FLOAT32_C(   -72.63), EASYSIMD_FLOAT32_C(  -169.24), EASYSIMD_FLOAT32_C(  -590.79), EASYSIMD_FLOAT32_C(   260.45),
        EASYSIMD_FLOAT32_C(  -976.15), EASYSIMD_FLOAT32_C(   322.63), EASYSIMD_FLOAT32_C(  -653.84), EASYSIMD_FLOAT32_C(   322.03) },
      { EASYSIMD_FLOAT32_C(   574.43), EASYSIMD_FLOAT32_C(   106.06), EASYSIMD_FLOAT32_C(   155.57), EASYSIMD_FLOAT32_C(  1117.75),
        EASYSIMD_FLOAT32_C(  -434.11), EASYSIMD_FLOAT32_C(  1173.03), EASYSIMD_FLOAT32_C( -1646.42), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(204),
      { EASYSIMD_FLOAT32_C(  -315.07), EASYSIMD_FLOAT32_C(   -30.91), EASYSIMD_FLOAT32_C(  -905.60), EASYSIMD_FLOAT32_C(   113.69),
        EASYSIMD_FLOAT32_C(   150.14), EASYSIMD_FLOAT32_C(   358.49), EASYSIMD_FLOAT32_C(  -919.27), EASYSIMD_FLOAT32_C(   969.26) },
      { EASYSIMD_FLOAT32_C(   381.09), EASYSIMD_FLOAT32_C(  -388.17), EASYSIMD_FLOAT32_C(  -169.50), EASYSIMD_FLOAT32_C(  -860.05),
        EASYSIMD_FLOAT32_C(  -258.73), EASYSIMD_FLOAT32_C(   -12.11), EASYSIMD_FLOAT32_C(   787.01), EASYSIMD_FLOAT32_C(  -983.43) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( -1075.10), EASYSIMD_FLOAT32_C(  -746.36),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -132.26), EASYSIMD_FLOAT32_C(   -14.17) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_add_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_add_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_add_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_add_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    easysimd_float64 src[4];
    uint8_t k;
    easysimd_float64 a[4];
    easysimd_float64 b[4];
    easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -720.34), EASYSIMD_FLOAT64_C(  -107.68), EASYSIMD_FLOAT64_C(   183.51), EASYSIMD_FLOAT64_C(  -654.87) },
      UINT8_C( 68),
      { EASYSIMD_FLOAT64_C(  -106.41), EASYSIMD_FLOAT64_C(  -445.92), EASYSIMD_FLOAT64_C(  -591.21), EASYSIMD_FLOAT64_C(  -311.88) },
      { EASYSIMD_FLOAT64_C(  -252.42), EASYSIMD_FLOAT64_C(   984.18), EASYSIMD_FLOAT64_C(  -244.24), EASYSIMD_FLOAT64_C(   574.23) },
      { EASYSIMD_FLOAT64_C(  -720.34), EASYSIMD_FLOAT64_C(  -107.68), EASYSIMD_FLOAT64_C(  -835.45), EASYSIMD_FLOAT64_C(  -654.87) } },
    { { EASYSIMD_FLOAT64_C(  -977.03), EASYSIMD_FLOAT64_C(   844.60), EASYSIMD_FLOAT64_C(   582.86), EASYSIMD_FLOAT64_C(  -943.17) },
      UINT8_C( 80),
      { EASYSIMD_FLOAT64_C(   292.14), EASYSIMD_FLOAT64_C(   107.65), EASYSIMD_FLOAT64_C(  -372.01), EASYSIMD_FLOAT64_C(   457.90) },
      { EASYSIMD_FLOAT64_C(   956.33), EASYSIMD_FLOAT64_C(  -215.79), EASYSIMD_FLOAT64_C(  -853.43), EASYSIMD_FLOAT64_C(   355.81) },
      { EASYSIMD_FLOAT64_C(  -977.03), EASYSIMD_FLOAT64_C(   844.60), EASYSIMD_FLOAT64_C(   582.86), EASYSIMD_FLOAT64_C(  -943.17) } },
    { { EASYSIMD_FLOAT64_C(   500.64), EASYSIMD_FLOAT64_C(   834.20), EASYSIMD_FLOAT64_C(  -623.91), EASYSIMD_FLOAT64_C(   -23.24) },
      UINT8_C( 64),
      { EASYSIMD_FLOAT64_C(  -344.25), EASYSIMD_FLOAT64_C(   869.08), EASYSIMD_FLOAT64_C(  -754.23), EASYSIMD_FLOAT64_C(     0.88) },
      { EASYSIMD_FLOAT64_C(  -220.89), EASYSIMD_FLOAT64_C(   139.35), EASYSIMD_FLOAT64_C(   554.96), EASYSIMD_FLOAT64_C(   187.90) },
      { EASYSIMD_FLOAT64_C(   500.64), EASYSIMD_FLOAT64_C(   834.20), EASYSIMD_FLOAT64_C(  -623.91), EASYSIMD_FLOAT64_C(   -23.24) } },
    { { EASYSIMD_FLOAT64_C(   827.47), EASYSIMD_FLOAT64_C(  -697.46), EASYSIMD_FLOAT64_C(   172.08), EASYSIMD_FLOAT64_C(  -416.77) },
      UINT8_C(215),
      { EASYSIMD_FLOAT64_C(   195.04), EASYSIMD_FLOAT64_C(  -572.17), EASYSIMD_FLOAT64_C(   459.62), EASYSIMD_FLOAT64_C(   251.87) },
      { EASYSIMD_FLOAT64_C(   115.11), EASYSIMD_FLOAT64_C(  -248.23), EASYSIMD_FLOAT64_C(  -640.49), EASYSIMD_FLOAT64_C(   743.09) },
      { EASYSIMD_FLOAT64_C(   310.15), EASYSIMD_FLOAT64_C(  -820.40), EASYSIMD_FLOAT64_C(  -180.87), EASYSIMD_FLOAT64_C(  -416.77) } },
    { { EASYSIMD_FLOAT64_C(  -790.33), EASYSIMD_FLOAT64_C(  -684.16), EASYSIMD_FLOAT64_C(  -472.70), EASYSIMD_FLOAT64_C(  -643.76) },
      UINT8_C( 62),
      { EASYSIMD_FLOAT64_C(  -972.06), EASYSIMD_FLOAT64_C(  -809.56), EASYSIMD_FLOAT64_C(  -952.26), EASYSIMD_FLOAT64_C(     4.70) },
      { EASYSIMD_FLOAT64_C(   252.70), EASYSIMD_FLOAT64_C(  -296.51), EASYSIMD_FLOAT64_C(  -126.22), EASYSIMD_FLOAT64_C(   498.47) },
      { EASYSIMD_FLOAT64_C(  -790.33), EASYSIMD_FLOAT64_C( -1106.07), EASYSIMD_FLOAT64_C( -1078.48), EASYSIMD_FLOAT64_C(   503.17) } },
    { { EASYSIMD_FLOAT64_C(   704.37), EASYSIMD_FLOAT64_C(   652.89), EASYSIMD_FLOAT64_C(  -362.18), EASYSIMD_FLOAT64_C(   259.33) },
      UINT8_C( 72),
      { EASYSIMD_FLOAT64_C(  -534.70), EASYSIMD_FLOAT64_C(   561.87), EASYSIMD_FLOAT64_C(  -987.13), EASYSIMD_FLOAT64_C(    48.53) },
      { EASYSIMD_FLOAT64_C(   438.64), EASYSIMD_FLOAT64_C(   207.91), EASYSIMD_FLOAT64_C(   476.35), EASYSIMD_FLOAT64_C(  -101.74) },
      { EASYSIMD_FLOAT64_C(   704.37), EASYSIMD_FLOAT64_C(   652.89), EASYSIMD_FLOAT64_C(  -362.18), EASYSIMD_FLOAT64_C(   -53.21) } },
    { { EASYSIMD_FLOAT64_C(  -540.22), EASYSIMD_FLOAT64_C(  -408.54), EASYSIMD_FLOAT64_C(   650.02), EASYSIMD_FLOAT64_C(  -180.71) },
      UINT8_C(143),
      { EASYSIMD_FLOAT64_C(   859.69), EASYSIMD_FLOAT64_C(   135.14), EASYSIMD_FLOAT64_C(  -138.14), EASYSIMD_FLOAT64_C(  -784.07) },
      { EASYSIMD_FLOAT64_C(  -193.22), EASYSIMD_FLOAT64_C(  -110.21), EASYSIMD_FLOAT64_C(  -593.64), EASYSIMD_FLOAT64_C(  -145.48) },
      { EASYSIMD_FLOAT64_C(   666.47), EASYSIMD_FLOAT64_C(    24.93), EASYSIMD_FLOAT64_C(  -731.78), EASYSIMD_FLOAT64_C(  -929.55) } },
    { { EASYSIMD_FLOAT64_C(   894.49), EASYSIMD_FLOAT64_C(   659.06), EASYSIMD_FLOAT64_C(   558.01), EASYSIMD_FLOAT64_C(  -231.72) },
      UINT8_C( 39),
      { EASYSIMD_FLOAT64_C(   262.38), EASYSIMD_FLOAT64_C(  -578.83), EASYSIMD_FLOAT64_C(   795.35), EASYSIMD_FLOAT64_C(  -478.29) },
      { EASYSIMD_FLOAT64_C(   261.96), EASYSIMD_FLOAT64_C(  -739.35), EASYSIMD_FLOAT64_C(  -916.42), EASYSIMD_FLOAT64_C(   274.83) },
      { EASYSIMD_FLOAT64_C(   524.34), EASYSIMD_FLOAT64_C( -1318.18), EASYSIMD_FLOAT64_C(  -121.07), EASYSIMD_FLOAT64_C(  -231.72) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_add_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_add_pd");
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
    easysimd__m256d r = easysimd_mm256_mask_add_pd(src, k, a, b);

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
test_easysimd_mm256_maskz_add_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  struct {
    uint8_t k;
    easysimd_float64 a[4];
    easysimd_float64 b[4];
    easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(137),
      { EASYSIMD_FLOAT64_C(  -184.21), EASYSIMD_FLOAT64_C(   474.43), EASYSIMD_FLOAT64_C(    52.81), EASYSIMD_FLOAT64_C(   416.02) },
      { EASYSIMD_FLOAT64_C(   614.99), EASYSIMD_FLOAT64_C(  -888.33), EASYSIMD_FLOAT64_C(   926.12), EASYSIMD_FLOAT64_C(  -222.13) },
      { EASYSIMD_FLOAT64_C(   430.78), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   193.89) } },
    { UINT8_C(189),
      { EASYSIMD_FLOAT64_C(   942.51), EASYSIMD_FLOAT64_C(   748.32), EASYSIMD_FLOAT64_C(  -217.96), EASYSIMD_FLOAT64_C(   429.85) },
      { EASYSIMD_FLOAT64_C(  -257.08), EASYSIMD_FLOAT64_C(   580.14), EASYSIMD_FLOAT64_C(  -583.68), EASYSIMD_FLOAT64_C(  -483.47) },
      { EASYSIMD_FLOAT64_C(   685.43), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -801.64), EASYSIMD_FLOAT64_C(   -53.62) } },
    { UINT8_C(139),
      { EASYSIMD_FLOAT64_C(   628.44), EASYSIMD_FLOAT64_C(  -776.33), EASYSIMD_FLOAT64_C(   138.18), EASYSIMD_FLOAT64_C(  -957.63) },
      { EASYSIMD_FLOAT64_C(  -398.42), EASYSIMD_FLOAT64_C(   399.51), EASYSIMD_FLOAT64_C(  -809.81), EASYSIMD_FLOAT64_C(   954.91) },
      { EASYSIMD_FLOAT64_C(   230.02), EASYSIMD_FLOAT64_C(  -376.82), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -2.72) } },
    { UINT8_C(  7),
      { EASYSIMD_FLOAT64_C(     6.58), EASYSIMD_FLOAT64_C(  -140.10), EASYSIMD_FLOAT64_C(   100.83), EASYSIMD_FLOAT64_C(  -511.87) },
      { EASYSIMD_FLOAT64_C(   675.70), EASYSIMD_FLOAT64_C(  -424.73), EASYSIMD_FLOAT64_C(   540.94), EASYSIMD_FLOAT64_C(    91.72) },
      { EASYSIMD_FLOAT64_C(   682.28), EASYSIMD_FLOAT64_C(  -564.83), EASYSIMD_FLOAT64_C(   641.77), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 25),
      { EASYSIMD_FLOAT64_C(   652.61), EASYSIMD_FLOAT64_C(    17.84), EASYSIMD_FLOAT64_C(   -31.87), EASYSIMD_FLOAT64_C(   -46.14) },
      { EASYSIMD_FLOAT64_C(   -39.65), EASYSIMD_FLOAT64_C(  -283.56), EASYSIMD_FLOAT64_C(   735.90), EASYSIMD_FLOAT64_C(  -609.80) },
      { EASYSIMD_FLOAT64_C(   612.96), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -655.94) } },
    { UINT8_C( 99),
      { EASYSIMD_FLOAT64_C(   316.04), EASYSIMD_FLOAT64_C(  -193.48), EASYSIMD_FLOAT64_C(   975.89), EASYSIMD_FLOAT64_C(    78.94) },
      { EASYSIMD_FLOAT64_C(  -565.04), EASYSIMD_FLOAT64_C(  -800.44), EASYSIMD_FLOAT64_C(  -782.88), EASYSIMD_FLOAT64_C(  -522.67) },
      { EASYSIMD_FLOAT64_C(  -249.00), EASYSIMD_FLOAT64_C(  -993.92), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(211),
      { EASYSIMD_FLOAT64_C(   616.62), EASYSIMD_FLOAT64_C(  -332.48), EASYSIMD_FLOAT64_C(  -243.94), EASYSIMD_FLOAT64_C(  -706.19) },
      { EASYSIMD_FLOAT64_C(   674.10), EASYSIMD_FLOAT64_C(   615.97), EASYSIMD_FLOAT64_C(   394.64), EASYSIMD_FLOAT64_C(  -837.77) },
      { EASYSIMD_FLOAT64_C(  1290.72), EASYSIMD_FLOAT64_C(   283.49), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(133),
      { EASYSIMD_FLOAT64_C(   969.90), EASYSIMD_FLOAT64_C(   703.17), EASYSIMD_FLOAT64_C(  -616.62), EASYSIMD_FLOAT64_C(  -839.84) },
      { EASYSIMD_FLOAT64_C(   355.77), EASYSIMD_FLOAT64_C(   401.22), EASYSIMD_FLOAT64_C(   128.29), EASYSIMD_FLOAT64_C(  -690.37) },
      { EASYSIMD_FLOAT64_C(  1325.67), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -488.33), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_add_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_add_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_add_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_add_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    const int8_t a[64];
    const int8_t b[64];
    const int8_t r[64];
  } test_vec[] = {
    { {  INT8_C(  16), -INT8_C(  23), -INT8_C(  42), -INT8_C( 110), -INT8_C(  66), -INT8_C(  42),  INT8_C(  81), -INT8_C( 107),
        -INT8_C(  64), -INT8_C(  89),  INT8_C( 109),  INT8_C(  69),  INT8_C(  32), -INT8_C(  26), -INT8_C(  52), -INT8_C( 121),
        -INT8_C(  91), -INT8_C( 119),  INT8_C(  20), -INT8_C( 127),  INT8_C(  41), -INT8_C(  19),  INT8_C(  18),  INT8_C(   3),
         INT8_C( 103),  INT8_C( 122),  INT8_C(  63), -INT8_C(  75),  INT8_C(  40), -INT8_C(  38), -INT8_C(  44),  INT8_C(  56),
        -INT8_C(  61), -INT8_C(  86), -INT8_C(  54), -INT8_C( 127),      INT8_MIN,  INT8_C(  27),  INT8_C(  22),  INT8_C(  64),
        -INT8_C(  61), -INT8_C( 125), -INT8_C( 122), -INT8_C(  29),  INT8_C( 105),  INT8_C(  82),  INT8_C( 106),  INT8_C(  14),
        -INT8_C(  37),  INT8_C( 126), -INT8_C( 113),  INT8_C(   5),  INT8_C( 107), -INT8_C(  95),  INT8_C(   8), -INT8_C(  46),
         INT8_C(  27),  INT8_C(  71), -INT8_C( 120),  INT8_C(  67),  INT8_C(  33),  INT8_C(  92),  INT8_C( 124), -INT8_C(  28) },
      {  INT8_C(   6),  INT8_C(  70),  INT8_C( 101), -INT8_C( 122),  INT8_C(  98),  INT8_C( 123), -INT8_C(  58),  INT8_C(  37),
        -INT8_C(   1),  INT8_C(  76),  INT8_C(   8),  INT8_C( 104), -INT8_C(  98),  INT8_C( 114),  INT8_C( 119),  INT8_C( 122),
        -INT8_C(  16),  INT8_C(   6),      INT8_MAX,  INT8_C(  91), -INT8_C(  88), -INT8_C( 121),  INT8_C(  45), -INT8_C(  61),
        -INT8_C(  50), -INT8_C(  75),  INT8_C(   7), -INT8_C(  17),  INT8_C(  17), -INT8_C( 125), -INT8_C(  45),  INT8_C(  23),
        -INT8_C(  55),  INT8_C(  56), -INT8_C(  99),  INT8_C(  43), -INT8_C(  77),  INT8_C( 100),  INT8_C(  80), -INT8_C(  78),
        -INT8_C(  80),  INT8_C(  88),  INT8_C(  27),  INT8_C(  79), -INT8_C(  54), -INT8_C( 110), -INT8_C(  55), -INT8_C(  70),
        -INT8_C( 104),  INT8_C(  72),  INT8_C(  21),  INT8_C(  64), -INT8_C(  49),  INT8_C(  67),  INT8_C(   4), -INT8_C(  99),
        -INT8_C(   8),  INT8_C(  11), -INT8_C( 116),  INT8_C(  10), -INT8_C( 114),  INT8_C(  95),  INT8_C(  33),  INT8_C(  87) },
      {  INT8_C(  22),  INT8_C(  47),  INT8_C(  59),  INT8_C(  24),  INT8_C(  32),  INT8_C(  81),  INT8_C(  23), -INT8_C(  70),
        -INT8_C(  65), -INT8_C(  13),  INT8_C( 117), -INT8_C(  83), -INT8_C(  66),  INT8_C(  88),  INT8_C(  67),  INT8_C(   1),
        -INT8_C( 107), -INT8_C( 113), -INT8_C( 109), -INT8_C(  36), -INT8_C(  47),  INT8_C( 116),  INT8_C(  63), -INT8_C(  58),
         INT8_C(  53),  INT8_C(  47),  INT8_C(  70), -INT8_C(  92),  INT8_C(  57),  INT8_C(  93), -INT8_C(  89),  INT8_C(  79),
        -INT8_C( 116), -INT8_C(  30),  INT8_C( 103), -INT8_C(  84),  INT8_C(  51),      INT8_MAX,  INT8_C( 102), -INT8_C(  14),
         INT8_C( 115), -INT8_C(  37), -INT8_C(  95),  INT8_C(  50),  INT8_C(  51), -INT8_C(  28),  INT8_C(  51), -INT8_C(  56),
         INT8_C( 115), -INT8_C(  58), -INT8_C(  92),  INT8_C(  69),  INT8_C(  58), -INT8_C(  28),  INT8_C(  12),  INT8_C( 111),
         INT8_C(  19),  INT8_C(  82),  INT8_C(  20),  INT8_C(  77), -INT8_C(  81), -INT8_C(  69), -INT8_C(  99),  INT8_C(  59) } },
    { { -INT8_C( 105), -INT8_C(  65), -INT8_C( 125),  INT8_C(  74),  INT8_C(  35), -INT8_C(  45), -INT8_C(   3), -INT8_C(  45),
         INT8_C(  44),  INT8_C(  24),  INT8_C(  34), -INT8_C(  10), -INT8_C(  86), -INT8_C(  21), -INT8_C(  79),  INT8_C(  66),
         INT8_C(  51), -INT8_C(  58), -INT8_C( 125),  INT8_C(   2),  INT8_C(   9), -INT8_C( 121), -INT8_C(  97),  INT8_C(   2),
        -INT8_C( 110),  INT8_C(  43),  INT8_C(  12),  INT8_C(  32), -INT8_C( 118),  INT8_C(  45),  INT8_C( 119),  INT8_C(  33),
        -INT8_C(  20), -INT8_C(   6),  INT8_C( 108),  INT8_C(  15), -INT8_C(  50),  INT8_C( 105), -INT8_C(  29), -INT8_C(   6),
        -INT8_C( 127),  INT8_C(   5), -INT8_C(  16),  INT8_C(  43), -INT8_C(  15), -INT8_C(  95),  INT8_C( 109),  INT8_C(  36),
         INT8_C( 104), -INT8_C(  16),  INT8_C(  39),  INT8_C( 113),  INT8_C( 119), -INT8_C(  58),  INT8_C( 115),  INT8_C(   9),
        -INT8_C(  14),      INT8_MAX,  INT8_C(  41),  INT8_C( 124), -INT8_C(  83), -INT8_C(  95), -INT8_C(  98), -INT8_C( 103) },
      { -INT8_C( 101),  INT8_C(  10), -INT8_C(  87),  INT8_C( 105),  INT8_C( 115), -INT8_C( 116),  INT8_C(  99), -INT8_C(  12),
        -INT8_C( 111),  INT8_C(  84),  INT8_C(  31), -INT8_C( 126), -INT8_C(  11), -INT8_C( 116), -INT8_C(  89),  INT8_C(  93),
         INT8_C( 125), -INT8_C(  50), -INT8_C(  49), -INT8_C(  12), -INT8_C( 108),  INT8_C(  66), -INT8_C(   2), -INT8_C( 122),
        -INT8_C(  62),  INT8_C(  39),  INT8_C(   3),  INT8_C( 111), -INT8_C(  56), -INT8_C(  95),  INT8_C(   8),  INT8_C( 100),
        -INT8_C(  85), -INT8_C(  79), -INT8_C(  51),  INT8_C(  30),  INT8_C(  61),  INT8_C(  49),  INT8_C(  18), -INT8_C(  49),
        -INT8_C( 123),  INT8_C(  49),  INT8_C(  81),  INT8_C( 122), -INT8_C(  67), -INT8_C(   8), -INT8_C(  40),  INT8_C(  58),
        -INT8_C(  58), -INT8_C(  89),  INT8_C(  47),  INT8_C(  91), -INT8_C(  23),  INT8_C(  45), -INT8_C(  31), -INT8_C(  85),
         INT8_C(  84), -INT8_C(  28),  INT8_C(  26),  INT8_C(  29), -INT8_C( 123),  INT8_C(  35), -INT8_C( 127),  INT8_C(  48) },
      {  INT8_C(  50), -INT8_C(  55),  INT8_C(  44), -INT8_C(  77), -INT8_C( 106),  INT8_C(  95),  INT8_C(  96), -INT8_C(  57),
        -INT8_C(  67),  INT8_C( 108),  INT8_C(  65),  INT8_C( 120), -INT8_C(  97),  INT8_C( 119),  INT8_C(  88), -INT8_C(  97),
        -INT8_C(  80), -INT8_C( 108),  INT8_C(  82), -INT8_C(  10), -INT8_C(  99), -INT8_C(  55), -INT8_C(  99), -INT8_C( 120),
         INT8_C(  84),  INT8_C(  82),  INT8_C(  15), -INT8_C( 113),  INT8_C(  82), -INT8_C(  50),      INT8_MAX, -INT8_C( 123),
        -INT8_C( 105), -INT8_C(  85),  INT8_C(  57),  INT8_C(  45),  INT8_C(  11), -INT8_C( 102), -INT8_C(  11), -INT8_C(  55),
         INT8_C(   6),  INT8_C(  54),  INT8_C(  65), -INT8_C(  91), -INT8_C(  82), -INT8_C( 103),  INT8_C(  69),  INT8_C(  94),
         INT8_C(  46), -INT8_C( 105),  INT8_C(  86), -INT8_C(  52),  INT8_C(  96), -INT8_C(  13),  INT8_C(  84), -INT8_C(  76),
         INT8_C(  70),  INT8_C(  99),  INT8_C(  67), -INT8_C( 103),  INT8_C(  50), -INT8_C(  60),  INT8_C(  31), -INT8_C(  55) } },
    { { -INT8_C(  44),  INT8_C(  78),  INT8_C(  78),  INT8_C(  18),      INT8_MAX,  INT8_C(  96), -INT8_C(  31),  INT8_C(   4),
        -INT8_C( 111),  INT8_C(  50),      INT8_MAX,  INT8_C(  79),  INT8_C(  43),  INT8_C(  87), -INT8_C( 119), -INT8_C(  15),
        -INT8_C(   2), -INT8_C(  72),  INT8_C(  76), -INT8_C(  25), -INT8_C(  27),  INT8_C(  46), -INT8_C( 109),  INT8_C(  58),
         INT8_C(  18), -INT8_C(  83),  INT8_C(  87), -INT8_C( 104), -INT8_C(  48), -INT8_C(  40), -INT8_C(  56), -INT8_C(  91),
         INT8_C(  38),  INT8_C(  23), -INT8_C(  73), -INT8_C(  90),  INT8_C( 119), -INT8_C( 104), -INT8_C(  86),  INT8_C(   9),
        -INT8_C(  54),  INT8_C(  41),  INT8_C(  88), -INT8_C(  11),      INT8_MIN, -INT8_C(  31), -INT8_C(  25),  INT8_C( 126),
        -INT8_C( 102),  INT8_C(  51),  INT8_C( 102),      INT8_MAX,  INT8_C(  97), -INT8_C(   7), -INT8_C(  71),  INT8_C( 116),
        -INT8_C(  90),  INT8_C(  16),  INT8_C(  12),  INT8_C( 119), -INT8_C(  24), -INT8_C(  44),  INT8_C(  28),  INT8_C(  15) },
      { -INT8_C(  21), -INT8_C(  45), -INT8_C(  75),  INT8_C(  99),  INT8_C( 107),  INT8_C(  95),  INT8_C( 108),  INT8_C(  53),
        -INT8_C( 119), -INT8_C(  60),  INT8_C(  43),  INT8_C(   9), -INT8_C(  91),  INT8_C(  18), -INT8_C( 120),  INT8_C(  63),
         INT8_C(  69), -INT8_C(  18), -INT8_C(  65), -INT8_C(  89), -INT8_C(  25),  INT8_C( 120),  INT8_C(  27), -INT8_C( 115),
        -INT8_C( 119),  INT8_C(  39),  INT8_C(   4),  INT8_C( 113), -INT8_C(   5),  INT8_C(  32),      INT8_MIN, -INT8_C(  25),
        -INT8_C(  13),  INT8_C(  53),  INT8_C(  74),  INT8_C(  94), -INT8_C( 107), -INT8_C(  74), -INT8_C( 108),  INT8_C(  30),
         INT8_C( 122), -INT8_C(  65),  INT8_C(  39),  INT8_C(  31), -INT8_C(  47), -INT8_C(  81),  INT8_C(  95),  INT8_C(  22),
        -INT8_C(  99),  INT8_C(  30), -INT8_C(  67), -INT8_C( 124), -INT8_C( 106), -INT8_C(  40),  INT8_C(  18),  INT8_C(  31),
        -INT8_C(   1),  INT8_C(  22), -INT8_C( 111), -INT8_C(   5),  INT8_C(  55),  INT8_C(  17), -INT8_C(  30),  INT8_C(  42) },
      { -INT8_C(  65),  INT8_C(  33),  INT8_C(   3),  INT8_C( 117), -INT8_C(  22), -INT8_C(  65),  INT8_C(  77),  INT8_C(  57),
         INT8_C(  26), -INT8_C(  10), -INT8_C(  86),  INT8_C(  88), -INT8_C(  48),  INT8_C( 105),  INT8_C(  17),  INT8_C(  48),
         INT8_C(  67), -INT8_C(  90),  INT8_C(  11), -INT8_C( 114), -INT8_C(  52), -INT8_C(  90), -INT8_C(  82), -INT8_C(  57),
        -INT8_C( 101), -INT8_C(  44),  INT8_C(  91),  INT8_C(   9), -INT8_C(  53), -INT8_C(   8),  INT8_C(  72), -INT8_C( 116),
         INT8_C(  25),  INT8_C(  76),  INT8_C(   1),  INT8_C(   4),  INT8_C(  12),  INT8_C(  78),  INT8_C(  62),  INT8_C(  39),
         INT8_C(  68), -INT8_C(  24),      INT8_MAX,  INT8_C(  20),  INT8_C(  81), -INT8_C( 112),  INT8_C(  70), -INT8_C( 108),
         INT8_C(  55),  INT8_C(  81),  INT8_C(  35),  INT8_C(   3), -INT8_C(   9), -INT8_C(  47), -INT8_C(  53), -INT8_C( 109),
        -INT8_C(  91),  INT8_C(  38), -INT8_C(  99),  INT8_C( 114),  INT8_C(  31), -INT8_C(  27), -INT8_C(   2),  INT8_C(  57) } },
    { {  INT8_C(  71),  INT8_C(  44), -INT8_C( 119), -INT8_C(  36), -INT8_C(  30),  INT8_C(  29), -INT8_C(   6),  INT8_C(  92),
        -INT8_C(  36),  INT8_C(  33),  INT8_C( 123), -INT8_C(  83), -INT8_C(  47), -INT8_C(  38), -INT8_C(  61),  INT8_C( 110),
        -INT8_C(   8), -INT8_C( 127), -INT8_C(  13), -INT8_C( 113),  INT8_C(  89),  INT8_C(   5), -INT8_C(  82),  INT8_C(  89),
         INT8_C(  27),  INT8_C(  63),  INT8_C(  84),  INT8_C(  82),  INT8_C(  81),  INT8_C(  54),  INT8_C( 125), -INT8_C( 104),
         INT8_C(  98),  INT8_C(   6),  INT8_C( 116),  INT8_C(  68),  INT8_C(  35),  INT8_C( 110), -INT8_C(  96), -INT8_C(   1),
        -INT8_C( 113),  INT8_C(  27), -INT8_C(  84),  INT8_C(  96), -INT8_C(  10),  INT8_C( 111), -INT8_C(  49), -INT8_C(  18),
        -INT8_C(  16), -INT8_C(  62),  INT8_C( 125),  INT8_C(  74), -INT8_C(  57),  INT8_C(  44), -INT8_C(  93), -INT8_C(  30),
         INT8_C( 107), -INT8_C(   9),  INT8_C(  53), -INT8_C(  68),  INT8_C(  45), -INT8_C(  78),  INT8_C(  84), -INT8_C( 113) },
      { -INT8_C(  72), -INT8_C(  56), -INT8_C(  45), -INT8_C(  37),  INT8_C(  54),  INT8_C( 115), -INT8_C(  38), -INT8_C(  58),
        -INT8_C( 114), -INT8_C( 122),  INT8_C(  38), -INT8_C( 124), -INT8_C(  11), -INT8_C(  11),  INT8_C( 115), -INT8_C(  26),
        -INT8_C(  73), -INT8_C(  16),  INT8_C(  48),  INT8_C( 126),  INT8_C(  28), -INT8_C(  45),  INT8_C(  97), -INT8_C( 120),
        -INT8_C(  54), -INT8_C( 106),  INT8_C(  68), -INT8_C(   9),  INT8_C(  72), -INT8_C( 103), -INT8_C( 122),  INT8_C(   0),
         INT8_C(  97),  INT8_C(  89), -INT8_C(  37), -INT8_C( 104), -INT8_C(  52), -INT8_C(  75),  INT8_C(  94),  INT8_C(  90),
         INT8_C(  59), -INT8_C( 124), -INT8_C(  33),  INT8_C(  48),  INT8_C( 122),  INT8_C(  82),  INT8_C(  22),  INT8_C(  49),
         INT8_C(  66),  INT8_C(  70), -INT8_C(  80),  INT8_C(  95),  INT8_C(  25),  INT8_C(  17), -INT8_C(  25), -INT8_C(  29),
        -INT8_C(  89),  INT8_C(  43), -INT8_C(  38), -INT8_C(  17), -INT8_C(  60),  INT8_C(  96), -INT8_C(  17),  INT8_C(  38) },
      { -INT8_C(   1), -INT8_C(  12),  INT8_C(  92), -INT8_C(  73),  INT8_C(  24), -INT8_C( 112), -INT8_C(  44),  INT8_C(  34),
         INT8_C( 106), -INT8_C(  89), -INT8_C(  95),  INT8_C(  49), -INT8_C(  58), -INT8_C(  49),  INT8_C(  54),  INT8_C(  84),
        -INT8_C(  81),  INT8_C( 113),  INT8_C(  35),  INT8_C(  13),  INT8_C( 117), -INT8_C(  40),  INT8_C(  15), -INT8_C(  31),
        -INT8_C(  27), -INT8_C(  43), -INT8_C( 104),  INT8_C(  73), -INT8_C( 103), -INT8_C(  49),  INT8_C(   3), -INT8_C( 104),
        -INT8_C(  61),  INT8_C(  95),  INT8_C(  79), -INT8_C(  36), -INT8_C(  17),  INT8_C(  35), -INT8_C(   2),  INT8_C(  89),
        -INT8_C(  54), -INT8_C(  97), -INT8_C( 117), -INT8_C( 112),  INT8_C( 112), -INT8_C(  63), -INT8_C(  27),  INT8_C(  31),
         INT8_C(  50),  INT8_C(   8),  INT8_C(  45), -INT8_C(  87), -INT8_C(  32),  INT8_C(  61), -INT8_C( 118), -INT8_C(  59),
         INT8_C(  18),  INT8_C(  34),  INT8_C(  15), -INT8_C(  85), -INT8_C(  15),  INT8_C(  18),  INT8_C(  67), -INT8_C(  75) } },
    { { -INT8_C(  71), -INT8_C(  54), -INT8_C(  66), -INT8_C( 123),      INT8_MAX,  INT8_C(  28), -INT8_C(  32), -INT8_C(  70),
        -INT8_C(  96), -INT8_C(  65), -INT8_C(  22),  INT8_C(  26),  INT8_C(  17),  INT8_C(   1),  INT8_C(  76),  INT8_C(  83),
         INT8_C(  71), -INT8_C(   4), -INT8_C(  78),  INT8_C(  97),  INT8_C(  13), -INT8_C( 103),  INT8_C(  68), -INT8_C(  76),
        -INT8_C(  59),  INT8_C(  31), -INT8_C(  93), -INT8_C( 119),      INT8_MAX, -INT8_C( 110), -INT8_C(  81),  INT8_C(  57),
         INT8_C(  92),  INT8_C( 109), -INT8_C(  66), -INT8_C(  37), -INT8_C( 119), -INT8_C(  98), -INT8_C( 107),  INT8_C(  42),
         INT8_C(  93),      INT8_MAX,  INT8_C(  68),  INT8_C( 110),      INT8_MIN, -INT8_C( 112), -INT8_C(  62), -INT8_C(  56),
        -INT8_C( 116),  INT8_C( 116),  INT8_C(  41), -INT8_C( 103),  INT8_C(  14),  INT8_C( 109),  INT8_C(  77), -INT8_C(  45),
        -INT8_C( 116), -INT8_C(  16),  INT8_C(  92),  INT8_C(  12), -INT8_C( 126),  INT8_C(  12),  INT8_C(  69), -INT8_C(  34) },
      {  INT8_C( 121),  INT8_C(   3), -INT8_C(  71),  INT8_C(   3), -INT8_C(  94),  INT8_C(  78),  INT8_C(  45), -INT8_C(   1),
        -INT8_C(  50),  INT8_C( 113),  INT8_C( 110),  INT8_C(  78),  INT8_C(   2),  INT8_C(  48),  INT8_C(  22), -INT8_C( 114),
        -INT8_C(  92),  INT8_C(  63),  INT8_C(  40), -INT8_C(  78), -INT8_C(  83),  INT8_C( 117), -INT8_C( 123),  INT8_C(  57),
         INT8_C( 102), -INT8_C(  30),  INT8_C(  69), -INT8_C(  24), -INT8_C(  18), -INT8_C( 118), -INT8_C(  57),  INT8_C( 103),
        -INT8_C( 114),      INT8_MIN,  INT8_C( 106),  INT8_C(  48), -INT8_C(  49), -INT8_C( 105),  INT8_C(  47), -INT8_C(  99),
         INT8_C(   9), -INT8_C(  99), -INT8_C(  21),  INT8_C(  11), -INT8_C(  51),  INT8_C(   2), -INT8_C( 103),  INT8_C( 114),
         INT8_C(  65), -INT8_C(  63),  INT8_C(  36), -INT8_C(  18),  INT8_C(  55), -INT8_C(  86),  INT8_C(  40), -INT8_C(  99),
        -INT8_C( 116),  INT8_C( 109), -INT8_C( 123),  INT8_C( 122), -INT8_C(   8),  INT8_C(  76), -INT8_C(  31), -INT8_C( 122) },
      {  INT8_C(  50), -INT8_C(  51),  INT8_C( 119), -INT8_C( 120),  INT8_C(  33),  INT8_C( 106),  INT8_C(  13), -INT8_C(  71),
         INT8_C( 110),  INT8_C(  48),  INT8_C(  88),  INT8_C( 104),  INT8_C(  19),  INT8_C(  49),  INT8_C(  98), -INT8_C(  31),
        -INT8_C(  21),  INT8_C(  59), -INT8_C(  38),  INT8_C(  19), -INT8_C(  70),  INT8_C(  14), -INT8_C(  55), -INT8_C(  19),
         INT8_C(  43),  INT8_C(   1), -INT8_C(  24),  INT8_C( 113),  INT8_C( 109),  INT8_C(  28),  INT8_C( 118), -INT8_C(  96),
        -INT8_C(  22), -INT8_C(  19),  INT8_C(  40),  INT8_C(  11),  INT8_C(  88),  INT8_C(  53), -INT8_C(  60), -INT8_C(  57),
         INT8_C( 102),  INT8_C(  28),  INT8_C(  47),  INT8_C( 121),  INT8_C(  77), -INT8_C( 110),  INT8_C(  91),  INT8_C(  58),
        -INT8_C(  51),  INT8_C(  53),  INT8_C(  77), -INT8_C( 121),  INT8_C(  69),  INT8_C(  23),  INT8_C( 117),  INT8_C( 112),
         INT8_C(  24),  INT8_C(  93), -INT8_C(  31), -INT8_C( 122),  INT8_C( 122),  INT8_C(  88),  INT8_C(  38),  INT8_C( 100) } },
    { { -INT8_C(  51),  INT8_C(  76), -INT8_C(  74), -INT8_C( 100), -INT8_C(  29), -INT8_C(  27),  INT8_C(  57), -INT8_C(  20),
        -INT8_C( 125),  INT8_C(  36), -INT8_C(   9),  INT8_C(  80),  INT8_C(  38), -INT8_C( 111), -INT8_C(  62),  INT8_C( 104),
         INT8_C(  82), -INT8_C(  25),  INT8_C(  86), -INT8_C( 119), -INT8_C( 111),  INT8_C( 126),  INT8_C(  38),  INT8_C(  29),
        -INT8_C(  20), -INT8_C(  84), -INT8_C( 105), -INT8_C(  28), -INT8_C(   8),  INT8_C( 120),  INT8_C( 106), -INT8_C(  59),
        -INT8_C(  60),  INT8_C(  32),  INT8_C(  97), -INT8_C(  88),  INT8_C(   5), -INT8_C( 102), -INT8_C( 108), -INT8_C( 120),
        -INT8_C(  65), -INT8_C( 116), -INT8_C(  39), -INT8_C(  27),  INT8_C(  29), -INT8_C( 101),  INT8_C(  77),  INT8_C( 111),
        -INT8_C( 126), -INT8_C(  92), -INT8_C(   7),  INT8_C(  19),  INT8_C(  34),  INT8_C(  31),  INT8_C(  48),  INT8_C(  14),
        -INT8_C(  53), -INT8_C(  57), -INT8_C(  14), -INT8_C(  60),  INT8_C(  64),  INT8_C(  92), -INT8_C( 119),  INT8_C(   4) },
      {  INT8_C( 124), -INT8_C(  21), -INT8_C(  84), -INT8_C( 126), -INT8_C( 123),  INT8_C(  65),  INT8_C(  10),  INT8_C(  68),
        -INT8_C(  51), -INT8_C(  29),  INT8_C(  42), -INT8_C(  22),      INT8_MAX,  INT8_C( 119),  INT8_C(  89),  INT8_C(   1),
         INT8_C(  27),  INT8_C(  82),  INT8_C(  21),  INT8_C(  62),  INT8_C( 114),  INT8_C(  69),  INT8_C(  76),  INT8_C(  61),
         INT8_C(  13),  INT8_C(  63),  INT8_C(   1),  INT8_C(  77), -INT8_C( 101), -INT8_C( 117),  INT8_C(  81),  INT8_C(  24),
         INT8_C( 118), -INT8_C(   2), -INT8_C( 102), -INT8_C(   5),  INT8_C(  63), -INT8_C(  92),  INT8_C(  64),  INT8_C(  12),
        -INT8_C( 120),  INT8_C( 106), -INT8_C(  10),  INT8_C(   7), -INT8_C(  31),  INT8_C(  79),  INT8_C(   8), -INT8_C(   3),
        -INT8_C(  94),  INT8_C(  29),  INT8_C(  59),  INT8_C(  20),  INT8_C(  99), -INT8_C( 121),  INT8_C(  81),  INT8_C( 112),
        -INT8_C(  58),  INT8_C(  83), -INT8_C(  67),  INT8_C(  98), -INT8_C(  34),  INT8_C(  14),  INT8_C( 122),  INT8_C(  84) },
      {  INT8_C(  73),  INT8_C(  55),  INT8_C(  98),  INT8_C(  30),  INT8_C( 104),  INT8_C(  38),  INT8_C(  67),  INT8_C(  48),
         INT8_C(  80),  INT8_C(   7),  INT8_C(  33),  INT8_C(  58), -INT8_C(  91),  INT8_C(   8),  INT8_C(  27),  INT8_C( 105),
         INT8_C( 109),  INT8_C(  57),  INT8_C( 107), -INT8_C(  57),  INT8_C(   3), -INT8_C(  61),  INT8_C( 114),  INT8_C(  90),
        -INT8_C(   7), -INT8_C(  21), -INT8_C( 104),  INT8_C(  49), -INT8_C( 109),  INT8_C(   3), -INT8_C(  69), -INT8_C(  35),
         INT8_C(  58),  INT8_C(  30), -INT8_C(   5), -INT8_C(  93),  INT8_C(  68),  INT8_C(  62), -INT8_C(  44), -INT8_C( 108),
         INT8_C(  71), -INT8_C(  10), -INT8_C(  49), -INT8_C(  20), -INT8_C(   2), -INT8_C(  22),  INT8_C(  85),  INT8_C( 108),
         INT8_C(  36), -INT8_C(  63),  INT8_C(  52),  INT8_C(  39), -INT8_C( 123), -INT8_C(  90), -INT8_C( 127),  INT8_C( 126),
        -INT8_C( 111),  INT8_C(  26), -INT8_C(  81),  INT8_C(  38),  INT8_C(  30),  INT8_C( 106),  INT8_C(   3),  INT8_C(  88) } },
    { {  INT8_C(  12),  INT8_C(  20),  INT8_C(  79),  INT8_C(  75), -INT8_C(  72), -INT8_C( 113),  INT8_C(  87),  INT8_C(  64),
        -INT8_C(   7),  INT8_C(  77),  INT8_C(  71), -INT8_C(  37), -INT8_C(  99),  INT8_C(  80), -INT8_C(  40),  INT8_C(  63),
         INT8_C( 109),  INT8_C(  19),  INT8_C(  83), -INT8_C(  48), -INT8_C( 102), -INT8_C(  92),  INT8_C(  64),  INT8_C(  97),
        -INT8_C(   9), -INT8_C(   3), -INT8_C(  61), -INT8_C(  43),  INT8_C(  12),  INT8_C(  61),  INT8_C(  41),  INT8_C(  24),
         INT8_C(  81),  INT8_C( 121),  INT8_C( 100),  INT8_C(   9),  INT8_C(   8), -INT8_C(  69),  INT8_C(  74),  INT8_C(   2),
         INT8_C(   9), -INT8_C( 111), -INT8_C(  35), -INT8_C(  90), -INT8_C(  31), -INT8_C(  75), -INT8_C(  27),  INT8_C(  79),
        -INT8_C(  56),  INT8_C(  56),  INT8_C(  31),  INT8_C(  98), -INT8_C(  36),  INT8_C(  96), -INT8_C(  61), -INT8_C(  44),
         INT8_C(  93), -INT8_C( 122), -INT8_C(  87),  INT8_C( 105), -INT8_C(  61), -INT8_C(  45), -INT8_C( 126),  INT8_C(  20) },
      {  INT8_C(  76), -INT8_C(  26),  INT8_C(  30),  INT8_C(  84), -INT8_C(  95),  INT8_C( 104),  INT8_C(  86), -INT8_C(  86),
        -INT8_C(   7),  INT8_C(  51),  INT8_C(  80), -INT8_C(  37), -INT8_C(  24),  INT8_C(  53),  INT8_C(  42), -INT8_C(  80),
         INT8_C( 109),  INT8_C(  73),  INT8_C(  19),  INT8_C(  74), -INT8_C(  87), -INT8_C(  42),  INT8_C(  30),  INT8_C(   7),
         INT8_C(  93), -INT8_C(  57),  INT8_C( 112),  INT8_C(  32), -INT8_C( 102), -INT8_C(  14),  INT8_C(  53), -INT8_C(  26),
        -INT8_C(  40),  INT8_C(  83),  INT8_C(  59),  INT8_C( 122), -INT8_C(  69), -INT8_C( 111),  INT8_C(  36), -INT8_C(  76),
        -INT8_C(  59),  INT8_C( 117), -INT8_C( 113), -INT8_C(  83), -INT8_C(  86), -INT8_C(  71),  INT8_C(  94),  INT8_C(  24),
         INT8_C(   3),  INT8_C( 113),  INT8_C(  98), -INT8_C(  84),  INT8_C(  71),      INT8_MIN, -INT8_C(  77), -INT8_C(  92),
         INT8_C(  71),  INT8_C(  36), -INT8_C(  59), -INT8_C(  30),  INT8_C(  22), -INT8_C(   6), -INT8_C(  56), -INT8_C(  17) },
      {  INT8_C(  88), -INT8_C(   6),  INT8_C( 109), -INT8_C(  97),  INT8_C(  89), -INT8_C(   9), -INT8_C(  83), -INT8_C(  22),
        -INT8_C(  14),      INT8_MIN, -INT8_C( 105), -INT8_C(  74), -INT8_C( 123), -INT8_C( 123),  INT8_C(   2), -INT8_C(  17),
        -INT8_C(  38),  INT8_C(  92),  INT8_C( 102),  INT8_C(  26),  INT8_C(  67),  INT8_C( 122),  INT8_C(  94),  INT8_C( 104),
         INT8_C(  84), -INT8_C(  60),  INT8_C(  51), -INT8_C(  11), -INT8_C(  90),  INT8_C(  47),  INT8_C(  94), -INT8_C(   2),
         INT8_C(  41), -INT8_C(  52), -INT8_C(  97), -INT8_C( 125), -INT8_C(  61),  INT8_C(  76),  INT8_C( 110), -INT8_C(  74),
        -INT8_C(  50),  INT8_C(   6),  INT8_C( 108),  INT8_C(  83), -INT8_C( 117),  INT8_C( 110),  INT8_C(  67),  INT8_C( 103),
        -INT8_C(  53), -INT8_C(  87), -INT8_C( 127),  INT8_C(  14),  INT8_C(  35), -INT8_C(  32),  INT8_C( 118),  INT8_C( 120),
        -INT8_C(  92), -INT8_C(  86),  INT8_C( 110),  INT8_C(  75), -INT8_C(  39), -INT8_C(  51),  INT8_C(  74),  INT8_C(   3) } },
    { {  INT8_C(  77),  INT8_C(   3),  INT8_C( 105),  INT8_C(   8), -INT8_C( 107), -INT8_C( 115), -INT8_C(  68),  INT8_C(  90),
         INT8_C(   2),  INT8_C(  76),  INT8_C(   7), -INT8_C(  83),  INT8_C(   5),  INT8_C( 101), -INT8_C(  59),  INT8_C(   8),
        -INT8_C(  42),  INT8_C(  39), -INT8_C(  75),  INT8_C(  30), -INT8_C(  89),  INT8_C( 104), -INT8_C(  62), -INT8_C(  18),
        -INT8_C( 116), -INT8_C( 121), -INT8_C(  48), -INT8_C(  93), -INT8_C( 127), -INT8_C( 103), -INT8_C( 110), -INT8_C(  50),
        -INT8_C( 100), -INT8_C(   5), -INT8_C(  42),  INT8_C(  49), -INT8_C( 120), -INT8_C( 109), -INT8_C( 117), -INT8_C( 117),
        -INT8_C(  33), -INT8_C( 109),  INT8_C(  56), -INT8_C(  28), -INT8_C(   8), -INT8_C(   3), -INT8_C(  19), -INT8_C(  49),
         INT8_C(  36), -INT8_C(  94), -INT8_C(  19), -INT8_C(  53),  INT8_C(  10), -INT8_C(  81), -INT8_C(  71), -INT8_C( 105),
         INT8_C(  55), -INT8_C( 118),  INT8_C(  58), -INT8_C(  72),  INT8_C(  35), -INT8_C(  52), -INT8_C( 121), -INT8_C(  65) },
      { -INT8_C(  57),  INT8_C(  93), -INT8_C(  15),  INT8_C(  79), -INT8_C(  16),  INT8_C( 124), -INT8_C(  38), -INT8_C(  49),
         INT8_C(  15),  INT8_C(  18), -INT8_C(  76),  INT8_C(   8),  INT8_C(  15), -INT8_C(  95), -INT8_C(  41),  INT8_C(  51),
         INT8_C(  67), -INT8_C(  60), -INT8_C(   2),  INT8_C(  77),  INT8_C( 115), -INT8_C(  72), -INT8_C(  28), -INT8_C(  86),
         INT8_C(  66),  INT8_C(  30),  INT8_C(  99),  INT8_C( 101), -INT8_C(  22), -INT8_C(  22),  INT8_C(  36), -INT8_C(  79),
         INT8_C(  71),  INT8_C(  21),  INT8_C(   1),  INT8_C(  56), -INT8_C( 110), -INT8_C(  37),  INT8_C(   7), -INT8_C(  95),
        -INT8_C(  18), -INT8_C(  69), -INT8_C(  87), -INT8_C(   3),  INT8_C(  92),      INT8_MIN,  INT8_C(  49), -INT8_C(  97),
         INT8_C(  68),  INT8_C(  47), -INT8_C(  19), -INT8_C(  72), -INT8_C(  25), -INT8_C(  47),  INT8_C(  98),  INT8_C(  41),
        -INT8_C(  16), -INT8_C(  59), -INT8_C( 114), -INT8_C(  38), -INT8_C(  81), -INT8_C(  77), -INT8_C( 116), -INT8_C(   9) },
      {  INT8_C(  20),  INT8_C(  96),  INT8_C(  90),  INT8_C(  87), -INT8_C( 123),  INT8_C(   9), -INT8_C( 106),  INT8_C(  41),
         INT8_C(  17),  INT8_C(  94), -INT8_C(  69), -INT8_C(  75),  INT8_C(  20),  INT8_C(   6), -INT8_C( 100),  INT8_C(  59),
         INT8_C(  25), -INT8_C(  21), -INT8_C(  77),  INT8_C( 107),  INT8_C(  26),  INT8_C(  32), -INT8_C(  90), -INT8_C( 104),
        -INT8_C(  50), -INT8_C(  91),  INT8_C(  51),  INT8_C(   8),  INT8_C( 107), -INT8_C( 125), -INT8_C(  74),      INT8_MAX,
        -INT8_C(  29),  INT8_C(  16), -INT8_C(  41),  INT8_C( 105),  INT8_C(  26),  INT8_C( 110), -INT8_C( 110),  INT8_C(  44),
        -INT8_C(  51),  INT8_C(  78), -INT8_C(  31), -INT8_C(  31),  INT8_C(  84),  INT8_C( 125),  INT8_C(  30),  INT8_C( 110),
         INT8_C( 104), -INT8_C(  47), -INT8_C(  38), -INT8_C( 125), -INT8_C(  15),      INT8_MIN,  INT8_C(  27), -INT8_C(  64),
         INT8_C(  39),  INT8_C(  79), -INT8_C(  56), -INT8_C( 110), -INT8_C(  46),      INT8_MAX,  INT8_C(  19), -INT8_C(  74) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_add_epi8(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_add_epi8");
    easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_add_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int8_t src[64];
    const easysimd__mmask64 k;
    const int8_t a[64];
    const int8_t b[64];
    const int8_t r[64];
  } test_vec[] = {
    { {  INT8_C(  20), -INT8_C(  19),  INT8_C(  38),  INT8_C(  45), -INT8_C(  89), -INT8_C(  95),  INT8_C(  34), -INT8_C(   3),
        -INT8_C(  21),  INT8_C(  99), -INT8_C(  35),  INT8_C(  73),  INT8_C(   3), -INT8_C( 119),  INT8_C(   1), -INT8_C(   7),
        -INT8_C( 123), -INT8_C(  90),  INT8_C(  32),  INT8_C(  73), -INT8_C(  12), -INT8_C( 115), -INT8_C(  93),  INT8_C(  56),
         INT8_C( 116),  INT8_C(  96), -INT8_C(  89), -INT8_C( 108),  INT8_C(  46), -INT8_C(  17),  INT8_C(  92),  INT8_C(  67),
        -INT8_C(  35), -INT8_C( 125),  INT8_C( 112), -INT8_C( 124),  INT8_C(  36), -INT8_C( 110), -INT8_C( 127),  INT8_C(  15),
        -INT8_C(  11),  INT8_C(  95),  INT8_C(  88), -INT8_C(   8), -INT8_C(  24),  INT8_C(  89), -INT8_C(  14),  INT8_C( 109),
        -INT8_C(   1),  INT8_C(  18), -INT8_C(  73), -INT8_C(  13), -INT8_C(  97),  INT8_C(  90),  INT8_C(  43),  INT8_C(  19),
        -INT8_C(  69), -INT8_C(  45), -INT8_C(  88), -INT8_C(  23), -INT8_C(  62),  INT8_C(   4),  INT8_C(  44), -INT8_C(  97) },
      UINT64_C( 2646890825868614791),
      {  INT8_C(   4),  INT8_C(  19),  INT8_C(  28), -INT8_C(  20),  INT8_C( 109),  INT8_C(  14),  INT8_C(  90),  INT8_C( 108),
         INT8_C(  32),  INT8_C(  17),  INT8_C(  96), -INT8_C(  64),  INT8_C( 107), -INT8_C( 117), -INT8_C(  45),  INT8_C(  38),
         INT8_C(  94),  INT8_C( 123),  INT8_C(  16),  INT8_C(  33),      INT8_MIN,  INT8_C(  60), -INT8_C(  64),  INT8_C(   7),
        -INT8_C(  39), -INT8_C(  28), -INT8_C(  77),  INT8_C(   7), -INT8_C( 119),  INT8_C( 110),  INT8_C(  43), -INT8_C( 115),
        -INT8_C( 127),  INT8_C(  72),  INT8_C( 121), -INT8_C(  18),  INT8_C(  86), -INT8_C(  45),  INT8_C(  91),  INT8_C( 119),
        -INT8_C(  28), -INT8_C(  69),  INT8_C(  55),  INT8_C(  80),  INT8_C(  70),  INT8_C(  10),  INT8_C( 118), -INT8_C(  91),
        -INT8_C( 122), -INT8_C( 122), -INT8_C(  58),  INT8_C(   6), -INT8_C(  61), -INT8_C( 122),  INT8_C(  13), -INT8_C( 100),
         INT8_C( 106), -INT8_C(  64), -INT8_C(  93), -INT8_C(  13),  INT8_C(  46), -INT8_C(  49),      INT8_MIN, -INT8_C(  80) },
      {  INT8_C(  23), -INT8_C(   6), -INT8_C(  98),  INT8_C( 109), -INT8_C(  51), -INT8_C(   7), -INT8_C(  28), -INT8_C(  78),
        -INT8_C(  76),  INT8_C(  27),  INT8_C(   2), -INT8_C(   5),  INT8_C(  38),  INT8_C( 120), -INT8_C(  96), -INT8_C(  84),
        -INT8_C(   1),  INT8_C( 102), -INT8_C(  78), -INT8_C(  62), -INT8_C(  20), -INT8_C(  65),  INT8_C(  94),  INT8_C(  87),
             INT8_MIN,  INT8_C(   1),  INT8_C(  74), -INT8_C(  82), -INT8_C(  48), -INT8_C(  53),  INT8_C(  94), -INT8_C(  25),
        -INT8_C(  59), -INT8_C(   3),  INT8_C(  85), -INT8_C( 110), -INT8_C(  10),  INT8_C(  57),  INT8_C(  68), -INT8_C(  85),
         INT8_C(  85),  INT8_C(  70), -INT8_C(  90),  INT8_C( 123), -INT8_C(  65),  INT8_C(  70),  INT8_C(  39), -INT8_C(  66),
        -INT8_C(  84), -INT8_C(  39),      INT8_MIN, -INT8_C( 104), -INT8_C( 104), -INT8_C(  34), -INT8_C(  17),  INT8_C(  24),
        -INT8_C(  33),  INT8_C(  58), -INT8_C(  57), -INT8_C(  80),  INT8_C(   5),  INT8_C(  37), -INT8_C( 105), -INT8_C(  54) },
      {  INT8_C(  27),  INT8_C(  13), -INT8_C(  70),  INT8_C(  45), -INT8_C(  89), -INT8_C(  95),  INT8_C(  34),  INT8_C(  30),
        -INT8_C(  21),  INT8_C(  99),  INT8_C(  98), -INT8_C(  69), -INT8_C( 111), -INT8_C( 119),  INT8_C(   1), -INT8_C(  46),
         INT8_C(  93), -INT8_C(  31),  INT8_C(  32),  INT8_C(  73), -INT8_C(  12), -INT8_C(   5), -INT8_C(  93),  INT8_C(  56),
         INT8_C(  89), -INT8_C(  27), -INT8_C(  89), -INT8_C(  75),  INT8_C(  46),  INT8_C(  57),  INT8_C(  92),  INT8_C( 116),
        -INT8_C(  35),  INT8_C(  69), -INT8_C(  50),      INT8_MIN,  INT8_C(  36),  INT8_C(  12), -INT8_C( 127),  INT8_C(  15),
         INT8_C(  57),  INT8_C(  95), -INT8_C(  35), -INT8_C(   8), -INT8_C(  24),  INT8_C(  80), -INT8_C(  14),  INT8_C(  99),
         INT8_C(  50),  INT8_C(  95), -INT8_C(  73), -INT8_C(  98),  INT8_C(  91),  INT8_C( 100),  INT8_C(  43), -INT8_C(  76),
        -INT8_C(  69), -INT8_C(  45),  INT8_C( 106), -INT8_C(  23), -INT8_C(  62), -INT8_C(  12),  INT8_C(  44), -INT8_C(  97) } },
    { {  INT8_C(  34), -INT8_C(  20),  INT8_C(  92),  INT8_C(  25),  INT8_C(  38), -INT8_C(  95), -INT8_C(  60),  INT8_C( 123),
        -INT8_C(  25),  INT8_C( 106), -INT8_C(  10), -INT8_C(  90), -INT8_C(  80),  INT8_C(  29),  INT8_C( 100),  INT8_C(  92),
        -INT8_C(  10), -INT8_C(  28), -INT8_C(  12), -INT8_C( 114), -INT8_C(  62), -INT8_C(  28), -INT8_C(  89), -INT8_C(  94),
         INT8_C(  30),  INT8_C( 110),  INT8_C(  82),  INT8_C(  35), -INT8_C( 109), -INT8_C(  23), -INT8_C(  19), -INT8_C(  74),
        -INT8_C(  42),  INT8_C(  73), -INT8_C(  49), -INT8_C(   4), -INT8_C(  22), -INT8_C( 109),  INT8_C( 119), -INT8_C(  46),
        -INT8_C(   3),  INT8_C( 109),  INT8_C( 120), -INT8_C(  83), -INT8_C( 118), -INT8_C(  35),  INT8_C(   9),      INT8_MIN,
        -INT8_C(  63), -INT8_C(   3),  INT8_C(  14), -INT8_C( 124), -INT8_C(  31), -INT8_C(  75),  INT8_C(  38), -INT8_C(   1),
         INT8_C(  35),  INT8_C( 120),  INT8_C(  34), -INT8_C(  73),  INT8_C(  97),  INT8_C(  15),  INT8_C( 109),  INT8_C(  55) },
      UINT64_C(14705847965410606169),
      {  INT8_C(  23), -INT8_C( 114),  INT8_C( 121), -INT8_C(  95),  INT8_C( 107), -INT8_C( 126),  INT8_C(  33),  INT8_C(  44),
             INT8_MAX,  INT8_C(  48), -INT8_C(  80),  INT8_C(  97), -INT8_C(  27), -INT8_C(  42),  INT8_C(  96),  INT8_C(   9),
         INT8_C(  78), -INT8_C( 125), -INT8_C(  64), -INT8_C(  80), -INT8_C( 110),  INT8_C(  45), -INT8_C(  25), -INT8_C(  21),
         INT8_C( 105),  INT8_C(  27),  INT8_C(  47),  INT8_C(  56), -INT8_C(  59),  INT8_C(  68),  INT8_C(   4), -INT8_C(  35),
        -INT8_C(  46),  INT8_C( 125),  INT8_C( 126),  INT8_C(  61), -INT8_C(   1), -INT8_C(  96),  INT8_C( 106),  INT8_C( 126),
        -INT8_C(  48),  INT8_C(  26), -INT8_C(  33), -INT8_C(  75), -INT8_C(  15),  INT8_C(  64), -INT8_C(  66),  INT8_C(  63),
        -INT8_C(  61),  INT8_C( 126), -INT8_C(  17),  INT8_C(  85), -INT8_C(  85), -INT8_C(  41),  INT8_C(  65),  INT8_C(  20),
        -INT8_C(  14),  INT8_C( 112),  INT8_C(  76), -INT8_C(  73), -INT8_C(  76),  INT8_C(  80), -INT8_C( 108), -INT8_C( 121) },
      { -INT8_C(  51),  INT8_C(  19), -INT8_C(  60), -INT8_C(  52), -INT8_C(  77),  INT8_C(  46),  INT8_C(  75), -INT8_C( 125),
         INT8_C(  73),  INT8_C(  42),  INT8_C(  56),  INT8_C(  58),  INT8_C( 106), -INT8_C(   9),  INT8_C( 121),  INT8_C(  45),
         INT8_C( 117),  INT8_C( 105), -INT8_C( 125),  INT8_C(  33),  INT8_C(  64), -INT8_C(  60),  INT8_C(  53),  INT8_C(  50),
         INT8_C(  52), -INT8_C( 126), -INT8_C(  23), -INT8_C(  24), -INT8_C(  46),  INT8_C( 126),  INT8_C( 111), -INT8_C(  96),
        -INT8_C( 111),  INT8_C(  52),  INT8_C( 108),  INT8_C(  68),  INT8_C(  98), -INT8_C(  73), -INT8_C(  57), -INT8_C(  85),
        -INT8_C(  30), -INT8_C(   1), -INT8_C(  27),  INT8_C(  76), -INT8_C(  10),  INT8_C(  95),  INT8_C( 122),  INT8_C( 108),
        -INT8_C(  56), -INT8_C(   3), -INT8_C( 115),  INT8_C(   8), -INT8_C(  63), -INT8_C(  62),  INT8_C(  58), -INT8_C(  11),
         INT8_C(  68),  INT8_C(  35), -INT8_C(  35),  INT8_C(  23), -INT8_C(  95),  INT8_C(  77), -INT8_C(  73),  INT8_C(  50) },
      { -INT8_C(  28), -INT8_C(  20),  INT8_C(  92),  INT8_C( 109),  INT8_C(  30), -INT8_C(  95),  INT8_C( 108),  INT8_C( 123),
        -INT8_C(  25),  INT8_C( 106), -INT8_C(  24), -INT8_C( 101),  INT8_C(  79), -INT8_C(  51),  INT8_C( 100),  INT8_C(  92),
        -INT8_C(  61), -INT8_C(  20), -INT8_C(  12), -INT8_C( 114), -INT8_C(  46), -INT8_C(  15), -INT8_C(  89), -INT8_C(  94),
        -INT8_C(  99), -INT8_C(  99),  INT8_C(  82),  INT8_C(  35), -INT8_C( 109), -INT8_C(  23),  INT8_C( 115), -INT8_C(  74),
         INT8_C(  99), -INT8_C(  79), -INT8_C(  22), -INT8_C( 127), -INT8_C(  22), -INT8_C( 109),  INT8_C(  49),  INT8_C(  41),
        -INT8_C(   3),  INT8_C(  25),  INT8_C( 120),  INT8_C(   1), -INT8_C( 118), -INT8_C(  97),  INT8_C(   9), -INT8_C(  85),
        -INT8_C( 117), -INT8_C(   3),  INT8_C( 124), -INT8_C( 124),  INT8_C( 108), -INT8_C(  75),  INT8_C(  38), -INT8_C(   1),
         INT8_C(  35),  INT8_C( 120),  INT8_C(  41), -INT8_C(  50),  INT8_C(  97),  INT8_C(  15),  INT8_C(  75), -INT8_C(  71) } },
    { { -INT8_C( 127),  INT8_C(  35),  INT8_C( 118), -INT8_C(  29), -INT8_C(  37),  INT8_C(  61), -INT8_C( 113), -INT8_C(  67),
         INT8_C(  61),  INT8_C( 116),  INT8_C(   9),  INT8_C(  51), -INT8_C(  45), -INT8_C( 125), -INT8_C(  97), -INT8_C( 101),
             INT8_MIN,  INT8_C(  44), -INT8_C(  93),  INT8_C(  65), -INT8_C(  17), -INT8_C(  35),  INT8_C(  54),  INT8_C(  51),
         INT8_C(   1),  INT8_C(  20),  INT8_C(  74), -INT8_C(  94),  INT8_C(  97),  INT8_C(   1), -INT8_C(  43), -INT8_C(  30),
         INT8_C(  37),  INT8_C(  75), -INT8_C(  59),  INT8_C(   0), -INT8_C( 119),  INT8_C(  84), -INT8_C(  67), -INT8_C(  58),
        -INT8_C(  55), -INT8_C(  58), -INT8_C(   7), -INT8_C( 100),  INT8_C(  74), -INT8_C( 103),  INT8_C(  56), -INT8_C(  54),
        -INT8_C(  59), -INT8_C(  37),  INT8_C(  12), -INT8_C(  76), -INT8_C(  71),  INT8_C(  66), -INT8_C(  24), -INT8_C(  70),
         INT8_C(  86),  INT8_C(  50),  INT8_C(  92), -INT8_C(  73),  INT8_C(  52),  INT8_C(  49), -INT8_C( 103),  INT8_C(  89) },
      UINT64_C( 8992587514113515389),
      { -INT8_C(  36), -INT8_C(  59),  INT8_C(  25),  INT8_C(  38),  INT8_C(  94),  INT8_C(  81), -INT8_C(  15),  INT8_C(  36),
         INT8_C(  44), -INT8_C(   3), -INT8_C(  40), -INT8_C(  27),  INT8_C(  63), -INT8_C(  64), -INT8_C(  97), -INT8_C( 106),
        -INT8_C(  13), -INT8_C(   4),  INT8_C(  77),  INT8_C(  39),  INT8_C(  45), -INT8_C(  25),      INT8_MIN, -INT8_C(  86),
         INT8_C(  70), -INT8_C(  39), -INT8_C(  80), -INT8_C(   7), -INT8_C(  17),  INT8_C( 124),  INT8_C( 118), -INT8_C(  53),
         INT8_C(  66), -INT8_C( 113), -INT8_C(  14), -INT8_C(  96), -INT8_C(  32), -INT8_C(  29), -INT8_C(  60),  INT8_C(  12),
        -INT8_C(  32), -INT8_C(  99), -INT8_C(  14),  INT8_C(  31),  INT8_C(  93), -INT8_C( 111), -INT8_C(  75),  INT8_C(  80),
        -INT8_C( 115),  INT8_C(   3),  INT8_C( 119), -INT8_C(  69), -INT8_C(  22), -INT8_C(   9),  INT8_C( 101),  INT8_C(  48),
        -INT8_C(  48),  INT8_C(  22),  INT8_C(  41), -INT8_C(  65), -INT8_C( 110), -INT8_C(  97), -INT8_C( 117), -INT8_C(  44) },
      {  INT8_C(  46),  INT8_C( 125),  INT8_C( 117),  INT8_C(  14),  INT8_C(  96),  INT8_C(  57),  INT8_C(  27),  INT8_C(  64),
        -INT8_C(  42),  INT8_C(  13),  INT8_C(  95),  INT8_C(  52), -INT8_C(  98),  INT8_C(  21), -INT8_C( 124),  INT8_C(  44),
         INT8_C(  24), -INT8_C(   4), -INT8_C(  25),  INT8_C(   2), -INT8_C(  13),  INT8_C(  76),  INT8_C(  50), -INT8_C(  60),
         INT8_C(  98),  INT8_C(  91), -INT8_C( 125), -INT8_C(  11), -INT8_C(   5),  INT8_C(  14), -INT8_C(  55),  INT8_C(  41),
        -INT8_C( 117),  INT8_C(  62),  INT8_C(  56), -INT8_C(  21),  INT8_C( 120),  INT8_C(  83),  INT8_C(  43),  INT8_C(  78),
         INT8_C(  96), -INT8_C( 117), -INT8_C( 126), -INT8_C(   2), -INT8_C(  96),  INT8_C(   7),  INT8_C(  42), -INT8_C(  72),
         INT8_C(   3),  INT8_C(  17), -INT8_C(  70), -INT8_C(  10),  INT8_C(  94), -INT8_C(  20), -INT8_C(  70), -INT8_C(  64),
         INT8_C(  71),  INT8_C(  62), -INT8_C(  75),  INT8_C(  66),  INT8_C(  76),      INT8_MAX,  INT8_C( 108), -INT8_C(  40) },
      {  INT8_C(  10),  INT8_C(  35), -INT8_C( 114),  INT8_C(  52), -INT8_C(  66), -INT8_C( 118),  INT8_C(  12), -INT8_C(  67),
         INT8_C(   2),  INT8_C(  10),  INT8_C(  55),  INT8_C(  25), -INT8_C(  35), -INT8_C( 125),  INT8_C(  35), -INT8_C( 101),
         INT8_C(  11),  INT8_C(  44), -INT8_C(  93),  INT8_C(  41),  INT8_C(  32), -INT8_C(  35), -INT8_C(  78),  INT8_C(  51),
         INT8_C(   1),  INT8_C(  52),  INT8_C(  51), -INT8_C(  94),  INT8_C(  97),  INT8_C(   1), -INT8_C(  43), -INT8_C(  30),
        -INT8_C(  51), -INT8_C(  51), -INT8_C(  59),  INT8_C(   0),  INT8_C(  88),  INT8_C(  54), -INT8_C(  67),  INT8_C(  90),
        -INT8_C(  55),  INT8_C(  40),  INT8_C( 116), -INT8_C( 100), -INT8_C(   3), -INT8_C( 103),  INT8_C(  56), -INT8_C(  54),
        -INT8_C(  59), -INT8_C(  37),  INT8_C(  49), -INT8_C(  79), -INT8_C(  71),  INT8_C(  66),  INT8_C(  31), -INT8_C(  16),
         INT8_C(  86),  INT8_C(  50), -INT8_C(  34),  INT8_C(   1), -INT8_C(  34),  INT8_C(  30), -INT8_C(   9),  INT8_C(  89) } },
    { { -INT8_C(  67), -INT8_C(  92), -INT8_C(  61),  INT8_C(  53), -INT8_C(   9), -INT8_C(  17), -INT8_C( 124),  INT8_C(  87),
         INT8_C( 122),  INT8_C(   6),  INT8_C(  85),  INT8_C(  26),  INT8_C(  13),      INT8_MIN, -INT8_C(  46),  INT8_C(  16),
        -INT8_C( 111), -INT8_C( 116),  INT8_C(   7), -INT8_C(  17),  INT8_C( 120), -INT8_C(  63), -INT8_C(  80), -INT8_C(  65),
        -INT8_C(   1),  INT8_C( 101),  INT8_C(   2),  INT8_C(  76), -INT8_C(  28),  INT8_C( 110),  INT8_C(  36), -INT8_C(  94),
         INT8_C(  18), -INT8_C(  25), -INT8_C(  41),  INT8_C(   9), -INT8_C(  42),  INT8_C(  91),  INT8_C(  96),  INT8_C(  80),
         INT8_C(  98), -INT8_C(  75),  INT8_C( 106),  INT8_C( 111),  INT8_C(  53),  INT8_C(  60),      INT8_MIN, -INT8_C(  57),
        -INT8_C(  56), -INT8_C( 121), -INT8_C(  74),  INT8_C(  64),  INT8_C(  72),  INT8_C( 102),  INT8_C(   0),  INT8_C(  72),
        -INT8_C(  52),  INT8_C(   2), -INT8_C( 108), -INT8_C(  80),  INT8_C( 112), -INT8_C(  72),  INT8_C(  82), -INT8_C( 126) },
      UINT64_C(16701295226602072735),
      { -INT8_C(  96),  INT8_C(  49),  INT8_C(  87), -INT8_C(  42),  INT8_C( 109), -INT8_C(  41), -INT8_C(  99),  INT8_C(  54),
         INT8_C(  94),  INT8_C(  83),  INT8_C( 118), -INT8_C(  90), -INT8_C(  70),  INT8_C( 118), -INT8_C(  18), -INT8_C( 122),
         INT8_C( 120), -INT8_C( 126),  INT8_C(  54), -INT8_C(  24),  INT8_C(  58), -INT8_C( 119),  INT8_C( 106), -INT8_C(  38),
        -INT8_C(  77), -INT8_C(  11),  INT8_C(  80),  INT8_C(  56), -INT8_C(  32),  INT8_C(  22),  INT8_C(  32), -INT8_C( 127),
         INT8_C(  71),  INT8_C( 119),  INT8_C(  87), -INT8_C(  75),  INT8_C(  78), -INT8_C(  12), -INT8_C(  21), -INT8_C(  84),
         INT8_C(  71),  INT8_C(  97),  INT8_C(  82),  INT8_C(   1), -INT8_C(  40),  INT8_C(  65), -INT8_C( 121),  INT8_C(  80),
        -INT8_C(  61), -INT8_C(  66),  INT8_C(  57), -INT8_C(   2),  INT8_C(  71), -INT8_C(  93), -INT8_C(  40), -INT8_C(   6),
        -INT8_C( 103),  INT8_C(  40),  INT8_C(  50),  INT8_C( 121),  INT8_C(  62),  INT8_C(  82), -INT8_C(   6), -INT8_C( 122) },
      { -INT8_C(  55),  INT8_C(  81),  INT8_C(  59),  INT8_C(  23),  INT8_C(  69),  INT8_C(  38), -INT8_C(  61), -INT8_C( 115),
        -INT8_C( 121),  INT8_C(  22), -INT8_C( 114),  INT8_C(  95),  INT8_C(  87),  INT8_C(  22), -INT8_C(  80),  INT8_C(  26),
        -INT8_C(  44), -INT8_C(  23),  INT8_C(  24),  INT8_C(  27), -INT8_C( 116), -INT8_C(  16),  INT8_C(  21),  INT8_C(  37),
         INT8_C(  24),  INT8_C(  71), -INT8_C(  97),  INT8_C(  87), -INT8_C( 102), -INT8_C( 103), -INT8_C(  35),  INT8_C(  99),
        -INT8_C(  21),  INT8_C(  24),  INT8_C( 123),  INT8_C(  48),  INT8_C(  62),  INT8_C(  62), -INT8_C(  67), -INT8_C(  59),
         INT8_C(  84),  INT8_C(  76),  INT8_C(  37), -INT8_C(  85),  INT8_C(  98), -INT8_C(  43), -INT8_C(  58),  INT8_C(  54),
        -INT8_C(  66), -INT8_C(  34),  INT8_C(  81),  INT8_C(  74), -INT8_C(  49),  INT8_C( 102),  INT8_C( 112), -INT8_C(  25),
        -INT8_C(  83),  INT8_C(  15),  INT8_C(  62),  INT8_C(  71), -INT8_C(  88),  INT8_C(  27), -INT8_C(  85), -INT8_C( 109) },
      {  INT8_C( 105), -INT8_C( 126), -INT8_C( 110), -INT8_C(  19), -INT8_C(  78), -INT8_C(  17), -INT8_C( 124), -INT8_C(  61),
         INT8_C( 122),  INT8_C( 105),  INT8_C(  85),  INT8_C(   5),  INT8_C(  13), -INT8_C( 116), -INT8_C(  46),  INT8_C(  16),
         INT8_C(  76),  INT8_C( 107),  INT8_C(   7),  INT8_C(   3),  INT8_C( 120), -INT8_C(  63), -INT8_C(  80), -INT8_C(   1),
        -INT8_C(   1),  INT8_C(  60), -INT8_C(  17),  INT8_C(  76),  INT8_C( 122), -INT8_C(  81), -INT8_C(   3), -INT8_C(  94),
         INT8_C(  50), -INT8_C(  25), -INT8_C(  46),  INT8_C(   9), -INT8_C(  42),  INT8_C(  91),  INT8_C(  96),  INT8_C( 113),
        -INT8_C( 101), -INT8_C(  83),  INT8_C( 106), -INT8_C(  84),  INT8_C(  53),  INT8_C(  22),  INT8_C(  77), -INT8_C( 122),
        -INT8_C(  56), -INT8_C( 100), -INT8_C( 118),  INT8_C(  64),  INT8_C(  72),  INT8_C( 102),  INT8_C(  72), -INT8_C(  31),
         INT8_C(  70),  INT8_C(  55),  INT8_C( 112), -INT8_C(  80),  INT8_C( 112),  INT8_C( 109), -INT8_C(  91),  INT8_C(  25) } },
    { {  INT8_C(  51),  INT8_C(  38), -INT8_C(  60),  INT8_C( 113),  INT8_C( 100), -INT8_C( 127),  INT8_C(  55), -INT8_C(  71),
        -INT8_C(  51),  INT8_C(  92),  INT8_C( 100),  INT8_C(  47),  INT8_C(  49),  INT8_C(  42),  INT8_C( 101), -INT8_C(  17),
         INT8_C(   9), -INT8_C(  74),  INT8_C(  57), -INT8_C(  40),  INT8_C(  28), -INT8_C(  87), -INT8_C(  65), -INT8_C(  54),
        -INT8_C(  72), -INT8_C(   2),  INT8_C(  17),  INT8_C(  97),  INT8_C(  25), -INT8_C(  68), -INT8_C(  12),  INT8_C(  77),
        -INT8_C(  30), -INT8_C(  72), -INT8_C(  66),  INT8_C(  71),  INT8_C(  58), -INT8_C(  11),  INT8_C(   0),  INT8_C(   7),
         INT8_C(  81),  INT8_C( 100),  INT8_C(  55), -INT8_C( 126), -INT8_C( 113), -INT8_C( 100),  INT8_C( 113), -INT8_C( 104),
         INT8_C(  83), -INT8_C(  85),  INT8_C( 112),  INT8_C( 111),  INT8_C(  84),  INT8_C(  47),  INT8_C(  57),  INT8_C(  13),
         INT8_C(  45),  INT8_C(  75),  INT8_C( 110),  INT8_C(  71),  INT8_C(   7),  INT8_C(  98), -INT8_C( 108), -INT8_C(  22) },
      UINT64_C(11050761772397056539),
      { -INT8_C( 107), -INT8_C( 109),  INT8_C(  28),  INT8_C(  36),  INT8_C(  48), -INT8_C( 115), -INT8_C(  68), -INT8_C( 125),
         INT8_C(  56),  INT8_C(  44), -INT8_C(  14), -INT8_C( 115),  INT8_C(  92),  INT8_C(  44), -INT8_C( 102), -INT8_C( 119),
         INT8_C( 119),  INT8_C(   8), -INT8_C(  48),  INT8_C( 126),  INT8_C( 106),  INT8_C( 100),  INT8_C( 104), -INT8_C( 123),
        -INT8_C(  73), -INT8_C( 103), -INT8_C(  38), -INT8_C(   1), -INT8_C(  54),  INT8_C(  55), -INT8_C( 104),  INT8_C(  96),
        -INT8_C(  54), -INT8_C(  76), -INT8_C( 124), -INT8_C(   6),  INT8_C(  66),  INT8_C(  65),  INT8_C( 125),  INT8_C( 122),
         INT8_C( 109),  INT8_C( 112),  INT8_C(   7), -INT8_C(  55), -INT8_C( 100), -INT8_C(  95),  INT8_C(  83),  INT8_C(  19),
        -INT8_C(  87),  INT8_C(  35), -INT8_C( 111),  INT8_C(  20), -INT8_C( 120), -INT8_C(   6), -INT8_C( 103),  INT8_C(  63),
        -INT8_C( 109),  INT8_C( 116),  INT8_C(  62),  INT8_C(  94), -INT8_C(  85), -INT8_C(  42), -INT8_C(  66),  INT8_C( 117) },
      { -INT8_C( 117),  INT8_C(  66),  INT8_C( 112), -INT8_C(  51), -INT8_C( 125), -INT8_C(  19),  INT8_C(  71), -INT8_C(  15),
         INT8_C(  93),  INT8_C(  79), -INT8_C(  70), -INT8_C(   7), -INT8_C(  16),  INT8_C(  13),  INT8_C(  12), -INT8_C( 102),
         INT8_C(  49), -INT8_C(  98), -INT8_C(  82), -INT8_C(  71), -INT8_C( 104),  INT8_C(  71), -INT8_C(   8),  INT8_C(  43),
        -INT8_C(  69),  INT8_C(  54), -INT8_C( 119),  INT8_C( 102),  INT8_C(  12),  INT8_C(  71), -INT8_C(  36), -INT8_C( 105),
        -INT8_C( 118),  INT8_C(  76),  INT8_C( 100),  INT8_C(  13),  INT8_C(  57), -INT8_C(  84), -INT8_C(   2), -INT8_C( 105),
        -INT8_C(   5), -INT8_C(  71), -INT8_C( 112), -INT8_C(  21), -INT8_C(  58), -INT8_C(  99), -INT8_C( 123), -INT8_C(   9),
         INT8_C(  59),  INT8_C(  51), -INT8_C(  80), -INT8_C(  45),  INT8_C( 123), -INT8_C(  88), -INT8_C(   2),  INT8_C(  54),
        -INT8_C(  34), -INT8_C( 120), -INT8_C(  99), -INT8_C(  21), -INT8_C(  49),  INT8_C( 121), -INT8_C( 126),  INT8_C(  89) },
      {  INT8_C(  32), -INT8_C(  43), -INT8_C(  60), -INT8_C(  15), -INT8_C(  77), -INT8_C( 127),  INT8_C(  55), -INT8_C(  71),
        -INT8_C(  51),  INT8_C( 123),  INT8_C( 100),  INT8_C(  47),  INT8_C(  76),  INT8_C(  42), -INT8_C(  90), -INT8_C(  17),
        -INT8_C(  88), -INT8_C(  74),  INT8_C(  57), -INT8_C(  40),  INT8_C(   2), -INT8_C(  85), -INT8_C(  65), -INT8_C(  54),
         INT8_C( 114), -INT8_C(   2),  INT8_C(  99),  INT8_C(  97), -INT8_C(  42), -INT8_C(  68),  INT8_C( 116),  INT8_C(  77),
        -INT8_C(  30), -INT8_C(  72), -INT8_C(  66),  INT8_C(   7),  INT8_C(  58), -INT8_C(  11),  INT8_C( 123),  INT8_C(   7),
         INT8_C( 104),  INT8_C( 100),  INT8_C(  55), -INT8_C( 126),  INT8_C(  98),  INT8_C(  62),  INT8_C( 113), -INT8_C( 104),
         INT8_C(  83), -INT8_C(  85),  INT8_C(  65), -INT8_C(  25),  INT8_C(   3),  INT8_C(  47), -INT8_C( 105),  INT8_C(  13),
         INT8_C( 113),  INT8_C(  75),  INT8_C( 110),  INT8_C(  73),  INT8_C( 122),  INT8_C(  98), -INT8_C( 108), -INT8_C(  50) } },
    { { -INT8_C(  59), -INT8_C(  25),  INT8_C( 103), -INT8_C(   2), -INT8_C( 109),  INT8_C( 101), -INT8_C( 107), -INT8_C( 114),
         INT8_C(  30),  INT8_C(  38),  INT8_C( 121), -INT8_C(  27), -INT8_C(  61), -INT8_C(   1), -INT8_C(  36), -INT8_C(   2),
         INT8_C(  50), -INT8_C( 115), -INT8_C(  47), -INT8_C(  83),  INT8_C(  53), -INT8_C(  49), -INT8_C(  28),  INT8_C(  20),
         INT8_C(  87), -INT8_C( 127), -INT8_C(   1),  INT8_C(  39), -INT8_C(   6), -INT8_C( 127),      INT8_MIN, -INT8_C(  65),
         INT8_C( 104), -INT8_C(  25), -INT8_C(  67), -INT8_C(   5),  INT8_C(  77),  INT8_C(  83), -INT8_C( 119),  INT8_C( 107),
         INT8_C( 121),  INT8_C(   3),  INT8_C(  80),  INT8_C(  60),  INT8_C(   2),  INT8_C(  45),  INT8_C(  58),  INT8_C(  52),
        -INT8_C(  70),  INT8_C(  11), -INT8_C(  30), -INT8_C(  17), -INT8_C(  38), -INT8_C(  58),  INT8_C(   3),  INT8_C(  50),
         INT8_C(  71),  INT8_C(   2),  INT8_C(  89),  INT8_C(  65), -INT8_C( 124), -INT8_C(  39),  INT8_C(   0), -INT8_C(  20) },
      UINT64_C( 9906073169161665985),
      {  INT8_C( 116), -INT8_C(  54), -INT8_C(  59),  INT8_C( 118), -INT8_C(   9), -INT8_C(   1), -INT8_C(  85), -INT8_C(  79),
         INT8_C(  10), -INT8_C( 115), -INT8_C(  96), -INT8_C(  27),  INT8_C(  83), -INT8_C(  92),  INT8_C(  23), -INT8_C( 102),
        -INT8_C(  90),  INT8_C( 112), -INT8_C(  37),  INT8_C(  42),  INT8_C(  73), -INT8_C(  37),  INT8_C(  23),  INT8_C(  10),
        -INT8_C( 104), -INT8_C(   1),  INT8_C(  24), -INT8_C(  87),  INT8_C( 112), -INT8_C( 110),  INT8_C(  50), -INT8_C(  27),
         INT8_C(  92), -INT8_C(   8),  INT8_C(  91),  INT8_C(  83), -INT8_C(   9),  INT8_C(   6),  INT8_C(   4),  INT8_C(   2),
        -INT8_C( 109), -INT8_C(  92), -INT8_C(  25), -INT8_C(  26),  INT8_C(  72), -INT8_C(   2),      INT8_MIN, -INT8_C(  17),
         INT8_C( 110),  INT8_C(  91),  INT8_C(  25), -INT8_C(  73),  INT8_C(  54),  INT8_C(  48), -INT8_C(  62), -INT8_C(  49),
         INT8_C(  47), -INT8_C(  38),  INT8_C( 120), -INT8_C(  96),  INT8_C( 108), -INT8_C(  86), -INT8_C( 123), -INT8_C(  56) },
      { -INT8_C(  94), -INT8_C(  32),  INT8_C(  27), -INT8_C( 102), -INT8_C(  25),  INT8_C(  31), -INT8_C( 100),  INT8_C( 122),
        -INT8_C(  60), -INT8_C( 125),  INT8_C(  97),  INT8_C(  12), -INT8_C( 127), -INT8_C(  31), -INT8_C(   5), -INT8_C(  17),
         INT8_C(  61),  INT8_C(  21), -INT8_C(  90),  INT8_C( 115),  INT8_C(  69),  INT8_C( 104),  INT8_C(  66),  INT8_C( 117),
         INT8_C(  67), -INT8_C(  70),  INT8_C(  21), -INT8_C(  81),  INT8_C( 101), -INT8_C( 102),  INT8_C( 120),  INT8_C(   7),
         INT8_C( 122), -INT8_C( 109), -INT8_C(  95),  INT8_C(  97), -INT8_C(  77),  INT8_C(  61), -INT8_C(  36),  INT8_C( 119),
        -INT8_C(  64),  INT8_C(  61), -INT8_C( 125),  INT8_C(  65),  INT8_C(  30),      INT8_MAX,  INT8_C(  48),  INT8_C(  91),
        -INT8_C( 108), -INT8_C(  41), -INT8_C(  49), -INT8_C(  39),  INT8_C(  63),  INT8_C(  17),  INT8_C(  78), -INT8_C( 126),
        -INT8_C(  52),  INT8_C(  99),  INT8_C(  50),  INT8_C(  49), -INT8_C(   3), -INT8_C(  86),  INT8_C(  56),  INT8_C( 120) },
      {  INT8_C(  22), -INT8_C(  25),  INT8_C( 103), -INT8_C(   2), -INT8_C( 109),  INT8_C( 101),  INT8_C(  71),  INT8_C(  43),
        -INT8_C(  50),  INT8_C(  38),  INT8_C(   1), -INT8_C(  15), -INT8_C(  44), -INT8_C( 123), -INT8_C(  36), -INT8_C( 119),
         INT8_C(  50), -INT8_C( 115), -INT8_C(  47), -INT8_C(  99),  INT8_C(  53),  INT8_C(  67),  INT8_C(  89),      INT8_MAX,
         INT8_C(  87), -INT8_C(  71),  INT8_C(  45),  INT8_C(  88), -INT8_C(   6), -INT8_C( 127),      INT8_MIN, -INT8_C(  65),
         INT8_C( 104), -INT8_C(  25), -INT8_C(  67), -INT8_C(   5), -INT8_C(  86),  INT8_C(  83), -INT8_C( 119),  INT8_C( 107),
         INT8_C(  83),  INT8_C(   3),  INT8_C(  80),  INT8_C(  60),  INT8_C( 102),  INT8_C( 125), -INT8_C(  80),  INT8_C(  52),
         INT8_C(   2),  INT8_C(  11), -INT8_C(  30), -INT8_C( 112),  INT8_C( 117),  INT8_C(  65),  INT8_C(  16),  INT8_C(  50),
        -INT8_C(   5),  INT8_C(   2),  INT8_C(  89), -INT8_C(  47), -INT8_C( 124), -INT8_C(  39),  INT8_C(   0),  INT8_C(  64) } },
    { {  INT8_C(  61), -INT8_C(  38), -INT8_C(  39), -INT8_C(  16),  INT8_C(  23), -INT8_C(  75),  INT8_C( 103), -INT8_C(  40),
        -INT8_C(  14), -INT8_C(  21),  INT8_C(  25),  INT8_C(  17),  INT8_C( 106),  INT8_C(  74),  INT8_C( 108), -INT8_C(   2),
         INT8_C(  33),  INT8_C(  59), -INT8_C(  41),  INT8_C(  96),  INT8_C(  77),  INT8_C(  38), -INT8_C(  29),  INT8_C(  25),
        -INT8_C( 119),  INT8_C(  21),  INT8_C(  74), -INT8_C( 121), -INT8_C(  65), -INT8_C( 126), -INT8_C(   1), -INT8_C(   4),
         INT8_C(  92), -INT8_C(  40), -INT8_C(  19),  INT8_C( 116), -INT8_C( 114),  INT8_C(  84),  INT8_C(  76),      INT8_MIN,
         INT8_C(  63),  INT8_C( 101), -INT8_C( 111), -INT8_C(  87), -INT8_C(  81), -INT8_C(   2), -INT8_C(  89), -INT8_C(  48),
         INT8_C(  57),      INT8_MAX,  INT8_C(  49), -INT8_C( 122), -INT8_C(  91),  INT8_C(  20), -INT8_C(  97),  INT8_C(  46),
         INT8_C(  41), -INT8_C(  23), -INT8_C(  75), -INT8_C(  24),  INT8_C( 108), -INT8_C(  76), -INT8_C(  28), -INT8_C(  56) },
      UINT64_C( 7321595316467978637),
      { -INT8_C(  18),  INT8_C(  45),  INT8_C(  15), -INT8_C(  99),  INT8_C(  43), -INT8_C(  74),  INT8_C( 110),  INT8_C( 100),
         INT8_C(  53), -INT8_C(  97), -INT8_C(  21), -INT8_C(  38), -INT8_C(  77), -INT8_C( 118),  INT8_C(   9), -INT8_C(  36),
         INT8_C( 116), -INT8_C(  66), -INT8_C(  60), -INT8_C(  32),  INT8_C( 115), -INT8_C(  88), -INT8_C(  88),  INT8_C(   0),
         INT8_C( 122), -INT8_C(  27),  INT8_C(  27), -INT8_C(  96),  INT8_C( 109), -INT8_C(  74),  INT8_C(   5),  INT8_C(  91),
        -INT8_C(  29),  INT8_C(  20), -INT8_C(   7),  INT8_C(  14), -INT8_C(  53),  INT8_C( 103),  INT8_C( 115),  INT8_C(   0),
         INT8_C(   6),  INT8_C(  94), -INT8_C(  37), -INT8_C(  71), -INT8_C(  24), -INT8_C(  28), -INT8_C( 107),  INT8_C(  92),
        -INT8_C(  94),  INT8_C(  89),  INT8_C(  60),  INT8_C(  21),  INT8_C(   1), -INT8_C(  27),  INT8_C(  21),  INT8_C( 123),
        -INT8_C(  54),  INT8_C(  48),  INT8_C(  27),  INT8_C(  55), -INT8_C(  25),  INT8_C(  33), -INT8_C( 109), -INT8_C(  54) },
      {  INT8_C(  53), -INT8_C( 116), -INT8_C(  39),  INT8_C(   0), -INT8_C(  13),  INT8_C(  76),  INT8_C(   1), -INT8_C(   7),
        -INT8_C(  86), -INT8_C(  36), -INT8_C(  78), -INT8_C( 110), -INT8_C(  64),  INT8_C(  71), -INT8_C(  17),  INT8_C(  98),
        -INT8_C(  96),  INT8_C(  43),  INT8_C( 120), -INT8_C(  95),  INT8_C(  16), -INT8_C( 115),  INT8_C(  29), -INT8_C(  38),
        -INT8_C(  66),  INT8_C(  56),  INT8_C(  18), -INT8_C(  91),  INT8_C(  89), -INT8_C(  91),  INT8_C( 111), -INT8_C( 113),
         INT8_C(  49),  INT8_C(  72), -INT8_C( 113),  INT8_C(  36), -INT8_C( 108), -INT8_C( 112),  INT8_C(  29),  INT8_C(  62),
         INT8_C( 108), -INT8_C(  49), -INT8_C(  47),  INT8_C(  44),  INT8_C(  22), -INT8_C(  64), -INT8_C( 113), -INT8_C(  74),
        -INT8_C(  21),  INT8_C(   7),  INT8_C(  87), -INT8_C(   4), -INT8_C( 108),  INT8_C( 116), -INT8_C(  42),  INT8_C(  82),
        -INT8_C(  83), -INT8_C(  24), -INT8_C(   9),  INT8_C(   6), -INT8_C( 115),  INT8_C( 103), -INT8_C( 107), -INT8_C(  66) },
      {  INT8_C(  35), -INT8_C(  38), -INT8_C(  24), -INT8_C(  99),  INT8_C(  23), -INT8_C(  75),  INT8_C( 103),  INT8_C(  93),
        -INT8_C(  33), -INT8_C(  21),  INT8_C(  25),  INT8_C(  17),  INT8_C( 115),  INT8_C(  74), -INT8_C(   8),  INT8_C(  62),
         INT8_C(  33),  INT8_C(  59),  INT8_C(  60), -INT8_C( 127), -INT8_C( 125),  INT8_C(  53), -INT8_C(  29),  INT8_C(  25),
         INT8_C(  56),  INT8_C(  29),  INT8_C(  74),  INT8_C(  69), -INT8_C(  58), -INT8_C( 126), -INT8_C(   1), -INT8_C(   4),
         INT8_C(  92),  INT8_C(  92), -INT8_C( 120),  INT8_C( 116), -INT8_C( 114), -INT8_C(   9),  INT8_C(  76),      INT8_MIN,
         INT8_C(  63),  INT8_C( 101), -INT8_C( 111), -INT8_C(  27), -INT8_C(  81), -INT8_C(   2), -INT8_C(  89),  INT8_C(  18),
        -INT8_C( 115),  INT8_C(  96),  INT8_C(  49),  INT8_C(  17), -INT8_C( 107),  INT8_C(  20), -INT8_C(  97), -INT8_C(  51),
         INT8_C( 119), -INT8_C(  23),  INT8_C(  18), -INT8_C(  24),  INT8_C( 108), -INT8_C( 120),  INT8_C(  40), -INT8_C(  56) } },
    { { -INT8_C(  81),  INT8_C(  37), -INT8_C(  30),  INT8_C(  68), -INT8_C(  75), -INT8_C(   1), -INT8_C( 126),  INT8_C(  34),
        -INT8_C(  50),  INT8_C(  83),  INT8_C(  78), -INT8_C(  28),  INT8_C(  19), -INT8_C(  35), -INT8_C( 102), -INT8_C(   1),
        -INT8_C(  28), -INT8_C(  14), -INT8_C(   5),  INT8_C( 121),  INT8_C( 102), -INT8_C(  47), -INT8_C(  53),  INT8_C(  19),
        -INT8_C(  70), -INT8_C(  61),  INT8_C(  26),  INT8_C(  71),  INT8_C(  42), -INT8_C(  81),  INT8_C(   6), -INT8_C(  39),
        -INT8_C(  44), -INT8_C(  24),  INT8_C(  29), -INT8_C( 118), -INT8_C(  24), -INT8_C(  96), -INT8_C(  84), -INT8_C(  74),
        -INT8_C(  13), -INT8_C(   6), -INT8_C( 101),  INT8_C(   7), -INT8_C(  40),  INT8_C(  53),  INT8_C(   6), -INT8_C(  68),
         INT8_C(  39),  INT8_C(   1),  INT8_C(  53), -INT8_C( 114), -INT8_C(  46),  INT8_C(   1), -INT8_C(  95), -INT8_C( 116),
        -INT8_C(  60), -INT8_C(  69), -INT8_C(  44), -INT8_C(  18),  INT8_C( 107), -INT8_C(  38), -INT8_C(  57),  INT8_C(  63) },
      UINT64_C( 8674343574248744386),
      {  INT8_C( 112), -INT8_C(   4),      INT8_MAX,  INT8_C(  72),  INT8_C(  49), -INT8_C( 123),  INT8_C(   4),  INT8_C(  89),
        -INT8_C( 122),  INT8_C(  58), -INT8_C(  25),  INT8_C(  89),  INT8_C(  59), -INT8_C( 120), -INT8_C(  27), -INT8_C(   1),
         INT8_C(  68), -INT8_C(  71), -INT8_C(  19), -INT8_C(  81), -INT8_C( 109), -INT8_C(  76), -INT8_C(  18),  INT8_C(  86),
        -INT8_C( 103), -INT8_C(  72),  INT8_C(   0),  INT8_C(  30),  INT8_C(  45),  INT8_C(  97), -INT8_C( 105), -INT8_C(  99),
         INT8_C(  93),  INT8_C(  22), -INT8_C(  27), -INT8_C( 113), -INT8_C( 100), -INT8_C(  22), -INT8_C(  24),  INT8_C(  34),
         INT8_C(  36), -INT8_C(  49),  INT8_C( 123),  INT8_C(  95),  INT8_C(  87),  INT8_C(  97),  INT8_C(  94), -INT8_C( 101),
         INT8_C(  26),  INT8_C(  75),  INT8_C(  74), -INT8_C(  82), -INT8_C(   1),  INT8_C(  57),  INT8_C(   4), -INT8_C( 103),
        -INT8_C(  15),  INT8_C(   4), -INT8_C(  73),  INT8_C(  30),  INT8_C( 102),  INT8_C(  78), -INT8_C(  68), -INT8_C(  61) },
      {  INT8_C( 101), -INT8_C(  95),  INT8_C(  82),  INT8_C(   1), -INT8_C( 117),  INT8_C(  58),  INT8_C(  35), -INT8_C(  81),
         INT8_C(   9), -INT8_C(  97),  INT8_C(  14),  INT8_C(  97),  INT8_C(   0),  INT8_C( 108), -INT8_C(   4),  INT8_C(  26),
        -INT8_C(  73),  INT8_C(  71), -INT8_C(  56), -INT8_C(  73),      INT8_MIN, -INT8_C(  52),  INT8_C(  80),  INT8_C( 113),
        -INT8_C(  47),  INT8_C(   7), -INT8_C( 113),  INT8_C(  55),  INT8_C(  86),  INT8_C(  75), -INT8_C(   6), -INT8_C(  69),
        -INT8_C(  19),  INT8_C(  77), -INT8_C(  68),  INT8_C( 120), -INT8_C( 121), -INT8_C(  33),  INT8_C(  40), -INT8_C( 111),
         INT8_C( 126),  INT8_C(  54), -INT8_C(  14),  INT8_C( 126), -INT8_C(  93), -INT8_C(  18), -INT8_C( 103),  INT8_C(  90),
         INT8_C(  53),  INT8_C(  97),  INT8_C(  17), -INT8_C(  75),  INT8_C(  46),  INT8_C(  97),  INT8_C(  38), -INT8_C(   1),
         INT8_C( 105), -INT8_C(  74),  INT8_C(  54), -INT8_C(  65),  INT8_C(   1),  INT8_C(  48),  INT8_C( 122), -INT8_C(  18) },
      { -INT8_C(  81), -INT8_C(  99), -INT8_C(  30),  INT8_C(  68), -INT8_C(  75), -INT8_C(   1),  INT8_C(  39),  INT8_C(   8),
        -INT8_C( 113),  INT8_C(  83), -INT8_C(  11), -INT8_C(  28),  INT8_C(  19), -INT8_C(  12), -INT8_C(  31),  INT8_C(  25),
        -INT8_C(   5), -INT8_C(  14), -INT8_C(   5),  INT8_C( 102),  INT8_C( 102), -INT8_C(  47),  INT8_C(  62), -INT8_C(  57),
        -INT8_C(  70), -INT8_C(  65),  INT8_C(  26),  INT8_C(  85),  INT8_C(  42), -INT8_C(  84),  INT8_C(   6),  INT8_C(  88),
         INT8_C(  74), -INT8_C(  24), -INT8_C(  95), -INT8_C( 118), -INT8_C(  24), -INT8_C(  96), -INT8_C(  84), -INT8_C(  77),
        -INT8_C(  94), -INT8_C(   6),  INT8_C( 109),  INT8_C(   7), -INT8_C(   6),  INT8_C(  79), -INT8_C(   9), -INT8_C(  68),
         INT8_C(  79),  INT8_C(   1),  INT8_C(  53), -INT8_C( 114), -INT8_C(  46), -INT8_C( 102),  INT8_C(  42), -INT8_C( 116),
        -INT8_C(  60), -INT8_C(  69), -INT8_C(  44), -INT8_C(  35),  INT8_C( 103),  INT8_C( 126),  INT8_C(  54),  INT8_C(  63) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi8(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_add_epi8(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_add_epi8");
   easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_add_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask64 k;
    const int8_t a[64];
    const int8_t b[64];
    const int8_t r[64];
  } test_vec[] = {
    { UINT64_C( 9122680976650596708),
      {  INT8_C(  68), -INT8_C(  57),  INT8_C(  11), -INT8_C( 119), -INT8_C( 113),  INT8_C( 124),  INT8_C(  91),  INT8_C(   1),
         INT8_C( 112), -INT8_C( 126),  INT8_C(  44), -INT8_C(  65),  INT8_C(  76), -INT8_C(  60), -INT8_C(  30),  INT8_C(  88),
        -INT8_C(   8),  INT8_C(  72), -INT8_C(  34), -INT8_C( 108),  INT8_C(   2),  INT8_C(  65), -INT8_C(  12),  INT8_C( 102),
         INT8_C(  50),  INT8_C( 109),  INT8_C(  10),  INT8_C(  54), -INT8_C(  77), -INT8_C(  91), -INT8_C(  76), -INT8_C(   9),
         INT8_C( 108), -INT8_C(  65),      INT8_MIN, -INT8_C(   5),  INT8_C(  60), -INT8_C(  36), -INT8_C(   4), -INT8_C(  84),
         INT8_C(  94),  INT8_C(  40),  INT8_C( 108), -INT8_C(  86), -INT8_C(  20),  INT8_C(  78),  INT8_C(   2), -INT8_C(  27),
        -INT8_C( 106), -INT8_C(  32),  INT8_C( 121), -INT8_C( 104),  INT8_C(  34),  INT8_C( 110), -INT8_C(   2),  INT8_C(  84),
        -INT8_C(  37),  INT8_C(   8), -INT8_C( 118), -INT8_C( 114), -INT8_C(  83),  INT8_C(  63), -INT8_C( 122),  INT8_C(  25) },
      { -INT8_C(   2),  INT8_C(   6),  INT8_C(  20),  INT8_C(  58), -INT8_C(  30),  INT8_C(  16), -INT8_C(  25),  INT8_C(  65),
         INT8_C(  56),  INT8_C(  83), -INT8_C(  21),  INT8_C(  37), -INT8_C(  95), -INT8_C(  18),  INT8_C(  10),  INT8_C(  55),
        -INT8_C(  50), -INT8_C( 125), -INT8_C(  49), -INT8_C(  16), -INT8_C(  15), -INT8_C(  51),  INT8_C(  69), -INT8_C(  52),
        -INT8_C(  43), -INT8_C(  49),  INT8_C(  91), -INT8_C( 125),  INT8_C(  14), -INT8_C(  31), -INT8_C( 100),  INT8_C(  13),
        -INT8_C(  25), -INT8_C(  79),  INT8_C(  71), -INT8_C(  54), -INT8_C(  63),  INT8_C(  46),  INT8_C(  11), -INT8_C(   6),
        -INT8_C( 127), -INT8_C(  10),  INT8_C(  31),  INT8_C(  34), -INT8_C(  28),  INT8_C(  41),  INT8_C(  89), -INT8_C(  77),
        -INT8_C(  84),  INT8_C(  40), -INT8_C(  93), -INT8_C(  98), -INT8_C(  11), -INT8_C(  24),  INT8_C( 106), -INT8_C(  53),
        -INT8_C(  72), -INT8_C(  59),  INT8_C(  78), -INT8_C(  58), -INT8_C(  90), -INT8_C(  22), -INT8_C(  45), -INT8_C( 114) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(  31),  INT8_C(   0),  INT8_C(   0), -INT8_C( 116),  INT8_C(  66),  INT8_C(   0),
        -INT8_C(  88),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  19), -INT8_C(  78), -INT8_C(  20), -INT8_C( 113),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 124), -INT8_C(  13),  INT8_C(  14),  INT8_C(  57),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C( 101),  INT8_C(   0),  INT8_C(   0), -INT8_C( 122),  INT8_C(   0),  INT8_C(   4),
         INT8_C(  83),  INT8_C( 112),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  30), -INT8_C( 117),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  91),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   8),  INT8_C(   0),  INT8_C(  54),  INT8_C(  23),  INT8_C(   0),  INT8_C(   0),  INT8_C(  31),
         INT8_C(   0), -INT8_C(  51), -INT8_C(  40),  INT8_C(  84),  INT8_C(  83),  INT8_C(  41),  INT8_C(  89),  INT8_C(   0) } },
    { UINT64_C(14652289079423015835),
      {  INT8_C(  89),  INT8_C( 118), -INT8_C(  19),  INT8_C(  62), -INT8_C(  97),  INT8_C(  71), -INT8_C(  15),  INT8_C(  75),
         INT8_C( 111), -INT8_C( 108), -INT8_C(  23),  INT8_C( 101),  INT8_C( 125),  INT8_C(  84),  INT8_C(  48),  INT8_C(  53),
         INT8_C(  25),  INT8_C( 126), -INT8_C(   5), -INT8_C(  64),  INT8_C( 104), -INT8_C(  49),  INT8_C(  78),  INT8_C(   4),
        -INT8_C(  22), -INT8_C(  90),  INT8_C(  97),  INT8_C(  51),  INT8_C(   9), -INT8_C(  72), -INT8_C(   2),  INT8_C(  98),
         INT8_C(  46), -INT8_C(  20), -INT8_C(  96), -INT8_C(  51),  INT8_C(  51), -INT8_C( 111),  INT8_C(  24), -INT8_C(  94),
         INT8_C(  38),  INT8_C(   2),  INT8_C(   7), -INT8_C(  93),  INT8_C(  86),  INT8_C(  55), -INT8_C(  40),  INT8_C( 111),
        -INT8_C(  75), -INT8_C(  45),  INT8_C(  47),  INT8_C(  30), -INT8_C(  94),  INT8_C( 125),  INT8_C(  34), -INT8_C( 116),
         INT8_C(  35), -INT8_C( 125), -INT8_C(  64),  INT8_C(  44),  INT8_C(  59), -INT8_C(  66), -INT8_C( 113),  INT8_C( 105) },
      { -INT8_C(  86),  INT8_C(  47),  INT8_C(  54), -INT8_C(  35), -INT8_C(  63),  INT8_C(  78),      INT8_MIN, -INT8_C(  25),
         INT8_C(  80), -INT8_C( 121), -INT8_C( 118), -INT8_C(  90), -INT8_C(  65),  INT8_C(  98),  INT8_C(  22),  INT8_C( 116),
         INT8_C(  53),  INT8_C(  69), -INT8_C( 110), -INT8_C(  40), -INT8_C(  61), -INT8_C(  76),  INT8_C( 100), -INT8_C(  26),
         INT8_C(  55),  INT8_C(  36),  INT8_C(  19),  INT8_C( 114), -INT8_C(  29), -INT8_C(  94), -INT8_C(  37), -INT8_C( 115),
        -INT8_C(  47),  INT8_C(  17),  INT8_C( 107), -INT8_C( 110),  INT8_C(  96), -INT8_C(  21),  INT8_C( 121), -INT8_C(  80),
         INT8_C( 114),  INT8_C(   3),  INT8_C(  87),  INT8_C(  49),  INT8_C( 101),  INT8_C( 109), -INT8_C(  90), -INT8_C( 101),
        -INT8_C(  78),  INT8_C(  56),  INT8_C( 115),  INT8_C( 117), -INT8_C(  19), -INT8_C(  41),  INT8_C(  92),  INT8_C(  36),
        -INT8_C(   4),  INT8_C( 111), -INT8_C( 105), -INT8_C(  33),  INT8_C(  17),  INT8_C( 114),  INT8_C( 108), -INT8_C(  30) },
      {  INT8_C(   3), -INT8_C(  91),  INT8_C(   0),  INT8_C(  27),  INT8_C(  96),  INT8_C(   0),  INT8_C(   0),  INT8_C(  50),
        -INT8_C(  65),  INT8_C(  27),  INT8_C(   0),  INT8_C(  11),  INT8_C(  60),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 104),  INT8_C(  43),  INT8_C(   0), -INT8_C(  78),  INT8_C(   0),
         INT8_C(  33),  INT8_C(   0),  INT8_C( 116), -INT8_C(  91), -INT8_C(  20),  INT8_C(   0), -INT8_C(  39),  INT8_C(   0),
        -INT8_C(   1),  INT8_C(   0),  INT8_C(   0),  INT8_C(  95),  INT8_C(   0),  INT8_C(   0), -INT8_C( 111),  INT8_C(   0),
        -INT8_C( 104),  INT8_C(   5),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  92),  INT8_C( 126),  INT8_C(   0),
         INT8_C( 103),  INT8_C(  11), -INT8_C(  94),  INT8_C(   0), -INT8_C( 113),  INT8_C(   0),  INT8_C( 126),  INT8_C(   0),
         INT8_C(  31), -INT8_C(  14),  INT8_C(   0),  INT8_C(  11),  INT8_C(   0),  INT8_C(   0), -INT8_C(   5),  INT8_C(  75) } },
    { UINT64_C( 3860973301387351940),
      { -INT8_C(  14), -INT8_C(  21),  INT8_C( 102),  INT8_C(  87),  INT8_C(  88),  INT8_C(  12), -INT8_C(  14),  INT8_C(  11),
         INT8_C(  69),  INT8_C( 101),      INT8_MIN,  INT8_C(  50),  INT8_C(  61), -INT8_C(  36),  INT8_C(  86),  INT8_C(  57),
         INT8_C(  75), -INT8_C(  19),  INT8_C(  24),  INT8_C(  92),  INT8_C(  96), -INT8_C( 124),  INT8_C(  63), -INT8_C(  28),
         INT8_C(  92), -INT8_C(  76), -INT8_C(  56),  INT8_C(  30), -INT8_C(  94),  INT8_C(  92),  INT8_C(  83), -INT8_C( 108),
         INT8_C(  72), -INT8_C(  70), -INT8_C(  20), -INT8_C(  96), -INT8_C(  58), -INT8_C(  34), -INT8_C(  85),  INT8_C(  11),
         INT8_C(  68),  INT8_C(  44),  INT8_C(  61), -INT8_C( 127),  INT8_C(   8), -INT8_C( 108), -INT8_C(  70),  INT8_C(  84),
        -INT8_C( 127), -INT8_C(  46), -INT8_C(  80), -INT8_C(  31),  INT8_C(  86), -INT8_C(  17), -INT8_C(  59), -INT8_C(  78),
        -INT8_C(  93), -INT8_C( 115), -INT8_C(  47),  INT8_C(  70), -INT8_C(  22),  INT8_C(  36), -INT8_C(  38),  INT8_C(  50) },
      { -INT8_C(  34), -INT8_C(  58), -INT8_C(  46), -INT8_C(  91), -INT8_C(  91),  INT8_C( 126), -INT8_C(  80), -INT8_C(  23),
        -INT8_C(  86), -INT8_C(  18),  INT8_C( 106), -INT8_C(  78), -INT8_C( 126),  INT8_C(  36),  INT8_C(   6),  INT8_C(   3),
        -INT8_C(  10), -INT8_C(  73), -INT8_C(  27),  INT8_C(  76), -INT8_C(  90), -INT8_C(  86), -INT8_C(   1),  INT8_C(  74),
         INT8_C(  56), -INT8_C(  48), -INT8_C( 112),  INT8_C(  34), -INT8_C(  12),  INT8_C( 106),  INT8_C(  84), -INT8_C(  45),
         INT8_C(  49),  INT8_C(  38),  INT8_C( 120), -INT8_C(  42), -INT8_C(  92),  INT8_C(  40), -INT8_C(  65),  INT8_C(  78),
         INT8_C(  22),  INT8_C(  41),  INT8_C(   1), -INT8_C( 104),  INT8_C(  77),  INT8_C(   7), -INT8_C( 100),  INT8_C(  67),
        -INT8_C(  66), -INT8_C( 127), -INT8_C( 113),  INT8_C( 101),  INT8_C(  43), -INT8_C( 114), -INT8_C(  81),  INT8_C(  99),
         INT8_C(  94),  INT8_C(  63), -INT8_C( 123),  INT8_C(  83), -INT8_C(  87), -INT8_C(  39),  INT8_C(  38), -INT8_C(  38) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(  56),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  12),
        -INT8_C(  17),  INT8_C(  83), -INT8_C(  22),  INT8_C(   0), -INT8_C(  65),  INT8_C(   0),  INT8_C(  92),  INT8_C(  60),
         INT8_C(  65),  INT8_C(   0), -INT8_C(   3),  INT8_C(   0),  INT8_C(   6),  INT8_C(  46),  INT8_C(  62),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  88),  INT8_C(   0),  INT8_C(   0), -INT8_C(  58), -INT8_C(  89),  INT8_C( 103),
         INT8_C(   0), -INT8_C(  32),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 106),  INT8_C(  89),
         INT8_C(   0),  INT8_C(  85),  INT8_C(  62),  INT8_C(  25),  INT8_C(   0), -INT8_C( 101),  INT8_C(  86), -INT8_C( 105),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  63),  INT8_C(   0), -INT8_C( 127),  INT8_C(   0),  INT8_C(   0),  INT8_C(  21),
         INT8_C(   1),  INT8_C(   0),  INT8_C(  86),  INT8_C(   0), -INT8_C( 109), -INT8_C(   3),  INT8_C(   0),  INT8_C(   0) } },
    { UINT64_C(15993249600680009216),
      { -INT8_C( 104), -INT8_C(  12),  INT8_C( 117), -INT8_C(  27), -INT8_C(   5),  INT8_C(  17),  INT8_C(  40), -INT8_C(  70),
        -INT8_C( 110), -INT8_C(  72),  INT8_C(  31), -INT8_C(  66),  INT8_C(  70), -INT8_C(  50),  INT8_C(  33), -INT8_C(  91),
         INT8_C(  13), -INT8_C(  89), -INT8_C(   8), -INT8_C(  74),      INT8_MIN,  INT8_C(  30), -INT8_C( 111),      INT8_MIN,
        -INT8_C(  68),  INT8_C(  65),  INT8_C(  37), -INT8_C( 126), -INT8_C(  79),  INT8_C(  24),  INT8_C(  95),  INT8_C(  73),
         INT8_C(  12), -INT8_C(  43),  INT8_C(  47),  INT8_C(   7), -INT8_C(  26),  INT8_C(  87), -INT8_C(  63),  INT8_C( 121),
         INT8_C(  15), -INT8_C(  32),  INT8_C(  55),  INT8_C(  86), -INT8_C(  82),  INT8_C(  88), -INT8_C(   5), -INT8_C(  69),
        -INT8_C(   1), -INT8_C(  13),  INT8_C( 114),      INT8_MIN,  INT8_C(  17),  INT8_C(   3),  INT8_C(   0), -INT8_C(  51),
         INT8_C(  68),  INT8_C(  37),  INT8_C(  79), -INT8_C(  11),  INT8_C(  61), -INT8_C(  81),  INT8_C(  63),  INT8_C(  73) },
      { -INT8_C( 124),  INT8_C( 110),  INT8_C(  81),  INT8_C( 106), -INT8_C(  59),  INT8_C(  18), -INT8_C(  29), -INT8_C(  43),
        -INT8_C(  13),  INT8_C(  26),  INT8_C(  43), -INT8_C(  95),  INT8_C( 115),  INT8_C(  38),  INT8_C(  93),  INT8_C( 114),
         INT8_C(  25), -INT8_C(  49), -INT8_C(  14),  INT8_C(  42), -INT8_C(  46), -INT8_C(  13), -INT8_C(   9),  INT8_C(  22),
         INT8_C(  24),  INT8_C(  70),  INT8_C(  12),  INT8_C(  86), -INT8_C(  11),  INT8_C(  75), -INT8_C(  97),  INT8_C( 121),
        -INT8_C(  71), -INT8_C(  16), -INT8_C(  28),  INT8_C( 126),  INT8_C(   3), -INT8_C(  57),  INT8_C(  83), -INT8_C(  10),
        -INT8_C(  30),  INT8_C( 126), -INT8_C( 105),  INT8_C(  85), -INT8_C(  92), -INT8_C(  12), -INT8_C(  57), -INT8_C(  67),
        -INT8_C(  61), -INT8_C(  70), -INT8_C(  25), -INT8_C( 107), -INT8_C(  83), -INT8_C(  34), -INT8_C(  84), -INT8_C(  59),
         INT8_C(  37), -INT8_C(  72),  INT8_C(  27),  INT8_C(  26),  INT8_C(   3), -INT8_C(  69), -INT8_C( 108), -INT8_C(  68) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  46),  INT8_C(  74),  INT8_C(  95), -INT8_C(  71),  INT8_C(   0),  INT8_C(   0),  INT8_C(  23),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  82),  INT8_C(  17),  INT8_C(   0), -INT8_C( 106),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  49),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),  INT8_C(   0), -INT8_C(  62),
         INT8_C(   0), -INT8_C(  59),  INT8_C(  19),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  20),  INT8_C( 111),
        -INT8_C(  15),  INT8_C(  94), -INT8_C(  50), -INT8_C(  85),  INT8_C(   0),  INT8_C(  76), -INT8_C(  62),  INT8_C(   0),
        -INT8_C(  62), -INT8_C(  83),  INT8_C(   0),  INT8_C(   0), -INT8_C(  66), -INT8_C(  31), -INT8_C(  84), -INT8_C( 110),
         INT8_C( 105),  INT8_C(   0),  INT8_C( 106),  INT8_C(  15),  INT8_C(  64),  INT8_C(   0), -INT8_C(  45),  INT8_C(   5) } },
    { UINT64_C( 2424218903589320875),
      {  INT8_C(  12),  INT8_C(  60),  INT8_C( 118), -INT8_C(  79),  INT8_C(  48),  INT8_C(  62),  INT8_C( 110), -INT8_C(  12),
        -INT8_C(   8),  INT8_C(  86), -INT8_C( 119), -INT8_C(  91),  INT8_C(  52),  INT8_C(  53),  INT8_C( 106),  INT8_C(  89),
        -INT8_C(  19), -INT8_C( 122),  INT8_C( 116), -INT8_C(  16),  INT8_C(  65),  INT8_C(   8), -INT8_C(  84), -INT8_C(  20),
             INT8_MIN, -INT8_C(  25), -INT8_C( 101), -INT8_C(  65),  INT8_C( 117),  INT8_C(  63), -INT8_C(  31), -INT8_C( 127),
         INT8_C( 123),  INT8_C(  87),  INT8_C(  50), -INT8_C(  84), -INT8_C( 107), -INT8_C(  95), -INT8_C(  96), -INT8_C( 115),
        -INT8_C(   9),  INT8_C(  41),  INT8_C(  50),  INT8_C(  43),  INT8_C(  95), -INT8_C(  99), -INT8_C( 123),  INT8_C(  76),
         INT8_C(  35), -INT8_C(   7),  INT8_C(  61),  INT8_C( 100),  INT8_C(   1), -INT8_C(  23),  INT8_C(  80), -INT8_C( 127),
        -INT8_C(  48), -INT8_C(  21),  INT8_C(  64),  INT8_C(  69),  INT8_C(  43),  INT8_C(  33), -INT8_C(  57), -INT8_C(  90) },
      {  INT8_C( 121), -INT8_C(   7),  INT8_C(  82),  INT8_C(  14), -INT8_C( 102), -INT8_C(  14), -INT8_C( 100), -INT8_C( 111),
         INT8_C(  28), -INT8_C(  50), -INT8_C(  67),  INT8_C( 123),  INT8_C( 107),  INT8_C(  66), -INT8_C(  57), -INT8_C( 114),
         INT8_C(  59),  INT8_C(   4), -INT8_C(  14),  INT8_C(  60), -INT8_C(  18),  INT8_C(  67), -INT8_C(  67), -INT8_C(  66),
         INT8_C(  46), -INT8_C(   3),  INT8_C(   4),  INT8_C(  89),  INT8_C(  31), -INT8_C(  53),  INT8_C(   0), -INT8_C( 104),
        -INT8_C(  60),  INT8_C(  82), -INT8_C(  90),  INT8_C(  95),  INT8_C(  69),  INT8_C(  66), -INT8_C(  16),  INT8_C(  97),
         INT8_C(  17), -INT8_C(  83), -INT8_C(  36),  INT8_C( 124), -INT8_C(  17), -INT8_C(  93),  INT8_C(  11),  INT8_C(  42),
        -INT8_C(  88), -INT8_C(   3),  INT8_C( 102), -INT8_C( 106),  INT8_C(  64),  INT8_C(  35),  INT8_C(  84),  INT8_C( 111),
         INT8_C(  33),  INT8_C(  88), -INT8_C(  56),  INT8_C(  64),  INT8_C(  35), -INT8_C(  56), -INT8_C(  40), -INT8_C(  24) },
      { -INT8_C( 123),  INT8_C(  53),  INT8_C(   0), -INT8_C(  65),  INT8_C(   0),  INT8_C(  48),  INT8_C(   0), -INT8_C( 123),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  32), -INT8_C(  97),  INT8_C( 119),  INT8_C(  49),  INT8_C(   0),
         INT8_C(   0), -INT8_C( 118),  INT8_C(   0),  INT8_C(  44),  INT8_C(  47),  INT8_C(  75),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  28), -INT8_C(  97),  INT8_C(  24),  INT8_C(   0),  INT8_C(  10),  INT8_C(   0),  INT8_C(  25),
         INT8_C(  63), -INT8_C(  87), -INT8_C(  40),  INT8_C(  11), -INT8_C(  38), -INT8_C(  29),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  42),  INT8_C(  14), -INT8_C(  89),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 118),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  93),  INT8_C(   0),  INT8_C(   0),  INT8_C(  12),  INT8_C(   0), -INT8_C(  16),
        -INT8_C(  15),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  23),  INT8_C(   0),  INT8_C(   0) } },
    { UINT64_C(15186480722153537051),
      { -INT8_C(  27), -INT8_C(  99),  INT8_C(  78), -INT8_C(  44),  INT8_C(  64),  INT8_C(  89), -INT8_C(   1), -INT8_C(  24),
         INT8_C(  87),  INT8_C( 101),  INT8_C( 126), -INT8_C( 105), -INT8_C( 119), -INT8_C(  45),  INT8_C(   6), -INT8_C(  86),
         INT8_C(  43), -INT8_C(  49), -INT8_C(  22),  INT8_C(  79), -INT8_C( 105), -INT8_C(  62),  INT8_C(  55), -INT8_C(  78),
         INT8_C(  64),  INT8_C( 126),  INT8_C(  18),  INT8_C(   1), -INT8_C(  75), -INT8_C(  45), -INT8_C(  45), -INT8_C( 102),
         INT8_C( 112),  INT8_C(  34),  INT8_C( 111), -INT8_C(  79),  INT8_C( 123),  INT8_C( 110), -INT8_C( 103), -INT8_C(  46),
        -INT8_C(  45),  INT8_C(  24),  INT8_C( 106),  INT8_C(  92), -INT8_C(  21),  INT8_C( 112),  INT8_C(   6),  INT8_C(  22),
         INT8_C(  63), -INT8_C(  16),  INT8_C( 101), -INT8_C(  41), -INT8_C(  78), -INT8_C( 100), -INT8_C( 119), -INT8_C(  13),
         INT8_C(  26), -INT8_C( 100), -INT8_C(  12), -INT8_C(  48),  INT8_C( 111), -INT8_C(  56),  INT8_C( 106), -INT8_C(  32) },
      { -INT8_C(  22), -INT8_C(  39), -INT8_C( 111),  INT8_C( 101),  INT8_C(  71),  INT8_C(  42),  INT8_C(  56),  INT8_C(  27),
         INT8_C(  66), -INT8_C(  94),  INT8_C( 119),  INT8_C(  45),  INT8_C(  18),  INT8_C( 126),  INT8_C(  68),  INT8_C(  82),
         INT8_C( 110), -INT8_C(  87),  INT8_C(  41),  INT8_C(  33),  INT8_C(  70), -INT8_C(  78),  INT8_C(  20),  INT8_C(  96),
         INT8_C(  78),  INT8_C(   8),  INT8_C(  48), -INT8_C(  66), -INT8_C(  48), -INT8_C( 101), -INT8_C(  98), -INT8_C(  70),
         INT8_C( 116),  INT8_C(  47),  INT8_C(  32), -INT8_C(  68),  INT8_C(  89),  INT8_C(  88), -INT8_C(  41), -INT8_C( 100),
        -INT8_C(   6),  INT8_C(  78), -INT8_C(  55),  INT8_C(  12), -INT8_C(  52),  INT8_C(  13),  INT8_C(  94),  INT8_C(  59),
        -INT8_C(  73), -INT8_C( 121),  INT8_C(  92), -INT8_C(   3),  INT8_C(  58),  INT8_C( 112),  INT8_C(  93), -INT8_C( 120),
         INT8_C( 120), -INT8_C( 114),  INT8_C(  70),  INT8_C(  73),  INT8_C(  41), -INT8_C(  28),  INT8_C(   3), -INT8_C(  99) },
      { -INT8_C(  49),  INT8_C( 118),  INT8_C(   0),  INT8_C(  57), -INT8_C( 121),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   7), -INT8_C(  11), -INT8_C(  60), -INT8_C( 101),  INT8_C(  81),  INT8_C(  74),  INT8_C(   0),
        -INT8_C( 103),  INT8_C( 120),  INT8_C(  19),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  75),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 110),  INT8_C( 113),  INT8_C(   0),
        -INT8_C(  28),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 112),  INT8_C( 110),
        -INT8_C(  51),  INT8_C( 102),  INT8_C(  51),  INT8_C(   0), -INT8_C(  73),  INT8_C( 125),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  10),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  26),  INT8_C( 123),
         INT8_C(   0),  INT8_C(  42),  INT8_C(   0),  INT8_C(   0), -INT8_C( 104),  INT8_C(   0),  INT8_C( 109),  INT8_C( 125) } },
    { UINT64_C( 8433325083901633299),
      {      INT8_MAX, -INT8_C(  46), -INT8_C( 126),  INT8_C(  75), -INT8_C(  32), -INT8_C(  32), -INT8_C( 122), -INT8_C( 105),
         INT8_C( 104), -INT8_C(  30), -INT8_C( 108), -INT8_C(  94),  INT8_C(  82), -INT8_C(  15),  INT8_C(  42), -INT8_C(  53),
             INT8_MAX,  INT8_C( 113),  INT8_C(  20), -INT8_C(  88),  INT8_C(  85),  INT8_C(  23),  INT8_C(  70),  INT8_C( 105),
         INT8_C(  59), -INT8_C(  97), -INT8_C(  42), -INT8_C(  74), -INT8_C(  48), -INT8_C(  33),  INT8_C(  44),  INT8_C(  79),
        -INT8_C(  79), -INT8_C(  82), -INT8_C( 102), -INT8_C( 111), -INT8_C( 114),  INT8_C(  33),  INT8_C(  40), -INT8_C(  10),
         INT8_C(   3), -INT8_C(  68), -INT8_C( 104),  INT8_C(  86), -INT8_C(  82), -INT8_C(  61),  INT8_C(  33),  INT8_C(  45),
         INT8_C(  52),  INT8_C(  53), -INT8_C(  42), -INT8_C( 119),  INT8_C(  76),  INT8_C(  28), -INT8_C(  14), -INT8_C( 121),
        -INT8_C(  69), -INT8_C(  56),  INT8_C(  62), -INT8_C( 117), -INT8_C(  89),  INT8_C( 106), -INT8_C(  38),  INT8_C(  89) },
      {  INT8_C(  24),  INT8_C( 117), -INT8_C(  22), -INT8_C(  90), -INT8_C( 106),  INT8_C(  19), -INT8_C(  99), -INT8_C( 103),
        -INT8_C(  49),  INT8_C(  53), -INT8_C(  17),  INT8_C( 125), -INT8_C(   8),  INT8_C(  16), -INT8_C(  85),  INT8_C(  44),
         INT8_C(  69), -INT8_C( 127), -INT8_C(  74), -INT8_C( 110), -INT8_C(  99), -INT8_C(  88),  INT8_C(  25),  INT8_C(  88),
         INT8_C( 113),  INT8_C(  87), -INT8_C(  28),  INT8_C(  24), -INT8_C(  63), -INT8_C(  66),  INT8_C( 113), -INT8_C(  39),
         INT8_C(  51),  INT8_C(  92),      INT8_MIN, -INT8_C(  55),  INT8_C( 111),  INT8_C(  29),  INT8_C(  99),  INT8_C(  62),
         INT8_C(  82),  INT8_C(  82), -INT8_C(  68),  INT8_C(  75),  INT8_C(  99),  INT8_C( 103),  INT8_C( 119), -INT8_C(  88),
        -INT8_C(  24),  INT8_C(  45),  INT8_C(  58), -INT8_C( 123), -INT8_C(  42),  INT8_C(  84), -INT8_C(  35),  INT8_C(  71),
        -INT8_C(  85), -INT8_C(  63),  INT8_C(  95),  INT8_C( 109),      INT8_MIN, -INT8_C(  47),  INT8_C(  70), -INT8_C(  77) },
      { -INT8_C( 105),  INT8_C(  71),  INT8_C(   0),  INT8_C(   0),  INT8_C( 118),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  55),  INT8_C(  23),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   1),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  60),  INT8_C(   0),  INT8_C(   0),  INT8_C(  58), -INT8_C(  14),  INT8_C(   0),  INT8_C(  95),  INT8_C(   0),
        -INT8_C(  84),  INT8_C(   0), -INT8_C(  70), -INT8_C(  50),  INT8_C(   0), -INT8_C(  99), -INT8_C(  99),  INT8_C(   0),
        -INT8_C(  28),  INT8_C(  10),  INT8_C(   0),  INT8_C(  90), -INT8_C(   3),  INT8_C(  62), -INT8_C( 117),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  17),  INT8_C(  42),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  28),  INT8_C(   0),  INT8_C(   0),  INT8_C(  14),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 102),  INT8_C(   0), -INT8_C(  99),  INT8_C(   0),  INT8_C(  39),  INT8_C(  59),  INT8_C(  32),  INT8_C(   0) } },
    { UINT64_C( 3952718891158717997),
      {  INT8_C(  50), -INT8_C( 106), -INT8_C( 127), -INT8_C( 107), -INT8_C(   3), -INT8_C(   8),  INT8_C(  62), -INT8_C(  27),
         INT8_C(  38),  INT8_C( 120),  INT8_C( 106), -INT8_C(   4), -INT8_C(  52),  INT8_C(  72),  INT8_C(  67),  INT8_C( 120),
         INT8_C(   9), -INT8_C(  94), -INT8_C(  27), -INT8_C( 119),  INT8_C( 115),  INT8_C(  43),  INT8_C(  61), -INT8_C(  96),
        -INT8_C(  14), -INT8_C(  70),  INT8_C(  60), -INT8_C(  43), -INT8_C( 102),  INT8_C(  23),  INT8_C(  11), -INT8_C(  52),
        -INT8_C(  83), -INT8_C( 116),  INT8_C(  98), -INT8_C(  85), -INT8_C( 123), -INT8_C(  96), -INT8_C( 112), -INT8_C(  85),
         INT8_C(  24), -INT8_C(   5), -INT8_C(  89), -INT8_C(  27),  INT8_C(  67), -INT8_C(  22),  INT8_C(  93),  INT8_C(  76),
        -INT8_C( 116),  INT8_C(  66), -INT8_C(  42),  INT8_C(   0),  INT8_C( 109),  INT8_C(  19), -INT8_C(  96),  INT8_C(  95),
        -INT8_C(  51), -INT8_C(  35),  INT8_C(  53),  INT8_C( 103), -INT8_C(  12),  INT8_C(  64),  INT8_C(  51), -INT8_C(  95) },
      { -INT8_C(  51), -INT8_C( 107),  INT8_C(  76),  INT8_C(  82),  INT8_C(  53), -INT8_C(  35), -INT8_C(   3),  INT8_C(  78),
        -INT8_C(  40), -INT8_C(  92),  INT8_C(  51),  INT8_C(  27), -INT8_C( 114), -INT8_C( 112),  INT8_C( 103),  INT8_C(  26),
        -INT8_C(  46),  INT8_C(  61),  INT8_C(  26),  INT8_C(  63),  INT8_C(  80), -INT8_C(  69), -INT8_C(  97),  INT8_C(  29),
        -INT8_C( 104), -INT8_C(  44), -INT8_C( 124), -INT8_C( 116),  INT8_C(  20), -INT8_C(  72),  INT8_C(  45), -INT8_C(  31),
         INT8_C(  77),  INT8_C( 122),  INT8_C(  51), -INT8_C( 125),  INT8_C(  87),  INT8_C(  48), -INT8_C(  47),  INT8_C(  47),
        -INT8_C(  44),  INT8_C(   4),  INT8_C(  74),  INT8_C(  98), -INT8_C( 108), -INT8_C(  79),  INT8_C( 125),  INT8_C( 102),
        -INT8_C(  17), -INT8_C( 105), -INT8_C(  91),  INT8_C(  63),  INT8_C(  82),  INT8_C(  68),  INT8_C(  93), -INT8_C(  22),
         INT8_C(  24), -INT8_C(  31),  INT8_C( 118),  INT8_C(  45), -INT8_C( 103), -INT8_C(  92),  INT8_C(  14), -INT8_C(  25) },
      { -INT8_C(   1),  INT8_C(   0), -INT8_C(  51), -INT8_C(  25),  INT8_C(   0), -INT8_C(  43),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  28), -INT8_C(  99),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  86), -INT8_C( 110),
        -INT8_C(  37),  INT8_C(   0), -INT8_C(   1), -INT8_C(  56), -INT8_C(  61), -INT8_C(  26), -INT8_C(  36),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  64),  INT8_C(  97), -INT8_C(  82),  INT8_C(   0),  INT8_C(   0), -INT8_C(  83),
        -INT8_C(   6),  INT8_C(   6),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  48),  INT8_C(  97), -INT8_C(  38),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 101), -INT8_C(  38), -INT8_C(  78),
         INT8_C(   0), -INT8_C(  39),  INT8_C(   0),  INT8_C(  63), -INT8_C(  65),  INT8_C(   0), -INT8_C(   3),  INT8_C(  73),
         INT8_C(   0), -INT8_C(  66), -INT8_C(  85),  INT8_C(   0), -INT8_C( 115), -INT8_C(  28),  INT8_C(   0),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_add_epi8(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_add_epi8");
    easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_add_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { {  INT16_C( 11452),  INT16_C( 18562),  INT16_C(  2675), -INT16_C( 17089),  INT16_C( 30550), -INT16_C( 14832), -INT16_C( 11694), -INT16_C( 20057),
         INT16_C( 31125),  INT16_C(  7807), -INT16_C( 11481),  INT16_C( 12519), -INT16_C(   410), -INT16_C( 11992),  INT16_C( 27537),  INT16_C( 19885),
         INT16_C( 12440),  INT16_C(  2965), -INT16_C( 11206), -INT16_C( 28472), -INT16_C( 10164), -INT16_C( 25001), -INT16_C(   342),  INT16_C( 16463),
        -INT16_C( 12424), -INT16_C( 24738),  INT16_C( 17826),  INT16_C(  2255), -INT16_C(  1981), -INT16_C( 11047), -INT16_C( 30877), -INT16_C(  1246) },
      { -INT16_C( 18505), -INT16_C(  3833), -INT16_C( 12404), -INT16_C( 10111), -INT16_C( 10072),  INT16_C( 21110), -INT16_C( 14633),  INT16_C( 20370),
        -INT16_C(  3947),  INT16_C( 14318), -INT16_C( 17098),  INT16_C( 31040),  INT16_C(  6581),  INT16_C(  6478),  INT16_C( 28832),  INT16_C( 22292),
         INT16_C(  6951), -INT16_C( 19640), -INT16_C( 13589), -INT16_C( 27765),  INT16_C(   674),  INT16_C( 31205),  INT16_C( 30920),  INT16_C( 24008),
        -INT16_C( 18840), -INT16_C( 24940), -INT16_C( 11148),  INT16_C( 10520),  INT16_C( 26350), -INT16_C( 29118),  INT16_C( 22486), -INT16_C(   538) },
      { -INT16_C(  7053),  INT16_C( 14729), -INT16_C(  9729), -INT16_C( 27200),  INT16_C( 20478),  INT16_C(  6278), -INT16_C( 26327),  INT16_C(   313),
         INT16_C( 27178),  INT16_C( 22125), -INT16_C( 28579), -INT16_C( 21977),  INT16_C(  6171), -INT16_C(  5514), -INT16_C(  9167), -INT16_C( 23359),
         INT16_C( 19391), -INT16_C( 16675), -INT16_C( 24795),  INT16_C(  9299), -INT16_C(  9490),  INT16_C(  6204),  INT16_C( 30578), -INT16_C( 25065),
        -INT16_C( 31264),  INT16_C( 15858),  INT16_C(  6678),  INT16_C( 12775),  INT16_C( 24369),  INT16_C( 25371), -INT16_C(  8391), -INT16_C(  1784) } },
    { {  INT16_C( 11890),  INT16_C( 23985),  INT16_C( 15608), -INT16_C( 25616), -INT16_C( 10690),  INT16_C(  1556), -INT16_C(  8882), -INT16_C( 18845),
        -INT16_C(  1901),  INT16_C(  1877),  INT16_C( 28108), -INT16_C( 17871),  INT16_C( 29651), -INT16_C( 22199),  INT16_C( 12234),  INT16_C( 15782),
         INT16_C( 22365),  INT16_C( 22170), -INT16_C( 29804), -INT16_C( 11535),  INT16_C(  1377), -INT16_C( 20519),  INT16_C( 15586),  INT16_C( 30309),
        -INT16_C( 17868),  INT16_C(   381), -INT16_C( 20953), -INT16_C(  1349),  INT16_C(  1058), -INT16_C(  4957),  INT16_C( 18995), -INT16_C( 28375) },
      { -INT16_C( 15199),  INT16_C( 13799), -INT16_C( 10161), -INT16_C( 20472), -INT16_C(  7715), -INT16_C( 16289), -INT16_C( 15331),  INT16_C( 21046),
        -INT16_C( 19585), -INT16_C( 22957),  INT16_C(  3682), -INT16_C( 31583),  INT16_C( 17427),  INT16_C( 18032), -INT16_C( 25970),  INT16_C( 12503),
        -INT16_C( 16802), -INT16_C( 21147),  INT16_C( 28054),  INT16_C( 29789), -INT16_C( 17330),  INT16_C( 27700),  INT16_C( 27264), -INT16_C(    66),
         INT16_C(  4381),  INT16_C( 32678),  INT16_C( 18207),  INT16_C( 12803),  INT16_C( 29835),  INT16_C(  6777),  INT16_C( 20494),  INT16_C( 27722) },
      { -INT16_C(  3309), -INT16_C( 27752),  INT16_C(  5447),  INT16_C( 19448), -INT16_C( 18405), -INT16_C( 14733), -INT16_C( 24213),  INT16_C(  2201),
        -INT16_C( 21486), -INT16_C( 21080),  INT16_C( 31790),  INT16_C( 16082), -INT16_C( 18458), -INT16_C(  4167), -INT16_C( 13736),  INT16_C( 28285),
         INT16_C(  5563),  INT16_C(  1023), -INT16_C(  1750),  INT16_C( 18254), -INT16_C( 15953),  INT16_C(  7181), -INT16_C( 22686),  INT16_C( 30243),
        -INT16_C( 13487), -INT16_C( 32477), -INT16_C(  2746),  INT16_C( 11454),  INT16_C( 30893),  INT16_C(  1820), -INT16_C( 26047), -INT16_C(   653) } },
    { { -INT16_C( 20721), -INT16_C( 23271),  INT16_C( 30237),  INT16_C( 27417),  INT16_C( 19762), -INT16_C( 19753), -INT16_C( 27209), -INT16_C( 10830),
         INT16_C( 22694), -INT16_C( 14764),  INT16_C( 22687),  INT16_C( 11000),  INT16_C( 29132), -INT16_C(  9660), -INT16_C( 28990), -INT16_C( 11962),
         INT16_C( 24382),  INT16_C( 23414), -INT16_C( 28459),  INT16_C(  1990), -INT16_C( 24867), -INT16_C( 27207),  INT16_C( 27443), -INT16_C(  9622),
        -INT16_C( 16701),  INT16_C( 25248), -INT16_C( 26602), -INT16_C(  7539), -INT16_C( 12022), -INT16_C( 13124),  INT16_C(   608), -INT16_C( 24931) },
      {  INT16_C(  4961),  INT16_C( 14073), -INT16_C( 16477), -INT16_C( 32451), -INT16_C(  2211), -INT16_C( 28394), -INT16_C( 32670),  INT16_C(  9835),
         INT16_C(  2878),  INT16_C( 21896),  INT16_C(  5539), -INT16_C( 21193), -INT16_C(  2841),  INT16_C( 18297),  INT16_C(  5878),  INT16_C( 22757),
        -INT16_C(  8662), -INT16_C( 12914), -INT16_C( 13155), -INT16_C(  1202),  INT16_C( 25795),  INT16_C(  9612), -INT16_C(  2076),  INT16_C(  9035),
        -INT16_C( 11262), -INT16_C( 23176), -INT16_C( 20503), -INT16_C( 12205), -INT16_C( 13149), -INT16_C( 26089), -INT16_C(   797),  INT16_C(  3570) },
      { -INT16_C( 15760), -INT16_C(  9198),  INT16_C( 13760), -INT16_C(  5034),  INT16_C( 17551),  INT16_C( 17389),  INT16_C(  5657), -INT16_C(   995),
         INT16_C( 25572),  INT16_C(  7132),  INT16_C( 28226), -INT16_C( 10193),  INT16_C( 26291),  INT16_C(  8637), -INT16_C( 23112),  INT16_C( 10795),
         INT16_C( 15720),  INT16_C( 10500),  INT16_C( 23922),  INT16_C(   788),  INT16_C(   928), -INT16_C( 17595),  INT16_C( 25367), -INT16_C(   587),
        -INT16_C( 27963),  INT16_C(  2072),  INT16_C( 18431), -INT16_C( 19744), -INT16_C( 25171),  INT16_C( 26323), -INT16_C(   189), -INT16_C( 21361) } },
    { { -INT16_C( 32550),  INT16_C( 30938),  INT16_C( 10572),  INT16_C(  3955), -INT16_C(   115),  INT16_C( 29237), -INT16_C( 32522), -INT16_C(  1899),
         INT16_C(  3412),  INT16_C( 16029), -INT16_C(  3908),  INT16_C( 24590),  INT16_C(  9917), -INT16_C( 24326), -INT16_C(  5086), -INT16_C(   595),
        -INT16_C( 30868), -INT16_C( 18059), -INT16_C(  5968),  INT16_C( 16072), -INT16_C(   537), -INT16_C(  8784),  INT16_C( 17790), -INT16_C( 11563),
         INT16_C( 29266),  INT16_C(  3600),  INT16_C(  8035),  INT16_C(  8302),  INT16_C( 26693),  INT16_C( 26560),  INT16_C( 27988), -INT16_C( 16028) },
      { -INT16_C(  9740), -INT16_C( 23174),  INT16_C( 17089), -INT16_C( 22301), -INT16_C( 27840), -INT16_C( 16763),  INT16_C( 23256),  INT16_C( 10896),
        -INT16_C( 24115),  INT16_C( 12344), -INT16_C( 22592),  INT16_C(  1360),  INT16_C(  4111),  INT16_C( 25708), -INT16_C( 11907),  INT16_C( 28965),
        -INT16_C( 24662),  INT16_C( 27670), -INT16_C(  1567),  INT16_C(  8468), -INT16_C( 25972),  INT16_C( 25823),  INT16_C( 28916), -INT16_C( 15986),
        -INT16_C( 14575), -INT16_C( 11791),  INT16_C( 16750),  INT16_C( 32214),  INT16_C( 16977), -INT16_C( 12575),  INT16_C(  1555), -INT16_C( 16832) },
      {  INT16_C( 23246),  INT16_C(  7764),  INT16_C( 27661), -INT16_C( 18346), -INT16_C( 27955),  INT16_C( 12474), -INT16_C(  9266),  INT16_C(  8997),
        -INT16_C( 20703),  INT16_C( 28373), -INT16_C( 26500),  INT16_C( 25950),  INT16_C( 14028),  INT16_C(  1382), -INT16_C( 16993),  INT16_C( 28370),
         INT16_C( 10006),  INT16_C(  9611), -INT16_C(  7535),  INT16_C( 24540), -INT16_C( 26509),  INT16_C( 17039), -INT16_C( 18830), -INT16_C( 27549),
         INT16_C( 14691), -INT16_C(  8191),  INT16_C( 24785), -INT16_C( 25020), -INT16_C( 21866),  INT16_C( 13985),  INT16_C( 29543),  INT16_C( 32676) } },
    { {  INT16_C( 22181), -INT16_C( 30934),  INT16_C( 15952), -INT16_C(  9048), -INT16_C( 30504), -INT16_C( 12991), -INT16_C( 12296),  INT16_C(  2446),
        -INT16_C( 32618),  INT16_C(  1242), -INT16_C( 20287),  INT16_C(  4994),  INT16_C( 25586),  INT16_C(  1761),  INT16_C(  8554),  INT16_C(  4036),
        -INT16_C(  4488), -INT16_C( 14186),  INT16_C( 16172),  INT16_C(  1444), -INT16_C(  6713), -INT16_C( 16430),  INT16_C( 24757),  INT16_C( 19400),
        -INT16_C( 23840), -INT16_C( 23984), -INT16_C( 11694),  INT16_C( 17589), -INT16_C( 27083), -INT16_C( 24758),  INT16_C(  3768),  INT16_C( 12463) },
      {  INT16_C( 17916),  INT16_C( 10744), -INT16_C( 25468),  INT16_C( 19246),  INT16_C(   130),  INT16_C( 14090), -INT16_C( 11680),  INT16_C( 16770),
        -INT16_C( 11660), -INT16_C( 14621), -INT16_C( 26460), -INT16_C(  9717),  INT16_C( 21806), -INT16_C(  6535),  INT16_C( 10340),  INT16_C( 24598),
         INT16_C(  3694), -INT16_C(  3447), -INT16_C( 18517),  INT16_C( 11582),  INT16_C( 18615),  INT16_C(  6244), -INT16_C(  6629), -INT16_C( 28839),
         INT16_C( 15545),  INT16_C( 23894),  INT16_C( 25044),  INT16_C(   567), -INT16_C( 20042),  INT16_C(  6889), -INT16_C(    39),  INT16_C( 18299) },
      { -INT16_C( 25439), -INT16_C( 20190), -INT16_C(  9516),  INT16_C( 10198), -INT16_C( 30374),  INT16_C(  1099), -INT16_C( 23976),  INT16_C( 19216),
         INT16_C( 21258), -INT16_C( 13379),  INT16_C( 18789), -INT16_C(  4723), -INT16_C( 18144), -INT16_C(  4774),  INT16_C( 18894),  INT16_C( 28634),
        -INT16_C(   794), -INT16_C( 17633), -INT16_C(  2345),  INT16_C( 13026),  INT16_C( 11902), -INT16_C( 10186),  INT16_C( 18128), -INT16_C(  9439),
        -INT16_C(  8295), -INT16_C(    90),  INT16_C( 13350),  INT16_C( 18156),  INT16_C( 18411), -INT16_C( 17869),  INT16_C(  3729),  INT16_C( 30762) } },
    { {  INT16_C(  1038), -INT16_C( 18118),  INT16_C( 30908),  INT16_C( 29670),  INT16_C( 19136), -INT16_C(  9333), -INT16_C(  7120), -INT16_C(  5781),
        -INT16_C( 16096), -INT16_C(  3001),  INT16_C( 32290), -INT16_C(  9993), -INT16_C(  8145),  INT16_C(  2547),  INT16_C( 28383), -INT16_C(  4784),
        -INT16_C( 30094),  INT16_C( 11942), -INT16_C( 29694), -INT16_C( 15454),  INT16_C( 11734),  INT16_C(  1950),  INT16_C(  2322),  INT16_C( 13040),
         INT16_C( 14282), -INT16_C(  5081),  INT16_C(  7862), -INT16_C(  6715), -INT16_C( 18178), -INT16_C(  8722),  INT16_C( 16166), -INT16_C( 26421) },
      {  INT16_C( 29129), -INT16_C( 13113),  INT16_C( 27134), -INT16_C( 11121),  INT16_C( 11670), -INT16_C( 22309), -INT16_C( 13257),  INT16_C(   475),
         INT16_C(   515), -INT16_C( 17938), -INT16_C( 19680),  INT16_C(  7839), -INT16_C( 29333), -INT16_C( 28165), -INT16_C( 14644), -INT16_C( 27095),
        -INT16_C(  4040),  INT16_C( 13922), -INT16_C(  3751), -INT16_C(  4086), -INT16_C(  6626),  INT16_C( 21912),  INT16_C( 29618), -INT16_C( 19113),
         INT16_C( 17781), -INT16_C( 27281),  INT16_C(  3832),  INT16_C( 25523), -INT16_C( 20581),  INT16_C( 26868),  INT16_C(  7541), -INT16_C( 20994) },
      {  INT16_C( 30167), -INT16_C( 31231), -INT16_C(  7494),  INT16_C( 18549),  INT16_C( 30806), -INT16_C( 31642), -INT16_C( 20377), -INT16_C(  5306),
        -INT16_C( 15581), -INT16_C( 20939),  INT16_C( 12610), -INT16_C(  2154),  INT16_C( 28058), -INT16_C( 25618),  INT16_C( 13739), -INT16_C( 31879),
         INT16_C( 31402),  INT16_C( 25864),  INT16_C( 32091), -INT16_C( 19540),  INT16_C(  5108),  INT16_C( 23862),  INT16_C( 31940), -INT16_C(  6073),
         INT16_C( 32063), -INT16_C( 32362),  INT16_C( 11694),  INT16_C( 18808),  INT16_C( 26777),  INT16_C( 18146),  INT16_C( 23707),  INT16_C( 18121) } },
    { {  INT16_C( 24590),  INT16_C( 26595), -INT16_C(  4527),  INT16_C( 28503), -INT16_C(  3884), -INT16_C( 31035),  INT16_C(  7267), -INT16_C(  9925),
        -INT16_C( 21919),  INT16_C( 22894),  INT16_C(  8888),  INT16_C( 21692), -INT16_C( 20271),  INT16_C( 18108), -INT16_C( 17715), -INT16_C(  9228),
        -INT16_C( 10470),  INT16_C( 27459), -INT16_C( 25915), -INT16_C( 26150), -INT16_C( 24694), -INT16_C(  4577),  INT16_C( 23483),  INT16_C(  7367),
         INT16_C( 13573), -INT16_C( 16779),  INT16_C( 12631),  INT16_C( 10258), -INT16_C( 12575), -INT16_C( 20625),  INT16_C( 25480), -INT16_C( 23926) },
      { -INT16_C( 12998),  INT16_C(    13), -INT16_C(  6296), -INT16_C(  3431), -INT16_C( 18041),  INT16_C( 17120), -INT16_C( 22764),  INT16_C(  6495),
        -INT16_C( 11043),  INT16_C( 13527), -INT16_C(  5882), -INT16_C(  6307), -INT16_C( 13129),  INT16_C( 16278),  INT16_C(  8495),  INT16_C( 27105),
        -INT16_C(  4370),  INT16_C( 22121),  INT16_C(   982),  INT16_C( 23881),  INT16_C( 10684), -INT16_C( 12129), -INT16_C(   303), -INT16_C( 20759),
        -INT16_C( 15917), -INT16_C(  9758),  INT16_C( 16298),  INT16_C( 25280),  INT16_C( 22283),  INT16_C( 15009), -INT16_C( 31880),  INT16_C( 26276) },
      {  INT16_C( 11592),  INT16_C( 26608), -INT16_C( 10823),  INT16_C( 25072), -INT16_C( 21925), -INT16_C( 13915), -INT16_C( 15497), -INT16_C(  3430),
         INT16_C( 32574), -INT16_C( 29115),  INT16_C(  3006),  INT16_C( 15385),  INT16_C( 32136), -INT16_C( 31150), -INT16_C(  9220),  INT16_C( 17877),
        -INT16_C( 14840), -INT16_C( 15956), -INT16_C( 24933), -INT16_C(  2269), -INT16_C( 14010), -INT16_C( 16706),  INT16_C( 23180), -INT16_C( 13392),
        -INT16_C(  2344), -INT16_C( 26537),  INT16_C( 28929), -INT16_C( 29998),  INT16_C(  9708), -INT16_C(  5616), -INT16_C(  6400),  INT16_C(  2350) } },
    { {  INT16_C(  3441),  INT16_C( 18365),  INT16_C(  1552), -INT16_C( 13148),  INT16_C( 17455),  INT16_C(   156), -INT16_C( 31166),  INT16_C(  5550),
        -INT16_C( 28345), -INT16_C(  3602), -INT16_C( 20528), -INT16_C(  9133), -INT16_C(  2810),  INT16_C( 32278), -INT16_C( 17800), -INT16_C(  5660),
        -INT16_C( 24120), -INT16_C( 10191), -INT16_C( 10841), -INT16_C( 10331),  INT16_C( 16665),  INT16_C( 23767), -INT16_C( 31033),  INT16_C(  3697),
         INT16_C( 24599), -INT16_C(  6400),  INT16_C( 21263),  INT16_C(  5571), -INT16_C(  9656), -INT16_C( 16237),  INT16_C( 30612),  INT16_C( 23722) },
      { -INT16_C(  9447), -INT16_C( 16331), -INT16_C(  9552), -INT16_C( 13673),  INT16_C( 28443), -INT16_C(  7386), -INT16_C( 26635),  INT16_C(  3313),
        -INT16_C(  3593),  INT16_C(  1779), -INT16_C( 18619), -INT16_C( 29413), -INT16_C( 20847),  INT16_C(  9550), -INT16_C(  2010),  INT16_C( 16258),
        -INT16_C( 18477), -INT16_C( 31745), -INT16_C( 26735), -INT16_C( 21427),  INT16_C( 29446), -INT16_C(  1137), -INT16_C( 32501),  INT16_C(   519),
        -INT16_C(  1422), -INT16_C( 18679),  INT16_C(  9393),  INT16_C( 16965), -INT16_C( 27693), -INT16_C(  1688), -INT16_C(  5493),  INT16_C( 24120) },
      { -INT16_C(  6006),  INT16_C(  2034), -INT16_C(  8000), -INT16_C( 26821), -INT16_C( 19638), -INT16_C(  7230),  INT16_C(  7735),  INT16_C(  8863),
        -INT16_C( 31938), -INT16_C(  1823),  INT16_C( 26389),  INT16_C( 26990), -INT16_C( 23657), -INT16_C( 23708), -INT16_C( 19810),  INT16_C( 10598),
         INT16_C( 22939),  INT16_C( 23600),  INT16_C( 27960), -INT16_C( 31758), -INT16_C( 19425),  INT16_C( 22630),  INT16_C(  2002),  INT16_C(  4216),
         INT16_C( 23177), -INT16_C( 25079),  INT16_C( 30656),  INT16_C( 22536),  INT16_C( 28187), -INT16_C( 17925),  INT16_C( 25119), -INT16_C( 17694) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_add_epi16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_add_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_add_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { {  INT16_C( 19989),  INT16_C( 17201), -INT16_C(   484), -INT16_C(  4807), -INT16_C( 18583),  INT16_C(  7804),  INT16_C( 26728),  INT16_C( 16294),
        -INT16_C( 31636),  INT16_C( 28135), -INT16_C( 32158),  INT16_C( 20064), -INT16_C( 16695),  INT16_C(  3157),  INT16_C( 32300),  INT16_C( 16873),
         INT16_C(  6860), -INT16_C(  5755), -INT16_C( 16872), -INT16_C( 32042),  INT16_C( 21110), -INT16_C(  8544),  INT16_C( 18106),  INT16_C(  9757),
         INT16_C(  1482),  INT16_C( 11668), -INT16_C(  2937),  INT16_C( 20859), -INT16_C( 12110), -INT16_C(  8611),  INT16_C( 17999),  INT16_C(  6944) },
      UINT32_C(2030347617),
      { -INT16_C( 25811), -INT16_C(  6217), -INT16_C( 10782), -INT16_C( 21490), -INT16_C( 23846),  INT16_C( 25049),  INT16_C( 21910),  INT16_C( 18610),
         INT16_C(  4133),  INT16_C( 29734),  INT16_C( 18006), -INT16_C( 18544), -INT16_C( 27413),  INT16_C( 20273),  INT16_C( 11375), -INT16_C( 25560),
        -INT16_C(  7992), -INT16_C( 21885), -INT16_C( 28235), -INT16_C( 28842),  INT16_C( 12339), -INT16_C( 13840), -INT16_C( 23675), -INT16_C( 21999),
         INT16_C( 14515),  INT16_C(  2335), -INT16_C( 20610),  INT16_C( 27329), -INT16_C(  3517), -INT16_C( 19783), -INT16_C(  7906), -INT16_C(  6578) },
      { -INT16_C( 11583),  INT16_C( 30352), -INT16_C(  6301), -INT16_C( 26875), -INT16_C(  2537), -INT16_C( 25504),  INT16_C( 29337),  INT16_C( 19526),
         INT16_C( 26026),  INT16_C( 10325),  INT16_C(  5652),  INT16_C( 22674),  INT16_C( 19208),  INT16_C(  9994),  INT16_C( 22829), -INT16_C(  4595),
        -INT16_C( 25045), -INT16_C( 29083),  INT16_C( 27269), -INT16_C( 25563), -INT16_C( 31136), -INT16_C(  1736),  INT16_C( 32504), -INT16_C( 23995),
        -INT16_C( 25628), -INT16_C(  1846),  INT16_C( 23985), -INT16_C( 17840),  INT16_C( 23464), -INT16_C( 10783), -INT16_C(  4428), -INT16_C(  8252) },
      {  INT16_C( 28142),  INT16_C( 17201), -INT16_C(   484), -INT16_C(  4807), -INT16_C( 18583), -INT16_C(   455), -INT16_C( 14289),  INT16_C( 16294),
         INT16_C( 30159),  INT16_C( 28135),  INT16_C( 23658),  INT16_C( 20064), -INT16_C( 16695),  INT16_C( 30267),  INT16_C( 32300), -INT16_C( 30155),
         INT16_C(  6860), -INT16_C(  5755), -INT16_C(   966), -INT16_C( 32042),  INT16_C( 21110), -INT16_C(  8544),  INT16_C( 18106),  INT16_C(  9757),
        -INT16_C( 11113),  INT16_C( 11668), -INT16_C(  2937),  INT16_C(  9489),  INT16_C( 19947), -INT16_C( 30566), -INT16_C( 12334),  INT16_C(  6944) } },
    { {  INT16_C( 10636),  INT16_C(  4461), -INT16_C( 27757), -INT16_C(  2899), -INT16_C(  6887),  INT16_C(  4589),  INT16_C( 13156),  INT16_C( 18611),
         INT16_C( 32206),  INT16_C( 32576), -INT16_C( 28198), -INT16_C( 31943),  INT16_C(  6892), -INT16_C( 24488),  INT16_C(  7177), -INT16_C( 27265),
        -INT16_C(  5051), -INT16_C(  9817),  INT16_C( 21631), -INT16_C( 26419), -INT16_C( 17862), -INT16_C( 24919),  INT16_C( 23789), -INT16_C( 17434),
         INT16_C(  9946), -INT16_C( 19397),  INT16_C( 29879), -INT16_C( 23753), -INT16_C( 28529), -INT16_C( 26557), -INT16_C( 15700), -INT16_C(  3539) },
      UINT32_C( 785110191),
      {  INT16_C( 28754),  INT16_C( 16385), -INT16_C(  6195), -INT16_C( 22533),  INT16_C( 13837), -INT16_C( 15013), -INT16_C( 27733),  INT16_C( 14952),
        -INT16_C( 21469), -INT16_C( 12334), -INT16_C(   146),  INT16_C(  7617), -INT16_C( 29484), -INT16_C(   692),  INT16_C(  4900),  INT16_C( 30560),
         INT16_C( 24963),  INT16_C( 20663), -INT16_C( 19896),  INT16_C( 22007),  INT16_C( 21481), -INT16_C( 27622), -INT16_C( 31770),  INT16_C(  2510),
        -INT16_C( 24529), -INT16_C( 25128), -INT16_C( 25953),  INT16_C( 29627),  INT16_C(  1830),  INT16_C( 19312), -INT16_C( 12262), -INT16_C( 25150) },
      {  INT16_C( 31025),  INT16_C( 31214), -INT16_C(  6869),  INT16_C(  5327), -INT16_C(  5832),  INT16_C(  7848),  INT16_C( 30316), -INT16_C( 25817),
         INT16_C(    22), -INT16_C( 18887), -INT16_C(  2918), -INT16_C( 16343), -INT16_C( 25861),  INT16_C(  5387), -INT16_C( 12950), -INT16_C( 25422),
        -INT16_C( 24506),  INT16_C( 29205), -INT16_C(  7034), -INT16_C( 16762),  INT16_C( 12238),  INT16_C( 15069),  INT16_C(  1189), -INT16_C( 17194),
         INT16_C(  3844), -INT16_C( 24974), -INT16_C( 25853), -INT16_C(   417),  INT16_C( 27189), -INT16_C( 24557), -INT16_C( 15048),  INT16_C( 32316) },
      { -INT16_C(  5757), -INT16_C( 17937), -INT16_C( 13064), -INT16_C( 17206), -INT16_C(  6887), -INT16_C(  7165),  INT16_C( 13156), -INT16_C( 10865),
         INT16_C( 32206),  INT16_C( 32576), -INT16_C(  3064), -INT16_C( 31943),  INT16_C( 10191), -INT16_C( 24488), -INT16_C(  8050),  INT16_C(  5138),
         INT16_C(   457), -INT16_C( 15668),  INT16_C( 21631),  INT16_C(  5245), -INT16_C( 17862), -INT16_C( 24919), -INT16_C( 30581), -INT16_C( 14684),
         INT16_C(  9946),  INT16_C( 15434),  INT16_C( 13730),  INT16_C( 29210), -INT16_C( 28529), -INT16_C(  5245), -INT16_C( 15700), -INT16_C(  3539) } },
    { {  INT16_C( 20838), -INT16_C(  4880),  INT16_C( 30518),  INT16_C(  1194), -INT16_C( 30810),  INT16_C( 19262),  INT16_C(  5260), -INT16_C( 28665),
         INT16_C( 31011),  INT16_C(  9775), -INT16_C( 29163),  INT16_C( 18980),  INT16_C( 14328),  INT16_C( 12522),  INT16_C(  9981),  INT16_C( 25519),
        -INT16_C( 24712), -INT16_C( 20913), -INT16_C(  1770), -INT16_C( 17230), -INT16_C(  3967),  INT16_C(  3336),  INT16_C(  3845),  INT16_C( 10397),
        -INT16_C( 13175), -INT16_C( 25009),  INT16_C( 29530),  INT16_C( 21480), -INT16_C( 11349), -INT16_C( 22397),  INT16_C( 13049),  INT16_C( 28939) },
      UINT32_C(3894368978),
      { -INT16_C( 21054), -INT16_C( 14367),  INT16_C( 32700),  INT16_C( 17903),  INT16_C( 15947), -INT16_C( 22813), -INT16_C( 13134),  INT16_C( 24057),
         INT16_C( 31903), -INT16_C( 26619),  INT16_C(  4271), -INT16_C( 32502),  INT16_C( 10602), -INT16_C( 17047),  INT16_C(  3835), -INT16_C( 17006),
         INT16_C( 29627),  INT16_C( 30852),  INT16_C( 29682),  INT16_C( 16061), -INT16_C( 24142),  INT16_C( 25828), -INT16_C(  8851),  INT16_C(  3265),
        -INT16_C( 14759),  INT16_C(  2212), -INT16_C( 20778),  INT16_C( 16521), -INT16_C(  3112), -INT16_C( 11267), -INT16_C( 28927), -INT16_C( 17008) },
      {  INT16_C(  5123), -INT16_C(  2763), -INT16_C(  3449),  INT16_C( 14643),  INT16_C(  6035),  INT16_C(   157),  INT16_C( 24308),  INT16_C( 19980),
        -INT16_C( 20188), -INT16_C(  1450), -INT16_C(  8097),  INT16_C( 14138),  INT16_C( 14547), -INT16_C( 11254), -INT16_C( 25913), -INT16_C( 13679),
        -INT16_C( 14674),  INT16_C( 14016), -INT16_C(  3143),  INT16_C( 19567),  INT16_C(  3339), -INT16_C(   179),  INT16_C( 22891), -INT16_C( 28595),
        -INT16_C( 23542),  INT16_C( 27274), -INT16_C( 14972),  INT16_C( 22433), -INT16_C( 21251), -INT16_C( 15317), -INT16_C( 17082), -INT16_C(  2673) },
      {  INT16_C( 20838), -INT16_C( 17130),  INT16_C( 30518),  INT16_C(  1194),  INT16_C( 21982),  INT16_C( 19262),  INT16_C( 11174), -INT16_C( 21499),
         INT16_C( 31011), -INT16_C( 28069), -INT16_C( 29163), -INT16_C( 18364),  INT16_C( 25149),  INT16_C( 12522), -INT16_C( 22078),  INT16_C( 25519),
         INT16_C( 14953), -INT16_C( 20668),  INT16_C( 26539), -INT16_C( 29908), -INT16_C( 20803),  INT16_C(  3336),  INT16_C(  3845),  INT16_C( 10397),
        -INT16_C( 13175), -INT16_C( 25009),  INT16_C( 29530), -INT16_C( 26582), -INT16_C( 11349), -INT16_C( 26584),  INT16_C( 19527), -INT16_C( 19681) } },
    { {  INT16_C( 20355),  INT16_C( 15403), -INT16_C( 26046),  INT16_C( 19849), -INT16_C( 10585),  INT16_C(  4941), -INT16_C( 26065),  INT16_C( 15011),
         INT16_C( 11582), -INT16_C( 15708),  INT16_C( 17906), -INT16_C(  4327),  INT16_C( 17905),  INT16_C( 14516),  INT16_C( 17154), -INT16_C( 31443),
         INT16_C( 22674), -INT16_C( 11070),  INT16_C( 19442), -INT16_C( 26078),  INT16_C( 28449),  INT16_C( 20653),  INT16_C( 20489),  INT16_C( 18570),
         INT16_C( 11901),  INT16_C( 28682),  INT16_C(  9332),  INT16_C( 25951),  INT16_C(  4969),  INT16_C( 27549), -INT16_C( 13738), -INT16_C(  5904) },
      UINT32_C( 364753442),
      {  INT16_C( 23630),  INT16_C( 22383), -INT16_C(  1620),  INT16_C( 10655), -INT16_C( 21976), -INT16_C( 25447), -INT16_C(  1586),  INT16_C( 14081),
        -INT16_C( 24820),  INT16_C( 25506), -INT16_C( 28055), -INT16_C( 29621),  INT16_C(  2117),  INT16_C( 17057),  INT16_C( 20711),  INT16_C( 13665),
        -INT16_C( 12116),  INT16_C( 22669),  INT16_C( 11465), -INT16_C(  3711),  INT16_C(  7126), -INT16_C( 23411), -INT16_C( 28908),  INT16_C(  8411),
         INT16_C( 32046), -INT16_C( 26749), -INT16_C( 12528),  INT16_C( 21795), -INT16_C( 15145), -INT16_C( 16489), -INT16_C(  2028), -INT16_C( 16140) },
      { -INT16_C( 32312), -INT16_C( 28136), -INT16_C( 25938), -INT16_C( 31613),  INT16_C(  4533), -INT16_C( 14039),  INT16_C(  1184), -INT16_C( 12567),
         INT16_C( 28034), -INT16_C( 28059), -INT16_C( 30404),  INT16_C(  5095),  INT16_C( 32333),  INT16_C( 25298), -INT16_C( 14473),  INT16_C( 16162),
         INT16_C( 15176), -INT16_C(  2351),  INT16_C( 21973), -INT16_C( 30085), -INT16_C( 23450),  INT16_C(  1619),  INT16_C( 15528),  INT16_C( 10964),
         INT16_C( 14761), -INT16_C(  6724), -INT16_C( 23614),  INT16_C(  4345), -INT16_C( 13534), -INT16_C( 26254), -INT16_C( 27502), -INT16_C(  9256) },
      {  INT16_C( 20355), -INT16_C(  5753), -INT16_C( 26046),  INT16_C( 19849), -INT16_C( 10585),  INT16_C( 26050), -INT16_C( 26065),  INT16_C( 15011),
         INT16_C( 11582), -INT16_C(  2553),  INT16_C( 17906), -INT16_C(  4327), -INT16_C( 31086), -INT16_C( 23181),  INT16_C( 17154),  INT16_C( 29827),
         INT16_C(  3060), -INT16_C( 11070), -INT16_C( 32098),  INT16_C( 31740), -INT16_C( 16324), -INT16_C( 21792),  INT16_C( 20489),  INT16_C( 19375),
        -INT16_C( 18729),  INT16_C( 28682),  INT16_C( 29394),  INT16_C( 25951), -INT16_C( 28679),  INT16_C( 27549), -INT16_C( 13738), -INT16_C(  5904) } },
    { { -INT16_C( 21809), -INT16_C( 23343),  INT16_C( 19711),  INT16_C( 25902), -INT16_C( 32272), -INT16_C( 26261),  INT16_C( 16318),  INT16_C( 26563),
        -INT16_C( 32648),  INT16_C( 15181),  INT16_C( 17955),  INT16_C( 17739), -INT16_C( 17135), -INT16_C( 23330), -INT16_C( 18607),  INT16_C(  8575),
         INT16_C( 20577),  INT16_C( 24773), -INT16_C(  2915), -INT16_C( 29243),  INT16_C( 12405),  INT16_C( 13094), -INT16_C(  5521), -INT16_C(  6245),
        -INT16_C(  6038), -INT16_C( 29406),  INT16_C( 27950),  INT16_C( 16339), -INT16_C( 20182),  INT16_C( 31971),  INT16_C( 25192), -INT16_C( 13923) },
      UINT32_C(1344889523),
      {  INT16_C(  1054), -INT16_C( 29185), -INT16_C( 25874),  INT16_C( 22645), -INT16_C( 26750), -INT16_C( 20251), -INT16_C( 18427),  INT16_C( 12272),
        -INT16_C( 11414), -INT16_C( 11605),  INT16_C( 18486), -INT16_C(  5732), -INT16_C( 14933),  INT16_C(   313),  INT16_C(  5812), -INT16_C( 11571),
        -INT16_C( 13030),  INT16_C(  2144), -INT16_C( 10905), -INT16_C(  5536),  INT16_C( 18028),  INT16_C( 29082), -INT16_C( 29954),  INT16_C( 26785),
         INT16_C( 19550), -INT16_C( 27589), -INT16_C( 10347),  INT16_C( 16509), -INT16_C( 18788),  INT16_C( 20545),  INT16_C(  4044), -INT16_C(  6365) },
      { -INT16_C( 31780),  INT16_C( 17391),  INT16_C( 20568), -INT16_C( 15315), -INT16_C( 14186), -INT16_C( 27594), -INT16_C( 10414), -INT16_C( 20227),
         INT16_C( 14371), -INT16_C( 18364), -INT16_C( 16113), -INT16_C( 21512),  INT16_C( 14967),  INT16_C( 17660),  INT16_C(  8009),  INT16_C(  9515),
         INT16_C(  6818), -INT16_C(  1432), -INT16_C( 27030),  INT16_C(   190), -INT16_C(  2978), -INT16_C( 20331), -INT16_C( 27957), -INT16_C(  4255),
        -INT16_C( 23094), -INT16_C(  9817), -INT16_C( 24473), -INT16_C(  8572), -INT16_C( 32550),  INT16_C(  8994),  INT16_C( 19871),  INT16_C( 16712) },
      { -INT16_C( 30726), -INT16_C( 11794),  INT16_C( 19711),  INT16_C( 25902),  INT16_C( 24600),  INT16_C( 17691),  INT16_C( 16318), -INT16_C(  7955),
        -INT16_C( 32648), -INT16_C( 29969),  INT16_C( 17955),  INT16_C( 17739), -INT16_C( 17135),  INT16_C( 17973),  INT16_C( 13821),  INT16_C(  8575),
        -INT16_C(  6212),  INT16_C( 24773), -INT16_C(  2915), -INT16_C(  5346),  INT16_C( 12405),  INT16_C(  8751), -INT16_C(  5521), -INT16_C(  6245),
        -INT16_C(  6038), -INT16_C( 29406),  INT16_C( 27950),  INT16_C( 16339),  INT16_C( 14198),  INT16_C( 31971),  INT16_C( 23915), -INT16_C( 13923) } },
    { { -INT16_C( 20376), -INT16_C( 11717), -INT16_C(  1466), -INT16_C( 23341),  INT16_C( 26862), -INT16_C( 17835), -INT16_C( 18694), -INT16_C( 15191),
         INT16_C( 20571), -INT16_C( 15715),  INT16_C(  8688), -INT16_C( 13663), -INT16_C( 15454),  INT16_C( 16877),  INT16_C( 13585),  INT16_C( 31107),
        -INT16_C( 16666),  INT16_C( 11339),  INT16_C(  7864), -INT16_C( 22575),  INT16_C(  9862), -INT16_C( 32671),  INT16_C(  2780),  INT16_C( 14148),
        -INT16_C(  7846),  INT16_C( 19450), -INT16_C( 25853), -INT16_C( 23275),  INT16_C(   862),  INT16_C( 28646),  INT16_C( 26936),  INT16_C(  7912) },
      UINT32_C(3763024936),
      { -INT16_C(  6078),  INT16_C(  7769), -INT16_C( 24846),  INT16_C( 19797),  INT16_C( 20351), -INT16_C( 32104), -INT16_C( 21014),  INT16_C( 18727),
         INT16_C(  3760), -INT16_C(  5704), -INT16_C( 24201), -INT16_C( 24825),  INT16_C( 21205),  INT16_C( 10112),  INT16_C(  1902), -INT16_C( 20480),
         INT16_C( 23280), -INT16_C(  7474),  INT16_C(  9464),  INT16_C( 30511), -INT16_C( 14477),  INT16_C( 24314),  INT16_C(  8565),  INT16_C(  9639),
         INT16_C( 24367), -INT16_C( 22770),  INT16_C(  5632), -INT16_C( 10938), -INT16_C( 14744), -INT16_C( 10243), -INT16_C(   562), -INT16_C( 16761) },
      {  INT16_C( 22103),  INT16_C( 20384), -INT16_C( 12166), -INT16_C(  4665), -INT16_C( 15977),  INT16_C(  3147), -INT16_C(  3358),  INT16_C(  4658),
         INT16_C( 16466),  INT16_C( 21177), -INT16_C(   170), -INT16_C( 16600),  INT16_C(  9670), -INT16_C( 27498),  INT16_C(  7458),  INT16_C( 31314),
        -INT16_C(  3469), -INT16_C(  4663), -INT16_C( 28478),  INT16_C( 23259),  INT16_C(  9809),  INT16_C( 13414), -INT16_C( 26599),  INT16_C( 27462),
        -INT16_C(    39),  INT16_C( 12221), -INT16_C(  6658), -INT16_C( 15122), -INT16_C( 31734),  INT16_C( 11608), -INT16_C( 21854),  INT16_C(  5543) },
      { -INT16_C( 20376), -INT16_C( 11717), -INT16_C(  1466),  INT16_C( 15132),  INT16_C( 26862), -INT16_C( 28957), -INT16_C( 18694), -INT16_C( 15191),
         INT16_C( 20571), -INT16_C( 15715), -INT16_C( 24371), -INT16_C( 13663),  INT16_C( 30875), -INT16_C( 17386),  INT16_C( 13585),  INT16_C( 31107),
         INT16_C( 19811), -INT16_C( 12137),  INT16_C(  7864), -INT16_C( 11766),  INT16_C(  9862), -INT16_C( 32671), -INT16_C( 18034),  INT16_C( 14148),
        -INT16_C(  7846),  INT16_C( 19450), -INT16_C( 25853), -INT16_C( 23275),  INT16_C(   862),  INT16_C(  1365), -INT16_C( 22416), -INT16_C( 11218) } },
    { {  INT16_C( 28829),  INT16_C( 24323), -INT16_C(  8703),  INT16_C( 21177),  INT16_C(  8196),  INT16_C(  7558), -INT16_C( 13128), -INT16_C( 28280),
         INT16_C( 18123), -INT16_C( 13631), -INT16_C( 20693),  INT16_C( 13966), -INT16_C(  6348), -INT16_C( 10653),  INT16_C(  2705),  INT16_C( 12011),
        -INT16_C(  4486),  INT16_C( 31630),  INT16_C( 18380), -INT16_C( 11826),  INT16_C( 21607),  INT16_C(  8430),  INT16_C( 30497), -INT16_C(  4943),
         INT16_C( 29373), -INT16_C(  5962),  INT16_C( 17698),  INT16_C( 22046), -INT16_C( 32468), -INT16_C( 17108),  INT16_C(  6027),  INT16_C(  1772) },
      UINT32_C(3531700742),
      { -INT16_C( 27996), -INT16_C( 15031), -INT16_C(  1527), -INT16_C( 14671),  INT16_C( 26733), -INT16_C( 28754), -INT16_C( 12883), -INT16_C(  9755),
         INT16_C(  4430), -INT16_C(  9578), -INT16_C( 32216),  INT16_C( 12000),  INT16_C( 25084), -INT16_C( 16895), -INT16_C( 23375),  INT16_C( 21991),
         INT16_C( 12342),  INT16_C( 16154), -INT16_C( 13526), -INT16_C( 26875), -INT16_C( 19405), -INT16_C(  8154),  INT16_C(  2945), -INT16_C( 12359),
         INT16_C( 20508),  INT16_C( 17833), -INT16_C( 30254), -INT16_C( 12429),  INT16_C( 29931), -INT16_C( 25459),  INT16_C( 29721),  INT16_C( 20465) },
      {  INT16_C(  2980), -INT16_C( 12657), -INT16_C( 27434),  INT16_C(  2662), -INT16_C( 29624), -INT16_C( 13846), -INT16_C( 23400), -INT16_C( 19303),
         INT16_C( 17140), -INT16_C( 14599),  INT16_C( 28108), -INT16_C( 18539),  INT16_C(  8929), -INT16_C(  1453),  INT16_C( 17558),  INT16_C( 14922),
        -INT16_C(  9905),  INT16_C(  9481),  INT16_C( 28525), -INT16_C( 18897),  INT16_C(  6907), -INT16_C( 27777),  INT16_C(  6334), -INT16_C( 19896),
         INT16_C( 16731),  INT16_C( 10104),  INT16_C(  3758), -INT16_C( 28450),  INT16_C( 12592), -INT16_C( 14454), -INT16_C( 11147), -INT16_C( 15359) },
      {  INT16_C( 28829), -INT16_C( 27688), -INT16_C( 28961),  INT16_C( 21177),  INT16_C(  8196),  INT16_C(  7558), -INT16_C( 13128), -INT16_C( 28280),
         INT16_C( 18123), -INT16_C( 24177), -INT16_C( 20693), -INT16_C(  6539), -INT16_C( 31523), -INT16_C( 18348), -INT16_C(  5817),  INT16_C( 12011),
         INT16_C(  2437),  INT16_C( 31630),  INT16_C( 18380), -INT16_C( 11826),  INT16_C( 21607),  INT16_C(  8430),  INT16_C( 30497), -INT16_C( 32255),
         INT16_C( 29373),  INT16_C( 27937),  INT16_C( 17698),  INT16_C( 22046), -INT16_C( 23013), -INT16_C( 17108),  INT16_C( 18574),  INT16_C(  5106) } },
    { {  INT16_C(  2733),  INT16_C(  7145),  INT16_C(  6521),  INT16_C( 30161),  INT16_C( 20531), -INT16_C(  3832),  INT16_C( 20585), -INT16_C( 15197),
         INT16_C(  7058),  INT16_C( 16619), -INT16_C( 14039),  INT16_C( 23248),  INT16_C( 23546),  INT16_C( 28449),  INT16_C(  8751), -INT16_C(  8909),
         INT16_C(  7213), -INT16_C( 22792), -INT16_C( 14027),  INT16_C( 26651),  INT16_C(  9241), -INT16_C( 32167), -INT16_C(   908),  INT16_C(  1606),
         INT16_C( 12568),  INT16_C( 16711),  INT16_C(  6138), -INT16_C(  2917), -INT16_C( 17294), -INT16_C( 23965), -INT16_C( 26913),  INT16_C(  3199) },
      UINT32_C(3904010163),
      { -INT16_C( 21774),  INT16_C( 26332),  INT16_C(  8871), -INT16_C( 16531), -INT16_C( 19372),  INT16_C( 19968), -INT16_C( 25397),  INT16_C( 15939),
        -INT16_C( 22952),  INT16_C( 14304),  INT16_C( 24381), -INT16_C(  4029), -INT16_C(  2346),  INT16_C(  5848),  INT16_C( 10692), -INT16_C( 18833),
         INT16_C( 19412),  INT16_C( 31516), -INT16_C( 30354), -INT16_C( 15814),  INT16_C( 14909),  INT16_C(  2320),  INT16_C( 21462),  INT16_C( 12103),
         INT16_C( 10234),  INT16_C( 14182), -INT16_C( 21882),  INT16_C( 23591), -INT16_C(    96),  INT16_C( 25714), -INT16_C(  7895), -INT16_C(   742) },
      {  INT16_C( 13869), -INT16_C( 25736), -INT16_C( 19776), -INT16_C(   675),  INT16_C( 28140), -INT16_C( 15610),  INT16_C( 19905), -INT16_C( 17422),
         INT16_C( 22644), -INT16_C(  1294),  INT16_C(  6402), -INT16_C( 23978), -INT16_C( 14312),  INT16_C( 16646),  INT16_C(  8362), -INT16_C( 10434),
        -INT16_C( 18857),  INT16_C(  6002), -INT16_C( 12440),  INT16_C( 21780),  INT16_C(  6972), -INT16_C(   744),  INT16_C(  2664), -INT16_C(  8776),
        -INT16_C( 21918),  INT16_C( 26071),  INT16_C( 11971), -INT16_C(  9209),  INT16_C(  3830), -INT16_C( 24547),  INT16_C( 23598), -INT16_C( 31369) },
      { -INT16_C(  7905),  INT16_C(   596),  INT16_C(  6521),  INT16_C( 30161),  INT16_C(  8768),  INT16_C(  4358),  INT16_C( 20585), -INT16_C(  1483),
        -INT16_C(   308),  INT16_C( 13010),  INT16_C( 30783),  INT16_C( 23248), -INT16_C( 16658),  INT16_C( 22494),  INT16_C( 19054), -INT16_C(  8909),
         INT16_C(  7213), -INT16_C( 28018), -INT16_C( 14027),  INT16_C( 26651),  INT16_C( 21881),  INT16_C(  1576), -INT16_C(   908),  INT16_C(  3327),
         INT16_C( 12568),  INT16_C( 16711),  INT16_C(  6138),  INT16_C( 14382), -INT16_C( 17294),  INT16_C(  1167),  INT16_C( 15703), -INT16_C( 32111) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_add_epi16(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_add_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_add_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { UINT32_C(1970569674),
      { -INT16_C(  4224), -INT16_C( 11503), -INT16_C( 18041),  INT16_C(  6436), -INT16_C(  9270), -INT16_C( 15814), -INT16_C( 27569),  INT16_C( 13042),
        -INT16_C( 23737), -INT16_C( 13044),  INT16_C(  7220), -INT16_C(   316),  INT16_C( 14494), -INT16_C(   908),  INT16_C( 27220), -INT16_C( 11118),
        -INT16_C( 23719), -INT16_C(  8025), -INT16_C( 13476),  INT16_C(  9977),  INT16_C( 13479), -INT16_C(  2328), -INT16_C(  9528),  INT16_C(  3881),
         INT16_C( 13693), -INT16_C( 19747), -INT16_C( 24239), -INT16_C(  4176),  INT16_C(  9433),  INT16_C( 11756),  INT16_C( 32398), -INT16_C(  6143) },
      { -INT16_C( 22239),  INT16_C( 32200), -INT16_C( 15756),  INT16_C(  7075), -INT16_C( 29450), -INT16_C( 16878),  INT16_C( 15206), -INT16_C(  6962),
        -INT16_C( 21648), -INT16_C( 15978),  INT16_C( 17996),  INT16_C(  9649), -INT16_C( 25237), -INT16_C(  1709),  INT16_C( 21531),  INT16_C( 15585),
        -INT16_C( 21763),  INT16_C( 29369),  INT16_C( 23660),  INT16_C( 25229), -INT16_C( 24600),  INT16_C( 20256), -INT16_C(  4390),  INT16_C( 18995),
        -INT16_C( 13927), -INT16_C(  6900), -INT16_C( 17137),  INT16_C( 31243),  INT16_C( 24154),  INT16_C( 30068),  INT16_C( 21938), -INT16_C( 20303) },
      {  INT16_C(     0),  INT16_C( 20697),  INT16_C(     0),  INT16_C( 13511),  INT16_C(     0),  INT16_C(     0), -INT16_C( 12363),  INT16_C(  6080),
         INT16_C( 20151),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4467),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 10184),  INT16_C(     0), -INT16_C( 11121),  INT16_C( 17928), -INT16_C( 13918),  INT16_C(     0),
        -INT16_C(   234),  INT16_C(     0),  INT16_C( 24160),  INT16_C(     0), -INT16_C( 31949), -INT16_C( 23712), -INT16_C( 11200),  INT16_C(     0) } },
    { UINT32_C(1797417727),
      { -INT16_C(  4529),  INT16_C( 10750),  INT16_C( 12764),  INT16_C( 30324), -INT16_C( 32518),  INT16_C(  2395),  INT16_C( 26173), -INT16_C( 26748),
        -INT16_C(  1852),  INT16_C( 30476), -INT16_C( 17075),  INT16_C( 19751),  INT16_C( 18727), -INT16_C(  4680), -INT16_C( 30984),  INT16_C( 18332),
        -INT16_C( 25996),  INT16_C( 20593), -INT16_C(  6709), -INT16_C( 14906),  INT16_C(  8805), -INT16_C( 23857),  INT16_C( 21384),  INT16_C( 19769),
         INT16_C( 17739), -INT16_C( 26428), -INT16_C(  5374),  INT16_C( 10725), -INT16_C( 25036),  INT16_C( 11286), -INT16_C( 19676), -INT16_C( 26508) },
      { -INT16_C(  6835),  INT16_C(  6632), -INT16_C( 20534),  INT16_C( 12254), -INT16_C( 21039),  INT16_C( 22993),  INT16_C(  2560),  INT16_C( 19366),
         INT16_C( 27215),  INT16_C( 20964), -INT16_C( 13995), -INT16_C( 30342), -INT16_C( 28569), -INT16_C( 29770),  INT16_C( 10819), -INT16_C( 28381),
         INT16_C(  3087), -INT16_C(  9814), -INT16_C( 30533), -INT16_C( 29688), -INT16_C(  9930),  INT16_C( 14053), -INT16_C( 29469),  INT16_C( 12930),
         INT16_C( 26358),  INT16_C( 19587), -INT16_C(   721), -INT16_C( 26667), -INT16_C( 29811), -INT16_C( 11998),  INT16_C( 18101), -INT16_C( 15262) },
      { -INT16_C( 11364),  INT16_C( 17382), -INT16_C(  7770), -INT16_C( 22958),  INT16_C( 11979),  INT16_C( 25388),  INT16_C( 28733), -INT16_C(  7382),
         INT16_C(     0), -INT16_C( 14096),  INT16_C(     0), -INT16_C( 10591),  INT16_C(     0),  INT16_C( 31086), -INT16_C( 20165),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 10779),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  9804),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 21439), -INT16_C(  6841),  INT16_C(     0), -INT16_C( 15942),  INT16_C(     0), -INT16_C(   712), -INT16_C(  1575),  INT16_C(     0) } },
    { UINT32_C( 228396114),
      {  INT16_C( 32382),  INT16_C( 24833), -INT16_C( 31990),  INT16_C(   403),  INT16_C(  5865),  INT16_C(  6221),  INT16_C(  8723), -INT16_C( 24145),
        -INT16_C( 11602),  INT16_C( 25458), -INT16_C( 11240),  INT16_C( 27176), -INT16_C( 14880),  INT16_C( 29815),  INT16_C(  4203), -INT16_C(  5825),
         INT16_C( 16526), -INT16_C( 26293), -INT16_C(  8509), -INT16_C( 21350), -INT16_C(  6155),  INT16_C(  2244),  INT16_C( 29705), -INT16_C( 18519),
         INT16_C(  6982),  INT16_C( 24091),  INT16_C( 17391), -INT16_C( 12344),  INT16_C( 16136),  INT16_C( 29508), -INT16_C( 31921), -INT16_C(  8867) },
      { -INT16_C( 22333), -INT16_C( 31114),  INT16_C(  4230),  INT16_C( 31538), -INT16_C(  2313),  INT16_C(   388),  INT16_C( 11626), -INT16_C( 20296),
        -INT16_C( 11447),  INT16_C( 14350), -INT16_C( 10730),  INT16_C(  7944),  INT16_C( 19477),  INT16_C( 25746), -INT16_C(  4145), -INT16_C( 28094),
        -INT16_C( 18281),  INT16_C(  7704),  INT16_C( 19145), -INT16_C( 16231),  INT16_C(  7488), -INT16_C( 21567),  INT16_C( 31307), -INT16_C( 27557),
         INT16_C( 27213),  INT16_C( 25804), -INT16_C( 11200),  INT16_C( 22147),  INT16_C(  5408), -INT16_C(  4166), -INT16_C(  1019), -INT16_C( 25471) },
      {  INT16_C(     0), -INT16_C(  6281),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3552),  INT16_C(     0),  INT16_C( 20349),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 21970), -INT16_C( 30416),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
        -INT16_C(  1755),  INT16_C(     0),  INT16_C( 10636),  INT16_C( 27955),  INT16_C(  1333),  INT16_C(     0),  INT16_C(     0),  INT16_C( 19460),
        -INT16_C( 31341),  INT16_C(     0),  INT16_C(  6191),  INT16_C(  9803),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(2126158261),
      {  INT16_C(   113), -INT16_C( 17201),  INT16_C( 10874), -INT16_C( 14512),  INT16_C(  7572), -INT16_C( 10965), -INT16_C( 20751),  INT16_C(  4651),
        -INT16_C(  6716), -INT16_C( 14079), -INT16_C( 31774), -INT16_C( 26779),  INT16_C(  8220),  INT16_C(    21),  INT16_C( 21364), -INT16_C(  6876),
        -INT16_C(  3245), -INT16_C( 12894), -INT16_C(  3555), -INT16_C( 19819), -INT16_C( 16369),  INT16_C(   391), -INT16_C( 19857),  INT16_C( 13075),
         INT16_C(  5271),  INT16_C( 31228),  INT16_C( 24983), -INT16_C( 19440),  INT16_C(  9601), -INT16_C(  2636), -INT16_C( 10119), -INT16_C( 13093) },
      {  INT16_C( 32203), -INT16_C(  5990),  INT16_C( 12143),  INT16_C( 32666),  INT16_C(  8687),  INT16_C( 24192), -INT16_C( 27693),  INT16_C( 27537),
        -INT16_C( 29273),  INT16_C( 16356), -INT16_C(  2577),  INT16_C( 28915), -INT16_C( 22758), -INT16_C( 27802),  INT16_C( 16767),  INT16_C( 19040),
        -INT16_C(  1346),  INT16_C( 11570), -INT16_C( 13015),  INT16_C(  6316),  INT16_C( 11502), -INT16_C( 15753),  INT16_C(  2239),  INT16_C( 26413),
         INT16_C(  4502), -INT16_C( 31322), -INT16_C( 26362),  INT16_C(  8693),  INT16_C( 23360), -INT16_C( 16460),  INT16_C(  5276),  INT16_C( 23049) },
      {  INT16_C( 32316),  INT16_C(     0),  INT16_C( 23017),  INT16_C(     0),  INT16_C( 16259),  INT16_C( 13227),  INT16_C(     0),  INT16_C( 32188),
         INT16_C( 29547),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2136), -INT16_C( 14538),  INT16_C(     0),  INT16_C(     0),  INT16_C( 12164),
         INT16_C(     0), -INT16_C(  1324),  INT16_C(     0), -INT16_C( 13503), -INT16_C(  4867), -INT16_C( 15362),  INT16_C(     0), -INT16_C( 26048),
         INT16_C(     0), -INT16_C(    94), -INT16_C(  1379), -INT16_C( 10747), -INT16_C( 32575), -INT16_C( 19096), -INT16_C(  4843),  INT16_C(     0) } },
    { UINT32_C( 931674894),
      { -INT16_C( 14495),  INT16_C(  8377), -INT16_C(  6449),  INT16_C( 25991),  INT16_C( 11767), -INT16_C(   278), -INT16_C(  7994),  INT16_C(  1567),
        -INT16_C( 11461), -INT16_C( 10043), -INT16_C( 12568), -INT16_C(  2510), -INT16_C( 17910),  INT16_C(  4654),  INT16_C( 32495),  INT16_C( 20489),
        -INT16_C( 15803),  INT16_C(  5232), -INT16_C(  1880), -INT16_C( 24454),  INT16_C( 25637), -INT16_C(  4962), -INT16_C( 17084), -INT16_C( 32526),
        -INT16_C( 18288),  INT16_C( 30808), -INT16_C( 30074), -INT16_C( 28561), -INT16_C( 25275),  INT16_C( 13475), -INT16_C( 21477),  INT16_C( 24708) },
      { -INT16_C(  2961),  INT16_C(  6004), -INT16_C(  4372),  INT16_C(  4791),  INT16_C( 21843), -INT16_C( 26626), -INT16_C(  4078), -INT16_C( 23785),
         INT16_C( 28584),  INT16_C( 12059), -INT16_C( 29958),  INT16_C( 16319),  INT16_C( 25127),  INT16_C( 17011), -INT16_C(  2289),  INT16_C( 32418),
         INT16_C(  6123), -INT16_C( 10091),  INT16_C( 19717),  INT16_C( 22762), -INT16_C(  5982), -INT16_C( 18960),  INT16_C(  2008), -INT16_C( 32424),
         INT16_C( 29559),  INT16_C( 29104),  INT16_C( 28670),  INT16_C(  9648),  INT16_C(  9170), -INT16_C(  7832),  INT16_C(  2586),  INT16_C(  1375) },
      {  INT16_C(     0),  INT16_C( 14381), -INT16_C( 10821),  INT16_C( 30782),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 17123),  INT16_C(  2016),  INT16_C(     0),  INT16_C( 13809),  INT16_C(  7217),  INT16_C( 21665),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  1692),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(   586),
         INT16_C( 11271), -INT16_C(  5624), -INT16_C(  1404),  INT16_C(     0), -INT16_C( 16105),  INT16_C(  5643),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C( 668857377),
      {  INT16_C( 28591), -INT16_C( 30567), -INT16_C(  3721), -INT16_C(  4599), -INT16_C( 18076),  INT16_C( 25183),  INT16_C(  3880), -INT16_C(  1400),
        -INT16_C(  4046),  INT16_C( 19675),  INT16_C( 15098),  INT16_C(  7249),  INT16_C( 12079),  INT16_C( 28739), -INT16_C( 15626), -INT16_C( 22956),
        -INT16_C(  4814), -INT16_C( 22226),  INT16_C( 14302),  INT16_C( 17303), -INT16_C(  2320),  INT16_C(  6309),  INT16_C( 11525),  INT16_C( 14099),
        -INT16_C(  4579),  INT16_C(  6275), -INT16_C( 11223),  INT16_C( 22580),  INT16_C( 30467), -INT16_C(  1336),  INT16_C(  7481),  INT16_C( 27552) },
      { -INT16_C( 12790), -INT16_C(  5868), -INT16_C( 21755), -INT16_C(  2772), -INT16_C( 11871), -INT16_C( 23027),  INT16_C(  8447),  INT16_C(  7389),
         INT16_C( 24591),  INT16_C( 14388),  INT16_C( 26677),  INT16_C( 14480),  INT16_C( 22751),  INT16_C(  6450), -INT16_C( 11659), -INT16_C( 32636),
        -INT16_C( 26208), -INT16_C( 23191), -INT16_C( 27324), -INT16_C(  6502), -INT16_C( 22426),  INT16_C( 25996),  INT16_C( 27336), -INT16_C( 10366),
        -INT16_C( 18742), -INT16_C(   241), -INT16_C( 24801), -INT16_C(   456),  INT16_C( 27384),  INT16_C( 27927), -INT16_C( 25539), -INT16_C(  8723) },
      {  INT16_C( 15801),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2156),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 23761),  INT16_C(     0), -INT16_C( 30706), -INT16_C( 30347), -INT16_C( 27285),  INT16_C(  9944),
        -INT16_C( 31022),  INT16_C(     0), -INT16_C( 13022),  INT16_C( 10801), -INT16_C( 24746),  INT16_C(     0), -INT16_C( 26675),  INT16_C(  3733),
        -INT16_C( 23321),  INT16_C(  6034),  INT16_C( 29512),  INT16_C(     0),  INT16_C(     0),  INT16_C( 26591),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(2038650421),
      { -INT16_C(  4923), -INT16_C( 29001),  INT16_C( 14678),  INT16_C(  8293),  INT16_C( 30192),  INT16_C(  3872),  INT16_C( 22548),  INT16_C(  3085),
         INT16_C(  9666), -INT16_C(   134),  INT16_C( 26561), -INT16_C(  2339),  INT16_C( 24766), -INT16_C( 22161), -INT16_C( 12419),  INT16_C( 17403),
        -INT16_C( 19525),  INT16_C(  4561),  INT16_C( 14060), -INT16_C(  9167),  INT16_C( 20907), -INT16_C( 16149), -INT16_C(  1623),  INT16_C( 27852),
         INT16_C( 17950), -INT16_C(  8341),  INT16_C( 18606),  INT16_C( 27861),  INT16_C( 17576),  INT16_C(  9749),  INT16_C(  4371), -INT16_C( 12695) },
      {  INT16_C( 15044), -INT16_C( 20257),  INT16_C(  4464),  INT16_C(  7309),  INT16_C( 30818),  INT16_C(  3292), -INT16_C( 22415), -INT16_C( 28808),
        -INT16_C(  7185), -INT16_C( 25234),  INT16_C( 17196), -INT16_C( 11255),  INT16_C(  7816), -INT16_C( 25606),  INT16_C( 25391), -INT16_C(  3222),
         INT16_C( 18845),  INT16_C(  3748),  INT16_C( 12634), -INT16_C( 17110),  INT16_C(  1705),  INT16_C(  7113),  INT16_C( 16814), -INT16_C( 25174),
         INT16_C(  6436),  INT16_C( 20538),  INT16_C( 17244), -INT16_C(  7131),  INT16_C(  8034), -INT16_C( 28288), -INT16_C(  5501),  INT16_C(  8325) },
      {  INT16_C( 10121),  INT16_C(     0),  INT16_C( 19142),  INT16_C(     0), -INT16_C(  4526),  INT16_C(  7164),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 25368), -INT16_C( 21779),  INT16_C(     0),  INT16_C( 32582),  INT16_C(     0),  INT16_C( 12972),  INT16_C(     0),
        -INT16_C(   680),  INT16_C(  8309),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2678),
         INT16_C( 24386),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20730),  INT16_C( 25610), -INT16_C( 18539), -INT16_C(  1130),  INT16_C(     0) } },
    { UINT32_C(2385389875),
      {  INT16_C(  5214),  INT16_C(  3358), -INT16_C( 13995),  INT16_C( 31146), -INT16_C(  6686),  INT16_C( 16074), -INT16_C(  4312), -INT16_C( 30173),
        -INT16_C( 23794), -INT16_C( 28388), -INT16_C( 24179), -INT16_C( 16206), -INT16_C(  7990),  INT16_C(  9294), -INT16_C( 26311), -INT16_C( 26841),
         INT16_C( 18093),  INT16_C(   676),  INT16_C( 20239), -INT16_C(  3716),  INT16_C( 17972),  INT16_C( 23599),  INT16_C( 21045),  INT16_C( 17383),
         INT16_C(  1013), -INT16_C( 32043), -INT16_C( 30812),  INT16_C( 28227), -INT16_C( 28313), -INT16_C( 24430), -INT16_C( 18133), -INT16_C( 10184) },
      { -INT16_C(  8961),  INT16_C(  3803),  INT16_C( 22315),  INT16_C( 24575),  INT16_C( 12189), -INT16_C( 11588), -INT16_C( 23679),  INT16_C( 30485),
        -INT16_C(  5466),  INT16_C( 19193),  INT16_C( 15473), -INT16_C(  9800),  INT16_C( 19150), -INT16_C(  1671), -INT16_C( 20221),  INT16_C(   977),
        -INT16_C( 21362), -INT16_C( 18159),  INT16_C(  4355), -INT16_C( 24551), -INT16_C( 10944), -INT16_C( 16014), -INT16_C( 30600),  INT16_C(  7736),
         INT16_C( 12914), -INT16_C(  7064),  INT16_C(  8302),  INT16_C( 15549),  INT16_C( 13930),  INT16_C( 27957),  INT16_C(  2024),  INT16_C( 30320) },
      { -INT16_C(  3747),  INT16_C(  7161),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5503),  INT16_C(  4486),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 29260),  INT16_C(     0),  INT16_C(     0), -INT16_C( 26006),  INT16_C(     0),  INT16_C(  7623),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 17483),  INT16_C( 24594), -INT16_C( 28267),  INT16_C(     0),  INT16_C(  7585),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 26429), -INT16_C( 22510), -INT16_C( 21760),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20136) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_add_epi16(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_add_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_add_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    int32_t a[16];
    int32_t b[16];
    int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  2023322181), -INT32_C(   153083711),  INT32_C(  1496228679), -INT32_C(  1879098037),  INT32_C(   556222349), -INT32_C(  1406744359),  INT32_C(   328840924),  INT32_C(   558351974),
         INT32_C(   363441491),  INT32_C(   353077710),  INT32_C(  2003712300),  INT32_C(    50752886), -INT32_C(  1926943052), -INT32_C(   767996938),  INT32_C(  1759891970), -INT32_C(   326488680) },
      { -INT32_C(  1476316198), -INT32_C(   390198084), -INT32_C(  1151325115),  INT32_C(  1321166490), -INT32_C(  1965235052),  INT32_C(   828118319), -INT32_C(  2137373976),  INT32_C(  1231823983),
         INT32_C(   334654807), -INT32_C(  1090801543),  INT32_C(  1987730396),  INT32_C(  1455765954),  INT32_C(  1289855261), -INT32_C(  1635893834), -INT32_C(   299952001), -INT32_C(  1825010884) },
      {  INT32_C(   795328917), -INT32_C(   543281795),  INT32_C(   344903564), -INT32_C(   557931547), -INT32_C(  1409012703), -INT32_C(   578626040), -INT32_C(  1808533052),  INT32_C(  1790175957),
         INT32_C(   698096298), -INT32_C(   737723833), -INT32_C(   303524600),  INT32_C(  1506518840), -INT32_C(   637087791),  INT32_C(  1891076524),  INT32_C(  1459939969),  INT32_C(  2143467732) } },
    { {  INT32_C(  1923492601), -INT32_C(  1238261286), -INT32_C(  1087525891),  INT32_C(    18215652),  INT32_C(  1229846163), -INT32_C(  1276589260),  INT32_C(   530712547), -INT32_C(  1951212910),
        -INT32_C(   553756668), -INT32_C(   141283334), -INT32_C(  1095319078),  INT32_C(  1186974643), -INT32_C(   158331710), -INT32_C(  1146521384),  INT32_C(   299584383),  INT32_C(   698191141) },
      { -INT32_C(   536372250), -INT32_C(  1529373494),  INT32_C(   291671389),  INT32_C(   441917784),  INT32_C(    84993837), -INT32_C(   557729185),  INT32_C(   737188869),  INT32_C(   257199401),
        -INT32_C(   219194328),  INT32_C(  1435944696), -INT32_C(  1402537901),  INT32_C(  1187429913),  INT32_C(    88922021), -INT32_C(  1763504751), -INT32_C(   759049303), -INT32_C(  1998449056) },
      {  INT32_C(  1387120351),  INT32_C(  1527332516), -INT32_C(   795854502),  INT32_C(   460133436),  INT32_C(  1314840000), -INT32_C(  1834318445),  INT32_C(  1267901416), -INT32_C(  1694013509),
        -INT32_C(   772950996),  INT32_C(  1294661362),  INT32_C(  1797110317), -INT32_C(  1920562740), -INT32_C(    69409689),  INT32_C(  1384941161), -INT32_C(   459464920), -INT32_C(  1300257915) } },
    { {  INT32_C(  1786433906), -INT32_C(   339799912),  INT32_C(   563553800), -INT32_C(  1989648668), -INT32_C(   963726283),  INT32_C(  1784443585), -INT32_C(  1506009531), -INT32_C(  1506927052),
        -INT32_C(  2012173840), -INT32_C(  1032597575), -INT32_C(   639431691), -INT32_C(  1637659799), -INT32_C(  1067126273), -INT32_C(  1456816029),  INT32_C(   307193822),  INT32_C(  1975025029) },
      { -INT32_C(   520239066), -INT32_C(  1918733928), -INT32_C(   446200452), -INT32_C(   796669231),  INT32_C(   529655739), -INT32_C(  2033665113), -INT32_C(  1466427614), -INT32_C(  1155706476),
        -INT32_C(  1315235047),  INT32_C(   138362252), -INT32_C(  1813141822),  INT32_C(   728002672), -INT32_C(    28641961), -INT32_C(   746319184), -INT32_C(  1099227863), -INT32_C(  2022074258) },
      {  INT32_C(  1266194840),  INT32_C(  2036433456),  INT32_C(   117353348),  INT32_C(  1508649397), -INT32_C(   434070544), -INT32_C(   249221528),  INT32_C(  1322530151),  INT32_C(  1632333768),
         INT32_C(   967558409), -INT32_C(   894235323),  INT32_C(  1842393783), -INT32_C(   909657127), -INT32_C(  1095768234),  INT32_C(  2091832083), -INT32_C(   792034041), -INT32_C(    47049229) } },
    { {  INT32_C(  1060705459),  INT32_C(   323450961), -INT32_C(  1901644770), -INT32_C(    71758940), -INT32_C(  1325792256),  INT32_C(  1082359318),  INT32_C(   167706267),  INT32_C(  1251047319),
        -INT32_C(   594883957),  INT32_C(  1626329410), -INT32_C(  1427204602), -INT32_C(  1582913631), -INT32_C(  1034772309), -INT32_C(  1174219490),  INT32_C(  1807941844),  INT32_C(    45438071) },
      {  INT32_C(  1625177886),  INT32_C(   398511377),  INT32_C(    96579172),  INT32_C(    27748182),  INT32_C(   650377479), -INT32_C(  1562327602),  INT32_C(  1007526853),  INT32_C(   373212152),
         INT32_C(   326573058),  INT32_C(  1311389674),  INT32_C(  1012133094),  INT32_C(  1530788435), -INT32_C(  1031732749), -INT32_C(  1939578426), -INT32_C(    53972476),  INT32_C(   923993909) },
      { -INT32_C(  1609083951),  INT32_C(   721962338), -INT32_C(  1805065598), -INT32_C(    44010758), -INT32_C(   675414777), -INT32_C(   479968284),  INT32_C(  1175233120),  INT32_C(  1624259471),
        -INT32_C(   268310899), -INT32_C(  1357248212), -INT32_C(   415071508), -INT32_C(    52125196), -INT32_C(  2066505058),  INT32_C(  1181169380),  INT32_C(  1753969368),  INT32_C(   969431980) } },
    { {  INT32_C(   223054371), -INT32_C(  1487178303), -INT32_C(  1243369631), -INT32_C(  1659887191), -INT32_C(   396390110), -INT32_C(   160119822),  INT32_C(  1794325813),  INT32_C(  1738671684),
         INT32_C(  1366683024), -INT32_C(   990261150),  INT32_C(   695852159),  INT32_C(   533105149),  INT32_C(   201860378),  INT32_C(   503479528), -INT32_C(    41355847), -INT32_C(  1956304133) },
      {  INT32_C(  2061359639),  INT32_C(   708761258), -INT32_C(  1336690766),  INT32_C(  1523521856),  INT32_C(   644273982),  INT32_C(   222586964),  INT32_C(  1493945694),  INT32_C(   266694903),
        -INT32_C(   192298422),  INT32_C(  1243531160), -INT32_C(  1090883202), -INT32_C(   937899382), -INT32_C(   168853855),  INT32_C(  1141060582), -INT32_C(   123859456), -INT32_C(   939031682) },
      { -INT32_C(  2010553286), -INT32_C(   778417045),  INT32_C(  1714906899), -INT32_C(   136365335),  INT32_C(   247883872),  INT32_C(    62467142), -INT32_C(  1006695789),  INT32_C(  2005366587),
         INT32_C(  1174384602),  INT32_C(   253270010), -INT32_C(   395031043), -INT32_C(   404794233),  INT32_C(    33006523),  INT32_C(  1644540110), -INT32_C(   165215303),  INT32_C(  1399631481) } },
    { { -INT32_C(   574844859), -INT32_C(   718808233), -INT32_C(   678223284), -INT32_C(  1918915604),  INT32_C(   260279849), -INT32_C(  1034647870),  INT32_C(   314241684), -INT32_C(  1160068747),
        -INT32_C(  1466460591), -INT32_C(  1099055503), -INT32_C(   862646048), -INT32_C(   463850309), -INT32_C(  2047550013), -INT32_C(   146323357), -INT32_C(  1358364102),  INT32_C(   359261123) },
      { -INT32_C(   339935111), -INT32_C(  1616299074),  INT32_C(   124468811),  INT32_C(   904643954),  INT32_C(    96133026),  INT32_C(  1643905575), -INT32_C(   955251452),  INT32_C(  1658616296),
         INT32_C(   944609913),  INT32_C(   551024341), -INT32_C(  1507376588), -INT32_C(  1428417784),  INT32_C(   447780594), -INT32_C(  1669616488), -INT32_C(  1704686414),  INT32_C(  2147237893) },
      { -INT32_C(   914779970),  INT32_C(  1959859989), -INT32_C(   553754473), -INT32_C(  1014271650),  INT32_C(   356412875),  INT32_C(   609257705), -INT32_C(   641009768),  INT32_C(   498547549),
        -INT32_C(   521850678), -INT32_C(   548031162),  INT32_C(  1924944660), -INT32_C(  1892268093), -INT32_C(  1599769419), -INT32_C(  1815939845),  INT32_C(  1231916780), -INT32_C(  1788468280) } },
    { { -INT32_C(  1346942502),  INT32_C(  1943047743), -INT32_C(   669321264), -INT32_C(    41683446),  INT32_C(   622277516), -INT32_C(  1849584929),  INT32_C(   606872862),  INT32_C(  1084434534),
        -INT32_C(  1309648270), -INT32_C(  1205485336), -INT32_C(  1030668361), -INT32_C(  1044442059),  INT32_C(   652662343), -INT32_C(  2017941400),  INT32_C(   866903245),  INT32_C(  2121551372) },
      { -INT32_C(  1875876696), -INT32_C(   616016604), -INT32_C(   912402028),  INT32_C(   881482989), -INT32_C(  1688506062), -INT32_C(   433974503),  INT32_C(    52088311), -INT32_C(  1014854117),
         INT32_C(   374584050), -INT32_C(  1678664953),  INT32_C(  1650757493),  INT32_C(   513273579),  INT32_C(  2025452127), -INT32_C(    60826875), -INT32_C(  1006667352), -INT32_C(   108625657) },
      {  INT32_C(  1072148098),  INT32_C(  1327031139), -INT32_C(  1581723292),  INT32_C(   839799543), -INT32_C(  1066228546),  INT32_C(  2011407864),  INT32_C(   658961173),  INT32_C(    69580417),
        -INT32_C(   935064220),  INT32_C(  1410817007),  INT32_C(   620089132), -INT32_C(   531168480), -INT32_C(  1616852826), -INT32_C(  2078768275), -INT32_C(   139764107),  INT32_C(  2012925715) } },
    { {  INT32_C(   974117171), -INT32_C(   371916684),  INT32_C(  2068593039), -INT32_C(  2019957976), -INT32_C(   637513003), -INT32_C(   707371219), -INT32_C(   543631912), -INT32_C(  1965547945),
         INT32_C(  1808132087),  INT32_C(  2002098919), -INT32_C(    51207724),  INT32_C(  1501793156),  INT32_C(   171148253), -INT32_C(  1159788062),  INT32_C(   899250142), -INT32_C(  1933545067) },
      {  INT32_C(  1089963352), -INT32_C(   206091233),  INT32_C(  1911532013),  INT32_C(   298480436), -INT32_C(   652476938), -INT32_C(   443287034),  INT32_C(   102378865), -INT32_C(   141370722),
         INT32_C(  2134346079), -INT32_C(  1015877930), -INT32_C(   885693801), -INT32_C(   874709035),  INT32_C(    61143037),  INT32_C(  1659386097),  INT32_C(    57148261),  INT32_C(  1039858397) },
      {  INT32_C(  2064080523), -INT32_C(   578007917), -INT32_C(   314842244), -INT32_C(  1721477540), -INT32_C(  1289989941), -INT32_C(  1150658253), -INT32_C(   441253047), -INT32_C(  2106918667),
        -INT32_C(   352489130),  INT32_C(   986220989), -INT32_C(   936901525),  INT32_C(   627084121),  INT32_C(   232291290),  INT32_C(   499598035),  INT32_C(   956398403), -INT32_C(   893686670) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_add_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_add_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_add_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1958532398), -INT32_C(   733777631), -INT32_C(    61793464),  INT32_C(  1915617450),  INT32_C(   759754662),  INT32_C(   196190852), -INT32_C(    77082310), -INT32_C(   259049954),
        -INT32_C(  2105748895), -INT32_C(  1940471997), -INT32_C(  2071418662),  INT32_C(  1324791464),  INT32_C(   695943077),  INT32_C(   456404449),  INT32_C(   471309310),  INT32_C(   856467154) },
      UINT16_C(34936),
      {  INT32_C(  1554501556),  INT32_C(   715824517), -INT32_C(  1118624036),  INT32_C(  1373210451), -INT32_C(   177344477),  INT32_C(   237533590),  INT32_C(  1254743298), -INT32_C(  1653861911),
         INT32_C(  1727599584), -INT32_C(  1919900495),  INT32_C(   491513034), -INT32_C(  1905384341), -INT32_C(  1434199276),  INT32_C(  1454943060),  INT32_C(  1923121545),  INT32_C(  1930431890) },
      { -INT32_C(   774305504),  INT32_C(  2002741677), -INT32_C(  1181439411), -INT32_C(   498662706),  INT32_C(   848088029), -INT32_C(     7846794),  INT32_C(  1483876805),  INT32_C(  1456177718),
         INT32_C(   958964875),  INT32_C(  1538295565), -INT32_C(    15448783),  INT32_C(   635525959), -INT32_C(  1655214810),  INT32_C(  1989992369),  INT32_C(  1053691400),  INT32_C(   479566224) },
      { -INT32_C(  1958532398), -INT32_C(   733777631), -INT32_C(    61793464),  INT32_C(   874547745),  INT32_C(   670743552),  INT32_C(   229686796), -INT32_C(  1556347193), -INT32_C(   259049954),
        -INT32_C(  2105748895), -INT32_C(  1940471997), -INT32_C(  2071418662), -INT32_C(  1269858382),  INT32_C(   695943077),  INT32_C(   456404449),  INT32_C(   471309310), -INT32_C(  1884969182) } },
    { {  INT32_C(  1263910205),  INT32_C(  1973814596), -INT32_C(  1837843894),  INT32_C(  1018647829),  INT32_C(  1943604930), -INT32_C(   152472083),  INT32_C(   338999428), -INT32_C(  1892628143),
        -INT32_C(   891648634), -INT32_C(   717258613),  INT32_C(  1332196154), -INT32_C(   896852472),  INT32_C(   440230956),  INT32_C(  1578117082),  INT32_C(   812795103), -INT32_C(  1799380210) },
      UINT16_C(39208),
      {  INT32_C(  1520693330),  INT32_C(   992292367),  INT32_C(  1834312339),  INT32_C(  1758160265), -INT32_C(  1197916758),  INT32_C(   155998432),  INT32_C(   196914162), -INT32_C(  1671477942),
         INT32_C(  1173750326), -INT32_C(  1015014608),  INT32_C(   120641150), -INT32_C(   445580485),  INT32_C(   429721913),  INT32_C(  1394797153), -INT32_C(   547364971),  INT32_C(  1518059044) },
      { -INT32_C(   257985856),  INT32_C(   196354189),  INT32_C(   823387382),  INT32_C(   420971488), -INT32_C(   315444084),  INT32_C(   876696990),  INT32_C(  1477681204),  INT32_C(   515084126),
        -INT32_C(  1911664127),  INT32_C(  1754972786), -INT32_C(  2019906137), -INT32_C(  1130319568),  INT32_C(    78238309),  INT32_C(  1530456615), -INT32_C(   390837366), -INT32_C(   620337190) },
      {  INT32_C(  1263910205),  INT32_C(  1973814596), -INT32_C(  1837843894), -INT32_C(  2115835543),  INT32_C(  1943604930),  INT32_C(  1032695422),  INT32_C(   338999428), -INT32_C(  1892628143),
        -INT32_C(   737913801), -INT32_C(   717258613),  INT32_C(  1332196154), -INT32_C(  1575900053),  INT32_C(   507960222),  INT32_C(  1578117082),  INT32_C(   812795103),  INT32_C(   897721854) } },
    { {  INT32_C(   745149881),  INT32_C(  2123629783), -INT32_C(   519754063),  INT32_C(  1167959519), -INT32_C(  1622587784), -INT32_C(  1141145295), -INT32_C(  1482379316), -INT32_C(   813520362),
        -INT32_C(  1745097537), -INT32_C(  1592422160), -INT32_C(  1635640386),  INT32_C(   954408896), -INT32_C(  1747440538),  INT32_C(  2035471277), -INT32_C(  1742670206),  INT32_C(  1617404833) },
      UINT16_C(25487),
      { -INT32_C(   397433816),  INT32_C(   690041539),  INT32_C(   197196126),  INT32_C(  1317344204), -INT32_C(  1427725047), -INT32_C(   670347960), -INT32_C(  1554513232),  INT32_C(   928348431),
        -INT32_C(   517954531),  INT32_C(   889864663),  INT32_C(   104975162), -INT32_C(   413874466), -INT32_C(  1265485205),  INT32_C(   948739463), -INT32_C(  1344543585),  INT32_C(  2078683229) },
      { -INT32_C(  1403255083), -INT32_C(  2115934649),  INT32_C(   260514353),  INT32_C(  1425529832), -INT32_C(  1660385003), -INT32_C(   975858650), -INT32_C(   713772936), -INT32_C(  1236247583),
        -INT32_C(  1453151135),  INT32_C(  1143620371),  INT32_C(  1314173542), -INT32_C(  1549644915), -INT32_C(    96425260), -INT32_C(  1228991170), -INT32_C(  1500760891), -INT32_C(   262349681) },
      { -INT32_C(  1800688899), -INT32_C(  1425893110),  INT32_C(   457710479), -INT32_C(  1552093260), -INT32_C(  1622587784), -INT32_C(  1141145295), -INT32_C(  1482379316), -INT32_C(   307899152),
        -INT32_C(  1971105666),  INT32_C(  2033485034), -INT32_C(  1635640386),  INT32_C(   954408896), -INT32_C(  1747440538), -INT32_C(   280251707),  INT32_C(  1449662820),  INT32_C(  1617404833) } },
    { { -INT32_C(  1667645815),  INT32_C(  1759560706),  INT32_C(    62272630),  INT32_C(  1403410815),  INT32_C(  1112401411), -INT32_C(  1040708101), -INT32_C(   798522303), -INT32_C(   356465567),
        -INT32_C(  2071569790), -INT32_C(  1796446690),  INT32_C(   446145435), -INT32_C(     9552132),  INT32_C(   541178660),  INT32_C(   165755592),  INT32_C(   534333630), -INT32_C(  1895196148) },
      UINT16_C(36852),
      { -INT32_C(  1616167517), -INT32_C(  1600251525), -INT32_C(  1648303915), -INT32_C(   660102886), -INT32_C(   151486231),  INT32_C(   243597594), -INT32_C(  2027906927),  INT32_C(   991479448),
         INT32_C(  2145043204), -INT32_C(  1306560035),  INT32_C(  1934614361),  INT32_C(  1783363200), -INT32_C(  1855962249), -INT32_C(   694098619),  INT32_C(   375242877), -INT32_C(  1957595769) },
      {  INT32_C(   352988216), -INT32_C(    20501851), -INT32_C(  1972300023), -INT32_C(  2064335859),  INT32_C(  1159091200), -INT32_C(  1239697863), -INT32_C(    36931466),  INT32_C(   629677805),
        -INT32_C(   281308342), -INT32_C(   957545795),  INT32_C(   659578393),  INT32_C(   447431706), -INT32_C(   782253672), -INT32_C(   293045641), -INT32_C(   538225422), -INT32_C(  1140493198) },
      { -INT32_C(  1667645815),  INT32_C(  1759560706),  INT32_C(   674363358),  INT32_C(  1403410815),  INT32_C(  1007604969), -INT32_C(   996100269), -INT32_C(  2064838393),  INT32_C(  1621157253),
         INT32_C(  1863734862),  INT32_C(  2030861466), -INT32_C(  1700774542), -INT32_C(  2064172390),  INT32_C(   541178660),  INT32_C(   165755592),  INT32_C(   534333630),  INT32_C(  1196878329) } },
    { { -INT32_C(   995409913),  INT32_C(  1552586818),  INT32_C(   293854198), -INT32_C(  1205129697),  INT32_C(  1737067504), -INT32_C(   128642811), -INT32_C(   656981658), -INT32_C(  1131029323),
         INT32_C(  1602240540), -INT32_C(   809825575),  INT32_C(    98582245),  INT32_C(  1555893356), -INT32_C(  1664858473), -INT32_C(  1097590440),  INT32_C(   261516378),  INT32_C(  1707813704) },
      UINT16_C(19308),
      {  INT32_C(   692123069), -INT32_C(  1735983871), -INT32_C(  1674294716), -INT32_C(  1101346461),  INT32_C(  2110648373), -INT32_C(  1998415588),  INT32_C(   986556132), -INT32_C(   495525595),
        -INT32_C(   687032618), -INT32_C(   126905676),  INT32_C(  1066706140), -INT32_C(  1560416659), -INT32_C(    98579490),  INT32_C(  1216479844), -INT32_C(   830255192),  INT32_C(   129038641) },
      {  INT32_C(  1675607215),  INT32_C(   710626894),  INT32_C(  1600843762), -INT32_C(  1140758563), -INT32_C(  1766448846), -INT32_C(   874563293), -INT32_C(  1181130104),  INT32_C(   180439643),
         INT32_C(  1433313286), -INT32_C(   511718930), -INT32_C(  1774130759), -INT32_C(  2091761071), -INT32_C(  2045114013), -INT32_C(   900597438), -INT32_C(  1232802981),  INT32_C(  1002456373) },
      { -INT32_C(   995409913),  INT32_C(  1552586818), -INT32_C(    73450954),  INT32_C(  2052862272),  INT32_C(  1737067504),  INT32_C(  1421988415), -INT32_C(   194573972), -INT32_C(  1131029323),
         INT32_C(   746280668), -INT32_C(   638624606),  INT32_C(    98582245),  INT32_C(   642789566), -INT32_C(  1664858473), -INT32_C(  1097590440), -INT32_C(  2063058173),  INT32_C(  1707813704) } },
    { { -INT32_C(   745525531), -INT32_C(  1313599240),  INT32_C(  1246230009), -INT32_C(  1697736137), -INT32_C(   450828125),  INT32_C(  1018130913), -INT32_C(  1846398116),  INT32_C(  1573761656),
        -INT32_C(   651076127),  INT32_C(  1737155949),  INT32_C(   296866266),  INT32_C(   246120299),  INT32_C(  1223936871), -INT32_C(  1719360707),  INT32_C(  1328248534),  INT32_C(   179107881) },
      UINT16_C(56661),
      { -INT32_C(  1431315650), -INT32_C(  1028105637),  INT32_C(  1661709350),  INT32_C(   637308751),  INT32_C(   796141318),  INT32_C(  1966678303), -INT32_C(  1053287170), -INT32_C(   950050167),
        -INT32_C(  1737421251), -INT32_C(  1906627992),  INT32_C(   636577494), -INT32_C(    78975243),  INT32_C(   891993877), -INT32_C(   559258656),  INT32_C(   144761471), -INT32_C(  2117009596) },
      {  INT32_C(  1964654861),  INT32_C(  1090811243), -INT32_C(   798558757), -INT32_C(   104025629),  INT32_C(  1345255024), -INT32_C(   651241382), -INT32_C(    18690374), -INT32_C(   629165363),
         INT32_C(  1599117811), -INT32_C(   375368690),  INT32_C(   767166281),  INT32_C(   673613496), -INT32_C(   696757124), -INT32_C(   424630740),  INT32_C(  1122275957),  INT32_C(   924672836) },
      {  INT32_C(   533339211), -INT32_C(  1313599240),  INT32_C(   863150593), -INT32_C(  1697736137),  INT32_C(  2141396342),  INT32_C(  1018130913), -INT32_C(  1071977544),  INT32_C(  1573761656),
        -INT32_C(   138303440),  INT32_C(  1737155949),  INT32_C(  1403743775),  INT32_C(   594638253),  INT32_C(   195236753), -INT32_C(  1719360707),  INT32_C(  1267037428), -INT32_C(  1192336760) } },
    { {  INT32_C(   194407933),  INT32_C(   183842753), -INT32_C(   164122818), -INT32_C(  1323410123), -INT32_C(   578251087), -INT32_C(  1312606148),  INT32_C(   250914762),  INT32_C(   138744075),
         INT32_C(  1058266238),  INT32_C(  1363740691), -INT32_C(   330858057), -INT32_C(  1868667426),  INT32_C(   929900283),  INT32_C(   686371166), -INT32_C(   482943528),  INT32_C(  1827372014) },
      UINT16_C(65367),
      {  INT32_C(  1420493429),  INT32_C(  1659128167), -INT32_C(   845524625),  INT32_C(  1542816642),  INT32_C(  1312697184), -INT32_C(    21353817),  INT32_C(   812213545), -INT32_C(   806411175),
        -INT32_C(  1910269145),  INT32_C(  1425082340), -INT32_C(   618558632),  INT32_C(  1849038606), -INT32_C(   373525438), -INT32_C(   941066594),  INT32_C(   888689115), -INT32_C(  1677465739) },
      {  INT32_C(  1730881154), -INT32_C(  2034557907), -INT32_C(  1251877721),  INT32_C(   908302323), -INT32_C(  1440751861),  INT32_C(   812713813), -INT32_C(   832280232), -INT32_C(   748001199),
        -INT32_C(  1137011314),  INT32_C(  1480783281), -INT32_C(   988961838),  INT32_C(  1174089786),  INT32_C(  1693391631),  INT32_C(  2073321762),  INT32_C(   457832906), -INT32_C(   269503647) },
      { -INT32_C(  1143592713), -INT32_C(   375429740), -INT32_C(  2097402346), -INT32_C(  1323410123), -INT32_C(   128054677), -INT32_C(  1312606148), -INT32_C(    20066687),  INT32_C(   138744075),
         INT32_C(  1247686837), -INT32_C(  1389101675), -INT32_C(  1607520470), -INT32_C(  1271838904),  INT32_C(  1319866193),  INT32_C(  1132255168),  INT32_C(  1346522021), -INT32_C(  1946969386) } },
    { { -INT32_C(    89446071), -INT32_C(   246158049), -INT32_C(   894017392), -INT32_C(  1609518447), -INT32_C(   284819507),  INT32_C(   728406368), -INT32_C(   213470318), -INT32_C(  1327286937),
         INT32_C(  2125106783),  INT32_C(   208665980), -INT32_C(   271112866), -INT32_C(  1534072873),  INT32_C(  1200919782), -INT32_C(  1066205650),  INT32_C(   431274162),  INT32_C(  1305057262) },
      UINT16_C(29477),
      {  INT32_C(   935232863),  INT32_C(  1390103916),  INT32_C(   278491106),  INT32_C(   550505326), -INT32_C(  1304853308),  INT32_C(  1107231259), -INT32_C(   421344651),  INT32_C(  1672843268),
        -INT32_C(  2120584427), -INT32_C(  1546357055),  INT32_C(  1404268005),  INT32_C(  1030980473),  INT32_C(   602909704),  INT32_C(   610594478), -INT32_C(  1140176968), -INT32_C(   316686121) },
      { -INT32_C(   194069965),  INT32_C(   362234416),  INT32_C(   694766256), -INT32_C(   697901874),  INT32_C(   939087241), -INT32_C(    77898173),  INT32_C(  2092394149),  INT32_C(  1500108326),
        -INT32_C(  1068574576), -INT32_C(   891886310), -INT32_C(    17613008), -INT32_C(  1529587429), -INT32_C(   237187666), -INT32_C(   789825749), -INT32_C(  1018322019),  INT32_C(   169719418) },
      {  INT32_C(   741162898), -INT32_C(   246158049),  INT32_C(   973257362), -INT32_C(  1609518447), -INT32_C(   284819507),  INT32_C(  1029333086), -INT32_C(   213470318), -INT32_C(  1327286937),
         INT32_C(  1105808293),  INT32_C(  1856723931), -INT32_C(   271112866), -INT32_C(  1534072873),  INT32_C(   365722038), -INT32_C(   179231271),  INT32_C(  2136468309),  INT32_C(  1305057262) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_add_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_add_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_add_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(52979),
      {  INT32_C(  1318030952), -INT32_C(   938502652), -INT32_C(  1205717630),  INT32_C(  1648076236),  INT32_C(  1874746093),  INT32_C(   507402795), -INT32_C(   271937240), -INT32_C(   581761675),
        -INT32_C(  1758731373), -INT32_C(    77579399),  INT32_C(  1018397296),  INT32_C(   345959975),  INT32_C(  1954766153), -INT32_C(   527253065), -INT32_C(   925934509), -INT32_C(   190504095) },
      {  INT32_C(  2139869190),  INT32_C(  2071653131), -INT32_C(  1799934611),  INT32_C(  1688819227),  INT32_C(  1792552115),  INT32_C(  1095396078),  INT32_C(   654908102),  INT32_C(  1125887549),
        -INT32_C(  1966954626),  INT32_C(      343186), -INT32_C(  2070626967), -INT32_C(   957793005),  INT32_C(  1479590250), -INT32_C(   224822484), -INT32_C(   770006379), -INT32_C(   837470896) },
      { -INT32_C(   837067154),  INT32_C(  1133150479),  INT32_C(           0),  INT32_C(           0), -INT32_C(   627669088),  INT32_C(  1602798873),  INT32_C(   382970862),  INT32_C(   544125874),
         INT32_C(           0), -INT32_C(    77236213), -INT32_C(  1052229671), -INT32_C(   611833030),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1695940888), -INT32_C(  1027974991) } },
    { UINT16_C(55260),
      {  INT32_C(   771752731), -INT32_C(  1410012863), -INT32_C(   687659861),  INT32_C(   885628063), -INT32_C(  1912151234), -INT32_C(   178513127),  INT32_C(    90486258),  INT32_C(   780260115),
         INT32_C(   408715991),  INT32_C(   381898859),  INT32_C(   351127156), -INT32_C(  1605847198), -INT32_C(  1288810598),  INT32_C(  1571392106),  INT32_C(  1382157631), -INT32_C(  1199512351) },
      { -INT32_C(   774841242), -INT32_C(  1578593492), -INT32_C(  1145711271),  INT32_C(   660340108), -INT32_C(  1210414772),  INT32_C(  1393853203),  INT32_C(  1923446417), -INT32_C(  1070979494),
         INT32_C(   798161410), -INT32_C(   422544755), -INT32_C(   593394353), -INT32_C(   821822334), -INT32_C(  1735991931), -INT32_C(   219440543),  INT32_C(  1801752848), -INT32_C(  1188327753) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1833371132),  INT32_C(  1545968171),  INT32_C(  1172401290),  INT32_C(           0),  INT32_C(  2013932675), -INT32_C(   290719379),
         INT32_C(  1206877401), -INT32_C(    40645896), -INT32_C(   242267197),  INT32_C(           0),  INT32_C(  1270164767),  INT32_C(           0), -INT32_C(  1111056817),  INT32_C(  1907127192) } },
    { UINT16_C(48520),
      { -INT32_C(  1067213763),  INT32_C(   495937176), -INT32_C(  1531636413), -INT32_C(  1080647249), -INT32_C(   383059406),  INT32_C(   279074440),  INT32_C(  1260751635), -INT32_C(  2116935613),
         INT32_C(  1413559740), -INT32_C(   562966373), -INT32_C(  1803343899), -INT32_C(    95217208), -INT32_C(  1662812652), -INT32_C(   408058412),  INT32_C(  1412616720), -INT32_C(  1344994061) },
      { -INT32_C(   737929671), -INT32_C(   877431322),  INT32_C(  1683961500),  INT32_C(  1667150415),  INT32_C(    67125552), -INT32_C(   672354873),  INT32_C(  1915428479),  INT32_C(  1545732131),
        -INT32_C(    63887850),  INT32_C(   952624283),  INT32_C(  1771841050),  INT32_C(   164494297),  INT32_C(    51301692), -INT32_C(   103024006),  INT32_C(   996935192),  INT32_C(   496537095) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   586503166),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   571203482),
         INT32_C(  1349671890),  INT32_C(           0), -INT32_C(    31502849),  INT32_C(    69277089), -INT32_C(  1611510960), -INT32_C(   511082418),  INT32_C(           0), -INT32_C(   848456966) } },
    { UINT16_C(51636),
      { -INT32_C(   516938744),  INT32_C(  1542126879),  INT32_C(  1147140298),  INT32_C(   188627698),  INT32_C(  1195813440), -INT32_C(   328868296),  INT32_C(  1413185447),  INT32_C(  1746649952),
         INT32_C(   105467111),  INT32_C(   341914697),  INT32_C(   525910060),  INT32_C(   992646906),  INT32_C(  2021814336), -INT32_C(   161159345), -INT32_C(   951345050),  INT32_C(  1244620387) },
      { -INT32_C(    61834830), -INT32_C(   653217363),  INT32_C(  1828218994),  INT32_C(  1067918079), -INT32_C(   491246957),  INT32_C(  2027428881),  INT32_C(   524231612),  INT32_C(  1013542538),
        -INT32_C(  1808221721), -INT32_C(   579975061), -INT32_C(  1337366863),  INT32_C(   485486985),  INT32_C(   754886427), -INT32_C(  2136680764), -INT32_C(  2069830662),  INT32_C(   968886610) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1319608004),  INT32_C(           0),  INT32_C(   704566483),  INT32_C(  1698560585),  INT32_C(           0), -INT32_C(  1534774806),
        -INT32_C(  1702754610),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1478133891),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1273791584), -INT32_C(  2081460299) } },
    { UINT16_C(63939),
      {  INT32_C(   732058786),  INT32_C(  1615303237), -INT32_C(    41073351),  INT32_C(   377368860),  INT32_C(  1738153493), -INT32_C(   358589913), -INT32_C(  1793561005),  INT32_C(  1300702122),
        -INT32_C(  1116198280), -INT32_C(   182533956),  INT32_C(   569617157), -INT32_C(   248024612), -INT32_C(  1235693169),  INT32_C(  2141321516),  INT32_C(   303348071),  INT32_C(  1432329437) },
      { -INT32_C(  2112694330), -INT32_C(  1653133161), -INT32_C(  1195480357),  INT32_C(  1789523675), -INT32_C(   215940409),  INT32_C(  1651753723),  INT32_C(  1484031867), -INT32_C(   374484189),
         INT32_C(  1114357931), -INT32_C(   857742352),  INT32_C(   696557133),  INT32_C(  1536372116),  INT32_C(   709866543), -INT32_C(   225590666),  INT32_C(  1833566537), -INT32_C(  2141783851) },
      { -INT32_C(  1380635544), -INT32_C(    37829924),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   309529138),  INT32_C(   926217933),
        -INT32_C(     1840349),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1288347504), -INT32_C(   525826626),  INT32_C(  1915730850),  INT32_C(  2136914608), -INT32_C(   709454414) } },
    { UINT16_C(49848),
      { -INT32_C(   736364480),  INT32_C(  1429188390), -INT32_C(   629113245), -INT32_C(  1966338752), -INT32_C(   470346226), -INT32_C(   966570738), -INT32_C(  1267784177),  INT32_C(   145220552),
         INT32_C(    48022236), -INT32_C(   715715727), -INT32_C(   894445686), -INT32_C(   212567068), -INT32_C(  1596568687), -INT32_C(  1469695335),  INT32_C(   677238112), -INT32_C(  1792015175) },
      {  INT32_C(   949423302), -INT32_C(  1592922601), -INT32_C(  1435714362), -INT32_C(   929185737), -INT32_C(  1519881204),  INT32_C(   239980462),  INT32_C(  1563863716),  INT32_C(  1978820270),
        -INT32_C(  1985115790),  INT32_C(  1043053176), -INT32_C(  1377265802),  INT32_C(  1668646487), -INT32_C(  1475813638),  INT32_C(  1370904237), -INT32_C(  1347425280), -INT32_C(  1004232366) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1399442807), -INT32_C(  1990227430), -INT32_C(   726590276),  INT32_C(           0),  INT32_C(  2124040822),
         INT32_C(           0),  INT32_C(   327337449),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   670187168),  INT32_C(  1498719755) } },
    { UINT16_C(53547),
      {  INT32_C(  1739507983),  INT32_C(  1237984079), -INT32_C(  1359883519),  INT32_C(   687908649), -INT32_C(   438784109),  INT32_C(  2074737744),  INT32_C(  1478424525),  INT32_C(  2136604527),
         INT32_C(   417728457),  INT32_C(   744665131), -INT32_C(  1394912381), -INT32_C(  1898521605), -INT32_C(   629887350),  INT32_C(  2018909611), -INT32_C(  2066648044),  INT32_C(  1023617652) },
      {  INT32_C(  1565911346),  INT32_C(   495564697),  INT32_C(   113861643), -INT32_C(   913006785), -INT32_C(   106690482),  INT32_C(   980548134), -INT32_C(   490847634), -INT32_C(  1625308819),
         INT32_C(  1157395882),  INT32_C(   929137964),  INT32_C(   691874538),  INT32_C(   418632394),  INT32_C(     1152986), -INT32_C(    12877167),  INT32_C(   853735877),  INT32_C(  1708196283) },
      { -INT32_C(   989547967),  INT32_C(  1733548776),  INT32_C(           0), -INT32_C(   225098136),  INT32_C(           0), -INT32_C(  1239681418),  INT32_C(           0),  INT32_C(           0),
         INT32_C(  1575124339),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   628734364),  INT32_C(           0), -INT32_C(  1212912167), -INT32_C(  1563153361) } },
    { UINT16_C(52598),
      { -INT32_C(    10086347), -INT32_C(  1005102614), -INT32_C(  2117785360),  INT32_C(  1870659754), -INT32_C(  1264491783), -INT32_C(   635800988), -INT32_C(  1837251777),  INT32_C(    63854798),
         INT32_C(  1510093936),  INT32_C(  2099124621), -INT32_C(   335617215), -INT32_C(   581206045),  INT32_C(  1167195361), -INT32_C(  1373590673),  INT32_C(  1027644783),  INT32_C(  1698697205) },
      { -INT32_C(   775994813), -INT32_C(  1672552869), -INT32_C(  1517859391), -INT32_C(  1383931188),  INT32_C(  1324553183),  INT32_C(   788272063), -INT32_C(  1502921296), -INT32_C(  1895060660),
         INT32_C(  1214303213),  INT32_C(  1793372073), -INT32_C(   938513412),  INT32_C(   762679630),  INT32_C(  1685809317),  INT32_C(   747796347),  INT32_C(    13827508), -INT32_C(  1785668184) },
      {  INT32_C(           0),  INT32_C(  1617311813),  INT32_C(   659322545),  INT32_C(           0),  INT32_C(    60061400),  INT32_C(   152471075),  INT32_C(   954794223),  INT32_C(           0),
        -INT32_C(  1570570147),  INT32_C(           0), -INT32_C(  1274130627),  INT32_C(   181473585),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1041472291), -INT32_C(    86970979) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_add_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_add_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_add_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 5896694048212443049),  INT64_C( 9000266092577364175),  INT64_C( 7152103947590144860),  INT64_C( 2989496802120002433),
         INT64_C( 7710631622698424498), -INT64_C( 3633641352504339518),  INT64_C( 4274662313843579209),  INT64_C( 6826149357438294289) },
      { -INT64_C(  164363539517042510), -INT64_C( 8691483022440823005),  INT64_C( 9220214710722807807), -INT64_C( 2401321110146592095),
        -INT64_C( 1183069234730910884),  INT64_C( 5562197665816815723),  INT64_C( 1124204932795639468), -INT64_C( 3326459772972193332) },
      {  INT64_C( 5732330508695400539),  INT64_C(  308783070136541170), -INT64_C( 2074425415396598949),  INT64_C(  588175691973410338),
         INT64_C( 6527562387967513614),  INT64_C( 1928556313312476205),  INT64_C( 5398867246639218677),  INT64_C( 3499689584466100957) } },
    { {  INT64_C( 5873253456280027845), -INT64_C( 7547985786885765724),  INT64_C(  958785414761629392),  INT64_C( 8879370812030102515),
         INT64_C( 4762149369024389598),  INT64_C(  798391932315570322),  INT64_C(  147097132267652539), -INT64_C( 3786220049007964093) },
      { -INT64_C( 6906651495143342010), -INT64_C( 8287694440800363594),  INT64_C( 2452371479376447222),  INT64_C( 6726846324779217826),
         INT64_C( 5025569198072523023),  INT64_C(  225235015271111619), -INT64_C( 2487938372584494983),  INT64_C( 3745242421369017476) },
      { -INT64_C( 1033398038863314165),  INT64_C( 2611063846023422298),  INT64_C( 3411156894138076614), -INT64_C( 2840526936900231275),
        -INT64_C( 8659025506612638995),  INT64_C( 1023626947586681941), -INT64_C( 2340841240316842444), -INT64_C(   40977627638946617) } },
    { {  INT64_C( 7387045378804169392),  INT64_C( 7348215347083393770), -INT64_C( 7432287296260602942), -INT64_C( 4746462990122596405),
         INT64_C( 7669772552352133735),  INT64_C( 5083821277037292091),  INT64_C( 5407731889132030559), -INT64_C( 3084302269135830938) },
      { -INT64_C( 1461330460425637939), -INT64_C( 3061426891990558023),  INT64_C( 5599758734307477482),  INT64_C( 1862788523933954198),
        -INT64_C( 4557890179386853341),  INT64_C( 6666876053459161657), -INT64_C( 4937214972124475832),  INT64_C( 6111833508638834029) },
      {  INT64_C( 5925714918378531453),  INT64_C( 4286788455092835747), -INT64_C( 1832528561953125460), -INT64_C( 2883674466188642207),
         INT64_C( 3111882372965280394), -INT64_C( 6696046743213097868),  INT64_C(  470516917007554727),  INT64_C( 3027531239503003091) } },
    { { -INT64_C( 7991663547628636080), -INT64_C( 2555292973839346502),  INT64_C( 4212139769629200532),  INT64_C( 1966319092590916547),
         INT64_C( 1506042142180667901), -INT64_C( 9075093079022557283),  INT64_C( 7143746535270586651), -INT64_C( 2897889499141433630) },
      {  INT64_C( 6953298079720946194), -INT64_C( 6437157297342791622),  INT64_C( 8555627167819425208), -INT64_C( 4217080419303877945),
        -INT64_C(  916288211658955227),  INT64_C( 3576356706803505520), -INT64_C( 6218269451284303702), -INT64_C( 1742958193093650601) },
      { -INT64_C( 1038365467907689886), -INT64_C( 8992450271182138124), -INT64_C( 5678977136260925876), -INT64_C( 2250761326712961398),
         INT64_C(  589753930521712674), -INT64_C( 5498736372219051763),  INT64_C(  925477083986282949), -INT64_C( 4640847692235084231) } },
    { {  INT64_C( 2312342974665588586), -INT64_C( 6729576343545367823), -INT64_C( 4578026214523853331), -INT64_C( 1074221180203122067),
        -INT64_C( 1195656230424156519),  INT64_C( 3385005156404397150), -INT64_C( 2575086539621213671),  INT64_C( 4660983342689947190) },
      {  INT64_C( 5690001192450114569),  INT64_C( 6765706558176579445), -INT64_C( 8375529455621185160), -INT64_C(  280638300551000014),
         INT64_C(  791220201005032380), -INT64_C( 5337991249511014582),  INT64_C( 2478776332018633862),  INT64_C( 7142732816633802545) },
      {  INT64_C( 8002344167115703155),  INT64_C(   36130214631211622),  INT64_C( 5493188403564513125), -INT64_C( 1354859480754122081),
        -INT64_C(  404436029419124139), -INT64_C( 1952986093106617432), -INT64_C(   96310207602579809), -INT64_C( 6643027914385801881) } },
    { {  INT64_C( 7590546826509362360),  INT64_C( 4799960603843565481), -INT64_C( 3764863488869189202), -INT64_C( 8485326154395304909),
         INT64_C(  125025846558150196),  INT64_C( 4919203572335817541),  INT64_C( 1811753159855661758),  INT64_C( 6393760326532469855) },
      {  INT64_C(  421764692607537793),  INT64_C(  501672283606598428), -INT64_C( 2545232539499374162),  INT64_C( 6202803407104615064),
         INT64_C( 4980991260009414746),  INT64_C( 2385761506151573452), -INT64_C( 7628987825040033081),  INT64_C( 5850290225876708869) },
      {  INT64_C( 8012311519116900153),  INT64_C( 5301632887450163909), -INT64_C( 6310096028368563364), -INT64_C( 2282522747290689845),
         INT64_C( 5106017106567564942),  INT64_C( 7304965078487390993), -INT64_C( 5817234665184371323), -INT64_C( 6202693521300372892) } },
    { {  INT64_C( 3861145535682141991),  INT64_C( 4704120286579625139),  INT64_C( 7310649930581147103),  INT64_C( 6132617560052451027),
        -INT64_C( 4220933801323952434), -INT64_C(  467755223424977465), -INT64_C( 9153765608270723279),  INT64_C( 8400169494660134417) },
      {  INT64_C( 4306311459952605676),  INT64_C( 1432426031515283149),  INT64_C( 1311843823099622919), -INT64_C( 3392084749394608174),
         INT64_C( 8992722739203577885),  INT64_C( 5779599678188505408),  INT64_C( 5119810430763850234),  INT64_C( 6804001435340987831) },
      {  INT64_C( 8167456995634747667),  INT64_C( 6136546318094908288),  INT64_C( 8622493753680770022),  INT64_C( 2740532810657842853),
         INT64_C( 4771788937879625451),  INT64_C( 5311844454763527943), -INT64_C( 4033955177506873045), -INT64_C( 3242573143708429368) } },
    { {  INT64_C(  896142439321910083),  INT64_C( 1197503498379252485), -INT64_C( 7856220743107108291), -INT64_C( 6406762567310591882),
        -INT64_C( 4058014011976186410),  INT64_C( 9080299469053222364), -INT64_C( 7078487466013880490),  INT64_C( 7199966683762914017) },
      {  INT64_C( 7780449457883481456),  INT64_C( 1824347912971095698), -INT64_C( 8415522727832944271),  INT64_C( 7418198203865008897),
         INT64_C( 4556395623730444353),  INT64_C( 7889010207409543840), -INT64_C( 3090529460147599642), -INT64_C( 1719435354305139514) },
      {  INT64_C( 8676591897205391539),  INT64_C( 3021851411350348183),  INT64_C( 2175000602769499054),  INT64_C( 1011435636554417015),
         INT64_C(  498381611754257943), -INT64_C( 1477434397246785412),  INT64_C( 8277727147548071484),  INT64_C( 5480531329457774503) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_add_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_add_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_add_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[16];
    const easysimd__mmask8 k;
    const int64_t a[16];
    const int64_t b[16];
    const int64_t r[16];
  } test_vec[] = {
    { {  INT64_C( 1637095571448452370), -INT64_C( 5010656489973054228),  INT64_C( 8096577095910365922),  INT64_C( 3926524403043278656),
         INT64_C( 1692383872749537703), -INT64_C( 7546382928320257262), -INT64_C( 6602394267959061769),  INT64_C( 2598636899144412341) },
      UINT8_C( 63),
      {  INT64_C( 3064278465490095078), -INT64_C( 1676615221711466009), -INT64_C( 5333405411065419087), -INT64_C( 3810230114928306775),
        -INT64_C( 1075017760364328478),  INT64_C( 2095704811519734998), -INT64_C(  299103093840977638),  INT64_C( 5108483185182444596) },
      { -INT64_C( 4648200900296301693), -INT64_C( 4629136759825157063), -INT64_C( 5066623773317061022),  INT64_C(  536973459407932105),
         INT64_C( 2486008889004565721), -INT64_C( 3554876755438703545),  INT64_C( 1436140625484484016),  INT64_C( 5713013447801749692) },
      { -INT64_C( 1583922434806206615), -INT64_C( 6305751981536623072),  INT64_C( 8046714889327071507), -INT64_C( 3273256655520374670),
         INT64_C( 1410991128640237243), -INT64_C( 1459171943918968547), -INT64_C( 6602394267959061769),  INT64_C( 2598636899144412341) } },
    { {  INT64_C( 3991194155833482583),  INT64_C( 7365863369617845245), -INT64_C( 6217348007288128678),  INT64_C(  148675600489051978),
        -INT64_C( 5748943111581392624), -INT64_C( 9084794923389396527),  INT64_C( 6675703621262608398), -INT64_C( 3278006165881122860) },
      UINT8_C( 79),
      {  INT64_C( 7078729567351001797),  INT64_C( 3976127268296180429),  INT64_C( 6764870419675162927), -INT64_C( 6394845513855835965),
        -INT64_C( 4382478565492243517), -INT64_C( 1011318967947184367), -INT64_C( 6311831277423214532),  INT64_C( 4236157129718335039) },
      {  INT64_C( 4311691048566315805),  INT64_C(  693258357862808300), -INT64_C( 5061911316372677582),  INT64_C( 5814609134873172224),
        -INT64_C( 2890120277031405697),  INT64_C(  273070111211249652),  INT64_C( 6337650268323962303), -INT64_C(  956874791454847436) },
      { -INT64_C( 7056323457792234014),  INT64_C( 4669385626158988729),  INT64_C( 1702959103302485345), -INT64_C(  580236378982663741),
        -INT64_C( 5748943111581392624), -INT64_C( 9084794923389396527),  INT64_C(   25818990900747771), -INT64_C( 3278006165881122860) } },
    { {  INT64_C( 5402490335443754038),  INT64_C( 7004459312563912287),  INT64_C( 6873494867043635124), -INT64_C( 1746693303777676963),
         INT64_C( 4412405986682822043), -INT64_C( 3277034903515019135),  INT64_C(  971442364987875570),  INT64_C( 6704628126445290098) },
      UINT8_C(252),
      { -INT64_C( 3199500174101950700),  INT64_C( 7068896874256776325), -INT64_C(  840732006067128670), -INT64_C( 7451465598208935429),
         INT64_C(  990872770473652578),  INT64_C( 1777037797882114565), -INT64_C( 3158904769779877244),  INT64_C( 6189642379913322441) },
      { -INT64_C( 2014230672746244489),  INT64_C( 1316129223197016245),  INT64_C( 6358081634684124815),  INT64_C( 2925524125942721361),
        -INT64_C( 7645647755206468574),  INT64_C( 8364597264550793588),  INT64_C( 2245635740289228099),  INT64_C( 9124008468664275140) },
      {  INT64_C( 5402490335443754038),  INT64_C( 7004459312563912287),  INT64_C( 5517349628616996145), -INT64_C( 4525941472266214068),
        -INT64_C( 6654774984732815996), -INT64_C( 8305109011276643463), -INT64_C(  913269029490649145), -INT64_C( 3133093225131954035) } },
    { {  INT64_C( 7876626396527707865),  INT64_C( 6327703798935457910), -INT64_C( 8444156093278868254),  INT64_C(  792525990600389412),
         INT64_C( 6542343655737491300), -INT64_C( 6733297332257473758),  INT64_C( 3495113324412254258), -INT64_C( 8894133035806391978) },
      UINT8_C( 48),
      { -INT64_C( 1618640895730195884),  INT64_C(  566130083197796387), -INT64_C( 3091365637900741985), -INT64_C( 8802714067975954187),
         INT64_C( 8931894081495034460), -INT64_C( 8463108217014804938), -INT64_C( 2811541516088205358), -INT64_C( 4054272745087766267) },
      {  INT64_C( 6129898402509662270),  INT64_C(  565315231888848484), -INT64_C( 8016080185148496634), -INT64_C( 3365171251436437734),
         INT64_C( 5232753838442094123), -INT64_C( 1806946338783921745),  INT64_C( 6678716485601335700),  INT64_C( 2537267084449117649) },
      {  INT64_C( 7876626396527707865),  INT64_C( 6327703798935457910), -INT64_C( 8444156093278868254),  INT64_C(  792525990600389412),
        -INT64_C( 4282096153772423033),  INT64_C( 8176689517910824933),  INT64_C( 3495113324412254258), -INT64_C( 8894133035806391978) } },
    { { -INT64_C( 4217327386109371060),  INT64_C( 1462146507223994500),  INT64_C( 9029403535350110895), -INT64_C( 6164557771088777128),
         INT64_C( 7967243682726010805), -INT64_C( 9152970505335981211),  INT64_C( 7521223655988276535),  INT64_C( 1078941248321503985) },
      UINT8_C( 10),
      { -INT64_C( 6444823229810484523), -INT64_C( 7166643799492954826),  INT64_C( 1160825679683284586),  INT64_C( 4107978185158323148),
        -INT64_C( 8042316938503522478),  INT64_C( 4355947116441623144),  INT64_C(  124837676903243996), -INT64_C( 1113239454258551314) },
      {  INT64_C( 5394206117329760241),  INT64_C(  790827237554372843), -INT64_C( 3320718750563147595), -INT64_C( 3521057494574767212),
        -INT64_C( 3689301451095683169),  INT64_C( 4102642388072787639), -INT64_C( 6298270799792855837),  INT64_C(  908597294068841711) },
      { -INT64_C( 4217327386109371060), -INT64_C( 6375816561938581983),  INT64_C( 9029403535350110895),  INT64_C(  586920690583555936),
         INT64_C( 7967243682726010805), -INT64_C( 9152970505335981211),  INT64_C( 7521223655988276535),  INT64_C( 1078941248321503985) } },
    { {  INT64_C( 7311693701301843659),  INT64_C( 7494898546895421768),  INT64_C( 2349409172957636062),  INT64_C( 4322479761028576388),
         INT64_C( 3265778120923777598), -INT64_C( 5310310381393437343), -INT64_C( 4003064257566966751),  INT64_C( 2693634056535957430) },
      UINT8_C( 63),
      {  INT64_C(   74681461099467337), -INT64_C( 5086377914583683253), -INT64_C( 8273458662043960522), -INT64_C(   39800438883330947),
         INT64_C( 3679636505814865579), -INT64_C( 5866531736128853600), -INT64_C( 3073049960134569313),  INT64_C( 1395686423709339305) },
      {  INT64_C( 6281452445510075920),  INT64_C( 3045217899379926812),  INT64_C( 7460303757460924507),  INT64_C( 1845390670211485473),
         INT64_C( 1096976101920587563), -INT64_C( 7954793774127551260),  INT64_C(  392601397348307534), -INT64_C( 8539621634010629797) },
      {  INT64_C( 6356133906609543257), -INT64_C( 2041160015203756441), -INT64_C(  813154904583036015),  INT64_C( 1805590231328154526),
         INT64_C( 4776612607735453142),  INT64_C( 4625418563453146756), -INT64_C( 4003064257566966751),  INT64_C( 2693634056535957430) } },
    { { -INT64_C( 5625659159720783894), -INT64_C( 4262733505137438704),  INT64_C( 4771074415986154316), -INT64_C( 2710563408861215365),
        -INT64_C( 9137340262048543309),  INT64_C( 6372485775011303733), -INT64_C(  224123893461729351),  INT64_C( 7083941961317845637) },
      UINT8_C(  4),
      {  INT64_C( 7269643312887620103),  INT64_C( 4329870181778099646),  INT64_C( 2564722579906344530),  INT64_C( 7190335853134220430),
        -INT64_C(  968852038973637098),  INT64_C( 1853343154121473663), -INT64_C( 4838903194234096357), -INT64_C(  824357888695620912) },
      {  INT64_C( 1274425862000582536),  INT64_C( 9189953907530268329),  INT64_C( 5306942928662607291), -INT64_C(  321439533223302985),
        -INT64_C(  340471119033620572), -INT64_C( 3077940849910492058),  INT64_C( 4642198055108443306), -INT64_C( 8432040435859988082) },
      { -INT64_C( 5625659159720783894), -INT64_C( 4262733505137438704),  INT64_C( 7871665508568951821), -INT64_C( 2710563408861215365),
        -INT64_C( 9137340262048543309),  INT64_C( 6372485775011303733), -INT64_C(  224123893461729351),  INT64_C( 7083941961317845637) } },
    { { -INT64_C( 9127382355256823033),  INT64_C( 6974267907656827098),  INT64_C( 9068262761557100815),  INT64_C( 1459580854064754385),
        -INT64_C( 2177275983803055828), -INT64_C( 5361079444635839613), -INT64_C( 2408539542357402585), -INT64_C( 5262782123028966956) },
      UINT8_C(216),
      { -INT64_C( 2761901989156618652), -INT64_C( 7396259151174703979),  INT64_C( 1620878075755917699), -INT64_C( 4915584061870677991),
        -INT64_C(  219395007845324972),  INT64_C( 3208968296463365233), -INT64_C( 3812486535545803012),  INT64_C( 7117239981973485491) },
      { -INT64_C( 6444525492333861076),  INT64_C( 5168757207706484966), -INT64_C( 7509645842022035381),  INT64_C( 3857445270331687960),
         INT64_C( 6839094782695310862), -INT64_C( 1825179838618698216),  INT64_C( 7833075129166066744), -INT64_C( 3860117335376243408) },
      { -INT64_C( 9127382355256823033),  INT64_C( 6974267907656827098),  INT64_C( 9068262761557100815), -INT64_C( 1058138791538990031),
         INT64_C( 6619699774849985890), -INT64_C( 5361079444635839613),  INT64_C( 4020588593620263732),  INT64_C( 3257122646597242083) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_add_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_add_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_add_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[16];
    const int64_t b[16];
    const int64_t r[16];
  } test_vec[] = {
    { UINT8_C(165),
      {  INT64_C( 6299320458837796671), -INT64_C( 3196421240547742572), -INT64_C( 9151855083952004989), -INT64_C( 2652966953870515301),
        -INT64_C( 3361856595458879637), -INT64_C( 8765515588673012554),  INT64_C( 4218943347121949634),  INT64_C( 8056360307695763285) },
      { -INT64_C( 6084423613766652800), -INT64_C(  986696027690857020),  INT64_C(  716507424025936408),  INT64_C( 2755580261000000714),
         INT64_C( 4185659851829194101), -INT64_C( 1359153785955268607),  INT64_C( 8651579458846990930),  INT64_C( 5266260289850313545) },
      {  INT64_C(  214896845071143871),  INT64_C(                   0), -INT64_C( 8435347659926068581),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C( 8322074699081270455),  INT64_C(                   0), -INT64_C( 5124123476163474786) } },
    { UINT8_C( 48),
      {  INT64_C( 1327964625155044601), -INT64_C( 1979941431104987422), -INT64_C( 6388061835839239302),  INT64_C( 6198577468949612625),
         INT64_C( 7878256497849969529), -INT64_C( 8253649976125538866),  INT64_C( 7274427282076993456), -INT64_C( 5985215513423679939) },
      { -INT64_C( 5628578266044451862),  INT64_C( 6350840359232373634), -INT64_C(  993721339898183746), -INT64_C( 7573227544723558906),
        -INT64_C( 7078269819051780816),  INT64_C( 8967324078724744818), -INT64_C( 5423879114017925356), -INT64_C(  850101963731351568) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(  799986678798188713),  INT64_C(  713674102599205952),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(169),
      { -INT64_C( 3269505804180125889),  INT64_C( 4898802782180457107), -INT64_C( 4122299440839867048), -INT64_C( 2805040416254433388),
        -INT64_C( 3881074597838727547),  INT64_C( 1644288571922952801), -INT64_C(   91554778652228748),  INT64_C( 5302276918373401890) },
      { -INT64_C( 6606572555650556850),  INT64_C( 2829390529692828527),  INT64_C( 1381995888231790022),  INT64_C( 6568329687495316506),
        -INT64_C( 1303457298250678015), -INT64_C( 2013848872050549965),  INT64_C( 1771957535492024468), -INT64_C( 3657734556536641579) },
      {  INT64_C( 8570665713878868877),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 3763289271240883118),
         INT64_C(                   0), -INT64_C(  369560300127597164),  INT64_C(                   0),  INT64_C( 1644542361836760311) } },
    { UINT8_C(222),
      {  INT64_C( 1693716282863260189), -INT64_C( 7181241992025315484), -INT64_C( 2718570591168046034), -INT64_C( 9033248451413530712),
        -INT64_C( 5937343786860347514),  INT64_C( 1858518704354021561), -INT64_C( 4687457667859782492), -INT64_C( 7792311420757763850) },
      { -INT64_C( 5252692508087571419), -INT64_C(  611453451093374081), -INT64_C( 3394024332202210286), -INT64_C( 5460606234653922919),
         INT64_C( 9036821187608596148),  INT64_C( 1013709022150741447), -INT64_C( 5106768477839482762),  INT64_C(  561708961651182727) },
      {  INT64_C(                   0), -INT64_C( 7792695443118689565), -INT64_C( 6112594923370256320),  INT64_C( 3952889387642097985),
         INT64_C( 3099477400748248634),  INT64_C(                   0),  INT64_C( 8652517928010286362), -INT64_C( 7230602459106581123) } },
    { UINT8_C(229),
      {  INT64_C( 8138391701483141613),  INT64_C( 4406625028354607943), -INT64_C( 1993379839983388751),  INT64_C( 2662541310383647862),
         INT64_C( 9046393778122708729),  INT64_C( 2568271637353789258), -INT64_C( 8121881179064237364), -INT64_C( 5039088444989734475) },
      { -INT64_C( 1722519523622035611),  INT64_C( 7561249774353008216), -INT64_C( 2405460785354645258),  INT64_C( 3464354200514345880),
        -INT64_C( 6718838163239081926), -INT64_C(  275183546372714198), -INT64_C( 6250246341167154373),  INT64_C( 2512751206208769253) },
      {  INT64_C( 6415872177861106002),  INT64_C(                   0), -INT64_C( 4398840625338034009),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C( 2293088090981075060),  INT64_C( 4074616553478159879), -INT64_C( 2526337238780965222) } },
    { UINT8_C(254),
      { -INT64_C( 3622607429175870549), -INT64_C( 6274694410419404970), -INT64_C( 7031979866514108454), -INT64_C( 6667111988167567258),
         INT64_C(   12644015949398435), -INT64_C( 6358628794173882517),  INT64_C( 4907679902253939692), -INT64_C( 1348765095626235475) },
      { -INT64_C( 3412187665191306502), -INT64_C( 2311003071927724424),  INT64_C( 6505582346217724197),  INT64_C( 4346333461565343769),
        -INT64_C( 8000778778988929343), -INT64_C( 5787191995171151651), -INT64_C( 7069248972678558756), -INT64_C( 1332976243435314173) },
      {  INT64_C(                   0), -INT64_C( 8585697482347129394), -INT64_C(  526397520296384257), -INT64_C( 2320778526602223489),
        -INT64_C( 7988134763039530908),  INT64_C( 6300923284364517448), -INT64_C( 2161569070424619064), -INT64_C( 2681741339061549648) } },
    { UINT8_C(239),
      { -INT64_C( 1385535232953346975),  INT64_C( 5696251178006254957),  INT64_C( 6906112230749870041), -INT64_C(  166219096561869968),
        -INT64_C( 4862855913802450804), -INT64_C(  444736920620238273), -INT64_C( 8760446760531417455), -INT64_C(  334961341082568769) },
      {  INT64_C( 2935809197118471858),  INT64_C( 3902790899556199184),  INT64_C( 6467643616834876965), -INT64_C( 5327742948472452442),
        -INT64_C( 4008634985254182324), -INT64_C( 5837191191359649246), -INT64_C( 7634820792522817257),  INT64_C( 1904947663936929972) },
      {  INT64_C( 1550273964165124883), -INT64_C( 8847701996147097475), -INT64_C( 5072988226124804610), -INT64_C( 5493962045034322410),
         INT64_C(                   0), -INT64_C( 6281928111979887519),  INT64_C( 2051476520655316904),  INT64_C( 1569986322854361203) } },
    { UINT8_C( 94),
      { -INT64_C( 5133576159156088793), -INT64_C( 3958400705177220649), -INT64_C( 8271053347050896680), -INT64_C( 8784986448452653061),
        -INT64_C( 2149372564095095867),  INT64_C( 3728957796702186606), -INT64_C( 4321223872130680659), -INT64_C( 7079217880864431396) },
      {  INT64_C( 4154637502148371899),  INT64_C( 2033637388041814953), -INT64_C( 5191631281194602905), -INT64_C( 5010619628260266496),
         INT64_C( 7826456547109668761),  INT64_C( 2465062992106081707),  INT64_C( 7649721765552376983),  INT64_C( 7524593379129367732) },
      {  INT64_C(                   0), -INT64_C( 1924763317135405696),  INT64_C( 4984059445464052031),  INT64_C( 4651137996996632059),
         INT64_C( 5677083983014572894),  INT64_C(                   0),  INT64_C( 3328497893421696324),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_add_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_add_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_add_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    easysimd_float32 a[16];
    easysimd_float32 b[16];
    easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   667.91), EASYSIMD_FLOAT32_C(   656.58), EASYSIMD_FLOAT32_C(  -596.78), EASYSIMD_FLOAT32_C(  -487.31),
        EASYSIMD_FLOAT32_C(  -822.62), EASYSIMD_FLOAT32_C(   812.31), EASYSIMD_FLOAT32_C(    89.92), EASYSIMD_FLOAT32_C(  -844.98),
        EASYSIMD_FLOAT32_C(  -326.84), EASYSIMD_FLOAT32_C(  -869.81), EASYSIMD_FLOAT32_C(  -327.10), EASYSIMD_FLOAT32_C(  -520.41),
        EASYSIMD_FLOAT32_C(   590.39), EASYSIMD_FLOAT32_C(   190.09), EASYSIMD_FLOAT32_C(  -999.49), EASYSIMD_FLOAT32_C(   279.05) },
      { EASYSIMD_FLOAT32_C(   510.55), EASYSIMD_FLOAT32_C(   821.50), EASYSIMD_FLOAT32_C(  -282.52), EASYSIMD_FLOAT32_C(   624.18),
        EASYSIMD_FLOAT32_C(   410.62), EASYSIMD_FLOAT32_C(  -938.89), EASYSIMD_FLOAT32_C(    71.79), EASYSIMD_FLOAT32_C(   376.91),
        EASYSIMD_FLOAT32_C(   674.13), EASYSIMD_FLOAT32_C(    85.78), EASYSIMD_FLOAT32_C(   -18.27), EASYSIMD_FLOAT32_C(   115.82),
        EASYSIMD_FLOAT32_C(  -281.68), EASYSIMD_FLOAT32_C(  -193.16), EASYSIMD_FLOAT32_C(  -673.77), EASYSIMD_FLOAT32_C(  -613.77) },
      { EASYSIMD_FLOAT32_C(  1178.46), EASYSIMD_FLOAT32_C(  1478.08), EASYSIMD_FLOAT32_C(  -879.31), EASYSIMD_FLOAT32_C(   136.87),
        EASYSIMD_FLOAT32_C(  -412.00), EASYSIMD_FLOAT32_C(  -126.58), EASYSIMD_FLOAT32_C(   161.71), EASYSIMD_FLOAT32_C(  -468.07),
        EASYSIMD_FLOAT32_C(   347.28), EASYSIMD_FLOAT32_C(  -784.02), EASYSIMD_FLOAT32_C(  -345.37), EASYSIMD_FLOAT32_C(  -404.59),
        EASYSIMD_FLOAT32_C(   308.71), EASYSIMD_FLOAT32_C(    -3.07), EASYSIMD_FLOAT32_C( -1673.26), EASYSIMD_FLOAT32_C(  -334.72) } },
    { { EASYSIMD_FLOAT32_C(  -536.58), EASYSIMD_FLOAT32_C(  -270.56), EASYSIMD_FLOAT32_C(  -101.08), EASYSIMD_FLOAT32_C(  -359.21),
        EASYSIMD_FLOAT32_C(  -458.24), EASYSIMD_FLOAT32_C(   988.84), EASYSIMD_FLOAT32_C(  -204.19), EASYSIMD_FLOAT32_C(   214.91),
        EASYSIMD_FLOAT32_C(  -880.97), EASYSIMD_FLOAT32_C(   468.71), EASYSIMD_FLOAT32_C(   694.51), EASYSIMD_FLOAT32_C(   709.42),
        EASYSIMD_FLOAT32_C(  -341.20), EASYSIMD_FLOAT32_C(   695.02), EASYSIMD_FLOAT32_C(   -11.52), EASYSIMD_FLOAT32_C(  -830.65) },
      { EASYSIMD_FLOAT32_C(   516.52), EASYSIMD_FLOAT32_C(   705.95), EASYSIMD_FLOAT32_C(   793.53), EASYSIMD_FLOAT32_C(   -72.87),
        EASYSIMD_FLOAT32_C(   767.06), EASYSIMD_FLOAT32_C(  -134.68), EASYSIMD_FLOAT32_C(  -695.95), EASYSIMD_FLOAT32_C(   441.19),
        EASYSIMD_FLOAT32_C(   951.11), EASYSIMD_FLOAT32_C(   285.78), EASYSIMD_FLOAT32_C(  -442.99), EASYSIMD_FLOAT32_C(  -330.57),
        EASYSIMD_FLOAT32_C(  -907.38), EASYSIMD_FLOAT32_C(  -116.76), EASYSIMD_FLOAT32_C(    55.65), EASYSIMD_FLOAT32_C(  -443.96) },
      { EASYSIMD_FLOAT32_C(   -20.07), EASYSIMD_FLOAT32_C(   435.40), EASYSIMD_FLOAT32_C(   692.45), EASYSIMD_FLOAT32_C(  -432.07),
        EASYSIMD_FLOAT32_C(   308.82), EASYSIMD_FLOAT32_C(   854.16), EASYSIMD_FLOAT32_C(  -900.14), EASYSIMD_FLOAT32_C(   656.10),
        EASYSIMD_FLOAT32_C(    70.14), EASYSIMD_FLOAT32_C(   754.49), EASYSIMD_FLOAT32_C(   251.51), EASYSIMD_FLOAT32_C(   378.85),
        EASYSIMD_FLOAT32_C( -1248.58), EASYSIMD_FLOAT32_C(   578.25), EASYSIMD_FLOAT32_C(    44.13), EASYSIMD_FLOAT32_C( -1274.61) } },
    { { EASYSIMD_FLOAT32_C(   612.68), EASYSIMD_FLOAT32_C(   954.57), EASYSIMD_FLOAT32_C(   196.83), EASYSIMD_FLOAT32_C(  -845.56),
        EASYSIMD_FLOAT32_C(   943.41), EASYSIMD_FLOAT32_C(   992.64), EASYSIMD_FLOAT32_C(   369.35), EASYSIMD_FLOAT32_C(  -937.56),
        EASYSIMD_FLOAT32_C(   461.35), EASYSIMD_FLOAT32_C(    63.86), EASYSIMD_FLOAT32_C(   771.86), EASYSIMD_FLOAT32_C(  -879.85),
        EASYSIMD_FLOAT32_C(  -241.12), EASYSIMD_FLOAT32_C(  -239.67), EASYSIMD_FLOAT32_C(  -710.49), EASYSIMD_FLOAT32_C(  -724.61) },
      { EASYSIMD_FLOAT32_C(  -533.71), EASYSIMD_FLOAT32_C(  -916.96), EASYSIMD_FLOAT32_C(   202.53), EASYSIMD_FLOAT32_C(  -766.65),
        EASYSIMD_FLOAT32_C(   -51.64), EASYSIMD_FLOAT32_C(   506.57), EASYSIMD_FLOAT32_C(   674.54), EASYSIMD_FLOAT32_C(  -100.53),
        EASYSIMD_FLOAT32_C(  -207.65), EASYSIMD_FLOAT32_C(  -768.46), EASYSIMD_FLOAT32_C(   568.90), EASYSIMD_FLOAT32_C(  -115.03),
        EASYSIMD_FLOAT32_C(   114.78), EASYSIMD_FLOAT32_C(  -375.45), EASYSIMD_FLOAT32_C(   441.01), EASYSIMD_FLOAT32_C(  -272.54) },
      { EASYSIMD_FLOAT32_C(    78.97), EASYSIMD_FLOAT32_C(    37.61), EASYSIMD_FLOAT32_C(   399.35), EASYSIMD_FLOAT32_C( -1612.21),
        EASYSIMD_FLOAT32_C(   891.77), EASYSIMD_FLOAT32_C(  1499.21), EASYSIMD_FLOAT32_C(  1043.89), EASYSIMD_FLOAT32_C( -1038.09),
        EASYSIMD_FLOAT32_C(   253.70), EASYSIMD_FLOAT32_C(  -704.60), EASYSIMD_FLOAT32_C(  1340.75), EASYSIMD_FLOAT32_C(  -994.87),
        EASYSIMD_FLOAT32_C(  -126.35), EASYSIMD_FLOAT32_C(  -615.12), EASYSIMD_FLOAT32_C(  -269.48), EASYSIMD_FLOAT32_C(  -997.15) } },
    { { EASYSIMD_FLOAT32_C(  -420.88), EASYSIMD_FLOAT32_C(  -362.16), EASYSIMD_FLOAT32_C(  -118.10), EASYSIMD_FLOAT32_C(  -477.47),
        EASYSIMD_FLOAT32_C(  -369.52), EASYSIMD_FLOAT32_C(  -748.75), EASYSIMD_FLOAT32_C(  -415.03), EASYSIMD_FLOAT32_C(  -908.17),
        EASYSIMD_FLOAT32_C(   315.11), EASYSIMD_FLOAT32_C(  -643.17), EASYSIMD_FLOAT32_C(  -788.02), EASYSIMD_FLOAT32_C(  -926.02),
        EASYSIMD_FLOAT32_C(   117.16), EASYSIMD_FLOAT32_C(  -498.52), EASYSIMD_FLOAT32_C(  -650.63), EASYSIMD_FLOAT32_C(   583.45) },
      { EASYSIMD_FLOAT32_C(  -415.48), EASYSIMD_FLOAT32_C(   551.90), EASYSIMD_FLOAT32_C(   816.80), EASYSIMD_FLOAT32_C(   532.88),
        EASYSIMD_FLOAT32_C(    58.47), EASYSIMD_FLOAT32_C(   491.34), EASYSIMD_FLOAT32_C(  -567.65), EASYSIMD_FLOAT32_C(   850.83),
        EASYSIMD_FLOAT32_C(   722.88), EASYSIMD_FLOAT32_C(  -998.75), EASYSIMD_FLOAT32_C(  -264.20), EASYSIMD_FLOAT32_C(  -162.34),
        EASYSIMD_FLOAT32_C(  -374.20), EASYSIMD_FLOAT32_C(  -823.19), EASYSIMD_FLOAT32_C(   565.12), EASYSIMD_FLOAT32_C(   204.92) },
      { EASYSIMD_FLOAT32_C(  -836.35), EASYSIMD_FLOAT32_C(   189.74), EASYSIMD_FLOAT32_C(   698.70), EASYSIMD_FLOAT32_C(    55.42),
        EASYSIMD_FLOAT32_C(  -311.05), EASYSIMD_FLOAT32_C(  -257.41), EASYSIMD_FLOAT32_C(  -982.68), EASYSIMD_FLOAT32_C(   -57.35),
        EASYSIMD_FLOAT32_C(  1037.99), EASYSIMD_FLOAT32_C( -1641.92), EASYSIMD_FLOAT32_C( -1052.22), EASYSIMD_FLOAT32_C( -1088.36),
        EASYSIMD_FLOAT32_C(  -257.04), EASYSIMD_FLOAT32_C( -1321.70), EASYSIMD_FLOAT32_C(   -85.51), EASYSIMD_FLOAT32_C(   788.38) } },
    { { EASYSIMD_FLOAT32_C(  -185.35), EASYSIMD_FLOAT32_C(  -552.99), EASYSIMD_FLOAT32_C(   727.46), EASYSIMD_FLOAT32_C(   445.13),
        EASYSIMD_FLOAT32_C(  -301.74), EASYSIMD_FLOAT32_C(  -687.57), EASYSIMD_FLOAT32_C(   536.96), EASYSIMD_FLOAT32_C(  -986.63),
        EASYSIMD_FLOAT32_C(  -330.75), EASYSIMD_FLOAT32_C(   748.93), EASYSIMD_FLOAT32_C(  -912.65), EASYSIMD_FLOAT32_C(   786.42),
        EASYSIMD_FLOAT32_C(  -749.58), EASYSIMD_FLOAT32_C(  -563.28), EASYSIMD_FLOAT32_C(   369.87), EASYSIMD_FLOAT32_C(  -165.06) },
      { EASYSIMD_FLOAT32_C(   988.62), EASYSIMD_FLOAT32_C(   186.67), EASYSIMD_FLOAT32_C(  -632.17), EASYSIMD_FLOAT32_C(    47.10),
        EASYSIMD_FLOAT32_C(  -321.99), EASYSIMD_FLOAT32_C(  -199.82), EASYSIMD_FLOAT32_C(  -102.08), EASYSIMD_FLOAT32_C(  -599.11),
        EASYSIMD_FLOAT32_C(  -198.57), EASYSIMD_FLOAT32_C(   633.73), EASYSIMD_FLOAT32_C(   238.55), EASYSIMD_FLOAT32_C(   427.23),
        EASYSIMD_FLOAT32_C(   810.54), EASYSIMD_FLOAT32_C(  -196.33), EASYSIMD_FLOAT32_C(  -367.85), EASYSIMD_FLOAT32_C(  -374.81) },
      { EASYSIMD_FLOAT32_C(   803.28), EASYSIMD_FLOAT32_C(  -366.32), EASYSIMD_FLOAT32_C(    95.28), EASYSIMD_FLOAT32_C(   492.23),
        EASYSIMD_FLOAT32_C(  -623.73), EASYSIMD_FLOAT32_C(  -887.39), EASYSIMD_FLOAT32_C(   434.88), EASYSIMD_FLOAT32_C( -1585.74),
        EASYSIMD_FLOAT32_C(  -529.32), EASYSIMD_FLOAT32_C(  1382.66), EASYSIMD_FLOAT32_C(  -674.10), EASYSIMD_FLOAT32_C(  1213.65),
        EASYSIMD_FLOAT32_C(    60.96), EASYSIMD_FLOAT32_C(  -759.61), EASYSIMD_FLOAT32_C(     2.02), EASYSIMD_FLOAT32_C(  -539.87) } },
    { { EASYSIMD_FLOAT32_C(   250.68), EASYSIMD_FLOAT32_C(  -640.39), EASYSIMD_FLOAT32_C(  -929.68), EASYSIMD_FLOAT32_C(   948.94),
        EASYSIMD_FLOAT32_C(  -327.96), EASYSIMD_FLOAT32_C(   607.27), EASYSIMD_FLOAT32_C(   962.31), EASYSIMD_FLOAT32_C(   341.29),
        EASYSIMD_FLOAT32_C(   356.21), EASYSIMD_FLOAT32_C(  -950.34), EASYSIMD_FLOAT32_C(   127.71), EASYSIMD_FLOAT32_C(   606.63),
        EASYSIMD_FLOAT32_C(  -513.62), EASYSIMD_FLOAT32_C(  -502.42), EASYSIMD_FLOAT32_C(  -558.43), EASYSIMD_FLOAT32_C(  -524.99) },
      { EASYSIMD_FLOAT32_C(   684.24), EASYSIMD_FLOAT32_C(  -190.61), EASYSIMD_FLOAT32_C(   522.10), EASYSIMD_FLOAT32_C(  -637.75),
        EASYSIMD_FLOAT32_C(   609.57), EASYSIMD_FLOAT32_C(  -579.97), EASYSIMD_FLOAT32_C(  -236.86), EASYSIMD_FLOAT32_C(  -589.00),
        EASYSIMD_FLOAT32_C(  -946.25), EASYSIMD_FLOAT32_C(  -998.31), EASYSIMD_FLOAT32_C(   838.23), EASYSIMD_FLOAT32_C(   864.29),
        EASYSIMD_FLOAT32_C(  -194.64), EASYSIMD_FLOAT32_C(  -529.61), EASYSIMD_FLOAT32_C(  -510.52), EASYSIMD_FLOAT32_C(  -943.96) },
      { EASYSIMD_FLOAT32_C(   934.92), EASYSIMD_FLOAT32_C(  -831.00), EASYSIMD_FLOAT32_C(  -407.58), EASYSIMD_FLOAT32_C(   311.19),
        EASYSIMD_FLOAT32_C(   281.61), EASYSIMD_FLOAT32_C(    27.30), EASYSIMD_FLOAT32_C(   725.45), EASYSIMD_FLOAT32_C(  -247.71),
        EASYSIMD_FLOAT32_C(  -590.04), EASYSIMD_FLOAT32_C( -1948.65), EASYSIMD_FLOAT32_C(   965.94), EASYSIMD_FLOAT32_C(  1470.92),
        EASYSIMD_FLOAT32_C(  -708.26), EASYSIMD_FLOAT32_C( -1032.04), EASYSIMD_FLOAT32_C( -1068.95), EASYSIMD_FLOAT32_C( -1468.95) } },
    { { EASYSIMD_FLOAT32_C(  -170.00), EASYSIMD_FLOAT32_C(  -440.20), EASYSIMD_FLOAT32_C(  -995.02), EASYSIMD_FLOAT32_C(   502.03),
        EASYSIMD_FLOAT32_C(  -832.92), EASYSIMD_FLOAT32_C(   967.29), EASYSIMD_FLOAT32_C(  -156.68), EASYSIMD_FLOAT32_C(   523.28),
        EASYSIMD_FLOAT32_C(  -983.05), EASYSIMD_FLOAT32_C(   971.03), EASYSIMD_FLOAT32_C(   129.91), EASYSIMD_FLOAT32_C(  -496.67),
        EASYSIMD_FLOAT32_C(  -531.39), EASYSIMD_FLOAT32_C(   571.48), EASYSIMD_FLOAT32_C(   -21.66), EASYSIMD_FLOAT32_C(  -847.15) },
      { EASYSIMD_FLOAT32_C(  -619.13), EASYSIMD_FLOAT32_C(  -499.55), EASYSIMD_FLOAT32_C(  -484.90), EASYSIMD_FLOAT32_C(   990.45),
        EASYSIMD_FLOAT32_C(   -79.53), EASYSIMD_FLOAT32_C(   278.24), EASYSIMD_FLOAT32_C(  -598.55), EASYSIMD_FLOAT32_C(   -25.77),
        EASYSIMD_FLOAT32_C(   279.93), EASYSIMD_FLOAT32_C(  -760.32), EASYSIMD_FLOAT32_C(  -161.48), EASYSIMD_FLOAT32_C(  -914.71),
        EASYSIMD_FLOAT32_C(  -289.93), EASYSIMD_FLOAT32_C(   328.00), EASYSIMD_FLOAT32_C(  -858.67), EASYSIMD_FLOAT32_C(   540.06) },
      { EASYSIMD_FLOAT32_C(  -789.13), EASYSIMD_FLOAT32_C(  -939.75), EASYSIMD_FLOAT32_C( -1479.92), EASYSIMD_FLOAT32_C(  1492.48),
        EASYSIMD_FLOAT32_C(  -912.45), EASYSIMD_FLOAT32_C(  1245.53), EASYSIMD_FLOAT32_C(  -755.23), EASYSIMD_FLOAT32_C(   497.51),
        EASYSIMD_FLOAT32_C(  -703.12), EASYSIMD_FLOAT32_C(   210.71), EASYSIMD_FLOAT32_C(   -31.57), EASYSIMD_FLOAT32_C( -1411.38),
        EASYSIMD_FLOAT32_C(  -821.33), EASYSIMD_FLOAT32_C(   899.48), EASYSIMD_FLOAT32_C(  -880.33), EASYSIMD_FLOAT32_C(  -307.09) } },
    { { EASYSIMD_FLOAT32_C(   887.80), EASYSIMD_FLOAT32_C(  -853.69), EASYSIMD_FLOAT32_C(    42.10), EASYSIMD_FLOAT32_C(  -945.12),
        EASYSIMD_FLOAT32_C(  -886.40), EASYSIMD_FLOAT32_C(   885.42), EASYSIMD_FLOAT32_C(   578.16), EASYSIMD_FLOAT32_C(  -869.46),
        EASYSIMD_FLOAT32_C(   856.45), EASYSIMD_FLOAT32_C(  -291.93), EASYSIMD_FLOAT32_C(  -366.12), EASYSIMD_FLOAT32_C(  -674.94),
        EASYSIMD_FLOAT32_C(  -720.45), EASYSIMD_FLOAT32_C(   612.22), EASYSIMD_FLOAT32_C(  -522.09), EASYSIMD_FLOAT32_C(  -339.57) },
      { EASYSIMD_FLOAT32_C(  -887.34), EASYSIMD_FLOAT32_C(    -6.99), EASYSIMD_FLOAT32_C(  -349.13), EASYSIMD_FLOAT32_C(    33.14),
        EASYSIMD_FLOAT32_C(  -728.74), EASYSIMD_FLOAT32_C(    52.32), EASYSIMD_FLOAT32_C(  -992.63), EASYSIMD_FLOAT32_C(   551.19),
        EASYSIMD_FLOAT32_C(   292.00), EASYSIMD_FLOAT32_C(  -154.11), EASYSIMD_FLOAT32_C(   636.48), EASYSIMD_FLOAT32_C(  -997.93),
        EASYSIMD_FLOAT32_C(  -826.11), EASYSIMD_FLOAT32_C(   777.81), EASYSIMD_FLOAT32_C(   542.14), EASYSIMD_FLOAT32_C(  -938.31) },
      { EASYSIMD_FLOAT32_C(     0.46), EASYSIMD_FLOAT32_C(  -860.68), EASYSIMD_FLOAT32_C(  -307.03), EASYSIMD_FLOAT32_C(  -911.99),
        EASYSIMD_FLOAT32_C( -1615.15), EASYSIMD_FLOAT32_C(   937.74), EASYSIMD_FLOAT32_C(  -414.47), EASYSIMD_FLOAT32_C(  -318.27),
        EASYSIMD_FLOAT32_C(  1148.46), EASYSIMD_FLOAT32_C(  -446.04), EASYSIMD_FLOAT32_C(   270.35), EASYSIMD_FLOAT32_C( -1672.87),
        EASYSIMD_FLOAT32_C( -1546.56), EASYSIMD_FLOAT32_C(  1390.02), EASYSIMD_FLOAT32_C(    20.05), EASYSIMD_FLOAT32_C( -1277.88) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_add_ps(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_add_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_add_round_ps (EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT32_C(  -215.09), EASYSIMD_FLOAT32_C(  -266.59), EASYSIMD_FLOAT32_C(   769.27), EASYSIMD_FLOAT32_C(   -61.17),
        EASYSIMD_FLOAT32_C(  -936.67), EASYSIMD_FLOAT32_C(  -448.41), EASYSIMD_FLOAT32_C(   -33.37), EASYSIMD_FLOAT32_C(   370.37),
        EASYSIMD_FLOAT32_C(  -916.11), EASYSIMD_FLOAT32_C(   186.95), EASYSIMD_FLOAT32_C(  -386.88), EASYSIMD_FLOAT32_C(   -97.73),
        EASYSIMD_FLOAT32_C(    79.46), EASYSIMD_FLOAT32_C(  -746.83), EASYSIMD_FLOAT32_C(   190.50), EASYSIMD_FLOAT32_C(   365.82) },
      { EASYSIMD_FLOAT32_C(  -270.17), EASYSIMD_FLOAT32_C(  -704.18), EASYSIMD_FLOAT32_C(   923.50), EASYSIMD_FLOAT32_C(   -67.36),
        EASYSIMD_FLOAT32_C(   860.14), EASYSIMD_FLOAT32_C(   420.86), EASYSIMD_FLOAT32_C(   887.16), EASYSIMD_FLOAT32_C(  -954.97),
        EASYSIMD_FLOAT32_C(   330.17), EASYSIMD_FLOAT32_C(   462.52), EASYSIMD_FLOAT32_C(   120.61), EASYSIMD_FLOAT32_C(   777.27),
        EASYSIMD_FLOAT32_C(  -364.13), EASYSIMD_FLOAT32_C(   127.27), EASYSIMD_FLOAT32_C(   583.11), EASYSIMD_FLOAT32_C(   420.77) },
      { EASYSIMD_FLOAT32_C(  -485.00), EASYSIMD_FLOAT32_C(  -971.00), EASYSIMD_FLOAT32_C(  1693.00), EASYSIMD_FLOAT32_C(  -129.00),
        EASYSIMD_FLOAT32_C(   -77.00), EASYSIMD_FLOAT32_C(   -28.00), EASYSIMD_FLOAT32_C(   854.00), EASYSIMD_FLOAT32_C(  -585.00),
        EASYSIMD_FLOAT32_C(  -586.00), EASYSIMD_FLOAT32_C(   649.00), EASYSIMD_FLOAT32_C(  -266.00), EASYSIMD_FLOAT32_C(   680.00),
        EASYSIMD_FLOAT32_C(  -285.00), EASYSIMD_FLOAT32_C(  -620.00), EASYSIMD_FLOAT32_C(   774.00), EASYSIMD_FLOAT32_C(   787.00) },
      { EASYSIMD_FLOAT32_C(  -486.00), EASYSIMD_FLOAT32_C(  -971.00), EASYSIMD_FLOAT32_C(  1692.00), EASYSIMD_FLOAT32_C(  -129.00),
        EASYSIMD_FLOAT32_C(   -77.00), EASYSIMD_FLOAT32_C(   -28.00), EASYSIMD_FLOAT32_C(   853.00), EASYSIMD_FLOAT32_C(  -585.00),
        EASYSIMD_FLOAT32_C(  -586.00), EASYSIMD_FLOAT32_C(   649.00), EASYSIMD_FLOAT32_C(  -267.00), EASYSIMD_FLOAT32_C(   679.00),
        EASYSIMD_FLOAT32_C(  -285.00), EASYSIMD_FLOAT32_C(  -620.00), EASYSIMD_FLOAT32_C(   773.00), EASYSIMD_FLOAT32_C(   786.00) },
      { EASYSIMD_FLOAT32_C(  -485.00), EASYSIMD_FLOAT32_C(  -970.00), EASYSIMD_FLOAT32_C(  1693.00), EASYSIMD_FLOAT32_C(  -128.00),
        EASYSIMD_FLOAT32_C(   -76.00), EASYSIMD_FLOAT32_C(   -27.00), EASYSIMD_FLOAT32_C(   854.00), EASYSIMD_FLOAT32_C(  -584.00),
        EASYSIMD_FLOAT32_C(  -585.00), EASYSIMD_FLOAT32_C(   650.00), EASYSIMD_FLOAT32_C(  -266.00), EASYSIMD_FLOAT32_C(   680.00),
        EASYSIMD_FLOAT32_C(  -284.00), EASYSIMD_FLOAT32_C(  -619.00), EASYSIMD_FLOAT32_C(   774.00), EASYSIMD_FLOAT32_C(   787.00) },
      { EASYSIMD_FLOAT32_C(  -485.00), EASYSIMD_FLOAT32_C(  -970.00), EASYSIMD_FLOAT32_C(  1692.00), EASYSIMD_FLOAT32_C(  -128.00),
        EASYSIMD_FLOAT32_C(   -76.00), EASYSIMD_FLOAT32_C(   -27.00), EASYSIMD_FLOAT32_C(   853.00), EASYSIMD_FLOAT32_C(  -584.00),
        EASYSIMD_FLOAT32_C(  -585.00), EASYSIMD_FLOAT32_C(   649.00), EASYSIMD_FLOAT32_C(  -266.00), EASYSIMD_FLOAT32_C(   679.00),
        EASYSIMD_FLOAT32_C(  -284.00), EASYSIMD_FLOAT32_C(  -619.00), EASYSIMD_FLOAT32_C(   773.00), EASYSIMD_FLOAT32_C(   786.00) },
      { EASYSIMD_FLOAT32_C(  -485.00), EASYSIMD_FLOAT32_C(  -971.00), EASYSIMD_FLOAT32_C(  1693.00), EASYSIMD_FLOAT32_C(  -129.00),
        EASYSIMD_FLOAT32_C(   -77.00), EASYSIMD_FLOAT32_C(   -28.00), EASYSIMD_FLOAT32_C(   854.00), EASYSIMD_FLOAT32_C(  -585.00),
        EASYSIMD_FLOAT32_C(  -586.00), EASYSIMD_FLOAT32_C(   649.00), EASYSIMD_FLOAT32_C(  -266.00), EASYSIMD_FLOAT32_C(   680.00),
        EASYSIMD_FLOAT32_C(  -285.00), EASYSIMD_FLOAT32_C(  -620.00), EASYSIMD_FLOAT32_C(   774.00), EASYSIMD_FLOAT32_C(   787.00) } },
    { { EASYSIMD_FLOAT32_C(   860.68), EASYSIMD_FLOAT32_C(   352.39), EASYSIMD_FLOAT32_C(  -640.40), EASYSIMD_FLOAT32_C(   924.01),
        EASYSIMD_FLOAT32_C(   903.98), EASYSIMD_FLOAT32_C(   326.23), EASYSIMD_FLOAT32_C(   294.38), EASYSIMD_FLOAT32_C(   987.87),
        EASYSIMD_FLOAT32_C(  -486.82), EASYSIMD_FLOAT32_C(   907.51), EASYSIMD_FLOAT32_C(  -109.86), EASYSIMD_FLOAT32_C(   592.64),
        EASYSIMD_FLOAT32_C(  -839.33), EASYSIMD_FLOAT32_C(  -919.36), EASYSIMD_FLOAT32_C(   -41.54), EASYSIMD_FLOAT32_C(  -109.50) },
      { EASYSIMD_FLOAT32_C(  -623.54), EASYSIMD_FLOAT32_C(  -118.04), EASYSIMD_FLOAT32_C(   823.15), EASYSIMD_FLOAT32_C(  -763.40),
        EASYSIMD_FLOAT32_C(  -697.18), EASYSIMD_FLOAT32_C(   710.30), EASYSIMD_FLOAT32_C(  -718.37), EASYSIMD_FLOAT32_C(   633.00),
        EASYSIMD_FLOAT32_C(   172.83), EASYSIMD_FLOAT32_C(   402.24), EASYSIMD_FLOAT32_C(   410.27), EASYSIMD_FLOAT32_C(   808.69),
        EASYSIMD_FLOAT32_C(  -470.49), EASYSIMD_FLOAT32_C(    -6.61), EASYSIMD_FLOAT32_C(   229.47), EASYSIMD_FLOAT32_C(  -609.80) },
      { EASYSIMD_FLOAT32_C(   237.00), EASYSIMD_FLOAT32_C(   234.00), EASYSIMD_FLOAT32_C(   183.00), EASYSIMD_FLOAT32_C(   161.00),
        EASYSIMD_FLOAT32_C(   207.00), EASYSIMD_FLOAT32_C(  1037.00), EASYSIMD_FLOAT32_C(  -424.00), EASYSIMD_FLOAT32_C(  1621.00),
        EASYSIMD_FLOAT32_C(  -314.00), EASYSIMD_FLOAT32_C(  1310.00), EASYSIMD_FLOAT32_C(   300.00), EASYSIMD_FLOAT32_C(  1401.00),
        EASYSIMD_FLOAT32_C( -1310.00), EASYSIMD_FLOAT32_C(  -926.00), EASYSIMD_FLOAT32_C(   188.00), EASYSIMD_FLOAT32_C(  -719.00) },
      { EASYSIMD_FLOAT32_C(   237.00), EASYSIMD_FLOAT32_C(   234.00), EASYSIMD_FLOAT32_C(   182.00), EASYSIMD_FLOAT32_C(   160.00),
        EASYSIMD_FLOAT32_C(   206.00), EASYSIMD_FLOAT32_C(  1036.00), EASYSIMD_FLOAT32_C(  -424.00), EASYSIMD_FLOAT32_C(  1620.00),
        EASYSIMD_FLOAT32_C(  -314.00), EASYSIMD_FLOAT32_C(  1309.00), EASYSIMD_FLOAT32_C(   300.00), EASYSIMD_FLOAT32_C(  1401.00),
        EASYSIMD_FLOAT32_C( -1310.00), EASYSIMD_FLOAT32_C(  -926.00), EASYSIMD_FLOAT32_C(   187.00), EASYSIMD_FLOAT32_C(  -720.00) },
      { EASYSIMD_FLOAT32_C(   238.00), EASYSIMD_FLOAT32_C(   235.00), EASYSIMD_FLOAT32_C(   183.00), EASYSIMD_FLOAT32_C(   161.00),
        EASYSIMD_FLOAT32_C(   207.00), EASYSIMD_FLOAT32_C(  1037.00), EASYSIMD_FLOAT32_C(  -423.00), EASYSIMD_FLOAT32_C(  1621.00),
        EASYSIMD_FLOAT32_C(  -313.00), EASYSIMD_FLOAT32_C(  1310.00), EASYSIMD_FLOAT32_C(   301.00), EASYSIMD_FLOAT32_C(  1402.00),
        EASYSIMD_FLOAT32_C( -1309.00), EASYSIMD_FLOAT32_C(  -925.00), EASYSIMD_FLOAT32_C(   188.00), EASYSIMD_FLOAT32_C(  -719.00) },
      { EASYSIMD_FLOAT32_C(   237.00), EASYSIMD_FLOAT32_C(   234.00), EASYSIMD_FLOAT32_C(   182.00), EASYSIMD_FLOAT32_C(   160.00),
        EASYSIMD_FLOAT32_C(   206.00), EASYSIMD_FLOAT32_C(  1036.00), EASYSIMD_FLOAT32_C(  -423.00), EASYSIMD_FLOAT32_C(  1620.00),
        EASYSIMD_FLOAT32_C(  -313.00), EASYSIMD_FLOAT32_C(  1309.00), EASYSIMD_FLOAT32_C(   300.00), EASYSIMD_FLOAT32_C(  1401.00),
        EASYSIMD_FLOAT32_C( -1309.00), EASYSIMD_FLOAT32_C(  -925.00), EASYSIMD_FLOAT32_C(   187.00), EASYSIMD_FLOAT32_C(  -719.00) },
      { EASYSIMD_FLOAT32_C(   237.00), EASYSIMD_FLOAT32_C(   234.00), EASYSIMD_FLOAT32_C(   183.00), EASYSIMD_FLOAT32_C(   161.00),
        EASYSIMD_FLOAT32_C(   207.00), EASYSIMD_FLOAT32_C(  1037.00), EASYSIMD_FLOAT32_C(  -424.00), EASYSIMD_FLOAT32_C(  1621.00),
        EASYSIMD_FLOAT32_C(  -314.00), EASYSIMD_FLOAT32_C(  1310.00), EASYSIMD_FLOAT32_C(   300.00), EASYSIMD_FLOAT32_C(  1401.00),
        EASYSIMD_FLOAT32_C( -1310.00), EASYSIMD_FLOAT32_C(  -926.00), EASYSIMD_FLOAT32_C(   188.00), EASYSIMD_FLOAT32_C(  -719.00) } },
    { { EASYSIMD_FLOAT32_C(  -654.23), EASYSIMD_FLOAT32_C(   589.07), EASYSIMD_FLOAT32_C(  -685.79), EASYSIMD_FLOAT32_C(  -750.25),
        EASYSIMD_FLOAT32_C(   -84.70), EASYSIMD_FLOAT32_C(   608.59), EASYSIMD_FLOAT32_C(  -762.37), EASYSIMD_FLOAT32_C(   428.48),
        EASYSIMD_FLOAT32_C(   516.10), EASYSIMD_FLOAT32_C(   127.77), EASYSIMD_FLOAT32_C(    21.13), EASYSIMD_FLOAT32_C(   676.77),
        EASYSIMD_FLOAT32_C(   208.41), EASYSIMD_FLOAT32_C(   979.59), EASYSIMD_FLOAT32_C(  -432.73), EASYSIMD_FLOAT32_C(   584.87) },
      { EASYSIMD_FLOAT32_C(  -138.45), EASYSIMD_FLOAT32_C(  -609.58), EASYSIMD_FLOAT32_C(   821.46), EASYSIMD_FLOAT32_C(   164.38),
        EASYSIMD_FLOAT32_C(  -899.28), EASYSIMD_FLOAT32_C(  -896.91), EASYSIMD_FLOAT32_C(  -202.63), EASYSIMD_FLOAT32_C(   273.55),
        EASYSIMD_FLOAT32_C(   505.33), EASYSIMD_FLOAT32_C(  -792.36), EASYSIMD_FLOAT32_C(    82.25), EASYSIMD_FLOAT32_C(  -965.16),
        EASYSIMD_FLOAT32_C(   201.03), EASYSIMD_FLOAT32_C(  -688.29), EASYSIMD_FLOAT32_C(  -574.96), EASYSIMD_FLOAT32_C(   546.80) },
      { EASYSIMD_FLOAT32_C(  -793.00), EASYSIMD_FLOAT32_C(   -21.00), EASYSIMD_FLOAT32_C(   136.00), EASYSIMD_FLOAT32_C(  -586.00),
        EASYSIMD_FLOAT32_C(  -984.00), EASYSIMD_FLOAT32_C(  -288.00), EASYSIMD_FLOAT32_C(  -965.00), EASYSIMD_FLOAT32_C(   702.00),
        EASYSIMD_FLOAT32_C(  1021.00), EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(  -288.00),
        EASYSIMD_FLOAT32_C(   409.00), EASYSIMD_FLOAT32_C(   291.00), EASYSIMD_FLOAT32_C( -1008.00), EASYSIMD_FLOAT32_C(  1132.00) },
      { EASYSIMD_FLOAT32_C(  -793.00), EASYSIMD_FLOAT32_C(   -21.00), EASYSIMD_FLOAT32_C(   135.00), EASYSIMD_FLOAT32_C(  -586.00),
        EASYSIMD_FLOAT32_C(  -984.00), EASYSIMD_FLOAT32_C(  -289.00), EASYSIMD_FLOAT32_C(  -965.00), EASYSIMD_FLOAT32_C(   702.00),
        EASYSIMD_FLOAT32_C(  1021.00), EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(  -289.00),
        EASYSIMD_FLOAT32_C(   409.00), EASYSIMD_FLOAT32_C(   291.00), EASYSIMD_FLOAT32_C( -1008.00), EASYSIMD_FLOAT32_C(  1131.00) },
      { EASYSIMD_FLOAT32_C(  -792.00), EASYSIMD_FLOAT32_C(   -20.00), EASYSIMD_FLOAT32_C(   136.00), EASYSIMD_FLOAT32_C(  -585.00),
        EASYSIMD_FLOAT32_C(  -983.00), EASYSIMD_FLOAT32_C(  -288.00), EASYSIMD_FLOAT32_C(  -965.00), EASYSIMD_FLOAT32_C(   703.00),
        EASYSIMD_FLOAT32_C(  1022.00), EASYSIMD_FLOAT32_C(  -664.00), EASYSIMD_FLOAT32_C(   104.00), EASYSIMD_FLOAT32_C(  -288.00),
        EASYSIMD_FLOAT32_C(   410.00), EASYSIMD_FLOAT32_C(   292.00), EASYSIMD_FLOAT32_C( -1007.00), EASYSIMD_FLOAT32_C(  1132.00) },
      { EASYSIMD_FLOAT32_C(  -792.00), EASYSIMD_FLOAT32_C(   -20.00), EASYSIMD_FLOAT32_C(   135.00), EASYSIMD_FLOAT32_C(  -585.00),
        EASYSIMD_FLOAT32_C(  -983.00), EASYSIMD_FLOAT32_C(  -288.00), EASYSIMD_FLOAT32_C(  -965.00), EASYSIMD_FLOAT32_C(   702.00),
        EASYSIMD_FLOAT32_C(  1021.00), EASYSIMD_FLOAT32_C(  -664.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(  -288.00),
        EASYSIMD_FLOAT32_C(   409.00), EASYSIMD_FLOAT32_C(   291.00), EASYSIMD_FLOAT32_C( -1007.00), EASYSIMD_FLOAT32_C(  1131.00) },
      { EASYSIMD_FLOAT32_C(  -793.00), EASYSIMD_FLOAT32_C(   -21.00), EASYSIMD_FLOAT32_C(   136.00), EASYSIMD_FLOAT32_C(  -586.00),
        EASYSIMD_FLOAT32_C(  -984.00), EASYSIMD_FLOAT32_C(  -288.00), EASYSIMD_FLOAT32_C(  -965.00), EASYSIMD_FLOAT32_C(   702.00),
        EASYSIMD_FLOAT32_C(  1021.00), EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(  -288.00),
        EASYSIMD_FLOAT32_C(   409.00), EASYSIMD_FLOAT32_C(   291.00), EASYSIMD_FLOAT32_C( -1008.00), EASYSIMD_FLOAT32_C(  1132.00) } },
    { { EASYSIMD_FLOAT32_C(   900.78), EASYSIMD_FLOAT32_C(  -260.75), EASYSIMD_FLOAT32_C(   796.56), EASYSIMD_FLOAT32_C(  -183.92),
        EASYSIMD_FLOAT32_C(  -652.16), EASYSIMD_FLOAT32_C(  -965.81), EASYSIMD_FLOAT32_C(  -755.43), EASYSIMD_FLOAT32_C(   863.94),
        EASYSIMD_FLOAT32_C(   161.96), EASYSIMD_FLOAT32_C(   265.69), EASYSIMD_FLOAT32_C(   540.71), EASYSIMD_FLOAT32_C(  -629.63),
        EASYSIMD_FLOAT32_C(   245.28), EASYSIMD_FLOAT32_C(  -892.02), EASYSIMD_FLOAT32_C(   955.23), EASYSIMD_FLOAT32_C(  -893.16) },
      { EASYSIMD_FLOAT32_C(  -501.60), EASYSIMD_FLOAT32_C(   776.70), EASYSIMD_FLOAT32_C(   271.21), EASYSIMD_FLOAT32_C(  -400.87),
        EASYSIMD_FLOAT32_C(   879.79), EASYSIMD_FLOAT32_C(  -931.42), EASYSIMD_FLOAT32_C(   872.68), EASYSIMD_FLOAT32_C(   385.12),
        EASYSIMD_FLOAT32_C(  -723.77), EASYSIMD_FLOAT32_C(   -45.08), EASYSIMD_FLOAT32_C(   419.96), EASYSIMD_FLOAT32_C(   477.26),
        EASYSIMD_FLOAT32_C(   266.64), EASYSIMD_FLOAT32_C(   845.00), EASYSIMD_FLOAT32_C(    24.06), EASYSIMD_FLOAT32_C(   167.42) },
      { EASYSIMD_FLOAT32_C(   399.00), EASYSIMD_FLOAT32_C(   516.00), EASYSIMD_FLOAT32_C(  1068.00), EASYSIMD_FLOAT32_C(  -585.00),
        EASYSIMD_FLOAT32_C(   228.00), EASYSIMD_FLOAT32_C( -1897.00), EASYSIMD_FLOAT32_C(   117.00), EASYSIMD_FLOAT32_C(  1249.00),
        EASYSIMD_FLOAT32_C(  -562.00), EASYSIMD_FLOAT32_C(   221.00), EASYSIMD_FLOAT32_C(   961.00), EASYSIMD_FLOAT32_C(  -152.00),
        EASYSIMD_FLOAT32_C(   512.00), EASYSIMD_FLOAT32_C(   -47.00), EASYSIMD_FLOAT32_C(   979.00), EASYSIMD_FLOAT32_C(  -726.00) },
      { EASYSIMD_FLOAT32_C(   399.00), EASYSIMD_FLOAT32_C(   515.00), EASYSIMD_FLOAT32_C(  1067.00), EASYSIMD_FLOAT32_C(  -585.00),
        EASYSIMD_FLOAT32_C(   227.00), EASYSIMD_FLOAT32_C( -1898.00), EASYSIMD_FLOAT32_C(   117.00), EASYSIMD_FLOAT32_C(  1249.00),
        EASYSIMD_FLOAT32_C(  -562.00), EASYSIMD_FLOAT32_C(   220.00), EASYSIMD_FLOAT32_C(   960.00), EASYSIMD_FLOAT32_C(  -153.00),
        EASYSIMD_FLOAT32_C(   511.00), EASYSIMD_FLOAT32_C(   -48.00), EASYSIMD_FLOAT32_C(   979.00), EASYSIMD_FLOAT32_C(  -726.00) },
      { EASYSIMD_FLOAT32_C(   400.00), EASYSIMD_FLOAT32_C(   516.00), EASYSIMD_FLOAT32_C(  1068.00), EASYSIMD_FLOAT32_C(  -584.00),
        EASYSIMD_FLOAT32_C(   228.00), EASYSIMD_FLOAT32_C( -1897.00), EASYSIMD_FLOAT32_C(   118.00), EASYSIMD_FLOAT32_C(  1250.00),
        EASYSIMD_FLOAT32_C(  -561.00), EASYSIMD_FLOAT32_C(   221.00), EASYSIMD_FLOAT32_C(   961.00), EASYSIMD_FLOAT32_C(  -152.00),
        EASYSIMD_FLOAT32_C(   512.00), EASYSIMD_FLOAT32_C(   -47.00), EASYSIMD_FLOAT32_C(   980.00), EASYSIMD_FLOAT32_C(  -725.00) },
      { EASYSIMD_FLOAT32_C(   399.00), EASYSIMD_FLOAT32_C(   515.00), EASYSIMD_FLOAT32_C(  1067.00), EASYSIMD_FLOAT32_C(  -584.00),
        EASYSIMD_FLOAT32_C(   227.00), EASYSIMD_FLOAT32_C( -1897.00), EASYSIMD_FLOAT32_C(   117.00), EASYSIMD_FLOAT32_C(  1249.00),
        EASYSIMD_FLOAT32_C(  -561.00), EASYSIMD_FLOAT32_C(   220.00), EASYSIMD_FLOAT32_C(   960.00), EASYSIMD_FLOAT32_C(  -152.00),
        EASYSIMD_FLOAT32_C(   511.00), EASYSIMD_FLOAT32_C(   -47.00), EASYSIMD_FLOAT32_C(   979.00), EASYSIMD_FLOAT32_C(  -725.00) },
      { EASYSIMD_FLOAT32_C(   399.00), EASYSIMD_FLOAT32_C(   516.00), EASYSIMD_FLOAT32_C(  1068.00), EASYSIMD_FLOAT32_C(  -585.00),
        EASYSIMD_FLOAT32_C(   228.00), EASYSIMD_FLOAT32_C( -1897.00), EASYSIMD_FLOAT32_C(   117.00), EASYSIMD_FLOAT32_C(  1249.00),
        EASYSIMD_FLOAT32_C(  -562.00), EASYSIMD_FLOAT32_C(   221.00), EASYSIMD_FLOAT32_C(   961.00), EASYSIMD_FLOAT32_C(  -152.00),
        EASYSIMD_FLOAT32_C(   512.00), EASYSIMD_FLOAT32_C(   -47.00), EASYSIMD_FLOAT32_C(   979.00), EASYSIMD_FLOAT32_C(  -726.00) } },
    { { EASYSIMD_FLOAT32_C(  -415.75), EASYSIMD_FLOAT32_C(  -179.38), EASYSIMD_FLOAT32_C(   983.50), EASYSIMD_FLOAT32_C(   -67.91),
        EASYSIMD_FLOAT32_C(  -145.20), EASYSIMD_FLOAT32_C(  -771.93), EASYSIMD_FLOAT32_C(  -203.97), EASYSIMD_FLOAT32_C(  -983.24),
        EASYSIMD_FLOAT32_C(   493.76), EASYSIMD_FLOAT32_C(  -663.26), EASYSIMD_FLOAT32_C(  -612.87), EASYSIMD_FLOAT32_C(  -260.96),
        EASYSIMD_FLOAT32_C(  -555.27), EASYSIMD_FLOAT32_C(  -657.64), EASYSIMD_FLOAT32_C(  -154.12), EASYSIMD_FLOAT32_C(   -56.87) },
      { EASYSIMD_FLOAT32_C(  -880.94), EASYSIMD_FLOAT32_C(  -882.91), EASYSIMD_FLOAT32_C(   542.26), EASYSIMD_FLOAT32_C(   998.84),
        EASYSIMD_FLOAT32_C(  -814.32), EASYSIMD_FLOAT32_C(   414.94), EASYSIMD_FLOAT32_C(   383.96), EASYSIMD_FLOAT32_C(  -538.09),
        EASYSIMD_FLOAT32_C(  -630.14), EASYSIMD_FLOAT32_C(  -196.08), EASYSIMD_FLOAT32_C(   939.17), EASYSIMD_FLOAT32_C(   636.49),
        EASYSIMD_FLOAT32_C(  -351.08), EASYSIMD_FLOAT32_C(   -36.77), EASYSIMD_FLOAT32_C(  -196.09), EASYSIMD_FLOAT32_C(   233.17) },
      { EASYSIMD_FLOAT32_C( -1297.00), EASYSIMD_FLOAT32_C( -1062.00), EASYSIMD_FLOAT32_C(  1526.00), EASYSIMD_FLOAT32_C(   931.00),
        EASYSIMD_FLOAT32_C(  -960.00), EASYSIMD_FLOAT32_C(  -357.00), EASYSIMD_FLOAT32_C(   180.00), EASYSIMD_FLOAT32_C( -1521.00),
        EASYSIMD_FLOAT32_C(  -136.00), EASYSIMD_FLOAT32_C(  -859.00), EASYSIMD_FLOAT32_C(   326.00), EASYSIMD_FLOAT32_C(   376.00),
        EASYSIMD_FLOAT32_C(  -906.00), EASYSIMD_FLOAT32_C(  -694.00), EASYSIMD_FLOAT32_C(  -350.00), EASYSIMD_FLOAT32_C(   176.00) },
      { EASYSIMD_FLOAT32_C( -1297.00), EASYSIMD_FLOAT32_C( -1063.00), EASYSIMD_FLOAT32_C(  1525.00), EASYSIMD_FLOAT32_C(   930.00),
        EASYSIMD_FLOAT32_C(  -960.00), EASYSIMD_FLOAT32_C(  -357.00), EASYSIMD_FLOAT32_C(   179.00), EASYSIMD_FLOAT32_C( -1522.00),
        EASYSIMD_FLOAT32_C(  -137.00), EASYSIMD_FLOAT32_C(  -860.00), EASYSIMD_FLOAT32_C(   326.00), EASYSIMD_FLOAT32_C(   375.00),
        EASYSIMD_FLOAT32_C(  -907.00), EASYSIMD_FLOAT32_C(  -695.00), EASYSIMD_FLOAT32_C(  -351.00), EASYSIMD_FLOAT32_C(   176.00) },
      { EASYSIMD_FLOAT32_C( -1296.00), EASYSIMD_FLOAT32_C( -1062.00), EASYSIMD_FLOAT32_C(  1526.00), EASYSIMD_FLOAT32_C(   931.00),
        EASYSIMD_FLOAT32_C(  -959.00), EASYSIMD_FLOAT32_C(  -356.00), EASYSIMD_FLOAT32_C(   180.00), EASYSIMD_FLOAT32_C( -1521.00),
        EASYSIMD_FLOAT32_C(  -136.00), EASYSIMD_FLOAT32_C(  -859.00), EASYSIMD_FLOAT32_C(   327.00), EASYSIMD_FLOAT32_C(   376.00),
        EASYSIMD_FLOAT32_C(  -906.00), EASYSIMD_FLOAT32_C(  -694.00), EASYSIMD_FLOAT32_C(  -350.00), EASYSIMD_FLOAT32_C(   177.00) },
      { EASYSIMD_FLOAT32_C( -1296.00), EASYSIMD_FLOAT32_C( -1062.00), EASYSIMD_FLOAT32_C(  1525.00), EASYSIMD_FLOAT32_C(   930.00),
        EASYSIMD_FLOAT32_C(  -959.00), EASYSIMD_FLOAT32_C(  -356.00), EASYSIMD_FLOAT32_C(   179.00), EASYSIMD_FLOAT32_C( -1521.00),
        EASYSIMD_FLOAT32_C(  -136.00), EASYSIMD_FLOAT32_C(  -859.00), EASYSIMD_FLOAT32_C(   326.00), EASYSIMD_FLOAT32_C(   375.00),
        EASYSIMD_FLOAT32_C(  -906.00), EASYSIMD_FLOAT32_C(  -694.00), EASYSIMD_FLOAT32_C(  -350.00), EASYSIMD_FLOAT32_C(   176.00) },
      { EASYSIMD_FLOAT32_C( -1297.00), EASYSIMD_FLOAT32_C( -1062.00), EASYSIMD_FLOAT32_C(  1526.00), EASYSIMD_FLOAT32_C(   931.00),
        EASYSIMD_FLOAT32_C(  -960.00), EASYSIMD_FLOAT32_C(  -357.00), EASYSIMD_FLOAT32_C(   180.00), EASYSIMD_FLOAT32_C( -1521.00),
        EASYSIMD_FLOAT32_C(  -136.00), EASYSIMD_FLOAT32_C(  -859.00), EASYSIMD_FLOAT32_C(   326.00), EASYSIMD_FLOAT32_C(   376.00),
        EASYSIMD_FLOAT32_C(  -906.00), EASYSIMD_FLOAT32_C(  -694.00), EASYSIMD_FLOAT32_C(  -350.00), EASYSIMD_FLOAT32_C(   176.00) } },
    { { EASYSIMD_FLOAT32_C(   783.85), EASYSIMD_FLOAT32_C(  -212.59), EASYSIMD_FLOAT32_C(  -834.74), EASYSIMD_FLOAT32_C(  -361.35),
        EASYSIMD_FLOAT32_C(    15.48), EASYSIMD_FLOAT32_C(   -38.71), EASYSIMD_FLOAT32_C(  -344.59), EASYSIMD_FLOAT32_C(  -490.76),
        EASYSIMD_FLOAT32_C(   298.03), EASYSIMD_FLOAT32_C(    42.54), EASYSIMD_FLOAT32_C(   248.28), EASYSIMD_FLOAT32_C(   742.76),
        EASYSIMD_FLOAT32_C(   384.90), EASYSIMD_FLOAT32_C(  -905.84), EASYSIMD_FLOAT32_C(  -314.12), EASYSIMD_FLOAT32_C(   503.95) },
      { EASYSIMD_FLOAT32_C(  -788.75), EASYSIMD_FLOAT32_C(  -771.86), EASYSIMD_FLOAT32_C(   502.79), EASYSIMD_FLOAT32_C(  -603.07),
        EASYSIMD_FLOAT32_C(   643.08), EASYSIMD_FLOAT32_C(  -113.24), EASYSIMD_FLOAT32_C(  -141.16), EASYSIMD_FLOAT32_C(  -987.06),
        EASYSIMD_FLOAT32_C(   690.68), EASYSIMD_FLOAT32_C(  -202.00), EASYSIMD_FLOAT32_C(   649.43), EASYSIMD_FLOAT32_C(  -660.40),
        EASYSIMD_FLOAT32_C(   761.23), EASYSIMD_FLOAT32_C(  -546.66), EASYSIMD_FLOAT32_C(   572.76), EASYSIMD_FLOAT32_C(   545.08) },
      { EASYSIMD_FLOAT32_C(    -5.00), EASYSIMD_FLOAT32_C(  -984.00), EASYSIMD_FLOAT32_C(  -332.00), EASYSIMD_FLOAT32_C(  -964.00),
        EASYSIMD_FLOAT32_C(   659.00), EASYSIMD_FLOAT32_C(  -152.00), EASYSIMD_FLOAT32_C(  -486.00), EASYSIMD_FLOAT32_C( -1478.00),
        EASYSIMD_FLOAT32_C(   989.00), EASYSIMD_FLOAT32_C(  -159.00), EASYSIMD_FLOAT32_C(   898.00), EASYSIMD_FLOAT32_C(    82.00),
        EASYSIMD_FLOAT32_C(  1146.00), EASYSIMD_FLOAT32_C( -1452.00), EASYSIMD_FLOAT32_C(   259.00), EASYSIMD_FLOAT32_C(  1049.00) },
      { EASYSIMD_FLOAT32_C(    -5.00), EASYSIMD_FLOAT32_C(  -985.00), EASYSIMD_FLOAT32_C(  -332.00), EASYSIMD_FLOAT32_C(  -965.00),
        EASYSIMD_FLOAT32_C(   658.00), EASYSIMD_FLOAT32_C(  -152.00), EASYSIMD_FLOAT32_C(  -486.00), EASYSIMD_FLOAT32_C( -1478.00),
        EASYSIMD_FLOAT32_C(   988.00), EASYSIMD_FLOAT32_C(  -160.00), EASYSIMD_FLOAT32_C(   897.00), EASYSIMD_FLOAT32_C(    82.00),
        EASYSIMD_FLOAT32_C(  1146.00), EASYSIMD_FLOAT32_C( -1453.00), EASYSIMD_FLOAT32_C(   258.00), EASYSIMD_FLOAT32_C(  1049.00) },
      { EASYSIMD_FLOAT32_C(    -4.00), EASYSIMD_FLOAT32_C(  -984.00), EASYSIMD_FLOAT32_C(  -331.00), EASYSIMD_FLOAT32_C(  -964.00),
        EASYSIMD_FLOAT32_C(   659.00), EASYSIMD_FLOAT32_C(  -151.00), EASYSIMD_FLOAT32_C(  -485.00), EASYSIMD_FLOAT32_C( -1477.00),
        EASYSIMD_FLOAT32_C(   989.00), EASYSIMD_FLOAT32_C(  -159.00), EASYSIMD_FLOAT32_C(   898.00), EASYSIMD_FLOAT32_C(    83.00),
        EASYSIMD_FLOAT32_C(  1147.00), EASYSIMD_FLOAT32_C( -1452.00), EASYSIMD_FLOAT32_C(   259.00), EASYSIMD_FLOAT32_C(  1050.00) },
      { EASYSIMD_FLOAT32_C(    -4.00), EASYSIMD_FLOAT32_C(  -984.00), EASYSIMD_FLOAT32_C(  -331.00), EASYSIMD_FLOAT32_C(  -964.00),
        EASYSIMD_FLOAT32_C(   658.00), EASYSIMD_FLOAT32_C(  -151.00), EASYSIMD_FLOAT32_C(  -485.00), EASYSIMD_FLOAT32_C( -1477.00),
        EASYSIMD_FLOAT32_C(   988.00), EASYSIMD_FLOAT32_C(  -159.00), EASYSIMD_FLOAT32_C(   897.00), EASYSIMD_FLOAT32_C(    82.00),
        EASYSIMD_FLOAT32_C(  1146.00), EASYSIMD_FLOAT32_C( -1452.00), EASYSIMD_FLOAT32_C(   258.00), EASYSIMD_FLOAT32_C(  1049.00) },
      { EASYSIMD_FLOAT32_C(    -5.00), EASYSIMD_FLOAT32_C(  -984.00), EASYSIMD_FLOAT32_C(  -332.00), EASYSIMD_FLOAT32_C(  -964.00),
        EASYSIMD_FLOAT32_C(   659.00), EASYSIMD_FLOAT32_C(  -152.00), EASYSIMD_FLOAT32_C(  -486.00), EASYSIMD_FLOAT32_C( -1478.00),
        EASYSIMD_FLOAT32_C(   989.00), EASYSIMD_FLOAT32_C(  -159.00), EASYSIMD_FLOAT32_C(   898.00), EASYSIMD_FLOAT32_C(    82.00),
        EASYSIMD_FLOAT32_C(  1146.00), EASYSIMD_FLOAT32_C( -1452.00), EASYSIMD_FLOAT32_C(   259.00), EASYSIMD_FLOAT32_C(  1049.00) } },
    { { EASYSIMD_FLOAT32_C(   240.75), EASYSIMD_FLOAT32_C(   738.02), EASYSIMD_FLOAT32_C(  -816.27), EASYSIMD_FLOAT32_C(  -743.77),
        EASYSIMD_FLOAT32_C(  -300.69), EASYSIMD_FLOAT32_C(  -160.86), EASYSIMD_FLOAT32_C(  -234.53), EASYSIMD_FLOAT32_C(   997.34),
        EASYSIMD_FLOAT32_C(   881.68), EASYSIMD_FLOAT32_C(  -986.25), EASYSIMD_FLOAT32_C(   740.09), EASYSIMD_FLOAT32_C(   266.57),
        EASYSIMD_FLOAT32_C(  -892.09), EASYSIMD_FLOAT32_C(  -574.02), EASYSIMD_FLOAT32_C(  -229.48), EASYSIMD_FLOAT32_C(  -680.84) },
      { EASYSIMD_FLOAT32_C(  -345.88), EASYSIMD_FLOAT32_C(  -726.68), EASYSIMD_FLOAT32_C(  -283.91), EASYSIMD_FLOAT32_C(  -702.81),
        EASYSIMD_FLOAT32_C(   160.07), EASYSIMD_FLOAT32_C(   574.93), EASYSIMD_FLOAT32_C(  -689.87), EASYSIMD_FLOAT32_C(  -149.25),
        EASYSIMD_FLOAT32_C(  -627.06), EASYSIMD_FLOAT32_C(   959.56), EASYSIMD_FLOAT32_C(   190.35), EASYSIMD_FLOAT32_C(  -865.83),
        EASYSIMD_FLOAT32_C(  -587.10), EASYSIMD_FLOAT32_C(  -236.89), EASYSIMD_FLOAT32_C(   679.24), EASYSIMD_FLOAT32_C(   653.66) },
      { EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C(    11.00), EASYSIMD_FLOAT32_C( -1100.00), EASYSIMD_FLOAT32_C( -1447.00),
        EASYSIMD_FLOAT32_C(  -141.00), EASYSIMD_FLOAT32_C(   414.00), EASYSIMD_FLOAT32_C(  -924.00), EASYSIMD_FLOAT32_C(   848.00),
        EASYSIMD_FLOAT32_C(   255.00), EASYSIMD_FLOAT32_C(   -27.00), EASYSIMD_FLOAT32_C(   930.00), EASYSIMD_FLOAT32_C(  -599.00),
        EASYSIMD_FLOAT32_C( -1479.00), EASYSIMD_FLOAT32_C(  -811.00), EASYSIMD_FLOAT32_C(   450.00), EASYSIMD_FLOAT32_C(   -27.00) },
      { EASYSIMD_FLOAT32_C(  -106.00), EASYSIMD_FLOAT32_C(    11.00), EASYSIMD_FLOAT32_C( -1101.00), EASYSIMD_FLOAT32_C( -1447.00),
        EASYSIMD_FLOAT32_C(  -141.00), EASYSIMD_FLOAT32_C(   414.00), EASYSIMD_FLOAT32_C(  -925.00), EASYSIMD_FLOAT32_C(   848.00),
        EASYSIMD_FLOAT32_C(   254.00), EASYSIMD_FLOAT32_C(   -27.00), EASYSIMD_FLOAT32_C(   930.00), EASYSIMD_FLOAT32_C(  -600.00),
        EASYSIMD_FLOAT32_C( -1480.00), EASYSIMD_FLOAT32_C(  -811.00), EASYSIMD_FLOAT32_C(   449.00), EASYSIMD_FLOAT32_C(   -28.00) },
      { EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C(    12.00), EASYSIMD_FLOAT32_C( -1100.00), EASYSIMD_FLOAT32_C( -1446.00),
        EASYSIMD_FLOAT32_C(  -140.00), EASYSIMD_FLOAT32_C(   415.00), EASYSIMD_FLOAT32_C(  -924.00), EASYSIMD_FLOAT32_C(   849.00),
        EASYSIMD_FLOAT32_C(   255.00), EASYSIMD_FLOAT32_C(   -26.00), EASYSIMD_FLOAT32_C(   931.00), EASYSIMD_FLOAT32_C(  -599.00),
        EASYSIMD_FLOAT32_C( -1479.00), EASYSIMD_FLOAT32_C(  -810.00), EASYSIMD_FLOAT32_C(   450.00), EASYSIMD_FLOAT32_C(   -27.00) },
      { EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C(    11.00), EASYSIMD_FLOAT32_C( -1100.00), EASYSIMD_FLOAT32_C( -1446.00),
        EASYSIMD_FLOAT32_C(  -140.00), EASYSIMD_FLOAT32_C(   414.00), EASYSIMD_FLOAT32_C(  -924.00), EASYSIMD_FLOAT32_C(   848.00),
        EASYSIMD_FLOAT32_C(   254.00), EASYSIMD_FLOAT32_C(   -26.00), EASYSIMD_FLOAT32_C(   930.00), EASYSIMD_FLOAT32_C(  -599.00),
        EASYSIMD_FLOAT32_C( -1479.00), EASYSIMD_FLOAT32_C(  -810.00), EASYSIMD_FLOAT32_C(   449.00), EASYSIMD_FLOAT32_C(   -27.00) },
      { EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C(    11.00), EASYSIMD_FLOAT32_C( -1100.00), EASYSIMD_FLOAT32_C( -1447.00),
        EASYSIMD_FLOAT32_C(  -141.00), EASYSIMD_FLOAT32_C(   414.00), EASYSIMD_FLOAT32_C(  -924.00), EASYSIMD_FLOAT32_C(   848.00),
        EASYSIMD_FLOAT32_C(   255.00), EASYSIMD_FLOAT32_C(   -27.00), EASYSIMD_FLOAT32_C(   930.00), EASYSIMD_FLOAT32_C(  -599.00),
        EASYSIMD_FLOAT32_C( -1479.00), EASYSIMD_FLOAT32_C(  -811.00), EASYSIMD_FLOAT32_C(   450.00), EASYSIMD_FLOAT32_C(   -27.00) } },
    { { EASYSIMD_FLOAT32_C(  -498.87), EASYSIMD_FLOAT32_C(   862.97), EASYSIMD_FLOAT32_C(   909.89), EASYSIMD_FLOAT32_C(   200.43),
        EASYSIMD_FLOAT32_C(  -297.89), EASYSIMD_FLOAT32_C(  -324.64), EASYSIMD_FLOAT32_C(   197.77), EASYSIMD_FLOAT32_C(  -416.21),
        EASYSIMD_FLOAT32_C(  -310.90), EASYSIMD_FLOAT32_C(   -62.14), EASYSIMD_FLOAT32_C(   850.36), EASYSIMD_FLOAT32_C(  -202.99),
        EASYSIMD_FLOAT32_C(   363.84), EASYSIMD_FLOAT32_C(  -379.12), EASYSIMD_FLOAT32_C(   116.18), EASYSIMD_FLOAT32_C(  -982.05) },
      { EASYSIMD_FLOAT32_C(  -105.80), EASYSIMD_FLOAT32_C(   832.27), EASYSIMD_FLOAT32_C(  -684.85), EASYSIMD_FLOAT32_C(  -945.73),
        EASYSIMD_FLOAT32_C(   407.21), EASYSIMD_FLOAT32_C(  -374.72), EASYSIMD_FLOAT32_C(   -94.97), EASYSIMD_FLOAT32_C(   780.14),
        EASYSIMD_FLOAT32_C(  -415.16), EASYSIMD_FLOAT32_C(  -904.63), EASYSIMD_FLOAT32_C(   914.31), EASYSIMD_FLOAT32_C(    -2.26),
        EASYSIMD_FLOAT32_C(  -141.52), EASYSIMD_FLOAT32_C(   593.55), EASYSIMD_FLOAT32_C(  -348.60), EASYSIMD_FLOAT32_C(   359.61) },
      { EASYSIMD_FLOAT32_C(  -605.00), EASYSIMD_FLOAT32_C(  1695.00), EASYSIMD_FLOAT32_C(   225.00), EASYSIMD_FLOAT32_C(  -745.00),
        EASYSIMD_FLOAT32_C(   109.00), EASYSIMD_FLOAT32_C(  -699.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(   364.00),
        EASYSIMD_FLOAT32_C(  -726.00), EASYSIMD_FLOAT32_C(  -967.00), EASYSIMD_FLOAT32_C(  1765.00), EASYSIMD_FLOAT32_C(  -205.00),
        EASYSIMD_FLOAT32_C(   222.00), EASYSIMD_FLOAT32_C(   214.00), EASYSIMD_FLOAT32_C(  -232.00), EASYSIMD_FLOAT32_C(  -622.00) },
      { EASYSIMD_FLOAT32_C(  -605.00), EASYSIMD_FLOAT32_C(  1695.00), EASYSIMD_FLOAT32_C(   225.00), EASYSIMD_FLOAT32_C(  -746.00),
        EASYSIMD_FLOAT32_C(   109.00), EASYSIMD_FLOAT32_C(  -700.00), EASYSIMD_FLOAT32_C(   102.00), EASYSIMD_FLOAT32_C(   363.00),
        EASYSIMD_FLOAT32_C(  -727.00), EASYSIMD_FLOAT32_C(  -967.00), EASYSIMD_FLOAT32_C(  1764.00), EASYSIMD_FLOAT32_C(  -206.00),
        EASYSIMD_FLOAT32_C(   222.00), EASYSIMD_FLOAT32_C(   214.00), EASYSIMD_FLOAT32_C(  -233.00), EASYSIMD_FLOAT32_C(  -623.00) },
      { EASYSIMD_FLOAT32_C(  -604.00), EASYSIMD_FLOAT32_C(  1696.00), EASYSIMD_FLOAT32_C(   226.00), EASYSIMD_FLOAT32_C(  -745.00),
        EASYSIMD_FLOAT32_C(   110.00), EASYSIMD_FLOAT32_C(  -699.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(   364.00),
        EASYSIMD_FLOAT32_C(  -726.00), EASYSIMD_FLOAT32_C(  -966.00), EASYSIMD_FLOAT32_C(  1765.00), EASYSIMD_FLOAT32_C(  -205.00),
        EASYSIMD_FLOAT32_C(   223.00), EASYSIMD_FLOAT32_C(   215.00), EASYSIMD_FLOAT32_C(  -232.00), EASYSIMD_FLOAT32_C(  -622.00) },
      { EASYSIMD_FLOAT32_C(  -604.00), EASYSIMD_FLOAT32_C(  1695.00), EASYSIMD_FLOAT32_C(   225.00), EASYSIMD_FLOAT32_C(  -745.00),
        EASYSIMD_FLOAT32_C(   109.00), EASYSIMD_FLOAT32_C(  -699.00), EASYSIMD_FLOAT32_C(   102.00), EASYSIMD_FLOAT32_C(   363.00),
        EASYSIMD_FLOAT32_C(  -726.00), EASYSIMD_FLOAT32_C(  -966.00), EASYSIMD_FLOAT32_C(  1764.00), EASYSIMD_FLOAT32_C(  -205.00),
        EASYSIMD_FLOAT32_C(   222.00), EASYSIMD_FLOAT32_C(   214.00), EASYSIMD_FLOAT32_C(  -232.00), EASYSIMD_FLOAT32_C(  -622.00) },
      { EASYSIMD_FLOAT32_C(  -605.00), EASYSIMD_FLOAT32_C(  1695.00), EASYSIMD_FLOAT32_C(   225.00), EASYSIMD_FLOAT32_C(  -745.00),
        EASYSIMD_FLOAT32_C(   109.00), EASYSIMD_FLOAT32_C(  -699.00), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(   364.00),
        EASYSIMD_FLOAT32_C(  -726.00), EASYSIMD_FLOAT32_C(  -967.00), EASYSIMD_FLOAT32_C(  1765.00), EASYSIMD_FLOAT32_C(  -205.00),
        EASYSIMD_FLOAT32_C(   222.00), EASYSIMD_FLOAT32_C(   214.00), EASYSIMD_FLOAT32_C(  -232.00), EASYSIMD_FLOAT32_C(  -622.00) } }
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

    r = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512_close(r, nearest_inf, 1);

    r = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512_close(r, neg_inf, 1);

    r = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512_close(r, pos_inf, 1);

    r = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
       r = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_add_round_ps");
    easysimd_assert_m512_close(r, direction, 1);
  }

  return 0;

  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512 nearest_inf = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512 neg_inf = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512 pos_inf = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512 zero = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512 direction = easysimd_mm512_add_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

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
test_easysimd_mm512_mask_add_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    easysimd_float32 src[16];
    easysimd__mmask16 k;
    easysimd_float32 a[16];
    easysimd_float32 b[16];
    easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   460.46), EASYSIMD_FLOAT32_C(  -331.63), EASYSIMD_FLOAT32_C(   681.04), EASYSIMD_FLOAT32_C(  -156.99),
        EASYSIMD_FLOAT32_C(  -543.60), EASYSIMD_FLOAT32_C(    94.06), EASYSIMD_FLOAT32_C(   923.51), EASYSIMD_FLOAT32_C(  -881.03),
        EASYSIMD_FLOAT32_C(  -751.28), EASYSIMD_FLOAT32_C(  -565.86), EASYSIMD_FLOAT32_C(  -825.81), EASYSIMD_FLOAT32_C(   899.74),
        EASYSIMD_FLOAT32_C(   180.53), EASYSIMD_FLOAT32_C(  -308.94), EASYSIMD_FLOAT32_C(   105.08), EASYSIMD_FLOAT32_C(  -812.17) },
      UINT16_C(46254),
      { EASYSIMD_FLOAT32_C(   159.98), EASYSIMD_FLOAT32_C(   930.16), EASYSIMD_FLOAT32_C(  -647.50), EASYSIMD_FLOAT32_C(  -273.39),
        EASYSIMD_FLOAT32_C(  -580.12), EASYSIMD_FLOAT32_C(  -662.69), EASYSIMD_FLOAT32_C(  -636.46), EASYSIMD_FLOAT32_C(   880.33),
        EASYSIMD_FLOAT32_C(     5.69), EASYSIMD_FLOAT32_C(  -955.42), EASYSIMD_FLOAT32_C(  -276.66), EASYSIMD_FLOAT32_C(   462.09),
        EASYSIMD_FLOAT32_C(   138.64), EASYSIMD_FLOAT32_C(  -353.15), EASYSIMD_FLOAT32_C(   581.06), EASYSIMD_FLOAT32_C(   387.36) },
      { EASYSIMD_FLOAT32_C(    80.99), EASYSIMD_FLOAT32_C(   755.25), EASYSIMD_FLOAT32_C(   287.10), EASYSIMD_FLOAT32_C(  -738.48),
        EASYSIMD_FLOAT32_C(  -553.70), EASYSIMD_FLOAT32_C(  -607.83), EASYSIMD_FLOAT32_C(  -550.66), EASYSIMD_FLOAT32_C(   141.56),
        EASYSIMD_FLOAT32_C(  -486.72), EASYSIMD_FLOAT32_C(   158.46), EASYSIMD_FLOAT32_C(    11.82), EASYSIMD_FLOAT32_C(  -371.24),
        EASYSIMD_FLOAT32_C(   774.24), EASYSIMD_FLOAT32_C(  -643.60), EASYSIMD_FLOAT32_C(   412.35), EASYSIMD_FLOAT32_C(   -65.78) },
      { EASYSIMD_FLOAT32_C(   460.46), EASYSIMD_FLOAT32_C(  1685.41), EASYSIMD_FLOAT32_C(  -360.40), EASYSIMD_FLOAT32_C( -1011.87),
        EASYSIMD_FLOAT32_C(  -543.60), EASYSIMD_FLOAT32_C( -1270.51), EASYSIMD_FLOAT32_C(   923.51), EASYSIMD_FLOAT32_C(  1021.89),
        EASYSIMD_FLOAT32_C(  -751.28), EASYSIMD_FLOAT32_C(  -565.86), EASYSIMD_FLOAT32_C(  -264.84), EASYSIMD_FLOAT32_C(   899.74),
        EASYSIMD_FLOAT32_C(   912.88), EASYSIMD_FLOAT32_C(  -996.76), EASYSIMD_FLOAT32_C(   105.08), EASYSIMD_FLOAT32_C(   321.58) } },
    { { EASYSIMD_FLOAT32_C(  -713.44), EASYSIMD_FLOAT32_C(   764.86), EASYSIMD_FLOAT32_C(   660.83), EASYSIMD_FLOAT32_C(  -293.56),
        EASYSIMD_FLOAT32_C(  -897.83), EASYSIMD_FLOAT32_C(  -975.63), EASYSIMD_FLOAT32_C(  -413.23), EASYSIMD_FLOAT32_C(   107.86),
        EASYSIMD_FLOAT32_C(  -931.06), EASYSIMD_FLOAT32_C(   310.12), EASYSIMD_FLOAT32_C(  -430.05), EASYSIMD_FLOAT32_C(   207.58),
        EASYSIMD_FLOAT32_C(   956.96), EASYSIMD_FLOAT32_C(  -848.99), EASYSIMD_FLOAT32_C(  -405.06), EASYSIMD_FLOAT32_C(    37.95) },
      UINT16_C(56892),
      { EASYSIMD_FLOAT32_C(   -92.73), EASYSIMD_FLOAT32_C(  -494.07), EASYSIMD_FLOAT32_C(   416.25), EASYSIMD_FLOAT32_C(  -318.49),
        EASYSIMD_FLOAT32_C(  -137.67), EASYSIMD_FLOAT32_C(  -171.40), EASYSIMD_FLOAT32_C(   615.73), EASYSIMD_FLOAT32_C(   148.89),
        EASYSIMD_FLOAT32_C(  -406.54), EASYSIMD_FLOAT32_C(   276.56), EASYSIMD_FLOAT32_C(   855.33), EASYSIMD_FLOAT32_C(  -304.37),
        EASYSIMD_FLOAT32_C(   300.92), EASYSIMD_FLOAT32_C(  -557.90), EASYSIMD_FLOAT32_C(   803.48), EASYSIMD_FLOAT32_C(   369.86) },
      { EASYSIMD_FLOAT32_C(   752.21), EASYSIMD_FLOAT32_C(  -626.57), EASYSIMD_FLOAT32_C(  -422.55), EASYSIMD_FLOAT32_C(   709.18),
        EASYSIMD_FLOAT32_C(  -475.56), EASYSIMD_FLOAT32_C(   172.39), EASYSIMD_FLOAT32_C(  -252.87), EASYSIMD_FLOAT32_C(  -569.31),
        EASYSIMD_FLOAT32_C(    54.43), EASYSIMD_FLOAT32_C(  -953.40), EASYSIMD_FLOAT32_C(  -216.76), EASYSIMD_FLOAT32_C(   328.64),
        EASYSIMD_FLOAT32_C(   795.42), EASYSIMD_FLOAT32_C(  -722.64), EASYSIMD_FLOAT32_C(  -883.86), EASYSIMD_FLOAT32_C(  -297.31) },
      { EASYSIMD_FLOAT32_C(  -713.44), EASYSIMD_FLOAT32_C(   764.86), EASYSIMD_FLOAT32_C(    -6.30), EASYSIMD_FLOAT32_C(   390.69),
        EASYSIMD_FLOAT32_C(  -613.23), EASYSIMD_FLOAT32_C(     0.99), EASYSIMD_FLOAT32_C(  -413.23), EASYSIMD_FLOAT32_C(   107.86),
        EASYSIMD_FLOAT32_C(  -931.06), EASYSIMD_FLOAT32_C(  -676.84), EASYSIMD_FLOAT32_C(   638.57), EASYSIMD_FLOAT32_C(    24.27),
        EASYSIMD_FLOAT32_C(  1096.34), EASYSIMD_FLOAT32_C(  -848.99), EASYSIMD_FLOAT32_C(   -80.38), EASYSIMD_FLOAT32_C(    72.56) } },
    { { EASYSIMD_FLOAT32_C(  -216.71), EASYSIMD_FLOAT32_C(   532.39), EASYSIMD_FLOAT32_C(   384.21), EASYSIMD_FLOAT32_C(   645.62),
        EASYSIMD_FLOAT32_C(  -639.01), EASYSIMD_FLOAT32_C(    -0.07), EASYSIMD_FLOAT32_C(  -205.49), EASYSIMD_FLOAT32_C(   -45.55),
        EASYSIMD_FLOAT32_C(  -723.51), EASYSIMD_FLOAT32_C(  -350.17), EASYSIMD_FLOAT32_C(   650.08), EASYSIMD_FLOAT32_C(   577.41),
        EASYSIMD_FLOAT32_C(    91.93), EASYSIMD_FLOAT32_C(   453.57), EASYSIMD_FLOAT32_C(   -52.73), EASYSIMD_FLOAT32_C(  -155.86) },
      UINT16_C( 2131),
      { EASYSIMD_FLOAT32_C(   347.05), EASYSIMD_FLOAT32_C(   565.37), EASYSIMD_FLOAT32_C(    80.17), EASYSIMD_FLOAT32_C(   142.47),
        EASYSIMD_FLOAT32_C(   842.73), EASYSIMD_FLOAT32_C(   196.31), EASYSIMD_FLOAT32_C(   845.17), EASYSIMD_FLOAT32_C(  -373.98),
        EASYSIMD_FLOAT32_C(  -271.30), EASYSIMD_FLOAT32_C(   229.37), EASYSIMD_FLOAT32_C(  -728.36), EASYSIMD_FLOAT32_C(    89.69),
        EASYSIMD_FLOAT32_C(  -770.69), EASYSIMD_FLOAT32_C(    66.14), EASYSIMD_FLOAT32_C(  -955.85), EASYSIMD_FLOAT32_C(  -494.20) },
      { EASYSIMD_FLOAT32_C(   715.97), EASYSIMD_FLOAT32_C(   694.23), EASYSIMD_FLOAT32_C(  -916.79), EASYSIMD_FLOAT32_C(  -192.10),
        EASYSIMD_FLOAT32_C(   147.79), EASYSIMD_FLOAT32_C(    30.48), EASYSIMD_FLOAT32_C(   652.05), EASYSIMD_FLOAT32_C(   -25.21),
        EASYSIMD_FLOAT32_C(  -444.80), EASYSIMD_FLOAT32_C(  -794.64), EASYSIMD_FLOAT32_C(   326.23), EASYSIMD_FLOAT32_C(   252.31),
        EASYSIMD_FLOAT32_C(   505.81), EASYSIMD_FLOAT32_C(  -891.64), EASYSIMD_FLOAT32_C(     3.84), EASYSIMD_FLOAT32_C(  -147.13) },
      { EASYSIMD_FLOAT32_C(  1063.03), EASYSIMD_FLOAT32_C(  1259.60), EASYSIMD_FLOAT32_C(   384.21), EASYSIMD_FLOAT32_C(   645.62),
        EASYSIMD_FLOAT32_C(   990.52), EASYSIMD_FLOAT32_C(    -0.07), EASYSIMD_FLOAT32_C(  1497.21), EASYSIMD_FLOAT32_C(   -45.55),
        EASYSIMD_FLOAT32_C(  -723.51), EASYSIMD_FLOAT32_C(  -350.17), EASYSIMD_FLOAT32_C(   650.08), EASYSIMD_FLOAT32_C(   342.00),
        EASYSIMD_FLOAT32_C(    91.93), EASYSIMD_FLOAT32_C(   453.57), EASYSIMD_FLOAT32_C(   -52.73), EASYSIMD_FLOAT32_C(  -155.86) } },
    { { EASYSIMD_FLOAT32_C(   673.73), EASYSIMD_FLOAT32_C(  -915.98), EASYSIMD_FLOAT32_C(   995.34), EASYSIMD_FLOAT32_C(   516.46),
        EASYSIMD_FLOAT32_C(   280.33), EASYSIMD_FLOAT32_C(   840.51), EASYSIMD_FLOAT32_C(  -857.52), EASYSIMD_FLOAT32_C(  -990.97),
        EASYSIMD_FLOAT32_C(    69.88), EASYSIMD_FLOAT32_C(  -585.88), EASYSIMD_FLOAT32_C(    98.72), EASYSIMD_FLOAT32_C(   299.19),
        EASYSIMD_FLOAT32_C(   480.27), EASYSIMD_FLOAT32_C(   142.87), EASYSIMD_FLOAT32_C(   804.99), EASYSIMD_FLOAT32_C(   196.24) },
      UINT16_C(47953),
      { EASYSIMD_FLOAT32_C(   861.56), EASYSIMD_FLOAT32_C(  -714.09), EASYSIMD_FLOAT32_C(  -273.80), EASYSIMD_FLOAT32_C(   367.37),
        EASYSIMD_FLOAT32_C(  -605.73), EASYSIMD_FLOAT32_C(   730.04), EASYSIMD_FLOAT32_C(  -779.76), EASYSIMD_FLOAT32_C(  -932.00),
        EASYSIMD_FLOAT32_C(   814.06), EASYSIMD_FLOAT32_C(  -784.42), EASYSIMD_FLOAT32_C(   584.46), EASYSIMD_FLOAT32_C(    94.39),
        EASYSIMD_FLOAT32_C(  -943.91), EASYSIMD_FLOAT32_C(   726.95), EASYSIMD_FLOAT32_C(   103.42), EASYSIMD_FLOAT32_C(   125.97) },
      { EASYSIMD_FLOAT32_C(  -858.93), EASYSIMD_FLOAT32_C(  -797.86), EASYSIMD_FLOAT32_C(  -574.84), EASYSIMD_FLOAT32_C(   621.33),
        EASYSIMD_FLOAT32_C(   345.01), EASYSIMD_FLOAT32_C(  -769.85), EASYSIMD_FLOAT32_C(  -182.42), EASYSIMD_FLOAT32_C(  -817.89),
        EASYSIMD_FLOAT32_C(  -881.65), EASYSIMD_FLOAT32_C(  -178.28), EASYSIMD_FLOAT32_C(  -833.00), EASYSIMD_FLOAT32_C(    37.03),
        EASYSIMD_FLOAT32_C(  -522.09), EASYSIMD_FLOAT32_C(   126.68), EASYSIMD_FLOAT32_C(  -489.08), EASYSIMD_FLOAT32_C(  -660.53) },
      { EASYSIMD_FLOAT32_C(     2.62), EASYSIMD_FLOAT32_C(  -915.98), EASYSIMD_FLOAT32_C(   995.34), EASYSIMD_FLOAT32_C(   516.46),
        EASYSIMD_FLOAT32_C(  -260.72), EASYSIMD_FLOAT32_C(   840.51), EASYSIMD_FLOAT32_C(  -962.19), EASYSIMD_FLOAT32_C(  -990.97),
        EASYSIMD_FLOAT32_C(   -67.59), EASYSIMD_FLOAT32_C(  -962.70), EASYSIMD_FLOAT32_C(    98.72), EASYSIMD_FLOAT32_C(   131.42),
        EASYSIMD_FLOAT32_C( -1466.00), EASYSIMD_FLOAT32_C(   853.63), EASYSIMD_FLOAT32_C(   804.99), EASYSIMD_FLOAT32_C(  -534.56) } },
    { { EASYSIMD_FLOAT32_C(   412.59), EASYSIMD_FLOAT32_C(   237.12), EASYSIMD_FLOAT32_C(   706.84), EASYSIMD_FLOAT32_C(   806.86),
        EASYSIMD_FLOAT32_C(   -32.84), EASYSIMD_FLOAT32_C(   927.07), EASYSIMD_FLOAT32_C(   874.86), EASYSIMD_FLOAT32_C(  -218.78),
        EASYSIMD_FLOAT32_C(  -857.35), EASYSIMD_FLOAT32_C(   459.32), EASYSIMD_FLOAT32_C(   875.61), EASYSIMD_FLOAT32_C(  -801.26),
        EASYSIMD_FLOAT32_C(   186.27), EASYSIMD_FLOAT32_C(   -20.97), EASYSIMD_FLOAT32_C(   324.71), EASYSIMD_FLOAT32_C(   327.34) },
      UINT16_C(16785),
      { EASYSIMD_FLOAT32_C(  -412.03), EASYSIMD_FLOAT32_C(  -124.71), EASYSIMD_FLOAT32_C(   135.41), EASYSIMD_FLOAT32_C(    65.88),
        EASYSIMD_FLOAT32_C(  -998.03), EASYSIMD_FLOAT32_C(   646.33), EASYSIMD_FLOAT32_C(   405.35), EASYSIMD_FLOAT32_C(   414.56),
        EASYSIMD_FLOAT32_C(  -116.55), EASYSIMD_FLOAT32_C(   112.18), EASYSIMD_FLOAT32_C(   221.42), EASYSIMD_FLOAT32_C(   850.61),
        EASYSIMD_FLOAT32_C(    39.26), EASYSIMD_FLOAT32_C(    96.28), EASYSIMD_FLOAT32_C(  -368.17), EASYSIMD_FLOAT32_C(   181.91) },
      { EASYSIMD_FLOAT32_C(  -444.40), EASYSIMD_FLOAT32_C(  -492.56), EASYSIMD_FLOAT32_C(   380.65), EASYSIMD_FLOAT32_C(   741.87),
        EASYSIMD_FLOAT32_C(   486.46), EASYSIMD_FLOAT32_C(  -294.64), EASYSIMD_FLOAT32_C(    69.20), EASYSIMD_FLOAT32_C(  -332.37),
        EASYSIMD_FLOAT32_C(  -544.77), EASYSIMD_FLOAT32_C(  -982.12), EASYSIMD_FLOAT32_C(   193.82), EASYSIMD_FLOAT32_C(  -564.75),
        EASYSIMD_FLOAT32_C(   784.12), EASYSIMD_FLOAT32_C(   902.11), EASYSIMD_FLOAT32_C(  -466.37), EASYSIMD_FLOAT32_C(  -627.91) },
      { EASYSIMD_FLOAT32_C(  -856.43), EASYSIMD_FLOAT32_C(   237.12), EASYSIMD_FLOAT32_C(   706.84), EASYSIMD_FLOAT32_C(   806.86),
        EASYSIMD_FLOAT32_C(  -511.57), EASYSIMD_FLOAT32_C(   927.07), EASYSIMD_FLOAT32_C(   874.86), EASYSIMD_FLOAT32_C(    82.19),
        EASYSIMD_FLOAT32_C(  -661.32), EASYSIMD_FLOAT32_C(   459.32), EASYSIMD_FLOAT32_C(   875.61), EASYSIMD_FLOAT32_C(  -801.26),
        EASYSIMD_FLOAT32_C(   186.27), EASYSIMD_FLOAT32_C(   -20.97), EASYSIMD_FLOAT32_C(  -834.55), EASYSIMD_FLOAT32_C(   327.34) } },
    { { EASYSIMD_FLOAT32_C(  -222.60), EASYSIMD_FLOAT32_C(   669.04), EASYSIMD_FLOAT32_C(   437.97), EASYSIMD_FLOAT32_C(  -220.63),
        EASYSIMD_FLOAT32_C(   315.37), EASYSIMD_FLOAT32_C(  -156.68), EASYSIMD_FLOAT32_C(  -806.07), EASYSIMD_FLOAT32_C(  -801.18),
        EASYSIMD_FLOAT32_C(   955.50), EASYSIMD_FLOAT32_C(   415.35), EASYSIMD_FLOAT32_C(  -950.57), EASYSIMD_FLOAT32_C(    -5.24),
        EASYSIMD_FLOAT32_C(  -488.38), EASYSIMD_FLOAT32_C(  -318.75), EASYSIMD_FLOAT32_C(  -823.33), EASYSIMD_FLOAT32_C(    67.22) },
      UINT16_C(17154),
      { EASYSIMD_FLOAT32_C(   896.17), EASYSIMD_FLOAT32_C(  -463.40), EASYSIMD_FLOAT32_C(   153.15), EASYSIMD_FLOAT32_C(   680.29),
        EASYSIMD_FLOAT32_C(  -561.29), EASYSIMD_FLOAT32_C(   686.78), EASYSIMD_FLOAT32_C(  -947.62), EASYSIMD_FLOAT32_C(   216.11),
        EASYSIMD_FLOAT32_C(   355.82), EASYSIMD_FLOAT32_C(   490.35), EASYSIMD_FLOAT32_C(   995.48), EASYSIMD_FLOAT32_C(  -328.82),
        EASYSIMD_FLOAT32_C(  -666.33), EASYSIMD_FLOAT32_C(  -810.59), EASYSIMD_FLOAT32_C(  -130.00), EASYSIMD_FLOAT32_C(  -710.83) },
      { EASYSIMD_FLOAT32_C(   604.75), EASYSIMD_FLOAT32_C(   -80.58), EASYSIMD_FLOAT32_C(   283.92), EASYSIMD_FLOAT32_C(  -883.63),
        EASYSIMD_FLOAT32_C(   600.68), EASYSIMD_FLOAT32_C(   460.59), EASYSIMD_FLOAT32_C(   183.59), EASYSIMD_FLOAT32_C(  -210.63),
        EASYSIMD_FLOAT32_C(    17.91), EASYSIMD_FLOAT32_C(   992.68), EASYSIMD_FLOAT32_C(   464.52), EASYSIMD_FLOAT32_C(   280.58),
        EASYSIMD_FLOAT32_C(   870.97), EASYSIMD_FLOAT32_C(  -192.70), EASYSIMD_FLOAT32_C(   998.48), EASYSIMD_FLOAT32_C(   767.14) },
      { EASYSIMD_FLOAT32_C(  -222.60), EASYSIMD_FLOAT32_C(  -543.97), EASYSIMD_FLOAT32_C(   437.97), EASYSIMD_FLOAT32_C(  -220.63),
        EASYSIMD_FLOAT32_C(   315.37), EASYSIMD_FLOAT32_C(  -156.68), EASYSIMD_FLOAT32_C(  -806.07), EASYSIMD_FLOAT32_C(  -801.18),
        EASYSIMD_FLOAT32_C(   373.72), EASYSIMD_FLOAT32_C(  1483.03), EASYSIMD_FLOAT32_C(  -950.57), EASYSIMD_FLOAT32_C(    -5.24),
        EASYSIMD_FLOAT32_C(  -488.38), EASYSIMD_FLOAT32_C(  -318.75), EASYSIMD_FLOAT32_C(   868.48), EASYSIMD_FLOAT32_C(    67.22) } },
    { { EASYSIMD_FLOAT32_C(   343.91), EASYSIMD_FLOAT32_C(   151.64), EASYSIMD_FLOAT32_C(   447.43), EASYSIMD_FLOAT32_C(   782.62),
        EASYSIMD_FLOAT32_C(  -161.58), EASYSIMD_FLOAT32_C(   499.81), EASYSIMD_FLOAT32_C(    -1.27), EASYSIMD_FLOAT32_C(  -805.77),
        EASYSIMD_FLOAT32_C(    -9.84), EASYSIMD_FLOAT32_C(    -5.79), EASYSIMD_FLOAT32_C(  -134.58), EASYSIMD_FLOAT32_C(   323.82),
        EASYSIMD_FLOAT32_C(   183.61), EASYSIMD_FLOAT32_C(   735.41), EASYSIMD_FLOAT32_C(   612.99), EASYSIMD_FLOAT32_C(  -211.63) },
      UINT16_C(55098),
      { EASYSIMD_FLOAT32_C(  -918.99), EASYSIMD_FLOAT32_C(  -490.60), EASYSIMD_FLOAT32_C(  -344.01), EASYSIMD_FLOAT32_C(   951.99),
        EASYSIMD_FLOAT32_C(   316.70), EASYSIMD_FLOAT32_C(  -345.53), EASYSIMD_FLOAT32_C(   719.12), EASYSIMD_FLOAT32_C(  -339.39),
        EASYSIMD_FLOAT32_C(   806.11), EASYSIMD_FLOAT32_C(   166.55), EASYSIMD_FLOAT32_C(  -556.77), EASYSIMD_FLOAT32_C(  -355.47),
        EASYSIMD_FLOAT32_C(  -333.64), EASYSIMD_FLOAT32_C(   441.96), EASYSIMD_FLOAT32_C(  -161.24), EASYSIMD_FLOAT32_C(   656.52) },
      { EASYSIMD_FLOAT32_C(  -563.83), EASYSIMD_FLOAT32_C(   704.18), EASYSIMD_FLOAT32_C(   -19.66), EASYSIMD_FLOAT32_C(   619.78),
        EASYSIMD_FLOAT32_C(   439.59), EASYSIMD_FLOAT32_C(  -406.67), EASYSIMD_FLOAT32_C(  -591.85), EASYSIMD_FLOAT32_C(  -905.57),
        EASYSIMD_FLOAT32_C(   490.24), EASYSIMD_FLOAT32_C(   312.88), EASYSIMD_FLOAT32_C(  -650.06), EASYSIMD_FLOAT32_C(   847.74),
        EASYSIMD_FLOAT32_C(   401.22), EASYSIMD_FLOAT32_C(   394.82), EASYSIMD_FLOAT32_C(   223.15), EASYSIMD_FLOAT32_C(   482.23) },
      { EASYSIMD_FLOAT32_C(   343.91), EASYSIMD_FLOAT32_C(   213.57), EASYSIMD_FLOAT32_C(   447.43), EASYSIMD_FLOAT32_C(  1571.77),
        EASYSIMD_FLOAT32_C(   756.29), EASYSIMD_FLOAT32_C(  -752.20), EASYSIMD_FLOAT32_C(    -1.27), EASYSIMD_FLOAT32_C(  -805.77),
        EASYSIMD_FLOAT32_C(  1296.35), EASYSIMD_FLOAT32_C(   479.44), EASYSIMD_FLOAT32_C( -1206.83), EASYSIMD_FLOAT32_C(   323.82),
        EASYSIMD_FLOAT32_C(    67.58), EASYSIMD_FLOAT32_C(   735.41), EASYSIMD_FLOAT32_C(    61.91), EASYSIMD_FLOAT32_C(  1138.75) } },
    { { EASYSIMD_FLOAT32_C(   904.21), EASYSIMD_FLOAT32_C(   879.14), EASYSIMD_FLOAT32_C(   434.21), EASYSIMD_FLOAT32_C(   220.91),
        EASYSIMD_FLOAT32_C(  -466.39), EASYSIMD_FLOAT32_C(   153.34), EASYSIMD_FLOAT32_C(   881.52), EASYSIMD_FLOAT32_C(  -660.28),
        EASYSIMD_FLOAT32_C(  -680.11), EASYSIMD_FLOAT32_C(  -675.25), EASYSIMD_FLOAT32_C(   -15.75), EASYSIMD_FLOAT32_C(   -13.75),
        EASYSIMD_FLOAT32_C(   766.71), EASYSIMD_FLOAT32_C(   823.02), EASYSIMD_FLOAT32_C(  -357.23), EASYSIMD_FLOAT32_C(  -797.13) },
      UINT16_C(62059),
      { EASYSIMD_FLOAT32_C(   543.68), EASYSIMD_FLOAT32_C(   411.16), EASYSIMD_FLOAT32_C(   554.42), EASYSIMD_FLOAT32_C(   -55.10),
        EASYSIMD_FLOAT32_C(  -194.03), EASYSIMD_FLOAT32_C(  -222.43), EASYSIMD_FLOAT32_C(  -572.87), EASYSIMD_FLOAT32_C(  -289.81),
        EASYSIMD_FLOAT32_C(  -343.29), EASYSIMD_FLOAT32_C(   861.34), EASYSIMD_FLOAT32_C(   931.10), EASYSIMD_FLOAT32_C(   190.32),
        EASYSIMD_FLOAT32_C(    14.68), EASYSIMD_FLOAT32_C(   812.62), EASYSIMD_FLOAT32_C(   530.05), EASYSIMD_FLOAT32_C(   334.57) },
      { EASYSIMD_FLOAT32_C(  -862.62), EASYSIMD_FLOAT32_C(  -485.70), EASYSIMD_FLOAT32_C(  -679.18), EASYSIMD_FLOAT32_C(   904.08),
        EASYSIMD_FLOAT32_C(  -662.68), EASYSIMD_FLOAT32_C(   -36.41), EASYSIMD_FLOAT32_C(  -893.04), EASYSIMD_FLOAT32_C(   864.51),
        EASYSIMD_FLOAT32_C(  -413.30), EASYSIMD_FLOAT32_C(   929.61), EASYSIMD_FLOAT32_C(  -168.70), EASYSIMD_FLOAT32_C(  -196.86),
        EASYSIMD_FLOAT32_C(  -839.59), EASYSIMD_FLOAT32_C(   892.52), EASYSIMD_FLOAT32_C(  -490.18), EASYSIMD_FLOAT32_C(   704.10) },
      { EASYSIMD_FLOAT32_C(  -318.94), EASYSIMD_FLOAT32_C(   -74.54), EASYSIMD_FLOAT32_C(   434.21), EASYSIMD_FLOAT32_C(   848.98),
        EASYSIMD_FLOAT32_C(  -466.39), EASYSIMD_FLOAT32_C(  -258.84), EASYSIMD_FLOAT32_C( -1465.91), EASYSIMD_FLOAT32_C(  -660.28),
        EASYSIMD_FLOAT32_C(  -680.11), EASYSIMD_FLOAT32_C(  1790.95), EASYSIMD_FLOAT32_C(   -15.75), EASYSIMD_FLOAT32_C(   -13.75),
        EASYSIMD_FLOAT32_C(  -824.91), EASYSIMD_FLOAT32_C(  1705.14), EASYSIMD_FLOAT32_C(    39.87), EASYSIMD_FLOAT32_C(  1038.67) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 src = easysimd_mm512_loadu_ps(test_vec[i].src);
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_add_ps(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_add_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_add_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  struct {
    easysimd__mmask16 k;
    easysimd_float32 a[16];
    easysimd_float32 b[16];
    easysimd_float32 r[16];
  } test_vec[] = {
    { UINT16_C( 7629),
      { EASYSIMD_FLOAT32_C(   914.45), EASYSIMD_FLOAT32_C(   855.01), EASYSIMD_FLOAT32_C(    38.80), EASYSIMD_FLOAT32_C(   492.72),
        EASYSIMD_FLOAT32_C(   597.40), EASYSIMD_FLOAT32_C(   253.75), EASYSIMD_FLOAT32_C(   616.90), EASYSIMD_FLOAT32_C(   108.68),
        EASYSIMD_FLOAT32_C(   217.16), EASYSIMD_FLOAT32_C(   439.38), EASYSIMD_FLOAT32_C(   724.30), EASYSIMD_FLOAT32_C(   474.66),
        EASYSIMD_FLOAT32_C(   870.80), EASYSIMD_FLOAT32_C(   -46.25), EASYSIMD_FLOAT32_C(  -743.93), EASYSIMD_FLOAT32_C(   176.79) },
      { EASYSIMD_FLOAT32_C(  -872.85), EASYSIMD_FLOAT32_C(   805.82), EASYSIMD_FLOAT32_C(   350.81), EASYSIMD_FLOAT32_C(  -515.94),
        EASYSIMD_FLOAT32_C(  -720.47), EASYSIMD_FLOAT32_C(   570.49), EASYSIMD_FLOAT32_C(   295.95), EASYSIMD_FLOAT32_C(   265.48),
        EASYSIMD_FLOAT32_C(   175.46), EASYSIMD_FLOAT32_C(  -217.20), EASYSIMD_FLOAT32_C(  -845.54), EASYSIMD_FLOAT32_C(   857.16),
        EASYSIMD_FLOAT32_C(   138.12), EASYSIMD_FLOAT32_C(  -599.93), EASYSIMD_FLOAT32_C(   503.35), EASYSIMD_FLOAT32_C(    52.57) },
      { EASYSIMD_FLOAT32_C(    41.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   389.61), EASYSIMD_FLOAT32_C(   -23.22),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   912.85), EASYSIMD_FLOAT32_C(   374.16),
        EASYSIMD_FLOAT32_C(   392.62), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -121.24), EASYSIMD_FLOAT32_C(  1331.82),
        EASYSIMD_FLOAT32_C(  1008.92), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(26184),
      { EASYSIMD_FLOAT32_C(   601.56), EASYSIMD_FLOAT32_C(  -314.52), EASYSIMD_FLOAT32_C(  -512.28), EASYSIMD_FLOAT32_C(   472.36),
        EASYSIMD_FLOAT32_C(   639.23), EASYSIMD_FLOAT32_C(  -256.21), EASYSIMD_FLOAT32_C(  -350.85), EASYSIMD_FLOAT32_C(   766.38),
        EASYSIMD_FLOAT32_C(  -450.39), EASYSIMD_FLOAT32_C(   999.96), EASYSIMD_FLOAT32_C(  -749.56), EASYSIMD_FLOAT32_C(  -170.85),
        EASYSIMD_FLOAT32_C(   570.45), EASYSIMD_FLOAT32_C(   546.39), EASYSIMD_FLOAT32_C(  -905.38), EASYSIMD_FLOAT32_C(  -254.09) },
      { EASYSIMD_FLOAT32_C(  -670.81), EASYSIMD_FLOAT32_C(  -750.92), EASYSIMD_FLOAT32_C(  -396.93), EASYSIMD_FLOAT32_C(   467.31),
        EASYSIMD_FLOAT32_C(  -350.85), EASYSIMD_FLOAT32_C(  -893.58), EASYSIMD_FLOAT32_C(  -480.12), EASYSIMD_FLOAT32_C(   -95.76),
        EASYSIMD_FLOAT32_C(  -351.43), EASYSIMD_FLOAT32_C(    65.16), EASYSIMD_FLOAT32_C(  -243.28), EASYSIMD_FLOAT32_C(  -555.53),
        EASYSIMD_FLOAT32_C(   227.35), EASYSIMD_FLOAT32_C(   717.89), EASYSIMD_FLOAT32_C(   457.53), EASYSIMD_FLOAT32_C(  -171.09) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   939.67),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -830.97), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  1065.13), EASYSIMD_FLOAT32_C(  -992.84), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  1264.28), EASYSIMD_FLOAT32_C(  -447.85), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(38268),
      { EASYSIMD_FLOAT32_C(   -49.61), EASYSIMD_FLOAT32_C(  -940.59), EASYSIMD_FLOAT32_C(  -932.20), EASYSIMD_FLOAT32_C(  -479.16),
        EASYSIMD_FLOAT32_C(   605.80), EASYSIMD_FLOAT32_C(  -837.58), EASYSIMD_FLOAT32_C(   266.75), EASYSIMD_FLOAT32_C(   934.99),
        EASYSIMD_FLOAT32_C(  -588.49), EASYSIMD_FLOAT32_C(   869.82), EASYSIMD_FLOAT32_C(   402.30), EASYSIMD_FLOAT32_C(    60.66),
        EASYSIMD_FLOAT32_C(   976.24), EASYSIMD_FLOAT32_C(   922.17), EASYSIMD_FLOAT32_C(   964.89), EASYSIMD_FLOAT32_C(  -375.20) },
      { EASYSIMD_FLOAT32_C(   -12.67), EASYSIMD_FLOAT32_C(  -278.39), EASYSIMD_FLOAT32_C(    69.27), EASYSIMD_FLOAT32_C(  -785.32),
        EASYSIMD_FLOAT32_C(  -560.49), EASYSIMD_FLOAT32_C(  -473.20), EASYSIMD_FLOAT32_C(    43.59), EASYSIMD_FLOAT32_C(  -157.12),
        EASYSIMD_FLOAT32_C(  -527.94), EASYSIMD_FLOAT32_C(   344.87), EASYSIMD_FLOAT32_C(  -114.53), EASYSIMD_FLOAT32_C(   161.10),
        EASYSIMD_FLOAT32_C(  -704.71), EASYSIMD_FLOAT32_C(  -305.55), EASYSIMD_FLOAT32_C(  -600.24), EASYSIMD_FLOAT32_C(   245.68) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -862.92), EASYSIMD_FLOAT32_C( -1264.48),
        EASYSIMD_FLOAT32_C(    45.31), EASYSIMD_FLOAT32_C( -1310.77), EASYSIMD_FLOAT32_C(   310.34), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C( -1116.44), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   287.77), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   271.52), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -129.52) } },
    { UINT16_C(37369),
      { EASYSIMD_FLOAT32_C(   -96.91), EASYSIMD_FLOAT32_C(   696.94), EASYSIMD_FLOAT32_C(  -897.85), EASYSIMD_FLOAT32_C(  -120.68),
        EASYSIMD_FLOAT32_C(   619.12), EASYSIMD_FLOAT32_C(  -932.96), EASYSIMD_FLOAT32_C(   504.13), EASYSIMD_FLOAT32_C(  -393.55),
        EASYSIMD_FLOAT32_C(  -211.35), EASYSIMD_FLOAT32_C(  -426.60), EASYSIMD_FLOAT32_C(  -178.87), EASYSIMD_FLOAT32_C(   228.16),
        EASYSIMD_FLOAT32_C(   100.20), EASYSIMD_FLOAT32_C(   864.72), EASYSIMD_FLOAT32_C(  -928.97), EASYSIMD_FLOAT32_C(   572.26) },
      { EASYSIMD_FLOAT32_C(   209.59), EASYSIMD_FLOAT32_C(   -43.49), EASYSIMD_FLOAT32_C(  -266.64), EASYSIMD_FLOAT32_C(   504.88),
        EASYSIMD_FLOAT32_C(   650.96), EASYSIMD_FLOAT32_C(   133.12), EASYSIMD_FLOAT32_C(  -249.44), EASYSIMD_FLOAT32_C(  -595.18),
        EASYSIMD_FLOAT32_C(   600.68), EASYSIMD_FLOAT32_C(  -482.93), EASYSIMD_FLOAT32_C(  -235.52), EASYSIMD_FLOAT32_C(  -769.33),
        EASYSIMD_FLOAT32_C(   550.34), EASYSIMD_FLOAT32_C(    59.13), EASYSIMD_FLOAT32_C(   272.16), EASYSIMD_FLOAT32_C(  -546.58) },
      { EASYSIMD_FLOAT32_C(   112.67), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   384.20),
        EASYSIMD_FLOAT32_C(  1270.07), EASYSIMD_FLOAT32_C(  -799.84), EASYSIMD_FLOAT32_C(   254.68), EASYSIMD_FLOAT32_C(  -988.73),
        EASYSIMD_FLOAT32_C(   389.33), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   650.54), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    25.68) } },
    { UINT16_C(17887),
      { EASYSIMD_FLOAT32_C(   410.27), EASYSIMD_FLOAT32_C(   802.77), EASYSIMD_FLOAT32_C(   458.15), EASYSIMD_FLOAT32_C(  -489.53),
        EASYSIMD_FLOAT32_C(   667.49), EASYSIMD_FLOAT32_C(   529.19), EASYSIMD_FLOAT32_C(  -917.27), EASYSIMD_FLOAT32_C(  -122.92),
        EASYSIMD_FLOAT32_C(  -514.30), EASYSIMD_FLOAT32_C(  -183.91), EASYSIMD_FLOAT32_C(  -618.04), EASYSIMD_FLOAT32_C(  -863.35),
        EASYSIMD_FLOAT32_C(   949.21), EASYSIMD_FLOAT32_C(   132.51), EASYSIMD_FLOAT32_C(  -458.53), EASYSIMD_FLOAT32_C(   549.89) },
      { EASYSIMD_FLOAT32_C(   649.59), EASYSIMD_FLOAT32_C(   305.95), EASYSIMD_FLOAT32_C(   780.56), EASYSIMD_FLOAT32_C(   199.92),
        EASYSIMD_FLOAT32_C(  -634.93), EASYSIMD_FLOAT32_C(    52.72), EASYSIMD_FLOAT32_C(   653.35), EASYSIMD_FLOAT32_C(   121.14),
        EASYSIMD_FLOAT32_C(  -572.98), EASYSIMD_FLOAT32_C(   -13.91), EASYSIMD_FLOAT32_C(   496.32), EASYSIMD_FLOAT32_C(   868.36),
        EASYSIMD_FLOAT32_C(   822.96), EASYSIMD_FLOAT32_C(  -522.04), EASYSIMD_FLOAT32_C(  -901.64), EASYSIMD_FLOAT32_C(   233.23) },
      { EASYSIMD_FLOAT32_C(  1059.85), EASYSIMD_FLOAT32_C(  1108.71), EASYSIMD_FLOAT32_C(  1238.71), EASYSIMD_FLOAT32_C(  -289.60),
        EASYSIMD_FLOAT32_C(    32.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -263.92), EASYSIMD_FLOAT32_C(    -1.78),
        EASYSIMD_FLOAT32_C( -1087.28), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -121.72), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( -1360.17), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(52085),
      { EASYSIMD_FLOAT32_C(  -357.48), EASYSIMD_FLOAT32_C(   207.25), EASYSIMD_FLOAT32_C(   708.05), EASYSIMD_FLOAT32_C(  -408.26),
        EASYSIMD_FLOAT32_C(  -660.23), EASYSIMD_FLOAT32_C(  -750.48), EASYSIMD_FLOAT32_C(  -858.37), EASYSIMD_FLOAT32_C(   989.35),
        EASYSIMD_FLOAT32_C(   555.47), EASYSIMD_FLOAT32_C(   922.19), EASYSIMD_FLOAT32_C(   189.28), EASYSIMD_FLOAT32_C(   920.54),
        EASYSIMD_FLOAT32_C(   -25.09), EASYSIMD_FLOAT32_C(  -157.38), EASYSIMD_FLOAT32_C(    41.68), EASYSIMD_FLOAT32_C(   401.93) },
      { EASYSIMD_FLOAT32_C(   828.72), EASYSIMD_FLOAT32_C(  -462.00), EASYSIMD_FLOAT32_C(   270.29), EASYSIMD_FLOAT32_C(   651.68),
        EASYSIMD_FLOAT32_C(    15.96), EASYSIMD_FLOAT32_C(   368.65), EASYSIMD_FLOAT32_C(  -115.09), EASYSIMD_FLOAT32_C(   296.68),
        EASYSIMD_FLOAT32_C(   -74.83), EASYSIMD_FLOAT32_C(  -371.39), EASYSIMD_FLOAT32_C(   244.89), EASYSIMD_FLOAT32_C(  -989.13),
        EASYSIMD_FLOAT32_C(  -544.95), EASYSIMD_FLOAT32_C(  -929.81), EASYSIMD_FLOAT32_C(   582.27), EASYSIMD_FLOAT32_C(    97.57) },
      { EASYSIMD_FLOAT32_C(   471.24), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   978.34), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -644.28), EASYSIMD_FLOAT32_C(  -381.83), EASYSIMD_FLOAT32_C(  -973.46), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   480.63), EASYSIMD_FLOAT32_C(   550.80), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -68.59),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   623.95), EASYSIMD_FLOAT32_C(   499.50) } },
    { UINT16_C(30320),
      { EASYSIMD_FLOAT32_C(   753.13), EASYSIMD_FLOAT32_C(  -204.17), EASYSIMD_FLOAT32_C(    15.84), EASYSIMD_FLOAT32_C(  -271.97),
        EASYSIMD_FLOAT32_C(   638.45), EASYSIMD_FLOAT32_C(  -942.48), EASYSIMD_FLOAT32_C(  -870.04), EASYSIMD_FLOAT32_C(   467.17),
        EASYSIMD_FLOAT32_C(  -404.47), EASYSIMD_FLOAT32_C(   400.26), EASYSIMD_FLOAT32_C(   118.85), EASYSIMD_FLOAT32_C(   611.49),
        EASYSIMD_FLOAT32_C(  -231.09), EASYSIMD_FLOAT32_C(  -996.24), EASYSIMD_FLOAT32_C(   -91.83), EASYSIMD_FLOAT32_C(   694.08) },
      { EASYSIMD_FLOAT32_C(  -367.63), EASYSIMD_FLOAT32_C(  -846.94), EASYSIMD_FLOAT32_C(   704.95), EASYSIMD_FLOAT32_C(    87.42),
        EASYSIMD_FLOAT32_C(  -776.75), EASYSIMD_FLOAT32_C(   287.22), EASYSIMD_FLOAT32_C(  -815.01), EASYSIMD_FLOAT32_C(   500.69),
        EASYSIMD_FLOAT32_C(  -422.46), EASYSIMD_FLOAT32_C(   874.30), EASYSIMD_FLOAT32_C(   117.89), EASYSIMD_FLOAT32_C(  -882.62),
        EASYSIMD_FLOAT32_C(   705.23), EASYSIMD_FLOAT32_C(  -275.56), EASYSIMD_FLOAT32_C(   212.68), EASYSIMD_FLOAT32_C(   458.36) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -138.30), EASYSIMD_FLOAT32_C(  -655.26), EASYSIMD_FLOAT32_C( -1685.05), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  1274.55), EASYSIMD_FLOAT32_C(   236.74), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   474.14), EASYSIMD_FLOAT32_C( -1271.80), EASYSIMD_FLOAT32_C(   120.85), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C( 6201),
      { EASYSIMD_FLOAT32_C(   716.62), EASYSIMD_FLOAT32_C(   744.75), EASYSIMD_FLOAT32_C(  -506.94), EASYSIMD_FLOAT32_C(  -514.47),
        EASYSIMD_FLOAT32_C(   748.50), EASYSIMD_FLOAT32_C(   401.23), EASYSIMD_FLOAT32_C(  -820.39), EASYSIMD_FLOAT32_C(  -619.12),
        EASYSIMD_FLOAT32_C(   554.30), EASYSIMD_FLOAT32_C(   884.56), EASYSIMD_FLOAT32_C(   468.30), EASYSIMD_FLOAT32_C(   777.54),
        EASYSIMD_FLOAT32_C(   171.78), EASYSIMD_FLOAT32_C(   653.28), EASYSIMD_FLOAT32_C(   278.23), EASYSIMD_FLOAT32_C(   749.31) },
      { EASYSIMD_FLOAT32_C(   527.58), EASYSIMD_FLOAT32_C(  -603.88), EASYSIMD_FLOAT32_C(   866.69), EASYSIMD_FLOAT32_C(   232.81),
        EASYSIMD_FLOAT32_C(   120.56), EASYSIMD_FLOAT32_C(    79.37), EASYSIMD_FLOAT32_C(  -308.83), EASYSIMD_FLOAT32_C(  -359.16),
        EASYSIMD_FLOAT32_C(   307.90), EASYSIMD_FLOAT32_C(  -122.44), EASYSIMD_FLOAT32_C(   799.56), EASYSIMD_FLOAT32_C(   593.95),
        EASYSIMD_FLOAT32_C(   193.92), EASYSIMD_FLOAT32_C(  -574.54), EASYSIMD_FLOAT32_C(  -524.47), EASYSIMD_FLOAT32_C(   -89.46) },
      { EASYSIMD_FLOAT32_C(  1244.20), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -281.66),
        EASYSIMD_FLOAT32_C(   869.07), EASYSIMD_FLOAT32_C(   480.61), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  1371.50),
        EASYSIMD_FLOAT32_C(   365.70), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_add_ps(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_add_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_add_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -775.47), EASYSIMD_FLOAT64_C(     0.19), EASYSIMD_FLOAT64_C(  -757.09), EASYSIMD_FLOAT64_C(  -410.73),
        EASYSIMD_FLOAT64_C(  -780.15), EASYSIMD_FLOAT64_C(  -708.07), EASYSIMD_FLOAT64_C(   367.35), EASYSIMD_FLOAT64_C(  -968.32) },
      { EASYSIMD_FLOAT64_C(   820.33), EASYSIMD_FLOAT64_C(   850.82), EASYSIMD_FLOAT64_C(   596.13), EASYSIMD_FLOAT64_C(  -878.99),
        EASYSIMD_FLOAT64_C(  -603.61), EASYSIMD_FLOAT64_C(   533.64), EASYSIMD_FLOAT64_C(  -994.37), EASYSIMD_FLOAT64_C(   846.35) },
      { EASYSIMD_FLOAT64_C(    44.87), EASYSIMD_FLOAT64_C(   851.01), EASYSIMD_FLOAT64_C(  -160.96), EASYSIMD_FLOAT64_C( -1289.72),
        EASYSIMD_FLOAT64_C( -1383.75), EASYSIMD_FLOAT64_C(  -174.43), EASYSIMD_FLOAT64_C(  -627.02), EASYSIMD_FLOAT64_C(  -121.96) } },
    { { EASYSIMD_FLOAT64_C(  -503.42), EASYSIMD_FLOAT64_C(  -250.77), EASYSIMD_FLOAT64_C(  -532.42), EASYSIMD_FLOAT64_C(   815.06),
        EASYSIMD_FLOAT64_C(  -419.11), EASYSIMD_FLOAT64_C(   224.41), EASYSIMD_FLOAT64_C(   -34.26), EASYSIMD_FLOAT64_C(  -803.36) },
      { EASYSIMD_FLOAT64_C(  -331.10), EASYSIMD_FLOAT64_C(  -474.33), EASYSIMD_FLOAT64_C(   866.30), EASYSIMD_FLOAT64_C(   560.33),
        EASYSIMD_FLOAT64_C(   467.15), EASYSIMD_FLOAT64_C(   279.38), EASYSIMD_FLOAT64_C(  -475.96), EASYSIMD_FLOAT64_C(   691.69) },
      { EASYSIMD_FLOAT64_C(  -834.52), EASYSIMD_FLOAT64_C(  -725.11), EASYSIMD_FLOAT64_C(   333.88), EASYSIMD_FLOAT64_C(  1375.40),
        EASYSIMD_FLOAT64_C(    48.04), EASYSIMD_FLOAT64_C(   503.79), EASYSIMD_FLOAT64_C(  -510.22), EASYSIMD_FLOAT64_C(  -111.67) } },
    { { EASYSIMD_FLOAT64_C(  -720.44), EASYSIMD_FLOAT64_C(  -233.05), EASYSIMD_FLOAT64_C(  -719.04), EASYSIMD_FLOAT64_C(  -500.58),
        EASYSIMD_FLOAT64_C(    58.88), EASYSIMD_FLOAT64_C(   648.31), EASYSIMD_FLOAT64_C(  -468.90), EASYSIMD_FLOAT64_C(  -120.79) },
      { EASYSIMD_FLOAT64_C(   499.13), EASYSIMD_FLOAT64_C(  -872.76), EASYSIMD_FLOAT64_C(     0.22), EASYSIMD_FLOAT64_C(   895.52),
        EASYSIMD_FLOAT64_C(   660.88), EASYSIMD_FLOAT64_C(     5.85), EASYSIMD_FLOAT64_C(   741.88), EASYSIMD_FLOAT64_C(  -842.54) },
      { EASYSIMD_FLOAT64_C(  -221.31), EASYSIMD_FLOAT64_C( -1105.81), EASYSIMD_FLOAT64_C(  -718.83), EASYSIMD_FLOAT64_C(   394.94),
        EASYSIMD_FLOAT64_C(   719.76), EASYSIMD_FLOAT64_C(   654.16), EASYSIMD_FLOAT64_C(   272.98), EASYSIMD_FLOAT64_C(  -963.33) } },
    { { EASYSIMD_FLOAT64_C(   755.08), EASYSIMD_FLOAT64_C(  -790.54), EASYSIMD_FLOAT64_C(   972.53), EASYSIMD_FLOAT64_C(  -664.03),
        EASYSIMD_FLOAT64_C(   433.87), EASYSIMD_FLOAT64_C(   -61.74), EASYSIMD_FLOAT64_C(  -467.39), EASYSIMD_FLOAT64_C(  -897.23) },
      { EASYSIMD_FLOAT64_C(   463.93), EASYSIMD_FLOAT64_C(  -601.09), EASYSIMD_FLOAT64_C(   663.10), EASYSIMD_FLOAT64_C(   -68.92),
        EASYSIMD_FLOAT64_C(   678.29), EASYSIMD_FLOAT64_C(  -812.86), EASYSIMD_FLOAT64_C(  -377.23), EASYSIMD_FLOAT64_C(   957.85) },
      { EASYSIMD_FLOAT64_C(  1219.01), EASYSIMD_FLOAT64_C( -1391.63), EASYSIMD_FLOAT64_C(  1635.63), EASYSIMD_FLOAT64_C(  -732.95),
        EASYSIMD_FLOAT64_C(  1112.16), EASYSIMD_FLOAT64_C(  -874.59), EASYSIMD_FLOAT64_C(  -844.62), EASYSIMD_FLOAT64_C(    60.62) } },
    { { EASYSIMD_FLOAT64_C(   -45.90), EASYSIMD_FLOAT64_C(   -96.28), EASYSIMD_FLOAT64_C(  -542.73), EASYSIMD_FLOAT64_C(  -987.02),
        EASYSIMD_FLOAT64_C(  -447.97), EASYSIMD_FLOAT64_C(   -11.63), EASYSIMD_FLOAT64_C(  -107.82), EASYSIMD_FLOAT64_C(  -948.84) },
      { EASYSIMD_FLOAT64_C(   115.60), EASYSIMD_FLOAT64_C(   892.40), EASYSIMD_FLOAT64_C(   946.68), EASYSIMD_FLOAT64_C(  -223.52),
        EASYSIMD_FLOAT64_C(  -101.75), EASYSIMD_FLOAT64_C(   688.56), EASYSIMD_FLOAT64_C(   -66.05), EASYSIMD_FLOAT64_C(  -346.67) },
      { EASYSIMD_FLOAT64_C(    69.70), EASYSIMD_FLOAT64_C(   796.12), EASYSIMD_FLOAT64_C(   403.95), EASYSIMD_FLOAT64_C( -1210.54),
        EASYSIMD_FLOAT64_C(  -549.72), EASYSIMD_FLOAT64_C(   676.92), EASYSIMD_FLOAT64_C(  -173.87), EASYSIMD_FLOAT64_C( -1295.52) } },
    { { EASYSIMD_FLOAT64_C(   898.01), EASYSIMD_FLOAT64_C(   -93.53), EASYSIMD_FLOAT64_C(   -10.70), EASYSIMD_FLOAT64_C(   331.89),
        EASYSIMD_FLOAT64_C(   844.74), EASYSIMD_FLOAT64_C(   521.91), EASYSIMD_FLOAT64_C(   434.66), EASYSIMD_FLOAT64_C(   308.66) },
      { EASYSIMD_FLOAT64_C(   920.82), EASYSIMD_FLOAT64_C(    97.76), EASYSIMD_FLOAT64_C(  -760.25), EASYSIMD_FLOAT64_C(   599.10),
        EASYSIMD_FLOAT64_C(   284.91), EASYSIMD_FLOAT64_C(  -137.49), EASYSIMD_FLOAT64_C(   556.96), EASYSIMD_FLOAT64_C(  -761.00) },
      { EASYSIMD_FLOAT64_C(  1818.83), EASYSIMD_FLOAT64_C(     4.23), EASYSIMD_FLOAT64_C(  -770.96), EASYSIMD_FLOAT64_C(   930.99),
        EASYSIMD_FLOAT64_C(  1129.64), EASYSIMD_FLOAT64_C(   384.42), EASYSIMD_FLOAT64_C(   991.61), EASYSIMD_FLOAT64_C(  -452.33) } },
    { { EASYSIMD_FLOAT64_C(   766.23), EASYSIMD_FLOAT64_C(  -985.78), EASYSIMD_FLOAT64_C(  -748.02), EASYSIMD_FLOAT64_C(  -681.74),
        EASYSIMD_FLOAT64_C(     2.59), EASYSIMD_FLOAT64_C(   144.16), EASYSIMD_FLOAT64_C(  -630.58), EASYSIMD_FLOAT64_C(  -881.80) },
      { EASYSIMD_FLOAT64_C(    36.57), EASYSIMD_FLOAT64_C(  -683.90), EASYSIMD_FLOAT64_C(  -105.32), EASYSIMD_FLOAT64_C(   934.82),
        EASYSIMD_FLOAT64_C(  -995.35), EASYSIMD_FLOAT64_C(   828.63), EASYSIMD_FLOAT64_C(  -411.86), EASYSIMD_FLOAT64_C(   902.67) },
      { EASYSIMD_FLOAT64_C(   802.80), EASYSIMD_FLOAT64_C( -1669.68), EASYSIMD_FLOAT64_C(  -853.34), EASYSIMD_FLOAT64_C(   253.08),
        EASYSIMD_FLOAT64_C(  -992.76), EASYSIMD_FLOAT64_C(   972.79), EASYSIMD_FLOAT64_C( -1042.44), EASYSIMD_FLOAT64_C(    20.86) } },
    { { EASYSIMD_FLOAT64_C(  -264.90), EASYSIMD_FLOAT64_C(   577.44), EASYSIMD_FLOAT64_C(   234.56), EASYSIMD_FLOAT64_C(  -420.17),
        EASYSIMD_FLOAT64_C(    99.35), EASYSIMD_FLOAT64_C(  -330.78), EASYSIMD_FLOAT64_C(   888.50), EASYSIMD_FLOAT64_C(    20.17) },
      { EASYSIMD_FLOAT64_C(   766.98), EASYSIMD_FLOAT64_C(  -871.76), EASYSIMD_FLOAT64_C(  -380.73), EASYSIMD_FLOAT64_C(    51.88),
        EASYSIMD_FLOAT64_C(    -9.24), EASYSIMD_FLOAT64_C(  -823.77), EASYSIMD_FLOAT64_C(   290.89), EASYSIMD_FLOAT64_C(  -243.01) },
      { EASYSIMD_FLOAT64_C(   502.08), EASYSIMD_FLOAT64_C(  -294.31), EASYSIMD_FLOAT64_C(  -146.17), EASYSIMD_FLOAT64_C(  -368.28),
        EASYSIMD_FLOAT64_C(    90.11), EASYSIMD_FLOAT64_C( -1154.55), EASYSIMD_FLOAT64_C(  1179.39), EASYSIMD_FLOAT64_C(  -222.84) } }
  };


  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_add_pd(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_add_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_add_round_pd (EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT64_C(   456.52), EASYSIMD_FLOAT64_C(  -438.71), EASYSIMD_FLOAT64_C(  -439.96), EASYSIMD_FLOAT64_C(  -841.37),
        EASYSIMD_FLOAT64_C(   236.65), EASYSIMD_FLOAT64_C(   757.81), EASYSIMD_FLOAT64_C(  -257.58), EASYSIMD_FLOAT64_C(   925.75) },
      { EASYSIMD_FLOAT64_C(  -304.33), EASYSIMD_FLOAT64_C(  -407.22), EASYSIMD_FLOAT64_C(  -277.23), EASYSIMD_FLOAT64_C(  -940.50),
        EASYSIMD_FLOAT64_C(   213.66), EASYSIMD_FLOAT64_C(   838.94), EASYSIMD_FLOAT64_C(  -922.54), EASYSIMD_FLOAT64_C(  -892.14) },
      { EASYSIMD_FLOAT64_C(   152.00), EASYSIMD_FLOAT64_C(  -846.00), EASYSIMD_FLOAT64_C(  -717.00), EASYSIMD_FLOAT64_C( -1782.00),
        EASYSIMD_FLOAT64_C(   450.00), EASYSIMD_FLOAT64_C(  1597.00), EASYSIMD_FLOAT64_C( -1180.00), EASYSIMD_FLOAT64_C(    34.00) },
      { EASYSIMD_FLOAT64_C(   152.00), EASYSIMD_FLOAT64_C(  -846.00), EASYSIMD_FLOAT64_C(  -718.00), EASYSIMD_FLOAT64_C( -1782.00),
        EASYSIMD_FLOAT64_C(   450.00), EASYSIMD_FLOAT64_C(  1596.00), EASYSIMD_FLOAT64_C( -1181.00), EASYSIMD_FLOAT64_C(    33.00) },
      { EASYSIMD_FLOAT64_C(   153.00), EASYSIMD_FLOAT64_C(  -845.00), EASYSIMD_FLOAT64_C(  -717.00), EASYSIMD_FLOAT64_C( -1781.00),
        EASYSIMD_FLOAT64_C(   451.00), EASYSIMD_FLOAT64_C(  1597.00), EASYSIMD_FLOAT64_C( -1180.00), EASYSIMD_FLOAT64_C(    34.00) },
      { EASYSIMD_FLOAT64_C(   152.00), EASYSIMD_FLOAT64_C(  -845.00), EASYSIMD_FLOAT64_C(  -717.00), EASYSIMD_FLOAT64_C( -1781.00),
        EASYSIMD_FLOAT64_C(   450.00), EASYSIMD_FLOAT64_C(  1596.00), EASYSIMD_FLOAT64_C( -1180.00), EASYSIMD_FLOAT64_C(    33.00) },
      { EASYSIMD_FLOAT64_C(   152.00), EASYSIMD_FLOAT64_C(  -846.00), EASYSIMD_FLOAT64_C(  -717.00), EASYSIMD_FLOAT64_C( -1782.00),
        EASYSIMD_FLOAT64_C(   450.00), EASYSIMD_FLOAT64_C(  1597.00), EASYSIMD_FLOAT64_C( -1180.00), EASYSIMD_FLOAT64_C(    34.00) } },
    { { EASYSIMD_FLOAT64_C(   671.22), EASYSIMD_FLOAT64_C(  -607.40), EASYSIMD_FLOAT64_C(  -837.87), EASYSIMD_FLOAT64_C(    78.42),
        EASYSIMD_FLOAT64_C(    17.88), EASYSIMD_FLOAT64_C(    67.16), EASYSIMD_FLOAT64_C(  -141.43), EASYSIMD_FLOAT64_C(   602.72) },
      { EASYSIMD_FLOAT64_C(   162.53), EASYSIMD_FLOAT64_C(  -227.13), EASYSIMD_FLOAT64_C(  -399.53), EASYSIMD_FLOAT64_C(  -978.99),
        EASYSIMD_FLOAT64_C(  -633.57), EASYSIMD_FLOAT64_C(   251.87), EASYSIMD_FLOAT64_C(   380.62), EASYSIMD_FLOAT64_C(   822.95) },
      { EASYSIMD_FLOAT64_C(   834.00), EASYSIMD_FLOAT64_C(  -835.00), EASYSIMD_FLOAT64_C( -1237.00), EASYSIMD_FLOAT64_C(  -901.00),
        EASYSIMD_FLOAT64_C(  -616.00), EASYSIMD_FLOAT64_C(   319.00), EASYSIMD_FLOAT64_C(   239.00), EASYSIMD_FLOAT64_C(  1426.00) },
      { EASYSIMD_FLOAT64_C(   833.00), EASYSIMD_FLOAT64_C(  -835.00), EASYSIMD_FLOAT64_C( -1238.00), EASYSIMD_FLOAT64_C(  -901.00),
        EASYSIMD_FLOAT64_C(  -616.00), EASYSIMD_FLOAT64_C(   319.00), EASYSIMD_FLOAT64_C(   239.00), EASYSIMD_FLOAT64_C(  1425.00) },
      { EASYSIMD_FLOAT64_C(   834.00), EASYSIMD_FLOAT64_C(  -834.00), EASYSIMD_FLOAT64_C( -1237.00), EASYSIMD_FLOAT64_C(  -900.00),
        EASYSIMD_FLOAT64_C(  -615.00), EASYSIMD_FLOAT64_C(   320.00), EASYSIMD_FLOAT64_C(   240.00), EASYSIMD_FLOAT64_C(  1426.00) },
      { EASYSIMD_FLOAT64_C(   833.00), EASYSIMD_FLOAT64_C(  -834.00), EASYSIMD_FLOAT64_C( -1237.00), EASYSIMD_FLOAT64_C(  -900.00),
        EASYSIMD_FLOAT64_C(  -615.00), EASYSIMD_FLOAT64_C(   319.00), EASYSIMD_FLOAT64_C(   239.00), EASYSIMD_FLOAT64_C(  1425.00) },
      { EASYSIMD_FLOAT64_C(   834.00), EASYSIMD_FLOAT64_C(  -835.00), EASYSIMD_FLOAT64_C( -1237.00), EASYSIMD_FLOAT64_C(  -901.00),
        EASYSIMD_FLOAT64_C(  -616.00), EASYSIMD_FLOAT64_C(   319.00), EASYSIMD_FLOAT64_C(   239.00), EASYSIMD_FLOAT64_C(  1426.00) } },
    { { EASYSIMD_FLOAT64_C(   813.16), EASYSIMD_FLOAT64_C(   940.66), EASYSIMD_FLOAT64_C(   981.58), EASYSIMD_FLOAT64_C(    49.81),
        EASYSIMD_FLOAT64_C(   698.47), EASYSIMD_FLOAT64_C(  -276.00), EASYSIMD_FLOAT64_C(   -24.44), EASYSIMD_FLOAT64_C(  -605.87) },
      { EASYSIMD_FLOAT64_C(   316.78), EASYSIMD_FLOAT64_C(   698.33), EASYSIMD_FLOAT64_C(  -546.36), EASYSIMD_FLOAT64_C(  -469.57),
        EASYSIMD_FLOAT64_C(   537.28), EASYSIMD_FLOAT64_C(  -468.91), EASYSIMD_FLOAT64_C(  -361.71), EASYSIMD_FLOAT64_C(   208.49) },
      { EASYSIMD_FLOAT64_C(  1130.00), EASYSIMD_FLOAT64_C(  1639.00), EASYSIMD_FLOAT64_C(   435.00), EASYSIMD_FLOAT64_C(  -420.00),
        EASYSIMD_FLOAT64_C(  1236.00), EASYSIMD_FLOAT64_C(  -745.00), EASYSIMD_FLOAT64_C(  -386.00), EASYSIMD_FLOAT64_C(  -397.00) },
      { EASYSIMD_FLOAT64_C(  1129.00), EASYSIMD_FLOAT64_C(  1638.00), EASYSIMD_FLOAT64_C(   435.00), EASYSIMD_FLOAT64_C(  -420.00),
        EASYSIMD_FLOAT64_C(  1235.00), EASYSIMD_FLOAT64_C(  -745.00), EASYSIMD_FLOAT64_C(  -387.00), EASYSIMD_FLOAT64_C(  -398.00) },
      { EASYSIMD_FLOAT64_C(  1130.00), EASYSIMD_FLOAT64_C(  1639.00), EASYSIMD_FLOAT64_C(   436.00), EASYSIMD_FLOAT64_C(  -419.00),
        EASYSIMD_FLOAT64_C(  1236.00), EASYSIMD_FLOAT64_C(  -744.00), EASYSIMD_FLOAT64_C(  -386.00), EASYSIMD_FLOAT64_C(  -397.00) },
      { EASYSIMD_FLOAT64_C(  1129.00), EASYSIMD_FLOAT64_C(  1638.00), EASYSIMD_FLOAT64_C(   435.00), EASYSIMD_FLOAT64_C(  -419.00),
        EASYSIMD_FLOAT64_C(  1235.00), EASYSIMD_FLOAT64_C(  -744.00), EASYSIMD_FLOAT64_C(  -386.00), EASYSIMD_FLOAT64_C(  -397.00) },
      { EASYSIMD_FLOAT64_C(  1130.00), EASYSIMD_FLOAT64_C(  1639.00), EASYSIMD_FLOAT64_C(   435.00), EASYSIMD_FLOAT64_C(  -420.00),
        EASYSIMD_FLOAT64_C(  1236.00), EASYSIMD_FLOAT64_C(  -745.00), EASYSIMD_FLOAT64_C(  -386.00), EASYSIMD_FLOAT64_C(  -397.00) } },
    { { EASYSIMD_FLOAT64_C(   -76.30), EASYSIMD_FLOAT64_C(  -199.58), EASYSIMD_FLOAT64_C(  -713.08), EASYSIMD_FLOAT64_C(   941.58),
        EASYSIMD_FLOAT64_C(   867.58), EASYSIMD_FLOAT64_C(   145.48), EASYSIMD_FLOAT64_C(   544.31), EASYSIMD_FLOAT64_C(    30.11) },
      { EASYSIMD_FLOAT64_C(   918.35), EASYSIMD_FLOAT64_C(  -855.23), EASYSIMD_FLOAT64_C(    51.12), EASYSIMD_FLOAT64_C(  -715.22),
        EASYSIMD_FLOAT64_C(   396.65), EASYSIMD_FLOAT64_C(  -568.27), EASYSIMD_FLOAT64_C(  -892.27), EASYSIMD_FLOAT64_C(   209.81) },
      { EASYSIMD_FLOAT64_C(   842.00), EASYSIMD_FLOAT64_C( -1055.00), EASYSIMD_FLOAT64_C(  -662.00), EASYSIMD_FLOAT64_C(   226.00),
        EASYSIMD_FLOAT64_C(  1264.00), EASYSIMD_FLOAT64_C(  -423.00), EASYSIMD_FLOAT64_C(  -348.00), EASYSIMD_FLOAT64_C(   240.00) },
      { EASYSIMD_FLOAT64_C(   842.00), EASYSIMD_FLOAT64_C( -1055.00), EASYSIMD_FLOAT64_C(  -662.00), EASYSIMD_FLOAT64_C(   226.00),
        EASYSIMD_FLOAT64_C(  1264.00), EASYSIMD_FLOAT64_C(  -423.00), EASYSIMD_FLOAT64_C(  -348.00), EASYSIMD_FLOAT64_C(   239.00) },
      { EASYSIMD_FLOAT64_C(   843.00), EASYSIMD_FLOAT64_C( -1054.00), EASYSIMD_FLOAT64_C(  -661.00), EASYSIMD_FLOAT64_C(   227.00),
        EASYSIMD_FLOAT64_C(  1265.00), EASYSIMD_FLOAT64_C(  -422.00), EASYSIMD_FLOAT64_C(  -347.00), EASYSIMD_FLOAT64_C(   240.00) },
      { EASYSIMD_FLOAT64_C(   842.00), EASYSIMD_FLOAT64_C( -1054.00), EASYSIMD_FLOAT64_C(  -661.00), EASYSIMD_FLOAT64_C(   226.00),
        EASYSIMD_FLOAT64_C(  1264.00), EASYSIMD_FLOAT64_C(  -422.00), EASYSIMD_FLOAT64_C(  -347.00), EASYSIMD_FLOAT64_C(   239.00) },
      { EASYSIMD_FLOAT64_C(   842.00), EASYSIMD_FLOAT64_C( -1055.00), EASYSIMD_FLOAT64_C(  -662.00), EASYSIMD_FLOAT64_C(   226.00),
        EASYSIMD_FLOAT64_C(  1264.00), EASYSIMD_FLOAT64_C(  -423.00), EASYSIMD_FLOAT64_C(  -348.00), EASYSIMD_FLOAT64_C(   240.00) } },
    { { EASYSIMD_FLOAT64_C(  -627.61), EASYSIMD_FLOAT64_C(  -910.68), EASYSIMD_FLOAT64_C(  -740.38), EASYSIMD_FLOAT64_C(  -929.14),
        EASYSIMD_FLOAT64_C(  -186.68), EASYSIMD_FLOAT64_C(   235.19), EASYSIMD_FLOAT64_C(  -535.01), EASYSIMD_FLOAT64_C(  -869.90) },
      { EASYSIMD_FLOAT64_C(   -66.48), EASYSIMD_FLOAT64_C(   -81.37), EASYSIMD_FLOAT64_C(  -339.47), EASYSIMD_FLOAT64_C(  -529.21),
        EASYSIMD_FLOAT64_C(   449.72), EASYSIMD_FLOAT64_C(   298.82), EASYSIMD_FLOAT64_C(   679.29), EASYSIMD_FLOAT64_C(  -626.58) },
      { EASYSIMD_FLOAT64_C(  -694.00), EASYSIMD_FLOAT64_C(  -992.00), EASYSIMD_FLOAT64_C( -1080.00), EASYSIMD_FLOAT64_C( -1458.00),
        EASYSIMD_FLOAT64_C(   263.00), EASYSIMD_FLOAT64_C(   534.00), EASYSIMD_FLOAT64_C(   144.00), EASYSIMD_FLOAT64_C( -1496.00) },
      { EASYSIMD_FLOAT64_C(  -695.00), EASYSIMD_FLOAT64_C(  -993.00), EASYSIMD_FLOAT64_C( -1080.00), EASYSIMD_FLOAT64_C( -1459.00),
        EASYSIMD_FLOAT64_C(   263.00), EASYSIMD_FLOAT64_C(   534.00), EASYSIMD_FLOAT64_C(   144.00), EASYSIMD_FLOAT64_C( -1497.00) },
      { EASYSIMD_FLOAT64_C(  -694.00), EASYSIMD_FLOAT64_C(  -992.00), EASYSIMD_FLOAT64_C( -1079.00), EASYSIMD_FLOAT64_C( -1458.00),
        EASYSIMD_FLOAT64_C(   264.00), EASYSIMD_FLOAT64_C(   535.00), EASYSIMD_FLOAT64_C(   145.00), EASYSIMD_FLOAT64_C( -1496.00) },
      { EASYSIMD_FLOAT64_C(  -694.00), EASYSIMD_FLOAT64_C(  -992.00), EASYSIMD_FLOAT64_C( -1079.00), EASYSIMD_FLOAT64_C( -1458.00),
        EASYSIMD_FLOAT64_C(   263.00), EASYSIMD_FLOAT64_C(   534.00), EASYSIMD_FLOAT64_C(   144.00), EASYSIMD_FLOAT64_C( -1496.00) },
      { EASYSIMD_FLOAT64_C(  -694.00), EASYSIMD_FLOAT64_C(  -992.00), EASYSIMD_FLOAT64_C( -1080.00), EASYSIMD_FLOAT64_C( -1458.00),
        EASYSIMD_FLOAT64_C(   263.00), EASYSIMD_FLOAT64_C(   534.00), EASYSIMD_FLOAT64_C(   144.00), EASYSIMD_FLOAT64_C( -1496.00) } },
    { { EASYSIMD_FLOAT64_C(  -900.75), EASYSIMD_FLOAT64_C(   966.20), EASYSIMD_FLOAT64_C(  -685.00), EASYSIMD_FLOAT64_C(   966.83),
        EASYSIMD_FLOAT64_C(   111.68), EASYSIMD_FLOAT64_C(   859.31), EASYSIMD_FLOAT64_C(    -3.06), EASYSIMD_FLOAT64_C(    30.04) },
      { EASYSIMD_FLOAT64_C(  -995.92), EASYSIMD_FLOAT64_C(  -951.95), EASYSIMD_FLOAT64_C(   314.82), EASYSIMD_FLOAT64_C(   400.73),
        EASYSIMD_FLOAT64_C(  -520.21), EASYSIMD_FLOAT64_C(   422.55), EASYSIMD_FLOAT64_C(  -389.46), EASYSIMD_FLOAT64_C(  -147.82) },
      { EASYSIMD_FLOAT64_C( -1897.00), EASYSIMD_FLOAT64_C(    14.00), EASYSIMD_FLOAT64_C(  -370.00), EASYSIMD_FLOAT64_C(  1368.00),
        EASYSIMD_FLOAT64_C(  -409.00), EASYSIMD_FLOAT64_C(  1282.00), EASYSIMD_FLOAT64_C(  -393.00), EASYSIMD_FLOAT64_C(  -118.00) },
      { EASYSIMD_FLOAT64_C( -1897.00), EASYSIMD_FLOAT64_C(    14.00), EASYSIMD_FLOAT64_C(  -371.00), EASYSIMD_FLOAT64_C(  1367.00),
        EASYSIMD_FLOAT64_C(  -409.00), EASYSIMD_FLOAT64_C(  1281.00), EASYSIMD_FLOAT64_C(  -393.00), EASYSIMD_FLOAT64_C(  -118.00) },
      { EASYSIMD_FLOAT64_C( -1896.00), EASYSIMD_FLOAT64_C(    15.00), EASYSIMD_FLOAT64_C(  -370.00), EASYSIMD_FLOAT64_C(  1368.00),
        EASYSIMD_FLOAT64_C(  -408.00), EASYSIMD_FLOAT64_C(  1282.00), EASYSIMD_FLOAT64_C(  -392.00), EASYSIMD_FLOAT64_C(  -117.00) },
      { EASYSIMD_FLOAT64_C( -1896.00), EASYSIMD_FLOAT64_C(    14.00), EASYSIMD_FLOAT64_C(  -370.00), EASYSIMD_FLOAT64_C(  1367.00),
        EASYSIMD_FLOAT64_C(  -408.00), EASYSIMD_FLOAT64_C(  1281.00), EASYSIMD_FLOAT64_C(  -392.00), EASYSIMD_FLOAT64_C(  -117.00) },
      { EASYSIMD_FLOAT64_C( -1897.00), EASYSIMD_FLOAT64_C(    14.00), EASYSIMD_FLOAT64_C(  -370.00), EASYSIMD_FLOAT64_C(  1368.00),
        EASYSIMD_FLOAT64_C(  -409.00), EASYSIMD_FLOAT64_C(  1282.00), EASYSIMD_FLOAT64_C(  -393.00), EASYSIMD_FLOAT64_C(  -118.00) } },
    { { EASYSIMD_FLOAT64_C(   511.87), EASYSIMD_FLOAT64_C(  -129.84), EASYSIMD_FLOAT64_C(   -76.97), EASYSIMD_FLOAT64_C(  -674.81),
        EASYSIMD_FLOAT64_C(  -894.65), EASYSIMD_FLOAT64_C(   388.02), EASYSIMD_FLOAT64_C(  -544.72), EASYSIMD_FLOAT64_C(    38.86) },
      { EASYSIMD_FLOAT64_C(  -693.35), EASYSIMD_FLOAT64_C(   115.81), EASYSIMD_FLOAT64_C(   509.66), EASYSIMD_FLOAT64_C(   756.37),
        EASYSIMD_FLOAT64_C(  -585.36), EASYSIMD_FLOAT64_C(   188.94), EASYSIMD_FLOAT64_C(  -870.21), EASYSIMD_FLOAT64_C(  -486.12) },
      { EASYSIMD_FLOAT64_C(  -181.00), EASYSIMD_FLOAT64_C(   -14.00), EASYSIMD_FLOAT64_C(   433.00), EASYSIMD_FLOAT64_C(    82.00),
        EASYSIMD_FLOAT64_C( -1480.00), EASYSIMD_FLOAT64_C(   577.00), EASYSIMD_FLOAT64_C( -1415.00), EASYSIMD_FLOAT64_C(  -447.00) },
      { EASYSIMD_FLOAT64_C(  -182.00), EASYSIMD_FLOAT64_C(   -15.00), EASYSIMD_FLOAT64_C(   432.00), EASYSIMD_FLOAT64_C(    81.00),
        EASYSIMD_FLOAT64_C( -1481.00), EASYSIMD_FLOAT64_C(   576.00), EASYSIMD_FLOAT64_C( -1415.00), EASYSIMD_FLOAT64_C(  -448.00) },
      { EASYSIMD_FLOAT64_C(  -181.00), EASYSIMD_FLOAT64_C(   -14.00), EASYSIMD_FLOAT64_C(   433.00), EASYSIMD_FLOAT64_C(    82.00),
        EASYSIMD_FLOAT64_C( -1480.00), EASYSIMD_FLOAT64_C(   577.00), EASYSIMD_FLOAT64_C( -1414.00), EASYSIMD_FLOAT64_C(  -447.00) },
      { EASYSIMD_FLOAT64_C(  -181.00), EASYSIMD_FLOAT64_C(   -14.00), EASYSIMD_FLOAT64_C(   432.00), EASYSIMD_FLOAT64_C(    81.00),
        EASYSIMD_FLOAT64_C( -1480.00), EASYSIMD_FLOAT64_C(   576.00), EASYSIMD_FLOAT64_C( -1414.00), EASYSIMD_FLOAT64_C(  -447.00) },
      { EASYSIMD_FLOAT64_C(  -181.00), EASYSIMD_FLOAT64_C(   -14.00), EASYSIMD_FLOAT64_C(   433.00), EASYSIMD_FLOAT64_C(    82.00),
        EASYSIMD_FLOAT64_C( -1480.00), EASYSIMD_FLOAT64_C(   577.00), EASYSIMD_FLOAT64_C( -1415.00), EASYSIMD_FLOAT64_C(  -447.00) } },
    { { EASYSIMD_FLOAT64_C(   155.14), EASYSIMD_FLOAT64_C(  -555.21), EASYSIMD_FLOAT64_C(  -519.29), EASYSIMD_FLOAT64_C(  -733.17),
        EASYSIMD_FLOAT64_C(  -695.90), EASYSIMD_FLOAT64_C(   477.65), EASYSIMD_FLOAT64_C(   296.86), EASYSIMD_FLOAT64_C(  -691.82) },
      { EASYSIMD_FLOAT64_C(   525.70), EASYSIMD_FLOAT64_C(  -388.32), EASYSIMD_FLOAT64_C(   708.91), EASYSIMD_FLOAT64_C(  -994.51),
        EASYSIMD_FLOAT64_C(  -965.77), EASYSIMD_FLOAT64_C(  -680.55), EASYSIMD_FLOAT64_C(  -142.34), EASYSIMD_FLOAT64_C(   546.10) },
      { EASYSIMD_FLOAT64_C(   681.00), EASYSIMD_FLOAT64_C(  -944.00), EASYSIMD_FLOAT64_C(   190.00), EASYSIMD_FLOAT64_C( -1728.00),
        EASYSIMD_FLOAT64_C( -1662.00), EASYSIMD_FLOAT64_C(  -203.00), EASYSIMD_FLOAT64_C(   155.00), EASYSIMD_FLOAT64_C(  -146.00) },
      { EASYSIMD_FLOAT64_C(   680.00), EASYSIMD_FLOAT64_C(  -944.00), EASYSIMD_FLOAT64_C(   189.00), EASYSIMD_FLOAT64_C( -1728.00),
        EASYSIMD_FLOAT64_C( -1662.00), EASYSIMD_FLOAT64_C(  -203.00), EASYSIMD_FLOAT64_C(   154.00), EASYSIMD_FLOAT64_C(  -146.00) },
      { EASYSIMD_FLOAT64_C(   681.00), EASYSIMD_FLOAT64_C(  -943.00), EASYSIMD_FLOAT64_C(   190.00), EASYSIMD_FLOAT64_C( -1727.00),
        EASYSIMD_FLOAT64_C( -1661.00), EASYSIMD_FLOAT64_C(  -202.00), EASYSIMD_FLOAT64_C(   155.00), EASYSIMD_FLOAT64_C(  -145.00) },
      { EASYSIMD_FLOAT64_C(   680.00), EASYSIMD_FLOAT64_C(  -943.00), EASYSIMD_FLOAT64_C(   189.00), EASYSIMD_FLOAT64_C( -1727.00),
        EASYSIMD_FLOAT64_C( -1661.00), EASYSIMD_FLOAT64_C(  -202.00), EASYSIMD_FLOAT64_C(   154.00), EASYSIMD_FLOAT64_C(  -145.00) },
      { EASYSIMD_FLOAT64_C(   681.00), EASYSIMD_FLOAT64_C(  -944.00), EASYSIMD_FLOAT64_C(   190.00), EASYSIMD_FLOAT64_C( -1728.00),
        EASYSIMD_FLOAT64_C( -1662.00), EASYSIMD_FLOAT64_C(  -203.00), EASYSIMD_FLOAT64_C(   155.00), EASYSIMD_FLOAT64_C(  -146.00) } }
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

    r = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512d_close(r, nearest_inf, 1);

    r = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512d_close(r, neg_inf, 1);

    r = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512d_close(r, pos_inf, 1);

    r = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512d_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_add_round_pd");
    easysimd_assert_m512d_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512d nearest_inf = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512d neg_inf = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512d pos_inf = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512d zero = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512d direction = easysimd_mm512_add_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

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
test_easysimd_mm512_mask_add_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 src[8];
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -241.95), EASYSIMD_FLOAT64_C(   348.31), EASYSIMD_FLOAT64_C(  -125.04), EASYSIMD_FLOAT64_C(  -245.69),
        EASYSIMD_FLOAT64_C(  -588.93), EASYSIMD_FLOAT64_C(  -276.58), EASYSIMD_FLOAT64_C(  -867.91), EASYSIMD_FLOAT64_C(   -10.44) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT64_C(   296.41), EASYSIMD_FLOAT64_C(  -576.00), EASYSIMD_FLOAT64_C(   367.45), EASYSIMD_FLOAT64_C(  -294.17),
        EASYSIMD_FLOAT64_C(  -916.48), EASYSIMD_FLOAT64_C(   266.60), EASYSIMD_FLOAT64_C(   288.43), EASYSIMD_FLOAT64_C(   471.22) },
      { EASYSIMD_FLOAT64_C(  -995.36), EASYSIMD_FLOAT64_C(   468.56), EASYSIMD_FLOAT64_C(    50.02), EASYSIMD_FLOAT64_C(   -51.51),
        EASYSIMD_FLOAT64_C(   997.69), EASYSIMD_FLOAT64_C(   806.19), EASYSIMD_FLOAT64_C(  -145.36), EASYSIMD_FLOAT64_C(   877.33) },
      { EASYSIMD_FLOAT64_C(  -698.96), EASYSIMD_FLOAT64_C(   348.31), EASYSIMD_FLOAT64_C(  -125.04), EASYSIMD_FLOAT64_C(  -245.69),
        EASYSIMD_FLOAT64_C(    81.21), EASYSIMD_FLOAT64_C(  -276.58), EASYSIMD_FLOAT64_C(  -867.91), EASYSIMD_FLOAT64_C(   -10.44) } },
    { { EASYSIMD_FLOAT64_C(  -303.10), EASYSIMD_FLOAT64_C(  -675.79), EASYSIMD_FLOAT64_C(   770.76), EASYSIMD_FLOAT64_C(   600.76),
        EASYSIMD_FLOAT64_C(  -105.79), EASYSIMD_FLOAT64_C(  -257.88), EASYSIMD_FLOAT64_C(  -641.18), EASYSIMD_FLOAT64_C(  -757.48) },
      UINT8_C(183),
      { EASYSIMD_FLOAT64_C(   113.13), EASYSIMD_FLOAT64_C(  -346.41), EASYSIMD_FLOAT64_C(  -659.51), EASYSIMD_FLOAT64_C(   245.22),
        EASYSIMD_FLOAT64_C(   643.14), EASYSIMD_FLOAT64_C(    43.25), EASYSIMD_FLOAT64_C(  -458.37), EASYSIMD_FLOAT64_C(  -932.86) },
      { EASYSIMD_FLOAT64_C(  -589.30), EASYSIMD_FLOAT64_C(   247.46), EASYSIMD_FLOAT64_C(  -849.33), EASYSIMD_FLOAT64_C(   677.31),
        EASYSIMD_FLOAT64_C(  -464.11), EASYSIMD_FLOAT64_C(   621.89), EASYSIMD_FLOAT64_C(   681.94), EASYSIMD_FLOAT64_C(  -995.54) },
      { EASYSIMD_FLOAT64_C(  -476.17), EASYSIMD_FLOAT64_C(   -98.95), EASYSIMD_FLOAT64_C( -1508.84), EASYSIMD_FLOAT64_C(   600.76),
        EASYSIMD_FLOAT64_C(   179.04), EASYSIMD_FLOAT64_C(   665.14), EASYSIMD_FLOAT64_C(  -641.18), EASYSIMD_FLOAT64_C( -1928.40) } },
    { { EASYSIMD_FLOAT64_C(  -328.10), EASYSIMD_FLOAT64_C(  -369.57), EASYSIMD_FLOAT64_C(  -997.86), EASYSIMD_FLOAT64_C(  -521.91),
        EASYSIMD_FLOAT64_C(   485.07), EASYSIMD_FLOAT64_C(   879.48), EASYSIMD_FLOAT64_C(   175.00), EASYSIMD_FLOAT64_C(   809.28) },
      UINT8_C( 91),
      { EASYSIMD_FLOAT64_C(  -224.24), EASYSIMD_FLOAT64_C(  -296.51), EASYSIMD_FLOAT64_C(  -607.64), EASYSIMD_FLOAT64_C(   134.57),
        EASYSIMD_FLOAT64_C(   -53.99), EASYSIMD_FLOAT64_C(  -990.57), EASYSIMD_FLOAT64_C(  -752.30), EASYSIMD_FLOAT64_C(   599.60) },
      { EASYSIMD_FLOAT64_C(  -650.08), EASYSIMD_FLOAT64_C(   492.93), EASYSIMD_FLOAT64_C(   242.74), EASYSIMD_FLOAT64_C(   393.17),
        EASYSIMD_FLOAT64_C(  -965.44), EASYSIMD_FLOAT64_C(   309.89), EASYSIMD_FLOAT64_C(   803.88), EASYSIMD_FLOAT64_C(   282.02) },
      { EASYSIMD_FLOAT64_C(  -874.32), EASYSIMD_FLOAT64_C(   196.42), EASYSIMD_FLOAT64_C(  -997.86), EASYSIMD_FLOAT64_C(   527.75),
        EASYSIMD_FLOAT64_C( -1019.43), EASYSIMD_FLOAT64_C(   879.48), EASYSIMD_FLOAT64_C(    51.58), EASYSIMD_FLOAT64_C(   809.28) } },
    { { EASYSIMD_FLOAT64_C(   460.56), EASYSIMD_FLOAT64_C(   481.18), EASYSIMD_FLOAT64_C(   817.91), EASYSIMD_FLOAT64_C(    82.44),
        EASYSIMD_FLOAT64_C(   163.12), EASYSIMD_FLOAT64_C(   822.36), EASYSIMD_FLOAT64_C(   754.35), EASYSIMD_FLOAT64_C(   793.56) },
      UINT8_C( 35),
      { EASYSIMD_FLOAT64_C(  -767.56), EASYSIMD_FLOAT64_C(   278.63), EASYSIMD_FLOAT64_C(   703.98), EASYSIMD_FLOAT64_C(   407.44),
        EASYSIMD_FLOAT64_C(    87.91), EASYSIMD_FLOAT64_C(   354.22), EASYSIMD_FLOAT64_C(  -816.81), EASYSIMD_FLOAT64_C(   791.41) },
      { EASYSIMD_FLOAT64_C(   746.58), EASYSIMD_FLOAT64_C(   317.77), EASYSIMD_FLOAT64_C(  -262.58), EASYSIMD_FLOAT64_C(   756.01),
        EASYSIMD_FLOAT64_C(   565.47), EASYSIMD_FLOAT64_C(  -662.99), EASYSIMD_FLOAT64_C(  -894.07), EASYSIMD_FLOAT64_C(    58.40) },
      { EASYSIMD_FLOAT64_C(   -20.98), EASYSIMD_FLOAT64_C(   596.40), EASYSIMD_FLOAT64_C(   817.91), EASYSIMD_FLOAT64_C(    82.44),
        EASYSIMD_FLOAT64_C(   163.12), EASYSIMD_FLOAT64_C(  -308.76), EASYSIMD_FLOAT64_C(   754.35), EASYSIMD_FLOAT64_C(   793.56) } },
    { { EASYSIMD_FLOAT64_C(   579.76), EASYSIMD_FLOAT64_C(   499.11), EASYSIMD_FLOAT64_C(    92.96), EASYSIMD_FLOAT64_C(  -110.35),
        EASYSIMD_FLOAT64_C(   302.99), EASYSIMD_FLOAT64_C(  -625.02), EASYSIMD_FLOAT64_C(  -649.80), EASYSIMD_FLOAT64_C(  -215.83) },
      UINT8_C(  3),
      { EASYSIMD_FLOAT64_C(   432.65), EASYSIMD_FLOAT64_C(   947.29), EASYSIMD_FLOAT64_C(  -984.75), EASYSIMD_FLOAT64_C(   186.99),
        EASYSIMD_FLOAT64_C(   740.85), EASYSIMD_FLOAT64_C(   839.76), EASYSIMD_FLOAT64_C(   419.43), EASYSIMD_FLOAT64_C(    19.48) },
      { EASYSIMD_FLOAT64_C(   543.74), EASYSIMD_FLOAT64_C(  -173.13), EASYSIMD_FLOAT64_C(  -892.61), EASYSIMD_FLOAT64_C(  -102.04),
        EASYSIMD_FLOAT64_C(    10.06), EASYSIMD_FLOAT64_C(   898.80), EASYSIMD_FLOAT64_C(  -355.45), EASYSIMD_FLOAT64_C(  -672.17) },
      { EASYSIMD_FLOAT64_C(   976.39), EASYSIMD_FLOAT64_C(   774.16), EASYSIMD_FLOAT64_C(    92.96), EASYSIMD_FLOAT64_C(  -110.35),
        EASYSIMD_FLOAT64_C(   302.99), EASYSIMD_FLOAT64_C(  -625.02), EASYSIMD_FLOAT64_C(  -649.80), EASYSIMD_FLOAT64_C(  -215.83) } },
    { { EASYSIMD_FLOAT64_C(  -363.79), EASYSIMD_FLOAT64_C(  -599.44), EASYSIMD_FLOAT64_C(   893.30), EASYSIMD_FLOAT64_C(   -26.77),
        EASYSIMD_FLOAT64_C(  -493.51), EASYSIMD_FLOAT64_C(   -48.30), EASYSIMD_FLOAT64_C(  -447.01), EASYSIMD_FLOAT64_C(  -994.40) },
      UINT8_C( 89),
      { EASYSIMD_FLOAT64_C(   442.63), EASYSIMD_FLOAT64_C(   308.59), EASYSIMD_FLOAT64_C(  -580.36), EASYSIMD_FLOAT64_C(   792.84),
        EASYSIMD_FLOAT64_C(  -907.24), EASYSIMD_FLOAT64_C(  -387.48), EASYSIMD_FLOAT64_C(   225.48), EASYSIMD_FLOAT64_C(  -959.95) },
      { EASYSIMD_FLOAT64_C(  -372.23), EASYSIMD_FLOAT64_C(  -587.52), EASYSIMD_FLOAT64_C(   780.90), EASYSIMD_FLOAT64_C(  -532.47),
        EASYSIMD_FLOAT64_C(   831.91), EASYSIMD_FLOAT64_C(  -199.62), EASYSIMD_FLOAT64_C(  -988.73), EASYSIMD_FLOAT64_C(  -341.22) },
      { EASYSIMD_FLOAT64_C(    70.40), EASYSIMD_FLOAT64_C(  -599.44), EASYSIMD_FLOAT64_C(   893.30), EASYSIMD_FLOAT64_C(   260.36),
        EASYSIMD_FLOAT64_C(   -75.33), EASYSIMD_FLOAT64_C(   -48.30), EASYSIMD_FLOAT64_C(  -763.25), EASYSIMD_FLOAT64_C(  -994.40) } },
    { { EASYSIMD_FLOAT64_C(   -92.23), EASYSIMD_FLOAT64_C(   -90.77), EASYSIMD_FLOAT64_C(   668.84), EASYSIMD_FLOAT64_C(  -193.43),
        EASYSIMD_FLOAT64_C(   553.78), EASYSIMD_FLOAT64_C(   996.67), EASYSIMD_FLOAT64_C(   442.78), EASYSIMD_FLOAT64_C(   954.34) },
      UINT8_C(200),
      { EASYSIMD_FLOAT64_C(  -583.99), EASYSIMD_FLOAT64_C(  -539.17), EASYSIMD_FLOAT64_C(  -158.32), EASYSIMD_FLOAT64_C(   -31.00),
        EASYSIMD_FLOAT64_C(  -533.56), EASYSIMD_FLOAT64_C(  -113.65), EASYSIMD_FLOAT64_C(  -588.37), EASYSIMD_FLOAT64_C(   775.02) },
      { EASYSIMD_FLOAT64_C(   305.99), EASYSIMD_FLOAT64_C(  -795.53), EASYSIMD_FLOAT64_C(   867.78), EASYSIMD_FLOAT64_C(   918.51),
        EASYSIMD_FLOAT64_C(   429.95), EASYSIMD_FLOAT64_C(   907.83), EASYSIMD_FLOAT64_C(  -453.72), EASYSIMD_FLOAT64_C(   842.43) },
      { EASYSIMD_FLOAT64_C(   -92.23), EASYSIMD_FLOAT64_C(   -90.77), EASYSIMD_FLOAT64_C(   668.84), EASYSIMD_FLOAT64_C(   887.51),
        EASYSIMD_FLOAT64_C(   553.78), EASYSIMD_FLOAT64_C(   996.67), EASYSIMD_FLOAT64_C( -1042.09), EASYSIMD_FLOAT64_C(  1617.45) } },
    { { EASYSIMD_FLOAT64_C(   688.73), EASYSIMD_FLOAT64_C(    13.81), EASYSIMD_FLOAT64_C(   674.34), EASYSIMD_FLOAT64_C(  -510.89),
        EASYSIMD_FLOAT64_C(    25.08), EASYSIMD_FLOAT64_C(  -666.88), EASYSIMD_FLOAT64_C(   396.88), EASYSIMD_FLOAT64_C(   934.31) },
      UINT8_C(155),
      { EASYSIMD_FLOAT64_C(  -796.55), EASYSIMD_FLOAT64_C(   488.09), EASYSIMD_FLOAT64_C(   998.63), EASYSIMD_FLOAT64_C(   646.24),
        EASYSIMD_FLOAT64_C(   442.43), EASYSIMD_FLOAT64_C(   888.61), EASYSIMD_FLOAT64_C(  -937.75), EASYSIMD_FLOAT64_C(   903.26) },
      { EASYSIMD_FLOAT64_C(  -269.71), EASYSIMD_FLOAT64_C(    31.25), EASYSIMD_FLOAT64_C(  -630.30), EASYSIMD_FLOAT64_C(   616.64),
        EASYSIMD_FLOAT64_C(   442.88), EASYSIMD_FLOAT64_C(  -855.28), EASYSIMD_FLOAT64_C(   -77.38), EASYSIMD_FLOAT64_C(   647.35) },
      { EASYSIMD_FLOAT64_C( -1066.26), EASYSIMD_FLOAT64_C(   519.34), EASYSIMD_FLOAT64_C(   674.34), EASYSIMD_FLOAT64_C(  1262.87),
        EASYSIMD_FLOAT64_C(   885.31), EASYSIMD_FLOAT64_C(  -666.88), EASYSIMD_FLOAT64_C(   396.88), EASYSIMD_FLOAT64_C(  1550.61) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d src = easysimd_mm512_loadu_pd(test_vec[i].src);
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d r = easysimd_mm512_mask_add_pd(src, test_vec[i].k, a, b);
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_add_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { UINT8_C(141),
      { EASYSIMD_FLOAT64_C(   539.39), EASYSIMD_FLOAT64_C(   127.65), EASYSIMD_FLOAT64_C(  -419.83), EASYSIMD_FLOAT64_C(  -509.25),
        EASYSIMD_FLOAT64_C(   614.81), EASYSIMD_FLOAT64_C(  -356.87), EASYSIMD_FLOAT64_C(  -437.81), EASYSIMD_FLOAT64_C(   217.95) },
      { EASYSIMD_FLOAT64_C(   -60.15), EASYSIMD_FLOAT64_C(  -699.30), EASYSIMD_FLOAT64_C(   963.74), EASYSIMD_FLOAT64_C(   851.36),
        EASYSIMD_FLOAT64_C(   773.07), EASYSIMD_FLOAT64_C(  -457.96), EASYSIMD_FLOAT64_C(  -310.92), EASYSIMD_FLOAT64_C(   852.62) },
      { EASYSIMD_FLOAT64_C(   479.24), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   543.91), EASYSIMD_FLOAT64_C(   342.11),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  1070.58) } },
    { UINT8_C(123),
      { EASYSIMD_FLOAT64_C(   902.15), EASYSIMD_FLOAT64_C(   661.09), EASYSIMD_FLOAT64_C(  -493.90), EASYSIMD_FLOAT64_C(   433.62),
        EASYSIMD_FLOAT64_C(  -884.72), EASYSIMD_FLOAT64_C(  -690.47), EASYSIMD_FLOAT64_C(  -391.44), EASYSIMD_FLOAT64_C(   -97.69) },
      { EASYSIMD_FLOAT64_C(  -732.29), EASYSIMD_FLOAT64_C(   446.84), EASYSIMD_FLOAT64_C(  -990.19), EASYSIMD_FLOAT64_C(   216.62),
        EASYSIMD_FLOAT64_C(  -720.09), EASYSIMD_FLOAT64_C(    35.61), EASYSIMD_FLOAT64_C(  -243.99), EASYSIMD_FLOAT64_C(   407.56) },
      { EASYSIMD_FLOAT64_C(   169.86), EASYSIMD_FLOAT64_C(  1107.93), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   650.24),
        EASYSIMD_FLOAT64_C( -1604.81), EASYSIMD_FLOAT64_C(  -654.86), EASYSIMD_FLOAT64_C(  -635.42), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(175),
      { EASYSIMD_FLOAT64_C(   246.76), EASYSIMD_FLOAT64_C(    22.37), EASYSIMD_FLOAT64_C(  -741.09), EASYSIMD_FLOAT64_C(   808.94),
        EASYSIMD_FLOAT64_C(  -759.68), EASYSIMD_FLOAT64_C(   198.75), EASYSIMD_FLOAT64_C(  -890.36), EASYSIMD_FLOAT64_C(  -795.93) },
      { EASYSIMD_FLOAT64_C(    50.12), EASYSIMD_FLOAT64_C(   882.71), EASYSIMD_FLOAT64_C(  -253.90), EASYSIMD_FLOAT64_C(   739.19),
        EASYSIMD_FLOAT64_C(   735.33), EASYSIMD_FLOAT64_C(   572.27), EASYSIMD_FLOAT64_C(   641.34), EASYSIMD_FLOAT64_C(   396.42) },
      { EASYSIMD_FLOAT64_C(   296.87), EASYSIMD_FLOAT64_C(   905.08), EASYSIMD_FLOAT64_C(  -994.99), EASYSIMD_FLOAT64_C(  1548.14),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   771.02), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -399.51) } },
    { UINT8_C( 10),
      { EASYSIMD_FLOAT64_C(    74.96), EASYSIMD_FLOAT64_C(   511.70), EASYSIMD_FLOAT64_C(  -612.10), EASYSIMD_FLOAT64_C(   683.53),
        EASYSIMD_FLOAT64_C(  -585.99), EASYSIMD_FLOAT64_C(  -344.39), EASYSIMD_FLOAT64_C(   130.37), EASYSIMD_FLOAT64_C(  -576.18) },
      { EASYSIMD_FLOAT64_C(   872.23), EASYSIMD_FLOAT64_C(   410.28), EASYSIMD_FLOAT64_C(   459.43), EASYSIMD_FLOAT64_C(  -371.75),
        EASYSIMD_FLOAT64_C(  -182.16), EASYSIMD_FLOAT64_C(    75.20), EASYSIMD_FLOAT64_C(   875.00), EASYSIMD_FLOAT64_C(   840.21) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   921.98), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   311.77),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 10),
      { EASYSIMD_FLOAT64_C(   683.95), EASYSIMD_FLOAT64_C(  -919.47), EASYSIMD_FLOAT64_C(  -467.14), EASYSIMD_FLOAT64_C(   793.59),
        EASYSIMD_FLOAT64_C(  -715.40), EASYSIMD_FLOAT64_C(   582.98), EASYSIMD_FLOAT64_C(   676.29), EASYSIMD_FLOAT64_C(    30.70) },
      { EASYSIMD_FLOAT64_C(   322.17), EASYSIMD_FLOAT64_C(   411.62), EASYSIMD_FLOAT64_C(  -397.03), EASYSIMD_FLOAT64_C(   -36.48),
        EASYSIMD_FLOAT64_C(  -191.96), EASYSIMD_FLOAT64_C(  -318.66), EASYSIMD_FLOAT64_C(  -961.52), EASYSIMD_FLOAT64_C(  -680.25) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -507.84), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   757.10),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(199),
      { EASYSIMD_FLOAT64_C(   722.01), EASYSIMD_FLOAT64_C(  -266.24), EASYSIMD_FLOAT64_C(   724.85), EASYSIMD_FLOAT64_C(  -147.62),
        EASYSIMD_FLOAT64_C(   157.58), EASYSIMD_FLOAT64_C(   597.08), EASYSIMD_FLOAT64_C(  -737.35), EASYSIMD_FLOAT64_C(  -383.00) },
      { EASYSIMD_FLOAT64_C(  -774.68), EASYSIMD_FLOAT64_C(    80.49), EASYSIMD_FLOAT64_C(   692.21), EASYSIMD_FLOAT64_C(  -899.67),
        EASYSIMD_FLOAT64_C(   -79.30), EASYSIMD_FLOAT64_C(    26.32), EASYSIMD_FLOAT64_C(   784.27), EASYSIMD_FLOAT64_C(     1.24) },
      { EASYSIMD_FLOAT64_C(   -52.67), EASYSIMD_FLOAT64_C(  -185.75), EASYSIMD_FLOAT64_C(  1417.06), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    46.93), EASYSIMD_FLOAT64_C(  -381.76) } },
    { UINT8_C(108),
      { EASYSIMD_FLOAT64_C(   577.86), EASYSIMD_FLOAT64_C(   285.84), EASYSIMD_FLOAT64_C(   142.16), EASYSIMD_FLOAT64_C(   254.16),
        EASYSIMD_FLOAT64_C(  -683.46), EASYSIMD_FLOAT64_C(  -535.67), EASYSIMD_FLOAT64_C(  -334.22), EASYSIMD_FLOAT64_C(   -80.49) },
      { EASYSIMD_FLOAT64_C(   427.85), EASYSIMD_FLOAT64_C(   473.82), EASYSIMD_FLOAT64_C(   600.85), EASYSIMD_FLOAT64_C(   466.33),
        EASYSIMD_FLOAT64_C(   793.57), EASYSIMD_FLOAT64_C(  -329.91), EASYSIMD_FLOAT64_C(   188.34), EASYSIMD_FLOAT64_C(  -472.67) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   743.02), EASYSIMD_FLOAT64_C(   720.48),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -865.57), EASYSIMD_FLOAT64_C(  -145.89), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 48),
      { EASYSIMD_FLOAT64_C(  -959.29), EASYSIMD_FLOAT64_C(   684.90), EASYSIMD_FLOAT64_C(   992.02), EASYSIMD_FLOAT64_C(  -696.63),
        EASYSIMD_FLOAT64_C(  -698.09), EASYSIMD_FLOAT64_C(  -782.66), EASYSIMD_FLOAT64_C(   383.86), EASYSIMD_FLOAT64_C(   994.11) },
      { EASYSIMD_FLOAT64_C(  -682.33), EASYSIMD_FLOAT64_C(  -695.44), EASYSIMD_FLOAT64_C(    20.43), EASYSIMD_FLOAT64_C(  -898.06),
        EASYSIMD_FLOAT64_C(   305.80), EASYSIMD_FLOAT64_C(  -420.39), EASYSIMD_FLOAT64_C(   679.80), EASYSIMD_FLOAT64_C(  -408.37) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(  -392.30), EASYSIMD_FLOAT64_C( -1203.04), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_add_pd(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_add_pd");
    #if defined(__EMSCRIPTEN__)
    (void) r;
    #else
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
    #endif
  }

  return 0;
}

static int
test_easysimd_mm512_addn_ps (EASYSIMD_MUNIT_TEST_ARGS) {
    #if 1
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   499.28), EASYSIMD_FLOAT32_C(   359.58), EASYSIMD_FLOAT32_C(   877.23), EASYSIMD_FLOAT32_C(  -540.17),
        EASYSIMD_FLOAT32_C(    48.01), EASYSIMD_FLOAT32_C(   198.12), EASYSIMD_FLOAT32_C(   542.60), EASYSIMD_FLOAT32_C(   808.54),
        EASYSIMD_FLOAT32_C(  -730.88), EASYSIMD_FLOAT32_C(  -679.34), EASYSIMD_FLOAT32_C(   320.00), EASYSIMD_FLOAT32_C(  -619.19),
        EASYSIMD_FLOAT32_C(   221.39), EASYSIMD_FLOAT32_C(  -818.66), EASYSIMD_FLOAT32_C(  -411.27), EASYSIMD_FLOAT32_C(   433.72) },
      { EASYSIMD_FLOAT32_C(  -850.51), EASYSIMD_FLOAT32_C(    80.93), EASYSIMD_FLOAT32_C(  -409.14), EASYSIMD_FLOAT32_C(   357.36),
        EASYSIMD_FLOAT32_C(   718.96), EASYSIMD_FLOAT32_C(  -604.43), EASYSIMD_FLOAT32_C(   907.01), EASYSIMD_FLOAT32_C(  -514.99),
        EASYSIMD_FLOAT32_C(  -522.68), EASYSIMD_FLOAT32_C(  -753.12), EASYSIMD_FLOAT32_C(  -548.85), EASYSIMD_FLOAT32_C(  -248.34),
        EASYSIMD_FLOAT32_C(   705.21), EASYSIMD_FLOAT32_C(   725.38), EASYSIMD_FLOAT32_C(  -383.67), EASYSIMD_FLOAT32_C(   204.49) },
      { EASYSIMD_FLOAT32_C(   351.23), EASYSIMD_FLOAT32_C(  -440.51), EASYSIMD_FLOAT32_C(  -468.09), EASYSIMD_FLOAT32_C(   182.81),
        EASYSIMD_FLOAT32_C(  -766.97), EASYSIMD_FLOAT32_C(   406.31), EASYSIMD_FLOAT32_C( -1449.61), EASYSIMD_FLOAT32_C(  -293.55),
        EASYSIMD_FLOAT32_C(  1253.56), EASYSIMD_FLOAT32_C(  1432.46), EASYSIMD_FLOAT32_C(   228.85), EASYSIMD_FLOAT32_C(   867.53),
        EASYSIMD_FLOAT32_C(  -926.60), EASYSIMD_FLOAT32_C(    93.28), EASYSIMD_FLOAT32_C(   794.94), EASYSIMD_FLOAT32_C(  -638.21) } },
    { { EASYSIMD_FLOAT32_C(    84.96), EASYSIMD_FLOAT32_C(  -506.44), EASYSIMD_FLOAT32_C(   664.32), EASYSIMD_FLOAT32_C(  -867.03),
        EASYSIMD_FLOAT32_C(   691.68), EASYSIMD_FLOAT32_C(   206.92), EASYSIMD_FLOAT32_C(   941.51), EASYSIMD_FLOAT32_C(   960.80),
        EASYSIMD_FLOAT32_C(   527.59), EASYSIMD_FLOAT32_C(   261.51), EASYSIMD_FLOAT32_C(  -658.40), EASYSIMD_FLOAT32_C(  -251.02),
        EASYSIMD_FLOAT32_C(   442.85), EASYSIMD_FLOAT32_C(   -69.66), EASYSIMD_FLOAT32_C(  -817.30), EASYSIMD_FLOAT32_C(   592.34) },
      { EASYSIMD_FLOAT32_C(  -988.74), EASYSIMD_FLOAT32_C(  -226.45), EASYSIMD_FLOAT32_C(   -50.30), EASYSIMD_FLOAT32_C(   730.22),
        EASYSIMD_FLOAT32_C(   169.12), EASYSIMD_FLOAT32_C(  -143.29), EASYSIMD_FLOAT32_C(  -784.76), EASYSIMD_FLOAT32_C(   646.44),
        EASYSIMD_FLOAT32_C(   103.59), EASYSIMD_FLOAT32_C(  -333.62), EASYSIMD_FLOAT32_C(  -601.90), EASYSIMD_FLOAT32_C(  -191.21),
        EASYSIMD_FLOAT32_C(  -608.24), EASYSIMD_FLOAT32_C(    14.43), EASYSIMD_FLOAT32_C(  -986.72), EASYSIMD_FLOAT32_C(   476.72) },
      { EASYSIMD_FLOAT32_C(   903.78), EASYSIMD_FLOAT32_C(   732.89), EASYSIMD_FLOAT32_C(  -614.02), EASYSIMD_FLOAT32_C(   136.81),
        EASYSIMD_FLOAT32_C(  -860.80), EASYSIMD_FLOAT32_C(   -63.63), EASYSIMD_FLOAT32_C(  -156.75), EASYSIMD_FLOAT32_C( -1607.24),
        EASYSIMD_FLOAT32_C(  -631.18), EASYSIMD_FLOAT32_C(    72.11), EASYSIMD_FLOAT32_C(  1260.30), EASYSIMD_FLOAT32_C(   442.23),
        EASYSIMD_FLOAT32_C(   165.39), EASYSIMD_FLOAT32_C(    55.23), EASYSIMD_FLOAT32_C(  1804.02), EASYSIMD_FLOAT32_C( -1069.06) } },
    { { EASYSIMD_FLOAT32_C(   507.98), EASYSIMD_FLOAT32_C(   677.60), EASYSIMD_FLOAT32_C(   609.69), EASYSIMD_FLOAT32_C(   199.66),
        EASYSIMD_FLOAT32_C(  -115.47), EASYSIMD_FLOAT32_C(   551.20), EASYSIMD_FLOAT32_C(   160.46), EASYSIMD_FLOAT32_C(  -587.89),
        EASYSIMD_FLOAT32_C(  -187.29), EASYSIMD_FLOAT32_C(   502.06), EASYSIMD_FLOAT32_C(   161.09), EASYSIMD_FLOAT32_C(  -744.44),
        EASYSIMD_FLOAT32_C(  -567.60), EASYSIMD_FLOAT32_C(   343.79), EASYSIMD_FLOAT32_C(   847.90), EASYSIMD_FLOAT32_C(  -556.34) },
      { EASYSIMD_FLOAT32_C(  -882.66), EASYSIMD_FLOAT32_C(  -202.40), EASYSIMD_FLOAT32_C(  -826.12), EASYSIMD_FLOAT32_C(   286.46),
        EASYSIMD_FLOAT32_C(   654.31), EASYSIMD_FLOAT32_C(  -610.88), EASYSIMD_FLOAT32_C(   -67.10), EASYSIMD_FLOAT32_C(  -242.10),
        EASYSIMD_FLOAT32_C(    55.50), EASYSIMD_FLOAT32_C(   330.99), EASYSIMD_FLOAT32_C(   566.69), EASYSIMD_FLOAT32_C(   447.26),
        EASYSIMD_FLOAT32_C(  -654.58), EASYSIMD_FLOAT32_C(   579.97), EASYSIMD_FLOAT32_C(   -76.01), EASYSIMD_FLOAT32_C(   853.40) },
      { EASYSIMD_FLOAT32_C(   374.68), EASYSIMD_FLOAT32_C(  -475.20), EASYSIMD_FLOAT32_C(   216.43), EASYSIMD_FLOAT32_C(  -486.12),
        EASYSIMD_FLOAT32_C(  -538.84), EASYSIMD_FLOAT32_C(    59.68), EASYSIMD_FLOAT32_C(   -93.36), EASYSIMD_FLOAT32_C(   829.99),
        EASYSIMD_FLOAT32_C(   131.79), EASYSIMD_FLOAT32_C(  -833.05), EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   297.18),
        EASYSIMD_FLOAT32_C(  1222.18), EASYSIMD_FLOAT32_C(  -923.76), EASYSIMD_FLOAT32_C(  -771.89), EASYSIMD_FLOAT32_C(  -297.06) } },
    { { EASYSIMD_FLOAT32_C(   257.57), EASYSIMD_FLOAT32_C(  -466.33), EASYSIMD_FLOAT32_C(    53.06), EASYSIMD_FLOAT32_C(  -857.90),
        EASYSIMD_FLOAT32_C(  -915.13), EASYSIMD_FLOAT32_C(  -786.49), EASYSIMD_FLOAT32_C(  -445.79), EASYSIMD_FLOAT32_C(  -102.42),
        EASYSIMD_FLOAT32_C(   715.57), EASYSIMD_FLOAT32_C(   715.30), EASYSIMD_FLOAT32_C(   153.14), EASYSIMD_FLOAT32_C(  -852.03),
        EASYSIMD_FLOAT32_C(    59.09), EASYSIMD_FLOAT32_C(     1.05), EASYSIMD_FLOAT32_C(  -408.37), EASYSIMD_FLOAT32_C(   176.43) },
      { EASYSIMD_FLOAT32_C(   798.65), EASYSIMD_FLOAT32_C(  -234.48), EASYSIMD_FLOAT32_C(  -537.11), EASYSIMD_FLOAT32_C(   452.96),
        EASYSIMD_FLOAT32_C(   154.64), EASYSIMD_FLOAT32_C(   395.79), EASYSIMD_FLOAT32_C(  -789.14), EASYSIMD_FLOAT32_C(  -789.86),
        EASYSIMD_FLOAT32_C(  -273.22), EASYSIMD_FLOAT32_C(   777.55), EASYSIMD_FLOAT32_C(   657.40), EASYSIMD_FLOAT32_C(    72.20),
        EASYSIMD_FLOAT32_C(   357.52), EASYSIMD_FLOAT32_C(  -418.61), EASYSIMD_FLOAT32_C(   -74.40), EASYSIMD_FLOAT32_C(  -384.91) },
      { EASYSIMD_FLOAT32_C( -1056.22), EASYSIMD_FLOAT32_C(   700.81), EASYSIMD_FLOAT32_C(   484.05), EASYSIMD_FLOAT32_C(   404.94),
        EASYSIMD_FLOAT32_C(   760.49), EASYSIMD_FLOAT32_C(   390.70), EASYSIMD_FLOAT32_C(  1234.93), EASYSIMD_FLOAT32_C(   892.28),
        EASYSIMD_FLOAT32_C(  -442.35), EASYSIMD_FLOAT32_C( -1492.85), EASYSIMD_FLOAT32_C(  -810.54), EASYSIMD_FLOAT32_C(   779.83),
        EASYSIMD_FLOAT32_C(  -416.61), EASYSIMD_FLOAT32_C(   417.56), EASYSIMD_FLOAT32_C(   482.77), EASYSIMD_FLOAT32_C(   208.48) } },
    { { EASYSIMD_FLOAT32_C(   115.07), EASYSIMD_FLOAT32_C(   978.65), EASYSIMD_FLOAT32_C(  -242.81), EASYSIMD_FLOAT32_C(   199.94),
        EASYSIMD_FLOAT32_C(  -807.83), EASYSIMD_FLOAT32_C(   311.40), EASYSIMD_FLOAT32_C(  -902.48), EASYSIMD_FLOAT32_C(   907.74),
        EASYSIMD_FLOAT32_C(    26.71), EASYSIMD_FLOAT32_C(   250.67), EASYSIMD_FLOAT32_C(  -944.29), EASYSIMD_FLOAT32_C(  -914.20),
        EASYSIMD_FLOAT32_C(  -748.28), EASYSIMD_FLOAT32_C(  -352.66), EASYSIMD_FLOAT32_C(   262.23), EASYSIMD_FLOAT32_C(  -949.63) },
      { EASYSIMD_FLOAT32_C(   412.86), EASYSIMD_FLOAT32_C(   725.13), EASYSIMD_FLOAT32_C(   503.33), EASYSIMD_FLOAT32_C(  -432.51),
        EASYSIMD_FLOAT32_C(   120.91), EASYSIMD_FLOAT32_C(   714.19), EASYSIMD_FLOAT32_C(  -222.37), EASYSIMD_FLOAT32_C(   847.70),
        EASYSIMD_FLOAT32_C(   491.73), EASYSIMD_FLOAT32_C(  -564.96), EASYSIMD_FLOAT32_C(   -80.11), EASYSIMD_FLOAT32_C(  -150.75),
        EASYSIMD_FLOAT32_C(    16.43), EASYSIMD_FLOAT32_C(   845.49), EASYSIMD_FLOAT32_C(   464.34), EASYSIMD_FLOAT32_C(  -868.51) },
      { EASYSIMD_FLOAT32_C(  -527.93), EASYSIMD_FLOAT32_C( -1703.78), EASYSIMD_FLOAT32_C(  -260.52), EASYSIMD_FLOAT32_C(   232.57),
        EASYSIMD_FLOAT32_C(   686.92), EASYSIMD_FLOAT32_C( -1025.59), EASYSIMD_FLOAT32_C(  1124.85), EASYSIMD_FLOAT32_C( -1755.44),
        EASYSIMD_FLOAT32_C(  -518.44), EASYSIMD_FLOAT32_C(   314.29), EASYSIMD_FLOAT32_C(  1024.40), EASYSIMD_FLOAT32_C(  1064.95),
        EASYSIMD_FLOAT32_C(   731.85), EASYSIMD_FLOAT32_C(  -492.83), EASYSIMD_FLOAT32_C(  -726.57), EASYSIMD_FLOAT32_C(  1818.14) } },
    { { EASYSIMD_FLOAT32_C(   824.15), EASYSIMD_FLOAT32_C(  -778.47), EASYSIMD_FLOAT32_C(   331.43), EASYSIMD_FLOAT32_C(  -983.69),
        EASYSIMD_FLOAT32_C(   532.93), EASYSIMD_FLOAT32_C(   428.96), EASYSIMD_FLOAT32_C(   924.05), EASYSIMD_FLOAT32_C(  -440.36),
        EASYSIMD_FLOAT32_C(  -320.37), EASYSIMD_FLOAT32_C(   979.76), EASYSIMD_FLOAT32_C(  -354.56), EASYSIMD_FLOAT32_C(   -68.66),
        EASYSIMD_FLOAT32_C(  -372.90), EASYSIMD_FLOAT32_C(   907.68), EASYSIMD_FLOAT32_C(   -18.29), EASYSIMD_FLOAT32_C(  -960.04) },
      { EASYSIMD_FLOAT32_C(   632.80), EASYSIMD_FLOAT32_C(  -514.96), EASYSIMD_FLOAT32_C(  -392.55), EASYSIMD_FLOAT32_C(  -246.28),
        EASYSIMD_FLOAT32_C(  -800.78), EASYSIMD_FLOAT32_C(   385.09), EASYSIMD_FLOAT32_C(  -398.59), EASYSIMD_FLOAT32_C(   690.95),
        EASYSIMD_FLOAT32_C(   820.12), EASYSIMD_FLOAT32_C(   521.31), EASYSIMD_FLOAT32_C(  -459.80), EASYSIMD_FLOAT32_C(  -163.45),
        EASYSIMD_FLOAT32_C(   366.80), EASYSIMD_FLOAT32_C(  -995.46), EASYSIMD_FLOAT32_C(   -31.95), EASYSIMD_FLOAT32_C(   190.95) },
      { EASYSIMD_FLOAT32_C( -1456.95), EASYSIMD_FLOAT32_C(  1293.43), EASYSIMD_FLOAT32_C(    61.12), EASYSIMD_FLOAT32_C(  1229.97),
        EASYSIMD_FLOAT32_C(   267.85), EASYSIMD_FLOAT32_C(  -814.05), EASYSIMD_FLOAT32_C(  -525.46), EASYSIMD_FLOAT32_C(  -250.59),
        EASYSIMD_FLOAT32_C(  -499.75), EASYSIMD_FLOAT32_C( -1501.07), EASYSIMD_FLOAT32_C(   814.36), EASYSIMD_FLOAT32_C(   232.11),
        EASYSIMD_FLOAT32_C(     6.10), EASYSIMD_FLOAT32_C(    87.78), EASYSIMD_FLOAT32_C(    50.24), EASYSIMD_FLOAT32_C(   769.09) } },
    { { EASYSIMD_FLOAT32_C(  -773.93), EASYSIMD_FLOAT32_C(  -700.52), EASYSIMD_FLOAT32_C(   207.26), EASYSIMD_FLOAT32_C(   759.00),
        EASYSIMD_FLOAT32_C(   728.44), EASYSIMD_FLOAT32_C(   131.31), EASYSIMD_FLOAT32_C(  -681.35), EASYSIMD_FLOAT32_C(  -591.94),
        EASYSIMD_FLOAT32_C(   111.08), EASYSIMD_FLOAT32_C(   -35.91), EASYSIMD_FLOAT32_C(   339.40), EASYSIMD_FLOAT32_C(   738.18),
        EASYSIMD_FLOAT32_C(  -128.23), EASYSIMD_FLOAT32_C(  -678.89), EASYSIMD_FLOAT32_C(   778.14), EASYSIMD_FLOAT32_C(  -495.43) },
      { EASYSIMD_FLOAT32_C(  -193.85), EASYSIMD_FLOAT32_C(  -614.40), EASYSIMD_FLOAT32_C(   258.29), EASYSIMD_FLOAT32_C(     5.37),
        EASYSIMD_FLOAT32_C(   770.68), EASYSIMD_FLOAT32_C(   859.70), EASYSIMD_FLOAT32_C(  -303.68), EASYSIMD_FLOAT32_C(   590.81),
        EASYSIMD_FLOAT32_C(   381.01), EASYSIMD_FLOAT32_C(   236.52), EASYSIMD_FLOAT32_C(  -572.64), EASYSIMD_FLOAT32_C(  -252.19),
        EASYSIMD_FLOAT32_C(   241.06), EASYSIMD_FLOAT32_C(   395.41), EASYSIMD_FLOAT32_C(   938.76), EASYSIMD_FLOAT32_C(   467.13) },
      { EASYSIMD_FLOAT32_C(   967.78), EASYSIMD_FLOAT32_C(  1314.92), EASYSIMD_FLOAT32_C(  -465.55), EASYSIMD_FLOAT32_C(  -764.37),
        EASYSIMD_FLOAT32_C( -1499.12), EASYSIMD_FLOAT32_C(  -991.01), EASYSIMD_FLOAT32_C(   985.03), EASYSIMD_FLOAT32_C(     1.13),
        EASYSIMD_FLOAT32_C(  -492.09), EASYSIMD_FLOAT32_C(  -200.61), EASYSIMD_FLOAT32_C(   233.24), EASYSIMD_FLOAT32_C(  -485.99),
        EASYSIMD_FLOAT32_C(  -112.83), EASYSIMD_FLOAT32_C(   283.48), EASYSIMD_FLOAT32_C( -1716.90), EASYSIMD_FLOAT32_C(    28.30) } },
    { { EASYSIMD_FLOAT32_C(   694.89), EASYSIMD_FLOAT32_C(   146.02), EASYSIMD_FLOAT32_C(   226.13), EASYSIMD_FLOAT32_C(   423.33),
        EASYSIMD_FLOAT32_C(  -722.67), EASYSIMD_FLOAT32_C(   544.78), EASYSIMD_FLOAT32_C(   831.39), EASYSIMD_FLOAT32_C(   388.41),
        EASYSIMD_FLOAT32_C(  -491.13), EASYSIMD_FLOAT32_C(   170.79), EASYSIMD_FLOAT32_C(   126.59), EASYSIMD_FLOAT32_C(   380.64),
        EASYSIMD_FLOAT32_C(   491.91), EASYSIMD_FLOAT32_C(   -95.27), EASYSIMD_FLOAT32_C(   885.21), EASYSIMD_FLOAT32_C(  -701.95) },
      { EASYSIMD_FLOAT32_C(   290.33), EASYSIMD_FLOAT32_C(   143.50), EASYSIMD_FLOAT32_C(   303.42), EASYSIMD_FLOAT32_C(    61.01),
        EASYSIMD_FLOAT32_C(     3.20), EASYSIMD_FLOAT32_C(   999.75), EASYSIMD_FLOAT32_C(  -348.18), EASYSIMD_FLOAT32_C(  -615.79),
        EASYSIMD_FLOAT32_C(   236.27), EASYSIMD_FLOAT32_C(    79.18), EASYSIMD_FLOAT32_C(   132.02), EASYSIMD_FLOAT32_C(  -522.67),
        EASYSIMD_FLOAT32_C(  -525.41), EASYSIMD_FLOAT32_C(    70.77), EASYSIMD_FLOAT32_C(   944.47), EASYSIMD_FLOAT32_C(  -830.52) },
      { EASYSIMD_FLOAT32_C(  -985.22), EASYSIMD_FLOAT32_C(  -289.52), EASYSIMD_FLOAT32_C(  -529.55), EASYSIMD_FLOAT32_C(  -484.34),
        EASYSIMD_FLOAT32_C(   719.47), EASYSIMD_FLOAT32_C( -1544.53), EASYSIMD_FLOAT32_C(  -483.21), EASYSIMD_FLOAT32_C(   227.38),
        EASYSIMD_FLOAT32_C(   254.86), EASYSIMD_FLOAT32_C(  -249.97), EASYSIMD_FLOAT32_C(  -258.61), EASYSIMD_FLOAT32_C(   142.03),
        EASYSIMD_FLOAT32_C(    33.50), EASYSIMD_FLOAT32_C(    24.50), EASYSIMD_FLOAT32_C( -1829.68), EASYSIMD_FLOAT32_C(  1532.47) } }
  };


  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_addn_ps(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_addn_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_addn_ps(a, b);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_addn_round_ps (EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT32_C(   -10.55), EASYSIMD_FLOAT32_C(  -966.51), EASYSIMD_FLOAT32_C(  -810.47), EASYSIMD_FLOAT32_C(  -660.32),
        EASYSIMD_FLOAT32_C(   594.58), EASYSIMD_FLOAT32_C(  -287.58), EASYSIMD_FLOAT32_C(  -361.67), EASYSIMD_FLOAT32_C(   246.96),
        EASYSIMD_FLOAT32_C(  -316.67), EASYSIMD_FLOAT32_C(   -60.93), EASYSIMD_FLOAT32_C(  -660.85), EASYSIMD_FLOAT32_C(  -606.59),
        EASYSIMD_FLOAT32_C(   670.42), EASYSIMD_FLOAT32_C(   741.50), EASYSIMD_FLOAT32_C(  -231.68), EASYSIMD_FLOAT32_C(   887.18) },
      { EASYSIMD_FLOAT32_C(    15.95), EASYSIMD_FLOAT32_C(   171.00), EASYSIMD_FLOAT32_C(  -504.30), EASYSIMD_FLOAT32_C(  -320.30),
        EASYSIMD_FLOAT32_C(   874.69), EASYSIMD_FLOAT32_C(   741.31), EASYSIMD_FLOAT32_C(   341.96), EASYSIMD_FLOAT32_C(  -994.75),
        EASYSIMD_FLOAT32_C(   156.16), EASYSIMD_FLOAT32_C(    -5.22), EASYSIMD_FLOAT32_C(  -623.99), EASYSIMD_FLOAT32_C(  -367.85),
        EASYSIMD_FLOAT32_C(   771.01), EASYSIMD_FLOAT32_C(  -393.43), EASYSIMD_FLOAT32_C(   184.56), EASYSIMD_FLOAT32_C(  -239.54) },
      { EASYSIMD_FLOAT32_C(    -5.00), EASYSIMD_FLOAT32_C(   796.00), EASYSIMD_FLOAT32_C(  1315.00), EASYSIMD_FLOAT32_C(   981.00),
        EASYSIMD_FLOAT32_C( -1469.00), EASYSIMD_FLOAT32_C(  -454.00), EASYSIMD_FLOAT32_C(    20.00), EASYSIMD_FLOAT32_C(   748.00),
        EASYSIMD_FLOAT32_C(   161.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(  1285.00), EASYSIMD_FLOAT32_C(   974.00),
        EASYSIMD_FLOAT32_C( -1441.00), EASYSIMD_FLOAT32_C(  -348.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(  -648.00) },
      { EASYSIMD_FLOAT32_C(    -6.00), EASYSIMD_FLOAT32_C(   795.00), EASYSIMD_FLOAT32_C(  1314.00), EASYSIMD_FLOAT32_C(   980.00),
        EASYSIMD_FLOAT32_C( -1470.00), EASYSIMD_FLOAT32_C(  -454.00), EASYSIMD_FLOAT32_C(    19.00), EASYSIMD_FLOAT32_C(   747.00),
        EASYSIMD_FLOAT32_C(   160.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(  1284.00), EASYSIMD_FLOAT32_C(   974.00),
        EASYSIMD_FLOAT32_C( -1442.00), EASYSIMD_FLOAT32_C(  -349.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(  -648.00) },
      { EASYSIMD_FLOAT32_C(    -5.00), EASYSIMD_FLOAT32_C(   796.00), EASYSIMD_FLOAT32_C(  1315.00), EASYSIMD_FLOAT32_C(   981.00),
        EASYSIMD_FLOAT32_C( -1469.00), EASYSIMD_FLOAT32_C(  -453.00), EASYSIMD_FLOAT32_C(    20.00), EASYSIMD_FLOAT32_C(   748.00),
        EASYSIMD_FLOAT32_C(   161.00), EASYSIMD_FLOAT32_C(    67.00), EASYSIMD_FLOAT32_C(  1285.00), EASYSIMD_FLOAT32_C(   975.00),
        EASYSIMD_FLOAT32_C( -1441.00), EASYSIMD_FLOAT32_C(  -348.00), EASYSIMD_FLOAT32_C(    48.00), EASYSIMD_FLOAT32_C(  -647.00) },
      { EASYSIMD_FLOAT32_C(    -5.00), EASYSIMD_FLOAT32_C(   795.00), EASYSIMD_FLOAT32_C(  1314.00), EASYSIMD_FLOAT32_C(   980.00),
        EASYSIMD_FLOAT32_C( -1469.00), EASYSIMD_FLOAT32_C(  -453.00), EASYSIMD_FLOAT32_C(    19.00), EASYSIMD_FLOAT32_C(   747.00),
        EASYSIMD_FLOAT32_C(   160.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(  1284.00), EASYSIMD_FLOAT32_C(   974.00),
        EASYSIMD_FLOAT32_C( -1441.00), EASYSIMD_FLOAT32_C(  -348.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(  -647.00) },
      { EASYSIMD_FLOAT32_C(    -5.00), EASYSIMD_FLOAT32_C(   796.00), EASYSIMD_FLOAT32_C(  1315.00), EASYSIMD_FLOAT32_C(   981.00),
        EASYSIMD_FLOAT32_C( -1469.00), EASYSIMD_FLOAT32_C(  -454.00), EASYSIMD_FLOAT32_C(    20.00), EASYSIMD_FLOAT32_C(   748.00),
        EASYSIMD_FLOAT32_C(   161.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(  1285.00), EASYSIMD_FLOAT32_C(   974.00),
        EASYSIMD_FLOAT32_C( -1441.00), EASYSIMD_FLOAT32_C(  -348.00), EASYSIMD_FLOAT32_C(    47.00), EASYSIMD_FLOAT32_C(  -648.00) } },
    { { EASYSIMD_FLOAT32_C(  -359.94), EASYSIMD_FLOAT32_C(   374.09), EASYSIMD_FLOAT32_C(   100.14), EASYSIMD_FLOAT32_C(  -765.36),
        EASYSIMD_FLOAT32_C(  -913.49), EASYSIMD_FLOAT32_C(   738.47), EASYSIMD_FLOAT32_C(   481.60), EASYSIMD_FLOAT32_C(  -230.15),
        EASYSIMD_FLOAT32_C(  -322.46), EASYSIMD_FLOAT32_C(   820.75), EASYSIMD_FLOAT32_C(   163.26), EASYSIMD_FLOAT32_C(  -652.04),
        EASYSIMD_FLOAT32_C(   562.26), EASYSIMD_FLOAT32_C(   931.57), EASYSIMD_FLOAT32_C(  -764.86), EASYSIMD_FLOAT32_C(  -421.80) },
      { EASYSIMD_FLOAT32_C(   102.57), EASYSIMD_FLOAT32_C(  -269.16), EASYSIMD_FLOAT32_C(   257.90), EASYSIMD_FLOAT32_C(   -22.74),
        EASYSIMD_FLOAT32_C(  -527.84), EASYSIMD_FLOAT32_C(  -400.14), EASYSIMD_FLOAT32_C(   -17.49), EASYSIMD_FLOAT32_C(   628.31),
        EASYSIMD_FLOAT32_C(   594.64), EASYSIMD_FLOAT32_C(   358.52), EASYSIMD_FLOAT32_C(  -739.54), EASYSIMD_FLOAT32_C(   365.66),
        EASYSIMD_FLOAT32_C(   965.09), EASYSIMD_FLOAT32_C(   445.02), EASYSIMD_FLOAT32_C(  -873.88), EASYSIMD_FLOAT32_C(  -394.85) },
      { EASYSIMD_FLOAT32_C(   257.00), EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C(  -358.00), EASYSIMD_FLOAT32_C(   788.00),
        EASYSIMD_FLOAT32_C(  1441.00), EASYSIMD_FLOAT32_C(  -338.00), EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(  -398.00),
        EASYSIMD_FLOAT32_C(  -272.00), EASYSIMD_FLOAT32_C( -1179.00), EASYSIMD_FLOAT32_C(   576.00), EASYSIMD_FLOAT32_C(   286.00),
        EASYSIMD_FLOAT32_C( -1527.00), EASYSIMD_FLOAT32_C( -1377.00), EASYSIMD_FLOAT32_C(  1639.00), EASYSIMD_FLOAT32_C(   817.00) },
      { EASYSIMD_FLOAT32_C(   257.00), EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C(  -359.00), EASYSIMD_FLOAT32_C(   788.00),
        EASYSIMD_FLOAT32_C(  1441.00), EASYSIMD_FLOAT32_C(  -339.00), EASYSIMD_FLOAT32_C(  -465.00), EASYSIMD_FLOAT32_C(  -399.00),
        EASYSIMD_FLOAT32_C(  -273.00), EASYSIMD_FLOAT32_C( -1180.00), EASYSIMD_FLOAT32_C(   576.00), EASYSIMD_FLOAT32_C(   286.00),
        EASYSIMD_FLOAT32_C( -1528.00), EASYSIMD_FLOAT32_C( -1377.00), EASYSIMD_FLOAT32_C(  1638.00), EASYSIMD_FLOAT32_C(   816.00) },
      { EASYSIMD_FLOAT32_C(   258.00), EASYSIMD_FLOAT32_C(  -104.00), EASYSIMD_FLOAT32_C(  -358.00), EASYSIMD_FLOAT32_C(   789.00),
        EASYSIMD_FLOAT32_C(  1442.00), EASYSIMD_FLOAT32_C(  -338.00), EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(  -398.00),
        EASYSIMD_FLOAT32_C(  -272.00), EASYSIMD_FLOAT32_C( -1179.00), EASYSIMD_FLOAT32_C(   577.00), EASYSIMD_FLOAT32_C(   287.00),
        EASYSIMD_FLOAT32_C( -1527.00), EASYSIMD_FLOAT32_C( -1376.00), EASYSIMD_FLOAT32_C(  1639.00), EASYSIMD_FLOAT32_C(   817.00) },
      { EASYSIMD_FLOAT32_C(   257.00), EASYSIMD_FLOAT32_C(  -104.00), EASYSIMD_FLOAT32_C(  -358.00), EASYSIMD_FLOAT32_C(   788.00),
        EASYSIMD_FLOAT32_C(  1441.00), EASYSIMD_FLOAT32_C(  -338.00), EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(  -398.00),
        EASYSIMD_FLOAT32_C(  -272.00), EASYSIMD_FLOAT32_C( -1179.00), EASYSIMD_FLOAT32_C(   576.00), EASYSIMD_FLOAT32_C(   286.00),
        EASYSIMD_FLOAT32_C( -1527.00), EASYSIMD_FLOAT32_C( -1376.00), EASYSIMD_FLOAT32_C(  1638.00), EASYSIMD_FLOAT32_C(   816.00) },
      { EASYSIMD_FLOAT32_C(   257.00), EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C(  -358.00), EASYSIMD_FLOAT32_C(   788.00),
        EASYSIMD_FLOAT32_C(  1441.00), EASYSIMD_FLOAT32_C(  -338.00), EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(  -398.00),
        EASYSIMD_FLOAT32_C(  -272.00), EASYSIMD_FLOAT32_C( -1179.00), EASYSIMD_FLOAT32_C(   576.00), EASYSIMD_FLOAT32_C(   286.00),
        EASYSIMD_FLOAT32_C( -1527.00), EASYSIMD_FLOAT32_C( -1377.00), EASYSIMD_FLOAT32_C(  1639.00), EASYSIMD_FLOAT32_C(   817.00) } },
    { { EASYSIMD_FLOAT32_C(  -180.89), EASYSIMD_FLOAT32_C(   226.26), EASYSIMD_FLOAT32_C(  -160.21), EASYSIMD_FLOAT32_C(   -94.38),
        EASYSIMD_FLOAT32_C(   -35.27), EASYSIMD_FLOAT32_C(  -678.61), EASYSIMD_FLOAT32_C(   675.47), EASYSIMD_FLOAT32_C(   642.26),
        EASYSIMD_FLOAT32_C(  -857.86), EASYSIMD_FLOAT32_C(  -161.27), EASYSIMD_FLOAT32_C(   990.22), EASYSIMD_FLOAT32_C(   704.40),
        EASYSIMD_FLOAT32_C(  -229.70), EASYSIMD_FLOAT32_C(  -774.63), EASYSIMD_FLOAT32_C(  -717.40), EASYSIMD_FLOAT32_C(   872.87) },
      { EASYSIMD_FLOAT32_C(   -43.79), EASYSIMD_FLOAT32_C(   540.51), EASYSIMD_FLOAT32_C(  -149.87), EASYSIMD_FLOAT32_C(   428.37),
        EASYSIMD_FLOAT32_C(  -859.63), EASYSIMD_FLOAT32_C(   832.64), EASYSIMD_FLOAT32_C(    56.68), EASYSIMD_FLOAT32_C(   735.02),
        EASYSIMD_FLOAT32_C(   191.16), EASYSIMD_FLOAT32_C(   317.14), EASYSIMD_FLOAT32_C(   100.67), EASYSIMD_FLOAT32_C(   156.25),
        EASYSIMD_FLOAT32_C(  -237.84), EASYSIMD_FLOAT32_C(   226.79), EASYSIMD_FLOAT32_C(   761.39), EASYSIMD_FLOAT32_C(   581.27) },
      { EASYSIMD_FLOAT32_C(   225.00), EASYSIMD_FLOAT32_C(  -767.00), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -334.00),
        EASYSIMD_FLOAT32_C(   895.00), EASYSIMD_FLOAT32_C(  -154.00), EASYSIMD_FLOAT32_C(  -732.00), EASYSIMD_FLOAT32_C( -1377.00),
        EASYSIMD_FLOAT32_C(   667.00), EASYSIMD_FLOAT32_C(  -156.00), EASYSIMD_FLOAT32_C( -1091.00), EASYSIMD_FLOAT32_C(  -861.00),
        EASYSIMD_FLOAT32_C(   468.00), EASYSIMD_FLOAT32_C(   548.00), EASYSIMD_FLOAT32_C(   -44.00), EASYSIMD_FLOAT32_C( -1454.00) },
      { EASYSIMD_FLOAT32_C(   224.00), EASYSIMD_FLOAT32_C(  -767.00), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -334.00),
        EASYSIMD_FLOAT32_C(   894.00), EASYSIMD_FLOAT32_C(  -155.00), EASYSIMD_FLOAT32_C(  -733.00), EASYSIMD_FLOAT32_C( -1378.00),
        EASYSIMD_FLOAT32_C(   666.00), EASYSIMD_FLOAT32_C(  -156.00), EASYSIMD_FLOAT32_C( -1091.00), EASYSIMD_FLOAT32_C(  -861.00),
        EASYSIMD_FLOAT32_C(   467.00), EASYSIMD_FLOAT32_C(   547.00), EASYSIMD_FLOAT32_C(   -44.00), EASYSIMD_FLOAT32_C( -1455.00) },
      { EASYSIMD_FLOAT32_C(   225.00), EASYSIMD_FLOAT32_C(  -766.00), EASYSIMD_FLOAT32_C(   311.00), EASYSIMD_FLOAT32_C(  -333.00),
        EASYSIMD_FLOAT32_C(   895.00), EASYSIMD_FLOAT32_C(  -154.00), EASYSIMD_FLOAT32_C(  -732.00), EASYSIMD_FLOAT32_C( -1377.00),
        EASYSIMD_FLOAT32_C(   667.00), EASYSIMD_FLOAT32_C(  -155.00), EASYSIMD_FLOAT32_C( -1090.00), EASYSIMD_FLOAT32_C(  -860.00),
        EASYSIMD_FLOAT32_C(   468.00), EASYSIMD_FLOAT32_C(   548.00), EASYSIMD_FLOAT32_C(   -43.00), EASYSIMD_FLOAT32_C( -1454.00) },
      { EASYSIMD_FLOAT32_C(   224.00), EASYSIMD_FLOAT32_C(  -766.00), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -333.00),
        EASYSIMD_FLOAT32_C(   894.00), EASYSIMD_FLOAT32_C(  -154.00), EASYSIMD_FLOAT32_C(  -732.00), EASYSIMD_FLOAT32_C( -1377.00),
        EASYSIMD_FLOAT32_C(   666.00), EASYSIMD_FLOAT32_C(  -155.00), EASYSIMD_FLOAT32_C( -1090.00), EASYSIMD_FLOAT32_C(  -860.00),
        EASYSIMD_FLOAT32_C(   467.00), EASYSIMD_FLOAT32_C(   547.00), EASYSIMD_FLOAT32_C(   -43.00), EASYSIMD_FLOAT32_C( -1454.00) },
      { EASYSIMD_FLOAT32_C(   225.00), EASYSIMD_FLOAT32_C(  -767.00), EASYSIMD_FLOAT32_C(   310.00), EASYSIMD_FLOAT32_C(  -334.00),
        EASYSIMD_FLOAT32_C(   895.00), EASYSIMD_FLOAT32_C(  -154.00), EASYSIMD_FLOAT32_C(  -732.00), EASYSIMD_FLOAT32_C( -1377.00),
        EASYSIMD_FLOAT32_C(   667.00), EASYSIMD_FLOAT32_C(  -156.00), EASYSIMD_FLOAT32_C( -1091.00), EASYSIMD_FLOAT32_C(  -861.00),
        EASYSIMD_FLOAT32_C(   468.00), EASYSIMD_FLOAT32_C(   548.00), EASYSIMD_FLOAT32_C(   -44.00), EASYSIMD_FLOAT32_C( -1454.00) } },
    { { EASYSIMD_FLOAT32_C(  -546.95), EASYSIMD_FLOAT32_C(  -398.82), EASYSIMD_FLOAT32_C(  -513.11), EASYSIMD_FLOAT32_C(   417.78),
        EASYSIMD_FLOAT32_C(   -77.43), EASYSIMD_FLOAT32_C(  -837.64), EASYSIMD_FLOAT32_C(    60.04), EASYSIMD_FLOAT32_C(    64.71),
        EASYSIMD_FLOAT32_C(     1.09), EASYSIMD_FLOAT32_C(    50.27), EASYSIMD_FLOAT32_C(  -230.89), EASYSIMD_FLOAT32_C(   771.39),
        EASYSIMD_FLOAT32_C(   275.63), EASYSIMD_FLOAT32_C(    51.71), EASYSIMD_FLOAT32_C(   644.26), EASYSIMD_FLOAT32_C(  -768.16) },
      { EASYSIMD_FLOAT32_C(  -407.78), EASYSIMD_FLOAT32_C(  -505.62), EASYSIMD_FLOAT32_C(   660.21), EASYSIMD_FLOAT32_C(  -267.41),
        EASYSIMD_FLOAT32_C(  -672.98), EASYSIMD_FLOAT32_C(  -283.11), EASYSIMD_FLOAT32_C(  -532.39), EASYSIMD_FLOAT32_C(   518.19),
        EASYSIMD_FLOAT32_C(  -965.97), EASYSIMD_FLOAT32_C(   568.28), EASYSIMD_FLOAT32_C(  -325.57), EASYSIMD_FLOAT32_C(  -203.80),
        EASYSIMD_FLOAT32_C(  -204.93), EASYSIMD_FLOAT32_C(  -564.17), EASYSIMD_FLOAT32_C(  -622.53), EASYSIMD_FLOAT32_C(   248.13) },
      { EASYSIMD_FLOAT32_C(   955.00), EASYSIMD_FLOAT32_C(   904.00), EASYSIMD_FLOAT32_C(  -147.00), EASYSIMD_FLOAT32_C(  -150.00),
        EASYSIMD_FLOAT32_C(   750.00), EASYSIMD_FLOAT32_C(  1121.00), EASYSIMD_FLOAT32_C(   472.00), EASYSIMD_FLOAT32_C(  -583.00),
        EASYSIMD_FLOAT32_C(   965.00), EASYSIMD_FLOAT32_C(  -619.00), EASYSIMD_FLOAT32_C(   556.00), EASYSIMD_FLOAT32_C(  -568.00),
        EASYSIMD_FLOAT32_C(   -71.00), EASYSIMD_FLOAT32_C(   512.00), EASYSIMD_FLOAT32_C(   -22.00), EASYSIMD_FLOAT32_C(   520.00) },
      { EASYSIMD_FLOAT32_C(   954.00), EASYSIMD_FLOAT32_C(   904.00), EASYSIMD_FLOAT32_C(  -148.00), EASYSIMD_FLOAT32_C(  -151.00),
        EASYSIMD_FLOAT32_C(   750.00), EASYSIMD_FLOAT32_C(  1120.00), EASYSIMD_FLOAT32_C(   472.00), EASYSIMD_FLOAT32_C(  -583.00),
        EASYSIMD_FLOAT32_C(   964.00), EASYSIMD_FLOAT32_C(  -619.00), EASYSIMD_FLOAT32_C(   556.00), EASYSIMD_FLOAT32_C(  -568.00),
        EASYSIMD_FLOAT32_C(   -71.00), EASYSIMD_FLOAT32_C(   512.00), EASYSIMD_FLOAT32_C(   -22.00), EASYSIMD_FLOAT32_C(   520.00) },
      { EASYSIMD_FLOAT32_C(   955.00), EASYSIMD_FLOAT32_C(   905.00), EASYSIMD_FLOAT32_C(  -147.00), EASYSIMD_FLOAT32_C(  -150.00),
        EASYSIMD_FLOAT32_C(   751.00), EASYSIMD_FLOAT32_C(  1121.00), EASYSIMD_FLOAT32_C(   473.00), EASYSIMD_FLOAT32_C(  -582.00),
        EASYSIMD_FLOAT32_C(   965.00), EASYSIMD_FLOAT32_C(  -618.00), EASYSIMD_FLOAT32_C(   557.00), EASYSIMD_FLOAT32_C(  -567.00),
        EASYSIMD_FLOAT32_C(   -70.00), EASYSIMD_FLOAT32_C(   513.00), EASYSIMD_FLOAT32_C(   -21.00), EASYSIMD_FLOAT32_C(   521.00) },
      { EASYSIMD_FLOAT32_C(   954.00), EASYSIMD_FLOAT32_C(   904.00), EASYSIMD_FLOAT32_C(  -147.00), EASYSIMD_FLOAT32_C(  -150.00),
        EASYSIMD_FLOAT32_C(   750.00), EASYSIMD_FLOAT32_C(  1120.00), EASYSIMD_FLOAT32_C(   472.00), EASYSIMD_FLOAT32_C(  -582.00),
        EASYSIMD_FLOAT32_C(   964.00), EASYSIMD_FLOAT32_C(  -618.00), EASYSIMD_FLOAT32_C(   556.00), EASYSIMD_FLOAT32_C(  -567.00),
        EASYSIMD_FLOAT32_C(   -70.00), EASYSIMD_FLOAT32_C(   512.00), EASYSIMD_FLOAT32_C(   -21.00), EASYSIMD_FLOAT32_C(   520.00) },
      { EASYSIMD_FLOAT32_C(   955.00), EASYSIMD_FLOAT32_C(   904.00), EASYSIMD_FLOAT32_C(  -147.00), EASYSIMD_FLOAT32_C(  -150.00),
        EASYSIMD_FLOAT32_C(   750.00), EASYSIMD_FLOAT32_C(  1121.00), EASYSIMD_FLOAT32_C(   472.00), EASYSIMD_FLOAT32_C(  -583.00),
        EASYSIMD_FLOAT32_C(   965.00), EASYSIMD_FLOAT32_C(  -619.00), EASYSIMD_FLOAT32_C(   556.00), EASYSIMD_FLOAT32_C(  -568.00),
        EASYSIMD_FLOAT32_C(   -71.00), EASYSIMD_FLOAT32_C(   512.00), EASYSIMD_FLOAT32_C(   -22.00), EASYSIMD_FLOAT32_C(   520.00) } },
    { { EASYSIMD_FLOAT32_C(    37.01), EASYSIMD_FLOAT32_C(  -135.64), EASYSIMD_FLOAT32_C(  -334.10), EASYSIMD_FLOAT32_C(   959.57),
        EASYSIMD_FLOAT32_C(    26.72), EASYSIMD_FLOAT32_C(   725.95), EASYSIMD_FLOAT32_C(    24.28), EASYSIMD_FLOAT32_C(  -972.20),
        EASYSIMD_FLOAT32_C(  -223.79), EASYSIMD_FLOAT32_C(   793.39), EASYSIMD_FLOAT32_C(   799.19), EASYSIMD_FLOAT32_C(  -948.16),
        EASYSIMD_FLOAT32_C(  -154.90), EASYSIMD_FLOAT32_C(   443.44), EASYSIMD_FLOAT32_C(  -716.32), EASYSIMD_FLOAT32_C(   437.33) },
      { EASYSIMD_FLOAT32_C(   937.83), EASYSIMD_FLOAT32_C(   943.89), EASYSIMD_FLOAT32_C(  -830.08), EASYSIMD_FLOAT32_C(  -735.15),
        EASYSIMD_FLOAT32_C(  -339.22), EASYSIMD_FLOAT32_C(  -362.48), EASYSIMD_FLOAT32_C(   783.04), EASYSIMD_FLOAT32_C(  -305.19),
        EASYSIMD_FLOAT32_C(  -794.19), EASYSIMD_FLOAT32_C(  -542.53), EASYSIMD_FLOAT32_C(   491.01), EASYSIMD_FLOAT32_C(     0.88),
        EASYSIMD_FLOAT32_C(  -106.70), EASYSIMD_FLOAT32_C(   868.48), EASYSIMD_FLOAT32_C(  -750.99), EASYSIMD_FLOAT32_C(   930.31) },
      { EASYSIMD_FLOAT32_C(  -975.00), EASYSIMD_FLOAT32_C(  -808.00), EASYSIMD_FLOAT32_C(  1164.00), EASYSIMD_FLOAT32_C(  -224.00),
        EASYSIMD_FLOAT32_C(   312.00), EASYSIMD_FLOAT32_C(  -363.00), EASYSIMD_FLOAT32_C(  -807.00), EASYSIMD_FLOAT32_C(  1277.00),
        EASYSIMD_FLOAT32_C(  1018.00), EASYSIMD_FLOAT32_C(  -251.00), EASYSIMD_FLOAT32_C( -1290.00), EASYSIMD_FLOAT32_C(   947.00),
        EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C( -1312.00), EASYSIMD_FLOAT32_C(  1467.00), EASYSIMD_FLOAT32_C( -1368.00) },
      { EASYSIMD_FLOAT32_C(  -975.00), EASYSIMD_FLOAT32_C(  -809.00), EASYSIMD_FLOAT32_C(  1164.00), EASYSIMD_FLOAT32_C(  -225.00),
        EASYSIMD_FLOAT32_C(   312.00), EASYSIMD_FLOAT32_C(  -364.00), EASYSIMD_FLOAT32_C(  -808.00), EASYSIMD_FLOAT32_C(  1277.00),
        EASYSIMD_FLOAT32_C(  1017.00), EASYSIMD_FLOAT32_C(  -251.00), EASYSIMD_FLOAT32_C( -1291.00), EASYSIMD_FLOAT32_C(   947.00),
        EASYSIMD_FLOAT32_C(   261.00), EASYSIMD_FLOAT32_C( -1312.00), EASYSIMD_FLOAT32_C(  1467.00), EASYSIMD_FLOAT32_C( -1368.00) },
      { EASYSIMD_FLOAT32_C(  -974.00), EASYSIMD_FLOAT32_C(  -808.00), EASYSIMD_FLOAT32_C(  1165.00), EASYSIMD_FLOAT32_C(  -224.00),
        EASYSIMD_FLOAT32_C(   313.00), EASYSIMD_FLOAT32_C(  -363.00), EASYSIMD_FLOAT32_C(  -807.00), EASYSIMD_FLOAT32_C(  1278.00),
        EASYSIMD_FLOAT32_C(  1018.00), EASYSIMD_FLOAT32_C(  -250.00), EASYSIMD_FLOAT32_C( -1290.00), EASYSIMD_FLOAT32_C(   948.00),
        EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C( -1311.00), EASYSIMD_FLOAT32_C(  1468.00), EASYSIMD_FLOAT32_C( -1367.00) },
      { EASYSIMD_FLOAT32_C(  -974.00), EASYSIMD_FLOAT32_C(  -808.00), EASYSIMD_FLOAT32_C(  1164.00), EASYSIMD_FLOAT32_C(  -224.00),
        EASYSIMD_FLOAT32_C(   312.00), EASYSIMD_FLOAT32_C(  -363.00), EASYSIMD_FLOAT32_C(  -807.00), EASYSIMD_FLOAT32_C(  1277.00),
        EASYSIMD_FLOAT32_C(  1017.00), EASYSIMD_FLOAT32_C(  -250.00), EASYSIMD_FLOAT32_C( -1290.00), EASYSIMD_FLOAT32_C(   947.00),
        EASYSIMD_FLOAT32_C(   261.00), EASYSIMD_FLOAT32_C( -1311.00), EASYSIMD_FLOAT32_C(  1467.00), EASYSIMD_FLOAT32_C( -1367.00) },
      { EASYSIMD_FLOAT32_C(  -975.00), EASYSIMD_FLOAT32_C(  -808.00), EASYSIMD_FLOAT32_C(  1164.00), EASYSIMD_FLOAT32_C(  -224.00),
        EASYSIMD_FLOAT32_C(   312.00), EASYSIMD_FLOAT32_C(  -363.00), EASYSIMD_FLOAT32_C(  -807.00), EASYSIMD_FLOAT32_C(  1277.00),
        EASYSIMD_FLOAT32_C(  1018.00), EASYSIMD_FLOAT32_C(  -251.00), EASYSIMD_FLOAT32_C( -1290.00), EASYSIMD_FLOAT32_C(   947.00),
        EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C( -1312.00), EASYSIMD_FLOAT32_C(  1467.00), EASYSIMD_FLOAT32_C( -1368.00) } },
    { { EASYSIMD_FLOAT32_C(  -267.16), EASYSIMD_FLOAT32_C(   -85.09), EASYSIMD_FLOAT32_C(   889.88), EASYSIMD_FLOAT32_C(   759.55),
        EASYSIMD_FLOAT32_C(  -359.14), EASYSIMD_FLOAT32_C(   -85.83), EASYSIMD_FLOAT32_C(   787.36), EASYSIMD_FLOAT32_C(   417.07),
        EASYSIMD_FLOAT32_C(  -292.44), EASYSIMD_FLOAT32_C(   586.55), EASYSIMD_FLOAT32_C(   468.91), EASYSIMD_FLOAT32_C(   552.66),
        EASYSIMD_FLOAT32_C(    29.99), EASYSIMD_FLOAT32_C(   752.59), EASYSIMD_FLOAT32_C(   -10.01), EASYSIMD_FLOAT32_C(   -32.18) },
      { EASYSIMD_FLOAT32_C(   696.48), EASYSIMD_FLOAT32_C(   159.91), EASYSIMD_FLOAT32_C(   232.67), EASYSIMD_FLOAT32_C(  -642.74),
        EASYSIMD_FLOAT32_C(   797.43), EASYSIMD_FLOAT32_C(    15.71), EASYSIMD_FLOAT32_C(    52.08), EASYSIMD_FLOAT32_C(  -996.76),
        EASYSIMD_FLOAT32_C(   473.19), EASYSIMD_FLOAT32_C(  -456.91), EASYSIMD_FLOAT32_C(     4.12), EASYSIMD_FLOAT32_C(  -633.51),
        EASYSIMD_FLOAT32_C(  -588.43), EASYSIMD_FLOAT32_C(   253.12), EASYSIMD_FLOAT32_C(  -703.21), EASYSIMD_FLOAT32_C(   144.41) },
      { EASYSIMD_FLOAT32_C(  -429.00), EASYSIMD_FLOAT32_C(   -75.00), EASYSIMD_FLOAT32_C( -1123.00), EASYSIMD_FLOAT32_C(  -117.00),
        EASYSIMD_FLOAT32_C(  -438.00), EASYSIMD_FLOAT32_C(    70.00), EASYSIMD_FLOAT32_C(  -839.00), EASYSIMD_FLOAT32_C(   580.00),
        EASYSIMD_FLOAT32_C(  -181.00), EASYSIMD_FLOAT32_C(  -130.00), EASYSIMD_FLOAT32_C(  -473.00), EASYSIMD_FLOAT32_C(    81.00),
        EASYSIMD_FLOAT32_C(   558.00), EASYSIMD_FLOAT32_C( -1006.00), EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(  -112.00) },
      { EASYSIMD_FLOAT32_C(  -430.00), EASYSIMD_FLOAT32_C(   -75.00), EASYSIMD_FLOAT32_C( -1123.00), EASYSIMD_FLOAT32_C(  -117.00),
        EASYSIMD_FLOAT32_C(  -439.00), EASYSIMD_FLOAT32_C(    70.00), EASYSIMD_FLOAT32_C(  -840.00), EASYSIMD_FLOAT32_C(   579.00),
        EASYSIMD_FLOAT32_C(  -181.00), EASYSIMD_FLOAT32_C(  -130.00), EASYSIMD_FLOAT32_C(  -474.00), EASYSIMD_FLOAT32_C(    80.00),
        EASYSIMD_FLOAT32_C(   558.00), EASYSIMD_FLOAT32_C( -1006.00), EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(  -113.00) },
      { EASYSIMD_FLOAT32_C(  -429.00), EASYSIMD_FLOAT32_C(   -74.00), EASYSIMD_FLOAT32_C( -1122.00), EASYSIMD_FLOAT32_C(  -116.00),
        EASYSIMD_FLOAT32_C(  -438.00), EASYSIMD_FLOAT32_C(    71.00), EASYSIMD_FLOAT32_C(  -839.00), EASYSIMD_FLOAT32_C(   580.00),
        EASYSIMD_FLOAT32_C(  -180.00), EASYSIMD_FLOAT32_C(  -129.00), EASYSIMD_FLOAT32_C(  -473.00), EASYSIMD_FLOAT32_C(    81.00),
        EASYSIMD_FLOAT32_C(   559.00), EASYSIMD_FLOAT32_C( -1005.00), EASYSIMD_FLOAT32_C(   714.00), EASYSIMD_FLOAT32_C(  -112.00) },
      { EASYSIMD_FLOAT32_C(  -429.00), EASYSIMD_FLOAT32_C(   -74.00), EASYSIMD_FLOAT32_C( -1122.00), EASYSIMD_FLOAT32_C(  -116.00),
        EASYSIMD_FLOAT32_C(  -438.00), EASYSIMD_FLOAT32_C(    70.00), EASYSIMD_FLOAT32_C(  -839.00), EASYSIMD_FLOAT32_C(   579.00),
        EASYSIMD_FLOAT32_C(  -180.00), EASYSIMD_FLOAT32_C(  -129.00), EASYSIMD_FLOAT32_C(  -473.00), EASYSIMD_FLOAT32_C(    80.00),
        EASYSIMD_FLOAT32_C(   558.00), EASYSIMD_FLOAT32_C( -1005.00), EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(  -112.00) },
      { EASYSIMD_FLOAT32_C(  -429.00), EASYSIMD_FLOAT32_C(   -75.00), EASYSIMD_FLOAT32_C( -1123.00), EASYSIMD_FLOAT32_C(  -117.00),
        EASYSIMD_FLOAT32_C(  -438.00), EASYSIMD_FLOAT32_C(    70.00), EASYSIMD_FLOAT32_C(  -839.00), EASYSIMD_FLOAT32_C(   580.00),
        EASYSIMD_FLOAT32_C(  -181.00), EASYSIMD_FLOAT32_C(  -130.00), EASYSIMD_FLOAT32_C(  -473.00), EASYSIMD_FLOAT32_C(    81.00),
        EASYSIMD_FLOAT32_C(   558.00), EASYSIMD_FLOAT32_C( -1006.00), EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(  -112.00) } },
    { { EASYSIMD_FLOAT32_C(  -831.96), EASYSIMD_FLOAT32_C(  -813.32), EASYSIMD_FLOAT32_C(   -96.04), EASYSIMD_FLOAT32_C(  -191.11),
        EASYSIMD_FLOAT32_C(   100.84), EASYSIMD_FLOAT32_C(  -308.68), EASYSIMD_FLOAT32_C(  -774.04), EASYSIMD_FLOAT32_C(   808.40),
        EASYSIMD_FLOAT32_C(  -722.14), EASYSIMD_FLOAT32_C(   694.87), EASYSIMD_FLOAT32_C(   361.07), EASYSIMD_FLOAT32_C(   307.85),
        EASYSIMD_FLOAT32_C(   447.46), EASYSIMD_FLOAT32_C(  -648.95), EASYSIMD_FLOAT32_C(  -724.33), EASYSIMD_FLOAT32_C(   143.95) },
      { EASYSIMD_FLOAT32_C(   510.96), EASYSIMD_FLOAT32_C(   508.35), EASYSIMD_FLOAT32_C(   501.21), EASYSIMD_FLOAT32_C(   308.39),
        EASYSIMD_FLOAT32_C(  -475.94), EASYSIMD_FLOAT32_C(  -446.72), EASYSIMD_FLOAT32_C(   311.63), EASYSIMD_FLOAT32_C(   997.24),
        EASYSIMD_FLOAT32_C(    96.37), EASYSIMD_FLOAT32_C(  -684.25), EASYSIMD_FLOAT32_C(  -636.27), EASYSIMD_FLOAT32_C(   507.94),
        EASYSIMD_FLOAT32_C(   568.87), EASYSIMD_FLOAT32_C(  -339.48), EASYSIMD_FLOAT32_C(  -347.65), EASYSIMD_FLOAT32_C(   736.91) },
      { EASYSIMD_FLOAT32_C(   321.00), EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(  -405.00), EASYSIMD_FLOAT32_C(  -117.00),
        EASYSIMD_FLOAT32_C(   375.00), EASYSIMD_FLOAT32_C(   755.00), EASYSIMD_FLOAT32_C(   462.00), EASYSIMD_FLOAT32_C( -1806.00),
        EASYSIMD_FLOAT32_C(   626.00), EASYSIMD_FLOAT32_C(   -11.00), EASYSIMD_FLOAT32_C(   275.00), EASYSIMD_FLOAT32_C(  -816.00),
        EASYSIMD_FLOAT32_C( -1016.00), EASYSIMD_FLOAT32_C(   988.00), EASYSIMD_FLOAT32_C(  1072.00), EASYSIMD_FLOAT32_C(  -881.00) },
      { EASYSIMD_FLOAT32_C(   321.00), EASYSIMD_FLOAT32_C(   304.00), EASYSIMD_FLOAT32_C(  -406.00), EASYSIMD_FLOAT32_C(  -118.00),
        EASYSIMD_FLOAT32_C(   375.00), EASYSIMD_FLOAT32_C(   755.00), EASYSIMD_FLOAT32_C(   462.00), EASYSIMD_FLOAT32_C( -1806.00),
        EASYSIMD_FLOAT32_C(   625.00), EASYSIMD_FLOAT32_C(   -11.00), EASYSIMD_FLOAT32_C(   275.00), EASYSIMD_FLOAT32_C(  -816.00),
        EASYSIMD_FLOAT32_C( -1017.00), EASYSIMD_FLOAT32_C(   988.00), EASYSIMD_FLOAT32_C(  1071.00), EASYSIMD_FLOAT32_C(  -881.00) },
      { EASYSIMD_FLOAT32_C(   322.00), EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(  -405.00), EASYSIMD_FLOAT32_C(  -117.00),
        EASYSIMD_FLOAT32_C(   376.00), EASYSIMD_FLOAT32_C(   756.00), EASYSIMD_FLOAT32_C(   463.00), EASYSIMD_FLOAT32_C( -1805.00),
        EASYSIMD_FLOAT32_C(   626.00), EASYSIMD_FLOAT32_C(   -10.00), EASYSIMD_FLOAT32_C(   276.00), EASYSIMD_FLOAT32_C(  -815.00),
        EASYSIMD_FLOAT32_C( -1016.00), EASYSIMD_FLOAT32_C(   989.00), EASYSIMD_FLOAT32_C(  1072.00), EASYSIMD_FLOAT32_C(  -880.00) },
      { EASYSIMD_FLOAT32_C(   321.00), EASYSIMD_FLOAT32_C(   304.00), EASYSIMD_FLOAT32_C(  -405.00), EASYSIMD_FLOAT32_C(  -117.00),
        EASYSIMD_FLOAT32_C(   375.00), EASYSIMD_FLOAT32_C(   755.00), EASYSIMD_FLOAT32_C(   462.00), EASYSIMD_FLOAT32_C( -1805.00),
        EASYSIMD_FLOAT32_C(   625.00), EASYSIMD_FLOAT32_C(   -10.00), EASYSIMD_FLOAT32_C(   275.00), EASYSIMD_FLOAT32_C(  -815.00),
        EASYSIMD_FLOAT32_C( -1016.00), EASYSIMD_FLOAT32_C(   988.00), EASYSIMD_FLOAT32_C(  1071.00), EASYSIMD_FLOAT32_C(  -880.00) },
      { EASYSIMD_FLOAT32_C(   321.00), EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(  -405.00), EASYSIMD_FLOAT32_C(  -117.00),
        EASYSIMD_FLOAT32_C(   375.00), EASYSIMD_FLOAT32_C(   755.00), EASYSIMD_FLOAT32_C(   462.00), EASYSIMD_FLOAT32_C( -1806.00),
        EASYSIMD_FLOAT32_C(   626.00), EASYSIMD_FLOAT32_C(   -11.00), EASYSIMD_FLOAT32_C(   275.00), EASYSIMD_FLOAT32_C(  -816.00),
        EASYSIMD_FLOAT32_C( -1016.00), EASYSIMD_FLOAT32_C(   988.00), EASYSIMD_FLOAT32_C(  1072.00), EASYSIMD_FLOAT32_C(  -881.00) } },
    { { EASYSIMD_FLOAT32_C(  -152.80), EASYSIMD_FLOAT32_C(   556.31), EASYSIMD_FLOAT32_C(  -454.20), EASYSIMD_FLOAT32_C(   948.04),
        EASYSIMD_FLOAT32_C(  -752.38), EASYSIMD_FLOAT32_C(  -228.24), EASYSIMD_FLOAT32_C(   756.44), EASYSIMD_FLOAT32_C(  -474.51),
        EASYSIMD_FLOAT32_C(  -533.37), EASYSIMD_FLOAT32_C(   117.51), EASYSIMD_FLOAT32_C(   833.34), EASYSIMD_FLOAT32_C(   914.09),
        EASYSIMD_FLOAT32_C(   468.56), EASYSIMD_FLOAT32_C(  -890.99), EASYSIMD_FLOAT32_C(    58.04), EASYSIMD_FLOAT32_C(   -20.47) },
      { EASYSIMD_FLOAT32_C(   617.36), EASYSIMD_FLOAT32_C(  -440.75), EASYSIMD_FLOAT32_C(  -712.08), EASYSIMD_FLOAT32_C(  -858.58),
        EASYSIMD_FLOAT32_C(   112.53), EASYSIMD_FLOAT32_C(   599.55), EASYSIMD_FLOAT32_C(  -861.34), EASYSIMD_FLOAT32_C(  -791.10),
        EASYSIMD_FLOAT32_C(   915.29), EASYSIMD_FLOAT32_C(  -497.61), EASYSIMD_FLOAT32_C(   716.84), EASYSIMD_FLOAT32_C(   484.16),
        EASYSIMD_FLOAT32_C(   162.91), EASYSIMD_FLOAT32_C(  -630.81), EASYSIMD_FLOAT32_C(   221.07), EASYSIMD_FLOAT32_C(  -989.89) },
      { EASYSIMD_FLOAT32_C(  -465.00), EASYSIMD_FLOAT32_C(  -116.00), EASYSIMD_FLOAT32_C(  1166.00), EASYSIMD_FLOAT32_C(   -89.00),
        EASYSIMD_FLOAT32_C(   640.00), EASYSIMD_FLOAT32_C(  -371.00), EASYSIMD_FLOAT32_C(   105.00), EASYSIMD_FLOAT32_C(  1266.00),
        EASYSIMD_FLOAT32_C(  -382.00), EASYSIMD_FLOAT32_C(   380.00), EASYSIMD_FLOAT32_C( -1550.00), EASYSIMD_FLOAT32_C( -1398.00),
        EASYSIMD_FLOAT32_C(  -631.00), EASYSIMD_FLOAT32_C(  1522.00), EASYSIMD_FLOAT32_C(  -279.00), EASYSIMD_FLOAT32_C(  1010.00) },
      { EASYSIMD_FLOAT32_C(  -465.00), EASYSIMD_FLOAT32_C(  -116.00), EASYSIMD_FLOAT32_C(  1166.00), EASYSIMD_FLOAT32_C(   -90.00),
        EASYSIMD_FLOAT32_C(   639.00), EASYSIMD_FLOAT32_C(  -372.00), EASYSIMD_FLOAT32_C(   104.00), EASYSIMD_FLOAT32_C(  1265.00),
        EASYSIMD_FLOAT32_C(  -382.00), EASYSIMD_FLOAT32_C(   380.00), EASYSIMD_FLOAT32_C( -1551.00), EASYSIMD_FLOAT32_C( -1399.00),
        EASYSIMD_FLOAT32_C(  -632.00), EASYSIMD_FLOAT32_C(  1521.00), EASYSIMD_FLOAT32_C(  -280.00), EASYSIMD_FLOAT32_C(  1010.00) },
      { EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(  -115.00), EASYSIMD_FLOAT32_C(  1167.00), EASYSIMD_FLOAT32_C(   -89.00),
        EASYSIMD_FLOAT32_C(   640.00), EASYSIMD_FLOAT32_C(  -371.00), EASYSIMD_FLOAT32_C(   105.00), EASYSIMD_FLOAT32_C(  1266.00),
        EASYSIMD_FLOAT32_C(  -381.00), EASYSIMD_FLOAT32_C(   381.00), EASYSIMD_FLOAT32_C( -1550.00), EASYSIMD_FLOAT32_C( -1398.00),
        EASYSIMD_FLOAT32_C(  -631.00), EASYSIMD_FLOAT32_C(  1522.00), EASYSIMD_FLOAT32_C(  -279.00), EASYSIMD_FLOAT32_C(  1011.00) },
      { EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(  -115.00), EASYSIMD_FLOAT32_C(  1166.00), EASYSIMD_FLOAT32_C(   -89.00),
        EASYSIMD_FLOAT32_C(   639.00), EASYSIMD_FLOAT32_C(  -371.00), EASYSIMD_FLOAT32_C(   104.00), EASYSIMD_FLOAT32_C(  1265.00),
        EASYSIMD_FLOAT32_C(  -381.00), EASYSIMD_FLOAT32_C(   380.00), EASYSIMD_FLOAT32_C( -1550.00), EASYSIMD_FLOAT32_C( -1398.00),
        EASYSIMD_FLOAT32_C(  -631.00), EASYSIMD_FLOAT32_C(  1521.00), EASYSIMD_FLOAT32_C(  -279.00), EASYSIMD_FLOAT32_C(  1010.00) },
      { EASYSIMD_FLOAT32_C(  -465.00), EASYSIMD_FLOAT32_C(  -116.00), EASYSIMD_FLOAT32_C(  1166.00), EASYSIMD_FLOAT32_C(   -89.00),
        EASYSIMD_FLOAT32_C(   640.00), EASYSIMD_FLOAT32_C(  -371.00), EASYSIMD_FLOAT32_C(   105.00), EASYSIMD_FLOAT32_C(  1266.00),
        EASYSIMD_FLOAT32_C(  -382.00), EASYSIMD_FLOAT32_C(   380.00), EASYSIMD_FLOAT32_C( -1550.00), EASYSIMD_FLOAT32_C( -1398.00),
        EASYSIMD_FLOAT32_C(  -631.00), EASYSIMD_FLOAT32_C(  1522.00), EASYSIMD_FLOAT32_C(  -279.00), EASYSIMD_FLOAT32_C(  1010.00) } }
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

    r = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512_close(r, nearest_inf, 1);

    r = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512_close(r, neg_inf, 1);

    r = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512_close(r, pos_inf, 1);

    r = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_addn_round_ps");
    easysimd_assert_m512_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512 nearest_inf = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512 neg_inf = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512 pos_inf = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512 zero = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512 direction = easysimd_mm512_addn_round_ps(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

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
test_easysimd_mm512_addn_pd (EASYSIMD_MUNIT_TEST_ARGS) {
    #if 1
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -783.21), EASYSIMD_FLOAT64_C(   170.60), EASYSIMD_FLOAT64_C(   592.80), EASYSIMD_FLOAT64_C(  -505.87),
        EASYSIMD_FLOAT64_C(  -284.62), EASYSIMD_FLOAT64_C(   424.19), EASYSIMD_FLOAT64_C(   882.53), EASYSIMD_FLOAT64_C(   224.25) },
      { EASYSIMD_FLOAT64_C(  -405.01), EASYSIMD_FLOAT64_C(     9.12), EASYSIMD_FLOAT64_C(  -395.11), EASYSIMD_FLOAT64_C(  -913.11),
        EASYSIMD_FLOAT64_C(   913.86), EASYSIMD_FLOAT64_C(  -509.90), EASYSIMD_FLOAT64_C(  -615.05), EASYSIMD_FLOAT64_C(   204.19) },
      { EASYSIMD_FLOAT64_C(  1188.22), EASYSIMD_FLOAT64_C(  -179.72), EASYSIMD_FLOAT64_C(  -197.69), EASYSIMD_FLOAT64_C(  1418.98),
        EASYSIMD_FLOAT64_C(  -629.24), EASYSIMD_FLOAT64_C(    85.71), EASYSIMD_FLOAT64_C(  -267.48), EASYSIMD_FLOAT64_C(  -428.44) } },
    { { EASYSIMD_FLOAT64_C(   633.59), EASYSIMD_FLOAT64_C(   688.37), EASYSIMD_FLOAT64_C(  -734.80), EASYSIMD_FLOAT64_C(  -363.21),
        EASYSIMD_FLOAT64_C(   688.12), EASYSIMD_FLOAT64_C(   -82.98), EASYSIMD_FLOAT64_C(    21.00), EASYSIMD_FLOAT64_C(   -75.61) },
      { EASYSIMD_FLOAT64_C(   996.20), EASYSIMD_FLOAT64_C(  -846.99), EASYSIMD_FLOAT64_C(   401.72), EASYSIMD_FLOAT64_C(  -529.21),
        EASYSIMD_FLOAT64_C(   223.79), EASYSIMD_FLOAT64_C(   346.19), EASYSIMD_FLOAT64_C(  -359.73), EASYSIMD_FLOAT64_C(   440.58) },
      { EASYSIMD_FLOAT64_C( -1629.79), EASYSIMD_FLOAT64_C(   158.62), EASYSIMD_FLOAT64_C(   333.08), EASYSIMD_FLOAT64_C(   892.42),
        EASYSIMD_FLOAT64_C(  -911.91), EASYSIMD_FLOAT64_C(  -263.21), EASYSIMD_FLOAT64_C(   338.73), EASYSIMD_FLOAT64_C(  -364.97) } },
    { { EASYSIMD_FLOAT64_C(  -483.21), EASYSIMD_FLOAT64_C(  -766.93), EASYSIMD_FLOAT64_C(   934.70), EASYSIMD_FLOAT64_C(   232.17),
        EASYSIMD_FLOAT64_C(   657.26), EASYSIMD_FLOAT64_C(   817.24), EASYSIMD_FLOAT64_C(  -543.58), EASYSIMD_FLOAT64_C(  -747.75) },
      { EASYSIMD_FLOAT64_C(  -173.64), EASYSIMD_FLOAT64_C(    61.31), EASYSIMD_FLOAT64_C(  -660.86), EASYSIMD_FLOAT64_C(  -259.78),
        EASYSIMD_FLOAT64_C(   551.41), EASYSIMD_FLOAT64_C(  -275.91), EASYSIMD_FLOAT64_C(   944.40), EASYSIMD_FLOAT64_C(   185.00) },
      { EASYSIMD_FLOAT64_C(   656.85), EASYSIMD_FLOAT64_C(   705.62), EASYSIMD_FLOAT64_C(  -273.84), EASYSIMD_FLOAT64_C(    27.61),
        EASYSIMD_FLOAT64_C( -1208.67), EASYSIMD_FLOAT64_C(  -541.33), EASYSIMD_FLOAT64_C(  -400.82), EASYSIMD_FLOAT64_C(   562.75) } },
    { { EASYSIMD_FLOAT64_C(  -587.54), EASYSIMD_FLOAT64_C(  -790.40), EASYSIMD_FLOAT64_C(   821.79), EASYSIMD_FLOAT64_C(  -899.43),
        EASYSIMD_FLOAT64_C(   126.62), EASYSIMD_FLOAT64_C(  -157.22), EASYSIMD_FLOAT64_C(    24.96), EASYSIMD_FLOAT64_C(   122.82) },
      { EASYSIMD_FLOAT64_C(    -4.21), EASYSIMD_FLOAT64_C(  -573.31), EASYSIMD_FLOAT64_C(   593.61), EASYSIMD_FLOAT64_C(  -780.42),
        EASYSIMD_FLOAT64_C(   772.88), EASYSIMD_FLOAT64_C(  -766.12), EASYSIMD_FLOAT64_C(   660.16), EASYSIMD_FLOAT64_C(  -710.33) },
      { EASYSIMD_FLOAT64_C(   591.75), EASYSIMD_FLOAT64_C(  1363.71), EASYSIMD_FLOAT64_C( -1415.40), EASYSIMD_FLOAT64_C(  1679.85),
        EASYSIMD_FLOAT64_C(  -899.50), EASYSIMD_FLOAT64_C(   923.34), EASYSIMD_FLOAT64_C(  -685.12), EASYSIMD_FLOAT64_C(   587.51) } },
    { { EASYSIMD_FLOAT64_C(  -533.05), EASYSIMD_FLOAT64_C(   594.86), EASYSIMD_FLOAT64_C(   521.84), EASYSIMD_FLOAT64_C(  -875.78),
        EASYSIMD_FLOAT64_C(   412.10), EASYSIMD_FLOAT64_C(   978.26), EASYSIMD_FLOAT64_C(  -623.53), EASYSIMD_FLOAT64_C(  -761.54) },
      { EASYSIMD_FLOAT64_C(    39.57), EASYSIMD_FLOAT64_C(  -284.39), EASYSIMD_FLOAT64_C(   -21.32), EASYSIMD_FLOAT64_C(  -409.02),
        EASYSIMD_FLOAT64_C(   439.69), EASYSIMD_FLOAT64_C(   -76.91), EASYSIMD_FLOAT64_C(   775.98), EASYSIMD_FLOAT64_C(   852.15) },
      { EASYSIMD_FLOAT64_C(   493.48), EASYSIMD_FLOAT64_C(  -310.47), EASYSIMD_FLOAT64_C(  -500.52), EASYSIMD_FLOAT64_C(  1284.80),
        EASYSIMD_FLOAT64_C(  -851.79), EASYSIMD_FLOAT64_C(  -901.35), EASYSIMD_FLOAT64_C(  -152.45), EASYSIMD_FLOAT64_C(   -90.61) } },
    { { EASYSIMD_FLOAT64_C(   132.69), EASYSIMD_FLOAT64_C(   597.77), EASYSIMD_FLOAT64_C(   952.72), EASYSIMD_FLOAT64_C(  -740.69),
        EASYSIMD_FLOAT64_C(  -559.45), EASYSIMD_FLOAT64_C(   -22.32), EASYSIMD_FLOAT64_C(   382.13), EASYSIMD_FLOAT64_C(   436.34) },
      { EASYSIMD_FLOAT64_C(   404.37), EASYSIMD_FLOAT64_C(   -24.25), EASYSIMD_FLOAT64_C(   655.93), EASYSIMD_FLOAT64_C(   177.24),
        EASYSIMD_FLOAT64_C(   209.63), EASYSIMD_FLOAT64_C(   316.08), EASYSIMD_FLOAT64_C(   466.91), EASYSIMD_FLOAT64_C(   676.58) },
      { EASYSIMD_FLOAT64_C(  -537.06), EASYSIMD_FLOAT64_C(  -573.52), EASYSIMD_FLOAT64_C( -1608.65), EASYSIMD_FLOAT64_C(   563.45),
        EASYSIMD_FLOAT64_C(   349.82), EASYSIMD_FLOAT64_C(  -293.76), EASYSIMD_FLOAT64_C(  -849.04), EASYSIMD_FLOAT64_C( -1112.92) } },
    { { EASYSIMD_FLOAT64_C(   -89.05), EASYSIMD_FLOAT64_C(   -11.25), EASYSIMD_FLOAT64_C(   800.80), EASYSIMD_FLOAT64_C(  -676.95),
        EASYSIMD_FLOAT64_C(   -32.99), EASYSIMD_FLOAT64_C(  -822.73), EASYSIMD_FLOAT64_C(  -438.49), EASYSIMD_FLOAT64_C(  -993.42) },
      { EASYSIMD_FLOAT64_C(  -107.13), EASYSIMD_FLOAT64_C(   540.19), EASYSIMD_FLOAT64_C(  -402.44), EASYSIMD_FLOAT64_C(  -667.43),
        EASYSIMD_FLOAT64_C(  -536.72), EASYSIMD_FLOAT64_C(  -626.46), EASYSIMD_FLOAT64_C(  -815.28), EASYSIMD_FLOAT64_C(   595.97) },
      { EASYSIMD_FLOAT64_C(   196.18), EASYSIMD_FLOAT64_C(  -528.94), EASYSIMD_FLOAT64_C(  -398.36), EASYSIMD_FLOAT64_C(  1344.38),
        EASYSIMD_FLOAT64_C(   569.71), EASYSIMD_FLOAT64_C(  1449.19), EASYSIMD_FLOAT64_C(  1253.77), EASYSIMD_FLOAT64_C(   397.45) } },
    { { EASYSIMD_FLOAT64_C(   971.30), EASYSIMD_FLOAT64_C(  -862.56), EASYSIMD_FLOAT64_C(   855.28), EASYSIMD_FLOAT64_C(  -588.15),
        EASYSIMD_FLOAT64_C(   115.12), EASYSIMD_FLOAT64_C(   237.41), EASYSIMD_FLOAT64_C(   848.20), EASYSIMD_FLOAT64_C(  -480.51) },
      { EASYSIMD_FLOAT64_C(  -786.84), EASYSIMD_FLOAT64_C(   504.12), EASYSIMD_FLOAT64_C(   696.73), EASYSIMD_FLOAT64_C(   422.79),
        EASYSIMD_FLOAT64_C(  -179.79), EASYSIMD_FLOAT64_C(   163.64), EASYSIMD_FLOAT64_C(    99.38), EASYSIMD_FLOAT64_C(   731.15) },
      { EASYSIMD_FLOAT64_C(  -184.46), EASYSIMD_FLOAT64_C(   358.44), EASYSIMD_FLOAT64_C( -1552.01), EASYSIMD_FLOAT64_C(   165.36),
        EASYSIMD_FLOAT64_C(    64.67), EASYSIMD_FLOAT64_C(  -401.05), EASYSIMD_FLOAT64_C(  -947.58), EASYSIMD_FLOAT64_C(  -250.64) } }
  };


  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_addn_pd(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_addn_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d r = easysimd_mm512_addn_pd(a, b);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_addn_round_pd (EASYSIMD_MUNIT_TEST_ARGS) {
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
    { { EASYSIMD_FLOAT64_C(  -509.51), EASYSIMD_FLOAT64_C(  -727.44), EASYSIMD_FLOAT64_C(   842.03), EASYSIMD_FLOAT64_C(  -583.32),
        EASYSIMD_FLOAT64_C(  -478.42), EASYSIMD_FLOAT64_C(   922.84), EASYSIMD_FLOAT64_C(  -835.47), EASYSIMD_FLOAT64_C(   283.92) },
      { EASYSIMD_FLOAT64_C(   999.11), EASYSIMD_FLOAT64_C(   -75.04), EASYSIMD_FLOAT64_C(   520.00), EASYSIMD_FLOAT64_C(  -832.47),
        EASYSIMD_FLOAT64_C(  -658.00), EASYSIMD_FLOAT64_C(   299.77), EASYSIMD_FLOAT64_C(   161.99), EASYSIMD_FLOAT64_C(  -828.57) },
      { EASYSIMD_FLOAT64_C(  -490.00), EASYSIMD_FLOAT64_C(   802.00), EASYSIMD_FLOAT64_C( -1362.00), EASYSIMD_FLOAT64_C(  1416.00),
        EASYSIMD_FLOAT64_C(  1136.00), EASYSIMD_FLOAT64_C( -1223.00), EASYSIMD_FLOAT64_C(   673.00), EASYSIMD_FLOAT64_C(   545.00) },
      { EASYSIMD_FLOAT64_C(  -490.00), EASYSIMD_FLOAT64_C(   802.00), EASYSIMD_FLOAT64_C( -1363.00), EASYSIMD_FLOAT64_C(  1415.00),
        EASYSIMD_FLOAT64_C(  1136.00), EASYSIMD_FLOAT64_C( -1223.00), EASYSIMD_FLOAT64_C(   673.00), EASYSIMD_FLOAT64_C(   544.00) },
      { EASYSIMD_FLOAT64_C(  -489.00), EASYSIMD_FLOAT64_C(   803.00), EASYSIMD_FLOAT64_C( -1362.00), EASYSIMD_FLOAT64_C(  1416.00),
        EASYSIMD_FLOAT64_C(  1137.00), EASYSIMD_FLOAT64_C( -1222.00), EASYSIMD_FLOAT64_C(   674.00), EASYSIMD_FLOAT64_C(   545.00) },
      { EASYSIMD_FLOAT64_C(  -489.00), EASYSIMD_FLOAT64_C(   802.00), EASYSIMD_FLOAT64_C( -1362.00), EASYSIMD_FLOAT64_C(  1415.00),
        EASYSIMD_FLOAT64_C(  1136.00), EASYSIMD_FLOAT64_C( -1222.00), EASYSIMD_FLOAT64_C(   673.00), EASYSIMD_FLOAT64_C(   544.00) },
      { EASYSIMD_FLOAT64_C(  -490.00), EASYSIMD_FLOAT64_C(   802.00), EASYSIMD_FLOAT64_C( -1362.00), EASYSIMD_FLOAT64_C(  1416.00),
        EASYSIMD_FLOAT64_C(  1136.00), EASYSIMD_FLOAT64_C( -1223.00), EASYSIMD_FLOAT64_C(   673.00), EASYSIMD_FLOAT64_C(   545.00) } },
    { { EASYSIMD_FLOAT64_C(  -565.65), EASYSIMD_FLOAT64_C(   -75.34), EASYSIMD_FLOAT64_C(   212.81), EASYSIMD_FLOAT64_C(   676.83),
        EASYSIMD_FLOAT64_C(  -104.74), EASYSIMD_FLOAT64_C(   438.89), EASYSIMD_FLOAT64_C(   -47.41), EASYSIMD_FLOAT64_C(   -27.86) },
      { EASYSIMD_FLOAT64_C(   915.72), EASYSIMD_FLOAT64_C(   904.65), EASYSIMD_FLOAT64_C(   946.06), EASYSIMD_FLOAT64_C(   401.55),
        EASYSIMD_FLOAT64_C(   766.97), EASYSIMD_FLOAT64_C(  -459.53), EASYSIMD_FLOAT64_C(  -612.55), EASYSIMD_FLOAT64_C(  -742.54) },
      { EASYSIMD_FLOAT64_C(  -350.00), EASYSIMD_FLOAT64_C(  -829.00), EASYSIMD_FLOAT64_C( -1159.00), EASYSIMD_FLOAT64_C( -1078.00),
        EASYSIMD_FLOAT64_C(  -662.00), EASYSIMD_FLOAT64_C(    21.00), EASYSIMD_FLOAT64_C(   660.00), EASYSIMD_FLOAT64_C(   770.00) },
      { EASYSIMD_FLOAT64_C(  -351.00), EASYSIMD_FLOAT64_C(  -830.00), EASYSIMD_FLOAT64_C( -1159.00), EASYSIMD_FLOAT64_C( -1079.00),
        EASYSIMD_FLOAT64_C(  -663.00), EASYSIMD_FLOAT64_C(    20.00), EASYSIMD_FLOAT64_C(   659.00), EASYSIMD_FLOAT64_C(   770.00) },
      { EASYSIMD_FLOAT64_C(  -350.00), EASYSIMD_FLOAT64_C(  -829.00), EASYSIMD_FLOAT64_C( -1158.00), EASYSIMD_FLOAT64_C( -1078.00),
        EASYSIMD_FLOAT64_C(  -662.00), EASYSIMD_FLOAT64_C(    21.00), EASYSIMD_FLOAT64_C(   660.00), EASYSIMD_FLOAT64_C(   771.00) },
      { EASYSIMD_FLOAT64_C(  -350.00), EASYSIMD_FLOAT64_C(  -829.00), EASYSIMD_FLOAT64_C( -1158.00), EASYSIMD_FLOAT64_C( -1078.00),
        EASYSIMD_FLOAT64_C(  -662.00), EASYSIMD_FLOAT64_C(    20.00), EASYSIMD_FLOAT64_C(   659.00), EASYSIMD_FLOAT64_C(   770.00) },
      { EASYSIMD_FLOAT64_C(  -350.00), EASYSIMD_FLOAT64_C(  -829.00), EASYSIMD_FLOAT64_C( -1159.00), EASYSIMD_FLOAT64_C( -1078.00),
        EASYSIMD_FLOAT64_C(  -662.00), EASYSIMD_FLOAT64_C(    21.00), EASYSIMD_FLOAT64_C(   660.00), EASYSIMD_FLOAT64_C(   770.00) } },
    { { EASYSIMD_FLOAT64_C(  -186.96), EASYSIMD_FLOAT64_C(  -770.51), EASYSIMD_FLOAT64_C(  -325.86), EASYSIMD_FLOAT64_C(   334.62),
        EASYSIMD_FLOAT64_C(  -847.67), EASYSIMD_FLOAT64_C(  -161.33), EASYSIMD_FLOAT64_C(  -381.47), EASYSIMD_FLOAT64_C(  -848.56) },
      { EASYSIMD_FLOAT64_C(   763.63), EASYSIMD_FLOAT64_C(  -861.47), EASYSIMD_FLOAT64_C(  -681.02), EASYSIMD_FLOAT64_C(  -894.37),
        EASYSIMD_FLOAT64_C(   438.30), EASYSIMD_FLOAT64_C(   480.96), EASYSIMD_FLOAT64_C(  -722.94), EASYSIMD_FLOAT64_C(   872.65) },
      { EASYSIMD_FLOAT64_C(  -577.00), EASYSIMD_FLOAT64_C(  1632.00), EASYSIMD_FLOAT64_C(  1007.00), EASYSIMD_FLOAT64_C(   560.00),
        EASYSIMD_FLOAT64_C(   409.00), EASYSIMD_FLOAT64_C(  -320.00), EASYSIMD_FLOAT64_C(  1104.00), EASYSIMD_FLOAT64_C(   -24.00) },
      { EASYSIMD_FLOAT64_C(  -577.00), EASYSIMD_FLOAT64_C(  1631.00), EASYSIMD_FLOAT64_C(  1006.00), EASYSIMD_FLOAT64_C(   559.00),
        EASYSIMD_FLOAT64_C(   409.00), EASYSIMD_FLOAT64_C(  -320.00), EASYSIMD_FLOAT64_C(  1104.00), EASYSIMD_FLOAT64_C(   -25.00) },
      { EASYSIMD_FLOAT64_C(  -576.00), EASYSIMD_FLOAT64_C(  1632.00), EASYSIMD_FLOAT64_C(  1007.00), EASYSIMD_FLOAT64_C(   560.00),
        EASYSIMD_FLOAT64_C(   410.00), EASYSIMD_FLOAT64_C(  -319.00), EASYSIMD_FLOAT64_C(  1105.00), EASYSIMD_FLOAT64_C(   -24.00) },
      { EASYSIMD_FLOAT64_C(  -576.00), EASYSIMD_FLOAT64_C(  1631.00), EASYSIMD_FLOAT64_C(  1006.00), EASYSIMD_FLOAT64_C(   559.00),
        EASYSIMD_FLOAT64_C(   409.00), EASYSIMD_FLOAT64_C(  -319.00), EASYSIMD_FLOAT64_C(  1104.00), EASYSIMD_FLOAT64_C(   -24.00) },
      { EASYSIMD_FLOAT64_C(  -577.00), EASYSIMD_FLOAT64_C(  1632.00), EASYSIMD_FLOAT64_C(  1007.00), EASYSIMD_FLOAT64_C(   560.00),
        EASYSIMD_FLOAT64_C(   409.00), EASYSIMD_FLOAT64_C(  -320.00), EASYSIMD_FLOAT64_C(  1104.00), EASYSIMD_FLOAT64_C(   -24.00) } },
    { { EASYSIMD_FLOAT64_C(  -594.38), EASYSIMD_FLOAT64_C(   489.87), EASYSIMD_FLOAT64_C(   549.47), EASYSIMD_FLOAT64_C(   300.88),
        EASYSIMD_FLOAT64_C(   -71.24), EASYSIMD_FLOAT64_C(  -497.93), EASYSIMD_FLOAT64_C(  -726.98), EASYSIMD_FLOAT64_C(  -155.52) },
      { EASYSIMD_FLOAT64_C(  -593.28), EASYSIMD_FLOAT64_C(  -780.92), EASYSIMD_FLOAT64_C(  -753.98), EASYSIMD_FLOAT64_C(  -826.31),
        EASYSIMD_FLOAT64_C(  -240.45), EASYSIMD_FLOAT64_C(  -366.52), EASYSIMD_FLOAT64_C(  -568.85), EASYSIMD_FLOAT64_C(   572.59) },
      { EASYSIMD_FLOAT64_C(  1188.00), EASYSIMD_FLOAT64_C(   291.00), EASYSIMD_FLOAT64_C(   205.00), EASYSIMD_FLOAT64_C(   525.00),
        EASYSIMD_FLOAT64_C(   312.00), EASYSIMD_FLOAT64_C(   864.00), EASYSIMD_FLOAT64_C(  1296.00), EASYSIMD_FLOAT64_C(  -417.00) },
      { EASYSIMD_FLOAT64_C(  1187.00), EASYSIMD_FLOAT64_C(   291.00), EASYSIMD_FLOAT64_C(   204.00), EASYSIMD_FLOAT64_C(   525.00),
        EASYSIMD_FLOAT64_C(   311.00), EASYSIMD_FLOAT64_C(   864.00), EASYSIMD_FLOAT64_C(  1295.00), EASYSIMD_FLOAT64_C(  -418.00) },
      { EASYSIMD_FLOAT64_C(  1188.00), EASYSIMD_FLOAT64_C(   292.00), EASYSIMD_FLOAT64_C(   205.00), EASYSIMD_FLOAT64_C(   526.00),
        EASYSIMD_FLOAT64_C(   312.00), EASYSIMD_FLOAT64_C(   865.00), EASYSIMD_FLOAT64_C(  1296.00), EASYSIMD_FLOAT64_C(  -417.00) },
      { EASYSIMD_FLOAT64_C(  1187.00), EASYSIMD_FLOAT64_C(   291.00), EASYSIMD_FLOAT64_C(   204.00), EASYSIMD_FLOAT64_C(   525.00),
        EASYSIMD_FLOAT64_C(   311.00), EASYSIMD_FLOAT64_C(   864.00), EASYSIMD_FLOAT64_C(  1295.00), EASYSIMD_FLOAT64_C(  -417.00) },
      { EASYSIMD_FLOAT64_C(  1188.00), EASYSIMD_FLOAT64_C(   291.00), EASYSIMD_FLOAT64_C(   205.00), EASYSIMD_FLOAT64_C(   525.00),
        EASYSIMD_FLOAT64_C(   312.00), EASYSIMD_FLOAT64_C(   864.00), EASYSIMD_FLOAT64_C(  1296.00), EASYSIMD_FLOAT64_C(  -417.00) } },
    { { EASYSIMD_FLOAT64_C(  -137.04), EASYSIMD_FLOAT64_C(   105.29), EASYSIMD_FLOAT64_C(   -92.80), EASYSIMD_FLOAT64_C(    15.29),
        EASYSIMD_FLOAT64_C(   943.96), EASYSIMD_FLOAT64_C(   525.74), EASYSIMD_FLOAT64_C(   166.73), EASYSIMD_FLOAT64_C(   707.58) },
      { EASYSIMD_FLOAT64_C(   664.27), EASYSIMD_FLOAT64_C(   485.71), EASYSIMD_FLOAT64_C(   813.21), EASYSIMD_FLOAT64_C(   102.57),
        EASYSIMD_FLOAT64_C(   -33.33), EASYSIMD_FLOAT64_C(  -909.73), EASYSIMD_FLOAT64_C(   -24.79), EASYSIMD_FLOAT64_C(   372.29) },
      { EASYSIMD_FLOAT64_C(  -527.00), EASYSIMD_FLOAT64_C(  -591.00), EASYSIMD_FLOAT64_C(  -720.00), EASYSIMD_FLOAT64_C(  -118.00),
        EASYSIMD_FLOAT64_C(  -911.00), EASYSIMD_FLOAT64_C(   384.00), EASYSIMD_FLOAT64_C(  -142.00), EASYSIMD_FLOAT64_C( -1080.00) },
      { EASYSIMD_FLOAT64_C(  -528.00), EASYSIMD_FLOAT64_C(  -591.00), EASYSIMD_FLOAT64_C(  -721.00), EASYSIMD_FLOAT64_C(  -118.00),
        EASYSIMD_FLOAT64_C(  -911.00), EASYSIMD_FLOAT64_C(   383.00), EASYSIMD_FLOAT64_C(  -142.00), EASYSIMD_FLOAT64_C( -1080.00) },
      { EASYSIMD_FLOAT64_C(  -527.00), EASYSIMD_FLOAT64_C(  -591.00), EASYSIMD_FLOAT64_C(  -720.00), EASYSIMD_FLOAT64_C(  -117.00),
        EASYSIMD_FLOAT64_C(  -910.00), EASYSIMD_FLOAT64_C(   384.00), EASYSIMD_FLOAT64_C(  -141.00), EASYSIMD_FLOAT64_C( -1079.00) },
      { EASYSIMD_FLOAT64_C(  -527.00), EASYSIMD_FLOAT64_C(  -591.00), EASYSIMD_FLOAT64_C(  -720.00), EASYSIMD_FLOAT64_C(  -117.00),
        EASYSIMD_FLOAT64_C(  -910.00), EASYSIMD_FLOAT64_C(   383.00), EASYSIMD_FLOAT64_C(  -141.00), EASYSIMD_FLOAT64_C( -1079.00) },
      { EASYSIMD_FLOAT64_C(  -527.00), EASYSIMD_FLOAT64_C(  -591.00), EASYSIMD_FLOAT64_C(  -720.00), EASYSIMD_FLOAT64_C(  -118.00),
        EASYSIMD_FLOAT64_C(  -911.00), EASYSIMD_FLOAT64_C(   384.00), EASYSIMD_FLOAT64_C(  -142.00), EASYSIMD_FLOAT64_C( -1080.00) } },
    { { EASYSIMD_FLOAT64_C(   580.14), EASYSIMD_FLOAT64_C(  -475.31), EASYSIMD_FLOAT64_C(  -326.83), EASYSIMD_FLOAT64_C(  -491.10),
        EASYSIMD_FLOAT64_C(    26.76), EASYSIMD_FLOAT64_C(   -53.81), EASYSIMD_FLOAT64_C(   353.38), EASYSIMD_FLOAT64_C(   433.48) },
      { EASYSIMD_FLOAT64_C(   165.27), EASYSIMD_FLOAT64_C(   599.40), EASYSIMD_FLOAT64_C(   607.17), EASYSIMD_FLOAT64_C(   924.82),
        EASYSIMD_FLOAT64_C(  -767.12), EASYSIMD_FLOAT64_C(  -961.68), EASYSIMD_FLOAT64_C(   497.41), EASYSIMD_FLOAT64_C(    95.84) },
      { EASYSIMD_FLOAT64_C(  -745.00), EASYSIMD_FLOAT64_C(  -124.00), EASYSIMD_FLOAT64_C(  -280.00), EASYSIMD_FLOAT64_C(  -434.00),
        EASYSIMD_FLOAT64_C(   740.00), EASYSIMD_FLOAT64_C(  1015.00), EASYSIMD_FLOAT64_C(  -851.00), EASYSIMD_FLOAT64_C(  -529.00) },
      { EASYSIMD_FLOAT64_C(  -746.00), EASYSIMD_FLOAT64_C(  -125.00), EASYSIMD_FLOAT64_C(  -281.00), EASYSIMD_FLOAT64_C(  -434.00),
        EASYSIMD_FLOAT64_C(   740.00), EASYSIMD_FLOAT64_C(  1015.00), EASYSIMD_FLOAT64_C(  -851.00), EASYSIMD_FLOAT64_C(  -530.00) },
      { EASYSIMD_FLOAT64_C(  -745.00), EASYSIMD_FLOAT64_C(  -124.00), EASYSIMD_FLOAT64_C(  -280.00), EASYSIMD_FLOAT64_C(  -433.00),
        EASYSIMD_FLOAT64_C(   741.00), EASYSIMD_FLOAT64_C(  1016.00), EASYSIMD_FLOAT64_C(  -850.00), EASYSIMD_FLOAT64_C(  -529.00) },
      { EASYSIMD_FLOAT64_C(  -745.00), EASYSIMD_FLOAT64_C(  -124.00), EASYSIMD_FLOAT64_C(  -280.00), EASYSIMD_FLOAT64_C(  -433.00),
        EASYSIMD_FLOAT64_C(   740.00), EASYSIMD_FLOAT64_C(  1015.00), EASYSIMD_FLOAT64_C(  -850.00), EASYSIMD_FLOAT64_C(  -529.00) },
      { EASYSIMD_FLOAT64_C(  -745.00), EASYSIMD_FLOAT64_C(  -124.00), EASYSIMD_FLOAT64_C(  -280.00), EASYSIMD_FLOAT64_C(  -434.00),
        EASYSIMD_FLOAT64_C(   740.00), EASYSIMD_FLOAT64_C(  1015.00), EASYSIMD_FLOAT64_C(  -851.00), EASYSIMD_FLOAT64_C(  -529.00) } },
    { { EASYSIMD_FLOAT64_C(   143.61), EASYSIMD_FLOAT64_C(  -595.39), EASYSIMD_FLOAT64_C(  -888.87), EASYSIMD_FLOAT64_C(    87.56),
        EASYSIMD_FLOAT64_C(   930.35), EASYSIMD_FLOAT64_C(   277.87), EASYSIMD_FLOAT64_C(  -204.86), EASYSIMD_FLOAT64_C(   594.61) },
      { EASYSIMD_FLOAT64_C(  -236.42), EASYSIMD_FLOAT64_C(  -391.64), EASYSIMD_FLOAT64_C(  -302.82), EASYSIMD_FLOAT64_C(   730.25),
        EASYSIMD_FLOAT64_C(  -301.37), EASYSIMD_FLOAT64_C(   672.39), EASYSIMD_FLOAT64_C(   102.54), EASYSIMD_FLOAT64_C(  -721.23) },
      { EASYSIMD_FLOAT64_C(    93.00), EASYSIMD_FLOAT64_C(   987.00), EASYSIMD_FLOAT64_C(  1192.00), EASYSIMD_FLOAT64_C(  -818.00),
        EASYSIMD_FLOAT64_C(  -629.00), EASYSIMD_FLOAT64_C(  -950.00), EASYSIMD_FLOAT64_C(   102.00), EASYSIMD_FLOAT64_C(   127.00) },
      { EASYSIMD_FLOAT64_C(    92.00), EASYSIMD_FLOAT64_C(   987.00), EASYSIMD_FLOAT64_C(  1191.00), EASYSIMD_FLOAT64_C(  -818.00),
        EASYSIMD_FLOAT64_C(  -629.00), EASYSIMD_FLOAT64_C(  -951.00), EASYSIMD_FLOAT64_C(   102.00), EASYSIMD_FLOAT64_C(   126.00) },
      { EASYSIMD_FLOAT64_C(    93.00), EASYSIMD_FLOAT64_C(   988.00), EASYSIMD_FLOAT64_C(  1192.00), EASYSIMD_FLOAT64_C(  -817.00),
        EASYSIMD_FLOAT64_C(  -628.00), EASYSIMD_FLOAT64_C(  -950.00), EASYSIMD_FLOAT64_C(   103.00), EASYSIMD_FLOAT64_C(   127.00) },
      { EASYSIMD_FLOAT64_C(    92.00), EASYSIMD_FLOAT64_C(   987.00), EASYSIMD_FLOAT64_C(  1191.00), EASYSIMD_FLOAT64_C(  -817.00),
        EASYSIMD_FLOAT64_C(  -628.00), EASYSIMD_FLOAT64_C(  -950.00), EASYSIMD_FLOAT64_C(   102.00), EASYSIMD_FLOAT64_C(   126.00) },
      { EASYSIMD_FLOAT64_C(    93.00), EASYSIMD_FLOAT64_C(   987.00), EASYSIMD_FLOAT64_C(  1192.00), EASYSIMD_FLOAT64_C(  -818.00),
        EASYSIMD_FLOAT64_C(  -629.00), EASYSIMD_FLOAT64_C(  -950.00), EASYSIMD_FLOAT64_C(   102.00), EASYSIMD_FLOAT64_C(   127.00) } },
    { { EASYSIMD_FLOAT64_C(  -802.92), EASYSIMD_FLOAT64_C(   775.71), EASYSIMD_FLOAT64_C(  -212.33), EASYSIMD_FLOAT64_C(   223.84),
        EASYSIMD_FLOAT64_C(  -278.10), EASYSIMD_FLOAT64_C(  -858.95), EASYSIMD_FLOAT64_C(  -342.68), EASYSIMD_FLOAT64_C(   887.17) },
      { EASYSIMD_FLOAT64_C(   740.45), EASYSIMD_FLOAT64_C(  -735.51), EASYSIMD_FLOAT64_C(   811.99), EASYSIMD_FLOAT64_C(   973.32),
        EASYSIMD_FLOAT64_C(  -697.19), EASYSIMD_FLOAT64_C(   309.39), EASYSIMD_FLOAT64_C(    69.16), EASYSIMD_FLOAT64_C(   446.41) },
      { EASYSIMD_FLOAT64_C(    62.00), EASYSIMD_FLOAT64_C(   -40.00), EASYSIMD_FLOAT64_C(  -600.00), EASYSIMD_FLOAT64_C( -1197.00),
        EASYSIMD_FLOAT64_C(   975.00), EASYSIMD_FLOAT64_C(   550.00), EASYSIMD_FLOAT64_C(   274.00), EASYSIMD_FLOAT64_C( -1334.00) },
      { EASYSIMD_FLOAT64_C(    62.00), EASYSIMD_FLOAT64_C(   -41.00), EASYSIMD_FLOAT64_C(  -600.00), EASYSIMD_FLOAT64_C( -1198.00),
        EASYSIMD_FLOAT64_C(   975.00), EASYSIMD_FLOAT64_C(   549.00), EASYSIMD_FLOAT64_C(   273.00), EASYSIMD_FLOAT64_C( -1334.00) },
      { EASYSIMD_FLOAT64_C(    63.00), EASYSIMD_FLOAT64_C(   -40.00), EASYSIMD_FLOAT64_C(  -599.00), EASYSIMD_FLOAT64_C( -1197.00),
        EASYSIMD_FLOAT64_C(   976.00), EASYSIMD_FLOAT64_C(   550.00), EASYSIMD_FLOAT64_C(   274.00), EASYSIMD_FLOAT64_C( -1333.00) },
      { EASYSIMD_FLOAT64_C(    62.00), EASYSIMD_FLOAT64_C(   -40.00), EASYSIMD_FLOAT64_C(  -599.00), EASYSIMD_FLOAT64_C( -1197.00),
        EASYSIMD_FLOAT64_C(   975.00), EASYSIMD_FLOAT64_C(   549.00), EASYSIMD_FLOAT64_C(   273.00), EASYSIMD_FLOAT64_C( -1333.00) },
      { EASYSIMD_FLOAT64_C(    62.00), EASYSIMD_FLOAT64_C(   -40.00), EASYSIMD_FLOAT64_C(  -600.00), EASYSIMD_FLOAT64_C( -1197.00),
        EASYSIMD_FLOAT64_C(   975.00), EASYSIMD_FLOAT64_C(   550.00), EASYSIMD_FLOAT64_C(   274.00), EASYSIMD_FLOAT64_C( -1334.00) } }
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

    r = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512d_close(r, nearest_inf, 1);

    r = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512d_close(r, neg_inf, 1);

    r = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512d_close(r, pos_inf, 1);

    r = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512d_close(r, zero, 1);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_addn_round_pd");
    easysimd_assert_m512d_close(r, direction, 1);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));

    easysimd__m512d nearest_inf = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512d neg_inf = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512d pos_inf = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512d zero = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512d direction = easysimd_mm512_addn_round_pd(a, b, EASYSIMD_MM_FROUND_CUR_DIRECTION);

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
test_easysimd_mm512_addsetc_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
    const easysimd__mmask16 k2_res;
  } test_vec[] = {
    { { -INT32_C(   113696077), -INT32_C(  1876844262), -INT32_C(  2000018427),  INT32_C(  1923166776), -INT32_C(   695242621), -INT32_C(   918719560),  INT32_C(   400629486), -INT32_C(   716666591),
        -INT32_C(   976322133),  INT32_C(   676785954),  INT32_C(  1286610964), -INT32_C(   708947374),  INT32_C(  2058046914), -INT32_C(  1371281216),  INT32_C(   684074247),  INT32_C(  1509756590) },
      { -INT32_C(  1306604657), -INT32_C(   807766853), -INT32_C(   417625452), -INT32_C(  1631790628), -INT32_C(   434542554),  INT32_C(  1469340752),  INT32_C(   796940929), -INT32_C(   125272728),
         INT32_C(    78292809), -INT32_C(  1328315365), -INT32_C(   342364658), -INT32_C(   309767225),  INT32_C(   215196347), -INT32_C(  2140968961),  INT32_C(   716235714), -INT32_C(  1440597919) },
      { -INT32_C(  1420300734),  INT32_C(  1610356181),  INT32_C(  1877323417),  INT32_C(   291376148), -INT32_C(  1129785175),  INT32_C(   550621192),  INT32_C(  1197570415), -INT32_C(   841939319),
        -INT32_C(   898029324), -INT32_C(   651529411),  INT32_C(   944246306), -INT32_C(  1018714599), -INT32_C(  2021724035),  INT32_C(   782717119),  INT32_C(  1400309961),  INT32_C(    69158671) },
      UINT16_C(44223) },
    { { -INT32_C(    72430369),  INT32_C(  1605075280),  INT32_C(   910836335),  INT32_C(  1361302421),  INT32_C(  1969092214),  INT32_C(   536199261),  INT32_C(    88778148), -INT32_C(  1112576802),
        -INT32_C(  1984406215),  INT32_C(  1307075550),  INT32_C(   998453925),  INT32_C(  2089592326), -INT32_C(   101586532),  INT32_C(  1293543081),  INT32_C(  1783784332),  INT32_C(   136774095) },
      {  INT32_C(  1016258654), -INT32_C(   376866237), -INT32_C(  1289483091),  INT32_C(  1311748274),  INT32_C(  1112023193), -INT32_C(  1836031738), -INT32_C(  1812143420),  INT32_C(  1117529316),
         INT32_C(  1199517188),  INT32_C(  1429211304), -INT32_C(   955755499), -INT32_C(  1659488508),  INT32_C(  1591762519), -INT32_C(  2081394497),  INT32_C(   907472210),  INT32_C(   360296977) },
      {  INT32_C(   943828285),  INT32_C(  1228209043), -INT32_C(   378646756), -INT32_C(  1621916601), -INT32_C(  1213851889), -INT32_C(  1299832477), -INT32_C(  1723365272),  INT32_C(     4952514),
        -INT32_C(   784889027), -INT32_C(  1558680442),  INT32_C(    42698426),  INT32_C(   430103818),  INT32_C(  1490175987), -INT32_C(   787851416), -INT32_C(  1603710754),  INT32_C(   497071072) },
      UINT16_C( 7299) },
    { { -INT32_C(  1990395680),  INT32_C(   366906624), -INT32_C(   421664799),  INT32_C(  1988358942),  INT32_C(   282354513),  INT32_C(   647218387), -INT32_C(  1034114639),  INT32_C(  1020777820),
        -INT32_C(   825936691), -INT32_C(  1562139455), -INT32_C(  1450655605),  INT32_C(    69143731),  INT32_C(  1125446511),  INT32_C(  1768531896), -INT32_C(  1389574831),  INT32_C(  1760101275) },
      { -INT32_C(   130633929), -INT32_C(   560260525), -INT32_C(  1903746086), -INT32_C(  1617778897),  INT32_C(  1390585754), -INT32_C(  1615049906), -INT32_C(  1421023216),  INT32_C(   588527339),
         INT32_C(   941312741),  INT32_C(  1058453092),  INT32_C(   164470234), -INT32_C(   542613692),  INT32_C(  1429309958), -INT32_C(   420155947), -INT32_C(  1047445035),  INT32_C(  1558488439) },
      { -INT32_C(  2121029609), -INT32_C(   193353901),  INT32_C(  1969556411),  INT32_C(   370580045),  INT32_C(  1672940267), -INT32_C(   967831519),  INT32_C(  1839829441),  INT32_C(  1609305159),
         INT32_C(   115376050), -INT32_C(   503686363), -INT32_C(  1286185371), -INT32_C(   473469961), -INT32_C(  1740210827),  INT32_C(  1348375949),  INT32_C(  1857947430), -INT32_C(   976377582) },
      UINT16_C(24909) },
    { {  INT32_C(  1419050992), -INT32_C(  1869370698), -INT32_C(  1936105401), -INT32_C(   966049088), -INT32_C(  1575248692),  INT32_C(  1602752650), -INT32_C(   937420463), -INT32_C(  1356593985),
        -INT32_C(  1174161404), -INT32_C(  1454729374), -INT32_C(  1221205001), -INT32_C(   226582491), -INT32_C(   946562755), -INT32_C(    98165591), -INT32_C(   188594379),  INT32_C(  1336141387) },
      {  INT32_C(      632734),  INT32_C(   900354878),  INT32_C(  1559093047), -INT32_C(  1118934144), -INT32_C(  1367023100),  INT32_C(   883469054),  INT32_C(  1026059249), -INT32_C(   259208111),
        -INT32_C(  1309632909),  INT32_C(   551983849), -INT32_C(    92417158),  INT32_C(  1136118590), -INT32_C(  1393476690), -INT32_C(   673146395),  INT32_C(  1444153604),  INT32_C(  1212588245) },
      {  INT32_C(  1419683726), -INT32_C(   969015820), -INT32_C(   377012354), -INT32_C(  2084983232),  INT32_C(  1352695504), -INT32_C(  1808745592),  INT32_C(    88638786), -INT32_C(  1615802096),
         INT32_C(  1811172983), -INT32_C(   902745525), -INT32_C(  1313622159),  INT32_C(   909536099),  INT32_C(  1954927851), -INT32_C(   771311986),  INT32_C(  1255559225), -INT32_C(  1746237664) },
      UINT16_C(32216) },
    { {  INT32_C(   553203254),  INT32_C(  1262542801), -INT32_C(   247087693),  INT32_C(   926219401),  INT32_C(   484648247), -INT32_C(  1007434561), -INT32_C(  1575417907), -INT32_C(   555065432),
         INT32_C(  1744757654),  INT32_C(  1974616002), -INT32_C(  2056783876),  INT32_C(   717003763), -INT32_C(  2142854975),  INT32_C(   826489444), -INT32_C(   372023999),  INT32_C(  1388887484) },
      {  INT32_C(  1656342176),  INT32_C(    47737605),  INT32_C(  1434926946), -INT32_C(  1686158118),  INT32_C(  1209779940),  INT32_C(  1098473216),  INT32_C(  2016103612), -INT32_C(  1446251767),
        -INT32_C(  1089764167),  INT32_C(  1388438512), -INT32_C(    39303134),  INT32_C(  1889019788), -INT32_C(   306662163), -INT32_C(   818990829), -INT32_C(  2042144131),  INT32_C(   103748173) },
      { -INT32_C(  2085421866),  INT32_C(  1310280406),  INT32_C(  1187839253), -INT32_C(   759938717),  INT32_C(  1694428187),  INT32_C(    91038655),  INT32_C(   440685705), -INT32_C(  2001317199),
         INT32_C(   654993487), -INT32_C(   931912782), -INT32_C(  2096087010), -INT32_C(  1688943745),  INT32_C(  1845450158),  INT32_C(     7498615),  INT32_C(  1880799166),  INT32_C(  1492635657) },
      UINT16_C(30180) },
    { { -INT32_C(  2017117289),  INT32_C(  1104774686),  INT32_C(  1530823119), -INT32_C(  1764960599), -INT32_C(  1652259702),  INT32_C(   862761910),  INT32_C(  1522185229),  INT32_C(  1566632390),
         INT32_C(  1139025444),  INT32_C(  2072297132), -INT32_C(   388513217),  INT32_C(   595567512), -INT32_C(   574618841), -INT32_C(  1022284362), -INT32_C(  1491219487), -INT32_C(   653951564) },
      {  INT32_C(  1344072099), -INT32_C(   406085465), -INT32_C(    87055774),  INT32_C(  1830637125),  INT32_C(   122347089), -INT32_C(   322282741), -INT32_C(   611064026),  INT32_C(   146053221),
         INT32_C(   693686402), -INT32_C(   770694032),  INT32_C(   214753478), -INT32_C(  2139493842), -INT32_C(   746077240),  INT32_C(  1170166303), -INT32_C(  1642048967),  INT32_C(  1839715563) },
      { -INT32_C(   673045190),  INT32_C(   698689221),  INT32_C(  1443767345),  INT32_C(    65676526), -INT32_C(  1529912613),  INT32_C(   540479169),  INT32_C(   911121203),  INT32_C(  1712685611),
         INT32_C(  1832711846),  INT32_C(  1301603100), -INT32_C(   173759739), -INT32_C(  1543926330), -INT32_C(  1320696081),  INT32_C(   147881941),  INT32_C(  1161698842),  INT32_C(  1185763999) },
      UINT16_C(62062) },
    { {  INT32_C(   345440164), -INT32_C(   353982685), -INT32_C(  1242123385),  INT32_C(  1698000797),  INT32_C(  1362672946),  INT32_C(  1217918735),  INT32_C(   887600969),  INT32_C(   815894156),
        -INT32_C(  1320863603),  INT32_C(  1721445343),  INT32_C(  2065404382),  INT32_C(   853561600),  INT32_C(   495196174),  INT32_C(  1499798287),  INT32_C(  1586318546),  INT32_C(  1754214362) },
      {  INT32_C(  1176097895), -INT32_C(   559106817),  INT32_C(  1163511621),  INT32_C(   645347864),  INT32_C(  1648622418), -INT32_C(   373577450), -INT32_C(   800634635), -INT32_C(   566700425),
        -INT32_C(  1440460374),  INT32_C(  1250480133), -INT32_C(  1332747880),  INT32_C(  1859585563),  INT32_C(   416291330), -INT32_C(  1191081021),  INT32_C(  1267288531), -INT32_C(   903233249) },
      {  INT32_C(  1521538059), -INT32_C(   913089502), -INT32_C(    78611764), -INT32_C(  1951618635), -INT32_C(  1283671932),  INT32_C(   844341285),  INT32_C(    86966334),  INT32_C(   249193731),
         INT32_C(  1533643319), -INT32_C(  1323041820),  INT32_C(   732656502), -INT32_C(  1581820133),  INT32_C(   911487504),  INT32_C(   308717266), -INT32_C(  1441360219),  INT32_C(   850981113) },
      UINT16_C(42466) },
    { {  INT32_C(   393498130), -INT32_C(  1235092450), -INT32_C(   110628643), -INT32_C(   127451402),  INT32_C(   470890328), -INT32_C(  1781198142),  INT32_C(  2078301787),  INT32_C(   826608159),
         INT32_C(  1984477528), -INT32_C(  1842501195), -INT32_C(  1886677863),  INT32_C(   713618130), -INT32_C(   347694807),  INT32_C(   125901739), -INT32_C(  1736285831), -INT32_C(  1010186389) },
      {  INT32_C(   892998016),  INT32_C(  1405577146), -INT32_C(   840739845),  INT32_C(  1878485829), -INT32_C(  1353040380), -INT32_C(   742991015), -INT32_C(  1469368259),  INT32_C(  2137798143),
         INT32_C(    28616262),  INT32_C(   139754509),  INT32_C(   349517775), -INT32_C(  1501311838),  INT32_C(  1683414538), -INT32_C(   164164423),  INT32_C(  1151312453),  INT32_C(   516164567) },
      {  INT32_C(  1286496146),  INT32_C(   170484696), -INT32_C(   951368488),  INT32_C(  1751034427), -INT32_C(   882150052),  INT32_C(  1770778139),  INT32_C(   608933528), -INT32_C(  1330560994),
         INT32_C(  2013093790), -INT32_C(  1702746686), -INT32_C(  1537160088), -INT32_C(   787693708),  INT32_C(  1335719731), -INT32_C(    38262684), -INT32_C(   584973378), -INT32_C(   494021822) },
      UINT16_C( 4206) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 k2_res = 0;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_addsetc_epi32(a, b, &k2_res);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_addsetc_epi32");
    easysimd_assert_m512i_i32(r, ==, easysimd_mm512_loadu_epi32(test_vec[i].r));
    easysimd_assert_equal_mmask16(k2_res, test_vec[i].k2_res);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k2_res = 0;
    easysimd__m512i r = easysimd_mm512_addsetc_epi32(a, b, &k2_res);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, k2_res, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_addsets_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
    const easysimd__mmask16 sign;
  } test_vec[] = {
    { {  INT32_C(   812441640), -INT32_C(   365312373),  INT32_C(   777504693), -INT32_C(  1212013777),  INT32_C(   507287899),  INT32_C(   394722069), -INT32_C(     2792485), -INT32_C(   209428789),
        -INT32_C(   903614146),  INT32_C(  1840537015),  INT32_C(  1402669860), -INT32_C(  1978966737),  INT32_C(   212420599),  INT32_C(   488845122),  INT32_C(  1578956946), -INT32_C(  1789812649) },
      {  INT32_C(  1230992785), -INT32_C(   155839534),  INT32_C(  1313493279), -INT32_C(  1512548946), -INT32_C(   558726756),  INT32_C(  1140577712),  INT32_C(   631314382),  INT32_C(  1236988599),
         INT32_C(   965876327),  INT32_C(  1278232621),  INT32_C(  1201306265),  INT32_C(  1810723791), -INT32_C(  1521901580),  INT32_C(  1122518132),  INT32_C(   308775259), -INT32_C(   480566661) },
      {  INT32_C(  2043434425), -INT32_C(   521151907),  INT32_C(  2090997972),  INT32_C(  1570404573), -INT32_C(    51438857),  INT32_C(  1535299781),  INT32_C(   628521897),  INT32_C(  1027559810),
         INT32_C(    62262181), -INT32_C(  1176197660), -INT32_C(  1690991171), -INT32_C(   168242946), -INT32_C(  1309480981),  INT32_C(  1611363254),  INT32_C(  1887732205),  INT32_C(  2024587986) },
      UINT16_C( 7698) },
    { {  INT32_C(  1763503420), -INT32_C(   826913739), -INT32_C(  1793699642), -INT32_C(  1207958589),  INT32_C(   375212450), -INT32_C(   396802675),  INT32_C(  1241235662),  INT32_C(   506222306),
         INT32_C(  2038974788),  INT32_C(  1548238485),  INT32_C(  1391550094),  INT32_C(    51049057), -INT32_C(   921082053),  INT32_C(  2058449580),  INT32_C(   365145139),  INT32_C(  1194651651) },
      { -INT32_C(   826229703), -INT32_C(  2010511110), -INT32_C(   942007194),  INT32_C(  1238033422), -INT32_C(   149756853), -INT32_C(  1989032874),  INT32_C(  1939813744),  INT32_C(  1589302053),
        -INT32_C(  1993573489), -INT32_C(   367962237), -INT32_C(  2119046029),  INT32_C(   483032272), -INT32_C(  1240212128),  INT32_C(   289441185), -INT32_C(   544874566),  INT32_C(  1094533042) },
      {  INT32_C(   937273717),  INT32_C(  1457542447),  INT32_C(  1559260460),  INT32_C(    30074833),  INT32_C(   225455597),  INT32_C(  1909131747), -INT32_C(  1113917890),  INT32_C(  2095524359),
         INT32_C(    45401299),  INT32_C(  1180276248), -INT32_C(   727495935),  INT32_C(   534081329),  INT32_C(  2133673115), -INT32_C(  1947076531), -INT32_C(   179729427), -INT32_C(  2005782603) },
      UINT16_C(58432) },
    { {  INT32_C(  1053452730),  INT32_C(   858315712), -INT32_C(  1732978233), -INT32_C(  1246462123), -INT32_C(    43202724),  INT32_C(   101624908),  INT32_C(  1038455691), -INT32_C(  1921113389),
         INT32_C(  1288390796), -INT32_C(   343870685),  INT32_C(   579024077),  INT32_C(   265828275),  INT32_C(  1259095294),  INT32_C(  2068913136), -INT32_C(  2118633554), -INT32_C(   451987879) },
      { -INT32_C(  1590502786), -INT32_C(  1685278002), -INT32_C(  1698885658),  INT32_C(  1168741702), -INT32_C(   913262887),  INT32_C(  2135220689),  INT32_C(  1912732696), -INT32_C(  1336471502),
        -INT32_C(  1202550294),  INT32_C(   575921723),  INT32_C(   884740590), -INT32_C(  2139527770), -INT32_C(   330757861),  INT32_C(    57446123), -INT32_C(  1116377718),  INT32_C(  1735249277) },
      { -INT32_C(   537050056), -INT32_C(   826962290),  INT32_C(   863103405), -INT32_C(    77720421), -INT32_C(   956465611), -INT32_C(  2058121699), -INT32_C(  1343778909),  INT32_C(  1037382405),
         INT32_C(    85840502),  INT32_C(   232051038),  INT32_C(  1463764667), -INT32_C(  1873699495),  INT32_C(   928337433),  INT32_C(  2126359259),  INT32_C(  1059956024),  INT32_C(  1283261398) },
      UINT16_C( 2171) },
    { { -INT32_C(  1843347626), -INT32_C(  1934330978),  INT32_C(   734032004), -INT32_C(   240436523),  INT32_C(   786297923),  INT32_C(   221399426),  INT32_C(   868919222), -INT32_C(   879020172),
        -INT32_C(  1789019145), -INT32_C(  1289678546),  INT32_C(  1457447297),  INT32_C(  1598523675),  INT32_C(     9250173),  INT32_C(   621657966), -INT32_C(   614934681),  INT32_C(    94827278) },
      { -INT32_C(   577109074), -INT32_C(  1785677036), -INT32_C(  1192464739),  INT32_C(  1947677687), -INT32_C(   948656808), -INT32_C(   873692828),  INT32_C(  1739015256), -INT32_C(   429109960),
         INT32_C(  1707280209),  INT32_C(  1610306498), -INT32_C(  1206327359),  INT32_C(  1932341018),  INT32_C(   960143828),  INT32_C(  2063869475), -INT32_C(  1562203286),  INT32_C(  1250512889) },
      {  INT32_C(  1874510596),  INT32_C(   574959282), -INT32_C(   458432735),  INT32_C(  1707241164), -INT32_C(   162358885), -INT32_C(   652293402), -INT32_C(  1687032818), -INT32_C(  1308130132),
        -INT32_C(    81738936),  INT32_C(   320627952),  INT32_C(   251119938), -INT32_C(   764102603),  INT32_C(   969394001), -INT32_C(  1609439855),  INT32_C(  2117829329),  INT32_C(  1345340167) },
      UINT16_C(10740) },
    { {  INT32_C(   414141526),  INT32_C(  1635297952), -INT32_C(  1407545199), -INT32_C(  1809889345),  INT32_C(   198007272), -INT32_C(   376974977),  INT32_C(  1988913533),  INT32_C(   247469496),
         INT32_C(    19361633), -INT32_C(  1419534566), -INT32_C(   296256209), -INT32_C(  1400736060),  INT32_C(  1320701903), -INT32_C(  1640480991),  INT32_C(  1628751016),  INT32_C(   980407513) },
      {  INT32_C(  1564251715),  INT32_C(  1678352181), -INT32_C(   531406820), -INT32_C(  1500719657),  INT32_C(  1190478885),  INT32_C(   753151363), -INT32_C(   896665359),  INT32_C(   251985100),
        -INT32_C(   932364141), -INT32_C(    80906529), -INT32_C(  1378123562),  INT32_C(  2052352085),  INT32_C(   817908140),  INT32_C(  1734124662),  INT32_C(  1748101532),  INT32_C(  2021144037) },
      {  INT32_C(  1978393241), -INT32_C(   981317163), -INT32_C(  1938952019),  INT32_C(   984358294),  INT32_C(  1388486157),  INT32_C(   376176386),  INT32_C(  1092248174),  INT32_C(   499454596),
        -INT32_C(   913002508), -INT32_C(  1500441095), -INT32_C(  1674379771),  INT32_C(   651616025),  INT32_C(  2138610043),  INT32_C(    93643671), -INT32_C(   918114748), -INT32_C(  1293415746) },
      UINT16_C(50950) },
    { {  INT32_C(  1430381942),  INT32_C(   827420251),  INT32_C(  1138699502),  INT32_C(  1102984084), -INT32_C(   227443076), -INT32_C(  1084633821), -INT32_C(  1691841866),  INT32_C(   890544319),
        -INT32_C(   527739515), -INT32_C(  1324229437), -INT32_C(  1644892152), -INT32_C(  1612795357),  INT32_C(  1418809137), -INT32_C(   770446820),  INT32_C(   879573876),  INT32_C(  1617527259) },
      { -INT32_C(  1723796266), -INT32_C(   649440560),  INT32_C(  1719025474),  INT32_C(   570774769), -INT32_C(  1082746973), -INT32_C(   158233983), -INT32_C(  1591017787),  INT32_C(  1442943872),
         INT32_C(  1492140680), -INT32_C(   701416812),  INT32_C(  1799137145), -INT32_C(  1634909445),  INT32_C(  1516045529),  INT32_C(  1397812878),  INT32_C(  1844738797), -INT32_C(  1765542386) },
      { -INT32_C(   293414324),  INT32_C(   177979691), -INT32_C(  1437242320),  INT32_C(  1673758853), -INT32_C(  1310190049), -INT32_C(  1242867804),  INT32_C(  1012107643), -INT32_C(  1961479105),
         INT32_C(   964401165), -INT32_C(  2025646249),  INT32_C(   154244993),  INT32_C(  1047262494), -INT32_C(  1360112630),  INT32_C(   627366058), -INT32_C(  1570654623), -INT32_C(   148015127) },
      UINT16_C(53941) },
    { { -INT32_C(   856771784),  INT32_C(  1738678510), -INT32_C(  1009590329), -INT32_C(    94281695), -INT32_C(   229327004), -INT32_C(  1706711635),  INT32_C(   755448351),  INT32_C(  1757662000),
         INT32_C(  1815392894), -INT32_C(  1714104622), -INT32_C(   698571083),  INT32_C(  1792065030),  INT32_C(   710681981), -INT32_C(   372923702),  INT32_C(   202886364),  INT32_C(   376756887) },
      {  INT32_C(  1585621132),  INT32_C(   888690302),  INT32_C(    51008765), -INT32_C(  1888560366), -INT32_C(   893728256),  INT32_C(  1219723116), -INT32_C(   480982453),  INT32_C(   838453413),
        -INT32_C(   275743888), -INT32_C(   819754798), -INT32_C(   271438372),  INT32_C(   142491656),  INT32_C(  2010265611),  INT32_C(    62949047), -INT32_C(   169470896),  INT32_C(  1294458845) },
      {  INT32_C(   728849348), -INT32_C(  1667598484), -INT32_C(   958581564), -INT32_C(  1982842061), -INT32_C(  1123055260), -INT32_C(   486988519),  INT32_C(   274465898), -INT32_C(  1698851883),
         INT32_C(  1539649006),  INT32_C(  1761107876), -INT32_C(   970009455),  INT32_C(  1934556686), -INT32_C(  1574019704), -INT32_C(   309974655),  INT32_C(    33415468),  INT32_C(  1671215732) },
      UINT16_C(13502) },
    { {  INT32_C(   742176602),  INT32_C(   469458751), -INT32_C(  1794453875),  INT32_C(   429820174),  INT32_C(  2039574721),  INT32_C(  1199329782),  INT32_C(  1111253605), -INT32_C(  1685036223),
         INT32_C(  1506331674), -INT32_C(  1183464660), -INT32_C(  1605468271), -INT32_C(   893785080),  INT32_C(  1396918877),  INT32_C(    26918811),  INT32_C(  1648613153),  INT32_C(  1442698042) },
      { -INT32_C(   860961376),  INT32_C(   444933001), -INT32_C(  1413819485),  INT32_C(   494236864),  INT32_C(  1517336766), -INT32_C(  1738863753),  INT32_C(   486186722),  INT32_C(   309459058),
         INT32_C(  1188962493), -INT32_C(   429825213), -INT32_C(   158196938),  INT32_C(  1309869967),  INT32_C(   933790912),  INT32_C(  1909457807),  INT32_C(   328059553), -INT32_C(  2144993342) },
      { -INT32_C(   118784774),  INT32_C(   914391752),  INT32_C(  1086693936),  INT32_C(   924057038), -INT32_C(   738055809), -INT32_C(   539533971),  INT32_C(  1597440327), -INT32_C(  1375577165),
        -INT32_C(  1599673129), -INT32_C(  1613289873), -INT32_C(  1763665209),  INT32_C(   416084887), -INT32_C(  1964257507),  INT32_C(  1936376618),  INT32_C(  1976672706), -INT32_C(   702295300) },
      UINT16_C(38833) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__mmask16 sign = 0;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_addsets_epi32(a, b, &sign);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_addsets_epi32");
    easysimd_assert_m512i_i32(r, ==, easysimd_mm512_loadu_epi32(test_vec[i].r));
    easysimd_assert_equal_mmask16(sign, test_vec[i].sign);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 sign = 0;
    easysimd__m512i r = easysimd_mm512_addsets_epi32(a, b, &sign);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, sign, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_addsets_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
    const easysimd__mmask16 sign;
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   438.37), EASYSIMD_FLOAT32_C(  -113.12), EASYSIMD_FLOAT32_C(  -634.46), EASYSIMD_FLOAT32_C(  -523.71),
        EASYSIMD_FLOAT32_C(  -865.08), EASYSIMD_FLOAT32_C(   955.88), EASYSIMD_FLOAT32_C(  -300.68), EASYSIMD_FLOAT32_C(    70.40),
        EASYSIMD_FLOAT32_C(   968.70), EASYSIMD_FLOAT32_C(   154.10), EASYSIMD_FLOAT32_C(  -969.47), EASYSIMD_FLOAT32_C(  -474.24),
        EASYSIMD_FLOAT32_C(   666.52), EASYSIMD_FLOAT32_C(   769.15), EASYSIMD_FLOAT32_C(  -317.61), EASYSIMD_FLOAT32_C(  -255.97) },
      { EASYSIMD_FLOAT32_C(  -798.32), EASYSIMD_FLOAT32_C(  -561.24), EASYSIMD_FLOAT32_C(   839.32), EASYSIMD_FLOAT32_C(   995.62),
        EASYSIMD_FLOAT32_C(   215.29), EASYSIMD_FLOAT32_C(   321.34), EASYSIMD_FLOAT32_C(   595.20), EASYSIMD_FLOAT32_C(  -406.92),
        EASYSIMD_FLOAT32_C(   162.48), EASYSIMD_FLOAT32_C(  -714.68), EASYSIMD_FLOAT32_C(   441.89), EASYSIMD_FLOAT32_C(  -844.96),
        EASYSIMD_FLOAT32_C(  -410.05), EASYSIMD_FLOAT32_C(   858.84), EASYSIMD_FLOAT32_C(    31.37), EASYSIMD_FLOAT32_C(  -971.69) },
      { EASYSIMD_FLOAT32_C(  -359.95), EASYSIMD_FLOAT32_C(  -674.36), EASYSIMD_FLOAT32_C(   204.86), EASYSIMD_FLOAT32_C(   471.91),
        EASYSIMD_FLOAT32_C(  -649.79), EASYSIMD_FLOAT32_C(  1277.22), EASYSIMD_FLOAT32_C(   294.52), EASYSIMD_FLOAT32_C(  -336.52),
        EASYSIMD_FLOAT32_C(  1131.18), EASYSIMD_FLOAT32_C(  -560.58), EASYSIMD_FLOAT32_C(  -527.58), EASYSIMD_FLOAT32_C( -1319.20),
        EASYSIMD_FLOAT32_C(   256.47), EASYSIMD_FLOAT32_C(  1627.99), EASYSIMD_FLOAT32_C(  -286.24), EASYSIMD_FLOAT32_C( -1227.66) },
      UINT16_C(52883) },
    { { EASYSIMD_FLOAT32_C(  -254.29), EASYSIMD_FLOAT32_C(   396.92), EASYSIMD_FLOAT32_C(  -495.39), EASYSIMD_FLOAT32_C(  -119.37),
        EASYSIMD_FLOAT32_C(   352.80), EASYSIMD_FLOAT32_C(   203.93), EASYSIMD_FLOAT32_C(   951.04), EASYSIMD_FLOAT32_C(   321.50),
        EASYSIMD_FLOAT32_C(  -641.98), EASYSIMD_FLOAT32_C(   981.56), EASYSIMD_FLOAT32_C(   847.25), EASYSIMD_FLOAT32_C(  -975.45),
        EASYSIMD_FLOAT32_C(   750.71), EASYSIMD_FLOAT32_C(  -470.36), EASYSIMD_FLOAT32_C(  -231.42), EASYSIMD_FLOAT32_C(   952.39) },
      { EASYSIMD_FLOAT32_C(   -31.60), EASYSIMD_FLOAT32_C(  -392.11), EASYSIMD_FLOAT32_C(   948.01), EASYSIMD_FLOAT32_C(  -816.31),
        EASYSIMD_FLOAT32_C(   929.23), EASYSIMD_FLOAT32_C(   543.21), EASYSIMD_FLOAT32_C(  -223.23), EASYSIMD_FLOAT32_C(    91.71),
        EASYSIMD_FLOAT32_C(   828.53), EASYSIMD_FLOAT32_C(  -781.34), EASYSIMD_FLOAT32_C(   246.76), EASYSIMD_FLOAT32_C(  -581.53),
        EASYSIMD_FLOAT32_C(  -922.51), EASYSIMD_FLOAT32_C(  -721.87), EASYSIMD_FLOAT32_C(  -553.21), EASYSIMD_FLOAT32_C(  -176.80) },
      { EASYSIMD_FLOAT32_C(  -285.89), EASYSIMD_FLOAT32_C(     4.81), EASYSIMD_FLOAT32_C(   452.62), EASYSIMD_FLOAT32_C(  -935.68),
        EASYSIMD_FLOAT32_C(  1282.03), EASYSIMD_FLOAT32_C(   747.14), EASYSIMD_FLOAT32_C(   727.81), EASYSIMD_FLOAT32_C(   413.21),
        EASYSIMD_FLOAT32_C(   186.55), EASYSIMD_FLOAT32_C(   200.22), EASYSIMD_FLOAT32_C(  1094.01), EASYSIMD_FLOAT32_C( -1556.98),
        EASYSIMD_FLOAT32_C(  -171.80), EASYSIMD_FLOAT32_C( -1192.23), EASYSIMD_FLOAT32_C(  -784.63), EASYSIMD_FLOAT32_C(   775.59) },
      UINT16_C(30729) },
    { { EASYSIMD_FLOAT32_C(   675.04), EASYSIMD_FLOAT32_C(   -48.60), EASYSIMD_FLOAT32_C(   703.84), EASYSIMD_FLOAT32_C(    27.84),
        EASYSIMD_FLOAT32_C(  -844.68), EASYSIMD_FLOAT32_C(   654.87), EASYSIMD_FLOAT32_C(  -650.66), EASYSIMD_FLOAT32_C(  -486.65),
        EASYSIMD_FLOAT32_C(   636.44), EASYSIMD_FLOAT32_C(  -803.41), EASYSIMD_FLOAT32_C(  -462.11), EASYSIMD_FLOAT32_C(   387.15),
        EASYSIMD_FLOAT32_C(  -273.77), EASYSIMD_FLOAT32_C(   306.47), EASYSIMD_FLOAT32_C(   339.55), EASYSIMD_FLOAT32_C(   694.63) },
      { EASYSIMD_FLOAT32_C(   914.36), EASYSIMD_FLOAT32_C(   287.56), EASYSIMD_FLOAT32_C(   878.32), EASYSIMD_FLOAT32_C(   843.60),
        EASYSIMD_FLOAT32_C(  -169.23), EASYSIMD_FLOAT32_C(  -344.90), EASYSIMD_FLOAT32_C(   -64.69), EASYSIMD_FLOAT32_C(  -340.70),
        EASYSIMD_FLOAT32_C(  -126.25), EASYSIMD_FLOAT32_C(  -817.94), EASYSIMD_FLOAT32_C(    77.77), EASYSIMD_FLOAT32_C(   -48.75),
        EASYSIMD_FLOAT32_C(  -539.81), EASYSIMD_FLOAT32_C(   524.56), EASYSIMD_FLOAT32_C(   774.45), EASYSIMD_FLOAT32_C(  -864.77) },
      { EASYSIMD_FLOAT32_C(  1589.40), EASYSIMD_FLOAT32_C(   238.96), EASYSIMD_FLOAT32_C(  1582.16), EASYSIMD_FLOAT32_C(   871.44),
        EASYSIMD_FLOAT32_C( -1013.91), EASYSIMD_FLOAT32_C(   309.97), EASYSIMD_FLOAT32_C(  -715.35), EASYSIMD_FLOAT32_C(  -827.35),
        EASYSIMD_FLOAT32_C(   510.19), EASYSIMD_FLOAT32_C( -1621.35), EASYSIMD_FLOAT32_C(  -384.34), EASYSIMD_FLOAT32_C(   338.40),
        EASYSIMD_FLOAT32_C(  -813.58), EASYSIMD_FLOAT32_C(   831.03), EASYSIMD_FLOAT32_C(  1114.00), EASYSIMD_FLOAT32_C(  -170.14) },
      UINT16_C(38608) },
    { { EASYSIMD_FLOAT32_C(  -524.05), EASYSIMD_FLOAT32_C(   478.29), EASYSIMD_FLOAT32_C(   163.07), EASYSIMD_FLOAT32_C(  -368.72),
        EASYSIMD_FLOAT32_C(   133.17), EASYSIMD_FLOAT32_C(   512.41), EASYSIMD_FLOAT32_C(   144.62), EASYSIMD_FLOAT32_C(  -230.40),
        EASYSIMD_FLOAT32_C(   709.01), EASYSIMD_FLOAT32_C(   682.51), EASYSIMD_FLOAT32_C(  -843.25), EASYSIMD_FLOAT32_C(  -564.76),
        EASYSIMD_FLOAT32_C(   -11.02), EASYSIMD_FLOAT32_C(   496.30), EASYSIMD_FLOAT32_C(  -870.13), EASYSIMD_FLOAT32_C(   -96.65) },
      { EASYSIMD_FLOAT32_C(  -216.14), EASYSIMD_FLOAT32_C(  -991.80), EASYSIMD_FLOAT32_C(  -253.06), EASYSIMD_FLOAT32_C(   614.63),
        EASYSIMD_FLOAT32_C(  -336.71), EASYSIMD_FLOAT32_C(   682.25), EASYSIMD_FLOAT32_C(  -726.08), EASYSIMD_FLOAT32_C(   537.05),
        EASYSIMD_FLOAT32_C(   864.31), EASYSIMD_FLOAT32_C(   351.69), EASYSIMD_FLOAT32_C(  -511.71), EASYSIMD_FLOAT32_C(  -675.50),
        EASYSIMD_FLOAT32_C(  -123.75), EASYSIMD_FLOAT32_C(  -737.25), EASYSIMD_FLOAT32_C(  -540.27), EASYSIMD_FLOAT32_C(   352.21) },
      { EASYSIMD_FLOAT32_C(  -740.19), EASYSIMD_FLOAT32_C(  -513.51), EASYSIMD_FLOAT32_C(   -89.99), EASYSIMD_FLOAT32_C(   245.91),
        EASYSIMD_FLOAT32_C(  -203.54), EASYSIMD_FLOAT32_C(  1194.66), EASYSIMD_FLOAT32_C(  -581.46), EASYSIMD_FLOAT32_C(   306.65),
        EASYSIMD_FLOAT32_C(  1573.32), EASYSIMD_FLOAT32_C(  1034.20), EASYSIMD_FLOAT32_C( -1354.96), EASYSIMD_FLOAT32_C( -1240.26),
        EASYSIMD_FLOAT32_C(  -134.77), EASYSIMD_FLOAT32_C(  -240.95), EASYSIMD_FLOAT32_C( -1410.40), EASYSIMD_FLOAT32_C(   255.56) },
      UINT16_C(31831) },
    { { EASYSIMD_FLOAT32_C(   741.04), EASYSIMD_FLOAT32_C(   622.81), EASYSIMD_FLOAT32_C(   983.48), EASYSIMD_FLOAT32_C(  -125.80),
        EASYSIMD_FLOAT32_C(   135.22), EASYSIMD_FLOAT32_C(   128.11), EASYSIMD_FLOAT32_C(   643.81), EASYSIMD_FLOAT32_C(  -155.77),
        EASYSIMD_FLOAT32_C(  -189.38), EASYSIMD_FLOAT32_C(   800.56), EASYSIMD_FLOAT32_C(   279.46), EASYSIMD_FLOAT32_C(   799.60),
        EASYSIMD_FLOAT32_C(   296.86), EASYSIMD_FLOAT32_C(   409.34), EASYSIMD_FLOAT32_C(  -297.05), EASYSIMD_FLOAT32_C(  -919.28) },
      { EASYSIMD_FLOAT32_C(   417.53), EASYSIMD_FLOAT32_C(   449.89), EASYSIMD_FLOAT32_C(   695.34), EASYSIMD_FLOAT32_C(  -919.18),
        EASYSIMD_FLOAT32_C(   132.14), EASYSIMD_FLOAT32_C(   969.26), EASYSIMD_FLOAT32_C(   617.87), EASYSIMD_FLOAT32_C(    -3.55),
        EASYSIMD_FLOAT32_C(   320.96), EASYSIMD_FLOAT32_C(  -893.84), EASYSIMD_FLOAT32_C(   320.96), EASYSIMD_FLOAT32_C(  -802.79),
        EASYSIMD_FLOAT32_C(  -631.09), EASYSIMD_FLOAT32_C(   780.69), EASYSIMD_FLOAT32_C(   549.41), EASYSIMD_FLOAT32_C(  -890.05) },
      { EASYSIMD_FLOAT32_C(  1158.57), EASYSIMD_FLOAT32_C(  1072.70), EASYSIMD_FLOAT32_C(  1678.82), EASYSIMD_FLOAT32_C( -1044.98),
        EASYSIMD_FLOAT32_C(   267.36), EASYSIMD_FLOAT32_C(  1097.37), EASYSIMD_FLOAT32_C(  1261.68), EASYSIMD_FLOAT32_C(  -159.32),
        EASYSIMD_FLOAT32_C(   131.58), EASYSIMD_FLOAT32_C(   -93.28), EASYSIMD_FLOAT32_C(   600.42), EASYSIMD_FLOAT32_C(    -3.19),
        EASYSIMD_FLOAT32_C(  -334.23), EASYSIMD_FLOAT32_C(  1190.03), EASYSIMD_FLOAT32_C(   252.36), EASYSIMD_FLOAT32_C( -1809.33) },
      UINT16_C(39560) },
    { { EASYSIMD_FLOAT32_C(   403.49), EASYSIMD_FLOAT32_C(   532.90), EASYSIMD_FLOAT32_C(   -15.85), EASYSIMD_FLOAT32_C(  -461.29),
        EASYSIMD_FLOAT32_C(  -339.00), EASYSIMD_FLOAT32_C(  -372.04), EASYSIMD_FLOAT32_C(   382.94), EASYSIMD_FLOAT32_C(   471.62),
        EASYSIMD_FLOAT32_C(  -571.48), EASYSIMD_FLOAT32_C(  -337.59), EASYSIMD_FLOAT32_C(   271.23), EASYSIMD_FLOAT32_C(   725.37),
        EASYSIMD_FLOAT32_C(  -928.26), EASYSIMD_FLOAT32_C(   974.18), EASYSIMD_FLOAT32_C(   806.09), EASYSIMD_FLOAT32_C(   489.27) },
      { EASYSIMD_FLOAT32_C(   424.07), EASYSIMD_FLOAT32_C(   501.43), EASYSIMD_FLOAT32_C(   570.10), EASYSIMD_FLOAT32_C(  -443.78),
        EASYSIMD_FLOAT32_C(   470.70), EASYSIMD_FLOAT32_C(   187.97), EASYSIMD_FLOAT32_C(   552.67), EASYSIMD_FLOAT32_C(  -208.35),
        EASYSIMD_FLOAT32_C(   294.14), EASYSIMD_FLOAT32_C(  -126.37), EASYSIMD_FLOAT32_C(   -11.14), EASYSIMD_FLOAT32_C(   663.05),
        EASYSIMD_FLOAT32_C(  -345.69), EASYSIMD_FLOAT32_C(  -461.73), EASYSIMD_FLOAT32_C(   773.00), EASYSIMD_FLOAT32_C(  -942.19) },
      { EASYSIMD_FLOAT32_C(   827.56), EASYSIMD_FLOAT32_C(  1034.33), EASYSIMD_FLOAT32_C(   554.25), EASYSIMD_FLOAT32_C(  -905.07),
        EASYSIMD_FLOAT32_C(   131.70), EASYSIMD_FLOAT32_C(  -184.07), EASYSIMD_FLOAT32_C(   935.61), EASYSIMD_FLOAT32_C(   263.27),
        EASYSIMD_FLOAT32_C(  -277.34), EASYSIMD_FLOAT32_C(  -463.96), EASYSIMD_FLOAT32_C(   260.09), EASYSIMD_FLOAT32_C(  1388.42),
        EASYSIMD_FLOAT32_C( -1273.95), EASYSIMD_FLOAT32_C(   512.45), EASYSIMD_FLOAT32_C(  1579.09), EASYSIMD_FLOAT32_C(  -452.92) },
      UINT16_C(37672) },
    { { EASYSIMD_FLOAT32_C(  -928.83), EASYSIMD_FLOAT32_C(  -242.85), EASYSIMD_FLOAT32_C(  -403.48), EASYSIMD_FLOAT32_C(  -267.83),
        EASYSIMD_FLOAT32_C(   385.10), EASYSIMD_FLOAT32_C(   979.46), EASYSIMD_FLOAT32_C(  -796.20), EASYSIMD_FLOAT32_C(   813.62),
        EASYSIMD_FLOAT32_C(  -358.13), EASYSIMD_FLOAT32_C(   475.02), EASYSIMD_FLOAT32_C(   538.99), EASYSIMD_FLOAT32_C(  -286.39),
        EASYSIMD_FLOAT32_C(   449.20), EASYSIMD_FLOAT32_C(   345.08), EASYSIMD_FLOAT32_C(  -797.12), EASYSIMD_FLOAT32_C(  -126.73) },
      { EASYSIMD_FLOAT32_C(  -153.49), EASYSIMD_FLOAT32_C(   772.98), EASYSIMD_FLOAT32_C(   429.49), EASYSIMD_FLOAT32_C(  -682.79),
        EASYSIMD_FLOAT32_C(   -39.05), EASYSIMD_FLOAT32_C(   -17.84), EASYSIMD_FLOAT32_C(   108.86), EASYSIMD_FLOAT32_C(  -744.91),
        EASYSIMD_FLOAT32_C(   855.79), EASYSIMD_FLOAT32_C(  -902.28), EASYSIMD_FLOAT32_C(   918.13), EASYSIMD_FLOAT32_C(  -489.90),
        EASYSIMD_FLOAT32_C(  -364.00), EASYSIMD_FLOAT32_C(   691.13), EASYSIMD_FLOAT32_C(  -432.09), EASYSIMD_FLOAT32_C(  -292.83) },
      { EASYSIMD_FLOAT32_C( -1082.32), EASYSIMD_FLOAT32_C(   530.13), EASYSIMD_FLOAT32_C(    26.01), EASYSIMD_FLOAT32_C(  -950.62),
        EASYSIMD_FLOAT32_C(   346.05), EASYSIMD_FLOAT32_C(   961.62), EASYSIMD_FLOAT32_C(  -687.34), EASYSIMD_FLOAT32_C(    68.71),
        EASYSIMD_FLOAT32_C(   497.66), EASYSIMD_FLOAT32_C(  -427.26), EASYSIMD_FLOAT32_C(  1457.12), EASYSIMD_FLOAT32_C(  -776.29),
        EASYSIMD_FLOAT32_C(    85.20), EASYSIMD_FLOAT32_C(  1036.21), EASYSIMD_FLOAT32_C( -1229.21), EASYSIMD_FLOAT32_C(  -419.56) },
      UINT16_C(51785) },
    { { EASYSIMD_FLOAT32_C(  -551.72), EASYSIMD_FLOAT32_C(   164.43), EASYSIMD_FLOAT32_C(   439.34), EASYSIMD_FLOAT32_C(   833.38),
        EASYSIMD_FLOAT32_C(   143.89), EASYSIMD_FLOAT32_C(   643.13), EASYSIMD_FLOAT32_C(   647.00), EASYSIMD_FLOAT32_C(   785.76),
        EASYSIMD_FLOAT32_C(   118.16), EASYSIMD_FLOAT32_C(   185.99), EASYSIMD_FLOAT32_C(  -500.63), EASYSIMD_FLOAT32_C(  -432.64),
        EASYSIMD_FLOAT32_C(  -468.93), EASYSIMD_FLOAT32_C(  -297.75), EASYSIMD_FLOAT32_C(   440.63), EASYSIMD_FLOAT32_C(   377.59) },
      { EASYSIMD_FLOAT32_C(  -524.77), EASYSIMD_FLOAT32_C(  -129.88), EASYSIMD_FLOAT32_C(   694.80), EASYSIMD_FLOAT32_C(   436.19),
        EASYSIMD_FLOAT32_C(   852.29), EASYSIMD_FLOAT32_C(  -196.34), EASYSIMD_FLOAT32_C(   691.27), EASYSIMD_FLOAT32_C(   708.07),
        EASYSIMD_FLOAT32_C(   -98.62), EASYSIMD_FLOAT32_C(   609.41), EASYSIMD_FLOAT32_C(  -781.82), EASYSIMD_FLOAT32_C(   537.38),
        EASYSIMD_FLOAT32_C(   300.54), EASYSIMD_FLOAT32_C(  -213.91), EASYSIMD_FLOAT32_C(  -755.46), EASYSIMD_FLOAT32_C(   748.81) },
      { EASYSIMD_FLOAT32_C( -1076.49), EASYSIMD_FLOAT32_C(    34.55), EASYSIMD_FLOAT32_C(  1134.14), EASYSIMD_FLOAT32_C(  1269.57),
        EASYSIMD_FLOAT32_C(   996.18), EASYSIMD_FLOAT32_C(   446.79), EASYSIMD_FLOAT32_C(  1338.27), EASYSIMD_FLOAT32_C(  1493.83),
        EASYSIMD_FLOAT32_C(    19.54), EASYSIMD_FLOAT32_C(   795.40), EASYSIMD_FLOAT32_C( -1282.45), EASYSIMD_FLOAT32_C(   104.74),
        EASYSIMD_FLOAT32_C(  -168.39), EASYSIMD_FLOAT32_C(  -511.66), EASYSIMD_FLOAT32_C(  -314.83), EASYSIMD_FLOAT32_C(  1126.40) },
      UINT16_C(29697) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__mmask16 sign = 0;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_addsets_ps(a, b, &sign);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_addsets_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
    easysimd_assert_equal_mmask16(sign, test_vec[i].sign);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask16 sign = 0;
    easysimd__m512 r = easysimd_mm512_addsets_ps(a, b, &sign);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, sign, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

static int
test_easysimd_mm512_addsets_round_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
  struct {
    easysimd_float32 a[16];
    easysimd_float32 b[16];
    easysimd_float32 nearest_inf[16];
    easysimd__mmask16 sign1;
    easysimd_float32 neg_inf[16];
    easysimd__mmask16 sign2;
    easysimd_float32 pos_inf[16];
    easysimd__mmask16 sign3;
    easysimd_float32 zero[16];
    easysimd__mmask16 sign4;
    easysimd_float32 direction[16];
    easysimd__mmask16 sign5;
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   810.75), EASYSIMD_FLOAT32_C(  -600.41), EASYSIMD_FLOAT32_C(   357.45), EASYSIMD_FLOAT32_C(   851.67),
        EASYSIMD_FLOAT32_C(   983.67), EASYSIMD_FLOAT32_C(   805.11), EASYSIMD_FLOAT32_C(   439.86), EASYSIMD_FLOAT32_C(  -199.30),
        EASYSIMD_FLOAT32_C(   974.34), EASYSIMD_FLOAT32_C(   734.82), EASYSIMD_FLOAT32_C(  -508.32), EASYSIMD_FLOAT32_C(  -466.23),
        EASYSIMD_FLOAT32_C(   435.28), EASYSIMD_FLOAT32_C(   972.07), EASYSIMD_FLOAT32_C(  -470.41), EASYSIMD_FLOAT32_C(   832.54) },
      { EASYSIMD_FLOAT32_C(   692.70), EASYSIMD_FLOAT32_C(  -480.49), EASYSIMD_FLOAT32_C(   468.30), EASYSIMD_FLOAT32_C(   205.48),
        EASYSIMD_FLOAT32_C(   685.04), EASYSIMD_FLOAT32_C(  -963.18), EASYSIMD_FLOAT32_C(   810.65), EASYSIMD_FLOAT32_C(   906.24),
        EASYSIMD_FLOAT32_C(  -668.98), EASYSIMD_FLOAT32_C(   439.50), EASYSIMD_FLOAT32_C(  -119.72), EASYSIMD_FLOAT32_C(  -423.77),
        EASYSIMD_FLOAT32_C(  -907.00), EASYSIMD_FLOAT32_C(   811.72), EASYSIMD_FLOAT32_C(   220.29), EASYSIMD_FLOAT32_C(   903.76) },
      { EASYSIMD_FLOAT32_C(  1503.00), EASYSIMD_FLOAT32_C( -1081.00), EASYSIMD_FLOAT32_C(   826.00), EASYSIMD_FLOAT32_C(  1057.00),
        EASYSIMD_FLOAT32_C(  1669.00), EASYSIMD_FLOAT32_C(  -158.00), EASYSIMD_FLOAT32_C(  1251.00), EASYSIMD_FLOAT32_C(   707.00),
        EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(  1174.00), EASYSIMD_FLOAT32_C(  -628.00), EASYSIMD_FLOAT32_C(  -890.00),
        EASYSIMD_FLOAT32_C(  -472.00), EASYSIMD_FLOAT32_C(  1784.00), EASYSIMD_FLOAT32_C(  -250.00), EASYSIMD_FLOAT32_C(  1736.00) },
      UINT16_C(23586),
      { EASYSIMD_FLOAT32_C(  1503.00), EASYSIMD_FLOAT32_C( -1081.00), EASYSIMD_FLOAT32_C(   825.00), EASYSIMD_FLOAT32_C(  1057.00),
        EASYSIMD_FLOAT32_C(  1668.00), EASYSIMD_FLOAT32_C(  -159.00), EASYSIMD_FLOAT32_C(  1250.00), EASYSIMD_FLOAT32_C(   706.00),
        EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(  1174.00), EASYSIMD_FLOAT32_C(  -629.00), EASYSIMD_FLOAT32_C(  -890.00),
        EASYSIMD_FLOAT32_C(  -472.00), EASYSIMD_FLOAT32_C(  1783.00), EASYSIMD_FLOAT32_C(  -251.00), EASYSIMD_FLOAT32_C(  1736.00) },
      UINT16_C(23586),
      { EASYSIMD_FLOAT32_C(  1504.00), EASYSIMD_FLOAT32_C( -1080.00), EASYSIMD_FLOAT32_C(   826.00), EASYSIMD_FLOAT32_C(  1058.00),
        EASYSIMD_FLOAT32_C(  1669.00), EASYSIMD_FLOAT32_C(  -158.00), EASYSIMD_FLOAT32_C(  1251.00), EASYSIMD_FLOAT32_C(   707.00),
        EASYSIMD_FLOAT32_C(   306.00), EASYSIMD_FLOAT32_C(  1175.00), EASYSIMD_FLOAT32_C(  -628.00), EASYSIMD_FLOAT32_C(  -890.00),
        EASYSIMD_FLOAT32_C(  -471.00), EASYSIMD_FLOAT32_C(  1784.00), EASYSIMD_FLOAT32_C(  -250.00), EASYSIMD_FLOAT32_C(  1737.00) },
      UINT16_C(23586),
      { EASYSIMD_FLOAT32_C(  1503.00), EASYSIMD_FLOAT32_C( -1080.00), EASYSIMD_FLOAT32_C(   825.00), EASYSIMD_FLOAT32_C(  1057.00),
        EASYSIMD_FLOAT32_C(  1668.00), EASYSIMD_FLOAT32_C(  -158.00), EASYSIMD_FLOAT32_C(  1250.00), EASYSIMD_FLOAT32_C(   706.00),
        EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(  1174.00), EASYSIMD_FLOAT32_C(  -628.00), EASYSIMD_FLOAT32_C(  -890.00),
        EASYSIMD_FLOAT32_C(  -471.00), EASYSIMD_FLOAT32_C(  1783.00), EASYSIMD_FLOAT32_C(  -250.00), EASYSIMD_FLOAT32_C(  1736.00) },
      UINT16_C(23586),
      { EASYSIMD_FLOAT32_C(  1503.00), EASYSIMD_FLOAT32_C( -1081.00), EASYSIMD_FLOAT32_C(   826.00), EASYSIMD_FLOAT32_C(  1057.00),
        EASYSIMD_FLOAT32_C(  1669.00), EASYSIMD_FLOAT32_C(  -158.00), EASYSIMD_FLOAT32_C(  1251.00), EASYSIMD_FLOAT32_C(   707.00),
        EASYSIMD_FLOAT32_C(   305.00), EASYSIMD_FLOAT32_C(  1174.00), EASYSIMD_FLOAT32_C(  -628.00), EASYSIMD_FLOAT32_C(  -890.00),
        EASYSIMD_FLOAT32_C(  -472.00), EASYSIMD_FLOAT32_C(  1784.00), EASYSIMD_FLOAT32_C(  -250.00), EASYSIMD_FLOAT32_C(  1736.00) },
      UINT16_C(23586) },
    { { EASYSIMD_FLOAT32_C(  -788.69), EASYSIMD_FLOAT32_C(  -422.26), EASYSIMD_FLOAT32_C(   755.43), EASYSIMD_FLOAT32_C(  -805.02),
        EASYSIMD_FLOAT32_C(  -617.14), EASYSIMD_FLOAT32_C(   195.29), EASYSIMD_FLOAT32_C(    -4.32), EASYSIMD_FLOAT32_C(  -642.81),
        EASYSIMD_FLOAT32_C(   -69.89), EASYSIMD_FLOAT32_C(   487.36), EASYSIMD_FLOAT32_C(  -109.04), EASYSIMD_FLOAT32_C(  -634.60),
        EASYSIMD_FLOAT32_C(   459.43), EASYSIMD_FLOAT32_C(   420.55), EASYSIMD_FLOAT32_C(  -802.06), EASYSIMD_FLOAT32_C(   152.13) },
      { EASYSIMD_FLOAT32_C(   940.07), EASYSIMD_FLOAT32_C(   666.24), EASYSIMD_FLOAT32_C(  -642.39), EASYSIMD_FLOAT32_C(   625.10),
        EASYSIMD_FLOAT32_C(   703.06), EASYSIMD_FLOAT32_C(  -831.74), EASYSIMD_FLOAT32_C(   531.34), EASYSIMD_FLOAT32_C(  -965.92),
        EASYSIMD_FLOAT32_C(   607.76), EASYSIMD_FLOAT32_C(  -588.37), EASYSIMD_FLOAT32_C(  -389.69), EASYSIMD_FLOAT32_C(   700.76),
        EASYSIMD_FLOAT32_C(  -776.66), EASYSIMD_FLOAT32_C(   830.60), EASYSIMD_FLOAT32_C(   604.52), EASYSIMD_FLOAT32_C(  -565.35) },
      { EASYSIMD_FLOAT32_C(   151.00), EASYSIMD_FLOAT32_C(   244.00), EASYSIMD_FLOAT32_C(   113.00), EASYSIMD_FLOAT32_C(  -180.00),
        EASYSIMD_FLOAT32_C(    86.00), EASYSIMD_FLOAT32_C(  -636.00), EASYSIMD_FLOAT32_C(   527.00), EASYSIMD_FLOAT32_C( -1609.00),
        EASYSIMD_FLOAT32_C(   538.00), EASYSIMD_FLOAT32_C(  -101.00), EASYSIMD_FLOAT32_C(  -499.00), EASYSIMD_FLOAT32_C(    66.00),
        EASYSIMD_FLOAT32_C(  -317.00), EASYSIMD_FLOAT32_C(  1251.00), EASYSIMD_FLOAT32_C(  -198.00), EASYSIMD_FLOAT32_C(  -413.00) },
      UINT16_C(54952),
      { EASYSIMD_FLOAT32_C(   151.00), EASYSIMD_FLOAT32_C(   243.00), EASYSIMD_FLOAT32_C(   113.00), EASYSIMD_FLOAT32_C(  -180.00),
        EASYSIMD_FLOAT32_C(    85.00), EASYSIMD_FLOAT32_C(  -637.00), EASYSIMD_FLOAT32_C(   527.00), EASYSIMD_FLOAT32_C( -1609.00),
        EASYSIMD_FLOAT32_C(   537.00), EASYSIMD_FLOAT32_C(  -102.00), EASYSIMD_FLOAT32_C(  -499.00), EASYSIMD_FLOAT32_C(    66.00),
        EASYSIMD_FLOAT32_C(  -318.00), EASYSIMD_FLOAT32_C(  1251.00), EASYSIMD_FLOAT32_C(  -198.00), EASYSIMD_FLOAT32_C(  -414.00) },
      UINT16_C(54952),
      { EASYSIMD_FLOAT32_C(   152.00), EASYSIMD_FLOAT32_C(   244.00), EASYSIMD_FLOAT32_C(   114.00), EASYSIMD_FLOAT32_C(  -179.00),
        EASYSIMD_FLOAT32_C(    86.00), EASYSIMD_FLOAT32_C(  -636.00), EASYSIMD_FLOAT32_C(   528.00), EASYSIMD_FLOAT32_C( -1608.00),
        EASYSIMD_FLOAT32_C(   538.00), EASYSIMD_FLOAT32_C(  -101.00), EASYSIMD_FLOAT32_C(  -498.00), EASYSIMD_FLOAT32_C(    67.00),
        EASYSIMD_FLOAT32_C(  -317.00), EASYSIMD_FLOAT32_C(  1252.00), EASYSIMD_FLOAT32_C(  -197.00), EASYSIMD_FLOAT32_C(  -413.00) },
      UINT16_C(54952),
      { EASYSIMD_FLOAT32_C(   151.00), EASYSIMD_FLOAT32_C(   243.00), EASYSIMD_FLOAT32_C(   113.00), EASYSIMD_FLOAT32_C(  -179.00),
        EASYSIMD_FLOAT32_C(    85.00), EASYSIMD_FLOAT32_C(  -636.00), EASYSIMD_FLOAT32_C(   527.00), EASYSIMD_FLOAT32_C( -1608.00),
        EASYSIMD_FLOAT32_C(   537.00), EASYSIMD_FLOAT32_C(  -101.00), EASYSIMD_FLOAT32_C(  -498.00), EASYSIMD_FLOAT32_C(    66.00),
        EASYSIMD_FLOAT32_C(  -317.00), EASYSIMD_FLOAT32_C(  1251.00), EASYSIMD_FLOAT32_C(  -197.00), EASYSIMD_FLOAT32_C(  -413.00) },
      UINT16_C(54952),
      { EASYSIMD_FLOAT32_C(   151.00), EASYSIMD_FLOAT32_C(   244.00), EASYSIMD_FLOAT32_C(   113.00), EASYSIMD_FLOAT32_C(  -180.00),
        EASYSIMD_FLOAT32_C(    86.00), EASYSIMD_FLOAT32_C(  -636.00), EASYSIMD_FLOAT32_C(   527.00), EASYSIMD_FLOAT32_C( -1609.00),
        EASYSIMD_FLOAT32_C(   538.00), EASYSIMD_FLOAT32_C(  -101.00), EASYSIMD_FLOAT32_C(  -499.00), EASYSIMD_FLOAT32_C(    66.00),
        EASYSIMD_FLOAT32_C(  -317.00), EASYSIMD_FLOAT32_C(  1251.00), EASYSIMD_FLOAT32_C(  -198.00), EASYSIMD_FLOAT32_C(  -413.00) },
      UINT16_C(54952) },
    { { EASYSIMD_FLOAT32_C(  -591.65), EASYSIMD_FLOAT32_C(   359.95), EASYSIMD_FLOAT32_C(  -370.38), EASYSIMD_FLOAT32_C(  -208.80),
        EASYSIMD_FLOAT32_C(  -444.76), EASYSIMD_FLOAT32_C(   625.30), EASYSIMD_FLOAT32_C(   148.39), EASYSIMD_FLOAT32_C(   485.35),
        EASYSIMD_FLOAT32_C(   112.67), EASYSIMD_FLOAT32_C(  -960.64), EASYSIMD_FLOAT32_C(   850.75), EASYSIMD_FLOAT32_C(  -427.90),
        EASYSIMD_FLOAT32_C(   459.91), EASYSIMD_FLOAT32_C(  -951.32), EASYSIMD_FLOAT32_C(   724.23), EASYSIMD_FLOAT32_C(   399.97) },
      { EASYSIMD_FLOAT32_C(   714.93), EASYSIMD_FLOAT32_C(  -918.17), EASYSIMD_FLOAT32_C(    25.07), EASYSIMD_FLOAT32_C(   417.99),
        EASYSIMD_FLOAT32_C(  -749.91), EASYSIMD_FLOAT32_C(  -443.58), EASYSIMD_FLOAT32_C(   452.07), EASYSIMD_FLOAT32_C(   857.85),
        EASYSIMD_FLOAT32_C(   -31.96), EASYSIMD_FLOAT32_C(  -937.62), EASYSIMD_FLOAT32_C(   558.61), EASYSIMD_FLOAT32_C(   191.39),
        EASYSIMD_FLOAT32_C(   892.98), EASYSIMD_FLOAT32_C(   163.13), EASYSIMD_FLOAT32_C(   626.03), EASYSIMD_FLOAT32_C(  -698.67) },
      { EASYSIMD_FLOAT32_C(   123.00), EASYSIMD_FLOAT32_C(  -558.00), EASYSIMD_FLOAT32_C(  -345.00), EASYSIMD_FLOAT32_C(   209.00),
        EASYSIMD_FLOAT32_C( -1195.00), EASYSIMD_FLOAT32_C(   182.00), EASYSIMD_FLOAT32_C(   600.00), EASYSIMD_FLOAT32_C(  1343.00),
        EASYSIMD_FLOAT32_C(    81.00), EASYSIMD_FLOAT32_C( -1898.00), EASYSIMD_FLOAT32_C(  1409.00), EASYSIMD_FLOAT32_C(  -237.00),
        EASYSIMD_FLOAT32_C(  1353.00), EASYSIMD_FLOAT32_C(  -788.00), EASYSIMD_FLOAT32_C(  1350.00), EASYSIMD_FLOAT32_C(  -299.00) },
      UINT16_C(43542),
      { EASYSIMD_FLOAT32_C(   123.00), EASYSIMD_FLOAT32_C(  -559.00), EASYSIMD_FLOAT32_C(  -346.00), EASYSIMD_FLOAT32_C(   209.00),
        EASYSIMD_FLOAT32_C( -1195.00), EASYSIMD_FLOAT32_C(   181.00), EASYSIMD_FLOAT32_C(   600.00), EASYSIMD_FLOAT32_C(  1343.00),
        EASYSIMD_FLOAT32_C(    80.00), EASYSIMD_FLOAT32_C( -1899.00), EASYSIMD_FLOAT32_C(  1409.00), EASYSIMD_FLOAT32_C(  -237.00),
        EASYSIMD_FLOAT32_C(  1352.00), EASYSIMD_FLOAT32_C(  -789.00), EASYSIMD_FLOAT32_C(  1350.00), EASYSIMD_FLOAT32_C(  -299.00) },
      UINT16_C(43542),
      { EASYSIMD_FLOAT32_C(   124.00), EASYSIMD_FLOAT32_C(  -558.00), EASYSIMD_FLOAT32_C(  -345.00), EASYSIMD_FLOAT32_C(   210.00),
        EASYSIMD_FLOAT32_C( -1194.00), EASYSIMD_FLOAT32_C(   182.00), EASYSIMD_FLOAT32_C(   601.00), EASYSIMD_FLOAT32_C(  1344.00),
        EASYSIMD_FLOAT32_C(    81.00), EASYSIMD_FLOAT32_C( -1898.00), EASYSIMD_FLOAT32_C(  1410.00), EASYSIMD_FLOAT32_C(  -236.00),
        EASYSIMD_FLOAT32_C(  1353.00), EASYSIMD_FLOAT32_C(  -788.00), EASYSIMD_FLOAT32_C(  1351.00), EASYSIMD_FLOAT32_C(  -298.00) },
      UINT16_C(43542),
      { EASYSIMD_FLOAT32_C(   123.00), EASYSIMD_FLOAT32_C(  -558.00), EASYSIMD_FLOAT32_C(  -345.00), EASYSIMD_FLOAT32_C(   209.00),
        EASYSIMD_FLOAT32_C( -1194.00), EASYSIMD_FLOAT32_C(   181.00), EASYSIMD_FLOAT32_C(   600.00), EASYSIMD_FLOAT32_C(  1343.00),
        EASYSIMD_FLOAT32_C(    80.00), EASYSIMD_FLOAT32_C( -1898.00), EASYSIMD_FLOAT32_C(  1409.00), EASYSIMD_FLOAT32_C(  -236.00),
        EASYSIMD_FLOAT32_C(  1352.00), EASYSIMD_FLOAT32_C(  -788.00), EASYSIMD_FLOAT32_C(  1350.00), EASYSIMD_FLOAT32_C(  -298.00) },
      UINT16_C(43542),
      { EASYSIMD_FLOAT32_C(   123.00), EASYSIMD_FLOAT32_C(  -558.00), EASYSIMD_FLOAT32_C(  -345.00), EASYSIMD_FLOAT32_C(   209.00),
        EASYSIMD_FLOAT32_C( -1195.00), EASYSIMD_FLOAT32_C(   182.00), EASYSIMD_FLOAT32_C(   600.00), EASYSIMD_FLOAT32_C(  1343.00),
        EASYSIMD_FLOAT32_C(    81.00), EASYSIMD_FLOAT32_C( -1898.00), EASYSIMD_FLOAT32_C(  1409.00), EASYSIMD_FLOAT32_C(  -237.00),
        EASYSIMD_FLOAT32_C(  1353.00), EASYSIMD_FLOAT32_C(  -788.00), EASYSIMD_FLOAT32_C(  1350.00), EASYSIMD_FLOAT32_C(  -299.00) },
      UINT16_C(43542) },
    { { EASYSIMD_FLOAT32_C(  -476.93), EASYSIMD_FLOAT32_C(  -744.34), EASYSIMD_FLOAT32_C(    92.53), EASYSIMD_FLOAT32_C(    78.31),
        EASYSIMD_FLOAT32_C(   880.96), EASYSIMD_FLOAT32_C(  -759.08), EASYSIMD_FLOAT32_C(  -436.34), EASYSIMD_FLOAT32_C(    -6.37),
        EASYSIMD_FLOAT32_C(  -719.72), EASYSIMD_FLOAT32_C(  -585.60), EASYSIMD_FLOAT32_C(   565.73), EASYSIMD_FLOAT32_C(   740.19),
        EASYSIMD_FLOAT32_C(  -536.91), EASYSIMD_FLOAT32_C(   289.95), EASYSIMD_FLOAT32_C(   140.16), EASYSIMD_FLOAT32_C(  -821.99) },
      { EASYSIMD_FLOAT32_C(   371.78), EASYSIMD_FLOAT32_C(  -834.77), EASYSIMD_FLOAT32_C(   596.01), EASYSIMD_FLOAT32_C(   621.87),
        EASYSIMD_FLOAT32_C(  -278.35), EASYSIMD_FLOAT32_C(    48.08), EASYSIMD_FLOAT32_C(   479.72), EASYSIMD_FLOAT32_C(   689.70),
        EASYSIMD_FLOAT32_C(   110.46), EASYSIMD_FLOAT32_C(    38.32), EASYSIMD_FLOAT32_C(  -118.92), EASYSIMD_FLOAT32_C(     3.45),
        EASYSIMD_FLOAT32_C(  -798.55), EASYSIMD_FLOAT32_C(  -492.88), EASYSIMD_FLOAT32_C(   304.78), EASYSIMD_FLOAT32_C(  -275.47) },
      { EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C( -1579.00), EASYSIMD_FLOAT32_C(   689.00), EASYSIMD_FLOAT32_C(   700.00),
        EASYSIMD_FLOAT32_C(   603.00), EASYSIMD_FLOAT32_C(  -711.00), EASYSIMD_FLOAT32_C(    43.00), EASYSIMD_FLOAT32_C(   683.00),
        EASYSIMD_FLOAT32_C(  -609.00), EASYSIMD_FLOAT32_C(  -547.00), EASYSIMD_FLOAT32_C(   447.00), EASYSIMD_FLOAT32_C(   744.00),
        EASYSIMD_FLOAT32_C( -1335.00), EASYSIMD_FLOAT32_C(  -203.00), EASYSIMD_FLOAT32_C(   445.00), EASYSIMD_FLOAT32_C( -1097.00) },
      UINT16_C(45859),
      { EASYSIMD_FLOAT32_C(  -106.00), EASYSIMD_FLOAT32_C( -1580.00), EASYSIMD_FLOAT32_C(   688.00), EASYSIMD_FLOAT32_C(   700.00),
        EASYSIMD_FLOAT32_C(   602.00), EASYSIMD_FLOAT32_C(  -711.00), EASYSIMD_FLOAT32_C(    43.00), EASYSIMD_FLOAT32_C(   683.00),
        EASYSIMD_FLOAT32_C(  -610.00), EASYSIMD_FLOAT32_C(  -548.00), EASYSIMD_FLOAT32_C(   446.00), EASYSIMD_FLOAT32_C(   743.00),
        EASYSIMD_FLOAT32_C( -1336.00), EASYSIMD_FLOAT32_C(  -203.00), EASYSIMD_FLOAT32_C(   444.00), EASYSIMD_FLOAT32_C( -1098.00) },
      UINT16_C(45859),
      { EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C( -1579.00), EASYSIMD_FLOAT32_C(   689.00), EASYSIMD_FLOAT32_C(   701.00),
        EASYSIMD_FLOAT32_C(   603.00), EASYSIMD_FLOAT32_C(  -711.00), EASYSIMD_FLOAT32_C(    44.00), EASYSIMD_FLOAT32_C(   684.00),
        EASYSIMD_FLOAT32_C(  -609.00), EASYSIMD_FLOAT32_C(  -547.00), EASYSIMD_FLOAT32_C(   447.00), EASYSIMD_FLOAT32_C(   744.00),
        EASYSIMD_FLOAT32_C( -1335.00), EASYSIMD_FLOAT32_C(  -202.00), EASYSIMD_FLOAT32_C(   445.00), EASYSIMD_FLOAT32_C( -1097.00) },
      UINT16_C(45859),
      { EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C( -1579.00), EASYSIMD_FLOAT32_C(   688.00), EASYSIMD_FLOAT32_C(   700.00),
        EASYSIMD_FLOAT32_C(   602.00), EASYSIMD_FLOAT32_C(  -711.00), EASYSIMD_FLOAT32_C(    43.00), EASYSIMD_FLOAT32_C(   683.00),
        EASYSIMD_FLOAT32_C(  -609.00), EASYSIMD_FLOAT32_C(  -547.00), EASYSIMD_FLOAT32_C(   446.00), EASYSIMD_FLOAT32_C(   743.00),
        EASYSIMD_FLOAT32_C( -1335.00), EASYSIMD_FLOAT32_C(  -202.00), EASYSIMD_FLOAT32_C(   444.00), EASYSIMD_FLOAT32_C( -1097.00) },
      UINT16_C(45859),
      { EASYSIMD_FLOAT32_C(  -105.00), EASYSIMD_FLOAT32_C( -1579.00), EASYSIMD_FLOAT32_C(   689.00), EASYSIMD_FLOAT32_C(   700.00),
        EASYSIMD_FLOAT32_C(   603.00), EASYSIMD_FLOAT32_C(  -711.00), EASYSIMD_FLOAT32_C(    43.00), EASYSIMD_FLOAT32_C(   683.00),
        EASYSIMD_FLOAT32_C(  -609.00), EASYSIMD_FLOAT32_C(  -547.00), EASYSIMD_FLOAT32_C(   447.00), EASYSIMD_FLOAT32_C(   744.00),
        EASYSIMD_FLOAT32_C( -1335.00), EASYSIMD_FLOAT32_C(  -203.00), EASYSIMD_FLOAT32_C(   445.00), EASYSIMD_FLOAT32_C( -1097.00) },
      UINT16_C(45859) },
    { { EASYSIMD_FLOAT32_C(  -237.23), EASYSIMD_FLOAT32_C(  -602.69), EASYSIMD_FLOAT32_C(   802.83), EASYSIMD_FLOAT32_C(  -356.27),
        EASYSIMD_FLOAT32_C(  -361.77), EASYSIMD_FLOAT32_C(  -633.51), EASYSIMD_FLOAT32_C(   637.36), EASYSIMD_FLOAT32_C(   -81.49),
        EASYSIMD_FLOAT32_C(  -219.11), EASYSIMD_FLOAT32_C(   203.09), EASYSIMD_FLOAT32_C(  -341.31), EASYSIMD_FLOAT32_C(   243.98),
        EASYSIMD_FLOAT32_C(  -506.96), EASYSIMD_FLOAT32_C(   798.85), EASYSIMD_FLOAT32_C(   422.00), EASYSIMD_FLOAT32_C(   864.82) },
      { EASYSIMD_FLOAT32_C(   964.08), EASYSIMD_FLOAT32_C(    18.00), EASYSIMD_FLOAT32_C(   486.70), EASYSIMD_FLOAT32_C(  -314.27),
        EASYSIMD_FLOAT32_C(  -933.92), EASYSIMD_FLOAT32_C(   -33.59), EASYSIMD_FLOAT32_C(  -624.57), EASYSIMD_FLOAT32_C(   176.55),
        EASYSIMD_FLOAT32_C(  -995.27), EASYSIMD_FLOAT32_C(   256.51), EASYSIMD_FLOAT32_C(  -820.00), EASYSIMD_FLOAT32_C(  -793.81),
        EASYSIMD_FLOAT32_C(   763.63), EASYSIMD_FLOAT32_C(   484.77), EASYSIMD_FLOAT32_C(   -69.29), EASYSIMD_FLOAT32_C(  -473.60) },
      { EASYSIMD_FLOAT32_C(   727.00), EASYSIMD_FLOAT32_C(  -585.00), EASYSIMD_FLOAT32_C(  1290.00), EASYSIMD_FLOAT32_C(  -671.00),
        EASYSIMD_FLOAT32_C( -1296.00), EASYSIMD_FLOAT32_C(  -667.00), EASYSIMD_FLOAT32_C(    13.00), EASYSIMD_FLOAT32_C(    95.00),
        EASYSIMD_FLOAT32_C( -1214.00), EASYSIMD_FLOAT32_C(   460.00), EASYSIMD_FLOAT32_C( -1161.00), EASYSIMD_FLOAT32_C(  -550.00),
        EASYSIMD_FLOAT32_C(   257.00), EASYSIMD_FLOAT32_C(  1284.00), EASYSIMD_FLOAT32_C(   353.00), EASYSIMD_FLOAT32_C(   391.00) },
      UINT16_C( 3386),
      { EASYSIMD_FLOAT32_C(   726.00), EASYSIMD_FLOAT32_C(  -585.00), EASYSIMD_FLOAT32_C(  1289.00), EASYSIMD_FLOAT32_C(  -671.00),
        EASYSIMD_FLOAT32_C( -1296.00), EASYSIMD_FLOAT32_C(  -668.00), EASYSIMD_FLOAT32_C(    12.00), EASYSIMD_FLOAT32_C(    95.00),
        EASYSIMD_FLOAT32_C( -1215.00), EASYSIMD_FLOAT32_C(   459.00), EASYSIMD_FLOAT32_C( -1162.00), EASYSIMD_FLOAT32_C(  -550.00),
        EASYSIMD_FLOAT32_C(   256.00), EASYSIMD_FLOAT32_C(  1283.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(   391.00) },
      UINT16_C( 3386),
      { EASYSIMD_FLOAT32_C(   727.00), EASYSIMD_FLOAT32_C(  -584.00), EASYSIMD_FLOAT32_C(  1290.00), EASYSIMD_FLOAT32_C(  -670.00),
        EASYSIMD_FLOAT32_C( -1295.00), EASYSIMD_FLOAT32_C(  -667.00), EASYSIMD_FLOAT32_C(    13.00), EASYSIMD_FLOAT32_C(    96.00),
        EASYSIMD_FLOAT32_C( -1214.00), EASYSIMD_FLOAT32_C(   460.00), EASYSIMD_FLOAT32_C( -1161.00), EASYSIMD_FLOAT32_C(  -549.00),
        EASYSIMD_FLOAT32_C(   257.00), EASYSIMD_FLOAT32_C(  1284.00), EASYSIMD_FLOAT32_C(   353.00), EASYSIMD_FLOAT32_C(   392.00) },
      UINT16_C( 3386),
      { EASYSIMD_FLOAT32_C(   726.00), EASYSIMD_FLOAT32_C(  -584.00), EASYSIMD_FLOAT32_C(  1289.00), EASYSIMD_FLOAT32_C(  -670.00),
        EASYSIMD_FLOAT32_C( -1295.00), EASYSIMD_FLOAT32_C(  -667.00), EASYSIMD_FLOAT32_C(    12.00), EASYSIMD_FLOAT32_C(    95.00),
        EASYSIMD_FLOAT32_C( -1214.00), EASYSIMD_FLOAT32_C(   459.00), EASYSIMD_FLOAT32_C( -1161.00), EASYSIMD_FLOAT32_C(  -549.00),
        EASYSIMD_FLOAT32_C(   256.00), EASYSIMD_FLOAT32_C(  1283.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(   391.00) },
      UINT16_C( 3386),
      { EASYSIMD_FLOAT32_C(   727.00), EASYSIMD_FLOAT32_C(  -585.00), EASYSIMD_FLOAT32_C(  1290.00), EASYSIMD_FLOAT32_C(  -671.00),
        EASYSIMD_FLOAT32_C( -1296.00), EASYSIMD_FLOAT32_C(  -667.00), EASYSIMD_FLOAT32_C(    13.00), EASYSIMD_FLOAT32_C(    95.00),
        EASYSIMD_FLOAT32_C( -1214.00), EASYSIMD_FLOAT32_C(   460.00), EASYSIMD_FLOAT32_C( -1161.00), EASYSIMD_FLOAT32_C(  -550.00),
        EASYSIMD_FLOAT32_C(   257.00), EASYSIMD_FLOAT32_C(  1284.00), EASYSIMD_FLOAT32_C(   353.00), EASYSIMD_FLOAT32_C(   391.00) },
      UINT16_C( 3386) },
    { { EASYSIMD_FLOAT32_C(   882.08), EASYSIMD_FLOAT32_C(  -266.45), EASYSIMD_FLOAT32_C(   170.14), EASYSIMD_FLOAT32_C(  -479.69),
        EASYSIMD_FLOAT32_C(   100.04), EASYSIMD_FLOAT32_C(  -192.50), EASYSIMD_FLOAT32_C(   438.81), EASYSIMD_FLOAT32_C(   880.93),
        EASYSIMD_FLOAT32_C(  -989.41), EASYSIMD_FLOAT32_C(  -902.49), EASYSIMD_FLOAT32_C(   124.91), EASYSIMD_FLOAT32_C(  -496.37),
        EASYSIMD_FLOAT32_C(   896.36), EASYSIMD_FLOAT32_C(  -453.09), EASYSIMD_FLOAT32_C(  -631.55), EASYSIMD_FLOAT32_C(   860.44) },
      { EASYSIMD_FLOAT32_C(   564.91), EASYSIMD_FLOAT32_C(   855.15), EASYSIMD_FLOAT32_C(  -453.83), EASYSIMD_FLOAT32_C(   631.00),
        EASYSIMD_FLOAT32_C(  -178.44), EASYSIMD_FLOAT32_C(   -78.40), EASYSIMD_FLOAT32_C(  -192.45), EASYSIMD_FLOAT32_C(  -173.70),
        EASYSIMD_FLOAT32_C(  -821.89), EASYSIMD_FLOAT32_C(   -12.46), EASYSIMD_FLOAT32_C(    32.48), EASYSIMD_FLOAT32_C(   941.74),
        EASYSIMD_FLOAT32_C(  -527.69), EASYSIMD_FLOAT32_C(   963.19), EASYSIMD_FLOAT32_C(  -531.86), EASYSIMD_FLOAT32_C(  -645.61) },
      { EASYSIMD_FLOAT32_C(  1447.00), EASYSIMD_FLOAT32_C(   589.00), EASYSIMD_FLOAT32_C(  -284.00), EASYSIMD_FLOAT32_C(   151.00),
        EASYSIMD_FLOAT32_C(   -78.00), EASYSIMD_FLOAT32_C(  -271.00), EASYSIMD_FLOAT32_C(   246.00), EASYSIMD_FLOAT32_C(   707.00),
        EASYSIMD_FLOAT32_C( -1811.00), EASYSIMD_FLOAT32_C(  -915.00), EASYSIMD_FLOAT32_C(   157.00), EASYSIMD_FLOAT32_C(   445.00),
        EASYSIMD_FLOAT32_C(   369.00), EASYSIMD_FLOAT32_C(   510.00), EASYSIMD_FLOAT32_C( -1163.00), EASYSIMD_FLOAT32_C(   215.00) },
      UINT16_C(17204),
      { EASYSIMD_FLOAT32_C(  1446.00), EASYSIMD_FLOAT32_C(   588.00), EASYSIMD_FLOAT32_C(  -284.00), EASYSIMD_FLOAT32_C(   151.00),
        EASYSIMD_FLOAT32_C(   -79.00), EASYSIMD_FLOAT32_C(  -271.00), EASYSIMD_FLOAT32_C(   246.00), EASYSIMD_FLOAT32_C(   707.00),
        EASYSIMD_FLOAT32_C( -1812.00), EASYSIMD_FLOAT32_C(  -915.00), EASYSIMD_FLOAT32_C(   157.00), EASYSIMD_FLOAT32_C(   445.00),
        EASYSIMD_FLOAT32_C(   368.00), EASYSIMD_FLOAT32_C(   510.00), EASYSIMD_FLOAT32_C( -1164.00), EASYSIMD_FLOAT32_C(   214.00) },
      UINT16_C(17204),
      { EASYSIMD_FLOAT32_C(  1447.00), EASYSIMD_FLOAT32_C(   589.00), EASYSIMD_FLOAT32_C(  -283.00), EASYSIMD_FLOAT32_C(   152.00),
        EASYSIMD_FLOAT32_C(   -78.00), EASYSIMD_FLOAT32_C(  -270.00), EASYSIMD_FLOAT32_C(   247.00), EASYSIMD_FLOAT32_C(   708.00),
        EASYSIMD_FLOAT32_C( -1811.00), EASYSIMD_FLOAT32_C(  -914.00), EASYSIMD_FLOAT32_C(   158.00), EASYSIMD_FLOAT32_C(   446.00),
        EASYSIMD_FLOAT32_C(   369.00), EASYSIMD_FLOAT32_C(   511.00), EASYSIMD_FLOAT32_C( -1163.00), EASYSIMD_FLOAT32_C(   215.00) },
      UINT16_C(17204),
      { EASYSIMD_FLOAT32_C(  1446.00), EASYSIMD_FLOAT32_C(   588.00), EASYSIMD_FLOAT32_C(  -283.00), EASYSIMD_FLOAT32_C(   151.00),
        EASYSIMD_FLOAT32_C(   -78.00), EASYSIMD_FLOAT32_C(  -270.00), EASYSIMD_FLOAT32_C(   246.00), EASYSIMD_FLOAT32_C(   707.00),
        EASYSIMD_FLOAT32_C( -1811.00), EASYSIMD_FLOAT32_C(  -914.00), EASYSIMD_FLOAT32_C(   157.00), EASYSIMD_FLOAT32_C(   445.00),
        EASYSIMD_FLOAT32_C(   368.00), EASYSIMD_FLOAT32_C(   510.00), EASYSIMD_FLOAT32_C( -1163.00), EASYSIMD_FLOAT32_C(   214.00) },
      UINT16_C(17204),
      { EASYSIMD_FLOAT32_C(  1447.00), EASYSIMD_FLOAT32_C(   589.00), EASYSIMD_FLOAT32_C(  -284.00), EASYSIMD_FLOAT32_C(   151.00),
        EASYSIMD_FLOAT32_C(   -78.00), EASYSIMD_FLOAT32_C(  -271.00), EASYSIMD_FLOAT32_C(   246.00), EASYSIMD_FLOAT32_C(   707.00),
        EASYSIMD_FLOAT32_C( -1811.00), EASYSIMD_FLOAT32_C(  -915.00), EASYSIMD_FLOAT32_C(   157.00), EASYSIMD_FLOAT32_C(   445.00),
        EASYSIMD_FLOAT32_C(   369.00), EASYSIMD_FLOAT32_C(   510.00), EASYSIMD_FLOAT32_C( -1163.00), EASYSIMD_FLOAT32_C(   215.00) },
      UINT16_C(17204) },
    { { EASYSIMD_FLOAT32_C(  -303.26), EASYSIMD_FLOAT32_C(   638.27), EASYSIMD_FLOAT32_C(  -125.30), EASYSIMD_FLOAT32_C(   796.78),
        EASYSIMD_FLOAT32_C(  -554.23), EASYSIMD_FLOAT32_C(  -686.49), EASYSIMD_FLOAT32_C(   677.71), EASYSIMD_FLOAT32_C(  -543.64),
        EASYSIMD_FLOAT32_C(  -588.99), EASYSIMD_FLOAT32_C(  -197.38), EASYSIMD_FLOAT32_C(   -40.01), EASYSIMD_FLOAT32_C(  -692.63),
        EASYSIMD_FLOAT32_C(   349.53), EASYSIMD_FLOAT32_C(   328.44), EASYSIMD_FLOAT32_C(  -832.19), EASYSIMD_FLOAT32_C(   -85.56) },
      { EASYSIMD_FLOAT32_C(   183.59), EASYSIMD_FLOAT32_C(  -286.02), EASYSIMD_FLOAT32_C(  -454.56), EASYSIMD_FLOAT32_C(  -994.84),
        EASYSIMD_FLOAT32_C(   635.58), EASYSIMD_FLOAT32_C(   352.98), EASYSIMD_FLOAT32_C(  -168.55), EASYSIMD_FLOAT32_C(   813.69),
        EASYSIMD_FLOAT32_C(  -659.47), EASYSIMD_FLOAT32_C(   863.93), EASYSIMD_FLOAT32_C(   755.43), EASYSIMD_FLOAT32_C(  -187.16),
        EASYSIMD_FLOAT32_C(   827.13), EASYSIMD_FLOAT32_C(  -776.43), EASYSIMD_FLOAT32_C(   167.23), EASYSIMD_FLOAT32_C(  -476.14) },
      { EASYSIMD_FLOAT32_C(  -120.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(  -580.00), EASYSIMD_FLOAT32_C(  -198.00),
        EASYSIMD_FLOAT32_C(    81.00), EASYSIMD_FLOAT32_C(  -334.00), EASYSIMD_FLOAT32_C(   509.00), EASYSIMD_FLOAT32_C(   270.00),
        EASYSIMD_FLOAT32_C( -1248.00), EASYSIMD_FLOAT32_C(   667.00), EASYSIMD_FLOAT32_C(   715.00), EASYSIMD_FLOAT32_C(  -880.00),
        EASYSIMD_FLOAT32_C(  1177.00), EASYSIMD_FLOAT32_C(  -448.00), EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(  -562.00) },
      UINT16_C(59693),
      { EASYSIMD_FLOAT32_C(  -120.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(  -580.00), EASYSIMD_FLOAT32_C(  -199.00),
        EASYSIMD_FLOAT32_C(    81.00), EASYSIMD_FLOAT32_C(  -334.00), EASYSIMD_FLOAT32_C(   509.00), EASYSIMD_FLOAT32_C(   270.00),
        EASYSIMD_FLOAT32_C( -1249.00), EASYSIMD_FLOAT32_C(   666.00), EASYSIMD_FLOAT32_C(   715.00), EASYSIMD_FLOAT32_C(  -880.00),
        EASYSIMD_FLOAT32_C(  1176.00), EASYSIMD_FLOAT32_C(  -448.00), EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(  -562.00) },
      UINT16_C(59693),
      { EASYSIMD_FLOAT32_C(  -119.00), EASYSIMD_FLOAT32_C(   353.00), EASYSIMD_FLOAT32_C(  -579.00), EASYSIMD_FLOAT32_C(  -198.00),
        EASYSIMD_FLOAT32_C(    82.00), EASYSIMD_FLOAT32_C(  -333.00), EASYSIMD_FLOAT32_C(   510.00), EASYSIMD_FLOAT32_C(   271.00),
        EASYSIMD_FLOAT32_C( -1248.00), EASYSIMD_FLOAT32_C(   667.00), EASYSIMD_FLOAT32_C(   716.00), EASYSIMD_FLOAT32_C(  -879.00),
        EASYSIMD_FLOAT32_C(  1177.00), EASYSIMD_FLOAT32_C(  -447.00), EASYSIMD_FLOAT32_C(  -664.00), EASYSIMD_FLOAT32_C(  -561.00) },
      UINT16_C(59693),
      { EASYSIMD_FLOAT32_C(  -119.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(  -579.00), EASYSIMD_FLOAT32_C(  -198.00),
        EASYSIMD_FLOAT32_C(    81.00), EASYSIMD_FLOAT32_C(  -333.00), EASYSIMD_FLOAT32_C(   509.00), EASYSIMD_FLOAT32_C(   270.00),
        EASYSIMD_FLOAT32_C( -1248.00), EASYSIMD_FLOAT32_C(   666.00), EASYSIMD_FLOAT32_C(   715.00), EASYSIMD_FLOAT32_C(  -879.00),
        EASYSIMD_FLOAT32_C(  1176.00), EASYSIMD_FLOAT32_C(  -447.00), EASYSIMD_FLOAT32_C(  -664.00), EASYSIMD_FLOAT32_C(  -561.00) },
      UINT16_C(59693),
      { EASYSIMD_FLOAT32_C(  -120.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(  -580.00), EASYSIMD_FLOAT32_C(  -198.00),
        EASYSIMD_FLOAT32_C(    81.00), EASYSIMD_FLOAT32_C(  -334.00), EASYSIMD_FLOAT32_C(   509.00), EASYSIMD_FLOAT32_C(   270.00),
        EASYSIMD_FLOAT32_C( -1248.00), EASYSIMD_FLOAT32_C(   667.00), EASYSIMD_FLOAT32_C(   715.00), EASYSIMD_FLOAT32_C(  -880.00),
        EASYSIMD_FLOAT32_C(  1177.00), EASYSIMD_FLOAT32_C(  -448.00), EASYSIMD_FLOAT32_C(  -665.00), EASYSIMD_FLOAT32_C(  -562.00) },
      UINT16_C(59693) },
    { { EASYSIMD_FLOAT32_C(   861.84), EASYSIMD_FLOAT32_C(  -958.08), EASYSIMD_FLOAT32_C(  -679.36), EASYSIMD_FLOAT32_C(  -692.39),
        EASYSIMD_FLOAT32_C(  -644.57), EASYSIMD_FLOAT32_C(   998.35), EASYSIMD_FLOAT32_C(  -236.03), EASYSIMD_FLOAT32_C(  -233.56),
        EASYSIMD_FLOAT32_C(  -199.03), EASYSIMD_FLOAT32_C(   723.96), EASYSIMD_FLOAT32_C(    73.81), EASYSIMD_FLOAT32_C(  -849.50),
        EASYSIMD_FLOAT32_C(    52.41), EASYSIMD_FLOAT32_C(   241.62), EASYSIMD_FLOAT32_C(    64.95), EASYSIMD_FLOAT32_C(  -764.00) },
      { EASYSIMD_FLOAT32_C(   955.60), EASYSIMD_FLOAT32_C(   610.39), EASYSIMD_FLOAT32_C(  -758.84), EASYSIMD_FLOAT32_C(   591.18),
        EASYSIMD_FLOAT32_C(   -36.63), EASYSIMD_FLOAT32_C(    72.61), EASYSIMD_FLOAT32_C(   404.87), EASYSIMD_FLOAT32_C(   303.90),
        EASYSIMD_FLOAT32_C(   -63.46), EASYSIMD_FLOAT32_C(   160.30), EASYSIMD_FLOAT32_C(  -883.26), EASYSIMD_FLOAT32_C(  -236.33),
        EASYSIMD_FLOAT32_C(   383.86), EASYSIMD_FLOAT32_C(   283.96), EASYSIMD_FLOAT32_C(   287.54), EASYSIMD_FLOAT32_C(   245.70) },
      { EASYSIMD_FLOAT32_C(  1817.00), EASYSIMD_FLOAT32_C(  -348.00), EASYSIMD_FLOAT32_C( -1438.00), EASYSIMD_FLOAT32_C(  -101.00),
        EASYSIMD_FLOAT32_C(  -681.00), EASYSIMD_FLOAT32_C(  1071.00), EASYSIMD_FLOAT32_C(   169.00), EASYSIMD_FLOAT32_C(    70.00),
        EASYSIMD_FLOAT32_C(  -262.00), EASYSIMD_FLOAT32_C(   884.00), EASYSIMD_FLOAT32_C(  -809.00), EASYSIMD_FLOAT32_C( -1086.00),
        EASYSIMD_FLOAT32_C(   436.00), EASYSIMD_FLOAT32_C(   526.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(  -518.00) },
      UINT16_C(36126),
      { EASYSIMD_FLOAT32_C(  1817.00), EASYSIMD_FLOAT32_C(  -348.00), EASYSIMD_FLOAT32_C( -1439.00), EASYSIMD_FLOAT32_C(  -102.00),
        EASYSIMD_FLOAT32_C(  -682.00), EASYSIMD_FLOAT32_C(  1070.00), EASYSIMD_FLOAT32_C(   168.00), EASYSIMD_FLOAT32_C(    70.00),
        EASYSIMD_FLOAT32_C(  -263.00), EASYSIMD_FLOAT32_C(   884.00), EASYSIMD_FLOAT32_C(  -810.00), EASYSIMD_FLOAT32_C( -1086.00),
        EASYSIMD_FLOAT32_C(   436.00), EASYSIMD_FLOAT32_C(   525.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(  -519.00) },
      UINT16_C(36126),
      { EASYSIMD_FLOAT32_C(  1818.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C( -1438.00), EASYSIMD_FLOAT32_C(  -101.00),
        EASYSIMD_FLOAT32_C(  -681.00), EASYSIMD_FLOAT32_C(  1071.00), EASYSIMD_FLOAT32_C(   169.00), EASYSIMD_FLOAT32_C(    71.00),
        EASYSIMD_FLOAT32_C(  -262.00), EASYSIMD_FLOAT32_C(   885.00), EASYSIMD_FLOAT32_C(  -809.00), EASYSIMD_FLOAT32_C( -1085.00),
        EASYSIMD_FLOAT32_C(   437.00), EASYSIMD_FLOAT32_C(   526.00), EASYSIMD_FLOAT32_C(   353.00), EASYSIMD_FLOAT32_C(  -518.00) },
      UINT16_C(36126),
      { EASYSIMD_FLOAT32_C(  1817.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C( -1438.00), EASYSIMD_FLOAT32_C(  -101.00),
        EASYSIMD_FLOAT32_C(  -681.00), EASYSIMD_FLOAT32_C(  1070.00), EASYSIMD_FLOAT32_C(   168.00), EASYSIMD_FLOAT32_C(    70.00),
        EASYSIMD_FLOAT32_C(  -262.00), EASYSIMD_FLOAT32_C(   884.00), EASYSIMD_FLOAT32_C(  -809.00), EASYSIMD_FLOAT32_C( -1085.00),
        EASYSIMD_FLOAT32_C(   436.00), EASYSIMD_FLOAT32_C(   525.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(  -518.00) },
      UINT16_C(36126),
      { EASYSIMD_FLOAT32_C(  1817.00), EASYSIMD_FLOAT32_C(  -348.00), EASYSIMD_FLOAT32_C( -1438.00), EASYSIMD_FLOAT32_C(  -101.00),
        EASYSIMD_FLOAT32_C(  -681.00), EASYSIMD_FLOAT32_C(  1071.00), EASYSIMD_FLOAT32_C(   169.00), EASYSIMD_FLOAT32_C(    70.00),
        EASYSIMD_FLOAT32_C(  -262.00), EASYSIMD_FLOAT32_C(   884.00), EASYSIMD_FLOAT32_C(  -809.00), EASYSIMD_FLOAT32_C( -1086.00),
        EASYSIMD_FLOAT32_C(   436.00), EASYSIMD_FLOAT32_C(   526.00), EASYSIMD_FLOAT32_C(   352.00), EASYSIMD_FLOAT32_C(  -518.00) },
      UINT16_C(36126) }
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
    easysimd__mmask16 sign1, sign2, sign3, sign4, sign5;
    sign1 = sign2 = sign3 = sign4 = sign5 = 0;

    r = easysimd_mm512_addsets_round_ps(a, b, &sign1, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd_assert_m512_close(r, nearest_inf, 1);
    easysimd_assert_equal_mmask16(sign1, test_vec[i].sign1);

    r = easysimd_mm512_addsets_round_ps(a, b, &sign2, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd_assert_m512_close(r, neg_inf, 1);
    easysimd_assert_equal_mmask16(sign2, test_vec[i].sign2);

    r = easysimd_mm512_addsets_round_ps(a, b, &sign3, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd_assert_m512_close(r, pos_inf, 1);
    easysimd_assert_equal_mmask16(sign3, test_vec[i].sign3);

    r = easysimd_mm512_addsets_round_ps(a, b, &sign4, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd_assert_m512_close(r, zero, 1);
    easysimd_assert_equal_mmask16(sign4, test_vec[i].sign4);

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_addsets_round_ps(a, b, &sign5, EASYSIMD_MM_FROUND_CUR_DIRECTION);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_addsets_round_ps");
    easysimd_assert_m512_close(r, direction, 1);
    easysimd_assert_equal_mmask16(sign5, test_vec[i].sign5);
  }

  return 0;
  #else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask16 sign1, sign2, sign3, sign4, sign5;
    sign1 = sign2 = sign3 = sign4 = sign5 = 0;

    easysimd__m512 nearest_inf = easysimd_mm512_addsets_round_ps(a, b, &sign1, EASYSIMD_MM_FROUND_TO_NEAREST_INT);
    easysimd__m512 neg_inf = easysimd_mm512_addsets_round_ps(a, b, &sign2, EASYSIMD_MM_FROUND_TO_NEG_INF);
    easysimd__m512 pos_inf = easysimd_mm512_addsets_round_ps(a, b, &sign3, EASYSIMD_MM_FROUND_TO_POS_INF);
    easysimd__m512 zero = easysimd_mm512_addsets_round_ps(a, b, &sign4, EASYSIMD_MM_FROUND_TO_ZERO);
    easysimd__m512 direction = easysimd_mm512_addsets_round_ps(a, b, &sign5, EASYSIMD_MM_FROUND_CUR_DIRECTION);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, nearest_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, sign1, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, neg_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, sign2, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, pos_inf, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, sign3, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, zero, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, sign4, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, direction, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_mmask16(2, sign5, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
  #endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_add_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_add_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_add_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_add_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_add_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_add_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_add_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_add_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_add_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_add_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_add_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_add_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_add_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_add_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_add_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_add_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_add_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_add_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_add_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_add_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_add_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_add_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_add_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_add_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_add_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_add_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_add_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_add_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_add_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_add_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_add_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_add_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_add_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_add_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_add_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_add_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_add_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_add_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_add_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_add_round_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_add_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_add_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_add_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_add_round_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_add_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_add_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_addn_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_addn_round_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_addn_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_addn_round_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_addsetc_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_addsets_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_addsets_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_addsets_round_ps)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
