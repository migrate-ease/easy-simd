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
 *   2020      Christopher Moore <moore@free.fr>
 */

#define EASYSIMD_TEST_X86_AVX512_INSN broadcast

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/broadcast.h>

static int
test_easysimd_mm_broadcastmw_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint16_t k;
    const int32_t r[4];
  } test_vec[8] = {
    { UINT16_C(16198),
      {  INT32_C(       16198),  INT32_C(       16198),  INT32_C(       16198),  INT32_C(       16198) } },
    { UINT16_C(  738),
      {  INT32_C(         738),  INT32_C(         738),  INT32_C(         738),  INT32_C(         738) } },
    { UINT16_C( 7768),
      {  INT32_C(        7768),  INT32_C(        7768),  INT32_C(        7768),  INT32_C(        7768) } },
    { UINT16_C(22648),
      {  INT32_C(       22648),  INT32_C(       22648),  INT32_C(       22648),  INT32_C(       22648) } },
    { UINT16_C(11844),
      {  INT32_C(       11844),  INT32_C(       11844),  INT32_C(       11844),  INT32_C(       11844) } },
    { UINT16_C(33300),
      {  INT32_C(       33300),  INT32_C(       33300),  INT32_C(       33300),  INT32_C(       33300) } },
    { UINT16_C(59721),
      {  INT32_C(       59721),  INT32_C(       59721),  INT32_C(       59721),  INT32_C(       59721) } },
    { UINT16_C(62372),
      {  INT32_C(       62372),  INT32_C(       62372),  INT32_C(       62372),  INT32_C(       62372) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_broadcastmw_epi32(k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_broadcastmw_epi32");

    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i r = easysimd_mm_broadcastmw_epi32(k);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_broadcastmb_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const int64_t r[2];
  } test_vec[8] = {
    { UINT8_C( 62),
      {  INT64_C(                  62),  INT64_C(                  62) } },
    { UINT8_C(212),
      {  INT64_C(                 212),  INT64_C(                 212) } },
    {    UINT8_MAX,
      {  INT64_C(                 255),  INT64_C(                 255) } },
    { UINT8_C( 65),
      {  INT64_C(                  65),  INT64_C(                  65) } },
    { UINT8_C(134),
      {  INT64_C(                 134),  INT64_C(                 134) } },
    { UINT8_C( 74),
      {  INT64_C(                  74),  INT64_C(                  74) } },
    { UINT8_C(196),
      {  INT64_C(                 196),  INT64_C(                 196) } },
    { UINT8_C(246),
      {  INT64_C(                 246),  INT64_C(                 246) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_broadcastmb_epi64(k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_broadcastmb_epi64");

    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i r = easysimd_mm_broadcastmb_epi64(k);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_broadcastb_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t src[16];
    uint16_t k;
    int8_t a[16];
    int8_t r[16];
  } test_vec[] = {
    { { -INT8_C(  89),  INT8_C( 112),  INT8_C(  17),  INT8_C(  91),  INT8_C( 121), -INT8_C( 101), -INT8_C(  67),  INT8_C( 125),
         INT8_C(  40), -INT8_C(  78), -INT8_C(   6),  INT8_C(  51),  INT8_C( 120), -INT8_C( 107),  INT8_C(  36), -INT8_C(  69) },
      UINT16_C(10540),
      { -INT8_C(  27), -INT8_C(  46), -INT8_C(  89), -INT8_C(  92),  INT8_C( 112), -INT8_C(  33),  INT8_C( 106),  INT8_C(  23),
         INT8_C(  81),  INT8_C(  23), -INT8_C(  46), -INT8_C(  72),  INT8_C(  90),  INT8_C( 122),  INT8_C(  41),  INT8_C( 107) },
      { -INT8_C(  89),  INT8_C( 112), -INT8_C(  27), -INT8_C(  27),  INT8_C( 121), -INT8_C(  27), -INT8_C(  67),  INT8_C( 125),
        -INT8_C(  27), -INT8_C(  78), -INT8_C(   6), -INT8_C(  27),  INT8_C( 120), -INT8_C(  27),  INT8_C(  36), -INT8_C(  69) } },
    { { -INT8_C(  43), -INT8_C(  94),  INT8_C(   7), -INT8_C( 110),  INT8_C(  31),  INT8_C(  47),  INT8_C(  68),  INT8_C(  26),
         INT8_C(  98), -INT8_C(  68), -INT8_C(  81), -INT8_C( 121),  INT8_C( 120), -INT8_C(  37), -INT8_C(  80),  INT8_C(  93) },
      UINT16_C(22701),
      {  INT8_C(   1),  INT8_C(  29),  INT8_C(  55),  INT8_C( 107),  INT8_C(  53), -INT8_C( 120), -INT8_C( 125),  INT8_C(   7),
         INT8_C(  65), -INT8_C(  35), -INT8_C( 127),  INT8_C( 106),  INT8_C(  73),  INT8_C(  87),  INT8_C(  12),  INT8_C(  80) },
      {  INT8_C(   1), -INT8_C(  94),  INT8_C(   1),  INT8_C(   1),  INT8_C(  31),  INT8_C(   1),  INT8_C(  68),  INT8_C(   1),
         INT8_C(  98), -INT8_C(  68), -INT8_C(  81),  INT8_C(   1),  INT8_C(   1), -INT8_C(  37),  INT8_C(   1),  INT8_C(  93) } },
    { { -INT8_C(  23),  INT8_C(  43),      INT8_MAX,  INT8_C(  46),  INT8_C(  69), -INT8_C(  31), -INT8_C(  22), -INT8_C(  11),
         INT8_C( 104),  INT8_C(  98), -INT8_C(  48),  INT8_C(  25), -INT8_C(  65),  INT8_C( 126),  INT8_C( 113), -INT8_C(  64) },
      UINT16_C(43163),
      {  INT8_C(  44), -INT8_C(  48),  INT8_C(  49), -INT8_C(  81), -INT8_C(  40),  INT8_C( 114), -INT8_C( 116),  INT8_C(  89),
        -INT8_C(  36), -INT8_C(  43), -INT8_C(  80), -INT8_C(  24),  INT8_C(  37), -INT8_C( 102),  INT8_C(  19), -INT8_C(  92) },
      {  INT8_C(  44),  INT8_C(  44),      INT8_MAX,  INT8_C(  44),  INT8_C(  44), -INT8_C(  31), -INT8_C(  22),  INT8_C(  44),
         INT8_C( 104),  INT8_C(  98), -INT8_C(  48),  INT8_C(  44), -INT8_C(  65),  INT8_C(  44),  INT8_C( 113),  INT8_C(  44) } },
    { { -INT8_C(  56),  INT8_C(  89), -INT8_C( 122), -INT8_C(  78),  INT8_C(  78), -INT8_C(  18),  INT8_C(  21),  INT8_C(  30),
         INT8_C(   7), -INT8_C(  44), -INT8_C( 100),  INT8_C( 120), -INT8_C( 107),  INT8_C(  56),  INT8_C(  33), -INT8_C(  63) },
      UINT16_C(21000),
      {  INT8_C( 112), -INT8_C(  32), -INT8_C(  60), -INT8_C(   4),  INT8_C(  58), -INT8_C(  96), -INT8_C(  46), -INT8_C(  22),
        -INT8_C( 120), -INT8_C(   9), -INT8_C( 124), -INT8_C( 101), -INT8_C( 100),  INT8_C(  76), -INT8_C(  12),  INT8_C(  34) },
      { -INT8_C(  56),  INT8_C(  89), -INT8_C( 122),  INT8_C( 112),  INT8_C(  78), -INT8_C(  18),  INT8_C(  21),  INT8_C(  30),
         INT8_C(   7),  INT8_C( 112), -INT8_C( 100),  INT8_C( 120),  INT8_C( 112),  INT8_C(  56),  INT8_C( 112), -INT8_C(  63) } },
    { { -INT8_C(   1),  INT8_C(  66),  INT8_C(  16),  INT8_C(  20),  INT8_C(  97),  INT8_C(  24), -INT8_C(  24), -INT8_C(   3),
        -INT8_C( 112),  INT8_C( 125),  INT8_C(  53), -INT8_C(  79),  INT8_C(  62),  INT8_C(  62),  INT8_C(   3), -INT8_C(  82) },
      UINT16_C(50974),
      { -INT8_C(  85),  INT8_C(  88),  INT8_C( 103),  INT8_C( 125),  INT8_C(  67), -INT8_C(  17),  INT8_C( 116), -INT8_C(  57),
        -INT8_C( 117),  INT8_C(  16),  INT8_C(  20),      INT8_MAX,  INT8_C(  50),  INT8_C(  19), -INT8_C(  62),  INT8_C(  67) },
      { -INT8_C(   1), -INT8_C(  85), -INT8_C(  85), -INT8_C(  85), -INT8_C(  85),  INT8_C(  24), -INT8_C(  24), -INT8_C(   3),
        -INT8_C(  85), -INT8_C(  85), -INT8_C(  85), -INT8_C(  79),  INT8_C(  62),  INT8_C(  62), -INT8_C(  85), -INT8_C(  85) } },
    { {  INT8_C(  39),  INT8_C(  35),  INT8_C(  91),  INT8_C(  15),  INT8_C(  32), -INT8_C(  21), -INT8_C( 115),  INT8_C(  86),
        -INT8_C(  99), -INT8_C(  53), -INT8_C( 108), -INT8_C(  96),  INT8_C( 122), -INT8_C(  78),  INT8_C( 104),  INT8_C(  37) },
      UINT16_C(53003),
      { -INT8_C(  94),  INT8_C(  78), -INT8_C(  65),  INT8_C(  22),  INT8_C(  21),  INT8_C(  74),  INT8_C(  39),  INT8_C(  41),
        -INT8_C(  55),  INT8_C(  89),  INT8_C(  60), -INT8_C( 117), -INT8_C( 100),  INT8_C(  99), -INT8_C(  82), -INT8_C(   9) },
      { -INT8_C(  94), -INT8_C(  94),  INT8_C(  91), -INT8_C(  94),  INT8_C(  32), -INT8_C(  21), -INT8_C( 115),  INT8_C(  86),
        -INT8_C(  94), -INT8_C(  94), -INT8_C(  94), -INT8_C(  94),  INT8_C( 122), -INT8_C(  78), -INT8_C(  94), -INT8_C(  94) } },
    { {  INT8_C( 115), -INT8_C(  49), -INT8_C(  29),  INT8_C(   0),  INT8_C(  37),      INT8_MIN, -INT8_C(  53), -INT8_C(  71),
         INT8_C(  32),  INT8_C(  69),  INT8_C( 107), -INT8_C( 120),  INT8_C( 106),  INT8_C( 118),  INT8_C(  88),  INT8_C(  12) },
      UINT16_C( 6084),
      {  INT8_C(  35), -INT8_C(  38),  INT8_C(  97),  INT8_C(  74),  INT8_C(   3),  INT8_C(  42), -INT8_C(  93),  INT8_C(  64),
        -INT8_C(  74),  INT8_C(  64), -INT8_C(  93),  INT8_C( 100),  INT8_C(  55),  INT8_C(  22),  INT8_C(  51),  INT8_C(  26) },
      {  INT8_C( 115), -INT8_C(  49),  INT8_C(  35),  INT8_C(   0),  INT8_C(  37),      INT8_MIN,  INT8_C(  35),  INT8_C(  35),
         INT8_C(  35),  INT8_C(  35),  INT8_C(  35), -INT8_C( 120),  INT8_C(  35),  INT8_C( 118),  INT8_C(  88),  INT8_C(  12) } },
    { {  INT8_C(  22),  INT8_C(  88), -INT8_C( 102), -INT8_C(  30),  INT8_C(  17), -INT8_C(  69),  INT8_C(  39),  INT8_C( 125),
         INT8_C(  67), -INT8_C( 110), -INT8_C(  13), -INT8_C( 101), -INT8_C(  98), -INT8_C(  72), -INT8_C(  78), -INT8_C(  63) },
      UINT16_C( 5010),
      {  INT8_C(  11), -INT8_C( 107),  INT8_C(  62), -INT8_C(  81), -INT8_C(  43), -INT8_C(  12), -INT8_C(  17),  INT8_C( 121),
         INT8_C(  88),  INT8_C(  38), -INT8_C( 113), -INT8_C( 116),  INT8_C(  65), -INT8_C(  90), -INT8_C(  28), -INT8_C(  37) },
      {  INT8_C(  22),  INT8_C(  11), -INT8_C( 102), -INT8_C(  30),  INT8_C(  11), -INT8_C(  69),  INT8_C(  39),  INT8_C(  11),
         INT8_C(  11),  INT8_C(  11), -INT8_C(  13), -INT8_C( 101),  INT8_C(  11), -INT8_C(  72), -INT8_C(  78), -INT8_C(  63) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi8(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_broadcastb_epi8(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_broadcastb_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i r = easysimd_mm_mask_broadcastb_epi8(src, k, a);

    easysimd_test_x86_write_i8x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_broadcastb_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t k;
    int8_t a[16];
    int8_t r[16];
  } test_vec[] = {
    { UINT16_C(18345),
      {  INT8_C(  44), -INT8_C( 105),  INT8_C(  99), -INT8_C(  33), -INT8_C(  80), -INT8_C(  51), -INT8_C(  57),  INT8_C(  84),
        -INT8_C( 102),  INT8_C( 110),  INT8_C(  65),  INT8_C( 111), -INT8_C(  73),  INT8_C(  13),  INT8_C(  81),  INT8_C(   2) },
      {  INT8_C(  44),  INT8_C(   0),  INT8_C(   0),  INT8_C(  44),  INT8_C(   0),  INT8_C(  44),  INT8_C(   0),  INT8_C(  44),
         INT8_C(  44),  INT8_C(  44),  INT8_C(  44),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  44),  INT8_C(   0) } },
    { UINT16_C(57022),
      { -INT8_C(  14), -INT8_C(  90), -INT8_C( 105), -INT8_C(  50), -INT8_C(  26),  INT8_C(  71), -INT8_C(  86),  INT8_C(  85),
        -INT8_C(  90),  INT8_C(  44),  INT8_C(  59),  INT8_C(  79),  INT8_C( 115),  INT8_C( 103), -INT8_C(  26), -INT8_C(  41) },
      {  INT8_C(   0), -INT8_C(  14), -INT8_C(  14), -INT8_C(  14), -INT8_C(  14), -INT8_C(  14),  INT8_C(   0), -INT8_C(  14),
         INT8_C(   0), -INT8_C(  14), -INT8_C(  14), -INT8_C(  14), -INT8_C(  14),  INT8_C(   0), -INT8_C(  14), -INT8_C(  14) } },
    { UINT16_C(38726),
      { -INT8_C(  92),  INT8_C(  14), -INT8_C(  21),  INT8_C(  62),  INT8_C( 124),  INT8_C(  45), -INT8_C(  83),  INT8_C(  51),
         INT8_C(  58), -INT8_C(   2),  INT8_C(  53), -INT8_C(   8), -INT8_C(  35),  INT8_C(  39), -INT8_C(  97),  INT8_C( 116) },
      {  INT8_C(   0), -INT8_C(  92), -INT8_C(  92),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  92),  INT8_C(   0),
        -INT8_C(  92), -INT8_C(  92), -INT8_C(  92),  INT8_C(   0), -INT8_C(  92),  INT8_C(   0),  INT8_C(   0), -INT8_C(  92) } },
    { UINT16_C(34293),
      { -INT8_C(  68), -INT8_C(  97), -INT8_C(  37),  INT8_C(  98), -INT8_C(  52),  INT8_C(  22), -INT8_C(  79),  INT8_C(  63),
         INT8_C( 125), -INT8_C( 105),  INT8_C(  22), -INT8_C(  61),  INT8_C(  46), -INT8_C(  69), -INT8_C(  47),  INT8_C(  26) },
      { -INT8_C(  68),  INT8_C(   0), -INT8_C(  68),  INT8_C(   0), -INT8_C(  68), -INT8_C(  68), -INT8_C(  68), -INT8_C(  68),
        -INT8_C(  68),  INT8_C(   0), -INT8_C(  68),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  68) } },
    { UINT16_C(19961),
      {  INT8_C(  71), -INT8_C(  89),      INT8_MIN, -INT8_C( 127), -INT8_C(  91), -INT8_C(  75),  INT8_C( 121), -INT8_C( 126),
        -INT8_C(  36),  INT8_C(  24), -INT8_C(   9), -INT8_C(  47), -INT8_C(  98), -INT8_C(  77),  INT8_C( 113),  INT8_C( 121) },
      {  INT8_C(  71),  INT8_C(   0),  INT8_C(   0),  INT8_C(  71),  INT8_C(  71),  INT8_C(  71),  INT8_C(  71),  INT8_C(  71),
         INT8_C(  71),  INT8_C(   0),  INT8_C(  71),  INT8_C(  71),  INT8_C(   0),  INT8_C(   0),  INT8_C(  71),  INT8_C(   0) } },
    { UINT16_C(15637),
      { -INT8_C( 113), -INT8_C(  58),  INT8_C( 124),  INT8_C(  12),  INT8_C(  93), -INT8_C( 109), -INT8_C(  49), -INT8_C( 116),
         INT8_C(  78), -INT8_C(  95), -INT8_C(  90),  INT8_C(  71), -INT8_C(  18), -INT8_C(  19), -INT8_C(  18),  INT8_C( 111) },
      { -INT8_C( 113),  INT8_C(   0), -INT8_C( 113),  INT8_C(   0), -INT8_C( 113),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C( 113),  INT8_C(   0), -INT8_C( 113), -INT8_C( 113), -INT8_C( 113), -INT8_C( 113),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(37998),
      {  INT8_C(  36), -INT8_C(  25),  INT8_C(  22),  INT8_C(   1),  INT8_C(   0),  INT8_C(  13), -INT8_C(  46), -INT8_C(  98),
        -INT8_C(  64),  INT8_C(  67),  INT8_C(  23), -INT8_C(  43),      INT8_MIN, -INT8_C(  90), -INT8_C( 101), -INT8_C(   3) },
      {  INT8_C(   0),  INT8_C(  36),  INT8_C(  36),  INT8_C(  36),  INT8_C(   0),  INT8_C(  36),  INT8_C(  36),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  36),  INT8_C(   0),  INT8_C(  36),  INT8_C(   0),  INT8_C(   0),  INT8_C(  36) } },
    { UINT16_C(63922),
      { -INT8_C( 112), -INT8_C( 127), -INT8_C( 123), -INT8_C(  34),  INT8_C(  34),  INT8_C(  43),  INT8_C(  37),  INT8_C(  17),
         INT8_C(  24),  INT8_C(  20),      INT8_MIN, -INT8_C( 122), -INT8_C(  88), -INT8_C(  92),  INT8_C( 109), -INT8_C(  66) },
      {  INT8_C(   0), -INT8_C( 112),  INT8_C(   0),  INT8_C(   0), -INT8_C( 112), -INT8_C( 112),  INT8_C(   0), -INT8_C( 112),
        -INT8_C( 112),  INT8_C(   0),  INT8_C(   0), -INT8_C( 112), -INT8_C( 112), -INT8_C( 112), -INT8_C( 112), -INT8_C( 112) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_broadcastb_epi8(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_broadcastb_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i r = easysimd_mm_maskz_broadcastb_epi8(k, a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_broadcastw_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t src[8];
    uint8_t k;
    int16_t a[8];
    int16_t r[8];
  } test_vec[] = {
    { { -INT16_C(  2424), -INT16_C( 20586), -INT16_C(  9613),  INT16_C( 26177), -INT16_C(  8075),  INT16_C( 10270), -INT16_C( 20319), -INT16_C( 21189) },
      UINT8_C( 70),
      {  INT16_C( 23673),  INT16_C( 27931), -INT16_C( 27573),  INT16_C( 29126),  INT16_C( 21028), -INT16_C( 13646), -INT16_C( 29130),  INT16_C( 11346) },
      { -INT16_C(  2424),  INT16_C( 23673),  INT16_C( 23673),  INT16_C( 26177), -INT16_C(  8075),  INT16_C( 10270),  INT16_C( 23673), -INT16_C( 21189) } },
    { {  INT16_C(   292), -INT16_C(   353),  INT16_C(  1603),  INT16_C(  9076), -INT16_C( 25564), -INT16_C( 10812),  INT16_C( 29143),  INT16_C( 20763) },
      UINT8_C(205),
      { -INT16_C( 16842), -INT16_C( 13544), -INT16_C( 30076), -INT16_C( 10513), -INT16_C( 18116), -INT16_C( 13811),  INT16_C( 14603),  INT16_C(  3311) },
      { -INT16_C( 16842), -INT16_C(   353), -INT16_C( 16842), -INT16_C( 16842), -INT16_C( 25564), -INT16_C( 10812), -INT16_C( 16842), -INT16_C( 16842) } },
    { { -INT16_C(  4647), -INT16_C(  8369),  INT16_C( 29281), -INT16_C(   765), -INT16_C( 10185), -INT16_C( 22315),  INT16_C(  9971),  INT16_C( 10870) },
      UINT8_C(228),
      { -INT16_C(  2674),  INT16_C(  6249),  INT16_C( 16356), -INT16_C( 25259),  INT16_C(  8012), -INT16_C( 31064), -INT16_C( 19442), -INT16_C(   929) },
      { -INT16_C(  4647), -INT16_C(  8369), -INT16_C(  2674), -INT16_C(   765), -INT16_C( 10185), -INT16_C(  2674), -INT16_C(  2674), -INT16_C(  2674) } },
    { {  INT16_C( 15876),  INT16_C( 30301),  INT16_C( 23361),  INT16_C(  6829),  INT16_C( 22064),  INT16_C( 22029),  INT16_C( 14284),  INT16_C( 23098) },
      UINT8_C( 44),
      {  INT16_C( 29603), -INT16_C(  7408), -INT16_C( 21048), -INT16_C(  6353), -INT16_C( 19115),  INT16_C(  2806), -INT16_C(  3564),  INT16_C( 21006) },
      {  INT16_C( 15876),  INT16_C( 30301),  INT16_C( 29603),  INT16_C( 29603),  INT16_C( 22064),  INT16_C( 29603),  INT16_C( 14284),  INT16_C( 23098) } },
    { { -INT16_C( 31665), -INT16_C( 21868), -INT16_C( 20942), -INT16_C( 30502),  INT16_C( 12475), -INT16_C(  3244), -INT16_C( 20885),  INT16_C(  3615) },
      UINT8_C( 33),
      { -INT16_C(  3792), -INT16_C(  8727), -INT16_C( 11999), -INT16_C( 10701),  INT16_C( 15815), -INT16_C( 17941),  INT16_C( 15691), -INT16_C( 12536) },
      { -INT16_C(  3792), -INT16_C( 21868), -INT16_C( 20942), -INT16_C( 30502),  INT16_C( 12475), -INT16_C(  3792), -INT16_C( 20885),  INT16_C(  3615) } },
    { { -INT16_C( 19503),  INT16_C( 32513), -INT16_C( 30323), -INT16_C( 16837),  INT16_C( 11997), -INT16_C( 29655),  INT16_C( 14157),  INT16_C( 32173) },
      UINT8_C( 41),
      {  INT16_C( 23447),  INT16_C( 26698),  INT16_C(  8334), -INT16_C( 13521), -INT16_C(  6133),  INT16_C( 18710), -INT16_C(  6672), -INT16_C( 23782) },
      {  INT16_C( 23447),  INT16_C( 32513), -INT16_C( 30323),  INT16_C( 23447),  INT16_C( 11997),  INT16_C( 23447),  INT16_C( 14157),  INT16_C( 32173) } },
    { { -INT16_C( 25881),  INT16_C( 28721), -INT16_C(  4139),  INT16_C(   846), -INT16_C(  9704),  INT16_C( 20304), -INT16_C( 12665),  INT16_C(  7800) },
      UINT8_C( 41),
      { -INT16_C( 31038), -INT16_C(  7241), -INT16_C( 32075), -INT16_C( 25106),  INT16_C( 14232),  INT16_C( 32142),  INT16_C( 12626), -INT16_C(  5020) },
      { -INT16_C( 31038),  INT16_C( 28721), -INT16_C(  4139), -INT16_C( 31038), -INT16_C(  9704), -INT16_C( 31038), -INT16_C( 12665),  INT16_C(  7800) } },
    { { -INT16_C( 10910),  INT16_C( 20929), -INT16_C( 15325), -INT16_C(   663), -INT16_C( 18156), -INT16_C(  7548), -INT16_C( 23759), -INT16_C(  3061) },
      UINT8_C( 41),
      { -INT16_C( 10302),  INT16_C( 17631),  INT16_C( 31941), -INT16_C(   548),  INT16_C( 23050),  INT16_C( 15439),  INT16_C( 15294), -INT16_C( 27746) },
      { -INT16_C( 10302),  INT16_C( 20929), -INT16_C( 15325), -INT16_C( 10302), -INT16_C( 18156), -INT16_C( 10302), -INT16_C( 23759), -INT16_C(  3061) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_broadcastw_epi16(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_broadcastw_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_mask_broadcastw_epi16(src, k, a);

    easysimd_test_x86_write_i16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_broadcastw_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int16_t a[8];
    int16_t r[8];
  } test_vec[] = {
    { UINT8_C(165),
      { -INT16_C( 13203),  INT16_C(  2936), -INT16_C( 17524),  INT16_C( 25122), -INT16_C( 14276),  INT16_C( 14845), -INT16_C(  2438), -INT16_C(   823) },
      { -INT16_C( 13203),  INT16_C(     0), -INT16_C( 13203),  INT16_C(     0),  INT16_C(     0), -INT16_C( 13203),  INT16_C(     0), -INT16_C( 13203) } },
    { UINT8_C(123),
      {  INT16_C(  7847), -INT16_C( 13146), -INT16_C( 16849), -INT16_C( 20512), -INT16_C( 30652), -INT16_C( 19884), -INT16_C(  1721),  INT16_C(  4895) },
      {  INT16_C(  7847),  INT16_C(  7847),  INT16_C(     0),  INT16_C(  7847),  INT16_C(  7847),  INT16_C(  7847),  INT16_C(  7847),  INT16_C(     0) } },
    { UINT8_C(113),
      { -INT16_C( 24789),  INT16_C( 19757),  INT16_C( 26881), -INT16_C(   234), -INT16_C( 28510),  INT16_C( 27637),  INT16_C( 29068), -INT16_C( 21742) },
      { -INT16_C( 24789),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 24789), -INT16_C( 24789), -INT16_C( 24789),  INT16_C(     0) } },
    { UINT8_C( 23),
      { -INT16_C(  9506), -INT16_C( 16426),  INT16_C(  6794), -INT16_C(  8633), -INT16_C( 28980), -INT16_C(  4905),  INT16_C( 18849),  INT16_C( 16663) },
      { -INT16_C(  9506), -INT16_C(  9506), -INT16_C(  9506),  INT16_C(     0), -INT16_C(  9506),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(118),
      {  INT16_C( 16996),  INT16_C( 31455), -INT16_C( 32447),  INT16_C( 14091), -INT16_C( 26644), -INT16_C(   344), -INT16_C( 16574),  INT16_C(  7644) },
      {  INT16_C(     0),  INT16_C( 16996),  INT16_C( 16996),  INT16_C(     0),  INT16_C( 16996),  INT16_C( 16996),  INT16_C( 16996),  INT16_C(     0) } },
    { UINT8_C(149),
      { -INT16_C( 22629), -INT16_C(  7248),  INT16_C( 31877),  INT16_C( 23665),  INT16_C(  4968),  INT16_C( 32677),  INT16_C(  6996), -INT16_C( 26908) },
      { -INT16_C( 22629),  INT16_C(     0), -INT16_C( 22629),  INT16_C(     0), -INT16_C( 22629),  INT16_C(     0),  INT16_C(     0), -INT16_C( 22629) } },
    { UINT8_C(250),
      { -INT16_C( 10146),  INT16_C( 27003),  INT16_C( 26383), -INT16_C( 18687),  INT16_C( 17253),  INT16_C( 17014),  INT16_C(  3168),  INT16_C(  2013) },
      {  INT16_C(     0), -INT16_C( 10146),  INT16_C(     0), -INT16_C( 10146), -INT16_C( 10146), -INT16_C( 10146), -INT16_C( 10146), -INT16_C( 10146) } },
    { UINT8_C(188),
      { -INT16_C( 29504),  INT16_C( 12856), -INT16_C( 24087), -INT16_C( 29115), -INT16_C( 26336),  INT16_C(  1194), -INT16_C( 23505),  INT16_C(  1891) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 29504), -INT16_C( 29504), -INT16_C( 29504), -INT16_C( 29504),  INT16_C(     0), -INT16_C( 29504) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_broadcastw_epi16(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_broadcastw_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_maskz_broadcastw_epi16(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_broadcastd_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[4];
    uint8_t k;
    int32_t a[4];
    int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(  1061752580),  INT32_C(   315929433), -INT32_C(   616253640), -INT32_C(  2063320894) },
      UINT8_C( 15),
      {  INT32_C(  1624558051),  INT32_C(     7000486),  INT32_C(  1539221024),  INT32_C(   894915141) },
      {  INT32_C(  1624558051),  INT32_C(  1624558051),  INT32_C(  1624558051),  INT32_C(  1624558051) } },
    { { -INT32_C(  1131538680), -INT32_C(  1544248852),  INT32_C(   493211621), -INT32_C(  1221793069) },
      UINT8_C(180),
      { -INT32_C(   765847808),  INT32_C(   686971521),  INT32_C(  1785548312), -INT32_C(  1116495195) },
      { -INT32_C(  1131538680), -INT32_C(  1544248852), -INT32_C(   765847808), -INT32_C(  1221793069) } },
    { { -INT32_C(   777441488), -INT32_C(   222933981), -INT32_C(  1664756815),  INT32_C(     5274879) },
      UINT8_C(147),
      {  INT32_C(    68539050),  INT32_C(   303840708),  INT32_C(  1287161770),  INT32_C(   696022522) },
      {  INT32_C(    68539050),  INT32_C(    68539050), -INT32_C(  1664756815),  INT32_C(     5274879) } },
    { {  INT32_C(  1783385630), -INT32_C(   669303292), -INT32_C(  2133346557), -INT32_C(  1324099833) },
      UINT8_C(169),
      {  INT32_C(  1701754152),  INT32_C(  1494188242), -INT32_C(  1387046088),  INT32_C(   650869976) },
      {  INT32_C(  1701754152), -INT32_C(   669303292), -INT32_C(  2133346557),  INT32_C(  1701754152) } },
    { {  INT32_C(   103429576),  INT32_C(   134808145), -INT32_C(  1324316198), -INT32_C(   983842403) },
      UINT8_C(119),
      {  INT32_C(  1229532105), -INT32_C(  1769823686),  INT32_C(  1903046645),  INT32_C(   809080059) },
      {  INT32_C(  1229532105),  INT32_C(  1229532105),  INT32_C(  1229532105), -INT32_C(   983842403) } },
    { { -INT32_C(  1048494146), -INT32_C(   761558456),  INT32_C(  1534020762),  INT32_C(  1892824231) },
      UINT8_C( 95),
      { -INT32_C(  1113933285),  INT32_C(  1806839868),  INT32_C(   845554590), -INT32_C(  1678731428) },
      { -INT32_C(  1113933285), -INT32_C(  1113933285), -INT32_C(  1113933285), -INT32_C(  1113933285) } },
    { { -INT32_C(  1562070760), -INT32_C(  1724074420),  INT32_C(  1497405477), -INT32_C(  2034650774) },
      UINT8_C(107),
      { -INT32_C(  2086190253),  INT32_C(   421597942),  INT32_C(   276190073), -INT32_C(   181923517) },
      { -INT32_C(  2086190253), -INT32_C(  2086190253),  INT32_C(  1497405477), -INT32_C(  2086190253) } },
    { { -INT32_C(  1421751307), -INT32_C(  1613702649), -INT32_C(   871749093),  INT32_C(   909611235) },
      UINT8_C(211),
      { -INT32_C(   255215138),  INT32_C(   761914330),  INT32_C(  1785755993),  INT32_C(  1835034018) },
      { -INT32_C(   255215138), -INT32_C(   255215138), -INT32_C(   871749093),  INT32_C(   909611235) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_broadcastd_epi32(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_broadcastd_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_broadcastd_epi32(src, k, a);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_broadcastd_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[4];
    int32_t r[4];
  } test_vec[] = {
    { UINT8_C( 32),
      { -INT32_C(   846784820),  INT32_C(  1142025677),  INT32_C(   206598447),  INT32_C(    97324153) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 68),
      {  INT32_C(  1155919615),  INT32_C(   668796541),  INT32_C(  1842023434),  INT32_C(   708504340) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1155919615),  INT32_C(           0) } },
    { UINT8_C(115),
      {  INT32_C(   409008135), -INT32_C(  1937076420),  INT32_C(  1754792859), -INT32_C(   144188408) },
      {  INT32_C(   409008135),  INT32_C(   409008135),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(194),
      { -INT32_C(  2000128853),  INT32_C(  1721094811),  INT32_C(  2052237632), -INT32_C(   880687661) },
      {  INT32_C(           0), -INT32_C(  2000128853),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 37),
      {  INT32_C(   615778202),  INT32_C(   723996563),  INT32_C(   520630200), -INT32_C(  1697920474) },
      {  INT32_C(   615778202),  INT32_C(           0),  INT32_C(   615778202),  INT32_C(           0) } },
    { UINT8_C(146),
      { -INT32_C(   396085933), -INT32_C(   292445028), -INT32_C(  1598855906),  INT32_C(   943380528) },
      {  INT32_C(           0), -INT32_C(   396085933),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(140),
      { -INT32_C(  2049193122), -INT32_C(     4942857),  INT32_C(  2127158195), -INT32_C(  1412342923) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  2049193122), -INT32_C(  2049193122) } },
    { UINT8_C(191),
      {  INT32_C(  1264863162), -INT32_C(   391085515),  INT32_C(  1556144418), -INT32_C(   457552871) },
      {  INT32_C(  1264863162),  INT32_C(  1264863162),  INT32_C(  1264863162),  INT32_C(  1264863162) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_broadcastd_epi32(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_broadcastd_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_broadcastd_epi32(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_broadcastq_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[2];
    uint8_t k;
    int64_t a[2];
    int64_t r[2];
  } test_vec[] = {
    { {  INT64_C(  476278597891656615),  INT64_C( 9043544297431459870) },
      UINT8_C(216),
      {  INT64_C( 5874910354222509643),  INT64_C( 6727357847445958991) },
      {  INT64_C(  476278597891656615),  INT64_C( 9043544297431459870) } },
    { { -INT64_C( 2159969174162907692), -INT64_C( 9142308109945752536) },
      UINT8_C(109),
      {  INT64_C( 2882187448170164177), -INT64_C( 1243602553959500663) },
      {  INT64_C( 2882187448170164177), -INT64_C( 9142308109945752536) } },
    { {  INT64_C( 7022882772841178639), -INT64_C( 5964708894871142964) },
      UINT8_C(199),
      {  INT64_C( 1269210647896087934), -INT64_C( 1548658535198259555) },
      {  INT64_C( 1269210647896087934),  INT64_C( 1269210647896087934) } },
    { {  INT64_C(  256414112091132023),  INT64_C( 1327872035226868366) },
      UINT8_C(233),
      {  INT64_C( 5416800953548810090), -INT64_C(  864839414204882824) },
      {  INT64_C( 5416800953548810090),  INT64_C( 1327872035226868366) } },
    { { -INT64_C( 6154945901804011249),  INT64_C( 1482337016489065500) },
      UINT8_C(120),
      {  INT64_C( 8045830058217874392), -INT64_C( 1035962949953245162) },
      { -INT64_C( 6154945901804011249),  INT64_C( 1482337016489065500) } },
    { {  INT64_C( 8581867732805461694),  INT64_C( 3238991189673885994) },
      UINT8_C(216),
      {  INT64_C( 3732664970893548335), -INT64_C( 6706892302974755415) },
      {  INT64_C( 8581867732805461694),  INT64_C( 3238991189673885994) } },
    { { -INT64_C( 4491684217698260843),  INT64_C( 5225090467373524342) },
      UINT8_C(156),
      {  INT64_C( 3020459374333132586), -INT64_C( 1360149735836149706) },
      { -INT64_C( 4491684217698260843),  INT64_C( 5225090467373524342) } },
    { {  INT64_C( 8030109772662284197), -INT64_C( 3990241367744308286) },
      UINT8_C(207),
      {  INT64_C(  956951097897585214), -INT64_C( 8045680800029290006) },
      {  INT64_C(  956951097897585214),  INT64_C(  956951097897585214) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_broadcastq_epi64(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_broadcastq_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_mask_broadcastq_epi64(src, k, a);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_broadcastq_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int64_t a[2];
    int64_t r[2];
  } test_vec[] = {
    { UINT8_C( 39),
      { -INT64_C( 6642969038660314305),  INT64_C( 3924815537170695404) },
      { -INT64_C( 6642969038660314305), -INT64_C( 6642969038660314305) } },
    { UINT8_C( 34),
      {  INT64_C( 1392957346000694211),  INT64_C( 3250903296019229987) },
      {  INT64_C(                   0),  INT64_C( 1392957346000694211) } },
    { UINT8_C( 66),
      { -INT64_C( 6186366251100534767), -INT64_C( 5290451604013050791) },
      {  INT64_C(                   0), -INT64_C( 6186366251100534767) } },
    { UINT8_C( 46),
      {  INT64_C( 9063803338964142599),  INT64_C( 5847438445869538295) },
      {  INT64_C(                   0),  INT64_C( 9063803338964142599) } },
    { UINT8_C(246),
      {  INT64_C( 1027079813968624646),  INT64_C( 5429308434134109116) },
      {  INT64_C(                   0),  INT64_C( 1027079813968624646) } },
    { UINT8_C(188),
      { -INT64_C( 1541976107072001612),  INT64_C( 3390408095085683425) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(184),
      {  INT64_C( 9008431689057424953), -INT64_C(  379427645697014352) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(214),
      { -INT64_C(  381253686285639622), -INT64_C( 1310028420739962742) },
      {  INT64_C(                   0), -INT64_C(  381253686285639622) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_broadcastq_epi64(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_broadcastq_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_broadcastq_epi64(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_broadcastss_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd_float32 src[4];
    uint8_t k;
    easysimd_float32 a[4];
    easysimd_float32 r[4];
  } test_vec[8] = {
    { { EASYSIMD_FLOAT32_C(  -264.71), EASYSIMD_FLOAT32_C(  -516.03), EASYSIMD_FLOAT32_C(  -883.40), EASYSIMD_FLOAT32_C(  -830.68) },
      UINT8_C(  7),
      { EASYSIMD_FLOAT32_C(   439.61), EASYSIMD_FLOAT32_C(  -674.22), EASYSIMD_FLOAT32_C(   289.48), EASYSIMD_FLOAT32_C(  -894.25) },
      { EASYSIMD_FLOAT32_C(   439.61), EASYSIMD_FLOAT32_C(   439.61), EASYSIMD_FLOAT32_C(   439.61), EASYSIMD_FLOAT32_C(  -830.68) } },
    { { EASYSIMD_FLOAT32_C(  -913.93), EASYSIMD_FLOAT32_C(  -969.80), EASYSIMD_FLOAT32_C(  -237.07), EASYSIMD_FLOAT32_C(   112.23) },
      UINT8_C(204),
      { EASYSIMD_FLOAT32_C(  -898.48), EASYSIMD_FLOAT32_C(   568.09), EASYSIMD_FLOAT32_C(  -579.22), EASYSIMD_FLOAT32_C(  -595.02) },
      { EASYSIMD_FLOAT32_C(  -913.93), EASYSIMD_FLOAT32_C(  -969.80), EASYSIMD_FLOAT32_C(  -898.48), EASYSIMD_FLOAT32_C(  -898.48) } },
    { { EASYSIMD_FLOAT32_C(  -840.78), EASYSIMD_FLOAT32_C(   600.23), EASYSIMD_FLOAT32_C(   554.81), EASYSIMD_FLOAT32_C(   489.25) },
      UINT8_C( 59),
      { EASYSIMD_FLOAT32_C(  -890.05), EASYSIMD_FLOAT32_C(  -406.44), EASYSIMD_FLOAT32_C(  -168.88), EASYSIMD_FLOAT32_C(  -765.50) },
      { EASYSIMD_FLOAT32_C(  -890.05), EASYSIMD_FLOAT32_C(  -890.05), EASYSIMD_FLOAT32_C(   554.81), EASYSIMD_FLOAT32_C(  -890.05) } },
    { { EASYSIMD_FLOAT32_C(  -134.81), EASYSIMD_FLOAT32_C(   708.39), EASYSIMD_FLOAT32_C(   -19.20), EASYSIMD_FLOAT32_C(   912.59) },
      UINT8_C(234),
      { EASYSIMD_FLOAT32_C(   464.77), EASYSIMD_FLOAT32_C(  -970.81), EASYSIMD_FLOAT32_C(  -387.00), EASYSIMD_FLOAT32_C(   197.36) },
      { EASYSIMD_FLOAT32_C(  -134.81), EASYSIMD_FLOAT32_C(   464.77), EASYSIMD_FLOAT32_C(   -19.20), EASYSIMD_FLOAT32_C(   464.77) } },
    { { EASYSIMD_FLOAT32_C(   468.81), EASYSIMD_FLOAT32_C(   -61.22), EASYSIMD_FLOAT32_C(  -513.16), EASYSIMD_FLOAT32_C(   574.56) },
      UINT8_C( 57),
      { EASYSIMD_FLOAT32_C(  -482.96), EASYSIMD_FLOAT32_C(  -662.51), EASYSIMD_FLOAT32_C(  -862.92), EASYSIMD_FLOAT32_C(  -893.60) },
      { EASYSIMD_FLOAT32_C(  -482.96), EASYSIMD_FLOAT32_C(   -61.22), EASYSIMD_FLOAT32_C(  -513.16), EASYSIMD_FLOAT32_C(  -482.96) } },
    { { EASYSIMD_FLOAT32_C(  -560.99), EASYSIMD_FLOAT32_C(   705.17), EASYSIMD_FLOAT32_C(  -472.82), EASYSIMD_FLOAT32_C(  -156.01) },
      UINT8_C( 85),
      { EASYSIMD_FLOAT32_C(  -872.59), EASYSIMD_FLOAT32_C(  -601.20), EASYSIMD_FLOAT32_C(   353.65), EASYSIMD_FLOAT32_C(  -261.54) },
      { EASYSIMD_FLOAT32_C(  -872.59), EASYSIMD_FLOAT32_C(   705.17), EASYSIMD_FLOAT32_C(  -872.59), EASYSIMD_FLOAT32_C(  -156.01) } },
    { { EASYSIMD_FLOAT32_C(  -491.25), EASYSIMD_FLOAT32_C(   947.21), EASYSIMD_FLOAT32_C(   569.59), EASYSIMD_FLOAT32_C(  -256.74) },
      UINT8_C(128),
      { EASYSIMD_FLOAT32_C(   277.97), EASYSIMD_FLOAT32_C(   724.06), EASYSIMD_FLOAT32_C(  -275.00), EASYSIMD_FLOAT32_C(   721.65) },
      { EASYSIMD_FLOAT32_C(  -491.25), EASYSIMD_FLOAT32_C(   947.21), EASYSIMD_FLOAT32_C(   569.59), EASYSIMD_FLOAT32_C(  -256.74) } },
    { { EASYSIMD_FLOAT32_C(   188.83), EASYSIMD_FLOAT32_C(  -245.81), EASYSIMD_FLOAT32_C(  -665.35), EASYSIMD_FLOAT32_C(  -613.81) },
      UINT8_C( 89),
      { EASYSIMD_FLOAT32_C(   273.43), EASYSIMD_FLOAT32_C(  -126.97), EASYSIMD_FLOAT32_C(   797.56), EASYSIMD_FLOAT32_C(  -701.72) },
      { EASYSIMD_FLOAT32_C(   273.43), EASYSIMD_FLOAT32_C(  -245.81), EASYSIMD_FLOAT32_C(  -665.35), EASYSIMD_FLOAT32_C(   273.43) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_broadcastss_ps(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_broadcastss_ps");
    easysimd_assert_m128_close(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_broadcastss_ps(src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_broadcast_i32x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[4];
    uint8_t k;
    int32_t a[4];
    int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(  1844158754),  INT32_C(   662299972),  INT32_C(  2092350862), -INT32_C(   644806509) },
      UINT8_C(207),
      { -INT32_C(  1047905698), -INT32_C(   922054928),  INT32_C(   223335737), -INT32_C(   533776065) },
      { -INT32_C(  1047905698), -INT32_C(   922054928), -INT32_C(  1047905698), -INT32_C(   922054928) } },
    { {  INT32_C(   153394212), -INT32_C(   745059563),  INT32_C(   174461953),  INT32_C(    64569253) },
      UINT8_C(121),
      { -INT32_C(   244661149), -INT32_C(  1557515314), -INT32_C(  1142802558),  INT32_C(    65061478) },
      { -INT32_C(   244661149), -INT32_C(   745059563),  INT32_C(   174461953), -INT32_C(  1557515314) } },
    { {  INT32_C(   840493543), -INT32_C(  1791693951), -INT32_C(  1858453935),  INT32_C(  2047491351) },
      UINT8_C(  2),
      { -INT32_C(  1479513228), -INT32_C(   869632875),  INT32_C(   422765910), -INT32_C(  1358940731) },
      {  INT32_C(   840493543), -INT32_C(   869632875), -INT32_C(  1858453935),  INT32_C(  2047491351) } },
    { {  INT32_C(   959459918), -INT32_C(  1534343834),  INT32_C(  1018895615), -INT32_C(  1690421978) },
      UINT8_C(160),
      { -INT32_C(  2093661681),  INT32_C(  1389953388),  INT32_C(  1763177012),  INT32_C(   632801010) },
      {  INT32_C(   959459918), -INT32_C(  1534343834),  INT32_C(  1018895615), -INT32_C(  1690421978) } },
    { { -INT32_C(  1148456458), -INT32_C(  1732628356),  INT32_C(   566163435),  INT32_C(  1153521973) },
      UINT8_C(156),
      { -INT32_C(   133642249), -INT32_C(  1825809759),  INT32_C(   948344434),  INT32_C(  1043311437) },
      { -INT32_C(  1148456458), -INT32_C(  1732628356), -INT32_C(   133642249), -INT32_C(  1825809759) } },
    { {  INT32_C(  1723525686), -INT32_C(  1672326491),  INT32_C(  1792111377),  INT32_C(   721819188) },
      UINT8_C(221),
      {  INT32_C(  1769874447), -INT32_C(   421850544),  INT32_C(  1127486616),  INT32_C(   762933827) },
      {  INT32_C(  1769874447), -INT32_C(  1672326491),  INT32_C(  1769874447), -INT32_C(   421850544) } },
    { {  INT32_C(  2144526381), -INT32_C(  1517261262),  INT32_C(  1457126208),  INT32_C(   271779073) },
      UINT8_C( 41),
      { -INT32_C(   998671694),  INT32_C(  1767661653),  INT32_C(   111976340), -INT32_C(   130819560) },
      { -INT32_C(   998671694), -INT32_C(  1517261262),  INT32_C(  1457126208),  INT32_C(  1767661653) } },
    { {  INT32_C(   455783340),  INT32_C(  1046204227), -INT32_C(  1371491927), -INT32_C(  1764273948) },
      UINT8_C(202),
      { -INT32_C(  1340057008),  INT32_C(  1430555062),  INT32_C(   275663670),  INT32_C(   834430590) },
      {  INT32_C(   455783340),  INT32_C(  1430555062), -INT32_C(  1371491927),  INT32_C(  1430555062) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_broadcast_i32x2(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_broadcast_i32x2");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_broadcast_i32x2(src, k, a);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_broadcast_i32x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[4];
    int32_t r[4];
  } test_vec[] = {
    { UINT8_C( 51),
      {  INT32_C(   146101113),  INT32_C(  1719092177),  INT32_C(   891871102),  INT32_C(  1249873453) },
      {  INT32_C(   146101113),  INT32_C(  1719092177),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(194),
      {  INT32_C(   948906278),  INT32_C(    57334533), -INT32_C(   767927918), -INT32_C(   229937761) },
      {  INT32_C(           0),  INT32_C(    57334533),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 35),
      { -INT32_C(   899234989),  INT32_C(  1386865706), -INT32_C(  1730095335),  INT32_C(  1254007329) },
      { -INT32_C(   899234989),  INT32_C(  1386865706),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 50),
      {  INT32_C(  1628262647), -INT32_C(  1900371885), -INT32_C(  1107601038),  INT32_C(   252714572) },
      {  INT32_C(           0), -INT32_C(  1900371885),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(132),
      { -INT32_C(  2090255910),  INT32_C(  1801093771), -INT32_C(   653362149),  INT32_C(   382746822) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  2090255910),  INT32_C(           0) } },
    { UINT8_C( 77),
      { -INT32_C(   319985103), -INT32_C(   213557256),  INT32_C(   739414300), -INT32_C(   637102431) },
      { -INT32_C(   319985103),  INT32_C(           0), -INT32_C(   319985103), -INT32_C(   213557256) } },
    {    UINT8_MAX,
      { -INT32_C(   461216118), -INT32_C(   547316271),  INT32_C(  1193224054), -INT32_C(  1485279939) },
      { -INT32_C(   461216118), -INT32_C(   547316271), -INT32_C(   461216118), -INT32_C(   547316271) } },
    { UINT8_C( 90),
      { -INT32_C(  1430610076), -INT32_C(  1522805101),  INT32_C(   154917891),  INT32_C(   496188343) },
      {  INT32_C(           0), -INT32_C(  1522805101),  INT32_C(           0), -INT32_C(  1522805101) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_broadcast_i32x2(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_broadcast_i32x2");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_broadcast_i32x2(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_broadcast_i32x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[4];
    int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(   575318251),  INT32_C(  1860327497), -INT32_C(  1028870877),  INT32_C(   155780813) },
      { -INT32_C(   575318251),  INT32_C(  1860327497), -INT32_C(   575318251),  INT32_C(  1860327497), -INT32_C(   575318251),  INT32_C(  1860327497), -INT32_C(   575318251),  INT32_C(  1860327497) } },
    { { -INT32_C(  1136402042),  INT32_C(  1838219724), -INT32_C(  1242567846), -INT32_C(  1245378913) },
      { -INT32_C(  1136402042),  INT32_C(  1838219724), -INT32_C(  1136402042),  INT32_C(  1838219724), -INT32_C(  1136402042),  INT32_C(  1838219724), -INT32_C(  1136402042),  INT32_C(  1838219724) } },
    { { -INT32_C(  1500349603), -INT32_C(   149654061), -INT32_C(   273039326),  INT32_C(  1308099270) },
      { -INT32_C(  1500349603), -INT32_C(   149654061), -INT32_C(  1500349603), -INT32_C(   149654061), -INT32_C(  1500349603), -INT32_C(   149654061), -INT32_C(  1500349603), -INT32_C(   149654061) } },
    { { -INT32_C(  1341572124), -INT32_C(  1759602115),  INT32_C(   625741190),  INT32_C(  1893339411) },
      { -INT32_C(  1341572124), -INT32_C(  1759602115), -INT32_C(  1341572124), -INT32_C(  1759602115), -INT32_C(  1341572124), -INT32_C(  1759602115), -INT32_C(  1341572124), -INT32_C(  1759602115) } },
    { {  INT32_C(  1612082572),  INT32_C(    72821474), -INT32_C(  1309470485), -INT32_C(   151065838) },
      {  INT32_C(  1612082572),  INT32_C(    72821474),  INT32_C(  1612082572),  INT32_C(    72821474),  INT32_C(  1612082572),  INT32_C(    72821474),  INT32_C(  1612082572),  INT32_C(    72821474) } },
    { {  INT32_C(  1688667943),  INT32_C(   687588770), -INT32_C(   447920174), -INT32_C(   447403944) },
      {  INT32_C(  1688667943),  INT32_C(   687588770),  INT32_C(  1688667943),  INT32_C(   687588770),  INT32_C(  1688667943),  INT32_C(   687588770),  INT32_C(  1688667943),  INT32_C(   687588770) } },
    { {  INT32_C(  2001038229), -INT32_C(  2139382635), -INT32_C(  1103991124), -INT32_C(  2135609255) },
      {  INT32_C(  2001038229), -INT32_C(  2139382635),  INT32_C(  2001038229), -INT32_C(  2139382635),  INT32_C(  2001038229), -INT32_C(  2139382635),  INT32_C(  2001038229), -INT32_C(  2139382635) } },
    { { -INT32_C(   622568392), -INT32_C(   217915615),  INT32_C(  2144882470),  INT32_C(   207891831) },
      { -INT32_C(   622568392), -INT32_C(   217915615), -INT32_C(   622568392), -INT32_C(   217915615), -INT32_C(   622568392), -INT32_C(   217915615), -INT32_C(   622568392), -INT32_C(   217915615) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_broadcast_i32x2(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_broadcast_i32x2");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m256i r = easysimd_mm256_broadcast_i32x2(a);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_broadcast_i32x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[8];
    uint8_t k;
    int32_t a[4];
    int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1892041890),  INT32_C(  1997810994),  INT32_C(  1201585791), -INT32_C(   891905703), -INT32_C(  1672309924), -INT32_C(  1321977855),  INT32_C(  1984388552), -INT32_C(  1849429197) },
      UINT8_C(174),
      {  INT32_C(   719331836), -INT32_C(   190228683), -INT32_C(  1890717451),  INT32_C(  1525356487) },
      { -INT32_C(  1892041890), -INT32_C(   190228683),  INT32_C(   719331836), -INT32_C(   190228683), -INT32_C(  1672309924), -INT32_C(   190228683),  INT32_C(  1984388552), -INT32_C(   190228683) } },
    { { -INT32_C(  1571059862),  INT32_C(   426380475),  INT32_C(  1179443283), -INT32_C(  1594565212),  INT32_C(   885708286),  INT32_C(   573076268),  INT32_C(   733050212), -INT32_C(   142238579) },
      UINT8_C( 36),
      { -INT32_C(   320890400), -INT32_C(   465569788),  INT32_C(   579372612),  INT32_C(  1327507834) },
      { -INT32_C(  1571059862),  INT32_C(   426380475), -INT32_C(   320890400), -INT32_C(  1594565212),  INT32_C(   885708286), -INT32_C(   465569788),  INT32_C(   733050212), -INT32_C(   142238579) } },
    { {  INT32_C(  1736201459), -INT32_C(   221536644), -INT32_C(   343935409),  INT32_C(  1527740027),  INT32_C(   323481359),  INT32_C(   754484967), -INT32_C(  2024898548), -INT32_C(  1646891351) },
      UINT8_C(195),
      { -INT32_C(   264305582), -INT32_C(   985714225),  INT32_C(   641739696),  INT32_C(   691444538) },
      { -INT32_C(   264305582), -INT32_C(   985714225), -INT32_C(   343935409),  INT32_C(  1527740027),  INT32_C(   323481359),  INT32_C(   754484967), -INT32_C(   264305582), -INT32_C(   985714225) } },
    { {  INT32_C(  1745963490), -INT32_C(  1032503999), -INT32_C(   110363509),  INT32_C(   633080018), -INT32_C(   619316212), -INT32_C(   559917779), -INT32_C(  1174085504),  INT32_C(  1575238267) },
      UINT8_C(132),
      {  INT32_C(   851822325),  INT32_C(   935167803), -INT32_C(    83249421),  INT32_C(  1862807155) },
      {  INT32_C(  1745963490), -INT32_C(  1032503999),  INT32_C(   851822325),  INT32_C(   633080018), -INT32_C(   619316212), -INT32_C(   559917779), -INT32_C(  1174085504),  INT32_C(   935167803) } },
    { { -INT32_C(  1717771452),  INT32_C(  1679391364), -INT32_C(  1176448129), -INT32_C(  1405272649),  INT32_C(  1054737155),  INT32_C(  2104859530), -INT32_C(   981893294), -INT32_C(   248217171) },
      UINT8_C(100),
      {  INT32_C(  1273531088),  INT32_C(  2009746851),  INT32_C(  1781433133), -INT32_C(   999367743) },
      { -INT32_C(  1717771452),  INT32_C(  1679391364),  INT32_C(  1273531088), -INT32_C(  1405272649),  INT32_C(  1054737155),  INT32_C(  2009746851),  INT32_C(  1273531088), -INT32_C(   248217171) } },
    { {  INT32_C(  1431219385), -INT32_C(  1582838750), -INT32_C(   967939003),  INT32_C(  1881817248),  INT32_C(  1857754058), -INT32_C(  1914337952), -INT32_C(   889777399), -INT32_C(  1467062802) },
      UINT8_C( 18),
      { -INT32_C(  1472922148),  INT32_C(   284022180),  INT32_C(  1672524579), -INT32_C(   248635170) },
      {  INT32_C(  1431219385),  INT32_C(   284022180), -INT32_C(   967939003),  INT32_C(  1881817248), -INT32_C(  1472922148), -INT32_C(  1914337952), -INT32_C(   889777399), -INT32_C(  1467062802) } },
    { {  INT32_C(  1632738524), -INT32_C(  1804935551),  INT32_C(   998454485), -INT32_C(  1622332478), -INT32_C(   867729112),  INT32_C(  2044474710), -INT32_C(   958559000), -INT32_C(  2001269844) },
      UINT8_C(167),
      { -INT32_C(   433526264), -INT32_C(  1984119724),  INT32_C(  1783363391), -INT32_C(   980227516) },
      { -INT32_C(   433526264), -INT32_C(  1984119724), -INT32_C(   433526264), -INT32_C(  1622332478), -INT32_C(   867729112), -INT32_C(  1984119724), -INT32_C(   958559000), -INT32_C(  1984119724) } },
    { {  INT32_C(  1729846834), -INT32_C(   967863238),  INT32_C(  2087917169), -INT32_C(   702284851),  INT32_C(   985418725),  INT32_C(  1203992584), -INT32_C(  1280176529),  INT32_C(   729302265) },
      UINT8_C(163),
      {  INT32_C(   685609619), -INT32_C(   140925727),  INT32_C(   314906135), -INT32_C(  2047304903) },
      {  INT32_C(   685609619), -INT32_C(   140925727),  INT32_C(  2087917169), -INT32_C(   702284851),  INT32_C(   985418725), -INT32_C(   140925727), -INT32_C(  1280176529), -INT32_C(   140925727) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_broadcast_i32x2(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_broadcast_i32x2");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m256i r = easysimd_mm256_mask_broadcast_i32x2(src, k, a);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_broadcast_i32x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[4];
    int32_t r[8];
  } test_vec[] = {
    { UINT8_C( 35),
      { -INT32_C(   637683677),  INT32_C(    65504755), -INT32_C(  1670289983), -INT32_C(    52375130) },
      { -INT32_C(   637683677),  INT32_C(    65504755),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(    65504755),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(163),
      {  INT32_C(   497086503), -INT32_C(  1656816878),  INT32_C(  2134922217),  INT32_C(  1134715787) },
      {  INT32_C(   497086503), -INT32_C(  1656816878),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1656816878),  INT32_C(           0), -INT32_C(  1656816878) } },
    { UINT8_C( 97),
      {  INT32_C(  1676031868), -INT32_C(  1412913350),  INT32_C(   612200004),  INT32_C(  1514872938) },
      {  INT32_C(  1676031868),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1412913350),  INT32_C(  1676031868),  INT32_C(           0) } },
    { UINT8_C(192),
      { -INT32_C(  1480888984),  INT32_C(  1257939978), -INT32_C(   961706716),  INT32_C(     4329161) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1480888984),  INT32_C(  1257939978) } },
    { UINT8_C(244),
      {  INT32_C(  1855732390),  INT32_C(  1658708197),  INT32_C(  1334003460),  INT32_C(   263734178) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1855732390),  INT32_C(           0),  INT32_C(  1855732390),  INT32_C(  1658708197),  INT32_C(  1855732390),  INT32_C(  1658708197) } },
    {    UINT8_MAX,
      {  INT32_C(  1503861087),  INT32_C(   283101027), -INT32_C(   803231602), -INT32_C(   495578200) },
      {  INT32_C(  1503861087),  INT32_C(   283101027),  INT32_C(  1503861087),  INT32_C(   283101027),  INT32_C(  1503861087),  INT32_C(   283101027),  INT32_C(  1503861087),  INT32_C(   283101027) } },
    { UINT8_C(175),
      { -INT32_C(  1030764571), -INT32_C(  1391881430), -INT32_C(  1678726173), -INT32_C(   738529350) },
      { -INT32_C(  1030764571), -INT32_C(  1391881430), -INT32_C(  1030764571), -INT32_C(  1391881430),  INT32_C(           0), -INT32_C(  1391881430),  INT32_C(           0), -INT32_C(  1391881430) } },
    { UINT8_C(147),
      {  INT32_C(   844772947),  INT32_C(  1725622343),  INT32_C(   796492473),  INT32_C(   739518820) },
      {  INT32_C(   844772947),  INT32_C(  1725622343),  INT32_C(           0),  INT32_C(           0),  INT32_C(   844772947),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1725622343) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_broadcast_i32x2(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_broadcast_i32x2");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m256i r = easysimd_mm256_maskz_broadcast_i32x2(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_broadcast_f32x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -751.49), EASYSIMD_FLOAT32_C(  -275.35), EASYSIMD_FLOAT32_C(   260.73), EASYSIMD_FLOAT32_C(    40.02) },
      { EASYSIMD_FLOAT32_C(  -751.49), EASYSIMD_FLOAT32_C(  -275.35), EASYSIMD_FLOAT32_C(  -751.49), EASYSIMD_FLOAT32_C(  -275.35),
        EASYSIMD_FLOAT32_C(  -751.49), EASYSIMD_FLOAT32_C(  -275.35), EASYSIMD_FLOAT32_C(  -751.49), EASYSIMD_FLOAT32_C(  -275.35) } },
    { { EASYSIMD_FLOAT32_C(   629.63), EASYSIMD_FLOAT32_C(   163.39), EASYSIMD_FLOAT32_C(   167.23), EASYSIMD_FLOAT32_C(   652.38) },
      { EASYSIMD_FLOAT32_C(   629.63), EASYSIMD_FLOAT32_C(   163.39), EASYSIMD_FLOAT32_C(   629.63), EASYSIMD_FLOAT32_C(   163.39),
        EASYSIMD_FLOAT32_C(   629.63), EASYSIMD_FLOAT32_C(   163.39), EASYSIMD_FLOAT32_C(   629.63), EASYSIMD_FLOAT32_C(   163.39) } },
    { { EASYSIMD_FLOAT32_C(   574.73), EASYSIMD_FLOAT32_C(  -529.99), EASYSIMD_FLOAT32_C(   389.79), EASYSIMD_FLOAT32_C(  -875.04) },
      { EASYSIMD_FLOAT32_C(   574.73), EASYSIMD_FLOAT32_C(  -529.99), EASYSIMD_FLOAT32_C(   574.73), EASYSIMD_FLOAT32_C(  -529.99),
        EASYSIMD_FLOAT32_C(   574.73), EASYSIMD_FLOAT32_C(  -529.99), EASYSIMD_FLOAT32_C(   574.73), EASYSIMD_FLOAT32_C(  -529.99) } },
    { { EASYSIMD_FLOAT32_C(  -790.15), EASYSIMD_FLOAT32_C(     7.90), EASYSIMD_FLOAT32_C(   834.33), EASYSIMD_FLOAT32_C(   549.92) },
      { EASYSIMD_FLOAT32_C(  -790.15), EASYSIMD_FLOAT32_C(     7.90), EASYSIMD_FLOAT32_C(  -790.15), EASYSIMD_FLOAT32_C(     7.90),
        EASYSIMD_FLOAT32_C(  -790.15), EASYSIMD_FLOAT32_C(     7.90), EASYSIMD_FLOAT32_C(  -790.15), EASYSIMD_FLOAT32_C(     7.90) } },
    { { EASYSIMD_FLOAT32_C(   494.62), EASYSIMD_FLOAT32_C(  -875.96), EASYSIMD_FLOAT32_C(  -221.96), EASYSIMD_FLOAT32_C(  -519.70) },
      { EASYSIMD_FLOAT32_C(   494.62), EASYSIMD_FLOAT32_C(  -875.96), EASYSIMD_FLOAT32_C(   494.62), EASYSIMD_FLOAT32_C(  -875.96),
        EASYSIMD_FLOAT32_C(   494.62), EASYSIMD_FLOAT32_C(  -875.96), EASYSIMD_FLOAT32_C(   494.62), EASYSIMD_FLOAT32_C(  -875.96) } },
    { { EASYSIMD_FLOAT32_C(  -583.03), EASYSIMD_FLOAT32_C(  -938.00), EASYSIMD_FLOAT32_C(   973.38), EASYSIMD_FLOAT32_C(  -468.70) },
      { EASYSIMD_FLOAT32_C(  -583.03), EASYSIMD_FLOAT32_C(  -938.00), EASYSIMD_FLOAT32_C(  -583.03), EASYSIMD_FLOAT32_C(  -938.00),
        EASYSIMD_FLOAT32_C(  -583.03), EASYSIMD_FLOAT32_C(  -938.00), EASYSIMD_FLOAT32_C(  -583.03), EASYSIMD_FLOAT32_C(  -938.00) } },
    { { EASYSIMD_FLOAT32_C(   521.04), EASYSIMD_FLOAT32_C(  -960.21), EASYSIMD_FLOAT32_C(  -215.76), EASYSIMD_FLOAT32_C(  -218.82) },
      { EASYSIMD_FLOAT32_C(   521.04), EASYSIMD_FLOAT32_C(  -960.21), EASYSIMD_FLOAT32_C(   521.04), EASYSIMD_FLOAT32_C(  -960.21),
        EASYSIMD_FLOAT32_C(   521.04), EASYSIMD_FLOAT32_C(  -960.21), EASYSIMD_FLOAT32_C(   521.04), EASYSIMD_FLOAT32_C(  -960.21) } },
    { { EASYSIMD_FLOAT32_C(   315.04), EASYSIMD_FLOAT32_C(   872.51), EASYSIMD_FLOAT32_C(   318.60), EASYSIMD_FLOAT32_C(   720.27) },
      { EASYSIMD_FLOAT32_C(   315.04), EASYSIMD_FLOAT32_C(   872.51), EASYSIMD_FLOAT32_C(   315.04), EASYSIMD_FLOAT32_C(   872.51),
        EASYSIMD_FLOAT32_C(   315.04), EASYSIMD_FLOAT32_C(   872.51), EASYSIMD_FLOAT32_C(   315.04), EASYSIMD_FLOAT32_C(   872.51) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_broadcast_f32x2(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_broadcast_f32x2");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_broadcast_f32x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 src[8];
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -155.94), EASYSIMD_FLOAT32_C(  -965.17), EASYSIMD_FLOAT32_C(   378.08), EASYSIMD_FLOAT32_C(   365.29),
        EASYSIMD_FLOAT32_C(  -495.97), EASYSIMD_FLOAT32_C(   311.10), EASYSIMD_FLOAT32_C(   575.79), EASYSIMD_FLOAT32_C(  -655.57) },
      UINT8_C( 85),
      { EASYSIMD_FLOAT32_C(   963.37), EASYSIMD_FLOAT32_C(  -596.05), EASYSIMD_FLOAT32_C(   183.95), EASYSIMD_FLOAT32_C(  -410.87) },
      { EASYSIMD_FLOAT32_C(   963.37), EASYSIMD_FLOAT32_C(  -965.17), EASYSIMD_FLOAT32_C(   963.37), EASYSIMD_FLOAT32_C(   365.29),
        EASYSIMD_FLOAT32_C(   963.37), EASYSIMD_FLOAT32_C(   311.10), EASYSIMD_FLOAT32_C(   963.37), EASYSIMD_FLOAT32_C(  -655.57) } },
    { { EASYSIMD_FLOAT32_C(   431.64), EASYSIMD_FLOAT32_C(   613.27), EASYSIMD_FLOAT32_C(  -834.97), EASYSIMD_FLOAT32_C(   711.68),
        EASYSIMD_FLOAT32_C(  -862.98), EASYSIMD_FLOAT32_C(   -74.52), EASYSIMD_FLOAT32_C(  -451.05), EASYSIMD_FLOAT32_C(  -751.41) },
      UINT8_C(193),
      { EASYSIMD_FLOAT32_C(   -39.01), EASYSIMD_FLOAT32_C(   325.90), EASYSIMD_FLOAT32_C(  -543.82), EASYSIMD_FLOAT32_C(    50.30) },
      { EASYSIMD_FLOAT32_C(   -39.01), EASYSIMD_FLOAT32_C(   613.27), EASYSIMD_FLOAT32_C(  -834.97), EASYSIMD_FLOAT32_C(   711.68),
        EASYSIMD_FLOAT32_C(  -862.98), EASYSIMD_FLOAT32_C(   -74.52), EASYSIMD_FLOAT32_C(   -39.01), EASYSIMD_FLOAT32_C(   325.90) } },
    { { EASYSIMD_FLOAT32_C(  -570.27), EASYSIMD_FLOAT32_C(  -600.03), EASYSIMD_FLOAT32_C(  -713.28), EASYSIMD_FLOAT32_C(   -16.45),
        EASYSIMD_FLOAT32_C(  -512.72), EASYSIMD_FLOAT32_C(   640.13), EASYSIMD_FLOAT32_C(   632.82), EASYSIMD_FLOAT32_C(  -156.53) },
      UINT8_C(110),
      { EASYSIMD_FLOAT32_C(   351.05), EASYSIMD_FLOAT32_C(    39.68), EASYSIMD_FLOAT32_C(   822.74), EASYSIMD_FLOAT32_C(  -140.05) },
      { EASYSIMD_FLOAT32_C(  -570.27), EASYSIMD_FLOAT32_C(    39.68), EASYSIMD_FLOAT32_C(   351.05), EASYSIMD_FLOAT32_C(    39.68),
        EASYSIMD_FLOAT32_C(  -512.72), EASYSIMD_FLOAT32_C(    39.68), EASYSIMD_FLOAT32_C(   351.05), EASYSIMD_FLOAT32_C(  -156.53) } },
    { { EASYSIMD_FLOAT32_C(   219.95), EASYSIMD_FLOAT32_C(   765.90), EASYSIMD_FLOAT32_C(   464.19), EASYSIMD_FLOAT32_C(  -363.72),
        EASYSIMD_FLOAT32_C(   978.16), EASYSIMD_FLOAT32_C(   -55.83), EASYSIMD_FLOAT32_C(  -268.61), EASYSIMD_FLOAT32_C(  -471.94) },
      UINT8_C(194),
      { EASYSIMD_FLOAT32_C(   300.83), EASYSIMD_FLOAT32_C(   122.56), EASYSIMD_FLOAT32_C(  -137.37), EASYSIMD_FLOAT32_C(  -830.55) },
      { EASYSIMD_FLOAT32_C(   219.95), EASYSIMD_FLOAT32_C(   122.56), EASYSIMD_FLOAT32_C(   464.19), EASYSIMD_FLOAT32_C(  -363.72),
        EASYSIMD_FLOAT32_C(   978.16), EASYSIMD_FLOAT32_C(   -55.83), EASYSIMD_FLOAT32_C(   300.83), EASYSIMD_FLOAT32_C(   122.56) } },
    { { EASYSIMD_FLOAT32_C(  -993.95), EASYSIMD_FLOAT32_C(   735.37), EASYSIMD_FLOAT32_C(  -715.04), EASYSIMD_FLOAT32_C(   363.48),
        EASYSIMD_FLOAT32_C(   997.38), EASYSIMD_FLOAT32_C(   957.48), EASYSIMD_FLOAT32_C(   411.04), EASYSIMD_FLOAT32_C(   318.40) },
      UINT8_C(  0),
      { EASYSIMD_FLOAT32_C(   944.29), EASYSIMD_FLOAT32_C(   688.98), EASYSIMD_FLOAT32_C(  -319.61), EASYSIMD_FLOAT32_C(   391.33) },
      { EASYSIMD_FLOAT32_C(  -993.95), EASYSIMD_FLOAT32_C(   735.37), EASYSIMD_FLOAT32_C(  -715.04), EASYSIMD_FLOAT32_C(   363.48),
        EASYSIMD_FLOAT32_C(   997.38), EASYSIMD_FLOAT32_C(   957.48), EASYSIMD_FLOAT32_C(   411.04), EASYSIMD_FLOAT32_C(   318.40) } },
    { { EASYSIMD_FLOAT32_C(  -917.62), EASYSIMD_FLOAT32_C(  -406.65), EASYSIMD_FLOAT32_C(  -532.97), EASYSIMD_FLOAT32_C(   298.17),
        EASYSIMD_FLOAT32_C(  -598.91), EASYSIMD_FLOAT32_C(   107.47), EASYSIMD_FLOAT32_C(   214.95), EASYSIMD_FLOAT32_C(   587.62) },
      UINT8_C(159),
      { EASYSIMD_FLOAT32_C(  -173.39), EASYSIMD_FLOAT32_C(  -170.67), EASYSIMD_FLOAT32_C(  -483.21), EASYSIMD_FLOAT32_C(   718.07) },
      { EASYSIMD_FLOAT32_C(  -173.39), EASYSIMD_FLOAT32_C(  -170.67), EASYSIMD_FLOAT32_C(  -173.39), EASYSIMD_FLOAT32_C(  -170.67),
        EASYSIMD_FLOAT32_C(  -173.39), EASYSIMD_FLOAT32_C(   107.47), EASYSIMD_FLOAT32_C(   214.95), EASYSIMD_FLOAT32_C(  -170.67) } },
    { { EASYSIMD_FLOAT32_C(   526.28), EASYSIMD_FLOAT32_C(  -786.80), EASYSIMD_FLOAT32_C(   286.87), EASYSIMD_FLOAT32_C(  -560.33),
        EASYSIMD_FLOAT32_C(   596.72), EASYSIMD_FLOAT32_C(   991.58), EASYSIMD_FLOAT32_C(  -572.23), EASYSIMD_FLOAT32_C(   587.29) },
      UINT8_C( 79),
      { EASYSIMD_FLOAT32_C(   221.82), EASYSIMD_FLOAT32_C(   117.18), EASYSIMD_FLOAT32_C(  -624.10), EASYSIMD_FLOAT32_C(   727.41) },
      { EASYSIMD_FLOAT32_C(   221.82), EASYSIMD_FLOAT32_C(   117.18), EASYSIMD_FLOAT32_C(   221.82), EASYSIMD_FLOAT32_C(   117.18),
        EASYSIMD_FLOAT32_C(   596.72), EASYSIMD_FLOAT32_C(   991.58), EASYSIMD_FLOAT32_C(   221.82), EASYSIMD_FLOAT32_C(   587.29) } },
    { { EASYSIMD_FLOAT32_C(  -473.57), EASYSIMD_FLOAT32_C(   647.70), EASYSIMD_FLOAT32_C(  -174.14), EASYSIMD_FLOAT32_C(  -701.99),
        EASYSIMD_FLOAT32_C(  -317.30), EASYSIMD_FLOAT32_C(  -833.25), EASYSIMD_FLOAT32_C(  -470.85), EASYSIMD_FLOAT32_C(   426.74) },
      UINT8_C(169),
      { EASYSIMD_FLOAT32_C(  -800.29), EASYSIMD_FLOAT32_C(  -506.53), EASYSIMD_FLOAT32_C(   682.63), EASYSIMD_FLOAT32_C(   942.35) },
      { EASYSIMD_FLOAT32_C(  -800.29), EASYSIMD_FLOAT32_C(   647.70), EASYSIMD_FLOAT32_C(  -174.14), EASYSIMD_FLOAT32_C(  -506.53),
        EASYSIMD_FLOAT32_C(  -317.30), EASYSIMD_FLOAT32_C(  -506.53), EASYSIMD_FLOAT32_C(  -470.85), EASYSIMD_FLOAT32_C(  -506.53) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_broadcast_f32x2(src, test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_broadcast_f32x2");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_broadcast_f32x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(167),
      { EASYSIMD_FLOAT32_C(   -73.48), EASYSIMD_FLOAT32_C(  -950.66), EASYSIMD_FLOAT32_C(   265.90), EASYSIMD_FLOAT32_C(  -988.50) },
      { EASYSIMD_FLOAT32_C(   -73.48), EASYSIMD_FLOAT32_C(  -950.66), EASYSIMD_FLOAT32_C(   -73.48), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -950.66), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -950.66) } },
    { UINT8_C(122),
      { EASYSIMD_FLOAT32_C(   490.14), EASYSIMD_FLOAT32_C(  -286.45), EASYSIMD_FLOAT32_C(  -424.27), EASYSIMD_FLOAT32_C(  -754.18) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -286.45), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -286.45),
        EASYSIMD_FLOAT32_C(   490.14), EASYSIMD_FLOAT32_C(  -286.45), EASYSIMD_FLOAT32_C(   490.14), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 66),
      { EASYSIMD_FLOAT32_C(  -622.52), EASYSIMD_FLOAT32_C(  -691.02), EASYSIMD_FLOAT32_C(    48.53), EASYSIMD_FLOAT32_C(  -368.74) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -691.02), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -622.52), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(140),
      { EASYSIMD_FLOAT32_C(   336.37), EASYSIMD_FLOAT32_C(  -709.34), EASYSIMD_FLOAT32_C(    65.79), EASYSIMD_FLOAT32_C(  -200.10) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   336.37), EASYSIMD_FLOAT32_C(  -709.34),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -709.34) } },
    { UINT8_C(151),
      { EASYSIMD_FLOAT32_C(   450.42), EASYSIMD_FLOAT32_C(   257.72), EASYSIMD_FLOAT32_C(  -507.45), EASYSIMD_FLOAT32_C(  -644.25) },
      { EASYSIMD_FLOAT32_C(   450.42), EASYSIMD_FLOAT32_C(   257.72), EASYSIMD_FLOAT32_C(   450.42), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   450.42), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   257.72) } },
    { UINT8_C( 11),
      { EASYSIMD_FLOAT32_C(  -161.31), EASYSIMD_FLOAT32_C(   845.16), EASYSIMD_FLOAT32_C(   584.32), EASYSIMD_FLOAT32_C(   641.28) },
      { EASYSIMD_FLOAT32_C(  -161.31), EASYSIMD_FLOAT32_C(   845.16), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   845.16),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 82),
      { EASYSIMD_FLOAT32_C(   565.26), EASYSIMD_FLOAT32_C(   325.20), EASYSIMD_FLOAT32_C(  -344.79), EASYSIMD_FLOAT32_C(  -940.47) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   325.20), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   565.26), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   565.26), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(152),
      { EASYSIMD_FLOAT32_C(   715.85), EASYSIMD_FLOAT32_C(  -726.67), EASYSIMD_FLOAT32_C(   812.36), EASYSIMD_FLOAT32_C(  -643.19) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -726.67),
        EASYSIMD_FLOAT32_C(   715.85), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -726.67) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_broadcast_f32x2(test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_broadcast_f32x2");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcast_f32x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -125.63), EASYSIMD_FLOAT32_C(   601.06), EASYSIMD_FLOAT32_C(    20.21), EASYSIMD_FLOAT32_C(  -317.28) },
      { EASYSIMD_FLOAT32_C(  -125.63), EASYSIMD_FLOAT32_C(   601.06), EASYSIMD_FLOAT32_C(  -125.63), EASYSIMD_FLOAT32_C(   601.06),
        EASYSIMD_FLOAT32_C(  -125.63), EASYSIMD_FLOAT32_C(   601.06), EASYSIMD_FLOAT32_C(  -125.63), EASYSIMD_FLOAT32_C(   601.06),
        EASYSIMD_FLOAT32_C(  -125.63), EASYSIMD_FLOAT32_C(   601.06), EASYSIMD_FLOAT32_C(  -125.63), EASYSIMD_FLOAT32_C(   601.06),
        EASYSIMD_FLOAT32_C(  -125.63), EASYSIMD_FLOAT32_C(   601.06), EASYSIMD_FLOAT32_C(  -125.63), EASYSIMD_FLOAT32_C(   601.06) } },
    { { EASYSIMD_FLOAT32_C(  -590.78), EASYSIMD_FLOAT32_C(   832.90), EASYSIMD_FLOAT32_C(   590.84), EASYSIMD_FLOAT32_C(   180.72) },
      { EASYSIMD_FLOAT32_C(  -590.78), EASYSIMD_FLOAT32_C(   832.90), EASYSIMD_FLOAT32_C(  -590.78), EASYSIMD_FLOAT32_C(   832.90),
        EASYSIMD_FLOAT32_C(  -590.78), EASYSIMD_FLOAT32_C(   832.90), EASYSIMD_FLOAT32_C(  -590.78), EASYSIMD_FLOAT32_C(   832.90),
        EASYSIMD_FLOAT32_C(  -590.78), EASYSIMD_FLOAT32_C(   832.90), EASYSIMD_FLOAT32_C(  -590.78), EASYSIMD_FLOAT32_C(   832.90),
        EASYSIMD_FLOAT32_C(  -590.78), EASYSIMD_FLOAT32_C(   832.90), EASYSIMD_FLOAT32_C(  -590.78), EASYSIMD_FLOAT32_C(   832.90) } },
    { { EASYSIMD_FLOAT32_C(  -605.74), EASYSIMD_FLOAT32_C(  -713.02), EASYSIMD_FLOAT32_C(   218.93), EASYSIMD_FLOAT32_C(  -470.99) },
      { EASYSIMD_FLOAT32_C(  -605.74), EASYSIMD_FLOAT32_C(  -713.02), EASYSIMD_FLOAT32_C(  -605.74), EASYSIMD_FLOAT32_C(  -713.02),
        EASYSIMD_FLOAT32_C(  -605.74), EASYSIMD_FLOAT32_C(  -713.02), EASYSIMD_FLOAT32_C(  -605.74), EASYSIMD_FLOAT32_C(  -713.02),
        EASYSIMD_FLOAT32_C(  -605.74), EASYSIMD_FLOAT32_C(  -713.02), EASYSIMD_FLOAT32_C(  -605.74), EASYSIMD_FLOAT32_C(  -713.02),
        EASYSIMD_FLOAT32_C(  -605.74), EASYSIMD_FLOAT32_C(  -713.02), EASYSIMD_FLOAT32_C(  -605.74), EASYSIMD_FLOAT32_C(  -713.02) } },
    { { EASYSIMD_FLOAT32_C(    61.13), EASYSIMD_FLOAT32_C(  -592.59), EASYSIMD_FLOAT32_C(   423.81), EASYSIMD_FLOAT32_C(   987.29) },
      { EASYSIMD_FLOAT32_C(    61.13), EASYSIMD_FLOAT32_C(  -592.59), EASYSIMD_FLOAT32_C(    61.13), EASYSIMD_FLOAT32_C(  -592.59),
        EASYSIMD_FLOAT32_C(    61.13), EASYSIMD_FLOAT32_C(  -592.59), EASYSIMD_FLOAT32_C(    61.13), EASYSIMD_FLOAT32_C(  -592.59),
        EASYSIMD_FLOAT32_C(    61.13), EASYSIMD_FLOAT32_C(  -592.59), EASYSIMD_FLOAT32_C(    61.13), EASYSIMD_FLOAT32_C(  -592.59),
        EASYSIMD_FLOAT32_C(    61.13), EASYSIMD_FLOAT32_C(  -592.59), EASYSIMD_FLOAT32_C(    61.13), EASYSIMD_FLOAT32_C(  -592.59) } },
    { { EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -99.23), EASYSIMD_FLOAT32_C(   300.10), EASYSIMD_FLOAT32_C(  -254.94) },
      { EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -99.23), EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -99.23),
        EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -99.23), EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -99.23),
        EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -99.23), EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -99.23),
        EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -99.23), EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -99.23) } },
    { { EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   285.14), EASYSIMD_FLOAT32_C(   318.61), EASYSIMD_FLOAT32_C(   956.19) },
      { EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   285.14), EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   285.14),
        EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   285.14), EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   285.14),
        EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   285.14), EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   285.14),
        EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   285.14), EASYSIMD_FLOAT32_C(  -727.78), EASYSIMD_FLOAT32_C(   285.14) } },
    { { EASYSIMD_FLOAT32_C(   704.27), EASYSIMD_FLOAT32_C(   738.40), EASYSIMD_FLOAT32_C(   301.28), EASYSIMD_FLOAT32_C(  -459.90) },
      { EASYSIMD_FLOAT32_C(   704.27), EASYSIMD_FLOAT32_C(   738.40), EASYSIMD_FLOAT32_C(   704.27), EASYSIMD_FLOAT32_C(   738.40),
        EASYSIMD_FLOAT32_C(   704.27), EASYSIMD_FLOAT32_C(   738.40), EASYSIMD_FLOAT32_C(   704.27), EASYSIMD_FLOAT32_C(   738.40),
        EASYSIMD_FLOAT32_C(   704.27), EASYSIMD_FLOAT32_C(   738.40), EASYSIMD_FLOAT32_C(   704.27), EASYSIMD_FLOAT32_C(   738.40),
        EASYSIMD_FLOAT32_C(   704.27), EASYSIMD_FLOAT32_C(   738.40), EASYSIMD_FLOAT32_C(   704.27), EASYSIMD_FLOAT32_C(   738.40) } },
    { { EASYSIMD_FLOAT32_C(   379.79), EASYSIMD_FLOAT32_C(  -819.17), EASYSIMD_FLOAT32_C(   172.39), EASYSIMD_FLOAT32_C(  -722.17) },
      { EASYSIMD_FLOAT32_C(   379.79), EASYSIMD_FLOAT32_C(  -819.17), EASYSIMD_FLOAT32_C(   379.79), EASYSIMD_FLOAT32_C(  -819.17),
        EASYSIMD_FLOAT32_C(   379.79), EASYSIMD_FLOAT32_C(  -819.17), EASYSIMD_FLOAT32_C(   379.79), EASYSIMD_FLOAT32_C(  -819.17),
        EASYSIMD_FLOAT32_C(   379.79), EASYSIMD_FLOAT32_C(  -819.17), EASYSIMD_FLOAT32_C(   379.79), EASYSIMD_FLOAT32_C(  -819.17),
        EASYSIMD_FLOAT32_C(   379.79), EASYSIMD_FLOAT32_C(  -819.17), EASYSIMD_FLOAT32_C(   379.79), EASYSIMD_FLOAT32_C(  -819.17) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_broadcast_f32x2(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_broadcast_f32x2");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcast_f32x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 src[16];
    const easysimd__mmask16 k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    16.97), EASYSIMD_FLOAT32_C(  -724.36), EASYSIMD_FLOAT32_C(  -251.03), EASYSIMD_FLOAT32_C(   955.86),
        EASYSIMD_FLOAT32_C(  -884.86), EASYSIMD_FLOAT32_C(    79.30), EASYSIMD_FLOAT32_C(   805.27), EASYSIMD_FLOAT32_C(   217.58),
        EASYSIMD_FLOAT32_C(   919.33), EASYSIMD_FLOAT32_C(  -770.42), EASYSIMD_FLOAT32_C(  -363.93), EASYSIMD_FLOAT32_C(  -528.80),
        EASYSIMD_FLOAT32_C(   387.46), EASYSIMD_FLOAT32_C(     8.94), EASYSIMD_FLOAT32_C(   238.55), EASYSIMD_FLOAT32_C(  -769.11) },
      UINT16_C(26495),
      { EASYSIMD_FLOAT32_C(  -701.54), EASYSIMD_FLOAT32_C(  -832.82), EASYSIMD_FLOAT32_C(   858.15), EASYSIMD_FLOAT32_C(   988.45) },
      { EASYSIMD_FLOAT32_C(  -701.54), EASYSIMD_FLOAT32_C(  -832.82), EASYSIMD_FLOAT32_C(  -701.54), EASYSIMD_FLOAT32_C(  -832.82),
        EASYSIMD_FLOAT32_C(  -701.54), EASYSIMD_FLOAT32_C(  -832.82), EASYSIMD_FLOAT32_C(  -701.54), EASYSIMD_FLOAT32_C(   217.58),
        EASYSIMD_FLOAT32_C(  -701.54), EASYSIMD_FLOAT32_C(  -832.82), EASYSIMD_FLOAT32_C(  -701.54), EASYSIMD_FLOAT32_C(  -528.80),
        EASYSIMD_FLOAT32_C(   387.46), EASYSIMD_FLOAT32_C(  -832.82), EASYSIMD_FLOAT32_C(  -701.54), EASYSIMD_FLOAT32_C(  -769.11) } },
    { { EASYSIMD_FLOAT32_C(   886.30), EASYSIMD_FLOAT32_C(   115.75), EASYSIMD_FLOAT32_C(  -627.06), EASYSIMD_FLOAT32_C(  -987.33),
        EASYSIMD_FLOAT32_C(  -126.79), EASYSIMD_FLOAT32_C(   964.00), EASYSIMD_FLOAT32_C(  -128.64), EASYSIMD_FLOAT32_C(   -75.15),
        EASYSIMD_FLOAT32_C(   949.72), EASYSIMD_FLOAT32_C(  -114.82), EASYSIMD_FLOAT32_C(   286.01), EASYSIMD_FLOAT32_C(  -995.38),
        EASYSIMD_FLOAT32_C(   721.81), EASYSIMD_FLOAT32_C(  -531.94), EASYSIMD_FLOAT32_C(  -379.35), EASYSIMD_FLOAT32_C(   301.40) },
      UINT16_C(55066),
      { EASYSIMD_FLOAT32_C(  -112.99), EASYSIMD_FLOAT32_C(   933.42), EASYSIMD_FLOAT32_C(   -66.18), EASYSIMD_FLOAT32_C(  -307.32) },
      { EASYSIMD_FLOAT32_C(   886.30), EASYSIMD_FLOAT32_C(   933.42), EASYSIMD_FLOAT32_C(  -627.06), EASYSIMD_FLOAT32_C(   933.42),
        EASYSIMD_FLOAT32_C(  -112.99), EASYSIMD_FLOAT32_C(   964.00), EASYSIMD_FLOAT32_C(  -128.64), EASYSIMD_FLOAT32_C(   -75.15),
        EASYSIMD_FLOAT32_C(  -112.99), EASYSIMD_FLOAT32_C(   933.42), EASYSIMD_FLOAT32_C(  -112.99), EASYSIMD_FLOAT32_C(  -995.38),
        EASYSIMD_FLOAT32_C(  -112.99), EASYSIMD_FLOAT32_C(  -531.94), EASYSIMD_FLOAT32_C(  -112.99), EASYSIMD_FLOAT32_C(   933.42) } },
    { { EASYSIMD_FLOAT32_C(   858.06), EASYSIMD_FLOAT32_C(  -630.09), EASYSIMD_FLOAT32_C(    82.49), EASYSIMD_FLOAT32_C(   401.49),
        EASYSIMD_FLOAT32_C(  -226.24), EASYSIMD_FLOAT32_C(  -448.63), EASYSIMD_FLOAT32_C(  -200.28), EASYSIMD_FLOAT32_C(  -144.91),
        EASYSIMD_FLOAT32_C(   574.72), EASYSIMD_FLOAT32_C(  -647.66), EASYSIMD_FLOAT32_C(   850.68), EASYSIMD_FLOAT32_C(  -645.45),
        EASYSIMD_FLOAT32_C(  -136.23), EASYSIMD_FLOAT32_C(   385.26), EASYSIMD_FLOAT32_C(  -998.08), EASYSIMD_FLOAT32_C(  -718.84) },
      UINT16_C(39639),
      { EASYSIMD_FLOAT32_C(  -394.96), EASYSIMD_FLOAT32_C(   -89.93), EASYSIMD_FLOAT32_C(   511.24), EASYSIMD_FLOAT32_C(   328.98) },
      { EASYSIMD_FLOAT32_C(  -394.96), EASYSIMD_FLOAT32_C(   -89.93), EASYSIMD_FLOAT32_C(  -394.96), EASYSIMD_FLOAT32_C(   401.49),
        EASYSIMD_FLOAT32_C(  -394.96), EASYSIMD_FLOAT32_C(  -448.63), EASYSIMD_FLOAT32_C(  -394.96), EASYSIMD_FLOAT32_C(   -89.93),
        EASYSIMD_FLOAT32_C(   574.72), EASYSIMD_FLOAT32_C(   -89.93), EASYSIMD_FLOAT32_C(   850.68), EASYSIMD_FLOAT32_C(   -89.93),
        EASYSIMD_FLOAT32_C(  -394.96), EASYSIMD_FLOAT32_C(   385.26), EASYSIMD_FLOAT32_C(  -998.08), EASYSIMD_FLOAT32_C(   -89.93) } },
    { { EASYSIMD_FLOAT32_C(  -783.73), EASYSIMD_FLOAT32_C(  -210.92), EASYSIMD_FLOAT32_C(  -991.67), EASYSIMD_FLOAT32_C(   979.95),
        EASYSIMD_FLOAT32_C(    49.71), EASYSIMD_FLOAT32_C(  -489.71), EASYSIMD_FLOAT32_C(  -591.16), EASYSIMD_FLOAT32_C(   388.37),
        EASYSIMD_FLOAT32_C(  -622.36), EASYSIMD_FLOAT32_C(    45.42), EASYSIMD_FLOAT32_C(  -553.07), EASYSIMD_FLOAT32_C(   498.54),
        EASYSIMD_FLOAT32_C(   904.46), EASYSIMD_FLOAT32_C(  -795.68), EASYSIMD_FLOAT32_C(  -943.60), EASYSIMD_FLOAT32_C(   933.59) },
      UINT16_C(44422),
      { EASYSIMD_FLOAT32_C(   213.33), EASYSIMD_FLOAT32_C(  -541.90), EASYSIMD_FLOAT32_C(   310.55), EASYSIMD_FLOAT32_C(  -596.77) },
      { EASYSIMD_FLOAT32_C(  -783.73), EASYSIMD_FLOAT32_C(  -541.90), EASYSIMD_FLOAT32_C(   213.33), EASYSIMD_FLOAT32_C(   979.95),
        EASYSIMD_FLOAT32_C(    49.71), EASYSIMD_FLOAT32_C(  -489.71), EASYSIMD_FLOAT32_C(  -591.16), EASYSIMD_FLOAT32_C(  -541.90),
        EASYSIMD_FLOAT32_C(   213.33), EASYSIMD_FLOAT32_C(    45.42), EASYSIMD_FLOAT32_C(   213.33), EASYSIMD_FLOAT32_C(  -541.90),
        EASYSIMD_FLOAT32_C(   904.46), EASYSIMD_FLOAT32_C(  -541.90), EASYSIMD_FLOAT32_C(  -943.60), EASYSIMD_FLOAT32_C(  -541.90) } },
    { { EASYSIMD_FLOAT32_C(     4.43), EASYSIMD_FLOAT32_C(   378.61), EASYSIMD_FLOAT32_C(  -660.44), EASYSIMD_FLOAT32_C(   -60.44),
        EASYSIMD_FLOAT32_C(   265.90), EASYSIMD_FLOAT32_C(   922.57), EASYSIMD_FLOAT32_C(  -447.45), EASYSIMD_FLOAT32_C(  -208.75),
        EASYSIMD_FLOAT32_C(  -386.55), EASYSIMD_FLOAT32_C(  -791.16), EASYSIMD_FLOAT32_C(   993.63), EASYSIMD_FLOAT32_C(  -107.89),
        EASYSIMD_FLOAT32_C(   758.84), EASYSIMD_FLOAT32_C(  -215.37), EASYSIMD_FLOAT32_C(   198.46), EASYSIMD_FLOAT32_C(  -486.35) },
      UINT16_C(19819),
      { EASYSIMD_FLOAT32_C(   413.19), EASYSIMD_FLOAT32_C(   527.77), EASYSIMD_FLOAT32_C(   286.90), EASYSIMD_FLOAT32_C(   -50.52) },
      { EASYSIMD_FLOAT32_C(   413.19), EASYSIMD_FLOAT32_C(   527.77), EASYSIMD_FLOAT32_C(  -660.44), EASYSIMD_FLOAT32_C(   527.77),
        EASYSIMD_FLOAT32_C(   265.90), EASYSIMD_FLOAT32_C(   527.77), EASYSIMD_FLOAT32_C(   413.19), EASYSIMD_FLOAT32_C(  -208.75),
        EASYSIMD_FLOAT32_C(   413.19), EASYSIMD_FLOAT32_C(  -791.16), EASYSIMD_FLOAT32_C(   413.19), EASYSIMD_FLOAT32_C(   527.77),
        EASYSIMD_FLOAT32_C(   758.84), EASYSIMD_FLOAT32_C(  -215.37), EASYSIMD_FLOAT32_C(   413.19), EASYSIMD_FLOAT32_C(  -486.35) } },
    { { EASYSIMD_FLOAT32_C(   968.23), EASYSIMD_FLOAT32_C(  -877.74), EASYSIMD_FLOAT32_C(  -102.63), EASYSIMD_FLOAT32_C(  -954.86),
        EASYSIMD_FLOAT32_C(  -411.69), EASYSIMD_FLOAT32_C(   708.12), EASYSIMD_FLOAT32_C(  -635.17), EASYSIMD_FLOAT32_C(   743.77),
        EASYSIMD_FLOAT32_C(   622.65), EASYSIMD_FLOAT32_C(   851.75), EASYSIMD_FLOAT32_C(  -569.83), EASYSIMD_FLOAT32_C(   908.51),
        EASYSIMD_FLOAT32_C(  -674.71), EASYSIMD_FLOAT32_C(   173.61), EASYSIMD_FLOAT32_C(  -162.66), EASYSIMD_FLOAT32_C(   200.03) },
      UINT16_C(57825),
      { EASYSIMD_FLOAT32_C(  -696.94), EASYSIMD_FLOAT32_C(   529.84), EASYSIMD_FLOAT32_C(  -942.89), EASYSIMD_FLOAT32_C(   880.87) },
      { EASYSIMD_FLOAT32_C(  -696.94), EASYSIMD_FLOAT32_C(  -877.74), EASYSIMD_FLOAT32_C(  -102.63), EASYSIMD_FLOAT32_C(  -954.86),
        EASYSIMD_FLOAT32_C(  -411.69), EASYSIMD_FLOAT32_C(   529.84), EASYSIMD_FLOAT32_C(  -696.94), EASYSIMD_FLOAT32_C(   529.84),
        EASYSIMD_FLOAT32_C(  -696.94), EASYSIMD_FLOAT32_C(   851.75), EASYSIMD_FLOAT32_C(  -569.83), EASYSIMD_FLOAT32_C(   908.51),
        EASYSIMD_FLOAT32_C(  -674.71), EASYSIMD_FLOAT32_C(   529.84), EASYSIMD_FLOAT32_C(  -696.94), EASYSIMD_FLOAT32_C(   529.84) } },
    { { EASYSIMD_FLOAT32_C(   733.15), EASYSIMD_FLOAT32_C(    63.36), EASYSIMD_FLOAT32_C(   903.02), EASYSIMD_FLOAT32_C(  -977.76),
        EASYSIMD_FLOAT32_C(   704.77), EASYSIMD_FLOAT32_C(   985.75), EASYSIMD_FLOAT32_C(  -492.96), EASYSIMD_FLOAT32_C(   872.57),
        EASYSIMD_FLOAT32_C(  -697.69), EASYSIMD_FLOAT32_C(   -32.06), EASYSIMD_FLOAT32_C(  -826.65), EASYSIMD_FLOAT32_C(   423.95),
        EASYSIMD_FLOAT32_C(  -668.70), EASYSIMD_FLOAT32_C(  -777.46), EASYSIMD_FLOAT32_C(  -794.02), EASYSIMD_FLOAT32_C(   931.91) },
      UINT16_C(22885),
      { EASYSIMD_FLOAT32_C(   241.78), EASYSIMD_FLOAT32_C(  -340.95), EASYSIMD_FLOAT32_C(  -411.67), EASYSIMD_FLOAT32_C(  -904.01) },
      { EASYSIMD_FLOAT32_C(   241.78), EASYSIMD_FLOAT32_C(    63.36), EASYSIMD_FLOAT32_C(   241.78), EASYSIMD_FLOAT32_C(  -977.76),
        EASYSIMD_FLOAT32_C(   704.77), EASYSIMD_FLOAT32_C(  -340.95), EASYSIMD_FLOAT32_C(   241.78), EASYSIMD_FLOAT32_C(   872.57),
        EASYSIMD_FLOAT32_C(   241.78), EASYSIMD_FLOAT32_C(   -32.06), EASYSIMD_FLOAT32_C(  -826.65), EASYSIMD_FLOAT32_C(  -340.95),
        EASYSIMD_FLOAT32_C(   241.78), EASYSIMD_FLOAT32_C(  -777.46), EASYSIMD_FLOAT32_C(   241.78), EASYSIMD_FLOAT32_C(   931.91) } },
    { { EASYSIMD_FLOAT32_C(   377.61), EASYSIMD_FLOAT32_C(   543.54), EASYSIMD_FLOAT32_C(  -676.81), EASYSIMD_FLOAT32_C(   796.04),
        EASYSIMD_FLOAT32_C(  -952.55), EASYSIMD_FLOAT32_C(   439.69), EASYSIMD_FLOAT32_C(  -139.34), EASYSIMD_FLOAT32_C(   103.48),
        EASYSIMD_FLOAT32_C(  -782.74), EASYSIMD_FLOAT32_C(   562.99), EASYSIMD_FLOAT32_C(   161.99), EASYSIMD_FLOAT32_C(   620.38),
        EASYSIMD_FLOAT32_C(   696.86), EASYSIMD_FLOAT32_C(    88.47), EASYSIMD_FLOAT32_C(   998.69), EASYSIMD_FLOAT32_C(  -955.66) },
      UINT16_C(13591),
      { EASYSIMD_FLOAT32_C(  -395.69), EASYSIMD_FLOAT32_C(  -372.87), EASYSIMD_FLOAT32_C(  -839.61), EASYSIMD_FLOAT32_C(   668.17) },
      { EASYSIMD_FLOAT32_C(  -395.69), EASYSIMD_FLOAT32_C(  -372.87), EASYSIMD_FLOAT32_C(  -395.69), EASYSIMD_FLOAT32_C(   796.04),
        EASYSIMD_FLOAT32_C(  -395.69), EASYSIMD_FLOAT32_C(   439.69), EASYSIMD_FLOAT32_C(  -139.34), EASYSIMD_FLOAT32_C(   103.48),
        EASYSIMD_FLOAT32_C(  -395.69), EASYSIMD_FLOAT32_C(   562.99), EASYSIMD_FLOAT32_C(  -395.69), EASYSIMD_FLOAT32_C(   620.38),
        EASYSIMD_FLOAT32_C(  -395.69), EASYSIMD_FLOAT32_C(  -372.87), EASYSIMD_FLOAT32_C(   998.69), EASYSIMD_FLOAT32_C(  -955.66) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 src = easysimd_mm512_loadu_ps(test_vec[i].src);
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_broadcast_f32x2(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_broadcast_f32x2");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcast_f32x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[16];
  } test_vec[] = {
            { UINT16_C(18884),
      { EASYSIMD_FLOAT32_C(   545.10), EASYSIMD_FLOAT32_C(  -550.17), EASYSIMD_FLOAT32_C(  -710.41), EASYSIMD_FLOAT32_C(   204.85) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   545.10), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   545.10), EASYSIMD_FLOAT32_C(  -550.17),
        EASYSIMD_FLOAT32_C(   545.10), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -550.17),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   545.10), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(16968),
      { EASYSIMD_FLOAT32_C(    51.85), EASYSIMD_FLOAT32_C(  -493.14), EASYSIMD_FLOAT32_C(  -214.52), EASYSIMD_FLOAT32_C(   484.86) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -493.14),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    51.85), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -493.14), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    51.85), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(55493),
      { EASYSIMD_FLOAT32_C(  -547.31), EASYSIMD_FLOAT32_C(  -681.83), EASYSIMD_FLOAT32_C(   567.76), EASYSIMD_FLOAT32_C(   376.14) },
      { EASYSIMD_FLOAT32_C(  -547.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -547.31), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -547.31), EASYSIMD_FLOAT32_C(  -681.83),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -681.83),
        EASYSIMD_FLOAT32_C(  -547.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -547.31), EASYSIMD_FLOAT32_C(  -681.83) } },
    { UINT16_C( 1280),
      { EASYSIMD_FLOAT32_C(   358.99), EASYSIMD_FLOAT32_C(  -507.35), EASYSIMD_FLOAT32_C(  -959.80), EASYSIMD_FLOAT32_C(   688.48) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   358.99), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   358.99), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(16569),
      { EASYSIMD_FLOAT32_C(  -988.71), EASYSIMD_FLOAT32_C(   789.03), EASYSIMD_FLOAT32_C(  -740.57), EASYSIMD_FLOAT32_C(  -739.46) },
      { EASYSIMD_FLOAT32_C(  -988.71), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   789.03),
        EASYSIMD_FLOAT32_C(  -988.71), EASYSIMD_FLOAT32_C(   789.03), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   789.03),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -988.71), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(26242),
      { EASYSIMD_FLOAT32_C(  -555.34), EASYSIMD_FLOAT32_C(   402.79), EASYSIMD_FLOAT32_C(  -274.64), EASYSIMD_FLOAT32_C(   159.53) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   402.79), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   402.79),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   402.79), EASYSIMD_FLOAT32_C(  -555.34), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   402.79), EASYSIMD_FLOAT32_C(  -555.34), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(39055),
      { EASYSIMD_FLOAT32_C(   -25.84), EASYSIMD_FLOAT32_C(  -228.90), EASYSIMD_FLOAT32_C(   813.40), EASYSIMD_FLOAT32_C(   762.90) },
      { EASYSIMD_FLOAT32_C(   -25.84), EASYSIMD_FLOAT32_C(  -228.90), EASYSIMD_FLOAT32_C(   -25.84), EASYSIMD_FLOAT32_C(  -228.90),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -228.90),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -228.90),
        EASYSIMD_FLOAT32_C(   -25.84), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -228.90) } },
    { UINT16_C(53187),
      { EASYSIMD_FLOAT32_C(  -400.08), EASYSIMD_FLOAT32_C(  -173.64), EASYSIMD_FLOAT32_C(  -349.66), EASYSIMD_FLOAT32_C(  -663.64) },
      { EASYSIMD_FLOAT32_C(  -400.08), EASYSIMD_FLOAT32_C(  -173.64), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -400.08), EASYSIMD_FLOAT32_C(  -173.64),
        EASYSIMD_FLOAT32_C(  -400.08), EASYSIMD_FLOAT32_C(  -173.64), EASYSIMD_FLOAT32_C(  -400.08), EASYSIMD_FLOAT32_C(  -173.64),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -400.08), EASYSIMD_FLOAT32_C(  -173.64) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_broadcast_f32x2(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_broadcast_f32x2");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcast_f32x8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[8];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -416.45), EASYSIMD_FLOAT32_C(   711.04), EASYSIMD_FLOAT32_C(   494.08), EASYSIMD_FLOAT32_C(    55.06),
        EASYSIMD_FLOAT32_C(  -527.80), EASYSIMD_FLOAT32_C(  -810.11), EASYSIMD_FLOAT32_C(   486.30), EASYSIMD_FLOAT32_C(  -695.23) },
      { EASYSIMD_FLOAT32_C(  -416.45), EASYSIMD_FLOAT32_C(   711.04), EASYSIMD_FLOAT32_C(   494.08), EASYSIMD_FLOAT32_C(    55.06),
        EASYSIMD_FLOAT32_C(  -527.80), EASYSIMD_FLOAT32_C(  -810.11), EASYSIMD_FLOAT32_C(   486.30), EASYSIMD_FLOAT32_C(  -695.23),
        EASYSIMD_FLOAT32_C(  -416.45), EASYSIMD_FLOAT32_C(   711.04), EASYSIMD_FLOAT32_C(   494.08), EASYSIMD_FLOAT32_C(    55.06),
        EASYSIMD_FLOAT32_C(  -527.80), EASYSIMD_FLOAT32_C(  -810.11), EASYSIMD_FLOAT32_C(   486.30), EASYSIMD_FLOAT32_C(  -695.23) } },
    { { EASYSIMD_FLOAT32_C(  -800.88), EASYSIMD_FLOAT32_C(  -452.72), EASYSIMD_FLOAT32_C(  -904.66), EASYSIMD_FLOAT32_C(  -614.99),
        EASYSIMD_FLOAT32_C(  -172.17), EASYSIMD_FLOAT32_C(   311.84), EASYSIMD_FLOAT32_C(  -833.25), EASYSIMD_FLOAT32_C(  -503.53) },
      { EASYSIMD_FLOAT32_C(  -800.88), EASYSIMD_FLOAT32_C(  -452.72), EASYSIMD_FLOAT32_C(  -904.66), EASYSIMD_FLOAT32_C(  -614.99),
        EASYSIMD_FLOAT32_C(  -172.17), EASYSIMD_FLOAT32_C(   311.84), EASYSIMD_FLOAT32_C(  -833.25), EASYSIMD_FLOAT32_C(  -503.53),
        EASYSIMD_FLOAT32_C(  -800.88), EASYSIMD_FLOAT32_C(  -452.72), EASYSIMD_FLOAT32_C(  -904.66), EASYSIMD_FLOAT32_C(  -614.99),
        EASYSIMD_FLOAT32_C(  -172.17), EASYSIMD_FLOAT32_C(   311.84), EASYSIMD_FLOAT32_C(  -833.25), EASYSIMD_FLOAT32_C(  -503.53) } },
    { { EASYSIMD_FLOAT32_C(  -875.06), EASYSIMD_FLOAT32_C(   874.51), EASYSIMD_FLOAT32_C(  -123.24), EASYSIMD_FLOAT32_C(   657.48),
        EASYSIMD_FLOAT32_C(   309.07), EASYSIMD_FLOAT32_C(   484.03), EASYSIMD_FLOAT32_C(  -839.17), EASYSIMD_FLOAT32_C(    10.32) },
      { EASYSIMD_FLOAT32_C(  -875.06), EASYSIMD_FLOAT32_C(   874.51), EASYSIMD_FLOAT32_C(  -123.24), EASYSIMD_FLOAT32_C(   657.48),
        EASYSIMD_FLOAT32_C(   309.07), EASYSIMD_FLOAT32_C(   484.03), EASYSIMD_FLOAT32_C(  -839.17), EASYSIMD_FLOAT32_C(    10.32),
        EASYSIMD_FLOAT32_C(  -875.06), EASYSIMD_FLOAT32_C(   874.51), EASYSIMD_FLOAT32_C(  -123.24), EASYSIMD_FLOAT32_C(   657.48),
        EASYSIMD_FLOAT32_C(   309.07), EASYSIMD_FLOAT32_C(   484.03), EASYSIMD_FLOAT32_C(  -839.17), EASYSIMD_FLOAT32_C(    10.32) } },
    { { EASYSIMD_FLOAT32_C(  -515.09), EASYSIMD_FLOAT32_C(   924.58), EASYSIMD_FLOAT32_C(  -659.21), EASYSIMD_FLOAT32_C(   676.36),
        EASYSIMD_FLOAT32_C(  -421.41), EASYSIMD_FLOAT32_C(  -682.12), EASYSIMD_FLOAT32_C(  -306.00), EASYSIMD_FLOAT32_C(  -939.89) },
      { EASYSIMD_FLOAT32_C(  -515.09), EASYSIMD_FLOAT32_C(   924.58), EASYSIMD_FLOAT32_C(  -659.21), EASYSIMD_FLOAT32_C(   676.36),
        EASYSIMD_FLOAT32_C(  -421.41), EASYSIMD_FLOAT32_C(  -682.12), EASYSIMD_FLOAT32_C(  -306.00), EASYSIMD_FLOAT32_C(  -939.89),
        EASYSIMD_FLOAT32_C(  -515.09), EASYSIMD_FLOAT32_C(   924.58), EASYSIMD_FLOAT32_C(  -659.21), EASYSIMD_FLOAT32_C(   676.36),
        EASYSIMD_FLOAT32_C(  -421.41), EASYSIMD_FLOAT32_C(  -682.12), EASYSIMD_FLOAT32_C(  -306.00), EASYSIMD_FLOAT32_C(  -939.89) } },
    { { EASYSIMD_FLOAT32_C(  -812.70), EASYSIMD_FLOAT32_C(   906.23), EASYSIMD_FLOAT32_C(  -979.37), EASYSIMD_FLOAT32_C(  -275.20),
        EASYSIMD_FLOAT32_C(   664.08), EASYSIMD_FLOAT32_C(  -809.85), EASYSIMD_FLOAT32_C(   934.39), EASYSIMD_FLOAT32_C(   280.51) },
      { EASYSIMD_FLOAT32_C(  -812.70), EASYSIMD_FLOAT32_C(   906.23), EASYSIMD_FLOAT32_C(  -979.37), EASYSIMD_FLOAT32_C(  -275.20),
        EASYSIMD_FLOAT32_C(   664.08), EASYSIMD_FLOAT32_C(  -809.85), EASYSIMD_FLOAT32_C(   934.39), EASYSIMD_FLOAT32_C(   280.51),
        EASYSIMD_FLOAT32_C(  -812.70), EASYSIMD_FLOAT32_C(   906.23), EASYSIMD_FLOAT32_C(  -979.37), EASYSIMD_FLOAT32_C(  -275.20),
        EASYSIMD_FLOAT32_C(   664.08), EASYSIMD_FLOAT32_C(  -809.85), EASYSIMD_FLOAT32_C(   934.39), EASYSIMD_FLOAT32_C(   280.51) } },
    { { EASYSIMD_FLOAT32_C(   461.56), EASYSIMD_FLOAT32_C(  -484.84), EASYSIMD_FLOAT32_C(  -776.35), EASYSIMD_FLOAT32_C(   -37.28),
        EASYSIMD_FLOAT32_C(  -552.72), EASYSIMD_FLOAT32_C(   358.22), EASYSIMD_FLOAT32_C(   561.82), EASYSIMD_FLOAT32_C(   465.10) },
      { EASYSIMD_FLOAT32_C(   461.56), EASYSIMD_FLOAT32_C(  -484.84), EASYSIMD_FLOAT32_C(  -776.35), EASYSIMD_FLOAT32_C(   -37.28),
        EASYSIMD_FLOAT32_C(  -552.72), EASYSIMD_FLOAT32_C(   358.22), EASYSIMD_FLOAT32_C(   561.82), EASYSIMD_FLOAT32_C(   465.10),
        EASYSIMD_FLOAT32_C(   461.56), EASYSIMD_FLOAT32_C(  -484.84), EASYSIMD_FLOAT32_C(  -776.35), EASYSIMD_FLOAT32_C(   -37.28),
        EASYSIMD_FLOAT32_C(  -552.72), EASYSIMD_FLOAT32_C(   358.22), EASYSIMD_FLOAT32_C(   561.82), EASYSIMD_FLOAT32_C(   465.10) } },
    { { EASYSIMD_FLOAT32_C(   996.67), EASYSIMD_FLOAT32_C(  -908.09), EASYSIMD_FLOAT32_C(  -292.64), EASYSIMD_FLOAT32_C(  -421.79),
        EASYSIMD_FLOAT32_C(  -984.50), EASYSIMD_FLOAT32_C(  -529.88), EASYSIMD_FLOAT32_C(   228.67), EASYSIMD_FLOAT32_C(  -756.34) },
      { EASYSIMD_FLOAT32_C(   996.67), EASYSIMD_FLOAT32_C(  -908.09), EASYSIMD_FLOAT32_C(  -292.64), EASYSIMD_FLOAT32_C(  -421.79),
        EASYSIMD_FLOAT32_C(  -984.50), EASYSIMD_FLOAT32_C(  -529.88), EASYSIMD_FLOAT32_C(   228.67), EASYSIMD_FLOAT32_C(  -756.34),
        EASYSIMD_FLOAT32_C(   996.67), EASYSIMD_FLOAT32_C(  -908.09), EASYSIMD_FLOAT32_C(  -292.64), EASYSIMD_FLOAT32_C(  -421.79),
        EASYSIMD_FLOAT32_C(  -984.50), EASYSIMD_FLOAT32_C(  -529.88), EASYSIMD_FLOAT32_C(   228.67), EASYSIMD_FLOAT32_C(  -756.34) } },
    { { EASYSIMD_FLOAT32_C(   236.36), EASYSIMD_FLOAT32_C(   442.90), EASYSIMD_FLOAT32_C(  -175.57), EASYSIMD_FLOAT32_C(  -799.66),
        EASYSIMD_FLOAT32_C(    97.65), EASYSIMD_FLOAT32_C(  -822.08), EASYSIMD_FLOAT32_C(  -738.45), EASYSIMD_FLOAT32_C(   923.13) },
      { EASYSIMD_FLOAT32_C(   236.36), EASYSIMD_FLOAT32_C(   442.90), EASYSIMD_FLOAT32_C(  -175.57), EASYSIMD_FLOAT32_C(  -799.66),
        EASYSIMD_FLOAT32_C(    97.65), EASYSIMD_FLOAT32_C(  -822.08), EASYSIMD_FLOAT32_C(  -738.45), EASYSIMD_FLOAT32_C(   923.13),
        EASYSIMD_FLOAT32_C(   236.36), EASYSIMD_FLOAT32_C(   442.90), EASYSIMD_FLOAT32_C(  -175.57), EASYSIMD_FLOAT32_C(  -799.66),
        EASYSIMD_FLOAT32_C(    97.65), EASYSIMD_FLOAT32_C(  -822.08), EASYSIMD_FLOAT32_C(  -738.45), EASYSIMD_FLOAT32_C(   923.13) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_broadcast_f32x8(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_broadcast_f32x8");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcast_f32x8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 src[16];
    const easysimd__mmask16 k;
    const easysimd_float32 a[8];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   280.29), EASYSIMD_FLOAT32_C(   838.38), EASYSIMD_FLOAT32_C(   622.29), EASYSIMD_FLOAT32_C(   762.17),
        EASYSIMD_FLOAT32_C(  -281.25), EASYSIMD_FLOAT32_C(   985.78), EASYSIMD_FLOAT32_C(    78.74), EASYSIMD_FLOAT32_C(  -555.08),
        EASYSIMD_FLOAT32_C(   759.89), EASYSIMD_FLOAT32_C(  -557.22), EASYSIMD_FLOAT32_C(   754.50), EASYSIMD_FLOAT32_C(   954.59),
        EASYSIMD_FLOAT32_C(  -153.57), EASYSIMD_FLOAT32_C(   932.38), EASYSIMD_FLOAT32_C(   449.83), EASYSIMD_FLOAT32_C(   378.57) },
      UINT16_C(36924),
      { EASYSIMD_FLOAT32_C(   598.39), EASYSIMD_FLOAT32_C(  -917.42), EASYSIMD_FLOAT32_C(   853.85), EASYSIMD_FLOAT32_C(   635.72),
        EASYSIMD_FLOAT32_C(   497.82), EASYSIMD_FLOAT32_C(   880.65), EASYSIMD_FLOAT32_C(  -930.36), EASYSIMD_FLOAT32_C(  -512.19) },
      { EASYSIMD_FLOAT32_C(   280.29), EASYSIMD_FLOAT32_C(   838.38), EASYSIMD_FLOAT32_C(   853.85), EASYSIMD_FLOAT32_C(   635.72),
        EASYSIMD_FLOAT32_C(   497.82), EASYSIMD_FLOAT32_C(   880.65), EASYSIMD_FLOAT32_C(    78.74), EASYSIMD_FLOAT32_C(  -555.08),
        EASYSIMD_FLOAT32_C(   759.89), EASYSIMD_FLOAT32_C(  -557.22), EASYSIMD_FLOAT32_C(   754.50), EASYSIMD_FLOAT32_C(   954.59),
        EASYSIMD_FLOAT32_C(   497.82), EASYSIMD_FLOAT32_C(   932.38), EASYSIMD_FLOAT32_C(   449.83), EASYSIMD_FLOAT32_C(  -512.19) } },
    { { EASYSIMD_FLOAT32_C(  -437.09), EASYSIMD_FLOAT32_C(  -187.22), EASYSIMD_FLOAT32_C(  -573.53), EASYSIMD_FLOAT32_C(   628.55),
        EASYSIMD_FLOAT32_C(    16.28), EASYSIMD_FLOAT32_C(  -343.67), EASYSIMD_FLOAT32_C(    13.33), EASYSIMD_FLOAT32_C(    92.74),
        EASYSIMD_FLOAT32_C(   617.88), EASYSIMD_FLOAT32_C(   659.02), EASYSIMD_FLOAT32_C(   114.72), EASYSIMD_FLOAT32_C(    86.74),
        EASYSIMD_FLOAT32_C(   -78.46), EASYSIMD_FLOAT32_C(  -669.19), EASYSIMD_FLOAT32_C(   913.81), EASYSIMD_FLOAT32_C(   480.88) },
      UINT16_C(25166),
      { EASYSIMD_FLOAT32_C(  -761.34), EASYSIMD_FLOAT32_C(   162.88), EASYSIMD_FLOAT32_C(  -410.95), EASYSIMD_FLOAT32_C(  -918.77),
        EASYSIMD_FLOAT32_C(   294.07), EASYSIMD_FLOAT32_C(   489.11), EASYSIMD_FLOAT32_C(   466.01), EASYSIMD_FLOAT32_C(   281.28) },
      { EASYSIMD_FLOAT32_C(  -437.09), EASYSIMD_FLOAT32_C(   162.88), EASYSIMD_FLOAT32_C(  -410.95), EASYSIMD_FLOAT32_C(  -918.77),
        EASYSIMD_FLOAT32_C(    16.28), EASYSIMD_FLOAT32_C(  -343.67), EASYSIMD_FLOAT32_C(   466.01), EASYSIMD_FLOAT32_C(    92.74),
        EASYSIMD_FLOAT32_C(   617.88), EASYSIMD_FLOAT32_C(   162.88), EASYSIMD_FLOAT32_C(   114.72), EASYSIMD_FLOAT32_C(    86.74),
        EASYSIMD_FLOAT32_C(   -78.46), EASYSIMD_FLOAT32_C(   489.11), EASYSIMD_FLOAT32_C(   466.01), EASYSIMD_FLOAT32_C(   480.88) } },
    { { EASYSIMD_FLOAT32_C(  -606.28), EASYSIMD_FLOAT32_C(   188.60), EASYSIMD_FLOAT32_C(  -142.85), EASYSIMD_FLOAT32_C(  -814.99),
        EASYSIMD_FLOAT32_C(   440.56), EASYSIMD_FLOAT32_C(   576.44), EASYSIMD_FLOAT32_C(   238.85), EASYSIMD_FLOAT32_C(   303.69),
        EASYSIMD_FLOAT32_C(   150.34), EASYSIMD_FLOAT32_C(   808.69), EASYSIMD_FLOAT32_C(  -362.83), EASYSIMD_FLOAT32_C(  -158.08),
        EASYSIMD_FLOAT32_C(  -803.96), EASYSIMD_FLOAT32_C(  -196.75), EASYSIMD_FLOAT32_C(  -727.89), EASYSIMD_FLOAT32_C(   308.53) },
      UINT16_C(23787),
      { EASYSIMD_FLOAT32_C(   944.26), EASYSIMD_FLOAT32_C(   110.45), EASYSIMD_FLOAT32_C(   407.09), EASYSIMD_FLOAT32_C(    45.91),
        EASYSIMD_FLOAT32_C(  -335.37), EASYSIMD_FLOAT32_C(  -560.84), EASYSIMD_FLOAT32_C(     3.97), EASYSIMD_FLOAT32_C(   760.14) },
      { EASYSIMD_FLOAT32_C(   944.26), EASYSIMD_FLOAT32_C(   110.45), EASYSIMD_FLOAT32_C(  -142.85), EASYSIMD_FLOAT32_C(    45.91),
        EASYSIMD_FLOAT32_C(   440.56), EASYSIMD_FLOAT32_C(  -560.84), EASYSIMD_FLOAT32_C(     3.97), EASYSIMD_FLOAT32_C(   760.14),
        EASYSIMD_FLOAT32_C(   150.34), EASYSIMD_FLOAT32_C(   808.69), EASYSIMD_FLOAT32_C(   407.09), EASYSIMD_FLOAT32_C(    45.91),
        EASYSIMD_FLOAT32_C(  -335.37), EASYSIMD_FLOAT32_C(  -196.75), EASYSIMD_FLOAT32_C(     3.97), EASYSIMD_FLOAT32_C(   308.53) } },
    { { EASYSIMD_FLOAT32_C(  -278.78), EASYSIMD_FLOAT32_C(   517.15), EASYSIMD_FLOAT32_C(  -283.92), EASYSIMD_FLOAT32_C(   114.05),
        EASYSIMD_FLOAT32_C(   798.05), EASYSIMD_FLOAT32_C(   868.23), EASYSIMD_FLOAT32_C(   258.92), EASYSIMD_FLOAT32_C(  -367.27),
        EASYSIMD_FLOAT32_C(  -720.23), EASYSIMD_FLOAT32_C(  -836.19), EASYSIMD_FLOAT32_C(   163.28), EASYSIMD_FLOAT32_C(   201.97),
        EASYSIMD_FLOAT32_C(   461.48), EASYSIMD_FLOAT32_C(    33.48), EASYSIMD_FLOAT32_C(   752.68), EASYSIMD_FLOAT32_C(   274.33) },
      UINT16_C( 9614),
      { EASYSIMD_FLOAT32_C(  -353.42), EASYSIMD_FLOAT32_C(    72.45), EASYSIMD_FLOAT32_C(  -313.79), EASYSIMD_FLOAT32_C(    54.95),
        EASYSIMD_FLOAT32_C(  -482.32), EASYSIMD_FLOAT32_C(  -268.09), EASYSIMD_FLOAT32_C(   146.77), EASYSIMD_FLOAT32_C(   772.72) },
      { EASYSIMD_FLOAT32_C(  -278.78), EASYSIMD_FLOAT32_C(    72.45), EASYSIMD_FLOAT32_C(  -313.79), EASYSIMD_FLOAT32_C(    54.95),
        EASYSIMD_FLOAT32_C(   798.05), EASYSIMD_FLOAT32_C(   868.23), EASYSIMD_FLOAT32_C(   258.92), EASYSIMD_FLOAT32_C(   772.72),
        EASYSIMD_FLOAT32_C(  -353.42), EASYSIMD_FLOAT32_C(  -836.19), EASYSIMD_FLOAT32_C(  -313.79), EASYSIMD_FLOAT32_C(   201.97),
        EASYSIMD_FLOAT32_C(   461.48), EASYSIMD_FLOAT32_C(  -268.09), EASYSIMD_FLOAT32_C(   752.68), EASYSIMD_FLOAT32_C(   274.33) } },
    { { EASYSIMD_FLOAT32_C(  -894.15), EASYSIMD_FLOAT32_C(    -6.16), EASYSIMD_FLOAT32_C(   455.15), EASYSIMD_FLOAT32_C(  -216.19),
        EASYSIMD_FLOAT32_C(   419.21), EASYSIMD_FLOAT32_C(  -283.83), EASYSIMD_FLOAT32_C(  -341.07), EASYSIMD_FLOAT32_C(  -431.79),
        EASYSIMD_FLOAT32_C(   825.19), EASYSIMD_FLOAT32_C(  -956.94), EASYSIMD_FLOAT32_C(   688.79), EASYSIMD_FLOAT32_C(   509.40),
        EASYSIMD_FLOAT32_C(  -511.22), EASYSIMD_FLOAT32_C(   -14.80), EASYSIMD_FLOAT32_C(  -763.30), EASYSIMD_FLOAT32_C(  -769.02) },
      UINT16_C(57357),
      { EASYSIMD_FLOAT32_C(  -152.14), EASYSIMD_FLOAT32_C(  -951.21), EASYSIMD_FLOAT32_C(   936.35), EASYSIMD_FLOAT32_C(  -713.46),
        EASYSIMD_FLOAT32_C(   933.97), EASYSIMD_FLOAT32_C(  -738.03), EASYSIMD_FLOAT32_C(     3.91), EASYSIMD_FLOAT32_C(  -225.68) },
      { EASYSIMD_FLOAT32_C(  -152.14), EASYSIMD_FLOAT32_C(    -6.16), EASYSIMD_FLOAT32_C(   936.35), EASYSIMD_FLOAT32_C(  -713.46),
        EASYSIMD_FLOAT32_C(   419.21), EASYSIMD_FLOAT32_C(  -283.83), EASYSIMD_FLOAT32_C(  -341.07), EASYSIMD_FLOAT32_C(  -431.79),
        EASYSIMD_FLOAT32_C(   825.19), EASYSIMD_FLOAT32_C(  -956.94), EASYSIMD_FLOAT32_C(   688.79), EASYSIMD_FLOAT32_C(   509.40),
        EASYSIMD_FLOAT32_C(  -511.22), EASYSIMD_FLOAT32_C(  -738.03), EASYSIMD_FLOAT32_C(     3.91), EASYSIMD_FLOAT32_C(  -225.68) } },
    { { EASYSIMD_FLOAT32_C(   958.35), EASYSIMD_FLOAT32_C(   959.55), EASYSIMD_FLOAT32_C(  -771.84), EASYSIMD_FLOAT32_C(  -312.71),
        EASYSIMD_FLOAT32_C(   261.02), EASYSIMD_FLOAT32_C(  -965.72), EASYSIMD_FLOAT32_C(  -898.55), EASYSIMD_FLOAT32_C(    98.86),
        EASYSIMD_FLOAT32_C(  -506.78), EASYSIMD_FLOAT32_C(   475.13), EASYSIMD_FLOAT32_C(  -561.78), EASYSIMD_FLOAT32_C(   145.04),
        EASYSIMD_FLOAT32_C(  -310.71), EASYSIMD_FLOAT32_C(  -100.99), EASYSIMD_FLOAT32_C(   656.93), EASYSIMD_FLOAT32_C(   955.62) },
      UINT16_C(55637),
      { EASYSIMD_FLOAT32_C(    64.66), EASYSIMD_FLOAT32_C(   704.14), EASYSIMD_FLOAT32_C(   421.81), EASYSIMD_FLOAT32_C(  -620.94),
        EASYSIMD_FLOAT32_C(  -124.06), EASYSIMD_FLOAT32_C(   858.04), EASYSIMD_FLOAT32_C(  -855.91), EASYSIMD_FLOAT32_C(   691.15) },
      { EASYSIMD_FLOAT32_C(    64.66), EASYSIMD_FLOAT32_C(   959.55), EASYSIMD_FLOAT32_C(   421.81), EASYSIMD_FLOAT32_C(  -312.71),
        EASYSIMD_FLOAT32_C(  -124.06), EASYSIMD_FLOAT32_C(  -965.72), EASYSIMD_FLOAT32_C(  -855.91), EASYSIMD_FLOAT32_C(    98.86),
        EASYSIMD_FLOAT32_C(    64.66), EASYSIMD_FLOAT32_C(   475.13), EASYSIMD_FLOAT32_C(  -561.78), EASYSIMD_FLOAT32_C(  -620.94),
        EASYSIMD_FLOAT32_C(  -124.06), EASYSIMD_FLOAT32_C(  -100.99), EASYSIMD_FLOAT32_C(  -855.91), EASYSIMD_FLOAT32_C(   691.15) } },
    { { EASYSIMD_FLOAT32_C(   165.52), EASYSIMD_FLOAT32_C(  -117.15), EASYSIMD_FLOAT32_C(  -914.50), EASYSIMD_FLOAT32_C(   -48.64),
        EASYSIMD_FLOAT32_C(   429.74), EASYSIMD_FLOAT32_C(   612.18), EASYSIMD_FLOAT32_C(   933.85), EASYSIMD_FLOAT32_C(  -778.14),
        EASYSIMD_FLOAT32_C(  -214.40), EASYSIMD_FLOAT32_C(   623.77), EASYSIMD_FLOAT32_C(  -288.84), EASYSIMD_FLOAT32_C(  -541.76),
        EASYSIMD_FLOAT32_C(   699.14), EASYSIMD_FLOAT32_C(   473.09), EASYSIMD_FLOAT32_C(  -762.45), EASYSIMD_FLOAT32_C(  -518.42) },
      UINT16_C(63181),
      { EASYSIMD_FLOAT32_C(   188.68), EASYSIMD_FLOAT32_C(  -923.58), EASYSIMD_FLOAT32_C(  -542.98), EASYSIMD_FLOAT32_C(   193.71),
        EASYSIMD_FLOAT32_C(  -319.51), EASYSIMD_FLOAT32_C(    46.76), EASYSIMD_FLOAT32_C(   -44.67), EASYSIMD_FLOAT32_C(  -768.90) },
      { EASYSIMD_FLOAT32_C(   188.68), EASYSIMD_FLOAT32_C(  -117.15), EASYSIMD_FLOAT32_C(  -542.98), EASYSIMD_FLOAT32_C(   193.71),
        EASYSIMD_FLOAT32_C(   429.74), EASYSIMD_FLOAT32_C(   612.18), EASYSIMD_FLOAT32_C(   -44.67), EASYSIMD_FLOAT32_C(  -768.90),
        EASYSIMD_FLOAT32_C(  -214.40), EASYSIMD_FLOAT32_C(  -923.58), EASYSIMD_FLOAT32_C(  -542.98), EASYSIMD_FLOAT32_C(  -541.76),
        EASYSIMD_FLOAT32_C(  -319.51), EASYSIMD_FLOAT32_C(    46.76), EASYSIMD_FLOAT32_C(   -44.67), EASYSIMD_FLOAT32_C(  -768.90) } },
    { { EASYSIMD_FLOAT32_C(  -857.07), EASYSIMD_FLOAT32_C(  -775.77), EASYSIMD_FLOAT32_C(  -351.82), EASYSIMD_FLOAT32_C(   984.69),
        EASYSIMD_FLOAT32_C(  -320.14), EASYSIMD_FLOAT32_C(  -636.62), EASYSIMD_FLOAT32_C(   297.63), EASYSIMD_FLOAT32_C(   186.04),
        EASYSIMD_FLOAT32_C(   780.35), EASYSIMD_FLOAT32_C(  -693.20), EASYSIMD_FLOAT32_C(  -589.12), EASYSIMD_FLOAT32_C(   731.33),
        EASYSIMD_FLOAT32_C(  -601.90), EASYSIMD_FLOAT32_C(  -195.41), EASYSIMD_FLOAT32_C(  -239.98), EASYSIMD_FLOAT32_C(   675.16) },
      UINT16_C(63687),
      { EASYSIMD_FLOAT32_C(   751.41), EASYSIMD_FLOAT32_C(   926.41), EASYSIMD_FLOAT32_C(   149.18), EASYSIMD_FLOAT32_C(  -662.14),
        EASYSIMD_FLOAT32_C(  -649.07), EASYSIMD_FLOAT32_C(  -858.90), EASYSIMD_FLOAT32_C(   465.33), EASYSIMD_FLOAT32_C(   831.66) },
      { EASYSIMD_FLOAT32_C(   751.41), EASYSIMD_FLOAT32_C(   926.41), EASYSIMD_FLOAT32_C(   149.18), EASYSIMD_FLOAT32_C(   984.69),
        EASYSIMD_FLOAT32_C(  -320.14), EASYSIMD_FLOAT32_C(  -636.62), EASYSIMD_FLOAT32_C(   465.33), EASYSIMD_FLOAT32_C(   831.66),
        EASYSIMD_FLOAT32_C(   780.35), EASYSIMD_FLOAT32_C(  -693.20), EASYSIMD_FLOAT32_C(  -589.12), EASYSIMD_FLOAT32_C(  -662.14),
        EASYSIMD_FLOAT32_C(  -649.07), EASYSIMD_FLOAT32_C(  -858.90), EASYSIMD_FLOAT32_C(   465.33), EASYSIMD_FLOAT32_C(   831.66) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 src = easysimd_mm512_loadu_ps(test_vec[i].src);
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_broadcast_f32x8(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_broadcast_f32x8");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcast_f32x8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const easysimd_float32 a[8];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { UINT16_C(49062),
      { EASYSIMD_FLOAT32_C(   -67.12), EASYSIMD_FLOAT32_C(  -144.98), EASYSIMD_FLOAT32_C(  -693.09), EASYSIMD_FLOAT32_C(  -717.03),
        EASYSIMD_FLOAT32_C(   833.33), EASYSIMD_FLOAT32_C(  -297.62), EASYSIMD_FLOAT32_C(  -166.55), EASYSIMD_FLOAT32_C(   748.74) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -144.98), EASYSIMD_FLOAT32_C(  -693.09), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -297.62), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   748.74),
        EASYSIMD_FLOAT32_C(   -67.12), EASYSIMD_FLOAT32_C(  -144.98), EASYSIMD_FLOAT32_C(  -693.09), EASYSIMD_FLOAT32_C(  -717.03),
        EASYSIMD_FLOAT32_C(   833.33), EASYSIMD_FLOAT32_C(  -297.62), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   748.74) } },
    { UINT16_C( 6586),
      { EASYSIMD_FLOAT32_C(  -140.91), EASYSIMD_FLOAT32_C(  -189.72), EASYSIMD_FLOAT32_C(  -663.50), EASYSIMD_FLOAT32_C(   613.12),
        EASYSIMD_FLOAT32_C(   772.89), EASYSIMD_FLOAT32_C(   -76.35), EASYSIMD_FLOAT32_C(   859.08), EASYSIMD_FLOAT32_C(   595.36) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -189.72), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   613.12),
        EASYSIMD_FLOAT32_C(   772.89), EASYSIMD_FLOAT32_C(   -76.35), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   595.36),
        EASYSIMD_FLOAT32_C(  -140.91), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   613.12),
        EASYSIMD_FLOAT32_C(   772.89), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(41380),
      { EASYSIMD_FLOAT32_C(  -218.39), EASYSIMD_FLOAT32_C(  -397.45), EASYSIMD_FLOAT32_C(    20.87), EASYSIMD_FLOAT32_C(   703.15),
        EASYSIMD_FLOAT32_C(  -126.69), EASYSIMD_FLOAT32_C(   776.77), EASYSIMD_FLOAT32_C(  -820.00), EASYSIMD_FLOAT32_C(   252.00) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    20.87), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   776.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   252.00),
        EASYSIMD_FLOAT32_C(  -218.39), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   776.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   252.00) } },
    { UINT16_C(14746),
      { EASYSIMD_FLOAT32_C(   488.59), EASYSIMD_FLOAT32_C(  -333.19), EASYSIMD_FLOAT32_C(    82.99), EASYSIMD_FLOAT32_C(   818.76),
        EASYSIMD_FLOAT32_C(   927.98), EASYSIMD_FLOAT32_C(   586.60), EASYSIMD_FLOAT32_C(   933.90), EASYSIMD_FLOAT32_C(    84.47) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -333.19), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   818.76),
        EASYSIMD_FLOAT32_C(   927.98), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    84.47),
        EASYSIMD_FLOAT32_C(   488.59), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   818.76),
        EASYSIMD_FLOAT32_C(   927.98), EASYSIMD_FLOAT32_C(   586.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(22430),
      { EASYSIMD_FLOAT32_C(  -788.60), EASYSIMD_FLOAT32_C(    -2.38), EASYSIMD_FLOAT32_C(   -57.26), EASYSIMD_FLOAT32_C(  -363.40),
        EASYSIMD_FLOAT32_C(   348.91), EASYSIMD_FLOAT32_C(   172.83), EASYSIMD_FLOAT32_C(   816.49), EASYSIMD_FLOAT32_C(   677.29) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -2.38), EASYSIMD_FLOAT32_C(   -57.26), EASYSIMD_FLOAT32_C(  -363.40),
        EASYSIMD_FLOAT32_C(   348.91), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   677.29),
        EASYSIMD_FLOAT32_C(  -788.60), EASYSIMD_FLOAT32_C(    -2.38), EASYSIMD_FLOAT32_C(   -57.26), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   348.91), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   816.49), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(53747),
      { EASYSIMD_FLOAT32_C(   -74.10), EASYSIMD_FLOAT32_C(   628.20), EASYSIMD_FLOAT32_C(   176.40), EASYSIMD_FLOAT32_C(   789.58),
        EASYSIMD_FLOAT32_C(   434.02), EASYSIMD_FLOAT32_C(   537.30), EASYSIMD_FLOAT32_C(   360.66), EASYSIMD_FLOAT32_C(  -306.64) },
      { EASYSIMD_FLOAT32_C(   -74.10), EASYSIMD_FLOAT32_C(   628.20), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   434.02), EASYSIMD_FLOAT32_C(   537.30), EASYSIMD_FLOAT32_C(   360.66), EASYSIMD_FLOAT32_C(  -306.64),
        EASYSIMD_FLOAT32_C(   -74.10), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   434.02), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   360.66), EASYSIMD_FLOAT32_C(  -306.64) } },
    { UINT16_C(57660),
      { EASYSIMD_FLOAT32_C(   529.43), EASYSIMD_FLOAT32_C(   185.72), EASYSIMD_FLOAT32_C(  -666.37), EASYSIMD_FLOAT32_C(   372.37),
        EASYSIMD_FLOAT32_C(   420.53), EASYSIMD_FLOAT32_C(   -76.09), EASYSIMD_FLOAT32_C(  -764.18), EASYSIMD_FLOAT32_C(   472.62) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -666.37), EASYSIMD_FLOAT32_C(   372.37),
        EASYSIMD_FLOAT32_C(   420.53), EASYSIMD_FLOAT32_C(   -76.09), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   529.43), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -76.09), EASYSIMD_FLOAT32_C(  -764.18), EASYSIMD_FLOAT32_C(   472.62) } },
    { UINT16_C(60506),
      { EASYSIMD_FLOAT32_C(  -796.21), EASYSIMD_FLOAT32_C(   148.32), EASYSIMD_FLOAT32_C(   781.59), EASYSIMD_FLOAT32_C(   218.77),
        EASYSIMD_FLOAT32_C(   802.35), EASYSIMD_FLOAT32_C(  -915.03), EASYSIMD_FLOAT32_C(  -953.21), EASYSIMD_FLOAT32_C(  -530.25) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   148.32), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   218.77),
        EASYSIMD_FLOAT32_C(   802.35), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -953.21), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   781.59), EASYSIMD_FLOAT32_C(   218.77),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -915.03), EASYSIMD_FLOAT32_C(  -953.21), EASYSIMD_FLOAT32_C(  -530.25) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_broadcast_f32x8(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_broadcast_f32x8");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcast_f64x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   715.32), EASYSIMD_FLOAT64_C(   317.58) },
      { EASYSIMD_FLOAT64_C(   715.32), EASYSIMD_FLOAT64_C(   317.58), EASYSIMD_FLOAT64_C(   715.32), EASYSIMD_FLOAT64_C(   317.58),
        EASYSIMD_FLOAT64_C(   715.32), EASYSIMD_FLOAT64_C(   317.58), EASYSIMD_FLOAT64_C(   715.32), EASYSIMD_FLOAT64_C(   317.58) } },
    { { EASYSIMD_FLOAT64_C(  -404.76), EASYSIMD_FLOAT64_C(  -835.80) },
      { EASYSIMD_FLOAT64_C(  -404.76), EASYSIMD_FLOAT64_C(  -835.80), EASYSIMD_FLOAT64_C(  -404.76), EASYSIMD_FLOAT64_C(  -835.80),
        EASYSIMD_FLOAT64_C(  -404.76), EASYSIMD_FLOAT64_C(  -835.80), EASYSIMD_FLOAT64_C(  -404.76), EASYSIMD_FLOAT64_C(  -835.80) } },
    { { EASYSIMD_FLOAT64_C(   653.97), EASYSIMD_FLOAT64_C(  -774.97) },
      { EASYSIMD_FLOAT64_C(   653.97), EASYSIMD_FLOAT64_C(  -774.97), EASYSIMD_FLOAT64_C(   653.97), EASYSIMD_FLOAT64_C(  -774.97),
        EASYSIMD_FLOAT64_C(   653.97), EASYSIMD_FLOAT64_C(  -774.97), EASYSIMD_FLOAT64_C(   653.97), EASYSIMD_FLOAT64_C(  -774.97) } },
    { { EASYSIMD_FLOAT64_C(  -843.04), EASYSIMD_FLOAT64_C(  -900.71) },
      { EASYSIMD_FLOAT64_C(  -843.04), EASYSIMD_FLOAT64_C(  -900.71), EASYSIMD_FLOAT64_C(  -843.04), EASYSIMD_FLOAT64_C(  -900.71),
        EASYSIMD_FLOAT64_C(  -843.04), EASYSIMD_FLOAT64_C(  -900.71), EASYSIMD_FLOAT64_C(  -843.04), EASYSIMD_FLOAT64_C(  -900.71) } },
    { { EASYSIMD_FLOAT64_C(  -197.71), EASYSIMD_FLOAT64_C(  -989.91) },
      { EASYSIMD_FLOAT64_C(  -197.71), EASYSIMD_FLOAT64_C(  -989.91), EASYSIMD_FLOAT64_C(  -197.71), EASYSIMD_FLOAT64_C(  -989.91),
        EASYSIMD_FLOAT64_C(  -197.71), EASYSIMD_FLOAT64_C(  -989.91), EASYSIMD_FLOAT64_C(  -197.71), EASYSIMD_FLOAT64_C(  -989.91) } },
    { { EASYSIMD_FLOAT64_C(   515.43), EASYSIMD_FLOAT64_C(   879.19) },
      { EASYSIMD_FLOAT64_C(   515.43), EASYSIMD_FLOAT64_C(   879.19), EASYSIMD_FLOAT64_C(   515.43), EASYSIMD_FLOAT64_C(   879.19),
        EASYSIMD_FLOAT64_C(   515.43), EASYSIMD_FLOAT64_C(   879.19), EASYSIMD_FLOAT64_C(   515.43), EASYSIMD_FLOAT64_C(   879.19) } },
    { { EASYSIMD_FLOAT64_C(   610.61), EASYSIMD_FLOAT64_C(   540.00) },
      { EASYSIMD_FLOAT64_C(   610.61), EASYSIMD_FLOAT64_C(   540.00), EASYSIMD_FLOAT64_C(   610.61), EASYSIMD_FLOAT64_C(   540.00),
        EASYSIMD_FLOAT64_C(   610.61), EASYSIMD_FLOAT64_C(   540.00), EASYSIMD_FLOAT64_C(   610.61), EASYSIMD_FLOAT64_C(   540.00) } },
    { { EASYSIMD_FLOAT64_C(  -234.86), EASYSIMD_FLOAT64_C(   751.29) },
      { EASYSIMD_FLOAT64_C(  -234.86), EASYSIMD_FLOAT64_C(   751.29), EASYSIMD_FLOAT64_C(  -234.86), EASYSIMD_FLOAT64_C(   751.29),
        EASYSIMD_FLOAT64_C(  -234.86), EASYSIMD_FLOAT64_C(   751.29), EASYSIMD_FLOAT64_C(  -234.86), EASYSIMD_FLOAT64_C(   751.29) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_broadcast_f64x2(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_broadcast_f64x2");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcast_f64x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 src[8];
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   749.13), EASYSIMD_FLOAT64_C(   675.39), EASYSIMD_FLOAT64_C(  -739.63), EASYSIMD_FLOAT64_C(  -916.53),
        EASYSIMD_FLOAT64_C(   -70.94), EASYSIMD_FLOAT64_C(  -224.46), EASYSIMD_FLOAT64_C(  -485.72), EASYSIMD_FLOAT64_C(   433.96) },
      UINT8_C(250),
      { EASYSIMD_FLOAT64_C(   424.19), EASYSIMD_FLOAT64_C(  -720.98) },
      { EASYSIMD_FLOAT64_C(   749.13), EASYSIMD_FLOAT64_C(  -720.98), EASYSIMD_FLOAT64_C(  -739.63), EASYSIMD_FLOAT64_C(  -720.98),
        EASYSIMD_FLOAT64_C(   424.19), EASYSIMD_FLOAT64_C(  -720.98), EASYSIMD_FLOAT64_C(   424.19), EASYSIMD_FLOAT64_C(  -720.98) } },
    { { EASYSIMD_FLOAT64_C(   461.33), EASYSIMD_FLOAT64_C(  -402.24), EASYSIMD_FLOAT64_C(  -437.75), EASYSIMD_FLOAT64_C(   785.96),
        EASYSIMD_FLOAT64_C(  -372.46), EASYSIMD_FLOAT64_C(   110.74), EASYSIMD_FLOAT64_C(  -831.39), EASYSIMD_FLOAT64_C(   846.99) },
      UINT8_C( 78),
      { EASYSIMD_FLOAT64_C(  -572.48), EASYSIMD_FLOAT64_C(   394.61) },
      { EASYSIMD_FLOAT64_C(   461.33), EASYSIMD_FLOAT64_C(   394.61), EASYSIMD_FLOAT64_C(  -572.48), EASYSIMD_FLOAT64_C(   394.61),
        EASYSIMD_FLOAT64_C(  -372.46), EASYSIMD_FLOAT64_C(   110.74), EASYSIMD_FLOAT64_C(  -572.48), EASYSIMD_FLOAT64_C(   846.99) } },
    { { EASYSIMD_FLOAT64_C(   215.35), EASYSIMD_FLOAT64_C(  -616.54), EASYSIMD_FLOAT64_C(  -262.30), EASYSIMD_FLOAT64_C(  -426.39),
        EASYSIMD_FLOAT64_C(  -336.22), EASYSIMD_FLOAT64_C(  -839.02), EASYSIMD_FLOAT64_C(   672.49), EASYSIMD_FLOAT64_C(   589.70) },
      UINT8_C(163),
      { EASYSIMD_FLOAT64_C(  -982.23), EASYSIMD_FLOAT64_C(  -416.77) },
      { EASYSIMD_FLOAT64_C(  -982.23), EASYSIMD_FLOAT64_C(  -416.77), EASYSIMD_FLOAT64_C(  -262.30), EASYSIMD_FLOAT64_C(  -426.39),
        EASYSIMD_FLOAT64_C(  -336.22), EASYSIMD_FLOAT64_C(  -416.77), EASYSIMD_FLOAT64_C(   672.49), EASYSIMD_FLOAT64_C(  -416.77) } },
    { { EASYSIMD_FLOAT64_C(  -578.35), EASYSIMD_FLOAT64_C(  -267.73), EASYSIMD_FLOAT64_C(   242.90), EASYSIMD_FLOAT64_C(   449.74),
        EASYSIMD_FLOAT64_C(   714.62), EASYSIMD_FLOAT64_C(   671.90), EASYSIMD_FLOAT64_C(   577.25), EASYSIMD_FLOAT64_C(   -88.86) },
      UINT8_C(222),
      { EASYSIMD_FLOAT64_C(   379.16), EASYSIMD_FLOAT64_C(   573.95) },
      { EASYSIMD_FLOAT64_C(  -578.35), EASYSIMD_FLOAT64_C(   573.95), EASYSIMD_FLOAT64_C(   379.16), EASYSIMD_FLOAT64_C(   573.95),
        EASYSIMD_FLOAT64_C(   379.16), EASYSIMD_FLOAT64_C(   671.90), EASYSIMD_FLOAT64_C(   379.16), EASYSIMD_FLOAT64_C(   573.95) } },
    { { EASYSIMD_FLOAT64_C(   428.10), EASYSIMD_FLOAT64_C(  -969.60), EASYSIMD_FLOAT64_C(  -117.58), EASYSIMD_FLOAT64_C(  -121.88),
        EASYSIMD_FLOAT64_C(  -513.12), EASYSIMD_FLOAT64_C(   -67.52), EASYSIMD_FLOAT64_C(  -880.81), EASYSIMD_FLOAT64_C(   257.25) },
      UINT8_C( 35),
      { EASYSIMD_FLOAT64_C(   -71.92), EASYSIMD_FLOAT64_C(  -682.64) },
      { EASYSIMD_FLOAT64_C(   -71.92), EASYSIMD_FLOAT64_C(  -682.64), EASYSIMD_FLOAT64_C(  -117.58), EASYSIMD_FLOAT64_C(  -121.88),
        EASYSIMD_FLOAT64_C(  -513.12), EASYSIMD_FLOAT64_C(  -682.64), EASYSIMD_FLOAT64_C(  -880.81), EASYSIMD_FLOAT64_C(   257.25) } },
    { { EASYSIMD_FLOAT64_C(   858.06), EASYSIMD_FLOAT64_C(  -576.56), EASYSIMD_FLOAT64_C(  -199.04), EASYSIMD_FLOAT64_C(   741.89),
        EASYSIMD_FLOAT64_C(   940.66), EASYSIMD_FLOAT64_C(  -320.73), EASYSIMD_FLOAT64_C(  -519.45), EASYSIMD_FLOAT64_C(  -359.73) },
      UINT8_C( 14),
      { EASYSIMD_FLOAT64_C(  -260.24), EASYSIMD_FLOAT64_C(   150.09) },
      { EASYSIMD_FLOAT64_C(   858.06), EASYSIMD_FLOAT64_C(   150.09), EASYSIMD_FLOAT64_C(  -260.24), EASYSIMD_FLOAT64_C(   150.09),
        EASYSIMD_FLOAT64_C(   940.66), EASYSIMD_FLOAT64_C(  -320.73), EASYSIMD_FLOAT64_C(  -519.45), EASYSIMD_FLOAT64_C(  -359.73) } },
    { { EASYSIMD_FLOAT64_C(   508.76), EASYSIMD_FLOAT64_C(   671.76), EASYSIMD_FLOAT64_C(   188.22), EASYSIMD_FLOAT64_C(  -524.84),
        EASYSIMD_FLOAT64_C(   958.74), EASYSIMD_FLOAT64_C(  -408.21), EASYSIMD_FLOAT64_C(  -756.34), EASYSIMD_FLOAT64_C(   260.63) },
      UINT8_C( 48),
      { EASYSIMD_FLOAT64_C(  -287.86), EASYSIMD_FLOAT64_C(   -66.95) },
      { EASYSIMD_FLOAT64_C(   508.76), EASYSIMD_FLOAT64_C(   671.76), EASYSIMD_FLOAT64_C(   188.22), EASYSIMD_FLOAT64_C(  -524.84),
        EASYSIMD_FLOAT64_C(  -287.86), EASYSIMD_FLOAT64_C(   -66.95), EASYSIMD_FLOAT64_C(  -756.34), EASYSIMD_FLOAT64_C(   260.63) } },
    { { EASYSIMD_FLOAT64_C(   741.62), EASYSIMD_FLOAT64_C(   389.31), EASYSIMD_FLOAT64_C(  -806.05), EASYSIMD_FLOAT64_C(   761.48),
        EASYSIMD_FLOAT64_C(   242.55), EASYSIMD_FLOAT64_C(   550.14), EASYSIMD_FLOAT64_C(   214.54), EASYSIMD_FLOAT64_C(  -176.03) },
      UINT8_C( 79),
      { EASYSIMD_FLOAT64_C(   639.90), EASYSIMD_FLOAT64_C(   881.52) },
      { EASYSIMD_FLOAT64_C(   639.90), EASYSIMD_FLOAT64_C(   881.52), EASYSIMD_FLOAT64_C(   639.90), EASYSIMD_FLOAT64_C(   881.52),
        EASYSIMD_FLOAT64_C(   242.55), EASYSIMD_FLOAT64_C(   550.14), EASYSIMD_FLOAT64_C(   639.90), EASYSIMD_FLOAT64_C(  -176.03) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d src = easysimd_mm512_loadu_pd(test_vec[i].src);
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_broadcast_f64x2(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_broadcast_f64x2");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcast_f64x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { UINT8_C( 32),
      { EASYSIMD_FLOAT64_C(    95.43), EASYSIMD_FLOAT64_C(  -111.80) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -111.80), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(212),
      { EASYSIMD_FLOAT64_C(   159.26), EASYSIMD_FLOAT64_C(   721.63) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   159.26), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   159.26), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   159.26), EASYSIMD_FLOAT64_C(   721.63) } },
    { UINT8_C(232),
      { EASYSIMD_FLOAT64_C(   -41.02), EASYSIMD_FLOAT64_C(   592.81) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   592.81),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   592.81), EASYSIMD_FLOAT64_C(   -41.02), EASYSIMD_FLOAT64_C(   592.81) } },
    { UINT8_C(112),
      { EASYSIMD_FLOAT64_C(    80.26), EASYSIMD_FLOAT64_C(   969.51) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(    80.26), EASYSIMD_FLOAT64_C(   969.51), EASYSIMD_FLOAT64_C(    80.26), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(215),
      { EASYSIMD_FLOAT64_C(   905.16), EASYSIMD_FLOAT64_C(  -968.55) },
      { EASYSIMD_FLOAT64_C(   905.16), EASYSIMD_FLOAT64_C(  -968.55), EASYSIMD_FLOAT64_C(   905.16), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   905.16), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   905.16), EASYSIMD_FLOAT64_C(  -968.55) } },
    { UINT8_C(135),
      { EASYSIMD_FLOAT64_C(   140.43), EASYSIMD_FLOAT64_C(   267.82) },
      { EASYSIMD_FLOAT64_C(   140.43), EASYSIMD_FLOAT64_C(   267.82), EASYSIMD_FLOAT64_C(   140.43), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   267.82) } },
    { UINT8_C(192),
      { EASYSIMD_FLOAT64_C(  -853.88), EASYSIMD_FLOAT64_C(   811.68) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -853.88), EASYSIMD_FLOAT64_C(   811.68) } },
    { UINT8_C( 17),
      { EASYSIMD_FLOAT64_C(  -661.24), EASYSIMD_FLOAT64_C(   561.84) },
      { EASYSIMD_FLOAT64_C(  -661.24), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(  -661.24), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_broadcast_f64x2(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_broadcast_f64x2");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm256_broadcast_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[4];
    int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(   763603352), -INT32_C(   240189883), -INT32_C(   961552276),  INT32_C(  1212572688) },
      {  INT32_C(   763603352), -INT32_C(   240189883), -INT32_C(   961552276),  INT32_C(  1212572688),  INT32_C(   763603352), -INT32_C(   240189883), -INT32_C(   961552276),  INT32_C(  1212572688) } },
    { { -INT32_C(   517854272),  INT32_C(   835986442), -INT32_C(   340743052),  INT32_C(  1912083673) },
      { -INT32_C(   517854272),  INT32_C(   835986442), -INT32_C(   340743052),  INT32_C(  1912083673), -INT32_C(   517854272),  INT32_C(   835986442), -INT32_C(   340743052),  INT32_C(  1912083673) } },
    { {  INT32_C(    44006333), -INT32_C(   420262534),  INT32_C(  1034723885), -INT32_C(   947457273) },
      {  INT32_C(    44006333), -INT32_C(   420262534),  INT32_C(  1034723885), -INT32_C(   947457273),  INT32_C(    44006333), -INT32_C(   420262534),  INT32_C(  1034723885), -INT32_C(   947457273) } },
    { {  INT32_C(   682207262),  INT32_C(  1096383949),  INT32_C(    53217578), -INT32_C(   629857251) },
      {  INT32_C(   682207262),  INT32_C(  1096383949),  INT32_C(    53217578), -INT32_C(   629857251),  INT32_C(   682207262),  INT32_C(  1096383949),  INT32_C(    53217578), -INT32_C(   629857251) } },
    { {  INT32_C(   450630816), -INT32_C(  1912549535),  INT32_C(  2043391346), -INT32_C(  1103081056) },
      {  INT32_C(   450630816), -INT32_C(  1912549535),  INT32_C(  2043391346), -INT32_C(  1103081056),  INT32_C(   450630816), -INT32_C(  1912549535),  INT32_C(  2043391346), -INT32_C(  1103081056) } },
    { { -INT32_C(   941168134), -INT32_C(  1861730201),  INT32_C(  1737765961), -INT32_C(   113178279) },
      { -INT32_C(   941168134), -INT32_C(  1861730201),  INT32_C(  1737765961), -INT32_C(   113178279), -INT32_C(   941168134), -INT32_C(  1861730201),  INT32_C(  1737765961), -INT32_C(   113178279) } },
    { {  INT32_C(  2115182109),  INT32_C(  1594627053),  INT32_C(  1624824000),  INT32_C(   589175081) },
      {  INT32_C(  2115182109),  INT32_C(  1594627053),  INT32_C(  1624824000),  INT32_C(   589175081),  INT32_C(  2115182109),  INT32_C(  1594627053),  INT32_C(  1624824000),  INT32_C(   589175081) } },
    { {  INT32_C(  1776944386), -INT32_C(  1896156603), -INT32_C(  2131390681), -INT32_C(  1233569896) },
      {  INT32_C(  1776944386), -INT32_C(  1896156603), -INT32_C(  2131390681), -INT32_C(  1233569896),  INT32_C(  1776944386), -INT32_C(  1896156603), -INT32_C(  2131390681), -INT32_C(  1233569896) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_broadcast_i32x4(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_broadcast_i32x4");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m256i r = easysimd_mm256_broadcast_i32x4(a);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_broadcast_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[8];
    uint8_t k;
    int32_t a[4];
    int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(   796052905),  INT32_C(    54514933), -INT32_C(   872615034), -INT32_C(     9558165), -INT32_C(  1691923527),  INT32_C(   110280943),  INT32_C(   270096342),  INT32_C(  1251283442) },
      UINT8_C( 67),
      { -INT32_C(   164095454),  INT32_C(  1283275609),  INT32_C(  1605912632),  INT32_C(    18462646) },
      { -INT32_C(   164095454),  INT32_C(  1283275609), -INT32_C(   872615034), -INT32_C(     9558165), -INT32_C(  1691923527),  INT32_C(   110280943),  INT32_C(  1605912632),  INT32_C(  1251283442) } },
    { { -INT32_C(  1628326690), -INT32_C(  1636436154),  INT32_C(   563119376),  INT32_C(  1013242394),  INT32_C(  1311939828),  INT32_C(   278573016), -INT32_C(  1385213193), -INT32_C(   407926775) },
      UINT8_C( 60),
      { -INT32_C(  1752988000), -INT32_C(  2136464901), -INT32_C(  1936012879), -INT32_C(   897526226) },
      { -INT32_C(  1628326690), -INT32_C(  1636436154), -INT32_C(  1936012879), -INT32_C(   897526226), -INT32_C(  1752988000), -INT32_C(  2136464901), -INT32_C(  1385213193), -INT32_C(   407926775) } },
    { { -INT32_C(  1197289976), -INT32_C(  1146113431), -INT32_C(  1429906142), -INT32_C(  1394103284),  INT32_C(   759392818),  INT32_C(  1018030987), -INT32_C(   473413707),  INT32_C(   648890653) },
      UINT8_C( 23),
      {  INT32_C(    41999952), -INT32_C(   366723955), -INT32_C(  1376334079),  INT32_C(   551592630) },
      {  INT32_C(    41999952), -INT32_C(   366723955), -INT32_C(  1376334079), -INT32_C(  1394103284),  INT32_C(    41999952),  INT32_C(  1018030987), -INT32_C(   473413707),  INT32_C(   648890653) } },
    { { -INT32_C(   777318938),  INT32_C(    42395579), -INT32_C(   115316304),  INT32_C(  1729119767), -INT32_C(  1318481628), -INT32_C(   828600627),  INT32_C(   326865501),  INT32_C(   456350517) },
      UINT8_C(105),
      { -INT32_C(   987435810), -INT32_C(   596302221), -INT32_C(  1930203578),  INT32_C(   279992959) },
      { -INT32_C(   987435810),  INT32_C(    42395579), -INT32_C(   115316304),  INT32_C(   279992959), -INT32_C(  1318481628), -INT32_C(   596302221), -INT32_C(  1930203578),  INT32_C(   456350517) } },
    { {  INT32_C(  1390240452), -INT32_C(  1867535362), -INT32_C(  2100968922), -INT32_C(   739516171),  INT32_C(  1083707341),  INT32_C(  2082213429), -INT32_C(    83357572),  INT32_C(   772520298) },
      UINT8_C( 27),
      { -INT32_C(  1827045144), -INT32_C(   239425233),  INT32_C(  1340488815),  INT32_C(   907852071) },
      { -INT32_C(  1827045144), -INT32_C(   239425233), -INT32_C(  2100968922),  INT32_C(   907852071), -INT32_C(  1827045144),  INT32_C(  2082213429), -INT32_C(    83357572),  INT32_C(   772520298) } },
    { {  INT32_C(  1617648722), -INT32_C(  1982011527), -INT32_C(  1443637008), -INT32_C(   859561245), -INT32_C(   765469278), -INT32_C(   171763322),  INT32_C(  2084940373), -INT32_C(  1246600861) },
      UINT8_C(190),
      {  INT32_C(    87495966), -INT32_C(   889863950), -INT32_C(   710041933),  INT32_C(  1064860002) },
      {  INT32_C(  1617648722), -INT32_C(   889863950), -INT32_C(   710041933),  INT32_C(  1064860002),  INT32_C(    87495966), -INT32_C(   889863950),  INT32_C(  2084940373),  INT32_C(  1064860002) } },
    { { -INT32_C(   221885735), -INT32_C(  1219970291),  INT32_C(  1645986816), -INT32_C(  1793011593), -INT32_C(   660973594), -INT32_C(   895315945), -INT32_C(  1851764946), -INT32_C(  1563420471) },
      UINT8_C( 98),
      {  INT32_C(  1383044246), -INT32_C(  1588451364),  INT32_C(   303608898),  INT32_C(   737717716) },
      { -INT32_C(   221885735), -INT32_C(  1588451364),  INT32_C(  1645986816), -INT32_C(  1793011593), -INT32_C(   660973594), -INT32_C(  1588451364),  INT32_C(   303608898), -INT32_C(  1563420471) } },
    { { -INT32_C(   683487161), -INT32_C(  1039790734), -INT32_C(   980707411), -INT32_C(    30986905), -INT32_C(  1638885695), -INT32_C(    12606787),  INT32_C(   722622295),  INT32_C(  1263995396) },
      UINT8_C(219),
      { -INT32_C(  1504894055), -INT32_C(  1068298455), -INT32_C(   953739110), -INT32_C(   695655105) },
      { -INT32_C(  1504894055), -INT32_C(  1068298455), -INT32_C(   980707411), -INT32_C(   695655105), -INT32_C(  1504894055), -INT32_C(    12606787), -INT32_C(   953739110), -INT32_C(   695655105) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_broadcast_i32x4(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_broadcast_i32x4");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m256i r = easysimd_mm256_mask_broadcast_i32x4(src, k, a);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_broadcast_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a[4];
    int32_t r[8];
  } test_vec[] = {
    { UINT8_C(184),
      { -INT32_C(   531867946), -INT32_C(   208982269), -INT32_C(   991738166),  INT32_C(  1343714841) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1343714841), -INT32_C(   531867946), -INT32_C(   208982269),  INT32_C(           0),  INT32_C(  1343714841) } },
    { UINT8_C(208),
      {  INT32_C(   616142666),  INT32_C(  1990685437), -INT32_C(  1231090783), -INT32_C(  1920182217) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   616142666),  INT32_C(           0), -INT32_C(  1231090783), -INT32_C(  1920182217) } },
    { UINT8_C(164),
      { -INT32_C(   120352660),  INT32_C(  1732156804),  INT32_C(  2027771745),  INT32_C(  1052945831) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  2027771745),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1732156804),  INT32_C(           0),  INT32_C(  1052945831) } },
    { UINT8_C(102),
      { -INT32_C(  1898431513),  INT32_C(  1369012914), -INT32_C(  1129721552), -INT32_C(   282505890) },
      {  INT32_C(           0),  INT32_C(  1369012914), -INT32_C(  1129721552),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1369012914), -INT32_C(  1129721552),  INT32_C(           0) } },
    { UINT8_C( 32),
      {  INT32_C(  1606251297), -INT32_C(  1212801318),  INT32_C(  1499750039), -INT32_C(   666842212) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1212801318),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(163),
      {  INT32_C(  1746766542), -INT32_C(  2059842085),  INT32_C(   869439242), -INT32_C(   111873146) },
      {  INT32_C(  1746766542), -INT32_C(  2059842085),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2059842085),  INT32_C(           0), -INT32_C(   111873146) } },
    { UINT8_C(176),
      {  INT32_C(  1775227827), -INT32_C(   272144758),  INT32_C(    12215487), -INT32_C(   959554244) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1775227827), -INT32_C(   272144758),  INT32_C(           0), -INT32_C(   959554244) } },
    { UINT8_C(123),
      {  INT32_C(  1875419446), -INT32_C(   116927962),  INT32_C(  1508674821),  INT32_C(  1510775943) },
      {  INT32_C(  1875419446), -INT32_C(   116927962),  INT32_C(           0),  INT32_C(  1510775943),  INT32_C(  1875419446), -INT32_C(   116927962),  INT32_C(  1508674821),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_broadcast_i32x4(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_broadcast_i32x4");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m256i r = easysimd_mm256_maskz_broadcast_i32x4(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_broadcast_f32x4 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -558.39), EASYSIMD_FLOAT32_C(  -943.50), EASYSIMD_FLOAT32_C(   652.52), EASYSIMD_FLOAT32_C(   945.52) },
      { EASYSIMD_FLOAT32_C(  -558.39), EASYSIMD_FLOAT32_C(  -943.50), EASYSIMD_FLOAT32_C(   652.52), EASYSIMD_FLOAT32_C(   945.52),
        EASYSIMD_FLOAT32_C(  -558.39), EASYSIMD_FLOAT32_C(  -943.50), EASYSIMD_FLOAT32_C(   652.52), EASYSIMD_FLOAT32_C(   945.52) } },
    { { EASYSIMD_FLOAT32_C(  -577.06), EASYSIMD_FLOAT32_C(  -623.59), EASYSIMD_FLOAT32_C(  -742.48), EASYSIMD_FLOAT32_C(  -807.52) },
      { EASYSIMD_FLOAT32_C(  -577.06), EASYSIMD_FLOAT32_C(  -623.59), EASYSIMD_FLOAT32_C(  -742.48), EASYSIMD_FLOAT32_C(  -807.52),
        EASYSIMD_FLOAT32_C(  -577.06), EASYSIMD_FLOAT32_C(  -623.59), EASYSIMD_FLOAT32_C(  -742.48), EASYSIMD_FLOAT32_C(  -807.52) } },
    { { EASYSIMD_FLOAT32_C(     0.46), EASYSIMD_FLOAT32_C(  -222.35), EASYSIMD_FLOAT32_C(   965.41), EASYSIMD_FLOAT32_C(  -320.94) },
      { EASYSIMD_FLOAT32_C(     0.46), EASYSIMD_FLOAT32_C(  -222.35), EASYSIMD_FLOAT32_C(   965.41), EASYSIMD_FLOAT32_C(  -320.94),
        EASYSIMD_FLOAT32_C(     0.46), EASYSIMD_FLOAT32_C(  -222.35), EASYSIMD_FLOAT32_C(   965.41), EASYSIMD_FLOAT32_C(  -320.94) } },
    { { EASYSIMD_FLOAT32_C(    34.85), EASYSIMD_FLOAT32_C(  -238.64), EASYSIMD_FLOAT32_C(  -834.61), EASYSIMD_FLOAT32_C(   763.48) },
      { EASYSIMD_FLOAT32_C(    34.85), EASYSIMD_FLOAT32_C(  -238.64), EASYSIMD_FLOAT32_C(  -834.61), EASYSIMD_FLOAT32_C(   763.48),
        EASYSIMD_FLOAT32_C(    34.85), EASYSIMD_FLOAT32_C(  -238.64), EASYSIMD_FLOAT32_C(  -834.61), EASYSIMD_FLOAT32_C(   763.48) } },
    { { EASYSIMD_FLOAT32_C(  -215.99), EASYSIMD_FLOAT32_C(  -214.29), EASYSIMD_FLOAT32_C(   432.66), EASYSIMD_FLOAT32_C(  -222.94) },
      { EASYSIMD_FLOAT32_C(  -215.99), EASYSIMD_FLOAT32_C(  -214.29), EASYSIMD_FLOAT32_C(   432.66), EASYSIMD_FLOAT32_C(  -222.94),
        EASYSIMD_FLOAT32_C(  -215.99), EASYSIMD_FLOAT32_C(  -214.29), EASYSIMD_FLOAT32_C(   432.66), EASYSIMD_FLOAT32_C(  -222.94) } },
    { { EASYSIMD_FLOAT32_C(  -994.85), EASYSIMD_FLOAT32_C(  -413.17), EASYSIMD_FLOAT32_C(  -100.86), EASYSIMD_FLOAT32_C(   836.37) },
      { EASYSIMD_FLOAT32_C(  -994.85), EASYSIMD_FLOAT32_C(  -413.17), EASYSIMD_FLOAT32_C(  -100.86), EASYSIMD_FLOAT32_C(   836.37),
        EASYSIMD_FLOAT32_C(  -994.85), EASYSIMD_FLOAT32_C(  -413.17), EASYSIMD_FLOAT32_C(  -100.86), EASYSIMD_FLOAT32_C(   836.37) } },
    { { EASYSIMD_FLOAT32_C(   809.63), EASYSIMD_FLOAT32_C(  -520.84), EASYSIMD_FLOAT32_C(   265.00), EASYSIMD_FLOAT32_C(  -111.67) },
      { EASYSIMD_FLOAT32_C(   809.63), EASYSIMD_FLOAT32_C(  -520.84), EASYSIMD_FLOAT32_C(   265.00), EASYSIMD_FLOAT32_C(  -111.67),
        EASYSIMD_FLOAT32_C(   809.63), EASYSIMD_FLOAT32_C(  -520.84), EASYSIMD_FLOAT32_C(   265.00), EASYSIMD_FLOAT32_C(  -111.67) } },
    { { EASYSIMD_FLOAT32_C(  -855.41), EASYSIMD_FLOAT32_C(  -875.73), EASYSIMD_FLOAT32_C(  -447.77), EASYSIMD_FLOAT32_C(   263.25) },
      { EASYSIMD_FLOAT32_C(  -855.41), EASYSIMD_FLOAT32_C(  -875.73), EASYSIMD_FLOAT32_C(  -447.77), EASYSIMD_FLOAT32_C(   263.25),
        EASYSIMD_FLOAT32_C(  -855.41), EASYSIMD_FLOAT32_C(  -875.73), EASYSIMD_FLOAT32_C(  -447.77), EASYSIMD_FLOAT32_C(   263.25) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_broadcast_f32x4(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_broadcast_f32x4");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_broadcast_f32x4 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 src[8];
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   113.43), EASYSIMD_FLOAT32_C(   410.75), EASYSIMD_FLOAT32_C(  -451.88), EASYSIMD_FLOAT32_C(  -725.60),
        EASYSIMD_FLOAT32_C(   921.94), EASYSIMD_FLOAT32_C(  -987.53), EASYSIMD_FLOAT32_C(   590.45), EASYSIMD_FLOAT32_C(  -298.17) },
      UINT8_C( 50),
      { EASYSIMD_FLOAT32_C(    44.31), EASYSIMD_FLOAT32_C(   797.52), EASYSIMD_FLOAT32_C(  -107.60), EASYSIMD_FLOAT32_C(  -484.17) },
      { EASYSIMD_FLOAT32_C(   113.43), EASYSIMD_FLOAT32_C(   797.52), EASYSIMD_FLOAT32_C(  -451.88), EASYSIMD_FLOAT32_C(  -725.60),
        EASYSIMD_FLOAT32_C(    44.31), EASYSIMD_FLOAT32_C(   797.52), EASYSIMD_FLOAT32_C(   590.45), EASYSIMD_FLOAT32_C(  -298.17) } },
    { { EASYSIMD_FLOAT32_C(   556.86), EASYSIMD_FLOAT32_C(  -797.02), EASYSIMD_FLOAT32_C(   402.24), EASYSIMD_FLOAT32_C(   441.25),
        EASYSIMD_FLOAT32_C(   142.97), EASYSIMD_FLOAT32_C(   883.64), EASYSIMD_FLOAT32_C(  -635.48), EASYSIMD_FLOAT32_C(  -488.89) },
      UINT8_C(165),
      { EASYSIMD_FLOAT32_C(  -333.03), EASYSIMD_FLOAT32_C(   703.87), EASYSIMD_FLOAT32_C(   -69.82), EASYSIMD_FLOAT32_C(   527.07) },
      { EASYSIMD_FLOAT32_C(  -333.03), EASYSIMD_FLOAT32_C(  -797.02), EASYSIMD_FLOAT32_C(   -69.82), EASYSIMD_FLOAT32_C(   441.25),
        EASYSIMD_FLOAT32_C(   142.97), EASYSIMD_FLOAT32_C(   703.87), EASYSIMD_FLOAT32_C(  -635.48), EASYSIMD_FLOAT32_C(   527.07) } },
    { { EASYSIMD_FLOAT32_C(   425.48), EASYSIMD_FLOAT32_C(   960.83), EASYSIMD_FLOAT32_C(   698.87), EASYSIMD_FLOAT32_C(  -175.48),
        EASYSIMD_FLOAT32_C(   789.83), EASYSIMD_FLOAT32_C(   633.19), EASYSIMD_FLOAT32_C(    85.22), EASYSIMD_FLOAT32_C(   351.45) },
      UINT8_C(206),
      { EASYSIMD_FLOAT32_C(   362.09), EASYSIMD_FLOAT32_C(  -387.94), EASYSIMD_FLOAT32_C(   -58.09), EASYSIMD_FLOAT32_C(  -381.37) },
      { EASYSIMD_FLOAT32_C(   425.48), EASYSIMD_FLOAT32_C(  -387.94), EASYSIMD_FLOAT32_C(   -58.09), EASYSIMD_FLOAT32_C(  -381.37),
        EASYSIMD_FLOAT32_C(   789.83), EASYSIMD_FLOAT32_C(   633.19), EASYSIMD_FLOAT32_C(   -58.09), EASYSIMD_FLOAT32_C(  -381.37) } },
    { { EASYSIMD_FLOAT32_C(   385.81), EASYSIMD_FLOAT32_C(   368.14), EASYSIMD_FLOAT32_C(  -607.80), EASYSIMD_FLOAT32_C(   623.02),
        EASYSIMD_FLOAT32_C(  -955.44), EASYSIMD_FLOAT32_C(  -138.05), EASYSIMD_FLOAT32_C(  -245.78), EASYSIMD_FLOAT32_C(  -750.22) },
      UINT8_C(110),
      { EASYSIMD_FLOAT32_C(   548.54), EASYSIMD_FLOAT32_C(  -618.32), EASYSIMD_FLOAT32_C(  -113.43), EASYSIMD_FLOAT32_C(  -437.94) },
      { EASYSIMD_FLOAT32_C(   385.81), EASYSIMD_FLOAT32_C(  -618.32), EASYSIMD_FLOAT32_C(  -113.43), EASYSIMD_FLOAT32_C(  -437.94),
        EASYSIMD_FLOAT32_C(  -955.44), EASYSIMD_FLOAT32_C(  -618.32), EASYSIMD_FLOAT32_C(  -113.43), EASYSIMD_FLOAT32_C(  -750.22) } },
    { { EASYSIMD_FLOAT32_C(  -510.40), EASYSIMD_FLOAT32_C(  -247.29), EASYSIMD_FLOAT32_C(  -272.50), EASYSIMD_FLOAT32_C(   154.15),
        EASYSIMD_FLOAT32_C(   745.34), EASYSIMD_FLOAT32_C(   865.17), EASYSIMD_FLOAT32_C(   893.80), EASYSIMD_FLOAT32_C(    79.97) },
      UINT8_C(108),
      { EASYSIMD_FLOAT32_C(  -178.61), EASYSIMD_FLOAT32_C(    31.69), EASYSIMD_FLOAT32_C(   669.52), EASYSIMD_FLOAT32_C(   693.51) },
      { EASYSIMD_FLOAT32_C(  -510.40), EASYSIMD_FLOAT32_C(  -247.29), EASYSIMD_FLOAT32_C(   669.52), EASYSIMD_FLOAT32_C(   693.51),
        EASYSIMD_FLOAT32_C(   745.34), EASYSIMD_FLOAT32_C(    31.69), EASYSIMD_FLOAT32_C(   669.52), EASYSIMD_FLOAT32_C(    79.97) } },
    { { EASYSIMD_FLOAT32_C(  -127.96), EASYSIMD_FLOAT32_C(  -619.72), EASYSIMD_FLOAT32_C(   284.07), EASYSIMD_FLOAT32_C(   372.86),
        EASYSIMD_FLOAT32_C(   649.51), EASYSIMD_FLOAT32_C(   278.96), EASYSIMD_FLOAT32_C(   407.00), EASYSIMD_FLOAT32_C(   484.63) },
      UINT8_C( 35),
      { EASYSIMD_FLOAT32_C(  -266.56), EASYSIMD_FLOAT32_C(  -110.85), EASYSIMD_FLOAT32_C(  -976.05), EASYSIMD_FLOAT32_C(  -446.86) },
      { EASYSIMD_FLOAT32_C(  -266.56), EASYSIMD_FLOAT32_C(  -110.85), EASYSIMD_FLOAT32_C(   284.07), EASYSIMD_FLOAT32_C(   372.86),
        EASYSIMD_FLOAT32_C(   649.51), EASYSIMD_FLOAT32_C(  -110.85), EASYSIMD_FLOAT32_C(   407.00), EASYSIMD_FLOAT32_C(   484.63) } },
    { { EASYSIMD_FLOAT32_C(  -413.34), EASYSIMD_FLOAT32_C(   993.71), EASYSIMD_FLOAT32_C(  -725.95), EASYSIMD_FLOAT32_C(   912.24),
        EASYSIMD_FLOAT32_C(    38.79), EASYSIMD_FLOAT32_C(  -113.15), EASYSIMD_FLOAT32_C(   355.83), EASYSIMD_FLOAT32_C(   489.44) },
      UINT8_C(174),
      { EASYSIMD_FLOAT32_C(   271.71), EASYSIMD_FLOAT32_C(   611.34), EASYSIMD_FLOAT32_C(   750.31), EASYSIMD_FLOAT32_C(   445.31) },
      { EASYSIMD_FLOAT32_C(  -413.34), EASYSIMD_FLOAT32_C(   611.34), EASYSIMD_FLOAT32_C(   750.31), EASYSIMD_FLOAT32_C(   445.31),
        EASYSIMD_FLOAT32_C(    38.79), EASYSIMD_FLOAT32_C(   611.34), EASYSIMD_FLOAT32_C(   355.83), EASYSIMD_FLOAT32_C(   445.31) } },
    { { EASYSIMD_FLOAT32_C(   394.72), EASYSIMD_FLOAT32_C(    -2.71), EASYSIMD_FLOAT32_C(   433.21), EASYSIMD_FLOAT32_C(   979.88),
        EASYSIMD_FLOAT32_C(   870.25), EASYSIMD_FLOAT32_C(   239.46), EASYSIMD_FLOAT32_C(   664.36), EASYSIMD_FLOAT32_C(   -21.11) },
      UINT8_C(236),
      { EASYSIMD_FLOAT32_C(    20.21), EASYSIMD_FLOAT32_C(  -364.92), EASYSIMD_FLOAT32_C(   870.25), EASYSIMD_FLOAT32_C(   218.91) },
      { EASYSIMD_FLOAT32_C(   394.72), EASYSIMD_FLOAT32_C(    -2.71), EASYSIMD_FLOAT32_C(   870.25), EASYSIMD_FLOAT32_C(   218.91),
        EASYSIMD_FLOAT32_C(   870.25), EASYSIMD_FLOAT32_C(  -364.92), EASYSIMD_FLOAT32_C(   870.25), EASYSIMD_FLOAT32_C(   218.91) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_broadcast_f32x4(src, test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_broadcast_f32x4");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_broadcast_f32x4 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(233),
      { EASYSIMD_FLOAT32_C(   749.31), EASYSIMD_FLOAT32_C(  -425.85), EASYSIMD_FLOAT32_C(   752.50), EASYSIMD_FLOAT32_C(  -794.87) },
      { EASYSIMD_FLOAT32_C(   749.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -794.87),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -425.85), EASYSIMD_FLOAT32_C(   752.50), EASYSIMD_FLOAT32_C(  -794.87) } },
    { UINT8_C(237),
      { EASYSIMD_FLOAT32_C(   236.00), EASYSIMD_FLOAT32_C(   493.54), EASYSIMD_FLOAT32_C(  -992.91), EASYSIMD_FLOAT32_C(   213.78) },
      { EASYSIMD_FLOAT32_C(   236.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -992.91), EASYSIMD_FLOAT32_C(   213.78),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   493.54), EASYSIMD_FLOAT32_C(  -992.91), EASYSIMD_FLOAT32_C(   213.78) } },
    { UINT8_C(229),
      { EASYSIMD_FLOAT32_C(   572.59), EASYSIMD_FLOAT32_C(  -505.20), EASYSIMD_FLOAT32_C(  -888.69), EASYSIMD_FLOAT32_C(  -168.99) },
      { EASYSIMD_FLOAT32_C(   572.59), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -888.69), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -505.20), EASYSIMD_FLOAT32_C(  -888.69), EASYSIMD_FLOAT32_C(  -168.99) } },
    { UINT8_C(115),
      { EASYSIMD_FLOAT32_C(   961.78), EASYSIMD_FLOAT32_C(   587.15), EASYSIMD_FLOAT32_C(   162.08), EASYSIMD_FLOAT32_C(   131.99) },
      { EASYSIMD_FLOAT32_C(   961.78), EASYSIMD_FLOAT32_C(   587.15), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   961.78), EASYSIMD_FLOAT32_C(   587.15), EASYSIMD_FLOAT32_C(   162.08), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(107),
      { EASYSIMD_FLOAT32_C(   722.82), EASYSIMD_FLOAT32_C(   519.77), EASYSIMD_FLOAT32_C(  -160.36), EASYSIMD_FLOAT32_C(   908.34) },
      { EASYSIMD_FLOAT32_C(   722.82), EASYSIMD_FLOAT32_C(   519.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   908.34),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   519.77), EASYSIMD_FLOAT32_C(  -160.36), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(148),
      { EASYSIMD_FLOAT32_C(   251.18), EASYSIMD_FLOAT32_C(  -347.86), EASYSIMD_FLOAT32_C(  -514.92), EASYSIMD_FLOAT32_C(  -206.57) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -514.92), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   251.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -206.57) } },
    { UINT8_C(156),
      { EASYSIMD_FLOAT32_C(   874.47), EASYSIMD_FLOAT32_C(  -711.75), EASYSIMD_FLOAT32_C(  -458.03), EASYSIMD_FLOAT32_C(  -188.74) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -458.03), EASYSIMD_FLOAT32_C(  -188.74),
        EASYSIMD_FLOAT32_C(   874.47), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -188.74) } },
    { UINT8_C( 78),
      { EASYSIMD_FLOAT32_C(  -804.36), EASYSIMD_FLOAT32_C(  -844.65), EASYSIMD_FLOAT32_C(   -82.05), EASYSIMD_FLOAT32_C(  -986.67) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -844.65), EASYSIMD_FLOAT32_C(   -82.05), EASYSIMD_FLOAT32_C(  -986.67),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -82.05), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_broadcast_f32x4(test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_broadcast_f32x4");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm256_broadcast_i64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[2];
    int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 6963200126134553685), -INT64_C( 7609148696822777319) },
      {  INT64_C( 6963200126134553685), -INT64_C( 7609148696822777319),  INT64_C( 6963200126134553685), -INT64_C( 7609148696822777319) } },
    { {  INT64_C( 7689890055132762340),  INT64_C( 4167909811627404681) },
      {  INT64_C( 7689890055132762340),  INT64_C( 4167909811627404681),  INT64_C( 7689890055132762340),  INT64_C( 4167909811627404681) } },
    { {  INT64_C( 7416897855265901807),  INT64_C( 8268906394609561496) },
      {  INT64_C( 7416897855265901807),  INT64_C( 8268906394609561496),  INT64_C( 7416897855265901807),  INT64_C( 8268906394609561496) } },
    { {  INT64_C( 4903385060140106079),  INT64_C( 5196095526355203839) },
      {  INT64_C( 4903385060140106079),  INT64_C( 5196095526355203839),  INT64_C( 4903385060140106079),  INT64_C( 5196095526355203839) } },
    { {  INT64_C( 5619583818761934921),  INT64_C( 1329752330496222324) },
      {  INT64_C( 5619583818761934921),  INT64_C( 1329752330496222324),  INT64_C( 5619583818761934921),  INT64_C( 1329752330496222324) } },
    { {  INT64_C( 7060166891766288379), -INT64_C( 1848901794818400074) },
      {  INT64_C( 7060166891766288379), -INT64_C( 1848901794818400074),  INT64_C( 7060166891766288379), -INT64_C( 1848901794818400074) } },
    { {  INT64_C( 7420939269704527353), -INT64_C( 6061320975806860004) },
      {  INT64_C( 7420939269704527353), -INT64_C( 6061320975806860004),  INT64_C( 7420939269704527353), -INT64_C( 6061320975806860004) } },
    { {  INT64_C(  841711701805077995),  INT64_C( 5494234673206457020) },
      {  INT64_C(  841711701805077995),  INT64_C( 5494234673206457020),  INT64_C(  841711701805077995),  INT64_C( 5494234673206457020) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_broadcast_i64x2(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_broadcast_i64x2");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m256i r = easysimd_mm256_broadcast_i64x2(a);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_broadcast_i64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[4];
    uint8_t k;
    int64_t a[2];
    int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 4796453011418044555), -INT64_C( 8391880723476145499), -INT64_C( 9072354220674262993), -INT64_C( 2615515623626030947) },
      UINT8_C(249),
      { -INT64_C(  252733318206852282), -INT64_C( 8063279822707740963) },
      { -INT64_C(  252733318206852282), -INT64_C( 8391880723476145499), -INT64_C( 9072354220674262993), -INT64_C( 8063279822707740963) } },
    { { -INT64_C( 3833132236413570390), -INT64_C( 6697121626444486077),  INT64_C( 7592219647877672958),  INT64_C( 7958543080088619561) },
      UINT8_C(222),
      {  INT64_C( 1452467414018661866), -INT64_C( 4869493674784768413) },
      { -INT64_C( 3833132236413570390), -INT64_C( 4869493674784768413),  INT64_C( 1452467414018661866), -INT64_C( 4869493674784768413) } },
    { {  INT64_C( 8338328027121896594), -INT64_C(  698157948673222396),  INT64_C( 5746877810758826050), -INT64_C( 2717312934043897024) },
      UINT8_C(246),
      { -INT64_C( 5240057883237849198), -INT64_C( 4633719099996522294) },
      {  INT64_C( 8338328027121896594), -INT64_C( 4633719099996522294), -INT64_C( 5240057883237849198), -INT64_C( 2717312934043897024) } },
    { {  INT64_C( 6041289868526116499),  INT64_C( 4744390130722890935),  INT64_C( 6103165342337857500), -INT64_C( 4446613938812783228) },
      UINT8_C(211),
      { -INT64_C( 7807757211585521419),  INT64_C( 6139113683256772321) },
      { -INT64_C( 7807757211585521419),  INT64_C( 6139113683256772321),  INT64_C( 6103165342337857500), -INT64_C( 4446613938812783228) } },
    { {  INT64_C( 6150003806904986888),  INT64_C( 2611320528409257462),  INT64_C( 6295953624456704158),  INT64_C( 3310955128234549567) },
      UINT8_C( 13),
      {  INT64_C( 4754186419141673681), -INT64_C( 3266352581484452665) },
      {  INT64_C( 4754186419141673681),  INT64_C( 2611320528409257462),  INT64_C( 4754186419141673681), -INT64_C( 3266352581484452665) } },
    { {  INT64_C( 8691842652742186920), -INT64_C(   55227171201008562),  INT64_C( 3705052284452022733),  INT64_C( 1789876768266574690) },
      UINT8_C( 98),
      {  INT64_C( 6056974368233846815), -INT64_C( 6852083384986682289) },
      {  INT64_C( 8691842652742186920), -INT64_C( 6852083384986682289),  INT64_C( 3705052284452022733),  INT64_C( 1789876768266574690) } },
    { {  INT64_C( 6333538254973571462), -INT64_C( 2319952215507736193),  INT64_C(  631820855396992599),  INT64_C( 5692843264007358441) },
      UINT8_C( 28),
      { -INT64_C( 6715447894484659442),  INT64_C(  859133116862838710) },
      {  INT64_C( 6333538254973571462), -INT64_C( 2319952215507736193), -INT64_C( 6715447894484659442),  INT64_C(  859133116862838710) } },
    { {  INT64_C( 7455932523806416097), -INT64_C( 1723611079420880402),  INT64_C( 5885364252486052947),  INT64_C( 3623941196090838842) },
      UINT8_C( 47),
      {  INT64_C( 1281192974290501645), -INT64_C( 9171065157438357530) },
      {  INT64_C( 1281192974290501645), -INT64_C( 9171065157438357530),  INT64_C( 1281192974290501645), -INT64_C( 9171065157438357530) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_broadcast_i64x2(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_broadcast_i64x2");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m256i r = easysimd_mm256_mask_broadcast_i64x2(src, k, a);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_broadcast_i64x2(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int64_t a[2];
    int64_t r[4];
  } test_vec[] = {
    { UINT8_C(107),
      { -INT64_C( 8169087908317108874),  INT64_C( 5014308086998621585) },
      { -INT64_C( 8169087908317108874),  INT64_C( 5014308086998621585),  INT64_C(                   0),  INT64_C( 5014308086998621585) } },
    { UINT8_C( 47),
      {  INT64_C( 5807962795727219461),  INT64_C( 6981521134502879584) },
      {  INT64_C( 5807962795727219461),  INT64_C( 6981521134502879584),  INT64_C( 5807962795727219461),  INT64_C( 6981521134502879584) } },
    { UINT8_C( 42),
      { -INT64_C( 5863883790438681568), -INT64_C(  916689746660146261) },
      {  INT64_C(                   0), -INT64_C(  916689746660146261),  INT64_C(                   0), -INT64_C(  916689746660146261) } },
    { UINT8_C( 69),
      { -INT64_C( 7706032070848653229),  INT64_C( 2241975587045280281) },
      { -INT64_C( 7706032070848653229),  INT64_C(                   0), -INT64_C( 7706032070848653229),  INT64_C(                   0) } },
    { UINT8_C(211),
      { -INT64_C( 6574751263339060769), -INT64_C( 7663389074012026612) },
      { -INT64_C( 6574751263339060769), -INT64_C( 7663389074012026612),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(120),
      { -INT64_C( 4527174446734819948),  INT64_C( 1672995136010262331) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 1672995136010262331) } },
    { UINT8_C(203),
      {  INT64_C( 3657100242925758389), -INT64_C( 3445298165243183628) },
      {  INT64_C( 3657100242925758389), -INT64_C( 3445298165243183628),  INT64_C(                   0), -INT64_C( 3445298165243183628) } },
    { UINT8_C(252),
      {  INT64_C( 4401554386958656465),  INT64_C(  701071025108503502) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 4401554386958656465),  INT64_C(  701071025108503502) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_broadcast_i64x2(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_broadcast_i64x2");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m256i r = easysimd_mm256_maskz_broadcast_i64x2(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_broadcast_f64x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -818.06), EASYSIMD_FLOAT64_C(   862.04) },
      { EASYSIMD_FLOAT64_C(  -818.06), EASYSIMD_FLOAT64_C(   862.04), EASYSIMD_FLOAT64_C(  -818.06), EASYSIMD_FLOAT64_C(   862.04) } },
    { { EASYSIMD_FLOAT64_C(   251.28), EASYSIMD_FLOAT64_C(  -807.49) },
      { EASYSIMD_FLOAT64_C(   251.28), EASYSIMD_FLOAT64_C(  -807.49), EASYSIMD_FLOAT64_C(   251.28), EASYSIMD_FLOAT64_C(  -807.49) } },
    { { EASYSIMD_FLOAT64_C(   489.47), EASYSIMD_FLOAT64_C(   521.73) },
      { EASYSIMD_FLOAT64_C(   489.47), EASYSIMD_FLOAT64_C(   521.73), EASYSIMD_FLOAT64_C(   489.47), EASYSIMD_FLOAT64_C(   521.73) } },
    { { EASYSIMD_FLOAT64_C(   697.15), EASYSIMD_FLOAT64_C(  -943.39) },
      { EASYSIMD_FLOAT64_C(   697.15), EASYSIMD_FLOAT64_C(  -943.39), EASYSIMD_FLOAT64_C(   697.15), EASYSIMD_FLOAT64_C(  -943.39) } },
    { { EASYSIMD_FLOAT64_C(   397.38), EASYSIMD_FLOAT64_C(   769.24) },
      { EASYSIMD_FLOAT64_C(   397.38), EASYSIMD_FLOAT64_C(   769.24), EASYSIMD_FLOAT64_C(   397.38), EASYSIMD_FLOAT64_C(   769.24) } },
    { { EASYSIMD_FLOAT64_C(   607.10), EASYSIMD_FLOAT64_C(  -411.28) },
      { EASYSIMD_FLOAT64_C(   607.10), EASYSIMD_FLOAT64_C(  -411.28), EASYSIMD_FLOAT64_C(   607.10), EASYSIMD_FLOAT64_C(  -411.28) } },
    { { EASYSIMD_FLOAT64_C(  -417.96), EASYSIMD_FLOAT64_C(  -732.77) },
      { EASYSIMD_FLOAT64_C(  -417.96), EASYSIMD_FLOAT64_C(  -732.77), EASYSIMD_FLOAT64_C(  -417.96), EASYSIMD_FLOAT64_C(  -732.77) } },
    { { EASYSIMD_FLOAT64_C(   409.47), EASYSIMD_FLOAT64_C(   -49.18) },
      { EASYSIMD_FLOAT64_C(   409.47), EASYSIMD_FLOAT64_C(   -49.18), EASYSIMD_FLOAT64_C(   409.47), EASYSIMD_FLOAT64_C(   -49.18) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_broadcast_f64x2(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_broadcast_f64x2");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm256_mask_broadcast_f64x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 src[4];
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -338.03), EASYSIMD_FLOAT64_C(   731.04), EASYSIMD_FLOAT64_C(   652.28), EASYSIMD_FLOAT64_C(   868.31) },
      UINT8_C(176),
      { EASYSIMD_FLOAT64_C(  -240.09), EASYSIMD_FLOAT64_C(   738.97) },
      { EASYSIMD_FLOAT64_C(  -338.03), EASYSIMD_FLOAT64_C(   731.04), EASYSIMD_FLOAT64_C(   652.28), EASYSIMD_FLOAT64_C(   868.31) } },
    { { EASYSIMD_FLOAT64_C(  -161.99), EASYSIMD_FLOAT64_C(  -539.33), EASYSIMD_FLOAT64_C(  -491.52), EASYSIMD_FLOAT64_C(   960.24) },
      UINT8_C( 23),
      { EASYSIMD_FLOAT64_C(   782.07), EASYSIMD_FLOAT64_C(   293.17) },
      { EASYSIMD_FLOAT64_C(   782.07), EASYSIMD_FLOAT64_C(   293.17), EASYSIMD_FLOAT64_C(   782.07), EASYSIMD_FLOAT64_C(   960.24) } },
    { { EASYSIMD_FLOAT64_C(  -948.97), EASYSIMD_FLOAT64_C(   718.70), EASYSIMD_FLOAT64_C(  -833.55), EASYSIMD_FLOAT64_C(   519.24) },
      UINT8_C(166),
      { EASYSIMD_FLOAT64_C(   879.34), EASYSIMD_FLOAT64_C(  -863.77) },
      { EASYSIMD_FLOAT64_C(  -948.97), EASYSIMD_FLOAT64_C(  -863.77), EASYSIMD_FLOAT64_C(   879.34), EASYSIMD_FLOAT64_C(   519.24) } },
    { { EASYSIMD_FLOAT64_C(   136.25), EASYSIMD_FLOAT64_C(   -99.23), EASYSIMD_FLOAT64_C(   178.08), EASYSIMD_FLOAT64_C(  -929.05) },
      UINT8_C( 20),
      { EASYSIMD_FLOAT64_C(  -614.75), EASYSIMD_FLOAT64_C(   -70.42) },
      { EASYSIMD_FLOAT64_C(   136.25), EASYSIMD_FLOAT64_C(   -99.23), EASYSIMD_FLOAT64_C(  -614.75), EASYSIMD_FLOAT64_C(  -929.05) } },
    { { EASYSIMD_FLOAT64_C(  -617.52), EASYSIMD_FLOAT64_C(  -721.29), EASYSIMD_FLOAT64_C(  -762.54), EASYSIMD_FLOAT64_C(    70.31) },
      UINT8_C(  5),
      { EASYSIMD_FLOAT64_C(  -322.15), EASYSIMD_FLOAT64_C(  -417.60) },
      { EASYSIMD_FLOAT64_C(  -322.15), EASYSIMD_FLOAT64_C(  -721.29), EASYSIMD_FLOAT64_C(  -322.15), EASYSIMD_FLOAT64_C(    70.31) } },
    { { EASYSIMD_FLOAT64_C(  -577.36), EASYSIMD_FLOAT64_C(   298.63), EASYSIMD_FLOAT64_C(  -985.58), EASYSIMD_FLOAT64_C(  -562.98) },
      UINT8_C(167),
      { EASYSIMD_FLOAT64_C(   -39.73), EASYSIMD_FLOAT64_C(   262.95) },
      { EASYSIMD_FLOAT64_C(   -39.73), EASYSIMD_FLOAT64_C(   262.95), EASYSIMD_FLOAT64_C(   -39.73), EASYSIMD_FLOAT64_C(  -562.98) } },
    { { EASYSIMD_FLOAT64_C(   943.89), EASYSIMD_FLOAT64_C(  -108.91), EASYSIMD_FLOAT64_C(  -463.93), EASYSIMD_FLOAT64_C(   675.74) },
      UINT8_C(200),
      { EASYSIMD_FLOAT64_C(  -918.41), EASYSIMD_FLOAT64_C(   364.14) },
      { EASYSIMD_FLOAT64_C(   943.89), EASYSIMD_FLOAT64_C(  -108.91), EASYSIMD_FLOAT64_C(  -463.93), EASYSIMD_FLOAT64_C(   364.14) } },
    { { EASYSIMD_FLOAT64_C(   -90.94), EASYSIMD_FLOAT64_C(  -345.61), EASYSIMD_FLOAT64_C(  -599.08), EASYSIMD_FLOAT64_C(  -818.15) },
      UINT8_C(109),
      { EASYSIMD_FLOAT64_C(  -714.87), EASYSIMD_FLOAT64_C(  -771.51) },
      { EASYSIMD_FLOAT64_C(  -714.87), EASYSIMD_FLOAT64_C(  -345.61), EASYSIMD_FLOAT64_C(  -714.87), EASYSIMD_FLOAT64_C(  -771.51) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_broadcast_f64x2(src, test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_broadcast_f64x2");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm256_maskz_broadcast_f64x2 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(197),
      { EASYSIMD_FLOAT64_C(  -215.62), EASYSIMD_FLOAT64_C(    35.19) },
      { EASYSIMD_FLOAT64_C(  -215.62), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -215.62), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(198),
      { EASYSIMD_FLOAT64_C(   716.52), EASYSIMD_FLOAT64_C(   473.89) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   473.89), EASYSIMD_FLOAT64_C(   716.52), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 85),
      { EASYSIMD_FLOAT64_C(   312.77), EASYSIMD_FLOAT64_C(   715.13) },
      { EASYSIMD_FLOAT64_C(   312.77), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   312.77), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(212),
      { EASYSIMD_FLOAT64_C(   527.96), EASYSIMD_FLOAT64_C(  -502.50) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   527.96), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 50),
      { EASYSIMD_FLOAT64_C(  -571.65), EASYSIMD_FLOAT64_C(   248.58) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   248.58), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(207),
      { EASYSIMD_FLOAT64_C(   234.22), EASYSIMD_FLOAT64_C(   607.13) },
      { EASYSIMD_FLOAT64_C(   234.22), EASYSIMD_FLOAT64_C(   607.13), EASYSIMD_FLOAT64_C(   234.22), EASYSIMD_FLOAT64_C(   607.13) } },
    { UINT8_C(  5),
      { EASYSIMD_FLOAT64_C(  -229.19), EASYSIMD_FLOAT64_C(   -58.91) },
      { EASYSIMD_FLOAT64_C(  -229.19), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -229.19), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(104),
      { EASYSIMD_FLOAT64_C(    -8.77), EASYSIMD_FLOAT64_C(   682.18) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   682.18) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_broadcast_f64x2(test_vec[i].k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_broadcast_f64x2");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcast_f32x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   241.63), EASYSIMD_FLOAT32_C(   962.32), EASYSIMD_FLOAT32_C(  -223.53), EASYSIMD_FLOAT32_C(  -221.69)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   241.63), EASYSIMD_FLOAT32_C(   962.32), EASYSIMD_FLOAT32_C(  -223.53), EASYSIMD_FLOAT32_C(  -221.69),
                         EASYSIMD_FLOAT32_C(   241.63), EASYSIMD_FLOAT32_C(   962.32), EASYSIMD_FLOAT32_C(  -223.53), EASYSIMD_FLOAT32_C(  -221.69),
                         EASYSIMD_FLOAT32_C(   241.63), EASYSIMD_FLOAT32_C(   962.32), EASYSIMD_FLOAT32_C(  -223.53), EASYSIMD_FLOAT32_C(  -221.69),
                         EASYSIMD_FLOAT32_C(   241.63), EASYSIMD_FLOAT32_C(   962.32), EASYSIMD_FLOAT32_C(  -223.53), EASYSIMD_FLOAT32_C(  -221.69)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   115.71), EASYSIMD_FLOAT32_C(  -206.04), EASYSIMD_FLOAT32_C(  -581.48), EASYSIMD_FLOAT32_C(   670.36)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   115.71), EASYSIMD_FLOAT32_C(  -206.04), EASYSIMD_FLOAT32_C(  -581.48), EASYSIMD_FLOAT32_C(   670.36),
                         EASYSIMD_FLOAT32_C(   115.71), EASYSIMD_FLOAT32_C(  -206.04), EASYSIMD_FLOAT32_C(  -581.48), EASYSIMD_FLOAT32_C(   670.36),
                         EASYSIMD_FLOAT32_C(   115.71), EASYSIMD_FLOAT32_C(  -206.04), EASYSIMD_FLOAT32_C(  -581.48), EASYSIMD_FLOAT32_C(   670.36),
                         EASYSIMD_FLOAT32_C(   115.71), EASYSIMD_FLOAT32_C(  -206.04), EASYSIMD_FLOAT32_C(  -581.48), EASYSIMD_FLOAT32_C(   670.36)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   297.45), EASYSIMD_FLOAT32_C(   193.39), EASYSIMD_FLOAT32_C(  -163.24), EASYSIMD_FLOAT32_C(  -775.87)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   297.45), EASYSIMD_FLOAT32_C(   193.39), EASYSIMD_FLOAT32_C(  -163.24), EASYSIMD_FLOAT32_C(  -775.87),
                         EASYSIMD_FLOAT32_C(   297.45), EASYSIMD_FLOAT32_C(   193.39), EASYSIMD_FLOAT32_C(  -163.24), EASYSIMD_FLOAT32_C(  -775.87),
                         EASYSIMD_FLOAT32_C(   297.45), EASYSIMD_FLOAT32_C(   193.39), EASYSIMD_FLOAT32_C(  -163.24), EASYSIMD_FLOAT32_C(  -775.87),
                         EASYSIMD_FLOAT32_C(   297.45), EASYSIMD_FLOAT32_C(   193.39), EASYSIMD_FLOAT32_C(  -163.24), EASYSIMD_FLOAT32_C(  -775.87)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -675.37), EASYSIMD_FLOAT32_C(   853.20), EASYSIMD_FLOAT32_C(  -377.67), EASYSIMD_FLOAT32_C(   233.14)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -675.37), EASYSIMD_FLOAT32_C(   853.20), EASYSIMD_FLOAT32_C(  -377.67), EASYSIMD_FLOAT32_C(   233.14),
                         EASYSIMD_FLOAT32_C(  -675.37), EASYSIMD_FLOAT32_C(   853.20), EASYSIMD_FLOAT32_C(  -377.67), EASYSIMD_FLOAT32_C(   233.14),
                         EASYSIMD_FLOAT32_C(  -675.37), EASYSIMD_FLOAT32_C(   853.20), EASYSIMD_FLOAT32_C(  -377.67), EASYSIMD_FLOAT32_C(   233.14),
                         EASYSIMD_FLOAT32_C(  -675.37), EASYSIMD_FLOAT32_C(   853.20), EASYSIMD_FLOAT32_C(  -377.67), EASYSIMD_FLOAT32_C(   233.14)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -156.08), EASYSIMD_FLOAT32_C(  -209.26), EASYSIMD_FLOAT32_C(    48.51), EASYSIMD_FLOAT32_C(  -627.76)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -156.08), EASYSIMD_FLOAT32_C(  -209.26), EASYSIMD_FLOAT32_C(    48.51), EASYSIMD_FLOAT32_C(  -627.76),
                         EASYSIMD_FLOAT32_C(  -156.08), EASYSIMD_FLOAT32_C(  -209.26), EASYSIMD_FLOAT32_C(    48.51), EASYSIMD_FLOAT32_C(  -627.76),
                         EASYSIMD_FLOAT32_C(  -156.08), EASYSIMD_FLOAT32_C(  -209.26), EASYSIMD_FLOAT32_C(    48.51), EASYSIMD_FLOAT32_C(  -627.76),
                         EASYSIMD_FLOAT32_C(  -156.08), EASYSIMD_FLOAT32_C(  -209.26), EASYSIMD_FLOAT32_C(    48.51), EASYSIMD_FLOAT32_C(  -627.76)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   108.40), EASYSIMD_FLOAT32_C(   970.37), EASYSIMD_FLOAT32_C(   934.72), EASYSIMD_FLOAT32_C(  -932.81)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   108.40), EASYSIMD_FLOAT32_C(   970.37), EASYSIMD_FLOAT32_C(   934.72), EASYSIMD_FLOAT32_C(  -932.81),
                         EASYSIMD_FLOAT32_C(   108.40), EASYSIMD_FLOAT32_C(   970.37), EASYSIMD_FLOAT32_C(   934.72), EASYSIMD_FLOAT32_C(  -932.81),
                         EASYSIMD_FLOAT32_C(   108.40), EASYSIMD_FLOAT32_C(   970.37), EASYSIMD_FLOAT32_C(   934.72), EASYSIMD_FLOAT32_C(  -932.81),
                         EASYSIMD_FLOAT32_C(   108.40), EASYSIMD_FLOAT32_C(   970.37), EASYSIMD_FLOAT32_C(   934.72), EASYSIMD_FLOAT32_C(  -932.81)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   690.58), EASYSIMD_FLOAT32_C(   836.42), EASYSIMD_FLOAT32_C(  -952.66), EASYSIMD_FLOAT32_C(    22.35)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   690.58), EASYSIMD_FLOAT32_C(   836.42), EASYSIMD_FLOAT32_C(  -952.66), EASYSIMD_FLOAT32_C(    22.35),
                         EASYSIMD_FLOAT32_C(   690.58), EASYSIMD_FLOAT32_C(   836.42), EASYSIMD_FLOAT32_C(  -952.66), EASYSIMD_FLOAT32_C(    22.35),
                         EASYSIMD_FLOAT32_C(   690.58), EASYSIMD_FLOAT32_C(   836.42), EASYSIMD_FLOAT32_C(  -952.66), EASYSIMD_FLOAT32_C(    22.35),
                         EASYSIMD_FLOAT32_C(   690.58), EASYSIMD_FLOAT32_C(   836.42), EASYSIMD_FLOAT32_C(  -952.66), EASYSIMD_FLOAT32_C(    22.35)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   740.10), EASYSIMD_FLOAT32_C(   159.65), EASYSIMD_FLOAT32_C(   -65.49), EASYSIMD_FLOAT32_C(   946.83)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   740.10), EASYSIMD_FLOAT32_C(   159.65), EASYSIMD_FLOAT32_C(   -65.49), EASYSIMD_FLOAT32_C(   946.83),
                         EASYSIMD_FLOAT32_C(   740.10), EASYSIMD_FLOAT32_C(   159.65), EASYSIMD_FLOAT32_C(   -65.49), EASYSIMD_FLOAT32_C(   946.83),
                         EASYSIMD_FLOAT32_C(   740.10), EASYSIMD_FLOAT32_C(   159.65), EASYSIMD_FLOAT32_C(   -65.49), EASYSIMD_FLOAT32_C(   946.83),
                         EASYSIMD_FLOAT32_C(   740.10), EASYSIMD_FLOAT32_C(   159.65), EASYSIMD_FLOAT32_C(   -65.49), EASYSIMD_FLOAT32_C(   946.83)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 a = test_vec[i].a;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_broadcast_f32x4(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_broadcast_f32x4");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcast_f32x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 src;
    easysimd__mmask16 k;
    easysimd__m128 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -476.82), EASYSIMD_FLOAT32_C(   687.27), EASYSIMD_FLOAT32_C(   239.12), EASYSIMD_FLOAT32_C(  -622.96),
                         EASYSIMD_FLOAT32_C(   479.82), EASYSIMD_FLOAT32_C(  -652.18), EASYSIMD_FLOAT32_C(   585.66), EASYSIMD_FLOAT32_C(  -840.39),
                         EASYSIMD_FLOAT32_C(  -680.47), EASYSIMD_FLOAT32_C(  -211.69), EASYSIMD_FLOAT32_C(   879.50), EASYSIMD_FLOAT32_C(   245.88),
                         EASYSIMD_FLOAT32_C(   689.68), EASYSIMD_FLOAT32_C(   107.64), EASYSIMD_FLOAT32_C(  -872.56), EASYSIMD_FLOAT32_C(  -586.10)),
      UINT16_C(63721),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   337.98), EASYSIMD_FLOAT32_C(  -931.30), EASYSIMD_FLOAT32_C(   -93.71), EASYSIMD_FLOAT32_C(   492.43)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   337.98), EASYSIMD_FLOAT32_C(  -931.30), EASYSIMD_FLOAT32_C(   -93.71), EASYSIMD_FLOAT32_C(   492.43),
                         EASYSIMD_FLOAT32_C(   337.98), EASYSIMD_FLOAT32_C(  -652.18), EASYSIMD_FLOAT32_C(   585.66), EASYSIMD_FLOAT32_C(  -840.39),
                         EASYSIMD_FLOAT32_C(   337.98), EASYSIMD_FLOAT32_C(  -931.30), EASYSIMD_FLOAT32_C(   -93.71), EASYSIMD_FLOAT32_C(   245.88),
                         EASYSIMD_FLOAT32_C(   337.98), EASYSIMD_FLOAT32_C(   107.64), EASYSIMD_FLOAT32_C(  -872.56), EASYSIMD_FLOAT32_C(   492.43)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   792.34), EASYSIMD_FLOAT32_C(  -828.98), EASYSIMD_FLOAT32_C(   152.82), EASYSIMD_FLOAT32_C(   261.49),
                         EASYSIMD_FLOAT32_C(  -674.96), EASYSIMD_FLOAT32_C(  -626.70), EASYSIMD_FLOAT32_C(  -365.50), EASYSIMD_FLOAT32_C(   522.39),
                         EASYSIMD_FLOAT32_C(   659.15), EASYSIMD_FLOAT32_C(   204.13), EASYSIMD_FLOAT32_C(   487.20), EASYSIMD_FLOAT32_C(   790.92),
                         EASYSIMD_FLOAT32_C(  -372.23), EASYSIMD_FLOAT32_C(  -362.18), EASYSIMD_FLOAT32_C(   725.62), EASYSIMD_FLOAT32_C(   817.00)),
      UINT16_C(44067),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -858.39), EASYSIMD_FLOAT32_C(   608.18), EASYSIMD_FLOAT32_C(   129.78), EASYSIMD_FLOAT32_C(  -779.98)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -858.39), EASYSIMD_FLOAT32_C(  -828.98), EASYSIMD_FLOAT32_C(   129.78), EASYSIMD_FLOAT32_C(   261.49),
                         EASYSIMD_FLOAT32_C(  -858.39), EASYSIMD_FLOAT32_C(   608.18), EASYSIMD_FLOAT32_C(  -365.50), EASYSIMD_FLOAT32_C(   522.39),
                         EASYSIMD_FLOAT32_C(   659.15), EASYSIMD_FLOAT32_C(   204.13), EASYSIMD_FLOAT32_C(   129.78), EASYSIMD_FLOAT32_C(   790.92),
                         EASYSIMD_FLOAT32_C(  -372.23), EASYSIMD_FLOAT32_C(  -362.18), EASYSIMD_FLOAT32_C(   129.78), EASYSIMD_FLOAT32_C(  -779.98)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   433.05), EASYSIMD_FLOAT32_C(   405.44), EASYSIMD_FLOAT32_C(   652.04), EASYSIMD_FLOAT32_C(  -453.75),
                         EASYSIMD_FLOAT32_C(    56.24), EASYSIMD_FLOAT32_C(   506.86), EASYSIMD_FLOAT32_C(  -127.57), EASYSIMD_FLOAT32_C(  -230.83),
                         EASYSIMD_FLOAT32_C(  -815.89), EASYSIMD_FLOAT32_C(   351.22), EASYSIMD_FLOAT32_C(  -739.81), EASYSIMD_FLOAT32_C(  -104.33),
                         EASYSIMD_FLOAT32_C(   331.38), EASYSIMD_FLOAT32_C(   749.42), EASYSIMD_FLOAT32_C(   151.95), EASYSIMD_FLOAT32_C(   -25.90)),
      UINT16_C(12331),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -159.95), EASYSIMD_FLOAT32_C(  -519.57), EASYSIMD_FLOAT32_C(   -66.62), EASYSIMD_FLOAT32_C(  -690.93)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   433.05), EASYSIMD_FLOAT32_C(   405.44), EASYSIMD_FLOAT32_C(   -66.62), EASYSIMD_FLOAT32_C(  -690.93),
                         EASYSIMD_FLOAT32_C(    56.24), EASYSIMD_FLOAT32_C(   506.86), EASYSIMD_FLOAT32_C(  -127.57), EASYSIMD_FLOAT32_C(  -230.83),
                         EASYSIMD_FLOAT32_C(  -815.89), EASYSIMD_FLOAT32_C(   351.22), EASYSIMD_FLOAT32_C(   -66.62), EASYSIMD_FLOAT32_C(  -104.33),
                         EASYSIMD_FLOAT32_C(  -159.95), EASYSIMD_FLOAT32_C(   749.42), EASYSIMD_FLOAT32_C(   -66.62), EASYSIMD_FLOAT32_C(  -690.93)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   417.50), EASYSIMD_FLOAT32_C(   245.21), EASYSIMD_FLOAT32_C(   960.01), EASYSIMD_FLOAT32_C(  -303.61),
                         EASYSIMD_FLOAT32_C(  -550.57), EASYSIMD_FLOAT32_C(   665.98), EASYSIMD_FLOAT32_C(  -521.00), EASYSIMD_FLOAT32_C(   239.39),
                         EASYSIMD_FLOAT32_C(   798.32), EASYSIMD_FLOAT32_C(   251.37), EASYSIMD_FLOAT32_C(  -596.78), EASYSIMD_FLOAT32_C(   840.69),
                         EASYSIMD_FLOAT32_C(  -684.92), EASYSIMD_FLOAT32_C(    87.08), EASYSIMD_FLOAT32_C(   734.84), EASYSIMD_FLOAT32_C(  -854.89)),
      UINT16_C(52021),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -116.62), EASYSIMD_FLOAT32_C(   -17.97), EASYSIMD_FLOAT32_C(   229.99), EASYSIMD_FLOAT32_C(  -771.72)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -116.62), EASYSIMD_FLOAT32_C(   -17.97), EASYSIMD_FLOAT32_C(   960.01), EASYSIMD_FLOAT32_C(  -303.61),
                         EASYSIMD_FLOAT32_C(  -116.62), EASYSIMD_FLOAT32_C(   665.98), EASYSIMD_FLOAT32_C(   229.99), EASYSIMD_FLOAT32_C(  -771.72),
                         EASYSIMD_FLOAT32_C(   798.32), EASYSIMD_FLOAT32_C(   251.37), EASYSIMD_FLOAT32_C(   229.99), EASYSIMD_FLOAT32_C(  -771.72),
                         EASYSIMD_FLOAT32_C(  -684.92), EASYSIMD_FLOAT32_C(   -17.97), EASYSIMD_FLOAT32_C(   734.84), EASYSIMD_FLOAT32_C(  -771.72)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -616.45), EASYSIMD_FLOAT32_C(   914.70), EASYSIMD_FLOAT32_C(  -963.67), EASYSIMD_FLOAT32_C(  -935.61),
                         EASYSIMD_FLOAT32_C(   106.52), EASYSIMD_FLOAT32_C(   367.48), EASYSIMD_FLOAT32_C(   -10.30), EASYSIMD_FLOAT32_C(   543.55),
                         EASYSIMD_FLOAT32_C(   142.17), EASYSIMD_FLOAT32_C(  -844.51), EASYSIMD_FLOAT32_C(  -959.58), EASYSIMD_FLOAT32_C(   913.58),
                         EASYSIMD_FLOAT32_C(  -227.61), EASYSIMD_FLOAT32_C(  -979.09), EASYSIMD_FLOAT32_C(  -746.95), EASYSIMD_FLOAT32_C(   363.67)),
      UINT16_C(46395),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -319.95), EASYSIMD_FLOAT32_C(  -241.48), EASYSIMD_FLOAT32_C(  -416.05), EASYSIMD_FLOAT32_C(  -700.83)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -319.95), EASYSIMD_FLOAT32_C(   914.70), EASYSIMD_FLOAT32_C(  -416.05), EASYSIMD_FLOAT32_C(  -700.83),
                         EASYSIMD_FLOAT32_C(   106.52), EASYSIMD_FLOAT32_C(  -241.48), EASYSIMD_FLOAT32_C(   -10.30), EASYSIMD_FLOAT32_C(  -700.83),
                         EASYSIMD_FLOAT32_C(   142.17), EASYSIMD_FLOAT32_C(  -844.51), EASYSIMD_FLOAT32_C(  -416.05), EASYSIMD_FLOAT32_C(  -700.83),
                         EASYSIMD_FLOAT32_C(  -319.95), EASYSIMD_FLOAT32_C(  -979.09), EASYSIMD_FLOAT32_C(  -416.05), EASYSIMD_FLOAT32_C(  -700.83)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   810.23), EASYSIMD_FLOAT32_C(  -571.66), EASYSIMD_FLOAT32_C(  -313.94), EASYSIMD_FLOAT32_C(   812.08),
                         EASYSIMD_FLOAT32_C(   905.89), EASYSIMD_FLOAT32_C(    95.84), EASYSIMD_FLOAT32_C(  -942.64), EASYSIMD_FLOAT32_C(   490.95),
                         EASYSIMD_FLOAT32_C(   432.01), EASYSIMD_FLOAT32_C(  -989.57), EASYSIMD_FLOAT32_C(  -908.07), EASYSIMD_FLOAT32_C(   843.06),
                         EASYSIMD_FLOAT32_C(  -567.12), EASYSIMD_FLOAT32_C(   561.55), EASYSIMD_FLOAT32_C(  -316.58), EASYSIMD_FLOAT32_C(  -224.94)),
      UINT16_C(28510),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   608.47), EASYSIMD_FLOAT32_C(   502.71), EASYSIMD_FLOAT32_C(   524.73), EASYSIMD_FLOAT32_C(  -206.66)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   810.23), EASYSIMD_FLOAT32_C(   502.71), EASYSIMD_FLOAT32_C(   524.73), EASYSIMD_FLOAT32_C(   812.08),
                         EASYSIMD_FLOAT32_C(   608.47), EASYSIMD_FLOAT32_C(   502.71), EASYSIMD_FLOAT32_C(   524.73), EASYSIMD_FLOAT32_C(  -206.66),
                         EASYSIMD_FLOAT32_C(   432.01), EASYSIMD_FLOAT32_C(   502.71), EASYSIMD_FLOAT32_C(  -908.07), EASYSIMD_FLOAT32_C(  -206.66),
                         EASYSIMD_FLOAT32_C(   608.47), EASYSIMD_FLOAT32_C(   502.71), EASYSIMD_FLOAT32_C(   524.73), EASYSIMD_FLOAT32_C(  -224.94)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -770.27), EASYSIMD_FLOAT32_C(  -598.61), EASYSIMD_FLOAT32_C(   672.88), EASYSIMD_FLOAT32_C(  -504.06),
                         EASYSIMD_FLOAT32_C(   481.78), EASYSIMD_FLOAT32_C(  -154.88), EASYSIMD_FLOAT32_C(  -363.51), EASYSIMD_FLOAT32_C(  -643.93),
                         EASYSIMD_FLOAT32_C(  -973.84), EASYSIMD_FLOAT32_C(  -599.20), EASYSIMD_FLOAT32_C(   230.44), EASYSIMD_FLOAT32_C(  -713.35),
                         EASYSIMD_FLOAT32_C(  -554.88), EASYSIMD_FLOAT32_C(  -858.98), EASYSIMD_FLOAT32_C(   -21.09), EASYSIMD_FLOAT32_C(  -441.11)),
      UINT16_C( 6749),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   687.60), EASYSIMD_FLOAT32_C(   681.66), EASYSIMD_FLOAT32_C(  -362.35), EASYSIMD_FLOAT32_C(  -482.20)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -770.27), EASYSIMD_FLOAT32_C(  -598.61), EASYSIMD_FLOAT32_C(   672.88), EASYSIMD_FLOAT32_C(  -482.20),
                         EASYSIMD_FLOAT32_C(   687.60), EASYSIMD_FLOAT32_C(  -154.88), EASYSIMD_FLOAT32_C(  -362.35), EASYSIMD_FLOAT32_C(  -643.93),
                         EASYSIMD_FLOAT32_C(  -973.84), EASYSIMD_FLOAT32_C(   681.66), EASYSIMD_FLOAT32_C(   230.44), EASYSIMD_FLOAT32_C(  -482.20),
                         EASYSIMD_FLOAT32_C(   687.60), EASYSIMD_FLOAT32_C(   681.66), EASYSIMD_FLOAT32_C(   -21.09), EASYSIMD_FLOAT32_C(  -482.20)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -454.36), EASYSIMD_FLOAT32_C(  -172.69), EASYSIMD_FLOAT32_C(   256.23), EASYSIMD_FLOAT32_C(   682.27),
                         EASYSIMD_FLOAT32_C(   -43.91), EASYSIMD_FLOAT32_C(  -300.48), EASYSIMD_FLOAT32_C(   916.93), EASYSIMD_FLOAT32_C(  -592.77),
                         EASYSIMD_FLOAT32_C(   939.83), EASYSIMD_FLOAT32_C(  -553.88), EASYSIMD_FLOAT32_C(  -796.09), EASYSIMD_FLOAT32_C(  -515.91),
                         EASYSIMD_FLOAT32_C(   623.85), EASYSIMD_FLOAT32_C(   359.37), EASYSIMD_FLOAT32_C(  -557.79), EASYSIMD_FLOAT32_C(   595.65)),
      UINT16_C( 8287),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -705.53), EASYSIMD_FLOAT32_C(   238.42), EASYSIMD_FLOAT32_C(   504.37), EASYSIMD_FLOAT32_C(   296.48)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -454.36), EASYSIMD_FLOAT32_C(  -172.69), EASYSIMD_FLOAT32_C(   504.37), EASYSIMD_FLOAT32_C(   682.27),
                         EASYSIMD_FLOAT32_C(   -43.91), EASYSIMD_FLOAT32_C(  -300.48), EASYSIMD_FLOAT32_C(   916.93), EASYSIMD_FLOAT32_C(  -592.77),
                         EASYSIMD_FLOAT32_C(   939.83), EASYSIMD_FLOAT32_C(   238.42), EASYSIMD_FLOAT32_C(  -796.09), EASYSIMD_FLOAT32_C(   296.48),
                         EASYSIMD_FLOAT32_C(  -705.53), EASYSIMD_FLOAT32_C(   238.42), EASYSIMD_FLOAT32_C(   504.37), EASYSIMD_FLOAT32_C(   296.48)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128 a = test_vec[i].a;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_broadcast_f32x4(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_broadcast_f32x4");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcast_f32x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m128 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { UINT16_C(12860),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -93.71), EASYSIMD_FLOAT32_C(   137.99), EASYSIMD_FLOAT32_C(   492.43), EASYSIMD_FLOAT32_C(   420.83)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   492.43), EASYSIMD_FLOAT32_C(   420.83),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   492.43), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   492.43), EASYSIMD_FLOAT32_C(   420.83),
                         EASYSIMD_FLOAT32_C(   -93.71), EASYSIMD_FLOAT32_C(   137.99), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(63770),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -652.18), EASYSIMD_FLOAT32_C(  -872.56), EASYSIMD_FLOAT32_C(   585.66), EASYSIMD_FLOAT32_C(  -586.10)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -652.18), EASYSIMD_FLOAT32_C(  -872.56), EASYSIMD_FLOAT32_C(   585.66), EASYSIMD_FLOAT32_C(  -586.10),
                         EASYSIMD_FLOAT32_C(  -652.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -586.10),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -586.10),
                         EASYSIMD_FLOAT32_C(  -652.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   585.66), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(26030),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   700.39), EASYSIMD_FLOAT32_C(   129.78), EASYSIMD_FLOAT32_C(   708.98), EASYSIMD_FLOAT32_C(  -779.98)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   129.78), EASYSIMD_FLOAT32_C(   708.98), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   129.78), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -779.98),
                         EASYSIMD_FLOAT32_C(   700.39), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   708.98), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   700.39), EASYSIMD_FLOAT32_C(   129.78), EASYSIMD_FLOAT32_C(   708.98), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(41122),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -362.18), EASYSIMD_FLOAT32_C(  -626.70), EASYSIMD_FLOAT32_C(   725.62), EASYSIMD_FLOAT32_C(  -365.50)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -362.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   725.62), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -362.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   725.62), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   725.62), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(49851),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -519.57), EASYSIMD_FLOAT32_C(  -632.83), EASYSIMD_FLOAT32_C(   -66.62), EASYSIMD_FLOAT32_C(  -181.94)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -519.57), EASYSIMD_FLOAT32_C(  -632.83), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -66.62), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -519.57), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -66.62), EASYSIMD_FLOAT32_C(  -181.94),
                         EASYSIMD_FLOAT32_C(  -519.57), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -66.62), EASYSIMD_FLOAT32_C(  -181.94)) },
    { UINT16_C(41826),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    56.24), EASYSIMD_FLOAT32_C(   749.42), EASYSIMD_FLOAT32_C(   506.86), EASYSIMD_FLOAT32_C(   151.95)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    56.24), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   506.86), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   506.86), EASYSIMD_FLOAT32_C(   151.95),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   749.42), EASYSIMD_FLOAT32_C(   506.86), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   506.86), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(19285),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    57.37), EASYSIMD_FLOAT32_C(   -17.97), EASYSIMD_FLOAT32_C(   347.13), EASYSIMD_FLOAT32_C(   229.99)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -17.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(    57.37), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   347.13), EASYSIMD_FLOAT32_C(   229.99),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -17.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   229.99),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -17.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   229.99)) },
    { UINT16_C(48133),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -684.92), EASYSIMD_FLOAT32_C(  -550.57), EASYSIMD_FLOAT32_C(    87.08), EASYSIMD_FLOAT32_C(   665.98)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -684.92), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    87.08), EASYSIMD_FLOAT32_C(   665.98),
                         EASYSIMD_FLOAT32_C(  -684.92), EASYSIMD_FLOAT32_C(  -550.57), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -550.57), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   665.98)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128 a = test_vec[i].a;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_broadcast_f32x4(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_broadcast_f32x4");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcast_f64x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  241.63), EASYSIMD_FLOAT64_C(  962.32),
                         EASYSIMD_FLOAT64_C( -223.53), EASYSIMD_FLOAT64_C( -221.69)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  241.63), EASYSIMD_FLOAT64_C(  962.32),
                         EASYSIMD_FLOAT64_C( -223.53), EASYSIMD_FLOAT64_C( -221.69),
                         EASYSIMD_FLOAT64_C(  241.63), EASYSIMD_FLOAT64_C(  962.32),
                         EASYSIMD_FLOAT64_C( -223.53), EASYSIMD_FLOAT64_C( -221.69)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  115.71), EASYSIMD_FLOAT64_C( -206.04),
                         EASYSIMD_FLOAT64_C( -581.48), EASYSIMD_FLOAT64_C(  670.36)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  115.71), EASYSIMD_FLOAT64_C( -206.04),
                         EASYSIMD_FLOAT64_C( -581.48), EASYSIMD_FLOAT64_C(  670.36),
                         EASYSIMD_FLOAT64_C(  115.71), EASYSIMD_FLOAT64_C( -206.04),
                         EASYSIMD_FLOAT64_C( -581.48), EASYSIMD_FLOAT64_C(  670.36)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  297.45), EASYSIMD_FLOAT64_C(  193.39),
                         EASYSIMD_FLOAT64_C( -163.24), EASYSIMD_FLOAT64_C( -775.87)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  297.45), EASYSIMD_FLOAT64_C(  193.39),
                         EASYSIMD_FLOAT64_C( -163.24), EASYSIMD_FLOAT64_C( -775.87),
                         EASYSIMD_FLOAT64_C(  297.45), EASYSIMD_FLOAT64_C(  193.39),
                         EASYSIMD_FLOAT64_C( -163.24), EASYSIMD_FLOAT64_C( -775.87)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -675.37), EASYSIMD_FLOAT64_C(  853.20),
                         EASYSIMD_FLOAT64_C( -377.67), EASYSIMD_FLOAT64_C(  233.14)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -675.37), EASYSIMD_FLOAT64_C(  853.20),
                         EASYSIMD_FLOAT64_C( -377.67), EASYSIMD_FLOAT64_C(  233.14),
                         EASYSIMD_FLOAT64_C( -675.37), EASYSIMD_FLOAT64_C(  853.20),
                         EASYSIMD_FLOAT64_C( -377.67), EASYSIMD_FLOAT64_C(  233.14)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -156.08), EASYSIMD_FLOAT64_C( -209.26),
                         EASYSIMD_FLOAT64_C(   48.51), EASYSIMD_FLOAT64_C( -627.76)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -156.08), EASYSIMD_FLOAT64_C( -209.26),
                         EASYSIMD_FLOAT64_C(   48.51), EASYSIMD_FLOAT64_C( -627.76),
                         EASYSIMD_FLOAT64_C( -156.08), EASYSIMD_FLOAT64_C( -209.26),
                         EASYSIMD_FLOAT64_C(   48.51), EASYSIMD_FLOAT64_C( -627.76)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  108.40), EASYSIMD_FLOAT64_C(  970.37),
                         EASYSIMD_FLOAT64_C(  934.72), EASYSIMD_FLOAT64_C( -932.81)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  108.40), EASYSIMD_FLOAT64_C(  970.37),
                         EASYSIMD_FLOAT64_C(  934.72), EASYSIMD_FLOAT64_C( -932.81),
                         EASYSIMD_FLOAT64_C(  108.40), EASYSIMD_FLOAT64_C(  970.37),
                         EASYSIMD_FLOAT64_C(  934.72), EASYSIMD_FLOAT64_C( -932.81)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  690.58), EASYSIMD_FLOAT64_C(  836.42),
                         EASYSIMD_FLOAT64_C( -952.66), EASYSIMD_FLOAT64_C(   22.35)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  690.58), EASYSIMD_FLOAT64_C(  836.42),
                         EASYSIMD_FLOAT64_C( -952.66), EASYSIMD_FLOAT64_C(   22.35),
                         EASYSIMD_FLOAT64_C(  690.58), EASYSIMD_FLOAT64_C(  836.42),
                         EASYSIMD_FLOAT64_C( -952.66), EASYSIMD_FLOAT64_C(   22.35)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  740.10), EASYSIMD_FLOAT64_C(  159.65),
                         EASYSIMD_FLOAT64_C(  -65.49), EASYSIMD_FLOAT64_C(  946.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  740.10), EASYSIMD_FLOAT64_C(  159.65),
                         EASYSIMD_FLOAT64_C(  -65.49), EASYSIMD_FLOAT64_C(  946.83),
                         EASYSIMD_FLOAT64_C(  740.10), EASYSIMD_FLOAT64_C(  159.65),
                         EASYSIMD_FLOAT64_C(  -65.49), EASYSIMD_FLOAT64_C(  946.83)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d a = test_vec[i].a;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_broadcast_f64x4(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_broadcast_f64x4");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcast_f64x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d src;
    easysimd__mmask8 k;
    easysimd__m256d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -396.88), EASYSIMD_FLOAT64_C(  354.04),
                         EASYSIMD_FLOAT64_C(  268.06), EASYSIMD_FLOAT64_C( -972.10),
                         EASYSIMD_FLOAT64_C( -213.85), EASYSIMD_FLOAT64_C( -574.68),
                         EASYSIMD_FLOAT64_C(  137.99), EASYSIMD_FLOAT64_C(  420.83)),
      UINT8_C( 60),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  337.98), EASYSIMD_FLOAT64_C( -931.30),
                         EASYSIMD_FLOAT64_C(  -93.71), EASYSIMD_FLOAT64_C(  492.43)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -396.88), EASYSIMD_FLOAT64_C(  354.04),
                         EASYSIMD_FLOAT64_C(  -93.71), EASYSIMD_FLOAT64_C(  492.43),
                         EASYSIMD_FLOAT64_C(  337.98), EASYSIMD_FLOAT64_C( -931.30),
                         EASYSIMD_FLOAT64_C(  137.99), EASYSIMD_FLOAT64_C(  420.83)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -680.47), EASYSIMD_FLOAT64_C( -211.69),
                         EASYSIMD_FLOAT64_C(  879.50), EASYSIMD_FLOAT64_C(  245.88),
                         EASYSIMD_FLOAT64_C(  689.68), EASYSIMD_FLOAT64_C(  107.64),
                         EASYSIMD_FLOAT64_C( -872.56), EASYSIMD_FLOAT64_C( -586.10)),
      UINT8_C( 26),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -622.96), EASYSIMD_FLOAT64_C(  479.82),
                         EASYSIMD_FLOAT64_C( -652.18), EASYSIMD_FLOAT64_C(  585.66)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -680.47), EASYSIMD_FLOAT64_C( -211.69),
                         EASYSIMD_FLOAT64_C(  879.50), EASYSIMD_FLOAT64_C(  585.66),
                         EASYSIMD_FLOAT64_C( -622.96), EASYSIMD_FLOAT64_C(  107.64),
                         EASYSIMD_FLOAT64_C( -652.18), EASYSIMD_FLOAT64_C( -586.10)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  408.02), EASYSIMD_FLOAT64_C(  662.99),
                         EASYSIMD_FLOAT64_C( -491.44), EASYSIMD_FLOAT64_C( -586.97),
                         EASYSIMD_FLOAT64_C( -858.39), EASYSIMD_FLOAT64_C(  608.18),
                         EASYSIMD_FLOAT64_C(  129.78), EASYSIMD_FLOAT64_C( -779.98)),
      UINT8_C(174),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  469.58), EASYSIMD_FLOAT64_C( -229.18),
                         EASYSIMD_FLOAT64_C(  700.39), EASYSIMD_FLOAT64_C(  708.98)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  469.58), EASYSIMD_FLOAT64_C(  662.99),
                         EASYSIMD_FLOAT64_C(  700.39), EASYSIMD_FLOAT64_C( -586.97),
                         EASYSIMD_FLOAT64_C(  469.58), EASYSIMD_FLOAT64_C( -229.18),
                         EASYSIMD_FLOAT64_C(  700.39), EASYSIMD_FLOAT64_C( -779.98)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -491.65), EASYSIMD_FLOAT64_C(  792.34),
                         EASYSIMD_FLOAT64_C( -828.98), EASYSIMD_FLOAT64_C(  152.82),
                         EASYSIMD_FLOAT64_C(  261.49), EASYSIMD_FLOAT64_C( -674.96),
                         EASYSIMD_FLOAT64_C( -626.70), EASYSIMD_FLOAT64_C( -365.50)),
      UINT8_C(162),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  790.92), EASYSIMD_FLOAT64_C( -372.23),
                         EASYSIMD_FLOAT64_C( -362.18), EASYSIMD_FLOAT64_C(  725.62)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  790.92), EASYSIMD_FLOAT64_C(  792.34),
                         EASYSIMD_FLOAT64_C( -362.18), EASYSIMD_FLOAT64_C(  152.82),
                         EASYSIMD_FLOAT64_C(  261.49), EASYSIMD_FLOAT64_C( -674.96),
                         EASYSIMD_FLOAT64_C( -362.18), EASYSIMD_FLOAT64_C( -365.50)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -230.83), EASYSIMD_FLOAT64_C( -480.11),
                         EASYSIMD_FLOAT64_C(  511.94), EASYSIMD_FLOAT64_C(  614.74),
                         EASYSIMD_FLOAT64_C(  794.95), EASYSIMD_FLOAT64_C( -331.37),
                         EASYSIMD_FLOAT64_C( -632.83), EASYSIMD_FLOAT64_C( -181.94)),
      UINT8_C(187),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  125.71), EASYSIMD_FLOAT64_C( -159.95),
                         EASYSIMD_FLOAT64_C( -519.57), EASYSIMD_FLOAT64_C(  -66.62)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  125.71), EASYSIMD_FLOAT64_C( -480.11),
                         EASYSIMD_FLOAT64_C( -519.57), EASYSIMD_FLOAT64_C(  -66.62),
                         EASYSIMD_FLOAT64_C(  125.71), EASYSIMD_FLOAT64_C( -331.37),
                         EASYSIMD_FLOAT64_C( -519.57), EASYSIMD_FLOAT64_C(  -66.62)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  870.57), EASYSIMD_FLOAT64_C( -815.89),
                         EASYSIMD_FLOAT64_C(  351.22), EASYSIMD_FLOAT64_C( -739.81),
                         EASYSIMD_FLOAT64_C( -104.33), EASYSIMD_FLOAT64_C(  331.38),
                         EASYSIMD_FLOAT64_C(  749.42), EASYSIMD_FLOAT64_C(  151.95)),
      UINT8_C( 98),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  652.04), EASYSIMD_FLOAT64_C( -453.75),
                         EASYSIMD_FLOAT64_C(   56.24), EASYSIMD_FLOAT64_C(  506.86)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  870.57), EASYSIMD_FLOAT64_C( -453.75),
                         EASYSIMD_FLOAT64_C(   56.24), EASYSIMD_FLOAT64_C( -739.81),
                         EASYSIMD_FLOAT64_C( -104.33), EASYSIMD_FLOAT64_C(  331.38),
                         EASYSIMD_FLOAT64_C(   56.24), EASYSIMD_FLOAT64_C(  151.95)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -854.89), EASYSIMD_FLOAT64_C(  376.16),
                         EASYSIMD_FLOAT64_C( -846.26), EASYSIMD_FLOAT64_C(  817.65),
                         EASYSIMD_FLOAT64_C( -403.95), EASYSIMD_FLOAT64_C( -116.62),
                         EASYSIMD_FLOAT64_C(  -17.97), EASYSIMD_FLOAT64_C(  229.99)),
      UINT8_C( 85),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -65.83), EASYSIMD_FLOAT64_C( -494.87),
                         EASYSIMD_FLOAT64_C(   57.37), EASYSIMD_FLOAT64_C(  347.13)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -854.89), EASYSIMD_FLOAT64_C( -494.87),
                         EASYSIMD_FLOAT64_C( -846.26), EASYSIMD_FLOAT64_C(  347.13),
                         EASYSIMD_FLOAT64_C( -403.95), EASYSIMD_FLOAT64_C( -494.87),
                         EASYSIMD_FLOAT64_C(  -17.97), EASYSIMD_FLOAT64_C(  347.13)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -700.83), EASYSIMD_FLOAT64_C( -289.50),
                         EASYSIMD_FLOAT64_C(  417.50), EASYSIMD_FLOAT64_C(  245.21),
                         EASYSIMD_FLOAT64_C(  960.01), EASYSIMD_FLOAT64_C( -303.61),
                         EASYSIMD_FLOAT64_C( -550.57), EASYSIMD_FLOAT64_C(  665.98)),
      UINT8_C(  5),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -596.78), EASYSIMD_FLOAT64_C(  840.69),
                         EASYSIMD_FLOAT64_C( -684.92), EASYSIMD_FLOAT64_C(   87.08)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -700.83), EASYSIMD_FLOAT64_C( -289.50),
                         EASYSIMD_FLOAT64_C(  417.50), EASYSIMD_FLOAT64_C(  245.21),
                         EASYSIMD_FLOAT64_C(  960.01), EASYSIMD_FLOAT64_C(  840.69),
                         EASYSIMD_FLOAT64_C( -550.57), EASYSIMD_FLOAT64_C(   87.08)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d src = test_vec[i].src;
    easysimd__m256d a = test_vec[i].a;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_broadcast_f64x4(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_broadcast_f64x4");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcast_f64x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m256d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { UINT8_C( 25),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -93.71), EASYSIMD_FLOAT64_C(  137.99),
                         EASYSIMD_FLOAT64_C(  492.43), EASYSIMD_FLOAT64_C(  420.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  420.83),
                         EASYSIMD_FLOAT64_C(  -93.71), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  420.83)) },
    { UINT8_C(223),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  354.04), EASYSIMD_FLOAT64_C( -261.67),
                         EASYSIMD_FLOAT64_C(  268.06), EASYSIMD_FLOAT64_C(  648.56)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  354.04), EASYSIMD_FLOAT64_C( -261.67),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  648.56),
                         EASYSIMD_FLOAT64_C(  354.04), EASYSIMD_FLOAT64_C( -261.67),
                         EASYSIMD_FLOAT64_C(  268.06), EASYSIMD_FLOAT64_C(  648.56)) },
    { UINT8_C(191),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  107.64), EASYSIMD_FLOAT64_C( -652.18),
                         EASYSIMD_FLOAT64_C( -872.56), EASYSIMD_FLOAT64_C(  585.66)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  107.64), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C( -872.56), EASYSIMD_FLOAT64_C(  585.66),
                         EASYSIMD_FLOAT64_C(  107.64), EASYSIMD_FLOAT64_C( -652.18),
                         EASYSIMD_FLOAT64_C( -872.56), EASYSIMD_FLOAT64_C(  585.66)) },
    { UINT8_C( 77),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -476.82), EASYSIMD_FLOAT64_C( -211.69),
                         EASYSIMD_FLOAT64_C(  687.27), EASYSIMD_FLOAT64_C(  879.50)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -211.69),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C( -476.82), EASYSIMD_FLOAT64_C( -211.69),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  879.50)) },
    { UINT8_C(216),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -229.18), EASYSIMD_FLOAT64_C(  608.18),
                         EASYSIMD_FLOAT64_C(  700.39), EASYSIMD_FLOAT64_C(  129.78)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -229.18), EASYSIMD_FLOAT64_C(  608.18),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  129.78),
                         EASYSIMD_FLOAT64_C( -229.18), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(196),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  408.02), EASYSIMD_FLOAT64_C( -213.85),
                         EASYSIMD_FLOAT64_C(  662.99), EASYSIMD_FLOAT64_C(  346.52)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  408.02), EASYSIMD_FLOAT64_C( -213.85),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -213.85),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(125),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  261.49), EASYSIMD_FLOAT64_C( -372.23),
                         EASYSIMD_FLOAT64_C( -674.96), EASYSIMD_FLOAT64_C( -362.18)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -372.23),
                         EASYSIMD_FLOAT64_C( -674.96), EASYSIMD_FLOAT64_C( -362.18),
                         EASYSIMD_FLOAT64_C(  261.49), EASYSIMD_FLOAT64_C( -372.23),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -362.18)) },
    { UINT8_C( 95),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  475.10), EASYSIMD_FLOAT64_C( -491.65),
                         EASYSIMD_FLOAT64_C(  659.15), EASYSIMD_FLOAT64_C(  792.34)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -491.65),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  792.34),
                         EASYSIMD_FLOAT64_C(  475.10), EASYSIMD_FLOAT64_C( -491.65),
                         EASYSIMD_FLOAT64_C(  659.15), EASYSIMD_FLOAT64_C(  792.34)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d a = test_vec[i].a;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_broadcast_f64x4(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_broadcast_f64x4");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcast_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm_set_epi32(INT32_C( 1322912216), INT32_C( -192131569), INT32_C(  457247766), INT32_C( 1585478853)),
      easysimd_mm512_set_epi32(INT32_C( 1322912216), INT32_C( -192131569), INT32_C(  457247766), INT32_C( 1585478853),
                            INT32_C( 1322912216), INT32_C( -192131569), INT32_C(  457247766), INT32_C( 1585478853),
                            INT32_C( 1322912216), INT32_C( -192131569), INT32_C(  457247766), INT32_C( 1585478853),
                            INT32_C( 1322912216), INT32_C( -192131569), INT32_C(  457247766), INT32_C( 1585478853)) },
    { easysimd_mm_set_epi32(INT32_C(  455358584), INT32_C( -549958328), INT32_C( 1779282555), INT32_C(-1938144165)),
      easysimd_mm512_set_epi32(INT32_C(  455358584), INT32_C( -549958328), INT32_C( 1779282555), INT32_C(-1938144165),
                            INT32_C(  455358584), INT32_C( -549958328), INT32_C( 1779282555), INT32_C(-1938144165),
                            INT32_C(  455358584), INT32_C( -549958328), INT32_C( 1779282555), INT32_C(-1938144165),
                            INT32_C(  455358584), INT32_C( -549958328), INT32_C( 1779282555), INT32_C(-1938144165)) },
    { easysimd_mm_set_epi32(INT32_C(   35244693), INT32_C( -163894097), INT32_C(  -32854349), INT32_C(-1300832792)),
      easysimd_mm512_set_epi32(INT32_C(   35244693), INT32_C( -163894097), INT32_C(  -32854349), INT32_C(-1300832792),
                            INT32_C(   35244693), INT32_C( -163894097), INT32_C(  -32854349), INT32_C(-1300832792),
                            INT32_C(   35244693), INT32_C( -163894097), INT32_C(  -32854349), INT32_C(-1300832792),
                            INT32_C(   35244693), INT32_C( -163894097), INT32_C(  -32854349), INT32_C(-1300832792)) },
    { easysimd_mm_set_epi32(INT32_C( 1137728540), INT32_C( 1602744474), INT32_C( -610393021), INT32_C(-1810116300)),
      easysimd_mm512_set_epi32(INT32_C( 1137728540), INT32_C( 1602744474), INT32_C( -610393021), INT32_C(-1810116300),
                            INT32_C( 1137728540), INT32_C( 1602744474), INT32_C( -610393021), INT32_C(-1810116300),
                            INT32_C( 1137728540), INT32_C( 1602744474), INT32_C( -610393021), INT32_C(-1810116300),
                            INT32_C( 1137728540), INT32_C( 1602744474), INT32_C( -610393021), INT32_C(-1810116300)) },
    { easysimd_mm_set_epi32(INT32_C(-1023450780), INT32_C(  840494259), INT32_C(-1087383364), INT32_C(-1604779562)),
      easysimd_mm512_set_epi32(INT32_C(-1023450780), INT32_C(  840494259), INT32_C(-1087383364), INT32_C(-1604779562),
                            INT32_C(-1023450780), INT32_C(  840494259), INT32_C(-1087383364), INT32_C(-1604779562),
                            INT32_C(-1023450780), INT32_C(  840494259), INT32_C(-1087383364), INT32_C(-1604779562),
                            INT32_C(-1023450780), INT32_C(  840494259), INT32_C(-1087383364), INT32_C(-1604779562)) },
    { easysimd_mm_set_epi32(INT32_C( 1284866833), INT32_C(   27132707), INT32_C(-1597877982), INT32_C(-1252321438)),
      easysimd_mm512_set_epi32(INT32_C( 1284866833), INT32_C(   27132707), INT32_C(-1597877982), INT32_C(-1252321438),
                            INT32_C( 1284866833), INT32_C(   27132707), INT32_C(-1597877982), INT32_C(-1252321438),
                            INT32_C( 1284866833), INT32_C(   27132707), INT32_C(-1597877982), INT32_C(-1252321438),
                            INT32_C( 1284866833), INT32_C(   27132707), INT32_C(-1597877982), INT32_C(-1252321438)) },
    { easysimd_mm_set_epi32(INT32_C( -165954025), INT32_C(  878840386), INT32_C( -802596544), INT32_C( 1574139347)),
      easysimd_mm512_set_epi32(INT32_C( -165954025), INT32_C(  878840386), INT32_C( -802596544), INT32_C( 1574139347),
                            INT32_C( -165954025), INT32_C(  878840386), INT32_C( -802596544), INT32_C( 1574139347),
                            INT32_C( -165954025), INT32_C(  878840386), INT32_C( -802596544), INT32_C( 1574139347),
                            INT32_C( -165954025), INT32_C(  878840386), INT32_C( -802596544), INT32_C( 1574139347)) },
    { easysimd_mm_set_epi32(INT32_C( -602275056), INT32_C(-1823359312), INT32_C( 1232365699), INT32_C(  345237769)),
      easysimd_mm512_set_epi32(INT32_C( -602275056), INT32_C(-1823359312), INT32_C( 1232365699), INT32_C(  345237769),
                            INT32_C( -602275056), INT32_C(-1823359312), INT32_C( 1232365699), INT32_C(  345237769),
                            INT32_C( -602275056), INT32_C(-1823359312), INT32_C( 1232365699), INT32_C(  345237769),
                            INT32_C( -602275056), INT32_C(-1823359312), INT32_C( 1232365699), INT32_C(  345237769)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_broadcast_i32x4(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_broadcast_i32x4");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcast_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask16 k;
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( 1479802474), INT32_C(  587294539), INT32_C( -174751528), INT32_C( 1465222154),
                            INT32_C( 1625882140), INT32_C(-1283973275), INT32_C(  567394727), INT32_C( 1808136008),
                            INT32_C(  324921956), INT32_C(-1888780980), INT32_C( -262803011), INT32_C( 2131227345),
                            INT32_C( -161180317), INT32_C( -569391310), INT32_C(  471197581), INT32_C( 2029664703)),
      UINT16_C(12686),
      easysimd_mm_set_epi32(INT32_C(-1929654026), INT32_C(-1217014324), INT32_C(  230292224), INT32_C( 1361651453)),
      easysimd_mm512_set_epi32(INT32_C( 1479802474), INT32_C(  587294539), INT32_C(  230292224), INT32_C( 1361651453),
                            INT32_C( 1625882140), INT32_C(-1283973275), INT32_C(  567394727), INT32_C( 1361651453),
                            INT32_C(-1929654026), INT32_C(-1888780980), INT32_C( -262803011), INT32_C( 2131227345),
                            INT32_C(-1929654026), INT32_C(-1217014324), INT32_C(  230292224), INT32_C( 2029664703)) },
    { easysimd_mm512_set_epi32(INT32_C( 1958214116), INT32_C( 2124258263), INT32_C(-1603442041), INT32_C(-1137458903),
                            INT32_C( -291704812), INT32_C( -523349105), INT32_C( -769676631), INT32_C(  359038153),
                            INT32_C( -860324016), INT32_C(  142975746), INT32_C( 1871956670), INT32_C(-2122929741),
                            INT32_C( 1007202856), INT32_C(-1693638626), INT32_C(-1497430440), INT32_C(  766142674)),
      UINT16_C( 3460),
      easysimd_mm_set_epi32(INT32_C(-1801778632), INT32_C(  793094568), INT32_C(  739597071), INT32_C( 1855829690)),
      easysimd_mm512_set_epi32(INT32_C( 1958214116), INT32_C( 2124258263), INT32_C(-1603442041), INT32_C(-1137458903),
                            INT32_C(-1801778632), INT32_C(  793094568), INT32_C( -769676631), INT32_C( 1855829690),
                            INT32_C(-1801778632), INT32_C(  142975746), INT32_C( 1871956670), INT32_C(-2122929741),
                            INT32_C( 1007202856), INT32_C(  793094568), INT32_C(-1497430440), INT32_C(  766142674)) },
    { easysimd_mm512_set_epi32(INT32_C( -491998875), INT32_C( -465346847), INT32_C( 1096008422), INT32_C( -151618100),
                            INT32_C( -483382033), INT32_C(-1500806456), INT32_C(  175505846), INT32_C( -698441328),
                            INT32_C( -515513970), INT32_C( 1679973349), INT32_C(-1523347194), INT32_C(   91392241),
                            INT32_C( -561919749), INT32_C( -634254878), INT32_C( -625316172), INT32_C(  -17019235)),
      UINT16_C(25030),
      easysimd_mm_set_epi32(INT32_C( -839244820), INT32_C(-1678825378), INT32_C(  464598558), INT32_C(-1198702193)),
      easysimd_mm512_set_epi32(INT32_C( -491998875), INT32_C(-1678825378), INT32_C(  464598558), INT32_C( -151618100),
                            INT32_C( -483382033), INT32_C(-1500806456), INT32_C(  175505846), INT32_C(-1198702193),
                            INT32_C( -839244820), INT32_C(-1678825378), INT32_C(-1523347194), INT32_C(   91392241),
                            INT32_C( -561919749), INT32_C(-1678825378), INT32_C(  464598558), INT32_C(  -17019235)) },
    { easysimd_mm512_set_epi32(INT32_C( 1319681857), INT32_C(  649867282), INT32_C(-1955467744), INT32_C(-1687114005),
                            INT32_C(-1950655074), INT32_C(-2040429697), INT32_C( 1764915437), INT32_C(  813475409),
                            INT32_C(-1622276195), INT32_C(  614665853), INT32_C( -661145222), INT32_C(  -43416876),
                            INT32_C(  954392932), INT32_C(-1003825870), INT32_C( -858676034), INT32_C( 1589986539)),
      UINT16_C(29308),
      easysimd_mm_set_epi32(INT32_C(-1945617369), INT32_C( -313192838), INT32_C( -614227976), INT32_C(  -73637500)),
      easysimd_mm512_set_epi32(INT32_C( 1319681857), INT32_C( -313192838), INT32_C( -614227976), INT32_C(  -73637500),
                            INT32_C(-1950655074), INT32_C(-2040429697), INT32_C( -614227976), INT32_C(  813475409),
                            INT32_C(-1622276195), INT32_C( -313192838), INT32_C( -614227976), INT32_C(  -73637500),
                            INT32_C(-1945617369), INT32_C( -313192838), INT32_C( -858676034), INT32_C( 1589986539)) },
    { easysimd_mm512_set_epi32(INT32_C(  482652005), INT32_C( 1083073699), INT32_C( -547163888), INT32_C(-1439583577),
                            INT32_C( -836573741), INT32_C(-2032318592), INT32_C( 1307381638), INT32_C( 2027662416),
                            INT32_C( 2001285861), INT32_C( 1074543972), INT32_C(-2107097596), INT32_C(-2025611729),
                            INT32_C(  962055101), INT32_C( 1886777199), INT32_C( 1689643613), INT32_C(-1874481648)),
      UINT16_C(45428),
      easysimd_mm_set_epi32(INT32_C(  110278011), INT32_C(-1940227644), INT32_C(-1803195700), INT32_C( 1287862649)),
      easysimd_mm512_set_epi32(INT32_C(  110278011), INT32_C( 1083073699), INT32_C(-1803195700), INT32_C( 1287862649),
                            INT32_C( -836573741), INT32_C(-2032318592), INT32_C( 1307381638), INT32_C( 1287862649),
                            INT32_C( 2001285861), INT32_C(-1940227644), INT32_C(-1803195700), INT32_C( 1287862649),
                            INT32_C(  962055101), INT32_C(-1940227644), INT32_C( 1689643613), INT32_C(-1874481648)) },
    { easysimd_mm512_set_epi32(INT32_C(  485695865), INT32_C( 1704586743), INT32_C(-1227241134), INT32_C(  279727823),
                            INT32_C( -480355834), INT32_C( 1374909005), INT32_C(-1706379633), INT32_C( 1300025155),
                            INT32_C( 1901096153), INT32_C(-1845297076), INT32_C(  188971064), INT32_C( 1903842318),
                            INT32_C(-1221674473), INT32_C(-1332164211), INT32_C(   23564349), INT32_C(-2098316192)),
      UINT16_C(21964),
      easysimd_mm_set_epi32(INT32_C(-1820692848), INT32_C( -830585945), INT32_C( 1667959054), INT32_C(-1758734041)),
      easysimd_mm512_set_epi32(INT32_C(  485695865), INT32_C( -830585945), INT32_C(-1227241134), INT32_C(-1758734041),
                            INT32_C( -480355834), INT32_C( -830585945), INT32_C(-1706379633), INT32_C(-1758734041),
                            INT32_C(-1820692848), INT32_C( -830585945), INT32_C(  188971064), INT32_C( 1903842318),
                            INT32_C(-1820692848), INT32_C( -830585945), INT32_C(   23564349), INT32_C(-2098316192)) },
    { easysimd_mm512_set_epi32(INT32_C(-1876069406), INT32_C( 1820341222), INT32_C(  987166931), INT32_C(-1021572249),
                            INT32_C(-1046533173), INT32_C(-1808511518), INT32_C( -283777637), INT32_C( -168486656),
                            INT32_C( 1250903497), INT32_C( 1175614584), INT32_C(  204391673), INT32_C( -667659280),
                            INT32_C( 2035348040), INT32_C( -596829354), INT32_C(-1607289004), INT32_C( -670488239)),
      UINT16_C(31159),
      easysimd_mm_set_epi32(INT32_C(-1492076939), INT32_C( 1502879171), INT32_C( 1497885207), INT32_C(-1325620059)),
      easysimd_mm512_set_epi32(INT32_C(-1876069406), INT32_C( 1502879171), INT32_C( 1497885207), INT32_C(-1325620059),
                            INT32_C(-1492076939), INT32_C(-1808511518), INT32_C( -283777637), INT32_C(-1325620059),
                            INT32_C(-1492076939), INT32_C( 1175614584), INT32_C( 1497885207), INT32_C(-1325620059),
                            INT32_C( 2035348040), INT32_C( 1502879171), INT32_C( 1497885207), INT32_C(-1325620059)) },
    { easysimd_mm512_set_epi32(INT32_C(-1346174896), INT32_C( 1223712250), INT32_C( 2029339086), INT32_C( 2108949315),
                            INT32_C(-1822742445), INT32_C( -343433299), INT32_C(-1626119528), INT32_C( 1735301543),
                            INT32_C(  766111295), INT32_C(  -80424103), INT32_C( 1232059506), INT32_C(-1681875170),
                            INT32_C( 1819208351), INT32_C( -734074357), INT32_C(   61937468), INT32_C(-1403575087)),
      UINT16_C(37926),
      easysimd_mm_set_epi32(INT32_C( 1656599178), INT32_C( 1293315993), INT32_C( -728433677), INT32_C( -125533424)),
      easysimd_mm512_set_epi32(INT32_C( 1656599178), INT32_C( 1223712250), INT32_C( 2029339086), INT32_C( -125533424),
                            INT32_C(-1822742445), INT32_C( 1293315993), INT32_C(-1626119528), INT32_C( 1735301543),
                            INT32_C(  766111295), INT32_C(  -80424103), INT32_C( -728433677), INT32_C(-1681875170),
                            INT32_C( 1819208351), INT32_C( 1293315993), INT32_C( -728433677), INT32_C(-1403575087)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = test_vec[i].a;
    easysimd__m512i src = test_vec[i].src;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_broadcast_i32x4(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_broadcast_i32x4");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcast_i32x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT16_C(57503),
      easysimd_mm_set_epi32(INT32_C(  913371223), INT32_C( 1946242675), INT32_C(-1851162974), INT32_C(-1090004303)),
      easysimd_mm512_set_epi32(INT32_C(  913371223), INT32_C( 1946242675), INT32_C(-1851162974), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(  913371223), INT32_C(          0), INT32_C(          0), INT32_C(-1090004303),
                            INT32_C(  913371223), INT32_C( 1946242675), INT32_C(-1851162974), INT32_C(-1090004303)) },
    { UINT16_C( 9830),
      easysimd_mm_set_epi32(INT32_C( -754702866), INT32_C(   59910169), INT32_C(-1421684089), INT32_C( 1688249563)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(-1421684089), INT32_C(          0),
                            INT32_C(          0), INT32_C(   59910169), INT32_C(-1421684089), INT32_C(          0),
                            INT32_C(          0), INT32_C(   59910169), INT32_C(-1421684089), INT32_C(          0),
                            INT32_C(          0), INT32_C(   59910169), INT32_C(-1421684089), INT32_C(          0)) },
    { UINT16_C(54973),
      easysimd_mm_set_epi32(INT32_C( 1295192258), INT32_C( 2064350366), INT32_C(-1387191485), INT32_C( 1585557386)),
      easysimd_mm512_set_epi32(INT32_C( 1295192258), INT32_C( 2064350366), INT32_C(          0), INT32_C( 1585557386),
                            INT32_C(          0), INT32_C( 2064350366), INT32_C(-1387191485), INT32_C(          0),
                            INT32_C( 1295192258), INT32_C(          0), INT32_C(-1387191485), INT32_C( 1585557386),
                            INT32_C( 1295192258), INT32_C( 2064350366), INT32_C(          0), INT32_C( 1585557386)) },
    { UINT16_C( 2571),
      easysimd_mm_set_epi32(INT32_C(  273665101), INT32_C( -889778981), INT32_C(  888851167), INT32_C(  342766140)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(  273665101), INT32_C(          0), INT32_C(  888851167), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(  273665101), INT32_C(          0), INT32_C(  888851167), INT32_C(  342766140)) },
    { UINT16_C(34156),
      easysimd_mm_set_epi32(INT32_C(  809684493), INT32_C( -666403540), INT32_C(-1117073828), INT32_C(-1916337185)),
      easysimd_mm512_set_epi32(INT32_C(  809684493), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( -666403540), INT32_C(          0), INT32_C(-1916337185),
                            INT32_C(          0), INT32_C( -666403540), INT32_C(-1117073828), INT32_C(          0),
                            INT32_C(  809684493), INT32_C( -666403540), INT32_C(          0), INT32_C(          0)) },
    { UINT16_C( 6544),
      easysimd_mm_set_epi32(INT32_C( 1692879261), INT32_C( -671588299), INT32_C( -258764942), INT32_C(-1633977409)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(-1633977409),
                            INT32_C( 1692879261), INT32_C(          0), INT32_C(          0), INT32_C(-1633977409),
                            INT32_C( 1692879261), INT32_C(          0), INT32_C(          0), INT32_C(-1633977409),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
    { UINT16_C(45909),
      easysimd_mm_set_epi32(INT32_C(  472486650), INT32_C( 1238366490), INT32_C(-1084360471), INT32_C(  686181072)),
      easysimd_mm512_set_epi32(INT32_C(  472486650), INT32_C(          0), INT32_C(-1084360471), INT32_C(  686181072),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1084360471), INT32_C(  686181072),
                            INT32_C(          0), INT32_C( 1238366490), INT32_C(          0), INT32_C(  686181072),
                            INT32_C(          0), INT32_C( 1238366490), INT32_C(          0), INT32_C(  686181072)) },
    { UINT16_C(56653),
      easysimd_mm_set_epi32(INT32_C( 1655322598), INT32_C( -841418169), INT32_C( -643403227), INT32_C(-1868778842)),
      easysimd_mm512_set_epi32(INT32_C( 1655322598), INT32_C( -841418169), INT32_C(          0), INT32_C(-1868778842),
                            INT32_C( 1655322598), INT32_C( -841418169), INT32_C(          0), INT32_C(-1868778842),
                            INT32_C(          0), INT32_C( -841418169), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1655322598), INT32_C( -841418169), INT32_C(          0), INT32_C(-1868778842)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_broadcast_i32x4(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_broadcast_i32x4");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcast_i64x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm256_set_epi64x(INT64_C( 2067253863170152603), INT64_C( 7322969156688688496),
                             INT64_C(-3040413397780943697), INT64_C( -347515311309491350)),
      easysimd_mm512_set_epi64(INT64_C( 2067253863170152603), INT64_C( 7322969156688688496),
                            INT64_C(-3040413397780943697), INT64_C( -347515311309491350),
                            INT64_C( 2067253863170152603), INT64_C( 7322969156688688496),
                            INT64_C(-3040413397780943697), INT64_C( -347515311309491350)) },
    { easysimd_mm256_set_epi64x(INT64_C(-8775907405261856642), INT64_C( 2994184764454707691),
                             INT64_C( 5740004668815682638), INT64_C(-6479861669953478300)),
      easysimd_mm512_set_epi64(INT64_C(-8775907405261856642), INT64_C( 2994184764454707691),
                            INT64_C( 5740004668815682638), INT64_C(-6479861669953478300),
                            INT64_C(-8775907405261856642), INT64_C( 2994184764454707691),
                            INT64_C( 5740004668815682638), INT64_C(-6479861669953478300)) },
    { easysimd_mm256_set_epi64x(INT64_C(-1508734178901937051), INT64_C(-9017252864562564261),
                             INT64_C( -273279204292504060), INT64_C(  619750219118375084)),
      easysimd_mm512_set_epi64(INT64_C(-1508734178901937051), INT64_C(-9017252864562564261),
                            INT64_C( -273279204292504060), INT64_C(  619750219118375084),
                            INT64_C(-1508734178901937051), INT64_C(-9017252864562564261),
                            INT64_C( -273279204292504060), INT64_C(  619750219118375084)) },
    { easysimd_mm256_set_epi64x(INT64_C( 5726987144774798582), INT64_C(-5242976599564634972),
                             INT64_C(-2397121704692329659), INT64_C( 8619348224440898856)),
      easysimd_mm512_set_epi64(INT64_C( 5726987144774798582), INT64_C(-5242976599564634972),
                            INT64_C(-2397121704692329659), INT64_C( 8619348224440898856),
                            INT64_C( 5726987144774798582), INT64_C(-5242976599564634972),
                            INT64_C(-2397121704692329659), INT64_C( 8619348224440898856)) },
    { easysimd_mm256_set_epi64x(INT64_C( 3770039990400590046), INT64_C(-4228023324121815234),
                             INT64_C(-2554402032947045809), INT64_C(-5734730006803594733)),
      easysimd_mm512_set_epi64(INT64_C( 3770039990400590046), INT64_C(-4228023324121815234),
                            INT64_C(-2554402032947045809), INT64_C(-5734730006803594733),
                            INT64_C( 3770039990400590046), INT64_C(-4228023324121815234),
                            INT64_C(-2554402032947045809), INT64_C(-5734730006803594733)) },
    { easysimd_mm256_set_epi64x(INT64_C(-7969300362390541280), INT64_C( 5131273406597805369),
                             INT64_C( 3164578103377175393), INT64_C( -896289702737256643)),
      easysimd_mm512_set_epi64(INT64_C(-7969300362390541280), INT64_C( 5131273406597805369),
                            INT64_C( 3164578103377175393), INT64_C( -896289702737256643),
                            INT64_C(-7969300362390541280), INT64_C( 5131273406597805369),
                            INT64_C( 3164578103377175393), INT64_C( -896289702737256643)) },
    { easysimd_mm256_set_epi64x(INT64_C( 6358202424481672256), INT64_C(-2088789378195753898),
                             INT64_C(-3832720361616382569), INT64_C(-1395499602347228816)),
      easysimd_mm512_set_epi64(INT64_C( 6358202424481672256), INT64_C(-2088789378195753898),
                            INT64_C(-3832720361616382569), INT64_C(-1395499602347228816),
                            INT64_C( 6358202424481672256), INT64_C(-2088789378195753898),
                            INT64_C(-3832720361616382569), INT64_C(-1395499602347228816)) },
    { easysimd_mm256_set_epi64x(INT64_C(-7005415045902450329), INT64_C(  454800303112400674),
                             INT64_C(  120562593220559221), INT64_C(-9183341893829321065)),
      easysimd_mm512_set_epi64(INT64_C(-7005415045902450329), INT64_C(  454800303112400674),
                            INT64_C(  120562593220559221), INT64_C(-9183341893829321065),
                            INT64_C(-7005415045902450329), INT64_C(  454800303112400674),
                            INT64_C(  120562593220559221), INT64_C(-9183341893829321065)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_broadcast_i64x4(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_broadcast_i64x4");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcast_i64x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m256i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C(-6314317108894035774), INT64_C( 8866317312363406147),
                            INT64_C( 6809917121524389565), INT64_C(-3241424127607560167),
                            INT64_C(-6106086665810303781), INT64_C(  633642393017577559),
                            INT64_C( 8359048641648361122), INT64_C(-4681532830833057633)),
      UINT8_C( 60),
      easysimd_mm256_set_epi64x(INT64_C( 3477568421141904684), INT64_C(-4797795556098898977),
                             INT64_C( 3208117008747973709), INT64_C(-3821571623174354209)),
      easysimd_mm512_set_epi64(INT64_C(-6314317108894035774), INT64_C( 8866317312363406147),
                            INT64_C( 3208117008747973709), INT64_C(-3821571623174354209),
                            INT64_C( 3477568421141904684), INT64_C(-4797795556098898977),
                            INT64_C( 8359048641648361122), INT64_C(-4681532830833057633)) },
    { easysimd_mm512_set_epi64(INT64_C( 1306125493676423142), INT64_C(-3613863514463636955),
                            INT64_C(-8026344006176744115), INT64_C( 2029314710784964890),
                            INT64_C(-4657292759333975344), INT64_C( 4825522705097247133),
                            INT64_C(-2884449776545067150), INT64_C(-7017879531382302320)),
      UINT8_C(  0),
      easysimd_mm256_set_epi64x(INT64_C(-4405193415265233332), INT64_C( 7250935849068321562),
                             INT64_C(-6027293339582699304), INT64_C(-3733599027822978693)),
      easysimd_mm512_set_epi64(INT64_C( 1306125493676423142), INT64_C(-3613863514463636955),
                            INT64_C(-8026344006176744115), INT64_C( 2029314710784964890),
                            INT64_C(-4657292759333975344), INT64_C( 4825522705097247133),
                            INT64_C(-2884449776545067150), INT64_C(-7017879531382302320)) },
    { easysimd_mm512_set_epi64(INT64_C( 4688717956956220153), INT64_C(-1915316091557446787),
                            INT64_C( 1577347929723399506), INT64_C(-7813885322626023749),
                            INT64_C(-6811547529988353683), INT64_C( 2997984888778655645),
                            INT64_C( 3443124806434765346), INT64_C( 5852240145563215278)),
      UINT8_C(246),
      easysimd_mm256_set_epi64x(INT64_C(-1891210360757244537), INT64_C( 6167039147883013727),
                             INT64_C( 3386552444698298512), INT64_C( 7545310155849572514)),
      easysimd_mm512_set_epi64(INT64_C(-1891210360757244537), INT64_C( 6167039147883013727),
                            INT64_C( 3386552444698298512), INT64_C( 7545310155849572514),
                            INT64_C(-6811547529988353683), INT64_C( 6167039147883013727),
                            INT64_C( 3386552444698298512), INT64_C( 5852240145563215278)) },
    { easysimd_mm512_set_epi64(INT64_C( 5038277295705077786), INT64_C(-8704670477732479640),
                            INT64_C(-4548397220420700343), INT64_C( 8046739269734052975),
                            INT64_C( 7094379553694909752), INT64_C( 4795143479989329521),
                            INT64_C(-4501545483124413586), INT64_C(-3553418787378740418)),
      UINT8_C( 91),
      easysimd_mm256_set_epi64x(INT64_C(-1193819960890806229), INT64_C( 1698145641448748604),
                             INT64_C(-5983907472113043464), INT64_C( 2399871967268573321)),
      easysimd_mm512_set_epi64(INT64_C( 5038277295705077786), INT64_C( 1698145641448748604),
                            INT64_C(-4548397220420700343), INT64_C( 2399871967268573321),
                            INT64_C(-1193819960890806229), INT64_C( 4795143479989329521),
                            INT64_C(-5983907472113043464), INT64_C( 2399871967268573321)) },
    { easysimd_mm512_set_epi64(INT64_C(-7015430497800685262), INT64_C( 6395476272833483099),
                            INT64_C(-7658177893206805688), INT64_C( 8616202346974378134),
                            INT64_C( 4658965153462790469), INT64_C(-8694270525310808014),
                            INT64_C(-6021620893121233714), INT64_C(-2734912706905093379)),
      UINT8_C(200),
      easysimd_mm256_set_epi64x(INT64_C(-1469383970610000896), INT64_C( 2906056864364420569),
                             INT64_C(-8420208282727167471), INT64_C(-2445653243165948933)),
      easysimd_mm512_set_epi64(INT64_C(-1469383970610000896), INT64_C( 2906056864364420569),
                            INT64_C(-7658177893206805688), INT64_C( 8616202346974378134),
                            INT64_C(-1469383970610000896), INT64_C(-8694270525310808014),
                            INT64_C(-6021620893121233714), INT64_C(-2734912706905093379)) },
    { easysimd_mm512_set_epi64(INT64_C(  -14573144697473529), INT64_C( 8194534140513027918),
                            INT64_C( 2864848388614962181), INT64_C(-8899252041456864412),
                            INT64_C( 6379752944219310901), INT64_C(-1860193003353627344),
                            INT64_C(-6904865090556452860), INT64_C( 3719036040063860682)),
      UINT8_C(242),
      easysimd_mm256_set_epi64x(INT64_C(-5869124324801971655), INT64_C( 4548184433513821860),
                             INT64_C( -866976878921007676), INT64_C( 2203520398864570966)),
      easysimd_mm512_set_epi64(INT64_C(-5869124324801971655), INT64_C( 4548184433513821860),
                            INT64_C( -866976878921007676), INT64_C( 2203520398864570966),
                            INT64_C( 6379752944219310901), INT64_C(-1860193003353627344),
                            INT64_C( -866976878921007676), INT64_C( 3719036040063860682)) },
    { easysimd_mm512_set_epi64(INT64_C( 6933317985964373307), INT64_C(-7912084547370987750),
                            INT64_C( 1434122569595023374), INT64_C(  372849821895528123),
                            INT64_C( -797096709674116855), INT64_C( 7124042714150240897),
                            INT64_C(  192820077199458500), INT64_C( 2333974304098521090)),
      UINT8_C(243),
      easysimd_mm256_set_epi64x(INT64_C( 9216760499566437432), INT64_C(-3611239802138142732),
                             INT64_C(-4586686018735308980), INT64_C(-4383556822793463465)),
      easysimd_mm512_set_epi64(INT64_C( 9216760499566437432), INT64_C(-3611239802138142732),
                            INT64_C(-4586686018735308980), INT64_C(-4383556822793463465),
                            INT64_C( -797096709674116855), INT64_C( 7124042714150240897),
                            INT64_C(-4586686018735308980), INT64_C(-4383556822793463465)) },
    { easysimd_mm512_set_epi64(INT64_C(-1447537183271280169), INT64_C( 3992622506060288146),
                            INT64_C(-4043997837551953925), INT64_C( 6303477149728220498),
                            INT64_C( 7148655265583700891), INT64_C(-2780283900793463061),
                            INT64_C( 3296623181868458839), INT64_C( 3808941703531633947)),
      UINT8_C(191),
      easysimd_mm256_set_epi64x(INT64_C( 4775871390633368548), INT64_C( 1184569154591270183),
                             INT64_C(-1750343127516454914), INT64_C( 3950749388527391085)),
      easysimd_mm512_set_epi64(INT64_C( 4775871390633368548), INT64_C( 3992622506060288146),
                            INT64_C(-1750343127516454914), INT64_C( 3950749388527391085),
                            INT64_C( 4775871390633368548), INT64_C( 1184569154591270183),
                            INT64_C(-1750343127516454914), INT64_C( 3950749388527391085)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_broadcast_i64x4(src, k, a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_broadcast_i64x4");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcast_i64x4(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m256i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT8_C( 81),
      easysimd_mm256_set_epi64x(INT64_C(-3226888659503117201), INT64_C( 7490209482650655404),
                             INT64_C(-9179276487306987344), INT64_C( 7055682156038845095)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C( 7490209482650655404),
                            INT64_C(                   0), INT64_C( 7055682156038845095),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 7055682156038845095)) },
    { UINT8_C(230),
      easysimd_mm256_set_epi64x(INT64_C( 6952848743567724070), INT64_C( 6398498157984007660),
                             INT64_C(-7276216502972313781), INT64_C( 4842545408380684085)),
      easysimd_mm512_set_epi64(INT64_C( 6952848743567724070), INT64_C( 6398498157984007660),
                            INT64_C(-7276216502972313781), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 6398498157984007660),
                            INT64_C(-7276216502972313781), INT64_C(                   0)) },
    { UINT8_C(115),
      easysimd_mm256_set_epi64x(INT64_C( -147426939517817059), INT64_C(-3374766540151601501),
                             INT64_C( 9013437962204473886), INT64_C( 2290211861166994880)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(-3374766540151601501),
                            INT64_C( 9013437962204473886), INT64_C( 2290211861166994880),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 9013437962204473886), INT64_C( 2290211861166994880)) },
    { UINT8_C(102),
      easysimd_mm256_set_epi64x(INT64_C(-8700458333795307779), INT64_C(-9147297996573979024),
                             INT64_C(-3649385965919135635), INT64_C( 1818037113458506686)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(-9147297996573979024),
                            INT64_C(-3649385965919135635), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(-9147297996573979024),
                            INT64_C(-3649385965919135635), INT64_C(                   0)) },
    { UINT8_C( 59),
      easysimd_mm256_set_epi64x(INT64_C( 8763762661767364639), INT64_C(-7194784414741958081),
                             INT64_C(-1605849263772874289), INT64_C(-2187551180549076287)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-1605849263772874289), INT64_C(-2187551180549076287),
                            INT64_C( 8763762661767364639), INT64_C(                   0),
                            INT64_C(-1605849263772874289), INT64_C(-2187551180549076287)) },
    { UINT8_C(119),
      easysimd_mm256_set_epi64x(INT64_C( 3282428208913039389), INT64_C(-2887297167729747289),
                             INT64_C( 6938672003976555894), INT64_C(-3765766577293323049)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(-2887297167729747289),
                            INT64_C( 6938672003976555894), INT64_C(-3765766577293323049),
                            INT64_C(                   0), INT64_C(-2887297167729747289),
                            INT64_C( 6938672003976555894), INT64_C(-3765766577293323049)) },
    { UINT8_C( 25),
      easysimd_mm256_set_epi64x(INT64_C(-4802008903577488206), INT64_C(-3983516919532966210),
                             INT64_C(-4702094198572773446), INT64_C( -958715043139892800)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( -958715043139892800),
                            INT64_C(-4802008903577488206), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( -958715043139892800)) },
    { UINT8_C(207),
      easysimd_mm256_set_epi64x(INT64_C( 2289318697780797186), INT64_C(-4515948424499803858),
                             INT64_C( 7316310196690749623), INT64_C( 4937967944726422430)),
      easysimd_mm512_set_epi64(INT64_C( 2289318697780797186), INT64_C(-4515948424499803858),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 2289318697780797186), INT64_C(-4515948424499803858),
                            INT64_C( 7316310196690749623), INT64_C( 4937967944726422430)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_broadcast_i64x4(test_vec[i].k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_broadcast_i64x4");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcastd_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm_set_epi32(INT32_C(-1051270324), INT32_C(-1977183446), INT32_C( -548195640), INT32_C(-1363461466)),
      easysimd_mm512_set_epi32(INT32_C(-1363461466), INT32_C(-1363461466), INT32_C(-1363461466), INT32_C(-1363461466),
                            INT32_C(-1363461466), INT32_C(-1363461466), INT32_C(-1363461466), INT32_C(-1363461466),
                            INT32_C(-1363461466), INT32_C(-1363461466), INT32_C(-1363461466), INT32_C(-1363461466),
                            INT32_C(-1363461466), INT32_C(-1363461466), INT32_C(-1363461466), INT32_C(-1363461466)) },
    { easysimd_mm_set_epi32(INT32_C(  979094891), INT32_C(  416506319), INT32_C( 2123490297), INT32_C(  200388421)),
      easysimd_mm512_set_epi32(INT32_C(  200388421), INT32_C(  200388421), INT32_C(  200388421), INT32_C(  200388421),
                            INT32_C(  200388421), INT32_C(  200388421), INT32_C(  200388421), INT32_C(  200388421),
                            INT32_C(  200388421), INT32_C(  200388421), INT32_C(  200388421), INT32_C(  200388421),
                            INT32_C(  200388421), INT32_C(  200388421), INT32_C(  200388421), INT32_C(  200388421)) },
    { easysimd_mm_set_epi32(INT32_C( 1927260635), INT32_C( 1201458882), INT32_C(-1448742498), INT32_C(-1111904220)),
      easysimd_mm512_set_epi32(INT32_C(-1111904220), INT32_C(-1111904220), INT32_C(-1111904220), INT32_C(-1111904220),
                            INT32_C(-1111904220), INT32_C(-1111904220), INT32_C(-1111904220), INT32_C(-1111904220),
                            INT32_C(-1111904220), INT32_C(-1111904220), INT32_C(-1111904220), INT32_C(-1111904220),
                            INT32_C(-1111904220), INT32_C(-1111904220), INT32_C(-1111904220), INT32_C(-1111904220)) },
    { easysimd_mm_set_epi32(INT32_C( -976455818), INT32_C(  542613123), INT32_C(  -15911923), INT32_C( -562895064)),
      easysimd_mm512_set_epi32(INT32_C( -562895064), INT32_C( -562895064), INT32_C( -562895064), INT32_C( -562895064),
                            INT32_C( -562895064), INT32_C( -562895064), INT32_C( -562895064), INT32_C( -562895064),
                            INT32_C( -562895064), INT32_C( -562895064), INT32_C( -562895064), INT32_C( -562895064),
                            INT32_C( -562895064), INT32_C( -562895064), INT32_C( -562895064), INT32_C( -562895064)) },
    { easysimd_mm_set_epi32(INT32_C(  836747087), INT32_C(-1431045412), INT32_C(-1356396683), INT32_C( 1489138473)),
      easysimd_mm512_set_epi32(INT32_C( 1489138473), INT32_C( 1489138473), INT32_C( 1489138473), INT32_C( 1489138473),
                            INT32_C( 1489138473), INT32_C( 1489138473), INT32_C( 1489138473), INT32_C( 1489138473),
                            INT32_C( 1489138473), INT32_C( 1489138473), INT32_C( 1489138473), INT32_C( 1489138473),
                            INT32_C( 1489138473), INT32_C( 1489138473), INT32_C( 1489138473), INT32_C( 1489138473)) },
    { easysimd_mm_set_epi32(INT32_C(-1783426961), INT32_C( -263517415), INT32_C(-1697630001), INT32_C( 2025142863)),
      easysimd_mm512_set_epi32(INT32_C( 2025142863), INT32_C( 2025142863), INT32_C( 2025142863), INT32_C( 2025142863),
                            INT32_C( 2025142863), INT32_C( 2025142863), INT32_C( 2025142863), INT32_C( 2025142863),
                            INT32_C( 2025142863), INT32_C( 2025142863), INT32_C( 2025142863), INT32_C( 2025142863),
                            INT32_C( 2025142863), INT32_C( 2025142863), INT32_C( 2025142863), INT32_C( 2025142863)) },
    { easysimd_mm_set_epi32(INT32_C(  300619496), INT32_C( -659754204), INT32_C(-1019736463), INT32_C( 1022872166)),
      easysimd_mm512_set_epi32(INT32_C( 1022872166), INT32_C( 1022872166), INT32_C( 1022872166), INT32_C( 1022872166),
                            INT32_C( 1022872166), INT32_C( 1022872166), INT32_C( 1022872166), INT32_C( 1022872166),
                            INT32_C( 1022872166), INT32_C( 1022872166), INT32_C( 1022872166), INT32_C( 1022872166),
                            INT32_C( 1022872166), INT32_C( 1022872166), INT32_C( 1022872166), INT32_C( 1022872166)) },
    { easysimd_mm_set_epi32(INT32_C( -274893610), INT32_C(  171227717), INT32_C( 1187872667), INT32_C( -590903223)),
      easysimd_mm512_set_epi32(INT32_C( -590903223), INT32_C( -590903223), INT32_C( -590903223), INT32_C( -590903223),
                            INT32_C( -590903223), INT32_C( -590903223), INT32_C( -590903223), INT32_C( -590903223),
                            INT32_C( -590903223), INT32_C( -590903223), INT32_C( -590903223), INT32_C( -590903223),
                            INT32_C( -590903223), INT32_C( -590903223), INT32_C( -590903223), INT32_C( -590903223)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_broadcastd_epi32(test_vec[i].a);
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcastd_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask16 k;
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( 1638944021), INT32_C( -385149059), INT32_C(  852916680), INT32_C(-1839015366),
                            INT32_C( 1146921463), INT32_C(  765234486), INT32_C( -388218844), INT32_C(-1402803832),
                            INT32_C( 1245942358), INT32_C( 2001202713), INT32_C(  868062804), INT32_C(-1988191751),
                            INT32_C(  807099340), INT32_C(  -38350755), INT32_C( -443928971), INT32_C( -432554813)),
      UINT16_C(24594),
      easysimd_mm_set_epi32(INT32_C( -255909174), INT32_C(-1302917278), INT32_C(  327520540), INT32_C(  176606543)),
      easysimd_mm512_set_epi32(INT32_C( 1638944021), INT32_C(  176606543), INT32_C(  176606543), INT32_C(-1839015366),
                            INT32_C( 1146921463), INT32_C(  765234486), INT32_C( -388218844), INT32_C(-1402803832),
                            INT32_C( 1245942358), INT32_C( 2001202713), INT32_C(  868062804), INT32_C(  176606543),
                            INT32_C(  807099340), INT32_C(  -38350755), INT32_C(  176606543), INT32_C( -432554813)) },
    { easysimd_mm512_set_epi32(INT32_C( -115460801), INT32_C( 1889676725), INT32_C(    2912775), INT32_C(-1289469215),
                            INT32_C( 1033489041), INT32_C(  147853139), INT32_C(  706073024), INT32_C( -130092746),
                            INT32_C( -799642653), INT32_C(-1439962375), INT32_C(-1798405841), INT32_C( 1190396108),
                            INT32_C(-1013986568), INT32_C(  994541610), INT32_C(-1127995400), INT32_C( 1108325476)),
      UINT16_C(40849),
      easysimd_mm_set_epi32(INT32_C(  250706831), INT32_C( -936079925), INT32_C(-1129184131), INT32_C(  803417186)),
      easysimd_mm512_set_epi32(INT32_C(  803417186), INT32_C( 1889676725), INT32_C(    2912775), INT32_C(  803417186),
                            INT32_C(  803417186), INT32_C(  803417186), INT32_C(  803417186), INT32_C(  803417186),
                            INT32_C(  803417186), INT32_C(-1439962375), INT32_C(-1798405841), INT32_C(  803417186),
                            INT32_C(-1013986568), INT32_C(  994541610), INT32_C(-1127995400), INT32_C(  803417186)) },
    { easysimd_mm512_set_epi32(INT32_C(  357625867), INT32_C( -157238200), INT32_C(  909767636), INT32_C( 1422277073),
                            INT32_C( 2123935701), INT32_C(-1040550911), INT32_C(  686758291), INT32_C(-2090356905),
                            INT32_C( -362358815), INT32_C( -482453842), INT32_C(  117787421), INT32_C( 1300554279),
                            INT32_C(-1085613264), INT32_C( -109297466), INT32_C(-1230203271), INT32_C(-1731521429)),
      UINT16_C(53728),
      easysimd_mm_set_epi32(INT32_C( -707786971), INT32_C( 1712040202), INT32_C(-2012675757), INT32_C(-1396559749)),
      easysimd_mm512_set_epi32(INT32_C(-1396559749), INT32_C(-1396559749), INT32_C(  909767636), INT32_C(-1396559749),
                            INT32_C( 2123935701), INT32_C(-1040550911), INT32_C(  686758291), INT32_C(-1396559749),
                            INT32_C(-1396559749), INT32_C(-1396559749), INT32_C(-1396559749), INT32_C( 1300554279),
                            INT32_C(-1085613264), INT32_C( -109297466), INT32_C(-1230203271), INT32_C(-1731521429)) },
    { easysimd_mm512_set_epi32(INT32_C( 2041534605), INT32_C( 1255681923), INT32_C( 1220121473), INT32_C( 1819952522),
                            INT32_C(-1737362693), INT32_C(  712438877), INT32_C(-1234448370), INT32_C(  217554028),
                            INT32_C(-1878093154), INT32_C( -741869417), INT32_C(  943666007), INT32_C(  622675686),
                            INT32_C( -269910912), INT32_C(  137195559), INT32_C(  469574756), INT32_C( 1490101689)),
      UINT16_C(50038),
      easysimd_mm_set_epi32(INT32_C( -272719467), INT32_C( -594597983), INT32_C( -820913821), INT32_C(  345700481)),
      easysimd_mm512_set_epi32(INT32_C(  345700481), INT32_C(  345700481), INT32_C( 1220121473), INT32_C( 1819952522),
                            INT32_C(-1737362693), INT32_C(  712438877), INT32_C(  345700481), INT32_C(  345700481),
                            INT32_C(-1878093154), INT32_C(  345700481), INT32_C(  345700481), INT32_C(  345700481),
                            INT32_C( -269910912), INT32_C(  345700481), INT32_C(  345700481), INT32_C( 1490101689)) },
    { easysimd_mm512_set_epi32(INT32_C(  605201121), INT32_C(    2188130), INT32_C( -956406632), INT32_C(-1144421408),
                            INT32_C(-2008693903), INT32_C( 1823632430), INT32_C( 2043624683), INT32_C(  457225971),
                            INT32_C( 1484257119), INT32_C(  719932227), INT32_C( 1722430058), INT32_C(  916001650),
                            INT32_C(  553469699), INT32_C(-2003831430), INT32_C(-1834906502), INT32_C(  225358926)),
      UINT16_C(22657),
      easysimd_mm_set_epi32(INT32_C(  290541765), INT32_C( -479926223), INT32_C( 2079119915), INT32_C( -331512500)),
      easysimd_mm512_set_epi32(INT32_C(  605201121), INT32_C( -331512500), INT32_C( -956406632), INT32_C( -331512500),
                            INT32_C( -331512500), INT32_C( 1823632430), INT32_C( 2043624683), INT32_C(  457225971),
                            INT32_C( -331512500), INT32_C(  719932227), INT32_C( 1722430058), INT32_C(  916001650),
                            INT32_C(  553469699), INT32_C(-2003831430), INT32_C(-1834906502), INT32_C( -331512500)) },
    { easysimd_mm512_set_epi32(INT32_C( -545987817), INT32_C(-1146550995), INT32_C(  963048631), INT32_C( -701605919),
                            INT32_C(  432096480), INT32_C(-2030393254), INT32_C(-1236899565), INT32_C(-1697034971),
                            INT32_C( -998012960), INT32_C(-1579141793), INT32_C( 1664269708), INT32_C( -667117157),
                            INT32_C( -708117814), INT32_C(   85211107), INT32_C(  909670673), INT32_C( 1616737139)),
      UINT16_C( 4531),
      easysimd_mm_set_epi32(INT32_C( -503580732), INT32_C(-1790221512), INT32_C(-1663970343), INT32_C( 1633501790)),
      easysimd_mm512_set_epi32(INT32_C( -545987817), INT32_C(-1146550995), INT32_C(  963048631), INT32_C( 1633501790),
                            INT32_C(  432096480), INT32_C(-2030393254), INT32_C(-1236899565), INT32_C( 1633501790),
                            INT32_C( 1633501790), INT32_C(-1579141793), INT32_C( 1633501790), INT32_C( 1633501790),
                            INT32_C( -708117814), INT32_C(   85211107), INT32_C( 1633501790), INT32_C( 1633501790)) },
    { easysimd_mm512_set_epi32(INT32_C(-1668661089), INT32_C( 1895031925), INT32_C( 2107029353), INT32_C(-1915428586),
                            INT32_C(  963718296), INT32_C( 1878898594), INT32_C( -403168746), INT32_C(  502390291),
                            INT32_C( 1855826407), INT32_C(-1442018177), INT32_C( -244961355), INT32_C( 1777042193),
                            INT32_C(  373997996), INT32_C( -684064874), INT32_C(  930695451), INT32_C(-1073438864)),
      UINT16_C(53861),
      easysimd_mm_set_epi32(INT32_C( 1599859635), INT32_C(  543659234), INT32_C(-1222091200), INT32_C(  817594139)),
      easysimd_mm512_set_epi32(INT32_C(  817594139), INT32_C(  817594139), INT32_C( 2107029353), INT32_C(  817594139),
                            INT32_C(  963718296), INT32_C( 1878898594), INT32_C(  817594139), INT32_C(  502390291),
                            INT32_C( 1855826407), INT32_C(  817594139), INT32_C(  817594139), INT32_C( 1777042193),
                            INT32_C(  373997996), INT32_C(  817594139), INT32_C(  930695451), INT32_C(  817594139)) },
    { easysimd_mm512_set_epi32(INT32_C( -831807470), INT32_C( -591553083), INT32_C( -492649784), INT32_C(-1394371521),
                            INT32_C(-1760655625), INT32_C( 2135736563), INT32_C(-2075134444), INT32_C( -933317766),
                            INT32_C( -731013025), INT32_C(-2091361347), INT32_C( 1562364760), INT32_C( -612070110),
                            INT32_C( 1365385309), INT32_C( -121237183), INT32_C( 1543044931), INT32_C(-1490381593)),
      UINT16_C(20921),
      easysimd_mm_set_epi32(INT32_C(-1466503600), INT32_C(  824864478), INT32_C(-1491396230), INT32_C(-1907140086)),
      easysimd_mm512_set_epi32(INT32_C( -831807470), INT32_C(-1907140086), INT32_C( -492649784), INT32_C(-1907140086),
                            INT32_C(-1760655625), INT32_C( 2135736563), INT32_C(-2075134444), INT32_C(-1907140086),
                            INT32_C(-1907140086), INT32_C(-2091361347), INT32_C(-1907140086), INT32_C(-1907140086),
                            INT32_C(-1907140086), INT32_C( -121237183), INT32_C( 1543044931), INT32_C(-1907140086)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_mask_broadcastd_epi32(test_vec[i].src, test_vec[i].k, test_vec[i].a);
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcastd_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT16_C(21274),
      easysimd_mm_set_epi32(INT32_C( 1459257075), INT32_C(  587801532), INT32_C( 1631678564), INT32_C(  715337051)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  715337051), INT32_C(          0), INT32_C(  715337051),
                            INT32_C(          0), INT32_C(          0), INT32_C(  715337051), INT32_C(  715337051),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(  715337051),
                            INT32_C(  715337051), INT32_C(          0), INT32_C(  715337051), INT32_C(          0)) },
    { UINT16_C(59357),
      easysimd_mm_set_epi32(INT32_C(-2022546688), INT32_C( 2145084340), INT32_C(   29275255), INT32_C( -827125259)),
      easysimd_mm512_set_epi32(INT32_C( -827125259), INT32_C( -827125259), INT32_C( -827125259), INT32_C(          0),
                            INT32_C(          0), INT32_C( -827125259), INT32_C( -827125259), INT32_C( -827125259),
                            INT32_C( -827125259), INT32_C( -827125259), INT32_C(          0), INT32_C( -827125259),
                            INT32_C( -827125259), INT32_C( -827125259), INT32_C(          0), INT32_C( -827125259)) },
    { UINT16_C(34446),
      easysimd_mm_set_epi32(INT32_C(  973425906), INT32_C( -935954345), INT32_C( 1285315081), INT32_C( 2142489532)),
      easysimd_mm512_set_epi32(INT32_C( 2142489532), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( 2142489532), INT32_C( 2142489532), INT32_C(          0),
                            INT32_C( 2142489532), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C( 2142489532), INT32_C( 2142489532), INT32_C( 2142489532), INT32_C(          0)) },
    { UINT16_C(33955),
      easysimd_mm_set_epi32(INT32_C(-1114656122), INT32_C( 1221674060), INT32_C( -740975665), INT32_C( 2132760332)),
      easysimd_mm512_set_epi32(INT32_C( 2132760332), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C( 2132760332), INT32_C(          0), INT32_C(          0),
                            INT32_C( 2132760332), INT32_C(          0), INT32_C( 2132760332), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C( 2132760332), INT32_C( 2132760332)) },
    { UINT16_C(52572),
      easysimd_mm_set_epi32(INT32_C( -724774954), INT32_C( -166426332), INT32_C(-1571631693), INT32_C( -124417294)),
      easysimd_mm512_set_epi32(INT32_C( -124417294), INT32_C( -124417294), INT32_C(          0), INT32_C(          0),
                            INT32_C( -124417294), INT32_C( -124417294), INT32_C(          0), INT32_C( -124417294),
                            INT32_C(          0), INT32_C( -124417294), INT32_C(          0), INT32_C( -124417294),
                            INT32_C( -124417294), INT32_C( -124417294), INT32_C(          0), INT32_C(          0)) },
    { UINT16_C(38931),
      easysimd_mm_set_epi32(INT32_C(-1992244525), INT32_C( -292982508), INT32_C( -691380397), INT32_C(-1292068161)),
      easysimd_mm512_set_epi32(INT32_C(-1292068161), INT32_C(          0), INT32_C(          0), INT32_C(-1292068161),
                            INT32_C(-1292068161), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(-1292068161),
                            INT32_C(          0), INT32_C(          0), INT32_C(-1292068161), INT32_C(-1292068161)) },
    { UINT16_C(32377),
      easysimd_mm_set_epi32(INT32_C( -766689829), INT32_C(-1724046912), INT32_C( 1799018744), INT32_C(  623047724)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  623047724), INT32_C(  623047724), INT32_C(  623047724),
                            INT32_C(  623047724), INT32_C(  623047724), INT32_C(  623047724), INT32_C(          0),
                            INT32_C(          0), INT32_C(  623047724), INT32_C(  623047724), INT32_C(  623047724),
                            INT32_C(  623047724), INT32_C(          0), INT32_C(          0), INT32_C(  623047724)) },
    { UINT16_C(18782),
      easysimd_mm_set_epi32(INT32_C(-2020669200), INT32_C( -170583969), INT32_C( -628885190), INT32_C(  818636447)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  818636447), INT32_C(          0), INT32_C(          0),
                            INT32_C(  818636447), INT32_C(          0), INT32_C(          0), INT32_C(  818636447),
                            INT32_C(          0), INT32_C(  818636447), INT32_C(          0), INT32_C(  818636447),
                            INT32_C(  818636447), INT32_C(  818636447), INT32_C(  818636447), INT32_C(          0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_maskz_broadcastd_epi32(test_vec[i].k, test_vec[i].a);
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcastq_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm_set_epi64x(INT64_C(-4515171658517540054), INT64_C(-2354482342678283610)),
      easysimd_mm512_set_epi64(INT64_C(-2354482342678283610), INT64_C(-2354482342678283610),
                            INT64_C(-2354482342678283610), INT64_C(-2354482342678283610),
                            INT64_C(-2354482342678283610), INT64_C(-2354482342678283610),
                            INT64_C(-2354482342678283610), INT64_C(-2354482342678283610)) },
    { easysimd_mm_set_epi64x(INT64_C( 4205180536942191055), INT64_C( 9120321379188715333)),
      easysimd_mm512_set_epi64(INT64_C( 9120321379188715333), INT64_C( 9120321379188715333),
                            INT64_C( 9120321379188715333), INT64_C( 9120321379188715333),
                            INT64_C( 9120321379188715333), INT64_C( 9120321379188715333),
                            INT64_C( 9120321379188715333), INT64_C( 9120321379188715333)) },
    { easysimd_mm_set_epi64x(INT64_C( 8277521399394651842), INT64_C(-6222301646052282332)),
      easysimd_mm512_set_epi64(INT64_C(-6222301646052282332), INT64_C(-6222301646052282332),
                            INT64_C(-6222301646052282332), INT64_C(-6222301646052282332),
                            INT64_C(-6222301646052282332), INT64_C(-6222301646052282332),
                            INT64_C(-6222301646052282332), INT64_C(-6222301646052282332)) },
    { easysimd_mm_set_epi64x(INT64_C(-4193845803756315005), INT64_C(  -68341185169397976)),
      easysimd_mm512_set_epi64(INT64_C(  -68341185169397976), INT64_C(  -68341185169397976),
                            INT64_C(  -68341185169397976), INT64_C(  -68341185169397976),
                            INT64_C(  -68341185169397976), INT64_C(  -68341185169397976),
                            INT64_C(  -68341185169397976), INT64_C(  -68341185169397976)) },
    { easysimd_mm_set_epi64x(INT64_C( 3593801376552188636), INT64_C(-5825679392398740695)),
      easysimd_mm512_set_epi64(INT64_C(-5825679392398740695), INT64_C(-5825679392398740695),
                            INT64_C(-5825679392398740695), INT64_C(-5825679392398740695),
                            INT64_C(-5825679392398740695), INT64_C(-5825679392398740695),
                            INT64_C(-5825679392398740695), INT64_C(-5825679392398740695)) },
    { easysimd_mm_set_epi64x(INT64_C(-7659760468268217575), INT64_C(-7291265332978304433)),
      easysimd_mm512_set_epi64(INT64_C(-7291265332978304433), INT64_C(-7291265332978304433),
                            INT64_C(-7291265332978304433), INT64_C(-7291265332978304433),
                            INT64_C(-7291265332978304433), INT64_C(-7291265332978304433),
                            INT64_C(-7291265332978304433), INT64_C(-7291265332978304433)) },
    { easysimd_mm_set_epi64x(INT64_C( 1291150907495215908), INT64_C(-4379734758100841882)),
      easysimd_mm512_set_epi64(INT64_C(-4379734758100841882), INT64_C(-4379734758100841882),
                            INT64_C(-4379734758100841882), INT64_C(-4379734758100841882),
                            INT64_C(-4379734758100841882), INT64_C(-4379734758100841882),
                            INT64_C(-4379734758100841882), INT64_C(-4379734758100841882)) },
    { easysimd_mm_set_epi64x(INT64_C(-1180659064658150843), INT64_C( 5101874260281362505)),
      easysimd_mm512_set_epi64(INT64_C( 5101874260281362505), INT64_C( 5101874260281362505),
                            INT64_C( 5101874260281362505), INT64_C( 5101874260281362505),
                            INT64_C( 5101874260281362505), INT64_C( 5101874260281362505),
                            INT64_C( 5101874260281362505), INT64_C( 5101874260281362505)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_broadcastq_epi64(test_vec[i].a);
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcastq_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 7039210974079555453), INT64_C( 3663249249268849210),
                            INT64_C( 4925990175430708534), INT64_C(-1667387235778762360),
                            INT64_C( 5351281682312326681), INT64_C( 3728301356360833529),
                            INT64_C( 3466465274179801181), INT64_C(-1906660408329519933)),
      UINT8_C( 18),
      easysimd_mm_set_epi64x(INT64_C(-5595987098075819748), INT64_C(  758519329013942001)),
      easysimd_mm512_set_epi64(INT64_C( 7039210974079555453), INT64_C( 3663249249268849210),
                            INT64_C( 4925990175430708534), INT64_C(  758519329013942001),
                            INT64_C( 5351281682312326681), INT64_C( 3728301356360833529),
                            INT64_C(  758519329013942001), INT64_C(-1906660408329519933)) },
    { easysimd_mm512_set_epi64(INT64_C( 8116099733890298375), INT64_C(-5538228106590303599),
                            INT64_C(  635024397322015168), INT64_C( -558744086021510173),
                            INT64_C(-6184591305598926545), INT64_C( 5112712356426664696),
                            INT64_C( 4271523692628158456), INT64_C( 4760221676782691018)),
      UINT8_C( 63),
      easysimd_mm_set_epi64x(INT64_C(-4849808913003762590), INT64_C( 7901381612815228817)),
      easysimd_mm512_set_epi64(INT64_C( 8116099733890298375), INT64_C(-5538228106590303599),
                            INT64_C( 7901381612815228817), INT64_C( 7901381612815228817),
                            INT64_C( 7901381612815228817), INT64_C( 7901381612815228817),
                            INT64_C( 7901381612815228817), INT64_C( 7901381612815228817)) },
    { easysimd_mm512_set_epi64(INT64_C( 3907422245001509329), INT64_C( 9122234377856250881),
                            INT64_C( 2949604402306461527), INT64_C(-1556319256029800786),
                            INT64_C(  505893122375737895), INT64_C(-4662673460798144314),
                            INT64_C(-5283682813813779349), INT64_C( 1076777643387686347)),
      UINT8_C( 72),
      easysimd_mm_set_epi64x(INT64_C(-5998178448496319999), INT64_C( 7927410529462710283)),
      easysimd_mm512_set_epi64(INT64_C( 3907422245001509329), INT64_C( 7927410529462710283),
                            INT64_C( 2949604402306461527), INT64_C(-1556319256029800786),
                            INT64_C( 7927410529462710283), INT64_C(-4662673460798144314),
                            INT64_C(-5283682813813779349), INT64_C( 1076777643387686347)) },
    { easysimd_mm512_set_epi64(INT64_C( 7816636564820325115), INT64_C( 3059901680174485518),
                            INT64_C(  934387437789942430), INT64_C(-3186304882973920425),
                            INT64_C( 2674371711409421440), INT64_C(  589250439531013220),
                            INT64_C( 6399938025556543269), INT64_C( 7353156679309525331)),
      UINT8_C(129),
      easysimd_mm_set_epi64x(INT64_C(-3132262719190613130), INT64_C( 8768324363382960003)),
      easysimd_mm512_set_epi64(INT64_C( 8768324363382960003), INT64_C( 3059901680174485518),
                            INT64_C(  934387437789942430), INT64_C(-3186304882973920425),
                            INT64_C( 2674371711409421440), INT64_C(  589250439531013220),
                            INT64_C( 6399938025556543269), INT64_C( 8768324363382960003)) },
    { easysimd_mm512_set_epi64(INT64_C(-8627274619235963858), INT64_C( 8777301179240593139),
                            INT64_C( 6374835785680112451), INT64_C( 7397780769673384818),
                            INT64_C( 2377134258823099770), INT64_C(-7880863417082399666),
                            INT64_C(-1171321188047181919), INT64_C(-3525798013683697535)),
      UINT8_C(224),
      easysimd_mm_set_epi64x(INT64_C(-5077429793204296991), INT64_C(    9397950127957144)),
      easysimd_mm512_set_epi64(INT64_C(    9397950127957144), INT64_C(    9397950127957144),
                            INT64_C(    9397950127957144), INT64_C( 7397780769673384818),
                            INT64_C( 2377134258823099770), INT64_C(-7880863417082399666),
                            INT64_C(-1171321188047181919), INT64_C(-3525798013683697535)) },
    { easysimd_mm512_set_epi64(INT64_C(-8720472620890953453), INT64_C(-7288709697316354080),
                            INT64_C(-6782362355017532020), INT64_C(-2865246368328647990),
                            INT64_C(  365978918730627345), INT64_C( 6943833138524147909),
                            INT64_C(-2061267430198683093), INT64_C(-1423835345422209809)),
      UINT8_C(224),
      easysimd_mm_set_epi64x(INT64_C(-2344999814881016531), INT64_C( 4136262378195933153)),
      easysimd_mm512_set_epi64(INT64_C( 4136262378195933153), INT64_C( 4136262378195933153),
                            INT64_C( 4136262378195933153), INT64_C(-2865246368328647990),
                            INT64_C(  365978918730627345), INT64_C( 6943833138524147909),
                            INT64_C(-2061267430198683093), INT64_C(-1423835345422209809)) },
    { easysimd_mm512_set_epi64(INT64_C(-1731596578336940525), INT64_C( 7970713727971134591),
                            INT64_C(-1052101006731803887), INT64_C( 1606309165200441238),
                            INT64_C( 3997306527802498928), INT64_C(-2162862772330994888),
                            INT64_C(-7146698203065400738), INT64_C(-4784261768320577101)),
      UINT8_C(162),
      easysimd_mm_set_epi64x(INT64_C( 8139100144857954153), INT64_C(-8226703133729805160)),
      easysimd_mm512_set_epi64(INT64_C(-8226703133729805160), INT64_C( 7970713727971134591),
                            INT64_C(-8226703133729805160), INT64_C( 1606309165200441238),
                            INT64_C( 3997306527802498928), INT64_C(-2162862772330994888),
                            INT64_C(-8226703133729805160), INT64_C(-4784261768320577101)) },
    { easysimd_mm512_set_epi64(INT64_C(-4008569278181826465), INT64_C(-8982328587921142952),
                            INT64_C(-2628821103943737251), INT64_C( -520709734501122237),
                            INT64_C(-6401140198895522893), INT64_C( 2334998633271287360),
                            INT64_C( 3511540092651127844), INT64_C( 3766647997225123999)),
      UINT8_C( 20),
      easysimd_mm_set_epi64x(INT64_C(-2115914707760868289), INT64_C(-7561958326757703437)),
      easysimd_mm512_set_epi64(INT64_C(-4008569278181826465), INT64_C(-8982328587921142952),
                            INT64_C(-2628821103943737251), INT64_C(-7561958326757703437),
                            INT64_C(-6401140198895522893), INT64_C(-7561958326757703437),
                            INT64_C( 3511540092651127844), INT64_C( 3766647997225123999)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_mask_broadcastq_epi64(test_vec[i].src, test_vec[i].k, test_vec[i].a);
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcastq_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT8_C( 26),
      easysimd_mm_set_epi64x(INT64_C( 2524588358110376036), INT64_C( 3072349241054123220)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 3072349241054123220),
                            INT64_C( 3072349241054123220), INT64_C(                   0),
                            INT64_C( 3072349241054123220), INT64_C(                   0)) },
    { UINT8_C(243),
      easysimd_mm_set_epi64x(INT64_C(  125736266274902517), INT64_C( 4529119523676940253)),
      easysimd_mm512_set_epi64(INT64_C( 4529119523676940253), INT64_C( 4529119523676940253),
                            INT64_C( 4529119523676940253), INT64_C( 4529119523676940253),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 4529119523676940253), INT64_C( 4529119523676940253)) },
    { UINT8_C(180),
      easysimd_mm_set_epi64x(INT64_C( 9201922475629043961), INT64_C(-5256397243355602176)),
      easysimd_mm512_set_epi64(INT64_C(-5256397243355602176), INT64_C(                   0),
                            INT64_C(-5256397243355602176), INT64_C(-5256397243355602176),
                            INT64_C(                   0), INT64_C(-5256397243355602176),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(  9),
      easysimd_mm_set_epi64x(INT64_C( 4382010425855345827), INT64_C( 4180832434708183127)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 4180832434708183127), INT64_C(                   0),
                            INT64_C(                   0), INT64_C( 4180832434708183127)) },
    { UINT8_C( 12),
      easysimd_mm_set_epi64x(INT64_C(-1597707644585397626), INT64_C( 5247050137625533391)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 5247050137625533391), INT64_C( 5247050137625533391),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(150),
      easysimd_mm_set_epi64x(INT64_C(-3112884720261363420), INT64_C(-6750106718621562126)),
      easysimd_mm512_set_epi64(INT64_C(-6750106718621562126), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(-6750106718621562126),
                            INT64_C(                   0), INT64_C(-6750106718621562126),
                            INT64_C(-6750106718621562126), INT64_C(                   0)) },
    { UINT8_C( 19),
      easysimd_mm_set_epi64x(INT64_C(-1258350286556471469), INT64_C(-5549390491787734701)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(-5549390491787734701),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-5549390491787734701), INT64_C(-5549390491787734701)) },
    { UINT8_C(211),
      easysimd_mm_set_epi64x(INT64_C( 7726726670994043948), INT64_C( 5635717459582615161)),
      easysimd_mm512_set_epi64(INT64_C( 5635717459582615161), INT64_C( 5635717459582615161),
                            INT64_C(                   0), INT64_C( 5635717459582615161),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 5635717459582615161), INT64_C( 5635717459582615161)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_maskz_broadcastq_epi64(test_vec[i].k, test_vec[i].a);
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcastss_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   104.48), EASYSIMD_FLOAT32_C(   410.97), EASYSIMD_FLOAT32_C(   548.32), EASYSIMD_FLOAT32_C(   631.04)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04),
                         EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04),
                         EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04),
                         EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   668.18), EASYSIMD_FLOAT32_C(  -107.97), EASYSIMD_FLOAT32_C(  -627.99), EASYSIMD_FLOAT32_C(  -347.00)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00),
                         EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00),
                         EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00),
                         EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00), EASYSIMD_FLOAT32_C(  -347.00)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   236.12), EASYSIMD_FLOAT32_C(  -776.74), EASYSIMD_FLOAT32_C(   643.82), EASYSIMD_FLOAT32_C(  -941.79)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79),
                         EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79),
                         EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79),
                         EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79), EASYSIMD_FLOAT32_C(  -941.79)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -401.52), EASYSIMD_FLOAT32_C(   338.53), EASYSIMD_FLOAT32_C(  -725.48), EASYSIMD_FLOAT32_C(   387.06)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06),
                         EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06),
                         EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06),
                         EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(   387.06)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   119.04), EASYSIMD_FLOAT32_C(   263.81), EASYSIMD_FLOAT32_C(   717.18), EASYSIMD_FLOAT32_C(  -996.30)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30),
                         EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30),
                         EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30),
                         EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(  -996.30)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -278.68), EASYSIMD_FLOAT32_C(   120.15), EASYSIMD_FLOAT32_C(   751.98), EASYSIMD_FLOAT32_C(   536.33)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33),
                         EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33),
                         EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33),
                         EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(   536.33)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -143.65), EASYSIMD_FLOAT32_C(   810.77), EASYSIMD_FLOAT32_C(  -448.76), EASYSIMD_FLOAT32_C(   234.43)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43),
                         EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43),
                         EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43),
                         EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   234.43)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -42.20), EASYSIMD_FLOAT32_C(  -923.83), EASYSIMD_FLOAT32_C(   357.03), EASYSIMD_FLOAT32_C(  -933.51)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51),
                         EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51),
                         EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51),
                         EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -933.51)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 r = easysimd_mm512_broadcastss_ps(test_vec[i].a);
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcastss_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512 src;
    easysimd__mmask16 k;
    easysimd__m128 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -227.30), EASYSIMD_FLOAT32_C(   999.04), EASYSIMD_FLOAT32_C(   956.07), EASYSIMD_FLOAT32_C(  -270.40),
                         EASYSIMD_FLOAT32_C(   132.00), EASYSIMD_FLOAT32_C(   480.19), EASYSIMD_FLOAT32_C(  -107.97), EASYSIMD_FLOAT32_C(  -347.00),
                         EASYSIMD_FLOAT32_C(  -927.52), EASYSIMD_FLOAT32_C(   -67.87), EASYSIMD_FLOAT32_C(   891.86), EASYSIMD_FLOAT32_C(  -870.50),
                         EASYSIMD_FLOAT32_C(   932.69), EASYSIMD_FLOAT32_C(   244.86), EASYSIMD_FLOAT32_C(  -621.59), EASYSIMD_FLOAT32_C(    36.25)),
      UINT16_C(30253),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   134.75), EASYSIMD_FLOAT32_C(   871.12), EASYSIMD_FLOAT32_C(   104.48), EASYSIMD_FLOAT32_C(   548.32)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -227.30), EASYSIMD_FLOAT32_C(   548.32), EASYSIMD_FLOAT32_C(   548.32), EASYSIMD_FLOAT32_C(   548.32),
                         EASYSIMD_FLOAT32_C(   132.00), EASYSIMD_FLOAT32_C(   548.32), EASYSIMD_FLOAT32_C(   548.32), EASYSIMD_FLOAT32_C(  -347.00),
                         EASYSIMD_FLOAT32_C(  -927.52), EASYSIMD_FLOAT32_C(   -67.87), EASYSIMD_FLOAT32_C(   548.32), EASYSIMD_FLOAT32_C(  -870.50),
                         EASYSIMD_FLOAT32_C(   548.32), EASYSIMD_FLOAT32_C(   548.32), EASYSIMD_FLOAT32_C(  -621.59), EASYSIMD_FLOAT32_C(   548.32)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -996.30), EASYSIMD_FLOAT32_C(   389.31), EASYSIMD_FLOAT32_C(   670.35), EASYSIMD_FLOAT32_C(   396.13),
                         EASYSIMD_FLOAT32_C(  -971.67), EASYSIMD_FLOAT32_C(   528.69), EASYSIMD_FLOAT32_C(   275.37), EASYSIMD_FLOAT32_C(   338.53),
                         EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(    29.64), EASYSIMD_FLOAT32_C(   199.34), EASYSIMD_FLOAT32_C(  -686.40),
                         EASYSIMD_FLOAT32_C(   717.18), EASYSIMD_FLOAT32_C(   416.06), EASYSIMD_FLOAT32_C(   645.78), EASYSIMD_FLOAT32_C(  -990.79)),
      UINT16_C(37933),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   442.28), EASYSIMD_FLOAT32_C(   811.14), EASYSIMD_FLOAT32_C(  -767.79), EASYSIMD_FLOAT32_C(   236.12)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   236.12), EASYSIMD_FLOAT32_C(   389.31), EASYSIMD_FLOAT32_C(   670.35), EASYSIMD_FLOAT32_C(   236.12),
                         EASYSIMD_FLOAT32_C(  -971.67), EASYSIMD_FLOAT32_C(   236.12), EASYSIMD_FLOAT32_C(   275.37), EASYSIMD_FLOAT32_C(   338.53),
                         EASYSIMD_FLOAT32_C(   387.06), EASYSIMD_FLOAT32_C(    29.64), EASYSIMD_FLOAT32_C(   236.12), EASYSIMD_FLOAT32_C(  -686.40),
                         EASYSIMD_FLOAT32_C(   236.12), EASYSIMD_FLOAT32_C(   236.12), EASYSIMD_FLOAT32_C(   645.78), EASYSIMD_FLOAT32_C(   236.12)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   810.77), EASYSIMD_FLOAT32_C(   234.43), EASYSIMD_FLOAT32_C(   101.39), EASYSIMD_FLOAT32_C(  -366.10),
                         EASYSIMD_FLOAT32_C(   855.96), EASYSIMD_FLOAT32_C(   -55.56), EASYSIMD_FLOAT32_C(   896.89), EASYSIMD_FLOAT32_C(   697.60),
                         EASYSIMD_FLOAT32_C(   120.15), EASYSIMD_FLOAT32_C(   536.33), EASYSIMD_FLOAT32_C(  -156.71), EASYSIMD_FLOAT32_C(  -331.13),
                         EASYSIMD_FLOAT32_C(  -143.65), EASYSIMD_FLOAT32_C(  -448.76), EASYSIMD_FLOAT32_C(  -628.22), EASYSIMD_FLOAT32_C(   318.72)),
      UINT16_C(19701),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -248.84), EASYSIMD_FLOAT32_C(   566.99), EASYSIMD_FLOAT32_C(  -650.08), EASYSIMD_FLOAT32_C(  -460.40)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   810.77), EASYSIMD_FLOAT32_C(  -460.40), EASYSIMD_FLOAT32_C(   101.39), EASYSIMD_FLOAT32_C(  -366.10),
                         EASYSIMD_FLOAT32_C(  -460.40), EASYSIMD_FLOAT32_C(  -460.40), EASYSIMD_FLOAT32_C(   896.89), EASYSIMD_FLOAT32_C(   697.60),
                         EASYSIMD_FLOAT32_C(  -460.40), EASYSIMD_FLOAT32_C(  -460.40), EASYSIMD_FLOAT32_C(  -460.40), EASYSIMD_FLOAT32_C(  -460.40),
                         EASYSIMD_FLOAT32_C(  -143.65), EASYSIMD_FLOAT32_C(  -460.40), EASYSIMD_FLOAT32_C(  -628.22), EASYSIMD_FLOAT32_C(  -460.40)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   779.71), EASYSIMD_FLOAT32_C(   196.66), EASYSIMD_FLOAT32_C(    -0.50), EASYSIMD_FLOAT32_C(  -273.06),
                         EASYSIMD_FLOAT32_C(   429.50), EASYSIMD_FLOAT32_C(   650.80), EASYSIMD_FLOAT32_C(   509.10), EASYSIMD_FLOAT32_C(   709.57),
                         EASYSIMD_FLOAT32_C(  -561.64), EASYSIMD_FLOAT32_C(  -923.83), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -304.13),
                         EASYSIMD_FLOAT32_C(   728.72), EASYSIMD_FLOAT32_C(  -511.49), EASYSIMD_FLOAT32_C(   144.42), EASYSIMD_FLOAT32_C(   848.91)),
      UINT16_C(27468),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   949.19), EASYSIMD_FLOAT32_C(  -102.63), EASYSIMD_FLOAT32_C(    87.04), EASYSIMD_FLOAT32_C(   914.16)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   779.71), EASYSIMD_FLOAT32_C(   914.16), EASYSIMD_FLOAT32_C(   914.16), EASYSIMD_FLOAT32_C(  -273.06),
                         EASYSIMD_FLOAT32_C(   914.16), EASYSIMD_FLOAT32_C(   650.80), EASYSIMD_FLOAT32_C(   914.16), EASYSIMD_FLOAT32_C(   914.16),
                         EASYSIMD_FLOAT32_C(  -561.64), EASYSIMD_FLOAT32_C(   914.16), EASYSIMD_FLOAT32_C(  -933.51), EASYSIMD_FLOAT32_C(  -304.13),
                         EASYSIMD_FLOAT32_C(   914.16), EASYSIMD_FLOAT32_C(   914.16), EASYSIMD_FLOAT32_C(   144.42), EASYSIMD_FLOAT32_C(   848.91)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -960.76), EASYSIMD_FLOAT32_C(  -613.57), EASYSIMD_FLOAT32_C(   864.92), EASYSIMD_FLOAT32_C(   278.02),
                         EASYSIMD_FLOAT32_C(   573.37), EASYSIMD_FLOAT32_C(   393.40), EASYSIMD_FLOAT32_C(  -782.91), EASYSIMD_FLOAT32_C(  -933.90),
                         EASYSIMD_FLOAT32_C(  -291.87), EASYSIMD_FLOAT32_C(   382.75), EASYSIMD_FLOAT32_C(   -62.73), EASYSIMD_FLOAT32_C(   163.52),
                         EASYSIMD_FLOAT32_C(    87.09), EASYSIMD_FLOAT32_C(  -486.60), EASYSIMD_FLOAT32_C(  -157.79), EASYSIMD_FLOAT32_C(  -247.69)),
      UINT16_C(56353),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -97.06), EASYSIMD_FLOAT32_C(    -2.41), EASYSIMD_FLOAT32_C(   418.81), EASYSIMD_FLOAT32_C(  -141.42)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -141.42), EASYSIMD_FLOAT32_C(  -141.42), EASYSIMD_FLOAT32_C(   864.92), EASYSIMD_FLOAT32_C(  -141.42),
                         EASYSIMD_FLOAT32_C(  -141.42), EASYSIMD_FLOAT32_C(  -141.42), EASYSIMD_FLOAT32_C(  -782.91), EASYSIMD_FLOAT32_C(  -933.90),
                         EASYSIMD_FLOAT32_C(  -291.87), EASYSIMD_FLOAT32_C(   382.75), EASYSIMD_FLOAT32_C(  -141.42), EASYSIMD_FLOAT32_C(   163.52),
                         EASYSIMD_FLOAT32_C(    87.09), EASYSIMD_FLOAT32_C(  -486.60), EASYSIMD_FLOAT32_C(  -157.79), EASYSIMD_FLOAT32_C(  -141.42)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -490.98), EASYSIMD_FLOAT32_C(  -718.54), EASYSIMD_FLOAT32_C(  -172.08), EASYSIMD_FLOAT32_C(   476.19),
                         EASYSIMD_FLOAT32_C(  -825.45), EASYSIMD_FLOAT32_C(  -528.02), EASYSIMD_FLOAT32_C(  -604.26), EASYSIMD_FLOAT32_C(  -201.78),
                         EASYSIMD_FLOAT32_C(  -105.47), EASYSIMD_FLOAT32_C(   619.70), EASYSIMD_FLOAT32_C(   603.28), EASYSIMD_FLOAT32_C(  -553.28),
                         EASYSIMD_FLOAT32_C(   787.83), EASYSIMD_FLOAT32_C(  -945.21), EASYSIMD_FLOAT32_C(  -786.09), EASYSIMD_FLOAT32_C(   628.77)),
      UINT16_C(51486),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    54.48), EASYSIMD_FLOAT32_C(   679.92), EASYSIMD_FLOAT32_C(  -550.45), EASYSIMD_FLOAT32_C(  -482.87)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -482.87), EASYSIMD_FLOAT32_C(  -482.87), EASYSIMD_FLOAT32_C(  -172.08), EASYSIMD_FLOAT32_C(   476.19),
                         EASYSIMD_FLOAT32_C(  -482.87), EASYSIMD_FLOAT32_C(  -528.02), EASYSIMD_FLOAT32_C(  -604.26), EASYSIMD_FLOAT32_C(  -482.87),
                         EASYSIMD_FLOAT32_C(  -105.47), EASYSIMD_FLOAT32_C(   619.70), EASYSIMD_FLOAT32_C(   603.28), EASYSIMD_FLOAT32_C(  -482.87),
                         EASYSIMD_FLOAT32_C(  -482.87), EASYSIMD_FLOAT32_C(  -482.87), EASYSIMD_FLOAT32_C(  -482.87), EASYSIMD_FLOAT32_C(   628.77)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -124.63), EASYSIMD_FLOAT32_C(  -948.04), EASYSIMD_FLOAT32_C(  -344.11), EASYSIMD_FLOAT32_C(  -424.86),
                         EASYSIMD_FLOAT32_C(   640.76), EASYSIMD_FLOAT32_C(  -243.42), EASYSIMD_FLOAT32_C(   962.71), EASYSIMD_FLOAT32_C(   314.11),
                         EASYSIMD_FLOAT32_C(   599.88), EASYSIMD_FLOAT32_C(  -844.53), EASYSIMD_FLOAT32_C(  -530.48), EASYSIMD_FLOAT32_C(   563.54),
                         EASYSIMD_FLOAT32_C(   165.16), EASYSIMD_FLOAT32_C(   384.17), EASYSIMD_FLOAT32_C(   149.22), EASYSIMD_FLOAT32_C(   712.14)),
      UINT16_C(53759),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -284.59), EASYSIMD_FLOAT32_C(  -286.48), EASYSIMD_FLOAT32_C(  -340.65), EASYSIMD_FLOAT32_C(   563.88)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   563.88), EASYSIMD_FLOAT32_C(   563.88), EASYSIMD_FLOAT32_C(  -344.11), EASYSIMD_FLOAT32_C(   563.88),
                         EASYSIMD_FLOAT32_C(   640.76), EASYSIMD_FLOAT32_C(  -243.42), EASYSIMD_FLOAT32_C(   962.71), EASYSIMD_FLOAT32_C(   563.88),
                         EASYSIMD_FLOAT32_C(   563.88), EASYSIMD_FLOAT32_C(   563.88), EASYSIMD_FLOAT32_C(   563.88), EASYSIMD_FLOAT32_C(   563.88),
                         EASYSIMD_FLOAT32_C(   563.88), EASYSIMD_FLOAT32_C(   563.88), EASYSIMD_FLOAT32_C(   563.88), EASYSIMD_FLOAT32_C(   563.88)) },
    { easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -434.52), EASYSIMD_FLOAT32_C(   678.42), EASYSIMD_FLOAT32_C(   -65.20), EASYSIMD_FLOAT32_C(  -319.19),
                         EASYSIMD_FLOAT32_C(   664.97), EASYSIMD_FLOAT32_C(     9.01), EASYSIMD_FLOAT32_C(  -334.08), EASYSIMD_FLOAT32_C(  -870.44),
                         EASYSIMD_FLOAT32_C(   269.08), EASYSIMD_FLOAT32_C(  -345.75), EASYSIMD_FLOAT32_C(  -732.77), EASYSIMD_FLOAT32_C(   374.12),
                         EASYSIMD_FLOAT32_C(  -491.24), EASYSIMD_FLOAT32_C(   525.54), EASYSIMD_FLOAT32_C(  -178.26), EASYSIMD_FLOAT32_C(  -733.62)),
      UINT16_C(50870),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -300.63), EASYSIMD_FLOAT32_C(  -396.75), EASYSIMD_FLOAT32_C(   745.02), EASYSIMD_FLOAT32_C(   369.43)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   369.43), EASYSIMD_FLOAT32_C(   369.43), EASYSIMD_FLOAT32_C(   -65.20), EASYSIMD_FLOAT32_C(  -319.19),
                         EASYSIMD_FLOAT32_C(   664.97), EASYSIMD_FLOAT32_C(   369.43), EASYSIMD_FLOAT32_C(   369.43), EASYSIMD_FLOAT32_C(  -870.44),
                         EASYSIMD_FLOAT32_C(   369.43), EASYSIMD_FLOAT32_C(  -345.75), EASYSIMD_FLOAT32_C(   369.43), EASYSIMD_FLOAT32_C(   369.43),
                         EASYSIMD_FLOAT32_C(  -491.24), EASYSIMD_FLOAT32_C(   369.43), EASYSIMD_FLOAT32_C(   369.43), EASYSIMD_FLOAT32_C(  -733.62)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 src = test_vec[i].src;
    easysimd__m128 a = test_vec[i].a;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_broadcastss_ps(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_broadcastss_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcastss_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    easysimd__m128 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { UINT16_C(25371),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   104.48), EASYSIMD_FLOAT32_C(   410.97), EASYSIMD_FLOAT32_C(   548.32), EASYSIMD_FLOAT32_C(   631.04)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   631.04),
                         EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   631.04), EASYSIMD_FLOAT32_C(   631.04)) },
    { UINT16_C(49342),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -559.08), EASYSIMD_FLOAT32_C(   480.19), EASYSIMD_FLOAT32_C(   668.18), EASYSIMD_FLOAT32_C(  -107.97)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -107.97), EASYSIMD_FLOAT32_C(  -107.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -107.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -107.97), EASYSIMD_FLOAT32_C(  -107.97),
                         EASYSIMD_FLOAT32_C(  -107.97), EASYSIMD_FLOAT32_C(  -107.97), EASYSIMD_FLOAT32_C(  -107.97), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(24820),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   811.14), EASYSIMD_FLOAT32_C(  -333.00), EASYSIMD_FLOAT32_C(  -767.79), EASYSIMD_FLOAT32_C(   825.12)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   825.12), EASYSIMD_FLOAT32_C(   825.12), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   825.12), EASYSIMD_FLOAT32_C(   825.12), EASYSIMD_FLOAT32_C(   825.12), EASYSIMD_FLOAT32_C(   825.12),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   825.12), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(45881),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -610.29), EASYSIMD_FLOAT32_C(  -971.67), EASYSIMD_FLOAT32_C(   997.86), EASYSIMD_FLOAT32_C(   528.69)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   528.69), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   528.69), EASYSIMD_FLOAT32_C(   528.69),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   528.69), EASYSIMD_FLOAT32_C(   528.69),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   528.69), EASYSIMD_FLOAT32_C(   528.69),
                         EASYSIMD_FLOAT32_C(   528.69), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   528.69)) },
    { UINT16_C(28771),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -248.84), EASYSIMD_FLOAT32_C(   102.57), EASYSIMD_FLOAT32_C(   566.99), EASYSIMD_FLOAT32_C(   900.54)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   900.54), EASYSIMD_FLOAT32_C(   900.54), EASYSIMD_FLOAT32_C(   900.54),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   900.54), EASYSIMD_FLOAT32_C(   900.54), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   900.54), EASYSIMD_FLOAT32_C(   900.54)) },
    { UINT16_C(61611),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   318.72), EASYSIMD_FLOAT32_C(  -366.10), EASYSIMD_FLOAT32_C(   625.17), EASYSIMD_FLOAT32_C(   855.96)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   855.96), EASYSIMD_FLOAT32_C(   855.96), EASYSIMD_FLOAT32_C(   855.96), EASYSIMD_FLOAT32_C(   855.96),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   855.96), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   855.96), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   855.96), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   855.96), EASYSIMD_FLOAT32_C(   855.96)) },
    { UINT16_C(55548),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   324.92), EASYSIMD_FLOAT32_C(  -304.13), EASYSIMD_FLOAT32_C(   949.19), EASYSIMD_FLOAT32_C(   617.60)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   617.60), EASYSIMD_FLOAT32_C(   617.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   617.60),
                         EASYSIMD_FLOAT32_C(   617.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   617.60), EASYSIMD_FLOAT32_C(   617.60), EASYSIMD_FLOAT32_C(   617.60), EASYSIMD_FLOAT32_C(   617.60),
                         EASYSIMD_FLOAT32_C(   617.60), EASYSIMD_FLOAT32_C(   617.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT16_C(15841),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   144.42), EASYSIMD_FLOAT32_C(    -0.50), EASYSIMD_FLOAT32_C(   848.91), EASYSIMD_FLOAT32_C(  -273.06)),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -273.06), EASYSIMD_FLOAT32_C(  -273.06),
                         EASYSIMD_FLOAT32_C(  -273.06), EASYSIMD_FLOAT32_C(  -273.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -273.06),
                         EASYSIMD_FLOAT32_C(  -273.06), EASYSIMD_FLOAT32_C(  -273.06), EASYSIMD_FLOAT32_C(  -273.06), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -273.06)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 a = test_vec[i].a;
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_broadcastss_ps(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_broadcastss_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcastsd_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  912.41), EASYSIMD_FLOAT64_C(  842.49)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  842.49), EASYSIMD_FLOAT64_C(  842.49),
                         EASYSIMD_FLOAT64_C(  842.49), EASYSIMD_FLOAT64_C(  842.49),
                         EASYSIMD_FLOAT64_C(  842.49), EASYSIMD_FLOAT64_C(  842.49),
                         EASYSIMD_FLOAT64_C(  842.49), EASYSIMD_FLOAT64_C(  842.49)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -93.61), EASYSIMD_FLOAT64_C( -903.55)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -903.55), EASYSIMD_FLOAT64_C( -903.55),
                         EASYSIMD_FLOAT64_C( -903.55), EASYSIMD_FLOAT64_C( -903.55),
                         EASYSIMD_FLOAT64_C( -903.55), EASYSIMD_FLOAT64_C( -903.55),
                         EASYSIMD_FLOAT64_C( -903.55), EASYSIMD_FLOAT64_C( -903.55)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -219.93), EASYSIMD_FLOAT64_C( -754.32)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -754.32), EASYSIMD_FLOAT64_C( -754.32),
                         EASYSIMD_FLOAT64_C( -754.32), EASYSIMD_FLOAT64_C( -754.32),
                         EASYSIMD_FLOAT64_C( -754.32), EASYSIMD_FLOAT64_C( -754.32),
                         EASYSIMD_FLOAT64_C( -754.32), EASYSIMD_FLOAT64_C( -754.32)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  137.51), EASYSIMD_FLOAT64_C(  527.47)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  527.47), EASYSIMD_FLOAT64_C(  527.47),
                         EASYSIMD_FLOAT64_C(  527.47), EASYSIMD_FLOAT64_C(  527.47),
                         EASYSIMD_FLOAT64_C(  527.47), EASYSIMD_FLOAT64_C(  527.47),
                         EASYSIMD_FLOAT64_C(  527.47), EASYSIMD_FLOAT64_C(  527.47)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  596.82), EASYSIMD_FLOAT64_C(  365.41)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  365.41), EASYSIMD_FLOAT64_C(  365.41),
                         EASYSIMD_FLOAT64_C(  365.41), EASYSIMD_FLOAT64_C(  365.41),
                         EASYSIMD_FLOAT64_C(  365.41), EASYSIMD_FLOAT64_C(  365.41),
                         EASYSIMD_FLOAT64_C(  365.41), EASYSIMD_FLOAT64_C(  365.41)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -229.59), EASYSIMD_FLOAT64_C( -642.88)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -642.88), EASYSIMD_FLOAT64_C( -642.88),
                         EASYSIMD_FLOAT64_C( -642.88), EASYSIMD_FLOAT64_C( -642.88),
                         EASYSIMD_FLOAT64_C( -642.88), EASYSIMD_FLOAT64_C( -642.88),
                         EASYSIMD_FLOAT64_C( -642.88), EASYSIMD_FLOAT64_C( -642.88)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  308.04), EASYSIMD_FLOAT64_C( -958.64)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -958.64), EASYSIMD_FLOAT64_C( -958.64),
                         EASYSIMD_FLOAT64_C( -958.64), EASYSIMD_FLOAT64_C( -958.64),
                         EASYSIMD_FLOAT64_C( -958.64), EASYSIMD_FLOAT64_C( -958.64),
                         EASYSIMD_FLOAT64_C( -958.64), EASYSIMD_FLOAT64_C( -958.64)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  991.16), EASYSIMD_FLOAT64_C( -172.14)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -172.14), EASYSIMD_FLOAT64_C( -172.14),
                         EASYSIMD_FLOAT64_C( -172.14), EASYSIMD_FLOAT64_C( -172.14),
                         EASYSIMD_FLOAT64_C( -172.14), EASYSIMD_FLOAT64_C( -172.14),
                         EASYSIMD_FLOAT64_C( -172.14), EASYSIMD_FLOAT64_C( -172.14)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d r = easysimd_mm512_broadcastsd_pd(test_vec[i].a);
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcastsd_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512d src;
    easysimd__mmask8 k;
    easysimd__m128d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -746.75), EASYSIMD_FLOAT64_C(  634.39),
                         EASYSIMD_FLOAT64_C( -651.68), EASYSIMD_FLOAT64_C( -903.55),
                         EASYSIMD_FLOAT64_C(  689.73), EASYSIMD_FLOAT64_C(  178.89),
                         EASYSIMD_FLOAT64_C( -342.04), EASYSIMD_FLOAT64_C( -292.58)),
      UINT8_C(162),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   63.63), EASYSIMD_FLOAT64_C(  912.41)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  912.41), EASYSIMD_FLOAT64_C(  634.39),
                         EASYSIMD_FLOAT64_C(  912.41), EASYSIMD_FLOAT64_C( -903.55),
                         EASYSIMD_FLOAT64_C(  689.73), EASYSIMD_FLOAT64_C(  178.89),
                         EASYSIMD_FLOAT64_C(  912.41), EASYSIMD_FLOAT64_C( -292.58)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -287.03), EASYSIMD_FLOAT64_C(  851.62),
                         EASYSIMD_FLOAT64_C(  765.97), EASYSIMD_FLOAT64_C(  137.51),
                         EASYSIMD_FLOAT64_C( -457.60), EASYSIMD_FLOAT64_C(  815.46),
                         EASYSIMD_FLOAT64_C(  365.41), EASYSIMD_FLOAT64_C(  250.27)),
      UINT8_C( 66),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -149.06), EASYSIMD_FLOAT64_C( -899.78)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -287.03), EASYSIMD_FLOAT64_C( -899.78),
                         EASYSIMD_FLOAT64_C(  765.97), EASYSIMD_FLOAT64_C(  137.51),
                         EASYSIMD_FLOAT64_C( -457.60), EASYSIMD_FLOAT64_C(  815.46),
                         EASYSIMD_FLOAT64_C( -899.78), EASYSIMD_FLOAT64_C(  250.27)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -958.64), EASYSIMD_FLOAT64_C(  523.36),
                         EASYSIMD_FLOAT64_C( -361.34), EASYSIMD_FLOAT64_C( -153.87),
                         EASYSIMD_FLOAT64_C( -642.88), EASYSIMD_FLOAT64_C(  573.19),
                         EASYSIMD_FLOAT64_C(  308.04), EASYSIMD_FLOAT64_C(  -38.88)),
      UINT8_C(115),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -758.12), EASYSIMD_FLOAT64_C(   12.83)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -958.64), EASYSIMD_FLOAT64_C(   12.83),
                         EASYSIMD_FLOAT64_C(   12.83), EASYSIMD_FLOAT64_C(   12.83),
                         EASYSIMD_FLOAT64_C( -642.88), EASYSIMD_FLOAT64_C(  573.19),
                         EASYSIMD_FLOAT64_C(   12.83), EASYSIMD_FLOAT64_C(   12.83)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -221.94), EASYSIMD_FLOAT64_C(  499.58),
                         EASYSIMD_FLOAT64_C(   49.04), EASYSIMD_FLOAT64_C( -205.69),
                         EASYSIMD_FLOAT64_C(  991.16), EASYSIMD_FLOAT64_C( -984.94),
                         EASYSIMD_FLOAT64_C(  224.44), EASYSIMD_FLOAT64_C(  644.01)),
      UINT8_C(  4),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   53.80), EASYSIMD_FLOAT64_C( -691.82)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -221.94), EASYSIMD_FLOAT64_C(  499.58),
                         EASYSIMD_FLOAT64_C(   49.04), EASYSIMD_FLOAT64_C( -205.69),
                         EASYSIMD_FLOAT64_C(  991.16), EASYSIMD_FLOAT64_C( -691.82),
                         EASYSIMD_FLOAT64_C(  224.44), EASYSIMD_FLOAT64_C(  644.01)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -225.42), EASYSIMD_FLOAT64_C( -144.08),
                         EASYSIMD_FLOAT64_C( -549.59), EASYSIMD_FLOAT64_C(  465.78),
                         EASYSIMD_FLOAT64_C( -316.69), EASYSIMD_FLOAT64_C( -133.94),
                         EASYSIMD_FLOAT64_C( -646.50), EASYSIMD_FLOAT64_C(  160.17)),
      UINT8_C(172),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  841.59), EASYSIMD_FLOAT64_C(  843.47)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  843.47), EASYSIMD_FLOAT64_C( -144.08),
                         EASYSIMD_FLOAT64_C(  843.47), EASYSIMD_FLOAT64_C(  465.78),
                         EASYSIMD_FLOAT64_C(  843.47), EASYSIMD_FLOAT64_C(  843.47),
                         EASYSIMD_FLOAT64_C( -646.50), EASYSIMD_FLOAT64_C(  160.17)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -560.31), EASYSIMD_FLOAT64_C( -882.49),
                         EASYSIMD_FLOAT64_C(  -54.78), EASYSIMD_FLOAT64_C( -896.38),
                         EASYSIMD_FLOAT64_C(  607.65), EASYSIMD_FLOAT64_C( -296.43),
                         EASYSIMD_FLOAT64_C(  124.51), EASYSIMD_FLOAT64_C( -913.38)),
      UINT8_C(201),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  625.18), EASYSIMD_FLOAT64_C(   54.43)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   54.43), EASYSIMD_FLOAT64_C(   54.43),
                         EASYSIMD_FLOAT64_C(  -54.78), EASYSIMD_FLOAT64_C( -896.38),
                         EASYSIMD_FLOAT64_C(   54.43), EASYSIMD_FLOAT64_C( -296.43),
                         EASYSIMD_FLOAT64_C(  124.51), EASYSIMD_FLOAT64_C(   54.43)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  349.28), EASYSIMD_FLOAT64_C( -559.74),
                         EASYSIMD_FLOAT64_C( -116.49), EASYSIMD_FLOAT64_C(  342.49),
                         EASYSIMD_FLOAT64_C( -608.07), EASYSIMD_FLOAT64_C(  778.83),
                         EASYSIMD_FLOAT64_C( -284.17), EASYSIMD_FLOAT64_C( -113.81)),
      UINT8_C(234),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  312.83), EASYSIMD_FLOAT64_C(  -27.64)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -27.64), EASYSIMD_FLOAT64_C(  -27.64),
                         EASYSIMD_FLOAT64_C(  -27.64), EASYSIMD_FLOAT64_C(  342.49),
                         EASYSIMD_FLOAT64_C(  -27.64), EASYSIMD_FLOAT64_C(  778.83),
                         EASYSIMD_FLOAT64_C(  -27.64), EASYSIMD_FLOAT64_C( -113.81)) },
    { easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  364.84), EASYSIMD_FLOAT64_C(   86.19),
                         EASYSIMD_FLOAT64_C( -699.29), EASYSIMD_FLOAT64_C(  244.26),
                         EASYSIMD_FLOAT64_C( -206.27), EASYSIMD_FLOAT64_C( -921.17),
                         EASYSIMD_FLOAT64_C(  483.42), EASYSIMD_FLOAT64_C( -935.00)),
      UINT8_C( 12),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -328.98), EASYSIMD_FLOAT64_C(  803.91)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  364.84), EASYSIMD_FLOAT64_C(   86.19),
                         EASYSIMD_FLOAT64_C( -699.29), EASYSIMD_FLOAT64_C(  244.26),
                         EASYSIMD_FLOAT64_C(  803.91), EASYSIMD_FLOAT64_C(  803.91),
                         EASYSIMD_FLOAT64_C(  483.42), EASYSIMD_FLOAT64_C( -935.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d r = easysimd_mm512_mask_broadcastsd_pd(test_vec[i].src, test_vec[i].k, test_vec[i].a);
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcastsd_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    easysimd__m128d a;
    easysimd__m512d r;
  } test_vec[8] = {
    { UINT8_C(128),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  912.41), EASYSIMD_FLOAT64_C(  842.49)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  842.49), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(  2),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -651.68), EASYSIMD_FLOAT64_C(  -93.61)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  -93.61), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(216),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  881.57), EASYSIMD_FLOAT64_C( -899.78)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -899.78), EASYSIMD_FLOAT64_C( -899.78),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C( -899.78),
                         EASYSIMD_FLOAT64_C( -899.78), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) },
    { UINT8_C(183),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -292.94), EASYSIMD_FLOAT64_C(  765.97)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  765.97), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  765.97), EASYSIMD_FLOAT64_C(  765.97),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  765.97),
                         EASYSIMD_FLOAT64_C(  765.97), EASYSIMD_FLOAT64_C(  765.97)) },
    { UINT8_C(169),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -758.12), EASYSIMD_FLOAT64_C(  593.03)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  593.03), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  593.03), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  593.03), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(  593.03)) },
    { UINT8_C(243),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  523.36), EASYSIMD_FLOAT64_C(  761.91)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  761.91), EASYSIMD_FLOAT64_C(  761.91),
                         EASYSIMD_FLOAT64_C(  761.91), EASYSIMD_FLOAT64_C(  761.91),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  761.91), EASYSIMD_FLOAT64_C(  761.91)) },
    { UINT8_C(109),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -984.94), EASYSIMD_FLOAT64_C(   53.80)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   53.80),
                         EASYSIMD_FLOAT64_C(   53.80), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(   53.80), EASYSIMD_FLOAT64_C(   53.80),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(   53.80)) },
    { UINT8_C(168),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  644.01), EASYSIMD_FLOAT64_C(  499.58)),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  499.58), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  499.58), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(  499.58), EASYSIMD_FLOAT64_C(    0.00),
                         EASYSIMD_FLOAT64_C(    0.00), EASYSIMD_FLOAT64_C(    0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512d r = easysimd_mm512_maskz_broadcastsd_pd(test_vec[i].k, test_vec[i].a);
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcastb_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm_set_epi8(INT8_C( -17), INT8_C(  88), INT8_C(-122), INT8_C(-119),
                        INT8_C( 111), INT8_C(  87), INT8_C( -76), INT8_C(  27),
                        INT8_C( -93), INT8_C(  -8), INT8_C( -17), INT8_C(  24),
                        INT8_C(  69), INT8_C( 116), INT8_C( -78), INT8_C(-124)),
      easysimd_mm512_set_epi8(INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124),
                           INT8_C(-124), INT8_C(-124), INT8_C(-124), INT8_C(-124)) },
    { easysimd_mm_set_epi8(INT8_C(  -5), INT8_C(-114), INT8_C( -86), INT8_C(  -2),
                        INT8_C(  33), INT8_C(  90), INT8_C( -50), INT8_C(  28),
                        INT8_C(  15), INT8_C(  12), INT8_C(  32), INT8_C(  54),
                        INT8_C( -15), INT8_C( -57), INT8_C(  36), INT8_C( -50)),
      easysimd_mm512_set_epi8(INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50)) },
    { easysimd_mm_set_epi8(INT8_C( -49), INT8_C( -76), INT8_C( -62), INT8_C( 118),
                        INT8_C(  -4), INT8_C( -25), INT8_C( -58), INT8_C( 126),
                        INT8_C(-115), INT8_C( 126), INT8_C(-104), INT8_C( 127),
                        INT8_C(  15), INT8_C(  41), INT8_C(  68), INT8_C(  31)),
      easysimd_mm512_set_epi8(INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31),
                           INT8_C(  31), INT8_C(  31), INT8_C(  31), INT8_C(  31)) },
    { easysimd_mm_set_epi8(INT8_C( -30), INT8_C( -23), INT8_C( -42), INT8_C( -27),
                        INT8_C(-102), INT8_C(  -5), INT8_C( -87), INT8_C(  98),
                        INT8_C(  33), INT8_C(  73), INT8_C( 125), INT8_C( 120),
                        INT8_C( -70), INT8_C(  59), INT8_C( 124), INT8_C(  46)),
      easysimd_mm512_set_epi8(INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46),
                           INT8_C(  46), INT8_C(  46), INT8_C(  46), INT8_C(  46)) },
    { easysimd_mm_set_epi8(INT8_C( -18), INT8_C(  28), INT8_C( -19), INT8_C( -73),
                        INT8_C( -19), INT8_C(  67), INT8_C(  79), INT8_C( -45),
                        INT8_C(-124), INT8_C(  80), INT8_C(-101), INT8_C(-122),
                        INT8_C( -54), INT8_C(  30), INT8_C( -16), INT8_C(  55)),
      easysimd_mm512_set_epi8(INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55)) },
    { easysimd_mm_set_epi8(INT8_C(   6), INT8_C(  -5), INT8_C(  37), INT8_C( -97),
                        INT8_C(  16), INT8_C(  -5), INT8_C( -18), INT8_C(  14),
                        INT8_C(-120), INT8_C( -59), INT8_C( -43), INT8_C( -97),
                        INT8_C( -71), INT8_C( -73), INT8_C( -73), INT8_C( -50)),
      easysimd_mm512_set_epi8(INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C( -50), INT8_C( -50), INT8_C( -50)) },
    { easysimd_mm_set_epi8(INT8_C( 119), INT8_C(  60), INT8_C(  63), INT8_C( -26),
                        INT8_C(  50), INT8_C(  56), INT8_C(  40), INT8_C(  -7),
                        INT8_C(  68), INT8_C( -11), INT8_C( -21), INT8_C( -77),
                        INT8_C(  56), INT8_C(-109), INT8_C(-118), INT8_C(-108)),
      easysimd_mm512_set_epi8(INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108),
                           INT8_C(-108), INT8_C(-108), INT8_C(-108), INT8_C(-108)) },
    { easysimd_mm_set_epi8(INT8_C(-112), INT8_C(  65), INT8_C(  26), INT8_C( -90),
                        INT8_C( -77), INT8_C(  72), INT8_C(   2), INT8_C(   4),
                        INT8_C( -52), INT8_C( -82), INT8_C( -18), INT8_C( -66),
                        INT8_C(-118), INT8_C( -10), INT8_C(  52), INT8_C( -40)),
      easysimd_mm512_set_epi8(INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40),
                           INT8_C( -40), INT8_C( -40), INT8_C( -40), INT8_C( -40)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_broadcastb_epi8(test_vec[i].a);
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_broadcastb_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask64 k;
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C(  65), INT8_C( -68), INT8_C( 102), INT8_C(-122),
                           INT8_C(  40), INT8_C(  19), INT8_C(-111), INT8_C(   8),
                           INT8_C( -58), INT8_C(-120), INT8_C( 111), INT8_C(  10),
                           INT8_C( -54), INT8_C(-100), INT8_C(  36), INT8_C(  27),
                           INT8_C(-106), INT8_C(-123), INT8_C( 120), INT8_C(  43),
                           INT8_C( -31), INT8_C(   4), INT8_C(  10), INT8_C(  96),
                           INT8_C( -40), INT8_C(  23), INT8_C(  31), INT8_C(  73),
                           INT8_C( -51), INT8_C(  91), INT8_C(  68), INT8_C( -23),
                           INT8_C(-108), INT8_C(  52), INT8_C(  23), INT8_C( 115),
                           INT8_C(  -4), INT8_C(  24), INT8_C( 106), INT8_C( -92),
                           INT8_C( 126), INT8_C(  -6), INT8_C(  16), INT8_C( 102),
                           INT8_C( -47), INT8_C(-116), INT8_C(  -4), INT8_C(  33),
                           INT8_C( -25), INT8_C(-108), INT8_C(-115), INT8_C(-104),
                           INT8_C( -39), INT8_C(  49), INT8_C(  72), INT8_C(  44),
                           INT8_C( -17), INT8_C( -66), INT8_C( -61), INT8_C( -68),
                           INT8_C( 124), INT8_C(  20), INT8_C(  64), INT8_C( -49)),
      UINT64_C(          2970261907),
      easysimd_mm_set_epi8(INT8_C( -78), INT8_C( -60), INT8_C(-122), INT8_C( -66),
                        INT8_C(   8), INT8_C( -42), INT8_C(  44), INT8_C(  45),
                        INT8_C(  37), INT8_C(  -9), INT8_C( -14), INT8_C(  38),
                        INT8_C( -85), INT8_C(  35), INT8_C(   8), INT8_C(-127)),
      easysimd_mm512_set_epi8(INT8_C(  65), INT8_C( -68), INT8_C( 102), INT8_C(-122),
                           INT8_C(  40), INT8_C(  19), INT8_C(-111), INT8_C(   8),
                           INT8_C( -58), INT8_C(-120), INT8_C( 111), INT8_C(  10),
                           INT8_C( -54), INT8_C(-100), INT8_C(  36), INT8_C(  27),
                           INT8_C(-106), INT8_C(-123), INT8_C( 120), INT8_C(  43),
                           INT8_C( -31), INT8_C(   4), INT8_C(  10), INT8_C(  96),
                           INT8_C( -40), INT8_C(  23), INT8_C(  31), INT8_C(  73),
                           INT8_C( -51), INT8_C(  91), INT8_C(  68), INT8_C( -23),
                           INT8_C(-127), INT8_C(  52), INT8_C(-127), INT8_C(-127),
                           INT8_C(  -4), INT8_C(  24), INT8_C( 106), INT8_C(-127),
                           INT8_C( 126), INT8_C(  -6), INT8_C(  16), INT8_C( 102),
                           INT8_C(-127), INT8_C(-116), INT8_C(-127), INT8_C(  33),
                           INT8_C(-127), INT8_C(-108), INT8_C(-115), INT8_C(-127),
                           INT8_C(-127), INT8_C(  49), INT8_C(  72), INT8_C(-127),
                           INT8_C(-127), INT8_C( -66), INT8_C( -61), INT8_C(-127),
                           INT8_C( 124), INT8_C(  20), INT8_C(-127), INT8_C(-127)) },
    { easysimd_mm512_set_epi8(INT8_C( -76), INT8_C(  58), INT8_C(  70), INT8_C(-106),
                           INT8_C( 120), INT8_C(  72), INT8_C(  -4), INT8_C( -60),
                           INT8_C( 104), INT8_C( 101), INT8_C(  53), INT8_C(-121),
                           INT8_C( 102), INT8_C(-115), INT8_C(  90), INT8_C(  31),
                           INT8_C(  11), INT8_C(  68), INT8_C(  48), INT8_C(   4),
                           INT8_C(  55), INT8_C( -83), INT8_C(  75), INT8_C( -60),
                           INT8_C( -54), INT8_C(  57), INT8_C(  70), INT8_C(-114),
                           INT8_C(  51), INT8_C( -72), INT8_C( -71), INT8_C(  17),
                           INT8_C(  48), INT8_C(  40), INT8_C(-108), INT8_C( -12),
                           INT8_C( -11), INT8_C( -71), INT8_C(-114), INT8_C( -36),
                           INT8_C( -92), INT8_C( 101), INT8_C(  30), INT8_C(  10),
                           INT8_C(  43), INT8_C(-116), INT8_C( -45), INT8_C(-104),
                           INT8_C(  99), INT8_C( 108), INT8_C(  90), INT8_C(   7),
                           INT8_C( 112), INT8_C(  86), INT8_C(-125), INT8_C(  88),
                           INT8_C(  27), INT8_C(  40), INT8_C(  10), INT8_C(-105),
                           INT8_C(  76), INT8_C(-101), INT8_C(  87), INT8_C( 112)),
      UINT64_C(           458960523),
      easysimd_mm_set_epi8(INT8_C(  70), INT8_C( -73), INT8_C( -42), INT8_C( -19),
                        INT8_C( 110), INT8_C( -58), INT8_C(-117), INT8_C(-100),
                        INT8_C(  52), INT8_C( -63), INT8_C( -88), INT8_C( -55),
                        INT8_C(  90), INT8_C( -15), INT8_C( -11), INT8_C( -21)),
      easysimd_mm512_set_epi8(INT8_C( -76), INT8_C(  58), INT8_C(  70), INT8_C(-106),
                           INT8_C( 120), INT8_C(  72), INT8_C(  -4), INT8_C( -60),
                           INT8_C( 104), INT8_C( 101), INT8_C(  53), INT8_C(-121),
                           INT8_C( 102), INT8_C(-115), INT8_C(  90), INT8_C(  31),
                           INT8_C(  11), INT8_C(  68), INT8_C(  48), INT8_C(   4),
                           INT8_C(  55), INT8_C( -83), INT8_C(  75), INT8_C( -60),
                           INT8_C( -54), INT8_C(  57), INT8_C(  70), INT8_C(-114),
                           INT8_C(  51), INT8_C( -72), INT8_C( -71), INT8_C(  17),
                           INT8_C(  48), INT8_C(  40), INT8_C(-108), INT8_C( -21),
                           INT8_C( -21), INT8_C( -71), INT8_C( -21), INT8_C( -21),
                           INT8_C( -92), INT8_C( -21), INT8_C(  30), INT8_C( -21),
                           INT8_C( -21), INT8_C(-116), INT8_C( -21), INT8_C( -21),
                           INT8_C(  99), INT8_C( 108), INT8_C( -21), INT8_C(   7),
                           INT8_C( -21), INT8_C( -21), INT8_C( -21), INT8_C(  88),
                           INT8_C( -21), INT8_C(  40), INT8_C(  10), INT8_C(-105),
                           INT8_C( -21), INT8_C(-101), INT8_C( -21), INT8_C( -21)) },
    { easysimd_mm512_set_epi8(INT8_C( -65), INT8_C(  -9), INT8_C( -93), INT8_C(-113),
                           INT8_C( -10), INT8_C(  74), INT8_C(  39), INT8_C(  57),
                           INT8_C(  91), INT8_C( -48), INT8_C(  11), INT8_C( -15),
                           INT8_C(  21), INT8_C( -88), INT8_C(  91), INT8_C(  87),
                           INT8_C(-120), INT8_C(-105), INT8_C( -47), INT8_C(  85),
                           INT8_C( -98), INT8_C(  22), INT8_C(-124), INT8_C(-124),
                           INT8_C(   2), INT8_C(-104), INT8_C(  27), INT8_C(  96),
                           INT8_C( -89), INT8_C(  31), INT8_C(  20), INT8_C(  31),
                           INT8_C( -95), INT8_C(  13), INT8_C(  37), INT8_C(  31),
                           INT8_C( -72), INT8_C(  83), INT8_C(  94), INT8_C(  52),
                           INT8_C(  41), INT8_C(  25), INT8_C( -42), INT8_C(-109),
                           INT8_C(  31), INT8_C(  88), INT8_C( -71), INT8_C( -89),
                           INT8_C( 103), INT8_C( -85), INT8_C( -29), INT8_C(  86),
                           INT8_C(  71), INT8_C(  28), INT8_C( -23), INT8_C(  28),
                           INT8_C( -53), INT8_C( -82), INT8_C(  58), INT8_C( -12),
                           INT8_C(  63), INT8_C(  39), INT8_C( -32), INT8_C( -94)),
      UINT64_C(          1058428392),
      easysimd_mm_set_epi8(INT8_C(  85), INT8_C( -11), INT8_C( -21), INT8_C(  66),
                        INT8_C(  72), INT8_C(  -7), INT8_C( -18), INT8_C(-121),
                        INT8_C(  56), INT8_C(  51), INT8_C( 101), INT8_C(  91),
                        INT8_C( -85), INT8_C( -32), INT8_C( -40), INT8_C( -81)),
      easysimd_mm512_set_epi8(INT8_C( -65), INT8_C(  -9), INT8_C( -93), INT8_C(-113),
                           INT8_C( -10), INT8_C(  74), INT8_C(  39), INT8_C(  57),
                           INT8_C(  91), INT8_C( -48), INT8_C(  11), INT8_C( -15),
                           INT8_C(  21), INT8_C( -88), INT8_C(  91), INT8_C(  87),
                           INT8_C(-120), INT8_C(-105), INT8_C( -47), INT8_C(  85),
                           INT8_C( -98), INT8_C(  22), INT8_C(-124), INT8_C(-124),
                           INT8_C(   2), INT8_C(-104), INT8_C(  27), INT8_C(  96),
                           INT8_C( -89), INT8_C(  31), INT8_C(  20), INT8_C(  31),
                           INT8_C( -95), INT8_C(  13), INT8_C( -81), INT8_C( -81),
                           INT8_C( -81), INT8_C( -81), INT8_C( -81), INT8_C( -81),
                           INT8_C(  41), INT8_C(  25), INT8_C( -42), INT8_C( -81),
                           INT8_C(  31), INT8_C( -81), INT8_C( -81), INT8_C( -89),
                           INT8_C( 103), INT8_C( -81), INT8_C( -29), INT8_C( -81),
                           INT8_C(  71), INT8_C( -81), INT8_C( -23), INT8_C( -81),
                           INT8_C( -81), INT8_C( -81), INT8_C( -81), INT8_C( -12),
                           INT8_C( -81), INT8_C(  39), INT8_C( -32), INT8_C( -94)) },
    { easysimd_mm512_set_epi8(INT8_C(  85), INT8_C(  18), INT8_C(-117), INT8_C( -50),
                           INT8_C(  -8), INT8_C( 126), INT8_C( 103), INT8_C( -42),
                           INT8_C( 107), INT8_C( -60), INT8_C( -85), INT8_C( 123),
                           INT8_C( -11), INT8_C(  41), INT8_C(  98), INT8_C( 115),
                           INT8_C(  14), INT8_C(  34), INT8_C(  89), INT8_C( 101),
                           INT8_C(  39), INT8_C(  26), INT8_C( 121), INT8_C(  70),
                           INT8_C( -20), INT8_C( -34), INT8_C( -11), INT8_C(  72),
                           INT8_C(   8), INT8_C( -24), INT8_C(-104), INT8_C(  61),
                           INT8_C(-108), INT8_C( -43), INT8_C( 102), INT8_C( 100),
                           INT8_C( -29), INT8_C( -21), INT8_C(  70), INT8_C( -28),
                           INT8_C( -21), INT8_C( -82), INT8_C( -18), INT8_C(   9),
                           INT8_C(  94), INT8_C( -32), INT8_C(  97), INT8_C( -86),
                           INT8_C(  87), INT8_C(  62), INT8_C(-118), INT8_C(  17),
                           INT8_C(  18), INT8_C(-126), INT8_C(  74), INT8_C( -83),
                           INT8_C( -46), INT8_C(-103), INT8_C( -21), INT8_C( 108),
                           INT8_C( -58), INT8_C(-126), INT8_C( -28), INT8_C(-112)),
      UINT64_C(           923153287),
      easysimd_mm_set_epi8(INT8_C(  73), INT8_C( -73), INT8_C( -11), INT8_C(  36),
                        INT8_C( -17), INT8_C(  70), INT8_C(-102), INT8_C(-111),
                        INT8_C(  27), INT8_C( -97), INT8_C(  -6), INT8_C(  -7),
                        INT8_C(  28), INT8_C( -52), INT8_C( -54), INT8_C( -50)),
      easysimd_mm512_set_epi8(INT8_C(  85), INT8_C(  18), INT8_C(-117), INT8_C( -50),
                           INT8_C(  -8), INT8_C( 126), INT8_C( 103), INT8_C( -42),
                           INT8_C( 107), INT8_C( -60), INT8_C( -85), INT8_C( 123),
                           INT8_C( -11), INT8_C(  41), INT8_C(  98), INT8_C( 115),
                           INT8_C(  14), INT8_C(  34), INT8_C(  89), INT8_C( 101),
                           INT8_C(  39), INT8_C(  26), INT8_C( 121), INT8_C(  70),
                           INT8_C( -20), INT8_C( -34), INT8_C( -11), INT8_C(  72),
                           INT8_C(   8), INT8_C( -24), INT8_C(-104), INT8_C(  61),
                           INT8_C(-108), INT8_C( -43), INT8_C( -50), INT8_C( -50),
                           INT8_C( -29), INT8_C( -50), INT8_C( -50), INT8_C( -50),
                           INT8_C( -21), INT8_C( -82), INT8_C( -18), INT8_C(   9),
                           INT8_C(  94), INT8_C( -50), INT8_C( -50), INT8_C( -86),
                           INT8_C(  87), INT8_C(  62), INT8_C( -50), INT8_C( -50),
                           INT8_C(  18), INT8_C(-126), INT8_C( -50), INT8_C( -50),
                           INT8_C( -50), INT8_C(-103), INT8_C( -21), INT8_C( 108),
                           INT8_C( -58), INT8_C( -50), INT8_C( -50), INT8_C( -50)) },
    { easysimd_mm512_set_epi8(INT8_C(  67), INT8_C(-107), INT8_C(  82), INT8_C(  55),
                           INT8_C(  64), INT8_C(  72), INT8_C( -53), INT8_C(  66),
                           INT8_C( -50), INT8_C( 103), INT8_C( -13), INT8_C(  78),
                           INT8_C(  15), INT8_C(  32), INT8_C(  76), INT8_C(  78),
                           INT8_C(  28), INT8_C( -98), INT8_C(-128), INT8_C(  80),
                           INT8_C( 106), INT8_C( -45), INT8_C(  79), INT8_C( 116),
                           INT8_C(  23), INT8_C(  31), INT8_C( 117), INT8_C( -12),
                           INT8_C( -59), INT8_C( -16), INT8_C(  98), INT8_C( -49),
                           INT8_C( 116), INT8_C( -82), INT8_C(  92), INT8_C(   1),
                           INT8_C(  30), INT8_C(-100), INT8_C(  61), INT8_C( -14),
                           INT8_C(  26), INT8_C( -40), INT8_C( -78), INT8_C( -85),
                           INT8_C( -24), INT8_C( -47), INT8_C( -93), INT8_C(  -1),
                           INT8_C(  21), INT8_C(  82), INT8_C( 119), INT8_C(  64),
                           INT8_C(  74), INT8_C( -53), INT8_C(  58), INT8_C(  33),
                           INT8_C(  14), INT8_C( 114), INT8_C(  35), INT8_C( 109),
                           INT8_C( -74), INT8_C( -59), INT8_C( -81), INT8_C(  16)),
      UINT64_C(           594368556),
      easysimd_mm_set_epi8(INT8_C(  26), INT8_C( -78), INT8_C(  32), INT8_C(  10),
                        INT8_C(-126), INT8_C(  64), INT8_C(  35), INT8_C( -54),
                        INT8_C( -42), INT8_C( -70), INT8_C( 114), INT8_C( 111),
                        INT8_C( 111), INT8_C(  11), INT8_C( 104), INT8_C(  39)),
      easysimd_mm512_set_epi8(INT8_C(  67), INT8_C(-107), INT8_C(  82), INT8_C(  55),
                           INT8_C(  64), INT8_C(  72), INT8_C( -53), INT8_C(  66),
                           INT8_C( -50), INT8_C( 103), INT8_C( -13), INT8_C(  78),
                           INT8_C(  15), INT8_C(  32), INT8_C(  76), INT8_C(  78),
                           INT8_C(  28), INT8_C( -98), INT8_C(-128), INT8_C(  80),
                           INT8_C( 106), INT8_C( -45), INT8_C(  79), INT8_C( 116),
                           INT8_C(  23), INT8_C(  31), INT8_C( 117), INT8_C( -12),
                           INT8_C( -59), INT8_C( -16), INT8_C(  98), INT8_C( -49),
                           INT8_C( 116), INT8_C( -82), INT8_C(  39), INT8_C(   1),
                           INT8_C(  30), INT8_C(-100), INT8_C(  39), INT8_C(  39),
                           INT8_C(  26), INT8_C(  39), INT8_C(  39), INT8_C( -85),
                           INT8_C(  39), INT8_C(  39), INT8_C( -93), INT8_C(  39),
                           INT8_C(  21), INT8_C(  39), INT8_C( 119), INT8_C(  39),
                           INT8_C(  39), INT8_C( -53), INT8_C(  58), INT8_C(  33),
                           INT8_C(  14), INT8_C( 114), INT8_C(  39), INT8_C( 109),
                           INT8_C(  39), INT8_C(  39), INT8_C( -81), INT8_C(  16)) },
    { easysimd_mm512_set_epi8(INT8_C( 124), INT8_C(  71), INT8_C(-128), INT8_C( 110),
                           INT8_C(-123), INT8_C( -14), INT8_C( 123), INT8_C( -42),
                           INT8_C(  94), INT8_C(  60), INT8_C( 116), INT8_C( -89),
                           INT8_C(  73), INT8_C( -61), INT8_C(  -3), INT8_C(-114),
                           INT8_C( -92), INT8_C( -78), INT8_C(  90), INT8_C(  44),
                           INT8_C( -84), INT8_C( -33), INT8_C( 116), INT8_C(  -6),
                           INT8_C( -44), INT8_C( 126), INT8_C( -26), INT8_C(  80),
                           INT8_C( -91), INT8_C(-125), INT8_C(  72), INT8_C(  -8),
                           INT8_C( -16), INT8_C(  95), INT8_C( -25), INT8_C( -16),
                           INT8_C( -52), INT8_C( 116), INT8_C( -23), INT8_C(-102),
                           INT8_C( 119), INT8_C( -76), INT8_C(  48), INT8_C(  26),
                           INT8_C(-128), INT8_C(  43), INT8_C(  99), INT8_C( -34),
                           INT8_C(-103), INT8_C( -40), INT8_C(  47), INT8_C(-112),
                           INT8_C(-117), INT8_C( 111), INT8_C(-126), INT8_C(-115),
                           INT8_C(  65), INT8_C( -55), INT8_C(  49), INT8_C(  37),
                           INT8_C(-110), INT8_C(-124), INT8_C( 126), INT8_C(  -2)),
      UINT64_C(          1610616610),
      easysimd_mm_set_epi8(INT8_C( -95), INT8_C(  29), INT8_C( -58), INT8_C( -87),
                        INT8_C(  73), INT8_C(  12), INT8_C( -29), INT8_C(  41),
                        INT8_C( -96), INT8_C( 122), INT8_C( -95), INT8_C( -33),
                        INT8_C(-128), INT8_C(   2), INT8_C( 115), INT8_C( 108)),
      easysimd_mm512_set_epi8(INT8_C( 124), INT8_C(  71), INT8_C(-128), INT8_C( 110),
                           INT8_C(-123), INT8_C( -14), INT8_C( 123), INT8_C( -42),
                           INT8_C(  94), INT8_C(  60), INT8_C( 116), INT8_C( -89),
                           INT8_C(  73), INT8_C( -61), INT8_C(  -3), INT8_C(-114),
                           INT8_C( -92), INT8_C( -78), INT8_C(  90), INT8_C(  44),
                           INT8_C( -84), INT8_C( -33), INT8_C( 116), INT8_C(  -6),
                           INT8_C( -44), INT8_C( 126), INT8_C( -26), INT8_C(  80),
                           INT8_C( -91), INT8_C(-125), INT8_C(  72), INT8_C(  -8),
                           INT8_C( -16), INT8_C( 108), INT8_C( 108), INT8_C( -16),
                           INT8_C( -52), INT8_C( 116), INT8_C( -23), INT8_C(-102),
                           INT8_C( 119), INT8_C( -76), INT8_C(  48), INT8_C(  26),
                           INT8_C(-128), INT8_C(  43), INT8_C(  99), INT8_C( -34),
                           INT8_C(-103), INT8_C( -40), INT8_C(  47), INT8_C(-112),
                           INT8_C( 108), INT8_C( 108), INT8_C( 108), INT8_C( 108),
                           INT8_C(  65), INT8_C( -55), INT8_C( 108), INT8_C(  37),
                           INT8_C(-110), INT8_C(-124), INT8_C( 108), INT8_C(  -2)) },
    { easysimd_mm512_set_epi8(INT8_C(  73), INT8_C( -95), INT8_C( -44), INT8_C( 123),
                           INT8_C( -34), INT8_C(-122), INT8_C( 105), INT8_C( -63),
                           INT8_C( -13), INT8_C( -78), INT8_C(  -7), INT8_C(  88),
                           INT8_C(-101), INT8_C(  60), INT8_C(  29), INT8_C( -15),
                           INT8_C(  87), INT8_C( -77), INT8_C(  65), INT8_C(  71),
                           INT8_C( 113), INT8_C(-124), INT8_C( -41), INT8_C( -18),
                           INT8_C(  37), INT8_C( -20), INT8_C( 112), INT8_C(  70),
                           INT8_C(  36), INT8_C( -80), INT8_C( 122), INT8_C( -28),
                           INT8_C( -45), INT8_C(-113), INT8_C(  68), INT8_C(  23),
                           INT8_C(  84), INT8_C(  56), INT8_C( -44), INT8_C( -61),
                           INT8_C( -78), INT8_C(   6), INT8_C(-108), INT8_C(  73),
                           INT8_C( -22), INT8_C( -71), INT8_C(   1), INT8_C(   7),
                           INT8_C(  47), INT8_C(  18), INT8_C(-127), INT8_C( 127),
                           INT8_C( -16), INT8_C( -48), INT8_C( -39), INT8_C( 106),
                           INT8_C(  27), INT8_C(  40), INT8_C( -58), INT8_C( -56),
                           INT8_C( -27), INT8_C(  17), INT8_C(  29), INT8_C( -46)),
      UINT64_C(          2168160586),
      easysimd_mm_set_epi8(INT8_C(  45), INT8_C(  89), INT8_C( -40), INT8_C(  94),
                        INT8_C( -55), INT8_C( -34), INT8_C(-119), INT8_C(-109),
                        INT8_C(   3), INT8_C(-117), INT8_C(-101), INT8_C(  63),
                        INT8_C( 122), INT8_C(  -4), INT8_C(-100), INT8_C( -84)),
      easysimd_mm512_set_epi8(INT8_C(  73), INT8_C( -95), INT8_C( -44), INT8_C( 123),
                           INT8_C( -34), INT8_C(-122), INT8_C( 105), INT8_C( -63),
                           INT8_C( -13), INT8_C( -78), INT8_C(  -7), INT8_C(  88),
                           INT8_C(-101), INT8_C(  60), INT8_C(  29), INT8_C( -15),
                           INT8_C(  87), INT8_C( -77), INT8_C(  65), INT8_C(  71),
                           INT8_C( 113), INT8_C(-124), INT8_C( -41), INT8_C( -18),
                           INT8_C(  37), INT8_C( -20), INT8_C( 112), INT8_C(  70),
                           INT8_C(  36), INT8_C( -80), INT8_C( 122), INT8_C( -28),
                           INT8_C( -84), INT8_C(-113), INT8_C(  68), INT8_C(  23),
                           INT8_C(  84), INT8_C(  56), INT8_C( -44), INT8_C( -84),
                           INT8_C( -78), INT8_C(   6), INT8_C( -84), INT8_C( -84),
                           INT8_C( -84), INT8_C( -71), INT8_C( -84), INT8_C( -84),
                           INT8_C( -84), INT8_C(  18), INT8_C(-127), INT8_C( 127),
                           INT8_C( -16), INT8_C( -48), INT8_C( -39), INT8_C( -84),
                           INT8_C(  27), INT8_C( -84), INT8_C( -58), INT8_C( -56),
                           INT8_C( -84), INT8_C(  17), INT8_C( -84), INT8_C( -46)) },
    { easysimd_mm512_set_epi8(INT8_C(  38), INT8_C( -12), INT8_C( -37), INT8_C(  58),
                           INT8_C(  89), INT8_C(-127), INT8_C( -11), INT8_C(  26),
                           INT8_C( -29), INT8_C(-122), INT8_C(  86), INT8_C(  69),
                           INT8_C(  63), INT8_C(  74), INT8_C(  90), INT8_C(  88),
                           INT8_C( -75), INT8_C( -43), INT8_C(  36), INT8_C(  61),
                           INT8_C( -19), INT8_C(  27), INT8_C(-123), INT8_C(  78),
                           INT8_C(  67), INT8_C(  58), INT8_C( -32), INT8_C(  42),
                           INT8_C(  25), INT8_C( -26), INT8_C( 122), INT8_C(-100),
                           INT8_C(-107), INT8_C( -53), INT8_C(-114), INT8_C(  63),
                           INT8_C(-100), INT8_C(  53), INT8_C( -32), INT8_C( -39),
                           INT8_C( -75), INT8_C(-119), INT8_C( -67), INT8_C(  96),
                           INT8_C(  -6), INT8_C( -22), INT8_C( -12), INT8_C(  19),
                           INT8_C( -51), INT8_C(  42), INT8_C(  39), INT8_C(-124),
                           INT8_C(  38), INT8_C( -95), INT8_C(-119), INT8_C(  -9),
                           INT8_C(  94), INT8_C( -51), INT8_C(   1), INT8_C( -64),
                           INT8_C( -67), INT8_C(-127), INT8_C( -33), INT8_C(  75)),
      UINT64_C(          3579095368),
      easysimd_mm_set_epi8(INT8_C( -71), INT8_C(-112), INT8_C(-122), INT8_C( -13),
                        INT8_C(-109), INT8_C(  21), INT8_C(  27), INT8_C(-109),
                        INT8_C(  55), INT8_C(   9), INT8_C( 117), INT8_C( -28),
                        INT8_C( -58), INT8_C(  -1), INT8_C(   3), INT8_C( -34)),
      easysimd_mm512_set_epi8(INT8_C(  38), INT8_C( -12), INT8_C( -37), INT8_C(  58),
                           INT8_C(  89), INT8_C(-127), INT8_C( -11), INT8_C(  26),
                           INT8_C( -29), INT8_C(-122), INT8_C(  86), INT8_C(  69),
                           INT8_C(  63), INT8_C(  74), INT8_C(  90), INT8_C(  88),
                           INT8_C( -75), INT8_C( -43), INT8_C(  36), INT8_C(  61),
                           INT8_C( -19), INT8_C(  27), INT8_C(-123), INT8_C(  78),
                           INT8_C(  67), INT8_C(  58), INT8_C( -32), INT8_C(  42),
                           INT8_C(  25), INT8_C( -26), INT8_C( 122), INT8_C(-100),
                           INT8_C( -34), INT8_C( -34), INT8_C(-114), INT8_C( -34),
                           INT8_C(-100), INT8_C( -34), INT8_C( -32), INT8_C( -34),
                           INT8_C( -75), INT8_C( -34), INT8_C( -67), INT8_C( -34),
                           INT8_C(  -6), INT8_C( -34), INT8_C( -12), INT8_C(  19),
                           INT8_C( -34), INT8_C(  42), INT8_C( -34), INT8_C(-124),
                           INT8_C( -34), INT8_C( -95), INT8_C(-119), INT8_C( -34),
                           INT8_C(  94), INT8_C( -34), INT8_C(   1), INT8_C( -64),
                           INT8_C( -34), INT8_C(-127), INT8_C( -33), INT8_C(  75)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_mask_broadcastb_epi8(test_vec[i].src, test_vec[i].k, test_vec[i].a);
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_broadcastb_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask64 k;
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT64_C(          2081702095),
      easysimd_mm_set_epi8(INT8_C( 126), INT8_C(  -6), INT8_C(  16), INT8_C( 102),
                        INT8_C( -47), INT8_C(-116), INT8_C(  -4), INT8_C(  33),
                        INT8_C( -25), INT8_C(-108), INT8_C(-115), INT8_C(-104),
                        INT8_C( -39), INT8_C(  49), INT8_C(  72), INT8_C(  44)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  44), INT8_C(  44), INT8_C(  44),
                           INT8_C(  44), INT8_C(  44), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  44),
                           INT8_C(   0), INT8_C(  44), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  44), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  44), INT8_C(  44), INT8_C(   0), INT8_C(   0),
                           INT8_C(  44), INT8_C(  44), INT8_C(  44), INT8_C(  44)) },
    { UINT64_C(          4229458596),
      easysimd_mm_set_epi8(INT8_C(-106), INT8_C(-123), INT8_C( 120), INT8_C(  43),
                        INT8_C( -31), INT8_C(   4), INT8_C(  10), INT8_C(  96),
                        INT8_C( -40), INT8_C(  23), INT8_C(  31), INT8_C(  73),
                        INT8_C( -51), INT8_C(  91), INT8_C(  68), INT8_C( -23)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -23), INT8_C( -23), INT8_C( -23), INT8_C( -23),
                           INT8_C( -23), INT8_C( -23), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -23),
                           INT8_C( -23), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -23), INT8_C( -23), INT8_C(   0),
                           INT8_C( -23), INT8_C(   0), INT8_C( -23), INT8_C(   0),
                           INT8_C( -23), INT8_C(   0), INT8_C( -23), INT8_C(   0),
                           INT8_C(   0), INT8_C( -23), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C(          3399230491),
      easysimd_mm_set_epi8(INT8_C( -40), INT8_C( -29), INT8_C(  78), INT8_C(  94),
                        INT8_C( -79), INT8_C(  10), INT8_C(-103), INT8_C(-109),
                        INT8_C(  65), INT8_C( -68), INT8_C( 102), INT8_C(-122),
                        INT8_C(  40), INT8_C(  19), INT8_C(-111), INT8_C(   8)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   8), INT8_C(   8), INT8_C(   0), INT8_C(   0),
                           INT8_C(   8), INT8_C(   0), INT8_C(   8), INT8_C(   0),
                           INT8_C(   8), INT8_C(   0), INT8_C(   0), INT8_C(   8),
                           INT8_C(   8), INT8_C(   8), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   8), INT8_C(   0),
                           INT8_C(   0), INT8_C(   8), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   8),
                           INT8_C(   8), INT8_C(   0), INT8_C(   8), INT8_C(   8)) },
    { UINT64_C(          2871199873),
      easysimd_mm_set_epi8(INT8_C(  27), INT8_C(  40), INT8_C(  10), INT8_C(-105),
                        INT8_C(  76), INT8_C(-101), INT8_C(  87), INT8_C( 112),
                        INT8_C( -78), INT8_C( -60), INT8_C(-122), INT8_C( -66),
                        INT8_C(   8), INT8_C( -42), INT8_C(  44), INT8_C(  45)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  45), INT8_C(   0), INT8_C(  45), INT8_C(   0),
                           INT8_C(  45), INT8_C(   0), INT8_C(  45), INT8_C(  45),
                           INT8_C(   0), INT8_C(   0), INT8_C(  45), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  45), INT8_C(  45),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  45), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  45), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  45)) },
    { UINT64_C(          1884717912),
      easysimd_mm_set_epi8(INT8_C(  48), INT8_C(  40), INT8_C(-108), INT8_C( -12),
                        INT8_C( -11), INT8_C( -71), INT8_C(-114), INT8_C( -36),
                        INT8_C( -92), INT8_C( 101), INT8_C(  30), INT8_C(  10),
                        INT8_C(  43), INT8_C(-116), INT8_C( -45), INT8_C(-104)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(-104), INT8_C(-104), INT8_C(-104),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(-104), INT8_C(   0), INT8_C(-104),
                           INT8_C(   0), INT8_C(-104), INT8_C(-104), INT8_C(   0),
                           INT8_C(-104), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(-104), INT8_C(-104),
                           INT8_C(   0), INT8_C(-104), INT8_C(   0), INT8_C(-104),
                           INT8_C(-104), INT8_C(   0), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C(           867744017),
      easysimd_mm_set_epi8(INT8_C( 104), INT8_C( 101), INT8_C(  53), INT8_C(-121),
                        INT8_C( 102), INT8_C(-115), INT8_C(  90), INT8_C(  31),
                        INT8_C(  11), INT8_C(  68), INT8_C(  48), INT8_C(   4),
                        INT8_C(  55), INT8_C( -83), INT8_C(  75), INT8_C( -60)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -60), INT8_C( -60),
                           INT8_C(   0), INT8_C(   0), INT8_C( -60), INT8_C( -60),
                           INT8_C( -60), INT8_C(   0), INT8_C( -60), INT8_C( -60),
                           INT8_C( -60), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -60), INT8_C(   0), INT8_C( -60), INT8_C( -60),
                           INT8_C( -60), INT8_C(   0), INT8_C(   0), INT8_C( -60),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -60),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -60)) },
    { UINT64_C(          2018049220),
      easysimd_mm_set_epi8(INT8_C(  52), INT8_C( -63), INT8_C( -88), INT8_C( -55),
                        INT8_C(  90), INT8_C( -15), INT8_C( -11), INT8_C( -21),
                        INT8_C( 100), INT8_C( -84), INT8_C( -92), INT8_C( -78),
                        INT8_C(  27), INT8_C(  91), INT8_C(  46), INT8_C(-117)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(-117), INT8_C(-117), INT8_C(-117),
                           INT8_C(-117), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(-117), INT8_C(   0), INT8_C(   0),
                           INT8_C(-117), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(-117), INT8_C(-117), INT8_C(-117), INT8_C(-117),
                           INT8_C(-117), INT8_C(-117), INT8_C(   0), INT8_C(   0),
                           INT8_C(-117), INT8_C(-117), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(-117), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C(          1858505628),
      easysimd_mm_set_epi8(INT8_C( 103), INT8_C( -85), INT8_C( -29), INT8_C(  86),
                        INT8_C(  71), INT8_C(  28), INT8_C( -23), INT8_C(  28),
                        INT8_C( -53), INT8_C( -82), INT8_C(  58), INT8_C( -12),
                        INT8_C(  63), INT8_C(  39), INT8_C( -32), INT8_C( -94)),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -94), INT8_C( -94), INT8_C(   0),
                           INT8_C( -94), INT8_C( -94), INT8_C( -94), INT8_C(   0),
                           INT8_C( -94), INT8_C( -94), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -94), INT8_C( -94), INT8_C(   0),
                           INT8_C( -94), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -94), INT8_C(   0), INT8_C( -94), INT8_C( -94),
                           INT8_C( -94), INT8_C(   0), INT8_C(   0), INT8_C( -94),
                           INT8_C( -94), INT8_C( -94), INT8_C(   0), INT8_C(   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_maskz_broadcastb_epi8(test_vec[i].k, test_vec[i].a);
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_broadcastw_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm_set_epi16(INT16_C( -4264), INT16_C(-31095), INT16_C( 28503), INT16_C(-19429),
                         INT16_C(-23560), INT16_C( -4328), INT16_C( 17780), INT16_C(-19836)),
      easysimd_mm512_set_epi16(INT16_C(-19836), INT16_C(-19836), INT16_C(-19836), INT16_C(-19836),
                            INT16_C(-19836), INT16_C(-19836), INT16_C(-19836), INT16_C(-19836),
                            INT16_C(-19836), INT16_C(-19836), INT16_C(-19836), INT16_C(-19836),
                            INT16_C(-19836), INT16_C(-19836), INT16_C(-19836), INT16_C(-19836),
                            INT16_C(-19836), INT16_C(-19836), INT16_C(-19836), INT16_C(-19836),
                            INT16_C(-19836), INT16_C(-19836), INT16_C(-19836), INT16_C(-19836),
                            INT16_C(-19836), INT16_C(-19836), INT16_C(-19836), INT16_C(-19836),
                            INT16_C(-19836), INT16_C(-19836), INT16_C(-19836), INT16_C(-19836)) },
    { easysimd_mm_set_epi16(INT16_C( -1138), INT16_C(-21762), INT16_C(  8538), INT16_C(-12772),
                         INT16_C(  3852), INT16_C(  8246), INT16_C( -3641), INT16_C(  9422)),
      easysimd_mm512_set_epi16(INT16_C(  9422), INT16_C(  9422), INT16_C(  9422), INT16_C(  9422),
                            INT16_C(  9422), INT16_C(  9422), INT16_C(  9422), INT16_C(  9422),
                            INT16_C(  9422), INT16_C(  9422), INT16_C(  9422), INT16_C(  9422),
                            INT16_C(  9422), INT16_C(  9422), INT16_C(  9422), INT16_C(  9422),
                            INT16_C(  9422), INT16_C(  9422), INT16_C(  9422), INT16_C(  9422),
                            INT16_C(  9422), INT16_C(  9422), INT16_C(  9422), INT16_C(  9422),
                            INT16_C(  9422), INT16_C(  9422), INT16_C(  9422), INT16_C(  9422),
                            INT16_C(  9422), INT16_C(  9422), INT16_C(  9422), INT16_C(  9422)) },
    { easysimd_mm_set_epi16(INT16_C(-12364), INT16_C(-15754), INT16_C(  -793), INT16_C(-14722),
                         INT16_C(-29314), INT16_C(-26497), INT16_C(  3881), INT16_C( 17439)),
      easysimd_mm512_set_epi16(INT16_C( 17439), INT16_C( 17439), INT16_C( 17439), INT16_C( 17439),
                            INT16_C( 17439), INT16_C( 17439), INT16_C( 17439), INT16_C( 17439),
                            INT16_C( 17439), INT16_C( 17439), INT16_C( 17439), INT16_C( 17439),
                            INT16_C( 17439), INT16_C( 17439), INT16_C( 17439), INT16_C( 17439),
                            INT16_C( 17439), INT16_C( 17439), INT16_C( 17439), INT16_C( 17439),
                            INT16_C( 17439), INT16_C( 17439), INT16_C( 17439), INT16_C( 17439),
                            INT16_C( 17439), INT16_C( 17439), INT16_C( 17439), INT16_C( 17439),
                            INT16_C( 17439), INT16_C( 17439), INT16_C( 17439), INT16_C( 17439)) },
    { easysimd_mm_set_epi16(INT16_C( -7447), INT16_C(-10523), INT16_C(-25861), INT16_C(-22174),
                         INT16_C(  8521), INT16_C( 32120), INT16_C(-17861), INT16_C( 31790)),
      easysimd_mm512_set_epi16(INT16_C( 31790), INT16_C( 31790), INT16_C( 31790), INT16_C( 31790),
                            INT16_C( 31790), INT16_C( 31790), INT16_C( 31790), INT16_C( 31790),
                            INT16_C( 31790), INT16_C( 31790), INT16_C( 31790), INT16_C( 31790),
                            INT16_C( 31790), INT16_C( 31790), INT16_C( 31790), INT16_C( 31790),
                            INT16_C( 31790), INT16_C( 31790), INT16_C( 31790), INT16_C( 31790),
                            INT16_C( 31790), INT16_C( 31790), INT16_C( 31790), INT16_C( 31790),
                            INT16_C( 31790), INT16_C( 31790), INT16_C( 31790), INT16_C( 31790),
                            INT16_C( 31790), INT16_C( 31790), INT16_C( 31790), INT16_C( 31790)) },
    { easysimd_mm_set_epi16(INT16_C( -4580), INT16_C( -4681), INT16_C( -4797), INT16_C( 20435),
                         INT16_C(-31664), INT16_C(-25722), INT16_C(-13794), INT16_C( -4041)),
      easysimd_mm512_set_epi16(INT16_C( -4041), INT16_C( -4041), INT16_C( -4041), INT16_C( -4041),
                            INT16_C( -4041), INT16_C( -4041), INT16_C( -4041), INT16_C( -4041),
                            INT16_C( -4041), INT16_C( -4041), INT16_C( -4041), INT16_C( -4041),
                            INT16_C( -4041), INT16_C( -4041), INT16_C( -4041), INT16_C( -4041),
                            INT16_C( -4041), INT16_C( -4041), INT16_C( -4041), INT16_C( -4041),
                            INT16_C( -4041), INT16_C( -4041), INT16_C( -4041), INT16_C( -4041),
                            INT16_C( -4041), INT16_C( -4041), INT16_C( -4041), INT16_C( -4041),
                            INT16_C( -4041), INT16_C( -4041), INT16_C( -4041), INT16_C( -4041)) },
    { easysimd_mm_set_epi16(INT16_C(  1787), INT16_C(  9631), INT16_C(  4347), INT16_C( -4594),
                         INT16_C(-30523), INT16_C(-10849), INT16_C(-17993), INT16_C(-18482)),
      easysimd_mm512_set_epi16(INT16_C(-18482), INT16_C(-18482), INT16_C(-18482), INT16_C(-18482),
                            INT16_C(-18482), INT16_C(-18482), INT16_C(-18482), INT16_C(-18482),
                            INT16_C(-18482), INT16_C(-18482), INT16_C(-18482), INT16_C(-18482),
                            INT16_C(-18482), INT16_C(-18482), INT16_C(-18482), INT16_C(-18482),
                            INT16_C(-18482), INT16_C(-18482), INT16_C(-18482), INT16_C(-18482),
                            INT16_C(-18482), INT16_C(-18482), INT16_C(-18482), INT16_C(-18482),
                            INT16_C(-18482), INT16_C(-18482), INT16_C(-18482), INT16_C(-18482),
                            INT16_C(-18482), INT16_C(-18482), INT16_C(-18482), INT16_C(-18482)) },
    { easysimd_mm_set_epi16(INT16_C( 30524), INT16_C( 16358), INT16_C( 12856), INT16_C( 10489),
                         INT16_C( 17653), INT16_C( -5197), INT16_C( 14483), INT16_C(-30060)),
      easysimd_mm512_set_epi16(INT16_C(-30060), INT16_C(-30060), INT16_C(-30060), INT16_C(-30060),
                            INT16_C(-30060), INT16_C(-30060), INT16_C(-30060), INT16_C(-30060),
                            INT16_C(-30060), INT16_C(-30060), INT16_C(-30060), INT16_C(-30060),
                            INT16_C(-30060), INT16_C(-30060), INT16_C(-30060), INT16_C(-30060),
                            INT16_C(-30060), INT16_C(-30060), INT16_C(-30060), INT16_C(-30060),
                            INT16_C(-30060), INT16_C(-30060), INT16_C(-30060), INT16_C(-30060),
                            INT16_C(-30060), INT16_C(-30060), INT16_C(-30060), INT16_C(-30060),
                            INT16_C(-30060), INT16_C(-30060), INT16_C(-30060), INT16_C(-30060)) },
    { easysimd_mm_set_epi16(INT16_C(-28607), INT16_C(  6822), INT16_C(-19640), INT16_C(   516),
                         INT16_C(-13138), INT16_C( -4418), INT16_C(-29962), INT16_C( 13528)),
      easysimd_mm512_set_epi16(INT16_C( 13528), INT16_C( 13528), INT16_C( 13528), INT16_C( 13528),
                            INT16_C( 13528), INT16_C( 13528), INT16_C( 13528), INT16_C( 13528),
                            INT16_C( 13528), INT16_C( 13528), INT16_C( 13528), INT16_C( 13528),
                            INT16_C( 13528), INT16_C( 13528), INT16_C( 13528), INT16_C( 13528),
                            INT16_C( 13528), INT16_C( 13528), INT16_C( 13528), INT16_C( 13528),
                            INT16_C( 13528), INT16_C( 13528), INT16_C( 13528), INT16_C( 13528),
                            INT16_C( 13528), INT16_C( 13528), INT16_C( 13528), INT16_C( 13528),
                            INT16_C( 13528), INT16_C( 13528), INT16_C( 13528), INT16_C( 13528)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i r = easysimd_mm512_broadcastw_epi16(test_vec[i].a);
    easysimd_assert_m512i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_broadcastmw_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_broadcastmb_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_broadcastb_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_broadcastb_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_broadcastw_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_broadcastw_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_broadcastd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_broadcastd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_broadcastq_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_broadcastq_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_broadcastss_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_broadcast_i32x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_broadcast_i32x2)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_broadcast_i32x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_broadcast_i32x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_broadcast_i32x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_broadcast_f32x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_broadcast_f32x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_broadcast_f32x2)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcast_f32x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcast_f32x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcast_f32x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcast_f32x8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcast_f32x8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcast_f32x8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcast_f64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcast_f64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcast_f64x2)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_broadcast_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_broadcast_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_broadcast_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_broadcast_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_broadcast_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_broadcast_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_broadcast_i64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_broadcast_i64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_broadcast_i64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_broadcast_f64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_broadcast_f64x2)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_broadcast_f64x2)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcast_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcast_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcast_f32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcast_f64x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcast_f64x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcast_f64x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcast_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcast_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcast_i32x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcast_i64x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcast_i64x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcast_i64x4)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcastd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcastd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcastd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcastq_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcastq_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcastq_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcastss_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcastss_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcastss_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcastsd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcastsd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcastsd_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcastb_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_broadcastb_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_broadcastb_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_broadcastw_epi16)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
