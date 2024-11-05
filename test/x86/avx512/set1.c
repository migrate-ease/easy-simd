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

#define EASYSIMD_TEST_X86_AVX512_INSN set1

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set1.h>

static int
test_easysimd_mm_mask_set1_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t src[16];
    easysimd__mmask16 k;
    int8_t a;
    int8_t r[16];
  } test_vec[] = {
    { {  INT8_C(  67),  INT8_C( 110),  INT8_C(  19), -INT8_C( 102),  INT8_C(  59),  INT8_C( 126), -INT8_C(  57),  INT8_C( 108),
        -INT8_C(  10),  INT8_C(  50),  INT8_C(   7),  INT8_C(  76), -INT8_C(  61),  INT8_C(  30),  INT8_C(  49),  INT8_C(  32) },
      UINT16_C(46725),
      -INT8_C(   2),
      { -INT8_C(   2),  INT8_C( 110), -INT8_C(   2), -INT8_C( 102),  INT8_C(  59),  INT8_C( 126), -INT8_C(  57), -INT8_C(   2),
        -INT8_C(  10), -INT8_C(   2), -INT8_C(   2),  INT8_C(  76), -INT8_C(   2), -INT8_C(   2),  INT8_C(  49), -INT8_C(   2) } },
    { {  INT8_C(  66), -INT8_C(  93), -INT8_C(  54), -INT8_C(  71),  INT8_C(  19), -INT8_C(  63), -INT8_C(  42),  INT8_C(  82),
        -INT8_C(  57), -INT8_C( 122),  INT8_C(  11),  INT8_C(  56), -INT8_C(  55),  INT8_C( 122),  INT8_C(  75),  INT8_C( 100) },
      UINT16_C(51893),
       INT8_C(  43),
      {  INT8_C(  43), -INT8_C(  93),  INT8_C(  43), -INT8_C(  71),  INT8_C(  43),  INT8_C(  43), -INT8_C(  42),  INT8_C(  43),
        -INT8_C(  57),  INT8_C(  43),  INT8_C(  11),  INT8_C(  43), -INT8_C(  55),  INT8_C( 122),  INT8_C(  43),  INT8_C(  43) } },
    { {  INT8_C(  33), -INT8_C(  64),  INT8_C(  93),  INT8_C(  40),  INT8_C(  13),  INT8_C(  33),  INT8_C(  70),  INT8_C(  62),
         INT8_C(  65), -INT8_C(  52), -INT8_C(  12),  INT8_C(  64),  INT8_C(  14), -INT8_C( 105),  INT8_C(  10), -INT8_C(  57) },
      UINT16_C(52138),
      -INT8_C(  99),
      {  INT8_C(  33), -INT8_C(  99),  INT8_C(  93), -INT8_C(  99),  INT8_C(  13), -INT8_C(  99),  INT8_C(  70), -INT8_C(  99),
        -INT8_C(  99), -INT8_C(  99), -INT8_C(  12), -INT8_C(  99),  INT8_C(  14), -INT8_C( 105), -INT8_C(  99), -INT8_C(  99) } },
    { { -INT8_C(   4), -INT8_C( 109),  INT8_C(  35),  INT8_C(   8), -INT8_C(  53), -INT8_C(  19), -INT8_C( 126),  INT8_C(  22),
         INT8_C(  81),  INT8_C(  55), -INT8_C(  32),  INT8_C( 124),  INT8_C(  88), -INT8_C(  95), -INT8_C(  39),      INT8_MIN },
      UINT16_C(64174),
      -INT8_C(  58),
      { -INT8_C(   4), -INT8_C(  58), -INT8_C(  58), -INT8_C(  58), -INT8_C(  53), -INT8_C(  58), -INT8_C( 126), -INT8_C(  58),
         INT8_C(  81), -INT8_C(  58), -INT8_C(  32), -INT8_C(  58), -INT8_C(  58), -INT8_C(  58), -INT8_C(  58), -INT8_C(  58) } },
    { { -INT8_C(  20),  INT8_C(  60), -INT8_C( 110), -INT8_C(  32),  INT8_C( 124), -INT8_C(  96),  INT8_C( 119), -INT8_C( 122),
         INT8_C( 103),  INT8_C(  33),  INT8_C(  82),  INT8_C(   5),  INT8_C(  29), -INT8_C(  27),  INT8_C(  40),  INT8_C(  37) },
      UINT16_C( 5552),
      -INT8_C(  89),
      { -INT8_C(  20),  INT8_C(  60), -INT8_C( 110), -INT8_C(  32), -INT8_C(  89), -INT8_C(  89),  INT8_C( 119), -INT8_C(  89),
        -INT8_C(  89),  INT8_C(  33), -INT8_C(  89),  INT8_C(   5), -INT8_C(  89), -INT8_C(  27),  INT8_C(  40),  INT8_C(  37) } },
    { { -INT8_C(  58),  INT8_C( 102), -INT8_C(  34), -INT8_C(  89), -INT8_C(  30),  INT8_C(  54),  INT8_C(  72), -INT8_C(  68),
        -INT8_C(  74), -INT8_C(  10), -INT8_C(  74),  INT8_C( 125), -INT8_C(  30), -INT8_C(  14),  INT8_C(  15), -INT8_C(  62) },
      UINT16_C(45166),
       INT8_C(  57),
      { -INT8_C(  58),  INT8_C(  57),  INT8_C(  57),  INT8_C(  57), -INT8_C(  30),  INT8_C(  57),  INT8_C(  57), -INT8_C(  68),
        -INT8_C(  74), -INT8_C(  10), -INT8_C(  74),  INT8_C( 125),  INT8_C(  57),  INT8_C(  57),  INT8_C(  15),  INT8_C(  57) } },
    { { -INT8_C(  11),  INT8_C(  23),  INT8_C(  90),  INT8_C(  71),  INT8_C(  28),  INT8_C( 119),  INT8_C(  44),  INT8_C(  69),
        -INT8_C(  99), -INT8_C(  36),  INT8_C(  90),  INT8_C(  68), -INT8_C(  94), -INT8_C(  63),  INT8_C(  35),  INT8_C(  73) },
      UINT16_C(22947),
      -INT8_C( 111),
      { -INT8_C( 111), -INT8_C( 111),  INT8_C(  90),  INT8_C(  71),  INT8_C(  28), -INT8_C( 111),  INT8_C(  44), -INT8_C( 111),
        -INT8_C( 111), -INT8_C(  36),  INT8_C(  90), -INT8_C( 111), -INT8_C( 111), -INT8_C(  63), -INT8_C( 111),  INT8_C(  73) } },
    { {  INT8_C(  95),  INT8_C(  16), -INT8_C( 121),  INT8_C(  22), -INT8_C( 115),  INT8_C( 105),  INT8_C(   8), -INT8_C( 100),
         INT8_C(  43),  INT8_C( 119),  INT8_C(  76),  INT8_C( 100),  INT8_C( 108),  INT8_C( 100), -INT8_C(  66), -INT8_C(  77) },
      UINT16_C(13952),
      -INT8_C(  33),
      {  INT8_C(  95),  INT8_C(  16), -INT8_C( 121),  INT8_C(  22), -INT8_C( 115),  INT8_C( 105),  INT8_C(   8), -INT8_C(  33),
         INT8_C(  43), -INT8_C(  33), -INT8_C(  33),  INT8_C( 100), -INT8_C(  33), -INT8_C(  33), -INT8_C(  66), -INT8_C(  77) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi8(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    int8_t a = test_vec[i].a;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_set1_epi8(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_set1_epi8");
    easysimd_assert_m128i_i8(r, ==, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    int8_t a = easysimd_test_codegen_random_i8();
    easysimd__m128i r = easysimd_mm_mask_set1_epi8(src, k, a);

    easysimd_test_x86_write_i8x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_set1_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    int8_t a;
    int8_t r[16];
  } test_vec[] = {
    { UINT16_C( 1710),
      -INT8_C( 125),
      {  INT8_C(   0), -INT8_C( 125), -INT8_C( 125), -INT8_C( 125),  INT8_C(   0), -INT8_C( 125),  INT8_C(   0), -INT8_C( 125),
         INT8_C(   0), -INT8_C( 125), -INT8_C( 125),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C( 4685),
       INT8_C(  46),
      {  INT8_C(  46),  INT8_C(   0),  INT8_C(  46),  INT8_C(  46),  INT8_C(   0),  INT8_C(   0),  INT8_C(  46),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  46),  INT8_C(   0),  INT8_C(   0),  INT8_C(  46),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C( 7424),
       INT8_C(  98),
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  98),  INT8_C(   0),  INT8_C(  98),  INT8_C(  98),  INT8_C(  98),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(55590),
       INT8_C(  31),
      {  INT8_C(   0),  INT8_C(  31),  INT8_C(  31),  INT8_C(   0),  INT8_C(   0),  INT8_C(  31),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  31),  INT8_C(   0),  INT8_C(   0),  INT8_C(  31),  INT8_C(  31),  INT8_C(   0),  INT8_C(  31),  INT8_C(  31) } },
    { UINT16_C(48235),
       INT8_C(  20),
      {  INT8_C(  20),  INT8_C(  20),  INT8_C(   0),  INT8_C(  20),  INT8_C(   0),  INT8_C(  20),  INT8_C(  20),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  20),  INT8_C(  20),  INT8_C(  20),  INT8_C(  20),  INT8_C(   0),  INT8_C(  20) } },
    { UINT16_C(15519),
       INT8_C(  99),
      {  INT8_C(  99),  INT8_C(  99),  INT8_C(  99),  INT8_C(  99),  INT8_C(  99),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  99),  INT8_C(  99),  INT8_C(  99),  INT8_C(  99),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(24253),
      -INT8_C(  69),
      { -INT8_C(  69),  INT8_C(   0), -INT8_C(  69), -INT8_C(  69), -INT8_C(  69), -INT8_C(  69),  INT8_C(   0), -INT8_C(  69),
         INT8_C(   0), -INT8_C(  69), -INT8_C(  69), -INT8_C(  69), -INT8_C(  69),  INT8_C(   0), -INT8_C(  69),  INT8_C(   0) } },
    { UINT16_C(19664),
      -INT8_C(  35),
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  35),  INT8_C(   0), -INT8_C(  35), -INT8_C(  35),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  35), -INT8_C(  35),  INT8_C(   0),  INT8_C(   0), -INT8_C(  35),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    int8_t a = test_vec[i].a;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_set1_epi8(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_set1_epi8");
    easysimd_assert_m128i_i8(r, ==, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    int8_t a = easysimd_test_codegen_random_i8();
    easysimd__m128i r = easysimd_mm_maskz_set1_epi8(k, a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_set1_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t src[8];
    easysimd__mmask8 k;
    int16_t a;
    int16_t r[8];
  } test_vec[] = {
    { { -INT16_C( 11323),  INT16_C(  8379),  INT16_C( 23831),  INT16_C( 15073), -INT16_C( 31577),  INT16_C( 14484), -INT16_C( 23324), -INT16_C(  1344) },
      UINT8_C( 49),
       INT16_C(   553),
      {  INT16_C(   553),  INT16_C(  8379),  INT16_C( 23831),  INT16_C( 15073),  INT16_C(   553),  INT16_C(   553), -INT16_C( 23324), -INT16_C(  1344) } },
    { {  INT16_C( 21965),  INT16_C(  6777), -INT16_C(  6727),  INT16_C( 30846), -INT16_C(   360),  INT16_C( 30638), -INT16_C( 32316), -INT16_C(  7118) },
      UINT8_C(152),
      -INT16_C( 14960),
      {  INT16_C( 21965),  INT16_C(  6777), -INT16_C(  6727), -INT16_C( 14960), -INT16_C( 14960),  INT16_C( 30638), -INT16_C( 32316), -INT16_C( 14960) } },
    { {  INT16_C( 14291),  INT16_C( 26441),  INT16_C( 11631),  INT16_C( 12043),  INT16_C( 15399),  INT16_C( 10841), -INT16_C( 20983),  INT16_C(  9123) },
      UINT8_C(103),
      -INT16_C( 24183),
      { -INT16_C( 24183), -INT16_C( 24183), -INT16_C( 24183),  INT16_C( 12043),  INT16_C( 15399), -INT16_C( 24183), -INT16_C( 24183),  INT16_C(  9123) } },
    { {  INT16_C(  8671), -INT16_C( 29280),  INT16_C( 25753), -INT16_C( 13554), -INT16_C( 22712),  INT16_C(  3419), -INT16_C( 28038), -INT16_C(  7850) },
      UINT8_C(  2),
      -INT16_C(  4988),
      {  INT16_C(  8671), -INT16_C(  4988),  INT16_C( 25753), -INT16_C( 13554), -INT16_C( 22712),  INT16_C(  3419), -INT16_C( 28038), -INT16_C(  7850) } },
    { { -INT16_C( 21711), -INT16_C( 30168),  INT16_C( 12757),  INT16_C( 31032), -INT16_C( 24491), -INT16_C(  2558),  INT16_C(  9087),  INT16_C(  3478) },
      UINT8_C(188),
       INT16_C(  7162),
      { -INT16_C( 21711), -INT16_C( 30168),  INT16_C(  7162),  INT16_C(  7162),  INT16_C(  7162),  INT16_C(  7162),  INT16_C(  9087),  INT16_C(  7162) } },
    { {  INT16_C( 17032), -INT16_C(  7230),  INT16_C( 15439), -INT16_C( 22922),  INT16_C( 30749),  INT16_C(  2346), -INT16_C( 10839),  INT16_C( 13361) },
      UINT8_C(171),
       INT16_C( 27747),
      {  INT16_C( 27747),  INT16_C( 27747),  INT16_C( 15439),  INT16_C( 27747),  INT16_C( 30749),  INT16_C( 27747), -INT16_C( 10839),  INT16_C( 27747) } },
    { { -INT16_C( 18396),  INT16_C(  9740), -INT16_C( 29522),  INT16_C( 17737),  INT16_C(  1689), -INT16_C( 19393), -INT16_C( 32114),  INT16_C( 29047) },
      UINT8_C(209),
      -INT16_C(  6221),
      { -INT16_C(  6221),  INT16_C(  9740), -INT16_C( 29522),  INT16_C( 17737), -INT16_C(  6221), -INT16_C( 19393), -INT16_C(  6221), -INT16_C(  6221) } },
    { { -INT16_C( 11913), -INT16_C( 24225),  INT16_C(  2522),  INT16_C(  3191),  INT16_C(  8765), -INT16_C( 22161),  INT16_C( 10054),  INT16_C( 27830) },
      UINT8_C(213),
      -INT16_C( 19134),
      { -INT16_C( 19134), -INT16_C( 24225), -INT16_C( 19134),  INT16_C(  3191), -INT16_C( 19134), -INT16_C( 22161), -INT16_C( 19134), -INT16_C( 19134) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    int16_t a = test_vec[i].a;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_set1_epi16(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_set1_epi16");
    easysimd_assert_m128i_i16(r, ==, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int16_t a = easysimd_test_codegen_random_i16();
    easysimd__m128i r = easysimd_mm_mask_set1_epi16(src, k, a);

    easysimd_test_x86_write_i16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_set1_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int16_t a;
    int16_t r[8];
  } test_vec[] = {
    { UINT8_C( 87),
       INT16_C(  3040),
      {  INT16_C(  3040),  INT16_C(  3040),  INT16_C(  3040),  INT16_C(     0),  INT16_C(  3040),  INT16_C(     0),  INT16_C(  3040),  INT16_C(     0) } },
    { UINT8_C(158),
      -INT16_C( 11424),
      {  INT16_C(     0), -INT16_C( 11424), -INT16_C( 11424), -INT16_C( 11424), -INT16_C( 11424),  INT16_C(     0),  INT16_C(     0), -INT16_C( 11424) } },
    { UINT8_C( 46),
      -INT16_C(  9970),
      {  INT16_C(     0), -INT16_C(  9970), -INT16_C(  9970), -INT16_C(  9970),  INT16_C(     0), -INT16_C(  9970),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(177),
      -INT16_C(  5285),
      { -INT16_C(  5285),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  5285), -INT16_C(  5285),  INT16_C(     0), -INT16_C(  5285) } },
    { UINT8_C(223),
       INT16_C(  2140),
      {  INT16_C(  2140),  INT16_C(  2140),  INT16_C(  2140),  INT16_C(  2140),  INT16_C(  2140),  INT16_C(     0),  INT16_C(  2140),  INT16_C(  2140) } },
    { UINT8_C( 65),
      -INT16_C(  7550),
      { -INT16_C(  7550),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  7550),  INT16_C(     0) } },
    { UINT8_C( 96),
      -INT16_C( 24851),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 24851), -INT16_C( 24851),  INT16_C(     0) } },
    { UINT8_C(116),
      -INT16_C(  9587),
      {  INT16_C(     0),  INT16_C(     0), -INT16_C(  9587),  INT16_C(     0), -INT16_C(  9587), -INT16_C(  9587), -INT16_C(  9587),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    int16_t a = test_vec[i].a;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_set1_epi16(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_set1_epi16");
    easysimd_assert_m128i_i16(r, ==, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int16_t a = easysimd_test_codegen_random_i16();
    easysimd__m128i r = easysimd_mm_maskz_set1_epi16(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_set1_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[4];
    easysimd__mmask8 k;
    int32_t a;
    int32_t r[4];
  } test_vec[8] = {
    { {  INT32_C(  1200844682),  INT32_C(  1187340862), -INT32_C(  1629798600),  INT32_C(   143781745) },
      UINT8_C(219),
       INT32_C(   424993589),
      {  INT32_C(   424993589),  INT32_C(   424993589), -INT32_C(  1629798600),  INT32_C(   424993589) } },
    { { -INT32_C(  1836113374), -INT32_C(   884814379),  INT32_C(   961989834), -INT32_C(  1032348320) },
      UINT8_C( 98),
      -INT32_C(  1734214979),
      { -INT32_C(  1836113374), -INT32_C(  1734214979),  INT32_C(   961989834), -INT32_C(  1032348320) } },
    { {  INT32_C(   696717976), -INT32_C(    44146150), -INT32_C(   652248906), -INT32_C(   709971449) },
      UINT8_C(244),
      -INT32_C(   809459847),
      {  INT32_C(   696717976), -INT32_C(    44146150), -INT32_C(   809459847), -INT32_C(   709971449) } },
    { {  INT32_C(  1332486360), -INT32_C(   603140382), -INT32_C(  2122996625), -INT32_C(  1902404052) },
      UINT8_C(253),
      -INT32_C(  1200274023),
      { -INT32_C(  1200274023), -INT32_C(   603140382), -INT32_C(  1200274023), -INT32_C(  1200274023) } },
    { { -INT32_C(   865436642), -INT32_C(   263889327),  INT32_C(  1070077215),  INT32_C(  1344411521) },
      UINT8_C( 35),
       INT32_C(  1942536190),
      {  INT32_C(  1942536190),  INT32_C(  1942536190),  INT32_C(  1070077215),  INT32_C(  1344411521) } },
    { { -INT32_C(   602737599), -INT32_C(   948629630),  INT32_C(    15019652), -INT32_C(   145575528) },
      UINT8_C(247),
       INT32_C(   185341506),
      {  INT32_C(   185341506),  INT32_C(   185341506),  INT32_C(   185341506), -INT32_C(   145575528) } },
    { {  INT32_C(  1998753109), -INT32_C(  1653192995),  INT32_C(    48163086),  INT32_C(   176470779) },
      UINT8_C( 48),
       INT32_C(   828280396),
      {  INT32_C(  1998753109), -INT32_C(  1653192995),  INT32_C(    48163086),  INT32_C(   176470779) } },
    { { -INT32_C(   505153905),  INT32_C(    52681453),  INT32_C(  1935224550), -INT32_C(  1756245935) },
      UINT8_C( 70),
      -INT32_C(   869227026),
      { -INT32_C(   505153905), -INT32_C(   869227026), -INT32_C(   869227026), -INT32_C(  1756245935) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    int32_t a = test_vec[i].a;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_set1_epi32(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_set1_epi32");
    easysimd_assert_m128i_i32(r, ==, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int32_t a = easysimd_test_codegen_random_i32();
    easysimd__m128i r = easysimd_mm_mask_set1_epi32(src, k, a);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_set1_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int32_t a;
    int32_t r[4];
  } test_vec[8] = {
    { UINT8_C(215),
       INT32_C(   445856074),
      {  INT32_C(   445856074),  INT32_C(   445856074),  INT32_C(   445856074),  INT32_C(           0) } },
    { UINT8_C(133),
       INT32_C(  2070245744),
      {  INT32_C(  2070245744),  INT32_C(           0),  INT32_C(  2070245744),  INT32_C(           0) } },
    { UINT8_C( 16),
      -INT32_C(   750891323),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 39),
      -INT32_C(   837668881),
      { -INT32_C(   837668881), -INT32_C(   837668881), -INT32_C(   837668881),  INT32_C(           0) } },
    { UINT8_C(139),
      -INT32_C(    49475813),
      { -INT32_C(    49475813), -INT32_C(    49475813),  INT32_C(           0), -INT32_C(    49475813) } },
    { UINT8_C(111),
      -INT32_C(  2015126534),
      { -INT32_C(  2015126534), -INT32_C(  2015126534), -INT32_C(  2015126534), -INT32_C(  2015126534) } },
    { UINT8_C(117),
       INT32_C(  1303302842),
      {  INT32_C(  1303302842),  INT32_C(           0),  INT32_C(  1303302842),  INT32_C(           0) } },
    { UINT8_C(236),
      -INT32_C(  1738621389),
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1738621389), -INT32_C(  1738621389) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    int32_t a = test_vec[i].a;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_set1_epi32(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_set1_epi32");
    easysimd_assert_m128i_i32(r, ==, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int32_t a = easysimd_test_codegen_random_i32();
    easysimd__m128i r = easysimd_mm_maskz_set1_epi32(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_set1_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[2];
    easysimd__mmask8 k;
    int64_t a;
    int64_t r[2];
  } test_vec[] = {
    { {  INT64_C(  494350939155323674), -INT64_C( 4178624510486204997) },
      UINT8_C(101),
      -INT64_C( 6489158888222278389),
      { -INT64_C( 6489158888222278389), -INT64_C( 4178624510486204997) } },
    { { -INT64_C( 4253711515777915129), -INT64_C( 2628032704799288958) },
      UINT8_C( 20),
       INT64_C(  303436373587591209),
      { -INT64_C( 4253711515777915129), -INT64_C( 2628032704799288958) } },
    { {  INT64_C( 1683793688790269558), -INT64_C( 8906777336423040230) },
      UINT8_C( 25),
      -INT64_C( 8223422777328881104),
      { -INT64_C( 8223422777328881104), -INT64_C( 8906777336423040230) } },
    { { -INT64_C( 4493652061445747672), -INT64_C( 7719653160885552453) },
      UINT8_C( 73),
      -INT64_C(  120426982544521280),
      { -INT64_C(  120426982544521280), -INT64_C( 7719653160885552453) } },
    { { -INT64_C(  626817374908380260),  INT64_C( 1765124209223826027) },
      UINT8_C( 44),
      -INT64_C( 3611677971922521790),
      { -INT64_C(  626817374908380260),  INT64_C( 1765124209223826027) } },
    { {  INT64_C( 7174822827191239885),  INT64_C( 9089748409036814242) },
      UINT8_C(128),
       INT64_C( 2641545570413157352),
      {  INT64_C( 7174822827191239885),  INT64_C( 9089748409036814242) } },
    { { -INT64_C( 5861444081117640420), -INT64_C( 1888861822249427952) },
      UINT8_C(185),
       INT64_C(  257127836442159491),
      {  INT64_C(  257127836442159491), -INT64_C( 1888861822249427952) } },
    { {  INT64_C( 7620050577522096123), -INT64_C( 2590484300305360635) },
      UINT8_C(199),
      -INT64_C( 2702182930654885069),
      { -INT64_C( 2702182930654885069), -INT64_C( 2702182930654885069) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    int64_t a = test_vec[i].a;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_set1_epi64(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_set1_epi64");
    easysimd_assert_m128i_i64(r, ==, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int64_t a = easysimd_test_codegen_random_i64();
    easysimd__m128i r = easysimd_mm_mask_set1_epi64(src, k, a);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_set1_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int64_t a;
    int64_t r[2];
  } test_vec[] = {
    { UINT8_C( 57),
       INT64_C( 6961350276914109806),
      {  INT64_C( 6961350276914109806),  INT64_C(                   0) } },
    { UINT8_C(194),
      -INT64_C(  943544836196275351),
      {  INT64_C(                   0), -INT64_C(  943544836196275351) } },
    { UINT8_C(117),
       INT64_C( 4094503565159725770),
      {  INT64_C( 4094503565159725770),  INT64_C(                   0) } },
    { UINT8_C(204),
      -INT64_C( 5741594724106135920),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(243),
       INT64_C( 3215639080034585345),
      {  INT64_C( 3215639080034585345),  INT64_C( 3215639080034585345) } },
    { UINT8_C( 54),
       INT64_C( 7122904459258696570),
      {  INT64_C(                   0),  INT64_C( 7122904459258696570) } },
    { UINT8_C(239),
      -INT64_C( 8240251668098763662),
      { -INT64_C( 8240251668098763662), -INT64_C( 8240251668098763662) } },
    { UINT8_C(194),
      -INT64_C( 6652451203756859546),
      {  INT64_C(                   0), -INT64_C( 6652451203756859546) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    int64_t a = test_vec[i].a;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_set1_epi64(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_set1_epi64");
    easysimd_assert_m128i_i64(r, ==, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int64_t a = easysimd_test_codegen_random_i64();
    easysimd__m128i r = easysimd_mm_maskz_set1_epi64(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_set1_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int8_t src[32];
    uint32_t k;
    int8_t a;
    int8_t r[32];
  } test_vec[] = {
    { {  INT8_C(  66), -INT8_C( 108), -INT8_C(   5), -INT8_C( 120),  INT8_C(  12), -INT8_C(  37),  INT8_C(  88),  INT8_C(  65),
         INT8_C(  38), -INT8_C(  27), -INT8_C(  94), -INT8_C(  85),  INT8_C(  67), -INT8_C( 104), -INT8_C(   7), -INT8_C(  80),
        -INT8_C( 108),  INT8_C(  49),  INT8_C(   0), -INT8_C(  32),  INT8_C(  67), -INT8_C(  10), -INT8_C( 118),  INT8_C(  43),
         INT8_C(  28), -INT8_C(  52), -INT8_C(  58),  INT8_C( 104),  INT8_C( 118),  INT8_C( 103), -INT8_C( 127), -INT8_C(  72) },
      UINT32_C( 138444027),
       INT8_C(  87),
      {  INT8_C(  87),  INT8_C(  87), -INT8_C(   5),  INT8_C(  87),  INT8_C(  87),  INT8_C(  87),  INT8_C(  87),  INT8_C(  87),
         INT8_C(  38), -INT8_C(  27),  INT8_C(  87),  INT8_C(  87),  INT8_C(  87),  INT8_C(  87),  INT8_C(  87), -INT8_C(  80),
        -INT8_C( 108),  INT8_C(  49),  INT8_C(   0), -INT8_C(  32),  INT8_C(  67), -INT8_C(  10),  INT8_C(  87),  INT8_C(  43),
         INT8_C(  28), -INT8_C(  52), -INT8_C(  58),  INT8_C(  87),  INT8_C( 118),  INT8_C( 103), -INT8_C( 127), -INT8_C(  72) } },
    { { -INT8_C( 104),  INT8_C(  73),  INT8_C( 125),  INT8_C( 126), -INT8_C(  21),  INT8_C(  40), -INT8_C(  63), -INT8_C( 125),
         INT8_C(  34),  INT8_C( 113),  INT8_C(  23),  INT8_C(  83),  INT8_C( 114), -INT8_C(   9), -INT8_C( 106),  INT8_C( 104),
        -INT8_C( 127), -INT8_C(  63), -INT8_C( 124),  INT8_C(  78), -INT8_C( 120), -INT8_C(  19), -INT8_C(  60), -INT8_C(  17),
         INT8_C( 110),  INT8_C( 124), -INT8_C(  22), -INT8_C(  21), -INT8_C(  68), -INT8_C(  14),  INT8_C(  66),  INT8_C(  85) },
      UINT32_C( 651411515),
      -INT8_C(  24),
      { -INT8_C(  24), -INT8_C(  24),  INT8_C( 125), -INT8_C(  24), -INT8_C(  24), -INT8_C(  24), -INT8_C(  63), -INT8_C( 125),
         INT8_C(  34),  INT8_C( 113),  INT8_C(  23),  INT8_C(  83),  INT8_C( 114), -INT8_C(   9), -INT8_C(  24), -INT8_C(  24),
        -INT8_C(  24), -INT8_C(  24), -INT8_C( 124),  INT8_C(  78), -INT8_C(  24), -INT8_C(  19), -INT8_C(  24), -INT8_C(  24),
         INT8_C( 110), -INT8_C(  24), -INT8_C(  24), -INT8_C(  21), -INT8_C(  68), -INT8_C(  24),  INT8_C(  66),  INT8_C(  85) } },
    { { -INT8_C( 108), -INT8_C(  87),  INT8_C(  10),  INT8_C(   6), -INT8_C(  64),  INT8_C(  94),  INT8_C( 120), -INT8_C(  72),
        -INT8_C(  12), -INT8_C(  32),  INT8_C(  57), -INT8_C(  74),  INT8_C( 101), -INT8_C( 121),  INT8_C(  62),  INT8_C(  82),
         INT8_C(  75),  INT8_C(  45), -INT8_C(  64), -INT8_C(  57),  INT8_C(  23), -INT8_C(  85), -INT8_C( 124),  INT8_C(  10),
        -INT8_C(  18), -INT8_C(  39),  INT8_C(  69), -INT8_C(  82), -INT8_C(  84),  INT8_C( 108), -INT8_C( 106),  INT8_C(  64) },
      UINT32_C(3594952981),
      -INT8_C(   1),
      { -INT8_C(   1), -INT8_C(  87), -INT8_C(   1),  INT8_C(   6), -INT8_C(   1),  INT8_C(  94),  INT8_C( 120), -INT8_C(  72),
        -INT8_C(   1), -INT8_C(  32),  INT8_C(  57), -INT8_C(  74),  INT8_C( 101), -INT8_C(   1),  INT8_C(  62), -INT8_C(   1),
         INT8_C(  75), -INT8_C(   1), -INT8_C(   1), -INT8_C(  57),  INT8_C(  23), -INT8_C(  85), -INT8_C(   1),  INT8_C(  10),
        -INT8_C(  18), -INT8_C(   1), -INT8_C(   1), -INT8_C(  82), -INT8_C(   1),  INT8_C( 108), -INT8_C(   1), -INT8_C(   1) } },
    { { -INT8_C(  66), -INT8_C( 114), -INT8_C(  13), -INT8_C(  97), -INT8_C(  57), -INT8_C(  87),  INT8_C(   4),  INT8_C(  79),
        -INT8_C(  25),  INT8_C(  86), -INT8_C( 102),  INT8_C(  20),  INT8_C(  22),  INT8_C(  98),  INT8_C(  44), -INT8_C(  62),
        -INT8_C(  26),  INT8_C(  54), -INT8_C(  80), -INT8_C(  65),  INT8_C( 123),  INT8_C(  94),  INT8_C( 107), -INT8_C(  25),
        -INT8_C(  12), -INT8_C(  85), -INT8_C(   3), -INT8_C( 107), -INT8_C(  14), -INT8_C(  45), -INT8_C( 108), -INT8_C(  80) },
      UINT32_C( 676300897),
       INT8_C(  49),
      {  INT8_C(  49), -INT8_C( 114), -INT8_C(  13), -INT8_C(  97), -INT8_C(  57),  INT8_C(  49),  INT8_C(  49),  INT8_C(  79),
        -INT8_C(  25),  INT8_C(  86), -INT8_C( 102),  INT8_C(  49),  INT8_C(  22),  INT8_C(  98),  INT8_C(  44),  INT8_C(  49),
         INT8_C(  49),  INT8_C(  49),  INT8_C(  49),  INT8_C(  49),  INT8_C( 123),  INT8_C(  94),  INT8_C(  49), -INT8_C(  25),
        -INT8_C(  12), -INT8_C(  85), -INT8_C(   3),  INT8_C(  49), -INT8_C(  14),  INT8_C(  49), -INT8_C( 108), -INT8_C(  80) } },
    { {  INT8_C(  83),  INT8_C( 119),  INT8_C(  25), -INT8_C(  87),  INT8_C(  18),  INT8_C(  45), -INT8_C(  64),  INT8_C( 116),
         INT8_C(  89), -INT8_C( 126),  INT8_C(  90), -INT8_C( 113),  INT8_C(  50),  INT8_C(  25),  INT8_C(  11), -INT8_C( 112),
        -INT8_C( 124), -INT8_C(  14), -INT8_C( 124),  INT8_C(  47), -INT8_C(  17),  INT8_C(  26),  INT8_C(  33), -INT8_C(  62),
        -INT8_C(  82), -INT8_C(  46),  INT8_C(  35),  INT8_C(  54),  INT8_C(  33),  INT8_C(  76),  INT8_C( 104),  INT8_C( 117) },
      UINT32_C(3575546307),
      -INT8_C(  82),
      { -INT8_C(  82), -INT8_C(  82),  INT8_C(  25), -INT8_C(  87),  INT8_C(  18),  INT8_C(  45), -INT8_C(  82), -INT8_C(  82),
        -INT8_C(  82), -INT8_C( 126),  INT8_C(  90), -INT8_C( 113),  INT8_C(  50),  INT8_C(  25),  INT8_C(  11), -INT8_C(  82),
        -INT8_C( 124), -INT8_C(  82), -INT8_C(  82), -INT8_C(  82), -INT8_C(  82),  INT8_C(  26),  INT8_C(  33), -INT8_C(  62),
        -INT8_C(  82), -INT8_C(  46), -INT8_C(  82),  INT8_C(  54), -INT8_C(  82),  INT8_C(  76), -INT8_C(  82), -INT8_C(  82) } },
    { { -INT8_C(  34),  INT8_C(  73),  INT8_C(   8),  INT8_C(  96), -INT8_C(  93), -INT8_C( 105), -INT8_C( 110), -INT8_C(  68),
        -INT8_C(  94),  INT8_C(  34),  INT8_C(  64), -INT8_C( 107), -INT8_C(  89),  INT8_C( 112), -INT8_C( 124), -INT8_C(  63),
        -INT8_C( 111),  INT8_C(  71),  INT8_C( 111),  INT8_C(  99),  INT8_C( 106), -INT8_C(  90), -INT8_C( 123), -INT8_C(  74),
         INT8_C(  14), -INT8_C(   6),  INT8_C( 122), -INT8_C( 113),  INT8_C(  24),  INT8_C(  79),  INT8_C(  61), -INT8_C(   9) },
      UINT32_C(1012352409),
      -INT8_C(  35),
      { -INT8_C(  35),  INT8_C(  73),  INT8_C(   8), -INT8_C(  35), -INT8_C(  35), -INT8_C( 105), -INT8_C( 110), -INT8_C(  35),
        -INT8_C(  35),  INT8_C(  34), -INT8_C(  35), -INT8_C( 107), -INT8_C(  89),  INT8_C( 112), -INT8_C(  35), -INT8_C(  63),
        -INT8_C(  35), -INT8_C(  35), -INT8_C(  35),  INT8_C(  99), -INT8_C(  35), -INT8_C(  90), -INT8_C(  35), -INT8_C(  74),
         INT8_C(  14), -INT8_C(   6), -INT8_C(  35), -INT8_C(  35), -INT8_C(  35), -INT8_C(  35),  INT8_C(  61), -INT8_C(   9) } },
    { { -INT8_C(  22), -INT8_C(   7),      INT8_MAX,  INT8_C(  12),  INT8_C(  57),  INT8_C(  20), -INT8_C(  77), -INT8_C(  87),
        -INT8_C( 103),  INT8_C( 116),  INT8_C(  59), -INT8_C(  32), -INT8_C(  28), -INT8_C(  98),  INT8_C(  74), -INT8_C( 118),
         INT8_C(  35),  INT8_C(   1), -INT8_C( 104),  INT8_C(  29),  INT8_C( 123),  INT8_C(  39),  INT8_C(  54), -INT8_C(  54),
         INT8_C( 100),  INT8_C(  45),  INT8_C(  99), -INT8_C(  86), -INT8_C( 124), -INT8_C(  96), -INT8_C( 121),  INT8_C( 110) },
      UINT32_C(3531277977),
       INT8_C(  27),
      {  INT8_C(  27), -INT8_C(   7),      INT8_MAX,  INT8_C(  27),  INT8_C(  27),  INT8_C(  20), -INT8_C(  77),  INT8_C(  27),
        -INT8_C( 103),  INT8_C(  27),  INT8_C(  27), -INT8_C(  32), -INT8_C(  28), -INT8_C(  98),  INT8_C(  74), -INT8_C( 118),
         INT8_C(  27),  INT8_C(  27), -INT8_C( 104),  INT8_C(  27),  INT8_C(  27),  INT8_C(  27),  INT8_C(  27), -INT8_C(  54),
         INT8_C( 100),  INT8_C(  27),  INT8_C(  99), -INT8_C(  86),  INT8_C(  27), -INT8_C(  96),  INT8_C(  27),  INT8_C(  27) } },
    { {  INT8_C(  46),  INT8_C( 124), -INT8_C(  76), -INT8_C(  93), -INT8_C(  73), -INT8_C( 108), -INT8_C( 121),  INT8_C(  85),
        -INT8_C(  34),  INT8_C(  17),  INT8_C( 121), -INT8_C(  33), -INT8_C(  87), -INT8_C( 106),  INT8_C(  90), -INT8_C(  48),
        -INT8_C(  52),  INT8_C(  37),  INT8_C(  52), -INT8_C(   7), -INT8_C( 120), -INT8_C(  34),  INT8_C( 126),  INT8_C(  40),
         INT8_C( 101), -INT8_C(  20), -INT8_C(  63),  INT8_C( 108),  INT8_C( 103), -INT8_C( 108), -INT8_C( 121), -INT8_C( 106) },
      UINT32_C(3342416656),
      -INT8_C(  49),
      {  INT8_C(  46),  INT8_C( 124), -INT8_C(  76), -INT8_C(  93), -INT8_C(  49), -INT8_C( 108), -INT8_C( 121),  INT8_C(  85),
        -INT8_C(  49), -INT8_C(  49),  INT8_C( 121), -INT8_C(  49), -INT8_C(  49), -INT8_C(  49),  INT8_C(  90), -INT8_C(  48),
        -INT8_C(  49),  INT8_C(  37),  INT8_C(  52), -INT8_C(  49), -INT8_C(  49), -INT8_C(  49),  INT8_C( 126),  INT8_C(  40),
        -INT8_C(  49), -INT8_C(  49), -INT8_C(  49),  INT8_C( 108),  INT8_C( 103), -INT8_C( 108), -INT8_C(  49), -INT8_C(  49) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi8(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    int8_t a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_set1_epi8(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_set1_epi8");
    easysimd_assert_m256i_i8(r, ==, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    int8_t a = easysimd_test_codegen_random_i8();
    easysimd__m256i r = easysimd_mm256_mask_set1_epi8(src, k, a);

    easysimd_test_x86_write_i8x32(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_set1_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint32_t k;
    int8_t a;
    int8_t r[32];
  } test_vec[] = {
    { UINT32_C(3065499726),
      -INT8_C(  71),
      {  INT8_C(   0), -INT8_C(  71), -INT8_C(  71), -INT8_C(  71),  INT8_C(   0),  INT8_C(   0), -INT8_C(  71),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  71),  INT8_C(   0), -INT8_C(  71), -INT8_C(  71),
        -INT8_C(  71), -INT8_C(  71), -INT8_C(  71),  INT8_C(   0), -INT8_C(  71), -INT8_C(  71),  INT8_C(   0), -INT8_C(  71),
         INT8_C(   0), -INT8_C(  71), -INT8_C(  71),  INT8_C(   0), -INT8_C(  71), -INT8_C(  71),  INT8_C(   0), -INT8_C(  71) } },
    { UINT32_C( 554845085),
      -INT8_C(  25),
      { -INT8_C(  25),  INT8_C(   0), -INT8_C(  25), -INT8_C(  25), -INT8_C(  25),  INT8_C(   0),  INT8_C(   0), -INT8_C(  25),
        -INT8_C(  25), -INT8_C(  25),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  25),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  25),  INT8_C(   0),  INT8_C(   0), -INT8_C(  25),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  25),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  25),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C( 800328439),
      -INT8_C( 114),
      { -INT8_C( 114), -INT8_C( 114), -INT8_C( 114),  INT8_C(   0), -INT8_C( 114), -INT8_C( 114), -INT8_C( 114), -INT8_C( 114),
         INT8_C(   0), -INT8_C( 114),  INT8_C(   0), -INT8_C( 114),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C( 114),  INT8_C(   0), -INT8_C( 114), -INT8_C( 114),  INT8_C(   0), -INT8_C( 114),
        -INT8_C( 114), -INT8_C( 114), -INT8_C( 114), -INT8_C( 114),  INT8_C(   0), -INT8_C( 114),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C( 353480848),
       INT8_C(  99),
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),  INT8_C(  99),  INT8_C(   0),  INT8_C(  99),
         INT8_C(  99),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  99),  INT8_C(   0),  INT8_C(  99),  INT8_C(   0),  INT8_C(  99),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C(3820294450),
       INT8_C(  40),
      {  INT8_C(   0),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),  INT8_C(  40),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  40),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  40),  INT8_C(   0),  INT8_C(  40),  INT8_C(   0),  INT8_C(  40),  INT8_C(  40),  INT8_C(   0),  INT8_C(  40),
         INT8_C(  40),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  40),  INT8_C(  40),  INT8_C(  40) } },
    { UINT32_C(2668800109),
       INT8_C( 112),
      {  INT8_C( 112),  INT8_C(   0),  INT8_C( 112),  INT8_C( 112),  INT8_C(   0),  INT8_C( 112),  INT8_C( 112),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 112),  INT8_C(   0),  INT8_C( 112),  INT8_C(   0),  INT8_C( 112),
         INT8_C(   0),  INT8_C( 112),  INT8_C(   0),  INT8_C(   0),  INT8_C( 112),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 112),  INT8_C( 112),  INT8_C( 112),  INT8_C( 112),  INT8_C( 112),  INT8_C(   0),  INT8_C(   0),  INT8_C( 112) } },
    { UINT32_C(1010888069),
      -INT8_C(  92),
      { -INT8_C(  92),  INT8_C(   0), -INT8_C(  92),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  92),
        -INT8_C(  92),  INT8_C(   0), -INT8_C(  92), -INT8_C(  92),  INT8_C(   0), -INT8_C(  92), -INT8_C(  92), -INT8_C(  92),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  92),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  92), -INT8_C(  92), -INT8_C(  92), -INT8_C(  92),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C( 199744249),
      -INT8_C(   5),
      { -INT8_C(   5),  INT8_C(   0),  INT8_C(   0), -INT8_C(   5), -INT8_C(   5), -INT8_C(   5), -INT8_C(   5), -INT8_C(   5),
         INT8_C(   0), -INT8_C(   5),  INT8_C(   0), -INT8_C(   5), -INT8_C(   5),  INT8_C(   0), -INT8_C(   5), -INT8_C(   5),
        -INT8_C(   5), -INT8_C(   5), -INT8_C(   5),  INT8_C(   0),  INT8_C(   0), -INT8_C(   5), -INT8_C(   5), -INT8_C(   5),
        -INT8_C(   5), -INT8_C(   5),  INT8_C(   0), -INT8_C(   5),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    int8_t a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_set1_epi8(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_set1_epi8");
    easysimd_assert_m256i_i8(r, ==, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    int8_t a = easysimd_test_codegen_random_i8();
    easysimd__m256i r = easysimd_mm256_maskz_set1_epi8(k, a);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_set1_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t src[16];
    uint16_t k;
    int16_t a;
    int16_t r[16];
  } test_vec[] = {
    { {  INT16_C(  7360), -INT16_C( 11859), -INT16_C( 29291),  INT16_C( 11386),  INT16_C( 19175),  INT16_C(  3320), -INT16_C(  3458),  INT16_C( 23957),
        -INT16_C( 17040),  INT16_C( 23746),  INT16_C( 11903),  INT16_C(  5060),  INT16_C( 23221), -INT16_C(  4061), -INT16_C(  5485),  INT16_C( 21439) },
      UINT16_C(27910),
      -INT16_C( 25564),
      {  INT16_C(  7360), -INT16_C( 25564), -INT16_C( 25564),  INT16_C( 11386),  INT16_C( 19175),  INT16_C(  3320), -INT16_C(  3458),  INT16_C( 23957),
        -INT16_C( 25564),  INT16_C( 23746), -INT16_C( 25564), -INT16_C( 25564),  INT16_C( 23221), -INT16_C( 25564), -INT16_C( 25564),  INT16_C( 21439) } },
    { { -INT16_C( 24838), -INT16_C(  7736), -INT16_C( 16152),  INT16_C( 26350), -INT16_C( 31822),  INT16_C(  8899), -INT16_C( 31168), -INT16_C( 16513),
         INT16_C( 17332),  INT16_C( 27346), -INT16_C(  2659),  INT16_C( 12378),  INT16_C(  6879), -INT16_C(  6525), -INT16_C( 22649), -INT16_C( 32382) },
      UINT16_C(19013),
       INT16_C( 11618),
      {  INT16_C( 11618), -INT16_C(  7736),  INT16_C( 11618),  INT16_C( 26350), -INT16_C( 31822),  INT16_C(  8899),  INT16_C( 11618), -INT16_C( 16513),
         INT16_C( 17332),  INT16_C( 11618), -INT16_C(  2659),  INT16_C( 11618),  INT16_C(  6879), -INT16_C(  6525),  INT16_C( 11618), -INT16_C( 32382) } },
    { {  INT16_C( 20490), -INT16_C( 17005),  INT16_C( 22483),  INT16_C(  5343),  INT16_C( 24285), -INT16_C( 28205), -INT16_C( 22879),  INT16_C( 16123),
         INT16_C( 22171),  INT16_C( 31598), -INT16_C(  3728), -INT16_C(  2207), -INT16_C(  7272), -INT16_C(  8840), -INT16_C(  9683),  INT16_C( 14090) },
      UINT16_C(40491),
      -INT16_C(   268),
      { -INT16_C(   268), -INT16_C(   268),  INT16_C( 22483), -INT16_C(   268),  INT16_C( 24285), -INT16_C(   268), -INT16_C( 22879),  INT16_C( 16123),
         INT16_C( 22171), -INT16_C(   268), -INT16_C(   268), -INT16_C(   268), -INT16_C(   268), -INT16_C(  8840), -INT16_C(  9683), -INT16_C(   268) } },
    { { -INT16_C( 11019), -INT16_C( 11758), -INT16_C(  6606), -INT16_C( 11165),  INT16_C( 24460),  INT16_C( 10002), -INT16_C( 32331),  INT16_C(  9634),
         INT16_C(   882),  INT16_C(  2844), -INT16_C( 27418),  INT16_C(  5096), -INT16_C(  3218), -INT16_C( 26293),  INT16_C( 16273), -INT16_C( 31080) },
      UINT16_C(43539),
       INT16_C( 18008),
      {  INT16_C( 18008),  INT16_C( 18008), -INT16_C(  6606), -INT16_C( 11165),  INT16_C( 18008),  INT16_C( 10002), -INT16_C( 32331),  INT16_C(  9634),
         INT16_C(   882),  INT16_C( 18008), -INT16_C( 27418),  INT16_C( 18008), -INT16_C(  3218),  INT16_C( 18008),  INT16_C( 16273),  INT16_C( 18008) } },
    { { -INT16_C( 17520),  INT16_C(  7194),  INT16_C( 11290), -INT16_C( 12476), -INT16_C(  6483),  INT16_C(  8436),  INT16_C(  4330), -INT16_C( 12245),
         INT16_C(  5028),  INT16_C(  5092),  INT16_C( 12038), -INT16_C( 26708),  INT16_C( 17518), -INT16_C( 32227),  INT16_C( 30191),  INT16_C( 32712) },
      UINT16_C(57905),
       INT16_C( 19356),
      {  INT16_C( 19356),  INT16_C(  7194),  INT16_C( 11290), -INT16_C( 12476),  INT16_C( 19356),  INT16_C( 19356),  INT16_C(  4330), -INT16_C( 12245),
         INT16_C(  5028),  INT16_C( 19356),  INT16_C( 12038), -INT16_C( 26708),  INT16_C( 17518),  INT16_C( 19356),  INT16_C( 19356),  INT16_C( 19356) } },
    { { -INT16_C(  8178), -INT16_C( 17381),  INT16_C(  4038), -INT16_C( 20260),  INT16_C(  1824), -INT16_C( 15231),  INT16_C( 25882),  INT16_C(  8663),
        -INT16_C( 31596),  INT16_C(   696), -INT16_C( 10552), -INT16_C( 18556),  INT16_C( 19531),  INT16_C( 31799), -INT16_C( 11474),  INT16_C( 15816) },
      UINT16_C(58291),
       INT16_C( 31225),
      {  INT16_C( 31225),  INT16_C( 31225),  INT16_C(  4038), -INT16_C( 20260),  INT16_C( 31225),  INT16_C( 31225),  INT16_C( 25882),  INT16_C( 31225),
         INT16_C( 31225),  INT16_C( 31225), -INT16_C( 10552), -INT16_C( 18556),  INT16_C( 19531),  INT16_C( 31225),  INT16_C( 31225),  INT16_C( 31225) } },
    { { -INT16_C( 10766),  INT16_C(  4650), -INT16_C( 21540), -INT16_C(  2345), -INT16_C( 20976), -INT16_C( 23529), -INT16_C( 12238), -INT16_C(  1114),
         INT16_C( 11174), -INT16_C(  3662), -INT16_C(  5769), -INT16_C( 22930),  INT16_C( 14012),  INT16_C( 28643), -INT16_C(  9191),  INT16_C(  3049) },
      UINT16_C( 5041),
      -INT16_C( 29410),
      { -INT16_C( 29410),  INT16_C(  4650), -INT16_C( 21540), -INT16_C(  2345), -INT16_C( 29410), -INT16_C( 29410), -INT16_C( 12238), -INT16_C( 29410),
        -INT16_C( 29410), -INT16_C( 29410), -INT16_C(  5769), -INT16_C( 22930), -INT16_C( 29410),  INT16_C( 28643), -INT16_C(  9191),  INT16_C(  3049) } },
    { { -INT16_C(  2626), -INT16_C( 12669), -INT16_C( 25693), -INT16_C( 10638),  INT16_C(  6251),  INT16_C(  4561), -INT16_C( 31933), -INT16_C( 17662),
         INT16_C( 28781),  INT16_C( 10593),  INT16_C( 17574), -INT16_C( 16487), -INT16_C( 32224), -INT16_C( 11829), -INT16_C(  5739),  INT16_C( 21342) },
      UINT16_C(57822),
      -INT16_C( 32479),
      { -INT16_C(  2626), -INT16_C( 32479), -INT16_C( 32479), -INT16_C( 32479), -INT16_C( 32479),  INT16_C(  4561), -INT16_C( 32479), -INT16_C( 32479),
        -INT16_C( 32479),  INT16_C( 10593),  INT16_C( 17574), -INT16_C( 16487), -INT16_C( 32224), -INT16_C( 32479), -INT16_C( 32479), -INT16_C( 32479) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    int16_t a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_set1_epi16(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_set1_epi16");
    easysimd_assert_m256i_i16(r, ==, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    int16_t a = easysimd_test_codegen_random_i16();
    easysimd__m256i r = easysimd_mm256_mask_set1_epi16(src, k, a);

    easysimd_test_x86_write_i16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_set1_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t k;
    int16_t a;
    int16_t r[16];
  } test_vec[] = {
    { UINT16_C(  719),
      -INT16_C( 31995),
      { -INT16_C( 31995), -INT16_C( 31995), -INT16_C( 31995), -INT16_C( 31995),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31995), -INT16_C( 31995),
         INT16_C(     0), -INT16_C( 31995),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(37937),
      -INT16_C(  7917),
      { -INT16_C(  7917),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  7917), -INT16_C(  7917),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C(  7917),  INT16_C(     0), -INT16_C(  7917),  INT16_C(     0),  INT16_C(     0), -INT16_C(  7917) } },
    { UINT16_C(10405),
      -INT16_C( 10427),
      { -INT16_C( 10427),  INT16_C(     0), -INT16_C( 10427),  INT16_C(     0),  INT16_C(     0), -INT16_C( 10427),  INT16_C(     0), -INT16_C( 10427),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 10427),  INT16_C(     0), -INT16_C( 10427),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(64058),
       INT16_C( 25275),
      {  INT16_C(     0),  INT16_C( 25275),  INT16_C(     0),  INT16_C( 25275),  INT16_C( 25275),  INT16_C( 25275),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 25275),  INT16_C(     0),  INT16_C( 25275),  INT16_C( 25275),  INT16_C( 25275),  INT16_C( 25275),  INT16_C( 25275) } },
    { UINT16_C(25447),
       INT16_C(  1908),
      {  INT16_C(  1908),  INT16_C(  1908),  INT16_C(  1908),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1908),  INT16_C(  1908),  INT16_C(     0),
         INT16_C(  1908),  INT16_C(  1908),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1908),  INT16_C(  1908),  INT16_C(     0) } },
    { UINT16_C(63955),
       INT16_C(  5108),
      {  INT16_C(  5108),  INT16_C(  5108),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5108),  INT16_C(     0),  INT16_C(  5108),  INT16_C(  5108),
         INT16_C(  5108),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5108),  INT16_C(  5108),  INT16_C(  5108),  INT16_C(  5108),  INT16_C(  5108) } },
    { UINT16_C(38966),
       INT16_C(  4109),
      {  INT16_C(     0),  INT16_C(  4109),  INT16_C(  4109),  INT16_C(     0),  INT16_C(  4109),  INT16_C(  4109),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4109),  INT16_C(  4109),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4109) } },
    { UINT16_C( 6272),
       INT16_C( 20235),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20235),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20235),  INT16_C( 20235),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    int16_t a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_set1_epi16(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_set1_epi16");
    easysimd_assert_m256i_i16(r, ==, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    int16_t a = easysimd_test_codegen_random_i16();
    easysimd__m256i r = easysimd_mm256_maskz_set1_epi16(k, a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_set1_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t src[8];
    uint8_t k;
    int32_t a;
    int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(   413691012), -INT32_C(   268949333),  INT32_C(   430635948),  INT32_C(   306318187),  INT32_C(  1876024143), -INT32_C(   230646691),  INT32_C(  1665506949), -INT32_C(    52074881) },
      UINT8_C(249),
       INT32_C(  1705370428),
      {  INT32_C(  1705370428), -INT32_C(   268949333),  INT32_C(   430635948),  INT32_C(  1705370428),  INT32_C(  1705370428),  INT32_C(  1705370428),  INT32_C(  1705370428),  INT32_C(  1705370428) } },
    { { -INT32_C(   686713636),  INT32_C(  1229072958),  INT32_C(  1217942636), -INT32_C(  1029372122), -INT32_C(   448227257),  INT32_C(  1147448285), -INT32_C(   851615600),  INT32_C(   540205636) },
      UINT8_C(118),
       INT32_C(  1840576323),
      { -INT32_C(   686713636),  INT32_C(  1840576323),  INT32_C(  1840576323), -INT32_C(  1029372122),  INT32_C(  1840576323),  INT32_C(  1840576323),  INT32_C(  1840576323),  INT32_C(   540205636) } },
    { { -INT32_C(  1898316487), -INT32_C(  1665916523),  INT32_C(  1608742599),  INT32_C(  1782368446), -INT32_C(  1912963027), -INT32_C(  1596864578),  INT32_C(  1008202233),  INT32_C(   581553128) },
      UINT8_C(201),
      -INT32_C(  1537298301),
      { -INT32_C(  1537298301), -INT32_C(  1665916523),  INT32_C(  1608742599), -INT32_C(  1537298301), -INT32_C(  1912963027), -INT32_C(  1596864578), -INT32_C(  1537298301), -INT32_C(  1537298301) } },
    { { -INT32_C(   630457500), -INT32_C(  1483093282), -INT32_C(  2016148729), -INT32_C(   985308675),  INT32_C(   616490547), -INT32_C(   938607619),  INT32_C(   663826340),  INT32_C(  1137504479) },
      UINT8_C(235),
       INT32_C(    46734647),
      {  INT32_C(    46734647),  INT32_C(    46734647), -INT32_C(  2016148729),  INT32_C(    46734647),  INT32_C(   616490547),  INT32_C(    46734647),  INT32_C(    46734647),  INT32_C(    46734647) } },
    { { -INT32_C(  1190563658), -INT32_C(  1497919420), -INT32_C(  1126597418),  INT32_C(   901381434),  INT32_C(   970621194),  INT32_C(    51904787),  INT32_C(    99507149),  INT32_C(   789034873) },
      UINT8_C( 40),
      -INT32_C(  1603475184),
      { -INT32_C(  1190563658), -INT32_C(  1497919420), -INT32_C(  1126597418), -INT32_C(  1603475184),  INT32_C(   970621194), -INT32_C(  1603475184),  INT32_C(    99507149),  INT32_C(   789034873) } },
    { {  INT32_C(   477500064), -INT32_C(   380226837),  INT32_C(  1844677867),  INT32_C(  1736453478), -INT32_C(  1590328507),  INT32_C(   672807537),  INT32_C(  1364216129), -INT32_C(   755909326) },
      UINT8_C(207),
      -INT32_C(  1698959768),
      { -INT32_C(  1698959768), -INT32_C(  1698959768), -INT32_C(  1698959768), -INT32_C(  1698959768), -INT32_C(  1590328507),  INT32_C(   672807537), -INT32_C(  1698959768), -INT32_C(  1698959768) } },
    { { -INT32_C(   779705275), -INT32_C(  1002966121), -INT32_C(   167076237),  INT32_C(   224898003),  INT32_C(   240029637), -INT32_C(  1656643616), -INT32_C(   110357615),  INT32_C(  1200826114) },
      UINT8_C(203),
       INT32_C(   207820825),
      {  INT32_C(   207820825),  INT32_C(   207820825), -INT32_C(   167076237),  INT32_C(   207820825),  INT32_C(   240029637), -INT32_C(  1656643616),  INT32_C(   207820825),  INT32_C(   207820825) } },
    { { -INT32_C(   310433969), -INT32_C(   591301327),  INT32_C(  1822543580), -INT32_C(  1135824867),  INT32_C(    72215025),  INT32_C(  2097563221), -INT32_C(   213365286), -INT32_C(  1275024539) },
      UINT8_C(211),
      -INT32_C(   184245633),
      { -INT32_C(   184245633), -INT32_C(   184245633),  INT32_C(  1822543580), -INT32_C(  1135824867), -INT32_C(   184245633),  INT32_C(  2097563221), -INT32_C(   184245633), -INT32_C(   184245633) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    int32_t a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_set1_epi32(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_set1_epi32");
    easysimd_assert_m256i_i32(r, ==, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int32_t a = easysimd_test_codegen_random_i32();
    easysimd__m256i r = easysimd_mm256_mask_set1_epi32(src, k, a);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_set1_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int32_t a;
    int32_t r[8];
  } test_vec[] = {
    { UINT8_C( 27),
      -INT32_C(  1521692143),
      { -INT32_C(  1521692143), -INT32_C(  1521692143),  INT32_C(           0), -INT32_C(  1521692143), -INT32_C(  1521692143),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(229),
       INT32_C(  1930250798),
      {  INT32_C(  1930250798),  INT32_C(           0),  INT32_C(  1930250798),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1930250798),  INT32_C(  1930250798),  INT32_C(  1930250798) } },
    { UINT8_C( 33),
      -INT32_C(  1445171897),
      { -INT32_C(  1445171897),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1445171897),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(212),
       INT32_C(   333127232),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   333127232),  INT32_C(           0),  INT32_C(   333127232),  INT32_C(           0),  INT32_C(   333127232),  INT32_C(   333127232) } },
    { UINT8_C( 23),
       INT32_C(  1749886928),
      {  INT32_C(  1749886928),  INT32_C(  1749886928),  INT32_C(  1749886928),  INT32_C(           0),  INT32_C(  1749886928),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 52),
       INT32_C(  1766647901),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1766647901),  INT32_C(           0),  INT32_C(  1766647901),  INT32_C(  1766647901),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 55),
      -INT32_C(  1274447257),
      { -INT32_C(  1274447257), -INT32_C(  1274447257), -INT32_C(  1274447257),  INT32_C(           0), -INT32_C(  1274447257), -INT32_C(  1274447257),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 31),
      -INT32_C(    60169490),
      { -INT32_C(    60169490), -INT32_C(    60169490), -INT32_C(    60169490), -INT32_C(    60169490), -INT32_C(    60169490),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    int32_t a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_set1_epi32(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_set1_epi32");
    easysimd_assert_m256i_i32(r, ==, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int32_t a = easysimd_test_codegen_random_i32();
    easysimd__m256i r = easysimd_mm256_maskz_set1_epi32(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_set1_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t src[4];
    uint8_t k;
    int64_t a;
    int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 3624902379470709091), -INT64_C( 6933247265999746167),  INT64_C( 4170734084856300846), -INT64_C( 7214467291406363028) },
      UINT8_C(241),
       INT64_C( 2772429684551765170),
      {  INT64_C( 2772429684551765170), -INT64_C( 6933247265999746167),  INT64_C( 4170734084856300846), -INT64_C( 7214467291406363028) } },
    { { -INT64_C(  495363296011031350),  INT64_C( 1472606151119379606), -INT64_C( 1167022479839044661), -INT64_C( 5551714683714584141) },
      UINT8_C( 68),
       INT64_C( 7000829972318823539),
      { -INT64_C(  495363296011031350),  INT64_C( 1472606151119379606),  INT64_C( 7000829972318823539), -INT64_C( 5551714683714584141) } },
    { { -INT64_C(  791521270094157192), -INT64_C( 1235920698989224129),  INT64_C( 1124924212022520063), -INT64_C(  121096332585691860) },
      UINT8_C(241),
       INT64_C( 1757106688014038576),
      {  INT64_C( 1757106688014038576), -INT64_C( 1235920698989224129),  INT64_C( 1124924212022520063), -INT64_C(  121096332585691860) } },
    { {  INT64_C(  779600348007663793),  INT64_C( 6502272919520291208), -INT64_C( 5832478734638043818), -INT64_C( 3934398845943637995) },
      UINT8_C(189),
      -INT64_C( 7753464165223654869),
      { -INT64_C( 7753464165223654869),  INT64_C( 6502272919520291208), -INT64_C( 7753464165223654869), -INT64_C( 7753464165223654869) } },
    { { -INT64_C( 2243719831191128368), -INT64_C( 6950728749207848555),  INT64_C( 7697355721565264352), -INT64_C( 1505418217185021413) },
      UINT8_C( 36),
       INT64_C( 5097994845249040757),
      { -INT64_C( 2243719831191128368), -INT64_C( 6950728749207848555),  INT64_C( 5097994845249040757), -INT64_C( 1505418217185021413) } },
    { { -INT64_C( 7564941548160469994), -INT64_C( 2503928914610022236),  INT64_C( 6223100441077008009),  INT64_C(  137711544904463189) },
      UINT8_C(115),
       INT64_C( 4668202188012594231),
      {  INT64_C( 4668202188012594231),  INT64_C( 4668202188012594231),  INT64_C( 6223100441077008009),  INT64_C(  137711544904463189) } },
    { { -INT64_C(  834564312802792032),  INT64_C( 3619254723834913281), -INT64_C(  117622696273125972), -INT64_C(  528070865827229886) },
      UINT8_C(216),
      -INT64_C( 1738115377675248938),
      { -INT64_C(  834564312802792032),  INT64_C( 3619254723834913281), -INT64_C(  117622696273125972), -INT64_C( 1738115377675248938) } },
    { { -INT64_C( 2474981982494865969),  INT64_C( 9193433367425880629), -INT64_C( 9113367360146243388),  INT64_C( 6113425963169647786) },
      UINT8_C(  5),
      -INT64_C( 8643846985020518467),
      { -INT64_C( 8643846985020518467),  INT64_C( 9193433367425880629), -INT64_C( 8643846985020518467),  INT64_C( 6113425963169647786) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    int64_t a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_set1_epi64(src, k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_set1_epi64");
    easysimd_assert_m256i_i64(r, ==, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int64_t a = easysimd_test_codegen_random_i64();
    easysimd__m256i r = easysimd_mm256_mask_set1_epi64(src, k, a);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_set1_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t k;
    int64_t a;
    int64_t r[4];
  } test_vec[] = {
    { UINT8_C( 85),
       INT64_C(  839805801920086922),
      {  INT64_C(  839805801920086922),  INT64_C(                   0),  INT64_C(  839805801920086922),  INT64_C(                   0) } },
    { UINT8_C(114),
      -INT64_C( 3668838919631134022),
      {  INT64_C(                   0), -INT64_C( 3668838919631134022),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(147),
      -INT64_C( 3495403880004831646),
      { -INT64_C( 3495403880004831646), -INT64_C( 3495403880004831646),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(194),
       INT64_C( 3059636036419074143),
      {  INT64_C(                   0),  INT64_C( 3059636036419074143),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(239),
       INT64_C( 7908633806112477453),
      {  INT64_C( 7908633806112477453),  INT64_C( 7908633806112477453),  INT64_C( 7908633806112477453),  INT64_C( 7908633806112477453) } },
    { UINT8_C(140),
      -INT64_C( 5851986927596961173),
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 5851986927596961173), -INT64_C( 5851986927596961173) } },
    { UINT8_C( 65),
       INT64_C( 4548153536634956701),
      {  INT64_C( 4548153536634956701),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 12),
       INT64_C( 4758343015648900640),
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 4758343015648900640),  INT64_C( 4758343015648900640) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    int64_t a = test_vec[i].a;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_set1_epi64(k, a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_set1_epi64");
    easysimd_assert_m256i_i64(r, ==, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    int64_t a = easysimd_test_codegen_random_i64();
    easysimd__m256i r = easysimd_mm256_maskz_set1_epi64(k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_set1_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a;
    const easysimd_float32 r[16];
  } test_vec[] = {
    { EASYSIMD_FLOAT32_C(  -130.28),
      { EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28),
        EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28),
        EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28),
        EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28), EASYSIMD_FLOAT32_C(  -130.28) } },
    { EASYSIMD_FLOAT32_C(   996.56),
      { EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56),
        EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56),
        EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56),
        EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56), EASYSIMD_FLOAT32_C(   996.56) } },
    { EASYSIMD_FLOAT32_C(  -437.56),
      { EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56),
        EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56),
        EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56),
        EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56), EASYSIMD_FLOAT32_C(  -437.56) } },
    { EASYSIMD_FLOAT32_C(  -653.34),
      { EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34),
        EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34),
        EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34),
        EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34), EASYSIMD_FLOAT32_C(  -653.34) } },
    { EASYSIMD_FLOAT32_C(  -547.09),
      { EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09),
        EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09),
        EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09),
        EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09), EASYSIMD_FLOAT32_C(  -547.09) } },
    { EASYSIMD_FLOAT32_C(  -670.08),
      { EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08),
        EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08),
        EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08),
        EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08), EASYSIMD_FLOAT32_C(  -670.08) } },
    { EASYSIMD_FLOAT32_C(  -380.10),
      { EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10),
        EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10),
        EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10),
        EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10), EASYSIMD_FLOAT32_C(  -380.10) } },
    { EASYSIMD_FLOAT32_C(   -89.44),
      { EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44),
        EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44),
        EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44),
        EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44), EASYSIMD_FLOAT32_C(   -89.44) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32 a = test_vec[i].a;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_set1_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_set1_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32 a = easysimd_test_codegen_random_f32(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_set1_ps(a);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_set1_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd_float64 a;
    easysimd__m512d r;
  } test_vec[8] = {
    { EASYSIMD_FLOAT64_C( -426.34),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -426.34), EASYSIMD_FLOAT64_C( -426.34),
                         EASYSIMD_FLOAT64_C( -426.34), EASYSIMD_FLOAT64_C( -426.34),
                         EASYSIMD_FLOAT64_C( -426.34), EASYSIMD_FLOAT64_C( -426.34),
                         EASYSIMD_FLOAT64_C( -426.34), EASYSIMD_FLOAT64_C( -426.34)) },
    { EASYSIMD_FLOAT64_C(  122.65),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  122.65), EASYSIMD_FLOAT64_C(  122.65),
                         EASYSIMD_FLOAT64_C(  122.65), EASYSIMD_FLOAT64_C(  122.65),
                         EASYSIMD_FLOAT64_C(  122.65), EASYSIMD_FLOAT64_C(  122.65),
                         EASYSIMD_FLOAT64_C(  122.65), EASYSIMD_FLOAT64_C(  122.65)) },
    { EASYSIMD_FLOAT64_C(  879.85),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  879.85), EASYSIMD_FLOAT64_C(  879.85),
                         EASYSIMD_FLOAT64_C(  879.85), EASYSIMD_FLOAT64_C(  879.85),
                         EASYSIMD_FLOAT64_C(  879.85), EASYSIMD_FLOAT64_C(  879.85),
                         EASYSIMD_FLOAT64_C(  879.85), EASYSIMD_FLOAT64_C(  879.85)) },
    { EASYSIMD_FLOAT64_C(  301.17),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  301.17), EASYSIMD_FLOAT64_C(  301.17),
                         EASYSIMD_FLOAT64_C(  301.17), EASYSIMD_FLOAT64_C(  301.17),
                         EASYSIMD_FLOAT64_C(  301.17), EASYSIMD_FLOAT64_C(  301.17),
                         EASYSIMD_FLOAT64_C(  301.17), EASYSIMD_FLOAT64_C(  301.17)) },
    { EASYSIMD_FLOAT64_C( -341.96),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -341.96), EASYSIMD_FLOAT64_C( -341.96),
                         EASYSIMD_FLOAT64_C( -341.96), EASYSIMD_FLOAT64_C( -341.96),
                         EASYSIMD_FLOAT64_C( -341.96), EASYSIMD_FLOAT64_C( -341.96),
                         EASYSIMD_FLOAT64_C( -341.96), EASYSIMD_FLOAT64_C( -341.96)) },
    { EASYSIMD_FLOAT64_C( -854.60),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -854.60), EASYSIMD_FLOAT64_C( -854.60),
                         EASYSIMD_FLOAT64_C( -854.60), EASYSIMD_FLOAT64_C( -854.60),
                         EASYSIMD_FLOAT64_C( -854.60), EASYSIMD_FLOAT64_C( -854.60),
                         EASYSIMD_FLOAT64_C( -854.60), EASYSIMD_FLOAT64_C( -854.60)) },
    { EASYSIMD_FLOAT64_C(  711.48),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  711.48), EASYSIMD_FLOAT64_C(  711.48),
                         EASYSIMD_FLOAT64_C(  711.48), EASYSIMD_FLOAT64_C(  711.48),
                         EASYSIMD_FLOAT64_C(  711.48), EASYSIMD_FLOAT64_C(  711.48),
                         EASYSIMD_FLOAT64_C(  711.48), EASYSIMD_FLOAT64_C(  711.48)) },
    { EASYSIMD_FLOAT64_C( -146.85),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -146.85), EASYSIMD_FLOAT64_C( -146.85),
                         EASYSIMD_FLOAT64_C( -146.85), EASYSIMD_FLOAT64_C( -146.85),
                         EASYSIMD_FLOAT64_C( -146.85), EASYSIMD_FLOAT64_C( -146.85),
                         EASYSIMD_FLOAT64_C( -146.85), EASYSIMD_FLOAT64_C( -146.85)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd_float64 a = test_vec[i].a;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_set1_pd(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_set1_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_set1_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    int8_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    {   15,
      easysimd_mm512_set_epi8(INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15),
                           INT8_C(  15), INT8_C(  15), INT8_C(  15), INT8_C(  15)) },
    {  124,
      easysimd_mm512_set_epi8(INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124),
                           INT8_C( 124), INT8_C( 124), INT8_C( 124), INT8_C( 124)) },
    {  -93,
      easysimd_mm512_set_epi8(INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93),
                           INT8_C( -93), INT8_C( -93), INT8_C( -93), INT8_C( -93)) },
    {  121,
      easysimd_mm512_set_epi8(INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121),
                           INT8_C( 121), INT8_C( 121), INT8_C( 121), INT8_C( 121)) },
    {  117,
      easysimd_mm512_set_epi8(INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117),
                           INT8_C( 117), INT8_C( 117), INT8_C( 117), INT8_C( 117)) },
    {   93,
      easysimd_mm512_set_epi8(INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93),
                           INT8_C(  93), INT8_C(  93), INT8_C(  93), INT8_C(  93)) },
    {   88,
      easysimd_mm512_set_epi8(INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88),
                           INT8_C(  88), INT8_C(  88), INT8_C(  88), INT8_C(  88)) },
    {  -73,
      easysimd_mm512_set_epi8(INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73)) }
  };


  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int8_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_set1_epi8(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_set1_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_set1_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask64 k;
    int8_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi8(INT8_C(  80), INT8_C(  13), INT8_C( -86), INT8_C( 103),
                           INT8_C(  30), INT8_C(  88), INT8_C( -63), INT8_C( -16),
                           INT8_C( -68), INT8_C( -20), INT8_C(  48), INT8_C( -36),
                           INT8_C( -97), INT8_C(-103), INT8_C(-104), INT8_C( -61),
                           INT8_C(-122), INT8_C( -83), INT8_C(  -3), INT8_C(-115),
                           INT8_C(  29), INT8_C(-112), INT8_C( 118), INT8_C(  53),
                           INT8_C(-107), INT8_C(-126), INT8_C(  41), INT8_C(-117),
                           INT8_C(  -4), INT8_C( -72), INT8_C(  -9), INT8_C(   2),
                           INT8_C(  10), INT8_C( -61), INT8_C( 116), INT8_C(   1),
                           INT8_C(  35), INT8_C( -78), INT8_C(  17), INT8_C( -82),
                           INT8_C( -14), INT8_C( 120), INT8_C( 120), INT8_C(  33),
                           INT8_C(  97), INT8_C(   4), INT8_C(-104), INT8_C(  67),
                           INT8_C( -86), INT8_C( -90), INT8_C( -95), INT8_C(  51),
                           INT8_C( -83), INT8_C(-120), INT8_C( 123), INT8_C(  -4),
                           INT8_C(  51), INT8_C( -66), INT8_C( -91), INT8_C(  51),
                           INT8_C(  -1), INT8_C(  32), INT8_C(  30), INT8_C(  92)),
      UINT64_C(12701675613368776088),
      INT8_C( -94),
      easysimd_mm512_set_epi8(INT8_C( -94), INT8_C(  13), INT8_C( -94), INT8_C( -94),
                           INT8_C(  30), INT8_C(  88), INT8_C( -63), INT8_C( -16),
                           INT8_C( -68), INT8_C( -94), INT8_C(  48), INT8_C( -36),
                           INT8_C( -97), INT8_C( -94), INT8_C(-104), INT8_C( -94),
                           INT8_C(-122), INT8_C( -94), INT8_C( -94), INT8_C(-115),
                           INT8_C( -94), INT8_C(-112), INT8_C( -94), INT8_C(  53),
                           INT8_C( -94), INT8_C(-126), INT8_C( -94), INT8_C(-117),
                           INT8_C( -94), INT8_C( -94), INT8_C(  -9), INT8_C(   2),
                           INT8_C( -94), INT8_C( -61), INT8_C( 116), INT8_C(   1),
                           INT8_C( -94), INT8_C( -78), INT8_C( -94), INT8_C( -94),
                           INT8_C( -14), INT8_C( 120), INT8_C( 120), INT8_C( -94),
                           INT8_C(  97), INT8_C( -94), INT8_C( -94), INT8_C(  67),
                           INT8_C( -86), INT8_C( -94), INT8_C( -94), INT8_C( -94),
                           INT8_C( -83), INT8_C( -94), INT8_C( 123), INT8_C( -94),
                           INT8_C( -94), INT8_C( -66), INT8_C( -91), INT8_C( -94),
                           INT8_C( -94), INT8_C(  32), INT8_C(  30), INT8_C(  92)) },
    { easysimd_mm512_set_epi8(INT8_C( -64), INT8_C( -80), INT8_C(  33), INT8_C(  -9),
                           INT8_C(   3), INT8_C(  93), INT8_C(  13), INT8_C( -28),
                           INT8_C(  79), INT8_C(  10), INT8_C( -42), INT8_C(-127),
                           INT8_C( 114), INT8_C(  78), INT8_C(  61), INT8_C(  67),
                           INT8_C(  95), INT8_C(  14), INT8_C(  28), INT8_C(  56),
                           INT8_C(  43), INT8_C( -20), INT8_C( -77), INT8_C(  83),
                           INT8_C( -68), INT8_C(  87), INT8_C( -96), INT8_C(  13),
                           INT8_C(  40), INT8_C( 107), INT8_C( -63), INT8_C(  -1),
                           INT8_C(  77), INT8_C(  21), INT8_C( -46), INT8_C( -12),
                           INT8_C(  42), INT8_C(  69), INT8_C(  51), INT8_C(  11),
                           INT8_C(-120), INT8_C(  65), INT8_C( -70), INT8_C( -19),
                           INT8_C( -95), INT8_C(  43), INT8_C(  -2), INT8_C( -62),
                           INT8_C( -16), INT8_C(  28), INT8_C(  29), INT8_C( -11),
                           INT8_C(  17), INT8_C( -18), INT8_C( 105), INT8_C(-119),
                           INT8_C(  60), INT8_C( 120), INT8_C(  38), INT8_C( -41),
                           INT8_C(  20), INT8_C( -30), INT8_C(  15), INT8_C( 112)),
      UINT64_C(15052494645983188959),
      INT8_C( -73),
      easysimd_mm512_set_epi8(INT8_C( -73), INT8_C( -73), INT8_C(  33), INT8_C( -73),
                           INT8_C(   3), INT8_C(  93), INT8_C(  13), INT8_C( -28),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C(-127),
                           INT8_C( 114), INT8_C( -73), INT8_C(  61), INT8_C( -73),
                           INT8_C(  95), INT8_C(  14), INT8_C( -73), INT8_C( -73),
                           INT8_C(  43), INT8_C( -73), INT8_C( -77), INT8_C(  83),
                           INT8_C( -68), INT8_C(  87), INT8_C( -73), INT8_C(  13),
                           INT8_C(  40), INT8_C( 107), INT8_C( -63), INT8_C( -73),
                           INT8_C(  77), INT8_C(  21), INT8_C( -46), INT8_C( -12),
                           INT8_C(  42), INT8_C( -73), INT8_C(  51), INT8_C( -73),
                           INT8_C(-120), INT8_C( -73), INT8_C( -70), INT8_C( -19),
                           INT8_C( -73), INT8_C( -73), INT8_C(  -2), INT8_C( -62),
                           INT8_C( -16), INT8_C(  28), INT8_C(  29), INT8_C( -73),
                           INT8_C(  17), INT8_C( -73), INT8_C( -73), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C(  38), INT8_C( -73),
                           INT8_C( -73), INT8_C( -73), INT8_C( -73), INT8_C( -73)) },
    { easysimd_mm512_set_epi8(INT8_C( 107), INT8_C( 126), INT8_C( -33), INT8_C(  83),
                           INT8_C(  46), INT8_C(  62), INT8_C( -81), INT8_C(  33),
                           INT8_C( -68), INT8_C(-126), INT8_C( -41), INT8_C( 125),
                           INT8_C( -96), INT8_C( -20), INT8_C(  62), INT8_C( -19),
                           INT8_C(  29), INT8_C( -96), INT8_C(  68), INT8_C( 119),
                           INT8_C( -36), INT8_C( -62), INT8_C( -27), INT8_C(-112),
                           INT8_C(-123), INT8_C(  55), INT8_C(-119), INT8_C(  -4),
                           INT8_C(  58), INT8_C(  28), INT8_C( -84), INT8_C( -38),
                           INT8_C(   1), INT8_C( -25), INT8_C( 107), INT8_C( -63),
                           INT8_C( -86), INT8_C(  88), INT8_C(  36), INT8_C(  53),
                           INT8_C( 109), INT8_C( -36), INT8_C( -70), INT8_C(-125),
                           INT8_C(  -3), INT8_C(-109), INT8_C( 121), INT8_C( -63),
                           INT8_C( 113), INT8_C( -92), INT8_C(  -4), INT8_C(-105),
                           INT8_C( -65), INT8_C(  26), INT8_C( -36), INT8_C(  87),
                           INT8_C(-101), INT8_C( -70), INT8_C(  -3), INT8_C(  26),
                           INT8_C( -88), INT8_C( -51), INT8_C(-123), INT8_C(  93)),
      UINT64_C( 2985661334514035835),
      INT8_C( 111),
      easysimd_mm512_set_epi8(INT8_C( 107), INT8_C( 126), INT8_C( 111), INT8_C(  83),
                           INT8_C( 111), INT8_C(  62), INT8_C( -81), INT8_C( 111),
                           INT8_C( -68), INT8_C( 111), INT8_C( 111), INT8_C( 125),
                           INT8_C( 111), INT8_C( 111), INT8_C( 111), INT8_C( 111),
                           INT8_C(  29), INT8_C( -96), INT8_C( 111), INT8_C( 111),
                           INT8_C( -36), INT8_C( -62), INT8_C( 111), INT8_C( 111),
                           INT8_C(-123), INT8_C(  55), INT8_C( 111), INT8_C(  -4),
                           INT8_C( 111), INT8_C(  28), INT8_C( 111), INT8_C( -38),
                           INT8_C(   1), INT8_C( -25), INT8_C( 111), INT8_C( 111),
                           INT8_C( 111), INT8_C( 111), INT8_C( 111), INT8_C( 111),
                           INT8_C( 109), INT8_C( 111), INT8_C( -70), INT8_C( 111),
                           INT8_C(  -3), INT8_C(-109), INT8_C( 111), INT8_C( 111),
                           INT8_C( 111), INT8_C( 111), INT8_C( 111), INT8_C(-105),
                           INT8_C( 111), INT8_C( 111), INT8_C( -36), INT8_C(  87),
                           INT8_C(-101), INT8_C( 111), INT8_C( 111), INT8_C( 111),
                           INT8_C( 111), INT8_C( -51), INT8_C( 111), INT8_C( 111)) },
    { easysimd_mm512_set_epi8(INT8_C( -63), INT8_C(  92), INT8_C( -41), INT8_C( -80),
                           INT8_C(-101), INT8_C(  86), INT8_C(  45), INT8_C(  45),
                           INT8_C( -41), INT8_C(-113), INT8_C( -17), INT8_C(-101),
                           INT8_C(-113), INT8_C( -69), INT8_C(  73), INT8_C(-124),
                           INT8_C(  90), INT8_C(-118), INT8_C(  31), INT8_C(-124),
                           INT8_C( -88), INT8_C(-116), INT8_C(   8), INT8_C( -37),
                           INT8_C( -41), INT8_C(  93), INT8_C( -86), INT8_C(  61),
                           INT8_C( -70), INT8_C( -88), INT8_C(  44), INT8_C( -34),
                           INT8_C( -21), INT8_C(-121), INT8_C(-124), INT8_C(-114),
                           INT8_C(  73), INT8_C(  92), INT8_C( -92), INT8_C(-115),
                           INT8_C(   6), INT8_C(-120), INT8_C(  89), INT8_C(-102),
                           INT8_C( -43), INT8_C(  33), INT8_C(  15), INT8_C(  -6),
                           INT8_C(-105), INT8_C(  66), INT8_C( -60), INT8_C(  54),
                           INT8_C( -95), INT8_C(  49), INT8_C(   1), INT8_C( 118),
                           INT8_C( -33), INT8_C( -35), INT8_C( -34), INT8_C( -10),
                           INT8_C( -70), INT8_C(  74), INT8_C( -10), INT8_C(  97)),
      UINT64_C(12556192675989742329),
      INT8_C(-120),
      easysimd_mm512_set_epi8(INT8_C(-120), INT8_C(  92), INT8_C(-120), INT8_C( -80),
                           INT8_C(-120), INT8_C(-120), INT8_C(-120), INT8_C(  45),
                           INT8_C( -41), INT8_C(-120), INT8_C( -17), INT8_C(-101),
                           INT8_C(-113), INT8_C( -69), INT8_C(  73), INT8_C(-124),
                           INT8_C(-120), INT8_C(-118), INT8_C(  31), INT8_C(-124),
                           INT8_C(-120), INT8_C(-120), INT8_C(-120), INT8_C( -37),
                           INT8_C(-120), INT8_C(  93), INT8_C(-120), INT8_C(-120),
                           INT8_C( -70), INT8_C(-120), INT8_C(-120), INT8_C( -34),
                           INT8_C(-120), INT8_C(-121), INT8_C(-124), INT8_C(-120),
                           INT8_C(  73), INT8_C(-120), INT8_C(-120), INT8_C(-120),
                           INT8_C(-120), INT8_C(-120), INT8_C(-120), INT8_C(-120),
                           INT8_C(-120), INT8_C(  33), INT8_C(-120), INT8_C(  -6),
                           INT8_C(-120), INT8_C(-120), INT8_C(-120), INT8_C(  54),
                           INT8_C( -95), INT8_C(  49), INT8_C(-120), INT8_C( 118),
                           INT8_C(-120), INT8_C(-120), INT8_C(-120), INT8_C(-120),
                           INT8_C(-120), INT8_C(  74), INT8_C( -10), INT8_C(-120)) },
    { easysimd_mm512_set_epi8(INT8_C(  21), INT8_C(  17), INT8_C(  22), INT8_C(-115),
                           INT8_C( 101), INT8_C(  -2), INT8_C( -32), INT8_C( -27),
                           INT8_C( -14), INT8_C(  47), INT8_C( 110), INT8_C( -88),
                           INT8_C(  23), INT8_C( -87), INT8_C( -20), INT8_C( 115),
                           INT8_C( 108), INT8_C( -54), INT8_C(-105), INT8_C( -94),
                           INT8_C(  96), INT8_C(-110), INT8_C( -87), INT8_C( 119),
                           INT8_C( 110), INT8_C( -13), INT8_C(  53), INT8_C( -27),
                           INT8_C( -59), INT8_C(  57), INT8_C( -46), INT8_C( -24),
                           INT8_C(  35), INT8_C(  26), INT8_C( 124), INT8_C( -28),
                           INT8_C( -68), INT8_C( -57), INT8_C(  75), INT8_C( -25),
                           INT8_C(-112), INT8_C( 112), INT8_C( 123), INT8_C(-108),
                           INT8_C( 115), INT8_C(  -6), INT8_C(  43), INT8_C(  52),
                           INT8_C( -91), INT8_C( -17), INT8_C(  93), INT8_C(  -2),
                           INT8_C( 116), INT8_C( -51), INT8_C(  70), INT8_C(  98),
                           INT8_C( 104), INT8_C( -69), INT8_C(-102), INT8_C(  77),
                           INT8_C(  82), INT8_C( 125), INT8_C(  42), INT8_C(  83)),
      UINT64_C(12090133344763257330),
      INT8_C(  55),
      easysimd_mm512_set_epi8(INT8_C(  55), INT8_C(  17), INT8_C(  55), INT8_C(-115),
                           INT8_C( 101), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C( 110), INT8_C( -88),
                           INT8_C(  55), INT8_C( -87), INT8_C( -20), INT8_C( 115),
                           INT8_C(  55), INT8_C(  55), INT8_C(-105), INT8_C( -94),
                           INT8_C(  55), INT8_C(-110), INT8_C( -87), INT8_C( 119),
                           INT8_C( 110), INT8_C( -13), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  57), INT8_C( -46), INT8_C( -24),
                           INT8_C(  55), INT8_C(  26), INT8_C( 124), INT8_C( -28),
                           INT8_C( -68), INT8_C(  55), INT8_C(  75), INT8_C(  55),
                           INT8_C(  55), INT8_C( 112), INT8_C(  55), INT8_C(  55),
                           INT8_C(  55), INT8_C(  -6), INT8_C(  43), INT8_C(  52),
                           INT8_C( -91), INT8_C( -17), INT8_C(  55), INT8_C(  -2),
                           INT8_C( 116), INT8_C( -51), INT8_C(  70), INT8_C(  55),
                           INT8_C(  55), INT8_C(  55), INT8_C(  55), INT8_C(  55),
                           INT8_C(  82), INT8_C( 125), INT8_C(  55), INT8_C(  83)) },
    { easysimd_mm512_set_epi8(INT8_C(-124), INT8_C( -37), INT8_C( -61), INT8_C( -35),
                           INT8_C( -22), INT8_C( -85), INT8_C(-117), INT8_C(-105),
                           INT8_C(  99), INT8_C( -62), INT8_C( 102), INT8_C( -31),
                           INT8_C(  82), INT8_C(  39), INT8_C(  49), INT8_C(  43),
                           INT8_C(  21), INT8_C(  16), INT8_C(  12), INT8_C(-125),
                           INT8_C(   2), INT8_C(-106), INT8_C(  -4), INT8_C( 100),
                           INT8_C( -12), INT8_C(  30), INT8_C( -39), INT8_C( -37),
                           INT8_C(  92), INT8_C( -43), INT8_C(  33), INT8_C(-124),
                           INT8_C(  48), INT8_C(   4), INT8_C(  31), INT8_C(  78),
                           INT8_C(-113), INT8_C( 115), INT8_C( 116), INT8_C( -62),
                           INT8_C(-109), INT8_C( -66), INT8_C(  43), INT8_C(-118),
                           INT8_C(-105), INT8_C( -11), INT8_C( 100), INT8_C(  41),
                           INT8_C(-104), INT8_C(-114), INT8_C(-105), INT8_C(  88),
                           INT8_C( -33), INT8_C(  -8), INT8_C(  41), INT8_C(  16),
                           INT8_C(   4), INT8_C(  89), INT8_C(  66), INT8_C(  27),
                           INT8_C( -63), INT8_C(  30), INT8_C( -95), INT8_C(  33)),
      UINT64_C(13436704833767296949),
      INT8_C(  18),
      easysimd_mm512_set_epi8(INT8_C(  18), INT8_C( -37), INT8_C(  18), INT8_C(  18),
                           INT8_C(  18), INT8_C( -85), INT8_C(  18), INT8_C(-105),
                           INT8_C(  99), INT8_C(  18), INT8_C(  18), INT8_C(  18),
                           INT8_C(  18), INT8_C(  39), INT8_C(  49), INT8_C(  43),
                           INT8_C(  18), INT8_C(  18), INT8_C(  12), INT8_C(-125),
                           INT8_C(   2), INT8_C(-106), INT8_C(  18), INT8_C(  18),
                           INT8_C(  18), INT8_C(  18), INT8_C( -39), INT8_C(  18),
                           INT8_C(  18), INT8_C( -43), INT8_C(  18), INT8_C(  18),
                           INT8_C(  48), INT8_C(   4), INT8_C(  31), INT8_C(  78),
                           INT8_C(  18), INT8_C( 115), INT8_C( 116), INT8_C( -62),
                           INT8_C(-109), INT8_C(  18), INT8_C(  18), INT8_C(  18),
                           INT8_C(  18), INT8_C( -11), INT8_C( 100), INT8_C(  18),
                           INT8_C(-104), INT8_C(  18), INT8_C(-105), INT8_C(  88),
                           INT8_C(  18), INT8_C(  18), INT8_C(  18), INT8_C(  18),
                           INT8_C(  18), INT8_C(  89), INT8_C(  18), INT8_C(  18),
                           INT8_C( -63), INT8_C(  18), INT8_C( -95), INT8_C(  18)) },
    { easysimd_mm512_set_epi8(INT8_C( -30), INT8_C( 101), INT8_C(  64), INT8_C( 107),
                           INT8_C( -34), INT8_C( -67), INT8_C( -96), INT8_C(  35),
                           INT8_C( 117), INT8_C(  76), INT8_C( 106), INT8_C( -82),
                           INT8_C( -48), INT8_C(  63), INT8_C(  11), INT8_C(  22),
                           INT8_C(  41), INT8_C(  95), INT8_C(-123), INT8_C( -90),
                           INT8_C(  67), INT8_C( -76), INT8_C(-105), INT8_C(  -7),
                           INT8_C( 115), INT8_C( 121), INT8_C( -52), INT8_C( -95),
                           INT8_C(-101), INT8_C(  64), INT8_C( -67), INT8_C( 107),
                           INT8_C(-104), INT8_C(  56), INT8_C(  89), INT8_C( -95),
                           INT8_C(  21), INT8_C( -42), INT8_C( -75), INT8_C(  45),
                           INT8_C( -86), INT8_C(  32), INT8_C(  27), INT8_C(-119),
                           INT8_C( -68), INT8_C(   5), INT8_C( -78), INT8_C( -36),
                           INT8_C( 125), INT8_C( 117), INT8_C( -63), INT8_C( -68),
                           INT8_C( -45), INT8_C( -77), INT8_C(   6), INT8_C(  68),
                           INT8_C(  79), INT8_C( -92), INT8_C(  67), INT8_C(  61),
                           INT8_C(  42), INT8_C(  26), INT8_C(-117), INT8_C( -55)),
      UINT64_C(14020412538477965079),
      INT8_C( -46),
      easysimd_mm512_set_epi8(INT8_C( -46), INT8_C( -46), INT8_C(  64), INT8_C( 107),
                           INT8_C( -34), INT8_C( -67), INT8_C( -46), INT8_C(  35),
                           INT8_C( -46), INT8_C(  76), INT8_C( 106), INT8_C( -46),
                           INT8_C( -48), INT8_C(  63), INT8_C( -46), INT8_C(  22),
                           INT8_C( -46), INT8_C(  95), INT8_C(-123), INT8_C( -90),
                           INT8_C(  67), INT8_C( -76), INT8_C( -46), INT8_C(  -7),
                           INT8_C( -46), INT8_C( -46), INT8_C( -46), INT8_C( -95),
                           INT8_C( -46), INT8_C(  64), INT8_C( -46), INT8_C( -46),
                           INT8_C( -46), INT8_C(  56), INT8_C( -46), INT8_C( -95),
                           INT8_C(  21), INT8_C( -42), INT8_C( -75), INT8_C(  45),
                           INT8_C( -86), INT8_C( -46), INT8_C( -46), INT8_C(-119),
                           INT8_C( -46), INT8_C(   5), INT8_C( -78), INT8_C( -46),
                           INT8_C( 125), INT8_C( -46), INT8_C( -46), INT8_C( -68),
                           INT8_C( -46), INT8_C( -46), INT8_C( -46), INT8_C( -46),
                           INT8_C(  79), INT8_C( -92), INT8_C(  67), INT8_C( -46),
                           INT8_C(  42), INT8_C( -46), INT8_C( -46), INT8_C( -46)) },
    { easysimd_mm512_set_epi8(INT8_C( -83), INT8_C( -73), INT8_C( -22), INT8_C(  98),
                           INT8_C( 126), INT8_C(  41), INT8_C( -28), INT8_C( 126),
                           INT8_C( -75), INT8_C(  91), INT8_C( -33), INT8_C( 103),
                           INT8_C( -63), INT8_C(  62), INT8_C(  83), INT8_C(   4),
                           INT8_C(  65), INT8_C( -22), INT8_C( 107), INT8_C(   8),
                           INT8_C(  31), INT8_C(-111), INT8_C(-114), INT8_C(-118),
                           INT8_C(   2), INT8_C(  76), INT8_C(  19), INT8_C( 127),
                           INT8_C( -37), INT8_C( -41), INT8_C(  91), INT8_C( -64),
                           INT8_C(-105), INT8_C( 127), INT8_C(-121), INT8_C(  84),
                           INT8_C( 124), INT8_C(  50), INT8_C( -86), INT8_C(-101),
                           INT8_C( -82), INT8_C( 121), INT8_C(  18), INT8_C( -17),
                           INT8_C( -55), INT8_C(-102), INT8_C( -81), INT8_C( -54),
                           INT8_C( -56), INT8_C(  -2), INT8_C( -68), INT8_C( 105),
                           INT8_C( -48), INT8_C( -90), INT8_C( -46), INT8_C(  63),
                           INT8_C( 126), INT8_C( -93), INT8_C(  46), INT8_C(-114),
                           INT8_C(  58), INT8_C( 110), INT8_C( 102), INT8_C( -93)),
      UINT64_C(14839809536761107867),
      INT8_C( 106),
      easysimd_mm512_set_epi8(INT8_C( 106), INT8_C( 106), INT8_C( -22), INT8_C(  98),
                           INT8_C( 106), INT8_C( 106), INT8_C( -28), INT8_C( 106),
                           INT8_C( 106), INT8_C( 106), INT8_C( 106), INT8_C( 106),
                           INT8_C( -63), INT8_C(  62), INT8_C(  83), INT8_C( 106),
                           INT8_C( 106), INT8_C( -22), INT8_C( 107), INT8_C( 106),
                           INT8_C( 106), INT8_C(-111), INT8_C(-114), INT8_C(-118),
                           INT8_C(   2), INT8_C(  76), INT8_C( 106), INT8_C( 127),
                           INT8_C( -37), INT8_C( 106), INT8_C( 106), INT8_C( -64),
                           INT8_C(-105), INT8_C( 127), INT8_C( 106), INT8_C(  84),
                           INT8_C( 124), INT8_C( 106), INT8_C( -86), INT8_C( 106),
                           INT8_C( -82), INT8_C( 121), INT8_C(  18), INT8_C( 106),
                           INT8_C( 106), INT8_C(-102), INT8_C( -81), INT8_C( 106),
                           INT8_C( 106), INT8_C(  -2), INT8_C( -68), INT8_C( 106),
                           INT8_C( -48), INT8_C( 106), INT8_C( -46), INT8_C( 106),
                           INT8_C( 106), INT8_C( -93), INT8_C(  46), INT8_C( 106),
                           INT8_C( 106), INT8_C( 110), INT8_C( 106), INT8_C( 106)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask64 k = test_vec[i].k;
    int8_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_set1_epi8(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_set1_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_set1_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask64 k;
    int8_t a;
    easysimd__m512i r;
  } test_vec[8] = {
   { UINT64_C( 2901368310709582274),
      INT8_C( -37),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C( -37), INT8_C(   0),
                           INT8_C( -37), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -37), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -37), INT8_C( -37),
                           INT8_C( -37), INT8_C(   0), INT8_C( -37), INT8_C( -37),
                           INT8_C( -37), INT8_C(   0), INT8_C( -37), INT8_C( -37),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -37),
                           INT8_C( -37), INT8_C(   0), INT8_C( -37), INT8_C( -37),
                           INT8_C(   0), INT8_C( -37), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -37), INT8_C(   0),
                           INT8_C(   0), INT8_C( -37), INT8_C( -37), INT8_C(   0),
                           INT8_C( -37), INT8_C(   0), INT8_C( -37), INT8_C( -37),
                           INT8_C(   0), INT8_C( -37), INT8_C( -37), INT8_C( -37),
                           INT8_C( -37), INT8_C(   0), INT8_C(   0), INT8_C( -37),
                           INT8_C( -37), INT8_C( -37), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -37), INT8_C(   0)) },
    { UINT64_C(15800639674747260058),
      INT8_C(  63),
      easysimd_mm512_set_epi8(INT8_C(  63), INT8_C(  63), INT8_C(   0), INT8_C(  63),
                           INT8_C(  63), INT8_C(   0), INT8_C(  63), INT8_C(  63),
                           INT8_C(   0), INT8_C(  63), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  63), INT8_C(  63), INT8_C(  63),
                           INT8_C(   0), INT8_C(   0), INT8_C(  63), INT8_C(   0),
                           INT8_C(   0), INT8_C(  63), INT8_C(  63), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  63),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  63),
                           INT8_C(  63), INT8_C(   0), INT8_C(   0), INT8_C(  63),
                           INT8_C(  63), INT8_C(  63), INT8_C(   0), INT8_C(  63),
                           INT8_C(   0), INT8_C(  63), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  63),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  63), INT8_C(  63), INT8_C(   0), INT8_C(   0),
                           INT8_C(  63), INT8_C(   0), INT8_C(   0), INT8_C(  63),
                           INT8_C(  63), INT8_C(   0), INT8_C(  63), INT8_C(   0)) },
    { UINT64_C(12860739080443979541),
      INT8_C(  53),
      easysimd_mm512_set_epi8(INT8_C(  53), INT8_C(   0), INT8_C(  53), INT8_C(  53),
                           INT8_C(   0), INT8_C(   0), INT8_C(  53), INT8_C(   0),
                           INT8_C(   0), INT8_C(  53), INT8_C(  53), INT8_C(  53),
                           INT8_C(  53), INT8_C(   0), INT8_C(  53), INT8_C(   0),
                           INT8_C(  53), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  53), INT8_C(  53), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  53), INT8_C(  53), INT8_C(   0), INT8_C(  53),
                           INT8_C(  53), INT8_C(  53), INT8_C(  53), INT8_C(  53),
                           INT8_C(   0), INT8_C(  53), INT8_C(  53), INT8_C(   0),
                           INT8_C(  53), INT8_C(  53), INT8_C(   0), INT8_C(  53),
                           INT8_C(  53), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  53), INT8_C(  53), INT8_C(  53), INT8_C(   0),
                           INT8_C(   0), INT8_C(  53), INT8_C(  53), INT8_C(  53),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  53),
                           INT8_C(   0), INT8_C(  53), INT8_C(   0), INT8_C(  53)) },
    { UINT64_C( 2595884503750725802),
      INT8_C(  78),
      easysimd_mm512_set_epi8(INT8_C(   0), INT8_C(   0), INT8_C(  78), INT8_C(   0),
                           INT8_C(   0), INT8_C(  78), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  78), INT8_C(  78), INT8_C(   0),
                           INT8_C(   0), INT8_C(  78), INT8_C(  78), INT8_C(   0),
                           INT8_C(  78), INT8_C(  78), INT8_C(  78), INT8_C(  78),
                           INT8_C(   0), INT8_C(   0), INT8_C(  78), INT8_C(  78),
                           INT8_C(   0), INT8_C(   0), INT8_C(  78), INT8_C(  78),
                           INT8_C(  78), INT8_C(  78), INT8_C(   0), INT8_C(  78),
                           INT8_C(  78), INT8_C(  78), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  78), INT8_C(  78), INT8_C(   0), INT8_C(  78),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  78), INT8_C(   0), INT8_C(  78), INT8_C(   0),
                           INT8_C(  78), INT8_C(   0), INT8_C(  78), INT8_C(   0)) },
    { UINT64_C(13286373173549182748),
      INT8_C( -67),
      easysimd_mm512_set_epi8(INT8_C( -67), INT8_C(   0), INT8_C( -67), INT8_C( -67),
                           INT8_C( -67), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -67), INT8_C( -67), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -67), INT8_C(   0),
                           INT8_C( -67), INT8_C(   0), INT8_C( -67), INT8_C(   0),
                           INT8_C( -67), INT8_C( -67), INT8_C( -67), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -67),
                           INT8_C( -67), INT8_C( -67), INT8_C(   0), INT8_C( -67),
                           INT8_C(   0), INT8_C(   0), INT8_C( -67), INT8_C(   0),
                           INT8_C( -67), INT8_C(   0), INT8_C( -67), INT8_C(   0),
                           INT8_C( -67), INT8_C( -67), INT8_C(   0), INT8_C( -67),
                           INT8_C( -67), INT8_C(   0), INT8_C( -67), INT8_C(   0),
                           INT8_C( -67), INT8_C( -67), INT8_C( -67), INT8_C( -67),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C( -67),
                           INT8_C( -67), INT8_C( -67), INT8_C(   0), INT8_C(   0)) },
    { UINT64_C(16804997844821669286),
      INT8_C( -98),
      easysimd_mm512_set_epi8(INT8_C( -98), INT8_C( -98), INT8_C( -98), INT8_C(   0),
                           INT8_C( -98), INT8_C(   0), INT8_C(   0), INT8_C( -98),
                           INT8_C(   0), INT8_C(   0), INT8_C( -98), INT8_C( -98),
                           INT8_C(   0), INT8_C( -98), INT8_C( -98), INT8_C( -98),
                           INT8_C(   0), INT8_C( -98), INT8_C(   0), INT8_C( -98),
                           INT8_C( -98), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -98), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C( -98), INT8_C( -98), INT8_C(   0), INT8_C(   0),
                           INT8_C( -98), INT8_C( -98), INT8_C( -98), INT8_C( -98),
                           INT8_C( -98), INT8_C( -98), INT8_C(   0), INT8_C(   0),
                           INT8_C( -98), INT8_C( -98), INT8_C(   0), INT8_C(   0),
                           INT8_C( -98), INT8_C( -98), INT8_C( -98), INT8_C( -98),
                           INT8_C(   0), INT8_C( -98), INT8_C(   0), INT8_C( -98),
                           INT8_C( -98), INT8_C(   0), INT8_C( -98), INT8_C(   0),
                           INT8_C(   0), INT8_C( -98), INT8_C( -98), INT8_C(   0)) },
    { UINT64_C(14388383136321922859),
      INT8_C( -31),
      easysimd_mm512_set_epi8(INT8_C( -31), INT8_C( -31), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -31), INT8_C( -31), INT8_C( -31),
                           INT8_C( -31), INT8_C(   0), INT8_C( -31), INT8_C(   0),
                           INT8_C( -31), INT8_C( -31), INT8_C(   0), INT8_C( -31),
                           INT8_C( -31), INT8_C( -31), INT8_C(   0), INT8_C(   0),
                           INT8_C( -31), INT8_C( -31), INT8_C( -31), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C( -31), INT8_C( -31),
                           INT8_C( -31), INT8_C(   0), INT8_C( -31), INT8_C(   0),
                           INT8_C( -31), INT8_C( -31), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -31), INT8_C(   0), INT8_C(   0),
                           INT8_C( -31), INT8_C(   0), INT8_C( -31), INT8_C( -31),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C( -31), INT8_C( -31), INT8_C(   0),
                           INT8_C(   0), INT8_C( -31), INT8_C( -31), INT8_C( -31),
                           INT8_C(   0), INT8_C(   0), INT8_C( -31), INT8_C(   0),
                           INT8_C( -31), INT8_C(   0), INT8_C( -31), INT8_C( -31)) },
    { UINT64_C( 9693935732927043828),
      INT8_C(  57),
      easysimd_mm512_set_epi8(INT8_C(  57), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  57), INT8_C(  57), INT8_C(   0),
                           INT8_C(  57), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  57), INT8_C(  57), INT8_C(  57),
                           INT8_C(  57), INT8_C(  57), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(  57), INT8_C(  57), INT8_C(  57),
                           INT8_C(   0), INT8_C(   0), INT8_C(  57), INT8_C(  57),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(   0), INT8_C(   0), INT8_C(  57), INT8_C(  57),
                           INT8_C(  57), INT8_C(  57), INT8_C(   0), INT8_C(   0),
                           INT8_C(  57), INT8_C(   0), INT8_C(   0), INT8_C(  57),
                           INT8_C(   0), INT8_C(  57), INT8_C(   0), INT8_C(  57),
                           INT8_C(   0), INT8_C(   0), INT8_C(   0), INT8_C(  57),
                           INT8_C(  57), INT8_C(   0), INT8_C(   0), INT8_C(   0),
                           INT8_C(  57), INT8_C(  57), INT8_C(  57), INT8_C(  57),
                           INT8_C(   0), INT8_C(  57), INT8_C(   0), INT8_C(   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask64 k = test_vec[i].k;
    int8_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_set1_epi8(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_set1_epi8");
    easysimd_assert_m512i_i8(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_set1_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    int16_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    {   -334,
      easysimd_mm512_set_epi16(INT16_C(  -334), INT16_C(  -334), INT16_C(  -334), INT16_C(  -334),
                            INT16_C(  -334), INT16_C(  -334), INT16_C(  -334), INT16_C(  -334),
                            INT16_C(  -334), INT16_C(  -334), INT16_C(  -334), INT16_C(  -334),
                            INT16_C(  -334), INT16_C(  -334), INT16_C(  -334), INT16_C(  -334),
                            INT16_C(  -334), INT16_C(  -334), INT16_C(  -334), INT16_C(  -334),
                            INT16_C(  -334), INT16_C(  -334), INT16_C(  -334), INT16_C(  -334),
                            INT16_C(  -334), INT16_C(  -334), INT16_C(  -334), INT16_C(  -334),
                            INT16_C(  -334), INT16_C(  -334), INT16_C(  -334), INT16_C(  -334)) },
    {  27900,
      easysimd_mm512_set_epi16(INT16_C( 27900), INT16_C( 27900), INT16_C( 27900), INT16_C( 27900),
                            INT16_C( 27900), INT16_C( 27900), INT16_C( 27900), INT16_C( 27900),
                            INT16_C( 27900), INT16_C( 27900), INT16_C( 27900), INT16_C( 27900),
                            INT16_C( 27900), INT16_C( 27900), INT16_C( 27900), INT16_C( 27900),
                            INT16_C( 27900), INT16_C( 27900), INT16_C( 27900), INT16_C( 27900),
                            INT16_C( 27900), INT16_C( 27900), INT16_C( 27900), INT16_C( 27900),
                            INT16_C( 27900), INT16_C( 27900), INT16_C( 27900), INT16_C( 27900),
                            INT16_C( 27900), INT16_C( 27900), INT16_C( 27900), INT16_C( 27900)) },
    {   9352,
      easysimd_mm512_set_epi16(INT16_C(  9352), INT16_C(  9352), INT16_C(  9352), INT16_C(  9352),
                            INT16_C(  9352), INT16_C(  9352), INT16_C(  9352), INT16_C(  9352),
                            INT16_C(  9352), INT16_C(  9352), INT16_C(  9352), INT16_C(  9352),
                            INT16_C(  9352), INT16_C(  9352), INT16_C(  9352), INT16_C(  9352),
                            INT16_C(  9352), INT16_C(  9352), INT16_C(  9352), INT16_C(  9352),
                            INT16_C(  9352), INT16_C(  9352), INT16_C(  9352), INT16_C(  9352),
                            INT16_C(  9352), INT16_C(  9352), INT16_C(  9352), INT16_C(  9352),
                            INT16_C(  9352), INT16_C(  9352), INT16_C(  9352), INT16_C(  9352)) },
    { -21903,
      easysimd_mm512_set_epi16(INT16_C(-21903), INT16_C(-21903), INT16_C(-21903), INT16_C(-21903),
                            INT16_C(-21903), INT16_C(-21903), INT16_C(-21903), INT16_C(-21903),
                            INT16_C(-21903), INT16_C(-21903), INT16_C(-21903), INT16_C(-21903),
                            INT16_C(-21903), INT16_C(-21903), INT16_C(-21903), INT16_C(-21903),
                            INT16_C(-21903), INT16_C(-21903), INT16_C(-21903), INT16_C(-21903),
                            INT16_C(-21903), INT16_C(-21903), INT16_C(-21903), INT16_C(-21903),
                            INT16_C(-21903), INT16_C(-21903), INT16_C(-21903), INT16_C(-21903),
                            INT16_C(-21903), INT16_C(-21903), INT16_C(-21903), INT16_C(-21903)) },
    {  32371,
      easysimd_mm512_set_epi16(INT16_C( 32371), INT16_C( 32371), INT16_C( 32371), INT16_C( 32371),
                            INT16_C( 32371), INT16_C( 32371), INT16_C( 32371), INT16_C( 32371),
                            INT16_C( 32371), INT16_C( 32371), INT16_C( 32371), INT16_C( 32371),
                            INT16_C( 32371), INT16_C( 32371), INT16_C( 32371), INT16_C( 32371),
                            INT16_C( 32371), INT16_C( 32371), INT16_C( 32371), INT16_C( 32371),
                            INT16_C( 32371), INT16_C( 32371), INT16_C( 32371), INT16_C( 32371),
                            INT16_C( 32371), INT16_C( 32371), INT16_C( 32371), INT16_C( 32371),
                            INT16_C( 32371), INT16_C( 32371), INT16_C( 32371), INT16_C( 32371)) },
    {    -49,
      easysimd_mm512_set_epi16(INT16_C(   -49), INT16_C(   -49), INT16_C(   -49), INT16_C(   -49),
                            INT16_C(   -49), INT16_C(   -49), INT16_C(   -49), INT16_C(   -49),
                            INT16_C(   -49), INT16_C(   -49), INT16_C(   -49), INT16_C(   -49),
                            INT16_C(   -49), INT16_C(   -49), INT16_C(   -49), INT16_C(   -49),
                            INT16_C(   -49), INT16_C(   -49), INT16_C(   -49), INT16_C(   -49),
                            INT16_C(   -49), INT16_C(   -49), INT16_C(   -49), INT16_C(   -49),
                            INT16_C(   -49), INT16_C(   -49), INT16_C(   -49), INT16_C(   -49),
                            INT16_C(   -49), INT16_C(   -49), INT16_C(   -49), INT16_C(   -49)) },
    {  18491,
      easysimd_mm512_set_epi16(INT16_C( 18491), INT16_C( 18491), INT16_C( 18491), INT16_C( 18491),
                            INT16_C( 18491), INT16_C( 18491), INT16_C( 18491), INT16_C( 18491),
                            INT16_C( 18491), INT16_C( 18491), INT16_C( 18491), INT16_C( 18491),
                            INT16_C( 18491), INT16_C( 18491), INT16_C( 18491), INT16_C( 18491),
                            INT16_C( 18491), INT16_C( 18491), INT16_C( 18491), INT16_C( 18491),
                            INT16_C( 18491), INT16_C( 18491), INT16_C( 18491), INT16_C( 18491),
                            INT16_C( 18491), INT16_C( 18491), INT16_C( 18491), INT16_C( 18491),
                            INT16_C( 18491), INT16_C( 18491), INT16_C( 18491), INT16_C( 18491)) },
    {  25038,
      easysimd_mm512_set_epi16(INT16_C( 25038), INT16_C( 25038), INT16_C( 25038), INT16_C( 25038),
                            INT16_C( 25038), INT16_C( 25038), INT16_C( 25038), INT16_C( 25038),
                            INT16_C( 25038), INT16_C( 25038), INT16_C( 25038), INT16_C( 25038),
                            INT16_C( 25038), INT16_C( 25038), INT16_C( 25038), INT16_C( 25038),
                            INT16_C( 25038), INT16_C( 25038), INT16_C( 25038), INT16_C( 25038),
                            INT16_C( 25038), INT16_C( 25038), INT16_C( 25038), INT16_C( 25038),
                            INT16_C( 25038), INT16_C( 25038), INT16_C( 25038), INT16_C( 25038),
                            INT16_C( 25038), INT16_C( 25038), INT16_C( 25038), INT16_C( 25038)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int16_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_set1_epi16(a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_set1_epi16");
    easysimd_assert_m512i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_set1_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask32 k;
    int16_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi16(INT16_C(   874), INT16_C( 15357), INT16_C(  3602), INT16_C( 11090),
                            INT16_C( 31475), INT16_C( 20808), INT16_C(-26328), INT16_C(-21794),
                            INT16_C(-24829), INT16_C(-15530), INT16_C( -9785), INT16_C( 22806),
                            INT16_C( -6385), INT16_C(-26604), INT16_C(-15351), INT16_C(-18936),
                            INT16_C( 28985), INT16_C( 24045), INT16_C(-25535), INT16_C(-25436),
                            INT16_C(   749), INT16_C( 15517), INT16_C(-24369), INT16_C(-27864),
                            INT16_C(  6864), INT16_C( 16553), INT16_C(  -407), INT16_C(-28514),
                            INT16_C( -9423), INT16_C(-29018), INT16_C(-11420), INT16_C(-22112)),
      UINT32_C(1121120418),
      INT16_C(-24380),
      easysimd_mm512_set_epi16(INT16_C(   874), INT16_C(-24380), INT16_C(  3602), INT16_C( 11090),
                            INT16_C( 31475), INT16_C( 20808), INT16_C(-24380), INT16_C(-21794),
                            INT16_C(-24380), INT16_C(-24380), INT16_C( -9785), INT16_C(-24380),
                            INT16_C( -6385), INT16_C(-26604), INT16_C(-24380), INT16_C(-18936),
                            INT16_C(-24380), INT16_C(-24380), INT16_C(-24380), INT16_C(-24380),
                            INT16_C(   749), INT16_C( 15517), INT16_C(-24369), INT16_C(-27864),
                            INT16_C(-24380), INT16_C( 16553), INT16_C(-24380), INT16_C(-28514),
                            INT16_C( -9423), INT16_C(-29018), INT16_C(-24380), INT16_C(-22112)) },
    { easysimd_mm512_set_epi16(INT16_C( 21630), INT16_C(    53), INT16_C(-30787), INT16_C( 11298),
                            INT16_C( 13120), INT16_C(-15891), INT16_C( 20834), INT16_C(  5170),
                            INT16_C( 11237), INT16_C(-32025), INT16_C(  2036), INT16_C(-32146),
                            INT16_C(  6820), INT16_C( 29964), INT16_C(-20673), INT16_C( -6255),
                            INT16_C( 12677), INT16_C(  5934), INT16_C( 18392), INT16_C(-16008),
                            INT16_C( -6967), INT16_C(-23263), INT16_C( 28759), INT16_C(  4932),
                            INT16_C(-20928), INT16_C(-12287), INT16_C(-21100), INT16_C(-15604),
                            INT16_C(-25734), INT16_C(-27889), INT16_C( 22154), INT16_C( 16749)),
      UINT32_C( 442706120),
      INT16_C(-18045),
      easysimd_mm512_set_epi16(INT16_C( 21630), INT16_C(    53), INT16_C(-30787), INT16_C(-18045),
                            INT16_C(-18045), INT16_C(-15891), INT16_C(-18045), INT16_C(  5170),
                            INT16_C( 11237), INT16_C(-18045), INT16_C(-18045), INT16_C(-32146),
                            INT16_C(  6820), INT16_C( 29964), INT16_C(-18045), INT16_C(-18045),
                            INT16_C( 12677), INT16_C(  5934), INT16_C(-18045), INT16_C(-16008),
                            INT16_C(-18045), INT16_C(-23263), INT16_C( 28759), INT16_C(  4932),
                            INT16_C(-18045), INT16_C(-18045), INT16_C(-21100), INT16_C(-15604),
                            INT16_C(-18045), INT16_C(-27889), INT16_C( 22154), INT16_C( 16749)) },
    { easysimd_mm512_set_epi16(INT16_C(-12675), INT16_C(-13885), INT16_C( -4000), INT16_C( 31908),
                            INT16_C( 16178), INT16_C( -8662), INT16_C(-27877), INT16_C(-11427),
                            INT16_C(-10847), INT16_C(  7965), INT16_C(-13767), INT16_C( 14192),
                            INT16_C( -3024), INT16_C(-20651), INT16_C(  1677), INT16_C(-14378),
                            INT16_C( 13823), INT16_C(-21716), INT16_C(-14569), INT16_C( 19205),
                            INT16_C(-19335), INT16_C( 31769), INT16_C(-13133), INT16_C(-12032),
                            INT16_C(-27851), INT16_C(-12954), INT16_C(-30941), INT16_C( 26210),
                            INT16_C( 10250), INT16_C(-12883), INT16_C(-31618), INT16_C(  -328)),
      UINT32_C(3083705480),
      INT16_C(  4440),
      easysimd_mm512_set_epi16(INT16_C(  4440), INT16_C(-13885), INT16_C(  4440), INT16_C(  4440),
                            INT16_C( 16178), INT16_C(  4440), INT16_C(  4440), INT16_C(  4440),
                            INT16_C(  4440), INT16_C(  4440), INT16_C(-13767), INT16_C( 14192),
                            INT16_C(  4440), INT16_C(  4440), INT16_C(  1677), INT16_C(  4440),
                            INT16_C(  4440), INT16_C(-21716), INT16_C(-14569), INT16_C(  4440),
                            INT16_C(  4440), INT16_C(  4440), INT16_C(-13133), INT16_C(-12032),
                            INT16_C(  4440), INT16_C(-12954), INT16_C(-30941), INT16_C( 26210),
                            INT16_C(  4440), INT16_C(-12883), INT16_C(-31618), INT16_C(  -328)) },
    { easysimd_mm512_set_epi16(INT16_C(-23201), INT16_C(  4909), INT16_C(-10596), INT16_C( 25003),
                            INT16_C( 25193), INT16_C(-28193), INT16_C(  7484), INT16_C( 22842),
                            INT16_C( 12827), INT16_C(-21490), INT16_C(-19021), INT16_C( 17939),
                            INT16_C( 14187), INT16_C( 31294), INT16_C(-22999), INT16_C( 25206),
                            INT16_C(-22002), INT16_C( 23505), INT16_C(-20713), INT16_C( 22238),
                            INT16_C( 29284), INT16_C( 28054), INT16_C(-21727), INT16_C( 30369),
                            INT16_C( 19358), INT16_C(  -623), INT16_C(  2386), INT16_C(  9395),
                            INT16_C(-11819), INT16_C( 28599), INT16_C(-11863), INT16_C( -4500)),
      UINT32_C(1729799485),
      INT16_C(   -51),
      easysimd_mm512_set_epi16(INT16_C(-23201), INT16_C(   -51), INT16_C(   -51), INT16_C( 25003),
                            INT16_C( 25193), INT16_C(   -51), INT16_C(   -51), INT16_C(   -51),
                            INT16_C( 12827), INT16_C(-21490), INT16_C(-19021), INT16_C(   -51),
                            INT16_C(   -51), INT16_C( 31294), INT16_C(   -51), INT16_C( 25206),
                            INT16_C(   -51), INT16_C( 23505), INT16_C(   -51), INT16_C( 22238),
                            INT16_C( 29284), INT16_C(   -51), INT16_C(-21727), INT16_C(   -51),
                            INT16_C( 19358), INT16_C(  -623), INT16_C(   -51), INT16_C(   -51),
                            INT16_C(   -51), INT16_C(   -51), INT16_C(-11863), INT16_C(   -51)) },
    { easysimd_mm512_set_epi16(INT16_C(-12929), INT16_C( -9559), INT16_C( -1255), INT16_C(-25300),
                            INT16_C( 24130), INT16_C( 22555), INT16_C(-26496), INT16_C(  4179),
                            INT16_C( 25227), INT16_C( 31028), INT16_C( 12492), INT16_C(-27096),
                            INT16_C( 22382), INT16_C( -5113), INT16_C(-30455), INT16_C( 15691),
                            INT16_C(-18605), INT16_C( -4278), INT16_C( 11441), INT16_C(-26478),
                            INT16_C( 11388), INT16_C(-27754), INT16_C(   607), INT16_C( -1601),
                            INT16_C(-14454), INT16_C(  1251), INT16_C( 27178), INT16_C( 11399),
                            INT16_C(  -184), INT16_C( 17990), INT16_C(-12132), INT16_C(-20400)),
      UINT32_C(3701546889),
      INT16_C( 26765),
      easysimd_mm512_set_epi16(INT16_C( 26765), INT16_C( 26765), INT16_C( -1255), INT16_C( 26765),
                            INT16_C( 26765), INT16_C( 26765), INT16_C(-26496), INT16_C(  4179),
                            INT16_C( 26765), INT16_C( 31028), INT16_C( 26765), INT16_C(-27096),
                            INT16_C( 22382), INT16_C( -5113), INT16_C(-30455), INT16_C( 26765),
                            INT16_C(-18605), INT16_C( -4278), INT16_C( 11441), INT16_C( 26765),
                            INT16_C( 26765), INT16_C( 26765), INT16_C( 26765), INT16_C( 26765),
                            INT16_C( 26765), INT16_C(  1251), INT16_C( 27178), INT16_C( 11399),
                            INT16_C( 26765), INT16_C( 17990), INT16_C(-12132), INT16_C( 26765)) },
    { easysimd_mm512_set_epi16(INT16_C( 23556), INT16_C( 11192), INT16_C(-13439), INT16_C( -2357),
                            INT16_C(   858), INT16_C( 27575), INT16_C( 20368), INT16_C(-20256),
                            INT16_C(-11019), INT16_C( -7073), INT16_C(-32385), INT16_C( 27749),
                            INT16_C( 17332), INT16_C(-28131), INT16_C( 22510), INT16_C(  -872),
                            INT16_C( 20986), INT16_C(-25896), INT16_C(  7561), INT16_C(-22951),
                            INT16_C( -9997), INT16_C( 18542), INT16_C( -1921), INT16_C(-16319),
                            INT16_C(-24759), INT16_C( 10467), INT16_C(  8453), INT16_C(  5278),
                            INT16_C(-22217), INT16_C( 17080), INT16_C( 16797), INT16_C( -9777)),
      UINT32_C(3298748633),
      INT16_C( -5240),
      easysimd_mm512_set_epi16(INT16_C( -5240), INT16_C( -5240), INT16_C(-13439), INT16_C( -2357),
                            INT16_C(   858), INT16_C( -5240), INT16_C( 20368), INT16_C(-20256),
                            INT16_C( -5240), INT16_C( -7073), INT16_C(-32385), INT16_C( -5240),
                            INT16_C( -5240), INT16_C( -5240), INT16_C( -5240), INT16_C(  -872),
                            INT16_C( -5240), INT16_C( -5240), INT16_C( -5240), INT16_C(-22951),
                            INT16_C( -5240), INT16_C( 18542), INT16_C( -1921), INT16_C(-16319),
                            INT16_C( -5240), INT16_C( -5240), INT16_C(  8453), INT16_C( -5240),
                            INT16_C( -5240), INT16_C( 17080), INT16_C( 16797), INT16_C( -5240)) },
    { easysimd_mm512_set_epi16(INT16_C(  -894), INT16_C( 15324), INT16_C(-23364), INT16_C( 25648),
                            INT16_C(  -512), INT16_C( 12172), INT16_C(-27706), INT16_C(-10514),
                            INT16_C(  1026), INT16_C( 20384), INT16_C(-25471), INT16_C( -3464),
                            INT16_C( 14827), INT16_C( 18045), INT16_C(-25826), INT16_C( 12664),
                            INT16_C(-16682), INT16_C( 16498), INT16_C( 29333), INT16_C(  -511),
                            INT16_C( 15382), INT16_C(-19710), INT16_C(-14139), INT16_C( 14459),
                            INT16_C( 16092), INT16_C(-12889), INT16_C(  -337), INT16_C( 29893),
                            INT16_C(-29467), INT16_C( -8274), INT16_C( 30322), INT16_C(-19138)),
      UINT32_C(3605268017),
      INT16_C(-14523),
      easysimd_mm512_set_epi16(INT16_C(-14523), INT16_C(-14523), INT16_C(-23364), INT16_C(-14523),
                            INT16_C(  -512), INT16_C(-14523), INT16_C(-14523), INT16_C(-10514),
                            INT16_C(-14523), INT16_C(-14523), INT16_C(-14523), INT16_C( -3464),
                            INT16_C( 14827), INT16_C(-14523), INT16_C(-25826), INT16_C( 12664),
                            INT16_C(-16682), INT16_C( 16498), INT16_C( 29333), INT16_C(  -511),
                            INT16_C( 15382), INT16_C(-14523), INT16_C(-14523), INT16_C( 14459),
                            INT16_C( 16092), INT16_C(-12889), INT16_C(-14523), INT16_C(-14523),
                            INT16_C(-29467), INT16_C( -8274), INT16_C( 30322), INT16_C(-14523)) },
    { easysimd_mm512_set_epi16(INT16_C( -6967), INT16_C(-20070), INT16_C( -8289), INT16_C(  -479),
                            INT16_C(-18969), INT16_C( -6012), INT16_C( 11721), INT16_C( 13564),
                            INT16_C( 19765), INT16_C( 23581), INT16_C(-21527), INT16_C( -2847),
                            INT16_C( 23178), INT16_C(-14967), INT16_C( 17682), INT16_C( 28255),
                            INT16_C(  8882), INT16_C( 14691), INT16_C(-27903), INT16_C( 28973),
                            INT16_C(   619), INT16_C(-10329), INT16_C( 25572), INT16_C(-13439),
                            INT16_C( -3930), INT16_C(  5659), INT16_C(  -675), INT16_C(-18004),
                            INT16_C(-26191), INT16_C(  5303), INT16_C(-13369), INT16_C( 21695)),
      UINT32_C( 349570055),
      INT16_C( 24210),
      easysimd_mm512_set_epi16(INT16_C( -6967), INT16_C(-20070), INT16_C( -8289), INT16_C( 24210),
                            INT16_C(-18969), INT16_C( 24210), INT16_C( 11721), INT16_C( 13564),
                            INT16_C( 24210), INT16_C( 24210), INT16_C(-21527), INT16_C( 24210),
                            INT16_C( 23178), INT16_C( 24210), INT16_C( 24210), INT16_C( 28255),
                            INT16_C(  8882), INT16_C( 14691), INT16_C(-27903), INT16_C( 28973),
                            INT16_C(   619), INT16_C( 24210), INT16_C( 25572), INT16_C(-13439),
                            INT16_C( -3930), INT16_C(  5659), INT16_C(  -675), INT16_C(-18004),
                            INT16_C(-26191), INT16_C( 24210), INT16_C( 24210), INT16_C( 24210)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask32 k = test_vec[i].k;
    int16_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_set1_epi16(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_set1_epi16");
    easysimd_assert_m512i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_set1_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask32 k;
    int16_t a;
    easysimd__m512i r;
  } test_vec[8] = {
   {  UINT32_C( 693683203),
      INT16_C(-16188),
      easysimd_mm512_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(-16188), INT16_C(     0),
                            INT16_C(-16188), INT16_C(     0), INT16_C(     0), INT16_C(-16188),
                            INT16_C(     0), INT16_C(-16188), INT16_C(     0), INT16_C(-16188),
                            INT16_C(-16188), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(-16188), INT16_C(-16188), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(-16188), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(-16188), INT16_C(-16188)) },
   {  UINT32_C(2322862674),
      INT16_C(-31832),
      easysimd_mm512_set_epi16(INT16_C(-31832), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(-31832), INT16_C(     0), INT16_C(-31832), INT16_C(     0),
                            INT16_C(     0), INT16_C(-31832), INT16_C(-31832), INT16_C(-31832),
                            INT16_C(     0), INT16_C(-31832), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(-31832),
                            INT16_C(     0), INT16_C(     0), INT16_C(-31832), INT16_C(     0),
                            INT16_C(     0), INT16_C(-31832), INT16_C(     0), INT16_C(-31832),
                            INT16_C(     0), INT16_C(     0), INT16_C(-31832), INT16_C(     0)) },
   {  UINT32_C(3196780114),
      INT16_C(  8083),
      easysimd_mm512_set_epi16(INT16_C(  8083), INT16_C(     0), INT16_C(  8083), INT16_C(  8083),
                            INT16_C(  8083), INT16_C(  8083), INT16_C(  8083), INT16_C(     0),
                            INT16_C(  8083), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(  8083), INT16_C(     0), INT16_C(  8083), INT16_C(     0),
                            INT16_C(  8083), INT16_C(  8083), INT16_C(  8083), INT16_C(  8083),
                            INT16_C(  8083), INT16_C(  8083), INT16_C(  8083), INT16_C(     0),
                            INT16_C(     0), INT16_C(  8083), INT16_C(     0), INT16_C(  8083),
                            INT16_C(     0), INT16_C(     0), INT16_C(  8083), INT16_C(     0)) },
   {  UINT32_C( 962615778),
      INT16_C( 10134),
      easysimd_mm512_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C( 10134), INT16_C( 10134),
                            INT16_C( 10134), INT16_C(     0), INT16_C(     0), INT16_C( 10134),
                            INT16_C(     0), INT16_C( 10134), INT16_C( 10134), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C( 10134), INT16_C(     0), INT16_C( 10134),
                            INT16_C( 10134), INT16_C(     0), INT16_C(     0), INT16_C( 10134),
                            INT16_C( 10134), INT16_C( 10134), INT16_C( 10134), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C( 10134), INT16_C(     0)) },
   {  UINT32_C(3651012064),
      INT16_C(-28841),
      easysimd_mm512_set_epi16(INT16_C(-28841), INT16_C(-28841), INT16_C(     0), INT16_C(-28841),
                            INT16_C(-28841), INT16_C(     0), INT16_C(     0), INT16_C(-28841),
                            INT16_C(-28841), INT16_C(     0), INT16_C(     0), INT16_C(-28841),
                            INT16_C(-28841), INT16_C(-28841), INT16_C(-28841), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(-28841), INT16_C(     0), INT16_C(-28841),
                            INT16_C(-28841), INT16_C(-28841), INT16_C(-28841), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
   {  UINT32_C(1153035128),
      INT16_C(  9546),
      easysimd_mm512_set_epi16(INT16_C(     0), INT16_C(  9546), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(  9546), INT16_C(     0), INT16_C(     0),
                            INT16_C(  9546), INT16_C(     0), INT16_C(  9546), INT16_C(  9546),
                            INT16_C(  9546), INT16_C(     0), INT16_C(     0), INT16_C(  9546),
                            INT16_C(  9546), INT16_C(  9546), INT16_C(  9546), INT16_C(     0),
                            INT16_C(  9546), INT16_C(     0), INT16_C(  9546), INT16_C(  9546),
                            INT16_C(     0), INT16_C(  9546), INT16_C(  9546), INT16_C(  9546),
                            INT16_C(  9546), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
   {  UINT32_C(2648275992),
      INT16_C(-29002),
      easysimd_mm512_set_epi16(INT16_C(-29002), INT16_C(     0), INT16_C(     0), INT16_C(-29002),
                            INT16_C(-29002), INT16_C(-29002), INT16_C(     0), INT16_C(-29002),
                            INT16_C(-29002), INT16_C(-29002), INT16_C(     0), INT16_C(-29002),
                            INT16_C(-29002), INT16_C(     0), INT16_C(     0), INT16_C(-29002),
                            INT16_C(     0), INT16_C(-29002), INT16_C(-29002), INT16_C(-29002),
                            INT16_C(-29002), INT16_C(-29002), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(-29002),
                            INT16_C(-29002), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
   {  UINT32_C(1548742660),
      INT16_C( 11362),
      easysimd_mm512_set_epi16(INT16_C(     0), INT16_C( 11362), INT16_C(     0), INT16_C( 11362),
                            INT16_C( 11362), INT16_C( 11362), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C( 11362), INT16_C(     0), INT16_C(     0),
                            INT16_C( 11362), INT16_C( 11362), INT16_C( 11362), INT16_C( 11362),
                            INT16_C( 11362), INT16_C( 11362), INT16_C( 11362), INT16_C( 11362),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                            INT16_C(     0), INT16_C( 11362), INT16_C(     0), INT16_C(     0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    int16_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_set1_epi16(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_set1_epi16");
    easysimd_assert_m512i_i16(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_set1_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    int32_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    {  1727286739,
      easysimd_mm512_set_epi32(INT32_C( 1727286739), INT32_C( 1727286739), INT32_C( 1727286739), INT32_C( 1727286739),
                            INT32_C( 1727286739), INT32_C( 1727286739), INT32_C( 1727286739), INT32_C( 1727286739),
                            INT32_C( 1727286739), INT32_C( 1727286739), INT32_C( 1727286739), INT32_C( 1727286739),
                            INT32_C( 1727286739), INT32_C( 1727286739), INT32_C( 1727286739), INT32_C( 1727286739)) },
    {  1944050466,
      easysimd_mm512_set_epi32(INT32_C( 1944050466), INT32_C( 1944050466), INT32_C( 1944050466), INT32_C( 1944050466),
                            INT32_C( 1944050466), INT32_C( 1944050466), INT32_C( 1944050466), INT32_C( 1944050466),
                            INT32_C( 1944050466), INT32_C( 1944050466), INT32_C( 1944050466), INT32_C( 1944050466),
                            INT32_C( 1944050466), INT32_C( 1944050466), INT32_C( 1944050466), INT32_C( 1944050466)) },
    { -1212539061,
      easysimd_mm512_set_epi32(INT32_C(-1212539061), INT32_C(-1212539061), INT32_C(-1212539061), INT32_C(-1212539061),
                            INT32_C(-1212539061), INT32_C(-1212539061), INT32_C(-1212539061), INT32_C(-1212539061),
                            INT32_C(-1212539061), INT32_C(-1212539061), INT32_C(-1212539061), INT32_C(-1212539061),
                            INT32_C(-1212539061), INT32_C(-1212539061), INT32_C(-1212539061), INT32_C(-1212539061)) },
    { -1654733061,
      easysimd_mm512_set_epi32(INT32_C(-1654733061), INT32_C(-1654733061), INT32_C(-1654733061), INT32_C(-1654733061),
                            INT32_C(-1654733061), INT32_C(-1654733061), INT32_C(-1654733061), INT32_C(-1654733061),
                            INT32_C(-1654733061), INT32_C(-1654733061), INT32_C(-1654733061), INT32_C(-1654733061),
                            INT32_C(-1654733061), INT32_C(-1654733061), INT32_C(-1654733061), INT32_C(-1654733061)) },
    { -1048158621,
      easysimd_mm512_set_epi32(INT32_C(-1048158621), INT32_C(-1048158621), INT32_C(-1048158621), INT32_C(-1048158621),
                            INT32_C(-1048158621), INT32_C(-1048158621), INT32_C(-1048158621), INT32_C(-1048158621),
                            INT32_C(-1048158621), INT32_C(-1048158621), INT32_C(-1048158621), INT32_C(-1048158621),
                            INT32_C(-1048158621), INT32_C(-1048158621), INT32_C(-1048158621), INT32_C(-1048158621)) },
    {  -676031020,
      easysimd_mm512_set_epi32(INT32_C( -676031020), INT32_C( -676031020), INT32_C( -676031020), INT32_C( -676031020),
                            INT32_C( -676031020), INT32_C( -676031020), INT32_C( -676031020), INT32_C( -676031020),
                            INT32_C( -676031020), INT32_C( -676031020), INT32_C( -676031020), INT32_C( -676031020),
                            INT32_C( -676031020), INT32_C( -676031020), INT32_C( -676031020), INT32_C( -676031020)) },
    {   651688918,
      easysimd_mm512_set_epi32(INT32_C(  651688918), INT32_C(  651688918), INT32_C(  651688918), INT32_C(  651688918),
                            INT32_C(  651688918), INT32_C(  651688918), INT32_C(  651688918), INT32_C(  651688918),
                            INT32_C(  651688918), INT32_C(  651688918), INT32_C(  651688918), INT32_C(  651688918),
                            INT32_C(  651688918), INT32_C(  651688918), INT32_C(  651688918), INT32_C(  651688918)) },
    { -1051556258,
      easysimd_mm512_set_epi32(INT32_C(-1051556258), INT32_C(-1051556258), INT32_C(-1051556258), INT32_C(-1051556258),
                            INT32_C(-1051556258), INT32_C(-1051556258), INT32_C(-1051556258), INT32_C(-1051556258),
                            INT32_C(-1051556258), INT32_C(-1051556258), INT32_C(-1051556258), INT32_C(-1051556258),
                            INT32_C(-1051556258), INT32_C(-1051556258), INT32_C(-1051556258), INT32_C(-1051556258)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int32_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_set1_epi32(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_set1_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_set1_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask16 k;
    int32_t a;
    easysimd__m512i r;
  } test_vec[8] = {
      { easysimd_mm512_set_epi32(INT32_C(-2133842294), INT32_C( 1453587049), INT32_C( 2146642803), INT32_C(-1231323727),
                            INT32_C( 1853533908), INT32_C(-1907653908), INT32_C(  564694133), INT32_C(-1137944481),
                            INT32_C(  355997036), INT32_C(   15257739), INT32_C( 1494729649), INT32_C( 1029796613),
                            INT32_C( 2106354400), INT32_C( 1763331885), INT32_C( -506942576), INT32_C( -586993838)),
      UINT16_C( 2584),
       -447334412,
      easysimd_mm512_set_epi32(INT32_C(-2133842294), INT32_C( 1453587049), INT32_C( 2146642803), INT32_C(-1231323727),
                            INT32_C( -447334412), INT32_C(-1907653908), INT32_C( -447334412), INT32_C(-1137944481),
                            INT32_C(  355997036), INT32_C(   15257739), INT32_C( 1494729649), INT32_C( -447334412),
                            INT32_C( -447334412), INT32_C( 1763331885), INT32_C( -506942576), INT32_C( -586993838)) },
    { easysimd_mm512_set_epi32(INT32_C( -998613800), INT32_C(  131555600), INT32_C( -778207920), INT32_C(-1881674086),
                            INT32_C(  920672518), INT32_C(-1753434445), INT32_C(  982638267), INT32_C( 1856541033),
                            INT32_C( -869506663), INT32_C( -394635465), INT32_C(-1596048192), INT32_C(  274218308),
                            INT32_C(  757893716), INT32_C(-2119306902), INT32_C(  364747827), INT32_C( -200526147)),
      UINT16_C(52133),
        749362876,
      easysimd_mm512_set_epi32(INT32_C(  749362876), INT32_C(  749362876), INT32_C( -778207920), INT32_C(-1881674086),
                            INT32_C(  749362876), INT32_C(-1753434445), INT32_C(  749362876), INT32_C(  749362876),
                            INT32_C(  749362876), INT32_C( -394635465), INT32_C(  749362876), INT32_C(  274218308),
                            INT32_C(  757893716), INT32_C(  749362876), INT32_C(  364747827), INT32_C(  749362876)) },
    { easysimd_mm512_set_epi32(INT32_C(-2009617550), INT32_C( 1354406381), INT32_C( 2028903938), INT32_C(-1425115920),
                            INT32_C(-1833209985), INT32_C( -485232115), INT32_C( -246273875), INT32_C(-1220668381),
                            INT32_C( 1710154952), INT32_C(-1764069342), INT32_C( -426734827), INT32_C(-1603498425),
                            INT32_C(-1463214772), INT32_C(-1312774926), INT32_C(  714085999), INT32_C( -352604741)),
      UINT16_C(50570),
        722829713,
      easysimd_mm512_set_epi32(INT32_C(  722829713), INT32_C(  722829713), INT32_C( 2028903938), INT32_C(-1425115920),
                            INT32_C(-1833209985), INT32_C(  722829713), INT32_C( -246273875), INT32_C(  722829713),
                            INT32_C(  722829713), INT32_C(-1764069342), INT32_C( -426734827), INT32_C(-1603498425),
                            INT32_C(  722829713), INT32_C(-1312774926), INT32_C(  722829713), INT32_C( -352604741)) },
    { easysimd_mm512_set_epi32(INT32_C(-1600817970), INT32_C( -289243644), INT32_C(  742005878), INT32_C( -612930926),
                            INT32_C(  717430896), INT32_C( 1787140065), INT32_C(-1405808293), INT32_C(  816556317),
                            INT32_C( 1747379900), INT32_C(-1006412100), INT32_C( 2116251350), INT32_C(-1238632202),
                            INT32_C( 1684739890), INT32_C( 1414060999), INT32_C(-2081867445), INT32_C( 1952705540)),
      UINT16_C(15423),
       1968604658,
      easysimd_mm512_set_epi32(INT32_C(-1600817970), INT32_C( -289243644), INT32_C( 1968604658), INT32_C( 1968604658),
                            INT32_C( 1968604658), INT32_C( 1968604658), INT32_C(-1405808293), INT32_C(  816556317),
                            INT32_C( 1747379900), INT32_C(-1006412100), INT32_C( 1968604658), INT32_C( 1968604658),
                            INT32_C( 1968604658), INT32_C( 1968604658), INT32_C( 1968604658), INT32_C( 1968604658)) },
    { easysimd_mm512_set_epi32(INT32_C( -666739030), INT32_C(-1370874438), INT32_C(-1476494318), INT32_C(-1101994537),
                            INT32_C(  338919471), INT32_C( -523657701), INT32_C( 1918205933), INT32_C( -933363441),
                            INT32_C(  191279486), INT32_C( -793805997), INT32_C(-1611569913), INT32_C(-1249963897),
                            INT32_C(-1384621234), INT32_C( 1593832662), INT32_C(  656079206), INT32_C(-1000644982)),
      UINT16_C(34631),
        997675190,
      easysimd_mm512_set_epi32(INT32_C(  997675190), INT32_C(-1370874438), INT32_C(-1476494318), INT32_C(-1101994537),
                            INT32_C(  338919471), INT32_C(  997675190), INT32_C(  997675190), INT32_C(  997675190),
                            INT32_C(  191279486), INT32_C(  997675190), INT32_C(-1611569913), INT32_C(-1249963897),
                            INT32_C(-1384621234), INT32_C(  997675190), INT32_C(  997675190), INT32_C(  997675190)) },
    { easysimd_mm512_set_epi32(INT32_C(  121649236), INT32_C( 1078857855), INT32_C( -789079366), INT32_C(  720922870),
                            INT32_C( 2041256669), INT32_C( -203208947), INT32_C( 1607011101), INT32_C(-1156829654),
                            INT32_C(  230848793), INT32_C( 1678224863), INT32_C( 2110278578), INT32_C(-1808926794),
                            INT32_C( 1395318189), INT32_C(  331190146), INT32_C(  150534496), INT32_C(  511594435)),
      UINT16_C(61391),
      -1035845727,
      easysimd_mm512_set_epi32(INT32_C(-1035845727), INT32_C(-1035845727), INT32_C(-1035845727), INT32_C(  720922870),
                            INT32_C(-1035845727), INT32_C(-1035845727), INT32_C(-1035845727), INT32_C(-1035845727),
                            INT32_C(-1035845727), INT32_C(-1035845727), INT32_C( 2110278578), INT32_C(-1808926794),
                            INT32_C(-1035845727), INT32_C(-1035845727), INT32_C(-1035845727), INT32_C(-1035845727)) },
    { easysimd_mm512_set_epi32(INT32_C( -439673063), INT32_C(  281345174), INT32_C( 1703672409), INT32_C( 1433894072),
                            INT32_C(-1374287391), INT32_C(-2054374124), INT32_C(-2087863688), INT32_C(  775409014),
                            INT32_C(  684629778), INT32_C(-1498533524), INT32_C( -208955538), INT32_C( 1063127700),
                            INT32_C(  429182470), INT32_C(-1892329828), INT32_C(  837229295), INT32_C( -115373033)),
      UINT16_C( 2879),
      -1796290912,
      easysimd_mm512_set_epi32(INT32_C( -439673063), INT32_C(  281345174), INT32_C( 1703672409), INT32_C( 1433894072),
                            INT32_C(-1796290912), INT32_C(-2054374124), INT32_C(-1796290912), INT32_C(-1796290912),
                            INT32_C(  684629778), INT32_C(-1498533524), INT32_C(-1796290912), INT32_C(-1796290912),
                            INT32_C(-1796290912), INT32_C(-1796290912), INT32_C(-1796290912), INT32_C(-1796290912)) },
    { easysimd_mm512_set_epi32(INT32_C(  211854878), INT32_C( 1120217162), INT32_C( 1399020352), INT32_C(-1730262794),
                            INT32_C( -217750907), INT32_C(-1958971298), INT32_C( 1308051941), INT32_C(  659156948),
                            INT32_C( -413755412), INT32_C(-1891691945), INT32_C(-1613764989), INT32_C( 1818229349),
                            INT32_C( 1838020027), INT32_C( 1546326520), INT32_C( 1564338027), INT32_C( 1340948138)),
      UINT16_C(26109),
        154532243,
      easysimd_mm512_set_epi32(INT32_C(  211854878), INT32_C(  154532243), INT32_C(  154532243), INT32_C(-1730262794),
                            INT32_C( -217750907), INT32_C(  154532243), INT32_C( 1308051941), INT32_C(  154532243),
                            INT32_C(  154532243), INT32_C(  154532243), INT32_C(  154532243), INT32_C(  154532243),
                            INT32_C(  154532243), INT32_C(  154532243), INT32_C( 1564338027), INT32_C(  154532243)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask16 k = test_vec[i].k;
    int32_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_set1_epi32(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_set1_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_set1_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask16 k;
    int32_t a;
    easysimd__m512i r;
  } test_vec[8] = {
   {  UINT16_C(55449),
       1161879327,
      easysimd_mm512_set_epi32(INT32_C( 1161879327), INT32_C( 1161879327), INT32_C(          0), INT32_C( 1161879327),
                            INT32_C( 1161879327), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1161879327), INT32_C(          0), INT32_C(          0), INT32_C( 1161879327),
                            INT32_C( 1161879327), INT32_C(          0), INT32_C(          0), INT32_C( 1161879327)) },
   {  UINT16_C(42205),
        491258437,
      easysimd_mm512_set_epi32(INT32_C(  491258437), INT32_C(          0), INT32_C(  491258437), INT32_C(          0),
                            INT32_C(          0), INT32_C(  491258437), INT32_C(          0), INT32_C(          0),
                            INT32_C(  491258437), INT32_C(  491258437), INT32_C(          0), INT32_C(  491258437),
                            INT32_C(  491258437), INT32_C(  491258437), INT32_C(          0), INT32_C(  491258437)) },
   {  UINT16_C(46294),
       1464671644,
      easysimd_mm512_set_epi32(INT32_C( 1464671644), INT32_C(          0), INT32_C( 1464671644), INT32_C( 1464671644),
                            INT32_C(          0), INT32_C( 1464671644), INT32_C(          0), INT32_C(          0),
                            INT32_C( 1464671644), INT32_C( 1464671644), INT32_C(          0), INT32_C( 1464671644),
                            INT32_C(          0), INT32_C( 1464671644), INT32_C( 1464671644), INT32_C(          0)) },
   {  UINT16_C(57846),
       1382569562,
      easysimd_mm512_set_epi32(INT32_C( 1382569562), INT32_C( 1382569562), INT32_C( 1382569562), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C( 1382569562),
                            INT32_C( 1382569562), INT32_C( 1382569562), INT32_C( 1382569562), INT32_C( 1382569562),
                            INT32_C(          0), INT32_C( 1382569562), INT32_C( 1382569562), INT32_C(          0)) },
   {  UINT16_C(64688),
        417592133,
      easysimd_mm512_set_epi32(INT32_C(  417592133), INT32_C(  417592133), INT32_C(  417592133), INT32_C(  417592133),
                            INT32_C(  417592133), INT32_C(  417592133), INT32_C(          0), INT32_C(          0),
                            INT32_C(  417592133), INT32_C(          0), INT32_C(  417592133), INT32_C(  417592133),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
   {  UINT16_C(40468),
        103154350,
      easysimd_mm512_set_epi32(INT32_C(  103154350), INT32_C(          0), INT32_C(          0), INT32_C(  103154350),
                            INT32_C(  103154350), INT32_C(  103154350), INT32_C(  103154350), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(  103154350),
                            INT32_C(          0), INT32_C(  103154350), INT32_C(          0), INT32_C(          0)) },
   {  UINT16_C(20696),
        487897671,
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(  487897671), INT32_C(          0), INT32_C(  487897671),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(  487897671), INT32_C(  487897671), INT32_C(          0), INT32_C(  487897671),
                            INT32_C(  487897671), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
   {  UINT16_C(47493),
        643357764,
      easysimd_mm512_set_epi32(INT32_C(  643357764), INT32_C(          0), INT32_C(  643357764), INT32_C(  643357764),
                            INT32_C(  643357764), INT32_C(          0), INT32_C(          0), INT32_C(  643357764),
                            INT32_C(  643357764), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(  643357764), INT32_C(          0), INT32_C(  643357764)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    int32_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_set1_epi32(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_set1_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_set1_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    int64_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    { -8789375007372599774,
      easysimd_mm512_set_epi64(INT64_C(-8789375007372599774), INT64_C(-8789375007372599774),
                            INT64_C(-8789375007372599774), INT64_C(-8789375007372599774),
                            INT64_C(-8789375007372599774), INT64_C(-8789375007372599774),
                            INT64_C(-8789375007372599774), INT64_C(-8789375007372599774)) },
    { -4285935604177939284,
      easysimd_mm512_set_epi64(INT64_C(-4285935604177939284), INT64_C(-4285935604177939284),
                            INT64_C(-4285935604177939284), INT64_C(-4285935604177939284),
                            INT64_C(-4285935604177939284), INT64_C(-4285935604177939284),
                            INT64_C(-4285935604177939284), INT64_C(-4285935604177939284)) },
    { -1541935515905504488,
      easysimd_mm512_set_epi64(INT64_C(-1541935515905504488), INT64_C(-1541935515905504488),
                            INT64_C(-1541935515905504488), INT64_C(-1541935515905504488),
                            INT64_C(-1541935515905504488), INT64_C(-1541935515905504488),
                            INT64_C(-1541935515905504488), INT64_C(-1541935515905504488)) },
    {  5952985382071947058,
      easysimd_mm512_set_epi64(INT64_C( 5952985382071947058), INT64_C( 5952985382071947058),
                            INT64_C( 5952985382071947058), INT64_C( 5952985382071947058),
                            INT64_C( 5952985382071947058), INT64_C( 5952985382071947058),
                            INT64_C( 5952985382071947058), INT64_C( 5952985382071947058)) },
    { -7162660555270519798,
      easysimd_mm512_set_epi64(INT64_C(-7162660555270519798), INT64_C(-7162660555270519798),
                            INT64_C(-7162660555270519798), INT64_C(-7162660555270519798),
                            INT64_C(-7162660555270519798), INT64_C(-7162660555270519798),
                            INT64_C(-7162660555270519798), INT64_C(-7162660555270519798)) },
    {  8404097979084250521,
      easysimd_mm512_set_epi64(INT64_C( 8404097979084250521), INT64_C( 8404097979084250521),
                            INT64_C( 8404097979084250521), INT64_C( 8404097979084250521),
                            INT64_C( 8404097979084250521), INT64_C( 8404097979084250521),
                            INT64_C( 8404097979084250521), INT64_C( 8404097979084250521)) },
    {   274863432779804064,
      easysimd_mm512_set_epi64(INT64_C(  274863432779804064), INT64_C(  274863432779804064),
                            INT64_C(  274863432779804064), INT64_C(  274863432779804064),
                            INT64_C(  274863432779804064), INT64_C(  274863432779804064),
                            INT64_C(  274863432779804064), INT64_C(  274863432779804064)) },
    { -6073562903357076278,
      easysimd_mm512_set_epi64(INT64_C(-6073562903357076278), INT64_C(-6073562903357076278),
                            INT64_C(-6073562903357076278), INT64_C(-6073562903357076278),
                            INT64_C(-6073562903357076278), INT64_C(-6073562903357076278),
                            INT64_C(-6073562903357076278), INT64_C(-6073562903357076278)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int64_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_set1_epi64(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_set1_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_set1_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i src;
    easysimd__mmask8 k;
    int64_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C( 1045216498523672669), INT64_C(-6036444540175881058),
                            INT64_C(-5911148920502355606), INT64_C(-7577028982327639795),
                            INT64_C(-2741592730704877834), INT64_C(-6453831303076951346),
                            INT64_C(-6689489276565790355), INT64_C(  202845396246057443)),
      UINT8_C(230),
        147395854529182590,
      easysimd_mm512_set_epi64(INT64_C(  147395854529182590), INT64_C(  147395854529182590),
                            INT64_C(  147395854529182590), INT64_C(-7577028982327639795),
                            INT64_C(-2741592730704877834), INT64_C(  147395854529182590),
                            INT64_C(  147395854529182590), INT64_C(  202845396246057443)) },
    { easysimd_mm512_set_epi64(INT64_C(-7718401035083209663), INT64_C(-3076780457048070953),
                            INT64_C( 6475016072843370494), INT64_C(-8381463578250516776),
                            INT64_C( 7440179812526306236), INT64_C(-1565233080792835049),
                            INT64_C(-3520705077242655190), INT64_C(  711599945422741640)),
      UINT8_C(183),
       7896918852801948623,
      easysimd_mm512_set_epi64(INT64_C( 7896918852801948623), INT64_C(-3076780457048070953),
                            INT64_C( 7896918852801948623), INT64_C( 7896918852801948623),
                            INT64_C( 7440179812526306236), INT64_C( 7896918852801948623),
                            INT64_C( 7896918852801948623), INT64_C( 7896918852801948623)) },
    { easysimd_mm512_set_epi64(INT64_C( 4486030894140599897), INT64_C( 6422628958957749227),
                            INT64_C(-5036188723709908563), INT64_C( 7249692644755604208),
                            INT64_C(-7968846935772652304), INT64_C(-1019958922473354647),
                            INT64_C( 5481721181155050457), INT64_C( 3220728135426515219)),
      UINT8_C(250),
       6737731418145878376,
      easysimd_mm512_set_epi64(INT64_C( 6737731418145878376), INT64_C( 6737731418145878376),
                            INT64_C( 6737731418145878376), INT64_C( 6737731418145878376),
                            INT64_C( 6737731418145878376), INT64_C(-1019958922473354647),
                            INT64_C( 6737731418145878376), INT64_C( 3220728135426515219)) },
    { easysimd_mm512_set_epi64(INT64_C(-6396453660831390526), INT64_C( 8933529613499491135),
                            INT64_C( -583608444119273487), INT64_C( 2774349158822651995),
                            INT64_C( 5342483589547515588), INT64_C(  169032945576329978),
                            INT64_C(-6862029605560509115), INT64_C( 6948715933942990141)),
      UINT8_C(144),
       5224961598009568585,
      easysimd_mm512_set_epi64(INT64_C( 5224961598009568585), INT64_C( 8933529613499491135),
                            INT64_C( -583608444119273487), INT64_C( 5224961598009568585),
                            INT64_C( 5342483589547515588), INT64_C(  169032945576329978),
                            INT64_C(-6862029605560509115), INT64_C( 6948715933942990141)) },
    { easysimd_mm512_set_epi64(INT64_C(-4346308446834850778), INT64_C( 2749670639259677889),
                            INT64_C(-1682235429196139261), INT64_C(-8570560540139381802),
                            INT64_C(-7853283901496397391), INT64_C(  153768084219711829),
                            INT64_C(-3210037353748455743), INT64_C(-4029896259883002015)),
      UINT8_C(214),
      -5146489163462262224,
      easysimd_mm512_set_epi64(INT64_C(-5146489163462262224), INT64_C(-5146489163462262224),
                            INT64_C(-1682235429196139261), INT64_C(-5146489163462262224),
                            INT64_C(-7853283901496397391), INT64_C(-5146489163462262224),
                            INT64_C(-5146489163462262224), INT64_C(-4029896259883002015)) },
    { easysimd_mm512_set_epi64(INT64_C( 6394437943527522650), INT64_C(-6125470791748892618),
                            INT64_C(-5975035781359101837), INT64_C( 4399409063692409934),
                            INT64_C(-8019209045639092618), INT64_C(-3157603671849839607),
                            INT64_C(-6814419689115640150), INT64_C( 5538401471960412489)),
      UINT8_C( 88),
       -748084489617986997,
      easysimd_mm512_set_epi64(INT64_C( 6394437943527522650), INT64_C( -748084489617986997),
                            INT64_C(-5975035781359101837), INT64_C( -748084489617986997),
                            INT64_C( -748084489617986997), INT64_C(-3157603671849839607),
                            INT64_C(-6814419689115640150), INT64_C( 5538401471960412489)) },
    { easysimd_mm512_set_epi64(INT64_C( 6475451416366061513), INT64_C( 3128457729014411682),
                            INT64_C( 4167134861407868007), INT64_C( 2076318686723048286),
                            INT64_C(  764926893292127387), INT64_C(-3471922167199587188),
                            INT64_C(-1007473193319966067), INT64_C(-7587900950013848349)),
      UINT8_C( 14),
      -3095861881784422408,
      easysimd_mm512_set_epi64(INT64_C( 6475451416366061513), INT64_C( 3128457729014411682),
                            INT64_C( 4167134861407868007), INT64_C( 2076318686723048286),
                            INT64_C(-3095861881784422408), INT64_C(-3095861881784422408),
                            INT64_C(-3095861881784422408), INT64_C(-7587900950013848349)) },
    { easysimd_mm512_set_epi64(INT64_C(-8918688664014182717), INT64_C(-5923824341695687917),
                            INT64_C(  597335319340416274), INT64_C(-6405873593024845306),
                            INT64_C( 9156616106305782892), INT64_C(-3930771615997737816),
                            INT64_C(-3489614562807589194), INT64_C(-6234599678791232286)),
      UINT8_C( 48),
      -2201472844397108415,
      easysimd_mm512_set_epi64(INT64_C(-8918688664014182717), INT64_C(-5923824341695687917),
                            INT64_C(-2201472844397108415), INT64_C(-2201472844397108415),
                            INT64_C( 9156616106305782892), INT64_C(-3930771615997737816),
                            INT64_C(-3489614562807589194), INT64_C(-6234599678791232286)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i src = test_vec[i].src;
    easysimd__mmask8 k = test_vec[i].k;
    int64_t a = test_vec[i].a;
    easysimd__m512i r ;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_set1_epi64(src, k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_set1_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_set1_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__mmask8 k;
    int64_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    { UINT8_C(207),
       9161374966470958313,
      easysimd_mm512_set_epi64(INT64_C( 9161374966470958313), INT64_C( 9161374966470958313),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C( 9161374966470958313), INT64_C( 9161374966470958313),
                            INT64_C( 9161374966470958313), INT64_C( 9161374966470958313)) },
    { UINT8_C( 52),
      -5504071340329784539,
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-5504071340329784539), INT64_C(-5504071340329784539),
                            INT64_C(                   0), INT64_C(-5504071340329784539),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C( 37),
      -4694012945600318045,
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-4694012945600318045), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(-4694012945600318045),
                            INT64_C(                   0), INT64_C(-4694012945600318045)) },
    { UINT8_C( 77),
      -4616382267006571958,
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(-4616382267006571958),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-4616382267006571958), INT64_C(-4616382267006571958),
                            INT64_C(                   0), INT64_C(-4616382267006571958)) },
    { UINT8_C( 33),
      -7296455954195359480,
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-7296455954195359480), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(-7296455954195359480)) },
    { UINT8_C( 47),
      -8949112185126954032,
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-8949112185126954032), INT64_C(                   0),
                            INT64_C(-8949112185126954032), INT64_C(-8949112185126954032),
                            INT64_C(-8949112185126954032), INT64_C(-8949112185126954032)) },
    { UINT8_C( 80),
       8577224771648710248,
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C( 8577224771648710248),
                            INT64_C(                   0), INT64_C( 8577224771648710248),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { UINT8_C(  2),
      -5341779416438471199,
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(-5341779416438471199), INT64_C(                   0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    int64_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_set1_epi64(k, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_set1_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_set1_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_set1_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_set1_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_set1_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_set1_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_set1_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_set1_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_set1_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_set1_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_set1_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_set1_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_set1_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_set1_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_set1_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_set1_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_set1_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_set1_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_set1_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_set1_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_set1_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_set1_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_set1_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_set1_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_set1_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_set1_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_set1_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_set1_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_set1_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_set1_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_set1_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
