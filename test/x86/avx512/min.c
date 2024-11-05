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

#define EASYSIMD_TEST_X86_AVX512_INSN min

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/min.h>

static int
test_easysimd_mm_min_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 4155869632876165358), -INT64_C( 2692958359382883742) },
      { -INT64_C( 3776697126024260794), -INT64_C( 3877824833156842306) },
      { -INT64_C( 3776697126024260794), -INT64_C( 3877824833156842306) } },
    { { -INT64_C( 7382236279970489878), -INT64_C( 7780508980250174071) },
      {  INT64_C( 6029034116166917095),  INT64_C( 5229138066487258090) },
      { -INT64_C( 7382236279970489878), -INT64_C( 7780508980250174071) } },
    { {  INT64_C( 8938284011006782522),  INT64_C( 8766730092105203399) },
      { -INT64_C( 2832194694302498614),  INT64_C(   60289270075565104) },
      { -INT64_C( 2832194694302498614),  INT64_C(   60289270075565104) } },
    { {  INT64_C( 7100906242900587036), -INT64_C( 6122889015273769989) },
      { -INT64_C( 7536565728220388959),  INT64_C( 6028062842704693217) },
      { -INT64_C( 7536565728220388959), -INT64_C( 6122889015273769989) } },
    { { -INT64_C( 8040071485393332115), -INT64_C( 6580563885196189152) },
      {  INT64_C( 6233997235692222341),  INT64_C( 1165487427226020002) },
      { -INT64_C( 8040071485393332115), -INT64_C( 6580563885196189152) } },
    { { -INT64_C( 7214340562406043972), -INT64_C( 6265269990646218906) },
      {  INT64_C( 9136705079189258241),  INT64_C( 2986876583229835712) },
      { -INT64_C( 7214340562406043972), -INT64_C( 6265269990646218906) } },
    { {  INT64_C( 8442661673344142099),  INT64_C( 1308554152143406939) },
      { -INT64_C( 7672351641247892181),  INT64_C( 8481522069228019229) },
      { -INT64_C( 7672351641247892181),  INT64_C( 1308554152143406939) } },
    { { -INT64_C( 1767048729170006026),  INT64_C( 8005304756172380405) },
      {  INT64_C( 1420319387247092043),  INT64_C( 1058998026940296666) },
      { -INT64_C( 1767048729170006026),  INT64_C( 1058998026940296666) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_min_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_min_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_min_epi64(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_min_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[2];
    const uint64_t b[2];
    const uint64_t r[2];
  } test_vec[] = {
    { { UINT64_C(15705053435677423232), UINT64_C( 3646219045073541607) },
      { UINT64_C(16690629444520513689), UINT64_C( 1286576560956260276) },
      { UINT64_C(15705053435677423232), UINT64_C( 1286576560956260276) } },
    { { UINT64_C( 7226888158000836237), UINT64_C(13511405959910402665) },
      { UINT64_C(10316598698398372920), UINT64_C(12955788668506427178) },
      { UINT64_C( 7226888158000836237), UINT64_C(12955788668506427178) } },
    { { UINT64_C( 7063174708541581604), UINT64_C(17543830329351115674) },
      { UINT64_C(15596638058030088712), UINT64_C( 2777699536025669043) },
      { UINT64_C( 7063174708541581604), UINT64_C( 2777699536025669043) } },
    { { UINT64_C(13448817351173058632), UINT64_C(10452835532788880736) },
      { UINT64_C( 6716809513673181104), UINT64_C( 8132588507219020211) },
      { UINT64_C( 6716809513673181104), UINT64_C( 8132588507219020211) } },
    { { UINT64_C(15009745975586694003), UINT64_C( 7116807479440950185) },
      { UINT64_C(15008797841575393089), UINT64_C(13067019783818805439) },
      { UINT64_C(15008797841575393089), UINT64_C( 7116807479440950185) } },
    { { UINT64_C(  379873967919859717), UINT64_C( 7813175991336382553) },
      { UINT64_C(16437753966765813297), UINT64_C(  727932795846632482) },
      { UINT64_C(  379873967919859717), UINT64_C(  727932795846632482) } },
    { { UINT64_C( 5876569660509348395), UINT64_C(17405508482599011829) },
      { UINT64_C(12177837326797366775), UINT64_C( 1878607897840962306) },
      { UINT64_C( 5876569660509348395), UINT64_C( 1878607897840962306) } },
    { { UINT64_C( 5463762821535928714), UINT64_C(17340538587810836468) },
      { UINT64_C( 4903307573568791772), UINT64_C(14690845621075319511) },
      { UINT64_C( 4903307573568791772), UINT64_C(14690845621075319511) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_min_epu64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_min_epu64");
    easysimd_test_x86_assert_equal_u64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__m128i r = easysimd_mm_min_epu64(a, b);

    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t src[16];
    const uint16_t k;
    const int8_t a[16];
    const int8_t b[16];
    const int8_t r[16];
  } test_vec[] = {
    { {  INT8_C(  88),  INT8_C(  54),  INT8_C(  23), -INT8_C(   1), -INT8_C(  93),  INT8_C(  11),  INT8_C( 119), -INT8_C(  61),
         INT8_C(  74), -INT8_C(  96), -INT8_C(   9), -INT8_C(  74),  INT8_C( 104),      INT8_MIN, -INT8_C(  91),  INT8_C(  37) },
      UINT16_C(24058),
      {  INT8_C(  37),  INT8_C(  81),  INT8_C( 126), -INT8_C(  27), -INT8_C( 110), -INT8_C(   4),  INT8_C(  22), -INT8_C(   3),
         INT8_C(  92),  INT8_C(  40), -INT8_C(  64), -INT8_C( 126), -INT8_C( 119),  INT8_C(  24), -INT8_C(  72), -INT8_C(  96) },
      {  INT8_C(  24),  INT8_C(  92), -INT8_C(  84), -INT8_C( 113),  INT8_C(  31), -INT8_C(  10),  INT8_C(  47),  INT8_C(  22),
        -INT8_C(  84), -INT8_C( 105), -INT8_C( 105),  INT8_C(  82), -INT8_C(  68), -INT8_C( 111), -INT8_C(  81), -INT8_C(  31) },
      {  INT8_C(  88),  INT8_C(  81),  INT8_C(  23), -INT8_C( 113), -INT8_C( 110), -INT8_C(  10),  INT8_C(  22), -INT8_C(   3),
        -INT8_C(  84), -INT8_C(  96), -INT8_C( 105), -INT8_C( 126), -INT8_C( 119),      INT8_MIN, -INT8_C(  81),  INT8_C(  37) } },
    { { -INT8_C(  29),  INT8_C(  46), -INT8_C(  58),  INT8_C( 117),  INT8_C(  42), -INT8_C(  36),  INT8_C( 114), -INT8_C( 121),
         INT8_C(   4),  INT8_C(  50),  INT8_C(   9), -INT8_C( 115),  INT8_C(  75), -INT8_C(  63),  INT8_C(  46),  INT8_C(  99) },
      UINT16_C(55837),
      { -INT8_C(  14),  INT8_C(  60), -INT8_C(  48),  INT8_C(  33),  INT8_C(  83),  INT8_C( 124), -INT8_C(  72), -INT8_C(  22),
        -INT8_C(  50),  INT8_C( 116),  INT8_C( 123),  INT8_C( 126),  INT8_C(  86),  INT8_C(  94), -INT8_C(  84),  INT8_C(  28) },
      { -INT8_C(  45), -INT8_C(  42), -INT8_C(   7),  INT8_C(  69),  INT8_C(  93), -INT8_C(   3),  INT8_C( 120),  INT8_C( 102),
        -INT8_C( 117), -INT8_C(  61),  INT8_C(  40), -INT8_C(  71),  INT8_C(  38),  INT8_C(  69), -INT8_C( 109),  INT8_C(  24) },
      { -INT8_C(  45),  INT8_C(  46), -INT8_C(  48),  INT8_C(  33),  INT8_C(  83), -INT8_C(  36),  INT8_C( 114), -INT8_C( 121),
         INT8_C(   4), -INT8_C(  61),  INT8_C(   9), -INT8_C(  71),  INT8_C(  38), -INT8_C(  63), -INT8_C( 109),  INT8_C(  24) } },
    { { -INT8_C( 126),  INT8_C(  99),  INT8_C(  57), -INT8_C(  43), -INT8_C(  33), -INT8_C(  15), -INT8_C(  65), -INT8_C(  82),
         INT8_C( 101),  INT8_C(  58),  INT8_C(  44), -INT8_C(  69), -INT8_C( 103), -INT8_C(  40), -INT8_C(  40),  INT8_C( 108) },
      UINT16_C(53678),
      { -INT8_C(  78),  INT8_C(  12), -INT8_C(  50),  INT8_C(  42),  INT8_C( 114),  INT8_C(  89), -INT8_C(  19), -INT8_C( 102),
         INT8_C(  18),  INT8_C(  19), -INT8_C(  32), -INT8_C(  91),  INT8_C(  43),  INT8_C(  98),  INT8_C(   8),  INT8_C( 100) },
      {  INT8_C(  55), -INT8_C(  24),  INT8_C(  85), -INT8_C(  10), -INT8_C( 106), -INT8_C(  70),  INT8_C(  48), -INT8_C(  62),
         INT8_C( 118), -INT8_C(  55), -INT8_C( 102),  INT8_C(  78),  INT8_C(  54),  INT8_C(  72),  INT8_C(  31), -INT8_C(  24) },
      { -INT8_C( 126), -INT8_C(  24), -INT8_C(  50), -INT8_C(  10), -INT8_C(  33), -INT8_C(  70), -INT8_C(  65), -INT8_C( 102),
         INT8_C(  18),  INT8_C(  58),  INT8_C(  44), -INT8_C(  69),  INT8_C(  43), -INT8_C(  40),  INT8_C(   8), -INT8_C(  24) } },
    { {  INT8_C(  84), -INT8_C(  19),  INT8_C(  18), -INT8_C(  57),  INT8_C(  71), -INT8_C(   1),  INT8_C(  97),  INT8_C(  89),
         INT8_C(  18),  INT8_C(  65), -INT8_C(   1),  INT8_C(  61), -INT8_C(  93),  INT8_C(   7), -INT8_C(  95), -INT8_C(  38) },
      UINT16_C(63215),
      { -INT8_C(  48), -INT8_C( 123), -INT8_C(  80),  INT8_C(   1),  INT8_C(  71),  INT8_C(  38), -INT8_C(  54), -INT8_C(  31),
         INT8_C( 116),  INT8_C(   0),  INT8_C(  42), -INT8_C( 109), -INT8_C(  24),  INT8_C( 126), -INT8_C( 127), -INT8_C(   6) },
      {  INT8_C(  69), -INT8_C(  56), -INT8_C(   7), -INT8_C(  89),  INT8_C(  33),  INT8_C(  11), -INT8_C(  24),  INT8_C(  32),
         INT8_C(  72), -INT8_C( 116),  INT8_C(  40), -INT8_C(  23),  INT8_C( 102),  INT8_C(  23), -INT8_C(  33),  INT8_C(  55) },
      { -INT8_C(  48), -INT8_C( 123), -INT8_C(  80), -INT8_C(  89),  INT8_C(  71),  INT8_C(  11), -INT8_C(  54), -INT8_C(  31),
         INT8_C(  18), -INT8_C( 116),  INT8_C(  40),  INT8_C(  61), -INT8_C(  24),  INT8_C(  23), -INT8_C( 127), -INT8_C(   6) } },
    { { -INT8_C(  99), -INT8_C( 112),  INT8_C(  56), -INT8_C(  28), -INT8_C(  74),  INT8_C(   2), -INT8_C(  58),  INT8_C(  43),
         INT8_C(   3), -INT8_C(  16), -INT8_C(  66), -INT8_C(  21),  INT8_C( 110),  INT8_C(  63), -INT8_C(  26), -INT8_C(  76) },
      UINT16_C(57095),
      {  INT8_C(  91),  INT8_C(  41), -INT8_C(  21),  INT8_C(  67),  INT8_C(  73),  INT8_C(  51), -INT8_C(  49),  INT8_C( 113),
         INT8_C(  29),  INT8_C(  54), -INT8_C( 119), -INT8_C(   4),  INT8_C( 109),  INT8_C(  38), -INT8_C( 116), -INT8_C(  91) },
      {  INT8_C(  10),  INT8_C(  67), -INT8_C(  89), -INT8_C(  48),  INT8_C( 110), -INT8_C(  86), -INT8_C(  64),  INT8_C(  44),
        -INT8_C( 106),  INT8_C(  47),  INT8_C( 108),  INT8_C( 124), -INT8_C(  29),  INT8_C( 115),  INT8_C(  91),  INT8_C(  62) },
      {  INT8_C(  10),  INT8_C(  41), -INT8_C(  89), -INT8_C(  28), -INT8_C(  74),  INT8_C(   2), -INT8_C(  58),  INT8_C(  43),
        -INT8_C( 106),  INT8_C(  47), -INT8_C( 119), -INT8_C(   4), -INT8_C(  29),  INT8_C(  63), -INT8_C( 116), -INT8_C(  91) } },
    { { -INT8_C( 100),  INT8_C(  70), -INT8_C( 127), -INT8_C(  26),  INT8_C( 122),  INT8_C(  81),  INT8_C(  87), -INT8_C( 105),
        -INT8_C( 121), -INT8_C(  32), -INT8_C( 109), -INT8_C(  12),  INT8_C(   6),  INT8_C(  32), -INT8_C( 103),  INT8_C(  17) },
      UINT16_C(16483),
      { -INT8_C(  31), -INT8_C(  47), -INT8_C(  21), -INT8_C(  94), -INT8_C(   3), -INT8_C( 127), -INT8_C(  47),  INT8_C( 105),
        -INT8_C(   3), -INT8_C(  76), -INT8_C(  35),  INT8_C(  88), -INT8_C(  14),  INT8_C( 121), -INT8_C(  97),  INT8_C( 115) },
      {  INT8_C(  95),  INT8_C(  25), -INT8_C(  60), -INT8_C(  73), -INT8_C(  80),  INT8_C(  75), -INT8_C( 105),  INT8_C(  67),
         INT8_C(  63), -INT8_C(  98),  INT8_C(  99), -INT8_C(  40), -INT8_C(  81), -INT8_C(  58),  INT8_C(  25), -INT8_C( 112) },
      { -INT8_C(  31), -INT8_C(  47), -INT8_C( 127), -INT8_C(  26),  INT8_C( 122), -INT8_C( 127), -INT8_C( 105), -INT8_C( 105),
        -INT8_C( 121), -INT8_C(  32), -INT8_C( 109), -INT8_C(  12),  INT8_C(   6),  INT8_C(  32), -INT8_C(  97),  INT8_C(  17) } },
    { { -INT8_C( 105),  INT8_C(   4),  INT8_C(  50), -INT8_C( 107), -INT8_C( 123),  INT8_C(   3), -INT8_C(   2), -INT8_C( 126),
        -INT8_C(  73), -INT8_C(  37), -INT8_C(  38), -INT8_C(  87),  INT8_C(  85),  INT8_C( 121),  INT8_C(  29), -INT8_C(  76) },
      UINT16_C(57746),
      {  INT8_C( 107),  INT8_C(  66),  INT8_C(  45),  INT8_C(   3), -INT8_C( 122),  INT8_C( 108), -INT8_C(  95), -INT8_C(  23),
         INT8_C(  69),  INT8_C(  80), -INT8_C(  80),  INT8_C(  94), -INT8_C(  32),  INT8_C(  71),  INT8_C(  98),  INT8_C(  19) },
      { -INT8_C(  36), -INT8_C(  25),  INT8_C(  22), -INT8_C(  37),  INT8_C( 105), -INT8_C(  50), -INT8_C(  74),  INT8_C(  67),
         INT8_C( 119),  INT8_C(  11), -INT8_C(  67), -INT8_C( 108), -INT8_C(  64),  INT8_C(  79),  INT8_C( 118),  INT8_C(  43) },
      { -INT8_C( 105), -INT8_C(  25),  INT8_C(  50), -INT8_C( 107), -INT8_C( 122),  INT8_C(   3), -INT8_C(   2), -INT8_C(  23),
         INT8_C(  69), -INT8_C(  37), -INT8_C(  38), -INT8_C(  87),  INT8_C(  85),  INT8_C(  71),  INT8_C(  98),  INT8_C(  19) } },
    { { -INT8_C( 110), -INT8_C(  93),  INT8_C(  46),  INT8_C(  24),  INT8_C(  15), -INT8_C(  49),  INT8_C(   1),  INT8_C(  84),
         INT8_C(  31), -INT8_C(  79), -INT8_C(  78),  INT8_C(   0), -INT8_C(   7),  INT8_C(  20),  INT8_C(  19), -INT8_C(  43) },
      UINT16_C(10747),
      { -INT8_C(  80),  INT8_C( 100), -INT8_C(   9),  INT8_C( 103), -INT8_C(  88),  INT8_C( 111),  INT8_C( 114),  INT8_C( 101),
         INT8_C(   3),  INT8_C(  50), -INT8_C(  76),  INT8_C( 121),  INT8_C(  94),  INT8_C(  70),  INT8_C(  28), -INT8_C( 116) },
      {  INT8_C(  94),  INT8_C(  44),  INT8_C(  92),  INT8_C(  96),      INT8_MIN,  INT8_C( 123),  INT8_C(  17),  INT8_C(  51),
         INT8_C( 123),  INT8_C(  10),  INT8_C(  71), -INT8_C( 114), -INT8_C(  32),  INT8_C(  67), -INT8_C(  72), -INT8_C( 112) },
      { -INT8_C(  80),  INT8_C(  44),  INT8_C(  46),  INT8_C(  96),      INT8_MIN,  INT8_C( 111),  INT8_C(  17),  INT8_C(  51),
         INT8_C(   3), -INT8_C(  79), -INT8_C(  78), -INT8_C( 114), -INT8_C(   7),  INT8_C(  67),  INT8_C(  19), -INT8_C(  43) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi8(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_epi8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_epi8");
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
    easysimd__m128i r = easysimd_mm_mask_min_epi8(src, k, a, b);

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
test_easysimd_mm_maskz_min_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const int8_t a[16];
    const int8_t b[16];
    const int8_t r[16];
  } test_vec[] = {
    { UINT16_C( 1307),
      { -INT8_C(  43), -INT8_C( 113),  INT8_C(  81), -INT8_C(  10),  INT8_C(  29),  INT8_C(  24),  INT8_C(  80), -INT8_C(  72),
        -INT8_C(  38),  INT8_C(  63), -INT8_C(  58), -INT8_C( 116),  INT8_C(   6), -INT8_C(   7),  INT8_C(   8),  INT8_C(  87) },
      { -INT8_C(  47), -INT8_C(  70),  INT8_C(  17),  INT8_C(   4),  INT8_C(   0), -INT8_C(  48),  INT8_C(  74), -INT8_C(  13),
        -INT8_C(  82), -INT8_C( 105), -INT8_C(   3), -INT8_C(  71), -INT8_C( 124),  INT8_C(  25), -INT8_C(  65),  INT8_C(  89) },
      { -INT8_C(  47), -INT8_C( 113),  INT8_C(   0), -INT8_C(  10),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  82),  INT8_C(   0), -INT8_C(  58),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C( 4264),
      {  INT8_C(  80), -INT8_C(  58),  INT8_C(  40), -INT8_C(  96),  INT8_C( 126),  INT8_C(   2), -INT8_C(  33),  INT8_C(  69),
        -INT8_C( 113), -INT8_C(  26),  INT8_C(  62), -INT8_C( 105),  INT8_C(  61),  INT8_C(  15),  INT8_C(  81),  INT8_C(  78) },
      {  INT8_C(  19),  INT8_C(  81),  INT8_C(  30),  INT8_C(  94),  INT8_C(  68), -INT8_C(  52), -INT8_C(  11),  INT8_C(  66),
        -INT8_C( 122),  INT8_C( 121),  INT8_C(  91),  INT8_C(  69), -INT8_C(  45),  INT8_C(   3),  INT8_C(  85),  INT8_C(  35) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  96),  INT8_C(   0), -INT8_C(  52),  INT8_C(   0),  INT8_C(  66),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  45),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(32201),
      { -INT8_C(  61),  INT8_C(  72),      INT8_MIN, -INT8_C(  93), -INT8_C( 115),  INT8_C(  15), -INT8_C( 119), -INT8_C(  53),
        -INT8_C(  90), -INT8_C(  58), -INT8_C(  38), -INT8_C(   9),  INT8_C(  20), -INT8_C(  18),  INT8_C(  72),  INT8_C(  50) },
      {  INT8_C(  76), -INT8_C( 115), -INT8_C(   1),  INT8_C(  65), -INT8_C(  49), -INT8_C( 123), -INT8_C(  69),  INT8_C(  42),
        -INT8_C(  54), -INT8_C( 114),  INT8_C(  45),  INT8_C(  31), -INT8_C(  79), -INT8_C(   9), -INT8_C( 100),  INT8_C( 116) },
      { -INT8_C(  61),  INT8_C(   0),  INT8_C(   0), -INT8_C(  93),  INT8_C(   0),  INT8_C(   0), -INT8_C( 119), -INT8_C(  53),
        -INT8_C(  90),  INT8_C(   0), -INT8_C(  38), -INT8_C(   9), -INT8_C(  79), -INT8_C(  18), -INT8_C( 100),  INT8_C(   0) } },
    { UINT16_C( 7231),
      {  INT8_C(  23), -INT8_C(  52),  INT8_C(  43), -INT8_C(  96), -INT8_C( 105), -INT8_C(  47),  INT8_C( 102),  INT8_C( 113),
        -INT8_C(  56),  INT8_C( 122),  INT8_C(  95),  INT8_C(  17), -INT8_C(  83), -INT8_C(  85), -INT8_C(  98), -INT8_C(  84) },
      { -INT8_C(  19),  INT8_C( 109),  INT8_C(  49), -INT8_C(  88), -INT8_C( 105), -INT8_C(   5),  INT8_C(  54), -INT8_C(  60),
         INT8_C(  26), -INT8_C(  25), -INT8_C(  69), -INT8_C(  74),  INT8_C(  91), -INT8_C(   6), -INT8_C(  45),  INT8_C( 115) },
      { -INT8_C(  19), -INT8_C(  52),  INT8_C(  43), -INT8_C(  96), -INT8_C( 105), -INT8_C(  47),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  69), -INT8_C(  74), -INT8_C(  83),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(65222),
      {  INT8_C(  19),  INT8_C(  93), -INT8_C(  48),  INT8_C( 122), -INT8_C(  49), -INT8_C( 104), -INT8_C(  12),  INT8_C(  46),
        -INT8_C(  87), -INT8_C(  95), -INT8_C(  38),  INT8_C(  71),  INT8_C(  77), -INT8_C(  57), -INT8_C(  76),  INT8_C( 126) },
      {  INT8_C( 111),  INT8_C(  75),  INT8_C( 121), -INT8_C(  91),  INT8_C(  16), -INT8_C( 109), -INT8_C( 116), -INT8_C(  53),
         INT8_C(  74), -INT8_C(  25), -INT8_C(  58),  INT8_C(  29),  INT8_C(  90), -INT8_C( 116),  INT8_C(  27),  INT8_C( 110) },
      {  INT8_C(   0),  INT8_C(  75), -INT8_C(  48),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 116), -INT8_C(  53),
         INT8_C(   0), -INT8_C(  95), -INT8_C(  58),  INT8_C(  29),  INT8_C(  77), -INT8_C( 116), -INT8_C(  76),  INT8_C( 110) } },
    { UINT16_C(60394),
      { -INT8_C(  24), -INT8_C(  71), -INT8_C( 124), -INT8_C(  36), -INT8_C(  25),  INT8_C(  45),  INT8_C( 126), -INT8_C(  63),
         INT8_C( 117), -INT8_C(  53), -INT8_C( 120),  INT8_C(  41),  INT8_C(  74), -INT8_C(   9),  INT8_C( 117), -INT8_C(  61) },
      { -INT8_C( 100), -INT8_C( 123),  INT8_C(  87),  INT8_C(  40),  INT8_C(  80), -INT8_C(  95),  INT8_C(  16),  INT8_C(  22),
        -INT8_C(  66),  INT8_C( 106), -INT8_C(  93), -INT8_C(  39), -INT8_C(  40), -INT8_C( 115), -INT8_C(  59), -INT8_C(  64) },
      {  INT8_C(   0), -INT8_C( 123),  INT8_C(   0), -INT8_C(  36),  INT8_C(   0), -INT8_C(  95),  INT8_C(  16), -INT8_C(  63),
        -INT8_C(  66), -INT8_C(  53),  INT8_C(   0), -INT8_C(  39),  INT8_C(   0), -INT8_C( 115), -INT8_C(  59), -INT8_C(  64) } },
    { UINT16_C(18758),
      { -INT8_C(  99),  INT8_C(  45),  INT8_C( 118),  INT8_C(  27), -INT8_C(  17), -INT8_C(  21), -INT8_C(  26),  INT8_C( 119),
         INT8_C(  21),  INT8_C(  48),  INT8_C( 111), -INT8_C( 118), -INT8_C(  12),  INT8_C(  11),  INT8_C(  15),  INT8_C(  75) },
      {  INT8_C(  52),  INT8_C(  95), -INT8_C(  20),  INT8_C(  68),  INT8_C( 118), -INT8_C(  86), -INT8_C(  82),  INT8_C(  25),
        -INT8_C( 125), -INT8_C( 121), -INT8_C(  90),  INT8_C(  72),  INT8_C(  71), -INT8_C(  20), -INT8_C( 111), -INT8_C(  28) },
      {  INT8_C(   0),  INT8_C(  45), -INT8_C(  20),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  82),  INT8_C(   0),
        -INT8_C( 125),  INT8_C(   0),  INT8_C(   0), -INT8_C( 118),  INT8_C(   0),  INT8_C(   0), -INT8_C( 111),  INT8_C(   0) } },
    { UINT16_C( 2073),
      { -INT8_C(   1),  INT8_C(   8), -INT8_C(  13), -INT8_C(  26),      INT8_MIN,  INT8_C(   8),  INT8_C(  22), -INT8_C(  17),
        -INT8_C( 110),  INT8_C(  10), -INT8_C(   6), -INT8_C(  95),  INT8_C(  85),  INT8_C(  46),  INT8_C(   1),  INT8_C(  65) },
      {  INT8_C( 114),  INT8_C( 119), -INT8_C(  21),  INT8_C(  33), -INT8_C( 112),  INT8_C( 111), -INT8_C(  88),  INT8_C(  54),
        -INT8_C(  73), -INT8_C(  17),  INT8_C(  34),  INT8_C(  73), -INT8_C(  44),  INT8_C(  59),  INT8_C(  81), -INT8_C(  45) },
      { -INT8_C(   1),  INT8_C(   0),  INT8_C(   0), -INT8_C(  26),      INT8_MIN,  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  95),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_epi8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__m128i r = easysimd_mm_maskz_min_epi8(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { { -INT16_C( 20569),  INT16_C( 20471),  INT16_C( 27166),  INT16_C(  8884),  INT16_C( 27036), -INT16_C(  1381), -INT16_C( 18257),  INT16_C(  3719) },
      UINT8_C(228),
      {  INT16_C( 28387),  INT16_C( 24164), -INT16_C( 26753), -INT16_C( 29990),  INT16_C( 26847),  INT16_C(  8810), -INT16_C(  1504), -INT16_C( 12087) },
      {  INT16_C(  6642),  INT16_C( 23790),  INT16_C(  4301),  INT16_C( 14072), -INT16_C(  3156),  INT16_C( 25830), -INT16_C(  2950),  INT16_C( 23880) },
      { -INT16_C( 20569),  INT16_C( 20471), -INT16_C( 26753),  INT16_C(  8884),  INT16_C( 27036),  INT16_C(  8810), -INT16_C(  2950), -INT16_C( 12087) } },
    { { -INT16_C( 21406), -INT16_C(  7749), -INT16_C( 27324),  INT16_C(  9067), -INT16_C( 10754),  INT16_C(  7749),  INT16_C(  3792), -INT16_C( 15634) },
      UINT8_C( 39),
      {  INT16_C(  7901), -INT16_C(  4619),  INT16_C( 11030),  INT16_C(  2457), -INT16_C(   751),  INT16_C(  1411), -INT16_C(  8123), -INT16_C(  3481) },
      {  INT16_C( 18844),  INT16_C( 12598),  INT16_C( 22964), -INT16_C( 30161),  INT16_C( 20126), -INT16_C( 21414),  INT16_C(  7228),  INT16_C(  6612) },
      {  INT16_C(  7901), -INT16_C(  4619),  INT16_C( 11030),  INT16_C(  9067), -INT16_C( 10754), -INT16_C( 21414),  INT16_C(  3792), -INT16_C( 15634) } },
    { { -INT16_C( 14022),  INT16_C( 20487), -INT16_C( 24332),  INT16_C(  1626), -INT16_C(  8802), -INT16_C(  7413),  INT16_C( 29630),  INT16_C( 23253) },
      UINT8_C(188),
      { -INT16_C( 29941),  INT16_C( 25712), -INT16_C(  1349),  INT16_C(  2306), -INT16_C( 20652),  INT16_C( 28741),  INT16_C( 24451),  INT16_C( 19626) },
      { -INT16_C(  1178),  INT16_C(  1600),  INT16_C( 18005),  INT16_C( 12964), -INT16_C( 30638), -INT16_C( 14864),  INT16_C( 19037),  INT16_C( 27009) },
      { -INT16_C( 14022),  INT16_C( 20487), -INT16_C(  1349),  INT16_C(  2306), -INT16_C( 30638), -INT16_C( 14864),  INT16_C( 29630),  INT16_C( 19626) } },
    { { -INT16_C(  3626), -INT16_C( 28211), -INT16_C( 12052),  INT16_C( 16538), -INT16_C(  8321),  INT16_C(   689),  INT16_C( 23358), -INT16_C( 23474) },
      UINT8_C( 86),
      { -INT16_C( 21618), -INT16_C( 10837), -INT16_C(  8625), -INT16_C( 10457), -INT16_C(  4914),  INT16_C(  6453), -INT16_C( 24979),  INT16_C( 24303) },
      { -INT16_C( 32661),  INT16_C( 15178), -INT16_C( 29926), -INT16_C(  1606), -INT16_C( 17348), -INT16_C( 26824), -INT16_C(  9206), -INT16_C( 26130) },
      { -INT16_C(  3626), -INT16_C( 10837), -INT16_C( 29926),  INT16_C( 16538), -INT16_C( 17348),  INT16_C(   689), -INT16_C( 24979), -INT16_C( 23474) } },
    { { -INT16_C( 26233), -INT16_C( 10386), -INT16_C( 27273),  INT16_C( 18094), -INT16_C(  7295), -INT16_C(  4513),  INT16_C( 20097), -INT16_C(  4788) },
      UINT8_C(206),
      {  INT16_C( 10391),  INT16_C(  8936), -INT16_C(  7709), -INT16_C( 24738), -INT16_C(  2791), -INT16_C(  2390),  INT16_C( 17379),  INT16_C( 32125) },
      {  INT16_C( 21681),  INT16_C( 18164),  INT16_C( 14851), -INT16_C(  6457), -INT16_C( 19047), -INT16_C(  6296),  INT16_C( 21761), -INT16_C( 26443) },
      { -INT16_C( 26233),  INT16_C(  8936), -INT16_C(  7709), -INT16_C( 24738), -INT16_C(  7295), -INT16_C(  4513),  INT16_C( 17379), -INT16_C( 26443) } },
    { { -INT16_C( 25219),  INT16_C( 24762),  INT16_C(  6271), -INT16_C( 26624), -INT16_C( 22002), -INT16_C(  3698),  INT16_C(  3309), -INT16_C( 24978) },
      UINT8_C( 96),
      { -INT16_C(  7069), -INT16_C( 25245),  INT16_C( 19115),  INT16_C( 24631),  INT16_C(  7858),  INT16_C(  1889), -INT16_C(  1324),  INT16_C( 29060) },
      { -INT16_C(  6732), -INT16_C( 12816), -INT16_C( 30235), -INT16_C( 28709), -INT16_C( 13289),  INT16_C(  9084),  INT16_C(  6715), -INT16_C( 24956) },
      { -INT16_C( 25219),  INT16_C( 24762),  INT16_C(  6271), -INT16_C( 26624), -INT16_C( 22002),  INT16_C(  1889), -INT16_C(  1324), -INT16_C( 24978) } },
    { { -INT16_C(  6146), -INT16_C( 22213),  INT16_C( 29233), -INT16_C(  7415),  INT16_C( 27281),  INT16_C( 26090),  INT16_C( 28516),  INT16_C(  6614) },
      UINT8_C( 84),
      { -INT16_C(  6457),  INT16_C( 20537), -INT16_C( 14143), -INT16_C( 29337), -INT16_C( 29884),  INT16_C( 24264),  INT16_C( 26127), -INT16_C(  2468) },
      {  INT16_C(  1442),  INT16_C(  5160),  INT16_C(  2830),  INT16_C( 30885),  INT16_C(  2806),  INT16_C( 26077), -INT16_C(  2335), -INT16_C( 22343) },
      { -INT16_C(  6146), -INT16_C( 22213), -INT16_C( 14143), -INT16_C(  7415), -INT16_C( 29884),  INT16_C( 26090), -INT16_C(  2335),  INT16_C(  6614) } },
    { { -INT16_C(  3364), -INT16_C( 25096),  INT16_C( 24506), -INT16_C(   470), -INT16_C(  3094), -INT16_C(  1700), -INT16_C( 18343), -INT16_C(  1040) },
      UINT8_C(189),
      {  INT16_C(  4120),  INT16_C(  9163),  INT16_C( 17333), -INT16_C( 16359),  INT16_C( 32288),  INT16_C(  5793),  INT16_C( 18743),  INT16_C( 10738) },
      { -INT16_C( 28863), -INT16_C( 24349), -INT16_C(  7750), -INT16_C( 21109), -INT16_C( 31683), -INT16_C(  2810),  INT16_C(   628), -INT16_C( 29518) },
      { -INT16_C( 28863), -INT16_C( 25096), -INT16_C(  7750), -INT16_C( 21109), -INT16_C( 31683), -INT16_C(  2810), -INT16_C( 18343), -INT16_C( 29518) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_epi16");
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
    easysimd__m128i r = easysimd_mm_mask_min_epi16(src, k, a, b);

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
test_easysimd_mm_maskz_min_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C( 68),
      { -INT16_C( 18108),  INT16_C( 19908), -INT16_C( 19504), -INT16_C(  9505), -INT16_C( 32339), -INT16_C(  9168),  INT16_C( 29058), -INT16_C(  1714) },
      {  INT16_C( 28509), -INT16_C( 13175), -INT16_C( 16617),  INT16_C(  1923), -INT16_C( 13087),  INT16_C(  7387), -INT16_C( 20963),  INT16_C( 25184) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 19504),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 20963),  INT16_C(     0) } },
    { UINT8_C(104),
      { -INT16_C( 20700), -INT16_C( 10440),  INT16_C(  4750),  INT16_C(  3973),  INT16_C( 24898), -INT16_C( 19311), -INT16_C( 30033),  INT16_C(  7953) },
      { -INT16_C(  8941), -INT16_C( 11722),  INT16_C( 15712),  INT16_C( 11699), -INT16_C( 12264), -INT16_C( 14518), -INT16_C( 21456),  INT16_C( 21807) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3973),  INT16_C(     0), -INT16_C( 19311), -INT16_C( 30033),  INT16_C(     0) } },
    { UINT8_C( 91),
      {  INT16_C( 11367),  INT16_C( 31210), -INT16_C(  1615),  INT16_C(  4796),  INT16_C( 28811),  INT16_C(  5570), -INT16_C(  7807),  INT16_C( 24105) },
      { -INT16_C(  1257),  INT16_C( 21950), -INT16_C(  5201),  INT16_C( 32621),  INT16_C( 13366), -INT16_C(  7505),  INT16_C(  1123), -INT16_C( 13762) },
      { -INT16_C(  1257),  INT16_C( 21950),  INT16_C(     0),  INT16_C(  4796),  INT16_C( 13366),  INT16_C(     0), -INT16_C(  7807),  INT16_C(     0) } },
    { UINT8_C( 49),
      {  INT16_C( 17448),  INT16_C(  8674), -INT16_C(  2816),  INT16_C( 28844), -INT16_C( 15689), -INT16_C( 26383),  INT16_C( 20459), -INT16_C(  6481) },
      {  INT16_C(  1037), -INT16_C(  1643),  INT16_C(  5234), -INT16_C( 22993),  INT16_C(  4548), -INT16_C( 14326), -INT16_C( 11185),  INT16_C( 30713) },
      {  INT16_C(  1037),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 15689), -INT16_C( 26383),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 24),
      { -INT16_C( 26148), -INT16_C( 12008), -INT16_C( 30651),  INT16_C(  1928),  INT16_C(  8313), -INT16_C( 14094), -INT16_C(  9777), -INT16_C( 11050) },
      { -INT16_C( 12434), -INT16_C( 31930), -INT16_C(  4866),  INT16_C(  3911),  INT16_C(  4086), -INT16_C( 13473), -INT16_C( 10743), -INT16_C(  6685) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1928),  INT16_C(  4086),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(111),
      { -INT16_C( 18692), -INT16_C( 31563), -INT16_C( 17346),  INT16_C( 24318), -INT16_C( 14673), -INT16_C( 30675),  INT16_C(   412),  INT16_C( 27638) },
      {  INT16_C( 31047),  INT16_C( 13417),  INT16_C( 31168), -INT16_C( 12246), -INT16_C(  2600), -INT16_C( 20775), -INT16_C( 16679), -INT16_C( 10978) },
      { -INT16_C( 18692), -INT16_C( 31563), -INT16_C( 17346), -INT16_C( 12246),  INT16_C(     0), -INT16_C( 30675), -INT16_C( 16679),  INT16_C(     0) } },
    { UINT8_C(116),
      {  INT16_C( 22995), -INT16_C( 28750),  INT16_C(  4183),  INT16_C(  7742), -INT16_C( 14787),  INT16_C( 16314),  INT16_C(  9917),  INT16_C( 13958) },
      { -INT16_C( 17777),  INT16_C(  2295), -INT16_C( 14363), -INT16_C(  9504), -INT16_C( 28768),  INT16_C( 24243), -INT16_C( 30547), -INT16_C( 32558) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 14363),  INT16_C(     0), -INT16_C( 28768),  INT16_C( 16314), -INT16_C( 30547),  INT16_C(     0) } },
    { UINT8_C(226),
      {  INT16_C(  3972), -INT16_C( 27591),  INT16_C( 22350),  INT16_C(  5329),  INT16_C(  4114),  INT16_C( 14545),  INT16_C(  2199),  INT16_C( 20935) },
      { -INT16_C( 12033), -INT16_C( 14794),  INT16_C(  4528),  INT16_C( 16230), -INT16_C( 15164),  INT16_C( 19948),  INT16_C( 27798),  INT16_C(  6703) },
      {  INT16_C(     0), -INT16_C( 27591),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 14545),  INT16_C(  2199),  INT16_C(  6703) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_maskz_min_epi16(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(   944734958), -INT32_C(   511194687),  INT32_C(  2146969672), -INT32_C(  1297487247) },
      UINT8_C(122),
      {  INT32_C(  1848923020),  INT32_C(  1655431646), -INT32_C(   371744281), -INT32_C(   772054189) },
      { -INT32_C(   577584365),  INT32_C(  1931834186),  INT32_C(  1457824875), -INT32_C(   623863987) },
      { -INT32_C(   944734958),  INT32_C(  1655431646),  INT32_C(  2146969672), -INT32_C(   772054189) } },
    { { -INT32_C(   934804246), -INT32_C(   869600283), -INT32_C(   390725228),  INT32_C(  2025435493) },
      UINT8_C(116),
      { -INT32_C(  1094757045),  INT32_C(   522859130), -INT32_C(  1385398250),  INT32_C(  1419200080) },
      { -INT32_C(  2110169202), -INT32_C(  1961491062),  INT32_C(  1844510395),  INT32_C(    48327095) },
      { -INT32_C(   934804246), -INT32_C(   869600283), -INT32_C(  1385398250),  INT32_C(  2025435493) } },
    { {  INT32_C(   968990910), -INT32_C(   380048430), -INT32_C(  1147747221), -INT32_C(  1727058421) },
      UINT8_C(141),
      {  INT32_C(  1293359944),  INT32_C(   805937970), -INT32_C(    51874157), -INT32_C(   121902505) },
      { -INT32_C(  1765083989),  INT32_C(   268546892),  INT32_C(  2015083594),  INT32_C(   319141323) },
      { -INT32_C(  1765083989), -INT32_C(   380048430), -INT32_C(    51874157), -INT32_C(   121902505) } },
    { {  INT32_C(    39919056),  INT32_C(  1395878592),  INT32_C(   927996896), -INT32_C(  1322317051) },
         UINT8_MAX,
      { -INT32_C(  1387575302),  INT32_C(   100096841),  INT32_C(   751923063), -INT32_C(  1828920203) },
      { -INT32_C(  1353515195),  INT32_C(  1301259570),  INT32_C(     5425141), -INT32_C(   251722762) },
      { -INT32_C(  1387575302),  INT32_C(   100096841),  INT32_C(     5425141), -INT32_C(  1828920203) } },
    { { -INT32_C(  1801565621),  INT32_C(   496604582),  INT32_C(  2051631621), -INT32_C(  1811135153) },
      UINT8_C( 68),
      {  INT32_C(    58082398), -INT32_C(  1694907437),  INT32_C(   428996886),  INT32_C(  1130660345) },
      { -INT32_C(  1242957793), -INT32_C(    54917486), -INT32_C(  1790233521), -INT32_C(  1629888448) },
      { -INT32_C(  1801565621),  INT32_C(   496604582), -INT32_C(  1790233521), -INT32_C(  1811135153) } },
    { { -INT32_C(   140423132),  INT32_C(   697473555), -INT32_C(  1924979820), -INT32_C(   992958556) },
      UINT8_C(160),
      { -INT32_C(  1070433862),  INT32_C(  1729113651),  INT32_C(  1520936314), -INT32_C(   830585474) },
      { -INT32_C(  2115865114),  INT32_C(   706022151), -INT32_C(   154164658),  INT32_C(   748065650) },
      { -INT32_C(   140423132),  INT32_C(   697473555), -INT32_C(  1924979820), -INT32_C(   992958556) } },
    { {  INT32_C(  1072547852),  INT32_C(  1889992182),  INT32_C(   550129058),  INT32_C(  2028947602) },
      UINT8_C(190),
      { -INT32_C(   591005231), -INT32_C(  1339297778),  INT32_C(  1378034111),  INT32_C(  2136887223) },
      {  INT32_C(   964009276), -INT32_C(  1864636861), -INT32_C(   115147600), -INT32_C(  1128817941) },
      {  INT32_C(  1072547852), -INT32_C(  1864636861), -INT32_C(   115147600), -INT32_C(  1128817941) } },
    { { -INT32_C(  1567064940),  INT32_C(   726909804), -INT32_C(  1686276380),  INT32_C(    35314629) },
      UINT8_C(120),
      {  INT32_C(  1991981968),  INT32_C(   321276695),  INT32_C(   150871917),  INT32_C(  1402780374) },
      {  INT32_C(   381632082),  INT32_C(   133884817),  INT32_C(  1154324072),  INT32_C(  1086115760) },
      { -INT32_C(  1567064940),  INT32_C(   726909804), -INT32_C(  1686276380),  INT32_C(  1086115760) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_epi32");
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
    easysimd__m128i r = easysimd_mm_mask_min_epi32(src, k, a, b);

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
test_easysimd_mm_maskz_min_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(124),
      { -INT32_C(  1060458904), -INT32_C(  1865228673), -INT32_C(  1205400912), -INT32_C(  1581811503) },
      { -INT32_C(  1068335697), -INT32_C(  1484484125), -INT32_C(   365047426),  INT32_C(  1768314624) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1205400912), -INT32_C(  1581811503) } },
    { UINT8_C(  5),
      {  INT32_C(   260319536),  INT32_C(    96408827), -INT32_C(  1261013189), -INT32_C(  1419544530) },
      {  INT32_C(  1536041930),  INT32_C(  2044278183), -INT32_C(   914766734),  INT32_C(  1540284970) },
      {  INT32_C(   260319536),  INT32_C(           0), -INT32_C(  1261013189),  INT32_C(           0) } },
    { UINT8_C( 11),
      {  INT32_C(  1728473682), -INT32_C(  1599993047),  INT32_C(  1523472098), -INT32_C(   601589319) },
      {  INT32_C(  1015316231),  INT32_C(   498007385),  INT32_C(  1497855862), -INT32_C(  1755012539) },
      {  INT32_C(  1015316231), -INT32_C(  1599993047),  INT32_C(           0), -INT32_C(  1755012539) } },
    { UINT8_C( 12),
      {  INT32_C(  1983250027), -INT32_C(   145173087),  INT32_C(   481407651),  INT32_C(  1445170646) },
      {  INT32_C(   246374161), -INT32_C(  2071606259), -INT32_C(  1228284397), -INT32_C(  1379770046) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1228284397), -INT32_C(  1379770046) } },
    { UINT8_C( 95),
      { -INT32_C(   855628552),  INT32_C(   795998332),  INT32_C(   906333609),  INT32_C(   273177521) },
      { -INT32_C(   702654966), -INT32_C(  1175805221), -INT32_C(   839081876),  INT32_C(  1512876386) },
      { -INT32_C(   855628552), -INT32_C(  1175805221), -INT32_C(   839081876),  INT32_C(   273177521) } },
    { UINT8_C(205),
      {  INT32_C(   625616941),  INT32_C(   651065753), -INT32_C(   623442818), -INT32_C(  1545279412) },
      { -INT32_C(  1468089594),  INT32_C(  1158953125), -INT32_C(   576200396),  INT32_C(  1001062926) },
      { -INT32_C(  1468089594),  INT32_C(           0), -INT32_C(   623442818), -INT32_C(  1545279412) } },
    { UINT8_C( 42),
      {  INT32_C(  1858298101),  INT32_C(   854386990), -INT32_C(  1451243839),  INT32_C(  1722753707) },
      { -INT32_C(   670345312),  INT32_C(  1292652652), -INT32_C(   111416585), -INT32_C(  1977379179) },
      {  INT32_C(           0),  INT32_C(   854386990),  INT32_C(           0), -INT32_C(  1977379179) } },
    { UINT8_C(246),
      { -INT32_C(   802883354), -INT32_C(  1416538140), -INT32_C(   128566570),  INT32_C(  1083751657) },
      {  INT32_C(   397177287),  INT32_C(  1729034877), -INT32_C(   319027115),  INT32_C(   316835371) },
      {  INT32_C(           0), -INT32_C(  1416538140), -INT32_C(   319027115),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_min_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 3545702636801390602),  INT64_C( 4550559686637204220) },
      UINT8_C( 19),
      { -INT64_C( 1929321506548656667), -INT64_C( 5668951964410020487) },
      { -INT64_C( 2637237285213080435),  INT64_C( 5027995616618756575) },
      { -INT64_C( 2637237285213080435), -INT64_C( 5668951964410020487) } },
    { { -INT64_C( 7885435600397570980), -INT64_C( 8451515986454304821) },
      UINT8_C(162),
      { -INT64_C( 7732637284888218581), -INT64_C( 7470845966569675537) },
      { -INT64_C( 5374970775944954546),  INT64_C( 5759889674561875231) },
      { -INT64_C( 7885435600397570980), -INT64_C( 7470845966569675537) } },
    { {  INT64_C( 5846184012508379954), -INT64_C( 7063164503641877194) },
      UINT8_C(150),
      {  INT64_C( 1157384022606035628), -INT64_C( 8719668941964419955) },
      { -INT64_C( 4491151319798084956), -INT64_C( 5426648400201314688) },
      {  INT64_C( 5846184012508379954), -INT64_C( 8719668941964419955) } },
    { {  INT64_C( 4438776474260264422), -INT64_C(  729295821974483697) },
      UINT8_C(143),
      {  INT64_C( 2703561752586120138),  INT64_C( 2494680067023412491) },
      { -INT64_C( 8266058322766846074),  INT64_C( 5812971217503495191) },
      { -INT64_C( 8266058322766846074),  INT64_C( 2494680067023412491) } },
    { { -INT64_C( 1406856045794171587),  INT64_C( 1145393740167489397) },
      UINT8_C(104),
      {  INT64_C( 4025573288228440503), -INT64_C( 8528958269576840813) },
      {  INT64_C( 1778172472159041377),  INT64_C(  911046904754827514) },
      { -INT64_C( 1406856045794171587),  INT64_C( 1145393740167489397) } },
    { {  INT64_C( 1568571545491787002), -INT64_C( 1092934833266851160) },
      UINT8_C( 21),
      {  INT64_C( 5084738623786854720),  INT64_C( 4389223415347399743) },
      { -INT64_C( 4769079754345253255),  INT64_C( 3638976610306890830) },
      { -INT64_C( 4769079754345253255), -INT64_C( 1092934833266851160) } },
    { { -INT64_C( 2052698443221563802),  INT64_C( 4598664255625174565) },
      UINT8_C( 51),
      {  INT64_C( 6174764085288003728), -INT64_C( 6621846583853507209) },
      { -INT64_C(  522649714348763327), -INT64_C( 5264991472037505793) },
      { -INT64_C(  522649714348763327), -INT64_C( 6621846583853507209) } },
    { { -INT64_C(    3435391363752858), -INT64_C(  780865789934486808) },
      UINT8_C( 37),
      { -INT64_C( 6656892752841699470), -INT64_C( 6693897950180321098) },
      { -INT64_C( 1852058426852788071),  INT64_C( 7912982856368574941) },
      { -INT64_C( 6656892752841699470), -INT64_C(  780865789934486808) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_epi64");
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
    easysimd__m128i r = easysimd_mm_mask_min_epi64(src, k, a, b);

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
test_easysimd_mm_maskz_min_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(126),
      { -INT64_C( 5966127717691432441),  INT64_C( 3884309341478988900) },
      {  INT64_C( 1500290266818674307), -INT64_C( 6333096121908281833) },
      {  INT64_C(                   0), -INT64_C( 6333096121908281833) } },
    { UINT8_C( 52),
      {  INT64_C( 7417802671717287551), -INT64_C(  701610495447526934) },
      {  INT64_C( 5855831336371273747),  INT64_C( 9069575643157684823) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(160),
      {  INT64_C(  919368724356139652), -INT64_C( 8728263296080903573) },
      {  INT64_C( 3632834041089690620),  INT64_C( 8401656355327316829) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(183),
      { -INT64_C( 7989739718555054989), -INT64_C( 3190556486063887934) },
      {  INT64_C( 5386117371421510903), -INT64_C( 4329736807765067899) },
      { -INT64_C( 7989739718555054989), -INT64_C( 4329736807765067899) } },
    { UINT8_C(251),
      {  INT64_C( 9087027738896397979),  INT64_C( 6613001208288696306) },
      { -INT64_C( 6983151280758577092), -INT64_C( 3692474292623161322) },
      { -INT64_C( 6983151280758577092), -INT64_C( 3692474292623161322) } },
    { UINT8_C( 41),
      {  INT64_C( 1682160708122191307),  INT64_C( 1645787412664035202) },
      { -INT64_C( 4141448541421803372), -INT64_C( 2852534427270558579) },
      { -INT64_C( 4141448541421803372),  INT64_C(                   0) } },
    { UINT8_C(232),
      { -INT64_C( 4763049640756494809), -INT64_C( 4504511835865906714) },
      { -INT64_C( 4689366023722177344), -INT64_C( 4851885784941515522) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(225),
      { -INT64_C( 8649104788464132142),  INT64_C( 5216190212666876128) },
      { -INT64_C( 2745373134994882079),  INT64_C( 2650696386964531012) },
      { -INT64_C( 8649104788464132142),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_min_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t src[16];
    const uint16_t k;
    const uint8_t a[16];
    const uint8_t b[16];
    const uint8_t r[16];
  } test_vec[] = {
    { { UINT8_C( 30), UINT8_C(207), UINT8_C( 11), UINT8_C(109), UINT8_C(199), UINT8_C(168), UINT8_C( 17), UINT8_C(125),
        UINT8_C(  0), UINT8_C(218), UINT8_C( 49), UINT8_C( 71), UINT8_C( 90), UINT8_C( 75), UINT8_C(235), UINT8_C(243) },
      UINT16_C( 5635),
      { UINT8_C( 87), UINT8_C(175), UINT8_C( 64), UINT8_C(163), UINT8_C(150), UINT8_C( 29), UINT8_C(116), UINT8_C( 48),
        UINT8_C(201), UINT8_C(111), UINT8_C(191), UINT8_C(153), UINT8_C(220), UINT8_C(221), UINT8_C(105), UINT8_C(231) },
      { UINT8_C( 75), UINT8_C( 48), UINT8_C(143), UINT8_C( 92), UINT8_C(173), UINT8_C(144), UINT8_C( 54), UINT8_C(222),
        UINT8_C(215), UINT8_C(144), UINT8_C( 41), UINT8_C(194), UINT8_C(131), UINT8_C( 44), UINT8_C(216), UINT8_C(218) },
      { UINT8_C( 75), UINT8_C( 48), UINT8_C( 11), UINT8_C(109), UINT8_C(199), UINT8_C(168), UINT8_C( 17), UINT8_C(125),
        UINT8_C(  0), UINT8_C(111), UINT8_C( 41), UINT8_C( 71), UINT8_C(131), UINT8_C( 75), UINT8_C(235), UINT8_C(243) } },
    { { UINT8_C(220), UINT8_C( 24), UINT8_C(125), UINT8_C(114), UINT8_C( 54), UINT8_C(242), UINT8_C(162),    UINT8_MAX,
        UINT8_C( 97), UINT8_C( 97), UINT8_C(152), UINT8_C( 62), UINT8_C( 62), UINT8_C(  1), UINT8_C( 37), UINT8_C(137) },
      UINT16_C(46385),
      { UINT8_C(229), UINT8_C(222), UINT8_C( 69), UINT8_C( 28), UINT8_C(188), UINT8_C( 28), UINT8_C(172), UINT8_C(229),
        UINT8_C(223), UINT8_C( 48), UINT8_C( 18), UINT8_C(183), UINT8_C( 10), UINT8_C(238), UINT8_C(208), UINT8_C(136) },
      { UINT8_C( 96), UINT8_C(  6), UINT8_C(122), UINT8_C(  2), UINT8_C(  5), UINT8_C(219), UINT8_C( 99), UINT8_C(157),
        UINT8_C( 25), UINT8_C(161), UINT8_C(159), UINT8_C( 63), UINT8_C( 43), UINT8_C(208), UINT8_C(244), UINT8_C( 16) },
      { UINT8_C( 96), UINT8_C( 24), UINT8_C(125), UINT8_C(114), UINT8_C(  5), UINT8_C( 28), UINT8_C(162),    UINT8_MAX,
        UINT8_C( 25), UINT8_C( 97), UINT8_C( 18), UINT8_C( 62), UINT8_C( 10), UINT8_C(208), UINT8_C( 37), UINT8_C( 16) } },
    { { UINT8_C(175), UINT8_C( 57), UINT8_C( 44), UINT8_C(107), UINT8_C( 85), UINT8_C(217), UINT8_C( 81), UINT8_C( 52),
        UINT8_C(  9), UINT8_C( 99), UINT8_C(236), UINT8_C( 19), UINT8_C( 81), UINT8_C(188), UINT8_C(155), UINT8_C(177) },
      UINT16_C( 5570),
      { UINT8_C(179), UINT8_C(199), UINT8_C(241), UINT8_C( 22), UINT8_C(100), UINT8_C( 10), UINT8_C(183), UINT8_C(  3),
        UINT8_C( 73), UINT8_C(226), UINT8_C(212), UINT8_C( 61), UINT8_C(243), UINT8_C(131), UINT8_C(118), UINT8_C( 31) },
      { UINT8_C(238), UINT8_C(204), UINT8_C(248), UINT8_C( 63), UINT8_C(  0), UINT8_C(  1), UINT8_C(162), UINT8_C(236),
        UINT8_C( 21), UINT8_C(243), UINT8_C(168), UINT8_C(176), UINT8_C(164), UINT8_C(106), UINT8_C(198), UINT8_C( 87) },
      { UINT8_C(175), UINT8_C(199), UINT8_C( 44), UINT8_C(107), UINT8_C( 85), UINT8_C(217), UINT8_C(162), UINT8_C(  3),
        UINT8_C( 21), UINT8_C( 99), UINT8_C(168), UINT8_C( 19), UINT8_C(164), UINT8_C(188), UINT8_C(155), UINT8_C(177) } },
    { { UINT8_C( 49), UINT8_C(183), UINT8_C(109), UINT8_C(150), UINT8_C(193), UINT8_C( 37), UINT8_C(153), UINT8_C( 11),
        UINT8_C(  7), UINT8_C(109), UINT8_C( 72), UINT8_C(250), UINT8_C(240), UINT8_C(191), UINT8_C( 26), UINT8_C(223) },
      UINT16_C( 4747),
      { UINT8_C( 30), UINT8_C(139), UINT8_C( 20), UINT8_C(193), UINT8_C(120), UINT8_C( 41), UINT8_C(180), UINT8_C( 32),
        UINT8_C(217), UINT8_C( 89), UINT8_C(139), UINT8_C(159), UINT8_C(176), UINT8_C(188), UINT8_C( 86), UINT8_C( 30) },
      { UINT8_C( 82), UINT8_C( 24), UINT8_C( 67), UINT8_C(236), UINT8_C( 35), UINT8_C( 74), UINT8_C( 89), UINT8_C(107),
        UINT8_C( 69), UINT8_C( 74), UINT8_C( 42), UINT8_C( 95), UINT8_C( 41), UINT8_C(181), UINT8_C(113), UINT8_C( 71) },
      { UINT8_C( 30), UINT8_C( 24), UINT8_C(109), UINT8_C(193), UINT8_C(193), UINT8_C( 37), UINT8_C(153), UINT8_C( 32),
        UINT8_C(  7), UINT8_C( 74), UINT8_C( 72), UINT8_C(250), UINT8_C( 41), UINT8_C(191), UINT8_C( 26), UINT8_C(223) } },
    { { UINT8_C( 65), UINT8_C(133), UINT8_C(  8), UINT8_C(185), UINT8_C(174), UINT8_C(189), UINT8_C(217), UINT8_C(136),
        UINT8_C( 22), UINT8_C(100), UINT8_C( 39), UINT8_C(198), UINT8_C( 33), UINT8_C(126), UINT8_C(228), UINT8_C(115) },
      UINT16_C(10134),
      { UINT8_C( 95), UINT8_C(185), UINT8_C(114), UINT8_C(185), UINT8_C( 36), UINT8_C(183), UINT8_C(  3), UINT8_C( 79),
        UINT8_C( 22), UINT8_C( 44), UINT8_C(  4), UINT8_C(135), UINT8_C(115), UINT8_C( 69), UINT8_C( 13), UINT8_C(124) },
      { UINT8_C(254), UINT8_C(187), UINT8_C( 57), UINT8_C(216), UINT8_C( 67), UINT8_C( 79), UINT8_C( 60), UINT8_C(107),
        UINT8_C( 21), UINT8_C( 93), UINT8_C(233), UINT8_C(250), UINT8_C(209), UINT8_C(127), UINT8_C( 33), UINT8_C( 48) },
      { UINT8_C( 65), UINT8_C(185), UINT8_C( 57), UINT8_C(185), UINT8_C( 36), UINT8_C(189), UINT8_C(217), UINT8_C( 79),
        UINT8_C( 21), UINT8_C( 44), UINT8_C(  4), UINT8_C(198), UINT8_C( 33), UINT8_C( 69), UINT8_C(228), UINT8_C(115) } },
    { { UINT8_C( 56), UINT8_C(147), UINT8_C(233), UINT8_C( 92), UINT8_C( 74), UINT8_C(236), UINT8_C(171), UINT8_C( 96),
        UINT8_C( 24), UINT8_C(176), UINT8_C(232), UINT8_C(140), UINT8_C(245), UINT8_C(245), UINT8_C(  8), UINT8_C(244) },
      UINT16_C(16816),
      { UINT8_C(204), UINT8_C(244), UINT8_C(144), UINT8_C(  8), UINT8_C( 95), UINT8_C(165), UINT8_C(102), UINT8_C( 72),
        UINT8_C(159), UINT8_C( 55), UINT8_C(199), UINT8_C(193), UINT8_C(103),    UINT8_MAX, UINT8_C( 84), UINT8_C( 81) },
      { UINT8_C( 91), UINT8_C(159), UINT8_C( 61), UINT8_C(  7),    UINT8_MAX, UINT8_C( 86), UINT8_C(183), UINT8_C(231),
        UINT8_C(226), UINT8_C(172), UINT8_C(220), UINT8_C(234), UINT8_C(160), UINT8_C(141), UINT8_C( 43), UINT8_C(108) },
      { UINT8_C( 56), UINT8_C(147), UINT8_C(233), UINT8_C( 92), UINT8_C( 95), UINT8_C( 86), UINT8_C(171), UINT8_C( 72),
        UINT8_C(159), UINT8_C(176), UINT8_C(232), UINT8_C(140), UINT8_C(245), UINT8_C(245), UINT8_C( 43), UINT8_C(244) } },
    { { UINT8_C(129), UINT8_C(187), UINT8_C(117), UINT8_C(224), UINT8_C( 96), UINT8_C(219), UINT8_C( 40), UINT8_C(  0),
        UINT8_C( 18), UINT8_C(239), UINT8_C(193), UINT8_C(121), UINT8_C(238), UINT8_C( 21), UINT8_C(202), UINT8_C( 73) },
      UINT16_C( 2228),
      { UINT8_C( 80), UINT8_C(180), UINT8_C( 94), UINT8_C(  7), UINT8_C(155), UINT8_C( 64), UINT8_C(180), UINT8_C(120),
        UINT8_C( 42), UINT8_C( 84), UINT8_C(  5), UINT8_C( 85), UINT8_C(193), UINT8_C(134), UINT8_C( 16), UINT8_C( 54) },
      { UINT8_C(102), UINT8_C(112), UINT8_C( 17), UINT8_C(142), UINT8_C(112), UINT8_C( 35), UINT8_C(125), UINT8_C( 49),
        UINT8_C(156), UINT8_C(107), UINT8_C( 71), UINT8_C(103), UINT8_C(180), UINT8_C(251), UINT8_C(111), UINT8_C(  5) },
      { UINT8_C(129), UINT8_C(187), UINT8_C( 17), UINT8_C(224), UINT8_C(112), UINT8_C( 35), UINT8_C( 40), UINT8_C( 49),
        UINT8_C( 18), UINT8_C(239), UINT8_C(193), UINT8_C( 85), UINT8_C(238), UINT8_C( 21), UINT8_C(202), UINT8_C( 73) } },
    { { UINT8_C(175), UINT8_C(205), UINT8_C( 12), UINT8_C( 75), UINT8_C( 13), UINT8_C(192), UINT8_C(195), UINT8_C( 55),
        UINT8_C( 21), UINT8_C(200), UINT8_C(140), UINT8_C(214), UINT8_C( 78), UINT8_C(156), UINT8_C( 12), UINT8_C(180) },
      UINT16_C( 7436),
      { UINT8_C( 66), UINT8_C(125), UINT8_C( 64), UINT8_C(191), UINT8_C(174), UINT8_C(220), UINT8_C( 42), UINT8_C(245),
        UINT8_C( 67), UINT8_C(222), UINT8_C(241), UINT8_C(178), UINT8_C(227), UINT8_C(160), UINT8_C(127), UINT8_C(240) },
      { UINT8_C(235), UINT8_C(140), UINT8_C(176), UINT8_C(174), UINT8_C(195), UINT8_C(197), UINT8_C(118), UINT8_C( 79),
        UINT8_C(155), UINT8_C(196), UINT8_C(235), UINT8_C(167), UINT8_C(120), UINT8_C(248), UINT8_C(196), UINT8_C(186) },
      { UINT8_C(175), UINT8_C(205), UINT8_C( 64), UINT8_C(174), UINT8_C( 13), UINT8_C(192), UINT8_C(195), UINT8_C( 55),
        UINT8_C( 67), UINT8_C(200), UINT8_C(235), UINT8_C(167), UINT8_C(120), UINT8_C(156), UINT8_C( 12), UINT8_C(180) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi8(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_epu8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_epu8");
    easysimd_test_x86_assert_equal_u8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u8x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_test_x86_random_u8x16();
    easysimd__m128i r = easysimd_mm_mask_min_epu8(src, k, a, b);

    easysimd_test_x86_write_u8x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_min_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const uint8_t a[16];
    const uint8_t b[16];
    const uint8_t r[16];
  } test_vec[] = {
    { UINT16_C(16714),
      { UINT8_C( 54), UINT8_C( 79), UINT8_C(118), UINT8_C( 46), UINT8_C(214), UINT8_C( 86), UINT8_C( 14), UINT8_C(182),
        UINT8_C(249), UINT8_C(106), UINT8_C( 88), UINT8_C( 92), UINT8_C(178), UINT8_C( 57), UINT8_C( 29), UINT8_C(108) },
      { UINT8_C( 33), UINT8_C(203), UINT8_C(229), UINT8_C(  7), UINT8_C(164), UINT8_C( 42), UINT8_C(175), UINT8_C( 19),
        UINT8_C( 18), UINT8_C(  0), UINT8_C( 61), UINT8_C(219), UINT8_C( 36), UINT8_C(135), UINT8_C( 28), UINT8_C( 90) },
      { UINT8_C(  0), UINT8_C( 79), UINT8_C(  0), UINT8_C(  7), UINT8_C(  0), UINT8_C(  0), UINT8_C( 14), UINT8_C(  0),
        UINT8_C( 18), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 28), UINT8_C(  0) } },
    { UINT16_C(37591),
      { UINT8_C(136), UINT8_C(173), UINT8_C(233), UINT8_C(150), UINT8_C(100), UINT8_C(226), UINT8_C(  0), UINT8_C(188),
        UINT8_C( 62), UINT8_C(178), UINT8_C(245), UINT8_C( 91), UINT8_C( 31), UINT8_C( 23), UINT8_C( 38), UINT8_C(  4) },
      { UINT8_C( 30), UINT8_C(203), UINT8_C( 46), UINT8_C(205), UINT8_C(222), UINT8_C( 64), UINT8_C(206), UINT8_C( 27),
        UINT8_C( 28), UINT8_C(242), UINT8_C(163), UINT8_C( 56), UINT8_C( 77), UINT8_C(122), UINT8_C(203), UINT8_C(213) },
      { UINT8_C( 30), UINT8_C(173), UINT8_C( 46), UINT8_C(  0), UINT8_C(100), UINT8_C(  0), UINT8_C(  0), UINT8_C( 27),
        UINT8_C(  0), UINT8_C(178), UINT8_C(  0), UINT8_C(  0), UINT8_C( 31), UINT8_C(  0), UINT8_C(  0), UINT8_C(  4) } },
    { UINT16_C(46119),
      { UINT8_C(108), UINT8_C(139), UINT8_C(150), UINT8_C(108), UINT8_C( 71), UINT8_C(212), UINT8_C( 31), UINT8_C( 61),
        UINT8_C( 47), UINT8_C( 62), UINT8_C( 84), UINT8_C( 86), UINT8_C( 66), UINT8_C(114), UINT8_C( 33), UINT8_C(113) },
      { UINT8_C( 64),    UINT8_MAX, UINT8_C(177), UINT8_C( 14), UINT8_C( 26), UINT8_C(205), UINT8_C(  0), UINT8_C(189),
        UINT8_C(  6), UINT8_C( 77), UINT8_C( 55), UINT8_C(209), UINT8_C( 35), UINT8_C( 95), UINT8_C(133), UINT8_C(143) },
      { UINT8_C( 64), UINT8_C(139), UINT8_C(150), UINT8_C(  0), UINT8_C(  0), UINT8_C(205), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C( 55), UINT8_C(  0), UINT8_C( 35), UINT8_C( 95), UINT8_C(  0), UINT8_C(113) } },
    { UINT16_C( 7146),
      { UINT8_C(251), UINT8_C( 50), UINT8_C(239), UINT8_C( 26), UINT8_C(111), UINT8_C( 30), UINT8_C( 88), UINT8_C(195),
        UINT8_C(116), UINT8_C(155), UINT8_C( 53), UINT8_C(149), UINT8_C( 12), UINT8_C(117), UINT8_C(148), UINT8_C(189) },
      { UINT8_C(131), UINT8_C(175), UINT8_C(139), UINT8_C(132), UINT8_C(108), UINT8_C(145), UINT8_C(209), UINT8_C(164),
        UINT8_C( 98), UINT8_C(244), UINT8_C(  3), UINT8_C(231), UINT8_C(131), UINT8_C(237), UINT8_C(  2), UINT8_C(127) },
      { UINT8_C(  0), UINT8_C( 50), UINT8_C(  0), UINT8_C( 26), UINT8_C(  0), UINT8_C( 30), UINT8_C( 88), UINT8_C(164),
        UINT8_C( 98), UINT8_C(155), UINT8_C(  0), UINT8_C(149), UINT8_C( 12), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0) } },
    { UINT16_C(61727),
      { UINT8_C(153), UINT8_C(142), UINT8_C( 15), UINT8_C(242), UINT8_C( 81), UINT8_C(132), UINT8_C(141), UINT8_C(135),
        UINT8_C( 25), UINT8_C(153), UINT8_C(252), UINT8_C(174), UINT8_C( 86), UINT8_C(128), UINT8_C( 93), UINT8_C(225) },
      { UINT8_C(  4), UINT8_C(201), UINT8_C(114), UINT8_C(213), UINT8_C(109), UINT8_C(212), UINT8_C(202), UINT8_C(112),
        UINT8_C(187), UINT8_C( 77), UINT8_C( 94), UINT8_C(189), UINT8_C(204), UINT8_C(125), UINT8_C(174), UINT8_C(102) },
      { UINT8_C(  4), UINT8_C(142), UINT8_C( 15), UINT8_C(213), UINT8_C( 81), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 25), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 86), UINT8_C(125), UINT8_C( 93), UINT8_C(102) } },
    { UINT16_C(48652),
      { UINT8_C( 88), UINT8_C( 93), UINT8_C( 66), UINT8_C(229), UINT8_C(228), UINT8_C( 91), UINT8_C(126), UINT8_C(225),
        UINT8_C(  9), UINT8_C(212), UINT8_C( 97), UINT8_C(102), UINT8_C(182), UINT8_C(101), UINT8_C( 48), UINT8_C( 40) },
      { UINT8_C( 58), UINT8_C(157), UINT8_C(253), UINT8_C(  4), UINT8_C( 14), UINT8_C(184), UINT8_C( 82), UINT8_C(108),
        UINT8_C(118), UINT8_C( 30), UINT8_C(233), UINT8_C( 36), UINT8_C(132), UINT8_C(245), UINT8_C(226), UINT8_C(220) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C( 66), UINT8_C(  4), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C( 30), UINT8_C( 97), UINT8_C( 36), UINT8_C(132), UINT8_C(101), UINT8_C(  0), UINT8_C( 40) } },
    { UINT16_C( 9299),
      { UINT8_C(193), UINT8_C( 55), UINT8_C(128), UINT8_C( 63), UINT8_C( 24), UINT8_C(137), UINT8_C( 20), UINT8_C(121),
        UINT8_C(240), UINT8_C(202), UINT8_C(222), UINT8_C( 32), UINT8_C(242), UINT8_C( 25), UINT8_C(189), UINT8_C(239) },
      { UINT8_C( 29), UINT8_C(203), UINT8_C(168), UINT8_C(111), UINT8_C( 55), UINT8_C( 30), UINT8_C(142), UINT8_C( 33),
        UINT8_C( 66), UINT8_C( 18), UINT8_C( 22), UINT8_C( 37), UINT8_C(239), UINT8_C(105), UINT8_C( 73), UINT8_C(176) },
      { UINT8_C( 29), UINT8_C( 55), UINT8_C(  0), UINT8_C(  0), UINT8_C( 24), UINT8_C(  0), UINT8_C( 20), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C( 22), UINT8_C(  0), UINT8_C(  0), UINT8_C( 25), UINT8_C(  0), UINT8_C(  0) } },
    { UINT16_C(51617),
      { UINT8_C(240), UINT8_C(185), UINT8_C( 83), UINT8_C(  4), UINT8_C( 51), UINT8_C( 67), UINT8_C(206), UINT8_C( 17),
        UINT8_C( 99), UINT8_C(192), UINT8_C( 42), UINT8_C( 32), UINT8_C(176), UINT8_C( 72), UINT8_C(236), UINT8_C( 88) },
      { UINT8_C(183), UINT8_C( 35), UINT8_C(118), UINT8_C( 69), UINT8_C( 68), UINT8_C(184), UINT8_C( 88), UINT8_C( 91),
        UINT8_C(221), UINT8_C( 71), UINT8_C(196), UINT8_C( 39), UINT8_C(247), UINT8_C(101), UINT8_C(240), UINT8_C(231) },
      { UINT8_C(183), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 67), UINT8_C(  0), UINT8_C( 17),
        UINT8_C( 99), UINT8_C(  0), UINT8_C(  0), UINT8_C( 32), UINT8_C(  0), UINT8_C(  0), UINT8_C(236), UINT8_C( 88) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_epu8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_epu8");
    easysimd_test_x86_assert_equal_u8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_test_x86_random_u8x16();
    easysimd__m128i r = easysimd_mm_maskz_min_epu8(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t src[8];
    const uint8_t k;
    const uint16_t a[8];
    const uint16_t b[8];
    const uint16_t r[8];
  } test_vec[] = {
    { { UINT16_C( 1141), UINT16_C( 9081), UINT16_C(41953), UINT16_C( 9241), UINT16_C( 2690), UINT16_C(26071), UINT16_C(22186), UINT16_C(38485) },
      UINT8_C(227),
      { UINT16_C(17414), UINT16_C(52134), UINT16_C(63163), UINT16_C(32615), UINT16_C( 3809), UINT16_C(55800), UINT16_C(45779), UINT16_C(55118) },
      { UINT16_C(29228), UINT16_C(53176), UINT16_C(56715), UINT16_C(38225), UINT16_C(47028), UINT16_C( 2623), UINT16_C(54540), UINT16_C( 4845) },
      { UINT16_C(17414), UINT16_C(52134), UINT16_C(41953), UINT16_C( 9241), UINT16_C( 2690), UINT16_C( 2623), UINT16_C(45779), UINT16_C( 4845) } },
    { { UINT16_C(37914), UINT16_C(54750), UINT16_C(17802), UINT16_C(27476), UINT16_C(19539), UINT16_C( 9797), UINT16_C(37887), UINT16_C(11262) },
      UINT8_C(  5),
      { UINT16_C(64182), UINT16_C(37776), UINT16_C( 9548), UINT16_C(  839), UINT16_C(21093), UINT16_C(14863), UINT16_C( 8767), UINT16_C(54100) },
      { UINT16_C(10496), UINT16_C(17757), UINT16_C(51582), UINT16_C(51864), UINT16_C(48910), UINT16_C(41417), UINT16_C(62653), UINT16_C(29607) },
      { UINT16_C(10496), UINT16_C(54750), UINT16_C( 9548), UINT16_C(27476), UINT16_C(19539), UINT16_C( 9797), UINT16_C(37887), UINT16_C(11262) } },
    { { UINT16_C(14319), UINT16_C(15111), UINT16_C(20061), UINT16_C(49726), UINT16_C(19872), UINT16_C(57596), UINT16_C(20847), UINT16_C(28595) },
      UINT8_C(122),
      { UINT16_C(46097), UINT16_C(56056), UINT16_C(49997), UINT16_C( 3304), UINT16_C(35212), UINT16_C(33225), UINT16_C(15408), UINT16_C(26736) },
      { UINT16_C(43843), UINT16_C(37573), UINT16_C(34793), UINT16_C(13874), UINT16_C( 4739), UINT16_C(54438), UINT16_C( 5574), UINT16_C(55119) },
      { UINT16_C(14319), UINT16_C(37573), UINT16_C(20061), UINT16_C( 3304), UINT16_C( 4739), UINT16_C(33225), UINT16_C( 5574), UINT16_C(28595) } },
    { { UINT16_C(18378), UINT16_C( 6065), UINT16_C(39178), UINT16_C(38691), UINT16_C(60450), UINT16_C(21272), UINT16_C(34856), UINT16_C(27835) },
      UINT8_C( 51),
      { UINT16_C(65152), UINT16_C( 1820), UINT16_C(21040), UINT16_C(17290), UINT16_C(24568), UINT16_C( 3593), UINT16_C(57518), UINT16_C(62936) },
      { UINT16_C(61329), UINT16_C(10752), UINT16_C(38674), UINT16_C(65100), UINT16_C(40879), UINT16_C(14118), UINT16_C(37466), UINT16_C(55914) },
      { UINT16_C(61329), UINT16_C( 1820), UINT16_C(39178), UINT16_C(38691), UINT16_C(24568), UINT16_C( 3593), UINT16_C(34856), UINT16_C(27835) } },
    { { UINT16_C(34448), UINT16_C(49633), UINT16_C(27864), UINT16_C(53508), UINT16_C( 3531), UINT16_C(31199), UINT16_C(47085), UINT16_C(32366) },
      UINT8_C(166),
      { UINT16_C(43118), UINT16_C( 1464), UINT16_C(46836), UINT16_C(38068), UINT16_C(60380), UINT16_C(28654), UINT16_C(51541), UINT16_C(56319) },
      { UINT16_C(49322), UINT16_C( 5812), UINT16_C(34244), UINT16_C(53729), UINT16_C(23140), UINT16_C( 7102), UINT16_C(15561), UINT16_C(14273) },
      { UINT16_C(34448), UINT16_C( 1464), UINT16_C(34244), UINT16_C(53508), UINT16_C( 3531), UINT16_C( 7102), UINT16_C(47085), UINT16_C(14273) } },
    { { UINT16_C(31204), UINT16_C(55613), UINT16_C(61743), UINT16_C( 2925), UINT16_C(23517), UINT16_C(12922), UINT16_C(31268), UINT16_C(53006) },
      UINT8_C( 58),
      { UINT16_C(58818), UINT16_C(18431), UINT16_C(53447), UINT16_C( 8619), UINT16_C(50831), UINT16_C(52202), UINT16_C( 8839), UINT16_C(  176) },
      { UINT16_C(35167), UINT16_C(20527), UINT16_C(15094), UINT16_C(20781), UINT16_C(24757), UINT16_C(12150), UINT16_C(17774), UINT16_C(12393) },
      { UINT16_C(31204), UINT16_C(18431), UINT16_C(61743), UINT16_C( 8619), UINT16_C(24757), UINT16_C(12150), UINT16_C(31268), UINT16_C(53006) } },
    { { UINT16_C(26666), UINT16_C(61815), UINT16_C( 8761), UINT16_C(51219), UINT16_C(65000), UINT16_C(28563), UINT16_C(17183), UINT16_C(32367) },
      UINT8_C(204),
      { UINT16_C(53150), UINT16_C(55490), UINT16_C( 5372), UINT16_C(23693), UINT16_C(48266), UINT16_C(53194), UINT16_C(64038), UINT16_C(36601) },
      { UINT16_C(60273), UINT16_C(37831), UINT16_C(36862), UINT16_C(64379), UINT16_C(59939), UINT16_C(26139), UINT16_C(39257), UINT16_C(63283) },
      { UINT16_C(26666), UINT16_C(61815), UINT16_C( 5372), UINT16_C(23693), UINT16_C(65000), UINT16_C(28563), UINT16_C(39257), UINT16_C(36601) } },
    { { UINT16_C(62824), UINT16_C(26064), UINT16_C(23817), UINT16_C(37825), UINT16_C(35866), UINT16_C(16482), UINT16_C(23686), UINT16_C(63694) },
      UINT8_C( 71),
      { UINT16_C(35734), UINT16_C( 9541), UINT16_C(16391), UINT16_C(61768), UINT16_C(44891), UINT16_C(62795), UINT16_C(17122), UINT16_C(55133) },
      { UINT16_C(49682), UINT16_C(28897), UINT16_C(29828), UINT16_C( 4234), UINT16_C(51927), UINT16_C(13206), UINT16_C(36504), UINT16_C(11898) },
      { UINT16_C(35734), UINT16_C( 9541), UINT16_C(16391), UINT16_C(37825), UINT16_C(35866), UINT16_C(16482), UINT16_C(17122), UINT16_C(63694) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_epu16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_epu16");
    easysimd_test_x86_assert_equal_u16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__m128i r = easysimd_mm_mask_min_epu16(src, k, a, b);

    easysimd_test_x86_write_u16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_min_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint16_t a[8];
    const uint16_t b[8];
    const uint16_t r[8];
  } test_vec[] = {
    { UINT8_C( 31),
      { UINT16_C(60227), UINT16_C(34386), UINT16_C(25529), UINT16_C(31465), UINT16_C( 2702), UINT16_C(54826), UINT16_C(33526), UINT16_C( 6541) },
      { UINT16_C(54264), UINT16_C(45150), UINT16_C(47403), UINT16_C(29326), UINT16_C(46461), UINT16_C(58217), UINT16_C(20901), UINT16_C(59650) },
      { UINT16_C(54264), UINT16_C(34386), UINT16_C(25529), UINT16_C(29326), UINT16_C( 2702), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C( 60),
      { UINT16_C(28500), UINT16_C(47094), UINT16_C(28761), UINT16_C(25413), UINT16_C( 7066), UINT16_C( 7257), UINT16_C(29353), UINT16_C(31764) },
      { UINT16_C(50384), UINT16_C(35239), UINT16_C( 6482), UINT16_C( 1799), UINT16_C(60034), UINT16_C(54189), UINT16_C(38636), UINT16_C(16400) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C( 6482), UINT16_C( 1799), UINT16_C( 7066), UINT16_C( 7257), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C(  5),
      { UINT16_C(63238), UINT16_C(30302), UINT16_C(49469), UINT16_C(22544), UINT16_C(11290), UINT16_C(36097), UINT16_C(32064), UINT16_C( 1117) },
      { UINT16_C(59172), UINT16_C(15703), UINT16_C(24302), UINT16_C(55488), UINT16_C(37643), UINT16_C(41412), UINT16_C( 1187), UINT16_C(43431) },
      { UINT16_C(59172), UINT16_C(    0), UINT16_C(24302), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C(251),
      { UINT16_C( 7941), UINT16_C(51000), UINT16_C(37167), UINT16_C(23521), UINT16_C(28306), UINT16_C( 4251), UINT16_C(41164), UINT16_C(45876) },
      { UINT16_C(29431), UINT16_C(21921), UINT16_C(31026), UINT16_C(50529), UINT16_C(  573), UINT16_C(16745), UINT16_C( 4777), UINT16_C(44860) },
      { UINT16_C( 7941), UINT16_C(21921), UINT16_C(    0), UINT16_C(23521), UINT16_C(  573), UINT16_C( 4251), UINT16_C( 4777), UINT16_C(44860) } },
    { UINT8_C( 50),
      { UINT16_C(30325), UINT16_C( 1633), UINT16_C(48471), UINT16_C(50840), UINT16_C(43096), UINT16_C(63634), UINT16_C(17885), UINT16_C(20463) },
      { UINT16_C(17894), UINT16_C(24449), UINT16_C(18086), UINT16_C(43164), UINT16_C(56751), UINT16_C(49746), UINT16_C(  281), UINT16_C(36596) },
      { UINT16_C(    0), UINT16_C( 1633), UINT16_C(    0), UINT16_C(    0), UINT16_C(43096), UINT16_C(49746), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C(119),
      { UINT16_C(37973), UINT16_C( 4814), UINT16_C(37933), UINT16_C(54635), UINT16_C(25382), UINT16_C(27570), UINT16_C(  339), UINT16_C(38993) },
      { UINT16_C(45186), UINT16_C(51518), UINT16_C(58956), UINT16_C(10616), UINT16_C(14904), UINT16_C(14659), UINT16_C(53550), UINT16_C(33968) },
      { UINT16_C(37973), UINT16_C( 4814), UINT16_C(37933), UINT16_C(    0), UINT16_C(14904), UINT16_C(14659), UINT16_C(  339), UINT16_C(    0) } },
    { UINT8_C(102),
      { UINT16_C(38527), UINT16_C( 5011), UINT16_C(26625), UINT16_C(25914), UINT16_C(42267), UINT16_C( 7352), UINT16_C(20727), UINT16_C(42911) },
      { UINT16_C(26766), UINT16_C(29940), UINT16_C( 7648), UINT16_C( 7085), UINT16_C(58976), UINT16_C(12873), UINT16_C(52631), UINT16_C( 5784) },
      { UINT16_C(    0), UINT16_C( 5011), UINT16_C( 7648), UINT16_C(    0), UINT16_C(    0), UINT16_C( 7352), UINT16_C(20727), UINT16_C(    0) } },
    { UINT8_C(100),
      { UINT16_C(10539), UINT16_C(37733), UINT16_C(51811), UINT16_C( 2478), UINT16_C(52098), UINT16_C(53760), UINT16_C(42858), UINT16_C(53856) },
      { UINT16_C(54683), UINT16_C(47538), UINT16_C(52610), UINT16_C(26649), UINT16_C(19223), UINT16_C(58623), UINT16_C( 5603), UINT16_C( 3656) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(51811), UINT16_C(    0), UINT16_C(    0), UINT16_C(53760), UINT16_C( 5603), UINT16_C(    0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_epu16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_epu16");
    easysimd_test_x86_assert_equal_u16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__m128i r = easysimd_mm_maskz_min_epu16(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t src[4];
    const uint8_t k;
    const uint32_t a[4];
    const uint32_t b[4];
    const uint32_t r[4];
  } test_vec[] = {
    { { UINT32_C( 559202074), UINT32_C(1527946495), UINT32_C( 760241483), UINT32_C(2986716576) },
      UINT8_C(112),
      { UINT32_C(1525949158), UINT32_C(1982923948), UINT32_C( 688874650), UINT32_C(2638429662) },
      { UINT32_C( 782066833), UINT32_C(3564763254), UINT32_C(4118062920), UINT32_C(2456102572) },
      { UINT32_C( 559202074), UINT32_C(1527946495), UINT32_C( 760241483), UINT32_C(2986716576) } },
    { { UINT32_C(4125907273), UINT32_C(4167835229), UINT32_C(1629584258), UINT32_C(1258185912) },
      UINT8_C(200),
      { UINT32_C(2470344859), UINT32_C(2564494065), UINT32_C(2906968454), UINT32_C(2432095798) },
      { UINT32_C(3790465987), UINT32_C(3529762135), UINT32_C(1787544582), UINT32_C(1580389827) },
      { UINT32_C(4125907273), UINT32_C(4167835229), UINT32_C(1629584258), UINT32_C(1580389827) } },
    { { UINT32_C(1056076109), UINT32_C( 181915011), UINT32_C(3585547166), UINT32_C(3043339762) },
      UINT8_C(152),
      { UINT32_C( 955291218), UINT32_C(3191784185), UINT32_C( 578922829), UINT32_C(1282400219) },
      { UINT32_C(2681253585), UINT32_C(2688408197), UINT32_C(1049760401), UINT32_C(3403040631) },
      { UINT32_C(1056076109), UINT32_C( 181915011), UINT32_C(3585547166), UINT32_C(1282400219) } },
    { { UINT32_C(3607283421), UINT32_C(3600105609), UINT32_C(3321435881), UINT32_C(3339806965) },
      UINT8_C( 22),
      { UINT32_C(3147523809), UINT32_C(3058449571), UINT32_C( 372083406), UINT32_C( 670300001) },
      { UINT32_C( 984664825), UINT32_C(1965262687), UINT32_C(3899385984), UINT32_C(3690935034) },
      { UINT32_C(3607283421), UINT32_C(1965262687), UINT32_C( 372083406), UINT32_C(3339806965) } },
    { { UINT32_C( 999791256), UINT32_C(2784093142), UINT32_C(3485146990), UINT32_C( 284601878) },
      UINT8_C(121),
      { UINT32_C( 785926823), UINT32_C(1454263917), UINT32_C(3947927225), UINT32_C( 813902741) },
      { UINT32_C(2785459906), UINT32_C(3474238384), UINT32_C( 367387494), UINT32_C(2156852697) },
      { UINT32_C( 785926823), UINT32_C(2784093142), UINT32_C(3485146990), UINT32_C( 813902741) } },
    { { UINT32_C(2913887807), UINT32_C(1828936884), UINT32_C(2287490035), UINT32_C(1102633854) },
      UINT8_C(154),
      { UINT32_C(1783293887), UINT32_C(3738245627), UINT32_C(4105692926), UINT32_C(3660789876) },
      { UINT32_C(1133502694), UINT32_C( 909573347), UINT32_C( 817217109), UINT32_C( 919271031) },
      { UINT32_C(2913887807), UINT32_C( 909573347), UINT32_C(2287490035), UINT32_C( 919271031) } },
    { { UINT32_C(3634369757), UINT32_C( 750154029), UINT32_C(3407899991), UINT32_C(2359710629) },
      UINT8_C( 52),
      { UINT32_C( 823643957), UINT32_C(3280358917), UINT32_C(4181374723), UINT32_C(2530635905) },
      { UINT32_C(2193862161), UINT32_C(3520786276), UINT32_C(1668785423), UINT32_C(2157380427) },
      { UINT32_C(3634369757), UINT32_C( 750154029), UINT32_C(1668785423), UINT32_C(2359710629) } },
    { { UINT32_C(3618811602), UINT32_C(4288297212), UINT32_C(1912133103), UINT32_C(1443352133) },
      UINT8_C(124),
      { UINT32_C(3135297994), UINT32_C(1489613491), UINT32_C( 748956713), UINT32_C(1929258179) },
      { UINT32_C( 258921942), UINT32_C(1157525103), UINT32_C( 898264934), UINT32_C(1102176374) },
      { UINT32_C(3618811602), UINT32_C(4288297212), UINT32_C( 748956713), UINT32_C(1102176374) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_epu32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_epu32");
    easysimd_test_x86_assert_equal_u32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__m128i r = easysimd_mm_mask_min_epu32(src, k, a, b);

    easysimd_test_x86_write_u32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_min_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint32_t a[4];
    const uint32_t b[4];
    const uint32_t r[4];
  } test_vec[] = {
    { UINT8_C( 63),
      { UINT32_C(2023924398), UINT32_C( 469478224), UINT32_C(1401277867), UINT32_C(  65951534) },
      { UINT32_C(3615860490), UINT32_C( 216985025), UINT32_C(  49337325), UINT32_C(3376545307) },
      { UINT32_C(2023924398), UINT32_C( 216985025), UINT32_C(  49337325), UINT32_C(  65951534) } },
    { UINT8_C(160),
      { UINT32_C(2414953188), UINT32_C( 171641917), UINT32_C(3912797842), UINT32_C( 619920252) },
      { UINT32_C(2934295488), UINT32_C(2359030201), UINT32_C(3769146849), UINT32_C(3279974879) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C(179),
      { UINT32_C(2129678961), UINT32_C( 454097805), UINT32_C(1872296243), UINT32_C(3090136301) },
      { UINT32_C(2456935841), UINT32_C( 376766072), UINT32_C( 401953958), UINT32_C(1187756244) },
      { UINT32_C(2129678961), UINT32_C( 376766072), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C( 10),
      { UINT32_C(3063465147), UINT32_C(3454710740), UINT32_C( 129653067), UINT32_C(1722315400) },
      { UINT32_C(3806215140), UINT32_C(  59307183), UINT32_C(2715262953), UINT32_C( 648813930) },
      { UINT32_C(         0), UINT32_C(  59307183), UINT32_C(         0), UINT32_C( 648813930) } },
    { UINT8_C(225),
      { UINT32_C(4155890756), UINT32_C( 524518342), UINT32_C(2963819069), UINT32_C( 781455091) },
      { UINT32_C(3789387500), UINT32_C(2647318782), UINT32_C(3574099127), UINT32_C(1555443224) },
      { UINT32_C(3789387500), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C( 10),
      { UINT32_C(4023473004), UINT32_C(3777818774), UINT32_C(2798967960), UINT32_C(3868394096) },
      { UINT32_C(3219420383), UINT32_C(2859893310), UINT32_C(3099741066), UINT32_C(1858215426) },
      { UINT32_C(         0), UINT32_C(2859893310), UINT32_C(         0), UINT32_C(1858215426) } },
    { UINT8_C(114),
      { UINT32_C(2215140755), UINT32_C(1713170825), UINT32_C(3218523069), UINT32_C(3399400790) },
      { UINT32_C( 587750817), UINT32_C( 531477460), UINT32_C(2485216629), UINT32_C(3137769256) },
      { UINT32_C(         0), UINT32_C( 531477460), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C(236),
      { UINT32_C(4168498958), UINT32_C( 532012124), UINT32_C(1869968818), UINT32_C(1896955667) },
      { UINT32_C(4198839367), UINT32_C(1198548194), UINT32_C( 359597190), UINT32_C( 402795274) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C( 359597190), UINT32_C( 402795274) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_epu32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_epu32");
    easysimd_test_x86_assert_equal_u32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__m128i r = easysimd_mm_maskz_min_epu32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t src[2];
    const uint8_t k;
    const uint64_t a[2];
    const uint64_t b[2];
    const uint64_t r[2];
  } test_vec[] = {
    { { UINT64_C( 7909944118711390905), UINT64_C( 7216622087783934449) },
      UINT8_C(108),
      { UINT64_C(16242549200856511381), UINT64_C(16304324022673841834) },
      { UINT64_C(16017495681919266963), UINT64_C(13344113077238168876) },
      { UINT64_C( 7909944118711390905), UINT64_C( 7216622087783934449) } },
    { { UINT64_C(15318190572921686595), UINT64_C( 1870862487186828740) },
      UINT8_C( 83),
      { UINT64_C(13248367149095416092), UINT64_C(15503242379281538579) },
      { UINT64_C( 6717917572681096016), UINT64_C( 6930191299243101749) },
      { UINT64_C( 6717917572681096016), UINT64_C( 6930191299243101749) } },
    { { UINT64_C(12744881951896237421), UINT64_C(11469747196611771016) },
      UINT8_C(149),
      { UINT64_C(11329477003641689390), UINT64_C( 3796324750857409350) },
      { UINT64_C( 7981083001223173712), UINT64_C( 8242533189319202580) },
      { UINT64_C( 7981083001223173712), UINT64_C(11469747196611771016) } },
    { { UINT64_C( 1251351760144394353), UINT64_C(14931405834110058180) },
      UINT8_C(216),
      { UINT64_C(16329203200526453000), UINT64_C( 3928713157012712120) },
      { UINT64_C(16178077351148602638), UINT64_C(14357808486675351654) },
      { UINT64_C( 1251351760144394353), UINT64_C(14931405834110058180) } },
    { { UINT64_C( 9431923871987060279), UINT64_C(12707500777008466519) },
      UINT8_C( 98),
      { UINT64_C( 8995348668739776604), UINT64_C( 5668346931223215684) },
      { UINT64_C(11211600845510061395), UINT64_C( 8264365996282592050) },
      { UINT64_C( 9431923871987060279), UINT64_C( 5668346931223215684) } },
    { { UINT64_C(16054970495537362269), UINT64_C( 7044751376870607605) },
      UINT8_C(169),
      { UINT64_C(17557800160287433180), UINT64_C( 6727669379846764307) },
      { UINT64_C( 9658204129783688796), UINT64_C( 2075303184134602023) },
      { UINT64_C( 9658204129783688796), UINT64_C( 7044751376870607605) } },
    { { UINT64_C(17276156564626146966), UINT64_C(11761515769092885300) },
      UINT8_C(251),
      { UINT64_C( 4264063039301989936), UINT64_C( 2458288215618703250) },
      { UINT64_C( 8619026211143609131), UINT64_C(16223480646226311389) },
      { UINT64_C( 4264063039301989936), UINT64_C( 2458288215618703250) } },
    { { UINT64_C( 3175059293030751255), UINT64_C( 5737814776138140514) },
      UINT8_C(187),
      { UINT64_C(14657773487161653150), UINT64_C( 6130909521154708637) },
      { UINT64_C( 9707822333299845699), UINT64_C( 8456907805292050384) },
      { UINT64_C( 9707822333299845699), UINT64_C( 6130909521154708637) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_epu64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_epu64");
    easysimd_test_x86_assert_equal_u64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_u64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__m128i r = easysimd_mm_mask_min_epu64(src, k, a, b);

    easysimd_test_x86_write_u64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_min_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint64_t a[2];
    const uint64_t b[2];
    const uint64_t r[2];
  } test_vec[] = {
    { UINT8_C(106),
      { UINT64_C( 4253339538086563960), UINT64_C(13538312543965967962) },
      { UINT64_C( 5760074560487414532), UINT64_C( 9145247365561123924) },
      { UINT64_C(                   0), UINT64_C( 9145247365561123924) } },
    { UINT8_C(130),
      { UINT64_C( 5760248893959361200), UINT64_C( 5086412720355838674) },
      { UINT64_C(17467459567420112687), UINT64_C( 9380848295581706299) },
      { UINT64_C(                   0), UINT64_C( 5086412720355838674) } },
    { UINT8_C( 74),
      { UINT64_C(10650294080994744440), UINT64_C(17584380567783104328) },
      { UINT64_C( 7922909840272399574), UINT64_C( 5156553194326696141) },
      { UINT64_C(                   0), UINT64_C( 5156553194326696141) } },
    { UINT8_C(218),
      { UINT64_C( 8028922047405179826), UINT64_C( 7678452984814257524) },
      { UINT64_C(15770581399589832506), UINT64_C( 7077391079223927195) },
      { UINT64_C(                   0), UINT64_C( 7077391079223927195) } },
    { UINT8_C( 52),
      { UINT64_C( 4036550119926302234), UINT64_C(10568805157372869656) },
      { UINT64_C(13319579066438564576), UINT64_C( 9688873839171810640) },
      { UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C( 82),
      { UINT64_C(15458179223939978937), UINT64_C( 6760946077570103042) },
      { UINT64_C( 2051298264098279297), UINT64_C(16659675036793747477) },
      { UINT64_C(                   0), UINT64_C( 6760946077570103042) } },
    { UINT8_C( 36),
      { UINT64_C(15129304028342539383), UINT64_C(14730379090556414263) },
      { UINT64_C(14161516300860626117), UINT64_C( 1336399876420888664) },
      { UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C( 45),
      { UINT64_C(16579678762602792650), UINT64_C(  208132526540259661) },
      { UINT64_C( 4849200166383473698), UINT64_C(  560963400517989059) },
      { UINT64_C( 4849200166383473698), UINT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_epu64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_epu64");
    easysimd_test_x86_assert_equal_u64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__m128i r = easysimd_mm_maskz_min_epu64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -202.58), EASYSIMD_FLOAT32_C(   283.64), EASYSIMD_FLOAT32_C(    34.31), EASYSIMD_FLOAT32_C(    33.94) },
      UINT8_C(171),
      { EASYSIMD_FLOAT32_C(   336.07), EASYSIMD_FLOAT32_C(    89.87), EASYSIMD_FLOAT32_C(  -178.02), EASYSIMD_FLOAT32_C(   -94.77) },
      { EASYSIMD_FLOAT32_C(  -404.66), EASYSIMD_FLOAT32_C(  -243.10), EASYSIMD_FLOAT32_C(   416.29), EASYSIMD_FLOAT32_C(   611.74) },
      { EASYSIMD_FLOAT32_C(  -404.66), EASYSIMD_FLOAT32_C(  -243.10), EASYSIMD_FLOAT32_C(    34.31), EASYSIMD_FLOAT32_C(   -94.77) } },
    { { EASYSIMD_FLOAT32_C(   812.69), EASYSIMD_FLOAT32_C(  -178.74), EASYSIMD_FLOAT32_C(  -617.73), EASYSIMD_FLOAT32_C(   656.54) },
      UINT8_C(254),
      { EASYSIMD_FLOAT32_C(    76.63), EASYSIMD_FLOAT32_C(   947.24), EASYSIMD_FLOAT32_C(  -675.01), EASYSIMD_FLOAT32_C(  -206.15) },
      { EASYSIMD_FLOAT32_C(    56.63), EASYSIMD_FLOAT32_C(   774.56), EASYSIMD_FLOAT32_C(   177.89), EASYSIMD_FLOAT32_C(   728.38) },
      { EASYSIMD_FLOAT32_C(   812.69), EASYSIMD_FLOAT32_C(   774.56), EASYSIMD_FLOAT32_C(  -675.01), EASYSIMD_FLOAT32_C(  -206.15) } },
    { { EASYSIMD_FLOAT32_C(   119.49), EASYSIMD_FLOAT32_C(  -542.58), EASYSIMD_FLOAT32_C(   558.84), EASYSIMD_FLOAT32_C(   457.52) },
      UINT8_C( 13),
      { EASYSIMD_FLOAT32_C(  -643.74), EASYSIMD_FLOAT32_C(  -258.84), EASYSIMD_FLOAT32_C(  -736.98), EASYSIMD_FLOAT32_C(   390.20) },
      { EASYSIMD_FLOAT32_C(  -926.35), EASYSIMD_FLOAT32_C(   599.08), EASYSIMD_FLOAT32_C(  -519.93), EASYSIMD_FLOAT32_C(  -104.37) },
      { EASYSIMD_FLOAT32_C(  -926.35), EASYSIMD_FLOAT32_C(  -542.58), EASYSIMD_FLOAT32_C(  -736.98), EASYSIMD_FLOAT32_C(  -104.37) } },
    { { EASYSIMD_FLOAT32_C(  -495.68), EASYSIMD_FLOAT32_C(    75.41), EASYSIMD_FLOAT32_C(   652.53), EASYSIMD_FLOAT32_C(   920.60) },
      UINT8_C(  9),
      { EASYSIMD_FLOAT32_C(   465.22), EASYSIMD_FLOAT32_C(  -258.14), EASYSIMD_FLOAT32_C(    69.43), EASYSIMD_FLOAT32_C(   121.75) },
      { EASYSIMD_FLOAT32_C(  -638.39), EASYSIMD_FLOAT32_C(  -853.94), EASYSIMD_FLOAT32_C(    68.99), EASYSIMD_FLOAT32_C(  -313.39) },
      { EASYSIMD_FLOAT32_C(  -638.39), EASYSIMD_FLOAT32_C(    75.41), EASYSIMD_FLOAT32_C(   652.53), EASYSIMD_FLOAT32_C(  -313.39) } },
    { { EASYSIMD_FLOAT32_C(   -60.09), EASYSIMD_FLOAT32_C(  -874.38), EASYSIMD_FLOAT32_C(  -538.83), EASYSIMD_FLOAT32_C(  -882.20) },
      UINT8_C(141),
      { EASYSIMD_FLOAT32_C(   580.66), EASYSIMD_FLOAT32_C(  -424.78), EASYSIMD_FLOAT32_C(   412.84), EASYSIMD_FLOAT32_C(    38.18) },
      { EASYSIMD_FLOAT32_C(   803.94), EASYSIMD_FLOAT32_C(   769.09), EASYSIMD_FLOAT32_C(   779.35), EASYSIMD_FLOAT32_C(  -933.04) },
      { EASYSIMD_FLOAT32_C(   580.66), EASYSIMD_FLOAT32_C(  -874.38), EASYSIMD_FLOAT32_C(   412.84), EASYSIMD_FLOAT32_C(  -933.04) } },
    { { EASYSIMD_FLOAT32_C(   159.29), EASYSIMD_FLOAT32_C(   852.99), EASYSIMD_FLOAT32_C(   666.04), EASYSIMD_FLOAT32_C(   639.36) },
      UINT8_C( 10),
      { EASYSIMD_FLOAT32_C(  -829.64), EASYSIMD_FLOAT32_C(  -285.23), EASYSIMD_FLOAT32_C(  -598.84), EASYSIMD_FLOAT32_C(  -909.04) },
      { EASYSIMD_FLOAT32_C(   401.92), EASYSIMD_FLOAT32_C(   866.38), EASYSIMD_FLOAT32_C(  -167.18), EASYSIMD_FLOAT32_C(  -528.65) },
      { EASYSIMD_FLOAT32_C(   159.29), EASYSIMD_FLOAT32_C(  -285.23), EASYSIMD_FLOAT32_C(   666.04), EASYSIMD_FLOAT32_C(  -909.04) } },
    { { EASYSIMD_FLOAT32_C(   -11.87), EASYSIMD_FLOAT32_C(   194.44), EASYSIMD_FLOAT32_C(  -382.59), EASYSIMD_FLOAT32_C(  -942.88) },
      UINT8_C( 38),
      { EASYSIMD_FLOAT32_C(   557.32), EASYSIMD_FLOAT32_C(  -817.26), EASYSIMD_FLOAT32_C(  -657.79), EASYSIMD_FLOAT32_C(   675.12) },
      { EASYSIMD_FLOAT32_C(  -963.26), EASYSIMD_FLOAT32_C(   922.87), EASYSIMD_FLOAT32_C(  -749.65), EASYSIMD_FLOAT32_C(   449.57) },
      { EASYSIMD_FLOAT32_C(   -11.87), EASYSIMD_FLOAT32_C(  -817.26), EASYSIMD_FLOAT32_C(  -749.65), EASYSIMD_FLOAT32_C(  -942.88) } },
    { { EASYSIMD_FLOAT32_C(   -38.94), EASYSIMD_FLOAT32_C(  -945.72), EASYSIMD_FLOAT32_C(   218.67), EASYSIMD_FLOAT32_C(  -259.60) },
      UINT8_C(196),
      { EASYSIMD_FLOAT32_C(  -622.05), EASYSIMD_FLOAT32_C(  -406.60), EASYSIMD_FLOAT32_C(   787.28), EASYSIMD_FLOAT32_C(  -982.69) },
      { EASYSIMD_FLOAT32_C(   342.02), EASYSIMD_FLOAT32_C(   957.63), EASYSIMD_FLOAT32_C(  -267.91), EASYSIMD_FLOAT32_C(   743.18) },
      { EASYSIMD_FLOAT32_C(   -38.94), EASYSIMD_FLOAT32_C(  -945.72), EASYSIMD_FLOAT32_C(  -267.91), EASYSIMD_FLOAT32_C(  -259.60) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_ps");
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
    easysimd__m128 r = easysimd_mm_mask_min_ps(src, k, a, b);

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
test_easysimd_mm_maskz_min_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(189),
      { EASYSIMD_FLOAT32_C(  -179.86), EASYSIMD_FLOAT32_C(  -933.06), EASYSIMD_FLOAT32_C(   376.94), EASYSIMD_FLOAT32_C(   857.01) },
      { EASYSIMD_FLOAT32_C(   -20.00), EASYSIMD_FLOAT32_C(  -949.98), EASYSIMD_FLOAT32_C(  -190.95), EASYSIMD_FLOAT32_C(   498.57) },
      { EASYSIMD_FLOAT32_C(  -179.86), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -190.95), EASYSIMD_FLOAT32_C(   498.57) } },
    { UINT8_C(116),
      { EASYSIMD_FLOAT32_C(  -499.54), EASYSIMD_FLOAT32_C(   204.37), EASYSIMD_FLOAT32_C(   377.86), EASYSIMD_FLOAT32_C(   395.36) },
      { EASYSIMD_FLOAT32_C(  -443.40), EASYSIMD_FLOAT32_C(  -637.05), EASYSIMD_FLOAT32_C(   212.38), EASYSIMD_FLOAT32_C(  -693.23) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   212.38), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 17),
      { EASYSIMD_FLOAT32_C(   608.66), EASYSIMD_FLOAT32_C(  -909.81), EASYSIMD_FLOAT32_C(  -944.46), EASYSIMD_FLOAT32_C(  -629.98) },
      { EASYSIMD_FLOAT32_C(  -662.00), EASYSIMD_FLOAT32_C(  -613.08), EASYSIMD_FLOAT32_C(    67.74), EASYSIMD_FLOAT32_C(   964.30) },
      { EASYSIMD_FLOAT32_C(  -662.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(225),
      { EASYSIMD_FLOAT32_C(  -252.76), EASYSIMD_FLOAT32_C(   100.53), EASYSIMD_FLOAT32_C(  -687.14), EASYSIMD_FLOAT32_C(    44.82) },
      { EASYSIMD_FLOAT32_C(   920.68), EASYSIMD_FLOAT32_C(  -620.20), EASYSIMD_FLOAT32_C(  -578.24), EASYSIMD_FLOAT32_C(   777.69) },
      { EASYSIMD_FLOAT32_C(  -252.76), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(122),
      { EASYSIMD_FLOAT32_C(  -528.22), EASYSIMD_FLOAT32_C(  -413.26), EASYSIMD_FLOAT32_C(  -141.63), EASYSIMD_FLOAT32_C(   488.95) },
      { EASYSIMD_FLOAT32_C(    87.20), EASYSIMD_FLOAT32_C(  -937.26), EASYSIMD_FLOAT32_C(  -133.19), EASYSIMD_FLOAT32_C(  -517.44) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -937.26), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -517.44) } },
    { UINT8_C(  7),
      { EASYSIMD_FLOAT32_C(   229.75), EASYSIMD_FLOAT32_C(   694.95), EASYSIMD_FLOAT32_C(   -73.89), EASYSIMD_FLOAT32_C(  -512.38) },
      { EASYSIMD_FLOAT32_C(   303.60), EASYSIMD_FLOAT32_C(    16.30), EASYSIMD_FLOAT32_C(  -456.85), EASYSIMD_FLOAT32_C(   673.63) },
      { EASYSIMD_FLOAT32_C(   229.75), EASYSIMD_FLOAT32_C(    16.30), EASYSIMD_FLOAT32_C(  -456.85), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(102),
      { EASYSIMD_FLOAT32_C(   -69.93), EASYSIMD_FLOAT32_C(  -258.64), EASYSIMD_FLOAT32_C(   318.59), EASYSIMD_FLOAT32_C(   -65.55) },
      { EASYSIMD_FLOAT32_C(   488.60), EASYSIMD_FLOAT32_C(  -580.87), EASYSIMD_FLOAT32_C(   247.31), EASYSIMD_FLOAT32_C(  -466.58) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -580.87), EASYSIMD_FLOAT32_C(   247.31), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 24),
      { EASYSIMD_FLOAT32_C(   627.11), EASYSIMD_FLOAT32_C(   -44.82), EASYSIMD_FLOAT32_C(  -882.51), EASYSIMD_FLOAT32_C(   -13.09) },
      { EASYSIMD_FLOAT32_C(   426.96), EASYSIMD_FLOAT32_C(  -295.77), EASYSIMD_FLOAT32_C(   845.28), EASYSIMD_FLOAT32_C(   -84.09) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -84.09) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_min_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_min_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(    76.49), EASYSIMD_FLOAT64_C(  -150.45) },
      UINT8_C(205),
      { EASYSIMD_FLOAT64_C(  -524.99), EASYSIMD_FLOAT64_C(   -51.00) },
      { EASYSIMD_FLOAT64_C(   697.00), EASYSIMD_FLOAT64_C(   189.14) },
      { EASYSIMD_FLOAT64_C(  -524.99), EASYSIMD_FLOAT64_C(  -150.45) } },
    { { EASYSIMD_FLOAT64_C(  -538.54), EASYSIMD_FLOAT64_C(   168.77) },
      UINT8_C(100),
      { EASYSIMD_FLOAT64_C(  -262.67), EASYSIMD_FLOAT64_C(   245.96) },
      { EASYSIMD_FLOAT64_C(   561.08), EASYSIMD_FLOAT64_C(  -951.66) },
      { EASYSIMD_FLOAT64_C(  -538.54), EASYSIMD_FLOAT64_C(   168.77) } },
    { { EASYSIMD_FLOAT64_C(  -357.57), EASYSIMD_FLOAT64_C(  -106.24) },
      UINT8_C(212),
      { EASYSIMD_FLOAT64_C(   749.56), EASYSIMD_FLOAT64_C(   212.79) },
      { EASYSIMD_FLOAT64_C(    80.76), EASYSIMD_FLOAT64_C(   401.86) },
      { EASYSIMD_FLOAT64_C(  -357.57), EASYSIMD_FLOAT64_C(  -106.24) } },
    { { EASYSIMD_FLOAT64_C(   608.27), EASYSIMD_FLOAT64_C(  -694.27) },
      UINT8_C(186),
      { EASYSIMD_FLOAT64_C(  -998.10), EASYSIMD_FLOAT64_C(   445.60) },
      { EASYSIMD_FLOAT64_C(   579.45), EASYSIMD_FLOAT64_C(  -194.68) },
      { EASYSIMD_FLOAT64_C(   608.27), EASYSIMD_FLOAT64_C(  -194.68) } },
    { { EASYSIMD_FLOAT64_C(  -101.23), EASYSIMD_FLOAT64_C(  -692.24) },
      UINT8_C( 88),
      { EASYSIMD_FLOAT64_C(   975.26), EASYSIMD_FLOAT64_C(   157.31) },
      { EASYSIMD_FLOAT64_C(   748.34), EASYSIMD_FLOAT64_C(  -549.73) },
      { EASYSIMD_FLOAT64_C(  -101.23), EASYSIMD_FLOAT64_C(  -692.24) } },
    { { EASYSIMD_FLOAT64_C(  -893.69), EASYSIMD_FLOAT64_C(   445.34) },
      UINT8_C(187),
      { EASYSIMD_FLOAT64_C(  -432.23), EASYSIMD_FLOAT64_C(  -385.89) },
      { EASYSIMD_FLOAT64_C(  -815.36), EASYSIMD_FLOAT64_C(   305.10) },
      { EASYSIMD_FLOAT64_C(  -815.36), EASYSIMD_FLOAT64_C(  -385.89) } },
    { { EASYSIMD_FLOAT64_C(   860.08), EASYSIMD_FLOAT64_C(   745.72) },
      UINT8_C(194),
      { EASYSIMD_FLOAT64_C(  -497.49), EASYSIMD_FLOAT64_C(  -360.52) },
      { EASYSIMD_FLOAT64_C(   678.88), EASYSIMD_FLOAT64_C(  -747.93) },
      { EASYSIMD_FLOAT64_C(   860.08), EASYSIMD_FLOAT64_C(  -747.93) } },
    { { EASYSIMD_FLOAT64_C(   852.27), EASYSIMD_FLOAT64_C(  -240.36) },
      UINT8_C(238),
      { EASYSIMD_FLOAT64_C(   460.53), EASYSIMD_FLOAT64_C(    65.36) },
      { EASYSIMD_FLOAT64_C(  -376.82), EASYSIMD_FLOAT64_C(   462.44) },
      { EASYSIMD_FLOAT64_C(   852.27), EASYSIMD_FLOAT64_C(    65.36) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_min_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_min_pd");
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
    easysimd__m128d r = easysimd_mm_mask_min_pd(src, k, a, b);

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
test_easysimd_mm_maskz_min_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C( 33),
      { EASYSIMD_FLOAT64_C(   -33.36), EASYSIMD_FLOAT64_C(  -409.50) },
      { EASYSIMD_FLOAT64_C(   452.09), EASYSIMD_FLOAT64_C(  -533.06) },
      { EASYSIMD_FLOAT64_C(   -33.36), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(131),
      { EASYSIMD_FLOAT64_C(    40.77), EASYSIMD_FLOAT64_C(   789.34) },
      { EASYSIMD_FLOAT64_C(   697.27), EASYSIMD_FLOAT64_C(  -293.26) },
      { EASYSIMD_FLOAT64_C(    40.77), EASYSIMD_FLOAT64_C(  -293.26) } },
    { UINT8_C(208),
      { EASYSIMD_FLOAT64_C(  -614.49), EASYSIMD_FLOAT64_C(  -346.16) },
      { EASYSIMD_FLOAT64_C(   483.73), EASYSIMD_FLOAT64_C(   815.41) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(168),
      { EASYSIMD_FLOAT64_C(   160.24), EASYSIMD_FLOAT64_C(   345.44) },
      { EASYSIMD_FLOAT64_C(   342.29), EASYSIMD_FLOAT64_C(   572.17) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(110),
      { EASYSIMD_FLOAT64_C(  -653.70), EASYSIMD_FLOAT64_C(   423.71) },
      { EASYSIMD_FLOAT64_C(  -753.00), EASYSIMD_FLOAT64_C(   818.26) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   423.71) } },
    { UINT8_C(195),
      { EASYSIMD_FLOAT64_C(   390.81), EASYSIMD_FLOAT64_C(   466.22) },
      { EASYSIMD_FLOAT64_C(   462.19), EASYSIMD_FLOAT64_C(  -301.16) },
      { EASYSIMD_FLOAT64_C(   390.81), EASYSIMD_FLOAT64_C(  -301.16) } },
    { UINT8_C(206),
      { EASYSIMD_FLOAT64_C(  -362.89), EASYSIMD_FLOAT64_C(   665.48) },
      { EASYSIMD_FLOAT64_C(    57.11), EASYSIMD_FLOAT64_C(  -910.80) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -910.80) } },
    { UINT8_C( 36),
      { EASYSIMD_FLOAT64_C(   101.20), EASYSIMD_FLOAT64_C(   129.97) },
      { EASYSIMD_FLOAT64_C(   921.76), EASYSIMD_FLOAT64_C(  -201.52) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_min_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_min_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_min_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_min_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   284.85), EASYSIMD_FLOAT32_C(  -637.09), EASYSIMD_FLOAT32_C(   147.92), EASYSIMD_FLOAT32_C(   352.35),
        EASYSIMD_FLOAT32_C(  -142.22), EASYSIMD_FLOAT32_C(   -33.05), EASYSIMD_FLOAT32_C(   949.75), EASYSIMD_FLOAT32_C(   276.70) },
      UINT8_C(251),
      { EASYSIMD_FLOAT32_C(  -398.58), EASYSIMD_FLOAT32_C(  -279.10), EASYSIMD_FLOAT32_C(  -146.14), EASYSIMD_FLOAT32_C(   789.41),
        EASYSIMD_FLOAT32_C(  -109.69), EASYSIMD_FLOAT32_C(   493.67), EASYSIMD_FLOAT32_C(  -353.00), EASYSIMD_FLOAT32_C(  -973.96) },
      { EASYSIMD_FLOAT32_C(   831.93), EASYSIMD_FLOAT32_C(  -842.13), EASYSIMD_FLOAT32_C(   602.63), EASYSIMD_FLOAT32_C(  -458.35),
        EASYSIMD_FLOAT32_C(  -178.64), EASYSIMD_FLOAT32_C(   819.81), EASYSIMD_FLOAT32_C(  -453.98), EASYSIMD_FLOAT32_C(   798.06) },
      { EASYSIMD_FLOAT32_C(  -398.58), EASYSIMD_FLOAT32_C(  -842.13), EASYSIMD_FLOAT32_C(   147.92), EASYSIMD_FLOAT32_C(  -458.35),
        EASYSIMD_FLOAT32_C(  -178.64), EASYSIMD_FLOAT32_C(   493.67), EASYSIMD_FLOAT32_C(  -453.98), EASYSIMD_FLOAT32_C(  -973.96) } },
    { { EASYSIMD_FLOAT32_C(    94.58), EASYSIMD_FLOAT32_C(  -416.30), EASYSIMD_FLOAT32_C(   856.61), EASYSIMD_FLOAT32_C(  -510.71),
        EASYSIMD_FLOAT32_C(   175.07), EASYSIMD_FLOAT32_C(   757.48), EASYSIMD_FLOAT32_C(   774.15), EASYSIMD_FLOAT32_C(   537.98) },
      UINT8_C(108),
      { EASYSIMD_FLOAT32_C(   126.49), EASYSIMD_FLOAT32_C(  -604.24), EASYSIMD_FLOAT32_C(   872.35), EASYSIMD_FLOAT32_C(    76.24),
        EASYSIMD_FLOAT32_C(   672.47), EASYSIMD_FLOAT32_C(   220.06), EASYSIMD_FLOAT32_C(   677.66), EASYSIMD_FLOAT32_C(  -606.64) },
      { EASYSIMD_FLOAT32_C(  -926.09), EASYSIMD_FLOAT32_C(   467.07), EASYSIMD_FLOAT32_C(   283.68), EASYSIMD_FLOAT32_C(   567.59),
        EASYSIMD_FLOAT32_C(  -885.93), EASYSIMD_FLOAT32_C(   309.72), EASYSIMD_FLOAT32_C(   399.51), EASYSIMD_FLOAT32_C(  -728.06) },
      { EASYSIMD_FLOAT32_C(    94.58), EASYSIMD_FLOAT32_C(  -416.30), EASYSIMD_FLOAT32_C(   283.68), EASYSIMD_FLOAT32_C(    76.24),
        EASYSIMD_FLOAT32_C(   175.07), EASYSIMD_FLOAT32_C(   220.06), EASYSIMD_FLOAT32_C(   399.51), EASYSIMD_FLOAT32_C(   537.98) } },
    { { EASYSIMD_FLOAT32_C(   -87.65), EASYSIMD_FLOAT32_C(   941.16), EASYSIMD_FLOAT32_C(    93.30), EASYSIMD_FLOAT32_C(  -267.85),
        EASYSIMD_FLOAT32_C(  -512.82), EASYSIMD_FLOAT32_C(  -108.65), EASYSIMD_FLOAT32_C(   826.74), EASYSIMD_FLOAT32_C(    70.89) },
      UINT8_C(239),
      { EASYSIMD_FLOAT32_C(  -683.97), EASYSIMD_FLOAT32_C(  -754.04), EASYSIMD_FLOAT32_C(  -494.55), EASYSIMD_FLOAT32_C(  -909.82),
        EASYSIMD_FLOAT32_C(   783.94), EASYSIMD_FLOAT32_C(   410.85), EASYSIMD_FLOAT32_C(   216.67), EASYSIMD_FLOAT32_C(  -820.29) },
      { EASYSIMD_FLOAT32_C(   283.21), EASYSIMD_FLOAT32_C(  -707.09), EASYSIMD_FLOAT32_C(   852.17), EASYSIMD_FLOAT32_C(  -496.74),
        EASYSIMD_FLOAT32_C(   970.57), EASYSIMD_FLOAT32_C(  -754.46), EASYSIMD_FLOAT32_C(  -422.83), EASYSIMD_FLOAT32_C(   437.65) },
      { EASYSIMD_FLOAT32_C(  -683.97), EASYSIMD_FLOAT32_C(  -754.04), EASYSIMD_FLOAT32_C(  -494.55), EASYSIMD_FLOAT32_C(  -909.82),
        EASYSIMD_FLOAT32_C(  -512.82), EASYSIMD_FLOAT32_C(  -754.46), EASYSIMD_FLOAT32_C(  -422.83), EASYSIMD_FLOAT32_C(  -820.29) } },
    { { EASYSIMD_FLOAT32_C(   529.21), EASYSIMD_FLOAT32_C(  -855.24), EASYSIMD_FLOAT32_C(   551.72), EASYSIMD_FLOAT32_C(  -161.07),
        EASYSIMD_FLOAT32_C(   544.28), EASYSIMD_FLOAT32_C(   823.66), EASYSIMD_FLOAT32_C(   751.28), EASYSIMD_FLOAT32_C(   485.44) },
      UINT8_C( 32),
      { EASYSIMD_FLOAT32_C(  -516.57), EASYSIMD_FLOAT32_C(   972.62), EASYSIMD_FLOAT32_C(   808.31), EASYSIMD_FLOAT32_C(  -689.83),
        EASYSIMD_FLOAT32_C(    43.51), EASYSIMD_FLOAT32_C(  -443.72), EASYSIMD_FLOAT32_C(  -373.80), EASYSIMD_FLOAT32_C(   289.47) },
      { EASYSIMD_FLOAT32_C(    61.73), EASYSIMD_FLOAT32_C(  -283.63), EASYSIMD_FLOAT32_C(    73.42), EASYSIMD_FLOAT32_C(  -527.42),
        EASYSIMD_FLOAT32_C(   933.04), EASYSIMD_FLOAT32_C(   253.12), EASYSIMD_FLOAT32_C(   755.79), EASYSIMD_FLOAT32_C(  -774.04) },
      { EASYSIMD_FLOAT32_C(   529.21), EASYSIMD_FLOAT32_C(  -855.24), EASYSIMD_FLOAT32_C(   551.72), EASYSIMD_FLOAT32_C(  -161.07),
        EASYSIMD_FLOAT32_C(   544.28), EASYSIMD_FLOAT32_C(  -443.72), EASYSIMD_FLOAT32_C(   751.28), EASYSIMD_FLOAT32_C(   485.44) } },
    { { EASYSIMD_FLOAT32_C(   105.30), EASYSIMD_FLOAT32_C(  -740.95), EASYSIMD_FLOAT32_C(  -803.47), EASYSIMD_FLOAT32_C(   350.84),
        EASYSIMD_FLOAT32_C(  -163.78), EASYSIMD_FLOAT32_C(   634.18), EASYSIMD_FLOAT32_C(  -119.95), EASYSIMD_FLOAT32_C(   -19.02) },
      UINT8_C(117),
      { EASYSIMD_FLOAT32_C(   718.98), EASYSIMD_FLOAT32_C(  -474.74), EASYSIMD_FLOAT32_C(     9.56), EASYSIMD_FLOAT32_C(   470.26),
        EASYSIMD_FLOAT32_C(  -989.30), EASYSIMD_FLOAT32_C(   926.51), EASYSIMD_FLOAT32_C(   953.69), EASYSIMD_FLOAT32_C(   983.32) },
      { EASYSIMD_FLOAT32_C(   734.82), EASYSIMD_FLOAT32_C(  -736.14), EASYSIMD_FLOAT32_C(    26.83), EASYSIMD_FLOAT32_C(  -708.90),
        EASYSIMD_FLOAT32_C(  -109.94), EASYSIMD_FLOAT32_C(  -683.70), EASYSIMD_FLOAT32_C(   352.82), EASYSIMD_FLOAT32_C(   606.43) },
      { EASYSIMD_FLOAT32_C(   718.98), EASYSIMD_FLOAT32_C(  -740.95), EASYSIMD_FLOAT32_C(     9.56), EASYSIMD_FLOAT32_C(   350.84),
        EASYSIMD_FLOAT32_C(  -989.30), EASYSIMD_FLOAT32_C(  -683.70), EASYSIMD_FLOAT32_C(   352.82), EASYSIMD_FLOAT32_C(   -19.02) } },
    { { EASYSIMD_FLOAT32_C(   389.72), EASYSIMD_FLOAT32_C(   825.40), EASYSIMD_FLOAT32_C(   539.47), EASYSIMD_FLOAT32_C(  -357.16),
        EASYSIMD_FLOAT32_C(   581.19), EASYSIMD_FLOAT32_C(   765.43), EASYSIMD_FLOAT32_C(   748.14), EASYSIMD_FLOAT32_C(   840.24) },
      UINT8_C(212),
      { EASYSIMD_FLOAT32_C(    98.97), EASYSIMD_FLOAT32_C(  -323.54), EASYSIMD_FLOAT32_C(   596.13), EASYSIMD_FLOAT32_C(   979.02),
        EASYSIMD_FLOAT32_C(   657.44), EASYSIMD_FLOAT32_C(  -217.97), EASYSIMD_FLOAT32_C(   698.01), EASYSIMD_FLOAT32_C(  -817.30) },
      { EASYSIMD_FLOAT32_C(   791.58), EASYSIMD_FLOAT32_C(   168.26), EASYSIMD_FLOAT32_C(  -806.60), EASYSIMD_FLOAT32_C(   718.10),
        EASYSIMD_FLOAT32_C(   121.95), EASYSIMD_FLOAT32_C(  -823.28), EASYSIMD_FLOAT32_C(   452.92), EASYSIMD_FLOAT32_C(   385.81) },
      { EASYSIMD_FLOAT32_C(   389.72), EASYSIMD_FLOAT32_C(   825.40), EASYSIMD_FLOAT32_C(  -806.60), EASYSIMD_FLOAT32_C(  -357.16),
        EASYSIMD_FLOAT32_C(   121.95), EASYSIMD_FLOAT32_C(   765.43), EASYSIMD_FLOAT32_C(   452.92), EASYSIMD_FLOAT32_C(  -817.30) } },
    { { EASYSIMD_FLOAT32_C(   203.55), EASYSIMD_FLOAT32_C(   744.02), EASYSIMD_FLOAT32_C(  -724.13), EASYSIMD_FLOAT32_C(   519.84),
        EASYSIMD_FLOAT32_C(    96.84), EASYSIMD_FLOAT32_C(   882.30), EASYSIMD_FLOAT32_C(   -90.44), EASYSIMD_FLOAT32_C(   -77.76) },
      UINT8_C(146),
      { EASYSIMD_FLOAT32_C(   552.40), EASYSIMD_FLOAT32_C(  -496.57), EASYSIMD_FLOAT32_C(   187.20), EASYSIMD_FLOAT32_C(   300.54),
        EASYSIMD_FLOAT32_C(  -656.33), EASYSIMD_FLOAT32_C(   149.16), EASYSIMD_FLOAT32_C(  -600.49), EASYSIMD_FLOAT32_C(    20.13) },
      { EASYSIMD_FLOAT32_C(  -254.71), EASYSIMD_FLOAT32_C(  -621.46), EASYSIMD_FLOAT32_C(  -322.43), EASYSIMD_FLOAT32_C(   527.32),
        EASYSIMD_FLOAT32_C(  -923.46), EASYSIMD_FLOAT32_C(  -139.73), EASYSIMD_FLOAT32_C(   318.90), EASYSIMD_FLOAT32_C(   244.81) },
      { EASYSIMD_FLOAT32_C(   203.55), EASYSIMD_FLOAT32_C(  -621.46), EASYSIMD_FLOAT32_C(  -724.13), EASYSIMD_FLOAT32_C(   519.84),
        EASYSIMD_FLOAT32_C(  -923.46), EASYSIMD_FLOAT32_C(   882.30), EASYSIMD_FLOAT32_C(   -90.44), EASYSIMD_FLOAT32_C(    20.13) } },
    { { EASYSIMD_FLOAT32_C(    53.67), EASYSIMD_FLOAT32_C(    37.00), EASYSIMD_FLOAT32_C(  -633.24), EASYSIMD_FLOAT32_C(   230.39),
        EASYSIMD_FLOAT32_C(  -510.08), EASYSIMD_FLOAT32_C(   752.57), EASYSIMD_FLOAT32_C(  -566.07), EASYSIMD_FLOAT32_C(  -766.07) },
      UINT8_C(240),
      { EASYSIMD_FLOAT32_C(   953.77), EASYSIMD_FLOAT32_C(   330.77), EASYSIMD_FLOAT32_C(   910.74), EASYSIMD_FLOAT32_C(  -136.67),
        EASYSIMD_FLOAT32_C(  -746.98), EASYSIMD_FLOAT32_C(   332.51), EASYSIMD_FLOAT32_C(  -584.27), EASYSIMD_FLOAT32_C(  -243.55) },
      { EASYSIMD_FLOAT32_C(  -480.29), EASYSIMD_FLOAT32_C(   716.27), EASYSIMD_FLOAT32_C(   100.12), EASYSIMD_FLOAT32_C(   668.87),
        EASYSIMD_FLOAT32_C(  -884.22), EASYSIMD_FLOAT32_C(  -879.76), EASYSIMD_FLOAT32_C(  -585.84), EASYSIMD_FLOAT32_C(  -505.68) },
      { EASYSIMD_FLOAT32_C(    53.67), EASYSIMD_FLOAT32_C(    37.00), EASYSIMD_FLOAT32_C(  -633.24), EASYSIMD_FLOAT32_C(   230.39),
        EASYSIMD_FLOAT32_C(  -884.22), EASYSIMD_FLOAT32_C(  -879.76), EASYSIMD_FLOAT32_C(  -585.84), EASYSIMD_FLOAT32_C(  -505.68) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_ps");
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
    easysimd__m256 r = easysimd_mm256_mask_min_ps(src, k, a, b);

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
test_easysimd_mm256_mask_min_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t src[32];
    const uint32_t k;
    const int8_t a[32];
    const int8_t b[32];
    const int8_t r[32];
  } test_vec[] = {
    { { -INT8_C( 105), -INT8_C(  85),  INT8_C( 112),  INT8_C(  57), -INT8_C(   3),  INT8_C(  92), -INT8_C(  36),  INT8_C(  62),
         INT8_C(  91), -INT8_C( 124),  INT8_C(  35), -INT8_C(  58), -INT8_C(  12),  INT8_C(  25),  INT8_C(  35),  INT8_C(  33),
        -INT8_C( 124), -INT8_C( 106),  INT8_C(  64), -INT8_C(  46), -INT8_C(  52),  INT8_C(  15), -INT8_C(  94), -INT8_C( 119),
        -INT8_C(  70), -INT8_C(  12), -INT8_C(  15),  INT8_C(  59), -INT8_C(  72), -INT8_C(  53), -INT8_C(  93),  INT8_C(  80) },
      UINT32_C(1938363510),
      {  INT8_C( 112),  INT8_C( 101), -INT8_C(  78), -INT8_C(  53), -INT8_C(  23), -INT8_C(  43), -INT8_C( 111), -INT8_C(  35),
        -INT8_C(  18), -INT8_C(  75), -INT8_C(   2),  INT8_C( 115),  INT8_C(  75),  INT8_C(  63),  INT8_C(  69),  INT8_C(  24),
         INT8_C(  78), -INT8_C(  25), -INT8_C(  95),  INT8_C(   8), -INT8_C(  37), -INT8_C( 110),  INT8_C(  67), -INT8_C( 108),
         INT8_C(  93), -INT8_C(  25), -INT8_C(  28), -INT8_C(  44), -INT8_C(   5),  INT8_C( 109),  INT8_C(  71),  INT8_C( 107) },
      { -INT8_C(  46), -INT8_C(   7),  INT8_C(  54), -INT8_C(  69), -INT8_C(  50), -INT8_C(  56), -INT8_C( 104), -INT8_C(  67),
         INT8_C( 125), -INT8_C( 105),  INT8_C(  48), -INT8_C(  56), -INT8_C(  42),  INT8_C( 117), -INT8_C(  32),  INT8_C(  36),
         INT8_C(  93), -INT8_C( 126),  INT8_C(  44),  INT8_C(  56),  INT8_C(  20),  INT8_C( 112), -INT8_C(  52),  INT8_C( 114),
         INT8_C(  87), -INT8_C(  80),  INT8_C(  70),  INT8_C(  82),  INT8_C(  29), -INT8_C( 115), -INT8_C(  67), -INT8_C(  17) },
      { -INT8_C( 105), -INT8_C(   7), -INT8_C(  78),  INT8_C(  57), -INT8_C(  50), -INT8_C(  56), -INT8_C( 111),  INT8_C(  62),
         INT8_C(  91), -INT8_C( 124), -INT8_C(   2), -INT8_C(  58), -INT8_C(  42),  INT8_C(  25),  INT8_C(  35),  INT8_C(  33),
         INT8_C(  78), -INT8_C( 106),  INT8_C(  64),  INT8_C(   8), -INT8_C(  52),  INT8_C(  15), -INT8_C(  94), -INT8_C( 108),
         INT8_C(  87), -INT8_C(  80), -INT8_C(  15),  INT8_C(  59), -INT8_C(   5), -INT8_C( 115), -INT8_C(  67),  INT8_C(  80) } },
    { { -INT8_C( 121), -INT8_C(  13), -INT8_C(  86),  INT8_C(  85), -INT8_C(  69),  INT8_C(  67),  INT8_C(  18),  INT8_C(  56),
        -INT8_C(  38),  INT8_C(  66),  INT8_C(   1), -INT8_C(  80), -INT8_C(  72), -INT8_C(  31), -INT8_C(  44),  INT8_C(  21),
         INT8_C(  99),  INT8_C(   0),  INT8_C(  77),  INT8_C( 120),  INT8_C( 112),  INT8_C(  26), -INT8_C(  22), -INT8_C(  57),
        -INT8_C(  54),  INT8_C(  48),  INT8_C(  25), -INT8_C(  24), -INT8_C(  67), -INT8_C(  42), -INT8_C(  41),  INT8_C(  68) },
      UINT32_C(2241495754),
      { -INT8_C(  59), -INT8_C(  84), -INT8_C(  66), -INT8_C(  97), -INT8_C(  17), -INT8_C(  65),  INT8_C(  79), -INT8_C(  89),
        -INT8_C(  96),  INT8_C(  35), -INT8_C(  68),  INT8_C(   4),  INT8_C(  35),  INT8_C(   9),  INT8_C( 124), -INT8_C( 108),
         INT8_C(  35),  INT8_C( 102),  INT8_C(  91), -INT8_C(  18), -INT8_C( 106),  INT8_C( 117), -INT8_C(  42),  INT8_C(  83),
         INT8_C(  75), -INT8_C(  83), -INT8_C( 104),  INT8_C(  21),  INT8_C(  47),  INT8_C(  50), -INT8_C( 101), -INT8_C(  12) },
      { -INT8_C(  34),  INT8_C(  89), -INT8_C( 109), -INT8_C(  51),  INT8_C(  24), -INT8_C(  30),  INT8_C( 116), -INT8_C(  72),
         INT8_C(   5),  INT8_C(  48), -INT8_C(  68),  INT8_C(  41),  INT8_C(  58),  INT8_C(  56), -INT8_C(  67),  INT8_C(  93),
        -INT8_C(  98),  INT8_C(  24),  INT8_C(  75),  INT8_C(  52), -INT8_C( 115),  INT8_C(  33), -INT8_C( 120), -INT8_C(  39),
        -INT8_C(  49),  INT8_C(  32), -INT8_C(  18), -INT8_C(   2),  INT8_C(  82), -INT8_C( 119), -INT8_C(  13),  INT8_C(  48) },
      { -INT8_C( 121), -INT8_C(  84), -INT8_C(  86), -INT8_C(  97), -INT8_C(  69),  INT8_C(  67),  INT8_C(  79), -INT8_C(  89),
        -INT8_C(  38),  INT8_C(  35),  INT8_C(   1), -INT8_C(  80), -INT8_C(  72), -INT8_C(  31), -INT8_C(  44), -INT8_C( 108),
         INT8_C(  99),  INT8_C(  24),  INT8_C(  77), -INT8_C(  18), -INT8_C( 115),  INT8_C(  26), -INT8_C(  22), -INT8_C(  39),
        -INT8_C(  49),  INT8_C(  48), -INT8_C( 104), -INT8_C(  24), -INT8_C(  67), -INT8_C(  42), -INT8_C(  41), -INT8_C(  12) } },
    { { -INT8_C(  30), -INT8_C( 122), -INT8_C(   2), -INT8_C(   6),  INT8_C( 105),  INT8_C( 114), -INT8_C(  77),  INT8_C( 110),
        -INT8_C(  93),  INT8_C( 111), -INT8_C( 105), -INT8_C(  35), -INT8_C(  88),  INT8_C(  84),  INT8_C(  58),  INT8_C(  70),
         INT8_C( 109), -INT8_C( 122),  INT8_C( 123), -INT8_C(   6), -INT8_C(  89),  INT8_C(   3), -INT8_C(  45),  INT8_C( 118),
         INT8_C(  35), -INT8_C(  62),  INT8_C( 117),  INT8_C( 117),  INT8_C(  75),  INT8_C( 104), -INT8_C(  91),  INT8_C(  46) },
      UINT32_C(1462281198),
      {  INT8_C(  22), -INT8_C(  37), -INT8_C(  58), -INT8_C(  71),  INT8_C(  75),  INT8_C(  93), -INT8_C( 106), -INT8_C(  13),
        -INT8_C(  78), -INT8_C(  48),  INT8_C(  57),  INT8_C(  31),  INT8_C(  86), -INT8_C(  76),  INT8_C(  25), -INT8_C(   2),
        -INT8_C(  73), -INT8_C(  19),  INT8_C( 116), -INT8_C(  38), -INT8_C(  81), -INT8_C(  23),  INT8_C(  79), -INT8_C(   6),
         INT8_C(  81), -INT8_C(  11),  INT8_C(  40),  INT8_C(  64), -INT8_C( 104),  INT8_C(  81), -INT8_C( 105), -INT8_C(  82) },
      {  INT8_C(  44),  INT8_C(  93),  INT8_C( 103),  INT8_C( 119), -INT8_C(  69), -INT8_C(   3),  INT8_C( 106),  INT8_C( 109),
        -INT8_C(  50), -INT8_C(  92), -INT8_C( 116),  INT8_C(  36),  INT8_C(  88), -INT8_C(  91),  INT8_C(  34),  INT8_C(  16),
        -INT8_C( 110), -INT8_C( 105), -INT8_C(  22),  INT8_C(  65),      INT8_MIN,  INT8_C(  58),  INT8_C(  60), -INT8_C(  46),
         INT8_C(  47),  INT8_C( 100),  INT8_C(  18), -INT8_C(  57), -INT8_C(  75), -INT8_C(  87),  INT8_C( 118), -INT8_C(  30) },
      { -INT8_C(  30), -INT8_C(  37), -INT8_C(  58), -INT8_C(  71),  INT8_C( 105), -INT8_C(   3), -INT8_C( 106), -INT8_C(  13),
        -INT8_C(  78), -INT8_C(  92), -INT8_C( 105), -INT8_C(  35), -INT8_C(  88), -INT8_C(  91),  INT8_C(  58), -INT8_C(   2),
         INT8_C( 109), -INT8_C( 122),  INT8_C( 123), -INT8_C(  38), -INT8_C(  89), -INT8_C(  23), -INT8_C(  45),  INT8_C( 118),
         INT8_C(  47), -INT8_C(  11),  INT8_C(  18),  INT8_C( 117), -INT8_C( 104),  INT8_C( 104), -INT8_C( 105),  INT8_C(  46) } },
    { {  INT8_C(   7), -INT8_C(  35),  INT8_C(  89), -INT8_C(  62), -INT8_C(  37), -INT8_C(  60),  INT8_C(  47), -INT8_C(  87),
         INT8_C( 104), -INT8_C(  69), -INT8_C(  51), -INT8_C(  64),  INT8_C(  96), -INT8_C(  16), -INT8_C(  48), -INT8_C(  13),
        -INT8_C( 121), -INT8_C(  69),  INT8_C(  52),  INT8_C(   7), -INT8_C(  11),  INT8_C( 112), -INT8_C(  39),  INT8_C(  36),
        -INT8_C(  43), -INT8_C(  21), -INT8_C(  21), -INT8_C( 118), -INT8_C( 107),  INT8_C(  97),  INT8_C( 108), -INT8_C( 100) },
      UINT32_C( 442418751),
      { -INT8_C( 118), -INT8_C( 115), -INT8_C(  61), -INT8_C(  14),  INT8_C(  72), -INT8_C( 112), -INT8_C(  78), -INT8_C(  88),
             INT8_MIN, -INT8_C( 125), -INT8_C( 101),  INT8_C(   7),  INT8_C(  62), -INT8_C(  48),  INT8_C(  15),  INT8_C(  51),
         INT8_C(  64), -INT8_C(  24),  INT8_C(  87),  INT8_C(  21), -INT8_C(  44),  INT8_C(  66), -INT8_C(  96),  INT8_C( 105),
        -INT8_C(  92),  INT8_C(  12),  INT8_C(   5), -INT8_C(  29), -INT8_C(  46),  INT8_C(  99), -INT8_C(   3),  INT8_C(  92) },
      { -INT8_C(  16), -INT8_C(  64),  INT8_C(  78),  INT8_C(  56),  INT8_C(  80),  INT8_C(   1), -INT8_C(  32), -INT8_C(  47),
        -INT8_C( 124),  INT8_C( 124), -INT8_C(  40), -INT8_C(  62),  INT8_C(  76), -INT8_C(  25), -INT8_C(  11), -INT8_C( 116),
        -INT8_C(  48),  INT8_C(  76), -INT8_C(  94), -INT8_C(  92), -INT8_C( 114),  INT8_C(  66),  INT8_C(  13),  INT8_C(  50),
         INT8_C(  78),  INT8_C(  18),  INT8_C(  21),  INT8_C(  33),  INT8_C( 117),  INT8_C(  18),  INT8_C( 125),  INT8_C( 101) },
      { -INT8_C( 118), -INT8_C( 115), -INT8_C(  61), -INT8_C(  14),  INT8_C(  72), -INT8_C( 112),  INT8_C(  47), -INT8_C(  87),
         INT8_C( 104), -INT8_C( 125), -INT8_C( 101), -INT8_C(  64),  INT8_C(  96), -INT8_C(  16), -INT8_C(  11), -INT8_C( 116),
        -INT8_C( 121), -INT8_C(  24), -INT8_C(  94), -INT8_C(  92), -INT8_C( 114),  INT8_C( 112), -INT8_C(  96),  INT8_C(  36),
        -INT8_C(  43),  INT8_C(  12), -INT8_C(  21), -INT8_C(  29), -INT8_C(  46),  INT8_C(  97),  INT8_C( 108), -INT8_C( 100) } },
    { { -INT8_C(  46), -INT8_C(  52), -INT8_C(  99),  INT8_C(  35), -INT8_C(  51),  INT8_C( 125), -INT8_C(  12),  INT8_C(  81),
        -INT8_C(   7), -INT8_C(  52),  INT8_C(  19),  INT8_C(  69), -INT8_C(  76),  INT8_C(   8), -INT8_C(  46), -INT8_C( 124),
         INT8_C(  84),  INT8_C( 116),  INT8_C(  40), -INT8_C(  30), -INT8_C(  74),  INT8_C(  53),  INT8_C(  21),  INT8_C(   4),
         INT8_C(  71),  INT8_C(  42),  INT8_C(  37), -INT8_C(  68),  INT8_C(  61), -INT8_C(  93),  INT8_C(  33),  INT8_C(  15) },
      UINT32_C(1009958511),
      {  INT8_C(  59),  INT8_C(  38), -INT8_C( 115),  INT8_C(  53), -INT8_C(  13), -INT8_C(  96),  INT8_C( 122), -INT8_C(  89),
        -INT8_C(  88),  INT8_C(  76),  INT8_C(  43), -INT8_C(   4), -INT8_C(  64),  INT8_C(  83), -INT8_C(  34),  INT8_C( 118),
        -INT8_C( 120), -INT8_C(  13),  INT8_C( 123), -INT8_C(  49),  INT8_C(  30), -INT8_C(  96), -INT8_C( 117),  INT8_C(  91),
         INT8_C(  67), -INT8_C(  84),  INT8_C( 106), -INT8_C(  78),  INT8_C( 106), -INT8_C(  99), -INT8_C(  18), -INT8_C(  91) },
      { -INT8_C(  61),  INT8_C( 123), -INT8_C(  38), -INT8_C(  74),  INT8_C(  27),  INT8_C(  85),  INT8_C(  93), -INT8_C(  61),
        -INT8_C(  95), -INT8_C( 120), -INT8_C(  65),  INT8_C(  98), -INT8_C(  37), -INT8_C(  98), -INT8_C(  40),  INT8_C(  99),
        -INT8_C( 111),  INT8_C(  83),  INT8_C(  50), -INT8_C(  81), -INT8_C(  12), -INT8_C(  67),  INT8_C(  10),  INT8_C(  55),
         INT8_C( 105),  INT8_C( 117), -INT8_C(  22), -INT8_C(  45),  INT8_C(  18), -INT8_C(  40),  INT8_C( 121), -INT8_C(  43) },
      { -INT8_C(  61),  INT8_C(  38), -INT8_C( 115), -INT8_C(  74), -INT8_C(  51), -INT8_C(  96),  INT8_C(  93),  INT8_C(  81),
        -INT8_C(   7), -INT8_C( 120), -INT8_C(  65), -INT8_C(   4), -INT8_C(  64), -INT8_C(  98), -INT8_C(  46),  INT8_C(  99),
         INT8_C(  84), -INT8_C(  13),  INT8_C(  40), -INT8_C(  30), -INT8_C(  12), -INT8_C(  96),  INT8_C(  21),  INT8_C(   4),
         INT8_C(  71),  INT8_C(  42), -INT8_C(  22), -INT8_C(  78),  INT8_C(  18), -INT8_C(  99),  INT8_C(  33),  INT8_C(  15) } },
    { {  INT8_C(  84),  INT8_C(  83), -INT8_C( 116),  INT8_C( 111), -INT8_C(  88), -INT8_C(  23),  INT8_C(  51),  INT8_C(  74),
         INT8_C( 114), -INT8_C(  14), -INT8_C(  84),  INT8_C(  77), -INT8_C( 112), -INT8_C( 124), -INT8_C(  79),  INT8_C(  34),
        -INT8_C(  40), -INT8_C(  29), -INT8_C(  47), -INT8_C(  52), -INT8_C(  95), -INT8_C(  36),  INT8_C(   3),  INT8_C(  10),
         INT8_C(  81), -INT8_C(  19), -INT8_C(  34),  INT8_C(  99), -INT8_C(  58),  INT8_C(  87),  INT8_C(  56),  INT8_C(  26) },
      UINT32_C(1401537706),
      { -INT8_C(  82), -INT8_C(  68), -INT8_C(  99),  INT8_C(  32), -INT8_C(  81),  INT8_C(  73),  INT8_C( 109),  INT8_C(  63),
        -INT8_C(  51),  INT8_C(  30),  INT8_C(  97), -INT8_C(  91),  INT8_C(   2),  INT8_C(  51),  INT8_C( 113), -INT8_C(  93),
         INT8_C(  15),  INT8_C( 117), -INT8_C(  83),  INT8_C(  96),  INT8_C(  98), -INT8_C( 117), -INT8_C(  61),  INT8_C(  40),
        -INT8_C(  30), -INT8_C(   5),  INT8_C(  66), -INT8_C( 115), -INT8_C(  64), -INT8_C(  52), -INT8_C(  32),  INT8_C( 110) },
      { -INT8_C( 120),  INT8_C( 125), -INT8_C( 114),  INT8_C(  55), -INT8_C(  58), -INT8_C(   5),  INT8_C( 119), -INT8_C( 109),
         INT8_C(  26), -INT8_C(  40),  INT8_C(  57),  INT8_C(  28),  INT8_C(  11), -INT8_C(  86), -INT8_C(  65),  INT8_C(  26),
         INT8_C(  31),  INT8_C( 108),  INT8_C( 122), -INT8_C( 126), -INT8_C(   8),  INT8_C(  61), -INT8_C(  86), -INT8_C(  38),
         INT8_C(  57), -INT8_C(  19),  INT8_C( 103), -INT8_C(   7), -INT8_C(  71),  INT8_C(  71),  INT8_C( 103),  INT8_C(  65) },
      {  INT8_C(  84), -INT8_C(  68), -INT8_C( 116),  INT8_C(  32), -INT8_C(  88), -INT8_C(   5),  INT8_C(  51), -INT8_C( 109),
         INT8_C( 114), -INT8_C(  14),  INT8_C(  57),  INT8_C(  77), -INT8_C( 112), -INT8_C( 124), -INT8_C(  65), -INT8_C(  93),
         INT8_C(  15), -INT8_C(  29), -INT8_C(  47), -INT8_C( 126), -INT8_C(  95), -INT8_C(  36),  INT8_C(   3), -INT8_C(  38),
        -INT8_C(  30), -INT8_C(  19), -INT8_C(  34),  INT8_C(  99), -INT8_C(  71),  INT8_C(  87), -INT8_C(  32),  INT8_C(  26) } },
    { { -INT8_C(  60), -INT8_C(  11),  INT8_C( 121), -INT8_C( 118), -INT8_C(  16), -INT8_C(  16),  INT8_C(  30),  INT8_C(  10),
        -INT8_C(  56),  INT8_C(  87),  INT8_C(  38), -INT8_C(  44),  INT8_C(   1), -INT8_C(  27), -INT8_C(  18),  INT8_C(  33),
         INT8_C(  82),  INT8_C( 105), -INT8_C(  93),  INT8_C(  74), -INT8_C(  90),  INT8_C(  77),  INT8_C(  36), -INT8_C(  33),
         INT8_C(  58), -INT8_C( 116), -INT8_C(  40), -INT8_C(  13), -INT8_C(  45),  INT8_C(  63),  INT8_C(  53), -INT8_C( 104) },
      UINT32_C( 623029812),
      { -INT8_C(  98),  INT8_C(  64),  INT8_C(  47),  INT8_C( 102), -INT8_C( 105),  INT8_C(  86),  INT8_C(  58), -INT8_C( 103),
         INT8_C(  59),  INT8_C(  41), -INT8_C(  70), -INT8_C( 115), -INT8_C( 110),  INT8_C(  93), -INT8_C(  41),  INT8_C(  56),
        -INT8_C(  86), -INT8_C(   4),  INT8_C(  24), -INT8_C(  27), -INT8_C( 120), -INT8_C(  16), -INT8_C(  40),  INT8_C(  91),
         INT8_C(  48),  INT8_C(  13), -INT8_C(  13),  INT8_C( 100), -INT8_C(  69),  INT8_C(  22), -INT8_C( 119),  INT8_C(  89) },
      {  INT8_C(  86), -INT8_C(  71), -INT8_C(  64), -INT8_C(  18),  INT8_C(  15), -INT8_C(   6), -INT8_C( 121),  INT8_C(  74),
         INT8_C(  35),  INT8_C(  65), -INT8_C(  40), -INT8_C(  75), -INT8_C(  98), -INT8_C(  81), -INT8_C(  18),  INT8_C(  72),
        -INT8_C(  85),  INT8_C(   6),  INT8_C(  45),  INT8_C(  51), -INT8_C(  10),  INT8_C(   6), -INT8_C( 113),  INT8_C(  38),
         INT8_C(  19), -INT8_C( 126), -INT8_C( 117), -INT8_C(  49), -INT8_C( 104),  INT8_C(  20),  INT8_C(  40), -INT8_C(  17) },
      { -INT8_C(  60), -INT8_C(  11), -INT8_C(  64), -INT8_C( 118), -INT8_C( 105), -INT8_C(   6),  INT8_C(  30),  INT8_C(  10),
        -INT8_C(  56),  INT8_C(  41), -INT8_C(  70), -INT8_C( 115),  INT8_C(   1), -INT8_C(  81), -INT8_C(  18),  INT8_C(  56),
         INT8_C(  82), -INT8_C(   4), -INT8_C(  93),  INT8_C(  74), -INT8_C(  90), -INT8_C(  16),  INT8_C(  36), -INT8_C(  33),
         INT8_C(  19), -INT8_C( 116), -INT8_C( 117), -INT8_C(  13), -INT8_C(  45),  INT8_C(  20),  INT8_C(  53), -INT8_C( 104) } },
    { { -INT8_C(  51), -INT8_C(  24), -INT8_C(  35), -INT8_C(  36), -INT8_C(  29),  INT8_C( 100),  INT8_C(  39),  INT8_C(   6),
        -INT8_C(  91), -INT8_C(   1), -INT8_C(  68),  INT8_C(  67), -INT8_C(  82), -INT8_C(  86), -INT8_C( 117),  INT8_C(  90),
        -INT8_C(  80), -INT8_C(  71), -INT8_C( 115), -INT8_C(  90), -INT8_C(  65),  INT8_C(  28), -INT8_C(  51), -INT8_C(  46),
        -INT8_C(  97),  INT8_C(  88), -INT8_C(  95),  INT8_C(  55),  INT8_C( 108), -INT8_C(  54),  INT8_C(  38),  INT8_C(  58) },
      UINT32_C(2501247922),
      {  INT8_C( 103),  INT8_C(  61), -INT8_C( 100),  INT8_C(  12),  INT8_C(  60),  INT8_C(  88),  INT8_C(  79), -INT8_C(  21),
         INT8_C(   2), -INT8_C(  37),  INT8_C(  69), -INT8_C(  78), -INT8_C( 108), -INT8_C(  46),  INT8_C(  88),  INT8_C(  83),
        -INT8_C(  17),  INT8_C(  37),  INT8_C(  37), -INT8_C( 114),  INT8_C( 125), -INT8_C(  57), -INT8_C(  59), -INT8_C(  22),
        -INT8_C( 111), -INT8_C(  20),  INT8_C(  36),  INT8_C(  67), -INT8_C(  17),  INT8_C(  58), -INT8_C(  39),  INT8_C(  87) },
      {  INT8_C( 120),  INT8_C( 117),  INT8_C(  99), -INT8_C(  76), -INT8_C(  51), -INT8_C(  77), -INT8_C(  97), -INT8_C(  49),
        -INT8_C( 114), -INT8_C(  28), -INT8_C( 127),  INT8_C(  34), -INT8_C(  73), -INT8_C(  39),  INT8_C( 117), -INT8_C(  90),
        -INT8_C(   1), -INT8_C( 102),  INT8_C(  52),  INT8_C( 124),  INT8_C(  97), -INT8_C(   7),  INT8_C( 102), -INT8_C(  14),
        -INT8_C(  27), -INT8_C( 118),  INT8_C(  54), -INT8_C(  43), -INT8_C(  59),  INT8_C(  15),  INT8_C(  44),  INT8_C(  61) },
      { -INT8_C(  51),  INT8_C(  61), -INT8_C(  35), -INT8_C(  36), -INT8_C(  51), -INT8_C(  77),  INT8_C(  39), -INT8_C(  49),
        -INT8_C( 114), -INT8_C(  37), -INT8_C(  68),  INT8_C(  67), -INT8_C(  82), -INT8_C(  86), -INT8_C( 117),  INT8_C(  90),
        -INT8_C(  80), -INT8_C( 102),  INT8_C(  37), -INT8_C(  90),  INT8_C(  97),  INT8_C(  28), -INT8_C(  51), -INT8_C(  46),
        -INT8_C( 111),  INT8_C(  88),  INT8_C(  36),  INT8_C(  55), -INT8_C(  59), -INT8_C(  54),  INT8_C(  38),  INT8_C(  61) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi8(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_epi8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_epi8");
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
    easysimd__m256i r = easysimd_mm256_mask_min_epi8(src, k, a, b);

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
test_easysimd_mm256_maskz_min_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t k;
    const int8_t a[32];
    const int8_t b[32];
    const int8_t r[32];
  } test_vec[] = {
    { UINT32_C(4202717729),
      {  INT8_C(   6),  INT8_C(  99),  INT8_C( 109),  INT8_C(  97),  INT8_C( 100), -INT8_C(  38), -INT8_C(  28), -INT8_C( 120),
        -INT8_C( 127), -INT8_C(  57),  INT8_C(  24),  INT8_C(  95), -INT8_C(  68), -INT8_C(  34),  INT8_C( 118), -INT8_C(  31),
         INT8_C(  16), -INT8_C(  82), -INT8_C( 109),  INT8_C(  57), -INT8_C(  96), -INT8_C(  38), -INT8_C(  71), -INT8_C(  34),
         INT8_C(  16),  INT8_C(  98), -INT8_C(  99),  INT8_C(  50), -INT8_C(  60),  INT8_C(  29),  INT8_C(  44), -INT8_C(  54) },
      { -INT8_C( 127), -INT8_C( 103),  INT8_C(  44), -INT8_C(  27),  INT8_C( 116),  INT8_C(  16),  INT8_C( 110), -INT8_C(  11),
        -INT8_C(  40), -INT8_C( 122),  INT8_C(  84), -INT8_C( 108),  INT8_C( 101), -INT8_C(  54),  INT8_C( 117),  INT8_C( 117),
         INT8_C( 121),  INT8_C(   9), -INT8_C(  82),  INT8_C(  25), -INT8_C(  29),  INT8_C( 103), -INT8_C(   8), -INT8_C(  13),
        -INT8_C(  55), -INT8_C( 107),  INT8_C(  37), -INT8_C( 114), -INT8_C(  78),  INT8_C(  82),  INT8_C(  88),  INT8_C(  51) },
      { -INT8_C( 127),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  38),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C( 122),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  54),  INT8_C( 117),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  34),
         INT8_C(   0), -INT8_C( 107),  INT8_C(   0), -INT8_C( 114), -INT8_C(  78),  INT8_C(  29),  INT8_C(  44), -INT8_C(  54) } },
    { UINT32_C(1595507947),
      { -INT8_C( 107), -INT8_C( 121),  INT8_C(  84),  INT8_C( 109),  INT8_C(  13), -INT8_C(  87),  INT8_C(   1),  INT8_C( 114),
         INT8_C( 115),  INT8_C( 118), -INT8_C(  25), -INT8_C(  20),      INT8_MAX, -INT8_C( 107),  INT8_C(   6),  INT8_C(  98),
        -INT8_C(   3), -INT8_C(   2),  INT8_C(  86), -INT8_C(  58), -INT8_C( 109),  INT8_C( 123),  INT8_C(  84),  INT8_C(  69),
        -INT8_C(  51), -INT8_C(  83),  INT8_C( 121), -INT8_C(  71),  INT8_C(  49), -INT8_C( 110),  INT8_C(  24), -INT8_C(  58) },
      {  INT8_C(  25),  INT8_C( 109),  INT8_C(  51),  INT8_C(  38),  INT8_C(  22),  INT8_C(  52), -INT8_C( 103), -INT8_C( 119),
        -INT8_C(  85),      INT8_MIN,  INT8_C( 118),  INT8_C(  42),  INT8_C(  22),  INT8_C( 124), -INT8_C( 115),  INT8_C(  19),
         INT8_C( 122), -INT8_C(  29), -INT8_C(  39),  INT8_C(  13),  INT8_C(  94),  INT8_C(  46),  INT8_C(  82),  INT8_C(  44),
        -INT8_C(  37), -INT8_C(  53), -INT8_C(  27),  INT8_C(  12),  INT8_C(  93), -INT8_C(   3), -INT8_C(  45),  INT8_C( 118) },
      { -INT8_C( 107), -INT8_C( 121),  INT8_C(   0),  INT8_C(  38),  INT8_C(   0), -INT8_C(  87), -INT8_C( 103), -INT8_C( 119),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  25),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  19),
        -INT8_C(   3),  INT8_C(   0),  INT8_C(   0), -INT8_C(  58), -INT8_C( 109),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  51), -INT8_C(  83), -INT8_C(  27), -INT8_C(  71),  INT8_C(  49),  INT8_C(   0), -INT8_C(  45),  INT8_C(   0) } },
    { UINT32_C(2157774442),
      {  INT8_C(  59),  INT8_C(  54),  INT8_C(  10), -INT8_C(  26), -INT8_C(  74),      INT8_MIN,  INT8_C(  16), -INT8_C(  52),
        -INT8_C(   4), -INT8_C(  99), -INT8_C(  33),  INT8_C( 118),      INT8_MIN, -INT8_C(  71), -INT8_C( 125), -INT8_C(  33),
        -INT8_C(  25), -INT8_C(  43),  INT8_C(  11), -INT8_C(  62), -INT8_C(  95), -INT8_C(  16), -INT8_C(  50), -INT8_C(   2),
        -INT8_C(  19), -INT8_C(  95),  INT8_C( 117),  INT8_C(  88), -INT8_C(  88),  INT8_C(  18), -INT8_C(  40), -INT8_C(  29) },
      {  INT8_C(  72), -INT8_C(  30), -INT8_C(  55), -INT8_C(   2),  INT8_C(  98), -INT8_C(  39), -INT8_C(  53),  INT8_C(  94),
         INT8_C( 119), -INT8_C(  86), -INT8_C(  44), -INT8_C(   9),  INT8_C(  99),  INT8_C(  87), -INT8_C(  42),  INT8_C(  74),
         INT8_C(  45), -INT8_C(  31),  INT8_C(  12), -INT8_C(  50), -INT8_C(  47), -INT8_C(  37), -INT8_C(  52), -INT8_C(  65),
         INT8_C( 124),  INT8_C(  65),  INT8_C(  23),  INT8_C(  36),  INT8_C(  83), -INT8_C(  17),  INT8_C(   7), -INT8_C( 101) },
      {  INT8_C(   0), -INT8_C(  30),  INT8_C(   0), -INT8_C(  26),  INT8_C(   0),      INT8_MIN, -INT8_C(  53),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  99), -INT8_C(  44),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  25),  INT8_C(   0),  INT8_C(  11), -INT8_C(  62), -INT8_C(  95),  INT8_C(   0),  INT8_C(   0), -INT8_C(  65),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 101) } },
    { UINT32_C( 882561234),
      { -INT8_C(  86),  INT8_C( 101), -INT8_C( 109),  INT8_C(  33),  INT8_C(  15),  INT8_C( 103),  INT8_C(  24),  INT8_C( 115),
        -INT8_C(  65), -INT8_C(  17), -INT8_C(  67), -INT8_C(  20), -INT8_C(  48), -INT8_C(  54), -INT8_C(  70), -INT8_C(  94),
        -INT8_C(  91), -INT8_C( 122),  INT8_C(  97),  INT8_C(  33), -INT8_C(  56),  INT8_C( 120),  INT8_C(  70),  INT8_C(  27),
         INT8_C( 103),  INT8_C(  77), -INT8_C(  73),  INT8_C(  57),  INT8_C(  30),  INT8_C(  81),  INT8_C( 110), -INT8_C(  56) },
      { -INT8_C(  74),  INT8_C(   1), -INT8_C(  23), -INT8_C(  59),  INT8_C( 104),  INT8_C(   1),  INT8_C(  56),  INT8_C(  39),
        -INT8_C(  16), -INT8_C(  10),  INT8_C(  19), -INT8_C(  63), -INT8_C(  64), -INT8_C(  51),  INT8_C(  99),  INT8_C( 101),
         INT8_C(  84), -INT8_C(  60), -INT8_C( 122),  INT8_C(  28),  INT8_C(  60), -INT8_C(  52),  INT8_C(  55), -INT8_C(  93),
         INT8_C(  26), -INT8_C(  18), -INT8_C(  35),  INT8_C(  56),  INT8_C(  63),  INT8_C(  75),  INT8_C(   0), -INT8_C(  11) },
      {  INT8_C(   0),  INT8_C(   1),  INT8_C(   0),  INT8_C(   0),  INT8_C(  15),  INT8_C(   0),  INT8_C(  24),  INT8_C(  39),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  64),  INT8_C(   0), -INT8_C(  70), -INT8_C(  94),
         INT8_C(   0), -INT8_C( 122),  INT8_C(   0),  INT8_C(  28), -INT8_C(  56),  INT8_C(   0),  INT8_C(   0), -INT8_C(  93),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  73),  INT8_C(   0),  INT8_C(  30),  INT8_C(  75),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C(3032213836),
      { -INT8_C(  22), -INT8_C(  13), -INT8_C(  36), -INT8_C(  37), -INT8_C(  23), -INT8_C(  17), -INT8_C( 100), -INT8_C(  87),
        -INT8_C(  67), -INT8_C(   1),  INT8_C(  14),  INT8_C(  17), -INT8_C(  61), -INT8_C( 107),  INT8_C(  45), -INT8_C(   1),
         INT8_C(  97),  INT8_C( 100), -INT8_C(  94),  INT8_C( 123),  INT8_C(  83),      INT8_MAX, -INT8_C(  77), -INT8_C( 110),
        -INT8_C(  54), -INT8_C(  77), -INT8_C( 120),  INT8_C(  22), -INT8_C( 100),  INT8_C(  67), -INT8_C(  53), -INT8_C( 121) },
      {  INT8_C(  54), -INT8_C(  89),  INT8_C(  98),  INT8_C(  32), -INT8_C( 106), -INT8_C(   2), -INT8_C(  55),  INT8_C(  83),
        -INT8_C(   3), -INT8_C(  40),  INT8_C( 100), -INT8_C(  64),  INT8_C( 109), -INT8_C( 111), -INT8_C(  65), -INT8_C(  50),
        -INT8_C(  10),  INT8_C(  97),  INT8_C(  74),  INT8_C(  73), -INT8_C(  31), -INT8_C(   3), -INT8_C(  37), -INT8_C(  85),
        -INT8_C(  79),  INT8_C(  99), -INT8_C(  62),  INT8_C(  77), -INT8_C(  90), -INT8_C( 115), -INT8_C(  44), -INT8_C(  35) },
      {  INT8_C(   0),  INT8_C(   0), -INT8_C(  36), -INT8_C(  37),  INT8_C(   0),  INT8_C(   0), -INT8_C( 100),  INT8_C(   0),
        -INT8_C(  67),  INT8_C(   0),  INT8_C(   0), -INT8_C(  64),  INT8_C(   0), -INT8_C( 111), -INT8_C(  65), -INT8_C(  50),
        -INT8_C(  10),  INT8_C(  97),  INT8_C(   0),  INT8_C(  73), -INT8_C(  31), -INT8_C(   3),  INT8_C(   0), -INT8_C( 110),
         INT8_C(   0),  INT8_C(   0), -INT8_C( 120),  INT8_C(   0), -INT8_C( 100), -INT8_C( 115),  INT8_C(   0), -INT8_C( 121) } },
    { UINT32_C(3405592116),
      {  INT8_C(  52), -INT8_C(  58),  INT8_C(  30),  INT8_C(  49), -INT8_C(  98), -INT8_C( 126), -INT8_C(  15),  INT8_C(  11),
         INT8_C(  20), -INT8_C(  80), -INT8_C(  38),  INT8_C(  10),  INT8_C(  18),  INT8_C(  36),  INT8_C(  83), -INT8_C(  13),
         INT8_C(  33),  INT8_C(  46), -INT8_C(  98), -INT8_C(  46), -INT8_C( 110),  INT8_C(  96),  INT8_C(  32),  INT8_C(  56),
        -INT8_C(  19), -INT8_C(  12),  INT8_C(  21),  INT8_C(  33),  INT8_C(  43),  INT8_C(  18), -INT8_C(  20),  INT8_C(  95) },
      { -INT8_C(  39),  INT8_C(  10), -INT8_C( 111),  INT8_C( 119), -INT8_C( 116), -INT8_C( 126), -INT8_C( 125), -INT8_C(  96),
         INT8_C(  51),  INT8_C(  93), -INT8_C(  86),  INT8_C(  69), -INT8_C( 127), -INT8_C(   3),  INT8_C(  56), -INT8_C(  94),
         INT8_C(  44), -INT8_C(  42),  INT8_C( 117), -INT8_C(  66),  INT8_C(  55), -INT8_C( 107), -INT8_C(  10),  INT8_C(  36),
        -INT8_C( 119),  INT8_C(  12),  INT8_C(  70), -INT8_C(  76),  INT8_C(  30),  INT8_C(  50),  INT8_C(  20), -INT8_C(   9) },
      {  INT8_C(   0),  INT8_C(   0), -INT8_C( 111),  INT8_C(   0), -INT8_C( 116), -INT8_C( 126),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  80), -INT8_C(  86),  INT8_C(   0), -INT8_C( 127), -INT8_C(   3),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  33),  INT8_C(   0), -INT8_C(  98), -INT8_C(  66), -INT8_C( 110), -INT8_C( 107), -INT8_C(  10),  INT8_C(  36),
         INT8_C(   0), -INT8_C(  12),  INT8_C(   0), -INT8_C(  76),  INT8_C(   0),  INT8_C(   0), -INT8_C(  20), -INT8_C(   9) } },
    { UINT32_C(3362759996),
      {  INT8_C(  39), -INT8_C(  14),  INT8_C( 105),  INT8_C(  90),  INT8_C(  79),  INT8_C(  19), -INT8_C(  97), -INT8_C(  48),
         INT8_C(  17), -INT8_C(  41),  INT8_C( 114),  INT8_C(  61), -INT8_C(  82), -INT8_C(  25), -INT8_C(   5), -INT8_C(  27),
         INT8_C( 124), -INT8_C(  15),  INT8_C(   9),  INT8_C(   6), -INT8_C(   3),  INT8_C(  79), -INT8_C(  70),  INT8_C(  28),
        -INT8_C( 127), -INT8_C(  50),  INT8_C(  19), -INT8_C(  67),  INT8_C( 115), -INT8_C( 126), -INT8_C( 122), -INT8_C( 101) },
      {  INT8_C( 116), -INT8_C(  17), -INT8_C(  11), -INT8_C(  61),  INT8_C(   2), -INT8_C( 107), -INT8_C( 109),  INT8_C(  19),
         INT8_C( 108),  INT8_C(   6),  INT8_C(  80),  INT8_C(  26), -INT8_C(  19),  INT8_C(  75), -INT8_C(   1),  INT8_C( 106),
         INT8_C(  61),  INT8_C(   9),  INT8_C( 112),  INT8_C(  58),  INT8_C(  88),  INT8_C(  42),  INT8_C(  86), -INT8_C(  38),
        -INT8_C(   7),  INT8_C( 106), -INT8_C( 105),  INT8_C( 108), -INT8_C(  20),  INT8_C(  29),  INT8_C(   7),  INT8_C(  97) },
      {  INT8_C(   0),  INT8_C(   0), -INT8_C(  11), -INT8_C(  61),  INT8_C(   2), -INT8_C( 107),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  17),  INT8_C(   0),  INT8_C(  80),  INT8_C(   0),  INT8_C(   0), -INT8_C(  25),  INT8_C(   0), -INT8_C(  27),
         INT8_C(  61), -INT8_C(  15),  INT8_C(   9),  INT8_C(   6),  INT8_C(   0),  INT8_C(  42), -INT8_C(  70),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  67),  INT8_C(   0),  INT8_C(   0), -INT8_C( 122), -INT8_C( 101) } },
    { UINT32_C( 254082316),
      { -INT8_C( 110), -INT8_C(  72),  INT8_C(  34), -INT8_C(   2), -INT8_C(  66),  INT8_C( 115),  INT8_C(  25), -INT8_C(  85),
        -INT8_C(  66),  INT8_C(  24),  INT8_C(  21), -INT8_C(   5),  INT8_C(  33), -INT8_C( 123),  INT8_C(  54),  INT8_C( 122),
        -INT8_C(  80), -INT8_C( 116),  INT8_C(  84), -INT8_C(  87), -INT8_C(  10), -INT8_C(  21),  INT8_C(  21), -INT8_C(  29),
         INT8_C(   9),  INT8_C(  29),  INT8_C(  68),  INT8_C(  21),  INT8_C(  26),  INT8_C( 104),  INT8_C(  36), -INT8_C(  84) },
      {  INT8_C(  32),  INT8_C(  71), -INT8_C(  86), -INT8_C(  34), -INT8_C(  70), -INT8_C(  61), -INT8_C( 118),  INT8_C( 120),
        -INT8_C(  36), -INT8_C(  97),  INT8_C( 116), -INT8_C(   3),  INT8_C(  37), -INT8_C(  86),  INT8_C( 119), -INT8_C(  43),
         INT8_C(  54), -INT8_C(  53),  INT8_C( 126),  INT8_C(  45), -INT8_C(  73), -INT8_C( 109),  INT8_C(  16), -INT8_C(  64),
        -INT8_C(  80),  INT8_C(  84), -INT8_C(  43), -INT8_C(  54), -INT8_C(  68), -INT8_C(   6),  INT8_C( 118), -INT8_C(  35) },
      {  INT8_C(   0),  INT8_C(   0), -INT8_C(  86), -INT8_C(  34),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  66),  INT8_C(   0),  INT8_C(  21), -INT8_C(   5),  INT8_C(  33), -INT8_C( 123),  INT8_C(  54), -INT8_C(  43),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  84),  INT8_C(   0),  INT8_C(   0), -INT8_C( 109),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  80),  INT8_C(  29), -INT8_C(  43), -INT8_C(  54),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_epi8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_epi8");
    easysimd_test_x86_assert_equal_i8x32(r, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = easysimd_mm256_maskz_min_epi8(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_min_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[16];
    const uint16_t k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { { -INT16_C( 28796),  INT16_C( 20977), -INT16_C( 28350), -INT16_C( 12256), -INT16_C( 24203),  INT16_C( 11506),  INT16_C( 26490),  INT16_C( 31186),
         INT16_C(  1538),  INT16_C( 25590),  INT16_C( 23552), -INT16_C(  6826), -INT16_C( 29465), -INT16_C( 21318), -INT16_C(  6501),  INT16_C(  8169) },
      UINT16_C(55926),
      { -INT16_C( 18320), -INT16_C( 28565), -INT16_C(  7799),  INT16_C( 31537), -INT16_C( 21747), -INT16_C(  7965), -INT16_C(  6875),  INT16_C(  7142),
        -INT16_C(  6584), -INT16_C( 24969),  INT16_C( 24268), -INT16_C( 31190), -INT16_C( 15094), -INT16_C(  3219), -INT16_C(  7196),  INT16_C( 21710) },
      {  INT16_C( 14747),  INT16_C(  9444),  INT16_C(  5402),  INT16_C( 10400), -INT16_C( 31807), -INT16_C(  6648), -INT16_C(  4504), -INT16_C( 20479),
         INT16_C( 30933), -INT16_C( 24241),  INT16_C( 31191), -INT16_C(  7897), -INT16_C( 27585),  INT16_C(  9173), -INT16_C( 23689),  INT16_C(  4984) },
      { -INT16_C( 28796), -INT16_C( 28565), -INT16_C(  7799), -INT16_C( 12256), -INT16_C( 31807), -INT16_C(  7965), -INT16_C(  6875),  INT16_C( 31186),
         INT16_C(  1538), -INT16_C( 24969),  INT16_C( 23552), -INT16_C( 31190), -INT16_C( 27585), -INT16_C( 21318), -INT16_C( 23689),  INT16_C(  4984) } },
    { {  INT16_C( 23772), -INT16_C(  2249), -INT16_C( 10382),  INT16_C( 13087),  INT16_C( 10074), -INT16_C( 15847),  INT16_C(  6677), -INT16_C(  5517),
        -INT16_C( 15726),  INT16_C( 27019), -INT16_C( 19653),  INT16_C( 31307),  INT16_C(  8263), -INT16_C( 16482),  INT16_C(  5827), -INT16_C( 24622) },
      UINT16_C( 2418),
      { -INT16_C(  7018), -INT16_C( 18975),  INT16_C( 15127),  INT16_C( 12508), -INT16_C(  3330),  INT16_C( 29002), -INT16_C(  8740),  INT16_C( 26675),
         INT16_C( 28230), -INT16_C( 28389),  INT16_C( 25321), -INT16_C( 30799),  INT16_C( 29729), -INT16_C(  3171),  INT16_C(  3860), -INT16_C( 21763) },
      { -INT16_C(  8460),  INT16_C(  2912),  INT16_C( 15385),  INT16_C(  5948), -INT16_C( 31186),  INT16_C(  2952), -INT16_C( 17565), -INT16_C( 21901),
        -INT16_C( 29142),  INT16_C(  4923), -INT16_C(  4624),  INT16_C(  4762),  INT16_C( 14177),  INT16_C( 29957),  INT16_C(   582),  INT16_C( 14880) },
      {  INT16_C( 23772), -INT16_C( 18975), -INT16_C( 10382),  INT16_C( 13087), -INT16_C( 31186),  INT16_C(  2952), -INT16_C( 17565), -INT16_C(  5517),
        -INT16_C( 29142),  INT16_C( 27019), -INT16_C( 19653), -INT16_C( 30799),  INT16_C(  8263), -INT16_C( 16482),  INT16_C(  5827), -INT16_C( 24622) } },
    { { -INT16_C( 32544), -INT16_C(  1466), -INT16_C( 32068), -INT16_C(  5359), -INT16_C( 26104),  INT16_C( 27894),  INT16_C( 26965),  INT16_C( 32534),
         INT16_C( 20983), -INT16_C(  6254),  INT16_C( 11326), -INT16_C( 24327), -INT16_C(   157), -INT16_C( 21995),  INT16_C( 13569), -INT16_C(  7452) },
      UINT16_C(10933),
      {  INT16_C( 29404), -INT16_C(  4692), -INT16_C( 19107),  INT16_C( 21383), -INT16_C(  8927),  INT16_C( 14268), -INT16_C( 19620), -INT16_C(  4216),
        -INT16_C( 14438), -INT16_C( 27621),  INT16_C( 32615),  INT16_C( 31891), -INT16_C( 27607),  INT16_C(  3506),  INT16_C( 26486),  INT16_C( 21048) },
      { -INT16_C(  6951),  INT16_C( 13888), -INT16_C( 14439), -INT16_C( 17783),  INT16_C( 17828),  INT16_C(   497),  INT16_C( 31480), -INT16_C( 27664),
         INT16_C(  2881), -INT16_C( 22489), -INT16_C( 17782), -INT16_C( 19676), -INT16_C( 10674), -INT16_C( 14911), -INT16_C(  1730),  INT16_C(  5911) },
      { -INT16_C(  6951), -INT16_C(  1466), -INT16_C( 19107), -INT16_C(  5359), -INT16_C(  8927),  INT16_C(   497),  INT16_C( 26965), -INT16_C( 27664),
         INT16_C( 20983), -INT16_C( 27621),  INT16_C( 11326), -INT16_C( 19676), -INT16_C(   157), -INT16_C( 14911),  INT16_C( 13569), -INT16_C(  7452) } },
    { {  INT16_C( 22493),  INT16_C( 30542), -INT16_C( 10465), -INT16_C( 15567),  INT16_C(  8989),  INT16_C(  5572), -INT16_C( 19299), -INT16_C(  8536),
        -INT16_C( 12352),  INT16_C( 19078), -INT16_C( 21879), -INT16_C(  9986), -INT16_C( 16511), -INT16_C( 16483), -INT16_C( 19272), -INT16_C( 27178) },
      UINT16_C( 9228),
      {  INT16_C( 11020),  INT16_C( 16124),  INT16_C(  6638), -INT16_C( 19615), -INT16_C(   466), -INT16_C( 10393),  INT16_C( 10204),  INT16_C( 25254),
         INT16_C( 12402),  INT16_C( 28684), -INT16_C( 29432), -INT16_C( 23249), -INT16_C(  6324),  INT16_C(  9049),  INT16_C( 25980), -INT16_C( 30393) },
      {  INT16_C( 17296),  INT16_C( 32711),  INT16_C( 10332), -INT16_C( 29902), -INT16_C( 26330),  INT16_C(   610),  INT16_C(  2241),  INT16_C( 13156),
         INT16_C( 28728),  INT16_C( 16547), -INT16_C( 11522),  INT16_C( 19173),  INT16_C( 16313),  INT16_C( 13677), -INT16_C( 19036),  INT16_C( 13758) },
      {  INT16_C( 22493),  INT16_C( 30542),  INT16_C(  6638), -INT16_C( 29902),  INT16_C(  8989),  INT16_C(  5572), -INT16_C( 19299), -INT16_C(  8536),
        -INT16_C( 12352),  INT16_C( 19078), -INT16_C( 29432), -INT16_C(  9986), -INT16_C( 16511),  INT16_C(  9049), -INT16_C( 19272), -INT16_C( 27178) } },
    { { -INT16_C( 31240),  INT16_C( 21940), -INT16_C(  6483), -INT16_C( 11296),  INT16_C( 17023),  INT16_C( 16597),  INT16_C( 14666), -INT16_C( 31885),
         INT16_C(  5802), -INT16_C( 22333), -INT16_C( 22040), -INT16_C( 24078),  INT16_C( 24808), -INT16_C( 29481), -INT16_C( 27371),  INT16_C(  3521) },
      UINT16_C(29979),
      { -INT16_C( 14238),  INT16_C( 16987), -INT16_C(  9316),  INT16_C( 29060), -INT16_C( 12517), -INT16_C( 28757),  INT16_C( 21842),  INT16_C(  5541),
        -INT16_C( 28931), -INT16_C(  4162), -INT16_C( 22993),  INT16_C(  1615),  INT16_C( 25651), -INT16_C(  2916), -INT16_C( 18574), -INT16_C( 11158) },
      { -INT16_C( 14977),  INT16_C(  6935), -INT16_C( 25696), -INT16_C( 17267),  INT16_C( 14442), -INT16_C( 17333), -INT16_C(  3955), -INT16_C( 29998),
        -INT16_C( 28546), -INT16_C( 20871), -INT16_C( 14025),  INT16_C( 27316),  INT16_C( 20525), -INT16_C( 24738), -INT16_C( 14329), -INT16_C( 30860) },
      { -INT16_C( 14977),  INT16_C(  6935), -INT16_C(  6483), -INT16_C( 17267), -INT16_C( 12517),  INT16_C( 16597),  INT16_C( 14666), -INT16_C( 31885),
        -INT16_C( 28931), -INT16_C( 22333), -INT16_C( 22993), -INT16_C( 24078),  INT16_C( 20525), -INT16_C( 24738), -INT16_C( 18574),  INT16_C(  3521) } },
    { { -INT16_C( 29810),  INT16_C( 11938),  INT16_C( 12070), -INT16_C( 28182),  INT16_C( 13671), -INT16_C(  2995),  INT16_C(  7974), -INT16_C( 23426),
        -INT16_C(  1872), -INT16_C(  6318),  INT16_C(  1985), -INT16_C(  4527), -INT16_C( 20649),  INT16_C( 24462),  INT16_C(   632),  INT16_C(  1766) },
      UINT16_C(34957),
      { -INT16_C( 19660),  INT16_C(  8120),  INT16_C(  8004), -INT16_C( 28076),  INT16_C( 31252), -INT16_C( 27983),  INT16_C( 24863),  INT16_C( 29066),
         INT16_C( 19272), -INT16_C( 26248), -INT16_C( 12230), -INT16_C( 14263), -INT16_C( 16081),  INT16_C(  5578),  INT16_C( 22471), -INT16_C(  1123) },
      {  INT16_C( 21770),  INT16_C( 20250),  INT16_C( 28533), -INT16_C( 30239), -INT16_C( 27927),  INT16_C(  2075), -INT16_C( 22796),  INT16_C( 15482),
        -INT16_C(  3343),  INT16_C( 11222),  INT16_C(  8130), -INT16_C(  3597), -INT16_C( 16928), -INT16_C( 22778), -INT16_C( 23532),  INT16_C(  8098) },
      { -INT16_C( 19660),  INT16_C( 11938),  INT16_C(  8004), -INT16_C( 30239),  INT16_C( 13671), -INT16_C(  2995),  INT16_C(  7974),  INT16_C( 15482),
        -INT16_C(  1872), -INT16_C(  6318),  INT16_C(  1985), -INT16_C( 14263), -INT16_C( 20649),  INT16_C( 24462),  INT16_C(   632), -INT16_C(  1123) } },
    { { -INT16_C( 16903),  INT16_C( 28270),  INT16_C( 20268),  INT16_C(  5623),  INT16_C(  5089), -INT16_C( 10978), -INT16_C( 26439), -INT16_C( 21998),
        -INT16_C(  6006),  INT16_C( 19926), -INT16_C( 14073), -INT16_C(  6338),  INT16_C( 17799), -INT16_C( 25714),  INT16_C( 12521), -INT16_C(  7494) },
      UINT16_C(10477),
      {  INT16_C(  6481),  INT16_C( 18551),  INT16_C( 22831),  INT16_C( 19803),  INT16_C(  5166),  INT16_C( 16613),  INT16_C( 28607), -INT16_C( 27352),
         INT16_C( 12220), -INT16_C(  1186), -INT16_C(  6890), -INT16_C( 23488),  INT16_C( 10625),  INT16_C( 15317), -INT16_C( 15861),  INT16_C( 23652) },
      { -INT16_C(  9252),  INT16_C(  2981),  INT16_C(    52),  INT16_C( 25432),  INT16_C( 15637), -INT16_C( 11101), -INT16_C( 13140),  INT16_C( 26985),
        -INT16_C( 14341),  INT16_C(  4708), -INT16_C( 23379),  INT16_C( 11958), -INT16_C( 29747), -INT16_C( 10135), -INT16_C( 12978),  INT16_C( 10805) },
      { -INT16_C(  9252),  INT16_C( 28270),  INT16_C(    52),  INT16_C( 19803),  INT16_C(  5089), -INT16_C( 11101), -INT16_C( 13140), -INT16_C( 27352),
        -INT16_C(  6006),  INT16_C( 19926), -INT16_C( 14073), -INT16_C( 23488),  INT16_C( 17799), -INT16_C( 10135),  INT16_C( 12521), -INT16_C(  7494) } },
    { { -INT16_C(  9559), -INT16_C(  8907), -INT16_C( 29222), -INT16_C(  4288), -INT16_C(  6966),  INT16_C( 30403),  INT16_C( 11440), -INT16_C( 21537),
         INT16_C( 17396), -INT16_C( 24131),  INT16_C( 29927), -INT16_C( 19249),  INT16_C( 14591),  INT16_C( 19853), -INT16_C( 15866), -INT16_C( 20617) },
      UINT16_C(44188),
      {  INT16_C( 30348), -INT16_C( 12999),  INT16_C(   870),  INT16_C( 10673),  INT16_C( 24954),  INT16_C( 22870),  INT16_C( 18956), -INT16_C( 13667),
        -INT16_C( 31509), -INT16_C( 17858),  INT16_C( 15673), -INT16_C( 14606), -INT16_C(  1909),  INT16_C(   648),  INT16_C(  9383),  INT16_C( 13487) },
      { -INT16_C(  5990),  INT16_C(     1), -INT16_C( 19732),  INT16_C( 26154), -INT16_C( 32749),  INT16_C(  8127),  INT16_C( 23754), -INT16_C( 18967),
         INT16_C( 10209),  INT16_C(  6767),  INT16_C( 24933), -INT16_C(  3872),  INT16_C( 26714),  INT16_C(   498), -INT16_C( 24180),  INT16_C(  9781) },
      { -INT16_C(  9559), -INT16_C(  8907), -INT16_C( 19732),  INT16_C( 10673), -INT16_C( 32749),  INT16_C( 30403),  INT16_C( 11440), -INT16_C( 18967),
         INT16_C( 17396), -INT16_C( 24131),  INT16_C( 15673), -INT16_C( 14606),  INT16_C( 14591),  INT16_C(   498), -INT16_C( 15866),  INT16_C(  9781) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_epi16");
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
    easysimd__m256i r = easysimd_mm256_mask_min_epi16(src, k, a, b);

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
test_easysimd_mm256_maskz_min_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C( 8513),
      { -INT16_C(  1093),  INT16_C( 17892), -INT16_C( 16269), -INT16_C(  6171),  INT16_C(  2750),  INT16_C( 13713), -INT16_C( 14113),  INT16_C( 23809),
        -INT16_C( 18187),  INT16_C(  1520), -INT16_C( 24200),  INT16_C( 19801),  INT16_C(  5483), -INT16_C(  7609), -INT16_C( 30478), -INT16_C( 20989) },
      { -INT16_C(  6269), -INT16_C(  2061), -INT16_C( 10072),  INT16_C( 26334),  INT16_C( 28898), -INT16_C( 15973), -INT16_C( 25544),  INT16_C( 11550),
         INT16_C(  3924), -INT16_C( 13262), -INT16_C( 29776),  INT16_C(  6938),  INT16_C( 24992), -INT16_C( 27651),  INT16_C(   234),  INT16_C( 27969) },
      { -INT16_C(  6269),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 25544),  INT16_C(     0),
        -INT16_C( 18187),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 27651),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(13544),
      { -INT16_C( 28572),  INT16_C( 17165), -INT16_C(  4106), -INT16_C( 28237), -INT16_C(  5199), -INT16_C( 12498), -INT16_C( 32232),  INT16_C( 19166),
        -INT16_C( 29105),  INT16_C( 27093),  INT16_C( 30122), -INT16_C( 22582), -INT16_C( 19448),  INT16_C( 18856), -INT16_C( 28638), -INT16_C( 31106) },
      { -INT16_C( 29920),  INT16_C(  5833),  INT16_C( 31866),  INT16_C( 11175), -INT16_C( 10905),  INT16_C( 32763), -INT16_C(  9896), -INT16_C( 22583),
        -INT16_C( 24984),  INT16_C(  4624), -INT16_C(  9708),  INT16_C(  7353),  INT16_C( 24975), -INT16_C( 20122), -INT16_C(  6927),  INT16_C(  4407) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28237),  INT16_C(     0), -INT16_C( 12498), -INT16_C( 32232), -INT16_C( 22583),
         INT16_C(     0),  INT16_C(     0), -INT16_C(  9708),  INT16_C(     0), -INT16_C( 19448), -INT16_C( 20122),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(  367),
      { -INT16_C(  5849), -INT16_C( 12419), -INT16_C(  6891),  INT16_C(  4260), -INT16_C(   924),  INT16_C( 12009),  INT16_C( 20899), -INT16_C( 19508),
        -INT16_C(  8093),  INT16_C(  7566),  INT16_C(  7677),  INT16_C( 25470),  INT16_C( 28878),  INT16_C(  1351), -INT16_C( 18815), -INT16_C( 22266) },
      { -INT16_C( 31585), -INT16_C( 19336),  INT16_C(  7273), -INT16_C( 12860), -INT16_C( 20967), -INT16_C( 17157), -INT16_C( 14081),  INT16_C( 25456),
        -INT16_C(   344), -INT16_C( 23168), -INT16_C(   485), -INT16_C(  5880),  INT16_C( 20334), -INT16_C(  3858), -INT16_C(  2811), -INT16_C( 23143) },
      { -INT16_C( 31585), -INT16_C( 19336), -INT16_C(  6891), -INT16_C( 12860),  INT16_C(     0), -INT16_C( 17157), -INT16_C( 14081),  INT16_C(     0),
        -INT16_C(  8093),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C( 4473),
      { -INT16_C(  7591),  INT16_C(  7725),  INT16_C( 18095), -INT16_C( 21556), -INT16_C( 13565),  INT16_C( 29555),  INT16_C(  6958), -INT16_C( 20879),
        -INT16_C( 29503), -INT16_C( 13907),  INT16_C(  7029),  INT16_C( 25369),  INT16_C(  7691), -INT16_C( 23464), -INT16_C( 11837),  INT16_C(  7605) },
      { -INT16_C(  7245),  INT16_C( 25403),  INT16_C(  1833),  INT16_C( 11278), -INT16_C( 32302),  INT16_C(   415),  INT16_C(  4252),  INT16_C( 23983),
         INT16_C( 23708),  INT16_C(  4391),  INT16_C( 16504), -INT16_C( 31883), -INT16_C( 12962),  INT16_C(  8744), -INT16_C(  8801),  INT16_C( 21055) },
      { -INT16_C(  7591),  INT16_C(     0),  INT16_C(     0), -INT16_C( 21556), -INT16_C( 32302),  INT16_C(   415),  INT16_C(  4252),  INT16_C(     0),
        -INT16_C( 29503),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 12962),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(31424),
      { -INT16_C(  5451), -INT16_C( 15487),  INT16_C( 21270), -INT16_C( 18876), -INT16_C(  7852),  INT16_C(  1222),  INT16_C( 25406),  INT16_C( 25952),
        -INT16_C( 10124), -INT16_C(  5723),  INT16_C(  1116), -INT16_C( 31561),  INT16_C( 22054),  INT16_C( 25953),  INT16_C(  8872),  INT16_C( 24287) },
      {  INT16_C( 24588),  INT16_C(  8737),  INT16_C( 26291),  INT16_C(  2264), -INT16_C( 24761), -INT16_C( 31476),  INT16_C( 27650),  INT16_C( 30443),
        -INT16_C( 28603), -INT16_C( 24224),  INT16_C(  6036), -INT16_C( 17883), -INT16_C( 31123),  INT16_C(  5407), -INT16_C(   344), -INT16_C( 19341) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 25406),  INT16_C( 25952),
         INT16_C(     0), -INT16_C( 24224),  INT16_C(     0), -INT16_C( 31561), -INT16_C( 31123),  INT16_C(  5407), -INT16_C(   344),  INT16_C(     0) } },
    { UINT16_C(38238),
      {  INT16_C(  4823), -INT16_C( 20485),  INT16_C( 16922),  INT16_C(  9806),  INT16_C( 20679), -INT16_C( 19822), -INT16_C( 10297),  INT16_C( 10051),
        -INT16_C( 10376), -INT16_C( 25282), -INT16_C( 21614), -INT16_C( 20188), -INT16_C( 13120),  INT16_C( 13488),  INT16_C(  3713),  INT16_C( 22729) },
      { -INT16_C( 15328),  INT16_C( 14855),  INT16_C( 22022), -INT16_C( 12960), -INT16_C(  3162),  INT16_C( 28032), -INT16_C( 15414),  INT16_C( 17300),
        -INT16_C( 11622),  INT16_C( 11488),  INT16_C(  1149),  INT16_C( 16094), -INT16_C( 28975),  INT16_C( 21106),  INT16_C( 15260), -INT16_C( 16982) },
      {  INT16_C(     0), -INT16_C( 20485),  INT16_C( 16922), -INT16_C( 12960), -INT16_C(  3162),  INT16_C(     0), -INT16_C( 15414),  INT16_C(     0),
        -INT16_C( 11622),  INT16_C(     0), -INT16_C( 21614),  INT16_C(     0), -INT16_C( 28975),  INT16_C(     0),  INT16_C(     0), -INT16_C( 16982) } },
    { UINT16_C(45567),
      {  INT16_C(  1527),  INT16_C( 22535), -INT16_C( 20782),  INT16_C( 21067),  INT16_C(  5403), -INT16_C( 20459), -INT16_C( 20392),  INT16_C( 14722),
         INT16_C(   220), -INT16_C( 17859),  INT16_C(  3646), -INT16_C( 20408), -INT16_C(  6816),  INT16_C(  2795), -INT16_C(  5470), -INT16_C( 26180) },
      { -INT16_C( 15377), -INT16_C( 15887),  INT16_C( 15473), -INT16_C( 29420),  INT16_C( 10578), -INT16_C( 21955), -INT16_C( 16423), -INT16_C( 18717),
         INT16_C(  8639), -INT16_C(   656), -INT16_C( 18129), -INT16_C( 28499), -INT16_C( 26466),  INT16_C( 16538),  INT16_C( 22146),  INT16_C( 29145) },
      { -INT16_C( 15377), -INT16_C( 15887), -INT16_C( 20782), -INT16_C( 29420),  INT16_C(  5403), -INT16_C( 21955), -INT16_C( 20392), -INT16_C( 18717),
         INT16_C(   220),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 26466),  INT16_C(  2795),  INT16_C(     0), -INT16_C( 26180) } },
    { UINT16_C(51994),
      { -INT16_C( 29901),  INT16_C( 18183),  INT16_C( 22808),  INT16_C( 21872),  INT16_C( 18948), -INT16_C(  6379), -INT16_C( 11264),  INT16_C( 28680),
         INT16_C( 14546),  INT16_C( 32553), -INT16_C( 14392),  INT16_C( 25112), -INT16_C( 26105), -INT16_C(  7751), -INT16_C( 11508),  INT16_C( 16300) },
      { -INT16_C( 19618),  INT16_C( 30598), -INT16_C(  2547),  INT16_C(  4556), -INT16_C(  7872),  INT16_C( 16632),  INT16_C(   438), -INT16_C( 30543),
        -INT16_C(  9671),  INT16_C(   263),  INT16_C(  8098), -INT16_C( 22173),  INT16_C(  7354), -INT16_C( 14710),  INT16_C( 14063),  INT16_C( 19973) },
      {  INT16_C(     0),  INT16_C( 18183),  INT16_C(     0),  INT16_C(  4556), -INT16_C(  7872),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
        -INT16_C(  9671),  INT16_C(   263),  INT16_C(     0), -INT16_C( 22173),  INT16_C(     0),  INT16_C(     0), -INT16_C( 11508),  INT16_C( 16300) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_maskz_min_epi16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_min_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(  1982281354), -INT32_C(    69447192), -INT32_C(  1692689455), -INT32_C(   649067272), -INT32_C(  1846296788),  INT32_C(  2055328544), -INT32_C(   948145349), -INT32_C(  1611812587) },
      UINT8_C(232),
      {  INT32_C(  1708135700), -INT32_C(  1942565647), -INT32_C(   343617049), -INT32_C(   535339743),  INT32_C(   587311184), -INT32_C(  1654752471), -INT32_C(  1447942665),  INT32_C(   663834899) },
      {  INT32_C(  1468883302),  INT32_C(   350470957), -INT32_C(  1241487211),  INT32_C(   379000773), -INT32_C(   382101312),  INT32_C(   193370131), -INT32_C(   793495363), -INT32_C(   268941943) },
      {  INT32_C(  1982281354), -INT32_C(    69447192), -INT32_C(  1692689455), -INT32_C(   535339743), -INT32_C(  1846296788), -INT32_C(  1654752471), -INT32_C(  1447942665), -INT32_C(   268941943) } },
    { { -INT32_C(   733575770), -INT32_C(   571987384),  INT32_C(  1469376658), -INT32_C(  1066587392), -INT32_C(   693524541), -INT32_C(    52350913), -INT32_C(   221407896), -INT32_C(  2115910182) },
      UINT8_C( 74),
      {  INT32_C(  1385321768),  INT32_C(   652505149),  INT32_C(   791034628),  INT32_C(  1358096041), -INT32_C(  1081096049),  INT32_C(  1059621802),  INT32_C(   488249944),  INT32_C(   593992699) },
      {  INT32_C(   762706672),  INT32_C(  1850956138),  INT32_C(  1050507669), -INT32_C(   275869857),  INT32_C(    28187991),  INT32_C(    21092008), -INT32_C(   333554704), -INT32_C(   435190026) },
      { -INT32_C(   733575770),  INT32_C(   652505149),  INT32_C(  1469376658), -INT32_C(   275869857), -INT32_C(   693524541), -INT32_C(    52350913), -INT32_C(   333554704), -INT32_C(  2115910182) } },
    { { -INT32_C(   367819392),  INT32_C(  1935173598),  INT32_C(  1085404640), -INT32_C(   617660540),  INT32_C(    98426204), -INT32_C(  1543102796),  INT32_C(  1871717497),  INT32_C(   710254762) },
      UINT8_C( 37),
      { -INT32_C(   821881752),  INT32_C(  1638954860),  INT32_C(  1759899688), -INT32_C(    54214369), -INT32_C(  1129264738), -INT32_C(   197831217), -INT32_C(  2053200667),  INT32_C(  1638582777) },
      {  INT32_C(  1244770013),  INT32_C(  1303109925), -INT32_C(   256536111), -INT32_C(   252937646),  INT32_C(   330079555), -INT32_C(   687349262),  INT32_C(  2120066437),  INT32_C(  1289684846) },
      { -INT32_C(   821881752),  INT32_C(  1935173598), -INT32_C(   256536111), -INT32_C(   617660540),  INT32_C(    98426204), -INT32_C(   687349262),  INT32_C(  1871717497),  INT32_C(   710254762) } },
    { { -INT32_C(   627699531), -INT32_C(  1037549071),  INT32_C(   615701970), -INT32_C(  1693147305),  INT32_C(   783204412),  INT32_C(   637908385), -INT32_C(   911973798),  INT32_C(   521503850) },
      UINT8_C(148),
      { -INT32_C(   326698325), -INT32_C(     4241374), -INT32_C(  1705516037), -INT32_C(  1193872648),  INT32_C(  1431962784),  INT32_C(  1823440905), -INT32_C(  1445562331),  INT32_C(   943584653) },
      {  INT32_C(   287687663),  INT32_C(   118613004), -INT32_C(  1079940921), -INT32_C(    92768422), -INT32_C(  2075143557),  INT32_C(  2012282450),  INT32_C(    69256823), -INT32_C(  1422041412) },
      { -INT32_C(   627699531), -INT32_C(  1037549071), -INT32_C(  1705516037), -INT32_C(  1693147305), -INT32_C(  2075143557),  INT32_C(   637908385), -INT32_C(   911973798), -INT32_C(  1422041412) } },
    { {  INT32_C(   767386145),  INT32_C(   221564486), -INT32_C(  1865558730), -INT32_C(   947239604),  INT32_C(  1766643991),  INT32_C(  1323384023), -INT32_C(  1085079293), -INT32_C(  2123722657) },
      UINT8_C(242),
      { -INT32_C(   180834777), -INT32_C(  1205123613),  INT32_C(  1459993362),  INT32_C(   510577733),  INT32_C(  1442240536), -INT32_C(  1168620359),  INT32_C(   656021399), -INT32_C(  1457939583) },
      {  INT32_C(   748573001), -INT32_C(  1444558185), -INT32_C(   889066875), -INT32_C(   806785098),  INT32_C(     2416455), -INT32_C(  1145340892),  INT32_C(   350475667), -INT32_C(  1178731408) },
      {  INT32_C(   767386145), -INT32_C(  1444558185), -INT32_C(  1865558730), -INT32_C(   947239604),  INT32_C(     2416455), -INT32_C(  1168620359),  INT32_C(   350475667), -INT32_C(  1457939583) } },
    { { -INT32_C(   437887922), -INT32_C(  1416705242),  INT32_C(  1802932149),  INT32_C(  1178231039),  INT32_C(  1665621567),  INT32_C(  1830748890),  INT32_C(  1199702743),  INT32_C(  1275084798) },
      UINT8_C(155),
      { -INT32_C(  1312738842),  INT32_C(  1332112832),  INT32_C(  1129239267),  INT32_C(  1803785484), -INT32_C(   565844260),  INT32_C(   129348357),  INT32_C(  1946614837), -INT32_C(   485469444) },
      {  INT32_C(  1150603652),  INT32_C(   580123454), -INT32_C(   647634227),  INT32_C(  1397024887), -INT32_C(   734950705),  INT32_C(  1927079485), -INT32_C(   555293982), -INT32_C(  1195248076) },
      { -INT32_C(  1312738842),  INT32_C(   580123454),  INT32_C(  1802932149),  INT32_C(  1397024887), -INT32_C(   734950705),  INT32_C(  1830748890),  INT32_C(  1199702743), -INT32_C(  1195248076) } },
    { {  INT32_C(   117200584),  INT32_C(   505974865), -INT32_C(   369652110),  INT32_C(  1161575542),  INT32_C(    52063686),  INT32_C(   896923219),  INT32_C(   202595288),  INT32_C(   449172818) },
      UINT8_C( 43),
      {  INT32_C(  1367089345), -INT32_C(   675046839), -INT32_C(   850547567),  INT32_C(  1435735016), -INT32_C(  1549232467),  INT32_C(  1736170763), -INT32_C(   960919567),  INT32_C(   250729292) },
      {  INT32_C(  1012887027), -INT32_C(  1743576313), -INT32_C(  1201249841), -INT32_C(  1592919564), -INT32_C(  1689995632), -INT32_C(  2080194669), -INT32_C(  1823818938), -INT32_C(  2103362674) },
      {  INT32_C(  1012887027), -INT32_C(  1743576313), -INT32_C(   369652110), -INT32_C(  1592919564),  INT32_C(    52063686), -INT32_C(  2080194669),  INT32_C(   202595288),  INT32_C(   449172818) } },
    { { -INT32_C(  1346502488), -INT32_C(   213331421),  INT32_C(   665562675),  INT32_C(   935901351),  INT32_C(    47385710),  INT32_C(   294049227),  INT32_C(   530895249), -INT32_C(  1247722228) },
      UINT8_C( 70),
      {  INT32_C(   845767776),  INT32_C(  1516592300), -INT32_C(  1073574905),  INT32_C(  1613642068), -INT32_C(   500486132), -INT32_C(  2005713737),  INT32_C(   647271137), -INT32_C(  1804842700) },
      {  INT32_C(  1522980526),  INT32_C(   984951602),  INT32_C(   200980407), -INT32_C(    43308816),  INT32_C(   283088473), -INT32_C(  1281862958),  INT32_C(   416951524),  INT32_C(   615270006) },
      { -INT32_C(  1346502488),  INT32_C(   984951602), -INT32_C(  1073574905),  INT32_C(   935901351),  INT32_C(    47385710),  INT32_C(   294049227),  INT32_C(   416951524), -INT32_C(  1247722228) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_epi32");
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
    easysimd__m256i r = easysimd_mm256_mask_min_epi32(src, k, a, b);

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
test_easysimd_mm256_maskz_min_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(234),
      { -INT32_C(  2114468469),  INT32_C(  1942096017),  INT32_C(    19464704), -INT32_C(  1908756045), -INT32_C(   667927624), -INT32_C(  1148003937), -INT32_C(  1700046748), -INT32_C(   393938595) },
      {  INT32_C(  1332313022), -INT32_C(  2067649661), -INT32_C(   511317202),  INT32_C(  1433387164), -INT32_C(  1708285957), -INT32_C(   581517447),  INT32_C(  1953956119), -INT32_C(  1201865478) },
      {  INT32_C(           0), -INT32_C(  2067649661),  INT32_C(           0), -INT32_C(  1908756045),  INT32_C(           0), -INT32_C(  1148003937), -INT32_C(  1700046748), -INT32_C(  1201865478) } },
    { UINT8_C(119),
      { -INT32_C(   235206715), -INT32_C(  1256226870), -INT32_C(  1001259004),  INT32_C(   264284016), -INT32_C(  1819780396), -INT32_C(  1314232912), -INT32_C(   643096867),  INT32_C(  1062232954) },
      {  INT32_C(   892357483), -INT32_C(   806727734), -INT32_C(  1064092592), -INT32_C(  1211149341),  INT32_C(  1581930670), -INT32_C(  1710164803), -INT32_C(  1938572526), -INT32_C(  1966291937) },
      { -INT32_C(   235206715), -INT32_C(  1256226870), -INT32_C(  1064092592),  INT32_C(           0), -INT32_C(  1819780396), -INT32_C(  1710164803), -INT32_C(  1938572526),  INT32_C(           0) } },
    { UINT8_C( 15),
      {  INT32_C(  1289404412), -INT32_C(   425940567), -INT32_C(  1865851844), -INT32_C(  2093055701), -INT32_C(  1069441845),  INT32_C(  1758649260),  INT32_C(   327638863),  INT32_C(   656544043) },
      {  INT32_C(  2037644496), -INT32_C(   497086555), -INT32_C(  1754125973),  INT32_C(  1964683434), -INT32_C(   113943732), -INT32_C(  2040461257), -INT32_C(  1835407257), -INT32_C(   910574599) },
      {  INT32_C(  1289404412), -INT32_C(   497086555), -INT32_C(  1865851844), -INT32_C(  2093055701),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(184),
      {  INT32_C(  1012744749), -INT32_C(   878166110),  INT32_C(  1635073969), -INT32_C(  1246827943),  INT32_C(   686597920), -INT32_C(   259034616), -INT32_C(   957800181),  INT32_C(   125743834) },
      { -INT32_C(  1773871884), -INT32_C(   849220581), -INT32_C(  2077305301), -INT32_C(   516301631), -INT32_C(  1962334845), -INT32_C(  1568958313), -INT32_C(  1821809479),  INT32_C(   194766614) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1246827943), -INT32_C(  1962334845), -INT32_C(  1568958313),  INT32_C(           0),  INT32_C(   125743834) } },
    { UINT8_C(195),
      { -INT32_C(   874536481), -INT32_C(   638145533), -INT32_C(  1214612774), -INT32_C(   650478668),  INT32_C(   477218436), -INT32_C(  1495985343),  INT32_C(  1690069372), -INT32_C(   483932412) },
      {  INT32_C(  1823344233), -INT32_C(  1924815694), -INT32_C(   767238114), -INT32_C(   525566373), -INT32_C(  2030297788), -INT32_C(  1389571536),  INT32_C(  1058138171),  INT32_C(   421673136) },
      { -INT32_C(   874536481), -INT32_C(  1924815694),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1058138171), -INT32_C(   483932412) } },
    { UINT8_C( 63),
      {  INT32_C(  1961985488), -INT32_C(  1433239862),  INT32_C(  1090938306),  INT32_C(   780527121), -INT32_C(  1268904990),  INT32_C(   552536887),  INT32_C(  1439706652),  INT32_C(   546629968) },
      {  INT32_C(   949257582), -INT32_C(   958191868), -INT32_C(  1660425844), -INT32_C(  1312059953), -INT32_C(   798611048),  INT32_C(  1374704949), -INT32_C(   727269244),  INT32_C(   401947305) },
      {  INT32_C(   949257582), -INT32_C(  1433239862), -INT32_C(  1660425844), -INT32_C(  1312059953), -INT32_C(  1268904990),  INT32_C(   552536887),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(192),
      { -INT32_C(  1329311863),  INT32_C(   456952370),  INT32_C(   535484818), -INT32_C(   826827612),  INT32_C(  1459848962),  INT32_C(   937120887),  INT32_C(   903917819),  INT32_C(   787871653) },
      {  INT32_C(  2044705095), -INT32_C(   694871228), -INT32_C(  1711964171),  INT32_C(   493333531), -INT32_C(  1418368460), -INT32_C(  1159507777), -INT32_C(  1511013632),  INT32_C(    47506875) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1511013632),  INT32_C(    47506875) } },
    { UINT8_C(158),
      { -INT32_C(   824017997), -INT32_C(  1866221552),  INT32_C(  1521180077),  INT32_C(   797886916), -INT32_C(  1896990402), -INT32_C(   527456228),  INT32_C(  2090546327), -INT32_C(  1155818232) },
      {  INT32_C(   696974616),  INT32_C(  1673088438),  INT32_C(  1874683050),  INT32_C(  1805536045), -INT32_C(  1577415547), -INT32_C(   880703180), -INT32_C(   968352579), -INT32_C(   763272263) },
      {  INT32_C(           0), -INT32_C(  1866221552),  INT32_C(  1521180077),  INT32_C(   797886916), -INT32_C(  1896990402),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1155818232) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_min_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_min_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 4331907611591426848),  INT64_C( 7179741506661127319),  INT64_C( 6204661658632865610), -INT64_C( 7763589255240675073) },
      {  INT64_C( 1885321053595470445), -INT64_C( 5812739310488792043),  INT64_C(  253446244252361287),  INT64_C( 8774674588071550077) },
      { -INT64_C( 4331907611591426848), -INT64_C( 5812739310488792043),  INT64_C(  253446244252361287), -INT64_C( 7763589255240675073) } },
    { { -INT64_C(  356335333448555150), -INT64_C( 5552301262614205474),  INT64_C( 3309131636722975717), -INT64_C( 1385099352185335231) },
      { -INT64_C( 7932264241868789752), -INT64_C(   38062778525362342),  INT64_C(   64973164566305033),  INT64_C( 8127896067192538214) },
      { -INT64_C( 7932264241868789752), -INT64_C( 5552301262614205474),  INT64_C(   64973164566305033), -INT64_C( 1385099352185335231) } },
    { { -INT64_C(  720213529153524290), -INT64_C( 6540856567963790048),  INT64_C( 6795445816205910926),  INT64_C( 8098047943145069705) },
      {  INT64_C( 3047496631871193528), -INT64_C( 3821128102866253896),  INT64_C( 2672999202167478211),  INT64_C( 1884045483976382210) },
      { -INT64_C(  720213529153524290), -INT64_C( 6540856567963790048),  INT64_C( 2672999202167478211),  INT64_C( 1884045483976382210) } },
    { { -INT64_C( 6814305874150029766),  INT64_C( 4019160804173862079),  INT64_C( 5045450248486510148), -INT64_C( 3468579210883556255) },
      { -INT64_C( 4345549215769605337), -INT64_C( 2953934110281783722),  INT64_C( 6768273608322576994),  INT64_C( 3548551773358713049) },
      { -INT64_C( 6814305874150029766), -INT64_C( 2953934110281783722),  INT64_C( 5045450248486510148), -INT64_C( 3468579210883556255) } },
    { {  INT64_C( 2958851569048703304), -INT64_C( 5475264088173720167),  INT64_C( 5710925477940280039), -INT64_C( 6382730560929898192) },
      {  INT64_C( 4056057296893226061),  INT64_C( 1861788960987299536), -INT64_C( 6131863675662946015), -INT64_C( 1178166695326079585) },
      {  INT64_C( 2958851569048703304), -INT64_C( 5475264088173720167), -INT64_C( 6131863675662946015), -INT64_C( 6382730560929898192) } },
    { {  INT64_C( 7841711004845119995), -INT64_C( 4475341492221757288), -INT64_C( 4631600239628131946), -INT64_C( 1696058342117630490) },
      {  INT64_C( 8720471780415997693), -INT64_C( 8219304653317958770),  INT64_C( 7992014156576147466),  INT64_C(  560816947587360500) },
      {  INT64_C( 7841711004845119995), -INT64_C( 8219304653317958770), -INT64_C( 4631600239628131946), -INT64_C( 1696058342117630490) } },
    { {  INT64_C( 3480620205797821427), -INT64_C( 4198113811290484168), -INT64_C( 6589653903805031787), -INT64_C( 2121109733291752833) },
      {  INT64_C(  966870759365816473), -INT64_C( 5974313977134658923),  INT64_C( 6748740652689458515),  INT64_C( 1489414375816441306) },
      {  INT64_C(  966870759365816473), -INT64_C( 5974313977134658923), -INT64_C( 6589653903805031787), -INT64_C( 2121109733291752833) } },
    { { -INT64_C(  393860026666622554),  INT64_C( 1245633143726851204),  INT64_C( 5017500827780453478), -INT64_C( 6666811800378505237) },
      {  INT64_C(  610702312484030746), -INT64_C( 8486209808269786262), -INT64_C(  454178508943503615), -INT64_C( 7667701093947686410) },
      { -INT64_C(  393860026666622554), -INT64_C( 8486209808269786262), -INT64_C(  454178508943503615), -INT64_C( 7667701093947686410) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_min_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_min_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_min_epi64(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_min_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 6091456721731875356),  INT64_C(  349251810589901802), -INT64_C( 1304165886565501088),  INT64_C( 2554581158109561049) },
      UINT8_C( 37),
      {  INT64_C( 9083505340867310321),  INT64_C( 8764337761691560566),  INT64_C( 8919503014486791235),  INT64_C( 8956998346225096566) },
      { -INT64_C( 7092412781258797032),  INT64_C( 3375308194118024347),  INT64_C( 4579007197368637780), -INT64_C( 3745669207373115907) },
      { -INT64_C( 7092412781258797032),  INT64_C(  349251810589901802),  INT64_C( 4579007197368637780),  INT64_C( 2554581158109561049) } },
    { { -INT64_C( 1113952244448844347),  INT64_C( 4415821113440757807),  INT64_C( 5298440204960163198),  INT64_C( 7666487372901642213) },
      UINT8_C(122),
      {  INT64_C( 5325155685096183620), -INT64_C( 5930836878382989298),  INT64_C( 8104300653754055099), -INT64_C( 5088363385082015725) },
      { -INT64_C(   71680522879749673),  INT64_C( 6795057324885160790), -INT64_C( 7291493241555171875),  INT64_C( 4298936966474556537) },
      { -INT64_C( 1113952244448844347), -INT64_C( 5930836878382989298),  INT64_C( 5298440204960163198), -INT64_C( 5088363385082015725) } },
    { { -INT64_C( 4249439341153937387), -INT64_C(  420611448169547480), -INT64_C(  281993381472036107), -INT64_C( 7949817079519274996) },
      UINT8_C(171),
      { -INT64_C( 2828858014912833376), -INT64_C( 3753205266838615996),  INT64_C( 1404361652346709320),  INT64_C( 4212027400676961846) },
      { -INT64_C( 5534768432667455559),  INT64_C( 6742610011050419477),  INT64_C( 7098598992397591029), -INT64_C( 1509375259459251656) },
      { -INT64_C( 5534768432667455559), -INT64_C( 3753205266838615996), -INT64_C(  281993381472036107), -INT64_C( 1509375259459251656) } },
    { {  INT64_C( 9173959342695898414),  INT64_C( 8938195587221092248),  INT64_C( 6505571857430534074),  INT64_C( 4728563085280637825) },
      UINT8_C( 24),
      {  INT64_C( 2805462395701616355),  INT64_C( 9068889167658654792),  INT64_C( 8937668379337015575),  INT64_C( 4502138768912727079) },
      {  INT64_C( 5045022237759241390), -INT64_C(  917816383064526560),  INT64_C(  326653759381431043), -INT64_C( 7721454655571028419) },
      {  INT64_C( 9173959342695898414),  INT64_C( 8938195587221092248),  INT64_C( 6505571857430534074), -INT64_C( 7721454655571028419) } },
    { {  INT64_C( 1162620231654403294), -INT64_C( 3533888873373988570),  INT64_C( 3305187963139944189),  INT64_C(  428578806484583800) },
      UINT8_C( 10),
      { -INT64_C( 8287174153161940637),  INT64_C( 7203830817386594530), -INT64_C(  587295677553060865), -INT64_C( 8146973058773665721) },
      { -INT64_C( 2840815932091733244),  INT64_C( 1061949077967923812), -INT64_C( 2861676299226771296), -INT64_C( 8198329736041175168) },
      {  INT64_C( 1162620231654403294),  INT64_C( 1061949077967923812),  INT64_C( 3305187963139944189), -INT64_C( 8198329736041175168) } },
    { {  INT64_C( 7451249579243446169), -INT64_C( 6183587380398638062),  INT64_C( 7491251884825288332), -INT64_C(  305035989568277332) },
      UINT8_C(234),
      {  INT64_C( 3428177141897726553), -INT64_C( 9142901101975456748), -INT64_C( 1958275082259776800),  INT64_C( 2744453372395784417) },
      {  INT64_C( 7776937379585590220), -INT64_C( 1561315261028817168),  INT64_C( 7883799141322571091), -INT64_C( 4837516976318730518) },
      {  INT64_C( 7451249579243446169), -INT64_C( 9142901101975456748),  INT64_C( 7491251884825288332), -INT64_C( 4837516976318730518) } },
    { { -INT64_C( 5659783653400869454), -INT64_C( 2629300051733352562),  INT64_C( 6741825364997083798), -INT64_C( 8565298009354262411) },
      UINT8_C(119),
      {  INT64_C( 8544147655108367003),  INT64_C( 4602355207479634542), -INT64_C( 5155758933474672585), -INT64_C( 7742222151787161564) },
      {  INT64_C( 2173033758605231788),  INT64_C( 3981256833114646100),  INT64_C( 2849837398617355076), -INT64_C( 2157844112106185600) },
      {  INT64_C( 2173033758605231788),  INT64_C( 3981256833114646100), -INT64_C( 5155758933474672585), -INT64_C( 8565298009354262411) } },
    { {  INT64_C( 2621303681002546578),  INT64_C( 5555865282966570723),  INT64_C( 8611919289888362329),  INT64_C( 2369735135998296522) },
      UINT8_C(163),
      { -INT64_C( 8568214000906378633),  INT64_C( 7168569608974301884),  INT64_C( 2052182767639910145), -INT64_C( 3142728185466488127) },
      {  INT64_C( 2844057713116763180), -INT64_C( 1399840024080872934),  INT64_C( 9156123427329535131), -INT64_C( 5710106359752639600) },
      { -INT64_C( 8568214000906378633), -INT64_C( 1399840024080872934),  INT64_C( 8611919289888362329),  INT64_C( 2369735135998296522) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_epi64");
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
    easysimd__m256i r = easysimd_mm256_mask_min_epi64(src, k, a, b);

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
test_easysimd_mm256_maskz_min_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C( 96),
      {  INT64_C( 1730361596847782667), -INT64_C( 7203312919499083209),  INT64_C( 5184377830440479147),  INT64_C( 5019212552685598948) },
      {  INT64_C( 8041917348705124046),  INT64_C( 4450113412410826772),  INT64_C( 1231281696049107364), -INT64_C( 5144798617863186660) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(190),
      { -INT64_C( 5337216539211449800),  INT64_C( 7347206984766526040), -INT64_C(  903071933928389129), -INT64_C( 6937066902642569988) },
      {  INT64_C( 7952062914032677073),  INT64_C( 2284089344761308026), -INT64_C( 1923896956279962244), -INT64_C( 7943051493708256579) },
      {  INT64_C(                   0),  INT64_C( 2284089344761308026), -INT64_C( 1923896956279962244), -INT64_C( 7943051493708256579) } },
    { UINT8_C( 77),
      {  INT64_C( 3900066234391279421),  INT64_C( 7261347547516398737), -INT64_C( 7807778948526563849), -INT64_C( 3747305438902433260) },
      { -INT64_C( 4330616543798861539), -INT64_C( 5152168821375875844),  INT64_C( 1435317864958577902),  INT64_C( 4785569545874325263) },
      { -INT64_C( 4330616543798861539),  INT64_C(                   0), -INT64_C( 7807778948526563849), -INT64_C( 3747305438902433260) } },
    { UINT8_C( 35),
      {  INT64_C( 2046068923469001433), -INT64_C( 3754711262746017756),  INT64_C( 7461017794814491701),  INT64_C( 7675768259446843296) },
      {  INT64_C( 1708052777572489920),  INT64_C( 6594097953005998520), -INT64_C( 3982484818652476627), -INT64_C( 3061972667005712505) },
      {  INT64_C( 1708052777572489920), -INT64_C( 3754711262746017756),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(139),
      { -INT64_C( 5601188065218499244), -INT64_C( 3909067862113191006),  INT64_C( 1796859081109083401), -INT64_C( 2513004706919544860) },
      { -INT64_C( 3007087114653688208),  INT64_C( 6480726387533094313),  INT64_C( 9162487242156073279), -INT64_C( 4026723463169026632) },
      { -INT64_C( 5601188065218499244), -INT64_C( 3909067862113191006),  INT64_C(                   0), -INT64_C( 4026723463169026632) } },
    { UINT8_C(210),
      { -INT64_C( 4836905997936797049), -INT64_C( 2905271500996344251),  INT64_C( 4220700670785862952),  INT64_C( 8649826026401223428) },
      {  INT64_C( 3463424155325436085), -INT64_C( 6169846266447586157), -INT64_C( 4167493478663270133), -INT64_C( 3795961542151783191) },
      {  INT64_C(                   0), -INT64_C( 6169846266447586157),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 25),
      {  INT64_C( 5485442824482922240), -INT64_C( 2653331525305718315),  INT64_C(  287213633167760893),  INT64_C( 2992049489232713314) },
      { -INT64_C( 7673896544521254395), -INT64_C( 4222716426106482816), -INT64_C( 4013318161404788222),  INT64_C( 5979576674154735530) },
      { -INT64_C( 7673896544521254395),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 2992049489232713314) } },
    { UINT8_C( 68),
      {  INT64_C( 6199654619694794418), -INT64_C(  416332310349180080),  INT64_C( 3979256617687351995), -INT64_C( 5090802577851544740) },
      {  INT64_C( 5168245568983686492),  INT64_C( 8490046962119006175), -INT64_C( 5315024528370027978), -INT64_C(  934174115876847287) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 5315024528370027978),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_min_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_min_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t src[32];
    const uint32_t k;
    const uint8_t a[32];
    const uint8_t b[32];
    const uint8_t r[32];
  } test_vec[] = {
    { { UINT8_C(117), UINT8_C(201), UINT8_C(159), UINT8_C(224), UINT8_C(232), UINT8_C( 23), UINT8_C(  7), UINT8_C(  2),
        UINT8_C( 34), UINT8_C( 80), UINT8_C( 49), UINT8_C( 12), UINT8_C( 20), UINT8_C(195), UINT8_C(248), UINT8_C(175),
        UINT8_C(175), UINT8_C(148), UINT8_C(164), UINT8_C(109), UINT8_C(169), UINT8_C(182), UINT8_C(237), UINT8_C( 57),
        UINT8_C(113), UINT8_C( 43), UINT8_C( 38), UINT8_C(245), UINT8_C(204), UINT8_C(231), UINT8_C(166), UINT8_C( 65) },
      UINT32_C(2569094576),
      { UINT8_C( 93), UINT8_C( 40), UINT8_C(155), UINT8_C(127), UINT8_C(120), UINT8_C(204), UINT8_C(139), UINT8_C(140),
        UINT8_C(143), UINT8_C(132), UINT8_C( 59), UINT8_C( 63), UINT8_C( 24), UINT8_C(223), UINT8_C(172), UINT8_C(193),
        UINT8_C(149), UINT8_C(153), UINT8_C(251), UINT8_C(  6), UINT8_C(197), UINT8_C( 33), UINT8_C(252), UINT8_C(145),
        UINT8_C(  9), UINT8_C(162), UINT8_C(210), UINT8_C(185), UINT8_C(231), UINT8_C(243), UINT8_C( 82), UINT8_C( 68) },
      { UINT8_C( 27), UINT8_C(238), UINT8_C(195), UINT8_C(147), UINT8_C(186), UINT8_C( 79), UINT8_C( 31), UINT8_C( 74),
        UINT8_C(211), UINT8_C( 90), UINT8_C(137), UINT8_C(235), UINT8_C( 57), UINT8_C( 53), UINT8_C(172), UINT8_C(207),
        UINT8_C(207), UINT8_C(167), UINT8_C(213), UINT8_C(148), UINT8_C(201), UINT8_C(209), UINT8_C( 37), UINT8_C(210),
        UINT8_C(115), UINT8_C(247), UINT8_C(139), UINT8_C( 91), UINT8_C(234), UINT8_C(222), UINT8_C(159), UINT8_C(  5) },
      { UINT8_C(117), UINT8_C(201), UINT8_C(159), UINT8_C(224), UINT8_C(120), UINT8_C( 79), UINT8_C(  7), UINT8_C( 74),
        UINT8_C(143), UINT8_C( 80), UINT8_C( 59), UINT8_C( 12), UINT8_C( 20), UINT8_C(195), UINT8_C(172), UINT8_C(175),
        UINT8_C(149), UINT8_C(148), UINT8_C(164), UINT8_C(109), UINT8_C(169), UINT8_C( 33), UINT8_C(237), UINT8_C( 57),
        UINT8_C(  9), UINT8_C( 43), UINT8_C( 38), UINT8_C( 91), UINT8_C(231), UINT8_C(231), UINT8_C(166), UINT8_C(  5) } },
    { { UINT8_C(204), UINT8_C( 99), UINT8_C(152), UINT8_C(134), UINT8_C(178), UINT8_C(183), UINT8_C(208), UINT8_C(133),
        UINT8_C( 17), UINT8_C( 89), UINT8_C(112), UINT8_C( 74), UINT8_C(143), UINT8_C( 28), UINT8_C( 25), UINT8_C( 94),
        UINT8_C(196), UINT8_C(239), UINT8_C(242), UINT8_C(141), UINT8_C(192), UINT8_C( 23), UINT8_C( 95), UINT8_C( 52),
        UINT8_C( 14), UINT8_C(234), UINT8_C(143), UINT8_C(248), UINT8_C(200), UINT8_C( 46), UINT8_C(253), UINT8_C(148) },
      UINT32_C(1125881233),
      { UINT8_C( 76), UINT8_C(235), UINT8_C(200), UINT8_C( 93), UINT8_C( 69), UINT8_C( 56), UINT8_C(167), UINT8_C(212),
        UINT8_C( 85), UINT8_C(193), UINT8_C( 50), UINT8_C( 25), UINT8_C(176), UINT8_C( 36), UINT8_C(166), UINT8_C(112),
        UINT8_C( 59), UINT8_C(  5), UINT8_C(164), UINT8_C( 73), UINT8_C(239), UINT8_C( 51), UINT8_C( 65), UINT8_C(184),
        UINT8_C( 98), UINT8_C( 62), UINT8_C( 76), UINT8_C(243), UINT8_C(211), UINT8_C(103), UINT8_C( 55), UINT8_C( 31) },
      { UINT8_C( 83),    UINT8_MAX, UINT8_C(124), UINT8_C(152), UINT8_C( 56), UINT8_C( 35), UINT8_C(108), UINT8_C(141),
        UINT8_C(228), UINT8_C(158), UINT8_C(166), UINT8_C(148), UINT8_C(194), UINT8_C( 76), UINT8_C(  5), UINT8_C(253),
        UINT8_C( 81), UINT8_C(169), UINT8_C( 70), UINT8_C( 64), UINT8_C(221), UINT8_C(135), UINT8_C(248), UINT8_C( 63),
        UINT8_C(197), UINT8_C( 69), UINT8_C( 50), UINT8_C(152), UINT8_C(172), UINT8_C(105), UINT8_C(183),    UINT8_MAX },
      { UINT8_C( 76), UINT8_C( 99), UINT8_C(152), UINT8_C(134), UINT8_C( 56), UINT8_C(183), UINT8_C(208), UINT8_C(141),
        UINT8_C( 85), UINT8_C( 89), UINT8_C( 50), UINT8_C( 74), UINT8_C(176), UINT8_C( 28), UINT8_C( 25), UINT8_C(112),
        UINT8_C( 59), UINT8_C(  5), UINT8_C(242), UINT8_C( 64), UINT8_C(221), UINT8_C( 23), UINT8_C( 95), UINT8_C( 52),
        UINT8_C( 98), UINT8_C( 62), UINT8_C(143), UINT8_C(248), UINT8_C(200), UINT8_C( 46), UINT8_C( 55), UINT8_C(148) } },
    { { UINT8_C(105), UINT8_C( 51), UINT8_C(151), UINT8_C(161), UINT8_C( 86), UINT8_C(  3), UINT8_C( 46), UINT8_C( 59),
        UINT8_C(161), UINT8_C(212), UINT8_C(207), UINT8_C( 99), UINT8_C( 32), UINT8_C(212), UINT8_C( 96), UINT8_C(113),
        UINT8_C(126), UINT8_C(166), UINT8_C(177), UINT8_C( 91), UINT8_C( 45), UINT8_C(170), UINT8_C(154), UINT8_C(242),
        UINT8_C(239), UINT8_C(204), UINT8_C(138), UINT8_C(155), UINT8_C( 54), UINT8_C( 65), UINT8_C(155), UINT8_C(159) },
      UINT32_C(3409982068),
      { UINT8_C( 54), UINT8_C(110), UINT8_C(  6), UINT8_C(215), UINT8_C( 66), UINT8_C(213), UINT8_C( 59), UINT8_C( 98),
        UINT8_C(170), UINT8_C(155), UINT8_C(211), UINT8_C( 40), UINT8_C( 66), UINT8_C(132), UINT8_C(131), UINT8_C(111),
        UINT8_C( 46), UINT8_C( 29), UINT8_C( 98), UINT8_C( 29), UINT8_C(233), UINT8_C(236), UINT8_C(185), UINT8_C( 31),
        UINT8_C( 46), UINT8_C( 84), UINT8_C(190), UINT8_C(162), UINT8_C(134), UINT8_C(254), UINT8_C(109), UINT8_C(188) },
      { UINT8_C(108), UINT8_C(115), UINT8_C(148), UINT8_C(174), UINT8_C( 73), UINT8_C(207), UINT8_C( 16), UINT8_C(243),
        UINT8_C(106), UINT8_C(227), UINT8_C( 27), UINT8_C(172), UINT8_C(104), UINT8_C(158), UINT8_C( 28), UINT8_C(150),
        UINT8_C(187), UINT8_C(126), UINT8_C(180), UINT8_C(164), UINT8_C(106), UINT8_C(109), UINT8_C(196), UINT8_C(152),
        UINT8_C(193), UINT8_C(130), UINT8_C( 59), UINT8_C( 71), UINT8_C(129), UINT8_C(168), UINT8_C(  4), UINT8_C(237) },
      { UINT8_C(105), UINT8_C( 51), UINT8_C(  6), UINT8_C(161), UINT8_C( 66), UINT8_C(207), UINT8_C( 16), UINT8_C( 59),
        UINT8_C(161), UINT8_C(155), UINT8_C(207), UINT8_C( 99), UINT8_C( 66), UINT8_C(132), UINT8_C( 96), UINT8_C(113),
        UINT8_C(126), UINT8_C(166), UINT8_C(177), UINT8_C( 91), UINT8_C( 45), UINT8_C(170), UINT8_C(185), UINT8_C(242),
        UINT8_C( 46), UINT8_C( 84), UINT8_C(138), UINT8_C( 71), UINT8_C( 54), UINT8_C( 65), UINT8_C(  4), UINT8_C(188) } },
    { { UINT8_C( 28), UINT8_C(152), UINT8_C(156), UINT8_C(101), UINT8_C(103), UINT8_C(172), UINT8_C( 88), UINT8_C(209),
        UINT8_C(144), UINT8_C(115), UINT8_C(126), UINT8_C(248), UINT8_C( 17), UINT8_C(154), UINT8_C(142), UINT8_C(204),
        UINT8_C( 24), UINT8_C( 66), UINT8_C(112), UINT8_C(130), UINT8_C(175), UINT8_C( 52), UINT8_C( 27), UINT8_C(112),
        UINT8_C(183), UINT8_C( 86), UINT8_C(184), UINT8_C( 56), UINT8_C(254), UINT8_C(188), UINT8_C( 37), UINT8_C( 26) },
      UINT32_C(3145711956),
      { UINT8_C(110), UINT8_C(215), UINT8_C(140), UINT8_C(254), UINT8_C( 74), UINT8_C( 10), UINT8_C(246), UINT8_C( 91),
        UINT8_C(164), UINT8_C(132), UINT8_C( 39), UINT8_C(188), UINT8_C(199), UINT8_C(152), UINT8_C( 63), UINT8_C(118),
        UINT8_C(204), UINT8_C( 90), UINT8_C(231), UINT8_C(131), UINT8_C(176), UINT8_C(159), UINT8_C(187), UINT8_C(174),
        UINT8_C( 91), UINT8_C(225), UINT8_C(201), UINT8_C(175), UINT8_C(162), UINT8_C( 72), UINT8_C(106), UINT8_C( 16) },
      { UINT8_C( 32), UINT8_C(246), UINT8_C( 14), UINT8_C(106), UINT8_C(  1), UINT8_C(  4), UINT8_C(198), UINT8_C(165),
        UINT8_C(137), UINT8_C(237), UINT8_C( 98), UINT8_C( 80), UINT8_C(133), UINT8_C(161), UINT8_C(198), UINT8_C( 82),
        UINT8_C(251), UINT8_C(173), UINT8_C(213), UINT8_C(171), UINT8_C( 76), UINT8_C(145), UINT8_C( 89), UINT8_C(167),
        UINT8_C(114), UINT8_C( 34), UINT8_C( 86), UINT8_C( 20), UINT8_C(107), UINT8_C(192), UINT8_C( 37), UINT8_C(139) },
      { UINT8_C( 28), UINT8_C(152), UINT8_C( 14), UINT8_C(101), UINT8_C(  1), UINT8_C(172), UINT8_C(198), UINT8_C(209),
        UINT8_C(137), UINT8_C(115), UINT8_C(126), UINT8_C(248), UINT8_C( 17), UINT8_C(154), UINT8_C( 63), UINT8_C( 82),
        UINT8_C(204), UINT8_C( 90), UINT8_C(213), UINT8_C(131), UINT8_C( 76), UINT8_C(145), UINT8_C( 89), UINT8_C(112),
        UINT8_C( 91), UINT8_C( 34), UINT8_C(184), UINT8_C( 20), UINT8_C(107), UINT8_C( 72), UINT8_C( 37), UINT8_C( 16) } },
    { { UINT8_C(183), UINT8_C( 51), UINT8_C(245), UINT8_C(184), UINT8_C( 56), UINT8_C(187), UINT8_C( 93), UINT8_C(193),
        UINT8_C(169), UINT8_C(191), UINT8_C( 17), UINT8_C( 46), UINT8_C( 96), UINT8_C(215), UINT8_C(128), UINT8_C( 91),
        UINT8_C(133), UINT8_C( 86), UINT8_C(  6), UINT8_C(209), UINT8_C(231), UINT8_C( 96), UINT8_C(121), UINT8_C( 89),
        UINT8_C(130), UINT8_C(207), UINT8_C(109), UINT8_C(237), UINT8_C(144), UINT8_C(146), UINT8_C(120), UINT8_C( 71) },
      UINT32_C(4278152902),
      { UINT8_C( 41), UINT8_C( 92), UINT8_C(191), UINT8_C(210), UINT8_C( 28), UINT8_C(208), UINT8_C(  1), UINT8_C(124),
        UINT8_C(167), UINT8_C(129), UINT8_C(216), UINT8_C( 44), UINT8_C(215), UINT8_C(222), UINT8_C(254), UINT8_C(190),
        UINT8_C( 62), UINT8_C(119), UINT8_C( 23), UINT8_C(193), UINT8_C( 70), UINT8_C(133), UINT8_C(174), UINT8_C(214),
        UINT8_C( 23), UINT8_C( 39), UINT8_C( 29), UINT8_C(221), UINT8_C(149), UINT8_C( 28), UINT8_C(219), UINT8_C(190) },
      { UINT8_C(121), UINT8_C(154), UINT8_C(145), UINT8_C(149), UINT8_C(106), UINT8_C(146), UINT8_C( 17), UINT8_C( 18),
        UINT8_C( 19), UINT8_C(233), UINT8_C( 62), UINT8_C(235), UINT8_C(200), UINT8_C( 60), UINT8_C(169), UINT8_C(  6),
        UINT8_C(179), UINT8_C(193), UINT8_C(199), UINT8_C(250), UINT8_C( 70), UINT8_C(118), UINT8_C(208), UINT8_C( 93),
        UINT8_C(157), UINT8_C(238), UINT8_C( 59), UINT8_C( 50), UINT8_C( 10), UINT8_C( 22), UINT8_C(240), UINT8_C(131) },
      { UINT8_C(183), UINT8_C( 92), UINT8_C(145), UINT8_C(184), UINT8_C( 56), UINT8_C(187), UINT8_C(  1), UINT8_C( 18),
        UINT8_C(169), UINT8_C(129), UINT8_C( 62), UINT8_C( 44), UINT8_C( 96), UINT8_C( 60), UINT8_C(169), UINT8_C( 91),
        UINT8_C( 62), UINT8_C(119), UINT8_C( 23), UINT8_C(193), UINT8_C( 70), UINT8_C(118), UINT8_C(174), UINT8_C( 93),
        UINT8_C(130), UINT8_C( 39), UINT8_C( 29), UINT8_C( 50), UINT8_C( 10), UINT8_C( 22), UINT8_C(219), UINT8_C(131) } },
    { { UINT8_C(177), UINT8_C(129), UINT8_C( 24), UINT8_C( 27), UINT8_C( 19), UINT8_C( 42), UINT8_C( 45), UINT8_C( 39),
        UINT8_C( 19), UINT8_C(108), UINT8_C( 18), UINT8_C(219), UINT8_C(168), UINT8_C(187), UINT8_C(226), UINT8_C( 92),
        UINT8_C(124), UINT8_C(169), UINT8_C( 86), UINT8_C(194), UINT8_C( 31), UINT8_C( 38), UINT8_C( 32), UINT8_C(188),
        UINT8_C( 20), UINT8_C( 91), UINT8_C(238), UINT8_C( 31), UINT8_C(113), UINT8_C(223), UINT8_C(162), UINT8_C( 34) },
      UINT32_C(1950268256),
      { UINT8_C(229), UINT8_C(107), UINT8_C(155), UINT8_C(248), UINT8_C(215), UINT8_C(173), UINT8_C(212), UINT8_C(128),
        UINT8_C(104), UINT8_C(182), UINT8_C(220), UINT8_C(229), UINT8_C( 95), UINT8_C( 50), UINT8_C(167), UINT8_C(127),
        UINT8_C( 88), UINT8_C(199), UINT8_C( 59), UINT8_C(109), UINT8_C( 34), UINT8_C( 42), UINT8_C(140), UINT8_C(148),
        UINT8_C(  9), UINT8_C( 46), UINT8_C(182), UINT8_C(105), UINT8_C(233), UINT8_C(244), UINT8_C(221), UINT8_C(206) },
      { UINT8_C( 96), UINT8_C(120), UINT8_C(199), UINT8_C( 55), UINT8_C( 37), UINT8_C(155), UINT8_C(183), UINT8_C(142),
        UINT8_C( 81), UINT8_C(147), UINT8_C(115), UINT8_C(176), UINT8_C(197), UINT8_C( 26), UINT8_C( 47), UINT8_C( 30),
        UINT8_C(226), UINT8_C(107), UINT8_C(139), UINT8_C(  4), UINT8_C(149), UINT8_C( 23), UINT8_C(152), UINT8_C(158),
        UINT8_C( 69), UINT8_C( 79), UINT8_C(  7), UINT8_C( 47), UINT8_C( 67), UINT8_C(229), UINT8_C(253), UINT8_C(163) },
      { UINT8_C(177), UINT8_C(129), UINT8_C( 24), UINT8_C( 27), UINT8_C( 19), UINT8_C(155), UINT8_C(183), UINT8_C( 39),
        UINT8_C( 81), UINT8_C(147), UINT8_C( 18), UINT8_C(176), UINT8_C( 95), UINT8_C( 26), UINT8_C(226), UINT8_C( 30),
        UINT8_C(124), UINT8_C(107), UINT8_C( 59), UINT8_C(  4), UINT8_C( 34), UINT8_C( 23), UINT8_C( 32), UINT8_C(188),
        UINT8_C( 20), UINT8_C( 91), UINT8_C(  7), UINT8_C( 31), UINT8_C( 67), UINT8_C(229), UINT8_C(221), UINT8_C( 34) } },
    { { UINT8_C( 93), UINT8_C(196), UINT8_C(219), UINT8_C(131), UINT8_C( 95), UINT8_C(146), UINT8_C( 17), UINT8_C(176),
        UINT8_C( 38), UINT8_C(132), UINT8_C( 97), UINT8_C(235), UINT8_C(158), UINT8_C(144), UINT8_C(  9), UINT8_C(128),
        UINT8_C(251), UINT8_C(148), UINT8_C(133), UINT8_C(144), UINT8_C(171), UINT8_C( 29), UINT8_C( 46), UINT8_C(241),
        UINT8_C(108), UINT8_C( 54), UINT8_C( 32), UINT8_C(176), UINT8_C( 27), UINT8_C( 29), UINT8_C( 83), UINT8_C(120) },
      UINT32_C(1106980578),
      { UINT8_C(193), UINT8_C( 12), UINT8_C(242), UINT8_C(231), UINT8_C(144), UINT8_C( 83), UINT8_C(210), UINT8_C( 47),
        UINT8_C(227), UINT8_C(220), UINT8_C(175), UINT8_C(223), UINT8_C(112), UINT8_C( 52), UINT8_C(111), UINT8_C( 28),
        UINT8_C( 82), UINT8_C(158), UINT8_C( 13), UINT8_C(190), UINT8_C(212), UINT8_C( 45), UINT8_C(110), UINT8_C(239),
        UINT8_C( 74), UINT8_C(194), UINT8_C(103), UINT8_C( 44), UINT8_C(240), UINT8_C( 99), UINT8_C(110), UINT8_C(177) },
      { UINT8_C(111), UINT8_C( 96), UINT8_C(152), UINT8_C(  0), UINT8_C(179), UINT8_C(107), UINT8_C( 47), UINT8_C(150),
        UINT8_C( 71), UINT8_C(222), UINT8_C(117), UINT8_C(183), UINT8_C( 19), UINT8_C(229), UINT8_C(211), UINT8_C(101),
        UINT8_C(131), UINT8_C(224), UINT8_C( 35), UINT8_C( 87), UINT8_C( 13), UINT8_C(146), UINT8_C( 70), UINT8_C( 88),
        UINT8_C( 84), UINT8_C(173), UINT8_C(132), UINT8_C( 68), UINT8_C( 16), UINT8_C(242), UINT8_C(246), UINT8_C(128) },
      { UINT8_C( 93), UINT8_C( 12), UINT8_C(219), UINT8_C(131), UINT8_C( 95), UINT8_C( 83), UINT8_C( 47), UINT8_C( 47),
        UINT8_C( 38), UINT8_C(220), UINT8_C(117), UINT8_C(183), UINT8_C(158), UINT8_C( 52), UINT8_C(  9), UINT8_C(128),
        UINT8_C( 82), UINT8_C(158), UINT8_C(133), UINT8_C( 87), UINT8_C( 13), UINT8_C( 45), UINT8_C( 70), UINT8_C( 88),
        UINT8_C( 74), UINT8_C( 54), UINT8_C( 32), UINT8_C(176), UINT8_C( 27), UINT8_C( 29), UINT8_C(110), UINT8_C(120) } },
    { { UINT8_C( 82), UINT8_C(142), UINT8_C(128), UINT8_C(  5), UINT8_C(249), UINT8_C(175), UINT8_C(156), UINT8_C( 64),
        UINT8_C(141), UINT8_C( 17), UINT8_C(248), UINT8_C(160), UINT8_C(246), UINT8_C(203), UINT8_C(  5), UINT8_C(121),
        UINT8_C(172), UINT8_C( 41), UINT8_C(208), UINT8_C(185), UINT8_C(187), UINT8_C( 22), UINT8_C( 17), UINT8_C( 15),
        UINT8_C(196), UINT8_C(150), UINT8_C( 83), UINT8_C(212), UINT8_C(136), UINT8_C( 73), UINT8_C( 84), UINT8_C(219) },
      UINT32_C(3521172696),
      { UINT8_C(131), UINT8_C(124), UINT8_C( 18), UINT8_C( 17), UINT8_C(142), UINT8_C( 10), UINT8_C(177), UINT8_C(132),
        UINT8_C(213), UINT8_C(183), UINT8_C(254), UINT8_C(129), UINT8_C(224), UINT8_C(206), UINT8_C( 59), UINT8_C(155),
        UINT8_C(229), UINT8_C( 76), UINT8_C(170), UINT8_C(169), UINT8_C(226), UINT8_C(253), UINT8_C(125), UINT8_C(107),
        UINT8_C( 71), UINT8_C(210), UINT8_C( 70), UINT8_C( 31), UINT8_C(166), UINT8_C( 38), UINT8_C(240), UINT8_C( 42) },
      { UINT8_C(163), UINT8_C(  2), UINT8_C( 59), UINT8_C( 49), UINT8_C( 12), UINT8_C(236), UINT8_C(181), UINT8_C(226),
        UINT8_C(163), UINT8_C(179), UINT8_C( 99), UINT8_C(131), UINT8_C(130), UINT8_C(158), UINT8_C( 30), UINT8_C(103),
        UINT8_C(235), UINT8_C(200), UINT8_C( 16), UINT8_C(205), UINT8_C(198), UINT8_C(141), UINT8_C( 56), UINT8_C( 13),
        UINT8_C( 95), UINT8_C(126), UINT8_C( 44), UINT8_C(  6), UINT8_C(165), UINT8_C( 28), UINT8_C( 48), UINT8_C( 72) },
      { UINT8_C( 82), UINT8_C(142), UINT8_C(128), UINT8_C( 17), UINT8_C( 12), UINT8_C(175), UINT8_C(177), UINT8_C(132),
        UINT8_C(141), UINT8_C( 17), UINT8_C( 99), UINT8_C(160), UINT8_C(130), UINT8_C(203), UINT8_C( 30), UINT8_C(103),
        UINT8_C(172), UINT8_C( 41), UINT8_C(208), UINT8_C(185), UINT8_C(187), UINT8_C(141), UINT8_C( 56), UINT8_C( 13),
        UINT8_C( 71), UINT8_C(150), UINT8_C( 83), UINT8_C(212), UINT8_C(165), UINT8_C( 73), UINT8_C( 48), UINT8_C( 42) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi8(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_epu8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_epu8");
    easysimd_test_x86_assert_equal_u8x32(r, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_u8x32();
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m256i b = easysimd_test_x86_random_u8x32();
    easysimd__m256i r = easysimd_mm256_mask_min_epu8(src, k, a, b);

    easysimd_test_x86_write_u8x32(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_min_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t k;
    const uint8_t a[32];
    const uint8_t b[32];
    const uint8_t r[32];
  } test_vec[] = {
    { UINT32_C(2035764098),
      { UINT8_C(150), UINT8_C( 16), UINT8_C(192), UINT8_C(117), UINT8_C(232), UINT8_C( 60), UINT8_C( 77), UINT8_C(162),
        UINT8_C(241), UINT8_C( 31), UINT8_C( 24), UINT8_C( 39), UINT8_C( 81), UINT8_C( 28), UINT8_C( 45), UINT8_C(171),
        UINT8_C( 90), UINT8_C(106), UINT8_C( 97), UINT8_C(163), UINT8_C( 43), UINT8_C( 44), UINT8_C(243), UINT8_C(194),
        UINT8_C( 81), UINT8_C(252), UINT8_C(182), UINT8_C(212), UINT8_C( 71), UINT8_C( 13), UINT8_C( 77), UINT8_C(221) },
      { UINT8_C( 29), UINT8_C( 13), UINT8_C( 83), UINT8_C(  5), UINT8_C( 73), UINT8_C(160), UINT8_C(168), UINT8_C( 58),
        UINT8_C(191), UINT8_C(192), UINT8_C( 98), UINT8_C( 16), UINT8_C(220), UINT8_C(143), UINT8_C(187), UINT8_C( 54),
        UINT8_C(250), UINT8_C( 28), UINT8_C(217), UINT8_C( 37), UINT8_C( 72), UINT8_C(204), UINT8_C(232), UINT8_C(154),
        UINT8_C(200), UINT8_C(158), UINT8_C(110), UINT8_C( 15), UINT8_C(171), UINT8_C(187), UINT8_C(237), UINT8_C(200) },
      { UINT8_C(  0), UINT8_C( 13), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 58),
        UINT8_C(191), UINT8_C( 31), UINT8_C(  0), UINT8_C( 16), UINT8_C(  0), UINT8_C(  0), UINT8_C( 45), UINT8_C(  0),
        UINT8_C( 90), UINT8_C( 28), UINT8_C( 97), UINT8_C(  0), UINT8_C( 43), UINT8_C(  0), UINT8_C(232), UINT8_C(  0),
        UINT8_C( 81), UINT8_C(  0), UINT8_C(  0), UINT8_C( 15), UINT8_C( 71), UINT8_C( 13), UINT8_C( 77), UINT8_C(  0) } },
    { UINT32_C( 298729672),
      { UINT8_C(224), UINT8_C(118), UINT8_C( 76), UINT8_C(159), UINT8_C( 54), UINT8_C(174), UINT8_C(175), UINT8_C( 18),
        UINT8_C( 61), UINT8_C(106), UINT8_C( 73), UINT8_C( 55), UINT8_C(134), UINT8_C( 34), UINT8_C( 93), UINT8_C(206),
        UINT8_C(239), UINT8_C( 69), UINT8_C(104), UINT8_C(183), UINT8_C(227), UINT8_C(214), UINT8_C(199), UINT8_C(142),
        UINT8_C(145), UINT8_C(180), UINT8_C( 86), UINT8_C( 89), UINT8_C(244), UINT8_C( 36), UINT8_C(107), UINT8_C(212) },
      { UINT8_C(154), UINT8_C(183), UINT8_C(115), UINT8_C(208), UINT8_C(101), UINT8_C( 34), UINT8_C(227), UINT8_C(162),
        UINT8_C(140), UINT8_C( 44), UINT8_C(218), UINT8_C( 18), UINT8_C( 78), UINT8_C( 55), UINT8_C(224), UINT8_C( 61),
        UINT8_C(124), UINT8_C( 73), UINT8_C(245), UINT8_C( 95), UINT8_C( 31), UINT8_C(188), UINT8_C(237), UINT8_C(177),
        UINT8_C(112), UINT8_C( 67), UINT8_C( 10), UINT8_C(100), UINT8_C(104), UINT8_C(117), UINT8_C( 56), UINT8_C(  2) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(159), UINT8_C(  0), UINT8_C(  0), UINT8_C(175), UINT8_C( 18),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 93), UINT8_C(  0),
        UINT8_C(  0), UINT8_C( 69), UINT8_C(104), UINT8_C( 95), UINT8_C(  0), UINT8_C(  0), UINT8_C(199), UINT8_C(142),
        UINT8_C(112), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(104), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0) } },
    { UINT32_C(2446568236),
      { UINT8_C(205), UINT8_C(182), UINT8_C( 52), UINT8_C( 89), UINT8_C(226), UINT8_C( 14), UINT8_C(107), UINT8_C( 48),
        UINT8_C( 69), UINT8_C( 75), UINT8_C(110), UINT8_C(193), UINT8_C(148), UINT8_C( 99), UINT8_C( 32), UINT8_C(180),
        UINT8_C( 31), UINT8_C( 13), UINT8_C(101), UINT8_C(143), UINT8_C( 80), UINT8_C(111), UINT8_C(243), UINT8_C(184),
        UINT8_C(229), UINT8_C( 43), UINT8_C(187), UINT8_C( 17), UINT8_C(214), UINT8_C(142), UINT8_C(163), UINT8_C(163) },
      { UINT8_C( 68), UINT8_C(215), UINT8_C(252), UINT8_C( 38), UINT8_C(229), UINT8_C(103), UINT8_C( 86), UINT8_C( 42),
        UINT8_C(178), UINT8_C(196), UINT8_C(235), UINT8_C( 71), UINT8_C( 39), UINT8_C( 11), UINT8_C(251), UINT8_C( 70),
        UINT8_C( 24), UINT8_C( 96), UINT8_C(213), UINT8_C(104), UINT8_C(207), UINT8_C(200), UINT8_C( 33), UINT8_C(180),
        UINT8_C(243), UINT8_C(220), UINT8_C(198), UINT8_C(201), UINT8_C(106), UINT8_C(105), UINT8_C(108), UINT8_C(174) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C( 52), UINT8_C( 38), UINT8_C(  0), UINT8_C( 14), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 69), UINT8_C( 75), UINT8_C(  0), UINT8_C( 71), UINT8_C(  0), UINT8_C( 11), UINT8_C(  0), UINT8_C( 70),
        UINT8_C( 24), UINT8_C( 13), UINT8_C(  0), UINT8_C(  0), UINT8_C( 80), UINT8_C(  0), UINT8_C( 33), UINT8_C(180),
        UINT8_C(229), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(106), UINT8_C(  0), UINT8_C(  0), UINT8_C(163) } },
    { UINT32_C( 634677312),
      { UINT8_C(207), UINT8_C( 42), UINT8_C( 79), UINT8_C(130), UINT8_C(239), UINT8_C( 58), UINT8_C(201), UINT8_C( 22),
        UINT8_C( 69), UINT8_C(196), UINT8_C( 93), UINT8_C( 93), UINT8_C( 36), UINT8_C( 50), UINT8_C(197), UINT8_C(243),
        UINT8_C(251), UINT8_C(230), UINT8_C(168), UINT8_C(238), UINT8_C(194), UINT8_C(110), UINT8_C(184), UINT8_C( 44),
        UINT8_C(215), UINT8_C( 36), UINT8_C(218), UINT8_C( 23), UINT8_C(141), UINT8_C(174), UINT8_C( 60), UINT8_C( 92) },
      { UINT8_C(217), UINT8_C(139), UINT8_C(222), UINT8_C(200), UINT8_C(197), UINT8_C(167), UINT8_C(222), UINT8_C( 10),
        UINT8_C(107), UINT8_C( 59), UINT8_C(103), UINT8_C(143), UINT8_C(110), UINT8_C( 44), UINT8_C(131), UINT8_C(105),
        UINT8_C( 19), UINT8_C( 43), UINT8_C( 87), UINT8_C(213), UINT8_C(153), UINT8_C( 15), UINT8_C(  2), UINT8_C(112),
        UINT8_C( 52), UINT8_C(220), UINT8_C(135), UINT8_C(193), UINT8_C(139), UINT8_C(195), UINT8_C( 29), UINT8_C(100) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(201), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 93), UINT8_C(  0), UINT8_C( 44), UINT8_C(131), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C( 87), UINT8_C(  0), UINT8_C(153), UINT8_C(  0), UINT8_C(  2), UINT8_C( 44),
        UINT8_C( 52), UINT8_C(  0), UINT8_C(135), UINT8_C(  0), UINT8_C(  0), UINT8_C(174), UINT8_C(  0), UINT8_C(  0) } },
    { UINT32_C( 321715278),
      { UINT8_C(163), UINT8_C( 10), UINT8_C( 29), UINT8_C( 15), UINT8_C( 70), UINT8_C(132), UINT8_C(158), UINT8_C(180),
        UINT8_C(176), UINT8_C( 33), UINT8_C( 29), UINT8_C(195), UINT8_C( 76), UINT8_C(116), UINT8_C(153), UINT8_C(229),
        UINT8_C(132), UINT8_C(155), UINT8_C( 85), UINT8_C(184), UINT8_C(119), UINT8_C(220), UINT8_C(121), UINT8_C(  2),
        UINT8_C(159), UINT8_C(150), UINT8_C(102), UINT8_C(237), UINT8_C(146), UINT8_C(146), UINT8_C(  0), UINT8_C( 54) },
      { UINT8_C(157), UINT8_C( 29), UINT8_C( 69), UINT8_C(227), UINT8_C(161), UINT8_C(227), UINT8_C(151), UINT8_C( 82),
        UINT8_C(  5), UINT8_C(180), UINT8_C( 21), UINT8_C( 81), UINT8_C( 40), UINT8_C(174), UINT8_C( 55), UINT8_C(172),
        UINT8_C( 73), UINT8_C(140), UINT8_C(100), UINT8_C(193), UINT8_C(105), UINT8_C(221), UINT8_C(195), UINT8_C(  8),
        UINT8_C(116), UINT8_C( 42), UINT8_C(246), UINT8_C(  6), UINT8_C(188), UINT8_C(246), UINT8_C( 60), UINT8_C( 89) },
      { UINT8_C(  0), UINT8_C( 10), UINT8_C( 29), UINT8_C( 15), UINT8_C(  0), UINT8_C(  0), UINT8_C(151), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C( 21), UINT8_C( 81), UINT8_C( 40), UINT8_C(116), UINT8_C( 55), UINT8_C(172),
        UINT8_C(  0), UINT8_C(  0), UINT8_C( 85), UINT8_C(184), UINT8_C(  0), UINT8_C(220), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(116), UINT8_C( 42), UINT8_C(  0), UINT8_C(  0), UINT8_C(146), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0) } },
    { UINT32_C(3040641300),
      { UINT8_C(101), UINT8_C(211), UINT8_C(  7), UINT8_C(106), UINT8_C(135), UINT8_C( 29), UINT8_C(187), UINT8_C(176),
        UINT8_C(203), UINT8_C(242), UINT8_C( 92), UINT8_C( 21), UINT8_C(127), UINT8_C(193), UINT8_C(214), UINT8_C(232),
        UINT8_C(158), UINT8_C(153), UINT8_C(240), UINT8_C( 18), UINT8_C(195), UINT8_C(230), UINT8_C( 25), UINT8_C(128),
        UINT8_C(221), UINT8_C( 85), UINT8_C(217), UINT8_C(241), UINT8_C(215), UINT8_C( 22), UINT8_C(166), UINT8_C( 60) },
      { UINT8_C(233), UINT8_C(174), UINT8_C(166), UINT8_C(113), UINT8_C(203), UINT8_C( 97), UINT8_C( 33), UINT8_C(150),
        UINT8_C( 84), UINT8_C(125), UINT8_C(171), UINT8_C(211), UINT8_C( 62), UINT8_C(129), UINT8_C(187), UINT8_C(221),
        UINT8_C( 27), UINT8_C(171), UINT8_C(239), UINT8_C(222), UINT8_C(146), UINT8_C(  8), UINT8_C( 94), UINT8_C(111),
        UINT8_C( 94), UINT8_C( 56), UINT8_C( 96), UINT8_C( 53), UINT8_C( 78), UINT8_C(  6), UINT8_C(113), UINT8_C( 55) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  7), UINT8_C(  0), UINT8_C(135), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 84), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(221),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(239), UINT8_C( 18), UINT8_C(146), UINT8_C(  8), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 94), UINT8_C(  0), UINT8_C( 96), UINT8_C(  0), UINT8_C( 78), UINT8_C(  6), UINT8_C(  0), UINT8_C( 55) } },
    { UINT32_C(2141722548),
      { UINT8_C(120), UINT8_C(201), UINT8_C( 22), UINT8_C(204), UINT8_C( 71), UINT8_C(193), UINT8_C(159), UINT8_C(133),
        UINT8_C( 67), UINT8_C( 90), UINT8_C( 98), UINT8_C( 94), UINT8_C(  6), UINT8_C( 82), UINT8_C( 60), UINT8_C(152),
        UINT8_C( 90), UINT8_C(155), UINT8_C(  7), UINT8_C(184), UINT8_C(211), UINT8_C(103), UINT8_C(237), UINT8_C( 33),
        UINT8_C(109), UINT8_C( 94), UINT8_C( 88), UINT8_C( 34), UINT8_C(117), UINT8_C(  1), UINT8_C(161), UINT8_C(238) },
      { UINT8_C(202), UINT8_C(183), UINT8_C(186), UINT8_C( 17), UINT8_C(121), UINT8_C( 90), UINT8_C(151), UINT8_C(188),
        UINT8_C(180), UINT8_C(249), UINT8_C( 26), UINT8_C(186), UINT8_C( 75), UINT8_C( 86), UINT8_C( 82), UINT8_C(166),
        UINT8_C(241), UINT8_C( 89), UINT8_C( 94), UINT8_C(196), UINT8_C(192), UINT8_C( 76), UINT8_C(229), UINT8_C( 46),
        UINT8_C(170), UINT8_C( 62), UINT8_C( 80), UINT8_C( 32), UINT8_C( 63), UINT8_C(241), UINT8_C( 14), UINT8_C(  9) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C( 22), UINT8_C(  0), UINT8_C( 71), UINT8_C( 90), UINT8_C(  0), UINT8_C(133),
        UINT8_C( 67), UINT8_C( 90), UINT8_C( 26), UINT8_C(  0), UINT8_C(  6), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(184), UINT8_C(  0), UINT8_C( 76), UINT8_C(  0), UINT8_C( 33),
        UINT8_C(109), UINT8_C( 62), UINT8_C( 80), UINT8_C( 32), UINT8_C( 63), UINT8_C(  1), UINT8_C( 14), UINT8_C(  0) } },
    { UINT32_C( 572246185),
      { UINT8_C( 34), UINT8_C(178), UINT8_C(222), UINT8_C(215), UINT8_C(171), UINT8_C(248), UINT8_C(145), UINT8_C(247),
        UINT8_C( 78), UINT8_C(228), UINT8_C(157), UINT8_C( 64), UINT8_C( 61), UINT8_C(251), UINT8_C(  4), UINT8_C(254),
        UINT8_C( 71), UINT8_C(234), UINT8_C( 44), UINT8_C(242), UINT8_C( 40), UINT8_C(124), UINT8_C( 18), UINT8_C(103),
        UINT8_C(109), UINT8_C( 32), UINT8_C(112), UINT8_C( 22), UINT8_C(232), UINT8_C(139), UINT8_C( 56), UINT8_C( 11) },
      { UINT8_C( 61), UINT8_C( 22), UINT8_C(226), UINT8_C(233), UINT8_C( 14), UINT8_C(115), UINT8_C(224), UINT8_C( 93),
        UINT8_C( 87), UINT8_C(125), UINT8_C(157), UINT8_C(149), UINT8_C(120), UINT8_C(161), UINT8_C(147), UINT8_C(192),
        UINT8_C(139), UINT8_C(191), UINT8_C(178), UINT8_C(179), UINT8_C( 59), UINT8_C(196), UINT8_C( 26), UINT8_C(168),
        UINT8_C(228), UINT8_C(139), UINT8_C(191), UINT8_C(204), UINT8_C( 22), UINT8_C(247), UINT8_C(215), UINT8_C( 84) },
      { UINT8_C( 34), UINT8_C(  0), UINT8_C(  0), UINT8_C(215), UINT8_C(  0), UINT8_C(115), UINT8_C(  0), UINT8_C( 93),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 64), UINT8_C(  0), UINT8_C(  0), UINT8_C(  4), UINT8_C(192),
        UINT8_C( 71), UINT8_C(191), UINT8_C(  0), UINT8_C(179), UINT8_C( 40), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C( 32), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(139), UINT8_C(  0), UINT8_C(  0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_epu8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_epu8");
    easysimd_test_x86_assert_equal_u8x32(r, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m256i b = easysimd_test_x86_random_u8x32();
    easysimd__m256i r = easysimd_mm256_maskz_min_epu8(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_min_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t src[16];
    const uint16_t k;
    const uint16_t a[16];
    const uint16_t b[16];
    const uint16_t r[16];
  } test_vec[] = {
    { { UINT16_C(27423), UINT16_C(11129), UINT16_C(11863), UINT16_C(64269), UINT16_C(29154), UINT16_C(25726), UINT16_C(40207), UINT16_C(64203),
        UINT16_C(56165), UINT16_C(11208), UINT16_C(  104), UINT16_C(51256), UINT16_C(25727), UINT16_C( 9422), UINT16_C(65153), UINT16_C(41068) },
      UINT16_C(58729),
      { UINT16_C(49355), UINT16_C(55571), UINT16_C(62907), UINT16_C(14922), UINT16_C(22873), UINT16_C( 9431), UINT16_C(15444), UINT16_C( 7423),
        UINT16_C(26728), UINT16_C(40988), UINT16_C(39728), UINT16_C(65029), UINT16_C(34495), UINT16_C(11260), UINT16_C(25894), UINT16_C(61712) },
      { UINT16_C( 9253), UINT16_C(57802), UINT16_C( 5145), UINT16_C(29467), UINT16_C(62062), UINT16_C(49815), UINT16_C(38702), UINT16_C(38622),
        UINT16_C(64255), UINT16_C(12087), UINT16_C(15510), UINT16_C(21805), UINT16_C(10690), UINT16_C(59521), UINT16_C(37262), UINT16_C(46041) },
      { UINT16_C( 9253), UINT16_C(11129), UINT16_C(11863), UINT16_C(14922), UINT16_C(29154), UINT16_C( 9431), UINT16_C(15444), UINT16_C(64203),
        UINT16_C(26728), UINT16_C(11208), UINT16_C(15510), UINT16_C(51256), UINT16_C(25727), UINT16_C(11260), UINT16_C(25894), UINT16_C(46041) } },
    { { UINT16_C(42165), UINT16_C(53140), UINT16_C(44984), UINT16_C( 9794), UINT16_C(55713), UINT16_C(53480), UINT16_C(50800), UINT16_C(28518),
        UINT16_C(40385), UINT16_C(22430), UINT16_C(52185), UINT16_C(39852), UINT16_C(11764), UINT16_C(33411), UINT16_C(23999), UINT16_C(29750) },
      UINT16_C(51713),
      { UINT16_C(47427), UINT16_C(34170), UINT16_C( 7136), UINT16_C(51295), UINT16_C(53227), UINT16_C(21135), UINT16_C(20543), UINT16_C(56815),
        UINT16_C(51623), UINT16_C(21417), UINT16_C(40292), UINT16_C(59521), UINT16_C(16416), UINT16_C(22085), UINT16_C(18100), UINT16_C(63520) },
      { UINT16_C(39679), UINT16_C(57213), UINT16_C(56502), UINT16_C(41384), UINT16_C(14252), UINT16_C(60403), UINT16_C(58247), UINT16_C(11976),
        UINT16_C(29100), UINT16_C( 4225), UINT16_C(  527), UINT16_C(12280), UINT16_C(15682), UINT16_C(63365), UINT16_C(42371), UINT16_C(33775) },
      { UINT16_C(39679), UINT16_C(53140), UINT16_C(44984), UINT16_C( 9794), UINT16_C(55713), UINT16_C(53480), UINT16_C(50800), UINT16_C(28518),
        UINT16_C(40385), UINT16_C( 4225), UINT16_C(52185), UINT16_C(12280), UINT16_C(11764), UINT16_C(33411), UINT16_C(18100), UINT16_C(33775) } },
    { { UINT16_C(27712), UINT16_C(63074), UINT16_C( 2633), UINT16_C(62871), UINT16_C(35649), UINT16_C(51424), UINT16_C(43118), UINT16_C( 6902),
        UINT16_C(30746), UINT16_C(10538), UINT16_C( 9082), UINT16_C(48472), UINT16_C(56672), UINT16_C(58548), UINT16_C(41858), UINT16_C(49767) },
      UINT16_C(51471),
      { UINT16_C(22712), UINT16_C(20692), UINT16_C( 5453), UINT16_C(11739), UINT16_C(18910), UINT16_C(54486), UINT16_C(61539), UINT16_C(36172),
        UINT16_C(50969), UINT16_C(29104), UINT16_C( 4484), UINT16_C(14414), UINT16_C(53493), UINT16_C(23771), UINT16_C(60051), UINT16_C(19237) },
      { UINT16_C(63811), UINT16_C(37019), UINT16_C(30223), UINT16_C(60862), UINT16_C(38079), UINT16_C( 8897), UINT16_C( 3716), UINT16_C(40368),
        UINT16_C(24789), UINT16_C(22798), UINT16_C(23665), UINT16_C(26257), UINT16_C(27692), UINT16_C(49090), UINT16_C(59478), UINT16_C(39179) },
      { UINT16_C(22712), UINT16_C(20692), UINT16_C( 5453), UINT16_C(11739), UINT16_C(35649), UINT16_C(51424), UINT16_C(43118), UINT16_C( 6902),
        UINT16_C(24789), UINT16_C(10538), UINT16_C( 9082), UINT16_C(14414), UINT16_C(56672), UINT16_C(58548), UINT16_C(59478), UINT16_C(19237) } },
    { { UINT16_C(42721), UINT16_C(61482), UINT16_C(59421), UINT16_C(56541), UINT16_C(40828), UINT16_C(  255), UINT16_C(44973), UINT16_C(33437),
        UINT16_C(43791), UINT16_C(33243), UINT16_C(27655), UINT16_C(13287), UINT16_C(43736), UINT16_C(12019), UINT16_C(65170), UINT16_C(29640) },
      UINT16_C(62116),
      { UINT16_C(49508), UINT16_C(16858), UINT16_C(22174), UINT16_C(40416), UINT16_C(36182), UINT16_C(62284), UINT16_C(23311), UINT16_C(60062),
        UINT16_C(42460), UINT16_C(50262), UINT16_C(11992), UINT16_C(52078), UINT16_C(   93), UINT16_C( 9673), UINT16_C(28275), UINT16_C(55063) },
      { UINT16_C(61743), UINT16_C(52505), UINT16_C(63815), UINT16_C(40298), UINT16_C(46727), UINT16_C(38544), UINT16_C(11794), UINT16_C(61057),
        UINT16_C(55251), UINT16_C(43954), UINT16_C( 8198), UINT16_C(25463), UINT16_C(16416), UINT16_C(38024), UINT16_C(40878), UINT16_C(56939) },
      { UINT16_C(42721), UINT16_C(61482), UINT16_C(22174), UINT16_C(56541), UINT16_C(40828), UINT16_C(38544), UINT16_C(44973), UINT16_C(60062),
        UINT16_C(43791), UINT16_C(43954), UINT16_C(27655), UINT16_C(13287), UINT16_C(   93), UINT16_C( 9673), UINT16_C(28275), UINT16_C(55063) } },
    { { UINT16_C(33936), UINT16_C(55211), UINT16_C( 5758), UINT16_C( 1396), UINT16_C( 1228), UINT16_C(56987), UINT16_C( 7218), UINT16_C( 1485),
        UINT16_C(32756), UINT16_C(64176), UINT16_C(10144), UINT16_C(49245), UINT16_C(58728), UINT16_C( 5716), UINT16_C(49284), UINT16_C( 5364) },
      UINT16_C(41028),
      { UINT16_C(49899), UINT16_C(24502), UINT16_C(33479), UINT16_C(25443), UINT16_C(38241), UINT16_C(11903), UINT16_C(29594), UINT16_C(19117),
        UINT16_C(19821), UINT16_C(51826), UINT16_C(55822), UINT16_C(25263), UINT16_C(13296), UINT16_C(58658), UINT16_C(26439), UINT16_C(12933) },
      { UINT16_C(15145), UINT16_C(61841), UINT16_C(62653), UINT16_C( 7764), UINT16_C(54153), UINT16_C( 9036), UINT16_C(64071), UINT16_C(46190),
        UINT16_C(57415), UINT16_C(21887), UINT16_C(11962), UINT16_C(43704), UINT16_C(55906), UINT16_C(43407), UINT16_C( 5185), UINT16_C(27612) },
      { UINT16_C(33936), UINT16_C(55211), UINT16_C(33479), UINT16_C( 1396), UINT16_C( 1228), UINT16_C(56987), UINT16_C(29594), UINT16_C( 1485),
        UINT16_C(32756), UINT16_C(64176), UINT16_C(10144), UINT16_C(49245), UINT16_C(58728), UINT16_C(43407), UINT16_C(49284), UINT16_C(12933) } },
    { { UINT16_C(27983), UINT16_C( 3420), UINT16_C(45154), UINT16_C(60203), UINT16_C(30851), UINT16_C(51727), UINT16_C(32114), UINT16_C(47487),
        UINT16_C(65117), UINT16_C( 5903), UINT16_C(50988), UINT16_C(36545), UINT16_C(20897), UINT16_C(58168), UINT16_C( 5221), UINT16_C(46414) },
      UINT16_C(43649),
      { UINT16_C(58306), UINT16_C(60762), UINT16_C(56783), UINT16_C(56933), UINT16_C(55208), UINT16_C(10075), UINT16_C(47249), UINT16_C(40997),
        UINT16_C(20943), UINT16_C(36967), UINT16_C( 2272), UINT16_C( 6369), UINT16_C(18411), UINT16_C(14636), UINT16_C(44540), UINT16_C(48867) },
      { UINT16_C(15761), UINT16_C(24747), UINT16_C( 4379), UINT16_C(49982), UINT16_C(39400), UINT16_C(31210), UINT16_C( 3921), UINT16_C( 8217),
        UINT16_C(32864), UINT16_C(16560), UINT16_C(37513), UINT16_C(29784), UINT16_C(34009), UINT16_C(54702), UINT16_C(37170), UINT16_C(50067) },
      { UINT16_C(15761), UINT16_C( 3420), UINT16_C(45154), UINT16_C(60203), UINT16_C(30851), UINT16_C(51727), UINT16_C(32114), UINT16_C( 8217),
        UINT16_C(65117), UINT16_C(16560), UINT16_C(50988), UINT16_C( 6369), UINT16_C(20897), UINT16_C(14636), UINT16_C( 5221), UINT16_C(48867) } },
    { { UINT16_C(16079), UINT16_C(59939), UINT16_C(24911), UINT16_C(14509), UINT16_C(38906), UINT16_C(19377), UINT16_C(52134), UINT16_C( 1643),
        UINT16_C( 6987), UINT16_C(54343), UINT16_C(40877), UINT16_C(34377), UINT16_C(63268), UINT16_C(22107), UINT16_C(61064), UINT16_C(22297) },
      UINT16_C(15405),
      { UINT16_C(31809), UINT16_C(61085), UINT16_C(38836), UINT16_C(26245), UINT16_C(11234), UINT16_C(19761), UINT16_C(31794), UINT16_C(31080),
        UINT16_C( 5713), UINT16_C(39448), UINT16_C(15516), UINT16_C(63633), UINT16_C( 6546), UINT16_C(44006), UINT16_C( 4977), UINT16_C(45799) },
      { UINT16_C(33936), UINT16_C(17569), UINT16_C( 9755), UINT16_C(64938), UINT16_C(56146), UINT16_C(33866), UINT16_C(45912), UINT16_C(43517),
        UINT16_C( 5577), UINT16_C(25923), UINT16_C(54354), UINT16_C(58461), UINT16_C(17645), UINT16_C(24208), UINT16_C(30551), UINT16_C(59153) },
      { UINT16_C(31809), UINT16_C(59939), UINT16_C( 9755), UINT16_C(26245), UINT16_C(38906), UINT16_C(19761), UINT16_C(52134), UINT16_C( 1643),
        UINT16_C( 6987), UINT16_C(54343), UINT16_C(15516), UINT16_C(58461), UINT16_C( 6546), UINT16_C(24208), UINT16_C(61064), UINT16_C(22297) } },
    { { UINT16_C(45820), UINT16_C( 5932), UINT16_C(55000), UINT16_C(10773), UINT16_C(24498), UINT16_C( 2734), UINT16_C(43794), UINT16_C(56243),
        UINT16_C(63169), UINT16_C( 4929), UINT16_C(40650), UINT16_C(47095), UINT16_C(34786), UINT16_C(14870), UINT16_C(10239), UINT16_C(64289) },
      UINT16_C(19929),
      { UINT16_C(45330), UINT16_C(10020), UINT16_C(55004), UINT16_C(35463), UINT16_C(39392), UINT16_C(37686), UINT16_C(63349), UINT16_C(46729),
        UINT16_C(21258), UINT16_C(  340), UINT16_C(14090), UINT16_C( 8329), UINT16_C(34929), UINT16_C(37447), UINT16_C( 8323), UINT16_C(38368) },
      { UINT16_C( 1234), UINT16_C(44733), UINT16_C(17626), UINT16_C(47672), UINT16_C(28381), UINT16_C(21069), UINT16_C(54885), UINT16_C(28424),
        UINT16_C(23849), UINT16_C(13169), UINT16_C(64148), UINT16_C( 1364), UINT16_C(39810), UINT16_C( 1431), UINT16_C(30652), UINT16_C(36506) },
      { UINT16_C( 1234), UINT16_C( 5932), UINT16_C(55000), UINT16_C(35463), UINT16_C(28381), UINT16_C( 2734), UINT16_C(54885), UINT16_C(28424),
        UINT16_C(21258), UINT16_C( 4929), UINT16_C(14090), UINT16_C( 1364), UINT16_C(34786), UINT16_C(14870), UINT16_C( 8323), UINT16_C(64289) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_epu16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_epu16");
    easysimd_test_x86_assert_equal_u16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_u16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_test_x86_random_u16x16();
    easysimd__m256i r = easysimd_mm256_mask_min_epu16(src, k, a, b);

    easysimd_test_x86_write_u16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_min_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const uint16_t a[16];
    const uint16_t b[16];
    const uint16_t r[16];
  } test_vec[] = {
    { UINT16_C(47374),
      { UINT16_C( 7229), UINT16_C( 7469), UINT16_C(33913), UINT16_C( 5786), UINT16_C( 4633), UINT16_C(44216), UINT16_C(17362), UINT16_C(33899),
        UINT16_C(42743), UINT16_C( 4424), UINT16_C(11343), UINT16_C( 3740), UINT16_C(46073), UINT16_C(53253), UINT16_C( 4871), UINT16_C(17546) },
      { UINT16_C(46896), UINT16_C(43361), UINT16_C(64315), UINT16_C(21952), UINT16_C(30733), UINT16_C(57345), UINT16_C(28091), UINT16_C(45668),
        UINT16_C(44307), UINT16_C(25284), UINT16_C(24793), UINT16_C(53872), UINT16_C(30227), UINT16_C( 6819), UINT16_C(11657), UINT16_C(47454) },
      { UINT16_C(    0), UINT16_C( 7469), UINT16_C(33913), UINT16_C( 5786), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(42743), UINT16_C(    0), UINT16_C(    0), UINT16_C( 3740), UINT16_C(30227), UINT16_C( 6819), UINT16_C(    0), UINT16_C(17546) } },
    { UINT16_C(49124),
      { UINT16_C( 8035), UINT16_C( 9146), UINT16_C(51316), UINT16_C(30363), UINT16_C(22184), UINT16_C( 3299), UINT16_C(62985), UINT16_C(52665),
        UINT16_C(37721), UINT16_C(51501), UINT16_C(16741), UINT16_C( 2111), UINT16_C(51547), UINT16_C(47669), UINT16_C( 6530), UINT16_C(58745) },
      { UINT16_C(13369), UINT16_C(44296), UINT16_C(41980), UINT16_C(42019), UINT16_C( 1786), UINT16_C(  944), UINT16_C(27389), UINT16_C(22224),
        UINT16_C(65021), UINT16_C(25119), UINT16_C(24382), UINT16_C(39531), UINT16_C(41000), UINT16_C(43604), UINT16_C(52666), UINT16_C(62352) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(41980), UINT16_C(    0), UINT16_C(    0), UINT16_C(  944), UINT16_C(27389), UINT16_C(22224),
        UINT16_C(37721), UINT16_C(25119), UINT16_C(16741), UINT16_C( 2111), UINT16_C(41000), UINT16_C(43604), UINT16_C(    0), UINT16_C(58745) } },
    { UINT16_C(38913),
      { UINT16_C(64928), UINT16_C(50236), UINT16_C(13985), UINT16_C(21194), UINT16_C(51001), UINT16_C( 2492), UINT16_C(47389), UINT16_C(15622),
        UINT16_C(17691), UINT16_C(34460), UINT16_C(50399), UINT16_C(13095), UINT16_C(57710), UINT16_C(65024), UINT16_C(  724), UINT16_C(29847) },
      { UINT16_C(54271), UINT16_C(41272), UINT16_C(  777), UINT16_C(17139), UINT16_C(45002), UINT16_C(59467), UINT16_C(20840), UINT16_C(33573),
        UINT16_C(49558), UINT16_C(29962), UINT16_C(12677), UINT16_C(62376), UINT16_C(43282), UINT16_C(59122), UINT16_C(35243), UINT16_C(43610) },
      { UINT16_C(54271), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(13095), UINT16_C(43282), UINT16_C(    0), UINT16_C(    0), UINT16_C(29847) } },
    { UINT16_C(37724),
      { UINT16_C(25931), UINT16_C(16022), UINT16_C(24743), UINT16_C(62189), UINT16_C(21832), UINT16_C(27971), UINT16_C(56025), UINT16_C(58158),
        UINT16_C(45903), UINT16_C(63508), UINT16_C( 9895), UINT16_C(39329), UINT16_C(19468), UINT16_C(26146), UINT16_C(32502), UINT16_C(17145) },
      { UINT16_C(36835), UINT16_C(35456), UINT16_C(28400), UINT16_C(14460), UINT16_C(49091), UINT16_C(40102), UINT16_C(54425), UINT16_C(59775),
        UINT16_C(37768), UINT16_C(12257), UINT16_C(33465), UINT16_C(50632), UINT16_C(60110), UINT16_C(50220), UINT16_C( 9576), UINT16_C(19206) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(24743), UINT16_C(14460), UINT16_C(21832), UINT16_C(    0), UINT16_C(54425), UINT16_C(    0),
        UINT16_C(37768), UINT16_C(12257), UINT16_C(    0), UINT16_C(    0), UINT16_C(19468), UINT16_C(    0), UINT16_C(    0), UINT16_C(17145) } },
    { UINT16_C(34741),
      { UINT16_C(42453), UINT16_C(20981), UINT16_C(47325), UINT16_C(33552), UINT16_C(43605), UINT16_C(54360), UINT16_C(57491), UINT16_C(29800),
        UINT16_C( 8463), UINT16_C(55286), UINT16_C(50407), UINT16_C( 5057), UINT16_C(10632), UINT16_C(36664), UINT16_C(60788), UINT16_C(18710) },
      { UINT16_C( 2962), UINT16_C(28826), UINT16_C(43715), UINT16_C( 6387), UINT16_C(19284), UINT16_C(59373), UINT16_C(21803), UINT16_C(14939),
        UINT16_C(20854), UINT16_C(23825), UINT16_C(53781), UINT16_C(40560), UINT16_C(43515), UINT16_C(28461), UINT16_C(17302), UINT16_C(10680) },
      { UINT16_C( 2962), UINT16_C(    0), UINT16_C(43715), UINT16_C(    0), UINT16_C(19284), UINT16_C(54360), UINT16_C(    0), UINT16_C(14939),
        UINT16_C( 8463), UINT16_C(23825), UINT16_C(50407), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(10680) } },
    { UINT16_C(21070),
      { UINT16_C( 4505), UINT16_C(36093), UINT16_C(20778), UINT16_C( 6104), UINT16_C(  825), UINT16_C(37996), UINT16_C(57918), UINT16_C(20454),
        UINT16_C(64320), UINT16_C(45090), UINT16_C( 7577), UINT16_C(50777), UINT16_C(61581), UINT16_C(17673), UINT16_C(22297), UINT16_C(45720) },
      { UINT16_C(38249), UINT16_C(37694), UINT16_C( 5862), UINT16_C( 8106), UINT16_C( 5658), UINT16_C(22708), UINT16_C(39672), UINT16_C(14503),
        UINT16_C(51605), UINT16_C(12265), UINT16_C(17127), UINT16_C(29941), UINT16_C(65330), UINT16_C(19385), UINT16_C(20822), UINT16_C(49149) },
      { UINT16_C(    0), UINT16_C(36093), UINT16_C( 5862), UINT16_C( 6104), UINT16_C(    0), UINT16_C(    0), UINT16_C(39672), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(12265), UINT16_C(    0), UINT16_C(    0), UINT16_C(61581), UINT16_C(    0), UINT16_C(20822), UINT16_C(    0) } },
    { UINT16_C(15590),
      { UINT16_C(52562), UINT16_C(64594), UINT16_C(27884), UINT16_C(40978), UINT16_C( 3012), UINT16_C(27706), UINT16_C(53315), UINT16_C(11317),
        UINT16_C( 7423), UINT16_C(62575), UINT16_C(41360), UINT16_C(19187), UINT16_C(19181), UINT16_C(60059), UINT16_C(33289), UINT16_C(23590) },
      { UINT16_C(31055), UINT16_C(15192), UINT16_C(27621), UINT16_C(43740), UINT16_C( 5750), UINT16_C(47382), UINT16_C(19430), UINT16_C(58854),
        UINT16_C(21864), UINT16_C(63706), UINT16_C(52726), UINT16_C(58178), UINT16_C(56855), UINT16_C( 8654), UINT16_C(62560), UINT16_C(44925) },
      { UINT16_C(    0), UINT16_C(15192), UINT16_C(27621), UINT16_C(    0), UINT16_C(    0), UINT16_C(27706), UINT16_C(19430), UINT16_C(11317),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(41360), UINT16_C(19187), UINT16_C(19181), UINT16_C( 8654), UINT16_C(    0), UINT16_C(    0) } },
    { UINT16_C(54637),
      { UINT16_C(21482), UINT16_C(50752), UINT16_C(46845), UINT16_C( 5085), UINT16_C(50032), UINT16_C(22110), UINT16_C(50857), UINT16_C(33707),
        UINT16_C(41407), UINT16_C(  336), UINT16_C(26757), UINT16_C(21471), UINT16_C(16265), UINT16_C( 1607), UINT16_C(46574), UINT16_C(55771) },
      { UINT16_C( 7176), UINT16_C( 1439), UINT16_C(31954), UINT16_C(16920), UINT16_C(30272), UINT16_C(59800), UINT16_C(17213), UINT16_C(64620),
        UINT16_C(48357), UINT16_C(27389), UINT16_C(56612), UINT16_C(44477), UINT16_C( 1052), UINT16_C( 2995), UINT16_C(36793), UINT16_C(49636) },
      { UINT16_C( 7176), UINT16_C(    0), UINT16_C(31954), UINT16_C( 5085), UINT16_C(    0), UINT16_C(22110), UINT16_C(17213), UINT16_C(    0),
        UINT16_C(41407), UINT16_C(    0), UINT16_C(26757), UINT16_C(    0), UINT16_C( 1052), UINT16_C(    0), UINT16_C(36793), UINT16_C(49636) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_epu16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_epu16");
    easysimd_test_x86_assert_equal_u16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_test_x86_random_u16x16();
    easysimd__m256i r = easysimd_mm256_maskz_min_epu16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_min_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t src[8];
    const uint8_t k;
    const uint32_t a[8];
    const uint32_t b[8];
    const uint32_t r[8];
  } test_vec[] = {
    { { UINT32_C(1430017915), UINT32_C(2031056027), UINT32_C(1221287139), UINT32_C(1538839602), UINT32_C(3314493745), UINT32_C(2781537059), UINT32_C( 984244606), UINT32_C(1422410969) },
      UINT8_C(156),
      { UINT32_C(2033691140), UINT32_C( 375173305), UINT32_C(1346937980), UINT32_C(2239865948), UINT32_C( 380126771), UINT32_C(1905544464), UINT32_C(1011535863), UINT32_C(2631442327) },
      { UINT32_C(  34934601), UINT32_C(1008234944), UINT32_C(1921802517), UINT32_C( 955714821), UINT32_C(1666097235), UINT32_C(3855934189), UINT32_C(1226907569), UINT32_C( 132512190) },
      { UINT32_C(1430017915), UINT32_C(2031056027), UINT32_C(1346937980), UINT32_C( 955714821), UINT32_C( 380126771), UINT32_C(2781537059), UINT32_C( 984244606), UINT32_C( 132512190) } },
    { { UINT32_C(3356097032), UINT32_C(2147754603), UINT32_C(2297598083), UINT32_C(4039174813), UINT32_C(2001931914), UINT32_C(2723948784), UINT32_C(  99319111), UINT32_C(2131546230) },
      UINT8_C(202),
      { UINT32_C( 926238485), UINT32_C(3703223628), UINT32_C(2457485992), UINT32_C( 270297602), UINT32_C(3858863038), UINT32_C(1831707632), UINT32_C(1592013454), UINT32_C(1395155774) },
      { UINT32_C(4136328618), UINT32_C(3134407954), UINT32_C(2320256392), UINT32_C(1973119159), UINT32_C(3965426940), UINT32_C(3445196863), UINT32_C(4163583418), UINT32_C(1263293344) },
      { UINT32_C(3356097032), UINT32_C(3134407954), UINT32_C(2297598083), UINT32_C( 270297602), UINT32_C(2001931914), UINT32_C(2723948784), UINT32_C(1592013454), UINT32_C(1263293344) } },
    { { UINT32_C(3259094960), UINT32_C(2759660572), UINT32_C( 422562145), UINT32_C( 781109810), UINT32_C(2770004582), UINT32_C( 762475378), UINT32_C(1361419697), UINT32_C(2694607344) },
      UINT8_C( 72),
      { UINT32_C(4066730718), UINT32_C(2840857055), UINT32_C(  47934776), UINT32_C(3848800763), UINT32_C(2522352931), UINT32_C( 474449279), UINT32_C( 470587818), UINT32_C( 325364789) },
      { UINT32_C(3993422095), UINT32_C( 194468563), UINT32_C(3255726791), UINT32_C(2661840507), UINT32_C(  53805188), UINT32_C( 790658181), UINT32_C(1263217685), UINT32_C(3898519769) },
      { UINT32_C(3259094960), UINT32_C(2759660572), UINT32_C( 422562145), UINT32_C(2661840507), UINT32_C(2770004582), UINT32_C( 762475378), UINT32_C( 470587818), UINT32_C(2694607344) } },
    { { UINT32_C(1289118841), UINT32_C(2237165246), UINT32_C(1548248800), UINT32_C(1627058396), UINT32_C(1969500144), UINT32_C(3248784556), UINT32_C(2299326640), UINT32_C( 426863520) },
      UINT8_C(207),
      { UINT32_C(3062785608), UINT32_C( 613815230), UINT32_C(1258353243), UINT32_C( 473653741), UINT32_C(1237889221), UINT32_C(1173981781), UINT32_C(  31818646), UINT32_C(1020395252) },
      { UINT32_C( 586309476), UINT32_C(3460729202), UINT32_C(1746487163), UINT32_C(1837454760), UINT32_C(1521896709), UINT32_C(1855958999), UINT32_C( 661619762), UINT32_C(3882041475) },
      { UINT32_C( 586309476), UINT32_C( 613815230), UINT32_C(1258353243), UINT32_C( 473653741), UINT32_C(1969500144), UINT32_C(3248784556), UINT32_C(  31818646), UINT32_C(1020395252) } },
    { { UINT32_C( 302667423), UINT32_C(1524650207), UINT32_C(1069808023), UINT32_C(1420576846), UINT32_C(1840145045), UINT32_C(1155223058), UINT32_C(1433094866), UINT32_C( 708693899) },
      UINT8_C( 37),
      { UINT32_C(2533637191), UINT32_C( 372203036), UINT32_C(1768189473), UINT32_C(2113910811), UINT32_C(3046075495), UINT32_C(2441598023), UINT32_C( 236772671), UINT32_C(1630750490) },
      { UINT32_C(2700621699), UINT32_C(3082168214), UINT32_C(2971736726), UINT32_C( 976101587), UINT32_C(3555704460), UINT32_C(3513022098), UINT32_C(1843429715), UINT32_C(1271796680) },
      { UINT32_C(2533637191), UINT32_C(1524650207), UINT32_C(1768189473), UINT32_C(1420576846), UINT32_C(1840145045), UINT32_C(2441598023), UINT32_C(1433094866), UINT32_C( 708693899) } },
    { { UINT32_C(3790325579), UINT32_C(2224595438), UINT32_C(2402662844), UINT32_C(1707697369), UINT32_C(3023616034), UINT32_C(2189794606), UINT32_C(3874448670), UINT32_C(3291594361) },
      UINT8_C(133),
      { UINT32_C(3195249949), UINT32_C(4152031293), UINT32_C(2446330157), UINT32_C(2327000786), UINT32_C( 196700014), UINT32_C(1378434029), UINT32_C(3905621802), UINT32_C(1584238401) },
      { UINT32_C(1914495284), UINT32_C( 107583449), UINT32_C(1939356064), UINT32_C(3741141871), UINT32_C(2699671219), UINT32_C( 485626865), UINT32_C(1661255202), UINT32_C(2193715789) },
      { UINT32_C(1914495284), UINT32_C(2224595438), UINT32_C(1939356064), UINT32_C(1707697369), UINT32_C(3023616034), UINT32_C(2189794606), UINT32_C(3874448670), UINT32_C(1584238401) } },
    { { UINT32_C( 754245203), UINT32_C( 355622261), UINT32_C( 109628054), UINT32_C(3387196950), UINT32_C( 778685756), UINT32_C(  55204832), UINT32_C(1734757913), UINT32_C( 334047424) },
      UINT8_C(  6),
      { UINT32_C( 981155805), UINT32_C(1020301426), UINT32_C(2673006105), UINT32_C(2296060858), UINT32_C(3764914564), UINT32_C(2734254931), UINT32_C(4200751314), UINT32_C( 637564489) },
      { UINT32_C( 660634549), UINT32_C( 610545931), UINT32_C(3267606023), UINT32_C(1447731154), UINT32_C(4231443112), UINT32_C(4036898846), UINT32_C(3672768656), UINT32_C( 738257526) },
      { UINT32_C( 754245203), UINT32_C( 610545931), UINT32_C(2673006105), UINT32_C(3387196950), UINT32_C( 778685756), UINT32_C(  55204832), UINT32_C(1734757913), UINT32_C( 334047424) } },
    { { UINT32_C(1884512613), UINT32_C(2576725906), UINT32_C(1079728238), UINT32_C(2694227447), UINT32_C(1989987672), UINT32_C(2372287229), UINT32_C(2976337978), UINT32_C(2682087482) },
      UINT8_C(201),
      { UINT32_C(3898281776), UINT32_C(4250334372), UINT32_C(4126447184), UINT32_C(4182610988), UINT32_C(1794556720), UINT32_C(2040890409), UINT32_C(1404262123), UINT32_C(1662800435) },
      { UINT32_C( 105609058), UINT32_C(3154354540), UINT32_C(1689385015), UINT32_C(3177054092), UINT32_C(3962000578), UINT32_C(3278228696), UINT32_C(1427577122), UINT32_C(3451466603) },
      { UINT32_C( 105609058), UINT32_C(2576725906), UINT32_C(1079728238), UINT32_C(3177054092), UINT32_C(1989987672), UINT32_C(2372287229), UINT32_C(1404262123), UINT32_C(1662800435) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_epu32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_epu32");
    easysimd_test_x86_assert_equal_u32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_u32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_test_x86_random_u32x8();
    easysimd__m256i r = easysimd_mm256_mask_min_epu32(src, k, a, b);

    easysimd_test_x86_write_u32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_min_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint32_t a[8];
    const uint32_t b[8];
    const uint32_t r[8];
  } test_vec[] = {
    { UINT8_C(171),
      { UINT32_C(   8242819), UINT32_C(1430307038), UINT32_C(2626824536), UINT32_C(1367445141), UINT32_C(1752624011), UINT32_C(2894406568), UINT32_C(1717997783), UINT32_C(4145096564) },
      { UINT32_C(3438776046), UINT32_C(2803971918), UINT32_C(4114854752), UINT32_C(3444032577), UINT32_C(1463139759), UINT32_C(3070474976), UINT32_C(3189598538), UINT32_C(2142645905) },
      { UINT32_C(   8242819), UINT32_C(1430307038), UINT32_C(         0), UINT32_C(1367445141), UINT32_C(         0), UINT32_C(2894406568), UINT32_C(         0), UINT32_C(2142645905) } },
    { UINT8_C(189),
      { UINT32_C(3842722733), UINT32_C( 541438573), UINT32_C(3110222837), UINT32_C(1063792514), UINT32_C( 522174308), UINT32_C( 745134019), UINT32_C( 599599348), UINT32_C(2346728670) },
      { UINT32_C(4117818248), UINT32_C(2467673758), UINT32_C(1934456817), UINT32_C( 196261286), UINT32_C( 942330229), UINT32_C(2640614312), UINT32_C(2579505851), UINT32_C(3861225566) },
      { UINT32_C(3842722733), UINT32_C(         0), UINT32_C(1934456817), UINT32_C( 196261286), UINT32_C( 522174308), UINT32_C( 745134019), UINT32_C(         0), UINT32_C(2346728670) } },
    { UINT8_C(139),
      { UINT32_C(1261034389), UINT32_C(1748811249), UINT32_C(3205476106), UINT32_C( 859052641), UINT32_C(3621481540), UINT32_C(4086528209), UINT32_C(3629198392), UINT32_C(3882104913) },
      { UINT32_C(  70421779), UINT32_C(1416458058), UINT32_C(2148826142), UINT32_C(3669182614), UINT32_C(2259783349), UINT32_C(1064911879), UINT32_C(3256404593), UINT32_C( 380206082) },
      { UINT32_C(  70421779), UINT32_C(1416458058), UINT32_C(         0), UINT32_C( 859052641), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C( 380206082) } },
    { UINT8_C(  9),
      { UINT32_C(1263803100), UINT32_C(  57256071), UINT32_C(  93972924), UINT32_C( 733639580), UINT32_C(1764900901), UINT32_C(2212131257), UINT32_C(  92708233), UINT32_C( 571448390) },
      { UINT32_C(1047356342), UINT32_C(3359758091), UINT32_C(1573772224), UINT32_C(1938327374), UINT32_C(2162014919), UINT32_C(3036919595), UINT32_C(2612693332), UINT32_C(3703425317) },
      { UINT32_C(1047356342), UINT32_C(         0), UINT32_C(         0), UINT32_C( 733639580), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C( 44),
      { UINT32_C(  37231147), UINT32_C( 918683739), UINT32_C(1418010573), UINT32_C(1629223079), UINT32_C(2374867925), UINT32_C( 685851294), UINT32_C(3326967036), UINT32_C(1710369082) },
      { UINT32_C(2674338371), UINT32_C(4157942058), UINT32_C(4031470153), UINT32_C( 676488787), UINT32_C(2696273665), UINT32_C( 499685153), UINT32_C(1306727699), UINT32_C(2192758335) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(1418010573), UINT32_C( 676488787), UINT32_C(         0), UINT32_C( 499685153), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C(  0),
      { UINT32_C(1126899993), UINT32_C(1368138487), UINT32_C(3567549550), UINT32_C(2916535758), UINT32_C( 432961154), UINT32_C(1395518526), UINT32_C(2777840335), UINT32_C(1185289517) },
      { UINT32_C( 764006710), UINT32_C(1635718643), UINT32_C(1614160786), UINT32_C(1930300656), UINT32_C(3230456962), UINT32_C(2551429576), UINT32_C(1631430196), UINT32_C(4054311867) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C(180),
      { UINT32_C(1185423153), UINT32_C(3252160926), UINT32_C(1269971263), UINT32_C( 600646983), UINT32_C(1810664881), UINT32_C(1201636513), UINT32_C(2768371905), UINT32_C(3629773735) },
      { UINT32_C(2954821906), UINT32_C(1249048331), UINT32_C(2006328368), UINT32_C(4204487497), UINT32_C(2456127473), UINT32_C(3436774411), UINT32_C(2893142788), UINT32_C(3783576526) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(1269971263), UINT32_C(         0), UINT32_C(1810664881), UINT32_C(1201636513), UINT32_C(         0), UINT32_C(3629773735) } },
    { UINT8_C(204),
      { UINT32_C(2614596003), UINT32_C( 667623939), UINT32_C( 460342200), UINT32_C(1695312862), UINT32_C(3580927696), UINT32_C(1389968503), UINT32_C(2032240046), UINT32_C(2906980874) },
      { UINT32_C(2538085779), UINT32_C(4156429375), UINT32_C( 890384215), UINT32_C(1788485530), UINT32_C( 893323965), UINT32_C(4102494534), UINT32_C(2825758878), UINT32_C(1045869482) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C( 460342200), UINT32_C(1695312862), UINT32_C(         0), UINT32_C(         0), UINT32_C(2032240046), UINT32_C(1045869482) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_epu32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_epu32");
    easysimd_test_x86_assert_equal_u32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_test_x86_random_u32x8();
    easysimd__m256i r = easysimd_mm256_maskz_min_epu32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_min_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[4];
    const uint64_t b[4];
    const uint64_t r[4];
  } test_vec[] = {
    { { UINT64_C(14136954648658228387), UINT64_C(14122769197619336272), UINT64_C( 3566256324707667062), UINT64_C( 6535772775597268132) },
      { UINT64_C(11929385965673629550), UINT64_C( 1660447968594590342), UINT64_C(14074056832479608036), UINT64_C(11458728863188743925) },
      { UINT64_C(11929385965673629550), UINT64_C( 1660447968594590342), UINT64_C( 3566256324707667062), UINT64_C( 6535772775597268132) } },
    { { UINT64_C(13715418239535122268), UINT64_C( 7793218051469897582), UINT64_C(15001734293276639822), UINT64_C(  226166958485868883) },
      { UINT64_C(17149437447712836104), UINT64_C( 7308537451528390009), UINT64_C( 3067235942614104550), UINT64_C(10862068287235034388) },
      { UINT64_C(13715418239535122268), UINT64_C( 7308537451528390009), UINT64_C( 3067235942614104550), UINT64_C(  226166958485868883) } },
    { { UINT64_C(17710640262563656027), UINT64_C(14940801620256409051), UINT64_C( 4709676887839943260), UINT64_C( 3986734448589343279) },
      { UINT64_C(15752423813684472362), UINT64_C( 5789137578188706206), UINT64_C( 4232784460929755215), UINT64_C(11311999320367544809) },
      { UINT64_C(15752423813684472362), UINT64_C( 5789137578188706206), UINT64_C( 4232784460929755215), UINT64_C( 3986734448589343279) } },
    { { UINT64_C( 9606140516362227061), UINT64_C(10426506319296797806), UINT64_C(18271325876933965132), UINT64_C( 5229451934901110915) },
      { UINT64_C( 2321618470057071820), UINT64_C(17724256810212476199), UINT64_C( 6087983584328287850), UINT64_C(14519565604266524260) },
      { UINT64_C( 2321618470057071820), UINT64_C(10426506319296797806), UINT64_C( 6087983584328287850), UINT64_C( 5229451934901110915) } },
    { { UINT64_C( 8035182903870373040), UINT64_C(11932797642265265658), UINT64_C( 2809015546188981718), UINT64_C(15034384589258573022) },
      { UINT64_C(14974386114565120025), UINT64_C( 5663852091218765754), UINT64_C( 4456576120803147504), UINT64_C( 3108817529210418484) },
      { UINT64_C( 8035182903870373040), UINT64_C( 5663852091218765754), UINT64_C( 2809015546188981718), UINT64_C( 3108817529210418484) } },
    { { UINT64_C(14231756403265787613), UINT64_C(17350211020697302018), UINT64_C(16262070189835861267), UINT64_C( 7056395717399988143) },
      { UINT64_C(15528857735366212017), UINT64_C(11707416810601134985), UINT64_C( 4928559314631924694), UINT64_C(11993073666033621897) },
      { UINT64_C(14231756403265787613), UINT64_C(11707416810601134985), UINT64_C( 4928559314631924694), UINT64_C( 7056395717399988143) } },
    { { UINT64_C(11680898276871275373), UINT64_C( 6579939949647981342), UINT64_C(12724995682853676223), UINT64_C(   34671535088115679) },
      { UINT64_C(16098782904859221424), UINT64_C(17918096930165855688), UINT64_C(12329419595982866498), UINT64_C(13049173345105336452) },
      { UINT64_C(11680898276871275373), UINT64_C( 6579939949647981342), UINT64_C(12329419595982866498), UINT64_C(   34671535088115679) } },
    { { UINT64_C( 9257404909311843032), UINT64_C( 2787577419073914493), UINT64_C( 7310937456848731390), UINT64_C(14356512369129878914) },
      { UINT64_C( 2244439277601602030), UINT64_C( 7559720138821792735), UINT64_C(11945701406548326785), UINT64_C(14219105198494466973) },
      { UINT64_C( 2244439277601602030), UINT64_C( 2787577419073914493), UINT64_C( 7310937456848731390), UINT64_C(14219105198494466973) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_min_epu64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_min_epu64");
    easysimd_test_x86_assert_equal_u64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_test_x86_random_u64x4();
    easysimd__m256i r = easysimd_mm256_min_epu64(a, b);

    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_min_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t src[4];
    const uint8_t k;
    const uint64_t a[4];
    const uint64_t b[4];
    const uint64_t r[4];
  } test_vec[] = {
    { { UINT64_C(15984356610832336043), UINT64_C( 5051243843610051791), UINT64_C( 3426842451974963443), UINT64_C( 9577255184787416496) },
      UINT8_C( 66),
      { UINT64_C(17466303433113377725), UINT64_C( 1916770759409582087), UINT64_C(10968049546418414830), UINT64_C(17968506247480169179) },
      { UINT64_C( 1365138291104302224), UINT64_C( 1271794257416972211), UINT64_C(18393488285970648502), UINT64_C( 4130774653450433310) },
      { UINT64_C(15984356610832336043), UINT64_C( 1271794257416972211), UINT64_C( 3426842451974963443), UINT64_C( 9577255184787416496) } },
    { { UINT64_C( 9715999235605324217), UINT64_C(10936070169890623120), UINT64_C( 5019967841546295943), UINT64_C(17444959482703495990) },
      UINT8_C(211),
      { UINT64_C( 1876080801136227920), UINT64_C(15817084448966163520), UINT64_C(11136073489478462674), UINT64_C(15253063579252976359) },
      { UINT64_C(13988557029745906810), UINT64_C( 9254402104305130514), UINT64_C(11446286268105060457), UINT64_C(16942582962434125002) },
      { UINT64_C( 1876080801136227920), UINT64_C( 9254402104305130514), UINT64_C( 5019967841546295943), UINT64_C(17444959482703495990) } },
    { { UINT64_C(15401205454343335033), UINT64_C( 2188141070547187758), UINT64_C( 6745327450644689478), UINT64_C(12844207009866323171) },
      UINT8_C( 55),
      { UINT64_C(16244992282409413680), UINT64_C(15948008976727625270), UINT64_C(   13175363201945619), UINT64_C(17472536919626318456) },
      { UINT64_C( 8710707431689515927), UINT64_C( 4791222173836868084), UINT64_C( 3752549326691101991), UINT64_C(14984022658049854526) },
      { UINT64_C( 8710707431689515927), UINT64_C( 4791222173836868084), UINT64_C(   13175363201945619), UINT64_C(12844207009866323171) } },
    { { UINT64_C(13916036870423127652), UINT64_C(11789261070934320170), UINT64_C( 2430759502479763659), UINT64_C( 9735957912288834070) },
      UINT8_C( 57),
      { UINT64_C(17422394294660378348), UINT64_C(16488258485992825128), UINT64_C( 5091012703326591430), UINT64_C(18327099689929405450) },
      { UINT64_C( 5502382948031945758), UINT64_C(11396117893328587142), UINT64_C(16417826092727857021), UINT64_C( 5753724771070132425) },
      { UINT64_C( 5502382948031945758), UINT64_C(11789261070934320170), UINT64_C( 2430759502479763659), UINT64_C( 5753724771070132425) } },
    { { UINT64_C( 6854786332195780018), UINT64_C( 1609977049594885344), UINT64_C( 2626583038932780728), UINT64_C( 8881336827017917375) },
      UINT8_C( 77),
      { UINT64_C(16641792872553075195), UINT64_C( 9919623498429344808), UINT64_C( 2412532667173321276), UINT64_C(11974621347640404744) },
      { UINT64_C(  292924037371876791), UINT64_C( 6982161775818346328), UINT64_C(15828067820859345369), UINT64_C(12633981208485466626) },
      { UINT64_C(  292924037371876791), UINT64_C( 1609977049594885344), UINT64_C( 2412532667173321276), UINT64_C(11974621347640404744) } },
    { { UINT64_C( 1234243279161724224), UINT64_C(10375258585816700536), UINT64_C(12883037871892994075), UINT64_C(12680364615903774033) },
      UINT8_C(194),
      { UINT64_C(  158626459785350098), UINT64_C(16458203401251508218), UINT64_C(11776426071267956062), UINT64_C( 9375582586959804325) },
      { UINT64_C( 2196754020221949838), UINT64_C(15848117048244319551), UINT64_C( 7919974430077911591), UINT64_C( 5571484626981485598) },
      { UINT64_C( 1234243279161724224), UINT64_C(15848117048244319551), UINT64_C(12883037871892994075), UINT64_C(12680364615903774033) } },
    { { UINT64_C( 4521125841102933627), UINT64_C( 2508849117930268792), UINT64_C( 5688135598173899195), UINT64_C( 1610720329445718094) },
      UINT8_C(252),
      { UINT64_C(11303253811843020321), UINT64_C( 7887695910264099459), UINT64_C(11727632821850601761), UINT64_C( 5220045971546557703) },
      { UINT64_C( 2889774554311126026), UINT64_C(10046785755061465436), UINT64_C(18221774052874127956), UINT64_C( 8934264738765938060) },
      { UINT64_C( 4521125841102933627), UINT64_C( 2508849117930268792), UINT64_C(11727632821850601761), UINT64_C( 5220045971546557703) } },
    { { UINT64_C(10312139460157483342), UINT64_C(12662992245869183173), UINT64_C(  358262089882384646), UINT64_C(11157583742388103658) },
      UINT8_C( 92),
      { UINT64_C(12369734149677747921), UINT64_C( 5742570658593168046), UINT64_C( 1034362975840349602), UINT64_C( 6815060647992800899) },
      { UINT64_C(17355446077742064259), UINT64_C( 4084286120787697864), UINT64_C(14171986083262789113), UINT64_C( 9269690047680612280) },
      { UINT64_C(10312139460157483342), UINT64_C(12662992245869183173), UINT64_C( 1034362975840349602), UINT64_C( 6815060647992800899) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_epu64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_epu64");
    easysimd_test_x86_assert_equal_u64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_u64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_test_x86_random_u64x4();
    easysimd__m256i r = easysimd_mm256_mask_min_epu64(src, k, a, b);

    easysimd_test_x86_write_u64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_min_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint64_t a[4];
    const uint64_t b[4];
    const uint64_t r[4];
  } test_vec[] = {
    { UINT8_C(208),
      { UINT64_C(13981713749565756830), UINT64_C(16354197353321479704), UINT64_C( 6538696355557747206), UINT64_C( 6561255083387152011) },
      { UINT64_C( 4282104262660267287), UINT64_C(10885315272113375829), UINT64_C(16329336841573972370), UINT64_C(18226963676238226139) },
      { UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C( 75),
      { UINT64_C(16027652552639031040), UINT64_C(16507389917415840540), UINT64_C(11028786584542272190), UINT64_C(11148514201947776574) },
      { UINT64_C(16939197969460176161), UINT64_C( 5209123847972815801), UINT64_C(16316975365534484192), UINT64_C( 9972101426828996400) },
      { UINT64_C(16027652552639031040), UINT64_C( 5209123847972815801), UINT64_C(                   0), UINT64_C( 9972101426828996400) } },
    { UINT8_C( 42),
      { UINT64_C( 6335970747046477589), UINT64_C( 2502077921139708499), UINT64_C(17321028889916903589), UINT64_C(11756052605562333942) },
      { UINT64_C( 6110756879491079843), UINT64_C( 9139746316500184855), UINT64_C( 4353026206370536823), UINT64_C( 5289862208194855480) },
      { UINT64_C(                   0), UINT64_C( 2502077921139708499), UINT64_C(                   0), UINT64_C( 5289862208194855480) } },
    { UINT8_C(123),
      { UINT64_C( 4331848820229717825), UINT64_C( 4998032040081609214), UINT64_C(   38217088485741633), UINT64_C(17930782537217887742) },
      { UINT64_C( 2483483557662446254), UINT64_C( 7051057158192336109), UINT64_C( 2987355687139501111), UINT64_C(13273614476772645766) },
      { UINT64_C( 2483483557662446254), UINT64_C( 4998032040081609214), UINT64_C(                   0), UINT64_C(13273614476772645766) } },
    { UINT8_C(172),
      { UINT64_C(17887156887729796147), UINT64_C( 1122374504741916921), UINT64_C(11018447353525438506), UINT64_C(10298988370367489767) },
      { UINT64_C(12155805487259761999), UINT64_C( 7972101710690962152), UINT64_C( 2789621799465376928), UINT64_C(  272842385500101088) },
      { UINT64_C(                   0), UINT64_C(                   0), UINT64_C( 2789621799465376928), UINT64_C(  272842385500101088) } },
    { UINT8_C( 18),
      { UINT64_C( 4226182332803038630), UINT64_C(13233807555014048110), UINT64_C(15301292525620741566), UINT64_C(  354248760178021763) },
      { UINT64_C( 9580116374378949449), UINT64_C( 2104642687080231586), UINT64_C(14719716626057800950), UINT64_C(11810846543796555653) },
      { UINT64_C(                   0), UINT64_C( 2104642687080231586), UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C( 57),
      { UINT64_C(14030634029994171562), UINT64_C(10774086306647585885), UINT64_C(  957765057354069973), UINT64_C(13840981218313055566) },
      { UINT64_C(10770790856233542759), UINT64_C(12591125408350771901), UINT64_C(10587044927545514657), UINT64_C(14321592354307199978) },
      { UINT64_C(10770790856233542759), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(13840981218313055566) } },
    { UINT8_C(232),
      { UINT64_C(10509912753372691859), UINT64_C( 6350947106033369561), UINT64_C( 4770209130782458507), UINT64_C( 9658645545293460382) },
      { UINT64_C(13289136100391201508), UINT64_C( 8905409694789261183), UINT64_C(   41084773728465761), UINT64_C( 8981072213237183017) },
      { UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0), UINT64_C( 8981072213237183017) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_epu64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_epu64");
    easysimd_test_x86_assert_equal_u64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_test_x86_random_u64x4();
    easysimd__m256i r = easysimd_mm256_maskz_min_epu64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_min_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(249),
      { EASYSIMD_FLOAT32_C(    36.85), EASYSIMD_FLOAT32_C(   137.63), EASYSIMD_FLOAT32_C(   500.99), EASYSIMD_FLOAT32_C(  -996.63),
        EASYSIMD_FLOAT32_C(   210.26), EASYSIMD_FLOAT32_C(  -355.36), EASYSIMD_FLOAT32_C(  -834.32), EASYSIMD_FLOAT32_C(   515.42) },
      { EASYSIMD_FLOAT32_C(   767.88), EASYSIMD_FLOAT32_C(  -775.80), EASYSIMD_FLOAT32_C(   396.61), EASYSIMD_FLOAT32_C(  -158.26),
        EASYSIMD_FLOAT32_C(  -359.00), EASYSIMD_FLOAT32_C(  -498.70), EASYSIMD_FLOAT32_C(   276.13), EASYSIMD_FLOAT32_C(  -542.07) },
      { EASYSIMD_FLOAT32_C(    36.85), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -996.63),
        EASYSIMD_FLOAT32_C(  -359.00), EASYSIMD_FLOAT32_C(  -498.70), EASYSIMD_FLOAT32_C(  -834.32), EASYSIMD_FLOAT32_C(  -542.07) } },
    { UINT8_C(123),
      { EASYSIMD_FLOAT32_C(  -998.54), EASYSIMD_FLOAT32_C(   206.26), EASYSIMD_FLOAT32_C(  -508.93), EASYSIMD_FLOAT32_C(  -275.50),
        EASYSIMD_FLOAT32_C(   618.54), EASYSIMD_FLOAT32_C(  -213.39), EASYSIMD_FLOAT32_C(  -541.88), EASYSIMD_FLOAT32_C(  -538.30) },
      { EASYSIMD_FLOAT32_C(   934.32), EASYSIMD_FLOAT32_C(  -845.22), EASYSIMD_FLOAT32_C(   494.31), EASYSIMD_FLOAT32_C(   899.12),
        EASYSIMD_FLOAT32_C(  -203.04), EASYSIMD_FLOAT32_C(   392.47), EASYSIMD_FLOAT32_C(   -64.02), EASYSIMD_FLOAT32_C(   934.59) },
      { EASYSIMD_FLOAT32_C(  -998.54), EASYSIMD_FLOAT32_C(  -845.22), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -275.50),
        EASYSIMD_FLOAT32_C(  -203.04), EASYSIMD_FLOAT32_C(  -213.39), EASYSIMD_FLOAT32_C(  -541.88), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(159),
      { EASYSIMD_FLOAT32_C(   -60.66), EASYSIMD_FLOAT32_C(   144.85), EASYSIMD_FLOAT32_C(   538.10), EASYSIMD_FLOAT32_C(   105.02),
        EASYSIMD_FLOAT32_C(  -339.73), EASYSIMD_FLOAT32_C(   305.98), EASYSIMD_FLOAT32_C(   329.21), EASYSIMD_FLOAT32_C(  -943.12) },
      { EASYSIMD_FLOAT32_C(  -852.28), EASYSIMD_FLOAT32_C(   970.22), EASYSIMD_FLOAT32_C(  -441.82), EASYSIMD_FLOAT32_C(   423.85),
        EASYSIMD_FLOAT32_C(  -571.86), EASYSIMD_FLOAT32_C(   102.64), EASYSIMD_FLOAT32_C(   425.31), EASYSIMD_FLOAT32_C(   634.40) },
      { EASYSIMD_FLOAT32_C(  -852.28), EASYSIMD_FLOAT32_C(   144.85), EASYSIMD_FLOAT32_C(  -441.82), EASYSIMD_FLOAT32_C(   105.02),
        EASYSIMD_FLOAT32_C(  -571.86), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -943.12) } },
    { UINT8_C(105),
      { EASYSIMD_FLOAT32_C(  -850.18), EASYSIMD_FLOAT32_C(   252.95), EASYSIMD_FLOAT32_C(  -619.68), EASYSIMD_FLOAT32_C(  -392.06),
        EASYSIMD_FLOAT32_C(   714.65), EASYSIMD_FLOAT32_C(  -685.36), EASYSIMD_FLOAT32_C(  -237.28), EASYSIMD_FLOAT32_C(   208.96) },
      { EASYSIMD_FLOAT32_C(  -786.24), EASYSIMD_FLOAT32_C(   559.68), EASYSIMD_FLOAT32_C(  -398.58), EASYSIMD_FLOAT32_C(   149.73),
        EASYSIMD_FLOAT32_C(   494.26), EASYSIMD_FLOAT32_C(   494.88), EASYSIMD_FLOAT32_C(  -910.92), EASYSIMD_FLOAT32_C(  -360.89) },
      { EASYSIMD_FLOAT32_C(  -850.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -392.06),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -685.36), EASYSIMD_FLOAT32_C(  -910.92), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(124),
      { EASYSIMD_FLOAT32_C(   194.10), EASYSIMD_FLOAT32_C(   299.38), EASYSIMD_FLOAT32_C(  -661.04), EASYSIMD_FLOAT32_C(  -476.69),
        EASYSIMD_FLOAT32_C(   356.26), EASYSIMD_FLOAT32_C(  -513.31), EASYSIMD_FLOAT32_C(  -506.47), EASYSIMD_FLOAT32_C(   914.44) },
      { EASYSIMD_FLOAT32_C(   910.54), EASYSIMD_FLOAT32_C(   -78.32), EASYSIMD_FLOAT32_C(    17.08), EASYSIMD_FLOAT32_C(   335.85),
        EASYSIMD_FLOAT32_C(  -443.92), EASYSIMD_FLOAT32_C(  -389.21), EASYSIMD_FLOAT32_C(   485.67), EASYSIMD_FLOAT32_C(   809.02) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -661.04), EASYSIMD_FLOAT32_C(  -476.69),
        EASYSIMD_FLOAT32_C(  -443.92), EASYSIMD_FLOAT32_C(  -513.31), EASYSIMD_FLOAT32_C(  -506.47), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(180),
      { EASYSIMD_FLOAT32_C(  -906.40), EASYSIMD_FLOAT32_C(   523.67), EASYSIMD_FLOAT32_C(   305.74), EASYSIMD_FLOAT32_C(  -143.68),
        EASYSIMD_FLOAT32_C(  -267.37), EASYSIMD_FLOAT32_C(   519.50), EASYSIMD_FLOAT32_C(  -584.01), EASYSIMD_FLOAT32_C(   334.05) },
      { EASYSIMD_FLOAT32_C(  -330.76), EASYSIMD_FLOAT32_C(   910.26), EASYSIMD_FLOAT32_C(  -171.07), EASYSIMD_FLOAT32_C(  -241.68),
        EASYSIMD_FLOAT32_C(  -450.63), EASYSIMD_FLOAT32_C(   861.91), EASYSIMD_FLOAT32_C(   952.42), EASYSIMD_FLOAT32_C(   848.75) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -171.07), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -450.63), EASYSIMD_FLOAT32_C(   519.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   334.05) } },
    { UINT8_C(109),
      { EASYSIMD_FLOAT32_C(  -524.27), EASYSIMD_FLOAT32_C(   205.00), EASYSIMD_FLOAT32_C(  -312.44), EASYSIMD_FLOAT32_C(   -30.74),
        EASYSIMD_FLOAT32_C(   119.44), EASYSIMD_FLOAT32_C(  -401.90), EASYSIMD_FLOAT32_C(   890.94), EASYSIMD_FLOAT32_C(  -863.48) },
      { EASYSIMD_FLOAT32_C(   933.95), EASYSIMD_FLOAT32_C(  -552.99), EASYSIMD_FLOAT32_C(  -252.70), EASYSIMD_FLOAT32_C(   419.62),
        EASYSIMD_FLOAT32_C(  -743.97), EASYSIMD_FLOAT32_C(   738.41), EASYSIMD_FLOAT32_C(   513.22), EASYSIMD_FLOAT32_C(   779.70) },
      { EASYSIMD_FLOAT32_C(  -524.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -312.44), EASYSIMD_FLOAT32_C(   -30.74),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -401.90), EASYSIMD_FLOAT32_C(   513.22), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 95),
      { EASYSIMD_FLOAT32_C(  -630.46), EASYSIMD_FLOAT32_C(  -487.67), EASYSIMD_FLOAT32_C(  -436.35), EASYSIMD_FLOAT32_C(  -214.47),
        EASYSIMD_FLOAT32_C(   846.38), EASYSIMD_FLOAT32_C(   232.89), EASYSIMD_FLOAT32_C(  -304.21), EASYSIMD_FLOAT32_C(  -324.69) },
      { EASYSIMD_FLOAT32_C(   991.21), EASYSIMD_FLOAT32_C(   245.16), EASYSIMD_FLOAT32_C(  -462.78), EASYSIMD_FLOAT32_C(   943.62),
        EASYSIMD_FLOAT32_C(    93.90), EASYSIMD_FLOAT32_C(  -261.90), EASYSIMD_FLOAT32_C(  -580.65), EASYSIMD_FLOAT32_C(  -701.10) },
      { EASYSIMD_FLOAT32_C(  -630.46), EASYSIMD_FLOAT32_C(  -487.67), EASYSIMD_FLOAT32_C(  -462.78), EASYSIMD_FLOAT32_C(  -214.47),
        EASYSIMD_FLOAT32_C(    93.90), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -580.65), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_min_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_min_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -889.51), EASYSIMD_FLOAT64_C(  -981.31), EASYSIMD_FLOAT64_C(  -808.86), EASYSIMD_FLOAT64_C(   413.94) },
      UINT8_C( 60),
      { EASYSIMD_FLOAT64_C(  -168.02), EASYSIMD_FLOAT64_C(  -942.82), EASYSIMD_FLOAT64_C(  -995.14), EASYSIMD_FLOAT64_C(  -624.86) },
      { EASYSIMD_FLOAT64_C(  -457.82), EASYSIMD_FLOAT64_C(  -876.39), EASYSIMD_FLOAT64_C(  -143.62), EASYSIMD_FLOAT64_C(   558.05) },
      { EASYSIMD_FLOAT64_C(  -889.51), EASYSIMD_FLOAT64_C(  -981.31), EASYSIMD_FLOAT64_C(  -995.14), EASYSIMD_FLOAT64_C(  -624.86) } },
    { { EASYSIMD_FLOAT64_C(   658.94), EASYSIMD_FLOAT64_C(  -521.62), EASYSIMD_FLOAT64_C(  -204.09), EASYSIMD_FLOAT64_C(  -256.96) },
      UINT8_C(143),
      { EASYSIMD_FLOAT64_C(  -575.62), EASYSIMD_FLOAT64_C(   552.08), EASYSIMD_FLOAT64_C(   172.13), EASYSIMD_FLOAT64_C(   343.65) },
      { EASYSIMD_FLOAT64_C(  -934.32), EASYSIMD_FLOAT64_C(  -958.18), EASYSIMD_FLOAT64_C(   223.89), EASYSIMD_FLOAT64_C(   -15.11) },
      { EASYSIMD_FLOAT64_C(  -934.32), EASYSIMD_FLOAT64_C(  -958.18), EASYSIMD_FLOAT64_C(   172.13), EASYSIMD_FLOAT64_C(   -15.11) } },
    { { EASYSIMD_FLOAT64_C(   735.48), EASYSIMD_FLOAT64_C(  -748.46), EASYSIMD_FLOAT64_C(   225.85), EASYSIMD_FLOAT64_C(  -407.10) },
      UINT8_C(221),
      { EASYSIMD_FLOAT64_C(   336.34), EASYSIMD_FLOAT64_C(  -388.42), EASYSIMD_FLOAT64_C(  -872.96), EASYSIMD_FLOAT64_C(  -249.72) },
      { EASYSIMD_FLOAT64_C(  -284.28), EASYSIMD_FLOAT64_C(   -40.98), EASYSIMD_FLOAT64_C(  -192.55), EASYSIMD_FLOAT64_C(  -279.42) },
      { EASYSIMD_FLOAT64_C(  -284.28), EASYSIMD_FLOAT64_C(  -748.46), EASYSIMD_FLOAT64_C(  -872.96), EASYSIMD_FLOAT64_C(  -279.42) } },
    { { EASYSIMD_FLOAT64_C(   334.16), EASYSIMD_FLOAT64_C(   349.63), EASYSIMD_FLOAT64_C(  -155.82), EASYSIMD_FLOAT64_C(  -809.46) },
      UINT8_C(  1),
      { EASYSIMD_FLOAT64_C(  -496.88), EASYSIMD_FLOAT64_C(  -331.09), EASYSIMD_FLOAT64_C(   703.60), EASYSIMD_FLOAT64_C(   246.16) },
      { EASYSIMD_FLOAT64_C(   810.27), EASYSIMD_FLOAT64_C(  -872.02), EASYSIMD_FLOAT64_C(  -201.76), EASYSIMD_FLOAT64_C(   -17.60) },
      { EASYSIMD_FLOAT64_C(  -496.88), EASYSIMD_FLOAT64_C(   349.63), EASYSIMD_FLOAT64_C(  -155.82), EASYSIMD_FLOAT64_C(  -809.46) } },
    { { EASYSIMD_FLOAT64_C(   471.63), EASYSIMD_FLOAT64_C(  -136.08), EASYSIMD_FLOAT64_C(    24.22), EASYSIMD_FLOAT64_C(  -304.47) },
      UINT8_C(115),
      { EASYSIMD_FLOAT64_C(  -240.30), EASYSIMD_FLOAT64_C(   -52.93), EASYSIMD_FLOAT64_C(    74.65), EASYSIMD_FLOAT64_C(   352.59) },
      { EASYSIMD_FLOAT64_C(  -117.03), EASYSIMD_FLOAT64_C(  -589.01), EASYSIMD_FLOAT64_C(   964.18), EASYSIMD_FLOAT64_C(    10.02) },
      { EASYSIMD_FLOAT64_C(  -240.30), EASYSIMD_FLOAT64_C(  -589.01), EASYSIMD_FLOAT64_C(    24.22), EASYSIMD_FLOAT64_C(  -304.47) } },
    { { EASYSIMD_FLOAT64_C(   161.27), EASYSIMD_FLOAT64_C(  -320.10), EASYSIMD_FLOAT64_C(   969.04), EASYSIMD_FLOAT64_C(   968.72) },
      UINT8_C( 94),
      { EASYSIMD_FLOAT64_C(   303.20), EASYSIMD_FLOAT64_C(   318.35), EASYSIMD_FLOAT64_C(  -755.34), EASYSIMD_FLOAT64_C(   493.74) },
      { EASYSIMD_FLOAT64_C(  -773.96), EASYSIMD_FLOAT64_C(  -252.22), EASYSIMD_FLOAT64_C(  -837.35), EASYSIMD_FLOAT64_C(   929.64) },
      { EASYSIMD_FLOAT64_C(   161.27), EASYSIMD_FLOAT64_C(  -252.22), EASYSIMD_FLOAT64_C(  -837.35), EASYSIMD_FLOAT64_C(   493.74) } },
    { { EASYSIMD_FLOAT64_C(   993.94), EASYSIMD_FLOAT64_C(   972.92), EASYSIMD_FLOAT64_C(  -942.38), EASYSIMD_FLOAT64_C(  -207.82) },
      UINT8_C( 96),
      { EASYSIMD_FLOAT64_C(   529.25), EASYSIMD_FLOAT64_C(   656.10), EASYSIMD_FLOAT64_C(   979.54), EASYSIMD_FLOAT64_C(  -775.22) },
      { EASYSIMD_FLOAT64_C(   504.90), EASYSIMD_FLOAT64_C(  -260.76), EASYSIMD_FLOAT64_C(   171.85), EASYSIMD_FLOAT64_C(  -420.45) },
      { EASYSIMD_FLOAT64_C(   993.94), EASYSIMD_FLOAT64_C(   972.92), EASYSIMD_FLOAT64_C(  -942.38), EASYSIMD_FLOAT64_C(  -207.82) } },
    { { EASYSIMD_FLOAT64_C(  -908.17), EASYSIMD_FLOAT64_C(  -945.18), EASYSIMD_FLOAT64_C(    -9.46), EASYSIMD_FLOAT64_C(  -943.99) },
      UINT8_C(193),
      { EASYSIMD_FLOAT64_C(  -848.19), EASYSIMD_FLOAT64_C(  -264.10), EASYSIMD_FLOAT64_C(    33.88), EASYSIMD_FLOAT64_C(  -879.47) },
      { EASYSIMD_FLOAT64_C(  -863.62), EASYSIMD_FLOAT64_C(  -662.91), EASYSIMD_FLOAT64_C(   438.88), EASYSIMD_FLOAT64_C(  -618.96) },
      { EASYSIMD_FLOAT64_C(  -863.62), EASYSIMD_FLOAT64_C(  -945.18), EASYSIMD_FLOAT64_C(    -9.46), EASYSIMD_FLOAT64_C(  -943.99) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_min_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_min_pd");
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
    easysimd__m256d r = easysimd_mm256_mask_min_pd(src, k, a, b);

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
test_easysimd_mm256_maskz_min_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(133),
      { EASYSIMD_FLOAT64_C(   355.60), EASYSIMD_FLOAT64_C(  -770.39), EASYSIMD_FLOAT64_C(   673.28), EASYSIMD_FLOAT64_C(   873.13) },
      { EASYSIMD_FLOAT64_C(  -242.45), EASYSIMD_FLOAT64_C(   106.64), EASYSIMD_FLOAT64_C(   101.12), EASYSIMD_FLOAT64_C(  -647.99) },
      { EASYSIMD_FLOAT64_C(  -242.45), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   101.12), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(232),
      { EASYSIMD_FLOAT64_C(  -654.19), EASYSIMD_FLOAT64_C(   890.81), EASYSIMD_FLOAT64_C(  -572.65), EASYSIMD_FLOAT64_C(   167.80) },
      { EASYSIMD_FLOAT64_C(  -138.13), EASYSIMD_FLOAT64_C(  -656.41), EASYSIMD_FLOAT64_C(   482.93), EASYSIMD_FLOAT64_C(    29.15) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    29.15) } },
    { UINT8_C( 57),
      { EASYSIMD_FLOAT64_C(  -563.34), EASYSIMD_FLOAT64_C(   764.24), EASYSIMD_FLOAT64_C(   -59.63), EASYSIMD_FLOAT64_C(   154.18) },
      { EASYSIMD_FLOAT64_C(   533.12), EASYSIMD_FLOAT64_C(  -193.89), EASYSIMD_FLOAT64_C(  -276.56), EASYSIMD_FLOAT64_C(   733.32) },
      { EASYSIMD_FLOAT64_C(  -563.34), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   154.18) } },
    { UINT8_C(251),
      { EASYSIMD_FLOAT64_C(   598.19), EASYSIMD_FLOAT64_C(  -803.44), EASYSIMD_FLOAT64_C(  -735.53), EASYSIMD_FLOAT64_C(   877.74) },
      { EASYSIMD_FLOAT64_C(   552.16), EASYSIMD_FLOAT64_C(  -505.92), EASYSIMD_FLOAT64_C(   551.02), EASYSIMD_FLOAT64_C(   425.29) },
      { EASYSIMD_FLOAT64_C(   552.16), EASYSIMD_FLOAT64_C(  -803.44), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   425.29) } },
    { UINT8_C( 59),
      { EASYSIMD_FLOAT64_C(  -342.34), EASYSIMD_FLOAT64_C(  -473.59), EASYSIMD_FLOAT64_C(   603.64), EASYSIMD_FLOAT64_C(  -781.22) },
      { EASYSIMD_FLOAT64_C(  -127.78), EASYSIMD_FLOAT64_C(   494.46), EASYSIMD_FLOAT64_C(  -353.87), EASYSIMD_FLOAT64_C(  -959.99) },
      { EASYSIMD_FLOAT64_C(  -342.34), EASYSIMD_FLOAT64_C(  -473.59), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -959.99) } },
    { UINT8_C(192),
      { EASYSIMD_FLOAT64_C(   -10.28), EASYSIMD_FLOAT64_C(   522.94), EASYSIMD_FLOAT64_C(   385.47), EASYSIMD_FLOAT64_C(  -314.76) },
      { EASYSIMD_FLOAT64_C(   959.60), EASYSIMD_FLOAT64_C(   149.71), EASYSIMD_FLOAT64_C(   625.61), EASYSIMD_FLOAT64_C(   113.79) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(242),
      { EASYSIMD_FLOAT64_C(  -568.28), EASYSIMD_FLOAT64_C(   837.23), EASYSIMD_FLOAT64_C(  -583.85), EASYSIMD_FLOAT64_C(  -357.14) },
      { EASYSIMD_FLOAT64_C(   435.42), EASYSIMD_FLOAT64_C(  -387.28), EASYSIMD_FLOAT64_C(   -92.67), EASYSIMD_FLOAT64_C(   313.16) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -387.28), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 83),
      { EASYSIMD_FLOAT64_C(   401.41), EASYSIMD_FLOAT64_C(  -135.82), EASYSIMD_FLOAT64_C(   590.16), EASYSIMD_FLOAT64_C(  -346.96) },
      { EASYSIMD_FLOAT64_C(   521.84), EASYSIMD_FLOAT64_C(  -883.44), EASYSIMD_FLOAT64_C(  -743.31), EASYSIMD_FLOAT64_C(   740.62) },
      { EASYSIMD_FLOAT64_C(   401.41), EASYSIMD_FLOAT64_C(  -883.44), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_min_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_min_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_min_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_min_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int8_t a[64];
    const int8_t b[64];
    const int8_t r[64];
  } test_vec[] = {
    { {  INT8_C(  99),  INT8_C(  57),  INT8_C(  67), -INT8_C(   9), -INT8_C(   6),  INT8_C(  33), -INT8_C( 124),  INT8_C(  36),
         INT8_C(  33),  INT8_C(  54), -INT8_C(  88), -INT8_C(  42), -INT8_C(   2),  INT8_C( 100), -INT8_C(  20), -INT8_C(  26),
         INT8_C(  12),  INT8_C(  68), -INT8_C(  19),  INT8_C(   5),  INT8_C(  93), -INT8_C(  21),      INT8_MAX,  INT8_C( 103),
         INT8_C( 108),  INT8_C(  29),  INT8_C(  35), -INT8_C(  11),  INT8_C(  48),  INT8_C(  37), -INT8_C(  11), -INT8_C( 108),
         INT8_C(  94),  INT8_C(  56), -INT8_C( 117),  INT8_C(  88),  INT8_C(  89),  INT8_C(  15),  INT8_C( 124),  INT8_C( 122),
         INT8_C(  69),  INT8_C(  37),  INT8_C(  80),  INT8_C(  67), -INT8_C( 119),  INT8_C(  60),  INT8_C(  42), -INT8_C( 107),
             INT8_MIN,  INT8_C(  23), -INT8_C( 102), -INT8_C(  34),  INT8_C(   2),  INT8_C(  26),  INT8_C(  69),  INT8_C( 111),
         INT8_C(  55),  INT8_C( 104),  INT8_C( 100),  INT8_C( 104), -INT8_C( 114),  INT8_C(  89), -INT8_C(   4), -INT8_C(  20) },
      { -INT8_C( 111), -INT8_C( 121),  INT8_C(  69), -INT8_C(  22), -INT8_C( 106), -INT8_C(  63),  INT8_C( 101), -INT8_C(  37),
        -INT8_C(  26), -INT8_C(  75),  INT8_C(  31),  INT8_C( 111), -INT8_C(  14),  INT8_C(  73),  INT8_C(   4),  INT8_C( 114),
         INT8_C(  96), -INT8_C(  97),  INT8_C(  80),  INT8_C(  98), -INT8_C(  71), -INT8_C( 106), -INT8_C(  47), -INT8_C(  16),
        -INT8_C(   2),  INT8_C(  54),  INT8_C(  88), -INT8_C( 116), -INT8_C( 113),  INT8_C(  84),  INT8_C( 121),  INT8_C(  33),
        -INT8_C(  37), -INT8_C(  66),  INT8_C(  11),  INT8_C( 113),      INT8_MAX,  INT8_C( 112),  INT8_C(  77),  INT8_C( 102),
         INT8_C(  38),  INT8_C( 108), -INT8_C(  43),  INT8_C(  24), -INT8_C(  75), -INT8_C(  38), -INT8_C( 118),  INT8_C(  21),
         INT8_C( 121), -INT8_C(  37),  INT8_C( 119),  INT8_C(  50),  INT8_C( 113),  INT8_C(  73),  INT8_C(  34),  INT8_C( 111),
             INT8_MAX,  INT8_C( 123), -INT8_C(   4),  INT8_C(  14), -INT8_C(  49),  INT8_C( 117),  INT8_C(  47), -INT8_C(  85) },
      { -INT8_C( 111), -INT8_C( 121),  INT8_C(  67), -INT8_C(  22), -INT8_C( 106), -INT8_C(  63), -INT8_C( 124), -INT8_C(  37),
        -INT8_C(  26), -INT8_C(  75), -INT8_C(  88), -INT8_C(  42), -INT8_C(  14),  INT8_C(  73), -INT8_C(  20), -INT8_C(  26),
         INT8_C(  12), -INT8_C(  97), -INT8_C(  19),  INT8_C(   5), -INT8_C(  71), -INT8_C( 106), -INT8_C(  47), -INT8_C(  16),
        -INT8_C(   2),  INT8_C(  29),  INT8_C(  35), -INT8_C( 116), -INT8_C( 113),  INT8_C(  37), -INT8_C(  11), -INT8_C( 108),
        -INT8_C(  37), -INT8_C(  66), -INT8_C( 117),  INT8_C(  88),  INT8_C(  89),  INT8_C(  15),  INT8_C(  77),  INT8_C( 102),
         INT8_C(  38),  INT8_C(  37), -INT8_C(  43),  INT8_C(  24), -INT8_C( 119), -INT8_C(  38), -INT8_C( 118), -INT8_C( 107),
             INT8_MIN, -INT8_C(  37), -INT8_C( 102), -INT8_C(  34),  INT8_C(   2),  INT8_C(  26),  INT8_C(  34),  INT8_C( 111),
         INT8_C(  55),  INT8_C( 104), -INT8_C(   4),  INT8_C(  14), -INT8_C( 114),  INT8_C(  89), -INT8_C(   4), -INT8_C(  85) } },
    { {  INT8_C(  51),  INT8_C(  59),  INT8_C(  28), -INT8_C(  78), -INT8_C(  85),  INT8_C( 105),  INT8_C(  24), -INT8_C(  47),
        -INT8_C(  43), -INT8_C(  18), -INT8_C(  23), -INT8_C( 118), -INT8_C(  56),  INT8_C( 116), -INT8_C(  97),  INT8_C(  65),
         INT8_C(  79),  INT8_C(  23),  INT8_C( 115), -INT8_C(  64),  INT8_C(  96), -INT8_C( 107),  INT8_C(  47), -INT8_C(  33),
         INT8_C(  16),  INT8_C(  43), -INT8_C(  19), -INT8_C(  32), -INT8_C(  96),  INT8_C(  29), -INT8_C( 117), -INT8_C(  45),
         INT8_C(  88), -INT8_C(  89), -INT8_C( 122),  INT8_C(   3),  INT8_C(  17), -INT8_C(  98), -INT8_C(  43), -INT8_C(  26),
        -INT8_C( 116), -INT8_C(  66),  INT8_C( 113),  INT8_C(  84),  INT8_C(  50),  INT8_C(  16), -INT8_C( 107), -INT8_C( 127),
         INT8_C(  39),  INT8_C(   8),  INT8_C(  65), -INT8_C( 121), -INT8_C(  98),  INT8_C( 113),  INT8_C( 102), -INT8_C(  82),
        -INT8_C( 100),  INT8_C(  84), -INT8_C( 114),  INT8_C(  61),  INT8_C( 113),  INT8_C(  25),  INT8_C(  16), -INT8_C(  55) },
      { -INT8_C(  63), -INT8_C( 106), -INT8_C(  52), -INT8_C(  46),  INT8_C(  53), -INT8_C(  95), -INT8_C(  72), -INT8_C(  63),
         INT8_C(  96),  INT8_C(  41),  INT8_C(  22), -INT8_C( 110),  INT8_C(  58), -INT8_C(  85),  INT8_C(  20),  INT8_C(  97),
        -INT8_C(  76),  INT8_C(  85), -INT8_C(  23),  INT8_C(  82), -INT8_C(  58),  INT8_C(  79),  INT8_C(   0),  INT8_C(  99),
        -INT8_C(  93), -INT8_C( 113), -INT8_C(  96),  INT8_C(  20), -INT8_C(  88), -INT8_C(  80), -INT8_C(  35),  INT8_C( 105),
         INT8_C(  71), -INT8_C(  86),  INT8_C(  59),  INT8_C( 124),  INT8_C(  75), -INT8_C(  12),  INT8_C(  61), -INT8_C(  85),
         INT8_C(  29),  INT8_C(  83),  INT8_C(  62),  INT8_C(  87), -INT8_C(   1),  INT8_C(  82), -INT8_C(  71), -INT8_C(  77),
        -INT8_C(  89), -INT8_C(  94),  INT8_C(   5),  INT8_C( 110), -INT8_C(  15),  INT8_C(   5), -INT8_C(  47), -INT8_C( 107),
        -INT8_C( 108),  INT8_C( 113), -INT8_C(  87),  INT8_C(  61),  INT8_C(  33), -INT8_C( 121), -INT8_C(  90),  INT8_C( 104) },
      { -INT8_C(  63), -INT8_C( 106), -INT8_C(  52), -INT8_C(  78), -INT8_C(  85), -INT8_C(  95), -INT8_C(  72), -INT8_C(  63),
        -INT8_C(  43), -INT8_C(  18), -INT8_C(  23), -INT8_C( 118), -INT8_C(  56), -INT8_C(  85), -INT8_C(  97),  INT8_C(  65),
        -INT8_C(  76),  INT8_C(  23), -INT8_C(  23), -INT8_C(  64), -INT8_C(  58), -INT8_C( 107),  INT8_C(   0), -INT8_C(  33),
        -INT8_C(  93), -INT8_C( 113), -INT8_C(  96), -INT8_C(  32), -INT8_C(  96), -INT8_C(  80), -INT8_C( 117), -INT8_C(  45),
         INT8_C(  71), -INT8_C(  89), -INT8_C( 122),  INT8_C(   3),  INT8_C(  17), -INT8_C(  98), -INT8_C(  43), -INT8_C(  85),
        -INT8_C( 116), -INT8_C(  66),  INT8_C(  62),  INT8_C(  84), -INT8_C(   1),  INT8_C(  16), -INT8_C( 107), -INT8_C( 127),
        -INT8_C(  89), -INT8_C(  94),  INT8_C(   5), -INT8_C( 121), -INT8_C(  98),  INT8_C(   5), -INT8_C(  47), -INT8_C( 107),
        -INT8_C( 108),  INT8_C(  84), -INT8_C( 114),  INT8_C(  61),  INT8_C(  33), -INT8_C( 121), -INT8_C(  90), -INT8_C(  55) } },
    { {  INT8_C(  49), -INT8_C(  30), -INT8_C(  28),  INT8_C( 124), -INT8_C(  42),  INT8_C(  34),  INT8_C(  40), -INT8_C(  13),
         INT8_C( 117),  INT8_C( 102),  INT8_C(  75),  INT8_C( 116), -INT8_C(  72),  INT8_C(   4),  INT8_C(  39),  INT8_C(  95),
        -INT8_C(  90),  INT8_C(  44), -INT8_C(  51), -INT8_C( 105),  INT8_C(  50), -INT8_C(  98),  INT8_C(  44), -INT8_C(  58),
         INT8_C(  15), -INT8_C(  42),  INT8_C(   3),  INT8_C(  49),  INT8_C(  93), -INT8_C(  86), -INT8_C( 103), -INT8_C( 114),
        -INT8_C( 116),  INT8_C( 126),  INT8_C(  10),  INT8_C(  98), -INT8_C(  96),  INT8_C(  50),  INT8_C(  85),  INT8_C(  21),
        -INT8_C( 104), -INT8_C(  96), -INT8_C( 118),  INT8_C(  80), -INT8_C(  92), -INT8_C(  79), -INT8_C(  80),  INT8_C(  74),
        -INT8_C(  34),  INT8_C( 125), -INT8_C(  30),  INT8_C(  16),  INT8_C(  28),  INT8_C(  14), -INT8_C(  42),  INT8_C(  43),
        -INT8_C(  28), -INT8_C(  38),  INT8_C(  92),  INT8_C(  65), -INT8_C( 124), -INT8_C(  10), -INT8_C(  49),  INT8_C(  16) },
      {  INT8_C( 116), -INT8_C(  38),  INT8_C( 114),  INT8_C(  20),  INT8_C(  12), -INT8_C(  57),  INT8_C(  41), -INT8_C(  91),
         INT8_C( 104), -INT8_C(  77), -INT8_C(  11),  INT8_C(  12),  INT8_C( 101), -INT8_C(  91),  INT8_C(  87),  INT8_C(  67),
         INT8_C(  35),  INT8_C(  57),  INT8_C(  83),  INT8_C(  63),  INT8_C(  71),  INT8_C(  41),  INT8_C( 106),  INT8_C(  44),
         INT8_C(   3), -INT8_C(  57),  INT8_C( 109), -INT8_C( 121), -INT8_C(  67),  INT8_C(  61), -INT8_C( 105),  INT8_C(  49),
         INT8_C(  23),  INT8_C(   9),  INT8_C(  69),  INT8_C(  35), -INT8_C(  47),  INT8_C( 110), -INT8_C(  56),  INT8_C(  57),
         INT8_C(  34), -INT8_C(  66),  INT8_C(  69), -INT8_C( 121),  INT8_C(  99), -INT8_C( 100), -INT8_C(  54), -INT8_C( 122),
        -INT8_C(  43),  INT8_C(  29), -INT8_C(  59),  INT8_C(  29),  INT8_C(  70),  INT8_C(  48),  INT8_C(  73),  INT8_C(  74),
        -INT8_C(   9), -INT8_C(  74), -INT8_C(  47), -INT8_C(  76), -INT8_C(  13),  INT8_C( 105), -INT8_C(  27),  INT8_C(  10) },
      {  INT8_C(  49), -INT8_C(  38), -INT8_C(  28),  INT8_C(  20), -INT8_C(  42), -INT8_C(  57),  INT8_C(  40), -INT8_C(  91),
         INT8_C( 104), -INT8_C(  77), -INT8_C(  11),  INT8_C(  12), -INT8_C(  72), -INT8_C(  91),  INT8_C(  39),  INT8_C(  67),
        -INT8_C(  90),  INT8_C(  44), -INT8_C(  51), -INT8_C( 105),  INT8_C(  50), -INT8_C(  98),  INT8_C(  44), -INT8_C(  58),
         INT8_C(   3), -INT8_C(  57),  INT8_C(   3), -INT8_C( 121), -INT8_C(  67), -INT8_C(  86), -INT8_C( 105), -INT8_C( 114),
        -INT8_C( 116),  INT8_C(   9),  INT8_C(  10),  INT8_C(  35), -INT8_C(  96),  INT8_C(  50), -INT8_C(  56),  INT8_C(  21),
        -INT8_C( 104), -INT8_C(  96), -INT8_C( 118), -INT8_C( 121), -INT8_C(  92), -INT8_C( 100), -INT8_C(  80), -INT8_C( 122),
        -INT8_C(  43),  INT8_C(  29), -INT8_C(  59),  INT8_C(  16),  INT8_C(  28),  INT8_C(  14), -INT8_C(  42),  INT8_C(  43),
        -INT8_C(  28), -INT8_C(  74), -INT8_C(  47), -INT8_C(  76), -INT8_C( 124), -INT8_C(  10), -INT8_C(  49),  INT8_C(  10) } },
    { {  INT8_C( 114),  INT8_C(  42),  INT8_C(  46),  INT8_C(  67), -INT8_C( 104), -INT8_C(  10),  INT8_C( 124), -INT8_C(  70),
        -INT8_C(  76), -INT8_C(  62),  INT8_C(  65),  INT8_C(  24),  INT8_C(  94),  INT8_C(  11), -INT8_C(  98),  INT8_C(  52),
         INT8_C(  40),  INT8_C( 100),  INT8_C(  81),  INT8_C( 111), -INT8_C( 108), -INT8_C( 102), -INT8_C(  71), -INT8_C( 117),
         INT8_C(  80), -INT8_C( 118),  INT8_C(  63),  INT8_C(  68), -INT8_C(  13),  INT8_C(  36),  INT8_C(  78),  INT8_C( 102),
         INT8_C(  78),  INT8_C( 124), -INT8_C(  87), -INT8_C(  26),  INT8_C( 115),  INT8_C(  38), -INT8_C(  95),  INT8_C(  39),
        -INT8_C(  24), -INT8_C(  30),  INT8_C(  63),  INT8_C(  70), -INT8_C(  18), -INT8_C(  34),  INT8_C( 122),  INT8_C(  22),
         INT8_C(  66), -INT8_C(  53), -INT8_C( 123), -INT8_C(  42),  INT8_C( 101),  INT8_C(  62),  INT8_C(  97), -INT8_C(  74),
        -INT8_C(  55), -INT8_C(  96), -INT8_C(   6), -INT8_C(  68), -INT8_C(  60),  INT8_C(  72),  INT8_C(  34),  INT8_C(  18) },
      { -INT8_C(  59), -INT8_C(  52), -INT8_C(   8),  INT8_C(  56), -INT8_C(  14), -INT8_C( 103),  INT8_C(  95), -INT8_C(  38),
         INT8_C( 124), -INT8_C(  97),  INT8_C(  32),  INT8_C( 106),  INT8_C( 125), -INT8_C( 101),      INT8_MIN, -INT8_C(  65),
         INT8_C( 102),  INT8_C(   6), -INT8_C( 107), -INT8_C(  52),  INT8_C(  68), -INT8_C(  10), -INT8_C( 126),  INT8_C(  13),
        -INT8_C( 106),  INT8_C( 124), -INT8_C(  54),  INT8_C(  90), -INT8_C(  60), -INT8_C(  20),  INT8_C( 108), -INT8_C( 119),
        -INT8_C(  72),  INT8_C( 100), -INT8_C(  63), -INT8_C(  86), -INT8_C(   2),  INT8_C(  33), -INT8_C( 124),  INT8_C( 122),
        -INT8_C(  64), -INT8_C(  91), -INT8_C(  28),  INT8_C(  61),  INT8_C(  64),  INT8_C( 100), -INT8_C(   4), -INT8_C(  90),
         INT8_C( 106), -INT8_C( 111),  INT8_C( 114), -INT8_C(  81), -INT8_C( 121), -INT8_C(  12), -INT8_C(  68),  INT8_C(  29),
         INT8_C( 112), -INT8_C( 122),  INT8_C( 119),  INT8_C(  53),  INT8_C( 115), -INT8_C(  29), -INT8_C(  66),  INT8_C(  43) },
      { -INT8_C(  59), -INT8_C(  52), -INT8_C(   8),  INT8_C(  56), -INT8_C( 104), -INT8_C( 103),  INT8_C(  95), -INT8_C(  70),
        -INT8_C(  76), -INT8_C(  97),  INT8_C(  32),  INT8_C(  24),  INT8_C(  94), -INT8_C( 101),      INT8_MIN, -INT8_C(  65),
         INT8_C(  40),  INT8_C(   6), -INT8_C( 107), -INT8_C(  52), -INT8_C( 108), -INT8_C( 102), -INT8_C( 126), -INT8_C( 117),
        -INT8_C( 106), -INT8_C( 118), -INT8_C(  54),  INT8_C(  68), -INT8_C(  60), -INT8_C(  20),  INT8_C(  78), -INT8_C( 119),
        -INT8_C(  72),  INT8_C( 100), -INT8_C(  87), -INT8_C(  86), -INT8_C(   2),  INT8_C(  33), -INT8_C( 124),  INT8_C(  39),
        -INT8_C(  64), -INT8_C(  91), -INT8_C(  28),  INT8_C(  61), -INT8_C(  18), -INT8_C(  34), -INT8_C(   4), -INT8_C(  90),
         INT8_C(  66), -INT8_C( 111), -INT8_C( 123), -INT8_C(  81), -INT8_C( 121), -INT8_C(  12), -INT8_C(  68), -INT8_C(  74),
        -INT8_C(  55), -INT8_C( 122), -INT8_C(   6), -INT8_C(  68), -INT8_C(  60), -INT8_C(  29), -INT8_C(  66),  INT8_C(  18) } },
    { {  INT8_C(  71),      INT8_MIN, -INT8_C(  42),  INT8_C(  69), -INT8_C(  95),  INT8_C(  90), -INT8_C(  65),  INT8_C(  97),
        -INT8_C(   1), -INT8_C(  93), -INT8_C(  98),  INT8_C(  63),  INT8_C(   8), -INT8_C( 102), -INT8_C(  26),  INT8_C( 114),
         INT8_C(  43),  INT8_C(  88),  INT8_C(  33), -INT8_C(  78),  INT8_C(  77), -INT8_C(  34), -INT8_C(  49), -INT8_C(  67),
         INT8_C( 100),  INT8_C(  70), -INT8_C(  14), -INT8_C(  41),  INT8_C(  41), -INT8_C(  79),  INT8_C(   3),  INT8_C( 112),
         INT8_C(  49), -INT8_C(  39), -INT8_C(  74), -INT8_C(  46),  INT8_C(  51),  INT8_C( 117),  INT8_C(  51),  INT8_C(  51),
         INT8_C(  25), -INT8_C(  47),  INT8_C( 114),  INT8_C(  33),  INT8_C( 107),  INT8_C(  88), -INT8_C( 109), -INT8_C( 106),
        -INT8_C(  79), -INT8_C(  75),  INT8_C(  72), -INT8_C(   2), -INT8_C( 109),  INT8_C(  23), -INT8_C(  69), -INT8_C(   9),
         INT8_C(  93), -INT8_C(  82), -INT8_C(  49), -INT8_C( 122),  INT8_C(  95), -INT8_C(  46), -INT8_C(  10), -INT8_C( 112) },
      { -INT8_C(  85), -INT8_C(  84),  INT8_C(  98), -INT8_C(  34),  INT8_C(  34), -INT8_C( 107),  INT8_C(  17),  INT8_C(  59),
         INT8_C( 102), -INT8_C( 124),  INT8_C(  92), -INT8_C(  47), -INT8_C(  36), -INT8_C(  17),  INT8_C( 103), -INT8_C( 115),
        -INT8_C(  92), -INT8_C(  81), -INT8_C( 117),  INT8_C(  55), -INT8_C(  58),  INT8_C(  71),  INT8_C(  47),  INT8_C(  35),
        -INT8_C(  11), -INT8_C(   2), -INT8_C(  87),  INT8_C(  84), -INT8_C(  48), -INT8_C(  97), -INT8_C(  28),  INT8_C( 123),
         INT8_C(  76),  INT8_C(  70),  INT8_C(  89),  INT8_C( 110), -INT8_C(  37),  INT8_C( 107), -INT8_C(  87),  INT8_C(  65),
        -INT8_C(  17),  INT8_C(   5),  INT8_C(  18), -INT8_C(  53), -INT8_C(  12),  INT8_C( 121),  INT8_C(  89), -INT8_C( 103),
         INT8_C(  40), -INT8_C(  28), -INT8_C(  48), -INT8_C(  18),  INT8_C(  43), -INT8_C(   1),  INT8_C(  17),  INT8_C(  32),
        -INT8_C(   3), -INT8_C(  70),  INT8_C( 116), -INT8_C(  51),  INT8_C(  89),  INT8_C(  88),  INT8_C(  72), -INT8_C(  91) },
      { -INT8_C(  85),      INT8_MIN, -INT8_C(  42), -INT8_C(  34), -INT8_C(  95), -INT8_C( 107), -INT8_C(  65),  INT8_C(  59),
        -INT8_C(   1), -INT8_C( 124), -INT8_C(  98), -INT8_C(  47), -INT8_C(  36), -INT8_C( 102), -INT8_C(  26), -INT8_C( 115),
        -INT8_C(  92), -INT8_C(  81), -INT8_C( 117), -INT8_C(  78), -INT8_C(  58), -INT8_C(  34), -INT8_C(  49), -INT8_C(  67),
        -INT8_C(  11), -INT8_C(   2), -INT8_C(  87), -INT8_C(  41), -INT8_C(  48), -INT8_C(  97), -INT8_C(  28),  INT8_C( 112),
         INT8_C(  49), -INT8_C(  39), -INT8_C(  74), -INT8_C(  46), -INT8_C(  37),  INT8_C( 107), -INT8_C(  87),  INT8_C(  51),
        -INT8_C(  17), -INT8_C(  47),  INT8_C(  18), -INT8_C(  53), -INT8_C(  12),  INT8_C(  88), -INT8_C( 109), -INT8_C( 106),
        -INT8_C(  79), -INT8_C(  75), -INT8_C(  48), -INT8_C(  18), -INT8_C( 109), -INT8_C(   1), -INT8_C(  69), -INT8_C(   9),
        -INT8_C(   3), -INT8_C(  82), -INT8_C(  49), -INT8_C( 122),  INT8_C(  89), -INT8_C(  46), -INT8_C(  10), -INT8_C( 112) } },
    { { -INT8_C(  98), -INT8_C(  94),  INT8_C(  19),  INT8_C( 121),  INT8_C(  13), -INT8_C(  68), -INT8_C(  70), -INT8_C(   4),
        -INT8_C(  63), -INT8_C(  52), -INT8_C(  57), -INT8_C(  74),  INT8_C(  69),  INT8_C(  32),  INT8_C(  79),  INT8_C( 109),
         INT8_C(   5),  INT8_C(  31),  INT8_C(  91),  INT8_C(  48),  INT8_C(  31),  INT8_C( 108),  INT8_C(  81),  INT8_C(  28),
         INT8_C(  38), -INT8_C(  59), -INT8_C(  22),      INT8_MIN,  INT8_C(  30),  INT8_C(  50),  INT8_C(  37), -INT8_C(  68),
        -INT8_C(  44),  INT8_C(  57),  INT8_C(  54), -INT8_C(  31), -INT8_C(  11), -INT8_C(  16), -INT8_C(  35), -INT8_C(  73),
        -INT8_C(  67), -INT8_C(  91),  INT8_C( 109),  INT8_C(   2), -INT8_C(  59), -INT8_C(  68),  INT8_C( 112), -INT8_C(  54),
        -INT8_C(  37), -INT8_C(  53), -INT8_C(   5), -INT8_C(   6),  INT8_C(  56),  INT8_C(  76),  INT8_C(  23),  INT8_C(  94),
         INT8_C(  17),  INT8_C(   1), -INT8_C(  34),  INT8_C(  47),  INT8_C(  51),  INT8_C(   4), -INT8_C(  20),  INT8_C(   8) },
      {  INT8_C(  61),  INT8_C(  34), -INT8_C(  23),  INT8_C(  50),  INT8_C(  18), -INT8_C(  57), -INT8_C(  23), -INT8_C(  49),
         INT8_C( 108),  INT8_C(  86), -INT8_C(  46),  INT8_C(  49),  INT8_C(  18),  INT8_C(  66), -INT8_C(   4), -INT8_C(  18),
         INT8_C(  13), -INT8_C(   9), -INT8_C(  24),  INT8_C(  69),  INT8_C(  67), -INT8_C(   1), -INT8_C(  92),  INT8_C(  84),
         INT8_C(   0), -INT8_C( 126), -INT8_C( 124),  INT8_C(  52), -INT8_C( 122),  INT8_C( 112),  INT8_C(  60), -INT8_C(  61),
        -INT8_C( 110),  INT8_C(  37), -INT8_C(  10), -INT8_C(  92), -INT8_C(  20), -INT8_C(  33),  INT8_C( 116),  INT8_C(  88),
         INT8_C(  54),  INT8_C(  70), -INT8_C( 118),  INT8_C(  72), -INT8_C( 120), -INT8_C( 122),  INT8_C(  54), -INT8_C( 107),
         INT8_C( 125),  INT8_C(  31), -INT8_C(  37), -INT8_C(  64),  INT8_C(  30),      INT8_MAX,  INT8_C(  20),  INT8_C(  31),
         INT8_C(   1), -INT8_C( 104),  INT8_C(  83), -INT8_C( 120),  INT8_C(   8), -INT8_C( 113),  INT8_C(  75), -INT8_C( 102) },
      { -INT8_C(  98), -INT8_C(  94), -INT8_C(  23),  INT8_C(  50),  INT8_C(  13), -INT8_C(  68), -INT8_C(  70), -INT8_C(  49),
        -INT8_C(  63), -INT8_C(  52), -INT8_C(  57), -INT8_C(  74),  INT8_C(  18),  INT8_C(  32), -INT8_C(   4), -INT8_C(  18),
         INT8_C(   5), -INT8_C(   9), -INT8_C(  24),  INT8_C(  48),  INT8_C(  31), -INT8_C(   1), -INT8_C(  92),  INT8_C(  28),
         INT8_C(   0), -INT8_C( 126), -INT8_C( 124),      INT8_MIN, -INT8_C( 122),  INT8_C(  50),  INT8_C(  37), -INT8_C(  68),
        -INT8_C( 110),  INT8_C(  37), -INT8_C(  10), -INT8_C(  92), -INT8_C(  20), -INT8_C(  33), -INT8_C(  35), -INT8_C(  73),
        -INT8_C(  67), -INT8_C(  91), -INT8_C( 118),  INT8_C(   2), -INT8_C( 120), -INT8_C( 122),  INT8_C(  54), -INT8_C( 107),
        -INT8_C(  37), -INT8_C(  53), -INT8_C(  37), -INT8_C(  64),  INT8_C(  30),  INT8_C(  76),  INT8_C(  20),  INT8_C(  31),
         INT8_C(   1), -INT8_C( 104), -INT8_C(  34), -INT8_C( 120),  INT8_C(   8), -INT8_C( 113), -INT8_C(  20), -INT8_C( 102) } },
    { { -INT8_C(  76),  INT8_C(  65),  INT8_C(  63), -INT8_C(  95),  INT8_C(  33), -INT8_C(  77), -INT8_C(   7),  INT8_C(  87),
        -INT8_C(   7), -INT8_C( 125), -INT8_C(  97), -INT8_C( 127),  INT8_C(   9), -INT8_C(  42),  INT8_C(  22), -INT8_C( 122),
        -INT8_C(  11), -INT8_C(  15),  INT8_C(  70),  INT8_C(  19),  INT8_C( 112),  INT8_C(  91),  INT8_C(  50),  INT8_C( 114),
        -INT8_C(  13), -INT8_C( 123), -INT8_C(   6), -INT8_C(   4),  INT8_C(  20),  INT8_C(  69), -INT8_C( 106), -INT8_C(  55),
        -INT8_C( 121), -INT8_C(  43),  INT8_C( 106), -INT8_C(  88), -INT8_C( 120),  INT8_C(  99), -INT8_C(   1), -INT8_C( 127),
        -INT8_C(  25), -INT8_C(  98),  INT8_C(   2), -INT8_C(  16),  INT8_C( 116),  INT8_C(  25),  INT8_C( 119),  INT8_C( 105),
         INT8_C(  10), -INT8_C(  67),  INT8_C( 125),  INT8_C( 123),  INT8_C(  24), -INT8_C(  81), -INT8_C(  19),  INT8_C(  12),
         INT8_C(  53), -INT8_C(  25),  INT8_C(   8),  INT8_C(  73),  INT8_C(  44), -INT8_C(  98),  INT8_C(  18), -INT8_C(  77) },
      {  INT8_C( 116),  INT8_C( 124),  INT8_C(  91), -INT8_C(   4), -INT8_C(  32),  INT8_C(  90),  INT8_C( 126), -INT8_C(  57),
        -INT8_C(   7),      INT8_MIN, -INT8_C(  73),  INT8_C( 109), -INT8_C( 103),  INT8_C(  46), -INT8_C(  41), -INT8_C(  92),
        -INT8_C(  20),  INT8_C(  84),  INT8_C(  31),  INT8_C(   4),  INT8_C(   3),  INT8_C(  12),  INT8_C(  16),  INT8_C(  56),
        -INT8_C(  13),  INT8_C(  24), -INT8_C( 126),  INT8_C(  31), -INT8_C(  73), -INT8_C( 108), -INT8_C(  45),  INT8_C(  43),
         INT8_C(  17),  INT8_C(  46),  INT8_C(  39), -INT8_C(  15), -INT8_C( 119), -INT8_C(  91), -INT8_C(  72), -INT8_C( 126),
         INT8_C(  38),  INT8_C( 111), -INT8_C(  17), -INT8_C(  65), -INT8_C(  98), -INT8_C(  58),  INT8_C(  99), -INT8_C( 118),
         INT8_C(  26), -INT8_C( 126), -INT8_C( 114),  INT8_C(  30), -INT8_C( 114), -INT8_C(  97),  INT8_C(  86), -INT8_C( 127),
        -INT8_C(  73), -INT8_C(  40), -INT8_C(  95),  INT8_C( 110),  INT8_C( 109),  INT8_C( 116), -INT8_C( 103),  INT8_C( 126) },
      { -INT8_C(  76),  INT8_C(  65),  INT8_C(  63), -INT8_C(  95), -INT8_C(  32), -INT8_C(  77), -INT8_C(   7), -INT8_C(  57),
        -INT8_C(   7),      INT8_MIN, -INT8_C(  97), -INT8_C( 127), -INT8_C( 103), -INT8_C(  42), -INT8_C(  41), -INT8_C( 122),
        -INT8_C(  20), -INT8_C(  15),  INT8_C(  31),  INT8_C(   4),  INT8_C(   3),  INT8_C(  12),  INT8_C(  16),  INT8_C(  56),
        -INT8_C(  13), -INT8_C( 123), -INT8_C( 126), -INT8_C(   4), -INT8_C(  73), -INT8_C( 108), -INT8_C( 106), -INT8_C(  55),
        -INT8_C( 121), -INT8_C(  43),  INT8_C(  39), -INT8_C(  88), -INT8_C( 120), -INT8_C(  91), -INT8_C(  72), -INT8_C( 127),
        -INT8_C(  25), -INT8_C(  98), -INT8_C(  17), -INT8_C(  65), -INT8_C(  98), -INT8_C(  58),  INT8_C(  99), -INT8_C( 118),
         INT8_C(  10), -INT8_C( 126), -INT8_C( 114),  INT8_C(  30), -INT8_C( 114), -INT8_C(  97), -INT8_C(  19), -INT8_C( 127),
        -INT8_C(  73), -INT8_C(  40), -INT8_C(  95),  INT8_C(  73),  INT8_C(  44), -INT8_C(  98), -INT8_C( 103), -INT8_C(  77) } },
    { { -INT8_C(  94), -INT8_C(  63),  INT8_C( 111),  INT8_C(  43),  INT8_C( 102),  INT8_C(  39), -INT8_C(  83), -INT8_C( 116),
        -INT8_C( 106), -INT8_C(  99),  INT8_C(  76),  INT8_C(  52),  INT8_C(  99), -INT8_C(  81), -INT8_C(  66),  INT8_C( 126),
         INT8_C(  50),  INT8_C(  77), -INT8_C( 100), -INT8_C(  64), -INT8_C(  20), -INT8_C(  14),  INT8_C(  66), -INT8_C(  93),
        -INT8_C(  53), -INT8_C(  29),  INT8_C(  18),  INT8_C(  56),  INT8_C(  87), -INT8_C(  85), -INT8_C(  74), -INT8_C(   7),
         INT8_C( 108),  INT8_C(  37),  INT8_C(  37), -INT8_C(  45),  INT8_C(  76), -INT8_C(  46),  INT8_C(  95), -INT8_C(  30),
         INT8_C( 111), -INT8_C(  85),  INT8_C(  23), -INT8_C(  45),  INT8_C(  91), -INT8_C(  43),  INT8_C(  81), -INT8_C( 115),
         INT8_C(  34), -INT8_C(  19),  INT8_C(  77),  INT8_C(  14), -INT8_C(  33), -INT8_C( 113), -INT8_C(  78), -INT8_C(  86),
         INT8_C( 114), -INT8_C(  60), -INT8_C(  30), -INT8_C(  55),  INT8_C( 111), -INT8_C( 104), -INT8_C(  61), -INT8_C(  36) },
      { -INT8_C(  67), -INT8_C(  24), -INT8_C(  81),  INT8_C(   9), -INT8_C(  70),  INT8_C(  14), -INT8_C(  20),  INT8_C(  42),
        -INT8_C(  70),  INT8_C(   3), -INT8_C(   3),  INT8_C(  21), -INT8_C(  40),  INT8_C(  78), -INT8_C(  94), -INT8_C(   5),
         INT8_C(  59), -INT8_C(  17),  INT8_C(   9),  INT8_C(  26),      INT8_MAX, -INT8_C(  69), -INT8_C(  59), -INT8_C(  15),
             INT8_MAX, -INT8_C(  89), -INT8_C(  69), -INT8_C(  17),  INT8_C(  64),  INT8_C( 126), -INT8_C(  53), -INT8_C(   3),
         INT8_C( 102),  INT8_C( 122),  INT8_C(   7),  INT8_C(  32), -INT8_C( 120), -INT8_C(  13),  INT8_C(  74),  INT8_C(  66),
        -INT8_C(  10),  INT8_C(  71),  INT8_C(  87), -INT8_C(  50), -INT8_C( 107), -INT8_C(   7), -INT8_C(  55), -INT8_C(  48),
        -INT8_C(  23), -INT8_C(  45), -INT8_C(  21),  INT8_C( 104), -INT8_C( 114), -INT8_C(  80),  INT8_C(  89),  INT8_C(  14),
         INT8_C(  87),  INT8_C(  20), -INT8_C(   3), -INT8_C( 105), -INT8_C( 110), -INT8_C(  56), -INT8_C( 107), -INT8_C(   8) },
      { -INT8_C(  94), -INT8_C(  63), -INT8_C(  81),  INT8_C(   9), -INT8_C(  70),  INT8_C(  14), -INT8_C(  83), -INT8_C( 116),
        -INT8_C( 106), -INT8_C(  99), -INT8_C(   3),  INT8_C(  21), -INT8_C(  40), -INT8_C(  81), -INT8_C(  94), -INT8_C(   5),
         INT8_C(  50), -INT8_C(  17), -INT8_C( 100), -INT8_C(  64), -INT8_C(  20), -INT8_C(  69), -INT8_C(  59), -INT8_C(  93),
        -INT8_C(  53), -INT8_C(  89), -INT8_C(  69), -INT8_C(  17),  INT8_C(  64), -INT8_C(  85), -INT8_C(  74), -INT8_C(   7),
         INT8_C( 102),  INT8_C(  37),  INT8_C(   7), -INT8_C(  45), -INT8_C( 120), -INT8_C(  46),  INT8_C(  74), -INT8_C(  30),
        -INT8_C(  10), -INT8_C(  85),  INT8_C(  23), -INT8_C(  50), -INT8_C( 107), -INT8_C(  43), -INT8_C(  55), -INT8_C( 115),
        -INT8_C(  23), -INT8_C(  45), -INT8_C(  21),  INT8_C(  14), -INT8_C( 114), -INT8_C( 113), -INT8_C(  78), -INT8_C(  86),
         INT8_C(  87), -INT8_C(  60), -INT8_C(  30), -INT8_C( 105), -INT8_C( 110), -INT8_C( 104), -INT8_C( 107), -INT8_C(  36) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_min_epi8(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_epi8");
    easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_min_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int8_t src[64];
    const easysimd__mmask64 k;
    const int8_t a[64];
    const int8_t b[64];
    const int8_t r[64];
  } test_vec[] = {
    { {  INT8_C(  69),  INT8_C(  77),  INT8_C( 104),  INT8_C(  74), -INT8_C(  56),  INT8_C(  84),  INT8_C( 106),  INT8_C( 109),
        -INT8_C(  29), -INT8_C(  33),  INT8_C(  11),  INT8_C(  57),  INT8_C(  28), -INT8_C( 111),  INT8_C( 107), -INT8_C(  70),
         INT8_C(  59), -INT8_C(  51),  INT8_C(   3), -INT8_C(  78),  INT8_C(  49),  INT8_C(   6),  INT8_C(  85), -INT8_C(  50),
         INT8_C(  24),  INT8_C(  48), -INT8_C(  61), -INT8_C(   7), -INT8_C( 125),  INT8_C(   2), -INT8_C(  65), -INT8_C(  55),
         INT8_C(  79),  INT8_C(  39),  INT8_C(  19),  INT8_C(  24),  INT8_C( 123),  INT8_C( 126), -INT8_C( 123),  INT8_C(  94),
         INT8_C(  93), -INT8_C( 112), -INT8_C( 105),  INT8_C( 121),  INT8_C(  34),  INT8_C(   3),  INT8_C(  52),  INT8_C(  93),
        -INT8_C(  48),  INT8_C(  55),  INT8_C(  15),  INT8_C(   2),  INT8_C(  61),  INT8_C( 100), -INT8_C(  48),  INT8_C(  85),
        -INT8_C( 108), -INT8_C( 108),  INT8_C(  79),  INT8_C(  24), -INT8_C( 106),  INT8_C(  14), -INT8_C(  31), -INT8_C(  27) },
      UINT64_C(14920570094114174006),
      {  INT8_C(  19), -INT8_C(  89),  INT8_C(  73),  INT8_C(  53), -INT8_C(  86),  INT8_C( 125), -INT8_C( 110),  INT8_C( 123),
        -INT8_C(  76), -INT8_C(  94),  INT8_C( 125), -INT8_C(  15),  INT8_C(   6),  INT8_C(  77),  INT8_C(  71), -INT8_C( 101),
        -INT8_C(  31), -INT8_C( 106), -INT8_C(  77),  INT8_C( 119), -INT8_C(  92), -INT8_C( 108),  INT8_C(  93), -INT8_C(  38),
        -INT8_C( 120),  INT8_C(  90), -INT8_C( 116), -INT8_C(   5), -INT8_C(  35), -INT8_C( 100), -INT8_C(  54), -INT8_C(  15),
         INT8_C(  67),  INT8_C(  19),  INT8_C(  38), -INT8_C(  18), -INT8_C( 112), -INT8_C(  71),  INT8_C( 105),  INT8_C(  68),
         INT8_C(  91), -INT8_C(  26),  INT8_C(  54),  INT8_C(  97),  INT8_C(  51),  INT8_C( 125), -INT8_C(   4),  INT8_C(  21),
         INT8_C(  19), -INT8_C(  81), -INT8_C( 116), -INT8_C(  73),  INT8_C(  67), -INT8_C(  23), -INT8_C( 110), -INT8_C(  52),
         INT8_C(  68),  INT8_C(  30), -INT8_C(  57),  INT8_C(  33), -INT8_C(  70), -INT8_C( 111),  INT8_C(  18), -INT8_C(   3) },
      { -INT8_C(  91),  INT8_C(  57), -INT8_C(  21),  INT8_C(  53), -INT8_C(  14),  INT8_C(  84),  INT8_C( 122),  INT8_C(  77),
         INT8_C(  58), -INT8_C(  80), -INT8_C(  82),  INT8_C( 110),  INT8_C(  45), -INT8_C(  85), -INT8_C( 125),  INT8_C(  64),
         INT8_C(  90),  INT8_C(  15), -INT8_C(   9), -INT8_C(  98), -INT8_C(   7), -INT8_C( 119),  INT8_C( 106),  INT8_C(  61),
        -INT8_C(  89),  INT8_C(  49),  INT8_C(  94),  INT8_C(  97), -INT8_C(  62),  INT8_C( 113),  INT8_C(  95),  INT8_C( 103),
        -INT8_C(  86),  INT8_C(  74), -INT8_C(  99), -INT8_C( 100), -INT8_C(  97),  INT8_C(  23), -INT8_C(  23), -INT8_C(  39),
        -INT8_C(  57), -INT8_C( 105),  INT8_C(  71), -INT8_C(  12),  INT8_C(  66), -INT8_C(  54),  INT8_C(  52), -INT8_C(  99),
        -INT8_C(  38),  INT8_C(  43),  INT8_C(  59), -INT8_C(  45), -INT8_C(  75), -INT8_C(  91),  INT8_C(  16),  INT8_C(  92),
        -INT8_C(  42),  INT8_C( 110), -INT8_C(  66), -INT8_C( 104), -INT8_C(  33),  INT8_C(  29),  INT8_C(   0), -INT8_C( 119) },
      {  INT8_C(  69), -INT8_C(  89), -INT8_C(  21),  INT8_C(  74), -INT8_C(  86),  INT8_C(  84),  INT8_C( 106),  INT8_C( 109),
        -INT8_C(  29), -INT8_C(  33), -INT8_C(  82),  INT8_C(  57),  INT8_C(   6), -INT8_C(  85), -INT8_C( 125), -INT8_C( 101),
        -INT8_C(  31), -INT8_C(  51), -INT8_C(  77), -INT8_C(  98), -INT8_C(  92), -INT8_C( 119),  INT8_C(  93), -INT8_C(  38),
        -INT8_C( 120),  INT8_C(  48), -INT8_C(  61), -INT8_C(   7), -INT8_C(  62), -INT8_C( 100), -INT8_C(  65), -INT8_C(  15),
         INT8_C(  79),  INT8_C(  19),  INT8_C(  19),  INT8_C(  24), -INT8_C( 112), -INT8_C(  71), -INT8_C(  23),  INT8_C(  94),
        -INT8_C(  57), -INT8_C( 105), -INT8_C( 105),  INT8_C( 121),  INT8_C(  34),  INT8_C(   3),  INT8_C(  52), -INT8_C(  99),
        -INT8_C(  48),  INT8_C(  55),  INT8_C(  15),  INT8_C(   2), -INT8_C(  75),  INT8_C( 100), -INT8_C(  48),  INT8_C(  85),
        -INT8_C(  42),  INT8_C(  30), -INT8_C(  66), -INT8_C( 104), -INT8_C( 106),  INT8_C(  14),  INT8_C(   0), -INT8_C( 119) } },
    { {  INT8_C( 103), -INT8_C(  99),  INT8_C(  37),  INT8_C(   6), -INT8_C(  76),  INT8_C(  14), -INT8_C(  32),  INT8_C( 123),
        -INT8_C(  90),  INT8_C(  39),  INT8_C( 111), -INT8_C(  24), -INT8_C(  14), -INT8_C(  93), -INT8_C( 123), -INT8_C(  52),
        -INT8_C(  50), -INT8_C(  64), -INT8_C(  97), -INT8_C( 125),  INT8_C( 101), -INT8_C(  81), -INT8_C(  32),  INT8_C(  59),
         INT8_C(  29), -INT8_C(  98), -INT8_C(  44), -INT8_C(   3), -INT8_C(  69), -INT8_C(  44), -INT8_C( 122),  INT8_C(  34),
         INT8_C( 113), -INT8_C(  84),  INT8_C(  41),  INT8_C(  37), -INT8_C(  70),  INT8_C(   9), -INT8_C(  96),  INT8_C(  96),
         INT8_C(  48),  INT8_C(  15),  INT8_C(  73),  INT8_C(  34), -INT8_C(  78), -INT8_C(  50), -INT8_C(  18),      INT8_MIN,
        -INT8_C( 113), -INT8_C( 115),  INT8_C(   4), -INT8_C(  12),  INT8_C(  60), -INT8_C(  28),  INT8_C(  48),  INT8_C(  90),
        -INT8_C( 126),  INT8_C(   4),  INT8_C(  87),  INT8_C(  61), -INT8_C(  40), -INT8_C(  35),  INT8_C(  95),  INT8_C(  73) },
      UINT64_C(14025351156619708553),
      {  INT8_C(  29), -INT8_C(  19), -INT8_C(  28), -INT8_C(  49), -INT8_C(  68), -INT8_C(  45),  INT8_C(  79),  INT8_C(  75),
         INT8_C(  96),  INT8_C(  83),  INT8_C(  63), -INT8_C(  99),  INT8_C(  55),  INT8_C( 111), -INT8_C(   9), -INT8_C(  71),
         INT8_C( 115),  INT8_C(  78), -INT8_C(  10),  INT8_C(  75),  INT8_C(  43),  INT8_C(  86), -INT8_C( 108), -INT8_C(  75),
        -INT8_C(  34),  INT8_C(   2), -INT8_C(   7),  INT8_C( 112),  INT8_C(  16), -INT8_C(  99),  INT8_C(  50),  INT8_C(  45),
        -INT8_C( 117),  INT8_C(  22), -INT8_C(   4),  INT8_C(  71), -INT8_C(  23),  INT8_C(  76), -INT8_C( 110),  INT8_C(  74),
        -INT8_C(  97), -INT8_C(  47), -INT8_C(  25), -INT8_C(  41),  INT8_C(  65), -INT8_C(  34), -INT8_C( 112), -INT8_C(  76),
         INT8_C(  44), -INT8_C( 121),  INT8_C(   0),  INT8_C(  87), -INT8_C(  35), -INT8_C( 108),  INT8_C(  12), -INT8_C(  69),
        -INT8_C( 105),  INT8_C(   5),  INT8_C(  43), -INT8_C(  89), -INT8_C(  93),  INT8_C(  93), -INT8_C(  43),  INT8_C(  46) },
      {  INT8_C( 116), -INT8_C(  47),  INT8_C( 117),  INT8_C(  93),  INT8_C(  29),  INT8_C(   7), -INT8_C(  89), -INT8_C(  67),
        -INT8_C(  40), -INT8_C( 114), -INT8_C( 108),  INT8_C(  25),  INT8_C( 108),  INT8_C(  36), -INT8_C(  50), -INT8_C( 104),
        -INT8_C(  85), -INT8_C(  50), -INT8_C(  16), -INT8_C( 120),  INT8_C(  98), -INT8_C(   4),  INT8_C(  68), -INT8_C(   7),
         INT8_C(   2),  INT8_C( 111), -INT8_C(  95), -INT8_C(  91), -INT8_C(  51),  INT8_C( 118), -INT8_C(  45),  INT8_C(  65),
         INT8_C(  71),  INT8_C(  72), -INT8_C(  98),  INT8_C( 101),  INT8_C(  79),  INT8_C(  70),  INT8_C(  34),  INT8_C(  39),
        -INT8_C(  44), -INT8_C(  74),  INT8_C(  65),  INT8_C(  65), -INT8_C(  38),  INT8_C(  15), -INT8_C(  39), -INT8_C( 122),
        -INT8_C(  35), -INT8_C(  55),  INT8_C(  14),  INT8_C(  63), -INT8_C(  58),  INT8_C(  82),  INT8_C(  57), -INT8_C(  56),
        -INT8_C(  62), -INT8_C(  38),  INT8_C( 109), -INT8_C( 113),  INT8_C(  80),  INT8_C(  64), -INT8_C(  48), -INT8_C( 105) },
      {  INT8_C(  29), -INT8_C(  99),  INT8_C(  37), -INT8_C(  49), -INT8_C(  76),  INT8_C(  14), -INT8_C(  32), -INT8_C(  67),
        -INT8_C(  90),  INT8_C(  39),  INT8_C( 111), -INT8_C(  99), -INT8_C(  14), -INT8_C(  93), -INT8_C( 123), -INT8_C( 104),
        -INT8_C(  50), -INT8_C(  50), -INT8_C(  16), -INT8_C( 120),  INT8_C( 101), -INT8_C(   4), -INT8_C( 108),  INT8_C(  59),
         INT8_C(  29), -INT8_C(  98), -INT8_C(  95), -INT8_C(   3), -INT8_C(  69), -INT8_C(  44), -INT8_C(  45),  INT8_C(  34),
        -INT8_C( 117), -INT8_C(  84),  INT8_C(  41),  INT8_C(  37), -INT8_C(  23),  INT8_C(   9), -INT8_C(  96),  INT8_C(  39),
         INT8_C(  48), -INT8_C(  74), -INT8_C(  25), -INT8_C(  41), -INT8_C(  78), -INT8_C(  50), -INT8_C(  18),      INT8_MIN,
        -INT8_C( 113), -INT8_C( 115),  INT8_C(   0), -INT8_C(  12),  INT8_C(  60), -INT8_C( 108),  INT8_C(  48), -INT8_C(  69),
        -INT8_C( 126), -INT8_C(  38),  INT8_C(  87),  INT8_C(  61), -INT8_C(  40), -INT8_C(  35), -INT8_C(  48), -INT8_C( 105) } },
    { { -INT8_C( 120),  INT8_C( 110), -INT8_C(   4), -INT8_C(  41), -INT8_C(  76),  INT8_C(  30), -INT8_C(   2), -INT8_C( 119),
        -INT8_C(  44),  INT8_C(  63), -INT8_C(  54), -INT8_C(  81),  INT8_C(  78), -INT8_C(  93),  INT8_C(  53),  INT8_C(  43),
         INT8_C( 109),  INT8_C(  67),  INT8_C( 107),  INT8_C(  51), -INT8_C( 106), -INT8_C(  92), -INT8_C(   5),  INT8_C(  88),
         INT8_C( 126),  INT8_C( 104), -INT8_C(  25), -INT8_C(  50), -INT8_C(  88), -INT8_C(  73),  INT8_C( 101),  INT8_C(  48),
         INT8_C(  37),  INT8_C(  98),  INT8_C(   7), -INT8_C(  38),      INT8_MIN,  INT8_C(   5),  INT8_C(  99),  INT8_C(  85),
         INT8_C(  69),  INT8_C(  45),  INT8_C(   4), -INT8_C( 109), -INT8_C(  48),  INT8_C(  57), -INT8_C(  65),  INT8_C(  61),
         INT8_C( 124),  INT8_C(  42),  INT8_C( 112),  INT8_C(  18), -INT8_C(  50),  INT8_C( 107),  INT8_C( 106),  INT8_C(  76),
        -INT8_C(  45),  INT8_C(  81),  INT8_C(  26),  INT8_C( 123),  INT8_C(   8),      INT8_MAX, -INT8_C(  85),  INT8_C(  46) },
      UINT64_C(18282199651996709601),
      { -INT8_C( 104), -INT8_C(  69), -INT8_C( 112),  INT8_C( 104), -INT8_C(  12),  INT8_C(  79), -INT8_C(  90),  INT8_C( 112),
         INT8_C( 121),  INT8_C(  22), -INT8_C( 125),  INT8_C(  71), -INT8_C( 126), -INT8_C(  19), -INT8_C( 109),  INT8_C(  85),
         INT8_C(  63), -INT8_C(  83), -INT8_C(  47),  INT8_C(  71),  INT8_C(  45),  INT8_C( 124),  INT8_C( 117),  INT8_C(  14),
         INT8_C(  47),  INT8_C( 125),  INT8_C( 112), -INT8_C(  25), -INT8_C(  24),  INT8_C(  39), -INT8_C(  28),      INT8_MIN,
        -INT8_C(  30),  INT8_C( 116), -INT8_C(  23), -INT8_C(  42), -INT8_C(  60), -INT8_C( 113),  INT8_C(  71),  INT8_C(  61),
        -INT8_C(  91), -INT8_C(  54), -INT8_C( 123),  INT8_C(  39), -INT8_C(  73),  INT8_C(  24),  INT8_C( 125), -INT8_C(  10),
        -INT8_C(  58),  INT8_C(  78),  INT8_C(  62), -INT8_C(  13), -INT8_C(  54), -INT8_C(  77),  INT8_C(   1), -INT8_C(   7),
         INT8_C(  49),  INT8_C( 114), -INT8_C(  32),  INT8_C(  25), -INT8_C( 103), -INT8_C(  60), -INT8_C( 102),  INT8_C( 124) },
      {  INT8_C(  57), -INT8_C( 125),  INT8_C(  82), -INT8_C(   3),  INT8_C(  18), -INT8_C( 103),  INT8_C(  58), -INT8_C(  73),
         INT8_C(  99), -INT8_C(  65), -INT8_C(  33),  INT8_C(  27), -INT8_C(  40),  INT8_C(  92),  INT8_C(  17), -INT8_C(  98),
        -INT8_C(  86),  INT8_C(  79), -INT8_C( 111),  INT8_C( 116),  INT8_C(   3), -INT8_C( 110),  INT8_C( 110),  INT8_C(  52),
         INT8_C(   4),  INT8_C(  78),  INT8_C(  77), -INT8_C(  98),  INT8_C(  19), -INT8_C(  25),  INT8_C(  26),  INT8_C(  76),
         INT8_C( 106),  INT8_C( 108),  INT8_C(  73),  INT8_C( 124),  INT8_C(   6), -INT8_C( 125),  INT8_C(  52),  INT8_C( 105),
         INT8_C(  67),  INT8_C(  19), -INT8_C( 124),  INT8_C(  27),  INT8_C( 111), -INT8_C( 106), -INT8_C(  71),  INT8_C(  25),
        -INT8_C(  27),  INT8_C(  74), -INT8_C( 115), -INT8_C(  24), -INT8_C(  36), -INT8_C(   5),  INT8_C(  28), -INT8_C(  31),
         INT8_C(  74),  INT8_C( 106),      INT8_MAX,  INT8_C(  93),  INT8_C(  81), -INT8_C( 103), -INT8_C(  87), -INT8_C(  68) },
      { -INT8_C( 104),  INT8_C( 110), -INT8_C(   4), -INT8_C(  41), -INT8_C(  76), -INT8_C( 103), -INT8_C(  90), -INT8_C(  73),
        -INT8_C(  44), -INT8_C(  65), -INT8_C(  54), -INT8_C(  81), -INT8_C( 126), -INT8_C(  19),  INT8_C(  53), -INT8_C(  98),
         INT8_C( 109),  INT8_C(  67),  INT8_C( 107),  INT8_C(  71), -INT8_C( 106), -INT8_C(  92), -INT8_C(   5),  INT8_C(  88),
         INT8_C( 126),  INT8_C(  78), -INT8_C(  25), -INT8_C(  50), -INT8_C(  88), -INT8_C(  25), -INT8_C(  28),  INT8_C(  48),
         INT8_C(  37),  INT8_C(  98),  INT8_C(   7), -INT8_C(  42), -INT8_C(  60), -INT8_C( 125),  INT8_C(  99),  INT8_C(  61),
        -INT8_C(  91), -INT8_C(  54),  INT8_C(   4),  INT8_C(  27), -INT8_C(  48), -INT8_C( 106), -INT8_C(  71),  INT8_C(  61),
        -INT8_C(  58),  INT8_C(  74), -INT8_C( 115),  INT8_C(  18), -INT8_C(  54), -INT8_C(  77),  INT8_C( 106), -INT8_C(  31),
         INT8_C(  49),  INT8_C(  81), -INT8_C(  32),  INT8_C(  25), -INT8_C( 103), -INT8_C( 103), -INT8_C( 102), -INT8_C(  68) } },
    { {  INT8_C(   5), -INT8_C(  14),  INT8_C(  56),  INT8_C(  11),  INT8_C( 117),  INT8_C( 108),  INT8_C( 117), -INT8_C(  72),
             INT8_MAX, -INT8_C(   7), -INT8_C(  45), -INT8_C(  18), -INT8_C( 113), -INT8_C( 116),  INT8_C(   7),  INT8_C( 117),
        -INT8_C(  42), -INT8_C( 107),  INT8_C(  93), -INT8_C(  77), -INT8_C( 112),  INT8_C( 122), -INT8_C( 108), -INT8_C(  38),
        -INT8_C(  28),  INT8_C(  19),  INT8_C(  55),  INT8_C(  53), -INT8_C(  84), -INT8_C(  32), -INT8_C(  15), -INT8_C(  79),
        -INT8_C(  46),  INT8_C(  42), -INT8_C(  67),  INT8_C(  72), -INT8_C( 106),  INT8_C(  50),  INT8_C(   0),  INT8_C(  22),
         INT8_C(  43), -INT8_C(  44),  INT8_C(   4), -INT8_C(  69),  INT8_C(  96),  INT8_C(  12),  INT8_C(  48),  INT8_C(  55),
        -INT8_C(  95), -INT8_C( 115), -INT8_C(  22),  INT8_C(  49),  INT8_C(   7),  INT8_C( 126),  INT8_C(  12), -INT8_C(  21),
        -INT8_C( 111),  INT8_C(  67),  INT8_C(  33),  INT8_C(  61),  INT8_C(  36),  INT8_C(  18), -INT8_C(  18), -INT8_C(  10) },
      UINT64_C(  714172237879356220),
      {  INT8_C(  19), -INT8_C(  19), -INT8_C(  60),  INT8_C( 115), -INT8_C(   7), -INT8_C(  12), -INT8_C(  86), -INT8_C( 102),
        -INT8_C( 127), -INT8_C( 108), -INT8_C(  52), -INT8_C( 119),  INT8_C(  18), -INT8_C(  40),  INT8_C( 116), -INT8_C(  93),
         INT8_C(  27), -INT8_C( 107), -INT8_C(  32),  INT8_C(  63), -INT8_C(  88), -INT8_C(  49),  INT8_C(  54), -INT8_C(  28),
         INT8_C( 122),  INT8_C( 116), -INT8_C(  73),  INT8_C(  88), -INT8_C(  77), -INT8_C(  96),  INT8_C(  97), -INT8_C(  58),
        -INT8_C( 114),  INT8_C(  37),  INT8_C(  58), -INT8_C( 121),  INT8_C(  25), -INT8_C(  28),  INT8_C(  34), -INT8_C( 102),
         INT8_C( 121), -INT8_C(  18),  INT8_C(  35), -INT8_C( 117), -INT8_C(  58), -INT8_C( 104),  INT8_C(  47), -INT8_C(  31),
         INT8_C(  45),  INT8_C(  15),  INT8_C(  33), -INT8_C(  43), -INT8_C(  34),  INT8_C(  87), -INT8_C(  70),  INT8_C(  89),
        -INT8_C(  53),  INT8_C( 113), -INT8_C(  79),      INT8_MAX,  INT8_C(  18),  INT8_C(  18),  INT8_C(  69), -INT8_C(  96) },
      {  INT8_C(  55),      INT8_MAX,  INT8_C(  39),  INT8_C(  80),  INT8_C( 100),  INT8_C(  73), -INT8_C(  22), -INT8_C(  35),
         INT8_C(  55),  INT8_C(  14),  INT8_C( 104), -INT8_C(   3), -INT8_C(  90), -INT8_C( 105), -INT8_C(  33), -INT8_C(  45),
        -INT8_C(  89),  INT8_C(   0), -INT8_C(  87), -INT8_C( 123),  INT8_C(  87),  INT8_C(  99), -INT8_C(  34),  INT8_C(  34),
        -INT8_C(  44), -INT8_C( 113), -INT8_C(  95), -INT8_C(  26), -INT8_C(  95), -INT8_C(  25), -INT8_C( 122), -INT8_C(  40),
         INT8_C( 102), -INT8_C(  82),  INT8_C(  40), -INT8_C(  54), -INT8_C(   9),  INT8_C(  19), -INT8_C(  89),  INT8_C(  47),
         INT8_C(  33),  INT8_C(  16),  INT8_C(  44), -INT8_C(  57), -INT8_C(  89),  INT8_C(  11), -INT8_C( 102),  INT8_C(  78),
         INT8_C(  11),  INT8_C(  67), -INT8_C(  44),  INT8_C(  98), -INT8_C(  90), -INT8_C(  78), -INT8_C( 123),  INT8_C( 123),
         INT8_C(  66),  INT8_C(  38),  INT8_C(  97), -INT8_C(  29),  INT8_C(  13), -INT8_C(  24), -INT8_C(  68),  INT8_C( 116) },
      {  INT8_C(   5), -INT8_C(  14), -INT8_C(  60),  INT8_C(  80), -INT8_C(   7), -INT8_C(  12),  INT8_C( 117), -INT8_C(  72),
        -INT8_C( 127), -INT8_C( 108), -INT8_C(  45), -INT8_C( 119), -INT8_C( 113), -INT8_C( 105),  INT8_C(   7), -INT8_C(  93),
        -INT8_C(  42), -INT8_C( 107), -INT8_C(  87), -INT8_C( 123), -INT8_C(  88), -INT8_C(  49), -INT8_C( 108), -INT8_C(  38),
        -INT8_C(  44), -INT8_C( 113),  INT8_C(  55),  INT8_C(  53), -INT8_C(  95), -INT8_C(  32), -INT8_C( 122), -INT8_C(  58),
        -INT8_C( 114),  INT8_C(  42),  INT8_C(  40), -INT8_C( 121), -INT8_C(   9),  INT8_C(  50), -INT8_C(  89), -INT8_C( 102),
         INT8_C(  33), -INT8_C(  18),  INT8_C(  35), -INT8_C( 117), -INT8_C(  89), -INT8_C( 104),  INT8_C(  48),  INT8_C(  55),
         INT8_C(  11), -INT8_C( 115), -INT8_C(  22), -INT8_C(  43),  INT8_C(   7), -INT8_C(  78), -INT8_C( 123),  INT8_C(  89),
        -INT8_C(  53),  INT8_C(  67),  INT8_C(  33), -INT8_C(  29),  INT8_C(  36),  INT8_C(  18), -INT8_C(  18), -INT8_C(  10) } },
    { { -INT8_C( 106), -INT8_C(  28),  INT8_C(  62), -INT8_C( 115), -INT8_C(   9), -INT8_C(  26), -INT8_C(  68),  INT8_C(  24),
        -INT8_C(  10), -INT8_C(  23), -INT8_C(  33), -INT8_C(  99), -INT8_C(  12),  INT8_C( 122), -INT8_C(  20),  INT8_C(   0),
        -INT8_C(  67), -INT8_C(  64),  INT8_C(  98),  INT8_C( 100),  INT8_C( 114), -INT8_C(  25), -INT8_C(  33), -INT8_C(  76),
         INT8_C(  14),  INT8_C(  64), -INT8_C( 104),  INT8_C(  27),  INT8_C(  40),  INT8_C(  84), -INT8_C( 113), -INT8_C(  66),
         INT8_C(  56), -INT8_C(  50),  INT8_C(  76),  INT8_C(  48), -INT8_C(  76),  INT8_C(   8),  INT8_C(  72), -INT8_C(  86),
        -INT8_C(  15),  INT8_C(  40),  INT8_C(  71), -INT8_C(  26), -INT8_C(  94),  INT8_C(  51), -INT8_C(  26),  INT8_C(  95),
        -INT8_C(  13),  INT8_C(  72), -INT8_C(  61),  INT8_C( 102),  INT8_C(  48), -INT8_C(  94),  INT8_C(  26),  INT8_C(  62),
        -INT8_C(  29), -INT8_C(  78),  INT8_C(  89),  INT8_C(  11),  INT8_C(   6), -INT8_C(  23), -INT8_C(  54),  INT8_C(  63) },
      UINT64_C( 1159033820397115063),
      { -INT8_C(  33),  INT8_C(  92), -INT8_C(  10), -INT8_C( 127), -INT8_C( 112), -INT8_C(  36), -INT8_C(  31), -INT8_C( 125),
         INT8_C(  36), -INT8_C(  92), -INT8_C(  23),  INT8_C(  84),  INT8_C(  71),  INT8_C(   4), -INT8_C( 110),  INT8_C(  42),
        -INT8_C(  74), -INT8_C(  20),  INT8_C(  53), -INT8_C(  67), -INT8_C(  43), -INT8_C(   1), -INT8_C(   4), -INT8_C( 116),
         INT8_C(  21),  INT8_C( 107), -INT8_C(   9),  INT8_C(  52),  INT8_C(  34),  INT8_C(  12),  INT8_C(  68),  INT8_C(   2),
         INT8_C( 104),  INT8_C(  58), -INT8_C( 125), -INT8_C(   8),  INT8_C(  22),  INT8_C( 100),  INT8_C( 124),  INT8_C(  58),
         INT8_C(   9),  INT8_C( 101), -INT8_C( 113),  INT8_C(  80),  INT8_C( 105),  INT8_C(  33),  INT8_C( 122),  INT8_C(  32),
         INT8_C(  13), -INT8_C(  81), -INT8_C(  35), -INT8_C(  30), -INT8_C(  81), -INT8_C(  39),  INT8_C( 110), -INT8_C(  60),
         INT8_C(  68),  INT8_C( 101), -INT8_C(   8),  INT8_C( 102),  INT8_C( 113),  INT8_C(  60),  INT8_C( 104), -INT8_C(  38) },
      {  INT8_C( 118), -INT8_C(  20), -INT8_C(  46), -INT8_C( 116),  INT8_C(  80),  INT8_C(  78), -INT8_C(  57),  INT8_C(  89),
        -INT8_C(  76),  INT8_C(  86), -INT8_C(  87),  INT8_C(  29),  INT8_C( 119),  INT8_C(  35),  INT8_C(  61), -INT8_C( 123),
        -INT8_C(  45),  INT8_C(  26),  INT8_C( 103), -INT8_C( 126), -INT8_C(  13), -INT8_C(  42),  INT8_C(  70),  INT8_C(  55),
         INT8_C(  59),  INT8_C(  63), -INT8_C(  98), -INT8_C(  83),  INT8_C( 123),  INT8_C(   6), -INT8_C( 121), -INT8_C(  14),
        -INT8_C(  14),  INT8_C(  89),  INT8_C( 126),  INT8_C(  67), -INT8_C(  88),  INT8_C(  69), -INT8_C( 100),  INT8_C(  92),
        -INT8_C( 101),  INT8_C(  70),  INT8_C( 121),  INT8_C(  19),  INT8_C( 105), -INT8_C(  73), -INT8_C( 104),  INT8_C(  60),
        -INT8_C(  47), -INT8_C(   1), -INT8_C(  66), -INT8_C(  59), -INT8_C(  43),  INT8_C(   5), -INT8_C(   4),  INT8_C(  17),
         INT8_C(  68), -INT8_C( 102), -INT8_C(  66), -INT8_C(  65), -INT8_C(  95),  INT8_C(  69), -INT8_C(  79), -INT8_C( 109) },
      { -INT8_C(  33), -INT8_C(  20), -INT8_C(  46), -INT8_C( 115), -INT8_C( 112), -INT8_C(  36), -INT8_C(  68), -INT8_C( 125),
        -INT8_C(  10), -INT8_C(  92), -INT8_C(  87), -INT8_C(  99),  INT8_C(  71),  INT8_C( 122), -INT8_C(  20),  INT8_C(   0),
        -INT8_C(  74), -INT8_C(  20),  INT8_C(  53), -INT8_C( 126),  INT8_C( 114), -INT8_C(  42), -INT8_C(   4), -INT8_C(  76),
         INT8_C(  21),  INT8_C(  63), -INT8_C( 104), -INT8_C(  83),  INT8_C(  40),  INT8_C(   6), -INT8_C( 121), -INT8_C(  66),
         INT8_C(  56),  INT8_C(  58), -INT8_C( 125), -INT8_C(   8), -INT8_C(  88),  INT8_C(   8),  INT8_C(  72), -INT8_C(  86),
        -INT8_C( 101),  INT8_C(  70), -INT8_C( 113), -INT8_C(  26),  INT8_C( 105), -INT8_C(  73), -INT8_C(  26),  INT8_C(  32),
        -INT8_C(  47),  INT8_C(  72), -INT8_C(  66),  INT8_C( 102), -INT8_C(  81), -INT8_C(  94),  INT8_C(  26),  INT8_C(  62),
        -INT8_C(  29), -INT8_C(  78),  INT8_C(  89),  INT8_C(  11), -INT8_C(  95), -INT8_C(  23), -INT8_C(  54),  INT8_C(  63) } },
    { { -INT8_C(  98),  INT8_C(  48), -INT8_C(  42),  INT8_C(  70),  INT8_C( 117),  INT8_C( 115), -INT8_C(  94),  INT8_C(  17),
        -INT8_C(  71),  INT8_C(  28),  INT8_C(  36),  INT8_C(  34), -INT8_C(  45), -INT8_C(  68),  INT8_C(  95), -INT8_C(  92),
        -INT8_C(  69),  INT8_C(  29),  INT8_C( 105), -INT8_C( 111),  INT8_C(  34),  INT8_C( 102), -INT8_C(  94),  INT8_C( 102),
         INT8_C(   0),  INT8_C(  96),  INT8_C(  38), -INT8_C(  95), -INT8_C(  91), -INT8_C(  41),  INT8_C(  53),  INT8_C(  67),
         INT8_C(   7),  INT8_C(  11), -INT8_C( 118),  INT8_C( 125),  INT8_C( 126),  INT8_C(  44), -INT8_C( 114),  INT8_C(  55),
         INT8_C(  72), -INT8_C(  78),  INT8_C(  90),  INT8_C(  27),  INT8_C( 110), -INT8_C(  71), -INT8_C(  64),  INT8_C(  41),
        -INT8_C(  42),  INT8_C(  41), -INT8_C(  70), -INT8_C(   7), -INT8_C( 113),  INT8_C(  92),  INT8_C(  95), -INT8_C( 112),
        -INT8_C(  68), -INT8_C( 123),  INT8_C(  49),  INT8_C(  97),  INT8_C(  93),  INT8_C( 102), -INT8_C(  91),  INT8_C( 100) },
      UINT64_C(11828826861962604402),
      {  INT8_C(  33), -INT8_C( 126), -INT8_C(  65), -INT8_C( 113),  INT8_C(  59),      INT8_MAX, -INT8_C(  71),  INT8_C(  17),
        -INT8_C(  87),  INT8_C( 115),  INT8_C(  10),  INT8_C(  56), -INT8_C(  48),  INT8_C( 106), -INT8_C(  56), -INT8_C( 116),
        -INT8_C(  17), -INT8_C(   6), -INT8_C(  18),  INT8_C(  76),  INT8_C(  96), -INT8_C( 109), -INT8_C(  79), -INT8_C(  46),
        -INT8_C(  62), -INT8_C( 110), -INT8_C(  61),  INT8_C(  29),  INT8_C(   2), -INT8_C(  21), -INT8_C(  63),  INT8_C(  35),
         INT8_C( 109), -INT8_C( 127), -INT8_C(  77), -INT8_C(  88),  INT8_C(   0),  INT8_C( 108), -INT8_C(  71), -INT8_C(  87),
        -INT8_C(  33), -INT8_C(  60), -INT8_C(  30), -INT8_C(  81),  INT8_C(  46), -INT8_C(  86),  INT8_C(  60),  INT8_C(  29),
        -INT8_C(  92),  INT8_C(  42),  INT8_C( 106),  INT8_C(   5), -INT8_C(  67),  INT8_C(  27), -INT8_C(  41),      INT8_MAX,
        -INT8_C(  83), -INT8_C( 102), -INT8_C( 100), -INT8_C(  81), -INT8_C( 123),  INT8_C(  94), -INT8_C(  45), -INT8_C(  14) },
      { -INT8_C(  33), -INT8_C( 122), -INT8_C( 102), -INT8_C(  33), -INT8_C(  14),  INT8_C(  84), -INT8_C( 119), -INT8_C(  47),
         INT8_C(  24),  INT8_C( 107), -INT8_C( 127),  INT8_C(  70),  INT8_C(  21), -INT8_C(  67),  INT8_C(  99), -INT8_C(  70),
        -INT8_C(  25), -INT8_C(  51), -INT8_C(  65), -INT8_C(  92), -INT8_C(  24), -INT8_C( 106),  INT8_C(  35), -INT8_C( 106),
         INT8_C(  49), -INT8_C(  65),  INT8_C(  69), -INT8_C(  74),  INT8_C(  29),  INT8_C(  24), -INT8_C(  87), -INT8_C(   4),
        -INT8_C(  98),  INT8_C(  67), -INT8_C(  36), -INT8_C( 112), -INT8_C( 105),  INT8_C( 101),  INT8_C(  98), -INT8_C(  81),
        -INT8_C(  48), -INT8_C(  29), -INT8_C(  11), -INT8_C(  27), -INT8_C(  96),  INT8_C(  89), -INT8_C(  97), -INT8_C( 121),
         INT8_C(  38),  INT8_C(  94),  INT8_C(  43),  INT8_C(  15), -INT8_C(  11),  INT8_C(  78), -INT8_C(  91),  INT8_C(  38),
         INT8_C(  13), -INT8_C(  22), -INT8_C(  36),  INT8_C(  43),  INT8_C(   3), -INT8_C( 123),  INT8_C(  39), -INT8_C(  95) },
      { -INT8_C(  98), -INT8_C( 126), -INT8_C(  42),  INT8_C(  70), -INT8_C(  14),  INT8_C(  84), -INT8_C( 119),  INT8_C(  17),
        -INT8_C(  87),  INT8_C( 107), -INT8_C( 127),  INT8_C(  56), -INT8_C(  45), -INT8_C(  67),  INT8_C(  95), -INT8_C(  92),
        -INT8_C(  25),  INT8_C(  29),  INT8_C( 105), -INT8_C( 111),  INT8_C(  34), -INT8_C( 109), -INT8_C(  79), -INT8_C( 106),
         INT8_C(   0),  INT8_C(  96),  INT8_C(  38), -INT8_C(  95),  INT8_C(   2), -INT8_C(  21), -INT8_C(  87), -INT8_C(   4),
        -INT8_C(  98), -INT8_C( 127), -INT8_C( 118), -INT8_C( 112), -INT8_C( 105),  INT8_C(  44), -INT8_C(  71),  INT8_C(  55),
        -INT8_C(  48), -INT8_C(  60), -INT8_C(  30), -INT8_C(  81),  INT8_C( 110), -INT8_C(  86), -INT8_C(  97),  INT8_C(  41),
        -INT8_C(  42),  INT8_C(  41), -INT8_C(  70),  INT8_C(   5), -INT8_C( 113),  INT8_C(  27),  INT8_C(  95), -INT8_C( 112),
        -INT8_C(  68), -INT8_C( 123), -INT8_C( 100),  INT8_C(  97),  INT8_C(  93), -INT8_C( 123), -INT8_C(  91), -INT8_C(  95) } },
    { { -INT8_C(  55),  INT8_C(   3),  INT8_C(  50),  INT8_C(  96),  INT8_C( 104), -INT8_C( 108),  INT8_C(  16),  INT8_C(  56),
         INT8_C( 119),  INT8_C(   5),  INT8_C(  30),  INT8_C(  23),  INT8_C(  94), -INT8_C(  67), -INT8_C(  98), -INT8_C( 123),
         INT8_C(  28), -INT8_C(  55), -INT8_C( 108),  INT8_C(  17),  INT8_C(  23),  INT8_C(  57),  INT8_C(  55),  INT8_C(  36),
         INT8_C(  35),  INT8_C(  19),  INT8_C(  79),  INT8_C(  38), -INT8_C( 103),  INT8_C( 119), -INT8_C(  56),  INT8_C(  98),
         INT8_C( 122), -INT8_C(   6), -INT8_C(  62), -INT8_C(  29), -INT8_C( 114), -INT8_C(  46),  INT8_C(  27),  INT8_C(   5),
        -INT8_C(  40),  INT8_C(  57),  INT8_C(  28),  INT8_C(  54), -INT8_C(   9), -INT8_C(  70), -INT8_C(  69),  INT8_C(  19),
        -INT8_C( 125),  INT8_C(  79),  INT8_C(  36), -INT8_C( 102), -INT8_C( 120),  INT8_C(  91), -INT8_C(  66), -INT8_C(  84),
         INT8_C( 110),  INT8_C(  14), -INT8_C(  46),  INT8_C(   7), -INT8_C( 123), -INT8_C( 102),  INT8_C( 105), -INT8_C(   1) },
      UINT64_C(15431583015668690068),
      {  INT8_C(  55),  INT8_C(  67),  INT8_C(  13),  INT8_C(  46), -INT8_C(   3), -INT8_C(  56),  INT8_C(  65),      INT8_MIN,
         INT8_C(  24),  INT8_C( 101),  INT8_C(  26), -INT8_C(  96), -INT8_C(  64), -INT8_C(  39),  INT8_C(  76),  INT8_C(  47),
        -INT8_C(  25),  INT8_C(  31),  INT8_C(  54),  INT8_C( 108), -INT8_C(  71), -INT8_C(  96),  INT8_C( 107),  INT8_C(  78),
        -INT8_C(  52),  INT8_C(  78),  INT8_C( 112), -INT8_C(  54),  INT8_C(  76), -INT8_C( 104), -INT8_C(  95), -INT8_C( 125),
        -INT8_C(  37), -INT8_C(  82), -INT8_C(  78), -INT8_C(  39),  INT8_C( 118), -INT8_C(  13),  INT8_C(  89), -INT8_C( 114),
         INT8_C(  89),  INT8_C( 116),  INT8_C(  47),  INT8_C(  25),  INT8_C(  77),  INT8_C( 123),  INT8_C(  72),  INT8_C(  52),
        -INT8_C( 102),      INT8_MAX, -INT8_C(  96),  INT8_C(  84),  INT8_C(  31),  INT8_C(  11), -INT8_C(  94), -INT8_C(  21),
         INT8_C(  89),  INT8_C(  18), -INT8_C(  75), -INT8_C(  91), -INT8_C(  86),  INT8_C(  86),  INT8_C(  41), -INT8_C( 122) },
      {  INT8_C(   4), -INT8_C(  37),  INT8_C(  95),  INT8_C( 123), -INT8_C(  50), -INT8_C(  72),  INT8_C(   9),  INT8_C(  39),
         INT8_C(  44),  INT8_C(  56),  INT8_C(  65),  INT8_C( 121), -INT8_C(  76), -INT8_C( 119), -INT8_C(  83),  INT8_C(  78),
         INT8_C(   8),  INT8_C(  77), -INT8_C(  94),  INT8_C(  39),  INT8_C(  89),  INT8_C(  68),  INT8_C(  18), -INT8_C(  78),
         INT8_C(  87), -INT8_C(  56),  INT8_C(  88),  INT8_C(   1),  INT8_C(  30), -INT8_C( 127), -INT8_C( 121),  INT8_C(  35),
         INT8_C(  92), -INT8_C(  26), -INT8_C(  98),  INT8_C(  42), -INT8_C(  97), -INT8_C(  89),  INT8_C(  82), -INT8_C(  53),
        -INT8_C(  32), -INT8_C( 109),  INT8_C(  69), -INT8_C( 108),  INT8_C(  28), -INT8_C(  14), -INT8_C(  30),  INT8_C(  37),
         INT8_C(  64), -INT8_C( 123),  INT8_C(  76), -INT8_C( 103), -INT8_C(  55),  INT8_C(  95),  INT8_C(  75),  INT8_C(  32),
         INT8_C(  39), -INT8_C(  93),  INT8_C(  34),  INT8_C(  69),  INT8_C(  36), -INT8_C(  87),  INT8_C( 104),      INT8_MIN },
      { -INT8_C(  55),  INT8_C(   3),  INT8_C(  13),  INT8_C(  96), -INT8_C(  50), -INT8_C( 108),  INT8_C(  16),      INT8_MIN,
         INT8_C( 119),  INT8_C(   5),  INT8_C(  26), -INT8_C(  96),  INT8_C(  94), -INT8_C( 119), -INT8_C(  98), -INT8_C( 123),
         INT8_C(  28),  INT8_C(  31), -INT8_C( 108),  INT8_C(  17),  INT8_C(  23), -INT8_C(  96),  INT8_C(  18), -INT8_C(  78),
         INT8_C(  35), -INT8_C(  56),  INT8_C(  79),  INT8_C(  38), -INT8_C( 103), -INT8_C( 127), -INT8_C(  56),  INT8_C(  98),
         INT8_C( 122), -INT8_C(  82), -INT8_C(  98), -INT8_C(  39), -INT8_C(  97), -INT8_C(  89),  INT8_C(  82), -INT8_C( 114),
        -INT8_C(  40), -INT8_C( 109),  INT8_C(  47), -INT8_C( 108),  INT8_C(  28), -INT8_C(  14), -INT8_C(  30),  INT8_C(  37),
        -INT8_C( 102), -INT8_C( 123), -INT8_C(  96), -INT8_C( 102), -INT8_C( 120),  INT8_C(  11), -INT8_C(  66), -INT8_C(  84),
         INT8_C( 110), -INT8_C(  93), -INT8_C(  75),  INT8_C(   7), -INT8_C(  86), -INT8_C( 102),  INT8_C(  41),      INT8_MIN } },
    { { -INT8_C( 112),  INT8_C(   6), -INT8_C(  85),  INT8_C(  47), -INT8_C(  82), -INT8_C(   3), -INT8_C(   6), -INT8_C( 114),
        -INT8_C( 112),  INT8_C(  63),  INT8_C(  34), -INT8_C(  84),  INT8_C(  50),  INT8_C(   4), -INT8_C(  47),  INT8_C( 114),
        -INT8_C( 119),  INT8_C(  30),  INT8_C(  11),  INT8_C(  83),  INT8_C( 125),  INT8_C(  86),  INT8_C( 115), -INT8_C(  92),
        -INT8_C(   6), -INT8_C( 107), -INT8_C(  23),  INT8_C(  30),  INT8_C(  63),  INT8_C(  82), -INT8_C(  97), -INT8_C(  49),
         INT8_C(  88),  INT8_C(  74), -INT8_C(   2),  INT8_C(   6),  INT8_C(  71), -INT8_C(   8), -INT8_C( 108), -INT8_C(  41),
         INT8_C(  56), -INT8_C(  74), -INT8_C( 125),  INT8_C( 106), -INT8_C(  69),  INT8_C(  85), -INT8_C(  36),  INT8_C(  68),
         INT8_C( 115), -INT8_C(  25), -INT8_C( 105), -INT8_C(  16),  INT8_C(  61),  INT8_C(  11), -INT8_C( 108),  INT8_C(  55),
        -INT8_C(  96),  INT8_C( 125),  INT8_C(  86), -INT8_C(  33), -INT8_C(  49), -INT8_C(  11), -INT8_C(  82),  INT8_C(  40) },
      UINT64_C(15951120570904390719),
      {  INT8_C( 121), -INT8_C(  32),  INT8_C(  71),  INT8_C(  52),  INT8_C(  53),  INT8_C(  35),  INT8_C( 121), -INT8_C(  88),
         INT8_C(  10),  INT8_C(  16), -INT8_C( 104),  INT8_C(  71),  INT8_C(  27),  INT8_C(  44),      INT8_MAX, -INT8_C(  68),
        -INT8_C(  86), -INT8_C(  43), -INT8_C( 101),  INT8_C( 121), -INT8_C(  54),  INT8_C(  74), -INT8_C(  95),  INT8_C(   9),
        -INT8_C(  10), -INT8_C(  48), -INT8_C( 113), -INT8_C( 101), -INT8_C( 109), -INT8_C(  20),  INT8_C( 120),  INT8_C(  12),
        -INT8_C(  52), -INT8_C(  65),  INT8_C(  65),  INT8_C(   2), -INT8_C(  30), -INT8_C(  70), -INT8_C(  86), -INT8_C(  20),
        -INT8_C(  54),  INT8_C(  67),  INT8_C(  52), -INT8_C(  26),  INT8_C( 111), -INT8_C(  77), -INT8_C(  94),  INT8_C(  25),
        -INT8_C( 120),  INT8_C(  61), -INT8_C( 109),  INT8_C(  82), -INT8_C( 121),  INT8_C(  52),  INT8_C(  91),  INT8_C( 126),
         INT8_C(   4), -INT8_C(  22),  INT8_C(  25), -INT8_C( 105), -INT8_C(  42), -INT8_C( 110), -INT8_C(  92), -INT8_C(  94) },
      {  INT8_C(  81), -INT8_C(  27), -INT8_C(  92),  INT8_C(  52), -INT8_C(  97),  INT8_C(  79),  INT8_C(  32),  INT8_C( 105),
        -INT8_C( 110),  INT8_C(  84),  INT8_C(  79),  INT8_C(   1),  INT8_C(   7), -INT8_C(  15),  INT8_C(  27), -INT8_C( 113),
         INT8_C(  47), -INT8_C(  82), -INT8_C(  31), -INT8_C(  74), -INT8_C(  30),  INT8_C(  60),  INT8_C(  52), -INT8_C(  25),
         INT8_C(  38),  INT8_C(  78),  INT8_C( 126), -INT8_C(   4), -INT8_C(  32),  INT8_C(  34), -INT8_C(  97),  INT8_C(  49),
         INT8_C(   7),  INT8_C(  67),  INT8_C( 101), -INT8_C(  90), -INT8_C( 110), -INT8_C( 122),  INT8_C(  16),  INT8_C(  36),
        -INT8_C(  38),  INT8_C(  95),  INT8_C(  38), -INT8_C(  30),  INT8_C(  81),  INT8_C(  65),  INT8_C( 113),      INT8_MIN,
        -INT8_C(  17),  INT8_C(  83),  INT8_C(  54), -INT8_C(  47), -INT8_C( 113),  INT8_C( 107), -INT8_C(  72), -INT8_C(  74),
        -INT8_C(  71),  INT8_C(  55), -INT8_C(  78), -INT8_C( 103),  INT8_C(  89),  INT8_C(  81), -INT8_C(  54),  INT8_C(  97) },
      {  INT8_C(  81), -INT8_C(  32), -INT8_C(  92),  INT8_C(  52), -INT8_C(  97),  INT8_C(  35), -INT8_C(   6), -INT8_C( 114),
        -INT8_C( 112),  INT8_C(  63), -INT8_C( 104),  INT8_C(   1),  INT8_C(  50), -INT8_C(  15), -INT8_C(  47), -INT8_C( 113),
        -INT8_C( 119), -INT8_C(  82), -INT8_C( 101), -INT8_C(  74),  INT8_C( 125),  INT8_C(  60),  INT8_C( 115), -INT8_C(  92),
        -INT8_C(   6), -INT8_C(  48), -INT8_C( 113),  INT8_C(  30),  INT8_C(  63),  INT8_C(  82), -INT8_C(  97),  INT8_C(  12),
        -INT8_C(  52),  INT8_C(  74),  INT8_C(  65),  INT8_C(   6),  INT8_C(  71), -INT8_C( 122), -INT8_C( 108), -INT8_C(  20),
        -INT8_C(  54),  INT8_C(  67), -INT8_C( 125),  INT8_C( 106), -INT8_C(  69),  INT8_C(  85), -INT8_C(  94),      INT8_MIN,
        -INT8_C( 120), -INT8_C(  25), -INT8_C( 109), -INT8_C(  47), -INT8_C( 121),  INT8_C(  11), -INT8_C(  72),  INT8_C(  55),
        -INT8_C(  71),  INT8_C( 125), -INT8_C(  78), -INT8_C( 105), -INT8_C(  42), -INT8_C(  11), -INT8_C(  92), -INT8_C(  94) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi8(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_epi8(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_epi8");
    easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask64 k;
    const int8_t a[64];
    const int8_t b[64];
    const int8_t r[64];
  } test_vec[] = {
    { UINT64_C(14163406931381216590),
      {  INT8_C( 107),  INT8_C( 119), -INT8_C(  72),  INT8_C(  51), -INT8_C(  83), -INT8_C(  51),  INT8_C( 101),  INT8_C(  96),
        -INT8_C(  35),  INT8_C(  42), -INT8_C( 114), -INT8_C(  41), -INT8_C(  96), -INT8_C(  15), -INT8_C(   3),  INT8_C( 112),
        -INT8_C(   7),  INT8_C(  37),  INT8_C(  15), -INT8_C( 120), -INT8_C(  21),  INT8_C( 107),  INT8_C(  77),  INT8_C(  58),
        -INT8_C(  48), -INT8_C( 107),  INT8_C( 100),  INT8_C(  97),  INT8_C(  29), -INT8_C(  14),  INT8_C(  37), -INT8_C( 120),
         INT8_C( 105), -INT8_C(  35), -INT8_C(  68),  INT8_C(  22), -INT8_C(  86),  INT8_C(  33),  INT8_C( 118), -INT8_C( 121),
         INT8_C(  75),  INT8_C(   4),  INT8_C(  94), -INT8_C(  21), -INT8_C(  10),  INT8_C(  91),  INT8_C(  91), -INT8_C(  17),
             INT8_MIN,  INT8_C( 106),  INT8_C( 119),  INT8_C( 108), -INT8_C(  42), -INT8_C(  60), -INT8_C(  90), -INT8_C(  90),
         INT8_C(  90),  INT8_C(  10),  INT8_C(   7),  INT8_C( 119), -INT8_C(   3),  INT8_C(  44), -INT8_C(   1),  INT8_C( 102) },
      {  INT8_C(   9), -INT8_C(  69),  INT8_C( 125), -INT8_C(  77), -INT8_C(  36), -INT8_C(  13),  INT8_C(  59),  INT8_C(  39),
        -INT8_C(   8), -INT8_C( 103),  INT8_C(  19), -INT8_C(  18), -INT8_C(  11),  INT8_C( 110), -INT8_C(  35),  INT8_C( 117),
        -INT8_C(  39),  INT8_C(  85), -INT8_C(  31), -INT8_C(  81),  INT8_C(  25), -INT8_C( 121),  INT8_C(  85),  INT8_C( 115),
        -INT8_C( 110),  INT8_C(  93), -INT8_C(  22), -INT8_C( 113), -INT8_C( 119), -INT8_C(  22), -INT8_C(  11), -INT8_C( 109),
        -INT8_C(  91),  INT8_C( 114),  INT8_C(  70), -INT8_C( 126),  INT8_C( 102), -INT8_C( 127), -INT8_C(  87),  INT8_C(  94),
         INT8_C(  27), -INT8_C(  68),  INT8_C(  76),  INT8_C(  16),  INT8_C(  43),  INT8_C(  41), -INT8_C( 123),  INT8_C(   4),
         INT8_C( 126),  INT8_C( 103), -INT8_C(  77), -INT8_C( 104), -INT8_C(  18),  INT8_C(   8),  INT8_C(  11),      INT8_MIN,
         INT8_C( 101), -INT8_C(  10),  INT8_C(  15), -INT8_C(  17), -INT8_C(  32),  INT8_C(   5), -INT8_C( 126), -INT8_C( 123) },
      {  INT8_C(   0), -INT8_C(  69), -INT8_C(  72), -INT8_C(  77),  INT8_C(   0),  INT8_C(   0),  INT8_C(  59),  INT8_C(   0),
        -INT8_C(  35),  INT8_C(   0), -INT8_C( 114),  INT8_C(   0),  INT8_C(   0), -INT8_C(  15), -INT8_C(  35),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 120),  INT8_C(   0),  INT8_C(   0),  INT8_C(  77),  INT8_C(   0),
         INT8_C(   0), -INT8_C( 107),  INT8_C(   0), -INT8_C( 113),  INT8_C(   0), -INT8_C(  22),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  86),  INT8_C(   0),  INT8_C(   0), -INT8_C( 121),
         INT8_C(  27), -INT8_C(  68),  INT8_C(  76),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  17),
         INT8_C(   0),  INT8_C( 103), -INT8_C(  77), -INT8_C( 104),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),      INT8_MIN,
         INT8_C(   0),  INT8_C(   0),  INT8_C(   7),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 126), -INT8_C( 123) } },
    { UINT64_C( 7294618956550621303),
      {  INT8_C( 109), -INT8_C( 121),  INT8_C( 117), -INT8_C( 104), -INT8_C(  79), -INT8_C(   6), -INT8_C( 100),  INT8_C(  47),
         INT8_C(  97),  INT8_C(  79), -INT8_C(  57),  INT8_C(  80),  INT8_C(  88), -INT8_C(  45), -INT8_C(  48), -INT8_C(  67),
        -INT8_C(  55), -INT8_C(  32), -INT8_C(  84), -INT8_C(  87), -INT8_C(  27),  INT8_C(  46),  INT8_C(  46),  INT8_C(  92),
        -INT8_C(   9),  INT8_C(  54),  INT8_C(  58),  INT8_C(  65), -INT8_C(  25),  INT8_C( 117), -INT8_C(  90),  INT8_C(  84),
        -INT8_C(   3),  INT8_C(  27), -INT8_C(  19), -INT8_C(  82),  INT8_C(  21), -INT8_C( 119), -INT8_C(  35),  INT8_C( 119),
        -INT8_C(  39), -INT8_C(  91), -INT8_C(  57),  INT8_C(  49),  INT8_C( 120), -INT8_C( 105), -INT8_C(  18),  INT8_C(  65),
         INT8_C( 119), -INT8_C( 101), -INT8_C(  22),  INT8_C(  92), -INT8_C(  55),  INT8_C(  24), -INT8_C(  71), -INT8_C(  64),
         INT8_C(  78), -INT8_C(  13),  INT8_C(   1),  INT8_C(  53),  INT8_C( 104), -INT8_C(  89), -INT8_C( 118),  INT8_C( 101) },
      { -INT8_C(  62),  INT8_C( 119),  INT8_C(  19), -INT8_C(  40),  INT8_C(   0), -INT8_C(  15),  INT8_C(  79), -INT8_C(  39),
        -INT8_C( 106),  INT8_C(  22),  INT8_C(  10),  INT8_C(  14), -INT8_C(  83), -INT8_C(   7),  INT8_C(  79),  INT8_C(  37),
        -INT8_C( 108),  INT8_C(  57), -INT8_C( 127),  INT8_C(  93),  INT8_C(  81),  INT8_C(  58),  INT8_C(  30), -INT8_C(  96),
         INT8_C(  45),  INT8_C(  31), -INT8_C(  43), -INT8_C( 106), -INT8_C(  57),  INT8_C(  95), -INT8_C(   5), -INT8_C( 119),
        -INT8_C(  42),  INT8_C(  15),  INT8_C(  97), -INT8_C(  41),  INT8_C(   0), -INT8_C(  80), -INT8_C(  80), -INT8_C( 106),
        -INT8_C(  58), -INT8_C(  69), -INT8_C(  92),  INT8_C( 116), -INT8_C(  76), -INT8_C(  13), -INT8_C( 103),  INT8_C(  72),
         INT8_C(  44),  INT8_C(  26), -INT8_C(  91),  INT8_C( 125),  INT8_C(  85), -INT8_C(  61),  INT8_C(  29), -INT8_C( 126),
        -INT8_C(  29), -INT8_C(  13),  INT8_C(  24), -INT8_C(  86),  INT8_C(  82),  INT8_C(  20),  INT8_C(  51),  INT8_C(  41) },
      { -INT8_C(  62), -INT8_C( 121),  INT8_C(  19),  INT8_C(   0), -INT8_C(  79), -INT8_C(  15), -INT8_C( 100),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  14),  INT8_C(   0),  INT8_C(   0), -INT8_C(  48), -INT8_C(  67),
        -INT8_C( 108), -INT8_C(  32), -INT8_C( 127),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(   9),  INT8_C(   0), -INT8_C(  43), -INT8_C( 106), -INT8_C(  57),  INT8_C(   0), -INT8_C(  90), -INT8_C( 119),
         INT8_C(   0),  INT8_C(  15),  INT8_C(   0), -INT8_C(  82),  INT8_C(   0),  INT8_C(   0), -INT8_C(  80),  INT8_C(   0),
        -INT8_C(  58),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  76), -INT8_C( 105),  INT8_C(   0),  INT8_C(  65),
         INT8_C(  44), -INT8_C( 101),  INT8_C(   0),  INT8_C(  92), -INT8_C(  55), -INT8_C(  61),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  29),  INT8_C(   0),  INT8_C(   1),  INT8_C(   0),  INT8_C(   0), -INT8_C(  89), -INT8_C( 118),  INT8_C(   0) } },
    { UINT64_C(  916957810133079331),
      {  INT8_C( 107),  INT8_C(  93),      INT8_MIN,  INT8_C(  31),  INT8_C(  80),  INT8_C(  25),  INT8_C( 103),  INT8_C( 124),
         INT8_C(  51),  INT8_C(  13), -INT8_C(   7), -INT8_C( 120), -INT8_C(  48),  INT8_C(  23),  INT8_C(  11), -INT8_C(  77),
         INT8_C(  10),  INT8_C(  35),  INT8_C(  93),  INT8_C(  92),  INT8_C(  55), -INT8_C( 111), -INT8_C( 123),  INT8_C(  90),
         INT8_C(  38), -INT8_C( 123),  INT8_C( 125),  INT8_C( 107),  INT8_C(  54),  INT8_C(  54),  INT8_C( 119), -INT8_C(  95),
        -INT8_C( 109), -INT8_C(   9), -INT8_C(  63), -INT8_C(  29),  INT8_C(  16),  INT8_C(  40),  INT8_C(  95),  INT8_C(  68),
         INT8_C(  53),  INT8_C(  89), -INT8_C(  52),  INT8_C(   6),  INT8_C( 112), -INT8_C(  41), -INT8_C(  71),  INT8_C( 122),
        -INT8_C(   5),  INT8_C(  23), -INT8_C(  42),  INT8_C(  50), -INT8_C(  88),  INT8_C(  92), -INT8_C( 115), -INT8_C(  50),
        -INT8_C(  31),  INT8_C(  10),  INT8_C(  57),  INT8_C(  23),  INT8_C(  65), -INT8_C(  79), -INT8_C(  71), -INT8_C(  44) },
      { -INT8_C(  88),  INT8_C( 122), -INT8_C(  72), -INT8_C(  71), -INT8_C(  94),  INT8_C(  23), -INT8_C(   3), -INT8_C(  40),
         INT8_C( 112), -INT8_C(  55), -INT8_C(  34), -INT8_C(  32), -INT8_C(  95), -INT8_C( 105),  INT8_C(  90), -INT8_C( 100),
        -INT8_C(  82),  INT8_C(  49), -INT8_C(  50),  INT8_C(  86), -INT8_C( 115),  INT8_C(  91),  INT8_C(  36),  INT8_C( 110),
         INT8_C( 102),  INT8_C(  94), -INT8_C( 122), -INT8_C(  89),  INT8_C(  15),  INT8_C(  63),  INT8_C( 123), -INT8_C(  73),
        -INT8_C(  71),  INT8_C(  51),  INT8_C( 112),  INT8_C(  91),  INT8_C(  75),  INT8_C( 109),  INT8_C(  51), -INT8_C(  69),
         INT8_C(  55),  INT8_C(  17), -INT8_C( 100), -INT8_C(  40), -INT8_C(  87), -INT8_C(  10),  INT8_C( 116),  INT8_C(  87),
         INT8_C(  39),  INT8_C(  66), -INT8_C(  82), -INT8_C(  76), -INT8_C(  98), -INT8_C(  46),  INT8_C(  35),  INT8_C(   4),
         INT8_C(  48), -INT8_C(  87), -INT8_C(  85),  INT8_C(  63), -INT8_C(  24),  INT8_C(  38), -INT8_C(   9), -INT8_C(  95) },
      { -INT8_C(  88),  INT8_C(  93),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  23),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  51),  INT8_C(   0), -INT8_C(  34),  INT8_C(   0), -INT8_C(  95),  INT8_C(   0),  INT8_C(   0), -INT8_C( 100),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  38), -INT8_C( 123),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  54),  INT8_C(   0),  INT8_C(   0),
        -INT8_C( 109),  INT8_C(   0), -INT8_C(  63),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  51),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  87), -INT8_C(  41),  INT8_C(   0),  INT8_C(  87),
        -INT8_C(   5),  INT8_C(   0),  INT8_C(   0), -INT8_C(  76), -INT8_C(  98), -INT8_C(  46),  INT8_C(   0), -INT8_C(  50),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  85),  INT8_C(  23),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT64_C(  891766420390307674),
      {  INT8_C(  65), -INT8_C(   4), -INT8_C(  28), -INT8_C(  22), -INT8_C(  13),  INT8_C(  88),  INT8_C(  66),  INT8_C(  26),
        -INT8_C( 102), -INT8_C(  16), -INT8_C(  49),  INT8_C(  56), -INT8_C(  62), -INT8_C(  14),  INT8_C(  60), -INT8_C(  13),
        -INT8_C( 101), -INT8_C(  25),  INT8_C(  50), -INT8_C( 125),  INT8_C(  14),  INT8_C(  41),  INT8_C(  36),  INT8_C( 104),
        -INT8_C( 111),  INT8_C(  32),  INT8_C(  13),  INT8_C( 102),  INT8_C(  80),  INT8_C( 109),  INT8_C( 114), -INT8_C( 110),
         INT8_C( 106),  INT8_C(  86),  INT8_C( 124),  INT8_C(  93), -INT8_C(  82), -INT8_C(  66),  INT8_C( 119),  INT8_C(  72),
        -INT8_C(  82),  INT8_C(  70), -INT8_C( 127),  INT8_C( 113),  INT8_C(  56), -INT8_C(  67),  INT8_C( 100), -INT8_C(  45),
        -INT8_C(  91), -INT8_C( 106),  INT8_C(  86), -INT8_C(  77), -INT8_C(  64),  INT8_C( 122),  INT8_C(  27),  INT8_C(  81),
        -INT8_C( 101),  INT8_C(  40), -INT8_C(  73), -INT8_C(  21), -INT8_C( 107),  INT8_C(  41),  INT8_C( 125), -INT8_C(   1) },
      {      INT8_MAX, -INT8_C(   6),  INT8_C(  92),  INT8_C(  45), -INT8_C(  72), -INT8_C(  44),  INT8_C( 117),  INT8_C( 103),
         INT8_C(  26), -INT8_C(  10), -INT8_C(  40),  INT8_C(  83), -INT8_C(  76),  INT8_C(  60),  INT8_C(  38),  INT8_C(  89),
        -INT8_C(  46),  INT8_C( 125),  INT8_C(  12), -INT8_C( 110), -INT8_C(   9),  INT8_C(  39), -INT8_C(  29), -INT8_C( 110),
         INT8_C(  79), -INT8_C( 102),  INT8_C( 126), -INT8_C(  28), -INT8_C(  61), -INT8_C(   5), -INT8_C(  28),  INT8_C(  66),
        -INT8_C(  11),  INT8_C(  64),  INT8_C( 111), -INT8_C(  82),  INT8_C(  20), -INT8_C(  27),  INT8_C(  21),  INT8_C(  47),
        -INT8_C(  37), -INT8_C(  19), -INT8_C( 126), -INT8_C( 113),  INT8_C(  41), -INT8_C(  88), -INT8_C(  24), -INT8_C(   5),
         INT8_C(  37), -INT8_C(  12), -INT8_C( 114),  INT8_C(  29),  INT8_C(  27),  INT8_C( 113), -INT8_C(  81),  INT8_C( 106),
         INT8_C(  12),  INT8_C(  45),  INT8_C(  79), -INT8_C(  49),  INT8_C(  41),  INT8_C(  51),  INT8_C(  18),  INT8_C(  30) },
      {  INT8_C(   0), -INT8_C(   6),  INT8_C(   0), -INT8_C(  22), -INT8_C(  72),  INT8_C(   0),  INT8_C(  66),  INT8_C(   0),
        -INT8_C( 102), -INT8_C(  16), -INT8_C(  49),  INT8_C(   0),  INT8_C(   0), -INT8_C(  14),  INT8_C(  38),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  12), -INT8_C( 125), -INT8_C(   9),  INT8_C(  39), -INT8_C(  29), -INT8_C( 110),
        -INT8_C( 111),  INT8_C(   0),  INT8_C(  13),  INT8_C(   0),  INT8_C(   0), -INT8_C(   5),  INT8_C(   0), -INT8_C( 110),
        -INT8_C(  11),  INT8_C(   0),  INT8_C( 111),  INT8_C(   0), -INT8_C(  82),  INT8_C(   0),  INT8_C(  21),  INT8_C(  47),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  41), -INT8_C(  88),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 113), -INT8_C(  81),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  73), -INT8_C(  49),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT64_C( 4807558957739311475),
      { -INT8_C(  50),  INT8_C(  57), -INT8_C(  47), -INT8_C(   9), -INT8_C(  31), -INT8_C(  70), -INT8_C(  13),  INT8_C(   7),
        -INT8_C(  82), -INT8_C( 127),  INT8_C(  36), -INT8_C(  54), -INT8_C(  14), -INT8_C(  45),  INT8_C(  52), -INT8_C(   2),
         INT8_C(   1), -INT8_C( 125), -INT8_C(  50),  INT8_C(  42), -INT8_C(  74), -INT8_C(  32),  INT8_C(  72),  INT8_C(  42),
         INT8_C(  97),  INT8_C(  21), -INT8_C(  78), -INT8_C(  56), -INT8_C(  10),  INT8_C( 105),  INT8_C(  10), -INT8_C(  59),
        -INT8_C(  94), -INT8_C(  37), -INT8_C(  68), -INT8_C( 125), -INT8_C( 107), -INT8_C(  81), -INT8_C( 118),  INT8_C(  68),
         INT8_C(  48), -INT8_C(  82),  INT8_C(  14),  INT8_C(  35), -INT8_C( 126),  INT8_C(  66),  INT8_C(  33), -INT8_C( 125),
        -INT8_C(  58), -INT8_C(  17), -INT8_C(  83),  INT8_C( 124), -INT8_C(  49), -INT8_C(  11), -INT8_C(  90),  INT8_C(  49),
         INT8_C(  10),  INT8_C(  88), -INT8_C(   7),  INT8_C(   1), -INT8_C(  63),  INT8_C(   3), -INT8_C(  58),  INT8_C(  99) },
      { -INT8_C(  34), -INT8_C( 126), -INT8_C(  25),  INT8_C( 116),  INT8_C(  50),  INT8_C( 113), -INT8_C(  72),  INT8_C(  98),
         INT8_C(  32), -INT8_C(  58), -INT8_C( 123), -INT8_C(  94),  INT8_C(   8), -INT8_C(  89),  INT8_C(  37), -INT8_C(  50),
        -INT8_C( 106), -INT8_C(  46),  INT8_C(  75),  INT8_C( 102), -INT8_C(  57), -INT8_C(  15), -INT8_C( 105), -INT8_C(  46),
         INT8_C(  74), -INT8_C( 112), -INT8_C(  45),  INT8_C(  11), -INT8_C( 109), -INT8_C( 103),  INT8_C( 111),  INT8_C( 113),
         INT8_C(  27),  INT8_C(  86), -INT8_C(  27),  INT8_C(  77), -INT8_C(  57), -INT8_C(  99), -INT8_C(  80), -INT8_C(  25),
         INT8_C(  99),  INT8_C(  53), -INT8_C( 119),  INT8_C( 108), -INT8_C(  36), -INT8_C(  82),  INT8_C(  58),  INT8_C( 115),
             INT8_MIN, -INT8_C( 123), -INT8_C(  39),  INT8_C(  72),  INT8_C( 119),  INT8_C( 112),  INT8_C(  26), -INT8_C(  63),
         INT8_C(   0), -INT8_C(  19), -INT8_C(  52), -INT8_C( 109), -INT8_C( 122),  INT8_C(  59),  INT8_C(   4), -INT8_C(  95) },
      { -INT8_C(  50), -INT8_C( 126),  INT8_C(   0),  INT8_C(   0), -INT8_C(  31), -INT8_C(  70), -INT8_C(  72),  INT8_C(   0),
        -INT8_C(  82),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  50),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  50),  INT8_C(  42),  INT8_C(   0),  INT8_C(   0), -INT8_C( 105), -INT8_C(  46),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  56),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  59),
         INT8_C(   0), -INT8_C(  37), -INT8_C(  68),  INT8_C(   0),  INT8_C(   0), -INT8_C(  99), -INT8_C( 118),  INT8_C(   0),
         INT8_C(  48),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  82),  INT8_C(  33), -INT8_C( 125),
             INT8_MIN, -INT8_C( 123), -INT8_C(  83),  INT8_C(   0), -INT8_C(  49), -INT8_C(  11),  INT8_C(   0), -INT8_C(  63),
         INT8_C(   0), -INT8_C(  19),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  58),  INT8_C(   0) } },
    { UINT64_C(16951724401100843665),
      { -INT8_C(  44), -INT8_C(  54),  INT8_C(  87), -INT8_C(  79),  INT8_C( 120), -INT8_C( 111),  INT8_C(  36), -INT8_C(   7),
         INT8_C(  23), -INT8_C(   3),  INT8_C(  65), -INT8_C( 114),  INT8_C( 109),  INT8_C(  91),  INT8_C(  79),  INT8_C( 109),
         INT8_C(  72),  INT8_C(  27),  INT8_C(   0), -INT8_C(  50),  INT8_C(  87),  INT8_C(   4),  INT8_C( 111), -INT8_C(  24),
        -INT8_C(  18),  INT8_C(  94),  INT8_C(  65),  INT8_C( 118), -INT8_C(   3), -INT8_C( 126),  INT8_C(  97), -INT8_C(  46),
         INT8_C(  76), -INT8_C(  72), -INT8_C( 125), -INT8_C(  60),  INT8_C(  73), -INT8_C(  89), -INT8_C(  67),  INT8_C(  96),
        -INT8_C(  92), -INT8_C(   2), -INT8_C(  18),  INT8_C(  17),  INT8_C(  89),  INT8_C(  61),  INT8_C( 126), -INT8_C(  95),
         INT8_C(  89),  INT8_C( 126),  INT8_C( 111), -INT8_C(  80), -INT8_C( 126), -INT8_C(  33), -INT8_C( 104),  INT8_C( 113),
         INT8_C(  61), -INT8_C(  38), -INT8_C(  25),  INT8_C(  59),  INT8_C(  92),  INT8_C(  72),  INT8_C(  13), -INT8_C(  88) },
      {  INT8_C(   0), -INT8_C( 112),  INT8_C( 108),  INT8_C(  73),  INT8_C(  55),  INT8_C(  42), -INT8_C(  86), -INT8_C(  37),
         INT8_C(  40), -INT8_C( 104), -INT8_C(  20), -INT8_C( 126), -INT8_C(  42),  INT8_C( 106),  INT8_C(  35),  INT8_C(  47),
        -INT8_C(  24), -INT8_C( 109), -INT8_C(  33),  INT8_C( 106),  INT8_C( 114),  INT8_C( 119), -INT8_C(  37), -INT8_C(  81),
         INT8_C(  81), -INT8_C(  62), -INT8_C(  22), -INT8_C(  83),  INT8_C(  10), -INT8_C(   9),  INT8_C(  85),  INT8_C(  10),
        -INT8_C( 121), -INT8_C(  62),  INT8_C(  84), -INT8_C(  66), -INT8_C(  20), -INT8_C(   2), -INT8_C( 103),  INT8_C(  20),
        -INT8_C( 106), -INT8_C( 123), -INT8_C( 106),  INT8_C( 108), -INT8_C(  17), -INT8_C(  70), -INT8_C( 101), -INT8_C(  41),
         INT8_C(  77),  INT8_C( 122),  INT8_C(  66), -INT8_C(  65), -INT8_C(  14),  INT8_C(  29),  INT8_C( 110),  INT8_C(  67),
        -INT8_C(  32),  INT8_C(  89), -INT8_C(  15), -INT8_C(  22),  INT8_C(  80),  INT8_C(  70), -INT8_C(  11), -INT8_C(  40) },
      { -INT8_C(  44),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  55),  INT8_C(   0),  INT8_C(   0), -INT8_C(  37),
         INT8_C(   0), -INT8_C( 104),  INT8_C(   0), -INT8_C( 126),  INT8_C(   0),  INT8_C(  91),  INT8_C(  35),  INT8_C(  47),
        -INT8_C(  24), -INT8_C( 109), -INT8_C(  33), -INT8_C(  50),  INT8_C(   0),  INT8_C(   4), -INT8_C(  37), -INT8_C(  81),
        -INT8_C(  18),  INT8_C(   0),  INT8_C(   0), -INT8_C(  83), -INT8_C(   3),  INT8_C(   0),  INT8_C(  85),  INT8_C(   0),
        -INT8_C( 121), -INT8_C(  72), -INT8_C( 125),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  20),
        -INT8_C( 106), -INT8_C( 123), -INT8_C( 106),  INT8_C(  17), -INT8_C(  17),  INT8_C(   0),  INT8_C(   0), -INT8_C(  95),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 104),  INT8_C(   0),
        -INT8_C(  32), -INT8_C(  38),  INT8_C(   0), -INT8_C(  22),  INT8_C(   0),  INT8_C(  70), -INT8_C(  11), -INT8_C(  88) } },
    { UINT64_C(15927314642776770824),
      { -INT8_C(  75), -INT8_C(  97),  INT8_C(  74), -INT8_C(  91),  INT8_C(  89), -INT8_C(  27),  INT8_C( 124), -INT8_C(  90),
         INT8_C(  96), -INT8_C(  66),  INT8_C( 101),  INT8_C(  82), -INT8_C(  36), -INT8_C(  44), -INT8_C( 107), -INT8_C(  68),
         INT8_C(  45), -INT8_C( 122), -INT8_C(  90),  INT8_C( 125), -INT8_C(  51), -INT8_C( 101),  INT8_C(  85), -INT8_C(  43),
        -INT8_C(  28), -INT8_C(  20), -INT8_C(  54),  INT8_C(  43),  INT8_C(  28), -INT8_C(  45),  INT8_C(   9), -INT8_C(  47),
         INT8_C( 114),  INT8_C(  83),  INT8_C( 118), -INT8_C(  52),  INT8_C(  56), -INT8_C(  13),  INT8_C( 114), -INT8_C( 104),
        -INT8_C(  79), -INT8_C(  40), -INT8_C(  22), -INT8_C( 115), -INT8_C(  84),      INT8_MIN,  INT8_C(  73), -INT8_C(  39),
         INT8_C(   6), -INT8_C(  16),  INT8_C(  86), -INT8_C(  45), -INT8_C( 117), -INT8_C(  84), -INT8_C(  87),  INT8_C( 112),
        -INT8_C( 104),  INT8_C( 115), -INT8_C( 101), -INT8_C(  76),  INT8_C(  70), -INT8_C(  92), -INT8_C( 123), -INT8_C(  72) },
      { -INT8_C(   9), -INT8_C(   4), -INT8_C( 124),  INT8_C(  48), -INT8_C(  17), -INT8_C(   9), -INT8_C(  56), -INT8_C(  96),
        -INT8_C(  49), -INT8_C(  77),  INT8_C(  46),  INT8_C( 123),  INT8_C(  51),  INT8_C( 119),  INT8_C(  84),  INT8_C(  57),
         INT8_C( 103), -INT8_C(  86),  INT8_C(  13), -INT8_C(  13),  INT8_C(  86), -INT8_C(  74),  INT8_C(  99), -INT8_C(  18),
         INT8_C(  41), -INT8_C(   2), -INT8_C(  94),  INT8_C( 111), -INT8_C(  93),  INT8_C(  40),  INT8_C(  39), -INT8_C( 102),
         INT8_C(  36), -INT8_C(  84), -INT8_C(  54),  INT8_C(  19), -INT8_C(  93), -INT8_C( 109), -INT8_C(  77),  INT8_C( 114),
         INT8_C(  70), -INT8_C(  31), -INT8_C(  19),  INT8_C( 121),  INT8_C(  89),  INT8_C(  65), -INT8_C(  78), -INT8_C(  64),
        -INT8_C(  21), -INT8_C(  65), -INT8_C(  77),  INT8_C(  66),  INT8_C( 117),  INT8_C(  22),  INT8_C(  48), -INT8_C(  98),
         INT8_C(  21), -INT8_C(  45),  INT8_C(  13), -INT8_C(  72), -INT8_C(   5),  INT8_C(  53),  INT8_C(  82),  INT8_C(  31) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  91),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  49),  INT8_C(   0),  INT8_C(   0),  INT8_C(  82),  INT8_C(   0),  INT8_C(   0), -INT8_C( 107),  INT8_C(   0),
         INT8_C(   0), -INT8_C( 122), -INT8_C(  90),  INT8_C(   0), -INT8_C(  51),  INT8_C(   0),  INT8_C(   0), -INT8_C(  43),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  94),  INT8_C(   0), -INT8_C(  93), -INT8_C(  45),  INT8_C(   9), -INT8_C( 102),
         INT8_C(  36), -INT8_C(  84), -INT8_C(  54),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  77),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  84),      INT8_MIN,  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  21),  INT8_C(   0),  INT8_C(   0), -INT8_C(  45),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C( 104),  INT8_C(   0), -INT8_C( 101), -INT8_C(  76), -INT8_C(   5),  INT8_C(   0), -INT8_C( 123), -INT8_C(  72) } },
    { UINT64_C(17795663523895975393),
      { -INT8_C(  57), -INT8_C(  29),  INT8_C( 111),  INT8_C(  32),  INT8_C(  36),  INT8_C(  33), -INT8_C(  32),  INT8_C(  15),
        -INT8_C(  31), -INT8_C( 108),  INT8_C(  81),  INT8_C(  86), -INT8_C(  86), -INT8_C( 126), -INT8_C(  11), -INT8_C(  65),
         INT8_C(  85),  INT8_C(   2),  INT8_C( 119),  INT8_C(  80),  INT8_C(  55), -INT8_C(  54),  INT8_C( 111),  INT8_C(  24),
        -INT8_C(  25), -INT8_C(  95), -INT8_C( 100), -INT8_C( 105), -INT8_C( 122), -INT8_C( 110), -INT8_C( 115),  INT8_C(  77),
         INT8_C( 117), -INT8_C(   4),  INT8_C( 109), -INT8_C( 103),  INT8_C(  29),  INT8_C(  78), -INT8_C(  87), -INT8_C(   2),
        -INT8_C(  30), -INT8_C(   6),  INT8_C(  85), -INT8_C( 116),  INT8_C( 124),  INT8_C(  74),  INT8_C(  76), -INT8_C(  47),
         INT8_C(  76), -INT8_C(  61),  INT8_C(  33), -INT8_C( 124), -INT8_C( 115), -INT8_C( 112), -INT8_C( 100),  INT8_C( 116),
         INT8_C(  49),  INT8_C(  57),  INT8_C(  11), -INT8_C(  72), -INT8_C(  53), -INT8_C( 104),  INT8_C(   5),  INT8_C(  65) },
      { -INT8_C( 108),  INT8_C( 115), -INT8_C(  38), -INT8_C(  78), -INT8_C(  63), -INT8_C( 125), -INT8_C(  80), -INT8_C(  93),
         INT8_C( 126),  INT8_C(   5),  INT8_C(  47), -INT8_C(   6),  INT8_C(  79),  INT8_C( 123), -INT8_C(  52), -INT8_C( 100),
         INT8_C(  63), -INT8_C(  19),  INT8_C(  32), -INT8_C(  52),  INT8_C( 126), -INT8_C(  68),  INT8_C(  65), -INT8_C(  81),
        -INT8_C(  11),  INT8_C(  76),  INT8_C( 103), -INT8_C(  63), -INT8_C(  27),  INT8_C( 109),  INT8_C(   2),  INT8_C( 121),
        -INT8_C(  32), -INT8_C(  36),  INT8_C(  43), -INT8_C(  95),  INT8_C(  96), -INT8_C(  36),  INT8_C(  68), -INT8_C(  34),
        -INT8_C(  31),  INT8_C( 115), -INT8_C(  40),  INT8_C(  49), -INT8_C(  17), -INT8_C(  92), -INT8_C(  51),  INT8_C(  46),
        -INT8_C( 110), -INT8_C(  19), -INT8_C(   6),  INT8_C(  16), -INT8_C(  87),  INT8_C(  59), -INT8_C(  65), -INT8_C(  97),
        -INT8_C( 120),  INT8_C(  39),  INT8_C(  96),  INT8_C( 109), -INT8_C( 108),  INT8_C(  98), -INT8_C(  26),  INT8_C( 116) },
      { -INT8_C( 108),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 125), -INT8_C(  80), -INT8_C(  93),
        -INT8_C(  31),  INT8_C(   0),  INT8_C(  47), -INT8_C(   6), -INT8_C(  86),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  19),  INT8_C(   0),  INT8_C(   0),  INT8_C(  55), -INT8_C(  68),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C( 100),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  77),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  29), -INT8_C(  36),  INT8_C(   0), -INT8_C(  34),
        -INT8_C(  31),  INT8_C(   0), -INT8_C(  40),  INT8_C(   0),  INT8_C(   0), -INT8_C(  92), -INT8_C(  51), -INT8_C(  47),
         INT8_C(   0), -INT8_C(  61), -INT8_C(   6),  INT8_C(   0), -INT8_C( 115), -INT8_C( 112), -INT8_C( 100), -INT8_C(  97),
         INT8_C(   0),  INT8_C(  39),  INT8_C(  11),  INT8_C(   0), -INT8_C( 108), -INT8_C( 104), -INT8_C(  26),  INT8_C(  65) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_epi8(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_epi8");
    easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_min_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint8_t a[64];
    const uint8_t b[64];
    const uint8_t r[64];
  } test_vec[] = {
    { { UINT8_C( 32), UINT8_C(217), UINT8_C( 27), UINT8_C(164), UINT8_C(193), UINT8_C(103), UINT8_C(171), UINT8_C(196),
        UINT8_C(224), UINT8_C(209), UINT8_C(  8), UINT8_C( 68), UINT8_C(152), UINT8_C(243), UINT8_C( 32), UINT8_C( 54),
        UINT8_C(148), UINT8_C(251), UINT8_C(191), UINT8_C(149), UINT8_C(223), UINT8_C( 23), UINT8_C(177), UINT8_C( 37),
        UINT8_C(144), UINT8_C(177), UINT8_C( 93), UINT8_C(117), UINT8_C(236), UINT8_C( 93), UINT8_C(156), UINT8_C( 12),
        UINT8_C( 54), UINT8_C(183), UINT8_C(176), UINT8_C(247), UINT8_C( 31), UINT8_C( 91), UINT8_C(187),    UINT8_MAX,
        UINT8_C( 44), UINT8_C(196), UINT8_C( 67), UINT8_C(197), UINT8_C(183), UINT8_C( 99), UINT8_C(251), UINT8_C( 75),
        UINT8_C( 94), UINT8_C(186), UINT8_C(224), UINT8_C( 61), UINT8_C(210), UINT8_C(145), UINT8_C( 99), UINT8_C( 98),
        UINT8_C( 66), UINT8_C(192), UINT8_C(215), UINT8_C( 47), UINT8_C( 29), UINT8_C(115), UINT8_C( 59), UINT8_C( 83) },
      { UINT8_C( 43), UINT8_C(236), UINT8_C( 75), UINT8_C( 74), UINT8_C( 71), UINT8_C(  6), UINT8_C( 73), UINT8_C(116),
        UINT8_C(202), UINT8_C(140), UINT8_C( 57), UINT8_C(129), UINT8_C(240), UINT8_C( 52), UINT8_C(204), UINT8_C( 78),
        UINT8_C(239), UINT8_C(172), UINT8_C(140), UINT8_C(193), UINT8_C( 62), UINT8_C(239), UINT8_C( 35), UINT8_C(128),
        UINT8_C(175), UINT8_C(250), UINT8_C(175), UINT8_C(204), UINT8_C(110), UINT8_C(235), UINT8_C( 31), UINT8_C(153),
        UINT8_C(215), UINT8_C(106), UINT8_C(227), UINT8_C( 30), UINT8_C(113), UINT8_C( 44), UINT8_C(146), UINT8_C( 59),
        UINT8_C(184), UINT8_C(203), UINT8_C(189), UINT8_C(168), UINT8_C(  0), UINT8_C(137), UINT8_C(247), UINT8_C(239),
        UINT8_C( 54), UINT8_C(131), UINT8_C(176), UINT8_C(116), UINT8_C(114), UINT8_C(211), UINT8_C(244), UINT8_C( 33),
        UINT8_C(205), UINT8_C(164), UINT8_C(237), UINT8_C( 59), UINT8_C(143), UINT8_C( 12), UINT8_C(212), UINT8_C(102) },
      { UINT8_C( 32), UINT8_C(217), UINT8_C( 27), UINT8_C( 74), UINT8_C( 71), UINT8_C(  6), UINT8_C( 73), UINT8_C(116),
        UINT8_C(202), UINT8_C(140), UINT8_C(  8), UINT8_C( 68), UINT8_C(152), UINT8_C( 52), UINT8_C( 32), UINT8_C( 54),
        UINT8_C(148), UINT8_C(172), UINT8_C(140), UINT8_C(149), UINT8_C( 62), UINT8_C( 23), UINT8_C( 35), UINT8_C( 37),
        UINT8_C(144), UINT8_C(177), UINT8_C( 93), UINT8_C(117), UINT8_C(110), UINT8_C( 93), UINT8_C( 31), UINT8_C( 12),
        UINT8_C( 54), UINT8_C(106), UINT8_C(176), UINT8_C( 30), UINT8_C( 31), UINT8_C( 44), UINT8_C(146), UINT8_C( 59),
        UINT8_C( 44), UINT8_C(196), UINT8_C( 67), UINT8_C(168), UINT8_C(  0), UINT8_C( 99), UINT8_C(247), UINT8_C( 75),
        UINT8_C( 54), UINT8_C(131), UINT8_C(176), UINT8_C( 61), UINT8_C(114), UINT8_C(145), UINT8_C( 99), UINT8_C( 33),
        UINT8_C( 66), UINT8_C(164), UINT8_C(215), UINT8_C( 47), UINT8_C( 29), UINT8_C( 12), UINT8_C( 59), UINT8_C( 83) } },
    { { UINT8_C(119), UINT8_C(183), UINT8_C(132), UINT8_C(232), UINT8_C(227), UINT8_C( 23), UINT8_C( 35), UINT8_C(156),
        UINT8_C(226), UINT8_C(224), UINT8_C( 68), UINT8_C(226), UINT8_C(106), UINT8_C( 59), UINT8_C(209), UINT8_C(160),
        UINT8_C(190), UINT8_C(129), UINT8_C( 20), UINT8_C( 48), UINT8_C( 84), UINT8_C(  8), UINT8_C( 81), UINT8_C( 34),
        UINT8_C(172), UINT8_C( 62), UINT8_C( 93), UINT8_C( 59), UINT8_C( 75), UINT8_C( 50), UINT8_C(161), UINT8_C(194),
        UINT8_C(233), UINT8_C( 38), UINT8_C(170), UINT8_C(205), UINT8_C( 61), UINT8_C(205), UINT8_C(105), UINT8_C( 31),
        UINT8_C(174), UINT8_C(173), UINT8_C(  2), UINT8_C( 24), UINT8_C(233), UINT8_C(211), UINT8_C(184), UINT8_C(167),
        UINT8_C( 85), UINT8_C(204), UINT8_C(216), UINT8_C(169), UINT8_C(212), UINT8_C( 41), UINT8_C(203), UINT8_C(129),
        UINT8_C(104), UINT8_C( 41), UINT8_C(188), UINT8_C(179), UINT8_C( 91), UINT8_C( 94), UINT8_C(117), UINT8_C( 68) },
      { UINT8_C(132), UINT8_C( 31), UINT8_C( 17), UINT8_C(193), UINT8_C(236), UINT8_C(122), UINT8_C(224), UINT8_C(154),
        UINT8_C( 40), UINT8_C(226), UINT8_C(178), UINT8_C( 17), UINT8_C(182), UINT8_C(106), UINT8_C(184), UINT8_C( 11),
        UINT8_C( 54), UINT8_C(144), UINT8_C(180), UINT8_C( 11), UINT8_C(186), UINT8_C(128), UINT8_C(140), UINT8_C( 34),
        UINT8_C(169), UINT8_C( 72), UINT8_C(213), UINT8_C(  4), UINT8_C(166), UINT8_C( 74), UINT8_C( 72), UINT8_C( 42),
        UINT8_C(105), UINT8_C( 90), UINT8_C(235), UINT8_C( 85), UINT8_C(212), UINT8_C(204), UINT8_C(240), UINT8_C(252),
        UINT8_C(174), UINT8_C(162), UINT8_C( 13), UINT8_C(100), UINT8_C( 13), UINT8_C(198), UINT8_C(111), UINT8_C( 67),
        UINT8_C( 86), UINT8_C( 36), UINT8_C( 78), UINT8_C( 16), UINT8_C(164), UINT8_C(218), UINT8_C( 50), UINT8_C( 77),
        UINT8_C( 35), UINT8_C(  7), UINT8_C( 81), UINT8_C(201), UINT8_C( 81), UINT8_C(153), UINT8_C(244), UINT8_C(186) },
      { UINT8_C(119), UINT8_C( 31), UINT8_C( 17), UINT8_C(193), UINT8_C(227), UINT8_C( 23), UINT8_C( 35), UINT8_C(154),
        UINT8_C( 40), UINT8_C(224), UINT8_C( 68), UINT8_C( 17), UINT8_C(106), UINT8_C( 59), UINT8_C(184), UINT8_C( 11),
        UINT8_C( 54), UINT8_C(129), UINT8_C( 20), UINT8_C( 11), UINT8_C( 84), UINT8_C(  8), UINT8_C( 81), UINT8_C( 34),
        UINT8_C(169), UINT8_C( 62), UINT8_C( 93), UINT8_C(  4), UINT8_C( 75), UINT8_C( 50), UINT8_C( 72), UINT8_C( 42),
        UINT8_C(105), UINT8_C( 38), UINT8_C(170), UINT8_C( 85), UINT8_C( 61), UINT8_C(204), UINT8_C(105), UINT8_C( 31),
        UINT8_C(174), UINT8_C(162), UINT8_C(  2), UINT8_C( 24), UINT8_C( 13), UINT8_C(198), UINT8_C(111), UINT8_C( 67),
        UINT8_C( 85), UINT8_C( 36), UINT8_C( 78), UINT8_C( 16), UINT8_C(164), UINT8_C( 41), UINT8_C( 50), UINT8_C( 77),
        UINT8_C( 35), UINT8_C(  7), UINT8_C( 81), UINT8_C(179), UINT8_C( 81), UINT8_C( 94), UINT8_C(117), UINT8_C( 68) } },
    { { UINT8_C(243), UINT8_C(223), UINT8_C( 16), UINT8_C(200), UINT8_C(171), UINT8_C(  0), UINT8_C(196), UINT8_C( 90),
        UINT8_C(162), UINT8_C(210), UINT8_C(190), UINT8_C(175), UINT8_C(152), UINT8_C( 46), UINT8_C(243), UINT8_C(238),
        UINT8_C( 82), UINT8_C( 65),    UINT8_MAX, UINT8_C(246), UINT8_C( 28), UINT8_C( 49), UINT8_C( 67), UINT8_C( 63),
        UINT8_C( 57), UINT8_C(148), UINT8_C(  8), UINT8_C(138), UINT8_C( 45), UINT8_C(252), UINT8_C( 69), UINT8_C( 33),
        UINT8_C(220), UINT8_C( 85), UINT8_C(233), UINT8_C(135), UINT8_C( 85), UINT8_C(173), UINT8_C(225), UINT8_C(247),
        UINT8_C(127), UINT8_C(160), UINT8_C(167), UINT8_C( 23), UINT8_C(206), UINT8_C(154), UINT8_C(  6), UINT8_C( 32),
        UINT8_C(219), UINT8_C(  5), UINT8_C( 22), UINT8_C(247), UINT8_C( 54), UINT8_C( 89), UINT8_C( 54), UINT8_C(111),
        UINT8_C(237), UINT8_C( 63), UINT8_C(250), UINT8_C( 26), UINT8_C( 59), UINT8_C( 63), UINT8_C( 59), UINT8_C( 23) },
      { UINT8_C(148), UINT8_C( 36), UINT8_C(159), UINT8_C(233), UINT8_C(210), UINT8_C(128), UINT8_C(224), UINT8_C( 81),
        UINT8_C( 32), UINT8_C(135), UINT8_C(105), UINT8_C(238), UINT8_C( 33), UINT8_C(111), UINT8_C( 14), UINT8_C(253),
        UINT8_C(116), UINT8_C( 36), UINT8_C(244), UINT8_C(170), UINT8_C(125), UINT8_C( 43), UINT8_C( 26), UINT8_C(106),
        UINT8_C(106), UINT8_C( 20), UINT8_C(133), UINT8_C(165), UINT8_C( 83), UINT8_C(192), UINT8_C(189), UINT8_C(231),
        UINT8_C(229), UINT8_C( 92), UINT8_C(208), UINT8_C(183), UINT8_C(220), UINT8_C(176), UINT8_C(  8), UINT8_C(253),
        UINT8_C( 56), UINT8_C(113), UINT8_C(235), UINT8_C( 89), UINT8_C(224), UINT8_C(250), UINT8_C( 86), UINT8_C( 84),
        UINT8_C( 30), UINT8_C( 75),    UINT8_MAX, UINT8_C(156), UINT8_C(118), UINT8_C( 25), UINT8_C(  6), UINT8_C(224),
        UINT8_C( 45), UINT8_C(139), UINT8_C(133), UINT8_C(128), UINT8_C( 76), UINT8_C( 66), UINT8_C(103), UINT8_C( 49) },
      { UINT8_C(148), UINT8_C( 36), UINT8_C( 16), UINT8_C(200), UINT8_C(171), UINT8_C(  0), UINT8_C(196), UINT8_C( 81),
        UINT8_C( 32), UINT8_C(135), UINT8_C(105), UINT8_C(175), UINT8_C( 33), UINT8_C( 46), UINT8_C( 14), UINT8_C(238),
        UINT8_C( 82), UINT8_C( 36), UINT8_C(244), UINT8_C(170), UINT8_C( 28), UINT8_C( 43), UINT8_C( 26), UINT8_C( 63),
        UINT8_C( 57), UINT8_C( 20), UINT8_C(  8), UINT8_C(138), UINT8_C( 45), UINT8_C(192), UINT8_C( 69), UINT8_C( 33),
        UINT8_C(220), UINT8_C( 85), UINT8_C(208), UINT8_C(135), UINT8_C( 85), UINT8_C(173), UINT8_C(  8), UINT8_C(247),
        UINT8_C( 56), UINT8_C(113), UINT8_C(167), UINT8_C( 23), UINT8_C(206), UINT8_C(154), UINT8_C(  6), UINT8_C( 32),
        UINT8_C( 30), UINT8_C(  5), UINT8_C( 22), UINT8_C(156), UINT8_C( 54), UINT8_C( 25), UINT8_C(  6), UINT8_C(111),
        UINT8_C( 45), UINT8_C( 63), UINT8_C(133), UINT8_C( 26), UINT8_C( 59), UINT8_C( 63), UINT8_C( 59), UINT8_C( 23) } },
    { { UINT8_C(158), UINT8_C( 55), UINT8_C(232), UINT8_C(123), UINT8_C(231), UINT8_C(240), UINT8_C(120), UINT8_C( 31),
        UINT8_C( 98), UINT8_C( 99), UINT8_C(121), UINT8_C( 66), UINT8_C( 93), UINT8_C(207), UINT8_C(151), UINT8_C(124),
        UINT8_C( 26), UINT8_C(150), UINT8_C( 24), UINT8_C(144), UINT8_C(175), UINT8_C( 30), UINT8_C(112), UINT8_C(220),
        UINT8_C(170), UINT8_C(246), UINT8_C( 92), UINT8_C(246), UINT8_C( 56), UINT8_C(195), UINT8_C( 39), UINT8_C(215),
        UINT8_C(250), UINT8_C( 15), UINT8_C( 82), UINT8_C(225),    UINT8_MAX, UINT8_C(202), UINT8_C(  1), UINT8_C( 97),
        UINT8_C( 45), UINT8_C(122), UINT8_C(164), UINT8_C(139), UINT8_C( 73), UINT8_C( 59), UINT8_C(  7), UINT8_C(100),
        UINT8_C(209), UINT8_C( 31), UINT8_C(244), UINT8_C(128), UINT8_C( 61), UINT8_C(101), UINT8_C( 92), UINT8_C(231),
        UINT8_C( 91), UINT8_C(184), UINT8_C(221), UINT8_C(147), UINT8_C(123), UINT8_C(  4), UINT8_C(106), UINT8_C(117) },
      { UINT8_C( 19), UINT8_C(188), UINT8_C( 86), UINT8_C( 19), UINT8_C(134), UINT8_C( 87), UINT8_C(116), UINT8_C(180),
        UINT8_C(209), UINT8_C( 24), UINT8_C( 63), UINT8_C( 27), UINT8_C( 83), UINT8_C( 70), UINT8_C(127), UINT8_C( 36),
        UINT8_C(101), UINT8_C(115), UINT8_C(164), UINT8_C(162), UINT8_C(216), UINT8_C(  0), UINT8_C(138), UINT8_C( 51),
        UINT8_C(184), UINT8_C(103), UINT8_C(199), UINT8_C( 51), UINT8_C(108), UINT8_C( 49), UINT8_C(168), UINT8_C(127),
        UINT8_C(238),    UINT8_MAX, UINT8_C(146), UINT8_C(116), UINT8_C( 86), UINT8_C(  7), UINT8_C( 40), UINT8_C( 40),
        UINT8_C( 31), UINT8_C(103), UINT8_C( 67), UINT8_C(115), UINT8_C(173), UINT8_C(194), UINT8_C(151), UINT8_C( 18),
        UINT8_C( 53), UINT8_C( 60), UINT8_C(181), UINT8_C( 14), UINT8_C( 60), UINT8_C( 63), UINT8_C( 65), UINT8_C(245),
        UINT8_C(166), UINT8_C(  8), UINT8_C( 40), UINT8_C( 18), UINT8_C( 58), UINT8_C(209), UINT8_C(146), UINT8_C( 40) },
      { UINT8_C( 19), UINT8_C( 55), UINT8_C( 86), UINT8_C( 19), UINT8_C(134), UINT8_C( 87), UINT8_C(116), UINT8_C( 31),
        UINT8_C( 98), UINT8_C( 24), UINT8_C( 63), UINT8_C( 27), UINT8_C( 83), UINT8_C( 70), UINT8_C(127), UINT8_C( 36),
        UINT8_C( 26), UINT8_C(115), UINT8_C( 24), UINT8_C(144), UINT8_C(175), UINT8_C(  0), UINT8_C(112), UINT8_C( 51),
        UINT8_C(170), UINT8_C(103), UINT8_C( 92), UINT8_C( 51), UINT8_C( 56), UINT8_C( 49), UINT8_C( 39), UINT8_C(127),
        UINT8_C(238), UINT8_C( 15), UINT8_C( 82), UINT8_C(116), UINT8_C( 86), UINT8_C(  7), UINT8_C(  1), UINT8_C( 40),
        UINT8_C( 31), UINT8_C(103), UINT8_C( 67), UINT8_C(115), UINT8_C( 73), UINT8_C( 59), UINT8_C(  7), UINT8_C( 18),
        UINT8_C( 53), UINT8_C( 31), UINT8_C(181), UINT8_C( 14), UINT8_C( 60), UINT8_C( 63), UINT8_C( 65), UINT8_C(231),
        UINT8_C( 91), UINT8_C(  8), UINT8_C( 40), UINT8_C( 18), UINT8_C( 58), UINT8_C(  4), UINT8_C(106), UINT8_C( 40) } },
    { { UINT8_C(208), UINT8_C( 36), UINT8_C(156), UINT8_C( 38), UINT8_C( 43), UINT8_C(197), UINT8_C( 78), UINT8_C( 75),
        UINT8_C( 44), UINT8_C(145), UINT8_C(190), UINT8_C(218), UINT8_C( 83), UINT8_C( 85), UINT8_C(236), UINT8_C(137),
        UINT8_C(145), UINT8_C(161), UINT8_C(151), UINT8_C(206), UINT8_C(224), UINT8_C(216), UINT8_C(195), UINT8_C(135),
        UINT8_C(225), UINT8_C(235), UINT8_C(153), UINT8_C( 27), UINT8_C(188), UINT8_C( 43), UINT8_C( 67), UINT8_C(140),
        UINT8_C( 80), UINT8_C(223), UINT8_C(179), UINT8_C(123), UINT8_C(164), UINT8_C(  1), UINT8_C(198), UINT8_C(209),
        UINT8_C(147), UINT8_C(132), UINT8_C(171), UINT8_C(230), UINT8_C(218), UINT8_C(151), UINT8_C(111), UINT8_C(107),
        UINT8_C( 57), UINT8_C(  6), UINT8_C( 57), UINT8_C( 25), UINT8_C(223), UINT8_C(252), UINT8_C(160), UINT8_C(192),
        UINT8_C(232), UINT8_C( 58), UINT8_C(219), UINT8_C(164), UINT8_C(101), UINT8_C( 30), UINT8_C( 49), UINT8_C(181) },
      { UINT8_C(253), UINT8_C(228), UINT8_C( 49), UINT8_C(162), UINT8_C(229), UINT8_C(247), UINT8_C(115), UINT8_C(120),
        UINT8_C(124), UINT8_C( 30), UINT8_C( 95), UINT8_C( 86), UINT8_C(181), UINT8_C(206), UINT8_C(193), UINT8_C(238),
        UINT8_C(213), UINT8_C(251), UINT8_C(  8), UINT8_C(180), UINT8_C(247), UINT8_C(168), UINT8_C(116), UINT8_C(223),
        UINT8_C(226), UINT8_C( 79), UINT8_C(132), UINT8_C( 72), UINT8_C(109), UINT8_C(181), UINT8_C(253), UINT8_C(106),
        UINT8_C(153), UINT8_C( 46), UINT8_C( 12), UINT8_C(126), UINT8_C( 38), UINT8_C(127), UINT8_C(247), UINT8_C(162),
        UINT8_C(157), UINT8_C( 86), UINT8_C(248), UINT8_C( 83), UINT8_C( 36), UINT8_C(185), UINT8_C( 65), UINT8_C(249),
        UINT8_C(180), UINT8_C( 73), UINT8_C(173), UINT8_C(172), UINT8_C(242), UINT8_C( 33), UINT8_C(139), UINT8_C(212),
        UINT8_C(112), UINT8_C( 15), UINT8_C( 28), UINT8_C(221), UINT8_C(196), UINT8_C( 26), UINT8_C( 72), UINT8_C( 93) },
      { UINT8_C(208), UINT8_C( 36), UINT8_C( 49), UINT8_C( 38), UINT8_C( 43), UINT8_C(197), UINT8_C( 78), UINT8_C( 75),
        UINT8_C( 44), UINT8_C( 30), UINT8_C( 95), UINT8_C( 86), UINT8_C( 83), UINT8_C( 85), UINT8_C(193), UINT8_C(137),
        UINT8_C(145), UINT8_C(161), UINT8_C(  8), UINT8_C(180), UINT8_C(224), UINT8_C(168), UINT8_C(116), UINT8_C(135),
        UINT8_C(225), UINT8_C( 79), UINT8_C(132), UINT8_C( 27), UINT8_C(109), UINT8_C( 43), UINT8_C( 67), UINT8_C(106),
        UINT8_C( 80), UINT8_C( 46), UINT8_C( 12), UINT8_C(123), UINT8_C( 38), UINT8_C(  1), UINT8_C(198), UINT8_C(162),
        UINT8_C(147), UINT8_C( 86), UINT8_C(171), UINT8_C( 83), UINT8_C( 36), UINT8_C(151), UINT8_C( 65), UINT8_C(107),
        UINT8_C( 57), UINT8_C(  6), UINT8_C( 57), UINT8_C( 25), UINT8_C(223), UINT8_C( 33), UINT8_C(139), UINT8_C(192),
        UINT8_C(112), UINT8_C( 15), UINT8_C( 28), UINT8_C(164), UINT8_C(101), UINT8_C( 26), UINT8_C( 49), UINT8_C( 93) } },
    { { UINT8_C( 72), UINT8_C( 84), UINT8_C(220), UINT8_C(110), UINT8_C(212), UINT8_C(211), UINT8_C( 16), UINT8_C(113),
        UINT8_C( 41), UINT8_C(  8), UINT8_C(196), UINT8_C( 77), UINT8_C(194), UINT8_C(  6), UINT8_C( 71), UINT8_C(118),
        UINT8_C( 79), UINT8_C(244), UINT8_C( 34), UINT8_C( 65), UINT8_C( 22), UINT8_C(174), UINT8_C( 22), UINT8_C(134),
        UINT8_C(189), UINT8_C( 50), UINT8_C(100), UINT8_C(130), UINT8_C( 76), UINT8_C(172), UINT8_C(223), UINT8_C(149),
        UINT8_C(  0), UINT8_C(187), UINT8_C(  3), UINT8_C(212), UINT8_C(142), UINT8_C( 20), UINT8_C( 70), UINT8_C(183),
        UINT8_C( 28), UINT8_C( 10), UINT8_C(  5), UINT8_C(222), UINT8_C( 16), UINT8_C( 76), UINT8_C( 85), UINT8_C( 96),
        UINT8_C( 64), UINT8_C(119), UINT8_C(161), UINT8_C( 86), UINT8_C( 37), UINT8_C(183), UINT8_C(221), UINT8_C(227),
        UINT8_C(234), UINT8_C( 65), UINT8_C(101), UINT8_C( 54), UINT8_C(237), UINT8_C( 68), UINT8_C(203), UINT8_C(237) },
      { UINT8_C(  0), UINT8_C(207), UINT8_C(194), UINT8_C(142), UINT8_C(227), UINT8_C(  8), UINT8_C( 70),    UINT8_MAX,
        UINT8_C( 18), UINT8_C( 75), UINT8_C(222), UINT8_C( 35), UINT8_C(151), UINT8_C( 51), UINT8_C(131), UINT8_C(215),
        UINT8_C(170), UINT8_C( 36), UINT8_C( 46), UINT8_C(208), UINT8_C(220), UINT8_C( 11), UINT8_C(179), UINT8_C(198),
        UINT8_C( 76), UINT8_C( 24), UINT8_C(252), UINT8_C( 57), UINT8_C( 92), UINT8_C(200), UINT8_C( 38), UINT8_C( 92),
        UINT8_C(151), UINT8_C(232), UINT8_C(235), UINT8_C(122), UINT8_C(240), UINT8_C( 49), UINT8_C(121), UINT8_C(  3),
        UINT8_C(124), UINT8_C( 87), UINT8_C( 38), UINT8_C( 19), UINT8_C(138), UINT8_C(169), UINT8_C(234), UINT8_C( 53),
        UINT8_C(205), UINT8_C( 24), UINT8_C(  5), UINT8_C(169), UINT8_C( 35), UINT8_C(184), UINT8_C(111), UINT8_C(111),
        UINT8_C(208), UINT8_C(108), UINT8_C(168), UINT8_C( 44), UINT8_C( 52), UINT8_C(207), UINT8_C(137), UINT8_C(203) },
      { UINT8_C(  0), UINT8_C( 84), UINT8_C(194), UINT8_C(110), UINT8_C(212), UINT8_C(  8), UINT8_C( 16), UINT8_C(113),
        UINT8_C( 18), UINT8_C(  8), UINT8_C(196), UINT8_C( 35), UINT8_C(151), UINT8_C(  6), UINT8_C( 71), UINT8_C(118),
        UINT8_C( 79), UINT8_C( 36), UINT8_C( 34), UINT8_C( 65), UINT8_C( 22), UINT8_C( 11), UINT8_C( 22), UINT8_C(134),
        UINT8_C( 76), UINT8_C( 24), UINT8_C(100), UINT8_C( 57), UINT8_C( 76), UINT8_C(172), UINT8_C( 38), UINT8_C( 92),
        UINT8_C(  0), UINT8_C(187), UINT8_C(  3), UINT8_C(122), UINT8_C(142), UINT8_C( 20), UINT8_C( 70), UINT8_C(  3),
        UINT8_C( 28), UINT8_C( 10), UINT8_C(  5), UINT8_C( 19), UINT8_C( 16), UINT8_C( 76), UINT8_C( 85), UINT8_C( 53),
        UINT8_C( 64), UINT8_C( 24), UINT8_C(  5), UINT8_C( 86), UINT8_C( 35), UINT8_C(183), UINT8_C(111), UINT8_C(111),
        UINT8_C(208), UINT8_C( 65), UINT8_C(101), UINT8_C( 44), UINT8_C( 52), UINT8_C( 68), UINT8_C(137), UINT8_C(203) } },
    { { UINT8_C(183), UINT8_C(116), UINT8_C( 69), UINT8_C(168), UINT8_C(165), UINT8_C(190), UINT8_C(171), UINT8_C( 33),
        UINT8_C( 22), UINT8_C(209), UINT8_C( 52), UINT8_C(160), UINT8_C(122), UINT8_C( 30), UINT8_C(213), UINT8_C( 71),
        UINT8_C( 55), UINT8_C(218), UINT8_C(241), UINT8_C( 90), UINT8_C(146), UINT8_C( 96), UINT8_C(202), UINT8_C( 98),
        UINT8_C(204), UINT8_C(114), UINT8_C(143), UINT8_C(  0), UINT8_C( 65), UINT8_C( 24), UINT8_C(203), UINT8_C(249),
        UINT8_C(140), UINT8_C( 16), UINT8_C(161), UINT8_C( 49), UINT8_C(207), UINT8_C( 76), UINT8_C( 82), UINT8_C(229),
        UINT8_C( 29), UINT8_C(134), UINT8_C(133), UINT8_C(151), UINT8_C(164), UINT8_C( 91), UINT8_C(222), UINT8_C(219),
        UINT8_C( 53), UINT8_C(207), UINT8_C( 54), UINT8_C(200), UINT8_C( 48), UINT8_C(  0), UINT8_C( 42), UINT8_C(252),
        UINT8_C(114), UINT8_C(185), UINT8_C(253), UINT8_C(180), UINT8_C(209), UINT8_C(200), UINT8_C(173), UINT8_C( 93) },
      { UINT8_C(217), UINT8_C( 78), UINT8_C(142), UINT8_C(168), UINT8_C(154), UINT8_C(224), UINT8_C(141), UINT8_C(183),
        UINT8_C(102), UINT8_C( 18), UINT8_C( 78), UINT8_C( 11), UINT8_C(109), UINT8_C( 44), UINT8_C(230), UINT8_C(163),
        UINT8_C(252), UINT8_C( 28), UINT8_C(107), UINT8_C( 44), UINT8_C( 28), UINT8_C(149), UINT8_C( 40), UINT8_C(143),
        UINT8_C( 79), UINT8_C( 37), UINT8_C( 67), UINT8_C( 32), UINT8_C(238), UINT8_C(240), UINT8_C(126), UINT8_C(199),
        UINT8_C( 62), UINT8_C( 12), UINT8_C(111), UINT8_C(216), UINT8_C(237), UINT8_C(252), UINT8_C(143), UINT8_C( 83),
        UINT8_C( 14), UINT8_C(221), UINT8_C( 94), UINT8_C(124), UINT8_C(  9), UINT8_C( 69), UINT8_C( 31), UINT8_C(  5),
        UINT8_C( 97), UINT8_C(138), UINT8_C( 49), UINT8_C(126), UINT8_C( 31), UINT8_C( 90), UINT8_C( 13), UINT8_C(110),
        UINT8_C(127), UINT8_C( 80), UINT8_C(143), UINT8_C(109), UINT8_C( 64), UINT8_C( 13), UINT8_C( 52), UINT8_C(126) },
      { UINT8_C(183), UINT8_C( 78), UINT8_C( 69), UINT8_C(168), UINT8_C(154), UINT8_C(190), UINT8_C(141), UINT8_C( 33),
        UINT8_C( 22), UINT8_C( 18), UINT8_C( 52), UINT8_C( 11), UINT8_C(109), UINT8_C( 30), UINT8_C(213), UINT8_C( 71),
        UINT8_C( 55), UINT8_C( 28), UINT8_C(107), UINT8_C( 44), UINT8_C( 28), UINT8_C( 96), UINT8_C( 40), UINT8_C( 98),
        UINT8_C( 79), UINT8_C( 37), UINT8_C( 67), UINT8_C(  0), UINT8_C( 65), UINT8_C( 24), UINT8_C(126), UINT8_C(199),
        UINT8_C( 62), UINT8_C( 12), UINT8_C(111), UINT8_C( 49), UINT8_C(207), UINT8_C( 76), UINT8_C( 82), UINT8_C( 83),
        UINT8_C( 14), UINT8_C(134), UINT8_C( 94), UINT8_C(124), UINT8_C(  9), UINT8_C( 69), UINT8_C( 31), UINT8_C(  5),
        UINT8_C( 53), UINT8_C(138), UINT8_C( 49), UINT8_C(126), UINT8_C( 31), UINT8_C(  0), UINT8_C( 13), UINT8_C(110),
        UINT8_C(114), UINT8_C( 80), UINT8_C(143), UINT8_C(109), UINT8_C( 64), UINT8_C( 13), UINT8_C( 52), UINT8_C( 93) } },
    { { UINT8_C( 25), UINT8_C(163), UINT8_C( 86), UINT8_C(  6), UINT8_C(159), UINT8_C(229), UINT8_C( 90), UINT8_C(174),
        UINT8_C(194), UINT8_C(184), UINT8_C( 42), UINT8_C(203), UINT8_C(253), UINT8_C( 73), UINT8_C(209), UINT8_C( 95),
        UINT8_C(211), UINT8_C(  2), UINT8_C(221), UINT8_C(242), UINT8_C( 92), UINT8_C(234), UINT8_C( 97), UINT8_C(220),
        UINT8_C( 58), UINT8_C(240), UINT8_C( 73), UINT8_C(122), UINT8_C(253), UINT8_C(126), UINT8_C(248), UINT8_C( 22),
        UINT8_C( 33), UINT8_C( 78), UINT8_C( 29), UINT8_C(193), UINT8_C( 51), UINT8_C(119), UINT8_C(111), UINT8_C(245),
        UINT8_C( 47), UINT8_C(153), UINT8_C(192), UINT8_C( 45), UINT8_C(226), UINT8_C(145), UINT8_C(140), UINT8_C(181),
        UINT8_C(148), UINT8_C(105), UINT8_C(167), UINT8_C(240), UINT8_C( 83), UINT8_C(  8), UINT8_C(204), UINT8_C(141),
        UINT8_C(248), UINT8_C( 22), UINT8_C(  7), UINT8_C(245), UINT8_C(148),    UINT8_MAX, UINT8_C( 12), UINT8_C(181) },
      { UINT8_C( 77), UINT8_C( 41), UINT8_C(118), UINT8_C(128), UINT8_C(160), UINT8_C(229), UINT8_C(117), UINT8_C(207),
        UINT8_C(126), UINT8_C( 53), UINT8_C(252), UINT8_C( 96), UINT8_C(199), UINT8_C(136), UINT8_C( 21), UINT8_C( 91),
        UINT8_C(241), UINT8_C(189), UINT8_C( 75), UINT8_C( 68), UINT8_C(197), UINT8_C( 24), UINT8_C(209), UINT8_C(190),
        UINT8_C( 46), UINT8_C(216), UINT8_C(179), UINT8_C(194), UINT8_C(215), UINT8_C(191), UINT8_C(119), UINT8_C( 36),
        UINT8_C(232), UINT8_C(238), UINT8_C(164), UINT8_C(136), UINT8_C(211), UINT8_C( 25), UINT8_C( 88), UINT8_C( 82),
        UINT8_C( 79), UINT8_C( 84), UINT8_C(178), UINT8_C( 22), UINT8_C(221), UINT8_C(200), UINT8_C(113), UINT8_C(206),
        UINT8_C(133), UINT8_C(188), UINT8_C( 19), UINT8_C( 74), UINT8_C(212), UINT8_C(228), UINT8_C(  8), UINT8_C(  2),
        UINT8_C(189), UINT8_C(188), UINT8_C(196), UINT8_C(148), UINT8_C(123), UINT8_C( 60), UINT8_C(185), UINT8_C(100) },
      { UINT8_C( 25), UINT8_C( 41), UINT8_C( 86), UINT8_C(  6), UINT8_C(159), UINT8_C(229), UINT8_C( 90), UINT8_C(174),
        UINT8_C(126), UINT8_C( 53), UINT8_C( 42), UINT8_C( 96), UINT8_C(199), UINT8_C( 73), UINT8_C( 21), UINT8_C( 91),
        UINT8_C(211), UINT8_C(  2), UINT8_C( 75), UINT8_C( 68), UINT8_C( 92), UINT8_C( 24), UINT8_C( 97), UINT8_C(190),
        UINT8_C( 46), UINT8_C(216), UINT8_C( 73), UINT8_C(122), UINT8_C(215), UINT8_C(126), UINT8_C(119), UINT8_C( 22),
        UINT8_C( 33), UINT8_C( 78), UINT8_C( 29), UINT8_C(136), UINT8_C( 51), UINT8_C( 25), UINT8_C( 88), UINT8_C( 82),
        UINT8_C( 47), UINT8_C( 84), UINT8_C(178), UINT8_C( 22), UINT8_C(221), UINT8_C(145), UINT8_C(113), UINT8_C(181),
        UINT8_C(133), UINT8_C(105), UINT8_C( 19), UINT8_C( 74), UINT8_C( 83), UINT8_C(  8), UINT8_C(  8), UINT8_C(  2),
        UINT8_C(189), UINT8_C( 22), UINT8_C(  7), UINT8_C(148), UINT8_C(123), UINT8_C( 60), UINT8_C( 12), UINT8_C(100) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_min_epu8(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_epu8");
    easysimd_test_x86_assert_equal_u8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_min_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int8_t src[64];
    const easysimd__mmask64 k;
    const uint8_t a[64];
    const uint8_t b[64];
    const uint8_t r[64];
  } test_vec[] = {
    { {  INT8_C(  25), -INT8_C( 115),  INT8_C(  79), -INT8_C( 105), -INT8_C( 118), -INT8_C(  48),  INT8_C(  60),  INT8_C( 124),
         INT8_C(  36),  INT8_C(  36),  INT8_C(  79), -INT8_C(  57),  INT8_C(  83),  INT8_C( 122), -INT8_C(  25),  INT8_C( 107),
        -INT8_C(  54),  INT8_C(  79), -INT8_C( 111),  INT8_C( 105),  INT8_C(  61), -INT8_C(   3), -INT8_C(  33), -INT8_C(  44),
        -INT8_C(  34),  INT8_C(  97),  INT8_C(  26),  INT8_C(   1),  INT8_C(  36), -INT8_C(  49), -INT8_C(  78),  INT8_C(  61),
         INT8_C(  92),  INT8_C(   1), -INT8_C(  44), -INT8_C(  26), -INT8_C(  47),  INT8_C(  16),  INT8_C(  99), -INT8_C(  10),
         INT8_C(  52), -INT8_C(  78), -INT8_C(  67), -INT8_C( 121),  INT8_C(  44), -INT8_C(  92), -INT8_C(  14), -INT8_C(  10),
        -INT8_C(  13), -INT8_C( 124),  INT8_C(  95),  INT8_C(  49), -INT8_C( 127),  INT8_C(  63),  INT8_C(   5),  INT8_C(  96),
        -INT8_C(  96),  INT8_C(  31),  INT8_C(  97), -INT8_C(  60), -INT8_C(  18),  INT8_C(  19),  INT8_C(   2),  INT8_C(  74) },
      UINT64_C( 2007643257620780565),
      { UINT8_C( 70), UINT8_C(153), UINT8_C(163), UINT8_C(114), UINT8_C( 61), UINT8_C(149), UINT8_C(104), UINT8_C( 49),
        UINT8_C( 25), UINT8_C(199), UINT8_C( 98), UINT8_C(155), UINT8_C(  6), UINT8_C(103), UINT8_C(251), UINT8_C(167),
        UINT8_C(135), UINT8_C( 92), UINT8_C(107), UINT8_C(117), UINT8_C(112), UINT8_C(109), UINT8_C(192), UINT8_C(133),
        UINT8_C( 68), UINT8_C(241), UINT8_C(107), UINT8_C( 43), UINT8_C(133), UINT8_C( 72), UINT8_C( 70), UINT8_C(203),
        UINT8_C(225), UINT8_C(233), UINT8_C( 61), UINT8_C( 31), UINT8_C(127), UINT8_C(165), UINT8_C( 80), UINT8_C(152),
        UINT8_C(108), UINT8_C(178), UINT8_C( 51), UINT8_C(115), UINT8_C( 25), UINT8_C( 46), UINT8_C( 26), UINT8_C(160),
        UINT8_C(139), UINT8_C(133), UINT8_C( 22), UINT8_C(251), UINT8_C(243), UINT8_C(214), UINT8_C(128), UINT8_C( 55),
        UINT8_C(199), UINT8_C(235), UINT8_C( 98), UINT8_C( 76), UINT8_C( 51), UINT8_C(168), UINT8_C( 23), UINT8_C( 21) },
      { UINT8_C(146), UINT8_C( 84), UINT8_C( 52), UINT8_C( 17), UINT8_C(249), UINT8_C(132), UINT8_C(169), UINT8_C(101),
        UINT8_C( 54), UINT8_C(221), UINT8_C(216), UINT8_C( 79), UINT8_C( 11), UINT8_C(242), UINT8_C(240), UINT8_C(150),
        UINT8_C(120), UINT8_C(  6), UINT8_C(145), UINT8_C(107), UINT8_C(220), UINT8_C( 17), UINT8_C(162), UINT8_C(163),
        UINT8_C(253), UINT8_C(  4), UINT8_C(239), UINT8_C( 48), UINT8_C(172), UINT8_C(  6), UINT8_C( 69), UINT8_C( 62),
        UINT8_C( 90), UINT8_C(121), UINT8_C( 79), UINT8_C( 83), UINT8_C(253), UINT8_C(249), UINT8_C(184), UINT8_C( 51),
        UINT8_C(214), UINT8_C(145), UINT8_C(131), UINT8_C(225), UINT8_C(131), UINT8_C(115), UINT8_C(120), UINT8_C(251),
        UINT8_C(121), UINT8_C(  9), UINT8_C(102), UINT8_C( 85), UINT8_C( 27), UINT8_C(  8), UINT8_C(248), UINT8_C( 24),
        UINT8_C( 12), UINT8_C(231), UINT8_C( 72), UINT8_C(185), UINT8_C(237), UINT8_C(142), UINT8_C(247), UINT8_C( 71) },
      { UINT8_C( 70), UINT8_C(141), UINT8_C( 52), UINT8_C(151), UINT8_C( 61), UINT8_C(208), UINT8_C( 60), UINT8_C(124),
        UINT8_C( 36), UINT8_C(199), UINT8_C( 98), UINT8_C(199), UINT8_C(  6), UINT8_C(122), UINT8_C(240), UINT8_C(150),
        UINT8_C(120), UINT8_C( 79), UINT8_C(145), UINT8_C(105), UINT8_C(112), UINT8_C( 17), UINT8_C(223), UINT8_C(212),
        UINT8_C(222), UINT8_C(  4), UINT8_C(107), UINT8_C(  1), UINT8_C( 36), UINT8_C(  6), UINT8_C( 69), UINT8_C( 62),
        UINT8_C( 90), UINT8_C(121), UINT8_C( 61), UINT8_C(230), UINT8_C(209), UINT8_C(165), UINT8_C( 80), UINT8_C( 51),
        UINT8_C( 52), UINT8_C(178), UINT8_C( 51), UINT8_C(135), UINT8_C( 25), UINT8_C(164), UINT8_C(242), UINT8_C(160),
        UINT8_C(243), UINT8_C(132), UINT8_C( 22), UINT8_C( 85), UINT8_C( 27), UINT8_C( 63), UINT8_C(128), UINT8_C( 24),
        UINT8_C( 12), UINT8_C(231), UINT8_C( 97), UINT8_C( 76), UINT8_C( 51), UINT8_C( 19), UINT8_C(  2), UINT8_C( 74) } },
    { {  INT8_C(   7),  INT8_C(  71), -INT8_C( 102),  INT8_C(   5),  INT8_C(  64),  INT8_C(  82),  INT8_C(  56),  INT8_C(  22),
        -INT8_C(  29), -INT8_C(  69), -INT8_C(   9),  INT8_C( 103),  INT8_C(  46),  INT8_C( 111),  INT8_C(  98), -INT8_C(  89),
         INT8_C( 121), -INT8_C(  55), -INT8_C(   4), -INT8_C( 108), -INT8_C(  47), -INT8_C(  12), -INT8_C(  84), -INT8_C(  34),
        -INT8_C(  37), -INT8_C(  12), -INT8_C( 105), -INT8_C(  56), -INT8_C( 126), -INT8_C( 114),  INT8_C(  15), -INT8_C( 118),
        -INT8_C(  43), -INT8_C(  87), -INT8_C( 113),  INT8_C(  21), -INT8_C(   4), -INT8_C(  57),  INT8_C(  43), -INT8_C(  33),
        -INT8_C( 125),  INT8_C(  35),  INT8_C(  70), -INT8_C(  79), -INT8_C( 110), -INT8_C(  87),  INT8_C(  89),  INT8_C(  11),
         INT8_C( 114),  INT8_C(  85), -INT8_C(  97),  INT8_C(  67),  INT8_C(  74),  INT8_C(  75),  INT8_C(  33),  INT8_C(  37),
         INT8_C(  64), -INT8_C(  72), -INT8_C(  18), -INT8_C(  62),  INT8_C(  71), -INT8_C(   3),  INT8_C(  76),  INT8_C(  28) },
      UINT64_C( 2774883277812718503),
      { UINT8_C(128), UINT8_C(201), UINT8_C(215), UINT8_C( 19), UINT8_C(114), UINT8_C( 48), UINT8_C( 30), UINT8_C(228),
        UINT8_C(134), UINT8_C(190), UINT8_C( 39), UINT8_C(208), UINT8_C(  9), UINT8_C( 73), UINT8_C(245), UINT8_C( 73),
        UINT8_C(  1), UINT8_C(227), UINT8_C( 12), UINT8_C( 72), UINT8_C(225), UINT8_C( 88), UINT8_C(101), UINT8_C(136),
        UINT8_C( 52), UINT8_C(151), UINT8_C( 43), UINT8_C(215), UINT8_C(244), UINT8_C(173), UINT8_C(253), UINT8_C(117),
        UINT8_C(118), UINT8_C(212), UINT8_C(136), UINT8_C(232), UINT8_C(  5), UINT8_C(166), UINT8_C(204), UINT8_C(139),
        UINT8_C(100), UINT8_C(244), UINT8_C( 91), UINT8_C(110), UINT8_C( 61), UINT8_C( 80), UINT8_C(183), UINT8_C( 62),
        UINT8_C( 52), UINT8_C(195), UINT8_C(135), UINT8_C( 21), UINT8_C( 28), UINT8_C(236), UINT8_C(157), UINT8_C( 80),
        UINT8_C(131), UINT8_C(200), UINT8_C( 39), UINT8_C(119), UINT8_C(117), UINT8_C( 36), UINT8_C(236), UINT8_C(236) },
      { UINT8_C(248), UINT8_C(116), UINT8_C(212), UINT8_C(253), UINT8_C( 27), UINT8_C(161), UINT8_C(136), UINT8_C(127),
        UINT8_C(149), UINT8_C(227), UINT8_C(237), UINT8_C(210), UINT8_C( 52), UINT8_C(165), UINT8_C( 16), UINT8_C(104),
        UINT8_C(104), UINT8_C(151), UINT8_C(125), UINT8_C(132), UINT8_C(131), UINT8_C( 26), UINT8_C(212), UINT8_C(  6),
        UINT8_C(226), UINT8_C(251), UINT8_C(126), UINT8_C( 87), UINT8_C( 31), UINT8_C(106), UINT8_C( 67), UINT8_C( 24),
        UINT8_C(223), UINT8_C( 24), UINT8_C( 21), UINT8_C(250), UINT8_C(185), UINT8_C(158), UINT8_C(121), UINT8_C( 78),
        UINT8_C(129), UINT8_C(103), UINT8_C( 32), UINT8_C(181), UINT8_C( 12), UINT8_C( 48), UINT8_C( 29), UINT8_C(116),
        UINT8_C(200), UINT8_C(154), UINT8_C(249), UINT8_C( 75), UINT8_C(180), UINT8_C(205), UINT8_C( 82), UINT8_C(150),
        UINT8_C(201), UINT8_C(208), UINT8_C(238), UINT8_C(232), UINT8_C( 58), UINT8_C( 49), UINT8_C(  0), UINT8_C( 25) },
      { UINT8_C(128), UINT8_C(116), UINT8_C(212), UINT8_C(  5), UINT8_C( 64), UINT8_C( 48), UINT8_C( 56), UINT8_C(127),
        UINT8_C(134), UINT8_C(190), UINT8_C(247), UINT8_C(208), UINT8_C(  9), UINT8_C(111), UINT8_C( 16), UINT8_C( 73),
        UINT8_C(121), UINT8_C(151), UINT8_C(252), UINT8_C(148), UINT8_C(131), UINT8_C( 26), UINT8_C(172), UINT8_C(222),
        UINT8_C( 52), UINT8_C(151), UINT8_C(151), UINT8_C(200), UINT8_C(130), UINT8_C(106), UINT8_C( 15), UINT8_C( 24),
        UINT8_C(118), UINT8_C( 24), UINT8_C(143), UINT8_C( 21), UINT8_C(252), UINT8_C(158), UINT8_C( 43), UINT8_C( 78),
        UINT8_C(100), UINT8_C( 35), UINT8_C( 32), UINT8_C(110), UINT8_C( 12), UINT8_C(169), UINT8_C( 29), UINT8_C( 11),
        UINT8_C(114), UINT8_C(154), UINT8_C(159), UINT8_C( 67), UINT8_C( 74), UINT8_C( 75), UINT8_C( 33), UINT8_C( 80),
        UINT8_C( 64), UINT8_C(200), UINT8_C( 39), UINT8_C(194), UINT8_C( 71), UINT8_C( 36), UINT8_C( 76), UINT8_C( 28) } },
    { {  INT8_C(  73),  INT8_C(  22),  INT8_C(  19),  INT8_C(   2), -INT8_C(  76), -INT8_C( 115),  INT8_C(  80),  INT8_C(  53),
        -INT8_C(  12),  INT8_C( 112), -INT8_C(  21),  INT8_C(   0), -INT8_C(  95),  INT8_C(   8),  INT8_C( 116),  INT8_C( 105),
        -INT8_C(  93),  INT8_C( 109), -INT8_C(  76),  INT8_C(  87),  INT8_C(  59),  INT8_C(   6), -INT8_C(  18),  INT8_C(   4),
        -INT8_C(  42), -INT8_C(  36), -INT8_C(  20),  INT8_C(  17),  INT8_C(  13), -INT8_C(  19),  INT8_C(  42),  INT8_C(  87),
         INT8_C(   3),  INT8_C(  62),  INT8_C(  89), -INT8_C(  73), -INT8_C(  53), -INT8_C(  86), -INT8_C(  20), -INT8_C(  65),
         INT8_C(  26), -INT8_C(  41), -INT8_C(  65), -INT8_C(  69), -INT8_C(  32),  INT8_C(  51),  INT8_C(  36), -INT8_C( 125),
        -INT8_C(  95), -INT8_C(  39), -INT8_C(  38), -INT8_C(  36), -INT8_C(  33), -INT8_C(  56), -INT8_C(  32), -INT8_C(  74),
        -INT8_C(  92), -INT8_C(  52), -INT8_C(  57), -INT8_C(  78), -INT8_C(  71), -INT8_C(  15),  INT8_C(   9), -INT8_C(  68) },
      UINT64_C( 2862424647028400687),
      { UINT8_C( 55), UINT8_C(120), UINT8_C(226), UINT8_C( 23), UINT8_C(172), UINT8_C(  7), UINT8_C(154), UINT8_C( 77),
        UINT8_C(224), UINT8_C(117), UINT8_C( 41), UINT8_C(191), UINT8_C( 61), UINT8_C(  9), UINT8_C(117), UINT8_C(226),
        UINT8_C(213), UINT8_C( 60), UINT8_C(148), UINT8_C(143), UINT8_C( 46), UINT8_C(157), UINT8_C( 75), UINT8_C( 93),
           UINT8_MAX, UINT8_C(191), UINT8_C( 88), UINT8_C( 12), UINT8_C( 31), UINT8_C( 17), UINT8_C( 51), UINT8_C( 86),
        UINT8_C(138), UINT8_C( 21), UINT8_C(110), UINT8_C( 54), UINT8_C( 28), UINT8_C(  8), UINT8_C(131), UINT8_C(252),
        UINT8_C(125), UINT8_C(172), UINT8_C(188), UINT8_C(187), UINT8_C(181), UINT8_C( 49), UINT8_C(157), UINT8_C(138),
        UINT8_C(110), UINT8_C( 49), UINT8_C( 25), UINT8_C(156), UINT8_C(206), UINT8_C(101), UINT8_C(249), UINT8_C(205),
        UINT8_C( 36), UINT8_C( 81), UINT8_C(217), UINT8_C( 67), UINT8_C( 99), UINT8_C( 12), UINT8_C(153), UINT8_C(237) },
      { UINT8_C( 34), UINT8_C(  7), UINT8_C( 35), UINT8_C( 62), UINT8_C( 16), UINT8_C(166), UINT8_C( 59), UINT8_C(141),
        UINT8_C( 82), UINT8_C(247), UINT8_C( 72), UINT8_C(  7), UINT8_C( 40), UINT8_C(229), UINT8_C(145), UINT8_C(150),
        UINT8_C( 22), UINT8_C(171), UINT8_C( 50), UINT8_C(228), UINT8_C( 16), UINT8_C( 44), UINT8_C(178), UINT8_C( 52),
        UINT8_C(125), UINT8_C(139), UINT8_C(119), UINT8_C(224), UINT8_C(152), UINT8_C( 16), UINT8_C(205), UINT8_C(186),
        UINT8_C( 24), UINT8_C(240), UINT8_C(248), UINT8_C( 40), UINT8_C(150), UINT8_C( 51), UINT8_C(181), UINT8_C(232),
        UINT8_C( 42), UINT8_C(254), UINT8_C(239), UINT8_C( 83), UINT8_C(227), UINT8_C(129), UINT8_C(233), UINT8_C(250),
        UINT8_C( 44), UINT8_C( 28), UINT8_C(222), UINT8_C( 60), UINT8_C( 72), UINT8_C(144), UINT8_C(112), UINT8_C(197),
        UINT8_C( 28), UINT8_C(231), UINT8_C(166), UINT8_C(180), UINT8_C(247), UINT8_C(115), UINT8_C(110), UINT8_C( 15) },
      { UINT8_C( 34), UINT8_C(  7), UINT8_C( 35), UINT8_C( 23), UINT8_C(180), UINT8_C(  7), UINT8_C( 80), UINT8_C( 53),
        UINT8_C(244), UINT8_C(117), UINT8_C(235), UINT8_C(  0), UINT8_C(161), UINT8_C(  9), UINT8_C(117), UINT8_C(105),
        UINT8_C( 22), UINT8_C( 60), UINT8_C(180), UINT8_C( 87), UINT8_C( 16), UINT8_C( 44), UINT8_C( 75), UINT8_C(  4),
        UINT8_C(214), UINT8_C(139), UINT8_C(236), UINT8_C( 12), UINT8_C( 31), UINT8_C( 16), UINT8_C( 51), UINT8_C( 86),
        UINT8_C(  3), UINT8_C( 62), UINT8_C(110), UINT8_C( 40), UINT8_C(203), UINT8_C(170), UINT8_C(236), UINT8_C(191),
        UINT8_C( 26), UINT8_C(215), UINT8_C(191), UINT8_C(187), UINT8_C(224), UINT8_C( 49), UINT8_C(157), UINT8_C(131),
        UINT8_C( 44), UINT8_C(217), UINT8_C(218), UINT8_C( 60), UINT8_C( 72), UINT8_C(101), UINT8_C(224), UINT8_C(197),
        UINT8_C( 28), UINT8_C( 81), UINT8_C(166), UINT8_C(178), UINT8_C(185), UINT8_C( 12), UINT8_C(  9), UINT8_C(188) } },
    { {  INT8_C( 100),  INT8_C( 102),  INT8_C(  55), -INT8_C(   6), -INT8_C( 102), -INT8_C(  19), -INT8_C(  29), -INT8_C(  60),
        -INT8_C(  21), -INT8_C(  46),  INT8_C(  23), -INT8_C(  50),  INT8_C(  83),  INT8_C(   1), -INT8_C(  56),      INT8_MAX,
         INT8_C(  29), -INT8_C(  89), -INT8_C(  69),  INT8_C( 101),  INT8_C(  55),  INT8_C(  43),  INT8_C(  42),  INT8_C(  83),
         INT8_C(  18), -INT8_C(  48),  INT8_C(   7),  INT8_C(  10),  INT8_C(  68),  INT8_C( 117),  INT8_C(  25), -INT8_C(  88),
        -INT8_C(  36),  INT8_C(  81), -INT8_C(  94),  INT8_C( 118),  INT8_C(  62), -INT8_C( 123),  INT8_C(  58),  INT8_C(  41),
         INT8_C(  88),  INT8_C(  82), -INT8_C(   9), -INT8_C(  85),  INT8_C(  83), -INT8_C(  64),  INT8_C(  43),  INT8_C( 112),
         INT8_C( 103), -INT8_C(  26), -INT8_C(  43), -INT8_C(  98),  INT8_C(  18), -INT8_C(   1), -INT8_C(  14),  INT8_C(  36),
        -INT8_C(  48), -INT8_C(   7),  INT8_C(  46),  INT8_C(  20),  INT8_C( 111),  INT8_C(  72), -INT8_C(  68),  INT8_C(  75) },
      UINT64_C( 4323732602566565529),
      { UINT8_C( 77), UINT8_C(247), UINT8_C(231), UINT8_C(160), UINT8_C(183), UINT8_C( 18), UINT8_C( 16), UINT8_C( 30),
        UINT8_C(249), UINT8_C(229), UINT8_C(189), UINT8_C( 11), UINT8_C(229), UINT8_C(175), UINT8_C( 47), UINT8_C(181),
        UINT8_C(168), UINT8_C( 94), UINT8_C(201), UINT8_C( 23), UINT8_C(166), UINT8_C(133), UINT8_C( 98), UINT8_C( 63),
        UINT8_C(227), UINT8_C( 35), UINT8_C( 22), UINT8_C(199), UINT8_C( 31), UINT8_C( 22), UINT8_C(  3), UINT8_C(108),
        UINT8_C( 13), UINT8_C(235), UINT8_C( 13), UINT8_C(197), UINT8_C(253), UINT8_C( 29), UINT8_C(227), UINT8_C(246),
        UINT8_C(  3), UINT8_C(160), UINT8_C(  1), UINT8_C(232), UINT8_C( 79), UINT8_C( 49), UINT8_C(157), UINT8_C(248),
        UINT8_C(143), UINT8_C(102), UINT8_C( 15), UINT8_C( 53), UINT8_C(235), UINT8_C(114), UINT8_C(116), UINT8_C(206),
        UINT8_C(149), UINT8_C(138), UINT8_C(150), UINT8_C(180), UINT8_C(160), UINT8_C(153), UINT8_C( 33), UINT8_C(173) },
      { UINT8_C(132), UINT8_C( 46), UINT8_C(114), UINT8_C(130), UINT8_C( 75), UINT8_C( 86), UINT8_C(120), UINT8_C( 78),
        UINT8_C(246), UINT8_C(122), UINT8_C( 54), UINT8_C( 70), UINT8_C(171), UINT8_C(211), UINT8_C( 62), UINT8_C( 58),
        UINT8_C( 57), UINT8_C( 77), UINT8_C(111), UINT8_C( 36), UINT8_C(191), UINT8_C(227), UINT8_C(243), UINT8_C( 85),
        UINT8_C(109), UINT8_C(137), UINT8_C(  9), UINT8_C( 13), UINT8_C( 34), UINT8_C( 42), UINT8_C(186), UINT8_C(167),
        UINT8_C( 88), UINT8_C( 45), UINT8_C( 41), UINT8_C(164), UINT8_C(131), UINT8_C(161), UINT8_C(242), UINT8_C(121),
        UINT8_C( 27), UINT8_C( 41), UINT8_C(191), UINT8_C(198), UINT8_C(252), UINT8_C(253), UINT8_C(  0), UINT8_C( 54),
        UINT8_C( 75), UINT8_C(111), UINT8_C( 90), UINT8_C( 10), UINT8_C( 82), UINT8_C( 77), UINT8_C( 95), UINT8_C(191),
        UINT8_C(214), UINT8_C(105), UINT8_C(204), UINT8_C(249), UINT8_C(147), UINT8_C(135), UINT8_C(160), UINT8_C(236) },
      { UINT8_C( 77), UINT8_C(102), UINT8_C( 55), UINT8_C(130), UINT8_C( 75), UINT8_C(237), UINT8_C(227), UINT8_C( 30),
        UINT8_C(235), UINT8_C(122), UINT8_C( 54), UINT8_C( 11), UINT8_C(171), UINT8_C(  1), UINT8_C( 47), UINT8_C(127),
        UINT8_C( 57), UINT8_C(167), UINT8_C(187), UINT8_C(101), UINT8_C( 55), UINT8_C( 43), UINT8_C( 98), UINT8_C( 63),
        UINT8_C(109), UINT8_C( 35), UINT8_C(  9), UINT8_C( 10), UINT8_C( 31), UINT8_C(117), UINT8_C(  3), UINT8_C(108),
        UINT8_C(220), UINT8_C( 81), UINT8_C( 13), UINT8_C(118), UINT8_C( 62), UINT8_C( 29), UINT8_C(227), UINT8_C(121),
        UINT8_C(  3), UINT8_C( 41), UINT8_C(247), UINT8_C(198), UINT8_C( 79), UINT8_C( 49), UINT8_C(  0), UINT8_C( 54),
        UINT8_C(103), UINT8_C(230), UINT8_C(213), UINT8_C(158), UINT8_C( 18),    UINT8_MAX, UINT8_C(242), UINT8_C( 36),
        UINT8_C(208), UINT8_C(249), UINT8_C(150), UINT8_C(180), UINT8_C(147), UINT8_C(135), UINT8_C(188), UINT8_C( 75) } },
    { { -INT8_C(  76), -INT8_C(  55), -INT8_C( 112),  INT8_C(  55),  INT8_C( 106), -INT8_C( 126), -INT8_C(  80), -INT8_C( 122),
        -INT8_C(  85),  INT8_C( 112),  INT8_C(  76), -INT8_C(  88),  INT8_C( 109),  INT8_C(  77), -INT8_C(  34), -INT8_C(  72),
        -INT8_C(  68),  INT8_C(  56), -INT8_C(  61),  INT8_C(  15), -INT8_C( 122),  INT8_C(  34), -INT8_C(  50),  INT8_C(  92),
        -INT8_C( 117), -INT8_C( 101),  INT8_C(  85),  INT8_C(  31),  INT8_C(  34), -INT8_C(  11),  INT8_C(  11), -INT8_C(  42),
        -INT8_C(  66), -INT8_C( 101),  INT8_C(  13),  INT8_C(  41),  INT8_C(  29), -INT8_C(  67), -INT8_C(  81), -INT8_C(  55),
         INT8_C(  45), -INT8_C(   5),  INT8_C( 113), -INT8_C( 101),  INT8_C(  72),  INT8_C(  79),  INT8_C(  83),  INT8_C(   5),
        -INT8_C( 121),  INT8_C(  22),  INT8_C(  20),  INT8_C(  13),  INT8_C(  57), -INT8_C(  30),  INT8_C( 106), -INT8_C(  60),
         INT8_C( 125), -INT8_C(  65), -INT8_C(  29), -INT8_C(  97), -INT8_C(  75), -INT8_C(  18),  INT8_C( 117),  INT8_C( 115) },
      UINT64_C( 7885885688587780745),
      { UINT8_C( 71), UINT8_C(225), UINT8_C(  8), UINT8_C(143), UINT8_C( 48), UINT8_C( 92), UINT8_C(148), UINT8_C(183),
        UINT8_C(114), UINT8_C(168), UINT8_C(197), UINT8_C(171), UINT8_C(139), UINT8_C( 47), UINT8_C(112), UINT8_C(  8),
        UINT8_C(238), UINT8_C( 83), UINT8_C(168), UINT8_C(163), UINT8_C( 66), UINT8_C( 29), UINT8_C( 23), UINT8_C(203),
        UINT8_C(160), UINT8_C(179), UINT8_C(114), UINT8_C(224),    UINT8_MAX, UINT8_C(226), UINT8_C( 77), UINT8_C( 70),
        UINT8_C(195), UINT8_C( 86), UINT8_C(213), UINT8_C(243), UINT8_C(178), UINT8_C(106), UINT8_C(171), UINT8_C( 36),
        UINT8_C( 18), UINT8_C(112), UINT8_C(208), UINT8_C(157), UINT8_C(159), UINT8_C( 64), UINT8_C(166), UINT8_C(141),
        UINT8_C(147), UINT8_C( 78), UINT8_C( 49), UINT8_C(213), UINT8_C(107), UINT8_C( 72), UINT8_C(161), UINT8_C( 11),
        UINT8_C(251), UINT8_C( 19), UINT8_C(235), UINT8_C(250), UINT8_C(246), UINT8_C( 57), UINT8_C( 64), UINT8_C(185) },
      { UINT8_C(143), UINT8_C( 22), UINT8_C(173), UINT8_C( 65), UINT8_C(128), UINT8_C( 88), UINT8_C(101), UINT8_C(146),
        UINT8_C(200), UINT8_C( 53), UINT8_C( 48), UINT8_C(103), UINT8_C(117), UINT8_C(214), UINT8_C(244), UINT8_C(  9),
        UINT8_C( 36), UINT8_C( 37), UINT8_C(222), UINT8_C(143), UINT8_C(109), UINT8_C(127), UINT8_C(155), UINT8_C(105),
        UINT8_C(147), UINT8_C(134), UINT8_C( 99), UINT8_C(137), UINT8_C(191), UINT8_C(164), UINT8_C( 66), UINT8_C( 78),
        UINT8_C(186), UINT8_C(239), UINT8_C(143), UINT8_C( 58), UINT8_C( 71), UINT8_C(245), UINT8_C(204), UINT8_C( 15),
        UINT8_C( 42), UINT8_C(252), UINT8_C(118), UINT8_C(160), UINT8_C(210), UINT8_C(107), UINT8_C(169), UINT8_C(246),
        UINT8_C(144), UINT8_C(135), UINT8_C(134), UINT8_C(254), UINT8_C(  7), UINT8_C( 33), UINT8_C(103), UINT8_C(154),
        UINT8_C(167), UINT8_C(202), UINT8_C( 35), UINT8_C(103), UINT8_C(110), UINT8_C(101), UINT8_C(181), UINT8_C( 40) },
      { UINT8_C( 71), UINT8_C(201), UINT8_C(144), UINT8_C( 65), UINT8_C(106), UINT8_C(130), UINT8_C(176), UINT8_C(146),
        UINT8_C(171), UINT8_C( 53), UINT8_C( 76), UINT8_C(168), UINT8_C(109), UINT8_C( 77), UINT8_C(222), UINT8_C(  8),
        UINT8_C(188), UINT8_C( 56), UINT8_C(168), UINT8_C(143), UINT8_C( 66), UINT8_C( 34), UINT8_C(206), UINT8_C(105),
        UINT8_C(147), UINT8_C(134), UINT8_C( 99), UINT8_C( 31), UINT8_C( 34), UINT8_C(164), UINT8_C( 11), UINT8_C( 70),
        UINT8_C(190), UINT8_C(155), UINT8_C( 13), UINT8_C( 41), UINT8_C( 29), UINT8_C(189), UINT8_C(171), UINT8_C(201),
        UINT8_C( 18), UINT8_C(112), UINT8_C(113), UINT8_C(157), UINT8_C( 72), UINT8_C( 79), UINT8_C(166), UINT8_C(  5),
        UINT8_C(135), UINT8_C( 22), UINT8_C( 20), UINT8_C( 13), UINT8_C(  7), UINT8_C( 33), UINT8_C(103), UINT8_C(196),
        UINT8_C(167), UINT8_C(191), UINT8_C( 35), UINT8_C(103), UINT8_C(181), UINT8_C( 57), UINT8_C( 64), UINT8_C(115) } },
    { {  INT8_C(  85),  INT8_C(  69),  INT8_C(  98), -INT8_C( 100),  INT8_C(  58),  INT8_C(  47), -INT8_C(  84),  INT8_C( 100),
         INT8_C(  43),  INT8_C(  34),  INT8_C(   4), -INT8_C(   2), -INT8_C( 115), -INT8_C(  83), -INT8_C(  12),  INT8_C(  30),
         INT8_C(  53),  INT8_C( 122),  INT8_C(  28),  INT8_C(  60), -INT8_C( 101), -INT8_C( 125), -INT8_C(  42),  INT8_C(  67),
         INT8_C(  77), -INT8_C(   7), -INT8_C(  86), -INT8_C(  68),  INT8_C(  94),  INT8_C(  95), -INT8_C(  28), -INT8_C(  77),
        -INT8_C(  92),  INT8_C(  71),  INT8_C(  80), -INT8_C(  34),  INT8_C( 118), -INT8_C(   4),  INT8_C(  67), -INT8_C(  95),
         INT8_C(  30),  INT8_C(  71), -INT8_C(  97), -INT8_C(  84), -INT8_C(  11), -INT8_C( 108), -INT8_C(  54),  INT8_C(  42),
         INT8_C(  14), -INT8_C(  26),  INT8_C( 102), -INT8_C(  86),  INT8_C( 105),  INT8_C(  60), -INT8_C(  19), -INT8_C(  74),
         INT8_C(  53), -INT8_C( 105),  INT8_C( 114), -INT8_C( 109), -INT8_C(  10),  INT8_C(  87),  INT8_C(  71), -INT8_C( 101) },
      UINT64_C(12805348455387600798),
      { UINT8_C(  4), UINT8_C( 85), UINT8_C( 93), UINT8_C(249), UINT8_C(233), UINT8_C( 39), UINT8_C( 35), UINT8_C(247),
        UINT8_C( 13), UINT8_C(137), UINT8_C(161), UINT8_C(118), UINT8_C(197), UINT8_C(142), UINT8_C( 45), UINT8_C(250),
        UINT8_C( 37), UINT8_C(159), UINT8_C(141), UINT8_C( 28), UINT8_C(246), UINT8_C(212), UINT8_C(183), UINT8_C(148),
        UINT8_C(107), UINT8_C( 48), UINT8_C(168), UINT8_C(254), UINT8_C(237), UINT8_C( 94), UINT8_C(176), UINT8_C(241),
        UINT8_C(179), UINT8_C( 13), UINT8_C(234), UINT8_C(156), UINT8_C( 53), UINT8_C( 13), UINT8_C(147), UINT8_C( 66),
        UINT8_C(150), UINT8_C( 53), UINT8_C(185), UINT8_C( 91), UINT8_C(195), UINT8_C(230), UINT8_C( 85), UINT8_C(233),
        UINT8_C(133), UINT8_C(226), UINT8_C(  5), UINT8_C(124), UINT8_C(183), UINT8_C(188), UINT8_C( 16), UINT8_C( 34),
        UINT8_C(236), UINT8_C(185), UINT8_C( 33), UINT8_C(217), UINT8_C( 23), UINT8_C(209), UINT8_C(202), UINT8_C(202) },
      { UINT8_C(222), UINT8_C(180), UINT8_C(102), UINT8_C( 19), UINT8_C(193), UINT8_C(249), UINT8_C( 86), UINT8_C( 87),
        UINT8_C( 46), UINT8_C( 15), UINT8_C(178), UINT8_C(242), UINT8_C(245), UINT8_C(  7), UINT8_C(219), UINT8_C(122),
        UINT8_C(234), UINT8_C(224), UINT8_C(246), UINT8_C(161), UINT8_C(156), UINT8_C(  7), UINT8_C(195), UINT8_C(136),
        UINT8_C(192), UINT8_C(228), UINT8_C( 98), UINT8_C(215), UINT8_C(181), UINT8_C( 44), UINT8_C(161), UINT8_C(148),
        UINT8_C(225), UINT8_C(  7), UINT8_C(167), UINT8_C(162), UINT8_C(  0), UINT8_C(253), UINT8_C(250), UINT8_C( 47),
        UINT8_C( 12), UINT8_C(172), UINT8_C( 33), UINT8_C(  1), UINT8_C(180), UINT8_C(252), UINT8_C(124), UINT8_C(158),
        UINT8_C(220), UINT8_C(114), UINT8_C( 63), UINT8_C(120), UINT8_C(121), UINT8_C(  2), UINT8_C(  0), UINT8_C( 57),
        UINT8_C(231), UINT8_C( 98), UINT8_C( 16), UINT8_C(156), UINT8_C(143), UINT8_C(177), UINT8_C( 48), UINT8_C(112) },
      { UINT8_C( 85), UINT8_C( 85), UINT8_C( 93), UINT8_C( 19), UINT8_C(193), UINT8_C( 47), UINT8_C(172), UINT8_C( 87),
        UINT8_C( 13), UINT8_C( 15), UINT8_C(161), UINT8_C(254), UINT8_C(197), UINT8_C(173), UINT8_C(244), UINT8_C(122),
        UINT8_C( 37), UINT8_C(122), UINT8_C( 28), UINT8_C( 28), UINT8_C(156), UINT8_C(  7), UINT8_C(183), UINT8_C( 67),
        UINT8_C( 77), UINT8_C(249), UINT8_C( 98), UINT8_C(188), UINT8_C(181), UINT8_C( 95), UINT8_C(228), UINT8_C(179),
        UINT8_C(179), UINT8_C(  7), UINT8_C( 80), UINT8_C(222), UINT8_C(  0), UINT8_C(252), UINT8_C( 67), UINT8_C( 47),
        UINT8_C( 30), UINT8_C( 71), UINT8_C( 33), UINT8_C(  1), UINT8_C(180), UINT8_C(230), UINT8_C(202), UINT8_C(158),
        UINT8_C(133), UINT8_C(230), UINT8_C(  5), UINT8_C(170), UINT8_C(121), UINT8_C(  2), UINT8_C(237), UINT8_C( 34),
        UINT8_C(231), UINT8_C(151), UINT8_C(114), UINT8_C(147), UINT8_C( 23), UINT8_C(177), UINT8_C( 71), UINT8_C(112) } },
    { { -INT8_C(  72), -INT8_C(  40),  INT8_C(  18), -INT8_C(  71), -INT8_C(  43),  INT8_C(  12), -INT8_C(  24), -INT8_C(  30),
        -INT8_C(  71),  INT8_C(   9), -INT8_C(  29),  INT8_C( 109),  INT8_C(   5),  INT8_C(  95),  INT8_C(  11), -INT8_C(  31),
        -INT8_C(  46),  INT8_C(  74),  INT8_C(  89),  INT8_C(  75),  INT8_C(  76),  INT8_C(  89), -INT8_C( 123),  INT8_C(  51),
        -INT8_C(  68), -INT8_C( 107), -INT8_C(  48),  INT8_C(  75),  INT8_C(  71),  INT8_C(   0), -INT8_C(  69), -INT8_C(   1),
        -INT8_C(  40), -INT8_C(  51), -INT8_C(  72), -INT8_C(  82), -INT8_C(  38), -INT8_C(  96), -INT8_C( 112), -INT8_C( 109),
        -INT8_C(  87),  INT8_C( 115),  INT8_C(   0), -INT8_C(  82), -INT8_C(  45),  INT8_C(  11), -INT8_C( 113), -INT8_C(  91),
         INT8_C(  85), -INT8_C(  24), -INT8_C(  16), -INT8_C(  95),  INT8_C(  66),  INT8_C( 117), -INT8_C(  43), -INT8_C(   2),
         INT8_C(  11), -INT8_C(  91),  INT8_C(  73),  INT8_C(  82), -INT8_C(  91),  INT8_C(   4),  INT8_C(  81),  INT8_C( 126) },
      UINT64_C( 6070496788944259793),
      { UINT8_C( 47), UINT8_C( 62), UINT8_C(  2), UINT8_C(  2), UINT8_C( 73), UINT8_C(146), UINT8_C(167), UINT8_C(158),
        UINT8_C(122), UINT8_C(152), UINT8_C( 64), UINT8_C(188), UINT8_C( 13), UINT8_C( 21), UINT8_C(186), UINT8_C( 24),
        UINT8_C(186), UINT8_C(  3), UINT8_C(106), UINT8_C( 95), UINT8_C(  7), UINT8_C(188), UINT8_C(221), UINT8_C(217),
        UINT8_C(198), UINT8_C(  9), UINT8_C(132), UINT8_C(112), UINT8_C(197), UINT8_C(195), UINT8_C(196), UINT8_C(245),
        UINT8_C(  1), UINT8_C(199), UINT8_C(247), UINT8_C( 75), UINT8_C( 89), UINT8_C(159), UINT8_C(233), UINT8_C(211),
        UINT8_C( 55), UINT8_C( 41), UINT8_C(144), UINT8_C( 68), UINT8_C( 62), UINT8_C( 74), UINT8_C( 93), UINT8_C(248),
        UINT8_C( 78), UINT8_C(199), UINT8_C( 88), UINT8_C( 85), UINT8_C(131), UINT8_C( 53), UINT8_C( 46), UINT8_C( 73),
        UINT8_C( 63), UINT8_C(179), UINT8_C(186), UINT8_C(  4), UINT8_C(118), UINT8_C(126), UINT8_C(249), UINT8_C(119) },
      { UINT8_C( 69), UINT8_C(241), UINT8_C(194), UINT8_C(158), UINT8_C(144), UINT8_C(172), UINT8_C(114), UINT8_C(199),
        UINT8_C(213), UINT8_C(  2), UINT8_C( 11), UINT8_C( 20), UINT8_C( 76), UINT8_C(104), UINT8_C( 12), UINT8_C(154),
        UINT8_C( 48), UINT8_C(100), UINT8_C(240), UINT8_C(179), UINT8_C(154), UINT8_C( 30), UINT8_C(253), UINT8_C(217),
        UINT8_C(209), UINT8_C(183), UINT8_C(221), UINT8_C( 71), UINT8_C( 53), UINT8_C(215), UINT8_C(191), UINT8_C(123),
        UINT8_C(200), UINT8_C(129), UINT8_C( 25), UINT8_C( 88), UINT8_C( 45), UINT8_C(139), UINT8_C( 31), UINT8_C(  3),
        UINT8_C(141), UINT8_C( 42), UINT8_C( 23), UINT8_C(218), UINT8_C(147), UINT8_C( 35), UINT8_C(116), UINT8_C(195),
        UINT8_C(136), UINT8_C(100), UINT8_C(118), UINT8_C( 34), UINT8_C(131), UINT8_C(115), UINT8_C(251), UINT8_C( 84),
        UINT8_C( 42), UINT8_C(216), UINT8_C(156), UINT8_C( 96), UINT8_C(175), UINT8_C( 91), UINT8_C(219), UINT8_C(119) },
      { UINT8_C( 47), UINT8_C(216), UINT8_C( 18), UINT8_C(185), UINT8_C( 73), UINT8_C( 12), UINT8_C(114), UINT8_C(158),
        UINT8_C(185), UINT8_C(  2), UINT8_C(227), UINT8_C( 20), UINT8_C(  5), UINT8_C( 95), UINT8_C( 11), UINT8_C(225),
        UINT8_C(210), UINT8_C( 74), UINT8_C(106), UINT8_C( 95), UINT8_C( 76), UINT8_C( 30), UINT8_C(133), UINT8_C( 51),
        UINT8_C(198), UINT8_C(  9), UINT8_C(208), UINT8_C( 71), UINT8_C( 71), UINT8_C(195), UINT8_C(187), UINT8_C(123),
        UINT8_C(216), UINT8_C(129), UINT8_C(184), UINT8_C( 75), UINT8_C(218), UINT8_C(139), UINT8_C(144), UINT8_C(  3),
        UINT8_C(169), UINT8_C(115), UINT8_C( 23), UINT8_C( 68), UINT8_C( 62), UINT8_C( 35), UINT8_C(143), UINT8_C(195),
        UINT8_C( 85), UINT8_C(100), UINT8_C( 88), UINT8_C( 34), UINT8_C(131), UINT8_C( 53), UINT8_C(213), UINT8_C(254),
        UINT8_C( 11), UINT8_C(165), UINT8_C(156), UINT8_C( 82), UINT8_C(118), UINT8_C(  4), UINT8_C(219), UINT8_C(126) } },
    { { -INT8_C(  36), -INT8_C(  12), -INT8_C(  49),  INT8_C(  10),      INT8_MIN, -INT8_C(  18),  INT8_C(  13),  INT8_C(  13),
         INT8_C(  25),  INT8_C(  36), -INT8_C(  25), -INT8_C(  84),  INT8_C(  71),  INT8_C(  92),  INT8_C( 111), -INT8_C(  49),
        -INT8_C(  64), -INT8_C(  27), -INT8_C(  15),  INT8_C(  67),  INT8_C(  89), -INT8_C(  20), -INT8_C( 104), -INT8_C( 125),
        -INT8_C(  59),  INT8_C(  52), -INT8_C(  29),  INT8_C( 116), -INT8_C( 113), -INT8_C(  66), -INT8_C(  20),  INT8_C( 107),
        -INT8_C(  77), -INT8_C(  69),  INT8_C( 117),  INT8_C(  51), -INT8_C(  86), -INT8_C( 126),  INT8_C(  64), -INT8_C(  61),
        -INT8_C(  90),  INT8_C(  40),  INT8_C( 111), -INT8_C(  18), -INT8_C( 124), -INT8_C(  34), -INT8_C(  67),  INT8_C(  68),
        -INT8_C(  61), -INT8_C(  81), -INT8_C( 120),  INT8_C(  28), -INT8_C( 101),  INT8_C(  32), -INT8_C(  96),  INT8_C(  96),
         INT8_C(  84), -INT8_C( 125), -INT8_C(  43), -INT8_C(  29),  INT8_C(  66), -INT8_C(  63),  INT8_C(  78), -INT8_C(  11) },
      UINT64_C(17143348107059709052),
      { UINT8_C(144), UINT8_C( 88), UINT8_C(219), UINT8_C( 20), UINT8_C( 54), UINT8_C(152), UINT8_C( 89), UINT8_C(250),
        UINT8_C( 71), UINT8_C(225), UINT8_C( 22), UINT8_C(227), UINT8_C(  1), UINT8_C(182), UINT8_C( 67), UINT8_C( 85),
        UINT8_C( 58), UINT8_C( 24), UINT8_C( 56), UINT8_C(124), UINT8_C(217), UINT8_C(134), UINT8_C(113), UINT8_C( 86),
        UINT8_C( 74), UINT8_C(153), UINT8_C(124), UINT8_C(145), UINT8_C(  1), UINT8_C(102), UINT8_C(126), UINT8_C(146),
        UINT8_C(190), UINT8_C( 89), UINT8_C(166), UINT8_C(245), UINT8_C(241),    UINT8_MAX, UINT8_C(239), UINT8_C( 57),
        UINT8_C(224), UINT8_C(  5), UINT8_C( 28), UINT8_C(225), UINT8_C(188), UINT8_C( 95), UINT8_C( 54), UINT8_C(246),
        UINT8_C(120), UINT8_C(110), UINT8_C(114), UINT8_C( 81), UINT8_C(245), UINT8_C(227), UINT8_C(167), UINT8_C( 63),
        UINT8_C(124), UINT8_C( 36), UINT8_C(208), UINT8_C(125), UINT8_C(138), UINT8_C( 78), UINT8_C( 15), UINT8_C( 72) },
      { UINT8_C(167), UINT8_C(182), UINT8_C( 61), UINT8_C(153), UINT8_C(181), UINT8_C( 44), UINT8_C(210), UINT8_C(150),
        UINT8_C( 50), UINT8_C(238), UINT8_C(119), UINT8_C(238), UINT8_C( 77), UINT8_C(174), UINT8_C(228), UINT8_C(197),
        UINT8_C( 28), UINT8_C( 86), UINT8_C( 23), UINT8_C( 17), UINT8_C( 57), UINT8_C(190), UINT8_C( 81), UINT8_C(181),
        UINT8_C(226), UINT8_C( 33), UINT8_C( 50), UINT8_C(108), UINT8_C(112), UINT8_C( 66), UINT8_C(181), UINT8_C( 23),
        UINT8_C(248), UINT8_C(242), UINT8_C(176), UINT8_C(173), UINT8_C( 31), UINT8_C(130), UINT8_C( 67), UINT8_C( 81),
        UINT8_C(112), UINT8_C(187), UINT8_C( 63), UINT8_C(190), UINT8_C(105), UINT8_C( 35), UINT8_C(131), UINT8_C(133),
        UINT8_C(121), UINT8_C(154), UINT8_C(151), UINT8_C(178), UINT8_C( 89), UINT8_C(232), UINT8_C(103), UINT8_C( 59),
        UINT8_C(  9), UINT8_C(153), UINT8_C(168), UINT8_C(121), UINT8_C(219), UINT8_C( 93), UINT8_C(145), UINT8_C(211) },
      { UINT8_C(220), UINT8_C(244), UINT8_C( 61), UINT8_C( 20), UINT8_C( 54), UINT8_C( 44), UINT8_C( 89), UINT8_C( 13),
        UINT8_C( 25), UINT8_C( 36), UINT8_C( 22), UINT8_C(172), UINT8_C( 71), UINT8_C( 92), UINT8_C( 67), UINT8_C( 85),
        UINT8_C(192), UINT8_C(229), UINT8_C(241), UINT8_C( 17), UINT8_C( 89), UINT8_C(134), UINT8_C(152), UINT8_C(131),
        UINT8_C(197), UINT8_C( 33), UINT8_C( 50), UINT8_C(116), UINT8_C(143), UINT8_C( 66), UINT8_C(236), UINT8_C(107),
        UINT8_C(179), UINT8_C( 89), UINT8_C(166), UINT8_C( 51), UINT8_C(170), UINT8_C(130), UINT8_C( 67), UINT8_C(195),
        UINT8_C(166), UINT8_C( 40), UINT8_C(111), UINT8_C(190), UINT8_C(132), UINT8_C( 35), UINT8_C( 54), UINT8_C( 68),
        UINT8_C(120), UINT8_C(175), UINT8_C(136), UINT8_C( 81), UINT8_C(155), UINT8_C(227), UINT8_C(103), UINT8_C( 59),
        UINT8_C(  9), UINT8_C(131), UINT8_C(168), UINT8_C(121), UINT8_C( 66), UINT8_C( 78), UINT8_C( 15), UINT8_C( 72) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi8(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_epu8(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_epu8");
    easysimd_test_x86_assert_equal_u8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask64 k;
    const uint8_t a[64];
    const uint8_t b[64];
    const uint8_t r[64];
  } test_vec[] = {
    { UINT64_C(17082286413152396594),
      { UINT8_C(176), UINT8_C(246), UINT8_C( 65), UINT8_C(226), UINT8_C( 97), UINT8_C(118), UINT8_C( 24), UINT8_C( 47),
        UINT8_C( 52), UINT8_C(202), UINT8_C( 14), UINT8_C( 44), UINT8_C(237), UINT8_C(159), UINT8_C( 11), UINT8_C(251),
        UINT8_C(132), UINT8_C(170), UINT8_C( 10), UINT8_C(208), UINT8_C( 36), UINT8_C( 92), UINT8_C( 27), UINT8_C( 86),
        UINT8_C(129), UINT8_C(137), UINT8_C(253), UINT8_C(125), UINT8_C(  1), UINT8_C( 13), UINT8_C(107), UINT8_C(178),
        UINT8_C(  3), UINT8_C(172), UINT8_C(148), UINT8_C(101), UINT8_C( 34), UINT8_C(172), UINT8_C(148), UINT8_C( 86),
        UINT8_C(119), UINT8_C(162), UINT8_C(131), UINT8_C(100), UINT8_C( 66), UINT8_C(142), UINT8_C( 95), UINT8_C(198),
        UINT8_C( 56), UINT8_C(106), UINT8_C(151), UINT8_C( 92), UINT8_C(198), UINT8_C(178), UINT8_C(178), UINT8_C( 72),
        UINT8_C( 59), UINT8_C(175), UINT8_C(197), UINT8_C( 61), UINT8_C(189), UINT8_C( 48), UINT8_C(239), UINT8_C(192) },
      { UINT8_C(220), UINT8_C(131), UINT8_C( 37), UINT8_C(254), UINT8_C( 48), UINT8_C(185), UINT8_C( 85), UINT8_C(167),
        UINT8_C( 92), UINT8_C(216), UINT8_C( 11), UINT8_C(158), UINT8_C(102), UINT8_C(107), UINT8_C(100), UINT8_C(158),
        UINT8_C(213), UINT8_C(251), UINT8_C(251), UINT8_C(155), UINT8_C(173), UINT8_C(173), UINT8_C(227), UINT8_C(233),
        UINT8_C( 93), UINT8_C(169), UINT8_C( 38), UINT8_C( 26), UINT8_C(217), UINT8_C( 21), UINT8_C(218), UINT8_C(182),
        UINT8_C(152), UINT8_C(  0), UINT8_C(180), UINT8_C(200), UINT8_C(185), UINT8_C(  9), UINT8_C(111), UINT8_C( 21),
        UINT8_C(225), UINT8_C(123), UINT8_C(179), UINT8_C( 71), UINT8_C(230), UINT8_C( 24), UINT8_C(230), UINT8_C(187),
        UINT8_C( 19), UINT8_C(225), UINT8_C( 86), UINT8_C(193), UINT8_C(142), UINT8_C( 58), UINT8_C(170), UINT8_C(235),
        UINT8_C(227), UINT8_C(208), UINT8_C(  5), UINT8_C(188), UINT8_C(229), UINT8_C(224), UINT8_C(114), UINT8_C(125) },
      { UINT8_C(  0), UINT8_C(131), UINT8_C(  0), UINT8_C(  0), UINT8_C( 48), UINT8_C(118), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 52), UINT8_C(  0), UINT8_C( 11), UINT8_C(  0), UINT8_C(  0), UINT8_C(107), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(170), UINT8_C( 10), UINT8_C(155), UINT8_C(  0), UINT8_C( 92), UINT8_C( 27), UINT8_C(  0),
        UINT8_C( 93), UINT8_C(137), UINT8_C( 38), UINT8_C(  0), UINT8_C(  0), UINT8_C( 13), UINT8_C(  0), UINT8_C(178),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(148), UINT8_C(101), UINT8_C( 34), UINT8_C(  9), UINT8_C(111), UINT8_C( 21),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 71), UINT8_C( 66), UINT8_C( 24), UINT8_C( 95), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(142), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 59), UINT8_C(  0), UINT8_C(  5), UINT8_C( 61), UINT8_C(  0), UINT8_C( 48), UINT8_C(114), UINT8_C(125) } },
    { UINT64_C( 1346494033941637088),
      { UINT8_C( 48), UINT8_C( 98), UINT8_C( 89), UINT8_C( 22), UINT8_C(122), UINT8_C( 63), UINT8_C(209), UINT8_C(142),
        UINT8_C( 32), UINT8_C( 40), UINT8_C( 79), UINT8_C(175), UINT8_C( 98), UINT8_C(249), UINT8_C(154), UINT8_C( 69),
        UINT8_C(201), UINT8_C(160), UINT8_C(  1), UINT8_C(174), UINT8_C(128), UINT8_C(116), UINT8_C( 43), UINT8_C( 96),
        UINT8_C(155), UINT8_C(113), UINT8_C(249), UINT8_C(203), UINT8_C( 39), UINT8_C(168), UINT8_C(221), UINT8_C( 87),
        UINT8_C( 11), UINT8_C( 55), UINT8_C(110), UINT8_C(133), UINT8_C(118), UINT8_C( 63), UINT8_C( 19), UINT8_C(151),
        UINT8_C(103), UINT8_C( 98), UINT8_C( 70), UINT8_C(201), UINT8_C( 91), UINT8_C(224), UINT8_C( 14), UINT8_C( 36),
        UINT8_C(128), UINT8_C( 16), UINT8_C(210), UINT8_C(  0), UINT8_C(132), UINT8_C(254), UINT8_C( 96), UINT8_C( 31),
        UINT8_C(111), UINT8_C( 90), UINT8_C(234), UINT8_C(150), UINT8_C(  2), UINT8_C(200), UINT8_C(238), UINT8_C( 13) },
      {    UINT8_MAX, UINT8_C( 92), UINT8_C(147), UINT8_C(117), UINT8_C(155), UINT8_C(166), UINT8_C( 12), UINT8_C(  3),
        UINT8_C(  9), UINT8_C( 82), UINT8_C(204), UINT8_C(100), UINT8_C( 51), UINT8_C(219), UINT8_C(137), UINT8_C(179),
        UINT8_C(235), UINT8_C( 91), UINT8_C(180), UINT8_C(111), UINT8_C( 89), UINT8_C( 20), UINT8_C(142), UINT8_C(201),
        UINT8_C(110), UINT8_C(120), UINT8_C( 95), UINT8_C(113), UINT8_C( 64), UINT8_C( 77), UINT8_C(126), UINT8_C( 63),
        UINT8_C(169), UINT8_C( 17), UINT8_C(181), UINT8_C( 69), UINT8_C(184), UINT8_C(193), UINT8_C( 72), UINT8_C(193),
        UINT8_C( 20), UINT8_C( 20), UINT8_C( 37), UINT8_C( 71), UINT8_C(239), UINT8_C(174), UINT8_C(250), UINT8_C(218),
        UINT8_C( 10), UINT8_C(174), UINT8_C( 73), UINT8_C( 99), UINT8_C(195), UINT8_C(215), UINT8_C( 44), UINT8_C( 49),
        UINT8_C( 80), UINT8_C(140), UINT8_C(162), UINT8_C(144), UINT8_C(217), UINT8_C( 33), UINT8_C(208), UINT8_C(131) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 63), UINT8_C( 12), UINT8_C(  3),
        UINT8_C(  9), UINT8_C( 40), UINT8_C( 79), UINT8_C(  0), UINT8_C(  0), UINT8_C(219), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C( 91), UINT8_C(  1), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 43), UINT8_C(  0),
        UINT8_C(110), UINT8_C(  0), UINT8_C(  0), UINT8_C(113), UINT8_C( 39), UINT8_C(  0), UINT8_C(  0), UINT8_C( 63),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(118), UINT8_C( 63), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 20), UINT8_C(  0), UINT8_C( 37), UINT8_C(  0), UINT8_C( 91), UINT8_C(174), UINT8_C(  0), UINT8_C( 36),
        UINT8_C( 10), UINT8_C( 16), UINT8_C( 73), UINT8_C(  0), UINT8_C(  0), UINT8_C(215), UINT8_C(  0), UINT8_C( 31),
        UINT8_C(  0), UINT8_C( 90), UINT8_C(  0), UINT8_C(  0), UINT8_C(  2), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0) } },
    { UINT64_C( 6533333581203801394),
      { UINT8_C( 36), UINT8_C(209), UINT8_C(161), UINT8_C( 20), UINT8_C(127), UINT8_C(156), UINT8_C(238), UINT8_C(137),
        UINT8_C( 74), UINT8_C( 56), UINT8_C(237), UINT8_C( 13), UINT8_C( 15), UINT8_C( 25), UINT8_C( 63), UINT8_C( 95),
        UINT8_C(165), UINT8_C(225), UINT8_C(240), UINT8_C(127), UINT8_C(  2), UINT8_C(192), UINT8_C(  2), UINT8_C( 53),
        UINT8_C( 69), UINT8_C(202), UINT8_C( 31), UINT8_C(139), UINT8_C(218), UINT8_C(203), UINT8_C(230), UINT8_C(254),
        UINT8_C(156), UINT8_C(135), UINT8_C( 18), UINT8_C( 27), UINT8_C( 35), UINT8_C(  1), UINT8_C(165), UINT8_C(110),
        UINT8_C( 57), UINT8_C(146), UINT8_C(123), UINT8_C( 72), UINT8_C(171), UINT8_C(186), UINT8_C(168), UINT8_C( 81),
        UINT8_C(156), UINT8_C(152), UINT8_C(208), UINT8_C(158), UINT8_C( 88), UINT8_C(210), UINT8_C(211), UINT8_C(157),
        UINT8_C(156), UINT8_C(243), UINT8_C( 40), UINT8_C(118), UINT8_C(190), UINT8_C( 14), UINT8_C(116), UINT8_C( 90) },
      { UINT8_C(150), UINT8_C(135), UINT8_C(117), UINT8_C(185), UINT8_C(136), UINT8_C( 26), UINT8_C( 39), UINT8_C(193),
        UINT8_C(172), UINT8_C(163), UINT8_C(  9), UINT8_C( 88), UINT8_C( 93), UINT8_C(177), UINT8_C(169), UINT8_C(249),
        UINT8_C( 73), UINT8_C(121), UINT8_C(152), UINT8_C(161), UINT8_C( 75), UINT8_C(107), UINT8_C( 62), UINT8_C(231),
        UINT8_C( 94), UINT8_C(103), UINT8_C( 93), UINT8_C( 28), UINT8_C(117), UINT8_C(209), UINT8_C(118), UINT8_C( 11),
        UINT8_C( 88), UINT8_C(236), UINT8_C(197), UINT8_C(224), UINT8_C(  6), UINT8_C(236), UINT8_C(161), UINT8_C(179),
        UINT8_C(143), UINT8_C(171), UINT8_C( 11), UINT8_C(237), UINT8_C( 92), UINT8_C(180), UINT8_C(230), UINT8_C(166),
        UINT8_C( 45), UINT8_C(126), UINT8_C( 71), UINT8_C(120), UINT8_C(234), UINT8_C(134), UINT8_C( 95), UINT8_C( 72),
        UINT8_C(237), UINT8_C(188), UINT8_C(101), UINT8_C( 98), UINT8_C(141), UINT8_C(219), UINT8_C(110), UINT8_C(230) },
      { UINT8_C(  0), UINT8_C(135), UINT8_C(  0), UINT8_C(  0), UINT8_C(127), UINT8_C( 26), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 74), UINT8_C(  0), UINT8_C(  9), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 95),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(127), UINT8_C(  0), UINT8_C(  0), UINT8_C(  2), UINT8_C( 53),
        UINT8_C(  0), UINT8_C(103), UINT8_C(  0), UINT8_C( 28), UINT8_C(  0), UINT8_C(203), UINT8_C(118), UINT8_C( 11),
        UINT8_C(  0), UINT8_C(135), UINT8_C( 18), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(161), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 92), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 45), UINT8_C(126), UINT8_C(  0), UINT8_C(120), UINT8_C(  0), UINT8_C(134), UINT8_C(  0), UINT8_C( 72),
        UINT8_C(  0), UINT8_C(188), UINT8_C(  0), UINT8_C( 98), UINT8_C(141), UINT8_C(  0), UINT8_C(110), UINT8_C(  0) } },
    { UINT64_C(12646503714455434183),
      { UINT8_C( 19), UINT8_C(140), UINT8_C(156), UINT8_C(111), UINT8_C( 64), UINT8_C(130), UINT8_C( 21), UINT8_C(109),
        UINT8_C(  1), UINT8_C( 93), UINT8_C(229), UINT8_C(235), UINT8_C(227), UINT8_C( 68), UINT8_C( 51), UINT8_C(208),
        UINT8_C(  0), UINT8_C(152), UINT8_C( 50), UINT8_C(141), UINT8_C(116), UINT8_C(160), UINT8_C(115), UINT8_C( 59),
        UINT8_C(211), UINT8_C( 58), UINT8_C(  9), UINT8_C(243), UINT8_C(162), UINT8_C(138), UINT8_C(162), UINT8_C(181),
        UINT8_C( 22), UINT8_C( 62), UINT8_C( 36), UINT8_C( 86), UINT8_C(192), UINT8_C( 58), UINT8_C(195), UINT8_C(193),
        UINT8_C(151), UINT8_C(168), UINT8_C(172), UINT8_C(122), UINT8_C(236), UINT8_C(224), UINT8_C( 74), UINT8_C(236),
        UINT8_C(120), UINT8_C(124), UINT8_C(122), UINT8_C(236), UINT8_C( 29), UINT8_C(237), UINT8_C( 40), UINT8_C(240),
        UINT8_C( 39), UINT8_C( 49), UINT8_C(227), UINT8_C(201), UINT8_C(188), UINT8_C(133), UINT8_C(126), UINT8_C(210) },
      { UINT8_C(195), UINT8_C(163), UINT8_C( 41), UINT8_C(132), UINT8_C(221), UINT8_C(236), UINT8_C( 69), UINT8_C(116),
        UINT8_C(149), UINT8_C(242), UINT8_C(238), UINT8_C(129), UINT8_C(210), UINT8_C( 56), UINT8_C(110), UINT8_C( 74),
        UINT8_C(180), UINT8_C(232), UINT8_C( 55), UINT8_C(209), UINT8_C(213), UINT8_C( 95), UINT8_C(194), UINT8_C(253),
        UINT8_C(144), UINT8_C(165), UINT8_C(198), UINT8_C( 76), UINT8_C( 43), UINT8_C( 69), UINT8_C( 31), UINT8_C(238),
        UINT8_C(232), UINT8_C( 72), UINT8_C(114), UINT8_C(197), UINT8_C( 52), UINT8_C(184), UINT8_C( 57), UINT8_C(201),
        UINT8_C(170), UINT8_C( 39), UINT8_C( 75), UINT8_C(124), UINT8_C( 95), UINT8_C(185), UINT8_C(198), UINT8_C( 19),
        UINT8_C(161), UINT8_C(253), UINT8_C(229), UINT8_C(118), UINT8_C( 92), UINT8_C(167), UINT8_C(115), UINT8_C(237),
        UINT8_C( 76), UINT8_C( 58), UINT8_C( 57), UINT8_C(119), UINT8_C(127), UINT8_C( 88), UINT8_C(102), UINT8_C(103) },
      { UINT8_C( 19), UINT8_C(140), UINT8_C( 41), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 21), UINT8_C(109),
        UINT8_C(  1), UINT8_C( 93), UINT8_C(  0), UINT8_C(  0), UINT8_C(210), UINT8_C( 56), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(152), UINT8_C( 50), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(115), UINT8_C( 59),
        UINT8_C(  0), UINT8_C( 58), UINT8_C(  9), UINT8_C( 76), UINT8_C(  0), UINT8_C(  0), UINT8_C( 31), UINT8_C(181),
        UINT8_C( 22), UINT8_C( 62), UINT8_C( 36), UINT8_C( 86), UINT8_C( 52), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(122), UINT8_C(  0), UINT8_C(185), UINT8_C( 74), UINT8_C(  0),
        UINT8_C(120), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(237),
        UINT8_C( 39), UINT8_C( 49), UINT8_C( 57), UINT8_C(119), UINT8_C(  0), UINT8_C( 88), UINT8_C(  0), UINT8_C(103) } },
    { UINT64_C( 4223925173246285984),
      { UINT8_C(140), UINT8_C(233), UINT8_C(182), UINT8_C(235), UINT8_C(162), UINT8_C(125), UINT8_C(254), UINT8_C( 67),
        UINT8_C(122), UINT8_C(227), UINT8_C(186), UINT8_C(215), UINT8_C(138), UINT8_C( 45), UINT8_C(196), UINT8_C(215),
        UINT8_C(103), UINT8_C(253), UINT8_C( 78), UINT8_C(230), UINT8_C( 86), UINT8_C(180), UINT8_C( 77), UINT8_C(246),
        UINT8_C(141), UINT8_C(121), UINT8_C(203), UINT8_C( 29), UINT8_C(222), UINT8_C(106), UINT8_C( 88), UINT8_C(106),
        UINT8_C( 83), UINT8_C( 14), UINT8_C( 85), UINT8_C(246), UINT8_C(139), UINT8_C( 84), UINT8_C( 57), UINT8_C(  6),
        UINT8_C( 55), UINT8_C(243), UINT8_C(221), UINT8_C(194), UINT8_C( 33), UINT8_C(161), UINT8_C(153), UINT8_C(136),
        UINT8_C(158), UINT8_C(231), UINT8_C(111), UINT8_C(244), UINT8_C(156), UINT8_C(188), UINT8_C(235), UINT8_C( 41),
        UINT8_C( 54), UINT8_C(182), UINT8_C( 70), UINT8_C( 20), UINT8_C( 32), UINT8_C(158), UINT8_C(127), UINT8_C(116) },
      { UINT8_C(173), UINT8_C(212), UINT8_C(106), UINT8_C( 56), UINT8_C( 40), UINT8_C(163), UINT8_C( 62), UINT8_C( 96),
        UINT8_C(151), UINT8_C( 27), UINT8_C( 34), UINT8_C(184), UINT8_C(188), UINT8_C(187), UINT8_C( 64), UINT8_C( 91),
        UINT8_C(162), UINT8_C(175), UINT8_C( 79), UINT8_C( 62), UINT8_C(108), UINT8_C( 58), UINT8_C(103), UINT8_C(162),
        UINT8_C(241), UINT8_C(174), UINT8_C(182), UINT8_C( 17), UINT8_C( 76), UINT8_C( 53), UINT8_C(133), UINT8_C(249),
        UINT8_C( 10), UINT8_C(239), UINT8_C( 50), UINT8_C( 50), UINT8_C(147), UINT8_C(112), UINT8_C(146), UINT8_C( 42),
        UINT8_C(140), UINT8_C(180), UINT8_C(226), UINT8_C( 72), UINT8_C(111), UINT8_C( 34), UINT8_C(163), UINT8_C( 18),
        UINT8_C(210), UINT8_C(243), UINT8_C( 80), UINT8_C( 62), UINT8_C( 45), UINT8_C(184), UINT8_C(224), UINT8_C( 30),
        UINT8_C(102), UINT8_C(150), UINT8_C( 48), UINT8_C(178), UINT8_C(204), UINT8_C(181), UINT8_C(172), UINT8_C(214) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(125), UINT8_C(  0), UINT8_C( 67),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(184), UINT8_C(138), UINT8_C(  0), UINT8_C( 64), UINT8_C( 91),
        UINT8_C(  0), UINT8_C(  0), UINT8_C( 78), UINT8_C( 62), UINT8_C(  0), UINT8_C( 58), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(141), UINT8_C(  0), UINT8_C(182), UINT8_C(  0), UINT8_C( 76), UINT8_C(  0), UINT8_C( 88), UINT8_C(106),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(139), UINT8_C(  0), UINT8_C(  0), UINT8_C(  6),
        UINT8_C( 55), UINT8_C(  0), UINT8_C(221), UINT8_C(  0), UINT8_C(  0), UINT8_C( 34), UINT8_C(153), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(231), UINT8_C( 80), UINT8_C( 62), UINT8_C( 45), UINT8_C(  0), UINT8_C(  0), UINT8_C( 30),
        UINT8_C(  0), UINT8_C(150), UINT8_C(  0), UINT8_C( 20), UINT8_C( 32), UINT8_C(158), UINT8_C(  0), UINT8_C(  0) } },
    { UINT64_C(15736310808235794085),
      { UINT8_C( 79), UINT8_C( 68), UINT8_C( 35), UINT8_C(191), UINT8_C(102), UINT8_C(198), UINT8_C(209), UINT8_C( 56),
        UINT8_C(185), UINT8_C( 33), UINT8_C(118), UINT8_C(231), UINT8_C(217), UINT8_C( 86), UINT8_C(  5), UINT8_C( 63),
        UINT8_C(237), UINT8_C( 53), UINT8_C(242), UINT8_C(185), UINT8_C(235), UINT8_C(158), UINT8_C(143), UINT8_C(144),
        UINT8_C(124), UINT8_C(151), UINT8_C(200), UINT8_C(202), UINT8_C( 50), UINT8_C( 42), UINT8_C(165), UINT8_C(130),
        UINT8_C(110), UINT8_C(200), UINT8_C( 65), UINT8_C(212), UINT8_C(142), UINT8_C( 18), UINT8_C( 13), UINT8_C( 72),
        UINT8_C( 51), UINT8_C(131), UINT8_C( 47), UINT8_C( 13), UINT8_C(218), UINT8_C( 52), UINT8_C( 76), UINT8_C(199),
        UINT8_C(106), UINT8_C( 62), UINT8_C(128), UINT8_C( 85), UINT8_C(220), UINT8_C( 15), UINT8_C(229), UINT8_C( 88),
        UINT8_C(166), UINT8_C(173), UINT8_C( 35), UINT8_C(217), UINT8_C(215), UINT8_C(200), UINT8_C( 91), UINT8_C( 69) },
      { UINT8_C(144), UINT8_C(156), UINT8_C( 25), UINT8_C( 30), UINT8_C(174), UINT8_C( 38), UINT8_C(102), UINT8_C(225),
        UINT8_C(170), UINT8_C(149), UINT8_C(238), UINT8_C(132), UINT8_C(202), UINT8_C( 59), UINT8_C( 75), UINT8_C( 52),
        UINT8_C(121), UINT8_C(203), UINT8_C(137), UINT8_C( 86), UINT8_C(218), UINT8_C(110), UINT8_C(174), UINT8_C(128),
        UINT8_C( 27), UINT8_C(209), UINT8_C( 89), UINT8_C(242), UINT8_C(153), UINT8_C(180), UINT8_C( 55), UINT8_C( 41),
        UINT8_C( 80), UINT8_C( 80), UINT8_C( 72), UINT8_C(254), UINT8_C(119), UINT8_C(174), UINT8_C(224), UINT8_C( 33),
        UINT8_C( 68), UINT8_C(206), UINT8_C(165), UINT8_C( 14), UINT8_C(  9), UINT8_C(240), UINT8_C( 66), UINT8_C(131),
        UINT8_C(187), UINT8_C(203), UINT8_C(217), UINT8_C(149), UINT8_C( 57), UINT8_C(135), UINT8_C( 21), UINT8_C( 84),
        UINT8_C( 89), UINT8_C(111), UINT8_C( 70), UINT8_C(242), UINT8_C( 35), UINT8_C(125), UINT8_C( 28), UINT8_C(116) },
      { UINT8_C( 79), UINT8_C(  0), UINT8_C( 25), UINT8_C(  0), UINT8_C(  0), UINT8_C( 38), UINT8_C(  0), UINT8_C( 56),
        UINT8_C(  0), UINT8_C( 33), UINT8_C(118), UINT8_C(132), UINT8_C(202), UINT8_C(  0), UINT8_C(  5), UINT8_C( 52),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 86), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(202), UINT8_C( 50), UINT8_C( 42), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C( 80), UINT8_C( 65), UINT8_C(212), UINT8_C(  0), UINT8_C(  0), UINT8_C( 13), UINT8_C(  0),
        UINT8_C( 51), UINT8_C(131), UINT8_C(  0), UINT8_C( 13), UINT8_C(  9), UINT8_C(  0), UINT8_C(  0), UINT8_C(131),
        UINT8_C(  0), UINT8_C( 62), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 15), UINT8_C( 21), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(111), UINT8_C(  0), UINT8_C(217), UINT8_C( 35), UINT8_C(  0), UINT8_C( 28), UINT8_C( 69) } },
    { UINT64_C( 6225472298320815309),
      { UINT8_C( 33), UINT8_C( 10), UINT8_C(100), UINT8_C( 42), UINT8_C(250), UINT8_C(166), UINT8_C(173), UINT8_C(181),
        UINT8_C(113), UINT8_C(134), UINT8_C( 74), UINT8_C(170), UINT8_C( 14), UINT8_C( 96), UINT8_C(254), UINT8_C(103),
        UINT8_C(207), UINT8_C( 68), UINT8_C( 89), UINT8_C(242), UINT8_C(193), UINT8_C(117), UINT8_C(102), UINT8_C(143),
        UINT8_C(217), UINT8_C(217), UINT8_C(211), UINT8_C(236), UINT8_C( 43), UINT8_C( 57), UINT8_C( 66), UINT8_C( 76),
        UINT8_C( 67), UINT8_C(167), UINT8_C(119), UINT8_C( 62), UINT8_C( 77), UINT8_C( 36), UINT8_C(243), UINT8_C(191),
        UINT8_C(171), UINT8_C( 62), UINT8_C(105), UINT8_C(185), UINT8_C(158), UINT8_C(104), UINT8_C( 32), UINT8_C(109),
        UINT8_C(172), UINT8_C(121), UINT8_C( 95), UINT8_C(110), UINT8_C(239), UINT8_C(198), UINT8_C(253), UINT8_C(200),
        UINT8_C(159), UINT8_C(208), UINT8_C(180), UINT8_C(202), UINT8_C(  9), UINT8_C(247), UINT8_C( 23), UINT8_C( 77) },
      { UINT8_C(158), UINT8_C(142), UINT8_C(139), UINT8_C(235), UINT8_C(178), UINT8_C(126), UINT8_C(170), UINT8_C( 93),
        UINT8_C(188), UINT8_C( 20), UINT8_C( 22), UINT8_C( 90), UINT8_C(124), UINT8_C( 54), UINT8_C(199), UINT8_C( 40),
        UINT8_C(176), UINT8_C( 39), UINT8_C(150), UINT8_C(159), UINT8_C(237), UINT8_C(147), UINT8_C(103), UINT8_C(140),
        UINT8_C(100), UINT8_C( 28), UINT8_C( 86), UINT8_C(109), UINT8_C( 19), UINT8_C(109), UINT8_C(186), UINT8_C(177),
        UINT8_C(251), UINT8_C( 69), UINT8_C(156), UINT8_C(174), UINT8_C(196), UINT8_C( 71), UINT8_C( 11), UINT8_C(128),
        UINT8_C( 91), UINT8_C( 34), UINT8_C(219), UINT8_C(215), UINT8_C( 88), UINT8_C(162),    UINT8_MAX, UINT8_C(  8),
        UINT8_C(201), UINT8_C(150), UINT8_C(167), UINT8_C(182), UINT8_C( 41), UINT8_C( 15), UINT8_C( 66), UINT8_C(141),
        UINT8_C( 43), UINT8_C(153), UINT8_C(251), UINT8_C( 62), UINT8_C(  6), UINT8_C(181), UINT8_C(239), UINT8_C(  2) },
      { UINT8_C( 33), UINT8_C(  0), UINT8_C(100), UINT8_C( 42), UINT8_C(  0), UINT8_C(  0), UINT8_C(170), UINT8_C( 93),
        UINT8_C(  0), UINT8_C(  0), UINT8_C( 22), UINT8_C(  0), UINT8_C(  0), UINT8_C( 54), UINT8_C(199), UINT8_C(  0),
        UINT8_C(  0), UINT8_C( 39), UINT8_C(  0), UINT8_C(  0), UINT8_C(193), UINT8_C(117), UINT8_C(102), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C( 86), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 66), UINT8_C(  0),
        UINT8_C(  0), UINT8_C( 69), UINT8_C(  0), UINT8_C(  0), UINT8_C( 77), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C( 34), UINT8_C(  0), UINT8_C(  0), UINT8_C( 88), UINT8_C(  0), UINT8_C( 32), UINT8_C(  0),
        UINT8_C(172), UINT8_C(  0), UINT8_C( 95), UINT8_C(  0), UINT8_C(  0), UINT8_C( 15), UINT8_C( 66), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(153), UINT8_C(180), UINT8_C(  0), UINT8_C(  6), UINT8_C(  0), UINT8_C( 23), UINT8_C(  0) } },
    { UINT64_C( 3260531169073073147),
      { UINT8_C(221), UINT8_C( 26), UINT8_C(  4), UINT8_C( 54), UINT8_C(189), UINT8_C(  4), UINT8_C( 62), UINT8_C(134),
        UINT8_C(154), UINT8_C(230), UINT8_C( 61), UINT8_C(195), UINT8_C(245), UINT8_C(127), UINT8_C( 81), UINT8_C( 32),
        UINT8_C( 24), UINT8_C( 76), UINT8_C( 94), UINT8_C( 31), UINT8_C(  1), UINT8_C( 77), UINT8_C( 33), UINT8_C(252),
        UINT8_C(216), UINT8_C(209), UINT8_C(187), UINT8_C(171), UINT8_C(140), UINT8_C(251), UINT8_C(216), UINT8_C(106),
        UINT8_C( 21), UINT8_C(221), UINT8_C(160), UINT8_C(210), UINT8_C(225), UINT8_C(222), UINT8_C( 89), UINT8_C(123),
        UINT8_C(196), UINT8_C(150), UINT8_C( 62), UINT8_C(185), UINT8_C( 21), UINT8_C(143), UINT8_C(217), UINT8_C( 46),
        UINT8_C(219), UINT8_C( 55), UINT8_C( 77), UINT8_C(221), UINT8_C(132), UINT8_C(110), UINT8_C(217), UINT8_C( 93),
        UINT8_C( 63), UINT8_C(149), UINT8_C(  8), UINT8_C(203), UINT8_C(144), UINT8_C(224), UINT8_C( 53), UINT8_C(165) },
      { UINT8_C(189), UINT8_C(213), UINT8_C(120), UINT8_C(158), UINT8_C(180), UINT8_C(209), UINT8_C( 25), UINT8_C(120),
        UINT8_C(103), UINT8_C( 88), UINT8_C( 50), UINT8_C(124), UINT8_C(231), UINT8_C( 11), UINT8_C(170), UINT8_C(195),
        UINT8_C( 67), UINT8_C(247), UINT8_C(160), UINT8_C(199), UINT8_C(101), UINT8_C(121), UINT8_C( 36), UINT8_C(164),
        UINT8_C( 14), UINT8_C( 44), UINT8_C(112), UINT8_C(158), UINT8_C( 13), UINT8_C(165), UINT8_C( 68), UINT8_C(202),
        UINT8_C(123), UINT8_C(188), UINT8_C(105), UINT8_C( 47), UINT8_C(141), UINT8_C(130), UINT8_C(167), UINT8_C(244),
        UINT8_C(218), UINT8_C(217), UINT8_C(112), UINT8_C(194), UINT8_C(229), UINT8_C( 27), UINT8_C(133), UINT8_C( 40),
        UINT8_C( 18), UINT8_C( 37), UINT8_C(239), UINT8_C(120), UINT8_C(158), UINT8_C( 20), UINT8_C( 28), UINT8_C(173),
        UINT8_C( 64), UINT8_C(140), UINT8_C( 75), UINT8_C( 77), UINT8_C( 50), UINT8_C(143), UINT8_C( 24), UINT8_C(173) },
      { UINT8_C(189), UINT8_C( 26), UINT8_C(  0), UINT8_C( 54), UINT8_C(180), UINT8_C(  4), UINT8_C( 25), UINT8_C(120),
        UINT8_C(103), UINT8_C( 88), UINT8_C(  0), UINT8_C(124), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 32),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  1), UINT8_C( 77), UINT8_C(  0), UINT8_C(164),
        UINT8_C( 14), UINT8_C( 44), UINT8_C(112), UINT8_C(158), UINT8_C( 13), UINT8_C(165), UINT8_C(  0), UINT8_C(106),
        UINT8_C(  0), UINT8_C(188), UINT8_C(  0), UINT8_C(  0), UINT8_C(141), UINT8_C(  0), UINT8_C( 89), UINT8_C(123),
        UINT8_C(196), UINT8_C(150), UINT8_C(  0), UINT8_C(185), UINT8_C( 21), UINT8_C( 27), UINT8_C(  0), UINT8_C( 40),
        UINT8_C( 18), UINT8_C( 37), UINT8_C( 77), UINT8_C(120), UINT8_C(132), UINT8_C( 20), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 63), UINT8_C(  0), UINT8_C(  8), UINT8_C( 77), UINT8_C(  0), UINT8_C(143), UINT8_C(  0), UINT8_C(  0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_epu8(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_epu8");
    easysimd_test_x86_assert_equal_u8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_min_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { {  INT16_C( 14691), -INT16_C(  2237),  INT16_C(  8698),  INT16_C(  9348),  INT16_C( 13857), -INT16_C( 10584),  INT16_C( 25854), -INT16_C(  6420),
         INT16_C( 17420),  INT16_C(  1517), -INT16_C(  5283),  INT16_C( 26495),  INT16_C(  7532), -INT16_C(  2781),  INT16_C(  9520), -INT16_C( 27403),
         INT16_C( 14430),  INT16_C( 22667),  INT16_C(  3929),  INT16_C( 31356),  INT16_C(  9541),  INT16_C( 17232),  INT16_C( 15497), -INT16_C( 27350),
         INT16_C(  6016), -INT16_C(  8550),  INT16_C(  6658),  INT16_C( 28485),  INT16_C( 26679),  INT16_C( 26724),  INT16_C( 22926), -INT16_C(  4868) },
      { -INT16_C( 30831), -INT16_C(  5563), -INT16_C( 15978), -INT16_C(  9371), -INT16_C( 18970),  INT16_C( 28447),  INT16_C( 18930),  INT16_C( 29188),
        -INT16_C( 24736),  INT16_C( 25168), -INT16_C( 26951), -INT16_C(  3887),  INT16_C( 14078), -INT16_C( 29608),  INT16_C( 21647),  INT16_C(  8569),
        -INT16_C( 16677),  INT16_C( 28939),  INT16_C( 28799),  INT16_C( 26189),  INT16_C( 27686),  INT16_C(  6357), -INT16_C(  9547),  INT16_C(  5514),
        -INT16_C(  9351),  INT16_C( 12919),  INT16_C( 18801),  INT16_C( 28450),  INT16_C( 31615),  INT16_C(  3836),  INT16_C( 30159), -INT16_C( 21713) },
      { -INT16_C( 30831), -INT16_C(  5563), -INT16_C( 15978), -INT16_C(  9371), -INT16_C( 18970), -INT16_C( 10584),  INT16_C( 18930), -INT16_C(  6420),
        -INT16_C( 24736),  INT16_C(  1517), -INT16_C( 26951), -INT16_C(  3887),  INT16_C(  7532), -INT16_C( 29608),  INT16_C(  9520), -INT16_C( 27403),
        -INT16_C( 16677),  INT16_C( 22667),  INT16_C(  3929),  INT16_C( 26189),  INT16_C(  9541),  INT16_C(  6357), -INT16_C(  9547), -INT16_C( 27350),
        -INT16_C(  9351), -INT16_C(  8550),  INT16_C(  6658),  INT16_C( 28450),  INT16_C( 26679),  INT16_C(  3836),  INT16_C( 22926), -INT16_C( 21713) } },
    { {  INT16_C( 15155), -INT16_C( 19940),  INT16_C( 27051), -INT16_C( 12008), -INT16_C(  4395), -INT16_C( 29975),  INT16_C( 29896),  INT16_C( 16799),
         INT16_C(  5967), -INT16_C( 16269), -INT16_C( 27296), -INT16_C(  8401),  INT16_C( 11024), -INT16_C(  7955),  INT16_C(  7584), -INT16_C( 11381),
        -INT16_C( 22696),  INT16_C(   902), -INT16_C( 25071), -INT16_C(  6443), -INT16_C( 16756),  INT16_C( 21617),  INT16_C(  4146), -INT16_C( 32363),
         INT16_C(  2087), -INT16_C( 30911),  INT16_C( 29086), -INT16_C( 20890),  INT16_C( 21660),  INT16_C( 15758),  INT16_C(  6513), -INT16_C( 14064) },
      { -INT16_C( 26943), -INT16_C( 11572), -INT16_C( 24267), -INT16_C( 15944),  INT16_C( 10592), -INT16_C( 28138), -INT16_C( 21702),  INT16_C( 24852),
         INT16_C( 21940),  INT16_C( 21225),  INT16_C( 20422),  INT16_C( 25344), -INT16_C( 28765),  INT16_C(  5280), -INT16_C( 20312),  INT16_C( 27101),
        -INT16_C( 21945),  INT16_C( 31803), -INT16_C(  2997), -INT16_C( 21699),  INT16_C( 21277),  INT16_C( 22334),  INT16_C( 21247), -INT16_C( 19527),
        -INT16_C( 23897),  INT16_C( 28165),  INT16_C(  1521), -INT16_C( 27183),  INT16_C( 29076),  INT16_C( 15785), -INT16_C( 30943),  INT16_C( 26790) },
      { -INT16_C( 26943), -INT16_C( 19940), -INT16_C( 24267), -INT16_C( 15944), -INT16_C(  4395), -INT16_C( 29975), -INT16_C( 21702),  INT16_C( 16799),
         INT16_C(  5967), -INT16_C( 16269), -INT16_C( 27296), -INT16_C(  8401), -INT16_C( 28765), -INT16_C(  7955), -INT16_C( 20312), -INT16_C( 11381),
        -INT16_C( 22696),  INT16_C(   902), -INT16_C( 25071), -INT16_C( 21699), -INT16_C( 16756),  INT16_C( 21617),  INT16_C(  4146), -INT16_C( 32363),
        -INT16_C( 23897), -INT16_C( 30911),  INT16_C(  1521), -INT16_C( 27183),  INT16_C( 21660),  INT16_C( 15758), -INT16_C( 30943), -INT16_C( 14064) } },
    { { -INT16_C(  7631),  INT16_C( 31972),  INT16_C(  8918), -INT16_C(  3288),  INT16_C( 26229),  INT16_C( 29771),  INT16_C(  1208),  INT16_C( 24359),
         INT16_C( 11430), -INT16_C( 26675), -INT16_C( 25038), -INT16_C( 14804), -INT16_C( 10737),  INT16_C( 12547), -INT16_C( 21923), -INT16_C( 29031),
         INT16_C( 32396),  INT16_C( 25098),  INT16_C( 12960),  INT16_C(  5461), -INT16_C( 24424),  INT16_C( 20618), -INT16_C( 20060),  INT16_C( 19120),
         INT16_C( 32222),  INT16_C(  4322),  INT16_C(  3612),  INT16_C( 11222), -INT16_C(  9500),  INT16_C( 16732), -INT16_C(  2428),  INT16_C(  4303) },
      { -INT16_C(  9612),  INT16_C(  5234), -INT16_C( 14580), -INT16_C( 23255), -INT16_C( 19608),  INT16_C(  3317), -INT16_C( 23195),  INT16_C( 17239),
         INT16_C( 14627),  INT16_C( 16211),  INT16_C( 10567),  INT16_C( 11370), -INT16_C( 14589), -INT16_C( 30867),  INT16_C( 15805),  INT16_C( 12695),
         INT16_C(  2327),  INT16_C(  9029),  INT16_C( 28369),  INT16_C( 14792), -INT16_C( 16862), -INT16_C( 30907), -INT16_C( 25501), -INT16_C( 31030),
         INT16_C(  7637),  INT16_C(  7621),  INT16_C( 12358),  INT16_C( 19017), -INT16_C( 18697), -INT16_C( 19247),  INT16_C( 27123),  INT16_C(  2789) },
      { -INT16_C(  9612),  INT16_C(  5234), -INT16_C( 14580), -INT16_C( 23255), -INT16_C( 19608),  INT16_C(  3317), -INT16_C( 23195),  INT16_C( 17239),
         INT16_C( 11430), -INT16_C( 26675), -INT16_C( 25038), -INT16_C( 14804), -INT16_C( 14589), -INT16_C( 30867), -INT16_C( 21923), -INT16_C( 29031),
         INT16_C(  2327),  INT16_C(  9029),  INT16_C( 12960),  INT16_C(  5461), -INT16_C( 24424), -INT16_C( 30907), -INT16_C( 25501), -INT16_C( 31030),
         INT16_C(  7637),  INT16_C(  4322),  INT16_C(  3612),  INT16_C( 11222), -INT16_C( 18697), -INT16_C( 19247), -INT16_C(  2428),  INT16_C(  2789) } },
    { {  INT16_C( 10866),  INT16_C( 17198), -INT16_C(  2408), -INT16_C( 17796), -INT16_C( 15692),  INT16_C(  6209),  INT16_C(  2910),  INT16_C( 13470),
         INT16_C( 25640),  INT16_C( 28497), -INT16_C( 25964), -INT16_C( 29767), -INT16_C( 30128),  INT16_C( 17471),  INT16_C(  9459),  INT16_C( 26190),
         INT16_C( 31822), -INT16_C(  6487),  INT16_C(  9843),  INT16_C( 10145), -INT16_C(  7448),  INT16_C( 17983), -INT16_C(  8466),  INT16_C(  5754),
        -INT16_C( 13502), -INT16_C( 10619),  INT16_C( 15973), -INT16_C( 18847), -INT16_C( 24375), -INT16_C( 17158),  INT16_C( 18628),  INT16_C(  4642) },
      { -INT16_C( 13115),  INT16_C( 14584), -INT16_C( 26126), -INT16_C(  9633), -INT16_C( 24708),  INT16_C( 27168), -INT16_C( 25731), -INT16_C( 16512),
         INT16_C(  1638), -INT16_C( 13163), -INT16_C(  2492),  INT16_C(  3458),  INT16_C( 31894),  INT16_C( 23242), -INT16_C(  4924), -INT16_C( 30356),
         INT16_C( 25784), -INT16_C( 21823),  INT16_C(  8702),  INT16_C( 31364), -INT16_C( 23104),  INT16_C( 15844),  INT16_C( 25664), -INT16_C( 22788),
        -INT16_C( 28310), -INT16_C( 20622), -INT16_C(  2937),  INT16_C(  7612), -INT16_C( 31120),  INT16_C( 13687), -INT16_C(  7309),  INT16_C( 11198) },
      { -INT16_C( 13115),  INT16_C( 14584), -INT16_C( 26126), -INT16_C( 17796), -INT16_C( 24708),  INT16_C(  6209), -INT16_C( 25731), -INT16_C( 16512),
         INT16_C(  1638), -INT16_C( 13163), -INT16_C( 25964), -INT16_C( 29767), -INT16_C( 30128),  INT16_C( 17471), -INT16_C(  4924), -INT16_C( 30356),
         INT16_C( 25784), -INT16_C( 21823),  INT16_C(  8702),  INT16_C( 10145), -INT16_C( 23104),  INT16_C( 15844), -INT16_C(  8466), -INT16_C( 22788),
        -INT16_C( 28310), -INT16_C( 20622), -INT16_C(  2937), -INT16_C( 18847), -INT16_C( 31120), -INT16_C( 17158), -INT16_C(  7309),  INT16_C(  4642) } },
    { { -INT16_C( 32697),  INT16_C( 17878),  INT16_C( 23201),  INT16_C( 25023), -INT16_C( 23553),  INT16_C( 16286), -INT16_C( 26104),  INT16_C( 29414),
         INT16_C( 22571), -INT16_C( 19935), -INT16_C(  8627), -INT16_C( 16945),  INT16_C( 18020), -INT16_C( 10254), -INT16_C( 20183),  INT16_C( 28675),
        -INT16_C(  9935), -INT16_C( 11594),  INT16_C( 30003),  INT16_C( 13107), -INT16_C( 12007),  INT16_C(  8562),  INT16_C( 22635), -INT16_C( 26989),
        -INT16_C( 19023), -INT16_C(   440),  INT16_C(  6035), -INT16_C(  2117), -INT16_C( 20899), -INT16_C( 31025), -INT16_C( 11681), -INT16_C( 28426) },
      { -INT16_C( 21333), -INT16_C(  8606), -INT16_C( 27358),  INT16_C( 15121), -INT16_C( 31642), -INT16_C( 11940), -INT16_C(  4132), -INT16_C( 29337),
        -INT16_C( 20572),  INT16_C( 14219),  INT16_C( 18374),  INT16_C(  9007), -INT16_C(   267),  INT16_C( 21673), -INT16_C( 24624),  INT16_C( 31716),
         INT16_C( 17996),  INT16_C( 28249),  INT16_C( 27611),  INT16_C( 16809),  INT16_C(  1519), -INT16_C( 13550),  INT16_C( 31220), -INT16_C( 26279),
        -INT16_C(  7128), -INT16_C(  4400), -INT16_C(   213),  INT16_C(  8209), -INT16_C( 17667), -INT16_C( 12940),  INT16_C( 22617), -INT16_C( 23224) },
      { -INT16_C( 32697), -INT16_C(  8606), -INT16_C( 27358),  INT16_C( 15121), -INT16_C( 31642), -INT16_C( 11940), -INT16_C( 26104), -INT16_C( 29337),
        -INT16_C( 20572), -INT16_C( 19935), -INT16_C(  8627), -INT16_C( 16945), -INT16_C(   267), -INT16_C( 10254), -INT16_C( 24624),  INT16_C( 28675),
        -INT16_C(  9935), -INT16_C( 11594),  INT16_C( 27611),  INT16_C( 13107), -INT16_C( 12007), -INT16_C( 13550),  INT16_C( 22635), -INT16_C( 26989),
        -INT16_C( 19023), -INT16_C(  4400), -INT16_C(   213), -INT16_C(  2117), -INT16_C( 20899), -INT16_C( 31025), -INT16_C( 11681), -INT16_C( 28426) } },
    { { -INT16_C( 23906),  INT16_C( 30995), -INT16_C( 17395), -INT16_C(   838), -INT16_C( 13119), -INT16_C( 18745),  INT16_C(  8261),  INT16_C( 27983),
         INT16_C(  7941),  INT16_C( 12379),  INT16_C( 27679),  INT16_C(  7249), -INT16_C( 15066), -INT16_C( 32534),  INT16_C( 12830), -INT16_C( 17371),
         INT16_C( 14804), -INT16_C(  7882), -INT16_C(  3851), -INT16_C( 18467), -INT16_C( 23107),  INT16_C(   621), -INT16_C( 17211), -INT16_C( 13712),
        -INT16_C( 13349), -INT16_C(  1285),  INT16_C( 19512),  INT16_C( 24087),  INT16_C(   273),  INT16_C( 12254),  INT16_C(  1075),  INT16_C(  2284) },
      {  INT16_C(  8765),  INT16_C( 13033), -INT16_C( 14574), -INT16_C( 12311),  INT16_C( 22124),  INT16_C( 12754),  INT16_C( 16914), -INT16_C(  4356),
        -INT16_C(  2291),  INT16_C( 17896), -INT16_C(   189),  INT16_C( 21668), -INT16_C( 32256),  INT16_C( 13444),  INT16_C( 28806), -INT16_C( 15556),
         INT16_C(  9618), -INT16_C( 23306), -INT16_C(  8212),  INT16_C( 22644),  INT16_C( 17974),  INT16_C( 18570), -INT16_C( 31096), -INT16_C( 27338),
         INT16_C(  8061), -INT16_C( 16165),  INT16_C( 32542),  INT16_C(  7956), -INT16_C( 26623), -INT16_C( 30637), -INT16_C( 28920), -INT16_C( 26037) },
      { -INT16_C( 23906),  INT16_C( 13033), -INT16_C( 17395), -INT16_C( 12311), -INT16_C( 13119), -INT16_C( 18745),  INT16_C(  8261), -INT16_C(  4356),
        -INT16_C(  2291),  INT16_C( 12379), -INT16_C(   189),  INT16_C(  7249), -INT16_C( 32256), -INT16_C( 32534),  INT16_C( 12830), -INT16_C( 17371),
         INT16_C(  9618), -INT16_C( 23306), -INT16_C(  8212), -INT16_C( 18467), -INT16_C( 23107),  INT16_C(   621), -INT16_C( 31096), -INT16_C( 27338),
        -INT16_C( 13349), -INT16_C( 16165),  INT16_C( 19512),  INT16_C(  7956), -INT16_C( 26623), -INT16_C( 30637), -INT16_C( 28920), -INT16_C( 26037) } },
    { {  INT16_C( 16820), -INT16_C( 24257), -INT16_C( 19679),  INT16_C( 22521), -INT16_C( 31751), -INT16_C( 32353), -INT16_C( 10743), -INT16_C( 31210),
        -INT16_C(  3595),  INT16_C(  4934),  INT16_C( 23408),  INT16_C( 29234), -INT16_C( 31245), -INT16_C(   774),  INT16_C( 17684), -INT16_C( 13930),
        -INT16_C( 10873), -INT16_C( 22422),  INT16_C( 25480), -INT16_C( 32257), -INT16_C( 24857), -INT16_C(  4094),  INT16_C(  6516),  INT16_C( 26999),
        -INT16_C( 17142),  INT16_C( 31613), -INT16_C( 20712),  INT16_C(  3309), -INT16_C(  6347),  INT16_C( 18696), -INT16_C( 25044), -INT16_C( 19694) },
      {  INT16_C( 31860), -INT16_C(   933),  INT16_C( 23264), -INT16_C( 14466), -INT16_C( 32519),  INT16_C( 28087),  INT16_C( 11929), -INT16_C( 23337),
         INT16_C( 21740),  INT16_C(  1055),  INT16_C(  3075),  INT16_C( 14352),  INT16_C(  6387),  INT16_C(  8066), -INT16_C( 27465),  INT16_C( 11219),
         INT16_C( 11793), -INT16_C(  3801), -INT16_C( 23159), -INT16_C( 32072),  INT16_C( 28454), -INT16_C( 16401), -INT16_C( 14690), -INT16_C( 30109),
        -INT16_C( 32230),  INT16_C(  7822), -INT16_C( 24690), -INT16_C( 32426), -INT16_C( 10057),  INT16_C( 28321),  INT16_C( 29805),  INT16_C( 32409) },
      {  INT16_C( 16820), -INT16_C( 24257), -INT16_C( 19679), -INT16_C( 14466), -INT16_C( 32519), -INT16_C( 32353), -INT16_C( 10743), -INT16_C( 31210),
        -INT16_C(  3595),  INT16_C(  1055),  INT16_C(  3075),  INT16_C( 14352), -INT16_C( 31245), -INT16_C(   774), -INT16_C( 27465), -INT16_C( 13930),
        -INT16_C( 10873), -INT16_C( 22422), -INT16_C( 23159), -INT16_C( 32257), -INT16_C( 24857), -INT16_C( 16401), -INT16_C( 14690), -INT16_C( 30109),
        -INT16_C( 32230),  INT16_C(  7822), -INT16_C( 24690), -INT16_C( 32426), -INT16_C( 10057),  INT16_C( 18696), -INT16_C( 25044), -INT16_C( 19694) } },
    { { -INT16_C( 15966),  INT16_C( 11119),  INT16_C( 10086), -INT16_C( 29523), -INT16_C( 25194),  INT16_C( 13388), -INT16_C( 20637),  INT16_C( 32446),
         INT16_C( 19762), -INT16_C( 16228), -INT16_C(  3348), -INT16_C( 23742), -INT16_C(  7221),  INT16_C( 14354), -INT16_C( 21673), -INT16_C(  1610),
         INT16_C(  9580), -INT16_C( 11483), -INT16_C( 11700), -INT16_C(  7585), -INT16_C( 21649), -INT16_C( 11497), -INT16_C( 10917), -INT16_C( 29359),
        -INT16_C(  4830),  INT16_C(  3661), -INT16_C( 28705), -INT16_C( 21838), -INT16_C( 15246), -INT16_C( 13854), -INT16_C( 26513), -INT16_C(  9021) },
      { -INT16_C(  5955),  INT16_C(  2479),  INT16_C(  3770),  INT16_C( 10988),  INT16_C(   954),  INT16_C(  5629),  INT16_C( 20184), -INT16_C(  1118),
        -INT16_C(  4293),  INT16_C(  6665), -INT16_C( 17537), -INT16_C(  3643), -INT16_C( 22657), -INT16_C(  4165),  INT16_C( 32320), -INT16_C(   565),
         INT16_C( 31334),  INT16_C(  8199), -INT16_C(  3192),  INT16_C( 16970),  INT16_C( 18422), -INT16_C( 12713), -INT16_C(  1643), -INT16_C( 12087),
        -INT16_C( 11287),  INT16_C( 26859), -INT16_C( 20338),  INT16_C(  3673),  INT16_C(  5207), -INT16_C( 26627), -INT16_C( 14190), -INT16_C(  1899) },
      { -INT16_C( 15966),  INT16_C(  2479),  INT16_C(  3770), -INT16_C( 29523), -INT16_C( 25194),  INT16_C(  5629), -INT16_C( 20637), -INT16_C(  1118),
        -INT16_C(  4293), -INT16_C( 16228), -INT16_C( 17537), -INT16_C( 23742), -INT16_C( 22657), -INT16_C(  4165), -INT16_C( 21673), -INT16_C(  1610),
         INT16_C(  9580), -INT16_C( 11483), -INT16_C( 11700), -INT16_C(  7585), -INT16_C( 21649), -INT16_C( 12713), -INT16_C( 10917), -INT16_C( 29359),
        -INT16_C( 11287),  INT16_C(  3661), -INT16_C( 28705), -INT16_C( 21838), -INT16_C( 15246), -INT16_C( 26627), -INT16_C( 26513), -INT16_C(  9021) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_min_epi16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_min_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { {  INT16_C(  9710),  INT16_C(  8237),  INT16_C( 26211),  INT16_C(  1978),  INT16_C( 14790),  INT16_C(  9775), -INT16_C( 11510), -INT16_C( 20158),
         INT16_C( 13508), -INT16_C( 18579),  INT16_C(  4844), -INT16_C(  4275), -INT16_C(  8222), -INT16_C( 28513),  INT16_C( 21293),  INT16_C(  7273),
        -INT16_C( 27015), -INT16_C(  9156), -INT16_C(  2308), -INT16_C( 15389),  INT16_C(  4911),  INT16_C( 14825),  INT16_C( 11238), -INT16_C( 21781),
         INT16_C( 22623),  INT16_C( 19298), -INT16_C( 20629),  INT16_C( 19770), -INT16_C(  9586), -INT16_C( 17187),  INT16_C( 17965), -INT16_C( 22824) },
      UINT32_C(3632403676),
      { -INT16_C( 31623),  INT16_C( 24436),  INT16_C( 24495),  INT16_C(  3593),  INT16_C( 27575),  INT16_C(  8794), -INT16_C( 27622), -INT16_C( 22161),
         INT16_C( 19566), -INT16_C( 25499),  INT16_C( 15762),  INT16_C( 28226), -INT16_C( 15023),  INT16_C( 23623), -INT16_C(  7637), -INT16_C( 23401),
         INT16_C(  2919),  INT16_C(  5635),  INT16_C(  3178),  INT16_C(  8485),  INT16_C( 32632), -INT16_C( 28092), -INT16_C( 19693), -INT16_C( 32197),
        -INT16_C( 24576), -INT16_C( 28130),  INT16_C( 24797),  INT16_C( 12033),  INT16_C( 18469),  INT16_C( 20619),  INT16_C(  8746), -INT16_C( 28172) },
      { -INT16_C(  2259), -INT16_C( 26712), -INT16_C( 13052),  INT16_C( 31929), -INT16_C(   692),  INT16_C( 24334),  INT16_C( 19120), -INT16_C( 20255),
        -INT16_C(    22), -INT16_C( 14269),  INT16_C( 17504), -INT16_C( 31241), -INT16_C( 32116), -INT16_C( 18730), -INT16_C( 13659), -INT16_C( 11704),
        -INT16_C(  3902), -INT16_C( 14742),  INT16_C(  9149),  INT16_C(  2370),  INT16_C( 20512), -INT16_C( 12184),  INT16_C( 19098), -INT16_C( 31359),
        -INT16_C( 15287), -INT16_C( 22195),  INT16_C( 17416), -INT16_C( 27601),  INT16_C(  1478),  INT16_C( 27466), -INT16_C( 27953), -INT16_C( 28354) },
      {  INT16_C(  9710),  INT16_C(  8237), -INT16_C( 13052),  INT16_C(  3593), -INT16_C(   692),  INT16_C(  9775), -INT16_C( 27622), -INT16_C( 22161),
         INT16_C( 13508), -INT16_C( 18579),  INT16_C( 15762), -INT16_C(  4275), -INT16_C( 32116), -INT16_C( 28513),  INT16_C( 21293),  INT16_C(  7273),
        -INT16_C( 27015), -INT16_C( 14742), -INT16_C(  2308), -INT16_C( 15389),  INT16_C(  4911),  INT16_C( 14825),  INT16_C( 11238), -INT16_C( 32197),
         INT16_C( 22623),  INT16_C( 19298), -INT16_C( 20629), -INT16_C( 27601),  INT16_C(  1478), -INT16_C( 17187), -INT16_C( 27953), -INT16_C( 28354) } },
    { { -INT16_C( 22398),  INT16_C( 16215), -INT16_C( 26165), -INT16_C(  5304), -INT16_C( 19990), -INT16_C( 31557),  INT16_C( 15611),  INT16_C( 17417),
         INT16_C( 22016),  INT16_C(  2286),  INT16_C(  7578),  INT16_C( 24988), -INT16_C(  6366), -INT16_C(  3636),  INT16_C(  2681), -INT16_C(   893),
        -INT16_C(  9550),  INT16_C( 32059), -INT16_C( 31628),  INT16_C( 24168),  INT16_C(  9269),  INT16_C( 12514), -INT16_C(  5024),  INT16_C( 24948),
         INT16_C( 25154), -INT16_C(  8855),  INT16_C(  1663), -INT16_C( 24258),  INT16_C(  2797),  INT16_C( 26259),  INT16_C(  5653), -INT16_C( 14494) },
      UINT32_C(1682284272),
      { -INT16_C( 23087),  INT16_C( 12935), -INT16_C(  1135), -INT16_C( 11373), -INT16_C(   930), -INT16_C(  8784), -INT16_C(  4606), -INT16_C(  4225),
         INT16_C(  4857),  INT16_C(  3670), -INT16_C( 18392),  INT16_C(  6357),  INT16_C(  6742),  INT16_C( 30845),  INT16_C( 16328), -INT16_C( 26161),
         INT16_C( 22244),  INT16_C( 30155),  INT16_C( 24146), -INT16_C( 20407), -INT16_C(  1701),  INT16_C( 23949),  INT16_C(  3304), -INT16_C(  7859),
        -INT16_C( 23778),  INT16_C( 18159), -INT16_C( 15269), -INT16_C( 19873), -INT16_C(  8993), -INT16_C( 22742), -INT16_C(  1509),  INT16_C(    64) },
      {  INT16_C(  3152), -INT16_C( 23947), -INT16_C( 16790), -INT16_C( 15022), -INT16_C(  8008), -INT16_C( 24541),  INT16_C( 28908),  INT16_C(  2945),
         INT16_C( 28691),  INT16_C( 28241), -INT16_C( 20428),  INT16_C(  4896),  INT16_C( 19340), -INT16_C( 22342), -INT16_C(  1211), -INT16_C( 27224),
         INT16_C(  7431),  INT16_C( 28984), -INT16_C( 29988), -INT16_C( 27593),  INT16_C( 23146),  INT16_C( 22324), -INT16_C( 18998), -INT16_C(  8862),
        -INT16_C( 19675),  INT16_C( 22859),  INT16_C( 27748), -INT16_C(  3987),  INT16_C( 10167), -INT16_C(   872),  INT16_C( 16418),  INT16_C( 10641) },
      { -INT16_C( 22398),  INT16_C( 16215), -INT16_C( 26165), -INT16_C(  5304), -INT16_C(  8008), -INT16_C( 24541), -INT16_C(  4606), -INT16_C(  4225),
         INT16_C( 22016),  INT16_C(  3670), -INT16_C( 20428),  INT16_C(  4896),  INT16_C(  6742), -INT16_C(  3636),  INT16_C(  2681), -INT16_C( 27224),
         INT16_C(  7431),  INT16_C( 32059), -INT16_C( 29988),  INT16_C( 24168),  INT16_C(  9269),  INT16_C( 12514), -INT16_C( 18998),  INT16_C( 24948),
         INT16_C( 25154), -INT16_C(  8855), -INT16_C( 15269), -INT16_C( 24258),  INT16_C(  2797), -INT16_C( 22742), -INT16_C(  1509), -INT16_C( 14494) } },
    { { -INT16_C( 13986),  INT16_C( 15003), -INT16_C( 11692), -INT16_C( 16690),  INT16_C(   556), -INT16_C(  2539),  INT16_C( 30647), -INT16_C(  9005),
         INT16_C(  7723), -INT16_C( 28875), -INT16_C( 23926),  INT16_C( 16767),  INT16_C(  6346), -INT16_C(  5059), -INT16_C( 12456), -INT16_C( 18922),
        -INT16_C( 20072), -INT16_C(  4880), -INT16_C( 16765), -INT16_C( 20565), -INT16_C( 16192),  INT16_C( 30629),  INT16_C( 30776),  INT16_C( 25427),
        -INT16_C( 30314),  INT16_C(  8690),  INT16_C( 28971), -INT16_C(  2718), -INT16_C( 24439), -INT16_C(  7454), -INT16_C(  1937),  INT16_C(  1944) },
      UINT32_C( 754223529),
      { -INT16_C( 32673), -INT16_C( 26753), -INT16_C( 11272), -INT16_C( 28934), -INT16_C(  5028), -INT16_C( 30801),  INT16_C(  4702), -INT16_C(  6275),
         INT16_C( 24498),  INT16_C(  8649),  INT16_C( 25175),  INT16_C(    40),  INT16_C(  7403),  INT16_C( 12844),  INT16_C(  1979),  INT16_C(  6970),
        -INT16_C( 17785),  INT16_C( 32690), -INT16_C( 21107), -INT16_C(  5875), -INT16_C( 16999), -INT16_C(  2192), -INT16_C(  4657), -INT16_C( 32289),
        -INT16_C( 22452), -INT16_C( 23646), -INT16_C( 13814), -INT16_C(  2653), -INT16_C( 12313), -INT16_C( 24024),  INT16_C( 25302),  INT16_C( 23997) },
      {  INT16_C( 28700), -INT16_C( 22052), -INT16_C(  5603), -INT16_C( 18798),  INT16_C(   935),  INT16_C( 30382), -INT16_C( 29200),  INT16_C( 15863),
        -INT16_C( 26315),  INT16_C( 16608), -INT16_C( 31645),  INT16_C( 18997),  INT16_C( 23891),  INT16_C( 10989), -INT16_C( 21824), -INT16_C(  9081),
         INT16_C( 25626),  INT16_C( 14214),  INT16_C(  6222), -INT16_C(  2578), -INT16_C( 25573),  INT16_C(  3179),  INT16_C( 25129),  INT16_C( 24137),
         INT16_C( 10747),  INT16_C( 24222), -INT16_C( 11091),  INT16_C(   425), -INT16_C( 27087), -INT16_C(  3797), -INT16_C( 19904),  INT16_C( 23502) },
      { -INT16_C( 32673),  INT16_C( 15003), -INT16_C( 11692), -INT16_C( 28934),  INT16_C(   556), -INT16_C( 30801),  INT16_C( 30647), -INT16_C(  6275),
        -INT16_C( 26315), -INT16_C( 28875), -INT16_C( 23926),  INT16_C(    40),  INT16_C(  6346), -INT16_C(  5059), -INT16_C( 12456), -INT16_C(  9081),
        -INT16_C( 20072), -INT16_C(  4880), -INT16_C( 21107), -INT16_C( 20565), -INT16_C( 25573), -INT16_C(  2192), -INT16_C(  4657), -INT16_C( 32289),
        -INT16_C( 30314),  INT16_C(  8690), -INT16_C( 13814), -INT16_C(  2653), -INT16_C( 24439), -INT16_C( 24024), -INT16_C(  1937),  INT16_C(  1944) } },
    { {  INT16_C( 21526),  INT16_C( 25746), -INT16_C( 32660), -INT16_C( 30631), -INT16_C( 15332),  INT16_C( 17812), -INT16_C(  8922),  INT16_C(  8612),
         INT16_C( 16902), -INT16_C( 19328),  INT16_C( 10518),  INT16_C( 18613), -INT16_C(  8001), -INT16_C(   199),  INT16_C(  1938), -INT16_C( 22182),
        -INT16_C(  4773), -INT16_C( 14323),  INT16_C( 26477), -INT16_C( 30128), -INT16_C(  7125),  INT16_C( 21199),  INT16_C( 29633), -INT16_C( 14477),
        -INT16_C(  3146), -INT16_C( 13189),  INT16_C( 12316), -INT16_C(  9452),  INT16_C( 19984), -INT16_C( 23589),  INT16_C( 13653), -INT16_C( 20148) },
      UINT32_C(2423871778),
      { -INT16_C(  5715),  INT16_C( 28222), -INT16_C( 20131),  INT16_C(  4917), -INT16_C( 20059), -INT16_C( 15905), -INT16_C(  2847), -INT16_C(  3427),
         INT16_C( 30786), -INT16_C( 26731), -INT16_C(  7763), -INT16_C( 12216), -INT16_C( 16070), -INT16_C(  1184),  INT16_C( 31370),  INT16_C( 14311),
         INT16_C(  9571), -INT16_C( 16219), -INT16_C(  9258),  INT16_C( 31699), -INT16_C( 19572),  INT16_C( 27965), -INT16_C(  9561), -INT16_C(  5793),
        -INT16_C(  2990), -INT16_C(   128), -INT16_C( 13867),  INT16_C(  4303),  INT16_C( 12170),  INT16_C(  5387), -INT16_C(  3415),  INT16_C(  3404) },
      { -INT16_C(  3561), -INT16_C(  4659), -INT16_C( 24115),  INT16_C( 22889), -INT16_C( 22956), -INT16_C(  1082),  INT16_C(  9856), -INT16_C( 11548),
         INT16_C( 25626), -INT16_C(  3887), -INT16_C( 24275), -INT16_C( 18432),  INT16_C(  3024),  INT16_C( 31437),  INT16_C(  6653),  INT16_C(  5255),
         INT16_C( 21515), -INT16_C( 10239),  INT16_C( 27381),  INT16_C( 18737), -INT16_C(  2032), -INT16_C( 28604),  INT16_C( 10270),  INT16_C( 14434),
         INT16_C( 13453), -INT16_C( 17880),  INT16_C( 10453), -INT16_C( 23182),  INT16_C( 16179),  INT16_C( 12319), -INT16_C( 22951),  INT16_C( 25668) },
      {  INT16_C( 21526), -INT16_C(  4659), -INT16_C( 32660), -INT16_C( 30631), -INT16_C( 15332), -INT16_C( 15905), -INT16_C(  8922),  INT16_C(  8612),
         INT16_C( 25626), -INT16_C( 19328),  INT16_C( 10518), -INT16_C( 18432), -INT16_C( 16070), -INT16_C(   199),  INT16_C(  6653), -INT16_C( 22182),
         INT16_C(  9571), -INT16_C( 14323),  INT16_C( 26477),  INT16_C( 18737), -INT16_C( 19572), -INT16_C( 28604), -INT16_C(  9561), -INT16_C( 14477),
        -INT16_C(  3146), -INT16_C( 13189),  INT16_C( 12316), -INT16_C(  9452),  INT16_C( 12170), -INT16_C( 23589),  INT16_C( 13653),  INT16_C(  3404) } },
    { {  INT16_C( 18171), -INT16_C(  4035),  INT16_C( 28336), -INT16_C( 16070),  INT16_C( 32358), -INT16_C( 31663), -INT16_C( 19289),  INT16_C( 13501),
        -INT16_C(  6680), -INT16_C( 16914),  INT16_C( 24846),  INT16_C( 16738), -INT16_C( 32096), -INT16_C(  1678), -INT16_C( 18904),  INT16_C(  9054),
        -INT16_C( 25604), -INT16_C( 21228),  INT16_C( 19977),  INT16_C( 28782), -INT16_C( 16436),  INT16_C( 29684), -INT16_C( 20109),  INT16_C( 23463),
        -INT16_C( 26985), -INT16_C( 23272),  INT16_C( 31735), -INT16_C( 26650),  INT16_C( 22781),  INT16_C(  9617), -INT16_C(  4337),  INT16_C(  2889) },
      UINT32_C(2478333322),
      { -INT16_C(  1818),  INT16_C( 23019), -INT16_C( 27991),  INT16_C( 16565), -INT16_C( 13016),  INT16_C(  8165), -INT16_C( 13240),  INT16_C( 17847),
         INT16_C( 18468),  INT16_C( 13163), -INT16_C( 19401), -INT16_C( 16065), -INT16_C(  2287), -INT16_C( 17324),  INT16_C( 22558),  INT16_C(  1075),
         INT16_C(  7760), -INT16_C(  1699),  INT16_C(  4785), -INT16_C(  9926),  INT16_C(  8160),  INT16_C( 10489), -INT16_C( 20245),  INT16_C(  4206),
        -INT16_C(  9736),  INT16_C( 12099), -INT16_C( 32115), -INT16_C( 24848),  INT16_C( 17530), -INT16_C( 26534), -INT16_C( 29284), -INT16_C(  4964) },
      { -INT16_C(  1620),  INT16_C( 24038),  INT16_C(  8204), -INT16_C(  5066),  INT16_C( 12095),  INT16_C( 11028), -INT16_C( 32033), -INT16_C( 10437),
         INT16_C( 32347), -INT16_C(  6138), -INT16_C(  2559),  INT16_C( 31622), -INT16_C(  8133), -INT16_C( 10477), -INT16_C( 20626),  INT16_C(  6852),
        -INT16_C( 21848), -INT16_C( 19337), -INT16_C( 21046),  INT16_C(  2464), -INT16_C( 18979), -INT16_C( 17356),  INT16_C( 28471), -INT16_C( 27756),
        -INT16_C( 25874), -INT16_C(  4229),  INT16_C(   657), -INT16_C( 13206),  INT16_C( 32226),  INT16_C( 20643),  INT16_C( 26412), -INT16_C( 11158) },
      {  INT16_C( 18171),  INT16_C( 23019),  INT16_C( 28336), -INT16_C(  5066),  INT16_C( 32358), -INT16_C( 31663), -INT16_C( 19289), -INT16_C( 10437),
         INT16_C( 18468), -INT16_C( 16914), -INT16_C( 19401), -INT16_C( 16065), -INT16_C(  8133), -INT16_C(  1678), -INT16_C( 20626),  INT16_C(  9054),
        -INT16_C( 25604), -INT16_C( 21228),  INT16_C( 19977), -INT16_C(  9926), -INT16_C( 18979), -INT16_C( 17356), -INT16_C( 20109), -INT16_C( 27756),
        -INT16_C( 25874), -INT16_C(  4229),  INT16_C( 31735), -INT16_C( 26650),  INT16_C( 17530),  INT16_C(  9617), -INT16_C(  4337), -INT16_C( 11158) } },
    { { -INT16_C(  7919), -INT16_C(  9335),  INT16_C( 10639),  INT16_C( 27877),  INT16_C(  6622),  INT16_C(  5672), -INT16_C( 17271),  INT16_C( 30633),
         INT16_C(  9303), -INT16_C(  6042), -INT16_C( 12250),  INT16_C(  2484),  INT16_C( 22349),  INT16_C( 31065), -INT16_C( 15169), -INT16_C( 12211),
        -INT16_C( 10587),  INT16_C( 13484), -INT16_C( 28416), -INT16_C(  8544), -INT16_C( 13910),  INT16_C( 13300), -INT16_C( 25211), -INT16_C(  9046),
         INT16_C(  4290), -INT16_C(  5948),  INT16_C( 30944),  INT16_C( 11761),  INT16_C( 19408), -INT16_C( 28762), -INT16_C(  3057), -INT16_C( 19361) },
      UINT32_C(3404270538),
      { -INT16_C( 25262), -INT16_C( 10118),  INT16_C(  9531), -INT16_C(   588),  INT16_C( 31029),  INT16_C(  5861), -INT16_C( 10255), -INT16_C( 16061),
        -INT16_C(  5598),  INT16_C( 12624), -INT16_C( 20258), -INT16_C( 22299), -INT16_C( 12613),  INT16_C( 22643),  INT16_C(  7256), -INT16_C( 21857),
         INT16_C(  6585), -INT16_C(  2942),  INT16_C( 14142),  INT16_C( 29937), -INT16_C( 10320), -INT16_C( 24182), -INT16_C( 12882), -INT16_C( 12189),
        -INT16_C( 19529), -INT16_C( 27391), -INT16_C(  6557),  INT16_C(  7998), -INT16_C( 20043),  INT16_C(  3447),  INT16_C(  5837), -INT16_C( 31049) },
      {  INT16_C( 14895),  INT16_C( 28283),  INT16_C( 27761),  INT16_C(  8674),  INT16_C( 27715), -INT16_C(  3646),  INT16_C(  9529), -INT16_C(  3647),
        -INT16_C( 15655),  INT16_C( 15494), -INT16_C( 15191),  INT16_C( 24155), -INT16_C( 11659),  INT16_C( 17003),  INT16_C(  8936),  INT16_C(  6345),
         INT16_C( 17500), -INT16_C( 12922),  INT16_C( 26800), -INT16_C(  2834), -INT16_C( 20012),  INT16_C(  3557), -INT16_C( 22570), -INT16_C( 20482),
        -INT16_C( 31383),  INT16_C(  4844),  INT16_C( 18249), -INT16_C( 16528), -INT16_C(  9446),  INT16_C(   513), -INT16_C( 13570),  INT16_C( 23066) },
      { -INT16_C(  7919), -INT16_C( 10118),  INT16_C( 10639), -INT16_C(   588),  INT16_C(  6622),  INT16_C(  5672), -INT16_C( 10255), -INT16_C( 16061),
        -INT16_C( 15655),  INT16_C( 12624), -INT16_C( 12250), -INT16_C( 22299),  INT16_C( 22349),  INT16_C( 31065), -INT16_C( 15169), -INT16_C( 12211),
         INT16_C(  6585),  INT16_C( 13484), -INT16_C( 28416), -INT16_C(  2834), -INT16_C( 13910), -INT16_C( 24182), -INT16_C( 22570), -INT16_C( 20482),
         INT16_C(  4290), -INT16_C( 27391),  INT16_C( 30944), -INT16_C( 16528),  INT16_C( 19408), -INT16_C( 28762), -INT16_C( 13570), -INT16_C( 31049) } },
    { { -INT16_C( 24562), -INT16_C( 16600),  INT16_C(  5640), -INT16_C(  9037), -INT16_C( 26425), -INT16_C( 24854), -INT16_C(  6081), -INT16_C( 22195),
         INT16_C( 14701), -INT16_C( 18501),  INT16_C( 11393), -INT16_C( 25738),  INT16_C( 30471),  INT16_C(  1437), -INT16_C( 18366),  INT16_C( 20576),
        -INT16_C( 30632),  INT16_C( 24847), -INT16_C( 15714),  INT16_C( 26173),  INT16_C( 10075), -INT16_C( 26108),  INT16_C( 20752),  INT16_C( 32067),
        -INT16_C(   117),  INT16_C(  3124), -INT16_C( 21973),  INT16_C( 12967),  INT16_C( 17442),  INT16_C( 25656), -INT16_C( 26372),  INT16_C( 21940) },
      UINT32_C(3199648800),
      {  INT16_C( 10267),  INT16_C( 11132), -INT16_C( 16518),  INT16_C(  1448), -INT16_C(  8770), -INT16_C(  5871), -INT16_C( 18297), -INT16_C( 22244),
         INT16_C( 21756), -INT16_C(  1779), -INT16_C( 15636),  INT16_C(  3150),  INT16_C(  1158),  INT16_C(  3274), -INT16_C(  4105),  INT16_C(  4846),
         INT16_C( 27159), -INT16_C( 28355), -INT16_C(  6615), -INT16_C(  5994), -INT16_C( 22589),  INT16_C( 19153), -INT16_C(  4769),  INT16_C( 23796),
         INT16_C(   321),  INT16_C( 11605), -INT16_C( 23613),  INT16_C( 18745),  INT16_C(  1191), -INT16_C( 25002),  INT16_C( 17651),  INT16_C(  2737) },
      { -INT16_C(  4434), -INT16_C( 10340),  INT16_C( 13012), -INT16_C( 26689), -INT16_C( 28198),  INT16_C( 14818), -INT16_C( 10626), -INT16_C( 16235),
        -INT16_C(  5417), -INT16_C( 25619),  INT16_C( 10125),  INT16_C( 13540),  INT16_C( 14891),  INT16_C(  7891), -INT16_C( 31618),  INT16_C( 11304),
        -INT16_C( 15246),  INT16_C( 18180), -INT16_C( 15369), -INT16_C( 11810), -INT16_C( 16300), -INT16_C( 11510), -INT16_C( 24426),  INT16_C( 28307),
        -INT16_C( 32630),  INT16_C(  6153), -INT16_C(  4697), -INT16_C( 11700),  INT16_C(  7976), -INT16_C( 22800),  INT16_C(  6563),  INT16_C(  5843) },
      { -INT16_C( 24562), -INT16_C( 16600),  INT16_C(  5640), -INT16_C(  9037), -INT16_C( 26425), -INT16_C(  5871), -INT16_C(  6081), -INT16_C( 22195),
         INT16_C( 14701), -INT16_C( 18501), -INT16_C( 15636), -INT16_C( 25738),  INT16_C( 30471),  INT16_C(  1437), -INT16_C( 31618),  INT16_C(  4846),
        -INT16_C( 30632), -INT16_C( 28355), -INT16_C( 15369),  INT16_C( 26173), -INT16_C( 22589), -INT16_C( 11510),  INT16_C( 20752),  INT16_C( 23796),
        -INT16_C(   117),  INT16_C(  6153), -INT16_C( 23613), -INT16_C( 11700),  INT16_C(  1191), -INT16_C( 25002), -INT16_C( 26372),  INT16_C(  2737) } },
    { { -INT16_C( 10275), -INT16_C( 11171),  INT16_C( 15258), -INT16_C(  4187), -INT16_C( 20228), -INT16_C( 27966),  INT16_C( 21840), -INT16_C(  9728),
         INT16_C(  2517),  INT16_C( 32242),  INT16_C( 16375),  INT16_C(  8015),  INT16_C( 16478),  INT16_C(   709), -INT16_C( 26535),  INT16_C( 13848),
         INT16_C( 30063),  INT16_C(  2571), -INT16_C( 20304), -INT16_C( 21255), -INT16_C( 17568), -INT16_C( 20417),  INT16_C( 16144), -INT16_C(  6773),
         INT16_C( 32073),  INT16_C( 16482), -INT16_C( 19780),  INT16_C(  7007),  INT16_C(  9458),  INT16_C( 19229),  INT16_C( 13757),  INT16_C( 11393) },
      UINT32_C(1513524394),
      {  INT16_C( 18154), -INT16_C(  1458), -INT16_C(  9851), -INT16_C( 12576),  INT16_C( 16982),  INT16_C(  4878),  INT16_C( 28148), -INT16_C(  6610),
         INT16_C( 19346),  INT16_C( 20273), -INT16_C( 19584),  INT16_C( 10875), -INT16_C( 19905),  INT16_C( 31876), -INT16_C( 29727), -INT16_C( 13286),
         INT16_C( 26833),  INT16_C( 22470), -INT16_C( 22975), -INT16_C( 26843),  INT16_C( 13545), -INT16_C(  8790), -INT16_C( 10079),  INT16_C( 13252),
        -INT16_C(  2781), -INT16_C( 23678), -INT16_C(   344), -INT16_C(  5939),  INT16_C( 21168), -INT16_C( 28316),  INT16_C( 32477), -INT16_C( 20643) },
      {  INT16_C(  9446),  INT16_C(  9990),  INT16_C( 11210), -INT16_C( 19521),  INT16_C( 26975),  INT16_C(   401),  INT16_C( 21826),  INT16_C( 25908),
        -INT16_C( 18614), -INT16_C(  3319), -INT16_C( 10571),  INT16_C( 26075),  INT16_C( 16168),  INT16_C(  1782),  INT16_C( 21694), -INT16_C( 23371),
        -INT16_C( 17544),  INT16_C( 17100), -INT16_C( 29722),  INT16_C( 18166), -INT16_C( 30732),  INT16_C( 13895),  INT16_C( 31708),  INT16_C(  9884),
        -INT16_C( 23246), -INT16_C(  6375), -INT16_C(  2949), -INT16_C( 23476),  INT16_C( 17204), -INT16_C(  3414),  INT16_C( 24471),  INT16_C(  3990) },
      { -INT16_C( 10275), -INT16_C(  1458),  INT16_C( 15258), -INT16_C( 19521), -INT16_C( 20228),  INT16_C(   401),  INT16_C( 21840), -INT16_C(  6610),
         INT16_C(  2517),  INT16_C( 32242), -INT16_C( 19584),  INT16_C( 10875),  INT16_C( 16478),  INT16_C(   709), -INT16_C( 26535), -INT16_C( 23371),
         INT16_C( 30063),  INT16_C( 17100), -INT16_C( 29722), -INT16_C( 21255), -INT16_C( 30732), -INT16_C(  8790),  INT16_C( 16144), -INT16_C(  6773),
         INT16_C( 32073), -INT16_C( 23678), -INT16_C( 19780), -INT16_C( 23476),  INT16_C( 17204),  INT16_C( 19229),  INT16_C( 24471),  INT16_C( 11393) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_epi16(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { UINT32_C(2426184294),
      {  INT16_C( 31879),  INT16_C( 10074),  INT16_C(  3945),  INT16_C( 30364),  INT16_C(  4967), -INT16_C( 14898), -INT16_C( 23290), -INT16_C( 27628),
        -INT16_C( 28502),  INT16_C( 18211),  INT16_C( 31392),  INT16_C(  1793), -INT16_C( 25315), -INT16_C(  2409), -INT16_C( 26870), -INT16_C( 28355),
        -INT16_C( 26861),  INT16_C( 31928),  INT16_C( 21670),  INT16_C(  3570), -INT16_C( 16281),  INT16_C( 28114), -INT16_C(  6298),  INT16_C(  4098),
         INT16_C(  9591),  INT16_C(  6231),  INT16_C( 22943), -INT16_C( 17377), -INT16_C( 18698),  INT16_C(   178), -INT16_C(  4275),  INT16_C( 24721) },
      {  INT16_C( 18823),  INT16_C( 11740), -INT16_C( 12387),  INT16_C(  1083),  INT16_C(  3471), -INT16_C(  2702),  INT16_C( 29940),  INT16_C( 27653),
         INT16_C( 23961),  INT16_C( 14468), -INT16_C( 23626), -INT16_C( 21259), -INT16_C( 22695), -INT16_C( 22611),  INT16_C( 16023),  INT16_C(  7687),
        -INT16_C(  7032),  INT16_C(  9547), -INT16_C( 31053),  INT16_C( 16938), -INT16_C( 25452), -INT16_C( 30664),  INT16_C( 15632), -INT16_C( 22028),
         INT16_C( 30874),  INT16_C( 20705), -INT16_C( 10725),  INT16_C( 30205), -INT16_C( 21890),  INT16_C(  5404),  INT16_C(  9192),  INT16_C( 28723) },
      {  INT16_C(     0),  INT16_C( 10074), -INT16_C( 12387),  INT16_C(     0),  INT16_C(     0), -INT16_C( 14898), -INT16_C( 23290),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 14468),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 22611),  INT16_C(     0), -INT16_C( 28355),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 31053),  INT16_C(  3570), -INT16_C( 25452),  INT16_C(     0),  INT16_C(     0), -INT16_C( 22028),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 21890),  INT16_C(     0),  INT16_C(     0),  INT16_C( 24721) } },
    { UINT32_C(3130424839),
      {  INT16_C( 13660),  INT16_C( 27681),  INT16_C(  5746),  INT16_C(  3349), -INT16_C(  2418), -INT16_C( 21923),  INT16_C( 23245),  INT16_C( 19231),
         INT16_C( 15108), -INT16_C(  4768), -INT16_C( 27810),  INT16_C( 26205), -INT16_C(  3311),  INT16_C(  5664),  INT16_C(  7603),  INT16_C(  4015),
        -INT16_C( 11950), -INT16_C( 14981), -INT16_C( 28441),  INT16_C( 30162),  INT16_C( 12167),  INT16_C( 21535),  INT16_C( 16010), -INT16_C( 29025),
        -INT16_C(   135), -INT16_C( 10117), -INT16_C(  9838), -INT16_C( 23746),  INT16_C( 24268), -INT16_C( 32582),  INT16_C( 27004), -INT16_C( 12657) },
      {  INT16_C(  2874),  INT16_C(  8595),  INT16_C( 26011),  INT16_C(  8855), -INT16_C( 18795),  INT16_C(  8054),  INT16_C(  5621),  INT16_C( 28333),
         INT16_C( 10516), -INT16_C( 22970), -INT16_C( 31742), -INT16_C( 12726),  INT16_C(  1251),  INT16_C( 24398), -INT16_C(  8595), -INT16_C( 22483),
        -INT16_C( 15895), -INT16_C( 31543),  INT16_C( 24614), -INT16_C( 17497),  INT16_C(  7447),  INT16_C(  3290), -INT16_C( 30669),  INT16_C( 18298),
        -INT16_C( 15951), -INT16_C( 19474),  INT16_C( 14405),  INT16_C( 10369), -INT16_C( 12228), -INT16_C( 22137), -INT16_C( 19026), -INT16_C( 26799) },
      {  INT16_C(  2874),  INT16_C(  8595),  INT16_C(  5746),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 22970), -INT16_C( 31742), -INT16_C( 12726), -INT16_C(  3311),  INT16_C(  5664), -INT16_C(  8595),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 31543), -INT16_C( 28441),  INT16_C(     0),  INT16_C(  7447),  INT16_C(     0),  INT16_C(     0), -INT16_C( 29025),
         INT16_C(     0), -INT16_C( 19474),  INT16_C(     0), -INT16_C( 23746), -INT16_C( 12228), -INT16_C( 32582),  INT16_C(     0), -INT16_C( 26799) } },
    { UINT32_C(2619022198),
      {  INT16_C( 13024),  INT16_C(  5022),  INT16_C(  6586),  INT16_C( 27482),  INT16_C( 18650),  INT16_C(  7966), -INT16_C( 24448), -INT16_C( 17336),
        -INT16_C( 12432),  INT16_C(  7782), -INT16_C( 18556), -INT16_C(  1355), -INT16_C( 12078),  INT16_C( 20119), -INT16_C(  4205),  INT16_C( 29664),
         INT16_C( 32545), -INT16_C(  9082), -INT16_C(  8040),  INT16_C( 29255),  INT16_C( 26153), -INT16_C( 22127), -INT16_C(  9978),  INT16_C( 30310),
        -INT16_C( 13143),  INT16_C( 11668),  INT16_C( 18819),  INT16_C( 22056), -INT16_C( 16615), -INT16_C( 21340), -INT16_C( 31570), -INT16_C( 12513) },
      { -INT16_C( 23293), -INT16_C( 25685), -INT16_C(  3194), -INT16_C( 20723), -INT16_C( 24743),  INT16_C( 24408), -INT16_C( 16776),  INT16_C(  8661),
         INT16_C( 27018),  INT16_C(  3663),  INT16_C( 30642), -INT16_C( 13468),  INT16_C(  2102), -INT16_C(  7048), -INT16_C( 26740), -INT16_C( 28493),
         INT16_C( 24381), -INT16_C( 15573),  INT16_C( 14674), -INT16_C( 21646), -INT16_C( 13608),  INT16_C( 20490), -INT16_C(  8311),  INT16_C(  4978),
        -INT16_C( 16056), -INT16_C(  1503), -INT16_C( 31432),  INT16_C( 28357),  INT16_C( 15757),  INT16_C(  6738),  INT16_C(  1493),  INT16_C(  4778) },
      {  INT16_C(     0), -INT16_C( 25685), -INT16_C(  3194),  INT16_C(     0), -INT16_C( 24743),  INT16_C(  7966), -INT16_C( 24448),  INT16_C(     0),
        -INT16_C( 12432),  INT16_C(  3663),  INT16_C(     0), -INT16_C( 13468), -INT16_C( 12078),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 24381), -INT16_C( 15573),  INT16_C(     0), -INT16_C( 21646), -INT16_C( 13608),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 31432),  INT16_C( 22056), -INT16_C( 16615),  INT16_C(     0),  INT16_C(     0), -INT16_C( 12513) } },
    { UINT32_C(3067467108),
      {  INT16_C( 27409), -INT16_C( 26057), -INT16_C( 22198), -INT16_C( 27986), -INT16_C( 12438), -INT16_C( 23924),  INT16_C( 21077), -INT16_C(  7664),
         INT16_C( 25231),  INT16_C( 25852), -INT16_C( 22937), -INT16_C( 13194),  INT16_C( 19324), -INT16_C( 30078), -INT16_C(  7022), -INT16_C( 23439),
        -INT16_C( 22449), -INT16_C( 26050), -INT16_C(  5039), -INT16_C( 17620), -INT16_C( 17988),  INT16_C(  4445),  INT16_C( 27915), -INT16_C( 25869),
        -INT16_C(  3889),  INT16_C( 14079),  INT16_C( 30102),  INT16_C(  4610), -INT16_C( 31295),  INT16_C( 21405),  INT16_C(  3689), -INT16_C( 18185) },
      {  INT16_C( 14006),  INT16_C(  1874),  INT16_C( 32546), -INT16_C(  8510),  INT16_C(  7992),  INT16_C( 17391), -INT16_C(  7284),  INT16_C( 23517),
        -INT16_C(  9005),  INT16_C( 27025), -INT16_C( 27566),  INT16_C(  4988),  INT16_C(  6425), -INT16_C( 32154),  INT16_C( 24103), -INT16_C(  8902),
        -INT16_C( 29292), -INT16_C( 18716), -INT16_C( 23028),  INT16_C( 17557), -INT16_C( 31547),  INT16_C( 20871),  INT16_C( 25703),  INT16_C( 15020),
         INT16_C( 15681), -INT16_C( 27740),  INT16_C(  8401), -INT16_C(  5466),  INT16_C(  3129),  INT16_C( 24684), -INT16_C( 22678), -INT16_C(   451) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 22198),  INT16_C(     0),  INT16_C(     0), -INT16_C( 23924), -INT16_C(  7284),  INT16_C(     0),
        -INT16_C(  9005),  INT16_C(     0), -INT16_C( 27566),  INT16_C(     0),  INT16_C(  6425),  INT16_C(     0), -INT16_C(  7022), -INT16_C( 23439),
        -INT16_C( 29292),  INT16_C(     0), -INT16_C( 23028),  INT16_C(     0), -INT16_C( 31547),  INT16_C(     0),  INT16_C( 25703), -INT16_C( 25869),
         INT16_C(     0), -INT16_C( 27740),  INT16_C(  8401),  INT16_C(     0), -INT16_C( 31295),  INT16_C( 21405),  INT16_C(     0), -INT16_C( 18185) } },
    { UINT32_C(1085612340),
      {  INT16_C(  3022),  INT16_C( 14045), -INT16_C( 30353), -INT16_C( 20368),  INT16_C(  5318), -INT16_C( 26557), -INT16_C(  5836),  INT16_C( 28034),
        -INT16_C(  4106),  INT16_C( 24781),  INT16_C(  2710), -INT16_C( 13729),  INT16_C(  5163), -INT16_C(  3574), -INT16_C( 29090),  INT16_C( 11390),
         INT16_C( 23449),  INT16_C(  2146), -INT16_C( 11292), -INT16_C( 21575), -INT16_C(   793),  INT16_C(  7235), -INT16_C( 14874), -INT16_C(  9079),
         INT16_C( 22452),  INT16_C( 19004), -INT16_C( 25759), -INT16_C( 29420),  INT16_C(  7855),  INT16_C(  3455), -INT16_C(   340),  INT16_C( 17722) },
      { -INT16_C( 25511),  INT16_C( 15950),  INT16_C(  1903),  INT16_C( 22505),  INT16_C( 11267), -INT16_C(  5773), -INT16_C(   783), -INT16_C( 22843),
         INT16_C(   595), -INT16_C( 18960),  INT16_C(  1437),  INT16_C( 19778), -INT16_C( 16093), -INT16_C( 12198), -INT16_C( 27457),  INT16_C(  6421),
         INT16_C( 25393), -INT16_C( 24489),  INT16_C( 16490),  INT16_C( 28407),  INT16_C( 27244),  INT16_C( 23895),  INT16_C(  7527), -INT16_C( 17917),
        -INT16_C(  3041), -INT16_C( 17297), -INT16_C( 19975),  INT16_C(  7177),  INT16_C( 25715),  INT16_C( 13036),  INT16_C(   760),  INT16_C( 10571) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 30353),  INT16_C(     0),  INT16_C(  5318), -INT16_C( 26557),  INT16_C(     0),  INT16_C(     0),
        -INT16_C(  4106),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 12198),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 23449),  INT16_C(     0), -INT16_C( 11292),  INT16_C(     0), -INT16_C(   793),  INT16_C(  7235),  INT16_C(     0), -INT16_C( 17917),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(   340),  INT16_C(     0) } },
    { UINT32_C(3502940773),
      { -INT16_C( 27348), -INT16_C( 27732), -INT16_C( 20558), -INT16_C( 11955), -INT16_C( 16989), -INT16_C( 25458), -INT16_C( 26770), -INT16_C(  7751),
        -INT16_C( 23045), -INT16_C(  3052),  INT16_C( 24487),  INT16_C(  3357), -INT16_C(  6398), -INT16_C(  6947),  INT16_C(  7081), -INT16_C( 10957),
        -INT16_C(  8272),  INT16_C( 25448), -INT16_C( 19058),  INT16_C( 12852), -INT16_C( 15758), -INT16_C(  7730), -INT16_C( 30886),  INT16_C( 21954),
        -INT16_C( 10707), -INT16_C( 11191),  INT16_C( 26422),  INT16_C( 14561), -INT16_C( 16818), -INT16_C(  2276),  INT16_C( 20441), -INT16_C( 30004) },
      {  INT16_C( 13358), -INT16_C( 16915),  INT16_C(  8682),  INT16_C( 23791), -INT16_C( 16924),  INT16_C( 15933),  INT16_C(    69),  INT16_C( 29331),
        -INT16_C(  8746),  INT16_C(  3142),  INT16_C( 10308), -INT16_C( 28092),  INT16_C( 25062), -INT16_C( 16246),  INT16_C( 22192), -INT16_C(  8374),
         INT16_C( 14219),  INT16_C( 30108), -INT16_C( 29864),  INT16_C( 15569),  INT16_C(  3912), -INT16_C( 29318),  INT16_C(  3599), -INT16_C(  6657),
         INT16_C( 18155),  INT16_C( 12274),  INT16_C( 13934),  INT16_C( 21697),  INT16_C( 19351),  INT16_C( 18452),  INT16_C( 24226),  INT16_C( 11559) },
      { -INT16_C( 27348),  INT16_C(     0), -INT16_C( 20558),  INT16_C(     0),  INT16_C(     0), -INT16_C( 25458), -INT16_C( 26770),  INT16_C(     0),
         INT16_C(     0), -INT16_C(  3052),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 16246),  INT16_C(     0), -INT16_C( 10957),
         INT16_C(     0),  INT16_C( 25448),  INT16_C(     0),  INT16_C( 12852),  INT16_C(     0),  INT16_C(     0), -INT16_C( 30886), -INT16_C(  6657),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 16818),  INT16_C(     0),  INT16_C( 20441), -INT16_C( 30004) } },
    { UINT32_C(4003644309),
      { -INT16_C( 23166), -INT16_C( 28380),  INT16_C(  9139), -INT16_C( 24969),  INT16_C( 26985), -INT16_C( 10291), -INT16_C( 29025),  INT16_C( 14124),
         INT16_C( 16602),  INT16_C( 31871), -INT16_C( 22881),  INT16_C( 13481),  INT16_C( 19305), -INT16_C( 18654),  INT16_C( 19902),  INT16_C( 16717),
         INT16_C( 29170), -INT16_C( 23086),  INT16_C( 18837), -INT16_C(   445),  INT16_C(  4274),  INT16_C( 21206),  INT16_C(   670),  INT16_C( 30857),
         INT16_C(  2114), -INT16_C(  7692), -INT16_C( 25170),  INT16_C(  5910),  INT16_C( 14568), -INT16_C( 22578),  INT16_C(  7045),  INT16_C( 30696) },
      { -INT16_C( 17779),  INT16_C(  8732),  INT16_C( 24324), -INT16_C( 18912), -INT16_C(  2449),  INT16_C(  3592), -INT16_C( 28168),  INT16_C( 15238),
         INT16_C( 31641),  INT16_C( 18204),  INT16_C( 12824),  INT16_C(   350),  INT16_C( 11371), -INT16_C(  3928), -INT16_C( 28600), -INT16_C( 10904),
        -INT16_C( 31670),  INT16_C( 20215),  INT16_C(  6116),  INT16_C( 21253),  INT16_C(  3342),  INT16_C(  1633), -INT16_C(  5985),  INT16_C( 14401),
         INT16_C( 24163),  INT16_C( 31616), -INT16_C(  8560), -INT16_C(  1156),  INT16_C(  9227),  INT16_C( 21484),  INT16_C( 21684), -INT16_C(   216) },
      { -INT16_C( 23166),  INT16_C(     0),  INT16_C(  9139),  INT16_C(     0), -INT16_C(  2449),  INT16_C(     0),  INT16_C(     0),  INT16_C( 14124),
         INT16_C( 16602),  INT16_C( 18204),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28600), -INT16_C( 10904),
         INT16_C(     0), -INT16_C( 23086),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1633),  INT16_C(     0),  INT16_C( 14401),
         INT16_C(     0), -INT16_C(  7692), -INT16_C( 25170), -INT16_C(  1156),  INT16_C(     0), -INT16_C( 22578),  INT16_C(  7045), -INT16_C(   216) } },
    { UINT32_C(3159171032),
      {  INT16_C( 29024), -INT16_C(   181), -INT16_C( 29607), -INT16_C( 17353), -INT16_C( 18454),  INT16_C( 31544), -INT16_C( 19306), -INT16_C( 24202),
         INT16_C( 25305), -INT16_C( 29196),  INT16_C(  7350), -INT16_C( 28788), -INT16_C(  9669),  INT16_C( 29003),  INT16_C( 23340), -INT16_C( 29514),
         INT16_C(   461),  INT16_C(  9867), -INT16_C( 15475),  INT16_C( 30947),  INT16_C(  7034),  INT16_C(  4339),  INT16_C( 27087), -INT16_C( 22351),
        -INT16_C( 23092), -INT16_C( 32202), -INT16_C( 15679), -INT16_C(  1007),  INT16_C( 23964), -INT16_C( 13970),  INT16_C(  9400), -INT16_C( 31403) },
      { -INT16_C(  7899), -INT16_C( 19796), -INT16_C( 28764),  INT16_C(  7722),  INT16_C(  7594),  INT16_C( 31023), -INT16_C(  8057),  INT16_C( 21282),
         INT16_C( 22662),  INT16_C( 18389), -INT16_C(  6374), -INT16_C( 18620), -INT16_C( 19900), -INT16_C(   896), -INT16_C( 10794), -INT16_C(  1150),
         INT16_C( 11958),  INT16_C( 23213), -INT16_C( 10051),  INT16_C( 26489), -INT16_C( 22283),  INT16_C( 31968),  INT16_C(   648),  INT16_C(  3791),
        -INT16_C( 23206),  INT16_C( 30038), -INT16_C( 25972), -INT16_C( 12244), -INT16_C( 21428),  INT16_C(  8908),  INT16_C( 20097),  INT16_C( 14365) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 17353), -INT16_C( 18454),  INT16_C(     0), -INT16_C( 19306), -INT16_C( 24202),
         INT16_C( 22662), -INT16_C( 29196), -INT16_C(  6374), -INT16_C( 28788), -INT16_C( 19900),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(   461),  INT16_C(     0), -INT16_C( 15475),  INT16_C( 26489),  INT16_C(     0),  INT16_C(     0),  INT16_C(   648),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 25972), -INT16_C( 12244), -INT16_C( 21428), -INT16_C( 13970),  INT16_C(     0), -INT16_C( 31403) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_epi16(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_min_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint16_t a[32];
    const uint16_t b[32];
    const uint16_t r[32];
  } test_vec[] = {
    { { UINT16_C( 1431), UINT16_C(60088), UINT16_C(59670), UINT16_C(46197), UINT16_C(28207), UINT16_C(42628), UINT16_C(38831), UINT16_C( 7804),
        UINT16_C(27384), UINT16_C(63519), UINT16_C(45488), UINT16_C(32855), UINT16_C(60315), UINT16_C(36942), UINT16_C(33367), UINT16_C(61079),
        UINT16_C(20616), UINT16_C(40664), UINT16_C(19769), UINT16_C(26706), UINT16_C(54972), UINT16_C(27406), UINT16_C(35693), UINT16_C(25993),
        UINT16_C(43253), UINT16_C(42333), UINT16_C(46425), UINT16_C(62502), UINT16_C(29856), UINT16_C(63365), UINT16_C( 7415), UINT16_C(32741) },
      { UINT16_C(48748), UINT16_C(42525), UINT16_C(28427), UINT16_C(50958), UINT16_C( 7494), UINT16_C(45875), UINT16_C(48296), UINT16_C(40217),
        UINT16_C(30309), UINT16_C(48706), UINT16_C(26667), UINT16_C(52403), UINT16_C(14557), UINT16_C(54467), UINT16_C(43348), UINT16_C(49491),
        UINT16_C(28775), UINT16_C(29287), UINT16_C(30175), UINT16_C( 9530), UINT16_C(28050), UINT16_C(15065), UINT16_C(61993), UINT16_C(36567),
        UINT16_C( 6760), UINT16_C(37965), UINT16_C(  130), UINT16_C(24416), UINT16_C( 9016), UINT16_C(35891), UINT16_C(34508), UINT16_C(13133) },
      { UINT16_C( 1431), UINT16_C(42525), UINT16_C(28427), UINT16_C(46197), UINT16_C( 7494), UINT16_C(42628), UINT16_C(38831), UINT16_C( 7804),
        UINT16_C(27384), UINT16_C(48706), UINT16_C(26667), UINT16_C(32855), UINT16_C(14557), UINT16_C(36942), UINT16_C(33367), UINT16_C(49491),
        UINT16_C(20616), UINT16_C(29287), UINT16_C(19769), UINT16_C( 9530), UINT16_C(28050), UINT16_C(15065), UINT16_C(35693), UINT16_C(25993),
        UINT16_C( 6760), UINT16_C(37965), UINT16_C(  130), UINT16_C(24416), UINT16_C( 9016), UINT16_C(35891), UINT16_C( 7415), UINT16_C(13133) } },
    { { UINT16_C(46326), UINT16_C(54950), UINT16_C(57386), UINT16_C(48379), UINT16_C(54349), UINT16_C(30455), UINT16_C(52934), UINT16_C(12037),
        UINT16_C(21224), UINT16_C(27587), UINT16_C( 9042), UINT16_C(35530), UINT16_C(65094), UINT16_C( 4886), UINT16_C(25732), UINT16_C(31558),
        UINT16_C(60440), UINT16_C(16977), UINT16_C(19660), UINT16_C( 6655), UINT16_C(63009), UINT16_C(59280), UINT16_C(38340), UINT16_C(44310),
        UINT16_C(55783), UINT16_C(14616), UINT16_C(58108), UINT16_C(17347), UINT16_C(55776), UINT16_C(25942), UINT16_C(39997), UINT16_C(22240) },
      { UINT16_C(12681), UINT16_C(21912), UINT16_C(38781), UINT16_C(40559), UINT16_C(65421), UINT16_C(21126), UINT16_C(40084), UINT16_C(31743),
        UINT16_C( 6006), UINT16_C(29364), UINT16_C(30713), UINT16_C(55989), UINT16_C( 2896), UINT16_C(36415), UINT16_C( 8104), UINT16_C(12772),
        UINT16_C(31824), UINT16_C(52614), UINT16_C(62740), UINT16_C(41324), UINT16_C(62196), UINT16_C(35059), UINT16_C(62094), UINT16_C( 1027),
        UINT16_C(46857), UINT16_C(  887), UINT16_C(11310), UINT16_C(32733), UINT16_C( 7224), UINT16_C(57357), UINT16_C(61755), UINT16_C(35601) },
      { UINT16_C(12681), UINT16_C(21912), UINT16_C(38781), UINT16_C(40559), UINT16_C(54349), UINT16_C(21126), UINT16_C(40084), UINT16_C(12037),
        UINT16_C( 6006), UINT16_C(27587), UINT16_C( 9042), UINT16_C(35530), UINT16_C( 2896), UINT16_C( 4886), UINT16_C( 8104), UINT16_C(12772),
        UINT16_C(31824), UINT16_C(16977), UINT16_C(19660), UINT16_C( 6655), UINT16_C(62196), UINT16_C(35059), UINT16_C(38340), UINT16_C( 1027),
        UINT16_C(46857), UINT16_C(  887), UINT16_C(11310), UINT16_C(17347), UINT16_C( 7224), UINT16_C(25942), UINT16_C(39997), UINT16_C(22240) } },
    { { UINT16_C(38765), UINT16_C(33112), UINT16_C(50317), UINT16_C(33059), UINT16_C( 5814), UINT16_C(17674), UINT16_C( 3337), UINT16_C( 4681),
        UINT16_C(49349), UINT16_C(62229), UINT16_C(62189), UINT16_C( 9586), UINT16_C(32526), UINT16_C(18693), UINT16_C( 5744), UINT16_C(57044),
        UINT16_C(11693), UINT16_C(14943), UINT16_C(33521), UINT16_C(43196), UINT16_C(50841), UINT16_C(41709), UINT16_C(14035), UINT16_C(39092),
        UINT16_C(51959), UINT16_C(58508), UINT16_C(65212), UINT16_C(51977), UINT16_C( 3710), UINT16_C(60948), UINT16_C(59684), UINT16_C(53708) },
      { UINT16_C(11286), UINT16_C( 1804), UINT16_C(51374), UINT16_C(18351), UINT16_C(40078), UINT16_C(25065), UINT16_C(40659), UINT16_C(51962),
        UINT16_C(34408), UINT16_C( 9390), UINT16_C(46980), UINT16_C(  751), UINT16_C( 1221), UINT16_C(59889), UINT16_C(48621), UINT16_C(  954),
        UINT16_C(50921), UINT16_C(38922), UINT16_C(47758), UINT16_C( 7391), UINT16_C(51542), UINT16_C(10622), UINT16_C(30823), UINT16_C(53235),
        UINT16_C(41470), UINT16_C(33523), UINT16_C(58200), UINT16_C( 7557), UINT16_C(30439), UINT16_C(54278), UINT16_C(49459), UINT16_C( 7639) },
      { UINT16_C(11286), UINT16_C( 1804), UINT16_C(50317), UINT16_C(18351), UINT16_C( 5814), UINT16_C(17674), UINT16_C( 3337), UINT16_C( 4681),
        UINT16_C(34408), UINT16_C( 9390), UINT16_C(46980), UINT16_C(  751), UINT16_C( 1221), UINT16_C(18693), UINT16_C( 5744), UINT16_C(  954),
        UINT16_C(11693), UINT16_C(14943), UINT16_C(33521), UINT16_C( 7391), UINT16_C(50841), UINT16_C(10622), UINT16_C(14035), UINT16_C(39092),
        UINT16_C(41470), UINT16_C(33523), UINT16_C(58200), UINT16_C( 7557), UINT16_C( 3710), UINT16_C(54278), UINT16_C(49459), UINT16_C( 7639) } },
    { { UINT16_C(57735), UINT16_C( 5813), UINT16_C(38043), UINT16_C(62002), UINT16_C(45149), UINT16_C(50203), UINT16_C( 3880), UINT16_C( 9875),
        UINT16_C(34736), UINT16_C( 2473), UINT16_C(11882), UINT16_C(20774), UINT16_C(11684), UINT16_C(55077), UINT16_C(64750), UINT16_C(30196),
        UINT16_C(43485), UINT16_C(31115), UINT16_C(48702), UINT16_C(39787), UINT16_C(34414), UINT16_C(38752), UINT16_C(62357), UINT16_C(18109),
        UINT16_C(26234), UINT16_C(58447), UINT16_C(30100), UINT16_C(14389), UINT16_C(23202), UINT16_C(36880), UINT16_C( 1110), UINT16_C(13318) },
      { UINT16_C(37294), UINT16_C(60589), UINT16_C( 6223), UINT16_C(48775), UINT16_C(59294), UINT16_C(13397), UINT16_C( 4827), UINT16_C(21882),
        UINT16_C(51577), UINT16_C( 3386), UINT16_C(28478), UINT16_C(57670), UINT16_C(22218), UINT16_C( 8305), UINT16_C(30554), UINT16_C( 2132),
        UINT16_C(  265), UINT16_C(22772), UINT16_C(31769), UINT16_C(47126), UINT16_C(27491), UINT16_C(16108), UINT16_C(26238), UINT16_C(63380),
        UINT16_C(52783), UINT16_C(27908), UINT16_C(19005), UINT16_C( 1870), UINT16_C(49312), UINT16_C(64296), UINT16_C(31799), UINT16_C(16387) },
      { UINT16_C(37294), UINT16_C( 5813), UINT16_C( 6223), UINT16_C(48775), UINT16_C(45149), UINT16_C(13397), UINT16_C( 3880), UINT16_C( 9875),
        UINT16_C(34736), UINT16_C( 2473), UINT16_C(11882), UINT16_C(20774), UINT16_C(11684), UINT16_C( 8305), UINT16_C(30554), UINT16_C( 2132),
        UINT16_C(  265), UINT16_C(22772), UINT16_C(31769), UINT16_C(39787), UINT16_C(27491), UINT16_C(16108), UINT16_C(26238), UINT16_C(18109),
        UINT16_C(26234), UINT16_C(27908), UINT16_C(19005), UINT16_C( 1870), UINT16_C(23202), UINT16_C(36880), UINT16_C( 1110), UINT16_C(13318) } },
    { { UINT16_C(63614), UINT16_C(38809), UINT16_C(44916), UINT16_C(55119), UINT16_C(15131), UINT16_C(39190), UINT16_C(43681), UINT16_C(53392),
        UINT16_C(38008), UINT16_C(46398), UINT16_C(36063), UINT16_C(32701), UINT16_C(58700), UINT16_C(33914), UINT16_C(32353), UINT16_C(57284),
        UINT16_C(23926), UINT16_C(60023), UINT16_C(50701), UINT16_C(10433), UINT16_C(55042), UINT16_C(41921), UINT16_C(20865), UINT16_C(63860),
        UINT16_C(45797), UINT16_C(50351), UINT16_C(27710), UINT16_C(35652), UINT16_C(48721), UINT16_C(45583), UINT16_C(54076), UINT16_C(45714) },
      { UINT16_C( 2353), UINT16_C(16028), UINT16_C(24271), UINT16_C(53606), UINT16_C(10037), UINT16_C(46965), UINT16_C(59768), UINT16_C(23984),
        UINT16_C(24475), UINT16_C(55586), UINT16_C(26315), UINT16_C( 7268), UINT16_C(29476), UINT16_C(25039), UINT16_C(24903), UINT16_C(30739),
        UINT16_C(45162), UINT16_C(14774), UINT16_C( 7182), UINT16_C(17163), UINT16_C(32835), UINT16_C(48122), UINT16_C(43881), UINT16_C( 1048),
        UINT16_C(14858), UINT16_C(55005), UINT16_C(17056), UINT16_C(50674), UINT16_C(49589), UINT16_C(64550), UINT16_C(14626), UINT16_C(35956) },
      { UINT16_C( 2353), UINT16_C(16028), UINT16_C(24271), UINT16_C(53606), UINT16_C(10037), UINT16_C(39190), UINT16_C(43681), UINT16_C(23984),
        UINT16_C(24475), UINT16_C(46398), UINT16_C(26315), UINT16_C( 7268), UINT16_C(29476), UINT16_C(25039), UINT16_C(24903), UINT16_C(30739),
        UINT16_C(23926), UINT16_C(14774), UINT16_C( 7182), UINT16_C(10433), UINT16_C(32835), UINT16_C(41921), UINT16_C(20865), UINT16_C( 1048),
        UINT16_C(14858), UINT16_C(50351), UINT16_C(17056), UINT16_C(35652), UINT16_C(48721), UINT16_C(45583), UINT16_C(14626), UINT16_C(35956) } },
    { { UINT16_C(10985), UINT16_C(63430), UINT16_C(53574), UINT16_C(35131), UINT16_C(13649), UINT16_C(47684), UINT16_C(24032), UINT16_C(60350),
        UINT16_C(39831), UINT16_C(14529), UINT16_C(46045), UINT16_C(37885), UINT16_C( 9077), UINT16_C(38799), UINT16_C( 1116), UINT16_C(17956),
        UINT16_C(59950), UINT16_C(30013), UINT16_C(30907), UINT16_C( 3326), UINT16_C(17326), UINT16_C(36550), UINT16_C(33952), UINT16_C(14201),
        UINT16_C(14879), UINT16_C(64879), UINT16_C(27886), UINT16_C(25488), UINT16_C( 8079), UINT16_C(60666), UINT16_C( 7715), UINT16_C(21042) },
      { UINT16_C(28424), UINT16_C(50119), UINT16_C(50664), UINT16_C(38607), UINT16_C(38152), UINT16_C(43044), UINT16_C(40473), UINT16_C(14816),
        UINT16_C(20440), UINT16_C(50742), UINT16_C(50876), UINT16_C(19241), UINT16_C( 9445), UINT16_C( 2359), UINT16_C(26946), UINT16_C(19291),
        UINT16_C( 8921), UINT16_C(49422), UINT16_C(57063), UINT16_C(61527), UINT16_C(31603), UINT16_C(36248), UINT16_C(30745), UINT16_C(62150),
        UINT16_C(64712), UINT16_C(33976), UINT16_C(58050), UINT16_C(42959), UINT16_C( 1798), UINT16_C(18608), UINT16_C( 2928), UINT16_C(18835) },
      { UINT16_C(10985), UINT16_C(50119), UINT16_C(50664), UINT16_C(35131), UINT16_C(13649), UINT16_C(43044), UINT16_C(24032), UINT16_C(14816),
        UINT16_C(20440), UINT16_C(14529), UINT16_C(46045), UINT16_C(19241), UINT16_C( 9077), UINT16_C( 2359), UINT16_C( 1116), UINT16_C(17956),
        UINT16_C( 8921), UINT16_C(30013), UINT16_C(30907), UINT16_C( 3326), UINT16_C(17326), UINT16_C(36248), UINT16_C(30745), UINT16_C(14201),
        UINT16_C(14879), UINT16_C(33976), UINT16_C(27886), UINT16_C(25488), UINT16_C( 1798), UINT16_C(18608), UINT16_C( 2928), UINT16_C(18835) } },
    { { UINT16_C(41517), UINT16_C( 5386), UINT16_C(24960), UINT16_C(62213), UINT16_C(40413), UINT16_C(63104), UINT16_C(17942), UINT16_C(57064),
        UINT16_C(41282), UINT16_C( 1122), UINT16_C(12675), UINT16_C(35244), UINT16_C(23608), UINT16_C(43473), UINT16_C(25960), UINT16_C(38386),
        UINT16_C(64775), UINT16_C(34730), UINT16_C(44894), UINT16_C(15226), UINT16_C(64333), UINT16_C(25394), UINT16_C( 6721), UINT16_C(33857),
        UINT16_C(41915), UINT16_C(16008), UINT16_C(13524), UINT16_C( 3527), UINT16_C(39313), UINT16_C(63926), UINT16_C(43262), UINT16_C( 1422) },
      { UINT16_C(14757), UINT16_C( 1164), UINT16_C( 1768), UINT16_C(13631), UINT16_C(28929), UINT16_C(17304), UINT16_C(55692), UINT16_C(18375),
        UINT16_C(20348), UINT16_C(20870), UINT16_C(19844), UINT16_C( 5470), UINT16_C( 5350), UINT16_C(58382), UINT16_C(40124), UINT16_C(25321),
        UINT16_C(30165), UINT16_C(48742), UINT16_C(42364), UINT16_C(32243), UINT16_C(35863), UINT16_C(41920), UINT16_C(34661), UINT16_C(58090),
        UINT16_C(28887), UINT16_C(23347), UINT16_C(37310), UINT16_C(42096), UINT16_C(32421), UINT16_C(24969), UINT16_C(29210), UINT16_C(61635) },
      { UINT16_C(14757), UINT16_C( 1164), UINT16_C( 1768), UINT16_C(13631), UINT16_C(28929), UINT16_C(17304), UINT16_C(17942), UINT16_C(18375),
        UINT16_C(20348), UINT16_C( 1122), UINT16_C(12675), UINT16_C( 5470), UINT16_C( 5350), UINT16_C(43473), UINT16_C(25960), UINT16_C(25321),
        UINT16_C(30165), UINT16_C(34730), UINT16_C(42364), UINT16_C(15226), UINT16_C(35863), UINT16_C(25394), UINT16_C( 6721), UINT16_C(33857),
        UINT16_C(28887), UINT16_C(16008), UINT16_C(13524), UINT16_C( 3527), UINT16_C(32421), UINT16_C(24969), UINT16_C(29210), UINT16_C( 1422) } },
    { { UINT16_C(10728), UINT16_C(25774), UINT16_C(41423), UINT16_C(59105), UINT16_C(41517), UINT16_C(37769), UINT16_C(29481), UINT16_C(  117),
        UINT16_C(43236), UINT16_C(41563), UINT16_C(52025), UINT16_C(56902), UINT16_C(53065), UINT16_C(25663), UINT16_C(  834), UINT16_C(10836),
        UINT16_C(  556), UINT16_C(64398), UINT16_C(28579), UINT16_C(53729), UINT16_C(27153), UINT16_C(15204), UINT16_C(55774), UINT16_C(49723),
        UINT16_C(38785), UINT16_C(47716), UINT16_C(43618), UINT16_C(44184), UINT16_C(55162), UINT16_C(48144), UINT16_C(25818), UINT16_C( 2022) },
      { UINT16_C(29798), UINT16_C( 2306), UINT16_C(58595), UINT16_C(62938), UINT16_C(15950), UINT16_C(11312), UINT16_C(27415), UINT16_C(39150),
        UINT16_C(20994), UINT16_C(25938), UINT16_C(60157), UINT16_C(30481), UINT16_C( 8642), UINT16_C(39987), UINT16_C( 6533), UINT16_C(60323),
        UINT16_C(42637), UINT16_C(28916), UINT16_C(53130), UINT16_C(55397), UINT16_C(38157), UINT16_C( 9477), UINT16_C(62209), UINT16_C(  957),
        UINT16_C( 4166), UINT16_C(17256), UINT16_C(31226), UINT16_C(48314), UINT16_C(60826), UINT16_C( 8025), UINT16_C(64518), UINT16_C(37642) },
      { UINT16_C(10728), UINT16_C( 2306), UINT16_C(41423), UINT16_C(59105), UINT16_C(15950), UINT16_C(11312), UINT16_C(27415), UINT16_C(  117),
        UINT16_C(20994), UINT16_C(25938), UINT16_C(52025), UINT16_C(30481), UINT16_C( 8642), UINT16_C(25663), UINT16_C(  834), UINT16_C(10836),
        UINT16_C(  556), UINT16_C(28916), UINT16_C(28579), UINT16_C(53729), UINT16_C(27153), UINT16_C( 9477), UINT16_C(55774), UINT16_C(  957),
        UINT16_C( 4166), UINT16_C(17256), UINT16_C(31226), UINT16_C(44184), UINT16_C(55162), UINT16_C( 8025), UINT16_C(25818), UINT16_C( 2022) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_min_epu16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_epu16");
    easysimd_test_x86_assert_equal_u16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_min_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const uint16_t a[32];
    const uint16_t b[32];
    const uint16_t r[32];
  } test_vec[] = {
    { { -INT16_C(  3079),  INT16_C(  8612), -INT16_C( 25643),  INT16_C( 29615), -INT16_C( 23644),  INT16_C( 18229),  INT16_C( 13600),  INT16_C( 16573),
         INT16_C( 11343),  INT16_C( 21901), -INT16_C( 11893), -INT16_C(  6952), -INT16_C( 26016),  INT16_C( 21084),  INT16_C(  1957), -INT16_C( 25055),
        -INT16_C( 14854), -INT16_C( 12097),  INT16_C( 28256),  INT16_C(  1347),  INT16_C( 30737),  INT16_C( 12620),  INT16_C(  2734), -INT16_C(   655),
        -INT16_C(   458), -INT16_C( 15790),  INT16_C( 11215),  INT16_C( 12454),  INT16_C(   709),  INT16_C( 27266), -INT16_C( 23798),  INT16_C(  1032) },
      UINT32_C(3386165097),
      { UINT16_C( 7056), UINT16_C(15993), UINT16_C(59941), UINT16_C(23355), UINT16_C(36585), UINT16_C(47133), UINT16_C(50361), UINT16_C(32488),
        UINT16_C(27590), UINT16_C(53481), UINT16_C(61710), UINT16_C(30677), UINT16_C(43449), UINT16_C(61249), UINT16_C( 4033), UINT16_C(21046),
        UINT16_C(44842), UINT16_C(20368), UINT16_C(52378), UINT16_C(33707), UINT16_C(51290), UINT16_C( 4923), UINT16_C( 9356), UINT16_C(21393),
        UINT16_C(31375), UINT16_C(40227), UINT16_C(63596), UINT16_C( 9493), UINT16_C(22178), UINT16_C(25364), UINT16_C(19045), UINT16_C(37045) },
      { UINT16_C(18170), UINT16_C(38111), UINT16_C(35346), UINT16_C(27671), UINT16_C(21075), UINT16_C(57215), UINT16_C( 4214), UINT16_C( 1330),
        UINT16_C(22155), UINT16_C(63395), UINT16_C(47182), UINT16_C(61468), UINT16_C(12302), UINT16_C(29524), UINT16_C( 2426), UINT16_C(29699),
        UINT16_C(58191), UINT16_C(24840), UINT16_C( 8045), UINT16_C(49357), UINT16_C(19570), UINT16_C(59552), UINT16_C(53853), UINT16_C(59630),
        UINT16_C(37160), UINT16_C(30687), UINT16_C(64329), UINT16_C(22375), UINT16_C(47915), UINT16_C(42442), UINT16_C(52933), UINT16_C( 5146) },
      { UINT16_C( 7056), UINT16_C( 8612), UINT16_C(39893), UINT16_C(23355), UINT16_C(41892), UINT16_C(47133), UINT16_C( 4214), UINT16_C(16573),
        UINT16_C(22155), UINT16_C(53481), UINT16_C(47182), UINT16_C(58584), UINT16_C(39520), UINT16_C(21084), UINT16_C( 2426), UINT16_C(21046),
        UINT16_C(50682), UINT16_C(53439), UINT16_C( 8045), UINT16_C( 1347), UINT16_C(19570), UINT16_C(12620), UINT16_C( 9356), UINT16_C(21393),
        UINT16_C(31375), UINT16_C(49746), UINT16_C(11215), UINT16_C( 9493), UINT16_C(  709), UINT16_C(27266), UINT16_C(19045), UINT16_C( 5146) } },
    { {  INT16_C(  8881),  INT16_C(  7798),  INT16_C( 17218), -INT16_C( 19233),  INT16_C( 32656), -INT16_C(  4708), -INT16_C( 30127),  INT16_C( 31445),
        -INT16_C( 19429),  INT16_C( 25841),  INT16_C( 22703), -INT16_C(  9541), -INT16_C( 31212), -INT16_C(  9857), -INT16_C( 26284),  INT16_C(  1517),
         INT16_C( 25532), -INT16_C(   477),  INT16_C(   679),  INT16_C( 14258),  INT16_C( 20097), -INT16_C( 11484), -INT16_C(  1575), -INT16_C(  2995),
         INT16_C( 16045),  INT16_C( 23641),  INT16_C(  5270), -INT16_C( 21962), -INT16_C( 19046), -INT16_C(  4477),  INT16_C( 29007),  INT16_C(  3059) },
      UINT32_C(2064193492),
      { UINT16_C(54793), UINT16_C(57966), UINT16_C(48079), UINT16_C(31959), UINT16_C(12537), UINT16_C(36824), UINT16_C( 3652), UINT16_C(57146),
        UINT16_C(48580), UINT16_C( 5069), UINT16_C(49454), UINT16_C(  798), UINT16_C(10200), UINT16_C(61822), UINT16_C(12770), UINT16_C(60300),
        UINT16_C(64007), UINT16_C(55246), UINT16_C(42421), UINT16_C(44627), UINT16_C(11477), UINT16_C( 6462), UINT16_C(30778), UINT16_C(65272),
        UINT16_C(50741), UINT16_C(25617), UINT16_C(12167), UINT16_C(24423), UINT16_C(58710), UINT16_C(14416), UINT16_C(56598), UINT16_C( 7716) },
      { UINT16_C(62167), UINT16_C(36341), UINT16_C(18583), UINT16_C(27707), UINT16_C(31092), UINT16_C(44933), UINT16_C(32497), UINT16_C(10157),
        UINT16_C(48964), UINT16_C(52107), UINT16_C(62190), UINT16_C(17706), UINT16_C(31447), UINT16_C(61053), UINT16_C(41303), UINT16_C(12044),
        UINT16_C(  403), UINT16_C(10940), UINT16_C(63305), UINT16_C(48790), UINT16_C( 7281), UINT16_C(25197), UINT16_C( 6810), UINT16_C(56969),
        UINT16_C( 5337), UINT16_C(51369), UINT16_C(54022), UINT16_C(56845), UINT16_C(35405), UINT16_C(42444), UINT16_C(55340), UINT16_C(49108) },
      { UINT16_C( 8881), UINT16_C( 7798), UINT16_C(18583), UINT16_C(46303), UINT16_C(12537), UINT16_C(60828), UINT16_C( 3652), UINT16_C(10157),
        UINT16_C(48580), UINT16_C( 5069), UINT16_C(49454), UINT16_C(55995), UINT16_C(10200), UINT16_C(55679), UINT16_C(39252), UINT16_C( 1517),
        UINT16_C(  403), UINT16_C(65059), UINT16_C(  679), UINT16_C(44627), UINT16_C(20097), UINT16_C(54052), UINT16_C(63961), UINT16_C(62541),
        UINT16_C( 5337), UINT16_C(25617), UINT16_C( 5270), UINT16_C(24423), UINT16_C(35405), UINT16_C(14416), UINT16_C(55340), UINT16_C( 3059) } },
    { { -INT16_C( 28455),  INT16_C(  8938), -INT16_C( 32633), -INT16_C(  1824),  INT16_C( 19868),  INT16_C( 13915), -INT16_C(  7064),  INT16_C( 16660),
        -INT16_C( 16903), -INT16_C(   247),  INT16_C(  5776), -INT16_C(  8483), -INT16_C( 22111), -INT16_C( 12925),  INT16_C( 22401),  INT16_C( 23180),
         INT16_C( 30439),  INT16_C( 28285),  INT16_C( 24055), -INT16_C( 27801), -INT16_C( 15701),  INT16_C(  5066), -INT16_C(  8538), -INT16_C( 24748),
         INT16_C( 24220),  INT16_C( 11423),  INT16_C( 31860),  INT16_C(  5386), -INT16_C( 29402), -INT16_C( 22558),  INT16_C( 28644), -INT16_C( 13566) },
      UINT32_C(3694821349),
      { UINT16_C(14947), UINT16_C( 2458), UINT16_C(61208), UINT16_C(46249), UINT16_C(18509), UINT16_C(49633), UINT16_C(60356), UINT16_C(60119),
        UINT16_C(47481), UINT16_C(23954), UINT16_C(37928), UINT16_C( 3625), UINT16_C(25363), UINT16_C(61418), UINT16_C(23044), UINT16_C(26487),
        UINT16_C( 4500), UINT16_C(44400), UINT16_C( 6400), UINT16_C(19809), UINT16_C(16993), UINT16_C( 9743), UINT16_C(58926), UINT16_C(42768),
        UINT16_C(41631), UINT16_C(51204), UINT16_C(11574), UINT16_C(18902), UINT16_C(49296), UINT16_C(37945), UINT16_C(45083), UINT16_C(45051) },
      { UINT16_C(27841), UINT16_C(49756), UINT16_C(48773), UINT16_C(59151), UINT16_C( 7680), UINT16_C(11789), UINT16_C( 7428), UINT16_C(42197),
        UINT16_C(56000), UINT16_C(63084), UINT16_C(16903), UINT16_C(38976), UINT16_C(30978), UINT16_C( 7468), UINT16_C(10281), UINT16_C(60109),
        UINT16_C(10644), UINT16_C( 6572), UINT16_C(48359), UINT16_C(59392), UINT16_C( 3546), UINT16_C(57110), UINT16_C(60459), UINT16_C(60291),
        UINT16_C(61382), UINT16_C(52705), UINT16_C( 8497), UINT16_C(13157), UINT16_C(37530), UINT16_C(50001), UINT16_C( 7866), UINT16_C(20142) },
      { UINT16_C(14947), UINT16_C( 8938), UINT16_C(48773), UINT16_C(63712), UINT16_C(19868), UINT16_C(11789), UINT16_C( 7428), UINT16_C(42197),
        UINT16_C(47481), UINT16_C(23954), UINT16_C(16903), UINT16_C( 3625), UINT16_C(25363), UINT16_C( 7468), UINT16_C(10281), UINT16_C(23180),
        UINT16_C(30439), UINT16_C( 6572), UINT16_C(24055), UINT16_C(19809), UINT16_C( 3546), UINT16_C( 9743), UINT16_C(56998), UINT16_C(40788),
        UINT16_C(24220), UINT16_C(11423), UINT16_C( 8497), UINT16_C(13157), UINT16_C(37530), UINT16_C(42978), UINT16_C( 7866), UINT16_C(20142) } },
    { {  INT16_C( 23111),  INT16_C( 12135),  INT16_C( 26646), -INT16_C(  3817),  INT16_C( 11637), -INT16_C( 24368),  INT16_C( 21273), -INT16_C(  8309),
         INT16_C( 27970),  INT16_C( 29613),  INT16_C(  4750),  INT16_C( 10662), -INT16_C(  2140),  INT16_C( 24300), -INT16_C( 26091),  INT16_C( 23980),
         INT16_C(  5365),  INT16_C(  2956), -INT16_C( 23684), -INT16_C(  3588), -INT16_C( 13104), -INT16_C(  5486),  INT16_C(  7455),  INT16_C( 25033),
         INT16_C( 30346),  INT16_C(  6612),  INT16_C( 31625),  INT16_C( 11586),  INT16_C( 11890), -INT16_C( 30580),  INT16_C( 14537), -INT16_C( 16667) },
      UINT32_C(3368644940),
      { UINT16_C(19602), UINT16_C(45774), UINT16_C(39017), UINT16_C(62483), UINT16_C(59406), UINT16_C(38669), UINT16_C(20323), UINT16_C(54725),
        UINT16_C(20861), UINT16_C(18013), UINT16_C(17033), UINT16_C(54788), UINT16_C(52915), UINT16_C(51102), UINT16_C(22676), UINT16_C( 9900),
        UINT16_C(31396), UINT16_C( 3800), UINT16_C(60434), UINT16_C( 8450), UINT16_C( 4052), UINT16_C(14264), UINT16_C(32094), UINT16_C(56076),
        UINT16_C(27342), UINT16_C(22562), UINT16_C( 9900), UINT16_C(24622), UINT16_C(52468), UINT16_C(34855), UINT16_C(54053), UINT16_C(51631) },
      { UINT16_C(34638), UINT16_C(24791), UINT16_C(55667), UINT16_C(18305), UINT16_C(15080), UINT16_C(18046), UINT16_C(35767), UINT16_C(34338),
        UINT16_C(17653), UINT16_C(41438), UINT16_C( 3178), UINT16_C(24321), UINT16_C(10712), UINT16_C(64999), UINT16_C(38652), UINT16_C(19143),
        UINT16_C(40478), UINT16_C(37291), UINT16_C(11384), UINT16_C(24793), UINT16_C(22374), UINT16_C( 7847), UINT16_C(51682), UINT16_C(55204),
        UINT16_C(33293), UINT16_C(30585), UINT16_C(31374), UINT16_C(26326), UINT16_C(48803), UINT16_C(41060), UINT16_C(11092), UINT16_C(29418) },
      { UINT16_C(23111), UINT16_C(12135), UINT16_C(39017), UINT16_C(18305), UINT16_C(11637), UINT16_C(41168), UINT16_C(20323), UINT16_C(57227),
        UINT16_C(17653), UINT16_C(29613), UINT16_C( 4750), UINT16_C(10662), UINT16_C(10712), UINT16_C(51102), UINT16_C(22676), UINT16_C(23980),
        UINT16_C(31396), UINT16_C( 2956), UINT16_C(41852), UINT16_C( 8450), UINT16_C(52432), UINT16_C(60050), UINT16_C(32094), UINT16_C(55204),
        UINT16_C(30346), UINT16_C( 6612), UINT16_C(31625), UINT16_C(24622), UINT16_C(11890), UINT16_C(34956), UINT16_C(11092), UINT16_C(29418) } },
    { { -INT16_C( 27191),  INT16_C( 16644), -INT16_C(  8766),  INT16_C( 10402),  INT16_C( 18740),  INT16_C(  5958), -INT16_C(  5614),  INT16_C(  8174),
         INT16_C( 26476), -INT16_C(  1386),  INT16_C( 28130), -INT16_C( 31391), -INT16_C( 15061),  INT16_C( 32549),  INT16_C(  4336), -INT16_C( 17934),
        -INT16_C(  2395),  INT16_C( 26619), -INT16_C( 25133),  INT16_C(  1936), -INT16_C( 10522), -INT16_C(  2018),  INT16_C(  3521),  INT16_C( 11543),
        -INT16_C( 21132),  INT16_C( 22056), -INT16_C( 30438),  INT16_C( 17884),  INT16_C(   334),  INT16_C( 16069), -INT16_C( 18671), -INT16_C( 18441) },
      UINT32_C(2149511853),
      { UINT16_C(42629), UINT16_C(18029), UINT16_C(33971), UINT16_C(10099), UINT16_C(39730), UINT16_C(19582), UINT16_C(23076), UINT16_C(29330),
        UINT16_C(22363), UINT16_C(28080), UINT16_C(43022), UINT16_C(47908), UINT16_C(17050), UINT16_C(10811), UINT16_C(49905), UINT16_C(30367),
        UINT16_C( 3432), UINT16_C( 7100), UINT16_C(12177), UINT16_C(49987), UINT16_C(49611), UINT16_C(61200), UINT16_C(41499), UINT16_C(30306),
        UINT16_C( 4857), UINT16_C( 2019), UINT16_C( 1978), UINT16_C(21954), UINT16_C(64842), UINT16_C(15231), UINT16_C( 7871), UINT16_C(10417) },
      { UINT16_C(27947), UINT16_C(48451), UINT16_C(34460), UINT16_C(26496), UINT16_C(36935), UINT16_C(25175), UINT16_C(47410), UINT16_C(11225),
        UINT16_C(48331), UINT16_C(34354), UINT16_C(62660), UINT16_C( 3803), UINT16_C(23281), UINT16_C(45385), UINT16_C(64120), UINT16_C(42201),
        UINT16_C( 7271), UINT16_C(  865), UINT16_C(57763), UINT16_C(60011), UINT16_C(49778), UINT16_C(42061), UINT16_C( 9851), UINT16_C(18128),
        UINT16_C(  738), UINT16_C(42700), UINT16_C(42999), UINT16_C(59572), UINT16_C(64769), UINT16_C(31385), UINT16_C(29431), UINT16_C(24094) },
      { UINT16_C(27947), UINT16_C(16644), UINT16_C(33971), UINT16_C(10099), UINT16_C(18740), UINT16_C(19582), UINT16_C(59922), UINT16_C(11225),
        UINT16_C(26476), UINT16_C(28080), UINT16_C(28130), UINT16_C(34145), UINT16_C(17050), UINT16_C(10811), UINT16_C(49905), UINT16_C(30367),
        UINT16_C(63141), UINT16_C(  865), UINT16_C(12177), UINT16_C(49987), UINT16_C(49611), UINT16_C(63518), UINT16_C( 3521), UINT16_C(11543),
        UINT16_C(44404), UINT16_C(22056), UINT16_C(35098), UINT16_C(17884), UINT16_C(  334), UINT16_C(16069), UINT16_C(46865), UINT16_C(10417) } },
    { {  INT16_C( 32655),  INT16_C( 12898), -INT16_C( 12960), -INT16_C( 11748),  INT16_C( 27023),  INT16_C(  2679),  INT16_C( 18319),  INT16_C( 29264),
         INT16_C(  7497),  INT16_C( 16408), -INT16_C( 12860), -INT16_C( 14807), -INT16_C( 15670), -INT16_C( 15808),  INT16_C( 24117), -INT16_C( 15328),
        -INT16_C( 32035),  INT16_C( 15862),  INT16_C(  4687), -INT16_C(  8688), -INT16_C( 30852),  INT16_C(  3048),  INT16_C( 14798),  INT16_C(  6013),
        -INT16_C( 27050),  INT16_C(  6744), -INT16_C( 32413),  INT16_C( 11744),  INT16_C(  8259),  INT16_C( 30959),  INT16_C(  4222),  INT16_C( 23356) },
      UINT32_C(3801690770),
      { UINT16_C(43312), UINT16_C(65228), UINT16_C(19170), UINT16_C(14357), UINT16_C(28128), UINT16_C(17234), UINT16_C(13294), UINT16_C(12912),
        UINT16_C(24659), UINT16_C(53930), UINT16_C(59248), UINT16_C(  557), UINT16_C(50713), UINT16_C(24292), UINT16_C(42351), UINT16_C(40735),
        UINT16_C(60494), UINT16_C(12445), UINT16_C(45878), UINT16_C( 5736), UINT16_C(47648), UINT16_C( 3929), UINT16_C(51693), UINT16_C(16705),
        UINT16_C(60201), UINT16_C(39187), UINT16_C(16594), UINT16_C(60572), UINT16_C(32775), UINT16_C(30282), UINT16_C(27173), UINT16_C(29462) },
      { UINT16_C(45910), UINT16_C(36003), UINT16_C( 2918), UINT16_C(34722), UINT16_C(64454), UINT16_C(45974), UINT16_C(55236), UINT16_C(61172),
        UINT16_C( 1986), UINT16_C(38279), UINT16_C( 9032), UINT16_C(20353), UINT16_C(52132), UINT16_C(51653), UINT16_C(56117), UINT16_C(35645),
        UINT16_C(57487), UINT16_C(62743), UINT16_C(47596), UINT16_C(45692), UINT16_C( 4788), UINT16_C(31077), UINT16_C(23273), UINT16_C(44135),
        UINT16_C(61025), UINT16_C(43329), UINT16_C(49682), UINT16_C(46840), UINT16_C(48781), UINT16_C(50047), UINT16_C(48281), UINT16_C(10318) },
      { UINT16_C(32655), UINT16_C(36003), UINT16_C(52576), UINT16_C(53788), UINT16_C(28128), UINT16_C( 2679), UINT16_C(18319), UINT16_C(12912),
        UINT16_C( 7497), UINT16_C(38279), UINT16_C(52676), UINT16_C(50729), UINT16_C(50713), UINT16_C(24292), UINT16_C(24117), UINT16_C(50208),
        UINT16_C(57487), UINT16_C(15862), UINT16_C( 4687), UINT16_C( 5736), UINT16_C( 4788), UINT16_C( 3048), UINT16_C(14798), UINT16_C(16705),
        UINT16_C(38486), UINT16_C(39187), UINT16_C(33123), UINT16_C(11744), UINT16_C( 8259), UINT16_C(30282), UINT16_C(27173), UINT16_C(10318) } },
    { {  INT16_C( 26269), -INT16_C( 30434), -INT16_C( 26081), -INT16_C( 11205), -INT16_C( 24403), -INT16_C( 27059), -INT16_C( 19206),  INT16_C( 23618),
        -INT16_C( 31838), -INT16_C( 19451), -INT16_C(   443), -INT16_C( 11414), -INT16_C(  5444),  INT16_C( 21910), -INT16_C(  7002),  INT16_C( 17278),
        -INT16_C( 25526),  INT16_C( 27340),  INT16_C(  1846), -INT16_C(  7362), -INT16_C( 29784), -INT16_C( 23942), -INT16_C( 17345), -INT16_C(  7682),
         INT16_C(  1088), -INT16_C( 31338),  INT16_C(     2), -INT16_C( 16808), -INT16_C(  4374), -INT16_C( 28397), -INT16_C( 28205),  INT16_C(  7636) },
      UINT32_C(1686610221),
      { UINT16_C(49488), UINT16_C(36851), UINT16_C(61822), UINT16_C(48753), UINT16_C( 2037), UINT16_C(63299), UINT16_C(39943), UINT16_C(62133),
        UINT16_C(51594), UINT16_C(23939), UINT16_C(22362), UINT16_C(34939), UINT16_C(  760), UINT16_C(41452), UINT16_C(13256), UINT16_C( 6385),
        UINT16_C(58613), UINT16_C(29608), UINT16_C( 6614), UINT16_C(52017), UINT16_C(29728), UINT16_C(10179), UINT16_C(30736), UINT16_C(39705),
        UINT16_C(40001), UINT16_C(40184), UINT16_C(29684), UINT16_C(60452), UINT16_C( 4214), UINT16_C(16013), UINT16_C(32579), UINT16_C(14422) },
      { UINT16_C(65123), UINT16_C(14763), UINT16_C(56343), UINT16_C(14085), UINT16_C(51281), UINT16_C(24927), UINT16_C(30784), UINT16_C(33532),
        UINT16_C(62741), UINT16_C( 2334), UINT16_C(17000), UINT16_C(57077), UINT16_C(33618), UINT16_C(38172), UINT16_C(29442), UINT16_C(26062),
        UINT16_C(31089), UINT16_C(35231), UINT16_C(42070), UINT16_C(42944), UINT16_C( 8044), UINT16_C(44040), UINT16_C( 1432), UINT16_C(44334),
        UINT16_C(19706), UINT16_C(25270), UINT16_C(43918), UINT16_C(57409), UINT16_C(23854), UINT16_C(12406), UINT16_C(17616), UINT16_C(17046) },
      { UINT16_C(49488), UINT16_C(35102), UINT16_C(56343), UINT16_C(14085), UINT16_C(41133), UINT16_C(24927), UINT16_C(46330), UINT16_C(23618),
        UINT16_C(51594), UINT16_C(46085), UINT16_C(65093), UINT16_C(54122), UINT16_C(60092), UINT16_C(38172), UINT16_C(58534), UINT16_C( 6385),
        UINT16_C(31089), UINT16_C(29608), UINT16_C( 6614), UINT16_C(58174), UINT16_C(35752), UINT16_C(41594), UINT16_C(48191), UINT16_C(39705),
        UINT16_C( 1088), UINT16_C(34198), UINT16_C(29684), UINT16_C(48728), UINT16_C(61162), UINT16_C(12406), UINT16_C(17616), UINT16_C( 7636) } },
    { {  INT16_C( 13757),  INT16_C(  5067), -INT16_C( 29735),  INT16_C( 17850), -INT16_C( 15445),  INT16_C( 17393),  INT16_C(  8392), -INT16_C( 15632),
        -INT16_C( 22932), -INT16_C(  1244),  INT16_C( 25937), -INT16_C( 32549),  INT16_C( 20931), -INT16_C( 27728),  INT16_C( 18069),  INT16_C( 21461),
        -INT16_C( 24453),  INT16_C( 21606),  INT16_C(  8492), -INT16_C( 10343), -INT16_C( 29724), -INT16_C( 21478),  INT16_C(  2731),  INT16_C(  5998),
        -INT16_C( 27984),  INT16_C(   274), -INT16_C(  4360), -INT16_C( 17535),  INT16_C( 12863), -INT16_C( 10930),  INT16_C(  9336), -INT16_C(  3032) },
      UINT32_C(4031286980),
      { UINT16_C(57709), UINT16_C( 6207), UINT16_C(44523), UINT16_C(39727), UINT16_C(16960), UINT16_C(14493), UINT16_C( 7728), UINT16_C(28659),
        UINT16_C(16720), UINT16_C(51524), UINT16_C(27749), UINT16_C(10941), UINT16_C( 1531), UINT16_C(43546), UINT16_C(58087), UINT16_C(21566),
        UINT16_C(32195), UINT16_C(44908), UINT16_C(39979), UINT16_C(27466), UINT16_C(59358), UINT16_C( 3747), UINT16_C(38406), UINT16_C(22141),
        UINT16_C(49879), UINT16_C(15647), UINT16_C(56366), UINT16_C(10599), UINT16_C(33250), UINT16_C(51668), UINT16_C( 4707), UINT16_C(10014) },
      { UINT16_C(35471), UINT16_C(47830), UINT16_C( 8230), UINT16_C( 1061), UINT16_C(51208), UINT16_C( 3602), UINT16_C(36958), UINT16_C(13924),
        UINT16_C(33874), UINT16_C(32883), UINT16_C(55904), UINT16_C(17066), UINT16_C(32347), UINT16_C(48908), UINT16_C(10896), UINT16_C( 8166),
        UINT16_C(48308), UINT16_C(56282), UINT16_C(65500), UINT16_C(58591), UINT16_C(62152), UINT16_C( 9970), UINT16_C(22402), UINT16_C(54364),
        UINT16_C(53211), UINT16_C(15188), UINT16_C(65193), UINT16_C( 1406), UINT16_C(35452), UINT16_C( 3268), UINT16_C(43700), UINT16_C(26668) },
      { UINT16_C(13757), UINT16_C( 5067), UINT16_C( 8230), UINT16_C(17850), UINT16_C(50091), UINT16_C(17393), UINT16_C( 7728), UINT16_C(13924),
        UINT16_C(42604), UINT16_C(32883), UINT16_C(27749), UINT16_C(10941), UINT16_C(20931), UINT16_C(37808), UINT16_C(18069), UINT16_C( 8166),
        UINT16_C(41083), UINT16_C(21606), UINT16_C( 8492), UINT16_C(27466), UINT16_C(35812), UINT16_C(44058), UINT16_C(22402), UINT16_C( 5998),
        UINT16_C(37552), UINT16_C(  274), UINT16_C(61176), UINT16_C(48001), UINT16_C(33250), UINT16_C( 3268), UINT16_C( 4707), UINT16_C(10014) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_epu16(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_epu16");
    easysimd_test_x86_assert_equal_u16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask32 k;
    const uint16_t a[32];
    const uint16_t b[32];
    const uint16_t r[32];
  } test_vec[] = {
    { UINT32_C(2325775073),
      { UINT16_C(31861), UINT16_C(36471), UINT16_C( 6552), UINT16_C(33113), UINT16_C(15773), UINT16_C(39996), UINT16_C(39459), UINT16_C( 8945),
        UINT16_C(22730), UINT16_C( 7867), UINT16_C(31654), UINT16_C(34707), UINT16_C(13565), UINT16_C(10514), UINT16_C(50106), UINT16_C(12078),
        UINT16_C(42303), UINT16_C(55230), UINT16_C( 6078), UINT16_C(23385), UINT16_C(38228), UINT16_C(30711), UINT16_C(59439), UINT16_C(64154),
        UINT16_C(21824), UINT16_C(58904), UINT16_C(44241), UINT16_C(52846), UINT16_C(32992), UINT16_C(39672), UINT16_C( 9795), UINT16_C(33481) },
      { UINT16_C(34763), UINT16_C(35162), UINT16_C(45982), UINT16_C(62436), UINT16_C(56136), UINT16_C(30570), UINT16_C( 1219), UINT16_C( 1137),
        UINT16_C(35418), UINT16_C(11242), UINT16_C(22582), UINT16_C( 5881), UINT16_C(61912), UINT16_C( 7088), UINT16_C(30999), UINT16_C(58014),
        UINT16_C(63489), UINT16_C(40811), UINT16_C(20395), UINT16_C(62354), UINT16_C(64810), UINT16_C(61034), UINT16_C(56321), UINT16_C(23538),
        UINT16_C(56422), UINT16_C(40070), UINT16_C(32821), UINT16_C( 3506), UINT16_C(25201), UINT16_C(35113), UINT16_C(51163), UINT16_C(56427) },
      { UINT16_C(31861), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(30570), UINT16_C( 1219), UINT16_C( 1137),
        UINT16_C(    0), UINT16_C( 7867), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(12078),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(30711), UINT16_C(    0), UINT16_C(23538),
        UINT16_C(    0), UINT16_C(40070), UINT16_C(    0), UINT16_C( 3506), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(33481) } },
    { UINT32_C(1786566591),
      { UINT16_C(50955), UINT16_C( 3391), UINT16_C(12707), UINT16_C( 2408), UINT16_C(61197), UINT16_C(17061), UINT16_C(22383), UINT16_C(57424),
        UINT16_C(31161), UINT16_C(38249), UINT16_C(54592), UINT16_C(65393), UINT16_C(60844), UINT16_C(53865), UINT16_C(50940), UINT16_C( 1827),
        UINT16_C(25229), UINT16_C(12564), UINT16_C(32147), UINT16_C(41274), UINT16_C(57452), UINT16_C(56291), UINT16_C(13111), UINT16_C(61883),
        UINT16_C( 9644), UINT16_C(60550), UINT16_C(63482), UINT16_C(42731), UINT16_C(21733), UINT16_C(57720), UINT16_C(39962), UINT16_C(43240) },
      { UINT16_C(65022), UINT16_C(37593), UINT16_C( 4986), UINT16_C(58931), UINT16_C( 5875), UINT16_C(11201), UINT16_C(31818), UINT16_C(63004),
        UINT16_C(41633), UINT16_C(39907), UINT16_C(52889), UINT16_C(32321), UINT16_C(47651), UINT16_C(15711), UINT16_C(18518), UINT16_C(21733),
        UINT16_C(48709), UINT16_C(49126), UINT16_C( 6610), UINT16_C(50597), UINT16_C(26160), UINT16_C(31472), UINT16_C( 3298), UINT16_C(33904),
        UINT16_C(21422), UINT16_C(18463), UINT16_C(24866), UINT16_C(17862), UINT16_C( 9755), UINT16_C(29058), UINT16_C(26734), UINT16_C(46021) },
      { UINT16_C(50955), UINT16_C( 3391), UINT16_C( 4986), UINT16_C( 2408), UINT16_C( 5875), UINT16_C(11201), UINT16_C(    0), UINT16_C(57424),
        UINT16_C(31161), UINT16_C(38249), UINT16_C(52889), UINT16_C(    0), UINT16_C(47651), UINT16_C(    0), UINT16_C(18518), UINT16_C( 1827),
        UINT16_C(    0), UINT16_C(    0), UINT16_C( 6610), UINT16_C(41274), UINT16_C(26160), UINT16_C(31472), UINT16_C( 3298), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(18463), UINT16_C(    0), UINT16_C(17862), UINT16_C(    0), UINT16_C(29058), UINT16_C(26734), UINT16_C(    0) } },
    { UINT32_C(4168264742),
      { UINT16_C(44669), UINT16_C(24431), UINT16_C(57531), UINT16_C(27107), UINT16_C(  819), UINT16_C(21937), UINT16_C(30820), UINT16_C(32666),
        UINT16_C( 7582), UINT16_C( 3312), UINT16_C(46469), UINT16_C(43967), UINT16_C(12641), UINT16_C(10148), UINT16_C(25160), UINT16_C(50460),
        UINT16_C(35856), UINT16_C(52004), UINT16_C( 2156), UINT16_C(40757), UINT16_C(58891), UINT16_C(28661), UINT16_C(36702), UINT16_C(64750),
        UINT16_C(57004), UINT16_C(12552), UINT16_C(51091), UINT16_C(62941), UINT16_C(33272), UINT16_C(16412), UINT16_C(14563), UINT16_C(62213) },
      { UINT16_C(10948), UINT16_C(12479), UINT16_C(62514), UINT16_C(15824), UINT16_C(50650), UINT16_C(14764), UINT16_C(39508), UINT16_C(  309),
        UINT16_C(15992), UINT16_C( 2866), UINT16_C( 3845), UINT16_C(65024), UINT16_C( 7312), UINT16_C(29502), UINT16_C(17493), UINT16_C( 6503),
        UINT16_C( 9838), UINT16_C(41034), UINT16_C( 6682), UINT16_C(62685), UINT16_C(35295), UINT16_C(13101), UINT16_C(25379), UINT16_C(39732),
        UINT16_C(26529), UINT16_C(42662), UINT16_C(42870), UINT16_C( 1956), UINT16_C(58307), UINT16_C( 6266), UINT16_C(57639), UINT16_C(38194) },
      { UINT16_C(    0), UINT16_C(12479), UINT16_C(57531), UINT16_C(    0), UINT16_C(    0), UINT16_C(14764), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C( 3845), UINT16_C(43967), UINT16_C(    0), UINT16_C(10148), UINT16_C(    0), UINT16_C( 6503),
        UINT16_C(    0), UINT16_C(41034), UINT16_C(    0), UINT16_C(    0), UINT16_C(35295), UINT16_C(13101), UINT16_C(25379), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 1956), UINT16_C(33272), UINT16_C( 6266), UINT16_C(14563), UINT16_C(38194) } },
    { UINT32_C( 557153287),
      { UINT16_C(17307), UINT16_C(48808), UINT16_C(56742), UINT16_C(18265), UINT16_C(65348), UINT16_C(47854), UINT16_C(37542), UINT16_C(27329),
        UINT16_C(15477), UINT16_C(40066), UINT16_C(46109), UINT16_C( 9521), UINT16_C(26160), UINT16_C(50758), UINT16_C(23672), UINT16_C( 4923),
        UINT16_C(58528), UINT16_C(18129), UINT16_C(10945), UINT16_C( 1422), UINT16_C(31786), UINT16_C(53439), UINT16_C(33038), UINT16_C(33850),
        UINT16_C(48573), UINT16_C(55840), UINT16_C(21105), UINT16_C(41727), UINT16_C(18104), UINT16_C(12648), UINT16_C(42146), UINT16_C(16964) },
      { UINT16_C( 5768), UINT16_C(18825), UINT16_C( 5952), UINT16_C(27214), UINT16_C( 3475), UINT16_C(41275), UINT16_C(30094), UINT16_C(19237),
        UINT16_C(17970), UINT16_C(42022), UINT16_C( 9624), UINT16_C(20550), UINT16_C(44651), UINT16_C( 3713), UINT16_C(50770), UINT16_C(55888),
        UINT16_C(55772), UINT16_C( 7203), UINT16_C(29168), UINT16_C(33671), UINT16_C(49791), UINT16_C( 3365), UINT16_C(18999), UINT16_C(27225),
        UINT16_C(32656), UINT16_C(10254), UINT16_C(21668), UINT16_C( 4217), UINT16_C(64002), UINT16_C(21790), UINT16_C(28352), UINT16_C(39983) },
      { UINT16_C( 5768), UINT16_C(18825), UINT16_C( 5952), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C( 9624), UINT16_C( 9521), UINT16_C(26160), UINT16_C( 3713), UINT16_C(23672), UINT16_C(    0),
        UINT16_C(55772), UINT16_C(    0), UINT16_C(10945), UINT16_C(    0), UINT16_C(31786), UINT16_C( 3365), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(32656), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(12648), UINT16_C(    0), UINT16_C(    0) } },
    { UINT32_C( 951669576),
      { UINT16_C(57602), UINT16_C(14673), UINT16_C(43563), UINT16_C(48291), UINT16_C(45353), UINT16_C(52708), UINT16_C(23813), UINT16_C( 2269),
        UINT16_C(64344), UINT16_C( 6237), UINT16_C(35946), UINT16_C(45749), UINT16_C(28383), UINT16_C(42218), UINT16_C(42670), UINT16_C(45287),
        UINT16_C(14471), UINT16_C(46057), UINT16_C(36322), UINT16_C( 2927), UINT16_C(21310), UINT16_C(17625), UINT16_C(46769), UINT16_C( 2380),
        UINT16_C(43442), UINT16_C( 7201), UINT16_C(54837), UINT16_C( 5582), UINT16_C(47172), UINT16_C(62137), UINT16_C(41055), UINT16_C(59042) },
      { UINT16_C(36057), UINT16_C(48025), UINT16_C( 2073), UINT16_C(22471), UINT16_C(41052), UINT16_C( 3483), UINT16_C(59222), UINT16_C( 2070),
        UINT16_C(14224), UINT16_C(50724), UINT16_C(61966), UINT16_C(21211), UINT16_C(38059), UINT16_C( 2629), UINT16_C(59188), UINT16_C( 3568),
        UINT16_C(35443), UINT16_C(36041), UINT16_C(37010), UINT16_C(61156), UINT16_C(32560), UINT16_C(34555), UINT16_C( 4455), UINT16_C(63375),
        UINT16_C(45897), UINT16_C(22461), UINT16_C(39078), UINT16_C(20905), UINT16_C(60972), UINT16_C(24923), UINT16_C(19414), UINT16_C(18798) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(22471), UINT16_C(    0), UINT16_C(    0), UINT16_C(23813), UINT16_C(    0),
        UINT16_C(14224), UINT16_C( 6237), UINT16_C(    0), UINT16_C(    0), UINT16_C(28383), UINT16_C(    0), UINT16_C(42670), UINT16_C(    0),
        UINT16_C(14471), UINT16_C(    0), UINT16_C(    0), UINT16_C( 2927), UINT16_C(21310), UINT16_C(17625), UINT16_C(    0), UINT16_C( 2380),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 5582), UINT16_C(47172), UINT16_C(24923), UINT16_C(    0), UINT16_C(    0) } },
    { UINT32_C(1758869461),
      { UINT16_C(21049), UINT16_C(41086), UINT16_C( 3427), UINT16_C(44184), UINT16_C(21952), UINT16_C(26115), UINT16_C(44526), UINT16_C( 6839),
        UINT16_C( 4763), UINT16_C(29051), UINT16_C(59998), UINT16_C(13243), UINT16_C(37153), UINT16_C(59803), UINT16_C(62027), UINT16_C(34016),
        UINT16_C(24132), UINT16_C(42789), UINT16_C(48491), UINT16_C(11348), UINT16_C(22290), UINT16_C(  146), UINT16_C(18948), UINT16_C(40987),
        UINT16_C(38492), UINT16_C(47633), UINT16_C(52352), UINT16_C(41710), UINT16_C(35165), UINT16_C(43147), UINT16_C(27515), UINT16_C(48941) },
      { UINT16_C(21194), UINT16_C(13671), UINT16_C(47887), UINT16_C( 8545), UINT16_C(62482), UINT16_C( 5922), UINT16_C(15678), UINT16_C(39607),
        UINT16_C(51411), UINT16_C(21589), UINT16_C(17301), UINT16_C(62198), UINT16_C(33228), UINT16_C(18587), UINT16_C(51436), UINT16_C(46599),
        UINT16_C(28186), UINT16_C(10732), UINT16_C(19753), UINT16_C(15434), UINT16_C(27713), UINT16_C(32595), UINT16_C( 2729), UINT16_C(32026),
        UINT16_C(28626), UINT16_C(26577), UINT16_C(51122), UINT16_C(32346), UINT16_C(62792), UINT16_C(13510), UINT16_C(52925), UINT16_C(55275) },
      { UINT16_C(21049), UINT16_C(    0), UINT16_C( 3427), UINT16_C(    0), UINT16_C(21952), UINT16_C(    0), UINT16_C(15678), UINT16_C( 6839),
        UINT16_C( 4763), UINT16_C(21589), UINT16_C(17301), UINT16_C(    0), UINT16_C(33228), UINT16_C(18587), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(10732), UINT16_C(19753), UINT16_C(    0), UINT16_C(22290), UINT16_C(    0), UINT16_C( 2729), UINT16_C(32026),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(32346), UINT16_C(    0), UINT16_C(13510), UINT16_C(27515), UINT16_C(    0) } },
    { UINT32_C(1711331132),
      { UINT16_C(62903), UINT16_C(24805),      UINT16_MAX, UINT16_C(53725), UINT16_C(44654), UINT16_C( 8249), UINT16_C(37749), UINT16_C(48543),
        UINT16_C(25992), UINT16_C(17906), UINT16_C(56627), UINT16_C(28700), UINT16_C( 7348), UINT16_C(55510), UINT16_C(30822), UINT16_C( 7486),
        UINT16_C( 9325), UINT16_C(27774), UINT16_C(23331), UINT16_C(37437), UINT16_C(30218), UINT16_C(32690), UINT16_C(20745), UINT16_C(37181),
        UINT16_C(12215), UINT16_C(60118), UINT16_C(61964), UINT16_C(49242), UINT16_C(12302), UINT16_C(30104), UINT16_C(55208), UINT16_C( 5522) },
      { UINT16_C( 4347), UINT16_C( 7809), UINT16_C(49004), UINT16_C(30384), UINT16_C(25397), UINT16_C(16373), UINT16_C(12980), UINT16_C(27600),
        UINT16_C(42849), UINT16_C(27990), UINT16_C(45209), UINT16_C(43053), UINT16_C(50913), UINT16_C(35101), UINT16_C(44957), UINT16_C(39071),
        UINT16_C( 8384), UINT16_C(11446), UINT16_C(26591), UINT16_C( 5538), UINT16_C(38858), UINT16_C(32340), UINT16_C( 9418), UINT16_C(11242),
        UINT16_C(16587), UINT16_C(26009), UINT16_C(50928), UINT16_C(53517), UINT16_C(10892), UINT16_C(10587), UINT16_C(64217), UINT16_C(39361) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(49004), UINT16_C(30384), UINT16_C(25397), UINT16_C( 8249), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(25992), UINT16_C(17906), UINT16_C(45209), UINT16_C(    0), UINT16_C( 7348), UINT16_C(    0), UINT16_C(30822), UINT16_C( 7486),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(26009), UINT16_C(50928), UINT16_C(    0), UINT16_C(    0), UINT16_C(10587), UINT16_C(55208), UINT16_C(    0) } },
    { UINT32_C(4207245338),
      { UINT16_C(25599), UINT16_C(51495), UINT16_C( 4487), UINT16_C(21492), UINT16_C(36177), UINT16_C(17080), UINT16_C(50516), UINT16_C(57363),
        UINT16_C(28399), UINT16_C(51210), UINT16_C(52072), UINT16_C(33634), UINT16_C(10051), UINT16_C( 8829), UINT16_C(35983), UINT16_C(36555),
        UINT16_C(62447), UINT16_C(30295), UINT16_C(19204), UINT16_C(22217), UINT16_C(33241), UINT16_C(11672), UINT16_C(43846), UINT16_C(13581),
        UINT16_C( 5914), UINT16_C(33534), UINT16_C(24803), UINT16_C( 9733), UINT16_C(33415), UINT16_C( 5705), UINT16_C( 5134), UINT16_C(64932) },
      { UINT16_C(64263), UINT16_C( 3188), UINT16_C(15687), UINT16_C( 8290), UINT16_C(64191), UINT16_C( 1357), UINT16_C(23205), UINT16_C(48955),
        UINT16_C(14706), UINT16_C(21826), UINT16_C(18329), UINT16_C( 8315), UINT16_C(50378), UINT16_C(55351), UINT16_C(56281), UINT16_C(57558),
        UINT16_C(19159), UINT16_C( 7916), UINT16_C(20103), UINT16_C(17982), UINT16_C(35656), UINT16_C(61004), UINT16_C(34789), UINT16_C(22445),
        UINT16_C(61376), UINT16_C(22956), UINT16_C(10295), UINT16_C(  377), UINT16_C(45292), UINT16_C(50649), UINT16_C(44940), UINT16_C(25510) },
      { UINT16_C(    0), UINT16_C( 3188), UINT16_C(    0), UINT16_C( 8290), UINT16_C(36177), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C( 8315), UINT16_C(10051), UINT16_C( 8829), UINT16_C(35983), UINT16_C(    0),
        UINT16_C(19159), UINT16_C(    0), UINT16_C(19204), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(34789), UINT16_C(13581),
        UINT16_C(    0), UINT16_C(22956), UINT16_C(    0), UINT16_C(  377), UINT16_C(33415), UINT16_C( 5705), UINT16_C( 5134), UINT16_C(25510) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_epu16(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_epu16");
    easysimd_test_x86_assert_equal_u16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_min_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(   851604017), -INT32_C(   751915793),  INT32_C(  1683302456),  INT32_C(  1978536698), -INT32_C(  1145431871), -INT32_C(  1558774560),  INT32_C(  1709346334), -INT32_C(  1299143295),
         INT32_C(   132469272),  INT32_C(   920327166),  INT32_C(   899297339),  INT32_C(    78285123), -INT32_C(  2118163039),  INT32_C(  2133120353), -INT32_C(   438040988), -INT32_C(  1013484373) },
      { -INT32_C(   976519994), -INT32_C(   889477489),  INT32_C(   419469014), -INT32_C(  1071797729),  INT32_C(  1900141839),  INT32_C(   384853426),  INT32_C(   385602923),  INT32_C(   282760009),
        -INT32_C(  1613388529),  INT32_C(   560582731), -INT32_C(  2059703962),  INT32_C(   591747092), -INT32_C(   392919499),  INT32_C(  1459521003), -INT32_C(  1553073574), -INT32_C(  1665972339) },
      { -INT32_C(   976519994), -INT32_C(   889477489),  INT32_C(   419469014), -INT32_C(  1071797729), -INT32_C(  1145431871), -INT32_C(  1558774560),  INT32_C(   385602923), -INT32_C(  1299143295),
        -INT32_C(  1613388529),  INT32_C(   560582731), -INT32_C(  2059703962),  INT32_C(    78285123), -INT32_C(  2118163039),  INT32_C(  1459521003), -INT32_C(  1553073574), -INT32_C(  1665972339) } },
    { {  INT32_C(   926648556), -INT32_C(  1084709543),  INT32_C(   574984974),  INT32_C(   558271212), -INT32_C(    66463215), -INT32_C(  1185740705), -INT32_C(  1889681406), -INT32_C(   215281657),
        -INT32_C(   232101991),  INT32_C(   447840780),  INT32_C(    37615126), -INT32_C(  1843166335), -INT32_C(  1114755747),  INT32_C(   930537781), -INT32_C(  1463364703),  INT32_C(  2107372004) },
      {  INT32_C(  1685046616),  INT32_C(  1585389640), -INT32_C(  1738490857), -INT32_C(  1674935489), -INT32_C(   447104848),  INT32_C(   991744154), -INT32_C(  2015042909),  INT32_C(   738492372),
        -INT32_C(  1919847611), -INT32_C(  1410658156),  INT32_C(   188959692),  INT32_C(  2141679054), -INT32_C(  1083965147),  INT32_C(  1962639825),  INT32_C(   939319139), -INT32_C(  1553727394) },
      {  INT32_C(   926648556), -INT32_C(  1084709543), -INT32_C(  1738490857), -INT32_C(  1674935489), -INT32_C(   447104848), -INT32_C(  1185740705), -INT32_C(  2015042909), -INT32_C(   215281657),
        -INT32_C(  1919847611), -INT32_C(  1410658156),  INT32_C(    37615126), -INT32_C(  1843166335), -INT32_C(  1114755747),  INT32_C(   930537781), -INT32_C(  1463364703), -INT32_C(  1553727394) } },
    { {  INT32_C(   137426292), -INT32_C(   776791291),  INT32_C(   903673446), -INT32_C(  2001435805),  INT32_C(  1430788228), -INT32_C(    37076071), -INT32_C(  2144025054),  INT32_C(   975476934),
        -INT32_C(  1841146739), -INT32_C(   698092176),  INT32_C(  1309360107),  INT32_C(  1205321667),  INT32_C(  1906122712), -INT32_C(  2073139358), -INT32_C(   217799891), -INT32_C(   919721925) },
      { -INT32_C(   312774531),  INT32_C(  1371783014), -INT32_C(  1046425602),  INT32_C(  1711896462), -INT32_C(   120019306),  INT32_C(   981222925),  INT32_C(   623739113),  INT32_C(   653155241),
         INT32_C(   840124876),  INT32_C(   126080520),  INT32_C(   885531557),  INT32_C(   815452570), -INT32_C(  2077724041), -INT32_C(  1564564295), -INT32_C(   825758683),  INT32_C(   334804295) },
      { -INT32_C(   312774531), -INT32_C(   776791291), -INT32_C(  1046425602), -INT32_C(  2001435805), -INT32_C(   120019306), -INT32_C(    37076071), -INT32_C(  2144025054),  INT32_C(   653155241),
        -INT32_C(  1841146739), -INT32_C(   698092176),  INT32_C(   885531557),  INT32_C(   815452570), -INT32_C(  2077724041), -INT32_C(  2073139358), -INT32_C(   825758683), -INT32_C(   919721925) } },
    { {  INT32_C(   121964543), -INT32_C(  2096182819), -INT32_C(  2017994772),  INT32_C(   548884904),  INT32_C(  2107957444), -INT32_C(  1457560700), -INT32_C(  1770526897), -INT32_C(  1683330148),
         INT32_C(  1352920946), -INT32_C(  1512853064),  INT32_C(   825002632), -INT32_C(  1622023205),  INT32_C(  1209857475), -INT32_C(  1477362600), -INT32_C(  1086428893),  INT32_C(  1197205716) },
      { -INT32_C(  1869087017),  INT32_C(   943024815), -INT32_C(   815177228),  INT32_C(   141539908),  INT32_C(   139496367), -INT32_C(   357613113),  INT32_C(  2141908394), -INT32_C(  1379531307),
        -INT32_C(  1304601341), -INT32_C(  1142263097),  INT32_C(   394941395), -INT32_C(  1121978099),  INT32_C(  1288007557),  INT32_C(  1530361009),  INT32_C(   937091426), -INT32_C(   370892570) },
      { -INT32_C(  1869087017), -INT32_C(  2096182819), -INT32_C(  2017994772),  INT32_C(   141539908),  INT32_C(   139496367), -INT32_C(  1457560700), -INT32_C(  1770526897), -INT32_C(  1683330148),
        -INT32_C(  1304601341), -INT32_C(  1512853064),  INT32_C(   394941395), -INT32_C(  1622023205),  INT32_C(  1209857475), -INT32_C(  1477362600), -INT32_C(  1086428893), -INT32_C(   370892570) } },
    { { -INT32_C(   996466179),  INT32_C(  1719633555), -INT32_C(   411170087), -INT32_C(  2002477821), -INT32_C(  1093310195),  INT32_C(  1058606301), -INT32_C(   747113235), -INT32_C(  1833149548),
         INT32_C(   274093949), -INT32_C(  1216882979), -INT32_C(   476121632), -INT32_C(  1620295022), -INT32_C(  2007154261),  INT32_C(   986216269), -INT32_C(    15909013),  INT32_C(   395430298) },
      { -INT32_C(    30873568), -INT32_C(  1632264258),  INT32_C(   646009748),  INT32_C(  1086778773),  INT32_C(  2076713774),  INT32_C(    95785114),  INT32_C(  1778762447), -INT32_C(  1400793461),
         INT32_C(  1017817470), -INT32_C(   589668536),  INT32_C(  1191402674),  INT32_C(  2022164809), -INT32_C(  2014097428), -INT32_C(  1349735968), -INT32_C(   149319317), -INT32_C(  1499227352) },
      { -INT32_C(   996466179), -INT32_C(  1632264258), -INT32_C(   411170087), -INT32_C(  2002477821), -INT32_C(  1093310195),  INT32_C(    95785114), -INT32_C(   747113235), -INT32_C(  1833149548),
         INT32_C(   274093949), -INT32_C(  1216882979), -INT32_C(   476121632), -INT32_C(  1620295022), -INT32_C(  2014097428), -INT32_C(  1349735968), -INT32_C(   149319317), -INT32_C(  1499227352) } },
    { { -INT32_C(  1914483388),  INT32_C(  1583988140),  INT32_C(  1671785497),  INT32_C(   584789045),  INT32_C(  1537855099), -INT32_C(   485804681), -INT32_C(   270916409),  INT32_C(    76905919),
         INT32_C(  1989245130),  INT32_C(  1339357750), -INT32_C(  1666025113),  INT32_C(   549359013), -INT32_C(   746821796),  INT32_C(  1689683869),  INT32_C(  1800638635), -INT32_C(   680531955) },
      { -INT32_C(  1756561311),  INT32_C(  1642471930),  INT32_C(  1073650074), -INT32_C(  2107589594), -INT32_C(  1051272156),  INT32_C(   237309027), -INT32_C(  1434879843), -INT32_C(  1048385440),
        -INT32_C(   480718872), -INT32_C(  1958461455), -INT32_C(    20233512),  INT32_C(   595667967),  INT32_C(  1793382151), -INT32_C(  2139616797), -INT32_C(   517213567), -INT32_C(  1012683302) },
      { -INT32_C(  1914483388),  INT32_C(  1583988140),  INT32_C(  1073650074), -INT32_C(  2107589594), -INT32_C(  1051272156), -INT32_C(   485804681), -INT32_C(  1434879843), -INT32_C(  1048385440),
        -INT32_C(   480718872), -INT32_C(  1958461455), -INT32_C(  1666025113),  INT32_C(   549359013), -INT32_C(   746821796), -INT32_C(  2139616797), -INT32_C(   517213567), -INT32_C(  1012683302) } },
    { {  INT32_C(  1839659900),  INT32_C(   318368314),  INT32_C(   739361837), -INT32_C(   162557201),  INT32_C(  1281373033), -INT32_C(  1110583236), -INT32_C(  1516308278),  INT32_C(   560480677),
         INT32_C(  2005863997),  INT32_C(   629836024),  INT32_C(   995203916), -INT32_C(  1775132627),  INT32_C(   266506707),  INT32_C(   885829481),  INT32_C(  1289317287), -INT32_C(   378650196) },
      {  INT32_C(  1197604175), -INT32_C(   781325435), -INT32_C(  1291010426),  INT32_C(   843660639),  INT32_C(   927083470), -INT32_C(  2106913061),  INT32_C(   651117689), -INT32_C(   737198715),
        -INT32_C(  1088655302), -INT32_C(   510621349), -INT32_C(  1500210105), -INT32_C(  1478894119), -INT32_C(   455206135), -INT32_C(  1553577431),  INT32_C(   348730766), -INT32_C(  1394026382) },
      {  INT32_C(  1197604175), -INT32_C(   781325435), -INT32_C(  1291010426), -INT32_C(   162557201),  INT32_C(   927083470), -INT32_C(  2106913061), -INT32_C(  1516308278), -INT32_C(   737198715),
        -INT32_C(  1088655302), -INT32_C(   510621349), -INT32_C(  1500210105), -INT32_C(  1775132627), -INT32_C(   455206135), -INT32_C(  1553577431),  INT32_C(   348730766), -INT32_C(  1394026382) } },
    { { -INT32_C(  1536490423), -INT32_C(   712574067),  INT32_C(  1887115927),  INT32_C(    18306296), -INT32_C(  1712982417), -INT32_C(   834909376), -INT32_C(   203291263),  INT32_C(   648072157),
         INT32_C(  1573587919), -INT32_C(  1640869625),  INT32_C(  1661971819),  INT32_C(  1902388738),  INT32_C(  1544177948),  INT32_C(   371934869),  INT32_C(   688459083), -INT32_C(  1471174184) },
      { -INT32_C(  1140516171), -INT32_C(   681953429), -INT32_C(   432379420), -INT32_C(  1403478128),  INT32_C(  2097767144), -INT32_C(   191679319),  INT32_C(   438148417), -INT32_C(    71144122),
        -INT32_C(   206059641), -INT32_C(   490073346),  INT32_C(   197723259), -INT32_C(  1934089821),  INT32_C(   738836867),  INT32_C(   908172789), -INT32_C(  2142224838),  INT32_C(   863769259) },
      { -INT32_C(  1536490423), -INT32_C(   712574067), -INT32_C(   432379420), -INT32_C(  1403478128), -INT32_C(  1712982417), -INT32_C(   834909376), -INT32_C(   203291263), -INT32_C(    71144122),
        -INT32_C(   206059641), -INT32_C(  1640869625),  INT32_C(   197723259), -INT32_C(  1934089821),  INT32_C(   738836867),  INT32_C(   371934869), -INT32_C(  2142224838), -INT32_C(  1471174184) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_min_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_epi32");
  easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_min_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(   122954917),  INT32_C(   657547680), -INT32_C(  1965187134),  INT32_C(  1812295177),  INT32_C(  1035689364), -INT32_C(  1614746190),  INT32_C(  1490439621), -INT32_C(  1382961144),
         INT32_C(  1958077908),  INT32_C(   161277511), -INT32_C(  1886160507),  INT32_C(  2080086247), -INT32_C(  1464224010),  INT32_C(  1766357411), -INT32_C(   826204474), -INT32_C(  1552133426) },
      UINT16_C(12600),
      { -INT32_C(  1290305769),  INT32_C(   455974024),  INT32_C(   649270315),  INT32_C(  1239198096), -INT32_C(  1698529198), -INT32_C(  1430746903), -INT32_C(   738425001),  INT32_C(  1090794538),
         INT32_C(  1005919155), -INT32_C(   480894281), -INT32_C(   972486090),  INT32_C(    84928179),  INT32_C(  1335873894), -INT32_C(  1309058982),  INT32_C(   159708639), -INT32_C(   582318038) },
      {  INT32_C(  1511604131), -INT32_C(  1740804255),  INT32_C(   761153401), -INT32_C(  1825411539), -INT32_C(  1713188289),  INT32_C(   155900714), -INT32_C(    99430704), -INT32_C(   103261098),
        -INT32_C(    44764772), -INT32_C(   627732128),  INT32_C(    84407512), -INT32_C(  1583859358),  INT32_C(   893090315),  INT32_C(   641631573), -INT32_C(  1440722860),  INT32_C(  1235548333) },
      {  INT32_C(   122954917),  INT32_C(   657547680), -INT32_C(  1965187134), -INT32_C(  1825411539), -INT32_C(  1713188289), -INT32_C(  1430746903),  INT32_C(  1490439621), -INT32_C(  1382961144),
        -INT32_C(    44764772),  INT32_C(   161277511), -INT32_C(  1886160507),  INT32_C(  2080086247),  INT32_C(   893090315), -INT32_C(  1309058982), -INT32_C(   826204474), -INT32_C(  1552133426) } },
    { {  INT32_C(  1246165225),  INT32_C(  1646582921),  INT32_C(   845622224),  INT32_C(  1892876388), -INT32_C(   794489222), -INT32_C(   386472812), -INT32_C(   510519756), -INT32_C(   131451377),
        -INT32_C(  1203605202),  INT32_C(   488269389), -INT32_C(   162561647), -INT32_C(    60415359), -INT32_C(   976483535),  INT32_C(   615367407), -INT32_C(   419086376), -INT32_C(  1528811402) },
      UINT16_C( 8865),
      {  INT32_C(  1988750940),  INT32_C(  1526209035),  INT32_C(  1987868944),  INT32_C(  1099083125),  INT32_C(   520319346), -INT32_C(  1688216427), -INT32_C(  1546922557), -INT32_C(   691639175),
         INT32_C(  1699499866),  INT32_C(  2059355241),  INT32_C(   871381950),  INT32_C(  1534423785),  INT32_C(  2038135012),  INT32_C(   404019796),  INT32_C(   683401135), -INT32_C(  1510047413) },
      {  INT32_C(   973753296),  INT32_C(  1303693711), -INT32_C(   293493755), -INT32_C(    79038953), -INT32_C(  1015757714),  INT32_C(  1306233246), -INT32_C(  1267296664), -INT32_C(   396790760),
         INT32_C(  1327654079),  INT32_C(   849139245), -INT32_C(  1843389061), -INT32_C(  2104661485), -INT32_C(   851115730), -INT32_C(   216391542), -INT32_C(   827879242), -INT32_C(   994639867) },
      {  INT32_C(   973753296),  INT32_C(  1646582921),  INT32_C(   845622224),  INT32_C(  1892876388), -INT32_C(   794489222), -INT32_C(  1688216427), -INT32_C(   510519756), -INT32_C(   691639175),
        -INT32_C(  1203605202),  INT32_C(   849139245), -INT32_C(   162561647), -INT32_C(    60415359), -INT32_C(   976483535), -INT32_C(   216391542), -INT32_C(   419086376), -INT32_C(  1528811402) } },
    { { -INT32_C(  1844192924),  INT32_C(   734310576), -INT32_C(   507648563),  INT32_C(  2103659087), -INT32_C(   699750325),  INT32_C(  2127128008), -INT32_C(    95588107), -INT32_C(   708901776),
        -INT32_C(  1922575651),  INT32_C(  1354247042),  INT32_C(  1597076752),  INT32_C(   199070911),  INT32_C(    81864508), -INT32_C(  2105365876), -INT32_C(  1971532006), -INT32_C(  1319158829) },
      UINT16_C(50702),
      { -INT32_C(   135098306),  INT32_C(   292291296),  INT32_C(  1067789410),  INT32_C(   409395511),  INT32_C(  1757606885), -INT32_C(   247997323), -INT32_C(   970126490), -INT32_C(    91472964),
        -INT32_C(  1393459509), -INT32_C(   474128767), -INT32_C(  1054710902),  INT32_C(   718899268), -INT32_C(   258827397), -INT32_C(   572339849), -INT32_C(  1868361772),  INT32_C(   361443402) },
      {  INT32_C(   801209518),  INT32_C(  1678933978),  INT32_C(   639972578),  INT32_C(   944832189),  INT32_C(   271180441), -INT32_C(  2131883092),  INT32_C(  1678872858),  INT32_C(  1887018177),
        -INT32_C(   207668456), -INT32_C(  1671974214), -INT32_C(  1547534874),  INT32_C(   366744443), -INT32_C(  1591409163),  INT32_C(   706876176),  INT32_C(  1720595365), -INT32_C(   405403697) },
      { -INT32_C(  1844192924),  INT32_C(   292291296),  INT32_C(   639972578),  INT32_C(   409395511), -INT32_C(   699750325),  INT32_C(  2127128008), -INT32_C(    95588107), -INT32_C(   708901776),
        -INT32_C(  1922575651), -INT32_C(  1671974214), -INT32_C(  1547534874),  INT32_C(   199070911),  INT32_C(    81864508), -INT32_C(  2105365876), -INT32_C(  1868361772), -INT32_C(   405403697) } },
    { { -INT32_C(    52791742),  INT32_C(   244855336),  INT32_C(   716331951),  INT32_C(  1665109614), -INT32_C(  1559927405),  INT32_C(   499984248),  INT32_C(   696539994), -INT32_C(  1525654942),
        -INT32_C(   123606064), -INT32_C(   871941603),  INT32_C(    66501013), -INT32_C(   630835641),  INT32_C(   326986651), -INT32_C(   332313966),  INT32_C(   118863269), -INT32_C(   525588977) },
      UINT16_C(19985),
      { -INT32_C(   544723240), -INT32_C(   258466310), -INT32_C(  2044272864),  INT32_C(   921878969),  INT32_C(   109085909), -INT32_C(  2034555535), -INT32_C(   642987475), -INT32_C(  2094547542),
        -INT32_C(   446517269), -INT32_C(   321455156), -INT32_C(  1804337958), -INT32_C(  1815452226),  INT32_C(  1519995881), -INT32_C(  1646177168), -INT32_C(   914911970),  INT32_C(   910990923) },
      {  INT32_C(   438087246), -INT32_C(  2079853911), -INT32_C(  1407681810), -INT32_C(   935337249), -INT32_C(  1641818067),  INT32_C(  1262158892), -INT32_C(   602623343), -INT32_C(  1626120111),
        -INT32_C(  1195823346),  INT32_C(   255639585),  INT32_C(   431772730),  INT32_C(  1692597046),  INT32_C(      132564), -INT32_C(  1706345207),  INT32_C(  1098342384), -INT32_C(   824145217) },
      { -INT32_C(   544723240),  INT32_C(   244855336),  INT32_C(   716331951),  INT32_C(  1665109614), -INT32_C(  1641818067),  INT32_C(   499984248),  INT32_C(   696539994), -INT32_C(  1525654942),
        -INT32_C(   123606064), -INT32_C(   321455156), -INT32_C(  1804337958), -INT32_C(  1815452226),  INT32_C(   326986651), -INT32_C(   332313966), -INT32_C(   914911970), -INT32_C(   525588977) } },
    { { -INT32_C(   628713031), -INT32_C(  1796619686),  INT32_C(  1286513942),  INT32_C(  1974505633), -INT32_C(  1636453739), -INT32_C(   533151248), -INT32_C(   534663392), -INT32_C(   223477447),
        -INT32_C(   187943782),  INT32_C(   210351862), -INT32_C(    61327525),  INT32_C(  1550911943), -INT32_C(  1409620037), -INT32_C(   930401624),  INT32_C(   464039138),  INT32_C(  1208899245) },
      UINT16_C(55946),
      { -INT32_C(   980385732), -INT32_C(   436409204),  INT32_C(  1525597160),  INT32_C(   423733535), -INT32_C(   531830443),  INT32_C(  1519201969),  INT32_C(  1471167049), -INT32_C(  1087227006),
         INT32_C(  1216660155), -INT32_C(  1758625362),  INT32_C(  1693522756),  INT32_C(   427635396), -INT32_C(   855979749),  INT32_C(  1093044215),  INT32_C(  1150867393), -INT32_C(   855389678) },
      {  INT32_C(   991266701),  INT32_C(  1305625096),  INT32_C(   582075229),  INT32_C(   272314101), -INT32_C(   270715400),  INT32_C(  2117075900), -INT32_C(   322778662),  INT32_C(   549111187),
         INT32_C(  1432145740),  INT32_C(  1872899602), -INT32_C(   409906190),  INT32_C(  2046348673), -INT32_C(  1083583230), -INT32_C(  1287808552), -INT32_C(   157286558),  INT32_C(   269900228) },
      { -INT32_C(   628713031), -INT32_C(   436409204),  INT32_C(  1286513942),  INT32_C(   272314101), -INT32_C(  1636453739), -INT32_C(   533151248), -INT32_C(   534663392), -INT32_C(  1087227006),
        -INT32_C(   187943782), -INT32_C(  1758625362), -INT32_C(    61327525),  INT32_C(   427635396), -INT32_C(  1083583230), -INT32_C(   930401624), -INT32_C(   157286558), -INT32_C(   855389678) } },
    { {  INT32_C(   996504105), -INT32_C(  1817573471), -INT32_C(   595968934),  INT32_C(   190149129),  INT32_C(   550157895), -INT32_C(  1160575144), -INT32_C(   894406138), -INT32_C(   170145844),
        -INT32_C(   634372039), -INT32_C(  1569859000),  INT32_C(   528410646), -INT32_C(  1574185894), -INT32_C(   356321902),  INT32_C(    61183485), -INT32_C(   741452537),  INT32_C(  1455991068) },
      UINT16_C(63721),
      { -INT32_C(  1630326480),  INT32_C(  1384573396),  INT32_C(   874962953),  INT32_C(  1143585154),  INT32_C(  1222190755), -INT32_C(   123805398),  INT32_C(  2107751092),  INT32_C(  1098222096),
        -INT32_C(  1864415044),  INT32_C(  1004692786), -INT32_C(   932247227),  INT32_C(  1695324354), -INT32_C(   374413633),  INT32_C(  2078362823),  INT32_C(   402162182), -INT32_C(   933728756) },
      { -INT32_C(   396871754), -INT32_C(   517784676),  INT32_C(   111776324), -INT32_C(   362039765),  INT32_C(  1674779036),  INT32_C(  1826534501),  INT32_C(  1115936566), -INT32_C(    83109051),
        -INT32_C(  1360764142), -INT32_C(   493942882), -INT32_C(   974636646), -INT32_C(  1951443729), -INT32_C(   739343763),  INT32_C(  1816120374), -INT32_C(   391200093), -INT32_C(  1343964771) },
      { -INT32_C(  1630326480), -INT32_C(  1817573471), -INT32_C(   595968934), -INT32_C(   362039765),  INT32_C(   550157895), -INT32_C(   123805398),  INT32_C(  1115936566), -INT32_C(    83109051),
        -INT32_C(   634372039), -INT32_C(  1569859000),  INT32_C(   528410646), -INT32_C(  1951443729), -INT32_C(   739343763),  INT32_C(  1816120374), -INT32_C(   391200093), -INT32_C(  1343964771) } },
    { { -INT32_C(  1151481827),  INT32_C(  1772022991),  INT32_C(   338593317),  INT32_C(  1218436570), -INT32_C(  1793356449), -INT32_C(    50242982), -INT32_C(  1176063972), -INT32_C(  2039952791),
         INT32_C(  1631765906), -INT32_C(   674504527),  INT32_C(  1105983846),  INT32_C(   914983895),  INT32_C(  1926013976),  INT32_C(   443600382), -INT32_C(   405580163), -INT32_C(  1301464288) },
      UINT16_C(44800),
      { -INT32_C(   561008365),  INT32_C(  1960375944), -INT32_C(  1056985289), -INT32_C(  1318840347), -INT32_C(    92314998),  INT32_C(  1330707580),  INT32_C(  1368027363),  INT32_C(   889227810),
        -INT32_C(  1005285317), -INT32_C(  1120342906), -INT32_C(  2139277413), -INT32_C(   667754162), -INT32_C(  1076711101), -INT32_C(  1861344595), -INT32_C(  1260218222),  INT32_C(  1575674402) },
      { -INT32_C(   131989902), -INT32_C(  2051712534), -INT32_C(   553307504),  INT32_C(  1454847763), -INT32_C(  1776973080),  INT32_C(  1059529644), -INT32_C(   571274821),  INT32_C(  1580981739),
        -INT32_C(   984195877),  INT32_C(  1179258038),  INT32_C(  1378177086),  INT32_C(  1890114951),  INT32_C(   319209063), -INT32_C(  1655558687),  INT32_C(   561661494), -INT32_C(     8407773) },
      { -INT32_C(  1151481827),  INT32_C(  1772022991),  INT32_C(   338593317),  INT32_C(  1218436570), -INT32_C(  1793356449), -INT32_C(    50242982), -INT32_C(  1176063972), -INT32_C(  2039952791),
        -INT32_C(  1005285317), -INT32_C(  1120342906), -INT32_C(  2139277413), -INT32_C(   667754162),  INT32_C(  1926013976), -INT32_C(  1861344595), -INT32_C(   405580163), -INT32_C(     8407773) } },
    { { -INT32_C(   926624238),  INT32_C(   537792482), -INT32_C(   428723105),  INT32_C(  2018908945), -INT32_C(  1165271847), -INT32_C(  1084760439),  INT32_C(  1205981732), -INT32_C(  1723441017),
         INT32_C(   409013046),  INT32_C(  2033807386), -INT32_C(  1252021340), -INT32_C(  1624394042), -INT32_C(  1671776238), -INT32_C(  1168395882),  INT32_C(   184695939), -INT32_C(   744208227) },
      UINT16_C( 1619),
      {  INT32_C(   628518380),  INT32_C(  1188043494), -INT32_C(    50554929), -INT32_C(  1867248074),  INT32_C(   105008042),  INT32_C(   121816325),  INT32_C(  1968234448), -INT32_C(  1619287117),
        -INT32_C(   138088175), -INT32_C(   633498613),  INT32_C(  1658206507), -INT32_C(   218985912), -INT32_C(   604490539), -INT32_C(   891143174), -INT32_C(   851496422), -INT32_C(   412304682) },
      { -INT32_C(  1226952533), -INT32_C(   258990907), -INT32_C(  1655544235), -INT32_C(   963623439),  INT32_C(  1939966073), -INT32_C(   566328125), -INT32_C(  1934918218), -INT32_C(   478996424),
         INT32_C(   228217416), -INT32_C(  1006753170), -INT32_C(  2107551599),  INT32_C(   256438677),  INT32_C(  1031989881),  INT32_C(   605798510),  INT32_C(  1991362110),  INT32_C(   660153566) },
      { -INT32_C(  1226952533), -INT32_C(   258990907), -INT32_C(   428723105),  INT32_C(  2018908945),  INT32_C(   105008042), -INT32_C(  1084760439), -INT32_C(  1934918218), -INT32_C(  1723441017),
         INT32_C(   409013046), -INT32_C(  1006753170), -INT32_C(  2107551599), -INT32_C(  1624394042), -INT32_C(  1671776238), -INT32_C(  1168395882),  INT32_C(   184695939), -INT32_C(   744208227) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(20760),
      {  INT32_C(   213830387), -INT32_C(     9123383),  INT32_C(   845115943),  INT32_C(     8198242), -INT32_C(    76107379), -INT32_C(   862611029),  INT32_C(  1709952286),  INT32_C(  1102447438),
        -INT32_C(  1773243187),  INT32_C(  1704313405), -INT32_C(  1768426444), -INT32_C(  1701374707),  INT32_C(  1905593798), -INT32_C(   985781337),  INT32_C(  1177168376), -INT32_C(   108470228) },
      { -INT32_C(  1852844460), -INT32_C(   856283752),  INT32_C(   643993113),  INT32_C(  1774254499), -INT32_C(  1361422841),  INT32_C(  2037586049),  INT32_C(  1841274177), -INT32_C(   781826179),
        -INT32_C(  1251805667),  INT32_C(   880892187), -INT32_C(  1973689113), -INT32_C(   453829667), -INT32_C(   225260175),  INT32_C(   661325286),  INT32_C(   529869730), -INT32_C(  1863255182) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(     8198242), -INT32_C(  1361422841),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(  1773243187),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   225260175),  INT32_C(           0),  INT32_C(   529869730),  INT32_C(           0) } },
    { UINT16_C(21489),
      { -INT32_C(   945025979), -INT32_C(  1666477247),  INT32_C(   280463389), -INT32_C(     2283155), -INT32_C(  2029665509), -INT32_C(  2135775253), -INT32_C(  1216666425), -INT32_C(    99979852),
        -INT32_C(  1161709959),  INT32_C(  1716939849),  INT32_C(  1635127028),  INT32_C(  1382110263),  INT32_C(    47801879), -INT32_C(   729642227),  INT32_C(  1686961840), -INT32_C(   463563157) },
      { -INT32_C(  1767956659), -INT32_C(  2130840181),  INT32_C(   987853571), -INT32_C(   544390457), -INT32_C(  1226742104), -INT32_C(  1567988494), -INT32_C(   855239070), -INT32_C(    89037395),
         INT32_C(   277893252),  INT32_C(  1234210118), -INT32_C(   930844415),  INT32_C(  1554452916),  INT32_C(  1762822519),  INT32_C(  1326161389),  INT32_C(  1612452531), -INT32_C(    77935241) },
      { -INT32_C(  1767956659),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2029665509), -INT32_C(  2135775253), -INT32_C(  1216666425), -INT32_C(    99979852),
        -INT32_C(  1161709959),  INT32_C(  1234210118),  INT32_C(           0),  INT32_C(           0),  INT32_C(    47801879),  INT32_C(           0),  INT32_C(  1612452531),  INT32_C(           0) } },
    { UINT16_C(60190),
      { -INT32_C(  1686608885),  INT32_C(   822966701), -INT32_C(   381501118),  INT32_C(   812825117),  INT32_C(   801988387), -INT32_C(   901676882),  INT32_C(   999864545), -INT32_C(  1087981901),
        -INT32_C(   950362342),  INT32_C(  1526294296),  INT32_C(  1178876712),  INT32_C(   427210485), -INT32_C(  1001897194), -INT32_C(  1534096957), -INT32_C(   186636479), -INT32_C(   139262243) },
      { -INT32_C(  1094840667), -INT32_C(  1642547339), -INT32_C(   387687181),  INT32_C(   687954451),  INT32_C(  1626163613),  INT32_C(   319126738), -INT32_C(  2146900573), -INT32_C(  1854424085),
         INT32_C(  1062155977), -INT32_C(   522360851), -INT32_C(   674643516), -INT32_C(  1207907813), -INT32_C(   434574060),  INT32_C(   234495338), -INT32_C(   292683262), -INT32_C(  2021718595) },
      {  INT32_C(           0), -INT32_C(  1642547339), -INT32_C(   387687181),  INT32_C(   687954451),  INT32_C(   801988387),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(   950362342), -INT32_C(   522360851),  INT32_C(           0), -INT32_C(  1207907813),  INT32_C(           0), -INT32_C(  1534096957), -INT32_C(   292683262), -INT32_C(  2021718595) } },
    { UINT16_C(52795),
      { -INT32_C(  1556796986), -INT32_C(   765134583), -INT32_C(   794984496),  INT32_C(  1337897271), -INT32_C(  1855117161), -INT32_C(  1013747915), -INT32_C(   590786211), -INT32_C(  1633024808),
         INT32_C(   927064109), -INT32_C(  1442208295),  INT32_C(  1534764580), -INT32_C(   274057129), -INT32_C(  1769990304), -INT32_C(   463924089),  INT32_C(  1036067429),  INT32_C(  1423665959) },
      {  INT32_C(   629873739), -INT32_C(   439380543), -INT32_C(  1824503493), -INT32_C(   494736766), -INT32_C(  1988623870),  INT32_C(  2070794774),  INT32_C(   465055476), -INT32_C(   445607014),
         INT32_C(  1879767983), -INT32_C(   866788976), -INT32_C(  1520462557), -INT32_C(  2105024128), -INT32_C(    99942173), -INT32_C(   965379886),  INT32_C(  1105342119),  INT32_C(  1898336961) },
      { -INT32_C(  1556796986), -INT32_C(   765134583),  INT32_C(           0), -INT32_C(   494736766), -INT32_C(  1988623870), -INT32_C(  1013747915),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0), -INT32_C(  1442208295), -INT32_C(  1520462557), -INT32_C(  2105024128),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1036067429),  INT32_C(  1423665959) } },
    { UINT16_C(12621),
      {  INT32_C(   923459297),  INT32_C(   164375978),  INT32_C(  1525304530), -INT32_C(   648360498),  INT32_C(  1028795591), -INT32_C(   731121166),  INT32_C(  1613114426),  INT32_C(  2140239005),
        -INT32_C(    55141294), -INT32_C(  1677360439), -INT32_C(  1644761137), -INT32_C(  2072555332),  INT32_C(  1858193788), -INT32_C(    62706494), -INT32_C(   161715880),  INT32_C(   796258013) },
      {  INT32_C(  1395338122),  INT32_C(  2096050349), -INT32_C(   602217185), -INT32_C(  1319071435),  INT32_C(   471867738), -INT32_C(  1525128371), -INT32_C(  1432652596), -INT32_C(   321318814),
        -INT32_C(   381680325),  INT32_C(  1432694581),  INT32_C(  1244757781),  INT32_C(  1794937104), -INT32_C(     7988046),  INT32_C(  1269079679),  INT32_C(  1979006995), -INT32_C(  1939681456) },
      {  INT32_C(   923459297),  INT32_C(           0), -INT32_C(   602217185), -INT32_C(  1319071435),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1432652596),  INT32_C(           0),
        -INT32_C(   381680325),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(     7988046), -INT32_C(    62706494),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(41684),
      { -INT32_C(   623834763), -INT32_C(  1889868194),  INT32_C(   757099057),  INT32_C(  1531564757),  INT32_C(  1996146897),  INT32_C(   162925843), -INT32_C(   455604606),  INT32_C(   126266514),
         INT32_C(   350378165),  INT32_C(  1872968766),  INT32_C(  2073871526),  INT32_C(  1758979478), -INT32_C(  1042361939),  INT32_C(  1623889118),  INT32_C(   759538330),  INT32_C(    53791566) },
      {  INT32_C(  1645680163), -INT32_C(   103695534),  INT32_C(   359951999),  INT32_C(    24988499), -INT32_C(    87925988), -INT32_C(  1973711633),  INT32_C(  2092408878), -INT32_C(  1887442069),
         INT32_C(  1425118978), -INT32_C(   783433134), -INT32_C(  2065251792),  INT32_C(   713384973), -INT32_C(  1356576833),  INT32_C(    20545491), -INT32_C(  1954680801), -INT32_C(   585433893) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   359951999),  INT32_C(           0), -INT32_C(    87925988),  INT32_C(           0), -INT32_C(   455604606), -INT32_C(  1887442069),
         INT32_C(           0), -INT32_C(   783433134),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(    20545491),  INT32_C(           0), -INT32_C(   585433893) } },
    { UINT16_C( 2963),
      {  INT32_C(  2144199986), -INT32_C(  1656619338),  INT32_C(   117526402), -INT32_C(  1655783303), -INT32_C(  1474485905), -INT32_C(  1617413086), -INT32_C(   509905721), -INT32_C(  2081673391),
        -INT32_C(   888948204),  INT32_C(   979911864),  INT32_C(   205613459), -INT32_C(  1716875479), -INT32_C(   767440976), -INT32_C(   881731069), -INT32_C(  1616114610),  INT32_C(  1344510267) },
      {  INT32_C(   186328659), -INT32_C(    28998806),  INT32_C(   353011436), -INT32_C(   978406379),  INT32_C(  2140663931),  INT32_C(   407505098),  INT32_C(  1354233364), -INT32_C(   492774769),
         INT32_C(  1810742016),  INT32_C(   711537214), -INT32_C(   851479624), -INT32_C(  1550651864), -INT32_C(  1474156066),  INT32_C(  1187015729), -INT32_C(   225020061),  INT32_C(  1389704786) },
      {  INT32_C(   186328659), -INT32_C(  1656619338),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1474485905),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2081673391),
        -INT32_C(   888948204),  INT32_C(   711537214),  INT32_C(           0), -INT32_C(  1716875479),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(49905),
      {  INT32_C(   653602749), -INT32_C(  1734693543),  INT32_C(   226935419), -INT32_C(  2009635739), -INT32_C(   856397812),  INT32_C(  1145329582),  INT32_C(   528127562), -INT32_C(  1495176216),
        -INT32_C(   204679526), -INT32_C(     7575932), -INT32_C(  1911811544), -INT32_C(  2095692937),  INT32_C(  1515195052), -INT32_C(  1398827934),  INT32_C(   315300138),  INT32_C(   532196485) },
      {  INT32_C(   118719875),  INT32_C(   335978475),  INT32_C(   681710257),  INT32_C(    44873814),  INT32_C(   610073794), -INT32_C(  1160709232), -INT32_C(  1697866987), -INT32_C(   877034168),
        -INT32_C(   170734582),  INT32_C(   487184491),  INT32_C(  1078307818),  INT32_C(   641921379),  INT32_C(  2102042605), -INT32_C(  1355342950), -INT32_C(    28769098), -INT32_C(  1815542903) },
      {  INT32_C(   118719875),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   856397812), -INT32_C(  1160709232), -INT32_C(  1697866987), -INT32_C(  1495176216),
         INT32_C(           0), -INT32_C(     7575932),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(    28769098), -INT32_C(  1815542903) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_min_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint32_t a[16];
    const uint32_t b[16];
    const uint32_t r[16];
  } test_vec[] = {
    { { UINT32_C(1289220019), UINT32_C(2127009800), UINT32_C(3136822521), UINT32_C(1627778699), UINT32_C(2140200231), UINT32_C(3389450485), UINT32_C( 107671816), UINT32_C( 779174523),
        UINT32_C( 981092401), UINT32_C(3669508833), UINT32_C(3918835806), UINT32_C(3326843295), UINT32_C(1799740278), UINT32_C(3409267651), UINT32_C(3100745788), UINT32_C( 350634722) },
      { UINT32_C(1817076107), UINT32_C(  21366435), UINT32_C(1458297527), UINT32_C(3927717492), UINT32_C(3562430737), UINT32_C(3919612844), UINT32_C( 245461291), UINT32_C(1059227572),
        UINT32_C(2343268584), UINT32_C( 764277110), UINT32_C(1065580747), UINT32_C(3207241646), UINT32_C(2912124928), UINT32_C( 932590347), UINT32_C(1497708453), UINT32_C(2811783102) },
      { UINT32_C(1289220019), UINT32_C(  21366435), UINT32_C(1458297527), UINT32_C(1627778699), UINT32_C(2140200231), UINT32_C(3389450485), UINT32_C( 107671816), UINT32_C( 779174523),
        UINT32_C( 981092401), UINT32_C( 764277110), UINT32_C(1065580747), UINT32_C(3207241646), UINT32_C(1799740278), UINT32_C( 932590347), UINT32_C(1497708453), UINT32_C( 350634722) } },
    { { UINT32_C(1295139799), UINT32_C(4286299956), UINT32_C(3846176311), UINT32_C(2661575070), UINT32_C(4098570473), UINT32_C( 271311211), UINT32_C(3614011416), UINT32_C(2927493591),
        UINT32_C(2029826116), UINT32_C(2809689968), UINT32_C( 328054645), UINT32_C( 162672928), UINT32_C(3590192489), UINT32_C(4158990815), UINT32_C(1909346201), UINT32_C(2501856336) },
      { UINT32_C(1829575677), UINT32_C( 135562642), UINT32_C(1545314620), UINT32_C(1013304787), UINT32_C(2853266379), UINT32_C( 631371660), UINT32_C(2526441542), UINT32_C(3106649788),
        UINT32_C(1680226769), UINT32_C(4218174398), UINT32_C(2941749212), UINT32_C( 552385877), UINT32_C(2898984224), UINT32_C( 986803188), UINT32_C(2563860699), UINT32_C(4015127582) },
      { UINT32_C(1295139799), UINT32_C( 135562642), UINT32_C(1545314620), UINT32_C(1013304787), UINT32_C(2853266379), UINT32_C( 271311211), UINT32_C(2526441542), UINT32_C(2927493591),
        UINT32_C(1680226769), UINT32_C(2809689968), UINT32_C( 328054645), UINT32_C( 162672928), UINT32_C(2898984224), UINT32_C( 986803188), UINT32_C(1909346201), UINT32_C(2501856336) } },
    { { UINT32_C(4099110965), UINT32_C(2414854067), UINT32_C(2621392455), UINT32_C( 599534339), UINT32_C( 500139560), UINT32_C(3445072369), UINT32_C( 660940809), UINT32_C(1511437861),
        UINT32_C(3780012590), UINT32_C(1886469417), UINT32_C(2265755780), UINT32_C(  61589723), UINT32_C(1075870286), UINT32_C( 604862491), UINT32_C(3310056096), UINT32_C(1461740072) },
      { UINT32_C(4114116300), UINT32_C( 812034476), UINT32_C( 884437593), UINT32_C(2302173755), UINT32_C(4173945053), UINT32_C(1897780944), UINT32_C(1899391048), UINT32_C(2529711818),
        UINT32_C(1905000645), UINT32_C(  60945066), UINT32_C(2671269988), UINT32_C(2552852667), UINT32_C(2576413384), UINT32_C( 285912521), UINT32_C(3766632470), UINT32_C(1551321751) },
      { UINT32_C(4099110965), UINT32_C( 812034476), UINT32_C( 884437593), UINT32_C( 599534339), UINT32_C( 500139560), UINT32_C(1897780944), UINT32_C( 660940809), UINT32_C(1511437861),
        UINT32_C(1905000645), UINT32_C(  60945066), UINT32_C(2265755780), UINT32_C(  61589723), UINT32_C(1075870286), UINT32_C( 285912521), UINT32_C(3310056096), UINT32_C(1461740072) } },
    { { UINT32_C(4123853643), UINT32_C(1509453557), UINT32_C(2180591814), UINT32_C(1763254944), UINT32_C(3707939348), UINT32_C(1844382807), UINT32_C(3813568844), UINT32_C( 121619900),
        UINT32_C(3187412168), UINT32_C(1092023418), UINT32_C(3317829413), UINT32_C(1177476145), UINT32_C(3710070918), UINT32_C(2303398460), UINT32_C(1080859012), UINT32_C( 642231390) },
      { UINT32_C( 870532024), UINT32_C(1551169847), UINT32_C( 975320585), UINT32_C(2558545938), UINT32_C(3178669185), UINT32_C( 977715638), UINT32_C(3095049050), UINT32_C( 400474463),
        UINT32_C(1011532036), UINT32_C(3281567418), UINT32_C( 134134517), UINT32_C(2359328267), UINT32_C(3645445666), UINT32_C( 823365847), UINT32_C(2733215299), UINT32_C(1421461327) },
      { UINT32_C( 870532024), UINT32_C(1509453557), UINT32_C( 975320585), UINT32_C(1763254944), UINT32_C(3178669185), UINT32_C( 977715638), UINT32_C(3095049050), UINT32_C( 121619900),
        UINT32_C(1011532036), UINT32_C(1092023418), UINT32_C( 134134517), UINT32_C(1177476145), UINT32_C(3645445666), UINT32_C( 823365847), UINT32_C(1080859012), UINT32_C( 642231390) } },
    { { UINT32_C(1116734600), UINT32_C(3070634178), UINT32_C(4005496035), UINT32_C(2776260482), UINT32_C(1283375989), UINT32_C(2524811603), UINT32_C(1865967135), UINT32_C(3049517613),
        UINT32_C(3103216630), UINT32_C(1584463227), UINT32_C(2219585281), UINT32_C(  53069454), UINT32_C(3712984970), UINT32_C(1484049464), UINT32_C(1606921266), UINT32_C(2484374174) },
      { UINT32_C(1481444317), UINT32_C( 179813641), UINT32_C(2056127468), UINT32_C(1417525194), UINT32_C(2536623198), UINT32_C(3404703128), UINT32_C(4029265490), UINT32_C( 495271232),
        UINT32_C(1366676040), UINT32_C(2069638287), UINT32_C(4210420272), UINT32_C(   5141154), UINT32_C(3600252734), UINT32_C(2007008805), UINT32_C(2087176508), UINT32_C(1318710278) },
      { UINT32_C(1116734600), UINT32_C( 179813641), UINT32_C(2056127468), UINT32_C(1417525194), UINT32_C(1283375989), UINT32_C(2524811603), UINT32_C(1865967135), UINT32_C( 495271232),
        UINT32_C(1366676040), UINT32_C(1584463227), UINT32_C(2219585281), UINT32_C(   5141154), UINT32_C(3600252734), UINT32_C(1484049464), UINT32_C(1606921266), UINT32_C(1318710278) } },
    { { UINT32_C(1302335422), UINT32_C(1808333883), UINT32_C(2288369126), UINT32_C(1837740847), UINT32_C(1480794163), UINT32_C(3822052263), UINT32_C(2992649900), UINT32_C(3775002915),
        UINT32_C(1143972104), UINT32_C(2209347485), UINT32_C(3825997237), UINT32_C(4216493512), UINT32_C(1548981685), UINT32_C( 624960121), UINT32_C(2094571609), UINT32_C(2724059545) },
      { UINT32_C( 400985210), UINT32_C( 966432132), UINT32_C(1931323050), UINT32_C(4050546491), UINT32_C(2119025157), UINT32_C(1034128868), UINT32_C(3350821677), UINT32_C(3462993748),
        UINT32_C( 669339555), UINT32_C(2405466340), UINT32_C(1644330534), UINT32_C(4065554669), UINT32_C( 393257010), UINT32_C(1532236846), UINT32_C(3827437199), UINT32_C(3367144229) },
      { UINT32_C( 400985210), UINT32_C( 966432132), UINT32_C(1931323050), UINT32_C(1837740847), UINT32_C(1480794163), UINT32_C(1034128868), UINT32_C(2992649900), UINT32_C(3462993748),
        UINT32_C( 669339555), UINT32_C(2209347485), UINT32_C(1644330534), UINT32_C(4065554669), UINT32_C( 393257010), UINT32_C( 624960121), UINT32_C(2094571609), UINT32_C(2724059545) } },
    { { UINT32_C(3220216026), UINT32_C(1045319704), UINT32_C(3164623054), UINT32_C(4088329152), UINT32_C(3255443348), UINT32_C(3256704563), UINT32_C(2443591788), UINT32_C(2790939083),
        UINT32_C( 157633265), UINT32_C(1766306714), UINT32_C(3274041347), UINT32_C(1874252763), UINT32_C( 624017650), UINT32_C(2347257631), UINT32_C(1511886479), UINT32_C(3623909351) },
      { UINT32_C(1541498305), UINT32_C( 465840408), UINT32_C(3974097169), UINT32_C(2942080445), UINT32_C(1976929622), UINT32_C(1795210716), UINT32_C( 868621643), UINT32_C(1426835092),
        UINT32_C(1152511276), UINT32_C( 660632854), UINT32_C( 471023455), UINT32_C( 717975508), UINT32_C(3651117309), UINT32_C(2839912541), UINT32_C(1390152637), UINT32_C(4255639505) },
      { UINT32_C(1541498305), UINT32_C( 465840408), UINT32_C(3164623054), UINT32_C(2942080445), UINT32_C(1976929622), UINT32_C(1795210716), UINT32_C( 868621643), UINT32_C(1426835092),
        UINT32_C( 157633265), UINT32_C( 660632854), UINT32_C( 471023455), UINT32_C( 717975508), UINT32_C( 624017650), UINT32_C(2347257631), UINT32_C(1390152637), UINT32_C(3623909351) } },
    { { UINT32_C(3930216660), UINT32_C( 756130510), UINT32_C(3041469921), UINT32_C(2447381652), UINT32_C( 309034933), UINT32_C(3720065055), UINT32_C(2351929275), UINT32_C(1401607807),
        UINT32_C(4248751151), UINT32_C(1328172910), UINT32_C( 151286644), UINT32_C(1016784007), UINT32_C(2202994020), UINT32_C(1885342389), UINT32_C( 570265506), UINT32_C(2507442022) },
      { UINT32_C(3247616595), UINT32_C(1980808194), UINT32_C(3061781551), UINT32_C(1576213241), UINT32_C(3588243999), UINT32_C(3997516108), UINT32_C( 906969808), UINT32_C( 483099849),
        UINT32_C( 954031414), UINT32_C(1219489049), UINT32_C(4227804674), UINT32_C(1750659656), UINT32_C(2151495732), UINT32_C(1248821881), UINT32_C(2390785733), UINT32_C( 967527426) },
      { UINT32_C(3247616595), UINT32_C( 756130510), UINT32_C(3041469921), UINT32_C(1576213241), UINT32_C( 309034933), UINT32_C(3720065055), UINT32_C( 906969808), UINT32_C( 483099849),
        UINT32_C( 954031414), UINT32_C(1219489049), UINT32_C( 151286644), UINT32_C(1016784007), UINT32_C(2151495732), UINT32_C(1248821881), UINT32_C( 570265506), UINT32_C( 967527426) } }
  };


    for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
      easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
      easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
      easysimd__m512i r;
      EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
        r = easysimd_mm512_min_epu32(a, b);
      } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_epu32");
      easysimd_test_x86_assert_equal_u32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
    }

  return 0;
}

static int
test_easysimd_mm512_mask_min_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const uint32_t a[16];
    const uint32_t b[16];
    const uint32_t r[16];
  } test_vec[] = {
    { {  INT32_C(  1219780957), -INT32_C(   361567476),  INT32_C(  1237692958),  INT32_C(  2064988168), -INT32_C(  1941892903),  INT32_C(  2002512175),  INT32_C(  1267491998),  INT32_C(   579990469),
         INT32_C(  1617642836),  INT32_C(  1313594672), -INT32_C(  1734930288),  INT32_C(   554937672), -INT32_C(   307342147), -INT32_C(   597423810),  INT32_C(   841478253),  INT32_C(   928364771) },
      UINT16_C(49405),
      { UINT32_C(3818728856), UINT32_C( 317926779), UINT32_C(3653188806), UINT32_C( 187596125), UINT32_C(3457510506), UINT32_C(1874756168), UINT32_C( 170369717), UINT32_C(1909073113),
        UINT32_C(3428083537), UINT32_C(1541294229), UINT32_C(3778322052), UINT32_C(2246926875), UINT32_C( 374538958), UINT32_C( 965022084), UINT32_C(2370022579), UINT32_C( 570297808) },
      { UINT32_C( 183325557), UINT32_C( 526765211), UINT32_C(2231409002), UINT32_C(3389713916), UINT32_C(1960926703), UINT32_C( 581789294), UINT32_C(3819958547), UINT32_C(1929752062),
        UINT32_C(2608722432), UINT32_C( 683336382), UINT32_C(2024651644), UINT32_C(2554509224), UINT32_C(2198610708), UINT32_C(2644883850), UINT32_C(2843759786), UINT32_C(  35423489) },
      { UINT32_C( 183325557), UINT32_C(3933399820), UINT32_C(2231409002), UINT32_C( 187596125), UINT32_C(1960926703), UINT32_C( 581789294), UINT32_C( 170369717), UINT32_C(1909073113),
        UINT32_C(1617642836), UINT32_C(1313594672), UINT32_C(2560037008), UINT32_C( 554937672), UINT32_C(3987625149), UINT32_C(3697543486), UINT32_C(2370022579), UINT32_C(  35423489) } },
    { {  INT32_C(   916298360), -INT32_C(   127969156), -INT32_C(  1150284781), -INT32_C(   665603132),  INT32_C(  1616601046), -INT32_C(  1006829543),  INT32_C(  1449950804),  INT32_C(  2069399811),
        -INT32_C(  1615661789),  INT32_C(  1620578637), -INT32_C(   518256611), -INT32_C(  1833275461), -INT32_C(   386787889),  INT32_C(  1789653014),  INT32_C(  1908414574), -INT32_C(   991094623) },
      UINT16_C(40718),
      { UINT32_C(4222638947), UINT32_C(3624193468), UINT32_C(1766309807), UINT32_C(1149113937), UINT32_C(2855572734), UINT32_C(3217203967), UINT32_C(  14181139), UINT32_C(2325734951),
        UINT32_C(4253437761), UINT32_C(3436546589), UINT32_C(2603949385), UINT32_C( 853521203), UINT32_C(1205605192), UINT32_C(3355877045), UINT32_C( 684318209), UINT32_C( 112355524) },
      { UINT32_C(3573758136), UINT32_C( 211933634), UINT32_C( 698865398), UINT32_C(3512436361), UINT32_C(1310209945), UINT32_C(3591774165), UINT32_C(3271483389), UINT32_C(  13152584),
        UINT32_C(2899692521), UINT32_C(2595780260), UINT32_C(3586416460), UINT32_C(2124881893), UINT32_C( 751615831), UINT32_C(3691242206), UINT32_C( 178127298), UINT32_C(2617927346) },
      { UINT32_C( 916298360), UINT32_C( 211933634), UINT32_C( 698865398), UINT32_C(1149113937), UINT32_C(1616601046), UINT32_C(3288137753), UINT32_C(1449950804), UINT32_C(2069399811),
        UINT32_C(2899692521), UINT32_C(2595780260), UINT32_C(2603949385), UINT32_C( 853521203), UINT32_C( 751615831), UINT32_C(1789653014), UINT32_C(1908414574), UINT32_C( 112355524) } },
    { { -INT32_C(   699867343), -INT32_C(  1586495403),  INT32_C(  1148597343), -INT32_C(  1413341868), -INT32_C(  1143501091),  INT32_C(   848812656), -INT32_C(  1908656676), -INT32_C(   852867429),
         INT32_C(  2057531941), -INT32_C(   786754702), -INT32_C(  1676307896), -INT32_C(  1941448785), -INT32_C(   699916699), -INT32_C(   720838663), -INT32_C(  1335671531), -INT32_C(  1317171573) },
      UINT16_C( 8192),
      { UINT32_C(1194619691), UINT32_C(1524202564), UINT32_C(1597081624), UINT32_C(1568511765), UINT32_C(1966896749), UINT32_C(2948223307), UINT32_C(2134722050), UINT32_C( 580926967),
        UINT32_C(4117353648), UINT32_C(1750024784), UINT32_C(3771171019), UINT32_C(2218607639), UINT32_C( 117078459), UINT32_C(3451237579), UINT32_C(4048351994), UINT32_C(3759467568) },
      { UINT32_C( 282426816), UINT32_C(2339906752), UINT32_C(3161145253), UINT32_C(1061267588), UINT32_C(3963960097), UINT32_C(3938057199), UINT32_C( 500893421), UINT32_C(3019829234),
        UINT32_C( 767808365), UINT32_C(2646097144), UINT32_C(4284031867), UINT32_C(3963525835), UINT32_C(3319366869), UINT32_C(1823445631), UINT32_C(2341112472), UINT32_C(3879635066) },
      { UINT32_C(3595099953), UINT32_C(2708471893), UINT32_C(1148597343), UINT32_C(2881625428), UINT32_C(3151466205), UINT32_C( 848812656), UINT32_C(2386310620), UINT32_C(3442099867),
        UINT32_C(2057531941), UINT32_C(3508212594), UINT32_C(2618659400), UINT32_C(2353518511), UINT32_C(3595050597), UINT32_C(1823445631), UINT32_C(2959295765), UINT32_C(2977795723) } },
    { {  INT32_C(  1393819995), -INT32_C(  1175401411), -INT32_C(  1162327313), -INT32_C(  1163462684), -INT32_C(    92307589), -INT32_C(  1436144110),  INT32_C(   842395832), -INT32_C(   736529544),
        -INT32_C(  1306055307), -INT32_C(   395634439),  INT32_C(  1185031266), -INT32_C(  1778366181), -INT32_C(   628064312),  INT32_C(  1720055469),  INT32_C(  1603844839), -INT32_C(  1556893138) },
      UINT16_C(23519),
      { UINT32_C(3245594965), UINT32_C(1692784065), UINT32_C( 481099803), UINT32_C( 647722390), UINT32_C(3575400784), UINT32_C(1200554927), UINT32_C(2532949347), UINT32_C(3069303136),
        UINT32_C(1937204402), UINT32_C(1440177209), UINT32_C(4067525724), UINT32_C(1243090170), UINT32_C(  69153877), UINT32_C(2605493816), UINT32_C(3425781100), UINT32_C(3498189598) },
      { UINT32_C(3242523015), UINT32_C(2971016021), UINT32_C(2594408352), UINT32_C(3924081555), UINT32_C( 686621680), UINT32_C( 499333553), UINT32_C(2649420927), UINT32_C(2674813975),
        UINT32_C(3143676518), UINT32_C(1835890381), UINT32_C(2416382205), UINT32_C(3162106828), UINT32_C(2699323374), UINT32_C( 532522912), UINT32_C(3015550875), UINT32_C(2052205332) },
      { UINT32_C(3242523015), UINT32_C(1692784065), UINT32_C( 481099803), UINT32_C( 647722390), UINT32_C( 686621680), UINT32_C(2858823186), UINT32_C(2532949347), UINT32_C(2674813975),
        UINT32_C(1937204402), UINT32_C(1440177209), UINT32_C(1185031266), UINT32_C(1243090170), UINT32_C(  69153877), UINT32_C(1720055469), UINT32_C(3015550875), UINT32_C(2738074158) } },
    { { -INT32_C(  1439321379),  INT32_C(   622371368),  INT32_C(  2142576563), -INT32_C(   113561845),  INT32_C(   916004758),  INT32_C(  1633048518),  INT32_C(   303305726),  INT32_C(   462186046),
         INT32_C(  1086702104),  INT32_C(   392551780),  INT32_C(   144055293), -INT32_C(   536751798), -INT32_C(  1240032272), -INT32_C(   266834702), -INT32_C(  1123865473), -INT32_C(  1411870829) },
      UINT16_C(40529),
      { UINT32_C(1367062252), UINT32_C(1684830413), UINT32_C(2184558208), UINT32_C(2904368790), UINT32_C(4095283164), UINT32_C(  35756543), UINT32_C( 798143574), UINT32_C(1271784287),
        UINT32_C(1738360985), UINT32_C(1103825345), UINT32_C(1455620288), UINT32_C(  50585638), UINT32_C(4025949679), UINT32_C( 217127094), UINT32_C( 742097868), UINT32_C(   7800935) },
      { UINT32_C( 308745297), UINT32_C(3729994270), UINT32_C(1496586035), UINT32_C(3881580791), UINT32_C( 198595669), UINT32_C( 957859692), UINT32_C(2992984907), UINT32_C(2897402971),
        UINT32_C( 264116977), UINT32_C(2146243148), UINT32_C( 551100713), UINT32_C(2919707993), UINT32_C(4139376009), UINT32_C(4029665701), UINT32_C(2141361188), UINT32_C(1630295152) },
      { UINT32_C( 308745297), UINT32_C( 622371368), UINT32_C(2142576563), UINT32_C(4181405451), UINT32_C( 198595669), UINT32_C(1633048518), UINT32_C( 798143574), UINT32_C( 462186046),
        UINT32_C(1086702104), UINT32_C(1103825345), UINT32_C( 551100713), UINT32_C(  50585638), UINT32_C(4025949679), UINT32_C(4028132594), UINT32_C(3171101823), UINT32_C(   7800935) } },
    { { -INT32_C(  1150227858),  INT32_C(   624582140), -INT32_C(   666496129), -INT32_C(   762884791), -INT32_C(   792182741),  INT32_C(   901838609), -INT32_C(    55221621),  INT32_C(   626909622),
        -INT32_C(   924791093), -INT32_C(  1427301845),  INT32_C(  2005087022), -INT32_C(  1404499327),  INT32_C(  1551635018), -INT32_C(  1785644023),  INT32_C(  1418806942), -INT32_C(   210112985) },
      UINT16_C(22972),
      { UINT32_C(2826234043), UINT32_C( 366781074), UINT32_C(1646222617), UINT32_C(2238999049), UINT32_C(1472298694), UINT32_C(2761842451), UINT32_C( 764593587), UINT32_C(1938182072),
        UINT32_C(3374119479), UINT32_C(3051354268), UINT32_C(1578696277), UINT32_C( 786664552), UINT32_C( 495363082), UINT32_C(3066110979), UINT32_C(2732807401), UINT32_C(3658836643) },
      { UINT32_C(  27472228), UINT32_C(2125890089), UINT32_C(3923562113), UINT32_C(1696120667), UINT32_C(1719901795), UINT32_C(2870822082), UINT32_C( 994902168), UINT32_C(3474285418),
        UINT32_C(3201350036), UINT32_C(3158083131), UINT32_C(2963675477), UINT32_C(1008058072), UINT32_C( 513972316), UINT32_C(1976156125), UINT32_C( 716249024), UINT32_C( 251250298) },
      { UINT32_C(3144739438), UINT32_C( 624582140), UINT32_C(1646222617), UINT32_C(1696120667), UINT32_C(1472298694), UINT32_C(2761842451), UINT32_C(4239745675), UINT32_C(1938182072),
        UINT32_C(3201350036), UINT32_C(2867665451), UINT32_C(2005087022), UINT32_C( 786664552), UINT32_C( 495363082), UINT32_C(2509323273), UINT32_C( 716249024), UINT32_C(4084854311) } },
    { { -INT32_C(  1144206977), -INT32_C(  1518925488), -INT32_C(    95085278),  INT32_C(   926313179), -INT32_C(   531244797),  INT32_C(  1481973656), -INT32_C(  1333590474),  INT32_C(  1287552205),
        -INT32_C(  1777890490), -INT32_C(  1237614700),  INT32_C(  2024837276), -INT32_C(    38803462),  INT32_C(  1490879936),  INT32_C(  1521562404),  INT32_C(   101332025), -INT32_C(   162281296) },
      UINT16_C(23123),
      { UINT32_C(3352946572), UINT32_C(1314354845), UINT32_C(2637517550), UINT32_C( 765654351), UINT32_C(4267755085), UINT32_C( 707959072), UINT32_C(4092847008), UINT32_C(1716340441),
        UINT32_C(3408733998), UINT32_C(2333705629), UINT32_C( 640175831), UINT32_C(2438187843), UINT32_C(2995762065), UINT32_C(3990667853), UINT32_C(2128662437), UINT32_C(1155804438) },
      { UINT32_C(4044296788), UINT32_C(1853630871), UINT32_C(3147081079), UINT32_C(  21817456), UINT32_C(3904101275), UINT32_C(1121292445), UINT32_C(1975629151), UINT32_C( 934913507),
        UINT32_C(1311361463), UINT32_C(1773970930), UINT32_C(3122942282), UINT32_C(3569119289), UINT32_C(3921506124), UINT32_C(1596756735), UINT32_C( 735374664), UINT32_C(1247973010) },
      { UINT32_C(3352946572), UINT32_C(1314354845), UINT32_C(4199882018), UINT32_C( 926313179), UINT32_C(3904101275), UINT32_C(1481973656), UINT32_C(1975629151), UINT32_C(1287552205),
        UINT32_C(2517076806), UINT32_C(1773970930), UINT32_C(2024837276), UINT32_C(2438187843), UINT32_C(2995762065), UINT32_C(1521562404), UINT32_C( 735374664), UINT32_C(4132686000) } },
    { {  INT32_C(  1234733911),  INT32_C(  2075284785), -INT32_C(   550053978), -INT32_C(  1816923577),  INT32_C(  1635610721),  INT32_C(  1270917379),  INT32_C(   678859926),  INT32_C(  2037569570),
        -INT32_C(  1782445212),  INT32_C(   101741920), -INT32_C(  1813690804), -INT32_C(  1708681160),  INT32_C(   217818121), -INT32_C(   480789683),  INT32_C(  1913376079),  INT32_C(   166428325) },
      UINT16_C(44681),
      { UINT32_C(2938366366), UINT32_C(3572854767), UINT32_C( 694955522), UINT32_C(3285022152), UINT32_C(3632142977), UINT32_C( 161861117), UINT32_C( 730286911), UINT32_C(4091088980),
        UINT32_C(3902995705), UINT32_C(1841076075), UINT32_C(2375493829), UINT32_C( 525362334), UINT32_C(2096680575), UINT32_C(3682966940), UINT32_C(1023806696), UINT32_C( 355524380) },
      { UINT32_C(1191039707), UINT32_C( 800373097), UINT32_C(2159823842), UINT32_C( 782175663), UINT32_C(2007734235), UINT32_C( 491991093), UINT32_C(1499093309), UINT32_C( 342854201),
        UINT32_C(3327880284), UINT32_C( 150277926), UINT32_C( 159953242), UINT32_C(2587371454), UINT32_C(4094813119), UINT32_C(1343317011), UINT32_C(4155141310), UINT32_C(1393236470) },
      { UINT32_C(1191039707), UINT32_C(2075284785), UINT32_C(3744913318), UINT32_C( 782175663), UINT32_C(1635610721), UINT32_C(1270917379), UINT32_C( 678859926), UINT32_C( 342854201),
        UINT32_C(2512522084), UINT32_C( 150277926), UINT32_C( 159953242), UINT32_C( 525362334), UINT32_C( 217818121), UINT32_C(1343317011), UINT32_C(1913376079), UINT32_C( 355524380) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_epu32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_epu32");
    easysimd_test_x86_assert_equal_u32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const uint32_t a[16];
    const uint32_t b[16];
    const uint32_t r[16];
  } test_vec[] = {
    { UINT16_C(39416),
      { UINT32_C(1465973120), UINT32_C(2116295527), UINT32_C(3331100915), UINT32_C(1680350051), UINT32_C( 383124669), UINT32_C(   7555327), UINT32_C( 691857575), UINT32_C(1187198150),
        UINT32_C(2493326125), UINT32_C( 823312958), UINT32_C(3069747026), UINT32_C(1998201018), UINT32_C( 596504612), UINT32_C(3760390456), UINT32_C(2953404393), UINT32_C(3237399699) },
      { UINT32_C( 777295087), UINT32_C(2824824662), UINT32_C(3227408134), UINT32_C(2604169591), UINT32_C(2730411369), UINT32_C(2961367494), UINT32_C(3546319680), UINT32_C(1200838232),
        UINT32_C(1081469162), UINT32_C(1441387855), UINT32_C(2752857900), UINT32_C( 708791744), UINT32_C(3654090259), UINT32_C( 545869535), UINT32_C( 838068697), UINT32_C( 712607552) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(1680350051), UINT32_C( 383124669), UINT32_C(   7555327), UINT32_C( 691857575), UINT32_C(1187198150),
        UINT32_C(1081469162), UINT32_C(         0), UINT32_C(         0), UINT32_C( 708791744), UINT32_C( 596504612), UINT32_C(         0), UINT32_C(         0), UINT32_C( 712607552) } },
    { UINT16_C(61039),
      { UINT32_C(1422114411), UINT32_C( 681308179), UINT32_C(3547749524), UINT32_C(1389463942), UINT32_C(3969954146), UINT32_C(3302324689), UINT32_C( 608900523), UINT32_C(2870131264),
        UINT32_C(2348799608), UINT32_C(1521785542), UINT32_C(2083334902), UINT32_C( 365887411), UINT32_C(2164354736), UINT32_C(2470828008), UINT32_C( 750227948), UINT32_C(3302476107) },
      { UINT32_C(1699731103), UINT32_C(1740571505), UINT32_C(3773099309), UINT32_C(2633413356), UINT32_C( 152958753), UINT32_C(3147588302), UINT32_C(1072124915), UINT32_C(3154362140),
        UINT32_C( 102847125), UINT32_C(2205081942), UINT32_C(3127136974), UINT32_C( 626416132), UINT32_C( 539915089), UINT32_C(3386624725), UINT32_C( 973652509), UINT32_C( 402000769) },
      { UINT32_C(1422114411), UINT32_C( 681308179), UINT32_C(3547749524), UINT32_C(1389463942), UINT32_C(         0), UINT32_C(3147588302), UINT32_C( 608900523), UINT32_C(         0),
        UINT32_C(         0), UINT32_C(1521785542), UINT32_C(2083334902), UINT32_C( 365887411), UINT32_C(         0), UINT32_C(2470828008), UINT32_C( 750227948), UINT32_C( 402000769) } },
    { UINT16_C( 5981),
      { UINT32_C(2348331805), UINT32_C(2615002679), UINT32_C(3606438528), UINT32_C( 910771719), UINT32_C(1090527078), UINT32_C(4026801896), UINT32_C(1325106520), UINT32_C(3127203996),
        UINT32_C(1128619532), UINT32_C(2782798628), UINT32_C( 209441541), UINT32_C(2151859481), UINT32_C(3435217892), UINT32_C(3116156257), UINT32_C(3876042571), UINT32_C( 463563791) },
      { UINT32_C(4015974346), UINT32_C( 261372938), UINT32_C( 689639183), UINT32_C(3098107604), UINT32_C(  42232481), UINT32_C(2075869232), UINT32_C( 123912951), UINT32_C(4179756078),
        UINT32_C(4125655531), UINT32_C(3439623357), UINT32_C(1626742667), UINT32_C( 504930173), UINT32_C( 958438665), UINT32_C(3585399773), UINT32_C(3436976029), UINT32_C( 113638939) },
      { UINT32_C(2348331805), UINT32_C(         0), UINT32_C( 689639183), UINT32_C( 910771719), UINT32_C(  42232481), UINT32_C(         0), UINT32_C( 123912951), UINT32_C(         0),
        UINT32_C(1128619532), UINT32_C(2782798628), UINT32_C( 209441541), UINT32_C(         0), UINT32_C( 958438665), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0) } },
    { UINT16_C(44415),
      { UINT32_C(4280892923), UINT32_C(   2012170), UINT32_C( 765434900), UINT32_C(3687491770), UINT32_C(2528552930), UINT32_C(1487754364), UINT32_C(3847735328), UINT32_C(3381843662),
        UINT32_C( 499694355), UINT32_C(2216552303), UINT32_C(1035058307), UINT32_C(1192786789), UINT32_C(2682113826), UINT32_C(1140296483), UINT32_C( 573066835), UINT32_C( 954972709) },
      { UINT32_C(3830887541), UINT32_C( 510161819), UINT32_C(2505775408), UINT32_C(3084678292), UINT32_C(1716959555), UINT32_C(2561232196), UINT32_C(3250246044), UINT32_C(  16360843),
        UINT32_C(4108603225), UINT32_C(4078063043), UINT32_C(4220022374), UINT32_C( 632448226), UINT32_C(1653278749), UINT32_C(4059706453), UINT32_C(2427630597), UINT32_C(3012602969) },
      { UINT32_C(3830887541), UINT32_C(   2012170), UINT32_C( 765434900), UINT32_C(3084678292), UINT32_C(1716959555), UINT32_C(1487754364), UINT32_C(3250246044), UINT32_C(         0),
        UINT32_C( 499694355), UINT32_C(         0), UINT32_C(1035058307), UINT32_C( 632448226), UINT32_C(         0), UINT32_C(1140296483), UINT32_C(         0), UINT32_C( 954972709) } },
    { UINT16_C(29947),
      { UINT32_C(3133259431), UINT32_C( 958933169), UINT32_C(3583838755), UINT32_C(3135093551), UINT32_C( 401486365), UINT32_C(3603690276), UINT32_C( 327296131), UINT32_C(2139586263),
        UINT32_C(3996731708), UINT32_C(2485608817), UINT32_C(2590623083), UINT32_C(2639545984), UINT32_C(2629059192), UINT32_C(3094576949), UINT32_C(2076964259), UINT32_C(2969195123) },
      { UINT32_C( 211694491), UINT32_C(4288726420), UINT32_C( 177801610), UINT32_C(3366448463), UINT32_C(1684298543), UINT32_C(2115819482), UINT32_C(1090119629), UINT32_C(3589337913),
        UINT32_C(3135344166), UINT32_C(3736699476), UINT32_C(3689501323), UINT32_C(1856213055), UINT32_C(3335653356), UINT32_C(2890198751), UINT32_C( 250363349), UINT32_C(1457773872) },
      { UINT32_C( 211694491), UINT32_C( 958933169), UINT32_C(         0), UINT32_C(3135093551), UINT32_C( 401486365), UINT32_C(2115819482), UINT32_C( 327296131), UINT32_C(2139586263),
        UINT32_C(         0), UINT32_C(         0), UINT32_C(2590623083), UINT32_C(         0), UINT32_C(2629059192), UINT32_C(2890198751), UINT32_C( 250363349), UINT32_C(         0) } },
    { UINT16_C(50539),
      { UINT32_C(3376922384), UINT32_C(2266747550), UINT32_C(1343707821), UINT32_C(2589459400), UINT32_C( 243808202), UINT32_C(3477888483), UINT32_C(1336704108), UINT32_C(3809745107),
        UINT32_C(1974295511), UINT32_C(3690776622), UINT32_C(3945534499), UINT32_C(3783689239), UINT32_C(2666532539), UINT32_C(3631037548), UINT32_C(2334595768), UINT32_C( 158284850) },
      { UINT32_C(3313441943), UINT32_C( 128023524), UINT32_C(2817772943), UINT32_C( 210270545), UINT32_C(4088035463), UINT32_C(1842026420), UINT32_C(1677259569), UINT32_C(3329058607),
        UINT32_C(1754066051), UINT32_C(4151258471), UINT32_C(1268671226), UINT32_C(1666655963), UINT32_C(1398145439), UINT32_C(1254105624), UINT32_C(1152235797), UINT32_C(2752125472) },
      { UINT32_C(3313441943), UINT32_C( 128023524), UINT32_C(         0), UINT32_C( 210270545), UINT32_C(         0), UINT32_C(1842026420), UINT32_C(1336704108), UINT32_C(         0),
        UINT32_C(1754066051), UINT32_C(         0), UINT32_C(1268671226), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(1152235797), UINT32_C( 158284850) } },
    { UINT16_C(38406),
      { UINT32_C(2076405260), UINT32_C(  64929125), UINT32_C(1596569864), UINT32_C(1935722524), UINT32_C(3700783388), UINT32_C(1888856771), UINT32_C(4169905902), UINT32_C(1720684890),
        UINT32_C(1692488447), UINT32_C( 409452304), UINT32_C(2507706745), UINT32_C(1963513945), UINT32_C( 340958545), UINT32_C( 897967943), UINT32_C(4146991261), UINT32_C(2707275169) },
      { UINT32_C( 520437519), UINT32_C(2000186878), UINT32_C(1460515070), UINT32_C(3670873480), UINT32_C(4209909683), UINT32_C(2754638598), UINT32_C( 630939267), UINT32_C( 717682971),
        UINT32_C( 910871352), UINT32_C( 917406264), UINT32_C(3129916210), UINT32_C(2207538128), UINT32_C(2155774842), UINT32_C(2049224438), UINT32_C( 664780812), UINT32_C(4048643513) },
      { UINT32_C(         0), UINT32_C(  64929125), UINT32_C(1460515070), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0),
        UINT32_C(         0), UINT32_C( 409452304), UINT32_C(2507706745), UINT32_C(         0), UINT32_C( 340958545), UINT32_C(         0), UINT32_C(         0), UINT32_C(2707275169) } },
    { UINT16_C(39728),
      { UINT32_C(3575474471), UINT32_C( 764432287), UINT32_C(2659737866), UINT32_C(1646330596), UINT32_C(2802849923), UINT32_C( 828841106), UINT32_C(2509643843), UINT32_C( 959497745),
        UINT32_C(3473821231), UINT32_C(2818351005), UINT32_C(3829826816), UINT32_C( 172451719), UINT32_C( 296900479), UINT32_C(3074562420), UINT32_C(1263327290), UINT32_C(3464789407) },
      { UINT32_C(1755157451), UINT32_C( 839948850), UINT32_C(2786481695), UINT32_C(1035034045), UINT32_C( 693003189), UINT32_C(3068170620), UINT32_C( 134360425), UINT32_C(1977058986),
        UINT32_C(1272804377), UINT32_C( 763227406), UINT32_C(  30708803), UINT32_C(2789115377), UINT32_C(1691323624), UINT32_C(2266738717), UINT32_C(2274303453), UINT32_C(3187435171) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C( 693003189), UINT32_C( 828841106), UINT32_C(         0), UINT32_C(         0),
        UINT32_C(1272804377), UINT32_C( 763227406), UINT32_C(         0), UINT32_C( 172451719), UINT32_C( 296900479), UINT32_C(         0), UINT32_C(         0), UINT32_C(3187435171) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_epu32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_epu32");
    easysimd_test_x86_assert_equal_u32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_min_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 8926487841177630399), -INT64_C(  198981365653089870), -INT64_C( 1142750507646648470), -INT64_C(  317195656128959639),
        -INT64_C( 3931785308317887090),  INT64_C( 4705300458382083023), -INT64_C( 6125974934284591212), -INT64_C( 4229869377263821291) },
      {  INT64_C( 3832889178177792081), -INT64_C( 5619417979985557013),  INT64_C( 5250275990352824560), -INT64_C( 8299037646502903848),
        -INT64_C( 6887931595367019315), -INT64_C( 5800242970600739152), -INT64_C( 3321196944853811346),  INT64_C( 3766050967148003707) },
      {  INT64_C( 3832889178177792081), -INT64_C( 5619417979985557013), -INT64_C( 1142750507646648470), -INT64_C( 8299037646502903848),
        -INT64_C( 6887931595367019315), -INT64_C( 5800242970600739152), -INT64_C( 6125974934284591212), -INT64_C( 4229869377263821291) } },
    { {  INT64_C( 1050976293818480703), -INT64_C( 3806282354404755521),  INT64_C( 5073116734339878557),  INT64_C( 4138878397256058581),
         INT64_C( 7630437703371053175), -INT64_C( 4836279826553257542),  INT64_C( 8589413019448134947), -INT64_C( 1787327138275053568) },
      { -INT64_C( 2590237040503332613),  INT64_C( 3260558896362943843),  INT64_C( 5421012663790113059),  INT64_C(  924494742396661374),
        -INT64_C( 7577660192562507305), -INT64_C( 4401609342217303629), -INT64_C( 3350745935121086999),  INT64_C( 1283124824431838128) },
      { -INT64_C( 2590237040503332613), -INT64_C( 3806282354404755521),  INT64_C( 5073116734339878557),  INT64_C(  924494742396661374),
        -INT64_C( 7577660192562507305), -INT64_C( 4836279826553257542), -INT64_C( 3350745935121086999), -INT64_C( 1787327138275053568) } },
    { { -INT64_C( 1013906683223900380),  INT64_C( 5047296433345988266),  INT64_C(  343110664857490078),  INT64_C( 4037922458203226557),
        -INT64_C( 6970307830657051628),  INT64_C( 3109722953645421443),  INT64_C( 6404083055616369286),  INT64_C( 5741061732584957280) },
      {  INT64_C( 4396047299971726445),  INT64_C( 7318427319765232344), -INT64_C( 1451250295857173272), -INT64_C(  986837498692948796),
        -INT64_C( 5962671533283020984), -INT64_C( 8321347385694200256), -INT64_C( 6285075766588685233),  INT64_C( 8831546987744011544) },
      { -INT64_C( 1013906683223900380),  INT64_C( 5047296433345988266), -INT64_C( 1451250295857173272), -INT64_C(  986837498692948796),
        -INT64_C( 6970307830657051628), -INT64_C( 8321347385694200256), -INT64_C( 6285075766588685233),  INT64_C( 5741061732584957280) } },
    { {  INT64_C( 8676110158968710116),  INT64_C( 7731585570087336219),  INT64_C( 6947940732284263648), -INT64_C( 1379073418233834703),
         INT64_C( 3467786099733453167), -INT64_C( 5472651092515833978), -INT64_C( 8124242631632333928),  INT64_C( 4101599252628782583) },
      {  INT64_C( 2697092806972772647),  INT64_C( 3548508411849563575), -INT64_C( 7992110764606245336), -INT64_C(  103727006372329330),
         INT64_C( 2725144442825305869), -INT64_C( 7651072137327765498), -INT64_C( 4408687360240459099), -INT64_C( 1509082058199506630) },
      {  INT64_C( 2697092806972772647),  INT64_C( 3548508411849563575), -INT64_C( 7992110764606245336), -INT64_C( 1379073418233834703),
         INT64_C( 2725144442825305869), -INT64_C( 7651072137327765498), -INT64_C( 8124242631632333928), -INT64_C( 1509082058199506630) } },
    { { -INT64_C(  527931665442977512), -INT64_C( 1151962406489465856), -INT64_C( 8412442278925230261),  INT64_C( 2101679115640527714),
         INT64_C( 3088995634827805172),  INT64_C( 3019834932107703725),  INT64_C( 8834066958588057787), -INT64_C( 3285405759755897787) },
      {  INT64_C(  790828539241303206), -INT64_C( 1723775649920610036), -INT64_C( 1614948779877418237), -INT64_C( 2634153652428517184),
        -INT64_C( 3742873095679366489),  INT64_C(   21051238396596533), -INT64_C( 8353416673669398652), -INT64_C( 8641390768869915133) },
      { -INT64_C(  527931665442977512), -INT64_C( 1723775649920610036), -INT64_C( 8412442278925230261), -INT64_C( 2634153652428517184),
        -INT64_C( 3742873095679366489),  INT64_C(   21051238396596533), -INT64_C( 8353416673669398652), -INT64_C( 8641390768869915133) } },
    { { -INT64_C( 7542515202943828282),  INT64_C( 6388713222282283692),  INT64_C( 8996946829836928643),  INT64_C( 7584845323688019673),
         INT64_C( 1549312393974173318),  INT64_C( 4789973744992811597),  INT64_C( 7431903165732223533),  INT64_C( 2845541178263328882) },
      { -INT64_C(   94417599201317582),  INT64_C(  219155580128816649),  INT64_C( 8757193430941735826),  INT64_C( 4570039869208635557),
         INT64_C( 1524621353927998584),  INT64_C( 8274211893155809273), -INT64_C( 1224388340765000318),  INT64_C( 1372931147674456002) },
      { -INT64_C( 7542515202943828282),  INT64_C(  219155580128816649),  INT64_C( 8757193430941735826),  INT64_C( 4570039869208635557),
         INT64_C( 1524621353927998584),  INT64_C( 4789973744992811597), -INT64_C( 1224388340765000318),  INT64_C( 1372931147674456002) } },
    { {  INT64_C( 7799500575434663965), -INT64_C( 8935688111334352212),  INT64_C( 7837686853406593420), -INT64_C( 5239013914309822050),
        -INT64_C( 7489453278118246352), -INT64_C( 1748202205642208200),  INT64_C( 8560079382561802676),  INT64_C( 9209292026337429115) },
      {  INT64_C(  218198956258274690),  INT64_C(  198432500651666302),  INT64_C( 8867918617604357571),  INT64_C( 4323278318117961522),
        -INT64_C( 3181035208830213620), -INT64_C( 3229805441535174948), -INT64_C( 1412582467337023766),  INT64_C( 6932003363654334014) },
      {  INT64_C(  218198956258274690), -INT64_C( 8935688111334352212),  INT64_C( 7837686853406593420), -INT64_C( 5239013914309822050),
        -INT64_C( 7489453278118246352), -INT64_C( 3229805441535174948), -INT64_C( 1412582467337023766),  INT64_C( 6932003363654334014) } },
    { {  INT64_C( 7761410313214745998), -INT64_C( 5040720063136112088),  INT64_C( 3961208308217706834),  INT64_C( 7040360772965132031),
         INT64_C( 2682451021070134079), -INT64_C( 1952758411399671972), -INT64_C( 7921855298783835423), -INT64_C( 2340858468243259824) },
      {  INT64_C( 2208987861044021125),  INT64_C( 5560872881523131573),  INT64_C( 6331837906581593530), -INT64_C( 2020993227263654797),
         INT64_C( 4369631314671149253), -INT64_C( 2825532546702053872), -INT64_C( 7481318849734381618),  INT64_C( 7641094149959821257) },
      {  INT64_C( 2208987861044021125), -INT64_C( 5040720063136112088),  INT64_C( 3961208308217706834), -INT64_C( 2020993227263654797),
         INT64_C( 2682451021070134079), -INT64_C( 2825532546702053872), -INT64_C( 7921855298783835423), -INT64_C( 2340858468243259824) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_min_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_min_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C(  832893060298135842), -INT64_C( 5562894295615803047), -INT64_C( 1796689962277743011), -INT64_C( 8401131206823195650),
        -INT64_C( 4109912428081810219), -INT64_C( 5014718200183997287),  INT64_C( 3666331645962441979), -INT64_C( 4580075145466234056) },
      UINT8_C(193),
      {  INT64_C( 8496871624244314792),  INT64_C( 6659570441877889100),  INT64_C(  256637315434178281), -INT64_C( 8235765515890559834),
         INT64_C( 2422877041933082993), -INT64_C( 1743458215748603543), -INT64_C( 4130420842561708445), -INT64_C( 4921469802541114559) },
      { -INT64_C( 4303888341886725553),  INT64_C( 3952193257319290613),  INT64_C( 2614215216499595104), -INT64_C( 4814349726284906198),
        -INT64_C( 8237322755203066736),  INT64_C( 5118158349405250813), -INT64_C( 6883735029549569936),  INT64_C( 5781570515681722896) },
      { -INT64_C( 4303888341886725553), -INT64_C( 5562894295615803047), -INT64_C( 1796689962277743011), -INT64_C( 8401131206823195650),
        -INT64_C( 4109912428081810219), -INT64_C( 5014718200183997287), -INT64_C( 6883735029549569936), -INT64_C( 4921469802541114559) } },
    { {  INT64_C( 4718904091369741911), -INT64_C( 4180438498205827890), -INT64_C( 2010697995710315519), -INT64_C( 4078960058780406701),
         INT64_C( 6359983766621364157),  INT64_C( 8018950981776397615),  INT64_C( 5009691578083009229),  INT64_C( 5000048401435977771) },
      UINT8_C( 92),
      { -INT64_C( 2270588773879888102), -INT64_C( 5970719450183368372),  INT64_C( 3589339532494881957), -INT64_C( 1704613768996492482),
        -INT64_C( 3913713297744642491),  INT64_C( 7038074796686442164),  INT64_C( 7963160395127278343),  INT64_C( 9148430668855327551) },
      { -INT64_C( 7978692037101868033), -INT64_C( 5829843022299240965),  INT64_C( 3541944710442524040),  INT64_C( 5797879096948003006),
        -INT64_C( 4769854329868991393),  INT64_C( 2580155333917106039),  INT64_C( 1210909003179937707), -INT64_C( 6830413334550409837) },
      {  INT64_C( 4718904091369741911), -INT64_C( 4180438498205827890),  INT64_C( 3541944710442524040), -INT64_C( 1704613768996492482),
        -INT64_C( 4769854329868991393),  INT64_C( 8018950981776397615),  INT64_C( 1210909003179937707),  INT64_C( 5000048401435977771) } },
    { { -INT64_C( 8375660580469603773),  INT64_C( 8127722742837032972),  INT64_C( 4443606607624775495),  INT64_C( 8488537840863589097),
         INT64_C( 7275914123013696346),  INT64_C( 7182070542727160693),  INT64_C( 9107868980994685310),  INT64_C( 4251248379543849049) },
      UINT8_C( 34),
      {  INT64_C( 4912828654192142458), -INT64_C( 3728767101341570837),  INT64_C( 6250680509905594559), -INT64_C( 4457595751606367862),
         INT64_C( 7637444705809767960), -INT64_C( 2382566796882584561), -INT64_C( 4973941905533218603), -INT64_C( 5235751640300310709) },
      { -INT64_C( 8762477137892421507), -INT64_C( 3258750163570093623),  INT64_C( 6328522899614665448), -INT64_C( 1444740217538179427),
        -INT64_C( 7346132058166107376),  INT64_C( 3867546778731722460),  INT64_C( 3365914675463987795), -INT64_C( 1078507789801054033) },
      { -INT64_C( 8375660580469603773), -INT64_C( 3728767101341570837),  INT64_C( 4443606607624775495),  INT64_C( 8488537840863589097),
         INT64_C( 7275914123013696346), -INT64_C( 2382566796882584561),  INT64_C( 9107868980994685310),  INT64_C( 4251248379543849049) } },
    { {  INT64_C( 1597799553933373121),  INT64_C( 5386197365871914556), -INT64_C( 2414669172691104321),  INT64_C(  563516506348709888),
         INT64_C( 1027601171459165169), -INT64_C( 8163239626675834252), -INT64_C( 4651554586725818523),  INT64_C( 7030526320972950851) },
      UINT8_C(190),
      {  INT64_C(  100379707640417578), -INT64_C( 1876770117458473161),  INT64_C( 7483966608753592381),  INT64_C( 2392319562060621315),
        -INT64_C( 7379581873131764794),  INT64_C(  337520141491791685), -INT64_C( 5113983927075384411), -INT64_C( 2452102142569226528) },
      { -INT64_C( 4314156524689552858),  INT64_C(  737533043426675056),  INT64_C( 3675695217147304338), -INT64_C( 5078930547069537905),
         INT64_C( 8093357759854147341), -INT64_C( 5756756880866615125),  INT64_C(  451762347170186685),  INT64_C( 6879584137258028844) },
      {  INT64_C( 1597799553933373121), -INT64_C( 1876770117458473161),  INT64_C( 3675695217147304338), -INT64_C( 5078930547069537905),
        -INT64_C( 7379581873131764794), -INT64_C( 5756756880866615125), -INT64_C( 4651554586725818523), -INT64_C( 2452102142569226528) } },
    { { -INT64_C( 3992093498748978648), -INT64_C( 2095311661344145124),  INT64_C(  808177189403223226),  INT64_C( 4483408289686348935),
         INT64_C( 8999598743634715646),  INT64_C( 7874723935063358784), -INT64_C(  400022725246174329), -INT64_C( 9219981684985610823) },
      UINT8_C( 47),
      {  INT64_C( 4445909899155577715), -INT64_C( 7138948265198806845),  INT64_C(  326522826273019130), -INT64_C( 3019656523962492963),
         INT64_C( 8457474992241223585), -INT64_C( 6624291842926359276),  INT64_C(  642321683028647965),  INT64_C( 1624965441752493907) },
      { -INT64_C( 1740476525155199312), -INT64_C( 7335961936294792221), -INT64_C(  738476390311090803),  INT64_C( 7665754087545942486),
         INT64_C( 1354075163570510096),  INT64_C( 2337034589997835864),  INT64_C( 3102303205289684342),  INT64_C(  319149691154905673) },
      { -INT64_C( 1740476525155199312), -INT64_C( 7335961936294792221), -INT64_C(  738476390311090803), -INT64_C( 3019656523962492963),
         INT64_C( 8999598743634715646), -INT64_C( 6624291842926359276), -INT64_C(  400022725246174329), -INT64_C( 9219981684985610823) } },
    { {  INT64_C( 3941689668636457586), -INT64_C( 7454731569466748201),  INT64_C( 5036923225950413670), -INT64_C( 6381400331417723784),
        -INT64_C( 2232102020741310224), -INT64_C( 1150433814732140467), -INT64_C( 1262255333637900230),  INT64_C( 7767244237088408814) },
      UINT8_C( 35),
      {  INT64_C( 5789954839688264210), -INT64_C( 8473017124609406631), -INT64_C( 2308394859603386506), -INT64_C(  776933732096796343),
        -INT64_C( 3850194564950105088), -INT64_C( 5944360945412576475), -INT64_C( 4972629915507261181),  INT64_C( 2205644646238804164) },
      {  INT64_C( 2280775221522701930),  INT64_C(  450244529351991387),  INT64_C( 7152598701790524441), -INT64_C( 9027845778457357702),
         INT64_C( 7015767115569108292),  INT64_C( 2642823581980128419),  INT64_C( 7937860376831382054),  INT64_C( 2062011840757903875) },
      {  INT64_C( 2280775221522701930), -INT64_C( 8473017124609406631),  INT64_C( 5036923225950413670), -INT64_C( 6381400331417723784),
        -INT64_C( 2232102020741310224), -INT64_C( 5944360945412576475), -INT64_C( 1262255333637900230),  INT64_C( 7767244237088408814) } },
    { { -INT64_C( 6757448935706788910), -INT64_C( 8981314151640388821), -INT64_C(  132169012556068408), -INT64_C( 4641320135375048542),
        -INT64_C( 3219000985198858062),  INT64_C( 6618996856079424762),  INT64_C( 3992795749843149935), -INT64_C( 1360570621868539022) },
      UINT8_C( 56),
      { -INT64_C( 8250958517886738763), -INT64_C( 2984625747510962498), -INT64_C( 1565386353147183470),  INT64_C( 8778380734301054547),
         INT64_C( 2751273211206153392),  INT64_C( 6546791058494934354),  INT64_C( 4120640716419253725),  INT64_C( 1610222026580317210) },
      {  INT64_C( 2660191223321941596), -INT64_C(  984877554862574285),  INT64_C( 1326423302095733576),  INT64_C( 4699587459626909115),
         INT64_C(  618945698813308831),  INT64_C(  804834549619212069), -INT64_C( 8870786378313866533), -INT64_C( 4754150703106763297) },
      { -INT64_C( 6757448935706788910), -INT64_C( 8981314151640388821), -INT64_C(  132169012556068408),  INT64_C( 4699587459626909115),
         INT64_C(  618945698813308831),  INT64_C(  804834549619212069),  INT64_C( 3992795749843149935), -INT64_C( 1360570621868539022) } },
    { {  INT64_C( 2281854940981758669), -INT64_C( 2282616902916723323),  INT64_C( 1023327771410205926), -INT64_C( 8986131827922839188),
        -INT64_C( 4180238476611585309), -INT64_C( 4656335156241738628),  INT64_C( 1669043992684008683), -INT64_C( 5208995898362430341) },
      UINT8_C(103),
      { -INT64_C( 1934739621314849574),  INT64_C( 2144694229274568824),  INT64_C(  146613306642718187), -INT64_C( 3902493397346731645),
         INT64_C( 7104597645489045027), -INT64_C( 4798306323316189631), -INT64_C( 7718235099635240486),  INT64_C( 7801320088162056844) },
      { -INT64_C( 7951746912189696423), -INT64_C( 5215654503480762610), -INT64_C( 4099547540182472586),  INT64_C( 6047074235598315318),
        -INT64_C( 3450536032744371653), -INT64_C( 5427313369072666341), -INT64_C(  560321285490784713),  INT64_C( 4306111947729238901) },
      { -INT64_C( 7951746912189696423), -INT64_C( 5215654503480762610), -INT64_C( 4099547540182472586), -INT64_C( 8986131827922839188),
        -INT64_C( 4180238476611585309), -INT64_C( 5427313369072666341), -INT64_C( 7718235099635240486), -INT64_C( 5208995898362430341) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(108),
      { -INT64_C( 2472476313199327400), -INT64_C( 7262106806340770128),  INT64_C( 6834306869630970124),  INT64_C( 7105441106113742796),
         INT64_C( 8800778905699404384),  INT64_C( 1890579520473873996), -INT64_C( 4856193287171761574), -INT64_C(  934068582790530660) },
      { -INT64_C( 1039388523364876465),  INT64_C( 1711595172339518615),  INT64_C( 3392599831143005544), -INT64_C( 3211666298667401496),
        -INT64_C( 7605313139610180130),  INT64_C( 4422812822769763980), -INT64_C( 8359901762306067398), -INT64_C( 8214719999122222681) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 3392599831143005544), -INT64_C( 3211666298667401496),
         INT64_C(                   0),  INT64_C( 1890579520473873996), -INT64_C( 8359901762306067398),  INT64_C(                   0) } },
    { UINT8_C( 24),
      { -INT64_C( 8433082092495695637),  INT64_C( 6061735838320902376), -INT64_C( 5151692566412399823), -INT64_C( 7560710701117362809),
        -INT64_C( 4749897011430283003),  INT64_C( 4624565765711033145),  INT64_C( 3336213663237940397), -INT64_C( 5869250613222001591) },
      { -INT64_C( 4397961263633100956),  INT64_C(  725300940480682688),  INT64_C( 1207786892513127405), -INT64_C( 8505451256934438241),
         INT64_C( 4353778509370568480), -INT64_C( 4775190202030487979), -INT64_C( 2008865721453290936), -INT64_C( 2977967092224219907) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 8505451256934438241),
        -INT64_C( 4749897011430283003),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 81),
      { -INT64_C( 1117162181707858065),  INT64_C( 5078791458224353302), -INT64_C( 3612327075680384306),  INT64_C( 2093990958789231003),
        -INT64_C(  165851541855019816), -INT64_C( 4816128180187270409), -INT64_C( 4706968442486475722),  INT64_C( 7010568091717021822) },
      {  INT64_C( 5713614752012538215),  INT64_C( 6770417838826093384), -INT64_C( 8074695672785431537),  INT64_C( 6703052751799872283),
         INT64_C( 5205867603656583831), -INT64_C( 2669378062331645840),  INT64_C( 4083353214102811146),  INT64_C(  341117350895773288) },
      { -INT64_C( 1117162181707858065),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(  165851541855019816),  INT64_C(                   0), -INT64_C( 4706968442486475722),  INT64_C(                   0) } },
    { UINT8_C( 14),
      { -INT64_C( 6716461120010390683), -INT64_C( 8622343426074058281), -INT64_C( 8954883270350900651),  INT64_C( 4084188523046836155),
        -INT64_C( 1676644108240503833), -INT64_C( 4063218342201841218), -INT64_C( 2693484496080584194),  INT64_C( 7562712012916624873) },
      { -INT64_C( 6711703869941774623), -INT64_C( 7232970539122945946),  INT64_C( 8326404236480264084),  INT64_C(  993926816314885858),
        -INT64_C( 1283565989249659735), -INT64_C( 3664692903285430805),  INT64_C( 1232031570882255389),  INT64_C( 1535234252872108052) },
      {  INT64_C(                   0), -INT64_C( 8622343426074058281), -INT64_C( 8954883270350900651),  INT64_C(  993926816314885858),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(139),
      { -INT64_C( 5326347886935170482), -INT64_C( 7184441242637328952),  INT64_C( 6007762368993161532),  INT64_C( 4403065706787086416),
        -INT64_C(  505112140204069514), -INT64_C( 1912216021743768551),  INT64_C( 7908758677676310882), -INT64_C( 3865752779607998418) },
      {  INT64_C( 6155028968523284548),  INT64_C( 7316280124921319931),  INT64_C( 8462988578737273063), -INT64_C( 3949115340514758893),
         INT64_C( 5493732190204019372), -INT64_C(  386896068955116230),  INT64_C( 2011827034937848880),  INT64_C( 1598976232500919777) },
      { -INT64_C( 5326347886935170482), -INT64_C( 7184441242637328952),  INT64_C(                   0), -INT64_C( 3949115340514758893),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 3865752779607998418) } },
    { UINT8_C(246),
      { -INT64_C( 6246612271875228051), -INT64_C( 6833167700991374205),  INT64_C(  723325325721313718),  INT64_C(  932679017548771142),
         INT64_C( 5723355825735612552), -INT64_C( 7580554467111982595), -INT64_C( 9174176340532890346),  INT64_C( 3810726624785736246) },
      {  INT64_C( 8340497693800989860), -INT64_C( 7780413115436496496), -INT64_C( 4496027792861248288),  INT64_C( 2659510273166480392),
        -INT64_C( 2852938791133060738),  INT64_C( 5939771209448687196), -INT64_C( 7755754307279084014), -INT64_C( 1243502087413107174) },
      {  INT64_C(                   0), -INT64_C( 7780413115436496496), -INT64_C( 4496027792861248288),  INT64_C(                   0),
        -INT64_C( 2852938791133060738), -INT64_C( 7580554467111982595), -INT64_C( 9174176340532890346), -INT64_C( 1243502087413107174) } },
    { UINT8_C( 18),
      {  INT64_C( 5837001892527006546), -INT64_C( 1091536548743881862), -INT64_C( 3701657805628016767),  INT64_C( 5592673567830195290),
        -INT64_C( 6509121848692508659),  INT64_C(  219067255490440655), -INT64_C(  981455309446150209), -INT64_C( 3049187875246727833) },
      {  INT64_C( 7341668478055522579),  INT64_C( 6326018816358633541), -INT64_C( 1758006394332818417),  INT64_C( 5082866555416025324),
        -INT64_C( 3128763075952134247), -INT64_C( 6150397998391422282), -INT64_C( 3665668545094446407), -INT64_C(  396868304485914363) },
      {  INT64_C(                   0), -INT64_C( 1091536548743881862),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C( 6509121848692508659),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(194),
      { -INT64_C( 3450559144571885311), -INT64_C( 6638282687324132415), -INT64_C( 6747991784971560600),  INT64_C( 1270012968322699986),
        -INT64_C( 2831191765649067993), -INT64_C( 7082227944931778558),  INT64_C( 6905509768676391929), -INT64_C(  143873471909406498) },
      {  INT64_C( 1277500638311446987), -INT64_C( 7959026466587100744), -INT64_C( 6340617704767718122), -INT64_C( 1997743254139355791),
         INT64_C( 8732902719398226192),  INT64_C( 5072162797323179321),  INT64_C( 3594412088944568175),  INT64_C( 2244128586564466728) },
      {  INT64_C(                   0), -INT64_C( 7959026466587100744),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C( 3594412088944568175), -INT64_C(  143873471909406498) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_min_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint64_t a[8];
    const uint64_t b[8];
    const uint64_t r[8];
  } test_vec[] = {
    { { UINT64_C( 9044969195737560887), UINT64_C(15694047717360971943), UINT64_C( 3921003438290561397), UINT64_C( 1509185112255430061),
        UINT64_C( 1186214072345967210), UINT64_C(13118825461627079671), UINT64_C( 6081951486148113239), UINT64_C(14942198529738257003) },
      { UINT64_C( 6633026429662010772), UINT64_C( 8358859155088518921), UINT64_C(15929527043916891157), UINT64_C( 7562250571942820542),
        UINT64_C(  171263199686277792), UINT64_C(12299977736181314337), UINT64_C( 2494845092740159312), UINT64_C(16658313952610403826) },
      { UINT64_C( 6633026429662010772), UINT64_C( 8358859155088518921), UINT64_C( 3921003438290561397), UINT64_C( 1509185112255430061),
        UINT64_C(  171263199686277792), UINT64_C(12299977736181314337), UINT64_C( 2494845092740159312), UINT64_C(14942198529738257003) } },
    { { UINT64_C( 2877320800269800435), UINT64_C(15600214605897743167), UINT64_C( 5833914546256537199), UINT64_C( 7434955301018025215),
        UINT64_C(11962478044776700631), UINT64_C(17356021396500277629), UINT64_C(12722029792799782879), UINT64_C(13750093188666139638) },
      { UINT64_C(16450416231898096958), UINT64_C(11419375756601050375), UINT64_C(17121957871900966204), UINT64_C(17848655622549509133),
        UINT64_C(17625936570506563151), UINT64_C( 9658802046534190252), UINT64_C( 6635510132672554546), UINT64_C(16596895574919311013) },
      { UINT64_C( 2877320800269800435), UINT64_C(11419375756601050375), UINT64_C( 5833914546256537199), UINT64_C( 7434955301018025215),
        UINT64_C(11962478044776700631), UINT64_C( 9658802046534190252), UINT64_C( 6635510132672554546), UINT64_C(13750093188666139638) } },
    { { UINT64_C( 9525738658961682514), UINT64_C(15531782222786834364), UINT64_C(13081574900903247839), UINT64_C( 3603838339067862149),
        UINT64_C(10715088087348781143), UINT64_C( 8921109267966698066), UINT64_C( 7703748995652963876), UINT64_C(13413279222790093586) },
      { UINT64_C(10130456184787408426), UINT64_C( 8671207557433601854), UINT64_C( 5291893621416263712), UINT64_C(12370650962216155025),
        UINT64_C( 8196062254738544376), UINT64_C(12916219912397734514), UINT64_C( 1306371004577737890), UINT64_C(13208129442792496416) },
      { UINT64_C( 9525738658961682514), UINT64_C( 8671207557433601854), UINT64_C( 5291893621416263712), UINT64_C( 3603838339067862149),
        UINT64_C( 8196062254738544376), UINT64_C( 8921109267966698066), UINT64_C( 1306371004577737890), UINT64_C(13208129442792496416) } },
    { { UINT64_C(12604914810315471178), UINT64_C( 8948852409597508073), UINT64_C( 3617927695919606177), UINT64_C( 4056089523943324628),
        UINT64_C(14124311914971738904), UINT64_C(14858062561793898715), UINT64_C(15177173618446665563), UINT64_C(12360322224545428321) },
      { UINT64_C(13510803141578210786), UINT64_C(10265853317895799163), UINT64_C(17292586229111154731), UINT64_C(17387772606307852303),
        UINT64_C(14744273155009629444), UINT64_C(16046585726442475503), UINT64_C( 6256346815122615381), UINT64_C(15952390729833648738) },
      { UINT64_C(12604914810315471178), UINT64_C( 8948852409597508073), UINT64_C( 3617927695919606177), UINT64_C( 4056089523943324628),
        UINT64_C(14124311914971738904), UINT64_C(14858062561793898715), UINT64_C( 6256346815122615381), UINT64_C(12360322224545428321) } },
    { { UINT64_C( 3292236047043268499), UINT64_C( 7421650784503145838), UINT64_C( 7658966040575608492), UINT64_C( 2082138057202156079),
        UINT64_C( 8338326768641418573), UINT64_C(12920145667649963989), UINT64_C(16360997679937263514), UINT64_C( 8347369299134638387) },
      { UINT64_C( 6696207814458529653), UINT64_C(12715509021071868348), UINT64_C( 4665271876632911143), UINT64_C(17213834719280277833),
        UINT64_C( 6021125334691799467), UINT64_C( 9915955935570285271), UINT64_C(11094655412113567658), UINT64_C( 8300958507587990731) },
      { UINT64_C( 3292236047043268499), UINT64_C( 7421650784503145838), UINT64_C( 4665271876632911143), UINT64_C( 2082138057202156079),
        UINT64_C( 6021125334691799467), UINT64_C( 9915955935570285271), UINT64_C(11094655412113567658), UINT64_C( 8300958507587990731) } },
    { { UINT64_C(14391592291341278158), UINT64_C(  644384583809552390), UINT64_C( 4767544504217523520), UINT64_C(17215213124685542317),
        UINT64_C(14414911635327323476), UINT64_C( 5783222324588298461), UINT64_C( 7508075331079635576), UINT64_C(14216673739421890621) },
      { UINT64_C( 9934121281375608658), UINT64_C(12121163729190784726), UINT64_C( 6786020984921528073), UINT64_C( 2357924465355721090),
        UINT64_C(12672024176126968742), UINT64_C(12752449938371551264), UINT64_C(15953052092863910372), UINT64_C(14814462500888715433) },
      { UINT64_C( 9934121281375608658), UINT64_C(  644384583809552390), UINT64_C( 4767544504217523520), UINT64_C( 2357924465355721090),
        UINT64_C(12672024176126968742), UINT64_C( 5783222324588298461), UINT64_C( 7508075331079635576), UINT64_C(14216673739421890621) } },
    { { UINT64_C(18363425136224291918), UINT64_C( 2302544724584525213), UINT64_C(12759129887644936409), UINT64_C(13591512307622011817),
        UINT64_C( 6442888255085264524), UINT64_C(15673531658565171241), UINT64_C(15831312885479221498), UINT64_C( 2966827195079318786) },
      { UINT64_C(17734329592381182216), UINT64_C( 5084807300864719542), UINT64_C(13804375898621320837), UINT64_C(10349654056069184987),
        UINT64_C(16875110073847920236), UINT64_C(17130341031865322025), UINT64_C(16018237363539150288), UINT64_C( 4866974850039053172) },
      { UINT64_C(17734329592381182216), UINT64_C( 2302544724584525213), UINT64_C(12759129887644936409), UINT64_C(10349654056069184987),
        UINT64_C( 6442888255085264524), UINT64_C(15673531658565171241), UINT64_C(15831312885479221498), UINT64_C( 2966827195079318786) } },
    { { UINT64_C( 8380262446387747885), UINT64_C(17261353027049745719), UINT64_C(15819958463686783402), UINT64_C(  458629218341151043),
        UINT64_C(13690295832671672637), UINT64_C( 3868847766836668065), UINT64_C( 4061241865194843161), UINT64_C(15000838980395742030) },
      { UINT64_C(16846935173581345929), UINT64_C( 3097730047321647164), UINT64_C(11291376720116703366), UINT64_C( 4512308540320450106),
        UINT64_C( 9994922769949521796), UINT64_C( 4502561380537360193), UINT64_C( 7573204294845409071), UINT64_C( 5793834518460226675) },
      { UINT64_C( 8380262446387747885), UINT64_C( 3097730047321647164), UINT64_C(11291376720116703366), UINT64_C(  458629218341151043),
        UINT64_C( 9994922769949521796), UINT64_C( 3868847766836668065), UINT64_C( 4061241865194843161), UINT64_C( 5793834518460226675) } }
  };


    for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
      easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
      easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
      easysimd__m512i r;
      EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
        r = easysimd_mm512_min_epu64(a, b);
      } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_epu64");
      easysimd_test_x86_assert_equal_u64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
    }


  return 0;
}

static int
test_easysimd_mm512_mask_min_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint64_t src[8];
    const easysimd__mmask8 k;
    const uint64_t a[8];
    const uint64_t b[8];
    const uint64_t r[8];
  } test_vec[] = {
    { { UINT64_C(13686068365161150177), UINT64_C(17532401932033290988), UINT64_C( 5071801821781528843), UINT64_C(16403007056865297244),
        UINT64_C( 7120014496599492455), UINT64_C(12424961470197204689), UINT64_C(16793558581555422693), UINT64_C( 6061130335265078328) },
      UINT8_C( 60),
      { UINT64_C( 4233125491838707086), UINT64_C(10328029007246474010), UINT64_C(16700761572999175385), UINT64_C(12360454758603227095),
        UINT64_C( 7709134865548524429), UINT64_C( 4020638450122872724), UINT64_C(14292889128457857353), UINT64_C(16452384284722015882) },
      { UINT64_C(15677771670748589271), UINT64_C(16217712503761038202), UINT64_C(16825922508532602316), UINT64_C(  583465947924305118),
        UINT64_C(15847536385992998607), UINT64_C(12873754930782077742), UINT64_C(10007543373175324056), UINT64_C( 2647227444091646595) },
      { UINT64_C(13686068365161150177), UINT64_C(17532401932033290988), UINT64_C(16700761572999175385), UINT64_C(  583465947924305118),
        UINT64_C( 7709134865548524429), UINT64_C( 4020638450122872724), UINT64_C(16793558581555422693), UINT64_C( 6061130335265078328) } },
    { { UINT64_C(15911821866855623830), UINT64_C(16267366106206012712), UINT64_C(11637225703779938120), UINT64_C(12368592210698255917),
        UINT64_C( 3136884699593629437), UINT64_C(13701353551232117754), UINT64_C(16639167544660590123), UINT64_C( 5706357946148470896) },
      UINT8_C(236),
      { UINT64_C(12628685456267017171), UINT64_C(12068772517308106069), UINT64_C(17237886769549067671), UINT64_C(14098677992813346478),
        UINT64_C(  557687355998967691), UINT64_C( 6783499416983102418), UINT64_C( 8662780305157358489), UINT64_C(10723595218708691230) },
      { UINT64_C(18099509682323512437), UINT64_C(17539437360796042819), UINT64_C(  963616319494393629), UINT64_C(16684126273053365484),
        UINT64_C(14982416636392123592), UINT64_C( 1805734526670431170), UINT64_C(12307318254061243088), UINT64_C(11334761829393033460) },
      { UINT64_C(15911821866855623830), UINT64_C(16267366106206012712), UINT64_C(  963616319494393629), UINT64_C(14098677992813346478),
        UINT64_C( 3136884699593629437), UINT64_C( 1805734526670431170), UINT64_C( 8662780305157358489), UINT64_C(10723595218708691230) } },
    { { UINT64_C( 6462801357531142997), UINT64_C(11165695382941698526), UINT64_C( 5275727633078416591), UINT64_C( 6232978593844025102),
        UINT64_C( 7442735538344824408), UINT64_C( 1357612375565516658), UINT64_C( 7262009690649735210), UINT64_C(14701656238056765358) },
      UINT8_C(212),
      { UINT64_C(14238345752624348222), UINT64_C(15881130087530681606), UINT64_C( 2468139358930330810), UINT64_C( 3709105674812583902),
        UINT64_C(11667404532817254175), UINT64_C(14434109435563340743), UINT64_C(18011190249750430502), UINT64_C(  986877581458312073) },
      { UINT64_C(14279540026880756974), UINT64_C( 8207680901140318224), UINT64_C( 7214829116955246861), UINT64_C(14176934525860560579),
        UINT64_C( 2263193020303077794), UINT64_C( 4125056658046886530), UINT64_C( 9053785262858994278), UINT64_C(13968737911998155071) },
      { UINT64_C( 6462801357531142997), UINT64_C(11165695382941698526), UINT64_C( 2468139358930330810), UINT64_C( 6232978593844025102),
        UINT64_C( 2263193020303077794), UINT64_C( 1357612375565516658), UINT64_C( 9053785262858994278), UINT64_C(  986877581458312073) } },
    { { UINT64_C( 2670014054728411358), UINT64_C(15479489228914707309), UINT64_C(17576630924552385204), UINT64_C(16584191199209331269),
        UINT64_C(13966472048727525643), UINT64_C(12281647096496047403), UINT64_C( 9945361146332153950), UINT64_C(18090139399423462687) },
      UINT8_C(158),
      { UINT64_C( 4678311457047357664), UINT64_C( 9410238406936922125), UINT64_C(13555361740489206266), UINT64_C( 3743519687341524288),
        UINT64_C(11892564712869913935), UINT64_C( 2081918817811710312), UINT64_C( 4434420541052136223), UINT64_C( 3171808324837586559) },
      { UINT64_C( 4731314915132307563), UINT64_C( 2125848550798755906), UINT64_C( 6083366755163151154), UINT64_C(17344816288595574443),
        UINT64_C(17311165913264073762), UINT64_C(11115399998573174821), UINT64_C( 5591559383428967536), UINT64_C(  840075166449319732) },
      { UINT64_C( 2670014054728411358), UINT64_C( 2125848550798755906), UINT64_C( 6083366755163151154), UINT64_C( 3743519687341524288),
        UINT64_C(11892564712869913935), UINT64_C(12281647096496047403), UINT64_C( 9945361146332153950), UINT64_C(  840075166449319732) } },
    { { UINT64_C( 6905219148390308919), UINT64_C( 7937529465821931812), UINT64_C( 6548318686262128880), UINT64_C( 8647398651486975500),
        UINT64_C( 8138340206561200215), UINT64_C( 2938075631335601242), UINT64_C(15318039516875029012), UINT64_C(13333693271013762897) },
      UINT8_C(211),
      { UINT64_C(12750129510458805094), UINT64_C( 9999416211519588748), UINT64_C( 4302951487301156811), UINT64_C(13443058330370918897),
        UINT64_C(14595395900362829473), UINT64_C( 5847712488547317132), UINT64_C(10452414521711032639), UINT64_C(11801361770630458297) },
      { UINT64_C(16686415356729759882), UINT64_C(10041808137218165891), UINT64_C( 4275724205147689764), UINT64_C(  845285298388116503),
        UINT64_C(14801473021009935398), UINT64_C(13567149509055081841), UINT64_C( 2955443355653350981), UINT64_C( 9225620924617870204) },
      { UINT64_C(12750129510458805094), UINT64_C( 9999416211519588748), UINT64_C( 6548318686262128880), UINT64_C( 8647398651486975500),
        UINT64_C(14595395900362829473), UINT64_C( 2938075631335601242), UINT64_C( 2955443355653350981), UINT64_C( 9225620924617870204) } },
    { { UINT64_C(12777110438840228409), UINT64_C(11505772749216688215), UINT64_C( 9338610774410931549), UINT64_C( 1337153306673208244),
        UINT64_C( 5859438178814300000), UINT64_C(15206665234644320015), UINT64_C(10133624556884291098), UINT64_C(  406494557947699128) },
      UINT8_C(184),
      { UINT64_C(12983882077076366394), UINT64_C( 1102534877555366212), UINT64_C( 3302954976424717377), UINT64_C( 5111540549564917774),
        UINT64_C(16799719242515063163), UINT64_C(10223283634242735664), UINT64_C( 5336063231971281106), UINT64_C( 3547553062589685737) },
      { UINT64_C( 7611019846458974737), UINT64_C(17504958627519838485), UINT64_C(  318030552002370019), UINT64_C( 7901061263188945854),
        UINT64_C( 2556569104159033559), UINT64_C(12938405019769943419), UINT64_C(18314515981370810379), UINT64_C( 3526064901190787878) },
      { UINT64_C(12777110438840228409), UINT64_C(11505772749216688215), UINT64_C( 9338610774410931549), UINT64_C( 5111540549564917774),
        UINT64_C( 2556569104159033559), UINT64_C(10223283634242735664), UINT64_C(10133624556884291098), UINT64_C( 3526064901190787878) } },
    { { UINT64_C( 5734133271646769761), UINT64_C(10805113565693548400), UINT64_C( 2817253359809709529), UINT64_C(  401269673593244425),
        UINT64_C(18360621039424426666), UINT64_C(  810140176142547231), UINT64_C(16726437048221039352), UINT64_C(  968796610631486152) },
      UINT8_C(174),
      { UINT64_C( 1300478550309899237), UINT64_C(17123900690666927481), UINT64_C( 9762314968632237429), UINT64_C( 4362289670808856319),
        UINT64_C( 8786019296073074050), UINT64_C(17927644309883270122), UINT64_C( 5596606879962404966), UINT64_C( 1053193104358151434) },
      { UINT64_C(16712816420545283958), UINT64_C( 4866017203671340044), UINT64_C( 6900659587811520809), UINT64_C( 1531608282320399782),
        UINT64_C( 1155307462581758756), UINT64_C( 9704736429709446542), UINT64_C(14703058879847928919), UINT64_C( 5576089834784541615) },
      { UINT64_C( 5734133271646769761), UINT64_C( 4866017203671340044), UINT64_C( 6900659587811520809), UINT64_C( 1531608282320399782),
        UINT64_C(18360621039424426666), UINT64_C( 9704736429709446542), UINT64_C(16726437048221039352), UINT64_C( 1053193104358151434) } },
    { { UINT64_C(17690269522951335275), UINT64_C(11396093078602260547), UINT64_C(10814645631567144227), UINT64_C(13877919579589776417),
        UINT64_C(13875784505449514547), UINT64_C( 2501056896776139216), UINT64_C( 8587099319091068846), UINT64_C( 9847781756449656469) },
      UINT8_C( 96),
      { UINT64_C(11843233039555306458), UINT64_C( 2753606095282139903), UINT64_C(  308764815373683506), UINT64_C(13145692697590837940),
        UINT64_C(16671372443939588868), UINT64_C(17896079815005372430), UINT64_C( 9181701415467997040), UINT64_C(12230082949249090598) },
      { UINT64_C(11823845224917039686), UINT64_C(11203606959026177468), UINT64_C( 4708070426801219340), UINT64_C( 2630100940438692657),
        UINT64_C( 6355538520778661880), UINT64_C( 5671253772179541486), UINT64_C( 6114334836932327038), UINT64_C( 1540417015589248862) },
      { UINT64_C(17690269522951335275), UINT64_C(11396093078602260547), UINT64_C(10814645631567144227), UINT64_C(13877919579589776417),
        UINT64_C(13875784505449514547), UINT64_C( 5671253772179541486), UINT64_C( 6114334836932327038), UINT64_C( 9847781756449656469) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_epu64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_epu64");
    easysimd_test_x86_assert_equal_u64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const uint64_t a[8];
    const uint64_t b[8];
    const uint64_t r[8];
  } test_vec[] = {
    { UINT8_C(137),
      { UINT64_C(16503860676597753011), UINT64_C(10173198177510302297), UINT64_C(13867996816985933723), UINT64_C(14223197835479593264),
        UINT64_C( 6797315046681642903), UINT64_C(10119936847272377100), UINT64_C( 6297465112980759043), UINT64_C(14817004633568920265) },
      { UINT64_C( 8610479422572363803), UINT64_C(14666477725613198348), UINT64_C(  895323810527390957), UINT64_C( 5483562667113684247),
        UINT64_C(17575837300568118792), UINT64_C(18233238772285260918), UINT64_C( 7275141121168275622), UINT64_C( 7897565958014819868) },
      { UINT64_C( 8610479422572363803), UINT64_C(                   0), UINT64_C(                   0), UINT64_C( 5483562667113684247),
        UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0), UINT64_C( 7897565958014819868) } },
    { UINT8_C( 32),
      { UINT64_C(  965453440346480645), UINT64_C( 2597480412835849195), UINT64_C(14683658851831641644), UINT64_C(12785893805021729670),
        UINT64_C( 6725287602638845822), UINT64_C(13292883976785195688), UINT64_C( 5079543337340562118), UINT64_C(14598988069481131580) },
      { UINT64_C(16810643562660248445), UINT64_C(11825809262842285785), UINT64_C(16548430274960523169), UINT64_C(13878519842170879363),
        UINT64_C(17063569526524652707), UINT64_C( 7100609541574822408), UINT64_C( 4214079208781242862), UINT64_C( 6172927327602362791) },
      { UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0),
        UINT64_C(                   0), UINT64_C( 7100609541574822408), UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C( 81),
      { UINT64_C( 1250682409119398232), UINT64_C(12537797938699714978), UINT64_C( 6487609060712814071), UINT64_C(17626352940028326568),
        UINT64_C(17704673040212099739), UINT64_C(12708320018572936675), UINT64_C(  506119906825191835), UINT64_C( 1196815613617705739) },
      { UINT64_C( 7510708422366651123), UINT64_C(16308506526666408682), UINT64_C( 9843625777144161333), UINT64_C(15217042879377567656),
        UINT64_C( 3502204316404015277), UINT64_C( 6611327300880544150), UINT64_C(10129030848468504459), UINT64_C( 4120185132418711930) },
      { UINT64_C( 1250682409119398232), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0),
        UINT64_C( 3502204316404015277), UINT64_C(                   0), UINT64_C(  506119906825191835), UINT64_C(                   0) } },
    { UINT8_C(182),
      { UINT64_C(13200380385403104070), UINT64_C( 2283950427686637567), UINT64_C(  949854131022183208), UINT64_C(18286254640890218808),
        UINT64_C(17076464433000661310), UINT64_C( 4163794530850822361), UINT64_C(10581800351259544448), UINT64_C( 9608541157854311332) },
      { UINT64_C(10520625028717473830), UINT64_C( 6405530274141770652), UINT64_C(10573043374034834126), UINT64_C(16662550890379303794),
        UINT64_C( 7386460613380802101), UINT64_C(10321679428363730189), UINT64_C(15153331130976395979), UINT64_C(13003238469991610829) },
      { UINT64_C(                   0), UINT64_C( 2283950427686637567), UINT64_C(  949854131022183208), UINT64_C(                   0),
        UINT64_C( 7386460613380802101), UINT64_C( 4163794530850822361), UINT64_C(                   0), UINT64_C( 9608541157854311332) } },
    { UINT8_C(207),
      { UINT64_C(16246454337152852847), UINT64_C(18218803880263834643), UINT64_C( 1452182024010633192), UINT64_C(10975630910976865722),
        UINT64_C(18370035455526890473), UINT64_C(12352213528684892629), UINT64_C(12780703332646111343), UINT64_C( 2424208970889818594) },
      { UINT64_C(17607137338204333680), UINT64_C(12834642397369754288), UINT64_C( 3620713026983568279), UINT64_C( 7540400133595034444),
        UINT64_C(11716990684039992199), UINT64_C( 7382300077774405502), UINT64_C( 1645842233799503701), UINT64_C( 9009238808538518695) },
      { UINT64_C(16246454337152852847), UINT64_C(12834642397369754288), UINT64_C( 1452182024010633192), UINT64_C( 7540400133595034444),
        UINT64_C(                   0), UINT64_C(                   0), UINT64_C( 1645842233799503701), UINT64_C( 2424208970889818594) } },
    { UINT8_C( 18),
      { UINT64_C(13719273708005369034), UINT64_C(  595023364407174586), UINT64_C(15186957891871631079), UINT64_C( 3662347238212395395),
        UINT64_C( 4100101387752888169), UINT64_C( 5495067080298623906), UINT64_C( 9357400296842007884), UINT64_C( 5990938598357247114) },
      { UINT64_C( 6603230027166989549), UINT64_C( 8829263283251169237), UINT64_C(11703835191775915757), UINT64_C(12844470392711392772),
        UINT64_C(17513021339003108598), UINT64_C( 7219187153931968391), UINT64_C( 1683707529127768995), UINT64_C(11736457234736280170) },
      { UINT64_C(                   0), UINT64_C(  595023364407174586), UINT64_C(                   0), UINT64_C(                   0),
        UINT64_C( 4100101387752888169), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C(  1),
      { UINT64_C(18033840468099289597), UINT64_C(15895408885510634047), UINT64_C(11694448209877226022), UINT64_C( 6363850544142776842),
        UINT64_C(13949436408542621505), UINT64_C(  385423765887322196), UINT64_C( 6480453700705208478), UINT64_C( 7953067963763408300) },
      { UINT64_C( 2570325922406888552), UINT64_C( 5372225549456400553), UINT64_C( 5974737005137327170), UINT64_C(17609276354712867524),
        UINT64_C(13050456509037570859), UINT64_C(18005333824593903081), UINT64_C(13994783000903710271), UINT64_C( 8252305664839553031) },
      { UINT64_C( 2570325922406888552), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0),
        UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C(127),
      { UINT64_C( 5032281689989802852), UINT64_C(13946228895540603775), UINT64_C( 4033959350094425984), UINT64_C( 8775527843854454324),
        UINT64_C( 1452478593801540699), UINT64_C(11147790180098647208), UINT64_C( 2825598596790776408), UINT64_C(18421193228111885336) },
      { UINT64_C(17381092220628831373), UINT64_C(15412611488663508016), UINT64_C( 1451578536620225035), UINT64_C( 6656855502611388094),
        UINT64_C( 9236236667495269618), UINT64_C(13085963715764425032), UINT64_C( 3905140362312904224), UINT64_C( 5447660485473759854) },
      { UINT64_C( 5032281689989802852), UINT64_C(13946228895540603775), UINT64_C( 1451578536620225035), UINT64_C( 6656855502611388094),
        UINT64_C( 1452478593801540699), UINT64_C(11147790180098647208), UINT64_C( 2825598596790776408), UINT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_epu64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_epu64");
    easysimd_test_x86_assert_equal_u64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_min_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   -30.78), EASYSIMD_FLOAT32_C(   230.02), EASYSIMD_FLOAT32_C(   650.41), EASYSIMD_FLOAT32_C(  -352.49),
        EASYSIMD_FLOAT32_C(   -59.64), EASYSIMD_FLOAT32_C(   790.85), EASYSIMD_FLOAT32_C(   797.78), EASYSIMD_FLOAT32_C(  -606.64),
        EASYSIMD_FLOAT32_C(   -87.74), EASYSIMD_FLOAT32_C(   822.54), EASYSIMD_FLOAT32_C(  -957.59), EASYSIMD_FLOAT32_C(   103.00),
        EASYSIMD_FLOAT32_C(   800.32), EASYSIMD_FLOAT32_C(  -762.75), EASYSIMD_FLOAT32_C(   593.42), EASYSIMD_FLOAT32_C(  -283.97) },
      { EASYSIMD_FLOAT32_C(  -801.78), EASYSIMD_FLOAT32_C(   192.58), EASYSIMD_FLOAT32_C(  -152.69), EASYSIMD_FLOAT32_C(  -913.41),
        EASYSIMD_FLOAT32_C(    31.03), EASYSIMD_FLOAT32_C(   411.15), EASYSIMD_FLOAT32_C(  -317.27), EASYSIMD_FLOAT32_C(    57.37),
        EASYSIMD_FLOAT32_C(  -966.49), EASYSIMD_FLOAT32_C(   636.65), EASYSIMD_FLOAT32_C(    28.95), EASYSIMD_FLOAT32_C(   832.42),
        EASYSIMD_FLOAT32_C(  -321.67), EASYSIMD_FLOAT32_C(  -832.42), EASYSIMD_FLOAT32_C(  -939.05), EASYSIMD_FLOAT32_C(   647.55) },
      { EASYSIMD_FLOAT32_C(  -801.78), EASYSIMD_FLOAT32_C(   192.58), EASYSIMD_FLOAT32_C(  -152.69), EASYSIMD_FLOAT32_C(  -913.41),
        EASYSIMD_FLOAT32_C(   -59.64), EASYSIMD_FLOAT32_C(   411.15), EASYSIMD_FLOAT32_C(  -317.27), EASYSIMD_FLOAT32_C(  -606.64),
        EASYSIMD_FLOAT32_C(  -966.49), EASYSIMD_FLOAT32_C(   636.65), EASYSIMD_FLOAT32_C(  -957.59), EASYSIMD_FLOAT32_C(   103.00),
        EASYSIMD_FLOAT32_C(  -321.67), EASYSIMD_FLOAT32_C(  -832.42), EASYSIMD_FLOAT32_C(  -939.05), EASYSIMD_FLOAT32_C(  -283.97) } },
    { { EASYSIMD_FLOAT32_C(   397.60), EASYSIMD_FLOAT32_C(   711.36), EASYSIMD_FLOAT32_C(  -704.94), EASYSIMD_FLOAT32_C(  -662.04),
        EASYSIMD_FLOAT32_C(   502.21), EASYSIMD_FLOAT32_C(  -907.16), EASYSIMD_FLOAT32_C(  -268.68), EASYSIMD_FLOAT32_C(  -585.53),
        EASYSIMD_FLOAT32_C(   915.38), EASYSIMD_FLOAT32_C(  -226.27), EASYSIMD_FLOAT32_C(   517.47), EASYSIMD_FLOAT32_C(   715.70),
        EASYSIMD_FLOAT32_C(    10.98), EASYSIMD_FLOAT32_C(   110.89), EASYSIMD_FLOAT32_C(  -568.27), EASYSIMD_FLOAT32_C(   209.20) },
      { EASYSIMD_FLOAT32_C(  -696.52), EASYSIMD_FLOAT32_C(   279.04), EASYSIMD_FLOAT32_C(   295.79), EASYSIMD_FLOAT32_C(   334.51),
        EASYSIMD_FLOAT32_C(  -309.81), EASYSIMD_FLOAT32_C(   978.52), EASYSIMD_FLOAT32_C(  -608.12), EASYSIMD_FLOAT32_C(  -276.30),
        EASYSIMD_FLOAT32_C(   615.17), EASYSIMD_FLOAT32_C(   420.83), EASYSIMD_FLOAT32_C(  -443.88), EASYSIMD_FLOAT32_C(  -706.50),
        EASYSIMD_FLOAT32_C(   588.41), EASYSIMD_FLOAT32_C(  -382.93), EASYSIMD_FLOAT32_C(   941.05), EASYSIMD_FLOAT32_C(   -13.98) },
      { EASYSIMD_FLOAT32_C(  -696.52), EASYSIMD_FLOAT32_C(   279.04), EASYSIMD_FLOAT32_C(  -704.94), EASYSIMD_FLOAT32_C(  -662.04),
        EASYSIMD_FLOAT32_C(  -309.81), EASYSIMD_FLOAT32_C(  -907.16), EASYSIMD_FLOAT32_C(  -608.12), EASYSIMD_FLOAT32_C(  -585.53),
        EASYSIMD_FLOAT32_C(   615.17), EASYSIMD_FLOAT32_C(  -226.27), EASYSIMD_FLOAT32_C(  -443.88), EASYSIMD_FLOAT32_C(  -706.50),
        EASYSIMD_FLOAT32_C(    10.98), EASYSIMD_FLOAT32_C(  -382.93), EASYSIMD_FLOAT32_C(  -568.27), EASYSIMD_FLOAT32_C(   -13.98) } },
    { { EASYSIMD_FLOAT32_C(  -671.57), EASYSIMD_FLOAT32_C(  -763.89), EASYSIMD_FLOAT32_C(   323.97), EASYSIMD_FLOAT32_C(   830.64),
        EASYSIMD_FLOAT32_C(  -671.05), EASYSIMD_FLOAT32_C(  -944.70), EASYSIMD_FLOAT32_C(  -754.89), EASYSIMD_FLOAT32_C(  -755.67),
        EASYSIMD_FLOAT32_C(  -170.97), EASYSIMD_FLOAT32_C(   762.58), EASYSIMD_FLOAT32_C(   960.04), EASYSIMD_FLOAT32_C(   840.01),
        EASYSIMD_FLOAT32_C(  -126.53), EASYSIMD_FLOAT32_C(  -608.23), EASYSIMD_FLOAT32_C(    49.21), EASYSIMD_FLOAT32_C(   176.95) },
      { EASYSIMD_FLOAT32_C(   670.81), EASYSIMD_FLOAT32_C(  -655.00), EASYSIMD_FLOAT32_C(  -488.53), EASYSIMD_FLOAT32_C(  -639.00),
        EASYSIMD_FLOAT32_C(  -676.48), EASYSIMD_FLOAT32_C(   -96.65), EASYSIMD_FLOAT32_C(    84.71), EASYSIMD_FLOAT32_C(   938.70),
        EASYSIMD_FLOAT32_C(  -675.82), EASYSIMD_FLOAT32_C(   640.83), EASYSIMD_FLOAT32_C(  -767.81), EASYSIMD_FLOAT32_C(   912.60),
        EASYSIMD_FLOAT32_C(  -742.11), EASYSIMD_FLOAT32_C(  -826.76), EASYSIMD_FLOAT32_C(  -101.39), EASYSIMD_FLOAT32_C(  -413.68) },
      { EASYSIMD_FLOAT32_C(  -671.57), EASYSIMD_FLOAT32_C(  -763.89), EASYSIMD_FLOAT32_C(  -488.53), EASYSIMD_FLOAT32_C(  -639.00),
        EASYSIMD_FLOAT32_C(  -676.48), EASYSIMD_FLOAT32_C(  -944.70), EASYSIMD_FLOAT32_C(  -754.89), EASYSIMD_FLOAT32_C(  -755.67),
        EASYSIMD_FLOAT32_C(  -675.82), EASYSIMD_FLOAT32_C(   640.83), EASYSIMD_FLOAT32_C(  -767.81), EASYSIMD_FLOAT32_C(   840.01),
        EASYSIMD_FLOAT32_C(  -742.11), EASYSIMD_FLOAT32_C(  -826.76), EASYSIMD_FLOAT32_C(  -101.39), EASYSIMD_FLOAT32_C(  -413.68) } },
    { { EASYSIMD_FLOAT32_C(  -590.65), EASYSIMD_FLOAT32_C(  -777.42), EASYSIMD_FLOAT32_C(  -583.04), EASYSIMD_FLOAT32_C(  -261.70),
        EASYSIMD_FLOAT32_C(  -722.12), EASYSIMD_FLOAT32_C(  -337.93), EASYSIMD_FLOAT32_C(   -17.36), EASYSIMD_FLOAT32_C(   106.91),
        EASYSIMD_FLOAT32_C(  -575.35), EASYSIMD_FLOAT32_C(   -57.33), EASYSIMD_FLOAT32_C(   -53.08), EASYSIMD_FLOAT32_C(   298.13),
        EASYSIMD_FLOAT32_C(   334.44), EASYSIMD_FLOAT32_C(   996.13), EASYSIMD_FLOAT32_C(  -524.92), EASYSIMD_FLOAT32_C(     5.25) },
      { EASYSIMD_FLOAT32_C(  -658.87), EASYSIMD_FLOAT32_C(   -13.45), EASYSIMD_FLOAT32_C(   366.26), EASYSIMD_FLOAT32_C(  -335.35),
        EASYSIMD_FLOAT32_C(   889.90), EASYSIMD_FLOAT32_C(  -549.04), EASYSIMD_FLOAT32_C(  -396.65), EASYSIMD_FLOAT32_C(  -785.92),
        EASYSIMD_FLOAT32_C(  -908.21), EASYSIMD_FLOAT32_C(  -164.46), EASYSIMD_FLOAT32_C(  -873.33), EASYSIMD_FLOAT32_C(  -650.32),
        EASYSIMD_FLOAT32_C(     8.78), EASYSIMD_FLOAT32_C(    25.28), EASYSIMD_FLOAT32_C(   -63.99), EASYSIMD_FLOAT32_C(   418.13) },
      { EASYSIMD_FLOAT32_C(  -658.87), EASYSIMD_FLOAT32_C(  -777.42), EASYSIMD_FLOAT32_C(  -583.04), EASYSIMD_FLOAT32_C(  -335.35),
        EASYSIMD_FLOAT32_C(  -722.12), EASYSIMD_FLOAT32_C(  -549.04), EASYSIMD_FLOAT32_C(  -396.65), EASYSIMD_FLOAT32_C(  -785.92),
        EASYSIMD_FLOAT32_C(  -908.21), EASYSIMD_FLOAT32_C(  -164.46), EASYSIMD_FLOAT32_C(  -873.33), EASYSIMD_FLOAT32_C(  -650.32),
        EASYSIMD_FLOAT32_C(     8.78), EASYSIMD_FLOAT32_C(    25.28), EASYSIMD_FLOAT32_C(  -524.92), EASYSIMD_FLOAT32_C(     5.25) } },
    { { EASYSIMD_FLOAT32_C(   247.87), EASYSIMD_FLOAT32_C(   352.97), EASYSIMD_FLOAT32_C(  -843.57), EASYSIMD_FLOAT32_C(   525.75),
        EASYSIMD_FLOAT32_C(  -984.96), EASYSIMD_FLOAT32_C(   139.07), EASYSIMD_FLOAT32_C(  -367.35), EASYSIMD_FLOAT32_C(  -560.31),
        EASYSIMD_FLOAT32_C(  -918.25), EASYSIMD_FLOAT32_C(   579.57), EASYSIMD_FLOAT32_C(   737.82), EASYSIMD_FLOAT32_C(   416.19),
        EASYSIMD_FLOAT32_C(   575.70), EASYSIMD_FLOAT32_C(  -787.10), EASYSIMD_FLOAT32_C(  -578.56), EASYSIMD_FLOAT32_C(   916.82) },
      { EASYSIMD_FLOAT32_C(   199.45), EASYSIMD_FLOAT32_C(   787.70), EASYSIMD_FLOAT32_C(  -418.53), EASYSIMD_FLOAT32_C(    89.35),
        EASYSIMD_FLOAT32_C(  -761.34), EASYSIMD_FLOAT32_C(   184.82), EASYSIMD_FLOAT32_C(   303.43), EASYSIMD_FLOAT32_C(  -669.55),
        EASYSIMD_FLOAT32_C(  -979.64), EASYSIMD_FLOAT32_C(   430.10), EASYSIMD_FLOAT32_C(  -319.87), EASYSIMD_FLOAT32_C(    29.14),
        EASYSIMD_FLOAT32_C(  -544.62), EASYSIMD_FLOAT32_C(   616.14), EASYSIMD_FLOAT32_C(  -552.73), EASYSIMD_FLOAT32_C(   703.25) },
      { EASYSIMD_FLOAT32_C(   199.45), EASYSIMD_FLOAT32_C(   352.97), EASYSIMD_FLOAT32_C(  -843.57), EASYSIMD_FLOAT32_C(    89.35),
        EASYSIMD_FLOAT32_C(  -984.96), EASYSIMD_FLOAT32_C(   139.07), EASYSIMD_FLOAT32_C(  -367.35), EASYSIMD_FLOAT32_C(  -669.55),
        EASYSIMD_FLOAT32_C(  -979.64), EASYSIMD_FLOAT32_C(   430.10), EASYSIMD_FLOAT32_C(  -319.87), EASYSIMD_FLOAT32_C(    29.14),
        EASYSIMD_FLOAT32_C(  -544.62), EASYSIMD_FLOAT32_C(  -787.10), EASYSIMD_FLOAT32_C(  -578.56), EASYSIMD_FLOAT32_C(   703.25) } },
    { { EASYSIMD_FLOAT32_C(   -30.90), EASYSIMD_FLOAT32_C(  -396.30), EASYSIMD_FLOAT32_C(   229.00), EASYSIMD_FLOAT32_C(   -15.86),
        EASYSIMD_FLOAT32_C(   742.77), EASYSIMD_FLOAT32_C(   861.65), EASYSIMD_FLOAT32_C(   423.84), EASYSIMD_FLOAT32_C(   824.52),
        EASYSIMD_FLOAT32_C(   441.22), EASYSIMD_FLOAT32_C(   161.66), EASYSIMD_FLOAT32_C(   240.71), EASYSIMD_FLOAT32_C(    16.92),
        EASYSIMD_FLOAT32_C(   374.56), EASYSIMD_FLOAT32_C(   662.15), EASYSIMD_FLOAT32_C(   -66.25), EASYSIMD_FLOAT32_C(  -425.99) },
      { EASYSIMD_FLOAT32_C(   449.85), EASYSIMD_FLOAT32_C(   515.22), EASYSIMD_FLOAT32_C(   663.36), EASYSIMD_FLOAT32_C(   688.52),
        EASYSIMD_FLOAT32_C(  -299.96), EASYSIMD_FLOAT32_C(   -33.22), EASYSIMD_FLOAT32_C(  -981.03), EASYSIMD_FLOAT32_C(  -279.61),
        EASYSIMD_FLOAT32_C(  -603.12), EASYSIMD_FLOAT32_C(  -300.90), EASYSIMD_FLOAT32_C(   749.53), EASYSIMD_FLOAT32_C(  -147.73),
        EASYSIMD_FLOAT32_C(  -684.77), EASYSIMD_FLOAT32_C(  -803.20), EASYSIMD_FLOAT32_C(  -444.48), EASYSIMD_FLOAT32_C(   284.34) },
      { EASYSIMD_FLOAT32_C(   -30.90), EASYSIMD_FLOAT32_C(  -396.30), EASYSIMD_FLOAT32_C(   229.00), EASYSIMD_FLOAT32_C(   -15.86),
        EASYSIMD_FLOAT32_C(  -299.96), EASYSIMD_FLOAT32_C(   -33.22), EASYSIMD_FLOAT32_C(  -981.03), EASYSIMD_FLOAT32_C(  -279.61),
        EASYSIMD_FLOAT32_C(  -603.12), EASYSIMD_FLOAT32_C(  -300.90), EASYSIMD_FLOAT32_C(   240.71), EASYSIMD_FLOAT32_C(  -147.73),
        EASYSIMD_FLOAT32_C(  -684.77), EASYSIMD_FLOAT32_C(  -803.20), EASYSIMD_FLOAT32_C(  -444.48), EASYSIMD_FLOAT32_C(  -425.99) } },
    { { EASYSIMD_FLOAT32_C(  -199.50), EASYSIMD_FLOAT32_C(   784.52), EASYSIMD_FLOAT32_C(  -731.52), EASYSIMD_FLOAT32_C(  -456.72),
        EASYSIMD_FLOAT32_C(   646.17), EASYSIMD_FLOAT32_C(   692.32), EASYSIMD_FLOAT32_C(  -632.20), EASYSIMD_FLOAT32_C(    87.40),
        EASYSIMD_FLOAT32_C(  -146.02), EASYSIMD_FLOAT32_C(   608.51), EASYSIMD_FLOAT32_C(  -895.68), EASYSIMD_FLOAT32_C(  -771.46),
        EASYSIMD_FLOAT32_C(   270.66), EASYSIMD_FLOAT32_C(    38.06), EASYSIMD_FLOAT32_C(  -197.45), EASYSIMD_FLOAT32_C(  -279.49) },
      { EASYSIMD_FLOAT32_C(  -446.72), EASYSIMD_FLOAT32_C(  -534.09), EASYSIMD_FLOAT32_C(  -590.97), EASYSIMD_FLOAT32_C(   253.32),
        EASYSIMD_FLOAT32_C(   432.69), EASYSIMD_FLOAT32_C(  -572.00), EASYSIMD_FLOAT32_C(   973.71), EASYSIMD_FLOAT32_C(   829.57),
        EASYSIMD_FLOAT32_C(   127.09), EASYSIMD_FLOAT32_C(   723.24), EASYSIMD_FLOAT32_C(  -318.16), EASYSIMD_FLOAT32_C(   442.33),
        EASYSIMD_FLOAT32_C(   920.04), EASYSIMD_FLOAT32_C(   237.36), EASYSIMD_FLOAT32_C(  -273.33), EASYSIMD_FLOAT32_C(  -279.46) },
      { EASYSIMD_FLOAT32_C(  -446.72), EASYSIMD_FLOAT32_C(  -534.09), EASYSIMD_FLOAT32_C(  -731.52), EASYSIMD_FLOAT32_C(  -456.72),
        EASYSIMD_FLOAT32_C(   432.69), EASYSIMD_FLOAT32_C(  -572.00), EASYSIMD_FLOAT32_C(  -632.20), EASYSIMD_FLOAT32_C(    87.40),
        EASYSIMD_FLOAT32_C(  -146.02), EASYSIMD_FLOAT32_C(   608.51), EASYSIMD_FLOAT32_C(  -895.68), EASYSIMD_FLOAT32_C(  -771.46),
        EASYSIMD_FLOAT32_C(   270.66), EASYSIMD_FLOAT32_C(    38.06), EASYSIMD_FLOAT32_C(  -273.33), EASYSIMD_FLOAT32_C(  -279.49) } },
    { { EASYSIMD_FLOAT32_C(    21.88), EASYSIMD_FLOAT32_C(    -4.85), EASYSIMD_FLOAT32_C(   263.82), EASYSIMD_FLOAT32_C(  -331.95),
        EASYSIMD_FLOAT32_C(  -312.53), EASYSIMD_FLOAT32_C(   631.61), EASYSIMD_FLOAT32_C(   755.44), EASYSIMD_FLOAT32_C(   541.44),
        EASYSIMD_FLOAT32_C(   240.12), EASYSIMD_FLOAT32_C(   859.76), EASYSIMD_FLOAT32_C(   769.98), EASYSIMD_FLOAT32_C(  -489.22),
        EASYSIMD_FLOAT32_C(  -102.18), EASYSIMD_FLOAT32_C(  -427.47), EASYSIMD_FLOAT32_C(   231.29), EASYSIMD_FLOAT32_C(   451.10) },
      { EASYSIMD_FLOAT32_C(    38.43), EASYSIMD_FLOAT32_C(   640.32), EASYSIMD_FLOAT32_C(  -295.58), EASYSIMD_FLOAT32_C(  -528.88),
        EASYSIMD_FLOAT32_C(  -931.68), EASYSIMD_FLOAT32_C(  -321.87), EASYSIMD_FLOAT32_C(  -699.30), EASYSIMD_FLOAT32_C(   195.41),
        EASYSIMD_FLOAT32_C(  -598.63), EASYSIMD_FLOAT32_C(   -17.46), EASYSIMD_FLOAT32_C(  -362.26), EASYSIMD_FLOAT32_C(  -678.60),
        EASYSIMD_FLOAT32_C(  -780.10), EASYSIMD_FLOAT32_C(   364.41), EASYSIMD_FLOAT32_C(    41.95), EASYSIMD_FLOAT32_C(   241.77) },
      { EASYSIMD_FLOAT32_C(    21.88), EASYSIMD_FLOAT32_C(    -4.85), EASYSIMD_FLOAT32_C(  -295.58), EASYSIMD_FLOAT32_C(  -528.88),
        EASYSIMD_FLOAT32_C(  -931.68), EASYSIMD_FLOAT32_C(  -321.87), EASYSIMD_FLOAT32_C(  -699.30), EASYSIMD_FLOAT32_C(   195.41),
        EASYSIMD_FLOAT32_C(  -598.63), EASYSIMD_FLOAT32_C(   -17.46), EASYSIMD_FLOAT32_C(  -362.26), EASYSIMD_FLOAT32_C(  -678.60),
        EASYSIMD_FLOAT32_C(  -780.10), EASYSIMD_FLOAT32_C(  -427.47), EASYSIMD_FLOAT32_C(    41.95), EASYSIMD_FLOAT32_C(   241.77) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_min_ps(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_min_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 src[16];
    const easysimd__mmask8 k;
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   -36.57), EASYSIMD_FLOAT32_C(    69.10), EASYSIMD_FLOAT32_C(  -983.85), EASYSIMD_FLOAT32_C(  -248.71),
        EASYSIMD_FLOAT32_C(   274.57), EASYSIMD_FLOAT32_C(   653.48), EASYSIMD_FLOAT32_C(   360.57), EASYSIMD_FLOAT32_C(   -54.84),
        EASYSIMD_FLOAT32_C(   250.76), EASYSIMD_FLOAT32_C(  -841.58), EASYSIMD_FLOAT32_C(  -977.10), EASYSIMD_FLOAT32_C(  -197.41),
        EASYSIMD_FLOAT32_C(  -411.19), EASYSIMD_FLOAT32_C(   191.35), EASYSIMD_FLOAT32_C(   -80.14), EASYSIMD_FLOAT32_C(   640.19) },
      UINT8_C( 32),
      { EASYSIMD_FLOAT32_C(   655.14), EASYSIMD_FLOAT32_C(   247.62), EASYSIMD_FLOAT32_C(  -329.20), EASYSIMD_FLOAT32_C(  -310.43),
        EASYSIMD_FLOAT32_C(   -42.22), EASYSIMD_FLOAT32_C(  -251.97), EASYSIMD_FLOAT32_C(  -986.50), EASYSIMD_FLOAT32_C(   336.60),
        EASYSIMD_FLOAT32_C(  -919.96), EASYSIMD_FLOAT32_C(  -555.47), EASYSIMD_FLOAT32_C(    -1.94), EASYSIMD_FLOAT32_C(  -858.96),
        EASYSIMD_FLOAT32_C(  -877.47), EASYSIMD_FLOAT32_C(  -290.85), EASYSIMD_FLOAT32_C(   104.46), EASYSIMD_FLOAT32_C(   191.63) },
      { EASYSIMD_FLOAT32_C(  -274.70), EASYSIMD_FLOAT32_C(   855.75), EASYSIMD_FLOAT32_C(  -533.80), EASYSIMD_FLOAT32_C(  -621.22),
        EASYSIMD_FLOAT32_C(   216.32), EASYSIMD_FLOAT32_C(   411.35), EASYSIMD_FLOAT32_C(   629.54), EASYSIMD_FLOAT32_C(   374.74),
        EASYSIMD_FLOAT32_C(   434.26), EASYSIMD_FLOAT32_C(  -567.87), EASYSIMD_FLOAT32_C(   963.55), EASYSIMD_FLOAT32_C(  -374.39),
        EASYSIMD_FLOAT32_C(   351.98), EASYSIMD_FLOAT32_C(   603.74), EASYSIMD_FLOAT32_C(   320.68), EASYSIMD_FLOAT32_C(     7.12) },
      { EASYSIMD_FLOAT32_C(   -36.57), EASYSIMD_FLOAT32_C(    69.10), EASYSIMD_FLOAT32_C(  -983.85), EASYSIMD_FLOAT32_C(  -248.71),
        EASYSIMD_FLOAT32_C(   274.57), EASYSIMD_FLOAT32_C(  -251.97), EASYSIMD_FLOAT32_C(   360.57), EASYSIMD_FLOAT32_C(   -54.84),
        EASYSIMD_FLOAT32_C(   250.76), EASYSIMD_FLOAT32_C(  -841.58), EASYSIMD_FLOAT32_C(  -977.10), EASYSIMD_FLOAT32_C(  -197.41),
        EASYSIMD_FLOAT32_C(  -411.19), EASYSIMD_FLOAT32_C(   191.35), EASYSIMD_FLOAT32_C(   -80.14), EASYSIMD_FLOAT32_C(   640.19) } },
    { { EASYSIMD_FLOAT32_C(  -148.64), EASYSIMD_FLOAT32_C(   991.49), EASYSIMD_FLOAT32_C(   696.69), EASYSIMD_FLOAT32_C(   809.15),
        EASYSIMD_FLOAT32_C(  -260.48), EASYSIMD_FLOAT32_C(   710.19), EASYSIMD_FLOAT32_C(   145.75), EASYSIMD_FLOAT32_C(  -180.44),
        EASYSIMD_FLOAT32_C(  -845.29), EASYSIMD_FLOAT32_C(  -856.19), EASYSIMD_FLOAT32_C(   -39.40), EASYSIMD_FLOAT32_C(  -722.76),
        EASYSIMD_FLOAT32_C(  -147.04), EASYSIMD_FLOAT32_C(  -934.94), EASYSIMD_FLOAT32_C(   468.87), EASYSIMD_FLOAT32_C(   578.26) },
      UINT8_C(246),
      { EASYSIMD_FLOAT32_C(   935.06), EASYSIMD_FLOAT32_C(   957.03), EASYSIMD_FLOAT32_C(   137.14), EASYSIMD_FLOAT32_C(   346.42),
        EASYSIMD_FLOAT32_C(   586.58), EASYSIMD_FLOAT32_C(  -488.13), EASYSIMD_FLOAT32_C(  -219.32), EASYSIMD_FLOAT32_C(  -981.30),
        EASYSIMD_FLOAT32_C(  -524.58), EASYSIMD_FLOAT32_C(   406.29), EASYSIMD_FLOAT32_C(   370.69), EASYSIMD_FLOAT32_C(  -920.84),
        EASYSIMD_FLOAT32_C(  -273.03), EASYSIMD_FLOAT32_C(  -622.19), EASYSIMD_FLOAT32_C(   -69.48), EASYSIMD_FLOAT32_C(  -281.54) },
      { EASYSIMD_FLOAT32_C(  -925.50), EASYSIMD_FLOAT32_C(  -260.33), EASYSIMD_FLOAT32_C(   457.98), EASYSIMD_FLOAT32_C(   784.68),
        EASYSIMD_FLOAT32_C(   885.42), EASYSIMD_FLOAT32_C(  -722.47), EASYSIMD_FLOAT32_C(   939.40), EASYSIMD_FLOAT32_C(  -970.77),
        EASYSIMD_FLOAT32_C(   238.13), EASYSIMD_FLOAT32_C(  -783.36), EASYSIMD_FLOAT32_C(  -117.81), EASYSIMD_FLOAT32_C(   303.19),
        EASYSIMD_FLOAT32_C(   685.51), EASYSIMD_FLOAT32_C(  -539.55), EASYSIMD_FLOAT32_C(   224.00), EASYSIMD_FLOAT32_C(   620.57) },
      { EASYSIMD_FLOAT32_C(  -148.64), EASYSIMD_FLOAT32_C(  -260.33), EASYSIMD_FLOAT32_C(   137.14), EASYSIMD_FLOAT32_C(   809.15),
        EASYSIMD_FLOAT32_C(   586.58), EASYSIMD_FLOAT32_C(  -722.47), EASYSIMD_FLOAT32_C(  -219.32), EASYSIMD_FLOAT32_C(  -981.30),
        EASYSIMD_FLOAT32_C(  -845.29), EASYSIMD_FLOAT32_C(  -856.19), EASYSIMD_FLOAT32_C(   -39.40), EASYSIMD_FLOAT32_C(  -722.76),
        EASYSIMD_FLOAT32_C(  -147.04), EASYSIMD_FLOAT32_C(  -934.94), EASYSIMD_FLOAT32_C(   468.87), EASYSIMD_FLOAT32_C(   578.26) } },
    { { EASYSIMD_FLOAT32_C(  -582.52), EASYSIMD_FLOAT32_C(  -638.86), EASYSIMD_FLOAT32_C(   -33.01), EASYSIMD_FLOAT32_C(  -995.94),
        EASYSIMD_FLOAT32_C(  -126.99), EASYSIMD_FLOAT32_C(   747.67), EASYSIMD_FLOAT32_C(  -977.24), EASYSIMD_FLOAT32_C(   348.44),
        EASYSIMD_FLOAT32_C(   153.95), EASYSIMD_FLOAT32_C(   393.45), EASYSIMD_FLOAT32_C(   427.60), EASYSIMD_FLOAT32_C(   880.92),
        EASYSIMD_FLOAT32_C(   771.26), EASYSIMD_FLOAT32_C(  -641.88), EASYSIMD_FLOAT32_C(  -400.62), EASYSIMD_FLOAT32_C(   845.75) },
         UINT8_MAX,
      { EASYSIMD_FLOAT32_C(  -942.64), EASYSIMD_FLOAT32_C(   630.44), EASYSIMD_FLOAT32_C(   -16.79), EASYSIMD_FLOAT32_C(  -665.11),
        EASYSIMD_FLOAT32_C(   569.83), EASYSIMD_FLOAT32_C(    12.44), EASYSIMD_FLOAT32_C(   573.02), EASYSIMD_FLOAT32_C(   786.47),
        EASYSIMD_FLOAT32_C(   894.64), EASYSIMD_FLOAT32_C(  -123.79), EASYSIMD_FLOAT32_C(   471.98), EASYSIMD_FLOAT32_C(  -644.91),
        EASYSIMD_FLOAT32_C(  -899.79), EASYSIMD_FLOAT32_C(    92.56), EASYSIMD_FLOAT32_C(  -227.43), EASYSIMD_FLOAT32_C(  -538.65) },
      { EASYSIMD_FLOAT32_C(  -940.45), EASYSIMD_FLOAT32_C(  -223.37), EASYSIMD_FLOAT32_C(   334.37), EASYSIMD_FLOAT32_C(   807.22),
        EASYSIMD_FLOAT32_C(  -200.61), EASYSIMD_FLOAT32_C(  -317.20), EASYSIMD_FLOAT32_C(   -38.83), EASYSIMD_FLOAT32_C(  -807.16),
        EASYSIMD_FLOAT32_C(  -889.60), EASYSIMD_FLOAT32_C(  -157.90), EASYSIMD_FLOAT32_C(   964.10), EASYSIMD_FLOAT32_C(  -531.48),
        EASYSIMD_FLOAT32_C(   441.48), EASYSIMD_FLOAT32_C(   809.85), EASYSIMD_FLOAT32_C(   566.31), EASYSIMD_FLOAT32_C(   498.84) },
      { EASYSIMD_FLOAT32_C(  -942.64), EASYSIMD_FLOAT32_C(  -223.37), EASYSIMD_FLOAT32_C(   -16.79), EASYSIMD_FLOAT32_C(  -665.11),
        EASYSIMD_FLOAT32_C(  -200.61), EASYSIMD_FLOAT32_C(  -317.20), EASYSIMD_FLOAT32_C(   -38.83), EASYSIMD_FLOAT32_C(  -807.16),
        EASYSIMD_FLOAT32_C(   153.95), EASYSIMD_FLOAT32_C(   393.45), EASYSIMD_FLOAT32_C(   427.60), EASYSIMD_FLOAT32_C(   880.92),
        EASYSIMD_FLOAT32_C(   771.26), EASYSIMD_FLOAT32_C(  -641.88), EASYSIMD_FLOAT32_C(  -400.62), EASYSIMD_FLOAT32_C(   845.75) } },
    { { EASYSIMD_FLOAT32_C(   440.29), EASYSIMD_FLOAT32_C(  -450.48), EASYSIMD_FLOAT32_C(   833.73), EASYSIMD_FLOAT32_C(    10.12),
        EASYSIMD_FLOAT32_C(   561.96), EASYSIMD_FLOAT32_C(   406.75), EASYSIMD_FLOAT32_C(  -203.40), EASYSIMD_FLOAT32_C(   456.60),
        EASYSIMD_FLOAT32_C(  -717.04), EASYSIMD_FLOAT32_C(  -731.42), EASYSIMD_FLOAT32_C(   811.69), EASYSIMD_FLOAT32_C(  -616.83),
        EASYSIMD_FLOAT32_C(   361.14), EASYSIMD_FLOAT32_C(  -415.74), EASYSIMD_FLOAT32_C(  -155.48), EASYSIMD_FLOAT32_C(   420.69) },
      UINT8_C(150),
      { EASYSIMD_FLOAT32_C(  -821.11), EASYSIMD_FLOAT32_C(   227.91), EASYSIMD_FLOAT32_C(  -839.72), EASYSIMD_FLOAT32_C(  -138.31),
        EASYSIMD_FLOAT32_C(  -810.92), EASYSIMD_FLOAT32_C(  -646.88), EASYSIMD_FLOAT32_C(   -27.91), EASYSIMD_FLOAT32_C(    31.18),
        EASYSIMD_FLOAT32_C(  -682.77), EASYSIMD_FLOAT32_C(   440.61), EASYSIMD_FLOAT32_C(  -527.34), EASYSIMD_FLOAT32_C(  -872.92),
        EASYSIMD_FLOAT32_C(     6.92), EASYSIMD_FLOAT32_C(   971.50), EASYSIMD_FLOAT32_C(   567.37), EASYSIMD_FLOAT32_C(   556.44) },
      { EASYSIMD_FLOAT32_C(   805.23), EASYSIMD_FLOAT32_C(  -422.50), EASYSIMD_FLOAT32_C(   118.40), EASYSIMD_FLOAT32_C(   211.98),
        EASYSIMD_FLOAT32_C(   374.09), EASYSIMD_FLOAT32_C(  -425.00), EASYSIMD_FLOAT32_C(   494.94), EASYSIMD_FLOAT32_C(   642.68),
        EASYSIMD_FLOAT32_C(  -613.31), EASYSIMD_FLOAT32_C(   878.11), EASYSIMD_FLOAT32_C(     3.82), EASYSIMD_FLOAT32_C(   -29.05),
        EASYSIMD_FLOAT32_C(  -277.36), EASYSIMD_FLOAT32_C(  -575.49), EASYSIMD_FLOAT32_C(  -668.17), EASYSIMD_FLOAT32_C(   -98.47) },
      { EASYSIMD_FLOAT32_C(   440.29), EASYSIMD_FLOAT32_C(  -422.50), EASYSIMD_FLOAT32_C(  -839.72), EASYSIMD_FLOAT32_C(    10.12),
        EASYSIMD_FLOAT32_C(  -810.92), EASYSIMD_FLOAT32_C(   406.75), EASYSIMD_FLOAT32_C(  -203.40), EASYSIMD_FLOAT32_C(    31.18),
        EASYSIMD_FLOAT32_C(  -717.04), EASYSIMD_FLOAT32_C(  -731.42), EASYSIMD_FLOAT32_C(   811.69), EASYSIMD_FLOAT32_C(  -616.83),
        EASYSIMD_FLOAT32_C(   361.14), EASYSIMD_FLOAT32_C(  -415.74), EASYSIMD_FLOAT32_C(  -155.48), EASYSIMD_FLOAT32_C(   420.69) } },
    { { EASYSIMD_FLOAT32_C(   652.42), EASYSIMD_FLOAT32_C(  -507.89), EASYSIMD_FLOAT32_C(   763.22), EASYSIMD_FLOAT32_C(   841.50),
        EASYSIMD_FLOAT32_C(  -154.77), EASYSIMD_FLOAT32_C(  -264.68), EASYSIMD_FLOAT32_C(  -127.32), EASYSIMD_FLOAT32_C(   162.46),
        EASYSIMD_FLOAT32_C(  -824.07), EASYSIMD_FLOAT32_C(   345.34), EASYSIMD_FLOAT32_C(   289.54), EASYSIMD_FLOAT32_C(   182.85),
        EASYSIMD_FLOAT32_C(   316.84), EASYSIMD_FLOAT32_C(  -143.09), EASYSIMD_FLOAT32_C(  -260.71), EASYSIMD_FLOAT32_C(   122.07) },
      UINT8_C( 26),
      { EASYSIMD_FLOAT32_C(   857.69), EASYSIMD_FLOAT32_C(  -665.95), EASYSIMD_FLOAT32_C(  -191.50), EASYSIMD_FLOAT32_C(  -567.30),
        EASYSIMD_FLOAT32_C(   828.99), EASYSIMD_FLOAT32_C(  -548.82), EASYSIMD_FLOAT32_C(  -180.61), EASYSIMD_FLOAT32_C(   707.10),
        EASYSIMD_FLOAT32_C(   455.00), EASYSIMD_FLOAT32_C(   790.33), EASYSIMD_FLOAT32_C(  -570.26), EASYSIMD_FLOAT32_C(   879.51),
        EASYSIMD_FLOAT32_C(  -877.84), EASYSIMD_FLOAT32_C(   331.27), EASYSIMD_FLOAT32_C(   531.93), EASYSIMD_FLOAT32_C(  -385.73) },
      { EASYSIMD_FLOAT32_C(    94.49), EASYSIMD_FLOAT32_C(   373.43), EASYSIMD_FLOAT32_C(   459.51), EASYSIMD_FLOAT32_C(   829.81),
        EASYSIMD_FLOAT32_C(  -753.89), EASYSIMD_FLOAT32_C(  -378.03), EASYSIMD_FLOAT32_C(  -994.27), EASYSIMD_FLOAT32_C(   591.46),
        EASYSIMD_FLOAT32_C(   911.50), EASYSIMD_FLOAT32_C(   188.58), EASYSIMD_FLOAT32_C(   -91.70), EASYSIMD_FLOAT32_C(  -231.58),
        EASYSIMD_FLOAT32_C(   927.87), EASYSIMD_FLOAT32_C(  -969.63), EASYSIMD_FLOAT32_C(  -797.18), EASYSIMD_FLOAT32_C(   785.57) },
      { EASYSIMD_FLOAT32_C(   652.42), EASYSIMD_FLOAT32_C(  -665.95), EASYSIMD_FLOAT32_C(   763.22), EASYSIMD_FLOAT32_C(  -567.30),
        EASYSIMD_FLOAT32_C(  -753.89), EASYSIMD_FLOAT32_C(  -264.68), EASYSIMD_FLOAT32_C(  -127.32), EASYSIMD_FLOAT32_C(   162.46),
        EASYSIMD_FLOAT32_C(  -824.07), EASYSIMD_FLOAT32_C(   345.34), EASYSIMD_FLOAT32_C(   289.54), EASYSIMD_FLOAT32_C(   182.85),
        EASYSIMD_FLOAT32_C(   316.84), EASYSIMD_FLOAT32_C(  -143.09), EASYSIMD_FLOAT32_C(  -260.71), EASYSIMD_FLOAT32_C(   122.07) } },
    { { EASYSIMD_FLOAT32_C(  -635.58), EASYSIMD_FLOAT32_C(    11.32), EASYSIMD_FLOAT32_C(  -781.74), EASYSIMD_FLOAT32_C(  -806.59),
        EASYSIMD_FLOAT32_C(   462.50), EASYSIMD_FLOAT32_C(    37.65), EASYSIMD_FLOAT32_C(   900.51), EASYSIMD_FLOAT32_C(   -82.50),
        EASYSIMD_FLOAT32_C(  -172.02), EASYSIMD_FLOAT32_C(  -669.76), EASYSIMD_FLOAT32_C(  -202.99), EASYSIMD_FLOAT32_C(   -49.85),
        EASYSIMD_FLOAT32_C(   661.51), EASYSIMD_FLOAT32_C(  -671.06), EASYSIMD_FLOAT32_C(   564.42), EASYSIMD_FLOAT32_C(  -244.00) },
      UINT8_C( 76),
      { EASYSIMD_FLOAT32_C(    23.92), EASYSIMD_FLOAT32_C(  -414.19), EASYSIMD_FLOAT32_C(   948.48), EASYSIMD_FLOAT32_C(   645.89),
        EASYSIMD_FLOAT32_C(  -408.46), EASYSIMD_FLOAT32_C(   539.94), EASYSIMD_FLOAT32_C(   557.39), EASYSIMD_FLOAT32_C(   780.12),
        EASYSIMD_FLOAT32_C(  -551.76), EASYSIMD_FLOAT32_C(  -674.19), EASYSIMD_FLOAT32_C(   708.00), EASYSIMD_FLOAT32_C(  -521.39),
        EASYSIMD_FLOAT32_C(  -471.37), EASYSIMD_FLOAT32_C(   493.56), EASYSIMD_FLOAT32_C(  -156.98), EASYSIMD_FLOAT32_C(   539.96) },
      { EASYSIMD_FLOAT32_C(   711.83), EASYSIMD_FLOAT32_C(    36.43), EASYSIMD_FLOAT32_C(     2.46), EASYSIMD_FLOAT32_C(  -250.52),
        EASYSIMD_FLOAT32_C(   -63.07), EASYSIMD_FLOAT32_C(   919.96), EASYSIMD_FLOAT32_C(   577.46), EASYSIMD_FLOAT32_C(   267.18),
        EASYSIMD_FLOAT32_C(  -283.03), EASYSIMD_FLOAT32_C(  -472.40), EASYSIMD_FLOAT32_C(   -71.31), EASYSIMD_FLOAT32_C(    45.91),
        EASYSIMD_FLOAT32_C(  -907.98), EASYSIMD_FLOAT32_C(   684.69), EASYSIMD_FLOAT32_C(  -251.73), EASYSIMD_FLOAT32_C(   115.95) },
      { EASYSIMD_FLOAT32_C(  -635.58), EASYSIMD_FLOAT32_C(    11.32), EASYSIMD_FLOAT32_C(     2.46), EASYSIMD_FLOAT32_C(  -250.52),
        EASYSIMD_FLOAT32_C(   462.50), EASYSIMD_FLOAT32_C(    37.65), EASYSIMD_FLOAT32_C(   557.39), EASYSIMD_FLOAT32_C(   -82.50),
        EASYSIMD_FLOAT32_C(  -172.02), EASYSIMD_FLOAT32_C(  -669.76), EASYSIMD_FLOAT32_C(  -202.99), EASYSIMD_FLOAT32_C(   -49.85),
        EASYSIMD_FLOAT32_C(   661.51), EASYSIMD_FLOAT32_C(  -671.06), EASYSIMD_FLOAT32_C(   564.42), EASYSIMD_FLOAT32_C(  -244.00) } },
    { { EASYSIMD_FLOAT32_C(  -729.51), EASYSIMD_FLOAT32_C(  -303.24), EASYSIMD_FLOAT32_C(  -238.16), EASYSIMD_FLOAT32_C(  -137.97),
        EASYSIMD_FLOAT32_C(  -763.30), EASYSIMD_FLOAT32_C(  -680.77), EASYSIMD_FLOAT32_C(  -357.84), EASYSIMD_FLOAT32_C(  -315.06),
        EASYSIMD_FLOAT32_C(  -354.96), EASYSIMD_FLOAT32_C(  -649.85), EASYSIMD_FLOAT32_C(   163.54), EASYSIMD_FLOAT32_C(   173.67),
        EASYSIMD_FLOAT32_C(   843.72), EASYSIMD_FLOAT32_C(  -993.43), EASYSIMD_FLOAT32_C(  -286.37), EASYSIMD_FLOAT32_C(   555.54) },
      UINT8_C(138),
      { EASYSIMD_FLOAT32_C(   716.09), EASYSIMD_FLOAT32_C(  -694.98), EASYSIMD_FLOAT32_C(   979.93), EASYSIMD_FLOAT32_C(   636.05),
        EASYSIMD_FLOAT32_C(   882.48), EASYSIMD_FLOAT32_C(   247.11), EASYSIMD_FLOAT32_C(  -646.99), EASYSIMD_FLOAT32_C(  -589.92),
        EASYSIMD_FLOAT32_C(  -824.20), EASYSIMD_FLOAT32_C(   398.92), EASYSIMD_FLOAT32_C(  -497.90), EASYSIMD_FLOAT32_C(   860.48),
        EASYSIMD_FLOAT32_C(  -852.81), EASYSIMD_FLOAT32_C(   618.05), EASYSIMD_FLOAT32_C(  -869.02), EASYSIMD_FLOAT32_C(  -156.05) },
      { EASYSIMD_FLOAT32_C(  -620.12), EASYSIMD_FLOAT32_C(    -6.99), EASYSIMD_FLOAT32_C(    80.65), EASYSIMD_FLOAT32_C(  -300.88),
        EASYSIMD_FLOAT32_C(   635.17), EASYSIMD_FLOAT32_C(   765.59), EASYSIMD_FLOAT32_C(   344.16), EASYSIMD_FLOAT32_C(   985.32),
        EASYSIMD_FLOAT32_C(   -70.87), EASYSIMD_FLOAT32_C(  -482.17), EASYSIMD_FLOAT32_C(   829.04), EASYSIMD_FLOAT32_C(   -64.30),
        EASYSIMD_FLOAT32_C(   231.46), EASYSIMD_FLOAT32_C(   384.58), EASYSIMD_FLOAT32_C(   978.69), EASYSIMD_FLOAT32_C(   -52.45) },
      { EASYSIMD_FLOAT32_C(  -729.51), EASYSIMD_FLOAT32_C(  -694.98), EASYSIMD_FLOAT32_C(  -238.16), EASYSIMD_FLOAT32_C(  -300.88),
        EASYSIMD_FLOAT32_C(  -763.30), EASYSIMD_FLOAT32_C(  -680.77), EASYSIMD_FLOAT32_C(  -357.84), EASYSIMD_FLOAT32_C(  -589.92),
        EASYSIMD_FLOAT32_C(  -354.96), EASYSIMD_FLOAT32_C(  -649.85), EASYSIMD_FLOAT32_C(   163.54), EASYSIMD_FLOAT32_C(   173.67),
        EASYSIMD_FLOAT32_C(   843.72), EASYSIMD_FLOAT32_C(  -993.43), EASYSIMD_FLOAT32_C(  -286.37), EASYSIMD_FLOAT32_C(   555.54) } },
    { { EASYSIMD_FLOAT32_C(   689.60), EASYSIMD_FLOAT32_C(   958.62), EASYSIMD_FLOAT32_C(  -416.41), EASYSIMD_FLOAT32_C(   572.08),
        EASYSIMD_FLOAT32_C(   205.73), EASYSIMD_FLOAT32_C(   -63.40), EASYSIMD_FLOAT32_C(   982.16), EASYSIMD_FLOAT32_C(   381.53),
        EASYSIMD_FLOAT32_C(  -664.48), EASYSIMD_FLOAT32_C(  -515.74), EASYSIMD_FLOAT32_C(   242.01), EASYSIMD_FLOAT32_C(  -517.29),
        EASYSIMD_FLOAT32_C(  -897.69), EASYSIMD_FLOAT32_C(   372.99), EASYSIMD_FLOAT32_C(   326.67), EASYSIMD_FLOAT32_C(  -517.81) },
      UINT8_C(170),
      { EASYSIMD_FLOAT32_C(  -592.68), EASYSIMD_FLOAT32_C(   181.31), EASYSIMD_FLOAT32_C(  -998.84), EASYSIMD_FLOAT32_C(  -827.09),
        EASYSIMD_FLOAT32_C(  -474.53), EASYSIMD_FLOAT32_C(   986.49), EASYSIMD_FLOAT32_C(   102.04), EASYSIMD_FLOAT32_C(    43.30),
        EASYSIMD_FLOAT32_C(   815.53), EASYSIMD_FLOAT32_C(  -962.26), EASYSIMD_FLOAT32_C(  -725.24), EASYSIMD_FLOAT32_C(   200.11),
        EASYSIMD_FLOAT32_C(  -983.57), EASYSIMD_FLOAT32_C(   222.30), EASYSIMD_FLOAT32_C(  -110.29), EASYSIMD_FLOAT32_C(   975.06) },
      { EASYSIMD_FLOAT32_C(   805.89), EASYSIMD_FLOAT32_C(  -538.21), EASYSIMD_FLOAT32_C(   180.79), EASYSIMD_FLOAT32_C(  -257.50),
        EASYSIMD_FLOAT32_C(  -556.06), EASYSIMD_FLOAT32_C(  -437.68), EASYSIMD_FLOAT32_C(    78.02), EASYSIMD_FLOAT32_C(   -71.80),
        EASYSIMD_FLOAT32_C(   804.33), EASYSIMD_FLOAT32_C(   560.73), EASYSIMD_FLOAT32_C(    30.51), EASYSIMD_FLOAT32_C(   177.31),
        EASYSIMD_FLOAT32_C(  -112.60), EASYSIMD_FLOAT32_C(   512.71), EASYSIMD_FLOAT32_C(   543.31), EASYSIMD_FLOAT32_C(   294.71) },
      { EASYSIMD_FLOAT32_C(   689.60), EASYSIMD_FLOAT32_C(  -538.21), EASYSIMD_FLOAT32_C(  -416.41), EASYSIMD_FLOAT32_C(  -827.09),
        EASYSIMD_FLOAT32_C(   205.73), EASYSIMD_FLOAT32_C(  -437.68), EASYSIMD_FLOAT32_C(   982.16), EASYSIMD_FLOAT32_C(   -71.80),
        EASYSIMD_FLOAT32_C(  -664.48), EASYSIMD_FLOAT32_C(  -515.74), EASYSIMD_FLOAT32_C(   242.01), EASYSIMD_FLOAT32_C(  -517.29),
        EASYSIMD_FLOAT32_C(  -897.69), EASYSIMD_FLOAT32_C(   372.99), EASYSIMD_FLOAT32_C(   326.67), EASYSIMD_FLOAT32_C(  -517.81) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 src = easysimd_mm512_loadu_ps(test_vec[i].src);
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_ps(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { UINT16_C( 9901),
      { EASYSIMD_FLOAT32_C(  -544.18), EASYSIMD_FLOAT32_C(   601.49), EASYSIMD_FLOAT32_C(  -304.11), EASYSIMD_FLOAT32_C(   528.27),
        EASYSIMD_FLOAT32_C(  -713.39), EASYSIMD_FLOAT32_C(  -743.38), EASYSIMD_FLOAT32_C(   273.42), EASYSIMD_FLOAT32_C(   -14.77),
        EASYSIMD_FLOAT32_C(  -697.44), EASYSIMD_FLOAT32_C(  -402.15), EASYSIMD_FLOAT32_C(   -91.86), EASYSIMD_FLOAT32_C(    77.83),
        EASYSIMD_FLOAT32_C(  -962.39), EASYSIMD_FLOAT32_C(   568.28), EASYSIMD_FLOAT32_C(   531.52), EASYSIMD_FLOAT32_C(    83.25) },
      { EASYSIMD_FLOAT32_C(   426.04), EASYSIMD_FLOAT32_C(  -179.26), EASYSIMD_FLOAT32_C(   988.33), EASYSIMD_FLOAT32_C(   950.60),
        EASYSIMD_FLOAT32_C(   437.57), EASYSIMD_FLOAT32_C(    78.30), EASYSIMD_FLOAT32_C(  -903.72), EASYSIMD_FLOAT32_C(  -100.80),
        EASYSIMD_FLOAT32_C(  -508.84), EASYSIMD_FLOAT32_C(  -791.31), EASYSIMD_FLOAT32_C(   900.01), EASYSIMD_FLOAT32_C(   830.49),
        EASYSIMD_FLOAT32_C(  -949.05), EASYSIMD_FLOAT32_C(  -690.37), EASYSIMD_FLOAT32_C(   246.57), EASYSIMD_FLOAT32_C(  -493.23) },
      { EASYSIMD_FLOAT32_C(  -544.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -304.11), EASYSIMD_FLOAT32_C(   528.27),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -743.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -100.80),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -791.31), EASYSIMD_FLOAT32_C(   -91.86), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -690.37), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C( 3227),
      { EASYSIMD_FLOAT32_C(  -964.97), EASYSIMD_FLOAT32_C(  -802.27), EASYSIMD_FLOAT32_C(  -800.92), EASYSIMD_FLOAT32_C(   308.45),
        EASYSIMD_FLOAT32_C(   182.97), EASYSIMD_FLOAT32_C(  -498.36), EASYSIMD_FLOAT32_C(   906.30), EASYSIMD_FLOAT32_C(  -908.89),
        EASYSIMD_FLOAT32_C(   579.47), EASYSIMD_FLOAT32_C(   943.91), EASYSIMD_FLOAT32_C(   659.39), EASYSIMD_FLOAT32_C(   111.00),
        EASYSIMD_FLOAT32_C(    27.16), EASYSIMD_FLOAT32_C(    85.43), EASYSIMD_FLOAT32_C(   931.73), EASYSIMD_FLOAT32_C(    15.49) },
      { EASYSIMD_FLOAT32_C(    36.03), EASYSIMD_FLOAT32_C(   369.30), EASYSIMD_FLOAT32_C(  -906.21), EASYSIMD_FLOAT32_C(   132.31),
        EASYSIMD_FLOAT32_C(  -731.50), EASYSIMD_FLOAT32_C(  -415.05), EASYSIMD_FLOAT32_C(   341.00), EASYSIMD_FLOAT32_C(  -831.49),
        EASYSIMD_FLOAT32_C(  -584.56), EASYSIMD_FLOAT32_C(   391.95), EASYSIMD_FLOAT32_C(  -521.86), EASYSIMD_FLOAT32_C(   662.01),
        EASYSIMD_FLOAT32_C(   898.72), EASYSIMD_FLOAT32_C(  -610.74), EASYSIMD_FLOAT32_C(   604.47), EASYSIMD_FLOAT32_C(   933.75) },
      { EASYSIMD_FLOAT32_C(  -964.97), EASYSIMD_FLOAT32_C(  -802.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   132.31),
        EASYSIMD_FLOAT32_C(  -731.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -908.89),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -521.86), EASYSIMD_FLOAT32_C(   111.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(42061),
      { EASYSIMD_FLOAT32_C(   242.20), EASYSIMD_FLOAT32_C(   769.96), EASYSIMD_FLOAT32_C(  -694.80), EASYSIMD_FLOAT32_C(   148.51),
        EASYSIMD_FLOAT32_C(   861.07), EASYSIMD_FLOAT32_C(   884.67), EASYSIMD_FLOAT32_C(    92.42), EASYSIMD_FLOAT32_C(   520.46),
        EASYSIMD_FLOAT32_C(    -4.33), EASYSIMD_FLOAT32_C(  -880.42), EASYSIMD_FLOAT32_C(  -394.11), EASYSIMD_FLOAT32_C(   -72.60),
        EASYSIMD_FLOAT32_C(   135.07), EASYSIMD_FLOAT32_C(   641.92), EASYSIMD_FLOAT32_C(  -703.30), EASYSIMD_FLOAT32_C(   228.86) },
      { EASYSIMD_FLOAT32_C(  -225.77), EASYSIMD_FLOAT32_C(  -434.80), EASYSIMD_FLOAT32_C(   813.81), EASYSIMD_FLOAT32_C(  -884.77),
        EASYSIMD_FLOAT32_C(  -266.29), EASYSIMD_FLOAT32_C(  -770.76), EASYSIMD_FLOAT32_C(   507.18), EASYSIMD_FLOAT32_C(   211.84),
        EASYSIMD_FLOAT32_C(   891.25), EASYSIMD_FLOAT32_C(   405.90), EASYSIMD_FLOAT32_C(   601.10), EASYSIMD_FLOAT32_C(   495.72),
        EASYSIMD_FLOAT32_C(   339.65), EASYSIMD_FLOAT32_C(  -811.90), EASYSIMD_FLOAT32_C(   299.27), EASYSIMD_FLOAT32_C(  -418.15) },
      { EASYSIMD_FLOAT32_C(  -225.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -694.80), EASYSIMD_FLOAT32_C(  -884.77),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    92.42), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -394.11), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -811.90), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -418.15) } },
    { UINT16_C(16182),
      { EASYSIMD_FLOAT32_C(   730.36), EASYSIMD_FLOAT32_C(   819.13), EASYSIMD_FLOAT32_C(   489.14), EASYSIMD_FLOAT32_C(  -177.22),
        EASYSIMD_FLOAT32_C(   339.58), EASYSIMD_FLOAT32_C(  -515.19), EASYSIMD_FLOAT32_C(   -57.64), EASYSIMD_FLOAT32_C(   945.47),
        EASYSIMD_FLOAT32_C(   412.21), EASYSIMD_FLOAT32_C(  -922.57), EASYSIMD_FLOAT32_C(   587.39), EASYSIMD_FLOAT32_C(   708.91),
        EASYSIMD_FLOAT32_C(   306.29), EASYSIMD_FLOAT32_C(  -638.38), EASYSIMD_FLOAT32_C(  -725.89), EASYSIMD_FLOAT32_C(   120.10) },
      { EASYSIMD_FLOAT32_C(  -523.15), EASYSIMD_FLOAT32_C(     7.82), EASYSIMD_FLOAT32_C(   349.34), EASYSIMD_FLOAT32_C(   984.04),
        EASYSIMD_FLOAT32_C(  -780.34), EASYSIMD_FLOAT32_C(   240.59), EASYSIMD_FLOAT32_C(   389.94), EASYSIMD_FLOAT32_C(   820.76),
        EASYSIMD_FLOAT32_C(  -263.69), EASYSIMD_FLOAT32_C(  -270.41), EASYSIMD_FLOAT32_C(  -991.14), EASYSIMD_FLOAT32_C(  -964.41),
        EASYSIMD_FLOAT32_C(   311.44), EASYSIMD_FLOAT32_C(   966.92), EASYSIMD_FLOAT32_C(   640.06), EASYSIMD_FLOAT32_C(    41.80) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     7.82), EASYSIMD_FLOAT32_C(   349.34), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -780.34), EASYSIMD_FLOAT32_C(  -515.19), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -263.69), EASYSIMD_FLOAT32_C(  -922.57), EASYSIMD_FLOAT32_C(  -991.14), EASYSIMD_FLOAT32_C(  -964.41),
        EASYSIMD_FLOAT32_C(   306.29), EASYSIMD_FLOAT32_C(  -638.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C( 5758),
      { EASYSIMD_FLOAT32_C(   864.58), EASYSIMD_FLOAT32_C(   125.63), EASYSIMD_FLOAT32_C(   614.01), EASYSIMD_FLOAT32_C(  -193.06),
        EASYSIMD_FLOAT32_C(    71.10), EASYSIMD_FLOAT32_C(    26.23), EASYSIMD_FLOAT32_C(  -115.63), EASYSIMD_FLOAT32_C(  -341.51),
        EASYSIMD_FLOAT32_C(  -264.86), EASYSIMD_FLOAT32_C(  -809.33), EASYSIMD_FLOAT32_C(    20.10), EASYSIMD_FLOAT32_C(     9.25),
        EASYSIMD_FLOAT32_C(   310.77), EASYSIMD_FLOAT32_C(   496.96), EASYSIMD_FLOAT32_C(  -982.93), EASYSIMD_FLOAT32_C(  -339.89) },
      { EASYSIMD_FLOAT32_C(   481.00), EASYSIMD_FLOAT32_C(  -763.28), EASYSIMD_FLOAT32_C(   900.70), EASYSIMD_FLOAT32_C(  -129.07),
        EASYSIMD_FLOAT32_C(  -942.51), EASYSIMD_FLOAT32_C(  -362.99), EASYSIMD_FLOAT32_C(   600.52), EASYSIMD_FLOAT32_C(  -933.65),
        EASYSIMD_FLOAT32_C(  -327.41), EASYSIMD_FLOAT32_C(   -88.04), EASYSIMD_FLOAT32_C(  -966.73), EASYSIMD_FLOAT32_C(  -687.35),
        EASYSIMD_FLOAT32_C(   953.77), EASYSIMD_FLOAT32_C(   819.32), EASYSIMD_FLOAT32_C(   441.85), EASYSIMD_FLOAT32_C(   818.35) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -763.28), EASYSIMD_FLOAT32_C(   614.01), EASYSIMD_FLOAT32_C(  -193.06),
        EASYSIMD_FLOAT32_C(  -942.51), EASYSIMD_FLOAT32_C(  -362.99), EASYSIMD_FLOAT32_C(  -115.63), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -809.33), EASYSIMD_FLOAT32_C(  -966.73), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   310.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C( 3949),
      { EASYSIMD_FLOAT32_C(  -374.71), EASYSIMD_FLOAT32_C(  -983.95), EASYSIMD_FLOAT32_C(  -917.91), EASYSIMD_FLOAT32_C(   509.66),
        EASYSIMD_FLOAT32_C(  -325.47), EASYSIMD_FLOAT32_C(  -182.77), EASYSIMD_FLOAT32_C(   700.33), EASYSIMD_FLOAT32_C(   694.64),
        EASYSIMD_FLOAT32_C(   826.48), EASYSIMD_FLOAT32_C(    11.09), EASYSIMD_FLOAT32_C(   191.60), EASYSIMD_FLOAT32_C(   843.55),
        EASYSIMD_FLOAT32_C(   671.20), EASYSIMD_FLOAT32_C(  -327.41), EASYSIMD_FLOAT32_C(  -919.73), EASYSIMD_FLOAT32_C(   571.90) },
      { EASYSIMD_FLOAT32_C(   543.53), EASYSIMD_FLOAT32_C(  -862.24), EASYSIMD_FLOAT32_C(  -791.09), EASYSIMD_FLOAT32_C(   144.05),
        EASYSIMD_FLOAT32_C(  -795.89), EASYSIMD_FLOAT32_C(  -118.50), EASYSIMD_FLOAT32_C(  -943.99), EASYSIMD_FLOAT32_C(  -762.62),
        EASYSIMD_FLOAT32_C(   194.16), EASYSIMD_FLOAT32_C(  -990.23), EASYSIMD_FLOAT32_C(  -943.30), EASYSIMD_FLOAT32_C(  -363.99),
        EASYSIMD_FLOAT32_C(   828.12), EASYSIMD_FLOAT32_C(     1.65), EASYSIMD_FLOAT32_C(   691.87), EASYSIMD_FLOAT32_C(  -546.59) },
      { EASYSIMD_FLOAT32_C(  -374.71), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -917.91), EASYSIMD_FLOAT32_C(   144.05),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -182.77), EASYSIMD_FLOAT32_C(  -943.99), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   194.16), EASYSIMD_FLOAT32_C(  -990.23), EASYSIMD_FLOAT32_C(  -943.30), EASYSIMD_FLOAT32_C(  -363.99),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C( 9530),
      { EASYSIMD_FLOAT32_C(   963.07), EASYSIMD_FLOAT32_C(   692.24), EASYSIMD_FLOAT32_C(  -408.80), EASYSIMD_FLOAT32_C(   663.40),
        EASYSIMD_FLOAT32_C(   386.88), EASYSIMD_FLOAT32_C(  -582.32), EASYSIMD_FLOAT32_C(  -325.51), EASYSIMD_FLOAT32_C(  -421.52),
        EASYSIMD_FLOAT32_C(  -738.77), EASYSIMD_FLOAT32_C(  -654.30), EASYSIMD_FLOAT32_C(   251.07), EASYSIMD_FLOAT32_C(  -658.50),
        EASYSIMD_FLOAT32_C(   917.60), EASYSIMD_FLOAT32_C(  -205.40), EASYSIMD_FLOAT32_C(  -520.74), EASYSIMD_FLOAT32_C(  -873.49) },
      { EASYSIMD_FLOAT32_C(   938.65), EASYSIMD_FLOAT32_C(  -316.63), EASYSIMD_FLOAT32_C(     8.01), EASYSIMD_FLOAT32_C(   994.66),
        EASYSIMD_FLOAT32_C(   -79.25), EASYSIMD_FLOAT32_C(  -797.83), EASYSIMD_FLOAT32_C(  -995.57), EASYSIMD_FLOAT32_C(   -22.54),
        EASYSIMD_FLOAT32_C(  -161.82), EASYSIMD_FLOAT32_C(   832.55), EASYSIMD_FLOAT32_C(   979.11), EASYSIMD_FLOAT32_C(  -469.95),
        EASYSIMD_FLOAT32_C(  -714.04), EASYSIMD_FLOAT32_C(    -3.19), EASYSIMD_FLOAT32_C(  -695.98), EASYSIMD_FLOAT32_C(  -750.96) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -316.63), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   663.40),
        EASYSIMD_FLOAT32_C(   -79.25), EASYSIMD_FLOAT32_C(  -797.83), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -738.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   251.07), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -205.40), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(20037),
      { EASYSIMD_FLOAT32_C(   912.44), EASYSIMD_FLOAT32_C(  -924.07), EASYSIMD_FLOAT32_C(   312.90), EASYSIMD_FLOAT32_C(  -413.07),
        EASYSIMD_FLOAT32_C(  -345.59), EASYSIMD_FLOAT32_C(   574.13), EASYSIMD_FLOAT32_C(   -67.37), EASYSIMD_FLOAT32_C(   905.48),
        EASYSIMD_FLOAT32_C(   915.63), EASYSIMD_FLOAT32_C(  -149.77), EASYSIMD_FLOAT32_C(  -299.92), EASYSIMD_FLOAT32_C(  -605.11),
        EASYSIMD_FLOAT32_C(   -23.27), EASYSIMD_FLOAT32_C(  -361.27), EASYSIMD_FLOAT32_C(    78.26), EASYSIMD_FLOAT32_C(   984.74) },
      { EASYSIMD_FLOAT32_C(  -366.61), EASYSIMD_FLOAT32_C(   999.01), EASYSIMD_FLOAT32_C(  -813.09), EASYSIMD_FLOAT32_C(  -362.18),
        EASYSIMD_FLOAT32_C(   -23.53), EASYSIMD_FLOAT32_C(    25.08), EASYSIMD_FLOAT32_C(  -529.63), EASYSIMD_FLOAT32_C(   -44.42),
        EASYSIMD_FLOAT32_C(   555.13), EASYSIMD_FLOAT32_C(  -243.67), EASYSIMD_FLOAT32_C(   952.39), EASYSIMD_FLOAT32_C(   859.15),
        EASYSIMD_FLOAT32_C(     5.37), EASYSIMD_FLOAT32_C(  -358.55), EASYSIMD_FLOAT32_C(  -245.64), EASYSIMD_FLOAT32_C(   -82.19) },
      { EASYSIMD_FLOAT32_C(  -366.61), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -813.09), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -529.63), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -243.67), EASYSIMD_FLOAT32_C(  -299.92), EASYSIMD_FLOAT32_C(  -605.11),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -245.64), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_ps(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_min_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   157.06), EASYSIMD_FLOAT64_C(   547.17), EASYSIMD_FLOAT64_C(   473.32), EASYSIMD_FLOAT64_C(  -357.52),
        EASYSIMD_FLOAT64_C(   296.15), EASYSIMD_FLOAT64_C(  -304.34), EASYSIMD_FLOAT64_C(   366.32), EASYSIMD_FLOAT64_C(   139.64) },
      { EASYSIMD_FLOAT64_C(  -267.42), EASYSIMD_FLOAT64_C(  -660.77), EASYSIMD_FLOAT64_C(   238.48), EASYSIMD_FLOAT64_C(  -953.73),
        EASYSIMD_FLOAT64_C(   511.02), EASYSIMD_FLOAT64_C(   236.50), EASYSIMD_FLOAT64_C(  -563.78), EASYSIMD_FLOAT64_C(  -854.07) },
      { EASYSIMD_FLOAT64_C(  -267.42), EASYSIMD_FLOAT64_C(  -660.77), EASYSIMD_FLOAT64_C(   238.48), EASYSIMD_FLOAT64_C(  -953.73),
        EASYSIMD_FLOAT64_C(   296.15), EASYSIMD_FLOAT64_C(  -304.34), EASYSIMD_FLOAT64_C(  -563.78), EASYSIMD_FLOAT64_C(  -854.07) } },
    { { EASYSIMD_FLOAT64_C(    95.48), EASYSIMD_FLOAT64_C(  -524.56), EASYSIMD_FLOAT64_C(  -361.69), EASYSIMD_FLOAT64_C(    32.98),
        EASYSIMD_FLOAT64_C(  -239.25), EASYSIMD_FLOAT64_C(   730.39), EASYSIMD_FLOAT64_C(  -362.23), EASYSIMD_FLOAT64_C(   -99.17) },
      { EASYSIMD_FLOAT64_C(  -102.36), EASYSIMD_FLOAT64_C(   530.27), EASYSIMD_FLOAT64_C(  -659.60), EASYSIMD_FLOAT64_C(  -663.13),
        EASYSIMD_FLOAT64_C(    28.77), EASYSIMD_FLOAT64_C(  -184.41), EASYSIMD_FLOAT64_C(   436.90), EASYSIMD_FLOAT64_C(  -814.16) },
      { EASYSIMD_FLOAT64_C(  -102.36), EASYSIMD_FLOAT64_C(  -524.56), EASYSIMD_FLOAT64_C(  -659.60), EASYSIMD_FLOAT64_C(  -663.13),
        EASYSIMD_FLOAT64_C(  -239.25), EASYSIMD_FLOAT64_C(  -184.41), EASYSIMD_FLOAT64_C(  -362.23), EASYSIMD_FLOAT64_C(  -814.16) } },
    { { EASYSIMD_FLOAT64_C(  -637.24), EASYSIMD_FLOAT64_C(   -89.78), EASYSIMD_FLOAT64_C(  -171.68), EASYSIMD_FLOAT64_C(   658.92),
        EASYSIMD_FLOAT64_C(   605.88), EASYSIMD_FLOAT64_C(  -805.36), EASYSIMD_FLOAT64_C(  -201.45), EASYSIMD_FLOAT64_C(  -661.54) },
      { EASYSIMD_FLOAT64_C(  -466.13), EASYSIMD_FLOAT64_C(  -962.97), EASYSIMD_FLOAT64_C(  -615.27), EASYSIMD_FLOAT64_C(  -955.11),
        EASYSIMD_FLOAT64_C(   273.54), EASYSIMD_FLOAT64_C(  -179.05), EASYSIMD_FLOAT64_C(  -809.18), EASYSIMD_FLOAT64_C(  -630.99) },
      { EASYSIMD_FLOAT64_C(  -637.24), EASYSIMD_FLOAT64_C(  -962.97), EASYSIMD_FLOAT64_C(  -615.27), EASYSIMD_FLOAT64_C(  -955.11),
        EASYSIMD_FLOAT64_C(   273.54), EASYSIMD_FLOAT64_C(  -805.36), EASYSIMD_FLOAT64_C(  -809.18), EASYSIMD_FLOAT64_C(  -661.54) } },
    { { EASYSIMD_FLOAT64_C(   296.39), EASYSIMD_FLOAT64_C(  -170.87), EASYSIMD_FLOAT64_C(   401.99), EASYSIMD_FLOAT64_C(  -942.85),
        EASYSIMD_FLOAT64_C(  -440.48), EASYSIMD_FLOAT64_C(  -960.24), EASYSIMD_FLOAT64_C(   -42.02), EASYSIMD_FLOAT64_C(   457.16) },
      { EASYSIMD_FLOAT64_C(   570.04), EASYSIMD_FLOAT64_C(   298.38), EASYSIMD_FLOAT64_C(   794.03), EASYSIMD_FLOAT64_C(  -401.19),
        EASYSIMD_FLOAT64_C(  -886.02), EASYSIMD_FLOAT64_C(   230.93), EASYSIMD_FLOAT64_C(  -215.35), EASYSIMD_FLOAT64_C(  -523.26) },
      { EASYSIMD_FLOAT64_C(   296.39), EASYSIMD_FLOAT64_C(  -170.87), EASYSIMD_FLOAT64_C(   401.99), EASYSIMD_FLOAT64_C(  -942.85),
        EASYSIMD_FLOAT64_C(  -886.02), EASYSIMD_FLOAT64_C(  -960.24), EASYSIMD_FLOAT64_C(  -215.35), EASYSIMD_FLOAT64_C(  -523.26) } },
    { { EASYSIMD_FLOAT64_C(  -858.85), EASYSIMD_FLOAT64_C(   612.96), EASYSIMD_FLOAT64_C(  -864.34), EASYSIMD_FLOAT64_C(   747.03),
        EASYSIMD_FLOAT64_C(   807.61), EASYSIMD_FLOAT64_C(   -65.79), EASYSIMD_FLOAT64_C(  -914.51), EASYSIMD_FLOAT64_C(  -658.53) },
      { EASYSIMD_FLOAT64_C(   -28.75), EASYSIMD_FLOAT64_C(  -529.78), EASYSIMD_FLOAT64_C(  -613.63), EASYSIMD_FLOAT64_C(  -755.21),
        EASYSIMD_FLOAT64_C(   291.18), EASYSIMD_FLOAT64_C(  -422.81), EASYSIMD_FLOAT64_C(  -386.20), EASYSIMD_FLOAT64_C(  -412.43) },
      { EASYSIMD_FLOAT64_C(  -858.85), EASYSIMD_FLOAT64_C(  -529.78), EASYSIMD_FLOAT64_C(  -864.34), EASYSIMD_FLOAT64_C(  -755.21),
        EASYSIMD_FLOAT64_C(   291.18), EASYSIMD_FLOAT64_C(  -422.81), EASYSIMD_FLOAT64_C(  -914.51), EASYSIMD_FLOAT64_C(  -658.53) } },
    { { EASYSIMD_FLOAT64_C(   406.31), EASYSIMD_FLOAT64_C(  -984.21), EASYSIMD_FLOAT64_C(  -355.28), EASYSIMD_FLOAT64_C(   965.83),
        EASYSIMD_FLOAT64_C(  -944.44), EASYSIMD_FLOAT64_C(   602.70), EASYSIMD_FLOAT64_C(   422.99), EASYSIMD_FLOAT64_C(   625.59) },
      { EASYSIMD_FLOAT64_C(   -98.92), EASYSIMD_FLOAT64_C(   217.03), EASYSIMD_FLOAT64_C(  -775.60), EASYSIMD_FLOAT64_C(    15.06),
        EASYSIMD_FLOAT64_C(  -552.04), EASYSIMD_FLOAT64_C(     9.05), EASYSIMD_FLOAT64_C(   491.80), EASYSIMD_FLOAT64_C(  -410.89) },
      { EASYSIMD_FLOAT64_C(   -98.92), EASYSIMD_FLOAT64_C(  -984.21), EASYSIMD_FLOAT64_C(  -775.60), EASYSIMD_FLOAT64_C(    15.06),
        EASYSIMD_FLOAT64_C(  -944.44), EASYSIMD_FLOAT64_C(     9.05), EASYSIMD_FLOAT64_C(   422.99), EASYSIMD_FLOAT64_C(  -410.89) } },
    { { EASYSIMD_FLOAT64_C(  -377.99), EASYSIMD_FLOAT64_C(   627.46), EASYSIMD_FLOAT64_C(  -663.86), EASYSIMD_FLOAT64_C(  -570.38),
        EASYSIMD_FLOAT64_C(  -438.33), EASYSIMD_FLOAT64_C(  -578.38), EASYSIMD_FLOAT64_C(  -228.91), EASYSIMD_FLOAT64_C(   532.92) },
      { EASYSIMD_FLOAT64_C(  -108.15), EASYSIMD_FLOAT64_C(   157.46), EASYSIMD_FLOAT64_C(   777.71), EASYSIMD_FLOAT64_C(  -816.98),
        EASYSIMD_FLOAT64_C(   734.65), EASYSIMD_FLOAT64_C(  -608.49), EASYSIMD_FLOAT64_C(  -229.41), EASYSIMD_FLOAT64_C(   140.96) },
      { EASYSIMD_FLOAT64_C(  -377.99), EASYSIMD_FLOAT64_C(   157.46), EASYSIMD_FLOAT64_C(  -663.86), EASYSIMD_FLOAT64_C(  -816.98),
        EASYSIMD_FLOAT64_C(  -438.33), EASYSIMD_FLOAT64_C(  -608.49), EASYSIMD_FLOAT64_C(  -229.41), EASYSIMD_FLOAT64_C(   140.96) } },
    { { EASYSIMD_FLOAT64_C(  -592.70), EASYSIMD_FLOAT64_C(   415.31), EASYSIMD_FLOAT64_C(   106.79), EASYSIMD_FLOAT64_C(  -537.14),
        EASYSIMD_FLOAT64_C(    18.00), EASYSIMD_FLOAT64_C(  -470.22), EASYSIMD_FLOAT64_C(  -911.55), EASYSIMD_FLOAT64_C(   919.08) },
      { EASYSIMD_FLOAT64_C(   746.81), EASYSIMD_FLOAT64_C(  -687.15), EASYSIMD_FLOAT64_C(   -65.86), EASYSIMD_FLOAT64_C(  -805.23),
        EASYSIMD_FLOAT64_C(   321.90), EASYSIMD_FLOAT64_C(  -574.06), EASYSIMD_FLOAT64_C(  -216.12), EASYSIMD_FLOAT64_C(   943.91) },
      { EASYSIMD_FLOAT64_C(  -592.70), EASYSIMD_FLOAT64_C(  -687.15), EASYSIMD_FLOAT64_C(   -65.86), EASYSIMD_FLOAT64_C(  -805.23),
        EASYSIMD_FLOAT64_C(    18.00), EASYSIMD_FLOAT64_C(  -574.06), EASYSIMD_FLOAT64_C(  -911.55), EASYSIMD_FLOAT64_C(   919.08) } }
  };

    for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
      easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
      easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
      easysimd__m512d r;
      EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
        r = easysimd_mm512_min_pd(a, b);
      } EASYSIMD_TEST_PERF_END("easysimd_mm512_min_pd");
      easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
    }

  return 0;
}

static int
test_easysimd_mm512_mask_min_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 src[8];
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -740.28), EASYSIMD_FLOAT64_C(   179.12), EASYSIMD_FLOAT64_C(   703.74), EASYSIMD_FLOAT64_C(  -735.57),
        EASYSIMD_FLOAT64_C(   454.00), EASYSIMD_FLOAT64_C(   876.10), EASYSIMD_FLOAT64_C(   -57.17), EASYSIMD_FLOAT64_C(   149.70) },
      UINT8_C(199),
      { EASYSIMD_FLOAT64_C(  -721.50), EASYSIMD_FLOAT64_C(   591.47), EASYSIMD_FLOAT64_C(   852.88), EASYSIMD_FLOAT64_C(  -826.41),
        EASYSIMD_FLOAT64_C(  -586.93), EASYSIMD_FLOAT64_C(   924.98), EASYSIMD_FLOAT64_C(    87.20), EASYSIMD_FLOAT64_C(  -494.10) },
      { EASYSIMD_FLOAT64_C(   281.40), EASYSIMD_FLOAT64_C(   142.23), EASYSIMD_FLOAT64_C(   313.92), EASYSIMD_FLOAT64_C(   960.39),
        EASYSIMD_FLOAT64_C(   379.03), EASYSIMD_FLOAT64_C(  -507.24), EASYSIMD_FLOAT64_C(   349.76), EASYSIMD_FLOAT64_C(  -502.39) },
      { EASYSIMD_FLOAT64_C(  -721.50), EASYSIMD_FLOAT64_C(   142.23), EASYSIMD_FLOAT64_C(   313.92), EASYSIMD_FLOAT64_C(  -735.57),
        EASYSIMD_FLOAT64_C(   454.00), EASYSIMD_FLOAT64_C(   876.10), EASYSIMD_FLOAT64_C(    87.20), EASYSIMD_FLOAT64_C(  -502.39) } },
    { { EASYSIMD_FLOAT64_C(  -361.36), EASYSIMD_FLOAT64_C(   147.20), EASYSIMD_FLOAT64_C(   968.90), EASYSIMD_FLOAT64_C(  -206.13),
        EASYSIMD_FLOAT64_C(  -276.91), EASYSIMD_FLOAT64_C(   774.59), EASYSIMD_FLOAT64_C(    53.59), EASYSIMD_FLOAT64_C(   902.22) },
      UINT8_C(242),
      { EASYSIMD_FLOAT64_C(   318.02), EASYSIMD_FLOAT64_C(   356.22), EASYSIMD_FLOAT64_C(   354.43), EASYSIMD_FLOAT64_C(  -739.16),
        EASYSIMD_FLOAT64_C(  -494.09), EASYSIMD_FLOAT64_C(  -704.46), EASYSIMD_FLOAT64_C(  -460.65), EASYSIMD_FLOAT64_C(  -902.62) },
      { EASYSIMD_FLOAT64_C(  -851.58), EASYSIMD_FLOAT64_C(  -287.06), EASYSIMD_FLOAT64_C(  -489.54), EASYSIMD_FLOAT64_C(  -926.59),
        EASYSIMD_FLOAT64_C(   800.14), EASYSIMD_FLOAT64_C(    16.35), EASYSIMD_FLOAT64_C(   354.81), EASYSIMD_FLOAT64_C(   -57.63) },
      { EASYSIMD_FLOAT64_C(  -361.36), EASYSIMD_FLOAT64_C(  -287.06), EASYSIMD_FLOAT64_C(   968.90), EASYSIMD_FLOAT64_C(  -206.13),
        EASYSIMD_FLOAT64_C(  -494.09), EASYSIMD_FLOAT64_C(  -704.46), EASYSIMD_FLOAT64_C(  -460.65), EASYSIMD_FLOAT64_C(  -902.62) } },
    { { EASYSIMD_FLOAT64_C(  -669.73), EASYSIMD_FLOAT64_C(   315.20), EASYSIMD_FLOAT64_C(  -678.61), EASYSIMD_FLOAT64_C(  -176.97),
        EASYSIMD_FLOAT64_C(  -335.04), EASYSIMD_FLOAT64_C(  -181.00), EASYSIMD_FLOAT64_C(   461.67), EASYSIMD_FLOAT64_C(   812.17) },
      UINT8_C(154),
      { EASYSIMD_FLOAT64_C(  -744.46), EASYSIMD_FLOAT64_C(  -464.74), EASYSIMD_FLOAT64_C(  -437.51), EASYSIMD_FLOAT64_C(   309.13),
        EASYSIMD_FLOAT64_C(  -562.52), EASYSIMD_FLOAT64_C(  -959.18), EASYSIMD_FLOAT64_C(  -372.85), EASYSIMD_FLOAT64_C(   793.70) },
      { EASYSIMD_FLOAT64_C(   395.24), EASYSIMD_FLOAT64_C(  -112.01), EASYSIMD_FLOAT64_C(  -700.39), EASYSIMD_FLOAT64_C(   690.79),
        EASYSIMD_FLOAT64_C(   427.34), EASYSIMD_FLOAT64_C(  -603.01), EASYSIMD_FLOAT64_C(   839.21), EASYSIMD_FLOAT64_C(  -859.72) },
      { EASYSIMD_FLOAT64_C(  -669.73), EASYSIMD_FLOAT64_C(  -464.74), EASYSIMD_FLOAT64_C(  -678.61), EASYSIMD_FLOAT64_C(   309.13),
        EASYSIMD_FLOAT64_C(  -562.52), EASYSIMD_FLOAT64_C(  -181.00), EASYSIMD_FLOAT64_C(   461.67), EASYSIMD_FLOAT64_C(  -859.72) } },
    { { EASYSIMD_FLOAT64_C(   -92.55), EASYSIMD_FLOAT64_C(   912.62), EASYSIMD_FLOAT64_C(   940.42), EASYSIMD_FLOAT64_C(   923.80),
        EASYSIMD_FLOAT64_C(   267.42), EASYSIMD_FLOAT64_C(  -117.22), EASYSIMD_FLOAT64_C(  -745.93), EASYSIMD_FLOAT64_C(  -417.38) },
      UINT8_C(242),
      { EASYSIMD_FLOAT64_C(    77.10), EASYSIMD_FLOAT64_C(   247.59), EASYSIMD_FLOAT64_C(  -976.82), EASYSIMD_FLOAT64_C(  -461.22),
        EASYSIMD_FLOAT64_C(    59.76), EASYSIMD_FLOAT64_C(  -188.92), EASYSIMD_FLOAT64_C(  -205.68), EASYSIMD_FLOAT64_C(   595.02) },
      { EASYSIMD_FLOAT64_C(   373.57), EASYSIMD_FLOAT64_C(  -896.55), EASYSIMD_FLOAT64_C(  -967.50), EASYSIMD_FLOAT64_C(   414.38),
        EASYSIMD_FLOAT64_C(  -269.40), EASYSIMD_FLOAT64_C(   826.19), EASYSIMD_FLOAT64_C(  -190.37), EASYSIMD_FLOAT64_C(   618.59) },
      { EASYSIMD_FLOAT64_C(   -92.55), EASYSIMD_FLOAT64_C(  -896.55), EASYSIMD_FLOAT64_C(   940.42), EASYSIMD_FLOAT64_C(   923.80),
        EASYSIMD_FLOAT64_C(  -269.40), EASYSIMD_FLOAT64_C(  -188.92), EASYSIMD_FLOAT64_C(  -205.68), EASYSIMD_FLOAT64_C(   595.02) } },
    { { EASYSIMD_FLOAT64_C(  -874.20), EASYSIMD_FLOAT64_C(  -499.59), EASYSIMD_FLOAT64_C(    45.93), EASYSIMD_FLOAT64_C(  -477.21),
        EASYSIMD_FLOAT64_C(  -660.38), EASYSIMD_FLOAT64_C(   186.20), EASYSIMD_FLOAT64_C(   430.24), EASYSIMD_FLOAT64_C(  -747.76) },
      UINT8_C(234),
      { EASYSIMD_FLOAT64_C(   354.05), EASYSIMD_FLOAT64_C(   519.67), EASYSIMD_FLOAT64_C(  -990.60), EASYSIMD_FLOAT64_C(   608.12),
        EASYSIMD_FLOAT64_C(  -897.71), EASYSIMD_FLOAT64_C(   213.58), EASYSIMD_FLOAT64_C(  -314.78), EASYSIMD_FLOAT64_C(   349.88) },
      { EASYSIMD_FLOAT64_C(   236.76), EASYSIMD_FLOAT64_C(   223.99), EASYSIMD_FLOAT64_C(  -590.36), EASYSIMD_FLOAT64_C(  -952.16),
        EASYSIMD_FLOAT64_C(  -981.69), EASYSIMD_FLOAT64_C(  -995.35), EASYSIMD_FLOAT64_C(   421.40), EASYSIMD_FLOAT64_C(  -878.24) },
      { EASYSIMD_FLOAT64_C(  -874.20), EASYSIMD_FLOAT64_C(   223.99), EASYSIMD_FLOAT64_C(    45.93), EASYSIMD_FLOAT64_C(  -952.16),
        EASYSIMD_FLOAT64_C(  -660.38), EASYSIMD_FLOAT64_C(  -995.35), EASYSIMD_FLOAT64_C(  -314.78), EASYSIMD_FLOAT64_C(  -878.24) } },
    { { EASYSIMD_FLOAT64_C(  -962.85), EASYSIMD_FLOAT64_C(  -164.21), EASYSIMD_FLOAT64_C(  -147.64), EASYSIMD_FLOAT64_C(   863.35),
        EASYSIMD_FLOAT64_C(   645.41), EASYSIMD_FLOAT64_C(  -529.05), EASYSIMD_FLOAT64_C(   989.15), EASYSIMD_FLOAT64_C(  -854.17) },
      UINT8_C(104),
      { EASYSIMD_FLOAT64_C(  -488.06), EASYSIMD_FLOAT64_C(  -514.55), EASYSIMD_FLOAT64_C(  -296.92), EASYSIMD_FLOAT64_C(   942.19),
        EASYSIMD_FLOAT64_C(  -262.31), EASYSIMD_FLOAT64_C(   829.69), EASYSIMD_FLOAT64_C(   296.23), EASYSIMD_FLOAT64_C(  -742.64) },
      { EASYSIMD_FLOAT64_C(   839.10), EASYSIMD_FLOAT64_C(   -95.65), EASYSIMD_FLOAT64_C(  -640.35), EASYSIMD_FLOAT64_C(    52.68),
        EASYSIMD_FLOAT64_C(   589.57), EASYSIMD_FLOAT64_C(   709.53), EASYSIMD_FLOAT64_C(  -710.56), EASYSIMD_FLOAT64_C(  -186.44) },
      { EASYSIMD_FLOAT64_C(  -962.85), EASYSIMD_FLOAT64_C(  -164.21), EASYSIMD_FLOAT64_C(  -147.64), EASYSIMD_FLOAT64_C(    52.68),
        EASYSIMD_FLOAT64_C(   645.41), EASYSIMD_FLOAT64_C(   709.53), EASYSIMD_FLOAT64_C(  -710.56), EASYSIMD_FLOAT64_C(  -854.17) } },
    { { EASYSIMD_FLOAT64_C(  -880.84), EASYSIMD_FLOAT64_C(  -662.73), EASYSIMD_FLOAT64_C(  -168.13), EASYSIMD_FLOAT64_C(  -876.18),
        EASYSIMD_FLOAT64_C(   758.68), EASYSIMD_FLOAT64_C(   -46.37), EASYSIMD_FLOAT64_C(  -839.03), EASYSIMD_FLOAT64_C(  -405.54) },
      UINT8_C(236),
      { EASYSIMD_FLOAT64_C(  -975.68), EASYSIMD_FLOAT64_C(  -760.13), EASYSIMD_FLOAT64_C(  -723.06), EASYSIMD_FLOAT64_C(  -986.53),
        EASYSIMD_FLOAT64_C(  -614.30), EASYSIMD_FLOAT64_C(   793.81), EASYSIMD_FLOAT64_C(  -474.59), EASYSIMD_FLOAT64_C(  -128.85) },
      { EASYSIMD_FLOAT64_C(  -503.11), EASYSIMD_FLOAT64_C(  -532.40), EASYSIMD_FLOAT64_C(   608.84), EASYSIMD_FLOAT64_C(  -673.42),
        EASYSIMD_FLOAT64_C(   763.83), EASYSIMD_FLOAT64_C(   866.20), EASYSIMD_FLOAT64_C(  -834.32), EASYSIMD_FLOAT64_C(  -331.82) },
      { EASYSIMD_FLOAT64_C(  -880.84), EASYSIMD_FLOAT64_C(  -662.73), EASYSIMD_FLOAT64_C(  -723.06), EASYSIMD_FLOAT64_C(  -986.53),
        EASYSIMD_FLOAT64_C(   758.68), EASYSIMD_FLOAT64_C(   793.81), EASYSIMD_FLOAT64_C(  -834.32), EASYSIMD_FLOAT64_C(  -331.82) } },
    { { EASYSIMD_FLOAT64_C(  -774.15), EASYSIMD_FLOAT64_C(   218.36), EASYSIMD_FLOAT64_C(  -742.25), EASYSIMD_FLOAT64_C(   935.38),
        EASYSIMD_FLOAT64_C(   507.80), EASYSIMD_FLOAT64_C(    71.31), EASYSIMD_FLOAT64_C(  -945.46), EASYSIMD_FLOAT64_C(   845.07) },
      UINT8_C(135),
      { EASYSIMD_FLOAT64_C(  -821.64), EASYSIMD_FLOAT64_C(   603.75), EASYSIMD_FLOAT64_C(  -143.18), EASYSIMD_FLOAT64_C(  -660.67),
        EASYSIMD_FLOAT64_C(  -801.79), EASYSIMD_FLOAT64_C(  -337.19), EASYSIMD_FLOAT64_C(  -636.35), EASYSIMD_FLOAT64_C(  -561.92) },
      { EASYSIMD_FLOAT64_C(   -60.25), EASYSIMD_FLOAT64_C(  -622.88), EASYSIMD_FLOAT64_C(  -176.22), EASYSIMD_FLOAT64_C(  -266.43),
        EASYSIMD_FLOAT64_C(   -97.47), EASYSIMD_FLOAT64_C(   694.93), EASYSIMD_FLOAT64_C(   230.45), EASYSIMD_FLOAT64_C(   370.13) },
      { EASYSIMD_FLOAT64_C(  -821.64), EASYSIMD_FLOAT64_C(  -622.88), EASYSIMD_FLOAT64_C(  -176.22), EASYSIMD_FLOAT64_C(   935.38),
        EASYSIMD_FLOAT64_C(   507.80), EASYSIMD_FLOAT64_C(    71.31), EASYSIMD_FLOAT64_C(  -945.46), EASYSIMD_FLOAT64_C(  -561.92) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d src = easysimd_mm512_loadu_pd(test_vec[i].src);
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_min_pd(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_min_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_min_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { UINT8_C(121),
      { EASYSIMD_FLOAT64_C(   840.17), EASYSIMD_FLOAT64_C(  -139.99), EASYSIMD_FLOAT64_C(  -628.50), EASYSIMD_FLOAT64_C(   530.40),
        EASYSIMD_FLOAT64_C(  -947.05), EASYSIMD_FLOAT64_C(  -129.73), EASYSIMD_FLOAT64_C(  -962.59), EASYSIMD_FLOAT64_C(   370.53) },
      { EASYSIMD_FLOAT64_C(  -874.72), EASYSIMD_FLOAT64_C(    38.87), EASYSIMD_FLOAT64_C(   333.13), EASYSIMD_FLOAT64_C(   818.57),
        EASYSIMD_FLOAT64_C(   354.94), EASYSIMD_FLOAT64_C(  -397.93), EASYSIMD_FLOAT64_C(  -985.56), EASYSIMD_FLOAT64_C(  -200.08) },
      { EASYSIMD_FLOAT64_C(  -874.72), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   530.40),
        EASYSIMD_FLOAT64_C(  -947.05), EASYSIMD_FLOAT64_C(  -397.93), EASYSIMD_FLOAT64_C(  -985.56), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(198),
      { EASYSIMD_FLOAT64_C(   177.52), EASYSIMD_FLOAT64_C(   545.59), EASYSIMD_FLOAT64_C(  -650.55), EASYSIMD_FLOAT64_C(  -557.90),
        EASYSIMD_FLOAT64_C(   528.51), EASYSIMD_FLOAT64_C(   639.67), EASYSIMD_FLOAT64_C(   580.27), EASYSIMD_FLOAT64_C(  -791.42) },
      { EASYSIMD_FLOAT64_C(   803.04), EASYSIMD_FLOAT64_C(  -947.55), EASYSIMD_FLOAT64_C(   983.07), EASYSIMD_FLOAT64_C(  -118.88),
        EASYSIMD_FLOAT64_C(    -1.42), EASYSIMD_FLOAT64_C(   763.30), EASYSIMD_FLOAT64_C(  -278.71), EASYSIMD_FLOAT64_C(   858.59) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -947.55), EASYSIMD_FLOAT64_C(  -650.55), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -278.71), EASYSIMD_FLOAT64_C(  -791.42) } },
    { UINT8_C(198),
      { EASYSIMD_FLOAT64_C(  -748.31), EASYSIMD_FLOAT64_C(   911.54), EASYSIMD_FLOAT64_C(     5.07), EASYSIMD_FLOAT64_C(  -710.90),
        EASYSIMD_FLOAT64_C(   282.08), EASYSIMD_FLOAT64_C(   130.34), EASYSIMD_FLOAT64_C(   327.97), EASYSIMD_FLOAT64_C(  -384.80) },
      { EASYSIMD_FLOAT64_C(   -51.09), EASYSIMD_FLOAT64_C(  -317.09), EASYSIMD_FLOAT64_C(   217.28), EASYSIMD_FLOAT64_C(   -36.65),
        EASYSIMD_FLOAT64_C(   482.83), EASYSIMD_FLOAT64_C(   174.63), EASYSIMD_FLOAT64_C(  -859.13), EASYSIMD_FLOAT64_C(    28.43) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -317.09), EASYSIMD_FLOAT64_C(     5.07), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -859.13), EASYSIMD_FLOAT64_C(  -384.80) } },
    { UINT8_C(251),
      { EASYSIMD_FLOAT64_C(  -417.03), EASYSIMD_FLOAT64_C(  -443.06), EASYSIMD_FLOAT64_C(   163.75), EASYSIMD_FLOAT64_C(  -836.77),
        EASYSIMD_FLOAT64_C(  -234.48), EASYSIMD_FLOAT64_C(   -33.21), EASYSIMD_FLOAT64_C(  -784.32), EASYSIMD_FLOAT64_C(  -251.41) },
      { EASYSIMD_FLOAT64_C(   847.91), EASYSIMD_FLOAT64_C(   214.26), EASYSIMD_FLOAT64_C(  -488.11), EASYSIMD_FLOAT64_C(  -430.80),
        EASYSIMD_FLOAT64_C(    72.85), EASYSIMD_FLOAT64_C(  -353.31), EASYSIMD_FLOAT64_C(  -179.10), EASYSIMD_FLOAT64_C(   -15.60) },
      { EASYSIMD_FLOAT64_C(  -417.03), EASYSIMD_FLOAT64_C(  -443.06), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -836.77),
        EASYSIMD_FLOAT64_C(  -234.48), EASYSIMD_FLOAT64_C(  -353.31), EASYSIMD_FLOAT64_C(  -784.32), EASYSIMD_FLOAT64_C(  -251.41) } },
    { UINT8_C(217),
      { EASYSIMD_FLOAT64_C(   110.00), EASYSIMD_FLOAT64_C(  -733.53), EASYSIMD_FLOAT64_C(  -217.90), EASYSIMD_FLOAT64_C(  -562.03),
        EASYSIMD_FLOAT64_C(  -118.32), EASYSIMD_FLOAT64_C(   731.01), EASYSIMD_FLOAT64_C(   120.88), EASYSIMD_FLOAT64_C(  -901.05) },
      { EASYSIMD_FLOAT64_C(  -305.64), EASYSIMD_FLOAT64_C(  -396.29), EASYSIMD_FLOAT64_C(   273.58), EASYSIMD_FLOAT64_C(  -164.78),
        EASYSIMD_FLOAT64_C(   632.14), EASYSIMD_FLOAT64_C(  -202.34), EASYSIMD_FLOAT64_C(   418.19), EASYSIMD_FLOAT64_C(  -810.93) },
      { EASYSIMD_FLOAT64_C(  -305.64), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -562.03),
        EASYSIMD_FLOAT64_C(  -118.32), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   120.88), EASYSIMD_FLOAT64_C(  -901.05) } },
    { UINT8_C(130),
      { EASYSIMD_FLOAT64_C(   581.42), EASYSIMD_FLOAT64_C(   -45.41), EASYSIMD_FLOAT64_C(   -71.79), EASYSIMD_FLOAT64_C(   797.11),
        EASYSIMD_FLOAT64_C(   703.18), EASYSIMD_FLOAT64_C(  -223.88), EASYSIMD_FLOAT64_C(    11.37), EASYSIMD_FLOAT64_C(  -784.93) },
      { EASYSIMD_FLOAT64_C(   345.32), EASYSIMD_FLOAT64_C(  -915.78), EASYSIMD_FLOAT64_C(  -138.24), EASYSIMD_FLOAT64_C(  -833.78),
        EASYSIMD_FLOAT64_C(    68.62), EASYSIMD_FLOAT64_C(  -486.48), EASYSIMD_FLOAT64_C(   276.21), EASYSIMD_FLOAT64_C(   335.10) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -915.78), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -784.93) } },
    { UINT8_C(163),
      { EASYSIMD_FLOAT64_C(   714.18), EASYSIMD_FLOAT64_C(  -783.23), EASYSIMD_FLOAT64_C(    26.63), EASYSIMD_FLOAT64_C(  -164.94),
        EASYSIMD_FLOAT64_C(  -684.28), EASYSIMD_FLOAT64_C(   720.98), EASYSIMD_FLOAT64_C(   438.77), EASYSIMD_FLOAT64_C(   589.30) },
      { EASYSIMD_FLOAT64_C(  -443.80), EASYSIMD_FLOAT64_C(    70.90), EASYSIMD_FLOAT64_C(  -613.04), EASYSIMD_FLOAT64_C(   974.39),
        EASYSIMD_FLOAT64_C(   259.98), EASYSIMD_FLOAT64_C(  -651.62), EASYSIMD_FLOAT64_C(   555.82), EASYSIMD_FLOAT64_C(  -785.43) },
      { EASYSIMD_FLOAT64_C(  -443.80), EASYSIMD_FLOAT64_C(  -783.23), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -651.62), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -785.43) } },
    { UINT8_C(162),
      { EASYSIMD_FLOAT64_C(   352.93), EASYSIMD_FLOAT64_C(   917.75), EASYSIMD_FLOAT64_C(  -947.30), EASYSIMD_FLOAT64_C(  -635.70),
        EASYSIMD_FLOAT64_C(  -867.18), EASYSIMD_FLOAT64_C(   398.02), EASYSIMD_FLOAT64_C(  -551.47), EASYSIMD_FLOAT64_C(    -5.42) },
      { EASYSIMD_FLOAT64_C(   564.24), EASYSIMD_FLOAT64_C(   517.15), EASYSIMD_FLOAT64_C(   508.10), EASYSIMD_FLOAT64_C(  -159.55),
        EASYSIMD_FLOAT64_C(  -147.76), EASYSIMD_FLOAT64_C(  -196.29), EASYSIMD_FLOAT64_C(  -445.36), EASYSIMD_FLOAT64_C(    69.01) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   517.15), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -196.29), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -5.42) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_min_pd(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_min_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_min_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_min_epu64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_epu64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_min_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_min_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_min_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_min_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_epu64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_min_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_min_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_epu8)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_epu16)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_epu32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_epu64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_min_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_min_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_min_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
