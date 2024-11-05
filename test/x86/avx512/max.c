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

#define EASYSIMD_TEST_X86_AVX512_INSN max

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/max.h>

static int
test_easysimd_mm_max_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 4213791572434006763), -INT64_C( 8748399666946504980) },
      {  INT64_C( 6159742423581674854),  INT64_C( 5570333955189588947) },
      {  INT64_C( 6159742423581674854),  INT64_C( 5570333955189588947) } },
    { {  INT64_C( 6684407759070615066),  INT64_C( 1959222748814722512) },
      {  INT64_C( 5641550714050555355), -INT64_C( 7955394924127081965) },
      {  INT64_C( 6684407759070615066),  INT64_C( 1959222748814722512) } },
    { {  INT64_C( 8531034467208256664), -INT64_C( 4222035683413304425) },
      { -INT64_C( 6861790877078340099), -INT64_C( 7658507518017324113) },
      {  INT64_C( 8531034467208256664), -INT64_C( 4222035683413304425) } },
    { {  INT64_C( 3493670262901917280), -INT64_C( 1653412685200647075) },
      {  INT64_C( 8921957194247882659), -INT64_C( 6847906575342781340) },
      {  INT64_C( 8921957194247882659), -INT64_C( 1653412685200647075) } },
    { { -INT64_C(  835379702286347105),  INT64_C( 7895734606987257278) },
      { -INT64_C( 9054364872487275248), -INT64_C( 1005470044254897877) },
      { -INT64_C(  835379702286347105),  INT64_C( 7895734606987257278) } },
    { { -INT64_C( 7928144233986608785), -INT64_C( 8237137056565086857) },
      { -INT64_C( 1773078236768875068), -INT64_C( 7998289437847762757) },
      { -INT64_C( 1773078236768875068), -INT64_C( 7998289437847762757) } },
    { {  INT64_C( 3708029009156057616),  INT64_C( 4958502219234455749) },
      { -INT64_C( 5182363356007875984), -INT64_C( 3060258868240685101) },
      {  INT64_C( 3708029009156057616),  INT64_C( 4958502219234455749) } },
    { {  INT64_C( 8421501155963117296), -INT64_C( 1755691957961141282) },
      {  INT64_C( 2771446960829548738), -INT64_C(  996805137112744817) },
      {  INT64_C( 8421501155963117296), -INT64_C(  996805137112744817) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_max_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_max_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_max_epi64(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_max_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[2];
    const uint64_t b[2];
    const uint64_t r[2];
  } test_vec[] = {
    { { UINT64_C(17285797308618695617), UINT64_C(16442623695073942225) },
      { UINT64_C(18028500205864895416), UINT64_C(12847214964503432789) },
      { UINT64_C(18028500205864895416), UINT64_C(16442623695073942225) } },
    { { UINT64_C( 3989069307419158876), UINT64_C( 5883301563140647583) },
      { UINT64_C(13139409561211193586), UINT64_C( 8971268455152920541) },
      { UINT64_C(13139409561211193586), UINT64_C( 8971268455152920541) } },
    { { UINT64_C( 5107159659050541378), UINT64_C(  133890891877670896) },
      { UINT64_C(11821025188716165365), UINT64_C(13185680391202282335) },
      { UINT64_C(11821025188716165365), UINT64_C(13185680391202282335) } },
    { { UINT64_C( 1981652191426831763), UINT64_C( 3003197490249172894) },
      { UINT64_C( 2532022191092862136), UINT64_C( 1932500018240066983) },
      { UINT64_C( 2532022191092862136), UINT64_C( 3003197490249172894) } },
    { { UINT64_C(  922064120580419460), UINT64_C(12158143327446265948) },
      { UINT64_C( 7807353639901018482), UINT64_C( 2197854038325198047) },
      { UINT64_C( 7807353639901018482), UINT64_C(12158143327446265948) } },
    { { UINT64_C(  361019875412333960), UINT64_C(10637871897382402569) },
      { UINT64_C(17418554545340532873), UINT64_C( 2583051450126385951) },
      { UINT64_C(17418554545340532873), UINT64_C(10637871897382402569) } },
    { { UINT64_C( 5213790952452892332), UINT64_C(14666502434440999382) },
      { UINT64_C(10785511388560905130), UINT64_C( 9860392634215695384) },
      { UINT64_C(10785511388560905130), UINT64_C(14666502434440999382) } },
    { { UINT64_C(16308441399540249036), UINT64_C( 2321432393704641579) },
      { UINT64_C( 4593822421250494901), UINT64_C(  212174087734366093) },
      { UINT64_C(16308441399540249036), UINT64_C( 2321432393704641579) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_max_epu64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_max_epu64");
    easysimd_test_x86_assert_equal_u64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__m128i r = easysimd_mm_max_epu64(a, b);

    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t src[16];
    const uint16_t k;
    const int8_t a[16];
    const int8_t b[16];
    const int8_t r[16];
  } test_vec[] = {
    { {  INT8_C(  93), -INT8_C(   7), -INT8_C(  57), -INT8_C(  50), -INT8_C(  65),  INT8_C(  28),  INT8_C(  97), -INT8_C( 110),
         INT8_C(  75),  INT8_C(   2), -INT8_C(   4),  INT8_C(  14),  INT8_C( 108), -INT8_C(  82),  INT8_C(  70), -INT8_C(  34) },
      UINT16_C(47027),
      {  INT8_C(  51),  INT8_C(   7), -INT8_C(  58),  INT8_C(  36),  INT8_C(  13), -INT8_C(  67), -INT8_C(  83), -INT8_C(  83),
         INT8_C(   3),  INT8_C(  46),  INT8_C(  10), -INT8_C(  13),  INT8_C(  20),  INT8_C( 103), -INT8_C(  20), -INT8_C(  37) },
      {  INT8_C(  54), -INT8_C(  85), -INT8_C(   9), -INT8_C( 105),  INT8_C(  61),  INT8_C(  66), -INT8_C( 103),  INT8_C(  57),
         INT8_C(  80),  INT8_C(   5), -INT8_C(  24), -INT8_C( 106), -INT8_C(  29), -INT8_C( 101),  INT8_C(  78),  INT8_C(  23) },
      {  INT8_C(  54),  INT8_C(   7), -INT8_C(  57), -INT8_C(  50),  INT8_C(  61),  INT8_C(  66),  INT8_C(  97),  INT8_C(  57),
         INT8_C(  80),  INT8_C(  46),  INT8_C(  10),  INT8_C(  14),  INT8_C(  20),  INT8_C( 103),  INT8_C(  70),  INT8_C(  23) } },
    { { -INT8_C(  94),  INT8_C(  20),  INT8_C(  59), -INT8_C(  80), -INT8_C(  47), -INT8_C(  24),  INT8_C(  93), -INT8_C(  43),
         INT8_C(  22),  INT8_C( 103), -INT8_C(  56),  INT8_C(  42), -INT8_C(  49), -INT8_C(  75),  INT8_C(   6),  INT8_C(   5) },
      UINT16_C(64864),
      { -INT8_C( 100), -INT8_C(  98),  INT8_C(  64),  INT8_C(  54), -INT8_C(  41), -INT8_C( 112),  INT8_C(  59), -INT8_C(  65),
         INT8_C(  39),  INT8_C(  31),  INT8_C(  91),  INT8_C( 117),  INT8_C(  54), -INT8_C(   3), -INT8_C( 119),  INT8_C( 113) },
      { -INT8_C(  83),  INT8_C(  90),  INT8_C(  89),  INT8_C(  11),  INT8_C(  47),  INT8_C( 112),  INT8_C( 114), -INT8_C(   8),
        -INT8_C( 102),  INT8_C(  65), -INT8_C(  83), -INT8_C(  96),  INT8_C(  70),  INT8_C(  13), -INT8_C(  98), -INT8_C(  29) },
      { -INT8_C(  94),  INT8_C(  20),  INT8_C(  59), -INT8_C(  80), -INT8_C(  47),  INT8_C( 112),  INT8_C( 114), -INT8_C(  43),
         INT8_C(  39),  INT8_C( 103),  INT8_C(  91),  INT8_C( 117),  INT8_C(  70),  INT8_C(  13), -INT8_C(  98),  INT8_C( 113) } },
    { { -INT8_C(  85), -INT8_C(  34),  INT8_C(  25), -INT8_C( 125),  INT8_C( 110),  INT8_C(  84),  INT8_C(  66), -INT8_C( 107),
         INT8_C( 115), -INT8_C(  99),  INT8_C(  10), -INT8_C(  87), -INT8_C( 101), -INT8_C( 109),  INT8_C(  26),  INT8_C(  72) },
      UINT16_C(29934),
      {  INT8_C(  83),  INT8_C(  29), -INT8_C(  28), -INT8_C(  58),  INT8_C(  21),  INT8_C( 126),  INT8_C(   7), -INT8_C(  62),
         INT8_C(  31),  INT8_C(  78), -INT8_C(  48), -INT8_C(  67),  INT8_C(  49),  INT8_C( 123), -INT8_C( 101),  INT8_C(  74) },
      { -INT8_C(   2),  INT8_C(   9), -INT8_C(  98),  INT8_C(  65), -INT8_C(  97),  INT8_C(  18), -INT8_C(  34), -INT8_C(  87),
        -INT8_C(  69),  INT8_C( 121),  INT8_C(  61), -INT8_C(  42), -INT8_C(  62),  INT8_C(  43),  INT8_C(  74),  INT8_C(  21) },
      { -INT8_C(  85),  INT8_C(  29), -INT8_C(  28),  INT8_C(  65),  INT8_C( 110),  INT8_C( 126),  INT8_C(   7), -INT8_C(  62),
         INT8_C( 115), -INT8_C(  99),  INT8_C(  61), -INT8_C(  87),  INT8_C(  49),  INT8_C( 123),  INT8_C(  74),  INT8_C(  72) } },
    { {  INT8_C(  72),  INT8_C(  46), -INT8_C(  37),  INT8_C(  94), -INT8_C(  84), -INT8_C(  29),  INT8_C(  32), -INT8_C(  53),
         INT8_C(  49), -INT8_C(  16), -INT8_C( 120),  INT8_C(  98),  INT8_C( 108),  INT8_C(  35), -INT8_C(  84),  INT8_C( 106) },
      UINT16_C(18989),
      { -INT8_C(  85), -INT8_C(  52),  INT8_C(  92), -INT8_C( 118),  INT8_C( 117),  INT8_C(  24),  INT8_C(   3), -INT8_C(  78),
        -INT8_C(  18), -INT8_C(  59), -INT8_C(  35),  INT8_C(  56), -INT8_C(  37),  INT8_C(  38),  INT8_C( 102), -INT8_C(  74) },
      { -INT8_C( 124),  INT8_C(  18), -INT8_C( 103), -INT8_C(  92), -INT8_C(  34), -INT8_C(  54), -INT8_C( 107),  INT8_C( 102),
         INT8_C(  44),  INT8_C(   1), -INT8_C( 118), -INT8_C(  40),  INT8_C( 107), -INT8_C(  73),  INT8_C(  35),  INT8_C(  23) },
      { -INT8_C(  85),  INT8_C(  46),  INT8_C(  92), -INT8_C(  92), -INT8_C(  84),  INT8_C(  24),  INT8_C(  32), -INT8_C(  53),
         INT8_C(  49),  INT8_C(   1), -INT8_C( 120),  INT8_C(  56),  INT8_C( 108),  INT8_C(  35),  INT8_C( 102),  INT8_C( 106) } },
    { { -INT8_C( 125),      INT8_MAX, -INT8_C(  95), -INT8_C(   8), -INT8_C( 105), -INT8_C(  92), -INT8_C(  85), -INT8_C( 123),
         INT8_C( 106), -INT8_C( 120), -INT8_C(  67),  INT8_C(  69), -INT8_C(  82),  INT8_C(  35), -INT8_C(   5),  INT8_C(  50) },
      UINT16_C(38198),
      { -INT8_C(  41),  INT8_C(  20),  INT8_C(  95),  INT8_C( 108),  INT8_C( 122), -INT8_C( 116),  INT8_C( 109),  INT8_C(   4),
         INT8_C( 100), -INT8_C(  40), -INT8_C(  69), -INT8_C( 121), -INT8_C(  17),  INT8_C(  62),  INT8_C(   7), -INT8_C( 112) },
      {  INT8_C(  55), -INT8_C(  98),  INT8_C(  53), -INT8_C(  30),  INT8_C(  36), -INT8_C(  97),  INT8_C( 106), -INT8_C(  31),
        -INT8_C(  28),  INT8_C(  25),  INT8_C(   5), -INT8_C(  33),  INT8_C(  75),  INT8_C(  59),  INT8_C( 116),  INT8_C(  34) },
      { -INT8_C( 125),  INT8_C(  20),  INT8_C(  95), -INT8_C(   8),  INT8_C( 122), -INT8_C(  97), -INT8_C(  85), -INT8_C( 123),
         INT8_C( 100), -INT8_C( 120),  INT8_C(   5),  INT8_C(  69),  INT8_C(  75),  INT8_C(  35), -INT8_C(   5),  INT8_C(  34) } },
    { {  INT8_C(  79), -INT8_C(  44), -INT8_C( 114), -INT8_C(  55),  INT8_C(  96), -INT8_C(   5), -INT8_C(  50), -INT8_C(  60),
        -INT8_C(  44), -INT8_C( 119),  INT8_C(  76), -INT8_C(  61), -INT8_C(  56),  INT8_C(  83),  INT8_C(  84), -INT8_C(   1) },
      UINT16_C(35313),
      { -INT8_C(  31),  INT8_C(  21),  INT8_C(  40),  INT8_C(  75), -INT8_C(   9),  INT8_C(  12),  INT8_C( 100), -INT8_C(   4),
        -INT8_C(  21), -INT8_C(  80),  INT8_C(  55),  INT8_C(  96), -INT8_C(  46), -INT8_C( 122),  INT8_C(  52),  INT8_C(  97) },
      {  INT8_C(  79), -INT8_C( 108),  INT8_C(  92),  INT8_C(  29),  INT8_C(  88),  INT8_C(  48), -INT8_C(  89), -INT8_C(  92),
        -INT8_C(  12),  INT8_C( 111), -INT8_C(   9),  INT8_C(  72),  INT8_C( 110), -INT8_C(  23), -INT8_C(  47),  INT8_C(  79) },
      {  INT8_C(  79), -INT8_C(  44), -INT8_C( 114), -INT8_C(  55),  INT8_C(  88),  INT8_C(  48),  INT8_C( 100), -INT8_C(   4),
        -INT8_C(  12), -INT8_C( 119),  INT8_C(  76),  INT8_C(  96), -INT8_C(  56),  INT8_C(  83),  INT8_C(  84),  INT8_C(  97) } },
    { { -INT8_C(   2), -INT8_C(   7), -INT8_C( 102), -INT8_C(  11),  INT8_C(   5), -INT8_C(   1), -INT8_C(  15), -INT8_C(  16),
        -INT8_C(  81),  INT8_C(  40),  INT8_C(  80), -INT8_C( 127), -INT8_C(  82), -INT8_C( 124), -INT8_C(  30), -INT8_C(   2) },
      UINT16_C(16152),
      {  INT8_C(  27),  INT8_C( 113),  INT8_C( 111), -INT8_C(  62),  INT8_C(  21),  INT8_C(  99),  INT8_C(  49),  INT8_C(  13),
        -INT8_C(  85), -INT8_C(  97), -INT8_C(  10),  INT8_C( 124), -INT8_C(  18), -INT8_C(  12),  INT8_C( 117), -INT8_C( 119) },
      { -INT8_C(  22),  INT8_C( 122), -INT8_C( 120), -INT8_C(  37),  INT8_C( 107),  INT8_C(  55),  INT8_C(   4), -INT8_C(  69),
        -INT8_C(  72), -INT8_C(  78),  INT8_C(  64), -INT8_C( 101), -INT8_C(  80),  INT8_C(  88), -INT8_C(  38), -INT8_C(  52) },
      { -INT8_C(   2), -INT8_C(   7), -INT8_C( 102), -INT8_C(  37),  INT8_C( 107), -INT8_C(   1), -INT8_C(  15), -INT8_C(  16),
        -INT8_C(  72), -INT8_C(  78),  INT8_C(  64),  INT8_C( 124), -INT8_C(  18),  INT8_C(  88), -INT8_C(  30), -INT8_C(   2) } },
    { { -INT8_C(  55),  INT8_C(  73), -INT8_C( 114), -INT8_C(  33), -INT8_C(  83), -INT8_C(  64), -INT8_C(  20),  INT8_C(  88),
         INT8_C(  95), -INT8_C(  30), -INT8_C(  43),  INT8_C(  78), -INT8_C(  42),  INT8_C(  74), -INT8_C(  41), -INT8_C(  64) },
      UINT16_C(24517),
      { -INT8_C( 100),  INT8_C(  48), -INT8_C( 106), -INT8_C(  96), -INT8_C(  21),  INT8_C(  78),  INT8_C(  82),  INT8_C(  43),
        -INT8_C(  23),  INT8_C(   3), -INT8_C( 124), -INT8_C(  61), -INT8_C(  49),  INT8_C(  77),  INT8_C(  13),  INT8_C(  93) },
      {  INT8_C(  44), -INT8_C(  70),  INT8_C(  29),  INT8_C(  24),  INT8_C(  18),  INT8_C( 125), -INT8_C(   6), -INT8_C(  25),
        -INT8_C(  53), -INT8_C(  47),  INT8_C(  50), -INT8_C(  94), -INT8_C( 111), -INT8_C(   9),  INT8_C(   1),  INT8_C(  45) },
      {  INT8_C(  44),  INT8_C(  73),  INT8_C(  29), -INT8_C(  33), -INT8_C(  83), -INT8_C(  64),  INT8_C(  82),  INT8_C(  43),
        -INT8_C(  23),  INT8_C(   3),  INT8_C(  50), -INT8_C(  61), -INT8_C(  49),  INT8_C(  74),  INT8_C(  13), -INT8_C(  64) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi8(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_epi8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_epi8");
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
    easysimd__m128i r = easysimd_mm_mask_max_epi8(src, k, a, b);

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
test_easysimd_mm_maskz_max_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const int8_t a[16];
    const int8_t b[16];
    const int8_t r[16];
  } test_vec[] = {
    { UINT16_C( 4713),
      {  INT8_C(  50),  INT8_C(  48), -INT8_C(  95),  INT8_C(  23),  INT8_C(  98), -INT8_C(  94),  INT8_C(  69), -INT8_C(  28),
         INT8_C( 110), -INT8_C( 104), -INT8_C( 121), -INT8_C(  47), -INT8_C(  23), -INT8_C(  98),  INT8_C(  23), -INT8_C(  33) },
      {  INT8_C( 102), -INT8_C(  30),  INT8_C(  88), -INT8_C(  88),  INT8_C(  11), -INT8_C( 100), -INT8_C(  95), -INT8_C(  79),
         INT8_C(  75), -INT8_C(  35), -INT8_C( 122),  INT8_C(  25),      INT8_MIN, -INT8_C(  17),  INT8_C(  43), -INT8_C(  78) },
      {  INT8_C( 102),  INT8_C(   0),  INT8_C(   0),  INT8_C(  23),  INT8_C(   0), -INT8_C(  94),  INT8_C(  69),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  35),  INT8_C(   0),  INT8_C(   0), -INT8_C(  23),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(52511),
      { -INT8_C(  55), -INT8_C( 127),  INT8_C( 111),  INT8_C(  14),  INT8_C( 101), -INT8_C(  35), -INT8_C(  90), -INT8_C(  20),
        -INT8_C(  81), -INT8_C( 113), -INT8_C( 118), -INT8_C(  58),  INT8_C( 111), -INT8_C(  16), -INT8_C(  88), -INT8_C(  57) },
      { -INT8_C( 104), -INT8_C(  77),  INT8_C(  99),  INT8_C(  58),  INT8_C( 100), -INT8_C(  82),  INT8_C(  23), -INT8_C(  22),
        -INT8_C(  56), -INT8_C( 105), -INT8_C(  38), -INT8_C(  13),  INT8_C(  73), -INT8_C(   7), -INT8_C(  64),  INT8_C(  18) },
      { -INT8_C(  55), -INT8_C(  77),  INT8_C( 111),  INT8_C(  58),  INT8_C( 101),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  56),  INT8_C(   0), -INT8_C(  38), -INT8_C(  13),  INT8_C(   0),  INT8_C(   0), -INT8_C(  64),  INT8_C(  18) } },
    { UINT16_C(12411),
      {  INT8_C(  32), -INT8_C(  32),  INT8_C(  13), -INT8_C(  58), -INT8_C(  51), -INT8_C(  68),  INT8_C(  86),  INT8_C(  87),
        -INT8_C( 126), -INT8_C(  59),  INT8_C(  72),  INT8_C(  43), -INT8_C( 116), -INT8_C(  32), -INT8_C(  34), -INT8_C(  17) },
      {  INT8_C(  26),  INT8_C(  67), -INT8_C(  98),  INT8_C(  49),  INT8_C(  45),  INT8_C( 102), -INT8_C(  56),  INT8_C(   7),
         INT8_C(  89),  INT8_C(  17),  INT8_C(   1),  INT8_C(  26),  INT8_C(  35),  INT8_C( 124),  INT8_C(  74),  INT8_C(  67) },
      {  INT8_C(  32),  INT8_C(  67),  INT8_C(   0),  INT8_C(  49),  INT8_C(  45),  INT8_C( 102),  INT8_C(  86),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  35),  INT8_C( 124),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(22364),
      {  INT8_C(  10),  INT8_C(  41),  INT8_C(  20),  INT8_C(  96), -INT8_C( 127), -INT8_C( 106),  INT8_C(  37), -INT8_C(  55),
        -INT8_C(  63), -INT8_C(  79), -INT8_C(  87), -INT8_C(  96), -INT8_C(  96), -INT8_C(  60), -INT8_C(  29),  INT8_C(  62) },
      { -INT8_C(  11),  INT8_C(  16), -INT8_C(  92), -INT8_C(  66),  INT8_C(  24), -INT8_C(   2), -INT8_C(  49),  INT8_C(  25),
         INT8_C(  24), -INT8_C(  13), -INT8_C( 107),  INT8_C(  98),  INT8_C(  54), -INT8_C(  15), -INT8_C(  71),  INT8_C(  64) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(  20),  INT8_C(  96),  INT8_C(  24),  INT8_C(   0),  INT8_C(  37),  INT8_C(   0),
         INT8_C(  24), -INT8_C(  13), -INT8_C(  87),  INT8_C(   0),  INT8_C(  54),  INT8_C(   0), -INT8_C(  29),  INT8_C(   0) } },
    { UINT16_C(52507),
      { -INT8_C(  96), -INT8_C( 100),  INT8_C( 100), -INT8_C(  59),  INT8_C( 101),  INT8_C(  37),  INT8_C( 118),  INT8_C(  14),
        -INT8_C(  59),  INT8_C(  23), -INT8_C(  46), -INT8_C(  88),  INT8_C(  85), -INT8_C(  56), -INT8_C(  71), -INT8_C(   6) },
      { -INT8_C( 122), -INT8_C(  47), -INT8_C(   8),  INT8_C(  85), -INT8_C(  22),  INT8_C(  16),  INT8_C(  72),      INT8_MAX,
         INT8_C( 114),      INT8_MAX,  INT8_C( 112),  INT8_C(  43), -INT8_C(  65), -INT8_C( 117), -INT8_C(   7),  INT8_C(  96) },
      { -INT8_C(  96), -INT8_C(  47),  INT8_C(   0),  INT8_C(  85),  INT8_C( 101),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 114),  INT8_C(   0),  INT8_C( 112),  INT8_C(  43),  INT8_C(   0),  INT8_C(   0), -INT8_C(   7),  INT8_C(  96) } },
    { UINT16_C(23847),
      {  INT8_C(  37), -INT8_C( 116), -INT8_C( 126), -INT8_C( 100), -INT8_C( 101),  INT8_C(  72), -INT8_C(  77),  INT8_C( 109),
        -INT8_C(  16),  INT8_C(   8),  INT8_C(  53), -INT8_C(  87),  INT8_C(   2), -INT8_C(  69),  INT8_C( 122), -INT8_C(   6) },
      {  INT8_C(  17),  INT8_C( 100),  INT8_C(  10),  INT8_C(  89), -INT8_C(  29),  INT8_C( 124), -INT8_C(  40),  INT8_C(  84),
        -INT8_C(  88), -INT8_C( 104), -INT8_C(  33), -INT8_C(  95), -INT8_C(   8),  INT8_C(   7), -INT8_C(   2),  INT8_C(  29) },
      {  INT8_C(  37),  INT8_C( 100),  INT8_C(  10),  INT8_C(   0),  INT8_C(   0),  INT8_C( 124),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  16),  INT8_C(   0),  INT8_C(  53), -INT8_C(  87),  INT8_C(   2),  INT8_C(   0),  INT8_C( 122),  INT8_C(   0) } },
    { UINT16_C(32915),
      { -INT8_C(  71),  INT8_C(  46), -INT8_C(  56),  INT8_C( 108), -INT8_C( 100), -INT8_C(  71),  INT8_C( 117), -INT8_C(  47),
         INT8_C(  98),  INT8_C( 119), -INT8_C( 115), -INT8_C(  35),  INT8_C( 114), -INT8_C(  98),  INT8_C(  65),  INT8_C( 124) },
      { -INT8_C(   9),  INT8_C(  37), -INT8_C(   7), -INT8_C(  48),  INT8_C( 121), -INT8_C(  95),  INT8_C( 104),  INT8_C(  88),
         INT8_C(  66),  INT8_C(  96),  INT8_C(  95),  INT8_C(  64),  INT8_C( 125), -INT8_C(  13), -INT8_C(  64),  INT8_C(  55) },
      { -INT8_C(   9),  INT8_C(  46),  INT8_C(   0),  INT8_C(   0),  INT8_C( 121),  INT8_C(   0),  INT8_C(   0),  INT8_C(  88),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 124) } },
    { UINT16_C(35105),
      { -INT8_C(  93), -INT8_C(  67),  INT8_C(  66),  INT8_C(  24), -INT8_C( 113), -INT8_C(  92), -INT8_C( 112),  INT8_C(  28),
        -INT8_C( 127),  INT8_C(   2), -INT8_C(  70), -INT8_C(  61),  INT8_C( 126), -INT8_C(  79), -INT8_C(  24),  INT8_C( 119) },
      { -INT8_C( 127),  INT8_C(  97),  INT8_C(  24), -INT8_C(  23), -INT8_C(  71),  INT8_C(  90),  INT8_C(  73),  INT8_C(  25),
        -INT8_C( 102), -INT8_C(  57),  INT8_C(  12),  INT8_C(  91), -INT8_C(   2),  INT8_C(  45), -INT8_C(  28), -INT8_C(  95) },
      { -INT8_C(  93),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  90),  INT8_C(   0),  INT8_C(   0),
        -INT8_C( 102),  INT8_C(   0),  INT8_C(   0),  INT8_C(  91),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 119) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_epi8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i b = easysimd_test_x86_random_i8x16();
    easysimd__m128i r = easysimd_mm_maskz_max_epi8(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { { -INT16_C( 26841),  INT16_C(  4813),  INT16_C(  8421), -INT16_C( 12482), -INT16_C( 15837), -INT16_C(  3438), -INT16_C( 24817),  INT16_C( 15439) },
      UINT8_C( 89),
      {  INT16_C( 21613), -INT16_C(  5524),  INT16_C( 21327),  INT16_C(  8373),  INT16_C( 22405),  INT16_C( 31921), -INT16_C(  8360), -INT16_C(  4189) },
      { -INT16_C( 18772), -INT16_C( 13100), -INT16_C( 23564), -INT16_C( 18705), -INT16_C(  7882), -INT16_C( 10811),  INT16_C(   305), -INT16_C( 25041) },
      {  INT16_C( 21613),  INT16_C(  4813),  INT16_C(  8421),  INT16_C(  8373),  INT16_C( 22405), -INT16_C(  3438),  INT16_C(   305),  INT16_C( 15439) } },
    { { -INT16_C( 25770), -INT16_C( 23160),  INT16_C( 15854),  INT16_C( 29893),  INT16_C( 30356), -INT16_C(  4880), -INT16_C( 27563),  INT16_C(   731) },
      UINT8_C( 74),
      { -INT16_C( 12625),  INT16_C( 21310), -INT16_C(  2882), -INT16_C( 24695),  INT16_C( 24249), -INT16_C( 17456),  INT16_C( 28301),  INT16_C( 10257) },
      { -INT16_C( 18698),  INT16_C( 13079), -INT16_C( 29829), -INT16_C(  3641), -INT16_C( 19589),  INT16_C(  3911),  INT16_C( 18830),  INT16_C( 15961) },
      { -INT16_C( 25770),  INT16_C( 21310),  INT16_C( 15854), -INT16_C(  3641),  INT16_C( 30356), -INT16_C(  4880),  INT16_C( 28301),  INT16_C(   731) } },
    { { -INT16_C( 26857), -INT16_C( 10863),  INT16_C(  6795),  INT16_C( 17781),  INT16_C( 17784),  INT16_C(  1536),  INT16_C(  4532), -INT16_C( 21970) },
      UINT8_C(199),
      { -INT16_C(  8635), -INT16_C( 12222),  INT16_C( 13221),  INT16_C( 22860),  INT16_C( 23418), -INT16_C( 15385),  INT16_C(  9653),  INT16_C( 19675) },
      { -INT16_C( 20298), -INT16_C( 12072),  INT16_C(  7461),  INT16_C( 27465),  INT16_C( 20253),  INT16_C( 11807), -INT16_C( 13955), -INT16_C( 15371) },
      { -INT16_C(  8635), -INT16_C( 12072),  INT16_C( 13221),  INT16_C( 17781),  INT16_C( 17784),  INT16_C(  1536),  INT16_C(  9653),  INT16_C( 19675) } },
    { {  INT16_C( 14247),  INT16_C( 19859), -INT16_C(  8342), -INT16_C(  6746), -INT16_C( 29381), -INT16_C(  3928), -INT16_C( 31821),  INT16_C( 26940) },
      UINT8_C( 52),
      {  INT16_C( 14868),  INT16_C( 12633), -INT16_C( 15229), -INT16_C( 11698),  INT16_C( 31971), -INT16_C( 21169),  INT16_C(  4721), -INT16_C( 22444) },
      { -INT16_C( 24154), -INT16_C( 31469), -INT16_C(  1977), -INT16_C( 10816), -INT16_C( 20320),  INT16_C(  9352), -INT16_C(  3603),  INT16_C(   344) },
      {  INT16_C( 14247),  INT16_C( 19859), -INT16_C(  1977), -INT16_C(  6746),  INT16_C( 31971),  INT16_C(  9352), -INT16_C( 31821),  INT16_C( 26940) } },
    { { -INT16_C( 20181), -INT16_C( 20941), -INT16_C( 32394),  INT16_C( 22912), -INT16_C( 12034),  INT16_C( 28422),  INT16_C( 23522), -INT16_C( 30696) },
      UINT8_C(252),
      {  INT16_C(  3627),  INT16_C(  9028),  INT16_C(  6606),  INT16_C( 32707), -INT16_C(  6239), -INT16_C( 28052),  INT16_C( 27967), -INT16_C(  3650) },
      {  INT16_C( 27808),  INT16_C(  8807), -INT16_C( 16147), -INT16_C( 17120), -INT16_C( 28729),  INT16_C(  8863),  INT16_C( 10407), -INT16_C( 11746) },
      { -INT16_C( 20181), -INT16_C( 20941),  INT16_C(  6606),  INT16_C( 32707), -INT16_C(  6239),  INT16_C(  8863),  INT16_C( 27967), -INT16_C(  3650) } },
    { {  INT16_C( 25142),  INT16_C(  1269), -INT16_C( 18053),  INT16_C(  7299), -INT16_C(  4192), -INT16_C(  8017),  INT16_C( 27997), -INT16_C(   559) },
      UINT8_C(217),
      {  INT16_C(  7992), -INT16_C(  1850), -INT16_C( 31937), -INT16_C( 12353), -INT16_C(  7901),  INT16_C( 19318),  INT16_C( 18688),  INT16_C( 25217) },
      { -INT16_C( 31426), -INT16_C(  2082), -INT16_C(  1527), -INT16_C(  1896),  INT16_C( 30889),  INT16_C(  5717),  INT16_C( 21321), -INT16_C( 32272) },
      {  INT16_C(  7992),  INT16_C(  1269), -INT16_C( 18053), -INT16_C(  1896),  INT16_C( 30889), -INT16_C(  8017),  INT16_C( 21321),  INT16_C( 25217) } },
    { { -INT16_C( 18830), -INT16_C( 19847),  INT16_C( 14650),  INT16_C( 23937), -INT16_C(  2278),  INT16_C(  6824),  INT16_C( 10560),  INT16_C( 32637) },
      UINT8_C(174),
      {  INT16_C( 30299),  INT16_C( 21943), -INT16_C( 20466), -INT16_C( 30977),  INT16_C(  5381),  INT16_C( 22735),  INT16_C( 20485), -INT16_C( 17205) },
      {  INT16_C( 32202),  INT16_C(  1014),  INT16_C( 21502), -INT16_C(  2787),  INT16_C( 14587),  INT16_C(  9270), -INT16_C( 19019),  INT16_C(  4306) },
      { -INT16_C( 18830),  INT16_C( 21943),  INT16_C( 21502), -INT16_C(  2787), -INT16_C(  2278),  INT16_C( 22735),  INT16_C( 10560),  INT16_C(  4306) } },
    { { -INT16_C( 30165),  INT16_C( 14949),  INT16_C( 25658),  INT16_C( 16320), -INT16_C( 28550),  INT16_C( 32664),  INT16_C( 25568), -INT16_C( 21957) },
      UINT8_C(224),
      { -INT16_C( 21199), -INT16_C( 31522), -INT16_C( 11317),  INT16_C(   895), -INT16_C( 23799), -INT16_C( 16712), -INT16_C( 14218),  INT16_C(   234) },
      {  INT16_C(  9261), -INT16_C( 28102),  INT16_C( 31204),  INT16_C( 29708), -INT16_C( 29935),  INT16_C( 29781), -INT16_C(    57), -INT16_C(  1964) },
      { -INT16_C( 30165),  INT16_C( 14949),  INT16_C( 25658),  INT16_C( 16320), -INT16_C( 28550),  INT16_C( 29781), -INT16_C(    57),  INT16_C(   234) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_epi16");
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
    easysimd__m128i r = easysimd_mm_mask_max_epi16(src, k, a, b);

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
test_easysimd_mm_maskz_max_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C(235),
      { -INT16_C( 17882), -INT16_C( 13702), -INT16_C( 27062),  INT16_C( 19532),  INT16_C(  3920),  INT16_C(   458),  INT16_C( 17143),  INT16_C( 22659) },
      {  INT16_C( 27738), -INT16_C( 19183),  INT16_C( 10934),  INT16_C( 32079), -INT16_C( 21962),  INT16_C( 25723),  INT16_C(  7310), -INT16_C( 19377) },
      {  INT16_C( 27738), -INT16_C( 13702),  INT16_C(     0),  INT16_C( 32079),  INT16_C(     0),  INT16_C( 25723),  INT16_C( 17143),  INT16_C( 22659) } },
    { UINT8_C(214),
      {  INT16_C( 32713),  INT16_C( 24352),  INT16_C( 27851), -INT16_C(  9553), -INT16_C( 20425),  INT16_C( 31185),  INT16_C( 10547), -INT16_C( 24365) },
      { -INT16_C( 30662),  INT16_C( 25942), -INT16_C( 11304), -INT16_C( 32101), -INT16_C(   178),  INT16_C( 27153), -INT16_C( 15026),  INT16_C(  5953) },
      {  INT16_C(     0),  INT16_C( 25942),  INT16_C( 27851),  INT16_C(     0), -INT16_C(   178),  INT16_C(     0),  INT16_C( 10547),  INT16_C(  5953) } },
    { UINT8_C( 68),
      {  INT16_C( 30305), -INT16_C( 12785), -INT16_C(  5851), -INT16_C( 10747),  INT16_C( 32442), -INT16_C(  7415), -INT16_C( 22191), -INT16_C(  9698) },
      { -INT16_C( 31745), -INT16_C( 11598),  INT16_C( 13342),  INT16_C(  7712), -INT16_C( 29883),  INT16_C(  2924), -INT16_C( 31540),  INT16_C( 11599) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 13342),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 22191),  INT16_C(     0) } },
    { UINT8_C(250),
      { -INT16_C(  1185),  INT16_C( 18464), -INT16_C(  2560),  INT16_C( 32259), -INT16_C(  6401), -INT16_C( 22064), -INT16_C( 22012), -INT16_C( 30808) },
      {  INT16_C( 31580), -INT16_C( 28506), -INT16_C( 15205),  INT16_C(  9942), -INT16_C(  7888), -INT16_C( 19214),  INT16_C(  8240), -INT16_C( 28753) },
      {  INT16_C(     0),  INT16_C( 18464),  INT16_C(     0),  INT16_C( 32259), -INT16_C(  6401), -INT16_C( 19214),  INT16_C(  8240), -INT16_C( 28753) } },
    { UINT8_C( 27),
      { -INT16_C( 10033), -INT16_C( 15076), -INT16_C( 25893), -INT16_C( 15932),  INT16_C( 28010),  INT16_C(  5318),  INT16_C( 19734), -INT16_C( 28304) },
      {  INT16_C(   499), -INT16_C( 18644),  INT16_C( 21463), -INT16_C( 18200), -INT16_C( 25531),  INT16_C( 26088),  INT16_C( 30795),  INT16_C(  6785) },
      {  INT16_C(   499), -INT16_C( 15076),  INT16_C(     0), -INT16_C( 15932),  INT16_C( 28010),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 80),
      { -INT16_C(  8291),  INT16_C( 14123), -INT16_C(  4956),  INT16_C(  4514), -INT16_C( 18766),  INT16_C(    39), -INT16_C( 18393),  INT16_C( 10483) },
      { -INT16_C( 21531),  INT16_C( 14591), -INT16_C( 18541),  INT16_C( 12157), -INT16_C(  7265),  INT16_C(  6011), -INT16_C( 27292),  INT16_C(   359) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  7265),  INT16_C(     0), -INT16_C( 18393),  INT16_C(     0) } },
    { UINT8_C(117),
      {  INT16_C( 14482),  INT16_C( 32537),  INT16_C( 10970), -INT16_C( 28367),  INT16_C( 12626),  INT16_C(  2744), -INT16_C(  8155), -INT16_C( 12049) },
      {  INT16_C( 10207), -INT16_C( 27037), -INT16_C( 27995), -INT16_C( 30667),  INT16_C( 19725), -INT16_C( 23572), -INT16_C(  4684),  INT16_C( 18200) },
      {  INT16_C( 14482),  INT16_C(     0),  INT16_C( 10970),  INT16_C(     0),  INT16_C( 19725),  INT16_C(  2744), -INT16_C(  4684),  INT16_C(     0) } },
    { UINT8_C( 37),
      { -INT16_C( 14799),  INT16_C( 23296), -INT16_C( 28169),  INT16_C( 10669), -INT16_C( 18359),  INT16_C( 10574),  INT16_C(  7847), -INT16_C( 12536) },
      { -INT16_C( 24959),  INT16_C(  4980), -INT16_C(   813),  INT16_C(  8225), -INT16_C( 15128), -INT16_C( 10795),  INT16_C(  7388),  INT16_C(  3578) },
      { -INT16_C( 14799),  INT16_C(     0), -INT16_C(   813),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10574),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_maskz_max_epi16(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(  2021470893),  INT32_C(   259783686),  INT32_C(   382612384),  INT32_C(   672577787) },
      UINT8_C(220),
      { -INT32_C(   910116272),  INT32_C(  1390097862), -INT32_C(  1994829942), -INT32_C(   701099612) },
      { -INT32_C(  1948471666),  INT32_C(  1546382377),  INT32_C(  1918321082), -INT32_C(  1488027561) },
      {  INT32_C(  2021470893),  INT32_C(   259783686),  INT32_C(  1918321082), -INT32_C(   701099612) } },
    { {  INT32_C(    24120890), -INT32_C(   850179261),  INT32_C(  1062694043),  INT32_C(   202739069) },
      UINT8_C( 60),
      { -INT32_C(   563767310),  INT32_C(    60342978), -INT32_C(  1722152423),  INT32_C(  1725170008) },
      { -INT32_C(  1112877711), -INT32_C(  1806141656),  INT32_C(  1544656846), -INT32_C(  1634198100) },
      {  INT32_C(    24120890), -INT32_C(   850179261),  INT32_C(  1544656846),  INT32_C(  1725170008) } },
    { {  INT32_C(  1987902900), -INT32_C(   646376257),  INT32_C(  1987236638),  INT32_C(  1188906708) },
      UINT8_C( 27),
      { -INT32_C(    29162617), -INT32_C(   221391013),  INT32_C(   111028713), -INT32_C(  1095025215) },
      { -INT32_C(   830590535),  INT32_C(  2129418155),  INT32_C(   273900489), -INT32_C(   953444032) },
      { -INT32_C(    29162617),  INT32_C(  2129418155),  INT32_C(  1987236638), -INT32_C(   953444032) } },
    { { -INT32_C(   138055780),  INT32_C(   803836486),  INT32_C(  2083948475),  INT32_C(  2117857732) },
      UINT8_C( 34),
      {  INT32_C(   281889977), -INT32_C(  1680257992),  INT32_C(   953936287), -INT32_C(  2066439659) },
      { -INT32_C(    87372952),  INT32_C(  1001847476),  INT32_C(   553660976),  INT32_C(   641957485) },
      { -INT32_C(   138055780),  INT32_C(  1001847476),  INT32_C(  2083948475),  INT32_C(  2117857732) } },
    { {  INT32_C(    37097930), -INT32_C(    56749987),  INT32_C(   238320121), -INT32_C(  2070804452) },
      UINT8_C(211),
      {  INT32_C(  1468497501),  INT32_C(  1736950324),  INT32_C(  1087678658), -INT32_C(    66389013) },
      {  INT32_C(  1096355121),  INT32_C(   607868331), -INT32_C(  1858057847),  INT32_C(   962905308) },
      {  INT32_C(  1468497501),  INT32_C(  1736950324),  INT32_C(   238320121), -INT32_C(  2070804452) } },
    { {  INT32_C(  2005986115),  INT32_C(  1893603246), -INT32_C(  1431194689), -INT32_C(   542655570) },
      UINT8_C(200),
      {  INT32_C(  1450385664), -INT32_C(  1512073124), -INT32_C(  1652461096), -INT32_C(  1042236715) },
      {  INT32_C(  1634686794), -INT32_C(   383721674),  INT32_C(  1285016464),  INT32_C(  1913943666) },
      {  INT32_C(  2005986115),  INT32_C(  1893603246), -INT32_C(  1431194689),  INT32_C(  1913943666) } },
    { { -INT32_C(   204961641), -INT32_C(   124147680), -INT32_C(   292218343),  INT32_C(   531592661) },
      UINT8_C(205),
      { -INT32_C(    33324770),  INT32_C(  1821306017), -INT32_C(   102835581),  INT32_C(  2006012399) },
      { -INT32_C(  1047034855),  INT32_C(   953847581), -INT32_C(  1710372571), -INT32_C(  1754846088) },
      { -INT32_C(    33324770), -INT32_C(   124147680), -INT32_C(   102835581),  INT32_C(  2006012399) } },
    { {  INT32_C(  1335192237), -INT32_C(   642047146), -INT32_C(   304899330),  INT32_C(    73688299) },
      UINT8_C(232),
      { -INT32_C(  1979267333),  INT32_C(  1773092512),  INT32_C(  2011318859),  INT32_C(   472218033) },
      {  INT32_C(   829584398), -INT32_C(   919647185), -INT32_C(  2085348321),  INT32_C(  2070657408) },
      {  INT32_C(  1335192237), -INT32_C(   642047146), -INT32_C(   304899330),  INT32_C(  2070657408) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_epi32");
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
    easysimd__m128i r = easysimd_mm_mask_max_epi32(src, k, a, b);

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
test_easysimd_mm_maskz_max_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(226),
      { -INT32_C(  1948686086), -INT32_C(   824966634),  INT32_C(  1853226320), -INT32_C(  1544600571) },
      { -INT32_C(  1267268680), -INT32_C(   392390876), -INT32_C(  2000391828),  INT32_C(  1718276460) },
      {  INT32_C(           0), -INT32_C(   392390876),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 57),
      {  INT32_C(  1179644484), -INT32_C(   996729402), -INT32_C(   674691693), -INT32_C(   124752395) },
      {  INT32_C(  2065450212),  INT32_C(  1390937313), -INT32_C(  1715572536),  INT32_C(   533865947) },
      {  INT32_C(  2065450212),  INT32_C(           0),  INT32_C(           0),  INT32_C(   533865947) } },
    { UINT8_C( 23),
      {  INT32_C(  1071473954),  INT32_C(    47358460), -INT32_C(   654857621),  INT32_C(  2126311226) },
      {  INT32_C(   257898251), -INT32_C(  1864912353), -INT32_C(  1788120976),  INT32_C(  1689029186) },
      {  INT32_C(  1071473954),  INT32_C(    47358460), -INT32_C(   654857621),  INT32_C(           0) } },
    { UINT8_C(239),
      {  INT32_C(   736928906),  INT32_C(   546762358), -INT32_C(   732270875),  INT32_C(  1658837290) },
      { -INT32_C(   360583624),  INT32_C(   895160773),  INT32_C(   108523644),  INT32_C(   670489757) },
      {  INT32_C(   736928906),  INT32_C(   895160773),  INT32_C(   108523644),  INT32_C(  1658837290) } },
    { UINT8_C(128),
      { -INT32_C(   789163294),  INT32_C(  1471485929),  INT32_C(  1250068849),  INT32_C(  1451484264) },
      {  INT32_C(  1998286181), -INT32_C(  1175236408), -INT32_C(  1554580793), -INT32_C(   769425936) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(207),
      {  INT32_C(   834183706), -INT32_C(   526249897),  INT32_C(  1984490898), -INT32_C(   572809616) },
      {  INT32_C(   161895097),  INT32_C(  1087397702),  INT32_C(   842036405), -INT32_C(  1325333865) },
      {  INT32_C(   834183706),  INT32_C(  1087397702),  INT32_C(  1984490898), -INT32_C(   572809616) } },
    { UINT8_C(164),
      { -INT32_C(   889462086),  INT32_C(  1918688133), -INT32_C(  1042099677),  INT32_C(     8044461) },
      { -INT32_C(  1002011803), -INT32_C(   948337069), -INT32_C(  1201689674),  INT32_C(  1734086829) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1042099677),  INT32_C(           0) } },
    { UINT8_C(243),
      {  INT32_C(   846737751),  INT32_C(  1599466125), -INT32_C(  1962142004),  INT32_C(   334564496) },
      { -INT32_C(   664357550), -INT32_C(   628216273), -INT32_C(  1652078963), -INT32_C(   107942238) },
      {  INT32_C(   846737751),  INT32_C(  1599466125),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_max_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C(  393583728673787521), -INT64_C(  950936937819902977) },
      UINT8_C( 11),
      {  INT64_C( 8046248441115452223),  INT64_C(  788248241575570872) },
      {  INT64_C(  299124995156217969), -INT64_C( 1961077236624206042) },
      {  INT64_C( 8046248441115452223),  INT64_C(  788248241575570872) } },
    { { -INT64_C( 4629671798220323843), -INT64_C( 8220440291225522105) },
      UINT8_C( 49),
      { -INT64_C( 5584236100734765915),  INT64_C(  688551592900050358) },
      {  INT64_C( 5623816840503418538),  INT64_C( 8958776587137510400) },
      {  INT64_C( 5623816840503418538), -INT64_C( 8220440291225522105) } },
    { { -INT64_C( 6685780647462232405),  INT64_C( 8260377733858823827) },
      UINT8_C(160),
      {  INT64_C( 3351396389492766386),  INT64_C( 1556418048083469559) },
      { -INT64_C( 1999697917857620394),  INT64_C( 3186894297759537368) },
      { -INT64_C( 6685780647462232405),  INT64_C( 8260377733858823827) } },
    { { -INT64_C(  232728104673642411), -INT64_C( 6710219083215090989) },
      UINT8_C(162),
      {  INT64_C( 7305081096024641250), -INT64_C( 5981612409295471992) },
      {  INT64_C( 2646900911484605624),  INT64_C( 5313704844411831961) },
      { -INT64_C(  232728104673642411),  INT64_C( 5313704844411831961) } },
    { {  INT64_C( 1968690494239258208), -INT64_C(  861384637685812949) },
      UINT8_C( 84),
      {  INT64_C( 2163366748121652413), -INT64_C( 7587210909814808254) },
      {  INT64_C( 2665080893473399200),  INT64_C( 2601485634663111430) },
      {  INT64_C( 1968690494239258208), -INT64_C(  861384637685812949) } },
    { { -INT64_C( 2456549104691394511), -INT64_C( 2581153863521178186) },
      UINT8_C(135),
      {  INT64_C(  333441923215502678),  INT64_C( 5226022198961726976) },
      { -INT64_C( 6527193360134615009), -INT64_C( 8908192429515865615) },
      {  INT64_C(  333441923215502678),  INT64_C( 5226022198961726976) } },
    { {  INT64_C( 8726496655238421100),  INT64_C( 3742021887560942885) },
      UINT8_C(165),
      {  INT64_C( 5945457744933677009), -INT64_C( 7814021635253446044) },
      {  INT64_C( 4829695804779045922),  INT64_C( 7660212408779516756) },
      {  INT64_C( 5945457744933677009),  INT64_C( 3742021887560942885) } },
    { {  INT64_C( 1303086956271053681), -INT64_C( 4817509533239372072) },
      UINT8_C(105),
      {  INT64_C( 5695764508914982448),  INT64_C( 3368220844976878786) },
      {  INT64_C( 3289298538136895823), -INT64_C( 8479569622381152732) },
      {  INT64_C( 5695764508914982448), -INT64_C( 4817509533239372072) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_epi64");
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
    easysimd__m128i r = easysimd_mm_mask_max_epi64(src, k, a, b);

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
test_easysimd_mm_maskz_max_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C( 31),
      { -INT64_C( 7584612073990247672), -INT64_C( 1119363056205936104) },
      {  INT64_C( 2619963443093917600), -INT64_C( 7957930320005171071) },
      {  INT64_C( 2619963443093917600), -INT64_C( 1119363056205936104) } },
    { UINT8_C(236),
      { -INT64_C( 8331937202411109316),  INT64_C( 5333309773587121193) },
      {  INT64_C( 2617520663315403223), -INT64_C( 8160452899807313132) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(195),
      { -INT64_C( 5090877146629024211), -INT64_C( 8004585666573452199) },
      { -INT64_C( 3420236469078541785),  INT64_C( 5372989097006816654) },
      { -INT64_C( 3420236469078541785),  INT64_C( 5372989097006816654) } },
    { UINT8_C(135),
      { -INT64_C( 8034009606663603562),  INT64_C( 3914156375052161835) },
      { -INT64_C( 6121495466144905571),  INT64_C(   31546547301374519) },
      { -INT64_C( 6121495466144905571),  INT64_C( 3914156375052161835) } },
    { UINT8_C(142),
      {  INT64_C( 4454045371208968237),  INT64_C( 1207227801841301826) },
      {  INT64_C( 2629832021565718067),  INT64_C(  465436302558324947) },
      {  INT64_C(                   0),  INT64_C( 1207227801841301826) } },
    { UINT8_C(184),
      {  INT64_C( 8218118765462858455),  INT64_C( 3277439581426626392) },
      { -INT64_C( 6121908935244397780),  INT64_C( 4427753557479190827) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 53),
      {  INT64_C( 4812976994204387284),  INT64_C( 5897567678892151774) },
      {  INT64_C(  836611909784637397), -INT64_C( 7024859189621991486) },
      {  INT64_C( 4812976994204387284),  INT64_C(                   0) } },
    { UINT8_C(182),
      { -INT64_C( 1139679135513636560), -INT64_C( 4436098737638548676) },
      { -INT64_C( 3142414687713631902),  INT64_C( 7186966765927051352) },
      {  INT64_C(                   0),  INT64_C( 7186966765927051352) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_max_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t src[16];
    const uint16_t k;
    const uint8_t a[16];
    const uint8_t b[16];
    const uint8_t r[16];
  } test_vec[] = {
    { { UINT8_C( 50), UINT8_C( 16), UINT8_C(225), UINT8_C(231), UINT8_C( 37), UINT8_C( 54), UINT8_C( 26), UINT8_C(142),
        UINT8_C(243), UINT8_C( 56), UINT8_C(190), UINT8_C( 78), UINT8_C(239), UINT8_C(203), UINT8_C(197), UINT8_C( 33) },
      UINT16_C( 8029),
      { UINT8_C(163), UINT8_C(172), UINT8_C(130), UINT8_C(180), UINT8_C( 40), UINT8_C( 10), UINT8_C( 13), UINT8_C( 34),
        UINT8_C(215), UINT8_C( 27), UINT8_C(249), UINT8_C(244), UINT8_C(167), UINT8_C( 44), UINT8_C(  5), UINT8_C(136) },
      { UINT8_C( 19), UINT8_C( 42), UINT8_C(191), UINT8_C( 45), UINT8_C(185), UINT8_C(178), UINT8_C(101), UINT8_C(119),
        UINT8_C(  0), UINT8_C( 84), UINT8_C( 67), UINT8_C(198), UINT8_C(117), UINT8_C(160), UINT8_C(229), UINT8_C( 24) },
      { UINT8_C(163), UINT8_C( 16), UINT8_C(191), UINT8_C(180), UINT8_C(185), UINT8_C( 54), UINT8_C(101), UINT8_C(142),
        UINT8_C(215), UINT8_C( 84), UINT8_C(249), UINT8_C(244), UINT8_C(167), UINT8_C(203), UINT8_C(197), UINT8_C( 33) } },
    { { UINT8_C( 76), UINT8_C(103), UINT8_C(204), UINT8_C(116), UINT8_C(113), UINT8_C(217), UINT8_C(150), UINT8_C( 73),
        UINT8_C(245), UINT8_C(144), UINT8_C( 61), UINT8_C(156), UINT8_C(188), UINT8_C( 66), UINT8_C( 36), UINT8_C(207) },
      UINT16_C(58221),
      { UINT8_C(252), UINT8_C( 38), UINT8_C(149), UINT8_C( 98), UINT8_C(157), UINT8_C(150), UINT8_C(182), UINT8_C(224),
        UINT8_C( 92), UINT8_C( 44), UINT8_C(128), UINT8_C( 65), UINT8_C( 68), UINT8_C(204), UINT8_C(168), UINT8_C( 17) },
      { UINT8_C( 64), UINT8_C( 25), UINT8_C(234), UINT8_C(215), UINT8_C( 98), UINT8_C(223), UINT8_C(103), UINT8_C(160),
        UINT8_C(123), UINT8_C( 35), UINT8_C(226), UINT8_C(160), UINT8_C(242), UINT8_C( 79), UINT8_C(131), UINT8_C(238) },
      { UINT8_C(252), UINT8_C(103), UINT8_C(234), UINT8_C(215), UINT8_C(113), UINT8_C(223), UINT8_C(182), UINT8_C( 73),
        UINT8_C(123), UINT8_C( 44), UINT8_C( 61), UINT8_C(156), UINT8_C(188), UINT8_C(204), UINT8_C(168), UINT8_C(238) } },
    { { UINT8_C(117), UINT8_C( 25), UINT8_C( 80), UINT8_C( 19), UINT8_C(175), UINT8_C(  7), UINT8_C(243), UINT8_C( 11),
        UINT8_C( 51), UINT8_C(116), UINT8_C( 76), UINT8_C(119), UINT8_C( 64), UINT8_C(244), UINT8_C(136), UINT8_C(129) },
      UINT16_C(29453),
      { UINT8_C( 88), UINT8_C(112), UINT8_C( 82), UINT8_C(191), UINT8_C( 16), UINT8_C(206), UINT8_C(226), UINT8_C(242),
        UINT8_C(110), UINT8_C(212), UINT8_C( 66), UINT8_C(241), UINT8_C(194), UINT8_C(183), UINT8_C( 10), UINT8_C( 19) },
      { UINT8_C(202), UINT8_C(185), UINT8_C( 26), UINT8_C(190), UINT8_C(196), UINT8_C( 77), UINT8_C( 50), UINT8_C( 16),
        UINT8_C(196), UINT8_C(114), UINT8_C(  4), UINT8_C( 77), UINT8_C(243), UINT8_C( 18), UINT8_C(192), UINT8_C( 75) },
      { UINT8_C(202), UINT8_C( 25), UINT8_C( 82), UINT8_C(191), UINT8_C(175), UINT8_C(  7), UINT8_C(243), UINT8_C( 11),
        UINT8_C(196), UINT8_C(212), UINT8_C( 76), UINT8_C(119), UINT8_C(243), UINT8_C(183), UINT8_C(192), UINT8_C(129) } },
    { { UINT8_C(130), UINT8_C( 18), UINT8_C( 10), UINT8_C(146), UINT8_C(224), UINT8_C(236), UINT8_C(132), UINT8_C( 78),
        UINT8_C(192), UINT8_C(198), UINT8_C( 64), UINT8_C(131), UINT8_C(126), UINT8_C( 74), UINT8_C(150), UINT8_C( 72) },
      UINT16_C(45060),
      { UINT8_C(  6), UINT8_C(200), UINT8_C(253), UINT8_C( 56), UINT8_C(217), UINT8_C(193), UINT8_C(171), UINT8_C(221),
        UINT8_C( 14), UINT8_C(158), UINT8_C(239), UINT8_C(206), UINT8_C(234), UINT8_C(113), UINT8_C(225), UINT8_C(244) },
      { UINT8_C(  3), UINT8_C(193), UINT8_C(225), UINT8_C(136), UINT8_C( 16), UINT8_C(161), UINT8_C( 78), UINT8_C( 80),
        UINT8_C( 36), UINT8_C(204), UINT8_C(154), UINT8_C(186), UINT8_C( 21), UINT8_C(158), UINT8_C(106), UINT8_C( 27) },
      { UINT8_C(130), UINT8_C( 18), UINT8_C(253), UINT8_C(146), UINT8_C(224), UINT8_C(236), UINT8_C(132), UINT8_C( 78),
        UINT8_C(192), UINT8_C(198), UINT8_C( 64), UINT8_C(131), UINT8_C(234), UINT8_C(158), UINT8_C(150), UINT8_C(244) } },
    { { UINT8_C(103), UINT8_C(103), UINT8_C( 84), UINT8_C( 64), UINT8_C( 41),    UINT8_MAX, UINT8_C( 29), UINT8_C( 55),
        UINT8_C(157), UINT8_C( 13), UINT8_C(  6), UINT8_C(135), UINT8_C(126), UINT8_C(231), UINT8_C(124), UINT8_C(130) },
      UINT16_C(23976),
      { UINT8_C( 10), UINT8_C(184), UINT8_C(254), UINT8_C( 88), UINT8_C(  8), UINT8_C( 35), UINT8_C( 37), UINT8_C(163),
        UINT8_C(221), UINT8_C( 58), UINT8_C( 65), UINT8_C( 72), UINT8_C( 85), UINT8_C(168), UINT8_C(175), UINT8_C(169) },
      { UINT8_C(232), UINT8_C(216), UINT8_C(168), UINT8_C(  6), UINT8_C( 16), UINT8_C( 70), UINT8_C( 19), UINT8_C( 22),
        UINT8_C(205), UINT8_C(145), UINT8_C(253), UINT8_C( 73), UINT8_C( 19), UINT8_C(165), UINT8_C(166), UINT8_C( 29) },
      { UINT8_C(103), UINT8_C(103), UINT8_C( 84), UINT8_C( 88), UINT8_C( 41), UINT8_C( 70), UINT8_C( 29), UINT8_C(163),
        UINT8_C(221), UINT8_C( 13), UINT8_C(253), UINT8_C( 73), UINT8_C( 85), UINT8_C(231), UINT8_C(175), UINT8_C(130) } },
    { { UINT8_C( 94), UINT8_C(165), UINT8_C(118), UINT8_C(102), UINT8_C(200), UINT8_C(155), UINT8_C(  9), UINT8_C(165),
        UINT8_C(213), UINT8_C( 75), UINT8_C(237), UINT8_C( 42), UINT8_C(243), UINT8_C(157), UINT8_C(212), UINT8_C(220) },
      UINT16_C(31861),
      { UINT8_C(226), UINT8_C(133), UINT8_C(194), UINT8_C(245), UINT8_C(155), UINT8_C(144), UINT8_C(134), UINT8_C(152),
        UINT8_C(217), UINT8_C(154), UINT8_C( 62), UINT8_C(128), UINT8_C(183), UINT8_C(156), UINT8_C( 37), UINT8_C( 45) },
      { UINT8_C(  2), UINT8_C(237), UINT8_C(200), UINT8_C( 12), UINT8_C(146), UINT8_C(157), UINT8_C( 87), UINT8_C(128),
        UINT8_C(200), UINT8_C( 74), UINT8_C( 29), UINT8_C(156), UINT8_C( 38), UINT8_C(146), UINT8_C( 24), UINT8_C(  8) },
      { UINT8_C(226), UINT8_C(165), UINT8_C(200), UINT8_C(102), UINT8_C(155), UINT8_C(157), UINT8_C(134), UINT8_C(165),
        UINT8_C(213), UINT8_C( 75), UINT8_C( 62), UINT8_C(156), UINT8_C(183), UINT8_C(156), UINT8_C( 37), UINT8_C(220) } },
    { { UINT8_C( 24), UINT8_C(219), UINT8_C(253), UINT8_C(179), UINT8_C(107), UINT8_C(132), UINT8_C( 76), UINT8_C( 68),
        UINT8_C( 30), UINT8_C(138), UINT8_C(196), UINT8_C(213), UINT8_C( 38), UINT8_C(233), UINT8_C(  3), UINT8_C( 40) },
      UINT16_C(52182),
      { UINT8_C( 52), UINT8_C(105), UINT8_C(105), UINT8_C(139), UINT8_C(233), UINT8_C( 49), UINT8_C(214), UINT8_C(  6),
        UINT8_C(205), UINT8_C(252), UINT8_C(152), UINT8_C(229), UINT8_C(  5), UINT8_C(176), UINT8_C(192), UINT8_C(  2) },
      { UINT8_C(100), UINT8_C( 43), UINT8_C(134), UINT8_C(176), UINT8_C(112), UINT8_C(164), UINT8_C( 58), UINT8_C( 52),
        UINT8_C(122), UINT8_C( 96), UINT8_C( 30), UINT8_C(125), UINT8_C(136), UINT8_C(244), UINT8_C( 72), UINT8_C(189) },
      { UINT8_C( 24), UINT8_C(105), UINT8_C(134), UINT8_C(179), UINT8_C(233), UINT8_C(132), UINT8_C(214), UINT8_C( 52),
        UINT8_C(205), UINT8_C(252), UINT8_C(196), UINT8_C(229), UINT8_C( 38), UINT8_C(233), UINT8_C(192), UINT8_C(189) } },
    { { UINT8_C( 93), UINT8_C(177), UINT8_C( 72), UINT8_C( 70), UINT8_C(226), UINT8_C( 30), UINT8_C( 76), UINT8_C(175),
        UINT8_C( 27), UINT8_C(229), UINT8_C(149), UINT8_C( 32), UINT8_C(149), UINT8_C( 85), UINT8_C( 34), UINT8_C(249) },
      UINT16_C(43393),
      { UINT8_C(169), UINT8_C(241), UINT8_C( 77), UINT8_C(227), UINT8_C( 37), UINT8_C(199), UINT8_C( 67), UINT8_C( 67),
        UINT8_C( 68), UINT8_C(204), UINT8_C( 56), UINT8_C(141), UINT8_C(137), UINT8_C(149), UINT8_C( 62), UINT8_C(209) },
      { UINT8_C(220), UINT8_C( 33), UINT8_C(240), UINT8_C( 40), UINT8_C(208), UINT8_C( 11), UINT8_C( 13), UINT8_C(101),
        UINT8_C( 43), UINT8_C(163), UINT8_C(187), UINT8_C( 77), UINT8_C(156), UINT8_C( 60), UINT8_C(246), UINT8_C( 70) },
      { UINT8_C(220), UINT8_C(177), UINT8_C( 72), UINT8_C( 70), UINT8_C(226), UINT8_C( 30), UINT8_C( 76), UINT8_C(101),
        UINT8_C( 68), UINT8_C(229), UINT8_C(149), UINT8_C(141), UINT8_C(149), UINT8_C(149), UINT8_C( 34), UINT8_C(209) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi8(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_epu8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_epu8");
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
    easysimd__m128i r = easysimd_mm_mask_max_epu8(src, k, a, b);

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
test_easysimd_mm_maskz_max_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const uint8_t a[16];
    const uint8_t b[16];
    const uint8_t r[16];
  } test_vec[] = {
    { UINT16_C(34381),
      { UINT8_C(180), UINT8_C( 42), UINT8_C(145), UINT8_C(228), UINT8_C( 26), UINT8_C(205), UINT8_C(191), UINT8_C(152),
        UINT8_C(103), UINT8_C(114), UINT8_C(103), UINT8_C(214), UINT8_C( 52), UINT8_C(202), UINT8_C(183), UINT8_C( 15) },
      { UINT8_C(107), UINT8_C( 53), UINT8_C(246), UINT8_C(206), UINT8_C(  9), UINT8_C( 78), UINT8_C(127), UINT8_C(167),
        UINT8_C( 85), UINT8_C(177), UINT8_C(227), UINT8_C( 18), UINT8_C( 20), UINT8_C( 49), UINT8_C(152), UINT8_C(201) },
      { UINT8_C(180), UINT8_C(  0), UINT8_C(246), UINT8_C(228), UINT8_C(  0), UINT8_C(  0), UINT8_C(191), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(177), UINT8_C(227), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(201) } },
    { UINT16_C(10587),
      { UINT8_C(173), UINT8_C(118), UINT8_C(246), UINT8_C(108), UINT8_C( 14), UINT8_C( 93), UINT8_C(222), UINT8_C(118),
        UINT8_C( 51), UINT8_C( 18), UINT8_C( 64), UINT8_C(235), UINT8_C( 33), UINT8_C(171), UINT8_C( 32), UINT8_C( 23) },
      { UINT8_C(121), UINT8_C( 41), UINT8_C(101), UINT8_C(248), UINT8_C(209), UINT8_C(186), UINT8_C(170), UINT8_C(180),
        UINT8_C(204), UINT8_C(190), UINT8_C(229), UINT8_C(100), UINT8_C(135), UINT8_C( 65), UINT8_C(141), UINT8_C( 52) },
      { UINT8_C(173), UINT8_C(118), UINT8_C(  0), UINT8_C(248), UINT8_C(209), UINT8_C(  0), UINT8_C(222), UINT8_C(  0),
        UINT8_C(204), UINT8_C(  0), UINT8_C(  0), UINT8_C(235), UINT8_C(  0), UINT8_C(171), UINT8_C(  0), UINT8_C(  0) } },
    { UINT16_C(33719),
      { UINT8_C(160), UINT8_C(197), UINT8_C(224), UINT8_C(126), UINT8_C( 59), UINT8_C( 20), UINT8_C(144), UINT8_C(123),
           UINT8_MAX, UINT8_C(178), UINT8_C( 38), UINT8_C( 31), UINT8_C(201), UINT8_C(160), UINT8_C( 72), UINT8_C( 47) },
      { UINT8_C(152), UINT8_C( 25), UINT8_C(233), UINT8_C( 66), UINT8_C(206), UINT8_C(182), UINT8_C(  1), UINT8_C(179),
        UINT8_C( 26), UINT8_C(136), UINT8_C(244), UINT8_C(168), UINT8_C(189), UINT8_C(171), UINT8_C( 43), UINT8_C( 93) },
      { UINT8_C(160), UINT8_C(197), UINT8_C(233), UINT8_C(  0), UINT8_C(206), UINT8_C(182), UINT8_C(  0), UINT8_C(179),
           UINT8_MAX, UINT8_C(178), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 93) } },
    { UINT16_C( 3185),
      { UINT8_C(220), UINT8_C(172), UINT8_C( 32), UINT8_C(108), UINT8_C( 40), UINT8_C( 31), UINT8_C( 30), UINT8_C( 78),
        UINT8_C( 62), UINT8_C(232), UINT8_C(238), UINT8_C(134), UINT8_C( 23), UINT8_C(135), UINT8_C(160), UINT8_C(  0) },
      { UINT8_C(201), UINT8_C(110), UINT8_C(182), UINT8_C(202), UINT8_C( 33), UINT8_C(209), UINT8_C( 83), UINT8_C( 22),
        UINT8_C(121), UINT8_C( 16), UINT8_C(193), UINT8_C(164), UINT8_C(109), UINT8_C( 50), UINT8_C(176), UINT8_C( 73) },
      { UINT8_C(220), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 40), UINT8_C(209), UINT8_C( 83), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(238), UINT8_C(164), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0) } },
    { UINT16_C(53471),
      { UINT8_C(182), UINT8_C(  7), UINT8_C(239), UINT8_C(212), UINT8_C( 85), UINT8_C( 45), UINT8_C(188), UINT8_C( 68),
        UINT8_C(180), UINT8_C(211), UINT8_C(203), UINT8_C( 84), UINT8_C(212), UINT8_C(148), UINT8_C(194), UINT8_C(138) },
      { UINT8_C( 95), UINT8_C(227), UINT8_C( 91), UINT8_C(178), UINT8_C(249), UINT8_C(212), UINT8_C(194), UINT8_C(187),
        UINT8_C(121), UINT8_C( 47), UINT8_C(237), UINT8_C( 41), UINT8_C(121), UINT8_C(204), UINT8_C(250), UINT8_C( 47) },
      { UINT8_C(182), UINT8_C(227), UINT8_C(239), UINT8_C(212), UINT8_C(249), UINT8_C(  0), UINT8_C(194), UINT8_C(187),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(212), UINT8_C(  0), UINT8_C(250), UINT8_C(138) } },
    { UINT16_C(59859),
      { UINT8_C(  3), UINT8_C( 41), UINT8_C( 23), UINT8_C(192), UINT8_C(109), UINT8_C(203), UINT8_C(147), UINT8_C( 56),
        UINT8_C( 31), UINT8_C(103), UINT8_C(204), UINT8_C(225), UINT8_C(242), UINT8_C( 43), UINT8_C(196), UINT8_C( 77) },
      { UINT8_C(221), UINT8_C(190), UINT8_C( 34), UINT8_C(159), UINT8_C(121), UINT8_C(155), UINT8_C(207), UINT8_C(102),
        UINT8_C(196), UINT8_C( 72), UINT8_C( 51), UINT8_C(190), UINT8_C(119), UINT8_C(  6), UINT8_C(168), UINT8_C(122) },
      { UINT8_C(221), UINT8_C(190), UINT8_C(  0), UINT8_C(  0), UINT8_C(121), UINT8_C(  0), UINT8_C(207), UINT8_C(102),
        UINT8_C(196), UINT8_C(  0), UINT8_C(  0), UINT8_C(225), UINT8_C(  0), UINT8_C( 43), UINT8_C(196), UINT8_C(122) } },
    { UINT16_C(48943),
      { UINT8_C( 58), UINT8_C(156), UINT8_C(138), UINT8_C(206), UINT8_C(212), UINT8_C(169), UINT8_C( 53), UINT8_C(161),
        UINT8_C(138), UINT8_C( 39), UINT8_C(204), UINT8_C( 78), UINT8_C(117), UINT8_C(170), UINT8_C( 12), UINT8_C(151) },
      { UINT8_C( 73), UINT8_C(133), UINT8_C( 50), UINT8_C( 24), UINT8_C(236), UINT8_C(246), UINT8_C( 96), UINT8_C( 31),
        UINT8_C(181), UINT8_C(215), UINT8_C( 37), UINT8_C( 93), UINT8_C( 82), UINT8_C( 85), UINT8_C( 28), UINT8_C(140) },
      { UINT8_C( 73), UINT8_C(156), UINT8_C(138), UINT8_C(206), UINT8_C(  0), UINT8_C(246), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(181), UINT8_C(215), UINT8_C(204), UINT8_C( 93), UINT8_C(117), UINT8_C(170), UINT8_C(  0), UINT8_C(151) } },
    { UINT16_C(42737),
      { UINT8_C( 90), UINT8_C(198), UINT8_C( 79), UINT8_C(144), UINT8_C(103), UINT8_C(217), UINT8_C(183), UINT8_C( 51),
        UINT8_C( 39), UINT8_C( 44), UINT8_C(221), UINT8_C( 52), UINT8_C(195), UINT8_C( 39), UINT8_C(185), UINT8_C(245) },
      { UINT8_C( 63), UINT8_C(165), UINT8_C(236), UINT8_C(160), UINT8_C(196), UINT8_C(161), UINT8_C(119), UINT8_C(234),
        UINT8_C(254), UINT8_C(201), UINT8_C( 63), UINT8_C( 26), UINT8_C( 86), UINT8_C( 48), UINT8_C(192), UINT8_C(176) },
      { UINT8_C( 90), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(196), UINT8_C(217), UINT8_C(183), UINT8_C(234),
        UINT8_C(  0), UINT8_C(201), UINT8_C(221), UINT8_C(  0), UINT8_C(  0), UINT8_C( 48), UINT8_C(  0), UINT8_C(245) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi8(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_epu8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_epu8");
    easysimd_test_x86_assert_equal_u8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_test_x86_random_u8x16();
    easysimd__m128i r = easysimd_mm_maskz_max_epu8(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t src[8];
    const uint8_t k;
    const uint16_t a[8];
    const uint16_t b[8];
    const uint16_t r[8];
  } test_vec[] = {
    { { UINT16_C(17453), UINT16_C(21033), UINT16_C(27915), UINT16_C(20630), UINT16_C(52793), UINT16_C(49885), UINT16_C( 7011), UINT16_C(16275) },
      UINT8_C( 60),
      { UINT16_C(26755), UINT16_C(36365), UINT16_C(29301), UINT16_C( 6329), UINT16_C( 1837), UINT16_C(27061), UINT16_C(64509), UINT16_C(16790) },
      { UINT16_C(59684), UINT16_C(37197), UINT16_C(40319), UINT16_C(19914), UINT16_C(35962), UINT16_C(38320), UINT16_C(61472), UINT16_C(41938) },
      { UINT16_C(17453), UINT16_C(21033), UINT16_C(40319), UINT16_C(19914), UINT16_C(35962), UINT16_C(38320), UINT16_C( 7011), UINT16_C(16275) } },
    { { UINT16_C(57176), UINT16_C(52530), UINT16_C(60241), UINT16_C(32742), UINT16_C(39922), UINT16_C(61672), UINT16_C(32662), UINT16_C(47665) },
      UINT8_C(104),
      { UINT16_C(19582), UINT16_C( 7143), UINT16_C(13334), UINT16_C(41877), UINT16_C(11236), UINT16_C(54467), UINT16_C(26365), UINT16_C(56364) },
      { UINT16_C(64152), UINT16_C(33837), UINT16_C(44256), UINT16_C(31606), UINT16_C(26261), UINT16_C( 5137), UINT16_C(52120), UINT16_C( 5756) },
      { UINT16_C(57176), UINT16_C(52530), UINT16_C(60241), UINT16_C(41877), UINT16_C(39922), UINT16_C(54467), UINT16_C(52120), UINT16_C(47665) } },
    { { UINT16_C(25367), UINT16_C(11826), UINT16_C(51095), UINT16_C(31697), UINT16_C(38130), UINT16_C(61264), UINT16_C(31994), UINT16_C(37835) },
      UINT8_C(118),
      { UINT16_C( 6137), UINT16_C(42326), UINT16_C(53645), UINT16_C(62522), UINT16_C(20194), UINT16_C(44684), UINT16_C(41674), UINT16_C(11717) },
      { UINT16_C(62420), UINT16_C(40132), UINT16_C(16580), UINT16_C(22670), UINT16_C(32400), UINT16_C( 3155), UINT16_C(58953), UINT16_C(17027) },
      { UINT16_C(25367), UINT16_C(42326), UINT16_C(53645), UINT16_C(31697), UINT16_C(32400), UINT16_C(44684), UINT16_C(58953), UINT16_C(37835) } },
    { { UINT16_C(55805), UINT16_C(35560), UINT16_C( 8875), UINT16_C(36222), UINT16_C( 2673), UINT16_C(15163), UINT16_C(  429), UINT16_C(33129) },
      UINT8_C(244),
      { UINT16_C( 7469), UINT16_C(28089), UINT16_C( 4524), UINT16_C(11005), UINT16_C( 2660), UINT16_C(19059), UINT16_C(46733), UINT16_C(26183) },
      { UINT16_C(53918), UINT16_C(49169), UINT16_C(40784), UINT16_C(23345), UINT16_C(28122), UINT16_C(56072), UINT16_C(35286), UINT16_C(  976) },
      { UINT16_C(55805), UINT16_C(35560), UINT16_C(40784), UINT16_C(36222), UINT16_C(28122), UINT16_C(56072), UINT16_C(46733), UINT16_C(26183) } },
    { { UINT16_C(35239), UINT16_C(21361), UINT16_C(28314), UINT16_C(65405), UINT16_C(61560), UINT16_C( 1353), UINT16_C(37286), UINT16_C(17516) },
      UINT8_C( 99),
      { UINT16_C( 1405), UINT16_C( 7347), UINT16_C( 3638), UINT16_C(41975), UINT16_C(53782), UINT16_C(41081), UINT16_C(32162), UINT16_C(11079) },
      { UINT16_C(39662), UINT16_C(23750), UINT16_C(50455), UINT16_C( 2005), UINT16_C(55822), UINT16_C(40878), UINT16_C(62022), UINT16_C(50178) },
      { UINT16_C(39662), UINT16_C(23750), UINT16_C(28314), UINT16_C(65405), UINT16_C(61560), UINT16_C(41081), UINT16_C(62022), UINT16_C(17516) } },
    { { UINT16_C(46839), UINT16_C(12000), UINT16_C(55236), UINT16_C(56273), UINT16_C(19370), UINT16_C(19579), UINT16_C(49864), UINT16_C(46712) },
      UINT8_C( 92),
      { UINT16_C( 4670), UINT16_C(  883), UINT16_C(31463), UINT16_C(49681), UINT16_C(45352), UINT16_C( 6920), UINT16_C(52403), UINT16_C(26898) },
      { UINT16_C(16557), UINT16_C(33838), UINT16_C( 2322), UINT16_C(23854), UINT16_C(31620), UINT16_C(17957), UINT16_C(56307), UINT16_C(12706) },
      { UINT16_C(46839), UINT16_C(12000), UINT16_C(31463), UINT16_C(49681), UINT16_C(45352), UINT16_C(19579), UINT16_C(56307), UINT16_C(46712) } },
    { { UINT16_C( 5613), UINT16_C(54580), UINT16_C(17807), UINT16_C(47255), UINT16_C(40950), UINT16_C(43731), UINT16_C(58732), UINT16_C( 6419) },
      UINT8_C( 38),
      { UINT16_C(40257), UINT16_C(19000), UINT16_C(38348), UINT16_C(18382), UINT16_C( 5306), UINT16_C(38202), UINT16_C(27574), UINT16_C(52098) },
      { UINT16_C(22431), UINT16_C(58459), UINT16_C( 5102), UINT16_C(36571), UINT16_C(34278), UINT16_C(52218), UINT16_C( 5016), UINT16_C(56049) },
      { UINT16_C( 5613), UINT16_C(58459), UINT16_C(38348), UINT16_C(47255), UINT16_C(40950), UINT16_C(52218), UINT16_C(58732), UINT16_C( 6419) } },
    { { UINT16_C(10672), UINT16_C(31780), UINT16_C(62398), UINT16_C(30915), UINT16_C(64775), UINT16_C(48653), UINT16_C(36968), UINT16_C( 1929) },
      UINT8_C(231),
      { UINT16_C(60644), UINT16_C(63446), UINT16_C(25799), UINT16_C(19677), UINT16_C(43358), UINT16_C(29156), UINT16_C(48794), UINT16_C(50209) },
      { UINT16_C(40675), UINT16_C(54914), UINT16_C(64353), UINT16_C(24541), UINT16_C(39688), UINT16_C(39111), UINT16_C(53029), UINT16_C( 2432) },
      { UINT16_C(60644), UINT16_C(63446), UINT16_C(64353), UINT16_C(30915), UINT16_C(64775), UINT16_C(39111), UINT16_C(53029), UINT16_C(50209) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_epu16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_epu16");
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
    easysimd__m128i r = easysimd_mm_mask_max_epu16(src, k, a, b);

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
test_easysimd_mm_maskz_max_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint16_t a[8];
    const uint16_t b[8];
    const uint16_t r[8];
  } test_vec[] = {
    { UINT8_C(246),
      { UINT16_C(16399), UINT16_C(59485), UINT16_C(37368), UINT16_C( 9231), UINT16_C(17262), UINT16_C(38376), UINT16_C(56829), UINT16_C(41685) },
      { UINT16_C(30153), UINT16_C(27239), UINT16_C(20972), UINT16_C(46696), UINT16_C(33424), UINT16_C(49164), UINT16_C(48194), UINT16_C(20919) },
      { UINT16_C(    0), UINT16_C(59485), UINT16_C(37368), UINT16_C(    0), UINT16_C(33424), UINT16_C(49164), UINT16_C(56829), UINT16_C(41685) } },
    { UINT8_C(253),
      { UINT16_C(14612), UINT16_C(42485), UINT16_C( 6473), UINT16_C(35860), UINT16_C(43265), UINT16_C(57225), UINT16_C(11390), UINT16_C(62376) },
      { UINT16_C( 5011), UINT16_C(58592), UINT16_C(38523), UINT16_C(65140), UINT16_C(13474), UINT16_C(24128), UINT16_C(37611), UINT16_C(   91) },
      { UINT16_C(14612), UINT16_C(    0), UINT16_C(38523), UINT16_C(65140), UINT16_C(43265), UINT16_C(57225), UINT16_C(37611), UINT16_C(62376) } },
    { UINT8_C(203),
      { UINT16_C(42320), UINT16_C(27156), UINT16_C(41401), UINT16_C(25451), UINT16_C(18986), UINT16_C(22241), UINT16_C(54771), UINT16_C( 1769) },
      { UINT16_C(52661), UINT16_C(19329), UINT16_C(32577), UINT16_C(30445), UINT16_C(19392), UINT16_C(21089), UINT16_C(24999), UINT16_C(63261) },
      { UINT16_C(52661), UINT16_C(27156), UINT16_C(    0), UINT16_C(30445), UINT16_C(    0), UINT16_C(    0), UINT16_C(54771), UINT16_C(63261) } },
    { UINT8_C(  7),
      { UINT16_C(24882), UINT16_C(54208), UINT16_C( 9165), UINT16_C( 6141), UINT16_C(21509), UINT16_C(55818), UINT16_C( 4157), UINT16_C( 2959) },
      { UINT16_C(55954), UINT16_C( 4428), UINT16_C(49863), UINT16_C( 4817), UINT16_C( 8996), UINT16_C(34233), UINT16_C(45377), UINT16_C(29580) },
      { UINT16_C(55954), UINT16_C(54208), UINT16_C(49863), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C( 18),
      { UINT16_C(17997), UINT16_C(28895), UINT16_C(63299), UINT16_C(38773), UINT16_C(20225), UINT16_C( 4821), UINT16_C(57566), UINT16_C(47268) },
      { UINT16_C(46380), UINT16_C(61311), UINT16_C(37511), UINT16_C(43539), UINT16_C(38987), UINT16_C(64747), UINT16_C(24101), UINT16_C(29199) },
      { UINT16_C(    0), UINT16_C(61311), UINT16_C(    0), UINT16_C(    0), UINT16_C(38987), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT8_C(164),
      { UINT16_C(58094), UINT16_C(58856), UINT16_C(32600), UINT16_C(42983), UINT16_C(63828), UINT16_C(13446), UINT16_C(16029), UINT16_C(21089) },
      { UINT16_C(20670), UINT16_C(20697), UINT16_C(33891), UINT16_C(64411), UINT16_C(39023), UINT16_C(52768), UINT16_C(37543), UINT16_C(38258) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(33891), UINT16_C(    0), UINT16_C(    0), UINT16_C(52768), UINT16_C(    0), UINT16_C(38258) } },
    { UINT8_C(117),
      { UINT16_C(31578), UINT16_C(56013), UINT16_C(29794), UINT16_C(23342), UINT16_C(25594), UINT16_C(14840), UINT16_C(19140), UINT16_C( 5367) },
      { UINT16_C(18212), UINT16_C(43127), UINT16_C(29410), UINT16_C(31255), UINT16_C(58771), UINT16_C( 9505), UINT16_C(46936), UINT16_C(45722) },
      { UINT16_C(31578), UINT16_C(    0), UINT16_C(29794), UINT16_C(    0), UINT16_C(58771), UINT16_C(14840), UINT16_C(46936), UINT16_C(    0) } },
    { UINT8_C( 50),
      { UINT16_C(35943), UINT16_C(56468), UINT16_C(61371), UINT16_C( 7894), UINT16_C( 4071), UINT16_C(12770), UINT16_C(62982), UINT16_C(19797) },
      { UINT16_C(64877), UINT16_C(57136), UINT16_C(43541), UINT16_C(64114), UINT16_C(39116), UINT16_C(33618), UINT16_C( 1330), UINT16_C(39605) },
      { UINT16_C(    0), UINT16_C(57136), UINT16_C(    0), UINT16_C(    0), UINT16_C(39116), UINT16_C(33618), UINT16_C(    0), UINT16_C(    0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_epu16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_epu16");
    easysimd_test_x86_assert_equal_u16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    easysimd__m128i r = easysimd_mm_maskz_max_epu16(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t src[4];
    const uint8_t k;
    const uint32_t a[4];
    const uint32_t b[4];
    const uint32_t r[4];
  } test_vec[] = {
    { { UINT32_C(2181125819), UINT32_C( 416210618), UINT32_C( 579449479), UINT32_C(1424403057) },
      UINT8_C( 72),
      { UINT32_C(1672096360), UINT32_C(2741766407), UINT32_C(2680685776), UINT32_C(3663384964) },
      { UINT32_C(2979323090), UINT32_C(1563995306), UINT32_C(3771619893), UINT32_C(2837979712) },
      { UINT32_C(2181125819), UINT32_C( 416210618), UINT32_C( 579449479), UINT32_C(3663384964) } },
    { { UINT32_C(1393349196), UINT32_C(2901833947), UINT32_C(  21741181), UINT32_C(1658627728) },
      UINT8_C(130),
      { UINT32_C( 489493360), UINT32_C(2790427212), UINT32_C(2061972056), UINT32_C( 767987803) },
      { UINT32_C( 352917916), UINT32_C(3465721104), UINT32_C(2791215872), UINT32_C(3760832879) },
      { UINT32_C(1393349196), UINT32_C(3465721104), UINT32_C(  21741181), UINT32_C(1658627728) } },
    { { UINT32_C( 553473748), UINT32_C( 952586208), UINT32_C(3719474818), UINT32_C(3658119230) },
      UINT8_C(145),
      { UINT32_C(3366055699), UINT32_C( 365523073), UINT32_C(2407821262), UINT32_C(3999556760) },
      { UINT32_C(2983101537), UINT32_C(4180870731), UINT32_C( 808915128), UINT32_C( 784470554) },
      { UINT32_C(3366055699), UINT32_C( 952586208), UINT32_C(3719474818), UINT32_C(3658119230) } },
    { { UINT32_C(2213962497), UINT32_C(2727919571), UINT32_C(3341884463), UINT32_C(3803616641) },
      UINT8_C( 26),
      { UINT32_C(2338689924), UINT32_C(3594739654), UINT32_C(2834396310), UINT32_C(2578063158) },
      { UINT32_C(3580701973), UINT32_C(3775139781), UINT32_C(3596798784), UINT32_C( 116409729) },
      { UINT32_C(2213962497), UINT32_C(3775139781), UINT32_C(3341884463), UINT32_C(2578063158) } },
    { { UINT32_C(2677102040), UINT32_C(1266013364), UINT32_C(2129880648), UINT32_C(2602081669) },
      UINT8_C(202),
      { UINT32_C(2492428421), UINT32_C(1070887284), UINT32_C( 415345363), UINT32_C(4042377114) },
      { UINT32_C( 748982360), UINT32_C(1819668229), UINT32_C(2163340259), UINT32_C(2420870155) },
      { UINT32_C(2677102040), UINT32_C(1819668229), UINT32_C(2129880648), UINT32_C(4042377114) } },
    { { UINT32_C(1881463548), UINT32_C( 531691851), UINT32_C(1043820963), UINT32_C(2418944056) },
      UINT8_C(184),
      { UINT32_C(3267280082), UINT32_C( 631581233), UINT32_C(2821727515), UINT32_C(1269088624) },
      { UINT32_C(3734377957), UINT32_C( 914535877), UINT32_C( 359579885), UINT32_C(3234791150) },
      { UINT32_C(1881463548), UINT32_C( 531691851), UINT32_C(1043820963), UINT32_C(3234791150) } },
    { { UINT32_C(3967978682), UINT32_C(3507562422), UINT32_C(3178840397), UINT32_C(3892846082) },
      UINT8_C( 51),
      { UINT32_C(1425589919), UINT32_C( 138489416), UINT32_C(2599835548), UINT32_C(2975119141) },
      { UINT32_C(1634156601), UINT32_C(2477668433), UINT32_C(3499453362), UINT32_C( 318995828) },
      { UINT32_C(1634156601), UINT32_C(2477668433), UINT32_C(3178840397), UINT32_C(3892846082) } },
    { { UINT32_C(2355624772), UINT32_C(3314919721), UINT32_C( 627018496), UINT32_C(2094445378) },
      UINT8_C(244),
      { UINT32_C(1967512893), UINT32_C(4129806475), UINT32_C(3949655918), UINT32_C(4113530362) },
      { UINT32_C(2384379109), UINT32_C(3700351825), UINT32_C(4129272642), UINT32_C(3337264009) },
      { UINT32_C(2355624772), UINT32_C(3314919721), UINT32_C(4129272642), UINT32_C(2094445378) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_epu32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_epu32");
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
    easysimd__m128i r = easysimd_mm_mask_max_epu32(src, k, a, b);

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
test_easysimd_mm_maskz_max_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint32_t a[4];
    const uint32_t b[4];
    const uint32_t r[4];
  } test_vec[] = {
    { UINT8_C(145),
      { UINT32_C( 944535113), UINT32_C(1545562700), UINT32_C(1113739340), UINT32_C(2746200230) },
      { UINT32_C(2327351264), UINT32_C(2572595969), UINT32_C( 181197061), UINT32_C(3617351310) },
      { UINT32_C(2327351264), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C(220),
      { UINT32_C(1395134440), UINT32_C(2141160494), UINT32_C(2535842535), UINT32_C( 544721041) },
      { UINT32_C( 874643841), UINT32_C( 825867096), UINT32_C(3988800647), UINT32_C(3368654816) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(3988800647), UINT32_C(3368654816) } },
    { UINT8_C(167),
      { UINT32_C(1993677810), UINT32_C(2640139451), UINT32_C(1110373497), UINT32_C(1841516395) },
      { UINT32_C( 751171441), UINT32_C(1958016560), UINT32_C(1314169270), UINT32_C(1576344939) },
      { UINT32_C(1993677810), UINT32_C(2640139451), UINT32_C(1314169270), UINT32_C(         0) } },
    { UINT8_C( 56),
      { UINT32_C( 536073162), UINT32_C( 630755377), UINT32_C( 244439743), UINT32_C(2491416221) },
      { UINT32_C(3116739523), UINT32_C(  24131935), UINT32_C(2876030606), UINT32_C(2112080307) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(2491416221) } },
    { UINT8_C(156),
      { UINT32_C(1741528279), UINT32_C( 254210869), UINT32_C(2192389252), UINT32_C(1598374323) },
      { UINT32_C(1086258694), UINT32_C( 751746926), UINT32_C(4108286251), UINT32_C( 865164636) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(4108286251), UINT32_C(1598374323) } },
    { UINT8_C(249),
      { UINT32_C(1362008926), UINT32_C(4141170369), UINT32_C( 749295595), UINT32_C(2603813020) },
      { UINT32_C(2248766407), UINT32_C(3132241473), UINT32_C(1914086933), UINT32_C(2523679287) },
      { UINT32_C(2248766407), UINT32_C(         0), UINT32_C(         0), UINT32_C(2603813020) } },
    { UINT8_C(229),
      { UINT32_C(3634816922), UINT32_C( 348363965), UINT32_C(1320284230), UINT32_C(2517978147) },
      { UINT32_C(2346163285), UINT32_C(4104229198), UINT32_C(4046197671), UINT32_C( 450282111) },
      { UINT32_C(3634816922), UINT32_C(         0), UINT32_C(4046197671), UINT32_C(         0) } },
    { UINT8_C(169),
      { UINT32_C( 426177149), UINT32_C(2791275446), UINT32_C(2026483244), UINT32_C(1607294915) },
      { UINT32_C(3350026550), UINT32_C( 225354490), UINT32_C(2425184462), UINT32_C(3006900022) },
      { UINT32_C(3350026550), UINT32_C(         0), UINT32_C(         0), UINT32_C(3006900022) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_epu32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_epu32");
    easysimd_test_x86_assert_equal_u32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u32x4();
    easysimd__m128i b = easysimd_test_x86_random_u32x4();
    easysimd__m128i r = easysimd_mm_maskz_max_epu32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t src[2];
    const uint8_t k;
    const uint64_t a[2];
    const uint64_t b[2];
    const uint64_t r[2];
  } test_vec[] = {
    { { UINT64_C( 8572992251741548408), UINT64_C(14288393164772435034) },
      UINT8_C( 77),
      { UINT64_C(10776967545687528553), UINT64_C(11433764720253765017) },
      { UINT64_C(  998986206729908000), UINT64_C(16599341875323613822) },
      { UINT64_C(10776967545687528553), UINT64_C(14288393164772435034) } },
    { { UINT64_C( 1045329296449796617), UINT64_C(11841352673519640646) },
      UINT8_C(158),
      { UINT64_C( 4005199824458033146), UINT64_C(18014849313421960259) },
      { UINT64_C( 1140954899181137357), UINT64_C(15657952533523172031) },
      { UINT64_C( 1045329296449796617), UINT64_C(18014849313421960259) } },
    { { UINT64_C( 3142533873170158687), UINT64_C(13268490619101335593) },
      UINT8_C( 18),
      { UINT64_C(12019421426403591825), UINT64_C( 6609979988928396068) },
      { UINT64_C(  935464204381724749), UINT64_C(15693548574106657939) },
      { UINT64_C( 3142533873170158687), UINT64_C(15693548574106657939) } },
    { { UINT64_C(   51708778024226038), UINT64_C(16668412220393929775) },
      UINT8_C( 56),
      { UINT64_C(15679282880179386774), UINT64_C( 7466398753925189880) },
      { UINT64_C(11728574536961208289), UINT64_C(11740922766092450464) },
      { UINT64_C(   51708778024226038), UINT64_C(16668412220393929775) } },
    { { UINT64_C(15954917459625567452), UINT64_C( 7427318172279409809) },
      UINT8_C(209),
      { UINT64_C( 8807874017009073238), UINT64_C(14417475842291004182) },
      { UINT64_C(11431132089423865809), UINT64_C(13259313506779680527) },
      { UINT64_C(11431132089423865809), UINT64_C( 7427318172279409809) } },
    { { UINT64_C( 3008842239512361351), UINT64_C( 4840939082192406657) },
      UINT8_C( 31),
      { UINT64_C( 5149250414741378523), UINT64_C(16214474509236414195) },
      { UINT64_C( 6252155225020648629), UINT64_C( 3681705172151052929) },
      { UINT64_C( 6252155225020648629), UINT64_C(16214474509236414195) } },
    { { UINT64_C(  837966503412844897), UINT64_C( 7007194941173090996) },
      UINT8_C(218),
      { UINT64_C(10651643591239640115), UINT64_C( 3202656245620919045) },
      { UINT64_C(13221248459677682616), UINT64_C( 5206377864943097816) },
      { UINT64_C(  837966503412844897), UINT64_C( 5206377864943097816) } },
    { { UINT64_C( 7738710719286880361), UINT64_C( 2830774526772269112) },
      UINT8_C(185),
      { UINT64_C( 4662594956675181964), UINT64_C( 9470962655152118592) },
      { UINT64_C(17860243301247566136), UINT64_C(14733400015177709776) },
      { UINT64_C(17860243301247566136), UINT64_C( 2830774526772269112) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_epu64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_epu64");
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
    easysimd__m128i r = easysimd_mm_mask_max_epu64(src, k, a, b);

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
test_easysimd_mm_maskz_max_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint64_t a[2];
    const uint64_t b[2];
    const uint64_t r[2];
  } test_vec[] = {
    { UINT8_C(153),
      { UINT64_C(15728809793400327584), UINT64_C(16524830235986673598) },
      { UINT64_C(16797280409181166251), UINT64_C( 9564189735436397434) },
      { UINT64_C(16797280409181166251), UINT64_C(                   0) } },
    { UINT8_C(160),
      { UINT64_C(11943167872409050890), UINT64_C( 5326939425990695749) },
      { UINT64_C(   26664625396398839), UINT64_C( 4849929545192931291) },
      { UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C(  9),
      { UINT64_C(15854523372985598746), UINT64_C( 8202555990467103175) },
      { UINT64_C(10862338256683841831), UINT64_C(15284675875292867756) },
      { UINT64_C(15854523372985598746), UINT64_C(                   0) } },
    { UINT8_C( 86),
      { UINT64_C(16531083229576795662), UINT64_C(15545455176249526335) },
      { UINT64_C(14539455081942589267), UINT64_C(  630319205878848597) },
      { UINT64_C(                   0), UINT64_C(15545455176249526335) } },
    { UINT8_C(237),
      { UINT64_C( 6812379523603671831), UINT64_C( 8821517970585689995) },
      { UINT64_C( 2962700385822148162), UINT64_C( 6730678324313687587) },
      { UINT64_C( 6812379523603671831), UINT64_C(                   0) } },
    { UINT8_C(215),
      { UINT64_C( 5903457366316001522), UINT64_C(17896643498940827079) },
      { UINT64_C(14501712030370964861), UINT64_C( 1555268147077954449) },
      { UINT64_C(14501712030370964861), UINT64_C(17896643498940827079) } },
    { UINT8_C( 20),
      { UINT64_C( 7607873224948741762), UINT64_C( 3524297654913380036) },
      { UINT64_C(   78597792086558231), UINT64_C(16990737145392321611) },
      { UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C(176),
      { UINT64_C(18117555019654084625), UINT64_C(14357818389790382423) },
      { UINT64_C(12840061458331911850), UINT64_C(14131459672969570356) },
      { UINT64_C(                   0), UINT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_epu64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_epu64");
    easysimd_test_x86_assert_equal_u64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_u64x2();
    easysimd__m128i b = easysimd_test_x86_random_u64x2();
    easysimd__m128i r = easysimd_mm_maskz_max_epu64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   158.46), EASYSIMD_FLOAT32_C(  -973.50), EASYSIMD_FLOAT32_C(   653.58), EASYSIMD_FLOAT32_C(   636.02) },
      UINT8_C(  2),
      { EASYSIMD_FLOAT32_C(  -222.30), EASYSIMD_FLOAT32_C(  -316.28), EASYSIMD_FLOAT32_C(   712.92), EASYSIMD_FLOAT32_C(  -745.84) },
      { EASYSIMD_FLOAT32_C(   -68.80), EASYSIMD_FLOAT32_C(  -270.75), EASYSIMD_FLOAT32_C(   223.13), EASYSIMD_FLOAT32_C(   -29.80) },
      { EASYSIMD_FLOAT32_C(   158.46), EASYSIMD_FLOAT32_C(  -270.75), EASYSIMD_FLOAT32_C(   653.58), EASYSIMD_FLOAT32_C(   636.02) } },
    { { EASYSIMD_FLOAT32_C(   746.00), EASYSIMD_FLOAT32_C(  -578.71), EASYSIMD_FLOAT32_C(   327.09), EASYSIMD_FLOAT32_C(  -975.07) },
      UINT8_C( 23),
      { EASYSIMD_FLOAT32_C(   729.87), EASYSIMD_FLOAT32_C(   476.71), EASYSIMD_FLOAT32_C(   572.62), EASYSIMD_FLOAT32_C(   356.48) },
      { EASYSIMD_FLOAT32_C(   117.45), EASYSIMD_FLOAT32_C(   909.23), EASYSIMD_FLOAT32_C(   768.03), EASYSIMD_FLOAT32_C(   717.51) },
      { EASYSIMD_FLOAT32_C(   729.87), EASYSIMD_FLOAT32_C(   909.23), EASYSIMD_FLOAT32_C(   768.03), EASYSIMD_FLOAT32_C(  -975.07) } },
    { { EASYSIMD_FLOAT32_C(  -558.59), EASYSIMD_FLOAT32_C(   436.71), EASYSIMD_FLOAT32_C(   661.89), EASYSIMD_FLOAT32_C(  -699.11) },
      UINT8_C(147),
      { EASYSIMD_FLOAT32_C(  -179.65), EASYSIMD_FLOAT32_C(  -672.61), EASYSIMD_FLOAT32_C(  -955.54), EASYSIMD_FLOAT32_C(  -543.63) },
      { EASYSIMD_FLOAT32_C(  -968.49), EASYSIMD_FLOAT32_C(  -177.84), EASYSIMD_FLOAT32_C(   140.09), EASYSIMD_FLOAT32_C(   744.43) },
      { EASYSIMD_FLOAT32_C(  -179.65), EASYSIMD_FLOAT32_C(  -177.84), EASYSIMD_FLOAT32_C(   661.89), EASYSIMD_FLOAT32_C(  -699.11) } },
    { { EASYSIMD_FLOAT32_C(    76.32), EASYSIMD_FLOAT32_C(  -928.71), EASYSIMD_FLOAT32_C(  -526.31), EASYSIMD_FLOAT32_C(  -700.55) },
      UINT8_C( 26),
      { EASYSIMD_FLOAT32_C(  -780.31), EASYSIMD_FLOAT32_C(  -279.26), EASYSIMD_FLOAT32_C(  -631.42), EASYSIMD_FLOAT32_C(  -755.38) },
      { EASYSIMD_FLOAT32_C(   257.12), EASYSIMD_FLOAT32_C(  -901.55), EASYSIMD_FLOAT32_C(   721.33), EASYSIMD_FLOAT32_C(  -170.27) },
      { EASYSIMD_FLOAT32_C(    76.32), EASYSIMD_FLOAT32_C(  -279.26), EASYSIMD_FLOAT32_C(  -526.31), EASYSIMD_FLOAT32_C(  -170.27) } },
    { { EASYSIMD_FLOAT32_C(   454.93), EASYSIMD_FLOAT32_C(  -161.22), EASYSIMD_FLOAT32_C(  -261.04), EASYSIMD_FLOAT32_C(   222.96) },
      UINT8_C(210),
      { EASYSIMD_FLOAT32_C(   180.37), EASYSIMD_FLOAT32_C(  -340.33), EASYSIMD_FLOAT32_C(  -781.82), EASYSIMD_FLOAT32_C(   481.26) },
      { EASYSIMD_FLOAT32_C(    50.56), EASYSIMD_FLOAT32_C(    38.52), EASYSIMD_FLOAT32_C(   808.64), EASYSIMD_FLOAT32_C(    95.02) },
      { EASYSIMD_FLOAT32_C(   454.93), EASYSIMD_FLOAT32_C(    38.52), EASYSIMD_FLOAT32_C(  -261.04), EASYSIMD_FLOAT32_C(   222.96) } },
    { { EASYSIMD_FLOAT32_C(   494.89), EASYSIMD_FLOAT32_C(   840.15), EASYSIMD_FLOAT32_C(   917.18), EASYSIMD_FLOAT32_C(  -365.02) },
      UINT8_C(240),
      { EASYSIMD_FLOAT32_C(    -6.50), EASYSIMD_FLOAT32_C(  -293.74), EASYSIMD_FLOAT32_C(  -941.73), EASYSIMD_FLOAT32_C(   292.95) },
      { EASYSIMD_FLOAT32_C(   747.75), EASYSIMD_FLOAT32_C(  -722.04), EASYSIMD_FLOAT32_C(  -986.30), EASYSIMD_FLOAT32_C(  -883.67) },
      { EASYSIMD_FLOAT32_C(   494.89), EASYSIMD_FLOAT32_C(   840.15), EASYSIMD_FLOAT32_C(   917.18), EASYSIMD_FLOAT32_C(  -365.02) } },
    { { EASYSIMD_FLOAT32_C(  -477.42), EASYSIMD_FLOAT32_C(   270.81), EASYSIMD_FLOAT32_C(  -785.22), EASYSIMD_FLOAT32_C(  -756.09) },
      UINT8_C( 47),
      { EASYSIMD_FLOAT32_C(   669.71), EASYSIMD_FLOAT32_C(    82.69), EASYSIMD_FLOAT32_C(  -160.49), EASYSIMD_FLOAT32_C(  -107.34) },
      { EASYSIMD_FLOAT32_C(   638.98), EASYSIMD_FLOAT32_C(  -980.12), EASYSIMD_FLOAT32_C(   552.33), EASYSIMD_FLOAT32_C(   857.16) },
      { EASYSIMD_FLOAT32_C(   669.71), EASYSIMD_FLOAT32_C(    82.69), EASYSIMD_FLOAT32_C(   552.33), EASYSIMD_FLOAT32_C(   857.16) } },
    { { EASYSIMD_FLOAT32_C(   501.13), EASYSIMD_FLOAT32_C(  -397.11), EASYSIMD_FLOAT32_C(  -104.32), EASYSIMD_FLOAT32_C(   309.78) },
      UINT8_C( 41),
      { EASYSIMD_FLOAT32_C(  -609.43), EASYSIMD_FLOAT32_C(   149.93), EASYSIMD_FLOAT32_C(   615.09), EASYSIMD_FLOAT32_C(    25.55) },
      { EASYSIMD_FLOAT32_C(  -265.49), EASYSIMD_FLOAT32_C(  -391.41), EASYSIMD_FLOAT32_C(   731.82), EASYSIMD_FLOAT32_C(  -207.21) },
      { EASYSIMD_FLOAT32_C(  -265.49), EASYSIMD_FLOAT32_C(  -397.11), EASYSIMD_FLOAT32_C(  -104.32), EASYSIMD_FLOAT32_C(    25.55) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_ps");
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
    easysimd__m128 r = easysimd_mm_mask_max_ps(src, k, a, b);

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
test_easysimd_mm_maskz_max_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C( 92),
      { EASYSIMD_FLOAT32_C(  -179.80), EASYSIMD_FLOAT32_C(    33.05), EASYSIMD_FLOAT32_C(   786.39), EASYSIMD_FLOAT32_C(   673.44) },
      { EASYSIMD_FLOAT32_C(  -863.70), EASYSIMD_FLOAT32_C(    10.45), EASYSIMD_FLOAT32_C(   450.47), EASYSIMD_FLOAT32_C(    40.38) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   786.39), EASYSIMD_FLOAT32_C(   673.44) } },
    { UINT8_C( 20),
      { EASYSIMD_FLOAT32_C(   263.56), EASYSIMD_FLOAT32_C(   422.30), EASYSIMD_FLOAT32_C(  -552.11), EASYSIMD_FLOAT32_C(  -751.63) },
      { EASYSIMD_FLOAT32_C(  -942.76), EASYSIMD_FLOAT32_C(  -781.47), EASYSIMD_FLOAT32_C(   328.71), EASYSIMD_FLOAT32_C(  -592.04) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   328.71), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(119),
      { EASYSIMD_FLOAT32_C(    93.32), EASYSIMD_FLOAT32_C(   349.59), EASYSIMD_FLOAT32_C(  -768.76), EASYSIMD_FLOAT32_C(  -398.17) },
      { EASYSIMD_FLOAT32_C(  -854.62), EASYSIMD_FLOAT32_C(   229.19), EASYSIMD_FLOAT32_C(   927.83), EASYSIMD_FLOAT32_C(  -240.12) },
      { EASYSIMD_FLOAT32_C(    93.32), EASYSIMD_FLOAT32_C(   349.59), EASYSIMD_FLOAT32_C(   927.83), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(173),
      { EASYSIMD_FLOAT32_C(  -266.95), EASYSIMD_FLOAT32_C(  -697.90), EASYSIMD_FLOAT32_C(   404.69), EASYSIMD_FLOAT32_C(  -573.96) },
      { EASYSIMD_FLOAT32_C(   122.30), EASYSIMD_FLOAT32_C(  -562.26), EASYSIMD_FLOAT32_C(  -787.58), EASYSIMD_FLOAT32_C(  -204.26) },
      { EASYSIMD_FLOAT32_C(   122.30), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   404.69), EASYSIMD_FLOAT32_C(  -204.26) } },
    { UINT8_C(204),
      { EASYSIMD_FLOAT32_C(   222.88), EASYSIMD_FLOAT32_C(  -753.79), EASYSIMD_FLOAT32_C(   614.42), EASYSIMD_FLOAT32_C(   650.99) },
      { EASYSIMD_FLOAT32_C(   509.77), EASYSIMD_FLOAT32_C(    36.72), EASYSIMD_FLOAT32_C(  -901.11), EASYSIMD_FLOAT32_C(   758.14) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   614.42), EASYSIMD_FLOAT32_C(   758.14) } },
    { UINT8_C(114),
      { EASYSIMD_FLOAT32_C(  -682.58), EASYSIMD_FLOAT32_C(    86.85), EASYSIMD_FLOAT32_C(   501.92), EASYSIMD_FLOAT32_C(  -388.44) },
      { EASYSIMD_FLOAT32_C(  -819.84), EASYSIMD_FLOAT32_C(  -148.48), EASYSIMD_FLOAT32_C(  -157.20), EASYSIMD_FLOAT32_C(  -218.01) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    86.85), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 79),
      { EASYSIMD_FLOAT32_C(  -928.01), EASYSIMD_FLOAT32_C(  -290.18), EASYSIMD_FLOAT32_C(   756.78), EASYSIMD_FLOAT32_C(   347.01) },
      { EASYSIMD_FLOAT32_C(   442.87), EASYSIMD_FLOAT32_C(  -941.12), EASYSIMD_FLOAT32_C(  -248.30), EASYSIMD_FLOAT32_C(   868.91) },
      { EASYSIMD_FLOAT32_C(   442.87), EASYSIMD_FLOAT32_C(  -290.18), EASYSIMD_FLOAT32_C(   756.78), EASYSIMD_FLOAT32_C(   868.91) } },
    { UINT8_C( 10),
      { EASYSIMD_FLOAT32_C(   189.45), EASYSIMD_FLOAT32_C(  -918.66), EASYSIMD_FLOAT32_C(   976.91), EASYSIMD_FLOAT32_C(   763.49) },
      { EASYSIMD_FLOAT32_C(   304.21), EASYSIMD_FLOAT32_C(  -776.88), EASYSIMD_FLOAT32_C(   377.91), EASYSIMD_FLOAT32_C(   -44.80) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -776.88), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   763.49) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_max_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_max_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -894.24), EASYSIMD_FLOAT64_C(   640.43) },
         UINT8_MAX,
      { EASYSIMD_FLOAT64_C(   -71.54), EASYSIMD_FLOAT64_C(    94.51) },
      { EASYSIMD_FLOAT64_C(  -892.19), EASYSIMD_FLOAT64_C(  -615.06) },
      { EASYSIMD_FLOAT64_C(   -71.54), EASYSIMD_FLOAT64_C(    94.51) } },
    { { EASYSIMD_FLOAT64_C(  -608.29), EASYSIMD_FLOAT64_C(  -997.63) },
      UINT8_C(245),
      { EASYSIMD_FLOAT64_C(    79.71), EASYSIMD_FLOAT64_C(   968.27) },
      { EASYSIMD_FLOAT64_C(   592.60), EASYSIMD_FLOAT64_C(  -184.56) },
      { EASYSIMD_FLOAT64_C(   592.60), EASYSIMD_FLOAT64_C(  -997.63) } },
    { { EASYSIMD_FLOAT64_C(   421.11), EASYSIMD_FLOAT64_C(  -214.84) },
      UINT8_C(125),
      { EASYSIMD_FLOAT64_C(  -994.24), EASYSIMD_FLOAT64_C(   367.19) },
      { EASYSIMD_FLOAT64_C(  -606.24), EASYSIMD_FLOAT64_C(   181.97) },
      { EASYSIMD_FLOAT64_C(  -606.24), EASYSIMD_FLOAT64_C(  -214.84) } },
    { { EASYSIMD_FLOAT64_C(  -380.52), EASYSIMD_FLOAT64_C(   -47.26) },
      UINT8_C(190),
      { EASYSIMD_FLOAT64_C(  -310.50), EASYSIMD_FLOAT64_C(   -89.67) },
      { EASYSIMD_FLOAT64_C(   888.80), EASYSIMD_FLOAT64_C(  -288.90) },
      { EASYSIMD_FLOAT64_C(  -380.52), EASYSIMD_FLOAT64_C(   -89.67) } },
    { { EASYSIMD_FLOAT64_C(   339.26), EASYSIMD_FLOAT64_C(   768.02) },
      UINT8_C(230),
      { EASYSIMD_FLOAT64_C(   445.02), EASYSIMD_FLOAT64_C(   408.45) },
      { EASYSIMD_FLOAT64_C(   975.22), EASYSIMD_FLOAT64_C(  -626.52) },
      { EASYSIMD_FLOAT64_C(   339.26), EASYSIMD_FLOAT64_C(   408.45) } },
    { { EASYSIMD_FLOAT64_C(  -497.03), EASYSIMD_FLOAT64_C(  -916.97) },
      UINT8_C(112),
      { EASYSIMD_FLOAT64_C(  -105.32), EASYSIMD_FLOAT64_C(  -914.60) },
      { EASYSIMD_FLOAT64_C(  -230.99), EASYSIMD_FLOAT64_C(   974.39) },
      { EASYSIMD_FLOAT64_C(  -497.03), EASYSIMD_FLOAT64_C(  -916.97) } },
    { { EASYSIMD_FLOAT64_C(  -946.33), EASYSIMD_FLOAT64_C(  -638.39) },
      UINT8_C( 22),
      { EASYSIMD_FLOAT64_C(   474.78), EASYSIMD_FLOAT64_C(   146.76) },
      { EASYSIMD_FLOAT64_C(  -328.03), EASYSIMD_FLOAT64_C(   480.54) },
      { EASYSIMD_FLOAT64_C(  -946.33), EASYSIMD_FLOAT64_C(   480.54) } },
    { { EASYSIMD_FLOAT64_C(  -486.05), EASYSIMD_FLOAT64_C(    65.72) },
      UINT8_C( 75),
      { EASYSIMD_FLOAT64_C(   133.43), EASYSIMD_FLOAT64_C(  -981.54) },
      { EASYSIMD_FLOAT64_C(    52.48), EASYSIMD_FLOAT64_C(   822.93) },
      { EASYSIMD_FLOAT64_C(   133.43), EASYSIMD_FLOAT64_C(   822.93) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_max_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_max_pd");
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
    easysimd__m128d r = easysimd_mm_mask_max_pd(src, k, a, b);

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
test_easysimd_mm_maskz_max_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(167),
      { EASYSIMD_FLOAT64_C(  -813.95), EASYSIMD_FLOAT64_C(  -259.60) },
      { EASYSIMD_FLOAT64_C(  -719.12), EASYSIMD_FLOAT64_C(   960.87) },
      { EASYSIMD_FLOAT64_C(  -719.12), EASYSIMD_FLOAT64_C(   960.87) } },
    { UINT8_C(  6),
      { EASYSIMD_FLOAT64_C(  -731.99), EASYSIMD_FLOAT64_C(   166.17) },
      { EASYSIMD_FLOAT64_C(   704.70), EASYSIMD_FLOAT64_C(  -631.18) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   166.17) } },
    { UINT8_C(165),
      { EASYSIMD_FLOAT64_C(   897.87), EASYSIMD_FLOAT64_C(  -162.50) },
      { EASYSIMD_FLOAT64_C(   824.24), EASYSIMD_FLOAT64_C(   154.61) },
      { EASYSIMD_FLOAT64_C(   897.87), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(119),
      { EASYSIMD_FLOAT64_C(  -537.88), EASYSIMD_FLOAT64_C(   -88.06) },
      { EASYSIMD_FLOAT64_C(  -331.90), EASYSIMD_FLOAT64_C(   -48.29) },
      { EASYSIMD_FLOAT64_C(  -331.90), EASYSIMD_FLOAT64_C(   -48.29) } },
    { UINT8_C( 34),
      { EASYSIMD_FLOAT64_C(  -734.60), EASYSIMD_FLOAT64_C(   836.82) },
      { EASYSIMD_FLOAT64_C(  -522.32), EASYSIMD_FLOAT64_C(   239.16) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   836.82) } },
    { UINT8_C(106),
      { EASYSIMD_FLOAT64_C(   613.71), EASYSIMD_FLOAT64_C(   915.49) },
      { EASYSIMD_FLOAT64_C(  -361.46), EASYSIMD_FLOAT64_C(   526.30) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   915.49) } },
    { UINT8_C(122),
      { EASYSIMD_FLOAT64_C(   456.40), EASYSIMD_FLOAT64_C(   712.35) },
      { EASYSIMD_FLOAT64_C(  -603.23), EASYSIMD_FLOAT64_C(   737.27) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   737.27) } },
    { UINT8_C(208),
      { EASYSIMD_FLOAT64_C(  -840.43), EASYSIMD_FLOAT64_C(  -994.72) },
      { EASYSIMD_FLOAT64_C(  -160.61), EASYSIMD_FLOAT64_C(   864.27) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_max_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_max_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_max_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t src[32];
    const uint32_t k;
    const int8_t a[32];
    const int8_t b[32];
    const int8_t r[32];
  } test_vec[] = {
    { {  INT8_C( 115), -INT8_C( 109), -INT8_C(  22),  INT8_C(   6),  INT8_C(  37),  INT8_C(  15), -INT8_C(  70), -INT8_C(  54),
         INT8_C(   9),  INT8_C(  30),  INT8_C(  35), -INT8_C(  44),  INT8_C(  82), -INT8_C( 100), -INT8_C(   4),  INT8_C(  48),
         INT8_C(  81),  INT8_C(  52),  INT8_C(  46), -INT8_C(  89), -INT8_C(  60),  INT8_C(  37), -INT8_C( 105),  INT8_C(  37),
             INT8_MIN,  INT8_C(  13), -INT8_C(  15),  INT8_C(  47), -INT8_C(  57), -INT8_C(  57),  INT8_C(  82),  INT8_C(  58) },
      UINT32_C(2134916442),
      {  INT8_C(  76), -INT8_C(   5),  INT8_C(  73),  INT8_C(  85),  INT8_C(  25),  INT8_C( 109),  INT8_C(  42),  INT8_C( 108),
         INT8_C(   9),  INT8_C(  38), -INT8_C( 100),  INT8_C(  91),  INT8_C(  90), -INT8_C(  54),  INT8_C(   2),  INT8_C(  30),
        -INT8_C(  16), -INT8_C( 103),  INT8_C(  67),  INT8_C( 112), -INT8_C(  89),  INT8_C(  52), -INT8_C(  97),  INT8_C( 110),
        -INT8_C(   5), -INT8_C(  15), -INT8_C(  88),  INT8_C(  86),  INT8_C(  46), -INT8_C(  24), -INT8_C(  43),  INT8_C( 123) },
      { -INT8_C(  29),  INT8_C(  31), -INT8_C(  48), -INT8_C(   3), -INT8_C( 116), -INT8_C(   6),  INT8_C( 105), -INT8_C( 107),
         INT8_C(  32),  INT8_C(   5), -INT8_C(  16),  INT8_C( 122), -INT8_C(  48), -INT8_C(  14), -INT8_C( 104), -INT8_C(  64),
        -INT8_C( 116), -INT8_C(  37),  INT8_C(  48),  INT8_C(  51),  INT8_C(  16), -INT8_C(  49), -INT8_C(  95),  INT8_C(  11),
        -INT8_C(  64),  INT8_C(  73),  INT8_C(  97), -INT8_C(  17),  INT8_C(  49),  INT8_C(  55),  INT8_C( 106),  INT8_C(  21) },
      {  INT8_C( 115),  INT8_C(  31), -INT8_C(  22),  INT8_C(  85),  INT8_C(  25),  INT8_C(  15),  INT8_C( 105), -INT8_C(  54),
         INT8_C(  32),  INT8_C(  30), -INT8_C(  16),  INT8_C( 122),  INT8_C(  90), -INT8_C(  14), -INT8_C(   4),  INT8_C(  48),
         INT8_C(  81),  INT8_C(  52),  INT8_C(  46), -INT8_C(  89), -INT8_C(  60),  INT8_C(  37), -INT8_C(  95),  INT8_C(  37),
        -INT8_C(   5),  INT8_C(  73),  INT8_C(  97),  INT8_C(  86),  INT8_C(  49),  INT8_C(  55),  INT8_C( 106),  INT8_C(  58) } },
    { {  INT8_C(  86),  INT8_C(  58),  INT8_C(  18), -INT8_C(  30),  INT8_C(  53),  INT8_C( 123),  INT8_C( 119),  INT8_C(  85),
             INT8_MIN,  INT8_C( 104), -INT8_C(  48),  INT8_C(  80),  INT8_C(  90),  INT8_C( 104),  INT8_C(  16), -INT8_C(  26),
         INT8_C(  68),  INT8_C(  64),  INT8_C(  25),  INT8_C(  84),  INT8_C(  15), -INT8_C(  70),  INT8_C(  95), -INT8_C(  48),
         INT8_C(   3), -INT8_C(  63), -INT8_C(  65),  INT8_C(  53), -INT8_C(   8),  INT8_C(  41),  INT8_C(  74),  INT8_C(  78) },
      UINT32_C(2553306211),
      { -INT8_C(  41), -INT8_C(  89), -INT8_C(  18),  INT8_C(  87),  INT8_C(  15), -INT8_C(  66), -INT8_C(  88),  INT8_C( 106),
         INT8_C(  38), -INT8_C(  72),  INT8_C(  80),  INT8_C( 106), -INT8_C(   7),  INT8_C( 106), -INT8_C(  66),  INT8_C(   8),
         INT8_C(  36),  INT8_C(  30), -INT8_C(  40),  INT8_C(  40), -INT8_C(  33), -INT8_C( 105),  INT8_C(  93), -INT8_C(  41),
        -INT8_C(  64), -INT8_C(  89),  INT8_C(  37),  INT8_C(  36),  INT8_C(   3),  INT8_C(  85), -INT8_C(  68), -INT8_C(  38) },
      { -INT8_C(   4), -INT8_C(  86),  INT8_C(  49),  INT8_C(  12),  INT8_C( 104), -INT8_C(  39),  INT8_C( 118), -INT8_C( 113),
        -INT8_C( 110), -INT8_C(  58), -INT8_C(   7), -INT8_C( 117),  INT8_C(  48), -INT8_C(  72), -INT8_C( 109),  INT8_C(  85),
        -INT8_C(  42),  INT8_C( 108),  INT8_C( 125), -INT8_C(  75),  INT8_C(   3), -INT8_C(  38), -INT8_C( 116), -INT8_C(  60),
        -INT8_C( 127), -INT8_C(  79), -INT8_C(  24), -INT8_C( 124),  INT8_C(   6), -INT8_C(  92),  INT8_C(  94),  INT8_C(   2) },
      { -INT8_C(   4), -INT8_C(  86),  INT8_C(  18), -INT8_C(  30),  INT8_C(  53), -INT8_C(  39),  INT8_C( 118),  INT8_C(  85),
             INT8_MIN,  INT8_C( 104),  INT8_C(  80),  INT8_C( 106),  INT8_C(  48),  INT8_C( 104), -INT8_C(  66), -INT8_C(  26),
         INT8_C(  68),  INT8_C(  64),  INT8_C(  25),  INT8_C(  84),  INT8_C(   3), -INT8_C(  38),  INT8_C(  95), -INT8_C(  48),
         INT8_C(   3), -INT8_C(  63), -INT8_C(  65),  INT8_C(  36),  INT8_C(   6),  INT8_C(  41),  INT8_C(  74),  INT8_C(   2) } },
    { {  INT8_C(  79), -INT8_C( 113),  INT8_C(  14), -INT8_C(  73),  INT8_C( 105), -INT8_C( 124),  INT8_C(  70), -INT8_C(   5),
         INT8_C(  75),  INT8_C(  64), -INT8_C( 122),  INT8_C( 123), -INT8_C(   8),  INT8_C(  25), -INT8_C(  48), -INT8_C(  50),
        -INT8_C( 123),  INT8_C(  77), -INT8_C( 125), -INT8_C( 119),  INT8_C(  39),  INT8_C(  15),  INT8_C(  77), -INT8_C(  88),
        -INT8_C(  64),  INT8_C(  53),  INT8_C(  44), -INT8_C(  58), -INT8_C(  39), -INT8_C( 118), -INT8_C(  56),  INT8_C(  40) },
      UINT32_C(2212550426),
      {  INT8_C(  91),  INT8_C(  38),  INT8_C( 126), -INT8_C(  90),  INT8_C( 102),  INT8_C(   4),  INT8_C(  34),  INT8_C(  94),
         INT8_C(  29), -INT8_C(  14),  INT8_C(  44), -INT8_C(  93),  INT8_C(  64), -INT8_C(  81),  INT8_C(  44),  INT8_C( 103),
        -INT8_C(  66),  INT8_C( 121),  INT8_C(  16),  INT8_C( 126), -INT8_C(  82),  INT8_C(  60),  INT8_C(  68), -INT8_C( 121),
        -INT8_C(  57),  INT8_C(  13), -INT8_C(  80), -INT8_C(  31), -INT8_C(  28), -INT8_C( 112),  INT8_C( 100),  INT8_C(  63) },
      { -INT8_C(  74), -INT8_C(  30), -INT8_C(  26),  INT8_C(  29), -INT8_C(  26),  INT8_C(   8),  INT8_C( 123),  INT8_C(   3),
        -INT8_C(   6), -INT8_C(  88), -INT8_C(  90),  INT8_C(  58),  INT8_C(  87), -INT8_C(  46), -INT8_C(  94),  INT8_C(  22),
         INT8_C(  75), -INT8_C(  78), -INT8_C( 108), -INT8_C(   7), -INT8_C(  18), -INT8_C(  39), -INT8_C( 127), -INT8_C(  75),
        -INT8_C(  26),  INT8_C(  49), -INT8_C( 106), -INT8_C(  54), -INT8_C(  63), -INT8_C(   6),  INT8_C(   9),  INT8_C( 119) },
      {  INT8_C(  79),  INT8_C(  38),  INT8_C(  14),  INT8_C(  29),  INT8_C( 102), -INT8_C( 124),  INT8_C(  70), -INT8_C(   5),
         INT8_C(  29), -INT8_C(  14),  INT8_C(  44),  INT8_C( 123),  INT8_C(  87),  INT8_C(  25),  INT8_C(  44),  INT8_C( 103),
        -INT8_C( 123),  INT8_C(  77), -INT8_C( 125), -INT8_C( 119),  INT8_C(  39),  INT8_C(  60),  INT8_C(  68), -INT8_C(  75),
        -INT8_C(  26),  INT8_C(  49),  INT8_C(  44), -INT8_C(  58), -INT8_C(  39), -INT8_C( 118), -INT8_C(  56),  INT8_C( 119) } },
    { { -INT8_C(  36), -INT8_C(  17), -INT8_C( 108), -INT8_C(  62), -INT8_C(   9),  INT8_C(  16), -INT8_C(  58), -INT8_C(  14),
        -INT8_C(  72),  INT8_C( 108),  INT8_C(  44),  INT8_C(  15),  INT8_C(  63), -INT8_C(  50),  INT8_C(  37), -INT8_C( 118),
             INT8_MIN, -INT8_C(  70), -INT8_C( 124),  INT8_C( 111), -INT8_C( 109),  INT8_C(   5),  INT8_C(  36),  INT8_C( 121),
         INT8_C(  54), -INT8_C(  69),  INT8_C(  67), -INT8_C(   9), -INT8_C(  75),  INT8_C(  76),  INT8_C( 110), -INT8_C( 110) },
      UINT32_C( 861143868),
      {  INT8_C(  19),  INT8_C(  26),  INT8_C(  37), -INT8_C(  53), -INT8_C( 121),  INT8_C(  82), -INT8_C(  38), -INT8_C(  58),
         INT8_C(  32),  INT8_C(   0),  INT8_C(  80), -INT8_C(  95), -INT8_C(  70), -INT8_C(  44),  INT8_C(  16),  INT8_C(  77),
        -INT8_C(  39),  INT8_C(  52), -INT8_C(  58),  INT8_C(  15), -INT8_C(  17),  INT8_C(   9),  INT8_C(   6), -INT8_C(  91),
         INT8_C(  85),  INT8_C( 117),  INT8_C(  55), -INT8_C( 111),  INT8_C( 120), -INT8_C( 117), -INT8_C(  59), -INT8_C( 117) },
      { -INT8_C(  90), -INT8_C(  22),  INT8_C(  86),  INT8_C(  45),  INT8_C(  60),  INT8_C(  48), -INT8_C(  13),  INT8_C(  93),
         INT8_C(  48),  INT8_C(  67), -INT8_C(   2), -INT8_C(  22),  INT8_C(  24),  INT8_C(  14),  INT8_C(  55), -INT8_C(  15),
         INT8_C(  66), -INT8_C(   3),  INT8_C(   1),  INT8_C(  50),  INT8_C(   6),  INT8_C(   7), -INT8_C(  41),  INT8_C(  92),
         INT8_C( 124),  INT8_C(  14), -INT8_C(  19), -INT8_C(  12), -INT8_C( 103), -INT8_C(  78),      INT8_MAX,  INT8_C(  63) },
      { -INT8_C(  36), -INT8_C(  17),  INT8_C(  86),  INT8_C(  45),  INT8_C(  60),  INT8_C(  82), -INT8_C(  58), -INT8_C(  14),
         INT8_C(  48),  INT8_C(  67),  INT8_C(  44),  INT8_C(  15),  INT8_C(  63), -INT8_C(  50),  INT8_C(  37), -INT8_C( 118),
             INT8_MIN, -INT8_C(  70),  INT8_C(   1),  INT8_C( 111),  INT8_C(   6),  INT8_C(   5),  INT8_C(   6),  INT8_C( 121),
         INT8_C( 124),  INT8_C( 117),  INT8_C(  67), -INT8_C(   9),  INT8_C( 120), -INT8_C(  78),  INT8_C( 110), -INT8_C( 110) } },
    { { -INT8_C(  99), -INT8_C(  43),  INT8_C( 108), -INT8_C(  39),  INT8_C(   6),  INT8_C(  95),  INT8_C(  54),  INT8_C(  54),
        -INT8_C(  93),  INT8_C(  52),  INT8_C(  33), -INT8_C(  69),  INT8_C(  66),  INT8_C(  88), -INT8_C(  84), -INT8_C( 123),
         INT8_C(  86), -INT8_C(  83), -INT8_C(  73),  INT8_C(  92), -INT8_C(  75), -INT8_C( 114), -INT8_C(  72),  INT8_C(  49),
        -INT8_C( 100), -INT8_C(  90),  INT8_C(  38),  INT8_C(  53),  INT8_C(  88), -INT8_C(  91),  INT8_C( 117), -INT8_C(  11) },
      UINT32_C(2177884539),
      {  INT8_C(  65),  INT8_C(   5), -INT8_C(  73), -INT8_C(  28),  INT8_C(  58), -INT8_C(  40), -INT8_C(  97),  INT8_C( 124),
         INT8_C(  49),  INT8_C(  75),  INT8_C(   1), -INT8_C( 121), -INT8_C(   7), -INT8_C(  72), -INT8_C(  29), -INT8_C(  82),
         INT8_C(  70), -INT8_C( 100), -INT8_C(  33), -INT8_C(  30),  INT8_C(  66),  INT8_C(   5),  INT8_C(  24), -INT8_C( 102),
        -INT8_C(  85), -INT8_C( 115), -INT8_C( 112),  INT8_C(  38),  INT8_C( 110),  INT8_C(  95), -INT8_C(  89), -INT8_C(  81) },
      {  INT8_C( 100),  INT8_C(  94), -INT8_C( 109), -INT8_C(  98),  INT8_C(  55),  INT8_C(  50),  INT8_C(  27),  INT8_C( 104),
         INT8_C( 126),  INT8_C(  28), -INT8_C(  17),  INT8_C( 119), -INT8_C(  43), -INT8_C(  46),  INT8_C(  37),  INT8_C(  27),
         INT8_C( 110),  INT8_C(   4), -INT8_C(   2), -INT8_C(  80),  INT8_C(  10),  INT8_C(  22),  INT8_C(  75), -INT8_C(  75),
        -INT8_C(  93), -INT8_C(  37), -INT8_C(  37),  INT8_C(  17),  INT8_C(  58), -INT8_C( 126), -INT8_C(  63), -INT8_C(  98) },
      {  INT8_C( 100),  INT8_C(  94),  INT8_C( 108), -INT8_C(  28),  INT8_C(  58),  INT8_C(  50),  INT8_C(  27),  INT8_C(  54),
         INT8_C( 126),  INT8_C(  52),  INT8_C(  33), -INT8_C(  69),  INT8_C(  66), -INT8_C(  46),  INT8_C(  37),  INT8_C(  27),
         INT8_C( 110),  INT8_C(   4), -INT8_C(   2), -INT8_C(  30), -INT8_C(  75), -INT8_C( 114),  INT8_C(  75), -INT8_C(  75),
        -INT8_C(  85), -INT8_C(  90),  INT8_C(  38),  INT8_C(  53),  INT8_C(  88), -INT8_C(  91),  INT8_C( 117), -INT8_C(  81) } },
    { { -INT8_C(  32),  INT8_C(  84),  INT8_C(  61),  INT8_C(  23), -INT8_C( 121),  INT8_C(  88),      INT8_MAX,  INT8_C(   5),
         INT8_C( 116),  INT8_C( 110),  INT8_C( 124),  INT8_C(  73),  INT8_C(  65), -INT8_C(  95),  INT8_C( 101), -INT8_C(  81),
        -INT8_C(  91),  INT8_C(  99),  INT8_C(  96), -INT8_C(  81),  INT8_C( 121), -INT8_C(  85),  INT8_C( 100),  INT8_C(  28),
        -INT8_C( 122),  INT8_C(  63),  INT8_C(  45), -INT8_C(  64), -INT8_C(  63), -INT8_C(  18),  INT8_C(  94), -INT8_C(  94) },
      UINT32_C(3401161539),
      { -INT8_C(  13),  INT8_C(  57), -INT8_C(  49),  INT8_C( 104), -INT8_C(  89),  INT8_C(  75), -INT8_C(  79), -INT8_C(  24),
        -INT8_C(  20),  INT8_C(  22), -INT8_C( 104), -INT8_C( 111),  INT8_C( 121), -INT8_C(   8),  INT8_C(  65), -INT8_C(  14),
        -INT8_C(  93), -INT8_C(  91),  INT8_C(  14),  INT8_C(  41), -INT8_C(  27),  INT8_C(  60), -INT8_C(  23), -INT8_C(  90),
         INT8_C(  42),  INT8_C(  71),  INT8_C(  72),  INT8_C( 109), -INT8_C(  29),  INT8_C(   2),  INT8_C(  55), -INT8_C(  42) },
      {  INT8_C(  59),  INT8_C(   6),  INT8_C(  62), -INT8_C(  30),  INT8_C(  81), -INT8_C(  16), -INT8_C(  53),  INT8_C(  61),
         INT8_C(   6),  INT8_C(  99), -INT8_C(  49),      INT8_MIN,  INT8_C(  91),  INT8_C(  16),  INT8_C( 114), -INT8_C(   2),
        -INT8_C(  75), -INT8_C( 127),  INT8_C(  39), -INT8_C( 102), -INT8_C(  67),  INT8_C(  16),  INT8_C(  65), -INT8_C(  25),
         INT8_C(  87), -INT8_C( 119),  INT8_C(  85),  INT8_C(  58), -INT8_C( 117), -INT8_C( 116),  INT8_C(  17), -INT8_C(  58) },
      {  INT8_C(  59),  INT8_C(  57),  INT8_C(  61),  INT8_C(  23), -INT8_C( 121),  INT8_C(  88), -INT8_C(  53),  INT8_C(   5),
         INT8_C(   6),  INT8_C(  99),  INT8_C( 124), -INT8_C( 111),  INT8_C( 121), -INT8_C(  95),  INT8_C( 101), -INT8_C(   2),
        -INT8_C(  75),  INT8_C(  99),  INT8_C(  96),  INT8_C(  41), -INT8_C(  27),  INT8_C(  60),  INT8_C( 100), -INT8_C(  25),
        -INT8_C( 122),  INT8_C(  71),  INT8_C(  45),  INT8_C( 109), -INT8_C(  63), -INT8_C(  18),  INT8_C(  55), -INT8_C(  42) } },
    { { -INT8_C( 109),  INT8_C(  79), -INT8_C(  87), -INT8_C(  28),  INT8_C(  63),  INT8_C( 116),  INT8_C(  34),  INT8_C(  70),
        -INT8_C(  41), -INT8_C(  15), -INT8_C(  58),  INT8_C(  50),  INT8_C(   1),  INT8_C(  56),  INT8_C(  48), -INT8_C(  74),
        -INT8_C(  71),  INT8_C(  87),  INT8_C(  81),  INT8_C( 118),  INT8_C( 103), -INT8_C( 110),  INT8_C(  94), -INT8_C(  66),
         INT8_C(  27), -INT8_C(  77), -INT8_C(   7), -INT8_C(  89),  INT8_C(  63),  INT8_C(  10),  INT8_C( 109), -INT8_C(  46) },
      UINT32_C(2578912857),
      { -INT8_C( 118), -INT8_C(  39), -INT8_C(  33),  INT8_C(  97), -INT8_C(  54), -INT8_C(  91), -INT8_C( 109), -INT8_C(  53),
        -INT8_C(  35), -INT8_C(  61), -INT8_C( 127), -INT8_C( 105),  INT8_C(  26), -INT8_C(  46),  INT8_C(  13), -INT8_C( 127),
         INT8_C( 100),  INT8_C( 107),  INT8_C(  64),      INT8_MIN,  INT8_C(  30),  INT8_C(  57),  INT8_C(  39),  INT8_C(  94),
         INT8_C(  67), -INT8_C( 108),  INT8_C(  48), -INT8_C( 100), -INT8_C(  85), -INT8_C(  25),  INT8_C(  53),  INT8_C(  53) },
      { -INT8_C(  64),  INT8_C(  20), -INT8_C( 105), -INT8_C( 118), -INT8_C(  71),  INT8_C(  42),  INT8_C(  85), -INT8_C( 105),
        -INT8_C(  18), -INT8_C(  41),  INT8_C(  46),  INT8_C(   8), -INT8_C(  87),  INT8_C(  59), -INT8_C( 118),  INT8_C(  14),
        -INT8_C(  89), -INT8_C(  54), -INT8_C( 114), -INT8_C(  59),  INT8_C(   3), -INT8_C(  75),  INT8_C(  35),  INT8_C(  70),
         INT8_C(  73),  INT8_C(  84), -INT8_C(  30), -INT8_C(  12),  INT8_C(  59),  INT8_C(  24),  INT8_C(  42), -INT8_C(   4) },
      { -INT8_C(  64),  INT8_C(  79), -INT8_C(  87),  INT8_C(  97), -INT8_C(  54),  INT8_C( 116),  INT8_C(  85),  INT8_C(  70),
        -INT8_C(  41), -INT8_C(  41),  INT8_C(  46),  INT8_C(  50),  INT8_C(  26),  INT8_C(  56),  INT8_C(  48), -INT8_C(  74),
         INT8_C( 100),  INT8_C( 107),  INT8_C(  64),  INT8_C( 118),  INT8_C(  30),  INT8_C(  57),  INT8_C(  94),  INT8_C(  94),
         INT8_C(  73), -INT8_C(  77), -INT8_C(   7), -INT8_C(  12),  INT8_C(  59),  INT8_C(  10),  INT8_C( 109),  INT8_C(  53) } },
    { {  INT8_C(  44), -INT8_C(  63), -INT8_C( 122), -INT8_C(  26), -INT8_C(  21), -INT8_C(  36),  INT8_C( 125), -INT8_C(  39),
        -INT8_C(  77), -INT8_C(  85), -INT8_C(  30),  INT8_C(  92), -INT8_C(  26),  INT8_C( 108),  INT8_C( 106), -INT8_C( 115),
         INT8_C(  54), -INT8_C(   8),  INT8_C(  83),  INT8_C(  57), -INT8_C(  83),  INT8_C( 118),      INT8_MAX, -INT8_C(   9),
        -INT8_C(  54),  INT8_C(  97), -INT8_C(  21),  INT8_C(   6),  INT8_C( 121),  INT8_C(  21),  INT8_C(   2), -INT8_C(  90) },
      UINT32_C(3263989974),
      {  INT8_C( 100),  INT8_C(   9), -INT8_C( 101),  INT8_C(  23), -INT8_C(  76),  INT8_C( 125),  INT8_C( 116), -INT8_C( 102),
        -INT8_C(  23), -INT8_C(  34),  INT8_C(  40),  INT8_C(  31), -INT8_C(  41),  INT8_C( 123),  INT8_C(  88), -INT8_C( 124),
        -INT8_C(  15), -INT8_C(  41),  INT8_C( 123), -INT8_C(  68),  INT8_C(  57),  INT8_C( 103), -INT8_C(  62), -INT8_C(  78),
         INT8_C( 124), -INT8_C(  60),  INT8_C(  88),  INT8_C(  83),  INT8_C(  76), -INT8_C(  28),  INT8_C(  21), -INT8_C(  79) },
      { -INT8_C(  19), -INT8_C(  80), -INT8_C(  56), -INT8_C(  95),  INT8_C(  46),  INT8_C(  60),  INT8_C(  60),  INT8_C(  23),
         INT8_C(  27),  INT8_C( 100),  INT8_C(  55), -INT8_C(  14), -INT8_C(  33), -INT8_C( 113),  INT8_C( 118), -INT8_C(  48),
         INT8_C( 103), -INT8_C(  14), -INT8_C( 116), -INT8_C(  96),  INT8_C(  89),  INT8_C(  78),  INT8_C(  82), -INT8_C(  43),
         INT8_C(  18), -INT8_C(  85),  INT8_C(  40),  INT8_C(  95), -INT8_C( 113),  INT8_C(  61),  INT8_C(  16),  INT8_C( 125) },
      {  INT8_C(  44),  INT8_C(   9), -INT8_C(  56), -INT8_C(  26),  INT8_C(  46), -INT8_C(  36),  INT8_C( 116),  INT8_C(  23),
        -INT8_C(  77), -INT8_C(  85), -INT8_C(  30),  INT8_C(  31), -INT8_C(  26),  INT8_C( 108),  INT8_C( 106), -INT8_C(  48),
         INT8_C(  54), -INT8_C(   8),  INT8_C( 123), -INT8_C(  68), -INT8_C(  83),  INT8_C( 118),      INT8_MAX, -INT8_C(  43),
        -INT8_C(  54), -INT8_C(  60), -INT8_C(  21),  INT8_C(   6),  INT8_C( 121),  INT8_C(  21),  INT8_C(  21),  INT8_C( 125) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi8(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_epi8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_epi8");
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
    easysimd__m256i r = easysimd_mm256_mask_max_epi8(src, k, a, b);

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
test_easysimd_mm256_maskz_max_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t k;
    const int8_t a[32];
    const int8_t b[32];
    const int8_t r[32];
  } test_vec[] = {
    { UINT32_C(2086591482),
      {  INT8_C(  69),  INT8_C(  47),  INT8_C(  12), -INT8_C(   5), -INT8_C(  81),  INT8_C(  30),  INT8_C( 114), -INT8_C(  92),
        -INT8_C(  60),  INT8_C(  48), -INT8_C( 109),  INT8_C(  54), -INT8_C(  60),  INT8_C(  92), -INT8_C(  18), -INT8_C(   3),
        -INT8_C(  83),  INT8_C(  75), -INT8_C(  26), -INT8_C(  50), -INT8_C(  61), -INT8_C(  48), -INT8_C(  82),  INT8_C(  20),
        -INT8_C(  53),  INT8_C(  23),  INT8_C(  53), -INT8_C(  58), -INT8_C(  13), -INT8_C( 109),  INT8_C(  66),  INT8_C(  56) },
      { -INT8_C(  61),  INT8_C(  78),  INT8_C(  51),  INT8_C( 114),  INT8_C( 108), -INT8_C(  90),  INT8_C(  22),  INT8_C(  48),
        -INT8_C(  42), -INT8_C(  87),  INT8_C( 103), -INT8_C( 102),  INT8_C(   5),  INT8_C(  85), -INT8_C( 105), -INT8_C(  77),
        -INT8_C(  96),  INT8_C( 126), -INT8_C( 127),  INT8_C(  99),  INT8_C(  78),  INT8_C(  47),  INT8_C( 119),  INT8_C(  26),
         INT8_C(  71), -INT8_C(  84), -INT8_C(  32),  INT8_C(  58),  INT8_C(  64),  INT8_C(  34),  INT8_C( 114),  INT8_C(   3) },
      {  INT8_C(   0),  INT8_C(  78),  INT8_C(   0),  INT8_C( 114),  INT8_C( 108),  INT8_C(  30),  INT8_C( 114),  INT8_C(  48),
        -INT8_C(  42),  INT8_C(  48),  INT8_C(   0),  INT8_C(  54),  INT8_C(   5),  INT8_C(   0), -INT8_C(  18), -INT8_C(   3),
         INT8_C(   0),  INT8_C( 126), -INT8_C(  26),  INT8_C(  99),  INT8_C(  78),  INT8_C(   0),  INT8_C( 119),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  53),  INT8_C(  58),  INT8_C(  64),  INT8_C(  34),  INT8_C( 114),  INT8_C(   0) } },
    { UINT32_C(3698697840),
      {  INT8_C(  76), -INT8_C( 116),  INT8_C(  13),  INT8_C(  34),  INT8_C(  53),  INT8_C( 116), -INT8_C(  68),  INT8_C(  59),
        -INT8_C(  55),  INT8_C(  84), -INT8_C(  18),  INT8_C( 105), -INT8_C(  46),  INT8_C( 111), -INT8_C(  52),  INT8_C(  32),
        -INT8_C(  97),  INT8_C(  67),  INT8_C(  58), -INT8_C(  26), -INT8_C(  17),  INT8_C(  26),  INT8_C(  32),  INT8_C(  47),
         INT8_C(  60), -INT8_C( 110),  INT8_C(  50), -INT8_C(  84),  INT8_C(  56), -INT8_C(  88), -INT8_C( 119), -INT8_C( 124) },
      {  INT8_C(  52), -INT8_C( 106), -INT8_C(  90),  INT8_C( 105),  INT8_C(  10),  INT8_C(  99), -INT8_C(  92), -INT8_C(  45),
        -INT8_C(  73), -INT8_C( 110),  INT8_C(  60), -INT8_C( 119),  INT8_C(   2),  INT8_C(   8), -INT8_C(  87), -INT8_C(  95),
         INT8_C(  75), -INT8_C(  28), -INT8_C( 121),  INT8_C(  58), -INT8_C(   2), -INT8_C(  89),  INT8_C( 106),  INT8_C(  59),
         INT8_C(  57), -INT8_C( 100), -INT8_C(  25),  INT8_C( 114),  INT8_C(  68),  INT8_C( 112), -INT8_C(  10),  INT8_C( 120) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  53),  INT8_C( 116), -INT8_C(  68),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  84),  INT8_C(  60),  INT8_C(   0),  INT8_C(   0),  INT8_C( 111),  INT8_C(   0),  INT8_C(  32),
         INT8_C(  75),  INT8_C(   0),  INT8_C(  58),  INT8_C(   0), -INT8_C(   2),  INT8_C(  26),  INT8_C( 106),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  50),  INT8_C( 114),  INT8_C(  68),  INT8_C(   0), -INT8_C(  10),  INT8_C( 120) } },
    { UINT32_C( 283286790),
      {  INT8_C(   0), -INT8_C( 122), -INT8_C(  29), -INT8_C(  73),  INT8_C(  25),  INT8_C(  31),  INT8_C(  64),  INT8_C(  27),
         INT8_C(  39), -INT8_C(  23), -INT8_C(  68),  INT8_C( 114), -INT8_C(  51),  INT8_C(  67), -INT8_C(  83), -INT8_C(  52),
        -INT8_C(  22),  INT8_C(  23),  INT8_C(   7),  INT8_C(  35), -INT8_C(  77), -INT8_C(  18), -INT8_C( 107), -INT8_C(   8),
         INT8_C(  95), -INT8_C( 116),  INT8_C( 112),  INT8_C( 101),  INT8_C(  41),  INT8_C(  82),  INT8_C( 118),  INT8_C(  41) },
      { -INT8_C(  39),  INT8_C(  89), -INT8_C(  32), -INT8_C(  14),  INT8_C( 121),  INT8_C(  32),  INT8_C(  13), -INT8_C(  96),
         INT8_C(   9), -INT8_C(  55),  INT8_C(  19), -INT8_C(  41),  INT8_C(  12), -INT8_C(  64), -INT8_C(  93), -INT8_C(  10),
        -INT8_C(  41), -INT8_C(  86),  INT8_C(  25), -INT8_C( 118), -INT8_C( 104), -INT8_C(  81), -INT8_C( 126), -INT8_C(   9),
         INT8_C(  59), -INT8_C(  13),  INT8_C(  93),  INT8_C( 100),  INT8_C(  69), -INT8_C(  45), -INT8_C( 115),  INT8_C(  30) },
      {  INT8_C(   0),  INT8_C(  89), -INT8_C(  29),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  39),  INT8_C(   0),  INT8_C(  19),  INT8_C( 114),  INT8_C(  12),  INT8_C(   0),  INT8_C(   0), -INT8_C(  10),
         INT8_C(   0),  INT8_C(  23),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  18), -INT8_C( 107), -INT8_C(   8),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  69),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C(2769317164),
      { -INT8_C( 115),  INT8_C(  29),  INT8_C(  70), -INT8_C( 106), -INT8_C(  26),  INT8_C(  89),  INT8_C( 109), -INT8_C(  14),
         INT8_C(  25),  INT8_C(  16), -INT8_C(  24), -INT8_C(  16), -INT8_C(  70),  INT8_C(   2),  INT8_C( 122),  INT8_C(  83),
        -INT8_C(  79), -INT8_C(   3),  INT8_C(  74), -INT8_C(  20), -INT8_C(  16), -INT8_C(  89),  INT8_C(  80),  INT8_C(  53),
         INT8_C( 122), -INT8_C(  35),  INT8_C(  84), -INT8_C(  89),  INT8_C(  74),  INT8_C( 100),  INT8_C(  76), -INT8_C(  41) },
      { -INT8_C( 126), -INT8_C( 110),  INT8_C( 109),  INT8_C( 104), -INT8_C(  21), -INT8_C(  37),  INT8_C(  91),  INT8_C(   4),
        -INT8_C(  21),  INT8_C(  67), -INT8_C(  12), -INT8_C(  90),  INT8_C(  69),  INT8_C( 111), -INT8_C(   7), -INT8_C(  10),
         INT8_C( 108),  INT8_C(  67), -INT8_C(  30),  INT8_C(  92), -INT8_C(  21),  INT8_C(  50), -INT8_C( 111),  INT8_C( 101),
         INT8_C(  15), -INT8_C(  27),  INT8_C(  12),  INT8_C(  89),  INT8_C(  74),  INT8_C(  89),  INT8_C(  48), -INT8_C(  52) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C( 109),  INT8_C( 104),  INT8_C(   0),  INT8_C(  89),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  25),  INT8_C(   0), -INT8_C(  12), -INT8_C(  16),  INT8_C(   0),  INT8_C( 111),  INT8_C( 122),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  16),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 122),  INT8_C(   0),  INT8_C(  84),  INT8_C(   0),  INT8_C(   0),  INT8_C( 100),  INT8_C(   0), -INT8_C(  41) } },
    { UINT32_C(3610549995),
      {  INT8_C( 121), -INT8_C( 113), -INT8_C(  37),  INT8_C( 100), -INT8_C(  45), -INT8_C(  48),  INT8_C(  10),  INT8_C(  24),
         INT8_C(  63),  INT8_C(   3),  INT8_C(  15), -INT8_C(  85),  INT8_C(  71), -INT8_C(  15),  INT8_C(   7),  INT8_C(  50),
         INT8_C(  36), -INT8_C( 104), -INT8_C( 105),  INT8_C(  51),  INT8_C( 126), -INT8_C(  92), -INT8_C( 115), -INT8_C(  56),
        -INT8_C(   3), -INT8_C(  67), -INT8_C( 108), -INT8_C(  24),  INT8_C(  91), -INT8_C(  56), -INT8_C(  65), -INT8_C(  44) },
      {  INT8_C(  88), -INT8_C( 101),  INT8_C(  57),  INT8_C(  43),  INT8_C( 107),  INT8_C(  67),  INT8_C(  67), -INT8_C(  86),
         INT8_C(  71),  INT8_C(  82),  INT8_C(  85), -INT8_C( 114),  INT8_C(  68),  INT8_C(  92), -INT8_C(  64),  INT8_C( 104),
        -INT8_C(  12),  INT8_C(  87), -INT8_C( 101),  INT8_C( 114), -INT8_C(   5),  INT8_C(  40),  INT8_C(  58), -INT8_C(   8),
        -INT8_C(  26), -INT8_C(  50), -INT8_C(  31),  INT8_C(  65), -INT8_C( 105), -INT8_C(  96),  INT8_C(  22), -INT8_C(  17) },
      {  INT8_C( 121), -INT8_C( 101),  INT8_C(   0),  INT8_C( 100),  INT8_C(   0),  INT8_C(  67),  INT8_C(  67),  INT8_C(  24),
         INT8_C(   0),  INT8_C(  82),  INT8_C(  85), -INT8_C(  85),  INT8_C(  71),  INT8_C(   0),  INT8_C(   0),  INT8_C( 104),
         INT8_C(   0),  INT8_C(   0), -INT8_C( 101),  INT8_C(   0),  INT8_C( 126),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(   3), -INT8_C(  50), -INT8_C(  31),  INT8_C(   0),  INT8_C(  91),  INT8_C(   0),  INT8_C(  22), -INT8_C(  17) } },
    { UINT32_C(2786742075),
      { -INT8_C( 110),  INT8_C(  93),  INT8_C(  80), -INT8_C(  39), -INT8_C(  80), -INT8_C(  91),  INT8_C( 103), -INT8_C(  12),
         INT8_C(   1),  INT8_C(  39),  INT8_C(  92), -INT8_C(  10),      INT8_MAX, -INT8_C(   9),  INT8_C( 104),  INT8_C( 122),
         INT8_C(  32), -INT8_C(  93),  INT8_C( 115),  INT8_C(   6),  INT8_C( 113),  INT8_C(  84),  INT8_C(  71),  INT8_C(   8),
        -INT8_C(  12),  INT8_C(  93), -INT8_C(   9),  INT8_C(  48), -INT8_C(  84),  INT8_C(  17), -INT8_C(  42),  INT8_C(  63) },
      {  INT8_C( 111),  INT8_C(  39),  INT8_C(  24),  INT8_C(  31), -INT8_C(  52),      INT8_MIN,  INT8_C(  19), -INT8_C(  50),
        -INT8_C(  89),  INT8_C( 111), -INT8_C(  60),  INT8_C(  38),  INT8_C( 102),  INT8_C(  44), -INT8_C(  95), -INT8_C( 122),
        -INT8_C(  49),  INT8_C(  20), -INT8_C( 116),  INT8_C(  65),  INT8_C( 104), -INT8_C(  44),  INT8_C(  73),  INT8_C(  92),
         INT8_C(  49),  INT8_C(  65), -INT8_C( 116), -INT8_C(  34),  INT8_C(  82),  INT8_C(  99),  INT8_C(  29), -INT8_C(  63) },
      {  INT8_C( 111),  INT8_C(  93),  INT8_C(   0),  INT8_C(  31), -INT8_C(  52), -INT8_C(  91),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   1),  INT8_C( 111),  INT8_C(  92),  INT8_C(  38),  INT8_C(   0),  INT8_C(   0),  INT8_C( 104),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  20),  INT8_C(   0),  INT8_C(  65),  INT8_C( 113),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  93), -INT8_C(   9),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),  INT8_C(   0),  INT8_C(  63) } },
    { UINT32_C(1457534346),
      { -INT8_C(  75), -INT8_C(  13),  INT8_C(  36),  INT8_C(  93),  INT8_C(  98), -INT8_C(  24), -INT8_C( 125), -INT8_C(  55),
         INT8_C(  21),  INT8_C(  36),  INT8_C(  79), -INT8_C(  28),  INT8_C(  56), -INT8_C(  36),  INT8_C(  37), -INT8_C(  96),
        -INT8_C(  80),  INT8_C( 111), -INT8_C(   3), -INT8_C(  31), -INT8_C(  80), -INT8_C( 119), -INT8_C(  65),  INT8_C(   2),
        -INT8_C(  20), -INT8_C(  36), -INT8_C(  60),  INT8_C( 118),  INT8_C(  18), -INT8_C(  92), -INT8_C(  51), -INT8_C(  57) },
      { -INT8_C( 104), -INT8_C(  15),  INT8_C(  36), -INT8_C(   6), -INT8_C(  38), -INT8_C(  88), -INT8_C(  61), -INT8_C(  17),
        -INT8_C(  52),  INT8_C(  19), -INT8_C(  45),  INT8_C(   5), -INT8_C(  17), -INT8_C(   7), -INT8_C(  91), -INT8_C(  97),
         INT8_C( 104), -INT8_C(  94),      INT8_MIN,  INT8_C(  24),  INT8_C(  44),  INT8_C(  64),  INT8_C(  26),  INT8_C(  24),
         INT8_C(  28), -INT8_C(  34), -INT8_C( 113),  INT8_C(  46), -INT8_C( 125),  INT8_C(  92), -INT8_C(  10),  INT8_C(  27) },
      {  INT8_C(   0), -INT8_C(  13),  INT8_C(   0),  INT8_C(  93),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  17),
         INT8_C(  21),  INT8_C(   0),  INT8_C(  79),  INT8_C(   0),  INT8_C(  56), -INT8_C(   7),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  64),  INT8_C(  26),  INT8_C(  24),
         INT8_C(   0), -INT8_C(  34), -INT8_C(  60),  INT8_C(   0),  INT8_C(  18),  INT8_C(   0), -INT8_C(  10),  INT8_C(   0) } },
    { UINT32_C( 655694413),
      { -INT8_C(  62), -INT8_C(  39),  INT8_C(  22), -INT8_C( 113), -INT8_C(  20), -INT8_C(  22), -INT8_C( 108), -INT8_C(  37),
        -INT8_C(  29),  INT8_C(  57),  INT8_C( 122),  INT8_C(  75), -INT8_C(  36), -INT8_C(   6),  INT8_C(  99),  INT8_C(   8),
         INT8_C(  58),  INT8_C( 125),  INT8_C(  32),  INT8_C(  87),  INT8_C(  92), -INT8_C(  81), -INT8_C( 123), -INT8_C(  33),
         INT8_C(  11),  INT8_C( 123), -INT8_C(   6),  INT8_C(  89), -INT8_C( 106),  INT8_C(  15),      INT8_MIN,  INT8_C(  88) },
      { -INT8_C(  24), -INT8_C( 105), -INT8_C(  25), -INT8_C(  44), -INT8_C( 127),  INT8_C( 123), -INT8_C(  81),  INT8_C( 100),
        -INT8_C(  75),  INT8_C(  41), -INT8_C(  81), -INT8_C( 111),  INT8_C(  36),  INT8_C(  18), -INT8_C( 103),  INT8_C(  94),
        -INT8_C( 113), -INT8_C(  71), -INT8_C(  75), -INT8_C(  21),  INT8_C( 105),  INT8_C(  59), -INT8_C(  54),  INT8_C( 116),
        -INT8_C(  74), -INT8_C(  60), -INT8_C(  51),  INT8_C(  76), -INT8_C(  44),  INT8_C(  78), -INT8_C(  91), -INT8_C(  68) },
      { -INT8_C(  24),  INT8_C(   0),  INT8_C(  22), -INT8_C(  44),  INT8_C(   0),  INT8_C(   0), -INT8_C(  81),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  57),  INT8_C(   0),  INT8_C(  75),  INT8_C(  36),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  58),  INT8_C(   0),  INT8_C(  32),  INT8_C(   0),  INT8_C( 105),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  11),  INT8_C( 123), -INT8_C(   6),  INT8_C(   0),  INT8_C(   0),  INT8_C(  78),  INT8_C(   0),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_epi8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_epi8");
    easysimd_test_x86_assert_equal_i8x32(r, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i b = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = easysimd_mm256_maskz_max_epi8(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[16];
    const uint16_t k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { { -INT16_C( 10002),  INT16_C(  7198),  INT16_C( 23061),  INT16_C( 12339),  INT16_C( 27326), -INT16_C( 25310), -INT16_C( 26374),  INT16_C( 24942),
        -INT16_C(  1398), -INT16_C(  7423),  INT16_C( 21321),  INT16_C( 23481), -INT16_C(  7682), -INT16_C( 28998), -INT16_C( 13793),  INT16_C(  3339) },
      UINT16_C(10659),
      { -INT16_C( 18391),  INT16_C( 23684),  INT16_C( 17128),  INT16_C(  2759), -INT16_C( 15904),  INT16_C( 20130),  INT16_C( 11554),  INT16_C(  9032),
        -INT16_C( 28400), -INT16_C( 13962),  INT16_C( 30189), -INT16_C( 22613), -INT16_C( 13821),  INT16_C(  3698),  INT16_C(  5591),  INT16_C(    55) },
      { -INT16_C( 17459), -INT16_C( 19108),  INT16_C(  9214), -INT16_C(  8513),  INT16_C( 25060),  INT16_C(  1580),  INT16_C( 29838), -INT16_C( 24791),
        -INT16_C( 24570), -INT16_C(  3224),  INT16_C(  4885),  INT16_C(  6298),  INT16_C(  3293), -INT16_C( 19418),  INT16_C( 23841), -INT16_C(  4428) },
      { -INT16_C( 17459),  INT16_C( 23684),  INT16_C( 23061),  INT16_C( 12339),  INT16_C( 27326),  INT16_C( 20130), -INT16_C( 26374),  INT16_C(  9032),
        -INT16_C( 24570), -INT16_C(  7423),  INT16_C( 21321),  INT16_C(  6298), -INT16_C(  7682),  INT16_C(  3698), -INT16_C( 13793),  INT16_C(  3339) } },
    { {  INT16_C(  4377),  INT16_C(  6051),  INT16_C( 25140),  INT16_C(  6645),  INT16_C(  8644),  INT16_C( 21023),  INT16_C( 18837), -INT16_C( 25615),
         INT16_C( 23273), -INT16_C(   370),  INT16_C( 10605),  INT16_C( 19222),  INT16_C( 15413),  INT16_C( 22527), -INT16_C( 19303), -INT16_C( 19899) },
      UINT16_C(59845),
      { -INT16_C(  1591), -INT16_C( 16821),  INT16_C(  3858),  INT16_C( 13023),  INT16_C( 30050),  INT16_C( 21371),  INT16_C( 25616), -INT16_C( 24659),
         INT16_C(  7010),  INT16_C( 30920), -INT16_C(   666),  INT16_C( 26036),  INT16_C( 19796), -INT16_C( 26087), -INT16_C(  8704), -INT16_C( 13949) },
      { -INT16_C( 12584), -INT16_C(  5496),  INT16_C( 26590),  INT16_C( 16412), -INT16_C( 26660), -INT16_C(  4717),  INT16_C( 16891),  INT16_C( 23948),
         INT16_C( 21596), -INT16_C( 15659), -INT16_C( 30383), -INT16_C( 23001),  INT16_C( 16855), -INT16_C( 10432), -INT16_C( 15585), -INT16_C(  2144) },
      { -INT16_C(  1591),  INT16_C(  6051),  INT16_C( 26590),  INT16_C(  6645),  INT16_C(  8644),  INT16_C( 21023),  INT16_C( 25616),  INT16_C( 23948),
         INT16_C( 21596), -INT16_C(   370),  INT16_C( 10605),  INT16_C( 26036),  INT16_C( 15413), -INT16_C( 10432), -INT16_C(  8704), -INT16_C(  2144) } },
    { {  INT16_C( 10385),  INT16_C( 28642), -INT16_C(   368),  INT16_C( 27823),  INT16_C( 17302), -INT16_C( 28327), -INT16_C(  6780), -INT16_C(  7953),
        -INT16_C( 15303), -INT16_C( 29790), -INT16_C( 14002),  INT16_C(  9521),  INT16_C( 28938),  INT16_C( 11004), -INT16_C( 25548), -INT16_C( 15071) },
      UINT16_C(  965),
      {  INT16_C( 21813), -INT16_C(  7166), -INT16_C( 26431),  INT16_C(  6951), -INT16_C( 21719),  INT16_C(  6144),  INT16_C( 14987),  INT16_C( 11741),
         INT16_C( 11205), -INT16_C(  2313),  INT16_C(   336),  INT16_C( 19559), -INT16_C( 25813),  INT16_C( 19944), -INT16_C( 21152), -INT16_C( 27312) },
      {  INT16_C( 20994), -INT16_C( 15238), -INT16_C( 24086),  INT16_C(  5343), -INT16_C(  8371), -INT16_C( 10196),  INT16_C(  2329), -INT16_C(  8698),
        -INT16_C(   716), -INT16_C( 31532),  INT16_C( 15358),  INT16_C( 10960), -INT16_C( 17962),  INT16_C( 14199), -INT16_C( 14490),  INT16_C( 27084) },
      {  INT16_C( 21813),  INT16_C( 28642), -INT16_C( 24086),  INT16_C( 27823),  INT16_C( 17302), -INT16_C( 28327),  INT16_C( 14987),  INT16_C( 11741),
         INT16_C( 11205), -INT16_C(  2313), -INT16_C( 14002),  INT16_C(  9521),  INT16_C( 28938),  INT16_C( 11004), -INT16_C( 25548), -INT16_C( 15071) } },
    { {  INT16_C( 17946),  INT16_C(  1069),  INT16_C(  3304),  INT16_C( 13592),  INT16_C( 17899),  INT16_C(  1293),  INT16_C(  4942), -INT16_C( 31773),
        -INT16_C( 18416),  INT16_C(  3847), -INT16_C(  9997), -INT16_C( 13767), -INT16_C( 20335), -INT16_C(  2303), -INT16_C( 12937), -INT16_C( 28320) },
      UINT16_C(36116),
      { -INT16_C(   874), -INT16_C( 20839), -INT16_C( 31439),  INT16_C( 16115),  INT16_C( 17034),  INT16_C( 27986),  INT16_C( 25285), -INT16_C( 13275),
         INT16_C(  6513), -INT16_C( 21852),  INT16_C( 13795), -INT16_C(  7078), -INT16_C( 11731), -INT16_C( 29263), -INT16_C( 15005), -INT16_C(  1765) },
      { -INT16_C( 19263), -INT16_C(  3416), -INT16_C( 25799), -INT16_C( 15567), -INT16_C( 31779), -INT16_C( 24015),  INT16_C( 22245),  INT16_C( 22383),
         INT16_C(  4975),  INT16_C( 20993),  INT16_C( 23625),  INT16_C( 30262), -INT16_C(  6098), -INT16_C( 28413),  INT16_C(  7853),  INT16_C( 28555) },
      {  INT16_C( 17946),  INT16_C(  1069), -INT16_C( 25799),  INT16_C( 13592),  INT16_C( 17034),  INT16_C(  1293),  INT16_C(  4942), -INT16_C( 31773),
         INT16_C(  6513),  INT16_C(  3847),  INT16_C( 23625),  INT16_C( 30262), -INT16_C( 20335), -INT16_C(  2303), -INT16_C( 12937),  INT16_C( 28555) } },
    { {  INT16_C( 13267),  INT16_C(  3169), -INT16_C( 27954), -INT16_C( 21296),  INT16_C(   277), -INT16_C(  1202), -INT16_C( 17065), -INT16_C( 14510),
         INT16_C( 21457),  INT16_C(  6681),  INT16_C( 20655), -INT16_C(  8816), -INT16_C( 27848), -INT16_C(  6801), -INT16_C(  1358), -INT16_C( 31404) },
      UINT16_C(46637),
      { -INT16_C(  1135),  INT16_C( 24904),  INT16_C( 24231), -INT16_C(  2462), -INT16_C( 17831), -INT16_C( 21581), -INT16_C( 31615), -INT16_C( 25858),
        -INT16_C( 20834),  INT16_C( 12010),  INT16_C(  8843), -INT16_C(  1342),  INT16_C( 29704),  INT16_C( 23796),  INT16_C(  8697), -INT16_C( 30190) },
      {  INT16_C( 23325), -INT16_C( 15124),  INT16_C( 20153),  INT16_C(  4794),  INT16_C( 28168), -INT16_C( 30275), -INT16_C( 17422), -INT16_C( 28380),
         INT16_C(  3689), -INT16_C(  2625), -INT16_C( 32463),  INT16_C( 14831), -INT16_C(  6923), -INT16_C(  4459), -INT16_C( 22523),  INT16_C(  8825) },
      {  INT16_C( 23325),  INT16_C(  3169),  INT16_C( 24231),  INT16_C(  4794),  INT16_C(   277), -INT16_C( 21581), -INT16_C( 17065), -INT16_C( 14510),
         INT16_C( 21457),  INT16_C( 12010),  INT16_C(  8843), -INT16_C(  8816),  INT16_C( 29704),  INT16_C( 23796), -INT16_C(  1358),  INT16_C(  8825) } },
    { {  INT16_C( 25859), -INT16_C( 17177), -INT16_C( 24141), -INT16_C( 17202), -INT16_C( 29937),  INT16_C(   581),  INT16_C( 26950), -INT16_C( 20333),
         INT16_C( 21112), -INT16_C( 22107), -INT16_C( 27436), -INT16_C( 13854),  INT16_C( 30584),  INT16_C( 32440),  INT16_C( 12575),  INT16_C(  8864) },
      UINT16_C(34710),
      {  INT16_C( 18910), -INT16_C( 21463),  INT16_C( 14341),  INT16_C( 19255),  INT16_C( 32314), -INT16_C( 12876),  INT16_C( 11310), -INT16_C( 11488),
        -INT16_C(  2859), -INT16_C( 18585), -INT16_C(  8003),  INT16_C( 29999),  INT16_C( 20062), -INT16_C(   346),  INT16_C( 15473),  INT16_C( 20358) },
      { -INT16_C( 20602), -INT16_C( 29700),  INT16_C( 13287),  INT16_C(  8918), -INT16_C( 29775), -INT16_C(  8209),  INT16_C(  4023), -INT16_C( 29262),
         INT16_C(  6659), -INT16_C( 16060),  INT16_C( 29690),  INT16_C( 22582), -INT16_C(  8766),  INT16_C( 13142), -INT16_C(  9191), -INT16_C( 24702) },
      {  INT16_C( 25859), -INT16_C( 21463),  INT16_C( 14341), -INT16_C( 17202),  INT16_C( 32314),  INT16_C(   581),  INT16_C( 26950), -INT16_C( 11488),
         INT16_C(  6659), -INT16_C( 16060),  INT16_C( 29690), -INT16_C( 13854),  INT16_C( 30584),  INT16_C( 32440),  INT16_C( 12575),  INT16_C( 20358) } },
    { {  INT16_C( 32395),  INT16_C( 29483),  INT16_C(   434),  INT16_C( 25493), -INT16_C( 31604),  INT16_C( 17475), -INT16_C(  2668), -INT16_C( 26671),
         INT16_C(  5391),  INT16_C(  2392), -INT16_C( 28791),  INT16_C( 19297), -INT16_C( 18324), -INT16_C( 31362),  INT16_C(   148),  INT16_C(  8229) },
      UINT16_C(20607),
      {  INT16_C( 12691),  INT16_C( 10321), -INT16_C(  8556), -INT16_C( 10324),  INT16_C( 16418), -INT16_C(  3123), -INT16_C(  9000),  INT16_C( 12296),
        -INT16_C( 28186),  INT16_C( 18367),  INT16_C( 11228),  INT16_C( 23295), -INT16_C( 27471), -INT16_C( 10661), -INT16_C(  9548),  INT16_C( 18214) },
      {  INT16_C( 30475), -INT16_C( 24721),  INT16_C(  6997),  INT16_C( 30583),  INT16_C( 17500),  INT16_C( 13418),  INT16_C( 29472),  INT16_C(  1636),
         INT16_C(  9220), -INT16_C(  7858),  INT16_C( 19791),  INT16_C(    59), -INT16_C( 26911), -INT16_C( 27178), -INT16_C(   912),  INT16_C( 31708) },
      {  INT16_C( 30475),  INT16_C( 10321),  INT16_C(  6997),  INT16_C( 30583),  INT16_C( 17500),  INT16_C( 13418),  INT16_C( 29472), -INT16_C( 26671),
         INT16_C(  5391),  INT16_C(  2392), -INT16_C( 28791),  INT16_C( 19297), -INT16_C( 26911), -INT16_C( 31362), -INT16_C(   912),  INT16_C(  8229) } },
    { {  INT16_C( 19316), -INT16_C( 14053), -INT16_C( 28057), -INT16_C( 15551), -INT16_C( 21546), -INT16_C(  2313),  INT16_C( 23326),  INT16_C(  9213),
         INT16_C( 19327), -INT16_C( 12540),  INT16_C( 16280),  INT16_C( 31439), -INT16_C( 22826),  INT16_C( 17935), -INT16_C(  4958),  INT16_C(  5826) },
      UINT16_C(56631),
      { -INT16_C( 24864),  INT16_C(  8559),  INT16_C( 17761),  INT16_C( 22732), -INT16_C(  5317),  INT16_C( 14516),  INT16_C( 13070),  INT16_C(  4739),
         INT16_C(  7170), -INT16_C( 11695),  INT16_C( 10134), -INT16_C( 23176),  INT16_C(  6766),  INT16_C( 12433), -INT16_C( 14031),  INT16_C(  4365) },
      {  INT16_C( 31847), -INT16_C( 14030), -INT16_C(   319), -INT16_C(   991), -INT16_C( 10775), -INT16_C(  2251), -INT16_C( 18423),  INT16_C(  2825),
         INT16_C( 23508),  INT16_C( 27357),  INT16_C( 21890), -INT16_C(  4080), -INT16_C( 24208), -INT16_C( 24288),  INT16_C( 11626), -INT16_C( 11598) },
      {  INT16_C( 31847),  INT16_C(  8559),  INT16_C( 17761), -INT16_C( 15551), -INT16_C(  5317),  INT16_C( 14516),  INT16_C( 23326),  INT16_C(  9213),
         INT16_C( 23508), -INT16_C( 12540),  INT16_C( 21890), -INT16_C(  4080),  INT16_C(  6766),  INT16_C( 17935),  INT16_C( 11626),  INT16_C(  4365) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_epi16");
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
    easysimd__m256i r = easysimd_mm256_mask_max_epi16(src, k, a, b);

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
test_easysimd_mm256_maskz_max_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C(36069),
      {  INT16_C( 26257),  INT16_C( 16392), -INT16_C( 16950),  INT16_C( 31082), -INT16_C( 29106), -INT16_C(  6261),  INT16_C(  6892), -INT16_C( 23904),
         INT16_C(  2310), -INT16_C( 12067), -INT16_C( 27778),  INT16_C( 19349),  INT16_C( 27104), -INT16_C( 31335),  INT16_C( 32293), -INT16_C( 18927) },
      {  INT16_C(  6628), -INT16_C( 20745),  INT16_C( 25046),  INT16_C(  9255), -INT16_C( 19729), -INT16_C(  9461), -INT16_C( 21299), -INT16_C( 11395),
         INT16_C( 23221),  INT16_C( 13219),  INT16_C( 14574), -INT16_C( 12673),  INT16_C(  6305), -INT16_C( 14509),  INT16_C( 25751),  INT16_C( 31613) },
      {  INT16_C( 26257),  INT16_C(     0),  INT16_C( 25046),  INT16_C(     0),  INT16_C(     0), -INT16_C(  6261),  INT16_C(  6892), -INT16_C( 11395),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 14574),  INT16_C( 19349),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31613) } },
    { UINT16_C(29822),
      {  INT16_C( 21546),  INT16_C( 20949), -INT16_C( 15239), -INT16_C( 31740), -INT16_C( 11872),  INT16_C(  7472), -INT16_C(  6492),  INT16_C( 18296),
         INT16_C( 26137), -INT16_C( 26496),  INT16_C(  8500), -INT16_C( 30799),  INT16_C( 18664),  INT16_C( 26347),  INT16_C( 27075), -INT16_C(  4646) },
      { -INT16_C( 20290),  INT16_C( 14143),  INT16_C( 17268),  INT16_C(  5307), -INT16_C(  5100), -INT16_C( 18382), -INT16_C( 21806), -INT16_C(  5121),
         INT16_C( 32528),  INT16_C( 17540),  INT16_C( 13729), -INT16_C( 30261), -INT16_C( 18819),  INT16_C( 16623), -INT16_C( 13792), -INT16_C(  8658) },
      {  INT16_C(     0),  INT16_C( 20949),  INT16_C( 17268),  INT16_C(  5307), -INT16_C(  5100),  INT16_C(  7472), -INT16_C(  6492),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 13729),  INT16_C(     0),  INT16_C( 18664),  INT16_C( 26347),  INT16_C( 27075),  INT16_C(     0) } },
    { UINT16_C(28026),
      { -INT16_C(  4587), -INT16_C( 12112), -INT16_C( 15357),  INT16_C( 13756), -INT16_C( 29060),  INT16_C( 31711), -INT16_C(  4230), -INT16_C(   261),
        -INT16_C( 25549), -INT16_C(   461), -INT16_C( 20443),  INT16_C(  5556), -INT16_C( 11024),  INT16_C(  7903),  INT16_C( 22962), -INT16_C( 14453) },
      {  INT16_C( 15175),  INT16_C( 19096),  INT16_C( 21759),  INT16_C( 31615),  INT16_C( 24291),  INT16_C( 24055), -INT16_C(  3507), -INT16_C( 32677),
        -INT16_C( 29042), -INT16_C( 19586),  INT16_C( 13118),  INT16_C( 11976), -INT16_C( 22777), -INT16_C( 17843), -INT16_C( 10240),  INT16_C( 18561) },
      {  INT16_C(     0),  INT16_C( 19096),  INT16_C(     0),  INT16_C( 31615),  INT16_C( 24291),  INT16_C( 31711), -INT16_C(  3507),  INT16_C(     0),
        -INT16_C( 25549),  INT16_C(     0),  INT16_C( 13118),  INT16_C( 11976),  INT16_C(     0),  INT16_C(  7903),  INT16_C( 22962),  INT16_C(     0) } },
    { UINT16_C( 6420),
      {  INT16_C(  5010),  INT16_C(  4718),  INT16_C( 20879), -INT16_C( 31120), -INT16_C( 16722),  INT16_C(  2424),  INT16_C(  1598), -INT16_C( 17001),
        -INT16_C( 10823), -INT16_C( 32016), -INT16_C(  2301),  INT16_C( 20521),  INT16_C( 10929),  INT16_C( 13097),  INT16_C( 15730),  INT16_C(  1100) },
      { -INT16_C( 17840), -INT16_C(  8426), -INT16_C( 30965), -INT16_C( 18075), -INT16_C(  8891), -INT16_C( 31806),  INT16_C( 23011), -INT16_C( 25280),
         INT16_C( 12334),  INT16_C( 12831),  INT16_C( 18472), -INT16_C(  9854), -INT16_C( 21646), -INT16_C(  7156),  INT16_C( 23016),  INT16_C( 14825) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 20879),  INT16_C(     0), -INT16_C(  8891),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 12334),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20521),  INT16_C( 10929),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(65299),
      {  INT16_C(  7960),  INT16_C( 32390), -INT16_C( 13352), -INT16_C( 25765),  INT16_C( 16207), -INT16_C( 28684),  INT16_C(  9180), -INT16_C(  1088),
        -INT16_C(  6059), -INT16_C( 10429), -INT16_C( 18751), -INT16_C( 12669),  INT16_C( 27546), -INT16_C( 31961),  INT16_C( 15012), -INT16_C( 17021) },
      {  INT16_C(  2393),  INT16_C( 12859), -INT16_C( 26923),  INT16_C(  9421), -INT16_C( 15915), -INT16_C( 20045),  INT16_C( 29668),  INT16_C( 14764),
        -INT16_C(  4005),  INT16_C(  7441), -INT16_C( 27482),  INT16_C( 16619),  INT16_C(  4863), -INT16_C( 23356),  INT16_C( 18252), -INT16_C( 22943) },
      {  INT16_C(  7960),  INT16_C( 32390),  INT16_C(     0),  INT16_C(     0),  INT16_C( 16207),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
        -INT16_C(  4005),  INT16_C(  7441), -INT16_C( 18751),  INT16_C( 16619),  INT16_C( 27546), -INT16_C( 23356),  INT16_C( 18252), -INT16_C( 17021) } },
    { UINT16_C(40016),
      {  INT16_C(  9688), -INT16_C( 23246),  INT16_C(  2121), -INT16_C(   666),  INT16_C( 19385),  INT16_C( 26224), -INT16_C( 13180), -INT16_C( 27306),
        -INT16_C(   791), -INT16_C( 11223),  INT16_C( 10556),  INT16_C(   230),  INT16_C( 13005),  INT16_C( 11847), -INT16_C( 26408), -INT16_C( 20278) },
      { -INT16_C(   835),  INT16_C(  1877), -INT16_C( 17404), -INT16_C( 16892),  INT16_C( 29703), -INT16_C( 29916),  INT16_C( 31296),  INT16_C( 10529),
         INT16_C( 19062), -INT16_C( 19715), -INT16_C(  7309),  INT16_C( 16563), -INT16_C(  1514), -INT16_C(  4498),  INT16_C( 14482),  INT16_C( 20639) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 29703),  INT16_C(     0),  INT16_C( 31296),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 10556),  INT16_C( 16563),  INT16_C( 13005),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20639) } },
    { UINT16_C(62517),
      {  INT16_C( 14679),  INT16_C( 23472), -INT16_C( 18441),  INT16_C(  7119),  INT16_C(  4163),  INT16_C( 25749),  INT16_C(  2873),  INT16_C( 14254),
         INT16_C(  8894),  INT16_C( 28954),  INT16_C( 12386), -INT16_C( 11925), -INT16_C(   481), -INT16_C( 16887),  INT16_C( 15950), -INT16_C( 23118) },
      {  INT16_C( 25464),  INT16_C( 28416), -INT16_C( 12518),  INT16_C( 23947),  INT16_C(  8415),  INT16_C(  6593),  INT16_C( 28716), -INT16_C(  5552),
         INT16_C( 27282), -INT16_C(  2981), -INT16_C( 14693), -INT16_C( 17723), -INT16_C( 12348),  INT16_C(  4728),  INT16_C( 10765), -INT16_C( 31305) },
      {  INT16_C( 25464),  INT16_C(     0), -INT16_C( 12518),  INT16_C(     0),  INT16_C(  8415),  INT16_C( 25749),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 12386),  INT16_C(     0), -INT16_C(   481),  INT16_C(  4728),  INT16_C( 15950), -INT16_C( 23118) } },
    { UINT16_C(46989),
      { -INT16_C( 22283), -INT16_C( 32633),  INT16_C( 26117), -INT16_C( 14432), -INT16_C( 13185), -INT16_C( 12489), -INT16_C( 13898),  INT16_C(  4410),
        -INT16_C( 10819), -INT16_C( 31784), -INT16_C( 25457),  INT16_C(  1874),  INT16_C( 24495),  INT16_C( 26161), -INT16_C( 16411), -INT16_C(  9698) },
      { -INT16_C( 23193),  INT16_C( 27738), -INT16_C(  1525), -INT16_C( 29901),  INT16_C( 27335),  INT16_C( 32090), -INT16_C( 27597), -INT16_C(  3697),
         INT16_C( 26473), -INT16_C(  1932), -INT16_C( 14845), -INT16_C( 19713),  INT16_C( 12581),  INT16_C(  2585),  INT16_C( 14320),  INT16_C( 22500) },
      { -INT16_C( 22283),  INT16_C(     0),  INT16_C( 26117), -INT16_C( 14432),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4410),
         INT16_C( 26473), -INT16_C(  1932), -INT16_C( 14845),  INT16_C(     0),  INT16_C( 24495),  INT16_C( 26161),  INT16_C(     0),  INT16_C( 22500) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_maskz_max_epi16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(  1788601513), -INT32_C(   865616670), -INT32_C(  1681679214),  INT32_C(   698797396), -INT32_C(  1433172952),  INT32_C(  1234936793), -INT32_C(  1343571131), -INT32_C(  1836999447) },
      UINT8_C(128),
      { -INT32_C(   647758564),  INT32_C(     7024484), -INT32_C(  1085012238),  INT32_C(   820477356), -INT32_C(  1274375663), -INT32_C(   386313427), -INT32_C(   623793858),  INT32_C(  1197171754) },
      { -INT32_C(   987709855), -INT32_C(   540701715),  INT32_C(  1033836945), -INT32_C(  1469151593),  INT32_C(  1163687960),  INT32_C(   154031819),  INT32_C(   702873599), -INT32_C(   999276701) },
      {  INT32_C(  1788601513), -INT32_C(   865616670), -INT32_C(  1681679214),  INT32_C(   698797396), -INT32_C(  1433172952),  INT32_C(  1234936793), -INT32_C(  1343571131),  INT32_C(  1197171754) } },
    { { -INT32_C(   360083203), -INT32_C(  1396093413), -INT32_C(     1415064),  INT32_C(   128407791), -INT32_C(  1689451568),  INT32_C(  1487240025), -INT32_C(   561870470), -INT32_C(   979176760) },
      UINT8_C(131),
      {  INT32_C(  2057219884), -INT32_C(   505263240), -INT32_C(  1915690699), -INT32_C(  1940007032), -INT32_C(  1612318684),  INT32_C(   639254173), -INT32_C(  1292961600), -INT32_C(   969559142) },
      { -INT32_C(   616442782),  INT32_C(  1421615903), -INT32_C(  1914598395), -INT32_C(  2011611549), -INT32_C(   735576266), -INT32_C(    33930947), -INT32_C(   726668999), -INT32_C(     6625892) },
      {  INT32_C(  2057219884),  INT32_C(  1421615903), -INT32_C(     1415064),  INT32_C(   128407791), -INT32_C(  1689451568),  INT32_C(  1487240025), -INT32_C(   561870470), -INT32_C(     6625892) } },
    { { -INT32_C(   656745543),  INT32_C(    70031103), -INT32_C(  2054091486), -INT32_C(  2129810613), -INT32_C(   413846102), -INT32_C(  1327214474), -INT32_C(   712731847),  INT32_C(   835985016) },
      UINT8_C(250),
      {  INT32_C(  1157171630),  INT32_C(  1114045749),  INT32_C(   965602446),  INT32_C(   786632697), -INT32_C(  1264203164),  INT32_C(  1122850222), -INT32_C(   138755111),  INT32_C(  1190259863) },
      {  INT32_C(   730524405),  INT32_C(  1986916839), -INT32_C(   693109795),  INT32_C(  1845859082),  INT32_C(   203663965), -INT32_C(   665972481),  INT32_C(  1808730323), -INT32_C(   357449228) },
      { -INT32_C(   656745543),  INT32_C(  1986916839), -INT32_C(  2054091486),  INT32_C(  1845859082),  INT32_C(   203663965),  INT32_C(  1122850222),  INT32_C(  1808730323),  INT32_C(  1190259863) } },
    { { -INT32_C(  1827325013),  INT32_C(   151618092), -INT32_C(  2015315843), -INT32_C(  1443437237), -INT32_C(  1900734065), -INT32_C(    43646167),  INT32_C(     6829323), -INT32_C(  1561716234) },
      UINT8_C( 84),
      { -INT32_C(  2122238465), -INT32_C(   151025090),  INT32_C(  1329759850), -INT32_C(  1780552836), -INT32_C(  1547735904),  INT32_C(   145669330),  INT32_C(  1040100900), -INT32_C(  1752063848) },
      {  INT32_C(   320410325),  INT32_C(   118102173),  INT32_C(   458706078), -INT32_C(   676317897),  INT32_C(  1954181026),  INT32_C(  1333536811),  INT32_C(  1871477719), -INT32_C(   251191781) },
      { -INT32_C(  1827325013),  INT32_C(   151618092),  INT32_C(  1329759850), -INT32_C(  1443437237),  INT32_C(  1954181026), -INT32_C(    43646167),  INT32_C(  1871477719), -INT32_C(  1561716234) } },
    { { -INT32_C(   855367632), -INT32_C(   690680264), -INT32_C(  1846465446),  INT32_C(    57188961),  INT32_C(  1031332369), -INT32_C(   494078965), -INT32_C(  1957619345),  INT32_C(  1736202295) },
      UINT8_C(120),
      { -INT32_C(  1884277376),  INT32_C(   921274122),  INT32_C(   446135160), -INT32_C(   970155037),  INT32_C(   131164435),  INT32_C(   242725877),  INT32_C(  1564803588), -INT32_C(    19550850) },
      { -INT32_C(   326269214), -INT32_C(  2061338867), -INT32_C(   710886926),  INT32_C(  1738263636),  INT32_C(   711945269),  INT32_C(   607774239),  INT32_C(  1719762664),  INT32_C(   224679467) },
      { -INT32_C(   855367632), -INT32_C(   690680264), -INT32_C(  1846465446),  INT32_C(  1738263636),  INT32_C(   711945269),  INT32_C(   607774239),  INT32_C(  1719762664),  INT32_C(  1736202295) } },
    { { -INT32_C(   369495332),  INT32_C(  1534008169),  INT32_C(   691015637),  INT32_C(   277990619),  INT32_C(  1480196152), -INT32_C(   830704666),  INT32_C(   490012146),  INT32_C(   808163411) },
      UINT8_C(138),
      {  INT32_C(  1089673508), -INT32_C(  1760211320),  INT32_C(  1265778303), -INT32_C(   813464881),  INT32_C(   817224637),  INT32_C(  1411547991),  INT32_C(  1353203895), -INT32_C(  1881483157) },
      {  INT32_C(  2060439281), -INT32_C(  1693326308), -INT32_C(   219773917), -INT32_C(  1010669050), -INT32_C(  1644923067), -INT32_C(  1292822789), -INT32_C(  1056794282),  INT32_C(  1666309489) },
      { -INT32_C(   369495332), -INT32_C(  1693326308),  INT32_C(   691015637), -INT32_C(   813464881),  INT32_C(  1480196152), -INT32_C(   830704666),  INT32_C(   490012146),  INT32_C(  1666309489) } },
    { { -INT32_C(   941809493),  INT32_C(   677637637),  INT32_C(  2031765874), -INT32_C(   113451853),  INT32_C(  1335242836), -INT32_C(  1660844217), -INT32_C(  1839266783), -INT32_C(  1930055455) },
      UINT8_C(208),
      { -INT32_C(  1042983982),  INT32_C(     3407286), -INT32_C(   189551592),  INT32_C(   424193257), -INT32_C(   899639486), -INT32_C(  1645478247),  INT32_C(   226393437),  INT32_C(  1172114035) },
      {  INT32_C(   335983453), -INT32_C(   938198352), -INT32_C(   809646106), -INT32_C(  1226242700),  INT32_C(   914377117), -INT32_C(  1529648313),  INT32_C(  1555124713), -INT32_C(  1197371557) },
      { -INT32_C(   941809493),  INT32_C(   677637637),  INT32_C(  2031765874), -INT32_C(   113451853),  INT32_C(   914377117), -INT32_C(  1660844217),  INT32_C(  1555124713),  INT32_C(  1172114035) } },
    { { -INT32_C(   221468606), -INT32_C(   927211294),  INT32_C(   479754408),  INT32_C(   450003325),  INT32_C(   307254218), -INT32_C(  1481235522), -INT32_C(   821860236),  INT32_C(   965256695) },
      UINT8_C( 77),
      {  INT32_C(   892283732),  INT32_C(  1591605222),  INT32_C(   282917263),  INT32_C(   517732043),  INT32_C(  1792929095),  INT32_C(   199197859),  INT32_C(   755150472), -INT32_C(  1954923722) },
      {  INT32_C(  1304471911),  INT32_C(   816618912),  INT32_C(  1631619222), -INT32_C(   981460098), -INT32_C(  1406116600),  INT32_C(  1773604833), -INT32_C(   191448387),  INT32_C(  1551831285) },
      {  INT32_C(  1304471911), -INT32_C(   927211294),  INT32_C(  1631619222),  INT32_C(   517732043),  INT32_C(   307254218), -INT32_C(  1481235522),  INT32_C(   755150472),  INT32_C(   965256695) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_epi32");
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
    easysimd__m256i r = easysimd_mm256_mask_max_epi32(src, k, a, b);

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
test_easysimd_mm256_maskz_max_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(220),
      {  INT32_C(   971490110),  INT32_C(  1627419383),  INT32_C(  1637187021),  INT32_C(  1942717964), -INT32_C(  1065892870), -INT32_C(   186308157),  INT32_C(  2045046850),  INT32_C(   324352980) },
      { -INT32_C(   179552770),  INT32_C(  2086096047), -INT32_C(   690033463),  INT32_C(  1816766834),  INT32_C(   791461996),  INT32_C(   740495850), -INT32_C(   693762303),  INT32_C(  1105853250) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1637187021),  INT32_C(  1942717964),  INT32_C(   791461996),  INT32_C(           0),  INT32_C(  2045046850),  INT32_C(  1105853250) } },
    { UINT8_C( 56),
      { -INT32_C(  2115488203),  INT32_C(  2034918541), -INT32_C(   336912318),  INT32_C(   710432618), -INT32_C(  1793816701), -INT32_C(  1298775637), -INT32_C(   487232281), -INT32_C(  1977928107) },
      { -INT32_C(    99941524), -INT32_C(  1435281817), -INT32_C(   527081610),  INT32_C(   973860278),  INT32_C(   550444917),  INT32_C(  1204970848),  INT32_C(   657115090),  INT32_C(  1790068221) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   973860278),  INT32_C(   550444917),  INT32_C(  1204970848),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 72),
      {  INT32_C(   330261693),  INT32_C(   915036631), -INT32_C(   588420369), -INT32_C(  1789843595),  INT32_C(  1542812150),  INT32_C(   187514180), -INT32_C(  1408674457), -INT32_C(   990612729) },
      { -INT32_C(  1361533993), -INT32_C(   320511235),  INT32_C(  1120457420), -INT32_C(   271115527), -INT32_C(   817181557),  INT32_C(  1893431305), -INT32_C(   736303923),  INT32_C(   781717591) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   271115527),  INT32_C(           0),  INT32_C(           0), -INT32_C(   736303923),  INT32_C(           0) } },
    { UINT8_C(180),
      { -INT32_C(   760095632), -INT32_C(  1818255935),  INT32_C(  2139939173), -INT32_C(  2079622216),  INT32_C(  1049549510), -INT32_C(  1727267147),  INT32_C(   737206043), -INT32_C(   388030857) },
      { -INT32_C(  1128623877), -INT32_C(  1823450835), -INT32_C(   233644998),  INT32_C(   511122776), -INT32_C(  1386478088),  INT32_C(   507995907), -INT32_C(  1119209658),  INT32_C(  1369778518) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  2139939173),  INT32_C(           0),  INT32_C(  1049549510),  INT32_C(   507995907),  INT32_C(           0),  INT32_C(  1369778518) } },
    { UINT8_C(186),
      { -INT32_C(  1176040096),  INT32_C(   989100637), -INT32_C(  1433213299),  INT32_C(  1654829149),  INT32_C(  1936085004), -INT32_C(   826702697), -INT32_C(   131828018),  INT32_C(  2092070172) },
      { -INT32_C(   533358205), -INT32_C(  1592121068),  INT32_C(  1833675792),  INT32_C(  1758522972), -INT32_C(   707054018), -INT32_C(  1985768262),  INT32_C(   662816779), -INT32_C(  1063046339) },
      {  INT32_C(           0),  INT32_C(   989100637),  INT32_C(           0),  INT32_C(  1758522972),  INT32_C(  1936085004), -INT32_C(   826702697),  INT32_C(           0),  INT32_C(  2092070172) } },
    { UINT8_C(204),
      {  INT32_C(    48275928),  INT32_C(  1746043323), -INT32_C(  1144750131), -INT32_C(  2047267505), -INT32_C(  1656697336),  INT32_C(   967362929),  INT32_C(  2105003850),  INT32_C(  1246312306) },
      { -INT32_C(  1806947624),  INT32_C(  2029805227),  INT32_C(   758366430), -INT32_C(   156029715),  INT32_C(  1821635578),  INT32_C(   111492028),  INT32_C(  2088967178),  INT32_C(   751225940) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   758366430), -INT32_C(   156029715),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2105003850),  INT32_C(  1246312306) } },
    { UINT8_C(246),
      {  INT32_C(  1906491411),  INT32_C(  2102336188),  INT32_C(  2053799246),  INT32_C(   594894896),  INT32_C(   786424307), -INT32_C(  1556552314),  INT32_C(   888648808), -INT32_C(  1909775493) },
      { -INT32_C(  1593848348),  INT32_C(   891113447), -INT32_C(    55539508), -INT32_C(   601938456), -INT32_C(  1945436666),  INT32_C(  1261388771),  INT32_C(  1920935671),  INT32_C(   771795530) },
      {  INT32_C(           0),  INT32_C(  2102336188),  INT32_C(  2053799246),  INT32_C(           0),  INT32_C(   786424307),  INT32_C(  1261388771),  INT32_C(  1920935671),  INT32_C(   771795530) } },
    { UINT8_C(119),
      {  INT32_C(  1331678720),  INT32_C(  1947964652),  INT32_C(  1767642948),  INT32_C(   879704118), -INT32_C(  2045248445),  INT32_C(  1383948843), -INT32_C(  1935871775), -INT32_C(   268186896) },
      { -INT32_C(  2076220776),  INT32_C(  1006131959), -INT32_C(  1482336911), -INT32_C(   790948723),  INT32_C(  1012331024),  INT32_C(   915330132), -INT32_C(  1262343484), -INT32_C(  1935292940) },
      {  INT32_C(  1331678720),  INT32_C(  1947964652),  INT32_C(  1767642948),  INT32_C(           0),  INT32_C(  1012331024),  INT32_C(  1383948843), -INT32_C(  1262343484),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_max_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_max_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 6775249007546690867),  INT64_C( 3293598981503999438),  INT64_C( 7055787360795357468),  INT64_C( 5961975932414894496) },
      {  INT64_C( 7084711508763613891), -INT64_C( 4156852515965537456),  INT64_C(  463699671822906581),  INT64_C( 7556072211687240453) },
      {  INT64_C( 7084711508763613891),  INT64_C( 3293598981503999438),  INT64_C( 7055787360795357468),  INT64_C( 7556072211687240453) } },
    { {  INT64_C( 1846309585767880361), -INT64_C( 8697705755991879973),  INT64_C( 3254417147380941759),  INT64_C( 3784031437107756151) },
      { -INT64_C( 2356209424094094486), -INT64_C( 3307982902718592546),  INT64_C( 2794239131909598066),  INT64_C( 9069340433540283933) },
      {  INT64_C( 1846309585767880361), -INT64_C( 3307982902718592546),  INT64_C( 3254417147380941759),  INT64_C( 9069340433540283933) } },
    { {  INT64_C( 4270255073758629216), -INT64_C( 4004295636348590457), -INT64_C( 6673444566300728178), -INT64_C( 3947642113804688217) },
      {  INT64_C( 7930680505998895222),  INT64_C( 5418805332573542934),  INT64_C( 7351259694688146907), -INT64_C( 1170276771780603761) },
      {  INT64_C( 7930680505998895222),  INT64_C( 5418805332573542934),  INT64_C( 7351259694688146907), -INT64_C( 1170276771780603761) } },
    { {  INT64_C( 8472345530599665471), -INT64_C( 1246349123261522772), -INT64_C( 5274186684885418839), -INT64_C( 8931638319806157148) },
      {  INT64_C( 5491363081284603489), -INT64_C( 3181114553703000359),  INT64_C( 2414502786125998501),  INT64_C(  414413234963886839) },
      {  INT64_C( 8472345530599665471), -INT64_C( 1246349123261522772),  INT64_C( 2414502786125998501),  INT64_C(  414413234963886839) } },
    { { -INT64_C( 9221969439169811944), -INT64_C( 4102982523159725807),  INT64_C( 5408566231530949233), -INT64_C( 5768749830339277496) },
      { -INT64_C( 4261974579797772366), -INT64_C(   67536890642807002),  INT64_C(  142422737656062809),  INT64_C( 8975920115504480585) },
      { -INT64_C( 4261974579797772366), -INT64_C(   67536890642807002),  INT64_C( 5408566231530949233),  INT64_C( 8975920115504480585) } },
    { { -INT64_C( 8498662601531197290),  INT64_C( 8540200509340061430), -INT64_C( 5169818537884880250), -INT64_C( 6565359789443371106) },
      {  INT64_C( 6849971084107626784),  INT64_C( 5520683914213253642),  INT64_C( 8935649490280169946), -INT64_C( 2112250063993136644) },
      {  INT64_C( 6849971084107626784),  INT64_C( 8540200509340061430),  INT64_C( 8935649490280169946), -INT64_C( 2112250063993136644) } },
    { { -INT64_C( 6680958544274547328),  INT64_C( 8028624646604205145), -INT64_C( 4502269475869362391), -INT64_C( 5694948861155776430) },
      {  INT64_C( 3410317650756722199), -INT64_C( 5904015127894680307), -INT64_C( 3155552735731529552), -INT64_C( 7817721755242856031) },
      {  INT64_C( 3410317650756722199),  INT64_C( 8028624646604205145), -INT64_C( 3155552735731529552), -INT64_C( 5694948861155776430) } },
    { {  INT64_C(  595205205382076990), -INT64_C(  439177614982541132),  INT64_C( 8707890019113540994), -INT64_C( 2399351387373059932) },
      {  INT64_C( 4820879710075614948), -INT64_C( 3672857801375641404), -INT64_C( 4805717823914412800),  INT64_C( 5037153454688933737) },
      {  INT64_C( 4820879710075614948), -INT64_C(  439177614982541132),  INT64_C( 8707890019113540994),  INT64_C( 5037153454688933737) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_max_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_max_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_max_epi64(a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 8253504771523755961), -INT64_C( 1287374294609573154),  INT64_C( 2232468619317105328),  INT64_C( 6655917424288895500) },
      UINT8_C(208),
      {  INT64_C(  737497700797036040),  INT64_C(  934179963856263410),  INT64_C( 1573881092470835321),  INT64_C( 4539650651662637470) },
      { -INT64_C( 4821989834927264310),  INT64_C( 6717393602609350001), -INT64_C( 3451445949607860607), -INT64_C( 5742368912251497767) },
      {  INT64_C( 8253504771523755961), -INT64_C( 1287374294609573154),  INT64_C( 2232468619317105328),  INT64_C( 6655917424288895500) } },
    { {  INT64_C( 3978876143351295150), -INT64_C( 5470295525949228333),  INT64_C( 1748753720997854785), -INT64_C(  920290933777873756) },
      UINT8_C(100),
      { -INT64_C( 2407629590506082828), -INT64_C( 6799936931765334332), -INT64_C( 7064749956751485827), -INT64_C( 5837554924356731104) },
      {  INT64_C( 5459778403541526083),  INT64_C( 1754517761539429973),  INT64_C( 5788916834881780064),  INT64_C( 5855775791319103708) },
      {  INT64_C( 3978876143351295150), -INT64_C( 5470295525949228333),  INT64_C( 5788916834881780064), -INT64_C(  920290933777873756) } },
    { {  INT64_C( 1287745529840186379),  INT64_C( 3929148599497820627),  INT64_C( 2944585926471315292), -INT64_C( 5270054152641264003) },
      UINT8_C(182),
      { -INT64_C( 5627385001227957994), -INT64_C( 3689083765251418803), -INT64_C(  113081227818804564), -INT64_C( 1245702865592648961) },
      { -INT64_C(  467555582354577239),  INT64_C( 8279367388465072446),  INT64_C( 4596154595446882194),  INT64_C( 8832393199491183052) },
      {  INT64_C( 1287745529840186379),  INT64_C( 8279367388465072446),  INT64_C( 4596154595446882194), -INT64_C( 5270054152641264003) } },
    { { -INT64_C( 4988636551366077410), -INT64_C(  499693509229072681),  INT64_C( 1313234998542955450), -INT64_C( 8266276371480282627) },
      UINT8_C(221),
      {  INT64_C( 5348046669627349139),  INT64_C( 6440053895450449385),  INT64_C(  934409597297252717), -INT64_C( 6805774602703485593) },
      { -INT64_C( 2033591859237689760), -INT64_C( 8120298940768159728),  INT64_C(  351140711305440576), -INT64_C( 4326479972157597115) },
      {  INT64_C( 5348046669627349139), -INT64_C(  499693509229072681),  INT64_C(  934409597297252717), -INT64_C( 4326479972157597115) } },
    { {  INT64_C( 4757497968267949864), -INT64_C( 5390660914188415361),  INT64_C( 9157340011627657587),  INT64_C( 3740739816217823683) },
      UINT8_C(191),
      { -INT64_C( 1608010075191814871),  INT64_C( 4207229025210224746),  INT64_C( 5965285476279809181),  INT64_C(   93608952027016901) },
      { -INT64_C( 2725837277196304922),  INT64_C( 4246101278541449522),  INT64_C( 1908972449647583650),  INT64_C( 5967537154931809271) },
      { -INT64_C( 1608010075191814871),  INT64_C( 4246101278541449522),  INT64_C( 5965285476279809181),  INT64_C( 5967537154931809271) } },
    { {  INT64_C( 1619816905148924464),  INT64_C( 1594474881276790088),  INT64_C( 6883400635207802904),  INT64_C( 5789939308985986459) },
      UINT8_C(  1),
      { -INT64_C( 5695932663829499667),  INT64_C( 6454568633650960836), -INT64_C( 5487599285505491075),  INT64_C( 2760746400132423759) },
      { -INT64_C(  778905919129766635),  INT64_C( 1004306636454516126),  INT64_C( 2466884987829006851),  INT64_C( 5360058940915781634) },
      { -INT64_C(  778905919129766635),  INT64_C( 1594474881276790088),  INT64_C( 6883400635207802904),  INT64_C( 5789939308985986459) } },
    { { -INT64_C( 7485551053303565066), -INT64_C( 7715756250705654449),  INT64_C( 6613634294350690002), -INT64_C( 1087061880707008150) },
      UINT8_C( 46),
      { -INT64_C( 4396145224961222477), -INT64_C(    7626448582746895), -INT64_C( 8446302839565154044),  INT64_C(  248952867036358475) },
      {  INT64_C( 5657837650094103693), -INT64_C( 4300299105657104246), -INT64_C( 6149126121932543027),  INT64_C( 3505037063960407633) },
      { -INT64_C( 7485551053303565066), -INT64_C(    7626448582746895), -INT64_C( 6149126121932543027),  INT64_C( 3505037063960407633) } },
    { { -INT64_C( 8310137459850456064),  INT64_C( 2220891786163719246),  INT64_C( 3451700006577108188),  INT64_C( 4244488969360823828) },
      UINT8_C(200),
      {  INT64_C( 5881233019100633305),  INT64_C( 8233636192900525109),  INT64_C( 4321343918044119037),  INT64_C( 8110050095341612879) },
      { -INT64_C( 2099135177295899064),  INT64_C( 4466600238782185483),  INT64_C( 2066054694918245644),  INT64_C( 1788047594771915208) },
      { -INT64_C( 8310137459850456064),  INT64_C( 2220891786163719246),  INT64_C( 3451700006577108188),  INT64_C( 8110050095341612879) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_epi64");
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
    easysimd__m256i r = easysimd_mm256_mask_max_epi64(src, k, a, b);

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
test_easysimd_mm256_maskz_max_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C( 40),
      {  INT64_C( 6823052431226507748),  INT64_C( 2748528600183232767),  INT64_C( 1777709018060054547), -INT64_C( 1630134821708865678) },
      { -INT64_C( 2804941188373119062), -INT64_C( 3636074256893346256),  INT64_C( 2844409872277308530), -INT64_C( 3986601431281990209) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 1630134821708865678) } },
    { UINT8_C( 62),
      { -INT64_C( 8296209343454872364), -INT64_C( 3431438574954331682), -INT64_C( 4546475928724323640),  INT64_C( 3101475288922434060) },
      { -INT64_C( 1063493741285203421),  INT64_C( 6958979073106281719), -INT64_C( 3386865506089166844), -INT64_C( 6224917488570933734) },
      {  INT64_C(                   0),  INT64_C( 6958979073106281719), -INT64_C( 3386865506089166844),  INT64_C( 3101475288922434060) } },
    { UINT8_C(200),
      {  INT64_C( 2027407736213035893), -INT64_C( 8002309978738858580),  INT64_C( 1091882327267946909), -INT64_C( 6255711863085623103) },
      { -INT64_C( 8984690232233031014), -INT64_C( 5267694133464464420),  INT64_C( 2438393949640145557),  INT64_C( 4927141035689732994) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 4927141035689732994) } },
    { UINT8_C(195),
      { -INT64_C( 3960884234811771084),  INT64_C( 9045365789907115406), -INT64_C( 8086215174176027045),  INT64_C( 1298020606343282055) },
      {  INT64_C( 5153602012608044277),  INT64_C( 6512980216426096546), -INT64_C( 4196679489727545586),  INT64_C( 8129851075651304675) },
      {  INT64_C( 5153602012608044277),  INT64_C( 9045365789907115406),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(164),
      {  INT64_C( 1727311534366731024),  INT64_C( 9127793963310702045), -INT64_C( 7108508006362587377), -INT64_C(  201949578917373306) },
      {  INT64_C( 1331741902706544819),  INT64_C( 1290100983697805609), -INT64_C( 6162648166156491973), -INT64_C( 3754758434617519622) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 6162648166156491973),  INT64_C(                   0) } },
    { UINT8_C(251),
      {  INT64_C( 4300165419826622264), -INT64_C( 6806821408140880143),  INT64_C( 8120114234295477974), -INT64_C( 5544119728667519063) },
      {  INT64_C( 8172315337517645443), -INT64_C( 3775342541814126853),  INT64_C( 8862857367797195183),  INT64_C( 6987226916257171656) },
      {  INT64_C( 8172315337517645443), -INT64_C( 3775342541814126853),  INT64_C(                   0),  INT64_C( 6987226916257171656) } },
    { UINT8_C(220),
      { -INT64_C( 7129253228316947153),  INT64_C(  397828204159699341), -INT64_C( 7288705383871933473),  INT64_C( 4925204154805591519) },
      {  INT64_C( 1150783514369111396), -INT64_C( 4028070846690653584),  INT64_C( 2318142427885072510), -INT64_C( 2510204313878551856) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 2318142427885072510),  INT64_C( 4925204154805591519) } },
    { UINT8_C(171),
      { -INT64_C( 4946855435426711423),  INT64_C( 8578068145198296355),  INT64_C( 7622299247714016324),  INT64_C( 6203919410078592118) },
      { -INT64_C( 6421122054128515336),  INT64_C( 3964082407983056645),  INT64_C(  651884585515138011), -INT64_C( 6499663286534476383) },
      { -INT64_C( 4946855435426711423),  INT64_C( 8578068145198296355),  INT64_C(                   0),  INT64_C( 6203919410078592118) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_max_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t src[32];
    const uint32_t k;
    const uint8_t a[32];
    const uint8_t b[32];
    const uint8_t r[32];
  } test_vec[] = {
    { { UINT8_C(194), UINT8_C(144), UINT8_C(165), UINT8_C(111), UINT8_C(239), UINT8_C(132), UINT8_C( 81), UINT8_C(251),
        UINT8_C(  4), UINT8_C(155), UINT8_C(133), UINT8_C( 69), UINT8_C( 40), UINT8_C(129), UINT8_C(130), UINT8_C( 52),
        UINT8_C(206), UINT8_C(163), UINT8_C(165), UINT8_C(155), UINT8_C(188), UINT8_C( 81), UINT8_C(183), UINT8_C(133),
        UINT8_C(114), UINT8_C(150), UINT8_C(228), UINT8_C( 66), UINT8_C(  2), UINT8_C(180), UINT8_C( 91), UINT8_C(196) },
      UINT32_C( 875757637),
      { UINT8_C(132), UINT8_C(132), UINT8_C( 47), UINT8_C(137), UINT8_C( 31), UINT8_C(180), UINT8_C(206), UINT8_C( 71),
        UINT8_C( 54), UINT8_C( 80), UINT8_C(123), UINT8_C(  4), UINT8_C(244), UINT8_C( 32), UINT8_C(159), UINT8_C(176),
        UINT8_C(113), UINT8_C( 86), UINT8_C( 53), UINT8_C(227), UINT8_C(237), UINT8_C( 26), UINT8_C( 38), UINT8_C(239),
        UINT8_C(206), UINT8_C(129), UINT8_C(179), UINT8_C( 19), UINT8_C(129), UINT8_C(230), UINT8_C( 72), UINT8_C(  6) },
      { UINT8_C(106), UINT8_C(119), UINT8_C(143), UINT8_C(138), UINT8_C( 44), UINT8_C( 93), UINT8_C(209), UINT8_C( 98),
        UINT8_C(173), UINT8_C( 77), UINT8_C(102), UINT8_C(161), UINT8_C(109), UINT8_C(  6), UINT8_C( 82), UINT8_C(223),
        UINT8_C( 92), UINT8_C(135), UINT8_C(194), UINT8_C( 73), UINT8_C(161), UINT8_C(232), UINT8_C( 56), UINT8_C(112),
        UINT8_C(105), UINT8_C(235), UINT8_C(131), UINT8_C(235), UINT8_C(209), UINT8_C(203), UINT8_C(241), UINT8_C( 60) },
      { UINT8_C(132), UINT8_C(144), UINT8_C(143), UINT8_C(111), UINT8_C(239), UINT8_C(132), UINT8_C(209), UINT8_C(251),
        UINT8_C(  4), UINT8_C(155), UINT8_C(133), UINT8_C( 69), UINT8_C( 40), UINT8_C(129), UINT8_C(130), UINT8_C( 52),
        UINT8_C(113), UINT8_C(135), UINT8_C(165), UINT8_C(155), UINT8_C(237), UINT8_C(232), UINT8_C(183), UINT8_C(133),
        UINT8_C(114), UINT8_C(150), UINT8_C(179), UINT8_C( 66), UINT8_C(209), UINT8_C(230), UINT8_C( 91), UINT8_C(196) } },
    { { UINT8_C( 67), UINT8_C(128), UINT8_C(198), UINT8_C(111), UINT8_C(221), UINT8_C(151), UINT8_C(209), UINT8_C(138),
        UINT8_C(228), UINT8_C( 55), UINT8_C( 44), UINT8_C( 82), UINT8_C( 61), UINT8_C(126), UINT8_C( 49), UINT8_C(154),
        UINT8_C(  5), UINT8_C(243), UINT8_C(227), UINT8_C(167), UINT8_C(220), UINT8_C( 28), UINT8_C( 23), UINT8_C( 69),
        UINT8_C(  7), UINT8_C(154), UINT8_C( 48), UINT8_C(217), UINT8_C(102), UINT8_C( 33), UINT8_C( 21), UINT8_C(169) },
      UINT32_C(2115558305),
      { UINT8_C(114), UINT8_C(233), UINT8_C(  9), UINT8_C( 87), UINT8_C( 32), UINT8_C( 53), UINT8_C(169), UINT8_C( 94),
        UINT8_C(179), UINT8_C(218), UINT8_C(248), UINT8_C(184), UINT8_C(205), UINT8_C(219), UINT8_C( 95), UINT8_C(169),
        UINT8_C(247), UINT8_C(118), UINT8_C(239),    UINT8_MAX, UINT8_C( 17), UINT8_C( 31), UINT8_C(216), UINT8_C(119),
        UINT8_C( 65), UINT8_C(237), UINT8_C( 32), UINT8_C(226), UINT8_C(200), UINT8_C( 56), UINT8_C( 97), UINT8_C( 58) },
      { UINT8_C( 33), UINT8_C(106), UINT8_C(145), UINT8_C( 65), UINT8_C(159), UINT8_C( 58), UINT8_C(159), UINT8_C( 82),
        UINT8_C( 20), UINT8_C(151), UINT8_C( 10), UINT8_C(226), UINT8_C(115), UINT8_C(106), UINT8_C(139), UINT8_C(106),
        UINT8_C(224), UINT8_C(122), UINT8_C(105), UINT8_C(241), UINT8_C(154), UINT8_C( 65), UINT8_C(104), UINT8_C(219),
        UINT8_C( 46), UINT8_C(136), UINT8_C(189), UINT8_C(246), UINT8_C(192), UINT8_C( 30), UINT8_C( 49), UINT8_C(225) },
      { UINT8_C(114), UINT8_C(128), UINT8_C(198), UINT8_C(111), UINT8_C(221), UINT8_C( 58), UINT8_C(209), UINT8_C( 94),
        UINT8_C(179), UINT8_C(218), UINT8_C( 44), UINT8_C(226), UINT8_C(205), UINT8_C(126), UINT8_C(139), UINT8_C(169),
        UINT8_C(  5), UINT8_C(243), UINT8_C(227),    UINT8_MAX, UINT8_C(154), UINT8_C( 28), UINT8_C( 23), UINT8_C( 69),
        UINT8_C(  7), UINT8_C(237), UINT8_C(189), UINT8_C(246), UINT8_C(200), UINT8_C( 56), UINT8_C( 97), UINT8_C(169) } },
    { { UINT8_C(136), UINT8_C(194), UINT8_C( 35), UINT8_C( 39), UINT8_C(253), UINT8_C(194), UINT8_C(121), UINT8_C( 17),
        UINT8_C( 90), UINT8_C(132), UINT8_C(243), UINT8_C(205), UINT8_C(238), UINT8_C(127), UINT8_C( 55), UINT8_C(206),
        UINT8_C(249), UINT8_C(161), UINT8_C(192), UINT8_C(147), UINT8_C(226), UINT8_C( 40), UINT8_C(110), UINT8_C( 17),
        UINT8_C(177), UINT8_C( 44), UINT8_C(  7), UINT8_C(113), UINT8_C( 74), UINT8_C( 56), UINT8_C( 83), UINT8_C(211) },
      UINT32_C(4177164027),
      { UINT8_C( 56), UINT8_C(116), UINT8_C(  9), UINT8_C(146), UINT8_C(248), UINT8_C(253), UINT8_C( 95), UINT8_C(230),
        UINT8_C(124), UINT8_C(151), UINT8_C(180), UINT8_C(117), UINT8_C( 56), UINT8_C(116), UINT8_C(  9), UINT8_C( 26),
        UINT8_C(157), UINT8_C(119), UINT8_C( 43), UINT8_C( 78), UINT8_C(163), UINT8_C( 51), UINT8_C(191), UINT8_C(238),
        UINT8_C(107), UINT8_C( 18), UINT8_C(193), UINT8_C(102), UINT8_C(136), UINT8_C(187), UINT8_C( 94), UINT8_C(193) },
      { UINT8_C( 47), UINT8_C(104), UINT8_C( 83), UINT8_C( 39), UINT8_C(101), UINT8_C(179), UINT8_C( 13), UINT8_C(225),
        UINT8_C( 74), UINT8_C(194), UINT8_C( 86), UINT8_C(130), UINT8_C( 54), UINT8_C( 95), UINT8_C(156), UINT8_C(211),
        UINT8_C(215), UINT8_C(200), UINT8_C( 33), UINT8_C(122), UINT8_C(251), UINT8_C(225), UINT8_C(104), UINT8_C(102),
        UINT8_C(243), UINT8_C( 41), UINT8_C(205), UINT8_C(124), UINT8_C(229), UINT8_C( 43), UINT8_C( 61), UINT8_C( 20) },
      { UINT8_C( 56), UINT8_C(116), UINT8_C( 35), UINT8_C(146), UINT8_C(248), UINT8_C(253), UINT8_C( 95), UINT8_C(230),
        UINT8_C( 90), UINT8_C(194), UINT8_C(180), UINT8_C(205), UINT8_C( 56), UINT8_C(116), UINT8_C(156), UINT8_C(206),
        UINT8_C(249), UINT8_C(200), UINT8_C(192), UINT8_C(122), UINT8_C(251), UINT8_C(225), UINT8_C(191), UINT8_C(238),
        UINT8_C(177), UINT8_C( 44), UINT8_C(  7), UINT8_C(124), UINT8_C(229), UINT8_C(187), UINT8_C( 94), UINT8_C(193) } },
    { { UINT8_C(147), UINT8_C(144), UINT8_C( 60), UINT8_C(248), UINT8_C( 67), UINT8_C( 73), UINT8_C(217), UINT8_C(141),
        UINT8_C( 11), UINT8_C( 48), UINT8_C( 15), UINT8_C( 66), UINT8_C(143), UINT8_C(172), UINT8_C( 21), UINT8_C(102),
        UINT8_C(116), UINT8_C( 55), UINT8_C(225), UINT8_C(111), UINT8_C( 24), UINT8_C( 73), UINT8_C(213), UINT8_C( 11),
        UINT8_C(115), UINT8_C(162), UINT8_C(135), UINT8_C( 88), UINT8_C(206), UINT8_C(196), UINT8_C(108), UINT8_C( 97) },
      UINT32_C(2556078165),
      { UINT8_C(242), UINT8_C( 51), UINT8_C( 38), UINT8_C(253), UINT8_C( 99), UINT8_C( 53), UINT8_C( 63), UINT8_C(243),
        UINT8_C(225), UINT8_C( 85), UINT8_C( 89), UINT8_C( 85), UINT8_C(140), UINT8_C( 58), UINT8_C(196), UINT8_C(164),
        UINT8_C(132), UINT8_C(154), UINT8_C(175), UINT8_C(247), UINT8_C( 60), UINT8_C( 55), UINT8_C( 79), UINT8_C( 10),
        UINT8_C(251), UINT8_C(187), UINT8_C(108), UINT8_C( 80), UINT8_C(100), UINT8_C(198), UINT8_C(233), UINT8_C( 86) },
      { UINT8_C(249), UINT8_C( 15), UINT8_C( 83), UINT8_C( 93), UINT8_C( 68), UINT8_C(147), UINT8_C( 80), UINT8_C( 38),
        UINT8_C(232), UINT8_C(169), UINT8_C(123), UINT8_C(116), UINT8_C(228), UINT8_C( 64), UINT8_C( 24), UINT8_C(104),
        UINT8_C(218), UINT8_C(199), UINT8_C( 95), UINT8_C( 22), UINT8_C(254), UINT8_C(174), UINT8_C( 33), UINT8_C(250),
        UINT8_C(105), UINT8_C(141), UINT8_C( 74), UINT8_C(205), UINT8_C( 83), UINT8_C( 51), UINT8_C( 35), UINT8_C( 76) },
      { UINT8_C(249), UINT8_C(144), UINT8_C( 83), UINT8_C(248), UINT8_C( 99), UINT8_C( 73), UINT8_C( 80), UINT8_C(141),
        UINT8_C( 11), UINT8_C( 48), UINT8_C( 15), UINT8_C(116), UINT8_C(143), UINT8_C( 64), UINT8_C( 21), UINT8_C(164),
        UINT8_C(116), UINT8_C(199), UINT8_C(225), UINT8_C(247), UINT8_C(254), UINT8_C( 73), UINT8_C( 79), UINT8_C( 11),
        UINT8_C(115), UINT8_C(162), UINT8_C(135), UINT8_C(205), UINT8_C(100), UINT8_C(196), UINT8_C(108), UINT8_C( 86) } },
    { { UINT8_C( 66), UINT8_C(119), UINT8_C(169), UINT8_C(135), UINT8_C( 10), UINT8_C(249), UINT8_C(173), UINT8_C(242),
        UINT8_C(163), UINT8_C( 40), UINT8_C(102), UINT8_C(135), UINT8_C(104), UINT8_C(126), UINT8_C(239), UINT8_C( 66),
        UINT8_C( 69), UINT8_C( 78), UINT8_C( 89), UINT8_C( 68), UINT8_C(252), UINT8_C(122), UINT8_C( 62), UINT8_C(101),
        UINT8_C(  7), UINT8_C(136), UINT8_C( 51), UINT8_C( 90), UINT8_C(188), UINT8_C( 86), UINT8_C(166), UINT8_C(254) },
      UINT32_C(3615838413),
      { UINT8_C( 73), UINT8_C( 50), UINT8_C(201), UINT8_C(236), UINT8_C( 91), UINT8_C( 47), UINT8_C(115), UINT8_C(195),
        UINT8_C(173), UINT8_C( 98), UINT8_C(  6), UINT8_C(243), UINT8_C(176), UINT8_C( 95), UINT8_C( 55), UINT8_C(172),
        UINT8_C(217), UINT8_C(117), UINT8_C( 18), UINT8_C(224), UINT8_C(253), UINT8_C( 69), UINT8_C( 58), UINT8_C(185),
        UINT8_C(155), UINT8_C(224), UINT8_C(184), UINT8_C(105), UINT8_C( 48), UINT8_C( 61), UINT8_C( 64), UINT8_C(122) },
      { UINT8_C(112), UINT8_C( 10), UINT8_C(102), UINT8_C(203), UINT8_C( 57), UINT8_C(218), UINT8_C(142), UINT8_C(231),
        UINT8_C( 60), UINT8_C(148), UINT8_C(218), UINT8_C(237), UINT8_C(243), UINT8_C( 17), UINT8_C(153), UINT8_C(204),
        UINT8_C(134), UINT8_C(171), UINT8_C(172), UINT8_C(131), UINT8_C(240), UINT8_C(230), UINT8_C( 61), UINT8_C(140),
        UINT8_C(199), UINT8_C(245), UINT8_C(245), UINT8_C(247), UINT8_C( 50), UINT8_C( 53), UINT8_C(113), UINT8_C(162) },
      { UINT8_C(112), UINT8_C(119), UINT8_C(201), UINT8_C(236), UINT8_C( 10), UINT8_C(249), UINT8_C(142), UINT8_C(231),
        UINT8_C(163), UINT8_C( 40), UINT8_C(102), UINT8_C(135), UINT8_C(243), UINT8_C(126), UINT8_C(153), UINT8_C( 66),
        UINT8_C(217), UINT8_C( 78), UINT8_C(172), UINT8_C( 68), UINT8_C(252), UINT8_C(122), UINT8_C( 62), UINT8_C(185),
        UINT8_C(199), UINT8_C(245), UINT8_C(245), UINT8_C( 90), UINT8_C( 50), UINT8_C( 86), UINT8_C(113), UINT8_C(162) } },
    { { UINT8_C( 63), UINT8_C(216), UINT8_C(109), UINT8_C(121), UINT8_C(178), UINT8_C(252), UINT8_C( 96), UINT8_C(238),
        UINT8_C(144), UINT8_C( 58), UINT8_C(219), UINT8_C(132), UINT8_C( 75), UINT8_C(117), UINT8_C( 80), UINT8_C(209),
        UINT8_C( 32), UINT8_C(253), UINT8_C( 84), UINT8_C( 17), UINT8_C(227), UINT8_C(145), UINT8_C(157), UINT8_C(170),
        UINT8_C(134), UINT8_C(146), UINT8_C(162), UINT8_C(185), UINT8_C(199), UINT8_C( 19), UINT8_C( 91), UINT8_C(  7) },
      UINT32_C(2642463211),
      { UINT8_C(197), UINT8_C(224), UINT8_C(140), UINT8_C( 85), UINT8_C( 26), UINT8_C(103), UINT8_C(217), UINT8_C(101),
        UINT8_C(220), UINT8_C( 42), UINT8_C( 54), UINT8_C(253), UINT8_C( 39), UINT8_C(138), UINT8_C( 14), UINT8_C( 10),
        UINT8_C( 28), UINT8_C(171), UINT8_C(181), UINT8_C(162), UINT8_C( 61), UINT8_C( 87), UINT8_C( 91), UINT8_C(  4),
        UINT8_C(106), UINT8_C(183), UINT8_C( 11), UINT8_C( 86), UINT8_C(128), UINT8_C(139), UINT8_C(243), UINT8_C( 69) },
      { UINT8_C(107), UINT8_C(127), UINT8_C(154), UINT8_C(133), UINT8_C(231), UINT8_C(116), UINT8_C(234), UINT8_C(195),
        UINT8_C(158), UINT8_C( 32), UINT8_C(192), UINT8_C(197), UINT8_C(171), UINT8_C(206), UINT8_C(207), UINT8_C(199),
        UINT8_C(121), UINT8_C(132), UINT8_C(105), UINT8_C(182), UINT8_C(219), UINT8_C(197), UINT8_C(187), UINT8_C( 70),
        UINT8_C(124), UINT8_C(198), UINT8_C(156), UINT8_C(252), UINT8_C( 82), UINT8_C(143), UINT8_C( 65), UINT8_C(189) },
      { UINT8_C(197), UINT8_C(224), UINT8_C(109), UINT8_C(133), UINT8_C(178), UINT8_C(116), UINT8_C(234), UINT8_C(195),
        UINT8_C(220), UINT8_C( 58), UINT8_C(219), UINT8_C(253), UINT8_C( 75), UINT8_C(117), UINT8_C(207), UINT8_C(199),
        UINT8_C( 32), UINT8_C(253), UINT8_C( 84), UINT8_C( 17), UINT8_C(227), UINT8_C(145), UINT8_C(157), UINT8_C( 70),
        UINT8_C(124), UINT8_C(146), UINT8_C(156), UINT8_C(252), UINT8_C(128), UINT8_C( 19), UINT8_C( 91), UINT8_C(189) } },
    { { UINT8_C( 15), UINT8_C(219), UINT8_C( 67), UINT8_C(246), UINT8_C( 79), UINT8_C( 45), UINT8_C(185), UINT8_C(237),
        UINT8_C( 78), UINT8_C(122), UINT8_C(178), UINT8_C(249), UINT8_C( 72), UINT8_C(130), UINT8_C(192), UINT8_C(194),
        UINT8_C(  6), UINT8_C( 41), UINT8_C(120), UINT8_C(226), UINT8_C(238), UINT8_C( 51), UINT8_C( 40), UINT8_C(106),
        UINT8_C(250), UINT8_C(196), UINT8_C(102), UINT8_C( 76), UINT8_C( 83), UINT8_C(167), UINT8_C(  9), UINT8_C( 98) },
      UINT32_C(3529002115),
      { UINT8_C(122), UINT8_C( 18), UINT8_C(192), UINT8_C(200), UINT8_C(140), UINT8_C(114), UINT8_C(193), UINT8_C(212),
        UINT8_C(244), UINT8_C(129), UINT8_C(150), UINT8_C(251), UINT8_C(170), UINT8_C( 15), UINT8_C(221), UINT8_C(153),
        UINT8_C( 66), UINT8_C(  5), UINT8_C(  3), UINT8_C( 60), UINT8_C(201), UINT8_C(106), UINT8_C(136), UINT8_C( 28),
        UINT8_C( 17), UINT8_C(146), UINT8_C(127), UINT8_C(148), UINT8_C(222), UINT8_C(215), UINT8_C(103), UINT8_C( 88) },
      { UINT8_C(233), UINT8_C( 39), UINT8_C( 32), UINT8_C(117), UINT8_C(153), UINT8_C(225), UINT8_C( 74), UINT8_C(142),
        UINT8_C( 98), UINT8_C(224), UINT8_C(137), UINT8_C( 13), UINT8_C(239), UINT8_C(102), UINT8_C(166), UINT8_C( 50),
        UINT8_C(107), UINT8_C(169), UINT8_C(110), UINT8_C( 52), UINT8_C( 19), UINT8_C(247), UINT8_C( 80), UINT8_C( 37),
        UINT8_C(137), UINT8_C(207), UINT8_C(185), UINT8_C(103), UINT8_C(167), UINT8_C( 32), UINT8_C(192), UINT8_C(144) },
      { UINT8_C(233), UINT8_C( 39), UINT8_C( 67), UINT8_C(246), UINT8_C( 79), UINT8_C( 45), UINT8_C(185), UINT8_C(212),
        UINT8_C( 78), UINT8_C(122), UINT8_C(150), UINT8_C(251), UINT8_C( 72), UINT8_C(130), UINT8_C(221), UINT8_C(194),
        UINT8_C(  6), UINT8_C( 41), UINT8_C(120), UINT8_C( 60), UINT8_C(201), UINT8_C( 51), UINT8_C(136), UINT8_C(106),
        UINT8_C(250), UINT8_C(207), UINT8_C(102), UINT8_C( 76), UINT8_C(222), UINT8_C(167), UINT8_C(192), UINT8_C(144) } },
    { { UINT8_C( 71), UINT8_C(224), UINT8_C(  6), UINT8_C(225), UINT8_C(194), UINT8_C( 80), UINT8_C(111), UINT8_C( 36),
        UINT8_C( 48), UINT8_C(248), UINT8_C( 49), UINT8_C( 32), UINT8_C( 94), UINT8_C(215), UINT8_C( 82), UINT8_C(201),
        UINT8_C(129), UINT8_C(192), UINT8_C(253), UINT8_C(148), UINT8_C(183), UINT8_C( 77), UINT8_C(185), UINT8_C( 64),
        UINT8_C( 29), UINT8_C(115), UINT8_C(168), UINT8_C(196), UINT8_C(147), UINT8_C(104), UINT8_C( 84), UINT8_C(219) },
      UINT32_C( 180116040),
      { UINT8_C(170), UINT8_C( 43), UINT8_C( 47), UINT8_C(219), UINT8_C( 35), UINT8_C( 96), UINT8_C(251), UINT8_C(129),
        UINT8_C( 56), UINT8_C( 77), UINT8_C( 74), UINT8_C(185), UINT8_C( 13), UINT8_C( 71), UINT8_C( 77), UINT8_C(197),
        UINT8_C(148), UINT8_C(  7), UINT8_C(  5), UINT8_C(177), UINT8_C(122), UINT8_C(173), UINT8_C(117), UINT8_C( 13),
        UINT8_C( 21), UINT8_C(202), UINT8_C(232), UINT8_C( 94), UINT8_C( 36), UINT8_C(164), UINT8_C(104), UINT8_C(207) },
      { UINT8_C(207), UINT8_C(151), UINT8_C(170), UINT8_C(242), UINT8_C(248), UINT8_C(165), UINT8_C(115), UINT8_C( 48),
        UINT8_C(242), UINT8_C(189), UINT8_C(233),    UINT8_MAX, UINT8_C(  4), UINT8_C( 54), UINT8_C(196), UINT8_C(153),
        UINT8_C( 61), UINT8_C(202), UINT8_C( 74), UINT8_C(183), UINT8_C(119), UINT8_C(192), UINT8_C(197), UINT8_C(141),
        UINT8_C(138), UINT8_C(173), UINT8_C(235), UINT8_C(174), UINT8_C( 82), UINT8_C( 83), UINT8_C(125), UINT8_C( 33) },
      { UINT8_C( 71), UINT8_C(224), UINT8_C(  6), UINT8_C(242), UINT8_C(194), UINT8_C( 80), UINT8_C(251), UINT8_C( 36),
        UINT8_C( 48), UINT8_C(189), UINT8_C( 49),    UINT8_MAX, UINT8_C( 13), UINT8_C(215), UINT8_C(196), UINT8_C(201),
        UINT8_C(129), UINT8_C(192), UINT8_C( 74), UINT8_C(183), UINT8_C(122), UINT8_C(192), UINT8_C(185), UINT8_C(141),
        UINT8_C( 29), UINT8_C(202), UINT8_C(168), UINT8_C(174), UINT8_C(147), UINT8_C(104), UINT8_C( 84), UINT8_C(219) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi8(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_epu8(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_epu8");
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
    easysimd__m256i r = easysimd_mm256_mask_max_epu8(src, k, a, b);

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
test_easysimd_mm256_maskz_max_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t k;
    const uint8_t a[32];
    const uint8_t b[32];
    const uint8_t r[32];
  } test_vec[] = {
    { UINT32_C(1473055444),
      { UINT8_C(162), UINT8_C(177), UINT8_C(254), UINT8_C(167), UINT8_C( 56), UINT8_C( 50), UINT8_C( 51), UINT8_C( 42),
        UINT8_C(113), UINT8_C( 54), UINT8_C( 97), UINT8_C( 77), UINT8_C(241), UINT8_C(143), UINT8_C(239), UINT8_C(132),
        UINT8_C(132), UINT8_C(251), UINT8_C(142), UINT8_C( 37), UINT8_C(232), UINT8_C(  4), UINT8_C(155), UINT8_C(149),
        UINT8_C(146), UINT8_C(103), UINT8_C( 58), UINT8_C(102), UINT8_C(114), UINT8_C(  8), UINT8_C(189), UINT8_C( 20) },
      { UINT8_C(185), UINT8_C(187), UINT8_C(187), UINT8_C(241), UINT8_C(237), UINT8_C(238), UINT8_C( 27), UINT8_C( 95),
        UINT8_C( 36), UINT8_C(125), UINT8_C(172), UINT8_C( 21), UINT8_C( 12), UINT8_C(155), UINT8_C(153), UINT8_C(145),
        UINT8_C(150), UINT8_C( 39), UINT8_C(182), UINT8_C(127), UINT8_C( 43), UINT8_C( 82), UINT8_C( 20), UINT8_C(189),
        UINT8_C(185), UINT8_C( 79), UINT8_C( 35), UINT8_C( 43), UINT8_C( 87), UINT8_C(225), UINT8_C( 63), UINT8_C( 16) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(254), UINT8_C(  0), UINT8_C(237), UINT8_C(  0), UINT8_C( 51), UINT8_C( 95),
        UINT8_C(  0), UINT8_C(125), UINT8_C(  0), UINT8_C( 77), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(150), UINT8_C(  0), UINT8_C(182), UINT8_C(127), UINT8_C(  0), UINT8_C(  0), UINT8_C(155), UINT8_C(189),
        UINT8_C(185), UINT8_C(103), UINT8_C( 58), UINT8_C(  0), UINT8_C(114), UINT8_C(  0), UINT8_C(189), UINT8_C(  0) } },
    { UINT32_C(2315385500),
      { UINT8_C(232), UINT8_C( 28), UINT8_C(233), UINT8_C( 12), UINT8_C(153), UINT8_C(149), UINT8_C( 33), UINT8_C(166),
        UINT8_C( 48), UINT8_C(187), UINT8_C( 55), UINT8_C(199), UINT8_C(226), UINT8_C(237), UINT8_C( 70), UINT8_C( 14),
        UINT8_C( 63), UINT8_C( 90), UINT8_C(203), UINT8_C(249), UINT8_C(169), UINT8_C(239), UINT8_C( 36), UINT8_C(  0),
        UINT8_C(208), UINT8_C(100), UINT8_C( 16), UINT8_C(108), UINT8_C( 94), UINT8_C( 17), UINT8_C(246), UINT8_C( 71) },
      { UINT8_C( 46), UINT8_C(223), UINT8_C( 83), UINT8_C(199), UINT8_C(116), UINT8_C(117), UINT8_C(109), UINT8_C(165),
        UINT8_C( 48), UINT8_C(164), UINT8_C(108), UINT8_C( 18), UINT8_C(146), UINT8_C(178), UINT8_C( 32), UINT8_C(209),
        UINT8_C( 12), UINT8_C(236), UINT8_C(202), UINT8_C(182), UINT8_C(219), UINT8_C(239), UINT8_C(182), UINT8_C(171),
        UINT8_C( 83), UINT8_C(199), UINT8_C( 23), UINT8_C(177), UINT8_C(216), UINT8_C( 14), UINT8_C(248), UINT8_C(  6) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C(233), UINT8_C(199), UINT8_C(153), UINT8_C(  0), UINT8_C(  0), UINT8_C(166),
        UINT8_C(  0), UINT8_C(187), UINT8_C(  0), UINT8_C(199), UINT8_C(226), UINT8_C(237), UINT8_C( 70), UINT8_C(209),
        UINT8_C( 63), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(199), UINT8_C(  0), UINT8_C(177), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 71) } },
    { UINT32_C(1657687277),
      { UINT8_C(193), UINT8_C( 59), UINT8_C(  7), UINT8_C(241), UINT8_C(224), UINT8_C(115), UINT8_C(  3), UINT8_C(114),
        UINT8_C( 37), UINT8_C( 36), UINT8_C( 67), UINT8_C( 49), UINT8_C( 16), UINT8_C( 14), UINT8_C(231), UINT8_C(235),
        UINT8_C(253), UINT8_C(158), UINT8_C(150), UINT8_C( 80), UINT8_C(101), UINT8_C(173), UINT8_C(  1), UINT8_C( 61),
        UINT8_C(187), UINT8_C(250), UINT8_C( 68), UINT8_C(169), UINT8_C( 70), UINT8_C( 18), UINT8_C( 11), UINT8_C(  7) },
      { UINT8_C( 77), UINT8_C( 18), UINT8_C(248), UINT8_C( 45), UINT8_C(133), UINT8_C(251), UINT8_C(159), UINT8_C(170),
        UINT8_C( 31), UINT8_C(227), UINT8_C(219), UINT8_C( 47), UINT8_C(241), UINT8_C(195), UINT8_C( 26), UINT8_C(238),
        UINT8_C( 97), UINT8_C(176), UINT8_C( 62), UINT8_C(198), UINT8_C( 94), UINT8_C( 63), UINT8_C(  3), UINT8_C( 25),
        UINT8_C( 57), UINT8_C( 71), UINT8_C(194), UINT8_C(127), UINT8_C( 89), UINT8_C(205), UINT8_C(134), UINT8_C(167) },
      { UINT8_C(193), UINT8_C(  0), UINT8_C(248), UINT8_C(241), UINT8_C(  0), UINT8_C(251), UINT8_C(159), UINT8_C(170),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(219), UINT8_C( 49), UINT8_C(  0), UINT8_C(  0), UINT8_C(231), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(176), UINT8_C(150), UINT8_C(198), UINT8_C(  0), UINT8_C(  0), UINT8_C(  3), UINT8_C( 61),
        UINT8_C(  0), UINT8_C(250), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(205), UINT8_C(134), UINT8_C(  0) } },
    { UINT32_C(1691647711),
      { UINT8_C(122), UINT8_C(116), UINT8_C( 14), UINT8_C(153), UINT8_C( 87), UINT8_C(234), UINT8_C(201), UINT8_C( 72),
        UINT8_C(173), UINT8_C(227), UINT8_C( 54), UINT8_C( 14), UINT8_C(148), UINT8_C(116), UINT8_C(212), UINT8_C(242),
        UINT8_C(179), UINT8_C(215), UINT8_C( 11), UINT8_C(237), UINT8_C( 31), UINT8_C(206), UINT8_C(108), UINT8_C(120),
        UINT8_C(155), UINT8_C(243), UINT8_C( 31), UINT8_C(123), UINT8_C(113), UINT8_C(244), UINT8_C(223), UINT8_C(235) },
      { UINT8_C(104), UINT8_C(238), UINT8_C(133), UINT8_C(191), UINT8_C(216), UINT8_C( 78), UINT8_C(  7), UINT8_C(133),
        UINT8_C( 49), UINT8_C( 61), UINT8_C(147), UINT8_C(197), UINT8_C(177), UINT8_C(103), UINT8_C(183), UINT8_C(100),
        UINT8_C( 62), UINT8_C(195), UINT8_C( 81), UINT8_C( 93), UINT8_C(145), UINT8_C(190), UINT8_C(214), UINT8_C( 44),
        UINT8_C(177), UINT8_C(245), UINT8_C(167), UINT8_C( 34), UINT8_C(233), UINT8_C(135), UINT8_C( 14), UINT8_C( 81) },
      { UINT8_C(122), UINT8_C(238), UINT8_C(133), UINT8_C(191), UINT8_C(216), UINT8_C(  0), UINT8_C(201), UINT8_C(133),
        UINT8_C(  0), UINT8_C(227), UINT8_C(147), UINT8_C(197), UINT8_C(177), UINT8_C(116), UINT8_C(212), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C( 81), UINT8_C(  0), UINT8_C(145), UINT8_C(  0), UINT8_C(214), UINT8_C(120),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(167), UINT8_C(  0), UINT8_C(  0), UINT8_C(244), UINT8_C(223), UINT8_C(  0) } },
    { UINT32_C(1292931957),
      { UINT8_C(225), UINT8_C( 23), UINT8_C(210), UINT8_C( 18), UINT8_C( 84), UINT8_C(101), UINT8_C(216), UINT8_C(  5),
        UINT8_C(204), UINT8_C(143), UINT8_C(106), UINT8_C( 10), UINT8_C( 82), UINT8_C(187), UINT8_C(104), UINT8_C(227),
        UINT8_C(121), UINT8_C( 62), UINT8_C( 16), UINT8_C( 42), UINT8_C( 51), UINT8_C(183), UINT8_C( 77), UINT8_C( 29),
        UINT8_C( 62), UINT8_C( 91), UINT8_C(110), UINT8_C(179), UINT8_C(238), UINT8_C(127), UINT8_C(  0), UINT8_C(207) },
      { UINT8_C(150), UINT8_C(210), UINT8_C(225), UINT8_C(235), UINT8_C( 55), UINT8_C(185), UINT8_C(240), UINT8_C(  3),
        UINT8_C( 73), UINT8_C( 90), UINT8_C( 14), UINT8_C(155), UINT8_C( 22), UINT8_C(118), UINT8_C(127), UINT8_C(143),
        UINT8_C(180), UINT8_C(143), UINT8_C(186), UINT8_C(231), UINT8_C( 70), UINT8_C(  7), UINT8_C(  4), UINT8_C(133),
        UINT8_C( 98), UINT8_C(115), UINT8_C( 56), UINT8_C( 80), UINT8_C(242), UINT8_C( 57), UINT8_C( 31), UINT8_C(136) },
      { UINT8_C(225), UINT8_C(  0), UINT8_C(225), UINT8_C(  0), UINT8_C( 84), UINT8_C(185), UINT8_C(240), UINT8_C(  0),
        UINT8_C(204), UINT8_C(143), UINT8_C(  0), UINT8_C(  0), UINT8_C( 82), UINT8_C(  0), UINT8_C(  0), UINT8_C(227),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 70), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C( 98), UINT8_C(  0), UINT8_C(110), UINT8_C(179), UINT8_C(  0), UINT8_C(  0), UINT8_C( 31), UINT8_C(  0) } },
    { UINT32_C(1131610123),
      { UINT8_C(186), UINT8_C(100), UINT8_C( 70), UINT8_C(  3), UINT8_C(190), UINT8_C( 84), UINT8_C(158), UINT8_C(212),
        UINT8_C(202), UINT8_C( 29), UINT8_C(100), UINT8_C(126), UINT8_C(172), UINT8_C( 30), UINT8_C(102), UINT8_C(243),
        UINT8_C( 37), UINT8_C(106), UINT8_C(120), UINT8_C(135), UINT8_C(221), UINT8_C(176), UINT8_C(215), UINT8_C(207),
        UINT8_C(233), UINT8_C(246), UINT8_C( 88), UINT8_C(245), UINT8_C(246), UINT8_C(203), UINT8_C( 56), UINT8_C(176) },
      { UINT8_C( 47), UINT8_C(126), UINT8_C(179), UINT8_C(238), UINT8_C(211), UINT8_C( 82), UINT8_C(194), UINT8_C(157),
        UINT8_C(111), UINT8_C( 38), UINT8_C( 28), UINT8_C( 28), UINT8_C( 68), UINT8_C(130), UINT8_C( 15), UINT8_C(105),
        UINT8_C(236), UINT8_C(135), UINT8_C(240), UINT8_C(202), UINT8_C( 55), UINT8_C(199), UINT8_C(153), UINT8_C( 33),
        UINT8_C(189), UINT8_C(241), UINT8_C( 22), UINT8_C(180), UINT8_C(189), UINT8_C( 78), UINT8_C(100), UINT8_C(236) },
      { UINT8_C(186), UINT8_C(126), UINT8_C(  0), UINT8_C(238), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(236), UINT8_C(135), UINT8_C(  0), UINT8_C(  0), UINT8_C(221), UINT8_C(199), UINT8_C(215), UINT8_C(  0),
        UINT8_C(233), UINT8_C(246), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(100), UINT8_C(  0) } },
    { UINT32_C(2681870540),
      { UINT8_C(106), UINT8_C(157), UINT8_C( 61), UINT8_C(217), UINT8_C(195), UINT8_C( 89), UINT8_C(245), UINT8_C(  8),
        UINT8_C(219), UINT8_C(  4), UINT8_C(113), UINT8_C(199), UINT8_C(139), UINT8_C( 98), UINT8_C(145), UINT8_C(195),
        UINT8_C( 41), UINT8_C( 43), UINT8_C(228), UINT8_C(231), UINT8_C( 28), UINT8_C(250), UINT8_C(155), UINT8_C(217),
        UINT8_C( 72),    UINT8_MAX, UINT8_C(198), UINT8_C( 20), UINT8_C( 23), UINT8_C(160), UINT8_C(180), UINT8_C(129) },
      { UINT8_C( 61), UINT8_C(241), UINT8_C( 91), UINT8_C(  1), UINT8_C( 74), UINT8_C( 80), UINT8_C(  9), UINT8_C( 37),
        UINT8_C( 85), UINT8_C(122), UINT8_C(236), UINT8_C(224), UINT8_C(220), UINT8_C(126), UINT8_C(163), UINT8_C(  6),
        UINT8_C(169), UINT8_C(135), UINT8_C(237), UINT8_C(197), UINT8_C(129), UINT8_C(136), UINT8_C(159), UINT8_C(201),
        UINT8_C(135), UINT8_C(101), UINT8_C(222), UINT8_C(159), UINT8_C(  5), UINT8_C(146), UINT8_C( 32), UINT8_C( 67) },
      { UINT8_C(  0), UINT8_C(  0), UINT8_C( 91), UINT8_C(217), UINT8_C(  0), UINT8_C(  0), UINT8_C(245), UINT8_C( 37),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(224), UINT8_C(220), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(135), UINT8_C(  0), UINT8_C(231), UINT8_C(129), UINT8_C(  0), UINT8_C(159), UINT8_C(217),
        UINT8_C(135),    UINT8_MAX, UINT8_C(222), UINT8_C(159), UINT8_C( 23), UINT8_C(  0), UINT8_C(  0), UINT8_C(129) } },
    { UINT32_C(3443817347),
      { UINT8_C(204), UINT8_C( 77), UINT8_C(242), UINT8_C( 33), UINT8_C(199), UINT8_C(222), UINT8_C(  1), UINT8_C(164),
        UINT8_C( 92), UINT8_C(165), UINT8_C(170), UINT8_C(  5), UINT8_C( 44), UINT8_C(151), UINT8_C(203), UINT8_C(174),
        UINT8_C( 31), UINT8_C(106), UINT8_C(119), UINT8_C(166), UINT8_C(207), UINT8_C( 85), UINT8_C( 69), UINT8_C(212),
        UINT8_C(231), UINT8_C(102), UINT8_C( 23), UINT8_C(106), UINT8_C(225), UINT8_C( 91), UINT8_C( 55), UINT8_C(173) },
      { UINT8_C(168), UINT8_C( 41), UINT8_C(206), UINT8_C(112), UINT8_C(  8), UINT8_C(208), UINT8_C( 20), UINT8_C(100),
        UINT8_C(117), UINT8_C(190), UINT8_C(106), UINT8_C(161), UINT8_C( 85), UINT8_C( 53), UINT8_C( 79), UINT8_C(116),
        UINT8_C(159), UINT8_C(199), UINT8_C( 26), UINT8_C(110), UINT8_C( 28), UINT8_C( 96), UINT8_C( 66), UINT8_C(  4),
        UINT8_C(198), UINT8_C( 90), UINT8_C(110), UINT8_C(167), UINT8_C(181), UINT8_C(166), UINT8_C( 85), UINT8_C( 94) },
      { UINT8_C(204), UINT8_C( 77), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(164),
        UINT8_C(117), UINT8_C(190), UINT8_C(  0), UINT8_C(161), UINT8_C( 85), UINT8_C(151), UINT8_C(203), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(119), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 69), UINT8_C(  0),
        UINT8_C(231), UINT8_C(  0), UINT8_C(110), UINT8_C(167), UINT8_C(  0), UINT8_C(  0), UINT8_C( 85), UINT8_C(173) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi8(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi8(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_epu8(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_epu8");
    easysimd_test_x86_assert_equal_u8x32(r, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_u8x32();
    easysimd__m256i b = easysimd_test_x86_random_u8x32();
    easysimd__m256i r = easysimd_mm256_maskz_max_epu8(k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t src[16];
    const uint16_t k;
    const uint16_t a[16];
    const uint16_t b[16];
    const uint16_t r[16];
  } test_vec[] = {
    { { UINT16_C(10219), UINT16_C(58132), UINT16_C(34764), UINT16_C(48659), UINT16_C(64581), UINT16_C(18878), UINT16_C(33330), UINT16_C(28898),
        UINT16_C(11596), UINT16_C(50215), UINT16_C(60653), UINT16_C(30545), UINT16_C(15514), UINT16_C(60453), UINT16_C(41871), UINT16_C(31245) },
      UINT16_C( 8650),
      { UINT16_C(38749), UINT16_C(28841), UINT16_C(61013), UINT16_C( 4972), UINT16_C(40759), UINT16_C( 6806), UINT16_C(57871), UINT16_C(13895),
        UINT16_C(13478), UINT16_C(63267), UINT16_C(48555), UINT16_C(53299), UINT16_C(50089), UINT16_C(46707), UINT16_C(15933), UINT16_C(39896) },
      { UINT16_C(33237), UINT16_C(10763), UINT16_C(30831), UINT16_C(42558), UINT16_C(54295), UINT16_C( 9920), UINT16_C( 1974), UINT16_C(23900),
        UINT16_C(32571), UINT16_C(58964), UINT16_C(34876), UINT16_C(58807), UINT16_C(10827), UINT16_C(34972), UINT16_C(29800), UINT16_C(15651) },
      { UINT16_C(10219), UINT16_C(28841), UINT16_C(34764), UINT16_C(42558), UINT16_C(64581), UINT16_C(18878), UINT16_C(57871), UINT16_C(23900),
        UINT16_C(32571), UINT16_C(50215), UINT16_C(60653), UINT16_C(30545), UINT16_C(15514), UINT16_C(46707), UINT16_C(41871), UINT16_C(31245) } },
    { { UINT16_C(12277), UINT16_C(25704), UINT16_C(42663), UINT16_C(48650), UINT16_C(52090), UINT16_C(12516), UINT16_C(16594), UINT16_C( 3725),
        UINT16_C(58048), UINT16_C(64756), UINT16_C(43882), UINT16_C(46562), UINT16_C(32470), UINT16_C(15933), UINT16_C(25074), UINT16_C(59260) },
      UINT16_C(58512),
      { UINT16_C(14155), UINT16_C(21898), UINT16_C( 1269), UINT16_C(55584), UINT16_C(62260), UINT16_C(49689), UINT16_C(55553), UINT16_C(62884),
        UINT16_C( 3798), UINT16_C(47265), UINT16_C(30659), UINT16_C(   54), UINT16_C(10421), UINT16_C(12641), UINT16_C(61711), UINT16_C(23061) },
      { UINT16_C(40744), UINT16_C( 7599), UINT16_C(53411), UINT16_C(55542), UINT16_C( 4291), UINT16_C(50330), UINT16_C(16105), UINT16_C(49081),
        UINT16_C(23116), UINT16_C( 3959), UINT16_C(44497), UINT16_C(34575), UINT16_C(29141), UINT16_C(58552), UINT16_C(52834), UINT16_C(35646) },
      { UINT16_C(12277), UINT16_C(25704), UINT16_C(42663), UINT16_C(48650), UINT16_C(62260), UINT16_C(12516), UINT16_C(16594), UINT16_C(62884),
        UINT16_C(58048), UINT16_C(64756), UINT16_C(44497), UINT16_C(46562), UINT16_C(32470), UINT16_C(58552), UINT16_C(61711), UINT16_C(35646) } },
    { { UINT16_C(61037), UINT16_C( 4520), UINT16_C(40894), UINT16_C(33257), UINT16_C(33711), UINT16_C(38981), UINT16_C(65217), UINT16_C( 3416),
        UINT16_C(53081), UINT16_C(10780), UINT16_C(11133), UINT16_C(21169), UINT16_C(27292), UINT16_C(65335), UINT16_C(30008), UINT16_C(42378) },
      UINT16_C(12899),
      { UINT16_C( 8630), UINT16_C(40913), UINT16_C(32930), UINT16_C(59170), UINT16_C(58137), UINT16_C(29158), UINT16_C(16368), UINT16_C( 3136),
        UINT16_C(48489), UINT16_C( 6968), UINT16_C(54288), UINT16_C(18309), UINT16_C(48595), UINT16_C(23996), UINT16_C( 8290), UINT16_C( 6544) },
      { UINT16_C(24897), UINT16_C(58552), UINT16_C(56290), UINT16_C(64459), UINT16_C(45502), UINT16_C(44908), UINT16_C(44272), UINT16_C(23227),
        UINT16_C(62314), UINT16_C(31349), UINT16_C(64200), UINT16_C(39873), UINT16_C(32183), UINT16_C( 6649), UINT16_C(35229), UINT16_C(57138) },
      { UINT16_C(24897), UINT16_C(58552), UINT16_C(40894), UINT16_C(33257), UINT16_C(33711), UINT16_C(44908), UINT16_C(44272), UINT16_C( 3416),
        UINT16_C(53081), UINT16_C(31349), UINT16_C(11133), UINT16_C(21169), UINT16_C(48595), UINT16_C(23996), UINT16_C(30008), UINT16_C(42378) } },
    { { UINT16_C(60394), UINT16_C(52419), UINT16_C(36550), UINT16_C(33991), UINT16_C(13120), UINT16_C(12339), UINT16_C(61408), UINT16_C(19082),
        UINT16_C(65506), UINT16_C(43716), UINT16_C(34297), UINT16_C(45126), UINT16_C(16130), UINT16_C(41162), UINT16_C(64712), UINT16_C(45695) },
      UINT16_C(17127),
      { UINT16_C(44415), UINT16_C(18128), UINT16_C( 4146), UINT16_C(25978), UINT16_C(23105), UINT16_C(52052), UINT16_C(14244), UINT16_C(26827),
        UINT16_C(50401), UINT16_C(10221), UINT16_C(61301), UINT16_C(16230), UINT16_C(11919), UINT16_C( 3643), UINT16_C( 9185), UINT16_C(24656) },
      { UINT16_C( 8656), UINT16_C(  678), UINT16_C( 8241), UINT16_C(29288), UINT16_C(48250), UINT16_C( 7742), UINT16_C( 2547), UINT16_C(54662),
        UINT16_C(29645), UINT16_C(17148), UINT16_C(25443), UINT16_C(62081), UINT16_C(48529), UINT16_C(29185), UINT16_C(20960), UINT16_C(45266) },
      { UINT16_C(44415), UINT16_C(18128), UINT16_C( 8241), UINT16_C(33991), UINT16_C(13120), UINT16_C(52052), UINT16_C(14244), UINT16_C(54662),
        UINT16_C(65506), UINT16_C(17148), UINT16_C(34297), UINT16_C(45126), UINT16_C(16130), UINT16_C(41162), UINT16_C(20960), UINT16_C(45695) } },
    { { UINT16_C(31090), UINT16_C(42163), UINT16_C( 7065), UINT16_C( 5142), UINT16_C(21719), UINT16_C(52018), UINT16_C(47453), UINT16_C(11168),
        UINT16_C(39980), UINT16_C(36717), UINT16_C(61439), UINT16_C(37250), UINT16_C(33708), UINT16_C(35843), UINT16_C(54996), UINT16_C(18236) },
      UINT16_C(61263),
      { UINT16_C(59627), UINT16_C(  266), UINT16_C(58108), UINT16_C(12118), UINT16_C(45997), UINT16_C(19944), UINT16_C( 5342), UINT16_C(19689),
        UINT16_C(59812), UINT16_C( 9787), UINT16_C(59258), UINT16_C(32169), UINT16_C(32115), UINT16_C(44883), UINT16_C(41668), UINT16_C(44959) },
      { UINT16_C(43403), UINT16_C(34737), UINT16_C( 1931), UINT16_C(14518), UINT16_C(40634), UINT16_C(39301), UINT16_C(28595), UINT16_C(22501),
        UINT16_C( 8280), UINT16_C(53885), UINT16_C( 9735), UINT16_C(31311), UINT16_C(41891), UINT16_C(26665), UINT16_C(51269), UINT16_C(53271) },
      { UINT16_C(59627), UINT16_C(34737), UINT16_C(58108), UINT16_C(14518), UINT16_C(21719), UINT16_C(52018), UINT16_C(28595), UINT16_C(11168),
        UINT16_C(59812), UINT16_C(53885), UINT16_C(59258), UINT16_C(32169), UINT16_C(33708), UINT16_C(44883), UINT16_C(51269), UINT16_C(53271) } },
    { { UINT16_C(51314), UINT16_C(64856), UINT16_C( 3791), UINT16_C(35382), UINT16_C(48045), UINT16_C(24611), UINT16_C( 2090), UINT16_C(33463),
        UINT16_C(13352), UINT16_C(12116), UINT16_C(42074), UINT16_C(64937), UINT16_C(53831), UINT16_C(35941), UINT16_C(32155), UINT16_C( 3421) },
      UINT16_C(46405),
      { UINT16_C( 5386), UINT16_C(16579), UINT16_C(28831), UINT16_C(49916), UINT16_C( 9936), UINT16_C(34762), UINT16_C(62121), UINT16_C(64955),
        UINT16_C( 5409), UINT16_C(51873), UINT16_C(59411), UINT16_C(30876), UINT16_C(14197), UINT16_C(54005), UINT16_C(15172), UINT16_C(20359) },
      { UINT16_C(19024), UINT16_C(61327), UINT16_C(35771), UINT16_C(35761), UINT16_C(31666), UINT16_C(23315), UINT16_C(52845), UINT16_C(36440),
        UINT16_C(64228), UINT16_C(63320), UINT16_C(62690), UINT16_C(22383), UINT16_C(25900), UINT16_C(28713), UINT16_C(45216), UINT16_C(61631) },
      { UINT16_C(19024), UINT16_C(64856), UINT16_C(35771), UINT16_C(35382), UINT16_C(48045), UINT16_C(24611), UINT16_C(62121), UINT16_C(33463),
        UINT16_C(64228), UINT16_C(12116), UINT16_C(62690), UINT16_C(64937), UINT16_C(25900), UINT16_C(54005), UINT16_C(32155), UINT16_C(61631) } },
    { { UINT16_C(20475), UINT16_C(46815), UINT16_C(37082), UINT16_C(35905), UINT16_C(21515), UINT16_C(30951), UINT16_C(16419), UINT16_C( 1798),
        UINT16_C(24122), UINT16_C( 7422), UINT16_C(27986), UINT16_C(32372), UINT16_C(40402), UINT16_C(29423), UINT16_C(44622), UINT16_C(18786) },
      UINT16_C(16893),
      { UINT16_C(55551), UINT16_C(16593), UINT16_C(56420), UINT16_C(19605), UINT16_C(47188), UINT16_C(23180), UINT16_C(50879), UINT16_C(48568),
        UINT16_C( 3042), UINT16_C(22058), UINT16_C(64905), UINT16_C(30964), UINT16_C(17007), UINT16_C(53799), UINT16_C( 9355), UINT16_C(35347) },
      { UINT16_C(58876), UINT16_C(25034), UINT16_C(24513), UINT16_C( 5805), UINT16_C(14615), UINT16_C(54896), UINT16_C(10751), UINT16_C(57747),
        UINT16_C(48692), UINT16_C(48440), UINT16_C(11451), UINT16_C(10806), UINT16_C(23918), UINT16_C(63996), UINT16_C( 4225), UINT16_C(32387) },
      { UINT16_C(58876), UINT16_C(46815), UINT16_C(56420), UINT16_C(19605), UINT16_C(47188), UINT16_C(54896), UINT16_C(50879), UINT16_C(57747),
        UINT16_C(48692), UINT16_C( 7422), UINT16_C(27986), UINT16_C(32372), UINT16_C(40402), UINT16_C(29423), UINT16_C( 9355), UINT16_C(18786) } },
    { { UINT16_C(19957), UINT16_C(46815), UINT16_C(36013), UINT16_C(50380), UINT16_C(15813), UINT16_C(50331), UINT16_C(11878), UINT16_C(39589),
        UINT16_C(56812), UINT16_C(42839), UINT16_C(36105), UINT16_C(30674), UINT16_C(52970), UINT16_C(27760), UINT16_C(62430), UINT16_C(54250) },
      UINT16_C(51521),
      { UINT16_C(61066), UINT16_C(22101), UINT16_C( 6834), UINT16_C(19859), UINT16_C(63966), UINT16_C(33660), UINT16_C(26771), UINT16_C(60257),
        UINT16_C(27152), UINT16_C(57976), UINT16_C(25570), UINT16_C(21168), UINT16_C(36815), UINT16_C(47430), UINT16_C(34658), UINT16_C(60546) },
      { UINT16_C(55157), UINT16_C(10051), UINT16_C(55025), UINT16_C(53109), UINT16_C(61904), UINT16_C(25426), UINT16_C(45913), UINT16_C(26958),
        UINT16_C(50974), UINT16_C(   75), UINT16_C(64554), UINT16_C(63826), UINT16_C(39051), UINT16_C(60850), UINT16_C(13343), UINT16_C(38106) },
      { UINT16_C(61066), UINT16_C(46815), UINT16_C(36013), UINT16_C(50380), UINT16_C(15813), UINT16_C(50331), UINT16_C(45913), UINT16_C(39589),
        UINT16_C(50974), UINT16_C(42839), UINT16_C(36105), UINT16_C(63826), UINT16_C(52970), UINT16_C(27760), UINT16_C(34658), UINT16_C(60546) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_epu16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_epu16");
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
    easysimd__m256i r = easysimd_mm256_mask_max_epu16(src, k, a, b);

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
test_easysimd_mm256_maskz_max_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const uint16_t a[16];
    const uint16_t b[16];
    const uint16_t r[16];
  } test_vec[] = {
    { UINT16_C( 9167),
      { UINT16_C(55246), UINT16_C(58099), UINT16_C(26684), UINT16_C(42656), UINT16_C(62730), UINT16_C(23003), UINT16_C(31337), UINT16_C(33568),
        UINT16_C(15848), UINT16_C(10979), UINT16_C(43329), UINT16_C(44932), UINT16_C(14929), UINT16_C(42581), UINT16_C( 9624), UINT16_C(26313) },
      { UINT16_C(48636), UINT16_C(14408), UINT16_C(59429), UINT16_C(12254), UINT16_C(47581), UINT16_C(18057), UINT16_C(43315), UINT16_C( 7113),
        UINT16_C(44518), UINT16_C(10054), UINT16_C(51798), UINT16_C(42967), UINT16_C(11268), UINT16_C(40013), UINT16_C( 5969), UINT16_C(19970) },
      { UINT16_C(55246), UINT16_C(58099), UINT16_C(59429), UINT16_C(42656), UINT16_C(    0), UINT16_C(    0), UINT16_C(43315), UINT16_C(33568),
        UINT16_C(44518), UINT16_C(10979), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(42581), UINT16_C(    0), UINT16_C(    0) } },
    { UINT16_C(19156),
      { UINT16_C(63878), UINT16_C(25906), UINT16_C( 3881), UINT16_C(45598), UINT16_C(21077), UINT16_C( 8027), UINT16_C(17005), UINT16_C(46028),
        UINT16_C( 8809), UINT16_C(16510), UINT16_C(33482), UINT16_C( 5997), UINT16_C(48671), UINT16_C( 8494), UINT16_C(  524), UINT16_C(37740) },
      { UINT16_C(40700), UINT16_C( 9720), UINT16_C( 5806), UINT16_C(  983), UINT16_C(12904), UINT16_C(54818), UINT16_C(61044), UINT16_C(56969),
        UINT16_C( 1809), UINT16_C(56094), UINT16_C(35722), UINT16_C(43506), UINT16_C( 8522), UINT16_C(22218), UINT16_C(13859), UINT16_C( 8169) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C( 5806), UINT16_C(    0), UINT16_C(21077), UINT16_C(    0), UINT16_C(61044), UINT16_C(56969),
        UINT16_C(    0), UINT16_C(56094), UINT16_C(    0), UINT16_C(43506), UINT16_C(    0), UINT16_C(    0), UINT16_C(13859), UINT16_C(    0) } },
    { UINT16_C(57813),
      { UINT16_C(33604), UINT16_C( 7160), UINT16_C(24710), UINT16_C(43342), UINT16_C(49718), UINT16_C(49303), UINT16_C(43168), UINT16_C(49095),
        UINT16_C(20867), UINT16_C(30282), UINT16_C(38138), UINT16_C(50583), UINT16_C(47851), UINT16_C(54523), UINT16_C(53466), UINT16_C( 7862) },
      { UINT16_C(44627), UINT16_C(55866), UINT16_C(34830), UINT16_C(17795), UINT16_C( 6730), UINT16_C(60165), UINT16_C(52419), UINT16_C(18090),
        UINT16_C(62494), UINT16_C( 6332), UINT16_C(21385), UINT16_C(29917), UINT16_C(55566), UINT16_C(59464), UINT16_C(65193), UINT16_C(64774) },
      { UINT16_C(44627), UINT16_C(    0), UINT16_C(34830), UINT16_C(    0), UINT16_C(49718), UINT16_C(    0), UINT16_C(52419), UINT16_C(49095),
        UINT16_C(62494), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(59464), UINT16_C(65193), UINT16_C(64774) } },
    { UINT16_C(16556),
      { UINT16_C(48087), UINT16_C(23240), UINT16_C( 4864), UINT16_C( 1396), UINT16_C(14334), UINT16_C(43217), UINT16_C(61310), UINT16_C(15004),
        UINT16_C( 9480), UINT16_C(58766), UINT16_C(40089), UINT16_C(58046), UINT16_C(26756), UINT16_C(35552), UINT16_C(36197), UINT16_C(15563) },
      { UINT16_C(37704), UINT16_C(18582), UINT16_C( 2726), UINT16_C(42061), UINT16_C( 7746), UINT16_C(49228), UINT16_C(59662), UINT16_C( 5882),
        UINT16_C(34830), UINT16_C(43259), UINT16_C(47652), UINT16_C(43146), UINT16_C(27170), UINT16_C(34611), UINT16_C(65271), UINT16_C(16323) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C( 4864), UINT16_C(42061), UINT16_C(    0), UINT16_C(49228), UINT16_C(    0), UINT16_C(15004),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(65271), UINT16_C(    0) } },
    { UINT16_C(22929),
      { UINT16_C(14471), UINT16_C(54371), UINT16_C(42460), UINT16_C(10739), UINT16_C(  357), UINT16_C(24594), UINT16_C( 8215), UINT16_C( 4840),
        UINT16_C( 3528), UINT16_C(21196), UINT16_C(61109), UINT16_C(59581), UINT16_C(46197), UINT16_C(14566), UINT16_C(30964), UINT16_C(31633) },
      { UINT16_C(62896), UINT16_C(35920), UINT16_C(17306), UINT16_C(  181), UINT16_C(51012), UINT16_C(23392), UINT16_C(18664), UINT16_C(45165),
        UINT16_C(14933), UINT16_C( 2819), UINT16_C(49192), UINT16_C(40691), UINT16_C(55924), UINT16_C(26838), UINT16_C(26706), UINT16_C(  740) },
      { UINT16_C(62896), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(51012), UINT16_C(    0), UINT16_C(    0), UINT16_C(45165),
        UINT16_C(14933), UINT16_C(    0), UINT16_C(    0), UINT16_C(59581), UINT16_C(55924), UINT16_C(    0), UINT16_C(30964), UINT16_C(    0) } },
    { UINT16_C(13405),
      { UINT16_C(63374), UINT16_C(17527), UINT16_C(48119), UINT16_C(22283), UINT16_C(62230), UINT16_C(33696), UINT16_C(62884), UINT16_C(42941),
        UINT16_C(58880), UINT16_C(62567), UINT16_C(56196), UINT16_C(23246), UINT16_C( 8260), UINT16_C(10434), UINT16_C( 7970), UINT16_C(45148) },
      { UINT16_C(54039), UINT16_C( 3828), UINT16_C(  142), UINT16_C(42086), UINT16_C( 1779), UINT16_C(38695), UINT16_C(58875), UINT16_C(64574),
        UINT16_C(42443), UINT16_C(20464), UINT16_C(48769), UINT16_C(50601), UINT16_C(27870), UINT16_C(  237), UINT16_C(18827), UINT16_C(41648) },
      { UINT16_C(63374), UINT16_C(    0), UINT16_C(48119), UINT16_C(42086), UINT16_C(62230), UINT16_C(    0), UINT16_C(62884), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(56196), UINT16_C(    0), UINT16_C(27870), UINT16_C(10434), UINT16_C(    0), UINT16_C(    0) } },
    { UINT16_C(42268),
      { UINT16_C(43697), UINT16_C( 6053), UINT16_C(38990), UINT16_C(29981), UINT16_C( 6192), UINT16_C(28250), UINT16_C( 9492), UINT16_C( 1044),
        UINT16_C(38260), UINT16_C( 7874), UINT16_C(41050), UINT16_C(18314), UINT16_C( 5536), UINT16_C(20880), UINT16_C(44216), UINT16_C(27126) },
      { UINT16_C(39766), UINT16_C(42112), UINT16_C(40243), UINT16_C(25369), UINT16_C(29877), UINT16_C(51922), UINT16_C(59033), UINT16_C( 3790),
        UINT16_C(37243), UINT16_C(54572), UINT16_C(46641), UINT16_C(53788), UINT16_C(44235), UINT16_C(33571), UINT16_C( 6488), UINT16_C(44780) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(40243), UINT16_C(29981), UINT16_C(29877), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(38260), UINT16_C(    0), UINT16_C(46641), UINT16_C(    0), UINT16_C(    0), UINT16_C(33571), UINT16_C(    0), UINT16_C(44780) } },
    { UINT16_C(27828),
      { UINT16_C(59218), UINT16_C(27401), UINT16_C(48971), UINT16_C( 7647), UINT16_C(31113), UINT16_C(22275), UINT16_C(32391), UINT16_C(46056),
        UINT16_C( 6739), UINT16_C(28521), UINT16_C(13548), UINT16_C( 3867), UINT16_C(29624), UINT16_C(42024), UINT16_C(56353), UINT16_C(29457) },
      { UINT16_C( 6851), UINT16_C( 3806), UINT16_C(48857), UINT16_C(25131), UINT16_C(11831), UINT16_C(48826), UINT16_C(41644), UINT16_C(65393),
        UINT16_C(55996), UINT16_C(43118), UINT16_C(35086), UINT16_C(50871), UINT16_C(57340), UINT16_C( 7531), UINT16_C(31931), UINT16_C(32656) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(48971), UINT16_C(    0), UINT16_C(31113), UINT16_C(48826), UINT16_C(    0), UINT16_C(65393),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(35086), UINT16_C(50871), UINT16_C(    0), UINT16_C(42024), UINT16_C(56353), UINT16_C(    0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_epu16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_epu16");
    easysimd_test_x86_assert_equal_u16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_u16x16();
    easysimd__m256i b = easysimd_test_x86_random_u16x16();
    easysimd__m256i r = easysimd_mm256_maskz_max_epu16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint32_t src[8];
    const uint8_t k;
    const uint32_t a[8];
    const uint32_t b[8];
    const uint32_t r[8];
  } test_vec[] = {
    { { UINT32_C(4240186635), UINT32_C(3284873715), UINT32_C(2066160930), UINT32_C(4024792529), UINT32_C(1726951484), UINT32_C(3076473132), UINT32_C(4188344794), UINT32_C(1351515973) },
      UINT8_C(156),
      { UINT32_C(2072988746), UINT32_C( 899502871), UINT32_C(4010154106), UINT32_C( 774698493), UINT32_C( 626692836), UINT32_C(  67048178), UINT32_C( 910817719), UINT32_C(3520240007) },
      { UINT32_C(4249641446), UINT32_C( 775088564), UINT32_C(4280170497), UINT32_C( 288180781), UINT32_C(3459745756), UINT32_C(1355953817), UINT32_C(3062242095), UINT32_C(2592561332) },
      { UINT32_C(4240186635), UINT32_C(3284873715), UINT32_C(4280170497), UINT32_C( 774698493), UINT32_C(3459745756), UINT32_C(3076473132), UINT32_C(4188344794), UINT32_C(3520240007) } },
    { { UINT32_C(1855509434), UINT32_C(3198012092), UINT32_C( 817740547), UINT32_C(3779258885), UINT32_C( 196114801), UINT32_C(3747316399), UINT32_C(1368777373), UINT32_C(4109114682) },
      UINT8_C(240),
      { UINT32_C(1319986052), UINT32_C(3142675200), UINT32_C( 314606120), UINT32_C(1032036804), UINT32_C(3555495505), UINT32_C(3429944298), UINT32_C(2114372193), UINT32_C( 846134190) },
      { UINT32_C(1585453918), UINT32_C(2920927878), UINT32_C( 415291732), UINT32_C(3428140154), UINT32_C(3164553682), UINT32_C(1854410765), UINT32_C(2162986962), UINT32_C(3904002698) },
      { UINT32_C(1855509434), UINT32_C(3198012092), UINT32_C( 817740547), UINT32_C(3779258885), UINT32_C(3555495505), UINT32_C(3429944298), UINT32_C(2162986962), UINT32_C(3904002698) } },
    { { UINT32_C(4232458870), UINT32_C(1487625988), UINT32_C(2993711928), UINT32_C(2189346223), UINT32_C( 339615239), UINT32_C(   8570670), UINT32_C(3766513238), UINT32_C(1053307592) },
      UINT8_C(100),
      { UINT32_C(1835612942), UINT32_C(1369817574), UINT32_C(4144060210), UINT32_C(4110320598), UINT32_C(2283934401), UINT32_C(  48112276), UINT32_C(3570122402), UINT32_C(2486700422) },
      { UINT32_C( 704749892), UINT32_C(2507974243), UINT32_C(3565977086), UINT32_C(3251145472), UINT32_C( 843770525), UINT32_C(2922653708), UINT32_C(1837301735), UINT32_C(1292024329) },
      { UINT32_C(4232458870), UINT32_C(1487625988), UINT32_C(4144060210), UINT32_C(2189346223), UINT32_C( 339615239), UINT32_C(2922653708), UINT32_C(3570122402), UINT32_C(1053307592) } },
    { { UINT32_C(3212313436), UINT32_C(2824139946), UINT32_C(1904009329), UINT32_C( 154289259), UINT32_C( 976976942), UINT32_C(2364043173), UINT32_C(2029611631), UINT32_C(2160458532) },
         UINT8_MAX,
      { UINT32_C( 833175357), UINT32_C(1956794771), UINT32_C( 299832269), UINT32_C(3258968134), UINT32_C(2473031971), UINT32_C(3405968225), UINT32_C(3908008685), UINT32_C(2112319551) },
      { UINT32_C(1135513775), UINT32_C(2931249633), UINT32_C(2864682596), UINT32_C(2725117567), UINT32_C(3627406455), UINT32_C(3047372744), UINT32_C(4053636017), UINT32_C(2993587459) },
      { UINT32_C(1135513775), UINT32_C(2931249633), UINT32_C(2864682596), UINT32_C(3258968134), UINT32_C(3627406455), UINT32_C(3405968225), UINT32_C(4053636017), UINT32_C(2993587459) } },
    { { UINT32_C(4160035861), UINT32_C(3534072941), UINT32_C(3262932291), UINT32_C(3680823651), UINT32_C(2259917502), UINT32_C(2201704401), UINT32_C(3983857898), UINT32_C(1939858013) },
      UINT8_C(254),
      { UINT32_C(1097624213), UINT32_C(1954823695), UINT32_C(2765637306), UINT32_C(1164096427), UINT32_C(3172395110), UINT32_C(4222064931), UINT32_C(4015625229), UINT32_C(3387870260) },
      { UINT32_C(1158306358), UINT32_C(1387958168), UINT32_C(2163643093), UINT32_C(2881837125), UINT32_C(1667882048), UINT32_C(2204045429), UINT32_C(3648174245), UINT32_C(3131203716) },
      { UINT32_C(4160035861), UINT32_C(1954823695), UINT32_C(2765637306), UINT32_C(2881837125), UINT32_C(3172395110), UINT32_C(4222064931), UINT32_C(4015625229), UINT32_C(3387870260) } },
    { { UINT32_C(1392487610), UINT32_C( 296073531), UINT32_C(2425461579), UINT32_C( 876369908), UINT32_C(2828576051), UINT32_C(1512830901), UINT32_C( 859020975), UINT32_C(3119371774) },
      UINT8_C(130),
      { UINT32_C(2797407212), UINT32_C(1290915504), UINT32_C(3074458208), UINT32_C(1676309694), UINT32_C(  51941900), UINT32_C(1555198910), UINT32_C(2086331814), UINT32_C(3221099474) },
      { UINT32_C(3479550751), UINT32_C(3927660170), UINT32_C(2527157208), UINT32_C(3707341776), UINT32_C(3688895005), UINT32_C( 725062277), UINT32_C(1252495992), UINT32_C(3288966565) },
      { UINT32_C(1392487610), UINT32_C(3927660170), UINT32_C(2425461579), UINT32_C( 876369908), UINT32_C(2828576051), UINT32_C(1512830901), UINT32_C( 859020975), UINT32_C(3288966565) } },
    { { UINT32_C(3952373345), UINT32_C(2648027077), UINT32_C(3677648395), UINT32_C( 515321089), UINT32_C(3304757055), UINT32_C(2733650218), UINT32_C(1777113027), UINT32_C(2653812285) },
      UINT8_C(100),
      { UINT32_C(1898547649), UINT32_C(3564947294), UINT32_C( 685070331), UINT32_C(2791895822), UINT32_C( 533736685), UINT32_C(3034739228), UINT32_C(1425099614), UINT32_C( 985239417) },
      { UINT32_C(1990976024), UINT32_C(2756323241), UINT32_C(2379030398), UINT32_C(   3355922), UINT32_C(2115961697), UINT32_C(3543269749), UINT32_C(3324519245), UINT32_C(3389055410) },
      { UINT32_C(3952373345), UINT32_C(2648027077), UINT32_C(2379030398), UINT32_C( 515321089), UINT32_C(3304757055), UINT32_C(3543269749), UINT32_C(3324519245), UINT32_C(2653812285) } },
    { { UINT32_C(1832955075), UINT32_C(1376881363), UINT32_C(3152010921), UINT32_C(1975194131), UINT32_C(2331236885), UINT32_C( 693970396), UINT32_C(4210001224), UINT32_C( 717549414) },
      UINT8_C(155),
      { UINT32_C(2389677828), UINT32_C(2268578216), UINT32_C(2996499104), UINT32_C(2294747054), UINT32_C( 660885762), UINT32_C( 879725998), UINT32_C(1822058876), UINT32_C( 822592557) },
      { UINT32_C(  79656539), UINT32_C(3616273975), UINT32_C(2542347753), UINT32_C( 924799029), UINT32_C(1348437153), UINT32_C(2391068177), UINT32_C(1710890552), UINT32_C(1050083811) },
      { UINT32_C(2389677828), UINT32_C(3616273975), UINT32_C(3152010921), UINT32_C(2294747054), UINT32_C(1348437153), UINT32_C( 693970396), UINT32_C(4210001224), UINT32_C(1050083811) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_epu32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_epu32");
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
    easysimd__m256i r = easysimd_mm256_mask_max_epu32(src, k, a, b);

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
test_easysimd_mm256_maskz_max_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint32_t a[8];
    const uint32_t b[8];
    const uint32_t r[8];
  } test_vec[] = {
    { UINT8_C(150),
      { UINT32_C( 762350959), UINT32_C(3882144441), UINT32_C( 798237324), UINT32_C(1844155283), UINT32_C(2340131842), UINT32_C( 730350155), UINT32_C( 702981549), UINT32_C(2780783926) },
      { UINT32_C(2899455987), UINT32_C(2392012290), UINT32_C(3955042136), UINT32_C(3176704443), UINT32_C(2286474045), UINT32_C(3266564117), UINT32_C(2901121654), UINT32_C(4065438719) },
      { UINT32_C(         0), UINT32_C(3882144441), UINT32_C(3955042136), UINT32_C(         0), UINT32_C(2340131842), UINT32_C(         0), UINT32_C(         0), UINT32_C(4065438719) } },
    { UINT8_C(218),
      { UINT32_C(1507630627), UINT32_C(1504799538), UINT32_C(3507788840), UINT32_C(3339637236), UINT32_C(3940390682), UINT32_C(3831537482), UINT32_C( 920915083), UINT32_C(2148587101) },
      { UINT32_C(2816077173), UINT32_C(2147519064), UINT32_C( 441586982), UINT32_C(  14835942), UINT32_C(1122746359), UINT32_C(3911600990), UINT32_C(3038710360), UINT32_C(1446392033) },
      { UINT32_C(         0), UINT32_C(2147519064), UINT32_C(         0), UINT32_C(3339637236), UINT32_C(3940390682), UINT32_C(         0), UINT32_C(3038710360), UINT32_C(2148587101) } },
    { UINT8_C( 29),
      { UINT32_C(2591423759), UINT32_C( 314635773), UINT32_C(2834946887), UINT32_C(2090858941), UINT32_C( 819651044), UINT32_C( 310952968), UINT32_C( 334708195), UINT32_C(2200979827) },
      { UINT32_C(1142793542), UINT32_C(3797343643), UINT32_C(1971998648), UINT32_C( 770779721), UINT32_C( 324914187), UINT32_C(1931928976), UINT32_C(2525436195), UINT32_C(2837034851) },
      { UINT32_C(2591423759), UINT32_C(         0), UINT32_C(2834946887), UINT32_C(2090858941), UINT32_C( 819651044), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C( 93),
      { UINT32_C( 351857974), UINT32_C(2479675972), UINT32_C(2413576805), UINT32_C(4288285235), UINT32_C(1301261927), UINT32_C(3983541204), UINT32_C(1095763594), UINT32_C(1453259296) },
      { UINT32_C( 728405735), UINT32_C(3602855793), UINT32_C(2892340089), UINT32_C( 229441445), UINT32_C(2170174381), UINT32_C(3362769470), UINT32_C(4027236304), UINT32_C(2705828025) },
      { UINT32_C( 728405735), UINT32_C(         0), UINT32_C(2892340089), UINT32_C(4288285235), UINT32_C(2170174381), UINT32_C(         0), UINT32_C(4027236304), UINT32_C(         0) } },
    { UINT8_C( 63),
      { UINT32_C(3903900849), UINT32_C( 660702859), UINT32_C(3939241707), UINT32_C(4120435130), UINT32_C(4248049971), UINT32_C(1221524616), UINT32_C(2936126982), UINT32_C(3085869573) },
      { UINT32_C(4204764783), UINT32_C( 253821220), UINT32_C(3388599823), UINT32_C(4223570375), UINT32_C( 871953067), UINT32_C(4118529775), UINT32_C(2326035845), UINT32_C(2386661919) },
      { UINT32_C(4204764783), UINT32_C( 660702859), UINT32_C(3939241707), UINT32_C(4223570375), UINT32_C(4248049971), UINT32_C(4118529775), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C( 48),
      { UINT32_C(3797191137), UINT32_C(2565956522), UINT32_C(3999316573), UINT32_C(1805212536), UINT32_C( 442158419), UINT32_C(3315552072), UINT32_C(2263165428), UINT32_C(1287091051) },
      { UINT32_C(2804812796), UINT32_C(3426688879), UINT32_C(1371185113), UINT32_C(1320965370), UINT32_C(1768429089), UINT32_C(1513031526), UINT32_C(2615153712), UINT32_C(2229770119) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(1768429089), UINT32_C(3315552072), UINT32_C(         0), UINT32_C(         0) } },
    { UINT8_C(163),
      { UINT32_C( 873605909), UINT32_C( 168681066), UINT32_C(3959709592), UINT32_C( 839733787), UINT32_C(3247994810), UINT32_C(3136484006), UINT32_C(1782681042), UINT32_C(2316158325) },
      { UINT32_C(1539252208), UINT32_C(2506476797), UINT32_C(1199728939), UINT32_C(1987678140), UINT32_C(2889355526), UINT32_C(3580242435), UINT32_C( 742369463), UINT32_C(1589070957) },
      { UINT32_C(1539252208), UINT32_C(2506476797), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(3580242435), UINT32_C(         0), UINT32_C(2316158325) } },
    { UINT8_C(107),
      { UINT32_C(1114159478), UINT32_C(2272198174), UINT32_C( 256095616), UINT32_C(1058454062), UINT32_C( 474137330), UINT32_C(3520272169), UINT32_C(2738749527), UINT32_C( 755997879) },
      { UINT32_C(1936684885), UINT32_C(4126924149), UINT32_C(3221569170), UINT32_C(3942587384), UINT32_C( 101073373), UINT32_C(2966936153), UINT32_C(2438141658), UINT32_C( 146694834) },
      { UINT32_C(1936684885), UINT32_C(4126924149), UINT32_C(         0), UINT32_C(3942587384), UINT32_C(         0), UINT32_C(3520272169), UINT32_C(2738749527), UINT32_C(         0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_epu32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_epu32");
    easysimd_test_x86_assert_equal_u32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u32x8();
    easysimd__m256i b = easysimd_test_x86_random_u32x8();
    easysimd__m256i r = easysimd_mm256_maskz_max_epu32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_max_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t a[4];
    const uint64_t b[4];
    const uint64_t r[4];
  } test_vec[] = {
    { { UINT64_C(15084256152724494727), UINT64_C( 3869690723534099100), UINT64_C(10895151074823079004), UINT64_C( 2925238184451248369) },
      { UINT64_C( 7223654816377781601), UINT64_C( 7120188840207916901), UINT64_C( 2758185617720924803), UINT64_C(15124232624513083419) },
      { UINT64_C(15084256152724494727), UINT64_C( 7120188840207916901), UINT64_C(10895151074823079004), UINT64_C(15124232624513083419) } },
    { { UINT64_C( 1872229384199283150), UINT64_C(14413350893056640669), UINT64_C(  339952406290292316), UINT64_C( 1218686873187201192) },
      { UINT64_C( 4941472052061612230), UINT64_C( 2735635267710306025), UINT64_C(15472518042175827659), UINT64_C(16640474205064690333) },
      { UINT64_C( 4941472052061612230), UINT64_C(14413350893056640669), UINT64_C(15472518042175827659), UINT64_C(16640474205064690333) } },
    { { UINT64_C( 2444034271948609023), UINT64_C( 9955283662999114042), UINT64_C(16769426217316114099), UINT64_C( 9112951926223710066) },
      { UINT64_C(  478242256950974538), UINT64_C(16945129187902616287), UINT64_C(15150818745314535439), UINT64_C(  588242838314223916) },
      { UINT64_C( 2444034271948609023), UINT64_C(16945129187902616287), UINT64_C(16769426217316114099), UINT64_C( 9112951926223710066) } },
    { { UINT64_C( 4216827596619074995), UINT64_C(15006841408476253985), UINT64_C(14023203834129158250), UINT64_C(10177175121877139318) },
      { UINT64_C( 1691544721354967011), UINT64_C(13740961115224887841), UINT64_C(13770921658045488323), UINT64_C(  255349622639397155) },
      { UINT64_C( 4216827596619074995), UINT64_C(15006841408476253985), UINT64_C(14023203834129158250), UINT64_C(10177175121877139318) } },
    { { UINT64_C( 5405651245795023095), UINT64_C(16870446985573593869), UINT64_C(17204418852046481338), UINT64_C(15784123714841130473) },
      { UINT64_C( 1622927580749909520), UINT64_C( 5282887905549201728), UINT64_C(15922211444570756157), UINT64_C( 7747302259496969155) },
      { UINT64_C( 5405651245795023095), UINT64_C(16870446985573593869), UINT64_C(17204418852046481338), UINT64_C(15784123714841130473) } },
    { { UINT64_C( 6368136156921809984), UINT64_C(12444870112236787932), UINT64_C(15236710052077502883), UINT64_C(10040493120790681999) },
      { UINT64_C(  197092022209084235), UINT64_C( 3325946303001492238), UINT64_C( 1659357145361407268), UINT64_C(14801177494413280145) },
      { UINT64_C( 6368136156921809984), UINT64_C(12444870112236787932), UINT64_C(15236710052077502883), UINT64_C(14801177494413280145) } },
    { { UINT64_C( 9330311033726713171), UINT64_C( 3395755951800641895), UINT64_C(12656077152494806275), UINT64_C( 5854103360491988462) },
      { UINT64_C(13917889053560827185), UINT64_C( 5650560998255567919), UINT64_C( 2775762428420083128), UINT64_C( 3851091171058520099) },
      { UINT64_C(13917889053560827185), UINT64_C( 5650560998255567919), UINT64_C(12656077152494806275), UINT64_C( 5854103360491988462) } },
    { { UINT64_C(12780370778996161601), UINT64_C(11293996400845197347), UINT64_C( 3927973487531169315), UINT64_C( 3260827046344333319) },
      { UINT64_C( 4411034052086733316), UINT64_C(11219706337194353460), UINT64_C( 2614191274457507581), UINT64_C( 7758136893588414097) },
      { UINT64_C(12780370778996161601), UINT64_C(11293996400845197347), UINT64_C( 3927973487531169315), UINT64_C( 7758136893588414097) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_max_epu64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_max_epu64");
    easysimd_test_x86_assert_equal_u64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_test_x86_random_u64x4();
    easysimd__m256i r = easysimd_mm256_max_epu64(a, b);

    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint64_t src[4];
    const uint8_t k;
    const uint64_t a[4];
    const uint64_t b[4];
    const uint64_t r[4];
  } test_vec[] = {
    { { UINT64_C( 3929334933440583288), UINT64_C(  171115212903878899), UINT64_C(14200085528364367472), UINT64_C( 9445949525071629045) },
      UINT8_C( 24),
      { UINT64_C(14562842093889598297), UINT64_C( 6900973274451886440), UINT64_C( 7011173708604393344), UINT64_C(10643677428899914972) },
      { UINT64_C( 2136909326273092131), UINT64_C( 9301401085899100949), UINT64_C(10319475950971519614), UINT64_C(13048866089246499400) },
      { UINT64_C( 3929334933440583288), UINT64_C(  171115212903878899), UINT64_C(14200085528364367472), UINT64_C(13048866089246499400) } },
    { { UINT64_C(13389266171217498365), UINT64_C(17838622304505905677), UINT64_C(15404846801702643846), UINT64_C(15486875097835828900) },
      UINT8_C( 63),
      { UINT64_C(11152773432119363711), UINT64_C(14633895461905486859), UINT64_C( 1188155465722777814), UINT64_C( 2248069862019150981) },
      { UINT64_C(13563327713524585799), UINT64_C( 3432132524724997746), UINT64_C(12383636902535231922), UINT64_C(14697727748145660984) },
      { UINT64_C(13563327713524585799), UINT64_C(14633895461905486859), UINT64_C(12383636902535231922), UINT64_C(14697727748145660984) } },
    { { UINT64_C(16309123524309275111), UINT64_C(13077143813707309555), UINT64_C( 4550144185700745811), UINT64_C( 8966404717518832672) },
      UINT8_C(226),
      { UINT64_C(  825059689234503130), UINT64_C(13589826530624585707), UINT64_C(18068445238153908547), UINT64_C(12469918519360775195) },
      { UINT64_C( 9193711145367396385), UINT64_C(12345875553520493662), UINT64_C( 4007651832243034221), UINT64_C( 6696415294772162158) },
      { UINT64_C(16309123524309275111), UINT64_C(13589826530624585707), UINT64_C( 4550144185700745811), UINT64_C( 8966404717518832672) } },
    { { UINT64_C( 1865755484676955090), UINT64_C( 8229490510665984035), UINT64_C(17974346093585399839), UINT64_C( 1835825183533487587) },
      UINT8_C( 68),
      { UINT64_C(  793785152084803170), UINT64_C( 4967051220835405513), UINT64_C( 8011841727642872216), UINT64_C( 3057891130641632514) },
      { UINT64_C( 5673426697455365454), UINT64_C(16619985005650496670), UINT64_C( 4485411767198118965), UINT64_C( 1414931575588124080) },
      { UINT64_C( 1865755484676955090), UINT64_C( 8229490510665984035), UINT64_C( 8011841727642872216), UINT64_C( 1835825183533487587) } },
    { { UINT64_C( 6564138009531559493), UINT64_C( 8611554575217669709), UINT64_C(14362688495808853464), UINT64_C(15588051857325407099) },
      UINT8_C(144),
      { UINT64_C(17430243080993088279), UINT64_C(13232325584126967910), UINT64_C(14119231269880379121), UINT64_C(11188962946771726805) },
      { UINT64_C(11852936826710103505), UINT64_C( 5570970728954507927), UINT64_C( 9984961641319768006), UINT64_C(16647425276980160487) },
      { UINT64_C( 6564138009531559493), UINT64_C( 8611554575217669709), UINT64_C(14362688495808853464), UINT64_C(15588051857325407099) } },
    { { UINT64_C(16789968416582941214), UINT64_C( 5066500652549697445), UINT64_C(12624054611223057394), UINT64_C( 1984718533562168206) },
      UINT8_C( 92),
      { UINT64_C(16760311463782742016), UINT64_C(18149136963283228953), UINT64_C( 3096351971129722308), UINT64_C( 2885754203759156366) },
      { UINT64_C( 7574125830366542522), UINT64_C(17811792821281457228), UINT64_C(17769333209358857400), UINT64_C(13639066818217230583) },
      { UINT64_C(16789968416582941214), UINT64_C( 5066500652549697445), UINT64_C(17769333209358857400), UINT64_C(13639066818217230583) } },
    { { UINT64_C( 7000577433307996781), UINT64_C( 7619184231630405009), UINT64_C( 8005836048415858363), UINT64_C(16352804942040187185) },
      UINT8_C( 55),
      { UINT64_C( 4935574415347589345), UINT64_C( 2145788196975625028), UINT64_C(14256150137990933046), UINT64_C(15969796149078526867) },
      { UINT64_C(16125219398079933884), UINT64_C( 5884148744540896652), UINT64_C(15449264556344884636), UINT64_C( 8697822782517387483) },
      { UINT64_C(16125219398079933884), UINT64_C( 5884148744540896652), UINT64_C(15449264556344884636), UINT64_C(16352804942040187185) } },
    { { UINT64_C( 6073038987549507532), UINT64_C( 9290231022289367681), UINT64_C(10438649822404797892), UINT64_C(14437727848630852648) },
      UINT8_C(155),
      { UINT64_C(12150350562619240281), UINT64_C(  928318934842358422), UINT64_C(16347876185801353518), UINT64_C( 8895353778767512762) },
      { UINT64_C( 2590756765643101653), UINT64_C(13593310141356277873), UINT64_C( 8337378668711143180), UINT64_C(  914999520134795541) },
      { UINT64_C(12150350562619240281), UINT64_C(13593310141356277873), UINT64_C(10438649822404797892), UINT64_C( 8895353778767512762) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_epu64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_epu64");
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
    easysimd__m256i r = easysimd_mm256_mask_max_epu64(src, k, a, b);

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
test_easysimd_mm256_maskz_max_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const uint64_t a[4];
    const uint64_t b[4];
    const uint64_t r[4];
  } test_vec[] = {
    { UINT8_C(218),
      { UINT64_C(13086692465506745133), UINT64_C(11475902327398686282), UINT64_C( 3914375883560339870), UINT64_C(14931950239931294715) },
      { UINT64_C( 1628182987174217580), UINT64_C( 8994461263073068501), UINT64_C( 7880814953567144455), UINT64_C(11861077586099259178) },
      { UINT64_C(                   0), UINT64_C(11475902327398686282), UINT64_C(                   0), UINT64_C(14931950239931294715) } },
    { UINT8_C(139),
      { UINT64_C( 7332544226700789621), UINT64_C(10497537546161157609), UINT64_C( 8873340538661885619), UINT64_C(   56954200876380220) },
      { UINT64_C( 8747027938570740447), UINT64_C( 1728484387204474532), UINT64_C(12345552332540477349), UINT64_C( 4517872798834296614) },
      { UINT64_C( 8747027938570740447), UINT64_C(10497537546161157609), UINT64_C(                   0), UINT64_C( 4517872798834296614) } },
    { UINT8_C(215),
      { UINT64_C( 5112396605645339807), UINT64_C( 6374777488591707394), UINT64_C(12407544126076637395), UINT64_C( 3813222409839567454) },
      { UINT64_C( 3921638242377683846), UINT64_C(12431200338366109394), UINT64_C( 2170987635147099343), UINT64_C(18127021989573045447) },
      { UINT64_C( 5112396605645339807), UINT64_C(12431200338366109394), UINT64_C(12407544126076637395), UINT64_C(                   0) } },
    { UINT8_C(113),
      { UINT64_C(11843019571036758290), UINT64_C( 6282565505232176866), UINT64_C(17333006710068454622), UINT64_C( 9941521868427552616) },
      { UINT64_C( 6463909524298374275), UINT64_C( 3986506292372582897), UINT64_C(  523410000572017038), UINT64_C(10263004312461224301) },
      { UINT64_C(11843019571036758290), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C( 32),
      { UINT64_C( 9573228201121390206), UINT64_C(10530565092756281003), UINT64_C( 7206666202757822224), UINT64_C(15233985753008564165) },
      { UINT64_C(11201182284277088411), UINT64_C( 8441809921312378989), UINT64_C( 6190776651661138124), UINT64_C(17445827123710155253) },
      { UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C(  7),
      { UINT64_C(11807755784622235865), UINT64_C(17502533627643720388), UINT64_C(12721499438842080392), UINT64_C(14593625358625854583) },
      { UINT64_C(15478615806819598041), UINT64_C(15859097695188670345), UINT64_C(16737625204459739457), UINT64_C( 2273452800080301777) },
      { UINT64_C(15478615806819598041), UINT64_C(17502533627643720388), UINT64_C(16737625204459739457), UINT64_C(                   0) } },
    { UINT8_C(107),
      { UINT64_C(17394962361080368070), UINT64_C( 4132089629581347394), UINT64_C( 9611406421621731219), UINT64_C( 8598193787934287098) },
      { UINT64_C( 7152486009502289718), UINT64_C( 4404752062743218246), UINT64_C( 6448643805700541378), UINT64_C( 7745227310806157652) },
      { UINT64_C(17394962361080368070), UINT64_C( 4404752062743218246), UINT64_C(                   0), UINT64_C( 8598193787934287098) } },
    { UINT8_C( 98),
      { UINT64_C( 9715645376626998481), UINT64_C(12102981836845661026), UINT64_C(13240124878050750266), UINT64_C( 1267295174622167968) },
      { UINT64_C( 7248387478487177746), UINT64_C( 8048089987773155117), UINT64_C(13160528656530519678), UINT64_C(17846218197027163738) },
      { UINT64_C(                   0), UINT64_C(12102981836845661026), UINT64_C(                   0), UINT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_epu64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_epu64");
    easysimd_test_x86_assert_equal_u64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_u64x4();
    easysimd__m256i b = easysimd_test_x86_random_u64x4();
    easysimd__m256i r = easysimd_mm256_maskz_max_epu64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_u64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   853.03), EASYSIMD_FLOAT32_C(  -681.76), EASYSIMD_FLOAT32_C(  -784.53), EASYSIMD_FLOAT32_C(   551.65),
        EASYSIMD_FLOAT32_C(    49.27), EASYSIMD_FLOAT32_C(   801.64), EASYSIMD_FLOAT32_C(   603.31), EASYSIMD_FLOAT32_C(   994.23) },
      UINT8_C( 15),
      { EASYSIMD_FLOAT32_C(   607.53), EASYSIMD_FLOAT32_C(   226.23), EASYSIMD_FLOAT32_C(   770.38), EASYSIMD_FLOAT32_C(   398.99),
        EASYSIMD_FLOAT32_C(   996.50), EASYSIMD_FLOAT32_C(  -740.46), EASYSIMD_FLOAT32_C(   845.65), EASYSIMD_FLOAT32_C(  -738.41) },
      { EASYSIMD_FLOAT32_C(  -557.20), EASYSIMD_FLOAT32_C(  -696.43), EASYSIMD_FLOAT32_C(   402.70), EASYSIMD_FLOAT32_C(   136.64),
        EASYSIMD_FLOAT32_C(  -259.49), EASYSIMD_FLOAT32_C(   -93.76), EASYSIMD_FLOAT32_C(   657.92), EASYSIMD_FLOAT32_C(   760.23) },
      { EASYSIMD_FLOAT32_C(   607.53), EASYSIMD_FLOAT32_C(   226.23), EASYSIMD_FLOAT32_C(   770.38), EASYSIMD_FLOAT32_C(   398.99),
        EASYSIMD_FLOAT32_C(    49.27), EASYSIMD_FLOAT32_C(   801.64), EASYSIMD_FLOAT32_C(   603.31), EASYSIMD_FLOAT32_C(   994.23) } },
    { { EASYSIMD_FLOAT32_C(  -888.94), EASYSIMD_FLOAT32_C(  -222.44), EASYSIMD_FLOAT32_C(    99.84), EASYSIMD_FLOAT32_C(   323.08),
        EASYSIMD_FLOAT32_C(  -589.40), EASYSIMD_FLOAT32_C(  -942.70), EASYSIMD_FLOAT32_C(   176.11), EASYSIMD_FLOAT32_C(  -271.16) },
      UINT8_C(189),
      { EASYSIMD_FLOAT32_C(  -272.24), EASYSIMD_FLOAT32_C(   778.11), EASYSIMD_FLOAT32_C(  -925.58), EASYSIMD_FLOAT32_C(  -668.93),
        EASYSIMD_FLOAT32_C(   772.33), EASYSIMD_FLOAT32_C(   885.40), EASYSIMD_FLOAT32_C(   938.60), EASYSIMD_FLOAT32_C(    -1.43) },
      { EASYSIMD_FLOAT32_C(   655.78), EASYSIMD_FLOAT32_C(   337.58), EASYSIMD_FLOAT32_C(    -4.93), EASYSIMD_FLOAT32_C(   915.32),
        EASYSIMD_FLOAT32_C(   183.24), EASYSIMD_FLOAT32_C(   256.67), EASYSIMD_FLOAT32_C(  -641.88), EASYSIMD_FLOAT32_C(   486.80) },
      { EASYSIMD_FLOAT32_C(   655.78), EASYSIMD_FLOAT32_C(  -222.44), EASYSIMD_FLOAT32_C(    -4.93), EASYSIMD_FLOAT32_C(   915.32),
        EASYSIMD_FLOAT32_C(   772.33), EASYSIMD_FLOAT32_C(   885.40), EASYSIMD_FLOAT32_C(   176.11), EASYSIMD_FLOAT32_C(   486.80) } },
    { { EASYSIMD_FLOAT32_C(  -340.63), EASYSIMD_FLOAT32_C(   494.76), EASYSIMD_FLOAT32_C(  -772.69), EASYSIMD_FLOAT32_C(   565.61),
        EASYSIMD_FLOAT32_C(   152.67), EASYSIMD_FLOAT32_C(   987.54), EASYSIMD_FLOAT32_C(   676.67), EASYSIMD_FLOAT32_C(   930.24) },
      UINT8_C(221),
      { EASYSIMD_FLOAT32_C(    -0.25), EASYSIMD_FLOAT32_C(  -659.16), EASYSIMD_FLOAT32_C(   144.69), EASYSIMD_FLOAT32_C(  -824.14),
        EASYSIMD_FLOAT32_C(    69.67), EASYSIMD_FLOAT32_C(   417.46), EASYSIMD_FLOAT32_C(   -96.39), EASYSIMD_FLOAT32_C(  -152.22) },
      { EASYSIMD_FLOAT32_C(   491.88), EASYSIMD_FLOAT32_C(   234.68), EASYSIMD_FLOAT32_C(  -379.88), EASYSIMD_FLOAT32_C(   377.28),
        EASYSIMD_FLOAT32_C(   173.28), EASYSIMD_FLOAT32_C(   618.68), EASYSIMD_FLOAT32_C(    33.05), EASYSIMD_FLOAT32_C(  -489.14) },
      { EASYSIMD_FLOAT32_C(   491.88), EASYSIMD_FLOAT32_C(   494.76), EASYSIMD_FLOAT32_C(   144.69), EASYSIMD_FLOAT32_C(   377.28),
        EASYSIMD_FLOAT32_C(   173.28), EASYSIMD_FLOAT32_C(   987.54), EASYSIMD_FLOAT32_C(    33.05), EASYSIMD_FLOAT32_C(  -152.22) } },
    { { EASYSIMD_FLOAT32_C(  -386.25), EASYSIMD_FLOAT32_C(   -51.63), EASYSIMD_FLOAT32_C(   694.10), EASYSIMD_FLOAT32_C(   870.42),
        EASYSIMD_FLOAT32_C(   306.49), EASYSIMD_FLOAT32_C(   180.90), EASYSIMD_FLOAT32_C(  -470.21), EASYSIMD_FLOAT32_C(  -198.75) },
      UINT8_C(127),
      { EASYSIMD_FLOAT32_C(  -904.60), EASYSIMD_FLOAT32_C(   953.92), EASYSIMD_FLOAT32_C(   395.75), EASYSIMD_FLOAT32_C(   772.07),
        EASYSIMD_FLOAT32_C(   884.16), EASYSIMD_FLOAT32_C(  -516.86), EASYSIMD_FLOAT32_C(  -228.18), EASYSIMD_FLOAT32_C(  -775.00) },
      { EASYSIMD_FLOAT32_C(   627.83), EASYSIMD_FLOAT32_C(   -52.33), EASYSIMD_FLOAT32_C(   294.67), EASYSIMD_FLOAT32_C(    45.29),
        EASYSIMD_FLOAT32_C(   851.28), EASYSIMD_FLOAT32_C(  -857.55), EASYSIMD_FLOAT32_C(  -462.83), EASYSIMD_FLOAT32_C(    85.97) },
      { EASYSIMD_FLOAT32_C(   627.83), EASYSIMD_FLOAT32_C(   953.92), EASYSIMD_FLOAT32_C(   395.75), EASYSIMD_FLOAT32_C(   772.07),
        EASYSIMD_FLOAT32_C(   884.16), EASYSIMD_FLOAT32_C(  -516.86), EASYSIMD_FLOAT32_C(  -228.18), EASYSIMD_FLOAT32_C(  -198.75) } },
    { { EASYSIMD_FLOAT32_C(  -237.43), EASYSIMD_FLOAT32_C(   914.44), EASYSIMD_FLOAT32_C(  -740.75), EASYSIMD_FLOAT32_C(  -618.75),
        EASYSIMD_FLOAT32_C(   -52.50), EASYSIMD_FLOAT32_C(  -229.89), EASYSIMD_FLOAT32_C(    -4.99), EASYSIMD_FLOAT32_C(   895.87) },
      UINT8_C(  3),
      { EASYSIMD_FLOAT32_C(  -134.57), EASYSIMD_FLOAT32_C(   202.36), EASYSIMD_FLOAT32_C(   645.11), EASYSIMD_FLOAT32_C(   395.21),
        EASYSIMD_FLOAT32_C(  -996.39), EASYSIMD_FLOAT32_C(    53.32), EASYSIMD_FLOAT32_C(   490.61), EASYSIMD_FLOAT32_C(   957.54) },
      { EASYSIMD_FLOAT32_C(  -550.93), EASYSIMD_FLOAT32_C(   262.68), EASYSIMD_FLOAT32_C(   841.70), EASYSIMD_FLOAT32_C(   -67.78),
        EASYSIMD_FLOAT32_C(  -965.50), EASYSIMD_FLOAT32_C(  -933.31), EASYSIMD_FLOAT32_C(  -439.96), EASYSIMD_FLOAT32_C(   -17.83) },
      { EASYSIMD_FLOAT32_C(  -134.57), EASYSIMD_FLOAT32_C(   262.68), EASYSIMD_FLOAT32_C(  -740.75), EASYSIMD_FLOAT32_C(  -618.75),
        EASYSIMD_FLOAT32_C(   -52.50), EASYSIMD_FLOAT32_C(  -229.89), EASYSIMD_FLOAT32_C(    -4.99), EASYSIMD_FLOAT32_C(   895.87) } },
    { { EASYSIMD_FLOAT32_C(   361.37), EASYSIMD_FLOAT32_C(   605.33), EASYSIMD_FLOAT32_C(  -166.55), EASYSIMD_FLOAT32_C(   503.82),
        EASYSIMD_FLOAT32_C(  -857.50), EASYSIMD_FLOAT32_C(   919.42), EASYSIMD_FLOAT32_C(  -733.61), EASYSIMD_FLOAT32_C(  -943.06) },
      UINT8_C(246),
      { EASYSIMD_FLOAT32_C(  -352.36), EASYSIMD_FLOAT32_C(     4.44), EASYSIMD_FLOAT32_C(   -51.22), EASYSIMD_FLOAT32_C(   642.64),
        EASYSIMD_FLOAT32_C(   -99.69), EASYSIMD_FLOAT32_C(   412.99), EASYSIMD_FLOAT32_C(  -491.93), EASYSIMD_FLOAT32_C(  -897.33) },
      { EASYSIMD_FLOAT32_C(    58.10), EASYSIMD_FLOAT32_C(   903.28), EASYSIMD_FLOAT32_C(  -893.72), EASYSIMD_FLOAT32_C(  -888.58),
        EASYSIMD_FLOAT32_C(   393.90), EASYSIMD_FLOAT32_C(  -936.18), EASYSIMD_FLOAT32_C(  -439.51), EASYSIMD_FLOAT32_C(  -343.42) },
      { EASYSIMD_FLOAT32_C(   361.37), EASYSIMD_FLOAT32_C(   903.28), EASYSIMD_FLOAT32_C(   -51.22), EASYSIMD_FLOAT32_C(   503.82),
        EASYSIMD_FLOAT32_C(   393.90), EASYSIMD_FLOAT32_C(   412.99), EASYSIMD_FLOAT32_C(  -439.51), EASYSIMD_FLOAT32_C(  -343.42) } },
    { { EASYSIMD_FLOAT32_C(   905.51), EASYSIMD_FLOAT32_C(   492.71), EASYSIMD_FLOAT32_C(  -308.92), EASYSIMD_FLOAT32_C(   972.21),
        EASYSIMD_FLOAT32_C(  -947.25), EASYSIMD_FLOAT32_C(   673.25), EASYSIMD_FLOAT32_C(   333.57), EASYSIMD_FLOAT32_C(   658.08) },
      UINT8_C( 69),
      { EASYSIMD_FLOAT32_C(  -162.61), EASYSIMD_FLOAT32_C(   800.58), EASYSIMD_FLOAT32_C(  -573.88), EASYSIMD_FLOAT32_C(   103.78),
        EASYSIMD_FLOAT32_C(   857.52), EASYSIMD_FLOAT32_C(  -395.21), EASYSIMD_FLOAT32_C(   751.42), EASYSIMD_FLOAT32_C(  -138.04) },
      { EASYSIMD_FLOAT32_C(   553.57), EASYSIMD_FLOAT32_C(   394.06), EASYSIMD_FLOAT32_C(   762.27), EASYSIMD_FLOAT32_C(   -33.44),
        EASYSIMD_FLOAT32_C(   902.13), EASYSIMD_FLOAT32_C(   864.94), EASYSIMD_FLOAT32_C(  -975.34), EASYSIMD_FLOAT32_C(   805.41) },
      { EASYSIMD_FLOAT32_C(   553.57), EASYSIMD_FLOAT32_C(   492.71), EASYSIMD_FLOAT32_C(   762.27), EASYSIMD_FLOAT32_C(   972.21),
        EASYSIMD_FLOAT32_C(  -947.25), EASYSIMD_FLOAT32_C(   673.25), EASYSIMD_FLOAT32_C(   751.42), EASYSIMD_FLOAT32_C(   658.08) } },
    { { EASYSIMD_FLOAT32_C(   971.22), EASYSIMD_FLOAT32_C(  -863.92), EASYSIMD_FLOAT32_C(   199.31), EASYSIMD_FLOAT32_C(  -964.97),
        EASYSIMD_FLOAT32_C(  -303.43), EASYSIMD_FLOAT32_C(   855.88), EASYSIMD_FLOAT32_C(   940.55), EASYSIMD_FLOAT32_C(  -810.72) },
      UINT8_C(216),
      { EASYSIMD_FLOAT32_C(   912.76), EASYSIMD_FLOAT32_C(  -757.97), EASYSIMD_FLOAT32_C(  -779.79), EASYSIMD_FLOAT32_C(   246.33),
        EASYSIMD_FLOAT32_C(   900.11), EASYSIMD_FLOAT32_C(  -273.09), EASYSIMD_FLOAT32_C(  -916.28), EASYSIMD_FLOAT32_C(   700.70) },
      { EASYSIMD_FLOAT32_C(   153.03), EASYSIMD_FLOAT32_C(   187.50), EASYSIMD_FLOAT32_C(   558.22), EASYSIMD_FLOAT32_C(   757.82),
        EASYSIMD_FLOAT32_C(   -61.08), EASYSIMD_FLOAT32_C(  -579.82), EASYSIMD_FLOAT32_C(   311.39), EASYSIMD_FLOAT32_C(  -667.02) },
      { EASYSIMD_FLOAT32_C(   971.22), EASYSIMD_FLOAT32_C(  -863.92), EASYSIMD_FLOAT32_C(   199.31), EASYSIMD_FLOAT32_C(   757.82),
        EASYSIMD_FLOAT32_C(   900.11), EASYSIMD_FLOAT32_C(   855.88), EASYSIMD_FLOAT32_C(   311.39), EASYSIMD_FLOAT32_C(   700.70) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_ps(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_ps");
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
    easysimd__m256 r = easysimd_mm256_mask_max_ps(src, k, a, b);

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
test_easysimd_mm256_maskz_max_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C( 54),
      { EASYSIMD_FLOAT32_C(  -224.38), EASYSIMD_FLOAT32_C(   768.36), EASYSIMD_FLOAT32_C(  -218.31), EASYSIMD_FLOAT32_C(   662.17),
        EASYSIMD_FLOAT32_C(   509.55), EASYSIMD_FLOAT32_C(   327.32), EASYSIMD_FLOAT32_C(  -956.59), EASYSIMD_FLOAT32_C(   329.19) },
      { EASYSIMD_FLOAT32_C(  -566.69), EASYSIMD_FLOAT32_C(  -791.92), EASYSIMD_FLOAT32_C(   140.39), EASYSIMD_FLOAT32_C(  -213.93),
        EASYSIMD_FLOAT32_C(   817.37), EASYSIMD_FLOAT32_C(  -656.04), EASYSIMD_FLOAT32_C(  -732.95), EASYSIMD_FLOAT32_C(   863.18) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   768.36), EASYSIMD_FLOAT32_C(   140.39), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   817.37), EASYSIMD_FLOAT32_C(   327.32), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(219),
      { EASYSIMD_FLOAT32_C(  -416.04), EASYSIMD_FLOAT32_C(   368.02), EASYSIMD_FLOAT32_C(  -814.31), EASYSIMD_FLOAT32_C(   785.30),
        EASYSIMD_FLOAT32_C(   998.83), EASYSIMD_FLOAT32_C(  -693.05), EASYSIMD_FLOAT32_C(   108.20), EASYSIMD_FLOAT32_C(   395.98) },
      { EASYSIMD_FLOAT32_C(  -736.13), EASYSIMD_FLOAT32_C(   -63.74), EASYSIMD_FLOAT32_C(  -172.44), EASYSIMD_FLOAT32_C(    20.53),
        EASYSIMD_FLOAT32_C(    12.70), EASYSIMD_FLOAT32_C(  -423.42), EASYSIMD_FLOAT32_C(   796.15), EASYSIMD_FLOAT32_C(  -218.93) },
      { EASYSIMD_FLOAT32_C(  -416.04), EASYSIMD_FLOAT32_C(   368.02), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   785.30),
        EASYSIMD_FLOAT32_C(   998.83), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   796.15), EASYSIMD_FLOAT32_C(   395.98) } },
    { UINT8_C(224),
      { EASYSIMD_FLOAT32_C(   458.32), EASYSIMD_FLOAT32_C(  -709.38), EASYSIMD_FLOAT32_C(  -314.41), EASYSIMD_FLOAT32_C(   501.73),
        EASYSIMD_FLOAT32_C(   619.81), EASYSIMD_FLOAT32_C(   118.90), EASYSIMD_FLOAT32_C(   709.81), EASYSIMD_FLOAT32_C(  -239.81) },
      { EASYSIMD_FLOAT32_C(   904.97), EASYSIMD_FLOAT32_C(   527.18), EASYSIMD_FLOAT32_C(   104.16), EASYSIMD_FLOAT32_C(  -827.98),
        EASYSIMD_FLOAT32_C(   390.37), EASYSIMD_FLOAT32_C(  -118.87), EASYSIMD_FLOAT32_C(  -244.02), EASYSIMD_FLOAT32_C(  -241.61) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   118.90), EASYSIMD_FLOAT32_C(   709.81), EASYSIMD_FLOAT32_C(  -239.81) } },
    { UINT8_C(104),
      { EASYSIMD_FLOAT32_C(  -458.72), EASYSIMD_FLOAT32_C(  -242.78), EASYSIMD_FLOAT32_C(   373.77), EASYSIMD_FLOAT32_C(   649.48),
        EASYSIMD_FLOAT32_C(  -846.81), EASYSIMD_FLOAT32_C(   637.64), EASYSIMD_FLOAT32_C(  -414.26), EASYSIMD_FLOAT32_C(   -19.25) },
      { EASYSIMD_FLOAT32_C(  -341.83), EASYSIMD_FLOAT32_C(   598.44), EASYSIMD_FLOAT32_C(   557.34), EASYSIMD_FLOAT32_C(  -545.68),
        EASYSIMD_FLOAT32_C(  -620.49), EASYSIMD_FLOAT32_C(   -84.39), EASYSIMD_FLOAT32_C(   912.65), EASYSIMD_FLOAT32_C(  -329.87) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   649.48),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   637.64), EASYSIMD_FLOAT32_C(   912.65), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 99),
      { EASYSIMD_FLOAT32_C(   414.38), EASYSIMD_FLOAT32_C(  -710.07), EASYSIMD_FLOAT32_C(  -279.91), EASYSIMD_FLOAT32_C(   124.19),
        EASYSIMD_FLOAT32_C(    50.12), EASYSIMD_FLOAT32_C(  -374.94), EASYSIMD_FLOAT32_C(  -348.63), EASYSIMD_FLOAT32_C(  -845.72) },
      { EASYSIMD_FLOAT32_C(  -202.92), EASYSIMD_FLOAT32_C(  -958.26), EASYSIMD_FLOAT32_C(    35.41), EASYSIMD_FLOAT32_C(   553.05),
        EASYSIMD_FLOAT32_C(  -199.87), EASYSIMD_FLOAT32_C(  -897.77), EASYSIMD_FLOAT32_C(  -905.67), EASYSIMD_FLOAT32_C(   557.35) },
      { EASYSIMD_FLOAT32_C(   414.38), EASYSIMD_FLOAT32_C(  -710.07), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -374.94), EASYSIMD_FLOAT32_C(  -348.63), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 77),
      { EASYSIMD_FLOAT32_C(   743.81), EASYSIMD_FLOAT32_C(   710.54), EASYSIMD_FLOAT32_C(   113.63), EASYSIMD_FLOAT32_C(  -670.45),
        EASYSIMD_FLOAT32_C(  -308.70), EASYSIMD_FLOAT32_C(   771.81), EASYSIMD_FLOAT32_C(   927.99), EASYSIMD_FLOAT32_C(  -751.37) },
      { EASYSIMD_FLOAT32_C(  -773.87), EASYSIMD_FLOAT32_C(  -692.50), EASYSIMD_FLOAT32_C(   164.24), EASYSIMD_FLOAT32_C(  -861.22),
        EASYSIMD_FLOAT32_C(   -22.38), EASYSIMD_FLOAT32_C(  -234.57), EASYSIMD_FLOAT32_C(   553.15), EASYSIMD_FLOAT32_C(   267.55) },
      { EASYSIMD_FLOAT32_C(   743.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   164.24), EASYSIMD_FLOAT32_C(  -670.45),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   927.99), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(173),
      { EASYSIMD_FLOAT32_C(  -322.66), EASYSIMD_FLOAT32_C(  -682.33), EASYSIMD_FLOAT32_C(  -889.43), EASYSIMD_FLOAT32_C(   328.71),
        EASYSIMD_FLOAT32_C(  -528.04), EASYSIMD_FLOAT32_C(   -92.35), EASYSIMD_FLOAT32_C(   370.46), EASYSIMD_FLOAT32_C(   507.36) },
      { EASYSIMD_FLOAT32_C(  -539.30), EASYSIMD_FLOAT32_C(  -829.41), EASYSIMD_FLOAT32_C(   609.59), EASYSIMD_FLOAT32_C(  -444.97),
        EASYSIMD_FLOAT32_C(   727.93), EASYSIMD_FLOAT32_C(    85.58), EASYSIMD_FLOAT32_C(  -701.16), EASYSIMD_FLOAT32_C(   438.47) },
      { EASYSIMD_FLOAT32_C(  -322.66), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   609.59), EASYSIMD_FLOAT32_C(   328.71),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    85.58), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   507.36) } },
    { UINT8_C(171),
      { EASYSIMD_FLOAT32_C(  -371.61), EASYSIMD_FLOAT32_C(  -870.23), EASYSIMD_FLOAT32_C(   971.02), EASYSIMD_FLOAT32_C(  -443.62),
        EASYSIMD_FLOAT32_C(  -621.60), EASYSIMD_FLOAT32_C(  -802.85), EASYSIMD_FLOAT32_C(  -136.12), EASYSIMD_FLOAT32_C(   542.64) },
      { EASYSIMD_FLOAT32_C(  -664.07), EASYSIMD_FLOAT32_C(   841.50), EASYSIMD_FLOAT32_C(  -691.93), EASYSIMD_FLOAT32_C(   889.08),
        EASYSIMD_FLOAT32_C(   109.05), EASYSIMD_FLOAT32_C(   793.58), EASYSIMD_FLOAT32_C(  -433.58), EASYSIMD_FLOAT32_C(   426.72) },
      { EASYSIMD_FLOAT32_C(  -371.61), EASYSIMD_FLOAT32_C(   841.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   889.08),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   793.58), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   542.64) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_ps(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_max_ps(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_max_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   767.74), EASYSIMD_FLOAT64_C(  -155.77), EASYSIMD_FLOAT64_C(   800.96), EASYSIMD_FLOAT64_C(   674.58) },
      UINT8_C(231),
      { EASYSIMD_FLOAT64_C(   698.80), EASYSIMD_FLOAT64_C(  -637.27), EASYSIMD_FLOAT64_C(   829.76), EASYSIMD_FLOAT64_C(  -823.81) },
      { EASYSIMD_FLOAT64_C(   274.74), EASYSIMD_FLOAT64_C(   769.97), EASYSIMD_FLOAT64_C(  -949.25), EASYSIMD_FLOAT64_C(  -155.65) },
      { EASYSIMD_FLOAT64_C(   698.80), EASYSIMD_FLOAT64_C(   769.97), EASYSIMD_FLOAT64_C(   829.76), EASYSIMD_FLOAT64_C(   674.58) } },
    { { EASYSIMD_FLOAT64_C(  -552.75), EASYSIMD_FLOAT64_C(   609.04), EASYSIMD_FLOAT64_C(    62.51), EASYSIMD_FLOAT64_C(  -484.34) },
      UINT8_C( 86),
      { EASYSIMD_FLOAT64_C(   906.52), EASYSIMD_FLOAT64_C(    61.24), EASYSIMD_FLOAT64_C(   729.25), EASYSIMD_FLOAT64_C(   362.70) },
      { EASYSIMD_FLOAT64_C(  -547.20), EASYSIMD_FLOAT64_C(  -529.34), EASYSIMD_FLOAT64_C(   981.98), EASYSIMD_FLOAT64_C(  -809.42) },
      { EASYSIMD_FLOAT64_C(  -552.75), EASYSIMD_FLOAT64_C(    61.24), EASYSIMD_FLOAT64_C(   981.98), EASYSIMD_FLOAT64_C(  -484.34) } },
    { { EASYSIMD_FLOAT64_C(   -44.09), EASYSIMD_FLOAT64_C(   730.77), EASYSIMD_FLOAT64_C(  -734.46), EASYSIMD_FLOAT64_C(   282.54) },
      UINT8_C(222),
      { EASYSIMD_FLOAT64_C(  -966.72), EASYSIMD_FLOAT64_C(  -873.23), EASYSIMD_FLOAT64_C(   957.33), EASYSIMD_FLOAT64_C(   707.86) },
      { EASYSIMD_FLOAT64_C(  -694.27), EASYSIMD_FLOAT64_C(   656.13), EASYSIMD_FLOAT64_C(  -929.41), EASYSIMD_FLOAT64_C(  -864.52) },
      { EASYSIMD_FLOAT64_C(   -44.09), EASYSIMD_FLOAT64_C(   656.13), EASYSIMD_FLOAT64_C(   957.33), EASYSIMD_FLOAT64_C(   707.86) } },
    { { EASYSIMD_FLOAT64_C(   832.32), EASYSIMD_FLOAT64_C(   345.33), EASYSIMD_FLOAT64_C(   905.45), EASYSIMD_FLOAT64_C(   883.07) },
      UINT8_C(126),
      { EASYSIMD_FLOAT64_C(  -647.29), EASYSIMD_FLOAT64_C(   492.11), EASYSIMD_FLOAT64_C(   252.19), EASYSIMD_FLOAT64_C(  -131.64) },
      { EASYSIMD_FLOAT64_C(  -751.66), EASYSIMD_FLOAT64_C(   158.71), EASYSIMD_FLOAT64_C(   929.61), EASYSIMD_FLOAT64_C(   977.59) },
      { EASYSIMD_FLOAT64_C(   832.32), EASYSIMD_FLOAT64_C(   492.11), EASYSIMD_FLOAT64_C(   929.61), EASYSIMD_FLOAT64_C(   977.59) } },
    { { EASYSIMD_FLOAT64_C(  -478.59), EASYSIMD_FLOAT64_C(  -617.60), EASYSIMD_FLOAT64_C(  -551.76), EASYSIMD_FLOAT64_C(  -496.61) },
      UINT8_C(227),
      { EASYSIMD_FLOAT64_C(   404.16), EASYSIMD_FLOAT64_C(  -765.84), EASYSIMD_FLOAT64_C(  -161.48), EASYSIMD_FLOAT64_C(  -313.30) },
      { EASYSIMD_FLOAT64_C(  -609.47), EASYSIMD_FLOAT64_C(  -128.20), EASYSIMD_FLOAT64_C(  -186.53), EASYSIMD_FLOAT64_C(  -652.14) },
      { EASYSIMD_FLOAT64_C(   404.16), EASYSIMD_FLOAT64_C(  -128.20), EASYSIMD_FLOAT64_C(  -551.76), EASYSIMD_FLOAT64_C(  -496.61) } },
    { { EASYSIMD_FLOAT64_C(  -420.34), EASYSIMD_FLOAT64_C(   119.20), EASYSIMD_FLOAT64_C(  -996.01), EASYSIMD_FLOAT64_C(  -349.75) },
      UINT8_C(203),
      { EASYSIMD_FLOAT64_C(   836.31), EASYSIMD_FLOAT64_C(   995.58), EASYSIMD_FLOAT64_C(   160.13), EASYSIMD_FLOAT64_C(   719.39) },
      { EASYSIMD_FLOAT64_C(  -814.74), EASYSIMD_FLOAT64_C(   512.84), EASYSIMD_FLOAT64_C(   211.50), EASYSIMD_FLOAT64_C(   437.46) },
      { EASYSIMD_FLOAT64_C(   836.31), EASYSIMD_FLOAT64_C(   995.58), EASYSIMD_FLOAT64_C(  -996.01), EASYSIMD_FLOAT64_C(   719.39) } },
    { { EASYSIMD_FLOAT64_C(  -618.80), EASYSIMD_FLOAT64_C(   459.83), EASYSIMD_FLOAT64_C(  -403.83), EASYSIMD_FLOAT64_C(  -689.19) },
      UINT8_C(227),
      { EASYSIMD_FLOAT64_C(   117.58), EASYSIMD_FLOAT64_C(  -306.79), EASYSIMD_FLOAT64_C(   885.66), EASYSIMD_FLOAT64_C(   620.97) },
      { EASYSIMD_FLOAT64_C(   266.19), EASYSIMD_FLOAT64_C(   289.82), EASYSIMD_FLOAT64_C(   855.14), EASYSIMD_FLOAT64_C(  -895.29) },
      { EASYSIMD_FLOAT64_C(   266.19), EASYSIMD_FLOAT64_C(   289.82), EASYSIMD_FLOAT64_C(  -403.83), EASYSIMD_FLOAT64_C(  -689.19) } },
    { { EASYSIMD_FLOAT64_C(   976.52), EASYSIMD_FLOAT64_C(  -754.33), EASYSIMD_FLOAT64_C(   -23.49), EASYSIMD_FLOAT64_C(  -210.01) },
      UINT8_C(152),
      { EASYSIMD_FLOAT64_C(   556.18), EASYSIMD_FLOAT64_C(   909.19), EASYSIMD_FLOAT64_C(  -402.48), EASYSIMD_FLOAT64_C(  -793.57) },
      { EASYSIMD_FLOAT64_C(   163.86), EASYSIMD_FLOAT64_C(  -566.17), EASYSIMD_FLOAT64_C(  -797.99), EASYSIMD_FLOAT64_C(  -676.00) },
      { EASYSIMD_FLOAT64_C(   976.52), EASYSIMD_FLOAT64_C(  -754.33), EASYSIMD_FLOAT64_C(   -23.49), EASYSIMD_FLOAT64_C(  -676.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_max_pd(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_max_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_max_pd(src, k, a, b);

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
test_easysimd_mm256_maskz_max_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C( 70),
      { EASYSIMD_FLOAT64_C(  -496.93), EASYSIMD_FLOAT64_C(   739.76), EASYSIMD_FLOAT64_C(   835.64), EASYSIMD_FLOAT64_C(   837.57) },
      { EASYSIMD_FLOAT64_C(   306.34), EASYSIMD_FLOAT64_C(   219.19), EASYSIMD_FLOAT64_C(  -941.34), EASYSIMD_FLOAT64_C(   282.73) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   739.76), EASYSIMD_FLOAT64_C(   835.64), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(248),
      { EASYSIMD_FLOAT64_C(   542.19), EASYSIMD_FLOAT64_C(   206.52), EASYSIMD_FLOAT64_C(  -620.89), EASYSIMD_FLOAT64_C(  -805.64) },
      { EASYSIMD_FLOAT64_C(  -532.61), EASYSIMD_FLOAT64_C(   210.13), EASYSIMD_FLOAT64_C(   547.14), EASYSIMD_FLOAT64_C(  -539.62) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -539.62) } },
    { UINT8_C(162),
      { EASYSIMD_FLOAT64_C(   721.35), EASYSIMD_FLOAT64_C(   128.33), EASYSIMD_FLOAT64_C(  -482.75), EASYSIMD_FLOAT64_C(   111.94) },
      { EASYSIMD_FLOAT64_C(   125.93), EASYSIMD_FLOAT64_C(  -253.39), EASYSIMD_FLOAT64_C(    35.27), EASYSIMD_FLOAT64_C(  -601.52) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   128.33), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(  9),
      { EASYSIMD_FLOAT64_C(  -570.21), EASYSIMD_FLOAT64_C(  -873.58), EASYSIMD_FLOAT64_C(   846.18), EASYSIMD_FLOAT64_C(  -458.54) },
      { EASYSIMD_FLOAT64_C(  -370.51), EASYSIMD_FLOAT64_C(   585.94), EASYSIMD_FLOAT64_C(  -622.89), EASYSIMD_FLOAT64_C(  -532.94) },
      { EASYSIMD_FLOAT64_C(  -370.51), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -458.54) } },
    { UINT8_C(164),
      { EASYSIMD_FLOAT64_C(   596.30), EASYSIMD_FLOAT64_C(  -474.28), EASYSIMD_FLOAT64_C(  -824.99), EASYSIMD_FLOAT64_C(   240.49) },
      { EASYSIMD_FLOAT64_C(  -932.09), EASYSIMD_FLOAT64_C(   381.53), EASYSIMD_FLOAT64_C(   619.60), EASYSIMD_FLOAT64_C(  -737.73) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   619.60), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(138),
      { EASYSIMD_FLOAT64_C(  -170.27), EASYSIMD_FLOAT64_C(   809.41), EASYSIMD_FLOAT64_C(  -690.70), EASYSIMD_FLOAT64_C(  -523.74) },
      { EASYSIMD_FLOAT64_C(   530.76), EASYSIMD_FLOAT64_C(   437.63), EASYSIMD_FLOAT64_C(    -6.50), EASYSIMD_FLOAT64_C(  -357.30) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   809.41), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -357.30) } },
    { UINT8_C( 82),
      { EASYSIMD_FLOAT64_C(   740.12), EASYSIMD_FLOAT64_C(   677.97), EASYSIMD_FLOAT64_C(   -37.95), EASYSIMD_FLOAT64_C(  -729.61) },
      { EASYSIMD_FLOAT64_C(  -892.24), EASYSIMD_FLOAT64_C(    88.47), EASYSIMD_FLOAT64_C(  -883.43), EASYSIMD_FLOAT64_C(  -350.78) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   677.97), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(159),
      { EASYSIMD_FLOAT64_C(   702.51), EASYSIMD_FLOAT64_C(    26.33), EASYSIMD_FLOAT64_C(  -814.97), EASYSIMD_FLOAT64_C(  -405.21) },
      { EASYSIMD_FLOAT64_C(  -377.37), EASYSIMD_FLOAT64_C(  -289.25), EASYSIMD_FLOAT64_C(  -230.20), EASYSIMD_FLOAT64_C(   863.12) },
      { EASYSIMD_FLOAT64_C(   702.51), EASYSIMD_FLOAT64_C(    26.33), EASYSIMD_FLOAT64_C(  -230.20), EASYSIMD_FLOAT64_C(   863.12) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_max_pd(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_max_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_max_pd(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_max_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      {  INT8_C(  99),  INT8_C(  57),  INT8_C(  69), -INT8_C(   9), -INT8_C(   6),  INT8_C(  33),  INT8_C( 101),  INT8_C(  36),
         INT8_C(  33),  INT8_C(  54),  INT8_C(  31),  INT8_C( 111), -INT8_C(   2),  INT8_C( 100),  INT8_C(   4),  INT8_C( 114),
         INT8_C(  96),  INT8_C(  68),  INT8_C(  80),  INT8_C(  98),  INT8_C(  93), -INT8_C(  21),      INT8_MAX,  INT8_C( 103),
         INT8_C( 108),  INT8_C(  54),  INT8_C(  88), -INT8_C(  11),  INT8_C(  48),  INT8_C(  84),  INT8_C( 121),  INT8_C(  33),
         INT8_C(  94),  INT8_C(  56),  INT8_C(  11),  INT8_C( 113),      INT8_MAX,  INT8_C( 112),  INT8_C( 124),  INT8_C( 122),
         INT8_C(  69),  INT8_C( 108),  INT8_C(  80),  INT8_C(  67), -INT8_C(  75),  INT8_C(  60),  INT8_C(  42),  INT8_C(  21),
         INT8_C( 121),  INT8_C(  23),  INT8_C( 119),  INT8_C(  50),  INT8_C( 113),  INT8_C(  73),  INT8_C(  69),  INT8_C( 111),
             INT8_MAX,  INT8_C( 123),  INT8_C( 100),  INT8_C( 104), -INT8_C(  49),  INT8_C( 117),  INT8_C(  47), -INT8_C(  20) } },
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
      {  INT8_C(  51),  INT8_C(  59),  INT8_C(  28), -INT8_C(  46),  INT8_C(  53),  INT8_C( 105),  INT8_C(  24), -INT8_C(  47),
         INT8_C(  96),  INT8_C(  41),  INT8_C(  22), -INT8_C( 110),  INT8_C(  58),  INT8_C( 116),  INT8_C(  20),  INT8_C(  97),
         INT8_C(  79),  INT8_C(  85),  INT8_C( 115),  INT8_C(  82),  INT8_C(  96),  INT8_C(  79),  INT8_C(  47),  INT8_C(  99),
         INT8_C(  16),  INT8_C(  43), -INT8_C(  19),  INT8_C(  20), -INT8_C(  88),  INT8_C(  29), -INT8_C(  35),  INT8_C( 105),
         INT8_C(  88), -INT8_C(  86),  INT8_C(  59),  INT8_C( 124),  INT8_C(  75), -INT8_C(  12),  INT8_C(  61), -INT8_C(  26),
         INT8_C(  29),  INT8_C(  83),  INT8_C( 113),  INT8_C(  87),  INT8_C(  50),  INT8_C(  82), -INT8_C(  71), -INT8_C(  77),
         INT8_C(  39),  INT8_C(   8),  INT8_C(  65),  INT8_C( 110), -INT8_C(  15),  INT8_C( 113),  INT8_C( 102), -INT8_C(  82),
        -INT8_C( 100),  INT8_C( 113), -INT8_C(  87),  INT8_C(  61),  INT8_C( 113),  INT8_C(  25),  INT8_C(  16),  INT8_C( 104) } },
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
      {  INT8_C( 116), -INT8_C(  30),  INT8_C( 114),  INT8_C( 124),  INT8_C(  12),  INT8_C(  34),  INT8_C(  41), -INT8_C(  13),
         INT8_C( 117),  INT8_C( 102),  INT8_C(  75),  INT8_C( 116),  INT8_C( 101),  INT8_C(   4),  INT8_C(  87),  INT8_C(  95),
         INT8_C(  35),  INT8_C(  57),  INT8_C(  83),  INT8_C(  63),  INT8_C(  71),  INT8_C(  41),  INT8_C( 106),  INT8_C(  44),
         INT8_C(  15), -INT8_C(  42),  INT8_C( 109),  INT8_C(  49),  INT8_C(  93),  INT8_C(  61), -INT8_C( 103),  INT8_C(  49),
         INT8_C(  23),  INT8_C( 126),  INT8_C(  69),  INT8_C(  98), -INT8_C(  47),  INT8_C( 110),  INT8_C(  85),  INT8_C(  57),
         INT8_C(  34), -INT8_C(  66),  INT8_C(  69),  INT8_C(  80),  INT8_C(  99), -INT8_C(  79), -INT8_C(  54),  INT8_C(  74),
        -INT8_C(  34),  INT8_C( 125), -INT8_C(  30),  INT8_C(  29),  INT8_C(  70),  INT8_C(  48),  INT8_C(  73),  INT8_C(  74),
        -INT8_C(   9), -INT8_C(  38),  INT8_C(  92),  INT8_C(  65), -INT8_C(  13),  INT8_C( 105), -INT8_C(  27),  INT8_C(  16) } },
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
      {  INT8_C( 114),  INT8_C(  42),  INT8_C(  46),  INT8_C(  67), -INT8_C(  14), -INT8_C(  10),  INT8_C( 124), -INT8_C(  38),
         INT8_C( 124), -INT8_C(  62),  INT8_C(  65),  INT8_C( 106),  INT8_C( 125),  INT8_C(  11), -INT8_C(  98),  INT8_C(  52),
         INT8_C( 102),  INT8_C( 100),  INT8_C(  81),  INT8_C( 111),  INT8_C(  68), -INT8_C(  10), -INT8_C(  71),  INT8_C(  13),
         INT8_C(  80),  INT8_C( 124),  INT8_C(  63),  INT8_C(  90), -INT8_C(  13),  INT8_C(  36),  INT8_C( 108),  INT8_C( 102),
         INT8_C(  78),  INT8_C( 124), -INT8_C(  63), -INT8_C(  26),  INT8_C( 115),  INT8_C(  38), -INT8_C(  95),  INT8_C( 122),
        -INT8_C(  24), -INT8_C(  30),  INT8_C(  63),  INT8_C(  70),  INT8_C(  64),  INT8_C( 100),  INT8_C( 122),  INT8_C(  22),
         INT8_C( 106), -INT8_C(  53),  INT8_C( 114), -INT8_C(  42),  INT8_C( 101),  INT8_C(  62),  INT8_C(  97),  INT8_C(  29),
         INT8_C( 112), -INT8_C(  96),  INT8_C( 119),  INT8_C(  53),  INT8_C( 115),  INT8_C(  72),  INT8_C(  34),  INT8_C(  43) } },
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
      {  INT8_C(  71), -INT8_C(  84),  INT8_C(  98),  INT8_C(  69),  INT8_C(  34),  INT8_C(  90),  INT8_C(  17),  INT8_C(  97),
         INT8_C( 102), -INT8_C(  93),  INT8_C(  92),  INT8_C(  63),  INT8_C(   8), -INT8_C(  17),  INT8_C( 103),  INT8_C( 114),
         INT8_C(  43),  INT8_C(  88),  INT8_C(  33),  INT8_C(  55),  INT8_C(  77),  INT8_C(  71),  INT8_C(  47),  INT8_C(  35),
         INT8_C( 100),  INT8_C(  70), -INT8_C(  14),  INT8_C(  84),  INT8_C(  41), -INT8_C(  79),  INT8_C(   3),  INT8_C( 123),
         INT8_C(  76),  INT8_C(  70),  INT8_C(  89),  INT8_C( 110),  INT8_C(  51),  INT8_C( 117),  INT8_C(  51),  INT8_C(  65),
         INT8_C(  25),  INT8_C(   5),  INT8_C( 114),  INT8_C(  33),  INT8_C( 107),  INT8_C( 121),  INT8_C(  89), -INT8_C( 103),
         INT8_C(  40), -INT8_C(  28),  INT8_C(  72), -INT8_C(   2),  INT8_C(  43),  INT8_C(  23),  INT8_C(  17),  INT8_C(  32),
         INT8_C(  93), -INT8_C(  70),  INT8_C( 116), -INT8_C(  51),  INT8_C(  95),  INT8_C(  88),  INT8_C(  72), -INT8_C(  91) } },
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
      {  INT8_C(  61),  INT8_C(  34),  INT8_C(  19),  INT8_C( 121),  INT8_C(  18), -INT8_C(  57), -INT8_C(  23), -INT8_C(   4),
         INT8_C( 108),  INT8_C(  86), -INT8_C(  46),  INT8_C(  49),  INT8_C(  69),  INT8_C(  66),  INT8_C(  79),  INT8_C( 109),
         INT8_C(  13),  INT8_C(  31),  INT8_C(  91),  INT8_C(  69),  INT8_C(  67),  INT8_C( 108),  INT8_C(  81),  INT8_C(  84),
         INT8_C(  38), -INT8_C(  59), -INT8_C(  22),  INT8_C(  52),  INT8_C(  30),  INT8_C( 112),  INT8_C(  60), -INT8_C(  61),
        -INT8_C(  44),  INT8_C(  57),  INT8_C(  54), -INT8_C(  31), -INT8_C(  11), -INT8_C(  16),  INT8_C( 116),  INT8_C(  88),
         INT8_C(  54),  INT8_C(  70),  INT8_C( 109),  INT8_C(  72), -INT8_C(  59), -INT8_C(  68),  INT8_C( 112), -INT8_C(  54),
         INT8_C( 125),  INT8_C(  31), -INT8_C(   5), -INT8_C(   6),  INT8_C(  56),      INT8_MAX,  INT8_C(  23),  INT8_C(  94),
         INT8_C(  17),  INT8_C(   1),  INT8_C(  83),  INT8_C(  47),  INT8_C(  51),  INT8_C(   4),  INT8_C(  75),  INT8_C(   8) } },
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
      {  INT8_C( 116),  INT8_C( 124),  INT8_C(  91), -INT8_C(   4),  INT8_C(  33),  INT8_C(  90),  INT8_C( 126),  INT8_C(  87),
        -INT8_C(   7), -INT8_C( 125), -INT8_C(  73),  INT8_C( 109),  INT8_C(   9),  INT8_C(  46),  INT8_C(  22), -INT8_C(  92),
        -INT8_C(  11),  INT8_C(  84),  INT8_C(  70),  INT8_C(  19),  INT8_C( 112),  INT8_C(  91),  INT8_C(  50),  INT8_C( 114),
        -INT8_C(  13),  INT8_C(  24), -INT8_C(   6),  INT8_C(  31),  INT8_C(  20),  INT8_C(  69), -INT8_C(  45),  INT8_C(  43),
         INT8_C(  17),  INT8_C(  46),  INT8_C( 106), -INT8_C(  15), -INT8_C( 119),  INT8_C(  99), -INT8_C(   1), -INT8_C( 126),
         INT8_C(  38),  INT8_C( 111),  INT8_C(   2), -INT8_C(  16),  INT8_C( 116),  INT8_C(  25),  INT8_C( 119),  INT8_C( 105),
         INT8_C(  26), -INT8_C(  67),  INT8_C( 125),  INT8_C( 123),  INT8_C(  24), -INT8_C(  81),  INT8_C(  86),  INT8_C(  12),
         INT8_C(  53), -INT8_C(  25),  INT8_C(   8),  INT8_C( 110),  INT8_C( 109),  INT8_C( 116),  INT8_C(  18),  INT8_C( 126) } },
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
      { -INT8_C(  67), -INT8_C(  24),  INT8_C( 111),  INT8_C(  43),  INT8_C( 102),  INT8_C(  39), -INT8_C(  20),  INT8_C(  42),
        -INT8_C(  70),  INT8_C(   3),  INT8_C(  76),  INT8_C(  52),  INT8_C(  99),  INT8_C(  78), -INT8_C(  66),  INT8_C( 126),
         INT8_C(  59),  INT8_C(  77),  INT8_C(   9),  INT8_C(  26),      INT8_MAX, -INT8_C(  14),  INT8_C(  66), -INT8_C(  15),
             INT8_MAX, -INT8_C(  29),  INT8_C(  18),  INT8_C(  56),  INT8_C(  87),  INT8_C( 126), -INT8_C(  53), -INT8_C(   3),
         INT8_C( 108),  INT8_C( 122),  INT8_C(  37),  INT8_C(  32),  INT8_C(  76), -INT8_C(  13),  INT8_C(  95),  INT8_C(  66),
         INT8_C( 111),  INT8_C(  71),  INT8_C(  87), -INT8_C(  45),  INT8_C(  91), -INT8_C(   7),  INT8_C(  81), -INT8_C(  48),
         INT8_C(  34), -INT8_C(  19),  INT8_C(  77),  INT8_C( 104), -INT8_C(  33), -INT8_C(  80),  INT8_C(  89),  INT8_C(  14),
         INT8_C( 114),  INT8_C(  20), -INT8_C(   3), -INT8_C(  55),  INT8_C( 111), -INT8_C(  56), -INT8_C(  61), -INT8_C(   8) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_epi8(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_epi8");
    easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int8_t src[64];
    const easysimd__mmask64 k;
    const int8_t a[64];
    const int8_t b[64];
    const int8_t r[64];
  } test_vec[] = {
    { {  INT8_C(  47), -INT8_C( 108), -INT8_C( 103),  INT8_C(  35),  INT8_C(  58), -INT8_C(  33), -INT8_C(  38),  INT8_C(  76),
         INT8_C(  90),  INT8_C(  55), -INT8_C(  80),  INT8_C(  59),  INT8_C( 118),  INT8_C( 110),  INT8_C( 101),  INT8_C(  47),
        -INT8_C( 123), -INT8_C(  34), -INT8_C( 109),  INT8_C(  94), -INT8_C( 124),  INT8_C( 125),  INT8_C( 100), -INT8_C(  88),
         INT8_C( 110), -INT8_C(   4), -INT8_C(  17), -INT8_C(  46),  INT8_C(  96), -INT8_C(  12), -INT8_C(  10), -INT8_C( 113),
        -INT8_C( 120), -INT8_C( 113), -INT8_C(  78), -INT8_C(  62),  INT8_C( 110), -INT8_C( 115),  INT8_C(  15), -INT8_C(  55),
        -INT8_C(  60), -INT8_C(  65),  INT8_C(   4),  INT8_C(  58),  INT8_C(  46),  INT8_C( 105),  INT8_C( 106), -INT8_C(  77),
         INT8_C(  72), -INT8_C(   3),  INT8_C(  17), -INT8_C(  52),  INT8_C( 122),  INT8_C( 118),  INT8_C( 116), -INT8_C(  24),
         INT8_C( 114),  INT8_C( 100), -INT8_C(  70), -INT8_C(  46),  INT8_C(  88), -INT8_C(  79),  INT8_C(  97), -INT8_C(  31) },
      UINT64_C( 7311790402542179392),
      {  INT8_C( 114),  INT8_C( 124), -INT8_C(  97), -INT8_C(  96), -INT8_C(  27),  INT8_C(   9),  INT8_C(  83),  INT8_C(  45),
         INT8_C(   6),  INT8_C( 101), -INT8_C(   7),      INT8_MIN, -INT8_C(  37),  INT8_C( 110),  INT8_C( 104),  INT8_C(  77),
        -INT8_C(  46),  INT8_C(  35),  INT8_C(  31),  INT8_C(  42), -INT8_C(  44),      INT8_MIN,  INT8_C(  11),  INT8_C(  20),
        -INT8_C( 108), -INT8_C(  81), -INT8_C(  61),  INT8_C(  53),  INT8_C(  97),  INT8_C(  59), -INT8_C( 102), -INT8_C(  45),
        -INT8_C(  73),  INT8_C(  58),  INT8_C( 115), -INT8_C(  99),  INT8_C(  67), -INT8_C(  57), -INT8_C(  54),  INT8_C(  74),
         INT8_C(  44), -INT8_C(  60), -INT8_C(  54),  INT8_C(   7),  INT8_C(  50),  INT8_C(  51),  INT8_C(  84),  INT8_C(   4),
         INT8_C(  86),  INT8_C( 115),  INT8_C(  46),  INT8_C(  42), -INT8_C(  13),  INT8_C(  58),  INT8_C(  62), -INT8_C( 120),
        -INT8_C(  23),  INT8_C(   2), -INT8_C(  67),  INT8_C(  74),  INT8_C(  61),  INT8_C(  88),  INT8_C(  30), -INT8_C(  11) },
      { -INT8_C( 110), -INT8_C( 111), -INT8_C( 110), -INT8_C(  43),  INT8_C(  88),  INT8_C(  92),  INT8_C(  31), -INT8_C( 124),
         INT8_C(  32), -INT8_C(  22), -INT8_C( 117),  INT8_C(  82),  INT8_C(  29), -INT8_C(  33),  INT8_C(  86),  INT8_C( 115),
         INT8_C(  82), -INT8_C( 123), -INT8_C(  99),  INT8_C(  70), -INT8_C(  65), -INT8_C(  37), -INT8_C(  50), -INT8_C(  88),
        -INT8_C(  35), -INT8_C( 117), -INT8_C(  14),  INT8_C(  27), -INT8_C(  29),  INT8_C(  16),  INT8_C(  16),  INT8_C( 117),
        -INT8_C(  94), -INT8_C(  94),  INT8_C(  75), -INT8_C(   6), -INT8_C(   2),  INT8_C( 106),      INT8_MAX,  INT8_C(  31),
         INT8_C(  84),  INT8_C(  10),  INT8_C( 113),  INT8_C( 113), -INT8_C(  22), -INT8_C(  56), -INT8_C(  28),  INT8_C(  60),
         INT8_C(  77), -INT8_C( 127), -INT8_C( 126),  INT8_C(  12),  INT8_C(  93),  INT8_C(  80), -INT8_C(  76),  INT8_C(  58),
        -INT8_C(  36), -INT8_C(  90),  INT8_C(  85), -INT8_C(  65), -INT8_C(  73),  INT8_C( 101),  INT8_C(  53),  INT8_C(  89) },
      {  INT8_C(  47), -INT8_C( 108), -INT8_C( 103),  INT8_C(  35),  INT8_C(  58), -INT8_C(  33),  INT8_C(  83),  INT8_C(  76),
         INT8_C(  90),  INT8_C(  55), -INT8_C(   7),  INT8_C(  59),  INT8_C(  29),  INT8_C( 110),  INT8_C( 101),  INT8_C(  47),
         INT8_C(  82),  INT8_C(  35), -INT8_C( 109),  INT8_C(  94), -INT8_C( 124), -INT8_C(  37),  INT8_C( 100),  INT8_C(  20),
        -INT8_C(  35), -INT8_C(  81), -INT8_C(  14),  INT8_C(  53),  INT8_C(  96),  INT8_C(  59), -INT8_C(  10),  INT8_C( 117),
        -INT8_C(  73), -INT8_C( 113), -INT8_C(  78), -INT8_C(  62),  INT8_C( 110),  INT8_C( 106),  INT8_C(  15),  INT8_C(  74),
        -INT8_C(  60),  INT8_C(  10),  INT8_C(   4),  INT8_C(  58),  INT8_C(  50),  INT8_C(  51),  INT8_C( 106),  INT8_C(  60),
         INT8_C(  72), -INT8_C(   3),  INT8_C(  17),  INT8_C(  42),  INT8_C(  93),  INT8_C(  80),  INT8_C(  62), -INT8_C(  24),
        -INT8_C(  23),  INT8_C( 100),  INT8_C(  85), -INT8_C(  46),  INT8_C(  88),  INT8_C( 101),  INT8_C(  53), -INT8_C(  31) } },
    { {  INT8_C(   7),      INT8_MIN,  INT8_C(  83),  INT8_C(   6), -INT8_C(  22), -INT8_C(  46),  INT8_C(  37),  INT8_C(  63),
        -INT8_C(  35), -INT8_C( 106), -INT8_C(  80), -INT8_C(  57),  INT8_C(  94), -INT8_C( 107),  INT8_C(   3), -INT8_C(  85),
         INT8_C(  22), -INT8_C( 122), -INT8_C(  73),  INT8_C( 115), -INT8_C(  42),  INT8_C( 107), -INT8_C(  82), -INT8_C(  78),
         INT8_C(  18),  INT8_C(   3),  INT8_C( 114), -INT8_C(  55),  INT8_C( 105), -INT8_C(  89),  INT8_C(  34),  INT8_C( 112),
         INT8_C(  39),  INT8_C( 117),  INT8_C( 118),  INT8_C(  17),  INT8_C(  72), -INT8_C( 101),  INT8_C(  80),  INT8_C(  37),
         INT8_C(  50),  INT8_C(   1), -INT8_C(  20), -INT8_C( 112), -INT8_C( 106), -INT8_C(  17),  INT8_C(  60), -INT8_C(  84),
         INT8_C( 117), -INT8_C(  13),  INT8_C(  32),  INT8_C(  76),  INT8_C(  95), -INT8_C(  50), -INT8_C(   2),  INT8_C( 113),
        -INT8_C(  47),  INT8_C( 112),  INT8_C(  58),  INT8_C(  58),  INT8_C(  23),  INT8_C(  92), -INT8_C(  85),  INT8_C(  62) },
      UINT64_C(17239393157654782417),
      { -INT8_C(  95),  INT8_C(  42),      INT8_MAX,  INT8_C(  55),  INT8_C(  26), -INT8_C(  69), -INT8_C(  28), -INT8_C( 113),
        -INT8_C(  81),  INT8_C(   4), -INT8_C(  37),  INT8_C(  14), -INT8_C(  46), -INT8_C(  38),      INT8_MAX, -INT8_C(  93),
         INT8_C(  74), -INT8_C(  71), -INT8_C(  34),  INT8_C(  98),  INT8_C(  21), -INT8_C( 119), -INT8_C(  96), -INT8_C(  26),
        -INT8_C(  86), -INT8_C(  16),  INT8_C(   0),  INT8_C( 103), -INT8_C( 111),  INT8_C(  62),  INT8_C(  86),  INT8_C(  50),
         INT8_C( 105), -INT8_C(  42),  INT8_C( 106), -INT8_C( 125), -INT8_C( 111),  INT8_C(  78),  INT8_C(  18),  INT8_C(  64),
         INT8_C(  82), -INT8_C(  18),  INT8_C(  78),  INT8_C(  36), -INT8_C(  56), -INT8_C(  51), -INT8_C(  57),  INT8_C(  18),
        -INT8_C( 122), -INT8_C(  91),  INT8_C( 116), -INT8_C( 101),  INT8_C(  46),  INT8_C(  21), -INT8_C( 126), -INT8_C(  39),
         INT8_C(   5), -INT8_C( 126),  INT8_C(  64), -INT8_C( 106), -INT8_C(  64), -INT8_C( 105), -INT8_C(  55),  INT8_C(  41) },
      {  INT8_C( 109),  INT8_C(  51), -INT8_C(  84), -INT8_C(   2), -INT8_C( 127), -INT8_C(  65),  INT8_C(  63), -INT8_C(  45),
        -INT8_C(  83), -INT8_C( 115), -INT8_C(   9),  INT8_C( 117),  INT8_C(  91), -INT8_C(  66), -INT8_C( 121), -INT8_C(  31),
         INT8_C( 100), -INT8_C(   4),  INT8_C( 125), -INT8_C( 110),  INT8_C(  17), -INT8_C(   1),  INT8_C( 107),  INT8_C(  22),
        -INT8_C( 127), -INT8_C(  84), -INT8_C(  83),  INT8_C(  65),  INT8_C(  67),  INT8_C( 118),  INT8_C( 107), -INT8_C(  80),
        -INT8_C(  87),  INT8_C(  23), -INT8_C(  82),  INT8_C(  42), -INT8_C(  42), -INT8_C(  19), -INT8_C(   3), -INT8_C( 125),
         INT8_C( 123), -INT8_C(  12), -INT8_C(   8), -INT8_C(  42), -INT8_C(  78),      INT8_MIN, -INT8_C(  73),  INT8_C(  22),
         INT8_C( 124),  INT8_C(  52), -INT8_C(  87), -INT8_C( 115),  INT8_C(  51),  INT8_C(  20), -INT8_C(  93), -INT8_C(  76),
        -INT8_C(  64),  INT8_C(  80), -INT8_C(  10),  INT8_C(   3), -INT8_C(  58),  INT8_C(  97), -INT8_C(  77),  INT8_C( 111) },
      {  INT8_C( 109),      INT8_MIN,  INT8_C(  83),  INT8_C(   6),  INT8_C(  26), -INT8_C(  46),  INT8_C(  63), -INT8_C(  45),
        -INT8_C(  81), -INT8_C( 106), -INT8_C(  80), -INT8_C(  57),  INT8_C(  94), -INT8_C(  38),  INT8_C(   3), -INT8_C(  85),
         INT8_C(  22), -INT8_C( 122), -INT8_C(  73),  INT8_C( 115),  INT8_C(  21),  INT8_C( 107),  INT8_C( 107), -INT8_C(  78),
        -INT8_C(  86),  INT8_C(   3),  INT8_C( 114),  INT8_C( 103),  INT8_C(  67), -INT8_C(  89),  INT8_C(  34),  INT8_C( 112),
         INT8_C( 105),  INT8_C( 117),  INT8_C( 106),  INT8_C(  42), -INT8_C(  42),  INT8_C(  78),  INT8_C(  80),  INT8_C(  64),
         INT8_C(  50),  INT8_C(   1), -INT8_C(  20), -INT8_C( 112), -INT8_C( 106), -INT8_C(  51),  INT8_C(  60),  INT8_C(  22),
         INT8_C( 117),  INT8_C(  52),  INT8_C( 116), -INT8_C( 101),  INT8_C(  51),  INT8_C(  21), -INT8_C(   2),  INT8_C( 113),
         INT8_C(   5),  INT8_C(  80),  INT8_C(  64),  INT8_C(   3),  INT8_C(  23),  INT8_C(  97), -INT8_C(  55),  INT8_C( 111) } },
    { {  INT8_C( 120),  INT8_C(  98), -INT8_C( 103),  INT8_C(  79),  INT8_C(  79), -INT8_C( 106), -INT8_C(  46), -INT8_C(  54),
        -INT8_C( 118), -INT8_C(  53), -INT8_C(  96),  INT8_C(  61),  INT8_C(  75),  INT8_C(  88),  INT8_C(  83), -INT8_C(  57),
        -INT8_C( 116), -INT8_C(   4),  INT8_C(  84), -INT8_C(  64),  INT8_C(  17), -INT8_C(   9),  INT8_C( 116), -INT8_C(  47),
         INT8_C(  72),  INT8_C( 106), -INT8_C(  43),  INT8_C(  14), -INT8_C(  53), -INT8_C( 120),  INT8_C( 126),  INT8_C(  68),
        -INT8_C(  22),  INT8_C(  23), -INT8_C( 109),  INT8_C(  58), -INT8_C(  82),  INT8_C( 101),  INT8_C(   4),  INT8_C(  56),
         INT8_C(  48), -INT8_C(  91),  INT8_C( 117),  INT8_C( 123), -INT8_C(   3), -INT8_C(  55),  INT8_C(  66), -INT8_C( 119),
        -INT8_C(  59), -INT8_C( 106),  INT8_C(  73), -INT8_C(  42), -INT8_C( 114), -INT8_C(  66), -INT8_C(  88), -INT8_C(  42),
         INT8_C(  40),  INT8_C( 125), -INT8_C(  28), -INT8_C(  12),  INT8_C(   5),  INT8_C(  98),  INT8_C(  56), -INT8_C(  16) },
      UINT64_C( 7016659003810433914),
      { -INT8_C(  45), -INT8_C(  42), -INT8_C(  36), -INT8_C(  48), -INT8_C(  97),  INT8_C(  31),  INT8_C(  90),  INT8_C( 100),
        -INT8_C(  75), -INT8_C(  93),  INT8_C(  59),  INT8_C(  67),  INT8_C(  97), -INT8_C(  29),  INT8_C(  25), -INT8_C( 118),
         INT8_C(  96), -INT8_C(   2),  INT8_C( 126),  INT8_C( 101),  INT8_C(  96), -INT8_C(  74),  INT8_C(  85), -INT8_C(  38),
        -INT8_C( 127),      INT8_MAX,  INT8_C(   2), -INT8_C(  79), -INT8_C(  82),  INT8_C(  99),  INT8_C(  18), -INT8_C( 127),
         INT8_C(  57), -INT8_C(  17),  INT8_C(  82), -INT8_C(  40),  INT8_C(  14), -INT8_C(  84),  INT8_C(  60), -INT8_C(  61),
         INT8_C(  79),  INT8_C( 119),  INT8_C(   7), -INT8_C(  79),  INT8_C(  90),  INT8_C(  32),  INT8_C(  59), -INT8_C(  70),
         INT8_C(  30), -INT8_C(  71),  INT8_C(  32),      INT8_MAX,  INT8_C( 111),  INT8_C( 117),  INT8_C(  89), -INT8_C(  16),
        -INT8_C(  11),  INT8_C(  92), -INT8_C(  95), -INT8_C(  93), -INT8_C(  65), -INT8_C(  76),  INT8_C(  36), -INT8_C(   8) },
      { -INT8_C(  93),  INT8_C( 118), -INT8_C(  48), -INT8_C(  79),  INT8_C(  34),  INT8_C(  12),  INT8_C( 116),  INT8_C( 114),
        -INT8_C( 124),  INT8_C( 123),  INT8_C(  35), -INT8_C(  34), -INT8_C( 100),  INT8_C(  94), -INT8_C( 103), -INT8_C(  70),
         INT8_C(  23), -INT8_C(  71),  INT8_C(  57), -INT8_C( 122),  INT8_C(  46), -INT8_C( 109),  INT8_C( 118),  INT8_C(  35),
        -INT8_C(  17),  INT8_C(  23), -INT8_C(  58), -INT8_C(  82), -INT8_C(  53), -INT8_C(  21), -INT8_C(  90),  INT8_C( 110),
         INT8_C(  97),  INT8_C( 118),  INT8_C(  31), -INT8_C( 124), -INT8_C( 126), -INT8_C( 108), -INT8_C(  10),  INT8_C(   6),
         INT8_C(  15),  INT8_C(  25), -INT8_C(  27), -INT8_C(  85),  INT8_C( 119),  INT8_C( 126),  INT8_C( 102), -INT8_C( 114),
         INT8_C(  55), -INT8_C(  97),  INT8_C(  20),  INT8_C( 101),  INT8_C(  50), -INT8_C( 118), -INT8_C( 119),  INT8_C(  33),
        -INT8_C(  95),  INT8_C(  79), -INT8_C(  49),  INT8_C( 109),  INT8_C(  58),  INT8_C( 117), -INT8_C(  37), -INT8_C( 100) },
      {  INT8_C( 120),  INT8_C( 118), -INT8_C( 103), -INT8_C(  48),  INT8_C(  34),  INT8_C(  31),  INT8_C( 116), -INT8_C(  54),
        -INT8_C(  75),  INT8_C( 123), -INT8_C(  96),  INT8_C(  67),  INT8_C(  75),  INT8_C(  88),  INT8_C(  25), -INT8_C(  70),
        -INT8_C( 116), -INT8_C(   2),  INT8_C(  84),  INT8_C( 101),  INT8_C(  17), -INT8_C(  74),  INT8_C( 116), -INT8_C(  47),
         INT8_C(  72),  INT8_C( 106), -INT8_C(  43), -INT8_C(  79), -INT8_C(  53),  INT8_C(  99),  INT8_C( 126),  INT8_C(  68),
        -INT8_C(  22),  INT8_C(  23), -INT8_C( 109),  INT8_C(  58),  INT8_C(  14), -INT8_C(  84),  INT8_C(   4),  INT8_C(  56),
         INT8_C(  48),  INT8_C( 119),  INT8_C(   7), -INT8_C(  79), -INT8_C(   3),  INT8_C( 126),  INT8_C(  66), -INT8_C( 119),
        -INT8_C(  59), -INT8_C( 106),  INT8_C(  73), -INT8_C(  42), -INT8_C( 114),  INT8_C( 117),  INT8_C(  89), -INT8_C(  42),
        -INT8_C(  11),  INT8_C( 125), -INT8_C(  28), -INT8_C(  12),  INT8_C(   5),  INT8_C( 117),  INT8_C(  36), -INT8_C(  16) } },
    { { -INT8_C(  21), -INT8_C(   5),  INT8_C(  32),  INT8_C( 110), -INT8_C( 113),  INT8_C(  22),  INT8_C( 116), -INT8_C(  98),
         INT8_C(  47),  INT8_C(  89),  INT8_C(  74), -INT8_C(  90), -INT8_C(  41), -INT8_C(  80),  INT8_C(  52),  INT8_C(  14),
         INT8_C(  79),  INT8_C(  72),  INT8_C( 116), -INT8_C( 126), -INT8_C(  46), -INT8_C(   3), -INT8_C(  93),  INT8_C( 115),
         INT8_C(  76),  INT8_C( 115), -INT8_C(  32), -INT8_C( 121), -INT8_C(  24), -INT8_C(  68),  INT8_C(  35), -INT8_C(  44),
        -INT8_C(  73),  INT8_C(  67),  INT8_C(  66),  INT8_C(  70),  INT8_C(  89), -INT8_C(  74), -INT8_C(  28), -INT8_C( 120),
         INT8_C(  16),  INT8_C(  46),  INT8_C(  46), -INT8_C(  25), -INT8_C(  34),  INT8_C(  98), -INT8_C(  10),  INT8_C(  46),
        -INT8_C(  86),  INT8_C( 106), -INT8_C(  80),  INT8_C( 124),  INT8_C( 103),  INT8_C(  83), -INT8_C(  17), -INT8_C(  77),
        -INT8_C(  58), -INT8_C(  48),  INT8_C(  58), -INT8_C(  81), -INT8_C( 116),  INT8_C(  93), -INT8_C( 125),  INT8_C(  67) },
      UINT64_C(10052436222502618528),
      { -INT8_C( 100), -INT8_C(  81),  INT8_C( 115),  INT8_C( 122),  INT8_C(  17),  INT8_C( 105), -INT8_C(  88), -INT8_C(  69),
        -INT8_C(  45),  INT8_C(  88),  INT8_C(  55),  INT8_C(  58), -INT8_C(  84),  INT8_C(  39), -INT8_C(  19),  INT8_C( 114),
        -INT8_C(   9),  INT8_C(  40),  INT8_C(  33), -INT8_C( 125), -INT8_C( 123), -INT8_C(  92), -INT8_C(  58),  INT8_C(  38),
         INT8_C( 105),  INT8_C(  79),  INT8_C(  31), -INT8_C(  27), -INT8_C(  68), -INT8_C(  95),  INT8_C( 112),  INT8_C(  88),
         INT8_C(  80), -INT8_C(  29), -INT8_C(  45),  INT8_C(  98),  INT8_C(  76),  INT8_C( 123),  INT8_C(  29),  INT8_C(  31),
        -INT8_C(  44),  INT8_C(  85),  INT8_C(  89),      INT8_MIN,  INT8_C( 124),  INT8_C(  71), -INT8_C(  14),  INT8_C( 115),
         INT8_C( 111),  INT8_C(  20), -INT8_C(  10), -INT8_C(  12), -INT8_C(  72), -INT8_C(  68),  INT8_C(  26),  INT8_C(  34),
         INT8_C(  11),  INT8_C(  58),  INT8_C(   7), -INT8_C(  57), -INT8_C(  37),  INT8_C( 119),  INT8_C(  32),  INT8_C(  43) },
      {  INT8_C(  91), -INT8_C(  13), -INT8_C( 115), -INT8_C(  89),  INT8_C( 110), -INT8_C(  85), -INT8_C(  57),  INT8_C(  66),
         INT8_C(   0),  INT8_C(  32), -INT8_C(  62),  INT8_C( 124),  INT8_C( 103), -INT8_C(  75), -INT8_C(  17), -INT8_C(  42),
        -INT8_C(  55), -INT8_C(  27), -INT8_C(  53), -INT8_C( 127), -INT8_C(  95), -INT8_C(  27), -INT8_C(  93), -INT8_C(  84),
         INT8_C(  31), -INT8_C(  86),  INT8_C( 115), -INT8_C(   6),  INT8_C(  34), -INT8_C( 109),  INT8_C(  38),  INT8_C( 125),
        -INT8_C( 122), -INT8_C(  77),  INT8_C(  36), -INT8_C(  11),  INT8_C(  94), -INT8_C(  21),  INT8_C(  55),  INT8_C(  94),
         INT8_C(  12), -INT8_C(   6), -INT8_C(  38),  INT8_C( 115), -INT8_C(  81), -INT8_C(  55),  INT8_C(  74),  INT8_C( 120),
        -INT8_C(  82),  INT8_C(  21), -INT8_C(   7),  INT8_C(  79), -INT8_C(   6), -INT8_C(  99), -INT8_C(   5),  INT8_C(  26),
         INT8_C(  71),  INT8_C( 111),  INT8_C(  20),  INT8_C( 105),  INT8_C(   2),  INT8_C(  58), -INT8_C(  26), -INT8_C( 119) },
      { -INT8_C(  21), -INT8_C(   5),  INT8_C(  32),  INT8_C( 110), -INT8_C( 113),  INT8_C( 105),  INT8_C( 116),  INT8_C(  66),
         INT8_C(   0),  INT8_C(  89),  INT8_C(  55), -INT8_C(  90), -INT8_C(  41), -INT8_C(  80), -INT8_C(  17),  INT8_C( 114),
        -INT8_C(   9),  INT8_C(  72),  INT8_C( 116), -INT8_C( 125), -INT8_C(  46), -INT8_C(   3), -INT8_C(  93),  INT8_C(  38),
         INT8_C( 105),  INT8_C( 115), -INT8_C(  32), -INT8_C(   6),  INT8_C(  34), -INT8_C(  95),  INT8_C( 112),  INT8_C( 125),
         INT8_C(  80), -INT8_C(  29),  INT8_C(  66),  INT8_C(  98),  INT8_C(  94),  INT8_C( 123),  INT8_C(  55), -INT8_C( 120),
         INT8_C(  12),  INT8_C(  46),  INT8_C(  89),  INT8_C( 115), -INT8_C(  34),  INT8_C(  71),  INT8_C(  74),  INT8_C(  46),
         INT8_C( 111),  INT8_C( 106), -INT8_C(  80),  INT8_C( 124),  INT8_C( 103),  INT8_C(  83), -INT8_C(  17),  INT8_C(  34),
         INT8_C(  71),  INT8_C( 111),  INT8_C(  58),  INT8_C( 105), -INT8_C( 116),  INT8_C(  93), -INT8_C( 125),  INT8_C(  43) } },
    { { -INT8_C(  18),  INT8_C(  11),  INT8_C( 126),  INT8_C(  76), -INT8_C(  10), -INT8_C(  75), -INT8_C(  85),  INT8_C(   2),
        -INT8_C(  81), -INT8_C( 123),  INT8_C( 118),  INT8_C(  94),  INT8_C(  79), -INT8_C(  64), -INT8_C(  42), -INT8_C(   3),
        -INT8_C(  43), -INT8_C(  48),  INT8_C(  77), -INT8_C(  49),  INT8_C( 109),  INT8_C(  72), -INT8_C(  23), -INT8_C(  76),
        -INT8_C(  73), -INT8_C(   2),  INT8_C(  30), -INT8_C(  70),  INT8_C(  56),  INT8_C(   4),  INT8_C(  67),  INT8_C(  38),
         INT8_C(  15), -INT8_C(  63),  INT8_C( 115),  INT8_C(   6),  INT8_C( 118),  INT8_C(  30),  INT8_C(   8),  INT8_C(  38),
        -INT8_C(  93),  INT8_C( 126), -INT8_C( 124), -INT8_C(  14),  INT8_C(  62),  INT8_C(  91), -INT8_C(  16),  INT8_C(  19),
         INT8_C(  43),  INT8_C(  61), -INT8_C(  29), -INT8_C( 104), -INT8_C( 123), -INT8_C(  52),  INT8_C(  76),  INT8_C(  61),
        -INT8_C(  54),  INT8_C( 106), -INT8_C(   9),  INT8_C(   3),  INT8_C( 111),  INT8_C(  58),  INT8_C(  41),  INT8_C( 126) },
      UINT64_C( 6816072392956484859),
      {  INT8_C(  11),  INT8_C(  28),  INT8_C(  80),  INT8_C(  74),  INT8_C( 119),  INT8_C(  64),  INT8_C(  93), -INT8_C(  94),
         INT8_C( 125),  INT8_C(  64),  INT8_C(  58),  INT8_C(   3),  INT8_C(  13), -INT8_C( 122),  INT8_C(  64), -INT8_C(  41),
        -INT8_C(  15),  INT8_C(  55), -INT8_C(  38),  INT8_C(  96),  INT8_C( 113),  INT8_C(   4), -INT8_C(  34),  INT8_C( 108),
        -INT8_C(  96),  INT8_C(  99), -INT8_C(  35),  INT8_C(  91), -INT8_C(  16),  INT8_C( 117), -INT8_C(  71), -INT8_C(   5),
        -INT8_C( 111),  INT8_C(   9),  INT8_C(  69),  INT8_C(   8),  INT8_C(  74), -INT8_C(  93), -INT8_C(  86), -INT8_C(  57),
        -INT8_C(  29), -INT8_C(  28), -INT8_C(  54), -INT8_C(  16),  INT8_C( 106),  INT8_C(  10), -INT8_C(  56),  INT8_C(  91),
         INT8_C(  65), -INT8_C(  94), -INT8_C(  69), -INT8_C(  78), -INT8_C(  90), -INT8_C( 102),  INT8_C(  30),  INT8_C(  71),
        -INT8_C(   3), -INT8_C(   4), -INT8_C(  94), -INT8_C(  19),  INT8_C( 113),  INT8_C(  91), -INT8_C(  24),  INT8_C(   2) },
      {  INT8_C( 100),  INT8_C(  46),  INT8_C(  10), -INT8_C(  82), -INT8_C(  47), -INT8_C(  76),  INT8_C( 118), -INT8_C(  76),
        -INT8_C( 104),  INT8_C(  64), -INT8_C(  91),  INT8_C(   2),  INT8_C(  75),  INT8_C( 109),  INT8_C(  94), -INT8_C( 116),
         INT8_C(  15),  INT8_C(  25),  INT8_C(  63), -INT8_C(  74), -INT8_C(  77),  INT8_C(  93), -INT8_C(   3), -INT8_C(  80),
         INT8_C(  89), -INT8_C(  97), -INT8_C(  99), -INT8_C(  54), -INT8_C(   6), -INT8_C( 122), -INT8_C(  52),  INT8_C(  94),
        -INT8_C(  76), -INT8_C(  42),  INT8_C(  13), -INT8_C( 123), -INT8_C( 118), -INT8_C( 125),  INT8_C(  57),  INT8_C(  34),
        -INT8_C(  61), -INT8_C(  34),  INT8_C(  37),  INT8_C(  14),  INT8_C(  75), -INT8_C( 125), -INT8_C( 101),  INT8_C(  91),
        -INT8_C( 100), -INT8_C(  38),  INT8_C(  17),  INT8_C(  80),  INT8_C(  55),  INT8_C(  14),  INT8_C(   0), -INT8_C( 111),
        -INT8_C(  83), -INT8_C(  98),  INT8_C(  91), -INT8_C(  89),  INT8_C(  36),  INT8_C(  40),  INT8_C(   5), -INT8_C(  40) },
      {  INT8_C( 100),  INT8_C(  46),  INT8_C( 126),  INT8_C(  74),  INT8_C( 119),  INT8_C(  64),  INT8_C( 118), -INT8_C(  76),
        -INT8_C(  81), -INT8_C( 123),  INT8_C(  58),  INT8_C(   3),  INT8_C(  75), -INT8_C(  64), -INT8_C(  42), -INT8_C(  41),
        -INT8_C(  43), -INT8_C(  48),  INT8_C(  63), -INT8_C(  49),  INT8_C( 109),  INT8_C(  72), -INT8_C(  23),  INT8_C( 108),
         INT8_C(  89), -INT8_C(   2),  INT8_C(  30), -INT8_C(  70), -INT8_C(   6),  INT8_C( 117), -INT8_C(  52),  INT8_C(  38),
         INT8_C(  15),  INT8_C(   9),  INT8_C( 115),  INT8_C(   8),  INT8_C(  74), -INT8_C(  93),  INT8_C(   8),  INT8_C(  34),
        -INT8_C(  29),  INT8_C( 126),  INT8_C(  37),  INT8_C(  14),  INT8_C(  62),  INT8_C(  91), -INT8_C(  16),  INT8_C(  91),
         INT8_C(  65), -INT8_C(  38),  INT8_C(  17), -INT8_C( 104),  INT8_C(  55), -INT8_C(  52),  INT8_C(  76),  INT8_C(  71),
        -INT8_C(  54), -INT8_C(   4),  INT8_C(  91), -INT8_C(  19),  INT8_C( 113),  INT8_C(  58),  INT8_C(   5),  INT8_C( 126) } },
    { { -INT8_C(   2),  INT8_C(  18),  INT8_C(  93), -INT8_C( 119), -INT8_C( 107), -INT8_C( 106), -INT8_C(  85),  INT8_C(  89),
         INT8_C( 117), -INT8_C(  48),  INT8_C( 103), -INT8_C(  64),  INT8_C(  83),  INT8_C(   2),  INT8_C(  27), -INT8_C(  16),
        -INT8_C(  36),  INT8_C(  44),  INT8_C(  64),  INT8_C(  20),  INT8_C(  58),  INT8_C(  64), -INT8_C(  91), -INT8_C(  25),
        -INT8_C(  34),  INT8_C(   0), -INT8_C( 114),  INT8_C(   2),  INT8_C(  40), -INT8_C( 108), -INT8_C(  38),  INT8_C(  39),
        -INT8_C(  90),  INT8_C(  55), -INT8_C(  80),  INT8_C(  60), -INT8_C(  50),  INT8_C(  91), -INT8_C( 107),  INT8_C(  67),
         INT8_C(  44), -INT8_C(   4),  INT8_C(   3),      INT8_MAX, -INT8_C(   1),  INT8_C(  31),  INT8_C( 111), -INT8_C(  37),
         INT8_C(  75), -INT8_C(  81), -INT8_C(  17), -INT8_C( 122), -INT8_C(  16), -INT8_C( 108),  INT8_C( 109), -INT8_C(  50),
        -INT8_C( 107), -INT8_C(   4), -INT8_C(  47), -INT8_C(  67), -INT8_C( 112), -INT8_C(  85), -INT8_C(  28),  INT8_C(  54) },
      UINT64_C( 2086301257730004195),
      {  INT8_C(   4), -INT8_C(   9), -INT8_C( 101),  INT8_C(   3),  INT8_C(  22),  INT8_C(  11), -INT8_C(  34),  INT8_C(  98),
        -INT8_C(  70), -INT8_C(  50), -INT8_C(  24), -INT8_C(  86),  INT8_C(  98),  INT8_C(  85),  INT8_C( 121), -INT8_C(   9),
         INT8_C(  81),  INT8_C(  74), -INT8_C(  75), -INT8_C(  31), -INT8_C(  11), -INT8_C( 103),  INT8_C(  24), -INT8_C(  40),
         INT8_C(  46), -INT8_C( 118), -INT8_C( 119),  INT8_C(  30), -INT8_C( 110),  INT8_C( 125),  INT8_C(  58), -INT8_C( 106),
         INT8_C( 117), -INT8_C(  43), -INT8_C( 103), -INT8_C( 117), -INT8_C(  32),  INT8_C( 119), -INT8_C(  19), -INT8_C( 101),
         INT8_C(  69), -INT8_C(  43),  INT8_C(  69), -INT8_C(  88),  INT8_C(  43), -INT8_C(  66), -INT8_C(  97),  INT8_C( 124),
         INT8_C(   8),  INT8_C(  84),  INT8_C(  94), -INT8_C(   2), -INT8_C(  18),  INT8_C( 118), -INT8_C(  42),  INT8_C(  28),
         INT8_C(   0),  INT8_C(  96),  INT8_C(  58), -INT8_C( 110), -INT8_C(  35),  INT8_C( 116),  INT8_C(  40),  INT8_C(  82) },
      {  INT8_C(  73), -INT8_C(  63), -INT8_C(  34),  INT8_C(  42),  INT8_C(  57), -INT8_C(  53), -INT8_C(  59),  INT8_C( 126),
        -INT8_C(  95),  INT8_C(  10),  INT8_C(  38), -INT8_C(  52), -INT8_C(  55), -INT8_C(  58),  INT8_C(  72), -INT8_C(  47),
         INT8_C(  26), -INT8_C(  90), -INT8_C(  49),  INT8_C(   8),  INT8_C(  28), -INT8_C(  90),  INT8_C(  36),  INT8_C(  29),
         INT8_C(   6),  INT8_C(  94), -INT8_C(  81), -INT8_C(  29), -INT8_C(  46), -INT8_C(  40),  INT8_C(  54),  INT8_C(  28),
        -INT8_C( 103),  INT8_C(  20),  INT8_C(  70), -INT8_C(  46), -INT8_C(  33),  INT8_C(  11),  INT8_C(  81),      INT8_MIN,
         INT8_C(  21),  INT8_C( 119),  INT8_C(  76), -INT8_C(  34),  INT8_C(  61), -INT8_C( 107), -INT8_C(  80),  INT8_C(  88),
         INT8_C(  59),      INT8_MAX,  INT8_C(  96),  INT8_C(  88),  INT8_C(  37), -INT8_C( 123),  INT8_C( 117),  INT8_C(  43),
        -INT8_C(  29),  INT8_C(  36),  INT8_C(  15), -INT8_C(  74), -INT8_C(   4),  INT8_C(  69), -INT8_C(  46), -INT8_C( 106) },
      {  INT8_C(  73), -INT8_C(   9),  INT8_C(  93), -INT8_C( 119), -INT8_C( 107),  INT8_C(  11), -INT8_C(  34),  INT8_C( 126),
         INT8_C( 117), -INT8_C(  48),  INT8_C(  38), -INT8_C(  64),  INT8_C(  98),  INT8_C(   2),  INT8_C(  27), -INT8_C(   9),
        -INT8_C(  36),  INT8_C(  74),  INT8_C(  64),  INT8_C(  20),  INT8_C(  28), -INT8_C(  90),  INT8_C(  36), -INT8_C(  25),
         INT8_C(  46),  INT8_C(   0), -INT8_C( 114),  INT8_C(   2), -INT8_C(  46),  INT8_C( 125), -INT8_C(  38),  INT8_C(  28),
        -INT8_C(  90),  INT8_C(  55), -INT8_C(  80),  INT8_C(  60), -INT8_C(  32),  INT8_C( 119),  INT8_C(  81), -INT8_C( 101),
         INT8_C(  69),  INT8_C( 119),  INT8_C(  76),      INT8_MAX, -INT8_C(   1),  INT8_C(  31),  INT8_C( 111), -INT8_C(  37),
         INT8_C(  75), -INT8_C(  81),  INT8_C(  96), -INT8_C( 122),  INT8_C(  37),  INT8_C( 118),  INT8_C( 117),  INT8_C(  43),
        -INT8_C( 107), -INT8_C(   4),  INT8_C(  58), -INT8_C(  74), -INT8_C(   4), -INT8_C(  85), -INT8_C(  28),  INT8_C(  54) } },
    { {  INT8_C(  89),  INT8_C(  24),  INT8_C( 104),  INT8_C(  56),  INT8_C(  35), -INT8_C(  71), -INT8_C(  71),  INT8_C(  56),
         INT8_C(  49),  INT8_C(   5),  INT8_C(  23),  INT8_C( 110), -INT8_C( 102), -INT8_C(  57), -INT8_C(  58), -INT8_C(  42),
         INT8_C(  70),  INT8_C(  39),  INT8_C(  46),  INT8_C( 108), -INT8_C(  84), -INT8_C(  93), -INT8_C( 105), -INT8_C( 113),
        -INT8_C(  57), -INT8_C(  90),  INT8_C(  69), -INT8_C(  60), -INT8_C(  21),  INT8_C(  23),  INT8_C(  90),  INT8_C(  68),
         INT8_C(  47), -INT8_C(  62),  INT8_C( 125),  INT8_C(  82),  INT8_C( 124),  INT8_C(  54), -INT8_C( 117), -INT8_C(  83),
         INT8_C(  59), -INT8_C(  94),  INT8_C(  27), -INT8_C(  42),  INT8_C( 105), -INT8_C(  30), -INT8_C(  84), -INT8_C(  81),
         INT8_C(   9), -INT8_C(  38),  INT8_C(  27), -INT8_C(  75),  INT8_C( 125), -INT8_C(  77),  INT8_C(  68),  INT8_C(  68),
         INT8_C(  89), -INT8_C( 118),  INT8_C(   8),  INT8_C(  69), -INT8_C(  95),  INT8_C(  98), -INT8_C( 119), -INT8_C(  47) },
      UINT64_C( 8669057908159481381),
      {  INT8_C(  80),  INT8_C( 105),  INT8_C(  78), -INT8_C(  71),  INT8_C(  75), -INT8_C(   6),  INT8_C( 105),  INT8_C(  84),
        -INT8_C(  44), -INT8_C( 124),  INT8_C(   9),  INT8_C(  81),  INT8_C(  55),  INT8_C(  78), -INT8_C( 107), -INT8_C( 111),
        -INT8_C(  40), -INT8_C(  98), -INT8_C(  42),  INT8_C( 121),  INT8_C(   0),  INT8_C(  95),  INT8_C(  74),  INT8_C(  37),
         INT8_C( 102),  INT8_C( 110), -INT8_C(  58), -INT8_C(  94),  INT8_C(  28),  INT8_C(  20),  INT8_C(  26),  INT8_C( 109),
         INT8_C( 126),  INT8_C( 104),  INT8_C(  38), -INT8_C(  55),  INT8_C(  98), -INT8_C( 113),  INT8_C(  30),  INT8_C(  54),
         INT8_C(  20),  INT8_C(  39), -INT8_C( 121),  INT8_C(  75),  INT8_C( 117),  INT8_C(  29), -INT8_C(  36),  INT8_C(  77),
        -INT8_C(  69), -INT8_C(  78), -INT8_C(  57), -INT8_C(  69),  INT8_C(  18),  INT8_C(  17), -INT8_C(  31),  INT8_C( 120),
             INT8_MAX, -INT8_C(  89),  INT8_C(  26), -INT8_C( 100), -INT8_C(  68),  INT8_C(  53),  INT8_C(   9),  INT8_C(  58) },
      { -INT8_C(  99),  INT8_C(  47),  INT8_C(   3),  INT8_C(   0), -INT8_C(  65),  INT8_C(  33),  INT8_C(  54), -INT8_C(  45),
         INT8_C(  73), -INT8_C(  66),  INT8_C(  30), -INT8_C(  66), -INT8_C(  37), -INT8_C(   5),  INT8_C(  12), -INT8_C( 106),
        -INT8_C(  83), -INT8_C(  45),  INT8_C(  81), -INT8_C(  65), -INT8_C(  28),  INT8_C(  50),  INT8_C(  55),  INT8_C( 100),
        -INT8_C(  38),  INT8_C(  82),  INT8_C(   0), -INT8_C( 106), -INT8_C( 121),  INT8_C(   9), -INT8_C(  48),  INT8_C(  36),
         INT8_C(  56), -INT8_C(  45),  INT8_C(  36), -INT8_C(   9), -INT8_C(  11),  INT8_C(  91), -INT8_C(  54),  INT8_C(  62),
         INT8_C(  25), -INT8_C(  23), -INT8_C(   4), -INT8_C(  12), -INT8_C(  28),  INT8_C(   8), -INT8_C( 118), -INT8_C( 111),
        -INT8_C(  37), -INT8_C(  37),  INT8_C(  81), -INT8_C(  64),  INT8_C(  14), -INT8_C( 120),  INT8_C(  36), -INT8_C(  24),
        -INT8_C(  38),  INT8_C(  36),  INT8_C( 126),  INT8_C(  97),  INT8_C(  45),  INT8_C(  78), -INT8_C( 122),  INT8_C( 101) },
      {  INT8_C(  80),  INT8_C(  24),  INT8_C(  78),  INT8_C(  56),  INT8_C(  35),  INT8_C(  33), -INT8_C(  71),  INT8_C(  56),
         INT8_C(  49), -INT8_C(  66),  INT8_C(  30),  INT8_C( 110), -INT8_C( 102), -INT8_C(  57), -INT8_C(  58), -INT8_C(  42),
        -INT8_C(  40), -INT8_C(  45),  INT8_C(  46),  INT8_C( 108), -INT8_C(  84),  INT8_C(  95), -INT8_C( 105), -INT8_C( 113),
         INT8_C( 102), -INT8_C(  90),  INT8_C(  69), -INT8_C(  60), -INT8_C(  21),  INT8_C(  20),  INT8_C(  90),  INT8_C( 109),
         INT8_C(  47), -INT8_C(  62),  INT8_C(  38), -INT8_C(   9),  INT8_C(  98),  INT8_C(  91), -INT8_C( 117), -INT8_C(  83),
         INT8_C(  59),  INT8_C(  39), -INT8_C(   4),  INT8_C(  75),  INT8_C( 105),  INT8_C(  29), -INT8_C(  84),  INT8_C(  77),
         INT8_C(   9), -INT8_C(  37),  INT8_C(  81), -INT8_C(  64),  INT8_C( 125), -INT8_C(  77),  INT8_C(  36),  INT8_C(  68),
         INT8_C(  89), -INT8_C( 118),  INT8_C(   8),  INT8_C(  97),  INT8_C(  45),  INT8_C(  78),  INT8_C(   9), -INT8_C(  47) } },
    { {  INT8_C(  33), -INT8_C(  86),  INT8_C(  93),  INT8_C(  22),  INT8_C(   5),  INT8_C(  39),  INT8_C(  84),  INT8_C(  30),
         INT8_C(  16),  INT8_C(  81),  INT8_C(  18), -INT8_C(  12),  INT8_C(  89), -INT8_C( 100), -INT8_C( 122),  INT8_C(  53),
         INT8_C( 120), -INT8_C(  41), -INT8_C(  11), -INT8_C( 122),  INT8_C(  95),  INT8_C(  25),  INT8_C( 110),  INT8_C(  58),
         INT8_C(  61), -INT8_C(  20), -INT8_C( 101),  INT8_C( 106),  INT8_C(  58),  INT8_C(  33), -INT8_C(  49),  INT8_C(  91),
        -INT8_C(  52),  INT8_C(  44),  INT8_C( 114), -INT8_C(  47),  INT8_C(  84), -INT8_C(  58), -INT8_C(  16),  INT8_C( 100),
         INT8_C(  23),  INT8_C(   2),  INT8_C(  89),  INT8_C( 113), -INT8_C(  97), -INT8_C(  33), -INT8_C(  90),  INT8_C(  23),
        -INT8_C(  74), -INT8_C( 101), -INT8_C(  99),  INT8_C(  21), -INT8_C(  76),  INT8_C(  11),  INT8_C(  79), -INT8_C(  15),
        -INT8_C(   9), -INT8_C(  21),  INT8_C(  91),  INT8_C(  49),  INT8_C(  12),  INT8_C(  42), -INT8_C( 116), -INT8_C(  40) },
      UINT64_C(15857062986774150743),
      { -INT8_C( 100),  INT8_C( 104),  INT8_C(  77),  INT8_C(  59),  INT8_C(  71), -INT8_C(  13),  INT8_C(  82), -INT8_C(   3),
        -INT8_C( 114), -INT8_C(  17),  INT8_C(  19),  INT8_C(  66), -INT8_C(   6),  INT8_C(  98),  INT8_C(  51), -INT8_C(  15),
         INT8_C(  77), -INT8_C( 114),  INT8_C(  34),  INT8_C(  90), -INT8_C(  71), -INT8_C(  81),  INT8_C(  50),  INT8_C(  16),
        -INT8_C(  83), -INT8_C(  36), -INT8_C(  69),  INT8_C( 114),  INT8_C( 118), -INT8_C(  54),  INT8_C(  79),  INT8_C(  19),
         INT8_C(  51), -INT8_C( 100),  INT8_C(  78),  INT8_C( 122), -INT8_C( 112), -INT8_C(  95),  INT8_C( 120),  INT8_C(  30),
        -INT8_C( 112), -INT8_C( 117),  INT8_C(  97), -INT8_C( 117), -INT8_C(  19), -INT8_C( 108),  INT8_C( 124),  INT8_C(  59),
         INT8_C(  35), -INT8_C(  97), -INT8_C( 107), -INT8_C(  36),  INT8_C(  78), -INT8_C(  57), -INT8_C(  20), -INT8_C(   5),
        -INT8_C(  92), -INT8_C(  89),  INT8_C( 110),  INT8_C(  26),  INT8_C( 113), -INT8_C(  67),  INT8_C(  45), -INT8_C(  92) },
      {  INT8_C(  89),  INT8_C( 124),  INT8_C(  31), -INT8_C(  23),  INT8_C(  29), -INT8_C( 105),  INT8_C(   8), -INT8_C(  83),
         INT8_C(  34),  INT8_C( 105),  INT8_C(  56),  INT8_C(  15), -INT8_C(   3), -INT8_C(  75),  INT8_C(  74),  INT8_C(  32),
         INT8_C(  84), -INT8_C(  33), -INT8_C(   4), -INT8_C(  94), -INT8_C(  89), -INT8_C(  24), -INT8_C(  99),  INT8_C(  75),
        -INT8_C( 113),  INT8_C(  11),  INT8_C( 101),  INT8_C(   1), -INT8_C(  56), -INT8_C( 109), -INT8_C(  91),  INT8_C(  34),
         INT8_C(  15), -INT8_C(  60),  INT8_C(  11),  INT8_C(  44),  INT8_C(  91),  INT8_C(  19), -INT8_C(  39),  INT8_C( 125),
         INT8_C( 124),  INT8_C(  18), -INT8_C( 115),  INT8_C( 122), -INT8_C(  57), -INT8_C(  41), -INT8_C( 102),  INT8_C(  27),
        -INT8_C(  73), -INT8_C( 105), -INT8_C(  67),  INT8_C(  94),      INT8_MAX,  INT8_C(  90), -INT8_C(  87),  INT8_C(  15),
         INT8_C( 102),  INT8_C(  14),  INT8_C(  16),  INT8_C(  46), -INT8_C(  95), -INT8_C(  75),  INT8_C(  80), -INT8_C(  80) },
      {  INT8_C(  89),  INT8_C( 124),  INT8_C(  77),  INT8_C(  22),  INT8_C(  71),  INT8_C(  39),  INT8_C(  82),  INT8_C(  30),
         INT8_C(  16),  INT8_C( 105),  INT8_C(  56),  INT8_C(  66), -INT8_C(   3),  INT8_C(  98),  INT8_C(  74),  INT8_C(  32),
         INT8_C( 120), -INT8_C(  33), -INT8_C(  11),  INT8_C(  90),  INT8_C(  95), -INT8_C(  24),  INT8_C( 110),  INT8_C(  75),
        -INT8_C(  83),  INT8_C(  11), -INT8_C( 101),  INT8_C( 114),  INT8_C(  58), -INT8_C(  54), -INT8_C(  49),  INT8_C(  34),
         INT8_C(  51),  INT8_C(  44),  INT8_C(  78), -INT8_C(  47),  INT8_C(  84), -INT8_C(  58),  INT8_C( 120),  INT8_C( 125),
         INT8_C(  23),  INT8_C(  18),  INT8_C(  89),  INT8_C( 122), -INT8_C(  19), -INT8_C(  33), -INT8_C(  90),  INT8_C(  59),
         INT8_C(  35), -INT8_C(  97), -INT8_C(  67),  INT8_C(  94), -INT8_C(  76),  INT8_C(  11),  INT8_C(  79), -INT8_C(  15),
        -INT8_C(   9), -INT8_C(  21),  INT8_C( 110),  INT8_C(  46),  INT8_C( 113),  INT8_C(  42),  INT8_C(  80), -INT8_C(  80) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi8(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_epi8(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_epi8");
    easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask64 k;
    const int8_t a[64];
    const int8_t b[64];
    const int8_t r[64];
  } test_vec[] = {
    { UINT64_C(15100473841603180603),
      { -INT8_C(  76), -INT8_C(  30),  INT8_C(   8),  INT8_C(  48),  INT8_C(  20), -INT8_C( 118), -INT8_C(  64), -INT8_C( 125),
        -INT8_C(  42), -INT8_C(   8), -INT8_C(  20),  INT8_C(  95),  INT8_C(  50), -INT8_C(  41), -INT8_C(  18), -INT8_C(  33),
        -INT8_C(  10), -INT8_C( 104),  INT8_C(  32), -INT8_C( 105),  INT8_C( 107),  INT8_C( 117), -INT8_C(   9), -INT8_C(  90),
         INT8_C( 109), -INT8_C(  52), -INT8_C(  86),  INT8_C(  97),  INT8_C( 116),  INT8_C(  58),  INT8_C(  50),  INT8_C(  40),
         INT8_C(  28),  INT8_C(  58),  INT8_C(  89),  INT8_C(  48), -INT8_C(  59),  INT8_C(  25), -INT8_C(  77), -INT8_C( 101),
         INT8_C(  17), -INT8_C(  96), -INT8_C(   5),  INT8_C(  68),  INT8_C( 119), -INT8_C(  23),  INT8_C(  35),  INT8_C( 110),
        -INT8_C( 127),  INT8_C(  67),  INT8_C(   5), -INT8_C(  19), -INT8_C(  72), -INT8_C(   4), -INT8_C( 109),  INT8_C(  37),
        -INT8_C(  56),  INT8_C(  62), -INT8_C( 122),  INT8_C(  61),  INT8_C( 120), -INT8_C(  72),  INT8_C( 101), -INT8_C( 108) },
      { -INT8_C(  14), -INT8_C(  66), -INT8_C(  60), -INT8_C(  73), -INT8_C(  41),  INT8_C( 120),  INT8_C(  83), -INT8_C(  23),
         INT8_C(  24),  INT8_C(  78),  INT8_C(  45), -INT8_C( 113),  INT8_C(  55),  INT8_C(  80), -INT8_C(   3), -INT8_C(  71),
        -INT8_C( 109),  INT8_C(   2), -INT8_C(  90),  INT8_C(  75), -INT8_C(   2),  INT8_C(  57),  INT8_C( 112), -INT8_C(  57),
         INT8_C( 119), -INT8_C(  10),  INT8_C(   4), -INT8_C(  17), -INT8_C(  82),  INT8_C( 105), -INT8_C( 125), -INT8_C(  96),
         INT8_C(  40),  INT8_C(  72),  INT8_C(  88), -INT8_C(   1), -INT8_C(  64), -INT8_C(  85), -INT8_C(  24), -INT8_C(  40),
        -INT8_C(   7),  INT8_C(  21),  INT8_C( 103),  INT8_C(  48),  INT8_C( 101),  INT8_C( 101), -INT8_C(  23), -INT8_C(   8),
         INT8_C( 103), -INT8_C( 113),  INT8_C(  67),  INT8_C( 102), -INT8_C(  55), -INT8_C(  77),  INT8_C(  45),  INT8_C(  64),
        -INT8_C(  87),  INT8_C(  49),  INT8_C(  48),  INT8_C(  87), -INT8_C( 102), -INT8_C(  77), -INT8_C(   8), -INT8_C(  62) },
      { -INT8_C(  14), -INT8_C(  30),  INT8_C(   0),  INT8_C(  48),  INT8_C(  20),  INT8_C( 120),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  95),  INT8_C(  55),  INT8_C(  80), -INT8_C(   3), -INT8_C(  33),
        -INT8_C(  10),  INT8_C(   0),  INT8_C(  32),  INT8_C(   0),  INT8_C( 107),  INT8_C(   0),  INT8_C( 112), -INT8_C(  57),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   4),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  89),  INT8_C(   0), -INT8_C(  59),  INT8_C(  25), -INT8_C(  24), -INT8_C(  40),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  68),  INT8_C(   0),  INT8_C( 101),  INT8_C(   0),  INT8_C( 110),
         INT8_C( 103),  INT8_C(  67),  INT8_C(  67),  INT8_C( 102),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  64),
        -INT8_C(  56),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 120),  INT8_C(   0),  INT8_C( 101), -INT8_C(  62) } },
    { UINT64_C(17623617764994470139),
      { -INT8_C(  64), -INT8_C(   5),  INT8_C(  36),  INT8_C(  37),  INT8_C(  96),  INT8_C(  14),  INT8_C(  30), -INT8_C(  57),
        -INT8_C(  99),  INT8_C(  97),  INT8_C(  45),  INT8_C( 102),  INT8_C(  21),  INT8_C(  90), -INT8_C(  89), -INT8_C(  66),
        -INT8_C( 117), -INT8_C(  41),  INT8_C(  22),  INT8_C(  38), -INT8_C( 118),  INT8_C(  14), -INT8_C(  24), -INT8_C( 122),
         INT8_C(  94), -INT8_C(  86),  INT8_C(  65),  INT8_C(  89),  INT8_C(  85), -INT8_C(  43),  INT8_C(  77),  INT8_C(  21),
        -INT8_C(  48),  INT8_C( 113),  INT8_C(  58),  INT8_C(  48),      INT8_MAX,  INT8_C(  88), -INT8_C(   9),  INT8_C(  29),
        -INT8_C(  70),  INT8_C(  37), -INT8_C( 125), -INT8_C(  49),      INT8_MAX,  INT8_C(  42), -INT8_C( 115),  INT8_C(  11),
         INT8_C(   1), -INT8_C(  93),  INT8_C(  49), -INT8_C( 116), -INT8_C(  79),  INT8_C(  25),  INT8_C(  18),  INT8_C(  15),
        -INT8_C(  60),  INT8_C(  83),  INT8_C( 104),  INT8_C(  25),  INT8_C(  40), -INT8_C(  75),  INT8_C(  46), -INT8_C(   8) },
      {  INT8_C(  39),  INT8_C( 104),  INT8_C(  40), -INT8_C(  90), -INT8_C(  63),  INT8_C(  32), -INT8_C(  61),  INT8_C( 123),
         INT8_C(  69),  INT8_C(  71),  INT8_C(  74), -INT8_C(  60),  INT8_C( 113), -INT8_C(  41), -INT8_C(  49),  INT8_C( 115),
         INT8_C( 123),  INT8_C(   0), -INT8_C(   1),  INT8_C(  44),  INT8_C(  26),  INT8_C(  17),  INT8_C(  60), -INT8_C(  34),
         INT8_C( 100), -INT8_C(  92), -INT8_C(   9), -INT8_C( 115),  INT8_C(  90),  INT8_C(  37), -INT8_C( 123), -INT8_C( 127),
        -INT8_C( 115), -INT8_C(  82),  INT8_C(  39),  INT8_C(  78), -INT8_C(  50), -INT8_C(  21), -INT8_C(  55),  INT8_C(  19),
         INT8_C(  50),  INT8_C(  19), -INT8_C(  41), -INT8_C(  93), -INT8_C(  21), -INT8_C(  89),  INT8_C(  22),  INT8_C( 102),
        -INT8_C(  89),  INT8_C(  21), -INT8_C( 110), -INT8_C(  63),  INT8_C(  38), -INT8_C(  50), -INT8_C(  97), -INT8_C( 117),
         INT8_C( 115), -INT8_C( 106),  INT8_C(  24), -INT8_C(  51), -INT8_C(  69), -INT8_C(  99),  INT8_C(  78),  INT8_C(  73) },
      {  INT8_C(  39),  INT8_C( 104),  INT8_C(   0),  INT8_C(  37),  INT8_C(  96),  INT8_C(  32),  INT8_C(  30),  INT8_C( 123),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 113),  INT8_C(   0), -INT8_C(  49),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  60), -INT8_C(  34),
         INT8_C( 100), -INT8_C(  86),  INT8_C(   0),  INT8_C(  89),  INT8_C(  90),  INT8_C(  37),  INT8_C(   0),  INT8_C(  21),
        -INT8_C(  48),  INT8_C( 113),  INT8_C(   0),  INT8_C(  78),      INT8_MAX,  INT8_C(  88), -INT8_C(   9),  INT8_C(  29),
         INT8_C(   0),  INT8_C(  37),  INT8_C(   0), -INT8_C(  49),  INT8_C(   0),  INT8_C(  42),  INT8_C(   0),  INT8_C( 102),
         INT8_C(   1),  INT8_C(  21),  INT8_C(   0),  INT8_C(   0),  INT8_C(  38),  INT8_C(   0),  INT8_C(   0),  INT8_C(  15),
         INT8_C(   0),  INT8_C(   0),  INT8_C( 104),  INT8_C(   0),  INT8_C(  40), -INT8_C(  75),  INT8_C(  78),  INT8_C(  73) } },
    { UINT64_C(10532900693886858571),
      {  INT8_C( 116),  INT8_C(   4),  INT8_C(  54),  INT8_C(  95), -INT8_C(  85),  INT8_C(  76), -INT8_C(  59),  INT8_C(  82),
         INT8_C(  98),  INT8_C(  88),  INT8_C(  20), -INT8_C( 120),  INT8_C(  38), -INT8_C(  77),  INT8_C(  19), -INT8_C( 103),
         INT8_C(  74),  INT8_C(  43),  INT8_C( 102),  INT8_C(   5), -INT8_C(  55), -INT8_C(  76),  INT8_C(  78),  INT8_C(  20),
         INT8_C(  42), -INT8_C(  26),  INT8_C(  46), -INT8_C( 118),  INT8_C(  71),  INT8_C(  90),  INT8_C(  29), -INT8_C(  69),
         INT8_C(  94),  INT8_C(  83),  INT8_C(  27),  INT8_C(   9), -INT8_C(  97), -INT8_C(  32),  INT8_C(  92),  INT8_C(   1),
         INT8_C(  56),  INT8_C( 112), -INT8_C( 118),  INT8_C(  95),  INT8_C(  35), -INT8_C(  99), -INT8_C(   8),  INT8_C( 109),
        -INT8_C(  55),  INT8_C(  95),  INT8_C( 115), -INT8_C( 110),  INT8_C(  19), -INT8_C(  63), -INT8_C(  90),  INT8_C(  61),
        -INT8_C(  89), -INT8_C(  44), -INT8_C(  56), -INT8_C(  18),  INT8_C(  47), -INT8_C(  27), -INT8_C(  86), -INT8_C( 115) },
      {  INT8_C(  56), -INT8_C(  59), -INT8_C( 105), -INT8_C(  41), -INT8_C(  91), -INT8_C(  13), -INT8_C(  39), -INT8_C(  34),
         INT8_C(  99),  INT8_C(  99),  INT8_C(  61), -INT8_C( 122),  INT8_C(   0),  INT8_C(  53), -INT8_C(  12), -INT8_C(  55),
        -INT8_C( 108),  INT8_C( 103),  INT8_C(  91), -INT8_C(  88),  INT8_C(  40),  INT8_C(   2), -INT8_C(  27), -INT8_C(  48),
        -INT8_C(  42), -INT8_C(  83), -INT8_C(  66),  INT8_C(   5), -INT8_C( 110),  INT8_C( 104), -INT8_C( 109), -INT8_C(  54),
         INT8_C(  45),  INT8_C(  42), -INT8_C(  94), -INT8_C(  45),  INT8_C(  29),  INT8_C( 123), -INT8_C(  79),      INT8_MIN,
        -INT8_C(  34), -INT8_C(  18),  INT8_C(   6), -INT8_C(  34),  INT8_C(  35), -INT8_C(   6), -INT8_C(  88), -INT8_C(  72),
         INT8_C(  97),  INT8_C(   3),  INT8_C(  96), -INT8_C( 118),  INT8_C(   5),  INT8_C(  69),  INT8_C(  90), -INT8_C(  36),
        -INT8_C(  13),  INT8_C(  24), -INT8_C(  31), -INT8_C( 123), -INT8_C( 127),  INT8_C( 116),  INT8_C(  80), -INT8_C(  82) },
      {  INT8_C( 116),  INT8_C(   4),  INT8_C(   0),  INT8_C(  95),  INT8_C(   0),  INT8_C(   0), -INT8_C(  39),  INT8_C(   0),
         INT8_C(  99),  INT8_C(   0),  INT8_C(  61),  INT8_C(   0),  INT8_C(  38),  INT8_C(  53),  INT8_C(  19),  INT8_C(   0),
         INT8_C(  74),  INT8_C( 103),  INT8_C( 102),  INT8_C(   0),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),  INT8_C(  20),
         INT8_C(  42),  INT8_C(   0),  INT8_C(   0),  INT8_C(   5),  INT8_C(  71),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 123),  INT8_C(  92),  INT8_C(   0),
         INT8_C(  56),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(   6), -INT8_C(   8),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C( 115), -INT8_C( 110),  INT8_C(   0),  INT8_C(  69),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  24),  INT8_C(   0),  INT8_C(   0),  INT8_C(  47),  INT8_C(   0),  INT8_C(   0), -INT8_C(  82) } },
    { UINT64_C( 5420982023349203614),
      {  INT8_C(  32),  INT8_C(  66),  INT8_C(  41),  INT8_C(  68),  INT8_C(  60), -INT8_C(  47), -INT8_C(   4), -INT8_C(  98),
        -INT8_C(  43),  INT8_C(  92),  INT8_C(  40), -INT8_C(  38), -INT8_C(  95), -INT8_C( 126), -INT8_C(  74), -INT8_C( 108),
        -INT8_C( 102), -INT8_C( 104),  INT8_C(  26),  INT8_C(  27),  INT8_C(  12),  INT8_C( 106), -INT8_C(  54), -INT8_C(  85),
         INT8_C(  92),  INT8_C(  75),  INT8_C( 102), -INT8_C(  55),  INT8_C( 126), -INT8_C(  94),  INT8_C(  20), -INT8_C(  98),
        -INT8_C(  28),  INT8_C(  61), -INT8_C(  30),  INT8_C(  32),  INT8_C(  15), -INT8_C(  34), -INT8_C(  66), -INT8_C(  28),
         INT8_C(  58), -INT8_C(  26), -INT8_C(  66), -INT8_C(  36),  INT8_C( 104),  INT8_C( 117),  INT8_C( 112),  INT8_C(   3),
         INT8_C(  13), -INT8_C( 118),  INT8_C(  30),  INT8_C(  25), -INT8_C(  12), -INT8_C(  24), -INT8_C(  60),  INT8_C(  80),
         INT8_C(  52),  INT8_C(  43),  INT8_C(  25), -INT8_C(  78), -INT8_C(  51),  INT8_C(  45),  INT8_C(  80), -INT8_C(  79) },
      {  INT8_C( 107),  INT8_C(  51), -INT8_C(  47),  INT8_C( 122),  INT8_C(  17), -INT8_C( 112),  INT8_C(  94),  INT8_C(  76),
         INT8_C( 118),  INT8_C(  28),  INT8_C(  40), -INT8_C(  33), -INT8_C( 111), -INT8_C( 104), -INT8_C(  30), -INT8_C(  98),
         INT8_C(  35),  INT8_C(   0), -INT8_C(  72),  INT8_C(  23), -INT8_C(  23),  INT8_C( 124),  INT8_C( 104),  INT8_C(  29),
        -INT8_C(  89), -INT8_C( 127), -INT8_C(  49),  INT8_C( 116), -INT8_C(  81),  INT8_C(  31),  INT8_C(  37),  INT8_C(  26),
         INT8_C(  82), -INT8_C(   9), -INT8_C( 108),  INT8_C( 100), -INT8_C( 121), -INT8_C(  14), -INT8_C(  80), -INT8_C(   3),
         INT8_C(  14), -INT8_C(  40), -INT8_C(  36), -INT8_C(  96),  INT8_C( 112), -INT8_C(  66),  INT8_C(  62), -INT8_C( 109),
        -INT8_C(  65), -INT8_C(  10), -INT8_C(  85), -INT8_C(  88),  INT8_C( 115),  INT8_C(  19), -INT8_C(  59),  INT8_C(  26),
        -INT8_C( 108), -INT8_C( 108), -INT8_C( 113),  INT8_C(  67), -INT8_C(  77), -INT8_C(  76),  INT8_C(  93),  INT8_C(   6) },
      {  INT8_C(   0),  INT8_C(  66),  INT8_C(  41),  INT8_C( 122),  INT8_C(  60),  INT8_C(   0),  INT8_C(   0),  INT8_C(  76),
         INT8_C(   0),  INT8_C(  92),  INT8_C(   0),  INT8_C(   0), -INT8_C(  95), -INT8_C( 104), -INT8_C(  30), -INT8_C(  98),
         INT8_C(  35),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  29),
         INT8_C(  92),  INT8_C(  75),  INT8_C(   0),  INT8_C( 116),  INT8_C( 126),  INT8_C(  31),  INT8_C(   0),  INT8_C(  26),
         INT8_C(  82),  INT8_C(   0), -INT8_C(  30),  INT8_C( 100),  INT8_C(   0), -INT8_C(  14), -INT8_C(  66),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  26),  INT8_C(   0),  INT8_C(   0),  INT8_C( 112),  INT8_C( 117),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  13), -INT8_C(  10),  INT8_C(   0),  INT8_C(  25),  INT8_C( 115),  INT8_C(  19),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  52),  INT8_C(  43),  INT8_C(   0),  INT8_C(  67),  INT8_C(   0),  INT8_C(   0),  INT8_C(  93),  INT8_C(   0) } },
    { UINT64_C(17451478119166439851),
      { -INT8_C(  14),  INT8_C(  12), -INT8_C( 110),  INT8_C(  98), -INT8_C(  53), -INT8_C(  48), -INT8_C(  10), -INT8_C( 118),
        -INT8_C(  57), -INT8_C(  95),  INT8_C(  50),  INT8_C(  58), -INT8_C(  76), -INT8_C(   9),  INT8_C(  84),  INT8_C(  72),
        -INT8_C( 117), -INT8_C(  29), -INT8_C( 116),  INT8_C(  62), -INT8_C( 104), -INT8_C(  23),  INT8_C(  68),  INT8_C(  67),
        -INT8_C(  37), -INT8_C(  82),  INT8_C( 118), -INT8_C(  66), -INT8_C(  56), -INT8_C(  90), -INT8_C(  80), -INT8_C(  70),
        -INT8_C(  78),  INT8_C(  66),  INT8_C(  29),  INT8_C( 125),  INT8_C(  19),  INT8_C(  19),  INT8_C(   7), -INT8_C(  38),
        -INT8_C(  76),  INT8_C(  57),  INT8_C(  20),  INT8_C( 104),  INT8_C(  48),  INT8_C( 104), -INT8_C(  80), -INT8_C(  69),
         INT8_C(  76),  INT8_C(  60), -INT8_C(   6), -INT8_C(  28),  INT8_C(  38),  INT8_C(  62),  INT8_C(  39),  INT8_C(   1),
        -INT8_C(  19), -INT8_C(  99), -INT8_C(  65), -INT8_C(  75),  INT8_C(  67),  INT8_C( 112),  INT8_C( 112), -INT8_C(  10) },
      { -INT8_C(  78), -INT8_C( 115),  INT8_C( 115), -INT8_C(  59), -INT8_C(  96),  INT8_C( 123), -INT8_C(  97),  INT8_C(  84),
        -INT8_C(  76), -INT8_C(  77), -INT8_C(  68), -INT8_C(  27),  INT8_C(  28),  INT8_C( 108), -INT8_C(  96),  INT8_C( 104),
        -INT8_C(  87), -INT8_C( 102),  INT8_C(  76), -INT8_C(  49), -INT8_C(  39),  INT8_C( 115), -INT8_C(  48), -INT8_C(  58),
         INT8_C(  17), -INT8_C( 113),  INT8_C( 123),  INT8_C(  84), -INT8_C(   1), -INT8_C(  21),  INT8_C(  74), -INT8_C(  78),
         INT8_C( 120), -INT8_C(  66),  INT8_C( 119),  INT8_C(  24),  INT8_C(  57),  INT8_C(  23),  INT8_C( 108), -INT8_C(  19),
        -INT8_C(  54),  INT8_C(  40), -INT8_C(  46), -INT8_C(  26), -INT8_C( 107),  INT8_C( 115),  INT8_C(  78),  INT8_C(  62),
         INT8_C(  13), -INT8_C( 102),  INT8_C(  13), -INT8_C(  26),  INT8_C(  14), -INT8_C(  35), -INT8_C(  84),  INT8_C(  31),
         INT8_C( 108),  INT8_C(  40),  INT8_C( 115),  INT8_C( 108),  INT8_C(  19), -INT8_C(  66),  INT8_C(  30), -INT8_C( 116) },
      { -INT8_C(  14),  INT8_C(  12),  INT8_C(   0),  INT8_C(  98),  INT8_C(   0),  INT8_C( 123),  INT8_C(   0),  INT8_C(  84),
        -INT8_C(  57),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  28),  INT8_C( 108),  INT8_C(  84),  INT8_C( 104),
         INT8_C(   0), -INT8_C(  29),  INT8_C(   0),  INT8_C(  62),  INT8_C(   0),  INT8_C( 115),  INT8_C(  68),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  82),  INT8_C(   0),  INT8_C(   0), -INT8_C(   1), -INT8_C(  21),  INT8_C(   0),  INT8_C(   0),
         INT8_C( 120),  INT8_C(  66),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  23),  INT8_C( 108), -INT8_C(  19),
         INT8_C(   0),  INT8_C(  57),  INT8_C(   0),  INT8_C( 104),  INT8_C(  48),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  38),  INT8_C(  62),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0),  INT8_C(  67),  INT8_C( 112),  INT8_C( 112), -INT8_C(  10) } },
    { UINT64_C( 8620472070220060028),
      {  INT8_C(  57),  INT8_C( 117),  INT8_C(  93), -INT8_C(  50), -INT8_C(  24), -INT8_C(  84),  INT8_C(  12), -INT8_C(  11),
         INT8_C(  70),  INT8_C(  25), -INT8_C(  36),  INT8_C(  84), -INT8_C(  10), -INT8_C( 120),  INT8_C( 115),  INT8_C(  99),
        -INT8_C(  80), -INT8_C(  25), -INT8_C(  49), -INT8_C(  60), -INT8_C(  91), -INT8_C(  19),  INT8_C(  80),  INT8_C(  33),
        -INT8_C( 126), -INT8_C(  12), -INT8_C(  42),  INT8_C(  47),  INT8_C(   5),  INT8_C( 120), -INT8_C(  90),  INT8_C(  63),
        -INT8_C(  19),  INT8_C(   3),  INT8_C(  13), -INT8_C(  43), -INT8_C(  81),  INT8_C(  26), -INT8_C(  53), -INT8_C(  10),
         INT8_C(  51), -INT8_C(  89),  INT8_C(  74),  INT8_C(  42),  INT8_C(  47), -INT8_C(  66), -INT8_C( 115), -INT8_C(  32),
        -INT8_C(  91),  INT8_C(  92), -INT8_C(  92),  INT8_C(  74),  INT8_C(  73), -INT8_C(  12),  INT8_C( 107), -INT8_C(  53),
        -INT8_C(  24),  INT8_C(  65), -INT8_C(   6), -INT8_C(  18), -INT8_C(  71), -INT8_C(  96),  INT8_C(  45), -INT8_C(  89) },
      { -INT8_C(  92),  INT8_C(  58),  INT8_C( 124),  INT8_C(  83),  INT8_C(  84),  INT8_C(  71),  INT8_C(  73), -INT8_C( 120),
        -INT8_C(  18), -INT8_C( 108), -INT8_C(  78),  INT8_C(  30),  INT8_C(  82),  INT8_C(  63), -INT8_C(   2), -INT8_C(   9),
        -INT8_C( 101), -INT8_C(  94),  INT8_C(  65), -INT8_C(  28), -INT8_C( 106), -INT8_C(  84), -INT8_C(  81),  INT8_C( 126),
        -INT8_C(  19), -INT8_C(  86),  INT8_C( 108), -INT8_C(  90),  INT8_C(  74), -INT8_C( 103),  INT8_C(  77), -INT8_C(  18),
        -INT8_C(  44), -INT8_C(  54),  INT8_C(  66),  INT8_C(  40),  INT8_C(  17), -INT8_C( 117), -INT8_C(  80),  INT8_C(   0),
         INT8_C(  31),  INT8_C(  98),  INT8_C(  30),  INT8_C( 113), -INT8_C(  95),  INT8_C(  28),  INT8_C( 104),  INT8_C(  60),
        -INT8_C(  66), -INT8_C(  87),  INT8_C(  32),  INT8_C(  84),  INT8_C(  85), -INT8_C(  48), -INT8_C(  46),  INT8_C(  66),
         INT8_C( 122),  INT8_C(  63), -INT8_C(  23), -INT8_C(  60), -INT8_C(  40),  INT8_C(  54), -INT8_C(  77), -INT8_C(  84) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C( 124),  INT8_C(  83),  INT8_C(  84),  INT8_C(  71),  INT8_C(  73),  INT8_C(   0),
         INT8_C(  70),  INT8_C(   0), -INT8_C(  36),  INT8_C(   0),  INT8_C(  82),  INT8_C(   0),  INT8_C(   0),  INT8_C(  99),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  65),  INT8_C(   0),  INT8_C(   0), -INT8_C(  19),  INT8_C(   0),  INT8_C( 126),
        -INT8_C(  19),  INT8_C(   0),  INT8_C( 108),  INT8_C(   0),  INT8_C(  74),  INT8_C( 120),  INT8_C(   0),  INT8_C(  63),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  66),  INT8_C(  40),  INT8_C(   0),  INT8_C(  26),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  51),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  47),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  92),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  12),  INT8_C(   0),  INT8_C(  66),
         INT8_C( 122),  INT8_C(  65), -INT8_C(   6),  INT8_C(   0), -INT8_C(  40),  INT8_C(  54),  INT8_C(  45),  INT8_C(   0) } },
    { UINT64_C(11534428380767581440),
      { -INT8_C(  24),  INT8_C(  48),  INT8_C(  17), -INT8_C( 119),  INT8_C(  76),  INT8_C( 122), -INT8_C(  58),  INT8_C(  10),
         INT8_C(  35), -INT8_C(  26),  INT8_C(  94),  INT8_C( 121), -INT8_C(  74),  INT8_C(  48), -INT8_C(  69),  INT8_C(  48),
         INT8_C( 111), -INT8_C(  92), -INT8_C(  11),  INT8_C(  72), -INT8_C(  37), -INT8_C(  88), -INT8_C(  12), -INT8_C(  37),
        -INT8_C(  99), -INT8_C(  55), -INT8_C(  19),  INT8_C(  29),  INT8_C(  79), -INT8_C(   1), -INT8_C(  67),  INT8_C(  55),
         INT8_C(  47), -INT8_C(  49), -INT8_C(  64),  INT8_C( 123),  INT8_C(  73), -INT8_C( 122), -INT8_C( 123),  INT8_C( 108),
         INT8_C( 109), -INT8_C(  29), -INT8_C(  27),  INT8_C(  35),  INT8_C(  20), -INT8_C(  95),  INT8_C(  84), -INT8_C( 125),
         INT8_C(  69),  INT8_C(  73), -INT8_C(  53),  INT8_C(  32), -INT8_C(  15), -INT8_C(  64), -INT8_C(   4), -INT8_C( 114),
        -INT8_C( 119), -INT8_C(  23), -INT8_C(  85), -INT8_C(  40), -INT8_C(  23),  INT8_C( 105),  INT8_C(  15),  INT8_C(  24) },
      {  INT8_C(  56), -INT8_C(  48), -INT8_C( 108), -INT8_C( 127),  INT8_C(  86),  INT8_C(  25), -INT8_C(  19), -INT8_C(  61),
        -INT8_C(   3), -INT8_C(  45), -INT8_C(  25),  INT8_C(  17),  INT8_C( 116),  INT8_C(  59), -INT8_C( 108), -INT8_C(  71),
        -INT8_C( 124),  INT8_C(  96), -INT8_C(  38),  INT8_C( 117),  INT8_C(  32), -INT8_C(  42),  INT8_C(   3), -INT8_C(  87),
        -INT8_C(  65), -INT8_C(  82), -INT8_C( 126), -INT8_C(  88),  INT8_C(  23), -INT8_C( 111), -INT8_C(  63),  INT8_C(  79),
         INT8_C(  97),  INT8_C(  85), -INT8_C(  48), -INT8_C(  72),  INT8_C( 110), -INT8_C(  66),  INT8_C( 123),  INT8_C( 107),
        -INT8_C( 111),  INT8_C(  98),  INT8_C( 124),  INT8_C(   5), -INT8_C(  99),  INT8_C(  17), -INT8_C(  66),  INT8_C(  33),
         INT8_C( 113), -INT8_C( 104), -INT8_C( 106), -INT8_C( 111),  INT8_C( 110), -INT8_C( 103),  INT8_C(  58),  INT8_C(  46),
         INT8_C(  72), -INT8_C(  68), -INT8_C(  42),  INT8_C(  95),  INT8_C(  78), -INT8_C( 105), -INT8_C(  81), -INT8_C(  81) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  35),  INT8_C(   0),  INT8_C(  94),  INT8_C(   0),  INT8_C( 116),  INT8_C(  59), -INT8_C(  69),  INT8_C(  48),
         INT8_C( 111),  INT8_C(   0), -INT8_C(  11),  INT8_C(   0),  INT8_C(  32),  INT8_C(   0),  INT8_C(   3), -INT8_C(  37),
         INT8_C(   0), -INT8_C(  55),  INT8_C(   0),  INT8_C(   0),  INT8_C(  79),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 108),
         INT8_C( 109),  INT8_C(   0),  INT8_C( 124),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  33),
         INT8_C(   0),  INT8_C(  73),  INT8_C(   0),  INT8_C(   0),  INT8_C( 110),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C( 105),  INT8_C(   0),  INT8_C(  24) } },
    { UINT64_C(14899846269867884524),
      {  INT8_C(  69),  INT8_C(  67), -INT8_C(  45), -INT8_C(  29),  INT8_C(  84), -INT8_C( 110),  INT8_C(   4), -INT8_C(  59),
         INT8_C(  42), -INT8_C( 101),  INT8_C(  86), -INT8_C( 103),  INT8_C(  52), -INT8_C( 112), -INT8_C(  57),  INT8_C( 124),
         INT8_C(  77), -INT8_C(  99), -INT8_C(  36), -INT8_C( 101),  INT8_C(  53), -INT8_C( 117),  INT8_C(  74),  INT8_C(  33),
         INT8_C(  10), -INT8_C(  78),  INT8_C( 124),  INT8_C(  72), -INT8_C( 107),  INT8_C(  67),  INT8_C(  22), -INT8_C(  38),
        -INT8_C( 122), -INT8_C(  22), -INT8_C(  67), -INT8_C(  38),  INT8_C( 124), -INT8_C(  62), -INT8_C(  97), -INT8_C(  90),
         INT8_C(  93), -INT8_C(  11),  INT8_C(  63), -INT8_C( 111), -INT8_C( 123),  INT8_C(   6),  INT8_C(  14), -INT8_C(  46),
        -INT8_C(  92), -INT8_C(  22),  INT8_C( 109), -INT8_C(  39),  INT8_C( 117), -INT8_C(  72), -INT8_C(   6),      INT8_MAX,
         INT8_C( 106),  INT8_C( 119), -INT8_C(  57), -INT8_C(   1), -INT8_C(  70), -INT8_C(  34), -INT8_C(  39),  INT8_C(  64) },
      { -INT8_C(  56), -INT8_C( 105),  INT8_C(  26),  INT8_C(  68),  INT8_C(  89), -INT8_C(  71), -INT8_C(  22), -INT8_C(  74),
        -INT8_C(  82),  INT8_C(  42),  INT8_C(  71),  INT8_C(  51),  INT8_C(  48),  INT8_C(  85),  INT8_C(   6), -INT8_C(  44),
         INT8_C(  63),  INT8_C( 115), -INT8_C(  83), -INT8_C(  76),  INT8_C(  43), -INT8_C(  88),  INT8_C(  52), -INT8_C( 107),
         INT8_C(  31), -INT8_C(   5), -INT8_C( 108), -INT8_C(  39), -INT8_C(  39),  INT8_C( 110),  INT8_C(  25), -INT8_C(  95),
         INT8_C(   5),  INT8_C(  51), -INT8_C(  27),  INT8_C(  94), -INT8_C(  20), -INT8_C(  48),  INT8_C(  20), -INT8_C( 102),
        -INT8_C(   6),  INT8_C(  91), -INT8_C(  51),  INT8_C(  42), -INT8_C(  79), -INT8_C(  45), -INT8_C(   1), -INT8_C(  16),
         INT8_C(  71), -INT8_C(  84), -INT8_C(  91),  INT8_C( 114),  INT8_C(  84), -INT8_C(  39),  INT8_C(   8),  INT8_C( 115),
        -INT8_C(  44), -INT8_C( 100),  INT8_C(  76), -INT8_C(  82),  INT8_C(  10),  INT8_C( 101),  INT8_C(  79),  INT8_C(  15) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(  26),  INT8_C(  68),  INT8_C(   0), -INT8_C(  71),  INT8_C(   4), -INT8_C(  59),
         INT8_C(  42),  INT8_C(  42),  INT8_C(  86),  INT8_C(  51),  INT8_C(  52),  INT8_C(  85),  INT8_C(   6),  INT8_C(   0),
         INT8_C(  77),  INT8_C( 115), -INT8_C(  36),  INT8_C(   0),  INT8_C(   0), -INT8_C(  88),  INT8_C(  74),  INT8_C(   0),
         INT8_C(  31), -INT8_C(   5),  INT8_C(   0),  INT8_C(  72), -INT8_C(  39),  INT8_C(   0),  INT8_C(  25),  INT8_C(   0),
         INT8_C(   5),  INT8_C(   0), -INT8_C(  27),  INT8_C(  94),  INT8_C( 124), -INT8_C(  48),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  93),  INT8_C(  91),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   6),  INT8_C(  14), -INT8_C(  16),
         INT8_C(   0), -INT8_C(  22),  INT8_C( 109),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   8),      INT8_MAX,
         INT8_C(   0),  INT8_C( 119),  INT8_C(  76), -INT8_C(   1),  INT8_C(   0),  INT8_C(   0),  INT8_C(  79),  INT8_C(  64) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_epi8(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_epi8");
    easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_max_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint8_t a[64];
    const uint8_t b[64];
    const uint8_t r[64];
  } test_vec[] = {
    { { UINT8_C( 59), UINT8_C(162), UINT8_C(  5), UINT8_C( 15), UINT8_C(230), UINT8_C(146), UINT8_C(192), UINT8_C( 69),
        UINT8_C(125), UINT8_C(138), UINT8_C(150), UINT8_C( 98), UINT8_C(245), UINT8_C(241), UINT8_C(112), UINT8_C(249),
        UINT8_C(152), UINT8_C(155), UINT8_C( 67), UINT8_C(153), UINT8_C(113), UINT8_C(166), UINT8_C(206), UINT8_C(155),
        UINT8_C( 29), UINT8_C(173), UINT8_C( 53), UINT8_C(  7), UINT8_C(175), UINT8_C(212), UINT8_C(130), UINT8_C(234),
        UINT8_C(118), UINT8_C(135), UINT8_C(249), UINT8_C( 92), UINT8_C( 26), UINT8_C(185), UINT8_C(161), UINT8_C(151),
        UINT8_C( 67), UINT8_C( 56), UINT8_C(249), UINT8_C( 57), UINT8_C( 41), UINT8_C(105), UINT8_C( 50), UINT8_C(193),
        UINT8_C(  4), UINT8_C(117), UINT8_C( 90), UINT8_C(118), UINT8_C( 27), UINT8_C( 40), UINT8_C( 17), UINT8_C( 56),
        UINT8_C(214), UINT8_C( 70), UINT8_C( 63), UINT8_C(133), UINT8_C( 26), UINT8_C(193), UINT8_C(111), UINT8_C(144) },
      { UINT8_C( 73), UINT8_C(104), UINT8_C(237), UINT8_C( 99), UINT8_C( 33), UINT8_C(142), UINT8_C(250), UINT8_C(100),
        UINT8_C(198), UINT8_C(244), UINT8_C(157), UINT8_C(240), UINT8_C( 93), UINT8_C(207), UINT8_C(177), UINT8_C( 98),
        UINT8_C( 68), UINT8_C( 12), UINT8_C(216), UINT8_C( 95), UINT8_C( 52), UINT8_C(233), UINT8_C(152), UINT8_C( 10),
        UINT8_C( 47), UINT8_C(215), UINT8_C(143), UINT8_C( 73), UINT8_C(153), UINT8_C(254), UINT8_C(217), UINT8_C(226),
        UINT8_C(102), UINT8_C(198), UINT8_C( 69), UINT8_C(135), UINT8_C( 85), UINT8_C( 63), UINT8_C(236), UINT8_C( 27),
        UINT8_C( 51), UINT8_C(137), UINT8_C( 11), UINT8_C(145), UINT8_C( 89), UINT8_C(189), UINT8_C(243), UINT8_C(157),
        UINT8_C(201), UINT8_C(203), UINT8_C(253), UINT8_C(253), UINT8_C(180), UINT8_C(149), UINT8_C(  8), UINT8_C(227),
        UINT8_C(108), UINT8_C(151), UINT8_C( 44), UINT8_C(  5), UINT8_C(150), UINT8_C(  5), UINT8_C(231), UINT8_C(252) },
      { UINT8_C( 73), UINT8_C(162), UINT8_C(237), UINT8_C( 99), UINT8_C(230), UINT8_C(146), UINT8_C(250), UINT8_C(100),
        UINT8_C(198), UINT8_C(244), UINT8_C(157), UINT8_C(240), UINT8_C(245), UINT8_C(241), UINT8_C(177), UINT8_C(249),
        UINT8_C(152), UINT8_C(155), UINT8_C(216), UINT8_C(153), UINT8_C(113), UINT8_C(233), UINT8_C(206), UINT8_C(155),
        UINT8_C( 47), UINT8_C(215), UINT8_C(143), UINT8_C( 73), UINT8_C(175), UINT8_C(254), UINT8_C(217), UINT8_C(234),
        UINT8_C(118), UINT8_C(198), UINT8_C(249), UINT8_C(135), UINT8_C( 85), UINT8_C(185), UINT8_C(236), UINT8_C(151),
        UINT8_C( 67), UINT8_C(137), UINT8_C(249), UINT8_C(145), UINT8_C( 89), UINT8_C(189), UINT8_C(243), UINT8_C(193),
        UINT8_C(201), UINT8_C(203), UINT8_C(253), UINT8_C(253), UINT8_C(180), UINT8_C(149), UINT8_C( 17), UINT8_C(227),
        UINT8_C(214), UINT8_C(151), UINT8_C( 63), UINT8_C(133), UINT8_C(150), UINT8_C(193), UINT8_C(231), UINT8_C(252) } },
    { { UINT8_C(204), UINT8_C( 44), UINT8_C(132), UINT8_C( 33), UINT8_C(108), UINT8_C(112), UINT8_C( 60), UINT8_C(159),
        UINT8_C(249), UINT8_C( 72), UINT8_C( 48), UINT8_C( 82), UINT8_C(  5), UINT8_C( 35), UINT8_C(240), UINT8_C(206),
        UINT8_C(238), UINT8_C(237), UINT8_C(203), UINT8_C(162), UINT8_C(130), UINT8_C(211), UINT8_C(133), UINT8_C(238),
        UINT8_C(107), UINT8_C(177), UINT8_C(244), UINT8_C(  1), UINT8_C(183), UINT8_C(219), UINT8_C(253), UINT8_C(131),
        UINT8_C(  8), UINT8_C(129), UINT8_C(164), UINT8_C(116), UINT8_C(241), UINT8_C(224), UINT8_C( 19), UINT8_C(235),
        UINT8_C( 40), UINT8_C( 68), UINT8_C( 61), UINT8_C( 45), UINT8_C(103), UINT8_C( 45), UINT8_C(251), UINT8_C( 86),
        UINT8_C( 26), UINT8_C(199), UINT8_C(248), UINT8_C(156), UINT8_C(154), UINT8_C(126), UINT8_C(139), UINT8_C(  5),
        UINT8_C( 47), UINT8_C(127), UINT8_C(  6), UINT8_C(230), UINT8_C( 90), UINT8_C(  4), UINT8_C(105), UINT8_C( 98) },
      { UINT8_C(133), UINT8_C( 13), UINT8_C(214), UINT8_C(119), UINT8_C(238), UINT8_C(234), UINT8_C( 98), UINT8_C( 22),
        UINT8_C( 46), UINT8_C(159), UINT8_C( 68), UINT8_C(149), UINT8_C(205), UINT8_C( 63), UINT8_C(235), UINT8_C(231),
        UINT8_C(  6), UINT8_C(228), UINT8_C(132), UINT8_C(161), UINT8_C( 98), UINT8_C( 15), UINT8_C(166), UINT8_C(145),
        UINT8_C(142), UINT8_C(173), UINT8_C(120), UINT8_C(232), UINT8_C(177), UINT8_C(225), UINT8_C( 75), UINT8_C( 54),
        UINT8_C(239), UINT8_C( 33), UINT8_C(173), UINT8_C(221), UINT8_C( 11), UINT8_C( 15), UINT8_C(243), UINT8_C( 57),
        UINT8_C(175), UINT8_C( 55), UINT8_C(207), UINT8_C(124), UINT8_C(119), UINT8_C(186), UINT8_C( 99), UINT8_C(125),
        UINT8_C(158), UINT8_C(231), UINT8_C( 30), UINT8_C(  0), UINT8_C(246), UINT8_C(197), UINT8_C(146), UINT8_C(132),
        UINT8_C(114), UINT8_C( 10), UINT8_C(109), UINT8_C( 35), UINT8_C(235), UINT8_C(184), UINT8_C( 89), UINT8_C(218) },
      { UINT8_C(204), UINT8_C( 44), UINT8_C(214), UINT8_C(119), UINT8_C(238), UINT8_C(234), UINT8_C( 98), UINT8_C(159),
        UINT8_C(249), UINT8_C(159), UINT8_C( 68), UINT8_C(149), UINT8_C(205), UINT8_C( 63), UINT8_C(240), UINT8_C(231),
        UINT8_C(238), UINT8_C(237), UINT8_C(203), UINT8_C(162), UINT8_C(130), UINT8_C(211), UINT8_C(166), UINT8_C(238),
        UINT8_C(142), UINT8_C(177), UINT8_C(244), UINT8_C(232), UINT8_C(183), UINT8_C(225), UINT8_C(253), UINT8_C(131),
        UINT8_C(239), UINT8_C(129), UINT8_C(173), UINT8_C(221), UINT8_C(241), UINT8_C(224), UINT8_C(243), UINT8_C(235),
        UINT8_C(175), UINT8_C( 68), UINT8_C(207), UINT8_C(124), UINT8_C(119), UINT8_C(186), UINT8_C(251), UINT8_C(125),
        UINT8_C(158), UINT8_C(231), UINT8_C(248), UINT8_C(156), UINT8_C(246), UINT8_C(197), UINT8_C(146), UINT8_C(132),
        UINT8_C(114), UINT8_C(127), UINT8_C(109), UINT8_C(230), UINT8_C(235), UINT8_C(184), UINT8_C(105), UINT8_C(218) } },
    { { UINT8_C(217), UINT8_C(  7), UINT8_C(183), UINT8_C(229), UINT8_C( 22), UINT8_C(171), UINT8_C( 30), UINT8_C(197),
        UINT8_C(226), UINT8_C(237), UINT8_C( 65), UINT8_C( 89), UINT8_C(168), UINT8_C(165), UINT8_C(215), UINT8_C( 70),
        UINT8_C(140), UINT8_C(245), UINT8_C( 71), UINT8_C(131), UINT8_C(186), UINT8_C(217), UINT8_C(  7), UINT8_C( 44),
        UINT8_C(227), UINT8_C(116), UINT8_C( 79), UINT8_C(206), UINT8_C( 44), UINT8_C(169), UINT8_C(169), UINT8_C(  6),
        UINT8_C(176), UINT8_C( 96), UINT8_C(235), UINT8_C(198), UINT8_C( 11), UINT8_C(  9), UINT8_C(140), UINT8_C(238),
        UINT8_C(247), UINT8_C(205), UINT8_C( 71), UINT8_C(159), UINT8_C(114), UINT8_C( 30), UINT8_C(229),    UINT8_MAX,
        UINT8_C( 20), UINT8_C( 44), UINT8_C(130), UINT8_C(206), UINT8_C(  5), UINT8_C(137), UINT8_C(251), UINT8_C(232),
        UINT8_C(254), UINT8_C( 74), UINT8_C(183), UINT8_C( 42), UINT8_C(243), UINT8_C( 96), UINT8_C( 48), UINT8_C(163) },
      { UINT8_C(192), UINT8_C( 27), UINT8_C(106), UINT8_C(204), UINT8_C( 37), UINT8_C(246), UINT8_C(186), UINT8_C( 28),
        UINT8_C(195), UINT8_C(  1), UINT8_C(187), UINT8_C( 54), UINT8_C( 32), UINT8_C(160), UINT8_C( 53), UINT8_C( 52),
        UINT8_C(205), UINT8_C(183), UINT8_C(  2), UINT8_C(210), UINT8_C( 64), UINT8_C(253), UINT8_C(187), UINT8_C( 62),
        UINT8_C( 72), UINT8_C(114), UINT8_C(105), UINT8_C( 59), UINT8_C(210), UINT8_C(153), UINT8_C(223), UINT8_C(146),
        UINT8_C(181), UINT8_C( 73), UINT8_C( 94), UINT8_C(218), UINT8_C( 63), UINT8_C( 24), UINT8_C(246), UINT8_C(  2),
        UINT8_C( 26), UINT8_C(177), UINT8_C( 56), UINT8_C( 58), UINT8_C( 81), UINT8_C(109), UINT8_C(110), UINT8_C( 30),
        UINT8_C( 36), UINT8_C(112), UINT8_C(241), UINT8_C(101), UINT8_C(110), UINT8_C(172), UINT8_C(163), UINT8_C(182),
        UINT8_C( 30), UINT8_C( 12), UINT8_C(241), UINT8_C(240), UINT8_C(166), UINT8_C(208), UINT8_C(130), UINT8_C( 91) },
      { UINT8_C(217), UINT8_C( 27), UINT8_C(183), UINT8_C(229), UINT8_C( 37), UINT8_C(246), UINT8_C(186), UINT8_C(197),
        UINT8_C(226), UINT8_C(237), UINT8_C(187), UINT8_C( 89), UINT8_C(168), UINT8_C(165), UINT8_C(215), UINT8_C( 70),
        UINT8_C(205), UINT8_C(245), UINT8_C( 71), UINT8_C(210), UINT8_C(186), UINT8_C(253), UINT8_C(187), UINT8_C( 62),
        UINT8_C(227), UINT8_C(116), UINT8_C(105), UINT8_C(206), UINT8_C(210), UINT8_C(169), UINT8_C(223), UINT8_C(146),
        UINT8_C(181), UINT8_C( 96), UINT8_C(235), UINT8_C(218), UINT8_C( 63), UINT8_C( 24), UINT8_C(246), UINT8_C(238),
        UINT8_C(247), UINT8_C(205), UINT8_C( 71), UINT8_C(159), UINT8_C(114), UINT8_C(109), UINT8_C(229),    UINT8_MAX,
        UINT8_C( 36), UINT8_C(112), UINT8_C(241), UINT8_C(206), UINT8_C(110), UINT8_C(172), UINT8_C(251), UINT8_C(232),
        UINT8_C(254), UINT8_C( 74), UINT8_C(241), UINT8_C(240), UINT8_C(243), UINT8_C(208), UINT8_C(130), UINT8_C(163) } },
    { { UINT8_C( 25), UINT8_C(225), UINT8_C( 53), UINT8_C( 88), UINT8_C(249), UINT8_C( 43), UINT8_C( 91), UINT8_C( 19),
        UINT8_C(220), UINT8_C(147), UINT8_C( 77), UINT8_C( 45), UINT8_C(  1), UINT8_C(187), UINT8_C( 76), UINT8_C( 37),
        UINT8_C( 44), UINT8_C( 61), UINT8_C(138), UINT8_C(154), UINT8_C(233), UINT8_C( 46), UINT8_C( 80), UINT8_C(  7),
        UINT8_C( 58), UINT8_C( 65), UINT8_C(247), UINT8_C(224), UINT8_C( 18), UINT8_C(121), UINT8_C( 59), UINT8_C( 43),
        UINT8_C( 90), UINT8_C(112), UINT8_C(132), UINT8_C( 84), UINT8_C(155), UINT8_C(223), UINT8_C(103), UINT8_C(119),
        UINT8_C(114), UINT8_C(181), UINT8_C(165), UINT8_C(115), UINT8_C(112), UINT8_C(241), UINT8_C(153), UINT8_C(156),
        UINT8_C( 46), UINT8_C( 35), UINT8_C( 54), UINT8_C( 23), UINT8_C( 81), UINT8_C(134), UINT8_C( 30), UINT8_C(140),
        UINT8_C(200), UINT8_C( 21), UINT8_C(108), UINT8_C(218), UINT8_C(142), UINT8_C(168), UINT8_C(  5), UINT8_C(233) },
      { UINT8_C( 24), UINT8_C(137), UINT8_C( 61), UINT8_C(180), UINT8_C(104), UINT8_C(164), UINT8_C( 43), UINT8_C(219),
        UINT8_C( 89), UINT8_C(208), UINT8_C( 78), UINT8_C(202), UINT8_C(193), UINT8_C(231), UINT8_C(102), UINT8_C(239),
        UINT8_C( 11), UINT8_C(157), UINT8_C(  6), UINT8_C( 92), UINT8_C( 35), UINT8_C( 36), UINT8_C(232), UINT8_C(235),
        UINT8_C( 57), UINT8_C( 85), UINT8_C(197), UINT8_C(200), UINT8_C(253), UINT8_C(203), UINT8_C(177), UINT8_C( 21),
        UINT8_C( 84), UINT8_C(238), UINT8_C(201), UINT8_C(189), UINT8_C(146), UINT8_C(245), UINT8_C(152), UINT8_C(236),
        UINT8_C(197), UINT8_C(230), UINT8_C(182), UINT8_C(135), UINT8_C(206), UINT8_C( 28), UINT8_C(118), UINT8_C(217),
        UINT8_C(185), UINT8_C(125), UINT8_C( 53), UINT8_C(221), UINT8_C(161), UINT8_C( 30), UINT8_C(200), UINT8_C(219),
        UINT8_C(115), UINT8_C(142), UINT8_C(163), UINT8_C(112), UINT8_C( 89), UINT8_C( 84), UINT8_C(133), UINT8_C(173) },
      { UINT8_C( 25), UINT8_C(225), UINT8_C( 61), UINT8_C(180), UINT8_C(249), UINT8_C(164), UINT8_C( 91), UINT8_C(219),
        UINT8_C(220), UINT8_C(208), UINT8_C( 78), UINT8_C(202), UINT8_C(193), UINT8_C(231), UINT8_C(102), UINT8_C(239),
        UINT8_C( 44), UINT8_C(157), UINT8_C(138), UINT8_C(154), UINT8_C(233), UINT8_C( 46), UINT8_C(232), UINT8_C(235),
        UINT8_C( 58), UINT8_C( 85), UINT8_C(247), UINT8_C(224), UINT8_C(253), UINT8_C(203), UINT8_C(177), UINT8_C( 43),
        UINT8_C( 90), UINT8_C(238), UINT8_C(201), UINT8_C(189), UINT8_C(155), UINT8_C(245), UINT8_C(152), UINT8_C(236),
        UINT8_C(197), UINT8_C(230), UINT8_C(182), UINT8_C(135), UINT8_C(206), UINT8_C(241), UINT8_C(153), UINT8_C(217),
        UINT8_C(185), UINT8_C(125), UINT8_C( 54), UINT8_C(221), UINT8_C(161), UINT8_C(134), UINT8_C(200), UINT8_C(219),
        UINT8_C(200), UINT8_C(142), UINT8_C(163), UINT8_C(218), UINT8_C(142), UINT8_C(168), UINT8_C(133), UINT8_C(233) } },
    { { UINT8_C( 66), UINT8_C( 79), UINT8_C(106), UINT8_C(212), UINT8_C( 68), UINT8_C(  2), UINT8_C(192), UINT8_C(  9),
        UINT8_C(233), UINT8_C(118), UINT8_C(144), UINT8_C(183), UINT8_C(147), UINT8_C(  7), UINT8_C(144), UINT8_C( 76),
        UINT8_C(132), UINT8_C(197), UINT8_C( 41), UINT8_C( 37), UINT8_C(227), UINT8_C(242), UINT8_C(  0), UINT8_C( 86),
        UINT8_C(128), UINT8_C(163), UINT8_C(198), UINT8_C(217), UINT8_C(247), UINT8_C( 76), UINT8_C(134), UINT8_C( 57),
        UINT8_C(155), UINT8_C(241), UINT8_C( 14), UINT8_C(223), UINT8_C(243), UINT8_C(206), UINT8_C(232), UINT8_C(220),
        UINT8_C( 69), UINT8_C(121), UINT8_C(147), UINT8_C(216), UINT8_C(128), UINT8_C( 35), UINT8_C( 36), UINT8_C(  4),
        UINT8_C(233), UINT8_C( 78), UINT8_C( 41), UINT8_C(204), UINT8_C( 64), UINT8_C( 42), UINT8_C( 35), UINT8_C(192),
        UINT8_C(205), UINT8_C(233), UINT8_C(153), UINT8_C(197), UINT8_C( 53), UINT8_C( 31), UINT8_C(254), UINT8_C(208) },
      { UINT8_C( 16), UINT8_C( 12), UINT8_C(175), UINT8_C(  4), UINT8_C(219), UINT8_C(152), UINT8_C(224), UINT8_C( 32),
        UINT8_C( 17), UINT8_C(116), UINT8_C(248), UINT8_C(145), UINT8_C(151), UINT8_C( 28), UINT8_C(149), UINT8_C(128),
        UINT8_C(106), UINT8_C(190), UINT8_C( 77), UINT8_C(170), UINT8_C(232), UINT8_C(112), UINT8_C(106), UINT8_C(182),
        UINT8_C( 89), UINT8_C(  3), UINT8_C(123), UINT8_C(143), UINT8_C( 35), UINT8_C(121), UINT8_C( 95), UINT8_C( 51),
        UINT8_C(134), UINT8_C( 15), UINT8_C( 55), UINT8_C( 97), UINT8_C(167), UINT8_C( 24), UINT8_C(129), UINT8_C(184),
        UINT8_C(140), UINT8_C(121), UINT8_C( 73), UINT8_C( 35), UINT8_C(149), UINT8_C(222), UINT8_C(164), UINT8_C(  0),
        UINT8_C(156), UINT8_C(241), UINT8_C(170), UINT8_C(133), UINT8_C( 97), UINT8_C( 21), UINT8_C( 59), UINT8_C(186),
        UINT8_C( 24), UINT8_C(182), UINT8_C( 73), UINT8_C( 59), UINT8_C( 47), UINT8_C(169), UINT8_C(111), UINT8_C(181) },
      { UINT8_C( 66), UINT8_C( 79), UINT8_C(175), UINT8_C(212), UINT8_C(219), UINT8_C(152), UINT8_C(224), UINT8_C( 32),
        UINT8_C(233), UINT8_C(118), UINT8_C(248), UINT8_C(183), UINT8_C(151), UINT8_C( 28), UINT8_C(149), UINT8_C(128),
        UINT8_C(132), UINT8_C(197), UINT8_C( 77), UINT8_C(170), UINT8_C(232), UINT8_C(242), UINT8_C(106), UINT8_C(182),
        UINT8_C(128), UINT8_C(163), UINT8_C(198), UINT8_C(217), UINT8_C(247), UINT8_C(121), UINT8_C(134), UINT8_C( 57),
        UINT8_C(155), UINT8_C(241), UINT8_C( 55), UINT8_C(223), UINT8_C(243), UINT8_C(206), UINT8_C(232), UINT8_C(220),
        UINT8_C(140), UINT8_C(121), UINT8_C(147), UINT8_C(216), UINT8_C(149), UINT8_C(222), UINT8_C(164), UINT8_C(  4),
        UINT8_C(233), UINT8_C(241), UINT8_C(170), UINT8_C(204), UINT8_C( 97), UINT8_C( 42), UINT8_C( 59), UINT8_C(192),
        UINT8_C(205), UINT8_C(233), UINT8_C(153), UINT8_C(197), UINT8_C( 53), UINT8_C(169), UINT8_C(254), UINT8_C(208) } },
    { { UINT8_C(184), UINT8_C(166), UINT8_C( 22), UINT8_C( 95), UINT8_C(190), UINT8_C(151), UINT8_C( 23), UINT8_C( 74),
        UINT8_C( 16), UINT8_C( 96), UINT8_C(110), UINT8_C(166), UINT8_C( 62), UINT8_C( 18), UINT8_C(166), UINT8_C(218),
        UINT8_C(  3), UINT8_C( 80), UINT8_C( 95), UINT8_C(100), UINT8_C(101), UINT8_C(154), UINT8_C( 30), UINT8_C(126),
        UINT8_C( 80), UINT8_C(104), UINT8_C(185), UINT8_C(128), UINT8_C( 17), UINT8_C( 40), UINT8_C( 53), UINT8_C(201),
        UINT8_C(207), UINT8_C( 76), UINT8_C( 40), UINT8_C(141), UINT8_C(227), UINT8_C( 63), UINT8_C(216), UINT8_C(244),
        UINT8_C(159), UINT8_C( 70), UINT8_C(154), UINT8_C(221), UINT8_C( 88), UINT8_C( 64), UINT8_C(183), UINT8_C( 91),
        UINT8_C(144), UINT8_C( 23), UINT8_C(191), UINT8_C(246), UINT8_C(177), UINT8_C(221), UINT8_C(116), UINT8_C(  2),
        UINT8_C( 69), UINT8_C( 45), UINT8_C(130), UINT8_C( 86), UINT8_C( 86), UINT8_C(183), UINT8_C( 31), UINT8_C( 37) },
      { UINT8_C(  3), UINT8_C( 71), UINT8_C(178), UINT8_C(231), UINT8_C(134), UINT8_C(138), UINT8_C(219), UINT8_C( 37),
        UINT8_C(208), UINT8_C(117), UINT8_C(  2), UINT8_C( 40), UINT8_C(181), UINT8_C(186), UINT8_C(131), UINT8_C( 69),
        UINT8_C(209), UINT8_C( 66), UINT8_C( 59), UINT8_C(130), UINT8_C( 32), UINT8_C(175), UINT8_C(132), UINT8_C(101),
        UINT8_C(221), UINT8_C(  6), UINT8_C(188), UINT8_C( 51), UINT8_C(190), UINT8_C(219), UINT8_C( 88), UINT8_C(193),
        UINT8_C( 35), UINT8_C( 10), UINT8_C(168), UINT8_C(169), UINT8_C(149), UINT8_C(131), UINT8_C(207), UINT8_C(101),
        UINT8_C(248), UINT8_C(209), UINT8_C(142), UINT8_C(173), UINT8_C(139), UINT8_C( 17), UINT8_C(243), UINT8_C( 92),
        UINT8_C( 84), UINT8_C( 46), UINT8_C(223), UINT8_C(116), UINT8_C(222), UINT8_C( 99), UINT8_C(217), UINT8_C(187),
        UINT8_C(106), UINT8_C(149), UINT8_C(238), UINT8_C( 40), UINT8_C(113), UINT8_C( 70), UINT8_C(233), UINT8_C(148) },
      { UINT8_C(184), UINT8_C(166), UINT8_C(178), UINT8_C(231), UINT8_C(190), UINT8_C(151), UINT8_C(219), UINT8_C( 74),
        UINT8_C(208), UINT8_C(117), UINT8_C(110), UINT8_C(166), UINT8_C(181), UINT8_C(186), UINT8_C(166), UINT8_C(218),
        UINT8_C(209), UINT8_C( 80), UINT8_C( 95), UINT8_C(130), UINT8_C(101), UINT8_C(175), UINT8_C(132), UINT8_C(126),
        UINT8_C(221), UINT8_C(104), UINT8_C(188), UINT8_C(128), UINT8_C(190), UINT8_C(219), UINT8_C( 88), UINT8_C(201),
        UINT8_C(207), UINT8_C( 76), UINT8_C(168), UINT8_C(169), UINT8_C(227), UINT8_C(131), UINT8_C(216), UINT8_C(244),
        UINT8_C(248), UINT8_C(209), UINT8_C(154), UINT8_C(221), UINT8_C(139), UINT8_C( 64), UINT8_C(243), UINT8_C( 92),
        UINT8_C(144), UINT8_C( 46), UINT8_C(223), UINT8_C(246), UINT8_C(222), UINT8_C(221), UINT8_C(217), UINT8_C(187),
        UINT8_C(106), UINT8_C(149), UINT8_C(238), UINT8_C( 86), UINT8_C(113), UINT8_C(183), UINT8_C(233), UINT8_C(148) } },
    { { UINT8_C( 80), UINT8_C(146), UINT8_C( 61), UINT8_C(229), UINT8_C( 21), UINT8_C( 12), UINT8_C( 75), UINT8_C( 14),
        UINT8_C(222), UINT8_C(217), UINT8_C(187), UINT8_C(105), UINT8_C(234), UINT8_C(174), UINT8_C(198), UINT8_C( 62),
        UINT8_C(221), UINT8_C(165), UINT8_C(178), UINT8_C(187), UINT8_C(  8), UINT8_C(140), UINT8_C(118), UINT8_C(114),
        UINT8_C( 33), UINT8_C(100), UINT8_C(154), UINT8_C(146), UINT8_C(170), UINT8_C(132), UINT8_C( 38), UINT8_C(250),
        UINT8_C( 22), UINT8_C(100), UINT8_C(224), UINT8_C( 43), UINT8_C(112), UINT8_C( 43), UINT8_C( 57), UINT8_C( 78),
        UINT8_C(  4), UINT8_C(245), UINT8_C(184), UINT8_C(238), UINT8_C(163), UINT8_C(126), UINT8_C( 45), UINT8_C(128),
        UINT8_C( 35), UINT8_C(223), UINT8_C( 59), UINT8_C( 43), UINT8_C(107), UINT8_C(177), UINT8_C(158), UINT8_C(141),
        UINT8_C( 21), UINT8_C( 56), UINT8_C( 31), UINT8_C(191), UINT8_C(188), UINT8_C( 70), UINT8_C(186), UINT8_C(210) },
      { UINT8_C(170), UINT8_C(154), UINT8_C(254), UINT8_C( 26), UINT8_C(197), UINT8_C( 55), UINT8_C(105), UINT8_C(201),
        UINT8_C( 44), UINT8_C( 33), UINT8_C(183), UINT8_C(208), UINT8_C(159), UINT8_C(228), UINT8_C( 80), UINT8_C(194),
        UINT8_C(196), UINT8_C(140), UINT8_C(237), UINT8_C( 47), UINT8_C( 61), UINT8_C(139), UINT8_C(188), UINT8_C( 83),
        UINT8_C(196), UINT8_C(220), UINT8_C( 18), UINT8_C(128), UINT8_C( 34), UINT8_C(204), UINT8_C( 83), UINT8_C(204),
        UINT8_C(102), UINT8_C( 81), UINT8_C(230), UINT8_C( 43), UINT8_C(136), UINT8_C( 79), UINT8_C(244), UINT8_C(181),
        UINT8_C(112), UINT8_C(172), UINT8_C(133), UINT8_C( 15), UINT8_C(144), UINT8_C(213), UINT8_C(209), UINT8_C( 84),
        UINT8_C( 97), UINT8_C(191), UINT8_C(132), UINT8_C(159), UINT8_C( 74), UINT8_C( 64), UINT8_C(242), UINT8_C( 14),
        UINT8_C( 28), UINT8_C(  4), UINT8_C(143), UINT8_C( 62), UINT8_C(209), UINT8_C(226), UINT8_C( 10), UINT8_C( 55) },
      { UINT8_C(170), UINT8_C(154), UINT8_C(254), UINT8_C(229), UINT8_C(197), UINT8_C( 55), UINT8_C(105), UINT8_C(201),
        UINT8_C(222), UINT8_C(217), UINT8_C(187), UINT8_C(208), UINT8_C(234), UINT8_C(228), UINT8_C(198), UINT8_C(194),
        UINT8_C(221), UINT8_C(165), UINT8_C(237), UINT8_C(187), UINT8_C( 61), UINT8_C(140), UINT8_C(188), UINT8_C(114),
        UINT8_C(196), UINT8_C(220), UINT8_C(154), UINT8_C(146), UINT8_C(170), UINT8_C(204), UINT8_C( 83), UINT8_C(250),
        UINT8_C(102), UINT8_C(100), UINT8_C(230), UINT8_C( 43), UINT8_C(136), UINT8_C( 79), UINT8_C(244), UINT8_C(181),
        UINT8_C(112), UINT8_C(245), UINT8_C(184), UINT8_C(238), UINT8_C(163), UINT8_C(213), UINT8_C(209), UINT8_C(128),
        UINT8_C( 97), UINT8_C(223), UINT8_C(132), UINT8_C(159), UINT8_C(107), UINT8_C(177), UINT8_C(242), UINT8_C(141),
        UINT8_C( 28), UINT8_C( 56), UINT8_C(143), UINT8_C(191), UINT8_C(209), UINT8_C(226), UINT8_C(186), UINT8_C(210) } },
    { { UINT8_C( 51), UINT8_C(241), UINT8_C( 99), UINT8_C(187), UINT8_C( 64), UINT8_C( 87), UINT8_C(112), UINT8_C(177),
        UINT8_C(  3), UINT8_C(245), UINT8_C(192), UINT8_C(148), UINT8_C(203), UINT8_C(146), UINT8_C(232), UINT8_C( 44),
        UINT8_C( 81), UINT8_C(108), UINT8_C(203), UINT8_C(155), UINT8_C(173), UINT8_C(189), UINT8_C(170), UINT8_C(201),
        UINT8_C(194), UINT8_C( 57), UINT8_C(  8), UINT8_C(147), UINT8_C( 27), UINT8_C( 18), UINT8_C(202), UINT8_C( 78),
        UINT8_C(  3), UINT8_C( 45), UINT8_C(  9), UINT8_C( 68), UINT8_C(133), UINT8_C(122), UINT8_C(245), UINT8_C(136),
        UINT8_C(111), UINT8_C(181), UINT8_C( 28), UINT8_C( 58), UINT8_C( 71), UINT8_C(  5), UINT8_C(103), UINT8_C(152),
        UINT8_C(113), UINT8_C( 50), UINT8_C( 52), UINT8_C( 30), UINT8_C(240), UINT8_C(222), UINT8_C(232), UINT8_C(178),
        UINT8_C( 23), UINT8_C(240), UINT8_C( 69), UINT8_C( 50), UINT8_C(  2), UINT8_C( 15), UINT8_C(128), UINT8_C(  6) },
      { UINT8_C( 61), UINT8_C(137), UINT8_C( 74), UINT8_C(194), UINT8_C(  3), UINT8_C( 63), UINT8_C( 74), UINT8_C(115),
        UINT8_C(244), UINT8_C(103), UINT8_C(173), UINT8_C( 60), UINT8_C(108), UINT8_C( 20), UINT8_C(212), UINT8_C(221),
        UINT8_C( 71), UINT8_C(  8), UINT8_C(252), UINT8_C( 55), UINT8_C(230), UINT8_C(228), UINT8_C(233), UINT8_C(253),
        UINT8_C(212), UINT8_C( 46), UINT8_C( 47), UINT8_C(214), UINT8_C( 61), UINT8_C(175), UINT8_C(220), UINT8_C(122),
        UINT8_C( 57), UINT8_C( 38), UINT8_C( 60), UINT8_C( 60), UINT8_C(101), UINT8_C(135), UINT8_C(175), UINT8_C( 90),
        UINT8_C(238), UINT8_C( 93), UINT8_C(150), UINT8_C( 90), UINT8_C(113), UINT8_C(106), UINT8_C( 55), UINT8_C(184),
        UINT8_C(115), UINT8_C( 51), UINT8_C(239), UINT8_C( 89), UINT8_C( 23), UINT8_C(216), UINT8_C( 87), UINT8_C(235),
        UINT8_C(  6), UINT8_C(134), UINT8_C(194), UINT8_C( 68), UINT8_C( 54), UINT8_C(158), UINT8_C(190), UINT8_C(111) },
      { UINT8_C( 61), UINT8_C(241), UINT8_C( 99), UINT8_C(194), UINT8_C( 64), UINT8_C( 87), UINT8_C(112), UINT8_C(177),
        UINT8_C(244), UINT8_C(245), UINT8_C(192), UINT8_C(148), UINT8_C(203), UINT8_C(146), UINT8_C(232), UINT8_C(221),
        UINT8_C( 81), UINT8_C(108), UINT8_C(252), UINT8_C(155), UINT8_C(230), UINT8_C(228), UINT8_C(233), UINT8_C(253),
        UINT8_C(212), UINT8_C( 57), UINT8_C( 47), UINT8_C(214), UINT8_C( 61), UINT8_C(175), UINT8_C(220), UINT8_C(122),
        UINT8_C( 57), UINT8_C( 45), UINT8_C( 60), UINT8_C( 68), UINT8_C(133), UINT8_C(135), UINT8_C(245), UINT8_C(136),
        UINT8_C(238), UINT8_C(181), UINT8_C(150), UINT8_C( 90), UINT8_C(113), UINT8_C(106), UINT8_C(103), UINT8_C(184),
        UINT8_C(115), UINT8_C( 51), UINT8_C(239), UINT8_C( 89), UINT8_C(240), UINT8_C(222), UINT8_C(232), UINT8_C(235),
        UINT8_C( 23), UINT8_C(240), UINT8_C(194), UINT8_C( 68), UINT8_C( 54), UINT8_C(158), UINT8_C(190), UINT8_C(111) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_epu8(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_epu8");
    easysimd_test_x86_assert_equal_u8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int8_t src[64];
    const easysimd__mmask64 k;
    const uint8_t a[64];
    const uint8_t b[64];
    const uint8_t r[64];
  } test_vec[] = {
    { { -INT8_C( 111),  INT8_C(  15),  INT8_C(   4), -INT8_C(  59),  INT8_C(  83),  INT8_C(  68), -INT8_C( 114),  INT8_C(   0),
        -INT8_C(  53),  INT8_C(  19),  INT8_C(  68), -INT8_C(  50),  INT8_C(  36), -INT8_C( 111), -INT8_C( 120),  INT8_C(  56),
         INT8_C(  81), -INT8_C( 111),  INT8_C(  12), -INT8_C( 107), -INT8_C(  59), -INT8_C( 106), -INT8_C(  82), -INT8_C(  55),
        -INT8_C(  82), -INT8_C(  56), -INT8_C(  95),  INT8_C(  57), -INT8_C( 111), -INT8_C(  99),  INT8_C(  95),  INT8_C(  34),
        -INT8_C(  84),  INT8_C(  99), -INT8_C(  24), -INT8_C(   1), -INT8_C(  89),  INT8_C( 118), -INT8_C(   1),  INT8_C( 114),
        -INT8_C( 118),  INT8_C(  67),  INT8_C(  64), -INT8_C(  82), -INT8_C(  44), -INT8_C(  56), -INT8_C(  25),  INT8_C(  38),
         INT8_C(  90), -INT8_C(  13), -INT8_C(  69),  INT8_C(  31), -INT8_C( 118),  INT8_C( 106), -INT8_C(  24),  INT8_C(  56),
         INT8_C(  50), -INT8_C( 119),  INT8_C( 113), -INT8_C(  60),  INT8_C(  38), -INT8_C(  48), -INT8_C(  26), -INT8_C(  46) },
      UINT64_C(14938106012493794868),
      { UINT8_C( 20), UINT8_C(142), UINT8_C(125), UINT8_C(232), UINT8_C( 87), UINT8_C(100), UINT8_C( 14), UINT8_C(177),
        UINT8_C( 88), UINT8_C(202), UINT8_C(208), UINT8_C(226), UINT8_C( 52), UINT8_C(184), UINT8_C( 26), UINT8_C(102),
        UINT8_C( 65), UINT8_C(139), UINT8_C( 42), UINT8_C(104), UINT8_C( 91), UINT8_C( 17), UINT8_C( 58), UINT8_C(143),
        UINT8_C(223), UINT8_C( 12), UINT8_C(107), UINT8_C( 36), UINT8_C(220), UINT8_C(185), UINT8_C(243), UINT8_C(240),
        UINT8_C( 71), UINT8_C(113), UINT8_C(217), UINT8_C(158), UINT8_C(213), UINT8_C(231), UINT8_C( 79), UINT8_C( 45),
        UINT8_C(177), UINT8_C( 31), UINT8_C( 15), UINT8_C(229), UINT8_C(215), UINT8_C( 41), UINT8_C( 76), UINT8_C( 25),
        UINT8_C(180), UINT8_C(118), UINT8_C(129), UINT8_C( 16), UINT8_C(135), UINT8_C(187), UINT8_C(159), UINT8_C(103),
        UINT8_C(199), UINT8_C( 10), UINT8_C(139), UINT8_C(164), UINT8_C(195), UINT8_C(127), UINT8_C(148), UINT8_C( 11) },
      { UINT8_C(240), UINT8_C(109), UINT8_C(169), UINT8_C(197), UINT8_C( 85), UINT8_C(249), UINT8_C(243), UINT8_C(  6),
        UINT8_C( 24), UINT8_C(  2), UINT8_C(236), UINT8_C(240), UINT8_C( 44), UINT8_C( 56), UINT8_C(  9), UINT8_C(224),
        UINT8_C(174), UINT8_C(138), UINT8_C(240), UINT8_C( 54), UINT8_C( 69), UINT8_C(144), UINT8_C(157), UINT8_C( 13),
        UINT8_C(154), UINT8_C( 40), UINT8_C(177), UINT8_C( 94), UINT8_C(167), UINT8_C( 69), UINT8_C(105), UINT8_C(151),
        UINT8_C(179), UINT8_C( 18), UINT8_C( 93), UINT8_C(  8), UINT8_C( 11), UINT8_C( 80), UINT8_C( 14), UINT8_C( 36),
        UINT8_C( 82), UINT8_C(250), UINT8_C( 20), UINT8_C(126), UINT8_C( 50), UINT8_C( 29), UINT8_C( 95), UINT8_C(225),
        UINT8_C(167), UINT8_C( 79), UINT8_C( 23), UINT8_C(236), UINT8_C(223), UINT8_C(180), UINT8_C(249), UINT8_C(122),
        UINT8_C(220), UINT8_C(170), UINT8_C(216), UINT8_C(132), UINT8_C(240), UINT8_C( 65), UINT8_C( 27), UINT8_C(163) },
      { UINT8_C(145), UINT8_C( 15), UINT8_C(169), UINT8_C(197), UINT8_C( 87), UINT8_C(249), UINT8_C(142), UINT8_C(  0),
        UINT8_C(203), UINT8_C(202), UINT8_C(236), UINT8_C(240), UINT8_C( 36), UINT8_C(145), UINT8_C( 26), UINT8_C(224),
        UINT8_C(174), UINT8_C(145), UINT8_C( 12), UINT8_C(149), UINT8_C( 91), UINT8_C(150), UINT8_C(157), UINT8_C(143),
        UINT8_C(223), UINT8_C( 40), UINT8_C(161), UINT8_C( 94), UINT8_C(220), UINT8_C(157), UINT8_C(243), UINT8_C(240),
        UINT8_C(179), UINT8_C( 99), UINT8_C(217),    UINT8_MAX, UINT8_C(167), UINT8_C(118), UINT8_C( 79), UINT8_C(114),
        UINT8_C(138), UINT8_C( 67), UINT8_C( 64), UINT8_C(174), UINT8_C(215), UINT8_C(200), UINT8_C( 95), UINT8_C(225),
        UINT8_C( 90), UINT8_C(118), UINT8_C(129), UINT8_C(236), UINT8_C(138), UINT8_C(106), UINT8_C(249), UINT8_C( 56),
        UINT8_C(220), UINT8_C(170), UINT8_C(216), UINT8_C(164), UINT8_C( 38), UINT8_C(208), UINT8_C(148), UINT8_C(163) } },
    { {  INT8_C(  83),  INT8_C( 120), -INT8_C(  85),  INT8_C(  95), -INT8_C(  56), -INT8_C(  71), -INT8_C( 125),  INT8_C(  27),
        -INT8_C(  76), -INT8_C( 105), -INT8_C( 103), -INT8_C(  26), -INT8_C(  76), -INT8_C(   8), -INT8_C(  57),  INT8_C(  91),
         INT8_C(  72), -INT8_C(  34),  INT8_C(  71),  INT8_C(  39), -INT8_C( 110),  INT8_C(  65), -INT8_C(  95),  INT8_C( 111),
        -INT8_C(  21),  INT8_C( 121), -INT8_C(  13), -INT8_C(  37), -INT8_C(  70),  INT8_C(  14),  INT8_C( 126),  INT8_C(  14),
        -INT8_C( 121),  INT8_C(  41),  INT8_C( 109),  INT8_C(  79), -INT8_C(  29), -INT8_C(  16),  INT8_C( 106), -INT8_C( 105),
        -INT8_C( 121),  INT8_C(   4),  INT8_C( 125),  INT8_C(  59), -INT8_C(   4),  INT8_C(  69), -INT8_C( 106),  INT8_C(  68),
         INT8_C(  35), -INT8_C(  35),  INT8_C( 108), -INT8_C(  74),  INT8_C(  30),  INT8_C(  13),  INT8_C(  37),  INT8_C(  10),
        -INT8_C( 121),  INT8_C(  24), -INT8_C(  27),  INT8_C(  65),  INT8_C(  38),  INT8_C( 100),  INT8_C(  79), -INT8_C(  83) },
      UINT64_C( 3677021611099012237),
      { UINT8_C(107), UINT8_C(133), UINT8_C(110), UINT8_C(104), UINT8_C(202), UINT8_C(  4), UINT8_C(172), UINT8_C(237),
        UINT8_C(226), UINT8_C( 24), UINT8_C(163), UINT8_C(  0), UINT8_C( 38), UINT8_C(200), UINT8_C( 10), UINT8_C(173),
        UINT8_C(224), UINT8_C(240), UINT8_C(238), UINT8_C(  7), UINT8_C( 84), UINT8_C( 62), UINT8_C(180), UINT8_C(225),
        UINT8_C(250), UINT8_C(177), UINT8_C( 82), UINT8_C(167), UINT8_C( 25), UINT8_C( 89), UINT8_C(218), UINT8_C(132),
        UINT8_C(222), UINT8_C( 73), UINT8_C(236), UINT8_C(168), UINT8_C( 77), UINT8_C(153), UINT8_C(150), UINT8_C( 47),
        UINT8_C(177), UINT8_C( 57), UINT8_C( 48), UINT8_C(215), UINT8_C(  2), UINT8_C( 58), UINT8_C(132), UINT8_C(226),
        UINT8_C( 42), UINT8_C(115), UINT8_C(233), UINT8_C(126), UINT8_C(177), UINT8_C(158), UINT8_C( 96), UINT8_C(171),
        UINT8_C( 79), UINT8_C(178), UINT8_C( 82), UINT8_C(104), UINT8_C( 11), UINT8_C( 45), UINT8_C(237), UINT8_C(234) },
      { UINT8_C(118), UINT8_C(217), UINT8_C(146), UINT8_C(195), UINT8_C(114), UINT8_C( 40), UINT8_C(243), UINT8_C( 36),
        UINT8_C( 98), UINT8_C( 35), UINT8_C(251), UINT8_C(100), UINT8_C( 93), UINT8_C(128), UINT8_C( 70), UINT8_C(136),
        UINT8_C(243), UINT8_C( 48), UINT8_C(  6), UINT8_C(164), UINT8_C(206), UINT8_C(102), UINT8_C( 79), UINT8_C( 29),
        UINT8_C( 24), UINT8_C(162), UINT8_C(134), UINT8_C( 36), UINT8_C(207), UINT8_C(115), UINT8_C( 14), UINT8_C( 69),
        UINT8_C( 76), UINT8_C(160), UINT8_C(  8), UINT8_C(191), UINT8_C(201), UINT8_C(251), UINT8_C(227), UINT8_C( 43),
        UINT8_C( 30), UINT8_C(222), UINT8_C(143), UINT8_C(124), UINT8_C( 94), UINT8_C(213), UINT8_C(  4), UINT8_C( 81),
        UINT8_C(  5), UINT8_C( 10), UINT8_C(245), UINT8_C(211), UINT8_C(113), UINT8_C( 69), UINT8_C(241), UINT8_C(137),
        UINT8_C(231), UINT8_C(119), UINT8_C(173), UINT8_C(182), UINT8_C(234), UINT8_C(187), UINT8_C(251), UINT8_C( 54) },
      { UINT8_C(118), UINT8_C(120), UINT8_C(146), UINT8_C(195), UINT8_C(200), UINT8_C(185), UINT8_C(131), UINT8_C(237),
        UINT8_C(180), UINT8_C(151), UINT8_C(251), UINT8_C(100), UINT8_C( 93), UINT8_C(200), UINT8_C(199), UINT8_C(173),
        UINT8_C(243), UINT8_C(222), UINT8_C(238), UINT8_C(164), UINT8_C(206), UINT8_C(102), UINT8_C(180), UINT8_C(225),
        UINT8_C(235), UINT8_C(121), UINT8_C(243), UINT8_C(219), UINT8_C(207), UINT8_C(115), UINT8_C(218), UINT8_C( 14),
        UINT8_C(135), UINT8_C( 41), UINT8_C(236), UINT8_C(191), UINT8_C(227), UINT8_C(251), UINT8_C(106), UINT8_C( 47),
        UINT8_C(177), UINT8_C(222), UINT8_C(143), UINT8_C( 59), UINT8_C(252), UINT8_C(213), UINT8_C(132), UINT8_C( 68),
        UINT8_C( 42), UINT8_C(115), UINT8_C(245), UINT8_C(182), UINT8_C( 30), UINT8_C( 13), UINT8_C( 37), UINT8_C( 10),
        UINT8_C(231), UINT8_C(178), UINT8_C(229), UINT8_C( 65), UINT8_C(234), UINT8_C(187), UINT8_C( 79), UINT8_C(173) } },
    { {  INT8_C(  92),  INT8_C(   3), -INT8_C(  11),  INT8_C(  37), -INT8_C(   1), -INT8_C(  40),  INT8_C(  80),  INT8_C(  29),
        -INT8_C(  73), -INT8_C(  33), -INT8_C( 103),  INT8_C(  21), -INT8_C(  76), -INT8_C(  99),  INT8_C( 103), -INT8_C(  70),
        -INT8_C(  88),  INT8_C(  92), -INT8_C( 115),  INT8_C(  25), -INT8_C(  95),  INT8_C( 126), -INT8_C(  94), -INT8_C( 120),
        -INT8_C(  11),  INT8_C(  80),  INT8_C(  62), -INT8_C(  33),  INT8_C(  11),  INT8_C(  57),  INT8_C(  22),  INT8_C( 103),
         INT8_C(  61),  INT8_C(  11), -INT8_C( 116),  INT8_C(  60), -INT8_C(  28), -INT8_C(  36),  INT8_C(  89), -INT8_C( 101),
        -INT8_C(  69), -INT8_C(  13), -INT8_C(  80),  INT8_C( 112), -INT8_C( 112),  INT8_C(  23),  INT8_C(  42),  INT8_C(  56),
         INT8_C( 116), -INT8_C(  73),  INT8_C(  81),  INT8_C(  21),  INT8_C(  54), -INT8_C(  12), -INT8_C(  98),  INT8_C(  43),
         INT8_C(  68), -INT8_C(  36),  INT8_C(  11),  INT8_C(  79),  INT8_C(  22),  INT8_C(  33), -INT8_C(  73),  INT8_C(  83) },
      UINT64_C(15829000539738161964),
      { UINT8_C(219), UINT8_C( 92), UINT8_C( 75), UINT8_C(108), UINT8_C(115), UINT8_C(117), UINT8_C(164), UINT8_C(231),
        UINT8_C( 45), UINT8_C(246), UINT8_C(253), UINT8_C( 99), UINT8_C(234), UINT8_C(155), UINT8_C(142), UINT8_C( 46),
        UINT8_C(119), UINT8_C(153), UINT8_C(125), UINT8_C(141), UINT8_C(186), UINT8_C( 52), UINT8_C(224), UINT8_C(231),
        UINT8_C(120), UINT8_C(111), UINT8_C(247), UINT8_C(152), UINT8_C( 88), UINT8_C(163), UINT8_C(115), UINT8_C( 51),
           UINT8_MAX, UINT8_C(191), UINT8_C(159), UINT8_C(114), UINT8_C( 52), UINT8_C( 68), UINT8_C( 90), UINT8_C( 97),
        UINT8_C( 58), UINT8_C( 87), UINT8_C(196), UINT8_C( 36), UINT8_C(242), UINT8_C( 83), UINT8_C( 82), UINT8_C(105),
        UINT8_C(236), UINT8_C(207), UINT8_C(247), UINT8_C(167), UINT8_C(  4), UINT8_C(215), UINT8_C(142), UINT8_C(124),
        UINT8_C( 71), UINT8_C(133), UINT8_C( 20), UINT8_C(159), UINT8_C( 40), UINT8_C(135), UINT8_C(210), UINT8_C( 39) },
      { UINT8_C( 70), UINT8_C(114), UINT8_C(154), UINT8_C(123), UINT8_C(182), UINT8_C(244), UINT8_C(220), UINT8_C(240),
        UINT8_C( 75), UINT8_C(161), UINT8_C( 20), UINT8_C( 61), UINT8_C(244), UINT8_C(102), UINT8_C(166), UINT8_C(224),
        UINT8_C( 53), UINT8_C(157), UINT8_C(135), UINT8_C( 57), UINT8_C(117), UINT8_C( 21), UINT8_C(181), UINT8_C(188),
        UINT8_C(155), UINT8_C(201), UINT8_C( 91), UINT8_C(195), UINT8_C( 81), UINT8_C( 45), UINT8_C(235), UINT8_C(151),
        UINT8_C(159), UINT8_C(133), UINT8_C( 18), UINT8_C( 85), UINT8_C(121), UINT8_C(239), UINT8_C( 69), UINT8_C(196),
        UINT8_C(144), UINT8_C( 89), UINT8_C(  1), UINT8_C(132), UINT8_C(191), UINT8_C(167), UINT8_C(100), UINT8_C(245),
        UINT8_C( 69), UINT8_C(236), UINT8_C( 46), UINT8_C(186), UINT8_C(  1), UINT8_C(228), UINT8_C(118), UINT8_C(156),
        UINT8_C(173), UINT8_C(209), UINT8_C( 96), UINT8_C(254), UINT8_C(254), UINT8_C( 75), UINT8_C(150), UINT8_C(158) },
      { UINT8_C( 92), UINT8_C(  3), UINT8_C(154), UINT8_C(123),    UINT8_MAX, UINT8_C(244), UINT8_C( 80), UINT8_C( 29),
        UINT8_C( 75), UINT8_C(246), UINT8_C(153), UINT8_C( 21), UINT8_C(180), UINT8_C(157), UINT8_C(166), UINT8_C(186),
        UINT8_C(119), UINT8_C(157), UINT8_C(135), UINT8_C(141), UINT8_C(161), UINT8_C(126), UINT8_C(162), UINT8_C(231),
        UINT8_C(245), UINT8_C( 80), UINT8_C( 62), UINT8_C(223), UINT8_C( 88), UINT8_C( 57), UINT8_C( 22), UINT8_C(103),
        UINT8_C( 61), UINT8_C( 11), UINT8_C(140), UINT8_C( 60), UINT8_C(228), UINT8_C(239), UINT8_C( 89), UINT8_C(155),
        UINT8_C(187), UINT8_C(243), UINT8_C(176), UINT8_C(132), UINT8_C(144), UINT8_C(167), UINT8_C(100), UINT8_C(245),
        UINT8_C(236), UINT8_C(236), UINT8_C( 81), UINT8_C(186), UINT8_C( 54), UINT8_C(228), UINT8_C(158), UINT8_C(156),
        UINT8_C(173), UINT8_C(209), UINT8_C( 11), UINT8_C(254), UINT8_C(254), UINT8_C( 33), UINT8_C(210), UINT8_C(158) } },
    { { -INT8_C(  48), -INT8_C(  88), -INT8_C(  13),  INT8_C(  73), -INT8_C( 105),  INT8_C(  57),  INT8_C(  13),  INT8_C(  39),
        -INT8_C( 110),  INT8_C(  14), -INT8_C(  85),  INT8_C(  82), -INT8_C(  75),  INT8_C(  16),  INT8_C(  71), -INT8_C(   6),
        -INT8_C(   4),  INT8_C( 117), -INT8_C(  76), -INT8_C(   3),  INT8_C(  89),  INT8_C(  42), -INT8_C( 102),  INT8_C(   7),
        -INT8_C(   5), -INT8_C(   6),  INT8_C(   5), -INT8_C(   6),  INT8_C(  69), -INT8_C( 101), -INT8_C( 104),  INT8_C(  21),
         INT8_C(  68), -INT8_C( 117),  INT8_C(  94), -INT8_C(  37), -INT8_C(  60),  INT8_C( 107),  INT8_C(   3),  INT8_C(  87),
         INT8_C( 121), -INT8_C(  82), -INT8_C(  87),  INT8_C(  46), -INT8_C(  66), -INT8_C(  16),  INT8_C(  41), -INT8_C(  70),
         INT8_C( 101), -INT8_C(  35), -INT8_C(  72), -INT8_C(  65),  INT8_C(   8),  INT8_C(  82), -INT8_C(  58),  INT8_C(   3),
         INT8_C(  76), -INT8_C(  53), -INT8_C(   3), -INT8_C( 111),  INT8_C( 103), -INT8_C( 107), -INT8_C(  90), -INT8_C(  85) },
      UINT64_C(16734401429087061025),
      { UINT8_C( 56), UINT8_C(229), UINT8_C( 22), UINT8_C(246), UINT8_C(213), UINT8_C( 63), UINT8_C(177), UINT8_C( 59),
        UINT8_C( 29), UINT8_C(105), UINT8_C(250), UINT8_C( 37), UINT8_C(187), UINT8_C(192), UINT8_C( 40), UINT8_C(  7),
        UINT8_C(139), UINT8_C( 38), UINT8_C(152), UINT8_C(242), UINT8_C(187), UINT8_C( 62), UINT8_C(157), UINT8_C(220),
        UINT8_C( 66), UINT8_C( 36), UINT8_C(194), UINT8_C(177), UINT8_C(173), UINT8_C(254), UINT8_C(153), UINT8_C(229),
        UINT8_C(228), UINT8_C(175), UINT8_C(220), UINT8_C(185), UINT8_C(239), UINT8_C(141), UINT8_C(244), UINT8_C( 12),
        UINT8_C(246), UINT8_C(238), UINT8_C( 49), UINT8_C(177), UINT8_C(174), UINT8_C( 89), UINT8_C(184), UINT8_C( 58),
        UINT8_C(127), UINT8_C( 80), UINT8_C( 44), UINT8_C( 59), UINT8_C(142), UINT8_C(202), UINT8_C( 23), UINT8_C(208),
        UINT8_C(238), UINT8_C(217), UINT8_C(129), UINT8_C(155), UINT8_C(216), UINT8_C( 26), UINT8_C(129), UINT8_C(188) },
      { UINT8_C(201), UINT8_C( 93), UINT8_C(117), UINT8_C(184), UINT8_C(234), UINT8_C(106), UINT8_C(196), UINT8_C(224),
        UINT8_C( 88), UINT8_C(245), UINT8_C(145), UINT8_C(  7), UINT8_C( 79), UINT8_C( 73), UINT8_C( 65), UINT8_C(206),
        UINT8_C(153), UINT8_C(109), UINT8_C(  9), UINT8_C( 39), UINT8_C( 55), UINT8_C( 33), UINT8_C(247), UINT8_C( 37),
        UINT8_C(250), UINT8_C(120), UINT8_C(193), UINT8_C(210), UINT8_C(146), UINT8_C( 66), UINT8_C(142), UINT8_C( 91),
        UINT8_C(159), UINT8_C(  4), UINT8_C( 20), UINT8_C(137), UINT8_C(110), UINT8_C(216), UINT8_C(105), UINT8_C(198),
        UINT8_C(206), UINT8_C(250), UINT8_C(205), UINT8_C( 29), UINT8_C( 67), UINT8_C( 14), UINT8_C(235), UINT8_C(220),
        UINT8_C(124), UINT8_C(245), UINT8_C(  3), UINT8_C(179), UINT8_C( 22), UINT8_C(250), UINT8_C(217), UINT8_C( 16),
        UINT8_C(114), UINT8_C(154), UINT8_C(227), UINT8_C(  4), UINT8_C(220), UINT8_C(113), UINT8_C( 95), UINT8_C(123) },
      { UINT8_C(201), UINT8_C(168), UINT8_C(243), UINT8_C( 73), UINT8_C(151), UINT8_C(106), UINT8_C( 13), UINT8_C( 39),
        UINT8_C(146), UINT8_C( 14), UINT8_C(250), UINT8_C( 82), UINT8_C(181), UINT8_C( 16), UINT8_C( 71), UINT8_C(250),
        UINT8_C(252), UINT8_C(109), UINT8_C(152), UINT8_C(253), UINT8_C( 89), UINT8_C( 42), UINT8_C(154), UINT8_C(220),
        UINT8_C(250), UINT8_C(250), UINT8_C(194), UINT8_C(250), UINT8_C( 69), UINT8_C(254), UINT8_C(153), UINT8_C(229),
        UINT8_C(228), UINT8_C(175), UINT8_C(220), UINT8_C(185), UINT8_C(196), UINT8_C(216), UINT8_C(244), UINT8_C( 87),
        UINT8_C(246), UINT8_C(174), UINT8_C(169), UINT8_C(177), UINT8_C(190), UINT8_C(240), UINT8_C( 41), UINT8_C(220),
        UINT8_C(101), UINT8_C(221), UINT8_C( 44), UINT8_C(179), UINT8_C(142), UINT8_C(250), UINT8_C(198), UINT8_C(  3),
        UINT8_C( 76), UINT8_C(203), UINT8_C(253), UINT8_C(155), UINT8_C(103), UINT8_C(113), UINT8_C(129), UINT8_C(188) } },
    { {  INT8_C( 117),  INT8_C( 115),  INT8_C(   4), -INT8_C(  29),  INT8_C(  76),  INT8_C( 109), -INT8_C(  86),  INT8_C(  26),
         INT8_C( 103),  INT8_C( 119),  INT8_C(  55), -INT8_C(  86), -INT8_C( 122),  INT8_C(  34), -INT8_C( 122),  INT8_C(   2),
         INT8_C(  23), -INT8_C( 119), -INT8_C(  75),  INT8_C(  45), -INT8_C( 125), -INT8_C( 114),  INT8_C(  62), -INT8_C(  11),
         INT8_C(  40),  INT8_C(  33), -INT8_C(   7),  INT8_C(   4), -INT8_C( 110),  INT8_C(  88),      INT8_MAX,  INT8_C(   8),
        -INT8_C(  52), -INT8_C( 125), -INT8_C(  21),  INT8_C(  24), -INT8_C(  16), -INT8_C( 107),  INT8_C(  50),  INT8_C(  87),
         INT8_C(  13),  INT8_C( 105),  INT8_C(   1), -INT8_C( 109), -INT8_C( 117), -INT8_C( 121), -INT8_C( 107), -INT8_C(  93),
         INT8_C(  16),  INT8_C(  74), -INT8_C(  48), -INT8_C( 109), -INT8_C(  39),  INT8_C(  14), -INT8_C( 120),  INT8_C(   1),
         INT8_C(  47), -INT8_C( 127),  INT8_C(   6), -INT8_C(  62), -INT8_C(  38), -INT8_C( 123), -INT8_C(  54), -INT8_C(  90) },
      UINT64_C( 6364131957554459913),
      { UINT8_C( 89), UINT8_C( 82), UINT8_C(235), UINT8_C(228), UINT8_C(218), UINT8_C(128), UINT8_C(135), UINT8_C(234),
        UINT8_C(202), UINT8_C( 88), UINT8_C(126), UINT8_C(163), UINT8_C(102), UINT8_C(  6), UINT8_C(165), UINT8_C(150),
        UINT8_C(136), UINT8_C(171), UINT8_C( 88), UINT8_C( 98), UINT8_C( 48), UINT8_C( 34), UINT8_C(  8), UINT8_C( 57),
        UINT8_C(215), UINT8_C(198), UINT8_C( 51), UINT8_C( 34), UINT8_C(182), UINT8_C(132), UINT8_C(122), UINT8_C( 15),
        UINT8_C(214), UINT8_C(101), UINT8_C(243), UINT8_C(176), UINT8_C(229), UINT8_C(123), UINT8_C(155), UINT8_C(176),
        UINT8_C(211), UINT8_C( 25), UINT8_C( 83), UINT8_C( 57), UINT8_C( 31), UINT8_C(248), UINT8_C(207), UINT8_C(167),
        UINT8_C(163), UINT8_C( 39), UINT8_C(  9), UINT8_C(212), UINT8_C( 73), UINT8_C( 17), UINT8_C( 13), UINT8_C( 33),
        UINT8_C(215), UINT8_C( 64), UINT8_C( 67), UINT8_C(141), UINT8_C(196), UINT8_C(190), UINT8_C(156), UINT8_C(155) },
      { UINT8_C( 35), UINT8_C(144), UINT8_C( 75), UINT8_C(  9), UINT8_C( 11), UINT8_C(230), UINT8_C(185), UINT8_C(222),
           UINT8_MAX, UINT8_C( 12), UINT8_C( 23), UINT8_C( 31), UINT8_C(  5), UINT8_C(231), UINT8_C(198), UINT8_C(168),
        UINT8_C( 14), UINT8_C(208), UINT8_C(124), UINT8_C( 88), UINT8_C(225), UINT8_C(138), UINT8_C(121), UINT8_C(185),
        UINT8_C(202), UINT8_C(188), UINT8_C( 70), UINT8_C(143), UINT8_C(122), UINT8_C(227), UINT8_C( 42), UINT8_C(158),
        UINT8_C(115), UINT8_C(117), UINT8_C(167), UINT8_C(126), UINT8_C( 92), UINT8_C( 96), UINT8_C( 92), UINT8_C( 91),
        UINT8_C(108), UINT8_C(115), UINT8_C(122), UINT8_C(113), UINT8_C( 90), UINT8_C( 65), UINT8_C( 26), UINT8_C(105),
        UINT8_C( 17), UINT8_C(150), UINT8_C(193), UINT8_C(242), UINT8_C( 32), UINT8_C( 58), UINT8_C(171), UINT8_C(235),
        UINT8_C(246), UINT8_C(242), UINT8_C(122), UINT8_C(113), UINT8_C(213), UINT8_C(164), UINT8_C( 15), UINT8_C( 72) },
      { UINT8_C( 89), UINT8_C(115), UINT8_C(  4), UINT8_C(228), UINT8_C( 76), UINT8_C(109), UINT8_C(170), UINT8_C( 26),
           UINT8_MAX, UINT8_C(119), UINT8_C(126), UINT8_C(170), UINT8_C(102), UINT8_C(231), UINT8_C(134), UINT8_C(168),
        UINT8_C( 23), UINT8_C(208), UINT8_C(124), UINT8_C( 98), UINT8_C(225), UINT8_C(138), UINT8_C( 62), UINT8_C(185),
        UINT8_C(215), UINT8_C( 33), UINT8_C(249), UINT8_C(143), UINT8_C(182), UINT8_C(227), UINT8_C(122), UINT8_C(158),
        UINT8_C(214), UINT8_C(117), UINT8_C(235), UINT8_C(176), UINT8_C(240), UINT8_C(149), UINT8_C(155), UINT8_C( 87),
        UINT8_C( 13), UINT8_C(105), UINT8_C(  1), UINT8_C(147), UINT8_C( 90), UINT8_C(248), UINT8_C(207), UINT8_C(167),
        UINT8_C(163), UINT8_C( 74), UINT8_C(208), UINT8_C(147), UINT8_C( 73), UINT8_C( 14), UINT8_C(171), UINT8_C(  1),
        UINT8_C( 47), UINT8_C(129), UINT8_C(  6), UINT8_C(141), UINT8_C(213), UINT8_C(133), UINT8_C(156), UINT8_C(166) } },
    { {  INT8_C(  25), -INT8_C(  74), -INT8_C(  58),  INT8_C( 117),  INT8_C(  22),  INT8_C(  34), -INT8_C(  47), -INT8_C( 126),
        -INT8_C( 107),  INT8_C(  75), -INT8_C(  12), -INT8_C(  16), -INT8_C( 116),  INT8_C(  14),  INT8_C(  89), -INT8_C(  99),
        -INT8_C(  92),  INT8_C(  26), -INT8_C( 112), -INT8_C(  59),  INT8_C(  84),  INT8_C(  59), -INT8_C(  80),  INT8_C(  74),
         INT8_C(  45),  INT8_C(  42), -INT8_C(  69),  INT8_C(   2), -INT8_C(  50), -INT8_C(  54),  INT8_C(  74), -INT8_C(  25),
             INT8_MIN,  INT8_C(  16),  INT8_C(  93), -INT8_C( 106),  INT8_C(  50),  INT8_C(  46),  INT8_C(  25), -INT8_C(  56),
         INT8_C( 121),  INT8_C(  13), -INT8_C(  72),  INT8_C(   6),  INT8_C(  27),  INT8_C(  17), -INT8_C(  93), -INT8_C(  65),
         INT8_C(  43),  INT8_C(  51), -INT8_C( 124),      INT8_MAX,  INT8_C( 111),  INT8_C(  52), -INT8_C(  55), -INT8_C( 100),
         INT8_C(  94), -INT8_C( 123), -INT8_C(  97),  INT8_C(  44),  INT8_C(  79), -INT8_C(  23),  INT8_C(  20), -INT8_C(  48) },
      UINT64_C( 1798202472849109498),
      { UINT8_C(140), UINT8_C(172), UINT8_C( 30), UINT8_C(167), UINT8_C(189), UINT8_C(194), UINT8_C(103), UINT8_C(232),
        UINT8_C(245), UINT8_C(235), UINT8_C(103), UINT8_C(100), UINT8_C( 32), UINT8_C( 49), UINT8_C(  1), UINT8_C(126),
        UINT8_C(182), UINT8_C(160), UINT8_C(171), UINT8_C(  5), UINT8_C(137), UINT8_C(191), UINT8_C(213), UINT8_C(131),
        UINT8_C( 48), UINT8_C( 60), UINT8_C(176), UINT8_C(207), UINT8_C(187), UINT8_C(164), UINT8_C(231), UINT8_C( 72),
        UINT8_C( 81), UINT8_C(  6), UINT8_C(239), UINT8_C( 14), UINT8_C(200), UINT8_C( 86), UINT8_C(247), UINT8_C(189),
        UINT8_C( 66), UINT8_C( 94), UINT8_C( 34), UINT8_C( 98), UINT8_C(143), UINT8_C( 35), UINT8_C(224), UINT8_C( 69),
        UINT8_C(195), UINT8_C(139), UINT8_C( 75), UINT8_C( 76), UINT8_C( 74), UINT8_C( 32), UINT8_C(208), UINT8_C(122),
        UINT8_C( 92), UINT8_C(128), UINT8_C( 73), UINT8_C( 24), UINT8_C( 36), UINT8_C( 49), UINT8_C( 96), UINT8_C(117) },
      { UINT8_C( 55), UINT8_C( 79), UINT8_C(132),    UINT8_MAX, UINT8_C(166), UINT8_C(123), UINT8_C(188), UINT8_C(232),
        UINT8_C(217), UINT8_C(222), UINT8_C( 74), UINT8_C(105), UINT8_C(  1), UINT8_C( 42), UINT8_C(174), UINT8_C(196),
        UINT8_C(182), UINT8_C(249), UINT8_C( 17), UINT8_C(  0), UINT8_C( 26), UINT8_C(225), UINT8_C(123), UINT8_C(118),
        UINT8_C( 97), UINT8_C(196), UINT8_C(142), UINT8_C(133), UINT8_C(245), UINT8_C(238), UINT8_C(251), UINT8_C( 44),
        UINT8_C( 62), UINT8_C(127), UINT8_C( 43), UINT8_C(228), UINT8_C(250), UINT8_C(232), UINT8_C(204), UINT8_C(211),
        UINT8_C(198), UINT8_C( 22), UINT8_C( 60), UINT8_C(200), UINT8_C( 64), UINT8_C(235), UINT8_C(140), UINT8_C(246),
        UINT8_C(228), UINT8_C(157), UINT8_C(247), UINT8_C(254), UINT8_C(126), UINT8_C(114), UINT8_C(117), UINT8_C(223),
        UINT8_C( 54), UINT8_C(  3), UINT8_C(101), UINT8_C( 44), UINT8_C(242), UINT8_C( 96), UINT8_C( 88), UINT8_C( 48) },
      { UINT8_C( 25), UINT8_C(172), UINT8_C(198),    UINT8_MAX, UINT8_C(189), UINT8_C(194), UINT8_C(188), UINT8_C(232),
        UINT8_C(245), UINT8_C( 75), UINT8_C(244), UINT8_C(240), UINT8_C( 32), UINT8_C( 49), UINT8_C(174), UINT8_C(157),
        UINT8_C(164), UINT8_C(249), UINT8_C(171), UINT8_C(197), UINT8_C( 84), UINT8_C(225), UINT8_C(213), UINT8_C( 74),
        UINT8_C( 45), UINT8_C( 42), UINT8_C(176), UINT8_C(207), UINT8_C(206), UINT8_C(238), UINT8_C( 74), UINT8_C(231),
        UINT8_C( 81), UINT8_C(127), UINT8_C(239), UINT8_C(228), UINT8_C(250), UINT8_C( 46), UINT8_C( 25), UINT8_C(211),
        UINT8_C(198), UINT8_C( 94), UINT8_C( 60), UINT8_C(200), UINT8_C(143), UINT8_C(235), UINT8_C(224), UINT8_C(191),
        UINT8_C( 43), UINT8_C( 51), UINT8_C(247), UINT8_C(127), UINT8_C(126), UINT8_C(114), UINT8_C(208), UINT8_C(223),
        UINT8_C( 94), UINT8_C(133), UINT8_C(159), UINT8_C( 44), UINT8_C(242), UINT8_C(233), UINT8_C( 20), UINT8_C(208) } },
    { { -INT8_C(  33), -INT8_C( 124),  INT8_C(  20), -INT8_C(  39),  INT8_C( 108), -INT8_C(  32), -INT8_C(  84),  INT8_C(  50),
        -INT8_C(  10), -INT8_C(  23), -INT8_C(   6),  INT8_C(  54), -INT8_C(  44), -INT8_C( 121),  INT8_C(  45), -INT8_C(  72),
         INT8_C(  36),  INT8_C(  36), -INT8_C(  73), -INT8_C(  93), -INT8_C( 106),  INT8_C(  44), -INT8_C( 126), -INT8_C(  52),
         INT8_C(  47), -INT8_C(  25), -INT8_C(   8),  INT8_C(  33),  INT8_C(  71),  INT8_C(  81),  INT8_C(  81),  INT8_C(  38),
        -INT8_C(  43),  INT8_C( 101), -INT8_C(   1),  INT8_C(  65),  INT8_C(  69), -INT8_C(  84),  INT8_C( 115),  INT8_C(  59),
        -INT8_C( 107),  INT8_C( 110),  INT8_C( 114),  INT8_C( 105), -INT8_C(  11), -INT8_C(  97),  INT8_C(  33),  INT8_C(  25),
        -INT8_C(  61), -INT8_C(  40), -INT8_C(  68),  INT8_C(  89),  INT8_C(   4),  INT8_C(  63),  INT8_C(  37),  INT8_C(  52),
         INT8_C(  38),  INT8_C(  30),  INT8_C(  85),  INT8_C( 110),  INT8_C( 111), -INT8_C(  89), -INT8_C( 108),  INT8_C(  68) },
      UINT64_C(15388228456940934156),
      { UINT8_C(102),    UINT8_MAX, UINT8_C( 62), UINT8_C( 91), UINT8_C(158), UINT8_C( 95), UINT8_C(117), UINT8_C( 97),
        UINT8_C( 56), UINT8_C( 49), UINT8_C(186), UINT8_C( 60), UINT8_C(112), UINT8_C(224), UINT8_C(112), UINT8_C(151),
        UINT8_C(254), UINT8_C(198), UINT8_C(  5), UINT8_C(109), UINT8_C(109), UINT8_C(153), UINT8_C(177), UINT8_C(121),
        UINT8_C( 45), UINT8_C( 54), UINT8_C(203), UINT8_C(109), UINT8_C( 46), UINT8_C( 89), UINT8_C( 66), UINT8_C(149),
        UINT8_C( 88), UINT8_C(128), UINT8_C(240), UINT8_C(247), UINT8_C(224), UINT8_C(101), UINT8_C( 88), UINT8_C( 24),
        UINT8_C(151), UINT8_C( 19), UINT8_C( 84), UINT8_C(  7), UINT8_C(243), UINT8_C(197), UINT8_C(158), UINT8_C(241),
        UINT8_C(139), UINT8_C(163), UINT8_C( 94), UINT8_C(248), UINT8_C( 61), UINT8_C( 15), UINT8_C(113), UINT8_C(106),
        UINT8_C( 69), UINT8_C( 61), UINT8_C(216), UINT8_C(115), UINT8_C(150), UINT8_C( 26), UINT8_C(  8), UINT8_C(238) },
      { UINT8_C(155), UINT8_C(249), UINT8_C(229), UINT8_C(123), UINT8_C( 94), UINT8_C( 62), UINT8_C(147), UINT8_C(245),
        UINT8_C( 81), UINT8_C(231), UINT8_C(253), UINT8_C( 68), UINT8_C(172), UINT8_C(155), UINT8_C( 53), UINT8_C( 55),
        UINT8_C( 63), UINT8_C(147), UINT8_C( 47), UINT8_C(124), UINT8_C(162), UINT8_C(161), UINT8_C(230), UINT8_C(231),
        UINT8_C(222), UINT8_C(190), UINT8_C( 90), UINT8_C(116), UINT8_C(217), UINT8_C( 99), UINT8_C( 98), UINT8_C(116),
        UINT8_C( 92), UINT8_C( 72), UINT8_C(239), UINT8_C(186), UINT8_C(134), UINT8_C(130), UINT8_C(176), UINT8_C(215),
        UINT8_C(105), UINT8_C(173), UINT8_C( 27), UINT8_C( 22), UINT8_C( 72), UINT8_C( 80), UINT8_C( 77), UINT8_C(135),
        UINT8_C(227), UINT8_C(125), UINT8_C(  3), UINT8_C(133), UINT8_C( 30), UINT8_C(234), UINT8_C(108), UINT8_C(252),
        UINT8_C(168), UINT8_C(198), UINT8_C(112), UINT8_C(129), UINT8_C( 41), UINT8_C(210), UINT8_C(245), UINT8_C(133) },
      { UINT8_C(223), UINT8_C(132), UINT8_C(229), UINT8_C(123), UINT8_C(108), UINT8_C(224), UINT8_C(172), UINT8_C( 50),
        UINT8_C(246), UINT8_C(233), UINT8_C(253), UINT8_C( 54), UINT8_C(172), UINT8_C(135), UINT8_C( 45), UINT8_C(151),
        UINT8_C(254), UINT8_C( 36), UINT8_C( 47), UINT8_C(163), UINT8_C(150), UINT8_C( 44), UINT8_C(130), UINT8_C(231),
        UINT8_C( 47), UINT8_C(190), UINT8_C(248), UINT8_C( 33), UINT8_C(217), UINT8_C( 81), UINT8_C( 98), UINT8_C( 38),
        UINT8_C(213), UINT8_C(101),    UINT8_MAX, UINT8_C( 65), UINT8_C( 69), UINT8_C(172), UINT8_C(176), UINT8_C( 59),
        UINT8_C(149), UINT8_C(110), UINT8_C(114), UINT8_C( 22), UINT8_C(243), UINT8_C(197), UINT8_C(158), UINT8_C(241),
        UINT8_C(227), UINT8_C(216), UINT8_C( 94), UINT8_C(248), UINT8_C(  4), UINT8_C( 63), UINT8_C( 37), UINT8_C(252),
        UINT8_C(168), UINT8_C( 30), UINT8_C(216), UINT8_C(110), UINT8_C(150), UINT8_C(167), UINT8_C(245), UINT8_C(238) } },
    { {  INT8_C(  26), -INT8_C(  28),  INT8_C(  64), -INT8_C(  96),  INT8_C( 102), -INT8_C(  16),  INT8_C( 119), -INT8_C(  48),
        -INT8_C(  99), -INT8_C( 110), -INT8_C(  26), -INT8_C(  27), -INT8_C(  30),  INT8_C(  51),  INT8_C( 109), -INT8_C(  59),
        -INT8_C(  80),  INT8_C( 112),  INT8_C(  74), -INT8_C(  50),  INT8_C(  90), -INT8_C(  74), -INT8_C(  54),  INT8_C(   3),
         INT8_C( 125),  INT8_C(  58), -INT8_C( 124), -INT8_C(  90),  INT8_C(  13),  INT8_C( 122),  INT8_C(  44),  INT8_C(  39),
         INT8_C(  94),  INT8_C( 108), -INT8_C(  56), -INT8_C(  59),  INT8_C(  92),  INT8_C(  63), -INT8_C( 107), -INT8_C(   7),
        -INT8_C(  46),  INT8_C( 123), -INT8_C(  34), -INT8_C(  76), -INT8_C(  82),  INT8_C(  75),  INT8_C( 122),  INT8_C(  95),
        -INT8_C(  68), -INT8_C(  60),  INT8_C(  45),  INT8_C(  22),  INT8_C( 123), -INT8_C(   8),  INT8_C(  25), -INT8_C(   8),
         INT8_C(  50), -INT8_C(  98), -INT8_C(  98),  INT8_C(  63),  INT8_C(  24), -INT8_C(  54),  INT8_C( 103),  INT8_C( 118) },
      UINT64_C( 4651040213508501302),
      { UINT8_C( 75), UINT8_C(106), UINT8_C(245), UINT8_C(250), UINT8_C(181), UINT8_C(111), UINT8_C( 89), UINT8_C(113),
        UINT8_C( 51), UINT8_C(134), UINT8_C(136), UINT8_C(174), UINT8_C(126), UINT8_C(161), UINT8_C(166), UINT8_C(177),
        UINT8_C( 63), UINT8_C( 69), UINT8_C(240), UINT8_C( 87), UINT8_C( 15), UINT8_C( 87), UINT8_C(206), UINT8_C( 70),
        UINT8_C(134), UINT8_C(  9), UINT8_C(216), UINT8_C(245), UINT8_C(218), UINT8_C(100), UINT8_C( 53), UINT8_C( 37),
        UINT8_C(206), UINT8_C( 42), UINT8_C( 31), UINT8_C(131), UINT8_C(153), UINT8_C(120), UINT8_C(245), UINT8_C(205),
           UINT8_MAX, UINT8_C(125), UINT8_C(123), UINT8_C(125), UINT8_C( 30), UINT8_C( 34), UINT8_C( 46), UINT8_C( 94),
        UINT8_C(103), UINT8_C( 31), UINT8_C(181), UINT8_C(118), UINT8_C(118), UINT8_C(131), UINT8_C(188), UINT8_C(253),
        UINT8_C(141), UINT8_C(149), UINT8_C(242), UINT8_C(103), UINT8_C(249), UINT8_C( 39), UINT8_C(140), UINT8_C(199) },
      { UINT8_C( 82), UINT8_C(172), UINT8_C( 74), UINT8_C(235), UINT8_C( 36), UINT8_C( 63), UINT8_C(184), UINT8_C( 35),
        UINT8_C(188), UINT8_C( 52), UINT8_C(161), UINT8_C(219), UINT8_C( 86), UINT8_C(207), UINT8_C( 57), UINT8_C(189),
        UINT8_C(238), UINT8_C(238), UINT8_C( 51), UINT8_C(101), UINT8_C(114), UINT8_C(240), UINT8_C( 98),    UINT8_MAX,
        UINT8_C(133), UINT8_C( 84), UINT8_C(102), UINT8_C(126), UINT8_C(123), UINT8_C(242), UINT8_C( 69), UINT8_C(205),
        UINT8_C(158), UINT8_C(143), UINT8_C(185), UINT8_C(195), UINT8_C(207), UINT8_C(113), UINT8_C(230), UINT8_C(139),
        UINT8_C(165), UINT8_C(135), UINT8_C(102), UINT8_C(251), UINT8_C( 87), UINT8_C(159), UINT8_C(184), UINT8_C( 69),
        UINT8_C(142), UINT8_C(236), UINT8_C(170), UINT8_C(  0), UINT8_C(220), UINT8_C( 12),    UINT8_MAX, UINT8_C( 97),
        UINT8_C( 96), UINT8_C(101), UINT8_C(223), UINT8_C(220), UINT8_C( 87), UINT8_C( 36), UINT8_C(169), UINT8_C(246) },
      { UINT8_C( 26), UINT8_C(172), UINT8_C(245), UINT8_C(160), UINT8_C(181), UINT8_C(111), UINT8_C(119), UINT8_C(208),
        UINT8_C(188), UINT8_C(134), UINT8_C(161), UINT8_C(219), UINT8_C(226), UINT8_C(207), UINT8_C(109), UINT8_C(197),
        UINT8_C(238), UINT8_C(238), UINT8_C( 74), UINT8_C(101), UINT8_C(114), UINT8_C(240), UINT8_C(202), UINT8_C(  3),
        UINT8_C(125), UINT8_C( 84), UINT8_C(132), UINT8_C(166), UINT8_C(218), UINT8_C(122), UINT8_C( 44), UINT8_C(205),
        UINT8_C( 94), UINT8_C(143), UINT8_C(185), UINT8_C(195), UINT8_C( 92), UINT8_C(120), UINT8_C(245), UINT8_C(249),
        UINT8_C(210), UINT8_C(123), UINT8_C(222), UINT8_C(180), UINT8_C( 87), UINT8_C( 75), UINT8_C(184), UINT8_C( 94),
        UINT8_C(142), UINT8_C(236), UINT8_C( 45), UINT8_C(118), UINT8_C(123), UINT8_C(248), UINT8_C( 25), UINT8_C(253),
        UINT8_C( 50), UINT8_C(158), UINT8_C(158), UINT8_C( 63), UINT8_C( 24), UINT8_C(202), UINT8_C(169), UINT8_C(118) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi8(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k  = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_epu8(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_epu8");
    easysimd_test_x86_assert_equal_u8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_epu8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask64 k;
    const uint8_t a[64];
    const uint8_t b[64];
    const uint8_t r[64];
  } test_vec[] = {
    { UINT64_C(  768989401926260750),
      { UINT8_C(200), UINT8_C(153), UINT8_C(184), UINT8_C(140), UINT8_C(198), UINT8_C(202), UINT8_C(130), UINT8_C(163),
        UINT8_C( 22), UINT8_C(136), UINT8_C(176), UINT8_C(145), UINT8_C( 38), UINT8_C( 80), UINT8_C(249), UINT8_C( 26),
        UINT8_C(160), UINT8_C( 32), UINT8_C(211), UINT8_C( 88), UINT8_C(134), UINT8_C(202), UINT8_C(182), UINT8_C(148),
        UINT8_C(163), UINT8_C(144), UINT8_C(253), UINT8_C(108), UINT8_C(143), UINT8_C(168), UINT8_C(118), UINT8_C( 87),
        UINT8_C( 65), UINT8_C( 47), UINT8_C(228), UINT8_C(  8), UINT8_C(249), UINT8_C(102), UINT8_C(171), UINT8_C( 15),
        UINT8_C(238), UINT8_C( 91), UINT8_C(160), UINT8_C( 21), UINT8_C(171), UINT8_C(153), UINT8_C( 47), UINT8_C( 75),
        UINT8_C(185), UINT8_C(  2), UINT8_C(163), UINT8_C( 64), UINT8_C(205), UINT8_C( 89), UINT8_C(212), UINT8_C(112),
        UINT8_C(233), UINT8_C(210), UINT8_C(220), UINT8_C(121), UINT8_C(122), UINT8_C( 83), UINT8_C(208), UINT8_C(188) },
      { UINT8_C(130), UINT8_C(180), UINT8_C(196), UINT8_C(123), UINT8_C( 26), UINT8_C(111), UINT8_C(138), UINT8_C(  9),
        UINT8_C(202), UINT8_C( 43), UINT8_C( 30), UINT8_C(117), UINT8_C(196), UINT8_C( 77), UINT8_C(192), UINT8_C(126),
        UINT8_C( 80), UINT8_C( 99), UINT8_C(190), UINT8_C( 29), UINT8_C(188), UINT8_C(146), UINT8_C(141), UINT8_C(165),
        UINT8_C(100), UINT8_C(105), UINT8_C( 30), UINT8_C(223), UINT8_C(188), UINT8_C(239), UINT8_C(155), UINT8_C( 62),
        UINT8_C(163), UINT8_C( 95), UINT8_C(185), UINT8_C(190), UINT8_C(206), UINT8_C( 68), UINT8_C(199), UINT8_C(152),
        UINT8_C(111), UINT8_C(229), UINT8_C( 13), UINT8_C( 51), UINT8_C( 50), UINT8_C(205), UINT8_C(177), UINT8_C(130),
        UINT8_C( 48), UINT8_C(111), UINT8_C(159), UINT8_C(236), UINT8_C(  2), UINT8_C( 44), UINT8_C(145), UINT8_C(102),
        UINT8_C(150), UINT8_C(176), UINT8_C( 69), UINT8_C( 82), UINT8_C(159), UINT8_C(224), UINT8_C(145), UINT8_C( 66) },
      { UINT8_C(  0), UINT8_C(180), UINT8_C(196), UINT8_C(140), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(145), UINT8_C(196), UINT8_C(  0), UINT8_C(249), UINT8_C(126),
        UINT8_C(  0), UINT8_C( 99), UINT8_C(  0), UINT8_C( 88), UINT8_C(188), UINT8_C(  0), UINT8_C(182), UINT8_C(165),
        UINT8_C(163), UINT8_C(  0), UINT8_C(  0), UINT8_C(223), UINT8_C(  0), UINT8_C(239), UINT8_C(155), UINT8_C(  0),
        UINT8_C(163), UINT8_C(  0), UINT8_C(  0), UINT8_C(190), UINT8_C(  0), UINT8_C(  0), UINT8_C(199), UINT8_C(152),
        UINT8_C(238), UINT8_C(229), UINT8_C(160), UINT8_C( 51), UINT8_C(171), UINT8_C(205), UINT8_C(177), UINT8_C(130),
        UINT8_C(185), UINT8_C(111), UINT8_C(  0), UINT8_C(236), UINT8_C(  0), UINT8_C( 89), UINT8_C(  0), UINT8_C(112),
        UINT8_C(  0), UINT8_C(210), UINT8_C(  0), UINT8_C(121), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0) } },
    { UINT64_C(18277234075670432319),
      { UINT8_C(172), UINT8_C(178), UINT8_C( 49), UINT8_C(223), UINT8_C(127), UINT8_C(226), UINT8_C( 97), UINT8_C(175),
        UINT8_C( 82), UINT8_C(  1), UINT8_C(155), UINT8_C( 84), UINT8_C( 45), UINT8_C( 45), UINT8_C(186), UINT8_C(195),
        UINT8_C(221), UINT8_C(  0), UINT8_C( 22), UINT8_C(124), UINT8_C(224), UINT8_C(167), UINT8_C(190), UINT8_C( 32),
        UINT8_C(241), UINT8_C(191), UINT8_C( 45), UINT8_C(128), UINT8_C(134), UINT8_C(211), UINT8_C(125), UINT8_C( 51),
        UINT8_C(133), UINT8_C(174), UINT8_C( 18), UINT8_C(  5), UINT8_C(145), UINT8_C(115), UINT8_C(180), UINT8_C(227),
        UINT8_C(116), UINT8_C( 80), UINT8_C( 55), UINT8_C(162), UINT8_C(125), UINT8_C(241), UINT8_C(101), UINT8_C( 90),
        UINT8_C(241), UINT8_C(123), UINT8_C(214), UINT8_C(210), UINT8_C( 34), UINT8_C(148), UINT8_C(242), UINT8_C( 20),
        UINT8_C( 83), UINT8_C( 31), UINT8_C(148), UINT8_C(218), UINT8_C(242), UINT8_C( 17), UINT8_C( 13), UINT8_C(120) },
      { UINT8_C(192), UINT8_C( 31), UINT8_C(125), UINT8_C( 81), UINT8_C(146), UINT8_C( 49), UINT8_C( 52), UINT8_C(  7),
        UINT8_C(129), UINT8_C(107), UINT8_C(169), UINT8_C(254), UINT8_C( 92), UINT8_C( 14), UINT8_C( 88), UINT8_C( 78),
        UINT8_C(138), UINT8_C( 46), UINT8_C( 32), UINT8_C(172), UINT8_C(195), UINT8_C( 18), UINT8_C(192), UINT8_C( 22),
        UINT8_C( 49), UINT8_C( 84), UINT8_C(240), UINT8_C( 36), UINT8_C(102), UINT8_C(253), UINT8_C(156), UINT8_C( 38),
        UINT8_C( 28), UINT8_C( 25), UINT8_C(119), UINT8_C(175), UINT8_C( 74), UINT8_C(171), UINT8_C(182), UINT8_C(204),
        UINT8_C( 22), UINT8_C( 95), UINT8_C(202), UINT8_C(114), UINT8_C(109), UINT8_C( 35), UINT8_C(192), UINT8_C(247),
        UINT8_C( 81), UINT8_C(224), UINT8_C(164), UINT8_C( 20), UINT8_C(242), UINT8_C(100), UINT8_C( 43), UINT8_C( 36),
        UINT8_C(185), UINT8_C( 27), UINT8_C( 72), UINT8_C( 31), UINT8_C( 25), UINT8_C(228), UINT8_C( 69), UINT8_C( 53) },
      { UINT8_C(192), UINT8_C(178), UINT8_C(125), UINT8_C(223), UINT8_C(146), UINT8_C(226), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(107), UINT8_C(  0), UINT8_C(254), UINT8_C(  0), UINT8_C(  0), UINT8_C(186), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(241), UINT8_C(  0), UINT8_C(240), UINT8_C(128), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(174), UINT8_C(119), UINT8_C(175), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(227),
        UINT8_C(116), UINT8_C( 95), UINT8_C(202), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(192), UINT8_C(247),
        UINT8_C(241), UINT8_C(  0), UINT8_C(214), UINT8_C(  0), UINT8_C(  0), UINT8_C(148), UINT8_C(  0), UINT8_C( 36),
        UINT8_C(185), UINT8_C(  0), UINT8_C(148), UINT8_C(218), UINT8_C(242), UINT8_C(228), UINT8_C( 69), UINT8_C(120) } },
    { UINT64_C( 9012717047676976381),
      { UINT8_C(249), UINT8_C(222), UINT8_C(239), UINT8_C(103), UINT8_C(  1), UINT8_C(176), UINT8_C( 94), UINT8_C( 82),
        UINT8_C(144), UINT8_C(  2), UINT8_C(103), UINT8_C(131), UINT8_C(103), UINT8_C(146), UINT8_C(167), UINT8_C( 32),
        UINT8_C(173), UINT8_C(239), UINT8_C( 63), UINT8_C(198), UINT8_C(211), UINT8_C(132), UINT8_C(252), UINT8_C(208),
        UINT8_C( 64), UINT8_C(224), UINT8_C( 23), UINT8_C(167), UINT8_C(123), UINT8_C( 43), UINT8_C( 36), UINT8_C(116),
        UINT8_C(  9), UINT8_C( 19), UINT8_C(219), UINT8_C( 10), UINT8_C(195), UINT8_C( 58), UINT8_C( 92), UINT8_C( 84),
        UINT8_C( 60), UINT8_C(195), UINT8_C(215), UINT8_C(163), UINT8_C( 85), UINT8_C(126), UINT8_C(195), UINT8_C(  3),
        UINT8_C(109), UINT8_C(  2), UINT8_C(201), UINT8_C( 64), UINT8_C(134), UINT8_C(197), UINT8_C( 16), UINT8_C(198),
        UINT8_C(166), UINT8_C( 39), UINT8_C(109), UINT8_C( 33), UINT8_C( 82), UINT8_C(145), UINT8_C(149), UINT8_C( 91) },
      { UINT8_C(165), UINT8_C(113), UINT8_C(101), UINT8_C(104), UINT8_C(171), UINT8_C(194), UINT8_C(188), UINT8_C(231),
        UINT8_C(133), UINT8_C(147), UINT8_C(139), UINT8_C(219), UINT8_C( 17), UINT8_C( 78), UINT8_C(222), UINT8_C(126),
        UINT8_C( 81), UINT8_C(167), UINT8_C(190), UINT8_C(215), UINT8_C(109), UINT8_C(206), UINT8_C(158), UINT8_C( 19),
        UINT8_C(246), UINT8_C( 11), UINT8_C( 52), UINT8_C( 72), UINT8_C(157), UINT8_C(201), UINT8_C(164), UINT8_C( 66),
        UINT8_C( 58), UINT8_C(  9), UINT8_C(170), UINT8_C(229), UINT8_C(203), UINT8_C(103), UINT8_C(205), UINT8_C( 81),
        UINT8_C(250), UINT8_C( 88), UINT8_C( 44), UINT8_C( 12), UINT8_C(166), UINT8_C( 10), UINT8_C(138), UINT8_C(247),
        UINT8_C(177), UINT8_C( 73), UINT8_C(207), UINT8_C( 30), UINT8_C( 23), UINT8_C(109), UINT8_C( 49), UINT8_C( 13),
        UINT8_C(120), UINT8_C(101), UINT8_C( 86), UINT8_C( 21), UINT8_C( 47), UINT8_C(250), UINT8_C( 87), UINT8_C(105) },
      { UINT8_C(249), UINT8_C(  0), UINT8_C(239), UINT8_C(104), UINT8_C(171), UINT8_C(194), UINT8_C(188), UINT8_C(231),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(139), UINT8_C(219), UINT8_C(103), UINT8_C(146), UINT8_C(  0), UINT8_C(126),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(190), UINT8_C(  0), UINT8_C(  0), UINT8_C(206), UINT8_C(252), UINT8_C(208),
        UINT8_C(246), UINT8_C(224), UINT8_C( 52), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(164), UINT8_C(  0),
        UINT8_C( 58), UINT8_C( 19), UINT8_C(219), UINT8_C(  0), UINT8_C(  0), UINT8_C(103), UINT8_C(205), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(195), UINT8_C(  0), UINT8_C(163), UINT8_C(166), UINT8_C(  0), UINT8_C(  0), UINT8_C(247),
        UINT8_C(177), UINT8_C( 73), UINT8_C(  0), UINT8_C(  0), UINT8_C(134), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(166), UINT8_C(  0), UINT8_C(109), UINT8_C( 33), UINT8_C( 82), UINT8_C(250), UINT8_C(149), UINT8_C(  0) } },
    { UINT64_C( 7142740249784812035),
      { UINT8_C(116), UINT8_C( 76), UINT8_C(111), UINT8_C( 26), UINT8_C( 86), UINT8_C(250), UINT8_C( 18), UINT8_C(  7),
        UINT8_C( 67), UINT8_C(225), UINT8_C( 38), UINT8_C( 90), UINT8_C( 78), UINT8_C( 87), UINT8_C(104), UINT8_C(198),
        UINT8_C(189), UINT8_C(190), UINT8_C(220), UINT8_C(236), UINT8_C(184), UINT8_C( 51), UINT8_C( 85), UINT8_C(187),
        UINT8_C( 53), UINT8_C(164), UINT8_C(138), UINT8_C(158), UINT8_C(192), UINT8_C(170), UINT8_C(  2), UINT8_C( 52),
        UINT8_C(246), UINT8_C(113), UINT8_C( 79), UINT8_C( 76), UINT8_C(107), UINT8_C( 97), UINT8_C( 84), UINT8_C(174),
        UINT8_C( 66), UINT8_C(122), UINT8_C(  9), UINT8_C(144), UINT8_C(209), UINT8_C(113), UINT8_C( 86), UINT8_C(142),
        UINT8_C( 47), UINT8_C( 50), UINT8_C(122), UINT8_C(231), UINT8_C(102), UINT8_C(208), UINT8_C(162), UINT8_C(155),
        UINT8_C(116), UINT8_C( 45), UINT8_C( 58), UINT8_C( 53), UINT8_C(215), UINT8_C( 60), UINT8_C(105), UINT8_C(206) },
      { UINT8_C(173), UINT8_C(184), UINT8_C( 26), UINT8_C( 25), UINT8_C( 25), UINT8_C(110), UINT8_C(199), UINT8_C( 91),
        UINT8_C(232), UINT8_C(208), UINT8_C(235), UINT8_C(186), UINT8_C( 65), UINT8_C( 66), UINT8_C( 72), UINT8_C(112),
        UINT8_C(116), UINT8_C(195), UINT8_C( 87), UINT8_C(218), UINT8_C(147), UINT8_C(250), UINT8_C(118), UINT8_C(  7),
        UINT8_C( 39), UINT8_C(176), UINT8_C( 60), UINT8_C(254), UINT8_C(236), UINT8_C(166), UINT8_C(204), UINT8_C(153),
        UINT8_C( 94), UINT8_C(231), UINT8_C(178), UINT8_C(120), UINT8_C( 85), UINT8_C(122), UINT8_C(211), UINT8_C( 62),
        UINT8_C( 74), UINT8_C(191), UINT8_C(248), UINT8_C(140), UINT8_C(  1), UINT8_C( 64), UINT8_C(252), UINT8_C(117),
        UINT8_C(  3), UINT8_C( 84), UINT8_C( 80), UINT8_C(150), UINT8_C( 78), UINT8_C(198), UINT8_C(158), UINT8_C(117),
        UINT8_C(118), UINT8_C(218), UINT8_C(115), UINT8_C( 98), UINT8_C(128), UINT8_C( 64), UINT8_C(251), UINT8_C(223) },
      { UINT8_C(173), UINT8_C(184), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(225), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(189), UINT8_C(195), UINT8_C(220), UINT8_C(236), UINT8_C(  0), UINT8_C(  0), UINT8_C(118), UINT8_C(  0),
        UINT8_C( 53), UINT8_C(176), UINT8_C(138), UINT8_C(254), UINT8_C(  0), UINT8_C(  0), UINT8_C(204), UINT8_C(153),
        UINT8_C(246), UINT8_C(  0), UINT8_C(  0), UINT8_C(120), UINT8_C(  0), UINT8_C(122), UINT8_C(211), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(248), UINT8_C(144), UINT8_C(209), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(208), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(118), UINT8_C(218), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 64), UINT8_C(251), UINT8_C(  0) } },
    { UINT64_C( 8266966419365146151),
      { UINT8_C(233), UINT8_C(178), UINT8_C(254), UINT8_C(234), UINT8_C(243), UINT8_C(251), UINT8_C( 96), UINT8_C(246),
        UINT8_C( 79), UINT8_C(176), UINT8_C(141), UINT8_C(157), UINT8_C(118), UINT8_C( 43), UINT8_C( 18), UINT8_C(236),
        UINT8_C(  5), UINT8_C(133), UINT8_C( 78), UINT8_C(134), UINT8_C(197), UINT8_C( 73), UINT8_C(101), UINT8_C(236),
        UINT8_C(247), UINT8_C(188), UINT8_C(105), UINT8_C( 31), UINT8_C(230), UINT8_C( 35), UINT8_C(146), UINT8_C(208),
        UINT8_C(214), UINT8_C(144), UINT8_C(186), UINT8_C(201), UINT8_C(139), UINT8_C( 26), UINT8_C(191), UINT8_C(218),
        UINT8_C(202), UINT8_C( 76), UINT8_C(119), UINT8_C( 64), UINT8_C(119), UINT8_C(137), UINT8_C( 44), UINT8_C(125),
        UINT8_C( 15), UINT8_C(122), UINT8_C(  3), UINT8_C(212), UINT8_C(196), UINT8_C(104), UINT8_C(193), UINT8_C(187),
        UINT8_C( 36), UINT8_C( 42), UINT8_C(219), UINT8_C( 10), UINT8_C( 77), UINT8_C(109), UINT8_C(218), UINT8_C( 35) },
      { UINT8_C(253), UINT8_C(149), UINT8_C(236), UINT8_C(137), UINT8_C(175), UINT8_C(172), UINT8_C( 99), UINT8_C(122),
        UINT8_C(248), UINT8_C(219), UINT8_C(186), UINT8_C(112), UINT8_C(100), UINT8_C(231), UINT8_C(237), UINT8_C(115),
        UINT8_C( 97), UINT8_C(240), UINT8_C( 72), UINT8_C( 37), UINT8_C( 88), UINT8_C(  9), UINT8_C(225), UINT8_C(124),
        UINT8_C( 51), UINT8_C(188), UINT8_C(134), UINT8_C(128), UINT8_C( 41), UINT8_C( 97), UINT8_C(164), UINT8_C( 38),
        UINT8_C(246), UINT8_C(144), UINT8_C(175), UINT8_C(165), UINT8_C( 60), UINT8_C( 19), UINT8_C( 31), UINT8_C( 53),
        UINT8_C(238), UINT8_C(218), UINT8_C(165), UINT8_C( 82), UINT8_C(193), UINT8_C(146), UINT8_C(198), UINT8_C( 34),
        UINT8_C(130), UINT8_C( 14), UINT8_C( 72), UINT8_C(218), UINT8_C( 23), UINT8_C( 41), UINT8_C( 86), UINT8_C( 74),
        UINT8_C(229), UINT8_C(220), UINT8_C(202), UINT8_C( 14), UINT8_C( 61), UINT8_C(110), UINT8_C( 52), UINT8_C( 51) },
      { UINT8_C(253), UINT8_C(178), UINT8_C(254), UINT8_C(  0), UINT8_C(  0), UINT8_C(251), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(219), UINT8_C(186), UINT8_C(157), UINT8_C(  0), UINT8_C(231), UINT8_C(  0), UINT8_C(236),
        UINT8_C( 97), UINT8_C(240), UINT8_C( 78), UINT8_C(  0), UINT8_C(197), UINT8_C(  0), UINT8_C(225), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(134), UINT8_C(128), UINT8_C(230), UINT8_C( 97), UINT8_C(164), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(201), UINT8_C(  0), UINT8_C( 26), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(218), UINT8_C(  0), UINT8_C( 82), UINT8_C(  0), UINT8_C(146), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(122), UINT8_C(  0), UINT8_C(218), UINT8_C(196), UINT8_C(104), UINT8_C(  0), UINT8_C(187),
        UINT8_C(  0), UINT8_C(220), UINT8_C(  0), UINT8_C(  0), UINT8_C( 77), UINT8_C(110), UINT8_C(218), UINT8_C(  0) } },
    { UINT64_C(16532987972821837055),
      { UINT8_C(210), UINT8_C( 21), UINT8_C( 55), UINT8_C(147), UINT8_C(167), UINT8_C(253), UINT8_C(182), UINT8_C( 41),
        UINT8_C( 11), UINT8_C(254), UINT8_C(  3), UINT8_C( 34), UINT8_C( 39), UINT8_C( 89), UINT8_C(108), UINT8_C( 12),
        UINT8_C( 54), UINT8_C( 55), UINT8_C( 26), UINT8_C(115), UINT8_C(165), UINT8_C( 78), UINT8_C(167), UINT8_C(164),
        UINT8_C( 50), UINT8_C(128), UINT8_C(224), UINT8_C( 41), UINT8_C(120), UINT8_C( 80), UINT8_C( 14), UINT8_C( 75),
        UINT8_C(102), UINT8_C( 70), UINT8_C(222), UINT8_C( 13), UINT8_C( 67), UINT8_C(148), UINT8_C( 55), UINT8_C( 79),
        UINT8_C(146), UINT8_C( 58), UINT8_C(113), UINT8_C(185), UINT8_C(148), UINT8_C(222), UINT8_C(197), UINT8_C(202),
        UINT8_C( 21), UINT8_C(223), UINT8_C( 61), UINT8_C(186), UINT8_C( 46), UINT8_C(228), UINT8_C( 95), UINT8_C( 96),
        UINT8_C(100), UINT8_C( 63), UINT8_C(138), UINT8_C(221), UINT8_C(143), UINT8_C(152), UINT8_C( 40), UINT8_C(245) },
      { UINT8_C(222), UINT8_C(  6), UINT8_C(  3), UINT8_C( 34), UINT8_C(155), UINT8_C( 58), UINT8_C(113), UINT8_C( 45),
        UINT8_C(116), UINT8_C(226), UINT8_C(231), UINT8_C(  8), UINT8_C(192), UINT8_C(172), UINT8_C(210), UINT8_C(213),
        UINT8_C(140), UINT8_C( 16), UINT8_C(144), UINT8_C(186), UINT8_C(244), UINT8_C(239), UINT8_C( 26), UINT8_C( 89),
        UINT8_C( 46), UINT8_C(164), UINT8_C( 54), UINT8_C(189), UINT8_C( 61), UINT8_C( 94), UINT8_C(179), UINT8_C( 27),
        UINT8_C(100), UINT8_C(182), UINT8_C( 61),    UINT8_MAX, UINT8_C(240), UINT8_C(174), UINT8_C( 45), UINT8_C(100),
        UINT8_C(145), UINT8_C( 20), UINT8_C(109), UINT8_C( 81), UINT8_C(192), UINT8_C( 63), UINT8_C( 39), UINT8_C( 76),
        UINT8_C( 79), UINT8_C(183), UINT8_C(  6), UINT8_C( 68), UINT8_C(166), UINT8_C( 33), UINT8_C(157), UINT8_C(212),
        UINT8_C(197), UINT8_C(211), UINT8_C(145), UINT8_C(  2), UINT8_C( 49), UINT8_C( 68), UINT8_C( 30), UINT8_C(149) },
      { UINT8_C(222), UINT8_C( 21), UINT8_C( 55), UINT8_C(147), UINT8_C(167), UINT8_C(253), UINT8_C(182), UINT8_C( 45),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(231), UINT8_C(  0), UINT8_C(  0), UINT8_C(172), UINT8_C(210), UINT8_C(213),
        UINT8_C(140), UINT8_C(  0), UINT8_C(  0), UINT8_C(186), UINT8_C(244), UINT8_C(  0), UINT8_C(167), UINT8_C(164),
        UINT8_C( 50), UINT8_C(164), UINT8_C(  0), UINT8_C(189), UINT8_C(120), UINT8_C( 94), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(102), UINT8_C(182), UINT8_C(222), UINT8_C(  0), UINT8_C(240), UINT8_C(174), UINT8_C( 55), UINT8_C(100),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(185), UINT8_C(192), UINT8_C(222), UINT8_C(197), UINT8_C(202),
        UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(166), UINT8_C(228), UINT8_C(157), UINT8_C(  0),
        UINT8_C(197), UINT8_C(  0), UINT8_C(145), UINT8_C(  0), UINT8_C(  0), UINT8_C(152), UINT8_C( 40), UINT8_C(245) } },
    { UINT64_C(11191376951180090362),
      { UINT8_C(214), UINT8_C(188), UINT8_C(236), UINT8_C(150), UINT8_C(251), UINT8_C( 19), UINT8_C(227), UINT8_C( 75),
        UINT8_C(202), UINT8_C(233), UINT8_C(143), UINT8_C(112), UINT8_C( 10), UINT8_C( 44), UINT8_C( 68), UINT8_C(208),
           UINT8_MAX, UINT8_C(214), UINT8_C(210), UINT8_C( 48), UINT8_C( 26), UINT8_C(240), UINT8_C(197), UINT8_C( 21),
        UINT8_C( 76), UINT8_C( 90),    UINT8_MAX, UINT8_C( 86), UINT8_C( 28), UINT8_C( 78), UINT8_C(241), UINT8_C(242),
        UINT8_C( 10), UINT8_C(221), UINT8_C(137), UINT8_C(  6), UINT8_C(241), UINT8_C(108), UINT8_C( 81), UINT8_C(187),
        UINT8_C( 85), UINT8_C(224), UINT8_C( 44), UINT8_C( 96), UINT8_C( 12), UINT8_C(112), UINT8_C( 48), UINT8_C( 11),
        UINT8_C( 70), UINT8_C(  2), UINT8_C( 59), UINT8_C( 97), UINT8_C(243), UINT8_C(  0), UINT8_C(118), UINT8_C( 63),
        UINT8_C( 91), UINT8_C(117), UINT8_C(149), UINT8_C(119), UINT8_C(196), UINT8_C(134), UINT8_C(106), UINT8_C(206) },
      { UINT8_C( 99), UINT8_C(243), UINT8_C(212), UINT8_C( 84), UINT8_C( 95), UINT8_C( 37), UINT8_C( 16), UINT8_C(180),
        UINT8_C(  5), UINT8_C( 60), UINT8_C( 20), UINT8_C( 17), UINT8_C(172), UINT8_C( 68), UINT8_C( 28), UINT8_C(243),
        UINT8_C( 71), UINT8_C( 87), UINT8_C( 84), UINT8_C( 58), UINT8_C( 88), UINT8_C(202), UINT8_C(121), UINT8_C(179),
        UINT8_C( 63), UINT8_C( 14), UINT8_C( 42), UINT8_C(  3), UINT8_C(148), UINT8_C(148), UINT8_C(210), UINT8_C(247),
        UINT8_C(135), UINT8_C(166), UINT8_C( 76), UINT8_C(230), UINT8_C(204), UINT8_C( 92), UINT8_C(155), UINT8_C(209),
        UINT8_C(152), UINT8_C(175), UINT8_C(227), UINT8_C( 68), UINT8_C(244),    UINT8_MAX, UINT8_C( 55), UINT8_C( 59),
        UINT8_C( 87), UINT8_C(139), UINT8_C(117), UINT8_C(175), UINT8_C( 85), UINT8_C(238), UINT8_C( 98), UINT8_C(149),
        UINT8_C(252), UINT8_C(140), UINT8_C(152), UINT8_C(144), UINT8_C( 33), UINT8_C(106), UINT8_C(135), UINT8_C(168) },
      { UINT8_C(  0), UINT8_C(243), UINT8_C(  0), UINT8_C(150), UINT8_C(251), UINT8_C( 37), UINT8_C(227), UINT8_C(180),
        UINT8_C(202), UINT8_C(233), UINT8_C(  0), UINT8_C(112), UINT8_C(172), UINT8_C(  0), UINT8_C( 68), UINT8_C(  0),
           UINT8_MAX, UINT8_C(  0), UINT8_C(210), UINT8_C(  0), UINT8_C( 88), UINT8_C(  0), UINT8_C(  0), UINT8_C(179),
        UINT8_C(  0), UINT8_C( 90), UINT8_C(  0), UINT8_C( 86), UINT8_C(  0), UINT8_C(148), UINT8_C(241), UINT8_C(247),
        UINT8_C(  0), UINT8_C(221), UINT8_C(  0), UINT8_C(230), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(224), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C( 55), UINT8_C( 59),
        UINT8_C( 87), UINT8_C(139), UINT8_C(117), UINT8_C(175), UINT8_C(  0), UINT8_C(  0), UINT8_C(118), UINT8_C(  0),
        UINT8_C(252), UINT8_C(140), UINT8_C(  0), UINT8_C(144), UINT8_C(196), UINT8_C(  0), UINT8_C(  0), UINT8_C(206) } },
    { UINT64_C(14388484244564333329),
      { UINT8_C(217), UINT8_C(145), UINT8_C( 12), UINT8_C(205), UINT8_C(145), UINT8_C( 67), UINT8_C(  8), UINT8_C(232),
        UINT8_C(207), UINT8_C(125), UINT8_C(151), UINT8_C( 36), UINT8_C(107), UINT8_C(249), UINT8_C(185), UINT8_C(103),
        UINT8_C(133), UINT8_C( 82), UINT8_C(247), UINT8_C(166), UINT8_C(188), UINT8_C(127), UINT8_C( 79), UINT8_C(205),
        UINT8_C( 82), UINT8_C(222), UINT8_C(170), UINT8_C(130), UINT8_C(  8), UINT8_C( 89), UINT8_C( 73), UINT8_C(225),
        UINT8_C(234), UINT8_C( 85), UINT8_C(175), UINT8_C(123), UINT8_C(153), UINT8_C(183), UINT8_C( 99), UINT8_C(104),
        UINT8_C( 53), UINT8_C(250), UINT8_C(140), UINT8_C(160), UINT8_C(243), UINT8_C( 70), UINT8_C(  8), UINT8_C(121),
        UINT8_C(152),    UINT8_MAX, UINT8_C( 31), UINT8_C( 84), UINT8_C(126), UINT8_C(110), UINT8_C( 34), UINT8_C(209),
        UINT8_C( 76), UINT8_C(204), UINT8_C( 83), UINT8_C( 84), UINT8_C( 37), UINT8_C(156), UINT8_C( 54), UINT8_C( 16) },
      { UINT8_C(242), UINT8_C(229), UINT8_C(139), UINT8_C(139), UINT8_C(156), UINT8_C(239), UINT8_C(243), UINT8_C(209),
        UINT8_C(233), UINT8_C(127), UINT8_C(114), UINT8_C(221), UINT8_C(197), UINT8_C(122), UINT8_C( 86), UINT8_C( 93),
        UINT8_C(121), UINT8_C(117), UINT8_C(178), UINT8_C(248), UINT8_C(228), UINT8_C(212), UINT8_C(201), UINT8_C( 48),
        UINT8_C(160), UINT8_C( 28), UINT8_C(133), UINT8_C(198), UINT8_C(184), UINT8_C(187), UINT8_C(214), UINT8_C(170),
        UINT8_C(160), UINT8_C( 97), UINT8_C( 53), UINT8_C( 60), UINT8_C( 80), UINT8_C( 40), UINT8_C( 14), UINT8_C( 58),
        UINT8_C(168), UINT8_C(128), UINT8_C( 23), UINT8_C(109), UINT8_C(250), UINT8_C(109), UINT8_C(203), UINT8_C(115),
        UINT8_C(226), UINT8_C(125), UINT8_C(107), UINT8_C(198), UINT8_C( 81), UINT8_C( 52), UINT8_C(247), UINT8_C(241),
        UINT8_C( 80), UINT8_C(124), UINT8_C(183), UINT8_C(  9), UINT8_C( 55), UINT8_C(141), UINT8_C(179), UINT8_C(215) },
      { UINT8_C(242), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(156), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(233), UINT8_C(127), UINT8_C(  0), UINT8_C(  0), UINT8_C(197), UINT8_C(  0), UINT8_C(185), UINT8_C(103),
        UINT8_C(133), UINT8_C(117), UINT8_C(247), UINT8_C(248), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(205),
        UINT8_C(160), UINT8_C(  0), UINT8_C(170), UINT8_C(198), UINT8_C(184), UINT8_C(  0), UINT8_C(214), UINT8_C(225),
        UINT8_C(234), UINT8_C( 97), UINT8_C(175), UINT8_C(123), UINT8_C(  0), UINT8_C(183), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0), UINT8_C(250), UINT8_C(  0), UINT8_C(160), UINT8_C(  0), UINT8_C(109), UINT8_C(  0), UINT8_C(  0),
        UINT8_C(  0),    UINT8_MAX, UINT8_C(107), UINT8_C(198), UINT8_C(  0), UINT8_C(110), UINT8_C(  0), UINT8_C(241),
        UINT8_C( 80), UINT8_C(204), UINT8_C(183), UINT8_C(  0), UINT8_C(  0), UINT8_C(  0), UINT8_C(179), UINT8_C(215) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi8(test_vec[i].b);
    const easysimd__mmask64 k  = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_epu8(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_epu8");
    easysimd_test_x86_assert_equal_u8x64(r, easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_max_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
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
      {  INT16_C( 14691), -INT16_C(  2237),  INT16_C(  8698),  INT16_C(  9348),  INT16_C( 13857),  INT16_C( 28447),  INT16_C( 25854),  INT16_C( 29188),
         INT16_C( 17420),  INT16_C( 25168), -INT16_C(  5283),  INT16_C( 26495),  INT16_C( 14078), -INT16_C(  2781),  INT16_C( 21647),  INT16_C(  8569),
         INT16_C( 14430),  INT16_C( 28939),  INT16_C( 28799),  INT16_C( 31356),  INT16_C( 27686),  INT16_C( 17232),  INT16_C( 15497),  INT16_C(  5514),
         INT16_C(  6016),  INT16_C( 12919),  INT16_C( 18801),  INT16_C( 28485),  INT16_C( 31615),  INT16_C( 26724),  INT16_C( 30159), -INT16_C(  4868) } },
    { {  INT16_C( 15155), -INT16_C( 19940),  INT16_C( 27051), -INT16_C( 12008), -INT16_C(  4395), -INT16_C( 29975),  INT16_C( 29896),  INT16_C( 16799),
         INT16_C(  5967), -INT16_C( 16269), -INT16_C( 27296), -INT16_C(  8401),  INT16_C( 11024), -INT16_C(  7955),  INT16_C(  7584), -INT16_C( 11381),
        -INT16_C( 22696),  INT16_C(   902), -INT16_C( 25071), -INT16_C(  6443), -INT16_C( 16756),  INT16_C( 21617),  INT16_C(  4146), -INT16_C( 32363),
         INT16_C(  2087), -INT16_C( 30911),  INT16_C( 29086), -INT16_C( 20890),  INT16_C( 21660),  INT16_C( 15758),  INT16_C(  6513), -INT16_C( 14064) },
      { -INT16_C( 26943), -INT16_C( 11572), -INT16_C( 24267), -INT16_C( 15944),  INT16_C( 10592), -INT16_C( 28138), -INT16_C( 21702),  INT16_C( 24852),
         INT16_C( 21940),  INT16_C( 21225),  INT16_C( 20422),  INT16_C( 25344), -INT16_C( 28765),  INT16_C(  5280), -INT16_C( 20312),  INT16_C( 27101),
        -INT16_C( 21945),  INT16_C( 31803), -INT16_C(  2997), -INT16_C( 21699),  INT16_C( 21277),  INT16_C( 22334),  INT16_C( 21247), -INT16_C( 19527),
        -INT16_C( 23897),  INT16_C( 28165),  INT16_C(  1521), -INT16_C( 27183),  INT16_C( 29076),  INT16_C( 15785), -INT16_C( 30943),  INT16_C( 26790) },
      {  INT16_C( 15155), -INT16_C( 11572),  INT16_C( 27051), -INT16_C( 12008),  INT16_C( 10592), -INT16_C( 28138),  INT16_C( 29896),  INT16_C( 24852),
         INT16_C( 21940),  INT16_C( 21225),  INT16_C( 20422),  INT16_C( 25344),  INT16_C( 11024),  INT16_C(  5280),  INT16_C(  7584),  INT16_C( 27101),
        -INT16_C( 21945),  INT16_C( 31803), -INT16_C(  2997), -INT16_C(  6443),  INT16_C( 21277),  INT16_C( 22334),  INT16_C( 21247), -INT16_C( 19527),
         INT16_C(  2087),  INT16_C( 28165),  INT16_C( 29086), -INT16_C( 20890),  INT16_C( 29076),  INT16_C( 15785),  INT16_C(  6513),  INT16_C( 26790) } },
    { { -INT16_C(  7631),  INT16_C( 31972),  INT16_C(  8918), -INT16_C(  3288),  INT16_C( 26229),  INT16_C( 29771),  INT16_C(  1208),  INT16_C( 24359),
         INT16_C( 11430), -INT16_C( 26675), -INT16_C( 25038), -INT16_C( 14804), -INT16_C( 10737),  INT16_C( 12547), -INT16_C( 21923), -INT16_C( 29031),
         INT16_C( 32396),  INT16_C( 25098),  INT16_C( 12960),  INT16_C(  5461), -INT16_C( 24424),  INT16_C( 20618), -INT16_C( 20060),  INT16_C( 19120),
         INT16_C( 32222),  INT16_C(  4322),  INT16_C(  3612),  INT16_C( 11222), -INT16_C(  9500),  INT16_C( 16732), -INT16_C(  2428),  INT16_C(  4303) },
      { -INT16_C(  9612),  INT16_C(  5234), -INT16_C( 14580), -INT16_C( 23255), -INT16_C( 19608),  INT16_C(  3317), -INT16_C( 23195),  INT16_C( 17239),
         INT16_C( 14627),  INT16_C( 16211),  INT16_C( 10567),  INT16_C( 11370), -INT16_C( 14589), -INT16_C( 30867),  INT16_C( 15805),  INT16_C( 12695),
         INT16_C(  2327),  INT16_C(  9029),  INT16_C( 28369),  INT16_C( 14792), -INT16_C( 16862), -INT16_C( 30907), -INT16_C( 25501), -INT16_C( 31030),
         INT16_C(  7637),  INT16_C(  7621),  INT16_C( 12358),  INT16_C( 19017), -INT16_C( 18697), -INT16_C( 19247),  INT16_C( 27123),  INT16_C(  2789) },
      { -INT16_C(  7631),  INT16_C( 31972),  INT16_C(  8918), -INT16_C(  3288),  INT16_C( 26229),  INT16_C( 29771),  INT16_C(  1208),  INT16_C( 24359),
         INT16_C( 14627),  INT16_C( 16211),  INT16_C( 10567),  INT16_C( 11370), -INT16_C( 10737),  INT16_C( 12547),  INT16_C( 15805),  INT16_C( 12695),
         INT16_C( 32396),  INT16_C( 25098),  INT16_C( 28369),  INT16_C( 14792), -INT16_C( 16862),  INT16_C( 20618), -INT16_C( 20060),  INT16_C( 19120),
         INT16_C( 32222),  INT16_C(  7621),  INT16_C( 12358),  INT16_C( 19017), -INT16_C(  9500),  INT16_C( 16732),  INT16_C( 27123),  INT16_C(  4303) } },
    { {  INT16_C( 10866),  INT16_C( 17198), -INT16_C(  2408), -INT16_C( 17796), -INT16_C( 15692),  INT16_C(  6209),  INT16_C(  2910),  INT16_C( 13470),
         INT16_C( 25640),  INT16_C( 28497), -INT16_C( 25964), -INT16_C( 29767), -INT16_C( 30128),  INT16_C( 17471),  INT16_C(  9459),  INT16_C( 26190),
         INT16_C( 31822), -INT16_C(  6487),  INT16_C(  9843),  INT16_C( 10145), -INT16_C(  7448),  INT16_C( 17983), -INT16_C(  8466),  INT16_C(  5754),
        -INT16_C( 13502), -INT16_C( 10619),  INT16_C( 15973), -INT16_C( 18847), -INT16_C( 24375), -INT16_C( 17158),  INT16_C( 18628),  INT16_C(  4642) },
      { -INT16_C( 13115),  INT16_C( 14584), -INT16_C( 26126), -INT16_C(  9633), -INT16_C( 24708),  INT16_C( 27168), -INT16_C( 25731), -INT16_C( 16512),
         INT16_C(  1638), -INT16_C( 13163), -INT16_C(  2492),  INT16_C(  3458),  INT16_C( 31894),  INT16_C( 23242), -INT16_C(  4924), -INT16_C( 30356),
         INT16_C( 25784), -INT16_C( 21823),  INT16_C(  8702),  INT16_C( 31364), -INT16_C( 23104),  INT16_C( 15844),  INT16_C( 25664), -INT16_C( 22788),
        -INT16_C( 28310), -INT16_C( 20622), -INT16_C(  2937),  INT16_C(  7612), -INT16_C( 31120),  INT16_C( 13687), -INT16_C(  7309),  INT16_C( 11198) },
      {  INT16_C( 10866),  INT16_C( 17198), -INT16_C(  2408), -INT16_C(  9633), -INT16_C( 15692),  INT16_C( 27168),  INT16_C(  2910),  INT16_C( 13470),
         INT16_C( 25640),  INT16_C( 28497), -INT16_C(  2492),  INT16_C(  3458),  INT16_C( 31894),  INT16_C( 23242),  INT16_C(  9459),  INT16_C( 26190),
         INT16_C( 31822), -INT16_C(  6487),  INT16_C(  9843),  INT16_C( 31364), -INT16_C(  7448),  INT16_C( 17983),  INT16_C( 25664),  INT16_C(  5754),
        -INT16_C( 13502), -INT16_C( 10619),  INT16_C( 15973),  INT16_C(  7612), -INT16_C( 24375),  INT16_C( 13687),  INT16_C( 18628),  INT16_C( 11198) } },
    { { -INT16_C( 32697),  INT16_C( 17878),  INT16_C( 23201),  INT16_C( 25023), -INT16_C( 23553),  INT16_C( 16286), -INT16_C( 26104),  INT16_C( 29414),
         INT16_C( 22571), -INT16_C( 19935), -INT16_C(  8627), -INT16_C( 16945),  INT16_C( 18020), -INT16_C( 10254), -INT16_C( 20183),  INT16_C( 28675),
        -INT16_C(  9935), -INT16_C( 11594),  INT16_C( 30003),  INT16_C( 13107), -INT16_C( 12007),  INT16_C(  8562),  INT16_C( 22635), -INT16_C( 26989),
        -INT16_C( 19023), -INT16_C(   440),  INT16_C(  6035), -INT16_C(  2117), -INT16_C( 20899), -INT16_C( 31025), -INT16_C( 11681), -INT16_C( 28426) },
      { -INT16_C( 21333), -INT16_C(  8606), -INT16_C( 27358),  INT16_C( 15121), -INT16_C( 31642), -INT16_C( 11940), -INT16_C(  4132), -INT16_C( 29337),
        -INT16_C( 20572),  INT16_C( 14219),  INT16_C( 18374),  INT16_C(  9007), -INT16_C(   267),  INT16_C( 21673), -INT16_C( 24624),  INT16_C( 31716),
         INT16_C( 17996),  INT16_C( 28249),  INT16_C( 27611),  INT16_C( 16809),  INT16_C(  1519), -INT16_C( 13550),  INT16_C( 31220), -INT16_C( 26279),
        -INT16_C(  7128), -INT16_C(  4400), -INT16_C(   213),  INT16_C(  8209), -INT16_C( 17667), -INT16_C( 12940),  INT16_C( 22617), -INT16_C( 23224) },
      { -INT16_C( 21333),  INT16_C( 17878),  INT16_C( 23201),  INT16_C( 25023), -INT16_C( 23553),  INT16_C( 16286), -INT16_C(  4132),  INT16_C( 29414),
         INT16_C( 22571),  INT16_C( 14219),  INT16_C( 18374),  INT16_C(  9007),  INT16_C( 18020),  INT16_C( 21673), -INT16_C( 20183),  INT16_C( 31716),
         INT16_C( 17996),  INT16_C( 28249),  INT16_C( 30003),  INT16_C( 16809),  INT16_C(  1519),  INT16_C(  8562),  INT16_C( 31220), -INT16_C( 26279),
        -INT16_C(  7128), -INT16_C(   440),  INT16_C(  6035),  INT16_C(  8209), -INT16_C( 17667), -INT16_C( 12940),  INT16_C( 22617), -INT16_C( 23224) } },
    { { -INT16_C( 23906),  INT16_C( 30995), -INT16_C( 17395), -INT16_C(   838), -INT16_C( 13119), -INT16_C( 18745),  INT16_C(  8261),  INT16_C( 27983),
         INT16_C(  7941),  INT16_C( 12379),  INT16_C( 27679),  INT16_C(  7249), -INT16_C( 15066), -INT16_C( 32534),  INT16_C( 12830), -INT16_C( 17371),
         INT16_C( 14804), -INT16_C(  7882), -INT16_C(  3851), -INT16_C( 18467), -INT16_C( 23107),  INT16_C(   621), -INT16_C( 17211), -INT16_C( 13712),
        -INT16_C( 13349), -INT16_C(  1285),  INT16_C( 19512),  INT16_C( 24087),  INT16_C(   273),  INT16_C( 12254),  INT16_C(  1075),  INT16_C(  2284) },
      {  INT16_C(  8765),  INT16_C( 13033), -INT16_C( 14574), -INT16_C( 12311),  INT16_C( 22124),  INT16_C( 12754),  INT16_C( 16914), -INT16_C(  4356),
        -INT16_C(  2291),  INT16_C( 17896), -INT16_C(   189),  INT16_C( 21668), -INT16_C( 32256),  INT16_C( 13444),  INT16_C( 28806), -INT16_C( 15556),
         INT16_C(  9618), -INT16_C( 23306), -INT16_C(  8212),  INT16_C( 22644),  INT16_C( 17974),  INT16_C( 18570), -INT16_C( 31096), -INT16_C( 27338),
         INT16_C(  8061), -INT16_C( 16165),  INT16_C( 32542),  INT16_C(  7956), -INT16_C( 26623), -INT16_C( 30637), -INT16_C( 28920), -INT16_C( 26037) },
      {  INT16_C(  8765),  INT16_C( 30995), -INT16_C( 14574), -INT16_C(   838),  INT16_C( 22124),  INT16_C( 12754),  INT16_C( 16914),  INT16_C( 27983),
         INT16_C(  7941),  INT16_C( 17896),  INT16_C( 27679),  INT16_C( 21668), -INT16_C( 15066),  INT16_C( 13444),  INT16_C( 28806), -INT16_C( 15556),
         INT16_C( 14804), -INT16_C(  7882), -INT16_C(  3851),  INT16_C( 22644),  INT16_C( 17974),  INT16_C( 18570), -INT16_C( 17211), -INT16_C( 13712),
         INT16_C(  8061), -INT16_C(  1285),  INT16_C( 32542),  INT16_C( 24087),  INT16_C(   273),  INT16_C( 12254),  INT16_C(  1075),  INT16_C(  2284) } },
    { {  INT16_C( 16820), -INT16_C( 24257), -INT16_C( 19679),  INT16_C( 22521), -INT16_C( 31751), -INT16_C( 32353), -INT16_C( 10743), -INT16_C( 31210),
        -INT16_C(  3595),  INT16_C(  4934),  INT16_C( 23408),  INT16_C( 29234), -INT16_C( 31245), -INT16_C(   774),  INT16_C( 17684), -INT16_C( 13930),
        -INT16_C( 10873), -INT16_C( 22422),  INT16_C( 25480), -INT16_C( 32257), -INT16_C( 24857), -INT16_C(  4094),  INT16_C(  6516),  INT16_C( 26999),
        -INT16_C( 17142),  INT16_C( 31613), -INT16_C( 20712),  INT16_C(  3309), -INT16_C(  6347),  INT16_C( 18696), -INT16_C( 25044), -INT16_C( 19694) },
      {  INT16_C( 31860), -INT16_C(   933),  INT16_C( 23264), -INT16_C( 14466), -INT16_C( 32519),  INT16_C( 28087),  INT16_C( 11929), -INT16_C( 23337),
         INT16_C( 21740),  INT16_C(  1055),  INT16_C(  3075),  INT16_C( 14352),  INT16_C(  6387),  INT16_C(  8066), -INT16_C( 27465),  INT16_C( 11219),
         INT16_C( 11793), -INT16_C(  3801), -INT16_C( 23159), -INT16_C( 32072),  INT16_C( 28454), -INT16_C( 16401), -INT16_C( 14690), -INT16_C( 30109),
        -INT16_C( 32230),  INT16_C(  7822), -INT16_C( 24690), -INT16_C( 32426), -INT16_C( 10057),  INT16_C( 28321),  INT16_C( 29805),  INT16_C( 32409) },
      {  INT16_C( 31860), -INT16_C(   933),  INT16_C( 23264),  INT16_C( 22521), -INT16_C( 31751),  INT16_C( 28087),  INT16_C( 11929), -INT16_C( 23337),
         INT16_C( 21740),  INT16_C(  4934),  INT16_C( 23408),  INT16_C( 29234),  INT16_C(  6387),  INT16_C(  8066),  INT16_C( 17684),  INT16_C( 11219),
         INT16_C( 11793), -INT16_C(  3801),  INT16_C( 25480), -INT16_C( 32072),  INT16_C( 28454), -INT16_C(  4094),  INT16_C(  6516),  INT16_C( 26999),
        -INT16_C( 17142),  INT16_C( 31613), -INT16_C( 20712),  INT16_C(  3309), -INT16_C(  6347),  INT16_C( 28321),  INT16_C( 29805),  INT16_C( 32409) } },
    { { -INT16_C( 15966),  INT16_C( 11119),  INT16_C( 10086), -INT16_C( 29523), -INT16_C( 25194),  INT16_C( 13388), -INT16_C( 20637),  INT16_C( 32446),
         INT16_C( 19762), -INT16_C( 16228), -INT16_C(  3348), -INT16_C( 23742), -INT16_C(  7221),  INT16_C( 14354), -INT16_C( 21673), -INT16_C(  1610),
         INT16_C(  9580), -INT16_C( 11483), -INT16_C( 11700), -INT16_C(  7585), -INT16_C( 21649), -INT16_C( 11497), -INT16_C( 10917), -INT16_C( 29359),
        -INT16_C(  4830),  INT16_C(  3661), -INT16_C( 28705), -INT16_C( 21838), -INT16_C( 15246), -INT16_C( 13854), -INT16_C( 26513), -INT16_C(  9021) },
      { -INT16_C(  5955),  INT16_C(  2479),  INT16_C(  3770),  INT16_C( 10988),  INT16_C(   954),  INT16_C(  5629),  INT16_C( 20184), -INT16_C(  1118),
        -INT16_C(  4293),  INT16_C(  6665), -INT16_C( 17537), -INT16_C(  3643), -INT16_C( 22657), -INT16_C(  4165),  INT16_C( 32320), -INT16_C(   565),
         INT16_C( 31334),  INT16_C(  8199), -INT16_C(  3192),  INT16_C( 16970),  INT16_C( 18422), -INT16_C( 12713), -INT16_C(  1643), -INT16_C( 12087),
        -INT16_C( 11287),  INT16_C( 26859), -INT16_C( 20338),  INT16_C(  3673),  INT16_C(  5207), -INT16_C( 26627), -INT16_C( 14190), -INT16_C(  1899) },
      { -INT16_C(  5955),  INT16_C( 11119),  INT16_C( 10086),  INT16_C( 10988),  INT16_C(   954),  INT16_C( 13388),  INT16_C( 20184),  INT16_C( 32446),
         INT16_C( 19762),  INT16_C(  6665), -INT16_C(  3348), -INT16_C(  3643), -INT16_C(  7221),  INT16_C( 14354),  INT16_C( 32320), -INT16_C(   565),
         INT16_C( 31334),  INT16_C(  8199), -INT16_C(  3192),  INT16_C( 16970),  INT16_C( 18422), -INT16_C( 11497), -INT16_C(  1643), -INT16_C( 12087),
        -INT16_C(  4830),  INT16_C( 26859), -INT16_C( 20338),  INT16_C(  3673),  INT16_C(  5207), -INT16_C( 13854), -INT16_C( 14190), -INT16_C(  1899) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_epi16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 31856),  INT16_C( 27575),  INT16_C( 18767),  INT16_C( 28047),  INT16_C( 25446), -INT16_C( 31139),  INT16_C(  7112),  INT16_C( 14683),
         INT16_C(  7442), -INT16_C(  5599),  INT16_C( 11607),  INT16_C( 12590),  INT16_C(  4923),  INT16_C( 25627),  INT16_C(  2220),  INT16_C( 15731),
         INT16_C( 11147), -INT16_C(  9560),  INT16_C( 14196), -INT16_C(  9657), -INT16_C( 23142),  INT16_C( 25441), -INT16_C( 17216), -INT16_C( 11620),
        -INT16_C( 16935),  INT16_C( 12733), -INT16_C(  5142),  INT16_C(  9826),  INT16_C( 32254), -INT16_C( 21622), -INT16_C(   379),  INT16_C(  4328) },
      UINT32_C(2649395241),
      { -INT16_C( 10026), -INT16_C( 26684),  INT16_C( 24980),  INT16_C( 28265),  INT16_C(  9758),  INT16_C(  2463),  INT16_C(   273),  INT16_C(  4143),
        -INT16_C( 18050),  INT16_C(   955), -INT16_C( 23625), -INT16_C(  8173), -INT16_C(   717), -INT16_C(  1411), -INT16_C(  2770),  INT16_C(  1371),
         INT16_C(  8397),  INT16_C( 25244),  INT16_C(  1409), -INT16_C( 24624),  INT16_C( 28460),  INT16_C( 15784), -INT16_C( 10384), -INT16_C(  4531),
         INT16_C(  2193),  INT16_C( 18673),  INT16_C(  1195), -INT16_C(  8663), -INT16_C( 23039),  INT16_C( 12248),  INT16_C( 13467),  INT16_C( 26932) },
      { -INT16_C( 12204), -INT16_C( 10805), -INT16_C( 25642),  INT16_C(   628),  INT16_C(  7434),  INT16_C( 31295), -INT16_C( 29196), -INT16_C( 31384),
         INT16_C( 22933),  INT16_C( 16846), -INT16_C(  2211),  INT16_C( 24095), -INT16_C(  1891),  INT16_C( 14733), -INT16_C( 15828), -INT16_C( 32606),
         INT16_C( 28050),  INT16_C( 26709), -INT16_C( 14072),  INT16_C(  4714), -INT16_C( 21786), -INT16_C(  9332), -INT16_C(  3017), -INT16_C( 13216),
         INT16_C( 11853), -INT16_C( 22003),  INT16_C( 11557), -INT16_C( 15608), -INT16_C( 27355),  INT16_C( 20988), -INT16_C( 25001), -INT16_C(  5423) },
      { -INT16_C( 10026),  INT16_C( 27575),  INT16_C( 18767),  INT16_C( 28265),  INT16_C( 25446),  INT16_C( 31295),  INT16_C(  7112),  INT16_C( 14683),
         INT16_C(  7442), -INT16_C(  5599),  INT16_C( 11607),  INT16_C( 12590), -INT16_C(   717),  INT16_C( 25627),  INT16_C(  2220),  INT16_C(  1371),
         INT16_C( 11147),  INT16_C( 26709),  INT16_C( 14196),  INT16_C(  4714), -INT16_C( 23142),  INT16_C( 15784), -INT16_C(  3017), -INT16_C(  4531),
         INT16_C( 11853),  INT16_C( 12733),  INT16_C( 11557), -INT16_C(  8663), -INT16_C( 23039), -INT16_C( 21622), -INT16_C(   379),  INT16_C( 26932) } },
    { {  INT16_C(  9739),  INT16_C(  4946), -INT16_C( 16913), -INT16_C( 10715), -INT16_C( 20121), -INT16_C( 24911),  INT16_C(  4517), -INT16_C(  3478),
         INT16_C( 30784),  INT16_C( 26012), -INT16_C( 23387), -INT16_C( 13784),  INT16_C(  9273), -INT16_C( 28389), -INT16_C(  4926), -INT16_C( 12933),
        -INT16_C( 13038),  INT16_C(   480),  INT16_C(  1418), -INT16_C(  3625), -INT16_C( 30538),  INT16_C( 23439), -INT16_C(  1382), -INT16_C(  9651),
        -INT16_C(  5774),  INT16_C(  5951),  INT16_C( 26765), -INT16_C( 14367), -INT16_C(   884),  INT16_C( 20312), -INT16_C( 11288), -INT16_C(  1508) },
      UINT32_C( 737934752),
      { -INT16_C( 21413), -INT16_C(  2796),  INT16_C( 25254),  INT16_C(  6351),  INT16_C(  3915), -INT16_C(  9937),  INT16_C(  4215),  INT16_C(   928),
        -INT16_C(  2036), -INT16_C(  2990),  INT16_C( 28619),  INT16_C( 27630), -INT16_C(  5780),  INT16_C( 28310), -INT16_C( 19524),  INT16_C(  6183),
         INT16_C( 15455),  INT16_C(  1293), -INT16_C(  8802), -INT16_C(  5859),  INT16_C( 19692),  INT16_C( 25538),  INT16_C( 25180),  INT16_C( 26726),
        -INT16_C( 18086),  INT16_C(  9564),  INT16_C( 18984), -INT16_C( 27503),  INT16_C( 10035), -INT16_C(  4094),  INT16_C( 10970),  INT16_C( 14600) },
      {  INT16_C(  5478),  INT16_C(  1086),  INT16_C( 23538), -INT16_C(  8467), -INT16_C( 20313),  INT16_C(   833), -INT16_C( 22510),  INT16_C( 28011),
        -INT16_C( 14495), -INT16_C( 30318),  INT16_C(  8977),  INT16_C( 17693),  INT16_C(  8011),  INT16_C(  9525),  INT16_C( 15689), -INT16_C( 20641),
        -INT16_C( 25262),  INT16_C( 17843), -INT16_C( 24071), -INT16_C( 24541),  INT16_C( 25937),  INT16_C( 25508),  INT16_C(  3853),  INT16_C( 28368),
         INT16_C( 25559), -INT16_C(  5897),  INT16_C(  5254), -INT16_C( 11987),  INT16_C( 25139),  INT16_C( 32247),  INT16_C( 22175), -INT16_C(  3540) },
      {  INT16_C(  9739),  INT16_C(  4946), -INT16_C( 16913), -INT16_C( 10715), -INT16_C( 20121),  INT16_C(   833),  INT16_C(  4517),  INT16_C( 28011),
        -INT16_C(  2036),  INT16_C( 26012),  INT16_C( 28619),  INT16_C( 27630),  INT16_C(  8011),  INT16_C( 28310),  INT16_C( 15689),  INT16_C(  6183),
         INT16_C( 15455),  INT16_C( 17843),  INT16_C(  1418), -INT16_C(  5859),  INT16_C( 25937),  INT16_C( 25538),  INT16_C( 25180),  INT16_C( 28368),
         INT16_C( 25559),  INT16_C(  9564),  INT16_C( 26765), -INT16_C( 11987), -INT16_C(   884),  INT16_C( 32247), -INT16_C( 11288), -INT16_C(  1508) } },
    { { -INT16_C(  7949), -INT16_C(  5065),  INT16_C( 23169), -INT16_C( 11635),  INT16_C( 12735), -INT16_C( 13259),  INT16_C(  1600),  INT16_C(  5946),
         INT16_C( 12649), -INT16_C(  4352),  INT16_C( 11589),  INT16_C( 31169), -INT16_C( 18288),  INT16_C( 12278),  INT16_C(  8718),  INT16_C(   289),
         INT16_C( 22530), -INT16_C( 31762),  INT16_C( 31667),  INT16_C( 29269), -INT16_C( 29780), -INT16_C(  5057),  INT16_C( 31121), -INT16_C(  1532),
         INT16_C(  1195), -INT16_C(  3863), -INT16_C( 21967), -INT16_C( 16023),  INT16_C( 24418),  INT16_C( 28913),  INT16_C(  4738), -INT16_C( 31630) },
      UINT32_C( 503865451),
      { -INT16_C( 12312),  INT16_C( 31091),  INT16_C( 30537), -INT16_C(  2957),  INT16_C( 23931), -INT16_C( 21020),  INT16_C( 19975),  INT16_C( 27246),
         INT16_C( 24493),  INT16_C( 12250),  INT16_C( 19570), -INT16_C(  8780), -INT16_C( 17236), -INT16_C( 30725), -INT16_C( 29927),  INT16_C(   526),
        -INT16_C( 32165), -INT16_C( 23429), -INT16_C(  4103),  INT16_C( 30104),  INT16_C( 31820),  INT16_C( 21282), -INT16_C( 28470),  INT16_C( 30909),
        -INT16_C( 26384),  INT16_C( 25255),  INT16_C( 23524), -INT16_C( 28353),  INT16_C( 14871),  INT16_C( 12568),  INT16_C( 10181),  INT16_C(  8243) },
      { -INT16_C( 20823), -INT16_C( 23868),  INT16_C( 23709), -INT16_C(  5865),  INT16_C( 14809), -INT16_C( 23747), -INT16_C(  1334), -INT16_C( 17893),
        -INT16_C( 15470),  INT16_C( 30492),  INT16_C( 23326),  INT16_C( 13832),  INT16_C(  8341),  INT16_C( 23143), -INT16_C( 26041), -INT16_C(  3973),
         INT16_C( 16200), -INT16_C(  6509), -INT16_C( 21860),  INT16_C( 30159),  INT16_C(  3300), -INT16_C( 20968),  INT16_C( 13319), -INT16_C( 26264),
        -INT16_C( 31497),  INT16_C(  5392),  INT16_C(  6367),  INT16_C( 29771), -INT16_C( 19911), -INT16_C( 32562),  INT16_C( 18764), -INT16_C( 27279) },
      { -INT16_C( 12312),  INT16_C( 31091),  INT16_C( 23169), -INT16_C(  2957),  INT16_C( 12735), -INT16_C( 21020),  INT16_C( 19975),  INT16_C(  5946),
         INT16_C( 12649), -INT16_C(  4352),  INT16_C( 11589),  INT16_C( 31169), -INT16_C( 18288),  INT16_C( 23143), -INT16_C( 26041),  INT16_C(   289),
         INT16_C( 22530), -INT16_C( 31762),  INT16_C( 31667),  INT16_C( 30159), -INT16_C( 29780), -INT16_C(  5057),  INT16_C( 31121), -INT16_C(  1532),
         INT16_C(  1195),  INT16_C( 25255),  INT16_C( 23524),  INT16_C( 29771),  INT16_C( 14871),  INT16_C( 28913),  INT16_C(  4738), -INT16_C( 31630) } },
    { {  INT16_C(  1161),  INT16_C(  9595),  INT16_C( 19118), -INT16_C( 28006), -INT16_C( 19881),  INT16_C( 24128), -INT16_C( 22298), -INT16_C(  8713),
         INT16_C(  2092),  INT16_C(  3059),  INT16_C( 15904),  INT16_C( 22911),  INT16_C( 20209),  INT16_C( 15834),  INT16_C( 19351),  INT16_C(  8402),
         INT16_C( 19791), -INT16_C(   699), -INT16_C(  8296), -INT16_C(  4208), -INT16_C( 12142),  INT16_C( 30797),  INT16_C( 17529), -INT16_C( 23210),
         INT16_C( 18764),  INT16_C( 28081),  INT16_C( 12423),  INT16_C( 30918), -INT16_C( 24450),  INT16_C(  5814), -INT16_C( 30485),  INT16_C( 14902) },
      UINT32_C(1849195734),
      { -INT16_C( 21864),  INT16_C(  4454), -INT16_C( 17170),  INT16_C( 15287),  INT16_C( 26629), -INT16_C( 29528),  INT16_C( 28312),  INT16_C(  5893),
        -INT16_C( 17649), -INT16_C(  1491),  INT16_C( 25411),  INT16_C(  6453),  INT16_C( 28127),  INT16_C( 15239), -INT16_C(  7115), -INT16_C( 13016),
        -INT16_C( 29042),  INT16_C( 32223), -INT16_C( 27062),  INT16_C( 20408),  INT16_C( 24830), -INT16_C( 26916), -INT16_C(  7730), -INT16_C(  8787),
        -INT16_C(  9572), -INT16_C(  8232),  INT16_C(  3390),  INT16_C(  7673), -INT16_C( 32646), -INT16_C( 20648), -INT16_C( 32411), -INT16_C(  3204) },
      {  INT16_C( 23311),  INT16_C( 23152),  INT16_C( 10481), -INT16_C(  4183), -INT16_C( 31352),  INT16_C( 22406),  INT16_C( 13158),  INT16_C(   564),
         INT16_C(  3086),  INT16_C( 19682), -INT16_C(  9447), -INT16_C( 27799), -INT16_C( 15781), -INT16_C( 16318), -INT16_C( 16573),  INT16_C( 21172),
         INT16_C(  9242),  INT16_C(  3244),  INT16_C( 22093), -INT16_C( 10757), -INT16_C( 32293),  INT16_C( 16940),  INT16_C( 25013), -INT16_C( 15548),
         INT16_C(  9837), -INT16_C( 30961),  INT16_C( 30721),  INT16_C( 23834),  INT16_C( 23866),  INT16_C( 32029), -INT16_C( 12004),  INT16_C( 14032) },
      {  INT16_C(  1161),  INT16_C( 23152),  INT16_C( 10481), -INT16_C( 28006),  INT16_C( 26629),  INT16_C( 24128),  INT16_C( 28312),  INT16_C(  5893),
         INT16_C(  2092),  INT16_C(  3059),  INT16_C( 25411),  INT16_C(  6453),  INT16_C( 28127),  INT16_C( 15239), -INT16_C(  7115),  INT16_C(  8402),
         INT16_C( 19791), -INT16_C(   699), -INT16_C(  8296),  INT16_C( 20408),  INT16_C( 24830),  INT16_C( 16940),  INT16_C( 17529), -INT16_C( 23210),
         INT16_C( 18764), -INT16_C(  8232),  INT16_C( 30721),  INT16_C( 23834), -INT16_C( 24450),  INT16_C( 32029), -INT16_C( 12004),  INT16_C( 14902) } },
    { {  INT16_C( 31990),  INT16_C( 17218),  INT16_C( 16082), -INT16_C( 20968),  INT16_C( 17855),  INT16_C( 29936),  INT16_C( 13478),  INT16_C(  4919),
         INT16_C( 18011),  INT16_C( 23706), -INT16_C( 19009), -INT16_C(  1607), -INT16_C( 10478),  INT16_C( 11895),  INT16_C( 18344), -INT16_C( 24988),
        -INT16_C( 22589), -INT16_C( 26911), -INT16_C(  1307), -INT16_C( 23484),  INT16_C( 13375), -INT16_C(  6887),  INT16_C( 20584), -INT16_C( 15368),
        -INT16_C( 27753),  INT16_C( 22048), -INT16_C(  9912),  INT16_C( 23119), -INT16_C( 14672),  INT16_C( 22920), -INT16_C(  5107), -INT16_C( 11785) },
      UINT32_C(2020071827),
      {  INT16_C( 14047),  INT16_C( 18423), -INT16_C(  4218),  INT16_C(  7435),  INT16_C( 11138), -INT16_C( 13709), -INT16_C( 15612), -INT16_C( 19164),
        -INT16_C( 21367), -INT16_C( 26866),  INT16_C(  1433),  INT16_C( 11368), -INT16_C( 12322), -INT16_C( 20059), -INT16_C( 15750),  INT16_C( 22979),
        -INT16_C( 17672),  INT16_C( 32416), -INT16_C( 21590),  INT16_C( 11420),  INT16_C(  4054), -INT16_C(  9225),  INT16_C(  7122),  INT16_C( 23696),
        -INT16_C( 24888),  INT16_C( 25075),  INT16_C( 23459), -INT16_C( 32115),  INT16_C( 12842), -INT16_C( 23501), -INT16_C(  2060), -INT16_C(  4867) },
      { -INT16_C( 25167),  INT16_C( 23403),  INT16_C(  1865),  INT16_C(  8072),  INT16_C( 32534), -INT16_C(  5638), -INT16_C( 30054),  INT16_C( 25157),
         INT16_C( 14376), -INT16_C( 13117),  INT16_C( 20883), -INT16_C( 17074), -INT16_C( 32381),  INT16_C( 30817),  INT16_C( 24184),  INT16_C( 10852),
        -INT16_C( 12293),  INT16_C( 17541),  INT16_C(  3542), -INT16_C(  4764),  INT16_C( 24204),  INT16_C( 10198),  INT16_C(  7145),  INT16_C(  4489),
         INT16_C( 19795), -INT16_C(  6435),  INT16_C( 11166),  INT16_C(  8611),  INT16_C(  1197),  INT16_C(  9625), -INT16_C(   414),  INT16_C( 23887) },
      {  INT16_C( 14047),  INT16_C( 23403),  INT16_C( 16082), -INT16_C( 20968),  INT16_C( 32534),  INT16_C( 29936),  INT16_C( 13478),  INT16_C( 25157),
         INT16_C( 14376),  INT16_C( 23706), -INT16_C( 19009),  INT16_C( 11368), -INT16_C( 12322),  INT16_C( 11895),  INT16_C( 24184),  INT16_C( 22979),
        -INT16_C( 12293),  INT16_C( 32416),  INT16_C(  3542), -INT16_C( 23484),  INT16_C( 13375),  INT16_C( 10198),  INT16_C(  7145), -INT16_C( 15368),
        -INT16_C( 27753),  INT16_C( 22048), -INT16_C(  9912),  INT16_C(  8611),  INT16_C( 12842),  INT16_C(  9625), -INT16_C(   414), -INT16_C( 11785) } },
    { { -INT16_C( 10803), -INT16_C( 23390),  INT16_C(  1762),  INT16_C( 28561),  INT16_C( 26468),  INT16_C( 19862),  INT16_C(  8066), -INT16_C( 10913),
         INT16_C( 15468),  INT16_C(  2747),  INT16_C( 24168),  INT16_C(  5420), -INT16_C( 15006), -INT16_C( 15302), -INT16_C( 30013), -INT16_C( 28383),
        -INT16_C( 15521),  INT16_C( 16693), -INT16_C( 14647),  INT16_C( 11952),  INT16_C( 17965), -INT16_C( 20613), -INT16_C(  9626), -INT16_C( 11644),
         INT16_C( 16151),  INT16_C( 32733),  INT16_C(  2461), -INT16_C(   108), -INT16_C( 12594), -INT16_C( 27965), -INT16_C(  7080), -INT16_C( 18653) },
      UINT32_C(1912166568),
      {  INT16_C(  7152),  INT16_C( 22266),  INT16_C( 32501),  INT16_C(  3112),  INT16_C(  1469),  INT16_C( 23179),  INT16_C(  7950), -INT16_C(  8871),
         INT16_C(  7406),  INT16_C( 18031), -INT16_C( 28160), -INT16_C( 22274), -INT16_C(  2070),  INT16_C(  2074), -INT16_C( 18016), -INT16_C( 28589),
         INT16_C( 19924), -INT16_C( 13594),  INT16_C(  4043), -INT16_C( 30506),  INT16_C( 25108),  INT16_C(  9186),  INT16_C( 15233),  INT16_C( 28416),
         INT16_C( 28503),  INT16_C( 22454), -INT16_C( 19455), -INT16_C(  5376),  INT16_C(  6827),  INT16_C( 19443),  INT16_C( 18131), -INT16_C( 22308) },
      { -INT16_C( 15725),  INT16_C( 24178),  INT16_C( 18641), -INT16_C(  6426), -INT16_C( 14166),  INT16_C( 11273),  INT16_C(  2307),  INT16_C( 23195),
         INT16_C( 20856),  INT16_C( 31153), -INT16_C( 20219), -INT16_C( 20380),  INT16_C( 22475), -INT16_C( 24580), -INT16_C( 10083),  INT16_C( 12359),
        -INT16_C( 18022),  INT16_C( 27790),  INT16_C( 29697), -INT16_C( 21422),  INT16_C( 23356),  INT16_C( 16344),  INT16_C( 29540), -INT16_C(  9063),
         INT16_C( 19141), -INT16_C( 13739), -INT16_C( 17924), -INT16_C( 14469),  INT16_C( 30480), -INT16_C( 21146), -INT16_C( 21169), -INT16_C(  5667) },
      { -INT16_C( 10803), -INT16_C( 23390),  INT16_C(  1762),  INT16_C(  3112),  INT16_C( 26468),  INT16_C( 23179),  INT16_C(  8066),  INT16_C( 23195),
         INT16_C( 15468),  INT16_C(  2747),  INT16_C( 24168), -INT16_C( 20380),  INT16_C( 22475), -INT16_C( 15302), -INT16_C( 10083), -INT16_C( 28383),
         INT16_C( 19924),  INT16_C( 16693), -INT16_C( 14647), -INT16_C( 21422),  INT16_C( 25108),  INT16_C( 16344),  INT16_C( 29540),  INT16_C( 28416),
         INT16_C( 28503),  INT16_C( 32733),  INT16_C(  2461), -INT16_C(   108),  INT16_C( 30480),  INT16_C( 19443),  INT16_C( 18131), -INT16_C( 18653) } },
    { {  INT16_C( 27494),  INT16_C( 26709), -INT16_C( 22561),  INT16_C(  6932), -INT16_C(  5118),  INT16_C( 26202), -INT16_C(  3233),  INT16_C(  9282),
        -INT16_C( 26819),  INT16_C( 14831),  INT16_C( 27216),  INT16_C( 24577),  INT16_C( 26593),  INT16_C( 12301), -INT16_C(  5611),  INT16_C( 31513),
         INT16_C( 28501),  INT16_C( 13539), -INT16_C(  2282),  INT16_C(  6479), -INT16_C( 22045),  INT16_C( 17279), -INT16_C( 15716), -INT16_C(  9625),
         INT16_C( 22105), -INT16_C( 21997),  INT16_C(  5312), -INT16_C( 24310),  INT16_C(  6268), -INT16_C( 28207), -INT16_C(  5374),  INT16_C( 22540) },
      UINT32_C(1888284762),
      {  INT16_C(  2437),  INT16_C(  8718),  INT16_C( 30155),  INT16_C(  9468),  INT16_C(  4044), -INT16_C( 29490), -INT16_C(  9948), -INT16_C( 24530),
        -INT16_C(    15), -INT16_C(  3279),  INT16_C( 15850),  INT16_C( 17483), -INT16_C( 10195),  INT16_C(  5557),  INT16_C( 16052),  INT16_C( 14816),
        -INT16_C(  4537),  INT16_C(  4699),  INT16_C( 22371),  INT16_C( 12087),  INT16_C(  1383), -INT16_C( 29764), -INT16_C(  5410), -INT16_C( 12501),
         INT16_C( 23785), -INT16_C( 11069),  INT16_C(  3737), -INT16_C( 14568), -INT16_C( 12826), -INT16_C( 25892), -INT16_C( 17396),  INT16_C( 21460) },
      {  INT16_C( 12202),  INT16_C(  3430), -INT16_C( 25209), -INT16_C(  4547), -INT16_C(  1630), -INT16_C( 32391), -INT16_C( 23325), -INT16_C( 13232),
         INT16_C(  4864), -INT16_C( 26208), -INT16_C( 18142),  INT16_C(  2144),  INT16_C( 15494), -INT16_C( 27997),  INT16_C( 30712), -INT16_C( 23834),
         INT16_C( 19622),  INT16_C( 11696), -INT16_C(  4631), -INT16_C( 29925), -INT16_C( 27418), -INT16_C( 14068),  INT16_C( 23864),  INT16_C( 14485),
         INT16_C( 13936), -INT16_C( 27950),  INT16_C( 13039),  INT16_C( 30107),  INT16_C( 15983),  INT16_C( 26376), -INT16_C(  4427),  INT16_C( 23306) },
      {  INT16_C( 27494),  INT16_C(  8718), -INT16_C( 22561),  INT16_C(  9468),  INT16_C(  4044),  INT16_C( 26202), -INT16_C(  9948),  INT16_C(  9282),
        -INT16_C( 26819),  INT16_C( 14831),  INT16_C( 27216),  INT16_C( 24577),  INT16_C( 15494),  INT16_C(  5557),  INT16_C( 30712),  INT16_C( 14816),
         INT16_C( 28501),  INT16_C( 13539),  INT16_C( 22371),  INT16_C( 12087), -INT16_C( 22045),  INT16_C( 17279), -INT16_C( 15716),  INT16_C( 14485),
         INT16_C( 22105), -INT16_C( 21997),  INT16_C(  5312), -INT16_C( 24310),  INT16_C( 15983),  INT16_C( 26376), -INT16_C(  4427),  INT16_C( 22540) } },
    { { -INT16_C( 17862),  INT16_C(  9097), -INT16_C( 23385), -INT16_C( 29266), -INT16_C( 17607),  INT16_C( 29014), -INT16_C(  5352), -INT16_C( 30550),
         INT16_C( 31777),  INT16_C(  4123), -INT16_C( 18770),  INT16_C(  7558), -INT16_C( 28940), -INT16_C( 22139), -INT16_C( 28804), -INT16_C( 18940),
        -INT16_C( 29367), -INT16_C(  3879), -INT16_C( 30926),  INT16_C( 27517), -INT16_C( 11454),  INT16_C( 23260), -INT16_C( 31042), -INT16_C(  7965),
        -INT16_C(   510), -INT16_C( 19984),  INT16_C( 30388), -INT16_C( 22322),  INT16_C( 21252), -INT16_C( 32687),  INT16_C( 21986),  INT16_C( 11062) },
      UINT32_C( 354095075),
      {  INT16_C( 23659),  INT16_C( 10804),  INT16_C(  6115), -INT16_C(  6902), -INT16_C(  1515), -INT16_C( 13930),  INT16_C( 25969),  INT16_C( 30065),
        -INT16_C( 15688), -INT16_C( 25610),  INT16_C( 11287), -INT16_C(  1338), -INT16_C(  7620), -INT16_C( 11505), -INT16_C( 28806), -INT16_C(  6484),
        -INT16_C(  7956), -INT16_C( 12528),  INT16_C(  6903),  INT16_C(  3252),  INT16_C( 19220), -INT16_C( 31275),  INT16_C( 18096),  INT16_C( 26875),
        -INT16_C(  3832),  INT16_C(  8195), -INT16_C( 13795),  INT16_C( 22810),  INT16_C( 10924),  INT16_C(  9772), -INT16_C(  9799), -INT16_C( 23284) },
      {  INT16_C(  7353), -INT16_C( 20108),  INT16_C( 10550),  INT16_C( 19389), -INT16_C( 27788),  INT16_C(  9424), -INT16_C( 13351), -INT16_C(  7540),
        -INT16_C( 28484), -INT16_C(  9726),  INT16_C(  7258),  INT16_C(  1587),  INT16_C( 24646),  INT16_C(    44),  INT16_C( 14649), -INT16_C(  3419),
         INT16_C(  6741), -INT16_C( 29533),  INT16_C( 24899), -INT16_C( 18473), -INT16_C( 22540), -INT16_C( 12837),  INT16_C( 26483),  INT16_C( 12207),
        -INT16_C( 19977),  INT16_C( 20745),  INT16_C( 15822),  INT16_C(  5207), -INT16_C( 31587), -INT16_C( 10732), -INT16_C( 17731),  INT16_C(  4808) },
      {  INT16_C( 23659),  INT16_C( 10804), -INT16_C( 23385), -INT16_C( 29266), -INT16_C( 17607),  INT16_C(  9424),  INT16_C( 25969),  INT16_C( 30065),
        -INT16_C( 15688), -INT16_C(  9726),  INT16_C( 11287),  INT16_C(  1587), -INT16_C( 28940), -INT16_C( 22139), -INT16_C( 28804), -INT16_C( 18940),
         INT16_C(  6741), -INT16_C( 12528), -INT16_C( 30926),  INT16_C(  3252),  INT16_C( 19220),  INT16_C( 23260), -INT16_C( 31042), -INT16_C(  7965),
        -INT16_C(  3832), -INT16_C( 19984),  INT16_C( 15822), -INT16_C( 22322),  INT16_C( 10924), -INT16_C( 32687),  INT16_C( 21986),  INT16_C( 11062) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_epi16(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask32 k;
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { UINT32_C(3465174033),
      {  INT16_C(   193), -INT16_C( 29734), -INT16_C( 20936),  INT16_C(    94),  INT16_C( 26271),  INT16_C( 26954), -INT16_C( 16407),  INT16_C( 11234),
        -INT16_C( 22766), -INT16_C( 30451),  INT16_C(   983), -INT16_C(  6005),  INT16_C(  5471),  INT16_C( 22967), -INT16_C( 24720),  INT16_C( 12840),
         INT16_C(   672), -INT16_C( 10051),  INT16_C(  7088),  INT16_C( 20697),  INT16_C(  9089),  INT16_C( 27577), -INT16_C( 25374), -INT16_C(  2666),
        -INT16_C( 23741),  INT16_C(  6782),  INT16_C(  2470),  INT16_C(  1283), -INT16_C( 17889), -INT16_C( 28834), -INT16_C( 30887), -INT16_C(  1599) },
      {  INT16_C( 32393),  INT16_C( 15058), -INT16_C( 21607),  INT16_C(  7050),  INT16_C( 17358), -INT16_C( 20346),  INT16_C(  7391),  INT16_C(  9125),
         INT16_C(  9152),  INT16_C( 26173),  INT16_C( 16429),  INT16_C( 19564), -INT16_C( 13574),  INT16_C( 21723), -INT16_C( 25263), -INT16_C(  9395),
         INT16_C(  7963), -INT16_C( 19179), -INT16_C( 24630), -INT16_C( 26416),  INT16_C( 22242), -INT16_C( 15799), -INT16_C(  4494),  INT16_C( 13029),
         INT16_C(  8722),  INT16_C( 16281),  INT16_C(  1379),  INT16_C( 23947),  INT16_C( 26319),  INT16_C(  8625), -INT16_C(   253),  INT16_C(  8188) },
      {  INT16_C( 32393),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 26271),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 16429),  INT16_C( 19564),  INT16_C(  5471),  INT16_C(     0), -INT16_C( 24720),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 10051),  INT16_C(     0),  INT16_C( 20697),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 13029),
         INT16_C(     0),  INT16_C( 16281),  INT16_C(  2470),  INT16_C( 23947),  INT16_C(     0),  INT16_C(     0), -INT16_C(   253),  INT16_C(  8188) } },
    { UINT32_C(3922989342),
      { -INT16_C( 13574),  INT16_C( 27732),  INT16_C( 14777), -INT16_C( 13409),  INT16_C( 14428), -INT16_C( 16630), -INT16_C( 27331),  INT16_C(  3100),
        -INT16_C( 12549), -INT16_C(   211),  INT16_C( 10701), -INT16_C(  5346), -INT16_C(  3526), -INT16_C(  5420),  INT16_C( 22166), -INT16_C( 28547),
        -INT16_C( 12000), -INT16_C(  9732), -INT16_C( 25845),  INT16_C( 26532), -INT16_C( 20781),  INT16_C(  4134),  INT16_C( 16963),  INT16_C( 16157),
         INT16_C( 18960), -INT16_C(  8898),  INT16_C( 23668), -INT16_C( 20791), -INT16_C( 25266), -INT16_C(  7015),  INT16_C(  5875),  INT16_C(  5236) },
      {  INT16_C( 28903), -INT16_C(  3347), -INT16_C( 28148), -INT16_C(  8359),  INT16_C( 32576), -INT16_C( 31504),  INT16_C(  3522), -INT16_C( 11581),
         INT16_C(   343), -INT16_C( 13392),  INT16_C( 31069), -INT16_C( 21638),  INT16_C(  4886),  INT16_C(  2703),  INT16_C(   809),  INT16_C(  4126),
         INT16_C(  2931),  INT16_C( 32515),  INT16_C( 23709), -INT16_C(  8609),  INT16_C( 20444), -INT16_C( 24990),  INT16_C(  9564), -INT16_C( 19600),
         INT16_C(  8230), -INT16_C( 31873), -INT16_C(  1639), -INT16_C( 20434), -INT16_C( 17140),  INT16_C( 13754), -INT16_C( 10048),  INT16_C( 13125) },
      {  INT16_C(     0),  INT16_C( 27732),  INT16_C( 14777), -INT16_C(  8359),  INT16_C( 32576),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(   343),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4886),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 23709),  INT16_C(     0),  INT16_C( 20444),  INT16_C(     0),  INT16_C( 16963),  INT16_C( 16157),
         INT16_C( 18960),  INT16_C(     0),  INT16_C(     0), -INT16_C( 20434),  INT16_C(     0),  INT16_C( 13754),  INT16_C(  5875),  INT16_C( 13125) } },
    { UINT32_C(2176010467),
      { -INT16_C( 16031), -INT16_C( 17121), -INT16_C( 28698),  INT16_C(  3184), -INT16_C(  4176),  INT16_C( 18831), -INT16_C( 16920), -INT16_C(  2823),
        -INT16_C( 19590),  INT16_C( 14889),  INT16_C( 28555),  INT16_C( 28525),  INT16_C(  8375),  INT16_C( 23792),  INT16_C( 20274), -INT16_C( 27683),
        -INT16_C(  1008), -INT16_C(  2480), -INT16_C( 15988),  INT16_C( 15362), -INT16_C( 28240), -INT16_C( 26235),  INT16_C( 32590), -INT16_C( 14195),
        -INT16_C( 18638), -INT16_C( 16894),  INT16_C( 28454), -INT16_C(  8915),  INT16_C(  7568), -INT16_C( 15814),  INT16_C(  5996),  INT16_C( 31830) },
      { -INT16_C( 23020), -INT16_C( 24462),  INT16_C( 29799),  INT16_C(  6364),  INT16_C( 24837),  INT16_C( 21425),  INT16_C( 16096),  INT16_C(  4891),
         INT16_C(  7669),  INT16_C(  7121), -INT16_C(   372),  INT16_C(  7417),  INT16_C( 13083), -INT16_C( 30753),  INT16_C( 13642),  INT16_C( 24067),
         INT16_C( 30171),  INT16_C( 17406), -INT16_C(  9495), -INT16_C(  4517),  INT16_C(  3132),  INT16_C(  7233),  INT16_C( 23626),  INT16_C( 16431),
         INT16_C(   121),  INT16_C(  1371),  INT16_C( 21758),  INT16_C(  6434),  INT16_C(   391), -INT16_C( 11616), -INT16_C( 23754),  INT16_C(  4400) },
      { -INT16_C( 16031), -INT16_C( 17121),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 21425),  INT16_C( 16096),  INT16_C(  4891),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 28525),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20274),  INT16_C(     0),
         INT16_C( 30171),  INT16_C( 17406),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3132),  INT16_C(  7233),  INT16_C(     0),  INT16_C( 16431),
         INT16_C(   121),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31830) } },
    { UINT32_C(  22294296),
      {  INT16_C( 12475),  INT16_C(  1634), -INT16_C( 28276),  INT16_C(  1350), -INT16_C( 24174), -INT16_C( 28661),  INT16_C( 11766),  INT16_C( 32170),
         INT16_C( 18990),  INT16_C( 25679), -INT16_C( 32530),  INT16_C(  1653), -INT16_C( 13649), -INT16_C( 18424), -INT16_C(  2183),  INT16_C( 13822),
         INT16_C( 24616), -INT16_C( 19397), -INT16_C( 32271), -INT16_C( 31814), -INT16_C( 15070),  INT16_C(  6164), -INT16_C( 16654),  INT16_C(  8342),
        -INT16_C(  6904), -INT16_C(  2428), -INT16_C(  1691),  INT16_C(  5373),  INT16_C(  1475),  INT16_C( 15821), -INT16_C( 13316),  INT16_C(  9330) },
      { -INT16_C( 21205),  INT16_C(  7385), -INT16_C( 27858),  INT16_C( 20640), -INT16_C( 19368),  INT16_C( 19049), -INT16_C(   142),  INT16_C( 31338),
        -INT16_C(  4380),  INT16_C( 19057),  INT16_C( 28391), -INT16_C( 21666),  INT16_C( 11123),  INT16_C( 28648),  INT16_C( 23286),  INT16_C(  8596),
         INT16_C( 27911),  INT16_C( 13630), -INT16_C(  8704),  INT16_C( 22661), -INT16_C(  4462),  INT16_C(  1186),  INT16_C(  3309), -INT16_C( 11650),
        -INT16_C(  4102), -INT16_C(  7908),  INT16_C( 31325), -INT16_C( 12148),  INT16_C( 29862), -INT16_C( 25536), -INT16_C( 11058), -INT16_C( 10818) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20640), -INT16_C( 19368),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 18990),  INT16_C( 25679),  INT16_C( 28391),  INT16_C(  1653),  INT16_C(     0),  INT16_C( 28648),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C(  8704),  INT16_C(     0), -INT16_C(  4462),  INT16_C(     0),  INT16_C(  3309),  INT16_C(     0),
        -INT16_C(  4102),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(1091238977),
      {  INT16_C( 15230),  INT16_C( 27760), -INT16_C(  4537),  INT16_C( 16702),  INT16_C( 23262),  INT16_C( 15138), -INT16_C( 20524),  INT16_C( 31244),
         INT16_C( 19491), -INT16_C(  3561), -INT16_C( 10976),  INT16_C( 25031), -INT16_C( 11567), -INT16_C( 21598),  INT16_C( 15202), -INT16_C(  8169),
        -INT16_C( 30858), -INT16_C( 17076), -INT16_C( 30091),  INT16_C( 21502),  INT16_C(  8420), -INT16_C( 18033), -INT16_C( 25649), -INT16_C(  3277),
         INT16_C( 19175),  INT16_C(  2021), -INT16_C( 21473), -INT16_C(  3992),  INT16_C(  2686), -INT16_C(  8037), -INT16_C( 19899), -INT16_C( 17471) },
      {  INT16_C(  3385), -INT16_C( 20616),  INT16_C( 30360),  INT16_C( 31746), -INT16_C( 28266),  INT16_C( 26165),  INT16_C( 26924),  INT16_C(  4953),
         INT16_C( 16051), -INT16_C( 11494), -INT16_C( 32022),  INT16_C( 27075),  INT16_C( 24460), -INT16_C( 11959),  INT16_C(  2577),  INT16_C( 19340),
         INT16_C(  1048), -INT16_C( 20230), -INT16_C(   902),  INT16_C(  4396),  INT16_C( 25230), -INT16_C( 17801), -INT16_C( 12085),  INT16_C( 32462),
        -INT16_C(  6130), -INT16_C(  1967),  INT16_C(  5483), -INT16_C(  2207), -INT16_C( 21644), -INT16_C( 31287),  INT16_C( 21941), -INT16_C( 12848) },
      {  INT16_C( 15230),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 26924),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 10976),  INT16_C( 27075),  INT16_C( 24460), -INT16_C( 11959),  INT16_C( 15202),  INT16_C( 19340),
         INT16_C(     0), -INT16_C( 17076),  INT16_C(     0),  INT16_C( 21502),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 19175),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 21941),  INT16_C(     0) } },
    { UINT32_C(3565013594),
      {  INT16_C( 23564), -INT16_C( 10481), -INT16_C(  8916),  INT16_C( 14933), -INT16_C( 22586),  INT16_C( 12595), -INT16_C( 27460),  INT16_C( 12328),
        -INT16_C(  3777), -INT16_C(  2635), -INT16_C( 31161), -INT16_C( 24126),  INT16_C( 16464),  INT16_C(  6005),  INT16_C( 23530), -INT16_C(  2452),
         INT16_C( 31927), -INT16_C(  6963),  INT16_C(  8793),  INT16_C(  7966),  INT16_C( 20937), -INT16_C( 31408),  INT16_C( 31206),  INT16_C(  9653),
         INT16_C( 27498), -INT16_C( 20198), -INT16_C(  8719),  INT16_C( 16722), -INT16_C( 14307),  INT16_C(  1881), -INT16_C( 15069), -INT16_C(  9475) },
      { -INT16_C( 13759), -INT16_C( 25666), -INT16_C(  8724), -INT16_C( 18758),  INT16_C(  2862),  INT16_C(  5179), -INT16_C(  3708), -INT16_C(  4550),
         INT16_C( 21596),  INT16_C( 19872), -INT16_C(  3535),  INT16_C( 20110), -INT16_C(  6214), -INT16_C(  8875),  INT16_C( 21165), -INT16_C(  4424),
         INT16_C( 30236),  INT16_C(  2441),  INT16_C( 17491), -INT16_C( 32065), -INT16_C(  1457), -INT16_C( 11370), -INT16_C( 12053),  INT16_C( 18369),
         INT16_C( 24869),  INT16_C( 22164),  INT16_C(  9044),  INT16_C(  3749), -INT16_C(  1526), -INT16_C( 18452), -INT16_C( 23475),  INT16_C( 27046) },
      {  INT16_C(     0), -INT16_C( 10481),  INT16_C(     0),  INT16_C( 14933),  INT16_C(  2862),  INT16_C(     0), -INT16_C(  3708),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 19872),  INT16_C(     0),  INT16_C( 20110),  INT16_C(     0),  INT16_C(     0),  INT16_C( 23530), -INT16_C(  2452),
         INT16_C( 31927),  INT16_C(     0),  INT16_C( 17491),  INT16_C(  7966),  INT16_C( 20937), -INT16_C( 11370),  INT16_C( 31206),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(  9044),  INT16_C(     0), -INT16_C(  1526),  INT16_C(     0), -INT16_C( 15069),  INT16_C( 27046) } },
    { UINT32_C(1852976922),
      { -INT16_C( 31188),  INT16_C(  6037),  INT16_C( 22359),  INT16_C( 31839), -INT16_C(  3144),  INT16_C(  3282),  INT16_C( 30486),  INT16_C(  8475),
         INT16_C(  1906), -INT16_C( 16424),  INT16_C( 32427), -INT16_C( 15064), -INT16_C( 25682),  INT16_C(  8499),  INT16_C(  9164), -INT16_C(  1820),
         INT16_C( 31146),  INT16_C(   272),  INT16_C( 28624), -INT16_C( 30339),  INT16_C( 20322),  INT16_C( 31125), -INT16_C( 20281),  INT16_C( 14746),
         INT16_C( 29367),  INT16_C( 25336),  INT16_C(  8433), -INT16_C( 24792),  INT16_C( 23483), -INT16_C( 30528), -INT16_C( 23425),  INT16_C( 10624) },
      { -INT16_C( 28642), -INT16_C(  4566), -INT16_C( 22529),  INT16_C( 25207),  INT16_C(  3574), -INT16_C( 16933),  INT16_C( 30141),  INT16_C( 30198),
        -INT16_C(  4377), -INT16_C( 10025), -INT16_C(   241), -INT16_C( 13705),  INT16_C( 14427), -INT16_C(  9646), -INT16_C( 11300), -INT16_C(  1533),
         INT16_C( 11619),  INT16_C( 25577),  INT16_C( 24788), -INT16_C( 13627), -INT16_C( 24467),  INT16_C( 11144),  INT16_C( 32277), -INT16_C(   864),
         INT16_C( 30573),  INT16_C( 31957),  INT16_C( 19575), -INT16_C( 11706), -INT16_C( 26236),  INT16_C( 25004), -INT16_C( 20628), -INT16_C( 12453) },
      {  INT16_C(     0),  INT16_C(  6037),  INT16_C(     0),  INT16_C( 31839),  INT16_C(  3574),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(  1906), -INT16_C( 10025),  INT16_C( 32427), -INT16_C( 13705),  INT16_C(     0),  INT16_C(  8499),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 25577),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20322),  INT16_C( 31125),  INT16_C( 32277),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 31957),  INT16_C( 19575), -INT16_C( 11706),  INT16_C(     0),  INT16_C( 25004), -INT16_C( 20628),  INT16_C(     0) } },
    { UINT32_C(2956084444),
      {  INT16_C(   663), -INT16_C( 21443), -INT16_C(  8831), -INT16_C(  4439),  INT16_C( 32341), -INT16_C( 13206), -INT16_C( 20278),  INT16_C( 20382),
         INT16_C( 19017), -INT16_C( 19024),  INT16_C(  3065), -INT16_C( 10875), -INT16_C( 18608), -INT16_C(  2683), -INT16_C(    81),  INT16_C( 17927),
         INT16_C( 17666), -INT16_C( 31757), -INT16_C( 25566),  INT16_C( 30577), -INT16_C(  9446), -INT16_C(  7101), -INT16_C(  7797), -INT16_C( 10957),
        -INT16_C(  7381),  INT16_C(  9354),  INT16_C(  4079),  INT16_C( 16377),  INT16_C( 32455),  INT16_C( 30260),  INT16_C( 15230), -INT16_C( 32580) },
      { -INT16_C( 20608), -INT16_C( 23805),  INT16_C( 29771),  INT16_C( 25882),  INT16_C( 24143), -INT16_C(  9654),  INT16_C( 32063),  INT16_C( 27567),
         INT16_C( 14945),  INT16_C( 20623), -INT16_C( 30391),  INT16_C(  4239), -INT16_C( 15609), -INT16_C( 31354),  INT16_C( 17406),  INT16_C( 32517),
         INT16_C(  2290),  INT16_C( 15906),  INT16_C( 15484), -INT16_C( 13405), -INT16_C(  4710), -INT16_C(  9562),  INT16_C( 21867), -INT16_C( 13243),
        -INT16_C( 11121), -INT16_C(  9956), -INT16_C( 21667),  INT16_C( 26089),  INT16_C( 28782),  INT16_C( 27882), -INT16_C(  3917), -INT16_C( 23061) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 29771),  INT16_C( 25882),  INT16_C( 32341),  INT16_C(     0),  INT16_C( 32063),  INT16_C( 27567),
         INT16_C(     0),  INT16_C(     0),  INT16_C(  3065),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 17406),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 15906),  INT16_C(     0),  INT16_C(     0), -INT16_C(  4710), -INT16_C(  7101),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 32455),  INT16_C( 30260),  INT16_C(     0), -INT16_C( 23061) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_epi16(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_max_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint16_t a[32];
    const uint16_t b[32];
    const uint16_t r[32];
  } test_vec[] = {
    { { UINT16_C(39893), UINT16_C(12168), UINT16_C(15734), UINT16_C(45768), UINT16_C( 4464), UINT16_C(63629), UINT16_C(35362), UINT16_C(32306),
        UINT16_C(62286), UINT16_C(19220), UINT16_C(14705), UINT16_C(45343), UINT16_C(57947), UINT16_C(40021), UINT16_C(16202), UINT16_C( 8136),
        UINT16_C(20699), UINT16_C(20814), UINT16_C( 5774), UINT16_C(65027), UINT16_C(37160), UINT16_C(19190), UINT16_C(10523), UINT16_C(27080),
        UINT16_C(56604), UINT16_C(36276), UINT16_C(54038), UINT16_C(28990), UINT16_C(37813), UINT16_C(65293), UINT16_C(54739), UINT16_C(44574) },
      { UINT16_C(27686), UINT16_C(46335), UINT16_C(  898), UINT16_C(43698), UINT16_C(43412), UINT16_C(45045), UINT16_C(48594), UINT16_C(60952),
        UINT16_C(52378), UINT16_C(45435), UINT16_C(47775), UINT16_C(21538), UINT16_C(12365), UINT16_C( 8275), UINT16_C(28933), UINT16_C(11214),
        UINT16_C(52957), UINT16_C(24543), UINT16_C(37585), UINT16_C(25866), UINT16_C(65339), UINT16_C( 3348), UINT16_C(11452), UINT16_C(22523),
        UINT16_C(30456), UINT16_C(38664), UINT16_C(10800), UINT16_C(32491), UINT16_C(15962), UINT16_C(24734), UINT16_C(28079), UINT16_C(35979) },
      { UINT16_C(39893), UINT16_C(46335), UINT16_C(15734), UINT16_C(45768), UINT16_C(43412), UINT16_C(63629), UINT16_C(48594), UINT16_C(60952),
        UINT16_C(62286), UINT16_C(45435), UINT16_C(47775), UINT16_C(45343), UINT16_C(57947), UINT16_C(40021), UINT16_C(28933), UINT16_C(11214),
        UINT16_C(52957), UINT16_C(24543), UINT16_C(37585), UINT16_C(65027), UINT16_C(65339), UINT16_C(19190), UINT16_C(11452), UINT16_C(27080),
        UINT16_C(56604), UINT16_C(38664), UINT16_C(54038), UINT16_C(32491), UINT16_C(37813), UINT16_C(65293), UINT16_C(54739), UINT16_C(44574) } },
    { { UINT16_C(27451), UINT16_C( 3307), UINT16_C(62973), UINT16_C(14449), UINT16_C(34292), UINT16_C(45381), UINT16_C(16561), UINT16_C(43272),
        UINT16_C( 4278), UINT16_C(59200), UINT16_C(11066), UINT16_C(38245), UINT16_C(  873), UINT16_C( 6389), UINT16_C(32880), UINT16_C(43940),
        UINT16_C(36843), UINT16_C(59575), UINT16_C(10373), UINT16_C(31008), UINT16_C(26029), UINT16_C(24106), UINT16_C(12965), UINT16_C(23559),
        UINT16_C(18242), UINT16_C(32067), UINT16_C(43122), UINT16_C(56082), UINT16_C( 1963), UINT16_C( 7411), UINT16_C(38791), UINT16_C(29639) },
      { UINT16_C(32551), UINT16_C(44123), UINT16_C(31911), UINT16_C(21797), UINT16_C(20705), UINT16_C(34739), UINT16_C(48002), UINT16_C(50659),
        UINT16_C( 9730), UINT16_C(30018), UINT16_C(21710), UINT16_C(31056), UINT16_C(17499), UINT16_C(58005), UINT16_C(24027), UINT16_C(  597),
        UINT16_C(45532), UINT16_C(33710), UINT16_C(54317), UINT16_C( 3800), UINT16_C(35876), UINT16_C(42645), UINT16_C(30791), UINT16_C(18795),
        UINT16_C(44446), UINT16_C(27838), UINT16_C( 3841), UINT16_C(23782), UINT16_C(31571), UINT16_C(11839), UINT16_C(38104), UINT16_C(46129) },
      { UINT16_C(32551), UINT16_C(44123), UINT16_C(62973), UINT16_C(21797), UINT16_C(34292), UINT16_C(45381), UINT16_C(48002), UINT16_C(50659),
        UINT16_C( 9730), UINT16_C(59200), UINT16_C(21710), UINT16_C(38245), UINT16_C(17499), UINT16_C(58005), UINT16_C(32880), UINT16_C(43940),
        UINT16_C(45532), UINT16_C(59575), UINT16_C(54317), UINT16_C(31008), UINT16_C(35876), UINT16_C(42645), UINT16_C(30791), UINT16_C(23559),
        UINT16_C(44446), UINT16_C(32067), UINT16_C(43122), UINT16_C(56082), UINT16_C(31571), UINT16_C(11839), UINT16_C(38791), UINT16_C(46129) } },
    { { UINT16_C(57157), UINT16_C(29240), UINT16_C( 4275), UINT16_C(55169), UINT16_C( 5788), UINT16_C(58238), UINT16_C(59791), UINT16_C(11565),
        UINT16_C(60311), UINT16_C(39066), UINT16_C(33018), UINT16_C(19957), UINT16_C(13563), UINT16_C(54396), UINT16_C(44488), UINT16_C( 3720),
        UINT16_C(49292), UINT16_C(16512), UINT16_C(  465), UINT16_C(27927), UINT16_C(38168), UINT16_C(42833), UINT16_C(32383), UINT16_C( 5844),
        UINT16_C(28265), UINT16_C(25774), UINT16_C(41966), UINT16_C(60081), UINT16_C(11735), UINT16_C(41150), UINT16_C(18138), UINT16_C(26542) },
      { UINT16_C(11783), UINT16_C(55463), UINT16_C(48688), UINT16_C(18501), UINT16_C(38484), UINT16_C(54255), UINT16_C(49940), UINT16_C(32489),
        UINT16_C(38706), UINT16_C( 8418), UINT16_C(37691), UINT16_C( 4618), UINT16_C(51393), UINT16_C(39858), UINT16_C(24591), UINT16_C( 5634),
        UINT16_C(43407), UINT16_C(49134), UINT16_C(13160), UINT16_C(48135), UINT16_C(63178), UINT16_C(56975), UINT16_C(30905), UINT16_C(60252),
        UINT16_C(15887), UINT16_C(18956), UINT16_C( 5842), UINT16_C(37725), UINT16_C( 4063), UINT16_C(60974), UINT16_C(12656), UINT16_C(65284) },
      { UINT16_C(57157), UINT16_C(55463), UINT16_C(48688), UINT16_C(55169), UINT16_C(38484), UINT16_C(58238), UINT16_C(59791), UINT16_C(32489),
        UINT16_C(60311), UINT16_C(39066), UINT16_C(37691), UINT16_C(19957), UINT16_C(51393), UINT16_C(54396), UINT16_C(44488), UINT16_C( 5634),
        UINT16_C(49292), UINT16_C(49134), UINT16_C(13160), UINT16_C(48135), UINT16_C(63178), UINT16_C(56975), UINT16_C(32383), UINT16_C(60252),
        UINT16_C(28265), UINT16_C(25774), UINT16_C(41966), UINT16_C(60081), UINT16_C(11735), UINT16_C(60974), UINT16_C(18138), UINT16_C(65284) } },
    { { UINT16_C(62170), UINT16_C(17086), UINT16_C(50469), UINT16_C(61438), UINT16_C(36283), UINT16_C(29902), UINT16_C(10757), UINT16_C( 5472),
        UINT16_C(27753), UINT16_C(15199), UINT16_C(48258), UINT16_C(25038), UINT16_C(64716), UINT16_C(15439), UINT16_C(21293), UINT16_C( 2107),
        UINT16_C(63813), UINT16_C(27466), UINT16_C(18878), UINT16_C(31066), UINT16_C(10454), UINT16_C(56557), UINT16_C(19795), UINT16_C(48369),
        UINT16_C(20665), UINT16_C(15607), UINT16_C(50445), UINT16_C(55709), UINT16_C(60865), UINT16_C(61205), UINT16_C(20544), UINT16_C(34551) },
      { UINT16_C(16713), UINT16_C( 2033), UINT16_C(19338), UINT16_C(24960), UINT16_C(28020), UINT16_C(51005), UINT16_C(11963), UINT16_C(29827),
        UINT16_C(31358), UINT16_C(35760), UINT16_C(20031), UINT16_C(  100), UINT16_C(31035), UINT16_C(31727), UINT16_C(59081), UINT16_C( 4609),
        UINT16_C(61992), UINT16_C(45593), UINT16_C(39230), UINT16_C(45587), UINT16_C(20487), UINT16_C(49785), UINT16_C(64638), UINT16_C(64822),
        UINT16_C(59254), UINT16_C(46472), UINT16_C(60725), UINT16_C(28853), UINT16_C(42342), UINT16_C(12523), UINT16_C(60811), UINT16_C(45890) },
      { UINT16_C(62170), UINT16_C(17086), UINT16_C(50469), UINT16_C(61438), UINT16_C(36283), UINT16_C(51005), UINT16_C(11963), UINT16_C(29827),
        UINT16_C(31358), UINT16_C(35760), UINT16_C(48258), UINT16_C(25038), UINT16_C(64716), UINT16_C(31727), UINT16_C(59081), UINT16_C( 4609),
        UINT16_C(63813), UINT16_C(45593), UINT16_C(39230), UINT16_C(45587), UINT16_C(20487), UINT16_C(56557), UINT16_C(64638), UINT16_C(64822),
        UINT16_C(59254), UINT16_C(46472), UINT16_C(60725), UINT16_C(55709), UINT16_C(60865), UINT16_C(61205), UINT16_C(60811), UINT16_C(45890) } },
    { { UINT16_C(23775), UINT16_C( 7526), UINT16_C(31221), UINT16_C(64719), UINT16_C(18634), UINT16_C(18622), UINT16_C(62788), UINT16_C(47685),
        UINT16_C(52956), UINT16_C( 4463), UINT16_C( 9659), UINT16_C( 8577), UINT16_C(27850), UINT16_C(21841), UINT16_C(37977), UINT16_C(14601),
        UINT16_C(28656), UINT16_C(58710), UINT16_C( 9960), UINT16_C(45794), UINT16_C(41070), UINT16_C(46075), UINT16_C(16533), UINT16_C(29037),
        UINT16_C(56590), UINT16_C(51586), UINT16_C(  770), UINT16_C(52459), UINT16_C(15472), UINT16_C(51489), UINT16_C(10960), UINT16_C(49154) },
      { UINT16_C(22937), UINT16_C(33446), UINT16_C(34943), UINT16_C(60724), UINT16_C(12072), UINT16_C(48800), UINT16_C( 3696), UINT16_C(32303),
        UINT16_C(45803), UINT16_C(60744), UINT16_C(13237), UINT16_C( 9657), UINT16_C(55919), UINT16_C(16623), UINT16_C(61701), UINT16_C(40448),
        UINT16_C(42570), UINT16_C(51488), UINT16_C(21806), UINT16_C(22455), UINT16_C(22404), UINT16_C(62485), UINT16_C(17509), UINT16_C(20595),
        UINT16_C(48118), UINT16_C(44093), UINT16_C(63214), UINT16_C(24017), UINT16_C(49361), UINT16_C(54941), UINT16_C(40626), UINT16_C(64628) },
      { UINT16_C(23775), UINT16_C(33446), UINT16_C(34943), UINT16_C(64719), UINT16_C(18634), UINT16_C(48800), UINT16_C(62788), UINT16_C(47685),
        UINT16_C(52956), UINT16_C(60744), UINT16_C(13237), UINT16_C( 9657), UINT16_C(55919), UINT16_C(21841), UINT16_C(61701), UINT16_C(40448),
        UINT16_C(42570), UINT16_C(58710), UINT16_C(21806), UINT16_C(45794), UINT16_C(41070), UINT16_C(62485), UINT16_C(17509), UINT16_C(29037),
        UINT16_C(56590), UINT16_C(51586), UINT16_C(63214), UINT16_C(52459), UINT16_C(49361), UINT16_C(54941), UINT16_C(40626), UINT16_C(64628) } },
    { { UINT16_C(38212), UINT16_C(29638), UINT16_C(32234), UINT16_C(28362), UINT16_C(57300), UINT16_C(14947), UINT16_C(54819), UINT16_C( 6794),
        UINT16_C(51345), UINT16_C(32710), UINT16_C(38846), UINT16_C(36828), UINT16_C(31320), UINT16_C( 2661), UINT16_C(55832), UINT16_C(23558),
        UINT16_C(52335), UINT16_C(22991), UINT16_C(39241), UINT16_C( 7879), UINT16_C(10872), UINT16_C(40024), UINT16_C(57856), UINT16_C(37302),
        UINT16_C(31914), UINT16_C(26896), UINT16_C(60691), UINT16_C(27640), UINT16_C(24167), UINT16_C(32629), UINT16_C(31800), UINT16_C(42971) },
      { UINT16_C(43848), UINT16_C(37376), UINT16_C(51012), UINT16_C(48560), UINT16_C( 2290), UINT16_C(62041), UINT16_C( 4074), UINT16_C(38276),
        UINT16_C(38027), UINT16_C(40702), UINT16_C(63105), UINT16_C(59402), UINT16_C(32596), UINT16_C(35943), UINT16_C(17403), UINT16_C(17459),
        UINT16_C(13294), UINT16_C(13014), UINT16_C(34555), UINT16_C(60911), UINT16_C(18574), UINT16_C(30943), UINT16_C(25431), UINT16_C(57869),
        UINT16_C( 3064), UINT16_C(31105), UINT16_C(35586), UINT16_C(22114), UINT16_C(51466), UINT16_C( 1763), UINT16_C( 5644), UINT16_C(64074) },
      { UINT16_C(43848), UINT16_C(37376), UINT16_C(51012), UINT16_C(48560), UINT16_C(57300), UINT16_C(62041), UINT16_C(54819), UINT16_C(38276),
        UINT16_C(51345), UINT16_C(40702), UINT16_C(63105), UINT16_C(59402), UINT16_C(32596), UINT16_C(35943), UINT16_C(55832), UINT16_C(23558),
        UINT16_C(52335), UINT16_C(22991), UINT16_C(39241), UINT16_C(60911), UINT16_C(18574), UINT16_C(40024), UINT16_C(57856), UINT16_C(57869),
        UINT16_C(31914), UINT16_C(31105), UINT16_C(60691), UINT16_C(27640), UINT16_C(51466), UINT16_C(32629), UINT16_C(31800), UINT16_C(64074) } },
    { { UINT16_C( 8266), UINT16_C(17709), UINT16_C( 7334), UINT16_C(13362), UINT16_C( 4453), UINT16_C(48300), UINT16_C(47733), UINT16_C(28063),
        UINT16_C( 8389), UINT16_C(51174), UINT16_C(18603), UINT16_C(46366), UINT16_C(  274), UINT16_C( 7867), UINT16_C( 1303), UINT16_C(24857),
        UINT16_C(17957), UINT16_C(52134), UINT16_C(55394), UINT16_C(51199), UINT16_C(44266), UINT16_C(24452), UINT16_C( 9062), UINT16_C(11212),
        UINT16_C(45635), UINT16_C(61171), UINT16_C( 4603), UINT16_C( 3491), UINT16_C(24338), UINT16_C(10539), UINT16_C(17508), UINT16_C(35467) },
      { UINT16_C(12682), UINT16_C(60757), UINT16_C(21770), UINT16_C(62644), UINT16_C(14337), UINT16_C(26451), UINT16_C( 8027), UINT16_C(40594),
        UINT16_C(34257), UINT16_C(52364), UINT16_C(12438), UINT16_C(43225), UINT16_C( 1423), UINT16_C(62418), UINT16_C(23881), UINT16_C(54397),
        UINT16_C(54158), UINT16_C(39105), UINT16_C(29992), UINT16_C(10636), UINT16_C(57262), UINT16_C( 2448), UINT16_C( 8958), UINT16_C(53416),
        UINT16_C(13480), UINT16_C(16028), UINT16_C(30308), UINT16_C(62439), UINT16_C(47483), UINT16_C(50407), UINT16_C(25622), UINT16_C(42136) },
      { UINT16_C(12682), UINT16_C(60757), UINT16_C(21770), UINT16_C(62644), UINT16_C(14337), UINT16_C(48300), UINT16_C(47733), UINT16_C(40594),
        UINT16_C(34257), UINT16_C(52364), UINT16_C(18603), UINT16_C(46366), UINT16_C( 1423), UINT16_C(62418), UINT16_C(23881), UINT16_C(54397),
        UINT16_C(54158), UINT16_C(52134), UINT16_C(55394), UINT16_C(51199), UINT16_C(57262), UINT16_C(24452), UINT16_C( 9062), UINT16_C(53416),
        UINT16_C(45635), UINT16_C(61171), UINT16_C(30308), UINT16_C(62439), UINT16_C(47483), UINT16_C(50407), UINT16_C(25622), UINT16_C(42136) } },
    { { UINT16_C(22839), UINT16_C(24381), UINT16_C(51663), UINT16_C(32136), UINT16_C( 6313), UINT16_C(42886), UINT16_C(11835), UINT16_C(58231),
        UINT16_C( 5219), UINT16_C(50977), UINT16_C( 2186), UINT16_C( 1467), UINT16_C(41665), UINT16_C(55241), UINT16_C(25094), UINT16_C(15996),
        UINT16_C(47547), UINT16_C(35485), UINT16_C( 9858), UINT16_C(11015), UINT16_C(36414), UINT16_C(31187), UINT16_C(19132), UINT16_C( 8028),
        UINT16_C(32350), UINT16_C(59623), UINT16_C(41606), UINT16_C(18669), UINT16_C(46916), UINT16_C(18975), UINT16_C(39705), UINT16_C(54408) },
      { UINT16_C( 9812), UINT16_C(55135), UINT16_C(26188), UINT16_C(35330), UINT16_C(54772), UINT16_C(45316), UINT16_C(24608), UINT16_C(32464),
        UINT16_C(47070), UINT16_C(25959), UINT16_C(21593), UINT16_C(40365), UINT16_C(52235), UINT16_C( 9448), UINT16_C(28776), UINT16_C(48377),
        UINT16_C(22678), UINT16_C(58003), UINT16_C(38590), UINT16_C(45933), UINT16_C(29035), UINT16_C(35684), UINT16_C(13521), UINT16_C(45066),
        UINT16_C(29164), UINT16_C(17685), UINT16_C(49861), UINT16_C(53731), UINT16_C(52110), UINT16_C(63221), UINT16_C(60987), UINT16_C(53939) },
      { UINT16_C(22839), UINT16_C(55135), UINT16_C(51663), UINT16_C(35330), UINT16_C(54772), UINT16_C(45316), UINT16_C(24608), UINT16_C(58231),
        UINT16_C(47070), UINT16_C(50977), UINT16_C(21593), UINT16_C(40365), UINT16_C(52235), UINT16_C(55241), UINT16_C(28776), UINT16_C(48377),
        UINT16_C(47547), UINT16_C(58003), UINT16_C(38590), UINT16_C(45933), UINT16_C(36414), UINT16_C(35684), UINT16_C(19132), UINT16_C(45066),
        UINT16_C(32350), UINT16_C(59623), UINT16_C(49861), UINT16_C(53731), UINT16_C(52110), UINT16_C(63221), UINT16_C(60987), UINT16_C(54408) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_epu16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_epu16");
    easysimd_test_x86_assert_equal_u16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const uint16_t a[32];
    const uint16_t b[32];
    const uint16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 17970), -INT16_C( 32047),  INT16_C( 11126),  INT16_C( 18607),  INT16_C( 28846), -INT16_C( 15027),  INT16_C( 10228),  INT16_C(  3519),
        -INT16_C(  3833), -INT16_C(  8977), -INT16_C( 12540), -INT16_C(  4980), -INT16_C( 26471), -INT16_C(  1341),  INT16_C( 26461),  INT16_C( 11071),
         INT16_C(  4128), -INT16_C( 26962),  INT16_C( 23868), -INT16_C(  5409),  INT16_C( 11469), -INT16_C( 15953),  INT16_C( 28243),  INT16_C( 23246),
        -INT16_C( 17056),  INT16_C( 25654), -INT16_C( 15732),  INT16_C(  9808),  INT16_C(  5210), -INT16_C( 18400),  INT16_C( 24443), -INT16_C( 25629) },
      UINT32_C(2872217967),
      { UINT16_C(17469), UINT16_C(36990), UINT16_C(19635), UINT16_C( 5098), UINT16_C( 8458), UINT16_C(38519), UINT16_C(51171), UINT16_C(16060),
        UINT16_C(56539), UINT16_C(22262), UINT16_C(55611), UINT16_C(44018), UINT16_C( 9323), UINT16_C(23126), UINT16_C(60469), UINT16_C(29206),
        UINT16_C(37936), UINT16_C(58114), UINT16_C(60641), UINT16_C(60406), UINT16_C(27917), UINT16_C(61825), UINT16_C(15925), UINT16_C( 4143),
        UINT16_C( 9498), UINT16_C(22119), UINT16_C(23038), UINT16_C(26881), UINT16_C(22397), UINT16_C(45763), UINT16_C(55875), UINT16_C(29732) },
      { UINT16_C( 9838), UINT16_C(20311), UINT16_C(19986), UINT16_C( 8250), UINT16_C(48315), UINT16_C(61457), UINT16_C(16634), UINT16_C( 5121),
        UINT16_C(26725), UINT16_C(25450), UINT16_C(27585), UINT16_C(16077), UINT16_C(37059), UINT16_C( 1776), UINT16_C( 5226), UINT16_C(55674),
        UINT16_C(53818), UINT16_C(19496), UINT16_C(25376), UINT16_C(56172), UINT16_C(32031), UINT16_C( 6604), UINT16_C(52669), UINT16_C( 8749),
        UINT16_C(38965), UINT16_C(63110), UINT16_C(21251), UINT16_C(50740), UINT16_C( 9443), UINT16_C(20173), UINT16_C(18232), UINT16_C(29223) },
      { UINT16_C(17469), UINT16_C(36990), UINT16_C(19986), UINT16_C( 8250), UINT16_C(28846), UINT16_C(61457), UINT16_C(51171), UINT16_C( 3519),
        UINT16_C(56539), UINT16_C(56559), UINT16_C(52996), UINT16_C(60556), UINT16_C(37059), UINT16_C(64195), UINT16_C(26461), UINT16_C(55674),
        UINT16_C( 4128), UINT16_C(58114), UINT16_C(23868), UINT16_C(60127), UINT16_C(32031), UINT16_C(61825), UINT16_C(28243), UINT16_C(23246),
        UINT16_C(38965), UINT16_C(63110), UINT16_C(49804), UINT16_C(50740), UINT16_C( 5210), UINT16_C(45763), UINT16_C(24443), UINT16_C(29732) } },
    { {  INT16_C( 20249),  INT16_C( 14782),  INT16_C( 11186), -INT16_C( 12011), -INT16_C(  7768),  INT16_C( 26346),  INT16_C(  6318), -INT16_C(  7288),
         INT16_C(  3760), -INT16_C( 19495),  INT16_C(  3425),  INT16_C( 17786),  INT16_C( 18225),  INT16_C( 27027), -INT16_C( 17778), -INT16_C( 22309),
        -INT16_C( 26359), -INT16_C( 17183), -INT16_C(  2364),  INT16_C( 28045),  INT16_C( 30935), -INT16_C( 31277),  INT16_C( 23440),  INT16_C( 16488),
         INT16_C( 16746), -INT16_C( 13325),  INT16_C( 27982),  INT16_C( 32528), -INT16_C( 23628),  INT16_C( 17384), -INT16_C( 15523),  INT16_C( 26603) },
      UINT32_C( 555994205),
      { UINT16_C(24872), UINT16_C(47136), UINT16_C(35005), UINT16_C(10232), UINT16_C(60618), UINT16_C( 6386), UINT16_C(  857), UINT16_C( 3736),
        UINT16_C(32934), UINT16_C( 1105), UINT16_C(15428), UINT16_C(41323), UINT16_C(36360), UINT16_C(52162), UINT16_C(20798), UINT16_C(26470),
        UINT16_C(34482), UINT16_C(28447), UINT16_C( 6158), UINT16_C(55446), UINT16_C(35076), UINT16_C(24049), UINT16_C(35212), UINT16_C(12907),
        UINT16_C(48137), UINT16_C(19766), UINT16_C(41464), UINT16_C(  494), UINT16_C(45359), UINT16_C(28364), UINT16_C(12802), UINT16_C(46293) },
      { UINT16_C(62648), UINT16_C(50980), UINT16_C(47628), UINT16_C( 4255), UINT16_C(36931), UINT16_C(53102), UINT16_C(55577), UINT16_C( 8962),
        UINT16_C(14486), UINT16_C(36464), UINT16_C(24538), UINT16_C( 2447), UINT16_C(23568), UINT16_C( 4727), UINT16_C(19598), UINT16_C(18374),
        UINT16_C(59969), UINT16_C(19726), UINT16_C(44453), UINT16_C(59486), UINT16_C(52286), UINT16_C(22456), UINT16_C(47781), UINT16_C(15226),
        UINT16_C(60402), UINT16_C(52426), UINT16_C(22858), UINT16_C(23254), UINT16_C(19893), UINT16_C(17516), UINT16_C(12954), UINT16_C(56203) },
      { UINT16_C(62648), UINT16_C(14782), UINT16_C(47628), UINT16_C(10232), UINT16_C(60618), UINT16_C(26346), UINT16_C(55577), UINT16_C(58248),
        UINT16_C( 3760), UINT16_C(46041), UINT16_C(24538), UINT16_C(41323), UINT16_C(18225), UINT16_C(27027), UINT16_C(20798), UINT16_C(26470),
        UINT16_C(59969), UINT16_C(28447), UINT16_C(63172), UINT16_C(28045), UINT16_C(30935), UINT16_C(24049), UINT16_C(23440), UINT16_C(16488),
        UINT16_C(60402), UINT16_C(52211), UINT16_C(27982), UINT16_C(32528), UINT16_C(41908), UINT16_C(28364), UINT16_C(50013), UINT16_C(26603) } },
    { { -INT16_C( 26339), -INT16_C( 15832), -INT16_C( 31162), -INT16_C( 31574),  INT16_C( 25170), -INT16_C(  1828),  INT16_C( 22044),  INT16_C(  3891),
        -INT16_C(   703), -INT16_C( 29733), -INT16_C( 20137),  INT16_C(  3301),  INT16_C( 20991), -INT16_C( 26288), -INT16_C(  9340), -INT16_C( 24204),
        -INT16_C( 25484), -INT16_C( 17565),  INT16_C(  3363),  INT16_C( 30015),  INT16_C(  7024), -INT16_C( 29587), -INT16_C( 24206), -INT16_C( 19557),
         INT16_C( 30622), -INT16_C(  2753),  INT16_C(  9256),  INT16_C(  9986),  INT16_C( 21110), -INT16_C(  1344),  INT16_C( 13358), -INT16_C( 23909) },
      UINT32_C(4099800785),
      { UINT16_C(55224), UINT16_C(10760), UINT16_C(41848), UINT16_C( 5854), UINT16_C( 7450), UINT16_C(17164), UINT16_C( 3649), UINT16_C(46954),
        UINT16_C(11104), UINT16_C(36529), UINT16_C(19551), UINT16_C(12337), UINT16_C(36426), UINT16_C(22052), UINT16_C(36395), UINT16_C(58577),
        UINT16_C(55653), UINT16_C(56590), UINT16_C(60541), UINT16_C(38899), UINT16_C(65289), UINT16_C(19418), UINT16_C(17677), UINT16_C(28162),
        UINT16_C(46192), UINT16_C(53244), UINT16_C(11520), UINT16_C(19200), UINT16_C( 9404), UINT16_C(59297), UINT16_C(29362), UINT16_C( 6091) },
      { UINT16_C(55884), UINT16_C(51700), UINT16_C(59590), UINT16_C(53344), UINT16_C(15335), UINT16_C(62747), UINT16_C( 7552), UINT16_C(61539),
        UINT16_C(24529), UINT16_C(53951), UINT16_C(49037), UINT16_C(18717), UINT16_C(48868), UINT16_C(38448), UINT16_C(64560), UINT16_C(31918),
        UINT16_C(41686), UINT16_C(40005), UINT16_C(42634), UINT16_C(29292), UINT16_C(34785), UINT16_C(24935), UINT16_C(51877), UINT16_C(30289),
        UINT16_C( 4137), UINT16_C(46664), UINT16_C(26064), UINT16_C(46335), UINT16_C(12323), UINT16_C(21578), UINT16_C(63532), UINT16_C(  720) },
      { UINT16_C(55884), UINT16_C(49704), UINT16_C(34374), UINT16_C(33962), UINT16_C(15335), UINT16_C(63708), UINT16_C( 7552), UINT16_C(61539),
        UINT16_C(64833), UINT16_C(53951), UINT16_C(49037), UINT16_C(18717), UINT16_C(48868), UINT16_C(38448), UINT16_C(64560), UINT16_C(58577),
        UINT16_C(55653), UINT16_C(47971), UINT16_C(60541), UINT16_C(38899), UINT16_C(65289), UINT16_C(35949), UINT16_C(51877), UINT16_C(45979),
        UINT16_C(30622), UINT16_C(62783), UINT16_C(26064), UINT16_C( 9986), UINT16_C(12323), UINT16_C(59297), UINT16_C(63532), UINT16_C( 6091) } },
    { {  INT16_C(  5787),  INT16_C(  9630),  INT16_C(  3004), -INT16_C( 25193), -INT16_C(   366),  INT16_C( 14334),  INT16_C( 20424), -INT16_C(  3410),
        -INT16_C(  2465),  INT16_C( 12200), -INT16_C( 22436),  INT16_C( 32739),  INT16_C( 11992),  INT16_C(  1235), -INT16_C( 23514), -INT16_C( 16122),
        -INT16_C( 23366),  INT16_C( 30439),  INT16_C( 32431),  INT16_C( 16915),  INT16_C(  4477),  INT16_C( 17785),  INT16_C( 10080), -INT16_C( 16585),
        -INT16_C(  8162),  INT16_C( 31471), -INT16_C( 11640),  INT16_C( 24825), -INT16_C( 13056),  INT16_C( 10084),  INT16_C( 27249),  INT16_C( 11240) },
      UINT32_C(3198275342),
      { UINT16_C(31173), UINT16_C( 9488), UINT16_C(18593), UINT16_C(49124), UINT16_C(54056), UINT16_C(45113), UINT16_C(12966), UINT16_C(42512),
        UINT16_C(29951), UINT16_C(28877), UINT16_C(46814), UINT16_C(60571), UINT16_C(15493), UINT16_C(54186), UINT16_C(43760), UINT16_C(46494),
        UINT16_C(44836), UINT16_C(50650), UINT16_C(49143), UINT16_C( 8068), UINT16_C(48530), UINT16_C(14543), UINT16_C(57327), UINT16_C(61407),
        UINT16_C(44115), UINT16_C(12639), UINT16_C(64354), UINT16_C(59421), UINT16_C(51255), UINT16_C(10427), UINT16_C(23154), UINT16_C(38621) },
      { UINT16_C(47113), UINT16_C(   91), UINT16_C(57207), UINT16_C( 2335), UINT16_C(61084), UINT16_C(35906), UINT16_C( 8653), UINT16_C( 8315),
        UINT16_C(56013), UINT16_C(12369), UINT16_C(28373), UINT16_C( 3352), UINT16_C(54070), UINT16_C(43317), UINT16_C( 4653), UINT16_C(13887),
        UINT16_C(39882), UINT16_C(16694), UINT16_C(21882), UINT16_C( 5963), UINT16_C(36163), UINT16_C( 4259), UINT16_C( 7854), UINT16_C(31536),
        UINT16_C(33272), UINT16_C(52907), UINT16_C(50160), UINT16_C( 9947), UINT16_C( 4247), UINT16_C(50383), UINT16_C( 3874), UINT16_C(60923) },
      { UINT16_C( 5787), UINT16_C( 9488), UINT16_C(57207), UINT16_C(49124), UINT16_C(65170), UINT16_C(14334), UINT16_C(20424), UINT16_C(62126),
        UINT16_C(56013), UINT16_C(28877), UINT16_C(46814), UINT16_C(60571), UINT16_C(11992), UINT16_C( 1235), UINT16_C(43760), UINT16_C(46494),
        UINT16_C(44836), UINT16_C(30439), UINT16_C(32431), UINT16_C(16915), UINT16_C( 4477), UINT16_C(14543), UINT16_C(10080), UINT16_C(61407),
        UINT16_C(57374), UINT16_C(52907), UINT16_C(64354), UINT16_C(59421), UINT16_C(51255), UINT16_C(50383), UINT16_C(27249), UINT16_C(60923) } },
    { {  INT16_C( 12714),  INT16_C(  9262),  INT16_C( 31111), -INT16_C( 13765), -INT16_C(  8698), -INT16_C( 19237),  INT16_C(  3068), -INT16_C(  2768),
        -INT16_C(  9331),  INT16_C( 32195), -INT16_C( 24929),  INT16_C( 13987),  INT16_C( 29614), -INT16_C( 12038), -INT16_C(  2686),  INT16_C( 11453),
        -INT16_C(  5081), -INT16_C( 20912), -INT16_C( 29595),  INT16_C( 27768),  INT16_C( 21354),  INT16_C( 26400),  INT16_C( 20575), -INT16_C(  5028),
         INT16_C(  7980), -INT16_C( 13463),  INT16_C(  3261),  INT16_C( 27393), -INT16_C(  1153),  INT16_C(   315), -INT16_C(  1551),  INT16_C(  6189) },
      UINT32_C(1254522597),
      { UINT16_C(55186), UINT16_C(61915), UINT16_C(14119), UINT16_C(21469), UINT16_C(18006), UINT16_C( 4894), UINT16_C( 8018), UINT16_C(53886),
        UINT16_C(47643), UINT16_C( 3283), UINT16_C(  435), UINT16_C(38948), UINT16_C(60031), UINT16_C(35298), UINT16_C(39208), UINT16_C(47869),
        UINT16_C(55664), UINT16_C(38827), UINT16_C(34832), UINT16_C(26603), UINT16_C( 2510), UINT16_C( 8570), UINT16_C(63785), UINT16_C(17651),
        UINT16_C(50867), UINT16_C(26192), UINT16_C(29895), UINT16_C(18174), UINT16_C(57438), UINT16_C(34511), UINT16_C(52601), UINT16_C(59713) },
      { UINT16_C(60582), UINT16_C(46721), UINT16_C(27765), UINT16_C(17181), UINT16_C(39029), UINT16_C(40548), UINT16_C(22417), UINT16_C(17634),
        UINT16_C(12830), UINT16_C(58794), UINT16_C(43174), UINT16_C( 1068), UINT16_C(64392), UINT16_C(  651), UINT16_C(52424), UINT16_C(28395),
        UINT16_C(27832), UINT16_C(11557), UINT16_C(17112), UINT16_C(20081), UINT16_C(54746), UINT16_C(27628), UINT16_C(53037), UINT16_C(19375),
        UINT16_C(22785), UINT16_C(43056), UINT16_C(23553), UINT16_C(35500), UINT16_C(14168), UINT16_C( 8332), UINT16_C(30467), UINT16_C(48271) },
      { UINT16_C(60582), UINT16_C( 9262), UINT16_C(27765), UINT16_C(51771), UINT16_C(56838), UINT16_C(40548), UINT16_C(22417), UINT16_C(53886),
        UINT16_C(56205), UINT16_C(58794), UINT16_C(43174), UINT16_C(38948), UINT16_C(64392), UINT16_C(35298), UINT16_C(52424), UINT16_C(11453),
        UINT16_C(60455), UINT16_C(38827), UINT16_C(34832), UINT16_C(27768), UINT16_C(21354), UINT16_C(26400), UINT16_C(63785), UINT16_C(19375),
        UINT16_C( 7980), UINT16_C(43056), UINT16_C( 3261), UINT16_C(35500), UINT16_C(64383), UINT16_C(  315), UINT16_C(52601), UINT16_C( 6189) } },
    { { -INT16_C( 19228), -INT16_C( 17175),  INT16_C( 23286), -INT16_C( 12022), -INT16_C(  2256),  INT16_C( 23868), -INT16_C(  4922), -INT16_C( 14424),
        -INT16_C( 10171),  INT16_C( 18287),  INT16_C(  7221), -INT16_C( 29231),  INT16_C( 23891),  INT16_C( 22445),  INT16_C( 15572), -INT16_C( 18413),
        -INT16_C(   784), -INT16_C(  6283),  INT16_C( 32599), -INT16_C( 30792), -INT16_C(  2954),  INT16_C( 15588), -INT16_C( 29472),  INT16_C(  9732),
         INT16_C( 29540), -INT16_C( 26259),  INT16_C( 16015), -INT16_C(  7386), -INT16_C( 11109),  INT16_C( 28474),  INT16_C( 19728),  INT16_C(   296) },
      UINT32_C(2699599177),
      { UINT16_C( 2964), UINT16_C(30159), UINT16_C(54167), UINT16_C(64667), UINT16_C( 2119), UINT16_C(54933), UINT16_C(48198), UINT16_C(57785),
        UINT16_C(62352), UINT16_C(41040), UINT16_C(30784), UINT16_C(35489), UINT16_C(35093), UINT16_C(12842), UINT16_C(21033), UINT16_C(48837),
        UINT16_C(37981), UINT16_C(62771), UINT16_C(52840), UINT16_C(45041), UINT16_C(34518), UINT16_C( 7301), UINT16_C(16194), UINT16_C(54013),
        UINT16_C(19762), UINT16_C(29555), UINT16_C( 5318), UINT16_C(56317), UINT16_C(10142), UINT16_C(50957), UINT16_C(53881), UINT16_C(55173) },
      { UINT16_C(47207), UINT16_C(53196), UINT16_C(48518), UINT16_C(23678), UINT16_C(  835), UINT16_C(34424), UINT16_C(30018), UINT16_C(30040),
        UINT16_C(52163), UINT16_C(35304), UINT16_C(58848), UINT16_C(32356), UINT16_C(29196), UINT16_C(34373), UINT16_C(52036), UINT16_C(43869),
        UINT16_C(10627), UINT16_C( 2682), UINT16_C(63718), UINT16_C(10598), UINT16_C(57340), UINT16_C(16047), UINT16_C( 2132), UINT16_C( 6067),
        UINT16_C(39891), UINT16_C(45984), UINT16_C( 1408), UINT16_C(36145), UINT16_C(30583), UINT16_C(47891), UINT16_C(28738), UINT16_C(50535) },
      { UINT16_C(47207), UINT16_C(48361), UINT16_C(23286), UINT16_C(64667), UINT16_C(63280), UINT16_C(23868), UINT16_C(48198), UINT16_C(51112),
        UINT16_C(62352), UINT16_C(18287), UINT16_C(58848), UINT16_C(35489), UINT16_C(35093), UINT16_C(22445), UINT16_C(15572), UINT16_C(48837),
        UINT16_C(64752), UINT16_C(59253), UINT16_C(32599), UINT16_C(45041), UINT16_C(62582), UINT16_C(16047), UINT16_C(16194), UINT16_C(54013),
        UINT16_C(29540), UINT16_C(39277), UINT16_C(16015), UINT16_C(58150), UINT16_C(54427), UINT16_C(50957), UINT16_C(19728), UINT16_C(55173) } },
    { { -INT16_C(  7783),  INT16_C( 32719),  INT16_C( 14042), -INT16_C( 10584),  INT16_C( 22549),  INT16_C( 26900), -INT16_C( 14240),  INT16_C( 13185),
         INT16_C(  8547), -INT16_C(  6937),  INT16_C(  6182), -INT16_C( 25231), -INT16_C( 31601), -INT16_C( 11943), -INT16_C( 16140), -INT16_C( 29289),
         INT16_C( 26273),  INT16_C( 31500), -INT16_C( 19300), -INT16_C( 20143),  INT16_C( 26124),  INT16_C( 27675), -INT16_C( 25554), -INT16_C( 28256),
        -INT16_C( 30787), -INT16_C(  7051), -INT16_C(  6497),  INT16_C( 12161), -INT16_C(  9622),  INT16_C( 24064), -INT16_C( 26726),  INT16_C( 15595) },
      UINT32_C(2595747838),
      { UINT16_C(26479), UINT16_C(40229), UINT16_C(50435), UINT16_C(49198), UINT16_C(42060), UINT16_C(60324), UINT16_C( 9866), UINT16_C(62746),
        UINT16_C( 6912), UINT16_C(39763), UINT16_C(16306), UINT16_C(45271), UINT16_C(36406), UINT16_C(57931), UINT16_C(38807), UINT16_C( 1691),
        UINT16_C(49406), UINT16_C(  419), UINT16_C(53893), UINT16_C(53697), UINT16_C(26230), UINT16_C(  188), UINT16_C(55180), UINT16_C(36085),
        UINT16_C(18930), UINT16_C(42023), UINT16_C(65160), UINT16_C(48725), UINT16_C(41101), UINT16_C( 9377), UINT16_C(15415), UINT16_C(13611) },
      { UINT16_C(52988), UINT16_C(33078), UINT16_C(63392), UINT16_C( 5714), UINT16_C( 3677), UINT16_C(59671), UINT16_C( 3301), UINT16_C(55158),
        UINT16_C(40277), UINT16_C(56700), UINT16_C(53660), UINT16_C(10652), UINT16_C(15729), UINT16_C(43085), UINT16_C(30841), UINT16_C(30173),
        UINT16_C( 4935), UINT16_C(59382), UINT16_C(18442), UINT16_C(26878), UINT16_C( 5462), UINT16_C(15441), UINT16_C(50977), UINT16_C(30483),
        UINT16_C(36709), UINT16_C(  340), UINT16_C(61536), UINT16_C(53546), UINT16_C(30509), UINT16_C(42617), UINT16_C(22256), UINT16_C(14107) },
      { UINT16_C(57753), UINT16_C(40229), UINT16_C(63392), UINT16_C(49198), UINT16_C(42060), UINT16_C(60324), UINT16_C( 9866), UINT16_C(62746),
        UINT16_C(40277), UINT16_C(56700), UINT16_C(53660), UINT16_C(40305), UINT16_C(36406), UINT16_C(57931), UINT16_C(38807), UINT16_C(30173),
        UINT16_C(49406), UINT16_C(59382), UINT16_C(53893), UINT16_C(45393), UINT16_C(26230), UINT16_C(15441), UINT16_C(39982), UINT16_C(36085),
        UINT16_C(34749), UINT16_C(42023), UINT16_C(59039), UINT16_C(53546), UINT16_C(41101), UINT16_C(24064), UINT16_C(38810), UINT16_C(14107) } },
    { {  INT16_C(  4457),  INT16_C( 29726),  INT16_C(  7257), -INT16_C( 20260),  INT16_C( 11569),  INT16_C( 21484), -INT16_C(    11),  INT16_C( 23242),
         INT16_C(  7823), -INT16_C(  4261), -INT16_C( 31473),  INT16_C( 15553),  INT16_C( 15100), -INT16_C(  4893), -INT16_C(   367), -INT16_C(  1501),
         INT16_C( 16912),  INT16_C( 26990),  INT16_C( 19038), -INT16_C( 28647),  INT16_C(  1400),  INT16_C( 28131), -INT16_C( 21243), -INT16_C( 27449),
         INT16_C(  8907), -INT16_C(  9597),  INT16_C( 17575), -INT16_C( 23785), -INT16_C(  1409),  INT16_C(  4240), -INT16_C( 19464),  INT16_C(  2058) },
      UINT32_C(1416788469),
      { UINT16_C(51089), UINT16_C(38568), UINT16_C(28532), UINT16_C(16170), UINT16_C(44433), UINT16_C(14362), UINT16_C(12786), UINT16_C(29148),
        UINT16_C(27691), UINT16_C( 9089), UINT16_C(35615), UINT16_C( 5420), UINT16_C(40452), UINT16_C(51305), UINT16_C(19753), UINT16_C(47619),
        UINT16_C(44052), UINT16_C(34896), UINT16_C(31259), UINT16_C(44487), UINT16_C(57640), UINT16_C( 6885), UINT16_C(49426), UINT16_C(15755),
        UINT16_C( 3117), UINT16_C(19809), UINT16_C(36247), UINT16_C(40034), UINT16_C(52011), UINT16_C(21604), UINT16_C(26392), UINT16_C(11279) },
      { UINT16_C(24339), UINT16_C(12212), UINT16_C(31706), UINT16_C(  732), UINT16_C(49501), UINT16_C(28444), UINT16_C(42883), UINT16_C(45229),
        UINT16_C( 3763), UINT16_C(19197), UINT16_C(24475), UINT16_C(50918), UINT16_C(18986), UINT16_C(16922), UINT16_C(10674), UINT16_C(50542),
        UINT16_C( 8841), UINT16_C(25588), UINT16_C(53406), UINT16_C(64357), UINT16_C(33170), UINT16_C( 5482), UINT16_C( 5928), UINT16_C(56261),
        UINT16_C(49957), UINT16_C(49189), UINT16_C( 3106), UINT16_C(19846), UINT16_C(41302), UINT16_C( 2191), UINT16_C(65226), UINT16_C(21454) },
      { UINT16_C(51089), UINT16_C(29726), UINT16_C(31706), UINT16_C(45276), UINT16_C(49501), UINT16_C(28444), UINT16_C(42883), UINT16_C(45229),
        UINT16_C(27691), UINT16_C(61275), UINT16_C(34063), UINT16_C(50918), UINT16_C(40452), UINT16_C(51305), UINT16_C(19753), UINT16_C(64035),
        UINT16_C(16912), UINT16_C(34896), UINT16_C(19038), UINT16_C(36889), UINT16_C(57640), UINT16_C( 6885), UINT16_C(49426), UINT16_C(38087),
        UINT16_C( 8907), UINT16_C(55939), UINT16_C(36247), UINT16_C(41751), UINT16_C(52011), UINT16_C( 4240), UINT16_C(65226), UINT16_C( 2058) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k  = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_epu16(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_epu16");
    easysimd_test_x86_assert_equal_u16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_epu16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask32 k;
    const uint16_t a[32];
    const uint16_t b[32];
    const uint16_t r[32];
  } test_vec[] = {
    { UINT32_C(2682918652),
      { UINT16_C( 3360), UINT16_C(28585), UINT16_C(34194), UINT16_C(19929), UINT16_C(64935), UINT16_C(52907), UINT16_C( 3569), UINT16_C(10171),
        UINT16_C( 8743), UINT16_C(32403), UINT16_C(16511), UINT16_C(31580), UINT16_C(18006), UINT16_C( 5403), UINT16_C( 9109), UINT16_C(46787),
        UINT16_C(27697), UINT16_C(49957), UINT16_C(65265), UINT16_C(38928), UINT16_C(48380), UINT16_C(60774), UINT16_C( 8649), UINT16_C(61460),
        UINT16_C(42820), UINT16_C(50031), UINT16_C(52200), UINT16_C(15934), UINT16_C(22801), UINT16_C(42580), UINT16_C( 6013), UINT16_C(44636) },
      { UINT16_C(33155), UINT16_C(29809), UINT16_C(33408), UINT16_C(31756), UINT16_C(29246), UINT16_C( 1897), UINT16_C(32147), UINT16_C(55287),
        UINT16_C(26148), UINT16_C( 3226), UINT16_C(55601), UINT16_C(16971), UINT16_C(40754), UINT16_C(45033), UINT16_C(17846), UINT16_C(14685),
        UINT16_C(53191), UINT16_C(18349), UINT16_C(47441), UINT16_C(36803), UINT16_C(11307), UINT16_C(48790), UINT16_C(36265), UINT16_C(52630),
        UINT16_C(12532), UINT16_C( 9690), UINT16_C( 9481), UINT16_C(15464), UINT16_C(20932), UINT16_C(31467), UINT16_C(18838), UINT16_C(23987) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(34194), UINT16_C(31756), UINT16_C(64935), UINT16_C(52907), UINT16_C(32147), UINT16_C(55287),
        UINT16_C(    0), UINT16_C(32403), UINT16_C(55601), UINT16_C(    0), UINT16_C(40754), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(49957), UINT16_C(    0), UINT16_C(38928), UINT16_C(    0), UINT16_C(60774), UINT16_C(36265), UINT16_C(61460),
        UINT16_C(42820), UINT16_C(50031), UINT16_C(52200), UINT16_C(15934), UINT16_C(22801), UINT16_C(    0), UINT16_C(    0), UINT16_C(44636) } },
    { UINT32_C(1772380184),
      { UINT16_C(36499), UINT16_C(15362), UINT16_C(38939), UINT16_C( 3850), UINT16_C(58569), UINT16_C(53813), UINT16_C(40201), UINT16_C(52494),
        UINT16_C(64238), UINT16_C(33863), UINT16_C(64067), UINT16_C(23522), UINT16_C(34394), UINT16_C(29636), UINT16_C(48366), UINT16_C(33207),
        UINT16_C(47434), UINT16_C(26046), UINT16_C(51282), UINT16_C( 7029), UINT16_C(43692), UINT16_C(46573), UINT16_C(64583), UINT16_C(13698),
        UINT16_C(51702), UINT16_C(14777), UINT16_C(39875), UINT16_C( 7572), UINT16_C(22562), UINT16_C( 4240), UINT16_C(18196), UINT16_C(24209) },
      { UINT16_C(20224), UINT16_C(21187), UINT16_C(14359), UINT16_C(50029), UINT16_C(23522), UINT16_C(10616), UINT16_C(64087), UINT16_C(19806),
        UINT16_C( 6339), UINT16_C(34438), UINT16_C( 6835), UINT16_C(54691), UINT16_C(13170), UINT16_C(34533), UINT16_C(30586), UINT16_C(31716),
        UINT16_C(42950), UINT16_C(57037), UINT16_C(15328), UINT16_C(49825), UINT16_C( 6806), UINT16_C(60908), UINT16_C(18964), UINT16_C(55354),
        UINT16_C(49250), UINT16_C( 5726), UINT16_C(  730), UINT16_C(19691), UINT16_C(53557), UINT16_C(45266), UINT16_C(46664), UINT16_C( 3627) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(50029), UINT16_C(58569), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(34533), UINT16_C(48366), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(51282), UINT16_C(    0), UINT16_C(    0), UINT16_C(60908), UINT16_C(    0), UINT16_C(55354),
        UINT16_C(51702), UINT16_C(    0), UINT16_C(    0), UINT16_C(19691), UINT16_C(    0), UINT16_C(45266), UINT16_C(46664), UINT16_C(    0) } },
    { UINT32_C(1038940253),
      { UINT16_C(60584), UINT16_C(48310), UINT16_C(61494), UINT16_C(39316), UINT16_C(62384), UINT16_C(35503), UINT16_C(39669), UINT16_C(10966),
        UINT16_C(43115), UINT16_C(46042), UINT16_C( 1374), UINT16_C(48322), UINT16_C(44798), UINT16_C(12793), UINT16_C(63804), UINT16_C(58619),
        UINT16_C(45541), UINT16_C( 7329), UINT16_C(13730), UINT16_C(21173), UINT16_C(25640), UINT16_C( 7645), UINT16_C(46078), UINT16_C(27208),
        UINT16_C( 8796), UINT16_C(47645), UINT16_C(57128), UINT16_C( 9846), UINT16_C(28814), UINT16_C(51799), UINT16_C(21097), UINT16_C(20399) },
      { UINT16_C(20484), UINT16_C(42603), UINT16_C( 8325), UINT16_C(44792), UINT16_C(54660), UINT16_C(33483), UINT16_C( 5001), UINT16_C(58860),
        UINT16_C( 2614), UINT16_C(24223), UINT16_C( 5865), UINT16_C(30596), UINT16_C(56198), UINT16_C(61250), UINT16_C(61742), UINT16_C(12862),
        UINT16_C(43329), UINT16_C(50904), UINT16_C(53449), UINT16_C(19828), UINT16_C(16550), UINT16_C(12240), UINT16_C(48211), UINT16_C(35092),
        UINT16_C(46022), UINT16_C(45287), UINT16_C(27593), UINT16_C(20263), UINT16_C(26951), UINT16_C(30015), UINT16_C(32090), UINT16_C(39847) },
      { UINT16_C(60584), UINT16_C(    0), UINT16_C(61494), UINT16_C(44792), UINT16_C(62384), UINT16_C(    0), UINT16_C(39669), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(48322), UINT16_C(56198), UINT16_C(61250), UINT16_C(63804), UINT16_C(58619),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(53449), UINT16_C(21173), UINT16_C(    0), UINT16_C(12240), UINT16_C(48211), UINT16_C(35092),
        UINT16_C(46022), UINT16_C(    0), UINT16_C(57128), UINT16_C(20263), UINT16_C(28814), UINT16_C(51799), UINT16_C(    0), UINT16_C(    0) } },
    { UINT32_C(4032986919),
      { UINT16_C( 3606), UINT16_C(27172), UINT16_C(14538), UINT16_C(37363), UINT16_C(56300), UINT16_C(46401), UINT16_C(26694), UINT16_C(36101),
        UINT16_C(17618), UINT16_C(11266), UINT16_C(43457), UINT16_C(59592), UINT16_C(10792), UINT16_C(30937), UINT16_C( 5888), UINT16_C( 5997),
        UINT16_C(37413), UINT16_C(61313), UINT16_C(29898), UINT16_C(46720), UINT16_C(49487), UINT16_C(38508), UINT16_C(28970), UINT16_C(64547),
        UINT16_C( 9909), UINT16_C(30248), UINT16_C(61647), UINT16_C(63583), UINT16_C(14362), UINT16_C( 7024), UINT16_C(56655), UINT16_C(29746) },
      { UINT16_C(45935), UINT16_C(14947), UINT16_C(58407), UINT16_C(30704), UINT16_C(23717), UINT16_C(53005), UINT16_C(12493), UINT16_C(33483),
        UINT16_C(62550), UINT16_C( 9977), UINT16_C(22756), UINT16_C(65310), UINT16_C(36496), UINT16_C(57114), UINT16_C(19563), UINT16_C(56147),
        UINT16_C(46847), UINT16_C( 9749), UINT16_C( 1434), UINT16_C(16541), UINT16_C(43618), UINT16_C(12047), UINT16_C(56283), UINT16_C(12722),
        UINT16_C(43983), UINT16_C(45911), UINT16_C(29955), UINT16_C(37810), UINT16_C(52227), UINT16_C(28530), UINT16_C(50456), UINT16_C( 5962) },
      { UINT16_C(45935), UINT16_C(27172), UINT16_C(58407), UINT16_C(    0), UINT16_C(    0), UINT16_C(53005), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(62550), UINT16_C(11266), UINT16_C(43457), UINT16_C(65310), UINT16_C(36496), UINT16_C(57114), UINT16_C(19563), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(61313), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(38508), UINT16_C(56283), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(52227), UINT16_C(28530), UINT16_C(56655), UINT16_C(29746) } },
    { UINT32_C( 373186427),
      { UINT16_C(25990), UINT16_C(25078), UINT16_C(43072), UINT16_C( 3986), UINT16_C(59987), UINT16_C(22211), UINT16_C(30047), UINT16_C(25577),
        UINT16_C(23362), UINT16_C(23250), UINT16_C( 7200), UINT16_C(39794), UINT16_C(45179), UINT16_C(57265), UINT16_C( 1931), UINT16_C( 4518),
        UINT16_C(40045), UINT16_C(44402), UINT16_C( 1348), UINT16_C(38845), UINT16_C(33007), UINT16_C(20205), UINT16_C(55029), UINT16_C(14257),
        UINT16_C(33585), UINT16_C(20882), UINT16_C( 1183), UINT16_C( 6892), UINT16_C(40628), UINT16_C(16378), UINT16_C(41125), UINT16_C( 4689) },
      { UINT16_C(49980), UINT16_C(32960), UINT16_C(32200), UINT16_C(46871), UINT16_C( 1277), UINT16_C(61958), UINT16_C(47066), UINT16_C( 2858),
        UINT16_C(48187), UINT16_C(55900), UINT16_C(18624), UINT16_C(29941), UINT16_C(61414), UINT16_C(36019), UINT16_C( 1167), UINT16_C(52126),
        UINT16_C(24264), UINT16_C(36939), UINT16_C(25307), UINT16_C(55368), UINT16_C(20070), UINT16_C(16587), UINT16_C(62725), UINT16_C(16459),
        UINT16_C(42929), UINT16_C(28955), UINT16_C( 4335), UINT16_C(55013), UINT16_C(39167), UINT16_C(36450), UINT16_C(  157), UINT16_C(25945) },
      { UINT16_C(49980), UINT16_C(32960), UINT16_C(    0), UINT16_C(46871), UINT16_C(59987), UINT16_C(61958), UINT16_C(47066), UINT16_C(    0),
        UINT16_C(48187), UINT16_C(55900), UINT16_C(18624), UINT16_C(39794), UINT16_C(61414), UINT16_C(    0), UINT16_C( 1931), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(44402), UINT16_C(25307), UINT16_C(55368), UINT16_C(33007), UINT16_C(20205), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(28955), UINT16_C( 4335), UINT16_C(    0), UINT16_C(40628), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } },
    { UINT32_C( 989176927),
      { UINT16_C(56971), UINT16_C(37292), UINT16_C(63443), UINT16_C(34001), UINT16_C(60574), UINT16_C(36341), UINT16_C(56060), UINT16_C(64355),
        UINT16_C(50546), UINT16_C( 3977), UINT16_C(58054), UINT16_C( 9588), UINT16_C(27270), UINT16_C(35935), UINT16_C(29351), UINT16_C(13304),
        UINT16_C(42064), UINT16_C( 9156), UINT16_C(38299), UINT16_C(14759), UINT16_C(40066), UINT16_C(32455), UINT16_C(10870), UINT16_C(59770),
        UINT16_C( 1008), UINT16_C(46840), UINT16_C(28134), UINT16_C(27867), UINT16_C(15063), UINT16_C(32505), UINT16_C(61869), UINT16_C(64945) },
      { UINT16_C(30102), UINT16_C(12577), UINT16_C(51211), UINT16_C(36203), UINT16_C(12901), UINT16_C(56075), UINT16_C(34140), UINT16_C(19652),
        UINT16_C(48521), UINT16_C(28418), UINT16_C(56618), UINT16_C(  475), UINT16_C(54296), UINT16_C(50559), UINT16_C(12742), UINT16_C(23746),
        UINT16_C(58278), UINT16_C(45453), UINT16_C(63660), UINT16_C( 4414), UINT16_C(18986), UINT16_C(34796), UINT16_C(45519), UINT16_C(22739),
        UINT16_C(54894), UINT16_C(39111), UINT16_C(41907), UINT16_C(52121), UINT16_C( 6263), UINT16_C(15760), UINT16_C(21321), UINT16_C(61593) },
      { UINT16_C(56971), UINT16_C(37292), UINT16_C(63443), UINT16_C(36203), UINT16_C(60574), UINT16_C(    0), UINT16_C(56060), UINT16_C(    0),
        UINT16_C(    0), UINT16_C(    0), UINT16_C(58054), UINT16_C(    0), UINT16_C(    0), UINT16_C(50559), UINT16_C(    0), UINT16_C(23746),
        UINT16_C(58278), UINT16_C(    0), UINT16_C(63660), UINT16_C(    0), UINT16_C(40066), UINT16_C(34796), UINT16_C(45519), UINT16_C(59770),
        UINT16_C(    0), UINT16_C(46840), UINT16_C(    0), UINT16_C(52121), UINT16_C(15063), UINT16_C(32505), UINT16_C(    0), UINT16_C(    0) } },
    { UINT32_C(3802212150),
      { UINT16_C(57386), UINT16_C(63953), UINT16_C(42129), UINT16_C(65362), UINT16_C( 6522), UINT16_C(11927), UINT16_C(12476), UINT16_C(13561),
        UINT16_C(35400), UINT16_C(37489), UINT16_C( 3037), UINT16_C( 4994), UINT16_C( 9010), UINT16_C(20982), UINT16_C(59651), UINT16_C(11675),
        UINT16_C(27849), UINT16_C(23079), UINT16_C(30993), UINT16_C(35673), UINT16_C(61586), UINT16_C(20409), UINT16_C(45856), UINT16_C(27011),
        UINT16_C(62525), UINT16_C( 6907), UINT16_C(32255), UINT16_C(12589), UINT16_C( 9120), UINT16_C(42115), UINT16_C( 7693), UINT16_C(54993) },
      { UINT16_C(63627), UINT16_C(39985), UINT16_C(35441), UINT16_C( 1063), UINT16_C(57723), UINT16_C(39763), UINT16_C(54932), UINT16_C(53508),
        UINT16_C(65482), UINT16_C(51947), UINT16_C( 6268), UINT16_C( 7675), UINT16_C(32316), UINT16_C(18881), UINT16_C(37533), UINT16_C(10271),
        UINT16_C(20619), UINT16_C(64708), UINT16_C(60379), UINT16_C(22016), UINT16_C(21452), UINT16_C(24817), UINT16_C(63017), UINT16_C(62513),
        UINT16_C( 7413), UINT16_C(29374), UINT16_C(47413), UINT16_C(29071), UINT16_C(20536), UINT16_C(54714), UINT16_C(55778), UINT16_C(28157) },
      { UINT16_C(    0), UINT16_C(63953), UINT16_C(42129), UINT16_C(    0), UINT16_C(57723), UINT16_C(39763), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(65482), UINT16_C(51947), UINT16_C( 6268), UINT16_C(    0), UINT16_C(    0), UINT16_C(20982), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(27849), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(24817), UINT16_C(    0), UINT16_C(62513),
        UINT16_C(    0), UINT16_C(29374), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(54714), UINT16_C(55778), UINT16_C(54993) } },
    { UINT32_C(  90882346),
      { UINT16_C(19646), UINT16_C(59353), UINT16_C( 2882), UINT16_C(14555), UINT16_C(39207), UINT16_C(23722), UINT16_C(14675), UINT16_C(35789),
        UINT16_C(34697), UINT16_C(27488), UINT16_C(23905), UINT16_C(35801), UINT16_C(17182), UINT16_C(51856), UINT16_C(60333), UINT16_C(27459),
        UINT16_C( 7479), UINT16_C(31315), UINT16_C(11816), UINT16_C(20402), UINT16_C(23752), UINT16_C( 7084), UINT16_C(31125), UINT16_C( 7846),
        UINT16_C( 1537), UINT16_C(25225), UINT16_C(25187), UINT16_C(33261), UINT16_C(32165), UINT16_C(21323), UINT16_C(36712), UINT16_C(40894) },
      { UINT16_C( 4524), UINT16_C(54297), UINT16_C(52032), UINT16_C( 2083), UINT16_C(53031), UINT16_C(48163), UINT16_C(51529), UINT16_C(19162),
        UINT16_C(25807), UINT16_C(12972), UINT16_C(39366), UINT16_C(27827), UINT16_C(65046), UINT16_C(32447), UINT16_C(32141), UINT16_C(14621),
        UINT16_C(14223), UINT16_C(53005), UINT16_C(12546), UINT16_C(10967), UINT16_C(64000), UINT16_C(18918), UINT16_C(49603), UINT16_C(37523),
        UINT16_C(16165), UINT16_C(60356), UINT16_C(30680), UINT16_C(61015), UINT16_C( 5749), UINT16_C(  876), UINT16_C(35476), UINT16_C( 9020) },
      { UINT16_C(    0), UINT16_C(59353), UINT16_C(    0), UINT16_C(14555), UINT16_C(    0), UINT16_C(48163), UINT16_C(    0), UINT16_C(    0),
        UINT16_C(34697), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(60333), UINT16_C(27459),
        UINT16_C(    0), UINT16_C(53005), UINT16_C(    0), UINT16_C(20402), UINT16_C(    0), UINT16_C(18918), UINT16_C(49603), UINT16_C(    0),
        UINT16_C(16165), UINT16_C(    0), UINT16_C(30680), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    const easysimd__mmask32 k  = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_epu16(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_epu16");
    easysimd_test_x86_assert_equal_u16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_max_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(    58487515),  INT32_C(  1267763067), -INT32_C(   198522948),  INT32_C(  1686761573), -INT32_C(   964970336), -INT32_C(   792797143), -INT32_C(   195002861), -INT32_C(  1960123547),
        -INT32_C(   343429776), -INT32_C(   248047820),  INT32_C(  1155883487), -INT32_C(   240554159),  INT32_C(  1354179623),  INT32_C(   404780548),  INT32_C(  1510768884), -INT32_C(   723175836) },
      {  INT32_C(   415263972),  INT32_C(  1661597572), -INT32_C(  1448612008), -INT32_C(  2053418914),  INT32_C(  2044023412), -INT32_C(  1114507576), -INT32_C(   635986570), -INT32_C(  1213203245),
        -INT32_C(   321884312), -INT32_C(  1102063258),  INT32_C(   644347848), -INT32_C(  1129643449),  INT32_C(   490045781), -INT32_C(   304429449),  INT32_C(   919138659),  INT32_C(  1458468845) },
      {  INT32_C(   415263972),  INT32_C(  1661597572), -INT32_C(   198522948),  INT32_C(  1686761573),  INT32_C(  2044023412), -INT32_C(   792797143), -INT32_C(   195002861), -INT32_C(  1213203245),
        -INT32_C(   321884312), -INT32_C(   248047820),  INT32_C(  1155883487), -INT32_C(   240554159),  INT32_C(  1354179623),  INT32_C(   404780548),  INT32_C(  1510768884),  INT32_C(  1458468845) } },
    { {  INT32_C(  1279442662),  INT32_C(  1611305623), -INT32_C(   796495479), -INT32_C(   913558924),  INT32_C(   719765939), -INT32_C(   367541881), -INT32_C(  1608392782),  INT32_C(  1022758742),
         INT32_C(  1686649037),  INT32_C(  1405391562),  INT32_C(  2015644420),  INT32_C(   809611389), -INT32_C(   111532174), -INT32_C(  1679527448), -INT32_C(  1489304239), -INT32_C(   505138924) },
      {  INT32_C(   876964969),  INT32_C(    59181823), -INT32_C(   763647147), -INT32_C(   838681508), -INT32_C(   859349789),  INT32_C(   510110669), -INT32_C(   993615184),  INT32_C(  1017490131),
         INT32_C(   359721750),  INT32_C(  1243150581), -INT32_C(    14904413),  INT32_C(   869080655),  INT32_C(  1207932282), -INT32_C(   244947392), -INT32_C(   608883704), -INT32_C(   334013482) },
      {  INT32_C(  1279442662),  INT32_C(  1611305623), -INT32_C(   763647147), -INT32_C(   838681508),  INT32_C(   719765939),  INT32_C(   510110669), -INT32_C(   993615184),  INT32_C(  1022758742),
         INT32_C(  1686649037),  INT32_C(  1405391562),  INT32_C(  2015644420),  INT32_C(   869080655),  INT32_C(  1207932282), -INT32_C(   244947392), -INT32_C(   608883704), -INT32_C(   334013482) } },
    { {  INT32_C(   990021702),  INT32_C(   595925632), -INT32_C(    47996498),  INT32_C(   959508671), -INT32_C(   964677755), -INT32_C(  1648892267), -INT32_C(   394761198),  INT32_C(   232100039),
        -INT32_C(  1740056808), -INT32_C(  1615081999),  INT32_C(   765320814),  INT32_C(  1416023503), -INT32_C(  1843730435), -INT32_C(   533671475),  INT32_C(    97036350),  INT32_C(  1343462712) },
      {  INT32_C(  1726503796), -INT32_C(  1761237975),  INT32_C(  1371906690),  INT32_C(  1839606640), -INT32_C(   520110062), -INT32_C(   792711278),  INT32_C(   282429656), -INT32_C(  1704859610),
         INT32_C(  1828735300),  INT32_C(  1879312109),  INT32_C(   415353256),  INT32_C(   126183413), -INT32_C(  1159232216), -INT32_C(  1937070156),  INT32_C(  1453154096), -INT32_C(  1930363320) },
      {  INT32_C(  1726503796),  INT32_C(   595925632),  INT32_C(  1371906690),  INT32_C(  1839606640), -INT32_C(   520110062), -INT32_C(   792711278),  INT32_C(   282429656),  INT32_C(   232100039),
         INT32_C(  1828735300),  INT32_C(  1879312109),  INT32_C(   765320814),  INT32_C(  1416023503), -INT32_C(  1159232216), -INT32_C(   533671475),  INT32_C(  1453154096),  INT32_C(  1343462712) } },
    { {  INT32_C(   905572679), -INT32_C(  1616511497), -INT32_C(  1128765753), -INT32_C(   154976818),  INT32_C(  2008067010), -INT32_C(  2113717678), -INT32_C(   505896807), -INT32_C(   429012578),
        -INT32_C(  1323604294),  INT32_C(   726712420), -INT32_C(   186185690),  INT32_C(   149596742), -INT32_C(  1468032427),  INT32_C(  1848280020), -INT32_C(  1035009245), -INT32_C(  2035761716) },
      { -INT32_C(  2026388701), -INT32_C(  1447917693), -INT32_C(   694249072), -INT32_C(  1713469372), -INT32_C(   146711005),  INT32_C(    73755873),  INT32_C(  1002878319), -INT32_C(  1782485390),
        -INT32_C(  1273104335),  INT32_C(   257871743), -INT32_C(  1377436567), -INT32_C(  1488534396),  INT32_C(    60786722),  INT32_C(  1661404404),  INT32_C(   731827897),  INT32_C(  1858166588) },
      {  INT32_C(   905572679), -INT32_C(  1447917693), -INT32_C(   694249072), -INT32_C(   154976818),  INT32_C(  2008067010),  INT32_C(    73755873),  INT32_C(  1002878319), -INT32_C(   429012578),
        -INT32_C(  1273104335),  INT32_C(   726712420), -INT32_C(   186185690),  INT32_C(   149596742),  INT32_C(    60786722),  INT32_C(  1848280020),  INT32_C(   731827897),  INT32_C(  1858166588) } },
    { { -INT32_C(   702357929),  INT32_C(   384204973),  INT32_C(    29608828), -INT32_C(  1314387313), -INT32_C(  2035005550),  INT32_C(    99204172), -INT32_C(   969832566),  INT32_C(  1026880230),
         INT32_C(  2098419664),  INT32_C(  1419049431),  INT32_C(  1414879173), -INT32_C(   217645727), -INT32_C(  1854293435),  INT32_C(     9855606), -INT32_C(   808990743), -INT32_C(  1995637831) },
      {  INT32_C(   705110098), -INT32_C(   562128103),  INT32_C(  1412682738),  INT32_C(   356989392),  INT32_C(  1789313523),  INT32_C(   225066275), -INT32_C(  1092865788),  INT32_C(  2135419181),
         INT32_C(   581520905), -INT32_C(   603904023), -INT32_C(   886033158),  INT32_C(  1625323373),  INT32_C(  1556776760), -INT32_C(   932629052), -INT32_C(  1819916954),  INT32_C(   924044846) },
      {  INT32_C(   705110098),  INT32_C(   384204973),  INT32_C(  1412682738),  INT32_C(   356989392),  INT32_C(  1789313523),  INT32_C(   225066275), -INT32_C(   969832566),  INT32_C(  2135419181),
         INT32_C(  2098419664),  INT32_C(  1419049431),  INT32_C(  1414879173),  INT32_C(  1625323373),  INT32_C(  1556776760),  INT32_C(     9855606), -INT32_C(   808990743),  INT32_C(   924044846) } },
    { {  INT32_C(   106609692), -INT32_C(   555590684), -INT32_C(    56028529), -INT32_C(  1034122615), -INT32_C(   719444207), -INT32_C(  1029863588), -INT32_C(    78240564),  INT32_C(   238184946),
         INT32_C(   152341541),  INT32_C(  1994979047), -INT32_C(  1837985528),  INT32_C(   743755547),  INT32_C(  1375826678), -INT32_C(   988504071), -INT32_C(  1245680957), -INT32_C(   104598573) },
      {  INT32_C(  1728239743), -INT32_C(   673322290), -INT32_C(  1754705796),  INT32_C(   365215007),  INT32_C(   677889327),  INT32_C(   669875044), -INT32_C(  1176719642),  INT32_C(   548577441),
         INT32_C(  1183298936),  INT32_C(   454911391), -INT32_C(   726432075),  INT32_C(  1927903043), -INT32_C(  1583722436), -INT32_C(  1312257845), -INT32_C(   680811210), -INT32_C(  1107878587) },
      {  INT32_C(  1728239743), -INT32_C(   555590684), -INT32_C(    56028529),  INT32_C(   365215007),  INT32_C(   677889327),  INT32_C(   669875044), -INT32_C(    78240564),  INT32_C(   548577441),
         INT32_C(  1183298936),  INT32_C(  1994979047), -INT32_C(   726432075),  INT32_C(  1927903043),  INT32_C(  1375826678), -INT32_C(   988504071), -INT32_C(   680811210), -INT32_C(   104598573) } },
    { {  INT32_C(  1912831954), -INT32_C(  1718803996), -INT32_C(   345161561), -INT32_C(   195209545),  INT32_C(  1905653926), -INT32_C(  1239196288),  INT32_C(  1200459266),  INT32_C(  2114225323),
        -INT32_C(   403699709), -INT32_C(   796885719),  INT32_C(  1975250366), -INT32_C(   378988221), -INT32_C(  1856242159),  INT32_C(  1581743708), -INT32_C(  1213803508),  INT32_C(  1547020888) },
      { -INT32_C(   616356430),  INT32_C(  1638712483), -INT32_C(   170498127), -INT32_C(  1847705472), -INT32_C(  1709033154), -INT32_C(  1007064649), -INT32_C(  1770283203), -INT32_C(    51204023),
         INT32_C(  2044147158), -INT32_C(  1411742727),  INT32_C(  1805693163),  INT32_C(   805142256),  INT32_C(  1875451832), -INT32_C(   969686391), -INT32_C(  1419989407),  INT32_C(   883379806) },
      {  INT32_C(  1912831954),  INT32_C(  1638712483), -INT32_C(   170498127), -INT32_C(   195209545),  INT32_C(  1905653926), -INT32_C(  1007064649),  INT32_C(  1200459266),  INT32_C(  2114225323),
         INT32_C(  2044147158), -INT32_C(   796885719),  INT32_C(  1975250366),  INT32_C(   805142256),  INT32_C(  1875451832),  INT32_C(  1581743708), -INT32_C(  1213803508),  INT32_C(  1547020888) } },
    { {  INT32_C(  2108522116), -INT32_C(   316111102),  INT32_C(   676907064), -INT32_C(    11053753), -INT32_C(    26336907),  INT32_C(  1170514403), -INT32_C(  1359994545), -INT32_C(   203253905),
         INT32_C(   393318421),  INT32_C(  1325701399), -INT32_C(  1451729566),  INT32_C(   665374642), -INT32_C(   735766800),  INT32_C(   119139000),  INT32_C(  2058684683), -INT32_C(  1251043168) },
      {  INT32_C(  1070456616), -INT32_C(   628108936), -INT32_C(   511506642), -INT32_C(   955765802), -INT32_C(    90493374),  INT32_C(   587314200),  INT32_C(  1570617277),  INT32_C(  1997671247),
         INT32_C(  1672929258), -INT32_C(   549632591),  INT32_C(   599834956),  INT32_C(   787139052),  INT32_C(   254313975), -INT32_C(   164484551),  INT32_C(   810799073), -INT32_C(   978885157) },
      {  INT32_C(  2108522116), -INT32_C(   316111102),  INT32_C(   676907064), -INT32_C(    11053753), -INT32_C(    26336907),  INT32_C(  1170514403),  INT32_C(  1570617277),  INT32_C(  1997671247),
         INT32_C(  1672929258),  INT32_C(  1325701399),  INT32_C(   599834956),  INT32_C(   787139052),  INT32_C(   254313975),  INT32_C(   119139000),  INT32_C(  2058684683), -INT32_C(   978885157) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1607096062), -INT32_C(   163478492),  INT32_C(  1326370578), -INT32_C(  1482691356),  INT32_C(    36315194),  INT32_C(   459690145),  INT32_C(   441074651),  INT32_C(   373429012),
        -INT32_C(   524912708),  INT32_C(   249035003), -INT32_C(  1403132473),  INT32_C(   206830802), -INT32_C(  1123123940), -INT32_C(  1462143795), -INT32_C(   876469321), -INT32_C(   203357130) },
      UINT16_C(39035),
      { -INT32_C(  1433372717), -INT32_C(   493856891),  INT32_C(  1457480195),  INT32_C(  2111044462), -INT32_C(  1829658439),  INT32_C(   196454728),  INT32_C(  1443884148), -INT32_C(  1292989473),
        -INT32_C(  2023916030), -INT32_C(   647369259), -INT32_C(  1120974769), -INT32_C(    46529724), -INT32_C(   326161245),  INT32_C(  1240941781),  INT32_C(   262080048), -INT32_C(  1815966319) },
      { -INT32_C(   518381813),  INT32_C(  1538950156),  INT32_C(   270068172), -INT32_C(  1878174740),  INT32_C(  1417452671),  INT32_C(   295531489),  INT32_C(   186727801), -INT32_C(   694230070),
         INT32_C(   246921474),  INT32_C(   157905213),  INT32_C(  1192919386),  INT32_C(  1406609364), -INT32_C(  1515760700),  INT32_C(  1068910022),  INT32_C(  1279973250), -INT32_C(  1138562630) },
      { -INT32_C(   518381813),  INT32_C(  1538950156),  INT32_C(  1326370578),  INT32_C(  2111044462),  INT32_C(  1417452671),  INT32_C(   295531489),  INT32_C(  1443884148),  INT32_C(   373429012),
        -INT32_C(   524912708),  INT32_C(   249035003), -INT32_C(  1403132473),  INT32_C(  1406609364), -INT32_C(   326161245), -INT32_C(  1462143795), -INT32_C(   876469321), -INT32_C(  1138562630) } },
    { { -INT32_C(   540354142), -INT32_C(  1511509174), -INT32_C(  1981021515), -INT32_C(   287456470), -INT32_C(   594312170),  INT32_C(  1260079561), -INT32_C(   610769375), -INT32_C(   241649073),
        -INT32_C(   556768620),  INT32_C(  1266923670), -INT32_C(   439062597),  INT32_C(  1221833010), -INT32_C(    14391754), -INT32_C(   800374609), -INT32_C(   173218906),  INT32_C(   837174429) },
      UINT16_C(46758),
      { -INT32_C(  1821492208),  INT32_C(  1543711111), -INT32_C(   519228146), -INT32_C(  1572387970), -INT32_C(  1897728189),  INT32_C(  1936820423),  INT32_C(  1672941181),  INT32_C(  1343839808),
         INT32_C(   585336986), -INT32_C(  1065425231), -INT32_C(  1700689124),  INT32_C(   305981903), -INT32_C(  1482678304),  INT32_C(   622530983),  INT32_C(  1602802207), -INT32_C(   877682128) },
      { -INT32_C(   605187286), -INT32_C(  1784976519), -INT32_C(   986759690),  INT32_C(   114781222),  INT32_C(   850229131), -INT32_C(  1470642040), -INT32_C(   888676197), -INT32_C(  1399408766),
        -INT32_C(  1014529206), -INT32_C(   447142930), -INT32_C(  2035644320),  INT32_C(  2139914996), -INT32_C(  2102248967), -INT32_C(  1658189566),  INT32_C(  1801990633),  INT32_C(   840433640) },
      { -INT32_C(   540354142),  INT32_C(  1543711111), -INT32_C(   519228146), -INT32_C(   287456470), -INT32_C(   594312170),  INT32_C(  1936820423), -INT32_C(   610769375),  INT32_C(  1343839808),
        -INT32_C(   556768620), -INT32_C(   447142930), -INT32_C(  1700689124),  INT32_C(  1221833010), -INT32_C(  1482678304),  INT32_C(   622530983), -INT32_C(   173218906),  INT32_C(   840433640) } },
    { {  INT32_C(  1911988098),  INT32_C(   576081858), -INT32_C(   861404969),  INT32_C(  2085303426),  INT32_C(  1878981997), -INT32_C(   267638777),  INT32_C(  1113355609), -INT32_C(   160140428),
        -INT32_C(   731420142), -INT32_C(  1846100551),  INT32_C(  1079877310),  INT32_C(  1086105810), -INT32_C(  1380992346),  INT32_C(  1016970466), -INT32_C(  1518405327),  INT32_C(  2140926573) },
      UINT16_C(  861),
      {  INT32_C(  1254102612),  INT32_C(    82411175),  INT32_C(  2074983359), -INT32_C(  1422502917), -INT32_C(  1654188032), -INT32_C(   761817004), -INT32_C(   658176963), -INT32_C(   673504637),
        -INT32_C(   534602696), -INT32_C(   639366374), -INT32_C(  1034579514), -INT32_C(   462582812), -INT32_C(   125643613), -INT32_C(  1446373012), -INT32_C(  1602121955), -INT32_C(   361210447) },
      {  INT32_C(   332045049),  INT32_C(  1810738853),  INT32_C(   606945856),  INT32_C(  1879677645), -INT32_C(   630682770), -INT32_C(  1048366172),  INT32_C(  1952515522),  INT32_C(  1532942690),
         INT32_C(   409872499),  INT32_C(   377773014),  INT32_C(  1782296989), -INT32_C(  1160035252),  INT32_C(  1939162063),  INT32_C(   959715446),  INT32_C(  2142082333), -INT32_C(   489026705) },
      {  INT32_C(  1254102612),  INT32_C(   576081858),  INT32_C(  2074983359),  INT32_C(  1879677645), -INT32_C(   630682770), -INT32_C(   267638777),  INT32_C(  1952515522), -INT32_C(   160140428),
         INT32_C(   409872499),  INT32_C(   377773014),  INT32_C(  1079877310),  INT32_C(  1086105810), -INT32_C(  1380992346),  INT32_C(  1016970466), -INT32_C(  1518405327),  INT32_C(  2140926573) } },
    { {  INT32_C(   167463219),  INT32_C(  1109426084),  INT32_C(  2091670320),  INT32_C(  1849132959),  INT32_C(  1105317067),  INT32_C(    41555428),  INT32_C(   427894698),  INT32_C(  1711037490),
         INT32_C(  1232074661), -INT32_C(  1500803210), -INT32_C(  1994180374), -INT32_C(  1963500865),  INT32_C(   181196838), -INT32_C(  1760803091), -INT32_C(  1598976402), -INT32_C(  1895387670) },
      UINT16_C(30116),
      {  INT32_C(  1677990616), -INT32_C(   476254528),  INT32_C(  1849514871), -INT32_C(  1304009754),  INT32_C(  2063086446),  INT32_C(  2064148170),  INT32_C(   220787207),  INT32_C(  1518521473),
        -INT32_C(  1480685850), -INT32_C(   343254412), -INT32_C(  1688614731), -INT32_C(  1722966229), -INT32_C(  1676392750), -INT32_C(  1290265428), -INT32_C(  1866448881), -INT32_C(   202751475) },
      {  INT32_C(  1016768712), -INT32_C(  1205394174),  INT32_C(   408125677), -INT32_C(   239951585), -INT32_C(  1819359513), -INT32_C(   246962462), -INT32_C(   209582106),  INT32_C(   317156426),
         INT32_C(   391086357), -INT32_C(  1815120218),  INT32_C(   380380151), -INT32_C(  1425514812),  INT32_C(   104764964),  INT32_C(   586712380), -INT32_C(   686392691), -INT32_C(    68551194) },
      {  INT32_C(   167463219),  INT32_C(  1109426084),  INT32_C(  1849514871),  INT32_C(  1849132959),  INT32_C(  1105317067),  INT32_C(  2064148170),  INT32_C(   427894698),  INT32_C(  1518521473),
         INT32_C(   391086357), -INT32_C(  1500803210),  INT32_C(   380380151), -INT32_C(  1963500865),  INT32_C(   104764964),  INT32_C(   586712380), -INT32_C(   686392691), -INT32_C(  1895387670) } },
    { {  INT32_C(   622016638), -INT32_C(  1497832785), -INT32_C(   910400507), -INT32_C(   428555070), -INT32_C(  1762806950), -INT32_C(   977672904),  INT32_C(  1167904607),  INT32_C(  1245808332),
         INT32_C(  1836012734),  INT32_C(  1007888438),  INT32_C(  1325781132), -INT32_C(   281707884),  INT32_C(  1703223853),  INT32_C(  1714109959), -INT32_C(   642988275),  INT32_C(   203746637) },
      UINT16_C(37697),
      { -INT32_C(  1917094023), -INT32_C(  1185068877), -INT32_C(   869011049),  INT32_C(  1726963936), -INT32_C(   257624379),  INT32_C(   163099229), -INT32_C(  1342831221), -INT32_C(  1958529263),
         INT32_C(  1645805230), -INT32_C(   585403066),  INT32_C(  1202343526),  INT32_C(  1940756910), -INT32_C(   328969841), -INT32_C(  1879761917),  INT32_C(   859761441), -INT32_C(   776044254) },
      { -INT32_C(   969681280), -INT32_C(  1314632117),  INT32_C(  1257787036),  INT32_C(  1992140263), -INT32_C(    94166537),  INT32_C(  1602836541),  INT32_C(  1720895556), -INT32_C(   919121847),
         INT32_C(  1905289766),  INT32_C(  1411527864),  INT32_C(  1771969410), -INT32_C(  1210098496), -INT32_C(  1145945475), -INT32_C(   551928933),  INT32_C(  1296411651),  INT32_C(   571899388) },
      { -INT32_C(   969681280), -INT32_C(  1497832785), -INT32_C(   910400507), -INT32_C(   428555070), -INT32_C(  1762806950), -INT32_C(   977672904),  INT32_C(  1720895556),  INT32_C(  1245808332),
         INT32_C(  1905289766),  INT32_C(  1411527864),  INT32_C(  1325781132), -INT32_C(   281707884), -INT32_C(   328969841),  INT32_C(  1714109959), -INT32_C(   642988275),  INT32_C(   571899388) } },
    { { -INT32_C(  1600936217),  INT32_C(  1559541210), -INT32_C(  1849322544),  INT32_C(  1816700399), -INT32_C(  2111309081),  INT32_C(   962674998), -INT32_C(   377051155),  INT32_C(   185310500),
         INT32_C(   514563651),  INT32_C(   612016212),  INT32_C(   582303795), -INT32_C(   863043867),  INT32_C(   776976120), -INT32_C(   446123785),  INT32_C(  2077158999), -INT32_C(   813180277) },
      UINT16_C(12920),
      {  INT32_C(  1741868269), -INT32_C(  1499003407), -INT32_C(  1230730201), -INT32_C(  1469276839),  INT32_C(   861430731),  INT32_C(   388149320), -INT32_C(  1292784341),  INT32_C(  1776642428),
         INT32_C(   668055350), -INT32_C(   456296259),  INT32_C(  1587180037), -INT32_C(   637139441), -INT32_C(  1307681174),  INT32_C(   986263566),  INT32_C(  1525463773),  INT32_C(  1522782500) },
      {  INT32_C(  1182897289),  INT32_C(   304762381), -INT32_C(   814692928),  INT32_C(   900363979), -INT32_C(   471287596), -INT32_C(   987909656), -INT32_C(  1877014164),  INT32_C(  1693115355),
        -INT32_C(  2069206153),  INT32_C(  2056705209),  INT32_C(  1699284633),  INT32_C(  1369109372), -INT32_C(  1825275221), -INT32_C(  1604759244),  INT32_C(   892368986), -INT32_C(   744940965) },
      { -INT32_C(  1600936217),  INT32_C(  1559541210), -INT32_C(  1849322544),  INT32_C(   900363979),  INT32_C(   861430731),  INT32_C(   388149320), -INT32_C(  1292784341),  INT32_C(   185310500),
         INT32_C(   514563651),  INT32_C(  2056705209),  INT32_C(   582303795), -INT32_C(   863043867), -INT32_C(  1307681174),  INT32_C(   986263566),  INT32_C(  2077158999), -INT32_C(   813180277) } },
    { {  INT32_C(  1045906309), -INT32_C(  1313280488),  INT32_C(  1897267956), -INT32_C(  1581075979),  INT32_C(  1731524147), -INT32_C(  1593340601),  INT32_C(  1641494278), -INT32_C(   701206447),
        -INT32_C(   871002956),  INT32_C(  1853738362), -INT32_C(   975203121),  INT32_C(  2019991877), -INT32_C(   555705705),  INT32_C(   780199720),  INT32_C(  1888442143),  INT32_C(  2068300999) },
      UINT16_C(23632),
      { -INT32_C(   987116985),  INT32_C(   408549688),  INT32_C(   616144574), -INT32_C(   155299562), -INT32_C(  1344346577), -INT32_C(  1543045868), -INT32_C(  1268199827), -INT32_C(  1861175223),
        -INT32_C(  1168754046), -INT32_C(   237850829),  INT32_C(  1662356557),  INT32_C(   207279069), -INT32_C(   826525510), -INT32_C(  1569537483), -INT32_C(   631776624),  INT32_C(   342583186) },
      { -INT32_C(   724581983), -INT32_C(  1111121552),  INT32_C(   169925165),  INT32_C(  1746369198), -INT32_C(   415837262),  INT32_C(   646621589),  INT32_C(   369156483), -INT32_C(   366318776),
        -INT32_C(  1665205972), -INT32_C(   933657445),  INT32_C(   215185758), -INT32_C(  1502287116),  INT32_C(  1385081789), -INT32_C(   679995308),  INT32_C(  1106082041),  INT32_C(   254482659) },
      {  INT32_C(  1045906309), -INT32_C(  1313280488),  INT32_C(  1897267956), -INT32_C(  1581075979), -INT32_C(   415837262), -INT32_C(  1593340601),  INT32_C(   369156483), -INT32_C(   701206447),
        -INT32_C(   871002956),  INT32_C(  1853738362),  INT32_C(  1662356557),  INT32_C(   207279069),  INT32_C(  1385081789),  INT32_C(   780199720),  INT32_C(  1106082041),  INT32_C(  2068300999) } },
    { { -INT32_C(  1398019567), -INT32_C(   864746386),  INT32_C(  1926842494), -INT32_C(   283620046),  INT32_C(  1279371000), -INT32_C(  1222329666),  INT32_C(   385421618),  INT32_C(   992289833),
         INT32_C(  2095567118),  INT32_C(  1397316821), -INT32_C(   691723612),  INT32_C(  1757797999),  INT32_C(  1135871876), -INT32_C(   201664319),  INT32_C(   319419370),  INT32_C(   642658072) },
      UINT16_C(13824),
      { -INT32_C(   325921373), -INT32_C(   301058263), -INT32_C(   741573363), -INT32_C(  1696968219), -INT32_C(  1905091692), -INT32_C(  1736287090), -INT32_C(  1094215056),  INT32_C(  1676986304),
        -INT32_C(   984643684), -INT32_C(   877371970), -INT32_C(  1063354149),  INT32_C(  1700427985), -INT32_C(  1561015021), -INT32_C(  1724221911), -INT32_C(   799538928),  INT32_C(  1681149128) },
      { -INT32_C(  1859484717), -INT32_C(  1134698783),  INT32_C(   813497182), -INT32_C(  2020223116), -INT32_C(   852915804),  INT32_C(   258434047),  INT32_C(   752926564), -INT32_C(   544140277),
         INT32_C(  2020653975), -INT32_C(   147534439),  INT32_C(  1026011593),  INT32_C(   751091080),  INT32_C(  1190784582), -INT32_C(  1235918767),  INT32_C(   736245023), -INT32_C(   519408823) },
      { -INT32_C(  1398019567), -INT32_C(   864746386),  INT32_C(  1926842494), -INT32_C(   283620046),  INT32_C(  1279371000), -INT32_C(  1222329666),  INT32_C(   385421618),  INT32_C(   992289833),
         INT32_C(  2095567118), -INT32_C(   147534439),  INT32_C(  1026011593),  INT32_C(  1757797999),  INT32_C(  1190784582), -INT32_C(  1235918767),  INT32_C(   319419370),  INT32_C(   642658072) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(46951),
      {  INT32_C(  1840390330), -INT32_C(   963235308), -INT32_C(   839300271), -INT32_C(  1164858157), -INT32_C(   133013581), -INT32_C(  1111663537), -INT32_C(    38173341),  INT32_C(  1907695798),
         INT32_C(  1423861568), -INT32_C(   350521958), -INT32_C(  1816652608),  INT32_C(  1934510528), -INT32_C(   143957848),  INT32_C(   548677820),  INT32_C(  1679650477), -INT32_C(   808070514) },
      { -INT32_C(   735857862), -INT32_C(   390119896),  INT32_C(   326924370),  INT32_C(  1787218625), -INT32_C(   413011414), -INT32_C(   939059686),  INT32_C(   304882820),  INT32_C(   836829687),
        -INT32_C(   586873420), -INT32_C(  1765424061), -INT32_C(    22462148),  INT32_C(   912797451),  INT32_C(  1008584993),  INT32_C(  1661215967),  INT32_C(  1064710216), -INT32_C(   445622479) },
      {  INT32_C(  1840390330), -INT32_C(   390119896),  INT32_C(   326924370),  INT32_C(           0),  INT32_C(           0), -INT32_C(   939059686),  INT32_C(   304882820),  INT32_C(           0),
         INT32_C(  1423861568), -INT32_C(   350521958), -INT32_C(    22462148),  INT32_C(           0),  INT32_C(  1008584993),  INT32_C(  1661215967),  INT32_C(           0), -INT32_C(   445622479) } },
    { UINT16_C(30044),
      { -INT32_C(  2009423678), -INT32_C(   540445130), -INT32_C(   603007628),  INT32_C(   681979915),  INT32_C(  1884063084),  INT32_C(  1604359401),  INT32_C(  1152831956),  INT32_C(  2042237878),
        -INT32_C(   385747789), -INT32_C(   540489110), -INT32_C(  1430530401),  INT32_C(  1926390022), -INT32_C(   790487321), -INT32_C(  2026929485),  INT32_C(   181134675), -INT32_C(  1417443848) },
      { -INT32_C(   460028807), -INT32_C(   289186738),  INT32_C(   966295091), -INT32_C(   945001504),  INT32_C(  1016565385),  INT32_C(  1690551825), -INT32_C(  1536258133), -INT32_C(  1907363564),
        -INT32_C(   999103371),  INT32_C(  1941058880), -INT32_C(  1817359693),  INT32_C(  1062885813), -INT32_C(   126094873),  INT32_C(  1667055543), -INT32_C(   502805554),  INT32_C(   846223037) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   966295091),  INT32_C(   681979915),  INT32_C(  1884063084),  INT32_C(           0),  INT32_C(  1152831956),  INT32_C(           0),
        -INT32_C(   385747789),  INT32_C(           0), -INT32_C(  1430530401),  INT32_C(           0), -INT32_C(   126094873),  INT32_C(  1667055543),  INT32_C(   181134675),  INT32_C(           0) } },
    { UINT16_C(57914),
      { -INT32_C(  1474855946), -INT32_C(  1678521362), -INT32_C(  1175148450),  INT32_C(  1672142055),  INT32_C(   832725716), -INT32_C(   855805755), -INT32_C(  1021134254), -INT32_C(   475701780),
        -INT32_C(   963920424), -INT32_C(   429752696),  INT32_C(   245323303),  INT32_C(   124865074),  INT32_C(  1899500460), -INT32_C(   700631677), -INT32_C(   593928209), -INT32_C(  1799405892) },
      { -INT32_C(  2091169029), -INT32_C(   261440055),  INT32_C(  1191053587), -INT32_C(    11702189),  INT32_C(   124814723), -INT32_C(  1428312645), -INT32_C(   913934835), -INT32_C(  1335999052),
         INT32_C(  1496562064), -INT32_C(  1991664266), -INT32_C(    87079001),  INT32_C(  1006247095),  INT32_C(  1564633762),  INT32_C(   621223704),  INT32_C(  1240370837),  INT32_C(  1677282515) },
      {  INT32_C(           0), -INT32_C(   261440055),  INT32_C(           0),  INT32_C(  1672142055),  INT32_C(   832725716), -INT32_C(   855805755),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0), -INT32_C(   429752696),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   621223704),  INT32_C(  1240370837),  INT32_C(  1677282515) } },
    { UINT16_C(11525),
      {  INT32_C(    97156028), -INT32_C(   733122299),  INT32_C(  1727071340), -INT32_C(  2117037249), -INT32_C(   140449552), -INT32_C(    58378995),  INT32_C(  2018007423),  INT32_C(  2040876732),
        -INT32_C(   830574391),  INT32_C(  1302580193),  INT32_C(   263427280),  INT32_C(   395412519),  INT32_C(  2047750508), -INT32_C(   428436377), -INT32_C(  1453408531), -INT32_C(   702413812) },
      {  INT32_C(  1436852596),  INT32_C(  1017333612),  INT32_C(    38557403),  INT32_C(  1192877530),  INT32_C(  1975592974), -INT32_C(  1453639748),  INT32_C(    38976245),  INT32_C(   853046718),
        -INT32_C(  2105050090), -INT32_C(  1614861628),  INT32_C(  1537346433), -INT32_C(   157107224), -INT32_C(  1620286493), -INT32_C(  1874278502),  INT32_C(  1066572673),  INT32_C(   644966928) },
      {  INT32_C(  1436852596),  INT32_C(           0),  INT32_C(  1727071340),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(   830574391),  INT32_C(           0),  INT32_C(  1537346433),  INT32_C(   395412519),  INT32_C(           0), -INT32_C(   428436377),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(63974),
      {  INT32_C(  1730390696), -INT32_C(   328031158), -INT32_C(  1566090752), -INT32_C(  1123644591),  INT32_C(  1938071594), -INT32_C(  1039268304), -INT32_C(  1221845435), -INT32_C(   324005052),
         INT32_C(   122999741),  INT32_C(  2029241976),  INT32_C(  1914346273),  INT32_C(  1345265702), -INT32_C(   238832703), -INT32_C(     4927047),  INT32_C(   867623151), -INT32_C(  1323276557) },
      { -INT32_C(  1313311687),  INT32_C(  1529457722),  INT32_C(  1842168903),  INT32_C(   633207908),  INT32_C(  1763148208),  INT32_C(  1114164050), -INT32_C(  1619714389), -INT32_C(  1102015100),
         INT32_C(  1148127241), -INT32_C(    73426508),  INT32_C(  1097362909),  INT32_C(   426190441), -INT32_C(   108822873), -INT32_C(   197399735), -INT32_C(  1902923510),  INT32_C(  1347216198) },
      {  INT32_C(           0),  INT32_C(  1529457722),  INT32_C(  1842168903),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1114164050), -INT32_C(  1221845435), -INT32_C(   324005052),
         INT32_C(  1148127241),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1345265702), -INT32_C(   108822873), -INT32_C(     4927047),  INT32_C(   867623151),  INT32_C(  1347216198) } },
    { UINT16_C(48364),
      {  INT32_C(   861249684),  INT32_C(    77607580), -INT32_C(   634779021), -INT32_C(  1504128733),  INT32_C(   110272971),  INT32_C(   699899030),  INT32_C(  1997405738), -INT32_C(   499910322),
         INT32_C(   890603673),  INT32_C(   758822586), -INT32_C(   485989184),  INT32_C(    25845814), -INT32_C(  1744364542), -INT32_C(   490618952),  INT32_C(   190435005),  INT32_C(  1642958023) },
      { -INT32_C(   829029868),  INT32_C(  2029834424),  INT32_C(  1801192501),  INT32_C(  1718412900), -INT32_C(  1157729534), -INT32_C(   274939854),  INT32_C(  1459287694), -INT32_C(  1749555326),
        -INT32_C(  1570419222),  INT32_C(  1394303262), -INT32_C(   893487259),  INT32_C(  1596992093), -INT32_C(   803655779),  INT32_C(  2109715951),  INT32_C(   785627819), -INT32_C(  1949988191) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1801192501),  INT32_C(  1718412900),  INT32_C(           0),  INT32_C(   699899030),  INT32_C(  1997405738), -INT32_C(   499910322),
         INT32_C(           0),  INT32_C(           0), -INT32_C(   485989184),  INT32_C(  1596992093), -INT32_C(   803655779),  INT32_C(  2109715951),  INT32_C(           0),  INT32_C(  1642958023) } },
    { UINT16_C(10968),
      {  INT32_C(  1233974830),  INT32_C(   130085193), -INT32_C(   332325445), -INT32_C(  1793339780),  INT32_C(  1581910686),  INT32_C(  1528362631), -INT32_C(   370820828),  INT32_C(  1930673477),
        -INT32_C(    54747213),  INT32_C(  1275296913),  INT32_C(   356005017), -INT32_C(  1582673149), -INT32_C(   436210595),  INT32_C(   239081450),  INT32_C(   385295825), -INT32_C(  1769403421) },
      {  INT32_C(   999507370),  INT32_C(  1518900929), -INT32_C(   831536949),  INT32_C(  1903106324), -INT32_C(   128553203),  INT32_C(  1460049542), -INT32_C(  1620181316), -INT32_C(  1288309239),
        -INT32_C(    51394501),  INT32_C(   710309727),  INT32_C(  1274594615), -INT32_C(   323131426),  INT32_C(  1575228374),  INT32_C(  1723132586), -INT32_C(   251321624),  INT32_C(  1420114456) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1903106324),  INT32_C(  1581910686),  INT32_C(           0), -INT32_C(   370820828),  INT32_C(  1930673477),
         INT32_C(           0),  INT32_C(  1275296913),  INT32_C(           0), -INT32_C(   323131426),  INT32_C(           0),  INT32_C(  1723132586),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(37891),
      { -INT32_C(  1492426160), -INT32_C(  2073279860),  INT32_C(  1273711502),  INT32_C(   459194935), -INT32_C(   754644961),  INT32_C(  1945497198), -INT32_C(  2068967713), -INT32_C(  1307004574),
        -INT32_C(  1621548269), -INT32_C(   198982042),  INT32_C(  1212091921), -INT32_C(   278684208),  INT32_C(   348350630),  INT32_C(   914929750),  INT32_C(   683292358), -INT32_C(    86256665) },
      {  INT32_C(  1553544438),  INT32_C(   189840634), -INT32_C(  1689022518), -INT32_C(   745884115),  INT32_C(  1978092831), -INT32_C(   861180154), -INT32_C(  1930074459),  INT32_C(   797364281),
        -INT32_C(     7594236), -INT32_C(  1509237541), -INT32_C(  1723769236),  INT32_C(   862767892),  INT32_C(   531190553),  INT32_C(  1760253123), -INT32_C(   201989958), -INT32_C(  1255965776) },
      {  INT32_C(  1553544438),  INT32_C(   189840634),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(  1212091921),  INT32_C(           0),  INT32_C(   531190553),  INT32_C(           0),  INT32_C(           0), -INT32_C(    86256665) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_max_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint32_t a[16];
    const uint32_t b[16];
    const uint32_t r[16];
  } test_vec[] = {
    { { UINT32_C(2243118331), UINT32_C(1578755567), UINT32_C(1899903292), UINT32_C(3548006149), UINT32_C(1967957742), UINT32_C(3104828962), UINT32_C(3870576571), UINT32_C(3545536983),
        UINT32_C(3428321501), UINT32_C( 690647533), UINT32_C(2946197674), UINT32_C(2659390895), UINT32_C(3725840315), UINT32_C(2023170749), UINT32_C(1214204785), UINT32_C(3239818212) },
      { UINT32_C(2827842747), UINT32_C(2429728997), UINT32_C(3493817632), UINT32_C(1047446146), UINT32_C(1327268242), UINT32_C( 348697507), UINT32_C(3797690366), UINT32_C(2527295706),
        UINT32_C(3510513900), UINT32_C( 157356264), UINT32_C(  14262653), UINT32_C(4114499427), UINT32_C(1799707336), UINT32_C( 192875789), UINT32_C( 250469172), UINT32_C(1067749459) },
      { UINT32_C(2827842747), UINT32_C(2429728997), UINT32_C(3493817632), UINT32_C(3548006149), UINT32_C(1967957742), UINT32_C(3104828962), UINT32_C(3870576571), UINT32_C(3545536983),
        UINT32_C(3510513900), UINT32_C( 690647533), UINT32_C(2946197674), UINT32_C(4114499427), UINT32_C(3725840315), UINT32_C(2023170749), UINT32_C(1214204785), UINT32_C(3239818212) } },
    { { UINT32_C(2836521920), UINT32_C(1907520243), UINT32_C(2003929875), UINT32_C(2590814162), UINT32_C( 369471753), UINT32_C(4062282942), UINT32_C(3003190879), UINT32_C(1592960414),
        UINT32_C(2080834440), UINT32_C(2314058102), UINT32_C( 369122884), UINT32_C( 380661005), UINT32_C(3710694686), UINT32_C(2580499769), UINT32_C(4182560859), UINT32_C(4267130486) },
      { UINT32_C(3095027522), UINT32_C(1564567320), UINT32_C(3530769093), UINT32_C(3454543023), UINT32_C( 329913561), UINT32_C(3165420129), UINT32_C(3233151050), UINT32_C(2025786678),
        UINT32_C(2217752940), UINT32_C(1726050977), UINT32_C(1664701876), UINT32_C(1378886009), UINT32_C(2540034870), UINT32_C(2673086805), UINT32_C(1080035593), UINT32_C(2193104662) },
      { UINT32_C(3095027522), UINT32_C(1907520243), UINT32_C(3530769093), UINT32_C(3454543023), UINT32_C( 369471753), UINT32_C(4062282942), UINT32_C(3233151050), UINT32_C(2025786678),
        UINT32_C(2217752940), UINT32_C(2314058102), UINT32_C(1664701876), UINT32_C(1378886009), UINT32_C(3710694686), UINT32_C(2673086805), UINT32_C(4182560859), UINT32_C(4267130486) } },
    { { UINT32_C(4178045272), UINT32_C( 258009179), UINT32_C(3060963645), UINT32_C(4027163322), UINT32_C(3532156541), UINT32_C(2306006144), UINT32_C(4241085157), UINT32_C(1233027825),
        UINT32_C(3326313835), UINT32_C(2882904942), UINT32_C(4133635900), UINT32_C(1743219689), UINT32_C(1496936409), UINT32_C( 820226891), UINT32_C(1848421501), UINT32_C(2579016494) },
      { UINT32_C(2657090352), UINT32_C(3662296222), UINT32_C(1708174459), UINT32_C(4039948055), UINT32_C(1900676390), UINT32_C( 782380465), UINT32_C( 144559833), UINT32_C(2862699897),
        UINT32_C(3997696336), UINT32_C(2982711861), UINT32_C(1427544126), UINT32_C(1984356944), UINT32_C(2565378279), UINT32_C(2529659581), UINT32_C(3533595736), UINT32_C( 159137977) },
      { UINT32_C(4178045272), UINT32_C(3662296222), UINT32_C(3060963645), UINT32_C(4039948055), UINT32_C(3532156541), UINT32_C(2306006144), UINT32_C(4241085157), UINT32_C(2862699897),
        UINT32_C(3997696336), UINT32_C(2982711861), UINT32_C(4133635900), UINT32_C(1984356944), UINT32_C(2565378279), UINT32_C(2529659581), UINT32_C(3533595736), UINT32_C(2579016494) } },
    { { UINT32_C(1995949121), UINT32_C(2502410071), UINT32_C(2817211735), UINT32_C( 119419167), UINT32_C(2124351169), UINT32_C(3893651088), UINT32_C(2210051018), UINT32_C( 881604339),
        UINT32_C(1386906619), UINT32_C(2598883906), UINT32_C( 792842767), UINT32_C(3291897603), UINT32_C(4114797925), UINT32_C( 115234620), UINT32_C(4253718538), UINT32_C(3392214735) },
      { UINT32_C(3692878746), UINT32_C(3178628013), UINT32_C(3656169686), UINT32_C(2107515415), UINT32_C( 863166711), UINT32_C(1094340663), UINT32_C(3091121385), UINT32_C(1954705370),
        UINT32_C(4166098507), UINT32_C(2008401825), UINT32_C(2538709375), UINT32_C(3138711491), UINT32_C( 133072591), UINT32_C(3225954519), UINT32_C(3346565100), UINT32_C(1094449910) },
      { UINT32_C(3692878746), UINT32_C(3178628013), UINT32_C(3656169686), UINT32_C(2107515415), UINT32_C(2124351169), UINT32_C(3893651088), UINT32_C(3091121385), UINT32_C(1954705370),
        UINT32_C(4166098507), UINT32_C(2598883906), UINT32_C(2538709375), UINT32_C(3291897603), UINT32_C(4114797925), UINT32_C(3225954519), UINT32_C(4253718538), UINT32_C(3392214735) } },
    { { UINT32_C( 960138392), UINT32_C(3551653716), UINT32_C(1416233617), UINT32_C(3222241009), UINT32_C(3704094213), UINT32_C( 328994854), UINT32_C(2379879575), UINT32_C(2798589198),
        UINT32_C(4141812130), UINT32_C(2311688440), UINT32_C(2212377746), UINT32_C(3074747826), UINT32_C( 311626731), UINT32_C(2988781339), UINT32_C(1363214147), UINT32_C(3069644564) },
      { UINT32_C( 279762712), UINT32_C(4204426855), UINT32_C(1551726762), UINT32_C(1360314725), UINT32_C(3898845133), UINT32_C( 446400727), UINT32_C(2607602567), UINT32_C(  38953962),
        UINT32_C(2719153722), UINT32_C( 513584244), UINT32_C(2323323172), UINT32_C(2832961499), UINT32_C( 227556918), UINT32_C(1294478278), UINT32_C(4041774086), UINT32_C( 854735607) },
      { UINT32_C( 960138392), UINT32_C(4204426855), UINT32_C(1551726762), UINT32_C(3222241009), UINT32_C(3898845133), UINT32_C( 446400727), UINT32_C(2607602567), UINT32_C(2798589198),
        UINT32_C(4141812130), UINT32_C(2311688440), UINT32_C(2323323172), UINT32_C(3074747826), UINT32_C( 311626731), UINT32_C(2988781339), UINT32_C(4041774086), UINT32_C(3069644564) } },
    { { UINT32_C(2916353337), UINT32_C(3603722417), UINT32_C(1684031369), UINT32_C( 202128342), UINT32_C(1058708857), UINT32_C(3482075848), UINT32_C(3451876566), UINT32_C(3909071535),
        UINT32_C(1754649527), UINT32_C(3443417411), UINT32_C(2117181096), UINT32_C(1384857305), UINT32_C(2744231387), UINT32_C(3178372583), UINT32_C(1099575954), UINT32_C(2603191012) },
      { UINT32_C(2701377117), UINT32_C(3362669088), UINT32_C(3125256160), UINT32_C(3087848157), UINT32_C(1583128183), UINT32_C(1293668027), UINT32_C(3834553600), UINT32_C(2373957423),
        UINT32_C(2519630710), UINT32_C(2774441157), UINT32_C( 425698619), UINT32_C(4006702199), UINT32_C(3310103818), UINT32_C(4229130236), UINT32_C(1021419789), UINT32_C(3486081113) },
      { UINT32_C(2916353337), UINT32_C(3603722417), UINT32_C(3125256160), UINT32_C(3087848157), UINT32_C(1583128183), UINT32_C(3482075848), UINT32_C(3834553600), UINT32_C(3909071535),
        UINT32_C(2519630710), UINT32_C(3443417411), UINT32_C(2117181096), UINT32_C(4006702199), UINT32_C(3310103818), UINT32_C(4229130236), UINT32_C(1099575954), UINT32_C(3486081113) } },
    { { UINT32_C(2825254883), UINT32_C(3478045587), UINT32_C(3773345129), UINT32_C( 600815897), UINT32_C(3823705063), UINT32_C(2430598275), UINT32_C(4140613789), UINT32_C(  80057889),
        UINT32_C( 564996749), UINT32_C(1475410926), UINT32_C(3258439848), UINT32_C(2028275345), UINT32_C(2774257186), UINT32_C(1748319178), UINT32_C( 475922939), UINT32_C( 622929047) },
      { UINT32_C(1011273294), UINT32_C(1905473225), UINT32_C(2670971662), UINT32_C(4078442961), UINT32_C(2996335591), UINT32_C(2853883310), UINT32_C(1724283087), UINT32_C(3951814556),
        UINT32_C(2116538805), UINT32_C( 368098055), UINT32_C(1471488902), UINT32_C( 608947516), UINT32_C(4023837504), UINT32_C(2157572273), UINT32_C(  98983784), UINT32_C(4243616327) },
      { UINT32_C(2825254883), UINT32_C(3478045587), UINT32_C(3773345129), UINT32_C(4078442961), UINT32_C(3823705063), UINT32_C(2853883310), UINT32_C(4140613789), UINT32_C(3951814556),
        UINT32_C(2116538805), UINT32_C(1475410926), UINT32_C(3258439848), UINT32_C(2028275345), UINT32_C(4023837504), UINT32_C(2157572273), UINT32_C( 475922939), UINT32_C(4243616327) } },
    { { UINT32_C(1266358083), UINT32_C(1482714066), UINT32_C(3417314702), UINT32_C( 602930146), UINT32_C(2400372190), UINT32_C( 487566261), UINT32_C(1361245706), UINT32_C(2874020456),
        UINT32_C(4244031786), UINT32_C(3260372788), UINT32_C(1334642028), UINT32_C(3732044800), UINT32_C(4134437953), UINT32_C( 957644079), UINT32_C(3683333747), UINT32_C(1938282825) },
      { UINT32_C(3597630882), UINT32_C(1100530900), UINT32_C(3381667529), UINT32_C(3836215970), UINT32_C(3050968710), UINT32_C( 133099155), UINT32_C(1860335909), UINT32_C(4108413266),
        UINT32_C(3150598375), UINT32_C(3741082389), UINT32_C( 732466313), UINT32_C( 336547982), UINT32_C(4190759526), UINT32_C(4244682968), UINT32_C(2221663025), UINT32_C( 863521868) },
      { UINT32_C(3597630882), UINT32_C(1482714066), UINT32_C(3417314702), UINT32_C(3836215970), UINT32_C(3050968710), UINT32_C( 487566261), UINT32_C(1860335909), UINT32_C(4108413266),
        UINT32_C(4244031786), UINT32_C(3741082389), UINT32_C(1334642028), UINT32_C(3732044800), UINT32_C(4190759526), UINT32_C(4244682968), UINT32_C(3683333747), UINT32_C(1938282825) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_epu32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_epu32");
    easysimd_test_x86_assert_equal_u32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint32_t src[16];
    const easysimd__mmask16 k;
    const uint32_t a[16];
    const uint32_t b[16];
    const uint32_t r[16];
  } test_vec[] = {
    { { UINT32_C(1064309063), UINT32_C(3842122435), UINT32_C( 973388532), UINT32_C( 462482396), UINT32_C(3579167017), UINT32_C(3341136776), UINT32_C(2124560456), UINT32_C( 856035564),
        UINT32_C(3849549346), UINT32_C(2378855833), UINT32_C( 214421296), UINT32_C(3827783610), UINT32_C(2713287961), UINT32_C(2020138544), UINT32_C(  49744406), UINT32_C(1027013915) },
      UINT16_C(43379),
      { UINT32_C(3978169378), UINT32_C(1639730841), UINT32_C(2193127003), UINT32_C( 335532378), UINT32_C(3690016627), UINT32_C(2682652584), UINT32_C(1083965706), UINT32_C(1625886526),
        UINT32_C(3041724188), UINT32_C(2971011414), UINT32_C(3660828544), UINT32_C( 334377888), UINT32_C( 183492450), UINT32_C(4054439399), UINT32_C( 338773462), UINT32_C(1903434325) },
      { UINT32_C(1999028769), UINT32_C(1260928459), UINT32_C(2888195084), UINT32_C(4055897231), UINT32_C(3673993203), UINT32_C(1523295620), UINT32_C(1081015531), UINT32_C( 951181846),
        UINT32_C(1890572196), UINT32_C( 549181460), UINT32_C(3285049652), UINT32_C(3920924149), UINT32_C(3234050108), UINT32_C(1092259670), UINT32_C(2726398091), UINT32_C( 265957994) },
      { UINT32_C(3978169378), UINT32_C(1639730841), UINT32_C( 973388532), UINT32_C( 462482396), UINT32_C(3690016627), UINT32_C(2682652584), UINT32_C(1083965706), UINT32_C( 856035564),
        UINT32_C(3041724188), UINT32_C(2378855833), UINT32_C( 214421296), UINT32_C(3920924149), UINT32_C(2713287961), UINT32_C(4054439399), UINT32_C(  49744406), UINT32_C(1903434325) } },
    { { UINT32_C( 511674633), UINT32_C(2503883361), UINT32_C( 290982684), UINT32_C(3573157272), UINT32_C( 328515261), UINT32_C(3629428301), UINT32_C(2709181750), UINT32_C( 296768519),
        UINT32_C(1060057054), UINT32_C(2245356905), UINT32_C( 295120249), UINT32_C(4175860026), UINT32_C(2617997903), UINT32_C(1601527849), UINT32_C(1023471413), UINT32_C( 575582276) },
      UINT16_C(32223),
      { UINT32_C( 921323873), UINT32_C(1701077966), UINT32_C(1542954613), UINT32_C(2731886230), UINT32_C(1476591331), UINT32_C(1581725534), UINT32_C(3255798644), UINT32_C( 255848109),
        UINT32_C( 105196087), UINT32_C(  40610189), UINT32_C(3730661960), UINT32_C(2357212073), UINT32_C(2414117425), UINT32_C( 787360698), UINT32_C(1693580727), UINT32_C( 594751723) },
      { UINT32_C(3895048538), UINT32_C(2867500130), UINT32_C(2693351671), UINT32_C(2888567163), UINT32_C(1178341516), UINT32_C(4067699259), UINT32_C( 307717415), UINT32_C(4030057110),
        UINT32_C(3872939651), UINT32_C(3935355891), UINT32_C(2257197323), UINT32_C(2939336227), UINT32_C(  32861894), UINT32_C(3220466072), UINT32_C(1708280783), UINT32_C(2572486421) },
      { UINT32_C(3895048538), UINT32_C(2867500130), UINT32_C(2693351671), UINT32_C(2888567163), UINT32_C(1476591331), UINT32_C(3629428301), UINT32_C(3255798644), UINT32_C(4030057110),
        UINT32_C(3872939651), UINT32_C(2245356905), UINT32_C(3730661960), UINT32_C(2939336227), UINT32_C(2414117425), UINT32_C(3220466072), UINT32_C(1708280783), UINT32_C( 575582276) } },
    { { UINT32_C(1501507174), UINT32_C(4232253425), UINT32_C(1283640617), UINT32_C(1241232515), UINT32_C(3142250531), UINT32_C( 679165529), UINT32_C(2676837769), UINT32_C(3124290388),
        UINT32_C(  34846481), UINT32_C(4026422982), UINT32_C(2788917283), UINT32_C(1475294772), UINT32_C(2148743718), UINT32_C(  44600952), UINT32_C( 799094491), UINT32_C( 720034073) },
      UINT16_C(64912),
      { UINT32_C( 710170156), UINT32_C(2175432518), UINT32_C( 230219294), UINT32_C(1229446710), UINT32_C( 131579998), UINT32_C(1664987842), UINT32_C(3409729249), UINT32_C(2898906240),
        UINT32_C(1758862626), UINT32_C(3001712788), UINT32_C(2495652446), UINT32_C(3755804544), UINT32_C(2313598151), UINT32_C(1223435110), UINT32_C(4178782329), UINT32_C( 396745972) },
      { UINT32_C(2390719481), UINT32_C(1497393659), UINT32_C(2364407819), UINT32_C(3479948040), UINT32_C(3864613248), UINT32_C(3979232628), UINT32_C(1659257454), UINT32_C( 410618654),
        UINT32_C(  27719942), UINT32_C(1851450978), UINT32_C(4026157287), UINT32_C(2495505684), UINT32_C( 712644534), UINT32_C(3407325533), UINT32_C( 154009067), UINT32_C(2384570248) },
      { UINT32_C(1501507174), UINT32_C(4232253425), UINT32_C(1283640617), UINT32_C(1241232515), UINT32_C(3864613248), UINT32_C( 679165529), UINT32_C(2676837769), UINT32_C(2898906240),
        UINT32_C(1758862626), UINT32_C(4026422982), UINT32_C(4026157287), UINT32_C(3755804544), UINT32_C(2313598151), UINT32_C(3407325533), UINT32_C(4178782329), UINT32_C(2384570248) } },
    { { UINT32_C(  42977184), UINT32_C(2507205038), UINT32_C(1183083058), UINT32_C(2245673679), UINT32_C(3081720922), UINT32_C(3900884733), UINT32_C(1274195907), UINT32_C(4141421398),
        UINT32_C(2314823899), UINT32_C(2216585554), UINT32_C(2747966164), UINT32_C(1042916580), UINT32_C(4143307000), UINT32_C(1658746783), UINT32_C(2108608551), UINT32_C(3212085220) },
      UINT16_C(28144),
      { UINT32_C(1725317704), UINT32_C(2416487110), UINT32_C(1999957070), UINT32_C( 542059563), UINT32_C(  26799650), UINT32_C(4291936081), UINT32_C(2961618236), UINT32_C(3156047476),
        UINT32_C(2116220088), UINT32_C(3960351390), UINT32_C(1113801239), UINT32_C(2439164783), UINT32_C(4069718689), UINT32_C(4143015097), UINT32_C(2393274393), UINT32_C(2806695150) },
      { UINT32_C(1445293496), UINT32_C(2923639959), UINT32_C(3857753718), UINT32_C(4218901337), UINT32_C( 132974925), UINT32_C(2281561965), UINT32_C(1159045975), UINT32_C( 535584615),
        UINT32_C(1685459660), UINT32_C(3155343686), UINT32_C(3114402655), UINT32_C(2746489174), UINT32_C(2427101474), UINT32_C(3608651648), UINT32_C(2988256331), UINT32_C(1490160011) },
      { UINT32_C(  42977184), UINT32_C(2507205038), UINT32_C(1183083058), UINT32_C(2245673679), UINT32_C( 132974925), UINT32_C(4291936081), UINT32_C(2961618236), UINT32_C(3156047476),
        UINT32_C(2116220088), UINT32_C(2216585554), UINT32_C(3114402655), UINT32_C(2746489174), UINT32_C(4143307000), UINT32_C(4143015097), UINT32_C(2988256331), UINT32_C(3212085220) } },
    { { UINT32_C(1639729179), UINT32_C(1612631553), UINT32_C( 655999185), UINT32_C(4224437721), UINT32_C(4018894191), UINT32_C(1757913629), UINT32_C(1511711950), UINT32_C( 162721005),
        UINT32_C( 896167476), UINT32_C( 244746300), UINT32_C( 557166408), UINT32_C(3961323645), UINT32_C(2480646262), UINT32_C( 435921483), UINT32_C(1953699206), UINT32_C( 914171138) },
      UINT16_C(59283),
      { UINT32_C(  40947820), UINT32_C( 330414302), UINT32_C(4145295066), UINT32_C(4137650714), UINT32_C( 412674589), UINT32_C(1999445764), UINT32_C( 278736787), UINT32_C(3539415142),
        UINT32_C(3738461952), UINT32_C(4210197792), UINT32_C(3471902388), UINT32_C(2915340432), UINT32_C(2143640955), UINT32_C( 267842172), UINT32_C(2283770658), UINT32_C(3294238404) },
      { UINT32_C(2661494398), UINT32_C(1738053043), UINT32_C( 724994459), UINT32_C(2497247769), UINT32_C(3541278039), UINT32_C(2984381071), UINT32_C(1631125917), UINT32_C(2519110424),
        UINT32_C(1966393793), UINT32_C(4191997022), UINT32_C(1847857749), UINT32_C(1677982733), UINT32_C( 674764441), UINT32_C(3201964576), UINT32_C( 874451740), UINT32_C(1758086567) },
      { UINT32_C(2661494398), UINT32_C(1738053043), UINT32_C( 655999185), UINT32_C(4224437721), UINT32_C(3541278039), UINT32_C(1757913629), UINT32_C(1511711950), UINT32_C(3539415142),
        UINT32_C(3738461952), UINT32_C(4210197792), UINT32_C(3471902388), UINT32_C(3961323645), UINT32_C(2480646262), UINT32_C(3201964576), UINT32_C(2283770658), UINT32_C(3294238404) } },
    { { UINT32_C(1826487822), UINT32_C( 526760650), UINT32_C(3649931724), UINT32_C( 507416709), UINT32_C(3343349415), UINT32_C(2894406032), UINT32_C(3688932660), UINT32_C(4182026986),
        UINT32_C(1919230376), UINT32_C(2828127195), UINT32_C(3665895252), UINT32_C(1459142575), UINT32_C(3323871029), UINT32_C(2507318112), UINT32_C( 862999368), UINT32_C(2787947773) },
      UINT16_C(37334),
      { UINT32_C(2858201368), UINT32_C(3687428441), UINT32_C(2207938699), UINT32_C(3989033167), UINT32_C( 143664022), UINT32_C( 693885368), UINT32_C( 954030348), UINT32_C( 399094783),
        UINT32_C(3200329317), UINT32_C(1654229719), UINT32_C(3538236419), UINT32_C(2596251652), UINT32_C(2225229772), UINT32_C( 883818024), UINT32_C(1449954135), UINT32_C(2741843518) },
      { UINT32_C( 862072668), UINT32_C(3163945913), UINT32_C( 864975407), UINT32_C(4023209251), UINT32_C(3312677021), UINT32_C(3321504110), UINT32_C(3927664300), UINT32_C(4170090652),
        UINT32_C(1898705079), UINT32_C( 455983339), UINT32_C(1582218299), UINT32_C(2790071305), UINT32_C(4201431180), UINT32_C(2378131169), UINT32_C(1769528012), UINT32_C( 442566242) },
      { UINT32_C(1826487822), UINT32_C(3687428441), UINT32_C(2207938699), UINT32_C( 507416709), UINT32_C(3312677021), UINT32_C(2894406032), UINT32_C(3927664300), UINT32_C(4170090652),
        UINT32_C(3200329317), UINT32_C(2828127195), UINT32_C(3665895252), UINT32_C(1459142575), UINT32_C(4201431180), UINT32_C(2507318112), UINT32_C( 862999368), UINT32_C(2741843518) } },
    { { UINT32_C(3784019446), UINT32_C(2298263629), UINT32_C(2129021812), UINT32_C(4079235943), UINT32_C(3589116148), UINT32_C(3278089462), UINT32_C(3895253894), UINT32_C(3607268833),
        UINT32_C(1756925210), UINT32_C(3136337222), UINT32_C(1731778304), UINT32_C(4267334922), UINT32_C(3839117293), UINT32_C(2041001971), UINT32_C(4083274514), UINT32_C(2076861536) },
      UINT16_C(33521),
      { UINT32_C(3543611363), UINT32_C( 715798514), UINT32_C(4169643422), UINT32_C(2269083059), UINT32_C(  12464729), UINT32_C( 231985323), UINT32_C(2373006275), UINT32_C(2433770158),
        UINT32_C(2355447706), UINT32_C( 498470783), UINT32_C(2014723780), UINT32_C( 218060211), UINT32_C( 856473224), UINT32_C(1379983246), UINT32_C(3252662546), UINT32_C(2924670740) },
      { UINT32_C(3057301303), UINT32_C(2345922759), UINT32_C(3775129902), UINT32_C(3354198847), UINT32_C(1341848001), UINT32_C(3953212376), UINT32_C(  11305452), UINT32_C(2813263472),
        UINT32_C(2103306422), UINT32_C( 117977561), UINT32_C(1508445210), UINT32_C(3491812879), UINT32_C(2820611024), UINT32_C(1116979542), UINT32_C(2990751554), UINT32_C(4099600702) },
      { UINT32_C(3543611363), UINT32_C(2298263629), UINT32_C(2129021812), UINT32_C(4079235943), UINT32_C(1341848001), UINT32_C(3953212376), UINT32_C(2373006275), UINT32_C(2813263472),
        UINT32_C(1756925210), UINT32_C( 498470783), UINT32_C(1731778304), UINT32_C(4267334922), UINT32_C(3839117293), UINT32_C(2041001971), UINT32_C(4083274514), UINT32_C(4099600702) } },
    { { UINT32_C(3010574298), UINT32_C(  62552552), UINT32_C(2489099141), UINT32_C(1248099706), UINT32_C(4008871064), UINT32_C(2268104261), UINT32_C(  54096837), UINT32_C(1073189733),
        UINT32_C( 871524427), UINT32_C(1731636450), UINT32_C(3405550416), UINT32_C(2819907600), UINT32_C( 697698020), UINT32_C(1387316876), UINT32_C(2673207866), UINT32_C(3370012029) },
      UINT16_C(53429),
      { UINT32_C( 847026172), UINT32_C(4224044287), UINT32_C(2858145174), UINT32_C( 330383485), UINT32_C( 450510185), UINT32_C(3842249871), UINT32_C(2436006323), UINT32_C(1180821322),
        UINT32_C(2121850239), UINT32_C(1081687722), UINT32_C(2448151571), UINT32_C(2124717076), UINT32_C(1771601625), UINT32_C(1162779794), UINT32_C(1742110749), UINT32_C(3870111591) },
      { UINT32_C(3211011605), UINT32_C(2013257060), UINT32_C(3322473138), UINT32_C(1615113606), UINT32_C(3200900139), UINT32_C(2516785016), UINT32_C(4278049431), UINT32_C( 652585745),
        UINT32_C( 904219089), UINT32_C(3651986727), UINT32_C(1453307343), UINT32_C(2377573474), UINT32_C( 961249216), UINT32_C( 768561046), UINT32_C( 975948841), UINT32_C(1214320759) },
      { UINT32_C(3211011605), UINT32_C(  62552552), UINT32_C(3322473138), UINT32_C(1248099706), UINT32_C(3200900139), UINT32_C(3842249871), UINT32_C(  54096837), UINT32_C(1180821322),
        UINT32_C( 871524427), UINT32_C(1731636450), UINT32_C(3405550416), UINT32_C(2819907600), UINT32_C(1771601625), UINT32_C(1387316876), UINT32_C(1742110749), UINT32_C(3870111591) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k  = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_epu32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_epu32");
    easysimd_test_x86_assert_equal_u32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_epu32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const uint32_t a[16];
    const uint32_t b[16];
    const uint32_t r[16];
  } test_vec[] = {
    { UINT16_C(21775),
      { UINT32_C(4091585630), UINT32_C(1443008046), UINT32_C( 544481075), UINT32_C(3827144372), UINT32_C(1458616163), UINT32_C( 822659735), UINT32_C(1373879613), UINT32_C(2544300601),
        UINT32_C(3314190231), UINT32_C(1226542357), UINT32_C(1617530796), UINT32_C(2303035173), UINT32_C(3588175166), UINT32_C(1007085567), UINT32_C(3733842341), UINT32_C(1920349147) },
      { UINT32_C(3493265594), UINT32_C( 957961101), UINT32_C( 144343778), UINT32_C(1200742153), UINT32_C( 320630804), UINT32_C(4249821784), UINT32_C(3889945611), UINT32_C(3394851087),
        UINT32_C(3751448914), UINT32_C(3323573220), UINT32_C(1070510901), UINT32_C(2793824146), UINT32_C( 683254736), UINT32_C(3508865221), UINT32_C(4088922340), UINT32_C(2763854162) },
      { UINT32_C(4091585630), UINT32_C(1443008046), UINT32_C( 544481075), UINT32_C(3827144372), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0),
        UINT32_C(3751448914), UINT32_C(         0), UINT32_C(1617530796), UINT32_C(         0), UINT32_C(3588175166), UINT32_C(         0), UINT32_C(4088922340), UINT32_C(         0) } },
    { UINT16_C(22434),
      { UINT32_C(2617935491), UINT32_C( 458178637), UINT32_C(  92004735), UINT32_C(1084771207), UINT32_C(2554883699), UINT32_C(4153945151), UINT32_C(3708348960), UINT32_C( 305507214),
        UINT32_C(2125348657), UINT32_C(4271570559), UINT32_C(1728320991), UINT32_C(3550981216), UINT32_C(1500246042), UINT32_C(1011876636), UINT32_C(2082101742), UINT32_C( 898518788) },
      { UINT32_C( 246693262), UINT32_C( 437014075), UINT32_C(3280015459), UINT32_C(3616942525), UINT32_C( 892403993), UINT32_C(4067590404), UINT32_C(3731852506), UINT32_C(1762983387),
        UINT32_C(1970784314), UINT32_C(2039514134), UINT32_C(1362891156), UINT32_C(1395249722), UINT32_C(3616103123), UINT32_C(3066756059), UINT32_C(1653881223), UINT32_C(1909172278) },
      { UINT32_C(         0), UINT32_C( 458178637), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(4153945151), UINT32_C(         0), UINT32_C(1762983387),
        UINT32_C(2125348657), UINT32_C(4271570559), UINT32_C(1728320991), UINT32_C(         0), UINT32_C(3616103123), UINT32_C(         0), UINT32_C(2082101742), UINT32_C(         0) } },
    { UINT16_C(17264),
      { UINT32_C(1992787686), UINT32_C( 998792191), UINT32_C(3591226029), UINT32_C(2670780438), UINT32_C(2191133624), UINT32_C(1455104449), UINT32_C(1325330819), UINT32_C(1234268002),
        UINT32_C(4122958069), UINT32_C(1630554036), UINT32_C( 540491274), UINT32_C(3602867998), UINT32_C( 878205298), UINT32_C(4253684602), UINT32_C(1733003269), UINT32_C(3987791351) },
      { UINT32_C(3923931189), UINT32_C(3242857143), UINT32_C(1877049680), UINT32_C(1531289832), UINT32_C(1938792185), UINT32_C(3060799921), UINT32_C(2568928417), UINT32_C(3464941209),
        UINT32_C(4139280446), UINT32_C(3417768570), UINT32_C(1815779716), UINT32_C( 868712249), UINT32_C(3483784733), UINT32_C( 293934959), UINT32_C(1823122387), UINT32_C(2956603506) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(2191133624), UINT32_C(3060799921), UINT32_C(2568928417), UINT32_C(         0),
        UINT32_C(4139280446), UINT32_C(3417768570), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(1823122387), UINT32_C(         0) } },
    { UINT16_C(62104),
      { UINT32_C(1593119398), UINT32_C( 402094557), UINT32_C(2912366821), UINT32_C( 168014947), UINT32_C( 153187203), UINT32_C( 783086724), UINT32_C(2589859424), UINT32_C(1972238031),
        UINT32_C(3872621064), UINT32_C(3774728955), UINT32_C(1586337019), UINT32_C(3429405001), UINT32_C(2295695620), UINT32_C(3719725693), UINT32_C(1870140576), UINT32_C( 316998922) },
      { UINT32_C(2197338247), UINT32_C(2120414851), UINT32_C(3554472074), UINT32_C(2241873281), UINT32_C(1275950542), UINT32_C(2552873975), UINT32_C(3775373783), UINT32_C( 770960550),
        UINT32_C( 682618021), UINT32_C(1822823138), UINT32_C(2202042882), UINT32_C(2517164231), UINT32_C(1306662229), UINT32_C(2951023576), UINT32_C(1402006701), UINT32_C(2122417113) },
      { UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(2241873281), UINT32_C(1275950542), UINT32_C(         0), UINT32_C(         0), UINT32_C(1972238031),
        UINT32_C(         0), UINT32_C(3774728955), UINT32_C(         0), UINT32_C(         0), UINT32_C(2295695620), UINT32_C(3719725693), UINT32_C(1870140576), UINT32_C(2122417113) } },
    { UINT16_C(12399),
      { UINT32_C(1279414694), UINT32_C(4274930878), UINT32_C(3487471303), UINT32_C( 249836332), UINT32_C(1696185472), UINT32_C(4216505963), UINT32_C(2608802586), UINT32_C(1338764969),
        UINT32_C(4271574592), UINT32_C( 452749650), UINT32_C( 736746239), UINT32_C(2386152973), UINT32_C(4143141770), UINT32_C( 871449881), UINT32_C( 432959600), UINT32_C(2674432607) },
      { UINT32_C(4204594088), UINT32_C(1813289325), UINT32_C(2157510259), UINT32_C(1443811788), UINT32_C(1045168676), UINT32_C(3094429255), UINT32_C( 231817390), UINT32_C(2192325338),
        UINT32_C(2860271933), UINT32_C(1427608034), UINT32_C(1540796303), UINT32_C(2779899008), UINT32_C( 786693862), UINT32_C(3940963388), UINT32_C(1861793684), UINT32_C( 804300017) },
      { UINT32_C(4204594088), UINT32_C(4274930878), UINT32_C(3487471303), UINT32_C(1443811788), UINT32_C(         0), UINT32_C(4216505963), UINT32_C(2608802586), UINT32_C(         0),
        UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(         0), UINT32_C(4143141770), UINT32_C(3940963388), UINT32_C(         0), UINT32_C(         0) } },
    { UINT16_C(28142),
      { UINT32_C(4043231449), UINT32_C(4238314790), UINT32_C(2581602536), UINT32_C(2828519365), UINT32_C(3690779637), UINT32_C(3063058878), UINT32_C(4032464127), UINT32_C(2354923699),
        UINT32_C(1065179929), UINT32_C(3493534952), UINT32_C(  23665468), UINT32_C(3618177506), UINT32_C(2461181652), UINT32_C( 910705975), UINT32_C(2082907081), UINT32_C(  67666923) },
      { UINT32_C(3326313950), UINT32_C(3734404770), UINT32_C(2095055002), UINT32_C(3579087105), UINT32_C(1718093359), UINT32_C( 345878603), UINT32_C(1066451795), UINT32_C( 625187143),
        UINT32_C(3236726558), UINT32_C(2678030853), UINT32_C(2199682946), UINT32_C( 945385480), UINT32_C(3265184118), UINT32_C(3319151473), UINT32_C(1174693887), UINT32_C( 510347008) },
      { UINT32_C(         0), UINT32_C(4238314790), UINT32_C(2581602536), UINT32_C(3579087105), UINT32_C(         0), UINT32_C(3063058878), UINT32_C(4032464127), UINT32_C(2354923699),
        UINT32_C(3236726558), UINT32_C(         0), UINT32_C(2199682946), UINT32_C(3618177506), UINT32_C(         0), UINT32_C(3319151473), UINT32_C(2082907081), UINT32_C(         0) } },
    { UINT16_C(22478),
      { UINT32_C(2128270559), UINT32_C(2415746163), UINT32_C( 973014496), UINT32_C(3707401789), UINT32_C( 236415800), UINT32_C( 880088624), UINT32_C(3363599708), UINT32_C(1931430548),
        UINT32_C(2465331486), UINT32_C( 908193366), UINT32_C( 829366771), UINT32_C(3473762711), UINT32_C(  98378964), UINT32_C(2537116475), UINT32_C(1549776328), UINT32_C( 516914944) },
      { UINT32_C(3467690104), UINT32_C(2718225070), UINT32_C(2329113587), UINT32_C(2975457500), UINT32_C(1068905988), UINT32_C(1389883273), UINT32_C(2779657893), UINT32_C( 784563893),
        UINT32_C(3992745022), UINT32_C( 965673286), UINT32_C(1371759220), UINT32_C(1174543426), UINT32_C(3699816530), UINT32_C(1278107047), UINT32_C(1240587411), UINT32_C(2574759258) },
      { UINT32_C(         0), UINT32_C(2718225070), UINT32_C(2329113587), UINT32_C(3707401789), UINT32_C(         0), UINT32_C(         0), UINT32_C(3363599708), UINT32_C(1931430548),
        UINT32_C(3992745022), UINT32_C( 965673286), UINT32_C(1371759220), UINT32_C(         0), UINT32_C(3699816530), UINT32_C(         0), UINT32_C(1549776328), UINT32_C(         0) } },
    { UINT16_C(29481),
      { UINT32_C( 359952262), UINT32_C(1803020712), UINT32_C(1015527738), UINT32_C(2247416319), UINT32_C(3823279029), UINT32_C(3653269224), UINT32_C( 915282623), UINT32_C( 967423923),
        UINT32_C(3461226022), UINT32_C(1094305031), UINT32_C(2122170494), UINT32_C(1325625754), UINT32_C(4097041932), UINT32_C( 466547548), UINT32_C(3243334669), UINT32_C( 989526548) },
      { UINT32_C( 520702232), UINT32_C(2371895822), UINT32_C(2634800387), UINT32_C(1542196814), UINT32_C(1347362804), UINT32_C( 543890706), UINT32_C(2397158522), UINT32_C(3486047159),
        UINT32_C( 871354660), UINT32_C( 398479124), UINT32_C(2075446061), UINT32_C(3470172377), UINT32_C(3525191360), UINT32_C(3186788931), UINT32_C(4249604934), UINT32_C(3553432751) },
      { UINT32_C( 520702232), UINT32_C(         0), UINT32_C(         0), UINT32_C(2247416319), UINT32_C(         0), UINT32_C(3653269224), UINT32_C(         0), UINT32_C(         0),
        UINT32_C(3461226022), UINT32_C(1094305031), UINT32_C(         0), UINT32_C(         0), UINT32_C(4097041932), UINT32_C(3186788931), UINT32_C(4249604934), UINT32_C(         0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k  = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_epu32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_epu32");
    easysimd_test_x86_assert_equal_u32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_max_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C( 2014748860252733256),  INT64_C( 8310013920193191870),  INT64_C( 8627312520704586488),  INT64_C( 2141911759069453499),
        -INT64_C( 6389418296559551015),  INT64_C( 1147794582725394137), -INT64_C( 1676750639909974469), -INT64_C( 4640073323493241832) },
      { -INT64_C( 2561841185703343390), -INT64_C( 1249391887751124606), -INT64_C(   60820695810971118), -INT64_C( 7359981615789075913),
         INT64_C( 1207483183845870561),  INT64_C( 7953448598462142656), -INT64_C( 5466291888746294979), -INT64_C( 7442288857198780570) },
      { -INT64_C( 2014748860252733256),  INT64_C( 8310013920193191870),  INT64_C( 8627312520704586488),  INT64_C( 2141911759069453499),
         INT64_C( 1207483183845870561),  INT64_C( 7953448598462142656), -INT64_C( 1676750639909974469), -INT64_C( 4640073323493241832) } },
    { { -INT64_C( 5523876103837639496), -INT64_C( 7916898971599699625),  INT64_C(  597247717906238999), -INT64_C( 9166832090146792840),
        -INT64_C( 2621060327344224774),  INT64_C( 7135864873221539104),  INT64_C( 1020136329125019005),  INT64_C( 2784075781284712027) },
      { -INT64_C( 6148623457466487989),  INT64_C( 7432952495457909369), -INT64_C( 5005559260738347037),  INT64_C( 1679985374884906749),
        -INT64_C( 8061281604004781076),  INT64_C( 5340561513718759629),  INT64_C( 5743579719547462811), -INT64_C(    2875137586523631) },
      { -INT64_C( 5523876103837639496),  INT64_C( 7432952495457909369),  INT64_C(  597247717906238999),  INT64_C( 1679985374884906749),
        -INT64_C( 2621060327344224774),  INT64_C( 7135864873221539104),  INT64_C( 5743579719547462811),  INT64_C( 2784075781284712027) } },
    { {  INT64_C( 4383012266978958380),  INT64_C( 4577364368851937494), -INT64_C( 6388656697539736908), -INT64_C( 4209606055450849407),
         INT64_C( 8431544031154538838), -INT64_C( 4978949266078501628), -INT64_C( 8016663458088935294),  INT64_C( 1388437709158368231) },
      {  INT64_C( 3303348722969628098),  INT64_C( 4940374856950974927),  INT64_C( 5655143890238066908),  INT64_C( 1886465463547764830),
        -INT64_C( 5641510730089101473),  INT64_C( 2585371054122109660), -INT64_C( 5252452426412311195),  INT64_C( 6601434808538294570) },
      {  INT64_C( 4383012266978958380),  INT64_C( 4940374856950974927),  INT64_C( 5655143890238066908),  INT64_C( 1886465463547764830),
         INT64_C( 8431544031154538838),  INT64_C( 2585371054122109660), -INT64_C( 5252452426412311195),  INT64_C( 6601434808538294570) } },
    { {  INT64_C( 5231662610203744520), -INT64_C( 7309967587911851997),  INT64_C( 5698322525522991876), -INT64_C( 1170531388486392213),
         INT64_C( 7006601834514720136), -INT64_C( 1340849032208375404), -INT64_C( 5158488212946367500),  INT64_C( 3610726547756825412) },
      { -INT64_C(  277989074899897211),  INT64_C( 4348552306242220163),  INT64_C( 5042271269930050548),  INT64_C( 9067590998807353594),
        -INT64_C( 2169217286705972095), -INT64_C( 3901904170037433516), -INT64_C( 1688844438773026999),  INT64_C( 1193843738599489820) },
      {  INT64_C( 5231662610203744520),  INT64_C( 4348552306242220163),  INT64_C( 5698322525522991876),  INT64_C( 9067590998807353594),
         INT64_C( 7006601834514720136), -INT64_C( 1340849032208375404), -INT64_C( 1688844438773026999),  INT64_C( 3610726547756825412) } },
    { { -INT64_C( 8458077628258463394), -INT64_C( 2655152826666718326),  INT64_C( 5910893379497374256),  INT64_C( 6077317384298206135),
        -INT64_C( 5149955032525746232), -INT64_C( 2375579881357988198), -INT64_C( 7208005823505813904),  INT64_C( 2153344412445400728) },
      {  INT64_C( 4426654303458685831),  INT64_C( 3809888892118205777),  INT64_C( 8727927568400571551), -INT64_C( 4637325287291944911),
        -INT64_C( 5495268532625817635),  INT64_C( 3990068972844426855),  INT64_C( 4524104472594102638), -INT64_C( 7579160454283014182) },
      {  INT64_C( 4426654303458685831),  INT64_C( 3809888892118205777),  INT64_C( 8727927568400571551),  INT64_C( 6077317384298206135),
        -INT64_C( 5149955032525746232),  INT64_C( 3990068972844426855),  INT64_C( 4524104472594102638),  INT64_C( 2153344412445400728) } },
    { { -INT64_C( 8124825606155844311), -INT64_C( 1025103812337405448),  INT64_C( 3791196745065660755), -INT64_C(  781348367953927463),
         INT64_C(  510241631673269597),  INT64_C( 4261352924285226927),  INT64_C( 5146831995218388190),  INT64_C( 2908201432506807451) },
      {  INT64_C( 5284343705789914174),  INT64_C( 2933424775004679313),  INT64_C( 2574035371966943235), -INT64_C( 3425015475534655101),
         INT64_C( 8621425594407462082), -INT64_C( 4407996268128690080), -INT64_C( 8745169126165367562), -INT64_C( 3035905454064194436) },
      {  INT64_C( 5284343705789914174),  INT64_C( 2933424775004679313),  INT64_C( 3791196745065660755), -INT64_C(  781348367953927463),
         INT64_C( 8621425594407462082),  INT64_C( 4261352924285226927),  INT64_C( 5146831995218388190),  INT64_C( 2908201432506807451) } },
    { { -INT64_C( 3328486192785982096),  INT64_C( 6591386827922128888),  INT64_C( 1372890451679030403), -INT64_C( 6948492173882826072),
        -INT64_C( 7908386253090405380), -INT64_C( 8266988188849292412),  INT64_C( 4834652249182707566),  INT64_C( 3878320804479318276) },
      {  INT64_C( 1189199396536043603), -INT64_C(  417638992092411491),  INT64_C( 8015308288830753118),  INT64_C( 2215899434236132178),
        -INT64_C( 2100493519837961412),  INT64_C( 8132584015426868053),  INT64_C( 5107547021236624391),  INT64_C( 3876353501048177889) },
      {  INT64_C( 1189199396536043603),  INT64_C( 6591386827922128888),  INT64_C( 8015308288830753118),  INT64_C( 2215899434236132178),
        -INT64_C( 2100493519837961412),  INT64_C( 8132584015426868053),  INT64_C( 5107547021236624391),  INT64_C( 3878320804479318276) } },
    { { -INT64_C( 5107689581159983115),  INT64_C( 7795298184369711019),  INT64_C( 2273683656811648850), -INT64_C( 1841523710254883005),
        -INT64_C( 1041669315400470673), -INT64_C( 1173225514552318234), -INT64_C( 7434946741277387404), -INT64_C( 6630911411376317150) },
      {  INT64_C( 4678115603191831476), -INT64_C( 1390694773466359001), -INT64_C( 3475530227149510185), -INT64_C( 7933973800668719092),
        -INT64_C( 8965691194758964488),  INT64_C( 4068996085191220754), -INT64_C( 7971608261304248861),  INT64_C( 1598416259887808960) },
      {  INT64_C( 4678115603191831476),  INT64_C( 7795298184369711019),  INT64_C( 2273683656811648850), -INT64_C( 1841523710254883005),
        -INT64_C( 1041669315400470673),  INT64_C( 4068996085191220754), -INT64_C( 7434946741277387404),  INT64_C( 1598416259887808960) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C( 5345905637586622780), -INT64_C( 5692600239673997336),  INT64_C( 3447846270946787721),  INT64_C( 5280092904555912861),
        -INT64_C( 7374479798287586610), -INT64_C( 1077563827958409956), -INT64_C( 3597455596750832390), -INT64_C( 6290796570429148701) },
      UINT8_C(195),
      { -INT64_C( 2798408533011031176), -INT64_C( 8323722415045507640), -INT64_C( 6501073103404244578),  INT64_C( 6574454888980230368),
        -INT64_C( 8583698441112538686), -INT64_C(  993740444610715368),  INT64_C( 6845927178435490791),  INT64_C( 7641789052216770766) },
      {  INT64_C( 2382210729759504071), -INT64_C( 8666897520686196366), -INT64_C(  301490291198423055), -INT64_C( 8685549501623195291),
        -INT64_C( 6243645810378086608),  INT64_C( 1395034114652840758), -INT64_C( 8511991153765892169),  INT64_C( 8522231367853247742) },
      {  INT64_C( 2382210729759504071), -INT64_C( 8323722415045507640),  INT64_C( 3447846270946787721),  INT64_C( 5280092904555912861),
        -INT64_C( 7374479798287586610), -INT64_C( 1077563827958409956),  INT64_C( 6845927178435490791),  INT64_C( 8522231367853247742) } },
    { { -INT64_C( 6425500390993734669),  INT64_C( 2477780700413772589),  INT64_C( 4234762695997843223),  INT64_C( 6426422704697006706),
         INT64_C( 1560030974883127184),  INT64_C( 1236739449352888987), -INT64_C( 8636759399566856274), -INT64_C( 1501263990414566037) },
      UINT8_C( 79),
      { -INT64_C( 5431255861060637778), -INT64_C(   60869167830620238),  INT64_C( 5947457634382975244),  INT64_C( 3719578040572664798),
        -INT64_C( 3041661328608864637),  INT64_C( 4850505679108994944), -INT64_C( 2301643962556783226), -INT64_C( 2882360576230638778) },
      { -INT64_C( 7882506535313852254), -INT64_C( 1826154263602861523),  INT64_C( 4213593490977851799),  INT64_C( 1699551908358170732),
         INT64_C( 7413057174183445309), -INT64_C( 8239208018902006942),  INT64_C( 7398183810386774652), -INT64_C(  877150009380392632) },
      { -INT64_C( 5431255861060637778), -INT64_C(   60869167830620238),  INT64_C( 5947457634382975244),  INT64_C( 3719578040572664798),
         INT64_C( 1560030974883127184),  INT64_C( 1236739449352888987),  INT64_C( 7398183810386774652), -INT64_C( 1501263990414566037) } },
    { { -INT64_C( 8367932021124452289), -INT64_C( 7392389279746512155),  INT64_C( 8548032402407722559), -INT64_C( 4462778595530258841),
         INT64_C( 6587387977829929911), -INT64_C( 1262522271584044604), -INT64_C( 6714083937197980371), -INT64_C( 4407667190287825521) },
      UINT8_C( 37),
      { -INT64_C( 4040301650481798641), -INT64_C( 3532683264081408467),  INT64_C( 3162559544451224715), -INT64_C( 2782105502057237140),
         INT64_C( 2554087405900726172), -INT64_C( 3038266968933144898), -INT64_C(  680311230200947139),  INT64_C( 6603569803635770881) },
      {  INT64_C( 6174565478091302550), -INT64_C( 4216439588620820643), -INT64_C( 5435642772517771760), -INT64_C(  965983240995224451),
         INT64_C( 8193506861635313353),  INT64_C( 6060601839996899790),  INT64_C( 8764427069845029947),  INT64_C(  977930121442107459) },
      {  INT64_C( 6174565478091302550), -INT64_C( 7392389279746512155),  INT64_C( 3162559544451224715), -INT64_C( 4462778595530258841),
         INT64_C( 6587387977829929911),  INT64_C( 6060601839996899790), -INT64_C( 6714083937197980371), -INT64_C( 4407667190287825521) } },
    { {  INT64_C( 1040932122173182444),  INT64_C( 6614521354654157619),  INT64_C( 8951443263840631236),  INT64_C( 3052223651288706826),
         INT64_C( 2093503034409339070), -INT64_C( 5214218449360489944), -INT64_C( 2247946204451705831),  INT64_C( 6126735624116300191) },
      UINT8_C( 81),
      { -INT64_C( 3245026734168648911), -INT64_C( 3501974344529788691),  INT64_C( 7945060601169295347),  INT64_C( 6237302025420545716),
         INT64_C( 1288061534104570797),  INT64_C( 1445871127478838621), -INT64_C( 1121403750364760708),  INT64_C( 8832611983379297047) },
      {  INT64_C( 3656474114891692168),  INT64_C( 6719797122166889484),  INT64_C(  676892280935610424), -INT64_C( 2844066805624499648),
        -INT64_C( 8964060507010756719), -INT64_C( 4062824591738794913),  INT64_C(  641094207007357930), -INT64_C( 7756996244792792527) },
      {  INT64_C( 3656474114891692168),  INT64_C( 6614521354654157619),  INT64_C( 8951443263840631236),  INT64_C( 3052223651288706826),
         INT64_C( 1288061534104570797), -INT64_C( 5214218449360489944),  INT64_C(  641094207007357930),  INT64_C( 6126735624116300191) } },
    { {  INT64_C( 9198201055202696620),  INT64_C( 3744281605296838303),  INT64_C(  155361891174003031), -INT64_C( 8667779074086453986),
        -INT64_C( 2064530701811011398), -INT64_C( 3809474135993542489),  INT64_C( 4903312945094209849),  INT64_C( 2788039795700764751) },
      UINT8_C(203),
      {  INT64_C( 8166135916793823324), -INT64_C( 7546994602521836797), -INT64_C( 1514616460234961510),  INT64_C( 3624410160372786534),
         INT64_C( 5712871432940116605),  INT64_C( 8751230606422650485), -INT64_C( 7697179325750759702),  INT64_C( 9173377252184196421) },
      { -INT64_C( 3073812990146499140),  INT64_C( 4045396086568825293), -INT64_C( 5902904741977044656),  INT64_C( 5310333901049834032),
         INT64_C( 8392925918063036485), -INT64_C( 7142633917275690662),  INT64_C( 4154060525654465934), -INT64_C( 3661392923705184166) },
      {  INT64_C( 8166135916793823324),  INT64_C( 4045396086568825293),  INT64_C(  155361891174003031),  INT64_C( 5310333901049834032),
        -INT64_C( 2064530701811011398), -INT64_C( 3809474135993542489),  INT64_C( 4154060525654465934),  INT64_C( 9173377252184196421) } },
    { { -INT64_C( 8916305743155461850),  INT64_C( 3889999190665486868), -INT64_C( 6724487464277502102),  INT64_C( 6744062616282929474),
         INT64_C(  642166417825401146), -INT64_C( 8238099514877536560), -INT64_C( 1268415667300607620), -INT64_C( 2136024915793875257) },
      UINT8_C( 89),
      {  INT64_C( 5099575821829228423), -INT64_C( 4422825203354485314), -INT64_C( 5301479173784706312),  INT64_C( 5601703632838683412),
         INT64_C( 2232950201730075270),  INT64_C( 6265034152244963141), -INT64_C( 3477225610252886207), -INT64_C( 2096160250809541420) },
      {  INT64_C( 8609358780718197500), -INT64_C( 2688200817640031491), -INT64_C( 1549061152969609738), -INT64_C( 6702643060250659651),
         INT64_C( 2900731760192951447),  INT64_C( 8405464573246957362), -INT64_C( 1665304729403160094),  INT64_C( 7900154688119597146) },
      {  INT64_C( 8609358780718197500),  INT64_C( 3889999190665486868), -INT64_C( 6724487464277502102),  INT64_C( 5601703632838683412),
         INT64_C( 2900731760192951447), -INT64_C( 8238099514877536560), -INT64_C( 1665304729403160094), -INT64_C( 2136024915793875257) } },
    { {  INT64_C( 5468346420173447312),  INT64_C( 9102827748989560416),  INT64_C( 8744400713309190215), -INT64_C( 1655886121147999037),
         INT64_C( 1522365889094368444),  INT64_C( 4253446389175105517), -INT64_C( 7253600422308512065), -INT64_C( 6294561215247757212) },
      UINT8_C(118),
      { -INT64_C( 3116982845547250359), -INT64_C( 3730946081185773773),  INT64_C( 4404028325641852594), -INT64_C( 3953085697309943180),
        -INT64_C( 4413148788968239537),  INT64_C( 4663888145844832927),  INT64_C( 1239924339176529291),  INT64_C( 9168451639147716339) },
      { -INT64_C( 3638578444647016911), -INT64_C( 4718238374845301322), -INT64_C( 8394449565981966127),  INT64_C( 5978874995294486346),
         INT64_C( 5191968197482538257),  INT64_C(  883007048760805457),  INT64_C( 3366154728906684562),  INT64_C(  121213393199281466) },
      {  INT64_C( 5468346420173447312), -INT64_C( 3730946081185773773),  INT64_C( 4404028325641852594), -INT64_C( 1655886121147999037),
         INT64_C( 5191968197482538257),  INT64_C( 4663888145844832927),  INT64_C( 3366154728906684562), -INT64_C( 6294561215247757212) } },
    { { -INT64_C( 6565879561257369593),  INT64_C( 1395950409363254807),  INT64_C( 1113784204694313569), -INT64_C( 4063627027580052055),
         INT64_C( 4814419655343004888), -INT64_C( 4376667185308370769),  INT64_C( 2211968311192289486),  INT64_C(  945847410351414426) },
      UINT8_C( 47),
      {  INT64_C(  175813054986664320),  INT64_C( 7202426655844114482),  INT64_C( 7370411586518292273),  INT64_C( 4769779031566164398),
         INT64_C( 6696417419348708578), -INT64_C( 6027373776964791532), -INT64_C( 2900702815815323122), -INT64_C( 2069394128779060925) },
      { -INT64_C( 3090666984198681451),  INT64_C( 2537820259726695697),  INT64_C( 8599404911143916120), -INT64_C( 3497109734551637870),
         INT64_C( 1560120159091181073), -INT64_C( 1117216409145740730), -INT64_C( 4316761598062364309),  INT64_C( 7638613325489083640) },
      {  INT64_C(  175813054986664320),  INT64_C( 7202426655844114482),  INT64_C( 8599404911143916120),  INT64_C( 4769779031566164398),
         INT64_C( 4814419655343004888), -INT64_C( 1117216409145740730),  INT64_C( 2211968311192289486),  INT64_C(  945847410351414426) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C( 63),
      { -INT64_C( 6926915754504825890), -INT64_C( 7397189029588016527), -INT64_C( 7536201324114600452),  INT64_C( 6873027948649756836),
         INT64_C( 2894290321799581881),  INT64_C(  816700812018984951),  INT64_C( 6055501674159253406),  INT64_C( 4688202719175287342) },
      {  INT64_C( 6036784323415033575),  INT64_C( 1096738046026444830), -INT64_C( 1724727222163454964),  INT64_C( 5040836808604616235),
        -INT64_C( 3018632514112339604), -INT64_C( 7198655944328992103),  INT64_C( 5702235678228126336), -INT64_C( 3633288481376912657) },
      {  INT64_C( 6036784323415033575),  INT64_C( 1096738046026444830), -INT64_C( 1724727222163454964),  INT64_C( 6873027948649756836),
         INT64_C( 2894290321799581881),  INT64_C(  816700812018984951),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(184),
      { -INT64_C( 2686037980477352586),  INT64_C( 8576831563879859274),  INT64_C( 2213495460442358366),  INT64_C( 1052519153161667820),
        -INT64_C( 7175387239475704863),  INT64_C( 4179388676098479531),  INT64_C( 8282322599611046765), -INT64_C( 7177909069199085635) },
      {  INT64_C( 2588330053329588476), -INT64_C( 6448119779903664530), -INT64_C(  414216551051786936), -INT64_C( 8994056214273878569),
         INT64_C( 7052000346529422146),  INT64_C( 9199497477670800075),  INT64_C( 7127946467432276915),  INT64_C( 3327072624578935331) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 1052519153161667820),
         INT64_C( 7052000346529422146),  INT64_C( 9199497477670800075),  INT64_C(                   0),  INT64_C( 3327072624578935331) } },
    { UINT8_C(125),
      {  INT64_C( 5730217847577176318),  INT64_C( 6367815225705925245),  INT64_C( 8503998119877496915), -INT64_C( 2821345899037151943),
        -INT64_C( 7526332901285017480),  INT64_C( 7867685305908925346), -INT64_C( 8024479576668368142),  INT64_C(  986784841744451071) },
      { -INT64_C(   90678527186484213), -INT64_C( 7750126914822143765),  INT64_C( 1296303397577867286), -INT64_C( 5328149050863916949),
         INT64_C( 2034107717976349152),  INT64_C( 3013351336526811034),  INT64_C( 7239417625381868371), -INT64_C( 2315568197507194245) },
      {  INT64_C( 5730217847577176318),  INT64_C(                   0),  INT64_C( 8503998119877496915), -INT64_C( 2821345899037151943),
         INT64_C( 2034107717976349152),  INT64_C( 7867685305908925346),  INT64_C( 7239417625381868371),  INT64_C(                   0) } },
    { UINT8_C(203),
      {  INT64_C( 7344835900854770617), -INT64_C( 8060982651336971462),  INT64_C( 6938056904573290297),  INT64_C(  400606287485627985),
        -INT64_C( 2372193426292044711),  INT64_C( 7637769824989187441), -INT64_C( 4200594613847357610), -INT64_C( 6252094350377282836) },
      {  INT64_C( 5560094904066545061), -INT64_C(  246649431242582022),  INT64_C( 3636942875797801024), -INT64_C( 4535223658831346922),
        -INT64_C( 3574955593694484677),  INT64_C( 1706173592363343371),  INT64_C( 1786597550360100179),  INT64_C( 1595838715907683656) },
      {  INT64_C( 7344835900854770617), -INT64_C(  246649431242582022),  INT64_C(                   0),  INT64_C(  400606287485627985),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C( 1786597550360100179),  INT64_C( 1595838715907683656) } },
    { UINT8_C(245),
      { -INT64_C( 5877030867296506832), -INT64_C( 3715859129705291377),  INT64_C(  645957393419470697), -INT64_C( 5771088594544141724),
        -INT64_C( 2854011911252233947), -INT64_C( 1134799871686743387),  INT64_C( 6432148469508345448),  INT64_C( 4979911498035570414) },
      {  INT64_C( 6342861248878227685), -INT64_C( 4341455357083846328), -INT64_C( 2132099336853627109), -INT64_C( 3617115179733502872),
         INT64_C( 7973624404321748892), -INT64_C( 7126665479367789317), -INT64_C( 4525248627699017890), -INT64_C( 8394965961651158070) },
      {  INT64_C( 6342861248878227685),  INT64_C(                   0),  INT64_C(  645957393419470697),  INT64_C(                   0),
         INT64_C( 7973624404321748892), -INT64_C( 1134799871686743387),  INT64_C( 6432148469508345448),  INT64_C( 4979911498035570414) } },
    { UINT8_C(222),
      { -INT64_C(  614896919360775558), -INT64_C(  284265650225361743),  INT64_C( 1631172314283728834),  INT64_C( 5716353073522496864),
         INT64_C( 8999391399028414570),  INT64_C( 3814613149780040719), -INT64_C( 4953202734526544487),  INT64_C( 4767771417161910021) },
      {  INT64_C( 5126806359902254559),  INT64_C( 2672371312965145199), -INT64_C( 2844141002291010704),  INT64_C( 2230932456099527132),
        -INT64_C( 6390064476090522414),  INT64_C( 2878537624090872896), -INT64_C( 5445561303790566207), -INT64_C( 3489904893107888077) },
      {  INT64_C(                   0),  INT64_C( 2672371312965145199),  INT64_C( 1631172314283728834),  INT64_C( 5716353073522496864),
         INT64_C( 8999391399028414570),  INT64_C(                   0), -INT64_C( 4953202734526544487),  INT64_C( 4767771417161910021) } },
    { UINT8_C(130),
      { -INT64_C( 5149760434878162343), -INT64_C( 8410024495666997783),  INT64_C( 7653612797919747466), -INT64_C( 7755419307346515584),
         INT64_C( 7130434581909215505), -INT64_C( 8823901891757185863), -INT64_C( 8663307170344210672),  INT64_C( 8446717037667167593) },
      {  INT64_C( 3655135608368600285), -INT64_C( 1834444071780572406),  INT64_C( 4340071271745262509),  INT64_C( 4615372009170313012),
         INT64_C( 7986940857370000940),  INT64_C( 8408218063844084211),  INT64_C( 3179651257537720592),  INT64_C( 5019210216756215811) },
      {  INT64_C(                   0), -INT64_C( 1834444071780572406),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 8446717037667167593) } },
    { UINT8_C(143),
      { -INT64_C( 3782633628547880386), -INT64_C( 1521082004566553898), -INT64_C( 1687144622622324945), -INT64_C( 2219352522735092526),
         INT64_C( 6898934679470024497), -INT64_C( 8782556560020516806), -INT64_C( 2112558692907286050),  INT64_C( 2752184211040743340) },
      { -INT64_C( 3762521568646160161), -INT64_C( 1079704945889903834),  INT64_C( 8723584410143104287),  INT64_C( 3328434238193702420),
         INT64_C( 5113379014405736858),  INT64_C( 3701614834299958875), -INT64_C( 8202336425020942875), -INT64_C( 8681593259977805048) },
      { -INT64_C( 3762521568646160161), -INT64_C( 1079704945889903834),  INT64_C( 8723584410143104287),  INT64_C( 3328434238193702420),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 2752184211040743340) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_max_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint64_t a[8];
    const uint64_t b[8];
    const uint64_t r[8];
  } test_vec[] = {
    { { UINT64_C(14405647450865401052), UINT64_C(15069161228696060266), UINT64_C( 1509971145447934469), UINT64_C( 8101907096340504770),
        UINT64_C( 6164751603358090941), UINT64_C( 9383154700410950170), UINT64_C(18016976009369467443), UINT64_C( 2291622348271100360) },
      { UINT64_C( 3206970931547568704), UINT64_C(16610341617040230789), UINT64_C(12085098299692660611), UINT64_C( 8937340941237469573),
        UINT64_C(16467737902515219530), UINT64_C( 7623021755386698802), UINT64_C( 1127830146735514912), UINT64_C(14219269727095073437) },
      { UINT64_C(14405647450865401052), UINT64_C(16610341617040230789), UINT64_C(12085098299692660611), UINT64_C( 8937340941237469573),
        UINT64_C(16467737902515219530), UINT64_C( 9383154700410950170), UINT64_C(18016976009369467443), UINT64_C(14219269727095073437) } },
    { { UINT64_C( 6210089108541471978), UINT64_C(11391298349110596807), UINT64_C(12917524493384956843), UINT64_C( 2607267771482651630),
        UINT64_C(14075891762244505820), UINT64_C( 4885709158955913905), UINT64_C(11424432347470654401), UINT64_C(15300194644856870904) },
      { UINT64_C( 1244168190067852165), UINT64_C(18129817794156475583), UINT64_C(14323520279921431161), UINT64_C( 9962047057146990452),
        UINT64_C(   36678889460405521), UINT64_C( 5204175241816293891), UINT64_C(15895518007174171139), UINT64_C(17264136708841574408) },
      { UINT64_C( 6210089108541471978), UINT64_C(18129817794156475583), UINT64_C(14323520279921431161), UINT64_C( 9962047057146990452),
        UINT64_C(14075891762244505820), UINT64_C( 5204175241816293891), UINT64_C(15895518007174171139), UINT64_C(17264136708841574408) } },
    { { UINT64_C(17573013795190645149), UINT64_C( 3223565956230952868), UINT64_C( 7978010633431821683), UINT64_C( 2806887743663833127),
        UINT64_C(15751309145066001587), UINT64_C(11776923128482875163), UINT64_C( 3101912537289879095), UINT64_C( 5468536085979105077) },
      { UINT64_C(16827848225631234424), UINT64_C( 1594447851292579405), UINT64_C(  665337386996375051), UINT64_C(  588752815020010311),
        UINT64_C(17098830368325704340), UINT64_C( 2309092160385546261), UINT64_C(14269491042304638762), UINT64_C( 1112056481645514710) },
      { UINT64_C(17573013795190645149), UINT64_C( 3223565956230952868), UINT64_C( 7978010633431821683), UINT64_C( 2806887743663833127),
        UINT64_C(17098830368325704340), UINT64_C(11776923128482875163), UINT64_C(14269491042304638762), UINT64_C( 5468536085979105077) } },
    { { UINT64_C( 8825010614149653995), UINT64_C(11054502478261248820), UINT64_C(15672700442101913658), UINT64_C(16354731852084225645),
        UINT64_C( 6391423864432060627), UINT64_C(15551222658663260873), UINT64_C( 8394166579517024418), UINT64_C( 4472729099770314040) },
      { UINT64_C( 3402232465802559675), UINT64_C( 9637485374303950922), UINT64_C(11177276091413450914), UINT64_C( 9876356383904594534),
        UINT64_C(17938858413209978205), UINT64_C( 3954335932376701816), UINT64_C( 1940485961097874159), UINT64_C(17567974339967170679) },
      { UINT64_C( 8825010614149653995), UINT64_C(11054502478261248820), UINT64_C(15672700442101913658), UINT64_C(16354731852084225645),
        UINT64_C(17938858413209978205), UINT64_C(15551222658663260873), UINT64_C( 8394166579517024418), UINT64_C(17567974339967170679) } },
    { { UINT64_C( 5631656596421883520), UINT64_C(10794998180465936132), UINT64_C( 2549552700474240916), UINT64_C(14417488366027623820),
        UINT64_C( 8759253289225669483), UINT64_C( 6224224011284527397), UINT64_C(12205035486931994769), UINT64_C( 7448356734173431628) },
      { UINT64_C(10924105908965889195), UINT64_C( 2272842877965085809), UINT64_C( 8417434579905554442), UINT64_C( 2803602349141564292),
        UINT64_C( 4162137255479578809), UINT64_C(17382759758752982157), UINT64_C(15617050106511530015), UINT64_C(16295502471031800707) },
      { UINT64_C(10924105908965889195), UINT64_C(10794998180465936132), UINT64_C( 8417434579905554442), UINT64_C(14417488366027623820),
        UINT64_C( 8759253289225669483), UINT64_C(17382759758752982157), UINT64_C(15617050106511530015), UINT64_C(16295502471031800707) } },
    { { UINT64_C(15768175752069868753), UINT64_C( 6254710672982425844), UINT64_C( 5906285979108238794), UINT64_C( 7072188056615276570),
        UINT64_C(17800706234978677473), UINT64_C(18131104183864196880), UINT64_C(12512143889682480005), UINT64_C( 5355929401625212000) },
      { UINT64_C( 3549259957032996936), UINT64_C( 4083665022662284416), UINT64_C( 3932540599173629267), UINT64_C(16273894252460147748),
        UINT64_C( 5917287713101074892), UINT64_C(  102931529247987585), UINT64_C( 5584880430196717940), UINT64_C(17400418183870654975) },
      { UINT64_C(15768175752069868753), UINT64_C( 6254710672982425844), UINT64_C( 5906285979108238794), UINT64_C(16273894252460147748),
        UINT64_C(17800706234978677473), UINT64_C(18131104183864196880), UINT64_C(12512143889682480005), UINT64_C(17400418183870654975) } },
    { { UINT64_C(10748399147852670469), UINT64_C( 5711470167293832339), UINT64_C(11936539738650585904), UINT64_C( 4312961039629910724),
        UINT64_C( 5261958865101175133), UINT64_C( 4076547143300272231), UINT64_C(  811835713104456953), UINT64_C(10893589821946888891) },
      { UINT64_C(16322749821918658439), UINT64_C(15644862852804973022), UINT64_C(11688457208457637859), UINT64_C( 9155749836091566399),
        UINT64_C(  242704158681732864), UINT64_C( 2092773298875761491), UINT64_C( 9241241581640975541), UINT64_C(10744190770404184997) },
      { UINT64_C(16322749821918658439), UINT64_C(15644862852804973022), UINT64_C(11936539738650585904), UINT64_C( 9155749836091566399),
        UINT64_C( 5261958865101175133), UINT64_C( 4076547143300272231), UINT64_C( 9241241581640975541), UINT64_C(10893589821946888891) } },
    { { UINT64_C( 7802607409231353631), UINT64_C(17098103143538831857), UINT64_C(12749631220573966126), UINT64_C( 1136992811779745949),
        UINT64_C( 4019072642946750272), UINT64_C( 4536438805688968654), UINT64_C(16642943881719938619), UINT64_C(17042992821668693125) },
      { UINT64_C( 4914566798546229686), UINT64_C( 9060749055168845681), UINT64_C(10298812095332117693), UINT64_C(11067745496159421695),
        UINT64_C( 6565063991793999456), UINT64_C( 7071102926157735521), UINT64_C(11501442069804147974), UINT64_C( 9860035617323917400) },
      { UINT64_C( 7802607409231353631), UINT64_C(17098103143538831857), UINT64_C(12749631220573966126), UINT64_C(11067745496159421695),
        UINT64_C( 6565063991793999456), UINT64_C( 7071102926157735521), UINT64_C(16642943881719938619), UINT64_C(17042992821668693125) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_epu64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_epu64");
    easysimd_test_x86_assert_equal_u64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const uint64_t src[8];
    const easysimd__mmask8 k;
    const uint64_t a[8];
    const uint64_t b[8];
    const uint64_t r[8];
  } test_vec[] = {
    { { UINT64_C(15079150202399567154), UINT64_C(11996494373043193425), UINT64_C( 4858319910155527770), UINT64_C(  972352950523515139),
        UINT64_C( 9273669397617985450), UINT64_C( 6895137495000279543), UINT64_C(17516062391167727514), UINT64_C( 9027402588605666910) },
      UINT8_C( 14),
      { UINT64_C(18041063940208484277), UINT64_C(15042407688237443251), UINT64_C(18332944238196593279), UINT64_C(11731964907599537659),
        UINT64_C( 8987609023050681518), UINT64_C(10932316678593427918), UINT64_C( 2782115345532745719), UINT64_C( 7728943035109109428) },
      { UINT64_C( 9226438237850586052), UINT64_C(  670224985266564510), UINT64_C(13034899466499912085), UINT64_C( 5393790103083929010),
        UINT64_C( 9733653087995233506), UINT64_C( 3362652464225287055), UINT64_C( 5130114089551221116), UINT64_C(   31683652269541024) },
      { UINT64_C(15079150202399567154), UINT64_C(15042407688237443251), UINT64_C(18332944238196593279), UINT64_C(11731964907599537659),
        UINT64_C( 9273669397617985450), UINT64_C( 6895137495000279543), UINT64_C(17516062391167727514), UINT64_C( 9027402588605666910) } },
    { { UINT64_C(13260189574033065877), UINT64_C( 5254016093149282851), UINT64_C( 5418383040647124820), UINT64_C(  980062302016497788),
        UINT64_C(17872975842067033384), UINT64_C(13467907369983042258), UINT64_C( 2192080535954959478), UINT64_C(16592195325271792829) },
      UINT8_C(127),
      { UINT64_C(11759546520185354191), UINT64_C( 7284224645044752641), UINT64_C( 9149668071190247015), UINT64_C( 4804815730559955166),
        UINT64_C( 5854350008285004633), UINT64_C( 8197122012874885029), UINT64_C(14870507672627431998), UINT64_C( 1625144965279357655) },
      { UINT64_C(12128455497450794927), UINT64_C( 9777535007936889397), UINT64_C(15947464007483874962), UINT64_C(12192288702057696864),
        UINT64_C(  545490614618743781), UINT64_C(17441305971535615802), UINT64_C(11040791336636937045), UINT64_C(18065883231876964371) },
      { UINT64_C(12128455497450794927), UINT64_C( 9777535007936889397), UINT64_C(15947464007483874962), UINT64_C(12192288702057696864),
        UINT64_C( 5854350008285004633), UINT64_C(17441305971535615802), UINT64_C(14870507672627431998), UINT64_C(16592195325271792829) } },
    { { UINT64_C(10495203897032007373), UINT64_C( 4205272321756622091), UINT64_C(12206669887595467155), UINT64_C(15221441089743756333),
        UINT64_C(11507005547386904778), UINT64_C( 8801554193032332806), UINT64_C(13147886965225929527), UINT64_C( 7107303191896206537) },
      UINT8_C( 87),
      { UINT64_C(18100964383557124884), UINT64_C(18317085556504403605), UINT64_C(11752773760238157987), UINT64_C(11584276992475588918),
        UINT64_C(14938721689529069345), UINT64_C( 3702237685978116894), UINT64_C( 6492111642770532350), UINT64_C( 1491688678203282567) },
      { UINT64_C(11547998559192908089), UINT64_C(  626389620384468462), UINT64_C( 6469868170235425866), UINT64_C( 6120989043794415850),
        UINT64_C(  486962808488418464), UINT64_C( 8082330919157154839), UINT64_C( 2924428514014766954), UINT64_C(13954112213641134392) },
      { UINT64_C(18100964383557124884), UINT64_C(18317085556504403605), UINT64_C(11752773760238157987), UINT64_C(15221441089743756333),
        UINT64_C(14938721689529069345), UINT64_C( 8801554193032332806), UINT64_C( 6492111642770532350), UINT64_C( 7107303191896206537) } },
    { { UINT64_C(10336506031410212436), UINT64_C(10808878990613346153), UINT64_C(13828135013600911234), UINT64_C( 4056257706092260712),
        UINT64_C( 4264090615561858342), UINT64_C( 4238391616941513998), UINT64_C( 8354143271978116009), UINT64_C(16135067853370687950) },
      UINT8_C(115),
      { UINT64_C( 9778582161350557071), UINT64_C( 7187597396203794251), UINT64_C(16150720662526160755), UINT64_C( 5735466887821251806),
        UINT64_C(12188616764912164597), UINT64_C( 5961779504480216729), UINT64_C(16946139457334422381), UINT64_C( 3651198003916621213) },
      { UINT64_C(16469655367179149703), UINT64_C( 4148089912404299358), UINT64_C(11249344253358650916), UINT64_C( 6766839682207067512),
        UINT64_C( 1113746667938756933), UINT64_C(14926580266168070432), UINT64_C(15469334059930397459), UINT64_C(10412167630026417995) },
      { UINT64_C(16469655367179149703), UINT64_C( 7187597396203794251), UINT64_C(13828135013600911234), UINT64_C( 4056257706092260712),
        UINT64_C(12188616764912164597), UINT64_C(14926580266168070432), UINT64_C(16946139457334422381), UINT64_C(16135067853370687950) } },
    { { UINT64_C(13037521338386924077), UINT64_C( 3152173500096068421), UINT64_C( 2856949971750403953), UINT64_C(15091220011794641043),
        UINT64_C( 7481214700885085834), UINT64_C(12113580427719439064), UINT64_C(15769385185188469460), UINT64_C( 8341273345579819341) },
      UINT8_C(181),
      { UINT64_C(15413357944398975345), UINT64_C( 5656721194440579222), UINT64_C( 3140818780600676653), UINT64_C(13475764358446679847),
        UINT64_C(12777751299412908826), UINT64_C( 4813184810654457993), UINT64_C(17673570581272616975), UINT64_C(18207569383574952618) },
      { UINT64_C(12965807433526704162), UINT64_C( 4217053884531690541), UINT64_C(15933902827174433116), UINT64_C(14830775423911159026),
        UINT64_C( 5032203140213722104), UINT64_C( 6893617963061478982), UINT64_C( 9885308373498002974), UINT64_C( 8612906137515065359) },
      { UINT64_C(15413357944398975345), UINT64_C( 3152173500096068421), UINT64_C(15933902827174433116), UINT64_C(15091220011794641043),
        UINT64_C(12777751299412908826), UINT64_C( 6893617963061478982), UINT64_C(15769385185188469460), UINT64_C(18207569383574952618) } },
    { { UINT64_C(15265355061528203899), UINT64_C(10149125018022601077), UINT64_C( 2021567634450834157), UINT64_C( 1730612183287884813),
        UINT64_C( 9390151511762050544), UINT64_C( 8134295338509571303), UINT64_C(15735299803182383157), UINT64_C( 9521852691968832879) },
      UINT8_C(  8),
      { UINT64_C(16518242809352028624), UINT64_C( 3790017080827875985), UINT64_C( 8016648382363851725), UINT64_C( 4662500432227290177),
        UINT64_C(17347534123791927432), UINT64_C(14703387462753003108), UINT64_C( 2986129441964166599), UINT64_C( 9428437529299088168) },
      { UINT64_C(11862790659114757714), UINT64_C(12036583450803500095), UINT64_C(12368601479159260821), UINT64_C( 6271574766159953036),
        UINT64_C(13722091476665001354), UINT64_C(  515106296207043230), UINT64_C( 8420372200233946796), UINT64_C(14268534173768294311) },
      { UINT64_C(15265355061528203899), UINT64_C(10149125018022601077), UINT64_C( 2021567634450834157), UINT64_C( 6271574766159953036),
        UINT64_C( 9390151511762050544), UINT64_C( 8134295338509571303), UINT64_C(15735299803182383157), UINT64_C( 9521852691968832879) } },
    { { UINT64_C( 6099843906370800745), UINT64_C( 6080977323875803881), UINT64_C(13412387178399721671), UINT64_C(10051869590686145918),
        UINT64_C(17906621146379522167), UINT64_C( 1421088320658887611), UINT64_C( 1832371980796509344), UINT64_C(13091773068631790337) },
      UINT8_C( 98),
      { UINT64_C(13184122309087191586), UINT64_C( 5698765551812369342), UINT64_C(  701439175578798551), UINT64_C(12793033908292149461),
        UINT64_C( 2520016210279398110), UINT64_C(16691094554133723712), UINT64_C( 7257091820740578423), UINT64_C(15672269395207192126) },
      { UINT64_C(10490592523055688720), UINT64_C( 5982054485007281677), UINT64_C( 4829747781398734354), UINT64_C(13224978256132870836),
        UINT64_C(14042155147592442620), UINT64_C( 6637992811178214383), UINT64_C( 9930442493608730249), UINT64_C( 7851393113686335894) },
      { UINT64_C( 6099843906370800745), UINT64_C( 5982054485007281677), UINT64_C(13412387178399721671), UINT64_C(10051869590686145918),
        UINT64_C(17906621146379522167), UINT64_C(16691094554133723712), UINT64_C( 9930442493608730249), UINT64_C(13091773068631790337) } },
    { { UINT64_C(15121567895206371427), UINT64_C( 2667181685818112348), UINT64_C(11184833735020380634), UINT64_C(10683045405007573348),
        UINT64_C(17339288014067399662), UINT64_C(15276380338257346111), UINT64_C(11113682348762444699), UINT64_C(14984232292701836076) },
      UINT8_C(170),
      { UINT64_C( 9179795368013897631), UINT64_C(17652788107595563206), UINT64_C(16841427232288840656), UINT64_C(12073192964517140018),
        UINT64_C(13904919900873149240), UINT64_C(10279215180359430514), UINT64_C(14444681820091568566), UINT64_C( 1339599723839504615) },
      { UINT64_C(  701298851474793473), UINT64_C(12412031583756540916), UINT64_C( 1338605017375463733), UINT64_C( 8491988394434510318),
        UINT64_C(15825135513782740222), UINT64_C(17630129911493384711), UINT64_C( 9179552724599956263), UINT64_C(18408217190605416335) },
      { UINT64_C(15121567895206371427), UINT64_C(17652788107595563206), UINT64_C(11184833735020380634), UINT64_C(12073192964517140018),
        UINT64_C(17339288014067399662), UINT64_C(17630129911493384711), UINT64_C(11113682348762444699), UINT64_C(18408217190605416335) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k  = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_epu64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_epu64");
    easysimd_test_x86_assert_equal_u64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_epu64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const uint64_t a[8];
    const uint64_t b[8];
    const uint64_t r[8];
  } test_vec[] = {
    { UINT8_C(126),
      { UINT64_C( 1027662572529108014), UINT64_C( 3982020794999916420), UINT64_C(  252009961771637167), UINT64_C( 6944830119902470000),
        UINT64_C( 8177651732383854531), UINT64_C( 6136279454117243937), UINT64_C( 3190361335639665290), UINT64_C( 4639110574336016079) },
      { UINT64_C(17706792351699958109), UINT64_C(16560034190430038741), UINT64_C(  171540522247766650), UINT64_C( 4299642270912835566),
        UINT64_C( 6280340608305682526), UINT64_C(10982772191809179434), UINT64_C(17409841952036131687), UINT64_C( 7185219446340624287) },
      { UINT64_C(                   0), UINT64_C(16560034190430038741), UINT64_C(  252009961771637167), UINT64_C( 6944830119902470000),
        UINT64_C( 8177651732383854531), UINT64_C(10982772191809179434), UINT64_C(17409841952036131687), UINT64_C(                   0) } },
    { UINT8_C( 95),
      { UINT64_C(14746537515878028604), UINT64_C( 2961542913226124044), UINT64_C(17158746807588402001), UINT64_C(10439438857185500281),
        UINT64_C(  537533619700089323), UINT64_C( 3863756309488230623), UINT64_C( 7116486656671533956), UINT64_C(17750869812158051699) },
      { UINT64_C(17835685094218491012), UINT64_C(10704785900324011637), UINT64_C(10313288350108698069), UINT64_C( 5323445825086816990),
        UINT64_C(  729949913378946834), UINT64_C( 7867492332007034251), UINT64_C(11449077962828912184), UINT64_C(17896011782137788749) },
      { UINT64_C(17835685094218491012), UINT64_C(10704785900324011637), UINT64_C(17158746807588402001), UINT64_C(10439438857185500281),
        UINT64_C(  729949913378946834), UINT64_C(                   0), UINT64_C(11449077962828912184), UINT64_C(                   0) } },
    { UINT8_C(231),
      { UINT64_C(15058776303739806243), UINT64_C( 6490766529442387433), UINT64_C( 9482530545143998208), UINT64_C( 2994157107972582207),
        UINT64_C( 8618082702921894277), UINT64_C(15440704395747197226), UINT64_C( 6385181889134682574), UINT64_C(17119462463658395236) },
      { UINT64_C(17049308509582536341), UINT64_C( 2520927636245114448), UINT64_C( 4320596734292729220), UINT64_C( 1455571422874629085),
        UINT64_C( 9806538951819323752), UINT64_C(10968703895700697793), UINT64_C( 2871091262402163655), UINT64_C( 2428178768665017886) },
      { UINT64_C(17049308509582536341), UINT64_C( 6490766529442387433), UINT64_C( 9482530545143998208), UINT64_C(                   0),
        UINT64_C(                   0), UINT64_C(15440704395747197226), UINT64_C( 6385181889134682574), UINT64_C(17119462463658395236) } },
    { UINT8_C(136),
      { UINT64_C( 6232498531844771050), UINT64_C( 4496566926057313270), UINT64_C( 6665802288877536568), UINT64_C(13421913059590741532),
        UINT64_C( 4845298489065145475), UINT64_C(16398533863126902665), UINT64_C(16684367445016058704), UINT64_C( 6372847278295785445) },
      { UINT64_C( 7384633303577389291), UINT64_C(18136363674212458276), UINT64_C(11825242876091692905), UINT64_C(10340487550843141714),
        UINT64_C( 6986845799318012082), UINT64_C( 7586842434398564770), UINT64_C( 8663440408587367128), UINT64_C( 1273148012031415415) },
      { UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(13421913059590741532),
        UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0), UINT64_C( 6372847278295785445) } },
    { UINT8_C(225),
      { UINT64_C( 5285113357058023794), UINT64_C( 7018212331969160621), UINT64_C(14663719519278070863), UINT64_C(13304261424280411040),
        UINT64_C(16915175217715658717), UINT64_C(  701397955142748011), UINT64_C(16274761387061887705), UINT64_C( 2101567219713574188) },
      { UINT64_C(14568108359905613828), UINT64_C( 6565086627270796376), UINT64_C(16808637488467487777), UINT64_C( 3434423712426485323),
        UINT64_C(16521345885245815582), UINT64_C( 7718345448690772800), UINT64_C( 1865917201317201982), UINT64_C(16524355500467569144) },
      { UINT64_C(14568108359905613828), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0),
        UINT64_C(                   0), UINT64_C( 7718345448690772800), UINT64_C(16274761387061887705), UINT64_C(16524355500467569144) } },
    { UINT8_C(137),
      { UINT64_C( 1486182007407207106), UINT64_C(  451144455750050289), UINT64_C( 7251056137478618775), UINT64_C(15593732495406515090),
        UINT64_C( 7652133053253779059), UINT64_C(17347214139548602424), UINT64_C( 1257974888838828525), UINT64_C( 2732094745310437885) },
      { UINT64_C( 3510965994342457815), UINT64_C(14145423620710999812), UINT64_C( 2145202301845235509), UINT64_C(16556105305213154795),
        UINT64_C( 9035608956746401084), UINT64_C( 2571601493381302805), UINT64_C(13496897546967549200), UINT64_C(15295662050699148881) },
      { UINT64_C( 3510965994342457815), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(16556105305213154795),
        UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(15295662050699148881) } },
    { UINT8_C( 33),
      { UINT64_C(13222032409913591930), UINT64_C(11469762365481118451), UINT64_C( 2921757168886570143), UINT64_C(12447831939719752834),
        UINT64_C( 7260247287519336862), UINT64_C(14823787487486306046), UINT64_C(15338298609045612297), UINT64_C(15015055251481992577) },
      { UINT64_C(  603653916778526090), UINT64_C(10880654425188327683), UINT64_C( 2172466323090179841), UINT64_C( 6108624688056998083),
        UINT64_C( 4663671193139716519), UINT64_C(13021438911752362385), UINT64_C(16731766677901344632), UINT64_C(15988803656344274117) },
      { UINT64_C(13222032409913591930), UINT64_C(                   0), UINT64_C(                   0), UINT64_C(                   0),
        UINT64_C(                   0), UINT64_C(14823787487486306046), UINT64_C(                   0), UINT64_C(                   0) } },
    { UINT8_C( 69),
      { UINT64_C( 2266932696683023667), UINT64_C( 5268796264959611957), UINT64_C( 7746005982826607501), UINT64_C( 2602657211870154293),
        UINT64_C(10984698831648104459), UINT64_C( 6314994226182374161), UINT64_C(11412843190501315216), UINT64_C(10139174519801578399) },
      { UINT64_C( 5418198939811626293), UINT64_C( 6751821412729974930), UINT64_C(17609837972454425691), UINT64_C(11801464494138644921),
        UINT64_C( 9690461018278110710), UINT64_C(14197763210179977694), UINT64_C(14818227143754472795), UINT64_C( 6401249366518474948) },
      { UINT64_C( 5418198939811626293), UINT64_C(                   0), UINT64_C(17609837972454425691), UINT64_C(                   0),
        UINT64_C(                   0), UINT64_C(                   0), UINT64_C(14818227143754472795), UINT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k  = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_epu64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_epu64");
    easysimd_test_x86_assert_equal_u64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_max_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -467.49), EASYSIMD_FLOAT32_C(    68.96), EASYSIMD_FLOAT32_C(    93.32), EASYSIMD_FLOAT32_C(  -192.23),
        EASYSIMD_FLOAT32_C(   206.98), EASYSIMD_FLOAT32_C(   -60.64), EASYSIMD_FLOAT32_C(   236.48), EASYSIMD_FLOAT32_C(  -938.37),
        EASYSIMD_FLOAT32_C(   629.22), EASYSIMD_FLOAT32_C(  -771.32), EASYSIMD_FLOAT32_C(  -922.17), EASYSIMD_FLOAT32_C(  -910.14),
        EASYSIMD_FLOAT32_C(  -100.30), EASYSIMD_FLOAT32_C(   480.89), EASYSIMD_FLOAT32_C(  -423.93), EASYSIMD_FLOAT32_C(   266.34) },
      { EASYSIMD_FLOAT32_C(  -755.61), EASYSIMD_FLOAT32_C(  -797.61), EASYSIMD_FLOAT32_C(  -135.50), EASYSIMD_FLOAT32_C(  -267.60),
        EASYSIMD_FLOAT32_C(  -951.29), EASYSIMD_FLOAT32_C(   951.42), EASYSIMD_FLOAT32_C(   213.79), EASYSIMD_FLOAT32_C(   234.83),
        EASYSIMD_FLOAT32_C(   263.79), EASYSIMD_FLOAT32_C(   144.04), EASYSIMD_FLOAT32_C(   457.99), EASYSIMD_FLOAT32_C(  -680.15),
        EASYSIMD_FLOAT32_C(  -615.84), EASYSIMD_FLOAT32_C(   601.67), EASYSIMD_FLOAT32_C(   458.35), EASYSIMD_FLOAT32_C(   -83.34) },
      { EASYSIMD_FLOAT32_C(  -467.49), EASYSIMD_FLOAT32_C(    68.96), EASYSIMD_FLOAT32_C(    93.32), EASYSIMD_FLOAT32_C(  -192.23),
        EASYSIMD_FLOAT32_C(   206.98), EASYSIMD_FLOAT32_C(   951.42), EASYSIMD_FLOAT32_C(   236.48), EASYSIMD_FLOAT32_C(   234.83),
        EASYSIMD_FLOAT32_C(   629.22), EASYSIMD_FLOAT32_C(   144.04), EASYSIMD_FLOAT32_C(   457.99), EASYSIMD_FLOAT32_C(  -680.15),
        EASYSIMD_FLOAT32_C(  -100.30), EASYSIMD_FLOAT32_C(   601.67), EASYSIMD_FLOAT32_C(   458.35), EASYSIMD_FLOAT32_C(   266.34) } },
    { { EASYSIMD_FLOAT32_C(  -329.37), EASYSIMD_FLOAT32_C(  -448.33), EASYSIMD_FLOAT32_C(   724.43), EASYSIMD_FLOAT32_C(   877.61),
        EASYSIMD_FLOAT32_C(   491.03), EASYSIMD_FLOAT32_C(   -39.09), EASYSIMD_FLOAT32_C(   939.24), EASYSIMD_FLOAT32_C(   120.25),
        EASYSIMD_FLOAT32_C(   189.59), EASYSIMD_FLOAT32_C(  -982.93), EASYSIMD_FLOAT32_C(   210.11), EASYSIMD_FLOAT32_C(  -910.71),
        EASYSIMD_FLOAT32_C(   497.97), EASYSIMD_FLOAT32_C(   786.19), EASYSIMD_FLOAT32_C(   355.63), EASYSIMD_FLOAT32_C(   742.36) },
      { EASYSIMD_FLOAT32_C(   988.58), EASYSIMD_FLOAT32_C(  -779.87), EASYSIMD_FLOAT32_C(  -525.24), EASYSIMD_FLOAT32_C(  -962.71),
        EASYSIMD_FLOAT32_C(  -828.45), EASYSIMD_FLOAT32_C(   688.56), EASYSIMD_FLOAT32_C(   272.12), EASYSIMD_FLOAT32_C(   435.34),
        EASYSIMD_FLOAT32_C(  -167.41), EASYSIMD_FLOAT32_C(  -269.90), EASYSIMD_FLOAT32_C(   755.19), EASYSIMD_FLOAT32_C(   216.75),
        EASYSIMD_FLOAT32_C(  -668.23), EASYSIMD_FLOAT32_C(   213.55), EASYSIMD_FLOAT32_C(  -866.59), EASYSIMD_FLOAT32_C(     2.41) },
      { EASYSIMD_FLOAT32_C(   988.58), EASYSIMD_FLOAT32_C(  -448.33), EASYSIMD_FLOAT32_C(   724.43), EASYSIMD_FLOAT32_C(   877.61),
        EASYSIMD_FLOAT32_C(   491.03), EASYSIMD_FLOAT32_C(   688.56), EASYSIMD_FLOAT32_C(   939.24), EASYSIMD_FLOAT32_C(   435.34),
        EASYSIMD_FLOAT32_C(   189.59), EASYSIMD_FLOAT32_C(  -269.90), EASYSIMD_FLOAT32_C(   755.19), EASYSIMD_FLOAT32_C(   216.75),
        EASYSIMD_FLOAT32_C(   497.97), EASYSIMD_FLOAT32_C(   786.19), EASYSIMD_FLOAT32_C(   355.63), EASYSIMD_FLOAT32_C(   742.36) } },
    { { EASYSIMD_FLOAT32_C(   765.22), EASYSIMD_FLOAT32_C(   857.85), EASYSIMD_FLOAT32_C(  -119.98), EASYSIMD_FLOAT32_C(   256.25),
        EASYSIMD_FLOAT32_C(  -181.25), EASYSIMD_FLOAT32_C(  -180.73), EASYSIMD_FLOAT32_C(  -623.50), EASYSIMD_FLOAT32_C(  -991.66),
        EASYSIMD_FLOAT32_C(  -163.66), EASYSIMD_FLOAT32_C(   586.61), EASYSIMD_FLOAT32_C(  -902.37), EASYSIMD_FLOAT32_C(  -665.69),
        EASYSIMD_FLOAT32_C(   372.80), EASYSIMD_FLOAT32_C(   453.26), EASYSIMD_FLOAT32_C(  -923.33), EASYSIMD_FLOAT32_C(   361.38) },
      { EASYSIMD_FLOAT32_C(   673.39), EASYSIMD_FLOAT32_C(  -448.57), EASYSIMD_FLOAT32_C(   398.67), EASYSIMD_FLOAT32_C(   844.95),
        EASYSIMD_FLOAT32_C(  -760.02), EASYSIMD_FLOAT32_C(  -329.21), EASYSIMD_FLOAT32_C(   280.29), EASYSIMD_FLOAT32_C(    72.58),
        EASYSIMD_FLOAT32_C(   400.89), EASYSIMD_FLOAT32_C(    35.48), EASYSIMD_FLOAT32_C(  -710.67), EASYSIMD_FLOAT32_C(   732.67),
        EASYSIMD_FLOAT32_C(  -750.97), EASYSIMD_FLOAT32_C(  -577.26), EASYSIMD_FLOAT32_C(  -264.92), EASYSIMD_FLOAT32_C(  -985.75) },
      { EASYSIMD_FLOAT32_C(   765.22), EASYSIMD_FLOAT32_C(   857.85), EASYSIMD_FLOAT32_C(   398.67), EASYSIMD_FLOAT32_C(   844.95),
        EASYSIMD_FLOAT32_C(  -181.25), EASYSIMD_FLOAT32_C(  -180.73), EASYSIMD_FLOAT32_C(   280.29), EASYSIMD_FLOAT32_C(    72.58),
        EASYSIMD_FLOAT32_C(   400.89), EASYSIMD_FLOAT32_C(   586.61), EASYSIMD_FLOAT32_C(  -710.67), EASYSIMD_FLOAT32_C(   732.67),
        EASYSIMD_FLOAT32_C(   372.80), EASYSIMD_FLOAT32_C(   453.26), EASYSIMD_FLOAT32_C(  -264.92), EASYSIMD_FLOAT32_C(   361.38) } },
    { { EASYSIMD_FLOAT32_C(  -719.41), EASYSIMD_FLOAT32_C(   615.10), EASYSIMD_FLOAT32_C(   270.50), EASYSIMD_FLOAT32_C(    99.34),
        EASYSIMD_FLOAT32_C(  -565.63), EASYSIMD_FLOAT32_C(   647.00), EASYSIMD_FLOAT32_C(   107.68), EASYSIMD_FLOAT32_C(   270.71),
        EASYSIMD_FLOAT32_C(   233.61), EASYSIMD_FLOAT32_C(   205.31), EASYSIMD_FLOAT32_C(   605.02), EASYSIMD_FLOAT32_C(  -393.59),
        EASYSIMD_FLOAT32_C(  -341.43), EASYSIMD_FLOAT32_C(   681.68), EASYSIMD_FLOAT32_C(   967.80), EASYSIMD_FLOAT32_C(  -668.04) },
      { EASYSIMD_FLOAT32_C(  -766.89), EASYSIMD_FLOAT32_C(   366.47), EASYSIMD_FLOAT32_C(  -823.10), EASYSIMD_FLOAT32_C(  -526.90),
        EASYSIMD_FLOAT32_C(  -962.74), EASYSIMD_FLOAT32_C(   457.19), EASYSIMD_FLOAT32_C(   545.67), EASYSIMD_FLOAT32_C(   438.16),
        EASYSIMD_FLOAT32_C(  -507.32), EASYSIMD_FLOAT32_C(   835.00), EASYSIMD_FLOAT32_C(   170.82), EASYSIMD_FLOAT32_C(  -258.30),
        EASYSIMD_FLOAT32_C(  -742.26), EASYSIMD_FLOAT32_C(   905.90), EASYSIMD_FLOAT32_C(  -244.05), EASYSIMD_FLOAT32_C(  -461.67) },
      { EASYSIMD_FLOAT32_C(  -719.41), EASYSIMD_FLOAT32_C(   615.10), EASYSIMD_FLOAT32_C(   270.50), EASYSIMD_FLOAT32_C(    99.34),
        EASYSIMD_FLOAT32_C(  -565.63), EASYSIMD_FLOAT32_C(   647.00), EASYSIMD_FLOAT32_C(   545.67), EASYSIMD_FLOAT32_C(   438.16),
        EASYSIMD_FLOAT32_C(   233.61), EASYSIMD_FLOAT32_C(   835.00), EASYSIMD_FLOAT32_C(   605.02), EASYSIMD_FLOAT32_C(  -258.30),
        EASYSIMD_FLOAT32_C(  -341.43), EASYSIMD_FLOAT32_C(   905.90), EASYSIMD_FLOAT32_C(   967.80), EASYSIMD_FLOAT32_C(  -461.67) } },
    { { EASYSIMD_FLOAT32_C(   521.00), EASYSIMD_FLOAT32_C(  -973.55), EASYSIMD_FLOAT32_C(   637.67), EASYSIMD_FLOAT32_C(   955.37),
        EASYSIMD_FLOAT32_C(   673.44), EASYSIMD_FLOAT32_C(  -254.65), EASYSIMD_FLOAT32_C(   226.08), EASYSIMD_FLOAT32_C(   -92.95),
        EASYSIMD_FLOAT32_C(   950.66), EASYSIMD_FLOAT32_C(  -168.90), EASYSIMD_FLOAT32_C(   513.47), EASYSIMD_FLOAT32_C(  -390.77),
        EASYSIMD_FLOAT32_C(  -487.22), EASYSIMD_FLOAT32_C(   481.27), EASYSIMD_FLOAT32_C(   -58.81), EASYSIMD_FLOAT32_C(  -254.11) },
      { EASYSIMD_FLOAT32_C(  -152.26), EASYSIMD_FLOAT32_C(   118.09), EASYSIMD_FLOAT32_C(   218.99), EASYSIMD_FLOAT32_C(  -115.00),
        EASYSIMD_FLOAT32_C(  -424.72), EASYSIMD_FLOAT32_C(  -235.34), EASYSIMD_FLOAT32_C(  -676.84), EASYSIMD_FLOAT32_C(    67.96),
        EASYSIMD_FLOAT32_C(  -400.33), EASYSIMD_FLOAT32_C(   493.98), EASYSIMD_FLOAT32_C(   809.66), EASYSIMD_FLOAT32_C(  -142.59),
        EASYSIMD_FLOAT32_C(   399.88), EASYSIMD_FLOAT32_C(  -434.39), EASYSIMD_FLOAT32_C(   395.74), EASYSIMD_FLOAT32_C(   -79.11) },
      { EASYSIMD_FLOAT32_C(   521.00), EASYSIMD_FLOAT32_C(   118.09), EASYSIMD_FLOAT32_C(   637.67), EASYSIMD_FLOAT32_C(   955.37),
        EASYSIMD_FLOAT32_C(   673.44), EASYSIMD_FLOAT32_C(  -235.34), EASYSIMD_FLOAT32_C(   226.08), EASYSIMD_FLOAT32_C(    67.96),
        EASYSIMD_FLOAT32_C(   950.66), EASYSIMD_FLOAT32_C(   493.98), EASYSIMD_FLOAT32_C(   809.66), EASYSIMD_FLOAT32_C(  -142.59),
        EASYSIMD_FLOAT32_C(   399.88), EASYSIMD_FLOAT32_C(   481.27), EASYSIMD_FLOAT32_C(   395.74), EASYSIMD_FLOAT32_C(   -79.11) } },
    { { EASYSIMD_FLOAT32_C(  -407.94), EASYSIMD_FLOAT32_C(    33.41), EASYSIMD_FLOAT32_C(  -123.74), EASYSIMD_FLOAT32_C(  -734.49),
        EASYSIMD_FLOAT32_C(   778.76), EASYSIMD_FLOAT32_C(  -897.66), EASYSIMD_FLOAT32_C(   172.56), EASYSIMD_FLOAT32_C(   729.42),
        EASYSIMD_FLOAT32_C(   -66.56), EASYSIMD_FLOAT32_C(  -313.97), EASYSIMD_FLOAT32_C(  -661.35), EASYSIMD_FLOAT32_C(   446.22),
        EASYSIMD_FLOAT32_C(  -832.70), EASYSIMD_FLOAT32_C(   279.83), EASYSIMD_FLOAT32_C(  -807.89), EASYSIMD_FLOAT32_C(    15.04) },
      { EASYSIMD_FLOAT32_C(  -602.07), EASYSIMD_FLOAT32_C(   411.10), EASYSIMD_FLOAT32_C(   900.04), EASYSIMD_FLOAT32_C(   -26.79),
        EASYSIMD_FLOAT32_C(  -824.23), EASYSIMD_FLOAT32_C(  -776.81), EASYSIMD_FLOAT32_C(  -958.83), EASYSIMD_FLOAT32_C(  -224.57),
        EASYSIMD_FLOAT32_C(   717.17), EASYSIMD_FLOAT32_C(   850.83), EASYSIMD_FLOAT32_C(   632.84), EASYSIMD_FLOAT32_C(   117.06),
        EASYSIMD_FLOAT32_C(  -583.55), EASYSIMD_FLOAT32_C(    28.58), EASYSIMD_FLOAT32_C(  -962.05), EASYSIMD_FLOAT32_C(     8.51) },
      { EASYSIMD_FLOAT32_C(  -407.94), EASYSIMD_FLOAT32_C(   411.10), EASYSIMD_FLOAT32_C(   900.04), EASYSIMD_FLOAT32_C(   -26.79),
        EASYSIMD_FLOAT32_C(   778.76), EASYSIMD_FLOAT32_C(  -776.81), EASYSIMD_FLOAT32_C(   172.56), EASYSIMD_FLOAT32_C(   729.42),
        EASYSIMD_FLOAT32_C(   717.17), EASYSIMD_FLOAT32_C(   850.83), EASYSIMD_FLOAT32_C(   632.84), EASYSIMD_FLOAT32_C(   446.22),
        EASYSIMD_FLOAT32_C(  -583.55), EASYSIMD_FLOAT32_C(   279.83), EASYSIMD_FLOAT32_C(  -807.89), EASYSIMD_FLOAT32_C(    15.04) } },
    { { EASYSIMD_FLOAT32_C(  -938.01), EASYSIMD_FLOAT32_C(   -85.80), EASYSIMD_FLOAT32_C(   274.02), EASYSIMD_FLOAT32_C(   840.75),
        EASYSIMD_FLOAT32_C(    16.55), EASYSIMD_FLOAT32_C(  -553.42), EASYSIMD_FLOAT32_C(   570.17), EASYSIMD_FLOAT32_C(   949.99),
        EASYSIMD_FLOAT32_C(   132.61), EASYSIMD_FLOAT32_C(   908.82), EASYSIMD_FLOAT32_C(   396.21), EASYSIMD_FLOAT32_C(   299.91),
        EASYSIMD_FLOAT32_C(   188.66), EASYSIMD_FLOAT32_C(   588.32), EASYSIMD_FLOAT32_C(  -685.05), EASYSIMD_FLOAT32_C(   586.58) },
      { EASYSIMD_FLOAT32_C(    -0.57), EASYSIMD_FLOAT32_C(  -785.02), EASYSIMD_FLOAT32_C(  -440.21), EASYSIMD_FLOAT32_C(   175.19),
        EASYSIMD_FLOAT32_C(  -561.82), EASYSIMD_FLOAT32_C(  -399.04), EASYSIMD_FLOAT32_C(   950.62), EASYSIMD_FLOAT32_C(  -844.65),
        EASYSIMD_FLOAT32_C(  -548.21), EASYSIMD_FLOAT32_C(   583.46), EASYSIMD_FLOAT32_C(   272.41), EASYSIMD_FLOAT32_C(  -131.76),
        EASYSIMD_FLOAT32_C(  -387.96), EASYSIMD_FLOAT32_C(   310.36), EASYSIMD_FLOAT32_C(   876.75), EASYSIMD_FLOAT32_C(  -325.97) },
      { EASYSIMD_FLOAT32_C(    -0.57), EASYSIMD_FLOAT32_C(   -85.80), EASYSIMD_FLOAT32_C(   274.02), EASYSIMD_FLOAT32_C(   840.75),
        EASYSIMD_FLOAT32_C(    16.55), EASYSIMD_FLOAT32_C(  -399.04), EASYSIMD_FLOAT32_C(   950.62), EASYSIMD_FLOAT32_C(   949.99),
        EASYSIMD_FLOAT32_C(   132.61), EASYSIMD_FLOAT32_C(   908.82), EASYSIMD_FLOAT32_C(   396.21), EASYSIMD_FLOAT32_C(   299.91),
        EASYSIMD_FLOAT32_C(   188.66), EASYSIMD_FLOAT32_C(   588.32), EASYSIMD_FLOAT32_C(   876.75), EASYSIMD_FLOAT32_C(   586.58) } },
    { { EASYSIMD_FLOAT32_C(  -775.44), EASYSIMD_FLOAT32_C(   150.76), EASYSIMD_FLOAT32_C(  -485.21), EASYSIMD_FLOAT32_C(   241.11),
        EASYSIMD_FLOAT32_C(   597.34), EASYSIMD_FLOAT32_C(  -915.04), EASYSIMD_FLOAT32_C(   191.10), EASYSIMD_FLOAT32_C(  -270.05),
        EASYSIMD_FLOAT32_C(   993.78), EASYSIMD_FLOAT32_C(  -412.69), EASYSIMD_FLOAT32_C(  -970.14), EASYSIMD_FLOAT32_C(   182.44),
        EASYSIMD_FLOAT32_C(  -824.37), EASYSIMD_FLOAT32_C(  -655.20), EASYSIMD_FLOAT32_C(  -230.98), EASYSIMD_FLOAT32_C(   175.06) },
      { EASYSIMD_FLOAT32_C(  -440.21), EASYSIMD_FLOAT32_C(   328.81), EASYSIMD_FLOAT32_C(  -649.75), EASYSIMD_FLOAT32_C(    -2.03),
        EASYSIMD_FLOAT32_C(   929.77), EASYSIMD_FLOAT32_C(  -699.13), EASYSIMD_FLOAT32_C(   153.32), EASYSIMD_FLOAT32_C(  -618.43),
        EASYSIMD_FLOAT32_C(   884.33), EASYSIMD_FLOAT32_C(  -574.27), EASYSIMD_FLOAT32_C(   249.80), EASYSIMD_FLOAT32_C(  -503.62),
        EASYSIMD_FLOAT32_C(   736.09), EASYSIMD_FLOAT32_C(   126.55), EASYSIMD_FLOAT32_C(   170.41), EASYSIMD_FLOAT32_C(   960.65) },
      { EASYSIMD_FLOAT32_C(  -440.21), EASYSIMD_FLOAT32_C(   328.81), EASYSIMD_FLOAT32_C(  -485.21), EASYSIMD_FLOAT32_C(   241.11),
        EASYSIMD_FLOAT32_C(   929.77), EASYSIMD_FLOAT32_C(  -699.13), EASYSIMD_FLOAT32_C(   191.10), EASYSIMD_FLOAT32_C(  -270.05),
        EASYSIMD_FLOAT32_C(   993.78), EASYSIMD_FLOAT32_C(  -412.69), EASYSIMD_FLOAT32_C(   249.80), EASYSIMD_FLOAT32_C(   182.44),
        EASYSIMD_FLOAT32_C(   736.09), EASYSIMD_FLOAT32_C(   126.55), EASYSIMD_FLOAT32_C(   170.41), EASYSIMD_FLOAT32_C(   960.65) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_ps(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 src[16];
    const easysimd__mmask8 k;
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   -88.50), EASYSIMD_FLOAT32_C(   -78.42), EASYSIMD_FLOAT32_C(   962.39), EASYSIMD_FLOAT32_C(   987.97),
        EASYSIMD_FLOAT32_C(  -302.96), EASYSIMD_FLOAT32_C(   654.54), EASYSIMD_FLOAT32_C(  -803.66), EASYSIMD_FLOAT32_C(   -72.57),
        EASYSIMD_FLOAT32_C(    33.69), EASYSIMD_FLOAT32_C(  -319.65), EASYSIMD_FLOAT32_C(  -278.97), EASYSIMD_FLOAT32_C(   -83.08),
        EASYSIMD_FLOAT32_C(   959.63), EASYSIMD_FLOAT32_C(   923.05), EASYSIMD_FLOAT32_C(  -533.10), EASYSIMD_FLOAT32_C(   171.10) },
      UINT8_C(175),
      { EASYSIMD_FLOAT32_C(    55.01), EASYSIMD_FLOAT32_C(   862.08), EASYSIMD_FLOAT32_C(   740.08), EASYSIMD_FLOAT32_C(   496.31),
        EASYSIMD_FLOAT32_C(   640.80), EASYSIMD_FLOAT32_C(   251.55), EASYSIMD_FLOAT32_C(   355.76), EASYSIMD_FLOAT32_C(  -259.95),
        EASYSIMD_FLOAT32_C(   393.94), EASYSIMD_FLOAT32_C(   515.66), EASYSIMD_FLOAT32_C(   507.69), EASYSIMD_FLOAT32_C(  -319.00),
        EASYSIMD_FLOAT32_C(   639.33), EASYSIMD_FLOAT32_C(   388.01), EASYSIMD_FLOAT32_C(   592.50), EASYSIMD_FLOAT32_C(  -439.09) },
      { EASYSIMD_FLOAT32_C(   350.40), EASYSIMD_FLOAT32_C(   580.47), EASYSIMD_FLOAT32_C(   257.94), EASYSIMD_FLOAT32_C(     4.94),
        EASYSIMD_FLOAT32_C(   776.81), EASYSIMD_FLOAT32_C(  -814.62), EASYSIMD_FLOAT32_C(  -961.37), EASYSIMD_FLOAT32_C(  -542.84),
        EASYSIMD_FLOAT32_C(   -93.60), EASYSIMD_FLOAT32_C(   -44.45), EASYSIMD_FLOAT32_C(  -583.22), EASYSIMD_FLOAT32_C(  -170.54),
        EASYSIMD_FLOAT32_C(   422.45), EASYSIMD_FLOAT32_C(   587.88), EASYSIMD_FLOAT32_C(   859.23), EASYSIMD_FLOAT32_C(  -522.55) },
      { EASYSIMD_FLOAT32_C(   350.40), EASYSIMD_FLOAT32_C(   862.08), EASYSIMD_FLOAT32_C(   740.08), EASYSIMD_FLOAT32_C(   496.31),
        EASYSIMD_FLOAT32_C(  -302.96), EASYSIMD_FLOAT32_C(   251.55), EASYSIMD_FLOAT32_C(  -803.66), EASYSIMD_FLOAT32_C(  -259.95),
        EASYSIMD_FLOAT32_C(    33.69), EASYSIMD_FLOAT32_C(  -319.65), EASYSIMD_FLOAT32_C(  -278.97), EASYSIMD_FLOAT32_C(   -83.08),
        EASYSIMD_FLOAT32_C(   959.63), EASYSIMD_FLOAT32_C(   923.05), EASYSIMD_FLOAT32_C(  -533.10), EASYSIMD_FLOAT32_C(   171.10) } },
    { { EASYSIMD_FLOAT32_C(   449.96), EASYSIMD_FLOAT32_C(   599.32), EASYSIMD_FLOAT32_C(   973.77), EASYSIMD_FLOAT32_C(    90.75),
        EASYSIMD_FLOAT32_C(  -149.14), EASYSIMD_FLOAT32_C(   329.53), EASYSIMD_FLOAT32_C(   830.80), EASYSIMD_FLOAT32_C(  -755.20),
        EASYSIMD_FLOAT32_C(  -154.81), EASYSIMD_FLOAT32_C(   338.49), EASYSIMD_FLOAT32_C(   -74.20), EASYSIMD_FLOAT32_C(  -515.48),
        EASYSIMD_FLOAT32_C(  -273.51), EASYSIMD_FLOAT32_C(  -481.70), EASYSIMD_FLOAT32_C(    45.43), EASYSIMD_FLOAT32_C(  -923.11) },
      UINT8_C(180),
      { EASYSIMD_FLOAT32_C(  -696.63), EASYSIMD_FLOAT32_C(    81.84), EASYSIMD_FLOAT32_C(   875.57), EASYSIMD_FLOAT32_C(  -511.25),
        EASYSIMD_FLOAT32_C(   120.47), EASYSIMD_FLOAT32_C(  -667.27), EASYSIMD_FLOAT32_C(   395.15), EASYSIMD_FLOAT32_C(  -923.99),
        EASYSIMD_FLOAT32_C(  -250.49), EASYSIMD_FLOAT32_C(  -775.40), EASYSIMD_FLOAT32_C(   498.46), EASYSIMD_FLOAT32_C(  -662.62),
        EASYSIMD_FLOAT32_C(  -916.16), EASYSIMD_FLOAT32_C(   975.92), EASYSIMD_FLOAT32_C(   787.34), EASYSIMD_FLOAT32_C(   683.15) },
      { EASYSIMD_FLOAT32_C(   949.68), EASYSIMD_FLOAT32_C(  -121.91), EASYSIMD_FLOAT32_C(  -465.99), EASYSIMD_FLOAT32_C(   279.22),
        EASYSIMD_FLOAT32_C(  -291.11), EASYSIMD_FLOAT32_C(  -221.19), EASYSIMD_FLOAT32_C(  -875.60), EASYSIMD_FLOAT32_C(  -952.62),
        EASYSIMD_FLOAT32_C(   704.61), EASYSIMD_FLOAT32_C(  -391.08), EASYSIMD_FLOAT32_C(  -226.12), EASYSIMD_FLOAT32_C(  -777.10),
        EASYSIMD_FLOAT32_C(   654.35), EASYSIMD_FLOAT32_C(  -149.23), EASYSIMD_FLOAT32_C(  -678.33), EASYSIMD_FLOAT32_C(   957.72) },
      { EASYSIMD_FLOAT32_C(   449.96), EASYSIMD_FLOAT32_C(   599.32), EASYSIMD_FLOAT32_C(   875.57), EASYSIMD_FLOAT32_C(    90.75),
        EASYSIMD_FLOAT32_C(   120.47), EASYSIMD_FLOAT32_C(  -221.19), EASYSIMD_FLOAT32_C(   830.80), EASYSIMD_FLOAT32_C(  -923.99),
        EASYSIMD_FLOAT32_C(  -154.81), EASYSIMD_FLOAT32_C(   338.49), EASYSIMD_FLOAT32_C(   -74.20), EASYSIMD_FLOAT32_C(  -515.48),
        EASYSIMD_FLOAT32_C(  -273.51), EASYSIMD_FLOAT32_C(  -481.70), EASYSIMD_FLOAT32_C(    45.43), EASYSIMD_FLOAT32_C(  -923.11) } },
    { { EASYSIMD_FLOAT32_C(   932.61), EASYSIMD_FLOAT32_C(  -802.76), EASYSIMD_FLOAT32_C(  -553.53), EASYSIMD_FLOAT32_C(    53.07),
        EASYSIMD_FLOAT32_C(  -470.04), EASYSIMD_FLOAT32_C(   841.61), EASYSIMD_FLOAT32_C(   129.09), EASYSIMD_FLOAT32_C(   279.47),
        EASYSIMD_FLOAT32_C(  -933.78), EASYSIMD_FLOAT32_C(  -372.45), EASYSIMD_FLOAT32_C(   616.85), EASYSIMD_FLOAT32_C(  -849.95),
        EASYSIMD_FLOAT32_C(  -396.53), EASYSIMD_FLOAT32_C(   404.19), EASYSIMD_FLOAT32_C(   833.21), EASYSIMD_FLOAT32_C(  -446.85) },
      UINT8_C( 58),
      { EASYSIMD_FLOAT32_C(  -632.78), EASYSIMD_FLOAT32_C(   832.37), EASYSIMD_FLOAT32_C(    -8.82), EASYSIMD_FLOAT32_C(   146.03),
        EASYSIMD_FLOAT32_C(   956.77), EASYSIMD_FLOAT32_C(    38.57), EASYSIMD_FLOAT32_C(  -149.36), EASYSIMD_FLOAT32_C(  -434.30),
        EASYSIMD_FLOAT32_C(   812.44), EASYSIMD_FLOAT32_C(    73.54), EASYSIMD_FLOAT32_C(  -779.95), EASYSIMD_FLOAT32_C(  -336.78),
        EASYSIMD_FLOAT32_C(   395.21), EASYSIMD_FLOAT32_C(  -822.23), EASYSIMD_FLOAT32_C(  -404.17), EASYSIMD_FLOAT32_C(   592.45) },
      { EASYSIMD_FLOAT32_C(  -375.77), EASYSIMD_FLOAT32_C(   648.90), EASYSIMD_FLOAT32_C(  -877.58), EASYSIMD_FLOAT32_C(  -534.15),
        EASYSIMD_FLOAT32_C(  -222.01), EASYSIMD_FLOAT32_C(   401.89), EASYSIMD_FLOAT32_C(  -467.94), EASYSIMD_FLOAT32_C(   405.54),
        EASYSIMD_FLOAT32_C(    18.74), EASYSIMD_FLOAT32_C(  -317.88), EASYSIMD_FLOAT32_C(  -990.99), EASYSIMD_FLOAT32_C(  -577.06),
        EASYSIMD_FLOAT32_C(  -484.68), EASYSIMD_FLOAT32_C(  -437.84), EASYSIMD_FLOAT32_C(  -294.78), EASYSIMD_FLOAT32_C(  -117.46) },
      { EASYSIMD_FLOAT32_C(   932.61), EASYSIMD_FLOAT32_C(   832.37), EASYSIMD_FLOAT32_C(  -553.53), EASYSIMD_FLOAT32_C(   146.03),
        EASYSIMD_FLOAT32_C(   956.77), EASYSIMD_FLOAT32_C(   401.89), EASYSIMD_FLOAT32_C(   129.09), EASYSIMD_FLOAT32_C(   279.47),
        EASYSIMD_FLOAT32_C(  -933.78), EASYSIMD_FLOAT32_C(  -372.45), EASYSIMD_FLOAT32_C(   616.85), EASYSIMD_FLOAT32_C(  -849.95),
        EASYSIMD_FLOAT32_C(  -396.53), EASYSIMD_FLOAT32_C(   404.19), EASYSIMD_FLOAT32_C(   833.21), EASYSIMD_FLOAT32_C(  -446.85) } },
    { { EASYSIMD_FLOAT32_C(  -605.48), EASYSIMD_FLOAT32_C(   696.41), EASYSIMD_FLOAT32_C(  -971.43), EASYSIMD_FLOAT32_C(  -648.70),
        EASYSIMD_FLOAT32_C(  -265.03), EASYSIMD_FLOAT32_C(  -120.79), EASYSIMD_FLOAT32_C(   -83.01), EASYSIMD_FLOAT32_C(  -452.58),
        EASYSIMD_FLOAT32_C(   952.75), EASYSIMD_FLOAT32_C(   137.04), EASYSIMD_FLOAT32_C(   210.63), EASYSIMD_FLOAT32_C(   347.96),
        EASYSIMD_FLOAT32_C(   314.80), EASYSIMD_FLOAT32_C(   806.46), EASYSIMD_FLOAT32_C(   -59.59), EASYSIMD_FLOAT32_C(   939.04) },
      UINT8_C( 95),
      { EASYSIMD_FLOAT32_C(    62.83), EASYSIMD_FLOAT32_C(  -595.12), EASYSIMD_FLOAT32_C(  -766.65), EASYSIMD_FLOAT32_C(  -535.28),
        EASYSIMD_FLOAT32_C(   -63.05), EASYSIMD_FLOAT32_C(   638.89), EASYSIMD_FLOAT32_C(   483.46), EASYSIMD_FLOAT32_C(   619.06),
        EASYSIMD_FLOAT32_C(   647.90), EASYSIMD_FLOAT32_C(   906.39), EASYSIMD_FLOAT32_C(  -865.61), EASYSIMD_FLOAT32_C(  -789.95),
        EASYSIMD_FLOAT32_C(  -388.38), EASYSIMD_FLOAT32_C(    16.93), EASYSIMD_FLOAT32_C(  -395.42), EASYSIMD_FLOAT32_C(  -691.98) },
      { EASYSIMD_FLOAT32_C(    45.50), EASYSIMD_FLOAT32_C(   -44.13), EASYSIMD_FLOAT32_C(    42.99), EASYSIMD_FLOAT32_C(   924.71),
        EASYSIMD_FLOAT32_C(   872.86), EASYSIMD_FLOAT32_C(   590.41), EASYSIMD_FLOAT32_C(   877.46), EASYSIMD_FLOAT32_C(     9.90),
        EASYSIMD_FLOAT32_C(  -198.96), EASYSIMD_FLOAT32_C(   225.42), EASYSIMD_FLOAT32_C(  -675.30), EASYSIMD_FLOAT32_C(  -392.50),
        EASYSIMD_FLOAT32_C(  -834.17), EASYSIMD_FLOAT32_C(  -736.26), EASYSIMD_FLOAT32_C(  -937.14), EASYSIMD_FLOAT32_C(   228.66) },
      { EASYSIMD_FLOAT32_C(    62.83), EASYSIMD_FLOAT32_C(   -44.13), EASYSIMD_FLOAT32_C(    42.99), EASYSIMD_FLOAT32_C(   924.71),
        EASYSIMD_FLOAT32_C(   872.86), EASYSIMD_FLOAT32_C(  -120.79), EASYSIMD_FLOAT32_C(   877.46), EASYSIMD_FLOAT32_C(  -452.58),
        EASYSIMD_FLOAT32_C(   952.75), EASYSIMD_FLOAT32_C(   137.04), EASYSIMD_FLOAT32_C(   210.63), EASYSIMD_FLOAT32_C(   347.96),
        EASYSIMD_FLOAT32_C(   314.80), EASYSIMD_FLOAT32_C(   806.46), EASYSIMD_FLOAT32_C(   -59.59), EASYSIMD_FLOAT32_C(   939.04) } },
    { { EASYSIMD_FLOAT32_C(  -331.37), EASYSIMD_FLOAT32_C(  -703.79), EASYSIMD_FLOAT32_C(   693.38), EASYSIMD_FLOAT32_C(   605.57),
        EASYSIMD_FLOAT32_C(   935.10), EASYSIMD_FLOAT32_C(   176.83), EASYSIMD_FLOAT32_C(   224.64), EASYSIMD_FLOAT32_C(   583.00),
        EASYSIMD_FLOAT32_C(    83.23), EASYSIMD_FLOAT32_C(   359.02), EASYSIMD_FLOAT32_C(   793.05), EASYSIMD_FLOAT32_C(   694.84),
        EASYSIMD_FLOAT32_C(  -624.05), EASYSIMD_FLOAT32_C(  -602.37), EASYSIMD_FLOAT32_C(  -997.13), EASYSIMD_FLOAT32_C(   421.45) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT32_C(    45.86), EASYSIMD_FLOAT32_C(   346.16), EASYSIMD_FLOAT32_C(   226.37), EASYSIMD_FLOAT32_C(  -363.73),
        EASYSIMD_FLOAT32_C(   223.61), EASYSIMD_FLOAT32_C(  -763.73), EASYSIMD_FLOAT32_C(   437.31), EASYSIMD_FLOAT32_C(  -550.97),
        EASYSIMD_FLOAT32_C(  -439.03), EASYSIMD_FLOAT32_C(  -955.19), EASYSIMD_FLOAT32_C(  -385.13), EASYSIMD_FLOAT32_C(  -175.29),
        EASYSIMD_FLOAT32_C(  -892.33), EASYSIMD_FLOAT32_C(   843.53), EASYSIMD_FLOAT32_C(   493.34), EASYSIMD_FLOAT32_C(  -596.12) },
      { EASYSIMD_FLOAT32_C(   536.91), EASYSIMD_FLOAT32_C(    98.91), EASYSIMD_FLOAT32_C(  -661.01), EASYSIMD_FLOAT32_C(  -286.26),
        EASYSIMD_FLOAT32_C(  -676.45), EASYSIMD_FLOAT32_C(   921.98), EASYSIMD_FLOAT32_C(   796.97), EASYSIMD_FLOAT32_C(   682.58),
        EASYSIMD_FLOAT32_C(   715.04), EASYSIMD_FLOAT32_C(   491.81), EASYSIMD_FLOAT32_C(  -941.47), EASYSIMD_FLOAT32_C(  -887.33),
        EASYSIMD_FLOAT32_C(   494.68), EASYSIMD_FLOAT32_C(   479.98), EASYSIMD_FLOAT32_C(   466.17), EASYSIMD_FLOAT32_C(  -459.46) },
      { EASYSIMD_FLOAT32_C(   536.91), EASYSIMD_FLOAT32_C(  -703.79), EASYSIMD_FLOAT32_C(   693.38), EASYSIMD_FLOAT32_C(   605.57),
        EASYSIMD_FLOAT32_C(   223.61), EASYSIMD_FLOAT32_C(   176.83), EASYSIMD_FLOAT32_C(   224.64), EASYSIMD_FLOAT32_C(   583.00),
        EASYSIMD_FLOAT32_C(    83.23), EASYSIMD_FLOAT32_C(   359.02), EASYSIMD_FLOAT32_C(   793.05), EASYSIMD_FLOAT32_C(   694.84),
        EASYSIMD_FLOAT32_C(  -624.05), EASYSIMD_FLOAT32_C(  -602.37), EASYSIMD_FLOAT32_C(  -997.13), EASYSIMD_FLOAT32_C(   421.45) } },
    { { EASYSIMD_FLOAT32_C(  -173.87), EASYSIMD_FLOAT32_C(  -307.46), EASYSIMD_FLOAT32_C(   176.81), EASYSIMD_FLOAT32_C(  -950.25),
        EASYSIMD_FLOAT32_C(   -71.19), EASYSIMD_FLOAT32_C(  -385.88), EASYSIMD_FLOAT32_C(  -501.22), EASYSIMD_FLOAT32_C(   489.78),
        EASYSIMD_FLOAT32_C(  -341.06), EASYSIMD_FLOAT32_C(   113.65), EASYSIMD_FLOAT32_C(  -685.50), EASYSIMD_FLOAT32_C(  -233.39),
        EASYSIMD_FLOAT32_C(   -42.82), EASYSIMD_FLOAT32_C(   807.84), EASYSIMD_FLOAT32_C(   170.49), EASYSIMD_FLOAT32_C(  -505.92) },
      UINT8_C(163),
      { EASYSIMD_FLOAT32_C(   509.48), EASYSIMD_FLOAT32_C(   207.82), EASYSIMD_FLOAT32_C(   230.31), EASYSIMD_FLOAT32_C(   431.47),
        EASYSIMD_FLOAT32_C(     4.79), EASYSIMD_FLOAT32_C(   -87.12), EASYSIMD_FLOAT32_C(   146.50), EASYSIMD_FLOAT32_C(  -503.40),
        EASYSIMD_FLOAT32_C(   -28.59), EASYSIMD_FLOAT32_C(   259.17), EASYSIMD_FLOAT32_C(   991.27), EASYSIMD_FLOAT32_C(  -548.61),
        EASYSIMD_FLOAT32_C(  -274.65), EASYSIMD_FLOAT32_C(  -468.19), EASYSIMD_FLOAT32_C(   277.53), EASYSIMD_FLOAT32_C(   417.89) },
      { EASYSIMD_FLOAT32_C(   708.62), EASYSIMD_FLOAT32_C(   327.28), EASYSIMD_FLOAT32_C(  -653.30), EASYSIMD_FLOAT32_C(  -677.26),
        EASYSIMD_FLOAT32_C(   826.06), EASYSIMD_FLOAT32_C(   836.49), EASYSIMD_FLOAT32_C(   -18.32), EASYSIMD_FLOAT32_C(   -60.30),
        EASYSIMD_FLOAT32_C(  -849.01), EASYSIMD_FLOAT32_C(   748.29), EASYSIMD_FLOAT32_C(   896.88), EASYSIMD_FLOAT32_C(   958.83),
        EASYSIMD_FLOAT32_C(   -81.22), EASYSIMD_FLOAT32_C(  -609.04), EASYSIMD_FLOAT32_C(  -134.42), EASYSIMD_FLOAT32_C(  -571.74) },
      { EASYSIMD_FLOAT32_C(   708.62), EASYSIMD_FLOAT32_C(   327.28), EASYSIMD_FLOAT32_C(   176.81), EASYSIMD_FLOAT32_C(  -950.25),
        EASYSIMD_FLOAT32_C(   -71.19), EASYSIMD_FLOAT32_C(   836.49), EASYSIMD_FLOAT32_C(  -501.22), EASYSIMD_FLOAT32_C(   -60.30),
        EASYSIMD_FLOAT32_C(  -341.06), EASYSIMD_FLOAT32_C(   113.65), EASYSIMD_FLOAT32_C(  -685.50), EASYSIMD_FLOAT32_C(  -233.39),
        EASYSIMD_FLOAT32_C(   -42.82), EASYSIMD_FLOAT32_C(   807.84), EASYSIMD_FLOAT32_C(   170.49), EASYSIMD_FLOAT32_C(  -505.92) } },
    { { EASYSIMD_FLOAT32_C(   598.79), EASYSIMD_FLOAT32_C(  -904.11), EASYSIMD_FLOAT32_C(   859.73), EASYSIMD_FLOAT32_C(  -396.42),
        EASYSIMD_FLOAT32_C(     8.77), EASYSIMD_FLOAT32_C(     6.23), EASYSIMD_FLOAT32_C(   100.17), EASYSIMD_FLOAT32_C(   980.19),
        EASYSIMD_FLOAT32_C(  -734.59), EASYSIMD_FLOAT32_C(    91.45), EASYSIMD_FLOAT32_C(  -568.42), EASYSIMD_FLOAT32_C(    -9.25),
        EASYSIMD_FLOAT32_C(   623.26), EASYSIMD_FLOAT32_C(   709.11), EASYSIMD_FLOAT32_C(  -591.36), EASYSIMD_FLOAT32_C(   331.88) },
      UINT8_C( 89),
      { EASYSIMD_FLOAT32_C(  -244.65), EASYSIMD_FLOAT32_C(   654.62), EASYSIMD_FLOAT32_C(  -137.56), EASYSIMD_FLOAT32_C(  -408.16),
        EASYSIMD_FLOAT32_C(  -363.71), EASYSIMD_FLOAT32_C(   802.14), EASYSIMD_FLOAT32_C(  -257.18), EASYSIMD_FLOAT32_C(  -615.42),
        EASYSIMD_FLOAT32_C(   699.03), EASYSIMD_FLOAT32_C(  -298.35), EASYSIMD_FLOAT32_C(   303.36), EASYSIMD_FLOAT32_C(  -910.01),
        EASYSIMD_FLOAT32_C(   567.23), EASYSIMD_FLOAT32_C(   731.63), EASYSIMD_FLOAT32_C(   688.78), EASYSIMD_FLOAT32_C(   663.12) },
      { EASYSIMD_FLOAT32_C(   591.36), EASYSIMD_FLOAT32_C(  -707.65), EASYSIMD_FLOAT32_C(  -328.11), EASYSIMD_FLOAT32_C(  -402.41),
        EASYSIMD_FLOAT32_C(   392.52), EASYSIMD_FLOAT32_C(  -347.92), EASYSIMD_FLOAT32_C(  -137.01), EASYSIMD_FLOAT32_C(  -516.03),
        EASYSIMD_FLOAT32_C(    83.66), EASYSIMD_FLOAT32_C(   853.75), EASYSIMD_FLOAT32_C(  -892.77), EASYSIMD_FLOAT32_C(  -207.23),
        EASYSIMD_FLOAT32_C(  -737.61), EASYSIMD_FLOAT32_C(   439.10), EASYSIMD_FLOAT32_C(   829.15), EASYSIMD_FLOAT32_C(    17.74) },
      { EASYSIMD_FLOAT32_C(   591.36), EASYSIMD_FLOAT32_C(  -904.11), EASYSIMD_FLOAT32_C(   859.73), EASYSIMD_FLOAT32_C(  -402.41),
        EASYSIMD_FLOAT32_C(   392.52), EASYSIMD_FLOAT32_C(     6.23), EASYSIMD_FLOAT32_C(  -137.01), EASYSIMD_FLOAT32_C(   980.19),
        EASYSIMD_FLOAT32_C(  -734.59), EASYSIMD_FLOAT32_C(    91.45), EASYSIMD_FLOAT32_C(  -568.42), EASYSIMD_FLOAT32_C(    -9.25),
        EASYSIMD_FLOAT32_C(   623.26), EASYSIMD_FLOAT32_C(   709.11), EASYSIMD_FLOAT32_C(  -591.36), EASYSIMD_FLOAT32_C(   331.88) } },
    { { EASYSIMD_FLOAT32_C(    93.72), EASYSIMD_FLOAT32_C(  -308.41), EASYSIMD_FLOAT32_C(   609.58), EASYSIMD_FLOAT32_C(   730.01),
        EASYSIMD_FLOAT32_C(  -506.26), EASYSIMD_FLOAT32_C(  -647.60), EASYSIMD_FLOAT32_C(  -885.40), EASYSIMD_FLOAT32_C(  -807.24),
        EASYSIMD_FLOAT32_C(    54.05), EASYSIMD_FLOAT32_C(   417.96), EASYSIMD_FLOAT32_C(  -717.25), EASYSIMD_FLOAT32_C(  -378.72),
        EASYSIMD_FLOAT32_C(   149.59), EASYSIMD_FLOAT32_C(   971.53), EASYSIMD_FLOAT32_C(  -715.60), EASYSIMD_FLOAT32_C(  -259.06) },
      UINT8_C(249),
      { EASYSIMD_FLOAT32_C(   -43.71), EASYSIMD_FLOAT32_C(   338.53), EASYSIMD_FLOAT32_C(   656.40), EASYSIMD_FLOAT32_C(   608.37),
        EASYSIMD_FLOAT32_C(  -798.47), EASYSIMD_FLOAT32_C(  -859.62), EASYSIMD_FLOAT32_C(  -307.97), EASYSIMD_FLOAT32_C(  -944.73),
        EASYSIMD_FLOAT32_C(  -752.40), EASYSIMD_FLOAT32_C(   484.80), EASYSIMD_FLOAT32_C(  -682.33), EASYSIMD_FLOAT32_C(   686.71),
        EASYSIMD_FLOAT32_C(   313.95), EASYSIMD_FLOAT32_C(   335.41), EASYSIMD_FLOAT32_C(  -219.57), EASYSIMD_FLOAT32_C(  -994.46) },
      { EASYSIMD_FLOAT32_C(   -55.01), EASYSIMD_FLOAT32_C(  -489.56), EASYSIMD_FLOAT32_C(  -500.73), EASYSIMD_FLOAT32_C(   297.39),
        EASYSIMD_FLOAT32_C(  -374.96), EASYSIMD_FLOAT32_C(  -307.96), EASYSIMD_FLOAT32_C(  -648.56), EASYSIMD_FLOAT32_C(  -957.00),
        EASYSIMD_FLOAT32_C(   -25.21), EASYSIMD_FLOAT32_C(   -27.28), EASYSIMD_FLOAT32_C(   192.58), EASYSIMD_FLOAT32_C(   -53.69),
        EASYSIMD_FLOAT32_C(   257.13), EASYSIMD_FLOAT32_C(   933.53), EASYSIMD_FLOAT32_C(   210.19), EASYSIMD_FLOAT32_C(  -786.58) },
      { EASYSIMD_FLOAT32_C(   -43.71), EASYSIMD_FLOAT32_C(  -308.41), EASYSIMD_FLOAT32_C(   609.58), EASYSIMD_FLOAT32_C(   608.37),
        EASYSIMD_FLOAT32_C(  -374.96), EASYSIMD_FLOAT32_C(  -307.96), EASYSIMD_FLOAT32_C(  -307.97), EASYSIMD_FLOAT32_C(  -944.73),
        EASYSIMD_FLOAT32_C(    54.05), EASYSIMD_FLOAT32_C(   417.96), EASYSIMD_FLOAT32_C(  -717.25), EASYSIMD_FLOAT32_C(  -378.72),
        EASYSIMD_FLOAT32_C(   149.59), EASYSIMD_FLOAT32_C(   971.53), EASYSIMD_FLOAT32_C(  -715.60), EASYSIMD_FLOAT32_C(  -259.06) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 src = easysimd_mm512_loadu_ps(test_vec[i].src);
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_ps(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { UINT16_C(49115),
      { EASYSIMD_FLOAT32_C(   433.64), EASYSIMD_FLOAT32_C(   811.97), EASYSIMD_FLOAT32_C(  -935.59), EASYSIMD_FLOAT32_C(  -291.75),
        EASYSIMD_FLOAT32_C(  -969.46), EASYSIMD_FLOAT32_C(   402.84), EASYSIMD_FLOAT32_C(  -536.08), EASYSIMD_FLOAT32_C(   -34.52),
        EASYSIMD_FLOAT32_C(   235.92), EASYSIMD_FLOAT32_C(  -199.87), EASYSIMD_FLOAT32_C(   393.12), EASYSIMD_FLOAT32_C(  -850.22),
        EASYSIMD_FLOAT32_C(  -499.40), EASYSIMD_FLOAT32_C(  -229.12), EASYSIMD_FLOAT32_C(   441.37), EASYSIMD_FLOAT32_C(  -903.49) },
      { EASYSIMD_FLOAT32_C(   235.24), EASYSIMD_FLOAT32_C(  -719.15), EASYSIMD_FLOAT32_C(  -316.51), EASYSIMD_FLOAT32_C(   336.59),
        EASYSIMD_FLOAT32_C(  -863.10), EASYSIMD_FLOAT32_C(   919.24), EASYSIMD_FLOAT32_C(  -654.44), EASYSIMD_FLOAT32_C(   266.97),
        EASYSIMD_FLOAT32_C(  -701.10), EASYSIMD_FLOAT32_C(   297.71), EASYSIMD_FLOAT32_C(   440.40), EASYSIMD_FLOAT32_C(  -385.85),
        EASYSIMD_FLOAT32_C(  -935.58), EASYSIMD_FLOAT32_C(   821.31), EASYSIMD_FLOAT32_C(   136.40), EASYSIMD_FLOAT32_C(   498.06) },
      { EASYSIMD_FLOAT32_C(   433.64), EASYSIMD_FLOAT32_C(   811.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   336.59),
        EASYSIMD_FLOAT32_C(  -863.10), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -536.08), EASYSIMD_FLOAT32_C(   266.97),
        EASYSIMD_FLOAT32_C(   235.92), EASYSIMD_FLOAT32_C(   297.71), EASYSIMD_FLOAT32_C(   440.40), EASYSIMD_FLOAT32_C(  -385.85),
        EASYSIMD_FLOAT32_C(  -499.40), EASYSIMD_FLOAT32_C(   821.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   498.06) } },
    { UINT16_C( 4724),
      { EASYSIMD_FLOAT32_C(  -793.69), EASYSIMD_FLOAT32_C(   663.82), EASYSIMD_FLOAT32_C(  -396.36), EASYSIMD_FLOAT32_C(  -329.77),
        EASYSIMD_FLOAT32_C(  -370.70), EASYSIMD_FLOAT32_C(   839.57), EASYSIMD_FLOAT32_C(   470.36), EASYSIMD_FLOAT32_C(  -977.58),
        EASYSIMD_FLOAT32_C(   989.34), EASYSIMD_FLOAT32_C(   970.96), EASYSIMD_FLOAT32_C(  -206.70), EASYSIMD_FLOAT32_C(   430.71),
        EASYSIMD_FLOAT32_C(  -932.53), EASYSIMD_FLOAT32_C(  -971.45), EASYSIMD_FLOAT32_C(   711.56), EASYSIMD_FLOAT32_C(  -249.04) },
      { EASYSIMD_FLOAT32_C(   365.13), EASYSIMD_FLOAT32_C(   848.47), EASYSIMD_FLOAT32_C(  -329.80), EASYSIMD_FLOAT32_C(   710.69),
        EASYSIMD_FLOAT32_C(   115.44), EASYSIMD_FLOAT32_C(   -30.90), EASYSIMD_FLOAT32_C(     8.40), EASYSIMD_FLOAT32_C(  -444.16),
        EASYSIMD_FLOAT32_C(   583.25), EASYSIMD_FLOAT32_C(    72.82), EASYSIMD_FLOAT32_C(  -622.85), EASYSIMD_FLOAT32_C(  -280.35),
        EASYSIMD_FLOAT32_C(  -429.13), EASYSIMD_FLOAT32_C(  -989.57), EASYSIMD_FLOAT32_C(   920.46), EASYSIMD_FLOAT32_C(  -222.82) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -329.80), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   115.44), EASYSIMD_FLOAT32_C(   839.57), EASYSIMD_FLOAT32_C(   470.36), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   970.96), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -429.13), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(54631),
      { EASYSIMD_FLOAT32_C(   447.41), EASYSIMD_FLOAT32_C(  -696.45), EASYSIMD_FLOAT32_C(  -636.33), EASYSIMD_FLOAT32_C(   -82.23),
        EASYSIMD_FLOAT32_C(  -674.04), EASYSIMD_FLOAT32_C(  -646.99), EASYSIMD_FLOAT32_C(  -111.28), EASYSIMD_FLOAT32_C(   119.27),
        EASYSIMD_FLOAT32_C(   783.72), EASYSIMD_FLOAT32_C(   -43.81), EASYSIMD_FLOAT32_C(   147.81), EASYSIMD_FLOAT32_C(   495.29),
        EASYSIMD_FLOAT32_C(   707.15), EASYSIMD_FLOAT32_C(  -487.06), EASYSIMD_FLOAT32_C(   343.75), EASYSIMD_FLOAT32_C(  -622.65) },
      { EASYSIMD_FLOAT32_C(  -776.37), EASYSIMD_FLOAT32_C(  -540.80), EASYSIMD_FLOAT32_C(   346.45), EASYSIMD_FLOAT32_C(   232.03),
        EASYSIMD_FLOAT32_C(    15.04), EASYSIMD_FLOAT32_C(   -70.30), EASYSIMD_FLOAT32_C(  -695.16), EASYSIMD_FLOAT32_C(   392.19),
        EASYSIMD_FLOAT32_C(   649.35), EASYSIMD_FLOAT32_C(  -124.29), EASYSIMD_FLOAT32_C(   402.62), EASYSIMD_FLOAT32_C(   569.81),
        EASYSIMD_FLOAT32_C(   652.89), EASYSIMD_FLOAT32_C(    76.87), EASYSIMD_FLOAT32_C(  -906.09), EASYSIMD_FLOAT32_C(   100.30) },
      { EASYSIMD_FLOAT32_C(   447.41), EASYSIMD_FLOAT32_C(  -540.80), EASYSIMD_FLOAT32_C(   346.45), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -70.30), EASYSIMD_FLOAT32_C(  -111.28), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   783.72), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   402.62), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   707.15), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   343.75), EASYSIMD_FLOAT32_C(   100.30) } },
    { UINT16_C(65099),
      { EASYSIMD_FLOAT32_C(  -981.93), EASYSIMD_FLOAT32_C(   706.38), EASYSIMD_FLOAT32_C(  -189.41), EASYSIMD_FLOAT32_C(   -93.21),
        EASYSIMD_FLOAT32_C(  -174.35), EASYSIMD_FLOAT32_C(  -405.68), EASYSIMD_FLOAT32_C(   862.98), EASYSIMD_FLOAT32_C(   973.46),
        EASYSIMD_FLOAT32_C(  -910.40), EASYSIMD_FLOAT32_C(   570.13), EASYSIMD_FLOAT32_C(  -513.60), EASYSIMD_FLOAT32_C(   433.36),
        EASYSIMD_FLOAT32_C(   947.48), EASYSIMD_FLOAT32_C(  -289.97), EASYSIMD_FLOAT32_C(   892.56), EASYSIMD_FLOAT32_C(   293.93) },
      { EASYSIMD_FLOAT32_C(   942.06), EASYSIMD_FLOAT32_C(   -92.40), EASYSIMD_FLOAT32_C(  -776.37), EASYSIMD_FLOAT32_C(  -753.10),
        EASYSIMD_FLOAT32_C(  -700.21), EASYSIMD_FLOAT32_C(   872.99), EASYSIMD_FLOAT32_C(   122.61), EASYSIMD_FLOAT32_C(   702.41),
        EASYSIMD_FLOAT32_C(   442.80), EASYSIMD_FLOAT32_C(  -224.50), EASYSIMD_FLOAT32_C(  -220.72), EASYSIMD_FLOAT32_C(   536.71),
        EASYSIMD_FLOAT32_C(   875.80), EASYSIMD_FLOAT32_C(  -840.31), EASYSIMD_FLOAT32_C(   994.29), EASYSIMD_FLOAT32_C(   893.87) },
      { EASYSIMD_FLOAT32_C(   942.06), EASYSIMD_FLOAT32_C(   706.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -93.21),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   862.98), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   570.13), EASYSIMD_FLOAT32_C(  -220.72), EASYSIMD_FLOAT32_C(   536.71),
        EASYSIMD_FLOAT32_C(   947.48), EASYSIMD_FLOAT32_C(  -289.97), EASYSIMD_FLOAT32_C(   994.29), EASYSIMD_FLOAT32_C(   893.87) } },
    { UINT16_C(30402),
      { EASYSIMD_FLOAT32_C(  -199.34), EASYSIMD_FLOAT32_C(  -308.28), EASYSIMD_FLOAT32_C(   399.20), EASYSIMD_FLOAT32_C(  -336.36),
        EASYSIMD_FLOAT32_C(  -334.82), EASYSIMD_FLOAT32_C(   488.81), EASYSIMD_FLOAT32_C(  -766.23), EASYSIMD_FLOAT32_C(   151.58),
        EASYSIMD_FLOAT32_C(   -77.83), EASYSIMD_FLOAT32_C(  -818.75), EASYSIMD_FLOAT32_C(   861.61), EASYSIMD_FLOAT32_C(  -185.28),
        EASYSIMD_FLOAT32_C(   475.18), EASYSIMD_FLOAT32_C(   803.67), EASYSIMD_FLOAT32_C(   722.32), EASYSIMD_FLOAT32_C(   698.81) },
      { EASYSIMD_FLOAT32_C(  -949.43), EASYSIMD_FLOAT32_C(  -977.90), EASYSIMD_FLOAT32_C(   571.80), EASYSIMD_FLOAT32_C(   173.18),
        EASYSIMD_FLOAT32_C(   724.51), EASYSIMD_FLOAT32_C(    14.60), EASYSIMD_FLOAT32_C(   948.68), EASYSIMD_FLOAT32_C(  -496.21),
        EASYSIMD_FLOAT32_C(  -448.69), EASYSIMD_FLOAT32_C(   824.48), EASYSIMD_FLOAT32_C(  -336.52), EASYSIMD_FLOAT32_C(  -454.40),
        EASYSIMD_FLOAT32_C(   718.35), EASYSIMD_FLOAT32_C(  -470.45), EASYSIMD_FLOAT32_C(   350.48), EASYSIMD_FLOAT32_C(  -480.99) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -308.28), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   948.68), EASYSIMD_FLOAT32_C(   151.58),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   824.48), EASYSIMD_FLOAT32_C(   861.61), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   718.35), EASYSIMD_FLOAT32_C(   803.67), EASYSIMD_FLOAT32_C(   722.32), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(63759),
      { EASYSIMD_FLOAT32_C(   182.66), EASYSIMD_FLOAT32_C(   886.45), EASYSIMD_FLOAT32_C(  -761.51), EASYSIMD_FLOAT32_C(   416.43),
        EASYSIMD_FLOAT32_C(    38.03), EASYSIMD_FLOAT32_C(   160.66), EASYSIMD_FLOAT32_C(   597.67), EASYSIMD_FLOAT32_C(  -100.36),
        EASYSIMD_FLOAT32_C(   975.38), EASYSIMD_FLOAT32_C(    72.85), EASYSIMD_FLOAT32_C(  -296.70), EASYSIMD_FLOAT32_C(   697.70),
        EASYSIMD_FLOAT32_C(  -228.34), EASYSIMD_FLOAT32_C(  -246.13), EASYSIMD_FLOAT32_C(   719.81), EASYSIMD_FLOAT32_C(  -656.54) },
      { EASYSIMD_FLOAT32_C(   927.05), EASYSIMD_FLOAT32_C(   444.32), EASYSIMD_FLOAT32_C(   358.06), EASYSIMD_FLOAT32_C(   875.72),
        EASYSIMD_FLOAT32_C(   948.10), EASYSIMD_FLOAT32_C(   909.36), EASYSIMD_FLOAT32_C(   700.21), EASYSIMD_FLOAT32_C(  -388.42),
        EASYSIMD_FLOAT32_C(  -545.04), EASYSIMD_FLOAT32_C(   418.56), EASYSIMD_FLOAT32_C(   141.13), EASYSIMD_FLOAT32_C(   805.45),
        EASYSIMD_FLOAT32_C(   937.57), EASYSIMD_FLOAT32_C(  -637.60), EASYSIMD_FLOAT32_C(  -444.87), EASYSIMD_FLOAT32_C(   120.23) },
      { EASYSIMD_FLOAT32_C(   927.05), EASYSIMD_FLOAT32_C(   886.45), EASYSIMD_FLOAT32_C(   358.06), EASYSIMD_FLOAT32_C(   875.72),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   975.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   805.45),
        EASYSIMD_FLOAT32_C(   937.57), EASYSIMD_FLOAT32_C(  -246.13), EASYSIMD_FLOAT32_C(   719.81), EASYSIMD_FLOAT32_C(   120.23) } },
    { UINT16_C(33342),
      { EASYSIMD_FLOAT32_C(  -463.34), EASYSIMD_FLOAT32_C(   286.87), EASYSIMD_FLOAT32_C(   954.28), EASYSIMD_FLOAT32_C(  -865.67),
        EASYSIMD_FLOAT32_C(  -813.49), EASYSIMD_FLOAT32_C(   929.67), EASYSIMD_FLOAT32_C(   207.18), EASYSIMD_FLOAT32_C(  -110.18),
        EASYSIMD_FLOAT32_C(   627.37), EASYSIMD_FLOAT32_C(   978.84), EASYSIMD_FLOAT32_C(   643.69), EASYSIMD_FLOAT32_C(   347.17),
        EASYSIMD_FLOAT32_C(  -677.70), EASYSIMD_FLOAT32_C(   570.73), EASYSIMD_FLOAT32_C(  -208.51), EASYSIMD_FLOAT32_C(   680.36) },
      { EASYSIMD_FLOAT32_C(   446.46), EASYSIMD_FLOAT32_C(  -260.41), EASYSIMD_FLOAT32_C(   589.73), EASYSIMD_FLOAT32_C(   146.67),
        EASYSIMD_FLOAT32_C(   351.17), EASYSIMD_FLOAT32_C(  -955.31), EASYSIMD_FLOAT32_C(  -434.77), EASYSIMD_FLOAT32_C(  -507.69),
        EASYSIMD_FLOAT32_C(   850.13), EASYSIMD_FLOAT32_C(  -497.20), EASYSIMD_FLOAT32_C(  -145.29), EASYSIMD_FLOAT32_C(  -594.74),
        EASYSIMD_FLOAT32_C(   623.03), EASYSIMD_FLOAT32_C(   103.56), EASYSIMD_FLOAT32_C(   198.89), EASYSIMD_FLOAT32_C(  -840.31) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   286.87), EASYSIMD_FLOAT32_C(   954.28), EASYSIMD_FLOAT32_C(   146.67),
        EASYSIMD_FLOAT32_C(   351.17), EASYSIMD_FLOAT32_C(   929.67), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   978.84), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   680.36) } },
    { UINT16_C(27663),
      { EASYSIMD_FLOAT32_C(  -705.98), EASYSIMD_FLOAT32_C(  -423.06), EASYSIMD_FLOAT32_C(    82.84), EASYSIMD_FLOAT32_C(   501.20),
        EASYSIMD_FLOAT32_C(   466.76), EASYSIMD_FLOAT32_C(  -289.79), EASYSIMD_FLOAT32_C(   480.05), EASYSIMD_FLOAT32_C(   110.45),
        EASYSIMD_FLOAT32_C(  -942.62), EASYSIMD_FLOAT32_C(   802.35), EASYSIMD_FLOAT32_C(  -318.82), EASYSIMD_FLOAT32_C(  -151.13),
        EASYSIMD_FLOAT32_C(   482.71), EASYSIMD_FLOAT32_C(  -872.36), EASYSIMD_FLOAT32_C(   588.47), EASYSIMD_FLOAT32_C(    72.44) },
      { EASYSIMD_FLOAT32_C(   274.30), EASYSIMD_FLOAT32_C(   -60.36), EASYSIMD_FLOAT32_C(   117.12), EASYSIMD_FLOAT32_C(   839.53),
        EASYSIMD_FLOAT32_C(   431.95), EASYSIMD_FLOAT32_C(   -32.74), EASYSIMD_FLOAT32_C(  -657.67), EASYSIMD_FLOAT32_C(  -713.34),
        EASYSIMD_FLOAT32_C(   372.52), EASYSIMD_FLOAT32_C(   965.36), EASYSIMD_FLOAT32_C(   390.22), EASYSIMD_FLOAT32_C(  -428.59),
        EASYSIMD_FLOAT32_C(  -874.95), EASYSIMD_FLOAT32_C(   780.65), EASYSIMD_FLOAT32_C(   724.58), EASYSIMD_FLOAT32_C(  -580.93) },
      { EASYSIMD_FLOAT32_C(   274.30), EASYSIMD_FLOAT32_C(   -60.36), EASYSIMD_FLOAT32_C(   117.12), EASYSIMD_FLOAT32_C(   839.53),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   390.22), EASYSIMD_FLOAT32_C(  -151.13),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   780.65), EASYSIMD_FLOAT32_C(   724.58), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_ps(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_max_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   117.27), EASYSIMD_FLOAT64_C(   605.12), EASYSIMD_FLOAT64_C(   -94.57), EASYSIMD_FLOAT64_C(  -717.29),
        EASYSIMD_FLOAT64_C(    -4.92), EASYSIMD_FLOAT64_C(  -587.44), EASYSIMD_FLOAT64_C(   507.51), EASYSIMD_FLOAT64_C(   596.61) },
      { EASYSIMD_FLOAT64_C(   821.81), EASYSIMD_FLOAT64_C(   612.82), EASYSIMD_FLOAT64_C(   712.15), EASYSIMD_FLOAT64_C(   612.33),
        EASYSIMD_FLOAT64_C(  -249.83), EASYSIMD_FLOAT64_C(  -211.57), EASYSIMD_FLOAT64_C(  -312.67), EASYSIMD_FLOAT64_C(   671.52) },
      { EASYSIMD_FLOAT64_C(   821.81), EASYSIMD_FLOAT64_C(   612.82), EASYSIMD_FLOAT64_C(   712.15), EASYSIMD_FLOAT64_C(   612.33),
        EASYSIMD_FLOAT64_C(    -4.92), EASYSIMD_FLOAT64_C(  -211.57), EASYSIMD_FLOAT64_C(   507.51), EASYSIMD_FLOAT64_C(   671.52) } },
    { { EASYSIMD_FLOAT64_C(   418.12), EASYSIMD_FLOAT64_C(   -94.27), EASYSIMD_FLOAT64_C(   381.31), EASYSIMD_FLOAT64_C(   262.88),
        EASYSIMD_FLOAT64_C(  -485.88), EASYSIMD_FLOAT64_C(  -131.08), EASYSIMD_FLOAT64_C(  -132.09), EASYSIMD_FLOAT64_C(  -583.59) },
      { EASYSIMD_FLOAT64_C(  -922.36), EASYSIMD_FLOAT64_C(   502.23), EASYSIMD_FLOAT64_C(   540.91), EASYSIMD_FLOAT64_C(  -336.70),
        EASYSIMD_FLOAT64_C(  -809.27), EASYSIMD_FLOAT64_C(   810.18), EASYSIMD_FLOAT64_C(   666.45), EASYSIMD_FLOAT64_C(   308.00) },
      { EASYSIMD_FLOAT64_C(   418.12), EASYSIMD_FLOAT64_C(   502.23), EASYSIMD_FLOAT64_C(   540.91), EASYSIMD_FLOAT64_C(   262.88),
        EASYSIMD_FLOAT64_C(  -485.88), EASYSIMD_FLOAT64_C(   810.18), EASYSIMD_FLOAT64_C(   666.45), EASYSIMD_FLOAT64_C(   308.00) } },
    { { EASYSIMD_FLOAT64_C(   415.30), EASYSIMD_FLOAT64_C(  -428.12), EASYSIMD_FLOAT64_C(   590.71), EASYSIMD_FLOAT64_C(  -589.62),
        EASYSIMD_FLOAT64_C(   -15.56), EASYSIMD_FLOAT64_C(    98.21), EASYSIMD_FLOAT64_C(  -993.01), EASYSIMD_FLOAT64_C(  -193.75) },
      { EASYSIMD_FLOAT64_C(  -288.97), EASYSIMD_FLOAT64_C(   719.14), EASYSIMD_FLOAT64_C(  -581.43), EASYSIMD_FLOAT64_C(   461.20),
        EASYSIMD_FLOAT64_C(  -492.43), EASYSIMD_FLOAT64_C(   105.90), EASYSIMD_FLOAT64_C(   132.72), EASYSIMD_FLOAT64_C(   925.69) },
      { EASYSIMD_FLOAT64_C(   415.30), EASYSIMD_FLOAT64_C(   719.14), EASYSIMD_FLOAT64_C(   590.71), EASYSIMD_FLOAT64_C(   461.20),
        EASYSIMD_FLOAT64_C(   -15.56), EASYSIMD_FLOAT64_C(   105.90), EASYSIMD_FLOAT64_C(   132.72), EASYSIMD_FLOAT64_C(   925.69) } },
    { { EASYSIMD_FLOAT64_C(  -988.37), EASYSIMD_FLOAT64_C(  -485.97), EASYSIMD_FLOAT64_C(   188.58), EASYSIMD_FLOAT64_C(  -474.25),
        EASYSIMD_FLOAT64_C(   382.95), EASYSIMD_FLOAT64_C(  -943.52), EASYSIMD_FLOAT64_C(   -57.85), EASYSIMD_FLOAT64_C(   460.59) },
      { EASYSIMD_FLOAT64_C(   558.72), EASYSIMD_FLOAT64_C(  -516.94), EASYSIMD_FLOAT64_C(  -876.11), EASYSIMD_FLOAT64_C(   749.44),
        EASYSIMD_FLOAT64_C(  -706.75), EASYSIMD_FLOAT64_C(   790.34), EASYSIMD_FLOAT64_C(    57.44), EASYSIMD_FLOAT64_C(   708.55) },
      { EASYSIMD_FLOAT64_C(   558.72), EASYSIMD_FLOAT64_C(  -485.97), EASYSIMD_FLOAT64_C(   188.58), EASYSIMD_FLOAT64_C(   749.44),
        EASYSIMD_FLOAT64_C(   382.95), EASYSIMD_FLOAT64_C(   790.34), EASYSIMD_FLOAT64_C(    57.44), EASYSIMD_FLOAT64_C(   708.55) } },
    { { EASYSIMD_FLOAT64_C(  -637.79), EASYSIMD_FLOAT64_C(  -351.85), EASYSIMD_FLOAT64_C(  -881.08), EASYSIMD_FLOAT64_C(   346.65),
        EASYSIMD_FLOAT64_C(   746.36), EASYSIMD_FLOAT64_C(  -874.09), EASYSIMD_FLOAT64_C(  -847.10), EASYSIMD_FLOAT64_C(  -542.61) },
      { EASYSIMD_FLOAT64_C(   845.05), EASYSIMD_FLOAT64_C(  -428.53), EASYSIMD_FLOAT64_C(   918.60), EASYSIMD_FLOAT64_C(  -647.38),
        EASYSIMD_FLOAT64_C(   677.37), EASYSIMD_FLOAT64_C(    51.31), EASYSIMD_FLOAT64_C(  -721.68), EASYSIMD_FLOAT64_C(   689.00) },
      { EASYSIMD_FLOAT64_C(   845.05), EASYSIMD_FLOAT64_C(  -351.85), EASYSIMD_FLOAT64_C(   918.60), EASYSIMD_FLOAT64_C(   346.65),
        EASYSIMD_FLOAT64_C(   746.36), EASYSIMD_FLOAT64_C(    51.31), EASYSIMD_FLOAT64_C(  -721.68), EASYSIMD_FLOAT64_C(   689.00) } },
    { { EASYSIMD_FLOAT64_C(   565.34), EASYSIMD_FLOAT64_C(   466.89), EASYSIMD_FLOAT64_C(  -785.25), EASYSIMD_FLOAT64_C(   -51.71),
        EASYSIMD_FLOAT64_C(   523.38), EASYSIMD_FLOAT64_C(   156.90), EASYSIMD_FLOAT64_C(  -591.12), EASYSIMD_FLOAT64_C(    82.09) },
      { EASYSIMD_FLOAT64_C(   639.96), EASYSIMD_FLOAT64_C(  -467.23), EASYSIMD_FLOAT64_C(  -168.46), EASYSIMD_FLOAT64_C(   933.21),
        EASYSIMD_FLOAT64_C(  -676.90), EASYSIMD_FLOAT64_C(   888.98), EASYSIMD_FLOAT64_C(   641.75), EASYSIMD_FLOAT64_C(  -314.68) },
      { EASYSIMD_FLOAT64_C(   639.96), EASYSIMD_FLOAT64_C(   466.89), EASYSIMD_FLOAT64_C(  -168.46), EASYSIMD_FLOAT64_C(   933.21),
        EASYSIMD_FLOAT64_C(   523.38), EASYSIMD_FLOAT64_C(   888.98), EASYSIMD_FLOAT64_C(   641.75), EASYSIMD_FLOAT64_C(    82.09) } },
    { { EASYSIMD_FLOAT64_C(  -462.87), EASYSIMD_FLOAT64_C(   760.67), EASYSIMD_FLOAT64_C(  -968.03), EASYSIMD_FLOAT64_C(  -716.51),
        EASYSIMD_FLOAT64_C(   886.59), EASYSIMD_FLOAT64_C(  -815.14), EASYSIMD_FLOAT64_C(  -259.11), EASYSIMD_FLOAT64_C(   731.64) },
      { EASYSIMD_FLOAT64_C(  -243.67), EASYSIMD_FLOAT64_C(  -340.52), EASYSIMD_FLOAT64_C(  -915.74), EASYSIMD_FLOAT64_C(  -566.30),
        EASYSIMD_FLOAT64_C(   710.79), EASYSIMD_FLOAT64_C(  -637.42), EASYSIMD_FLOAT64_C(  -877.29), EASYSIMD_FLOAT64_C(   276.14) },
      { EASYSIMD_FLOAT64_C(  -243.67), EASYSIMD_FLOAT64_C(   760.67), EASYSIMD_FLOAT64_C(  -915.74), EASYSIMD_FLOAT64_C(  -566.30),
        EASYSIMD_FLOAT64_C(   886.59), EASYSIMD_FLOAT64_C(  -637.42), EASYSIMD_FLOAT64_C(  -259.11), EASYSIMD_FLOAT64_C(   731.64) } },
    { { EASYSIMD_FLOAT64_C(   829.47), EASYSIMD_FLOAT64_C(  -662.55), EASYSIMD_FLOAT64_C(  -775.57), EASYSIMD_FLOAT64_C(   352.85),
        EASYSIMD_FLOAT64_C(   494.35), EASYSIMD_FLOAT64_C(  -366.69), EASYSIMD_FLOAT64_C(  -565.06), EASYSIMD_FLOAT64_C(   134.31) },
      { EASYSIMD_FLOAT64_C(   166.07), EASYSIMD_FLOAT64_C(   266.48), EASYSIMD_FLOAT64_C(    67.52), EASYSIMD_FLOAT64_C(   489.17),
        EASYSIMD_FLOAT64_C(   155.45), EASYSIMD_FLOAT64_C(  -290.73), EASYSIMD_FLOAT64_C(  -825.51), EASYSIMD_FLOAT64_C(   692.58) },
      { EASYSIMD_FLOAT64_C(   829.47), EASYSIMD_FLOAT64_C(   266.48), EASYSIMD_FLOAT64_C(    67.52), EASYSIMD_FLOAT64_C(   489.17),
        EASYSIMD_FLOAT64_C(   494.35), EASYSIMD_FLOAT64_C(  -290.73), EASYSIMD_FLOAT64_C(  -565.06), EASYSIMD_FLOAT64_C(   692.58) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_max_pd(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_max_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_max_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 src[8];
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   912.18), EASYSIMD_FLOAT64_C(   251.75), EASYSIMD_FLOAT64_C(   555.54), EASYSIMD_FLOAT64_C(  -456.14),
        EASYSIMD_FLOAT64_C(   118.39), EASYSIMD_FLOAT64_C(  -179.75), EASYSIMD_FLOAT64_C(  -362.75), EASYSIMD_FLOAT64_C(  -441.43) },
         UINT8_MAX,
      { EASYSIMD_FLOAT64_C(   852.32), EASYSIMD_FLOAT64_C(  -482.61), EASYSIMD_FLOAT64_C(   790.90), EASYSIMD_FLOAT64_C(   430.23),
        EASYSIMD_FLOAT64_C(   991.14), EASYSIMD_FLOAT64_C(  -742.88), EASYSIMD_FLOAT64_C(  -953.53), EASYSIMD_FLOAT64_C(  -242.10) },
      { EASYSIMD_FLOAT64_C(   221.50), EASYSIMD_FLOAT64_C(  -769.53), EASYSIMD_FLOAT64_C(   584.91), EASYSIMD_FLOAT64_C(  -479.69),
        EASYSIMD_FLOAT64_C(   132.60), EASYSIMD_FLOAT64_C(   485.66), EASYSIMD_FLOAT64_C(   494.57), EASYSIMD_FLOAT64_C(   677.48) },
      { EASYSIMD_FLOAT64_C(   852.32), EASYSIMD_FLOAT64_C(  -482.61), EASYSIMD_FLOAT64_C(   790.90), EASYSIMD_FLOAT64_C(   430.23),
        EASYSIMD_FLOAT64_C(   991.14), EASYSIMD_FLOAT64_C(   485.66), EASYSIMD_FLOAT64_C(   494.57), EASYSIMD_FLOAT64_C(   677.48) } },
    { { EASYSIMD_FLOAT64_C(   714.12), EASYSIMD_FLOAT64_C(  -800.14), EASYSIMD_FLOAT64_C(  -471.78), EASYSIMD_FLOAT64_C(  -757.58),
        EASYSIMD_FLOAT64_C(   282.32), EASYSIMD_FLOAT64_C(    76.37), EASYSIMD_FLOAT64_C(  -845.40), EASYSIMD_FLOAT64_C(  -465.93) },
      UINT8_C(187),
      { EASYSIMD_FLOAT64_C(  -301.54), EASYSIMD_FLOAT64_C(   652.46), EASYSIMD_FLOAT64_C(   452.17), EASYSIMD_FLOAT64_C(   335.71),
        EASYSIMD_FLOAT64_C(  -788.96), EASYSIMD_FLOAT64_C(    82.72), EASYSIMD_FLOAT64_C(   188.03), EASYSIMD_FLOAT64_C(  -271.58) },
      { EASYSIMD_FLOAT64_C(  -126.37), EASYSIMD_FLOAT64_C(  -381.74), EASYSIMD_FLOAT64_C(  -280.44), EASYSIMD_FLOAT64_C(   130.75),
        EASYSIMD_FLOAT64_C(  -335.28), EASYSIMD_FLOAT64_C(   477.46), EASYSIMD_FLOAT64_C(  -647.75), EASYSIMD_FLOAT64_C(  -104.81) },
      { EASYSIMD_FLOAT64_C(  -126.37), EASYSIMD_FLOAT64_C(   652.46), EASYSIMD_FLOAT64_C(  -471.78), EASYSIMD_FLOAT64_C(   335.71),
        EASYSIMD_FLOAT64_C(  -335.28), EASYSIMD_FLOAT64_C(   477.46), EASYSIMD_FLOAT64_C(  -845.40), EASYSIMD_FLOAT64_C(  -104.81) } },
    { { EASYSIMD_FLOAT64_C(    62.37), EASYSIMD_FLOAT64_C(  -127.44), EASYSIMD_FLOAT64_C(  -972.21), EASYSIMD_FLOAT64_C(  -451.97),
        EASYSIMD_FLOAT64_C(  -632.87), EASYSIMD_FLOAT64_C(   705.27), EASYSIMD_FLOAT64_C(  -737.85), EASYSIMD_FLOAT64_C(  -433.01) },
         UINT8_MAX,
      { EASYSIMD_FLOAT64_C(  -495.44), EASYSIMD_FLOAT64_C(   849.31), EASYSIMD_FLOAT64_C(   309.86), EASYSIMD_FLOAT64_C(  -340.84),
        EASYSIMD_FLOAT64_C(  -616.63), EASYSIMD_FLOAT64_C(   941.78), EASYSIMD_FLOAT64_C(   357.62), EASYSIMD_FLOAT64_C(  -964.16) },
      { EASYSIMD_FLOAT64_C(   393.95), EASYSIMD_FLOAT64_C(  -306.67), EASYSIMD_FLOAT64_C(  -753.13), EASYSIMD_FLOAT64_C(  -523.33),
        EASYSIMD_FLOAT64_C(   881.36), EASYSIMD_FLOAT64_C(   -24.71), EASYSIMD_FLOAT64_C(   350.30), EASYSIMD_FLOAT64_C(  -500.38) },
      { EASYSIMD_FLOAT64_C(   393.95), EASYSIMD_FLOAT64_C(   849.31), EASYSIMD_FLOAT64_C(   309.86), EASYSIMD_FLOAT64_C(  -340.84),
        EASYSIMD_FLOAT64_C(   881.36), EASYSIMD_FLOAT64_C(   941.78), EASYSIMD_FLOAT64_C(   357.62), EASYSIMD_FLOAT64_C(  -500.38) } },
    { { EASYSIMD_FLOAT64_C(   694.85), EASYSIMD_FLOAT64_C(  -518.96), EASYSIMD_FLOAT64_C(   164.34), EASYSIMD_FLOAT64_C(   172.31),
        EASYSIMD_FLOAT64_C(  -166.71), EASYSIMD_FLOAT64_C(  -940.46), EASYSIMD_FLOAT64_C(  -765.32), EASYSIMD_FLOAT64_C(   705.85) },
      UINT8_C(121),
      { EASYSIMD_FLOAT64_C(  -217.29), EASYSIMD_FLOAT64_C(  -927.02), EASYSIMD_FLOAT64_C(   792.60), EASYSIMD_FLOAT64_C(    44.86),
        EASYSIMD_FLOAT64_C(  -360.03), EASYSIMD_FLOAT64_C(  -973.91), EASYSIMD_FLOAT64_C(   549.42), EASYSIMD_FLOAT64_C(  -510.72) },
      { EASYSIMD_FLOAT64_C(   335.95), EASYSIMD_FLOAT64_C(  -791.41), EASYSIMD_FLOAT64_C(  -127.34), EASYSIMD_FLOAT64_C(   277.73),
        EASYSIMD_FLOAT64_C(   566.21), EASYSIMD_FLOAT64_C(   -91.51), EASYSIMD_FLOAT64_C(  -328.32), EASYSIMD_FLOAT64_C(  -740.46) },
      { EASYSIMD_FLOAT64_C(   335.95), EASYSIMD_FLOAT64_C(  -518.96), EASYSIMD_FLOAT64_C(   164.34), EASYSIMD_FLOAT64_C(   277.73),
        EASYSIMD_FLOAT64_C(   566.21), EASYSIMD_FLOAT64_C(   -91.51), EASYSIMD_FLOAT64_C(   549.42), EASYSIMD_FLOAT64_C(   705.85) } },
    { { EASYSIMD_FLOAT64_C(   155.37), EASYSIMD_FLOAT64_C(   148.35), EASYSIMD_FLOAT64_C(  -859.10), EASYSIMD_FLOAT64_C(  -869.34),
        EASYSIMD_FLOAT64_C(  -501.36), EASYSIMD_FLOAT64_C(  -359.48), EASYSIMD_FLOAT64_C(   825.51), EASYSIMD_FLOAT64_C(   -20.31) },
      UINT8_C(220),
      { EASYSIMD_FLOAT64_C(    -2.17), EASYSIMD_FLOAT64_C(   812.98), EASYSIMD_FLOAT64_C(   864.40), EASYSIMD_FLOAT64_C(   232.51),
        EASYSIMD_FLOAT64_C(   518.84), EASYSIMD_FLOAT64_C(   951.72), EASYSIMD_FLOAT64_C(  -984.78), EASYSIMD_FLOAT64_C(   591.82) },
      { EASYSIMD_FLOAT64_C(   744.32), EASYSIMD_FLOAT64_C(    60.08), EASYSIMD_FLOAT64_C(  -768.21), EASYSIMD_FLOAT64_C(   770.41),
        EASYSIMD_FLOAT64_C(  -390.49), EASYSIMD_FLOAT64_C(  -278.93), EASYSIMD_FLOAT64_C(   106.36), EASYSIMD_FLOAT64_C(  -181.91) },
      { EASYSIMD_FLOAT64_C(   155.37), EASYSIMD_FLOAT64_C(   148.35), EASYSIMD_FLOAT64_C(   864.40), EASYSIMD_FLOAT64_C(   770.41),
        EASYSIMD_FLOAT64_C(   518.84), EASYSIMD_FLOAT64_C(  -359.48), EASYSIMD_FLOAT64_C(   106.36), EASYSIMD_FLOAT64_C(   591.82) } },
    { { EASYSIMD_FLOAT64_C(   593.73), EASYSIMD_FLOAT64_C(  -615.91), EASYSIMD_FLOAT64_C(  -615.70), EASYSIMD_FLOAT64_C(  -497.78),
        EASYSIMD_FLOAT64_C(    55.77), EASYSIMD_FLOAT64_C(  -356.16), EASYSIMD_FLOAT64_C(   657.59), EASYSIMD_FLOAT64_C(  -795.89) },
      UINT8_C(145),
      { EASYSIMD_FLOAT64_C(   788.25), EASYSIMD_FLOAT64_C(  -297.24), EASYSIMD_FLOAT64_C(   425.26), EASYSIMD_FLOAT64_C(   613.76),
        EASYSIMD_FLOAT64_C(   682.44), EASYSIMD_FLOAT64_C(   230.12), EASYSIMD_FLOAT64_C(  -388.41), EASYSIMD_FLOAT64_C(   495.42) },
      { EASYSIMD_FLOAT64_C(    94.51), EASYSIMD_FLOAT64_C(   844.10), EASYSIMD_FLOAT64_C(    14.26), EASYSIMD_FLOAT64_C(    46.24),
        EASYSIMD_FLOAT64_C(   859.32), EASYSIMD_FLOAT64_C(  -393.92), EASYSIMD_FLOAT64_C(  -209.45), EASYSIMD_FLOAT64_C(   -80.60) },
      { EASYSIMD_FLOAT64_C(   788.25), EASYSIMD_FLOAT64_C(  -615.91), EASYSIMD_FLOAT64_C(  -615.70), EASYSIMD_FLOAT64_C(  -497.78),
        EASYSIMD_FLOAT64_C(   859.32), EASYSIMD_FLOAT64_C(  -356.16), EASYSIMD_FLOAT64_C(   657.59), EASYSIMD_FLOAT64_C(   495.42) } },
    { { EASYSIMD_FLOAT64_C(  -162.13), EASYSIMD_FLOAT64_C(  -439.04), EASYSIMD_FLOAT64_C(   528.91), EASYSIMD_FLOAT64_C(   558.95),
        EASYSIMD_FLOAT64_C(   667.32), EASYSIMD_FLOAT64_C(  -653.00), EASYSIMD_FLOAT64_C(   152.68), EASYSIMD_FLOAT64_C(  -948.59) },
      UINT8_C(192),
      { EASYSIMD_FLOAT64_C(   654.90), EASYSIMD_FLOAT64_C(   107.18), EASYSIMD_FLOAT64_C(   375.14), EASYSIMD_FLOAT64_C(   312.49),
        EASYSIMD_FLOAT64_C(   311.29), EASYSIMD_FLOAT64_C(  -840.12), EASYSIMD_FLOAT64_C(   100.74), EASYSIMD_FLOAT64_C(  -985.95) },
      { EASYSIMD_FLOAT64_C(   585.14), EASYSIMD_FLOAT64_C(  -285.51), EASYSIMD_FLOAT64_C(   696.49), EASYSIMD_FLOAT64_C(  -184.75),
        EASYSIMD_FLOAT64_C(   326.08), EASYSIMD_FLOAT64_C(   191.91), EASYSIMD_FLOAT64_C(   909.77), EASYSIMD_FLOAT64_C(   170.18) },
      { EASYSIMD_FLOAT64_C(  -162.13), EASYSIMD_FLOAT64_C(  -439.04), EASYSIMD_FLOAT64_C(   528.91), EASYSIMD_FLOAT64_C(   558.95),
        EASYSIMD_FLOAT64_C(   667.32), EASYSIMD_FLOAT64_C(  -653.00), EASYSIMD_FLOAT64_C(   909.77), EASYSIMD_FLOAT64_C(   170.18) } },
    { { EASYSIMD_FLOAT64_C(  -793.83), EASYSIMD_FLOAT64_C(   -44.00), EASYSIMD_FLOAT64_C(    29.50), EASYSIMD_FLOAT64_C(  -187.75),
        EASYSIMD_FLOAT64_C(   746.56), EASYSIMD_FLOAT64_C(   948.90), EASYSIMD_FLOAT64_C(   650.12), EASYSIMD_FLOAT64_C(  -692.48) },
      UINT8_C(200),
      { EASYSIMD_FLOAT64_C(   209.07), EASYSIMD_FLOAT64_C(   974.84), EASYSIMD_FLOAT64_C(   824.81), EASYSIMD_FLOAT64_C(  -638.25),
        EASYSIMD_FLOAT64_C(  -973.75), EASYSIMD_FLOAT64_C(  -443.88), EASYSIMD_FLOAT64_C(  -983.35), EASYSIMD_FLOAT64_C(   133.43) },
      { EASYSIMD_FLOAT64_C(   931.26), EASYSIMD_FLOAT64_C(   329.14), EASYSIMD_FLOAT64_C(  -555.28), EASYSIMD_FLOAT64_C(  -908.86),
        EASYSIMD_FLOAT64_C(  -570.13), EASYSIMD_FLOAT64_C(  -541.23), EASYSIMD_FLOAT64_C(   676.28), EASYSIMD_FLOAT64_C(   144.37) },
      { EASYSIMD_FLOAT64_C(  -793.83), EASYSIMD_FLOAT64_C(   -44.00), EASYSIMD_FLOAT64_C(    29.50), EASYSIMD_FLOAT64_C(  -638.25),
        EASYSIMD_FLOAT64_C(   746.56), EASYSIMD_FLOAT64_C(   948.90), EASYSIMD_FLOAT64_C(   676.28), EASYSIMD_FLOAT64_C(   144.37) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d src = easysimd_mm512_loadu_pd(test_vec[i].src);
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_max_pd(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_max_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_max_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { UINT8_C(124),
      { EASYSIMD_FLOAT64_C(   659.04), EASYSIMD_FLOAT64_C(  -119.01), EASYSIMD_FLOAT64_C(   237.02), EASYSIMD_FLOAT64_C(  -321.23),
        EASYSIMD_FLOAT64_C(   -24.75), EASYSIMD_FLOAT64_C(   582.04), EASYSIMD_FLOAT64_C(  -389.52), EASYSIMD_FLOAT64_C(   699.41) },
      { EASYSIMD_FLOAT64_C(   180.67), EASYSIMD_FLOAT64_C(   -25.56), EASYSIMD_FLOAT64_C(  -928.91), EASYSIMD_FLOAT64_C(   898.38),
        EASYSIMD_FLOAT64_C(  -813.04), EASYSIMD_FLOAT64_C(  -166.50), EASYSIMD_FLOAT64_C(    96.18), EASYSIMD_FLOAT64_C(  -720.66) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   237.02), EASYSIMD_FLOAT64_C(   898.38),
        EASYSIMD_FLOAT64_C(   -24.75), EASYSIMD_FLOAT64_C(   582.04), EASYSIMD_FLOAT64_C(    96.18), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 43),
      { EASYSIMD_FLOAT64_C(  -775.97), EASYSIMD_FLOAT64_C(  -789.28), EASYSIMD_FLOAT64_C(   689.62), EASYSIMD_FLOAT64_C(   225.24),
        EASYSIMD_FLOAT64_C(   957.81), EASYSIMD_FLOAT64_C(  -143.72), EASYSIMD_FLOAT64_C(   478.66), EASYSIMD_FLOAT64_C(   320.21) },
      { EASYSIMD_FLOAT64_C(   845.85), EASYSIMD_FLOAT64_C(   504.25), EASYSIMD_FLOAT64_C(    94.13), EASYSIMD_FLOAT64_C(   696.20),
        EASYSIMD_FLOAT64_C(  -502.89), EASYSIMD_FLOAT64_C(  -685.24), EASYSIMD_FLOAT64_C(   355.24), EASYSIMD_FLOAT64_C(   378.11) },
      { EASYSIMD_FLOAT64_C(   845.85), EASYSIMD_FLOAT64_C(   504.25), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   696.20),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -143.72), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(238),
      { EASYSIMD_FLOAT64_C(  -965.99), EASYSIMD_FLOAT64_C(  -646.65), EASYSIMD_FLOAT64_C(   133.82), EASYSIMD_FLOAT64_C(  -355.50),
        EASYSIMD_FLOAT64_C(  -947.23), EASYSIMD_FLOAT64_C(  -685.51), EASYSIMD_FLOAT64_C(   618.94), EASYSIMD_FLOAT64_C(  -876.14) },
      { EASYSIMD_FLOAT64_C(  -787.13), EASYSIMD_FLOAT64_C(   805.90), EASYSIMD_FLOAT64_C(   -42.65), EASYSIMD_FLOAT64_C(   309.05),
        EASYSIMD_FLOAT64_C(  -914.76), EASYSIMD_FLOAT64_C(   958.41), EASYSIMD_FLOAT64_C(   533.08), EASYSIMD_FLOAT64_C(  -704.04) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   805.90), EASYSIMD_FLOAT64_C(   133.82), EASYSIMD_FLOAT64_C(   309.05),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   958.41), EASYSIMD_FLOAT64_C(   618.94), EASYSIMD_FLOAT64_C(  -704.04) } },
    { UINT8_C( 68),
      { EASYSIMD_FLOAT64_C(  -241.67), EASYSIMD_FLOAT64_C(  -746.23), EASYSIMD_FLOAT64_C(  -495.69), EASYSIMD_FLOAT64_C(  -763.01),
        EASYSIMD_FLOAT64_C(   573.99), EASYSIMD_FLOAT64_C(  -649.84), EASYSIMD_FLOAT64_C(   741.23), EASYSIMD_FLOAT64_C(  -331.89) },
      { EASYSIMD_FLOAT64_C(  -953.63), EASYSIMD_FLOAT64_C(  -761.65), EASYSIMD_FLOAT64_C(   -17.12), EASYSIMD_FLOAT64_C(   401.61),
        EASYSIMD_FLOAT64_C(   616.45), EASYSIMD_FLOAT64_C(  -465.34), EASYSIMD_FLOAT64_C(   435.63), EASYSIMD_FLOAT64_C(   969.81) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   -17.12), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   741.23), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(161),
      { EASYSIMD_FLOAT64_C(  -919.88), EASYSIMD_FLOAT64_C(  -977.43), EASYSIMD_FLOAT64_C(   982.97), EASYSIMD_FLOAT64_C(   699.06),
        EASYSIMD_FLOAT64_C(  -853.57), EASYSIMD_FLOAT64_C(  -804.15), EASYSIMD_FLOAT64_C(   504.96), EASYSIMD_FLOAT64_C(   103.79) },
      { EASYSIMD_FLOAT64_C(   504.90), EASYSIMD_FLOAT64_C(   590.20), EASYSIMD_FLOAT64_C(    62.20), EASYSIMD_FLOAT64_C(    37.98),
        EASYSIMD_FLOAT64_C(   886.16), EASYSIMD_FLOAT64_C(  -289.77), EASYSIMD_FLOAT64_C(   796.31), EASYSIMD_FLOAT64_C(  -860.07) },
      { EASYSIMD_FLOAT64_C(   504.90), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -289.77), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   103.79) } },
    { UINT8_C( 77),
      { EASYSIMD_FLOAT64_C(  -966.71), EASYSIMD_FLOAT64_C(   713.91), EASYSIMD_FLOAT64_C(   564.71), EASYSIMD_FLOAT64_C(   774.53),
        EASYSIMD_FLOAT64_C(  -617.98), EASYSIMD_FLOAT64_C(   611.07), EASYSIMD_FLOAT64_C(  -987.13), EASYSIMD_FLOAT64_C(   364.90) },
      { EASYSIMD_FLOAT64_C(    12.69), EASYSIMD_FLOAT64_C(   629.33), EASYSIMD_FLOAT64_C(   899.56), EASYSIMD_FLOAT64_C(  -551.68),
        EASYSIMD_FLOAT64_C(   599.14), EASYSIMD_FLOAT64_C(   568.04), EASYSIMD_FLOAT64_C(  -471.56), EASYSIMD_FLOAT64_C(   621.71) },
      { EASYSIMD_FLOAT64_C(    12.69), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   899.56), EASYSIMD_FLOAT64_C(   774.53),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -471.56), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(168),
      { EASYSIMD_FLOAT64_C(  -772.50), EASYSIMD_FLOAT64_C(   768.14), EASYSIMD_FLOAT64_C(   746.85), EASYSIMD_FLOAT64_C(   732.46),
        EASYSIMD_FLOAT64_C(  -128.07), EASYSIMD_FLOAT64_C(   251.75), EASYSIMD_FLOAT64_C(   322.66), EASYSIMD_FLOAT64_C(   934.13) },
      { EASYSIMD_FLOAT64_C(  -710.27), EASYSIMD_FLOAT64_C(   208.82), EASYSIMD_FLOAT64_C(  -355.64), EASYSIMD_FLOAT64_C(  -913.97),
        EASYSIMD_FLOAT64_C(   348.75), EASYSIMD_FLOAT64_C(   858.91), EASYSIMD_FLOAT64_C(  -880.67), EASYSIMD_FLOAT64_C(    62.66) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   732.46),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   858.91), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   934.13) } },
    { UINT8_C(222),
      { EASYSIMD_FLOAT64_C(   893.86), EASYSIMD_FLOAT64_C(   444.68), EASYSIMD_FLOAT64_C(    34.69), EASYSIMD_FLOAT64_C(   906.73),
        EASYSIMD_FLOAT64_C(  -190.42), EASYSIMD_FLOAT64_C(  -952.63), EASYSIMD_FLOAT64_C(   536.06), EASYSIMD_FLOAT64_C(  -290.86) },
      { EASYSIMD_FLOAT64_C(  -504.31), EASYSIMD_FLOAT64_C(   135.19), EASYSIMD_FLOAT64_C(  -722.83), EASYSIMD_FLOAT64_C(    24.13),
        EASYSIMD_FLOAT64_C(  -243.10), EASYSIMD_FLOAT64_C(   828.18), EASYSIMD_FLOAT64_C(   251.63), EASYSIMD_FLOAT64_C(  -474.96) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   444.68), EASYSIMD_FLOAT64_C(    34.69), EASYSIMD_FLOAT64_C(   906.73),
        EASYSIMD_FLOAT64_C(  -190.42), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   536.06), EASYSIMD_FLOAT64_C(  -290.86) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_max_pd(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_max_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_max_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_max_epu64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_epu64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_max_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_max_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_max_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_max_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_epu64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_max_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_max_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_epu8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_epu8)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_epu16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_epu16)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_epu32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_epu32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_epu64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_epu64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_max_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_max_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_max_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
