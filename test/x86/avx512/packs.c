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

#define EASYSIMD_TEST_X86_AVX512_INSN packs

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/packs.h>

static int
test_easysimd_mm_mask_packs_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int8_t src[16];
    const uint16_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int8_t r[16];
  } test_vec[] = {
    { {  INT8_C(  75),  INT8_C(  71), -INT8_C(  22), -INT8_C(  70), -INT8_C(  92),  INT8_C( 105),  INT8_C( 120), -INT8_C(  65),
         INT8_C(  87),  INT8_C(  88), -INT8_C( 107),  INT8_C(  55),  INT8_C(  82),  INT8_C(  62), -INT8_C( 116), -INT8_C(  22) },
      UINT16_C(24601),
      {  INT16_C(  8476),  INT16_C(   387),  INT16_C(   861), -INT16_C(  3943),  INT16_C( 12030), -INT16_C( 15852),  INT16_C( 24432),  INT16_C( 23050) },
      { -INT16_C( 20967), -INT16_C( 28221),  INT16_C(  6765),  INT16_C(  1001),  INT16_C( 15442), -INT16_C(  8639),  INT16_C( 23078),  INT16_C( 16958) },
      {      INT8_MAX,  INT8_C(  71), -INT8_C(  22),      INT8_MIN,      INT8_MAX,  INT8_C( 105),  INT8_C( 120), -INT8_C(  65),
         INT8_C(  87),  INT8_C(  88), -INT8_C( 107),  INT8_C(  55),  INT8_C(  82),      INT8_MIN,      INT8_MAX, -INT8_C(  22) } },
    { {  INT8_C( 124), -INT8_C(  63),  INT8_C(  67), -INT8_C(  39), -INT8_C(  60), -INT8_C(  35), -INT8_C(  55), -INT8_C(  61),
         INT8_C(  11), -INT8_C(  35), -INT8_C( 123),  INT8_C( 123),  INT8_C(  60), -INT8_C( 113), -INT8_C(  43),  INT8_C(  86) },
      UINT16_C(38974),
      { -INT16_C( 21529), -INT16_C( 11854),  INT16_C(  1198), -INT16_C(  4339),  INT16_C( 13283),  INT16_C(  8522), -INT16_C( 14730), -INT16_C( 17949) },
      { -INT16_C( 22625),  INT16_C( 27030), -INT16_C( 24214), -INT16_C(  4026), -INT16_C( 31972), -INT16_C(  3713), -INT16_C( 16935), -INT16_C( 16247) },
      {  INT8_C( 124),      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX, -INT8_C(  55), -INT8_C(  61),
         INT8_C(  11), -INT8_C(  35), -INT8_C( 123),      INT8_MIN,      INT8_MIN, -INT8_C( 113), -INT8_C(  43),      INT8_MIN } },
    { {  INT8_C( 105),  INT8_C(  60), -INT8_C( 111),  INT8_C(  23),  INT8_C(  64), -INT8_C(  98),  INT8_C(   7),  INT8_C(  35),
        -INT8_C(  46),  INT8_C(  81),  INT8_C(  69),  INT8_C(  72),  INT8_C(  23),  INT8_C(  40),  INT8_C(   1), -INT8_C(  74) },
      UINT16_C(39119),
      {  INT16_C( 14879),  INT16_C( 26169),  INT16_C( 22058), -INT16_C( 22039), -INT16_C( 15801), -INT16_C( 11929), -INT16_C( 12158),  INT16_C(  5133) },
      {  INT16_C( 19943), -INT16_C(  4430), -INT16_C( 31631), -INT16_C( 18881),  INT16_C( 22220), -INT16_C( 12578), -INT16_C( 21235),  INT16_C( 11366) },
      {      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,  INT8_C(  64), -INT8_C(  98),      INT8_MIN,      INT8_MAX,
        -INT8_C(  46),  INT8_C(  81),  INT8_C(  69),      INT8_MIN,      INT8_MAX,  INT8_C(  40),  INT8_C(   1),      INT8_MAX } },
    { { -INT8_C(  25), -INT8_C(  97), -INT8_C( 110),  INT8_C(  17), -INT8_C(  11),  INT8_C( 123), -INT8_C(  69),  INT8_C(  61),
         INT8_C(  61),  INT8_C(  34),  INT8_C(  14), -INT8_C(  64), -INT8_C(  14),  INT8_C(  27), -INT8_C(  44), -INT8_C(  39) },
      UINT16_C(34408),
      { -INT16_C(  9784),  INT16_C(  1803), -INT16_C( 10353),  INT16_C( 27998),  INT16_C( 27557),  INT16_C(  2843),  INT16_C(   663),  INT16_C( 10923) },
      { -INT16_C( 24556), -INT16_C( 12379), -INT16_C(  7203), -INT16_C(  5135), -INT16_C(  7261),  INT16_C( 30470),  INT16_C( 28604), -INT16_C( 31491) },
      { -INT8_C(  25), -INT8_C(  97), -INT8_C( 110),      INT8_MAX, -INT8_C(  11),      INT8_MAX,      INT8_MAX,  INT8_C(  61),
         INT8_C(  61),      INT8_MIN,      INT8_MIN, -INT8_C(  64), -INT8_C(  14),  INT8_C(  27), -INT8_C(  44),      INT8_MIN } },
    { {  INT8_C(  72),  INT8_C(   8), -INT8_C( 116), -INT8_C(  40), -INT8_C(  32), -INT8_C(  22),  INT8_C(  69), -INT8_C( 123),
         INT8_C(  85),  INT8_C(  96), -INT8_C( 111), -INT8_C(  20),  INT8_C(  99),  INT8_C(  60),  INT8_C(  22),  INT8_C( 119) },
      UINT16_C(48348),
      { -INT16_C( 17850),  INT16_C( 14239),  INT16_C( 17061), -INT16_C( 21478), -INT16_C( 10567), -INT16_C( 18917),  INT16_C( 25435), -INT16_C(  6209) },
      { -INT16_C( 24773), -INT16_C( 32303),  INT16_C(  9764), -INT16_C( 18975),  INT16_C( 17426),  INT16_C( 10737), -INT16_C( 12613),  INT16_C(   485) },
      {  INT8_C(  72),  INT8_C(   8),      INT8_MAX,      INT8_MIN,      INT8_MIN, -INT8_C(  22),      INT8_MAX,      INT8_MIN,
         INT8_C(  85),  INT8_C(  96),      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,  INT8_C(  22),      INT8_MAX } },
    { { -INT8_C( 120), -INT8_C( 124),  INT8_C(  56),  INT8_C(  45), -INT8_C(  58),  INT8_C(  82), -INT8_C(  39),      INT8_MAX,
         INT8_C(  41), -INT8_C(  12),  INT8_C(  53), -INT8_C( 124),  INT8_C(  88), -INT8_C(  12),  INT8_C( 107), -INT8_C( 109) },
      UINT16_C(15507),
      { -INT16_C( 18412), -INT16_C(  2462),  INT16_C( 29805),  INT16_C( 24378), -INT16_C(  2403), -INT16_C( 32211), -INT16_C( 18953),  INT16_C( 12294) },
      { -INT16_C( 13086), -INT16_C( 17278), -INT16_C( 21685), -INT16_C( 32336),  INT16_C(  2095), -INT16_C( 25995),  INT16_C(  2460), -INT16_C( 20266) },
      {      INT8_MIN,      INT8_MIN,  INT8_C(  56),  INT8_C(  45),      INT8_MIN,  INT8_C(  82), -INT8_C(  39),      INT8_MAX,
         INT8_C(  41), -INT8_C(  12),      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MIN,  INT8_C( 107), -INT8_C( 109) } },
    { { -INT8_C(  63),  INT8_C(  56), -INT8_C(  90),  INT8_C(  46), -INT8_C(  83), -INT8_C(  31), -INT8_C( 115),  INT8_C(  74),
        -INT8_C(  41), -INT8_C(  70), -INT8_C(  51), -INT8_C(  50),  INT8_C( 111), -INT8_C(  45), -INT8_C(   2),  INT8_C(  82) },
      UINT16_C(33184),
      { -INT16_C(  5362), -INT16_C( 16852),  INT16_C( 23660), -INT16_C(  7481),  INT16_C( 25590), -INT16_C( 12821), -INT16_C( 21485), -INT16_C( 17915) },
      { -INT16_C( 19750),  INT16_C( 26779),  INT16_C( 29437), -INT16_C( 13790), -INT16_C( 28096),  INT16_C( 16285),  INT16_C( 15844), -INT16_C(  3392) },
      { -INT8_C(  63),  INT8_C(  56), -INT8_C(  90),  INT8_C(  46), -INT8_C(  83),      INT8_MIN, -INT8_C( 115),      INT8_MIN,
             INT8_MIN, -INT8_C(  70), -INT8_C(  51), -INT8_C(  50),  INT8_C( 111), -INT8_C(  45), -INT8_C(   2),      INT8_MIN } },
    { {  INT8_C(  41), -INT8_C(  20), -INT8_C(  80), -INT8_C( 107),  INT8_C(  72),  INT8_C( 119),  INT8_C( 119),  INT8_C(  63),
        -INT8_C(  38),  INT8_C(  98),  INT8_C(  12), -INT8_C(  18),  INT8_C(  14),  INT8_C(  17), -INT8_C(  88), -INT8_C(  23) },
      UINT16_C(17348),
      { -INT16_C( 16047),  INT16_C( 29621), -INT16_C(  2677),  INT16_C( 10245), -INT16_C(  5836), -INT16_C(  2970), -INT16_C( 28709), -INT16_C( 29471) },
      {  INT16_C( 10532), -INT16_C( 25597), -INT16_C(  8600),  INT16_C( 29950),  INT16_C(  3532),  INT16_C( 29830),  INT16_C( 19190),  INT16_C( 18359) },
      {  INT8_C(  41), -INT8_C(  20),      INT8_MIN, -INT8_C( 107),  INT8_C(  72),  INT8_C( 119),      INT8_MIN,      INT8_MIN,
             INT8_MAX,      INT8_MIN,  INT8_C(  12), -INT8_C(  18),  INT8_C(  14),  INT8_C(  17),      INT8_MAX, -INT8_C(  23) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi8(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_packs_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_packs_epi16");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_mask_packs_epi16(src, k, a, b);

    easysimd_test_x86_write_i8x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_packs_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int8_t r[16];
  } test_vec[] = {
    { UINT16_C( 6525),
      { -INT16_C( 19034), -INT16_C( 16008),  INT16_C( 29635),  INT16_C( 30709),  INT16_C( 10769), -INT16_C( 10017), -INT16_C(  8020),  INT16_C( 26431) },
      { -INT16_C( 23688), -INT16_C( 19925),  INT16_C( 10158), -INT16_C( 22010), -INT16_C( 32057),  INT16_C( 19801), -INT16_C( 10495), -INT16_C( 22682) },
      {      INT8_MIN,  INT8_C(   0),      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,  INT8_C(   0),
             INT8_MIN,  INT8_C(   0),  INT8_C(   0),      INT8_MIN,      INT8_MIN,  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(56972),
      {  INT16_C( 20328),  INT16_C( 23889),  INT16_C( 25542), -INT16_C( 23161),  INT16_C( 13115),  INT16_C( 31366), -INT16_C(   357), -INT16_C( 14819) },
      { -INT16_C( 13392), -INT16_C( 18706), -INT16_C( 19083), -INT16_C( 12487),  INT16_C( 14850),  INT16_C( 26790),  INT16_C( 13025),  INT16_C( 18759) },
      {  INT8_C(   0),  INT8_C(   0),      INT8_MAX,      INT8_MIN,  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),      INT8_MIN,
         INT8_C(   0),      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,  INT8_C(   0),      INT8_MAX,      INT8_MAX } },
    { UINT16_C(39041),
      {  INT16_C( 18343),  INT16_C( 12027),  INT16_C( 14317),  INT16_C( 29538), -INT16_C(   591), -INT16_C( 12431),  INT16_C(  8643), -INT16_C( 20070) },
      {  INT16_C(  4311),  INT16_C(  4199),  INT16_C( 27103), -INT16_C( 31414),  INT16_C( 11218),  INT16_C(  6583),  INT16_C( 14453),  INT16_C(  7345) },
      {      INT8_MAX,  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),      INT8_MIN,
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),      INT8_MIN,      INT8_MAX,  INT8_C(   0),  INT8_C(   0),      INT8_MAX } },
    { UINT16_C(44415),
      {  INT16_C( 27722), -INT16_C( 21276), -INT16_C( 27169),  INT16_C( 20649),  INT16_C( 28004), -INT16_C(   143),  INT16_C( 18718), -INT16_C( 31473) },
      { -INT16_C(  4519), -INT16_C( 23313), -INT16_C( 16013),  INT16_C( 10959),  INT16_C( 17626), -INT16_C( 29854), -INT16_C(  7840), -INT16_C( 21704) },
      {      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,  INT8_C(   0),
             INT8_MIN,  INT8_C(   0),      INT8_MIN,      INT8_MAX,  INT8_C(   0),      INT8_MIN,  INT8_C(   0),      INT8_MIN } },
    { UINT16_C( 7246),
      {  INT16_C( 11607),  INT16_C(   434),  INT16_C(  5758), -INT16_C(  4242), -INT16_C( 29675),  INT16_C(  9272), -INT16_C( 28142),  INT16_C(   274) },
      { -INT16_C( 31434),  INT16_C(  1474), -INT16_C( 25425),  INT16_C(  4426), -INT16_C( 21977),  INT16_C( 24819),  INT16_C( 16725), -INT16_C( 21124) },
      {  INT8_C(   0),      INT8_MAX,      INT8_MAX,      INT8_MIN,  INT8_C(   0),  INT8_C(   0),      INT8_MIN,  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),      INT8_MIN,      INT8_MAX,      INT8_MIN,  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(11886),
      { -INT16_C(  4946),  INT16_C(  7237),  INT16_C( 23260),  INT16_C(  5288), -INT16_C( 17793), -INT16_C( 28250), -INT16_C(  9029),  INT16_C( 32023) },
      { -INT16_C( 14622),  INT16_C( 11289),  INT16_C( 16856), -INT16_C( 13354),  INT16_C( 11425),  INT16_C(  7436),  INT16_C( 31449), -INT16_C( 30900) },
      {  INT8_C(   0),      INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C(   0),      INT8_MIN,      INT8_MIN,  INT8_C(   0),
         INT8_C(   0),      INT8_MAX,      INT8_MAX,      INT8_MIN,  INT8_C(   0),      INT8_MAX,  INT8_C(   0),  INT8_C(   0) } },
    { UINT16_C(37223),
      {  INT16_C( 17315),  INT16_C( 19435),  INT16_C( 27223), -INT16_C(   506), -INT16_C( 15876),  INT16_C(  5082), -INT16_C( 17345),  INT16_C( 22745) },
      { -INT16_C( 19992), -INT16_C( 16487),  INT16_C( 14972), -INT16_C( 30485), -INT16_C( 15272), -INT16_C( 23549),  INT16_C( 27211), -INT16_C(  4555) },
      {      INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C(   0),  INT8_C(   0),      INT8_MAX,      INT8_MIN,  INT8_C(   0),
             INT8_MIN,  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),      INT8_MIN,  INT8_C(   0),  INT8_C(   0),      INT8_MIN } },
    { UINT16_C( 8365),
      {  INT16_C(  1081),  INT16_C( 16267), -INT16_C( 30974), -INT16_C(  8959),  INT16_C( 16538),  INT16_C( 29593), -INT16_C( 32104),  INT16_C( 12837) },
      { -INT16_C( 24255),  INT16_C( 11372), -INT16_C( 15318),  INT16_C( 11760),  INT16_C( 15208), -INT16_C( 25193),  INT16_C( 17449),  INT16_C( 25278) },
      {      INT8_MAX,  INT8_C(   0),      INT8_MIN,      INT8_MIN,  INT8_C(   0),      INT8_MAX,  INT8_C(   0),      INT8_MAX,
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),      INT8_MIN,  INT8_C(   0),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_packs_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_packs_epi16");
    easysimd_test_x86_assert_equal_i8x16(r, easysimd_mm_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_maskz_packs_epi16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_packs_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int16_t r[8];
  } test_vec[] = {
    { {  INT16_C( 27659), -INT16_C( 26950), -INT16_C( 16287), -INT16_C( 26946),  INT16_C(  9385), -INT16_C( 31350),  INT16_C( 27571), -INT16_C( 10223) },
      UINT8_C(149),
      { -INT32_C(   218270700),  INT32_C(  2143187570),  INT32_C(  1114977016),  INT32_C(  1431157993) },
      {  INT32_C(   934798199), -INT32_C(   958378591),  INT32_C(  1132029399), -INT32_C(  1948757642) },
      {        INT16_MIN, -INT16_C( 26950),        INT16_MAX, -INT16_C( 26946),        INT16_MAX, -INT16_C( 31350),  INT16_C( 27571),        INT16_MIN } },
    { { -INT16_C( 10811),  INT16_C( 14461),  INT16_C( 15431),  INT16_C( 16311),  INT16_C( 11630),  INT16_C( 22657), -INT16_C( 12567),  INT16_C( 24749) },
      UINT8_C(177),
      { -INT32_C(  1319921820), -INT32_C(   578217608), -INT32_C(   464204654),  INT32_C(  2041175972) },
      { -INT32_C(  1732124324), -INT32_C(   972619623),  INT32_C(  1353670530),  INT32_C(  1895960588) },
      {        INT16_MIN,  INT16_C( 14461),  INT16_C( 15431),  INT16_C( 16311),        INT16_MIN,        INT16_MIN, -INT16_C( 12567),        INT16_MAX } },
    { {  INT16_C( 21927),  INT16_C(  7970), -INT16_C( 21650),  INT16_C(   253),  INT16_C( 20855),  INT16_C(  7140), -INT16_C( 29136), -INT16_C( 29547) },
      UINT8_C(111),
      {  INT32_C(  1443374422), -INT32_C(  1948725716), -INT32_C(  1902696066), -INT32_C(  2144008149) },
      { -INT32_C(   689023701), -INT32_C(  1555108270),  INT32_C(  1641245139),  INT32_C(  1422942206) },
      {        INT16_MAX,        INT16_MIN,        INT16_MIN,        INT16_MIN,  INT16_C( 20855),        INT16_MIN,        INT16_MAX, -INT16_C( 29547) } },
    { { -INT16_C(  9852), -INT16_C( 20309), -INT16_C( 31833),  INT16_C(  9531), -INT16_C( 11348), -INT16_C( 10317), -INT16_C(  5669),  INT16_C(  1623) },
      UINT8_C( 62),
      {  INT32_C(   881909061), -INT32_C(  1811467477),  INT32_C(  1704159238),  INT32_C(   300607288) },
      {  INT32_C(   381262482), -INT32_C(  1446846762),  INT32_C(  2072287890),  INT32_C(   934906865) },
      { -INT16_C(  9852),        INT16_MIN,        INT16_MAX,        INT16_MAX,        INT16_MAX,        INT16_MIN, -INT16_C(  5669),  INT16_C(  1623) } },
    { {  INT16_C( 18792), -INT16_C( 27797),  INT16_C( 29308), -INT16_C( 32217), -INT16_C( 17702),  INT16_C(  4839), -INT16_C( 11870),  INT16_C( 13348) },
      UINT8_C(108),
      { -INT32_C(  1153283363), -INT32_C(  1488065779), -INT32_C(    90650513), -INT32_C(   899494015) },
      { -INT32_C(  1404635846), -INT32_C(   679032803), -INT32_C(  2122737232), -INT32_C(  1695699267) },
      {  INT16_C( 18792), -INT16_C( 27797),        INT16_MIN,        INT16_MIN, -INT16_C( 17702),        INT16_MIN,        INT16_MIN,  INT16_C( 13348) } },
    { {  INT16_C( 12280),  INT16_C(  1365), -INT16_C( 23782), -INT16_C( 30036),  INT16_C( 17771), -INT16_C(  4732), -INT16_C(  6380),  INT16_C( 20407) },
      UINT8_C(220),
      { -INT32_C(   956695554),  INT32_C(   460771714), -INT32_C(   119998390),  INT32_C(   368145125) },
      {  INT32_C(  1781528263), -INT32_C(   388580957),  INT32_C(   637322046),  INT32_C(  2013350778) },
      {  INT16_C( 12280),  INT16_C(  1365),        INT16_MIN,        INT16_MAX,        INT16_MAX, -INT16_C(  4732),        INT16_MAX,        INT16_MAX } },
    { { -INT16_C(  1209), -INT16_C( 14017), -INT16_C( 18996),  INT16_C(  5860), -INT16_C( 17235), -INT16_C( 27889),  INT16_C(    46), -INT16_C(  2648) },
      UINT8_C(246),
      { -INT32_C(  1852219177), -INT32_C(   103841482), -INT32_C(   915147650), -INT32_C(   267326219) },
      { -INT32_C(   524494549),  INT32_C(  2039403453), -INT32_C(   492363294), -INT32_C(  1596416567) },
      { -INT16_C(  1209),        INT16_MIN,        INT16_MIN,  INT16_C(  5860),        INT16_MIN,        INT16_MAX,        INT16_MIN,        INT16_MIN } },
    { {  INT16_C( 29437),  INT16_C( 13105),  INT16_C(   243),  INT16_C( 28972), -INT16_C( 24588), -INT16_C(  5573),  INT16_C( 19339), -INT16_C( 18726) },
      UINT8_C( 37),
      {  INT32_C(  1793234839),  INT32_C(  1179409445), -INT32_C(  1609617917),  INT32_C(  2023599878) },
      { -INT32_C(   512962335), -INT32_C(  1663640068),  INT32_C(  1680326680),  INT32_C(   831119002) },
      {        INT16_MAX,  INT16_C( 13105),        INT16_MIN,  INT16_C( 28972), -INT16_C( 24588),        INT16_MIN,  INT16_C( 19339), -INT16_C( 18726) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_packs_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_packs_epi32");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_packs_epi32(src, k, a, b);

    easysimd_test_x86_write_i16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_packs_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C( 72),
      { -INT32_C(   800349623), -INT32_C(   479582045),  INT32_C(  1132191169), -INT32_C(  1534808830) },
      { -INT32_C(   556879846), -INT32_C(   616039520), -INT32_C(   704322414), -INT32_C(   350328926) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),        INT16_MIN,  INT16_C(     0),  INT16_C(     0),        INT16_MIN,  INT16_C(     0) } },
    { UINT8_C(  9),
      { -INT32_C(  1850950807),  INT32_C(    55807781), -INT32_C(  1207593462), -INT32_C(   875386597) },
      {  INT32_C(  1919725687), -INT32_C(   603699209), -INT32_C(  1283532212),  INT32_C(  1656515321) },
      {        INT16_MIN,  INT16_C(     0),  INT16_C(     0),        INT16_MIN,  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 37),
      { -INT32_C(   146017176), -INT32_C(   587116985), -INT32_C(    34031277),  INT32_C(  1014285451) },
      {  INT32_C(  1999890224), -INT32_C(   960294933), -INT32_C(   121670002), -INT32_C(  1709366990) },
      {        INT16_MIN,  INT16_C(     0),        INT16_MIN,  INT16_C(     0),  INT16_C(     0),        INT16_MIN,  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 21),
      { -INT32_C(  1235447448),  INT32_C(  1275738771), -INT32_C(   153614542),  INT32_C(  1646662779) },
      {  INT32_C(  1464770119), -INT32_C(   656075679), -INT32_C(   200614445),  INT32_C(  1661642234) },
      {        INT16_MIN,  INT16_C(     0),        INT16_MIN,  INT16_C(     0),        INT16_MAX,  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 54),
      { -INT32_C(  1597433498),  INT32_C(   718476835),  INT32_C(    44485102), -INT32_C(  1907816208) },
      {  INT32_C(  1794089046),  INT32_C(  1648215941),  INT32_C(  2002596562), -INT32_C(  1565671364) },
      {  INT16_C(     0),        INT16_MAX,        INT16_MAX,  INT16_C(     0),        INT16_MAX,        INT16_MAX,  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(217),
      { -INT32_C(  1912781961), -INT32_C(   545577194), -INT32_C(   691044915),  INT32_C(  1730960838) },
      {  INT32_C(   351049549),  INT32_C(   115757012),  INT32_C(  1816288684), -INT32_C(  2109348597) },
      {        INT16_MIN,  INT16_C(     0),  INT16_C(     0),        INT16_MAX,        INT16_MAX,  INT16_C(     0),        INT16_MAX,        INT16_MIN } },
    { UINT8_C( 40),
      {  INT32_C(  1782452034),  INT32_C(   121052554),  INT32_C(  1254952429), -INT32_C(   778619846) },
      {  INT32_C(  1889905697), -INT32_C(   266556270), -INT32_C(   738490130),  INT32_C(   284917197) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),        INT16_MIN,  INT16_C(     0),        INT16_MIN,  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(140),
      {  INT32_C(  1461090873), -INT32_C(  1086054991),  INT32_C(   536448746), -INT32_C(   767505882) },
      {  INT32_C(   459583599),  INT32_C(  1409963212), -INT32_C(   870195889),  INT32_C(   307769817) },
      {  INT16_C(     0),  INT16_C(     0),        INT16_MAX,        INT16_MIN,  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),        INT16_MAX } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_packs_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_packs_epi32");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_packs_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_packs_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const int8_t r[64];
  } test_vec[] = {
    { {  INT16_C(   148),  INT16_C(    89),  INT16_C(    44),  INT16_C(   120),  INT16_C(   172),  INT16_C(    95),  INT16_C(   152),  INT16_C(    63),
         INT16_C(   158),  INT16_C(    87),  INT16_C(   102),  INT16_C(   236),  INT16_C(   153),  INT16_C(   222),  INT16_C(   143),  INT16_C(   196),
         INT16_C(   171),  INT16_C(   232),  INT16_C(    34),  INT16_C(   217),  INT16_C(   125),  INT16_C(   165),  INT16_C(   230),  INT16_C(     5),
         INT16_C(    46),  INT16_C(   252),  INT16_C(   228),  INT16_C(    53),  INT16_C(    41),  INT16_C(   126),  INT16_C(    57),  INT16_C(   220) },
      {  INT16_C( 25061), -INT16_C( 16956),  INT16_C(  9603),  INT16_C( 21142), -INT16_C( 12382), -INT16_C( 18441), -INT16_C(  9035),  INT16_C( 14780),
         INT16_C(  6155), -INT16_C( 24779),  INT16_C(  7677),  INT16_C( 31444), -INT16_C(  6074), -INT16_C(     7),  INT16_C( 15393),  INT16_C(  1755),
        -INT16_C( 24419),  INT16_C(  8387),  INT16_C( 23237),  INT16_C( 26482),  INT16_C( 27177), -INT16_C(  8674), -INT16_C(  9658),  INT16_C( 20759),
         INT16_C( 19954), -INT16_C(  4111), -INT16_C( 14998), -INT16_C( 20118),  INT16_C( 25517), -INT16_C( 12368), -INT16_C( 29793),  INT16_C( 15573) },
      {      INT8_MAX,  INT8_C(  89),  INT8_C(  44),  INT8_C( 120),      INT8_MAX,  INT8_C(  95),      INT8_MAX,  INT8_C(  63),
             INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,
             INT8_MAX,  INT8_C(  87),  INT8_C( 102),      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,
             INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN, -INT8_C(   7),      INT8_MAX,      INT8_MAX,
             INT8_MAX,      INT8_MAX,  INT8_C(  34),      INT8_MAX,  INT8_C( 125),      INT8_MAX,      INT8_MAX,  INT8_C(   5),
             INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX,
         INT8_C(  46),      INT8_MAX,      INT8_MAX,  INT8_C(  53),  INT8_C(  41),  INT8_C( 126),  INT8_C(  57),      INT8_MAX,
             INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX } },
    { {  INT16_C(   153),  INT16_C(   240),  INT16_C(   207),  INT16_C(    28),  INT16_C(   117),  INT16_C(   127),  INT16_C(    17),  INT16_C(    66),
         INT16_C(   194),  INT16_C(   201),  INT16_C(   155),  INT16_C(    53),  INT16_C(    42),  INT16_C(   157),  INT16_C(   217),  INT16_C(   225),
         INT16_C(    54),  INT16_C(   101),  INT16_C(    41),  INT16_C(    62),  INT16_C(   123),  INT16_C(   238),  INT16_C(   142),  INT16_C(   235),
         INT16_C(    97),  INT16_C(   216),  INT16_C(    46),  INT16_C(   251),  INT16_C(    17),  INT16_C(    14),  INT16_C(   114),  INT16_C(    93) },
      { -INT16_C( 16216), -INT16_C( 21054),  INT16_C( 17641), -INT16_C( 30485), -INT16_C( 22081),  INT16_C( 19574), -INT16_C( 22985), -INT16_C( 30664),
        -INT16_C(  5113),  INT16_C(  1120),  INT16_C( 27931),  INT16_C( 29440), -INT16_C( 26242),  INT16_C( 26753),  INT16_C( 28683), -INT16_C( 19259),
        -INT16_C( 30671),  INT16_C(  6753),  INT16_C( 19916), -INT16_C( 29790),  INT16_C(  6390),  INT16_C( 11736),  INT16_C(  4286), -INT16_C( 14667),
         INT16_C(  5628),  INT16_C(  6090), -INT16_C( 13694),  INT16_C(   139),  INT16_C(  3171),  INT16_C( 28521),  INT16_C( 11901), -INT16_C( 20957) },
      {      INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C(  28),  INT8_C( 117),      INT8_MAX,  INT8_C(  17),  INT8_C(  66),
             INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MIN,
             INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C(  53),  INT8_C(  42),      INT8_MAX,      INT8_MAX,      INT8_MAX,
             INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,
         INT8_C(  54),  INT8_C( 101),  INT8_C(  41),  INT8_C(  62),  INT8_C( 123),      INT8_MAX,      INT8_MAX,      INT8_MAX,
             INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,
         INT8_C(  97),      INT8_MAX,  INT8_C(  46),      INT8_MAX,  INT8_C(  17),  INT8_C(  14),  INT8_C( 114),  INT8_C(  93),
             INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN } },
    { {  INT16_C(   132),  INT16_C(   130),  INT16_C(   107),  INT16_C(   199),  INT16_C(   230),  INT16_C(    66),  INT16_C(   170),  INT16_C(   242),
         INT16_C(   210),  INT16_C(    66),  INT16_C(   149),  INT16_C(     0),  INT16_C(   172),  INT16_C(    30),  INT16_C(   146),  INT16_C(   145),
         INT16_C(   149),  INT16_C(   232),  INT16_C(    33),  INT16_C(   131),  INT16_C(   165),  INT16_C(   253),  INT16_C(   205),  INT16_C(    15),
         INT16_C(   250),  INT16_C(    61),  INT16_C(   149),  INT16_C(    48),  INT16_C(   173),  INT16_C(    27),  INT16_C(    27),  INT16_C(    86) },
      { -INT16_C( 16208), -INT16_C( 20417), -INT16_C(  4127), -INT16_C(  5836), -INT16_C(  1644), -INT16_C(  7194), -INT16_C( 10553),  INT16_C( 26611),
         INT16_C( 17872),  INT16_C( 24484), -INT16_C(  7718),  INT16_C(  7056), -INT16_C(  8306), -INT16_C( 12746), -INT16_C(  7174), -INT16_C( 21724),
         INT16_C( 25507), -INT16_C( 31653), -INT16_C( 28846), -INT16_C(  6547),  INT16_C( 21641),  INT16_C( 20682), -INT16_C( 17110), -INT16_C(  1097),
         INT16_C( 23298), -INT16_C(  9126), -INT16_C(  5572), -INT16_C( 13321),  INT16_C( 11721), -INT16_C( 15207), -INT16_C( 17136), -INT16_C( 19601) },
      {      INT8_MAX,      INT8_MAX,  INT8_C( 107),      INT8_MAX,      INT8_MAX,  INT8_C(  66),      INT8_MAX,      INT8_MAX,
             INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,
             INT8_MAX,  INT8_C(  66),      INT8_MAX,  INT8_C(   0),      INT8_MAX,  INT8_C(  30),      INT8_MAX,      INT8_MAX,
             INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,
             INT8_MAX,      INT8_MAX,  INT8_C(  33),      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C(  15),
             INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,
             INT8_MAX,  INT8_C(  61),      INT8_MAX,  INT8_C(  48),      INT8_MAX,  INT8_C(  27),  INT8_C(  27),  INT8_C(  86),
             INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN } },
    { {  INT16_C(   202),  INT16_C(   115),  INT16_C(   165),  INT16_C(   227),  INT16_C(    36),  INT16_C(    36),  INT16_C(   234),  INT16_C(   227),
         INT16_C(   121),  INT16_C(   129),  INT16_C(   182),  INT16_C(    45),  INT16_C(   229),  INT16_C(   244),  INT16_C(    96),  INT16_C(   196),
         INT16_C(   223),  INT16_C(   133),  INT16_C(   145),  INT16_C(   126),  INT16_C(   155),  INT16_C(   150),  INT16_C(   193),  INT16_C(   202),
         INT16_C(    56),  INT16_C(   159),  INT16_C(   152),  INT16_C(   210),  INT16_C(   190),  INT16_C(    32),  INT16_C(   109),  INT16_C(    73) },
      {  INT16_C(  7245), -INT16_C( 11570),  INT16_C( 13997),  INT16_C( 25424), -INT16_C(  3119),  INT16_C( 22265),  INT16_C( 29620), -INT16_C(  4320),
         INT16_C( 27819), -INT16_C( 25970),  INT16_C( 23300), -INT16_C( 32404),  INT16_C( 12825),  INT16_C( 14242), -INT16_C( 31073), -INT16_C(  4991),
         INT16_C( 20386),  INT16_C( 20670),  INT16_C(  3974),  INT16_C( 22451), -INT16_C( 21502), -INT16_C( 18770), -INT16_C( 12769), -INT16_C( 13402),
         INT16_C( 13370),  INT16_C( 15973), -INT16_C( 11889), -INT16_C( 22336),  INT16_C( 25091), -INT16_C( 23840),  INT16_C( 25064), -INT16_C( 29809) },
      {      INT8_MAX,  INT8_C( 115),      INT8_MAX,      INT8_MAX,  INT8_C(  36),  INT8_C(  36),      INT8_MAX,      INT8_MAX,
             INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,
         INT8_C( 121),      INT8_MAX,      INT8_MAX,  INT8_C(  45),      INT8_MAX,      INT8_MAX,  INT8_C(  96),      INT8_MAX,
             INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,
             INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C( 126),      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,
             INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,
         INT8_C(  56),      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C(  32),  INT8_C( 109),  INT8_C(  73),
             INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MIN } },
    { {  INT16_C(    77),  INT16_C(    54),  INT16_C(   142),  INT16_C(    94),  INT16_C(    60),  INT16_C(    90),  INT16_C(   187),  INT16_C(    69),
         INT16_C(   138),  INT16_C(   127),  INT16_C(    67),  INT16_C(    94),  INT16_C(     7),  INT16_C(   142),  INT16_C(   143),  INT16_C(    25),
         INT16_C(   244),  INT16_C(    57),  INT16_C(   221),  INT16_C(   188),  INT16_C(   173),  INT16_C(    36),  INT16_C(    59),  INT16_C(    87),
         INT16_C(   236),  INT16_C(    32),  INT16_C(   254),  INT16_C(   213),  INT16_C(   127),  INT16_C(   110),  INT16_C(   124),  INT16_C(   235) },
      { -INT16_C( 10640), -INT16_C(  3547), -INT16_C( 16972), -INT16_C( 12881), -INT16_C( 14998), -INT16_C( 11535),  INT16_C( 23041), -INT16_C( 14807),
         INT16_C(    71),  INT16_C( 30695),  INT16_C( 26110),  INT16_C(   844), -INT16_C( 20252), -INT16_C(  3215), -INT16_C(  2004), -INT16_C( 25122),
         INT16_C(   975), -INT16_C( 31857),  INT16_C( 16064),  INT16_C( 10832),  INT16_C( 16900),  INT16_C(  1532),  INT16_C(  9884), -INT16_C(  7221),
        -INT16_C( 19930),  INT16_C(  9306), -INT16_C( 22760), -INT16_C(   985), -INT16_C( 26281), -INT16_C( 31761), -INT16_C( 12655),  INT16_C( 24608) },
      {  INT8_C(  77),  INT8_C(  54),      INT8_MAX,  INT8_C(  94),  INT8_C(  60),  INT8_C(  90),      INT8_MAX,  INT8_C(  69),
             INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MIN,
             INT8_MAX,      INT8_MAX,  INT8_C(  67),  INT8_C(  94),  INT8_C(   7),      INT8_MAX,      INT8_MAX,  INT8_C(  25),
         INT8_C(  71),      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,
             INT8_MAX,  INT8_C(  57),      INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C(  36),  INT8_C(  59),  INT8_C(  87),
             INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,
             INT8_MAX,  INT8_C(  32),      INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C( 110),  INT8_C( 124),      INT8_MAX,
             INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX } },
    { {  INT16_C(   176),  INT16_C(   146),  INT16_C(    52),  INT16_C(   242),  INT16_C(   185),  INT16_C(    18),  INT16_C(   195),  INT16_C(     5),
         INT16_C(    80),  INT16_C(   141),  INT16_C(    80),  INT16_C(    78),  INT16_C(   121),  INT16_C(   123),  INT16_C(   242),  INT16_C(    25),
         INT16_C(   191),  INT16_C(   145),  INT16_C(   103),  INT16_C(   105),  INT16_C(   123),  INT16_C(   255),  INT16_C(   113),  INT16_C(   179),
         INT16_C(    45),  INT16_C(   185),  INT16_C(   203),  INT16_C(   103),  INT16_C(   218),  INT16_C(   140),  INT16_C(   190),  INT16_C(   111) },
      {  INT16_C( 20605),  INT16_C( 28672), -INT16_C( 31817), -INT16_C( 10023),  INT16_C( 21758),  INT16_C( 15575), -INT16_C(  9018), -INT16_C( 30480),
         INT16_C( 12553), -INT16_C( 30911),  INT16_C( 18940),  INT16_C( 16623), -INT16_C( 11997), -INT16_C(  3892),  INT16_C( 29071),  INT16_C(  3167),
         INT16_C( 24513),  INT16_C( 31100),  INT16_C( 21986), -INT16_C(  7855),  INT16_C( 10410),  INT16_C( 28701),  INT16_C(  3332),  INT16_C(  3832),
         INT16_C( 14654),  INT16_C( 14997), -INT16_C( 31613), -INT16_C( 22917),  INT16_C( 18262), -INT16_C(  6762), -INT16_C(  2631),  INT16_C( 31474) },
      {      INT8_MAX,      INT8_MAX,  INT8_C(  52),      INT8_MAX,      INT8_MAX,  INT8_C(  18),      INT8_MAX,  INT8_C(   5),
             INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,
         INT8_C(  80),      INT8_MAX,  INT8_C(  80),  INT8_C(  78),  INT8_C( 121),  INT8_C( 123),      INT8_MAX,  INT8_C(  25),
             INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MAX,
             INT8_MAX,      INT8_MAX,  INT8_C( 103),  INT8_C( 105),  INT8_C( 123),      INT8_MAX,  INT8_C( 113),      INT8_MAX,
             INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,
         INT8_C(  45),      INT8_MAX,      INT8_MAX,  INT8_C( 103),      INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C( 111),
             INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX } },
    { {  INT16_C(   110),  INT16_C(    55),  INT16_C(    68),  INT16_C(   110),  INT16_C(    53),  INT16_C(   113),  INT16_C(   214),  INT16_C(   129),
         INT16_C(    21),  INT16_C(   146),  INT16_C(    55),  INT16_C(   239),  INT16_C(   207),  INT16_C(    55),  INT16_C(   199),  INT16_C(    25),
         INT16_C(   165),  INT16_C(   249),  INT16_C(   104),  INT16_C(    87),  INT16_C(    69),  INT16_C(   225),  INT16_C(    72),  INT16_C(    43),
         INT16_C(    30),  INT16_C(   246),  INT16_C(   246),  INT16_C(   212),  INT16_C(   187),  INT16_C(   139),  INT16_C(   189),  INT16_C(   183) },
      { -INT16_C(  2717),  INT16_C( 19889),  INT16_C(  6237), -INT16_C(  1116),  INT16_C( 27742),  INT16_C( 31196),  INT16_C( 16308),  INT16_C(  4516),
         INT16_C( 25181), -INT16_C( 19704), -INT16_C(  4520),  INT16_C(  7815), -INT16_C( 27991),  INT16_C( 11177),  INT16_C( 20048), -INT16_C( 19486),
        -INT16_C( 27837), -INT16_C( 24576), -INT16_C( 23380),  INT16_C(  2716),  INT16_C( 30736), -INT16_C( 14973),  INT16_C( 10423),  INT16_C(  5590),
        -INT16_C(  8566), -INT16_C(  7480),  INT16_C( 20428),  INT16_C( 29953), -INT16_C( 21791),  INT16_C( 12704), -INT16_C( 31752),  INT16_C( 15332) },
      {  INT8_C( 110),  INT8_C(  55),  INT8_C(  68),  INT8_C( 110),  INT8_C(  53),  INT8_C( 113),      INT8_MAX,      INT8_MAX,
             INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,
         INT8_C(  21),      INT8_MAX,  INT8_C(  55),      INT8_MAX,      INT8_MAX,  INT8_C(  55),      INT8_MAX,  INT8_C(  25),
             INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,
             INT8_MAX,      INT8_MAX,  INT8_C( 104),  INT8_C(  87),  INT8_C(  69),      INT8_MAX,  INT8_C(  72),  INT8_C(  43),
             INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MAX,
         INT8_C(  30),      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,
             INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MAX } },
    { {  INT16_C(   228),  INT16_C(   194),  INT16_C(   120),  INT16_C(   153),  INT16_C(    80),  INT16_C(   168),  INT16_C(    52),  INT16_C(     2),
         INT16_C(   133),  INT16_C(   223),  INT16_C(   229),  INT16_C(   181),  INT16_C(   245),  INT16_C(   136),  INT16_C(   203),  INT16_C(   143),
         INT16_C(   160),  INT16_C(    56),  INT16_C(    30),  INT16_C(     8),  INT16_C(    47),  INT16_C(   230),  INT16_C(   109),  INT16_C(   119),
         INT16_C(   204),  INT16_C(   198),  INT16_C(   171),  INT16_C(    66),  INT16_C(    99),  INT16_C(    25),  INT16_C(   142),  INT16_C(   222) },
      { -INT16_C(  1490),  INT16_C( 17943), -INT16_C(  6120), -INT16_C( 31153), -INT16_C(   232),  INT16_C( 31852),  INT16_C( 21613),  INT16_C( 24563),
         INT16_C( 18720), -INT16_C( 11738), -INT16_C( 23819), -INT16_C( 27116), -INT16_C(  8443),  INT16_C( 13231),  INT16_C( 22637), -INT16_C( 25582),
         INT16_C( 10578),  INT16_C( 27362),  INT16_C( 12561),  INT16_C( 10736),  INT16_C( 23601), -INT16_C( 24923), -INT16_C( 26448), -INT16_C( 12035),
         INT16_C(  9186), -INT16_C( 10333), -INT16_C( 18491), -INT16_C( 13715),  INT16_C(  7318),  INT16_C(  1278),  INT16_C(  4212), -INT16_C( 14688) },
      {      INT8_MAX,      INT8_MAX,  INT8_C( 120),      INT8_MAX,  INT8_C(  80),      INT8_MAX,  INT8_C(  52),  INT8_C(   2),
             INT8_MIN,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MAX,
             INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,
             INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MIN,
             INT8_MAX,  INT8_C(  56),  INT8_C(  30),  INT8_C(   8),  INT8_C(  47),      INT8_MAX,  INT8_C( 109),  INT8_C( 119),
             INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,
             INT8_MAX,      INT8_MAX,      INT8_MAX,  INT8_C(  66),  INT8_C(  99),  INT8_C(  25),      INT8_MAX,      INT8_MAX,
             INT8_MAX,      INT8_MIN,      INT8_MIN,      INT8_MIN,      INT8_MAX,      INT8_MAX,      INT8_MAX,      INT8_MIN } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_packs_epi16(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_packs_epi16");
    easysimd_test_x86_assert_equal_i8x64(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_packs_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int16_t r[32];
  } test_vec[] = {
    { {  INT32_C(  1800617241), -INT32_C(   686819306),  INT32_C(   140214962),  INT32_C(  1970280150), -INT32_C(  1837652367), -INT32_C(   601751898), -INT32_C(   689735000), -INT32_C(   924571217),
         INT32_C(  2083731302), -INT32_C(   497859792), -INT32_C(  1679118651),  INT32_C(   907041733), -INT32_C(  1463253247),  INT32_C(   780462469),  INT32_C(   319055716), -INT32_C(   153359984) },
      {  INT32_C(       20594),  INT32_C(        5683),  INT32_C(       14769),  INT32_C(       31344),  INT32_C(       53026),  INT32_C(       34557),  INT32_C(       40345),  INT32_C(        2963),
         INT32_C(       54363),  INT32_C(       16618),  INT32_C(        9337),  INT32_C(       42910),  INT32_C(       17526),  INT32_C(       29642),  INT32_C(       24336),  INT32_C(       22890) },
      {       INT16_MAX,       INT16_MIN,       INT16_MAX,       INT16_MAX,  INT16_C( 20594),  INT16_C(  5683),  INT16_C( 14769),  INT16_C( 31344),
              INT16_MIN,       INT16_MIN,       INT16_MIN,       INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C(  2963),
              INT16_MAX,       INT16_MIN,       INT16_MIN,       INT16_MAX,       INT16_MAX,  INT16_C( 16618),  INT16_C(  9337),       INT16_MAX,
              INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MIN,  INT16_C( 17526),  INT16_C( 29642),  INT16_C( 24336),  INT16_C( 22890) } },
    { {  INT32_C(  1949157033),  INT32_C(    11802708),  INT32_C(   304426676),  INT32_C(   968475415), -INT32_C(   360894332),  INT32_C(   408831907), -INT32_C(  2122813782), -INT32_C(  1143217646),
        -INT32_C(    47249240), -INT32_C(   721558496),  INT32_C(   702947858),  INT32_C(  1784914150),  INT32_C(  1934942416), -INT32_C(   762531288),  INT32_C(   861144097), -INT32_C(  1880215578) },
      {  INT32_C(       22668),  INT32_C(        4908),  INT32_C(       37437),  INT32_C(       33788),  INT32_C(       43254),  INT32_C(        9339),  INT32_C(       27991),  INT32_C(       13820),
         INT32_C(       25741),  INT32_C(       48503),  INT32_C(       32847),  INT32_C(       54531),  INT32_C(       40829),  INT32_C(         707),  INT32_C(       50543),  INT32_C(       49659) },
      {       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C( 22668),  INT16_C(  4908),       INT16_MAX,       INT16_MAX,
              INT16_MIN,       INT16_MAX,       INT16_MIN,       INT16_MIN,       INT16_MAX,  INT16_C(  9339),  INT16_C( 27991),  INT16_C( 13820),
              INT16_MIN,       INT16_MIN,       INT16_MAX,       INT16_MAX,  INT16_C( 25741),       INT16_MAX,       INT16_MAX,       INT16_MAX,
              INT16_MAX,       INT16_MIN,       INT16_MAX,       INT16_MIN,       INT16_MAX,  INT16_C(   707),       INT16_MAX,       INT16_MAX } },
    { {  INT32_C(    86345971),  INT32_C(   264412482), -INT32_C(  1500573103), -INT32_C(   109342115),  INT32_C(   144242828), -INT32_C(  1207280655), -INT32_C(   729908619), -INT32_C(   644449819),
         INT32_C(  1155447553), -INT32_C(  1437360040),  INT32_C(   273736626),  INT32_C(    17419125), -INT32_C(  1274436925),  INT32_C(  1936528637),  INT32_C(  1934093198),  INT32_C(  1699536228) },
      {  INT32_C(       61865),  INT32_C(       32155),  INT32_C(       21901),  INT32_C(       31319),  INT32_C(       13870),  INT32_C(         681),  INT32_C(       60022),  INT32_C(       26448),
         INT32_C(       47193),  INT32_C(       54837),  INT32_C(       38444),  INT32_C(       37648),  INT32_C(       22729),  INT32_C(       24922),  INT32_C(       12875),  INT32_C(       32922) },
      {       INT16_MAX,       INT16_MAX,       INT16_MIN,       INT16_MIN,       INT16_MAX,  INT16_C( 32155),  INT16_C( 21901),  INT16_C( 31319),
              INT16_MAX,       INT16_MIN,       INT16_MIN,       INT16_MIN,  INT16_C( 13870),  INT16_C(   681),       INT16_MAX,  INT16_C( 26448),
              INT16_MAX,       INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,
              INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C( 22729),  INT16_C( 24922),  INT16_C( 12875),       INT16_MAX } },
    { { -INT32_C(  1959201899), -INT32_C(   949850649), -INT32_C(  1973514704), -INT32_C(   199397871), -INT32_C(  2008225875), -INT32_C(  1091983526),  INT32_C(   183514231),  INT32_C(  1703578320),
         INT32_C(  1710277245),  INT32_C(  1613517360), -INT32_C(   236221728), -INT32_C(  1494873863),  INT32_C(  1227764463),  INT32_C(  1359419353),  INT32_C(   475789388),  INT32_C(     8513154) },
      {  INT32_C(       55397),  INT32_C(       42041),  INT32_C(        5526),  INT32_C(        7355),  INT32_C(       34917),  INT32_C(       19929),  INT32_C(       59241),  INT32_C(       50151),
         INT32_C(        8347),  INT32_C(       64196),  INT32_C(        9487),  INT32_C(       34113),  INT32_C(       46605),  INT32_C(       30723),  INT32_C(       13664),  INT32_C(       46072) },
      {       INT16_MIN,       INT16_MIN,       INT16_MIN,       INT16_MIN,       INT16_MAX,       INT16_MAX,  INT16_C(  5526),  INT16_C(  7355),
              INT16_MIN,       INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C( 19929),       INT16_MAX,       INT16_MAX,
              INT16_MAX,       INT16_MAX,       INT16_MIN,       INT16_MIN,  INT16_C(  8347),       INT16_MAX,  INT16_C(  9487),       INT16_MAX,
              INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C( 30723),  INT16_C( 13664),       INT16_MAX } },
    { {  INT32_C(  1926468500),  INT32_C(  1617729640), -INT32_C(   913998862), -INT32_C(    95500731), -INT32_C(  2135925907),  INT32_C(  1543091009),  INT32_C(  2022725920),  INT32_C(   875268256),
        -INT32_C(  2069430500), -INT32_C(  1981541737), -INT32_C(   749573491), -INT32_C(  1647468496),  INT32_C(  1008631291),  INT32_C(  1368921904),  INT32_C(   281618544), -INT32_C(   851053391) },
      {  INT32_C(       35409),  INT32_C(       35604),  INT32_C(       53342),  INT32_C(         621),  INT32_C(       55615),  INT32_C(        4650),  INT32_C(       45091),  INT32_C(       56189),
         INT32_C(       20837),  INT32_C(       41949),  INT32_C(       59251),  INT32_C(        4073),  INT32_C(        4072),  INT32_C(       65313),  INT32_C(       60847),  INT32_C(         200) },
      {       INT16_MAX,       INT16_MAX,       INT16_MIN,       INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C(   621),
              INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C(  4650),       INT16_MAX,       INT16_MAX,
              INT16_MIN,       INT16_MIN,       INT16_MIN,       INT16_MIN,  INT16_C( 20837),       INT16_MAX,       INT16_MAX,  INT16_C(  4073),
              INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MIN,  INT16_C(  4072),       INT16_MAX,       INT16_MAX,  INT16_C(   200) } },
    { { -INT32_C(    11457029), -INT32_C(  2019348825), -INT32_C(   781314454),  INT32_C(  1692424183),  INT32_C(  2138294656), -INT32_C(   511798053), -INT32_C(  2050085159),  INT32_C(  1451595355),
         INT32_C(  1784076227),  INT32_C(  1878128901),  INT32_C(   121659151),  INT32_C(   929767863), -INT32_C(   977871126),  INT32_C(  1269183858), -INT32_C(  1093569437), -INT32_C(   837528054) },
      {  INT32_C(       13112),  INT32_C(       29602),  INT32_C(       16506),  INT32_C(       61047),  INT32_C(       13747),  INT32_C(       50817),  INT32_C(       55684),  INT32_C(       54951),
         INT32_C(       26121),  INT32_C(       37849),  INT32_C(       37587),  INT32_C(       64384),  INT32_C(       56369),  INT32_C(       23714),  INT32_C(       44085),  INT32_C(       49538) },
      {       INT16_MIN,       INT16_MIN,       INT16_MIN,       INT16_MAX,  INT16_C( 13112),  INT16_C( 29602),  INT16_C( 16506),       INT16_MAX,
              INT16_MAX,       INT16_MIN,       INT16_MIN,       INT16_MAX,  INT16_C( 13747),       INT16_MAX,       INT16_MAX,       INT16_MAX,
              INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C( 26121),       INT16_MAX,       INT16_MAX,       INT16_MAX,
              INT16_MIN,       INT16_MAX,       INT16_MIN,       INT16_MIN,       INT16_MAX,  INT16_C( 23714),       INT16_MAX,       INT16_MAX } },
    { { -INT32_C(   987198532), -INT32_C(   984088265), -INT32_C(  1923601323), -INT32_C(   259401609), -INT32_C(  1697859060),  INT32_C(  1895263852),  INT32_C(  1377578132), -INT32_C(   988504311),
         INT32_C(  1636449322), -INT32_C(  1842879683), -INT32_C(  2044690673),  INT32_C(  1685498199), -INT32_C(   805420445),  INT32_C(  1145042352),  INT32_C(   731274018),  INT32_C(   636529402) },
      {  INT32_C(        9350),  INT32_C(       27830),  INT32_C(       34034),  INT32_C(       58088),  INT32_C(       23217),  INT32_C(       65182),  INT32_C(       17961),  INT32_C(       50795),
         INT32_C(       61930),  INT32_C(       52317),  INT32_C(       63056),  INT32_C(       25561),  INT32_C(       64189),  INT32_C(       51192),  INT32_C(       28685),  INT32_C(       52790) },
      {       INT16_MIN,       INT16_MIN,       INT16_MIN,       INT16_MIN,  INT16_C(  9350),  INT16_C( 27830),       INT16_MAX,       INT16_MAX,
              INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MIN,  INT16_C( 23217),       INT16_MAX,  INT16_C( 17961),       INT16_MAX,
              INT16_MAX,       INT16_MIN,       INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C( 25561),
              INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C( 28685),       INT16_MAX } },
    { {  INT32_C(   180297835),  INT32_C(   953556161),  INT32_C(   623781484),  INT32_C(  2106066782),  INT32_C(   225920402),  INT32_C(   852783265), -INT32_C(   861675119), -INT32_C(   979707558),
        -INT32_C(  1143973382), -INT32_C(   487348619),  INT32_C(   721887693), -INT32_C(  1146581207),  INT32_C(  1992827092),  INT32_C(   564698256), -INT32_C(   655537283),  INT32_C(   530417445) },
      {  INT32_C(       22234),  INT32_C(       57656),  INT32_C(        5900),  INT32_C(       41682),  INT32_C(       25880),  INT32_C(       46214),  INT32_C(       12684),  INT32_C(       56400),
         INT32_C(       43826),  INT32_C(       59020),  INT32_C(       30717),  INT32_C(        8729),  INT32_C(       41351),  INT32_C(       65365),  INT32_C(       32049),  INT32_C(       41305) },
      {       INT16_MAX,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C( 22234),       INT16_MAX,  INT16_C(  5900),       INT16_MAX,
              INT16_MAX,       INT16_MAX,       INT16_MIN,       INT16_MIN,  INT16_C( 25880),       INT16_MAX,  INT16_C( 12684),       INT16_MAX,
              INT16_MIN,       INT16_MIN,       INT16_MAX,       INT16_MIN,       INT16_MAX,       INT16_MAX,  INT16_C( 30717),  INT16_C(  8729),
              INT16_MAX,       INT16_MAX,       INT16_MIN,       INT16_MAX,       INT16_MAX,       INT16_MAX,  INT16_C( 32049),       INT16_MAX } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_packs_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_packs_epi32");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_packs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_packs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_packs_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_packs_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_packs_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_packs_epi32)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
