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

#define EASYSIMD_TEST_X86_AVX512_INSN mullo

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/mullo.h>

static int
test_easysimd_mm_mullo_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 7003481009132932494), -INT64_C( 7667798585923800569) },
      {  INT64_C( 6639436646181008104), -INT64_C( 1194794770925058758) },
      {  INT64_C( 8134128333384481616),  INT64_C( 5428086916014176406) } },
    { { -INT64_C( 8796497104294716882),  INT64_C( 1938781159720333443) },
      {  INT64_C( 8732500842347641697),  INT64_C( 2235008008049178008) },
      { -INT64_C( 2563302690534636178),  INT64_C( 2559411276045446344) } },
    { { -INT64_C( 2795511239844968911), -INT64_C( 2498070093529117430) },
      { -INT64_C( 1223041383134051251),  INT64_C( 8910099503978443415) },
      {  INT64_C( 6108241859548192957), -INT64_C( 8607368767973120794) } },
    { {  INT64_C( 8646452157465892046), -INT64_C( 8483646742528316693) },
      {  INT64_C( 8039801476561172044),  INT64_C( 7986182450776086287) },
      { -INT64_C( 1991157513899204312),  INT64_C( 1603652410206396613) } },
    { {  INT64_C( 5302868014911455154), -INT64_C( 6952725137998198084) },
      { -INT64_C( 4309146916798424597),  INT64_C( 8753857237457119597) },
      {  INT64_C( 9210923512008612454), -INT64_C( 6838806124820173300) } },
    { {  INT64_C( 6743911336865435633),  INT64_C( 2912659343957903481) },
      { -INT64_C( 1086742435948941967), -INT64_C( 1475358812545415098) },
      {  INT64_C( 2991293503853553249), -INT64_C( 3726215866947561194) } },
    { { -INT64_C( 4069616554646823289),  INT64_C( 2629819278636079897) },
      { -INT64_C( 3738210315258926676),  INT64_C( 2027850822437116990) },
      {  INT64_C( 2869612186737041844),  INT64_C( 5971484610053721102) } },
    { {  INT64_C( 5301803492984512126), -INT64_C( 3351770876921043035) },
      {  INT64_C( 4240737166549894791), -INT64_C( 7931799831860562885) },
      {  INT64_C( 1623154472100513906),  INT64_C( 2852638113367860999) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mullo_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mullo_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_mullo_epi64(a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_mullo_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { { -INT16_C(  7516), -INT16_C( 20891),  INT16_C( 31761), -INT16_C( 27740), -INT16_C( 26871), -INT16_C( 14527),  INT16_C(  8140),  INT16_C( 16510) },
      UINT8_C( 72),
      { -INT16_C(  2927),  INT16_C( 23693), -INT16_C(  7103), -INT16_C( 26974),  INT16_C(   722), -INT16_C( 22382), -INT16_C(   822), -INT16_C( 21428) },
      { -INT16_C(  1183), -INT16_C(  8514),  INT16_C( 20895),  INT16_C( 14055), -INT16_C( 20590), -INT16_C( 19966),  INT16_C( 17197), -INT16_C( 16390) },
      { -INT16_C(  7516), -INT16_C( 20891),  INT16_C( 31761),  INT16_C(  6190), -INT16_C( 26871), -INT16_C( 14527),  INT16_C( 19842),  INT16_C( 16510) } },
    { { -INT16_C( 30665),  INT16_C( 30747), -INT16_C( 17044),  INT16_C( 15886), -INT16_C( 24385), -INT16_C( 30233),  INT16_C( 13212), -INT16_C(   458) },
      UINT8_C( 46),
      { -INT16_C(  8972),  INT16_C( 17870),  INT16_C(  1219),  INT16_C( 29399), -INT16_C( 30457),  INT16_C( 19104),  INT16_C( 24452),  INT16_C(  3201) },
      { -INT16_C(  1670),  INT16_C( 14200), -INT16_C( 18937), -INT16_C( 22537), -INT16_C( 32611), -INT16_C( 11965),  INT16_C( 16822), -INT16_C( 21761) },
      { -INT16_C( 30665), -INT16_C(  1392), -INT16_C( 15531),  INT16_C(  3697), -INT16_C( 24385),  INT16_C( 10208),  INT16_C( 13212), -INT16_C(   458) } },
    { { -INT16_C( 13027), -INT16_C(  7697), -INT16_C( 14382), -INT16_C(  9901), -INT16_C(  3248), -INT16_C( 11229), -INT16_C( 23470), -INT16_C( 13088) },
      UINT8_C(157),
      {  INT16_C(  1112),  INT16_C(  4004),  INT16_C( 19451),  INT16_C( 31660),  INT16_C( 32142), -INT16_C( 12238), -INT16_C(  9091),  INT16_C( 19181) },
      { -INT16_C( 12596), -INT16_C( 27876), -INT16_C(  2782),  INT16_C(  5603), -INT16_C( 18408), -INT16_C( 17304),  INT16_C( 13464), -INT16_C(  3751) },
      {  INT16_C( 17952), -INT16_C(  7697),  INT16_C( 20054), -INT16_C( 14972), -INT16_C( 10928), -INT16_C( 11229), -INT16_C( 23470),  INT16_C( 10597) } },
    { { -INT16_C(   712),  INT16_C( 13056), -INT16_C( 21432), -INT16_C( 10321), -INT16_C(  7894), -INT16_C( 22617), -INT16_C( 27459), -INT16_C( 30223) },
      UINT8_C( 99),
      {  INT16_C(  7182),  INT16_C(   901), -INT16_C( 26112), -INT16_C( 18404), -INT16_C( 10238),  INT16_C( 14160),  INT16_C( 16690),  INT16_C( 12143) },
      { -INT16_C( 23743), -INT16_C(  4488),  INT16_C( 20306),  INT16_C( 13080), -INT16_C( 16394), -INT16_C( 29968),  INT16_C( 31408), -INT16_C( 16659) },
      {  INT16_C(  2446),  INT16_C( 19544), -INT16_C( 21432), -INT16_C( 10321), -INT16_C(  7894), -INT16_C(  1280), -INT16_C( 22944), -INT16_C( 30223) } },
    { {  INT16_C( 29334), -INT16_C( 26942), -INT16_C(  8691),  INT16_C(  3918), -INT16_C( 24650), -INT16_C(  6074), -INT16_C( 18720),  INT16_C(  8728) },
      UINT8_C( 89),
      {  INT16_C(  4240), -INT16_C(  8277), -INT16_C(  8664), -INT16_C(  6187),  INT16_C( 24526),  INT16_C( 18583),  INT16_C( 22093), -INT16_C( 16417) },
      {  INT16_C( 29976), -INT16_C(  2356), -INT16_C(  9020),  INT16_C( 25516), -INT16_C( 27358), -INT16_C( 10173),  INT16_C( 26029),  INT16_C( 15665) },
      {  INT16_C( 23936), -INT16_C( 26942), -INT16_C(  8691),  INT16_C(  8732), -INT16_C( 24740), -INT16_C(  6074), -INT16_C( 19703),  INT16_C(  8728) } },
    { { -INT16_C(  9099), -INT16_C( 25316), -INT16_C(  3654), -INT16_C( 30332),  INT16_C(  7248), -INT16_C( 25135), -INT16_C( 20366), -INT16_C( 30115) },
      UINT8_C( 38),
      { -INT16_C( 32727),  INT16_C(  1514),  INT16_C( 19756), -INT16_C( 16088),  INT16_C(   144), -INT16_C(  2450), -INT16_C( 21710),  INT16_C(  3691) },
      {  INT16_C(  2503), -INT16_C( 18231),  INT16_C( 21133), -INT16_C( 22263), -INT16_C( 23005), -INT16_C( 11237), -INT16_C( 23293),  INT16_C( 11770) },
      { -INT16_C(  9099), -INT16_C( 11078), -INT16_C( 26308), -INT16_C( 30332),  INT16_C(  7248),  INT16_C(  5530), -INT16_C( 20366), -INT16_C( 30115) } },
    { { -INT16_C(  7131),  INT16_C( 21042),  INT16_C( 23089), -INT16_C( 16109), -INT16_C( 32165), -INT16_C( 29257),  INT16_C(  9005), -INT16_C(  2661) },
      UINT8_C( 44),
      { -INT16_C( 21148), -INT16_C( 18759),  INT16_C( 25526),  INT16_C( 24026), -INT16_C( 20866),  INT16_C(  9312), -INT16_C( 29272), -INT16_C( 29623) },
      { -INT16_C( 25664),  INT16_C(  6845),  INT16_C( 32431),  INT16_C( 12661),  INT16_C(   566),  INT16_C( 22878),  INT16_C( 21406),  INT16_C(   645) },
      { -INT16_C(  7131),  INT16_C( 21042), -INT16_C( 17046), -INT16_C( 24926), -INT16_C( 32165), -INT16_C( 17600),  INT16_C(  9005), -INT16_C(  2661) } },
    { {  INT16_C( 15873), -INT16_C( 18503), -INT16_C( 27743),  INT16_C(  8212),  INT16_C( 30017), -INT16_C(  5820), -INT16_C( 29438), -INT16_C( 15755) },
      UINT8_C( 41),
      { -INT16_C(  8910), -INT16_C( 20264),  INT16_C(  2386),  INT16_C( 21990),  INT16_C( 16231), -INT16_C( 17421), -INT16_C(  2620),  INT16_C(   956) },
      {  INT16_C( 29614),  INT16_C( 16804), -INT16_C( 15224), -INT16_C(   638),  INT16_C( 27400), -INT16_C( 26881), -INT16_C( 15648),  INT16_C(  4799) },
      { -INT16_C( 12804), -INT16_C( 18503), -INT16_C( 27743), -INT16_C(  4916),  INT16_C( 30017), -INT16_C( 26355), -INT16_C( 29438), -INT16_C( 15755) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mullo_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mullo_epi16");
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
    easysimd__m128i r = easysimd_mm_mask_mullo_epi16(src, k, a, b);

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
test_easysimd_mm_maskz_mullo_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int16_t a[8];
    const int16_t b[8];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C(178),
      { -INT16_C( 13570), -INT16_C( 12895),  INT16_C(   807), -INT16_C( 15707),  INT16_C( 26574),  INT16_C( 11544), -INT16_C(  5415), -INT16_C(  2636) },
      { -INT16_C(  2982), -INT16_C( 28694),  INT16_C(  2149),  INT16_C( 20608),  INT16_C(  6361),  INT16_C( 15466),  INT16_C( 11964), -INT16_C( 17681) },
      {  INT16_C(     0), -INT16_C(  7126),  INT16_C(     0),  INT16_C(     0),  INT16_C( 19870),  INT16_C( 19440),  INT16_C(     0),  INT16_C( 11020) } },
    { UINT8_C(248),
      { -INT16_C( 30576), -INT16_C( 27872), -INT16_C(  7635), -INT16_C( 27550), -INT16_C( 28678), -INT16_C(  6802),  INT16_C( 25411),  INT16_C( 14399) },
      { -INT16_C( 12722),  INT16_C( 22173), -INT16_C(  4785),  INT16_C( 26416),  INT16_C( 27736), -INT16_C( 31197), -INT16_C(  8869), -INT16_C(  5250) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 16480), -INT16_C(  2576), -INT16_C(  3574),  INT16_C(  8145), -INT16_C( 31742) } },
    { UINT8_C(101),
      {  INT16_C( 32670), -INT16_C( 32622),  INT16_C( 10209),  INT16_C( 28795),  INT16_C( 24725), -INT16_C(  1869), -INT16_C(  5217),  INT16_C( 28230) },
      { -INT16_C( 25208),  INT16_C( 30397),  INT16_C(  9421),  INT16_C( 14798),  INT16_C( 21575),  INT16_C(  9365), -INT16_C( 32558),  INT16_C( 29066) },
      { -INT16_C( 19984),  INT16_C(     0), -INT16_C( 27859),  INT16_C(     0),  INT16_C(     0), -INT16_C(  5073), -INT16_C( 14226),  INT16_C(     0) } },
    {    UINT8_MAX,
      { -INT16_C(  3812),  INT16_C( 17376),  INT16_C( 20588), -INT16_C( 13096), -INT16_C( 12028), -INT16_C(  4244), -INT16_C(  9705), -INT16_C( 19336) },
      { -INT16_C(  4457), -INT16_C( 17535), -INT16_C( 17476),  INT16_C(  4098),  INT16_C(  9808), -INT16_C( 12062),  INT16_C( 21424), -INT16_C( 12848) },
      {  INT16_C( 16260), -INT16_C( 11296), -INT16_C(  3248),  INT16_C(  6576), -INT16_C(  5824),  INT16_C(  7512),  INT16_C( 25808), -INT16_C( 18048) } },
    { UINT8_C( 69),
      {  INT16_C(  4272),  INT16_C(   433),  INT16_C( 32489), -INT16_C( 17915), -INT16_C(  2838), -INT16_C( 15151), -INT16_C( 31124),  INT16_C( 23131) },
      {  INT16_C(  5639), -INT16_C( 15850),  INT16_C(  9752),  INT16_C( 15890), -INT16_C(  7415),  INT16_C( 23791), -INT16_C( 17229),  INT16_C( 25505) },
      { -INT16_C( 27440),  INT16_C(     0),  INT16_C( 31704),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 19844),  INT16_C(     0) } },
    { UINT8_C(204),
      {  INT16_C( 25683), -INT16_C( 11851),  INT16_C( 28521),  INT16_C( 24251),  INT16_C( 32577), -INT16_C( 14390),  INT16_C(  9690), -INT16_C(  3890) },
      { -INT16_C( 28357),  INT16_C( 25096),  INT16_C( 18083), -INT16_C( 31125), -INT16_C( 14539), -INT16_C(  3783), -INT16_C( 25239), -INT16_C( 17218) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 23077),  INT16_C( 31273),  INT16_C(     0),  INT16_C(     0),  INT16_C( 14442),  INT16_C(   228) } },
    { UINT8_C(  1),
      { -INT16_C( 29325), -INT16_C(  7317), -INT16_C( 14008), -INT16_C( 14556), -INT16_C(  5229), -INT16_C( 18271), -INT16_C( 28231),  INT16_C( 19188) },
      {  INT16_C( 22169), -INT16_C(  8210),  INT16_C( 29889), -INT16_C( 30699),  INT16_C(  1710),  INT16_C( 19441), -INT16_C( 21052),  INT16_C( 14412) },
      {  INT16_C( 11195),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 58),
      {  INT16_C(  7095), -INT16_C( 32638),  INT16_C( 18751),  INT16_C( 10772), -INT16_C( 13078),  INT16_C( 31715),  INT16_C( 11968),  INT16_C(  5652) },
      { -INT16_C(  3044), -INT16_C( 28457),  INT16_C( 24585),  INT16_C(  3902), -INT16_C( 30383), -INT16_C(    44),  INT16_C(  3286), -INT16_C( 29383) },
      {  INT16_C(     0),  INT16_C(  3374),  INT16_C(     0),  INT16_C( 23768),  INT16_C(  4106), -INT16_C( 19204),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi16(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mullo_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mullo_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_maskz_mullo_epi16(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_mullo_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(   238839905),  INT32_C(   122071456), -INT32_C(  1379780119), -INT32_C(   575635921) },
      UINT8_C(242),
      {  INT32_C(   427433813),  INT32_C(   220362657),  INT32_C(   955103350),  INT32_C(   248971383) },
      {  INT32_C(  1823394243),  INT32_C(  1230353679), -INT32_C(   159906952),  INT32_C(   149444019) },
      { -INT32_C(   238839905), -INT32_C(   988883089), -INT32_C(  1379780119), -INT32_C(   575635921) } },
    { {  INT32_C(   371352180),  INT32_C(  1344488665),  INT32_C(  1938297084), -INT32_C(   746495984) },
      UINT8_C( 41),
      { -INT32_C(   466075601), -INT32_C(  1738767979),  INT32_C(  1330402297), -INT32_C(  1631366085) },
      { -INT32_C(  1166550666),  INT32_C(   230082556), -INT32_C(  1340266160), -INT32_C(   640028246) },
      { -INT32_C(   600937302),  INT32_C(  1344488665),  INT32_C(  1938297084), -INT32_C(  1190888914) } },
    { { -INT32_C(   960622287), -INT32_C(  1939924333), -INT32_C(  1445221523),  INT32_C(  1984405503) },
      UINT8_C(120),
      { -INT32_C(  2039140162),  INT32_C(   282493671), -INT32_C(  1866758497),  INT32_C(  1891734623) },
      {  INT32_C(  1828948050), -INT32_C(  1831169817),  INT32_C(   177308523), -INT32_C(  1987901494) },
      { -INT32_C(   960622287), -INT32_C(  1939924333), -INT32_C(  1445221523),  INT32_C(   124431350) } },
    { {  INT32_C(   521140280),  INT32_C(   422569338), -INT32_C(   894768533), -INT32_C(   784700545) },
      UINT8_C(243),
      { -INT32_C(   824558018), -INT32_C(  1673958375),  INT32_C(    90653950), -INT32_C(  1086459705) },
      { -INT32_C(   466002689),  INT32_C(  1984909964),  INT32_C(  1760893436), -INT32_C(  1856256173) },
      { -INT32_C(  1919924286), -INT32_C(  1245831252), -INT32_C(   894768533), -INT32_C(   784700545) } },
    { {  INT32_C(   509556229), -INT32_C(  1598318174), -INT32_C(  1532615971),  INT32_C(   291758866) },
      UINT8_C( 64),
      { -INT32_C(   288557668),  INT32_C(  1575699012), -INT32_C(     5221576), -INT32_C(   469482834) },
      {  INT32_C(   981935009),  INT32_C(     1517534), -INT32_C(  1324172339), -INT32_C(  1158601954) },
      {  INT32_C(   509556229), -INT32_C(  1598318174), -INT32_C(  1532615971),  INT32_C(   291758866) } },
    { {  INT32_C(  1554627864),  INT32_C(   934909183), -INT32_C(  1791596057),  INT32_C(  1299856299) },
      UINT8_C( 94),
      {  INT32_C(   675055361),  INT32_C(  1526086815),  INT32_C(  1903798094),  INT32_C(  1435055000) },
      {  INT32_C(  1884612060),  INT32_C(   123178142), -INT32_C(    38539838),  INT32_C(  1750794343) },
      {  INT32_C(  1554627864), -INT32_C(  1071030750), -INT32_C(  1348600036), -INT32_C(  2126790104) } },
    { {  INT32_C(   647010183),  INT32_C(   562136787), -INT32_C(   980223443),  INT32_C(   169483053) },
      UINT8_C(  0),
      { -INT32_C(    73500049), -INT32_C(  1078090030),  INT32_C(  1495710553), -INT32_C(  1361015274) },
      { -INT32_C(  1518270690), -INT32_C(  2100124792),  INT32_C(  1370462261),  INT32_C(   559004082) },
      {  INT32_C(   647010183),  INT32_C(   562136787), -INT32_C(   980223443),  INT32_C(   169483053) } },
    { {  INT32_C(   102559796), -INT32_C(   272246122), -INT32_C(  1421284459), -INT32_C(  1755764615) },
      UINT8_C( 47),
      {  INT32_C(  2125937883), -INT32_C(  1481426673), -INT32_C(  1571224343),  INT32_C(  1188461398) },
      {  INT32_C(  1910299799), -INT32_C(  1945711711),  INT32_C(   990229011), -INT32_C(   429155061) },
      {  INT32_C(   764746029),  INT32_C(   542099311), -INT32_C(  1114480821),  INT32_C(  1707936434) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mullo_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mullo_epi32");
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
    easysimd__m128i r = easysimd_mm_mask_mullo_epi32(src, k, a, b);

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
test_easysimd_mm_maskz_mullo_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C( 39),
      {  INT32_C(    90574524), -INT32_C(   286224350), -INT32_C(  1582339213),  INT32_C(  1958593920) },
      { -INT32_C(    42119523),  INT32_C(   374246540),  INT32_C(  1827425632),  INT32_C(  1133738375) },
      {  INT32_C(  1709199692),  INT32_C(   280849048), -INT32_C(   149036512),  INT32_C(           0) } },
    { UINT8_C(135),
      { -INT32_C(  1985394439), -INT32_C(  1526884296),  INT32_C(   203726407), -INT32_C(  1431725988) },
      { -INT32_C(  1590254059),  INT32_C(  1107447029), -INT32_C(  1295421896), -INT32_C(    80147454) },
      {  INT32_C(   230322797), -INT32_C(   799847016),  INT32_C(   130294152),  INT32_C(           0) } },
    { UINT8_C( 85),
      {  INT32_C(  2072872419),  INT32_C(   549597570), -INT32_C(   310587819),  INT32_C(   503457400) },
      { -INT32_C(  1458330531),  INT32_C(   350377638),  INT32_C(   739677215), -INT32_C(  1333718323) },
      {  INT32_C(  1460146039),  INT32_C(           0),  INT32_C(  1085546571),  INT32_C(           0) } },
    { UINT8_C(151),
      {  INT32_C(  1075391246), -INT32_C(  1131070995), -INT32_C(   600538186),  INT32_C(   691622533) },
      { -INT32_C(  1127226522),  INT32_C(  1507583173),  INT32_C(   203819002), -INT32_C(  1750870136) },
      { -INT32_C(   209284716), -INT32_C(   114579615), -INT32_C(  1880712772),  INT32_C(           0) } },
    { UINT8_C(  2),
      { -INT32_C(   151988292), -INT32_C(   273896340),  INT32_C(   846563552), -INT32_C(  1516724542) },
      {  INT32_C(  1382700142),  INT32_C(   927777584), -INT32_C(  1061135895), -INT32_C(  1178380292) },
      {  INT32_C(           0), -INT32_C(   607741888),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 46),
      {  INT32_C(  1603973043), -INT32_C(   482375077), -INT32_C(  1650101761),  INT32_C(  1577798154) },
      {  INT32_C(  2005818804),  INT32_C(    56739498), -INT32_C(   570482298), -INT32_C(  1760839452) },
      {  INT32_C(           0), -INT32_C(  1968225682), -INT32_C(  2008249734), -INT32_C(   206364440) } },
    { UINT8_C(103),
      {  INT32_C(   801306277), -INT32_C(  1490115019), -INT32_C(  1766732725),  INT32_C(   894046167) },
      {  INT32_C(  1692385950),  INT32_C(  1156244003), -INT32_C(  1708537887),  INT32_C(  1996603602) },
      { -INT32_C(  2030550058),  INT32_C(   177046335),  INT32_C(  2115361515),  INT32_C(           0) } },
    { UINT8_C(183),
      {  INT32_C(  1793894340), -INT32_C(  1581935659),  INT32_C(  1417235268),  INT32_C(  1492299414) },
      {  INT32_C(  1870354061),  INT32_C(   122732352), -INT32_C(  1445336344), -INT32_C(  1335865108) },
      { -INT32_C(    69189900), -INT32_C(   331292608),  INT32_C(   492592544),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mullo_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mullo_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_mullo_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_mullo_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 5012251697601585881), -INT64_C(  308287235841070820) },
      UINT8_C(154),
      {  INT64_C( 2698439524592938388),  INT64_C( 4270563744594112206) },
      { -INT64_C( 6507458759045468804),  INT64_C( 8078338271933440964) },
      {  INT64_C( 5012251697601585881),  INT64_C( 5913429564128624568) } },
    { { -INT64_C( 1221810160544032654), -INT64_C( 3511507729965761296) },
      UINT8_C(203),
      {  INT64_C( 3998120136216699868),  INT64_C( 8335743013792972343) },
      { -INT64_C( 2700889159394618542), -INT64_C( 5960516505628144163) },
      { -INT64_C( 6698518447485661064),  INT64_C(  742027466533224571) } },
    { {  INT64_C( 3795658164190760801), -INT64_C( 7823000758962307359) },
      UINT8_C(177),
      {  INT64_C( 4063701154969691908), -INT64_C(  524749447695310504) },
      { -INT64_C( 7575008561253241827),  INT64_C( 5793997798346579097) },
      { -INT64_C(  712353094803055756), -INT64_C( 7823000758962307359) } },
    { {  INT64_C( 4284116092212197521),  INT64_C( 5926253553320558109) },
      UINT8_C(116),
      {  INT64_C( 1078017491261477171), -INT64_C( 3623521846670218054) },
      { -INT64_C( 1883371713334283390), -INT64_C( 7837833038605442435) },
      {  INT64_C( 4284116092212197521),  INT64_C( 5926253553320558109) } },
    { {  INT64_C( 2665819173114840766), -INT64_C( 8830997003190576752) },
      UINT8_C(242),
      {  INT64_C( 4794714160184837153),  INT64_C( 1190135067391971162) },
      { -INT64_C( 3704037576123255772), -INT64_C( 3469929303203955482) },
      {  INT64_C( 2665819173114840766), -INT64_C( 8733423008951934244) } },
    { { -INT64_C( 3676232027887357287), -INT64_C(  698313225441993290) },
      UINT8_C(139),
      { -INT64_C( 3292753005216071991),  INT64_C( 4505939439377200026) },
      { -INT64_C( 2789561894561627860), -INT64_C( 6795472413008361458) },
      {  INT64_C( 4168170210759857036),  INT64_C( 4875883178927759980) } },
    { {  INT64_C( 1443215089447972767),  INT64_C(  347212408579917959) },
      UINT8_C(204),
      {  INT64_C( 7569391955160373121), -INT64_C( 7940722581911293556) },
      { -INT64_C( 8014965841574821399), -INT64_C( 6162774589309429726) },
      {  INT64_C( 1443215089447972767),  INT64_C(  347212408579917959) } },
    { { -INT64_C( 1274884407158144764),  INT64_C( 6509926262228357092) },
      UINT8_C(232),
      { -INT64_C( 5285676344725628829),  INT64_C( 5574201574053401322) },
      {  INT64_C( 2146421033994829874),  INT64_C( 5299305088317034426) },
      { -INT64_C( 1274884407158144764),  INT64_C( 6509926262228357092) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_mullo_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_mullo_epi64");
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
    easysimd__m128i r = easysimd_mm_mask_mullo_epi64(src, k, a, b);

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
test_easysimd_mm_maskz_mullo_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[2];
    const int64_t b[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(247),
      {  INT64_C( 1955809977294461516), -INT64_C( 4602105515836606233) },
      { -INT64_C( 3734601322628869756), -INT64_C( 9139940866071669280) },
      {  INT64_C( 3403554178002754352), -INT64_C(  228457257205367520) } },
    { UINT8_C(130),
      { -INT64_C(  122884140846390795), -INT64_C( 7661985247517875938) },
      { -INT64_C( 1219838769285829078),  INT64_C( 7054790321470981116) },
      {  INT64_C(                   0), -INT64_C( 4513833018810666104) } },
    { UINT8_C(  7),
      { -INT64_C( 1025540811991956167), -INT64_C( 8914723085010736139) },
      { -INT64_C( 3512650976251119103), -INT64_C( 4224205423378782797) },
      {  INT64_C( 2686132250603548473), -INT64_C( 5761075171044956849) } },
    { UINT8_C( 96),
      {  INT64_C( 1904849123458247662),  INT64_C( 5828278965284932951) },
      {  INT64_C(  870790563494635475), -INT64_C( 4836481383300305179) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(200),
      {  INT64_C( 3678334399075006034), -INT64_C(  591843122006593855) },
      { -INT64_C( 5853154214311679179),  INT64_C( 8802911267246130692) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(112),
      {  INT64_C( 3095178267302637345), -INT64_C( 2314864825668539823) },
      { -INT64_C( 3764782214699499695),  INT64_C( 7923360394372129355) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(157),
      {  INT64_C( 9197480278889177098), -INT64_C( 1278719726877192194) },
      { -INT64_C( 2284100450217886413), -INT64_C( 3369367863538429390) },
      { -INT64_C( 8509267355006483458),  INT64_C(                   0) } },
    { UINT8_C( 58),
      {  INT64_C( 1736224281996502039),  INT64_C( 5172441662676666498) },
      { -INT64_C( 1722374970958034521),  INT64_C(  929010315275778065) },
      {  INT64_C(                   0),  INT64_C( 1685910991925280930) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_epi64(test_vec[i].b);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_mullo_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_mullo_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_mullo_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_mullo_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[16];
    const uint16_t k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { {  INT16_C( 25725),  INT16_C(  1722),  INT16_C( 19664), -INT16_C( 27872), -INT16_C(  9238), -INT16_C( 11282), -INT16_C( 28990),  INT16_C(  8870),
        -INT16_C(  4411),  INT16_C( 25830),  INT16_C( 27143),  INT16_C( 25752), -INT16_C(  1020),  INT16_C(  6497),  INT16_C( 16804),  INT16_C(  8481) },
      UINT16_C(56485),
      {  INT16_C( 29991),  INT16_C( 18472),  INT16_C(  4872), -INT16_C(  2525), -INT16_C(  6682), -INT16_C( 29563),  INT16_C( 18951), -INT16_C(  4741),
        -INT16_C( 32082),  INT16_C( 18007),  INT16_C( 23526),  INT16_C( 18243), -INT16_C(  6283), -INT16_C( 27000),  INT16_C( 11785),  INT16_C( 12402) },
      { -INT16_C( 25693), -INT16_C( 21384), -INT16_C( 25682), -INT16_C( 27486),  INT16_C( 10113), -INT16_C( 30687), -INT16_C( 25487),  INT16_C(  8054),
        -INT16_C( 13026),  INT16_C(  1126), -INT16_C( 22231), -INT16_C( 25012), -INT16_C( 11120), -INT16_C( 26316), -INT16_C( 22782), -INT16_C( 22838) },
      {  INT16_C( 13525),  INT16_C(  1722), -INT16_C( 14480), -INT16_C( 27872), -INT16_C(  9238), -INT16_C( 15067), -INT16_C( 28990),  INT16_C( 23474),
        -INT16_C(  4411),  INT16_C( 25830), -INT16_C( 29226), -INT16_C( 32284),  INT16_C(  5584),  INT16_C(  6497),  INT16_C( 15122),  INT16_C(  9716) } },
    { {  INT16_C( 16962), -INT16_C(  4014), -INT16_C(  2850),  INT16_C( 24452), -INT16_C( 23268), -INT16_C( 29209),  INT16_C( 23873),  INT16_C( 24493),
         INT16_C(  4907),  INT16_C( 21604), -INT16_C( 20292),  INT16_C( 19698),  INT16_C(  9860), -INT16_C( 30746), -INT16_C( 20275),  INT16_C(  3885) },
      UINT16_C(32754),
      { -INT16_C( 12033), -INT16_C( 31629), -INT16_C( 28881),  INT16_C(  5929),  INT16_C( 27421), -INT16_C( 13708), -INT16_C( 24630),  INT16_C( 11997),
        -INT16_C( 26125), -INT16_C(  6690),  INT16_C( 25573), -INT16_C( 13556), -INT16_C(  9750),  INT16_C(  6011),  INT16_C( 28393), -INT16_C(  5994) },
      {  INT16_C(  2366),  INT16_C( 28268), -INT16_C( 26983), -INT16_C( 18811), -INT16_C(  1791), -INT16_C( 13440),  INT16_C( 23961), -INT16_C( 29446),
        -INT16_C(  9994), -INT16_C(  9358),  INT16_C( 32315),  INT16_C(  9639),  INT16_C(  8791),  INT16_C( 16444), -INT16_C( 11632), -INT16_C( 12503) },
      {  INT16_C( 16962),  INT16_C( 19076), -INT16_C(  2850),  INT16_C( 24452), -INT16_C( 24547),  INT16_C( 13824), -INT16_C(  7750), -INT16_C( 24622),
        -INT16_C(  2174),  INT16_C( 18140), -INT16_C( 17465),  INT16_C( 12500),  INT16_C(  8838),  INT16_C( 16596), -INT16_C( 31472),  INT16_C(  3885) } },
    { { -INT16_C( 27172),  INT16_C( 30013), -INT16_C( 15829),  INT16_C( 11307), -INT16_C( 21573),  INT16_C( 21752), -INT16_C(  3576), -INT16_C(   287),
         INT16_C( 21450),  INT16_C(  1753), -INT16_C( 32559),  INT16_C( 10283),  INT16_C( 26787),  INT16_C( 13161), -INT16_C( 28102),  INT16_C(  5634) },
      UINT16_C(16167),
      {  INT16_C( 21387), -INT16_C( 18943), -INT16_C( 17025),  INT16_C( 30561),  INT16_C( 26897), -INT16_C(  3479),  INT16_C( 13415),  INT16_C( 16709),
         INT16_C(  5690),  INT16_C( 26049),  INT16_C( 25663), -INT16_C( 22323),  INT16_C(  2200), -INT16_C( 26054),  INT16_C( 24862), -INT16_C( 21798) },
      { -INT16_C(  9292),  INT16_C( 13408), -INT16_C( 15720), -INT16_C( 21845),  INT16_C(  5419), -INT16_C( 27748), -INT16_C(  7607), -INT16_C( 31788),
        -INT16_C( 27144),  INT16_C( 14312), -INT16_C( 18694), -INT16_C( 27937),  INT16_C(  6590), -INT16_C(  9172),  INT16_C(  1659),  INT16_C( 12166) },
      { -INT16_C( 22852),  INT16_C( 29792), -INT16_C( 16024),  INT16_C( 11307), -INT16_C( 21573),  INT16_C(   764), -INT16_C(  3576), -INT16_C(   287),
         INT16_C( 18992), -INT16_C( 21016), -INT16_C( 20602), -INT16_C(  2925),  INT16_C( 14544),  INT16_C( 23032), -INT16_C( 28102),  INT16_C(  5634) } },
    { { -INT16_C(  6174),  INT16_C( 31331),  INT16_C(  4009), -INT16_C( 11228), -INT16_C( 16092),  INT16_C( 28007),  INT16_C( 15267), -INT16_C( 25616),
        -INT16_C( 10031), -INT16_C( 13357), -INT16_C( 19826),  INT16_C( 19549), -INT16_C( 30260),  INT16_C( 18217), -INT16_C( 20592),  INT16_C( 29302) },
      UINT16_C(55958),
      {  INT16_C( 16364),  INT16_C(  4585),  INT16_C(  3348),  INT16_C( 31698),  INT16_C( 30074),  INT16_C( 27319), -INT16_C( 30704), -INT16_C(  7358),
        -INT16_C( 11949), -INT16_C( 20330),  INT16_C( 25117),  INT16_C( 17977), -INT16_C( 13911),  INT16_C(  8182), -INT16_C( 29637),  INT16_C( 10489) },
      { -INT16_C(  7476), -INT16_C(  8135),  INT16_C(  3055),  INT16_C( 26971),  INT16_C(  4736), -INT16_C( 28461),  INT16_C(  5786), -INT16_C(  4748),
         INT16_C(  2791),  INT16_C(  1181), -INT16_C( 10388),  INT16_C(  5451),  INT16_C( 16800), -INT16_C(  9164),  INT16_C( 11981), -INT16_C( 26364) },
      { -INT16_C(  6174), -INT16_C(  8991),  INT16_C(  4524), -INT16_C( 11228),  INT16_C( 20736),  INT16_C( 28007),  INT16_C( 15267),  INT16_C(  5096),
        -INT16_C( 10031), -INT16_C( 23554), -INT16_C( 19826),  INT16_C( 16307), -INT16_C(  3424),  INT16_C( 18217), -INT16_C(  6849),  INT16_C( 29924) } },
    { {  INT16_C( 15632),  INT16_C(   121), -INT16_C( 10936), -INT16_C( 14231),  INT16_C( 15847), -INT16_C( 32168), -INT16_C( 13229),  INT16_C( 14959),
         INT16_C(  3542),  INT16_C( 16958), -INT16_C( 30236), -INT16_C( 31657), -INT16_C( 29494), -INT16_C( 26528),  INT16_C( 25786), -INT16_C( 13775) },
      UINT16_C(43937),
      { -INT16_C(  5686),  INT16_C( 13440),  INT16_C( 26545),  INT16_C(  2673), -INT16_C( 15127),  INT16_C( 22998), -INT16_C( 20994),  INT16_C( 15462),
         INT16_C( 19183),  INT16_C( 18374), -INT16_C( 28466),  INT16_C( 12243), -INT16_C( 29400),  INT16_C( 23187),  INT16_C( 13655),  INT16_C(  8709) },
      { -INT16_C( 31458), -INT16_C( 12202), -INT16_C( 14356), -INT16_C( 10534), -INT16_C( 20341), -INT16_C( 30417), -INT16_C( 27299),  INT16_C( 19909),
        -INT16_C( 29729), -INT16_C( 21100),  INT16_C( 26396),  INT16_C( 17628),  INT16_C( 28916),  INT16_C( 19358), -INT16_C( 23643), -INT16_C( 15507) },
      {  INT16_C( 22444),  INT16_C(   121), -INT16_C( 10936), -INT16_C( 14231),  INT16_C( 15847),  INT16_C(  1098), -INT16_C( 13229),  INT16_C( 10366),
         INT16_C(  2865),  INT16_C( 19576), -INT16_C( 30236),  INT16_C(  9556), -INT16_C( 29494), -INT16_C(  2118),  INT16_C( 25786),  INT16_C( 19233) } },
    { { -INT16_C( 15576),  INT16_C(  5523),  INT16_C( 28042),  INT16_C(  5611),  INT16_C(  6686),  INT16_C( 31646),  INT16_C( 25775), -INT16_C( 28984),
         INT16_C( 23791),  INT16_C(  2875),  INT16_C(  6339), -INT16_C( 18608), -INT16_C(  4472),  INT16_C( 11523),  INT16_C( 28818), -INT16_C( 17680) },
      UINT16_C(33844),
      { -INT16_C( 16689), -INT16_C( 17679),  INT16_C(  4052),  INT16_C( 29396), -INT16_C( 31861),  INT16_C( 21462), -INT16_C( 14831),  INT16_C( 19888),
         INT16_C( 29649),  INT16_C(  8549), -INT16_C(  4821),  INT16_C( 11792), -INT16_C( 24038),  INT16_C(  2718), -INT16_C( 11684),  INT16_C( 11406) },
      { -INT16_C( 32623),  INT16_C( 26086), -INT16_C( 17521),  INT16_C(  6871), -INT16_C( 20930),  INT16_C( 20590),  INT16_C(  7796),  INT16_C( 17821),
         INT16_C(   657), -INT16_C( 17305),  INT16_C( 30703),  INT16_C(  2538), -INT16_C( 30439),  INT16_C( 29971), -INT16_C( 23973), -INT16_C(  4959) },
      { -INT16_C( 15576),  INT16_C(  5523), -INT16_C( 19604),  INT16_C(  5611),  INT16_C( 21930), -INT16_C(  6668),  INT16_C( 25775), -INT16_C( 28984),
         INT16_C( 23791),  INT16_C(  2875),  INT16_C( 26661), -INT16_C( 18608), -INT16_C(  4472),  INT16_C( 11523),  INT16_C( 28818), -INT16_C(  4786) } },
    { { -INT16_C( 30686), -INT16_C( 20143),  INT16_C( 10563), -INT16_C( 32308),  INT16_C( 15063),  INT16_C( 19409),  INT16_C( 28248), -INT16_C(  5744),
        -INT16_C(  2192),  INT16_C( 24486), -INT16_C( 28562), -INT16_C( 30872),  INT16_C( 31769),  INT16_C( 30205), -INT16_C( 25058),  INT16_C( 16481) },
      UINT16_C(45862),
      {  INT16_C( 27121), -INT16_C( 16932), -INT16_C( 19477), -INT16_C( 17161),  INT16_C( 20478), -INT16_C( 29141), -INT16_C( 25799), -INT16_C(  8314),
        -INT16_C(  2821),  INT16_C( 25455), -INT16_C( 30340),  INT16_C( 31199), -INT16_C(   514),  INT16_C( 24343),  INT16_C( 15933),  INT16_C( 12050) },
      { -INT16_C(  4441), -INT16_C( 27924), -INT16_C(  7007), -INT16_C( 24753),  INT16_C( 31283),  INT16_C( 27694), -INT16_C( 19435),  INT16_C(  4171),
        -INT16_C( 17496),  INT16_C(  9332),  INT16_C( 21316),  INT16_C( 17053), -INT16_C( 19119), -INT16_C( 29023), -INT16_C( 19213), -INT16_C( 25923) },
      { -INT16_C( 30686),  INT16_C( 32464),  INT16_C( 29387), -INT16_C( 32308),  INT16_C( 15063), -INT16_C( 20550),  INT16_C( 28248), -INT16_C(  5744),
         INT16_C(  7608), -INT16_C( 21940), -INT16_C( 28562), -INT16_C( 30872), -INT16_C(  3234), -INT16_C( 28809), -INT16_C( 25058), -INT16_C( 27574) } },
    { { -INT16_C( 21854),  INT16_C( 17453),  INT16_C( 31886), -INT16_C( 15901),  INT16_C(  4598),  INT16_C(  2862),  INT16_C( 31173),  INT16_C( 28188),
        -INT16_C( 28620),  INT16_C( 30866),  INT16_C( 12515),  INT16_C( 13498),  INT16_C( 23781), -INT16_C( 10045), -INT16_C( 32752), -INT16_C( 19854) },
      UINT16_C(40746),
      { -INT16_C( 18186), -INT16_C(  9701),  INT16_C(  4474), -INT16_C( 22293), -INT16_C( 20195),  INT16_C( 14625),  INT16_C( 22047), -INT16_C( 20023),
        -INT16_C( 21298), -INT16_C( 30239), -INT16_C( 14623), -INT16_C( 23323), -INT16_C(  2658),  INT16_C(  4388),  INT16_C( 20391), -INT16_C( 24912) },
      { -INT16_C( 13305), -INT16_C( 32392),  INT16_C( 25565), -INT16_C(  1495),  INT16_C( 19220),  INT16_C( 13107), -INT16_C(   863),  INT16_C( 28645),
        -INT16_C( 14679), -INT16_C( 29960), -INT16_C(  8819),  INT16_C( 11054),  INT16_C( 21202),  INT16_C( 31292), -INT16_C(  4703), -INT16_C( 22248) },
      { -INT16_C( 21854), -INT16_C( 10328),  INT16_C( 31886), -INT16_C( 29789),  INT16_C(  4598), -INT16_C(  2925),  INT16_C( 31173),  INT16_C( 28188),
         INT16_C( 26622), -INT16_C(  9224), -INT16_C( 14611),  INT16_C(  6182),  INT16_C(  6044), -INT16_C( 10045), -INT16_C( 32752),  INT16_C(  4224) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mullo_epi16(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mullo_epi16");
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
    easysimd__m256i r = easysimd_mm256_mask_mullo_epi16(src, k, a, b);

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
test_easysimd_mm256_maskz_mullo_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    const int16_t a[16];
    const int16_t b[16];
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C(39226),
      { -INT16_C( 16962), -INT16_C( 10601),  INT16_C( 30182), -INT16_C( 17824), -INT16_C( 24395),  INT16_C(  2498),  INT16_C( 26182),  INT16_C(  8669),
         INT16_C(  8629), -INT16_C( 14693),  INT16_C( 27828),  INT16_C(  8120),  INT16_C(  1037), -INT16_C( 19417),  INT16_C( 25308), -INT16_C( 26035) },
      { -INT16_C(  6881),  INT16_C(  1392), -INT16_C( 12198),  INT16_C(  4287), -INT16_C( 32144), -INT16_C( 18919), -INT16_C(  2328), -INT16_C( 25128),
         INT16_C( 29463), -INT16_C( 13469),  INT16_C(  7135), -INT16_C(  4885),  INT16_C(  4640), -INT16_C(   864), -INT16_C(  4492), -INT16_C( 27498) },
      {  INT16_C(     0), -INT16_C( 10992),  INT16_C(     0),  INT16_C(  3488),  INT16_C( 14640), -INT16_C(  8206),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 22083),  INT16_C(     0),  INT16_C(     0), -INT16_C( 16920),  INT16_C( 27552),  INT16_C(     0),  INT16_C(     0), -INT16_C(  4834) } },
    { UINT16_C( 1747),
      {  INT16_C( 11673),  INT16_C( 22999),  INT16_C( 18237),  INT16_C( 22235), -INT16_C( 15362), -INT16_C( 10676),  INT16_C( 25696), -INT16_C( 15543),
         INT16_C( 10287),  INT16_C(  6878), -INT16_C(   492), -INT16_C( 19411), -INT16_C( 24070), -INT16_C( 28510),  INT16_C( 30005), -INT16_C( 12393) },
      {  INT16_C( 28323), -INT16_C(  8152),  INT16_C(   949), -INT16_C( 19657), -INT16_C( 31802),  INT16_C(  9865), -INT16_C( 11545),  INT16_C(  6121),
        -INT16_C( 14342),  INT16_C(  3633),  INT16_C( 24262), -INT16_C( 16189),  INT16_C( 25856),  INT16_C( 13649), -INT16_C(  5925),  INT16_C( 32260) },
      { -INT16_C( 14741),  INT16_C( 10648),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28556),  INT16_C(     0),  INT16_C( 21152),  INT16_C( 19569),
         INT16_C(     0),  INT16_C( 18558), -INT16_C(  9352),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(11350),
      {  INT16_C(  2910), -INT16_C( 27345), -INT16_C(  2625),  INT16_C( 18457),  INT16_C(    27),  INT16_C(  1051),  INT16_C(  5399),  INT16_C( 18892),
        -INT16_C( 28124), -INT16_C(  6233), -INT16_C( 22702), -INT16_C( 23732),  INT16_C( 10205), -INT16_C(  7797), -INT16_C(  7771),  INT16_C(  1038) },
      {  INT16_C( 15853), -INT16_C( 21351), -INT16_C( 19917),  INT16_C( 20212),  INT16_C(  4019), -INT16_C( 13741),  INT16_C(  7973),  INT16_C( 18707),
        -INT16_C( 17487),  INT16_C(   816),  INT16_C( 31842),  INT16_C( 16295),  INT16_C( 12964),  INT16_C( 18721),  INT16_C( 12052),  INT16_C(   333) },
      {  INT16_C(     0), -INT16_C( 17129), -INT16_C( 15603),  INT16_C(     0), -INT16_C( 22559),  INT16_C(     0), -INT16_C( 10925),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 15004),  INT16_C( 14996),  INT16_C(     0), -INT16_C( 18965),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(59244),
      { -INT16_C( 24659), -INT16_C( 24167),  INT16_C( 19694),  INT16_C( 16817), -INT16_C( 10729),  INT16_C( 10848),  INT16_C(  4383),  INT16_C( 20453),
         INT16_C( 18452), -INT16_C( 17461),  INT16_C( 28551), -INT16_C( 22290),  INT16_C(   697),  INT16_C(  1751),  INT16_C( 17411), -INT16_C( 20243) },
      { -INT16_C( 30749), -INT16_C( 11951),  INT16_C(   723), -INT16_C(  5614),  INT16_C( 29400), -INT16_C(  2283), -INT16_C(  1405), -INT16_C( 26554),
         INT16_C(  4674), -INT16_C( 13741),  INT16_C( 16769),  INT16_C( 14962),  INT16_C( 19011),  INT16_C( 17985),  INT16_C( 11918),  INT16_C( 29174) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 17450),  INT16_C( 26738),  INT16_C(     0),  INT16_C(  6624),  INT16_C(  2269),  INT16_C(     0),
        -INT16_C(   728),  INT16_C(  4305),  INT16_C( 31239),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31081),  INT16_C( 17322), -INT16_C( 24386) } },
    { UINT16_C(18613),
      { -INT16_C( 30397),  INT16_C( 21834),  INT16_C(  9075), -INT16_C( 30520),  INT16_C( 19226),  INT16_C( 24963), -INT16_C( 14877),  INT16_C( 14195),
        -INT16_C(  2929),  INT16_C(   632), -INT16_C( 17361),  INT16_C( 28748), -INT16_C(  9726), -INT16_C(  1634),  INT16_C( 21579), -INT16_C( 29119) },
      { -INT16_C( 29731),  INT16_C( 20708), -INT16_C( 21330), -INT16_C( 13863),  INT16_C( 23799), -INT16_C(  9430), -INT16_C( 25311), -INT16_C( 20206),
        -INT16_C( 30063), -INT16_C( 16205), -INT16_C(   186),  INT16_C( 18736), -INT16_C( 12327),  INT16_C(  9282), -INT16_C( 31965),  INT16_C(   179) },
      { -INT16_C(  8233),  INT16_C(     0),  INT16_C( 23594),  INT16_C(     0), -INT16_C( 12778),  INT16_C(  4222),  INT16_C(     0),  INT16_C( 26902),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 17856),  INT16_C(     0),  INT16_C(     0), -INT16_C(  6335),  INT16_C(     0) } },
    { UINT16_C(38670),
      { -INT16_C( 17072),  INT16_C( 10563),  INT16_C( 14982), -INT16_C( 20347), -INT16_C( 22763),  INT16_C( 10061), -INT16_C(  8616),  INT16_C(  2994),
        -INT16_C(  1889), -INT16_C( 12534), -INT16_C(  7359), -INT16_C( 31842), -INT16_C( 16121), -INT16_C( 17914),  INT16_C(  5569),  INT16_C(  4689) },
      { -INT16_C( 27438),  INT16_C( 22587), -INT16_C( 15921), -INT16_C(  7160),  INT16_C( 21864), -INT16_C( 16372), -INT16_C( 16845), -INT16_C( 11573),
        -INT16_C( 10826), -INT16_C(  1886),  INT16_C( 16568), -INT16_C( 16517), -INT16_C( 32254), -INT16_C( 15494), -INT16_C( 13417),  INT16_C( 27093) },
      {  INT16_C(     0), -INT16_C( 30095),  INT16_C( 22618), -INT16_C(  2008),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(  3082), -INT16_C( 19372), -INT16_C( 26952),  INT16_C(     0),  INT16_C(  4110),  INT16_C(     0),  INT16_C(     0),  INT16_C( 30309) } },
    { UINT16_C( 4448),
      {  INT16_C( 12225), -INT16_C( 13870),  INT16_C( 14867),  INT16_C(  7966),  INT16_C( 20986), -INT16_C( 14883), -INT16_C( 27612), -INT16_C( 14694),
         INT16_C( 21132),  INT16_C(  1798),  INT16_C(  2065), -INT16_C( 29815),  INT16_C(  8396), -INT16_C( 24233), -INT16_C( 18551),  INT16_C( 19122) },
      { -INT16_C( 31514), -INT16_C(  1773),  INT16_C( 12734), -INT16_C( 18407), -INT16_C(  2429), -INT16_C( 22659),  INT16_C(  6026),  INT16_C(  5741),
         INT16_C( 29545),  INT16_C( 31518), -INT16_C( 22660),  INT16_C( 18438),  INT16_C( 24008),  INT16_C( 20969), -INT16_C( 25580), -INT16_C(  1380) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 14359),  INT16_C(  5992),  INT16_C(     0),
        -INT16_C( 16532),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 17568),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(44832),
      { -INT16_C(  8204),  INT16_C(  3553),  INT16_C( 25751),  INT16_C(  5379), -INT16_C( 29173),  INT16_C( 30764), -INT16_C( 26972), -INT16_C( 15637),
         INT16_C( 26385),  INT16_C(  5994),  INT16_C( 12975), -INT16_C( 26251), -INT16_C( 30333),  INT16_C(  7989),  INT16_C( 21892),  INT16_C( 30927) },
      { -INT16_C( 20428), -INT16_C( 13179), -INT16_C( 30700),  INT16_C(  8161),  INT16_C(  3350), -INT16_C( 17513), -INT16_C( 32093), -INT16_C( 19331),
        -INT16_C(  6166), -INT16_C( 26164),  INT16_C( 16665), -INT16_C( 25294),  INT16_C( 26570),  INT16_C( 20156), -INT16_C( 29763), -INT16_C(  3642) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1524),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 29558),  INT16_C(   632),  INT16_C( 25111), -INT16_C( 17958),  INT16_C(     0),  INT16_C(  4332),  INT16_C(     0),  INT16_C( 20250) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi16(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mullo_epi16(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mullo_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_maskz_mullo_epi16(k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_mullo_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1775595335),  INT32_C(   143742195),  INT32_C(  1077658783),  INT32_C(  1789862081),  INT32_C(  1962191079),  INT32_C(  1486824069),  INT32_C(   382917748), -INT32_C(  2101351735) },
      UINT8_C(122),
      {  INT32_C(  1030559977),  INT32_C(  1859941801), -INT32_C(   785441615),  INT32_C(  1958255052), -INT32_C(  1342624627), -INT32_C(  1474080308),  INT32_C(   225524003), -INT32_C(   494406664) },
      { -INT32_C(  1256196853),  INT32_C(   455343210), -INT32_C(   454209000),  INT32_C(  2019075563), -INT32_C(  1641590062), -INT32_C(   951694685),  INT32_C(  2094315396), -INT32_C(  1252107094) },
      { -INT32_C(  1775595335), -INT32_C(   734653446),  INT32_C(  1077658783), -INT32_C(  1632261564), -INT32_C(    98126422),  INT32_C(   904531172),  INT32_C(   590707468), -INT32_C(  2101351735) } },
    { { -INT32_C(  1150648751), -INT32_C(  1831367302), -INT32_C(   898120481),  INT32_C(   994234217), -INT32_C(   975607519),  INT32_C(   948707252), -INT32_C(  2135662378),  INT32_C(   221647804) },
      UINT8_C(145),
      {  INT32_C(   772524448),  INT32_C(  1678613920), -INT32_C(   456271852), -INT32_C(  2080044774),  INT32_C(    37276386), -INT32_C(  1210552234),  INT32_C(   930306340),  INT32_C(   801669519) },
      { -INT32_C(   362949814), -INT32_C(  2058458255),  INT32_C(  1567169347),  INT32_C(   132214564), -INT32_C(  1878451911), -INT32_C(  1371020919), -INT32_C(   907691462), -INT32_C(  2047234501) },
      { -INT32_C(  2051594176), -INT32_C(  1831367302), -INT32_C(   898120481),  INT32_C(   994234217),  INT32_C(  1281834578),  INT32_C(   948707252), -INT32_C(  2135662378), -INT32_C(  1705243147) } },
    { { -INT32_C(   227584383),  INT32_C(    75021761), -INT32_C(    43851303), -INT32_C(  1979432112), -INT32_C(   434500259),  INT32_C(   697590255),  INT32_C(  1475574299), -INT32_C(  1445139416) },
      UINT8_C( 66),
      {  INT32_C(   151297100), -INT32_C(   169736172), -INT32_C(  1371086742), -INT32_C(   234106652),  INT32_C(  1273098730),  INT32_C(     6687366), -INT32_C(   383205891), -INT32_C(   433270374) },
      { -INT32_C(  2115030931), -INT32_C(  1552428488), -INT32_C(  1773027918),  INT32_C(  2005425293), -INT32_C(   742233779),  INT32_C(  1909663860), -INT32_C(  2141455386),  INT32_C(   979797965) },
      { -INT32_C(   227584383), -INT32_C(  1319228320), -INT32_C(    43851303), -INT32_C(  1979432112), -INT32_C(   434500259),  INT32_C(   697590255),  INT32_C(  1976614990), -INT32_C(  1445139416) } },
    { { -INT32_C(   272869961), -INT32_C(   644730073),  INT32_C(  2104550384), -INT32_C(  1913325505), -INT32_C(   698304926), -INT32_C(  1001966370), -INT32_C(    62610897), -INT32_C(   533222871) },
      UINT8_C(  0),
      {  INT32_C(   640143603),  INT32_C(  1175847266),  INT32_C(  1770361713),  INT32_C(  1036718727), -INT32_C(  1491361421), -INT32_C(  1948851991), -INT32_C(   810167516), -INT32_C(    36727542) },
      { -INT32_C(   937167259),  INT32_C(  1745762807),  INT32_C(  1406309324),  INT32_C(   428907942),  INT32_C(   683715391), -INT32_C(  1330407797),  INT32_C(  1937729641),  INT32_C(  1668304638) },
      { -INT32_C(   272869961), -INT32_C(   644730073),  INT32_C(  2104550384), -INT32_C(  1913325505), -INT32_C(   698304926), -INT32_C(  1001966370), -INT32_C(    62610897), -INT32_C(   533222871) } },
    { {  INT32_C(  1009488708), -INT32_C(  1734067764),  INT32_C(  1944811213),  INT32_C(  1401715476), -INT32_C(  1300542425),  INT32_C(  1281502946), -INT32_C(  1782586985),  INT32_C(  1962422319) },
      UINT8_C(195),
      {  INT32_C(  1569763364), -INT32_C(   886429612), -INT32_C(  1881170668),  INT32_C(  1974874665), -INT32_C(   614962771), -INT32_C(  1384995637), -INT32_C(  1814296733),  INT32_C(   609701888) },
      {  INT32_C(  1434576640),  INT32_C(   589343759),  INT32_C(  1941176137), -INT32_C(   555194063), -INT32_C(  1632026414),  INT32_C(  1212886244),  INT32_C(   870000435),  INT32_C(  2018980472) },
      {  INT32_C(  1158970368), -INT32_C(  1055863572),  INT32_C(  1944811213),  INT32_C(  1401715476), -INT32_C(  1300542425),  INT32_C(  1281502946),  INT32_C(    78024121),  INT32_C(  1183678464) } },
    { {  INT32_C(   701356313), -INT32_C(   833819259),  INT32_C(   490864620),  INT32_C(  1006316137),  INT32_C(  1339667818),  INT32_C(   345449697), -INT32_C(  1001885108), -INT32_C(  1103323227) },
      UINT8_C(120),
      { -INT32_C(   134355190),  INT32_C(   870632500),  INT32_C(   932970765), -INT32_C(  1297950467), -INT32_C(   694947407), -INT32_C(    98392184), -INT32_C(  1885346065),  INT32_C(   738680354) },
      {  INT32_C(  2032403781), -INT32_C(   559085359),  INT32_C(   102123785), -INT32_C(   742868959),  INT32_C(   833178537), -INT32_C(   500446221), -INT32_C(   747517007),  INT32_C(  1845524777) },
      {  INT32_C(   701356313), -INT32_C(   833819259),  INT32_C(   490864620), -INT32_C(  1882468195),  INT32_C(  1850894553),  INT32_C(   496360984),  INT32_C(   103624767), -INT32_C(  1103323227) } },
    { {  INT32_C(  1340613758),  INT32_C(   892179756),  INT32_C(     3949534), -INT32_C(  1529613061),  INT32_C(   869628992), -INT32_C(   132841145), -INT32_C(   171210804),  INT32_C(  2120469248) },
      UINT8_C(239),
      { -INT32_C(   535048885),  INT32_C(  1052725754), -INT32_C(  2126921843),  INT32_C(   247586450), -INT32_C(  1269435213), -INT32_C(  1853862646), -INT32_C(   460229352),  INT32_C(   617811929) },
      { -INT32_C(   704253988), -INT32_C(   854277056),  INT32_C(   357453443), -INT32_C(   551350228),  INT32_C(   244611076), -INT32_C(   576777019),  INT32_C(  1673605258),  INT32_C(   461870399) },
      {  INT32_C(  1090810228), -INT32_C(  1004542336), -INT32_C(  1495270617),  INT32_C(  1127309592),  INT32_C(   869628992),  INT32_C(  2029392050),  INT32_C(   192257776),  INT32_C(   647574375) } },
    { { -INT32_C(   990737276), -INT32_C(   745470384), -INT32_C(  2132222124), -INT32_C(   195032081),  INT32_C(  1208153219), -INT32_C(  1842961912),  INT32_C(   318105554),  INT32_C(     2981244) },
      UINT8_C(  9),
      {  INT32_C(   643482655),  INT32_C(   897199445),  INT32_C(   556071702),  INT32_C(  1336219739), -INT32_C(  1118311141), -INT32_C(    74454509),  INT32_C(  1551344095), -INT32_C(   295274289) },
      { -INT32_C(  1844133828),  INT32_C(    63410157), -INT32_C(   450499702),  INT32_C(   523553284), -INT32_C(   874738761),  INT32_C(  1439066997), -INT32_C(   592364275), -INT32_C(   221636683) },
      {  INT32_C(   281622340), -INT32_C(   745470384), -INT32_C(  2132222124),  INT32_C(  2068328300),  INT32_C(  1208153219), -INT32_C(  1842961912),  INT32_C(   318105554),  INT32_C(     2981244) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mullo_epi32(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mullo_epi32");
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
    easysimd__m256i r = easysimd_mm256_mask_mullo_epi32(src, k, a, b);

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
test_easysimd_mm256_maskz_mullo_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C( 59),
      { -INT32_C(   732971701), -INT32_C(  1393922402), -INT32_C(  2008046331),  INT32_C(   175244323),  INT32_C(   287574992), -INT32_C(  1512324802),  INT32_C(   157428349),  INT32_C(  1011111152) },
      { -INT32_C(  1341090799),  INT32_C(   140311042), -INT32_C(  1013929056),  INT32_C(  2144141999),  INT32_C(  1267790093),  INT32_C(   821128370), -INT32_C(  2043063402), -INT32_C(  1178436185) },
      { -INT32_C(  1754599941), -INT32_C(    89970372),  INT32_C(           0), -INT32_C(  1258284563),  INT32_C(  1660119440), -INT32_C(  1817148132),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 17),
      { -INT32_C(   854300206),  INT32_C(  1886198981), -INT32_C(  1373622100), -INT32_C(   272916483), -INT32_C(  1667168464),  INT32_C(  1261621752), -INT32_C(  1997293558),  INT32_C(  1301916795) },
      { -INT32_C(   635785963),  INT32_C(  1967818697), -INT32_C(  1255970121),  INT32_C(   983883530), -INT32_C(   556382746),  INT32_C(   556337175),  INT32_C(  1000938688), -INT32_C(   578206776) },
      {  INT32_C(   830707770),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1044617952),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(240),
      {  INT32_C(   733656995),  INT32_C(  1826762498),  INT32_C(   846632787), -INT32_C(  2129088453), -INT32_C(  1902578042),  INT32_C(  1011857695), -INT32_C(  1509651869), -INT32_C(  1214848493) },
      { -INT32_C(  1679667047), -INT32_C(   754400128), -INT32_C(  1761247652), -INT32_C(  1256710865),  INT32_C(   843362323), -INT32_C(   848391318),  INT32_C(   829649693), -INT32_C(   286783147) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1718512626),  INT32_C(   513897430), -INT32_C(   165206473), -INT32_C(   834421937) } },
    { UINT8_C( 90),
      { -INT32_C(  1898280246),  INT32_C(   283815314), -INT32_C(   817921358),  INT32_C(  1273165210), -INT32_C(   860548039), -INT32_C(   152468861), -INT32_C(    28632331), -INT32_C(   866633470) },
      {  INT32_C(  1432040131), -INT32_C(  1855568417),  INT32_C(  1650501063), -INT32_C(   743619942), -INT32_C(   627088809), -INT32_C(   640644636), -INT32_C(  1495852125),  INT32_C(   410136660) },
      {  INT32_C(           0), -INT32_C(   681472978),  INT32_C(           0),  INT32_C(   283865764), -INT32_C(    55660193),  INT32_C(           0),  INT32_C(  1466761471),  INT32_C(           0) } },
    { UINT8_C( 98),
      {  INT32_C(   323120589),  INT32_C(  2044384211),  INT32_C(  1980972084),  INT32_C(  1271785449),  INT32_C(   254781318), -INT32_C(  1833760649),  INT32_C(   283597280), -INT32_C(  1720516661) },
      {  INT32_C(  1085060204), -INT32_C(  1128692088), -INT32_C(  1405956925),  INT32_C(   989331635),  INT32_C(   524887975),  INT32_C(   280099888),  INT32_C(   555784277),  INT32_C(    79336087) },
      {  INT32_C(           0),  INT32_C(  1318844952),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   137214544),  INT32_C(   372103008),  INT32_C(           0) } },
    { UINT8_C( 71),
      { -INT32_C(   321960858), -INT32_C(   911242243), -INT32_C(  1115857731),  INT32_C(  2070263636), -INT32_C(    39091200), -INT32_C(   833438923),  INT32_C(  1852142555), -INT32_C(  1816827603) },
      { -INT32_C(  1434418003), -INT32_C(   864801009), -INT32_C(   544542325), -INT32_C(  1470369880), -INT32_C(  1465579917), -INT32_C(  1653147454), -INT32_C(  1727275925), -INT32_C(   215171003) },
      { -INT32_C(   699019026),  INT32_C(   379627219), -INT32_C(  2008987745),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   828416137),  INT32_C(           0) } },
    { UINT8_C( 68),
      { -INT32_C(   598499924),  INT32_C(    40312849), -INT32_C(  1716893782), -INT32_C(  1475587166), -INT32_C(   261442312),  INT32_C(   123471915), -INT32_C(   733154029), -INT32_C(   837271775) },
      { -INT32_C(   290820899),  INT32_C(   921768332), -INT32_C(    87057576), -INT32_C(   408691730), -INT32_C(  1160311409),  INT32_C(   700527381),  INT32_C(  1258098216),  INT32_C(   706221389) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   496856976),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   322196728),  INT32_C(           0) } },
    { UINT8_C(129),
      { -INT32_C(   754116158), -INT32_C(  1523825910), -INT32_C(   309058030), -INT32_C(   679642167), -INT32_C(  2031339694),  INT32_C(   128849401),  INT32_C(   676657170), -INT32_C(   760643824) },
      { -INT32_C(  1566132328),  INT32_C(   222810874), -INT32_C(  1023747080), -INT32_C(  1449560234), -INT32_C(  1473280593), -INT32_C(  1380983397), -INT32_C(   422247466),  INT32_C(   448364418) },
      {  INT32_C(   620558640),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2102692320) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mullo_epi32(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mullo_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_mullo_epi32(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_mullo_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C(  520082817828986921),  INT64_C( 8023463689801559350), -INT64_C( 5879730096713353828),  INT64_C(  568953497153312829) },
      UINT8_C( 51),
      { -INT64_C( 9013222835968947095), -INT64_C( 5466057686821231242), -INT64_C( 5395308376931092765),  INT64_C( 6995759249387492237) },
      { -INT64_C( 4120511839159863623), -INT64_C( 7541432930849125595),  INT64_C( 4393994464028961960), -INT64_C( 8788717714465169234) },
      {  INT64_C( 2689609917733297633),  INT64_C( 2298818927741336590), -INT64_C( 5879730096713353828),  INT64_C(  568953497153312829) } },
    { {  INT64_C( 4755983850957774313),  INT64_C( 7564989370628198065), -INT64_C( 4695381871747565846),  INT64_C( 1214684540193600647) },
      UINT8_C(138),
      { -INT64_C( 7491761850105181520), -INT64_C( 2379940849915499312),  INT64_C( 6601341812062720365),  INT64_C( 8865779843682463762) },
      {  INT64_C( 7836220041842504157),  INT64_C( 4938886212002387780),  INT64_C( 4091507232336279812), -INT64_C( 5103492223627474459) },
      {  INT64_C( 4755983850957774313), -INT64_C( 8976035945448761536), -INT64_C( 4695381871747565846),  INT64_C( 7387090326243238426) } },
    { {  INT64_C( 1908013133866270066),  INT64_C( 7769080337392989622),  INT64_C(  227605636909050570),  INT64_C( 1332468045871112167) },
      UINT8_C(221),
      { -INT64_C( 8091615465401278370),  INT64_C( 4445442144810254741), -INT64_C( 6215143635946856135),  INT64_C( 6832060694360851219) },
      { -INT64_C( 1754978535042415493),  INT64_C( 4080682372747336785),  INT64_C( 8039997757757390112), -INT64_C( 5426500858522968750) },
      { -INT64_C(  296113273369459414),  INT64_C( 7769080337392989622), -INT64_C( 8780701213821141984), -INT64_C( 1051299569105018090) } },
    { {  INT64_C( 7993730034098113974),  INT64_C( 6576227677841374476),  INT64_C( 8375808150498268017), -INT64_C( 8673237089312596298) },
      UINT8_C(133),
      {  INT64_C( 4388775978221080684), -INT64_C( 8445987395790638109),  INT64_C( 8261890710604388121), -INT64_C( 6492560940007713048) },
      {  INT64_C( 8869392617079670618),  INT64_C( 8871418788190588508), -INT64_C( 3280566758671529677), -INT64_C( 2194916000268108629) },
      {  INT64_C( 1945927670549953016),  INT64_C( 6576227677841374476), -INT64_C(  871626594942600453), -INT64_C( 8673237089312596298) } },
    { { -INT64_C( 3336655547545220194),  INT64_C( 6070149850465600608),  INT64_C( 7640590685533088331), -INT64_C( 1083158238774478192) },
      UINT8_C(228),
      {  INT64_C( 6621465556779869874), -INT64_C( 8260497942834796913), -INT64_C( 3130082443469745877),  INT64_C( 5747462416747635973) },
      { -INT64_C( 2922634642243511235), -INT64_C( 1743661905388966819), -INT64_C( 3079488182100289067), -INT64_C( 4897814679315519206) },
      { -INT64_C( 3336655547545220194),  INT64_C( 6070149850465600608), -INT64_C( 1565806005239678009), -INT64_C( 1083158238774478192) } },
    { { -INT64_C( 1174804453781284205), -INT64_C( 5489608811927791059),  INT64_C( 5025002523474441751),  INT64_C( 8865904207941431502) },
      UINT8_C(222),
      { -INT64_C( 6752760578312218917), -INT64_C( 5141928788493323632), -INT64_C( 5149257831578105002), -INT64_C( 4753518748497815514) },
      {  INT64_C( 7573687714824485233),  INT64_C( 2599029770544388950),  INT64_C( 4911010597612614950),  INT64_C( 6044524363530553154) },
      { -INT64_C( 1174804453781284205),  INT64_C( 8144506131349122144),  INT64_C( 5537978013949361348), -INT64_C(  135877096167619636) } },
    { {  INT64_C( 3977214902867377647), -INT64_C( 1624213130843248510),  INT64_C(  142285324596827543), -INT64_C( 3640306127844579514) },
      UINT8_C(197),
      { -INT64_C( 1175195655192799929), -INT64_C( 3843932829843849106), -INT64_C( 4847972899265880845), -INT64_C( 6738116923351244624) },
      {  INT64_C( 7976578630800122876), -INT64_C(  882382270705539967), -INT64_C( 8821954816291550591), -INT64_C( 8143428888450749186) },
      { -INT64_C( 6715757094490626332), -INT64_C( 1624213130843248510),  INT64_C( 1816142787695154291), -INT64_C( 3640306127844579514) } },
    { { -INT64_C( 3581483970705895971),  INT64_C( 8167989241281104487), -INT64_C( 2582010790847922224), -INT64_C(  133082544098351354) },
         UINT8_MAX,
      { -INT64_C(  944600161142842247),  INT64_C( 7221333596884323296),  INT64_C( 7961552733658602652), -INT64_C( 5294387917025700268) },
      { -INT64_C( 3493359622533426326),  INT64_C( 6299248054343395170),  INT64_C( 2287348997356957001),  INT64_C( 5140063880074382945) },
      {  INT64_C(  421543917218956570),  INT64_C( 2466667764409445312), -INT64_C( 8404476125275859844),  INT64_C(   99793918213623252) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_mullo_epi64(src, k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_mullo_epi64");
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
    easysimd__m256i r = easysimd_mm256_mask_mullo_epi64(src, k, a, b);

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
test_easysimd_mm256_maskz_mullo_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    const int64_t a[4];
    const int64_t b[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C( 54),
      { -INT64_C( 2366293040992043937),  INT64_C(  747280349158370104), -INT64_C( 5061673564891707890), -INT64_C( 4549385071156811736) },
      { -INT64_C( 8937927500660404974),  INT64_C( 1843257626250587142), -INT64_C( 5474469991132415593), -INT64_C( 6156639940922097298) },
      {  INT64_C(                   0), -INT64_C( 3316705242088266928),  INT64_C( 2048222847983661122),  INT64_C(                   0) } },
    { UINT8_C( 69),
      { -INT64_C( 2116670149865979776), -INT64_C(  473433489796158650),  INT64_C(  233969210253228811),  INT64_C( 2094530746237193876) },
      { -INT64_C(  315717386307657211), -INT64_C( 7117022689686811773), -INT64_C( 8232107355703164676), -INT64_C( 6524190148755104837) },
      { -INT64_C( 8246578036898217344),  INT64_C(                   0), -INT64_C( 8855241344367042860),  INT64_C(                   0) } },
    { UINT8_C( 13),
      { -INT64_C( 8500615771280220116),  INT64_C( 5102219412247431824), -INT64_C( 6300591619510924249), -INT64_C( 8602177256400820442) },
      { -INT64_C( 5776407347403595442), -INT64_C( 3302739777031114146),  INT64_C( 6759961547409615323),  INT64_C( 6053274433100847388) },
      { -INT64_C(  295502983944865432),  INT64_C(                   0),  INT64_C( 8067846852678181981),  INT64_C( 8423023593207627304) } },
    { UINT8_C(238),
      {  INT64_C( 7674622459874484494), -INT64_C(  547213126474046147),  INT64_C( 8782173620608667534),  INT64_C( 7563184632376547706) },
      { -INT64_C(  264010609219468683), -INT64_C( 2091657779058480016),  INT64_C( 3267792211271886720),  INT64_C( 2423546468678097120) },
      {  INT64_C(                   0), -INT64_C( 1732763026446678352), -INT64_C(  548026270609676032), -INT64_C( 1869470255258060096) } },
    { UINT8_C( 45),
      { -INT64_C( 3411221200583782868), -INT64_C( 4043661130002552719), -INT64_C(  930935487672434003),  INT64_C( 2104520162553122524) },
      {  INT64_C( 8792430584213610703), -INT64_C( 2518428977067185755),  INT64_C(  760587368684226176), -INT64_C( 4647074966702838155) },
      { -INT64_C( 7167137094307875436),  INT64_C(                   0), -INT64_C(  619675231766584192), -INT64_C(  924435204480986484) } },
    { UINT8_C(101),
      {  INT64_C( 2412496705840673751),  INT64_C( 5981394710996041924),  INT64_C( 6547260976826260800), -INT64_C( 7257540058292797027) },
      {  INT64_C( 7068095733159737085), -INT64_C( 2969307474661447765),  INT64_C( 8360961122345241475),  INT64_C( 6465654333526921818) },
      {  INT64_C( 6918814636998514043),  INT64_C(                   0),  INT64_C( 5010538314892392384),  INT64_C(                   0) } },
    { UINT8_C( 84),
      { -INT64_C( 3764916499640070614),  INT64_C( 5665869188936251448), -INT64_C( 6059706168016285581), -INT64_C( 1521547348259961361) },
      {  INT64_C( 7124061468214941102), -INT64_C( 1781135766810771976), -INT64_C(  687543283379573390), -INT64_C( 8499868251083998112) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 6280748171903289910),  INT64_C(                   0) } },
    { UINT8_C( 59),
      { -INT64_C( 5205660368379732983),  INT64_C( 8276228550377061447), -INT64_C( 7475291365455649146),  INT64_C( 8472677639813096178) },
      {  INT64_C(  633882472542632347), -INT64_C( 8056010604140169284),  INT64_C(  911612899081956339), -INT64_C( 5334520241850082539) },
      {  INT64_C( 9023006035371178611),  INT64_C( 5390768574830593316),  INT64_C(                   0),  INT64_C(  787200493539927002) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_epi64(test_vec[i].b);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_mullo_epi64(k, a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_mullo_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_mullo_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mullo_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 31159), -INT16_C( 12097),  INT16_C( 29918),  INT16_C(  1377),  INT16_C( 32398),  INT16_C(  6544), -INT16_C( 30801),  INT16_C( 30357),
         INT16_C( 14887),  INT16_C(  9940),  INT16_C(  3636), -INT16_C( 23516),  INT16_C( 28807), -INT16_C(  9340), -INT16_C( 22852),  INT16_C(  1558),
        -INT16_C( 10708),  INT16_C(  2774),  INT16_C( 14154), -INT16_C( 10224), -INT16_C( 24394),  INT16_C( 26098), -INT16_C( 30937),  INT16_C( 20444),
        -INT16_C( 20287), -INT16_C(  2699), -INT16_C( 26177),  INT16_C( 18073),  INT16_C(  7434), -INT16_C( 14815),  INT16_C( 14275), -INT16_C(  3892) },
      { -INT16_C( 23795),  INT16_C( 22778),  INT16_C(  2778), -INT16_C( 28624),  INT16_C(  8875), -INT16_C( 11530), -INT16_C( 11607),  INT16_C( 27169),
        -INT16_C( 26750),  INT16_C( 16736), -INT16_C(  1744),  INT16_C( 14983), -INT16_C( 22505), -INT16_C(  9727), -INT16_C( 12832), -INT16_C(  4662),
        -INT16_C( 14992),  INT16_C( 19269),  INT16_C( 30415),  INT16_C( 31451), -INT16_C( 11880),  INT16_C( 16973),  INT16_C( 28323),  INT16_C(  9900),
         INT16_C(  3077),  INT16_C( 13927), -INT16_C(  4346),  INT16_C(  7536),  INT16_C( 29079),  INT16_C( 30711), -INT16_C( 15809), -INT16_C( 20635) },
      {  INT16_C( 19637), -INT16_C( 32122),  INT16_C( 12556), -INT16_C( 28112),  INT16_C( 25818), -INT16_C( 20384),  INT16_C(  8327), -INT16_C(  1227),
        -INT16_C( 30514),  INT16_C( 25472),  INT16_C( 15808), -INT16_C( 18692), -INT16_C( 19423),  INT16_C( 17284),  INT16_C( 28800),  INT16_C( 11100),
        -INT16_C( 28864), -INT16_C( 25170), -INT16_C( 12074),  INT16_C( 30128),  INT16_C(   528),  INT16_C(  3530), -INT16_C( 12331),  INT16_C( 20432),
         INT16_C( 32709),  INT16_C( 28691), -INT16_C(  5254),  INT16_C( 14320), -INT16_C( 29978), -INT16_C( 32553),  INT16_C( 32509),  INT16_C( 29820) } },
    { { -INT16_C( 21881),  INT16_C( 22266), -INT16_C( 10720), -INT16_C( 17967),  INT16_C(  7847),  INT16_C( 19451), -INT16_C( 22644), -INT16_C( 28047),
        -INT16_C( 10060), -INT16_C( 17720),  INT16_C( 14535),  INT16_C( 24535), -INT16_C( 12630), -INT16_C(  5674),  INT16_C( 15248),  INT16_C(  6040),
        -INT16_C( 27674),  INT16_C(  1646),  INT16_C( 16233),  INT16_C(  4287), -INT16_C( 17827), -INT16_C(  5797), -INT16_C( 13214),  INT16_C(  5755),
         INT16_C( 17317),  INT16_C( 27856), -INT16_C( 22660),  INT16_C(  9931), -INT16_C( 23947),  INT16_C(  1551), -INT16_C( 22563), -INT16_C( 15587) },
      { -INT16_C( 29894), -INT16_C( 23606), -INT16_C( 30262),  INT16_C( 10164),  INT16_C(  3908), -INT16_C( 23023), -INT16_C( 29476), -INT16_C( 32324),
        -INT16_C( 29488),  INT16_C( 19693), -INT16_C( 18125), -INT16_C( 22414), -INT16_C( 32421),  INT16_C( 14510), -INT16_C( 13272),  INT16_C( 25596),
        -INT16_C( 14761),  INT16_C(  8710), -INT16_C( 17841), -INT16_C( 27831),  INT16_C( 23242), -INT16_C( 22983), -INT16_C(  2585), -INT16_C( 18649),
         INT16_C(  5249), -INT16_C( 19453),  INT16_C( 30157),  INT16_C( 10333),  INT16_C(  3062),  INT16_C(  7777),  INT16_C( 24023),  INT16_C( 12161) },
      { -INT16_C(  4202), -INT16_C( 12476),  INT16_C(  5440),  INT16_C( 32244), -INT16_C(  4772), -INT16_C( 12885), -INT16_C( 29616),  INT16_C( 31740),
        -INT16_C( 32192),  INT16_C( 19240),  INT16_C(  7845), -INT16_C( 14914),  INT16_C(  8302), -INT16_C( 16524),  INT16_C(  3712),  INT16_C(   416),
         INT16_C( 10026), -INT16_C( 15724), -INT16_C(  9369),  INT16_C( 29559), -INT16_C( 16542), -INT16_C(  2237),  INT16_C( 13934),  INT16_C( 22973),
        -INT16_C(  1499), -INT16_C( 31120), -INT16_C( 13748), -INT16_C( 12353),  INT16_C(  9070),  INT16_C(  3503),  INT16_C( 17307), -INT16_C( 23395) } },
    { { -INT16_C( 30685),  INT16_C( 29265), -INT16_C( 26046),  INT16_C(  3078),  INT16_C( 16373), -INT16_C(  9038), -INT16_C(  9931), -INT16_C( 18797),
        -INT16_C( 26898), -INT16_C( 17557), -INT16_C( 14325),  INT16_C(   484),  INT16_C( 17875), -INT16_C( 21729), -INT16_C( 24158), -INT16_C( 14886),
         INT16_C( 11049),  INT16_C( 27447),  INT16_C( 15813), -INT16_C( 17800),  INT16_C( 10877), -INT16_C( 19818),  INT16_C( 10500), -INT16_C(  3480),
        -INT16_C( 11329), -INT16_C( 13651), -INT16_C( 28261),  INT16_C( 28619), -INT16_C(  5162),  INT16_C( 30746), -INT16_C(  2932), -INT16_C( 19139) },
      {  INT16_C( 29983), -INT16_C(  7136), -INT16_C( 26446),  INT16_C( 12191),  INT16_C( 13763), -INT16_C( 14367),  INT16_C( 19039),  INT16_C(  7865),
         INT16_C( 26141), -INT16_C( 17943), -INT16_C( 19208), -INT16_C( 12760),  INT16_C( 17055),  INT16_C( 11079), -INT16_C( 31690),  INT16_C( 21984),
         INT16_C(   505), -INT16_C( 21447), -INT16_C( 10087),  INT16_C( 23771), -INT16_C( 17138),  INT16_C( 27939), -INT16_C(  9209),  INT16_C(  9355),
         INT16_C( 29763),  INT16_C( 15325),  INT16_C(  1321), -INT16_C( 14327),  INT16_C( 20551),  INT16_C( 32244), -INT16_C( 11051), -INT16_C( 12590) },
      {  INT16_C( 31549),  INT16_C( 28192),  INT16_C( 29156), -INT16_C( 28230),  INT16_C( 28831),  INT16_C( 22130), -INT16_C(  4949),  INT16_C( 10811),
        -INT16_C(  4874), -INT16_C(  6301), -INT16_C( 31064), -INT16_C( 15456), -INT16_C( 15347), -INT16_C( 21863), -INT16_C( 24532), -INT16_C( 32576),
         INT16_C(  9185), -INT16_C( 11457),  INT16_C(  8893), -INT16_C( 23384), -INT16_C( 25642),  INT16_C( 18562), -INT16_C( 28900),  INT16_C( 15992),
        -INT16_C(  2307), -INT16_C( 10663),  INT16_C( 22739), -INT16_C( 31197),  INT16_C( 18522),  INT16_C( 10952),  INT16_C( 26748), -INT16_C( 15862) } },
    { {  INT16_C(  3285),  INT16_C( 28538),  INT16_C( 22244), -INT16_C(  3381), -INT16_C(  4333),  INT16_C(  6751), -INT16_C(  5173),  INT16_C(  3646),
         INT16_C(  7263), -INT16_C( 30647),  INT16_C( 21281),  INT16_C( 26961),  INT16_C( 17827),  INT16_C( 30950), -INT16_C( 18151), -INT16_C(  4281),
        -INT16_C( 15931), -INT16_C( 22178),  INT16_C( 10519),  INT16_C( 10908), -INT16_C(  1256), -INT16_C(  7100), -INT16_C( 31770),  INT16_C( 18162),
         INT16_C( 15519), -INT16_C( 16178),  INT16_C(  8079),  INT16_C( 12841),  INT16_C(  4196),  INT16_C( 32427), -INT16_C(  3383), -INT16_C( 29075) },
      { -INT16_C( 13389), -INT16_C( 13513), -INT16_C( 11276),  INT16_C(  3573),  INT16_C( 15055), -INT16_C( 18959), -INT16_C(  7235),  INT16_C( 23803),
        -INT16_C( 13793), -INT16_C( 20964),  INT16_C( 18153),  INT16_C( 20193), -INT16_C( 29610),  INT16_C(  8140),  INT16_C( 14718),  INT16_C( 12717),
        -INT16_C(  7164), -INT16_C(  1796), -INT16_C(  3400), -INT16_C( 30971), -INT16_C(  2516), -INT16_C(  5828),  INT16_C( 14554), -INT16_C(  1723),
         INT16_C( 24834), -INT16_C(  5208), -INT16_C( 30297), -INT16_C(   711),  INT16_C(  1301), -INT16_C( 27876), -INT16_C( 14018),  INT16_C( 17092) },
      { -INT16_C(  8209), -INT16_C( 20170), -INT16_C( 17072), -INT16_C( 21689), -INT16_C( 24995), -INT16_C(   401),  INT16_C(  5599),  INT16_C( 16074),
         INT16_C( 25985), -INT16_C( 31236), -INT16_C( 20727),  INT16_C( 15921), -INT16_C( 30526),  INT16_C( 12616), -INT16_C( 21682),  INT16_C( 18939),
         INT16_C( 31508), -INT16_C( 14200),  INT16_C( 18056),  INT16_C(  6412),  INT16_C( 14368),  INT16_C( 25584), -INT16_C( 24100), -INT16_C( 32454),
        -INT16_C( 18370), -INT16_C( 24272),  INT16_C(  7497), -INT16_C( 20447),  INT16_C( 19508),  INT16_C(  2996), -INT16_C( 25170),  INT16_C(  9588) } },
    { { -INT16_C( 15954),  INT16_C( 26171),  INT16_C( 16563), -INT16_C(  8211),  INT16_C( 10551),  INT16_C(  4552),  INT16_C(  3425),  INT16_C( 25354),
        -INT16_C( 19858),  INT16_C(  5711), -INT16_C( 30661),  INT16_C( 20499),  INT16_C( 12430), -INT16_C( 13085), -INT16_C( 22279), -INT16_C( 22769),
         INT16_C( 19049),  INT16_C(  7181), -INT16_C(  1398), -INT16_C( 15877), -INT16_C( 15580), -INT16_C( 31278), -INT16_C(  8752),  INT16_C( 16105),
         INT16_C( 14479), -INT16_C( 13484),  INT16_C( 26816),  INT16_C( 19995), -INT16_C(   104), -INT16_C( 28389),  INT16_C( 10919),  INT16_C(  4153) },
      {  INT16_C( 18036), -INT16_C(   468),  INT16_C( 10049),  INT16_C( 26048), -INT16_C( 27926), -INT16_C( 17686), -INT16_C( 11409), -INT16_C(     8),
         INT16_C( 19723), -INT16_C( 13110), -INT16_C(  6731),  INT16_C( 19738),  INT16_C( 13796), -INT16_C( 29730),  INT16_C(  5983), -INT16_C( 11365),
        -INT16_C( 14498), -INT16_C( 24622), -INT16_C( 27922), -INT16_C( 10236), -INT16_C(  4572), -INT16_C( 27502), -INT16_C( 29758), -INT16_C( 12909),
         INT16_C( 24024), -INT16_C( 29287), -INT16_C( 19390),  INT16_C( 10202), -INT16_C( 18199),  INT16_C( 18866),  INT16_C( 20176),  INT16_C( 11804) },
      {  INT16_C( 22232),  INT16_C(  7204), -INT16_C( 19853),  INT16_C( 29376),  INT16_C(  2630), -INT16_C( 28464), -INT16_C( 16369), -INT16_C(  6224),
        -INT16_C( 16198), -INT16_C( 29098),  INT16_C(  6327), -INT16_C( 10002), -INT16_C( 23432), -INT16_C(  4646),  INT16_C(  4967), -INT16_C( 31979),
        -INT16_C(  3698),  INT16_C(  5546), -INT16_C( 24500), -INT16_C( 12308), -INT16_C(  5872), -INT16_C( 17980),  INT16_C(  1952), -INT16_C( 19253),
        -INT16_C( 21592), -INT16_C( 14028),  INT16_C(   384), -INT16_C( 24578), -INT16_C(  7848), -INT16_C( 26682), -INT16_C( 30288),  INT16_C(  1084) } },
    { { -INT16_C(  4587),  INT16_C(  1229), -INT16_C( 11904), -INT16_C( 23076),  INT16_C( 28607), -INT16_C( 32455), -INT16_C( 13062), -INT16_C( 11697),
        -INT16_C(  6103),  INT16_C( 27487),  INT16_C( 14748), -INT16_C( 31086),  INT16_C( 17905), -INT16_C( 15921), -INT16_C(  5229), -INT16_C( 22289),
        -INT16_C( 17190),  INT16_C( 23212), -INT16_C( 30323),  INT16_C( 19967),  INT16_C( 14584), -INT16_C(  3378),  INT16_C(  7428),  INT16_C( 11716),
         INT16_C(  8966), -INT16_C( 23911),  INT16_C( 11100),  INT16_C( 19752), -INT16_C(  2192),  INT16_C(   783), -INT16_C(   285), -INT16_C( 16980) },
      {  INT16_C( 22715),  INT16_C( 18455),  INT16_C(  6113), -INT16_C(  9835),  INT16_C( 25679),  INT16_C( 21707), -INT16_C( 28799), -INT16_C( 30847),
         INT16_C(  6834),  INT16_C(  3626),  INT16_C( 21062), -INT16_C( 18852),  INT16_C( 27466),  INT16_C( 11706),  INT16_C( 26217),  INT16_C(  9450),
         INT16_C(   446), -INT16_C( 24467),  INT16_C(   536),  INT16_C( 26745),  INT16_C( 17766), -INT16_C(  5956),  INT16_C( 15828), -INT16_C( 30865),
        -INT16_C( 26280), -INT16_C( 24939), -INT16_C(  3604),  INT16_C( 13908),  INT16_C(  3676), -INT16_C( 14749),  INT16_C( 19828),  INT16_C( 13290) },
      {  INT16_C(  8535),  INT16_C(  5739), -INT16_C( 24192),  INT16_C(  1292),  INT16_C(  6129),  INT16_C( 11315), -INT16_C(  4102), -INT16_C( 23857),
        -INT16_C( 27006), -INT16_C( 12394), -INT16_C( 18264),  INT16_C( 10360), -INT16_C(  3414),  INT16_C( 13158),  INT16_C( 12619),  INT16_C(  1654),
         INT16_C(   972),  INT16_C(  6972), -INT16_C(   200),  INT16_C( 30087), -INT16_C( 30000), -INT16_C(   184), -INT16_C(  1200),  INT16_C( 13308),
        -INT16_C( 24560),  INT16_C(  4365), -INT16_C( 27440), -INT16_C( 16096),  INT16_C(  3136), -INT16_C( 14131), -INT16_C( 14884), -INT16_C( 23752) } },
    { {  INT16_C( 22350),  INT16_C( 26579),  INT16_C( 19546), -INT16_C( 16177), -INT16_C( 29807),  INT16_C( 26280),  INT16_C(  6344),  INT16_C(  8429),
        -INT16_C( 32079), -INT16_C( 25154),  INT16_C(  4980), -INT16_C( 12077),  INT16_C( 13857), -INT16_C( 26986), -INT16_C( 32381), -INT16_C( 11575),
        -INT16_C( 25384),  INT16_C( 12857),  INT16_C(  2280),  INT16_C( 31475), -INT16_C( 25709),  INT16_C( 23520), -INT16_C( 12877),  INT16_C( 25980),
         INT16_C( 14927), -INT16_C( 15614), -INT16_C( 10675),  INT16_C( 28564),  INT16_C( 10764), -INT16_C( 28667), -INT16_C( 12629), -INT16_C( 31646) },
      { -INT16_C( 25750),  INT16_C( 21174), -INT16_C( 22109),  INT16_C( 14028), -INT16_C( 21435), -INT16_C(  1903),  INT16_C(  3449), -INT16_C( 13987),
         INT16_C( 24648), -INT16_C( 27252),  INT16_C(  8246),  INT16_C( 16900),  INT16_C(  2379), -INT16_C(  2350),  INT16_C( 13527),  INT16_C( 16762),
         INT16_C( 12751),  INT16_C( 29332),  INT16_C( 24794),  INT16_C(  8104),  INT16_C( 14861), -INT16_C( 31208),  INT16_C( 30023), -INT16_C( 28849),
        -INT16_C(  9003),  INT16_C(  2853),  INT16_C( 10748),  INT16_C( 18254),  INT16_C(  8243),  INT16_C(  2622), -INT16_C( 18347),  INT16_C(  9292) },
      {  INT16_C( 24652),  INT16_C( 26114),  INT16_C(  1870),  INT16_C( 20212),  INT16_C(  2581), -INT16_C(  6872), -INT16_C(  8568),  INT16_C(  2841),
         INT16_C(  8648), -INT16_C(  9752), -INT16_C( 25992), -INT16_C( 22196),  INT16_C(  1195), -INT16_C( 21748),  INT16_C( 24837),  INT16_C( 31946),
         INT16_C( 10920),  INT16_C( 27380), -INT16_C( 27248),  INT16_C(  7288),  INT16_C( 13431), -INT16_C(  8960), -INT16_C(  9307), -INT16_C( 27324),
         INT16_C( 26555),  INT16_C( 17738),  INT16_C( 18636),  INT16_C(  2840), -INT16_C(  8092),  INT16_C(  4918), -INT16_C( 31033),  INT16_C(  5400) } },
    { { -INT16_C(  7959), -INT16_C( 15209),  INT16_C( 16192),  INT16_C( 19939), -INT16_C(  1159), -INT16_C( 15916),  INT16_C(  9073),  INT16_C( 18000),
         INT16_C( 30207), -INT16_C(   942), -INT16_C( 24417), -INT16_C( 11709), -INT16_C( 32320),  INT16_C(  5596),  INT16_C( 10298),  INT16_C(  9018),
        -INT16_C( 12024),  INT16_C( 18919), -INT16_C( 13552), -INT16_C( 30058),  INT16_C( 27334),  INT16_C( 14155), -INT16_C( 25714), -INT16_C( 29314),
        -INT16_C( 12271), -INT16_C( 20343), -INT16_C( 12944),  INT16_C( 12418),  INT16_C( 24142), -INT16_C( 30650), -INT16_C( 32633), -INT16_C( 28756) },
      { -INT16_C( 27823),  INT16_C( 25048),  INT16_C( 28510),  INT16_C(  9707),  INT16_C( 14041),  INT16_C( 26460), -INT16_C(  9518), -INT16_C(  7179),
         INT16_C( 32426),  INT16_C(  6803),  INT16_C(  5451), -INT16_C( 26037), -INT16_C( 28301), -INT16_C(  1502), -INT16_C( 12783),  INT16_C( 25226),
         INT16_C( 25186), -INT16_C( 16189), -INT16_C( 20527), -INT16_C( 21531),  INT16_C( 17125), -INT16_C( 18670),  INT16_C(  1820), -INT16_C( 14438),
         INT16_C( 11654), -INT16_C( 11807),  INT16_C( 11330), -INT16_C( 18837), -INT16_C( 28995), -INT16_C( 12624),  INT16_C( 14940), -INT16_C( 16848) },
      { -INT16_C(  2887),  INT16_C(  5736), -INT16_C(  1664),  INT16_C( 20065), -INT16_C( 20591), -INT16_C(  3024),  INT16_C( 19634),  INT16_C( 14992),
        -INT16_C(  8874),  INT16_C( 14102),  INT16_C(  6549), -INT16_C(  6239),  INT16_C(  2368), -INT16_C( 16584),  INT16_C( 22490),  INT16_C( 12612),
         INT16_C(  5392), -INT16_C( 29963), -INT16_C( 18416),  INT16_C( 10798), -INT16_C( 28898), -INT16_C( 32698), -INT16_C(  6776),  INT16_C(  4044),
        -INT16_C(  6682),  INT16_C(   361),  INT16_C( 14048), -INT16_C( 19882), -INT16_C(  7274),  INT16_C(  1056), -INT16_C( 14716), -INT16_C( 26560) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mullo_epi16(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mullo_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mullo_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(   355975199), -INT32_C(  1700022260),  INT32_C(  1271212220),  INT32_C(  1338643536),  INT32_C(   295713745),  INT32_C(   236686063),  INT32_C(  1041828026),  INT32_C(  2021440918),
         INT32_C(   140657916), -INT32_C(  1197273604), -INT32_C(  2063308747),  INT32_C(  1305857660), -INT32_C(   111250166), -INT32_C(  1190692353),  INT32_C(   418914690), -INT32_C(   628067106) },
      { -INT32_C(  1243417927),  INT32_C(   896435711),  INT32_C(  1773826797),  INT32_C(  1253478208),  INT32_C(    54727684),  INT32_C(   280841102),  INT32_C(  1244181868), -INT32_C(   500909783),
        -INT32_C(  1416100181),  INT32_C(  2044724876), -INT32_C(  1193043336),  INT32_C(   771987754),  INT32_C(  1009927854), -INT32_C(    45289583), -INT32_C(   850955100), -INT32_C(   642749651) },
      {  INT32_C(  1891294105), -INT32_C(   657289228), -INT32_C(  1505595892),  INT32_C(  1690158080), -INT32_C(   438287548),  INT32_C(  1735267730),  INT32_C(   113979512), -INT32_C(  2045525242),
         INT32_C(   820116564),  INT32_C(   603950544),  INT32_C(  2143271640),  INT32_C(  1089469528), -INT32_C(  2044180276),  INT32_C(    47524463), -INT32_C(   751673528),  INT32_C(  1390236934) } },
    { { -INT32_C(    24885134), -INT32_C(   965254066),  INT32_C(   679369470), -INT32_C(  1571323404),  INT32_C(  1507756488),  INT32_C(   475474552), -INT32_C(   857104738),  INT32_C(  2091227402),
         INT32_C(   796535265), -INT32_C(  1946816115),  INT32_C(  1085568076), -INT32_C(  1109259275),  INT32_C(   202883220), -INT32_C(  2010616086),  INT32_C(   374608140), -INT32_C(  1936524885) },
      { -INT32_C(  1346630622),  INT32_C(  1245426174),  INT32_C(   462090021), -INT32_C(  1898418950),  INT32_C(   379252524),  INT32_C(  1788789341),  INT32_C(  2122380243),  INT32_C(   252384236),
         INT32_C(   499042079), -INT32_C(  1637353096), -INT32_C(   491130392), -INT32_C(  1989111459), -INT32_C(   560002431), -INT32_C(  1605878068),  INT32_C(   488556849), -INT32_C(    80991780) },
      {  INT32_C(   581891876),  INT32_C(   550073188), -INT32_C(  1474148170),  INT32_C(  1915213896),  INT32_C(  1194419808),  INT32_C(  1072258456), -INT32_C(  1189235142),  INT32_C(  1551029048),
         INT32_C(  1213659455),  INT32_C(   150363928), -INT32_C(  1477381920),  INT32_C(  1544822017), -INT32_C(   110505324), -INT32_C(  1125839240), -INT32_C(  1562267828), -INT32_C(   357633548) } },
    { {  INT32_C(  1763306480), -INT32_C(   855146268), -INT32_C(   810565518), -INT32_C(   765910959), -INT32_C(   139331542),  INT32_C(  1738012982), -INT32_C(  1635404350), -INT32_C(   811945505),
        -INT32_C(  2126990436), -INT32_C(  1521598669),  INT32_C(  1366687231),  INT32_C(  1210306077), -INT32_C(    46148410), -INT32_C(  1872439602),  INT32_C(  1781524875),  INT32_C(   926599579) },
      { -INT32_C(  1346866564), -INT32_C(  1319827790),  INT32_C(   553830916),  INT32_C(  1583949464), -INT32_C(   933517062),  INT32_C(   173654142),  INT32_C(  1148487849), -INT32_C(   847466927),
        -INT32_C(   746834911),  INT32_C(  1065668923),  INT32_C(   861964187), -INT32_C(  1483552083), -INT32_C(   244257422),  INT32_C(  1476118957), -INT32_C(  1566871727),  INT32_C(  1064245022) },
      {  INT32_C(   404695104),  INT32_C(   737212040),  INT32_C(   528283080),  INT32_C(   519058968), -INT32_C(  1248466684), -INT32_C(  1216665452), -INT32_C(  1833169646), -INT32_C(  1881417329),
        -INT32_C(  1067788516),  INT32_C(   502084545), -INT32_C(  1744682395),  INT32_C(  2019503001),  INT32_C(  1527254572),  INT32_C(  1590093622), -INT32_C(  1131075077),  INT32_C(  1345686826) } },
    { { -INT32_C(  2028803252),  INT32_C(  1489409725), -INT32_C(   896784867),  INT32_C(  1668423408), -INT32_C(  1185619445), -INT32_C(    66039893), -INT32_C(   593581122),  INT32_C(   253431235),
        -INT32_C(  1248449032), -INT32_C(   519152444),  INT32_C(  1940691586), -INT32_C(  1009377608), -INT32_C(  1417926144),  INT32_C(   933727353), -INT32_C(    82557640),  INT32_C(  1242181458) },
      {  INT32_C(   553689181),  INT32_C(  2114064124),  INT32_C(  1626451624), -INT32_C(   870070324),  INT32_C(  1786224881),  INT32_C(  1688346156), -INT32_C(  1252018589),  INT32_C(  1107323365),
         INT32_C(    90374153), -INT32_C(  1232837106), -INT32_C(   518621932),  INT32_C(   783104317),  INT32_C(   110699993), -INT32_C(  1486210237), -INT32_C(   698561807),  INT32_C(  1025072179) },
      { -INT32_C(  1326515556),  INT32_C(  1932834828), -INT32_C(   282216184),  INT32_C(  1875720000),  INT32_C(   659336283),  INT32_C(  1918483300),  INT32_C(   821526138), -INT32_C(   505918865),
         INT32_C(    19897784),  INT32_C(   885992120),  INT32_C(  1967488040),  INT32_C(   691292632),  INT32_C(  1750178304),  INT32_C(  1196664491),  INT32_C(   215177656), -INT32_C(  1844779690) } },
    { {  INT32_C(  1782742108), -INT32_C(   148846878),  INT32_C(  2044212796),  INT32_C(  1235715440), -INT32_C(   296795990),  INT32_C(  1821751931), -INT32_C(  1220284028), -INT32_C(  1426826162),
        -INT32_C(  1156237352),  INT32_C(   967980541), -INT32_C(   592278932), -INT32_C(  1171957233),  INT32_C(   380138906), -INT32_C(  1283310289),  INT32_C(  2087372078), -INT32_C(    98083039) },
      { -INT32_C(  1816839018), -INT32_C(   573741199),  INT32_C(    12156913),  INT32_C(  1958404057), -INT32_C(  2088082860), -INT32_C(   835318625), -INT32_C(   213212974), -INT32_C(  1779600897),
         INT32_C(   522822317), -INT32_C(    84085239),  INT32_C(  1341896309), -INT32_C(   373049963), -INT32_C(  1217639144),  INT32_C(   730178137),  INT32_C(  1109381186), -INT32_C(   271119295) },
      { -INT32_C(  1725534744), -INT32_C(  1569278014),  INT32_C(   684642940),  INT32_C(   261925872),  INT32_C(  1650871240), -INT32_C(   978413979), -INT32_C(   388587960),  INT32_C(     6053810),
        -INT32_C(   196776712),  INT32_C(  2117141477), -INT32_C(   672789668),  INT32_C(    14091707),  INT32_C(  2116180080), -INT32_C(  2123758761), -INT32_C(  1873922596),  INT32_C(   271299425) } },
    { { -INT32_C(  1207041873),  INT32_C(  1823673078),  INT32_C(  1438363328),  INT32_C(  2067693155),  INT32_C(   607365835), -INT32_C(  1890535348), -INT32_C(   892244088),  INT32_C(   716810363),
        -INT32_C(  1612462167), -INT32_C(  1844734255), -INT32_C(  1477982652),  INT32_C(   253961796),  INT32_C(   489969360), -INT32_C(  1750301682),  INT32_C(  1851882995), -INT32_C(   828827099) },
      { -INT32_C(  1301381919),  INT32_C(  1447328018), -INT32_C(  2063782848),  INT32_C(   580132946), -INT32_C(  2059417482),  INT32_C(  1058859852), -INT32_C(  1901232792),  INT32_C(  2019313303),
        -INT32_C(   735393086), -INT32_C(  2077593788), -INT32_C(   318232421), -INT32_C(  1089495992), -INT32_C(  1337700508), -INT32_C(  1544593350),  INT32_C(  1949408733), -INT32_C(  1494446621) },
      { -INT32_C(  2049614385), -INT32_C(   769848500),  INT32_C(  1689563136), -INT32_C(  2129863754), -INT32_C(  1942728302), -INT32_C(  2123072880),  INT32_C(  2004006720),  INT32_C(   734528141),
        -INT32_C(  1679644654),  INT32_C(  1405267588),  INT32_C(   107981612), -INT32_C(  1490013408), -INT32_C(  1636034240),  INT32_C(  1101867820),  INT32_C(  1273676231), -INT32_C(  1656451121) } },
    { { -INT32_C(  1669720488),  INT32_C(   539010437),  INT32_C(   353183949),  INT32_C(   701767109), -INT32_C(  1495656340),  INT32_C(  1430899064),  INT32_C(  1254718054),  INT32_C(  1626387720),
         INT32_C(  1375496908), -INT32_C(   596501489),  INT32_C(   166887236),  INT32_C(   137610908),  INT32_C(  1471090143),  INT32_C(  1034811606),  INT32_C(  2072475251), -INT32_C(   119834836) },
      { -INT32_C(   246818847),  INT32_C(   936229875), -INT32_C(   683557061),  INT32_C(  1709208710),  INT32_C(  1471975297), -INT32_C(    90936953),  INT32_C(   209001440),  INT32_C(  1946439826),
         INT32_C(   442846503),  INT32_C(  1146237449),  INT32_C(  2015073266), -INT32_C(  1998718201),  INT32_C(   316643722), -INT32_C(   485723133),  INT32_C(   586121871), -INT32_C(    90770478) },
      {  INT32_C(  1193904984),  INT32_C(  1986413631),  INT32_C(   847665727), -INT32_C(   941303522), -INT32_C(  1965383060), -INT32_C(   420776376),  INT32_C(  1758583616), -INT32_C(   168182128),
         INT32_C(  1141874964),  INT32_C(    38529671), -INT32_C(   226731448),  INT32_C(  1372676676), -INT32_C(  1494695626), -INT32_C(   103619966), -INT32_C(  1506897859),  INT32_C(  1106150936) } },
    { {  INT32_C(  1226111808),  INT32_C(   529360429),  INT32_C(     9939449),  INT32_C(   797471908),  INT32_C(   289499150),  INT32_C(  1811172828), -INT32_C(  1567759409),  INT32_C(   379331542),
         INT32_C(  1281404958),  INT32_C(   275508503),  INT32_C(   990970774), -INT32_C(  2056611465), -INT32_C(   577328383), -INT32_C(   934704392),  INT32_C(  1131075181),  INT32_C(   408553210) },
      { -INT32_C(   849036618),  INT32_C(  1054789799),  INT32_C(  1249505235), -INT32_C(  1966021752), -INT32_C(  2040043890),  INT32_C(  1582215409), -INT32_C(  2120042361),  INT32_C(  1973025982),
         INT32_C(  1564671670), -INT32_C(  1583669042), -INT32_C(  1729424369), -INT32_C(  2061321225),  INT32_C(   319588642), -INT32_C(  1049535942), -INT32_C(   784198893), -INT32_C(   985212145) },
      { -INT32_C(  1737023616), -INT32_C(   112707749),  INT32_C(  1700892475),  INT32_C(  1831559968), -INT32_C(   748729404),  INT32_C(   395086876),  INT32_C(   747498025), -INT32_C(  1274068780),
        -INT32_C(  1992885932), -INT32_C(  1201100670),  INT32_C(   620293578), -INT32_C(   859701551),  INT32_C(   647118626), -INT32_C(  1963152336),  INT32_C(   871498007), -INT32_C(  1147517274) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mullo_epi32(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mullo_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mask_mullo_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(   466803588), -INT32_C(  1704804295), -INT32_C(   294739476), -INT32_C(    39638017), -INT32_C(   523648680), -INT32_C(  1903034581), -INT32_C(  1147760279), -INT32_C(   982530232),
         INT32_C(   631872747),  INT32_C(  1220480092), -INT32_C(  1405735507), -INT32_C(  1314268583), -INT32_C(   946769253), -INT32_C(   615177614), -INT32_C(    57218124), -INT32_C(  1648294478) },
      UINT16_C(27554),
      { -INT32_C(  2106065214),  INT32_C(  2108630087),  INT32_C(  2052524241),  INT32_C(  1273885626),  INT32_C(   258891706), -INT32_C(   788913606), -INT32_C(   522802146),  INT32_C(   206272585),
        -INT32_C(  1114717578), -INT32_C(  1220854298),  INT32_C(     3314246),  INT32_C(  1061887877), -INT32_C(  1202800258), -INT32_C(    91731749),  INT32_C(  1054498548),  INT32_C(  1296704982) },
      { -INT32_C(   854927129),  INT32_C(  1535460629),  INT32_C(  1566291928),  INT32_C(  1419618262),  INT32_C(  1007479648),  INT32_C(   674665524), -INT32_C(   916057869),  INT32_C(   471248949),
        -INT32_C(  1628888696),  INT32_C(  1056534118), -INT32_C(    73640667),  INT32_C(  1548696060),  INT32_C(  1486379812), -INT32_C(   478032144),  INT32_C(   330098654),  INT32_C(   540001176) },
      { -INT32_C(   466803588), -INT32_C(  1710744365), -INT32_C(   294739476), -INT32_C(    39638017), -INT32_C(   523648680),  INT32_C(  1456765896), -INT32_C(  1147760279), -INT32_C(  1063774435),
        -INT32_C(   330398544),  INT32_C(  1592360356), -INT32_C(  1405735507), -INT32_C(  1912429588), -INT32_C(   946769253),  INT32_C(  1908377424),  INT32_C(  1962705816), -INT32_C(  1648294478) } },
    { {  INT32_C(  1253972452), -INT32_C(  1383483257),  INT32_C(   162014477), -INT32_C(  2107246498),  INT32_C(  1138490963), -INT32_C(  1423549236), -INT32_C(   608251069),  INT32_C(  2063396502),
        -INT32_C(  1899709945),  INT32_C(  2134592882), -INT32_C(   796334990),  INT32_C(   794029788), -INT32_C(  1200410900), -INT32_C(   849110646),  INT32_C(    44638828),  INT32_C(   394044688) },
      UINT16_C(16735),
      { -INT32_C(   510733659), -INT32_C(   624623279),  INT32_C(   617128401),  INT32_C(  1129493712), -INT32_C(   790766484),  INT32_C(  1408387498), -INT32_C(   923270580), -INT32_C(  1106684135),
         INT32_C(  2074056745),  INT32_C(  1783981209), -INT32_C(   695263995),  INT32_C(  1025106385),  INT32_C(  1745745598), -INT32_C(  1933836480),  INT32_C(   441758465),  INT32_C(   886660618) },
      { -INT32_C(  1884325642), -INT32_C(   486996771), -INT32_C(   222787551),  INT32_C(   674222698),  INT32_C(   143670728),  INT32_C(  1049905980),  INT32_C(   140044798),  INT32_C(  1027354951),
        -INT32_C(  2016613462),  INT32_C(   275367407), -INT32_C(  1207819698), -INT32_C(  1126157580), -INT32_C(  1413123985), -INT32_C(  1175889477), -INT32_C(  1983823294),  INT32_C(   499580531) },
      { -INT32_C(   607731058), -INT32_C(  1176475923),  INT32_C(   873194481),  INT32_C(  1791065632),  INT32_C(  1738141792), -INT32_C(  1423549236),  INT32_C(  1246327656),  INT32_C(  2063396502),
         INT32_C(  2018815546),  INT32_C(  2134592882), -INT32_C(   796334990),  INT32_C(   794029788), -INT32_C(  1200410900), -INT32_C(   849110646),  INT32_C(   482175042),  INT32_C(   394044688) } },
    { { -INT32_C(   643525911), -INT32_C(  1494675880),  INT32_C(   610200624), -INT32_C(  1914683874),  INT32_C(  1765320110),  INT32_C(  1092756223), -INT32_C(   674503836), -INT32_C(   873098783),
         INT32_C(  2091161892), -INT32_C(   685601369), -INT32_C(  1745125255),  INT32_C(  1814355134), -INT32_C(  2133500543), -INT32_C(   473761921), -INT32_C(  1128624678),  INT32_C(  1116188446) },
      UINT16_C(11081),
      { -INT32_C(   524750658), -INT32_C(  1017105720), -INT32_C(   291561783),  INT32_C(  1598759306),  INT32_C(  1666632353),  INT32_C(  1743794605),  INT32_C(  1947668461),  INT32_C(   278880337),
         INT32_C(   418404176),  INT32_C(  1406947721), -INT32_C(    96371857),  INT32_C(  1046056092),  INT32_C(    60928086), -INT32_C(   831876383), -INT32_C(   230522463),  INT32_C(   839049697) },
      { -INT32_C(  1035275464), -INT32_C(  1290459580),  INT32_C(  1051547298),  INT32_C(   947652578), -INT32_C(  1740955977),  INT32_C(  1365681584),  INT32_C(   138651687), -INT32_C(  1053145463),
         INT32_C(  2105771321),  INT32_C(  1295030443), -INT32_C(   796074258), -INT32_C(  1677195035), -INT32_C(   701218010),  INT32_C(   271031017), -INT32_C(   887592126), -INT32_C(   359902287) },
      {  INT32_C(  1206190992), -INT32_C(  1494675880),  INT32_C(   610200624), -INT32_C(  1659477548),  INT32_C(  1765320110),  INT32_C(  1092756223),  INT32_C(  1694364955), -INT32_C(   873098783),
         INT32_C(   684720336),  INT32_C(  1396952707), -INT32_C(  1745125255),  INT32_C(   123709324), -INT32_C(  2133500543),  INT32_C(  1805781193), -INT32_C(  1128624678),  INT32_C(  1116188446) } },
    { { -INT32_C(  2090397480), -INT32_C(  1747871832),  INT32_C(  1516723573), -INT32_C(  1930006427), -INT32_C(  1671288141),  INT32_C(   111971012), -INT32_C(  1496201739), -INT32_C(   258974184),
         INT32_C(   376698734),  INT32_C(    78464142),  INT32_C(   123606433),  INT32_C(   949179781),  INT32_C(  1154872703),  INT32_C(  1951039871),  INT32_C(  1578769478), -INT32_C(   397497734) },
      UINT16_C(49825),
      { -INT32_C(  1408880386), -INT32_C(  1832802252),  INT32_C(  1122453167),  INT32_C(  1396139902), -INT32_C(   170543189),  INT32_C(  1175526187), -INT32_C(   923759750),  INT32_C(  1921684083),
        -INT32_C(   165768766), -INT32_C(   393683143), -INT32_C(  1557499867), -INT32_C(  2097716777),  INT32_C(  1148701720),  INT32_C(  1636469223), -INT32_C(  2010482156), -INT32_C(   822430708) },
      {  INT32_C(  2126780485),  INT32_C(   476466679), -INT32_C(  1799384899),  INT32_C(   169260786),  INT32_C(  1783532930),  INT32_C(   734779414),  INT32_C(  1605629267), -INT32_C(   282153558),
        -INT32_C(  1133579579), -INT32_C(    52898753),  INT32_C(  1469093733), -INT32_C(   798906802),  INT32_C(  1245360180), -INT32_C(   613087608), -INT32_C(  1522849541), -INT32_C(  1701484075) },
      {  INT32_C(  2025522294), -INT32_C(  1747871832),  INT32_C(  1516723573), -INT32_C(  1930006427), -INT32_C(  1671288141),  INT32_C(   947756466), -INT32_C(  1496201739),  INT32_C(  1655109470),
         INT32_C(   376698734),  INT32_C(   111289095),  INT32_C(   123606433),  INT32_C(   949179781),  INT32_C(  1154872703),  INT32_C(  1951039871),  INT32_C(   726978972),  INT32_C(   225229308) } },
    { { -INT32_C(  1688861861),  INT32_C(  1016540887),  INT32_C(   345188550),  INT32_C(    48559566),  INT32_C(   760029093),  INT32_C(   537510437), -INT32_C(  1060748053),  INT32_C(   140204973),
         INT32_C(   899920222), -INT32_C(  1502463008),  INT32_C(   834274659), -INT32_C(  1623941382), -INT32_C(   489848387),  INT32_C(   772003395), -INT32_C(   940586726), -INT32_C(  2100344284) },
      UINT16_C(29691),
      {  INT32_C(   699325367),  INT32_C(   992940417),  INT32_C(  1994008898), -INT32_C(  1762158647),  INT32_C(  2104245114),  INT32_C(  1481016937),  INT32_C(   480406093), -INT32_C(  1550868756),
        -INT32_C(    70435463), -INT32_C(  1858667442), -INT32_C(   301527003),  INT32_C(   579141544), -INT32_C(  1549799366), -INT32_C(  1006836362),  INT32_C(  1004576335), -INT32_C(  1226936516) },
      { -INT32_C(    72242259), -INT32_C(   846403673), -INT32_C(  1598319368), -INT32_C(   842907501), -INT32_C(  1485807312), -INT32_C(  1788187578),  INT32_C(  1204832779),  INT32_C(  1744678586),
         INT32_C(    23244378), -INT32_C(  1899040874),  INT32_C(   372148867), -INT32_C(    85724982), -INT32_C(  1751035055), -INT32_C(   886305600),  INT32_C(   269679702),  INT32_C(    74911914) },
      {  INT32_C(  1316272043), -INT32_C(   200551897),  INT32_C(   345188550), -INT32_C(  1854904213),  INT32_C(   573858016), -INT32_C(  1468592970), -INT32_C(   414919857),  INT32_C(  2006572920),
         INT32_C(   121756298), -INT32_C(  1232558156),  INT32_C(   834274659), -INT32_C(  1623941382), -INT32_C(   352858022), -INT32_C(   625631104),  INT32_C(  1255731850), -INT32_C(  2100344284) } },
    { {  INT32_C(  1409735358),  INT32_C(  1289934025),  INT32_C(   677515358),  INT32_C(  1361265920),  INT32_C(  1491649688),  INT32_C(   656610512), -INT32_C(  1154009584),  INT32_C(    79671110),
         INT32_C(  1381614985), -INT32_C(   123847782),  INT32_C(  1277231180), -INT32_C(   576830395), -INT32_C(   650738168), -INT32_C(  1426040421), -INT32_C(   714721393),  INT32_C(  1876567782) },
      UINT16_C(13291),
      {  INT32_C(  1601078721), -INT32_C(  1621116290), -INT32_C(  1511807993), -INT32_C(  1205081214), -INT32_C(  1005467964), -INT32_C(   654532238),  INT32_C(  1358881398),  INT32_C(   327412306),
        -INT32_C(   311168401), -INT32_C(  1282616660), -INT32_C(   111644809),  INT32_C(   531727451), -INT32_C(  1125858742), -INT32_C(   627711901),  INT32_C(   338334658), -INT32_C(   316100995) },
      {  INT32_C(  1289395104), -INT32_C(   469735571),  INT32_C(   836589782),  INT32_C(   659656412), -INT32_C(  1260178095), -INT32_C(   678529003), -INT32_C(  1980974836),  INT32_C(   125178983),
         INT32_C(   475222447), -INT32_C(  1912515656), -INT32_C(  2000626004), -INT32_C(  1095822995),  INT32_C(  1534235462),  INT32_C(   389153035),  INT32_C(   564207290), -INT32_C(   517466318) },
      { -INT32_C(  1033276512), -INT32_C(  1748297306),  INT32_C(   677515358), -INT32_C(   192774216),  INT32_C(  1491649688), -INT32_C(  2086490534), -INT32_C(   396237944),  INT32_C(  1747824382),
        -INT32_C(  1074472735), -INT32_C(   265212000),  INT32_C(  1277231180), -INT32_C(   576830395), -INT32_C(   720265668), -INT32_C(  1297676479), -INT32_C(   714721393),  INT32_C(  1876567782) } },
    { {  INT32_C(   553548648),  INT32_C(  2075131855),  INT32_C(  1241739229), -INT32_C(   972508288),  INT32_C(  1361148742),  INT32_C(   912872316), -INT32_C(  1537799566), -INT32_C(  1970897119),
        -INT32_C(   911571718),  INT32_C(  1615092099),  INT32_C(  1219184840),  INT32_C(  1091482619), -INT32_C(  1433260242), -INT32_C(   169804925),  INT32_C(   664352517), -INT32_C(  1313792074) },
      UINT16_C(23459),
      { -INT32_C(  1078647174),  INT32_C(   839351687), -INT32_C(   723189050), -INT32_C(   721152957), -INT32_C(  1613658178),  INT32_C(   366401148), -INT32_C(  1372221955), -INT32_C(  1207248834),
        -INT32_C(  2038972417),  INT32_C(    45645372), -INT32_C(  1009279616),  INT32_C(  1889131441), -INT32_C(   569415070),  INT32_C(  1056171328), -INT32_C(  1326700430),  INT32_C(     6944257) },
      { -INT32_C(   226041675), -INT32_C(   537641377), -INT32_C(  1918645285),  INT32_C(   150813862), -INT32_C(   421131098),  INT32_C(  1680136945),  INT32_C(    51646722), -INT32_C(  1123844857),
        -INT32_C(  1112569506), -INT32_C(  1550015545),  INT32_C(   355483503),  INT32_C(   555625851),  INT32_C(   705168441), -INT32_C(   510776098),  INT32_C(  1172611901),  INT32_C(  2114119712) },
      { -INT32_C(  2070219710), -INT32_C(  1629862119),  INT32_C(  1241739229), -INT32_C(   972508288),  INT32_C(  1361148742),  INT32_C(   458588860), -INT32_C(  1537799566), -INT32_C(  1566822478),
        -INT32_C(   133630302), -INT32_C(  1235594076),  INT32_C(  1219184840), -INT32_C(  1253528821),  INT32_C(   371734482), -INT32_C(   169804925),  INT32_C(  1166948650), -INT32_C(  1313792074) } },
    { {  INT32_C(   960213361), -INT32_C(  1008936876), -INT32_C(  1814492137),  INT32_C(  1924462393), -INT32_C(   627262213),  INT32_C(   649800681), -INT32_C(   294936626), -INT32_C(   110269049),
         INT32_C(  1932699678), -INT32_C(  1741287808),  INT32_C(  1395330842),  INT32_C(    46522118), -INT32_C(  2049154660),  INT32_C(  1521194892), -INT32_C(  1102506186), -INT32_C(  1548241276) },
      UINT16_C(59742),
      {  INT32_C(  1291312918), -INT32_C(  1571024521),  INT32_C(   696345188),  INT32_C(  1082793316),  INT32_C(  1322719138), -INT32_C(  1167782287), -INT32_C(  2089752116), -INT32_C(  1569927284),
         INT32_C(   636445614), -INT32_C(   658027660),  INT32_C(   302074029),  INT32_C(   139627366),  INT32_C(   341191330),  INT32_C(    80657208),  INT32_C(   830947237), -INT32_C(  1126894834) },
      { -INT32_C(   891174058), -INT32_C(  1146968050),  INT32_C(  1456317424), -INT32_C(   782294994), -INT32_C(  2098875062),  INT32_C(   377926513),  INT32_C(     4656626),  INT32_C(  1455168256),
        -INT32_C(   333406754),  INT32_C(   917029445), -INT32_C(  1819511451), -INT32_C(   547034219),  INT32_C(   308365729), -INT32_C(   282531843), -INT32_C(   185569292),  INT32_C(  1766501515) },
      {  INT32_C(   960213361),  INT32_C(  1187843202), -INT32_C(  1392121408),  INT32_C(  1754908664),  INT32_C(  1171021524),  INT32_C(   649800681), -INT32_C(   735027496), -INT32_C(   110269049),
        -INT32_C(  1820711196), -INT32_C(  1741287808),  INT32_C(  1395330842), -INT32_C(   338049954), -INT32_C(  2049154660), -INT32_C(   738292136), -INT32_C(   936299452), -INT32_C(  1694932838) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mullo_epi32(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mullo_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_maskz_mullo_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(24017),
      { -INT32_C(   266698699),  INT32_C(   176354454),  INT32_C(   403676777),  INT32_C(  2072272096),  INT32_C(  1766446988),  INT32_C(  2109727987),  INT32_C(   675115709), -INT32_C(  1366946183),
         INT32_C(   614375566), -INT32_C(    30531180), -INT32_C(  1625932353),  INT32_C(   639277722),  INT32_C(  1703896177),  INT32_C(   115494472),  INT32_C(   976101569), -INT32_C(  1108822994) },
      { -INT32_C(   371095723),  INT32_C(  1743196328), -INT32_C(   418972083),  INT32_C(   168632472), -INT32_C(   848323196), -INT32_C(  1395437077), -INT32_C(  1595539087),  INT32_C(   190697398),
        -INT32_C(    17547690), -INT32_C(  1671046066),  INT32_C(  1921215450),  INT32_C(   276599179),  INT32_C(   433974062), -INT32_C(  1362710467),  INT32_C(  1733209265), -INT32_C(   781014149) },
      { -INT32_C(  1644329831),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   620232752),  INT32_C(           0),  INT32_C(   912286317), -INT32_C(   302571258),
        -INT32_C(  1551608908),  INT32_C(           0),  INT32_C(  1803315622),  INT32_C(  1151019934), -INT32_C(  1618688178),  INT32_C(           0), -INT32_C(   642494095),  INT32_C(           0) } },
    { UINT16_C(26602),
      {  INT32_C(   876820687),  INT32_C(  1486822868),  INT32_C(   216607631),  INT32_C(   418846523), -INT32_C(   154651600),  INT32_C(   832797155),  INT32_C(  1407000289), -INT32_C(  1078278160),
        -INT32_C(   722141697), -INT32_C(  1439919334), -INT32_C(   105507394),  INT32_C(  1544662316), -INT32_C(   984360478),  INT32_C(   888600147),  INT32_C(    76010260), -INT32_C(  1698413926) },
      {  INT32_C(  1517205567),  INT32_C(   168073803), -INT32_C(   620512593),  INT32_C(  1228346727),  INT32_C(  1091472110), -INT32_C(  1787492992),  INT32_C(  1939471832),  INT32_C(  2114805055),
         INT32_C(  1641577237), -INT32_C(   999564267), -INT32_C(    23105898),  INT32_C(  1917310595), -INT32_C(   508340639),  INT32_C(   863381851),  INT32_C(  1705381926), -INT32_C(  2082229395) },
      {  INT32_C(           0),  INT32_C(  1843348764),  INT32_C(           0),  INT32_C(   553594813),  INT32_C(           0), -INT32_C(   690134912), -INT32_C(   779130152),  INT32_C(  1355375632),
         INT32_C(   474634987), -INT32_C(  1477416670), -INT32_C(  1828036268),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1954470529), -INT32_C(   330132232),  INT32_C(           0) } },
    { UINT16_C(48174),
      {  INT32_C(  1335378916), -INT32_C(  1497551097),  INT32_C(  1954365741),  INT32_C(  1724571315),  INT32_C(   915350975), -INT32_C(    12143271), -INT32_C(    21777638),  INT32_C(   448454966),
         INT32_C(   745099813), -INT32_C(  1361893503),  INT32_C(   455299176), -INT32_C(   310252242),  INT32_C(  1814237459), -INT32_C(   513054266),  INT32_C(  1407131165),  INT32_C(   628005120) },
      {  INT32_C(  1834080235),  INT32_C(  1713054974), -INT32_C(  1568588172),  INT32_C(  1066402604), -INT32_C(   626281708),  INT32_C(   985339421),  INT32_C(   898472501),  INT32_C(   526056243),
        -INT32_C(   779310125),  INT32_C(  1144563664),  INT32_C(   317176294), -INT32_C(   799967300),  INT32_C(  1202453546),  INT32_C(  1199662610),  INT32_C(   880545537), -INT32_C(   581708278) },
      {  INT32_C(           0), -INT32_C(   905009934),  INT32_C(  1882680932),  INT32_C(  1563058116),  INT32_C(           0), -INT32_C(  1600442091),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0), -INT32_C(   995269264),  INT32_C(  1010339784), -INT32_C(   280985314), -INT32_C(   596093972),  INT32_C(           0), -INT32_C(  2059929088) } },
    { UINT16_C(57218),
      { -INT32_C(   410561873),  INT32_C(  2107665814), -INT32_C(   789291649), -INT32_C(   657711315), -INT32_C(   398467482), -INT32_C(  1560854490), -INT32_C(   931593868), -INT32_C(  1901593633),
        -INT32_C(   478859699),  INT32_C(   442570139), -INT32_C(  1595255438), -INT32_C(   612845964), -INT32_C(  1144801387),  INT32_C(  1818082039), -INT32_C(  1707813189),  INT32_C(   505994193) },
      { -INT32_C(  1526555382),  INT32_C(   633365427),  INT32_C(   767929016), -INT32_C(   167231903), -INT32_C(   307115019),  INT32_C(  1079578245), -INT32_C(  1227125275),  INT32_C(  1926562664),
         INT32_C(  1410914209), -INT32_C(   226895814), -INT32_C(   484491390),  INT32_C(  1926834045),  INT32_C(  2002750194),  INT32_C(  2126035097), -INT32_C(  1372286139),  INT32_C(   924846486) },
      {  INT32_C(           0), -INT32_C(  1072848414),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   893587560),
         INT32_C(   298633581), -INT32_C(  1602441954), -INT32_C(   791383580), -INT32_C(   569326940),  INT32_C(  1665457370),  INT32_C(           0), -INT32_C(  1618525849),  INT32_C(  2093688182) } },
    { UINT16_C(14560),
      {  INT32_C(    84941451),  INT32_C(   742691597),  INT32_C(  1347731830), -INT32_C(  1814411725),  INT32_C(  1984656318),  INT32_C(   638161393),  INT32_C(  1596956479),  INT32_C(  1654132951),
         INT32_C(   929540138), -INT32_C(  1302025413), -INT32_C(  1610434452), -INT32_C(  1137451778),  INT32_C(  1093828176),  INT32_C(  1315388175), -INT32_C(  1297180709), -INT32_C(   787200345) },
      {  INT32_C(   705264878), -INT32_C(  1797493465),  INT32_C(   590667301),  INT32_C(   182478778), -INT32_C(   179563803),  INT32_C(   692302670), -INT32_C(   237244086),  INT32_C(   650374967),
        -INT32_C(  1823421333),  INT32_C(  1579625529), -INT32_C(   998155510), -INT32_C(  1479581246), -INT32_C(  1029956748),  INT32_C(   434888910),  INT32_C(   151701201),  INT32_C(   556780981) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   319355502), -INT32_C(   493660362), -INT32_C(   467052751),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1452558724),  INT32_C(  1995680832),  INT32_C(   411280914),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(32665),
      { -INT32_C(   609496396), -INT32_C(  1305037520),  INT32_C(  1209268345),  INT32_C(  1013155743),  INT32_C(   874263113),  INT32_C(  1442508107),  INT32_C(   623096054), -INT32_C(  2052801327),
        -INT32_C(  1084207217),  INT32_C(  2121373188),  INT32_C(   767985038),  INT32_C(  1449732620), -INT32_C(  1484094116),  INT32_C(  1778156915),  INT32_C(   110042933),  INT32_C(  1804284892) },
      { -INT32_C(  2027230333),  INT32_C(   252025985),  INT32_C(   758959137),  INT32_C(  1384359670), -INT32_C(  1627845077), -INT32_C(   938936941), -INT32_C(   238119147),  INT32_C(  1297897930),
        -INT32_C(   975862204),  INT32_C(  1138022946), -INT32_C(  1670311770), -INT32_C(   487656266), -INT32_C(  1786714366), -INT32_C(   228751139), -INT32_C(   354211041), -INT32_C(   935903356) },
      { -INT32_C(  1595929060),  INT32_C(           0),  INT32_C(           0), -INT32_C(   437097270),  INT32_C(  1115516995),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1908254186),
         INT32_C(  2083847164), -INT32_C(   869968760), -INT32_C(  1832814060),  INT32_C(  1509905544), -INT32_C(  2023027016), -INT32_C(   693416889), -INT32_C(   208556437),  INT32_C(           0) } },
    { UINT16_C( 3269),
      {  INT32_C(  1642588301), -INT32_C(  1670214357),  INT32_C(   412166186), -INT32_C(  1962896630), -INT32_C(  2045518551), -INT32_C(  1296944177),  INT32_C(  1425094173), -INT32_C(  1939753217),
        -INT32_C(   890353506), -INT32_C(    10067755), -INT32_C(  1827080312), -INT32_C(  1273096053), -INT32_C(  1002819083), -INT32_C(  2122912668),  INT32_C(   584476451), -INT32_C(  1129433315) },
      {  INT32_C(  1418108031), -INT32_C(  2058031876),  INT32_C(  1863871716),  INT32_C(  2032350852), -INT32_C(   868393625),  INT32_C(  1833808714),  INT32_C(   932127514), -INT32_C(   638370470),
        -INT32_C(   701597222),  INT32_C(  1247511142),  INT32_C(  1924756462),  INT32_C(   300670121), -INT32_C(  2082658247), -INT32_C(   168809765), -INT32_C(  1473478834), -INT32_C(  1753079619) },
      { -INT32_C(   510637581),  INT32_C(           0),  INT32_C(   405372264),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   270038514),  INT32_C(  2114066598),
         INT32_C(           0),  INT32_C(           0),  INT32_C(  1773156464), -INT32_C(  2020694077),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(45209),
      { -INT32_C(   919404691),  INT32_C(    37494857),  INT32_C(  2111760018), -INT32_C(   727377673), -INT32_C(  1929477989), -INT32_C(  1593095051),  INT32_C(  2009188597), -INT32_C(   853058721),
        -INT32_C(  1533650598),  INT32_C(   195482233), -INT32_C(  1350007368),  INT32_C(   948120989),  INT32_C(   583303853), -INT32_C(  1010577202), -INT32_C(   130382440),  INT32_C(   986014176) },
      {  INT32_C(   870210490), -INT32_C(   432110291), -INT32_C(  1483356662), -INT32_C(  1595991565),  INT32_C(  1774363803),  INT32_C(   204310132), -INT32_C(   352033013), -INT32_C(  2094675511),
         INT32_C(  1387725860), -INT32_C(  1824983671), -INT32_C(  1355100740), -INT32_C(  2125456922),  INT32_C(   837489341), -INT32_C(  1556211560),  INT32_C(  1217282687),  INT32_C(   818721804) },
      { -INT32_C(  2023795662),  INT32_C(           0),  INT32_C(           0),  INT32_C(   391106933),  INT32_C(  1894853081),  INT32_C(           0),  INT32_C(           0), -INT32_C(   866167657),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1911315527),  INT32_C(  1404783184),  INT32_C(           0), -INT32_C(  2003691904) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mullo_epi32(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mullo_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_mullo_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C( 7276399494919765408),  INT64_C( 1140737814182390863),  INT64_C( 5128450863443417125), -INT64_C( 4432927000091364114),
         INT64_C( 5523916718480553175), -INT64_C( 5305339792474383536), -INT64_C( 3465699268408248643),  INT64_C( 2554832144775723089) },
      { -INT64_C( 8587281231950883958),  INT64_C(  503056571225908404), -INT64_C( 2676107359274193875),  INT64_C( 1672526017044740038),
        -INT64_C(  858936007358257324),  INT64_C( 5793375297201630230),  INT64_C( 7116596281780014432), -INT64_C( 6496815945330896847) },
      {  INT64_C( 3922742681560147904),  INT64_C( 4620791748233184140),  INT64_C(  851216417278171777), -INT64_C( 1190296736307408364),
         INT64_C( 1378929952311793548),  INT64_C( 4873209146495113440),  INT64_C( 2464055324771688416),  INT64_C( 1108359865003415425) } },
    { { -INT64_C(  196111198931815149),  INT64_C( 1487530055727677489),  INT64_C( 7913037010577523409),  INT64_C(  290486511234467089),
        -INT64_C( 5738537555279267464),  INT64_C( 5757757010771032606), -INT64_C( 5164065649718336085), -INT64_C(  231897427188481582) },
      { -INT64_C( 2482687639701217188), -INT64_C( 1927656935600975758),  INT64_C( 6690698168802081915),  INT64_C( 8297011599678436515),
        -INT64_C( 6150552353196048376),  INT64_C( 7316618671447233386),  INT64_C( 2070355623333549782), -INT64_C( 3683829302332957866) },
      { -INT64_C( 1230758827760454956), -INT64_C( 5927549697110727214),  INT64_C( 2363660759024408171), -INT64_C( 5452344992416387629),
         INT64_C( 7763018449882041280), -INT64_C( 6141822801155011988), -INT64_C( 8858409054522675470), -INT64_C( 6030529403316977012) } },
    { {  INT64_C( 8199426575332224744),  INT64_C( 1502688516282606731),  INT64_C( 1840335914152452853),  INT64_C( 2277526422145125898),
        -INT64_C( 1077193300047352172), -INT64_C( 5206089964380085394),  INT64_C( 7890217913626735778),  INT64_C(  131685676004148861) },
      { -INT64_C( 8404655765538926075), -INT64_C( 6536983639933599027), -INT64_C( 7350847537000256468),  INT64_C( 1584739371927287953),
         INT64_C( 1430699125694831411),  INT64_C( 3911633607770444447),  INT64_C( 8001983799082963798), -INT64_C( 5972217883958355462) },
      {  INT64_C( 5435074734414117512),  INT64_C(  848024873861110095),  INT64_C( 8756322826649820700),  INT64_C(  111337622553420714),
        -INT64_C( 5553129130112195716),  INT64_C( 6148045471856704850), -INT64_C( 4591956312971398036),  INT64_C( 3556536575428446994) } },
    { {  INT64_C( 1972813565672595207),  INT64_C( 4911464487690568681), -INT64_C( 6221790302362472871), -INT64_C( 8699879704935328740),
        -INT64_C(   69748521156984976),  INT64_C( 2695716415581898296), -INT64_C(  116524169027575799),  INT64_C( 6278255270268507417) },
      {  INT64_C( 7002336735367853079),  INT64_C( 4901024959436300050),  INT64_C( 9219255907483738384),  INT64_C( 8409409477347476853),
         INT64_C( 6248971275899893806),  INT64_C( 4498312274212306670),  INT64_C( 4647190125031287704), -INT64_C(  110248236998092889) },
      { -INT64_C( 1029492883668798047), -INT64_C(  227993853892002462), -INT64_C( 1074705543597941104),  INT64_C( 3154255811917036748),
         INT64_C( 4522632547863619104), -INT64_C( 8931616509133465584),  INT64_C( 8539031425778174040), -INT64_C( 7931963603058392497) } },
    { { -INT64_C( 2962508837424915895), -INT64_C( 1749246796418312372), -INT64_C( 5484431015266620712),  INT64_C( 1930127592929132794),
         INT64_C( 8649596481791532874),  INT64_C( 5681199081799748001),  INT64_C( 1897656054644325526), -INT64_C( 8628422729589781163) },
      {  INT64_C( 1473738604998813902), -INT64_C( 6299732057240869170), -INT64_C( 5993552954973846632),  INT64_C( 3851444542902614189),
        -INT64_C( 4430236962704642052), -INT64_C(  233456674969057272),  INT64_C( 6421999668870515739),  INT64_C(  205677972169224766) },
      {  INT64_C( 8031120826698845886), -INT64_C( 1234391273960569048),  INT64_C( 2834417870197225536), -INT64_C( 7235177657389754126),
        -INT64_C( 6520163573906840872),  INT64_C( 6793754019647706376),  INT64_C( 8920601512406543314),  INT64_C( 6732102160926348438) } },
    { { -INT64_C( 5360138234218018074),  INT64_C( 6187543719664657022),  INT64_C( 6315145198318424734), -INT64_C( 8799515115247507495),
        -INT64_C( 4016089049832860703),  INT64_C( 4841239465566016159), -INT64_C( 5220743037595027685),  INT64_C( 4773102437363255576) },
      {  INT64_C( 4097171309200448971),  INT64_C( 3102207258693633829), -INT64_C( 1959649905838351816), -INT64_C( 6550528698588896395),
         INT64_C( 2325993733310048117), -INT64_C( 3289325811250079459),  INT64_C( 1891993040322083217),  INT64_C( 3069211624116050968) },
      { -INT64_C( 1809998440936131998), -INT64_C( 2952638630758257098), -INT64_C( 1187552136975409520), -INT64_C( 4243860769376640723),
         INT64_C( 6566393083509890261), -INT64_C( 2468994343804640509), -INT64_C( 1442733347385935541), -INT64_C( 8405917601467368896) } },
    { {  INT64_C(  435021680658174702),  INT64_C( 8883224828139860954), -INT64_C(  142679337897283514), -INT64_C( 1525161184780487434),
         INT64_C( 7695486902644969436), -INT64_C( 3299969141588361265),  INT64_C( 2034151651194768727), -INT64_C( 7954309438129099849) },
      {  INT64_C( 1920258898138656684),  INT64_C(  324268715209360929), -INT64_C( 5235299325893919467),  INT64_C( 7560425280505454502),
        -INT64_C( 8801358150647922059), -INT64_C( 3710206641833857071), -INT64_C( 1970501592326618384), -INT64_C( 8312470538224201601) },
      {  INT64_C( 5676074521177418216), -INT64_C(  910568989285577958),  INT64_C(  409357228781118398), -INT64_C( 1964937244117261948),
        -INT64_C( 3947509227010066548), -INT64_C( 4993975677055379201), -INT64_C(  651796780195156080),  INT64_C( 7229882830951751625) } },
    { {  INT64_C( 7230292342120204398),  INT64_C( 3195977096830814935),  INT64_C( 4632167452126241319),  INT64_C( 2483208205941639380),
        -INT64_C( 4368407057809423512), -INT64_C( 3584956944779951146), -INT64_C(  932109746298463218),  INT64_C( 1574664120515333123) },
      { -INT64_C(  952668435319238888), -INT64_C( 3158636569908930124), -INT64_C( 2543489417689670240), -INT64_C(  143148825021476543),
         INT64_C( 4200391177020773437),  INT64_C( 3782391006793977827),  INT64_C( 1788959122527625758),  INT64_C( 1305210277960509241) },
      {  INT64_C( 8626278698609130576),  INT64_C( 4557386793741950508),  INT64_C( 2471011749057364832), -INT64_C( 1997869337248083500),
        -INT64_C( 7859209769517338680),  INT64_C( 8381761652651274434),  INT64_C( 8615990717482182052),  INT64_C( 9037655839035646379) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mullo_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mullo_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_mullo_epi64(a, b);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_mullo_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C(  729836585798542379), -INT64_C( 6191138873035295413), -INT64_C( 6557060457650445803), -INT64_C( 1379488973163926383),
        -INT64_C( 6602062264895586066), -INT64_C( 4908370929389535688),  INT64_C( 1441122753815371288), -INT64_C(   84331369717891631) },
      UINT8_C(165),
      { -INT64_C( 3984943729738880585), -INT64_C( 6227869380790785404), -INT64_C( 1602085284807447538),  INT64_C( 4095170091102612681),
        -INT64_C(  435893185022864800), -INT64_C( 7438814593801821873), -INT64_C(  592699238248615950),  INT64_C( 1708265820354166780) },
      {  INT64_C( 1961938639546050508), -INT64_C( 6505456588995318379), -INT64_C( 8210262703864659223), -INT64_C(  299842113384073976),
        -INT64_C( 2229995068113217754),  INT64_C( 2701517922953675301),  INT64_C( 5844347716422664187),  INT64_C(  768962031260804480) },
      {  INT64_C( 9018579314672393940), -INT64_C( 6191138873035295413), -INT64_C( 3246515954088053570), -INT64_C( 1379488973163926383),
        -INT64_C( 6602062264895586066),  INT64_C( 4119794038072307307),  INT64_C( 1441122753815371288), -INT64_C( 1922087849073612288) } },
    { { -INT64_C( 8923027931836651267), -INT64_C( 2328965856717165492), -INT64_C( 5372082743235778087),  INT64_C( 7393802729537503917),
        -INT64_C( 1234871807015675656), -INT64_C( 1456312965334157306),  INT64_C( 7779827336417654540),  INT64_C( 7264474889879116801) },
      UINT8_C( 49),
      { -INT64_C( 3431530060248269953),  INT64_C( 5577652159419403063), -INT64_C(  334823696025016103),  INT64_C( 1386769572017194100),
         INT64_C(  859851985018229383),  INT64_C( 8609981563102679065),  INT64_C( 8291173089531627637), -INT64_C( 3848961109373454355) },
      { -INT64_C( 1535771656790211888),  INT64_C( 6250036277154485415), -INT64_C( 6732746524617700578), -INT64_C( 6917606207311834894),
         INT64_C( 7383355539331475387),  INT64_C( 4845142186618752774),  INT64_C( 1462245060662831346), -INT64_C( 2130686237881045504) },
      {  INT64_C( 6326270904952640816), -INT64_C( 2328965856717165492), -INT64_C( 5372082743235778087),  INT64_C( 7393802729537503917),
        -INT64_C( 3555715552234612323), -INT64_C(  620859519121762410),  INT64_C( 7779827336417654540),  INT64_C( 7264474889879116801) } },
    { { -INT64_C( 6245115390525001424),  INT64_C( 6475558779287109492),  INT64_C( 3363426060711509366),  INT64_C( 8053117005776943032),
        -INT64_C( 8545103011484127204),  INT64_C( 4571716331621251762),  INT64_C( 8473951218183991869),  INT64_C( 2245913040179730300) },
      UINT8_C( 71),
      {  INT64_C( 4750986751040609929), -INT64_C( 1566045990555089401), -INT64_C( 2368018980219211999), -INT64_C( 7724516641146216592),
         INT64_C( 1614749434419096031), -INT64_C( 2583269879455059260),  INT64_C( 6357014390881111312), -INT64_C(  422304705664836987) },
      {  INT64_C( 8573317337294435286), -INT64_C( 8605013875480193028),  INT64_C( 4820850228060034982),  INT64_C( 5208682840351491635),
        -INT64_C( 1148373973467305239),  INT64_C( 4773883348318208831),  INT64_C( 7878123108229456212),  INT64_C(  376053991718200957) },
      {  INT64_C( 2683438101229408646),  INT64_C( 1743163939602468836), -INT64_C(  544185592354535834),  INT64_C( 8053117005776943032),
        -INT64_C( 8545103011484127204),  INT64_C( 4571716331621251762), -INT64_C( 7940183934620509888),  INT64_C( 2245913040179730300) } },
    { { -INT64_C( 3723719820607265427), -INT64_C( 8335666794035152269),  INT64_C( 6236345123360088548), -INT64_C( 5243738859421415911),
        -INT64_C(  270942164520150556),  INT64_C( 7577689572667801957), -INT64_C( 3265932468015347842),  INT64_C( 6294696869092652121) },
      UINT8_C( 27),
      { -INT64_C( 4087965510851573388), -INT64_C( 3348508743216340564),  INT64_C( 9198986566797352508),  INT64_C( 1999961793450101861),
        -INT64_C( 5485033571427391981),  INT64_C( 2232734153944345888),  INT64_C( 5344673133006587681),  INT64_C( 8946033874059665251) },
      { -INT64_C( 9164132136526733032), -INT64_C(  818343610135375965), -INT64_C( 8331996684265385425),  INT64_C( 1860477034020949996),
        -INT64_C( 9097540776900401185),  INT64_C( 4783009804660927385), -INT64_C( 4939685941056437828), -INT64_C( 6476489869700888316) },
      {  INT64_C( 2960305622537393888), -INT64_C(  906341423620111228),  INT64_C( 6236345123360088548), -INT64_C( 6438249983781857252),
        -INT64_C( 3830332445180784755),  INT64_C( 7577689572667801957), -INT64_C( 3265932468015347842),  INT64_C( 6294696869092652121) } },
    { { -INT64_C(  385157406433913271),  INT64_C( 1541856529297616400),  INT64_C( 8525714809004145863), -INT64_C( 8294335000124563082),
         INT64_C( 9100748376889346545),  INT64_C( 4716965877627597576),  INT64_C( 6029238931698446531), -INT64_C( 5523255087971948407) },
      UINT8_C( 62),
      {  INT64_C(  391579325531161868), -INT64_C( 5217491332368510772), -INT64_C( 6897008323538400106),  INT64_C( 8874930681355113847),
        -INT64_C( 3570421012977494500),  INT64_C( 5124877235586724130),  INT64_C( 1305567367413543685), -INT64_C( 5526998010162151069) },
      { -INT64_C( 6715266716883931394), -INT64_C( 5945267556116172922),  INT64_C( 3857976099592635613), -INT64_C( 5903001194568644407),
         INT64_C( 5810906626284992130), -INT64_C( 5313958558045851924),  INT64_C( 9048497686137695114),  INT64_C( 8043134187804203328) },
      { -INT64_C(  385157406433913271), -INT64_C( 3497937404388327736),  INT64_C( 1145353416295183742), -INT64_C( 1733635906982374801),
        -INT64_C( 8926391678862712264), -INT64_C( 5095569917418313896),  INT64_C( 6029238931698446531), -INT64_C( 5523255087971948407) } },
    { { -INT64_C( 5385155297311371991),  INT64_C( 6652532232454523843),  INT64_C( 8657971515397603574),  INT64_C( 2468261740822398693),
        -INT64_C( 8443587639644933549), -INT64_C( 7182545797212922843),  INT64_C( 4370864759114173432),  INT64_C( 7678442851722465396) },
      UINT8_C(  1),
      {  INT64_C( 8512738271435458468),  INT64_C( 8640922712616475001), -INT64_C( 3179248027469735169), -INT64_C( 9049889041781469672),
        -INT64_C(  239636896207064868), -INT64_C( 9106041492510750276), -INT64_C( 7959883120215363854), -INT64_C( 6593725288558267617) },
      {  INT64_C( 6778348040095530642), -INT64_C( 4011538663983350192),  INT64_C( 3647639802598998933), -INT64_C( 8294031848402610056),
        -INT64_C( 7445661539612114772), -INT64_C( 1346929951115693392),  INT64_C( 5280360410434489089), -INT64_C( 3406569441955514829) },
      { -INT64_C( 8348983762339622008),  INT64_C( 6652532232454523843),  INT64_C( 8657971515397603574),  INT64_C( 2468261740822398693),
        -INT64_C( 8443587639644933549), -INT64_C( 7182545797212922843),  INT64_C( 4370864759114173432),  INT64_C( 7678442851722465396) } },
    { { -INT64_C( 4823697095508796881),  INT64_C( 4125739499965872635),  INT64_C(  826575283049872477), -INT64_C( 6917262459686878708),
        -INT64_C( 2227551294420804276),  INT64_C( 5264275769709123998), -INT64_C(  101911591561765710),  INT64_C(  865305477929422321) },
      UINT8_C( 75),
      { -INT64_C( 3108588222747089639),  INT64_C( 2049430028868956488),  INT64_C( 6728673132763085696), -INT64_C( 2327189175316659142),
         INT64_C( 4818282024351426190),  INT64_C( 2848370248908840363), -INT64_C( 2496409664517335444),  INT64_C( 3342508341254780860) },
      {  INT64_C( 2601695723304485087), -INT64_C(  468629940379640910), -INT64_C( 3806579558338652499),  INT64_C( 5446454399227294407),
        -INT64_C( 7867999789833204687),  INT64_C( 2513407121684127322), -INT64_C( 1271248577037674623), -INT64_C( 8302409709385307632) },
      { -INT64_C( 6917413805314961209), -INT64_C( 4493462822794699760),  INT64_C(  826575283049872477), -INT64_C(  369958332524307178),
        -INT64_C( 2227551294420804276),  INT64_C( 5264275769709123998),  INT64_C( 4802102561678000236),  INT64_C(  865305477929422321) } },
    { { -INT64_C( 8923595506190980170), -INT64_C(    1040692697885596),  INT64_C( 2105972738812243492),  INT64_C( 3349658295405743765),
         INT64_C( 2087961278768525899), -INT64_C( 2990405878580681471), -INT64_C( 8103846022874067105), -INT64_C( 3634691311858146584) },
      UINT8_C(173),
      { -INT64_C( 6826751993656819117),  INT64_C( 5820948978787926647), -INT64_C( 2337691747996781493), -INT64_C( 3458431923039113384),
         INT64_C( 6866955790600726128),  INT64_C( 1560373895109255189),  INT64_C( 2548838152013607363),  INT64_C( 6811545928080015417) },
      {  INT64_C(   10510144778180830),  INT64_C( 4672200375688162806),  INT64_C( 1503393978714722799), -INT64_C( 6112138079658879776),
        -INT64_C( 8875020850941577238),  INT64_C( 1747426518331894660),  INT64_C(  116160248969036204), -INT64_C( 6666454749366163708) },
      { -INT64_C( 1458927617591817222), -INT64_C(    1040692697885596),  INT64_C( 4720741135015038213),  INT64_C( 4847006065579150592),
         INT64_C( 2087961278768525899),  INT64_C(  158011998963700180), -INT64_C( 8103846022874067105), -INT64_C( 8817779993455759388) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_mullo_epi64(src, k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_mullo_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_mask_mullo_epi64(src, k, a, b);

    easysimd_test_x86_write_i64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_mullo_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(166),
      {  INT64_C( 1566578716985433741),  INT64_C( 7705851137453423901), -INT64_C( 3019567392384255134), -INT64_C( 1609474621163921108),
        -INT64_C(  234563356794797155), -INT64_C( 4442756635622395610),  INT64_C( 8209309251973934499),  INT64_C(  953106639382409716) },
      { -INT64_C(  704121953975916960),  INT64_C( 4415565693540725285), -INT64_C( 3755648224986889429), -INT64_C( 3300762003670542925),
        -INT64_C( 9204430357908035715), -INT64_C( 7956649539714799427), -INT64_C( 2489678971164038960),  INT64_C( 7024825851942309432) },
      {  INT64_C(                   0), -INT64_C( 5461253484368560335), -INT64_C( 3410461064160682122),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C( 4679105397226751730),  INT64_C(                   0), -INT64_C( 6572085854105352864) } },
    { UINT8_C( 24),
      {  INT64_C( 1916269490336935566),  INT64_C( 7716937217037364986), -INT64_C( 4670951679614140654), -INT64_C( 2108523189441931405),
         INT64_C( 2215862433530785971),  INT64_C( 7219409096622358593), -INT64_C( 6112081542344227523), -INT64_C( 8532899807024394146) },
      { -INT64_C( 1909263852232617667), -INT64_C( 6780773538729368017),  INT64_C( 2558674358714187679),  INT64_C( 1975791466564735500),
        -INT64_C( 5240503829199480388),  INT64_C( 7631382318899851235),  INT64_C( 4559264822167926620),  INT64_C( 4796206637257923477) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 5514898441174282084),
        -INT64_C(  336860443627372940),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 38),
      { -INT64_C( 2887821213782072820),  INT64_C( 1303521731358569373), -INT64_C( 7304129968718137014), -INT64_C( 5667670569884051802),
         INT64_C(  632094305195525637), -INT64_C( 5042102554117052711),  INT64_C( 8442959366172723987), -INT64_C( 5416140749615379328) },
      {  INT64_C( 1115598816253480967),  INT64_C( 3168010862585921229), -INT64_C( 6153632148122452140),  INT64_C( 5302889892472176948),
         INT64_C( 7889522184886630711), -INT64_C( 2398269560602328972), -INT64_C( 8055078962222435231), -INT64_C( 2060385095143717155) },
      {  INT64_C(                   0),  INT64_C( 2313095729384348345),  INT64_C( 5700842769875796552),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C( 2829775093527143852),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 68),
      {  INT64_C( 5678848701379137828), -INT64_C( 8718767583692039172), -INT64_C( 4092800362142790727), -INT64_C( 2363167439758011674),
         INT64_C( 7477392013956553128),  INT64_C( 1080307920459439460),  INT64_C( 4186824092514771345), -INT64_C( 8177587054173319447) },
      {  INT64_C( 2975413455097805944), -INT64_C( 1672350216245718937), -INT64_C( 5977026826604958337),  INT64_C( 1033809858586032100),
        -INT64_C( 8106976139577041520),  INT64_C( 4308561369521397388),  INT64_C( 4590042522978223682),  INT64_C( 7635961851968816211) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 5924809946151835079),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C( 7034653389564120418),  INT64_C(                   0) } },
    { UINT8_C(191),
      {  INT64_C( 6607473608306966249),  INT64_C( 7356721666683606465), -INT64_C( 2895812556834660510), -INT64_C( 7265404944248751606),
         INT64_C( 5363632713817461719),  INT64_C( 4453559596511954696), -INT64_C( 7277382071545589929), -INT64_C( 6876376738515406779) },
      { -INT64_C( 3595803490683432122), -INT64_C( 3908275161702975061),  INT64_C( 5602464777289039252),  INT64_C( 4687921732731062181),
        -INT64_C( 8016641644615908822), -INT64_C( 8169571527686687133),  INT64_C( 3219355234391528023), -INT64_C(  903630572388421319) },
      { -INT64_C( 6422214943003132234),  INT64_C( 4006792853618032875), -INT64_C( 8236592363162632536), -INT64_C(  856508707380875662),
        -INT64_C( 1417812232340361402),  INT64_C( 1302239434754091032),  INT64_C(                   0), -INT64_C( 6254706188380195747) } },
    { UINT8_C(106),
      { -INT64_C( 7416337167723535227),  INT64_C( 1884512935793191158),  INT64_C( 2702666871544579797), -INT64_C( 5730302579591760537),
        -INT64_C( 6989179079071119164), -INT64_C( 8215525899413514917), -INT64_C( 3353678233373139684),  INT64_C(  569522023120790028) },
      {  INT64_C( 1470831796092356130), -INT64_C( 8409078747231758330), -INT64_C( 8805574690864422512),  INT64_C( 2953696180912827826),
        -INT64_C( 1966282080309252887),  INT64_C( 3471974736456756533),  INT64_C(  868623366572673296),  INT64_C( 6256905525003405733) },
      {  INT64_C(                   0),  INT64_C( 5873809776532587972),  INT64_C(                   0),  INT64_C(  736990109261544606),
         INT64_C(                   0),  INT64_C( 6432388083619883735), -INT64_C( 5325504837829014080),  INT64_C(                   0) } },
    { UINT8_C(107),
      {  INT64_C( 6654117580162791623),  INT64_C( 3804745822285659343),  INT64_C( 3311721752833119832),  INT64_C( 4252787232216817819),
         INT64_C( 5256746922930032104),  INT64_C( 8291921980585078707), -INT64_C( 3117977234669592791),  INT64_C( 3101135453956166489) },
      { -INT64_C( 3206082314010203969), -INT64_C( 8920571642360264796),  INT64_C(  607928429799667298),  INT64_C( 1048457245795671326),
        -INT64_C( 6396862978288366010),  INT64_C( 1093700355183077966),  INT64_C( 2830259517850861200),  INT64_C( 4779469273511304695) },
      {  INT64_C( 2549886014622561401),  INT64_C(  848787660662250908),  INT64_C(                   0), -INT64_C( 8793923934948056790),
         INT64_C(                   0),  INT64_C( 8130508531397535370),  INT64_C( 6297203244015318288),  INT64_C(                   0) } },
    { UINT8_C(100),
      { -INT64_C( 3769213952439375717), -INT64_C( 4206213037797385610),  INT64_C( 4919326242153357362),  INT64_C( 8860976163628026750),
        -INT64_C( 1957220384749297878),  INT64_C(  591262195031232232),  INT64_C( 8566297254994182773), -INT64_C( 3639468549698084572) },
      { -INT64_C( 3335849104720241823), -INT64_C( 5487226857826768388), -INT64_C( 1376921948666253706),  INT64_C( 5891112868269583953),
         INT64_C( 4865530575896811978), -INT64_C( 5660762795556878721),  INT64_C( 4742576215768940317), -INT64_C(  345942427750186255) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 4713441053421448436),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(  601379350474036504),  INT64_C( 3639142549259653697),  INT64_C(                   0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_mullo_epi64(k, a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_mullo_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_maskz_mullo_epi64(k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mullox_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C( 7276399494919765408),  INT64_C( 1140737814182390863),  INT64_C( 5128450863443417125), -INT64_C( 4432927000091364114),
         INT64_C( 5523916718480553175), -INT64_C( 5305339792474383536), -INT64_C( 3465699268408248643),  INT64_C( 2554832144775723089) },
      { -INT64_C( 8587281231950883958),  INT64_C(  503056571225908404), -INT64_C( 2676107359274193875),  INT64_C( 1672526017044740038),
        -INT64_C(  858936007358257324),  INT64_C( 5793375297201630230),  INT64_C( 7116596281780014432), -INT64_C( 6496815945330896847) },
      {  INT64_C( 3922742681560147904),  INT64_C( 4620791748233184140),  INT64_C(  851216417278171777), -INT64_C( 1190296736307408364),
         INT64_C( 1378929952311793548),  INT64_C( 4873209146495113440),  INT64_C( 2464055324771688416),  INT64_C( 1108359865003415425) } },
    { { -INT64_C(  196111198931815149),  INT64_C( 1487530055727677489),  INT64_C( 7913037010577523409),  INT64_C(  290486511234467089),
        -INT64_C( 5738537555279267464),  INT64_C( 5757757010771032606), -INT64_C( 5164065649718336085), -INT64_C(  231897427188481582) },
      { -INT64_C( 2482687639701217188), -INT64_C( 1927656935600975758),  INT64_C( 6690698168802081915),  INT64_C( 8297011599678436515),
        -INT64_C( 6150552353196048376),  INT64_C( 7316618671447233386),  INT64_C( 2070355623333549782), -INT64_C( 3683829302332957866) },
      { -INT64_C( 1230758827760454956), -INT64_C( 5927549697110727214),  INT64_C( 2363660759024408171), -INT64_C( 5452344992416387629),
         INT64_C( 7763018449882041280), -INT64_C( 6141822801155011988), -INT64_C( 8858409054522675470), -INT64_C( 6030529403316977012) } },
    { {  INT64_C( 8199426575332224744),  INT64_C( 1502688516282606731),  INT64_C( 1840335914152452853),  INT64_C( 2277526422145125898),
        -INT64_C( 1077193300047352172), -INT64_C( 5206089964380085394),  INT64_C( 7890217913626735778),  INT64_C(  131685676004148861) },
      { -INT64_C( 8404655765538926075), -INT64_C( 6536983639933599027), -INT64_C( 7350847537000256468),  INT64_C( 1584739371927287953),
         INT64_C( 1430699125694831411),  INT64_C( 3911633607770444447),  INT64_C( 8001983799082963798), -INT64_C( 5972217883958355462) },
      {  INT64_C( 5435074734414117512),  INT64_C(  848024873861110095),  INT64_C( 8756322826649820700),  INT64_C(  111337622553420714),
        -INT64_C( 5553129130112195716),  INT64_C( 6148045471856704850), -INT64_C( 4591956312971398036),  INT64_C( 3556536575428446994) } },
    { {  INT64_C( 1972813565672595207),  INT64_C( 4911464487690568681), -INT64_C( 6221790302362472871), -INT64_C( 8699879704935328740),
        -INT64_C(   69748521156984976),  INT64_C( 2695716415581898296), -INT64_C(  116524169027575799),  INT64_C( 6278255270268507417) },
      {  INT64_C( 7002336735367853079),  INT64_C( 4901024959436300050),  INT64_C( 9219255907483738384),  INT64_C( 8409409477347476853),
         INT64_C( 6248971275899893806),  INT64_C( 4498312274212306670),  INT64_C( 4647190125031287704), -INT64_C(  110248236998092889) },
      { -INT64_C( 1029492883668798047), -INT64_C(  227993853892002462), -INT64_C( 1074705543597941104),  INT64_C( 3154255811917036748),
         INT64_C( 4522632547863619104), -INT64_C( 8931616509133465584),  INT64_C( 8539031425778174040), -INT64_C( 7931963603058392497) } },
    { { -INT64_C( 2962508837424915895), -INT64_C( 1749246796418312372), -INT64_C( 5484431015266620712),  INT64_C( 1930127592929132794),
         INT64_C( 8649596481791532874),  INT64_C( 5681199081799748001),  INT64_C( 1897656054644325526), -INT64_C( 8628422729589781163) },
      {  INT64_C( 1473738604998813902), -INT64_C( 6299732057240869170), -INT64_C( 5993552954973846632),  INT64_C( 3851444542902614189),
        -INT64_C( 4430236962704642052), -INT64_C(  233456674969057272),  INT64_C( 6421999668870515739),  INT64_C(  205677972169224766) },
      {  INT64_C( 8031120826698845886), -INT64_C( 1234391273960569048),  INT64_C( 2834417870197225536), -INT64_C( 7235177657389754126),
        -INT64_C( 6520163573906840872),  INT64_C( 6793754019647706376),  INT64_C( 8920601512406543314),  INT64_C( 6732102160926348438) } },
    { { -INT64_C( 5360138234218018074),  INT64_C( 6187543719664657022),  INT64_C( 6315145198318424734), -INT64_C( 8799515115247507495),
        -INT64_C( 4016089049832860703),  INT64_C( 4841239465566016159), -INT64_C( 5220743037595027685),  INT64_C( 4773102437363255576) },
      {  INT64_C( 4097171309200448971),  INT64_C( 3102207258693633829), -INT64_C( 1959649905838351816), -INT64_C( 6550528698588896395),
         INT64_C( 2325993733310048117), -INT64_C( 3289325811250079459),  INT64_C( 1891993040322083217),  INT64_C( 3069211624116050968) },
      { -INT64_C( 1809998440936131998), -INT64_C( 2952638630758257098), -INT64_C( 1187552136975409520), -INT64_C( 4243860769376640723),
         INT64_C( 6566393083509890261), -INT64_C( 2468994343804640509), -INT64_C( 1442733347385935541), -INT64_C( 8405917601467368896) } },
    { {  INT64_C(  435021680658174702),  INT64_C( 8883224828139860954), -INT64_C(  142679337897283514), -INT64_C( 1525161184780487434),
         INT64_C( 7695486902644969436), -INT64_C( 3299969141588361265),  INT64_C( 2034151651194768727), -INT64_C( 7954309438129099849) },
      {  INT64_C( 1920258898138656684),  INT64_C(  324268715209360929), -INT64_C( 5235299325893919467),  INT64_C( 7560425280505454502),
        -INT64_C( 8801358150647922059), -INT64_C( 3710206641833857071), -INT64_C( 1970501592326618384), -INT64_C( 8312470538224201601) },
      {  INT64_C( 5676074521177418216), -INT64_C(  910568989285577958),  INT64_C(  409357228781118398), -INT64_C( 1964937244117261948),
        -INT64_C( 3947509227010066548), -INT64_C( 4993975677055379201), -INT64_C(  651796780195156080),  INT64_C( 7229882830951751625) } },
    { {  INT64_C( 7230292342120204398),  INT64_C( 3195977096830814935),  INT64_C( 4632167452126241319),  INT64_C( 2483208205941639380),
        -INT64_C( 4368407057809423512), -INT64_C( 3584956944779951146), -INT64_C(  932109746298463218),  INT64_C( 1574664120515333123) },
      { -INT64_C(  952668435319238888), -INT64_C( 3158636569908930124), -INT64_C( 2543489417689670240), -INT64_C(  143148825021476543),
         INT64_C( 4200391177020773437),  INT64_C( 3782391006793977827),  INT64_C( 1788959122527625758),  INT64_C( 1305210277960509241) },
      {  INT64_C( 8626278698609130576),  INT64_C( 4557386793741950508),  INT64_C( 2471011749057364832), -INT64_C( 1997869337248083500),
        -INT64_C( 7859209769517338680),  INT64_C( 8381761652651274434),  INT64_C( 8615990717482182052),  INT64_C( 9037655839035646379) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mullox_epi64(a, b);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mullox_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_mullox_epi64(a, b);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mullo_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mullo_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mullo_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mullo_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mullo_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_mullo_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_mullo_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mullo_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mullo_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mullo_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mullo_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_mullo_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_mullo_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mullo_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mullo_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mullo_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mullo_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mullo_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_mullo_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_mullo_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mullox_epi64)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
