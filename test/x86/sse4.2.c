/* Copyright (c) 2017 Evan Nemerson <evan@nemerson.com>
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
 */

#define EASYSIMD_TESTS_CURRENT_ISAX sse4_2
#include <test/x86/test-sse2.h>
#include <easysimd/x86/sse4.2.h>

static int
test_easysimd_mm_cmpestrs_8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    int la;
    easysimd__m128i b;
    int lb;
    int r;
  } test_vec[8] = {
    { easysimd_mm_set_epi8(INT8_C(-105), INT8_C(-116), INT8_C( -45), INT8_C(-102),
                        INT8_C(  -3), INT8_C(  92), INT8_C( -99), INT8_C( 100),
                        INT8_C(  30), INT8_C(-115), INT8_C(  82), INT8_C(  84),
                        INT8_C(-106), INT8_C(  66), INT8_C(-107), INT8_C( 116)),
      0 ,
      easysimd_mm_set_epi8(INT8_C( -89), INT8_C(  65), INT8_C(  68), INT8_C( -29),
                        INT8_C(-101), INT8_C( 113), INT8_C( -11), INT8_C(  53),
                        INT8_C(  -5), INT8_C( -76), INT8_C(  28), INT8_C(-120),
                        INT8_C(  64), INT8_C(  43), INT8_C(-127), INT8_C( -44)),
      2 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C( 103), INT8_C(  89), INT8_C( 106), INT8_C( -90),
                        INT8_C(  18), INT8_C(  23), INT8_C( 117), INT8_C(   6),
                        INT8_C( -91), INT8_C( -40), INT8_C( 108), INT8_C(-127),
                        INT8_C( -29), INT8_C( -39), INT8_C(  49), INT8_C( -85)),
      5 ,
      easysimd_mm_set_epi8(INT8_C(-104), INT8_C( 100), INT8_C( -73), INT8_C( -23),
                        INT8_C( -48), INT8_C(  87), INT8_C(-118), INT8_C(  66),
                        INT8_C( -75), INT8_C(  35), INT8_C(  -1), INT8_C( 111),
                        INT8_C( -30), INT8_C(  -6), INT8_C(  10), INT8_C(  91)),
      10 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C(  84), INT8_C(  21), INT8_C(  91), INT8_C( -41),
                        INT8_C(  25), INT8_C( -24), INT8_C(  93), INT8_C(-124),
                        INT8_C( -97), INT8_C( -88), INT8_C( 113), INT8_C(  85),
                        INT8_C(  42), INT8_C( -93), INT8_C( -37), INT8_C( -18)),
      8 ,
      easysimd_mm_set_epi8(INT8_C( 117), INT8_C( -42), INT8_C(-112), INT8_C( -67),
                        INT8_C(  -7), INT8_C( -85), INT8_C(  -4), INT8_C( 125),
                        INT8_C(-127), INT8_C( -75), INT8_C(-125), INT8_C( 109),
                        INT8_C(  50), INT8_C( -16), INT8_C(  22), INT8_C(  86)),
      12 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C( 109), INT8_C(  78), INT8_C(  15), INT8_C( 113),
                        INT8_C(-118), INT8_C( -55), INT8_C(-119), INT8_C(  -4),
                        INT8_C(  29), INT8_C(  32), INT8_C(-107), INT8_C(-117),
                        INT8_C(  79), INT8_C(  29), INT8_C( 126), INT8_C( -75)),
      16 ,
      easysimd_mm_set_epi8(INT8_C(  -7), INT8_C(  48), INT8_C( 112), INT8_C(  -3),
                        INT8_C(  35), INT8_C( -21), INT8_C( -53), INT8_C(-114),
                        INT8_C( -78), INT8_C(  -5), INT8_C( -11), INT8_C(  91),
                        INT8_C(  53), INT8_C( -34), INT8_C( -19), INT8_C(  11)),
      8 ,
      0 },
    { easysimd_mm_set_epi8(INT8_C(  39), INT8_C(  98), INT8_C( -40), INT8_C( -94),
                        INT8_C( -37), INT8_C( -39), INT8_C(  -6), INT8_C( -18),
                        INT8_C( -44), INT8_C( 119), INT8_C( -96), INT8_C(  81),
                        INT8_C(-117), INT8_C(-126), INT8_C(  94), INT8_C( -52)),
      0 ,
      easysimd_mm_set_epi8(INT8_C(  52), INT8_C( -46), INT8_C(  -6), INT8_C( -85),
                        INT8_C(  63), INT8_C(  85), INT8_C( -29), INT8_C( -39),
                        INT8_C( -42), INT8_C(  92), INT8_C( -15), INT8_C(  -6),
                        INT8_C( -75), INT8_C( -86), INT8_C( -68), INT8_C( 108)),
      3 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C(  60), INT8_C( -84), INT8_C(  55), INT8_C(  82),
                        INT8_C( -32), INT8_C( -86), INT8_C( -19), INT8_C(   6),
                        INT8_C( -73), INT8_C( -96), INT8_C(  56), INT8_C(-116),
                        INT8_C(  40), INT8_C( -91), INT8_C( -58), INT8_C( -53)),
      5 ,
      easysimd_mm_set_epi8(INT8_C(-125), INT8_C(-121), INT8_C(  94), INT8_C( -81),
                        INT8_C(  51), INT8_C( -18), INT8_C(  57), INT8_C( 114),
                        INT8_C(  65), INT8_C(  21), INT8_C(   1), INT8_C( 122),
                        INT8_C( -29), INT8_C( -17), INT8_C( 114), INT8_C(  17)),
      6 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C(   7), INT8_C(-112), INT8_C(-109), INT8_C(  25),
                        INT8_C(  65), INT8_C(   3), INT8_C(  18), INT8_C( -17),
                        INT8_C(-117), INT8_C( -64), INT8_C( 123), INT8_C( 112),
                        INT8_C( -54), INT8_C( -32), INT8_C( -28), INT8_C( -54)),
      2 ,
      easysimd_mm_set_epi8(INT8_C(  20), INT8_C( -94), INT8_C( -95), INT8_C( -11),
                        INT8_C( -10), INT8_C(  45), INT8_C( -14), INT8_C(-103),
                        INT8_C(-109), INT8_C(-101), INT8_C( 112), INT8_C(  -4),
                        INT8_C(  62), INT8_C(-110), INT8_C( 100), INT8_C(  78)),
      14 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C(  94), INT8_C( -96), INT8_C( -41), INT8_C(-127),
                        INT8_C( 109), INT8_C( -92), INT8_C(  60), INT8_C(  85),
                        INT8_C( -80), INT8_C( -69), INT8_C( -10), INT8_C( 113),
                        INT8_C( -86), INT8_C(  12), INT8_C( -11), INT8_C(  93)),
      0 ,
      easysimd_mm_set_epi8(INT8_C(  -1), INT8_C( -87), INT8_C( -78), INT8_C(  26),
                        INT8_C(  30), INT8_C( 110), INT8_C( -36), INT8_C(  70),
                        INT8_C(-126), INT8_C( -29), INT8_C( -65), INT8_C( -41),
                        INT8_C( -71), INT8_C(   1), INT8_C( 121), INT8_C(-119)),
      10 ,
      1 }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int r = easysimd_mm_cmpestrs(test_vec[i].a, test_vec[i].la, test_vec[i].b, test_vec[i].lb, 0);
    easysimd_assert_equal_i(r, test_vec[i].r);
  }
  return 0;
}

static int
test_easysimd_mm_cmpestrs_16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    int la;
    easysimd__m128i b;
    int lb;
    int r;
  } test_vec[8] = {
    { easysimd_mm_set_epi16(INT16_C(-26740), INT16_C(-11366), INT16_C(  -676), INT16_C(-25244),
                         INT16_C(  7821), INT16_C( 21076), INT16_C(-27070), INT16_C(-27276)),
      6 ,
      easysimd_mm_set_epi16(INT16_C(-22719), INT16_C( 17635), INT16_C(-25743), INT16_C( -2763),
                         INT16_C( -1100), INT16_C(  7304), INT16_C( 16427), INT16_C(-32300)),
      7 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 26457), INT16_C( 27302), INT16_C(  4631), INT16_C( 29958),
                         INT16_C(-23080), INT16_C( 27777), INT16_C( -7207), INT16_C( 12715)),
      5 ,
      easysimd_mm_set_epi16(INT16_C(-26524), INT16_C(-18455), INT16_C(-12201), INT16_C(-30142),
                         INT16_C(-19165), INT16_C(  -145), INT16_C( -7430), INT16_C(  2651)),
      7 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 21525), INT16_C( 23511), INT16_C(  6632), INT16_C( 23940),
                         INT16_C(-24664), INT16_C( 29013), INT16_C( 10915), INT16_C( -9234)),
      5 ,
      easysimd_mm_set_epi16(INT16_C( 30166), INT16_C(-28483), INT16_C( -1621), INT16_C(  -899),
                         INT16_C(-32331), INT16_C(-31891), INT16_C( 13040), INT16_C(  5718)),
      6 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 27982), INT16_C(  3953), INT16_C(-30007), INT16_C(-30212),
                         INT16_C(  7456), INT16_C(-27253), INT16_C( 20253), INT16_C( 32437)),
      7 ,
      easysimd_mm_set_epi16(INT16_C( -1744), INT16_C( 28925), INT16_C(  9195), INT16_C(-13426),
                         INT16_C(-19717), INT16_C( -2725), INT16_C( 13790), INT16_C( -4853)),
      5 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 10082), INT16_C(-10078), INT16_C( -9255), INT16_C( -1298),
                         INT16_C(-11145), INT16_C(-24495), INT16_C(-29822), INT16_C( 24268)),
      2 ,
      easysimd_mm_set_epi16(INT16_C( 13522), INT16_C( -1365), INT16_C( 16213), INT16_C( -7207),
                         INT16_C(-10660), INT16_C( -3590), INT16_C(-19030), INT16_C(-17300)),
      7 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 15532), INT16_C( 14162), INT16_C( -8022), INT16_C( -4858),
                         INT16_C(-18528), INT16_C( 14476), INT16_C( 10405), INT16_C(-14645)),
      7 ,
      easysimd_mm_set_epi16(INT16_C(-31865), INT16_C( 24239), INT16_C( 13294), INT16_C( 14706),
                         INT16_C( 16661), INT16_C(   378), INT16_C( -7185), INT16_C( 29201)),
      4 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C(  1936), INT16_C(-27879), INT16_C( 16643), INT16_C(  4847),
                         INT16_C(-29760), INT16_C( 31600), INT16_C(-13600), INT16_C( -6966)),
      7 ,
      easysimd_mm_set_epi16(INT16_C(  5282), INT16_C(-24075), INT16_C( -2515), INT16_C( -3431),
                         INT16_C(-27749), INT16_C( 28924), INT16_C( 16018), INT16_C( 25678)),
      0 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 24224), INT16_C(-10367), INT16_C( 28068), INT16_C( 15445),
                         INT16_C(-20293), INT16_C( -2447), INT16_C(-22004), INT16_C( -2723)),
      7 ,
      easysimd_mm_set_epi16(INT16_C(   -87), INT16_C(-19942), INT16_C(  7790), INT16_C( -9146),
                         INT16_C(-32029), INT16_C(-16425), INT16_C(-18175), INT16_C( 31113)),
      2 ,
      1 }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int r = easysimd_mm_cmpestrs(test_vec[i].a, test_vec[i].la, test_vec[i].b, test_vec[i].lb, 1);
    easysimd_assert_equal_i(r, test_vec[i].r);
  }
  return 0;
}

static int
test_easysimd_mm_cmpestrz_8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    int la;
    easysimd__m128i b;
    int lb;
    int r;
  } test_vec[8] = {
    { easysimd_mm_set_epi8(INT8_C(  91), INT8_C(  17), INT8_C( -35), INT8_C( -83),
                        INT8_C(  65), INT8_C( -69), INT8_C( -33), INT8_C(  -2),
                        INT8_C( -59), INT8_C( -56), INT8_C( -20), INT8_C(-124),
                        INT8_C( -68), INT8_C( -33), INT8_C( -98), INT8_C( 119)),
      9 ,
      easysimd_mm_set_epi8(INT8_C( -24), INT8_C(  33), INT8_C(  95), INT8_C(   8),
                        INT8_C(  67), INT8_C( -46), INT8_C( 123), INT8_C( -89),
                        INT8_C( -36), INT8_C(  19), INT8_C( -12), INT8_C( 108),
                        INT8_C(  70), INT8_C( -86), INT8_C( 125), INT8_C(  63)),
      9 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C(  31), INT8_C( -36), INT8_C(  70), INT8_C( -37),
                        INT8_C( 120), INT8_C(  70), INT8_C(  10), INT8_C(  73),
                        INT8_C(  94), INT8_C( -22), INT8_C( 117), INT8_C(-123),
                        INT8_C( -97), INT8_C( -97), INT8_C(  94), INT8_C( -19)),
      15 ,
      easysimd_mm_set_epi8(INT8_C(-111), INT8_C(  66), INT8_C( -59), INT8_C(  54),
                        INT8_C( 102), INT8_C(-108), INT8_C(-128), INT8_C(-104),
                        INT8_C(  81), INT8_C(  46), INT8_C(-110), INT8_C(  86),
                        INT8_C(  82), INT8_C(  23), INT8_C( -59), INT8_C(  19)),
      1 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C( 100), INT8_C(  86), INT8_C(  40), INT8_C( -10),
                        INT8_C( -78), INT8_C(  38), INT8_C(  31), INT8_C(  81),
                        INT8_C(-107), INT8_C( 114), INT8_C( 112), INT8_C(  93),
                        INT8_C(-101), INT8_C(  10), INT8_C(   0), INT8_C(-128)),
      6 ,
      easysimd_mm_set_epi8(INT8_C( -95), INT8_C(  81), INT8_C( -72), INT8_C( -74),
                        INT8_C( -66), INT8_C(-106), INT8_C(  76), INT8_C( -42),
                        INT8_C(-123), INT8_C( -44), INT8_C(-103), INT8_C(  44),
                        INT8_C( -40), INT8_C( 125), INT8_C( -32), INT8_C(-115)),
      6 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C(  40), INT8_C( -63), INT8_C(  76), INT8_C(  45),
                        INT8_C(-113), INT8_C( -94), INT8_C(  -5), INT8_C( -14),
                        INT8_C( -18), INT8_C(  63), INT8_C( -52), INT8_C( -78),
                        INT8_C(-108), INT8_C(  41), INT8_C(   7), INT8_C(  43)),
      0 ,
      easysimd_mm_set_epi8(INT8_C( -66), INT8_C(  82), INT8_C(  59), INT8_C(  48),
                        INT8_C( 110), INT8_C(  49), INT8_C(  62), INT8_C( -91),
                        INT8_C( -57), INT8_C(  18), INT8_C(  30), INT8_C(  38),
                        INT8_C(  -3), INT8_C( -35), INT8_C(  -6), INT8_C( -54)),
      1 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C(  76), INT8_C(  37), INT8_C( -49), INT8_C( -67),
                        INT8_C(  68), INT8_C(-123), INT8_C(  61), INT8_C( -77),
                        INT8_C(  82), INT8_C(  19), INT8_C(  13), INT8_C( -91),
                        INT8_C( -17), INT8_C( 115), INT8_C( -42), INT8_C(-127)),
      7 ,
      easysimd_mm_set_epi8(INT8_C( -99), INT8_C(  -9), INT8_C( -89), INT8_C(  91),
                        INT8_C(-125), INT8_C( -63), INT8_C(  83), INT8_C(  47),
                        INT8_C(  61), INT8_C(-124), INT8_C( -87), INT8_C(  -5),
                        INT8_C(  94), INT8_C( -25), INT8_C( -16), INT8_C( -76)),
      6 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C( -34), INT8_C( -22), INT8_C( -14), INT8_C(  -6),
                        INT8_C( -18), INT8_C(  91), INT8_C(  -8), INT8_C( 121),
                        INT8_C( 119), INT8_C( 123), INT8_C(  80), INT8_C( 126),
                        INT8_C( -31), INT8_C( -48), INT8_C(  62), INT8_C( -34)),
      11 ,
      easysimd_mm_set_epi8(INT8_C(  31), INT8_C( -81), INT8_C( -83), INT8_C(  83),
                        INT8_C( -41), INT8_C( 100), INT8_C(   3), INT8_C(-110),
                        INT8_C( 111), INT8_C(-115), INT8_C( -38), INT8_C( 116),
                        INT8_C(  30), INT8_C(  34), INT8_C( 109), INT8_C(  42)),
      0 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C( -33), INT8_C(-111), INT8_C( -19), INT8_C(-122),
                        INT8_C( -36), INT8_C( -20), INT8_C(  35), INT8_C(  47),
                        INT8_C(-115), INT8_C( -67), INT8_C(   0), INT8_C( -15),
                        INT8_C( -72), INT8_C( -50), INT8_C( -50), INT8_C( -72)),
      8 ,
      easysimd_mm_set_epi8(INT8_C(-110), INT8_C(-118), INT8_C(  33), INT8_C(  44),
                        INT8_C(  69), INT8_C( -27), INT8_C( -37), INT8_C(  -9),
                        INT8_C(  64), INT8_C( -92), INT8_C(  60), INT8_C( 108),
                        INT8_C( 106), INT8_C(  83), INT8_C( -30), INT8_C(  83)),
      2 ,
      1 },
    { easysimd_mm_set_epi8(INT8_C(  77), INT8_C(-108), INT8_C(  64), INT8_C(  98),
                        INT8_C( -64), INT8_C(  49), INT8_C( -82), INT8_C(  37),
                        INT8_C(  71), INT8_C(  88), INT8_C(-109), INT8_C( -84),
                        INT8_C( 109), INT8_C( -36), INT8_C(  -4), INT8_C( -89)),
      3 ,
      easysimd_mm_set_epi8(INT8_C( -71), INT8_C( -17), INT8_C( -84), INT8_C( 102),
                        INT8_C( 127), INT8_C(  91), INT8_C( -22), INT8_C(  87),
                        INT8_C(   2), INT8_C(-127), INT8_C( -31), INT8_C(-119),
                        INT8_C(  31), INT8_C(  -5), INT8_C( 114), INT8_C( -61)),
      6 ,
      1 }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int r = easysimd_mm_cmpestrz(test_vec[i].a, test_vec[i].la, test_vec[i].b, test_vec[i].lb, 0);
    easysimd_assert_equal_i(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_cmpestrz_16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    int la;
    easysimd__m128i b;
    int lb;
    int r;
  } test_vec[8] = {
    { easysimd_mm_set_epi16(INT16_C( 23313), INT16_C( -8787), INT16_C( 16827), INT16_C( -8194),
                         INT16_C(-14904), INT16_C( -4988), INT16_C(-17185), INT16_C(-24969)),
      8 ,
      easysimd_mm_set_epi16(INT16_C( -6111), INT16_C( 24328), INT16_C( 17362), INT16_C( 31655),
                         INT16_C( -9197), INT16_C( -2964), INT16_C( 18090), INT16_C( 32063)),
      5 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C(  8156), INT16_C( 18139), INT16_C( 30790), INT16_C(  2633),
                         INT16_C( 24298), INT16_C( 30085), INT16_C(-24673), INT16_C( 24301)),
      0 ,
      easysimd_mm_set_epi16(INT16_C(-28350), INT16_C(-15050), INT16_C( 26260), INT16_C(-32616),
                         INT16_C( 20782), INT16_C(-28074), INT16_C( 21015), INT16_C(-15085)),
      2 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 25686), INT16_C( 10486), INT16_C(-19930), INT16_C(  8017),
                         INT16_C(-27278), INT16_C( 28765), INT16_C(-25846), INT16_C(   128)),
      8 ,
      easysimd_mm_set_epi16(INT16_C(-24239), INT16_C(-18250), INT16_C(-16746), INT16_C( 19670),
                         INT16_C(-31276), INT16_C(-26324), INT16_C(-10115), INT16_C( -8051)),
      1 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 10433), INT16_C( 19501), INT16_C(-28766), INT16_C( -1038),
                         INT16_C( -4545), INT16_C(-13134), INT16_C(-27607), INT16_C(  1835)),
      4 ,
      easysimd_mm_set_epi16(INT16_C(-16814), INT16_C( 15152), INT16_C( 28209), INT16_C( 16037),
                         INT16_C(-14574), INT16_C(  7718), INT16_C(  -547), INT16_C( -1334)),
      3 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 19493), INT16_C(-12355), INT16_C( 17541), INT16_C( 15795),
                         INT16_C( 21011), INT16_C(  3493), INT16_C( -4237), INT16_C(-10623)),
      2 ,
      easysimd_mm_set_epi16(INT16_C(-25097), INT16_C(-22693), INT16_C(-31807), INT16_C( 21295),
                         INT16_C( 15748), INT16_C(-22021), INT16_C( 24295), INT16_C( -3916)),
      3 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( -8470), INT16_C( -3334), INT16_C( -4517), INT16_C( -1927),
                         INT16_C( 30587), INT16_C( 20606), INT16_C( -7728), INT16_C( 16094)),
      2 ,
      easysimd_mm_set_epi16(INT16_C(  8111), INT16_C(-21165), INT16_C(-10396), INT16_C(   914),
                         INT16_C( 28557), INT16_C( -9612), INT16_C(  7714), INT16_C( 27946)),
      5 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( -8303), INT16_C( -4730), INT16_C( -8980), INT16_C(  9007),
                         INT16_C(-29251), INT16_C(   241), INT16_C(-18226), INT16_C(-12616)),
      0 ,
      easysimd_mm_set_epi16(INT16_C(-28022), INT16_C(  8492), INT16_C( 17893), INT16_C( -9225),
                         INT16_C( 16548), INT16_C( 15468), INT16_C( 27219), INT16_C( -7597)),
      4 ,
      1 },
    { easysimd_mm_set_epi16(INT16_C( 19860), INT16_C( 16482), INT16_C(-16335), INT16_C(-20955),
                         INT16_C( 18264), INT16_C(-27732), INT16_C( 28124), INT16_C(  -857)),
      1 ,
      easysimd_mm_set_epi16(INT16_C(-17937), INT16_C(-21402), INT16_C( 32603), INT16_C( -5545),
                         INT16_C(   641), INT16_C( -7799), INT16_C(  8187), INT16_C( 29379)),
      0 ,
      1 }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int r = easysimd_mm_cmpestrz(test_vec[i].a, test_vec[i].la, test_vec[i].b, test_vec[i].lb, 1);
    easysimd_assert_equal_i(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_cmpgt_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m128i b;
    easysimd__m128i r;
  } test_vec[8] = {
    { easysimd_mm_set_epi64x(INT64_C(-2149331112921330677), INT64_C( 3189038460188560982)),
      easysimd_mm_set_epi64x(INT64_C( -996047549682722220), INT64_C(-3995545326609437904)),
      easysimd_mm_set_epi64x( INT64_C(0), ~INT64_C(0)) },
    { easysimd_mm_set_epi64x(INT64_C( 3213898448913237846), INT64_C( 9188286366666087308)),
      easysimd_mm_set_epi64x(INT64_C( 2918885787365950970), INT64_C( 6780053140456787494)),
      easysimd_mm_set_epi64x(~INT64_C(0), ~INT64_C(0)) },
    { easysimd_mm_set_epi64x(INT64_C(-6480415937191367948), INT64_C( 6434069133602920016)),
      easysimd_mm_set_epi64x(INT64_C( 8054577307931165184), INT64_C( 2226222084862743618)),
      easysimd_mm_set_epi64x( INT64_C(0), ~INT64_C(0)) },
    { easysimd_mm_set_epi64x(INT64_C(-6197561420805751907), INT64_C( 4778870285233423339)),
      easysimd_mm_set_epi64x(INT64_C( 1839658993612937599), INT64_C( -902367911293731861)),
      easysimd_mm_set_epi64x( INT64_C(0), ~INT64_C(0)) },
    { easysimd_mm_set_epi64x(INT64_C( 5091127324004768664), INT64_C(-2002251908801446460)),
      easysimd_mm_set_epi64x(INT64_C(-9056506211008935561), INT64_C(-6487933609077704174)),
      easysimd_mm_set_epi64x(~INT64_C(0), ~INT64_C(0)) },
    { easysimd_mm_set_epi64x(INT64_C(-4743149223868910453), INT64_C(-4137271544350199785)),
      easysimd_mm_set_epi64x(INT64_C( 4762909370147937560), INT64_C( 6560801355595049799)),
      easysimd_mm_set_epi64x( INT64_C(0),  INT64_C(0)) },
    { easysimd_mm_set_epi64x(INT64_C(  913044205052582612), INT64_C(-2362244502684338485)),
      easysimd_mm_set_epi64x(INT64_C( -603710511502052754), INT64_C(-3179203207537477667)),
      easysimd_mm_set_epi64x(~INT64_C(0), ~INT64_C(0)) },
    { easysimd_mm_set_epi64x(INT64_C( 6753725813089147170), INT64_C( 7031124288307654085)),
      easysimd_mm_set_epi64x(INT64_C( 5046765831366456160), INT64_C( 6981054579474564569)),
      easysimd_mm_set_epi64x(~INT64_C(0), ~INT64_C(0)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r,
                 a = test_vec[i].a,
                 b = test_vec[i].b;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpgt_epi64(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpgt_epi64 on easysimd");
    easysimd_assert_m128i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_cmpistrs_8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m128i b;
    int r;
  } test_vec[] = {
    { easysimd_mm_set_epi8(INT8_C(  25), INT8_C(  54), INT8_C( -66), INT8_C( -16),
                        INT8_C(  66), INT8_C(-116), INT8_C( -35), INT8_C(  78),
                        INT8_C( 107), INT8_C(  11), INT8_C(-110), INT8_C(  90),
                        INT8_C(  -2), INT8_C(-109), INT8_C( -34), INT8_C(  53)),
      easysimd_mm_set_epi8(INT8_C( -86), INT8_C(-125), INT8_C( -30), INT8_C(   1),
                        INT8_C(  69), INT8_C( -79), INT8_C( -16), INT8_C(  34),
                        INT8_C(  73), INT8_C(  71), INT8_C( -50), INT8_C( -27),
                        INT8_C( -56), INT8_C(-106), INT8_C( -90), INT8_C( 104)),
      0 },
    { easysimd_mm_set_epi8(INT8_C(0), INT8_C( -93), INT8_C(  -2), INT8_C( -97),
                        INT8_C(-117), INT8_C( -46), INT8_C(-107), INT8_C(-101),
                        INT8_C(-104), INT8_C( -97), INT8_C(-123), INT8_C( -15),
                        INT8_C( 101), INT8_C( 123), INT8_C(-123), INT8_C(  -2)),
      easysimd_mm_set_epi8(INT8_C(  -6), INT8_C(   9), INT8_C(  43), INT8_C(-128),
                        INT8_C( -64), INT8_C(  71), INT8_C( -48), INT8_C(  11),
                        INT8_C(  61), INT8_C( -61), INT8_C(  55), INT8_C(-108),
                        INT8_C(  95), INT8_C( -26), INT8_C( -76), INT8_C(  92)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(  74), INT8_C(0), INT8_C(  48), INT8_C( 106),
                        INT8_C( -25), INT8_C(  49), INT8_C( -66), INT8_C(  38),
                        INT8_C( -18), INT8_C(-127), INT8_C(  20), INT8_C( -68),
                        INT8_C( 117), INT8_C(-114), INT8_C( 113), INT8_C( -43)),
      easysimd_mm_set_epi8(INT8_C(  19), INT8_C(  27), INT8_C(  69), INT8_C(   3),
                        INT8_C(  75), INT8_C( -73), INT8_C(  19), INT8_C( -16),
                        INT8_C( -20), INT8_C( -75), INT8_C( -47), INT8_C( -90),
                        INT8_C(-126), INT8_C(  82), INT8_C( -85), INT8_C(  65)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -36), INT8_C(-128), INT8_C( 0), INT8_C( -37),
                        INT8_C(-116), INT8_C( 107), INT8_C( -26), INT8_C(-121),
                        INT8_C( -65), INT8_C( 100), INT8_C(  78), INT8_C(   8),
                        INT8_C(-100), INT8_C( -73), INT8_C( -59), INT8_C( -67)),
      easysimd_mm_set_epi8(INT8_C(-124), INT8_C( -83), INT8_C( -63), INT8_C( -32),
                        INT8_C(  28), INT8_C( 100), INT8_C(  27), INT8_C(  38),
                        INT8_C( -55), INT8_C(  20), INT8_C( -89), INT8_C( -37),
                        INT8_C(  91), INT8_C(  56), INT8_C( -14), INT8_C( -98)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(-111), INT8_C( -83), INT8_C( 125), INT8_C(  0),
                        INT8_C(  53), INT8_C(  48), INT8_C( -61), INT8_C( -87),
                        INT8_C(  65), INT8_C( 121), INT8_C(  71), INT8_C(  10),
                        INT8_C( 118), INT8_C( -63), INT8_C( -96), INT8_C(   9)),
      easysimd_mm_set_epi8(INT8_C( -41), INT8_C(  -1), INT8_C( -57), INT8_C( 113),
                        INT8_C( 101), INT8_C(  39), INT8_C(  86), INT8_C(   5),
                        INT8_C(  19), INT8_C(  -8), INT8_C( 110), INT8_C(  44),
                        INT8_C(-100), INT8_C( -52), INT8_C(-126), INT8_C(  -3)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -18), INT8_C(   10), INT8_C(  22), INT8_C( -30),
                        INT8_C( 0), INT8_C(  75), INT8_C(  26), INT8_C( 106),
                        INT8_C( -59), INT8_C(-112), INT8_C(  62), INT8_C(   5),
                        INT8_C(  -4), INT8_C( -40), INT8_C(  68), INT8_C(  77)),
      easysimd_mm_set_epi8(INT8_C( -23), INT8_C(  71), INT8_C(  21), INT8_C(-100),
                        INT8_C(  36), INT8_C( -96), INT8_C( -10), INT8_C(  20),
                        INT8_C( -22), INT8_C( 110), INT8_C(  98), INT8_C(  67),
                        INT8_C(  12), INT8_C( -74), INT8_C( -50), INT8_C(  32)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( 106), INT8_C( -84), INT8_C(  30), INT8_C(  79),
                        INT8_C( 124), INT8_C( 0), INT8_C( -53), INT8_C( -99),
                        INT8_C( -15), INT8_C( 108), INT8_C( -91), INT8_C(   4),
                        INT8_C(  21), INT8_C(  48), INT8_C(  29), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C( 100), INT8_C( 100), INT8_C(  71), INT8_C(  90),
                        INT8_C( -52), INT8_C( 119), INT8_C( -64), INT8_C(-104),
                        INT8_C(  16), INT8_C( -98), INT8_C(  37), INT8_C(  -2),
                        INT8_C(  -6), INT8_C( -12), INT8_C( 117), INT8_C(  87)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(0), INT8_C(-117),
                        INT8_C( 102), INT8_C( -52), INT8_C( -52), INT8_C( -25),
                        INT8_C(  -6), INT8_C( 112), INT8_C( 116), INT8_C(  39)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(10), INT8_C(0),
                        INT8_C( 102), INT8_C( -52), INT8_C( -52), INT8_C( -25),
                        INT8_C(  -6), INT8_C( 112), INT8_C( 116), INT8_C(  39)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(10), INT8_C(-117),
                        INT8_C( 0), INT8_C( -52), INT8_C( -52), INT8_C( -25),
                        INT8_C(  -6), INT8_C( 112), INT8_C( 116), INT8_C(  39)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(10), INT8_C(-117),
                        INT8_C( 102), INT8_C( 0), INT8_C( -52), INT8_C( -25),
                        INT8_C(  -6), INT8_C( 112), INT8_C( 116), INT8_C(  39)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(10), INT8_C(-117),
                        INT8_C( 102), INT8_C( -52), INT8_C( 0), INT8_C( -25),
                        INT8_C(  -6), INT8_C( 112), INT8_C( 116), INT8_C(  39)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(20), INT8_C(-117),
                        INT8_C( 102), INT8_C( -52), INT8_C( -52), INT8_C( 0),
                        INT8_C(  -6), INT8_C( 112), INT8_C( 116), INT8_C(  39)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(50), INT8_C(-117),
                        INT8_C( 102), INT8_C( -52), INT8_C( -52), INT8_C( -25),
                        INT8_C(  0), INT8_C( 112), INT8_C( 116), INT8_C(  39)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(60), INT8_C(-117),
                        INT8_C( 102), INT8_C( -52), INT8_C( -52), INT8_C( -25),
                        INT8_C(  -6), INT8_C( 0), INT8_C( 116), INT8_C(  39)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(70), INT8_C(-117),
                        INT8_C( 102), INT8_C( -52), INT8_C( -52), INT8_C( -25),
                        INT8_C(  -6), INT8_C( 112), INT8_C( 0), INT8_C(  39)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -55), INT8_C(  16), INT8_C( -12), INT8_C(-128),
                        INT8_C( -68), INT8_C( 111), INT8_C(80), INT8_C(-117),
                        INT8_C( 102), INT8_C( -52), INT8_C( -52), INT8_C( -25),
                        INT8_C(  -6), INT8_C( 112), INT8_C( 116), INT8_C(  0)),
      easysimd_mm_set_epi8(INT8_C(  29), INT8_C( -72), INT8_C(  47), INT8_C(  93),
                        INT8_C( -90), INT8_C( 115), INT8_C(  36), INT8_C( -93),
                        INT8_C( 106), INT8_C(  -6), INT8_C( -91), INT8_C(  34),
                        INT8_C( -44), INT8_C( -69), INT8_C( 123), INT8_C(  51)),
      1 }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int r = easysimd_mm_cmpistrs(test_vec[i].a, test_vec[i].b, 0);
    easysimd_assert_equal_i(r, test_vec[i].r);
  }

  return 0;
}
static int
test_easysimd_mm_cmpistrs_16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m128i b;
    int r;
  } test_vec[] = {
    { easysimd_mm_set_epi16(INT16_C(  6454), INT16_C(-16656), INT16_C( 17036), INT16_C( -8882),
                         INT16_C( 27403), INT16_C(-28070), INT16_C(  -365), INT16_C( -8651)),
      easysimd_mm_set_epi16(INT16_C(-21885), INT16_C( -7679), INT16_C( 17841), INT16_C( -4062),
                         INT16_C( 18759), INT16_C(-12571), INT16_C(-14186), INT16_C(-22936)),
      0 },
    { easysimd_mm_set_epi16(INT16_C(0), INT16_C(  -353), INT16_C(-29742), INT16_C(-27237),
                         INT16_C(-26465), INT16_C(-31247), INT16_C( 25979), INT16_C(-31234)),
      easysimd_mm_set_epi16(INT16_C( -1527), INT16_C( 11136), INT16_C(-16313), INT16_C(-12277),
                         INT16_C( 15811), INT16_C( 14228), INT16_C( 24550), INT16_C(-19364)),
      1 },
    { easysimd_mm_set_epi16(INT16_C( 19077), INT16_C( 0), INT16_C( -6351), INT16_C(-16858),
                         INT16_C( -4479), INT16_C(  5308), INT16_C( 30094), INT16_C( 29141)),
      easysimd_mm_set_epi16(INT16_C(  4891), INT16_C( 17667), INT16_C( 19383), INT16_C(  5104),
                         INT16_C( -4939), INT16_C(-11866), INT16_C(-32174), INT16_C(-21695)),
      1 },
    { easysimd_mm_set_epi16(INT16_C( -9088), INT16_C( -7717), INT16_C(0), INT16_C( -6521),
                         INT16_C(-16540), INT16_C( 19976), INT16_C(-25417), INT16_C(-14915)),
      easysimd_mm_set_epi16(INT16_C(-31571), INT16_C(-15904), INT16_C(  7268), INT16_C(  6950),
                         INT16_C(-14060), INT16_C(-22565), INT16_C( 23352), INT16_C( -3426)),
      1 },
    { easysimd_mm_set_epi16(INT16_C(-28243), INT16_C( 32023), INT16_C( 13616), INT16_C(0),
                         INT16_C( 16761), INT16_C( 18186), INT16_C( 30401), INT16_C(-24567)),
      easysimd_mm_set_epi16(INT16_C(-10241), INT16_C(-14479), INT16_C( 25895), INT16_C( 22021),
                         INT16_C(  5112), INT16_C( 28204), INT16_C(-25396), INT16_C(-32003)),
      1 },
    { easysimd_mm_set_epi16(INT16_C( -4608), INT16_C(  5858), INT16_C(-12725), INT16_C(  6762),
                         INT16_C(0), INT16_C( 15877), INT16_C(  -808), INT16_C( 17485)),
      easysimd_mm_set_epi16(INT16_C( -5817), INT16_C(  5532), INT16_C(  9376), INT16_C( -2540),
                         INT16_C( -5522), INT16_C( 25155), INT16_C(  3254), INT16_C(-12768)),
      1 },
    { easysimd_mm_set_epi16(INT16_C( 27308), INT16_C(  7759), INT16_C( 31856), INT16_C(-13411),
                         INT16_C( -3732), INT16_C(0), INT16_C(  5424), INT16_C(  7625)),
      easysimd_mm_set_epi16(INT16_C( 25700), INT16_C( 18266), INT16_C(-13193), INT16_C(-16232),
                         INT16_C(  4254), INT16_C(  9726), INT16_C( -1292), INT16_C( 30039)),
      1 },
    { easysimd_mm_set_epi16(INT16_C(-14064), INT16_C( -2944), INT16_C(-17297), INT16_C(-26741),
                         INT16_C( 26316), INT16_C(-13081), INT16_C( 0), INT16_C( 29735)),
      easysimd_mm_set_epi16(INT16_C(  7608), INT16_C( 12125), INT16_C(-22925), INT16_C(  9379),
                         INT16_C( 27386), INT16_C(-23262), INT16_C(-11077), INT16_C( 31539)),
      1 },
    { easysimd_mm_set_epi16(INT16_C(-14064), INT16_C( -2944), INT16_C(-17297), INT16_C(-26741),
                         INT16_C( 26316), INT16_C(-13081), INT16_C( 70), INT16_C( 0)),
      easysimd_mm_set_epi16(INT16_C(  7608), INT16_C( 12125), INT16_C(-22925), INT16_C(  9379),
                         INT16_C( 27386), INT16_C(-23262), INT16_C(-11077), INT16_C( 31539)),
      1 }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int r = easysimd_mm_cmpistrs(test_vec[i].a, test_vec[i].b, 1);
    easysimd_assert_equal_i(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_cmpistrz_8(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m128i b;
    int r;
  } test_vec[] = {
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  99),
                        INT8_C( 115), INT8_C( -51), INT8_C(-105), INT8_C(  30),
                        INT8_C(  95), INT8_C( -27), INT8_C(  27), INT8_C(-118)),
      0 },
    { easysimd_mm_set_epi8(INT8_C( -20), INT8_C( -41), INT8_C( -11), INT8_C(  77),
                        INT8_C(   7), INT8_C( -34), INT8_C(  46), INT8_C( -70),
                        INT8_C(  58), INT8_C(  42), INT8_C(  57), INT8_C(  56),
                        INT8_C(  69), INT8_C( -64), INT8_C(-121), INT8_C(  96)),
      easysimd_mm_set_epi8(INT8_C(  0), INT8_C( -55), INT8_C( -68), INT8_C(   5),
                        INT8_C(  46), INT8_C(  24), INT8_C(-116), INT8_C( -73),
                        INT8_C(  22), INT8_C( -42), INT8_C( -48), INT8_C(  76),
                        INT8_C(  14), INT8_C(  67), INT8_C(  97), INT8_C(-116)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -50), INT8_C(  57), INT8_C(  48), INT8_C(-101),
                        INT8_C(   3), INT8_C( 113), INT8_C( 104), INT8_C(-118),
                        INT8_C(  74), INT8_C( -93), INT8_C( -56), INT8_C(  86),
                        INT8_C( -14), INT8_C( -37), INT8_C(  55), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C( 119), INT8_C( 0), INT8_C(-110), INT8_C(  99),
                        INT8_C(  83), INT8_C( -37), INT8_C( -75), INT8_C( -18),
                        INT8_C( 109), INT8_C(  -9), INT8_C(  40), INT8_C(  86),
                        INT8_C( -54), INT8_C( -27), INT8_C( -52), INT8_C(  75)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(-109), INT8_C( 127), INT8_C( -99), INT8_C( -62),
                        INT8_C(  99), INT8_C(-120), INT8_C(  41), INT8_C(-123),
                        INT8_C( -92), INT8_C( 114), INT8_C(  53), INT8_C(  90),
                        INT8_C(  -5), INT8_C( -27), INT8_C(  98), INT8_C( -67)),
      easysimd_mm_set_epi8(INT8_C(  80), INT8_C(  26), INT8_C( 0), INT8_C(-117),
                        INT8_C( -50), INT8_C( -38), INT8_C( -56), INT8_C( -22),
                        INT8_C(  51), INT8_C( -76), INT8_C(  55), INT8_C( -49),
                        INT8_C(  57), INT8_C(  60), INT8_C( -63), INT8_C(-107)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(  21), INT8_C(   6), INT8_C(  94), INT8_C(  46),
                        INT8_C(  20), INT8_C( -10), INT8_C( -62), INT8_C(  -7),
                        INT8_C(  32), INT8_C( -63), INT8_C( 113), INT8_C( -62),
                        INT8_C(   0), INT8_C(  63), INT8_C(  77), INT8_C( -53)),
      easysimd_mm_set_epi8(INT8_C( 118), INT8_C(   90), INT8_C(  98), INT8_C(0),
                        INT8_C( -82), INT8_C(  25), INT8_C( -11), INT8_C(  94),
                        INT8_C( 100), INT8_C(   3), INT8_C(-109), INT8_C(-117),
                        INT8_C( -61), INT8_C( 100), INT8_C(-120), INT8_C( -94)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(  54), INT8_C( -82), INT8_C(  50), INT8_C(  20),
                        INT8_C( -78), INT8_C(  25), INT8_C( -39), INT8_C( 113),
                        INT8_C( -88), INT8_C( -49), INT8_C(-105), INT8_C(  11),
                        INT8_C(  21), INT8_C( -81), INT8_C( -49), INT8_C( 113)),
      easysimd_mm_set_epi8(INT8_C(   7), INT8_C( -95), INT8_C(  34), INT8_C( -90),
                        INT8_C(  0), INT8_C(  98), INT8_C( -10), INT8_C(  55),
                        INT8_C( 125), INT8_C(  77), INT8_C(  23), INT8_C(  95),
                        INT8_C(  75), INT8_C(  43), INT8_C(  52), INT8_C(  72)),
      1 },
    { easysimd_mm_set_epi8(INT8_C( -47), INT8_C(  15), INT8_C(-110), INT8_C( -19),
                        INT8_C( -43), INT8_C( -27), INT8_C(  31), INT8_C( -52),
                        INT8_C(  95), INT8_C( -61), INT8_C(  75), INT8_C( 103),
                        INT8_C( -10), INT8_C(  24), INT8_C(  91), INT8_C( -50)),
      easysimd_mm_set_epi8(INT8_C(-116), INT8_C(-113), INT8_C(  47), INT8_C( -63),
                        INT8_C(  35), INT8_C( 0), INT8_C(  63), INT8_C(  12),
                        INT8_C(   7), INT8_C( 120), INT8_C( -97), INT8_C(  84),
                        INT8_C( 125), INT8_C( -85), INT8_C(-110), INT8_C( -21)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(  98), INT8_C( -51), INT8_C(  74), INT8_C( 114),
                        INT8_C(-123), INT8_C(  80), INT8_C(  99), INT8_C( -50),
                        INT8_C(  52), INT8_C(  86), INT8_C( -10), INT8_C( -16),
                        INT8_C(-121), INT8_C(  99), INT8_C(-115), INT8_C( 124)),
      easysimd_mm_set_epi8(INT8_C( -84), INT8_C(-104), INT8_C(  72), INT8_C( -97),
                        INT8_C(  90), INT8_C( -38), INT8_C( 0), INT8_C( -55),
                        INT8_C(-118), INT8_C(-106), INT8_C(-109), INT8_C( 101),
                        INT8_C(  87), INT8_C(-102), INT8_C( -96), INT8_C( -13)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  0),
                        INT8_C( 115), INT8_C( -51), INT8_C(-105), INT8_C(  30),
                        INT8_C(  95), INT8_C( -27), INT8_C(  27), INT8_C(-118)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  99),
                        INT8_C( 0), INT8_C( -51), INT8_C(-105), INT8_C(  30),
                        INT8_C(  95), INT8_C( -27), INT8_C(  27), INT8_C(-118)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  99),
                        INT8_C( 115), INT8_C( 0), INT8_C(-105), INT8_C(  30),
                        INT8_C(  95), INT8_C( -27), INT8_C(  27), INT8_C(-118)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  99),
                        INT8_C( 115), INT8_C( -51), INT8_C(0), INT8_C(  30),
                        INT8_C(  95), INT8_C( -27), INT8_C(  27), INT8_C(-118)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  99),
                        INT8_C( 115), INT8_C( -51), INT8_C(-105), INT8_C(  0),
                        INT8_C(  95), INT8_C( -27), INT8_C(  27), INT8_C(-118)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  99),
                        INT8_C( 115), INT8_C( -51), INT8_C(-105), INT8_C(  30),
                        INT8_C(  0), INT8_C( -27), INT8_C(  27), INT8_C(-118)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  99),
                        INT8_C( 115), INT8_C( -51), INT8_C(-105), INT8_C(  30),
                        INT8_C(  95), INT8_C( 0), INT8_C(  27), INT8_C(-118)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  99),
                        INT8_C( 115), INT8_C( -51), INT8_C(-105), INT8_C(  30),
                        INT8_C(  95), INT8_C( -27), INT8_C(  0), INT8_C(-118)),
      1 },
    { easysimd_mm_set_epi8(INT8_C(   1), INT8_C(  77), INT8_C( -64), INT8_C(-123),
                        INT8_C(  49), INT8_C( -50), INT8_C(  60), INT8_C(  57),
                        INT8_C(  64), INT8_C( -70), INT8_C(  56), INT8_C( -69),
                        INT8_C(-103), INT8_C( -41), INT8_C(  82), INT8_C( -55)),
      easysimd_mm_set_epi8(INT8_C(-103), INT8_C( -36), INT8_C( -57), INT8_C( -38),
                        INT8_C( 108), INT8_C( -48), INT8_C( -86), INT8_C(  99),
                        INT8_C( 115), INT8_C( -51), INT8_C(-105), INT8_C(  30),
                        INT8_C(  95), INT8_C( -27), INT8_C(  27), INT8_C(0)),
      1 }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int r = easysimd_mm_cmpistrz(test_vec[i].a, test_vec[i].b, 0);
    easysimd_assert_equal_i(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_cmpistrz_16(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128i a;
    easysimd__m128i b;
    int r;
  } test_vec[] = {
    { easysimd_mm_set_epi16(INT16_C(   333), INT16_C(-16251), INT16_C( 12750), INT16_C( 15417),
                         INT16_C( 16570), INT16_C( 14523), INT16_C(-26153), INT16_C( 21193)),
      easysimd_mm_set_epi16(INT16_C(-26148), INT16_C(-14374), INT16_C( 27856), INT16_C(-21917),
                         INT16_C( 29645), INT16_C(-26850), INT16_C( 24549), INT16_C(  7050)),
      0 },
    { easysimd_mm_set_epi16(INT16_C( -4905), INT16_C( -2739), INT16_C(  2014), INT16_C( 11962),
                         INT16_C( 14890), INT16_C( 14648), INT16_C( 17856), INT16_C(-30880)),
      easysimd_mm_set_epi16(INT16_C( 0), INT16_C(-17403), INT16_C( 11800), INT16_C(-29513),
                         INT16_C(  5846), INT16_C(-12212), INT16_C(  3651), INT16_C( 24972)),
      1 },
    { easysimd_mm_set_epi16(INT16_C(-12743), INT16_C( 12443), INT16_C(   881), INT16_C( 26762),
                         INT16_C( 19107), INT16_C(-14250), INT16_C( -3365), INT16_C( 14281)),
      easysimd_mm_set_epi16(INT16_C( 30693), INT16_C(0), INT16_C( 21467), INT16_C(-18962),
                         INT16_C( 28151), INT16_C( 10326), INT16_C(-13595), INT16_C(-13237)),
      1 },
    { easysimd_mm_set_epi16(INT16_C(-27777), INT16_C(-25150), INT16_C( 25480), INT16_C( 10629),
                         INT16_C(-23438), INT16_C( 13658), INT16_C( -1051), INT16_C( 25277)),
      easysimd_mm_set_epi16(INT16_C( 20506), INT16_C( 31627), INT16_C(0), INT16_C(-14102),
                         INT16_C( 13236), INT16_C( 14287), INT16_C( 14652), INT16_C(-15979)),
      1 },
    { easysimd_mm_set_epi16(INT16_C(  5382), INT16_C( 24110), INT16_C(  5366), INT16_C(-15623),
                         INT16_C(  8385), INT16_C( 29122), INT16_C(    63), INT16_C( 19915)),
      easysimd_mm_set_epi16(INT16_C( 30208), INT16_C( 25244), INT16_C(-20967), INT16_C( 0),
                         INT16_C( 25603), INT16_C(-27765), INT16_C(-15516), INT16_C(-30558)),
      1 },
    { easysimd_mm_set_epi16(INT16_C( 13998), INT16_C( 12820), INT16_C(-19943), INT16_C( -9871),
                         INT16_C(-22321), INT16_C(-26869), INT16_C(  5551), INT16_C(-12431)),
      easysimd_mm_set_epi16(INT16_C(  1953), INT16_C(  8870), INT16_C( -1694), INT16_C( -2505),
                         INT16_C( 0), INT16_C(  5983), INT16_C( 19243), INT16_C( 13384)),
      1 },
    { easysimd_mm_set_epi16(INT16_C(-12017), INT16_C(-27923), INT16_C(-10779), INT16_C(  8140),
                         INT16_C( 24515), INT16_C( 19303), INT16_C( -2536), INT16_C( 23502)),
      easysimd_mm_set_epi16(INT16_C(-29553), INT16_C( 12225), INT16_C(  9080), INT16_C( 16140),
                         INT16_C(  1912), INT16_C(0), INT16_C( 32171), INT16_C(-27925)),
      1 },
    { easysimd_mm_set_epi16(INT16_C( 25293), INT16_C( 19058), INT16_C(-31408), INT16_C( 25550),
                         INT16_C( 13398), INT16_C( -2320), INT16_C(-30877), INT16_C(-29316)),
      easysimd_mm_set_epi16(INT16_C(-21352), INT16_C( 18591), INT16_C( 23258), INT16_C(-24887),
                         INT16_C(-30058), INT16_C(-27803), INT16_C( 0), INT16_C(-24333)),
      1 },
    { easysimd_mm_set_epi16(INT16_C( 25293), INT16_C( 19058), INT16_C(-31408), INT16_C( 25550),
                         INT16_C( 13398), INT16_C( -2320), INT16_C(-30877), INT16_C(-29316)),
      easysimd_mm_set_epi16(INT16_C(-21352), INT16_C( 18591), INT16_C( 23258), INT16_C(-24887),
                         INT16_C(-30058), INT16_C(-27803), INT16_C( 870), INT16_C(0)),
      1 }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int r = easysimd_mm_cmpistrz(test_vec[i].a, test_vec[i].b, 1);
    easysimd_assert_equal_i(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_crc32_u8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    uint32_t crc;
    uint8_t v;
    uint32_t r;
  } test_vec[] = {
    { UINT32_C(3488119326),
      UINT8_C(233),
      UINT32_C( 661382116) },
    { UINT32_C(4181338815),
      UINT8_C(106),
      UINT32_C(3873165213) },
    { UINT32_C(3611029619),
      UINT8_C(190),
      UINT32_C(2087866855) },
    { UINT32_C(3633137044),
      UINT8_C(206),
      UINT32_C( 975142830) },
    { UINT32_C(3701195429),
      UINT8_C( 59),
      UINT32_C(1041029362) },
    { UINT32_C(1574265292),
      UINT8_C( 54),
      UINT32_C(2563871276) },
    { UINT32_C( 464550963),
      UINT8_C( 75),
      UINT32_C(4217027774) },
    { UINT32_C(3547716249),
      UINT8_C(211),
      UINT32_C( 709509214) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    uint32_t crc = test_vec[i].crc;
    uint8_t v = test_vec[i].v;
    uint32_t r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_crc32_u8(crc, v);
    }
    EASYSIMD_TEST_PERF_END("_mm_crc32_u8");
    easysimd_assert_equal_u32(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_crc32_u16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    uint32_t crc;
    uint16_t v;
    uint32_t r;
  } test_vec[] = {
    { UINT32_C( 728173782),
      UINT16_C(58051),
      UINT32_C( 765801584) },
    { UINT32_C(2531395991),
      UINT16_C(57124),
      UINT32_C(2048446530) },
    { UINT32_C( 297646163),
      UINT16_C( 4793),
      UINT32_C( 145203338) },
    { UINT32_C(4018813906),
      UINT16_C( 4093),
      UINT32_C(1871435995) },
    { UINT32_C(1176812284),
      UINT16_C(48677),
      UINT32_C(1916618632) },
    { UINT32_C(1019935701),
      UINT16_C(36390),
      UINT32_C( 873790012) },
    { UINT32_C(  26721567),
      UINT16_C(47956),
      UINT32_C(1883589466) },
    { UINT32_C(2658379744),
      UINT16_C(11705),
      UINT32_C(2809192825) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    uint32_t crc = test_vec[i].crc;
    uint16_t v = test_vec[i].v;
    uint32_t r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_crc32_u16(crc, v);
    }
    EASYSIMD_TEST_PERF_END("_mm_crc32_u16");
    easysimd_assert_equal_u32(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_crc32_u32 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    uint32_t crc;
    uint32_t v;
    uint32_t r;
  } test_vec[] = {
    { UINT32_C(2436525653),
      UINT32_C(2335302948),
      UINT32_C(3283443050) },
    { UINT32_C(1145760123),
      UINT32_C(3888075817),
      UINT32_C(1275307424) },
    { UINT32_C(1404614118),
      UINT32_C(1676357820),
      UINT32_C(2140092727) },
    { UINT32_C( 546365338),
      UINT32_C(2107344167),
      UINT32_C(3150313630) },
    { UINT32_C( 386848243),
      UINT32_C( 899891386),
      UINT32_C(3310319573) },
    { UINT32_C(1383787817),
      UINT32_C( 674838849),
      UINT32_C(4185068584) },
    { UINT32_C(2877026799),
      UINT32_C(3155060257),
      UINT32_C(1654064964) },
    { UINT32_C(1826397765),
      UINT32_C( 401176356),
      UINT32_C(1688688127) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    uint32_t crc = test_vec[i].crc;
    uint32_t v = test_vec[i].v;
    uint32_t r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_crc32_u32(crc, v);
    }
    EASYSIMD_TEST_PERF_END("_mm_crc32_u32");
    easysimd_assert_equal_u32(r, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm_crc32_u64 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    uint64_t crc;
    uint64_t v;
    uint64_t r;
  } test_vec[] = {
    { UINT64_C(10964460371209988374),
      UINT64_C(14849487482734297659),
      UINT64_C(          2530609228) },
    { UINT64_C(14906864906438122131),
      UINT64_C(10579630055528908036),
      UINT64_C(          2336937406) },
    { UINT64_C( 8450238593151902479),
      UINT64_C(14846135117717324041),
      UINT64_C(          2389161291) },
    { UINT64_C(15754071801993691947),
      UINT64_C(17187741549636385145),
      UINT64_C(          2628533589) },
    { UINT64_C(17686444891285660866),
      UINT64_C(12477846746303524896),
      UINT64_C(          1813528429) },
    { UINT64_C( 3308212454223314746),
      UINT64_C( 1686784245036627611),
      UINT64_C(           721365030) },
    { UINT64_C(  157211343182889549),
      UINT64_C(14854147642213948918),
      UINT64_C(          1805070678) },
    { UINT64_C( 7018798198485263495),
      UINT64_C( 9253000792826939901),
      UINT64_C(          1576406668) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    uint64_t crc = test_vec[i].crc;
    uint64_t v = test_vec[i].v;
    uint64_t r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_crc32_u64(crc, v);
    }
    EASYSIMD_TEST_PERF_END("_mm_crc32_u64");
    easysimd_assert_equal_u64(r, test_vec[i].r);
  }

  return 0;
}

#if (!defined(__clang__))
easysimd__m128i
easysimd_mm_load_epu8(void const* mem_addr) {
  easysimd__m128i_private r_;
  for(size_t i = 0; i < sizeof(r_.u8) / sizeof(r_.u8[0]); i++){
    r_.u8[i] = *(((uint8_t *)mem_addr) + i);
  }
  return easysimd__m128i_from_private(r_);
}

static int
test_easysimd_mm_cmpistrm_8bit(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    uint8_t  a[16];
    uint8_t  b[16];
    uint16_t r[8];
  } test_vec[] = {
/*0*/
    { { UINT8_C( 85), UINT8_C(110), UINT8_C(132), UINT8_C( 93), UINT8_C(179), UINT8_C( 55), UINT8_C(220), UINT8_C(173),
        UINT8_C(241), UINT8_C(143), UINT8_C(132), UINT8_C( 70), UINT8_C(174), UINT8_C( 40), UINT8_C(109), UINT8_C( 30) },
      { UINT8_C(234), UINT8_C(246), UINT8_C(123), UINT8_C(206), UINT8_C(182), UINT8_C( 51), UINT8_C(246), UINT8_C( 16),
        UINT8_C(132), UINT8_C(205), UINT8_C(196), UINT8_C(231), UINT8_C( 42), UINT8_C( 77), UINT8_C(177), UINT8_C(128) },
      { UINT16_C(  256), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(231), UINT8_C(174), UINT8_C(226), UINT8_C( 37), UINT8_C(128), UINT8_C(  9), UINT8_C( 71), UINT8_C(226),
        UINT8_C(107), UINT8_C(177), UINT8_C( 99), UINT8_C( 59), UINT8_C( 72), UINT8_C( 59), UINT8_C(111), UINT8_C( 56) },
      { UINT8_C(167), UINT8_C(179), UINT8_C(  7), UINT8_C(169), UINT8_C(226), UINT8_C(114), UINT8_C(206), UINT8_C(167),
        UINT8_C(134), UINT8_C(214), UINT8_C(150), UINT8_C( 32), UINT8_C(211), UINT8_C(145), UINT8_C(197), UINT8_C(187) },
      { UINT16_C(   16), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 63), UINT8_C(167), UINT8_C(224), UINT8_C(191), UINT8_C(176), UINT8_C( 40), UINT8_C(161), UINT8_C( 27),
        UINT8_C(217), UINT8_C(  5), UINT8_C( 87), UINT8_C( 33), UINT8_C( 64), UINT8_C(198), UINT8_C( 90), UINT8_C(231) },
      { UINT8_C(121), UINT8_C( 97), UINT8_C(145), UINT8_C( 91), UINT8_C(211), UINT8_C( 95), UINT8_C(  2), UINT8_C( 89),
        UINT8_C( 53), UINT8_C(152), UINT8_C(121), UINT8_C(  8), UINT8_C( 41), UINT8_C( 63), UINT8_C(195), UINT8_C(104) },
      { UINT16_C( 8192), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(230), UINT8_C(164), UINT8_C( 39), UINT8_C(151), UINT8_C(204), UINT8_C(201), UINT8_C(178), UINT8_C(165),
        UINT8_C(206), UINT8_C(  9), UINT8_C(199), UINT8_C( 14), UINT8_C(207), UINT8_C( 33), UINT8_C(246), UINT8_C( 72) },
      { UINT8_C(130), UINT8_C(135), UINT8_C(163), UINT8_C( 85), UINT8_C(230), UINT8_C(165), UINT8_C(174), UINT8_C( 27),
        UINT8_C( 62), UINT8_C( 40), UINT8_C( 35), UINT8_C(103), UINT8_C(103), UINT8_C(231), UINT8_C(208), UINT8_C( 77) },
      { UINT16_C(   48), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(110), UINT8_C( 43), UINT8_C(  7), UINT8_C( 47), UINT8_C(194), UINT8_C(  4), UINT8_C(189), UINT8_C( 98),
        UINT8_C(199), UINT8_C( 90), UINT8_C(210), UINT8_C(172), UINT8_C(237), UINT8_C(139), UINT8_C( 18), UINT8_C(  7) },
      { UINT8_C(231), UINT8_C(206), UINT8_C(  7), UINT8_C(232), UINT8_C( 56), UINT8_C( 34), UINT8_C( 40), UINT8_C(202),
        UINT8_C( 97), UINT8_C(207), UINT8_C(195), UINT8_C(134), UINT8_C( 70), UINT8_C(  9), UINT8_C( 55), UINT8_C(180) },
      { UINT16_C(    4), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 52), UINT8_C( 62), UINT8_C(227), UINT8_C(246), UINT8_C( 66), UINT8_C(161), UINT8_C( 89), UINT8_C( 10),
        UINT8_C(251), UINT8_C( 43), UINT8_C(182), UINT8_C(233), UINT8_C(182), UINT8_C(200), UINT8_C(240), UINT8_C(157) },
      { UINT8_C(150), UINT8_C(248), UINT8_C(134), UINT8_C(206), UINT8_C( 26), UINT8_C(174), UINT8_C(152), UINT8_C(123),
        UINT8_C(125), UINT8_C( 91), UINT8_C(  2), UINT8_C(195), UINT8_C(101), UINT8_C( 57), UINT8_C(119), UINT8_C(153) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(119), UINT8_C( 91), UINT8_C(144), UINT8_C(186), UINT8_C(252), UINT8_C(233), UINT8_C(196), UINT8_C(247),
        UINT8_C( 20), UINT8_C(122), UINT8_C(224), UINT8_C(203), UINT8_C( 66), UINT8_C(209), UINT8_C(104), UINT8_C(217) },
      { UINT8_C(201), UINT8_C(238), UINT8_C(167), UINT8_C(227), UINT8_C(156), UINT8_C( 64), UINT8_C( 95), UINT8_C( 25),
        UINT8_C(155), UINT8_C( 97), UINT8_C(220), UINT8_C(  0), UINT8_C(154), UINT8_C( 84), UINT8_C(154), UINT8_C( 17) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(175), UINT8_C( 42), UINT8_C(203), UINT8_C(171), UINT8_C( 19), UINT8_C(143), UINT8_C(162), UINT8_C( 39),
        UINT8_C(  9), UINT8_C(131), UINT8_C(242), UINT8_C( 76), UINT8_C( 84), UINT8_C( 91), UINT8_C( 37), UINT8_C( 29) },
      { UINT8_C( 73), UINT8_C(204), UINT8_C(  0), UINT8_C(230), UINT8_C( 12), UINT8_C( 95),    UINT8_MAX, UINT8_C(168),
        UINT8_C(192), UINT8_C(220), UINT8_C(168), UINT8_C( 90), UINT8_C( 48), UINT8_C( 66), UINT8_C(108), UINT8_C(223) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(108), UINT8_C( 55), UINT8_C(138), UINT8_C(127), UINT8_C(199), UINT8_C( 44), UINT8_C(167), UINT8_C(208),
        UINT8_C(175), UINT8_C(153), UINT8_C( 28), UINT8_C(  3), UINT8_C(244), UINT8_C( 65), UINT8_C( 32), UINT8_C( 62) },
      { UINT8_C( 14), UINT8_C( 33), UINT8_C( 36), UINT8_C( 26), UINT8_C(128), UINT8_C( 35), UINT8_C(194), UINT8_C( 65),
           UINT8_MAX, UINT8_C(107), UINT8_C(155), UINT8_C( 47), UINT8_C(173), UINT8_C(  7), UINT8_C( 14), UINT8_C( 26) },
      { UINT16_C(  128), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 63), UINT8_C(152), UINT8_C(153), UINT8_C(  6), UINT8_C(197), UINT8_C( 64), UINT8_C(214), UINT8_C(116),
        UINT8_C(218), UINT8_C(243), UINT8_C(120), UINT8_C(206), UINT8_C( 52), UINT8_C(152), UINT8_C( 12), UINT8_C( 66) },
      { UINT8_C(185), UINT8_C( 48), UINT8_C( 93), UINT8_C( 58), UINT8_C( 84), UINT8_C( 31), UINT8_C(123), UINT8_C( 83),
        UINT8_C(138), UINT8_C( 22), UINT8_C(131), UINT8_C( 56), UINT8_C( 30), UINT8_C(145), UINT8_C( 82), UINT8_C( 93) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*2*/
    { { UINT8_C( 80), UINT8_C( 69), UINT8_C( 36), UINT8_C( 96), UINT8_C(225), UINT8_C(232), UINT8_C( 92), UINT8_C( 96),
        UINT8_C( 99), UINT8_C( 60), UINT8_C( 91), UINT8_C( 53), UINT8_C( 40), UINT8_C(174), UINT8_C( 40), UINT8_C( 54) },
      { UINT8_C( 48), UINT8_C(177), UINT8_C( 20), UINT8_C(  6), UINT8_C( 83), UINT8_C(  3), UINT8_C( 99), UINT8_C( 75),
        UINT8_C(175), UINT8_C(238), UINT8_C(204), UINT8_C( 96), UINT8_C( 96), UINT8_C(173), UINT8_C(176), UINT8_C(176) },
      { UINT16_C( 6208), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(242), UINT8_C(213), UINT8_C( 16), UINT8_C(211), UINT8_C(189), UINT8_C(108), UINT8_C( 51), UINT8_C( 32),
        UINT8_C(168), UINT8_C(142), UINT8_C( 85), UINT8_C(209), UINT8_C( 61), UINT8_C(126), UINT8_C(  7), UINT8_C(109) },
      { UINT8_C( 47), UINT8_C( 27), UINT8_C(115), UINT8_C(130), UINT8_C( 31), UINT8_C(214), UINT8_C(205), UINT8_C(206),
        UINT8_C(196), UINT8_C(153), UINT8_C( 47), UINT8_C( 36), UINT8_C( 70), UINT8_C(223), UINT8_C(212), UINT8_C( 57) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(180), UINT8_C(228), UINT8_C( 12), UINT8_C(113), UINT8_C( 81), UINT8_C( 64), UINT8_C(146), UINT8_C(249),
        UINT8_C(206), UINT8_C(231), UINT8_C(202), UINT8_C( 11), UINT8_C(101), UINT8_C(210), UINT8_C(120), UINT8_C(148) },
      { UINT8_C(237), UINT8_C(235), UINT8_C( 22), UINT8_C( 12), UINT8_C(194), UINT8_C(227), UINT8_C(219), UINT8_C(134),
        UINT8_C(124), UINT8_C( 10), UINT8_C(171), UINT8_C(195), UINT8_C(233), UINT8_C(127), UINT8_C(252), UINT8_C(158) },
      { UINT16_C(    8), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(100), UINT8_C(  8), UINT8_C( 15), UINT8_C(181), UINT8_C( 72), UINT8_C(161), UINT8_C(174), UINT8_C( 23),
        UINT8_C(137), UINT8_C(121), UINT8_C( 34), UINT8_C(238), UINT8_C( 75), UINT8_C(155), UINT8_C(131), UINT8_C( 56) },
      { UINT8_C(134), UINT8_C(153), UINT8_C( 69), UINT8_C( 72), UINT8_C(125), UINT8_C( 32), UINT8_C(207), UINT8_C(249),
        UINT8_C( 42), UINT8_C(122), UINT8_C(188), UINT8_C( 19), UINT8_C(249), UINT8_C(184), UINT8_C(177), UINT8_C( 93) },
      { UINT16_C(    8), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(193), UINT8_C(193), UINT8_C( 18), UINT8_C(  9), UINT8_C( 98), UINT8_C(193), UINT8_C( 32), UINT8_C(235),
        UINT8_C( 58), UINT8_C( 67), UINT8_C(218), UINT8_C(133), UINT8_C(222), UINT8_C( 93), UINT8_C(189), UINT8_C(100) },
      { UINT8_C(246), UINT8_C(  2), UINT8_C(173), UINT8_C(115), UINT8_C( 34), UINT8_C(124), UINT8_C(109), UINT8_C( 76),
        UINT8_C(246), UINT8_C( 41), UINT8_C( 96), UINT8_C(239), UINT8_C(226), UINT8_C( 17), UINT8_C( 77), UINT8_C(163) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(210), UINT8_C( 95), UINT8_C(172), UINT8_C( 53), UINT8_C( 32), UINT8_C(205), UINT8_C( 32), UINT8_C( 90),
        UINT8_C( 16), UINT8_C(250), UINT8_C(223), UINT8_C(238), UINT8_C( 87), UINT8_C(157), UINT8_C( 82), UINT8_C( 78) },
      { UINT8_C(159),    UINT8_MAX, UINT8_C(193), UINT8_C(194), UINT8_C(123), UINT8_C( 46), UINT8_C( 14), UINT8_C(113),
        UINT8_C( 88), UINT8_C(110), UINT8_C( 97), UINT8_C( 58), UINT8_C(128), UINT8_C(174), UINT8_C(221), UINT8_C( 82) },
      { UINT16_C(32768), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 13), UINT8_C(137), UINT8_C(135), UINT8_C( 46), UINT8_C( 86), UINT8_C(168), UINT8_C(136), UINT8_C(102),
        UINT8_C(162), UINT8_C(104), UINT8_C( 84), UINT8_C(250), UINT8_C(  5), UINT8_C(167), UINT8_C( 72), UINT8_C(164) },
      { UINT8_C(166), UINT8_C(  9), UINT8_C(102), UINT8_C( 34), UINT8_C( 56), UINT8_C(117), UINT8_C(147), UINT8_C(144),
        UINT8_C(227), UINT8_C(244), UINT8_C(202), UINT8_C( 99), UINT8_C(162), UINT8_C(167), UINT8_C(182), UINT8_C(176) },
      { UINT16_C(12292), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 48), UINT8_C( 61), UINT8_C(222), UINT8_C(135), UINT8_C(229), UINT8_C(102), UINT8_C(237), UINT8_C(136),
        UINT8_C(206), UINT8_C( 66), UINT8_C(130), UINT8_C(211), UINT8_C(233), UINT8_C(202), UINT8_C(120), UINT8_C(143) },
      { UINT8_C(211), UINT8_C(222), UINT8_C(177), UINT8_C( 11), UINT8_C( 83), UINT8_C( 69), UINT8_C(155), UINT8_C( 55),
        UINT8_C( 57), UINT8_C(101), UINT8_C(154), UINT8_C(220), UINT8_C( 12), UINT8_C( 80), UINT8_C(140), UINT8_C( 61) },
      { UINT16_C(32771), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(142), UINT8_C(106), UINT8_C(196), UINT8_C(115), UINT8_C(208), UINT8_C(177), UINT8_C(251), UINT8_C(159),
        UINT8_C(243), UINT8_C(125), UINT8_C(114), UINT8_C(220), UINT8_C( 71), UINT8_C(234), UINT8_C(108), UINT8_C( 27) },
      { UINT8_C(201), UINT8_C( 29), UINT8_C( 38), UINT8_C( 28), UINT8_C( 98), UINT8_C(194), UINT8_C( 83), UINT8_C(156),
        UINT8_C( 39), UINT8_C(238), UINT8_C(120), UINT8_C( 52), UINT8_C( 62), UINT8_C(  4), UINT8_C(113), UINT8_C(204) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(110), UINT8_C( 53), UINT8_C( 64), UINT8_C( 62), UINT8_C(230), UINT8_C( 59), UINT8_C(221), UINT8_C(218),
        UINT8_C(185), UINT8_C( 80), UINT8_C(182), UINT8_C(  0), UINT8_C( 58), UINT8_C( 34), UINT8_C( 27), UINT8_C(  3) },
      { UINT8_C( 64), UINT8_C( 66), UINT8_C( 32), UINT8_C(162), UINT8_C(  4), UINT8_C(115), UINT8_C( 62), UINT8_C( 43),
        UINT8_C( 97), UINT8_C(182), UINT8_C( 95), UINT8_C(160), UINT8_C(186), UINT8_C(208), UINT8_C(108), UINT8_C( 40) },
      { UINT16_C(  577), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*4*/
    { { UINT8_C(221), UINT8_C( 90), UINT8_C( 80), UINT8_C( 70), UINT8_C(194), UINT8_C(176), UINT8_C(239), UINT8_C( 38),
        UINT8_C( 47), UINT8_C(204), UINT8_C(241), UINT8_C(206), UINT8_C( 24), UINT8_C( 66), UINT8_C(210), UINT8_C(237) },
      {    UINT8_MAX, UINT8_C( 36), UINT8_C(185), UINT8_C(251), UINT8_C(227), UINT8_C( 43), UINT8_C(144), UINT8_C(179),
        UINT8_C(222), UINT8_C( 11), UINT8_C(  5), UINT8_C(179), UINT8_C( 75), UINT8_C( 63), UINT8_C(162), UINT8_C( 40) },
      { UINT16_C(63990), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(153), UINT8_C(243), UINT8_C(111), UINT8_C( 92), UINT8_C(163), UINT8_C( 94), UINT8_C(130), UINT8_C(210),
        UINT8_C( 42), UINT8_C(115), UINT8_C(161), UINT8_C( 66), UINT8_C(181), UINT8_C(115), UINT8_C( 47), UINT8_C(180) },

      { UINT8_C(151), UINT8_C(232), UINT8_C(176), UINT8_C(123), UINT8_C( 19), UINT8_C( 64), UINT8_C( 46), UINT8_C(241),
        UINT8_C( 76), UINT8_C( 51), UINT8_C(165), UINT8_C(151), UINT8_C(114), UINT8_C( 71), UINT8_C(192), UINT8_C( 12) },
      { UINT16_C(32751), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 58), UINT8_C( 47), UINT8_C(104), UINT8_C(221), UINT8_C(141), UINT8_C(234), UINT8_C(176), UINT8_C(183),
        UINT8_C( 94), UINT8_C( 81), UINT8_C(249), UINT8_C( 19), UINT8_C(196), UINT8_C( 40), UINT8_C(200), UINT8_C( 92) },
      { UINT8_C( 16), UINT8_C(120), UINT8_C(215), UINT8_C( 35), UINT8_C(184), UINT8_C(  5), UINT8_C( 21), UINT8_C(  4),
        UINT8_C( 56), UINT8_C(186), UINT8_C(156), UINT8_C(170), UINT8_C(  1), UINT8_C( 92), UINT8_C(182), UINT8_C( 60) },
      { UINT16_C(19990), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(139), UINT8_C( 30), UINT8_C( 25), UINT8_C( 24), UINT8_C(  9), UINT8_C(201), UINT8_C(207), UINT8_C(103),
        UINT8_C( 26), UINT8_C(200), UINT8_C(122), UINT8_C(223), UINT8_C(240), UINT8_C( 66), UINT8_C( 59), UINT8_C(  0) },
      { UINT8_C(186), UINT8_C( 18), UINT8_C( 35), UINT8_C(115), UINT8_C( 23), UINT8_C( 56), UINT8_C(119), UINT8_C( 79),
        UINT8_C(242), UINT8_C( 19), UINT8_C(249), UINT8_C(244), UINT8_C(111), UINT8_C(176), UINT8_C( 48), UINT8_C(250) },
      { UINT16_C(29439), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(206), UINT8_C( 73), UINT8_C( 18), UINT8_C(215), UINT8_C( 19), UINT8_C(225), UINT8_C( 62), UINT8_C( 45),
        UINT8_C(169), UINT8_C(185), UINT8_C( 12), UINT8_C(153), UINT8_C(251), UINT8_C( 71), UINT8_C(153), UINT8_C(182) },
      { UINT8_C( 89), UINT8_C(189), UINT8_C( 41), UINT8_C(112), UINT8_C(245), UINT8_C(160), UINT8_C(191), UINT8_C(232),
        UINT8_C(180), UINT8_C(185), UINT8_C(220), UINT8_C( 35), UINT8_C(105), UINT8_C( 12), UINT8_C( 30), UINT8_C( 55) },
      { UINT16_C(65391), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 85), UINT8_C( 48), UINT8_C( 15), UINT8_C(104), UINT8_C( 18), UINT8_C( 77), UINT8_C(150), UINT8_C(187),
        UINT8_C(  6), UINT8_C(162), UINT8_C( 85), UINT8_C(  2), UINT8_C(234), UINT8_C(238), UINT8_C(184), UINT8_C( 67) },
      { UINT8_C(171), UINT8_C(225), UINT8_C(180), UINT8_C(161), UINT8_C(129), UINT8_C(115), UINT8_C(137), UINT8_C( 53),
        UINT8_C( 44), UINT8_C(101), UINT8_C( 89), UINT8_C(149), UINT8_C(113), UINT8_C(119), UINT8_C(205), UINT8_C(198) },
      { UINT16_C(16381), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(167), UINT8_C(220), UINT8_C( 47), UINT8_C(185), UINT8_C( 41), UINT8_C(197), UINT8_C(117), UINT8_C( 48),
        UINT8_C(103), UINT8_C(202), UINT8_C( 50), UINT8_C( 81), UINT8_C(184), UINT8_C(234), UINT8_C(149), UINT8_C(100) },
      { UINT8_C(203), UINT8_C( 73), UINT8_C(  5), UINT8_C( 76), UINT8_C(188), UINT8_C(142), UINT8_C(130), UINT8_C(233),
        UINT8_C(243), UINT8_C(219), UINT8_C(126), UINT8_C(100), UINT8_C( 82), UINT8_C( 75), UINT8_C( 42), UINT8_C(249) },
      { UINT16_C(32507), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 39), UINT8_C( 89), UINT8_C(179), UINT8_C( 81), UINT8_C( 30), UINT8_C( 40), UINT8_C(129), UINT8_C(134),
        UINT8_C(242), UINT8_C(179), UINT8_C(215), UINT8_C(170), UINT8_C(157), UINT8_C(108), UINT8_C( 14), UINT8_C(104) },
      { UINT8_C(181), UINT8_C( 19), UINT8_C(180), UINT8_C(114), UINT8_C(161), UINT8_C( 54), UINT8_C( 91), UINT8_C(148),
        UINT8_C( 17), UINT8_C(217), UINT8_C(248), UINT8_C( 99), UINT8_C( 37), UINT8_C( 35), UINT8_C( 93), UINT8_C( 76) },
      { UINT16_C(63842), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(124), UINT8_C( 16), UINT8_C(157), UINT8_C(155), UINT8_C( 56), UINT8_C( 30), UINT8_C( 33), UINT8_C( 42),
        UINT8_C(209), UINT8_C(248), UINT8_C(212), UINT8_C(110), UINT8_C(101), UINT8_C(227), UINT8_C(214), UINT8_C( 26) },
      { UINT8_C(246), UINT8_C(139), UINT8_C(140), UINT8_C(152), UINT8_C(193), UINT8_C(231), UINT8_C( 44), UINT8_C(211),
        UINT8_C(193), UINT8_C( 37), UINT8_C( 54), UINT8_C(230), UINT8_C( 72), UINT8_C(147), UINT8_C( 50), UINT8_C(196) },
      { UINT16_C(43967), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(163), UINT8_C(208), UINT8_C( 95), UINT8_C(219), UINT8_C(238), UINT8_C(128), UINT8_C(  5), UINT8_C(192),
        UINT8_C(121), UINT8_C(218), UINT8_C( 46), UINT8_C(222), UINT8_C(189), UINT8_C(  5), UINT8_C(248), UINT8_C(179) },
      { UINT8_C(144), UINT8_C(133), UINT8_C( 75), UINT8_C( 81), UINT8_C(108), UINT8_C(120), UINT8_C( 36), UINT8_C( 45),
        UINT8_C(157), UINT8_C( 91), UINT8_C( 19), UINT8_C(229), UINT8_C(238), UINT8_C( 70), UINT8_C(169), UINT8_C(146) },
      { UINT16_C(59391), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*6*/
    { { UINT8_C(108), UINT8_C( 40), UINT8_C(173), UINT8_C(140), UINT8_C(185), UINT8_C(236), UINT8_C( 68), UINT8_C(120),
        UINT8_C(105), UINT8_C(164), UINT8_C(130), UINT8_C(253), UINT8_C(198), UINT8_C( 62), UINT8_C(165), UINT8_C(158) },
      { UINT8_C(  1), UINT8_C(125), UINT8_C( 84), UINT8_C(142), UINT8_C(200), UINT8_C(112), UINT8_C(165), UINT8_C( 43),
        UINT8_C( 95), UINT8_C(139), UINT8_C(227), UINT8_C(191), UINT8_C(193), UINT8_C(165), UINT8_C(112), UINT8_C( 45) },
      { UINT16_C(65533), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(205), UINT8_C( 29), UINT8_C(185), UINT8_C(135), UINT8_C(  9), UINT8_C(253),    UINT8_MAX, UINT8_C(114),
        UINT8_C(161), UINT8_C(130), UINT8_C(111), UINT8_C(104), UINT8_C(192), UINT8_C( 21), UINT8_C(  6), UINT8_C(193) },
      { UINT8_C(146), UINT8_C( 90), UINT8_C( 79), UINT8_C( 90), UINT8_C(202), UINT8_C(245), UINT8_C(133), UINT8_C( 41),
        UINT8_C(128), UINT8_C(104), UINT8_C(232), UINT8_C( 65), UINT8_C( 13), UINT8_C( 88), UINT8_C(110), UINT8_C(219) },
      { UINT16_C(65214), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(118), UINT8_C( 39), UINT8_C( 98), UINT8_C(127), UINT8_C( 37), UINT8_C( 97), UINT8_C(242), UINT8_C(198),
        UINT8_C(227), UINT8_C( 97), UINT8_C( 46), UINT8_C(163), UINT8_C(118), UINT8_C( 52), UINT8_C(101), UINT8_C(  8) },
      { UINT8_C(142), UINT8_C(180), UINT8_C( 98), UINT8_C( 88), UINT8_C(169), UINT8_C(231), UINT8_C(130), UINT8_C( 42),
        UINT8_C( 80), UINT8_C(106), UINT8_C(107), UINT8_C( 93), UINT8_C(195), UINT8_C(218), UINT8_C( 56), UINT8_C( 57) },
      { UINT16_C(53164), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(  1), UINT8_C(154), UINT8_C(184), UINT8_C( 38), UINT8_C(252), UINT8_C(170), UINT8_C(237), UINT8_C(223),
        UINT8_C( 12), UINT8_C( 27), UINT8_C(131), UINT8_C(130), UINT8_C( 80), UINT8_C(232), UINT8_C(139), UINT8_C(222) },
      { UINT8_C(156), UINT8_C(237), UINT8_C( 55), UINT8_C( 70), UINT8_C(213), UINT8_C(185), UINT8_C(112), UINT8_C( 37),
        UINT8_C( 35), UINT8_C(219), UINT8_C(130), UINT8_C(230), UINT8_C(181), UINT8_C(187), UINT8_C( 31), UINT8_C(183) },
      { UINT16_C(64435), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 85), UINT8_C(216), UINT8_C(221), UINT8_C( 81), UINT8_C(130), UINT8_C(202), UINT8_C( 49), UINT8_C(142),
        UINT8_C(230), UINT8_C(180), UINT8_C( 17), UINT8_C( 54), UINT8_C(156), UINT8_C(156), UINT8_C( 20), UINT8_C( 56) },
      { UINT8_C(137), UINT8_C( 75), UINT8_C(126), UINT8_C( 94), UINT8_C(  4), UINT8_C(238), UINT8_C(131), UINT8_C( 40),
        UINT8_C(202), UINT8_C(  6), UINT8_C( 14), UINT8_C(127), UINT8_C(193), UINT8_C( 46), UINT8_C( 54), UINT8_C( 22) },
      { UINT16_C(63475), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(  6), UINT8_C( 20), UINT8_C(104), UINT8_C(136), UINT8_C(222), UINT8_C(153), UINT8_C( 23), UINT8_C(196),
        UINT8_C( 77), UINT8_C( 40), UINT8_C(250), UINT8_C(233), UINT8_C(196), UINT8_C( 15), UINT8_C( 33), UINT8_C( 77) },
      { UINT8_C( 90), UINT8_C(160), UINT8_C(172), UINT8_C( 95), UINT8_C(142), UINT8_C( 47), UINT8_C(135), UINT8_C( 88),
        UINT8_C( 53), UINT8_C(149), UINT8_C(216), UINT8_C(246), UINT8_C(195), UINT8_C( 14), UINT8_C( 13), UINT8_C(201) },
      { UINT16_C(60704), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 34), UINT8_C(117), UINT8_C( 82), UINT8_C(  1), UINT8_C( 14), UINT8_C(105), UINT8_C(197), UINT8_C( 91),
        UINT8_C(145), UINT8_C(192), UINT8_C( 68), UINT8_C( 85), UINT8_C(207), UINT8_C(101), UINT8_C(162), UINT8_C( 41) },
      { UINT8_C(  5), UINT8_C( 78), UINT8_C(136), UINT8_C(148), UINT8_C(126), UINT8_C( 15), UINT8_C(236), UINT8_C(179),
        UINT8_C(165), UINT8_C(196), UINT8_C(170), UINT8_C(104), UINT8_C(211), UINT8_C(183), UINT8_C( 50), UINT8_C(245) },
      { UINT16_C(65515), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 44), UINT8_C(132), UINT8_C(246), UINT8_C( 58), UINT8_C(237), UINT8_C(188), UINT8_C(149), UINT8_C(126),
        UINT8_C(124), UINT8_C(217), UINT8_C(211), UINT8_C( 75), UINT8_C( 62), UINT8_C(117), UINT8_C(116), UINT8_C( 68) },
      { UINT8_C(196), UINT8_C(253), UINT8_C(216), UINT8_C( 66), UINT8_C( 12), UINT8_C(196), UINT8_C(245), UINT8_C(177),
        UINT8_C(137), UINT8_C(159), UINT8_C( 26), UINT8_C( 92), UINT8_C( 86), UINT8_C( 76), UINT8_C( 81), UINT8_C(130) },
      { UINT16_C(32511), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(208), UINT8_C( 72), UINT8_C(188), UINT8_C(189), UINT8_C(  4), UINT8_C( 81), UINT8_C( 59), UINT8_C(128),
        UINT8_C( 42), UINT8_C( 14), UINT8_C(203), UINT8_C(105), UINT8_C(131), UINT8_C( 63), UINT8_C(173), UINT8_C( 71) },
      { UINT8_C( 60), UINT8_C(133), UINT8_C(137), UINT8_C( 73), UINT8_C( 73), UINT8_C(127), UINT8_C(250), UINT8_C(210),
        UINT8_C( 30), UINT8_C( 20), UINT8_C( 46), UINT8_C(117), UINT8_C( 96), UINT8_C(128), UINT8_C(247), UINT8_C( 48) },
      { UINT16_C(55263), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(200), UINT8_C(180), UINT8_C(237), UINT8_C(204), UINT8_C(  5), UINT8_C( 40), UINT8_C( 76), UINT8_C( 48),
        UINT8_C( 54), UINT8_C( 23), UINT8_C(153), UINT8_C(186), UINT8_C( 86), UINT8_C( 70), UINT8_C(  1), UINT8_C(147) },
      { UINT8_C(203), UINT8_C(139), UINT8_C(220), UINT8_C( 20), UINT8_C( 10), UINT8_C(214), UINT8_C(231), UINT8_C( 40),
        UINT8_C(235), UINT8_C( 21), UINT8_C(157), UINT8_C( 75), UINT8_C(149), UINT8_C(149), UINT8_C(124), UINT8_C( 93) },
      { UINT16_C( 1688), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*8*/
    { { UINT8_C(161), UINT8_C(134), UINT8_C(134), UINT8_C( 79), UINT8_C(117), UINT8_C( 29), UINT8_C(112), UINT8_C(  3),
        UINT8_C(155), UINT8_C(  1), UINT8_C( 99), UINT8_C(129), UINT8_C(110), UINT8_C(167), UINT8_C(241), UINT8_C(  5) },
      { UINT8_C(161), UINT8_C( 46), UINT8_C(246), UINT8_C(170), UINT8_C(117), UINT8_C(104), UINT8_C(112), UINT8_C(120),
        UINT8_C( 76), UINT8_C(101), UINT8_C( 47), UINT8_C(248), UINT8_C( 91), UINT8_C( 29), UINT8_C(241), UINT8_C(246) },
      { UINT16_C(16465), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(163), UINT8_C(105), UINT8_C( 69), UINT8_C( 26), UINT8_C(137), UINT8_C(181), UINT8_C( 30), UINT8_C( 33),
        UINT8_C(183), UINT8_C(129), UINT8_C(206), UINT8_C( 37), UINT8_C( 40), UINT8_C(147), UINT8_C( 42), UINT8_C(201) },
      { UINT8_C(193), UINT8_C( 33), UINT8_C(115), UINT8_C( 55), UINT8_C(137), UINT8_C(214), UINT8_C(175), UINT8_C( 33),
        UINT8_C( 59), UINT8_C(223), UINT8_C(206), UINT8_C(150), UINT8_C(252), UINT8_C(177), UINT8_C(140), UINT8_C(159) },
      { UINT16_C( 1168), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 26), UINT8_C(209), UINT8_C(185), UINT8_C(160), UINT8_C(134), UINT8_C(158), UINT8_C(193), UINT8_C( 61),
        UINT8_C( 88), UINT8_C(220), UINT8_C( 98), UINT8_C(128), UINT8_C(141), UINT8_C(141), UINT8_C( 74), UINT8_C(184) },
      { UINT8_C(174), UINT8_C(189), UINT8_C(239), UINT8_C( 55), UINT8_C(147), UINT8_C(158), UINT8_C( 13), UINT8_C(206),
        UINT8_C(125), UINT8_C(220), UINT8_C(100), UINT8_C(121), UINT8_C(141), UINT8_C(240), UINT8_C( 24), UINT8_C(168) },
      { UINT16_C( 4640), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(193), UINT8_C(210), UINT8_C( 72), UINT8_C( 72), UINT8_C(169), UINT8_C( 10), UINT8_C(133), UINT8_C(  2),
        UINT8_C(109), UINT8_C(232), UINT8_C(130), UINT8_C(100), UINT8_C(117), UINT8_C(204), UINT8_C( 28), UINT8_C( 35) },
      { UINT8_C(138), UINT8_C( 11), UINT8_C( 90), UINT8_C( 29), UINT8_C(169), UINT8_C(104), UINT8_C(236), UINT8_C( 39),
        UINT8_C( 68), UINT8_C( 80), UINT8_C(160), UINT8_C(209), UINT8_C( 65), UINT8_C(185), UINT8_C(121), UINT8_C(  2) },
      { UINT16_C(   16), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(139), UINT8_C(194), UINT8_C( 74), UINT8_C( 52), UINT8_C(204), UINT8_C(208), UINT8_C( 54), UINT8_C( 57),
        UINT8_C(184), UINT8_C(185), UINT8_C(157), UINT8_C( 45), UINT8_C(133), UINT8_C(185), UINT8_C( 80), UINT8_C( 15) },
      { UINT8_C(196), UINT8_C(170), UINT8_C( 45), UINT8_C(110), UINT8_C( 18), UINT8_C( 25), UINT8_C(149), UINT8_C( 86),
        UINT8_C(105), UINT8_C( 53), UINT8_C( 40), UINT8_C(170), UINT8_C(238), UINT8_C(161), UINT8_C(173), UINT8_C(121) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 99), UINT8_C(247), UINT8_C(174), UINT8_C( 47), UINT8_C(199), UINT8_C(135), UINT8_C(105), UINT8_C(127),
        UINT8_C(189), UINT8_C(  6), UINT8_C(172), UINT8_C(171), UINT8_C(192), UINT8_C(252), UINT8_C( 50), UINT8_C(132) },
      { UINT8_C(167), UINT8_C( 95), UINT8_C(242), UINT8_C(185), UINT8_C(120), UINT8_C(135), UINT8_C( 16), UINT8_C(226),
        UINT8_C(189), UINT8_C( 56), UINT8_C(140), UINT8_C(171), UINT8_C(217), UINT8_C( 57), UINT8_C( 37), UINT8_C( 61) },
      { UINT16_C( 2336), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 49), UINT8_C(211), UINT8_C(108), UINT8_C(248), UINT8_C(183), UINT8_C(213), UINT8_C(120), UINT8_C( 85),
        UINT8_C(201), UINT8_C(241), UINT8_C(120), UINT8_C(156), UINT8_C( 33), UINT8_C(170), UINT8_C( 32), UINT8_C(200) },
      { UINT8_C( 10), UINT8_C( 19), UINT8_C(129), UINT8_C(130), UINT8_C(154), UINT8_C(145), UINT8_C(100), UINT8_C(200),
        UINT8_C(201), UINT8_C(241), UINT8_C(  3), UINT8_C(163), UINT8_C( 42), UINT8_C( 40), UINT8_C(224), UINT8_C( 91) },
      { UINT16_C(  768), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(251), UINT8_C( 76), UINT8_C( 84), UINT8_C(178), UINT8_C( 34), UINT8_C(204), UINT8_C(  7), UINT8_C(254),
        UINT8_C(240), UINT8_C(194), UINT8_C(154), UINT8_C( 17), UINT8_C( 42), UINT8_C(186), UINT8_C(217), UINT8_C( 52) },
      { UINT8_C(205), UINT8_C( 91), UINT8_C(182), UINT8_C(104), UINT8_C(236), UINT8_C( 27), UINT8_C(191), UINT8_C(182),
        UINT8_C( 12), UINT8_C(194), UINT8_C( 89), UINT8_C( 54), UINT8_C(234), UINT8_C(186), UINT8_C(146), UINT8_C(229) },
      { UINT16_C( 8704), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(133), UINT8_C(230), UINT8_C(152), UINT8_C(213), UINT8_C(178), UINT8_C(159), UINT8_C(139), UINT8_C(162),
        UINT8_C( 31), UINT8_C( 63), UINT8_C(180), UINT8_C( 73), UINT8_C(250), UINT8_C(141), UINT8_C(125), UINT8_C(199) },
      { UINT8_C(232), UINT8_C( 51), UINT8_C( 47), UINT8_C(213), UINT8_C( 78), UINT8_C(239), UINT8_C(139), UINT8_C( 90),
        UINT8_C(177), UINT8_C(228), UINT8_C(145), UINT8_C(156), UINT8_C( 29), UINT8_C( 35), UINT8_C(129), UINT8_C(162) },
      { UINT16_C(   72), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(  9), UINT8_C( 32), UINT8_C( 32), UINT8_C(187), UINT8_C(185), UINT8_C( 32), UINT8_C( 93), UINT8_C(216),
        UINT8_C( 47), UINT8_C( 17), UINT8_C( 33), UINT8_C( 41), UINT8_C(159), UINT8_C(158), UINT8_C(240), UINT8_C(135) },
      { UINT8_C(209), UINT8_C( 32), UINT8_C( 92), UINT8_C( 32), UINT8_C( 15), UINT8_C(231), UINT8_C(122), UINT8_C(192),
        UINT8_C(203), UINT8_C( 11), UINT8_C( 92), UINT8_C(232), UINT8_C( 46), UINT8_C(222), UINT8_C(139), UINT8_C(135) },
      { UINT16_C(32770), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, }
  };

#define test_case_template(start, step, imm8) ({                        \
  for(size_t i = (start); i < (start) + (step); i++){                   \
    easysimd__m128i a = easysimd_mm_loadu_epi8((void *)&(test_vec[i].a[0]));  \
    easysimd__m128i b = easysimd_mm_loadu_epi8((void *)&(test_vec[i].b[0]));  \
    easysimd__m128i r = easysimd_mm_loadu_epi8((void *)&(test_vec[i].r[0]));  \
    easysimd__m128i ret = easysimd_mm_cmpistrm(a, b, (imm8));                 \
    easysimd_assert_m128i_u16(r, ==, ret);}                                \
})

  test_case_template( 0, 10, 0);
  test_case_template(10, 10, 2);
  test_case_template(20, 10, 4);
  test_case_template(30, 10, 6);
  test_case_template(40, 10, 8);

#undef test_case_template


#ifdef EASYSIMD_ENABLE_TEST_PERF
  for(size_t i = 40; i < 50; i++){
    easysimd__m128i a = easysimd_mm_loadu_epi8((void *)&(test_vec[i].a[0]));
    easysimd__m128i b = easysimd_mm_loadu_epi8((void *)&(test_vec[i].b[0]));
    easysimd__m128i r = easysimd_mm_loadu_epi8((void *)&(test_vec[i].r[0]));
    easysimd__m128i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm_cmpistrm(a, b, 8);
    }
    EASYSIMD_TEST_PERF_END("_mm_cmpistrm 8bit on easysimd");
    easysimd_assert_m128i_u16(r, ==, ret);
  }

#endif
  return 0;
 
}


easysimd__m128i
easysimd_mm_load_epu16(void const* mem_addr) {
  easysimd__m128i_private r_;
  for(size_t i = 0; i < sizeof(r_.u16) / sizeof(r_.u16[0]); i++){
    r_.u16[i] = *(((uint16_t *)mem_addr) + i);
  }
  return easysimd__m128i_from_private(r_);
}

static int
test_easysimd_mm_cmpistrm_16bit(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    uint16_t a[8];
    uint16_t b[8];
    uint16_t r[8];
  } test_vec[] = {
/*1*/
    { { UINT16_C(31970), UINT16_C(28414), UINT16_C(64008), UINT16_C(40360), UINT16_C( 1812), UINT16_C(19280), UINT16_C(20652), UINT16_C(31490) },
      { UINT16_C(31970), UINT16_C( 6881), UINT16_C(42202), UINT16_C(59206), UINT16_C(61054), UINT16_C(24518), UINT16_C(40891), UINT16_C(40651) },
      { UINT16_C(    1), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(51483), UINT16_C( 9228), UINT16_C(59499), UINT16_C(55233), UINT16_C( 4796), UINT16_C(26659), UINT16_C(59499), UINT16_C( 7651) },
      { UINT16_C(50272), UINT16_C(51483), UINT16_C(32104), UINT16_C(58914), UINT16_C(59499), UINT16_C(10053), UINT16_C( 7651), UINT16_C(59499) },
      { UINT16_C(  210), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(53722), UINT16_C(40391), UINT16_C(34950), UINT16_C(17013), UINT16_C(39066), UINT16_C(64938), UINT16_C(36285), UINT16_C( 7450) },
      { UINT16_C(20817), UINT16_C(47448), UINT16_C(53722), UINT16_C(15008), UINT16_C(58723), UINT16_C(60001), UINT16_C( 9974), UINT16_C(53389) },
      { UINT16_C(    4), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(21751), UINT16_C(32109), UINT16_C(58077), UINT16_C(30655), UINT16_C(27002), UINT16_C(14196), UINT16_C(36598), UINT16_C(18261) },
      { UINT16_C(44511), UINT16_C(44545), UINT16_C(41256), UINT16_C(21751), UINT16_C(18822), UINT16_C(31861), UINT16_C(  879), UINT16_C(26188) },
      { UINT16_C(    8), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(47703), UINT16_C(13540), UINT16_C(41884), UINT16_C( 6060), UINT16_C( 8205), UINT16_C(  846), UINT16_C(41903), UINT16_C(36427) },
      { UINT16_C(19537), UINT16_C(31036), UINT16_C( 9453), UINT16_C(29444), UINT16_C(31085), UINT16_C(47703), UINT16_C(15484), UINT16_C(54339) },
      { UINT16_C(   32), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(10230), UINT16_C(37640), UINT16_C(46282), UINT16_C(55210), UINT16_C(63701), UINT16_C(34011), UINT16_C( 9884), UINT16_C(60690) },
      { UINT16_C(20338), UINT16_C(46282), UINT16_C(27251), UINT16_C(57810), UINT16_C(49891), UINT16_C(24765), UINT16_C(  255), UINT16_C(62772) },
      { UINT16_C(    2), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(15399), UINT16_C(62088), UINT16_C(13041), UINT16_C(50889), UINT16_C(42027), UINT16_C(51018), UINT16_C(23754), UINT16_C(15540) },
      { UINT16_C( 6827), UINT16_C( 8091), UINT16_C(28292), UINT16_C(51018), UINT16_C(48432), UINT16_C(12231), UINT16_C(64446), UINT16_C(51018) },
      { UINT16_C(  136), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(44344), UINT16_C(10711), UINT16_C(41440), UINT16_C( 3055), UINT16_C(14661), UINT16_C( 4306), UINT16_C(10711), UINT16_C(16716) },
      { UINT16_C(59552), UINT16_C( 9312), UINT16_C(24662), UINT16_C(34443), UINT16_C(10711), UINT16_C(56246), UINT16_C(56142), UINT16_C(34497) },
      { UINT16_C(   16), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(39048), UINT16_C(26799), UINT16_C(40505), UINT16_C(32627), UINT16_C(26799), UINT16_C(28047), UINT16_C(56267), UINT16_C(27566) },
      { UINT16_C(26799), UINT16_C( 6543), UINT16_C( 7022), UINT16_C(26799), UINT16_C(22126), UINT16_C(48231), UINT16_C(10289), UINT16_C(26799) },
      { UINT16_C(  137), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(62144), UINT16_C(64034), UINT16_C(38289), UINT16_C(26745), UINT16_C( 2267), UINT16_C(42709), UINT16_C(33763), UINT16_C(42770) },
      { UINT16_C(41361), UINT16_C(65472), UINT16_C(24764), UINT16_C(10891), UINT16_C(62134), UINT16_C(59367), UINT16_C(10778), UINT16_C(55969) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*3*/
    { { UINT16_C( 9554), UINT16_C(51291), UINT16_C( 3966), UINT16_C(37595), UINT16_C(36496), UINT16_C(45813), UINT16_C(16907), UINT16_C(63598) },
      { UINT16_C(13805), UINT16_C( 9554), UINT16_C( 9554), UINT16_C(52334), UINT16_C(29630), UINT16_C(40341), UINT16_C( 9554), UINT16_C(59076) },
      { UINT16_C(   70), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C( 8431), UINT16_C(28078), UINT16_C(35119), UINT16_C(49407), UINT16_C(62744), UINT16_C( 9074), UINT16_C(57399), UINT16_C( 9243) },
      { UINT16_C(12822), UINT16_C(49407), UINT16_C(26922), UINT16_C(59638), UINT16_C(35804), UINT16_C(49407), UINT16_C(19029), UINT16_C(17750) },
      { UINT16_C(   34), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C( 1386), UINT16_C(44655), UINT16_C(45710), UINT16_C(42585), UINT16_C(52391), UINT16_C(57033), UINT16_C(44655), UINT16_C(44655) },
      { UINT16_C(64790), UINT16_C(16621), UINT16_C(58214), UINT16_C(17192), UINT16_C(44655), UINT16_C(50355), UINT16_C( 2808), UINT16_C(25097) },
      { UINT16_C(   16), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(48143), UINT16_C(40443), UINT16_C(21870), UINT16_C( 5444), UINT16_C( 3361), UINT16_C(52723), UINT16_C(62962), UINT16_C( 2192) },
      { UINT16_C(62962), UINT16_C(62962), UINT16_C(29024), UINT16_C(53147), UINT16_C(62962), UINT16_C( 6036), UINT16_C(40281), UINT16_C(62962) },
      { UINT16_C(  147), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(30041), UINT16_C(50949), UINT16_C(39621), UINT16_C(60380), UINT16_C(53079), UINT16_C(39621), UINT16_C(18628), UINT16_C(46673) },
      { UINT16_C(39621), UINT16_C( 9743), UINT16_C(43532), UINT16_C(46673), UINT16_C(35321), UINT16_C(21059), UINT16_C(39621), UINT16_C(32954) },
      { UINT16_C(   73), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(49201), UINT16_C(64328), UINT16_C( 9225), UINT16_C(24806), UINT16_C(49201), UINT16_C(47273), UINT16_C(64487), UINT16_C(44399) },
      { UINT16_C(49201), UINT16_C(49201), UINT16_C(51240), UINT16_C( 8909), UINT16_C( 4178), UINT16_C(31092), UINT16_C(49201), UINT16_C(65273) },
      { UINT16_C(   67), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(16879), UINT16_C(63737), UINT16_C(57446), UINT16_C(23129), UINT16_C(29677), UINT16_C(26130), UINT16_C(16879), UINT16_C(37651) },
      { UINT16_C(59135), UINT16_C(16879), UINT16_C(  431), UINT16_C(  330), UINT16_C(48657), UINT16_C(56954), UINT16_C(29677), UINT16_C(56540) },
      { UINT16_C(   66), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(54709), UINT16_C( 7125), UINT16_C(62869), UINT16_C(13429), UINT16_C(34608), UINT16_C(11931), UINT16_C(62869), UINT16_C( 2241) },
      { UINT16_C(62869), UINT16_C(17456), UINT16_C(31479), UINT16_C(34608), UINT16_C(48953), UINT16_C( 9958), UINT16_C(49714), UINT16_C(59139) },
      { UINT16_C(    9), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(55448), UINT16_C(19714), UINT16_C(30470), UINT16_C(13954), UINT16_C( 7679), UINT16_C( 2148), UINT16_C( 9675), UINT16_C(24592) },
      { UINT16_C(13954), UINT16_C( 4772), UINT16_C(59835), UINT16_C(24592), UINT16_C(  424), UINT16_C(13954), UINT16_C( 7875), UINT16_C(24592) },
      { UINT16_C(  169), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(50678), UINT16_C(64681), UINT16_C(11068), UINT16_C(15154), UINT16_C(38728), UINT16_C( 4931), UINT16_C(21692), UINT16_C(55156) },
      { UINT16_C( 6293), UINT16_C(64681), UINT16_C( 1026), UINT16_C(43589), UINT16_C(38728), UINT16_C(21692), UINT16_C(18558), UINT16_C(29732) },
      { UINT16_C(   50), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*5*/
    { { UINT16_C(56195), UINT16_C(10798), UINT16_C(23389), UINT16_C(53101), UINT16_C(16867), UINT16_C(11410), UINT16_C(38283), UINT16_C(15567) },
      { UINT16_C(18998), UINT16_C(49773), UINT16_C(58607), UINT16_C(19052), UINT16_C(61613), UINT16_C(48105), UINT16_C(29700), UINT16_C(34586) },
      { UINT16_C(  226), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(18511), UINT16_C(44209), UINT16_C( 8099), UINT16_C(34683), UINT16_C( 3424), UINT16_C(60339), UINT16_C(33698), UINT16_C(55335) },
      { UINT16_C(38349), UINT16_C(48282), UINT16_C( 1657), UINT16_C( 9735), UINT16_C(61686), UINT16_C(64225), UINT16_C(64356), UINT16_C(45954) },
      { UINT16_C(  139), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(13124), UINT16_C(59231), UINT16_C(55890), UINT16_C(45678), UINT16_C( 8936), UINT16_C(35485), UINT16_C(50597), UINT16_C(29283) },
      { UINT16_C(64858), UINT16_C(54062), UINT16_C(13572), UINT16_C(64249), UINT16_C(56102), UINT16_C(35573), UINT16_C(30678), UINT16_C( 6718) },
      { UINT16_C(  118), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(40362), UINT16_C(64770), UINT16_C(28792), UINT16_C(24751), UINT16_C(19858), UINT16_C(14314), UINT16_C(19730), UINT16_C(27817) },
      { UINT16_C(55371), UINT16_C(20287), UINT16_C(14349), UINT16_C(13129), UINT16_C(15891), UINT16_C(60094), UINT16_C(64693), UINT16_C(24580) },
      { UINT16_C(  227), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C( 1689), UINT16_C( 4445), UINT16_C( 3191), UINT16_C( 2417), UINT16_C(23641), UINT16_C(27457), UINT16_C(60073), UINT16_C(62679) },
      { UINT16_C( 5826), UINT16_C(53315), UINT16_C(36175), UINT16_C(25091), UINT16_C(49611), UINT16_C(33100), UINT16_C(20925), UINT16_C(22497) },
      { UINT16_C(    8), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(15959), UINT16_C(52840), UINT16_C(55882), UINT16_C(42200), UINT16_C( 6454), UINT16_C(57103), UINT16_C(59139), UINT16_C(50900) },
      { UINT16_C( 6141), UINT16_C(19606), UINT16_C(39332), UINT16_C(28847), UINT16_C(64347), UINT16_C( 6385), UINT16_C(53836), UINT16_C(42095) },
      { UINT16_C(  206), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(55312), UINT16_C(23154), UINT16_C(19122), UINT16_C(59646), UINT16_C( 3683), UINT16_C(26567), UINT16_C(39925), UINT16_C(61997) },
      { UINT16_C(50099), UINT16_C(22335), UINT16_C(61020), UINT16_C(47047), UINT16_C(47337), UINT16_C(14032), UINT16_C(16266), UINT16_C(39642) },
      { UINT16_C(  255), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(19479), UINT16_C(51701), UINT16_C(62359), UINT16_C(64177), UINT16_C(30977), UINT16_C(63073), UINT16_C(36372), UINT16_C(51177) },
      { UINT16_C(10321), UINT16_C(44575), UINT16_C(58902), UINT16_C(65381), UINT16_C(13727), UINT16_C(10549), UINT16_C( 3957), UINT16_C(36036) },
      { UINT16_C(  134), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(47452), UINT16_C(62294), UINT16_C( 1964), UINT16_C(44781), UINT16_C(20352), UINT16_C(38308), UINT16_C(36317), UINT16_C(12124) },
      { UINT16_C(31669), UINT16_C(52189), UINT16_C(16994), UINT16_C(  459), UINT16_C(  120), UINT16_C(60714), UINT16_C(60944), UINT16_C(27769) },
      { UINT16_C(  231), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(53159), UINT16_C(21599), UINT16_C(19671), UINT16_C(22274), UINT16_C(42651), UINT16_C(31212), UINT16_C(18740), UINT16_C(59816) },
      { UINT16_C(34244), UINT16_C( 9909), UINT16_C(32967), UINT16_C(16167), UINT16_C(21120), UINT16_C(36908), UINT16_C(42560), UINT16_C(59644) },
      { UINT16_C(  245), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*7*/
    { { UINT16_C( 6556), UINT16_C(29418), UINT16_C( 1017), UINT16_C(45159), UINT16_C(49662), UINT16_C(60395), UINT16_C(13058), UINT16_C( 6217) },
      { UINT16_C(54770), UINT16_C(26584), UINT16_C(51271), UINT16_C(50853), UINT16_C(61949), UINT16_C(51867), UINT16_C( 4304), UINT16_C(27729) },
      { UINT16_C(  175), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(15401), UINT16_C( 9183), UINT16_C(17983), UINT16_C(15827), UINT16_C(48647), UINT16_C( 2344), UINT16_C(29169), UINT16_C(58401) },
      { UINT16_C(64070), UINT16_C(36171), UINT16_C(61634), UINT16_C(48980), UINT16_C(61409), UINT16_C(45449), UINT16_C(56319), UINT16_C(10526) },
      { UINT16_C(   93), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(64791), UINT16_C(22092), UINT16_C( 8003), UINT16_C(19091), UINT16_C(48093), UINT16_C(53075), UINT16_C(29740), UINT16_C(29363) },
      { UINT16_C(65134), UINT16_C(12544), UINT16_C(21742), UINT16_C(53232), UINT16_C(31299), UINT16_C(17281), UINT16_C(40789), UINT16_C(27756) },
      { UINT16_C(   39), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(47260), UINT16_C(57282), UINT16_C(21975), UINT16_C(46121), UINT16_C(31760), UINT16_C(15491), UINT16_C(14064), UINT16_C(24494) },
      { UINT16_C(44596), UINT16_C( 8848), UINT16_C(32770), UINT16_C(18162), UINT16_C(29690), UINT16_C(20361), UINT16_C(62738), UINT16_C(44731) },
      { UINT16_C(   40), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(32173), UINT16_C(33933), UINT16_C(46802), UINT16_C(57912), UINT16_C(48178), UINT16_C( 8734), UINT16_C(52722), UINT16_C(10113) },
      { UINT16_C( 4475), UINT16_C(32329), UINT16_C(15250), UINT16_C(36036), UINT16_C(19886), UINT16_C(49372), UINT16_C(38722), UINT16_C(61294) },
      { UINT16_C(  161), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(64277), UINT16_C(59251), UINT16_C(43953), UINT16_C(58314), UINT16_C(59495), UINT16_C(23046), UINT16_C(34741), UINT16_C(12673) },
      { UINT16_C(51865), UINT16_C(11183), UINT16_C(29446), UINT16_C(46263), UINT16_C(37824), UINT16_C(  629), UINT16_C(58155), UINT16_C(16625) },
      { UINT16_C(  251), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(25823), UINT16_C(36903), UINT16_C(61711), UINT16_C(30580), UINT16_C(31450), UINT16_C(36817), UINT16_C(20993), UINT16_C(39616) },
      { UINT16_C(28444), UINT16_C( 8901), UINT16_C(32226), UINT16_C(41687), UINT16_C(19472), UINT16_C(15268), UINT16_C(38191), UINT16_C( 3707) },
      { UINT16_C(  179), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(41977), UINT16_C( 2463), UINT16_C( 5012), UINT16_C(28288), UINT16_C(20877), UINT16_C(36606), UINT16_C(48803), UINT16_C(48937) },
      { UINT16_C(60974), UINT16_C( 4322), UINT16_C(47467), UINT16_C(31923), UINT16_C(22277), UINT16_C(13495), UINT16_C(13293), UINT16_C(58947) },
      { UINT16_C(  245), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(58070), UINT16_C(27375), UINT16_C(28661), UINT16_C(33497), UINT16_C(55232), UINT16_C(25360), UINT16_C(14741), UINT16_C(49955) },
      { UINT16_C( 1320), UINT16_C(37844), UINT16_C(34750), UINT16_C(49935), UINT16_C(51166), UINT16_C(52215), UINT16_C(15098), UINT16_C(53426) },
      { UINT16_C(   65), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(41244), UINT16_C( 4410), UINT16_C( 4881), UINT16_C(53651), UINT16_C(42218), UINT16_C(32821), UINT16_C(22749), UINT16_C( 1347) },
      { UINT16_C( 5981), UINT16_C( 7065), UINT16_C(43166), UINT16_C(32222), UINT16_C(54639), UINT16_C(26952), UINT16_C(64016), UINT16_C(11321) },
      { UINT16_C(   84), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*9*/
    { { UINT16_C(20194), UINT16_C(13489), UINT16_C(28148), UINT16_C(46609), UINT16_C(40781), UINT16_C(42334), UINT16_C(64675), UINT16_C(64675) },
      { UINT16_C(57383), UINT16_C(59409), UINT16_C(27037), UINT16_C(56767), UINT16_C(64675), UINT16_C(63245), UINT16_C(34097), UINT16_C( 5034) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(23763), UINT16_C(51016), UINT16_C(45134), UINT16_C( 6014), UINT16_C(61868), UINT16_C(16060), UINT16_C(23763), UINT16_C( 2515) },
      { UINT16_C(58473), UINT16_C(23763), UINT16_C(45134), UINT16_C(23763), UINT16_C(61868), UINT16_C(56808), UINT16_C(23763), UINT16_C(18928) },
      { UINT16_C(   84), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(14574), UINT16_C(47121), UINT16_C(36754), UINT16_C(35535), UINT16_C(42842), UINT16_C(19913), UINT16_C(39956), UINT16_C(32086) },
      { UINT16_C(14574), UINT16_C(52868), UINT16_C(26871), UINT16_C(41919), UINT16_C(42842), UINT16_C(53376), UINT16_C(28730), UINT16_C(10266) },
      { UINT16_C(   17), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(11177), UINT16_C(15328), UINT16_C(10720), UINT16_C( 9669), UINT16_C(36410), UINT16_C(20082), UINT16_C(10720), UINT16_C(43980) },
      { UINT16_C(20495), UINT16_C( 1657), UINT16_C(10720), UINT16_C( 5033), UINT16_C(10720), UINT16_C( 6883), UINT16_C(10720), UINT16_C(16963) },
      { UINT16_C(   68), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C( 9000), UINT16_C(57981), UINT16_C(17363), UINT16_C( 3335), UINT16_C(57981), UINT16_C(64604), UINT16_C(57981), UINT16_C(57981) },
      { UINT16_C( 8312), UINT16_C(57981), UINT16_C(65369), UINT16_C(14916), UINT16_C(57981), UINT16_C(49748), UINT16_C(57981), UINT16_C(19972) },
      { UINT16_C(   82), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(33467), UINT16_C(36400), UINT16_C(14533), UINT16_C(38555), UINT16_C(63409), UINT16_C(62354), UINT16_C(63409), UINT16_C(38979) },
      { UINT16_C(39514), UINT16_C(36400), UINT16_C(14533), UINT16_C(49901), UINT16_C(16950), UINT16_C(63409), UINT16_C(63409), UINT16_C(38057) },
      { UINT16_C(   70), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(55818), UINT16_C(53026), UINT16_C(48658), UINT16_C(50022), UINT16_C(63669), UINT16_C(54710), UINT16_C(64050), UINT16_C(35949) },
      { UINT16_C(13972), UINT16_C(11583), UINT16_C(11588), UINT16_C(31471), UINT16_C(29551), UINT16_C(18646), UINT16_C(32764), UINT16_C( 1757) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(65369), UINT16_C(27606), UINT16_C(65369), UINT16_C(29487), UINT16_C(58676), UINT16_C(26184), UINT16_C(65369), UINT16_C(29682) },
      { UINT16_C(13035), UINT16_C(65369), UINT16_C(65369), UINT16_C(52906), UINT16_C(32772), UINT16_C(   22), UINT16_C(65369), UINT16_C(22790) },
      { UINT16_C(   68), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(56563), UINT16_C(45253), UINT16_C(62488), UINT16_C(19747), UINT16_C(42528), UINT16_C(47539), UINT16_C(42528), UINT16_C( 3116) },
      { UINT16_C(52696), UINT16_C(14140), UINT16_C(62488), UINT16_C(25093), UINT16_C(42528), UINT16_C(26466), UINT16_C(42528), UINT16_C(  704) },
      { UINT16_C(   84), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(34117), UINT16_C(23986), UINT16_C(54905), UINT16_C(21418), UINT16_C(24129), UINT16_C(25100), UINT16_C(14340), UINT16_C(56430) },
      { UINT16_C(43526), UINT16_C(25619), UINT16_C( 6288), UINT16_C(21418), UINT16_C(10291), UINT16_C(25100), UINT16_C( 8080), UINT16_C(54596) },
      { UINT16_C(   40), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, }
  };

#define test_case_template(start, step, imm8) ({                         \
  for(size_t i = (start); i < (start) + (step); i++){                    \
    easysimd__m128i a = easysimd_mm_loadu_epi16((void *)&(test_vec[i].a[0]));  \
    easysimd__m128i b = easysimd_mm_loadu_epi16((void *)&(test_vec[i].b[0]));  \
    easysimd__m128i r = easysimd_mm_loadu_epi16((void *)&(test_vec[i].r[0]));  \
    easysimd__m128i ret = easysimd_mm_cmpistrm(a, b, (imm8));                  \
    easysimd_assert_m128i_u16(r, ==, ret);}                                 \
})

  test_case_template( 0, 10, 1);
  test_case_template(10, 10, 3);
  test_case_template(20, 10, 5);
  test_case_template(30, 10, 7);
  test_case_template(40, 10, 9);

#undef test_case_template

  return 0;
}


static int
test_easysimd_mm_cmpestrm_8bit(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    uint8_t  a[16];
    uint8_t  b[16];
    uint16_t r[8];
  } test_vec[] = {
/*0*/
    { { UINT8_C( 85), UINT8_C(110), UINT8_C(132), UINT8_C( 93), UINT8_C(179), UINT8_C( 55), UINT8_C(220), UINT8_C(173),
        UINT8_C(241), UINT8_C(143), UINT8_C(132), UINT8_C( 70), UINT8_C(174), UINT8_C( 40), UINT8_C(109), UINT8_C( 30) },
      { UINT8_C(234), UINT8_C(246), UINT8_C(123), UINT8_C(206), UINT8_C(182), UINT8_C( 51), UINT8_C(246), UINT8_C( 16),
        UINT8_C(132), UINT8_C(205), UINT8_C(196), UINT8_C(231), UINT8_C( 42), UINT8_C( 77), UINT8_C(177), UINT8_C(128) },
      { UINT16_C(  256), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(231), UINT8_C(174), UINT8_C(226), UINT8_C( 37), UINT8_C(128), UINT8_C(  9), UINT8_C( 71), UINT8_C(226),
        UINT8_C(107), UINT8_C(177), UINT8_C( 99), UINT8_C( 59), UINT8_C( 72), UINT8_C( 59), UINT8_C(111), UINT8_C( 56) },
      { UINT8_C(167), UINT8_C(179), UINT8_C(  7), UINT8_C(169), UINT8_C(226), UINT8_C(114), UINT8_C(206), UINT8_C(167),
        UINT8_C(134), UINT8_C(214), UINT8_C(150), UINT8_C( 32), UINT8_C(211), UINT8_C(145), UINT8_C(197), UINT8_C(187) },
      { UINT16_C(   16), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 63), UINT8_C(167), UINT8_C(224), UINT8_C(191), UINT8_C(176), UINT8_C( 40), UINT8_C(161), UINT8_C( 27),
        UINT8_C(217), UINT8_C(  5), UINT8_C( 87), UINT8_C( 33), UINT8_C( 64), UINT8_C(198), UINT8_C( 90), UINT8_C(231) },
      { UINT8_C(121), UINT8_C( 97), UINT8_C(145), UINT8_C( 91), UINT8_C(211), UINT8_C( 95), UINT8_C(  2), UINT8_C( 89),
        UINT8_C( 53), UINT8_C(152), UINT8_C(121), UINT8_C(  8), UINT8_C( 41), UINT8_C( 63), UINT8_C(195), UINT8_C(104) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(230), UINT8_C(164), UINT8_C( 39), UINT8_C(151), UINT8_C(204), UINT8_C(201), UINT8_C(178), UINT8_C(165),
        UINT8_C(206), UINT8_C(  9), UINT8_C(199), UINT8_C( 14), UINT8_C(207), UINT8_C( 33), UINT8_C(246), UINT8_C( 72) },
      { UINT8_C(130), UINT8_C(135), UINT8_C(163), UINT8_C( 85), UINT8_C(230), UINT8_C(165), UINT8_C(174), UINT8_C( 27),
        UINT8_C( 62), UINT8_C( 40), UINT8_C( 35), UINT8_C(103), UINT8_C(103), UINT8_C(231), UINT8_C(208), UINT8_C( 77) },
      { UINT16_C(   16), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(110), UINT8_C( 43), UINT8_C(  7), UINT8_C( 47), UINT8_C(194), UINT8_C(  4), UINT8_C(189), UINT8_C( 98),
        UINT8_C(199), UINT8_C( 90), UINT8_C(210), UINT8_C(172), UINT8_C(237), UINT8_C(139), UINT8_C( 18), UINT8_C(  7) },
      { UINT8_C(231), UINT8_C(206), UINT8_C(  7), UINT8_C(232), UINT8_C( 56), UINT8_C( 34), UINT8_C( 40), UINT8_C(202),
        UINT8_C( 97), UINT8_C(207), UINT8_C(195), UINT8_C(134), UINT8_C( 70), UINT8_C(  9), UINT8_C( 55), UINT8_C(180) },
      { UINT16_C(    4), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 52), UINT8_C( 62), UINT8_C(227), UINT8_C(246), UINT8_C( 66), UINT8_C(161), UINT8_C( 89), UINT8_C( 10),
        UINT8_C(251), UINT8_C( 43), UINT8_C(182), UINT8_C(233), UINT8_C(182), UINT8_C(200), UINT8_C(240), UINT8_C(157) },
      { UINT8_C(150), UINT8_C(248), UINT8_C(134), UINT8_C(206), UINT8_C( 26), UINT8_C(174), UINT8_C(152), UINT8_C(123),
        UINT8_C(125), UINT8_C( 91), UINT8_C(  2), UINT8_C(195), UINT8_C(101), UINT8_C( 57), UINT8_C(119), UINT8_C(153) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(119), UINT8_C( 91), UINT8_C(144), UINT8_C(186), UINT8_C(252), UINT8_C(233), UINT8_C(196), UINT8_C(247),
        UINT8_C( 20), UINT8_C(122), UINT8_C(224), UINT8_C(203), UINT8_C( 66), UINT8_C(209), UINT8_C(104), UINT8_C(217) },
      { UINT8_C(201), UINT8_C(238), UINT8_C(167), UINT8_C(227), UINT8_C(156), UINT8_C( 64), UINT8_C( 95), UINT8_C( 25),
        UINT8_C(155), UINT8_C( 97), UINT8_C(220), UINT8_C(  0), UINT8_C(154), UINT8_C( 84), UINT8_C(154), UINT8_C( 17) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(175), UINT8_C( 42), UINT8_C(203), UINT8_C(171), UINT8_C( 19), UINT8_C(143), UINT8_C(162), UINT8_C( 39),
        UINT8_C(  9), UINT8_C(131), UINT8_C(242), UINT8_C( 76), UINT8_C( 84), UINT8_C( 91), UINT8_C( 37), UINT8_C( 29) },
      { UINT8_C( 73), UINT8_C(204), UINT8_C(  0), UINT8_C(230), UINT8_C( 12), UINT8_C( 95),    UINT8_MAX, UINT8_C(168),
        UINT8_C(192), UINT8_C(220), UINT8_C(168), UINT8_C( 90), UINT8_C( 48), UINT8_C( 66), UINT8_C(108), UINT8_C(223) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(108), UINT8_C( 55), UINT8_C(138), UINT8_C(127), UINT8_C(199), UINT8_C( 44), UINT8_C(167), UINT8_C(208),
        UINT8_C(175), UINT8_C(153), UINT8_C( 28), UINT8_C(  3), UINT8_C(244), UINT8_C( 65), UINT8_C( 32), UINT8_C( 62) },
      { UINT8_C( 14), UINT8_C( 33), UINT8_C( 36), UINT8_C( 26), UINT8_C(128), UINT8_C( 35), UINT8_C(194), UINT8_C( 65),
           UINT8_MAX, UINT8_C(107), UINT8_C(155), UINT8_C( 47), UINT8_C(173), UINT8_C(  7), UINT8_C( 14), UINT8_C( 26) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 63), UINT8_C(152), UINT8_C(153), UINT8_C(  6), UINT8_C(197), UINT8_C( 64), UINT8_C(214), UINT8_C(116),
        UINT8_C(218), UINT8_C(243), UINT8_C(120), UINT8_C(206), UINT8_C( 52), UINT8_C(152), UINT8_C( 12), UINT8_C( 66) },
      { UINT8_C(185), UINT8_C( 48), UINT8_C( 93), UINT8_C( 58), UINT8_C( 84), UINT8_C( 31), UINT8_C(123), UINT8_C( 83),
        UINT8_C(138), UINT8_C( 22), UINT8_C(131), UINT8_C( 56), UINT8_C( 30), UINT8_C(145), UINT8_C( 82), UINT8_C( 93) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*2*/
    { { UINT8_C( 80), UINT8_C( 69), UINT8_C( 36), UINT8_C( 96), UINT8_C(225), UINT8_C(232), UINT8_C( 92), UINT8_C( 96),
        UINT8_C( 99), UINT8_C( 60), UINT8_C( 91), UINT8_C( 53), UINT8_C( 40), UINT8_C(174), UINT8_C( 40), UINT8_C( 54) },
      { UINT8_C( 48), UINT8_C(177), UINT8_C( 20), UINT8_C(  6), UINT8_C( 83), UINT8_C(  3), UINT8_C( 99), UINT8_C( 75),
        UINT8_C(175), UINT8_C(238), UINT8_C(204), UINT8_C( 96), UINT8_C( 96), UINT8_C(173), UINT8_C(176), UINT8_C(176) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(242), UINT8_C(213), UINT8_C( 16), UINT8_C(211), UINT8_C(189), UINT8_C(108), UINT8_C( 51), UINT8_C( 32),
        UINT8_C(168), UINT8_C(142), UINT8_C( 85), UINT8_C(209), UINT8_C( 61), UINT8_C(126), UINT8_C(  7), UINT8_C(109) },
      { UINT8_C( 47), UINT8_C( 27), UINT8_C(115), UINT8_C(130), UINT8_C( 31), UINT8_C(214), UINT8_C(205), UINT8_C(206),
        UINT8_C(196), UINT8_C(153), UINT8_C( 47), UINT8_C( 36), UINT8_C( 70), UINT8_C(223), UINT8_C(212), UINT8_C( 57) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(180), UINT8_C(228), UINT8_C( 12), UINT8_C(113), UINT8_C( 81), UINT8_C( 64), UINT8_C(146), UINT8_C(249),
        UINT8_C(206), UINT8_C(231), UINT8_C(202), UINT8_C( 11), UINT8_C(101), UINT8_C(210), UINT8_C(120), UINT8_C(148) },
      { UINT8_C(237), UINT8_C(235), UINT8_C( 22), UINT8_C( 12), UINT8_C(194), UINT8_C(227), UINT8_C(219), UINT8_C(134),
        UINT8_C(124), UINT8_C( 10), UINT8_C(171), UINT8_C(195), UINT8_C(233), UINT8_C(127), UINT8_C(252), UINT8_C(158) },
      { UINT16_C(    8), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(100), UINT8_C(  8), UINT8_C( 15), UINT8_C(181), UINT8_C( 72), UINT8_C(161), UINT8_C(174), UINT8_C( 23),
        UINT8_C(137), UINT8_C(121), UINT8_C( 34), UINT8_C(238), UINT8_C( 75), UINT8_C(155), UINT8_C(131), UINT8_C( 56) },
      { UINT8_C(134), UINT8_C(153), UINT8_C( 69), UINT8_C( 72), UINT8_C(125), UINT8_C( 32), UINT8_C(207), UINT8_C(249),
        UINT8_C( 42), UINT8_C(122), UINT8_C(188), UINT8_C( 19), UINT8_C(249), UINT8_C(184), UINT8_C(177), UINT8_C( 93) },
      { UINT16_C(    8), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(193), UINT8_C(193), UINT8_C( 18), UINT8_C(  9), UINT8_C( 98), UINT8_C(193), UINT8_C( 32), UINT8_C(235),
        UINT8_C( 58), UINT8_C( 67), UINT8_C(218), UINT8_C(133), UINT8_C(222), UINT8_C( 93), UINT8_C(189), UINT8_C(100) },
      { UINT8_C(246), UINT8_C(  2), UINT8_C(173), UINT8_C(115), UINT8_C( 34), UINT8_C(124), UINT8_C(109), UINT8_C( 76),
        UINT8_C(246), UINT8_C( 41), UINT8_C( 96), UINT8_C(239), UINT8_C(226), UINT8_C( 17), UINT8_C( 77), UINT8_C(163) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(210), UINT8_C( 95), UINT8_C(172), UINT8_C( 53), UINT8_C( 32), UINT8_C(205), UINT8_C( 32), UINT8_C( 90),
        UINT8_C( 16), UINT8_C(250), UINT8_C(223), UINT8_C(238), UINT8_C( 87), UINT8_C(157), UINT8_C( 82), UINT8_C( 78) },
      { UINT8_C(159),    UINT8_MAX, UINT8_C(193), UINT8_C(194), UINT8_C(123), UINT8_C( 46), UINT8_C( 14), UINT8_C(113),
        UINT8_C( 88), UINT8_C(110), UINT8_C( 97), UINT8_C( 58), UINT8_C(128), UINT8_C(174), UINT8_C(221), UINT8_C( 82) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 13), UINT8_C(137), UINT8_C(135), UINT8_C( 46), UINT8_C( 86), UINT8_C(168), UINT8_C(136), UINT8_C(102),
        UINT8_C(162), UINT8_C(104), UINT8_C( 84), UINT8_C(250), UINT8_C(  5), UINT8_C(167), UINT8_C( 72), UINT8_C(164) },
      { UINT8_C(166), UINT8_C(  9), UINT8_C(102), UINT8_C( 34), UINT8_C( 56), UINT8_C(117), UINT8_C(147), UINT8_C(144),
        UINT8_C(227), UINT8_C(244), UINT8_C(202), UINT8_C( 99), UINT8_C(162), UINT8_C(167), UINT8_C(182), UINT8_C(176) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 48), UINT8_C( 61), UINT8_C(222), UINT8_C(135), UINT8_C(229), UINT8_C(102), UINT8_C(237), UINT8_C(136),
        UINT8_C(206), UINT8_C( 66), UINT8_C(130), UINT8_C(211), UINT8_C(233), UINT8_C(202), UINT8_C(120), UINT8_C(143) },
      { UINT8_C(211), UINT8_C(222), UINT8_C(177), UINT8_C( 11), UINT8_C( 83), UINT8_C( 69), UINT8_C(155), UINT8_C( 55),
        UINT8_C( 57), UINT8_C(101), UINT8_C(154), UINT8_C(220), UINT8_C( 12), UINT8_C( 80), UINT8_C(140), UINT8_C( 61) },
      { UINT16_C(    2), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(142), UINT8_C(106), UINT8_C(196), UINT8_C(115), UINT8_C(208), UINT8_C(177), UINT8_C(251), UINT8_C(159),
        UINT8_C(243), UINT8_C(125), UINT8_C(114), UINT8_C(220), UINT8_C( 71), UINT8_C(234), UINT8_C(108), UINT8_C( 27) },
      { UINT8_C(201), UINT8_C( 29), UINT8_C( 38), UINT8_C( 28), UINT8_C( 98), UINT8_C(194), UINT8_C( 83), UINT8_C(156),
        UINT8_C( 39), UINT8_C(238), UINT8_C(120), UINT8_C( 52), UINT8_C( 62), UINT8_C(  4), UINT8_C(113), UINT8_C(204) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(110), UINT8_C( 53), UINT8_C( 64), UINT8_C( 62), UINT8_C(230), UINT8_C( 59), UINT8_C(221), UINT8_C(218),
        UINT8_C(185), UINT8_C( 80), UINT8_C(182), UINT8_C(  0), UINT8_C( 58), UINT8_C( 34), UINT8_C( 27), UINT8_C(  3) },
      { UINT8_C( 64), UINT8_C( 66), UINT8_C( 32), UINT8_C(162), UINT8_C(  4), UINT8_C(115), UINT8_C( 62), UINT8_C( 43),
        UINT8_C( 97), UINT8_C(182), UINT8_C( 95), UINT8_C(160), UINT8_C(186), UINT8_C(208), UINT8_C(108), UINT8_C( 40) },
      { UINT16_C(   65), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*4*/
    { { UINT8_C(221), UINT8_C( 90), UINT8_C( 80), UINT8_C( 70), UINT8_C(194), UINT8_C(176), UINT8_C(239), UINT8_C( 38),
        UINT8_C( 47), UINT8_C(204), UINT8_C(241), UINT8_C(206), UINT8_C( 24), UINT8_C( 66), UINT8_C(210), UINT8_C(237) },
      {    UINT8_MAX, UINT8_C( 36), UINT8_C(185), UINT8_C(251), UINT8_C(227), UINT8_C( 43), UINT8_C(144), UINT8_C(179),
        UINT8_C(222), UINT8_C( 11), UINT8_C(  5), UINT8_C(179), UINT8_C( 75), UINT8_C( 63), UINT8_C(162), UINT8_C( 40) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(153), UINT8_C(243), UINT8_C(111), UINT8_C( 92), UINT8_C(163), UINT8_C( 94), UINT8_C(130), UINT8_C(210),
        UINT8_C( 42), UINT8_C(115), UINT8_C(161), UINT8_C( 66), UINT8_C(181), UINT8_C(115), UINT8_C( 47), UINT8_C(180) },
      { UINT8_C(151), UINT8_C(232), UINT8_C(176), UINT8_C(123), UINT8_C( 19), UINT8_C( 64), UINT8_C( 46), UINT8_C(241),
        UINT8_C( 76), UINT8_C( 51), UINT8_C(165), UINT8_C(151), UINT8_C(114), UINT8_C( 71), UINT8_C(192), UINT8_C( 12) },
      { UINT16_C( 1158), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 58), UINT8_C( 47), UINT8_C(104), UINT8_C(221), UINT8_C(141), UINT8_C(234), UINT8_C(176), UINT8_C(183),
        UINT8_C( 94), UINT8_C( 81), UINT8_C(249), UINT8_C( 19), UINT8_C(196), UINT8_C( 40), UINT8_C(200), UINT8_C( 92) },
      { UINT8_C( 16), UINT8_C(120), UINT8_C(215), UINT8_C( 35), UINT8_C(184), UINT8_C(  5), UINT8_C( 21), UINT8_C(  4),
        UINT8_C( 56), UINT8_C(186), UINT8_C(156), UINT8_C(170), UINT8_C(  1), UINT8_C( 92), UINT8_C(182), UINT8_C( 60) },
      { UINT16_C( 1558), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(139), UINT8_C( 30), UINT8_C( 25), UINT8_C( 24), UINT8_C(  9), UINT8_C(201), UINT8_C(207), UINT8_C(103),
        UINT8_C( 26), UINT8_C(200), UINT8_C(122), UINT8_C(223), UINT8_C(240), UINT8_C( 66), UINT8_C( 59), UINT8_C(  0) },
      { UINT8_C(186), UINT8_C( 18), UINT8_C( 35), UINT8_C(115), UINT8_C( 23), UINT8_C( 56), UINT8_C(119), UINT8_C( 79),
        UINT8_C(242), UINT8_C( 19), UINT8_C(249), UINT8_C(244), UINT8_C(111), UINT8_C(176), UINT8_C( 48), UINT8_C(250) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(206), UINT8_C( 73), UINT8_C( 18), UINT8_C(215), UINT8_C( 19), UINT8_C(225), UINT8_C( 62), UINT8_C( 45),
        UINT8_C(169), UINT8_C(185), UINT8_C( 12), UINT8_C(153), UINT8_C(251), UINT8_C( 71), UINT8_C(153), UINT8_C(182) },
      { UINT8_C( 89), UINT8_C(189), UINT8_C( 41), UINT8_C(112), UINT8_C(245), UINT8_C(160), UINT8_C(191), UINT8_C(232),
        UINT8_C(180), UINT8_C(185), UINT8_C(220), UINT8_C( 35), UINT8_C(105), UINT8_C( 12), UINT8_C( 30), UINT8_C( 55) },
      { UINT16_C(  879), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 85), UINT8_C( 48), UINT8_C( 15), UINT8_C(104), UINT8_C( 18), UINT8_C( 77), UINT8_C(150), UINT8_C(187),
        UINT8_C(  6), UINT8_C(162), UINT8_C( 85), UINT8_C(  2), UINT8_C(234), UINT8_C(238), UINT8_C(184), UINT8_C( 67) },
      { UINT8_C(171), UINT8_C(225), UINT8_C(180), UINT8_C(161), UINT8_C(129), UINT8_C(115), UINT8_C(137), UINT8_C( 53),
        UINT8_C( 44), UINT8_C(101), UINT8_C( 89), UINT8_C(149), UINT8_C(113), UINT8_C(119), UINT8_C(205), UINT8_C(198) },
      { UINT16_C( 1920), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(167), UINT8_C(220), UINT8_C( 47), UINT8_C(185), UINT8_C( 41), UINT8_C(197), UINT8_C(117), UINT8_C( 48),
        UINT8_C(103), UINT8_C(202), UINT8_C( 50), UINT8_C( 81), UINT8_C(184), UINT8_C(234), UINT8_C(149), UINT8_C(100) },
      { UINT8_C(203), UINT8_C( 73), UINT8_C(  5), UINT8_C( 76), UINT8_C(188), UINT8_C(142), UINT8_C(130), UINT8_C(233),
        UINT8_C(243), UINT8_C(219), UINT8_C(126), UINT8_C(100), UINT8_C( 82), UINT8_C( 75), UINT8_C( 42), UINT8_C(249) },
      { UINT16_C( 1659), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 39), UINT8_C( 89), UINT8_C(179), UINT8_C( 81), UINT8_C( 30), UINT8_C( 40), UINT8_C(129), UINT8_C(134),
        UINT8_C(242), UINT8_C(179), UINT8_C(215), UINT8_C(170), UINT8_C(157), UINT8_C(108), UINT8_C( 14), UINT8_C(104) },
      { UINT8_C(181), UINT8_C( 19), UINT8_C(180), UINT8_C(114), UINT8_C(161), UINT8_C( 54), UINT8_C( 91), UINT8_C(148),
        UINT8_C( 17), UINT8_C(217), UINT8_C(248), UINT8_C( 99), UINT8_C( 37), UINT8_C( 35), UINT8_C( 93), UINT8_C( 76) },
      { UINT16_C(   32), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(124), UINT8_C( 16), UINT8_C(157), UINT8_C(155), UINT8_C( 56), UINT8_C( 30), UINT8_C( 33), UINT8_C( 42),
        UINT8_C(209), UINT8_C(248), UINT8_C(212), UINT8_C(110), UINT8_C(101), UINT8_C(227), UINT8_C(214), UINT8_C( 26) },
      { UINT8_C(246), UINT8_C(139), UINT8_C(140), UINT8_C(152), UINT8_C(193), UINT8_C(231), UINT8_C( 44), UINT8_C(211),
        UINT8_C(193), UINT8_C( 37), UINT8_C( 54), UINT8_C(230), UINT8_C( 72), UINT8_C(147), UINT8_C( 50), UINT8_C(196) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(163), UINT8_C(208), UINT8_C( 95), UINT8_C(219), UINT8_C(238), UINT8_C(128), UINT8_C(  5), UINT8_C(192),
        UINT8_C(121), UINT8_C(218), UINT8_C( 46), UINT8_C(222), UINT8_C(189), UINT8_C(  5), UINT8_C(248), UINT8_C(179) },
      { UINT8_C(144), UINT8_C(133), UINT8_C( 75), UINT8_C( 81), UINT8_C(108), UINT8_C(120), UINT8_C( 36), UINT8_C( 45),
        UINT8_C(157), UINT8_C( 91), UINT8_C( 19), UINT8_C(229), UINT8_C(238), UINT8_C( 70), UINT8_C(169), UINT8_C(146) },
      { UINT16_C(  307), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*6*/
    { { UINT8_C(108), UINT8_C( 40), UINT8_C(173), UINT8_C(140), UINT8_C(185), UINT8_C(236), UINT8_C( 68), UINT8_C(120),
        UINT8_C(105), UINT8_C(164), UINT8_C(130), UINT8_C(253), UINT8_C(198), UINT8_C( 62), UINT8_C(165), UINT8_C(158) },
      { UINT8_C(  1), UINT8_C(125), UINT8_C( 84), UINT8_C(142), UINT8_C(200), UINT8_C(112), UINT8_C(165), UINT8_C( 43),
        UINT8_C( 95), UINT8_C(139), UINT8_C(227), UINT8_C(191), UINT8_C(193), UINT8_C(165), UINT8_C(112), UINT8_C( 45) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(205), UINT8_C( 29), UINT8_C(185), UINT8_C(135), UINT8_C(  9), UINT8_C(253),    UINT8_MAX, UINT8_C(114),
        UINT8_C(161), UINT8_C(130), UINT8_C(111), UINT8_C(104), UINT8_C(192), UINT8_C( 21), UINT8_C(  6), UINT8_C(193) },
      { UINT8_C(146), UINT8_C( 90), UINT8_C( 79), UINT8_C( 90), UINT8_C(202), UINT8_C(245), UINT8_C(133), UINT8_C( 41),
        UINT8_C(128), UINT8_C(104), UINT8_C(232), UINT8_C( 65), UINT8_C( 13), UINT8_C( 88), UINT8_C(110), UINT8_C(219) },
      { UINT16_C( 1056), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(118), UINT8_C( 39), UINT8_C( 98), UINT8_C(127), UINT8_C( 37), UINT8_C( 97), UINT8_C(242), UINT8_C(198),
        UINT8_C(227), UINT8_C( 97), UINT8_C( 46), UINT8_C(163), UINT8_C(118), UINT8_C( 52), UINT8_C(101), UINT8_C(  8) },
      { UINT8_C(142), UINT8_C(180), UINT8_C( 98), UINT8_C( 88), UINT8_C(169), UINT8_C(231), UINT8_C(130), UINT8_C( 42),
        UINT8_C( 80), UINT8_C(106), UINT8_C(107), UINT8_C( 93), UINT8_C(195), UINT8_C(218), UINT8_C( 56), UINT8_C( 57) },
      { UINT16_C( 1540), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(  1), UINT8_C(154), UINT8_C(184), UINT8_C( 38), UINT8_C(252), UINT8_C(170), UINT8_C(237), UINT8_C(223),
        UINT8_C( 12), UINT8_C( 27), UINT8_C(131), UINT8_C(130), UINT8_C( 80), UINT8_C(232), UINT8_C(139), UINT8_C(222) },
      { UINT8_C(156), UINT8_C(237), UINT8_C( 55), UINT8_C( 70), UINT8_C(213), UINT8_C(185), UINT8_C(112), UINT8_C( 37),
        UINT8_C( 35), UINT8_C(219), UINT8_C(130), UINT8_C(230), UINT8_C(181), UINT8_C(187), UINT8_C( 31), UINT8_C(183) },
      { UINT16_C(  946), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 85), UINT8_C(216), UINT8_C(221), UINT8_C( 81), UINT8_C(130), UINT8_C(202), UINT8_C( 49), UINT8_C(142),
        UINT8_C(230), UINT8_C(180), UINT8_C( 17), UINT8_C( 54), UINT8_C(156), UINT8_C(156), UINT8_C( 20), UINT8_C( 56) },
      { UINT8_C(137), UINT8_C( 75), UINT8_C(126), UINT8_C( 94), UINT8_C(  4), UINT8_C(238), UINT8_C(131), UINT8_C( 40),
        UINT8_C(202), UINT8_C(  6), UINT8_C( 14), UINT8_C(127), UINT8_C(193), UINT8_C( 46), UINT8_C( 54), UINT8_C( 22) },
      { UINT16_C( 1714), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(  6), UINT8_C( 20), UINT8_C(104), UINT8_C(136), UINT8_C(222), UINT8_C(153), UINT8_C( 23), UINT8_C(196),
        UINT8_C( 77), UINT8_C( 40), UINT8_C(250), UINT8_C(233), UINT8_C(196), UINT8_C( 15), UINT8_C( 33), UINT8_C( 77) },
      { UINT8_C( 90), UINT8_C(160), UINT8_C(172), UINT8_C( 95), UINT8_C(142), UINT8_C( 47), UINT8_C(135), UINT8_C( 88),
        UINT8_C( 53), UINT8_C(149), UINT8_C(216), UINT8_C(246), UINT8_C(195), UINT8_C( 14), UINT8_C( 13), UINT8_C(201) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 34), UINT8_C(117), UINT8_C( 82), UINT8_C(  1), UINT8_C( 14), UINT8_C(105), UINT8_C(197), UINT8_C( 91),
        UINT8_C(145), UINT8_C(192), UINT8_C( 68), UINT8_C( 85), UINT8_C(207), UINT8_C(101), UINT8_C(162), UINT8_C( 41) },
      { UINT8_C(  5), UINT8_C( 78), UINT8_C(136), UINT8_C(148), UINT8_C(126), UINT8_C( 15), UINT8_C(236), UINT8_C(179),
        UINT8_C(165), UINT8_C(196), UINT8_C(170), UINT8_C(104), UINT8_C(211), UINT8_C(183), UINT8_C( 50), UINT8_C(245) },
      { UINT16_C(    2), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 44), UINT8_C(132), UINT8_C(246), UINT8_C( 58), UINT8_C(237), UINT8_C(188), UINT8_C(149), UINT8_C(126),
        UINT8_C(124), UINT8_C(217), UINT8_C(211), UINT8_C( 75), UINT8_C( 62), UINT8_C(117), UINT8_C(116), UINT8_C( 68) },
      { UINT8_C(196), UINT8_C(253), UINT8_C(216), UINT8_C( 66), UINT8_C( 12), UINT8_C(196), UINT8_C(245), UINT8_C(177),
        UINT8_C(137), UINT8_C(159), UINT8_C( 26), UINT8_C( 92), UINT8_C( 86), UINT8_C( 76), UINT8_C( 81), UINT8_C(130) },
      { UINT16_C( 1042), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(208), UINT8_C( 72), UINT8_C(188), UINT8_C(189), UINT8_C(  4), UINT8_C( 81), UINT8_C( 59), UINT8_C(128),
        UINT8_C( 42), UINT8_C( 14), UINT8_C(203), UINT8_C(105), UINT8_C(131), UINT8_C( 63), UINT8_C(173), UINT8_C( 71) },
      { UINT8_C( 60), UINT8_C(133), UINT8_C(137), UINT8_C( 73), UINT8_C( 73), UINT8_C(127), UINT8_C(250), UINT8_C(210),
        UINT8_C( 30), UINT8_C( 20), UINT8_C( 46), UINT8_C(117), UINT8_C( 96), UINT8_C(128), UINT8_C(247), UINT8_C( 48) },
      { UINT16_C( 1985), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(200), UINT8_C(180), UINT8_C(237), UINT8_C(204), UINT8_C(  5), UINT8_C( 40), UINT8_C( 76), UINT8_C( 48),
        UINT8_C( 54), UINT8_C( 23), UINT8_C(153), UINT8_C(186), UINT8_C( 86), UINT8_C( 70), UINT8_C(  1), UINT8_C(147) },
      { UINT8_C(203), UINT8_C(139), UINT8_C(220), UINT8_C( 20), UINT8_C( 10), UINT8_C(214), UINT8_C(231), UINT8_C( 40),
        UINT8_C(235), UINT8_C( 21), UINT8_C(157), UINT8_C( 75), UINT8_C(149), UINT8_C(149), UINT8_C(124), UINT8_C( 93) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*8*/
    { { UINT8_C(161), UINT8_C(134), UINT8_C(134), UINT8_C( 79), UINT8_C(117), UINT8_C( 29), UINT8_C(112), UINT8_C(  3),
        UINT8_C(155), UINT8_C(  1), UINT8_C( 99), UINT8_C(129), UINT8_C(110), UINT8_C(167), UINT8_C(241), UINT8_C(  5) },
      { UINT8_C(161), UINT8_C( 46), UINT8_C(246), UINT8_C(170), UINT8_C(117), UINT8_C(104), UINT8_C(112), UINT8_C(120),
        UINT8_C( 76), UINT8_C(101), UINT8_C( 47), UINT8_C(248), UINT8_C( 91), UINT8_C( 29), UINT8_C(241), UINT8_C(246) },
      { UINT16_C(63505), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(163), UINT8_C(105), UINT8_C( 69), UINT8_C( 26), UINT8_C(137), UINT8_C(181), UINT8_C( 30), UINT8_C( 33),
        UINT8_C(183), UINT8_C(129), UINT8_C(206), UINT8_C( 37), UINT8_C( 40), UINT8_C(147), UINT8_C( 42), UINT8_C(201) },
      { UINT8_C(193), UINT8_C( 33), UINT8_C(115), UINT8_C( 55), UINT8_C(137), UINT8_C(214), UINT8_C(175), UINT8_C( 33),
        UINT8_C( 59), UINT8_C(223), UINT8_C(206), UINT8_C(150), UINT8_C(252), UINT8_C(177), UINT8_C(140), UINT8_C(159) },
      { UINT16_C(63504), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 26), UINT8_C(209), UINT8_C(185), UINT8_C(160), UINT8_C(134), UINT8_C(158), UINT8_C(193), UINT8_C( 61),
        UINT8_C( 88), UINT8_C(220), UINT8_C( 98), UINT8_C(128), UINT8_C(141), UINT8_C(141), UINT8_C( 74), UINT8_C(184) },
      { UINT8_C(174), UINT8_C(189), UINT8_C(239), UINT8_C( 55), UINT8_C(147), UINT8_C(158), UINT8_C( 13), UINT8_C(206),
        UINT8_C(125), UINT8_C(220), UINT8_C(100), UINT8_C(121), UINT8_C(141), UINT8_C(240), UINT8_C( 24), UINT8_C(168) },
      { UINT16_C(63488), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(193), UINT8_C(210), UINT8_C( 72), UINT8_C( 72), UINT8_C(169), UINT8_C( 10), UINT8_C(133), UINT8_C(  2),
        UINT8_C(109), UINT8_C(232), UINT8_C(130), UINT8_C(100), UINT8_C(117), UINT8_C(204), UINT8_C( 28), UINT8_C( 35) },
      { UINT8_C(138), UINT8_C( 11), UINT8_C( 90), UINT8_C( 29), UINT8_C(169), UINT8_C(104), UINT8_C(236), UINT8_C( 39),
        UINT8_C( 68), UINT8_C( 80), UINT8_C(160), UINT8_C(209), UINT8_C( 65), UINT8_C(185), UINT8_C(121), UINT8_C(  2) },
      { UINT16_C(63504), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(139), UINT8_C(194), UINT8_C( 74), UINT8_C( 52), UINT8_C(204), UINT8_C(208), UINT8_C( 54), UINT8_C( 57),
        UINT8_C(184), UINT8_C(185), UINT8_C(157), UINT8_C( 45), UINT8_C(133), UINT8_C(185), UINT8_C( 80), UINT8_C( 15) },
      { UINT8_C(196), UINT8_C(170), UINT8_C( 45), UINT8_C(110), UINT8_C( 18), UINT8_C( 25), UINT8_C(149), UINT8_C( 86),
        UINT8_C(105), UINT8_C( 53), UINT8_C( 40), UINT8_C(170), UINT8_C(238), UINT8_C(161), UINT8_C(173), UINT8_C(121) },
      { UINT16_C(63488), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 99), UINT8_C(247), UINT8_C(174), UINT8_C( 47), UINT8_C(199), UINT8_C(135), UINT8_C(105), UINT8_C(127),
        UINT8_C(189), UINT8_C(  6), UINT8_C(172), UINT8_C(171), UINT8_C(192), UINT8_C(252), UINT8_C( 50), UINT8_C(132) },
      { UINT8_C(167), UINT8_C( 95), UINT8_C(242), UINT8_C(185), UINT8_C(120), UINT8_C(135), UINT8_C( 16), UINT8_C(226),
        UINT8_C(189), UINT8_C( 56), UINT8_C(140), UINT8_C(171), UINT8_C(217), UINT8_C( 57), UINT8_C( 37), UINT8_C( 61) },
      { UINT16_C(63488), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C( 49), UINT8_C(211), UINT8_C(108), UINT8_C(248), UINT8_C(183), UINT8_C(213), UINT8_C(120), UINT8_C( 85),
        UINT8_C(201), UINT8_C(241), UINT8_C(120), UINT8_C(156), UINT8_C( 33), UINT8_C(170), UINT8_C( 32), UINT8_C(200) },
      { UINT8_C( 10), UINT8_C( 19), UINT8_C(129), UINT8_C(130), UINT8_C(154), UINT8_C(145), UINT8_C(100), UINT8_C(200),
        UINT8_C(201), UINT8_C(241), UINT8_C(  3), UINT8_C(163), UINT8_C( 42), UINT8_C( 40), UINT8_C(224), UINT8_C( 91) },
      { UINT16_C(63488), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(251), UINT8_C( 76), UINT8_C( 84), UINT8_C(178), UINT8_C( 34), UINT8_C(204), UINT8_C(  7), UINT8_C(254),
        UINT8_C(240), UINT8_C(194), UINT8_C(154), UINT8_C( 17), UINT8_C( 42), UINT8_C(186), UINT8_C(217), UINT8_C( 52) },
      { UINT8_C(205), UINT8_C( 91), UINT8_C(182), UINT8_C(104), UINT8_C(236), UINT8_C( 27), UINT8_C(191), UINT8_C(182),
        UINT8_C( 12), UINT8_C(194), UINT8_C( 89), UINT8_C( 54), UINT8_C(234), UINT8_C(186), UINT8_C(146), UINT8_C(229) },
      { UINT16_C(63488), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(133), UINT8_C(230), UINT8_C(152), UINT8_C(213), UINT8_C(178), UINT8_C(159), UINT8_C(139), UINT8_C(162),
        UINT8_C( 31), UINT8_C( 63), UINT8_C(180), UINT8_C( 73), UINT8_C(250), UINT8_C(141), UINT8_C(125), UINT8_C(199) },
      { UINT8_C(232), UINT8_C( 51), UINT8_C( 47), UINT8_C(213), UINT8_C( 78), UINT8_C(239), UINT8_C(139), UINT8_C( 90),
        UINT8_C(177), UINT8_C(228), UINT8_C(145), UINT8_C(156), UINT8_C( 29), UINT8_C( 35), UINT8_C(129), UINT8_C(162) },
      { UINT16_C(63496), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT8_C(  9), UINT8_C( 32), UINT8_C( 32), UINT8_C(187), UINT8_C(185), UINT8_C( 32), UINT8_C( 93), UINT8_C(216),
        UINT8_C( 47), UINT8_C( 17), UINT8_C( 33), UINT8_C( 41), UINT8_C(159), UINT8_C(158), UINT8_C(240), UINT8_C(135) },
      { UINT8_C(209), UINT8_C( 32), UINT8_C( 92), UINT8_C( 32), UINT8_C( 15), UINT8_C(231), UINT8_C(122), UINT8_C(192),
        UINT8_C(203), UINT8_C( 11), UINT8_C( 92), UINT8_C(232), UINT8_C( 46), UINT8_C(222), UINT8_C(139), UINT8_C(135) },
      { UINT16_C(63490), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, }
  };

#define test_case_template(start, step, imm8) ({                        \
  for(size_t i = (start); i < (start) + (step); i++){                   \
    easysimd__m128i a = easysimd_mm_loadu_epi8((void *)&(test_vec[i].a[0]));  \
    easysimd__m128i b = easysimd_mm_loadu_epi8((void *)&(test_vec[i].b[0]));  \
    easysimd__m128i r = easysimd_mm_loadu_epi8((void *)&(test_vec[i].r[0]));  \
    easysimd__m128i ret;                                                   \
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {                           \
      ret = easysimd_mm_cmpestrm(a, 5, b, 11, imm8);                       \
    }                                                                   \
    EASYSIMD_TEST_PERF_END("_mm_cmpestrm");                                \
    easysimd_assert_m128i_u16(r, ==, ret);}                                \
})

  test_case_template( 0, 10, 0);
  test_case_template(10, 10, 2);
  test_case_template(20, 10, 4);
  test_case_template(30, 10, 6);
  test_case_template(40, 10, 8);

#undef test_case_template

  return 0;
}

static int
test_easysimd_mm_cmpestrm_16bit(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    uint16_t a[8];
    uint16_t b[8];
    uint16_t r[8];
  } test_vec[] = {
/*1*/
    { { UINT16_C(31970), UINT16_C(28414), UINT16_C(64008), UINT16_C(40360), UINT16_C( 1812), UINT16_C(19280), UINT16_C(20652), UINT16_C(31490) },
      { UINT16_C(31970), UINT16_C( 6881), UINT16_C(42202), UINT16_C(59206), UINT16_C(61054), UINT16_C(24518), UINT16_C(40891), UINT16_C(40651) },
      { UINT16_C(    1), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(51483), UINT16_C( 9228), UINT16_C(59499), UINT16_C(55233), UINT16_C( 4796), UINT16_C(26659), UINT16_C(59499), UINT16_C( 7651) },
      { UINT16_C(50272), UINT16_C(51483), UINT16_C(32104), UINT16_C(58914), UINT16_C(59499), UINT16_C(10053), UINT16_C( 7651), UINT16_C(59499) },
      { UINT16_C(    2), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(53722), UINT16_C(40391), UINT16_C(34950), UINT16_C(17013), UINT16_C(39066), UINT16_C(64938), UINT16_C(36285), UINT16_C( 7450) },
      { UINT16_C(20817), UINT16_C(47448), UINT16_C(53722), UINT16_C(15008), UINT16_C(58723), UINT16_C(60001), UINT16_C( 9974), UINT16_C(53389) },
      { UINT16_C(    4), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(21751), UINT16_C(32109), UINT16_C(58077), UINT16_C(30655), UINT16_C(27002), UINT16_C(14196), UINT16_C(36598), UINT16_C(18261) },
      { UINT16_C(44511), UINT16_C(44545), UINT16_C(41256), UINT16_C(21751), UINT16_C(18822), UINT16_C(31861), UINT16_C(  879), UINT16_C(26188) },
      { UINT16_C(    8), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(47703), UINT16_C(13540), UINT16_C(41884), UINT16_C( 6060), UINT16_C( 8205), UINT16_C(  846), UINT16_C(41903), UINT16_C(36427) },
      { UINT16_C(19537), UINT16_C(31036), UINT16_C( 9453), UINT16_C(29444), UINT16_C(31085), UINT16_C(47703), UINT16_C(15484), UINT16_C(54339) },
      { UINT16_C(   32), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(10230), UINT16_C(37640), UINT16_C(46282), UINT16_C(55210), UINT16_C(63701), UINT16_C(34011), UINT16_C( 9884), UINT16_C(60690) },
      { UINT16_C(20338), UINT16_C(46282), UINT16_C(27251), UINT16_C(57810), UINT16_C(49891), UINT16_C(24765), UINT16_C(  255), UINT16_C(62772) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(15399), UINT16_C(62088), UINT16_C(13041), UINT16_C(50889), UINT16_C(42027), UINT16_C(51018), UINT16_C(23754), UINT16_C(15540) },
      { UINT16_C( 6827), UINT16_C( 8091), UINT16_C(28292), UINT16_C(51018), UINT16_C(48432), UINT16_C(12231), UINT16_C(64446), UINT16_C(51018) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(44344), UINT16_C(10711), UINT16_C(41440), UINT16_C( 3055), UINT16_C(14661), UINT16_C( 4306), UINT16_C(10711), UINT16_C(16716) },
      { UINT16_C(59552), UINT16_C( 9312), UINT16_C(24662), UINT16_C(34443), UINT16_C(10711), UINT16_C(56246), UINT16_C(56142), UINT16_C(34497) },
      { UINT16_C(   16), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(39048), UINT16_C(26799), UINT16_C(40505), UINT16_C(32627), UINT16_C(26799), UINT16_C(28047), UINT16_C(56267), UINT16_C(27566) },
      { UINT16_C(26799), UINT16_C( 6543), UINT16_C( 7022), UINT16_C(26799), UINT16_C(22126), UINT16_C(48231), UINT16_C(10289), UINT16_C(26799) },
      { UINT16_C(    9), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(62144), UINT16_C(64034), UINT16_C(38289), UINT16_C(26745), UINT16_C( 2267), UINT16_C(42709), UINT16_C(33763), UINT16_C(42770) },
      { UINT16_C(41361), UINT16_C(65472), UINT16_C(24764), UINT16_C(10891), UINT16_C(62134), UINT16_C(59367), UINT16_C(10778), UINT16_C(55969) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*3*/
    { { UINT16_C( 9554), UINT16_C(51291), UINT16_C( 3966), UINT16_C(37595), UINT16_C(36496), UINT16_C(45813), UINT16_C(16907), UINT16_C(63598) },
      { UINT16_C(13805), UINT16_C( 9554), UINT16_C( 9554), UINT16_C(52334), UINT16_C(29630), UINT16_C(40341), UINT16_C( 9554), UINT16_C(59076) },
      { UINT16_C(    6), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C( 8431), UINT16_C(28078), UINT16_C(35119), UINT16_C(49407), UINT16_C(62744), UINT16_C( 9074), UINT16_C(57399), UINT16_C( 9243) },
      { UINT16_C(12822), UINT16_C(49407), UINT16_C(26922), UINT16_C(59638), UINT16_C(35804), UINT16_C(49407), UINT16_C(19029), UINT16_C(17750) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C( 1386), UINT16_C(44655), UINT16_C(45710), UINT16_C(42585), UINT16_C(52391), UINT16_C(57033), UINT16_C(44655), UINT16_C(44655) },
      { UINT16_C(64790), UINT16_C(16621), UINT16_C(58214), UINT16_C(17192), UINT16_C(44655), UINT16_C(50355), UINT16_C( 2808), UINT16_C(25097) },
      { UINT16_C(   16), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(48143), UINT16_C(40443), UINT16_C(21870), UINT16_C( 5444), UINT16_C( 3361), UINT16_C(52723), UINT16_C(62962), UINT16_C( 2192) },
      { UINT16_C(62962), UINT16_C(62962), UINT16_C(29024), UINT16_C(53147), UINT16_C(62962), UINT16_C( 6036), UINT16_C(40281), UINT16_C(62962) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(30041), UINT16_C(50949), UINT16_C(39621), UINT16_C(60380), UINT16_C(53079), UINT16_C(39621), UINT16_C(18628), UINT16_C(46673) },
      { UINT16_C(39621), UINT16_C( 9743), UINT16_C(43532), UINT16_C(46673), UINT16_C(35321), UINT16_C(21059), UINT16_C(39621), UINT16_C(32954) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(49201), UINT16_C(64328), UINT16_C( 9225), UINT16_C(24806), UINT16_C(49201), UINT16_C(47273), UINT16_C(64487), UINT16_C(44399) },
      { UINT16_C(49201), UINT16_C(49201), UINT16_C(51240), UINT16_C( 8909), UINT16_C( 4178), UINT16_C(31092), UINT16_C(49201), UINT16_C(65273) },
      { UINT16_C(    3), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(16879), UINT16_C(63737), UINT16_C(57446), UINT16_C(23129), UINT16_C(29677), UINT16_C(26130), UINT16_C(16879), UINT16_C(37651) },
      { UINT16_C(59135), UINT16_C(16879), UINT16_C(  431), UINT16_C(  330), UINT16_C(48657), UINT16_C(56954), UINT16_C(29677), UINT16_C(56540) },
      { UINT16_C(    2), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(54709), UINT16_C( 7125), UINT16_C(62869), UINT16_C(13429), UINT16_C(34608), UINT16_C(11931), UINT16_C(62869), UINT16_C( 2241) },
      { UINT16_C(62869), UINT16_C(17456), UINT16_C(31479), UINT16_C(34608), UINT16_C(48953), UINT16_C( 9958), UINT16_C(49714), UINT16_C(59139) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(55448), UINT16_C(19714), UINT16_C(30470), UINT16_C(13954), UINT16_C( 7679), UINT16_C( 2148), UINT16_C( 9675), UINT16_C(24592) },
      { UINT16_C(13954), UINT16_C( 4772), UINT16_C(59835), UINT16_C(24592), UINT16_C(  424), UINT16_C(13954), UINT16_C( 7875), UINT16_C(24592) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(50678), UINT16_C(64681), UINT16_C(11068), UINT16_C(15154), UINT16_C(38728), UINT16_C( 4931), UINT16_C(21692), UINT16_C(55156) },
      { UINT16_C( 6293), UINT16_C(64681), UINT16_C( 1026), UINT16_C(43589), UINT16_C(38728), UINT16_C(21692), UINT16_C(18558), UINT16_C(29732) },
      { UINT16_C(    2), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*5*/
    { { UINT16_C(56195), UINT16_C(10798), UINT16_C(23389), UINT16_C(53101), UINT16_C(16867), UINT16_C(11410), UINT16_C(38283), UINT16_C(15567) },
      { UINT16_C(18998), UINT16_C(49773), UINT16_C(58607), UINT16_C(19052), UINT16_C(61613), UINT16_C(48105), UINT16_C(29700), UINT16_C(34586) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(18511), UINT16_C(44209), UINT16_C( 8099), UINT16_C(34683), UINT16_C( 3424), UINT16_C(60339), UINT16_C(33698), UINT16_C(55335) },
      { UINT16_C(38349), UINT16_C(48282), UINT16_C( 1657), UINT16_C( 9735), UINT16_C(61686), UINT16_C(64225), UINT16_C(64356), UINT16_C(45954) },
      { UINT16_C(    1), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(13124), UINT16_C(59231), UINT16_C(55890), UINT16_C(45678), UINT16_C( 8936), UINT16_C(35485), UINT16_C(50597), UINT16_C(29283) },
      { UINT16_C(64858), UINT16_C(54062), UINT16_C(13572), UINT16_C(64249), UINT16_C(56102), UINT16_C(35573), UINT16_C(30678), UINT16_C( 6718) },
      { UINT16_C(   54), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(40362), UINT16_C(64770), UINT16_C(28792), UINT16_C(24751), UINT16_C(19858), UINT16_C(14314), UINT16_C(19730), UINT16_C(27817) },
      { UINT16_C(55371), UINT16_C(20287), UINT16_C(14349), UINT16_C(13129), UINT16_C(15891), UINT16_C(60094), UINT16_C(64693), UINT16_C(24580) },
      { UINT16_C(   33), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C( 1689), UINT16_C( 4445), UINT16_C( 3191), UINT16_C( 2417), UINT16_C(23641), UINT16_C(27457), UINT16_C(60073), UINT16_C(62679) },
      { UINT16_C( 5826), UINT16_C(53315), UINT16_C(36175), UINT16_C(25091), UINT16_C(49611), UINT16_C(33100), UINT16_C(20925), UINT16_C(22497) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(15959), UINT16_C(52840), UINT16_C(55882), UINT16_C(42200), UINT16_C( 6454), UINT16_C(57103), UINT16_C(59139), UINT16_C(50900) },
      { UINT16_C( 6141), UINT16_C(19606), UINT16_C(39332), UINT16_C(28847), UINT16_C(64347), UINT16_C( 6385), UINT16_C(53836), UINT16_C(42095) },
      { UINT16_C(   14), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(55312), UINT16_C(23154), UINT16_C(19122), UINT16_C(59646), UINT16_C( 3683), UINT16_C(26567), UINT16_C(39925), UINT16_C(61997) },
      { UINT16_C(50099), UINT16_C(22335), UINT16_C(61020), UINT16_C(47047), UINT16_C(47337), UINT16_C(14032), UINT16_C(16266), UINT16_C(39642) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(19479), UINT16_C(51701), UINT16_C(62359), UINT16_C(64177), UINT16_C(30977), UINT16_C(63073), UINT16_C(36372), UINT16_C(51177) },
      { UINT16_C(10321), UINT16_C(44575), UINT16_C(58902), UINT16_C(65381), UINT16_C(13727), UINT16_C(10549), UINT16_C( 3957), UINT16_C(36036) },
      { UINT16_C(    2), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(47452), UINT16_C(62294), UINT16_C( 1964), UINT16_C(44781), UINT16_C(20352), UINT16_C(38308), UINT16_C(36317), UINT16_C(12124) },
      { UINT16_C(31669), UINT16_C(52189), UINT16_C(16994), UINT16_C(  459), UINT16_C(  120), UINT16_C(60714), UINT16_C(60944), UINT16_C(27769) },
      { UINT16_C(   34), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(53159), UINT16_C(21599), UINT16_C(19671), UINT16_C(22274), UINT16_C(42651), UINT16_C(31212), UINT16_C(18740), UINT16_C(59816) },
      { UINT16_C(34244), UINT16_C( 9909), UINT16_C(32967), UINT16_C(16167), UINT16_C(21120), UINT16_C(36908), UINT16_C(42560), UINT16_C(59644) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*7*/
    { { UINT16_C( 6556), UINT16_C(29418), UINT16_C( 1017), UINT16_C(45159), UINT16_C(49662), UINT16_C(60395), UINT16_C(13058), UINT16_C( 6217) },
      { UINT16_C(54770), UINT16_C(26584), UINT16_C(51271), UINT16_C(50853), UINT16_C(61949), UINT16_C(51867), UINT16_C( 4304), UINT16_C(27729) },
      { UINT16_C(    2), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(15401), UINT16_C( 9183), UINT16_C(17983), UINT16_C(15827), UINT16_C(48647), UINT16_C( 2344), UINT16_C(29169), UINT16_C(58401) },
      { UINT16_C(64070), UINT16_C(36171), UINT16_C(61634), UINT16_C(48980), UINT16_C(61409), UINT16_C(45449), UINT16_C(56319), UINT16_C(10526) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(64791), UINT16_C(22092), UINT16_C( 8003), UINT16_C(19091), UINT16_C(48093), UINT16_C(53075), UINT16_C(29740), UINT16_C(29363) },
      { UINT16_C(65134), UINT16_C(12544), UINT16_C(21742), UINT16_C(53232), UINT16_C(31299), UINT16_C(17281), UINT16_C(40789), UINT16_C(27756) },
      { UINT16_C(   39), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(47260), UINT16_C(57282), UINT16_C(21975), UINT16_C(46121), UINT16_C(31760), UINT16_C(15491), UINT16_C(14064), UINT16_C(24494) },
      { UINT16_C(44596), UINT16_C( 8848), UINT16_C(32770), UINT16_C(18162), UINT16_C(29690), UINT16_C(20361), UINT16_C(62738), UINT16_C(44731) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(32173), UINT16_C(33933), UINT16_C(46802), UINT16_C(57912), UINT16_C(48178), UINT16_C( 8734), UINT16_C(52722), UINT16_C(10113) },
      { UINT16_C( 4475), UINT16_C(32329), UINT16_C(15250), UINT16_C(36036), UINT16_C(19886), UINT16_C(49372), UINT16_C(38722), UINT16_C(61294) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(64277), UINT16_C(59251), UINT16_C(43953), UINT16_C(58314), UINT16_C(59495), UINT16_C(23046), UINT16_C(34741), UINT16_C(12673) },
      { UINT16_C(51865), UINT16_C(11183), UINT16_C(29446), UINT16_C(46263), UINT16_C(37824), UINT16_C(  629), UINT16_C(58155), UINT16_C(16625) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(25823), UINT16_C(36903), UINT16_C(61711), UINT16_C(30580), UINT16_C(31450), UINT16_C(36817), UINT16_C(20993), UINT16_C(39616) },
      { UINT16_C(28444), UINT16_C( 8901), UINT16_C(32226), UINT16_C(41687), UINT16_C(19472), UINT16_C(15268), UINT16_C(38191), UINT16_C( 3707) },
      { UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(41977), UINT16_C( 2463), UINT16_C( 5012), UINT16_C(28288), UINT16_C(20877), UINT16_C(36606), UINT16_C(48803), UINT16_C(48937) },
      { UINT16_C(60974), UINT16_C( 4322), UINT16_C(47467), UINT16_C(31923), UINT16_C(22277), UINT16_C(13495), UINT16_C(13293), UINT16_C(58947) },
      { UINT16_C(    5), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(58070), UINT16_C(27375), UINT16_C(28661), UINT16_C(33497), UINT16_C(55232), UINT16_C(25360), UINT16_C(14741), UINT16_C(49955) },
      { UINT16_C( 1320), UINT16_C(37844), UINT16_C(34750), UINT16_C(49935), UINT16_C(51166), UINT16_C(52215), UINT16_C(15098), UINT16_C(53426) },
      { UINT16_C(    1), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(41244), UINT16_C( 4410), UINT16_C( 4881), UINT16_C(53651), UINT16_C(42218), UINT16_C(32821), UINT16_C(22749), UINT16_C( 1347) },
      { UINT16_C( 5981), UINT16_C( 7065), UINT16_C(43166), UINT16_C(32222), UINT16_C(54639), UINT16_C(26952), UINT16_C(64016), UINT16_C(11321) },
      { UINT16_C(   20), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
/*9*/
    { { UINT16_C(20194), UINT16_C(13489), UINT16_C(28148), UINT16_C(46609), UINT16_C(40781), UINT16_C(42334), UINT16_C(64675), UINT16_C(64675) },
      { UINT16_C(57383), UINT16_C(59409), UINT16_C(27037), UINT16_C(56767), UINT16_C(64675), UINT16_C(63245), UINT16_C(34097), UINT16_C( 5034) },
      { UINT16_C(  192), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(23763), UINT16_C(51016), UINT16_C(45134), UINT16_C( 6014), UINT16_C(61868), UINT16_C(16060), UINT16_C(23763), UINT16_C( 2515) },
      { UINT16_C(58473), UINT16_C(23763), UINT16_C(45134), UINT16_C(23763), UINT16_C(61868), UINT16_C(56808), UINT16_C(23763), UINT16_C(18928) },
      { UINT16_C(  192), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(14574), UINT16_C(47121), UINT16_C(36754), UINT16_C(35535), UINT16_C(42842), UINT16_C(19913), UINT16_C(39956), UINT16_C(32086) },
      { UINT16_C(14574), UINT16_C(52868), UINT16_C(26871), UINT16_C(41919), UINT16_C(42842), UINT16_C(53376), UINT16_C(28730), UINT16_C(10266) },
      { UINT16_C(  193), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(11177), UINT16_C(15328), UINT16_C(10720), UINT16_C( 9669), UINT16_C(36410), UINT16_C(20082), UINT16_C(10720), UINT16_C(43980) },
      { UINT16_C(20495), UINT16_C( 1657), UINT16_C(10720), UINT16_C( 5033), UINT16_C(10720), UINT16_C( 6883), UINT16_C(10720), UINT16_C(16963) },
      { UINT16_C(  192), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C( 9000), UINT16_C(57981), UINT16_C(17363), UINT16_C( 3335), UINT16_C(57981), UINT16_C(64604), UINT16_C(57981), UINT16_C(57981) },
      { UINT16_C( 8312), UINT16_C(57981), UINT16_C(65369), UINT16_C(14916), UINT16_C(57981), UINT16_C(49748), UINT16_C(57981), UINT16_C(19972) },
      { UINT16_C(  194), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(33467), UINT16_C(36400), UINT16_C(14533), UINT16_C(38555), UINT16_C(63409), UINT16_C(62354), UINT16_C(63409), UINT16_C(38979) },
      { UINT16_C(39514), UINT16_C(36400), UINT16_C(14533), UINT16_C(49901), UINT16_C(16950), UINT16_C(63409), UINT16_C(63409), UINT16_C(38057) },
      { UINT16_C(  194), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(55818), UINT16_C(53026), UINT16_C(48658), UINT16_C(50022), UINT16_C(63669), UINT16_C(54710), UINT16_C(64050), UINT16_C(35949) },
      { UINT16_C(13972), UINT16_C(11583), UINT16_C(11588), UINT16_C(31471), UINT16_C(29551), UINT16_C(18646), UINT16_C(32764), UINT16_C( 1757) },
      { UINT16_C(  192), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(65369), UINT16_C(27606), UINT16_C(65369), UINT16_C(29487), UINT16_C(58676), UINT16_C(26184), UINT16_C(65369), UINT16_C(29682) },
      { UINT16_C(13035), UINT16_C(65369), UINT16_C(65369), UINT16_C(52906), UINT16_C(32772), UINT16_C(   22), UINT16_C(65369), UINT16_C(22790) },
      { UINT16_C(  192), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(56563), UINT16_C(45253), UINT16_C(62488), UINT16_C(19747), UINT16_C(42528), UINT16_C(47539), UINT16_C(42528), UINT16_C( 3116) },
      { UINT16_C(52696), UINT16_C(14140), UINT16_C(62488), UINT16_C(25093), UINT16_C(42528), UINT16_C(26466), UINT16_C(42528), UINT16_C(  704) },
      { UINT16_C(  192), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, },
    { { UINT16_C(34117), UINT16_C(23986), UINT16_C(54905), UINT16_C(21418), UINT16_C(24129), UINT16_C(25100), UINT16_C(14340), UINT16_C(56430) },
      { UINT16_C(43526), UINT16_C(25619), UINT16_C( 6288), UINT16_C(21418), UINT16_C(10291), UINT16_C(25100), UINT16_C( 8080), UINT16_C(54596) },
      { UINT16_C(  192), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) }, }
  };

#define test_case_template(start, step, imm8) ({                         \
  for(size_t i = (start); i < (start) + (step); i++){                    \
    easysimd__m128i a = easysimd_mm_loadu_epi16((void *)&(test_vec[i].a[0]));  \
    easysimd__m128i b = easysimd_mm_loadu_epi16((void *)&(test_vec[i].b[0]));  \
    easysimd__m128i r = easysimd_mm_loadu_epi16((void *)&(test_vec[i].r[0]));  \
    easysimd__m128i ret = easysimd_mm_cmpestrm(a, 2, b, 6, (imm8));            \
    easysimd_assert_m128i_u16(r, ==, ret);}                                 \
})

  test_case_template( 0, 10, 1);
  test_case_template(10, 10, 3);
  test_case_template(20, 10, 5);
  test_case_template(30, 10, 7);
  test_case_template(40, 10, 9);

#undef test_case_template

  return 0;
}

static int
test_easysimd_mm_cmpestri_8bit(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    uint8_t  a[16];
    uint8_t  b[16];
    int32_t  r;
  } test_vec[] = {
/*0*/
    { { UINT8_C(130), UINT8_C( 99), UINT8_C(186), UINT8_C( 79), UINT8_C( 24), UINT8_C( 76), UINT8_C(114), UINT8_C(146),
        UINT8_C( 49), UINT8_C(194), UINT8_C(142), UINT8_C(114), UINT8_C(131), UINT8_C(140), UINT8_C( 72), UINT8_C(167) },
      { UINT8_C(172), UINT8_C( 64), UINT8_C(215), UINT8_C(130), UINT8_C(142), UINT8_C(152), UINT8_C(169), UINT8_C(233),
        UINT8_C(124), UINT8_C( 63), UINT8_C(228), UINT8_C( 13), UINT8_C( 78), UINT8_C( 14), UINT8_C( 74), UINT8_C(208) },
        INT32_C(   3)  , },
    { { UINT8_C(201), UINT8_C( 74), UINT8_C(155), UINT8_C( 76), UINT8_C(170), UINT8_C(201), UINT8_C( 78), UINT8_C( 17),
        UINT8_C(198), UINT8_C( 93), UINT8_C( 18), UINT8_C(119), UINT8_C(146), UINT8_C( 83), UINT8_C( 71), UINT8_C(140) },
      { UINT8_C( 39), UINT8_C( 59), UINT8_C(172), UINT8_C(197), UINT8_C( 76), UINT8_C(150), UINT8_C( 30), UINT8_C(242),
        UINT8_C( 75), UINT8_C(131), UINT8_C(119), UINT8_C(191), UINT8_C(220), UINT8_C(190), UINT8_C(219), UINT8_C(165) },
        INT32_C(   4)  , },
    { { UINT8_C( 28), UINT8_C( 80), UINT8_C(  3), UINT8_C(192), UINT8_C(224), UINT8_C(109), UINT8_C(130), UINT8_C(  7),
        UINT8_C(160), UINT8_C(121), UINT8_C( 47), UINT8_C(182), UINT8_C( 21), UINT8_C(159), UINT8_C(170), UINT8_C( 18) },
      { UINT8_C( 89), UINT8_C( 80), UINT8_C(  9), UINT8_C(185), UINT8_C(240), UINT8_C(231), UINT8_C(230), UINT8_C(121),
        UINT8_C( 63), UINT8_C( 49), UINT8_C(164), UINT8_C( 75), UINT8_C(209), UINT8_C(213), UINT8_C( 32), UINT8_C(238) },
        INT32_C(   1)  , },
    { { UINT8_C( 76), UINT8_C( 14), UINT8_C(249), UINT8_C( 19), UINT8_C( 57), UINT8_C( 27), UINT8_C(164), UINT8_C( 91),
        UINT8_C( 25), UINT8_C( 56), UINT8_C(234), UINT8_C(253), UINT8_C(238), UINT8_C(223), UINT8_C(227), UINT8_C(152) },
      { UINT8_C( 57), UINT8_C( 15), UINT8_C(228), UINT8_C(169), UINT8_C(245), UINT8_C(219), UINT8_C( 85), UINT8_C( 34),
        UINT8_C(251), UINT8_C(205), UINT8_C(154), UINT8_C( 49), UINT8_C(218), UINT8_C(151), UINT8_C(164), UINT8_C( 38) },
        INT32_C(   0)  , },
    { { UINT8_C(165), UINT8_C(157), UINT8_C( 57), UINT8_C(222), UINT8_C(185), UINT8_C(222), UINT8_C( 58), UINT8_C(210),
        UINT8_C( 22), UINT8_C( 36), UINT8_C(207), UINT8_C(  4), UINT8_C(  3), UINT8_C(179), UINT8_C(156), UINT8_C( 60) },
      { UINT8_C(194), UINT8_C(129), UINT8_C(230), UINT8_C(183), UINT8_C( 92), UINT8_C( 59), UINT8_C(217), UINT8_C( 88),
        UINT8_C(  8), UINT8_C(115), UINT8_C(137), UINT8_C(226), UINT8_C( 10), UINT8_C( 46), UINT8_C(  9), UINT8_C(175) },
        INT32_C(  16)  , },
/*2*/
    { { UINT8_C(143), UINT8_C(201), UINT8_C(102), UINT8_C(168), UINT8_C( 11), UINT8_C( 18), UINT8_C(200), UINT8_C( 28),
        UINT8_C(124), UINT8_C( 14), UINT8_C( 21), UINT8_C(238), UINT8_C(136), UINT8_C(144), UINT8_C(251), UINT8_C( 93) },
      { UINT8_C(220), UINT8_C(246), UINT8_C(226), UINT8_C(174), UINT8_C(164), UINT8_C(126), UINT8_C(234), UINT8_C(104),
        UINT8_C( 11), UINT8_C( 24), UINT8_C( 39), UINT8_C(216), UINT8_C(144), UINT8_C(152), UINT8_C( 43), UINT8_C( 31) },
        INT32_C(  16)  , },
    { { UINT8_C(158), UINT8_C(139), UINT8_C(230), UINT8_C( 65), UINT8_C(  8), UINT8_C(114), UINT8_C(175), UINT8_C( 83),
        UINT8_C( 15), UINT8_C(247), UINT8_C(246), UINT8_C(112), UINT8_C(109), UINT8_C(137), UINT8_C(116), UINT8_C(130) },
      { UINT8_C(108), UINT8_C(105), UINT8_C(172), UINT8_C( 49), UINT8_C( 10), UINT8_C(230), UINT8_C(223), UINT8_C(152),
        UINT8_C(245), UINT8_C(180), UINT8_C(176), UINT8_C(222), UINT8_C( 40), UINT8_C(173), UINT8_C(229), UINT8_C(199) },
        INT32_C(   5)  , },
    { { UINT8_C( 27), UINT8_C( 85), UINT8_C(242), UINT8_C(156), UINT8_C(132), UINT8_C(127), UINT8_C( 33), UINT8_C(249),
        UINT8_C( 90), UINT8_C( 64), UINT8_C(242), UINT8_C( 42), UINT8_C(213), UINT8_C(237), UINT8_C(138), UINT8_C(182) },
      { UINT8_C( 70), UINT8_C( 60), UINT8_C(140), UINT8_C( 37), UINT8_C(140), UINT8_C(237), UINT8_C( 27), UINT8_C(228),
        UINT8_C(  9), UINT8_C(250), UINT8_C( 52), UINT8_C(168), UINT8_C(131), UINT8_C( 48), UINT8_C( 38), UINT8_C(158) },
        INT32_C(   6)  , },
    { { UINT8_C(157), UINT8_C( 50), UINT8_C(133), UINT8_C( 30), UINT8_C( 73), UINT8_C(244), UINT8_C( 80), UINT8_C(212),
        UINT8_C(252), UINT8_C( 84), UINT8_C(120), UINT8_C(146), UINT8_C( 11), UINT8_C(229), UINT8_C(239), UINT8_C(251) },
      { UINT8_C(104), UINT8_C( 14), UINT8_C(178), UINT8_C( 11), UINT8_C(168), UINT8_C(127), UINT8_C( 56), UINT8_C( 83),
        UINT8_C(168), UINT8_C(227), UINT8_C(  2), UINT8_C(213), UINT8_C( 59), UINT8_C(187), UINT8_C(228), UINT8_C(216) },
        INT32_C(  16)  , },
    { { UINT8_C( 61), UINT8_C(245), UINT8_C(203), UINT8_C( 52), UINT8_C(110), UINT8_C(107), UINT8_C(129), UINT8_C(241),
        UINT8_C(110), UINT8_C( 58), UINT8_C(231), UINT8_C(241), UINT8_C( 24), UINT8_C( 86), UINT8_C(116), UINT8_C( 56) },
      { UINT8_C(142), UINT8_C(175), UINT8_C( 80), UINT8_C(110), UINT8_C(117), UINT8_C(169), UINT8_C(170), UINT8_C(170),
        UINT8_C( 28), UINT8_C(114), UINT8_C(132), UINT8_C(204), UINT8_C(160), UINT8_C(217), UINT8_C(202), UINT8_C(222) },
        INT32_C(   3)  , }
  };

#define test_case_template(start, step, imm8) ({                        \
  for(size_t i = (start); i < (start) + (step); i++){                   \
    easysimd__m128i a = easysimd_mm_loadu_epi8((void *)&(test_vec[i].a[0]));  \
    easysimd__m128i b = easysimd_mm_loadu_epi8((void *)&(test_vec[i].b[0]));  \
    int32_t      ret;                                                   \
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {                           \
      ret = easysimd_mm_cmpestri(a, 5, b, 8, imm8);                        \
    }                                                                   \
    EASYSIMD_TEST_PERF_END("_mm_cmpestri");                                \
    easysimd_assert_equal_i32(ret, test_vec[i].r);}                        \
})

  test_case_template( 0, 5, 0);
  test_case_template( 5, 5, 2);

#undef test_case_template

  return 0;
}

static int
test_easysimd_mm_cmpistrc_8bit(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint8_t a[16];
    uint8_t b[16];
    int32_t r;
  } test_vec[] = {
/*0*/
    { { UINT8_C( 87), UINT8_C( 84), UINT8_C(200), UINT8_C(246), UINT8_C( 87), UINT8_C(110), UINT8_C( 38), UINT8_C(142),
        UINT8_C(241), UINT8_C(196), UINT8_C(254), UINT8_C(178), UINT8_C(224), UINT8_C( 16), UINT8_C(162), UINT8_C(195) },
      { UINT8_C(  8), UINT8_C( 55), UINT8_C( 46), UINT8_C( 44), UINT8_C( 10), UINT8_C( 58), UINT8_C(220), UINT8_C(179),
        UINT8_C( 35), UINT8_C( 18), UINT8_C( 38), UINT8_C(231), UINT8_C(164), UINT8_C(  9), UINT8_C( 52), UINT8_C(251) },
       INT32_C(           1) },
    { { UINT8_C( 93), UINT8_C(252), UINT8_C(241), UINT8_C(181), UINT8_C(106), UINT8_C( 23), UINT8_C( 67), UINT8_C( 91),
        UINT8_C(219), UINT8_C( 65), UINT8_C( 14), UINT8_C(188), UINT8_C( 81), UINT8_C(176), UINT8_C(127), UINT8_C( 89) },
      { UINT8_C(231), UINT8_C(173), UINT8_C(133), UINT8_C(241), UINT8_C(232), UINT8_C( 97), UINT8_C(164), UINT8_C( 11),
        UINT8_C(115), UINT8_C(203), UINT8_C(242), UINT8_C( 23), UINT8_C(212), UINT8_C( 39), UINT8_C( 18), UINT8_C( 50) },
       INT32_C(           1) },
    { { UINT8_C( 35), UINT8_C(  3), UINT8_C(231), UINT8_C(142), UINT8_C( 26), UINT8_C( 42), UINT8_C(233), UINT8_C(246),
        UINT8_C(107), UINT8_C(247), UINT8_C(178), UINT8_C(188), UINT8_C(168), UINT8_C( 49), UINT8_C( 21), UINT8_C(143) },
      { UINT8_C(223), UINT8_C(154), UINT8_C(129), UINT8_C(199), UINT8_C(251), UINT8_C( 37), UINT8_C(210), UINT8_C(111),
        UINT8_C(240), UINT8_C(197), UINT8_C(134), UINT8_C(197), UINT8_C(236), UINT8_C(153), UINT8_C(247), UINT8_C( 15) },
       INT32_C(           1) },
    { { UINT8_C(156), UINT8_C(222), UINT8_C(157), UINT8_C(183), UINT8_C(  8), UINT8_C(135), UINT8_C(173), UINT8_C(115),
        UINT8_C(126), UINT8_C( 95), UINT8_C( 47), UINT8_C( 38), UINT8_C(144), UINT8_C( 68), UINT8_C(182), UINT8_C(111) },
      { UINT8_C(222), UINT8_C( 55), UINT8_C( 54), UINT8_C(217), UINT8_C( 92), UINT8_C(  9), UINT8_C( 72), UINT8_C( 77),
        UINT8_C(206), UINT8_C(207), UINT8_C( 18), UINT8_C(186), UINT8_C(104), UINT8_C(  9), UINT8_C(201), UINT8_C(  4) },
       INT32_C(           1) },
    { { UINT8_C(231), UINT8_C(103), UINT8_C(187), UINT8_C(239), UINT8_C(238), UINT8_C(104), UINT8_C( 98), UINT8_C(108),
        UINT8_C(199), UINT8_C(145), UINT8_C(147), UINT8_C( 88), UINT8_C(213), UINT8_C( 73), UINT8_C(199), UINT8_C(179) },
      { UINT8_C(128), UINT8_C(254), UINT8_C(140), UINT8_C(220), UINT8_C(  7), UINT8_C(213), UINT8_C( 41), UINT8_C(213),
        UINT8_C(164), UINT8_C( 59), UINT8_C(143), UINT8_C( 12), UINT8_C( 68), UINT8_C( 88), UINT8_C( 16), UINT8_C( 43) },
       INT32_C(           1) },
    { { UINT8_C(191), UINT8_C(204), UINT8_C( 26), UINT8_C(173), UINT8_C( 52), UINT8_C(124), UINT8_C( 26), UINT8_C(252),
        UINT8_C( 13), UINT8_C(173), UINT8_C( 84), UINT8_C(226), UINT8_C(246), UINT8_C( 27), UINT8_C(149), UINT8_C(118) },
      { UINT8_C( 25), UINT8_C( 34), UINT8_C( 82), UINT8_C( 32), UINT8_C(247), UINT8_C(124), UINT8_C(245), UINT8_C(155),
        UINT8_C(183), UINT8_C(132), UINT8_C(167), UINT8_C(252), UINT8_C(221), UINT8_C(183), UINT8_C( 39), UINT8_C(156) },
       INT32_C(           1) },
    { { UINT8_C(131), UINT8_C( 66), UINT8_C( 74), UINT8_C(184), UINT8_C(190), UINT8_C(100), UINT8_C(180), UINT8_C(204),
        UINT8_C( 17), UINT8_C(  8), UINT8_C(174), UINT8_C(  7), UINT8_C( 35), UINT8_C( 68), UINT8_C(125), UINT8_C( 61) },
      { UINT8_C(102), UINT8_C(207), UINT8_C( 93), UINT8_C( 93), UINT8_C( 75), UINT8_C( 83), UINT8_C(248), UINT8_C(  3),
        UINT8_C(215), UINT8_C(159),    UINT8_MAX, UINT8_C(180), UINT8_C( 86), UINT8_C( 38), UINT8_C( 81), UINT8_C(218) },
       INT32_C(           1) },
    { { UINT8_C(104), UINT8_C(155), UINT8_C(146), UINT8_C( 39),    UINT8_MAX, UINT8_C( 70), UINT8_C(243), UINT8_C( 16),
        UINT8_C( 78), UINT8_C(161), UINT8_C( 23), UINT8_C(113), UINT8_C(229), UINT8_C(148), UINT8_C(174), UINT8_C( 75) },
      { UINT8_C( 99), UINT8_C( 12), UINT8_C(168), UINT8_C(175), UINT8_C( 95), UINT8_C(160), UINT8_C(178), UINT8_C( 54),
        UINT8_C( 63), UINT8_C(177), UINT8_C(235), UINT8_C(150), UINT8_C(215), UINT8_C( 60), UINT8_C(112), UINT8_C( 64) },
       INT32_C(           0) }
  };
                      \
  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi8((void *)&(test_vec[i].a[0]));
    easysimd__m128i b = easysimd_mm_loadu_epi8((void *)&(test_vec[i].b[0]));
    int32_t r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpistrc(a, b, 0);
    } EASYSIMD_TEST_PERF_END("_mm_cmpistrc_8bit");
    easysimd_assert_equal_i32(r, test_vec[i].r);
  }                                
  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u8x16();
    easysimd__m128i b = easysimd_test_x86_random_u8x16();
    int32_t r = easysimd_mm_cmpistrc(a, b, 0);

    easysimd_test_x86_write_u8x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u8x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cmpistrc_16bit(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    uint16_t a[8];
    uint16_t b[8];
    int32_t r;
  } test_vec[] = {
/*0*/
    { { UINT16_C(  727), UINT16_C(54887), UINT16_C(23112), UINT16_C(38630), UINT16_C(65019), UINT16_C(57607), UINT16_C(46737), UINT16_C(62508) },
      { UINT16_C(54722), UINT16_C( 8611), UINT16_C(21877), UINT16_C(46423), UINT16_C(16902), UINT16_C(56907), UINT16_C(47998), UINT16_C(21790) },
       INT32_C(           0) },
    { { UINT16_C(34237), UINT16_C( 1323), UINT16_C( 4575), UINT16_C(55963), UINT16_C(41486), UINT16_C(40891), UINT16_C(59480), UINT16_C( 6804) },
      { UINT16_C(14269), UINT16_C(12859), UINT16_C(37773), UINT16_C(37863), UINT16_C(13013), UINT16_C(21617), UINT16_C(36845), UINT16_C(43689) },
       INT32_C(           1) },
    { { UINT16_C(54548), UINT16_C(62383), UINT16_C(19174), UINT16_C(62926), UINT16_C(35309), UINT16_C(17812), UINT16_C(10353), UINT16_C(11872) },
      { UINT16_C(39776), UINT16_C(60769), UINT16_C(18478), UINT16_C( 1152), UINT16_C(62075), UINT16_C(26712), UINT16_C(  385), UINT16_C(38419) },
       INT32_C(           1) },
    { { UINT16_C(49878), UINT16_C(48521), UINT16_C(22285), UINT16_C(64178), UINT16_C(18145), UINT16_C(21055), UINT16_C(40815), UINT16_C(53121) },
      { UINT16_C(57915), UINT16_C(27068), UINT16_C(15402), UINT16_C(42349), UINT16_C(50478), UINT16_C(45070), UINT16_C( 8647), UINT16_C(40262) },
       INT32_C(           1) },
    { { UINT16_C(53219), UINT16_C(61530), UINT16_C( 3111), UINT16_C( 2282), UINT16_C(10835), UINT16_C(49754), UINT16_C(56265), UINT16_C( 1169) },
      { UINT16_C(19901), UINT16_C(59502), UINT16_C(56201), UINT16_C(47245), UINT16_C(39841), UINT16_C(26728), UINT16_C(44732), UINT16_C(40965) },
       INT32_C(           1) },
    { { UINT16_C(24701), UINT16_C(42128), UINT16_C(31596), UINT16_C(49068), UINT16_C( 1957), UINT16_C(28289), UINT16_C( 4834), UINT16_C(41075) },
      { UINT16_C(57695), UINT16_C(59784), UINT16_C( 5564), UINT16_C(23969), UINT16_C( 2481), UINT16_C(28101), UINT16_C(52151), UINT16_C(13325) },
       INT32_C(           0) },
    { { UINT16_C(40491), UINT16_C(38873), UINT16_C(34073), UINT16_C(48727), UINT16_C(55436), UINT16_C(28460), UINT16_C(40939), UINT16_C(18959) },
      { UINT16_C(38784), UINT16_C(15667), UINT16_C(54444), UINT16_C(23962), UINT16_C(24797), UINT16_C(38091), UINT16_C(55339), UINT16_C(22217) },
       INT32_C(           1) },
    { { UINT16_C(41590), UINT16_C(36845), UINT16_C(17447), UINT16_C(46157), UINT16_C(31261), UINT16_C( 2083), UINT16_C(12825), UINT16_C(39506) },
      { UINT16_C(34505), UINT16_C(30167), UINT16_C(29018), UINT16_C(14547), UINT16_C(40657), UINT16_C(64716), UINT16_C(38262), UINT16_C(60754) },
       INT32_C(           1) }
  };

  for(size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16((void *)&(test_vec[i].a[0]));
    easysimd__m128i b = easysimd_mm_loadu_epi16((void *)&(test_vec[i].b[0]));
    int32_t r;
#ifndef EASYSIMD_ENABLE_TEST_PERF
    r = easysimd_mm_cmpistrc(a, b, 0);
#else
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cmpistrc(a, b, 0);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cmpistrc_16bit");
#endif
    easysimd_assert_equal_i32(r, test_vec[i].r);
  }                                
  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_u16x8();
    easysimd__m128i b = easysimd_test_x86_random_u16x8();
    int32_t r = easysimd_mm_cmpistrc(a, b, 0);

    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_u16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}
#endif

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpestrs_8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpestrs_16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpestrz_8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpestrz_16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpgt_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpistrs_8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpistrs_16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpistrz_8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpistrz_16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_crc32_u8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_crc32_u16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_crc32_u32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_crc32_u64)
#if ((defined(EASYSIMD_ARM_NEON_A64V8_NATIVE) || defined(EASYSIMD_ARM_SVE_NATIVE) || defined(EASYSIMD_X86_AVX_NATIVE)) && (!defined(__clang__)))
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpistrm_8bit)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpistrm_16bit)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpestrm_8bit)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpestrm_16bit)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpestri_8bit)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpistrc_8bit)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cmpistrc_16bit)
#endif
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/test-x86-footer.h>
