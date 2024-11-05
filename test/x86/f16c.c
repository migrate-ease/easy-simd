/* Copyright (c) 2020 Evan Nemerson <evan@nemerson.com>
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

#define EASYSIMD_TESTS_CURRENT_ISAX f16c
#include <easysimd/x86/f16c.h>
#include <test/x86/test-avx.h>

static int
test_easysimd_mm_cvtps_ph (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const int16_t r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -922.45), EASYSIMD_FLOAT32_C(  -417.52), EASYSIMD_FLOAT32_C(   576.56), EASYSIMD_FLOAT32_C(   -16.40) },
      { -INT16_C(  7371), -INT16_C(  8570),  INT16_C( 24705), -INT16_C( 13286),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { EASYSIMD_FLOAT32_C(  -847.35), EASYSIMD_FLOAT32_C(  -868.69), EASYSIMD_FLOAT32_C(   190.03), EASYSIMD_FLOAT32_C(  -263.75) },
      { -INT16_C(  7521), -INT16_C(  7479),  INT16_C( 23024), -INT16_C(  9185),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { EASYSIMD_FLOAT32_C(   550.95), EASYSIMD_FLOAT32_C(   691.22), EASYSIMD_FLOAT32_C(   972.58), EASYSIMD_FLOAT32_C(   645.93) },
      {  INT16_C( 24654),  INT16_C( 24934),  INT16_C( 25497),  INT16_C( 24844),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { EASYSIMD_FLOAT32_C(  -961.75), EASYSIMD_FLOAT32_C(   626.33), EASYSIMD_FLOAT32_C(   597.48), EASYSIMD_FLOAT32_C(   793.15) },
      { -INT16_C(  7292),  INT16_C( 24805),  INT16_C( 24747),  INT16_C( 25138),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { EASYSIMD_FLOAT32_C(  -582.37), EASYSIMD_FLOAT32_C(  -225.09), EASYSIMD_FLOAT32_C(   -65.32), EASYSIMD_FLOAT32_C(   452.55) },
      { -INT16_C(  8051), -INT16_C(  9463), -INT16_C( 11243),  INT16_C( 24338),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { EASYSIMD_FLOAT32_C(   125.78), EASYSIMD_FLOAT32_C(  -683.39), EASYSIMD_FLOAT32_C(  -348.27), EASYSIMD_FLOAT32_C(  -309.07) },
      {  INT16_C( 22492), -INT16_C(  7849), -INT16_C(  8847), -INT16_C(  9004),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { EASYSIMD_FLOAT32_C(  -141.60), EASYSIMD_FLOAT32_C(   503.26), EASYSIMD_FLOAT32_C(  -451.69), EASYSIMD_FLOAT32_C(  -298.51) },
      { -INT16_C( 10131),  INT16_C( 24541), -INT16_C(  8433), -INT16_C(  9046),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { EASYSIMD_FLOAT32_C(   899.79), EASYSIMD_FLOAT32_C(   611.12), EASYSIMD_FLOAT32_C(  -363.24), EASYSIMD_FLOAT32_C(   977.33) },
      {  INT16_C( 25352),  INT16_C( 24774), -INT16_C(  8787),  INT16_C( 25507),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128i r;
#ifndef EASYSIMD_ENABLE_TEST_PERF
    r = easysimd_mm_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC);
#else
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cvtps_ph");
#endif
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_x_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(-1000.0f, 1000.0f);
    easysimd__m128i r = easysimd_mm_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_cvtph_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t a[8];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { UINT16_C(57665), UINT16_C(57418), UINT16_C(25491), UINT16_C(23593), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) },
      { EASYSIMD_FLOAT32_C(  -672.50), EASYSIMD_FLOAT32_C(  -549.00), EASYSIMD_FLOAT32_C(   969.50), EASYSIMD_FLOAT32_C(   266.25) } },
    { { UINT16_C(24529), UINT16_C(25120), UINT16_C(57577), UINT16_C(55544), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) },
      { EASYSIMD_FLOAT32_C(   500.25), EASYSIMD_FLOAT32_C(   784.00), EASYSIMD_FLOAT32_C(  -628.50), EASYSIMD_FLOAT32_C(  -159.00) } },
    { { UINT16_C(56338), UINT16_C(24788), UINT16_C(58146), UINT16_C(25256), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) },
      { EASYSIMD_FLOAT32_C(  -260.50), EASYSIMD_FLOAT32_C(   618.00), EASYSIMD_FLOAT32_C(  -913.00), EASYSIMD_FLOAT32_C(   852.00) } },
    { { UINT16_C(25457), UINT16_C(24637), UINT16_C(23890), UINT16_C(56982), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) },
      { EASYSIMD_FLOAT32_C(   952.50), EASYSIMD_FLOAT32_C(   542.50), EASYSIMD_FLOAT32_C(   340.50), EASYSIMD_FLOAT32_C(  -421.50) } },
    { { UINT16_C(22950), UINT16_C(21640), UINT16_C(54779), UINT16_C(22774), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) },
      { EASYSIMD_FLOAT32_C(   180.75), EASYSIMD_FLOAT32_C(    72.50), EASYSIMD_FLOAT32_C(   -95.69), EASYSIMD_FLOAT32_C(   158.75) } },
    { { UINT16_C(24293), UINT16_C(55275), UINT16_C(56350), UINT16_C(24325), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) },
      { EASYSIMD_FLOAT32_C(   441.25), EASYSIMD_FLOAT32_C(  -126.69), EASYSIMD_FLOAT32_C(  -263.50), EASYSIMD_FLOAT32_C(   449.25) } },
    { { UINT16_C(22402), UINT16_C(24502), UINT16_C(57033), UINT16_C(56678), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) },
      { EASYSIMD_FLOAT32_C(   120.12), EASYSIMD_FLOAT32_C(   493.50), EASYSIMD_FLOAT32_C(  -434.25), EASYSIMD_FLOAT32_C(  -345.50) } },
    { { UINT16_C(57623), UINT16_C(22385), UINT16_C(56989), UINT16_C(56592), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0), UINT16_C(    0) },
      { EASYSIMD_FLOAT32_C(  -651.50), EASYSIMD_FLOAT32_C(   119.06), EASYSIMD_FLOAT32_C(  -423.25), EASYSIMD_FLOAT32_C(  -324.00) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_x_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128 r;
#ifndef EASYSIMD_ENABLE_TEST_PERF
    r = easysimd_mm_cvtph_ps(a);
#else
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_cvtph_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_cvtph_ps");
#endif
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_mm_cvtps_ph(easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0)), EASYSIMD_MM_FROUND_NO_EXC);
    easysimd__m128 r = easysimd_mm_cvtph_ps(a);

    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_f16c_round_trip (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    #if !defined(EASYSIMD_FAST_NANS)
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -423.73),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   160.78) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -423.73),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   160.78) } },
    #endif
    { { EASYSIMD_FLOAT32_C(  -981.67), EASYSIMD_FLOAT32_C(   318.53), EASYSIMD_FLOAT32_C(  -890.10), EASYSIMD_FLOAT32_C(   443.08) },
      { EASYSIMD_FLOAT32_C(  -981.67), EASYSIMD_FLOAT32_C(   318.53), EASYSIMD_FLOAT32_C(  -890.10), EASYSIMD_FLOAT32_C(   443.08) } },
    { { EASYSIMD_FLOAT32_C(  -316.07), EASYSIMD_FLOAT32_C(   436.20), EASYSIMD_FLOAT32_C(   506.18), EASYSIMD_FLOAT32_C(   324.31) },
      { EASYSIMD_FLOAT32_C(  -316.07), EASYSIMD_FLOAT32_C(   436.20), EASYSIMD_FLOAT32_C(   506.18), EASYSIMD_FLOAT32_C(   324.31) } },
    { { EASYSIMD_FLOAT32_C(  -232.13), EASYSIMD_FLOAT32_C(   547.61), EASYSIMD_FLOAT32_C(   521.27), EASYSIMD_FLOAT32_C(  -153.90) },
      { EASYSIMD_FLOAT32_C(  -232.13), EASYSIMD_FLOAT32_C(   547.61), EASYSIMD_FLOAT32_C(   521.27), EASYSIMD_FLOAT32_C(  -153.90) } },
    { { EASYSIMD_FLOAT32_C(   819.91), EASYSIMD_FLOAT32_C(   215.00), EASYSIMD_FLOAT32_C(   715.88), EASYSIMD_FLOAT32_C(   525.54) },
      { EASYSIMD_FLOAT32_C(   819.91), EASYSIMD_FLOAT32_C(   215.00), EASYSIMD_FLOAT32_C(   715.88), EASYSIMD_FLOAT32_C(   525.54) } },
    { { EASYSIMD_FLOAT32_C(  -199.45), EASYSIMD_FLOAT32_C(  -914.59), EASYSIMD_FLOAT32_C(  -600.24), EASYSIMD_FLOAT32_C(  -579.28) },
      { EASYSIMD_FLOAT32_C(  -199.45), EASYSIMD_FLOAT32_C(  -914.59), EASYSIMD_FLOAT32_C(  -600.24), EASYSIMD_FLOAT32_C(  -579.28) } },
    { { EASYSIMD_FLOAT32_C(   950.34), EASYSIMD_FLOAT32_C(   142.00), EASYSIMD_FLOAT32_C(  -931.01), EASYSIMD_FLOAT32_C(   915.71) },
      { EASYSIMD_FLOAT32_C(   950.34), EASYSIMD_FLOAT32_C(   142.00), EASYSIMD_FLOAT32_C(  -931.01), EASYSIMD_FLOAT32_C(   915.71) } },
    { { EASYSIMD_FLOAT32_C(  -390.34), EASYSIMD_FLOAT32_C(   580.78), EASYSIMD_FLOAT32_C(  -866.48), EASYSIMD_FLOAT32_C(  -588.08) },
      { EASYSIMD_FLOAT32_C(  -390.34), EASYSIMD_FLOAT32_C(   580.78), EASYSIMD_FLOAT32_C(  -866.48), EASYSIMD_FLOAT32_C(  -588.08) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 r = easysimd_mm_cvtph_ps(easysimd_mm_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC));
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 0);
  }

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m128)];
  easysimd_test_x86_random_f32x4_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_extract_f32x4(i, 2, 0, values);
    easysimd__m128 r = a; // easysimd_mm_cvtph_ps(easysimd_mm_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC));

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cvtps_ph (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const int16_t r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -266.25), EASYSIMD_FLOAT32_C(  -994.56), EASYSIMD_FLOAT32_C(  -351.71), EASYSIMD_FLOAT32_C(   290.36),
        EASYSIMD_FLOAT32_C(  -637.78), EASYSIMD_FLOAT32_C(   495.06), EASYSIMD_FLOAT32_C(  -635.46), EASYSIMD_FLOAT32_C(  -352.22) },
      { -INT16_C(  9175), -INT16_C(  7227), -INT16_C(  8833),  INT16_C( 23689), -INT16_C(  7940),  INT16_C( 24508), -INT16_C(  7945), -INT16_C(  8831) } },
    { { EASYSIMD_FLOAT32_C(  -832.50), EASYSIMD_FLOAT32_C(   748.25), EASYSIMD_FLOAT32_C(  -953.26), EASYSIMD_FLOAT32_C(  -335.37),
        EASYSIMD_FLOAT32_C(   855.42), EASYSIMD_FLOAT32_C(  -551.65), EASYSIMD_FLOAT32_C(   369.44), EASYSIMD_FLOAT32_C(   315.27) },
      { -INT16_C(  7551),  INT16_C( 25048), -INT16_C(  7309), -INT16_C(  8899),  INT16_C( 25263), -INT16_C(  8113),  INT16_C( 24006),  INT16_C( 23789) } },
    { { EASYSIMD_FLOAT32_C(  -690.18), EASYSIMD_FLOAT32_C(   370.38), EASYSIMD_FLOAT32_C(   -92.70), EASYSIMD_FLOAT32_C(   797.51),
        EASYSIMD_FLOAT32_C(   286.45), EASYSIMD_FLOAT32_C(   853.41), EASYSIMD_FLOAT32_C(  -941.28), EASYSIMD_FLOAT32_C(   941.87) },
      { -INT16_C(  7836),  INT16_C( 24010), -INT16_C( 10805),  INT16_C( 25147),  INT16_C( 23674),  INT16_C( 25259), -INT16_C(  7333),  INT16_C( 25436) } },
    { { EASYSIMD_FLOAT32_C(   873.16), EASYSIMD_FLOAT32_C(   513.64), EASYSIMD_FLOAT32_C(   399.26), EASYSIMD_FLOAT32_C(  -985.07),
        EASYSIMD_FLOAT32_C(   503.49), EASYSIMD_FLOAT32_C(  -978.18), EASYSIMD_FLOAT32_C(  -844.37), EASYSIMD_FLOAT32_C(  -762.76) },
      {  INT16_C( 25298),  INT16_C( 24579),  INT16_C( 24125), -INT16_C(  7246),  INT16_C( 24542), -INT16_C(  7260), -INT16_C(  7527), -INT16_C(  7690) } },
    { { EASYSIMD_FLOAT32_C(  -972.74), EASYSIMD_FLOAT32_C(  -196.09), EASYSIMD_FLOAT32_C(   527.61), EASYSIMD_FLOAT32_C(  -610.53),
        EASYSIMD_FLOAT32_C(  -701.03), EASYSIMD_FLOAT32_C(   892.15), EASYSIMD_FLOAT32_C(    37.25), EASYSIMD_FLOAT32_C(  -533.53) },
      { -INT16_C(  7271), -INT16_C(  9695),  INT16_C( 24607), -INT16_C(  7995), -INT16_C(  7814),  INT16_C( 25336),  INT16_C( 20648), -INT16_C(  8149) } },
    { { EASYSIMD_FLOAT32_C(   640.40), EASYSIMD_FLOAT32_C(    83.99), EASYSIMD_FLOAT32_C(   131.10), EASYSIMD_FLOAT32_C(   495.82),
        EASYSIMD_FLOAT32_C(   532.34), EASYSIMD_FLOAT32_C(  -499.46), EASYSIMD_FLOAT32_C(  -188.91), EASYSIMD_FLOAT32_C(   842.16) },
      {  INT16_C( 24833),  INT16_C( 21824),  INT16_C( 22553),  INT16_C( 24511),  INT16_C( 24617), -INT16_C(  8242), -INT16_C(  9753),  INT16_C( 25236) } },
    { { EASYSIMD_FLOAT32_C(   870.92), EASYSIMD_FLOAT32_C(   718.39), EASYSIMD_FLOAT32_C(   639.67), EASYSIMD_FLOAT32_C(   157.37),
        EASYSIMD_FLOAT32_C(   571.81), EASYSIMD_FLOAT32_C(   698.39), EASYSIMD_FLOAT32_C(    99.25), EASYSIMD_FLOAT32_C(   444.96) },
      {  INT16_C( 25294),  INT16_C( 24989),  INT16_C( 24831),  INT16_C( 22763),  INT16_C( 24696),  INT16_C( 24949),  INT16_C( 22068),  INT16_C( 24308) } },
    { { EASYSIMD_FLOAT32_C(   212.02), EASYSIMD_FLOAT32_C(  -501.49), EASYSIMD_FLOAT32_C(   459.89), EASYSIMD_FLOAT32_C(  -284.49),
        EASYSIMD_FLOAT32_C(  -479.67), EASYSIMD_FLOAT32_C(   615.52), EASYSIMD_FLOAT32_C(   -47.25), EASYSIMD_FLOAT32_C(  -452.42) },
      {  INT16_C( 23200), -INT16_C(  8234),  INT16_C( 24368), -INT16_C(  9102), -INT16_C(  8321),  INT16_C( 24783), -INT16_C( 11800), -INT16_C(  8430) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m128i r;
#ifndef EASYSIMD_ENABLE_TEST_PERF
    r = easysimd_mm256_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC);
#else
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cvtps_ph");
#endif
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_x_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_f32x8(-1000.0f, 1000.0f);
    easysimd__m128i r = easysimd_mm256_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_cvtph_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
      { { UINT16_C(57990), UINT16_C(54973), UINT16_C(57953), UINT16_C(57620), UINT16_C(56879), UINT16_C(24954), UINT16_C(22493), UINT16_C(23782) },
        { EASYSIMD_FLOAT32_C(  -835.00), EASYSIMD_FLOAT32_C(  -107.81), EASYSIMD_FLOAT32_C(  -816.50), EASYSIMD_FLOAT32_C(  -650.00),
          EASYSIMD_FLOAT32_C(  -395.75), EASYSIMD_FLOAT32_C(   701.00), EASYSIMD_FLOAT32_C(   125.81), EASYSIMD_FLOAT32_C(   313.50) } },
      { { UINT16_C(57584), UINT16_C(55980), UINT16_C(25240), UINT16_C(56663), UINT16_C(22603), UINT16_C(58177), UINT16_C(25022), UINT16_C(25135) },
        { EASYSIMD_FLOAT32_C(  -632.00), EASYSIMD_FLOAT32_C(  -213.50), EASYSIMD_FLOAT32_C(   844.00), EASYSIMD_FLOAT32_C(  -341.75),
          EASYSIMD_FLOAT32_C(   137.38), EASYSIMD_FLOAT32_C(  -928.50), EASYSIMD_FLOAT32_C(   735.00), EASYSIMD_FLOAT32_C(   791.50) } },
      { { UINT16_C(23799), UINT16_C(58131), UINT16_C(56002), UINT16_C(56914), UINT16_C(58108), UINT16_C(21095), UINT16_C(57559), UINT16_C(57517) },
        { EASYSIMD_FLOAT32_C(   317.75), EASYSIMD_FLOAT32_C(  -905.50), EASYSIMD_FLOAT32_C(  -216.25), EASYSIMD_FLOAT32_C(  -404.50),
          EASYSIMD_FLOAT32_C(  -894.00), EASYSIMD_FLOAT32_C(    51.22), EASYSIMD_FLOAT32_C(  -619.50), EASYSIMD_FLOAT32_C(  -598.50) } },
      { { UINT16_C(23890), UINT16_C(24728), UINT16_C(54649), UINT16_C(57838), UINT16_C(57887), UINT16_C(24135), UINT16_C(57569), UINT16_C(57557) },
        { EASYSIMD_FLOAT32_C(   340.50), EASYSIMD_FLOAT32_C(   588.00), EASYSIMD_FLOAT32_C(   -87.56), EASYSIMD_FLOAT32_C(  -759.00),
          EASYSIMD_FLOAT32_C(  -783.50), EASYSIMD_FLOAT32_C(   401.75), EASYSIMD_FLOAT32_C(  -624.50), EASYSIMD_FLOAT32_C(  -618.50) } },
      { { UINT16_C(57732), UINT16_C(57060), UINT16_C(56369), UINT16_C(54878), UINT16_C(57800), UINT16_C(25267), UINT16_C(57897), UINT16_C(56785) },
        { EASYSIMD_FLOAT32_C(  -706.00), EASYSIMD_FLOAT32_C(  -441.00), EASYSIMD_FLOAT32_C(  -268.25), EASYSIMD_FLOAT32_C(  -101.88),
          EASYSIMD_FLOAT32_C(  -740.00), EASYSIMD_FLOAT32_C(   857.50), EASYSIMD_FLOAT32_C(  -788.50), EASYSIMD_FLOAT32_C(  -372.25) } },
      { { UINT16_C(56720), UINT16_C(58209), UINT16_C(23672), UINT16_C(25115), UINT16_C(58065), UINT16_C(19770), UINT16_C(24698), UINT16_C(24308) },
        { EASYSIMD_FLOAT32_C(  -356.00), EASYSIMD_FLOAT32_C(  -944.50), EASYSIMD_FLOAT32_C(   286.00), EASYSIMD_FLOAT32_C(   781.50),
          EASYSIMD_FLOAT32_C(  -872.50), EASYSIMD_FLOAT32_C(    20.91), EASYSIMD_FLOAT32_C(   573.00), EASYSIMD_FLOAT32_C(   445.00) } },
      { { UINT16_C(22324), UINT16_C(57607), UINT16_C(58239), UINT16_C(23275), UINT16_C(24160), UINT16_C(57478), UINT16_C(24797), UINT16_C(56285) },
        { EASYSIMD_FLOAT32_C(   115.25), EASYSIMD_FLOAT32_C(  -643.50), EASYSIMD_FLOAT32_C(  -959.50), EASYSIMD_FLOAT32_C(   221.38),
          EASYSIMD_FLOAT32_C(   408.00), EASYSIMD_FLOAT32_C(  -579.00), EASYSIMD_FLOAT32_C(   622.50), EASYSIMD_FLOAT32_C(  -251.62) } },
      { { UINT16_C(58301), UINT16_C(57156), UINT16_C(51507), UINT16_C(57868), UINT16_C(25426), UINT16_C(23988), UINT16_C(56866), UINT16_C(57859) },
        { EASYSIMD_FLOAT32_C(  -990.50), EASYSIMD_FLOAT32_C(  -465.00), EASYSIMD_FLOAT32_C(   -10.40), EASYSIMD_FLOAT32_C(  -774.00),
          EASYSIMD_FLOAT32_C(   937.00), EASYSIMD_FLOAT32_C(   365.00), EASYSIMD_FLOAT32_C(  -392.50), EASYSIMD_FLOAT32_C(  -769.50) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m256 r;
#ifndef EASYSIMD_ENABLE_TEST_PERF
    r = easysimd_mm256_cvtph_ps(a);
#else
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_cvtph_ps(a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_cvtph_ps");
#endif
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_mm256_cvtps_ph(easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0)), EASYSIMD_MM_FROUND_NO_EXC);
    easysimd__m256 r = easysimd_mm256_cvtph_ps(a);

    easysimd_test_x86_write_u16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_f16c_round_trip (EASYSIMD_MUNIT_TEST_ARGS) {
  #if 1
    static const struct {
      const easysimd_float32 a[8];
      const easysimd_float32 r[8];
    } test_vec[] = {
      #if !defined(EASYSIMD_FAST_NANS)
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -474.87),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -588.08),
        EASYSIMD_FLOAT32_C(   450.80), EASYSIMD_FLOAT32_C(   475.77), EASYSIMD_FLOAT32_C(   919.35), EASYSIMD_FLOAT32_C(   544.67) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -474.87),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -588.08),
        EASYSIMD_FLOAT32_C(   450.80), EASYSIMD_FLOAT32_C(   475.77), EASYSIMD_FLOAT32_C(   919.35), EASYSIMD_FLOAT32_C(   544.67) } },
      #endif
    { { EASYSIMD_FLOAT32_C(  -372.84), EASYSIMD_FLOAT32_C(  -960.44), EASYSIMD_FLOAT32_C(   -69.78), EASYSIMD_FLOAT32_C(  -757.33),
        EASYSIMD_FLOAT32_C(   645.16), EASYSIMD_FLOAT32_C(   893.71), EASYSIMD_FLOAT32_C(   672.61), EASYSIMD_FLOAT32_C(  -845.50) },
      { EASYSIMD_FLOAT32_C(  -372.84), EASYSIMD_FLOAT32_C(  -960.44), EASYSIMD_FLOAT32_C(   -69.78), EASYSIMD_FLOAT32_C(  -757.33),
        EASYSIMD_FLOAT32_C(   645.16), EASYSIMD_FLOAT32_C(   893.71), EASYSIMD_FLOAT32_C(   672.61), EASYSIMD_FLOAT32_C(  -845.50) } },
    { { EASYSIMD_FLOAT32_C(  -863.46), EASYSIMD_FLOAT32_C(    81.15), EASYSIMD_FLOAT32_C(  -636.08), EASYSIMD_FLOAT32_C(   587.35),
        EASYSIMD_FLOAT32_C(  -443.08), EASYSIMD_FLOAT32_C(  -716.73), EASYSIMD_FLOAT32_C(   132.01), EASYSIMD_FLOAT32_C(   728.65) },
      { EASYSIMD_FLOAT32_C(  -863.46), EASYSIMD_FLOAT32_C(    81.15), EASYSIMD_FLOAT32_C(  -636.08), EASYSIMD_FLOAT32_C(   587.35),
        EASYSIMD_FLOAT32_C(  -443.08), EASYSIMD_FLOAT32_C(  -716.73), EASYSIMD_FLOAT32_C(   132.01), EASYSIMD_FLOAT32_C(   728.65) } },
    { { EASYSIMD_FLOAT32_C(   708.45), EASYSIMD_FLOAT32_C(  -619.18), EASYSIMD_FLOAT32_C(   477.33), EASYSIMD_FLOAT32_C(   353.61),
        EASYSIMD_FLOAT32_C(  -725.48), EASYSIMD_FLOAT32_C(   149.94), EASYSIMD_FLOAT32_C(   508.10), EASYSIMD_FLOAT32_C(   238.34) },
      { EASYSIMD_FLOAT32_C(   708.45), EASYSIMD_FLOAT32_C(  -619.18), EASYSIMD_FLOAT32_C(   477.33), EASYSIMD_FLOAT32_C(   353.61),
        EASYSIMD_FLOAT32_C(  -725.48), EASYSIMD_FLOAT32_C(   149.94), EASYSIMD_FLOAT32_C(   508.10), EASYSIMD_FLOAT32_C(   238.34) } },
    { { EASYSIMD_FLOAT32_C(   491.58), EASYSIMD_FLOAT32_C(  -199.94), EASYSIMD_FLOAT32_C(   616.91), EASYSIMD_FLOAT32_C(  -951.51),
        EASYSIMD_FLOAT32_C(    83.34), EASYSIMD_FLOAT32_C(  -251.08), EASYSIMD_FLOAT32_C(   777.15), EASYSIMD_FLOAT32_C(  -532.99) },
      { EASYSIMD_FLOAT32_C(   491.58), EASYSIMD_FLOAT32_C(  -199.94), EASYSIMD_FLOAT32_C(   616.91), EASYSIMD_FLOAT32_C(  -951.51),
        EASYSIMD_FLOAT32_C(    83.34), EASYSIMD_FLOAT32_C(  -251.08), EASYSIMD_FLOAT32_C(   777.15), EASYSIMD_FLOAT32_C(  -532.99) } },
    { { EASYSIMD_FLOAT32_C(  -504.03), EASYSIMD_FLOAT32_C(    48.55), EASYSIMD_FLOAT32_C(  -511.25), EASYSIMD_FLOAT32_C(  -229.50),
        EASYSIMD_FLOAT32_C(  -801.51), EASYSIMD_FLOAT32_C(   996.85), EASYSIMD_FLOAT32_C(  -991.16), EASYSIMD_FLOAT32_C(   -14.58) },
      { EASYSIMD_FLOAT32_C(  -504.03), EASYSIMD_FLOAT32_C(    48.55), EASYSIMD_FLOAT32_C(  -511.25), EASYSIMD_FLOAT32_C(  -229.50),
        EASYSIMD_FLOAT32_C(  -801.51), EASYSIMD_FLOAT32_C(   996.85), EASYSIMD_FLOAT32_C(  -991.16), EASYSIMD_FLOAT32_C(   -14.58) } },
    { { EASYSIMD_FLOAT32_C(  -294.23), EASYSIMD_FLOAT32_C(   817.94), EASYSIMD_FLOAT32_C(  -541.68), EASYSIMD_FLOAT32_C(   789.10),
        EASYSIMD_FLOAT32_C(  -433.13), EASYSIMD_FLOAT32_C(  -764.53), EASYSIMD_FLOAT32_C(  -743.88), EASYSIMD_FLOAT32_C(  -704.08) },
      { EASYSIMD_FLOAT32_C(  -294.23), EASYSIMD_FLOAT32_C(   817.94), EASYSIMD_FLOAT32_C(  -541.68), EASYSIMD_FLOAT32_C(   789.10),
        EASYSIMD_FLOAT32_C(  -433.13), EASYSIMD_FLOAT32_C(  -764.53), EASYSIMD_FLOAT32_C(  -743.88), EASYSIMD_FLOAT32_C(  -704.08) } },
    { { EASYSIMD_FLOAT32_C(   252.24), EASYSIMD_FLOAT32_C(    43.86), EASYSIMD_FLOAT32_C(  -697.71), EASYSIMD_FLOAT32_C(   450.73),
        EASYSIMD_FLOAT32_C(    40.71), EASYSIMD_FLOAT32_C(  -688.87), EASYSIMD_FLOAT32_C(  -563.85), EASYSIMD_FLOAT32_C(   319.17) },
      { EASYSIMD_FLOAT32_C(   252.24), EASYSIMD_FLOAT32_C(    43.86), EASYSIMD_FLOAT32_C(  -697.71), EASYSIMD_FLOAT32_C(   450.73),
        EASYSIMD_FLOAT32_C(    40.71), EASYSIMD_FLOAT32_C(  -688.87), EASYSIMD_FLOAT32_C(  -563.85), EASYSIMD_FLOAT32_C(   319.17) } }
    };

    for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
      easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
      easysimd__m256 r = easysimd_mm256_cvtph_ps(easysimd_mm256_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC));
      easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 0);
    }

    return 0;
  #else
    fputc('\n', stdout);
    easysimd_float32 values[8 * 2 * sizeof(easysimd__m256)];
    easysimd_test_x86_random_f32x8_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

    for (size_t i = 0 ; i < 8 ; i++) {
      easysimd__m256 a = easysimd_test_x86_random_extract_f32x8(i, 2, 0, values);
      easysimd__m256 r = a; // easysimd_mm256_cvtph_ps(easysimd_mm256_cvtps_ph(a, EASYSIMD_MM_FROUND_NO_EXC));

      easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
    }
    return 1;
  #endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cvtps_ph)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_cvtph_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_f16c_round_trip)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cvtps_ph)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_cvtph_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_f16c_round_trip)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/test-x86-footer.h>
