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
 *   2020      kitegi <kitegi@users.noreply.github.com>
 */

#define EASYSIMD_TEST_X86_AVX512_INSN fnmsub

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/fnmsub.h>

static int
test_easysimd_mm_mask3_fnmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd__mmask8 k;
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -712.60), EASYSIMD_FLOAT32_C(   220.11), EASYSIMD_FLOAT32_C(  -369.14), EASYSIMD_FLOAT32_C(   411.96) },
      UINT8_C( 93),
      { EASYSIMD_FLOAT32_C(  -908.52), EASYSIMD_FLOAT32_C(   811.01), EASYSIMD_FLOAT32_C(  -307.45), EASYSIMD_FLOAT32_C(  -450.54) },
      { EASYSIMD_FLOAT32_C(  -623.44), EASYSIMD_FLOAT32_C(  -823.08), EASYSIMD_FLOAT32_C(  -416.95), EASYSIMD_FLOAT32_C(    97.71) },
      { EASYSIMD_FLOAT32_C(-646787.88), EASYSIMD_FLOAT32_C(  -823.08), EASYSIMD_FLOAT32_C(-113075.15), EASYSIMD_FLOAT32_C(185506.75) } },
    { { EASYSIMD_FLOAT32_C(  -976.56), EASYSIMD_FLOAT32_C(  -276.09), EASYSIMD_FLOAT32_C(  -906.71), EASYSIMD_FLOAT32_C(  -829.31) },
      UINT8_C( 85),
      { EASYSIMD_FLOAT32_C(   433.86), EASYSIMD_FLOAT32_C(  -637.29), EASYSIMD_FLOAT32_C(  -726.28), EASYSIMD_FLOAT32_C(  -562.66) },
      { EASYSIMD_FLOAT32_C(  -634.92), EASYSIMD_FLOAT32_C(  -827.22), EASYSIMD_FLOAT32_C(  -696.57), EASYSIMD_FLOAT32_C(  -741.48) },
      { EASYSIMD_FLOAT32_C(424325.22), EASYSIMD_FLOAT32_C(  -827.22), EASYSIMD_FLOAT32_C(-657828.81), EASYSIMD_FLOAT32_C(  -741.48) } },
    { { EASYSIMD_FLOAT32_C(   444.36), EASYSIMD_FLOAT32_C(  -212.39), EASYSIMD_FLOAT32_C(   891.21), EASYSIMD_FLOAT32_C(  -365.01) },
      UINT8_C(248),
      { EASYSIMD_FLOAT32_C(  -821.39), EASYSIMD_FLOAT32_C(   855.10), EASYSIMD_FLOAT32_C(   801.23), EASYSIMD_FLOAT32_C(   590.58) },
      { EASYSIMD_FLOAT32_C(   355.08), EASYSIMD_FLOAT32_C(   892.71), EASYSIMD_FLOAT32_C(   401.59), EASYSIMD_FLOAT32_C(  -952.38) },
      { EASYSIMD_FLOAT32_C(   355.08), EASYSIMD_FLOAT32_C(   892.71), EASYSIMD_FLOAT32_C(   401.59), EASYSIMD_FLOAT32_C(216520.00) } },
    { { EASYSIMD_FLOAT32_C(  -557.83), EASYSIMD_FLOAT32_C(   778.15), EASYSIMD_FLOAT32_C(  -775.45), EASYSIMD_FLOAT32_C(    25.23) },
      UINT8_C(166),
      { EASYSIMD_FLOAT32_C(  -752.02), EASYSIMD_FLOAT32_C(   749.14), EASYSIMD_FLOAT32_C(   -30.85), EASYSIMD_FLOAT32_C(  -581.32) },
      { EASYSIMD_FLOAT32_C(   -87.51), EASYSIMD_FLOAT32_C(  -596.99), EASYSIMD_FLOAT32_C(  -218.62), EASYSIMD_FLOAT32_C(   186.21) },
      { EASYSIMD_FLOAT32_C(   -87.51), EASYSIMD_FLOAT32_C(-582346.31), EASYSIMD_FLOAT32_C(-23704.01), EASYSIMD_FLOAT32_C(   186.21) } },
    { { EASYSIMD_FLOAT32_C(  -159.65), EASYSIMD_FLOAT32_C(   146.47), EASYSIMD_FLOAT32_C(   358.98), EASYSIMD_FLOAT32_C(   143.78) },
      UINT8_C( 55),
      { EASYSIMD_FLOAT32_C(  -196.66), EASYSIMD_FLOAT32_C(   931.38), EASYSIMD_FLOAT32_C(   296.20), EASYSIMD_FLOAT32_C(   438.33) },
      { EASYSIMD_FLOAT32_C(   101.75), EASYSIMD_FLOAT32_C(   474.81), EASYSIMD_FLOAT32_C(   293.42), EASYSIMD_FLOAT32_C(   -97.02) },
      { EASYSIMD_FLOAT32_C(-31498.52), EASYSIMD_FLOAT32_C(-136894.05), EASYSIMD_FLOAT32_C(-106623.30), EASYSIMD_FLOAT32_C(   -97.02) } },
    { { EASYSIMD_FLOAT32_C(    65.39), EASYSIMD_FLOAT32_C(  -351.50), EASYSIMD_FLOAT32_C(  -204.31), EASYSIMD_FLOAT32_C(  -533.02) },
      UINT8_C( 93),
      { EASYSIMD_FLOAT32_C(   237.86), EASYSIMD_FLOAT32_C(  -754.87), EASYSIMD_FLOAT32_C(   -79.33), EASYSIMD_FLOAT32_C(  -736.91) },
      { EASYSIMD_FLOAT32_C(   120.98), EASYSIMD_FLOAT32_C(   168.65), EASYSIMD_FLOAT32_C(  -987.78), EASYSIMD_FLOAT32_C(  -909.87) },
      { EASYSIMD_FLOAT32_C(-15674.65), EASYSIMD_FLOAT32_C(   168.65), EASYSIMD_FLOAT32_C(-15220.13), EASYSIMD_FLOAT32_C(-391877.91) } },
    { { EASYSIMD_FLOAT32_C(   587.33), EASYSIMD_FLOAT32_C(   -75.29), EASYSIMD_FLOAT32_C(  -506.86), EASYSIMD_FLOAT32_C(  -631.29) },
      UINT8_C(111),
      { EASYSIMD_FLOAT32_C(   333.49), EASYSIMD_FLOAT32_C(   515.18), EASYSIMD_FLOAT32_C(   469.90), EASYSIMD_FLOAT32_C(  -522.74) },
      { EASYSIMD_FLOAT32_C(   -79.83), EASYSIMD_FLOAT32_C(  -726.76), EASYSIMD_FLOAT32_C(  -591.35), EASYSIMD_FLOAT32_C(  -783.63) },
      { EASYSIMD_FLOAT32_C(-195788.86), EASYSIMD_FLOAT32_C( 39514.66), EASYSIMD_FLOAT32_C(238764.84), EASYSIMD_FLOAT32_C(-329216.91) } },
    { { EASYSIMD_FLOAT32_C(   711.56), EASYSIMD_FLOAT32_C(   510.39), EASYSIMD_FLOAT32_C(   691.18), EASYSIMD_FLOAT32_C(     4.98) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT32_C(  -243.43), EASYSIMD_FLOAT32_C(   653.48), EASYSIMD_FLOAT32_C(   209.06), EASYSIMD_FLOAT32_C(   223.54) },
      { EASYSIMD_FLOAT32_C(  -650.40), EASYSIMD_FLOAT32_C(  -553.08), EASYSIMD_FLOAT32_C(   468.67), EASYSIMD_FLOAT32_C(   270.27) },
      { EASYSIMD_FLOAT32_C(173865.45), EASYSIMD_FLOAT32_C(  -553.08), EASYSIMD_FLOAT32_C(   468.67), EASYSIMD_FLOAT32_C(   270.27) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask3_fnmsub_ps(a, b, c, test_vec[i].k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask3_fnmsub_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 c = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask3_fnmsub_ps(a, b, c, k);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fnmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd__mmask8 k;
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -776.82), EASYSIMD_FLOAT32_C(  -766.95), EASYSIMD_FLOAT32_C(   717.86), EASYSIMD_FLOAT32_C(   825.09) },
      UINT8_C(  2),
      { EASYSIMD_FLOAT32_C(   675.88), EASYSIMD_FLOAT32_C(   746.51), EASYSIMD_FLOAT32_C(   941.63), EASYSIMD_FLOAT32_C(   874.32) },
      { EASYSIMD_FLOAT32_C(    94.07), EASYSIMD_FLOAT32_C(  -949.00), EASYSIMD_FLOAT32_C(  -180.24), EASYSIMD_FLOAT32_C(  -190.97) },
      { EASYSIMD_FLOAT32_C(  -776.82), EASYSIMD_FLOAT32_C(573484.88), EASYSIMD_FLOAT32_C(   717.86), EASYSIMD_FLOAT32_C(   825.09) } },
    { { EASYSIMD_FLOAT32_C(   540.11), EASYSIMD_FLOAT32_C(  -566.52), EASYSIMD_FLOAT32_C(  -569.06), EASYSIMD_FLOAT32_C(   253.73) },
      UINT8_C(212),
      { EASYSIMD_FLOAT32_C(  -992.97), EASYSIMD_FLOAT32_C(  -696.94), EASYSIMD_FLOAT32_C(   784.15), EASYSIMD_FLOAT32_C(  -744.60) },
      { EASYSIMD_FLOAT32_C(  -906.40), EASYSIMD_FLOAT32_C(  -583.15), EASYSIMD_FLOAT32_C(  -789.78), EASYSIMD_FLOAT32_C(   664.80) },
      { EASYSIMD_FLOAT32_C(   540.11), EASYSIMD_FLOAT32_C(  -566.52), EASYSIMD_FLOAT32_C(447018.19), EASYSIMD_FLOAT32_C(   253.73) } },
    { { EASYSIMD_FLOAT32_C(     2.75), EASYSIMD_FLOAT32_C(   861.25), EASYSIMD_FLOAT32_C(  -519.43), EASYSIMD_FLOAT32_C(   544.64) },
      UINT8_C(137),
      { EASYSIMD_FLOAT32_C(  -296.26), EASYSIMD_FLOAT32_C(   777.70), EASYSIMD_FLOAT32_C(   944.73), EASYSIMD_FLOAT32_C(  -471.17) },
      { EASYSIMD_FLOAT32_C(  -978.67), EASYSIMD_FLOAT32_C(   620.62), EASYSIMD_FLOAT32_C(  -724.66), EASYSIMD_FLOAT32_C(   962.96) },
      { EASYSIMD_FLOAT32_C(  1793.39), EASYSIMD_FLOAT32_C(   861.25), EASYSIMD_FLOAT32_C(  -519.43), EASYSIMD_FLOAT32_C(255655.09) } },
    { { EASYSIMD_FLOAT32_C(   494.93), EASYSIMD_FLOAT32_C(   369.42), EASYSIMD_FLOAT32_C(  -986.04), EASYSIMD_FLOAT32_C(  -685.31) },
      UINT8_C(107),
      { EASYSIMD_FLOAT32_C(   554.07), EASYSIMD_FLOAT32_C(  -251.83), EASYSIMD_FLOAT32_C(  -390.61), EASYSIMD_FLOAT32_C(  -192.20) },
      { EASYSIMD_FLOAT32_C(   987.12), EASYSIMD_FLOAT32_C(  -383.58), EASYSIMD_FLOAT32_C(   110.86), EASYSIMD_FLOAT32_C(   771.27) },
      { EASYSIMD_FLOAT32_C(-275213.00), EASYSIMD_FLOAT32_C( 93414.62), EASYSIMD_FLOAT32_C(  -986.04), EASYSIMD_FLOAT32_C(-132487.84) } },
    { { EASYSIMD_FLOAT32_C(  -128.18), EASYSIMD_FLOAT32_C(   204.46), EASYSIMD_FLOAT32_C(  -811.88), EASYSIMD_FLOAT32_C(    82.05) },
      UINT8_C( 25),
      { EASYSIMD_FLOAT32_C(   190.86), EASYSIMD_FLOAT32_C(   -56.70), EASYSIMD_FLOAT32_C(   349.83), EASYSIMD_FLOAT32_C(  -264.49) },
      { EASYSIMD_FLOAT32_C(   170.17), EASYSIMD_FLOAT32_C(  -946.43), EASYSIMD_FLOAT32_C(  -486.80), EASYSIMD_FLOAT32_C(   114.90) },
      { EASYSIMD_FLOAT32_C( 24294.26), EASYSIMD_FLOAT32_C(   204.46), EASYSIMD_FLOAT32_C(  -811.88), EASYSIMD_FLOAT32_C( 21586.50) } },
    { { EASYSIMD_FLOAT32_C(  -417.60), EASYSIMD_FLOAT32_C(  -465.46), EASYSIMD_FLOAT32_C(  -264.48), EASYSIMD_FLOAT32_C(  -142.25) },
      UINT8_C( 55),
      { EASYSIMD_FLOAT32_C(  -769.55), EASYSIMD_FLOAT32_C(  -772.84), EASYSIMD_FLOAT32_C(  -488.55), EASYSIMD_FLOAT32_C(  -454.86) },
      { EASYSIMD_FLOAT32_C(  -594.39), EASYSIMD_FLOAT32_C(  -934.48), EASYSIMD_FLOAT32_C(   293.31), EASYSIMD_FLOAT32_C(    15.00) },
      { EASYSIMD_FLOAT32_C(-320769.72), EASYSIMD_FLOAT32_C(-358791.66), EASYSIMD_FLOAT32_C(-129505.02), EASYSIMD_FLOAT32_C(  -142.25) } },
    { { EASYSIMD_FLOAT32_C(  -126.69), EASYSIMD_FLOAT32_C(   280.43), EASYSIMD_FLOAT32_C(   631.43), EASYSIMD_FLOAT32_C(   984.17) },
      UINT8_C(140),
      { EASYSIMD_FLOAT32_C(  -496.75), EASYSIMD_FLOAT32_C(   188.63), EASYSIMD_FLOAT32_C(   239.82), EASYSIMD_FLOAT32_C(   585.30) },
      { EASYSIMD_FLOAT32_C(  -942.10), EASYSIMD_FLOAT32_C(  -569.32), EASYSIMD_FLOAT32_C(  -471.40), EASYSIMD_FLOAT32_C(   407.72) },
      { EASYSIMD_FLOAT32_C(  -126.69), EASYSIMD_FLOAT32_C(   280.43), EASYSIMD_FLOAT32_C(-150958.14), EASYSIMD_FLOAT32_C(-576442.44) } },
    { { EASYSIMD_FLOAT32_C(   166.19), EASYSIMD_FLOAT32_C(   698.76), EASYSIMD_FLOAT32_C(   461.30), EASYSIMD_FLOAT32_C(   679.39) },
      UINT8_C(107),
      { EASYSIMD_FLOAT32_C(  -956.30), EASYSIMD_FLOAT32_C(  -786.07), EASYSIMD_FLOAT32_C(   549.18), EASYSIMD_FLOAT32_C(   -98.55) },
      { EASYSIMD_FLOAT32_C(  -288.58), EASYSIMD_FLOAT32_C(   779.63), EASYSIMD_FLOAT32_C(   128.61), EASYSIMD_FLOAT32_C(   222.87) },
      { EASYSIMD_FLOAT32_C(159216.08), EASYSIMD_FLOAT32_C(548494.69), EASYSIMD_FLOAT32_C(   461.30), EASYSIMD_FLOAT32_C( 66731.02) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_fnmsub_ps(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_fnmsub_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 c = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_fnmsub_ps(a, k, b, c);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fnmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd__mmask8 k;
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    45.22), EASYSIMD_FLOAT32_C(  -841.74), EASYSIMD_FLOAT32_C(  -771.30), EASYSIMD_FLOAT32_C(   915.34) },
      UINT8_C(201),
      { EASYSIMD_FLOAT32_C(  -553.12), EASYSIMD_FLOAT32_C(   645.04), EASYSIMD_FLOAT32_C(   597.06), EASYSIMD_FLOAT32_C(  -923.84) },
      { EASYSIMD_FLOAT32_C(  -317.95), EASYSIMD_FLOAT32_C(  -305.13), EASYSIMD_FLOAT32_C(  -743.20), EASYSIMD_FLOAT32_C(  -533.81) },
      { EASYSIMD_FLOAT32_C( 25330.04), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(846161.56) } },
    { { EASYSIMD_FLOAT32_C(  -622.50), EASYSIMD_FLOAT32_C(   186.96), EASYSIMD_FLOAT32_C(   248.91), EASYSIMD_FLOAT32_C(   889.54) },
      UINT8_C(133),
      { EASYSIMD_FLOAT32_C(   409.18), EASYSIMD_FLOAT32_C(  -381.54), EASYSIMD_FLOAT32_C(  -959.89), EASYSIMD_FLOAT32_C(   248.73) },
      { EASYSIMD_FLOAT32_C(  -528.83), EASYSIMD_FLOAT32_C(   -48.66), EASYSIMD_FLOAT32_C(  -812.96), EASYSIMD_FLOAT32_C(    86.16) },
      { EASYSIMD_FLOAT32_C(255243.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(239739.19), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   697.16), EASYSIMD_FLOAT32_C(   615.11), EASYSIMD_FLOAT32_C(    84.31), EASYSIMD_FLOAT32_C(   309.97) },
      UINT8_C(200),
      { EASYSIMD_FLOAT32_C(  -870.47), EASYSIMD_FLOAT32_C(   468.23), EASYSIMD_FLOAT32_C(  -230.06), EASYSIMD_FLOAT32_C(  -955.13) },
      { EASYSIMD_FLOAT32_C(  -397.88), EASYSIMD_FLOAT32_C(   216.81), EASYSIMD_FLOAT32_C(   689.91), EASYSIMD_FLOAT32_C(  -800.82) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(296862.47) } },
    { { EASYSIMD_FLOAT32_C(   292.98), EASYSIMD_FLOAT32_C(  -628.04), EASYSIMD_FLOAT32_C(  -105.95), EASYSIMD_FLOAT32_C(   549.78) },
      UINT8_C(120),
      { EASYSIMD_FLOAT32_C(   271.55), EASYSIMD_FLOAT32_C(  -263.26), EASYSIMD_FLOAT32_C(  -912.94), EASYSIMD_FLOAT32_C(   161.09) },
      { EASYSIMD_FLOAT32_C(   400.46), EASYSIMD_FLOAT32_C(   496.24), EASYSIMD_FLOAT32_C(   779.55), EASYSIMD_FLOAT32_C(   440.57) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-89004.63) } },
    { { EASYSIMD_FLOAT32_C(  -255.03), EASYSIMD_FLOAT32_C(  -749.28), EASYSIMD_FLOAT32_C(  -608.09), EASYSIMD_FLOAT32_C(   -67.99) },
      UINT8_C( 37),
      { EASYSIMD_FLOAT32_C(  -910.93), EASYSIMD_FLOAT32_C(  -452.88), EASYSIMD_FLOAT32_C(  -578.80), EASYSIMD_FLOAT32_C(   399.04) },
      { EASYSIMD_FLOAT32_C(    88.35), EASYSIMD_FLOAT32_C(  -449.28), EASYSIMD_FLOAT32_C(  -132.73), EASYSIMD_FLOAT32_C(   858.29) },
      { EASYSIMD_FLOAT32_C(-232402.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-351829.78), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -404.41), EASYSIMD_FLOAT32_C(   469.39), EASYSIMD_FLOAT32_C(    75.10), EASYSIMD_FLOAT32_C(  -714.50) },
      UINT8_C(215),
      { EASYSIMD_FLOAT32_C(  -631.92), EASYSIMD_FLOAT32_C(  -342.54), EASYSIMD_FLOAT32_C(  -437.38), EASYSIMD_FLOAT32_C(   917.85) },
      { EASYSIMD_FLOAT32_C(   495.61), EASYSIMD_FLOAT32_C(   834.17), EASYSIMD_FLOAT32_C(  -345.41), EASYSIMD_FLOAT32_C(   582.67) },
      { EASYSIMD_FLOAT32_C(-256050.38), EASYSIMD_FLOAT32_C(159950.69), EASYSIMD_FLOAT32_C( 33192.65), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(    -4.74), EASYSIMD_FLOAT32_C(  -944.95), EASYSIMD_FLOAT32_C(    78.91), EASYSIMD_FLOAT32_C(  -225.19) },
      UINT8_C( 21),
      { EASYSIMD_FLOAT32_C(   823.88), EASYSIMD_FLOAT32_C(    25.53), EASYSIMD_FLOAT32_C(   887.53), EASYSIMD_FLOAT32_C(  -244.11) },
      { EASYSIMD_FLOAT32_C(  -637.58), EASYSIMD_FLOAT32_C(   976.60), EASYSIMD_FLOAT32_C(   303.00), EASYSIMD_FLOAT32_C(  -216.39) },
      { EASYSIMD_FLOAT32_C(  4542.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-70338.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   375.64), EASYSIMD_FLOAT32_C(  -608.65), EASYSIMD_FLOAT32_C(   334.33), EASYSIMD_FLOAT32_C(  -757.09) },
      UINT8_C(245),
      { EASYSIMD_FLOAT32_C(   929.92), EASYSIMD_FLOAT32_C(   712.30), EASYSIMD_FLOAT32_C(   324.74), EASYSIMD_FLOAT32_C(  -784.58) },
      { EASYSIMD_FLOAT32_C(   380.87), EASYSIMD_FLOAT32_C(   692.82), EASYSIMD_FLOAT32_C(  -127.12), EASYSIMD_FLOAT32_C(   943.49) },
      { EASYSIMD_FLOAT32_C(-349696.03), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-108443.20), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_fnmsub_ps(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_fnmsub_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 c = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_fnmsub_ps(k, a, b, c);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask3_fnmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd__mmask8 k;
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -59.95), EASYSIMD_FLOAT64_C(   356.35) },
      UINT8_C(133),
      { EASYSIMD_FLOAT64_C(   -31.14), EASYSIMD_FLOAT64_C(   642.39) },
      { EASYSIMD_FLOAT64_C(   297.20), EASYSIMD_FLOAT64_C(  -187.94) },
      { EASYSIMD_FLOAT64_C( -2164.04), EASYSIMD_FLOAT64_C(  -187.94) } },
    { { EASYSIMD_FLOAT64_C(  -197.29), EASYSIMD_FLOAT64_C(   110.25) },
      UINT8_C( 50),
      { EASYSIMD_FLOAT64_C(   145.41), EASYSIMD_FLOAT64_C(  -244.11) },
      { EASYSIMD_FLOAT64_C(  -559.15), EASYSIMD_FLOAT64_C(  -171.87) },
      { EASYSIMD_FLOAT64_C(  -559.15), EASYSIMD_FLOAT64_C( 27085.00) } },
    { { EASYSIMD_FLOAT64_C(   660.21), EASYSIMD_FLOAT64_C(  -546.75) },
      UINT8_C(247),
      { EASYSIMD_FLOAT64_C(   659.65), EASYSIMD_FLOAT64_C(  -521.83) },
      { EASYSIMD_FLOAT64_C(   170.54), EASYSIMD_FLOAT64_C(  -287.82) },
      { EASYSIMD_FLOAT64_C(-435678.07), EASYSIMD_FLOAT64_C(-285022.73) } },
    { { EASYSIMD_FLOAT64_C(  -392.53), EASYSIMD_FLOAT64_C(  -375.78) },
      UINT8_C(180),
      { EASYSIMD_FLOAT64_C(   909.00), EASYSIMD_FLOAT64_C(   659.60) },
      { EASYSIMD_FLOAT64_C(  -106.24), EASYSIMD_FLOAT64_C(  -236.52) },
      { EASYSIMD_FLOAT64_C(  -106.24), EASYSIMD_FLOAT64_C(  -236.52) } },
    { { EASYSIMD_FLOAT64_C(  -554.79), EASYSIMD_FLOAT64_C(   -19.24) },
      UINT8_C( 89),
      { EASYSIMD_FLOAT64_C(   385.26), EASYSIMD_FLOAT64_C(  -662.89) },
      { EASYSIMD_FLOAT64_C(  -921.82), EASYSIMD_FLOAT64_C(  -645.89) },
      { EASYSIMD_FLOAT64_C(214660.22), EASYSIMD_FLOAT64_C(  -645.89) } },
    { { EASYSIMD_FLOAT64_C(   979.51), EASYSIMD_FLOAT64_C(   375.38) },
      UINT8_C(147),
      { EASYSIMD_FLOAT64_C(  -217.78), EASYSIMD_FLOAT64_C(  -514.37) },
      { EASYSIMD_FLOAT64_C(     7.56), EASYSIMD_FLOAT64_C(   927.63) },
      { EASYSIMD_FLOAT64_C(213310.13), EASYSIMD_FLOAT64_C(192156.58) } },
    { { EASYSIMD_FLOAT64_C(   241.52), EASYSIMD_FLOAT64_C(   448.41) },
      UINT8_C( 90),
      { EASYSIMD_FLOAT64_C(   -98.27), EASYSIMD_FLOAT64_C(   901.66) },
      { EASYSIMD_FLOAT64_C(  -636.60), EASYSIMD_FLOAT64_C(  -438.62) },
      { EASYSIMD_FLOAT64_C(  -636.60), EASYSIMD_FLOAT64_C(-403874.74) } },
    { { EASYSIMD_FLOAT64_C(  -620.17), EASYSIMD_FLOAT64_C(   533.93) },
      UINT8_C(101),
      { EASYSIMD_FLOAT64_C(   -12.71), EASYSIMD_FLOAT64_C(  -841.85) },
      { EASYSIMD_FLOAT64_C(  -382.14), EASYSIMD_FLOAT64_C(  -103.70) },
      { EASYSIMD_FLOAT64_C( -7500.22), EASYSIMD_FLOAT64_C(  -103.70) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask3_fnmsub_pd(a, b, c, test_vec[i].k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask3_fnmsub_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d c = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask3_fnmsub_pd(a, b, c, k);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fnmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd__mmask8 k;
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -651.10), EASYSIMD_FLOAT64_C(  -451.98) },
      UINT8_C(159),
      { EASYSIMD_FLOAT64_C(    30.55), EASYSIMD_FLOAT64_C(  -922.49) },
      { EASYSIMD_FLOAT64_C(   470.14), EASYSIMD_FLOAT64_C(  -546.94) },
      { EASYSIMD_FLOAT64_C( 19420.97), EASYSIMD_FLOAT64_C(-416400.09) } },
    { { EASYSIMD_FLOAT64_C(   527.39), EASYSIMD_FLOAT64_C(   699.95) },
      UINT8_C( 20),
      { EASYSIMD_FLOAT64_C(  -817.63), EASYSIMD_FLOAT64_C(  -542.87) },
      { EASYSIMD_FLOAT64_C(  -510.27), EASYSIMD_FLOAT64_C(   -49.36) },
      { EASYSIMD_FLOAT64_C(   527.39), EASYSIMD_FLOAT64_C(   699.95) } },
    { { EASYSIMD_FLOAT64_C(   887.98), EASYSIMD_FLOAT64_C(  -827.66) },
      UINT8_C(  8),
      { EASYSIMD_FLOAT64_C(   628.68), EASYSIMD_FLOAT64_C(   750.25) },
      { EASYSIMD_FLOAT64_C(  -133.93), EASYSIMD_FLOAT64_C(  -410.52) },
      { EASYSIMD_FLOAT64_C(   887.98), EASYSIMD_FLOAT64_C(  -827.66) } },
    { { EASYSIMD_FLOAT64_C(  -354.72), EASYSIMD_FLOAT64_C(   666.08) },
      UINT8_C(254),
      { EASYSIMD_FLOAT64_C(  -258.88), EASYSIMD_FLOAT64_C(   917.38) },
      { EASYSIMD_FLOAT64_C(   678.74), EASYSIMD_FLOAT64_C(  -441.38) },
      { EASYSIMD_FLOAT64_C(  -354.72), EASYSIMD_FLOAT64_C(-610607.09) } },
    { { EASYSIMD_FLOAT64_C(  -254.17), EASYSIMD_FLOAT64_C(  -402.57) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT64_C(    94.73), EASYSIMD_FLOAT64_C(   145.45) },
      { EASYSIMD_FLOAT64_C(   327.21), EASYSIMD_FLOAT64_C(  -874.72) },
      { EASYSIMD_FLOAT64_C( 23750.31), EASYSIMD_FLOAT64_C(  -402.57) } },
    { { EASYSIMD_FLOAT64_C(   222.96), EASYSIMD_FLOAT64_C(  -202.64) },
      UINT8_C( 72),
      { EASYSIMD_FLOAT64_C(  -249.64), EASYSIMD_FLOAT64_C(  -502.70) },
      { EASYSIMD_FLOAT64_C(   275.86), EASYSIMD_FLOAT64_C(   -67.27) },
      { EASYSIMD_FLOAT64_C(   222.96), EASYSIMD_FLOAT64_C(  -202.64) } },
    { { EASYSIMD_FLOAT64_C(   -45.57), EASYSIMD_FLOAT64_C(   765.60) },
      UINT8_C( 99),
      { EASYSIMD_FLOAT64_C(  -157.59), EASYSIMD_FLOAT64_C(   937.93) },
      { EASYSIMD_FLOAT64_C(   923.38), EASYSIMD_FLOAT64_C(  -528.91) },
      { EASYSIMD_FLOAT64_C( -8104.76), EASYSIMD_FLOAT64_C(-717550.30) } },
    { { EASYSIMD_FLOAT64_C(   688.18), EASYSIMD_FLOAT64_C(  -210.55) },
      UINT8_C( 11),
      { EASYSIMD_FLOAT64_C(  -666.54), EASYSIMD_FLOAT64_C(  -544.47) },
      { EASYSIMD_FLOAT64_C(   961.97), EASYSIMD_FLOAT64_C(    74.58) },
      { EASYSIMD_FLOAT64_C(457737.53), EASYSIMD_FLOAT64_C(-114712.74) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_fnmsub_pd(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_fnmsub_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d c = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_fnmsub_pd(a, k, b, c);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fnmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd__mmask8 k;
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   820.79), EASYSIMD_FLOAT64_C(  -727.37) },
      UINT8_C(114),
      { EASYSIMD_FLOAT64_C(    13.36), EASYSIMD_FLOAT64_C(   876.37) },
      { EASYSIMD_FLOAT64_C(   502.06), EASYSIMD_FLOAT64_C(  -690.95) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(638136.20) } },
    { { EASYSIMD_FLOAT64_C(  -709.79), EASYSIMD_FLOAT64_C(   836.99) },
      UINT8_C(201),
      { EASYSIMD_FLOAT64_C(  -446.38), EASYSIMD_FLOAT64_C(  -251.78) },
      { EASYSIMD_FLOAT64_C(   369.13), EASYSIMD_FLOAT64_C(   476.61) },
      { EASYSIMD_FLOAT64_C(-317205.19), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -142.82), EASYSIMD_FLOAT64_C(   989.38) },
      UINT8_C(100),
      { EASYSIMD_FLOAT64_C(  -358.50), EASYSIMD_FLOAT64_C(   570.18) },
      { EASYSIMD_FLOAT64_C(  -800.06), EASYSIMD_FLOAT64_C(  -857.20) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(    59.69), EASYSIMD_FLOAT64_C(   435.80) },
      UINT8_C(219),
      { EASYSIMD_FLOAT64_C(  -538.79), EASYSIMD_FLOAT64_C(   926.39) },
      { EASYSIMD_FLOAT64_C(  -668.90), EASYSIMD_FLOAT64_C(   959.37) },
      { EASYSIMD_FLOAT64_C( 32829.28), EASYSIMD_FLOAT64_C(-404680.13) } },
    { { EASYSIMD_FLOAT64_C(  -494.38), EASYSIMD_FLOAT64_C(    62.71) },
      UINT8_C(119),
      { EASYSIMD_FLOAT64_C(  -673.59), EASYSIMD_FLOAT64_C(   335.33) },
      { EASYSIMD_FLOAT64_C(  -242.90), EASYSIMD_FLOAT64_C(   339.78) },
      { EASYSIMD_FLOAT64_C(-332766.52), EASYSIMD_FLOAT64_C(-21368.32) } },
    { { EASYSIMD_FLOAT64_C(   211.70), EASYSIMD_FLOAT64_C(  -740.84) },
      UINT8_C( 46),
      { EASYSIMD_FLOAT64_C(   501.92), EASYSIMD_FLOAT64_C(  -903.85) },
      { EASYSIMD_FLOAT64_C(   -53.87), EASYSIMD_FLOAT64_C(  -944.47) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-668663.76) } },
    { { EASYSIMD_FLOAT64_C(  -155.63), EASYSIMD_FLOAT64_C(  -684.74) },
      UINT8_C( 88),
      { EASYSIMD_FLOAT64_C(   701.56), EASYSIMD_FLOAT64_C(  -695.37) },
      { EASYSIMD_FLOAT64_C(  -554.98), EASYSIMD_FLOAT64_C(  -656.94) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   874.81), EASYSIMD_FLOAT64_C(  -355.04) },
      UINT8_C(205),
      { EASYSIMD_FLOAT64_C(   -65.50), EASYSIMD_FLOAT64_C(  -919.24) },
      { EASYSIMD_FLOAT64_C(   595.91), EASYSIMD_FLOAT64_C(   395.71) },
      { EASYSIMD_FLOAT64_C( 56704.14), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_fnmsub_pd(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_fnmsub_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d c = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_fnmsub_pd(k, a, b, c);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_fnmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const easysimd__mmask8 k;
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -507.43), EASYSIMD_FLOAT32_C(   457.08), EASYSIMD_FLOAT32_C(   961.95), EASYSIMD_FLOAT32_C(    92.66),
        EASYSIMD_FLOAT32_C(    85.47), EASYSIMD_FLOAT32_C(  -625.37), EASYSIMD_FLOAT32_C(   827.39), EASYSIMD_FLOAT32_C(   917.44) },
      UINT8_C( 92),
      { EASYSIMD_FLOAT32_C(  -373.97), EASYSIMD_FLOAT32_C(  -930.16), EASYSIMD_FLOAT32_C(  -903.75), EASYSIMD_FLOAT32_C(  -173.81),
        EASYSIMD_FLOAT32_C(   -98.47), EASYSIMD_FLOAT32_C(   945.67), EASYSIMD_FLOAT32_C(   229.85), EASYSIMD_FLOAT32_C(   792.87) },
      { EASYSIMD_FLOAT32_C(   -71.16), EASYSIMD_FLOAT32_C(   799.89), EASYSIMD_FLOAT32_C(   -59.82), EASYSIMD_FLOAT32_C(  -575.65),
        EASYSIMD_FLOAT32_C(   256.82), EASYSIMD_FLOAT32_C(  -650.94), EASYSIMD_FLOAT32_C(   511.94), EASYSIMD_FLOAT32_C(  -236.83) },
      { EASYSIMD_FLOAT32_C(  -507.43), EASYSIMD_FLOAT32_C(   457.08), EASYSIMD_FLOAT32_C(869422.12), EASYSIMD_FLOAT32_C( 16680.88),
        EASYSIMD_FLOAT32_C(  8159.41), EASYSIMD_FLOAT32_C(  -625.37), EASYSIMD_FLOAT32_C(-190687.53), EASYSIMD_FLOAT32_C(   917.44) } },
    { { EASYSIMD_FLOAT32_C(  -652.78), EASYSIMD_FLOAT32_C(    83.33), EASYSIMD_FLOAT32_C(  -963.49), EASYSIMD_FLOAT32_C(  -731.54),
        EASYSIMD_FLOAT32_C(   586.95), EASYSIMD_FLOAT32_C(  -217.19), EASYSIMD_FLOAT32_C(  -238.98), EASYSIMD_FLOAT32_C(    44.03) },
      UINT8_C(144),
      { EASYSIMD_FLOAT32_C(   853.68), EASYSIMD_FLOAT32_C(  -870.50), EASYSIMD_FLOAT32_C(   119.39), EASYSIMD_FLOAT32_C(   681.07),
        EASYSIMD_FLOAT32_C(  -953.06), EASYSIMD_FLOAT32_C(  -494.33), EASYSIMD_FLOAT32_C(  -692.90), EASYSIMD_FLOAT32_C(  -883.22) },
      { EASYSIMD_FLOAT32_C(  -398.09), EASYSIMD_FLOAT32_C(   133.29), EASYSIMD_FLOAT32_C(    18.31), EASYSIMD_FLOAT32_C(  -452.42),
        EASYSIMD_FLOAT32_C(  -636.86), EASYSIMD_FLOAT32_C(  -188.82), EASYSIMD_FLOAT32_C(   476.43), EASYSIMD_FLOAT32_C(  -836.97) },
      { EASYSIMD_FLOAT32_C(  -652.78), EASYSIMD_FLOAT32_C(    83.33), EASYSIMD_FLOAT32_C(  -963.49), EASYSIMD_FLOAT32_C(  -731.54),
        EASYSIMD_FLOAT32_C(560035.44), EASYSIMD_FLOAT32_C(  -217.19), EASYSIMD_FLOAT32_C(  -238.98), EASYSIMD_FLOAT32_C( 39725.14) } },
    { { EASYSIMD_FLOAT32_C(   751.36), EASYSIMD_FLOAT32_C(   900.78), EASYSIMD_FLOAT32_C(   419.85), EASYSIMD_FLOAT32_C(  -899.58),
        EASYSIMD_FLOAT32_C(   412.72), EASYSIMD_FLOAT32_C(  -816.98), EASYSIMD_FLOAT32_C(  -552.36), EASYSIMD_FLOAT32_C(  -503.95) },
      UINT8_C(124),
      { EASYSIMD_FLOAT32_C(  -283.90), EASYSIMD_FLOAT32_C(  -917.00), EASYSIMD_FLOAT32_C(     2.34), EASYSIMD_FLOAT32_C(   477.12),
        EASYSIMD_FLOAT32_C(   127.03), EASYSIMD_FLOAT32_C(   747.09), EASYSIMD_FLOAT32_C(   330.80), EASYSIMD_FLOAT32_C(   256.53) },
      { EASYSIMD_FLOAT32_C(  -133.52), EASYSIMD_FLOAT32_C(    11.88), EASYSIMD_FLOAT32_C(   303.46), EASYSIMD_FLOAT32_C(   372.15),
        EASYSIMD_FLOAT32_C(   318.98), EASYSIMD_FLOAT32_C(   420.24), EASYSIMD_FLOAT32_C(   974.07), EASYSIMD_FLOAT32_C(  -547.73) },
      { EASYSIMD_FLOAT32_C(   751.36), EASYSIMD_FLOAT32_C(   900.78), EASYSIMD_FLOAT32_C( -1285.91), EASYSIMD_FLOAT32_C(428835.47),
        EASYSIMD_FLOAT32_C(-52746.80), EASYSIMD_FLOAT32_C(609937.38), EASYSIMD_FLOAT32_C(181746.61), EASYSIMD_FLOAT32_C(  -503.95) } },
    { { EASYSIMD_FLOAT32_C(  -561.45), EASYSIMD_FLOAT32_C(  -478.35), EASYSIMD_FLOAT32_C(  -184.60), EASYSIMD_FLOAT32_C(   249.73),
        EASYSIMD_FLOAT32_C(   998.07), EASYSIMD_FLOAT32_C(   -21.57), EASYSIMD_FLOAT32_C(     1.08), EASYSIMD_FLOAT32_C(   898.85) },
      UINT8_C( 58),
      { EASYSIMD_FLOAT32_C(   101.50), EASYSIMD_FLOAT32_C(   311.57), EASYSIMD_FLOAT32_C(  -418.70), EASYSIMD_FLOAT32_C(   549.14),
        EASYSIMD_FLOAT32_C(   807.62), EASYSIMD_FLOAT32_C(  -199.17), EASYSIMD_FLOAT32_C(  -734.77), EASYSIMD_FLOAT32_C(   890.63) },
      { EASYSIMD_FLOAT32_C(   803.16), EASYSIMD_FLOAT32_C(   742.35), EASYSIMD_FLOAT32_C(    17.66), EASYSIMD_FLOAT32_C(   550.26),
        EASYSIMD_FLOAT32_C(    73.15), EASYSIMD_FLOAT32_C(  -725.82), EASYSIMD_FLOAT32_C(  -583.26), EASYSIMD_FLOAT32_C(  -914.97) },
      { EASYSIMD_FLOAT32_C(  -561.45), EASYSIMD_FLOAT32_C(148297.17), EASYSIMD_FLOAT32_C(  -184.60), EASYSIMD_FLOAT32_C(-137687.00),
        EASYSIMD_FLOAT32_C(-806134.44), EASYSIMD_FLOAT32_C( -3570.28), EASYSIMD_FLOAT32_C(     1.08), EASYSIMD_FLOAT32_C(   898.85) } },
    { { EASYSIMD_FLOAT32_C(   577.65), EASYSIMD_FLOAT32_C(   788.89), EASYSIMD_FLOAT32_C(   404.01), EASYSIMD_FLOAT32_C(    -2.11),
        EASYSIMD_FLOAT32_C(   762.96), EASYSIMD_FLOAT32_C(   856.27), EASYSIMD_FLOAT32_C(   436.44), EASYSIMD_FLOAT32_C(  -715.39) },
      UINT8_C( 86),
      { EASYSIMD_FLOAT32_C(  -313.84), EASYSIMD_FLOAT32_C(  -717.32), EASYSIMD_FLOAT32_C(   650.11), EASYSIMD_FLOAT32_C(   687.25),
        EASYSIMD_FLOAT32_C(  -818.47), EASYSIMD_FLOAT32_C(  -951.61), EASYSIMD_FLOAT32_C(  -211.26), EASYSIMD_FLOAT32_C(   493.10) },
      { EASYSIMD_FLOAT32_C(  -370.31), EASYSIMD_FLOAT32_C(  -662.12), EASYSIMD_FLOAT32_C(   300.72), EASYSIMD_FLOAT32_C(   430.52),
        EASYSIMD_FLOAT32_C(  -396.89), EASYSIMD_FLOAT32_C(   191.35), EASYSIMD_FLOAT32_C(   233.68), EASYSIMD_FLOAT32_C(  -654.53) },
      { EASYSIMD_FLOAT32_C(   577.65), EASYSIMD_FLOAT32_C(566548.69), EASYSIMD_FLOAT32_C(-262951.66), EASYSIMD_FLOAT32_C(    -2.11),
        EASYSIMD_FLOAT32_C(624856.75), EASYSIMD_FLOAT32_C(   856.27), EASYSIMD_FLOAT32_C( 91968.63), EASYSIMD_FLOAT32_C(  -715.39) } },
    { { EASYSIMD_FLOAT32_C(  -790.99), EASYSIMD_FLOAT32_C(  -216.06), EASYSIMD_FLOAT32_C(   418.62), EASYSIMD_FLOAT32_C(  -516.81),
        EASYSIMD_FLOAT32_C(   200.68), EASYSIMD_FLOAT32_C(   503.65), EASYSIMD_FLOAT32_C(  -939.17), EASYSIMD_FLOAT32_C(   -10.43) },
      UINT8_C( 59),
      { EASYSIMD_FLOAT32_C(    58.72), EASYSIMD_FLOAT32_C(  -247.47), EASYSIMD_FLOAT32_C(  -236.07), EASYSIMD_FLOAT32_C(  -504.84),
        EASYSIMD_FLOAT32_C(    37.13), EASYSIMD_FLOAT32_C(   435.60), EASYSIMD_FLOAT32_C(   181.32), EASYSIMD_FLOAT32_C(   319.81) },
      { EASYSIMD_FLOAT32_C(    85.71), EASYSIMD_FLOAT32_C(  -131.43), EASYSIMD_FLOAT32_C(   501.34), EASYSIMD_FLOAT32_C(   134.10),
        EASYSIMD_FLOAT32_C(   657.31), EASYSIMD_FLOAT32_C(    -5.56), EASYSIMD_FLOAT32_C(   763.79), EASYSIMD_FLOAT32_C(   995.19) },
      { EASYSIMD_FLOAT32_C( 46361.22), EASYSIMD_FLOAT32_C(-53336.94), EASYSIMD_FLOAT32_C(   418.62), EASYSIMD_FLOAT32_C(-261040.45),
        EASYSIMD_FLOAT32_C( -8108.56), EASYSIMD_FLOAT32_C(-219384.38), EASYSIMD_FLOAT32_C(  -939.17), EASYSIMD_FLOAT32_C(   -10.43) } },
    { { EASYSIMD_FLOAT32_C(  -704.83), EASYSIMD_FLOAT32_C(   194.31), EASYSIMD_FLOAT32_C(  -401.69), EASYSIMD_FLOAT32_C(   486.52),
        EASYSIMD_FLOAT32_C(  -572.02), EASYSIMD_FLOAT32_C(   -56.22), EASYSIMD_FLOAT32_C(   695.52), EASYSIMD_FLOAT32_C(   211.92) },
      UINT8_C(201),
      { EASYSIMD_FLOAT32_C(  -821.29), EASYSIMD_FLOAT32_C(  -587.40), EASYSIMD_FLOAT32_C(   866.04), EASYSIMD_FLOAT32_C(  -760.45),
        EASYSIMD_FLOAT32_C(   402.17), EASYSIMD_FLOAT32_C(  -226.30), EASYSIMD_FLOAT32_C(   298.27), EASYSIMD_FLOAT32_C(  -845.30) },
      { EASYSIMD_FLOAT32_C(   537.63), EASYSIMD_FLOAT32_C(   793.43), EASYSIMD_FLOAT32_C(   191.83), EASYSIMD_FLOAT32_C(   -26.77),
        EASYSIMD_FLOAT32_C(   -25.25), EASYSIMD_FLOAT32_C(  -488.36), EASYSIMD_FLOAT32_C(  -941.06), EASYSIMD_FLOAT32_C(   843.32) },
      { EASYSIMD_FLOAT32_C(-579407.44), EASYSIMD_FLOAT32_C(   194.31), EASYSIMD_FLOAT32_C(  -401.69), EASYSIMD_FLOAT32_C(370000.91),
        EASYSIMD_FLOAT32_C(  -572.02), EASYSIMD_FLOAT32_C(   -56.22), EASYSIMD_FLOAT32_C(-206511.69), EASYSIMD_FLOAT32_C(178292.66) } },
    { { EASYSIMD_FLOAT32_C(  -987.02), EASYSIMD_FLOAT32_C(   193.05), EASYSIMD_FLOAT32_C(   500.63), EASYSIMD_FLOAT32_C(     7.43),
        EASYSIMD_FLOAT32_C(   -43.16), EASYSIMD_FLOAT32_C(   495.82), EASYSIMD_FLOAT32_C(   302.60), EASYSIMD_FLOAT32_C(  -848.86) },
      UINT8_C(109),
      { EASYSIMD_FLOAT32_C(  -210.88), EASYSIMD_FLOAT32_C(  -420.87), EASYSIMD_FLOAT32_C(    37.91), EASYSIMD_FLOAT32_C(  -515.36),
        EASYSIMD_FLOAT32_C(   791.05), EASYSIMD_FLOAT32_C(   400.30), EASYSIMD_FLOAT32_C(  -336.65), EASYSIMD_FLOAT32_C(  -796.36) },
      { EASYSIMD_FLOAT32_C(   266.35), EASYSIMD_FLOAT32_C(   -97.10), EASYSIMD_FLOAT32_C(   605.81), EASYSIMD_FLOAT32_C(  -959.95),
        EASYSIMD_FLOAT32_C(  -798.83), EASYSIMD_FLOAT32_C(   760.51), EASYSIMD_FLOAT32_C(   577.68), EASYSIMD_FLOAT32_C(   994.59) },
      { EASYSIMD_FLOAT32_C(-208409.12), EASYSIMD_FLOAT32_C(   193.05), EASYSIMD_FLOAT32_C(-19584.69), EASYSIMD_FLOAT32_C(  4789.07),
        EASYSIMD_FLOAT32_C(   -43.16), EASYSIMD_FLOAT32_C(-199237.27), EASYSIMD_FLOAT32_C(101292.61), EASYSIMD_FLOAT32_C(  -848.86) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_fnmsub_ps(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_fnmsub_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 c = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_fnmsub_ps(a, k, b, c);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_fnmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const easysimd__mmask8 k;
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -968.52), EASYSIMD_FLOAT32_C(   -53.91), EASYSIMD_FLOAT32_C(   901.74), EASYSIMD_FLOAT32_C(   749.29),
        EASYSIMD_FLOAT32_C(   200.42), EASYSIMD_FLOAT32_C(   136.89), EASYSIMD_FLOAT32_C(   788.37), EASYSIMD_FLOAT32_C(   138.42) },
      UINT8_C(240),
      { EASYSIMD_FLOAT32_C(   455.04), EASYSIMD_FLOAT32_C(  -328.20), EASYSIMD_FLOAT32_C(   567.89), EASYSIMD_FLOAT32_C(  -391.83),
        EASYSIMD_FLOAT32_C(  -650.38), EASYSIMD_FLOAT32_C(   178.44), EASYSIMD_FLOAT32_C(   899.97), EASYSIMD_FLOAT32_C(   625.65) },
      { EASYSIMD_FLOAT32_C(   371.14), EASYSIMD_FLOAT32_C(  -954.51), EASYSIMD_FLOAT32_C(   933.41), EASYSIMD_FLOAT32_C(   666.47),
        EASYSIMD_FLOAT32_C(   972.24), EASYSIMD_FLOAT32_C(   297.52), EASYSIMD_FLOAT32_C(   791.83), EASYSIMD_FLOAT32_C(   517.54) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(129376.91), EASYSIMD_FLOAT32_C(-24724.17), EASYSIMD_FLOAT32_C(-710301.12), EASYSIMD_FLOAT32_C(-87120.02) } },
    { { EASYSIMD_FLOAT32_C(  -901.84), EASYSIMD_FLOAT32_C(   458.34), EASYSIMD_FLOAT32_C(   837.59), EASYSIMD_FLOAT32_C(  -314.40),
        EASYSIMD_FLOAT32_C(  -586.87), EASYSIMD_FLOAT32_C(   559.62), EASYSIMD_FLOAT32_C(  -282.93), EASYSIMD_FLOAT32_C(   359.22) },
      UINT8_C(250),
      { EASYSIMD_FLOAT32_C(  -533.64), EASYSIMD_FLOAT32_C(  -440.36), EASYSIMD_FLOAT32_C(  -401.75), EASYSIMD_FLOAT32_C(  -745.27),
        EASYSIMD_FLOAT32_C(   698.06), EASYSIMD_FLOAT32_C(   166.72), EASYSIMD_FLOAT32_C(   709.77), EASYSIMD_FLOAT32_C(  -630.14) },
      { EASYSIMD_FLOAT32_C(  -265.39), EASYSIMD_FLOAT32_C(  -682.06), EASYSIMD_FLOAT32_C(  -280.52), EASYSIMD_FLOAT32_C(   913.05),
        EASYSIMD_FLOAT32_C(  -782.09), EASYSIMD_FLOAT32_C(  -654.87), EASYSIMD_FLOAT32_C(   284.18), EASYSIMD_FLOAT32_C(  -736.60) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(202516.66), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-235225.94),
        EASYSIMD_FLOAT32_C(410452.56), EASYSIMD_FLOAT32_C(-92644.98), EASYSIMD_FLOAT32_C(200531.03), EASYSIMD_FLOAT32_C(227095.48) } },
    { { EASYSIMD_FLOAT32_C(  -721.46), EASYSIMD_FLOAT32_C(   -49.35), EASYSIMD_FLOAT32_C(  -764.36), EASYSIMD_FLOAT32_C(   576.05),
        EASYSIMD_FLOAT32_C(  -257.52), EASYSIMD_FLOAT32_C(   753.18), EASYSIMD_FLOAT32_C(   674.21), EASYSIMD_FLOAT32_C(  -799.18) },
      UINT8_C( 26),
      { EASYSIMD_FLOAT32_C(  -640.19), EASYSIMD_FLOAT32_C(  -386.05), EASYSIMD_FLOAT32_C(   150.39), EASYSIMD_FLOAT32_C(    76.88),
        EASYSIMD_FLOAT32_C(   973.17), EASYSIMD_FLOAT32_C(  -388.26), EASYSIMD_FLOAT32_C(   543.25), EASYSIMD_FLOAT32_C(  -467.19) },
      { EASYSIMD_FLOAT32_C(   209.99), EASYSIMD_FLOAT32_C(   797.98), EASYSIMD_FLOAT32_C(  -769.14), EASYSIMD_FLOAT32_C(  -623.29),
        EASYSIMD_FLOAT32_C(   507.75), EASYSIMD_FLOAT32_C(  -399.28), EASYSIMD_FLOAT32_C(   111.32), EASYSIMD_FLOAT32_C(   825.69) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-19849.55), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-43663.43),
        EASYSIMD_FLOAT32_C(250102.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   320.20), EASYSIMD_FLOAT32_C(    24.37), EASYSIMD_FLOAT32_C(  -956.40), EASYSIMD_FLOAT32_C(   665.33),
        EASYSIMD_FLOAT32_C(  -691.45), EASYSIMD_FLOAT32_C(  -692.99), EASYSIMD_FLOAT32_C(   943.86), EASYSIMD_FLOAT32_C(   259.20) },
      UINT8_C(220),
      { EASYSIMD_FLOAT32_C(   519.92), EASYSIMD_FLOAT32_C(  -998.32), EASYSIMD_FLOAT32_C(  -704.17), EASYSIMD_FLOAT32_C(   194.13),
        EASYSIMD_FLOAT32_C(  -797.50), EASYSIMD_FLOAT32_C(   886.60), EASYSIMD_FLOAT32_C(   553.94), EASYSIMD_FLOAT32_C(  -183.55) },
      { EASYSIMD_FLOAT32_C(    36.99), EASYSIMD_FLOAT32_C(  -369.18), EASYSIMD_FLOAT32_C(  -210.38), EASYSIMD_FLOAT32_C(   648.73),
        EASYSIMD_FLOAT32_C(  -825.94), EASYSIMD_FLOAT32_C(   322.43), EASYSIMD_FLOAT32_C(  -141.28), EASYSIMD_FLOAT32_C(   972.04) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-673257.81), EASYSIMD_FLOAT32_C(-129809.24),
        EASYSIMD_FLOAT32_C(-550605.44), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-522700.53), EASYSIMD_FLOAT32_C( 46604.12) } },
    { { EASYSIMD_FLOAT32_C(   553.29), EASYSIMD_FLOAT32_C(   235.44), EASYSIMD_FLOAT32_C(   479.78), EASYSIMD_FLOAT32_C(  -845.99),
        EASYSIMD_FLOAT32_C(  -653.25), EASYSIMD_FLOAT32_C(   305.47), EASYSIMD_FLOAT32_C(   474.21), EASYSIMD_FLOAT32_C(   371.12) },
      UINT8_C(114),
      { EASYSIMD_FLOAT32_C(   139.54), EASYSIMD_FLOAT32_C(   679.67), EASYSIMD_FLOAT32_C(   656.09), EASYSIMD_FLOAT32_C(    83.40),
        EASYSIMD_FLOAT32_C(   -61.13), EASYSIMD_FLOAT32_C(  -801.27), EASYSIMD_FLOAT32_C(  -396.68), EASYSIMD_FLOAT32_C(   -59.45) },
      { EASYSIMD_FLOAT32_C(  -505.43), EASYSIMD_FLOAT32_C(   797.44), EASYSIMD_FLOAT32_C(   143.05), EASYSIMD_FLOAT32_C(  -618.83),
        EASYSIMD_FLOAT32_C(   351.38), EASYSIMD_FLOAT32_C(   959.50), EASYSIMD_FLOAT32_C(   418.15), EASYSIMD_FLOAT32_C(   982.20) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-160818.94), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-40284.55), EASYSIMD_FLOAT32_C(243804.45), EASYSIMD_FLOAT32_C(187691.45), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -250.88), EASYSIMD_FLOAT32_C(    66.88), EASYSIMD_FLOAT32_C(  -843.74), EASYSIMD_FLOAT32_C(  -928.46),
        EASYSIMD_FLOAT32_C(   925.61), EASYSIMD_FLOAT32_C(  -871.70), EASYSIMD_FLOAT32_C(   624.84), EASYSIMD_FLOAT32_C(   161.04) },
      UINT8_C(110),
      { EASYSIMD_FLOAT32_C(   778.85), EASYSIMD_FLOAT32_C(   507.80), EASYSIMD_FLOAT32_C(   -86.44), EASYSIMD_FLOAT32_C(   253.06),
        EASYSIMD_FLOAT32_C(  -121.08), EASYSIMD_FLOAT32_C(  -737.36), EASYSIMD_FLOAT32_C(  -607.41), EASYSIMD_FLOAT32_C(  -441.41) },
      { EASYSIMD_FLOAT32_C(   918.72), EASYSIMD_FLOAT32_C(   475.99), EASYSIMD_FLOAT32_C(   497.45), EASYSIMD_FLOAT32_C(  -882.54),
        EASYSIMD_FLOAT32_C(  -920.69), EASYSIMD_FLOAT32_C(  -562.00), EASYSIMD_FLOAT32_C(  -387.98), EASYSIMD_FLOAT32_C(   876.75) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-34437.65), EASYSIMD_FLOAT32_C(-73430.34), EASYSIMD_FLOAT32_C(235838.64),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-642194.69), EASYSIMD_FLOAT32_C(379922.03), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   581.05), EASYSIMD_FLOAT32_C(    -6.81), EASYSIMD_FLOAT32_C(   228.13), EASYSIMD_FLOAT32_C(   540.55),
        EASYSIMD_FLOAT32_C(  -588.66), EASYSIMD_FLOAT32_C(   210.33), EASYSIMD_FLOAT32_C(  -710.33), EASYSIMD_FLOAT32_C(   478.22) },
      UINT8_C( 96),
      { EASYSIMD_FLOAT32_C(  -638.79), EASYSIMD_FLOAT32_C(   403.83), EASYSIMD_FLOAT32_C(   494.89), EASYSIMD_FLOAT32_C(   986.05),
        EASYSIMD_FLOAT32_C(  -435.12), EASYSIMD_FLOAT32_C(   102.97), EASYSIMD_FLOAT32_C(   764.90), EASYSIMD_FLOAT32_C(  -927.32) },
      { EASYSIMD_FLOAT32_C(  -983.47), EASYSIMD_FLOAT32_C(    17.96), EASYSIMD_FLOAT32_C(   -48.41), EASYSIMD_FLOAT32_C(  -720.84),
        EASYSIMD_FLOAT32_C(   410.55), EASYSIMD_FLOAT32_C(   510.18), EASYSIMD_FLOAT32_C(  -802.12), EASYSIMD_FLOAT32_C(  -113.45) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-22167.86), EASYSIMD_FLOAT32_C(544133.56), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(     7.63), EASYSIMD_FLOAT32_C(  -684.66), EASYSIMD_FLOAT32_C(   -34.14), EASYSIMD_FLOAT32_C(   445.64),
        EASYSIMD_FLOAT32_C(   -72.64), EASYSIMD_FLOAT32_C(  -157.39), EASYSIMD_FLOAT32_C(    26.69), EASYSIMD_FLOAT32_C(   920.55) },
      UINT8_C(249),
      { EASYSIMD_FLOAT32_C(  -432.76), EASYSIMD_FLOAT32_C(  -668.11), EASYSIMD_FLOAT32_C(   281.08), EASYSIMD_FLOAT32_C(  -143.09),
        EASYSIMD_FLOAT32_C(   810.11), EASYSIMD_FLOAT32_C(  -352.33), EASYSIMD_FLOAT32_C(   218.12), EASYSIMD_FLOAT32_C(   213.94) },
      { EASYSIMD_FLOAT32_C(  -857.44), EASYSIMD_FLOAT32_C(   204.18), EASYSIMD_FLOAT32_C(   778.82), EASYSIMD_FLOAT32_C(   245.53),
        EASYSIMD_FLOAT32_C(   -30.92), EASYSIMD_FLOAT32_C(   851.49), EASYSIMD_FLOAT32_C(   262.05), EASYSIMD_FLOAT32_C(   987.04) },
      { EASYSIMD_FLOAT32_C(  4159.40), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 63521.10),
        EASYSIMD_FLOAT32_C( 58877.31), EASYSIMD_FLOAT32_C(-56304.70), EASYSIMD_FLOAT32_C( -6083.67), EASYSIMD_FLOAT32_C(-197929.52) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_fnmsub_ps(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_fnmsub_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 c = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_fnmsub_ps(k, a, b, c);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_fnmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const easysimd__mmask8 k;
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -880.96), EASYSIMD_FLOAT64_C(  -745.62), EASYSIMD_FLOAT64_C(  -485.51), EASYSIMD_FLOAT64_C(   -22.80) },
      UINT8_C(155),
      { EASYSIMD_FLOAT64_C(   346.35), EASYSIMD_FLOAT64_C(   -55.07), EASYSIMD_FLOAT64_C(  -755.67), EASYSIMD_FLOAT64_C(   690.92) },
      { EASYSIMD_FLOAT64_C(  -157.94), EASYSIMD_FLOAT64_C(  -998.07), EASYSIMD_FLOAT64_C(  -690.44), EASYSIMD_FLOAT64_C(  -826.44) },
      { EASYSIMD_FLOAT64_C(305278.44), EASYSIMD_FLOAT64_C(-40063.22), EASYSIMD_FLOAT64_C(  -485.51), EASYSIMD_FLOAT64_C( 16579.42) } },
    { { EASYSIMD_FLOAT64_C(  -364.04), EASYSIMD_FLOAT64_C(  -235.63), EASYSIMD_FLOAT64_C(   219.25), EASYSIMD_FLOAT64_C(  -710.04) },
      UINT8_C(204),
      { EASYSIMD_FLOAT64_C(   669.78), EASYSIMD_FLOAT64_C(    39.32), EASYSIMD_FLOAT64_C(  -820.75), EASYSIMD_FLOAT64_C(   132.98) },
      { EASYSIMD_FLOAT64_C(  -653.60), EASYSIMD_FLOAT64_C(    48.50), EASYSIMD_FLOAT64_C(  -324.50), EASYSIMD_FLOAT64_C(   753.13) },
      { EASYSIMD_FLOAT64_C(  -364.04), EASYSIMD_FLOAT64_C(  -235.63), EASYSIMD_FLOAT64_C(180273.94), EASYSIMD_FLOAT64_C( 93667.99) } },
    { { EASYSIMD_FLOAT64_C(  -503.47), EASYSIMD_FLOAT64_C(   -43.71), EASYSIMD_FLOAT64_C(   180.30), EASYSIMD_FLOAT64_C(   405.65) },
      UINT8_C(164),
      { EASYSIMD_FLOAT64_C(   299.34), EASYSIMD_FLOAT64_C(   660.03), EASYSIMD_FLOAT64_C(  -194.72), EASYSIMD_FLOAT64_C(  -723.46) },
      { EASYSIMD_FLOAT64_C(  -608.58), EASYSIMD_FLOAT64_C(  -848.38), EASYSIMD_FLOAT64_C(   221.47), EASYSIMD_FLOAT64_C(  -364.25) },
      { EASYSIMD_FLOAT64_C(  -503.47), EASYSIMD_FLOAT64_C(   -43.71), EASYSIMD_FLOAT64_C( 34886.55), EASYSIMD_FLOAT64_C(   405.65) } },
    { { EASYSIMD_FLOAT64_C(   842.54), EASYSIMD_FLOAT64_C(  -936.47), EASYSIMD_FLOAT64_C(  -362.32), EASYSIMD_FLOAT64_C(  -847.90) },
      UINT8_C( 12),
      { EASYSIMD_FLOAT64_C(   273.65), EASYSIMD_FLOAT64_C(   -83.53), EASYSIMD_FLOAT64_C(   456.35), EASYSIMD_FLOAT64_C(   563.61) },
      { EASYSIMD_FLOAT64_C(   567.09), EASYSIMD_FLOAT64_C(   126.12), EASYSIMD_FLOAT64_C(  -397.07), EASYSIMD_FLOAT64_C(   746.34) },
      { EASYSIMD_FLOAT64_C(   842.54), EASYSIMD_FLOAT64_C(  -936.47), EASYSIMD_FLOAT64_C(165741.80), EASYSIMD_FLOAT64_C(477138.58) } },
    { { EASYSIMD_FLOAT64_C(  -740.90), EASYSIMD_FLOAT64_C(   -50.67), EASYSIMD_FLOAT64_C(  -205.16), EASYSIMD_FLOAT64_C(   -65.40) },
      UINT8_C(195),
      { EASYSIMD_FLOAT64_C(   291.37), EASYSIMD_FLOAT64_C(   890.89), EASYSIMD_FLOAT64_C(   882.76), EASYSIMD_FLOAT64_C(  -302.98) },
      { EASYSIMD_FLOAT64_C(  -818.32), EASYSIMD_FLOAT64_C(   182.10), EASYSIMD_FLOAT64_C(  -642.96), EASYSIMD_FLOAT64_C(   -13.05) },
      { EASYSIMD_FLOAT64_C(216694.35), EASYSIMD_FLOAT64_C( 44959.30), EASYSIMD_FLOAT64_C(  -205.16), EASYSIMD_FLOAT64_C(   -65.40) } },
    { { EASYSIMD_FLOAT64_C(   458.65), EASYSIMD_FLOAT64_C(  -251.54), EASYSIMD_FLOAT64_C(   138.58), EASYSIMD_FLOAT64_C(  -319.88) },
      UINT8_C(218),
      { EASYSIMD_FLOAT64_C(   -18.88), EASYSIMD_FLOAT64_C(  -256.34), EASYSIMD_FLOAT64_C(  -978.11), EASYSIMD_FLOAT64_C(   133.21) },
      { EASYSIMD_FLOAT64_C(   -19.25), EASYSIMD_FLOAT64_C(   295.54), EASYSIMD_FLOAT64_C(  -950.32), EASYSIMD_FLOAT64_C(  -562.90) },
      { EASYSIMD_FLOAT64_C(   458.65), EASYSIMD_FLOAT64_C(-64775.30), EASYSIMD_FLOAT64_C(   138.58), EASYSIMD_FLOAT64_C( 43174.11) } },
    { { EASYSIMD_FLOAT64_C(  -140.85), EASYSIMD_FLOAT64_C(   616.77), EASYSIMD_FLOAT64_C(   563.22), EASYSIMD_FLOAT64_C(   462.08) },
      UINT8_C(  5),
      { EASYSIMD_FLOAT64_C(   822.32), EASYSIMD_FLOAT64_C(  -588.59), EASYSIMD_FLOAT64_C(  -842.05), EASYSIMD_FLOAT64_C(  -243.08) },
      { EASYSIMD_FLOAT64_C(   113.87), EASYSIMD_FLOAT64_C(   449.31), EASYSIMD_FLOAT64_C(  -352.19), EASYSIMD_FLOAT64_C(    -3.37) },
      { EASYSIMD_FLOAT64_C(115709.90), EASYSIMD_FLOAT64_C(   616.77), EASYSIMD_FLOAT64_C(474611.59), EASYSIMD_FLOAT64_C(   462.08) } },
    { { EASYSIMD_FLOAT64_C(  -853.67), EASYSIMD_FLOAT64_C(  -170.52), EASYSIMD_FLOAT64_C(  -821.27), EASYSIMD_FLOAT64_C(  -496.63) },
      UINT8_C(251),
      { EASYSIMD_FLOAT64_C(   637.38), EASYSIMD_FLOAT64_C(   251.83), EASYSIMD_FLOAT64_C(   -44.99), EASYSIMD_FLOAT64_C(  -682.50) },
      { EASYSIMD_FLOAT64_C(  -363.96), EASYSIMD_FLOAT64_C(   936.13), EASYSIMD_FLOAT64_C(    61.15), EASYSIMD_FLOAT64_C(  -342.07) },
      { EASYSIMD_FLOAT64_C(544476.14), EASYSIMD_FLOAT64_C( 42005.92), EASYSIMD_FLOAT64_C(  -821.27), EASYSIMD_FLOAT64_C(-338607.90) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_fnmsub_pd(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_fnmsub_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d c = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_fnmsub_pd(a, k, b, c);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_fnmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const easysimd__mmask8 k;
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -237.04), EASYSIMD_FLOAT64_C(   513.27), EASYSIMD_FLOAT64_C(   706.16), EASYSIMD_FLOAT64_C(  -853.25) },
      UINT8_C( 97),
      { EASYSIMD_FLOAT64_C(   440.43), EASYSIMD_FLOAT64_C(  -857.27), EASYSIMD_FLOAT64_C(  -948.93), EASYSIMD_FLOAT64_C(    82.33) },
      { EASYSIMD_FLOAT64_C(   204.04), EASYSIMD_FLOAT64_C(  -426.68), EASYSIMD_FLOAT64_C(   235.37), EASYSIMD_FLOAT64_C(   756.06) },
      { EASYSIMD_FLOAT64_C(104195.49), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   949.35), EASYSIMD_FLOAT64_C(   130.34), EASYSIMD_FLOAT64_C(   299.53), EASYSIMD_FLOAT64_C(  -959.03) },
      UINT8_C(138),
      { EASYSIMD_FLOAT64_C(  -998.88), EASYSIMD_FLOAT64_C(   820.02), EASYSIMD_FLOAT64_C(   224.34), EASYSIMD_FLOAT64_C(   -54.35) },
      { EASYSIMD_FLOAT64_C(  -683.47), EASYSIMD_FLOAT64_C(   808.18), EASYSIMD_FLOAT64_C(   172.12), EASYSIMD_FLOAT64_C(  -124.67) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-107689.59), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-51998.61) } },
    { { EASYSIMD_FLOAT64_C(   658.36), EASYSIMD_FLOAT64_C(   422.70), EASYSIMD_FLOAT64_C(   795.08), EASYSIMD_FLOAT64_C(  -213.27) },
      UINT8_C( 70),
      { EASYSIMD_FLOAT64_C(  -441.96), EASYSIMD_FLOAT64_C(  -700.00), EASYSIMD_FLOAT64_C(  -481.03), EASYSIMD_FLOAT64_C(  -295.21) },
      { EASYSIMD_FLOAT64_C(  -919.06), EASYSIMD_FLOAT64_C(   959.40), EASYSIMD_FLOAT64_C(  -152.48), EASYSIMD_FLOAT64_C(  -867.98) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(294930.60), EASYSIMD_FLOAT64_C(382609.81), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(    41.74), EASYSIMD_FLOAT64_C(  -948.44), EASYSIMD_FLOAT64_C(  -294.66), EASYSIMD_FLOAT64_C(  -722.89) },
      UINT8_C(126),
      { EASYSIMD_FLOAT64_C(  -345.32), EASYSIMD_FLOAT64_C(   407.45), EASYSIMD_FLOAT64_C(   107.15), EASYSIMD_FLOAT64_C(  -304.35) },
      { EASYSIMD_FLOAT64_C(  -214.79), EASYSIMD_FLOAT64_C(   108.27), EASYSIMD_FLOAT64_C(  -484.33), EASYSIMD_FLOAT64_C(  -990.44) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(386333.61), EASYSIMD_FLOAT64_C( 32057.15), EASYSIMD_FLOAT64_C(-219021.13) } },
    { { EASYSIMD_FLOAT64_C(  -946.09), EASYSIMD_FLOAT64_C(  -167.80), EASYSIMD_FLOAT64_C(   817.74), EASYSIMD_FLOAT64_C(   226.04) },
      UINT8_C(227),
      { EASYSIMD_FLOAT64_C(   476.10), EASYSIMD_FLOAT64_C(  -351.26), EASYSIMD_FLOAT64_C(   502.61), EASYSIMD_FLOAT64_C(  -737.17) },
      { EASYSIMD_FLOAT64_C(   461.55), EASYSIMD_FLOAT64_C(  -939.35), EASYSIMD_FLOAT64_C(  -437.18), EASYSIMD_FLOAT64_C(   980.52) },
      { EASYSIMD_FLOAT64_C(449971.90), EASYSIMD_FLOAT64_C(-58002.08), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -234.56), EASYSIMD_FLOAT64_C(  -356.23), EASYSIMD_FLOAT64_C(   939.92), EASYSIMD_FLOAT64_C(   612.96) },
      UINT8_C(111),
      { EASYSIMD_FLOAT64_C(   -18.34), EASYSIMD_FLOAT64_C(   664.52), EASYSIMD_FLOAT64_C(   481.12), EASYSIMD_FLOAT64_C(   258.76) },
      { EASYSIMD_FLOAT64_C(   472.14), EASYSIMD_FLOAT64_C(  -864.20), EASYSIMD_FLOAT64_C(  -333.79), EASYSIMD_FLOAT64_C(  -420.71) },
      { EASYSIMD_FLOAT64_C( -4773.97), EASYSIMD_FLOAT64_C(237586.16), EASYSIMD_FLOAT64_C(-451880.52), EASYSIMD_FLOAT64_C(-158188.82) } },
    { { EASYSIMD_FLOAT64_C(  -168.55), EASYSIMD_FLOAT64_C(   451.42), EASYSIMD_FLOAT64_C(   687.56), EASYSIMD_FLOAT64_C(   347.12) },
      UINT8_C( 35),
      { EASYSIMD_FLOAT64_C(   741.47), EASYSIMD_FLOAT64_C(  -820.68), EASYSIMD_FLOAT64_C(   278.71), EASYSIMD_FLOAT64_C(   -32.50) },
      { EASYSIMD_FLOAT64_C(   886.85), EASYSIMD_FLOAT64_C(  -245.18), EASYSIMD_FLOAT64_C(   616.24), EASYSIMD_FLOAT64_C(   389.46) },
      { EASYSIMD_FLOAT64_C(124087.92), EASYSIMD_FLOAT64_C(370716.55), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(    17.64), EASYSIMD_FLOAT64_C(    77.79), EASYSIMD_FLOAT64_C(   450.10), EASYSIMD_FLOAT64_C(   580.47) },
      UINT8_C(128),
      { EASYSIMD_FLOAT64_C(  -784.46), EASYSIMD_FLOAT64_C(  -775.77), EASYSIMD_FLOAT64_C(    -1.77), EASYSIMD_FLOAT64_C(   828.50) },
      { EASYSIMD_FLOAT64_C(     0.02), EASYSIMD_FLOAT64_C(   979.89), EASYSIMD_FLOAT64_C(   493.02), EASYSIMD_FLOAT64_C(  -518.86) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_fnmsub_pd(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_fnmsub_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d c = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_fnmsub_pd(k, a, b, c);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_fnmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 c[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    38.95), EASYSIMD_FLOAT32_C(   -13.06), EASYSIMD_FLOAT32_C(    27.88), EASYSIMD_FLOAT32_C(    62.28),
        EASYSIMD_FLOAT32_C(    66.10), EASYSIMD_FLOAT32_C(   -82.38), EASYSIMD_FLOAT32_C(     5.33), EASYSIMD_FLOAT32_C(    59.08),
        EASYSIMD_FLOAT32_C(    11.50), EASYSIMD_FLOAT32_C(     2.93), EASYSIMD_FLOAT32_C(    86.03), EASYSIMD_FLOAT32_C(   -85.53),
        EASYSIMD_FLOAT32_C(    41.10), EASYSIMD_FLOAT32_C(   -96.18), EASYSIMD_FLOAT32_C(    60.27), EASYSIMD_FLOAT32_C(    88.72) },
      { EASYSIMD_FLOAT32_C(    84.81), EASYSIMD_FLOAT32_C(    87.90), EASYSIMD_FLOAT32_C(    94.42), EASYSIMD_FLOAT32_C(   -72.92),
        EASYSIMD_FLOAT32_C(   -94.27), EASYSIMD_FLOAT32_C(    82.63), EASYSIMD_FLOAT32_C(    39.45), EASYSIMD_FLOAT32_C(    66.67),
        EASYSIMD_FLOAT32_C(    16.72), EASYSIMD_FLOAT32_C(     0.77), EASYSIMD_FLOAT32_C(   -40.95), EASYSIMD_FLOAT32_C(    62.84),
        EASYSIMD_FLOAT32_C(   -22.56), EASYSIMD_FLOAT32_C(   -32.15), EASYSIMD_FLOAT32_C(    60.85), EASYSIMD_FLOAT32_C(   -83.61) },
      { EASYSIMD_FLOAT32_C(    54.78), EASYSIMD_FLOAT32_C(   -11.27), EASYSIMD_FLOAT32_C(    78.67), EASYSIMD_FLOAT32_C(    20.88),
        EASYSIMD_FLOAT32_C(     6.35), EASYSIMD_FLOAT32_C(   -16.00), EASYSIMD_FLOAT32_C(   -20.03), EASYSIMD_FLOAT32_C(   -82.16),
        EASYSIMD_FLOAT32_C(    86.93), EASYSIMD_FLOAT32_C(   -34.00), EASYSIMD_FLOAT32_C(   -67.68), EASYSIMD_FLOAT32_C(    28.02),
        EASYSIMD_FLOAT32_C(   -30.18), EASYSIMD_FLOAT32_C(    92.59), EASYSIMD_FLOAT32_C(    16.74), EASYSIMD_FLOAT32_C(   -45.37) },
      { EASYSIMD_FLOAT32_C( -3358.13), EASYSIMD_FLOAT32_C(  1159.24), EASYSIMD_FLOAT32_C( -2711.10), EASYSIMD_FLOAT32_C(  4520.58),
        EASYSIMD_FLOAT32_C(  6224.90), EASYSIMD_FLOAT32_C(  6823.06), EASYSIMD_FLOAT32_C(  -190.24), EASYSIMD_FLOAT32_C( -3856.70),
        EASYSIMD_FLOAT32_C(  -279.21), EASYSIMD_FLOAT32_C(    31.74), EASYSIMD_FLOAT32_C(  3590.61), EASYSIMD_FLOAT32_C(  5346.69),
        EASYSIMD_FLOAT32_C(   957.40), EASYSIMD_FLOAT32_C( -3184.78), EASYSIMD_FLOAT32_C( -3684.17), EASYSIMD_FLOAT32_C(  7463.25) } },
    { { EASYSIMD_FLOAT32_C(    80.49), EASYSIMD_FLOAT32_C(    11.16), EASYSIMD_FLOAT32_C(   -18.30), EASYSIMD_FLOAT32_C(    86.22),
        EASYSIMD_FLOAT32_C(    -6.21), EASYSIMD_FLOAT32_C(   -78.85), EASYSIMD_FLOAT32_C(    52.89), EASYSIMD_FLOAT32_C(   -89.49),
        EASYSIMD_FLOAT32_C(    21.92), EASYSIMD_FLOAT32_C(   -88.06), EASYSIMD_FLOAT32_C(    73.35), EASYSIMD_FLOAT32_C(    99.36),
        EASYSIMD_FLOAT32_C(   -20.21), EASYSIMD_FLOAT32_C(    34.20), EASYSIMD_FLOAT32_C(   -84.24), EASYSIMD_FLOAT32_C(   -65.43) },
      { EASYSIMD_FLOAT32_C(   -77.07), EASYSIMD_FLOAT32_C(    94.43), EASYSIMD_FLOAT32_C(    55.45), EASYSIMD_FLOAT32_C(    29.28),
        EASYSIMD_FLOAT32_C(   -21.57), EASYSIMD_FLOAT32_C(   -64.58), EASYSIMD_FLOAT32_C(    47.12), EASYSIMD_FLOAT32_C(   -34.64),
        EASYSIMD_FLOAT32_C(     1.42), EASYSIMD_FLOAT32_C(    79.44), EASYSIMD_FLOAT32_C(    93.38), EASYSIMD_FLOAT32_C(    71.24),
        EASYSIMD_FLOAT32_C(    72.03), EASYSIMD_FLOAT32_C(    10.12), EASYSIMD_FLOAT32_C(   -74.14), EASYSIMD_FLOAT32_C(    52.52) },
      { EASYSIMD_FLOAT32_C(   -78.71), EASYSIMD_FLOAT32_C(     7.57), EASYSIMD_FLOAT32_C(    38.73), EASYSIMD_FLOAT32_C(    15.08),
        EASYSIMD_FLOAT32_C(    28.72), EASYSIMD_FLOAT32_C(    -8.38), EASYSIMD_FLOAT32_C(    25.59), EASYSIMD_FLOAT32_C(   -49.36),
        EASYSIMD_FLOAT32_C(     3.56), EASYSIMD_FLOAT32_C(    -1.06), EASYSIMD_FLOAT32_C(   -50.00), EASYSIMD_FLOAT32_C(    83.35),
        EASYSIMD_FLOAT32_C(   -66.86), EASYSIMD_FLOAT32_C(   -34.24), EASYSIMD_FLOAT32_C(   -82.08), EASYSIMD_FLOAT32_C(   -43.92) },
      { EASYSIMD_FLOAT32_C(  6282.07), EASYSIMD_FLOAT32_C( -1061.41), EASYSIMD_FLOAT32_C(   976.01), EASYSIMD_FLOAT32_C( -2539.60),
        EASYSIMD_FLOAT32_C(  -162.67), EASYSIMD_FLOAT32_C( -5083.75), EASYSIMD_FLOAT32_C( -2517.77), EASYSIMD_FLOAT32_C( -3050.57),
        EASYSIMD_FLOAT32_C(   -34.69), EASYSIMD_FLOAT32_C(  6996.55), EASYSIMD_FLOAT32_C( -6799.42), EASYSIMD_FLOAT32_C( -7161.76),
        EASYSIMD_FLOAT32_C(  1522.59), EASYSIMD_FLOAT32_C(  -311.86), EASYSIMD_FLOAT32_C( -6163.47), EASYSIMD_FLOAT32_C(  3480.30) } },
    { { EASYSIMD_FLOAT32_C(   -39.81), EASYSIMD_FLOAT32_C(    73.37), EASYSIMD_FLOAT32_C(    85.36), EASYSIMD_FLOAT32_C(    38.62),
        EASYSIMD_FLOAT32_C(   -91.20), EASYSIMD_FLOAT32_C(    32.48), EASYSIMD_FLOAT32_C(   -96.03), EASYSIMD_FLOAT32_C(    10.22),
        EASYSIMD_FLOAT32_C(    11.92), EASYSIMD_FLOAT32_C(    97.35), EASYSIMD_FLOAT32_C(   -18.55), EASYSIMD_FLOAT32_C(   -16.05),
        EASYSIMD_FLOAT32_C(     7.48), EASYSIMD_FLOAT32_C(     7.32), EASYSIMD_FLOAT32_C(   -63.53), EASYSIMD_FLOAT32_C(    28.76) },
      { EASYSIMD_FLOAT32_C(   -85.12), EASYSIMD_FLOAT32_C(    75.20), EASYSIMD_FLOAT32_C(   -56.16), EASYSIMD_FLOAT32_C(    43.60),
        EASYSIMD_FLOAT32_C(   -33.18), EASYSIMD_FLOAT32_C(    69.43), EASYSIMD_FLOAT32_C(    94.24), EASYSIMD_FLOAT32_C(    70.38),
        EASYSIMD_FLOAT32_C(   -31.63), EASYSIMD_FLOAT32_C(   -55.76), EASYSIMD_FLOAT32_C(    53.73), EASYSIMD_FLOAT32_C(     1.51),
        EASYSIMD_FLOAT32_C(    10.00), EASYSIMD_FLOAT32_C(    71.65), EASYSIMD_FLOAT32_C(    57.59), EASYSIMD_FLOAT32_C(    70.19) },
      { EASYSIMD_FLOAT32_C(    45.02), EASYSIMD_FLOAT32_C(    42.95), EASYSIMD_FLOAT32_C(     8.80), EASYSIMD_FLOAT32_C(    53.81),
        EASYSIMD_FLOAT32_C(   -24.57), EASYSIMD_FLOAT32_C(    12.78), EASYSIMD_FLOAT32_C(   -35.97), EASYSIMD_FLOAT32_C(    87.34),
        EASYSIMD_FLOAT32_C(    10.13), EASYSIMD_FLOAT32_C(    45.48), EASYSIMD_FLOAT32_C(   -28.70), EASYSIMD_FLOAT32_C(   -82.39),
        EASYSIMD_FLOAT32_C(   -47.20), EASYSIMD_FLOAT32_C(     7.76), EASYSIMD_FLOAT32_C(    46.37), EASYSIMD_FLOAT32_C(   -32.32) },
      { EASYSIMD_FLOAT32_C( -3433.65), EASYSIMD_FLOAT32_C( -5560.37), EASYSIMD_FLOAT32_C(  4785.02), EASYSIMD_FLOAT32_C( -1737.64),
        EASYSIMD_FLOAT32_C( -3001.45), EASYSIMD_FLOAT32_C( -2267.87), EASYSIMD_FLOAT32_C(  9085.84), EASYSIMD_FLOAT32_C(  -806.62),
        EASYSIMD_FLOAT32_C(   366.90), EASYSIMD_FLOAT32_C(  5382.76), EASYSIMD_FLOAT32_C(  1025.39), EASYSIMD_FLOAT32_C(   106.63),
        EASYSIMD_FLOAT32_C(   -27.60), EASYSIMD_FLOAT32_C(  -532.24), EASYSIMD_FLOAT32_C(  3612.32), EASYSIMD_FLOAT32_C( -1986.34) } },
    { { EASYSIMD_FLOAT32_C(   -17.04), EASYSIMD_FLOAT32_C(    90.21), EASYSIMD_FLOAT32_C(   -88.72), EASYSIMD_FLOAT32_C(    49.79),
        EASYSIMD_FLOAT32_C(    59.64), EASYSIMD_FLOAT32_C(   -94.48), EASYSIMD_FLOAT32_C(    20.17), EASYSIMD_FLOAT32_C(   -71.99),
        EASYSIMD_FLOAT32_C(   -50.24), EASYSIMD_FLOAT32_C(   -26.10), EASYSIMD_FLOAT32_C(    29.52), EASYSIMD_FLOAT32_C(    59.76),
        EASYSIMD_FLOAT32_C(   -54.46), EASYSIMD_FLOAT32_C(   -12.89), EASYSIMD_FLOAT32_C(    29.94), EASYSIMD_FLOAT32_C(    90.56) },
      { EASYSIMD_FLOAT32_C(   -69.94), EASYSIMD_FLOAT32_C(   -61.25), EASYSIMD_FLOAT32_C(    44.38), EASYSIMD_FLOAT32_C(     5.48),
        EASYSIMD_FLOAT32_C(    51.53), EASYSIMD_FLOAT32_C(   -91.59), EASYSIMD_FLOAT32_C(    -7.17), EASYSIMD_FLOAT32_C(   -38.34),
        EASYSIMD_FLOAT32_C(    53.89), EASYSIMD_FLOAT32_C(    64.12), EASYSIMD_FLOAT32_C(   -20.73), EASYSIMD_FLOAT32_C(   -93.31),
        EASYSIMD_FLOAT32_C(   -28.11), EASYSIMD_FLOAT32_C(   -74.36), EASYSIMD_FLOAT32_C(   -25.63), EASYSIMD_FLOAT32_C(    54.85) },
      { EASYSIMD_FLOAT32_C(   -84.15), EASYSIMD_FLOAT32_C(   -14.35), EASYSIMD_FLOAT32_C(     4.64), EASYSIMD_FLOAT32_C(    75.49),
        EASYSIMD_FLOAT32_C(    -8.83), EASYSIMD_FLOAT32_C(   -75.19), EASYSIMD_FLOAT32_C(   -96.50), EASYSIMD_FLOAT32_C(    40.92),
        EASYSIMD_FLOAT32_C(    -1.30), EASYSIMD_FLOAT32_C(    33.03), EASYSIMD_FLOAT32_C(     0.68), EASYSIMD_FLOAT32_C(    44.25),
        EASYSIMD_FLOAT32_C(   -79.86), EASYSIMD_FLOAT32_C(   -69.37), EASYSIMD_FLOAT32_C(    34.81), EASYSIMD_FLOAT32_C(   -49.80) },
      { EASYSIMD_FLOAT32_C( -1107.63), EASYSIMD_FLOAT32_C(  5539.71), EASYSIMD_FLOAT32_C(  3932.75), EASYSIMD_FLOAT32_C(  -348.34),
        EASYSIMD_FLOAT32_C( -3064.42), EASYSIMD_FLOAT32_C( -8578.23), EASYSIMD_FLOAT32_C(   241.12), EASYSIMD_FLOAT32_C( -2801.02),
        EASYSIMD_FLOAT32_C(  2708.73), EASYSIMD_FLOAT32_C(  1640.50), EASYSIMD_FLOAT32_C(   611.27), EASYSIMD_FLOAT32_C(  5531.96),
        EASYSIMD_FLOAT32_C( -1451.01), EASYSIMD_FLOAT32_C(  -889.13), EASYSIMD_FLOAT32_C(   732.55), EASYSIMD_FLOAT32_C( -4917.42) } },
    { { EASYSIMD_FLOAT32_C(   -30.63), EASYSIMD_FLOAT32_C(   -20.81), EASYSIMD_FLOAT32_C(    55.68), EASYSIMD_FLOAT32_C(   -79.10),
        EASYSIMD_FLOAT32_C(   -12.41), EASYSIMD_FLOAT32_C(   -51.49), EASYSIMD_FLOAT32_C(   -17.44), EASYSIMD_FLOAT32_C(   -58.52),
        EASYSIMD_FLOAT32_C(   -87.36), EASYSIMD_FLOAT32_C(    61.82), EASYSIMD_FLOAT32_C(   -51.83), EASYSIMD_FLOAT32_C(   -15.47),
        EASYSIMD_FLOAT32_C(    87.46), EASYSIMD_FLOAT32_C(    22.53), EASYSIMD_FLOAT32_C(   -60.62), EASYSIMD_FLOAT32_C(   -96.69) },
      { EASYSIMD_FLOAT32_C(   -91.82), EASYSIMD_FLOAT32_C(    44.02), EASYSIMD_FLOAT32_C(    78.80), EASYSIMD_FLOAT32_C(    -0.65),
        EASYSIMD_FLOAT32_C(    68.82), EASYSIMD_FLOAT32_C(    82.31), EASYSIMD_FLOAT32_C(   -59.73), EASYSIMD_FLOAT32_C(   -32.48),
        EASYSIMD_FLOAT32_C(    15.33), EASYSIMD_FLOAT32_C(    40.95), EASYSIMD_FLOAT32_C(   -88.23), EASYSIMD_FLOAT32_C(    35.47),
        EASYSIMD_FLOAT32_C(    71.58), EASYSIMD_FLOAT32_C(    46.58), EASYSIMD_FLOAT32_C(    85.67), EASYSIMD_FLOAT32_C(   -59.05) },
      { EASYSIMD_FLOAT32_C(   -74.23), EASYSIMD_FLOAT32_C(    41.35), EASYSIMD_FLOAT32_C(   -38.15), EASYSIMD_FLOAT32_C(    13.36),
        EASYSIMD_FLOAT32_C(    89.87), EASYSIMD_FLOAT32_C(    44.41), EASYSIMD_FLOAT32_C(    54.84), EASYSIMD_FLOAT32_C(   -97.49),
        EASYSIMD_FLOAT32_C(     6.23), EASYSIMD_FLOAT32_C(   -97.00), EASYSIMD_FLOAT32_C(   -12.97), EASYSIMD_FLOAT32_C(    -6.30),
        EASYSIMD_FLOAT32_C(    25.54), EASYSIMD_FLOAT32_C(    26.41), EASYSIMD_FLOAT32_C(    -2.99), EASYSIMD_FLOAT32_C(    33.72) },
      { EASYSIMD_FLOAT32_C( -2738.22), EASYSIMD_FLOAT32_C(   874.71), EASYSIMD_FLOAT32_C( -4349.43), EASYSIMD_FLOAT32_C(   -64.77),
        EASYSIMD_FLOAT32_C(   764.19), EASYSIMD_FLOAT32_C(  4193.73), EASYSIMD_FLOAT32_C( -1096.53), EASYSIMD_FLOAT32_C( -1803.24),
        EASYSIMD_FLOAT32_C(  1333.00), EASYSIMD_FLOAT32_C( -2434.53), EASYSIMD_FLOAT32_C( -4559.99), EASYSIMD_FLOAT32_C(   555.02),
        EASYSIMD_FLOAT32_C( -6285.93), EASYSIMD_FLOAT32_C( -1075.86), EASYSIMD_FLOAT32_C(  5196.31), EASYSIMD_FLOAT32_C( -5743.26) } },
    { { EASYSIMD_FLOAT32_C(   -29.57), EASYSIMD_FLOAT32_C(   -24.19), EASYSIMD_FLOAT32_C(   -66.94), EASYSIMD_FLOAT32_C(   -60.75),
        EASYSIMD_FLOAT32_C(   -41.88), EASYSIMD_FLOAT32_C(   -26.67), EASYSIMD_FLOAT32_C(     6.77), EASYSIMD_FLOAT32_C(    73.45),
        EASYSIMD_FLOAT32_C(   -85.71), EASYSIMD_FLOAT32_C(    18.54), EASYSIMD_FLOAT32_C(     8.92), EASYSIMD_FLOAT32_C(    85.87),
        EASYSIMD_FLOAT32_C(   -34.88), EASYSIMD_FLOAT32_C(    -5.41), EASYSIMD_FLOAT32_C(   -73.18), EASYSIMD_FLOAT32_C(    -9.11) },
      { EASYSIMD_FLOAT32_C(   -64.05), EASYSIMD_FLOAT32_C(   -11.32), EASYSIMD_FLOAT32_C(   -95.75), EASYSIMD_FLOAT32_C(   -74.18),
        EASYSIMD_FLOAT32_C(   -66.91), EASYSIMD_FLOAT32_C(    59.08), EASYSIMD_FLOAT32_C(   -71.68), EASYSIMD_FLOAT32_C(    39.32),
        EASYSIMD_FLOAT32_C(    62.09), EASYSIMD_FLOAT32_C(    15.35), EASYSIMD_FLOAT32_C(   -66.98), EASYSIMD_FLOAT32_C(   -12.37),
        EASYSIMD_FLOAT32_C(   -58.24), EASYSIMD_FLOAT32_C(    30.03), EASYSIMD_FLOAT32_C(   -78.66), EASYSIMD_FLOAT32_C(    12.19) },
      { EASYSIMD_FLOAT32_C(   -94.16), EASYSIMD_FLOAT32_C(   -45.59), EASYSIMD_FLOAT32_C(    51.44), EASYSIMD_FLOAT32_C(   -36.04),
        EASYSIMD_FLOAT32_C(    27.74), EASYSIMD_FLOAT32_C(   -41.79), EASYSIMD_FLOAT32_C(   -62.59), EASYSIMD_FLOAT32_C(    42.03),
        EASYSIMD_FLOAT32_C(    76.75), EASYSIMD_FLOAT32_C(    46.33), EASYSIMD_FLOAT32_C(    27.90), EASYSIMD_FLOAT32_C(   -58.12),
        EASYSIMD_FLOAT32_C(   -59.08), EASYSIMD_FLOAT32_C(    54.72), EASYSIMD_FLOAT32_C(    32.77), EASYSIMD_FLOAT32_C(   -23.13) },
      { EASYSIMD_FLOAT32_C( -1799.80), EASYSIMD_FLOAT32_C(  -228.24), EASYSIMD_FLOAT32_C( -6460.95), EASYSIMD_FLOAT32_C( -4470.40),
        EASYSIMD_FLOAT32_C( -2829.93), EASYSIMD_FLOAT32_C(  1617.45), EASYSIMD_FLOAT32_C(   547.86), EASYSIMD_FLOAT32_C( -2930.08),
        EASYSIMD_FLOAT32_C(  5244.98), EASYSIMD_FLOAT32_C(  -330.92), EASYSIMD_FLOAT32_C(   569.56), EASYSIMD_FLOAT32_C(  1120.33),
        EASYSIMD_FLOAT32_C( -1972.33), EASYSIMD_FLOAT32_C(   107.74), EASYSIMD_FLOAT32_C( -5789.11), EASYSIMD_FLOAT32_C(   134.18) } },
    { { EASYSIMD_FLOAT32_C(   -56.60), EASYSIMD_FLOAT32_C(    37.01), EASYSIMD_FLOAT32_C(     2.69), EASYSIMD_FLOAT32_C(   -23.52),
        EASYSIMD_FLOAT32_C(    -3.90), EASYSIMD_FLOAT32_C(    31.01), EASYSIMD_FLOAT32_C(   -84.20), EASYSIMD_FLOAT32_C(   -41.81),
        EASYSIMD_FLOAT32_C(   -53.64), EASYSIMD_FLOAT32_C(   -51.18), EASYSIMD_FLOAT32_C(    45.81), EASYSIMD_FLOAT32_C(   -11.87),
        EASYSIMD_FLOAT32_C(    78.85), EASYSIMD_FLOAT32_C(    67.16), EASYSIMD_FLOAT32_C(   -99.68), EASYSIMD_FLOAT32_C(    84.69) },
      { EASYSIMD_FLOAT32_C(   -78.44), EASYSIMD_FLOAT32_C(    51.75), EASYSIMD_FLOAT32_C(   -51.36), EASYSIMD_FLOAT32_C(    49.31),
        EASYSIMD_FLOAT32_C(   -90.04), EASYSIMD_FLOAT32_C(   -13.95), EASYSIMD_FLOAT32_C(    -8.66), EASYSIMD_FLOAT32_C(    86.71),
        EASYSIMD_FLOAT32_C(   -67.62), EASYSIMD_FLOAT32_C(   -80.76), EASYSIMD_FLOAT32_C(   -71.41), EASYSIMD_FLOAT32_C(   -26.70),
        EASYSIMD_FLOAT32_C(    73.96), EASYSIMD_FLOAT32_C(    61.35), EASYSIMD_FLOAT32_C(    50.17), EASYSIMD_FLOAT32_C(   -82.65) },
      { EASYSIMD_FLOAT32_C(    -1.64), EASYSIMD_FLOAT32_C(   -47.14), EASYSIMD_FLOAT32_C(    -6.17), EASYSIMD_FLOAT32_C(    94.46),
        EASYSIMD_FLOAT32_C(    83.87), EASYSIMD_FLOAT32_C(     9.63), EASYSIMD_FLOAT32_C(   -47.35), EASYSIMD_FLOAT32_C(   -69.77),
        EASYSIMD_FLOAT32_C(    58.45), EASYSIMD_FLOAT32_C(    98.46), EASYSIMD_FLOAT32_C(    18.36), EASYSIMD_FLOAT32_C(    37.30),
        EASYSIMD_FLOAT32_C(    65.61), EASYSIMD_FLOAT32_C(    18.67), EASYSIMD_FLOAT32_C(    21.98), EASYSIMD_FLOAT32_C(    87.18) },
      { EASYSIMD_FLOAT32_C( -4438.06), EASYSIMD_FLOAT32_C( -1868.13), EASYSIMD_FLOAT32_C(   144.33), EASYSIMD_FLOAT32_C(  1065.31),
        EASYSIMD_FLOAT32_C(  -435.03), EASYSIMD_FLOAT32_C(   422.96), EASYSIMD_FLOAT32_C(  -681.82), EASYSIMD_FLOAT32_C(  3695.12),
        EASYSIMD_FLOAT32_C( -3685.59), EASYSIMD_FLOAT32_C( -4231.76), EASYSIMD_FLOAT32_C(  3252.93), EASYSIMD_FLOAT32_C(  -354.23),
        EASYSIMD_FLOAT32_C( -5897.36), EASYSIMD_FLOAT32_C( -4138.94), EASYSIMD_FLOAT32_C(  4978.97), EASYSIMD_FLOAT32_C(  6912.45) } },
    { { EASYSIMD_FLOAT32_C(   -29.57), EASYSIMD_FLOAT32_C(    70.63), EASYSIMD_FLOAT32_C(    36.48), EASYSIMD_FLOAT32_C(   -19.61),
        EASYSIMD_FLOAT32_C(   -43.32), EASYSIMD_FLOAT32_C(   -72.18), EASYSIMD_FLOAT32_C(   -32.90), EASYSIMD_FLOAT32_C(   -10.94),
        EASYSIMD_FLOAT32_C(   -52.94), EASYSIMD_FLOAT32_C(    -4.31), EASYSIMD_FLOAT32_C(    62.36), EASYSIMD_FLOAT32_C(   -78.99),
        EASYSIMD_FLOAT32_C(   -42.96), EASYSIMD_FLOAT32_C(    12.54), EASYSIMD_FLOAT32_C(   -61.64), EASYSIMD_FLOAT32_C(    55.40) },
      { EASYSIMD_FLOAT32_C(    65.40), EASYSIMD_FLOAT32_C(    32.20), EASYSIMD_FLOAT32_C(    49.86), EASYSIMD_FLOAT32_C(    49.27),
        EASYSIMD_FLOAT32_C(   -58.17), EASYSIMD_FLOAT32_C(   -97.49), EASYSIMD_FLOAT32_C(    79.50), EASYSIMD_FLOAT32_C(   -99.71),
        EASYSIMD_FLOAT32_C(   -99.03), EASYSIMD_FLOAT32_C(    -2.14), EASYSIMD_FLOAT32_C(    37.59), EASYSIMD_FLOAT32_C(    66.58),
        EASYSIMD_FLOAT32_C(   -83.46), EASYSIMD_FLOAT32_C(   -40.43), EASYSIMD_FLOAT32_C(    53.75), EASYSIMD_FLOAT32_C(   -13.04) },
      { EASYSIMD_FLOAT32_C(   -69.80), EASYSIMD_FLOAT32_C(    -9.76), EASYSIMD_FLOAT32_C(    67.35), EASYSIMD_FLOAT32_C(   -13.12),
        EASYSIMD_FLOAT32_C(    18.06), EASYSIMD_FLOAT32_C(   -65.56), EASYSIMD_FLOAT32_C(    75.94), EASYSIMD_FLOAT32_C(    65.11),
        EASYSIMD_FLOAT32_C(    30.13), EASYSIMD_FLOAT32_C(    38.30), EASYSIMD_FLOAT32_C(    86.12), EASYSIMD_FLOAT32_C(    87.17),
        EASYSIMD_FLOAT32_C(   -49.16), EASYSIMD_FLOAT32_C(   -75.51), EASYSIMD_FLOAT32_C(    42.57), EASYSIMD_FLOAT32_C(   -83.76) },
      { EASYSIMD_FLOAT32_C(  2003.68), EASYSIMD_FLOAT32_C( -2264.53), EASYSIMD_FLOAT32_C( -1886.24), EASYSIMD_FLOAT32_C(   979.30),
        EASYSIMD_FLOAT32_C( -2537.98), EASYSIMD_FLOAT32_C( -6971.27), EASYSIMD_FLOAT32_C(  2539.61), EASYSIMD_FLOAT32_C( -1155.94),
        EASYSIMD_FLOAT32_C( -5272.78), EASYSIMD_FLOAT32_C(   -47.52), EASYSIMD_FLOAT32_C( -2430.23), EASYSIMD_FLOAT32_C(  5171.98),
        EASYSIMD_FLOAT32_C( -3536.28), EASYSIMD_FLOAT32_C(   582.50), EASYSIMD_FLOAT32_C(  3270.58), EASYSIMD_FLOAT32_C(   806.18) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 c = easysimd_mm512_loadu_ps(test_vec[i].c);
    easysimd__m512 r = easysimd_mm512_fnmsub_ps(a, b, c);
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_fnmsub_ps(a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_fnmsub_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_fnmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const easysimd__mmask16 k;
    const easysimd_float32 b[16];
    const easysimd_float32 c[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -691.94), EASYSIMD_FLOAT32_C(   375.81), EASYSIMD_FLOAT32_C(  -524.24), EASYSIMD_FLOAT32_C(  -741.91),
        EASYSIMD_FLOAT32_C(  -213.14), EASYSIMD_FLOAT32_C(  -139.59), EASYSIMD_FLOAT32_C(   352.73), EASYSIMD_FLOAT32_C(  -694.26),
        EASYSIMD_FLOAT32_C(   -40.83), EASYSIMD_FLOAT32_C(  -216.72), EASYSIMD_FLOAT32_C(   257.07), EASYSIMD_FLOAT32_C(   948.03),
        EASYSIMD_FLOAT32_C(  -995.27), EASYSIMD_FLOAT32_C(  -370.34), EASYSIMD_FLOAT32_C(  -540.02), EASYSIMD_FLOAT32_C(   902.18) },
      UINT16_C(11903),
      { EASYSIMD_FLOAT32_C(    38.00), EASYSIMD_FLOAT32_C(   382.55), EASYSIMD_FLOAT32_C(  -715.42), EASYSIMD_FLOAT32_C(   891.00),
        EASYSIMD_FLOAT32_C(   502.16), EASYSIMD_FLOAT32_C(   898.57), EASYSIMD_FLOAT32_C(   106.26), EASYSIMD_FLOAT32_C(   766.52),
        EASYSIMD_FLOAT32_C(  -303.33), EASYSIMD_FLOAT32_C(   -79.04), EASYSIMD_FLOAT32_C(  -949.06), EASYSIMD_FLOAT32_C(  -799.36),
        EASYSIMD_FLOAT32_C(  -741.62), EASYSIMD_FLOAT32_C(  -641.00), EASYSIMD_FLOAT32_C(   576.44), EASYSIMD_FLOAT32_C(  -265.86) },
      { EASYSIMD_FLOAT32_C(  -382.91), EASYSIMD_FLOAT32_C(  -636.70), EASYSIMD_FLOAT32_C(   594.55), EASYSIMD_FLOAT32_C(   969.82),
        EASYSIMD_FLOAT32_C(  -330.96), EASYSIMD_FLOAT32_C(  -446.28), EASYSIMD_FLOAT32_C(  -246.90), EASYSIMD_FLOAT32_C(   926.11),
        EASYSIMD_FLOAT32_C(  -498.25), EASYSIMD_FLOAT32_C(  -242.17), EASYSIMD_FLOAT32_C(  -444.23), EASYSIMD_FLOAT32_C(   -38.27),
        EASYSIMD_FLOAT32_C(  -340.00), EASYSIMD_FLOAT32_C(  -158.83), EASYSIMD_FLOAT32_C(  -368.27), EASYSIMD_FLOAT32_C(   698.01) },
      { EASYSIMD_FLOAT32_C( 26676.63), EASYSIMD_FLOAT32_C(-143129.41), EASYSIMD_FLOAT32_C(-375646.31), EASYSIMD_FLOAT32_C(660072.00),
        EASYSIMD_FLOAT32_C(107361.34), EASYSIMD_FLOAT32_C(125877.66), EASYSIMD_FLOAT32_C(-37234.19), EASYSIMD_FLOAT32_C(  -694.26),
        EASYSIMD_FLOAT32_C(   -40.83), EASYSIMD_FLOAT32_C(-16887.38), EASYSIMD_FLOAT32_C(244419.09), EASYSIMD_FLOAT32_C(757855.50),
        EASYSIMD_FLOAT32_C(  -995.27), EASYSIMD_FLOAT32_C(-237229.11), EASYSIMD_FLOAT32_C(  -540.02), EASYSIMD_FLOAT32_C(   902.18) } },
    { { EASYSIMD_FLOAT32_C(  -776.28), EASYSIMD_FLOAT32_C(   -83.69), EASYSIMD_FLOAT32_C(   589.00), EASYSIMD_FLOAT32_C(   725.88),
        EASYSIMD_FLOAT32_C(  -185.11), EASYSIMD_FLOAT32_C(  -304.73), EASYSIMD_FLOAT32_C(   492.40), EASYSIMD_FLOAT32_C(   511.56),
        EASYSIMD_FLOAT32_C(   616.23), EASYSIMD_FLOAT32_C(   543.34), EASYSIMD_FLOAT32_C(   712.19), EASYSIMD_FLOAT32_C(   874.61),
        EASYSIMD_FLOAT32_C(   902.34), EASYSIMD_FLOAT32_C(   288.64), EASYSIMD_FLOAT32_C(  -391.26), EASYSIMD_FLOAT32_C(  -480.57) },
      UINT16_C( 6632),
      { EASYSIMD_FLOAT32_C(  -510.75), EASYSIMD_FLOAT32_C(  -679.01), EASYSIMD_FLOAT32_C(  -242.99), EASYSIMD_FLOAT32_C(   242.35),
        EASYSIMD_FLOAT32_C(  -752.90), EASYSIMD_FLOAT32_C(   258.76), EASYSIMD_FLOAT32_C(  -999.82), EASYSIMD_FLOAT32_C(  -197.13),
        EASYSIMD_FLOAT32_C(  -779.51), EASYSIMD_FLOAT32_C(  -339.82), EASYSIMD_FLOAT32_C(   644.05), EASYSIMD_FLOAT32_C(  -147.78),
        EASYSIMD_FLOAT32_C(  -641.81), EASYSIMD_FLOAT32_C(   867.77), EASYSIMD_FLOAT32_C(   768.54), EASYSIMD_FLOAT32_C(   947.19) },
      { EASYSIMD_FLOAT32_C(   593.65), EASYSIMD_FLOAT32_C(  -416.57), EASYSIMD_FLOAT32_C(  -357.54), EASYSIMD_FLOAT32_C(    86.06),
        EASYSIMD_FLOAT32_C(  -905.02), EASYSIMD_FLOAT32_C(  -741.31), EASYSIMD_FLOAT32_C(  -370.60), EASYSIMD_FLOAT32_C(   807.18),
        EASYSIMD_FLOAT32_C(  -866.71), EASYSIMD_FLOAT32_C(  -468.25), EASYSIMD_FLOAT32_C(    95.81), EASYSIMD_FLOAT32_C(  -257.96),
        EASYSIMD_FLOAT32_C(    51.18), EASYSIMD_FLOAT32_C(  -252.25), EASYSIMD_FLOAT32_C(   -54.67), EASYSIMD_FLOAT32_C(   540.43) },
      { EASYSIMD_FLOAT32_C(  -776.28), EASYSIMD_FLOAT32_C(   -83.69), EASYSIMD_FLOAT32_C(   589.00), EASYSIMD_FLOAT32_C(-176003.09),
        EASYSIMD_FLOAT32_C(  -185.11), EASYSIMD_FLOAT32_C( 79593.25), EASYSIMD_FLOAT32_C(492681.97), EASYSIMD_FLOAT32_C(100036.65),
        EASYSIMD_FLOAT32_C(481224.16), EASYSIMD_FLOAT32_C(   543.34), EASYSIMD_FLOAT32_C(   712.19), EASYSIMD_FLOAT32_C(129507.82),
        EASYSIMD_FLOAT32_C(579079.69), EASYSIMD_FLOAT32_C(   288.64), EASYSIMD_FLOAT32_C(  -391.26), EASYSIMD_FLOAT32_C(  -480.57) } },
    { { EASYSIMD_FLOAT32_C(    68.74), EASYSIMD_FLOAT32_C(   702.34), EASYSIMD_FLOAT32_C(  -217.21), EASYSIMD_FLOAT32_C(   315.84),
        EASYSIMD_FLOAT32_C(   -38.90), EASYSIMD_FLOAT32_C(  -217.03), EASYSIMD_FLOAT32_C(  -881.29), EASYSIMD_FLOAT32_C(   181.59),
        EASYSIMD_FLOAT32_C(   443.15), EASYSIMD_FLOAT32_C(   762.76), EASYSIMD_FLOAT32_C(  -966.18), EASYSIMD_FLOAT32_C(   801.34),
        EASYSIMD_FLOAT32_C(   630.52), EASYSIMD_FLOAT32_C(   802.35), EASYSIMD_FLOAT32_C(   748.53), EASYSIMD_FLOAT32_C(   224.18) },
      UINT16_C(10548),
      { EASYSIMD_FLOAT32_C(  -689.76), EASYSIMD_FLOAT32_C(  -519.24), EASYSIMD_FLOAT32_C(  -350.33), EASYSIMD_FLOAT32_C(   -60.36),
        EASYSIMD_FLOAT32_C(  -712.06), EASYSIMD_FLOAT32_C(  -217.04), EASYSIMD_FLOAT32_C(   471.38), EASYSIMD_FLOAT32_C(   383.75),
        EASYSIMD_FLOAT32_C(   525.00), EASYSIMD_FLOAT32_C(  -477.44), EASYSIMD_FLOAT32_C(  -868.49), EASYSIMD_FLOAT32_C(  -529.67),
        EASYSIMD_FLOAT32_C(  -937.01), EASYSIMD_FLOAT32_C(   200.25), EASYSIMD_FLOAT32_C(  -827.33), EASYSIMD_FLOAT32_C(  -154.22) },
      { EASYSIMD_FLOAT32_C(  -483.91), EASYSIMD_FLOAT32_C(   133.77), EASYSIMD_FLOAT32_C(   628.74), EASYSIMD_FLOAT32_C(  -365.21),
        EASYSIMD_FLOAT32_C(  -684.64), EASYSIMD_FLOAT32_C(    71.89), EASYSIMD_FLOAT32_C(  -602.45), EASYSIMD_FLOAT32_C(  -650.82),
        EASYSIMD_FLOAT32_C(  -126.77), EASYSIMD_FLOAT32_C(  -971.93), EASYSIMD_FLOAT32_C(  -848.47), EASYSIMD_FLOAT32_C(  -378.24),
        EASYSIMD_FLOAT32_C(   252.25), EASYSIMD_FLOAT32_C(  -462.69), EASYSIMD_FLOAT32_C(    12.74), EASYSIMD_FLOAT32_C(   562.49) },
      { EASYSIMD_FLOAT32_C(    68.74), EASYSIMD_FLOAT32_C(   702.34), EASYSIMD_FLOAT32_C(-76723.92), EASYSIMD_FLOAT32_C(   315.84),
        EASYSIMD_FLOAT32_C(-27014.49), EASYSIMD_FLOAT32_C(-47176.08), EASYSIMD_FLOAT32_C(  -881.29), EASYSIMD_FLOAT32_C(   181.59),
        EASYSIMD_FLOAT32_C(-232526.98), EASYSIMD_FLOAT32_C(   762.76), EASYSIMD_FLOAT32_C(  -966.18), EASYSIMD_FLOAT32_C(424824.00),
        EASYSIMD_FLOAT32_C(   630.52), EASYSIMD_FLOAT32_C(-160207.89), EASYSIMD_FLOAT32_C(   748.53), EASYSIMD_FLOAT32_C(   224.18) } },
    { { EASYSIMD_FLOAT32_C(    18.08), EASYSIMD_FLOAT32_C(   662.41), EASYSIMD_FLOAT32_C(  -497.87), EASYSIMD_FLOAT32_C(   306.02),
        EASYSIMD_FLOAT32_C(  -554.62), EASYSIMD_FLOAT32_C(   973.51), EASYSIMD_FLOAT32_C(  -310.23), EASYSIMD_FLOAT32_C(   970.38),
        EASYSIMD_FLOAT32_C(  -503.93), EASYSIMD_FLOAT32_C(  -178.72), EASYSIMD_FLOAT32_C(  -559.29), EASYSIMD_FLOAT32_C(  -440.93),
        EASYSIMD_FLOAT32_C(  -978.47), EASYSIMD_FLOAT32_C(  -386.62), EASYSIMD_FLOAT32_C(   404.85), EASYSIMD_FLOAT32_C(  -462.39) },
      UINT16_C(60451),
      { EASYSIMD_FLOAT32_C(   172.41), EASYSIMD_FLOAT32_C(  -937.49), EASYSIMD_FLOAT32_C(  -894.52), EASYSIMD_FLOAT32_C(   569.96),
        EASYSIMD_FLOAT32_C(  -588.31), EASYSIMD_FLOAT32_C(   -21.28), EASYSIMD_FLOAT32_C(   598.03), EASYSIMD_FLOAT32_C(  -436.78),
        EASYSIMD_FLOAT32_C(   600.47), EASYSIMD_FLOAT32_C(  -149.72), EASYSIMD_FLOAT32_C(   100.53), EASYSIMD_FLOAT32_C(  -386.78),
        EASYSIMD_FLOAT32_C(  -587.23), EASYSIMD_FLOAT32_C(  -881.39), EASYSIMD_FLOAT32_C(  -724.37), EASYSIMD_FLOAT32_C(   -85.10) },
      { EASYSIMD_FLOAT32_C(   424.63), EASYSIMD_FLOAT32_C(  -279.00), EASYSIMD_FLOAT32_C(  -111.59), EASYSIMD_FLOAT32_C(  -885.60),
        EASYSIMD_FLOAT32_C(  -308.62), EASYSIMD_FLOAT32_C(   384.49), EASYSIMD_FLOAT32_C(   -64.32), EASYSIMD_FLOAT32_C(   132.10),
        EASYSIMD_FLOAT32_C(   943.55), EASYSIMD_FLOAT32_C(   -42.79), EASYSIMD_FLOAT32_C(   745.48), EASYSIMD_FLOAT32_C(   348.40),
        EASYSIMD_FLOAT32_C(   494.82), EASYSIMD_FLOAT32_C(   492.63), EASYSIMD_FLOAT32_C(  -618.01), EASYSIMD_FLOAT32_C(  -332.77) },
      { EASYSIMD_FLOAT32_C( -3541.80), EASYSIMD_FLOAT32_C(621281.75), EASYSIMD_FLOAT32_C(  -497.87), EASYSIMD_FLOAT32_C(   306.02),
        EASYSIMD_FLOAT32_C(  -554.62), EASYSIMD_FLOAT32_C( 20331.80), EASYSIMD_FLOAT32_C(  -310.23), EASYSIMD_FLOAT32_C(   970.38),
        EASYSIMD_FLOAT32_C(  -503.93), EASYSIMD_FLOAT32_C(  -178.72), EASYSIMD_FLOAT32_C( 55479.94), EASYSIMD_FLOAT32_C(-170891.31),
        EASYSIMD_FLOAT32_C(  -978.47), EASYSIMD_FLOAT32_C(-341255.62), EASYSIMD_FLOAT32_C(293879.19), EASYSIMD_FLOAT32_C(-39016.62) } },
    { { EASYSIMD_FLOAT32_C(   555.14), EASYSIMD_FLOAT32_C(  -512.52), EASYSIMD_FLOAT32_C(  -762.82), EASYSIMD_FLOAT32_C(   966.83),
        EASYSIMD_FLOAT32_C(   466.19), EASYSIMD_FLOAT32_C(   835.21), EASYSIMD_FLOAT32_C(  -469.95), EASYSIMD_FLOAT32_C(    66.67),
        EASYSIMD_FLOAT32_C(  -314.50), EASYSIMD_FLOAT32_C(   630.58), EASYSIMD_FLOAT32_C(   679.88), EASYSIMD_FLOAT32_C(    98.27),
        EASYSIMD_FLOAT32_C(   749.19), EASYSIMD_FLOAT32_C(   955.51), EASYSIMD_FLOAT32_C(  -986.83), EASYSIMD_FLOAT32_C(   173.82) },
      UINT16_C(13236),
      { EASYSIMD_FLOAT32_C(   288.22), EASYSIMD_FLOAT32_C(   367.90), EASYSIMD_FLOAT32_C(  -713.93), EASYSIMD_FLOAT32_C(  -776.10),
        EASYSIMD_FLOAT32_C(  -500.01), EASYSIMD_FLOAT32_C(  -770.37), EASYSIMD_FLOAT32_C(   181.11), EASYSIMD_FLOAT32_C(  -754.53),
        EASYSIMD_FLOAT32_C(   578.03), EASYSIMD_FLOAT32_C(  -324.07), EASYSIMD_FLOAT32_C(   738.10), EASYSIMD_FLOAT32_C(   960.02),
        EASYSIMD_FLOAT32_C(   343.15), EASYSIMD_FLOAT32_C(   293.24), EASYSIMD_FLOAT32_C(  -552.50), EASYSIMD_FLOAT32_C(   580.33) },
      { EASYSIMD_FLOAT32_C(   260.07), EASYSIMD_FLOAT32_C(   913.69), EASYSIMD_FLOAT32_C(   415.55), EASYSIMD_FLOAT32_C(   790.11),
        EASYSIMD_FLOAT32_C(   -19.65), EASYSIMD_FLOAT32_C(  -898.95), EASYSIMD_FLOAT32_C(   420.69), EASYSIMD_FLOAT32_C(  -339.77),
        EASYSIMD_FLOAT32_C(   199.32), EASYSIMD_FLOAT32_C(   169.88), EASYSIMD_FLOAT32_C(  -384.26), EASYSIMD_FLOAT32_C(   212.49),
        EASYSIMD_FLOAT32_C(  -656.30), EASYSIMD_FLOAT32_C(   292.26), EASYSIMD_FLOAT32_C(  -885.92), EASYSIMD_FLOAT32_C(   631.92) },
      { EASYSIMD_FLOAT32_C(   555.14), EASYSIMD_FLOAT32_C(  -512.52), EASYSIMD_FLOAT32_C(-545015.62), EASYSIMD_FLOAT32_C(   966.83),
        EASYSIMD_FLOAT32_C(233119.33), EASYSIMD_FLOAT32_C(644319.69), EASYSIMD_FLOAT32_C(  -469.95), EASYSIMD_FLOAT32_C( 50644.29),
        EASYSIMD_FLOAT32_C(181591.12), EASYSIMD_FLOAT32_C(204182.20), EASYSIMD_FLOAT32_C(   679.88), EASYSIMD_FLOAT32_C(    98.27),
        EASYSIMD_FLOAT32_C(-256428.25), EASYSIMD_FLOAT32_C(-280486.00), EASYSIMD_FLOAT32_C(  -986.83), EASYSIMD_FLOAT32_C(   173.82) } },
    { { EASYSIMD_FLOAT32_C(  -339.85), EASYSIMD_FLOAT32_C(  -599.84), EASYSIMD_FLOAT32_C(   855.82), EASYSIMD_FLOAT32_C(   160.15),
        EASYSIMD_FLOAT32_C(  -370.22), EASYSIMD_FLOAT32_C(    36.92), EASYSIMD_FLOAT32_C(   405.62), EASYSIMD_FLOAT32_C(  -792.19),
        EASYSIMD_FLOAT32_C(   712.85), EASYSIMD_FLOAT32_C(   143.72), EASYSIMD_FLOAT32_C(  -832.17), EASYSIMD_FLOAT32_C(    56.00),
        EASYSIMD_FLOAT32_C(  -563.04), EASYSIMD_FLOAT32_C(  -384.67), EASYSIMD_FLOAT32_C(  -363.67), EASYSIMD_FLOAT32_C(   697.02) },
      UINT16_C(10970),
      { EASYSIMD_FLOAT32_C(   487.13), EASYSIMD_FLOAT32_C(   509.36), EASYSIMD_FLOAT32_C(  -847.07), EASYSIMD_FLOAT32_C(   -92.18),
        EASYSIMD_FLOAT32_C(  -830.40), EASYSIMD_FLOAT32_C(   352.25), EASYSIMD_FLOAT32_C(  -922.30), EASYSIMD_FLOAT32_C(  -214.66),
        EASYSIMD_FLOAT32_C(  -435.26), EASYSIMD_FLOAT32_C(  -578.60), EASYSIMD_FLOAT32_C(  -922.40), EASYSIMD_FLOAT32_C(  -321.17),
        EASYSIMD_FLOAT32_C(  -946.68), EASYSIMD_FLOAT32_C(  -262.25), EASYSIMD_FLOAT32_C(    78.98), EASYSIMD_FLOAT32_C(   909.14) },
      { EASYSIMD_FLOAT32_C(   897.90), EASYSIMD_FLOAT32_C(   708.77), EASYSIMD_FLOAT32_C(   -53.94), EASYSIMD_FLOAT32_C(   303.52),
        EASYSIMD_FLOAT32_C(   916.58), EASYSIMD_FLOAT32_C(  -341.09), EASYSIMD_FLOAT32_C(  -552.76), EASYSIMD_FLOAT32_C(  -915.59),
        EASYSIMD_FLOAT32_C(   714.91), EASYSIMD_FLOAT32_C(  -115.80), EASYSIMD_FLOAT32_C(  -300.27), EASYSIMD_FLOAT32_C(  -648.75),
        EASYSIMD_FLOAT32_C(  -418.78), EASYSIMD_FLOAT32_C(   228.74), EASYSIMD_FLOAT32_C(  -596.87), EASYSIMD_FLOAT32_C(  -931.65) },
      { EASYSIMD_FLOAT32_C(  -339.85), EASYSIMD_FLOAT32_C(304825.72), EASYSIMD_FLOAT32_C(   855.82), EASYSIMD_FLOAT32_C( 14459.11),
        EASYSIMD_FLOAT32_C(-308347.28), EASYSIMD_FLOAT32_C(    36.92), EASYSIMD_FLOAT32_C(374656.06), EASYSIMD_FLOAT32_C(-169135.92),
        EASYSIMD_FLOAT32_C(   712.85), EASYSIMD_FLOAT32_C( 83272.19), EASYSIMD_FLOAT32_C(  -832.17), EASYSIMD_FLOAT32_C( 18634.27),
        EASYSIMD_FLOAT32_C(  -563.04), EASYSIMD_FLOAT32_C(-101108.45), EASYSIMD_FLOAT32_C(  -363.67), EASYSIMD_FLOAT32_C(   697.02) } },
    { { EASYSIMD_FLOAT32_C(  -261.89), EASYSIMD_FLOAT32_C(  -443.94), EASYSIMD_FLOAT32_C(   -23.83), EASYSIMD_FLOAT32_C(   -92.29),
        EASYSIMD_FLOAT32_C(   908.31), EASYSIMD_FLOAT32_C(    53.88), EASYSIMD_FLOAT32_C(   693.05), EASYSIMD_FLOAT32_C(  -526.95),
        EASYSIMD_FLOAT32_C(   475.28), EASYSIMD_FLOAT32_C(   770.65), EASYSIMD_FLOAT32_C(   151.88), EASYSIMD_FLOAT32_C(   528.60),
        EASYSIMD_FLOAT32_C(  -491.60), EASYSIMD_FLOAT32_C(  -769.14), EASYSIMD_FLOAT32_C(   437.74), EASYSIMD_FLOAT32_C(  -593.70) },
      UINT16_C(52097),
      { EASYSIMD_FLOAT32_C(   709.82), EASYSIMD_FLOAT32_C(   856.21), EASYSIMD_FLOAT32_C(    42.71), EASYSIMD_FLOAT32_C(  -842.94),
        EASYSIMD_FLOAT32_C(   940.61), EASYSIMD_FLOAT32_C(  -242.38), EASYSIMD_FLOAT32_C(    41.26), EASYSIMD_FLOAT32_C(  -359.65),
        EASYSIMD_FLOAT32_C(   108.87), EASYSIMD_FLOAT32_C(   622.47), EASYSIMD_FLOAT32_C(   869.09), EASYSIMD_FLOAT32_C(   512.00),
        EASYSIMD_FLOAT32_C(   690.82), EASYSIMD_FLOAT32_C(  -392.80), EASYSIMD_FLOAT32_C(  -931.95), EASYSIMD_FLOAT32_C(  -333.00) },
      { EASYSIMD_FLOAT32_C(   514.91), EASYSIMD_FLOAT32_C(   976.36), EASYSIMD_FLOAT32_C(   720.88), EASYSIMD_FLOAT32_C(   207.96),
        EASYSIMD_FLOAT32_C(  -550.59), EASYSIMD_FLOAT32_C(   196.16), EASYSIMD_FLOAT32_C(   -21.40), EASYSIMD_FLOAT32_C(   601.29),
        EASYSIMD_FLOAT32_C(  -275.24), EASYSIMD_FLOAT32_C(   487.00), EASYSIMD_FLOAT32_C(   832.16), EASYSIMD_FLOAT32_C(  -837.50),
        EASYSIMD_FLOAT32_C(   893.31), EASYSIMD_FLOAT32_C(   771.79), EASYSIMD_FLOAT32_C(  -453.70), EASYSIMD_FLOAT32_C(   603.13) },
      { EASYSIMD_FLOAT32_C(185379.86), EASYSIMD_FLOAT32_C(  -443.94), EASYSIMD_FLOAT32_C(   -23.83), EASYSIMD_FLOAT32_C(   -92.29),
        EASYSIMD_FLOAT32_C(   908.31), EASYSIMD_FLOAT32_C(    53.88), EASYSIMD_FLOAT32_C(   693.05), EASYSIMD_FLOAT32_C(-190118.86),
        EASYSIMD_FLOAT32_C(-51468.50), EASYSIMD_FLOAT32_C(-480193.50), EASYSIMD_FLOAT32_C(   151.88), EASYSIMD_FLOAT32_C(-269805.69),
        EASYSIMD_FLOAT32_C(  -491.60), EASYSIMD_FLOAT32_C(  -769.14), EASYSIMD_FLOAT32_C(408405.47), EASYSIMD_FLOAT32_C(-198305.23) } },
    { { EASYSIMD_FLOAT32_C(   627.99), EASYSIMD_FLOAT32_C(   589.01), EASYSIMD_FLOAT32_C(   760.19), EASYSIMD_FLOAT32_C(   568.61),
        EASYSIMD_FLOAT32_C(  -653.37), EASYSIMD_FLOAT32_C(  -198.55), EASYSIMD_FLOAT32_C(  -791.05), EASYSIMD_FLOAT32_C(   455.49),
        EASYSIMD_FLOAT32_C(  -576.07), EASYSIMD_FLOAT32_C(  -921.96), EASYSIMD_FLOAT32_C(   -32.51), EASYSIMD_FLOAT32_C(  -885.25),
        EASYSIMD_FLOAT32_C(  -314.76), EASYSIMD_FLOAT32_C(    35.54), EASYSIMD_FLOAT32_C(  -218.25), EASYSIMD_FLOAT32_C(  -799.85) },
      UINT16_C( 9083),
      { EASYSIMD_FLOAT32_C(   408.10), EASYSIMD_FLOAT32_C(   461.32), EASYSIMD_FLOAT32_C(   698.79), EASYSIMD_FLOAT32_C(  -613.30),
        EASYSIMD_FLOAT32_C(    62.62), EASYSIMD_FLOAT32_C(  -576.45), EASYSIMD_FLOAT32_C(   873.71), EASYSIMD_FLOAT32_C(  -105.23),
        EASYSIMD_FLOAT32_C(  -413.95), EASYSIMD_FLOAT32_C(   767.01), EASYSIMD_FLOAT32_C(  -333.44), EASYSIMD_FLOAT32_C(   132.34),
        EASYSIMD_FLOAT32_C(   370.14), EASYSIMD_FLOAT32_C(  -705.45), EASYSIMD_FLOAT32_C(  -278.65), EASYSIMD_FLOAT32_C(   130.34) },
      { EASYSIMD_FLOAT32_C(   863.16), EASYSIMD_FLOAT32_C(    67.98), EASYSIMD_FLOAT32_C(   931.79), EASYSIMD_FLOAT32_C(  -927.89),
        EASYSIMD_FLOAT32_C(  -476.53), EASYSIMD_FLOAT32_C(  -644.28), EASYSIMD_FLOAT32_C(  -849.85), EASYSIMD_FLOAT32_C(   490.96),
        EASYSIMD_FLOAT32_C(  -529.53), EASYSIMD_FLOAT32_C(  -164.61), EASYSIMD_FLOAT32_C(  -473.49), EASYSIMD_FLOAT32_C(   252.22),
        EASYSIMD_FLOAT32_C(    35.53), EASYSIMD_FLOAT32_C(   538.42), EASYSIMD_FLOAT32_C(   754.84), EASYSIMD_FLOAT32_C(  -556.37) },
      { EASYSIMD_FLOAT32_C(-257145.88), EASYSIMD_FLOAT32_C(-271790.06), EASYSIMD_FLOAT32_C(   760.19), EASYSIMD_FLOAT32_C(349656.38),
        EASYSIMD_FLOAT32_C( 41390.56), EASYSIMD_FLOAT32_C(-113809.87), EASYSIMD_FLOAT32_C(691998.19), EASYSIMD_FLOAT32_C(   455.49),
        EASYSIMD_FLOAT32_C(-237934.66), EASYSIMD_FLOAT32_C(707317.19), EASYSIMD_FLOAT32_C(   -32.51), EASYSIMD_FLOAT32_C(  -885.25),
        EASYSIMD_FLOAT32_C(  -314.76), EASYSIMD_FLOAT32_C( 24533.27), EASYSIMD_FLOAT32_C(  -218.25), EASYSIMD_FLOAT32_C(  -799.85) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 c = easysimd_mm512_loadu_ps(test_vec[i].c);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_fnmsub_ps(a, test_vec[i].k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_fnmsub_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 c = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_mask_fnmsub_ps(a, k, b, c);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_fnmsub_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const easysimd__mmask16 k;
    const easysimd_float32 b[16];
    const easysimd_float32 c[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    44.99), EASYSIMD_FLOAT32_C(   351.65), EASYSIMD_FLOAT32_C(  -345.68), EASYSIMD_FLOAT32_C(   479.09),
        EASYSIMD_FLOAT32_C(  -936.06), EASYSIMD_FLOAT32_C(   606.33), EASYSIMD_FLOAT32_C(  -629.45), EASYSIMD_FLOAT32_C(   420.61),
        EASYSIMD_FLOAT32_C(    89.97), EASYSIMD_FLOAT32_C(  -260.87), EASYSIMD_FLOAT32_C(   503.11), EASYSIMD_FLOAT32_C(  -233.25),
        EASYSIMD_FLOAT32_C(  -111.14), EASYSIMD_FLOAT32_C(  -270.94), EASYSIMD_FLOAT32_C(   597.04), EASYSIMD_FLOAT32_C(   818.28) },
      UINT16_C(10801),
      { EASYSIMD_FLOAT32_C(  -623.87), EASYSIMD_FLOAT32_C(   391.29), EASYSIMD_FLOAT32_C(  -974.19), EASYSIMD_FLOAT32_C(   508.70),
        EASYSIMD_FLOAT32_C(  -696.93), EASYSIMD_FLOAT32_C(  -731.69), EASYSIMD_FLOAT32_C(  -897.64), EASYSIMD_FLOAT32_C(  -301.77),
        EASYSIMD_FLOAT32_C(    37.56), EASYSIMD_FLOAT32_C(   817.41), EASYSIMD_FLOAT32_C(   106.41), EASYSIMD_FLOAT32_C(  -810.21),
        EASYSIMD_FLOAT32_C(  -432.99), EASYSIMD_FLOAT32_C(  -848.61), EASYSIMD_FLOAT32_C(   541.44), EASYSIMD_FLOAT32_C(   221.34) },
      { EASYSIMD_FLOAT32_C(   630.49), EASYSIMD_FLOAT32_C(   605.38), EASYSIMD_FLOAT32_C(  -172.34), EASYSIMD_FLOAT32_C(  -998.96),
        EASYSIMD_FLOAT32_C(    25.99), EASYSIMD_FLOAT32_C(   917.64), EASYSIMD_FLOAT32_C(  -259.83), EASYSIMD_FLOAT32_C(  -470.90),
        EASYSIMD_FLOAT32_C(  -315.62), EASYSIMD_FLOAT32_C(   629.03), EASYSIMD_FLOAT32_C(   258.16), EASYSIMD_FLOAT32_C(  -718.58),
        EASYSIMD_FLOAT32_C(   447.31), EASYSIMD_FLOAT32_C(   965.94), EASYSIMD_FLOAT32_C(   291.03), EASYSIMD_FLOAT32_C(   823.44) },
      { EASYSIMD_FLOAT32_C( 27437.42), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-652394.31), EASYSIMD_FLOAT32_C(442728.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(212608.70), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-188262.91),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-230888.33), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   357.23), EASYSIMD_FLOAT32_C(   316.85), EASYSIMD_FLOAT32_C(   332.14), EASYSIMD_FLOAT32_C(   660.30),
        EASYSIMD_FLOAT32_C(   585.16), EASYSIMD_FLOAT32_C(   434.49), EASYSIMD_FLOAT32_C(  -641.47), EASYSIMD_FLOAT32_C(  -377.28),
        EASYSIMD_FLOAT32_C(   251.91), EASYSIMD_FLOAT32_C(   464.94), EASYSIMD_FLOAT32_C(  -187.49), EASYSIMD_FLOAT32_C(   818.92),
        EASYSIMD_FLOAT32_C(   616.33), EASYSIMD_FLOAT32_C(  -646.05), EASYSIMD_FLOAT32_C(    40.25), EASYSIMD_FLOAT32_C(   246.82) },
      UINT16_C( 8571),
      { EASYSIMD_FLOAT32_C(   247.86), EASYSIMD_FLOAT32_C(   -14.68), EASYSIMD_FLOAT32_C(   785.55), EASYSIMD_FLOAT32_C(   988.04),
        EASYSIMD_FLOAT32_C(   514.42), EASYSIMD_FLOAT32_C(  -530.06), EASYSIMD_FLOAT32_C(   617.07), EASYSIMD_FLOAT32_C(  -227.43),
        EASYSIMD_FLOAT32_C(  -248.64), EASYSIMD_FLOAT32_C(    64.37), EASYSIMD_FLOAT32_C(  -261.49), EASYSIMD_FLOAT32_C(  -957.61),
        EASYSIMD_FLOAT32_C(  -112.19), EASYSIMD_FLOAT32_C(  -904.26), EASYSIMD_FLOAT32_C(   359.24), EASYSIMD_FLOAT32_C(  -780.05) },
      { EASYSIMD_FLOAT32_C(   756.04), EASYSIMD_FLOAT32_C(   -55.61), EASYSIMD_FLOAT32_C(   654.44), EASYSIMD_FLOAT32_C(  -885.43),
        EASYSIMD_FLOAT32_C(   567.11), EASYSIMD_FLOAT32_C(   -93.65), EASYSIMD_FLOAT32_C(   579.51), EASYSIMD_FLOAT32_C(  -620.38),
        EASYSIMD_FLOAT32_C(  -274.74), EASYSIMD_FLOAT32_C(   195.84), EASYSIMD_FLOAT32_C(  -266.44), EASYSIMD_FLOAT32_C(   765.52),
        EASYSIMD_FLOAT32_C(  -557.33), EASYSIMD_FLOAT32_C(  -307.11), EASYSIMD_FLOAT32_C(   633.44), EASYSIMD_FLOAT32_C(   690.53) },
      { EASYSIMD_FLOAT32_C(-89299.07), EASYSIMD_FLOAT32_C(  4706.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-651517.38),
        EASYSIMD_FLOAT32_C(-301585.09), EASYSIMD_FLOAT32_C(230399.42), EASYSIMD_FLOAT32_C(395252.38), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C( 62909.64), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-583890.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   678.21), EASYSIMD_FLOAT32_C(   418.99), EASYSIMD_FLOAT32_C(   678.56), EASYSIMD_FLOAT32_C(   192.63),
        EASYSIMD_FLOAT32_C(   888.92), EASYSIMD_FLOAT32_C(   295.63), EASYSIMD_FLOAT32_C(   965.21), EASYSIMD_FLOAT32_C(  -359.72),
        EASYSIMD_FLOAT32_C(  -640.00), EASYSIMD_FLOAT32_C(  -296.28), EASYSIMD_FLOAT32_C(  -317.33), EASYSIMD_FLOAT32_C(   247.81),
        EASYSIMD_FLOAT32_C(  -200.54), EASYSIMD_FLOAT32_C(  -958.09), EASYSIMD_FLOAT32_C(   467.76), EASYSIMD_FLOAT32_C(  -444.49) },
      UINT16_C(62097),
      { EASYSIMD_FLOAT32_C(  -329.92), EASYSIMD_FLOAT32_C(  -446.59), EASYSIMD_FLOAT32_C(  -971.46), EASYSIMD_FLOAT32_C(  -750.41),
        EASYSIMD_FLOAT32_C(   -66.97), EASYSIMD_FLOAT32_C(  -246.19), EASYSIMD_FLOAT32_C(   445.43), EASYSIMD_FLOAT32_C(   666.59),
        EASYSIMD_FLOAT32_C(  -480.68), EASYSIMD_FLOAT32_C(   888.10), EASYSIMD_FLOAT32_C(  -640.51), EASYSIMD_FLOAT32_C(  -847.24),
        EASYSIMD_FLOAT32_C(   578.63), EASYSIMD_FLOAT32_C(  -962.30), EASYSIMD_FLOAT32_C(   571.75), EASYSIMD_FLOAT32_C(   257.19) },
      { EASYSIMD_FLOAT32_C(   230.33), EASYSIMD_FLOAT32_C(   460.67), EASYSIMD_FLOAT32_C(  -447.18), EASYSIMD_FLOAT32_C(   195.54),
        EASYSIMD_FLOAT32_C(  -899.04), EASYSIMD_FLOAT32_C(   -87.17), EASYSIMD_FLOAT32_C(   899.25), EASYSIMD_FLOAT32_C(  -216.37),
        EASYSIMD_FLOAT32_C(  -839.36), EASYSIMD_FLOAT32_C(  -301.29), EASYSIMD_FLOAT32_C(  -174.46), EASYSIMD_FLOAT32_C(   628.40),
        EASYSIMD_FLOAT32_C(   254.22), EASYSIMD_FLOAT32_C(   811.84), EASYSIMD_FLOAT32_C(  -249.40), EASYSIMD_FLOAT32_C(   924.30) },
      { EASYSIMD_FLOAT32_C(223524.73), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C( 60430.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(240002.14),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(263427.53), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(115784.24), EASYSIMD_FLOAT32_C(-922781.81), EASYSIMD_FLOAT32_C(-267192.38), EASYSIMD_FLOAT32_C(113394.09) } },
    { { EASYSIMD_FLOAT32_C(  -634.75), EASYSIMD_FLOAT32_C(  -220.86), EASYSIMD_FLOAT32_C(  -826.11), EASYSIMD_FLOAT32_C(   298.28),
        EASYSIMD_FLOAT32_C(   532.95), EASYSIMD_FLOAT32_C(   619.32), EASYSIMD_FLOAT32_C(   -35.13), EASYSIMD_FLOAT32_C(  -947.73),
        EASYSIMD_FLOAT32_C(   507.42), EASYSIMD_FLOAT32_C(   324.36), EASYSIMD_FLOAT32_C(  -794.97), EASYSIMD_FLOAT32_C(    86.05),
        EASYSIMD_FLOAT32_C(   362.06), EASYSIMD_FLOAT32_C(   776.78), EASYSIMD_FLOAT32_C(  -656.76), EASYSIMD_FLOAT32_C(  -407.61) },
      UINT16_C(42868),
      { EASYSIMD_FLOAT32_C(   787.92), EASYSIMD_FLOAT32_C(   338.41), EASYSIMD_FLOAT32_C(   808.89), EASYSIMD_FLOAT32_C(   687.17),
        EASYSIMD_FLOAT32_C(  -877.96), EASYSIMD_FLOAT32_C(   969.53), EASYSIMD_FLOAT32_C(  -614.11), EASYSIMD_FLOAT32_C(   -52.43),
        EASYSIMD_FLOAT32_C(   597.93), EASYSIMD_FLOAT32_C(   640.11), EASYSIMD_FLOAT32_C(  -240.59), EASYSIMD_FLOAT32_C(  -651.48),
        EASYSIMD_FLOAT32_C(   564.41), EASYSIMD_FLOAT32_C(   124.66), EASYSIMD_FLOAT32_C(   127.66), EASYSIMD_FLOAT32_C(   738.30) },
      { EASYSIMD_FLOAT32_C(  -577.07), EASYSIMD_FLOAT32_C(  -339.39), EASYSIMD_FLOAT32_C(   357.62), EASYSIMD_FLOAT32_C(   387.80),
        EASYSIMD_FLOAT32_C(  -287.12), EASYSIMD_FLOAT32_C(  -134.95), EASYSIMD_FLOAT32_C(  -287.84), EASYSIMD_FLOAT32_C(   -82.09),
        EASYSIMD_FLOAT32_C(   951.10), EASYSIMD_FLOAT32_C(  -925.79), EASYSIMD_FLOAT32_C(  -305.31), EASYSIMD_FLOAT32_C(  -705.66),
        EASYSIMD_FLOAT32_C(  -333.40), EASYSIMD_FLOAT32_C(   932.14), EASYSIMD_FLOAT32_C(   190.41), EASYSIMD_FLOAT32_C(  -545.48) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(667874.50), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(468195.94), EASYSIMD_FLOAT32_C(-600314.44), EASYSIMD_FLOAT32_C(-21285.84), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-304352.75), EASYSIMD_FLOAT32_C(-206700.27), EASYSIMD_FLOAT32_C(-190956.52), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-97765.54), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(301483.91) } },
    { { EASYSIMD_FLOAT32_C(   270.55), EASYSIMD_FLOAT32_C(    -0.70), EASYSIMD_FLOAT32_C(  -858.31), EASYSIMD_FLOAT32_C(   392.58),
        EASYSIMD_FLOAT32_C(   -31.17), EASYSIMD_FLOAT32_C(  -472.42), EASYSIMD_FLOAT32_C(  -659.85), EASYSIMD_FLOAT32_C(  -433.25),
        EASYSIMD_FLOAT32_C(  -832.31), EASYSIMD_FLOAT32_C(    99.56), EASYSIMD_FLOAT32_C(   -84.72), EASYSIMD_FLOAT32_C(   732.10),
        EASYSIMD_FLOAT32_C(  -775.78), EASYSIMD_FLOAT32_C(  -957.06), EASYSIMD_FLOAT32_C(   470.39), EASYSIMD_FLOAT32_C(  -352.85) },
      UINT16_C(30727),
      { EASYSIMD_FLOAT32_C(  -965.05), EASYSIMD_FLOAT32_C(   416.42), EASYSIMD_FLOAT32_C(   693.06), EASYSIMD_FLOAT32_C(  -252.89),
        EASYSIMD_FLOAT32_C(  -665.67), EASYSIMD_FLOAT32_C(   644.16), EASYSIMD_FLOAT32_C(  -178.68), EASYSIMD_FLOAT32_C(    29.02),
        EASYSIMD_FLOAT32_C(   938.50), EASYSIMD_FLOAT32_C(   487.92), EASYSIMD_FLOAT32_C(   -38.84), EASYSIMD_FLOAT32_C(   128.91),
        EASYSIMD_FLOAT32_C(   942.43), EASYSIMD_FLOAT32_C(  -768.29), EASYSIMD_FLOAT32_C(  -871.79), EASYSIMD_FLOAT32_C(  -915.87) },
      { EASYSIMD_FLOAT32_C(   624.29), EASYSIMD_FLOAT32_C(    97.04), EASYSIMD_FLOAT32_C(  -388.29), EASYSIMD_FLOAT32_C(   964.44),
        EASYSIMD_FLOAT32_C(   663.79), EASYSIMD_FLOAT32_C(  -220.60), EASYSIMD_FLOAT32_C(    64.00), EASYSIMD_FLOAT32_C(  -420.94),
        EASYSIMD_FLOAT32_C(  -488.50), EASYSIMD_FLOAT32_C(   288.22), EASYSIMD_FLOAT32_C(  -378.00), EASYSIMD_FLOAT32_C(   981.89),
        EASYSIMD_FLOAT32_C(   935.37), EASYSIMD_FLOAT32_C(   325.54), EASYSIMD_FLOAT32_C(  -190.09), EASYSIMD_FLOAT32_C(   970.32) },
      { EASYSIMD_FLOAT32_C(260469.97), EASYSIMD_FLOAT32_C(   194.45), EASYSIMD_FLOAT32_C(595248.62), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-95356.90),
        EASYSIMD_FLOAT32_C(730183.00), EASYSIMD_FLOAT32_C(-735625.19), EASYSIMD_FLOAT32_C(410271.41), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -258.03), EASYSIMD_FLOAT32_C(  -497.03), EASYSIMD_FLOAT32_C(  -282.58), EASYSIMD_FLOAT32_C(    76.30),
        EASYSIMD_FLOAT32_C(  -852.87), EASYSIMD_FLOAT32_C(   538.74), EASYSIMD_FLOAT32_C(  -894.68), EASYSIMD_FLOAT32_C(  -914.37),
        EASYSIMD_FLOAT32_C(    26.66), EASYSIMD_FLOAT32_C(    66.47), EASYSIMD_FLOAT32_C(   214.54), EASYSIMD_FLOAT32_C(   -30.91),
        EASYSIMD_FLOAT32_C(   298.18), EASYSIMD_FLOAT32_C(   342.75), EASYSIMD_FLOAT32_C(    53.22), EASYSIMD_FLOAT32_C(   -77.53) },
      UINT16_C(56105),
      { EASYSIMD_FLOAT32_C(  -113.09), EASYSIMD_FLOAT32_C(  -896.43), EASYSIMD_FLOAT32_C(  -555.67), EASYSIMD_FLOAT32_C(   950.91),
        EASYSIMD_FLOAT32_C(  -317.36), EASYSIMD_FLOAT32_C(   -44.18), EASYSIMD_FLOAT32_C(   239.13), EASYSIMD_FLOAT32_C(   304.64),
        EASYSIMD_FLOAT32_C(   -62.29), EASYSIMD_FLOAT32_C(   174.50), EASYSIMD_FLOAT32_C(  -369.82), EASYSIMD_FLOAT32_C(   747.62),
        EASYSIMD_FLOAT32_C(   144.82), EASYSIMD_FLOAT32_C(   372.15), EASYSIMD_FLOAT32_C(  -749.41), EASYSIMD_FLOAT32_C(   862.24) },
      { EASYSIMD_FLOAT32_C(  -551.55), EASYSIMD_FLOAT32_C(  -602.28), EASYSIMD_FLOAT32_C(   400.98), EASYSIMD_FLOAT32_C(  -446.23),
        EASYSIMD_FLOAT32_C(  -516.65), EASYSIMD_FLOAT32_C(  -572.36), EASYSIMD_FLOAT32_C(   620.24), EASYSIMD_FLOAT32_C(   697.89),
        EASYSIMD_FLOAT32_C(   396.73), EASYSIMD_FLOAT32_C(   -81.58), EASYSIMD_FLOAT32_C(    40.64), EASYSIMD_FLOAT32_C(  -550.05),
        EASYSIMD_FLOAT32_C(   840.89), EASYSIMD_FLOAT32_C(   480.42), EASYSIMD_FLOAT32_C(  -885.13), EASYSIMD_FLOAT32_C(  -272.20) },
      { EASYSIMD_FLOAT32_C(-28629.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-72108.21),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 24373.89), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  1263.92), EASYSIMD_FLOAT32_C(-11517.44), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 23658.98),
        EASYSIMD_FLOAT32_C(-44023.32), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 40768.73), EASYSIMD_FLOAT32_C( 67121.67) } },
    { { EASYSIMD_FLOAT32_C(   583.99), EASYSIMD_FLOAT32_C(  -440.80), EASYSIMD_FLOAT32_C(  -321.29), EASYSIMD_FLOAT32_C(  -733.37),
        EASYSIMD_FLOAT32_C(   515.02), EASYSIMD_FLOAT32_C(   917.84), EASYSIMD_FLOAT32_C(   571.27), EASYSIMD_FLOAT32_C(  -547.26),
        EASYSIMD_FLOAT32_C(    92.34), EASYSIMD_FLOAT32_C(  -798.55), EASYSIMD_FLOAT32_C(  -799.64), EASYSIMD_FLOAT32_C(  -762.84),
        EASYSIMD_FLOAT32_C(   573.60), EASYSIMD_FLOAT32_C(  -549.05), EASYSIMD_FLOAT32_C(  -900.60), EASYSIMD_FLOAT32_C(  -977.95) },
      UINT16_C(38526),
      { EASYSIMD_FLOAT32_C(  -424.18), EASYSIMD_FLOAT32_C(   332.01), EASYSIMD_FLOAT32_C(   928.01), EASYSIMD_FLOAT32_C(  -803.94),
        EASYSIMD_FLOAT32_C(    29.90), EASYSIMD_FLOAT32_C(   324.74), EASYSIMD_FLOAT32_C(   114.48), EASYSIMD_FLOAT32_C(  -929.47),
        EASYSIMD_FLOAT32_C(   774.68), EASYSIMD_FLOAT32_C(   -44.64), EASYSIMD_FLOAT32_C(   550.95), EASYSIMD_FLOAT32_C(   889.56),
        EASYSIMD_FLOAT32_C(   683.16), EASYSIMD_FLOAT32_C(   134.95), EASYSIMD_FLOAT32_C(  -551.24), EASYSIMD_FLOAT32_C(  -638.13) },
      { EASYSIMD_FLOAT32_C(   401.58), EASYSIMD_FLOAT32_C(   963.78), EASYSIMD_FLOAT32_C(  -720.29), EASYSIMD_FLOAT32_C(   -27.15),
        EASYSIMD_FLOAT32_C(  -583.48), EASYSIMD_FLOAT32_C(   372.06), EASYSIMD_FLOAT32_C(   174.30), EASYSIMD_FLOAT32_C(  -383.12),
        EASYSIMD_FLOAT32_C(   609.22), EASYSIMD_FLOAT32_C(  -252.10), EASYSIMD_FLOAT32_C(    67.82), EASYSIMD_FLOAT32_C(   708.61),
        EASYSIMD_FLOAT32_C(  -230.04), EASYSIMD_FLOAT32_C(   916.48), EASYSIMD_FLOAT32_C(   208.99), EASYSIMD_FLOAT32_C(   345.77) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(145386.23), EASYSIMD_FLOAT32_C(298880.62), EASYSIMD_FLOAT32_C(-589558.38),
        EASYSIMD_FLOAT32_C(-14815.62), EASYSIMD_FLOAT32_C(-298431.44), EASYSIMD_FLOAT32_C(-65573.29), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-35395.17), EASYSIMD_FLOAT32_C(440493.88), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-391630.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-624405.00) } },
    { { EASYSIMD_FLOAT32_C(   248.49), EASYSIMD_FLOAT32_C(   137.00), EASYSIMD_FLOAT32_C(   541.83), EASYSIMD_FLOAT32_C(  -721.61),
        EASYSIMD_FLOAT32_C(  -538.26), EASYSIMD_FLOAT32_C(  -343.69), EASYSIMD_FLOAT32_C(  -651.08), EASYSIMD_FLOAT32_C(  -763.58),
        EASYSIMD_FLOAT32_C(   611.67), EASYSIMD_FLOAT32_C(   899.87), EASYSIMD_FLOAT32_C(  -874.02), EASYSIMD_FLOAT32_C(   294.83),
        EASYSIMD_FLOAT32_C(    34.82), EASYSIMD_FLOAT32_C(  -425.26), EASYSIMD_FLOAT32_C(   656.70), EASYSIMD_FLOAT32_C(  -563.60) },
      UINT16_C(63278),
      { EASYSIMD_FLOAT32_C(   409.24), EASYSIMD_FLOAT32_C(   -44.96), EASYSIMD_FLOAT32_C(   308.47), EASYSIMD_FLOAT32_C(  -416.46),
        EASYSIMD_FLOAT32_C(   571.92), EASYSIMD_FLOAT32_C(   -82.31), EASYSIMD_FLOAT32_C(   331.45), EASYSIMD_FLOAT32_C(  -360.26),
        EASYSIMD_FLOAT32_C(  -373.70), EASYSIMD_FLOAT32_C(  -898.60), EASYSIMD_FLOAT32_C(  -443.78), EASYSIMD_FLOAT32_C(   835.29),
        EASYSIMD_FLOAT32_C(   447.17), EASYSIMD_FLOAT32_C(   804.72), EASYSIMD_FLOAT32_C(   -27.71), EASYSIMD_FLOAT32_C(   -11.00) },
      { EASYSIMD_FLOAT32_C(  -916.89), EASYSIMD_FLOAT32_C(   434.03), EASYSIMD_FLOAT32_C(   645.31), EASYSIMD_FLOAT32_C(  -567.97),
        EASYSIMD_FLOAT32_C(   670.45), EASYSIMD_FLOAT32_C(   256.98), EASYSIMD_FLOAT32_C(  -668.09), EASYSIMD_FLOAT32_C(   796.43),
        EASYSIMD_FLOAT32_C(  -448.19), EASYSIMD_FLOAT32_C(   366.73), EASYSIMD_FLOAT32_C(  -628.83), EASYSIMD_FLOAT32_C(  -791.49),
        EASYSIMD_FLOAT32_C(   803.12), EASYSIMD_FLOAT32_C(   -90.31), EASYSIMD_FLOAT32_C(  -855.08), EASYSIMD_FLOAT32_C(   212.37) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  5725.49), EASYSIMD_FLOAT32_C(-167783.62), EASYSIMD_FLOAT32_C(-299953.72),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-28546.10), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(229029.27), EASYSIMD_FLOAT32_C(808256.38), EASYSIMD_FLOAT32_C(-387243.75), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-16373.58), EASYSIMD_FLOAT32_C(342305.53), EASYSIMD_FLOAT32_C( 19052.24), EASYSIMD_FLOAT32_C( -6411.97) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 c = easysimd_mm512_loadu_ps(test_vec[i].c);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_fnmsub_ps(test_vec[i].k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_fnmsub_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 c = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_maskz_fnmsub_ps(k, a, b, c);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_fnmsub_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 c[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(    65.73), EASYSIMD_FLOAT64_C(    16.84), EASYSIMD_FLOAT64_C(   -62.22), EASYSIMD_FLOAT64_C(    41.28),
        EASYSIMD_FLOAT64_C(    41.92), EASYSIMD_FLOAT64_C(    45.60), EASYSIMD_FLOAT64_C(    20.93), EASYSIMD_FLOAT64_C(     8.26) },
      { EASYSIMD_FLOAT64_C(     7.90), EASYSIMD_FLOAT64_C(    31.90), EASYSIMD_FLOAT64_C(   -14.89), EASYSIMD_FLOAT64_C(   -98.38),
        EASYSIMD_FLOAT64_C(   -98.27), EASYSIMD_FLOAT64_C(    60.31), EASYSIMD_FLOAT64_C(   -39.17), EASYSIMD_FLOAT64_C(    82.51) },
      { EASYSIMD_FLOAT64_C(    35.14), EASYSIMD_FLOAT64_C(   -85.00), EASYSIMD_FLOAT64_C(   -10.86), EASYSIMD_FLOAT64_C(    33.90),
        EASYSIMD_FLOAT64_C(   -68.39), EASYSIMD_FLOAT64_C(   -87.95), EASYSIMD_FLOAT64_C(   -87.51), EASYSIMD_FLOAT64_C(   -78.91) },
      { EASYSIMD_FLOAT64_C(  -554.41), EASYSIMD_FLOAT64_C(  -452.20), EASYSIMD_FLOAT64_C(  -915.60), EASYSIMD_FLOAT64_C(  4027.23),
        EASYSIMD_FLOAT64_C(  4187.87), EASYSIMD_FLOAT64_C( -2662.19), EASYSIMD_FLOAT64_C(   907.34), EASYSIMD_FLOAT64_C(  -602.62) } },
    { { EASYSIMD_FLOAT64_C(   -48.41), EASYSIMD_FLOAT64_C(    52.12), EASYSIMD_FLOAT64_C(   -76.82), EASYSIMD_FLOAT64_C(   -20.81),
        EASYSIMD_FLOAT64_C(    40.27), EASYSIMD_FLOAT64_C(   -85.67), EASYSIMD_FLOAT64_C(   -20.01), EASYSIMD_FLOAT64_C(     6.00) },
      { EASYSIMD_FLOAT64_C(    31.17), EASYSIMD_FLOAT64_C(    17.77), EASYSIMD_FLOAT64_C(   -52.72), EASYSIMD_FLOAT64_C(   -26.91),
        EASYSIMD_FLOAT64_C(   -36.63), EASYSIMD_FLOAT64_C(    68.22), EASYSIMD_FLOAT64_C(    81.34), EASYSIMD_FLOAT64_C(    71.27) },
      { EASYSIMD_FLOAT64_C(     0.11), EASYSIMD_FLOAT64_C(   -33.55), EASYSIMD_FLOAT64_C(    72.89), EASYSIMD_FLOAT64_C(     1.84),
        EASYSIMD_FLOAT64_C(   -73.23), EASYSIMD_FLOAT64_C(   -66.27), EASYSIMD_FLOAT64_C(   -15.65), EASYSIMD_FLOAT64_C(    61.91) },
      { EASYSIMD_FLOAT64_C(  1508.83), EASYSIMD_FLOAT64_C(  -892.62), EASYSIMD_FLOAT64_C( -4122.84), EASYSIMD_FLOAT64_C(  -561.84),
        EASYSIMD_FLOAT64_C(  1548.32), EASYSIMD_FLOAT64_C(  5910.68), EASYSIMD_FLOAT64_C(  1643.26), EASYSIMD_FLOAT64_C(  -489.53) } },
    { { EASYSIMD_FLOAT64_C(   -51.27), EASYSIMD_FLOAT64_C(    73.48), EASYSIMD_FLOAT64_C(    -4.19), EASYSIMD_FLOAT64_C(   -19.66),
        EASYSIMD_FLOAT64_C(    85.53), EASYSIMD_FLOAT64_C(     8.30), EASYSIMD_FLOAT64_C(     1.43), EASYSIMD_FLOAT64_C(   -62.88) },
      { EASYSIMD_FLOAT64_C(   -39.58), EASYSIMD_FLOAT64_C(    24.60), EASYSIMD_FLOAT64_C(    16.31), EASYSIMD_FLOAT64_C(   -99.31),
        EASYSIMD_FLOAT64_C(    38.94), EASYSIMD_FLOAT64_C(    96.30), EASYSIMD_FLOAT64_C(     6.69), EASYSIMD_FLOAT64_C(   -29.89) },
      { EASYSIMD_FLOAT64_C(    14.08), EASYSIMD_FLOAT64_C(    53.98), EASYSIMD_FLOAT64_C(    43.19), EASYSIMD_FLOAT64_C(    77.45),
        EASYSIMD_FLOAT64_C(    22.19), EASYSIMD_FLOAT64_C(    24.54), EASYSIMD_FLOAT64_C(    48.72), EASYSIMD_FLOAT64_C(   -77.69) },
      { EASYSIMD_FLOAT64_C( -2043.35), EASYSIMD_FLOAT64_C( -1861.59), EASYSIMD_FLOAT64_C(    25.15), EASYSIMD_FLOAT64_C( -2029.88),
        EASYSIMD_FLOAT64_C( -3352.73), EASYSIMD_FLOAT64_C(  -823.83), EASYSIMD_FLOAT64_C(   -58.29), EASYSIMD_FLOAT64_C( -1801.79) } },
    { { EASYSIMD_FLOAT64_C(    90.99), EASYSIMD_FLOAT64_C(    21.61), EASYSIMD_FLOAT64_C(    24.15), EASYSIMD_FLOAT64_C(   -82.24),
        EASYSIMD_FLOAT64_C(    55.34), EASYSIMD_FLOAT64_C(   -91.51), EASYSIMD_FLOAT64_C(    79.67), EASYSIMD_FLOAT64_C(   -95.94) },
      { EASYSIMD_FLOAT64_C(    81.97), EASYSIMD_FLOAT64_C(   -24.52), EASYSIMD_FLOAT64_C(   -15.60), EASYSIMD_FLOAT64_C(    67.51),
        EASYSIMD_FLOAT64_C(    83.77), EASYSIMD_FLOAT64_C(    85.83), EASYSIMD_FLOAT64_C(   -95.37), EASYSIMD_FLOAT64_C(   -55.81) },
      { EASYSIMD_FLOAT64_C(    10.43), EASYSIMD_FLOAT64_C(    20.94), EASYSIMD_FLOAT64_C(   -55.12), EASYSIMD_FLOAT64_C(   -50.63),
        EASYSIMD_FLOAT64_C(    17.24), EASYSIMD_FLOAT64_C(    51.57), EASYSIMD_FLOAT64_C(    19.47), EASYSIMD_FLOAT64_C(   -68.68) },
      { EASYSIMD_FLOAT64_C( -7468.88), EASYSIMD_FLOAT64_C(   508.94), EASYSIMD_FLOAT64_C(   431.86), EASYSIMD_FLOAT64_C(  5602.65),
        EASYSIMD_FLOAT64_C( -4653.07), EASYSIMD_FLOAT64_C(  7802.73), EASYSIMD_FLOAT64_C(  7578.66), EASYSIMD_FLOAT64_C( -5285.73) } },
    { { EASYSIMD_FLOAT64_C(     5.55), EASYSIMD_FLOAT64_C(   -37.33), EASYSIMD_FLOAT64_C(   -91.23), EASYSIMD_FLOAT64_C(   -72.26),
        EASYSIMD_FLOAT64_C(    87.21), EASYSIMD_FLOAT64_C(    57.48), EASYSIMD_FLOAT64_C(   -49.95), EASYSIMD_FLOAT64_C(    78.20) },
      { EASYSIMD_FLOAT64_C(   -20.90), EASYSIMD_FLOAT64_C(    74.20), EASYSIMD_FLOAT64_C(    95.96), EASYSIMD_FLOAT64_C(   -65.57),
        EASYSIMD_FLOAT64_C(    82.69), EASYSIMD_FLOAT64_C(    75.63), EASYSIMD_FLOAT64_C(   -61.50), EASYSIMD_FLOAT64_C(    64.66) },
      { EASYSIMD_FLOAT64_C(   -48.89), EASYSIMD_FLOAT64_C(    22.90), EASYSIMD_FLOAT64_C(    32.17), EASYSIMD_FLOAT64_C(   -65.12),
        EASYSIMD_FLOAT64_C(     8.73), EASYSIMD_FLOAT64_C(    36.80), EASYSIMD_FLOAT64_C(   -20.93), EASYSIMD_FLOAT64_C(   -80.84) },
      { EASYSIMD_FLOAT64_C(   164.88), EASYSIMD_FLOAT64_C(  2746.99), EASYSIMD_FLOAT64_C(  8722.26), EASYSIMD_FLOAT64_C( -4672.97),
        EASYSIMD_FLOAT64_C( -7220.12), EASYSIMD_FLOAT64_C( -4384.01), EASYSIMD_FLOAT64_C( -3051.00), EASYSIMD_FLOAT64_C( -4975.57) } },
    { { EASYSIMD_FLOAT64_C(   -42.25), EASYSIMD_FLOAT64_C(    23.95), EASYSIMD_FLOAT64_C(   -31.47), EASYSIMD_FLOAT64_C(    74.99),
        EASYSIMD_FLOAT64_C(   -24.48), EASYSIMD_FLOAT64_C(    88.00), EASYSIMD_FLOAT64_C(   -93.69), EASYSIMD_FLOAT64_C(    81.07) },
      { EASYSIMD_FLOAT64_C(   -49.33), EASYSIMD_FLOAT64_C(   -84.92), EASYSIMD_FLOAT64_C(   -91.19), EASYSIMD_FLOAT64_C(   -62.12),
        EASYSIMD_FLOAT64_C(    72.56), EASYSIMD_FLOAT64_C(   -41.14), EASYSIMD_FLOAT64_C(   -83.92), EASYSIMD_FLOAT64_C(   -48.34) },
      { EASYSIMD_FLOAT64_C(   -66.94), EASYSIMD_FLOAT64_C(   -87.96), EASYSIMD_FLOAT64_C(   -13.91), EASYSIMD_FLOAT64_C(   -84.25),
        EASYSIMD_FLOAT64_C(    87.67), EASYSIMD_FLOAT64_C(    24.58), EASYSIMD_FLOAT64_C(    80.41), EASYSIMD_FLOAT64_C(   -61.22) },
      { EASYSIMD_FLOAT64_C( -2017.25), EASYSIMD_FLOAT64_C(  2121.79), EASYSIMD_FLOAT64_C( -2855.84), EASYSIMD_FLOAT64_C(  4742.63),
        EASYSIMD_FLOAT64_C(  1688.60), EASYSIMD_FLOAT64_C(  3595.74), EASYSIMD_FLOAT64_C( -7942.87), EASYSIMD_FLOAT64_C(  3980.14) } },
    { { EASYSIMD_FLOAT64_C(   -52.52), EASYSIMD_FLOAT64_C(    12.58), EASYSIMD_FLOAT64_C(   -26.34), EASYSIMD_FLOAT64_C(    56.21),
        EASYSIMD_FLOAT64_C(   -50.62), EASYSIMD_FLOAT64_C(    52.73), EASYSIMD_FLOAT64_C(    75.37), EASYSIMD_FLOAT64_C(     7.13) },
      { EASYSIMD_FLOAT64_C(   -23.32), EASYSIMD_FLOAT64_C(   -56.10), EASYSIMD_FLOAT64_C(   -17.88), EASYSIMD_FLOAT64_C(    52.20),
        EASYSIMD_FLOAT64_C(   -68.10), EASYSIMD_FLOAT64_C(   -11.57), EASYSIMD_FLOAT64_C(    33.27), EASYSIMD_FLOAT64_C(   -17.43) },
      { EASYSIMD_FLOAT64_C(     3.50), EASYSIMD_FLOAT64_C(    42.08), EASYSIMD_FLOAT64_C(    20.45), EASYSIMD_FLOAT64_C(   -23.94),
        EASYSIMD_FLOAT64_C(   -99.05), EASYSIMD_FLOAT64_C(    36.53), EASYSIMD_FLOAT64_C(    27.72), EASYSIMD_FLOAT64_C(   -66.00) },
      { EASYSIMD_FLOAT64_C( -1228.27), EASYSIMD_FLOAT64_C(   663.66), EASYSIMD_FLOAT64_C(  -491.41), EASYSIMD_FLOAT64_C( -2910.22),
        EASYSIMD_FLOAT64_C( -3348.17), EASYSIMD_FLOAT64_C(   573.56), EASYSIMD_FLOAT64_C( -2535.28), EASYSIMD_FLOAT64_C(   190.28) } },
    { { EASYSIMD_FLOAT64_C(    48.57), EASYSIMD_FLOAT64_C(   -86.20), EASYSIMD_FLOAT64_C(   -50.25), EASYSIMD_FLOAT64_C(    36.24),
        EASYSIMD_FLOAT64_C(    38.39), EASYSIMD_FLOAT64_C(   -69.84), EASYSIMD_FLOAT64_C(    75.01), EASYSIMD_FLOAT64_C(    85.87) },
      { EASYSIMD_FLOAT64_C(    42.73), EASYSIMD_FLOAT64_C(   -51.33), EASYSIMD_FLOAT64_C(    42.08), EASYSIMD_FLOAT64_C(    92.12),
        EASYSIMD_FLOAT64_C(   -98.60), EASYSIMD_FLOAT64_C(    17.44), EASYSIMD_FLOAT64_C(    -0.76), EASYSIMD_FLOAT64_C(   -21.92) },
      { EASYSIMD_FLOAT64_C(    61.34), EASYSIMD_FLOAT64_C(    81.36), EASYSIMD_FLOAT64_C(   -69.73), EASYSIMD_FLOAT64_C(    93.24),
        EASYSIMD_FLOAT64_C(   -30.21), EASYSIMD_FLOAT64_C(    63.55), EASYSIMD_FLOAT64_C(   -24.19), EASYSIMD_FLOAT64_C(    73.29) },
      { EASYSIMD_FLOAT64_C( -2136.74), EASYSIMD_FLOAT64_C( -4506.01), EASYSIMD_FLOAT64_C(  2184.25), EASYSIMD_FLOAT64_C( -3431.67),
        EASYSIMD_FLOAT64_C(  3815.46), EASYSIMD_FLOAT64_C(  1154.46), EASYSIMD_FLOAT64_C(    81.20), EASYSIMD_FLOAT64_C(  1808.98) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d c = easysimd_mm512_loadu_pd(test_vec[i].c);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_fnmsub_pd(a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_fnmsub_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask3_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask3_fnmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fnmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fnmsub_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fnmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fnmsub_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_fnmsub_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
