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

#define EASYSIMD_TEST_X86_AVX512_INSN fnmadd

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/fnmadd.h>

static int
test_easysimd_mm_mask3_fnmadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const uint8_t k;
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   -37.58), EASYSIMD_FLOAT32_C(  -482.63), EASYSIMD_FLOAT32_C(  -268.25), EASYSIMD_FLOAT32_C(   659.52) },
      UINT8_C(237),
      { EASYSIMD_FLOAT32_C(   607.93), EASYSIMD_FLOAT32_C(   515.45), EASYSIMD_FLOAT32_C(  -655.36), EASYSIMD_FLOAT32_C(    49.08) },
      { EASYSIMD_FLOAT32_C(   898.29), EASYSIMD_FLOAT32_C(   807.36), EASYSIMD_FLOAT32_C(  -609.11), EASYSIMD_FLOAT32_C(    99.59) },
      { EASYSIMD_FLOAT32_C( 23744.30), EASYSIMD_FLOAT32_C(   807.36), EASYSIMD_FLOAT32_C(-176409.42), EASYSIMD_FLOAT32_C(-32269.65) } },
    { { EASYSIMD_FLOAT32_C(   437.67), EASYSIMD_FLOAT32_C(  -112.13), EASYSIMD_FLOAT32_C(  -945.60), EASYSIMD_FLOAT32_C(   152.96) },
      UINT8_C(218),
      { EASYSIMD_FLOAT32_C(   788.14), EASYSIMD_FLOAT32_C(  -314.54), EASYSIMD_FLOAT32_C(  -333.15), EASYSIMD_FLOAT32_C(  -189.59) },
      { EASYSIMD_FLOAT32_C(   189.20), EASYSIMD_FLOAT32_C(  -784.92), EASYSIMD_FLOAT32_C(  -731.01), EASYSIMD_FLOAT32_C(   441.16) },
      { EASYSIMD_FLOAT32_C(   189.20), EASYSIMD_FLOAT32_C(-36054.29), EASYSIMD_FLOAT32_C(  -731.01), EASYSIMD_FLOAT32_C( 29440.85) } },
    { { EASYSIMD_FLOAT32_C(   -49.94), EASYSIMD_FLOAT32_C(   908.86), EASYSIMD_FLOAT32_C(  -766.64), EASYSIMD_FLOAT32_C(  -571.37) },
      UINT8_C(  5),
      { EASYSIMD_FLOAT32_C(   195.79), EASYSIMD_FLOAT32_C(   -54.00), EASYSIMD_FLOAT32_C(   660.32), EASYSIMD_FLOAT32_C(  -144.69) },
      { EASYSIMD_FLOAT32_C(  -161.65), EASYSIMD_FLOAT32_C(   268.25), EASYSIMD_FLOAT32_C(  -629.25), EASYSIMD_FLOAT32_C(   182.99) },
      { EASYSIMD_FLOAT32_C(  9616.10), EASYSIMD_FLOAT32_C(   268.25), EASYSIMD_FLOAT32_C(505598.50), EASYSIMD_FLOAT32_C(   182.99) } },
    { { EASYSIMD_FLOAT32_C(  -682.67), EASYSIMD_FLOAT32_C(  -730.96), EASYSIMD_FLOAT32_C(    -9.65), EASYSIMD_FLOAT32_C(  -291.78) },
      UINT8_C( 97),
      { EASYSIMD_FLOAT32_C(  -571.97), EASYSIMD_FLOAT32_C(   596.09), EASYSIMD_FLOAT32_C(   423.03), EASYSIMD_FLOAT32_C(   580.99) },
      { EASYSIMD_FLOAT32_C(  -942.89), EASYSIMD_FLOAT32_C(   211.17), EASYSIMD_FLOAT32_C(  -733.56), EASYSIMD_FLOAT32_C(  -276.05) },
      { EASYSIMD_FLOAT32_C(-391409.59), EASYSIMD_FLOAT32_C(   211.17), EASYSIMD_FLOAT32_C(  -733.56), EASYSIMD_FLOAT32_C(  -276.05) } },
    { { EASYSIMD_FLOAT32_C(  -978.42), EASYSIMD_FLOAT32_C(   455.65), EASYSIMD_FLOAT32_C(   -60.97), EASYSIMD_FLOAT32_C(  -709.43) },
      UINT8_C(123),
      { EASYSIMD_FLOAT32_C(   889.09), EASYSIMD_FLOAT32_C(  -800.58), EASYSIMD_FLOAT32_C(   130.17), EASYSIMD_FLOAT32_C(  -682.28) },
      { EASYSIMD_FLOAT32_C(   128.00), EASYSIMD_FLOAT32_C(  -674.04), EASYSIMD_FLOAT32_C(   263.72), EASYSIMD_FLOAT32_C(  -211.68) },
      { EASYSIMD_FLOAT32_C(870031.44), EASYSIMD_FLOAT32_C(364110.25), EASYSIMD_FLOAT32_C(   263.72), EASYSIMD_FLOAT32_C(-484241.59) } },
    { { EASYSIMD_FLOAT32_C(   181.27), EASYSIMD_FLOAT32_C(  -897.93), EASYSIMD_FLOAT32_C(  -943.43), EASYSIMD_FLOAT32_C(   552.02) },
      UINT8_C( 39),
      { EASYSIMD_FLOAT32_C(  -626.10), EASYSIMD_FLOAT32_C(   821.06), EASYSIMD_FLOAT32_C(  -724.59), EASYSIMD_FLOAT32_C(    82.13) },
      { EASYSIMD_FLOAT32_C(   189.69), EASYSIMD_FLOAT32_C(  -296.56), EASYSIMD_FLOAT32_C(  -321.79), EASYSIMD_FLOAT32_C(  -387.28) },
      { EASYSIMD_FLOAT32_C(113682.84), EASYSIMD_FLOAT32_C(736957.81), EASYSIMD_FLOAT32_C(-683921.75), EASYSIMD_FLOAT32_C(  -387.28) } },
    { { EASYSIMD_FLOAT32_C(  -715.57), EASYSIMD_FLOAT32_C(  -264.68), EASYSIMD_FLOAT32_C(   823.89), EASYSIMD_FLOAT32_C(  -449.13) },
      UINT8_C( 99),
      { EASYSIMD_FLOAT32_C(   845.47), EASYSIMD_FLOAT32_C(  -993.48), EASYSIMD_FLOAT32_C(  -601.70), EASYSIMD_FLOAT32_C(  -863.96) },
      { EASYSIMD_FLOAT32_C(   -96.67), EASYSIMD_FLOAT32_C(  -712.61), EASYSIMD_FLOAT32_C(  -664.53), EASYSIMD_FLOAT32_C(  -966.50) },
      { EASYSIMD_FLOAT32_C(604896.25), EASYSIMD_FLOAT32_C(-263666.91), EASYSIMD_FLOAT32_C(  -664.53), EASYSIMD_FLOAT32_C(  -966.50) } },
    { { EASYSIMD_FLOAT32_C(  -394.90), EASYSIMD_FLOAT32_C(   463.47), EASYSIMD_FLOAT32_C(  -640.55), EASYSIMD_FLOAT32_C(   868.82) },
      UINT8_C(172),
      { EASYSIMD_FLOAT32_C(   540.72), EASYSIMD_FLOAT32_C(   970.89), EASYSIMD_FLOAT32_C(  -691.64), EASYSIMD_FLOAT32_C(    92.74) },
      { EASYSIMD_FLOAT32_C(   255.95), EASYSIMD_FLOAT32_C(  -317.73), EASYSIMD_FLOAT32_C(   -86.20), EASYSIMD_FLOAT32_C(   531.36) },
      { EASYSIMD_FLOAT32_C(   255.95), EASYSIMD_FLOAT32_C(  -317.73), EASYSIMD_FLOAT32_C(-443116.19), EASYSIMD_FLOAT32_C(-80043.01) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask3_fnmadd_ps(a, b, c, k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask3_fnmadd_ps");
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
    easysimd__m128 r = easysimd_mm_mask3_fnmadd_ps(a, b, c, k);

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
test_easysimd_mm_mask_fnmadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const uint8_t k;
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -703.34), EASYSIMD_FLOAT32_C(  -624.64), EASYSIMD_FLOAT32_C(   178.64), EASYSIMD_FLOAT32_C(   806.53) },
      UINT8_C( 58),
      { EASYSIMD_FLOAT32_C(  -726.95), EASYSIMD_FLOAT32_C(   960.57), EASYSIMD_FLOAT32_C(   529.60), EASYSIMD_FLOAT32_C(   913.16) },
      { EASYSIMD_FLOAT32_C(  -774.33), EASYSIMD_FLOAT32_C(  -990.00), EASYSIMD_FLOAT32_C(   213.92), EASYSIMD_FLOAT32_C(   738.83) },
      { EASYSIMD_FLOAT32_C(  -703.34), EASYSIMD_FLOAT32_C(599020.44), EASYSIMD_FLOAT32_C(   178.64), EASYSIMD_FLOAT32_C(-735752.12) } },
    { { EASYSIMD_FLOAT32_C(   978.97), EASYSIMD_FLOAT32_C(  -476.72), EASYSIMD_FLOAT32_C(  -915.50), EASYSIMD_FLOAT32_C(   930.88) },
      UINT8_C(113),
      { EASYSIMD_FLOAT32_C(   529.39), EASYSIMD_FLOAT32_C(   943.14), EASYSIMD_FLOAT32_C(   829.34), EASYSIMD_FLOAT32_C(   797.83) },
      { EASYSIMD_FLOAT32_C(   425.09), EASYSIMD_FLOAT32_C(   143.21), EASYSIMD_FLOAT32_C(  -132.62), EASYSIMD_FLOAT32_C(  -418.26) },
      { EASYSIMD_FLOAT32_C(-517831.84), EASYSIMD_FLOAT32_C(  -476.72), EASYSIMD_FLOAT32_C(  -915.50), EASYSIMD_FLOAT32_C(   930.88) } },
    { { EASYSIMD_FLOAT32_C(    72.91), EASYSIMD_FLOAT32_C(   210.95), EASYSIMD_FLOAT32_C(  -808.40), EASYSIMD_FLOAT32_C(   164.22) },
      UINT8_C(186),
      { EASYSIMD_FLOAT32_C(  -511.74), EASYSIMD_FLOAT32_C(   539.57), EASYSIMD_FLOAT32_C(   993.20), EASYSIMD_FLOAT32_C(  -705.21) },
      { EASYSIMD_FLOAT32_C(  -604.07), EASYSIMD_FLOAT32_C(  -733.75), EASYSIMD_FLOAT32_C(  -744.64), EASYSIMD_FLOAT32_C(   925.53) },
      { EASYSIMD_FLOAT32_C(    72.91), EASYSIMD_FLOAT32_C(-114556.04), EASYSIMD_FLOAT32_C(  -808.40), EASYSIMD_FLOAT32_C(116735.12) } },
    { { EASYSIMD_FLOAT32_C(  -820.58), EASYSIMD_FLOAT32_C(  -518.97), EASYSIMD_FLOAT32_C(   935.53), EASYSIMD_FLOAT32_C(   393.34) },
      UINT8_C(176),
      { EASYSIMD_FLOAT32_C(   914.51), EASYSIMD_FLOAT32_C(   916.62), EASYSIMD_FLOAT32_C(  -695.64), EASYSIMD_FLOAT32_C(   845.38) },
      { EASYSIMD_FLOAT32_C(  -565.97), EASYSIMD_FLOAT32_C(   833.75), EASYSIMD_FLOAT32_C(   788.52), EASYSIMD_FLOAT32_C(  -736.62) },
      { EASYSIMD_FLOAT32_C(  -820.58), EASYSIMD_FLOAT32_C(  -518.97), EASYSIMD_FLOAT32_C(   935.53), EASYSIMD_FLOAT32_C(   393.34) } },
    { { EASYSIMD_FLOAT32_C(   631.58), EASYSIMD_FLOAT32_C(   213.61), EASYSIMD_FLOAT32_C(   406.58), EASYSIMD_FLOAT32_C(  -501.04) },
      UINT8_C( 28),
      { EASYSIMD_FLOAT32_C(  -520.50), EASYSIMD_FLOAT32_C(   709.91), EASYSIMD_FLOAT32_C(   986.96), EASYSIMD_FLOAT32_C(   643.71) },
      { EASYSIMD_FLOAT32_C(  -475.53), EASYSIMD_FLOAT32_C(  -524.78), EASYSIMD_FLOAT32_C(   183.29), EASYSIMD_FLOAT32_C(  -482.33) },
      { EASYSIMD_FLOAT32_C(   631.58), EASYSIMD_FLOAT32_C(   213.61), EASYSIMD_FLOAT32_C(-401094.91), EASYSIMD_FLOAT32_C(322042.12) } },
    { { EASYSIMD_FLOAT32_C(  -229.99), EASYSIMD_FLOAT32_C(   579.22), EASYSIMD_FLOAT32_C(  -216.08), EASYSIMD_FLOAT32_C(    25.37) },
      UINT8_C( 20),
      { EASYSIMD_FLOAT32_C(   -36.66), EASYSIMD_FLOAT32_C(   506.40), EASYSIMD_FLOAT32_C(   440.28), EASYSIMD_FLOAT32_C(  -643.32) },
      { EASYSIMD_FLOAT32_C(   726.25), EASYSIMD_FLOAT32_C(   354.79), EASYSIMD_FLOAT32_C(  -726.70), EASYSIMD_FLOAT32_C(  -969.39) },
      { EASYSIMD_FLOAT32_C(  -229.99), EASYSIMD_FLOAT32_C(   579.22), EASYSIMD_FLOAT32_C( 94409.00), EASYSIMD_FLOAT32_C(    25.37) } },
    { { EASYSIMD_FLOAT32_C(   200.17), EASYSIMD_FLOAT32_C(  -292.67), EASYSIMD_FLOAT32_C(   864.36), EASYSIMD_FLOAT32_C(   -11.30) },
      UINT8_C(204),
      { EASYSIMD_FLOAT32_C(   495.94), EASYSIMD_FLOAT32_C(  -797.69), EASYSIMD_FLOAT32_C(  -622.71), EASYSIMD_FLOAT32_C(   994.90) },
      { EASYSIMD_FLOAT32_C(   997.67), EASYSIMD_FLOAT32_C(  -143.21), EASYSIMD_FLOAT32_C(   704.80), EASYSIMD_FLOAT32_C(   984.63) },
      { EASYSIMD_FLOAT32_C(   200.17), EASYSIMD_FLOAT32_C(  -292.67), EASYSIMD_FLOAT32_C(538950.44), EASYSIMD_FLOAT32_C( 12227.00) } },
    { { EASYSIMD_FLOAT32_C(  -499.50), EASYSIMD_FLOAT32_C(  -770.73), EASYSIMD_FLOAT32_C(  -540.16), EASYSIMD_FLOAT32_C(   683.79) },
      UINT8_C(117),
      { EASYSIMD_FLOAT32_C(   229.85), EASYSIMD_FLOAT32_C(   263.01), EASYSIMD_FLOAT32_C(   530.86), EASYSIMD_FLOAT32_C(  -744.78) },
      { EASYSIMD_FLOAT32_C(  -232.23), EASYSIMD_FLOAT32_C(  -505.80), EASYSIMD_FLOAT32_C(   761.61), EASYSIMD_FLOAT32_C(  -791.95) },
      { EASYSIMD_FLOAT32_C(114577.85), EASYSIMD_FLOAT32_C(  -770.73), EASYSIMD_FLOAT32_C(287510.94), EASYSIMD_FLOAT32_C(   683.79) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_fnmadd_ps(a, k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_fnmadd_ps");
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
    easysimd__m128 r = easysimd_mm_mask_fnmadd_ps(a, k, b, c);

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
test_easysimd_mm_maskz_fnmadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const uint8_t k;
    const easysimd_float32 b[4];
    const easysimd_float32 c[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
{ { EASYSIMD_FLOAT32_C(   912.93), EASYSIMD_FLOAT32_C(  -796.24), EASYSIMD_FLOAT32_C(  -164.15), EASYSIMD_FLOAT32_C(  -312.96) },
      UINT8_C( 54),
      { EASYSIMD_FLOAT32_C(   804.16), EASYSIMD_FLOAT32_C(   779.22), EASYSIMD_FLOAT32_C(   884.41), EASYSIMD_FLOAT32_C(  -452.24) },
      { EASYSIMD_FLOAT32_C(  -430.16), EASYSIMD_FLOAT32_C(   660.82), EASYSIMD_FLOAT32_C(  -603.03), EASYSIMD_FLOAT32_C(   246.41) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(621106.94), EASYSIMD_FLOAT32_C(144572.86), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -370.68), EASYSIMD_FLOAT32_C(  -918.03), EASYSIMD_FLOAT32_C(   454.27), EASYSIMD_FLOAT32_C(   699.09) },
      UINT8_C(131),
      { EASYSIMD_FLOAT32_C(  -641.04), EASYSIMD_FLOAT32_C(   203.43), EASYSIMD_FLOAT32_C(   671.68), EASYSIMD_FLOAT32_C(  -496.85) },
      { EASYSIMD_FLOAT32_C(  -167.08), EASYSIMD_FLOAT32_C(  -114.75), EASYSIMD_FLOAT32_C(  -863.42), EASYSIMD_FLOAT32_C(   -86.86) },
      { EASYSIMD_FLOAT32_C(-237787.77), EASYSIMD_FLOAT32_C(186640.09), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -232.90), EASYSIMD_FLOAT32_C(  -821.82), EASYSIMD_FLOAT32_C(   482.48), EASYSIMD_FLOAT32_C(   277.18) },
      UINT8_C(120),
      { EASYSIMD_FLOAT32_C(   395.40), EASYSIMD_FLOAT32_C(   480.94), EASYSIMD_FLOAT32_C(   464.33), EASYSIMD_FLOAT32_C(  -917.55) },
      { EASYSIMD_FLOAT32_C(   562.34), EASYSIMD_FLOAT32_C(   268.49), EASYSIMD_FLOAT32_C(   861.67), EASYSIMD_FLOAT32_C(   446.75) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(254773.25) } },
    { { EASYSIMD_FLOAT32_C(   816.25), EASYSIMD_FLOAT32_C(  -568.49), EASYSIMD_FLOAT32_C(   107.56), EASYSIMD_FLOAT32_C(  -786.79) },
      UINT8_C( 26),
      { EASYSIMD_FLOAT32_C(   736.88), EASYSIMD_FLOAT32_C(  -704.82), EASYSIMD_FLOAT32_C(   132.19), EASYSIMD_FLOAT32_C(   435.97) },
      { EASYSIMD_FLOAT32_C(   -50.61), EASYSIMD_FLOAT32_C(   491.15), EASYSIMD_FLOAT32_C(  -360.60), EASYSIMD_FLOAT32_C(  -378.93) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-400191.97), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(342637.88) } },
    { { EASYSIMD_FLOAT32_C(   994.30), EASYSIMD_FLOAT32_C(   472.32), EASYSIMD_FLOAT32_C(   506.32), EASYSIMD_FLOAT32_C(  -869.12) },
      UINT8_C(214),
      { EASYSIMD_FLOAT32_C(  -726.59), EASYSIMD_FLOAT32_C(  -690.94), EASYSIMD_FLOAT32_C(   867.94), EASYSIMD_FLOAT32_C(   550.59) },
      { EASYSIMD_FLOAT32_C(   -62.46), EASYSIMD_FLOAT32_C(   263.34), EASYSIMD_FLOAT32_C(    31.53), EASYSIMD_FLOAT32_C(  -598.14) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(326608.12), EASYSIMD_FLOAT32_C(-439423.84), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   345.79), EASYSIMD_FLOAT32_C(  -406.13), EASYSIMD_FLOAT32_C(   670.35), EASYSIMD_FLOAT32_C(   207.46) },
      UINT8_C(236),
      { EASYSIMD_FLOAT32_C(   486.60), EASYSIMD_FLOAT32_C(   638.97), EASYSIMD_FLOAT32_C(   148.19), EASYSIMD_FLOAT32_C(   699.81) },
      { EASYSIMD_FLOAT32_C(   316.89), EASYSIMD_FLOAT32_C(  -114.93), EASYSIMD_FLOAT32_C(   995.00), EASYSIMD_FLOAT32_C(  -550.92) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-98344.16), EASYSIMD_FLOAT32_C(-145733.52) } },
    { { EASYSIMD_FLOAT32_C(  -678.96), EASYSIMD_FLOAT32_C(   -55.62), EASYSIMD_FLOAT32_C(   940.23), EASYSIMD_FLOAT32_C(   -39.55) },
      UINT8_C( 55),
      { EASYSIMD_FLOAT32_C(   934.52), EASYSIMD_FLOAT32_C(  -567.23), EASYSIMD_FLOAT32_C(    71.77), EASYSIMD_FLOAT32_C(  -934.60) },
      { EASYSIMD_FLOAT32_C(  -181.77), EASYSIMD_FLOAT32_C(   345.18), EASYSIMD_FLOAT32_C(  -625.54), EASYSIMD_FLOAT32_C(  -313.83) },
      { EASYSIMD_FLOAT32_C(634320.00), EASYSIMD_FLOAT32_C(-31204.15), EASYSIMD_FLOAT32_C(-68105.84), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -104.23), EASYSIMD_FLOAT32_C(   311.99), EASYSIMD_FLOAT32_C(   949.52), EASYSIMD_FLOAT32_C(   927.31) },
      UINT8_C( 73),
      { EASYSIMD_FLOAT32_C(   295.31), EASYSIMD_FLOAT32_C(  -478.82), EASYSIMD_FLOAT32_C(   384.21), EASYSIMD_FLOAT32_C(  -497.23) },
      { EASYSIMD_FLOAT32_C(  -438.20), EASYSIMD_FLOAT32_C(  -129.19), EASYSIMD_FLOAT32_C(  -858.26), EASYSIMD_FLOAT32_C(   709.99) },
      { EASYSIMD_FLOAT32_C( 30341.96), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(461796.38) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 a = easysimd_mm_loadu_ps(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 b = easysimd_mm_loadu_ps(test_vec[i].b);
    easysimd__m128 c = easysimd_mm_loadu_ps(test_vec[i].c);
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_fnmadd_ps(k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_fnmadd_ps");
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
    easysimd__m128 r = easysimd_mm_maskz_fnmadd_ps(k, a, b, c);

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
test_easysimd_mm_mask3_fnmadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const uint8_t k;
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -27.16), EASYSIMD_FLOAT64_C(   -21.78) },
      UINT8_C(120),
      { EASYSIMD_FLOAT64_C(   -96.30), EASYSIMD_FLOAT64_C(   155.09) },
      { EASYSIMD_FLOAT64_C(  -859.82), EASYSIMD_FLOAT64_C(   695.22) },
      { EASYSIMD_FLOAT64_C(  -859.82), EASYSIMD_FLOAT64_C(   695.22) } },
    { { EASYSIMD_FLOAT64_C(   -77.71), EASYSIMD_FLOAT64_C(   319.40) },
      UINT8_C(248),
      { EASYSIMD_FLOAT64_C(   181.49), EASYSIMD_FLOAT64_C(   512.01) },
      { EASYSIMD_FLOAT64_C(  -697.40), EASYSIMD_FLOAT64_C(  -198.77) },
      { EASYSIMD_FLOAT64_C(  -697.40), EASYSIMD_FLOAT64_C(  -198.77) } },
    { { EASYSIMD_FLOAT64_C(   576.74), EASYSIMD_FLOAT64_C(  -421.33) },
      UINT8_C(243),
      { EASYSIMD_FLOAT64_C(   -26.90), EASYSIMD_FLOAT64_C(   100.22) },
      { EASYSIMD_FLOAT64_C(  -429.13), EASYSIMD_FLOAT64_C(  -987.51) },
      { EASYSIMD_FLOAT64_C( 15085.18), EASYSIMD_FLOAT64_C( 41238.18) } },
    { { EASYSIMD_FLOAT64_C(   354.43), EASYSIMD_FLOAT64_C(  -493.21) },
      UINT8_C(238),
      { EASYSIMD_FLOAT64_C(   564.34), EASYSIMD_FLOAT64_C(   888.27) },
      { EASYSIMD_FLOAT64_C(   143.07), EASYSIMD_FLOAT64_C(  -410.28) },
      { EASYSIMD_FLOAT64_C(   143.07), EASYSIMD_FLOAT64_C(437693.37) } },
    { { EASYSIMD_FLOAT64_C(  -584.74), EASYSIMD_FLOAT64_C(   946.73) },
      UINT8_C(154),
      { EASYSIMD_FLOAT64_C(   388.10), EASYSIMD_FLOAT64_C(   -75.05) },
      { EASYSIMD_FLOAT64_C(  -635.03), EASYSIMD_FLOAT64_C(  -708.20) },
      { EASYSIMD_FLOAT64_C(  -635.03), EASYSIMD_FLOAT64_C( 70343.89) } },
    { { EASYSIMD_FLOAT64_C(  -919.96), EASYSIMD_FLOAT64_C(  -494.86) },
      UINT8_C( 39),
      { EASYSIMD_FLOAT64_C(     2.32), EASYSIMD_FLOAT64_C(   824.55) },
      { EASYSIMD_FLOAT64_C(  -166.93), EASYSIMD_FLOAT64_C(  -816.18) },
      { EASYSIMD_FLOAT64_C(  1967.38), EASYSIMD_FLOAT64_C(407220.63) } },
    { { EASYSIMD_FLOAT64_C(   336.56), EASYSIMD_FLOAT64_C(   135.66) },
      UINT8_C(215),
      { EASYSIMD_FLOAT64_C(   -86.71), EASYSIMD_FLOAT64_C(   714.33) },
      { EASYSIMD_FLOAT64_C(  -434.97), EASYSIMD_FLOAT64_C(   886.39) },
      { EASYSIMD_FLOAT64_C( 28748.15), EASYSIMD_FLOAT64_C(-96019.62) } },
    { { EASYSIMD_FLOAT64_C(  -185.45), EASYSIMD_FLOAT64_C(   135.90) },
      UINT8_C( 90),
      { EASYSIMD_FLOAT64_C(  -831.02), EASYSIMD_FLOAT64_C(   642.69) },
      { EASYSIMD_FLOAT64_C(   566.35), EASYSIMD_FLOAT64_C(   733.31) },
      { EASYSIMD_FLOAT64_C(   566.35), EASYSIMD_FLOAT64_C(-86608.26) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask3_fnmadd_pd(a, b, c, k);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask3_fnmadd_pd");
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
    easysimd__m128d r = easysimd_mm_mask3_fnmadd_pd(a, b, c, k);

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
test_easysimd_mm_mask_fnmadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const uint8_t k;
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   988.49), EASYSIMD_FLOAT64_C(   242.81) },
      UINT8_C(195),
      { EASYSIMD_FLOAT64_C(  -826.03), EASYSIMD_FLOAT64_C(   818.66) },
      { EASYSIMD_FLOAT64_C(  -181.05), EASYSIMD_FLOAT64_C(  -631.72) },
      { EASYSIMD_FLOAT64_C(816341.34), EASYSIMD_FLOAT64_C(-199410.55) } },
    { { EASYSIMD_FLOAT64_C(   484.99), EASYSIMD_FLOAT64_C(  -371.82) },
      UINT8_C( 29),
      { EASYSIMD_FLOAT64_C(   661.77), EASYSIMD_FLOAT64_C(   365.90) },
      { EASYSIMD_FLOAT64_C(  -793.32), EASYSIMD_FLOAT64_C(  -873.67) },
      { EASYSIMD_FLOAT64_C(-321745.15), EASYSIMD_FLOAT64_C(  -371.82) } },
    { { EASYSIMD_FLOAT64_C(   696.93), EASYSIMD_FLOAT64_C(   483.67) },
      UINT8_C( 97),
      { EASYSIMD_FLOAT64_C(  -903.12), EASYSIMD_FLOAT64_C(  -696.53) },
      { EASYSIMD_FLOAT64_C(  -325.19), EASYSIMD_FLOAT64_C(   980.31) },
      { EASYSIMD_FLOAT64_C(629086.23), EASYSIMD_FLOAT64_C(   483.67) } },
    { { EASYSIMD_FLOAT64_C(  -693.92), EASYSIMD_FLOAT64_C(   168.39) },
      UINT8_C(153),
      { EASYSIMD_FLOAT64_C(  -706.16), EASYSIMD_FLOAT64_C(  -258.34) },
      { EASYSIMD_FLOAT64_C(  -875.64), EASYSIMD_FLOAT64_C(    44.65) },
      { EASYSIMD_FLOAT64_C(-490894.19), EASYSIMD_FLOAT64_C(   168.39) } },
    { { EASYSIMD_FLOAT64_C(   988.35), EASYSIMD_FLOAT64_C(   464.27) },
      UINT8_C(139),
      { EASYSIMD_FLOAT64_C(   976.84), EASYSIMD_FLOAT64_C(  -292.93) },
      { EASYSIMD_FLOAT64_C(   837.32), EASYSIMD_FLOAT64_C(  -849.19) },
      { EASYSIMD_FLOAT64_C(-964622.49), EASYSIMD_FLOAT64_C(135149.42) } },
    { { EASYSIMD_FLOAT64_C(  -474.27), EASYSIMD_FLOAT64_C(  -343.73) },
      UINT8_C(  5),
      { EASYSIMD_FLOAT64_C(  -989.28), EASYSIMD_FLOAT64_C(   284.45) },
      { EASYSIMD_FLOAT64_C(  -325.68), EASYSIMD_FLOAT64_C(   672.49) },
      { EASYSIMD_FLOAT64_C(-469511.51), EASYSIMD_FLOAT64_C(  -343.73) } },
    { { EASYSIMD_FLOAT64_C(  -349.65), EASYSIMD_FLOAT64_C(  -119.01) },
      UINT8_C( 88),
      { EASYSIMD_FLOAT64_C(  -652.72), EASYSIMD_FLOAT64_C(  -635.33) },
      { EASYSIMD_FLOAT64_C(   -58.75), EASYSIMD_FLOAT64_C(  -555.84) },
      { EASYSIMD_FLOAT64_C(  -349.65), EASYSIMD_FLOAT64_C(  -119.01) } },
    { { EASYSIMD_FLOAT64_C(  -331.87), EASYSIMD_FLOAT64_C(   616.06) },
      UINT8_C(  4),
      { EASYSIMD_FLOAT64_C(   -25.79), EASYSIMD_FLOAT64_C(  -215.55) },
      { EASYSIMD_FLOAT64_C(     4.13), EASYSIMD_FLOAT64_C(   268.06) },
      { EASYSIMD_FLOAT64_C(  -331.87), EASYSIMD_FLOAT64_C(   616.06) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_fnmadd_pd(a, k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_fnmadd_pd");
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
    easysimd__m128d r = easysimd_mm_mask_fnmadd_pd(a, k, b, c);

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
test_easysimd_mm_maskz_fnmadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const uint8_t k;
    const easysimd_float64 b[2];
    const easysimd_float64 c[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -103.36), EASYSIMD_FLOAT64_C(  -673.31) },
      UINT8_C(166),
      { EASYSIMD_FLOAT64_C(   632.87), EASYSIMD_FLOAT64_C(   699.12) },
      { EASYSIMD_FLOAT64_C(   140.53), EASYSIMD_FLOAT64_C(   832.31) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(471556.80) } },
    { { EASYSIMD_FLOAT64_C(   -92.96), EASYSIMD_FLOAT64_C(   894.21) },
      UINT8_C(197),
      { EASYSIMD_FLOAT64_C(  -422.35), EASYSIMD_FLOAT64_C(   209.08) },
      { EASYSIMD_FLOAT64_C(  -419.29), EASYSIMD_FLOAT64_C(    -3.44) },
      { EASYSIMD_FLOAT64_C(-39680.95), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -451.09), EASYSIMD_FLOAT64_C(   557.81) },
      UINT8_C( 26),
      { EASYSIMD_FLOAT64_C(   912.49), EASYSIMD_FLOAT64_C(  -944.76) },
      { EASYSIMD_FLOAT64_C(   509.60), EASYSIMD_FLOAT64_C(  -785.60) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(526210.98) } },
    { { EASYSIMD_FLOAT64_C(   330.49), EASYSIMD_FLOAT64_C(  -209.86) },
      UINT8_C( 62),
      { EASYSIMD_FLOAT64_C(  -753.05), EASYSIMD_FLOAT64_C(  -418.16) },
      { EASYSIMD_FLOAT64_C(   276.04), EASYSIMD_FLOAT64_C(  -663.40) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-88418.46) } },
    { { EASYSIMD_FLOAT64_C(   841.14), EASYSIMD_FLOAT64_C(  -343.68) },
      UINT8_C(216),
      { EASYSIMD_FLOAT64_C(  -262.22), EASYSIMD_FLOAT64_C(   -16.99) },
      { EASYSIMD_FLOAT64_C(  -718.47), EASYSIMD_FLOAT64_C(  -629.34) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -317.87), EASYSIMD_FLOAT64_C(   422.06) },
      UINT8_C(252),
      { EASYSIMD_FLOAT64_C(   589.17), EASYSIMD_FLOAT64_C(   316.27) },
      { EASYSIMD_FLOAT64_C(   336.00), EASYSIMD_FLOAT64_C(  -833.18) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -474.64), EASYSIMD_FLOAT64_C(   916.70) },
      UINT8_C( 36),
      { EASYSIMD_FLOAT64_C(    74.27), EASYSIMD_FLOAT64_C(   474.52) },
      { EASYSIMD_FLOAT64_C(   961.44), EASYSIMD_FLOAT64_C(   -13.24) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   529.75), EASYSIMD_FLOAT64_C(   471.03) },
      UINT8_C( 90),
      { EASYSIMD_FLOAT64_C(  -139.76), EASYSIMD_FLOAT64_C(  -738.82) },
      { EASYSIMD_FLOAT64_C(   -61.21), EASYSIMD_FLOAT64_C(   107.19) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(348113.57) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d a = easysimd_mm_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d b = easysimd_mm_loadu_pd(test_vec[i].b);
    easysimd__m128d c = easysimd_mm_loadu_pd(test_vec[i].c);
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_fnmadd_pd(k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_fnmadd_pd");
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
    easysimd__m128d r = easysimd_mm_maskz_fnmadd_pd(k, a, b, c);

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
test_easysimd_mm256_mask_fnmadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const uint8_t k;
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   992.46), EASYSIMD_FLOAT32_C(   376.51), EASYSIMD_FLOAT32_C(  -371.26), EASYSIMD_FLOAT32_C(  -192.45),
        EASYSIMD_FLOAT32_C(   950.17), EASYSIMD_FLOAT32_C(   867.23), EASYSIMD_FLOAT32_C(   832.80), EASYSIMD_FLOAT32_C(   791.58) },
      UINT8_C(249),
      { EASYSIMD_FLOAT32_C(   731.15), EASYSIMD_FLOAT32_C(  -474.92), EASYSIMD_FLOAT32_C(  -698.46), EASYSIMD_FLOAT32_C(  -180.77),
        EASYSIMD_FLOAT32_C(  -981.53), EASYSIMD_FLOAT32_C(   951.75), EASYSIMD_FLOAT32_C(   499.07), EASYSIMD_FLOAT32_C(  -138.79) },
      { EASYSIMD_FLOAT32_C(  -331.47), EASYSIMD_FLOAT32_C(  -549.32), EASYSIMD_FLOAT32_C(   389.28), EASYSIMD_FLOAT32_C(    26.28),
        EASYSIMD_FLOAT32_C(  -734.47), EASYSIMD_FLOAT32_C(  -443.76), EASYSIMD_FLOAT32_C(  -907.32), EASYSIMD_FLOAT32_C(  -141.22) },
      { EASYSIMD_FLOAT32_C(-725968.69), EASYSIMD_FLOAT32_C(   376.51), EASYSIMD_FLOAT32_C(  -371.26), EASYSIMD_FLOAT32_C(-34762.91),
        EASYSIMD_FLOAT32_C(931885.88), EASYSIMD_FLOAT32_C(-825829.88), EASYSIMD_FLOAT32_C(-416532.81), EASYSIMD_FLOAT32_C(109722.16) } },
    { { EASYSIMD_FLOAT32_C(  -310.94), EASYSIMD_FLOAT32_C(  -742.46), EASYSIMD_FLOAT32_C(   598.01), EASYSIMD_FLOAT32_C(  -135.92),
        EASYSIMD_FLOAT32_C(  -438.21), EASYSIMD_FLOAT32_C(   502.76), EASYSIMD_FLOAT32_C(  -143.46), EASYSIMD_FLOAT32_C(   938.30) },
      UINT8_C( 95),
      { EASYSIMD_FLOAT32_C(   664.08), EASYSIMD_FLOAT32_C(   888.48), EASYSIMD_FLOAT32_C(   998.73), EASYSIMD_FLOAT32_C(   496.88),
        EASYSIMD_FLOAT32_C(   680.06), EASYSIMD_FLOAT32_C(  -275.65), EASYSIMD_FLOAT32_C(   228.03), EASYSIMD_FLOAT32_C(  -794.86) },
      { EASYSIMD_FLOAT32_C(    25.89), EASYSIMD_FLOAT32_C(  -952.74), EASYSIMD_FLOAT32_C(  -776.39), EASYSIMD_FLOAT32_C(   -22.36),
        EASYSIMD_FLOAT32_C(   546.33), EASYSIMD_FLOAT32_C(    84.82), EASYSIMD_FLOAT32_C(   646.17), EASYSIMD_FLOAT32_C(   997.01) },
      { EASYSIMD_FLOAT32_C(206514.94), EASYSIMD_FLOAT32_C(658708.12), EASYSIMD_FLOAT32_C(-598026.88), EASYSIMD_FLOAT32_C( 67513.57),
        EASYSIMD_FLOAT32_C(298555.44), EASYSIMD_FLOAT32_C(   502.76), EASYSIMD_FLOAT32_C( 33359.36), EASYSIMD_FLOAT32_C(   938.30) } },
    { { EASYSIMD_FLOAT32_C(  -525.90), EASYSIMD_FLOAT32_C(  -327.55), EASYSIMD_FLOAT32_C(  -737.45), EASYSIMD_FLOAT32_C(    30.34),
        EASYSIMD_FLOAT32_C(  -234.88), EASYSIMD_FLOAT32_C(   121.33), EASYSIMD_FLOAT32_C(   719.41), EASYSIMD_FLOAT32_C(    22.66) },
      UINT8_C( 90),
      { EASYSIMD_FLOAT32_C(  -416.52), EASYSIMD_FLOAT32_C(   584.45), EASYSIMD_FLOAT32_C(  -777.91), EASYSIMD_FLOAT32_C(   440.02),
        EASYSIMD_FLOAT32_C(   522.75), EASYSIMD_FLOAT32_C(  -646.41), EASYSIMD_FLOAT32_C(   104.10), EASYSIMD_FLOAT32_C(   411.23) },
      { EASYSIMD_FLOAT32_C(  -647.68), EASYSIMD_FLOAT32_C(  -399.02), EASYSIMD_FLOAT32_C(    91.29), EASYSIMD_FLOAT32_C(    76.67),
        EASYSIMD_FLOAT32_C(   829.02), EASYSIMD_FLOAT32_C(   296.43), EASYSIMD_FLOAT32_C(  -897.45), EASYSIMD_FLOAT32_C(   876.28) },
      { EASYSIMD_FLOAT32_C(  -525.90), EASYSIMD_FLOAT32_C(191037.58), EASYSIMD_FLOAT32_C(  -737.45), EASYSIMD_FLOAT32_C(-13273.54),
        EASYSIMD_FLOAT32_C(123612.55), EASYSIMD_FLOAT32_C(   121.33), EASYSIMD_FLOAT32_C(-75788.03), EASYSIMD_FLOAT32_C(    22.66) } },
    { { EASYSIMD_FLOAT32_C(   520.03), EASYSIMD_FLOAT32_C(    80.19), EASYSIMD_FLOAT32_C(   422.61), EASYSIMD_FLOAT32_C(  -395.15),
        EASYSIMD_FLOAT32_C(  -273.64), EASYSIMD_FLOAT32_C(   419.62), EASYSIMD_FLOAT32_C(    78.96), EASYSIMD_FLOAT32_C(   398.81) },
      UINT8_C( 90),
      { EASYSIMD_FLOAT32_C(  -890.70), EASYSIMD_FLOAT32_C(  -836.07), EASYSIMD_FLOAT32_C(  -196.50), EASYSIMD_FLOAT32_C(   828.71),
        EASYSIMD_FLOAT32_C(   186.59), EASYSIMD_FLOAT32_C(   522.83), EASYSIMD_FLOAT32_C(  -587.81), EASYSIMD_FLOAT32_C(  -228.95) },
      { EASYSIMD_FLOAT32_C(   744.93), EASYSIMD_FLOAT32_C(   852.20), EASYSIMD_FLOAT32_C(  -706.20), EASYSIMD_FLOAT32_C(  -901.48),
        EASYSIMD_FLOAT32_C(   -43.70), EASYSIMD_FLOAT32_C(   705.03), EASYSIMD_FLOAT32_C(  -549.16), EASYSIMD_FLOAT32_C(   557.29) },
      { EASYSIMD_FLOAT32_C(   520.03), EASYSIMD_FLOAT32_C( 67896.66), EASYSIMD_FLOAT32_C(   422.61), EASYSIMD_FLOAT32_C(326563.28),
        EASYSIMD_FLOAT32_C( 51014.79), EASYSIMD_FLOAT32_C(   419.62), EASYSIMD_FLOAT32_C( 45864.32), EASYSIMD_FLOAT32_C(   398.81) } },
    { { EASYSIMD_FLOAT32_C(  -203.68), EASYSIMD_FLOAT32_C(   527.50), EASYSIMD_FLOAT32_C(   386.30), EASYSIMD_FLOAT32_C(  -907.26),
        EASYSIMD_FLOAT32_C(   630.05), EASYSIMD_FLOAT32_C(   262.58), EASYSIMD_FLOAT32_C(   612.78), EASYSIMD_FLOAT32_C(  -289.75) },
      UINT8_C(  0),
      { EASYSIMD_FLOAT32_C(  -782.37), EASYSIMD_FLOAT32_C(   436.61), EASYSIMD_FLOAT32_C(  -895.19), EASYSIMD_FLOAT32_C(   296.59),
        EASYSIMD_FLOAT32_C(  -164.59), EASYSIMD_FLOAT32_C(   786.98), EASYSIMD_FLOAT32_C(   405.89), EASYSIMD_FLOAT32_C(    -0.66) },
      { EASYSIMD_FLOAT32_C(  -409.53), EASYSIMD_FLOAT32_C(   234.59), EASYSIMD_FLOAT32_C(  -814.06), EASYSIMD_FLOAT32_C(  -886.69),
        EASYSIMD_FLOAT32_C(   646.78), EASYSIMD_FLOAT32_C(   -43.02), EASYSIMD_FLOAT32_C(   858.23), EASYSIMD_FLOAT32_C(   498.98) },
      { EASYSIMD_FLOAT32_C(  -203.68), EASYSIMD_FLOAT32_C(   527.50), EASYSIMD_FLOAT32_C(   386.30), EASYSIMD_FLOAT32_C(  -907.26),
        EASYSIMD_FLOAT32_C(   630.05), EASYSIMD_FLOAT32_C(   262.58), EASYSIMD_FLOAT32_C(   612.78), EASYSIMD_FLOAT32_C(  -289.75) } },
    { { EASYSIMD_FLOAT32_C(   250.78), EASYSIMD_FLOAT32_C(   956.75), EASYSIMD_FLOAT32_C(  -544.71), EASYSIMD_FLOAT32_C(   -44.19),
        EASYSIMD_FLOAT32_C(  -592.41), EASYSIMD_FLOAT32_C(  -987.43), EASYSIMD_FLOAT32_C(   752.13), EASYSIMD_FLOAT32_C(   935.09) },
      UINT8_C(218),
      { EASYSIMD_FLOAT32_C(   844.87), EASYSIMD_FLOAT32_C(   565.14), EASYSIMD_FLOAT32_C(  -338.54), EASYSIMD_FLOAT32_C(   457.65),
        EASYSIMD_FLOAT32_C(  -724.61), EASYSIMD_FLOAT32_C(   346.65), EASYSIMD_FLOAT32_C(   675.28), EASYSIMD_FLOAT32_C(   712.00) },
      { EASYSIMD_FLOAT32_C(   451.46), EASYSIMD_FLOAT32_C(   -28.13), EASYSIMD_FLOAT32_C(  -452.59), EASYSIMD_FLOAT32_C(   238.44),
        EASYSIMD_FLOAT32_C(  -622.24), EASYSIMD_FLOAT32_C(   546.75), EASYSIMD_FLOAT32_C(   828.91), EASYSIMD_FLOAT32_C(   612.35) },
      { EASYSIMD_FLOAT32_C(   250.78), EASYSIMD_FLOAT32_C(-540725.81), EASYSIMD_FLOAT32_C(  -544.71), EASYSIMD_FLOAT32_C( 20461.99),
        EASYSIMD_FLOAT32_C(-429888.44), EASYSIMD_FLOAT32_C(  -987.43), EASYSIMD_FLOAT32_C(-507069.47), EASYSIMD_FLOAT32_C(-665171.75) } },
    { { EASYSIMD_FLOAT32_C(   732.69), EASYSIMD_FLOAT32_C(   942.22), EASYSIMD_FLOAT32_C(   259.13), EASYSIMD_FLOAT32_C(  -310.33),
        EASYSIMD_FLOAT32_C(   800.45), EASYSIMD_FLOAT32_C(  -241.89), EASYSIMD_FLOAT32_C(   940.46), EASYSIMD_FLOAT32_C(   757.20) },
      UINT8_C( 11),
      { EASYSIMD_FLOAT32_C(  -103.73), EASYSIMD_FLOAT32_C(  -835.21), EASYSIMD_FLOAT32_C(   225.97), EASYSIMD_FLOAT32_C(  -351.60),
        EASYSIMD_FLOAT32_C(  -900.12), EASYSIMD_FLOAT32_C(  -375.15), EASYSIMD_FLOAT32_C(  -506.73), EASYSIMD_FLOAT32_C(   665.02) },
      { EASYSIMD_FLOAT32_C(   286.31), EASYSIMD_FLOAT32_C(   950.92), EASYSIMD_FLOAT32_C(   940.41), EASYSIMD_FLOAT32_C(  -367.05),
        EASYSIMD_FLOAT32_C(   626.21), EASYSIMD_FLOAT32_C(   652.40), EASYSIMD_FLOAT32_C(  -915.59), EASYSIMD_FLOAT32_C(  -401.92) },
      { EASYSIMD_FLOAT32_C( 76288.25), EASYSIMD_FLOAT32_C(787902.50), EASYSIMD_FLOAT32_C(   259.13), EASYSIMD_FLOAT32_C(-109479.07),
        EASYSIMD_FLOAT32_C(   800.45), EASYSIMD_FLOAT32_C(  -241.89), EASYSIMD_FLOAT32_C(   940.46), EASYSIMD_FLOAT32_C(   757.20) } },
    { { EASYSIMD_FLOAT32_C(  -800.19), EASYSIMD_FLOAT32_C(   322.85), EASYSIMD_FLOAT32_C(   -24.16), EASYSIMD_FLOAT32_C(   746.57),
        EASYSIMD_FLOAT32_C(   151.76), EASYSIMD_FLOAT32_C(  -411.81), EASYSIMD_FLOAT32_C(   479.26), EASYSIMD_FLOAT32_C(    93.98) },
      UINT8_C( 50),
      { EASYSIMD_FLOAT32_C(  -831.07), EASYSIMD_FLOAT32_C(  -105.57), EASYSIMD_FLOAT32_C(  -394.56), EASYSIMD_FLOAT32_C(  -890.61),
        EASYSIMD_FLOAT32_C(  -348.37), EASYSIMD_FLOAT32_C(   818.84), EASYSIMD_FLOAT32_C(     5.66), EASYSIMD_FLOAT32_C(  -183.58) },
      { EASYSIMD_FLOAT32_C(    44.81), EASYSIMD_FLOAT32_C(   654.06), EASYSIMD_FLOAT32_C(   -83.71), EASYSIMD_FLOAT32_C(   669.66),
        EASYSIMD_FLOAT32_C(  -852.67), EASYSIMD_FLOAT32_C(  -418.69), EASYSIMD_FLOAT32_C(   -44.03), EASYSIMD_FLOAT32_C(  -901.75) },
      { EASYSIMD_FLOAT32_C(  -800.19), EASYSIMD_FLOAT32_C( 34737.33), EASYSIMD_FLOAT32_C(   -24.16), EASYSIMD_FLOAT32_C(   746.57),
        EASYSIMD_FLOAT32_C( 52015.96), EASYSIMD_FLOAT32_C(336787.81), EASYSIMD_FLOAT32_C(   479.26), EASYSIMD_FLOAT32_C(    93.98) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_fnmadd_ps(a, k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_fnmadd_ps");
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
    easysimd__m256 r = easysimd_mm256_mask_fnmadd_ps(a, k, b, c);

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
test_easysimd_mm256_maskz_fnmadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const uint8_t k;
    const easysimd_float32 b[8];
    const easysimd_float32 c[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -666.73), EASYSIMD_FLOAT32_C(   664.40), EASYSIMD_FLOAT32_C(   721.52), EASYSIMD_FLOAT32_C(   209.45),
        EASYSIMD_FLOAT32_C(    63.30), EASYSIMD_FLOAT32_C(   160.66), EASYSIMD_FLOAT32_C(  -448.31), EASYSIMD_FLOAT32_C(  -312.71) },
      UINT8_C( 75),
      { EASYSIMD_FLOAT32_C(  -679.19), EASYSIMD_FLOAT32_C(  -882.64), EASYSIMD_FLOAT32_C(    53.79), EASYSIMD_FLOAT32_C(   903.12),
        EASYSIMD_FLOAT32_C(  -163.16), EASYSIMD_FLOAT32_C(   459.92), EASYSIMD_FLOAT32_C(   320.55), EASYSIMD_FLOAT32_C(    95.64) },
      { EASYSIMD_FLOAT32_C(   663.68), EASYSIMD_FLOAT32_C(   987.45), EASYSIMD_FLOAT32_C(  -563.05), EASYSIMD_FLOAT32_C(   -56.49),
        EASYSIMD_FLOAT32_C(  -927.92), EASYSIMD_FLOAT32_C(   712.63), EASYSIMD_FLOAT32_C(   245.49), EASYSIMD_FLOAT32_C(   503.48) },
      { EASYSIMD_FLOAT32_C(-452172.66), EASYSIMD_FLOAT32_C(587413.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-189214.97),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(143951.25), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -679.10), EASYSIMD_FLOAT32_C(   458.28), EASYSIMD_FLOAT32_C(  -779.96), EASYSIMD_FLOAT32_C(  -655.78),
        EASYSIMD_FLOAT32_C(  -768.43), EASYSIMD_FLOAT32_C(  -142.87), EASYSIMD_FLOAT32_C(  -322.51), EASYSIMD_FLOAT32_C(   895.97) },
      UINT8_C(161),
      { EASYSIMD_FLOAT32_C(   886.95), EASYSIMD_FLOAT32_C(   -40.73), EASYSIMD_FLOAT32_C(   739.31), EASYSIMD_FLOAT32_C(  -561.36),
        EASYSIMD_FLOAT32_C(   646.56), EASYSIMD_FLOAT32_C(  -132.44), EASYSIMD_FLOAT32_C(  -240.55), EASYSIMD_FLOAT32_C(   763.92) },
      { EASYSIMD_FLOAT32_C(   921.35), EASYSIMD_FLOAT32_C(  -337.43), EASYSIMD_FLOAT32_C(  -399.24), EASYSIMD_FLOAT32_C(   381.28),
        EASYSIMD_FLOAT32_C(   983.11), EASYSIMD_FLOAT32_C(   696.40), EASYSIMD_FLOAT32_C(    44.96), EASYSIMD_FLOAT32_C(   970.57) },
      { EASYSIMD_FLOAT32_C(603249.12), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-18225.30), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-683478.81) } },
    { { EASYSIMD_FLOAT32_C(  -866.65), EASYSIMD_FLOAT32_C(   988.47), EASYSIMD_FLOAT32_C(  -957.35), EASYSIMD_FLOAT32_C(   845.98),
        EASYSIMD_FLOAT32_C(   233.96), EASYSIMD_FLOAT32_C(   546.13), EASYSIMD_FLOAT32_C(  -833.12), EASYSIMD_FLOAT32_C(  -307.77) },
      UINT8_C( 75),
      { EASYSIMD_FLOAT32_C(  -488.90), EASYSIMD_FLOAT32_C(   -76.20), EASYSIMD_FLOAT32_C(  -376.70), EASYSIMD_FLOAT32_C(   188.59),
        EASYSIMD_FLOAT32_C(  -180.23), EASYSIMD_FLOAT32_C(   201.95), EASYSIMD_FLOAT32_C(    75.54), EASYSIMD_FLOAT32_C(   779.04) },
      { EASYSIMD_FLOAT32_C(   -58.74), EASYSIMD_FLOAT32_C(   514.18), EASYSIMD_FLOAT32_C(   425.61), EASYSIMD_FLOAT32_C(   808.83),
        EASYSIMD_FLOAT32_C(  -726.37), EASYSIMD_FLOAT32_C(   189.53), EASYSIMD_FLOAT32_C(   730.18), EASYSIMD_FLOAT32_C(   -63.80) },
      { EASYSIMD_FLOAT32_C(-423763.94), EASYSIMD_FLOAT32_C( 75835.59), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-158734.53),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 63664.07), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   790.29), EASYSIMD_FLOAT32_C(   111.45), EASYSIMD_FLOAT32_C(   -80.68), EASYSIMD_FLOAT32_C(   486.70),
        EASYSIMD_FLOAT32_C(  -843.59), EASYSIMD_FLOAT32_C(  -110.12), EASYSIMD_FLOAT32_C(   620.05), EASYSIMD_FLOAT32_C(  -855.12) },
      UINT8_C( 44),
      { EASYSIMD_FLOAT32_C(   466.03), EASYSIMD_FLOAT32_C(   378.84), EASYSIMD_FLOAT32_C(  -521.34), EASYSIMD_FLOAT32_C(   632.91),
        EASYSIMD_FLOAT32_C(  -928.93), EASYSIMD_FLOAT32_C(  -755.18), EASYSIMD_FLOAT32_C(  -855.99), EASYSIMD_FLOAT32_C(    -5.13) },
      { EASYSIMD_FLOAT32_C(  -131.88), EASYSIMD_FLOAT32_C(   332.61), EASYSIMD_FLOAT32_C(   814.64), EASYSIMD_FLOAT32_C(  -929.93),
        EASYSIMD_FLOAT32_C(  -591.85), EASYSIMD_FLOAT32_C(   593.68), EASYSIMD_FLOAT32_C(    11.34), EASYSIMD_FLOAT32_C(   922.33) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-41247.07), EASYSIMD_FLOAT32_C(-308967.22),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-82566.74), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(    19.29), EASYSIMD_FLOAT32_C(  -179.84), EASYSIMD_FLOAT32_C(  -804.04), EASYSIMD_FLOAT32_C(  -791.18),
        EASYSIMD_FLOAT32_C(  -449.66), EASYSIMD_FLOAT32_C(   132.16), EASYSIMD_FLOAT32_C(   999.11), EASYSIMD_FLOAT32_C(   661.80) },
      UINT8_C(  8),
      { EASYSIMD_FLOAT32_C(   485.81), EASYSIMD_FLOAT32_C(   818.21), EASYSIMD_FLOAT32_C(   -58.64), EASYSIMD_FLOAT32_C(   105.86),
        EASYSIMD_FLOAT32_C(   963.09), EASYSIMD_FLOAT32_C(   873.89), EASYSIMD_FLOAT32_C(  -428.11), EASYSIMD_FLOAT32_C(   341.93) },
      { EASYSIMD_FLOAT32_C(  -647.45), EASYSIMD_FLOAT32_C(  -795.20), EASYSIMD_FLOAT32_C(   413.00), EASYSIMD_FLOAT32_C(  -402.63),
        EASYSIMD_FLOAT32_C(  -651.19), EASYSIMD_FLOAT32_C(  -592.13), EASYSIMD_FLOAT32_C(   465.50), EASYSIMD_FLOAT32_C(   681.42) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 83351.68),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -777.49), EASYSIMD_FLOAT32_C(   535.57), EASYSIMD_FLOAT32_C(  -910.44), EASYSIMD_FLOAT32_C(   816.19),
        EASYSIMD_FLOAT32_C(  -453.09), EASYSIMD_FLOAT32_C(  -988.11), EASYSIMD_FLOAT32_C(  -164.52), EASYSIMD_FLOAT32_C(   367.07) },
      UINT8_C(214),
      { EASYSIMD_FLOAT32_C(    44.30), EASYSIMD_FLOAT32_C(   917.42), EASYSIMD_FLOAT32_C(   340.02), EASYSIMD_FLOAT32_C(    43.42),
        EASYSIMD_FLOAT32_C(   579.22), EASYSIMD_FLOAT32_C(   391.50), EASYSIMD_FLOAT32_C(  -470.77), EASYSIMD_FLOAT32_C(   397.43) },
      { EASYSIMD_FLOAT32_C(  -667.14), EASYSIMD_FLOAT32_C(   635.09), EASYSIMD_FLOAT32_C(   360.52), EASYSIMD_FLOAT32_C(  -793.24),
        EASYSIMD_FLOAT32_C(  -793.03), EASYSIMD_FLOAT32_C(  -297.55), EASYSIMD_FLOAT32_C(  -440.70), EASYSIMD_FLOAT32_C(  -588.23) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-490707.53), EASYSIMD_FLOAT32_C(309928.34), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(261645.75), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-77891.78), EASYSIMD_FLOAT32_C(-146472.86) } },
    { { EASYSIMD_FLOAT32_C(  -884.55), EASYSIMD_FLOAT32_C(   156.68), EASYSIMD_FLOAT32_C(  -239.42), EASYSIMD_FLOAT32_C(  -476.68),
        EASYSIMD_FLOAT32_C(  -377.82), EASYSIMD_FLOAT32_C(  -558.00), EASYSIMD_FLOAT32_C(  -254.17), EASYSIMD_FLOAT32_C(  -842.25) },
      UINT8_C( 10),
      { EASYSIMD_FLOAT32_C(  -437.98), EASYSIMD_FLOAT32_C(  -295.35), EASYSIMD_FLOAT32_C(  -456.55), EASYSIMD_FLOAT32_C(   397.50),
        EASYSIMD_FLOAT32_C(  -928.27), EASYSIMD_FLOAT32_C(  -248.69), EASYSIMD_FLOAT32_C(  -558.19), EASYSIMD_FLOAT32_C(   989.15) },
      { EASYSIMD_FLOAT32_C(  -908.67), EASYSIMD_FLOAT32_C(   485.22), EASYSIMD_FLOAT32_C(   568.36), EASYSIMD_FLOAT32_C(   482.83),
        EASYSIMD_FLOAT32_C(  -985.55), EASYSIMD_FLOAT32_C(   -34.21), EASYSIMD_FLOAT32_C(   815.69), EASYSIMD_FLOAT32_C(   649.53) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 46760.66), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(189963.12),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -673.69), EASYSIMD_FLOAT32_C(  -977.55), EASYSIMD_FLOAT32_C(   856.51), EASYSIMD_FLOAT32_C(    28.76),
        EASYSIMD_FLOAT32_C(  -418.25), EASYSIMD_FLOAT32_C(  -731.72), EASYSIMD_FLOAT32_C(   144.21), EASYSIMD_FLOAT32_C(   738.43) },
      UINT8_C(171),
      { EASYSIMD_FLOAT32_C(   667.54), EASYSIMD_FLOAT32_C(  -639.39), EASYSIMD_FLOAT32_C(   470.86), EASYSIMD_FLOAT32_C(  -586.63),
        EASYSIMD_FLOAT32_C(  -481.64), EASYSIMD_FLOAT32_C(  -997.57), EASYSIMD_FLOAT32_C(   -24.61), EASYSIMD_FLOAT32_C(   223.01) },
      { EASYSIMD_FLOAT32_C(  -454.12), EASYSIMD_FLOAT32_C(  -627.10), EASYSIMD_FLOAT32_C(   294.74), EASYSIMD_FLOAT32_C(   297.19),
        EASYSIMD_FLOAT32_C(  -185.30), EASYSIMD_FLOAT32_C(   283.88), EASYSIMD_FLOAT32_C(   388.52), EASYSIMD_FLOAT32_C(  -700.08) },
      { EASYSIMD_FLOAT32_C(449260.88), EASYSIMD_FLOAT32_C(-625662.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 17168.67),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-729658.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-165377.34) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 a = easysimd_mm256_loadu_ps(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 b = easysimd_mm256_loadu_ps(test_vec[i].b);
    easysimd__m256 c = easysimd_mm256_loadu_ps(test_vec[i].c);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_fnmadd_ps(k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_fnmadd_ps");
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
    easysimd__m256 r = easysimd_mm256_maskz_fnmadd_ps(k, a, b, c);

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
test_easysimd_mm256_mask_fnmadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const uint8_t k;
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -556.38), EASYSIMD_FLOAT64_C(   499.03), EASYSIMD_FLOAT64_C(   -78.86), EASYSIMD_FLOAT64_C(   387.11) },
      UINT8_C(177),
      { EASYSIMD_FLOAT64_C(   284.82), EASYSIMD_FLOAT64_C(  -781.42), EASYSIMD_FLOAT64_C(  -116.64), EASYSIMD_FLOAT64_C(   421.58) },
      { EASYSIMD_FLOAT64_C(  -593.05), EASYSIMD_FLOAT64_C(   142.14), EASYSIMD_FLOAT64_C(   847.44), EASYSIMD_FLOAT64_C(   193.25) },
      { EASYSIMD_FLOAT64_C(157875.10), EASYSIMD_FLOAT64_C(   499.03), EASYSIMD_FLOAT64_C(   -78.86), EASYSIMD_FLOAT64_C(   387.11) } },
    { { EASYSIMD_FLOAT64_C(  -281.44), EASYSIMD_FLOAT64_C(  -637.08), EASYSIMD_FLOAT64_C(   697.03), EASYSIMD_FLOAT64_C(  -386.86) },
      UINT8_C(206),
      { EASYSIMD_FLOAT64_C(   163.20), EASYSIMD_FLOAT64_C(   180.86), EASYSIMD_FLOAT64_C(  -232.97), EASYSIMD_FLOAT64_C(   445.94) },
      { EASYSIMD_FLOAT64_C(  -242.33), EASYSIMD_FLOAT64_C(  -608.58), EASYSIMD_FLOAT64_C(   146.60), EASYSIMD_FLOAT64_C(  -380.62) },
      { EASYSIMD_FLOAT64_C(  -281.44), EASYSIMD_FLOAT64_C(114613.71), EASYSIMD_FLOAT64_C(162533.68), EASYSIMD_FLOAT64_C(172135.73) } },
    { { EASYSIMD_FLOAT64_C(    68.94), EASYSIMD_FLOAT64_C(   189.41), EASYSIMD_FLOAT64_C(   236.59), EASYSIMD_FLOAT64_C(   -51.58) },
      UINT8_C( 59),
      { EASYSIMD_FLOAT64_C(   680.22), EASYSIMD_FLOAT64_C(  -552.55), EASYSIMD_FLOAT64_C(   548.32), EASYSIMD_FLOAT64_C(    67.32) },
      { EASYSIMD_FLOAT64_C(   137.89), EASYSIMD_FLOAT64_C(  -166.86), EASYSIMD_FLOAT64_C(   285.90), EASYSIMD_FLOAT64_C(  -978.75) },
      { EASYSIMD_FLOAT64_C(-46756.48), EASYSIMD_FLOAT64_C(104491.64), EASYSIMD_FLOAT64_C(   236.59), EASYSIMD_FLOAT64_C(  2493.62) } },
    { { EASYSIMD_FLOAT64_C(  -745.28), EASYSIMD_FLOAT64_C(   692.85), EASYSIMD_FLOAT64_C(   163.39), EASYSIMD_FLOAT64_C(  -897.85) },
      UINT8_C(153),
      { EASYSIMD_FLOAT64_C(   881.96), EASYSIMD_FLOAT64_C(  -534.93), EASYSIMD_FLOAT64_C(  -416.86), EASYSIMD_FLOAT64_C(  -504.90) },
      { EASYSIMD_FLOAT64_C(   154.43), EASYSIMD_FLOAT64_C(   746.33), EASYSIMD_FLOAT64_C(   675.96), EASYSIMD_FLOAT64_C(   921.46) },
      { EASYSIMD_FLOAT64_C(657461.58), EASYSIMD_FLOAT64_C(   692.85), EASYSIMD_FLOAT64_C(   163.39), EASYSIMD_FLOAT64_C(-452403.00) } },
    { { EASYSIMD_FLOAT64_C(   192.27), EASYSIMD_FLOAT64_C(  -566.37), EASYSIMD_FLOAT64_C(  -687.12), EASYSIMD_FLOAT64_C(  -661.13) },
      UINT8_C(200),
      { EASYSIMD_FLOAT64_C(   381.82), EASYSIMD_FLOAT64_C(   528.29), EASYSIMD_FLOAT64_C(  -710.40), EASYSIMD_FLOAT64_C(  -669.76) },
      { EASYSIMD_FLOAT64_C(  -844.53), EASYSIMD_FLOAT64_C(   969.82), EASYSIMD_FLOAT64_C(  -222.30), EASYSIMD_FLOAT64_C(   703.79) },
      { EASYSIMD_FLOAT64_C(   192.27), EASYSIMD_FLOAT64_C(  -566.37), EASYSIMD_FLOAT64_C(  -687.12), EASYSIMD_FLOAT64_C(-442094.64) } },
    { { EASYSIMD_FLOAT64_C(    37.14), EASYSIMD_FLOAT64_C(   915.58), EASYSIMD_FLOAT64_C(  -463.07), EASYSIMD_FLOAT64_C(  -676.95) },
      UINT8_C(226),
      { EASYSIMD_FLOAT64_C(  -208.36), EASYSIMD_FLOAT64_C(  -984.10), EASYSIMD_FLOAT64_C(   100.23), EASYSIMD_FLOAT64_C(  -106.20) },
      { EASYSIMD_FLOAT64_C(   -98.00), EASYSIMD_FLOAT64_C(   -17.82), EASYSIMD_FLOAT64_C(   358.87), EASYSIMD_FLOAT64_C(   485.14) },
      { EASYSIMD_FLOAT64_C(    37.14), EASYSIMD_FLOAT64_C(901004.46), EASYSIMD_FLOAT64_C(  -463.07), EASYSIMD_FLOAT64_C(  -676.95) } },
    { { EASYSIMD_FLOAT64_C(   477.28), EASYSIMD_FLOAT64_C(  -486.70), EASYSIMD_FLOAT64_C(   231.47), EASYSIMD_FLOAT64_C(   153.24) },
      UINT8_C( 37),
      { EASYSIMD_FLOAT64_C(  -576.26), EASYSIMD_FLOAT64_C(   586.87), EASYSIMD_FLOAT64_C(  -252.36), EASYSIMD_FLOAT64_C(  -237.38) },
      { EASYSIMD_FLOAT64_C(  -360.12), EASYSIMD_FLOAT64_C(  -870.53), EASYSIMD_FLOAT64_C(  -709.10), EASYSIMD_FLOAT64_C(   -70.52) },
      { EASYSIMD_FLOAT64_C(274677.25), EASYSIMD_FLOAT64_C(  -486.70), EASYSIMD_FLOAT64_C( 57704.67), EASYSIMD_FLOAT64_C(   153.24) } },
    { { EASYSIMD_FLOAT64_C(  -540.29), EASYSIMD_FLOAT64_C(  -553.63), EASYSIMD_FLOAT64_C(  -100.70), EASYSIMD_FLOAT64_C(   237.41) },
      UINT8_C(176),
      { EASYSIMD_FLOAT64_C(   936.44), EASYSIMD_FLOAT64_C(   152.99), EASYSIMD_FLOAT64_C(  -312.91), EASYSIMD_FLOAT64_C(  -740.51) },
      { EASYSIMD_FLOAT64_C(    89.83), EASYSIMD_FLOAT64_C(   478.74), EASYSIMD_FLOAT64_C(  -724.61), EASYSIMD_FLOAT64_C(  -809.95) },
      { EASYSIMD_FLOAT64_C(  -540.29), EASYSIMD_FLOAT64_C(  -553.63), EASYSIMD_FLOAT64_C(  -100.70), EASYSIMD_FLOAT64_C(   237.41) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_fnmadd_pd(a, k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_fnmadd_pd");
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
    easysimd__m256d r = easysimd_mm256_mask_fnmadd_pd(a, k, b, c);

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
test_easysimd_mm256_maskz_fnmadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const uint8_t k;
    const easysimd_float64 b[4];
    const easysimd_float64 c[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   933.18), EASYSIMD_FLOAT64_C(   381.68), EASYSIMD_FLOAT64_C(   359.69), EASYSIMD_FLOAT64_C(   585.60) },
      UINT8_C(183),
      { EASYSIMD_FLOAT64_C(  -502.52), EASYSIMD_FLOAT64_C(   838.42), EASYSIMD_FLOAT64_C(   394.49), EASYSIMD_FLOAT64_C(   254.58) },
      { EASYSIMD_FLOAT64_C(   139.64), EASYSIMD_FLOAT64_C(  -149.91), EASYSIMD_FLOAT64_C(   931.05), EASYSIMD_FLOAT64_C(  -188.04) },
      { EASYSIMD_FLOAT64_C(469081.25), EASYSIMD_FLOAT64_C(-320158.06), EASYSIMD_FLOAT64_C(-140963.06), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -305.50), EASYSIMD_FLOAT64_C(   475.84), EASYSIMD_FLOAT64_C(   539.06), EASYSIMD_FLOAT64_C(  -580.64) },
      UINT8_C(150),
      { EASYSIMD_FLOAT64_C(   163.04), EASYSIMD_FLOAT64_C(   350.35), EASYSIMD_FLOAT64_C(  -666.20), EASYSIMD_FLOAT64_C(   -43.83) },
      { EASYSIMD_FLOAT64_C(   828.53), EASYSIMD_FLOAT64_C(    67.39), EASYSIMD_FLOAT64_C(    47.86), EASYSIMD_FLOAT64_C(   948.28) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-166643.15), EASYSIMD_FLOAT64_C(359169.63), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   522.67), EASYSIMD_FLOAT64_C(   843.67), EASYSIMD_FLOAT64_C(  -440.76), EASYSIMD_FLOAT64_C(   350.88) },
      UINT8_C(121),
      { EASYSIMD_FLOAT64_C(  -507.59), EASYSIMD_FLOAT64_C(  -267.44), EASYSIMD_FLOAT64_C(  -724.44), EASYSIMD_FLOAT64_C(  -921.98) },
      { EASYSIMD_FLOAT64_C(  -760.96), EASYSIMD_FLOAT64_C(  -226.96), EASYSIMD_FLOAT64_C(   916.44), EASYSIMD_FLOAT64_C(   633.52) },
      { EASYSIMD_FLOAT64_C(264541.11), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(324137.86) } },
    { { EASYSIMD_FLOAT64_C(  -972.38), EASYSIMD_FLOAT64_C(    56.07), EASYSIMD_FLOAT64_C(  -516.39), EASYSIMD_FLOAT64_C(   958.67) },
      UINT8_C(184),
      { EASYSIMD_FLOAT64_C(   178.11), EASYSIMD_FLOAT64_C(   434.51), EASYSIMD_FLOAT64_C(   407.09), EASYSIMD_FLOAT64_C(   597.47) },
      { EASYSIMD_FLOAT64_C(   141.18), EASYSIMD_FLOAT64_C(  -429.87), EASYSIMD_FLOAT64_C(   -52.18), EASYSIMD_FLOAT64_C(   474.98) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-572301.58) } },
    { { EASYSIMD_FLOAT64_C(   526.30), EASYSIMD_FLOAT64_C(  -223.65), EASYSIMD_FLOAT64_C(  -457.64), EASYSIMD_FLOAT64_C(  -425.84) },
      UINT8_C(237),
      { EASYSIMD_FLOAT64_C(  -934.97), EASYSIMD_FLOAT64_C(  -582.17), EASYSIMD_FLOAT64_C(   283.87), EASYSIMD_FLOAT64_C(   415.91) },
      { EASYSIMD_FLOAT64_C(   333.69), EASYSIMD_FLOAT64_C(   776.28), EASYSIMD_FLOAT64_C(  -851.53), EASYSIMD_FLOAT64_C(   609.26) },
      { EASYSIMD_FLOAT64_C(492408.40), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(129058.74), EASYSIMD_FLOAT64_C(177720.37) } },
    { { EASYSIMD_FLOAT64_C(   854.30), EASYSIMD_FLOAT64_C(  -612.50), EASYSIMD_FLOAT64_C(  -617.70), EASYSIMD_FLOAT64_C(   770.73) },
      UINT8_C( 77),
      { EASYSIMD_FLOAT64_C(  -590.08), EASYSIMD_FLOAT64_C(  -173.19), EASYSIMD_FLOAT64_C(  -495.36), EASYSIMD_FLOAT64_C(  -631.41) },
      { EASYSIMD_FLOAT64_C(  -305.16), EASYSIMD_FLOAT64_C(   682.75), EASYSIMD_FLOAT64_C(   803.10), EASYSIMD_FLOAT64_C(  -898.06) },
      { EASYSIMD_FLOAT64_C(503800.18), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(-305180.77), EASYSIMD_FLOAT64_C(485748.57) } },
    { { EASYSIMD_FLOAT64_C(   280.23), EASYSIMD_FLOAT64_C(   -55.72), EASYSIMD_FLOAT64_C(  -327.93), EASYSIMD_FLOAT64_C(  -771.95) },
      UINT8_C(135),
      { EASYSIMD_FLOAT64_C(  -801.63), EASYSIMD_FLOAT64_C(     4.40), EASYSIMD_FLOAT64_C(   -38.39), EASYSIMD_FLOAT64_C(  -227.48) },
      { EASYSIMD_FLOAT64_C(   729.03), EASYSIMD_FLOAT64_C(    26.64), EASYSIMD_FLOAT64_C(   190.35), EASYSIMD_FLOAT64_C(    12.90) },
      { EASYSIMD_FLOAT64_C(225369.80), EASYSIMD_FLOAT64_C(   271.81), EASYSIMD_FLOAT64_C(-12398.88), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -557.45), EASYSIMD_FLOAT64_C(  -475.96), EASYSIMD_FLOAT64_C(  -210.82), EASYSIMD_FLOAT64_C(  -408.99) },
      UINT8_C(199),
      { EASYSIMD_FLOAT64_C(  -356.52), EASYSIMD_FLOAT64_C(   -21.48), EASYSIMD_FLOAT64_C(  -484.40), EASYSIMD_FLOAT64_C(  -585.78) },
      { EASYSIMD_FLOAT64_C(    -0.46), EASYSIMD_FLOAT64_C(   -74.49), EASYSIMD_FLOAT64_C(   241.02), EASYSIMD_FLOAT64_C(   504.18) },
      { EASYSIMD_FLOAT64_C(-198742.53), EASYSIMD_FLOAT64_C(-10298.11), EASYSIMD_FLOAT64_C(-101880.19), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d a = easysimd_mm256_loadu_pd(test_vec[i].a);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d b = easysimd_mm256_loadu_pd(test_vec[i].b);
    easysimd__m256d c = easysimd_mm256_loadu_pd(test_vec[i].c);
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_fnmadd_pd(k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_fnmadd_pd");
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
    easysimd__m256d r = easysimd_mm256_maskz_fnmadd_pd(k, a, b, c);

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
test_easysimd_mm512_fnmadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 c[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   -61.24), EASYSIMD_FLOAT32_C(    66.05), EASYSIMD_FLOAT32_C(   -37.34), EASYSIMD_FLOAT32_C(   -97.14),
        EASYSIMD_FLOAT32_C(    46.24), EASYSIMD_FLOAT32_C(    -6.81), EASYSIMD_FLOAT32_C(    13.43), EASYSIMD_FLOAT32_C(   -17.56),
        EASYSIMD_FLOAT32_C(   -58.55), EASYSIMD_FLOAT32_C(   -25.60), EASYSIMD_FLOAT32_C(   -35.79), EASYSIMD_FLOAT32_C(    89.11),
        EASYSIMD_FLOAT32_C(    42.18), EASYSIMD_FLOAT32_C(    22.19), EASYSIMD_FLOAT32_C(    87.14), EASYSIMD_FLOAT32_C(   -55.50) },
      { EASYSIMD_FLOAT32_C(    51.65), EASYSIMD_FLOAT32_C(   -43.85), EASYSIMD_FLOAT32_C(   -44.69), EASYSIMD_FLOAT32_C(     2.09),
        EASYSIMD_FLOAT32_C(    37.82), EASYSIMD_FLOAT32_C(     8.27), EASYSIMD_FLOAT32_C(   -31.95), EASYSIMD_FLOAT32_C(    70.84),
        EASYSIMD_FLOAT32_C(   -62.34), EASYSIMD_FLOAT32_C(    50.49), EASYSIMD_FLOAT32_C(   -46.36), EASYSIMD_FLOAT32_C(   -70.84),
        EASYSIMD_FLOAT32_C(    55.23), EASYSIMD_FLOAT32_C(   -80.94), EASYSIMD_FLOAT32_C(    23.06), EASYSIMD_FLOAT32_C(    93.98) },
      { EASYSIMD_FLOAT32_C(    85.11), EASYSIMD_FLOAT32_C(    85.72), EASYSIMD_FLOAT32_C(    96.84), EASYSIMD_FLOAT32_C(    31.35),
        EASYSIMD_FLOAT32_C(   -21.08), EASYSIMD_FLOAT32_C(    10.28), EASYSIMD_FLOAT32_C(   -86.21), EASYSIMD_FLOAT32_C(    20.37),
        EASYSIMD_FLOAT32_C(    84.68), EASYSIMD_FLOAT32_C(   -22.01), EASYSIMD_FLOAT32_C(     9.48), EASYSIMD_FLOAT32_C(    26.86),
        EASYSIMD_FLOAT32_C(   -99.81), EASYSIMD_FLOAT32_C(    -3.39), EASYSIMD_FLOAT32_C(    71.36), EASYSIMD_FLOAT32_C(    51.84) },
      { EASYSIMD_FLOAT32_C(  3248.16), EASYSIMD_FLOAT32_C(  2982.01), EASYSIMD_FLOAT32_C( -1571.88), EASYSIMD_FLOAT32_C(   234.37),
        EASYSIMD_FLOAT32_C( -1769.88), EASYSIMD_FLOAT32_C(    66.60), EASYSIMD_FLOAT32_C(   342.88), EASYSIMD_FLOAT32_C(  1264.32),
        EASYSIMD_FLOAT32_C( -3565.33), EASYSIMD_FLOAT32_C(  1270.53), EASYSIMD_FLOAT32_C( -1649.74), EASYSIMD_FLOAT32_C(  6339.41),
        EASYSIMD_FLOAT32_C( -2429.41), EASYSIMD_FLOAT32_C(  1792.67), EASYSIMD_FLOAT32_C( -1938.09), EASYSIMD_FLOAT32_C(  5267.73) } },
    { { EASYSIMD_FLOAT32_C(    52.77), EASYSIMD_FLOAT32_C(   -73.33), EASYSIMD_FLOAT32_C(   -46.07), EASYSIMD_FLOAT32_C(    -9.41),
        EASYSIMD_FLOAT32_C(    34.95), EASYSIMD_FLOAT32_C(    21.97), EASYSIMD_FLOAT32_C(   -38.57), EASYSIMD_FLOAT32_C(    72.60),
        EASYSIMD_FLOAT32_C(   -27.54), EASYSIMD_FLOAT32_C(    15.07), EASYSIMD_FLOAT32_C(   -98.24), EASYSIMD_FLOAT32_C(   -72.31),
        EASYSIMD_FLOAT32_C(    34.13), EASYSIMD_FLOAT32_C(    24.83), EASYSIMD_FLOAT32_C(   -78.32), EASYSIMD_FLOAT32_C(    19.24) },
      { EASYSIMD_FLOAT32_C(    10.55), EASYSIMD_FLOAT32_C(   -81.48), EASYSIMD_FLOAT32_C(   -49.41), EASYSIMD_FLOAT32_C(    89.47),
        EASYSIMD_FLOAT32_C(    28.80), EASYSIMD_FLOAT32_C(   -35.62), EASYSIMD_FLOAT32_C(     9.83), EASYSIMD_FLOAT32_C(    13.47),
        EASYSIMD_FLOAT32_C(    42.37), EASYSIMD_FLOAT32_C(   -80.69), EASYSIMD_FLOAT32_C(   -59.67), EASYSIMD_FLOAT32_C(    42.56),
        EASYSIMD_FLOAT32_C(    15.92), EASYSIMD_FLOAT32_C(   -88.31), EASYSIMD_FLOAT32_C(    -5.60), EASYSIMD_FLOAT32_C(   -31.31) },
      { EASYSIMD_FLOAT32_C(   -61.63), EASYSIMD_FLOAT32_C(    48.33), EASYSIMD_FLOAT32_C(    59.28), EASYSIMD_FLOAT32_C(    73.32),
        EASYSIMD_FLOAT32_C(   -29.70), EASYSIMD_FLOAT32_C(   -79.29), EASYSIMD_FLOAT32_C(    45.92), EASYSIMD_FLOAT32_C(    42.77),
        EASYSIMD_FLOAT32_C(    35.79), EASYSIMD_FLOAT32_C(    47.68), EASYSIMD_FLOAT32_C(    70.46), EASYSIMD_FLOAT32_C(   -30.08),
        EASYSIMD_FLOAT32_C(   -27.50), EASYSIMD_FLOAT32_C(    92.14), EASYSIMD_FLOAT32_C(    89.16), EASYSIMD_FLOAT32_C(    83.05) },
      { EASYSIMD_FLOAT32_C(  -618.35), EASYSIMD_FLOAT32_C( -5926.60), EASYSIMD_FLOAT32_C( -2217.04), EASYSIMD_FLOAT32_C(   915.23),
        EASYSIMD_FLOAT32_C( -1036.26), EASYSIMD_FLOAT32_C(   703.28), EASYSIMD_FLOAT32_C(   425.06), EASYSIMD_FLOAT32_C(  -935.15),
        EASYSIMD_FLOAT32_C(  1202.66), EASYSIMD_FLOAT32_C(  1263.68), EASYSIMD_FLOAT32_C( -5791.52), EASYSIMD_FLOAT32_C(  3047.43),
        EASYSIMD_FLOAT32_C(  -570.85), EASYSIMD_FLOAT32_C(  2284.88), EASYSIMD_FLOAT32_C(  -349.43), EASYSIMD_FLOAT32_C(   685.45) } },
    { { EASYSIMD_FLOAT32_C(   -89.34), EASYSIMD_FLOAT32_C(   -60.25), EASYSIMD_FLOAT32_C(    72.52), EASYSIMD_FLOAT32_C(    39.45),
        EASYSIMD_FLOAT32_C(     4.13), EASYSIMD_FLOAT32_C(   -17.65), EASYSIMD_FLOAT32_C(   -47.07), EASYSIMD_FLOAT32_C(   -53.49),
        EASYSIMD_FLOAT32_C(     1.66), EASYSIMD_FLOAT32_C(    -6.74), EASYSIMD_FLOAT32_C(    89.07), EASYSIMD_FLOAT32_C(   -82.41),
        EASYSIMD_FLOAT32_C(     4.96), EASYSIMD_FLOAT32_C(   -16.53), EASYSIMD_FLOAT32_C(   -13.72), EASYSIMD_FLOAT32_C(    43.33) },
      { EASYSIMD_FLOAT32_C(   -68.20), EASYSIMD_FLOAT32_C(   -54.44), EASYSIMD_FLOAT32_C(    16.64), EASYSIMD_FLOAT32_C(     2.10),
        EASYSIMD_FLOAT32_C(   -33.73), EASYSIMD_FLOAT32_C(   -37.44), EASYSIMD_FLOAT32_C(   -55.13), EASYSIMD_FLOAT32_C(   -97.94),
        EASYSIMD_FLOAT32_C(   -89.76), EASYSIMD_FLOAT32_C(   -84.67), EASYSIMD_FLOAT32_C(   -28.02), EASYSIMD_FLOAT32_C(   -17.26),
        EASYSIMD_FLOAT32_C(   -92.53), EASYSIMD_FLOAT32_C(   -38.86), EASYSIMD_FLOAT32_C(   -34.20), EASYSIMD_FLOAT32_C(   -81.88) },
      { EASYSIMD_FLOAT32_C(     0.89), EASYSIMD_FLOAT32_C(   -61.69), EASYSIMD_FLOAT32_C(    57.58), EASYSIMD_FLOAT32_C(   -94.97),
        EASYSIMD_FLOAT32_C(    20.67), EASYSIMD_FLOAT32_C(   -89.49), EASYSIMD_FLOAT32_C(   -48.46), EASYSIMD_FLOAT32_C(   -77.67),
        EASYSIMD_FLOAT32_C(     3.77), EASYSIMD_FLOAT32_C(   -59.39), EASYSIMD_FLOAT32_C(   -60.08), EASYSIMD_FLOAT32_C(   -91.27),
        EASYSIMD_FLOAT32_C(    24.08), EASYSIMD_FLOAT32_C(    26.19), EASYSIMD_FLOAT32_C(    52.05), EASYSIMD_FLOAT32_C(    55.88) },
      { EASYSIMD_FLOAT32_C( -6092.10), EASYSIMD_FLOAT32_C( -3341.70), EASYSIMD_FLOAT32_C( -1149.15), EASYSIMD_FLOAT32_C(  -177.82),
        EASYSIMD_FLOAT32_C(   159.97), EASYSIMD_FLOAT32_C(  -750.31), EASYSIMD_FLOAT32_C( -2643.43), EASYSIMD_FLOAT32_C( -5316.48),
        EASYSIMD_FLOAT32_C(   152.77), EASYSIMD_FLOAT32_C(  -630.07), EASYSIMD_FLOAT32_C(  2435.66), EASYSIMD_FLOAT32_C( -1513.67),
        EASYSIMD_FLOAT32_C(   483.03), EASYSIMD_FLOAT32_C(  -616.17), EASYSIMD_FLOAT32_C(  -417.17), EASYSIMD_FLOAT32_C(  3603.74) } },
    { { EASYSIMD_FLOAT32_C(    71.75), EASYSIMD_FLOAT32_C(   -31.30), EASYSIMD_FLOAT32_C(   -42.02), EASYSIMD_FLOAT32_C(   -61.98),
        EASYSIMD_FLOAT32_C(    31.26), EASYSIMD_FLOAT32_C(     2.85), EASYSIMD_FLOAT32_C(   -59.92), EASYSIMD_FLOAT32_C(    41.49),
        EASYSIMD_FLOAT32_C(    18.19), EASYSIMD_FLOAT32_C(    12.07), EASYSIMD_FLOAT32_C(   -75.76), EASYSIMD_FLOAT32_C(    25.65),
        EASYSIMD_FLOAT32_C(    73.21), EASYSIMD_FLOAT32_C(    -9.97), EASYSIMD_FLOAT32_C(    43.78), EASYSIMD_FLOAT32_C(   -25.90) },
      { EASYSIMD_FLOAT32_C(    28.35), EASYSIMD_FLOAT32_C(     1.36), EASYSIMD_FLOAT32_C(   -20.87), EASYSIMD_FLOAT32_C(   -50.99),
        EASYSIMD_FLOAT32_C(    11.86), EASYSIMD_FLOAT32_C(    30.66), EASYSIMD_FLOAT32_C(   -28.66), EASYSIMD_FLOAT32_C(   -84.37),
        EASYSIMD_FLOAT32_C(    71.27), EASYSIMD_FLOAT32_C(    11.26), EASYSIMD_FLOAT32_C(   -75.64), EASYSIMD_FLOAT32_C(    -4.65),
        EASYSIMD_FLOAT32_C(   -62.54), EASYSIMD_FLOAT32_C(    76.41), EASYSIMD_FLOAT32_C(   -48.78), EASYSIMD_FLOAT32_C(   -90.79) },
      { EASYSIMD_FLOAT32_C(   -54.89), EASYSIMD_FLOAT32_C(     9.21), EASYSIMD_FLOAT32_C(   -52.77), EASYSIMD_FLOAT32_C(    76.37),
        EASYSIMD_FLOAT32_C(   -87.94), EASYSIMD_FLOAT32_C(   -12.68), EASYSIMD_FLOAT32_C(    17.86), EASYSIMD_FLOAT32_C(    30.24),
        EASYSIMD_FLOAT32_C(    99.38), EASYSIMD_FLOAT32_C(    42.10), EASYSIMD_FLOAT32_C(   -44.10), EASYSIMD_FLOAT32_C(    72.59),
        EASYSIMD_FLOAT32_C(   -67.87), EASYSIMD_FLOAT32_C(    99.68), EASYSIMD_FLOAT32_C(   -53.31), EASYSIMD_FLOAT32_C(    60.48) },
      { EASYSIMD_FLOAT32_C( -2089.00), EASYSIMD_FLOAT32_C(    51.78), EASYSIMD_FLOAT32_C(  -929.73), EASYSIMD_FLOAT32_C( -3083.99),
        EASYSIMD_FLOAT32_C(  -458.68), EASYSIMD_FLOAT32_C(  -100.06), EASYSIMD_FLOAT32_C( -1699.45), EASYSIMD_FLOAT32_C(  3530.75),
        EASYSIMD_FLOAT32_C( -1197.02), EASYSIMD_FLOAT32_C(   -93.81), EASYSIMD_FLOAT32_C( -5774.59), EASYSIMD_FLOAT32_C(   191.86),
        EASYSIMD_FLOAT32_C(  4510.68), EASYSIMD_FLOAT32_C(   861.49), EASYSIMD_FLOAT32_C(  2082.28), EASYSIMD_FLOAT32_C( -2290.98) } },
    { { EASYSIMD_FLOAT32_C(     1.03), EASYSIMD_FLOAT32_C(    25.82), EASYSIMD_FLOAT32_C(   -90.51), EASYSIMD_FLOAT32_C(   -87.11),
        EASYSIMD_FLOAT32_C(   -43.51), EASYSIMD_FLOAT32_C(   -19.16), EASYSIMD_FLOAT32_C(   -71.47), EASYSIMD_FLOAT32_C(   -72.24),
        EASYSIMD_FLOAT32_C(    92.10), EASYSIMD_FLOAT32_C(   -47.11), EASYSIMD_FLOAT32_C(    23.11), EASYSIMD_FLOAT32_C(   -70.45),
        EASYSIMD_FLOAT32_C(   -70.70), EASYSIMD_FLOAT32_C(    74.33), EASYSIMD_FLOAT32_C(   -61.24), EASYSIMD_FLOAT32_C(   -25.59) },
      { EASYSIMD_FLOAT32_C(   -16.46), EASYSIMD_FLOAT32_C(   -14.01), EASYSIMD_FLOAT32_C(   -49.22), EASYSIMD_FLOAT32_C(    -4.41),
        EASYSIMD_FLOAT32_C(    73.31), EASYSIMD_FLOAT32_C(    68.64), EASYSIMD_FLOAT32_C(   -74.16), EASYSIMD_FLOAT32_C(    72.69),
        EASYSIMD_FLOAT32_C(    10.74), EASYSIMD_FLOAT32_C(   -18.27), EASYSIMD_FLOAT32_C(    45.28), EASYSIMD_FLOAT32_C(    42.87),
        EASYSIMD_FLOAT32_C(   -18.59), EASYSIMD_FLOAT32_C(    91.98), EASYSIMD_FLOAT32_C(     3.34), EASYSIMD_FLOAT32_C(    82.44) },
      { EASYSIMD_FLOAT32_C(    17.80), EASYSIMD_FLOAT32_C(    12.84), EASYSIMD_FLOAT32_C(    95.33), EASYSIMD_FLOAT32_C(    74.28),
        EASYSIMD_FLOAT32_C(    93.67), EASYSIMD_FLOAT32_C(   -76.14), EASYSIMD_FLOAT32_C(   -97.96), EASYSIMD_FLOAT32_C(    85.77),
        EASYSIMD_FLOAT32_C(   -23.25), EASYSIMD_FLOAT32_C(    25.15), EASYSIMD_FLOAT32_C(   -84.68), EASYSIMD_FLOAT32_C(     6.05),
        EASYSIMD_FLOAT32_C(    -0.52), EASYSIMD_FLOAT32_C(   -45.92), EASYSIMD_FLOAT32_C(    80.46), EASYSIMD_FLOAT32_C(    83.01) },
      { EASYSIMD_FLOAT32_C(    34.75), EASYSIMD_FLOAT32_C(   374.58), EASYSIMD_FLOAT32_C( -4359.57), EASYSIMD_FLOAT32_C(  -309.88),
        EASYSIMD_FLOAT32_C(  3283.39), EASYSIMD_FLOAT32_C(  1239.00), EASYSIMD_FLOAT32_C( -5398.18), EASYSIMD_FLOAT32_C(  5336.90),
        EASYSIMD_FLOAT32_C( -1012.40), EASYSIMD_FLOAT32_C(  -835.55), EASYSIMD_FLOAT32_C( -1131.10), EASYSIMD_FLOAT32_C(  3026.24),
        EASYSIMD_FLOAT32_C( -1314.83), EASYSIMD_FLOAT32_C( -6882.79), EASYSIMD_FLOAT32_C(   285.00), EASYSIMD_FLOAT32_C(  2192.65) } },
    { { EASYSIMD_FLOAT32_C(    40.08), EASYSIMD_FLOAT32_C(   -68.77), EASYSIMD_FLOAT32_C(   -21.40), EASYSIMD_FLOAT32_C(    13.39),
        EASYSIMD_FLOAT32_C(    99.87), EASYSIMD_FLOAT32_C(     4.44), EASYSIMD_FLOAT32_C(   -13.92), EASYSIMD_FLOAT32_C(    10.61),
        EASYSIMD_FLOAT32_C(    86.17), EASYSIMD_FLOAT32_C(   -68.63), EASYSIMD_FLOAT32_C(   -46.52), EASYSIMD_FLOAT32_C(   -32.42),
        EASYSIMD_FLOAT32_C(   -76.66), EASYSIMD_FLOAT32_C(    56.82), EASYSIMD_FLOAT32_C(   -49.98), EASYSIMD_FLOAT32_C(    41.14) },
      { EASYSIMD_FLOAT32_C(   -30.34), EASYSIMD_FLOAT32_C(   -54.64), EASYSIMD_FLOAT32_C(    15.43), EASYSIMD_FLOAT32_C(   -36.67),
        EASYSIMD_FLOAT32_C(   -30.78), EASYSIMD_FLOAT32_C(    17.47), EASYSIMD_FLOAT32_C(   -50.90), EASYSIMD_FLOAT32_C(    45.97),
        EASYSIMD_FLOAT32_C(   -57.39), EASYSIMD_FLOAT32_C(   -35.58), EASYSIMD_FLOAT32_C(   -47.99), EASYSIMD_FLOAT32_C(    42.09),
        EASYSIMD_FLOAT32_C(    18.50), EASYSIMD_FLOAT32_C(   -67.53), EASYSIMD_FLOAT32_C(    25.10), EASYSIMD_FLOAT32_C(   -41.42) },
      { EASYSIMD_FLOAT32_C(   -36.29), EASYSIMD_FLOAT32_C(   -96.30), EASYSIMD_FLOAT32_C(    71.97), EASYSIMD_FLOAT32_C(   -36.42),
        EASYSIMD_FLOAT32_C(     8.14), EASYSIMD_FLOAT32_C(   -41.94), EASYSIMD_FLOAT32_C(    74.19), EASYSIMD_FLOAT32_C(    -5.68),
        EASYSIMD_FLOAT32_C(   -10.58), EASYSIMD_FLOAT32_C(   -72.33), EASYSIMD_FLOAT32_C(    61.90), EASYSIMD_FLOAT32_C(    12.77),
        EASYSIMD_FLOAT32_C(    84.49), EASYSIMD_FLOAT32_C(   -88.07), EASYSIMD_FLOAT32_C(   -46.09), EASYSIMD_FLOAT32_C(   -45.85) },
      { EASYSIMD_FLOAT32_C(  1179.74), EASYSIMD_FLOAT32_C( -3853.89), EASYSIMD_FLOAT32_C(   402.17), EASYSIMD_FLOAT32_C(   454.59),
        EASYSIMD_FLOAT32_C(  3082.14), EASYSIMD_FLOAT32_C(  -119.51), EASYSIMD_FLOAT32_C(  -634.34), EASYSIMD_FLOAT32_C(  -493.42),
        EASYSIMD_FLOAT32_C(  4934.72), EASYSIMD_FLOAT32_C( -2514.19), EASYSIMD_FLOAT32_C( -2170.59), EASYSIMD_FLOAT32_C(  1377.33),
        EASYSIMD_FLOAT32_C(  1502.70), EASYSIMD_FLOAT32_C(  3748.98), EASYSIMD_FLOAT32_C(  1208.41), EASYSIMD_FLOAT32_C(  1658.17) } },
    { { EASYSIMD_FLOAT32_C(   -42.71), EASYSIMD_FLOAT32_C(    69.33), EASYSIMD_FLOAT32_C(    17.48), EASYSIMD_FLOAT32_C(    26.51),
        EASYSIMD_FLOAT32_C(   -13.20), EASYSIMD_FLOAT32_C(    66.58), EASYSIMD_FLOAT32_C(   -27.52), EASYSIMD_FLOAT32_C(    29.41),
        EASYSIMD_FLOAT32_C(   -69.00), EASYSIMD_FLOAT32_C(    24.49), EASYSIMD_FLOAT32_C(   -28.50), EASYSIMD_FLOAT32_C(    49.50),
        EASYSIMD_FLOAT32_C(    56.96), EASYSIMD_FLOAT32_C(    96.60), EASYSIMD_FLOAT32_C(   -91.91), EASYSIMD_FLOAT32_C(   -79.34) },
      { EASYSIMD_FLOAT32_C(   -99.70), EASYSIMD_FLOAT32_C(    80.06), EASYSIMD_FLOAT32_C(   -15.76), EASYSIMD_FLOAT32_C(     8.44),
        EASYSIMD_FLOAT32_C(   -61.89), EASYSIMD_FLOAT32_C(   -41.56), EASYSIMD_FLOAT32_C(   -97.24), EASYSIMD_FLOAT32_C(    27.54),
        EASYSIMD_FLOAT32_C(   -13.89), EASYSIMD_FLOAT32_C(    64.67), EASYSIMD_FLOAT32_C(   -59.70), EASYSIMD_FLOAT32_C(   -29.40),
        EASYSIMD_FLOAT32_C(    76.59), EASYSIMD_FLOAT32_C(    -5.79), EASYSIMD_FLOAT32_C(    24.75), EASYSIMD_FLOAT32_C(   -66.12) },
      { EASYSIMD_FLOAT32_C(   -36.45), EASYSIMD_FLOAT32_C(   -57.76), EASYSIMD_FLOAT32_C(    60.39), EASYSIMD_FLOAT32_C(    50.35),
        EASYSIMD_FLOAT32_C(   -91.18), EASYSIMD_FLOAT32_C(   -67.13), EASYSIMD_FLOAT32_C(   -20.24), EASYSIMD_FLOAT32_C(   -60.18),
        EASYSIMD_FLOAT32_C(    57.36), EASYSIMD_FLOAT32_C(    51.26), EASYSIMD_FLOAT32_C(    89.32), EASYSIMD_FLOAT32_C(    14.31),
        EASYSIMD_FLOAT32_C(    47.86), EASYSIMD_FLOAT32_C(    97.41), EASYSIMD_FLOAT32_C(    34.98), EASYSIMD_FLOAT32_C(    48.16) },
      { EASYSIMD_FLOAT32_C( -4294.64), EASYSIMD_FLOAT32_C( -5608.32), EASYSIMD_FLOAT32_C(   335.87), EASYSIMD_FLOAT32_C(  -173.39),
        EASYSIMD_FLOAT32_C(  -908.13), EASYSIMD_FLOAT32_C(  2699.94), EASYSIMD_FLOAT32_C( -2696.28), EASYSIMD_FLOAT32_C(  -870.13),
        EASYSIMD_FLOAT32_C(  -901.05), EASYSIMD_FLOAT32_C( -1532.51), EASYSIMD_FLOAT32_C( -1612.13), EASYSIMD_FLOAT32_C(  1469.61),
        EASYSIMD_FLOAT32_C( -4314.71), EASYSIMD_FLOAT32_C(   656.72), EASYSIMD_FLOAT32_C(  2309.75), EASYSIMD_FLOAT32_C( -5197.80) } },
    { { EASYSIMD_FLOAT32_C(    77.47), EASYSIMD_FLOAT32_C(   -80.78), EASYSIMD_FLOAT32_C(   -43.40), EASYSIMD_FLOAT32_C(   -84.42),
        EASYSIMD_FLOAT32_C(   -22.34), EASYSIMD_FLOAT32_C(   -40.64), EASYSIMD_FLOAT32_C(    43.12), EASYSIMD_FLOAT32_C(    63.76),
        EASYSIMD_FLOAT32_C(   -75.97), EASYSIMD_FLOAT32_C(    83.43), EASYSIMD_FLOAT32_C(   -65.64), EASYSIMD_FLOAT32_C(   -99.38),
        EASYSIMD_FLOAT32_C(   -22.36), EASYSIMD_FLOAT32_C(    59.12), EASYSIMD_FLOAT32_C(   -65.50), EASYSIMD_FLOAT32_C(    41.19) },
      { EASYSIMD_FLOAT32_C(   -98.64), EASYSIMD_FLOAT32_C(    94.89), EASYSIMD_FLOAT32_C(    -8.47), EASYSIMD_FLOAT32_C(   -89.83),
        EASYSIMD_FLOAT32_C(   -72.24), EASYSIMD_FLOAT32_C(    71.29), EASYSIMD_FLOAT32_C(   -50.01), EASYSIMD_FLOAT32_C(    85.11),
        EASYSIMD_FLOAT32_C(    22.55), EASYSIMD_FLOAT32_C(   -60.68), EASYSIMD_FLOAT32_C(    -0.57), EASYSIMD_FLOAT32_C(   -29.59),
        EASYSIMD_FLOAT32_C(   -63.27), EASYSIMD_FLOAT32_C(   -65.59), EASYSIMD_FLOAT32_C(   -81.44), EASYSIMD_FLOAT32_C(   -85.80) },
      { EASYSIMD_FLOAT32_C(   -46.37), EASYSIMD_FLOAT32_C(   -24.84), EASYSIMD_FLOAT32_C(   -70.22), EASYSIMD_FLOAT32_C(    31.29),
        EASYSIMD_FLOAT32_C(    34.52), EASYSIMD_FLOAT32_C(    72.90), EASYSIMD_FLOAT32_C(    -4.95), EASYSIMD_FLOAT32_C(    58.55),
        EASYSIMD_FLOAT32_C(    56.33), EASYSIMD_FLOAT32_C(    29.42), EASYSIMD_FLOAT32_C(    59.17), EASYSIMD_FLOAT32_C(   -66.03),
        EASYSIMD_FLOAT32_C(   -11.46), EASYSIMD_FLOAT32_C(    93.67), EASYSIMD_FLOAT32_C(    75.15), EASYSIMD_FLOAT32_C(   -10.11) },
      { EASYSIMD_FLOAT32_C(  7595.27), EASYSIMD_FLOAT32_C(  7640.37), EASYSIMD_FLOAT32_C(  -437.82), EASYSIMD_FLOAT32_C( -7552.16),
        EASYSIMD_FLOAT32_C( -1579.32), EASYSIMD_FLOAT32_C(  2970.13), EASYSIMD_FLOAT32_C(  2151.48), EASYSIMD_FLOAT32_C( -5368.06),
        EASYSIMD_FLOAT32_C(  1769.45), EASYSIMD_FLOAT32_C(  5091.95), EASYSIMD_FLOAT32_C(    21.76), EASYSIMD_FLOAT32_C( -3006.68),
        EASYSIMD_FLOAT32_C( -1426.18), EASYSIMD_FLOAT32_C(  3971.35), EASYSIMD_FLOAT32_C( -5259.17), EASYSIMD_FLOAT32_C(  3523.99) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 c = easysimd_mm512_loadu_ps(test_vec[i].c);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_fnmadd_ps(a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_fnmadd_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_fnmadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const uint16_t k;
    const easysimd_float32 b[16];
    const easysimd_float32 c[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -746.67), EASYSIMD_FLOAT32_C(  -419.75), EASYSIMD_FLOAT32_C(   154.68), EASYSIMD_FLOAT32_C(  -173.73),
        EASYSIMD_FLOAT32_C(    73.51), EASYSIMD_FLOAT32_C(  -916.36), EASYSIMD_FLOAT32_C(   -66.90), EASYSIMD_FLOAT32_C(  -200.84),
        EASYSIMD_FLOAT32_C(   477.39), EASYSIMD_FLOAT32_C(    20.72), EASYSIMD_FLOAT32_C(  -936.43), EASYSIMD_FLOAT32_C(  -949.30),
        EASYSIMD_FLOAT32_C(  -398.99), EASYSIMD_FLOAT32_C(   888.70), EASYSIMD_FLOAT32_C(  -856.82), EASYSIMD_FLOAT32_C(    57.58) },
      UINT16_C(29274),
      { EASYSIMD_FLOAT32_C(  -869.87), EASYSIMD_FLOAT32_C(  -631.08), EASYSIMD_FLOAT32_C(  -100.97), EASYSIMD_FLOAT32_C(  -396.26),
        EASYSIMD_FLOAT32_C(  -157.06), EASYSIMD_FLOAT32_C(   317.43), EASYSIMD_FLOAT32_C(   939.87), EASYSIMD_FLOAT32_C(   520.92),
        EASYSIMD_FLOAT32_C(  -346.01), EASYSIMD_FLOAT32_C(   601.10), EASYSIMD_FLOAT32_C(  -911.23), EASYSIMD_FLOAT32_C(  -137.28),
        EASYSIMD_FLOAT32_C(   517.11), EASYSIMD_FLOAT32_C(  -657.90), EASYSIMD_FLOAT32_C(   442.96), EASYSIMD_FLOAT32_C(  -328.21) },
      { EASYSIMD_FLOAT32_C(   168.37), EASYSIMD_FLOAT32_C(  -483.52), EASYSIMD_FLOAT32_C(  -244.57), EASYSIMD_FLOAT32_C(  -898.53),
        EASYSIMD_FLOAT32_C(   315.64), EASYSIMD_FLOAT32_C(  -767.19), EASYSIMD_FLOAT32_C(   122.19), EASYSIMD_FLOAT32_C(   379.20),
        EASYSIMD_FLOAT32_C(  -716.49), EASYSIMD_FLOAT32_C(   723.21), EASYSIMD_FLOAT32_C(   267.90), EASYSIMD_FLOAT32_C(  -573.31),
        EASYSIMD_FLOAT32_C(  -219.21), EASYSIMD_FLOAT32_C(  -570.53), EASYSIMD_FLOAT32_C(  -352.50), EASYSIMD_FLOAT32_C(   -89.08) },
      { EASYSIMD_FLOAT32_C(  -746.67), EASYSIMD_FLOAT32_C(-265379.38), EASYSIMD_FLOAT32_C(   154.68), EASYSIMD_FLOAT32_C(-69740.78),
        EASYSIMD_FLOAT32_C( 11861.12), EASYSIMD_FLOAT32_C(  -916.36), EASYSIMD_FLOAT32_C( 62999.50), EASYSIMD_FLOAT32_C(  -200.84),
        EASYSIMD_FLOAT32_C(   477.39), EASYSIMD_FLOAT32_C(-11731.58), EASYSIMD_FLOAT32_C(  -936.43), EASYSIMD_FLOAT32_C(  -949.30),
        EASYSIMD_FLOAT32_C(206102.50), EASYSIMD_FLOAT32_C(584105.25), EASYSIMD_FLOAT32_C(379184.47), EASYSIMD_FLOAT32_C(    57.58) } },
    { { EASYSIMD_FLOAT32_C(  -201.60), EASYSIMD_FLOAT32_C(   546.53), EASYSIMD_FLOAT32_C(   514.66), EASYSIMD_FLOAT32_C(   641.33),
        EASYSIMD_FLOAT32_C(  -136.04), EASYSIMD_FLOAT32_C(   454.53), EASYSIMD_FLOAT32_C(   162.25), EASYSIMD_FLOAT32_C(   517.95),
        EASYSIMD_FLOAT32_C(    55.63), EASYSIMD_FLOAT32_C(   251.01), EASYSIMD_FLOAT32_C(  -619.34), EASYSIMD_FLOAT32_C(  -427.26),
        EASYSIMD_FLOAT32_C(   593.11), EASYSIMD_FLOAT32_C(   823.63), EASYSIMD_FLOAT32_C(   244.53), EASYSIMD_FLOAT32_C(  -238.52) },
      UINT16_C(45739),
      { EASYSIMD_FLOAT32_C(  -137.05), EASYSIMD_FLOAT32_C(   655.74), EASYSIMD_FLOAT32_C(  -767.23), EASYSIMD_FLOAT32_C(   985.14),
        EASYSIMD_FLOAT32_C(    34.95), EASYSIMD_FLOAT32_C(  -483.72), EASYSIMD_FLOAT32_C(   708.35), EASYSIMD_FLOAT32_C(  -697.15),
        EASYSIMD_FLOAT32_C(   -57.03), EASYSIMD_FLOAT32_C(  -510.86), EASYSIMD_FLOAT32_C(  -267.68), EASYSIMD_FLOAT32_C(   590.47),
        EASYSIMD_FLOAT32_C(   400.05), EASYSIMD_FLOAT32_C(   530.72), EASYSIMD_FLOAT32_C(   137.00), EASYSIMD_FLOAT32_C(   -85.28) },
      { EASYSIMD_FLOAT32_C(   172.05), EASYSIMD_FLOAT32_C(  -999.04), EASYSIMD_FLOAT32_C(  -630.76), EASYSIMD_FLOAT32_C(  -665.70),
        EASYSIMD_FLOAT32_C(   518.90), EASYSIMD_FLOAT32_C(   424.88), EASYSIMD_FLOAT32_C(   585.31), EASYSIMD_FLOAT32_C(   899.57),
        EASYSIMD_FLOAT32_C(   997.61), EASYSIMD_FLOAT32_C(   178.42), EASYSIMD_FLOAT32_C(   723.20), EASYSIMD_FLOAT32_C(   242.14),
        EASYSIMD_FLOAT32_C(   939.91), EASYSIMD_FLOAT32_C(  -936.70), EASYSIMD_FLOAT32_C(   242.09), EASYSIMD_FLOAT32_C(  -197.14) },
      { EASYSIMD_FLOAT32_C(-27457.23), EASYSIMD_FLOAT32_C(-359380.62), EASYSIMD_FLOAT32_C(   514.66), EASYSIMD_FLOAT32_C(-632465.56),
        EASYSIMD_FLOAT32_C(  -136.04), EASYSIMD_FLOAT32_C(220290.12), EASYSIMD_FLOAT32_C(   162.25), EASYSIMD_FLOAT32_C(361988.44),
        EASYSIMD_FLOAT32_C(    55.63), EASYSIMD_FLOAT32_C(128409.38), EASYSIMD_FLOAT32_C(  -619.34), EASYSIMD_FLOAT32_C(  -427.26),
        EASYSIMD_FLOAT32_C(-236333.73), EASYSIMD_FLOAT32_C(-438053.59), EASYSIMD_FLOAT32_C(   244.53), EASYSIMD_FLOAT32_C(-20538.13) } },
    { { EASYSIMD_FLOAT32_C(   719.05), EASYSIMD_FLOAT32_C(   474.86), EASYSIMD_FLOAT32_C(  -212.00), EASYSIMD_FLOAT32_C(  -246.00),
        EASYSIMD_FLOAT32_C(   991.13), EASYSIMD_FLOAT32_C(  -503.65), EASYSIMD_FLOAT32_C(    56.85), EASYSIMD_FLOAT32_C(   -65.90),
        EASYSIMD_FLOAT32_C(   -14.51), EASYSIMD_FLOAT32_C(   789.17), EASYSIMD_FLOAT32_C(  -475.44), EASYSIMD_FLOAT32_C(  -614.46),
        EASYSIMD_FLOAT32_C(   319.89), EASYSIMD_FLOAT32_C(   661.56), EASYSIMD_FLOAT32_C(   300.26), EASYSIMD_FLOAT32_C(  -508.07) },
      UINT16_C(56897),
      { EASYSIMD_FLOAT32_C(  -173.77), EASYSIMD_FLOAT32_C(   181.42), EASYSIMD_FLOAT32_C(    94.38), EASYSIMD_FLOAT32_C(  -588.46),
        EASYSIMD_FLOAT32_C(    80.99), EASYSIMD_FLOAT32_C(    91.99), EASYSIMD_FLOAT32_C(   589.97), EASYSIMD_FLOAT32_C(  -195.82),
        EASYSIMD_FLOAT32_C(  -665.87), EASYSIMD_FLOAT32_C(   529.87), EASYSIMD_FLOAT32_C(  -132.51), EASYSIMD_FLOAT32_C(   576.22),
        EASYSIMD_FLOAT32_C(  -667.27), EASYSIMD_FLOAT32_C(  -413.47), EASYSIMD_FLOAT32_C(    51.08), EASYSIMD_FLOAT32_C(   120.73) },
      { EASYSIMD_FLOAT32_C(   340.53), EASYSIMD_FLOAT32_C(    42.21), EASYSIMD_FLOAT32_C(   617.08), EASYSIMD_FLOAT32_C(  -602.62),
        EASYSIMD_FLOAT32_C(   976.31), EASYSIMD_FLOAT32_C(  -397.43), EASYSIMD_FLOAT32_C(  -813.46), EASYSIMD_FLOAT32_C(  -499.13),
        EASYSIMD_FLOAT32_C(   -11.88), EASYSIMD_FLOAT32_C(   506.43), EASYSIMD_FLOAT32_C(  -837.57), EASYSIMD_FLOAT32_C(  -711.62),
        EASYSIMD_FLOAT32_C(   998.36), EASYSIMD_FLOAT32_C(   824.94), EASYSIMD_FLOAT32_C(   957.88), EASYSIMD_FLOAT32_C(  -175.41) },
      { EASYSIMD_FLOAT32_C(125289.85), EASYSIMD_FLOAT32_C(   474.86), EASYSIMD_FLOAT32_C(  -212.00), EASYSIMD_FLOAT32_C(  -246.00),
        EASYSIMD_FLOAT32_C(   991.13), EASYSIMD_FLOAT32_C(  -503.65), EASYSIMD_FLOAT32_C(-34353.25), EASYSIMD_FLOAT32_C(   -65.90),
        EASYSIMD_FLOAT32_C(   -14.51), EASYSIMD_FLOAT32_C(-417651.06), EASYSIMD_FLOAT32_C(-63838.12), EASYSIMD_FLOAT32_C(353352.50),
        EASYSIMD_FLOAT32_C(214451.38), EASYSIMD_FLOAT32_C(   661.56), EASYSIMD_FLOAT32_C(-14379.40), EASYSIMD_FLOAT32_C( 61163.88) } },
    { { EASYSIMD_FLOAT32_C(     6.36), EASYSIMD_FLOAT32_C(    52.26), EASYSIMD_FLOAT32_C(   236.14), EASYSIMD_FLOAT32_C(  -912.65),
        EASYSIMD_FLOAT32_C(  -855.75), EASYSIMD_FLOAT32_C(  -173.90), EASYSIMD_FLOAT32_C(  -108.47), EASYSIMD_FLOAT32_C(  -521.62),
        EASYSIMD_FLOAT32_C(  -644.03), EASYSIMD_FLOAT32_C(   759.02), EASYSIMD_FLOAT32_C(  -945.40), EASYSIMD_FLOAT32_C(  -311.30),
        EASYSIMD_FLOAT32_C(  -654.45), EASYSIMD_FLOAT32_C(   105.68), EASYSIMD_FLOAT32_C(   809.43), EASYSIMD_FLOAT32_C(   686.08) },
      UINT16_C(59671),
      { EASYSIMD_FLOAT32_C(  -916.54), EASYSIMD_FLOAT32_C(  -875.81), EASYSIMD_FLOAT32_C(  -970.91), EASYSIMD_FLOAT32_C(  -730.00),
        EASYSIMD_FLOAT32_C(  -374.94), EASYSIMD_FLOAT32_C(    17.21), EASYSIMD_FLOAT32_C(   776.43), EASYSIMD_FLOAT32_C(  -212.51),
        EASYSIMD_FLOAT32_C(   305.59), EASYSIMD_FLOAT32_C(   774.80), EASYSIMD_FLOAT32_C(  -387.57), EASYSIMD_FLOAT32_C(   263.47),
        EASYSIMD_FLOAT32_C(  -400.61), EASYSIMD_FLOAT32_C(   618.79), EASYSIMD_FLOAT32_C(  -684.27), EASYSIMD_FLOAT32_C(   835.53) },
      { EASYSIMD_FLOAT32_C(   706.14), EASYSIMD_FLOAT32_C(  -540.01), EASYSIMD_FLOAT32_C(  -338.37), EASYSIMD_FLOAT32_C(  -402.33),
        EASYSIMD_FLOAT32_C(   -61.63), EASYSIMD_FLOAT32_C(    17.60), EASYSIMD_FLOAT32_C(  -643.31), EASYSIMD_FLOAT32_C(    -7.03),
        EASYSIMD_FLOAT32_C(   706.30), EASYSIMD_FLOAT32_C(  -297.76), EASYSIMD_FLOAT32_C(  -901.35), EASYSIMD_FLOAT32_C(   515.73),
        EASYSIMD_FLOAT32_C(  -611.68), EASYSIMD_FLOAT32_C(  -753.47), EASYSIMD_FLOAT32_C(   -57.75), EASYSIMD_FLOAT32_C(  -528.22) },
      { EASYSIMD_FLOAT32_C(  6535.33), EASYSIMD_FLOAT32_C( 45229.82), EASYSIMD_FLOAT32_C(228932.31), EASYSIMD_FLOAT32_C(  -912.65),
        EASYSIMD_FLOAT32_C(-320916.53), EASYSIMD_FLOAT32_C(  -173.90), EASYSIMD_FLOAT32_C(  -108.47), EASYSIMD_FLOAT32_C(  -521.62),
        EASYSIMD_FLOAT32_C(197515.44), EASYSIMD_FLOAT32_C(   759.02), EASYSIMD_FLOAT32_C(  -945.40), EASYSIMD_FLOAT32_C( 82533.94),
        EASYSIMD_FLOAT32_C(  -654.45), EASYSIMD_FLOAT32_C(-66147.20), EASYSIMD_FLOAT32_C(553810.94), EASYSIMD_FLOAT32_C(-573768.69) } },
    { { EASYSIMD_FLOAT32_C(  -629.28), EASYSIMD_FLOAT32_C(   -28.66), EASYSIMD_FLOAT32_C(  -258.21), EASYSIMD_FLOAT32_C(    -4.22),
        EASYSIMD_FLOAT32_C(   988.56), EASYSIMD_FLOAT32_C(  -481.78), EASYSIMD_FLOAT32_C(   783.27), EASYSIMD_FLOAT32_C(   294.14),
        EASYSIMD_FLOAT32_C(  -706.99), EASYSIMD_FLOAT32_C(  -604.30), EASYSIMD_FLOAT32_C(  -442.38), EASYSIMD_FLOAT32_C(  -107.60),
        EASYSIMD_FLOAT32_C(  -985.51), EASYSIMD_FLOAT32_C(  -126.65), EASYSIMD_FLOAT32_C(  -272.07), EASYSIMD_FLOAT32_C(   720.64) },
      UINT16_C(12722),
      { EASYSIMD_FLOAT32_C(  -681.69), EASYSIMD_FLOAT32_C(  -728.30), EASYSIMD_FLOAT32_C(  -592.84), EASYSIMD_FLOAT32_C(  -325.00),
        EASYSIMD_FLOAT32_C(   264.67), EASYSIMD_FLOAT32_C(  -886.54), EASYSIMD_FLOAT32_C(   377.24), EASYSIMD_FLOAT32_C(   363.32),
        EASYSIMD_FLOAT32_C(   629.19), EASYSIMD_FLOAT32_C(   765.56), EASYSIMD_FLOAT32_C(   609.85), EASYSIMD_FLOAT32_C(  -428.56),
        EASYSIMD_FLOAT32_C(  -762.66), EASYSIMD_FLOAT32_C(   980.57), EASYSIMD_FLOAT32_C(   542.79), EASYSIMD_FLOAT32_C(   -20.87) },
      { EASYSIMD_FLOAT32_C(   -23.64), EASYSIMD_FLOAT32_C(   531.34), EASYSIMD_FLOAT32_C(   497.35), EASYSIMD_FLOAT32_C(  -240.37),
        EASYSIMD_FLOAT32_C(  -174.51), EASYSIMD_FLOAT32_C(   790.36), EASYSIMD_FLOAT32_C(   155.33), EASYSIMD_FLOAT32_C(   383.10),
        EASYSIMD_FLOAT32_C(  -317.24), EASYSIMD_FLOAT32_C(   169.82), EASYSIMD_FLOAT32_C(  -743.55), EASYSIMD_FLOAT32_C(   410.69),
        EASYSIMD_FLOAT32_C(  -109.54), EASYSIMD_FLOAT32_C(   589.79), EASYSIMD_FLOAT32_C(  -199.75), EASYSIMD_FLOAT32_C(   208.77) },
      { EASYSIMD_FLOAT32_C(  -629.28), EASYSIMD_FLOAT32_C(-20341.74), EASYSIMD_FLOAT32_C(  -258.21), EASYSIMD_FLOAT32_C(    -4.22),
        EASYSIMD_FLOAT32_C(-261816.70), EASYSIMD_FLOAT32_C(-426326.84), EASYSIMD_FLOAT32_C(   783.27), EASYSIMD_FLOAT32_C(-106483.85),
        EASYSIMD_FLOAT32_C(444513.78), EASYSIMD_FLOAT32_C(  -604.30), EASYSIMD_FLOAT32_C(  -442.38), EASYSIMD_FLOAT32_C(  -107.60),
        EASYSIMD_FLOAT32_C(-751718.62), EASYSIMD_FLOAT32_C(124778.98), EASYSIMD_FLOAT32_C(  -272.07), EASYSIMD_FLOAT32_C(   720.64) } },
    { { EASYSIMD_FLOAT32_C(   861.49), EASYSIMD_FLOAT32_C(   207.40), EASYSIMD_FLOAT32_C(   883.76), EASYSIMD_FLOAT32_C(   126.16),
        EASYSIMD_FLOAT32_C(   320.86), EASYSIMD_FLOAT32_C(   261.00), EASYSIMD_FLOAT32_C(  -510.52), EASYSIMD_FLOAT32_C(   -49.95),
        EASYSIMD_FLOAT32_C(    26.56), EASYSIMD_FLOAT32_C(  -900.67), EASYSIMD_FLOAT32_C(   521.49), EASYSIMD_FLOAT32_C(   263.91),
        EASYSIMD_FLOAT32_C(  -920.09), EASYSIMD_FLOAT32_C(    64.28), EASYSIMD_FLOAT32_C(  -756.96), EASYSIMD_FLOAT32_C(    56.26) },
      UINT16_C(59065),
      { EASYSIMD_FLOAT32_C(   815.89), EASYSIMD_FLOAT32_C(   421.10), EASYSIMD_FLOAT32_C(   530.75), EASYSIMD_FLOAT32_C(   -28.78),
        EASYSIMD_FLOAT32_C(  -195.79), EASYSIMD_FLOAT32_C(  -786.49), EASYSIMD_FLOAT32_C(  -858.96), EASYSIMD_FLOAT32_C(    60.66),
        EASYSIMD_FLOAT32_C(   624.20), EASYSIMD_FLOAT32_C(    31.50), EASYSIMD_FLOAT32_C(  -349.56), EASYSIMD_FLOAT32_C(  -575.55),
        EASYSIMD_FLOAT32_C(  -759.73), EASYSIMD_FLOAT32_C(  -488.07), EASYSIMD_FLOAT32_C(   631.85), EASYSIMD_FLOAT32_C(  -875.97) },
      { EASYSIMD_FLOAT32_C(   638.09), EASYSIMD_FLOAT32_C(   -47.29), EASYSIMD_FLOAT32_C(   385.03), EASYSIMD_FLOAT32_C(  -872.43),
        EASYSIMD_FLOAT32_C(   902.76), EASYSIMD_FLOAT32_C(  -588.40), EASYSIMD_FLOAT32_C(  -773.10), EASYSIMD_FLOAT32_C(   424.25),
        EASYSIMD_FLOAT32_C(   675.50), EASYSIMD_FLOAT32_C(  -693.19), EASYSIMD_FLOAT32_C(  -511.47), EASYSIMD_FLOAT32_C(   918.54),
        EASYSIMD_FLOAT32_C(   363.07), EASYSIMD_FLOAT32_C(    84.14), EASYSIMD_FLOAT32_C(   658.93), EASYSIMD_FLOAT32_C(   178.96) },
      { EASYSIMD_FLOAT32_C(-702243.00), EASYSIMD_FLOAT32_C(   207.40), EASYSIMD_FLOAT32_C(   883.76), EASYSIMD_FLOAT32_C(  2758.46),
        EASYSIMD_FLOAT32_C( 63723.94), EASYSIMD_FLOAT32_C(204685.48), EASYSIMD_FLOAT32_C(  -510.52), EASYSIMD_FLOAT32_C(  3454.22),
        EASYSIMD_FLOAT32_C(    26.56), EASYSIMD_FLOAT32_C( 27677.91), EASYSIMD_FLOAT32_C(181780.58), EASYSIMD_FLOAT32_C(   263.91),
        EASYSIMD_FLOAT32_C(  -920.09), EASYSIMD_FLOAT32_C( 31457.28), EASYSIMD_FLOAT32_C(478944.09), EASYSIMD_FLOAT32_C( 49461.03) } },
    { { EASYSIMD_FLOAT32_C(  -494.75), EASYSIMD_FLOAT32_C(   189.67), EASYSIMD_FLOAT32_C(  -849.82), EASYSIMD_FLOAT32_C(   309.45),
        EASYSIMD_FLOAT32_C(   403.18), EASYSIMD_FLOAT32_C(  -708.78), EASYSIMD_FLOAT32_C(  -629.89), EASYSIMD_FLOAT32_C(    27.38),
        EASYSIMD_FLOAT32_C(   322.72), EASYSIMD_FLOAT32_C(    20.55), EASYSIMD_FLOAT32_C(   451.83), EASYSIMD_FLOAT32_C(   562.99),
        EASYSIMD_FLOAT32_C(   532.48), EASYSIMD_FLOAT32_C(    83.69), EASYSIMD_FLOAT32_C(   687.02), EASYSIMD_FLOAT32_C(   170.57) },
      UINT16_C(39540),
      { EASYSIMD_FLOAT32_C(   298.14), EASYSIMD_FLOAT32_C(   939.16), EASYSIMD_FLOAT32_C(   483.65), EASYSIMD_FLOAT32_C(   525.04),
        EASYSIMD_FLOAT32_C(   363.41), EASYSIMD_FLOAT32_C(   159.15), EASYSIMD_FLOAT32_C(   831.85), EASYSIMD_FLOAT32_C(   851.94),
        EASYSIMD_FLOAT32_C(    77.69), EASYSIMD_FLOAT32_C(   194.92), EASYSIMD_FLOAT32_C(   -63.92), EASYSIMD_FLOAT32_C(  -263.38),
        EASYSIMD_FLOAT32_C(  -626.12), EASYSIMD_FLOAT32_C(   441.32), EASYSIMD_FLOAT32_C(   926.29), EASYSIMD_FLOAT32_C(  -475.94) },
      { EASYSIMD_FLOAT32_C(  -249.23), EASYSIMD_FLOAT32_C(   329.48), EASYSIMD_FLOAT32_C(  -184.71), EASYSIMD_FLOAT32_C(   120.88),
        EASYSIMD_FLOAT32_C(  -643.14), EASYSIMD_FLOAT32_C(  -861.99), EASYSIMD_FLOAT32_C(  -858.57), EASYSIMD_FLOAT32_C(   808.70),
        EASYSIMD_FLOAT32_C(   701.00), EASYSIMD_FLOAT32_C(   673.91), EASYSIMD_FLOAT32_C(  -107.62), EASYSIMD_FLOAT32_C(   388.02),
        EASYSIMD_FLOAT32_C(  -155.52), EASYSIMD_FLOAT32_C(   -71.22), EASYSIMD_FLOAT32_C(  -539.92), EASYSIMD_FLOAT32_C(  -857.37) },
      { EASYSIMD_FLOAT32_C(  -494.75), EASYSIMD_FLOAT32_C(   189.67), EASYSIMD_FLOAT32_C(410830.72), EASYSIMD_FLOAT32_C(   309.45),
        EASYSIMD_FLOAT32_C(-147162.78), EASYSIMD_FLOAT32_C(111940.34), EASYSIMD_FLOAT32_C(523115.44), EASYSIMD_FLOAT32_C(    27.38),
        EASYSIMD_FLOAT32_C(   322.72), EASYSIMD_FLOAT32_C( -3331.70), EASYSIMD_FLOAT32_C(   451.83), EASYSIMD_FLOAT32_C(148668.33),
        EASYSIMD_FLOAT32_C(333240.84), EASYSIMD_FLOAT32_C(    83.69), EASYSIMD_FLOAT32_C(   687.02), EASYSIMD_FLOAT32_C( 80323.72) } },
    { { EASYSIMD_FLOAT32_C(  -132.05), EASYSIMD_FLOAT32_C(   943.73), EASYSIMD_FLOAT32_C(   667.67), EASYSIMD_FLOAT32_C(  -768.64),
        EASYSIMD_FLOAT32_C(   102.88), EASYSIMD_FLOAT32_C(   499.52), EASYSIMD_FLOAT32_C(  -916.71), EASYSIMD_FLOAT32_C(  -819.43),
        EASYSIMD_FLOAT32_C(  -305.55), EASYSIMD_FLOAT32_C(    19.37), EASYSIMD_FLOAT32_C(   -82.80), EASYSIMD_FLOAT32_C(    68.33),
        EASYSIMD_FLOAT32_C(  -539.31), EASYSIMD_FLOAT32_C(  -156.51), EASYSIMD_FLOAT32_C(   592.39), EASYSIMD_FLOAT32_C(   211.47) },
      UINT16_C(58754),
      { EASYSIMD_FLOAT32_C(  -667.65), EASYSIMD_FLOAT32_C(  -470.17), EASYSIMD_FLOAT32_C(  -454.31), EASYSIMD_FLOAT32_C(  -526.22),
        EASYSIMD_FLOAT32_C(  -661.48), EASYSIMD_FLOAT32_C(  -753.30), EASYSIMD_FLOAT32_C(  -852.31), EASYSIMD_FLOAT32_C(   230.91),
        EASYSIMD_FLOAT32_C(   634.72), EASYSIMD_FLOAT32_C(    -7.82), EASYSIMD_FLOAT32_C(  -840.31), EASYSIMD_FLOAT32_C(  -905.20),
        EASYSIMD_FLOAT32_C(   134.81), EASYSIMD_FLOAT32_C(    27.64), EASYSIMD_FLOAT32_C(  -961.47), EASYSIMD_FLOAT32_C(  -197.52) },
      { EASYSIMD_FLOAT32_C(   258.99), EASYSIMD_FLOAT32_C(   141.41), EASYSIMD_FLOAT32_C(  -698.00), EASYSIMD_FLOAT32_C(   342.29),
        EASYSIMD_FLOAT32_C(   321.98), EASYSIMD_FLOAT32_C(    -3.55), EASYSIMD_FLOAT32_C(  -638.34), EASYSIMD_FLOAT32_C(  -760.82),
        EASYSIMD_FLOAT32_C(  -935.22), EASYSIMD_FLOAT32_C(  -177.65), EASYSIMD_FLOAT32_C(    82.67), EASYSIMD_FLOAT32_C(   657.17),
        EASYSIMD_FLOAT32_C(  -966.18), EASYSIMD_FLOAT32_C(   255.63), EASYSIMD_FLOAT32_C(  -935.14), EASYSIMD_FLOAT32_C(  -633.83) },
      { EASYSIMD_FLOAT32_C(  -132.05), EASYSIMD_FLOAT32_C(443854.94), EASYSIMD_FLOAT32_C(   667.67), EASYSIMD_FLOAT32_C(  -768.64),
        EASYSIMD_FLOAT32_C(   102.88), EASYSIMD_FLOAT32_C(   499.52), EASYSIMD_FLOAT32_C(  -916.71), EASYSIMD_FLOAT32_C(188453.77),
        EASYSIMD_FLOAT32_C(193003.45), EASYSIMD_FLOAT32_C(    19.37), EASYSIMD_FLOAT32_C(-69495.00), EASYSIMD_FLOAT32_C(    68.33),
        EASYSIMD_FLOAT32_C(  -539.31), EASYSIMD_FLOAT32_C(  4581.57), EASYSIMD_FLOAT32_C(568630.06), EASYSIMD_FLOAT32_C( 41135.73) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 c = easysimd_mm512_loadu_ps(test_vec[i].c);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_fnmadd_ps(a, k, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_fnmadd_ps");
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
    easysimd__m512 r = easysimd_mm512_mask_fnmadd_ps(a, k, b, c);

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
test_easysimd_mm512_maskz_fnmadd_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const uint16_t k;
    const easysimd_float32 b[16];
    const easysimd_float32 c[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -293.28), EASYSIMD_FLOAT32_C(  -573.86), EASYSIMD_FLOAT32_C(   853.71), EASYSIMD_FLOAT32_C(   788.97),
        EASYSIMD_FLOAT32_C(   468.02), EASYSIMD_FLOAT32_C(   955.58), EASYSIMD_FLOAT32_C(   683.20), EASYSIMD_FLOAT32_C(   930.94),
        EASYSIMD_FLOAT32_C(   114.61), EASYSIMD_FLOAT32_C(  -361.92), EASYSIMD_FLOAT32_C(  -443.38), EASYSIMD_FLOAT32_C(   337.66),
        EASYSIMD_FLOAT32_C(  -746.40), EASYSIMD_FLOAT32_C(   549.48), EASYSIMD_FLOAT32_C(   -55.65), EASYSIMD_FLOAT32_C(  -920.72) },
      UINT16_C(16902),
      { EASYSIMD_FLOAT32_C(   772.46), EASYSIMD_FLOAT32_C(   847.31), EASYSIMD_FLOAT32_C(  -471.87), EASYSIMD_FLOAT32_C(   511.86),
        EASYSIMD_FLOAT32_C(   999.27), EASYSIMD_FLOAT32_C(   690.77), EASYSIMD_FLOAT32_C(    35.04), EASYSIMD_FLOAT32_C(   308.09),
        EASYSIMD_FLOAT32_C(   780.66), EASYSIMD_FLOAT32_C(  -238.92), EASYSIMD_FLOAT32_C(   433.02), EASYSIMD_FLOAT32_C(  -978.64),
        EASYSIMD_FLOAT32_C(   832.52), EASYSIMD_FLOAT32_C(  -860.26), EASYSIMD_FLOAT32_C(  -552.50), EASYSIMD_FLOAT32_C(   686.23) },
      { EASYSIMD_FLOAT32_C(   928.72), EASYSIMD_FLOAT32_C(   915.52), EASYSIMD_FLOAT32_C(   641.81), EASYSIMD_FLOAT32_C(   611.92),
        EASYSIMD_FLOAT32_C(   846.46), EASYSIMD_FLOAT32_C(  -243.58), EASYSIMD_FLOAT32_C(  -750.00), EASYSIMD_FLOAT32_C(  -596.92),
        EASYSIMD_FLOAT32_C(  -905.91), EASYSIMD_FLOAT32_C(  -496.40), EASYSIMD_FLOAT32_C(   952.55), EASYSIMD_FLOAT32_C(    38.44),
        EASYSIMD_FLOAT32_C(  -417.11), EASYSIMD_FLOAT32_C(   510.62), EASYSIMD_FLOAT32_C(   649.99), EASYSIMD_FLOAT32_C(  -644.66) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(487152.84), EASYSIMD_FLOAT32_C(403481.97), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-86966.33), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-30096.63), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   357.93), EASYSIMD_FLOAT32_C(  -821.88), EASYSIMD_FLOAT32_C(   867.21), EASYSIMD_FLOAT32_C(   357.21),
        EASYSIMD_FLOAT32_C(   868.89), EASYSIMD_FLOAT32_C(   -97.76), EASYSIMD_FLOAT32_C(  -334.71), EASYSIMD_FLOAT32_C(   649.54),
        EASYSIMD_FLOAT32_C(   663.32), EASYSIMD_FLOAT32_C(  -901.69), EASYSIMD_FLOAT32_C(   670.90), EASYSIMD_FLOAT32_C(   495.84),
        EASYSIMD_FLOAT32_C(  -761.94), EASYSIMD_FLOAT32_C(  -881.60), EASYSIMD_FLOAT32_C(   182.07), EASYSIMD_FLOAT32_C(  -833.23) },
      UINT16_C( 4952),
      { EASYSIMD_FLOAT32_C(   778.69), EASYSIMD_FLOAT32_C(   880.38), EASYSIMD_FLOAT32_C(   580.30), EASYSIMD_FLOAT32_C(  -971.30),
        EASYSIMD_FLOAT32_C(  -716.55), EASYSIMD_FLOAT32_C(   674.39), EASYSIMD_FLOAT32_C(  -467.70), EASYSIMD_FLOAT32_C(  -763.99),
        EASYSIMD_FLOAT32_C(  -287.17), EASYSIMD_FLOAT32_C(   115.18), EASYSIMD_FLOAT32_C(   746.62), EASYSIMD_FLOAT32_C(  -637.18),
        EASYSIMD_FLOAT32_C(   470.53), EASYSIMD_FLOAT32_C(   104.56), EASYSIMD_FLOAT32_C(  -459.06), EASYSIMD_FLOAT32_C(   337.73) },
      { EASYSIMD_FLOAT32_C(  -538.24), EASYSIMD_FLOAT32_C(  -590.17), EASYSIMD_FLOAT32_C(  -760.02), EASYSIMD_FLOAT32_C(   127.06),
        EASYSIMD_FLOAT32_C(  -940.62), EASYSIMD_FLOAT32_C(   903.30), EASYSIMD_FLOAT32_C(   225.37), EASYSIMD_FLOAT32_C(   730.28),
        EASYSIMD_FLOAT32_C(   399.14), EASYSIMD_FLOAT32_C(   463.43), EASYSIMD_FLOAT32_C(   848.68), EASYSIMD_FLOAT32_C(  -418.79),
        EASYSIMD_FLOAT32_C(   630.20), EASYSIMD_FLOAT32_C(   882.60), EASYSIMD_FLOAT32_C(   405.09), EASYSIMD_FLOAT32_C(   408.90) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(347085.12),
        EASYSIMD_FLOAT32_C(621662.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-156318.48), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(190884.75), EASYSIMD_FLOAT32_C(104320.09), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(359145.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   762.97), EASYSIMD_FLOAT32_C(   -14.61), EASYSIMD_FLOAT32_C(   437.59), EASYSIMD_FLOAT32_C(  -953.57),
        EASYSIMD_FLOAT32_C(  -340.22), EASYSIMD_FLOAT32_C(   969.89), EASYSIMD_FLOAT32_C(  -717.57), EASYSIMD_FLOAT32_C(   372.61),
        EASYSIMD_FLOAT32_C(    85.08), EASYSIMD_FLOAT32_C(  -970.94), EASYSIMD_FLOAT32_C(   735.44), EASYSIMD_FLOAT32_C(  -444.39),
        EASYSIMD_FLOAT32_C(   133.61), EASYSIMD_FLOAT32_C(  -723.62), EASYSIMD_FLOAT32_C(   893.34), EASYSIMD_FLOAT32_C(   595.38) },
      UINT16_C(38265),
      { EASYSIMD_FLOAT32_C(  -277.57), EASYSIMD_FLOAT32_C(  -254.41), EASYSIMD_FLOAT32_C(  -963.39), EASYSIMD_FLOAT32_C(   947.80),
        EASYSIMD_FLOAT32_C(  -524.14), EASYSIMD_FLOAT32_C(   435.75), EASYSIMD_FLOAT32_C(   411.23), EASYSIMD_FLOAT32_C(  -675.46),
        EASYSIMD_FLOAT32_C(  -983.04), EASYSIMD_FLOAT32_C(    41.43), EASYSIMD_FLOAT32_C(  -792.86), EASYSIMD_FLOAT32_C(   422.05),
        EASYSIMD_FLOAT32_C(  -549.67), EASYSIMD_FLOAT32_C(   970.11), EASYSIMD_FLOAT32_C(  -592.56), EASYSIMD_FLOAT32_C(   887.92) },
      { EASYSIMD_FLOAT32_C(  -983.46), EASYSIMD_FLOAT32_C(    67.22), EASYSIMD_FLOAT32_C(   857.81), EASYSIMD_FLOAT32_C(  -701.03),
        EASYSIMD_FLOAT32_C(  -560.16), EASYSIMD_FLOAT32_C(   -57.11), EASYSIMD_FLOAT32_C(  -671.97), EASYSIMD_FLOAT32_C(  -824.73),
        EASYSIMD_FLOAT32_C(   498.50), EASYSIMD_FLOAT32_C(   461.65), EASYSIMD_FLOAT32_C(  -548.35), EASYSIMD_FLOAT32_C(   391.84),
        EASYSIMD_FLOAT32_C(    57.03), EASYSIMD_FLOAT32_C(   137.86), EASYSIMD_FLOAT32_C(   525.15), EASYSIMD_FLOAT32_C(   779.46) },
      { EASYSIMD_FLOAT32_C(210794.12), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(903092.62),
        EASYSIMD_FLOAT32_C(-178883.08), EASYSIMD_FLOAT32_C(-422686.69), EASYSIMD_FLOAT32_C(294414.34), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C( 84135.55), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(582552.56), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C( 73498.44), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-527870.38) } },
    { { EASYSIMD_FLOAT32_C(   883.44), EASYSIMD_FLOAT32_C(   561.77), EASYSIMD_FLOAT32_C(   727.26), EASYSIMD_FLOAT32_C(  -640.69),
        EASYSIMD_FLOAT32_C(    -2.48), EASYSIMD_FLOAT32_C(   138.49), EASYSIMD_FLOAT32_C(  -316.15), EASYSIMD_FLOAT32_C(    14.48),
        EASYSIMD_FLOAT32_C(  -820.08), EASYSIMD_FLOAT32_C(  -109.01), EASYSIMD_FLOAT32_C(  -563.47), EASYSIMD_FLOAT32_C(  -369.75),
        EASYSIMD_FLOAT32_C(  -138.90), EASYSIMD_FLOAT32_C(  -156.03), EASYSIMD_FLOAT32_C(  -481.83), EASYSIMD_FLOAT32_C(  -122.35) },
      UINT16_C(  270),
      { EASYSIMD_FLOAT32_C(   176.62), EASYSIMD_FLOAT32_C(  -648.97), EASYSIMD_FLOAT32_C(   318.87), EASYSIMD_FLOAT32_C(   504.65),
        EASYSIMD_FLOAT32_C(  -473.70), EASYSIMD_FLOAT32_C(  -182.63), EASYSIMD_FLOAT32_C(   -33.70), EASYSIMD_FLOAT32_C(   -22.05),
        EASYSIMD_FLOAT32_C(  -790.79), EASYSIMD_FLOAT32_C(  -976.67), EASYSIMD_FLOAT32_C(  -884.19), EASYSIMD_FLOAT32_C(   734.37),
        EASYSIMD_FLOAT32_C(   802.79), EASYSIMD_FLOAT32_C(   999.25), EASYSIMD_FLOAT32_C(   296.13), EASYSIMD_FLOAT32_C(   530.05) },
      { EASYSIMD_FLOAT32_C(  -641.44), EASYSIMD_FLOAT32_C(  -706.35), EASYSIMD_FLOAT32_C(  -331.46), EASYSIMD_FLOAT32_C(    42.41),
        EASYSIMD_FLOAT32_C(   308.13), EASYSIMD_FLOAT32_C(  -151.53), EASYSIMD_FLOAT32_C(   933.40), EASYSIMD_FLOAT32_C(   744.66),
        EASYSIMD_FLOAT32_C(   478.72), EASYSIMD_FLOAT32_C(  -205.49), EASYSIMD_FLOAT32_C(  -411.37), EASYSIMD_FLOAT32_C(   996.88),
        EASYSIMD_FLOAT32_C(   672.15), EASYSIMD_FLOAT32_C(  -500.18), EASYSIMD_FLOAT32_C(  -627.13), EASYSIMD_FLOAT32_C(  -151.23) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(363865.53), EASYSIMD_FLOAT32_C(-232232.84), EASYSIMD_FLOAT32_C(323366.62),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-648032.31), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -149.16), EASYSIMD_FLOAT32_C(   691.74), EASYSIMD_FLOAT32_C(  -646.57), EASYSIMD_FLOAT32_C(   377.14),
        EASYSIMD_FLOAT32_C(  -490.88), EASYSIMD_FLOAT32_C(   319.73), EASYSIMD_FLOAT32_C(  -644.91), EASYSIMD_FLOAT32_C(  -281.67),
        EASYSIMD_FLOAT32_C(   343.06), EASYSIMD_FLOAT32_C(  -529.11), EASYSIMD_FLOAT32_C(  -547.31), EASYSIMD_FLOAT32_C(   145.85),
        EASYSIMD_FLOAT32_C(  -529.86), EASYSIMD_FLOAT32_C(   748.82), EASYSIMD_FLOAT32_C(  -324.10), EASYSIMD_FLOAT32_C(  -171.30) },
      UINT16_C(36127),
      { EASYSIMD_FLOAT32_C(   871.11), EASYSIMD_FLOAT32_C(   350.61), EASYSIMD_FLOAT32_C(  -807.09), EASYSIMD_FLOAT32_C(   804.51),
        EASYSIMD_FLOAT32_C(    95.27), EASYSIMD_FLOAT32_C(   671.63), EASYSIMD_FLOAT32_C(  -400.98), EASYSIMD_FLOAT32_C(   683.89),
        EASYSIMD_FLOAT32_C(   668.51), EASYSIMD_FLOAT32_C(  -728.83), EASYSIMD_FLOAT32_C(  -816.29), EASYSIMD_FLOAT32_C(  -958.62),
        EASYSIMD_FLOAT32_C(   119.94), EASYSIMD_FLOAT32_C(    34.56), EASYSIMD_FLOAT32_C(   733.12), EASYSIMD_FLOAT32_C(   473.37) },
      { EASYSIMD_FLOAT32_C(  -588.30), EASYSIMD_FLOAT32_C(  -757.76), EASYSIMD_FLOAT32_C(  -206.90), EASYSIMD_FLOAT32_C(  -233.22),
        EASYSIMD_FLOAT32_C(   -39.43), EASYSIMD_FLOAT32_C(  -863.84), EASYSIMD_FLOAT32_C(   237.67), EASYSIMD_FLOAT32_C(   413.26),
        EASYSIMD_FLOAT32_C(   282.02), EASYSIMD_FLOAT32_C(   707.82), EASYSIMD_FLOAT32_C(   162.08), EASYSIMD_FLOAT32_C(   957.92),
        EASYSIMD_FLOAT32_C(  -463.48), EASYSIMD_FLOAT32_C(   204.56), EASYSIMD_FLOAT32_C(   302.37), EASYSIMD_FLOAT32_C(  -592.37) },
      { EASYSIMD_FLOAT32_C(129346.47), EASYSIMD_FLOAT32_C(-243288.72), EASYSIMD_FLOAT32_C(-522047.12), EASYSIMD_FLOAT32_C(-303646.12),
        EASYSIMD_FLOAT32_C( 46726.71), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-229057.03), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-446601.56), EASYSIMD_FLOAT32_C(140772.66),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 80495.91) } },
    { { EASYSIMD_FLOAT32_C(  -444.84), EASYSIMD_FLOAT32_C(   495.28), EASYSIMD_FLOAT32_C(  -787.86), EASYSIMD_FLOAT32_C(   650.43),
        EASYSIMD_FLOAT32_C(   166.91), EASYSIMD_FLOAT32_C(  -188.84), EASYSIMD_FLOAT32_C(   334.32), EASYSIMD_FLOAT32_C(  -164.58),
        EASYSIMD_FLOAT32_C(    82.33), EASYSIMD_FLOAT32_C(   518.04), EASYSIMD_FLOAT32_C(  -123.19), EASYSIMD_FLOAT32_C(  -797.72),
        EASYSIMD_FLOAT32_C(  -447.41), EASYSIMD_FLOAT32_C(  -390.07), EASYSIMD_FLOAT32_C(   675.65), EASYSIMD_FLOAT32_C(   -35.71) },
      UINT16_C(49659),
      { EASYSIMD_FLOAT32_C(   731.07), EASYSIMD_FLOAT32_C(   812.74), EASYSIMD_FLOAT32_C(  -395.08), EASYSIMD_FLOAT32_C(   -31.26),
        EASYSIMD_FLOAT32_C(   226.00), EASYSIMD_FLOAT32_C(   886.93), EASYSIMD_FLOAT32_C(  -323.44), EASYSIMD_FLOAT32_C(  -611.92),
        EASYSIMD_FLOAT32_C(   844.85), EASYSIMD_FLOAT32_C(   213.08), EASYSIMD_FLOAT32_C(   592.64), EASYSIMD_FLOAT32_C(   147.22),
        EASYSIMD_FLOAT32_C(   620.71), EASYSIMD_FLOAT32_C(  -852.20), EASYSIMD_FLOAT32_C(  -357.50), EASYSIMD_FLOAT32_C(   832.85) },
      { EASYSIMD_FLOAT32_C(   798.23), EASYSIMD_FLOAT32_C(   809.41), EASYSIMD_FLOAT32_C(  -355.99), EASYSIMD_FLOAT32_C(   132.55),
        EASYSIMD_FLOAT32_C(  -355.17), EASYSIMD_FLOAT32_C(   726.34), EASYSIMD_FLOAT32_C(  -349.41), EASYSIMD_FLOAT32_C(   521.64),
        EASYSIMD_FLOAT32_C(   928.62), EASYSIMD_FLOAT32_C(   203.18), EASYSIMD_FLOAT32_C(  -868.43), EASYSIMD_FLOAT32_C(   604.26),
        EASYSIMD_FLOAT32_C(  -832.53), EASYSIMD_FLOAT32_C(   -16.26), EASYSIMD_FLOAT32_C(  -926.98), EASYSIMD_FLOAT32_C(   898.54) },
      { EASYSIMD_FLOAT32_C(326007.41), EASYSIMD_FLOAT32_C(-401724.47), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C( 20464.99),
        EASYSIMD_FLOAT32_C(-38076.83), EASYSIMD_FLOAT32_C(168214.20), EASYSIMD_FLOAT32_C(107783.05), EASYSIMD_FLOAT32_C(-100188.15),
        EASYSIMD_FLOAT32_C(-68627.88), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(240617.91), EASYSIMD_FLOAT32_C( 30639.61) } },
    { { EASYSIMD_FLOAT32_C(  -203.52), EASYSIMD_FLOAT32_C(  -322.07), EASYSIMD_FLOAT32_C(  -132.72), EASYSIMD_FLOAT32_C(  -977.52),
        EASYSIMD_FLOAT32_C(  -435.13), EASYSIMD_FLOAT32_C(   543.84), EASYSIMD_FLOAT32_C(  -589.44), EASYSIMD_FLOAT32_C(  -590.28),
        EASYSIMD_FLOAT32_C(  -243.08), EASYSIMD_FLOAT32_C(  -996.81), EASYSIMD_FLOAT32_C(   556.93), EASYSIMD_FLOAT32_C(  -622.37),
        EASYSIMD_FLOAT32_C(  -849.01), EASYSIMD_FLOAT32_C(  -800.57), EASYSIMD_FLOAT32_C(  -789.53), EASYSIMD_FLOAT32_C(   949.22) },
      UINT16_C(26655),
      { EASYSIMD_FLOAT32_C(    81.77), EASYSIMD_FLOAT32_C(  -346.33), EASYSIMD_FLOAT32_C(  -419.18), EASYSIMD_FLOAT32_C(   732.36),
        EASYSIMD_FLOAT32_C(  -824.69), EASYSIMD_FLOAT32_C(  -490.56), EASYSIMD_FLOAT32_C(   -64.46), EASYSIMD_FLOAT32_C(  -693.12),
        EASYSIMD_FLOAT32_C(  -886.30), EASYSIMD_FLOAT32_C(   103.00), EASYSIMD_FLOAT32_C(   290.63), EASYSIMD_FLOAT32_C(  -813.28),
        EASYSIMD_FLOAT32_C(     1.54), EASYSIMD_FLOAT32_C(  -912.89), EASYSIMD_FLOAT32_C(  -135.34), EASYSIMD_FLOAT32_C(   868.82) },
      { EASYSIMD_FLOAT32_C(  -890.41), EASYSIMD_FLOAT32_C(   429.52), EASYSIMD_FLOAT32_C(   412.67), EASYSIMD_FLOAT32_C(  -479.85),
        EASYSIMD_FLOAT32_C(   839.24), EASYSIMD_FLOAT32_C(  -830.41), EASYSIMD_FLOAT32_C(  -476.66), EASYSIMD_FLOAT32_C(   396.18),
        EASYSIMD_FLOAT32_C(  -452.79), EASYSIMD_FLOAT32_C(  -325.67), EASYSIMD_FLOAT32_C(   595.61), EASYSIMD_FLOAT32_C(  -242.32),
        EASYSIMD_FLOAT32_C(  -376.45), EASYSIMD_FLOAT32_C(   604.45), EASYSIMD_FLOAT32_C(   612.17), EASYSIMD_FLOAT32_C(   705.32) },
      { EASYSIMD_FLOAT32_C( 15751.42), EASYSIMD_FLOAT32_C(-111112.98), EASYSIMD_FLOAT32_C(-55220.90), EASYSIMD_FLOAT32_C(715416.69),
        EASYSIMD_FLOAT32_C(-358008.12), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-506403.41),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-730227.94), EASYSIMD_FLOAT32_C(-106242.82), EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(  -741.88), EASYSIMD_FLOAT32_C(  -807.01), EASYSIMD_FLOAT32_C(   437.68), EASYSIMD_FLOAT32_C(  -566.58),
        EASYSIMD_FLOAT32_C(  -297.57), EASYSIMD_FLOAT32_C(  -626.78), EASYSIMD_FLOAT32_C(  -259.69), EASYSIMD_FLOAT32_C(  -183.87),
        EASYSIMD_FLOAT32_C(   476.22), EASYSIMD_FLOAT32_C(  -969.07), EASYSIMD_FLOAT32_C(     2.86), EASYSIMD_FLOAT32_C(  -522.24),
        EASYSIMD_FLOAT32_C(  -881.96), EASYSIMD_FLOAT32_C(   867.51), EASYSIMD_FLOAT32_C(  -653.41), EASYSIMD_FLOAT32_C(  -772.37) },
      UINT16_C( 6898),
      { EASYSIMD_FLOAT32_C(  -252.23), EASYSIMD_FLOAT32_C(   136.27), EASYSIMD_FLOAT32_C(   928.84), EASYSIMD_FLOAT32_C(   271.11),
        EASYSIMD_FLOAT32_C(  -467.55), EASYSIMD_FLOAT32_C(  -523.95), EASYSIMD_FLOAT32_C(   945.44), EASYSIMD_FLOAT32_C(  -871.94),
        EASYSIMD_FLOAT32_C(   233.74), EASYSIMD_FLOAT32_C(  -431.01), EASYSIMD_FLOAT32_C(   732.50), EASYSIMD_FLOAT32_C(  -154.10),
        EASYSIMD_FLOAT32_C(  -725.68), EASYSIMD_FLOAT32_C(   990.62), EASYSIMD_FLOAT32_C(    38.89), EASYSIMD_FLOAT32_C(   712.00) },
      { EASYSIMD_FLOAT32_C(  -575.96), EASYSIMD_FLOAT32_C(   741.32), EASYSIMD_FLOAT32_C(  -914.79), EASYSIMD_FLOAT32_C(   164.35),
        EASYSIMD_FLOAT32_C(  -442.54), EASYSIMD_FLOAT32_C(   561.43), EASYSIMD_FLOAT32_C(   195.28), EASYSIMD_FLOAT32_C(   560.31),
        EASYSIMD_FLOAT32_C(  -960.80), EASYSIMD_FLOAT32_C(   313.32), EASYSIMD_FLOAT32_C(   427.82), EASYSIMD_FLOAT32_C(  -614.21),
        EASYSIMD_FLOAT32_C(   540.94), EASYSIMD_FLOAT32_C(  -275.14), EASYSIMD_FLOAT32_C(  -854.96), EASYSIMD_FLOAT32_C(  -711.28) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(110712.58), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-139571.41), EASYSIMD_FLOAT32_C(-327839.97), EASYSIMD_FLOAT32_C(245716.59), EASYSIMD_FLOAT32_C(-159763.30),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-417365.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(-81091.40),
        EASYSIMD_FLOAT32_C(-639479.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 a = easysimd_mm512_loadu_ps(test_vec[i].a);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 b = easysimd_mm512_loadu_ps(test_vec[i].b);
    easysimd__m512 c = easysimd_mm512_loadu_ps(test_vec[i].c);
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_fnmadd_ps(k, a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_fnmadd_ps");
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
    easysimd__m512 r = easysimd_mm512_maskz_fnmadd_ps(k, a, b, c);

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
test_easysimd_mm512_fnmadd_pd (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 c[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -33.42), EASYSIMD_FLOAT64_C(    18.77), EASYSIMD_FLOAT64_C(   -64.80), EASYSIMD_FLOAT64_C(   -31.25),
        EASYSIMD_FLOAT64_C(    13.14), EASYSIMD_FLOAT64_C(   -25.39), EASYSIMD_FLOAT64_C(   -32.34), EASYSIMD_FLOAT64_C(     4.66) },
      { EASYSIMD_FLOAT64_C(    75.32), EASYSIMD_FLOAT64_C(    64.02), EASYSIMD_FLOAT64_C(    55.28), EASYSIMD_FLOAT64_C(    78.84),
        EASYSIMD_FLOAT64_C(    52.87), EASYSIMD_FLOAT64_C(   -80.75), EASYSIMD_FLOAT64_C(   -10.62), EASYSIMD_FLOAT64_C(    16.16) },
      { EASYSIMD_FLOAT64_C(    95.97), EASYSIMD_FLOAT64_C(    62.64), EASYSIMD_FLOAT64_C(   -14.97), EASYSIMD_FLOAT64_C(   -48.39),
        EASYSIMD_FLOAT64_C(    80.72), EASYSIMD_FLOAT64_C(    32.09), EASYSIMD_FLOAT64_C(    81.91), EASYSIMD_FLOAT64_C(    89.26) },
      { EASYSIMD_FLOAT64_C(  2613.16), EASYSIMD_FLOAT64_C( -1139.02), EASYSIMD_FLOAT64_C(  3567.17), EASYSIMD_FLOAT64_C(  2415.36),
        EASYSIMD_FLOAT64_C(  -613.99), EASYSIMD_FLOAT64_C( -2018.15), EASYSIMD_FLOAT64_C(  -261.54), EASYSIMD_FLOAT64_C(    13.95) } },
    { { EASYSIMD_FLOAT64_C(   -94.10), EASYSIMD_FLOAT64_C(    84.15), EASYSIMD_FLOAT64_C(     9.79), EASYSIMD_FLOAT64_C(    -9.15),
        EASYSIMD_FLOAT64_C(     9.78), EASYSIMD_FLOAT64_C(   -92.82), EASYSIMD_FLOAT64_C(   -64.66), EASYSIMD_FLOAT64_C(    76.36) },
      { EASYSIMD_FLOAT64_C(    25.95), EASYSIMD_FLOAT64_C(   -29.46), EASYSIMD_FLOAT64_C(   -54.89), EASYSIMD_FLOAT64_C(   -60.91),
        EASYSIMD_FLOAT64_C(    45.15), EASYSIMD_FLOAT64_C(    12.77), EASYSIMD_FLOAT64_C(    43.76), EASYSIMD_FLOAT64_C(    20.47) },
      { EASYSIMD_FLOAT64_C(   -23.21), EASYSIMD_FLOAT64_C(    -0.96), EASYSIMD_FLOAT64_C(    -0.69), EASYSIMD_FLOAT64_C(   -70.34),
        EASYSIMD_FLOAT64_C(    18.29), EASYSIMD_FLOAT64_C(    88.69), EASYSIMD_FLOAT64_C(    45.82), EASYSIMD_FLOAT64_C(    14.26) },
      { EASYSIMD_FLOAT64_C(  2418.68), EASYSIMD_FLOAT64_C(  2478.10), EASYSIMD_FLOAT64_C(   536.68), EASYSIMD_FLOAT64_C(  -627.67),
        EASYSIMD_FLOAT64_C(  -423.28), EASYSIMD_FLOAT64_C(  1274.00), EASYSIMD_FLOAT64_C(  2875.34), EASYSIMD_FLOAT64_C( -1548.83) } },
    { { EASYSIMD_FLOAT64_C(    51.33), EASYSIMD_FLOAT64_C(   -69.15), EASYSIMD_FLOAT64_C(    65.87), EASYSIMD_FLOAT64_C(    32.05),
        EASYSIMD_FLOAT64_C(    62.93), EASYSIMD_FLOAT64_C(    47.79), EASYSIMD_FLOAT64_C(    21.31), EASYSIMD_FLOAT64_C(    68.84) },
      { EASYSIMD_FLOAT64_C(    31.94), EASYSIMD_FLOAT64_C(   -68.91), EASYSIMD_FLOAT64_C(   -40.32), EASYSIMD_FLOAT64_C(   -58.28),
        EASYSIMD_FLOAT64_C(   -61.72), EASYSIMD_FLOAT64_C(    -4.98), EASYSIMD_FLOAT64_C(   -81.91), EASYSIMD_FLOAT64_C(    64.23) },
      { EASYSIMD_FLOAT64_C(    65.56), EASYSIMD_FLOAT64_C(   -36.80), EASYSIMD_FLOAT64_C(   -96.68), EASYSIMD_FLOAT64_C(    10.71),
        EASYSIMD_FLOAT64_C(    75.97), EASYSIMD_FLOAT64_C(    47.08), EASYSIMD_FLOAT64_C(   -68.83), EASYSIMD_FLOAT64_C(   -47.24) },
      { EASYSIMD_FLOAT64_C( -1573.92), EASYSIMD_FLOAT64_C( -4801.93), EASYSIMD_FLOAT64_C(  2559.20), EASYSIMD_FLOAT64_C(  1878.58),
        EASYSIMD_FLOAT64_C(  3960.01), EASYSIMD_FLOAT64_C(   285.07), EASYSIMD_FLOAT64_C(  1676.67), EASYSIMD_FLOAT64_C( -4468.83) } },
    { { EASYSIMD_FLOAT64_C(   -53.88), EASYSIMD_FLOAT64_C(    30.48), EASYSIMD_FLOAT64_C(   -17.58), EASYSIMD_FLOAT64_C(    64.42),
        EASYSIMD_FLOAT64_C(    19.17), EASYSIMD_FLOAT64_C(   -71.76), EASYSIMD_FLOAT64_C(   -21.32), EASYSIMD_FLOAT64_C(   -29.50) },
      { EASYSIMD_FLOAT64_C(   -40.91), EASYSIMD_FLOAT64_C(   -55.45), EASYSIMD_FLOAT64_C(   -97.45), EASYSIMD_FLOAT64_C(   -77.98),
        EASYSIMD_FLOAT64_C(    92.34), EASYSIMD_FLOAT64_C(    23.85), EASYSIMD_FLOAT64_C(    90.86), EASYSIMD_FLOAT64_C(    24.27) },
      { EASYSIMD_FLOAT64_C(    54.95), EASYSIMD_FLOAT64_C(   -49.46), EASYSIMD_FLOAT64_C(    66.00), EASYSIMD_FLOAT64_C(    93.22),
        EASYSIMD_FLOAT64_C(    45.56), EASYSIMD_FLOAT64_C(    84.08), EASYSIMD_FLOAT64_C(    57.45), EASYSIMD_FLOAT64_C(    11.12) },
      { EASYSIMD_FLOAT64_C( -2149.28), EASYSIMD_FLOAT64_C(  1640.66), EASYSIMD_FLOAT64_C( -1647.17), EASYSIMD_FLOAT64_C(  5116.69),
        EASYSIMD_FLOAT64_C( -1724.60), EASYSIMD_FLOAT64_C(  1795.56), EASYSIMD_FLOAT64_C(  1994.59), EASYSIMD_FLOAT64_C(   727.09) } },
    { { EASYSIMD_FLOAT64_C(   -52.72), EASYSIMD_FLOAT64_C(    60.77), EASYSIMD_FLOAT64_C(   -78.17), EASYSIMD_FLOAT64_C(   -76.75),
        EASYSIMD_FLOAT64_C(     7.85), EASYSIMD_FLOAT64_C(   -47.00), EASYSIMD_FLOAT64_C(   -23.99), EASYSIMD_FLOAT64_C(    53.98) },
      { EASYSIMD_FLOAT64_C(    83.49), EASYSIMD_FLOAT64_C(    58.43), EASYSIMD_FLOAT64_C(    18.39), EASYSIMD_FLOAT64_C(     2.66),
        EASYSIMD_FLOAT64_C(    86.67), EASYSIMD_FLOAT64_C(    97.07), EASYSIMD_FLOAT64_C(    73.16), EASYSIMD_FLOAT64_C(   -54.24) },
      { EASYSIMD_FLOAT64_C(   -58.38), EASYSIMD_FLOAT64_C(    75.70), EASYSIMD_FLOAT64_C(   -32.22), EASYSIMD_FLOAT64_C(   -66.05),
        EASYSIMD_FLOAT64_C(    -0.44), EASYSIMD_FLOAT64_C(   -41.36), EASYSIMD_FLOAT64_C(    58.23), EASYSIMD_FLOAT64_C(   -45.50) },
      { EASYSIMD_FLOAT64_C(  4343.21), EASYSIMD_FLOAT64_C( -3475.09), EASYSIMD_FLOAT64_C(  1405.33), EASYSIMD_FLOAT64_C(   138.11),
        EASYSIMD_FLOAT64_C(  -680.80), EASYSIMD_FLOAT64_C(  4520.93), EASYSIMD_FLOAT64_C(  1813.34), EASYSIMD_FLOAT64_C(  2882.38) } },
    { { EASYSIMD_FLOAT64_C(     9.18), EASYSIMD_FLOAT64_C(    24.22), EASYSIMD_FLOAT64_C(   -52.28), EASYSIMD_FLOAT64_C(   -45.26),
        EASYSIMD_FLOAT64_C(     8.30), EASYSIMD_FLOAT64_C(   -94.83), EASYSIMD_FLOAT64_C(    65.86), EASYSIMD_FLOAT64_C(    55.58) },
      { EASYSIMD_FLOAT64_C(    65.95), EASYSIMD_FLOAT64_C(    87.69), EASYSIMD_FLOAT64_C(    78.83), EASYSIMD_FLOAT64_C(   -26.20),
        EASYSIMD_FLOAT64_C(   -59.31), EASYSIMD_FLOAT64_C(   -45.15), EASYSIMD_FLOAT64_C(   -72.22), EASYSIMD_FLOAT64_C(   -75.82) },
      { EASYSIMD_FLOAT64_C(   -86.72), EASYSIMD_FLOAT64_C(    46.17), EASYSIMD_FLOAT64_C(    26.84), EASYSIMD_FLOAT64_C(    99.95),
        EASYSIMD_FLOAT64_C(    43.24), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(   -54.29), EASYSIMD_FLOAT64_C(    84.85) },
      { EASYSIMD_FLOAT64_C(  -692.14), EASYSIMD_FLOAT64_C( -2077.68), EASYSIMD_FLOAT64_C(  4148.07), EASYSIMD_FLOAT64_C( -1085.86),
        EASYSIMD_FLOAT64_C(   535.51), EASYSIMD_FLOAT64_C( -4281.57), EASYSIMD_FLOAT64_C(  4702.12), EASYSIMD_FLOAT64_C(  4298.93) } },
    { { EASYSIMD_FLOAT64_C(   -24.30), EASYSIMD_FLOAT64_C(    13.49), EASYSIMD_FLOAT64_C(   -81.19), EASYSIMD_FLOAT64_C(    75.26),
        EASYSIMD_FLOAT64_C(    72.13), EASYSIMD_FLOAT64_C(    77.04), EASYSIMD_FLOAT64_C(   -70.24), EASYSIMD_FLOAT64_C(   -18.69) },
      { EASYSIMD_FLOAT64_C(     1.26), EASYSIMD_FLOAT64_C(   -22.52), EASYSIMD_FLOAT64_C(    36.06), EASYSIMD_FLOAT64_C(   -90.44),
        EASYSIMD_FLOAT64_C(   -17.34), EASYSIMD_FLOAT64_C(     1.92), EASYSIMD_FLOAT64_C(    65.15), EASYSIMD_FLOAT64_C(   -51.39) },
      { EASYSIMD_FLOAT64_C(   -10.39), EASYSIMD_FLOAT64_C(    43.98), EASYSIMD_FLOAT64_C(    22.41), EASYSIMD_FLOAT64_C(    30.30),
        EASYSIMD_FLOAT64_C(    98.83), EASYSIMD_FLOAT64_C(    50.19), EASYSIMD_FLOAT64_C(    54.48), EASYSIMD_FLOAT64_C(   -87.89) },
      { EASYSIMD_FLOAT64_C(    20.23), EASYSIMD_FLOAT64_C(   347.77), EASYSIMD_FLOAT64_C(  2950.12), EASYSIMD_FLOAT64_C(  6836.81),
        EASYSIMD_FLOAT64_C(  1349.56), EASYSIMD_FLOAT64_C(   -97.73), EASYSIMD_FLOAT64_C(  4630.62), EASYSIMD_FLOAT64_C( -1048.37) } },
    { { EASYSIMD_FLOAT64_C(    -3.64), EASYSIMD_FLOAT64_C(   -18.68), EASYSIMD_FLOAT64_C(   -87.95), EASYSIMD_FLOAT64_C(   -60.40),
        EASYSIMD_FLOAT64_C(    81.32), EASYSIMD_FLOAT64_C(   -42.24), EASYSIMD_FLOAT64_C(   -75.55), EASYSIMD_FLOAT64_C(   -42.97) },
      { EASYSIMD_FLOAT64_C(    71.25), EASYSIMD_FLOAT64_C(   -56.74), EASYSIMD_FLOAT64_C(   -67.71), EASYSIMD_FLOAT64_C(    43.39),
        EASYSIMD_FLOAT64_C(   -79.71), EASYSIMD_FLOAT64_C(   -37.95), EASYSIMD_FLOAT64_C(   -75.30), EASYSIMD_FLOAT64_C(    21.55) },
      { EASYSIMD_FLOAT64_C(    39.53), EASYSIMD_FLOAT64_C(    60.76), EASYSIMD_FLOAT64_C(    31.12), EASYSIMD_FLOAT64_C(   -77.81),
        EASYSIMD_FLOAT64_C(   -37.33), EASYSIMD_FLOAT64_C(    -3.74), EASYSIMD_FLOAT64_C(   -29.20), EASYSIMD_FLOAT64_C(    52.28) },
      { EASYSIMD_FLOAT64_C(   298.88), EASYSIMD_FLOAT64_C(  -999.14), EASYSIMD_FLOAT64_C( -5923.97), EASYSIMD_FLOAT64_C(  2542.95),
        EASYSIMD_FLOAT64_C(  6444.69), EASYSIMD_FLOAT64_C( -1606.75), EASYSIMD_FLOAT64_C( -5718.11), EASYSIMD_FLOAT64_C(   978.28) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512d a = easysimd_mm512_loadu_pd(test_vec[i].a);
    easysimd__m512d b = easysimd_mm512_loadu_pd(test_vec[i].b);
    easysimd__m512d c = easysimd_mm512_loadu_pd(test_vec[i].c);
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_fnmadd_pd(a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_fnmadd_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask3_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask3_fnmadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fnmadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fnmadd_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fnmadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fnmadd_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_fnmadd_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
