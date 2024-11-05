/* Copyright (c) 2019 Evan Nemerson <evan@nemerson.com>
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

#define EASYSIMD_TESTS_CURRENT_ISAX fma
#include <easysimd/x86/fma.h>
#include <test/x86/test-avx.h>

static int
test_easysimd_mm_fmadd_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -364.57), EASYSIMD_FLOAT64_C( -702.81)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -890.80), EASYSIMD_FLOAT64_C( -433.89)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  486.26), EASYSIMD_FLOAT64_C( -304.02)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(325245.22), EASYSIMD_FLOAT64_C(304638.21)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  278.35), EASYSIMD_FLOAT64_C( -601.69)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -850.98), EASYSIMD_FLOAT64_C(   10.48)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -876.47), EASYSIMD_FLOAT64_C( -253.46)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-237746.75), EASYSIMD_FLOAT64_C(-6559.17)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -495.32), EASYSIMD_FLOAT64_C(  626.54)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  243.15), EASYSIMD_FLOAT64_C( -595.67)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  652.27), EASYSIMD_FLOAT64_C(  684.47)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-119784.79), EASYSIMD_FLOAT64_C(-372526.61)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -675.35), EASYSIMD_FLOAT64_C( -855.85)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  863.96), EASYSIMD_FLOAT64_C( -244.88)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  266.15), EASYSIMD_FLOAT64_C( -217.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-583209.24), EASYSIMD_FLOAT64_C(209362.65)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -945.02), EASYSIMD_FLOAT64_C( -266.12)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  248.34), EASYSIMD_FLOAT64_C( -754.68)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  686.19), EASYSIMD_FLOAT64_C(  201.29)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-234000.08), EASYSIMD_FLOAT64_C(201036.73)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -923.40), EASYSIMD_FLOAT64_C(  347.92)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -299.09), EASYSIMD_FLOAT64_C( -322.35)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -172.16), EASYSIMD_FLOAT64_C(  792.83)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(276007.55), EASYSIMD_FLOAT64_C(-111359.18)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -997.96), EASYSIMD_FLOAT64_C( -774.36)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  336.57), EASYSIMD_FLOAT64_C( -666.28)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   18.66), EASYSIMD_FLOAT64_C(  857.72)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-335864.74), EASYSIMD_FLOAT64_C(516798.30)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  425.15), EASYSIMD_FLOAT64_C( -554.19)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -602.50), EASYSIMD_FLOAT64_C( -329.67)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -208.43), EASYSIMD_FLOAT64_C(  819.37)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-256361.30), EASYSIMD_FLOAT64_C(183519.19)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fmadd_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fmadd_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256d a;
    easysimd__m256d b;
    easysimd__m256d c;
    easysimd__m256d r;
  } test_vec[8] = {
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   463.71), EASYSIMD_FLOAT64_C(  -551.83),
                         EASYSIMD_FLOAT64_C(   568.05), EASYSIMD_FLOAT64_C(  -826.17)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   440.29), EASYSIMD_FLOAT64_C(   762.39),
                         EASYSIMD_FLOAT64_C(  -806.23), EASYSIMD_FLOAT64_C(  -848.48)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   222.93), EASYSIMD_FLOAT64_C(  -604.06),
                         EASYSIMD_FLOAT64_C(  -844.49), EASYSIMD_FLOAT64_C(   221.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(204389.81), EASYSIMD_FLOAT64_C(-421313.73),
                         EASYSIMD_FLOAT64_C(-458823.44), EASYSIMD_FLOAT64_C(701210.22)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   418.00), EASYSIMD_FLOAT64_C(  -725.82),
                         EASYSIMD_FLOAT64_C(   -54.90), EASYSIMD_FLOAT64_C(  -342.22)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   280.70), EASYSIMD_FLOAT64_C(   983.58),
                         EASYSIMD_FLOAT64_C(  -289.88), EASYSIMD_FLOAT64_C(   305.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -404.92), EASYSIMD_FLOAT64_C(  -664.17),
                         EASYSIMD_FLOAT64_C(   164.15), EASYSIMD_FLOAT64_C(  -785.83)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(116927.68), EASYSIMD_FLOAT64_C(-714566.21),
                         EASYSIMD_FLOAT64_C( 16078.56), EASYSIMD_FLOAT64_C(-105265.60)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   267.84), EASYSIMD_FLOAT64_C(   153.22),
                         EASYSIMD_FLOAT64_C(   565.53), EASYSIMD_FLOAT64_C(    45.62)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   253.29), EASYSIMD_FLOAT64_C(  -448.85),
                         EASYSIMD_FLOAT64_C(  -379.10), EASYSIMD_FLOAT64_C(   896.99)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   305.51), EASYSIMD_FLOAT64_C(   -18.42),
                         EASYSIMD_FLOAT64_C(   560.02), EASYSIMD_FLOAT64_C(  -441.54)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 68146.70), EASYSIMD_FLOAT64_C(-68791.22),
                         EASYSIMD_FLOAT64_C(-213832.40), EASYSIMD_FLOAT64_C( 40479.14)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   698.02), EASYSIMD_FLOAT64_C(  -282.65),
                         EASYSIMD_FLOAT64_C(  -531.77), EASYSIMD_FLOAT64_C(  -673.05)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -227.36), EASYSIMD_FLOAT64_C(   165.86),
                         EASYSIMD_FLOAT64_C(  -853.86), EASYSIMD_FLOAT64_C(   210.39)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -443.02), EASYSIMD_FLOAT64_C(  -362.32),
                         EASYSIMD_FLOAT64_C(   833.55), EASYSIMD_FLOAT64_C(   692.62)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-159144.85), EASYSIMD_FLOAT64_C(-47242.65),
                         EASYSIMD_FLOAT64_C(454890.68), EASYSIMD_FLOAT64_C(-140910.37)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -601.68), EASYSIMD_FLOAT64_C(   654.88),
                         EASYSIMD_FLOAT64_C(   957.42), EASYSIMD_FLOAT64_C(   563.37)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -958.65), EASYSIMD_FLOAT64_C(   523.00),
                         EASYSIMD_FLOAT64_C(  -211.18), EASYSIMD_FLOAT64_C(  -889.28)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   180.36), EASYSIMD_FLOAT64_C(   481.63),
                         EASYSIMD_FLOAT64_C(  -222.77), EASYSIMD_FLOAT64_C(   -51.21)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(576980.89), EASYSIMD_FLOAT64_C(342983.87),
                         EASYSIMD_FLOAT64_C(-202410.73), EASYSIMD_FLOAT64_C(-501044.88)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   731.42), EASYSIMD_FLOAT64_C(  -631.15),
                         EASYSIMD_FLOAT64_C(  -982.89), EASYSIMD_FLOAT64_C(  -397.65)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    69.37), EASYSIMD_FLOAT64_C(  -394.43),
                         EASYSIMD_FLOAT64_C(   -18.09), EASYSIMD_FLOAT64_C(   272.24)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   920.14), EASYSIMD_FLOAT64_C(  -196.58),
                         EASYSIMD_FLOAT64_C(   324.68), EASYSIMD_FLOAT64_C(  -193.62)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 51658.75), EASYSIMD_FLOAT64_C(248747.91),
                         EASYSIMD_FLOAT64_C( 18105.16), EASYSIMD_FLOAT64_C(-108449.86)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -561.20), EASYSIMD_FLOAT64_C(  -459.54),
                         EASYSIMD_FLOAT64_C(  -681.08), EASYSIMD_FLOAT64_C(   -49.66)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   197.69), EASYSIMD_FLOAT64_C(  -813.71),
                         EASYSIMD_FLOAT64_C(  -990.48), EASYSIMD_FLOAT64_C(  -180.87)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -453.27), EASYSIMD_FLOAT64_C(  -557.45),
                         EASYSIMD_FLOAT64_C(  -780.15), EASYSIMD_FLOAT64_C(   693.73)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-111396.90), EASYSIMD_FLOAT64_C(373374.84),
                         EASYSIMD_FLOAT64_C(673815.97), EASYSIMD_FLOAT64_C(  9675.73)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   297.04), EASYSIMD_FLOAT64_C(   950.40),
                         EASYSIMD_FLOAT64_C(  -454.41), EASYSIMD_FLOAT64_C(   419.22)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   303.25), EASYSIMD_FLOAT64_C(  -917.33),
                         EASYSIMD_FLOAT64_C(   128.78), EASYSIMD_FLOAT64_C(   208.96)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   735.53), EASYSIMD_FLOAT64_C(   976.90),
                         EASYSIMD_FLOAT64_C(   803.26), EASYSIMD_FLOAT64_C(   610.54)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 90812.91), EASYSIMD_FLOAT64_C(-870853.53),
                         EASYSIMD_FLOAT64_C(-57715.66), EASYSIMD_FLOAT64_C( 88210.75)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d r = easysimd_mm256_fmadd_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmadd_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    68.47), EASYSIMD_FLOAT32_C(   -20.99), EASYSIMD_FLOAT32_C(  -768.39), EASYSIMD_FLOAT32_C(   464.52)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   297.92), EASYSIMD_FLOAT32_C(   902.90), EASYSIMD_FLOAT32_C(   496.10), EASYSIMD_FLOAT32_C(  -932.73)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -341.38), EASYSIMD_FLOAT32_C(  -852.40), EASYSIMD_FLOAT32_C(   426.68), EASYSIMD_FLOAT32_C(   755.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 20057.20), EASYSIMD_FLOAT32_C(-19804.27), EASYSIMD_FLOAT32_C(-380771.59), EASYSIMD_FLOAT32_C(-432516.62)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   969.18), EASYSIMD_FLOAT32_C(   318.32), EASYSIMD_FLOAT32_C(  -273.65), EASYSIMD_FLOAT32_C(    39.39)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   995.10), EASYSIMD_FLOAT32_C(   620.67), EASYSIMD_FLOAT32_C(   664.82), EASYSIMD_FLOAT32_C(   711.85)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   473.88), EASYSIMD_FLOAT32_C(   360.15), EASYSIMD_FLOAT32_C(  -250.82), EASYSIMD_FLOAT32_C(   -88.76)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(964904.88), EASYSIMD_FLOAT32_C(197931.83), EASYSIMD_FLOAT32_C(-182178.80), EASYSIMD_FLOAT32_C( 27951.01)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   164.14), EASYSIMD_FLOAT32_C(  -848.02), EASYSIMD_FLOAT32_C(   235.35), EASYSIMD_FLOAT32_C(  -999.97)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   466.54), EASYSIMD_FLOAT32_C(    41.59), EASYSIMD_FLOAT32_C(  -619.09), EASYSIMD_FLOAT32_C(   332.19)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -720.05), EASYSIMD_FLOAT32_C(    91.37), EASYSIMD_FLOAT32_C(     3.41), EASYSIMD_FLOAT32_C(  -151.75)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 75857.83), EASYSIMD_FLOAT32_C(-35177.78), EASYSIMD_FLOAT32_C(-145699.44), EASYSIMD_FLOAT32_C(-332331.78)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -664.98), EASYSIMD_FLOAT32_C(  -765.11), EASYSIMD_FLOAT32_C(  -950.95), EASYSIMD_FLOAT32_C(   967.68)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   463.42), EASYSIMD_FLOAT32_C(   310.01), EASYSIMD_FLOAT32_C(  -859.78), EASYSIMD_FLOAT32_C(  -247.59)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   372.25), EASYSIMD_FLOAT32_C(  -546.43), EASYSIMD_FLOAT32_C(   -18.65), EASYSIMD_FLOAT32_C(  -608.78)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-307792.78), EASYSIMD_FLOAT32_C(-237738.19), EASYSIMD_FLOAT32_C(817589.19), EASYSIMD_FLOAT32_C(-240196.67)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   739.81), EASYSIMD_FLOAT32_C(  -275.42), EASYSIMD_FLOAT32_C(  -462.27), EASYSIMD_FLOAT32_C(  -299.55)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -308.92), EASYSIMD_FLOAT32_C(   948.18), EASYSIMD_FLOAT32_C(  -344.73), EASYSIMD_FLOAT32_C(  -942.49)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   780.24), EASYSIMD_FLOAT32_C(   819.52), EASYSIMD_FLOAT32_C(  -913.65), EASYSIMD_FLOAT32_C(   715.95)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-227761.88), EASYSIMD_FLOAT32_C(-260328.23), EASYSIMD_FLOAT32_C(158444.69), EASYSIMD_FLOAT32_C(283038.81)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -766.06), EASYSIMD_FLOAT32_C(  -563.42), EASYSIMD_FLOAT32_C(  -122.27), EASYSIMD_FLOAT32_C(  -338.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   496.12), EASYSIMD_FLOAT32_C(  -751.97), EASYSIMD_FLOAT32_C(   655.86), EASYSIMD_FLOAT32_C(   174.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -985.27), EASYSIMD_FLOAT32_C(   574.75), EASYSIMD_FLOAT32_C(   212.10), EASYSIMD_FLOAT32_C(  -683.32)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-381042.97), EASYSIMD_FLOAT32_C(424249.66), EASYSIMD_FLOAT32_C(-79979.90), EASYSIMD_FLOAT32_C(-59630.52)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -663.22), EASYSIMD_FLOAT32_C(   549.14), EASYSIMD_FLOAT32_C(   733.90), EASYSIMD_FLOAT32_C(   785.76)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -149.60), EASYSIMD_FLOAT32_C(  -221.89), EASYSIMD_FLOAT32_C(  -452.29), EASYSIMD_FLOAT32_C(   -18.14)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   979.50), EASYSIMD_FLOAT32_C(  -484.31), EASYSIMD_FLOAT32_C(  -965.78), EASYSIMD_FLOAT32_C(  -291.28)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(100197.21), EASYSIMD_FLOAT32_C(-122332.99), EASYSIMD_FLOAT32_C(-332901.44), EASYSIMD_FLOAT32_C(-14544.97)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    82.89), EASYSIMD_FLOAT32_C(  -639.53), EASYSIMD_FLOAT32_C(   680.97), EASYSIMD_FLOAT32_C(  -745.76)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   377.68), EASYSIMD_FLOAT32_C(  -229.15), EASYSIMD_FLOAT32_C(   986.42), EASYSIMD_FLOAT32_C(  -430.87)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   506.24), EASYSIMD_FLOAT32_C(  -791.48), EASYSIMD_FLOAT32_C(  -896.55), EASYSIMD_FLOAT32_C(  -775.82)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 31812.13), EASYSIMD_FLOAT32_C(145756.81), EASYSIMD_FLOAT32_C(670825.81), EASYSIMD_FLOAT32_C(320549.81)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fmadd_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fmadd_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256 a;
    easysimd__m256 b;
    easysimd__m256 c;
    easysimd__m256 r;
  } test_vec[8] = {
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    39.90), EASYSIMD_FLOAT32_C(    46.80),
                         EASYSIMD_FLOAT32_C(   -90.30), EASYSIMD_FLOAT32_C(   -57.20),
                         EASYSIMD_FLOAT32_C(    71.50), EASYSIMD_FLOAT32_C(    75.00),
                         EASYSIMD_FLOAT32_C(    -0.30), EASYSIMD_FLOAT32_C(    14.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -90.60), EASYSIMD_FLOAT32_C(   -15.30),
                         EASYSIMD_FLOAT32_C(   -46.70), EASYSIMD_FLOAT32_C(    73.50),
                         EASYSIMD_FLOAT32_C(   -27.40), EASYSIMD_FLOAT32_C(   -79.00),
                         EASYSIMD_FLOAT32_C(   -14.10), EASYSIMD_FLOAT32_C(    22.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -19.50), EASYSIMD_FLOAT32_C(    61.50),
                         EASYSIMD_FLOAT32_C(   -38.80), EASYSIMD_FLOAT32_C(   -19.20),
                         EASYSIMD_FLOAT32_C(    54.40), EASYSIMD_FLOAT32_C(   -71.00),
                         EASYSIMD_FLOAT32_C(   -11.30), EASYSIMD_FLOAT32_C(    -2.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -3634.44), EASYSIMD_FLOAT32_C(  -654.54),
                         EASYSIMD_FLOAT32_C(  4178.21), EASYSIMD_FLOAT32_C( -4223.40),
                         EASYSIMD_FLOAT32_C( -1904.70), EASYSIMD_FLOAT32_C( -5996.00),
                         EASYSIMD_FLOAT32_C(    -7.07), EASYSIMD_FLOAT32_C(   322.88)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    56.00), EASYSIMD_FLOAT32_C(   -61.10),
                         EASYSIMD_FLOAT32_C(   -84.20), EASYSIMD_FLOAT32_C(    -8.30),
                         EASYSIMD_FLOAT32_C(    96.60), EASYSIMD_FLOAT32_C(    92.70),
                         EASYSIMD_FLOAT32_C(   -19.40), EASYSIMD_FLOAT32_C(   -41.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -20.80), EASYSIMD_FLOAT32_C(   -77.90),
                         EASYSIMD_FLOAT32_C(    22.80), EASYSIMD_FLOAT32_C(   -62.40),
                         EASYSIMD_FLOAT32_C(    47.20), EASYSIMD_FLOAT32_C(    23.30),
                         EASYSIMD_FLOAT32_C(   -14.70), EASYSIMD_FLOAT32_C(     1.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    -2.50), EASYSIMD_FLOAT32_C(   -40.20),
                         EASYSIMD_FLOAT32_C(   -64.40), EASYSIMD_FLOAT32_C(    46.00),
                         EASYSIMD_FLOAT32_C(    19.60), EASYSIMD_FLOAT32_C(    30.00),
                         EASYSIMD_FLOAT32_C(    23.60), EASYSIMD_FLOAT32_C(    20.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -1167.30), EASYSIMD_FLOAT32_C(  4719.49),
                         EASYSIMD_FLOAT32_C( -1984.16), EASYSIMD_FLOAT32_C(   563.92),
                         EASYSIMD_FLOAT32_C(  4579.12), EASYSIMD_FLOAT32_C(  2189.91),
                         EASYSIMD_FLOAT32_C(   308.78), EASYSIMD_FLOAT32_C(   -53.74)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -73.60), EASYSIMD_FLOAT32_C(   -63.70),
                         EASYSIMD_FLOAT32_C(    -7.10), EASYSIMD_FLOAT32_C(   -70.90),
                         EASYSIMD_FLOAT32_C(    23.30), EASYSIMD_FLOAT32_C(    22.20),
                         EASYSIMD_FLOAT32_C(     4.90), EASYSIMD_FLOAT32_C(   -85.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    75.60), EASYSIMD_FLOAT32_C(    -6.90),
                         EASYSIMD_FLOAT32_C(    73.70), EASYSIMD_FLOAT32_C(   -85.70),
                         EASYSIMD_FLOAT32_C(   -25.90), EASYSIMD_FLOAT32_C(   -59.90),
                         EASYSIMD_FLOAT32_C(   -56.20), EASYSIMD_FLOAT32_C(   -30.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    54.00), EASYSIMD_FLOAT32_C(   -79.70),
                         EASYSIMD_FLOAT32_C(    71.20), EASYSIMD_FLOAT32_C(   -74.20),
                         EASYSIMD_FLOAT32_C(   -48.90), EASYSIMD_FLOAT32_C(    -7.20),
                         EASYSIMD_FLOAT32_C(   -59.10), EASYSIMD_FLOAT32_C(   -84.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -5510.16), EASYSIMD_FLOAT32_C(   359.83),
                         EASYSIMD_FLOAT32_C(  -452.07), EASYSIMD_FLOAT32_C(  6001.93),
                         EASYSIMD_FLOAT32_C(  -652.37), EASYSIMD_FLOAT32_C( -1336.98),
                         EASYSIMD_FLOAT32_C(  -334.48), EASYSIMD_FLOAT32_C(  2534.01)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    57.50), EASYSIMD_FLOAT32_C(    93.40),
                         EASYSIMD_FLOAT32_C(    -2.20), EASYSIMD_FLOAT32_C(    77.20),
                         EASYSIMD_FLOAT32_C(    79.40), EASYSIMD_FLOAT32_C(   -81.10),
                         EASYSIMD_FLOAT32_C(    25.80), EASYSIMD_FLOAT32_C(    -5.40)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -36.80), EASYSIMD_FLOAT32_C(   -46.10),
                         EASYSIMD_FLOAT32_C(    57.50), EASYSIMD_FLOAT32_C(    47.70),
                         EASYSIMD_FLOAT32_C(    38.00), EASYSIMD_FLOAT32_C(    48.30),
                         EASYSIMD_FLOAT32_C(    86.60), EASYSIMD_FLOAT32_C(    85.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    92.60), EASYSIMD_FLOAT32_C(    68.60),
                         EASYSIMD_FLOAT32_C(   -48.10), EASYSIMD_FLOAT32_C(   -53.80),
                         EASYSIMD_FLOAT32_C(   -45.80), EASYSIMD_FLOAT32_C(    33.60),
                         EASYSIMD_FLOAT32_C(    47.80), EASYSIMD_FLOAT32_C(    61.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -2023.40), EASYSIMD_FLOAT32_C( -4237.14),
                         EASYSIMD_FLOAT32_C(  -174.60), EASYSIMD_FLOAT32_C(  3628.64),
                         EASYSIMD_FLOAT32_C(  2971.40), EASYSIMD_FLOAT32_C( -3883.53),
                         EASYSIMD_FLOAT32_C(  2282.08), EASYSIMD_FLOAT32_C(  -400.94)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    39.30), EASYSIMD_FLOAT32_C(    47.70),
                         EASYSIMD_FLOAT32_C(   -46.40), EASYSIMD_FLOAT32_C(    22.40),
                         EASYSIMD_FLOAT32_C(   -47.70), EASYSIMD_FLOAT32_C(   -87.50),
                         EASYSIMD_FLOAT32_C(    56.70), EASYSIMD_FLOAT32_C(   -98.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    47.80), EASYSIMD_FLOAT32_C(    25.10),
                         EASYSIMD_FLOAT32_C(    86.40), EASYSIMD_FLOAT32_C(    20.80),
                         EASYSIMD_FLOAT32_C(   -68.30), EASYSIMD_FLOAT32_C(    -7.70),
                         EASYSIMD_FLOAT32_C(    87.10), EASYSIMD_FLOAT32_C(    24.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    30.50), EASYSIMD_FLOAT32_C(    80.40),
                         EASYSIMD_FLOAT32_C(   -81.20), EASYSIMD_FLOAT32_C(   -60.10),
                         EASYSIMD_FLOAT32_C(   -62.20), EASYSIMD_FLOAT32_C(    51.30),
                         EASYSIMD_FLOAT32_C(   -56.00), EASYSIMD_FLOAT32_C(   -52.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  1909.04), EASYSIMD_FLOAT32_C(  1277.67),
                         EASYSIMD_FLOAT32_C( -4090.16), EASYSIMD_FLOAT32_C(   405.82),
                         EASYSIMD_FLOAT32_C(  3195.71), EASYSIMD_FLOAT32_C(   725.05),
                         EASYSIMD_FLOAT32_C(  4882.57), EASYSIMD_FLOAT32_C( -2412.10)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    35.30), EASYSIMD_FLOAT32_C(   -51.40),
                         EASYSIMD_FLOAT32_C(   -71.80), EASYSIMD_FLOAT32_C(    28.30),
                         EASYSIMD_FLOAT32_C(    41.70), EASYSIMD_FLOAT32_C(   -29.90),
                         EASYSIMD_FLOAT32_C(    47.10), EASYSIMD_FLOAT32_C(   -23.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -72.20), EASYSIMD_FLOAT32_C(     5.10),
                         EASYSIMD_FLOAT32_C(    50.30), EASYSIMD_FLOAT32_C(     8.80),
                         EASYSIMD_FLOAT32_C(    10.30), EASYSIMD_FLOAT32_C(    88.00),
                         EASYSIMD_FLOAT32_C(   -32.10), EASYSIMD_FLOAT32_C(   -71.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    92.50), EASYSIMD_FLOAT32_C(   -22.70),
                         EASYSIMD_FLOAT32_C(   -32.50), EASYSIMD_FLOAT32_C(   -64.00),
                         EASYSIMD_FLOAT32_C(    53.40), EASYSIMD_FLOAT32_C(    57.00),
                         EASYSIMD_FLOAT32_C(    85.20), EASYSIMD_FLOAT32_C(    51.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -2456.16), EASYSIMD_FLOAT32_C(  -284.84),
                         EASYSIMD_FLOAT32_C( -3644.04), EASYSIMD_FLOAT32_C(   185.04),
                         EASYSIMD_FLOAT32_C(   482.91), EASYSIMD_FLOAT32_C( -2574.20),
                         EASYSIMD_FLOAT32_C( -1426.71), EASYSIMD_FLOAT32_C(  1739.20)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    62.00), EASYSIMD_FLOAT32_C(   -58.50),
                         EASYSIMD_FLOAT32_C(   -89.10), EASYSIMD_FLOAT32_C(    51.50),
                         EASYSIMD_FLOAT32_C(     2.30), EASYSIMD_FLOAT32_C(   -87.50),
                         EASYSIMD_FLOAT32_C(   -72.60), EASYSIMD_FLOAT32_C(    96.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -25.70), EASYSIMD_FLOAT32_C(    80.90),
                         EASYSIMD_FLOAT32_C(   -77.80), EASYSIMD_FLOAT32_C(     4.90),
                         EASYSIMD_FLOAT32_C(    70.20), EASYSIMD_FLOAT32_C(    32.70),
                         EASYSIMD_FLOAT32_C(   -60.70), EASYSIMD_FLOAT32_C(    68.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -99.00), EASYSIMD_FLOAT32_C(   -12.20),
                         EASYSIMD_FLOAT32_C(    41.70), EASYSIMD_FLOAT32_C(     9.80),
                         EASYSIMD_FLOAT32_C(   -34.40), EASYSIMD_FLOAT32_C(   -50.10),
                         EASYSIMD_FLOAT32_C(    35.40), EASYSIMD_FLOAT32_C(    62.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -1692.40), EASYSIMD_FLOAT32_C( -4744.85),
                         EASYSIMD_FLOAT32_C(  6973.68), EASYSIMD_FLOAT32_C(   262.15),
                         EASYSIMD_FLOAT32_C(   127.06), EASYSIMD_FLOAT32_C( -2911.35),
                         EASYSIMD_FLOAT32_C(  4442.22), EASYSIMD_FLOAT32_C(  6611.00)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -40.00), EASYSIMD_FLOAT32_C(    62.80),
                         EASYSIMD_FLOAT32_C(   -40.00), EASYSIMD_FLOAT32_C(    16.60),
                         EASYSIMD_FLOAT32_C(    60.10), EASYSIMD_FLOAT32_C(    22.60),
                         EASYSIMD_FLOAT32_C(   -12.40), EASYSIMD_FLOAT32_C(    91.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -98.70), EASYSIMD_FLOAT32_C(    17.00),
                         EASYSIMD_FLOAT32_C(   -23.90), EASYSIMD_FLOAT32_C(    29.60),
                         EASYSIMD_FLOAT32_C(   -52.60), EASYSIMD_FLOAT32_C(   -30.60),
                         EASYSIMD_FLOAT32_C(    43.40), EASYSIMD_FLOAT32_C(    76.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    61.00), EASYSIMD_FLOAT32_C(   -10.10),
                         EASYSIMD_FLOAT32_C(    48.20), EASYSIMD_FLOAT32_C(    50.20),
                         EASYSIMD_FLOAT32_C(    12.20), EASYSIMD_FLOAT32_C(    64.80),
                         EASYSIMD_FLOAT32_C(   -68.90), EASYSIMD_FLOAT32_C(   -86.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  4009.00), EASYSIMD_FLOAT32_C(  1057.50),
                         EASYSIMD_FLOAT32_C(  1004.20), EASYSIMD_FLOAT32_C(   541.56),
                         EASYSIMD_FLOAT32_C( -3149.06), EASYSIMD_FLOAT32_C(  -626.76),
                         EASYSIMD_FLOAT32_C(  -607.06), EASYSIMD_FLOAT32_C(  6898.45)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 a = test_vec[i].a;
    easysimd__m256 b = test_vec[i].b;
    easysimd__m256 c = test_vec[i].c;
    easysimd__m256 r;
#ifndef EASYSIMD_ENABLE_TEST_PERF
    r = easysimd_mm256_fmadd_ps(a, b, c);
#else
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_fmadd_ps(a, b, c);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_fmadd_ps");
#endif
    easysimd_assert_m256_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmadd_sd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   96.50), EASYSIMD_FLOAT64_C(  -99.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   16.90), EASYSIMD_FLOAT64_C(  -76.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   53.90), EASYSIMD_FLOAT64_C(    6.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   96.50), EASYSIMD_FLOAT64_C( 7648.00)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   53.00), EASYSIMD_FLOAT64_C(   -2.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -68.70), EASYSIMD_FLOAT64_C(  -11.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -76.30), EASYSIMD_FLOAT64_C(   62.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   53.00), EASYSIMD_FLOAT64_C(   87.47)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   34.00), EASYSIMD_FLOAT64_C(   30.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -42.50), EASYSIMD_FLOAT64_C(   32.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -36.20), EASYSIMD_FLOAT64_C(   36.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   34.00), EASYSIMD_FLOAT64_C( 1033.45)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -75.50), EASYSIMD_FLOAT64_C(  -58.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   29.70), EASYSIMD_FLOAT64_C(  -42.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   95.20), EASYSIMD_FLOAT64_C(   92.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -75.50), EASYSIMD_FLOAT64_C( 2545.40)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -59.60), EASYSIMD_FLOAT64_C(   12.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -84.80), EASYSIMD_FLOAT64_C(   50.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   42.20), EASYSIMD_FLOAT64_C(  -77.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -59.60), EASYSIMD_FLOAT64_C(  572.86)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   78.40), EASYSIMD_FLOAT64_C(  -77.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -76.10), EASYSIMD_FLOAT64_C(  -11.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    5.80), EASYSIMD_FLOAT64_C(  -75.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   78.40), EASYSIMD_FLOAT64_C(  771.10)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    6.70), EASYSIMD_FLOAT64_C(   47.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   24.30), EASYSIMD_FLOAT64_C(   93.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   93.40), EASYSIMD_FLOAT64_C(  -50.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    6.70), EASYSIMD_FLOAT64_C( 4405.26)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -59.60), EASYSIMD_FLOAT64_C(  -73.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    1.20), EASYSIMD_FLOAT64_C(   10.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   45.00), EASYSIMD_FLOAT64_C(  -86.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -59.60), EASYSIMD_FLOAT64_C( -886.56)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fmadd_sd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmadd_ss(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   29.70), EASYSIMD_FLOAT32_C(  -13.10), EASYSIMD_FLOAT32_C(  -92.70), EASYSIMD_FLOAT32_C(   44.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   25.90), EASYSIMD_FLOAT32_C(   67.70), EASYSIMD_FLOAT32_C(  -12.20), EASYSIMD_FLOAT32_C(   72.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   45.60), EASYSIMD_FLOAT32_C(   36.90), EASYSIMD_FLOAT32_C(  -98.40), EASYSIMD_FLOAT32_C(  -64.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   29.70), EASYSIMD_FLOAT32_C(  -13.10), EASYSIMD_FLOAT32_C(  -92.70), EASYSIMD_FLOAT32_C( 3169.76)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   16.70), EASYSIMD_FLOAT32_C(   85.50), EASYSIMD_FLOAT32_C(   89.70), EASYSIMD_FLOAT32_C(  -23.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   70.80), EASYSIMD_FLOAT32_C(   99.80), EASYSIMD_FLOAT32_C(  -87.00), EASYSIMD_FLOAT32_C(    9.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   42.40), EASYSIMD_FLOAT32_C(   38.10), EASYSIMD_FLOAT32_C(  -58.60), EASYSIMD_FLOAT32_C(  -71.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   16.70), EASYSIMD_FLOAT32_C(   85.50), EASYSIMD_FLOAT32_C(   89.70), EASYSIMD_FLOAT32_C( -285.90)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -85.70), EASYSIMD_FLOAT32_C(   66.60), EASYSIMD_FLOAT32_C(  -84.60), EASYSIMD_FLOAT32_C(  -90.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   61.30), EASYSIMD_FLOAT32_C(  -91.00), EASYSIMD_FLOAT32_C(  -35.60), EASYSIMD_FLOAT32_C(  -66.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   76.30), EASYSIMD_FLOAT32_C(  -46.00), EASYSIMD_FLOAT32_C(   54.10), EASYSIMD_FLOAT32_C(   17.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -85.70), EASYSIMD_FLOAT32_C(   66.60), EASYSIMD_FLOAT32_C(  -84.60), EASYSIMD_FLOAT32_C( 6017.75)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   14.40), EASYSIMD_FLOAT32_C(  -25.60), EASYSIMD_FLOAT32_C(  -65.60), EASYSIMD_FLOAT32_C(  -71.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   65.40), EASYSIMD_FLOAT32_C(   95.90), EASYSIMD_FLOAT32_C(   51.70), EASYSIMD_FLOAT32_C(  -84.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -47.60), EASYSIMD_FLOAT32_C(  -50.00), EASYSIMD_FLOAT32_C(   88.40), EASYSIMD_FLOAT32_C(  -28.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   14.40), EASYSIMD_FLOAT32_C(  -25.60), EASYSIMD_FLOAT32_C(  -65.60), EASYSIMD_FLOAT32_C( 5970.70)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   29.50), EASYSIMD_FLOAT32_C(  -26.70), EASYSIMD_FLOAT32_C(    8.30), EASYSIMD_FLOAT32_C(  -34.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   76.70), EASYSIMD_FLOAT32_C(  -34.90), EASYSIMD_FLOAT32_C(  -78.80), EASYSIMD_FLOAT32_C(   84.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   45.30), EASYSIMD_FLOAT32_C(  -18.40), EASYSIMD_FLOAT32_C(  -36.50), EASYSIMD_FLOAT32_C(  -89.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   29.50), EASYSIMD_FLOAT32_C(  -26.70), EASYSIMD_FLOAT32_C(    8.30), EASYSIMD_FLOAT32_C(-3021.15)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   64.00), EASYSIMD_FLOAT32_C(   46.60), EASYSIMD_FLOAT32_C(  -17.50), EASYSIMD_FLOAT32_C(   24.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -67.40), EASYSIMD_FLOAT32_C(  -16.40), EASYSIMD_FLOAT32_C(   38.30), EASYSIMD_FLOAT32_C(  -92.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -66.80), EASYSIMD_FLOAT32_C(   10.60), EASYSIMD_FLOAT32_C(   -6.70), EASYSIMD_FLOAT32_C(  -49.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   64.00), EASYSIMD_FLOAT32_C(   46.60), EASYSIMD_FLOAT32_C(  -17.50), EASYSIMD_FLOAT32_C(-2273.43)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -98.50), EASYSIMD_FLOAT32_C(   15.30), EASYSIMD_FLOAT32_C(  -33.40), EASYSIMD_FLOAT32_C(    4.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   97.00), EASYSIMD_FLOAT32_C(  -35.60), EASYSIMD_FLOAT32_C(   63.50), EASYSIMD_FLOAT32_C(  -94.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -9.90), EASYSIMD_FLOAT32_C(  -97.20), EASYSIMD_FLOAT32_C(  -13.80), EASYSIMD_FLOAT32_C(   11.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -98.50), EASYSIMD_FLOAT32_C(   15.30), EASYSIMD_FLOAT32_C(  -33.40), EASYSIMD_FLOAT32_C( -441.04)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -11.00), EASYSIMD_FLOAT32_C(  -65.00), EASYSIMD_FLOAT32_C(  -76.20), EASYSIMD_FLOAT32_C(   54.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   37.10), EASYSIMD_FLOAT32_C(  -97.90), EASYSIMD_FLOAT32_C(  -36.50), EASYSIMD_FLOAT32_C(   50.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -26.80), EASYSIMD_FLOAT32_C(  -74.90), EASYSIMD_FLOAT32_C(  -84.40), EASYSIMD_FLOAT32_C(   35.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -11.00), EASYSIMD_FLOAT32_C(  -65.00), EASYSIMD_FLOAT32_C(  -76.20), EASYSIMD_FLOAT32_C( 2814.06)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fmadd_ss(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmaddsub_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -72.20), EASYSIMD_FLOAT64_C(   74.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   41.60), EASYSIMD_FLOAT64_C(  -13.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   18.90), EASYSIMD_FLOAT64_C(   65.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-2984.62), EASYSIMD_FLOAT64_C(-1069.70)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   14.70), EASYSIMD_FLOAT64_C(   97.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   47.70), EASYSIMD_FLOAT64_C(   86.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   75.80), EASYSIMD_FLOAT64_C(   19.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  776.99), EASYSIMD_FLOAT64_C( 8443.10)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -24.00), EASYSIMD_FLOAT64_C(   39.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -0.00), EASYSIMD_FLOAT64_C(   42.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -66.10), EASYSIMD_FLOAT64_C(  -55.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -66.10), EASYSIMD_FLOAT64_C( 1731.00)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -45.20), EASYSIMD_FLOAT64_C(   65.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -64.40), EASYSIMD_FLOAT64_C(   58.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   79.80), EASYSIMD_FLOAT64_C(   19.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 2990.68), EASYSIMD_FLOAT64_C( 3756.50)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   14.50), EASYSIMD_FLOAT64_C(  -64.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   92.20), EASYSIMD_FLOAT64_C(  -68.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -2.50), EASYSIMD_FLOAT64_C(  -96.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 1334.40), EASYSIMD_FLOAT64_C( 4561.52)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   61.50), EASYSIMD_FLOAT64_C(   42.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   77.20), EASYSIMD_FLOAT64_C(   23.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   32.20), EASYSIMD_FLOAT64_C(   12.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 4780.00), EASYSIMD_FLOAT64_C(  975.82)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -38.20), EASYSIMD_FLOAT64_C(    8.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -10.10), EASYSIMD_FLOAT64_C(   98.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -14.20), EASYSIMD_FLOAT64_C(   22.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  371.62), EASYSIMD_FLOAT64_C(  776.06)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    8.40), EASYSIMD_FLOAT64_C(  -30.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -39.30), EASYSIMD_FLOAT64_C(   73.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   25.30), EASYSIMD_FLOAT64_C(    2.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -304.82), EASYSIMD_FLOAT64_C(-2256.28)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fmaddsub_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fmaddsub_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256d a;
    easysimd__m256d b;
    easysimd__m256d c;
    easysimd__m256d r;
  } test_vec[8] = {
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -52.10), EASYSIMD_FLOAT64_C(  -92.00),
                         EASYSIMD_FLOAT64_C(  -82.90), EASYSIMD_FLOAT64_C(  -49.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -49.30), EASYSIMD_FLOAT64_C(  -97.40),
                         EASYSIMD_FLOAT64_C(   58.80), EASYSIMD_FLOAT64_C(   67.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   23.30), EASYSIMD_FLOAT64_C(   87.10),
                         EASYSIMD_FLOAT64_C(   71.70), EASYSIMD_FLOAT64_C(   97.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 2591.83), EASYSIMD_FLOAT64_C( 8873.70),
                         EASYSIMD_FLOAT64_C(-4802.82), EASYSIMD_FLOAT64_C(-3409.50)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -83.80), EASYSIMD_FLOAT64_C(   50.40),
                         EASYSIMD_FLOAT64_C(  -94.80), EASYSIMD_FLOAT64_C(  -86.80)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    3.10), EASYSIMD_FLOAT64_C(  -46.80),
                         EASYSIMD_FLOAT64_C(   -3.10), EASYSIMD_FLOAT64_C(   83.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -11.70), EASYSIMD_FLOAT64_C(   76.10),
                         EASYSIMD_FLOAT64_C(   44.50), EASYSIMD_FLOAT64_C(   28.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -271.48), EASYSIMD_FLOAT64_C(-2434.82),
                         EASYSIMD_FLOAT64_C(  338.38), EASYSIMD_FLOAT64_C(-7232.40)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -64.40), EASYSIMD_FLOAT64_C(   40.90),
                         EASYSIMD_FLOAT64_C(   36.80), EASYSIMD_FLOAT64_C(   -1.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -57.50), EASYSIMD_FLOAT64_C(    5.00),
                         EASYSIMD_FLOAT64_C(  -21.50), EASYSIMD_FLOAT64_C(   -1.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   21.60), EASYSIMD_FLOAT64_C(  -36.20),
                         EASYSIMD_FLOAT64_C(  -67.50), EASYSIMD_FLOAT64_C(  -19.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 3724.60), EASYSIMD_FLOAT64_C(  240.70),
                         EASYSIMD_FLOAT64_C( -858.70), EASYSIMD_FLOAT64_C(   21.00)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -37.10), EASYSIMD_FLOAT64_C(    2.20),
                         EASYSIMD_FLOAT64_C(  -99.10), EASYSIMD_FLOAT64_C(   78.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -60.20), EASYSIMD_FLOAT64_C(   29.30),
                         EASYSIMD_FLOAT64_C(    2.50), EASYSIMD_FLOAT64_C(  -40.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   59.60), EASYSIMD_FLOAT64_C(  -28.40),
                         EASYSIMD_FLOAT64_C(   58.10), EASYSIMD_FLOAT64_C(   96.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 2293.02), EASYSIMD_FLOAT64_C(   92.86),
                         EASYSIMD_FLOAT64_C( -189.65), EASYSIMD_FLOAT64_C(-3232.72)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -88.70), EASYSIMD_FLOAT64_C(  -20.50),
                         EASYSIMD_FLOAT64_C(   28.00), EASYSIMD_FLOAT64_C(  -13.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -49.60), EASYSIMD_FLOAT64_C(  -13.90),
                         EASYSIMD_FLOAT64_C(   71.80), EASYSIMD_FLOAT64_C(  -29.40)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -95.70), EASYSIMD_FLOAT64_C(   48.30),
                         EASYSIMD_FLOAT64_C(   78.20), EASYSIMD_FLOAT64_C(   -6.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 4303.82), EASYSIMD_FLOAT64_C(  236.65),
                         EASYSIMD_FLOAT64_C( 2088.60), EASYSIMD_FLOAT64_C(  408.78)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   74.60), EASYSIMD_FLOAT64_C(   40.20),
                         EASYSIMD_FLOAT64_C(   -4.40), EASYSIMD_FLOAT64_C(   51.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   21.60), EASYSIMD_FLOAT64_C(  -83.50),
                         EASYSIMD_FLOAT64_C(   -2.00), EASYSIMD_FLOAT64_C(   -6.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   13.70), EASYSIMD_FLOAT64_C(   39.10),
                         EASYSIMD_FLOAT64_C(   92.60), EASYSIMD_FLOAT64_C(  -41.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 1625.06), EASYSIMD_FLOAT64_C(-3395.80),
                         EASYSIMD_FLOAT64_C(  101.40), EASYSIMD_FLOAT64_C( -296.68)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -14.60), EASYSIMD_FLOAT64_C(  -32.40),
                         EASYSIMD_FLOAT64_C(   94.80), EASYSIMD_FLOAT64_C(   -5.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   15.40), EASYSIMD_FLOAT64_C(  -34.50),
                         EASYSIMD_FLOAT64_C(   91.60), EASYSIMD_FLOAT64_C(  -58.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -70.70), EASYSIMD_FLOAT64_C(  -91.10),
                         EASYSIMD_FLOAT64_C(  -42.30), EASYSIMD_FLOAT64_C(   64.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -295.54), EASYSIMD_FLOAT64_C( 1208.90),
                         EASYSIMD_FLOAT64_C( 8641.38), EASYSIMD_FLOAT64_C(  240.02)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   68.20), EASYSIMD_FLOAT64_C(  -45.40),
                         EASYSIMD_FLOAT64_C(   33.10), EASYSIMD_FLOAT64_C(   17.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   52.00), EASYSIMD_FLOAT64_C(   24.80),
                         EASYSIMD_FLOAT64_C(    6.10), EASYSIMD_FLOAT64_C(   68.80)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   30.10), EASYSIMD_FLOAT64_C(   11.20),
                         EASYSIMD_FLOAT64_C(  -78.00), EASYSIMD_FLOAT64_C(  -47.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 3576.50), EASYSIMD_FLOAT64_C(-1137.12),
                         EASYSIMD_FLOAT64_C(  123.91), EASYSIMD_FLOAT64_C( 1223.98)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d r = easysimd_mm256_fmaddsub_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmaddsub_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -75.30), EASYSIMD_FLOAT32_C(   37.60), EASYSIMD_FLOAT32_C(   76.00), EASYSIMD_FLOAT32_C(   -4.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -15.80), EASYSIMD_FLOAT32_C(   64.20), EASYSIMD_FLOAT32_C(   50.90), EASYSIMD_FLOAT32_C(   26.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   64.80), EASYSIMD_FLOAT32_C(  -10.00), EASYSIMD_FLOAT32_C(  -97.40), EASYSIMD_FLOAT32_C(  -90.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 1254.54), EASYSIMD_FLOAT32_C( 2423.92), EASYSIMD_FLOAT32_C( 3771.00), EASYSIMD_FLOAT32_C(  -19.32)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -48.70), EASYSIMD_FLOAT32_C(   50.40), EASYSIMD_FLOAT32_C(  -22.00), EASYSIMD_FLOAT32_C(   76.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -80.30), EASYSIMD_FLOAT32_C(  -99.30), EASYSIMD_FLOAT32_C(  -86.10), EASYSIMD_FLOAT32_C(   30.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -41.10), EASYSIMD_FLOAT32_C(   57.20), EASYSIMD_FLOAT32_C(  -41.90), EASYSIMD_FLOAT32_C(  -88.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 3869.51), EASYSIMD_FLOAT32_C(-5061.92), EASYSIMD_FLOAT32_C( 1852.30), EASYSIMD_FLOAT32_C( 2388.14)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   99.70), EASYSIMD_FLOAT32_C(    2.10), EASYSIMD_FLOAT32_C(   41.80), EASYSIMD_FLOAT32_C(  -15.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   16.20), EASYSIMD_FLOAT32_C(  -74.30), EASYSIMD_FLOAT32_C(  -71.40), EASYSIMD_FLOAT32_C(   51.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   71.10), EASYSIMD_FLOAT32_C(  -90.60), EASYSIMD_FLOAT32_C(  -33.50), EASYSIMD_FLOAT32_C(  -68.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 1686.24), EASYSIMD_FLOAT32_C(  -65.43), EASYSIMD_FLOAT32_C(-3018.02), EASYSIMD_FLOAT32_C( -717.44)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   39.90), EASYSIMD_FLOAT32_C(   12.10), EASYSIMD_FLOAT32_C(  -93.10), EASYSIMD_FLOAT32_C(  -73.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -72.70), EASYSIMD_FLOAT32_C(  -61.90), EASYSIMD_FLOAT32_C(    1.90), EASYSIMD_FLOAT32_C(   89.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -63.40), EASYSIMD_FLOAT32_C(  -46.10), EASYSIMD_FLOAT32_C(   50.20), EASYSIMD_FLOAT32_C(  -74.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-2964.13), EASYSIMD_FLOAT32_C( -702.89), EASYSIMD_FLOAT32_C( -126.69), EASYSIMD_FLOAT32_C(-6494.10)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -37.90), EASYSIMD_FLOAT32_C(   16.10), EASYSIMD_FLOAT32_C(   65.80), EASYSIMD_FLOAT32_C(   65.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -95.90), EASYSIMD_FLOAT32_C(    9.30), EASYSIMD_FLOAT32_C(   33.70), EASYSIMD_FLOAT32_C(  -30.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    4.30), EASYSIMD_FLOAT32_C(  -27.90), EASYSIMD_FLOAT32_C(  -62.30), EASYSIMD_FLOAT32_C(  -71.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 3638.91), EASYSIMD_FLOAT32_C(  177.63), EASYSIMD_FLOAT32_C( 2155.16), EASYSIMD_FLOAT32_C(-1936.76)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   40.20), EASYSIMD_FLOAT32_C(  -28.10), EASYSIMD_FLOAT32_C(  -39.20), EASYSIMD_FLOAT32_C(   15.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -33.70), EASYSIMD_FLOAT32_C(  -55.90), EASYSIMD_FLOAT32_C(   -9.80), EASYSIMD_FLOAT32_C(  -88.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    3.20), EASYSIMD_FLOAT32_C(  -50.90), EASYSIMD_FLOAT32_C(   35.30), EASYSIMD_FLOAT32_C(  -45.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-1351.54), EASYSIMD_FLOAT32_C( 1621.69), EASYSIMD_FLOAT32_C(  419.46), EASYSIMD_FLOAT32_C(-1277.70)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    2.60), EASYSIMD_FLOAT32_C(   70.50), EASYSIMD_FLOAT32_C(   56.20), EASYSIMD_FLOAT32_C(    5.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -66.40), EASYSIMD_FLOAT32_C(   95.00), EASYSIMD_FLOAT32_C(   95.50), EASYSIMD_FLOAT32_C(  -15.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -60.10), EASYSIMD_FLOAT32_C(  -25.30), EASYSIMD_FLOAT32_C(  -69.10), EASYSIMD_FLOAT32_C(  -77.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( -232.74), EASYSIMD_FLOAT32_C( 6722.80), EASYSIMD_FLOAT32_C( 5298.00), EASYSIMD_FLOAT32_C(  -16.11)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -91.20), EASYSIMD_FLOAT32_C(   32.90), EASYSIMD_FLOAT32_C(   -8.90), EASYSIMD_FLOAT32_C(  -97.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -78.50), EASYSIMD_FLOAT32_C(   49.50), EASYSIMD_FLOAT32_C(   63.70), EASYSIMD_FLOAT32_C(  -83.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   10.30), EASYSIMD_FLOAT32_C(   73.30), EASYSIMD_FLOAT32_C(  -68.20), EASYSIMD_FLOAT32_C(   60.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 7169.50), EASYSIMD_FLOAT32_C( 1555.25), EASYSIMD_FLOAT32_C( -635.13), EASYSIMD_FLOAT32_C( 8015.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fmaddsub_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fmaddsub_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256 a;
    easysimd__m256 b;
    easysimd__m256 c;
    easysimd__m256 r;
  } test_vec[8] = {
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -61.10), EASYSIMD_FLOAT32_C(  -95.60),
                         EASYSIMD_FLOAT32_C(   56.00), EASYSIMD_FLOAT32_C(   46.30),
                         EASYSIMD_FLOAT32_C(  -62.80), EASYSIMD_FLOAT32_C(   38.90),
                         EASYSIMD_FLOAT32_C(  -92.60), EASYSIMD_FLOAT32_C(   65.40)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -38.30), EASYSIMD_FLOAT32_C(   -1.90),
                         EASYSIMD_FLOAT32_C(  -28.00), EASYSIMD_FLOAT32_C(  -43.20),
                         EASYSIMD_FLOAT32_C(  -19.40), EASYSIMD_FLOAT32_C(   57.60),
                         EASYSIMD_FLOAT32_C(  -97.20), EASYSIMD_FLOAT32_C(   81.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   93.20), EASYSIMD_FLOAT32_C(  -43.00),
                         EASYSIMD_FLOAT32_C(  -47.40), EASYSIMD_FLOAT32_C(  -77.00),
                         EASYSIMD_FLOAT32_C(  -59.90), EASYSIMD_FLOAT32_C(   17.90),
                         EASYSIMD_FLOAT32_C(   -9.60), EASYSIMD_FLOAT32_C(  -61.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 2433.33), EASYSIMD_FLOAT32_C(  224.64),
                         EASYSIMD_FLOAT32_C(-1615.40), EASYSIMD_FLOAT32_C(-1923.16),
                         EASYSIMD_FLOAT32_C( 1158.42), EASYSIMD_FLOAT32_C( 2222.74),
                         EASYSIMD_FLOAT32_C( 8991.12), EASYSIMD_FLOAT32_C( 5371.78)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -23.40), EASYSIMD_FLOAT32_C(  -24.60),
                         EASYSIMD_FLOAT32_C(   35.70), EASYSIMD_FLOAT32_C(   59.90),
                         EASYSIMD_FLOAT32_C(  -91.00), EASYSIMD_FLOAT32_C(  -25.40),
                         EASYSIMD_FLOAT32_C(  -88.30), EASYSIMD_FLOAT32_C(  -99.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -30.70), EASYSIMD_FLOAT32_C(   97.10),
                         EASYSIMD_FLOAT32_C(   86.90), EASYSIMD_FLOAT32_C(  -81.10),
                         EASYSIMD_FLOAT32_C(  -71.30), EASYSIMD_FLOAT32_C(  -61.20),
                         EASYSIMD_FLOAT32_C(  -26.10), EASYSIMD_FLOAT32_C(   31.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -74.30), EASYSIMD_FLOAT32_C(  -19.40),
                         EASYSIMD_FLOAT32_C(  -70.80), EASYSIMD_FLOAT32_C(  -13.00),
                         EASYSIMD_FLOAT32_C(   82.90), EASYSIMD_FLOAT32_C(  -75.70),
                         EASYSIMD_FLOAT32_C(  -31.50), EASYSIMD_FLOAT32_C(   73.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  644.08), EASYSIMD_FLOAT32_C(-2369.26),
                         EASYSIMD_FLOAT32_C( 3031.53), EASYSIMD_FLOAT32_C(-4844.89),
                         EASYSIMD_FLOAT32_C( 6571.20), EASYSIMD_FLOAT32_C( 1630.18),
                         EASYSIMD_FLOAT32_C( 2273.13), EASYSIMD_FLOAT32_C(-3227.18)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   70.20), EASYSIMD_FLOAT32_C(  -20.40),
                         EASYSIMD_FLOAT32_C(  -51.50), EASYSIMD_FLOAT32_C(   82.30),
                         EASYSIMD_FLOAT32_C(   31.30), EASYSIMD_FLOAT32_C(   17.80),
                         EASYSIMD_FLOAT32_C(  -39.60), EASYSIMD_FLOAT32_C(   66.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -72.50), EASYSIMD_FLOAT32_C(   52.00),
                         EASYSIMD_FLOAT32_C(  -54.80), EASYSIMD_FLOAT32_C(   14.00),
                         EASYSIMD_FLOAT32_C(   91.80), EASYSIMD_FLOAT32_C(  -80.70),
                         EASYSIMD_FLOAT32_C(  -97.90), EASYSIMD_FLOAT32_C(  -99.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -76.10), EASYSIMD_FLOAT32_C(   26.90),
                         EASYSIMD_FLOAT32_C(   24.90), EASYSIMD_FLOAT32_C(  -50.60),
                         EASYSIMD_FLOAT32_C(   66.90), EASYSIMD_FLOAT32_C(   82.40),
                         EASYSIMD_FLOAT32_C(   98.50), EASYSIMD_FLOAT32_C(    9.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-5165.60), EASYSIMD_FLOAT32_C(-1087.70),
                         EASYSIMD_FLOAT32_C( 2847.10), EASYSIMD_FLOAT32_C( 1202.80),
                         EASYSIMD_FLOAT32_C( 2940.24), EASYSIMD_FLOAT32_C(-1518.86),
                         EASYSIMD_FLOAT32_C( 3975.34), EASYSIMD_FLOAT32_C(-6622.80)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -28.40), EASYSIMD_FLOAT32_C(   22.80),
                         EASYSIMD_FLOAT32_C(   16.40), EASYSIMD_FLOAT32_C(   80.20),
                         EASYSIMD_FLOAT32_C(  -24.10), EASYSIMD_FLOAT32_C(  -83.00),
                         EASYSIMD_FLOAT32_C(  -74.10), EASYSIMD_FLOAT32_C(  -49.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -73.70), EASYSIMD_FLOAT32_C(   59.00),
                         EASYSIMD_FLOAT32_C(   36.90), EASYSIMD_FLOAT32_C(    7.50),
                         EASYSIMD_FLOAT32_C(  -74.80), EASYSIMD_FLOAT32_C(  -84.40),
                         EASYSIMD_FLOAT32_C(   79.60), EASYSIMD_FLOAT32_C(  -90.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -76.40), EASYSIMD_FLOAT32_C(   28.10),
                         EASYSIMD_FLOAT32_C(  -13.60), EASYSIMD_FLOAT32_C(  -71.50),
                         EASYSIMD_FLOAT32_C(  -52.20), EASYSIMD_FLOAT32_C(  -30.20),
                         EASYSIMD_FLOAT32_C(  -62.60), EASYSIMD_FLOAT32_C(    2.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 2016.68), EASYSIMD_FLOAT32_C( 1317.10),
                         EASYSIMD_FLOAT32_C(  591.56), EASYSIMD_FLOAT32_C(  673.00),
                         EASYSIMD_FLOAT32_C( 1750.48), EASYSIMD_FLOAT32_C( 7035.40),
                         EASYSIMD_FLOAT32_C(-5960.96), EASYSIMD_FLOAT32_C( 4496.42)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   74.80), EASYSIMD_FLOAT32_C(   79.60),
                         EASYSIMD_FLOAT32_C(  -91.30), EASYSIMD_FLOAT32_C(   86.60),
                         EASYSIMD_FLOAT32_C(   41.70), EASYSIMD_FLOAT32_C(  -74.30),
                         EASYSIMD_FLOAT32_C(  -75.60), EASYSIMD_FLOAT32_C(   28.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   96.00), EASYSIMD_FLOAT32_C(   10.10),
                         EASYSIMD_FLOAT32_C(  -63.40), EASYSIMD_FLOAT32_C(   96.90),
                         EASYSIMD_FLOAT32_C(   66.20), EASYSIMD_FLOAT32_C(  -75.30),
                         EASYSIMD_FLOAT32_C(  -11.80), EASYSIMD_FLOAT32_C(   30.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   31.70), EASYSIMD_FLOAT32_C(  -47.90),
                         EASYSIMD_FLOAT32_C(   27.70), EASYSIMD_FLOAT32_C(   40.70),
                         EASYSIMD_FLOAT32_C(  -22.80), EASYSIMD_FLOAT32_C(   35.80),
                         EASYSIMD_FLOAT32_C(  -30.10), EASYSIMD_FLOAT32_C(   88.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 7212.50), EASYSIMD_FLOAT32_C(  851.86),
                         EASYSIMD_FLOAT32_C( 5816.12), EASYSIMD_FLOAT32_C( 8350.84),
                         EASYSIMD_FLOAT32_C( 2737.74), EASYSIMD_FLOAT32_C( 5558.99),
                         EASYSIMD_FLOAT32_C(  861.98), EASYSIMD_FLOAT32_C(  786.95)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   91.80), EASYSIMD_FLOAT32_C(  -99.10),
                         EASYSIMD_FLOAT32_C(  -91.30), EASYSIMD_FLOAT32_C(   69.40),
                         EASYSIMD_FLOAT32_C(   38.40), EASYSIMD_FLOAT32_C(  -90.40),
                         EASYSIMD_FLOAT32_C(   62.20), EASYSIMD_FLOAT32_C(  -62.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   82.70), EASYSIMD_FLOAT32_C(  -63.90),
                         EASYSIMD_FLOAT32_C(   57.00), EASYSIMD_FLOAT32_C(  -53.70),
                         EASYSIMD_FLOAT32_C(  -62.00), EASYSIMD_FLOAT32_C(   87.90),
                         EASYSIMD_FLOAT32_C(  -60.70), EASYSIMD_FLOAT32_C(  -94.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   65.30), EASYSIMD_FLOAT32_C(   61.10),
                         EASYSIMD_FLOAT32_C(  -35.30), EASYSIMD_FLOAT32_C(  -37.60),
                         EASYSIMD_FLOAT32_C(    3.40), EASYSIMD_FLOAT32_C(   10.20),
                         EASYSIMD_FLOAT32_C(   25.70), EASYSIMD_FLOAT32_C(   31.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 7657.16), EASYSIMD_FLOAT32_C( 6271.39),
                         EASYSIMD_FLOAT32_C(-5239.40), EASYSIMD_FLOAT32_C(-3689.18),
                         EASYSIMD_FLOAT32_C(-2377.40), EASYSIMD_FLOAT32_C(-7956.36),
                         EASYSIMD_FLOAT32_C(-3749.84), EASYSIMD_FLOAT32_C( 5875.15)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -9.70), EASYSIMD_FLOAT32_C(   54.70),
                         EASYSIMD_FLOAT32_C(  -66.40), EASYSIMD_FLOAT32_C(  -34.70),
                         EASYSIMD_FLOAT32_C(  -27.90), EASYSIMD_FLOAT32_C(   92.40),
                         EASYSIMD_FLOAT32_C(  -11.40), EASYSIMD_FLOAT32_C(   14.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -0.90), EASYSIMD_FLOAT32_C(  -71.50),
                         EASYSIMD_FLOAT32_C(   67.00), EASYSIMD_FLOAT32_C(  -56.30),
                         EASYSIMD_FLOAT32_C(   74.40), EASYSIMD_FLOAT32_C(    9.80),
                         EASYSIMD_FLOAT32_C(   -5.30), EASYSIMD_FLOAT32_C(   63.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -47.10), EASYSIMD_FLOAT32_C(   81.20),
                         EASYSIMD_FLOAT32_C(   31.00), EASYSIMD_FLOAT32_C(   11.50),
                         EASYSIMD_FLOAT32_C(   67.80), EASYSIMD_FLOAT32_C(  -14.20),
                         EASYSIMD_FLOAT32_C(  -62.80), EASYSIMD_FLOAT32_C(   84.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -38.37), EASYSIMD_FLOAT32_C(-3992.25),
                         EASYSIMD_FLOAT32_C(-4417.80), EASYSIMD_FLOAT32_C( 1942.11),
                         EASYSIMD_FLOAT32_C(-2007.96), EASYSIMD_FLOAT32_C(  919.72),
                         EASYSIMD_FLOAT32_C(   -2.38), EASYSIMD_FLOAT32_C(  846.78)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -14.10), EASYSIMD_FLOAT32_C(  -90.60),
                         EASYSIMD_FLOAT32_C(   37.70), EASYSIMD_FLOAT32_C(   63.50),
                         EASYSIMD_FLOAT32_C(  -67.90), EASYSIMD_FLOAT32_C(  -75.70),
                         EASYSIMD_FLOAT32_C(   48.30), EASYSIMD_FLOAT32_C(   69.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   21.20), EASYSIMD_FLOAT32_C(  -56.80),
                         EASYSIMD_FLOAT32_C(  -51.20), EASYSIMD_FLOAT32_C(  -55.60),
                         EASYSIMD_FLOAT32_C(   65.10), EASYSIMD_FLOAT32_C(   21.30),
                         EASYSIMD_FLOAT32_C(  -29.20), EASYSIMD_FLOAT32_C(  -61.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   55.80), EASYSIMD_FLOAT32_C(  -16.50),
                         EASYSIMD_FLOAT32_C(   90.30), EASYSIMD_FLOAT32_C(   10.50),
                         EASYSIMD_FLOAT32_C(  -35.10), EASYSIMD_FLOAT32_C(    8.40),
                         EASYSIMD_FLOAT32_C(  -35.70), EASYSIMD_FLOAT32_C(   70.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -243.12), EASYSIMD_FLOAT32_C( 5162.58),
                         EASYSIMD_FLOAT32_C(-1839.94), EASYSIMD_FLOAT32_C(-3541.10),
                         EASYSIMD_FLOAT32_C(-4455.39), EASYSIMD_FLOAT32_C(-1620.81),
                         EASYSIMD_FLOAT32_C(-1446.06), EASYSIMD_FLOAT32_C(-4370.48)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 r = easysimd_mm256_fmaddsub_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmsub_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -86.00), EASYSIMD_FLOAT64_C(  -88.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   79.20), EASYSIMD_FLOAT64_C(  -72.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   69.30), EASYSIMD_FLOAT64_C(  -94.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-6880.50), EASYSIMD_FLOAT64_C( 6543.29)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   29.00), EASYSIMD_FLOAT64_C(  -23.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   88.90), EASYSIMD_FLOAT64_C(   98.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -27.80), EASYSIMD_FLOAT64_C(  -64.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 2605.90), EASYSIMD_FLOAT64_C(-2196.40)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -56.40), EASYSIMD_FLOAT64_C(   49.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   79.10), EASYSIMD_FLOAT64_C(  -51.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -66.70), EASYSIMD_FLOAT64_C(   16.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-4394.54), EASYSIMD_FLOAT64_C(-2596.43)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -73.50), EASYSIMD_FLOAT64_C(   25.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   95.60), EASYSIMD_FLOAT64_C(   38.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   96.40), EASYSIMD_FLOAT64_C(   40.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-7123.00), EASYSIMD_FLOAT64_C(  939.01)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   96.70), EASYSIMD_FLOAT64_C(  -25.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -31.20), EASYSIMD_FLOAT64_C(  -59.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -18.60), EASYSIMD_FLOAT64_C(  -15.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-2998.44), EASYSIMD_FLOAT64_C( 1567.31)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -63.20), EASYSIMD_FLOAT64_C(  -69.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -27.30), EASYSIMD_FLOAT64_C(   57.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   17.60), EASYSIMD_FLOAT64_C(   32.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 1707.76), EASYSIMD_FLOAT64_C(-4066.03)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -33.50), EASYSIMD_FLOAT64_C(   64.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -68.40), EASYSIMD_FLOAT64_C(  -49.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   44.70), EASYSIMD_FLOAT64_C(   88.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 2246.70), EASYSIMD_FLOAT64_C(-3271.94)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -79.60), EASYSIMD_FLOAT64_C(  -61.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -29.20), EASYSIMD_FLOAT64_C(  -21.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -94.70), EASYSIMD_FLOAT64_C(  -26.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 2419.02), EASYSIMD_FLOAT64_C( 1325.96)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fmsub_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fmsub_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256d a;
    easysimd__m256d b;
    easysimd__m256d c;
    easysimd__m256d r;
  } test_vec[8] = {
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   34.80), EASYSIMD_FLOAT64_C(   57.60),
                         EASYSIMD_FLOAT64_C(   21.20), EASYSIMD_FLOAT64_C(   58.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -15.50), EASYSIMD_FLOAT64_C(  -85.90),
                         EASYSIMD_FLOAT64_C(   76.40), EASYSIMD_FLOAT64_C(   37.40)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -67.00), EASYSIMD_FLOAT64_C(  -15.40),
                         EASYSIMD_FLOAT64_C(   94.00), EASYSIMD_FLOAT64_C(  -95.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -472.40), EASYSIMD_FLOAT64_C(-4932.44),
                         EASYSIMD_FLOAT64_C( 1525.68), EASYSIMD_FLOAT64_C( 2290.88)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   79.40), EASYSIMD_FLOAT64_C(  -18.40),
                         EASYSIMD_FLOAT64_C(  -87.30), EASYSIMD_FLOAT64_C(  -43.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   46.70), EASYSIMD_FLOAT64_C(  -61.00),
                         EASYSIMD_FLOAT64_C(   22.50), EASYSIMD_FLOAT64_C(  -19.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -92.50), EASYSIMD_FLOAT64_C(   24.60),
                         EASYSIMD_FLOAT64_C(   48.50), EASYSIMD_FLOAT64_C(   81.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 3800.48), EASYSIMD_FLOAT64_C( 1097.80),
                         EASYSIMD_FLOAT64_C(-2012.75), EASYSIMD_FLOAT64_C(  762.31)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -53.30), EASYSIMD_FLOAT64_C(   37.50),
                         EASYSIMD_FLOAT64_C(  -12.20), EASYSIMD_FLOAT64_C(   77.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   44.30), EASYSIMD_FLOAT64_C(   68.70),
                         EASYSIMD_FLOAT64_C(   45.00), EASYSIMD_FLOAT64_C(  -94.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   10.90), EASYSIMD_FLOAT64_C(  -78.60),
                         EASYSIMD_FLOAT64_C(   59.40), EASYSIMD_FLOAT64_C(   54.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-2372.09), EASYSIMD_FLOAT64_C( 2654.85),
                         EASYSIMD_FLOAT64_C( -608.40), EASYSIMD_FLOAT64_C(-7380.98)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -45.20), EASYSIMD_FLOAT64_C(  -98.30),
                         EASYSIMD_FLOAT64_C(    6.30), EASYSIMD_FLOAT64_C(  -64.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   47.00), EASYSIMD_FLOAT64_C(  -17.30),
                         EASYSIMD_FLOAT64_C(   90.50), EASYSIMD_FLOAT64_C(   33.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -47.40), EASYSIMD_FLOAT64_C(  -48.00),
                         EASYSIMD_FLOAT64_C(   92.50), EASYSIMD_FLOAT64_C(  -62.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-2077.00), EASYSIMD_FLOAT64_C( 1748.59),
                         EASYSIMD_FLOAT64_C(  477.65), EASYSIMD_FLOAT64_C(-2069.14)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -66.50), EASYSIMD_FLOAT64_C(   50.50),
                         EASYSIMD_FLOAT64_C(  -60.50), EASYSIMD_FLOAT64_C(   97.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -77.70), EASYSIMD_FLOAT64_C(  -31.10),
                         EASYSIMD_FLOAT64_C(   56.50), EASYSIMD_FLOAT64_C(  -49.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -78.10), EASYSIMD_FLOAT64_C(  -33.20),
                         EASYSIMD_FLOAT64_C(   60.50), EASYSIMD_FLOAT64_C(   91.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 5245.15), EASYSIMD_FLOAT64_C(-1537.35),
                         EASYSIMD_FLOAT64_C(-3478.75), EASYSIMD_FLOAT64_C(-4956.45)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   97.80), EASYSIMD_FLOAT64_C(    3.10),
                         EASYSIMD_FLOAT64_C(   -8.70), EASYSIMD_FLOAT64_C(   56.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   88.60), EASYSIMD_FLOAT64_C(  -73.80),
                         EASYSIMD_FLOAT64_C(   92.30), EASYSIMD_FLOAT64_C(   21.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -81.80), EASYSIMD_FLOAT64_C(  -53.80),
                         EASYSIMD_FLOAT64_C(  -76.80), EASYSIMD_FLOAT64_C(  -90.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 8746.88), EASYSIMD_FLOAT64_C( -174.98),
                         EASYSIMD_FLOAT64_C( -726.21), EASYSIMD_FLOAT64_C( 1313.55)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -20.60), EASYSIMD_FLOAT64_C(  -46.30),
                         EASYSIMD_FLOAT64_C(   51.00), EASYSIMD_FLOAT64_C(   60.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -26.30), EASYSIMD_FLOAT64_C(  -65.50),
                         EASYSIMD_FLOAT64_C(  -31.40), EASYSIMD_FLOAT64_C(   -0.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -79.80), EASYSIMD_FLOAT64_C(   98.80),
                         EASYSIMD_FLOAT64_C(   31.60), EASYSIMD_FLOAT64_C(  -29.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  621.58), EASYSIMD_FLOAT64_C( 2933.85),
                         EASYSIMD_FLOAT64_C(-1633.00), EASYSIMD_FLOAT64_C(   16.90)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   73.70), EASYSIMD_FLOAT64_C(  -28.30),
                         EASYSIMD_FLOAT64_C(   -1.90), EASYSIMD_FLOAT64_C(  -61.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -19.60), EASYSIMD_FLOAT64_C(  -92.40),
                         EASYSIMD_FLOAT64_C(  -22.30), EASYSIMD_FLOAT64_C(  -53.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -75.90), EASYSIMD_FLOAT64_C(   72.50),
                         EASYSIMD_FLOAT64_C(  -50.10), EASYSIMD_FLOAT64_C(   18.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-1368.62), EASYSIMD_FLOAT64_C( 2542.42),
                         EASYSIMD_FLOAT64_C(   92.47), EASYSIMD_FLOAT64_C( 3296.85)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d r = easysimd_mm256_fmsub_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmsub_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -36.50), EASYSIMD_FLOAT32_C(   13.70), EASYSIMD_FLOAT32_C(   -3.10), EASYSIMD_FLOAT32_C(   21.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   80.60), EASYSIMD_FLOAT32_C(   11.30), EASYSIMD_FLOAT32_C(   96.80), EASYSIMD_FLOAT32_C(  -38.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -8.50), EASYSIMD_FLOAT32_C(  -28.20), EASYSIMD_FLOAT32_C(  -26.80), EASYSIMD_FLOAT32_C(  -95.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-2933.40), EASYSIMD_FLOAT32_C(  183.01), EASYSIMD_FLOAT32_C( -273.28), EASYSIMD_FLOAT32_C( -722.48)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   59.20), EASYSIMD_FLOAT32_C(   -6.20), EASYSIMD_FLOAT32_C(  -52.90), EASYSIMD_FLOAT32_C(  -75.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   61.80), EASYSIMD_FLOAT32_C(  -76.10), EASYSIMD_FLOAT32_C(  -87.70), EASYSIMD_FLOAT32_C(  -40.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   34.20), EASYSIMD_FLOAT32_C(   37.10), EASYSIMD_FLOAT32_C(    7.30), EASYSIMD_FLOAT32_C(   67.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 3624.36), EASYSIMD_FLOAT32_C(  434.72), EASYSIMD_FLOAT32_C( 4632.03), EASYSIMD_FLOAT32_C( 2998.05)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -85.40), EASYSIMD_FLOAT32_C(   36.60), EASYSIMD_FLOAT32_C(  -55.80), EASYSIMD_FLOAT32_C(    5.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -37.10), EASYSIMD_FLOAT32_C(   37.80), EASYSIMD_FLOAT32_C(   -6.30), EASYSIMD_FLOAT32_C(   90.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   16.90), EASYSIMD_FLOAT32_C(  -83.90), EASYSIMD_FLOAT32_C(   82.90), EASYSIMD_FLOAT32_C(   23.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 3151.44), EASYSIMD_FLOAT32_C( 1467.38), EASYSIMD_FLOAT32_C(  268.64), EASYSIMD_FLOAT32_C(  510.36)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   59.30), EASYSIMD_FLOAT32_C(   97.10), EASYSIMD_FLOAT32_C(   -5.30), EASYSIMD_FLOAT32_C(  -37.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -7.50), EASYSIMD_FLOAT32_C(   42.80), EASYSIMD_FLOAT32_C(  -32.50), EASYSIMD_FLOAT32_C(  -34.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   25.50), EASYSIMD_FLOAT32_C(   87.80), EASYSIMD_FLOAT32_C(   95.90), EASYSIMD_FLOAT32_C(  -68.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( -470.25), EASYSIMD_FLOAT32_C( 4068.08), EASYSIMD_FLOAT32_C(   76.35), EASYSIMD_FLOAT32_C( 1362.34)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -87.90), EASYSIMD_FLOAT32_C(  -35.50), EASYSIMD_FLOAT32_C(  -15.00), EASYSIMD_FLOAT32_C(   72.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   78.40), EASYSIMD_FLOAT32_C(   83.00), EASYSIMD_FLOAT32_C(   34.70), EASYSIMD_FLOAT32_C(   -8.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   99.60), EASYSIMD_FLOAT32_C(   96.00), EASYSIMD_FLOAT32_C(   45.40), EASYSIMD_FLOAT32_C(  -79.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-6990.96), EASYSIMD_FLOAT32_C(-3042.50), EASYSIMD_FLOAT32_C( -565.90), EASYSIMD_FLOAT32_C( -538.90)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   26.30), EASYSIMD_FLOAT32_C(   69.80), EASYSIMD_FLOAT32_C(  -48.50), EASYSIMD_FLOAT32_C(  -58.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   85.00), EASYSIMD_FLOAT32_C(  -97.40), EASYSIMD_FLOAT32_C(   16.90), EASYSIMD_FLOAT32_C(  -37.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -22.30), EASYSIMD_FLOAT32_C(   21.90), EASYSIMD_FLOAT32_C(  -79.20), EASYSIMD_FLOAT32_C(  -99.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 2257.80), EASYSIMD_FLOAT32_C(-6820.42), EASYSIMD_FLOAT32_C( -740.45), EASYSIMD_FLOAT32_C( 2281.25)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   24.00), EASYSIMD_FLOAT32_C(   51.40), EASYSIMD_FLOAT32_C(  -24.70), EASYSIMD_FLOAT32_C(  -32.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   34.10), EASYSIMD_FLOAT32_C(   90.10), EASYSIMD_FLOAT32_C(   39.10), EASYSIMD_FLOAT32_C(  -33.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   63.90), EASYSIMD_FLOAT32_C(  -54.20), EASYSIMD_FLOAT32_C(  -27.60), EASYSIMD_FLOAT32_C(   31.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  754.50), EASYSIMD_FLOAT32_C( 4685.34), EASYSIMD_FLOAT32_C( -938.17), EASYSIMD_FLOAT32_C( 1044.05)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   45.70), EASYSIMD_FLOAT32_C(  -95.60), EASYSIMD_FLOAT32_C(   14.60), EASYSIMD_FLOAT32_C(   -3.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -90.50), EASYSIMD_FLOAT32_C(  -20.20), EASYSIMD_FLOAT32_C(   91.40), EASYSIMD_FLOAT32_C(   25.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -26.90), EASYSIMD_FLOAT32_C(   29.30), EASYSIMD_FLOAT32_C(   77.50), EASYSIMD_FLOAT32_C(  -80.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-4108.95), EASYSIMD_FLOAT32_C( 1901.82), EASYSIMD_FLOAT32_C( 1256.94), EASYSIMD_FLOAT32_C(   -5.34)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fmsub_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fmsub_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256 a;
    easysimd__m256 b;
    easysimd__m256 c;
    easysimd__m256 r;
  } test_vec[8] = {
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   71.60), EASYSIMD_FLOAT32_C(   70.70),
                         EASYSIMD_FLOAT32_C(   40.60), EASYSIMD_FLOAT32_C(   -9.30),
                         EASYSIMD_FLOAT32_C(  -79.10), EASYSIMD_FLOAT32_C(   52.30),
                         EASYSIMD_FLOAT32_C(  -67.90), EASYSIMD_FLOAT32_C(   25.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -51.50), EASYSIMD_FLOAT32_C(   43.80),
                         EASYSIMD_FLOAT32_C(   41.70), EASYSIMD_FLOAT32_C(  -77.20),
                         EASYSIMD_FLOAT32_C(   -5.00), EASYSIMD_FLOAT32_C(   96.70),
                         EASYSIMD_FLOAT32_C(  -13.50), EASYSIMD_FLOAT32_C(   -2.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -28.70), EASYSIMD_FLOAT32_C(  -28.30),
                         EASYSIMD_FLOAT32_C(    1.80), EASYSIMD_FLOAT32_C(  -81.10),
                         EASYSIMD_FLOAT32_C(  -82.10), EASYSIMD_FLOAT32_C(  -69.80),
                         EASYSIMD_FLOAT32_C(   42.10), EASYSIMD_FLOAT32_C(   74.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-3658.70), EASYSIMD_FLOAT32_C( 3124.96),
                         EASYSIMD_FLOAT32_C( 1691.22), EASYSIMD_FLOAT32_C(  799.06),
                         EASYSIMD_FLOAT32_C(  477.60), EASYSIMD_FLOAT32_C( 5127.21),
                         EASYSIMD_FLOAT32_C(  874.55), EASYSIMD_FLOAT32_C( -144.09)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -49.00), EASYSIMD_FLOAT32_C(  -78.70),
                         EASYSIMD_FLOAT32_C(  -72.10), EASYSIMD_FLOAT32_C(   26.10),
                         EASYSIMD_FLOAT32_C(  -91.90), EASYSIMD_FLOAT32_C(    1.40),
                         EASYSIMD_FLOAT32_C(   89.80), EASYSIMD_FLOAT32_C(   94.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -13.70), EASYSIMD_FLOAT32_C(    0.70),
                         EASYSIMD_FLOAT32_C(   57.80), EASYSIMD_FLOAT32_C(   33.00),
                         EASYSIMD_FLOAT32_C(  -83.50), EASYSIMD_FLOAT32_C(   -8.10),
                         EASYSIMD_FLOAT32_C(   91.30), EASYSIMD_FLOAT32_C(   65.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -67.90), EASYSIMD_FLOAT32_C(  -56.40),
                         EASYSIMD_FLOAT32_C(    5.90), EASYSIMD_FLOAT32_C(    2.40),
                         EASYSIMD_FLOAT32_C(   91.80), EASYSIMD_FLOAT32_C(   50.80),
                         EASYSIMD_FLOAT32_C(   64.70), EASYSIMD_FLOAT32_C(  -56.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  739.20), EASYSIMD_FLOAT32_C(    1.31),
                         EASYSIMD_FLOAT32_C(-4173.28), EASYSIMD_FLOAT32_C(  858.90),
                         EASYSIMD_FLOAT32_C( 7581.85), EASYSIMD_FLOAT32_C(  -62.14),
                         EASYSIMD_FLOAT32_C( 8134.04), EASYSIMD_FLOAT32_C( 6197.94)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   40.00), EASYSIMD_FLOAT32_C(   -5.30),
                         EASYSIMD_FLOAT32_C(   85.00), EASYSIMD_FLOAT32_C(   83.70),
                         EASYSIMD_FLOAT32_C(   96.80), EASYSIMD_FLOAT32_C(  -59.70),
                         EASYSIMD_FLOAT32_C(  -72.50), EASYSIMD_FLOAT32_C(   -8.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   77.50), EASYSIMD_FLOAT32_C(   50.00),
                         EASYSIMD_FLOAT32_C(   72.40), EASYSIMD_FLOAT32_C(   98.40),
                         EASYSIMD_FLOAT32_C(   69.10), EASYSIMD_FLOAT32_C(   35.80),
                         EASYSIMD_FLOAT32_C(  -92.90), EASYSIMD_FLOAT32_C(   63.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   73.80), EASYSIMD_FLOAT32_C(  -94.30),
                         EASYSIMD_FLOAT32_C(  -79.50), EASYSIMD_FLOAT32_C(   64.60),
                         EASYSIMD_FLOAT32_C(   63.40), EASYSIMD_FLOAT32_C(  -65.00),
                         EASYSIMD_FLOAT32_C(   75.20), EASYSIMD_FLOAT32_C(   48.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 3026.20), EASYSIMD_FLOAT32_C( -170.70),
                         EASYSIMD_FLOAT32_C( 6233.50), EASYSIMD_FLOAT32_C( 8171.48),
                         EASYSIMD_FLOAT32_C( 6625.48), EASYSIMD_FLOAT32_C(-2072.26),
                         EASYSIMD_FLOAT32_C( 6660.05), EASYSIMD_FLOAT32_C( -564.67)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -82.20), EASYSIMD_FLOAT32_C(   93.30),
                         EASYSIMD_FLOAT32_C(    9.70), EASYSIMD_FLOAT32_C(   -2.70),
                         EASYSIMD_FLOAT32_C(   86.00), EASYSIMD_FLOAT32_C(  -20.80),
                         EASYSIMD_FLOAT32_C(   67.70), EASYSIMD_FLOAT32_C(  -47.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -83.70), EASYSIMD_FLOAT32_C(   12.00),
                         EASYSIMD_FLOAT32_C(   23.10), EASYSIMD_FLOAT32_C(  -42.00),
                         EASYSIMD_FLOAT32_C(   46.30), EASYSIMD_FLOAT32_C(   48.20),
                         EASYSIMD_FLOAT32_C(   86.90), EASYSIMD_FLOAT32_C(  -91.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -48.60), EASYSIMD_FLOAT32_C(   92.30),
                         EASYSIMD_FLOAT32_C(  -12.70), EASYSIMD_FLOAT32_C(  -48.20),
                         EASYSIMD_FLOAT32_C(   60.90), EASYSIMD_FLOAT32_C(   43.20),
                         EASYSIMD_FLOAT32_C(  -71.30), EASYSIMD_FLOAT32_C(  -56.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 6928.74), EASYSIMD_FLOAT32_C( 1027.30),
                         EASYSIMD_FLOAT32_C(  236.77), EASYSIMD_FLOAT32_C(  161.60),
                         EASYSIMD_FLOAT32_C( 3920.90), EASYSIMD_FLOAT32_C(-1045.76),
                         EASYSIMD_FLOAT32_C( 5954.43), EASYSIMD_FLOAT32_C( 4375.40)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   32.60), EASYSIMD_FLOAT32_C(   90.30),
                         EASYSIMD_FLOAT32_C(  -31.90), EASYSIMD_FLOAT32_C(   33.60),
                         EASYSIMD_FLOAT32_C(   47.40), EASYSIMD_FLOAT32_C(   49.30),
                         EASYSIMD_FLOAT32_C(  -73.00), EASYSIMD_FLOAT32_C(   55.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -72.60), EASYSIMD_FLOAT32_C(   34.70),
                         EASYSIMD_FLOAT32_C(   -8.30), EASYSIMD_FLOAT32_C(  -47.40),
                         EASYSIMD_FLOAT32_C(  -91.00), EASYSIMD_FLOAT32_C(  -99.10),
                         EASYSIMD_FLOAT32_C(  -84.60), EASYSIMD_FLOAT32_C(  -13.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -84.20), EASYSIMD_FLOAT32_C(   35.00),
                         EASYSIMD_FLOAT32_C(  -58.10), EASYSIMD_FLOAT32_C(   81.70),
                         EASYSIMD_FLOAT32_C(    1.20), EASYSIMD_FLOAT32_C(  -33.20),
                         EASYSIMD_FLOAT32_C(   36.00), EASYSIMD_FLOAT32_C(  -80.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-2282.56), EASYSIMD_FLOAT32_C( 3098.41),
                         EASYSIMD_FLOAT32_C(  322.87), EASYSIMD_FLOAT32_C(-1674.34),
                         EASYSIMD_FLOAT32_C(-4314.60), EASYSIMD_FLOAT32_C(-4852.43),
                         EASYSIMD_FLOAT32_C( 6139.80), EASYSIMD_FLOAT32_C( -672.40)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -58.90), EASYSIMD_FLOAT32_C(   53.10),
                         EASYSIMD_FLOAT32_C(  -76.60), EASYSIMD_FLOAT32_C(   83.00),
                         EASYSIMD_FLOAT32_C(   91.20), EASYSIMD_FLOAT32_C(  -33.50),
                         EASYSIMD_FLOAT32_C(  -65.20), EASYSIMD_FLOAT32_C(  -55.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -47.40), EASYSIMD_FLOAT32_C(  -20.10),
                         EASYSIMD_FLOAT32_C(  -89.40), EASYSIMD_FLOAT32_C(   87.90),
                         EASYSIMD_FLOAT32_C(  -65.50), EASYSIMD_FLOAT32_C(  -20.70),
                         EASYSIMD_FLOAT32_C(   88.30), EASYSIMD_FLOAT32_C(   20.40)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   50.40), EASYSIMD_FLOAT32_C(   75.40),
                         EASYSIMD_FLOAT32_C(   79.80), EASYSIMD_FLOAT32_C(    5.10),
                         EASYSIMD_FLOAT32_C(   -6.50), EASYSIMD_FLOAT32_C(  -47.90),
                         EASYSIMD_FLOAT32_C(   48.50), EASYSIMD_FLOAT32_C(  -69.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 2741.46), EASYSIMD_FLOAT32_C(-1142.71),
                         EASYSIMD_FLOAT32_C( 6768.24), EASYSIMD_FLOAT32_C( 7290.60),
                         EASYSIMD_FLOAT32_C(-5967.10), EASYSIMD_FLOAT32_C(  741.35),
                         EASYSIMD_FLOAT32_C(-5805.66), EASYSIMD_FLOAT32_C(-1052.10)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    8.30), EASYSIMD_FLOAT32_C(   22.80),
                         EASYSIMD_FLOAT32_C(  -55.20), EASYSIMD_FLOAT32_C(  -62.40),
                         EASYSIMD_FLOAT32_C(  -29.10), EASYSIMD_FLOAT32_C(   56.20),
                         EASYSIMD_FLOAT32_C(   96.20), EASYSIMD_FLOAT32_C(   45.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    9.40), EASYSIMD_FLOAT32_C(  -58.60),
                         EASYSIMD_FLOAT32_C(  -71.50), EASYSIMD_FLOAT32_C(   52.70),
                         EASYSIMD_FLOAT32_C(  -96.40), EASYSIMD_FLOAT32_C(   75.70),
                         EASYSIMD_FLOAT32_C(   -3.70), EASYSIMD_FLOAT32_C(   35.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -87.20), EASYSIMD_FLOAT32_C(  -73.80),
                         EASYSIMD_FLOAT32_C(  -51.80), EASYSIMD_FLOAT32_C(   49.30),
                         EASYSIMD_FLOAT32_C(    9.90), EASYSIMD_FLOAT32_C(   32.40),
                         EASYSIMD_FLOAT32_C(  -44.20), EASYSIMD_FLOAT32_C(   88.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  165.22), EASYSIMD_FLOAT32_C(-1262.28),
                         EASYSIMD_FLOAT32_C( 3998.60), EASYSIMD_FLOAT32_C(-3337.78),
                         EASYSIMD_FLOAT32_C( 2795.34), EASYSIMD_FLOAT32_C( 4221.94),
                         EASYSIMD_FLOAT32_C( -311.74), EASYSIMD_FLOAT32_C( 1545.54)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -71.10), EASYSIMD_FLOAT32_C(  -36.70),
                         EASYSIMD_FLOAT32_C(    2.00), EASYSIMD_FLOAT32_C(  -19.80),
                         EASYSIMD_FLOAT32_C(  -33.20), EASYSIMD_FLOAT32_C(   94.30),
                         EASYSIMD_FLOAT32_C(    1.20), EASYSIMD_FLOAT32_C(   43.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   91.40), EASYSIMD_FLOAT32_C(   40.00),
                         EASYSIMD_FLOAT32_C(   26.00), EASYSIMD_FLOAT32_C(   80.90),
                         EASYSIMD_FLOAT32_C(  -92.20), EASYSIMD_FLOAT32_C(  -86.10),
                         EASYSIMD_FLOAT32_C(   71.10), EASYSIMD_FLOAT32_C(   10.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   29.80), EASYSIMD_FLOAT32_C(  -33.80),
                         EASYSIMD_FLOAT32_C(  -52.50), EASYSIMD_FLOAT32_C(   52.00),
                         EASYSIMD_FLOAT32_C(  -20.10), EASYSIMD_FLOAT32_C(  -49.80),
                         EASYSIMD_FLOAT32_C(   36.10), EASYSIMD_FLOAT32_C(   37.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-6528.34), EASYSIMD_FLOAT32_C(-1434.20),
                         EASYSIMD_FLOAT32_C(  104.50), EASYSIMD_FLOAT32_C(-1653.82),
                         EASYSIMD_FLOAT32_C( 3081.14), EASYSIMD_FLOAT32_C(-8069.43),
                         EASYSIMD_FLOAT32_C(   49.22), EASYSIMD_FLOAT32_C(  402.35)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 r = easysimd_mm256_fmsub_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmsub_sd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   43.20), EASYSIMD_FLOAT64_C(  -60.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -12.80), EASYSIMD_FLOAT64_C(   56.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -27.10), EASYSIMD_FLOAT64_C(   60.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   43.20), EASYSIMD_FLOAT64_C(-3461.40)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -79.30), EASYSIMD_FLOAT64_C(   88.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   44.30), EASYSIMD_FLOAT64_C(   37.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   77.70), EASYSIMD_FLOAT64_C(   22.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -79.30), EASYSIMD_FLOAT64_C( 3302.86)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -1.70), EASYSIMD_FLOAT64_C(  -49.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   87.10), EASYSIMD_FLOAT64_C(  -41.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -71.70), EASYSIMD_FLOAT64_C(   16.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -1.70), EASYSIMD_FLOAT64_C( 2025.60)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -81.20), EASYSIMD_FLOAT64_C(   22.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -90.90), EASYSIMD_FLOAT64_C(   95.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -81.00), EASYSIMD_FLOAT64_C(  -21.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -81.20), EASYSIMD_FLOAT64_C( 2113.20)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   86.00), EASYSIMD_FLOAT64_C(   69.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -68.40), EASYSIMD_FLOAT64_C(  -83.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   95.80), EASYSIMD_FLOAT64_C(   94.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   86.00), EASYSIMD_FLOAT64_C(-5903.08)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -19.30), EASYSIMD_FLOAT64_C(  -49.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -62.80), EASYSIMD_FLOAT64_C(   42.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   30.00), EASYSIMD_FLOAT64_C(  -69.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -19.30), EASYSIMD_FLOAT64_C(-2001.60)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   23.40), EASYSIMD_FLOAT64_C(  -19.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   85.50), EASYSIMD_FLOAT64_C(   56.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   27.00), EASYSIMD_FLOAT64_C(  -47.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   23.40), EASYSIMD_FLOAT64_C(-1062.35)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -59.40), EASYSIMD_FLOAT64_C(   23.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -67.50), EASYSIMD_FLOAT64_C(   79.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   98.40), EASYSIMD_FLOAT64_C(  -48.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -59.40), EASYSIMD_FLOAT64_C( 1870.40)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fmsub_sd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmsub_ss(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   81.20), EASYSIMD_FLOAT32_C(   26.30), EASYSIMD_FLOAT32_C(   21.90), EASYSIMD_FLOAT32_C(   41.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   37.70), EASYSIMD_FLOAT32_C(   61.40), EASYSIMD_FLOAT32_C(   87.60), EASYSIMD_FLOAT32_C(  -37.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   85.80), EASYSIMD_FLOAT32_C(  -48.50), EASYSIMD_FLOAT32_C(   52.10), EASYSIMD_FLOAT32_C(   67.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   81.20), EASYSIMD_FLOAT32_C(   26.30), EASYSIMD_FLOAT32_C(   21.90), EASYSIMD_FLOAT32_C(-1622.66)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -23.60), EASYSIMD_FLOAT32_C(  -82.80), EASYSIMD_FLOAT32_C(   55.80), EASYSIMD_FLOAT32_C(  -90.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   53.10), EASYSIMD_FLOAT32_C(  -75.20), EASYSIMD_FLOAT32_C(  -26.00), EASYSIMD_FLOAT32_C(   93.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   47.50), EASYSIMD_FLOAT32_C(   39.90), EASYSIMD_FLOAT32_C(  -49.20), EASYSIMD_FLOAT32_C(  -86.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -23.60), EASYSIMD_FLOAT32_C(  -82.80), EASYSIMD_FLOAT32_C(   55.80), EASYSIMD_FLOAT32_C(-8383.24)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    0.90), EASYSIMD_FLOAT32_C(  -99.10), EASYSIMD_FLOAT32_C(   26.00), EASYSIMD_FLOAT32_C(   32.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   29.60), EASYSIMD_FLOAT32_C(  -93.20), EASYSIMD_FLOAT32_C(  -96.10), EASYSIMD_FLOAT32_C(   87.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   76.20), EASYSIMD_FLOAT32_C(  -98.50), EASYSIMD_FLOAT32_C(    4.10), EASYSIMD_FLOAT32_C(  -66.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    0.90), EASYSIMD_FLOAT32_C(  -99.10), EASYSIMD_FLOAT32_C(   26.00), EASYSIMD_FLOAT32_C( 2910.25)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -58.80), EASYSIMD_FLOAT32_C(    0.70), EASYSIMD_FLOAT32_C(  -50.10), EASYSIMD_FLOAT32_C(  -58.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   53.70), EASYSIMD_FLOAT32_C(  -83.00), EASYSIMD_FLOAT32_C(  -66.70), EASYSIMD_FLOAT32_C(   96.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -97.40), EASYSIMD_FLOAT32_C(   97.80), EASYSIMD_FLOAT32_C(   93.40), EASYSIMD_FLOAT32_C(  -82.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -58.80), EASYSIMD_FLOAT32_C(    0.70), EASYSIMD_FLOAT32_C(  -50.10), EASYSIMD_FLOAT32_C(-5578.26)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   26.20), EASYSIMD_FLOAT32_C(    0.50), EASYSIMD_FLOAT32_C(   53.40), EASYSIMD_FLOAT32_C(   40.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -60.30), EASYSIMD_FLOAT32_C(  -94.00), EASYSIMD_FLOAT32_C(   14.10), EASYSIMD_FLOAT32_C(  -94.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   75.80), EASYSIMD_FLOAT32_C(   16.70), EASYSIMD_FLOAT32_C(   -3.80), EASYSIMD_FLOAT32_C(  -98.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   26.20), EASYSIMD_FLOAT32_C(    0.50), EASYSIMD_FLOAT32_C(   53.40), EASYSIMD_FLOAT32_C(-3719.30)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    6.90), EASYSIMD_FLOAT32_C(   37.30), EASYSIMD_FLOAT32_C(   95.60), EASYSIMD_FLOAT32_C(   26.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    6.20), EASYSIMD_FLOAT32_C(   51.70), EASYSIMD_FLOAT32_C(  -27.80), EASYSIMD_FLOAT32_C(   35.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   96.60), EASYSIMD_FLOAT32_C(   16.30), EASYSIMD_FLOAT32_C(  -87.40), EASYSIMD_FLOAT32_C(   51.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    6.90), EASYSIMD_FLOAT32_C(   37.30), EASYSIMD_FLOAT32_C(   95.60), EASYSIMD_FLOAT32_C(  881.72)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -41.80), EASYSIMD_FLOAT32_C(  -50.90), EASYSIMD_FLOAT32_C(   94.30), EASYSIMD_FLOAT32_C(   92.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -51.70), EASYSIMD_FLOAT32_C(   66.70), EASYSIMD_FLOAT32_C(   35.70), EASYSIMD_FLOAT32_C(   84.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -89.60), EASYSIMD_FLOAT32_C(  -35.50), EASYSIMD_FLOAT32_C(  -45.20), EASYSIMD_FLOAT32_C(  -87.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -41.80), EASYSIMD_FLOAT32_C(  -50.90), EASYSIMD_FLOAT32_C(   94.30), EASYSIMD_FLOAT32_C( 7940.85)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   60.00), EASYSIMD_FLOAT32_C(   45.70), EASYSIMD_FLOAT32_C(   16.60), EASYSIMD_FLOAT32_C(   40.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -13.60), EASYSIMD_FLOAT32_C(  -11.50), EASYSIMD_FLOAT32_C(  -61.10), EASYSIMD_FLOAT32_C(  -64.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   96.70), EASYSIMD_FLOAT32_C(  -80.10), EASYSIMD_FLOAT32_C(   37.00), EASYSIMD_FLOAT32_C(   74.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   60.00), EASYSIMD_FLOAT32_C(   45.70), EASYSIMD_FLOAT32_C(   16.60), EASYSIMD_FLOAT32_C(-2687.64)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fmsub_ss(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmsubadd_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -50.60), EASYSIMD_FLOAT64_C(  -67.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -79.80), EASYSIMD_FLOAT64_C(  -83.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   34.70), EASYSIMD_FLOAT64_C(  -10.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 4003.18), EASYSIMD_FLOAT64_C( 5567.60)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -31.80), EASYSIMD_FLOAT64_C(  -73.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -80.50), EASYSIMD_FLOAT64_C(   26.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -10.90), EASYSIMD_FLOAT64_C(  -36.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 2570.80), EASYSIMD_FLOAT64_C(-1982.58)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    7.90), EASYSIMD_FLOAT64_C(  -20.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   91.90), EASYSIMD_FLOAT64_C(  -31.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -17.90), EASYSIMD_FLOAT64_C(  -72.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  743.91), EASYSIMD_FLOAT64_C(  579.30)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    0.90), EASYSIMD_FLOAT64_C(   20.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   75.20), EASYSIMD_FLOAT64_C(  -63.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   33.00), EASYSIMD_FLOAT64_C(   76.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   34.68), EASYSIMD_FLOAT64_C(-1240.34)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   45.60), EASYSIMD_FLOAT64_C(  -62.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -70.50), EASYSIMD_FLOAT64_C(   21.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -20.00), EASYSIMD_FLOAT64_C(   73.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-3194.80), EASYSIMD_FLOAT64_C(-1239.30)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -10.30), EASYSIMD_FLOAT64_C(  -71.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   63.70), EASYSIMD_FLOAT64_C(  -56.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -15.40), EASYSIMD_FLOAT64_C(   29.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -640.71), EASYSIMD_FLOAT64_C( 4083.15)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -1.00), EASYSIMD_FLOAT64_C(   -6.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -84.60), EASYSIMD_FLOAT64_C(  -37.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   53.20), EASYSIMD_FLOAT64_C(  -28.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   31.40), EASYSIMD_FLOAT64_C(  231.63)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -13.10), EASYSIMD_FLOAT64_C(   -9.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -34.00), EASYSIMD_FLOAT64_C(  -63.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   31.60), EASYSIMD_FLOAT64_C(  -13.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  413.80), EASYSIMD_FLOAT64_C(  561.30)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fmsubadd_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fmsubadd_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256d a;
    easysimd__m256d b;
    easysimd__m256d c;
    easysimd__m256d r;
  } test_vec[8] = {
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -49.40), EASYSIMD_FLOAT64_C(  -57.60),
                         EASYSIMD_FLOAT64_C(  -73.20), EASYSIMD_FLOAT64_C(  -70.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   -0.10), EASYSIMD_FLOAT64_C(   46.20),
                         EASYSIMD_FLOAT64_C(  -46.70), EASYSIMD_FLOAT64_C(  -70.80)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   79.00), EASYSIMD_FLOAT64_C(  -79.60),
                         EASYSIMD_FLOAT64_C(   19.80), EASYSIMD_FLOAT64_C(  -16.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -74.06), EASYSIMD_FLOAT64_C(-2740.72),
                         EASYSIMD_FLOAT64_C( 3398.64), EASYSIMD_FLOAT64_C( 4947.08)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -12.30), EASYSIMD_FLOAT64_C(   53.50),
                         EASYSIMD_FLOAT64_C(  -97.80), EASYSIMD_FLOAT64_C(  -85.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   67.10), EASYSIMD_FLOAT64_C(  -30.10),
                         EASYSIMD_FLOAT64_C(   -0.30), EASYSIMD_FLOAT64_C(  -23.80)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   49.80), EASYSIMD_FLOAT64_C(   87.30),
                         EASYSIMD_FLOAT64_C(  -23.10), EASYSIMD_FLOAT64_C(   15.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -875.13), EASYSIMD_FLOAT64_C(-1523.05),
                         EASYSIMD_FLOAT64_C(   52.44), EASYSIMD_FLOAT64_C( 2043.66)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -27.20), EASYSIMD_FLOAT64_C(  -72.40),
                         EASYSIMD_FLOAT64_C(   53.20), EASYSIMD_FLOAT64_C(   -9.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -71.40), EASYSIMD_FLOAT64_C(    0.20),
                         EASYSIMD_FLOAT64_C(  -61.10), EASYSIMD_FLOAT64_C(  -97.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   85.60), EASYSIMD_FLOAT64_C(   27.60),
                         EASYSIMD_FLOAT64_C(   19.30), EASYSIMD_FLOAT64_C(   46.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 1856.48), EASYSIMD_FLOAT64_C(   13.12),
                         EASYSIMD_FLOAT64_C(-3269.82), EASYSIMD_FLOAT64_C(  970.95)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   42.40), EASYSIMD_FLOAT64_C(  -47.00),
                         EASYSIMD_FLOAT64_C(   57.40), EASYSIMD_FLOAT64_C(  -79.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -85.60), EASYSIMD_FLOAT64_C(  -55.10),
                         EASYSIMD_FLOAT64_C(    8.90), EASYSIMD_FLOAT64_C(   -9.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   78.80), EASYSIMD_FLOAT64_C(   18.80),
                         EASYSIMD_FLOAT64_C(  -90.80), EASYSIMD_FLOAT64_C(   46.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-3708.24), EASYSIMD_FLOAT64_C( 2608.50),
                         EASYSIMD_FLOAT64_C(  601.66), EASYSIMD_FLOAT64_C(  817.15)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   34.00), EASYSIMD_FLOAT64_C(   57.40),
                         EASYSIMD_FLOAT64_C(   76.30), EASYSIMD_FLOAT64_C(   99.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   74.30), EASYSIMD_FLOAT64_C(   64.30),
                         EASYSIMD_FLOAT64_C(  -88.20), EASYSIMD_FLOAT64_C(  -42.40)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -20.50), EASYSIMD_FLOAT64_C(   98.80),
                         EASYSIMD_FLOAT64_C(  -81.30), EASYSIMD_FLOAT64_C(    9.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 2546.70), EASYSIMD_FLOAT64_C( 3789.62),
                         EASYSIMD_FLOAT64_C(-6648.36), EASYSIMD_FLOAT64_C(-4196.38)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   26.30), EASYSIMD_FLOAT64_C(  -10.40),
                         EASYSIMD_FLOAT64_C(  -16.90), EASYSIMD_FLOAT64_C(  -91.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   -5.60), EASYSIMD_FLOAT64_C(  -40.40),
                         EASYSIMD_FLOAT64_C(   57.90), EASYSIMD_FLOAT64_C(   93.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -66.10), EASYSIMD_FLOAT64_C(  -60.00),
                         EASYSIMD_FLOAT64_C(  -42.50), EASYSIMD_FLOAT64_C(  -45.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -81.18), EASYSIMD_FLOAT64_C(  360.16),
                         EASYSIMD_FLOAT64_C( -936.01), EASYSIMD_FLOAT64_C(-8628.72)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -77.70), EASYSIMD_FLOAT64_C(   79.90),
                         EASYSIMD_FLOAT64_C(   16.20), EASYSIMD_FLOAT64_C(  -77.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   18.40), EASYSIMD_FLOAT64_C(   71.60),
                         EASYSIMD_FLOAT64_C(  -95.70), EASYSIMD_FLOAT64_C(  -21.80)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -83.90), EASYSIMD_FLOAT64_C(   14.30),
                         EASYSIMD_FLOAT64_C(  -44.90), EASYSIMD_FLOAT64_C(   72.40)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-1345.78), EASYSIMD_FLOAT64_C( 5735.14),
                         EASYSIMD_FLOAT64_C(-1505.44), EASYSIMD_FLOAT64_C( 1757.54)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   68.20), EASYSIMD_FLOAT64_C(   18.60),
                         EASYSIMD_FLOAT64_C(   38.50), EASYSIMD_FLOAT64_C(   98.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -61.10), EASYSIMD_FLOAT64_C(  -31.60),
                         EASYSIMD_FLOAT64_C(   70.50), EASYSIMD_FLOAT64_C(   85.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   94.30), EASYSIMD_FLOAT64_C(   91.40),
                         EASYSIMD_FLOAT64_C(  -28.70), EASYSIMD_FLOAT64_C(   64.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-4261.32), EASYSIMD_FLOAT64_C( -496.36),
                         EASYSIMD_FLOAT64_C( 2742.95), EASYSIMD_FLOAT64_C( 8456.80)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d r = easysimd_mm256_fmsubadd_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fmsubadd_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -29.90), EASYSIMD_FLOAT32_C(  -50.10), EASYSIMD_FLOAT32_C(   13.10), EASYSIMD_FLOAT32_C(   52.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -63.90), EASYSIMD_FLOAT32_C(  -96.40), EASYSIMD_FLOAT32_C(   84.20), EASYSIMD_FLOAT32_C(  -48.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   53.80), EASYSIMD_FLOAT32_C(   -3.40), EASYSIMD_FLOAT32_C(   13.90), EASYSIMD_FLOAT32_C(  -46.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 1856.81), EASYSIMD_FLOAT32_C( 4826.24), EASYSIMD_FLOAT32_C( 1089.12), EASYSIMD_FLOAT32_C(-2566.86)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   35.60), EASYSIMD_FLOAT32_C(    8.10), EASYSIMD_FLOAT32_C(  -35.10), EASYSIMD_FLOAT32_C(   22.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   77.40), EASYSIMD_FLOAT32_C(  -43.50), EASYSIMD_FLOAT32_C(  -53.00), EASYSIMD_FLOAT32_C(   60.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   15.60), EASYSIMD_FLOAT32_C(   -4.70), EASYSIMD_FLOAT32_C(   24.20), EASYSIMD_FLOAT32_C(  -46.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 2739.84), EASYSIMD_FLOAT32_C( -357.05), EASYSIMD_FLOAT32_C( 1836.10), EASYSIMD_FLOAT32_C( 1309.14)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -92.90), EASYSIMD_FLOAT32_C(   31.90), EASYSIMD_FLOAT32_C(  -29.90), EASYSIMD_FLOAT32_C(  -95.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   46.90), EASYSIMD_FLOAT32_C(  -89.80), EASYSIMD_FLOAT32_C(   18.10), EASYSIMD_FLOAT32_C(  -72.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   79.60), EASYSIMD_FLOAT32_C(  -32.40), EASYSIMD_FLOAT32_C(   -3.60), EASYSIMD_FLOAT32_C(  -57.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-4436.61), EASYSIMD_FLOAT32_C(-2897.02), EASYSIMD_FLOAT32_C( -537.59), EASYSIMD_FLOAT32_C( 6871.21)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   68.50), EASYSIMD_FLOAT32_C(   48.90), EASYSIMD_FLOAT32_C(   86.30), EASYSIMD_FLOAT32_C(   72.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -80.00), EASYSIMD_FLOAT32_C(  -44.60), EASYSIMD_FLOAT32_C(   -3.60), EASYSIMD_FLOAT32_C(  -91.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -57.30), EASYSIMD_FLOAT32_C(    2.10), EASYSIMD_FLOAT32_C(  -33.70), EASYSIMD_FLOAT32_C(  -13.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-5422.70), EASYSIMD_FLOAT32_C(-2178.84), EASYSIMD_FLOAT32_C( -276.98), EASYSIMD_FLOAT32_C(-6581.91)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -38.10), EASYSIMD_FLOAT32_C(  -61.30), EASYSIMD_FLOAT32_C(   38.90), EASYSIMD_FLOAT32_C(  -79.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   64.20), EASYSIMD_FLOAT32_C(   71.60), EASYSIMD_FLOAT32_C(  -99.30), EASYSIMD_FLOAT32_C(  -87.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -46.40), EASYSIMD_FLOAT32_C(   45.20), EASYSIMD_FLOAT32_C(  -56.00), EASYSIMD_FLOAT32_C(    0.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-2399.62), EASYSIMD_FLOAT32_C(-4343.88), EASYSIMD_FLOAT32_C(-3806.77), EASYSIMD_FLOAT32_C( 6915.36)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -88.30), EASYSIMD_FLOAT32_C(  -23.50), EASYSIMD_FLOAT32_C(   48.80), EASYSIMD_FLOAT32_C(  -55.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -31.80), EASYSIMD_FLOAT32_C(   50.50), EASYSIMD_FLOAT32_C(  -24.10), EASYSIMD_FLOAT32_C(  -80.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -18.70), EASYSIMD_FLOAT32_C(  -24.70), EASYSIMD_FLOAT32_C(  -56.50), EASYSIMD_FLOAT32_C(   57.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 2826.64), EASYSIMD_FLOAT32_C(-1211.45), EASYSIMD_FLOAT32_C(-1119.58), EASYSIMD_FLOAT32_C( 4476.72)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   48.60), EASYSIMD_FLOAT32_C(   33.60), EASYSIMD_FLOAT32_C(    8.60), EASYSIMD_FLOAT32_C(   57.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   52.40), EASYSIMD_FLOAT32_C(    2.70), EASYSIMD_FLOAT32_C(   57.50), EASYSIMD_FLOAT32_C(  -10.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -26.60), EASYSIMD_FLOAT32_C(  -67.20), EASYSIMD_FLOAT32_C(    5.80), EASYSIMD_FLOAT32_C(   75.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 2573.24), EASYSIMD_FLOAT32_C(   23.52), EASYSIMD_FLOAT32_C(  488.70), EASYSIMD_FLOAT32_C( -527.16)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -25.60), EASYSIMD_FLOAT32_C(   57.60), EASYSIMD_FLOAT32_C(  -91.00), EASYSIMD_FLOAT32_C(   53.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -92.00), EASYSIMD_FLOAT32_C(   35.10), EASYSIMD_FLOAT32_C(    8.60), EASYSIMD_FLOAT32_C(    0.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -41.20), EASYSIMD_FLOAT32_C(  -81.00), EASYSIMD_FLOAT32_C(  -21.80), EASYSIMD_FLOAT32_C(  -49.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 2396.40), EASYSIMD_FLOAT32_C( 1940.76), EASYSIMD_FLOAT32_C( -760.80), EASYSIMD_FLOAT32_C(  -49.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fmsubadd_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fmsubadd_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256 a;
    easysimd__m256 b;
    easysimd__m256 c;
    easysimd__m256 r;
  } test_vec[8] = {
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   80.60), EASYSIMD_FLOAT32_C(  -80.20),
                         EASYSIMD_FLOAT32_C(   25.10), EASYSIMD_FLOAT32_C(   54.40),
                         EASYSIMD_FLOAT32_C(  -94.50), EASYSIMD_FLOAT32_C(  -99.70),
                         EASYSIMD_FLOAT32_C(   67.30), EASYSIMD_FLOAT32_C(   -5.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -98.10), EASYSIMD_FLOAT32_C(  -47.30),
                         EASYSIMD_FLOAT32_C(  -82.80), EASYSIMD_FLOAT32_C(  -26.80),
                         EASYSIMD_FLOAT32_C(   87.80), EASYSIMD_FLOAT32_C(   71.10),
                         EASYSIMD_FLOAT32_C(   92.80), EASYSIMD_FLOAT32_C(  -97.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -36.00), EASYSIMD_FLOAT32_C(  -59.40),
                         EASYSIMD_FLOAT32_C(  -69.40), EASYSIMD_FLOAT32_C(   50.50),
                         EASYSIMD_FLOAT32_C(   70.50), EASYSIMD_FLOAT32_C(   26.60),
                         EASYSIMD_FLOAT32_C(   29.70), EASYSIMD_FLOAT32_C(  -14.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-7870.86), EASYSIMD_FLOAT32_C( 3734.06),
                         EASYSIMD_FLOAT32_C(-2008.88), EASYSIMD_FLOAT32_C(-1407.42),
                         EASYSIMD_FLOAT32_C(-8367.60), EASYSIMD_FLOAT32_C(-7062.07),
                         EASYSIMD_FLOAT32_C( 6215.74), EASYSIMD_FLOAT32_C(  553.02)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -68.50), EASYSIMD_FLOAT32_C(  -56.10),
                         EASYSIMD_FLOAT32_C(   89.00), EASYSIMD_FLOAT32_C(  -96.30),
                         EASYSIMD_FLOAT32_C(   41.10), EASYSIMD_FLOAT32_C(  -67.50),
                         EASYSIMD_FLOAT32_C(   59.30), EASYSIMD_FLOAT32_C(  -62.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -50.40), EASYSIMD_FLOAT32_C(  -79.00),
                         EASYSIMD_FLOAT32_C(   93.10), EASYSIMD_FLOAT32_C(  -46.20),
                         EASYSIMD_FLOAT32_C(  -86.10), EASYSIMD_FLOAT32_C(   19.30),
                         EASYSIMD_FLOAT32_C(  -62.90), EASYSIMD_FLOAT32_C(  -49.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -28.70), EASYSIMD_FLOAT32_C(  -24.80),
                         EASYSIMD_FLOAT32_C(   30.30), EASYSIMD_FLOAT32_C(  -97.00),
                         EASYSIMD_FLOAT32_C(  -57.70), EASYSIMD_FLOAT32_C(  -32.40),
                         EASYSIMD_FLOAT32_C(   -8.20), EASYSIMD_FLOAT32_C(   75.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 3481.10), EASYSIMD_FLOAT32_C( 4407.10),
                         EASYSIMD_FLOAT32_C( 8255.60), EASYSIMD_FLOAT32_C( 4352.06),
                         EASYSIMD_FLOAT32_C(-3481.01), EASYSIMD_FLOAT32_C(-1335.15),
                         EASYSIMD_FLOAT32_C(-3721.77), EASYSIMD_FLOAT32_C( 3176.17)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -40.40), EASYSIMD_FLOAT32_C(   50.40),
                         EASYSIMD_FLOAT32_C(    3.90), EASYSIMD_FLOAT32_C(  -96.60),
                         EASYSIMD_FLOAT32_C(   84.00), EASYSIMD_FLOAT32_C(   63.30),
                         EASYSIMD_FLOAT32_C(   71.70), EASYSIMD_FLOAT32_C(   -5.40)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   88.10), EASYSIMD_FLOAT32_C(   75.30),
                         EASYSIMD_FLOAT32_C(  -17.10), EASYSIMD_FLOAT32_C(  -27.60),
                         EASYSIMD_FLOAT32_C(   47.20), EASYSIMD_FLOAT32_C(  -72.70),
                         EASYSIMD_FLOAT32_C(  -49.20), EASYSIMD_FLOAT32_C(  -33.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   96.20), EASYSIMD_FLOAT32_C(   64.10),
                         EASYSIMD_FLOAT32_C(   96.10), EASYSIMD_FLOAT32_C(  -18.70),
                         EASYSIMD_FLOAT32_C(  -31.60), EASYSIMD_FLOAT32_C(   43.60),
                         EASYSIMD_FLOAT32_C(  -90.90), EASYSIMD_FLOAT32_C(  -27.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-3655.44), EASYSIMD_FLOAT32_C( 3859.22),
                         EASYSIMD_FLOAT32_C( -162.79), EASYSIMD_FLOAT32_C( 2647.46),
                         EASYSIMD_FLOAT32_C( 3996.40), EASYSIMD_FLOAT32_C(-4558.31),
                         EASYSIMD_FLOAT32_C(-3436.74), EASYSIMD_FLOAT32_C(  151.44)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -59.60), EASYSIMD_FLOAT32_C(  -83.90),
                         EASYSIMD_FLOAT32_C(   58.10), EASYSIMD_FLOAT32_C(   -6.90),
                         EASYSIMD_FLOAT32_C(   99.80), EASYSIMD_FLOAT32_C(  -64.30),
                         EASYSIMD_FLOAT32_C(   87.70), EASYSIMD_FLOAT32_C(   55.40)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   53.60), EASYSIMD_FLOAT32_C(   89.30),
                         EASYSIMD_FLOAT32_C(  -70.60), EASYSIMD_FLOAT32_C(   99.70),
                         EASYSIMD_FLOAT32_C(   -5.30), EASYSIMD_FLOAT32_C(    5.60),
                         EASYSIMD_FLOAT32_C(   86.80), EASYSIMD_FLOAT32_C(   -0.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   15.90), EASYSIMD_FLOAT32_C(  -12.20),
                         EASYSIMD_FLOAT32_C(   93.70), EASYSIMD_FLOAT32_C(  -91.90),
                         EASYSIMD_FLOAT32_C(   34.20), EASYSIMD_FLOAT32_C(  -64.50),
                         EASYSIMD_FLOAT32_C(   97.10), EASYSIMD_FLOAT32_C(   -8.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-3210.46), EASYSIMD_FLOAT32_C(-7504.47),
                         EASYSIMD_FLOAT32_C(-4195.56), EASYSIMD_FLOAT32_C( -779.83),
                         EASYSIMD_FLOAT32_C( -563.14), EASYSIMD_FLOAT32_C( -424.58),
                         EASYSIMD_FLOAT32_C( 7515.26), EASYSIMD_FLOAT32_C(  -19.08)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   49.20), EASYSIMD_FLOAT32_C(  -59.10),
                         EASYSIMD_FLOAT32_C(  -10.90), EASYSIMD_FLOAT32_C(  -67.30),
                         EASYSIMD_FLOAT32_C(   52.90), EASYSIMD_FLOAT32_C(   -9.10),
                         EASYSIMD_FLOAT32_C(  -30.60), EASYSIMD_FLOAT32_C(  -79.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -82.90), EASYSIMD_FLOAT32_C(   24.10),
                         EASYSIMD_FLOAT32_C(    5.20), EASYSIMD_FLOAT32_C(   -4.60),
                         EASYSIMD_FLOAT32_C(  -64.40), EASYSIMD_FLOAT32_C(   -6.30),
                         EASYSIMD_FLOAT32_C(   88.20), EASYSIMD_FLOAT32_C(   59.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   59.30), EASYSIMD_FLOAT32_C(  -23.80),
                         EASYSIMD_FLOAT32_C(   86.10), EASYSIMD_FLOAT32_C(   45.80),
                         EASYSIMD_FLOAT32_C(  -77.20), EASYSIMD_FLOAT32_C(    3.40),
                         EASYSIMD_FLOAT32_C(   70.60), EASYSIMD_FLOAT32_C(  -87.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-4137.98), EASYSIMD_FLOAT32_C(-1448.11),
                         EASYSIMD_FLOAT32_C( -142.78), EASYSIMD_FLOAT32_C(  355.38),
                         EASYSIMD_FLOAT32_C(-3329.56), EASYSIMD_FLOAT32_C(   60.73),
                         EASYSIMD_FLOAT32_C(-2769.52), EASYSIMD_FLOAT32_C(-4769.72)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -41.90), EASYSIMD_FLOAT32_C(   60.40),
                         EASYSIMD_FLOAT32_C(  -79.60), EASYSIMD_FLOAT32_C(   95.50),
                         EASYSIMD_FLOAT32_C(   31.30), EASYSIMD_FLOAT32_C(  -95.40),
                         EASYSIMD_FLOAT32_C(   27.30), EASYSIMD_FLOAT32_C(   96.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   46.90), EASYSIMD_FLOAT32_C(  -42.30),
                         EASYSIMD_FLOAT32_C(   95.50), EASYSIMD_FLOAT32_C(  -75.00),
                         EASYSIMD_FLOAT32_C(   48.70), EASYSIMD_FLOAT32_C(   76.90),
                         EASYSIMD_FLOAT32_C(   81.90), EASYSIMD_FLOAT32_C(   70.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -69.40), EASYSIMD_FLOAT32_C(   89.00),
                         EASYSIMD_FLOAT32_C(  -88.50), EASYSIMD_FLOAT32_C(   76.60),
                         EASYSIMD_FLOAT32_C(  -55.90), EASYSIMD_FLOAT32_C(  -98.10),
                         EASYSIMD_FLOAT32_C(  -24.00), EASYSIMD_FLOAT32_C(  -35.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-1895.71), EASYSIMD_FLOAT32_C(-2465.92),
                         EASYSIMD_FLOAT32_C(-7513.30), EASYSIMD_FLOAT32_C(-7085.90),
                         EASYSIMD_FLOAT32_C( 1580.21), EASYSIMD_FLOAT32_C(-7434.36),
                         EASYSIMD_FLOAT32_C( 2259.87), EASYSIMD_FLOAT32_C( 6756.89)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   12.90), EASYSIMD_FLOAT32_C(   65.20),
                         EASYSIMD_FLOAT32_C(   56.70), EASYSIMD_FLOAT32_C(   39.40),
                         EASYSIMD_FLOAT32_C(  -25.60), EASYSIMD_FLOAT32_C(   -1.40),
                         EASYSIMD_FLOAT32_C(   44.70), EASYSIMD_FLOAT32_C(  -72.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -49.00), EASYSIMD_FLOAT32_C(   97.00),
                         EASYSIMD_FLOAT32_C(  -63.50), EASYSIMD_FLOAT32_C(  -40.00),
                         EASYSIMD_FLOAT32_C(   48.40), EASYSIMD_FLOAT32_C(   30.20),
                         EASYSIMD_FLOAT32_C(  -73.80), EASYSIMD_FLOAT32_C(  -79.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    5.30), EASYSIMD_FLOAT32_C(   28.90),
                         EASYSIMD_FLOAT32_C(   61.30), EASYSIMD_FLOAT32_C(   -5.70),
                         EASYSIMD_FLOAT32_C(   39.10), EASYSIMD_FLOAT32_C(  -88.70),
                         EASYSIMD_FLOAT32_C(   17.20), EASYSIMD_FLOAT32_C(    0.40)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -637.40), EASYSIMD_FLOAT32_C( 6353.30),
                         EASYSIMD_FLOAT32_C(-3661.75), EASYSIMD_FLOAT32_C(-1581.70),
                         EASYSIMD_FLOAT32_C(-1278.14), EASYSIMD_FLOAT32_C( -130.98),
                         EASYSIMD_FLOAT32_C(-3316.06), EASYSIMD_FLOAT32_C( 5724.40)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    1.50), EASYSIMD_FLOAT32_C(  -69.70),
                         EASYSIMD_FLOAT32_C(  -80.70), EASYSIMD_FLOAT32_C(    7.80),
                         EASYSIMD_FLOAT32_C(  -92.30), EASYSIMD_FLOAT32_C(   11.90),
                         EASYSIMD_FLOAT32_C(   59.30), EASYSIMD_FLOAT32_C(  -21.40)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -78.70), EASYSIMD_FLOAT32_C(  -69.80),
                         EASYSIMD_FLOAT32_C(   38.10), EASYSIMD_FLOAT32_C(   22.10),
                         EASYSIMD_FLOAT32_C(  -96.20), EASYSIMD_FLOAT32_C(   60.20),
                         EASYSIMD_FLOAT32_C(   49.80), EASYSIMD_FLOAT32_C(  -68.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   52.80), EASYSIMD_FLOAT32_C(    2.20),
                         EASYSIMD_FLOAT32_C(  -17.20), EASYSIMD_FLOAT32_C(   60.50),
                         EASYSIMD_FLOAT32_C(  -86.40), EASYSIMD_FLOAT32_C(  -89.40),
                         EASYSIMD_FLOAT32_C(  -67.80), EASYSIMD_FLOAT32_C(    4.40)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -170.85), EASYSIMD_FLOAT32_C( 4867.26),
                         EASYSIMD_FLOAT32_C(-3057.47), EASYSIMD_FLOAT32_C(  232.88),
                         EASYSIMD_FLOAT32_C( 8965.66), EASYSIMD_FLOAT32_C(  626.98),
                         EASYSIMD_FLOAT32_C( 3020.94), EASYSIMD_FLOAT32_C( 1470.30)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 r = easysimd_mm256_fmsubadd_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fnmadd_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -93.70), EASYSIMD_FLOAT64_C(   14.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    8.90), EASYSIMD_FLOAT64_C(  -15.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -87.90), EASYSIMD_FLOAT64_C(  -34.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  746.03), EASYSIMD_FLOAT64_C(  194.16)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   38.10), EASYSIMD_FLOAT64_C(  -13.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   63.40), EASYSIMD_FLOAT64_C(  -68.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   13.30), EASYSIMD_FLOAT64_C(  -61.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-2402.24), EASYSIMD_FLOAT64_C( -969.76)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   94.40), EASYSIMD_FLOAT64_C(   89.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -60.60), EASYSIMD_FLOAT64_C(  -24.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -87.30), EASYSIMD_FLOAT64_C(   84.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 5633.34), EASYSIMD_FLOAT64_C( 2305.33)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -6.90), EASYSIMD_FLOAT64_C(   88.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    6.60), EASYSIMD_FLOAT64_C(  -57.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   35.50), EASYSIMD_FLOAT64_C(   30.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   81.04), EASYSIMD_FLOAT64_C( 5163.44)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   51.80), EASYSIMD_FLOAT64_C(   95.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -57.60), EASYSIMD_FLOAT64_C(  -59.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -97.40), EASYSIMD_FLOAT64_C(  -60.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 2886.28), EASYSIMD_FLOAT64_C( 5650.60)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   71.70), EASYSIMD_FLOAT64_C(  -99.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   27.40), EASYSIMD_FLOAT64_C(   37.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   62.10), EASYSIMD_FLOAT64_C(   17.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-1902.48), EASYSIMD_FLOAT64_C( 3785.16)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   43.60), EASYSIMD_FLOAT64_C(   78.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -37.30), EASYSIMD_FLOAT64_C(   -4.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -23.90), EASYSIMD_FLOAT64_C(   -9.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 1602.38), EASYSIMD_FLOAT64_C(  369.24)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   26.20), EASYSIMD_FLOAT64_C(  -96.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   57.90), EASYSIMD_FLOAT64_C(   91.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   97.90), EASYSIMD_FLOAT64_C(   18.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-1419.08), EASYSIMD_FLOAT64_C( 8857.20)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fnmadd_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fnmadd_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256d a;
    easysimd__m256d b;
    easysimd__m256d c;
    easysimd__m256d r;
  } test_vec[8] = {
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   17.60), EASYSIMD_FLOAT64_C(  -99.20),
                         EASYSIMD_FLOAT64_C(   64.80), EASYSIMD_FLOAT64_C(  -66.40)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -84.50), EASYSIMD_FLOAT64_C(   62.70),
                         EASYSIMD_FLOAT64_C(   -1.00), EASYSIMD_FLOAT64_C(   62.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    6.20), EASYSIMD_FLOAT64_C(  -52.40),
                         EASYSIMD_FLOAT64_C(  -54.70), EASYSIMD_FLOAT64_C(   93.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 1493.40), EASYSIMD_FLOAT64_C( 6167.44),
                         EASYSIMD_FLOAT64_C(   10.10), EASYSIMD_FLOAT64_C( 4210.10)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -70.30), EASYSIMD_FLOAT64_C(   67.00),
                         EASYSIMD_FLOAT64_C(   26.40), EASYSIMD_FLOAT64_C(   52.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    8.30), EASYSIMD_FLOAT64_C(   -6.70),
                         EASYSIMD_FLOAT64_C(  -38.30), EASYSIMD_FLOAT64_C(  -42.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   77.70), EASYSIMD_FLOAT64_C(   26.30),
                         EASYSIMD_FLOAT64_C(   10.50), EASYSIMD_FLOAT64_C(   36.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  661.19), EASYSIMD_FLOAT64_C(  475.20),
                         EASYSIMD_FLOAT64_C( 1021.62), EASYSIMD_FLOAT64_C( 2231.00)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -57.10), EASYSIMD_FLOAT64_C(   58.80),
                         EASYSIMD_FLOAT64_C(   93.20), EASYSIMD_FLOAT64_C(  -86.80)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   59.10), EASYSIMD_FLOAT64_C(   76.50),
                         EASYSIMD_FLOAT64_C(   45.10), EASYSIMD_FLOAT64_C(   67.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   60.20), EASYSIMD_FLOAT64_C(   65.10),
                         EASYSIMD_FLOAT64_C(  -17.00), EASYSIMD_FLOAT64_C(  -84.40)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 3434.81), EASYSIMD_FLOAT64_C(-4433.10),
                         EASYSIMD_FLOAT64_C(-4220.32), EASYSIMD_FLOAT64_C( 5791.96)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   34.60), EASYSIMD_FLOAT64_C(   -5.80),
                         EASYSIMD_FLOAT64_C(   89.80), EASYSIMD_FLOAT64_C(  -83.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   43.00), EASYSIMD_FLOAT64_C(    3.10),
                         EASYSIMD_FLOAT64_C(  -37.70), EASYSIMD_FLOAT64_C(  -40.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   26.40), EASYSIMD_FLOAT64_C(  -59.60),
                         EASYSIMD_FLOAT64_C(  -71.50), EASYSIMD_FLOAT64_C(   60.40)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-1461.40), EASYSIMD_FLOAT64_C(  -41.62),
                         EASYSIMD_FLOAT64_C( 3313.96), EASYSIMD_FLOAT64_C(-3317.52)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   40.20), EASYSIMD_FLOAT64_C(  -24.50),
                         EASYSIMD_FLOAT64_C(  -31.60), EASYSIMD_FLOAT64_C(    3.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   39.80), EASYSIMD_FLOAT64_C(   32.70),
                         EASYSIMD_FLOAT64_C(   20.30), EASYSIMD_FLOAT64_C(   49.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   69.00), EASYSIMD_FLOAT64_C(    7.80),
                         EASYSIMD_FLOAT64_C(   99.70), EASYSIMD_FLOAT64_C(   49.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-1530.96), EASYSIMD_FLOAT64_C(  808.95),
                         EASYSIMD_FLOAT64_C(  741.18), EASYSIMD_FLOAT64_C( -114.81)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -82.30), EASYSIMD_FLOAT64_C(   -8.50),
                         EASYSIMD_FLOAT64_C(  -80.50), EASYSIMD_FLOAT64_C(    9.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   52.10), EASYSIMD_FLOAT64_C(  -96.40),
                         EASYSIMD_FLOAT64_C(    3.00), EASYSIMD_FLOAT64_C(  -86.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -93.70), EASYSIMD_FLOAT64_C(    8.90),
                         EASYSIMD_FLOAT64_C(   46.10), EASYSIMD_FLOAT64_C(  -50.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 4194.13), EASYSIMD_FLOAT64_C( -810.50),
                         EASYSIMD_FLOAT64_C(  287.60), EASYSIMD_FLOAT64_C(  731.70)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   72.30), EASYSIMD_FLOAT64_C(   96.70),
                         EASYSIMD_FLOAT64_C(  -51.00), EASYSIMD_FLOAT64_C(  -38.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   37.50), EASYSIMD_FLOAT64_C(   93.30),
                         EASYSIMD_FLOAT64_C(   79.70), EASYSIMD_FLOAT64_C(   71.40)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   54.00), EASYSIMD_FLOAT64_C(    6.80),
                         EASYSIMD_FLOAT64_C(  -77.40), EASYSIMD_FLOAT64_C(  -48.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-2657.25), EASYSIMD_FLOAT64_C(-9015.31),
                         EASYSIMD_FLOAT64_C( 3987.30), EASYSIMD_FLOAT64_C( 2665.10)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -52.40), EASYSIMD_FLOAT64_C(  -75.40),
                         EASYSIMD_FLOAT64_C(  -96.00), EASYSIMD_FLOAT64_C(  -23.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -21.90), EASYSIMD_FLOAT64_C(  -53.30),
                         EASYSIMD_FLOAT64_C(  -90.50), EASYSIMD_FLOAT64_C(  -18.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -63.30), EASYSIMD_FLOAT64_C(  -23.10),
                         EASYSIMD_FLOAT64_C(  -88.90), EASYSIMD_FLOAT64_C(   67.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-1210.86), EASYSIMD_FLOAT64_C(-4041.92),
                         EASYSIMD_FLOAT64_C(-8776.90), EASYSIMD_FLOAT64_C( -355.63)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d r = easysimd_mm256_fnmadd_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fnmadd_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    5.20), EASYSIMD_FLOAT32_C(   59.60), EASYSIMD_FLOAT32_C(   87.70), EASYSIMD_FLOAT32_C(   47.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -48.20), EASYSIMD_FLOAT32_C(  -88.00), EASYSIMD_FLOAT32_C(   90.80), EASYSIMD_FLOAT32_C(  -22.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -19.00), EASYSIMD_FLOAT32_C(   40.90), EASYSIMD_FLOAT32_C(   74.00), EASYSIMD_FLOAT32_C(   71.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  231.64), EASYSIMD_FLOAT32_C( 5285.70), EASYSIMD_FLOAT32_C(-7889.16), EASYSIMD_FLOAT32_C( 1150.39)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   23.70), EASYSIMD_FLOAT32_C(   46.10), EASYSIMD_FLOAT32_C(   -5.90), EASYSIMD_FLOAT32_C(   49.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    6.50), EASYSIMD_FLOAT32_C(   83.40), EASYSIMD_FLOAT32_C(  -86.10), EASYSIMD_FLOAT32_C(   15.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -10.40), EASYSIMD_FLOAT32_C(  -37.00), EASYSIMD_FLOAT32_C(  -97.90), EASYSIMD_FLOAT32_C(   43.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( -164.45), EASYSIMD_FLOAT32_C(-3881.74), EASYSIMD_FLOAT32_C( -605.89), EASYSIMD_FLOAT32_C( -725.28)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -41.10), EASYSIMD_FLOAT32_C(   98.60), EASYSIMD_FLOAT32_C(  -66.40), EASYSIMD_FLOAT32_C(   31.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   80.90), EASYSIMD_FLOAT32_C(  -40.10), EASYSIMD_FLOAT32_C(  -24.70), EASYSIMD_FLOAT32_C(    7.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   99.20), EASYSIMD_FLOAT32_C(  -40.90), EASYSIMD_FLOAT32_C(  -69.50), EASYSIMD_FLOAT32_C(    9.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 3424.19), EASYSIMD_FLOAT32_C( 3912.96), EASYSIMD_FLOAT32_C(-1709.58), EASYSIMD_FLOAT32_C( -237.37)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    0.30), EASYSIMD_FLOAT32_C(   18.10), EASYSIMD_FLOAT32_C(  -38.40), EASYSIMD_FLOAT32_C(  -54.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -5.80), EASYSIMD_FLOAT32_C(   84.90), EASYSIMD_FLOAT32_C(  -77.80), EASYSIMD_FLOAT32_C(  -32.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -23.10), EASYSIMD_FLOAT32_C(    3.00), EASYSIMD_FLOAT32_C(    5.40), EASYSIMD_FLOAT32_C(   61.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -21.36), EASYSIMD_FLOAT32_C(-1533.69), EASYSIMD_FLOAT32_C(-2982.12), EASYSIMD_FLOAT32_C(-1714.11)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -95.30), EASYSIMD_FLOAT32_C(  -61.60), EASYSIMD_FLOAT32_C(  -95.50), EASYSIMD_FLOAT32_C(  -55.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -67.20), EASYSIMD_FLOAT32_C(   95.00), EASYSIMD_FLOAT32_C(   94.10), EASYSIMD_FLOAT32_C(   87.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   45.20), EASYSIMD_FLOAT32_C(  -12.10), EASYSIMD_FLOAT32_C(  -17.00), EASYSIMD_FLOAT32_C(  -48.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-6358.96), EASYSIMD_FLOAT32_C( 5839.90), EASYSIMD_FLOAT32_C( 8969.55), EASYSIMD_FLOAT32_C( 4767.04)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -85.20), EASYSIMD_FLOAT32_C(  -17.40), EASYSIMD_FLOAT32_C(    5.50), EASYSIMD_FLOAT32_C(   51.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   84.50), EASYSIMD_FLOAT32_C(    0.60), EASYSIMD_FLOAT32_C(   61.30), EASYSIMD_FLOAT32_C(   -9.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -44.50), EASYSIMD_FLOAT32_C(  -83.00), EASYSIMD_FLOAT32_C(  -17.60), EASYSIMD_FLOAT32_C(  -95.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( 7154.90), EASYSIMD_FLOAT32_C(  -72.56), EASYSIMD_FLOAT32_C( -354.75), EASYSIMD_FLOAT32_C(  393.20)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   40.00), EASYSIMD_FLOAT32_C(  -99.20), EASYSIMD_FLOAT32_C(  -45.30), EASYSIMD_FLOAT32_C(   65.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    3.60), EASYSIMD_FLOAT32_C(  -27.50), EASYSIMD_FLOAT32_C(   92.40), EASYSIMD_FLOAT32_C(  -74.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   91.90), EASYSIMD_FLOAT32_C(   62.60), EASYSIMD_FLOAT32_C(   33.10), EASYSIMD_FLOAT32_C(   17.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -52.10), EASYSIMD_FLOAT32_C(-2665.40), EASYSIMD_FLOAT32_C( 4218.82), EASYSIMD_FLOAT32_C( 4842.10)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   46.00), EASYSIMD_FLOAT32_C(    7.80), EASYSIMD_FLOAT32_C(   62.40), EASYSIMD_FLOAT32_C(   98.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   60.60), EASYSIMD_FLOAT32_C(  -96.70), EASYSIMD_FLOAT32_C(   86.60), EASYSIMD_FLOAT32_C(   94.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   57.50), EASYSIMD_FLOAT32_C(  -34.30), EASYSIMD_FLOAT32_C(  -42.40), EASYSIMD_FLOAT32_C(  -32.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-2730.10), EASYSIMD_FLOAT32_C(  719.96), EASYSIMD_FLOAT32_C(-5446.24), EASYSIMD_FLOAT32_C(-9302.38)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fnmadd_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fnmadd_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256 a;
    easysimd__m256 b;
    easysimd__m256 c;
    easysimd__m256 r;
  } test_vec[8] = {
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -74.50), EASYSIMD_FLOAT32_C(   76.00),
                         EASYSIMD_FLOAT32_C(  -65.60), EASYSIMD_FLOAT32_C(  -57.80),
                         EASYSIMD_FLOAT32_C(   48.90), EASYSIMD_FLOAT32_C(   17.90),
                         EASYSIMD_FLOAT32_C(   92.90), EASYSIMD_FLOAT32_C(   17.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -79.20), EASYSIMD_FLOAT32_C(   67.50),
                         EASYSIMD_FLOAT32_C(  -50.60), EASYSIMD_FLOAT32_C(   96.50),
                         EASYSIMD_FLOAT32_C(  -92.70), EASYSIMD_FLOAT32_C(   12.20),
                         EASYSIMD_FLOAT32_C(  -41.10), EASYSIMD_FLOAT32_C(  -24.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -20.70), EASYSIMD_FLOAT32_C(   76.20),
                         EASYSIMD_FLOAT32_C(  -47.10), EASYSIMD_FLOAT32_C(  -61.40),
                         EASYSIMD_FLOAT32_C(   55.90), EASYSIMD_FLOAT32_C(   79.30),
                         EASYSIMD_FLOAT32_C(  -95.40), EASYSIMD_FLOAT32_C(   98.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-5921.10), EASYSIMD_FLOAT32_C(-5053.80),
                         EASYSIMD_FLOAT32_C(-3366.46), EASYSIMD_FLOAT32_C( 5516.30),
                         EASYSIMD_FLOAT32_C( 4588.93), EASYSIMD_FLOAT32_C( -139.08),
                         EASYSIMD_FLOAT32_C( 3722.79), EASYSIMD_FLOAT32_C(  527.18)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   34.30), EASYSIMD_FLOAT32_C(   90.50),
                         EASYSIMD_FLOAT32_C(  -43.40), EASYSIMD_FLOAT32_C(  -95.00),
                         EASYSIMD_FLOAT32_C(  -62.70), EASYSIMD_FLOAT32_C(  -17.10),
                         EASYSIMD_FLOAT32_C(   30.50), EASYSIMD_FLOAT32_C(    1.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -23.20), EASYSIMD_FLOAT32_C(   28.90),
                         EASYSIMD_FLOAT32_C(   78.70), EASYSIMD_FLOAT32_C(    6.50),
                         EASYSIMD_FLOAT32_C(  -13.60), EASYSIMD_FLOAT32_C(    7.60),
                         EASYSIMD_FLOAT32_C(  -56.70), EASYSIMD_FLOAT32_C(   52.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -24.30), EASYSIMD_FLOAT32_C(   65.20),
                         EASYSIMD_FLOAT32_C(   27.90), EASYSIMD_FLOAT32_C(  -88.40),
                         EASYSIMD_FLOAT32_C(  -43.70), EASYSIMD_FLOAT32_C(   61.70),
                         EASYSIMD_FLOAT32_C(  -22.10), EASYSIMD_FLOAT32_C(  -51.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  771.46), EASYSIMD_FLOAT32_C(-2550.25),
                         EASYSIMD_FLOAT32_C( 3443.48), EASYSIMD_FLOAT32_C(  529.10),
                         EASYSIMD_FLOAT32_C( -896.42), EASYSIMD_FLOAT32_C(  191.66),
                         EASYSIMD_FLOAT32_C( 1707.25), EASYSIMD_FLOAT32_C( -104.10)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   95.30), EASYSIMD_FLOAT32_C(  -81.70),
                         EASYSIMD_FLOAT32_C(   51.00), EASYSIMD_FLOAT32_C(    6.50),
                         EASYSIMD_FLOAT32_C(   46.00), EASYSIMD_FLOAT32_C(   76.10),
                         EASYSIMD_FLOAT32_C(  -72.70), EASYSIMD_FLOAT32_C(   10.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   67.00), EASYSIMD_FLOAT32_C(  -43.80),
                         EASYSIMD_FLOAT32_C(    5.70), EASYSIMD_FLOAT32_C(    9.00),
                         EASYSIMD_FLOAT32_C(   39.70), EASYSIMD_FLOAT32_C(  -47.40),
                         EASYSIMD_FLOAT32_C(  -89.40), EASYSIMD_FLOAT32_C(  -69.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    5.10), EASYSIMD_FLOAT32_C(  -27.10),
                         EASYSIMD_FLOAT32_C(   24.30), EASYSIMD_FLOAT32_C(  -90.10),
                         EASYSIMD_FLOAT32_C(   48.70), EASYSIMD_FLOAT32_C(   91.00),
                         EASYSIMD_FLOAT32_C(   80.80), EASYSIMD_FLOAT32_C(  -24.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-6380.00), EASYSIMD_FLOAT32_C(-3605.56),
                         EASYSIMD_FLOAT32_C( -266.40), EASYSIMD_FLOAT32_C( -148.60),
                         EASYSIMD_FLOAT32_C(-1777.50), EASYSIMD_FLOAT32_C( 3698.14),
                         EASYSIMD_FLOAT32_C(-6418.58), EASYSIMD_FLOAT32_C(  673.31)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    5.80), EASYSIMD_FLOAT32_C(    2.80),
                         EASYSIMD_FLOAT32_C(   37.80), EASYSIMD_FLOAT32_C(  -55.50),
                         EASYSIMD_FLOAT32_C(   60.80), EASYSIMD_FLOAT32_C(  -46.40),
                         EASYSIMD_FLOAT32_C(  -53.70), EASYSIMD_FLOAT32_C(  -55.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   76.00), EASYSIMD_FLOAT32_C(   65.10),
                         EASYSIMD_FLOAT32_C(   67.70), EASYSIMD_FLOAT32_C(  -84.20),
                         EASYSIMD_FLOAT32_C(   63.00), EASYSIMD_FLOAT32_C(  -82.10),
                         EASYSIMD_FLOAT32_C(  -55.20), EASYSIMD_FLOAT32_C(   20.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   60.20), EASYSIMD_FLOAT32_C(  -85.50),
                         EASYSIMD_FLOAT32_C(   58.00), EASYSIMD_FLOAT32_C(   40.40),
                         EASYSIMD_FLOAT32_C(   31.70), EASYSIMD_FLOAT32_C(   -6.20),
                         EASYSIMD_FLOAT32_C(   83.70), EASYSIMD_FLOAT32_C(  -68.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -380.60), EASYSIMD_FLOAT32_C( -267.78),
                         EASYSIMD_FLOAT32_C(-2501.06), EASYSIMD_FLOAT32_C(-4632.70),
                         EASYSIMD_FLOAT32_C(-3798.70), EASYSIMD_FLOAT32_C(-3815.64),
                         EASYSIMD_FLOAT32_C(-2880.54), EASYSIMD_FLOAT32_C( 1051.07)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -30.40), EASYSIMD_FLOAT32_C(   81.70),
                         EASYSIMD_FLOAT32_C(  -68.60), EASYSIMD_FLOAT32_C(   46.50),
                         EASYSIMD_FLOAT32_C(   53.40), EASYSIMD_FLOAT32_C(   -1.10),
                         EASYSIMD_FLOAT32_C(  -70.80), EASYSIMD_FLOAT32_C(   10.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   99.90), EASYSIMD_FLOAT32_C(  -78.30),
                         EASYSIMD_FLOAT32_C(  -52.60), EASYSIMD_FLOAT32_C(   28.60),
                         EASYSIMD_FLOAT32_C(   62.90), EASYSIMD_FLOAT32_C(  -65.50),
                         EASYSIMD_FLOAT32_C(  -51.00), EASYSIMD_FLOAT32_C(   -0.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   25.50), EASYSIMD_FLOAT32_C(    2.70),
                         EASYSIMD_FLOAT32_C(   99.80), EASYSIMD_FLOAT32_C(  -76.10),
                         EASYSIMD_FLOAT32_C(   -4.50), EASYSIMD_FLOAT32_C(    7.40),
                         EASYSIMD_FLOAT32_C(   81.50), EASYSIMD_FLOAT32_C(    1.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 3062.46), EASYSIMD_FLOAT32_C( 6399.81),
                         EASYSIMD_FLOAT32_C(-3508.56), EASYSIMD_FLOAT32_C(-1406.00),
                         EASYSIMD_FLOAT32_C(-3363.36), EASYSIMD_FLOAT32_C(  -64.65),
                         EASYSIMD_FLOAT32_C(-3529.30), EASYSIMD_FLOAT32_C(    3.54)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   78.00), EASYSIMD_FLOAT32_C(   45.70),
                         EASYSIMD_FLOAT32_C(   59.30), EASYSIMD_FLOAT32_C(   35.50),
                         EASYSIMD_FLOAT32_C(   91.10), EASYSIMD_FLOAT32_C(  -96.00),
                         EASYSIMD_FLOAT32_C(  -93.90), EASYSIMD_FLOAT32_C(   -0.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -80.10), EASYSIMD_FLOAT32_C(   37.30),
                         EASYSIMD_FLOAT32_C(   94.60), EASYSIMD_FLOAT32_C(  -45.10),
                         EASYSIMD_FLOAT32_C(  -34.70), EASYSIMD_FLOAT32_C(  -33.50),
                         EASYSIMD_FLOAT32_C(  -17.00), EASYSIMD_FLOAT32_C(  -46.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -72.60), EASYSIMD_FLOAT32_C(   47.00),
                         EASYSIMD_FLOAT32_C(  -93.40), EASYSIMD_FLOAT32_C(  -34.90),
                         EASYSIMD_FLOAT32_C(   77.40), EASYSIMD_FLOAT32_C(  -96.40),
                         EASYSIMD_FLOAT32_C(   74.90), EASYSIMD_FLOAT32_C(   16.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 6175.20), EASYSIMD_FLOAT32_C(-1657.61),
                         EASYSIMD_FLOAT32_C(-5703.18), EASYSIMD_FLOAT32_C( 1566.15),
                         EASYSIMD_FLOAT32_C( 3238.57), EASYSIMD_FLOAT32_C(-3312.40),
                         EASYSIMD_FLOAT32_C(-1521.40), EASYSIMD_FLOAT32_C(   12.25)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -42.00), EASYSIMD_FLOAT32_C(  100.00),
                         EASYSIMD_FLOAT32_C(  -84.50), EASYSIMD_FLOAT32_C(   27.60),
                         EASYSIMD_FLOAT32_C(   27.10), EASYSIMD_FLOAT32_C(  -76.60),
                         EASYSIMD_FLOAT32_C(  -36.20), EASYSIMD_FLOAT32_C(   16.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   98.50), EASYSIMD_FLOAT32_C(  -46.90),
                         EASYSIMD_FLOAT32_C(  -21.70), EASYSIMD_FLOAT32_C(   90.80),
                         EASYSIMD_FLOAT32_C(   42.70), EASYSIMD_FLOAT32_C(   48.80),
                         EASYSIMD_FLOAT32_C(   91.30), EASYSIMD_FLOAT32_C(   90.10)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   74.40), EASYSIMD_FLOAT32_C(  -15.10),
                         EASYSIMD_FLOAT32_C(   42.70), EASYSIMD_FLOAT32_C(  -90.90),
                         EASYSIMD_FLOAT32_C(  -30.80), EASYSIMD_FLOAT32_C(   48.00),
                         EASYSIMD_FLOAT32_C(   12.60), EASYSIMD_FLOAT32_C(   59.70)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 4211.40), EASYSIMD_FLOAT32_C( 4674.90),
                         EASYSIMD_FLOAT32_C(-1790.95), EASYSIMD_FLOAT32_C(-2596.98),
                         EASYSIMD_FLOAT32_C(-1187.97), EASYSIMD_FLOAT32_C( 3786.08),
                         EASYSIMD_FLOAT32_C( 3317.66), EASYSIMD_FLOAT32_C(-1426.95)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -49.00), EASYSIMD_FLOAT32_C(   74.90),
                         EASYSIMD_FLOAT32_C(  -48.00), EASYSIMD_FLOAT32_C(   46.70),
                         EASYSIMD_FLOAT32_C(    4.40), EASYSIMD_FLOAT32_C(   44.70),
                         EASYSIMD_FLOAT32_C(  -68.40), EASYSIMD_FLOAT32_C(   74.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    8.00), EASYSIMD_FLOAT32_C(  -94.30),
                         EASYSIMD_FLOAT32_C(   -6.20), EASYSIMD_FLOAT32_C(  -21.50),
                         EASYSIMD_FLOAT32_C(   61.90), EASYSIMD_FLOAT32_C(   14.50),
                         EASYSIMD_FLOAT32_C(  -69.00), EASYSIMD_FLOAT32_C(  -34.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   11.20), EASYSIMD_FLOAT32_C(   22.30),
                         EASYSIMD_FLOAT32_C(  -35.00), EASYSIMD_FLOAT32_C(   30.60),
                         EASYSIMD_FLOAT32_C(   72.90), EASYSIMD_FLOAT32_C(   97.50),
                         EASYSIMD_FLOAT32_C(    2.70), EASYSIMD_FLOAT32_C(   72.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  403.20), EASYSIMD_FLOAT32_C( 7085.37),
                         EASYSIMD_FLOAT32_C( -332.60), EASYSIMD_FLOAT32_C( 1034.65),
                         EASYSIMD_FLOAT32_C( -199.46), EASYSIMD_FLOAT32_C( -550.65),
                         EASYSIMD_FLOAT32_C(-4716.90), EASYSIMD_FLOAT32_C( 2643.05)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 r = easysimd_mm256_fnmadd_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fnmadd_sd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   47.80), EASYSIMD_FLOAT64_C(  -80.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   55.10), EASYSIMD_FLOAT64_C(   17.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -52.00), EASYSIMD_FLOAT64_C(   -7.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   47.80), EASYSIMD_FLOAT64_C( 1388.51)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -16.50), EASYSIMD_FLOAT64_C(   77.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -98.70), EASYSIMD_FLOAT64_C(  -77.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -99.70), EASYSIMD_FLOAT64_C(   69.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -16.50), EASYSIMD_FLOAT64_C( 6099.00)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   40.70), EASYSIMD_FLOAT64_C(  -56.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -41.00), EASYSIMD_FLOAT64_C(  -43.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -9.90), EASYSIMD_FLOAT64_C(   48.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   40.70), EASYSIMD_FLOAT64_C(-2370.30)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -21.60), EASYSIMD_FLOAT64_C(  -51.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   91.50), EASYSIMD_FLOAT64_C(   24.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   83.10), EASYSIMD_FLOAT64_C(   15.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -21.60), EASYSIMD_FLOAT64_C( 1303.02)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   36.60), EASYSIMD_FLOAT64_C(   90.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   25.20), EASYSIMD_FLOAT64_C(  -17.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   21.60), EASYSIMD_FLOAT64_C(   29.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   36.60), EASYSIMD_FLOAT64_C( 1606.04)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   28.70), EASYSIMD_FLOAT64_C(  -13.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -49.50), EASYSIMD_FLOAT64_C(    1.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   78.40), EASYSIMD_FLOAT64_C(   70.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   28.70), EASYSIMD_FLOAT64_C(   95.08)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -83.30), EASYSIMD_FLOAT64_C(  -83.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   89.10), EASYSIMD_FLOAT64_C(    5.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -0.10), EASYSIMD_FLOAT64_C(   56.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -83.30), EASYSIMD_FLOAT64_C(  529.60)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -8.80), EASYSIMD_FLOAT64_C(   91.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   82.10), EASYSIMD_FLOAT64_C(  -69.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   92.70), EASYSIMD_FLOAT64_C(  -85.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -8.80), EASYSIMD_FLOAT64_C( 6255.45)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fnmadd_sd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fnmadd_ss(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -92.60), EASYSIMD_FLOAT32_C(  -98.70), EASYSIMD_FLOAT32_C(   10.90), EASYSIMD_FLOAT32_C(  -61.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   48.20), EASYSIMD_FLOAT32_C(    4.60), EASYSIMD_FLOAT32_C(  -98.40), EASYSIMD_FLOAT32_C(   56.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -83.90), EASYSIMD_FLOAT32_C(   54.30), EASYSIMD_FLOAT32_C(   54.70), EASYSIMD_FLOAT32_C(   20.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -92.60), EASYSIMD_FLOAT32_C(  -98.70), EASYSIMD_FLOAT32_C(   10.90), EASYSIMD_FLOAT32_C( 3491.60)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -61.10), EASYSIMD_FLOAT32_C(  -33.00), EASYSIMD_FLOAT32_C(  -47.10), EASYSIMD_FLOAT32_C(   31.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   47.10), EASYSIMD_FLOAT32_C(  -73.50), EASYSIMD_FLOAT32_C(  -40.70), EASYSIMD_FLOAT32_C(  -95.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -73.00), EASYSIMD_FLOAT32_C(  -68.20), EASYSIMD_FLOAT32_C(   35.20), EASYSIMD_FLOAT32_C(   48.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -61.10), EASYSIMD_FLOAT32_C(  -33.00), EASYSIMD_FLOAT32_C(  -47.10), EASYSIMD_FLOAT32_C( 3053.48)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -17.60), EASYSIMD_FLOAT32_C(  -75.20), EASYSIMD_FLOAT32_C(  -94.50), EASYSIMD_FLOAT32_C(   95.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   56.20), EASYSIMD_FLOAT32_C(  -24.90), EASYSIMD_FLOAT32_C(    6.00), EASYSIMD_FLOAT32_C(  -33.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -8.10), EASYSIMD_FLOAT32_C(   95.10), EASYSIMD_FLOAT32_C(  -66.20), EASYSIMD_FLOAT32_C(   51.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -17.60), EASYSIMD_FLOAT32_C(  -75.20), EASYSIMD_FLOAT32_C(  -94.50), EASYSIMD_FLOAT32_C( 3222.10)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   15.40), EASYSIMD_FLOAT32_C(  -42.20), EASYSIMD_FLOAT32_C(  -38.90), EASYSIMD_FLOAT32_C(  -40.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -84.90), EASYSIMD_FLOAT32_C(  -51.00), EASYSIMD_FLOAT32_C(  -45.70), EASYSIMD_FLOAT32_C(   14.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -82.20), EASYSIMD_FLOAT32_C(   60.00), EASYSIMD_FLOAT32_C(  -19.40), EASYSIMD_FLOAT32_C(   90.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   15.40), EASYSIMD_FLOAT32_C(  -42.20), EASYSIMD_FLOAT32_C(  -38.90), EASYSIMD_FLOAT32_C(  667.54)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -29.00), EASYSIMD_FLOAT32_C(   45.90), EASYSIMD_FLOAT32_C(  -65.60), EASYSIMD_FLOAT32_C(   -2.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -46.90), EASYSIMD_FLOAT32_C(    6.70), EASYSIMD_FLOAT32_C(  -97.90), EASYSIMD_FLOAT32_C(  -72.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -7.70), EASYSIMD_FLOAT32_C(  -29.90), EASYSIMD_FLOAT32_C(   69.80), EASYSIMD_FLOAT32_C(  -66.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -29.00), EASYSIMD_FLOAT32_C(   45.90), EASYSIMD_FLOAT32_C(  -65.60), EASYSIMD_FLOAT32_C( -226.00)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -58.10), EASYSIMD_FLOAT32_C(  -47.10), EASYSIMD_FLOAT32_C(   68.70), EASYSIMD_FLOAT32_C(   33.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   40.90), EASYSIMD_FLOAT32_C(  -18.60), EASYSIMD_FLOAT32_C(  -92.90), EASYSIMD_FLOAT32_C(   19.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -14.90), EASYSIMD_FLOAT32_C(   50.40), EASYSIMD_FLOAT32_C(  -64.40), EASYSIMD_FLOAT32_C(   -4.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -58.10), EASYSIMD_FLOAT32_C(  -47.10), EASYSIMD_FLOAT32_C(   68.70), EASYSIMD_FLOAT32_C( -667.08)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -90.60), EASYSIMD_FLOAT32_C(  -45.50), EASYSIMD_FLOAT32_C(  -20.60), EASYSIMD_FLOAT32_C(  -95.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -94.80), EASYSIMD_FLOAT32_C(   21.50), EASYSIMD_FLOAT32_C(   77.40), EASYSIMD_FLOAT32_C(  -58.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -8.20), EASYSIMD_FLOAT32_C(   56.80), EASYSIMD_FLOAT32_C(   16.40), EASYSIMD_FLOAT32_C(  -52.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -90.60), EASYSIMD_FLOAT32_C(  -45.50), EASYSIMD_FLOAT32_C(  -20.60), EASYSIMD_FLOAT32_C(-5677.45)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   16.00), EASYSIMD_FLOAT32_C(  -22.20), EASYSIMD_FLOAT32_C(  -70.50), EASYSIMD_FLOAT32_C(  -57.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   86.70), EASYSIMD_FLOAT32_C(   31.60), EASYSIMD_FLOAT32_C(  -15.30), EASYSIMD_FLOAT32_C(  -77.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -72.10), EASYSIMD_FLOAT32_C(   13.20), EASYSIMD_FLOAT32_C(   17.70), EASYSIMD_FLOAT32_C(  -65.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   16.00), EASYSIMD_FLOAT32_C(  -22.20), EASYSIMD_FLOAT32_C(  -70.50), EASYSIMD_FLOAT32_C(-4512.02)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fnmadd_ss(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fnmsub_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -85.20), EASYSIMD_FLOAT64_C(  -77.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   22.30), EASYSIMD_FLOAT64_C(   10.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   40.90), EASYSIMD_FLOAT64_C(   66.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( 1859.06), EASYSIMD_FLOAT64_C(  709.30)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   12.10), EASYSIMD_FLOAT64_C(  -42.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -71.60), EASYSIMD_FLOAT64_C(  -43.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   73.80), EASYSIMD_FLOAT64_C(  -65.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  792.56), EASYSIMD_FLOAT64_C(-1809.03)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   47.90), EASYSIMD_FLOAT64_C(    8.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    4.00), EASYSIMD_FLOAT64_C(  -70.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -58.60), EASYSIMD_FLOAT64_C(  -21.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -133.00), EASYSIMD_FLOAT64_C(  637.26)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   70.80), EASYSIMD_FLOAT64_C(  -62.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   22.60), EASYSIMD_FLOAT64_C(  -27.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   58.00), EASYSIMD_FLOAT64_C(   35.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-1658.08), EASYSIMD_FLOAT64_C(-1784.22)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -59.60), EASYSIMD_FLOAT64_C(  -26.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   10.40), EASYSIMD_FLOAT64_C(   -3.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -36.80), EASYSIMD_FLOAT64_C(  -20.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  656.64), EASYSIMD_FLOAT64_C(  -78.96)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   75.80), EASYSIMD_FLOAT64_C(  -40.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   82.60), EASYSIMD_FLOAT64_C(   14.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -69.00), EASYSIMD_FLOAT64_C(   52.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(-6192.08), EASYSIMD_FLOAT64_C(  551.05)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -54.60), EASYSIMD_FLOAT64_C(    2.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   17.90), EASYSIMD_FLOAT64_C(   72.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   95.70), EASYSIMD_FLOAT64_C(   56.80)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  881.64), EASYSIMD_FLOAT64_C( -202.40)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   21.40), EASYSIMD_FLOAT64_C(   40.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   42.50), EASYSIMD_FLOAT64_C(   29.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   24.40), EASYSIMD_FLOAT64_C(  -57.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C( -933.90), EASYSIMD_FLOAT64_C(-1112.22)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fnmsub_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fnmsub_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256d a;
    easysimd__m256d b;
    easysimd__m256d c;
    easysimd__m256d r;
  } test_vec[8] = {
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -97.30), EASYSIMD_FLOAT64_C(   40.60),
                         EASYSIMD_FLOAT64_C(  -78.70), EASYSIMD_FLOAT64_C(    0.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   43.40), EASYSIMD_FLOAT64_C(  -67.40),
                         EASYSIMD_FLOAT64_C(   62.50), EASYSIMD_FLOAT64_C(   -5.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   51.10), EASYSIMD_FLOAT64_C(   66.40),
                         EASYSIMD_FLOAT64_C(   79.40), EASYSIMD_FLOAT64_C(    4.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 4171.72), EASYSIMD_FLOAT64_C( 2670.04),
                         EASYSIMD_FLOAT64_C( 4839.35), EASYSIMD_FLOAT64_C(   -0.78)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -37.90), EASYSIMD_FLOAT64_C(  -91.10),
                         EASYSIMD_FLOAT64_C(   99.40), EASYSIMD_FLOAT64_C(  -64.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   55.70), EASYSIMD_FLOAT64_C(  -31.70),
                         EASYSIMD_FLOAT64_C(   33.10), EASYSIMD_FLOAT64_C(   94.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   18.80), EASYSIMD_FLOAT64_C(   56.10),
                         EASYSIMD_FLOAT64_C(  -19.80), EASYSIMD_FLOAT64_C(  -98.50)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 2092.23), EASYSIMD_FLOAT64_C(-2943.97),
                         EASYSIMD_FLOAT64_C(-3270.34), EASYSIMD_FLOAT64_C( 6190.28)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   44.20), EASYSIMD_FLOAT64_C(   98.00),
                         EASYSIMD_FLOAT64_C(  -20.60), EASYSIMD_FLOAT64_C(   99.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -36.50), EASYSIMD_FLOAT64_C(   37.70),
                         EASYSIMD_FLOAT64_C(   27.10), EASYSIMD_FLOAT64_C(  -85.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -54.80), EASYSIMD_FLOAT64_C(   46.70),
                         EASYSIMD_FLOAT64_C(  -59.70), EASYSIMD_FLOAT64_C(  -80.00)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 1668.10), EASYSIMD_FLOAT64_C(-3741.30),
                         EASYSIMD_FLOAT64_C(  617.96), EASYSIMD_FLOAT64_C( 8512.00)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -13.40), EASYSIMD_FLOAT64_C(   16.00),
                         EASYSIMD_FLOAT64_C(  -82.10), EASYSIMD_FLOAT64_C(   27.40)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -41.30), EASYSIMD_FLOAT64_C(   84.40),
                         EASYSIMD_FLOAT64_C(  -52.10), EASYSIMD_FLOAT64_C(   16.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    7.30), EASYSIMD_FLOAT64_C(  -49.40),
                         EASYSIMD_FLOAT64_C(  -31.90), EASYSIMD_FLOAT64_C(   69.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( -560.72), EASYSIMD_FLOAT64_C(-1301.00),
                         EASYSIMD_FLOAT64_C(-4245.51), EASYSIMD_FLOAT64_C( -524.14)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -16.80), EASYSIMD_FLOAT64_C(  -78.00),
                         EASYSIMD_FLOAT64_C(  -43.90), EASYSIMD_FLOAT64_C(  -53.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -11.30), EASYSIMD_FLOAT64_C(  -83.60),
                         EASYSIMD_FLOAT64_C(  -78.30), EASYSIMD_FLOAT64_C(   -1.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -94.20), EASYSIMD_FLOAT64_C(   36.20),
                         EASYSIMD_FLOAT64_C(   66.40), EASYSIMD_FLOAT64_C(   12.70)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -95.64), EASYSIMD_FLOAT64_C(-6557.00),
                         EASYSIMD_FLOAT64_C(-3503.77), EASYSIMD_FLOAT64_C(  -71.66)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   94.80), EASYSIMD_FLOAT64_C(   27.60),
                         EASYSIMD_FLOAT64_C(    5.70), EASYSIMD_FLOAT64_C(  -73.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   11.80), EASYSIMD_FLOAT64_C(  -83.40),
                         EASYSIMD_FLOAT64_C(   89.00), EASYSIMD_FLOAT64_C(   39.10)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   61.70), EASYSIMD_FLOAT64_C(   98.90),
                         EASYSIMD_FLOAT64_C(   -6.00), EASYSIMD_FLOAT64_C(  -89.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(-1180.34), EASYSIMD_FLOAT64_C( 2202.94),
                         EASYSIMD_FLOAT64_C( -501.30), EASYSIMD_FLOAT64_C( 2978.69)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -66.00), EASYSIMD_FLOAT64_C(  -99.10),
                         EASYSIMD_FLOAT64_C(  -51.20), EASYSIMD_FLOAT64_C(   98.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   58.10), EASYSIMD_FLOAT64_C(  -66.70),
                         EASYSIMD_FLOAT64_C(  -86.20), EASYSIMD_FLOAT64_C(   25.30)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -71.40), EASYSIMD_FLOAT64_C(   40.80),
                         EASYSIMD_FLOAT64_C(  -71.40), EASYSIMD_FLOAT64_C(    8.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 3906.00), EASYSIMD_FLOAT64_C(-6650.77),
                         EASYSIMD_FLOAT64_C(-4342.04), EASYSIMD_FLOAT64_C(-2493.36)) },
    { easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   63.90), EASYSIMD_FLOAT64_C(    7.50),
                         EASYSIMD_FLOAT64_C(   -0.00), EASYSIMD_FLOAT64_C(  -97.90)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -39.10), EASYSIMD_FLOAT64_C(  -73.10),
                         EASYSIMD_FLOAT64_C(  -53.20), EASYSIMD_FLOAT64_C(   81.20)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -32.20), EASYSIMD_FLOAT64_C(   71.70),
                         EASYSIMD_FLOAT64_C(   39.30), EASYSIMD_FLOAT64_C(  -11.60)),
      easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C( 2530.69), EASYSIMD_FLOAT64_C(  476.55),
                         EASYSIMD_FLOAT64_C(  -39.30), EASYSIMD_FLOAT64_C( 7961.08)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d r = easysimd_mm256_fnmsub_pd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fnmsub_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   18.60), EASYSIMD_FLOAT32_C(  -96.60), EASYSIMD_FLOAT32_C(  -17.10), EASYSIMD_FLOAT32_C(  -50.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    3.20), EASYSIMD_FLOAT32_C(  -15.90), EASYSIMD_FLOAT32_C(   83.80), EASYSIMD_FLOAT32_C(  -57.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -50.90), EASYSIMD_FLOAT32_C(  -53.70), EASYSIMD_FLOAT32_C(   66.30), EASYSIMD_FLOAT32_C(   53.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -8.62), EASYSIMD_FLOAT32_C(-1482.24), EASYSIMD_FLOAT32_C( 1366.68), EASYSIMD_FLOAT32_C(-2967.35)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   19.40), EASYSIMD_FLOAT32_C(   81.00), EASYSIMD_FLOAT32_C(    0.70), EASYSIMD_FLOAT32_C(   26.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -29.00), EASYSIMD_FLOAT32_C(   15.30), EASYSIMD_FLOAT32_C(  -89.70), EASYSIMD_FLOAT32_C(  -71.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   61.60), EASYSIMD_FLOAT32_C(  -84.00), EASYSIMD_FLOAT32_C(  -77.60), EASYSIMD_FLOAT32_C(   49.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  501.00), EASYSIMD_FLOAT32_C(-1155.30), EASYSIMD_FLOAT32_C(  140.39), EASYSIMD_FLOAT32_C( 1831.86)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   10.90), EASYSIMD_FLOAT32_C(  -73.80), EASYSIMD_FLOAT32_C(  -37.10), EASYSIMD_FLOAT32_C(   92.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   27.80), EASYSIMD_FLOAT32_C(   31.90), EASYSIMD_FLOAT32_C(  -77.70), EASYSIMD_FLOAT32_C(  -29.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   72.70), EASYSIMD_FLOAT32_C(   50.20), EASYSIMD_FLOAT32_C(  -64.40), EASYSIMD_FLOAT32_C(   81.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( -375.72), EASYSIMD_FLOAT32_C( 2304.02), EASYSIMD_FLOAT32_C(-2818.27), EASYSIMD_FLOAT32_C( 2668.64)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -2.50), EASYSIMD_FLOAT32_C(  -77.00), EASYSIMD_FLOAT32_C(  -97.10), EASYSIMD_FLOAT32_C(   -6.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    1.40), EASYSIMD_FLOAT32_C(   38.10), EASYSIMD_FLOAT32_C(   96.80), EASYSIMD_FLOAT32_C(  -90.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -31.70), EASYSIMD_FLOAT32_C(  -86.40), EASYSIMD_FLOAT32_C(  -62.20), EASYSIMD_FLOAT32_C(  -64.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   35.20), EASYSIMD_FLOAT32_C( 3020.10), EASYSIMD_FLOAT32_C( 9461.48), EASYSIMD_FLOAT32_C( -495.78)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -29.50), EASYSIMD_FLOAT32_C(  -45.60), EASYSIMD_FLOAT32_C(  -87.90), EASYSIMD_FLOAT32_C(  -82.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -16.40), EASYSIMD_FLOAT32_C(  -50.10), EASYSIMD_FLOAT32_C(  -30.70), EASYSIMD_FLOAT32_C(  -73.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   49.20), EASYSIMD_FLOAT32_C(   55.00), EASYSIMD_FLOAT32_C(   57.30), EASYSIMD_FLOAT32_C(  -33.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( -533.00), EASYSIMD_FLOAT32_C(-2339.56), EASYSIMD_FLOAT32_C(-2755.83), EASYSIMD_FLOAT32_C(-6001.90)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -90.10), EASYSIMD_FLOAT32_C(   83.90), EASYSIMD_FLOAT32_C(  -87.40), EASYSIMD_FLOAT32_C(  -87.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -1.40), EASYSIMD_FLOAT32_C(  -10.10), EASYSIMD_FLOAT32_C(   29.30), EASYSIMD_FLOAT32_C(  -74.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -21.90), EASYSIMD_FLOAT32_C(   46.80), EASYSIMD_FLOAT32_C(  -76.50), EASYSIMD_FLOAT32_C(  -94.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C( -104.24), EASYSIMD_FLOAT32_C(  800.59), EASYSIMD_FLOAT32_C( 2637.32), EASYSIMD_FLOAT32_C(-6435.64)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -29.30), EASYSIMD_FLOAT32_C(  -94.30), EASYSIMD_FLOAT32_C(   -8.20), EASYSIMD_FLOAT32_C(  -67.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -39.50), EASYSIMD_FLOAT32_C(   47.60), EASYSIMD_FLOAT32_C(   50.70), EASYSIMD_FLOAT32_C(   19.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    2.50), EASYSIMD_FLOAT32_C(   40.50), EASYSIMD_FLOAT32_C(  -73.30), EASYSIMD_FLOAT32_C(    7.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(-1159.85), EASYSIMD_FLOAT32_C( 4448.18), EASYSIMD_FLOAT32_C(  489.04), EASYSIMD_FLOAT32_C( 1309.86)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   15.30), EASYSIMD_FLOAT32_C(   14.60), EASYSIMD_FLOAT32_C(  -68.80), EASYSIMD_FLOAT32_C(   92.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -34.50), EASYSIMD_FLOAT32_C(   77.40), EASYSIMD_FLOAT32_C(   73.70), EASYSIMD_FLOAT32_C(  -25.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -70.20), EASYSIMD_FLOAT32_C(   -4.40), EASYSIMD_FLOAT32_C(  -93.70), EASYSIMD_FLOAT32_C(   16.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  598.05), EASYSIMD_FLOAT32_C(-1125.64), EASYSIMD_FLOAT32_C( 5164.26), EASYSIMD_FLOAT32_C( 2312.98)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fnmsub_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm256_fnmsub_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m256 a;
    easysimd__m256 b;
    easysimd__m256 c;
    easysimd__m256 r;
  } test_vec[8] = {
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -91.80), EASYSIMD_FLOAT32_C(  -53.10),
                         EASYSIMD_FLOAT32_C(  -79.10), EASYSIMD_FLOAT32_C(   50.50),
                         EASYSIMD_FLOAT32_C(  -81.20), EASYSIMD_FLOAT32_C(  -11.90),
                         EASYSIMD_FLOAT32_C(  -72.60), EASYSIMD_FLOAT32_C(   13.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   11.90), EASYSIMD_FLOAT32_C(   -8.00),
                         EASYSIMD_FLOAT32_C(   73.10), EASYSIMD_FLOAT32_C(   73.00),
                         EASYSIMD_FLOAT32_C(  -15.70), EASYSIMD_FLOAT32_C(   33.70),
                         EASYSIMD_FLOAT32_C(  -36.30), EASYSIMD_FLOAT32_C(  -25.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -90.00), EASYSIMD_FLOAT32_C(  -13.00),
                         EASYSIMD_FLOAT32_C(  -28.10), EASYSIMD_FLOAT32_C(  -49.40),
                         EASYSIMD_FLOAT32_C(  -74.60), EASYSIMD_FLOAT32_C(  -32.00),
                         EASYSIMD_FLOAT32_C(  -63.50), EASYSIMD_FLOAT32_C(  -18.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 1182.42), EASYSIMD_FLOAT32_C( -411.80),
                         EASYSIMD_FLOAT32_C( 5810.31), EASYSIMD_FLOAT32_C(-3637.10),
                         EASYSIMD_FLOAT32_C(-1200.24), EASYSIMD_FLOAT32_C(  433.03),
                         EASYSIMD_FLOAT32_C(-2571.88), EASYSIMD_FLOAT32_C(  369.78)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -57.30), EASYSIMD_FLOAT32_C(   71.50),
                         EASYSIMD_FLOAT32_C(   39.90), EASYSIMD_FLOAT32_C(  -77.10),
                         EASYSIMD_FLOAT32_C(   -9.90), EASYSIMD_FLOAT32_C(  -16.00),
                         EASYSIMD_FLOAT32_C(   74.80), EASYSIMD_FLOAT32_C(   77.40)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -53.60), EASYSIMD_FLOAT32_C(   49.60),
                         EASYSIMD_FLOAT32_C(   94.20), EASYSIMD_FLOAT32_C(    1.20),
                         EASYSIMD_FLOAT32_C(  -56.30), EASYSIMD_FLOAT32_C(   26.10),
                         EASYSIMD_FLOAT32_C(  -23.40), EASYSIMD_FLOAT32_C(  -47.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   82.40), EASYSIMD_FLOAT32_C(  -13.50),
                         EASYSIMD_FLOAT32_C(  -97.40), EASYSIMD_FLOAT32_C(   84.50),
                         EASYSIMD_FLOAT32_C(  -48.30), EASYSIMD_FLOAT32_C(   98.50),
                         EASYSIMD_FLOAT32_C(  -91.50), EASYSIMD_FLOAT32_C(   24.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-3153.68), EASYSIMD_FLOAT32_C(-3532.90),
                         EASYSIMD_FLOAT32_C(-3661.18), EASYSIMD_FLOAT32_C(    8.02),
                         EASYSIMD_FLOAT32_C( -509.07), EASYSIMD_FLOAT32_C(  319.10),
                         EASYSIMD_FLOAT32_C( 1841.82), EASYSIMD_FLOAT32_C( 3613.20)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   11.60), EASYSIMD_FLOAT32_C(   59.90),
                         EASYSIMD_FLOAT32_C(   -3.20), EASYSIMD_FLOAT32_C(    4.40),
                         EASYSIMD_FLOAT32_C(  -98.80), EASYSIMD_FLOAT32_C(   29.00),
                         EASYSIMD_FLOAT32_C(  -86.20), EASYSIMD_FLOAT32_C(   19.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   63.60), EASYSIMD_FLOAT32_C(  -94.60),
                         EASYSIMD_FLOAT32_C(  -81.40), EASYSIMD_FLOAT32_C(    9.90),
                         EASYSIMD_FLOAT32_C(  -69.00), EASYSIMD_FLOAT32_C(  -83.90),
                         EASYSIMD_FLOAT32_C(   22.00), EASYSIMD_FLOAT32_C(  -56.30)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   80.90), EASYSIMD_FLOAT32_C(   -7.90),
                         EASYSIMD_FLOAT32_C(  -92.10), EASYSIMD_FLOAT32_C(   65.40),
                         EASYSIMD_FLOAT32_C(  -26.30), EASYSIMD_FLOAT32_C(  -26.90),
                         EASYSIMD_FLOAT32_C(  -44.20), EASYSIMD_FLOAT32_C(  -39.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -818.66), EASYSIMD_FLOAT32_C( 5674.44),
                         EASYSIMD_FLOAT32_C( -168.38), EASYSIMD_FLOAT32_C( -108.96),
                         EASYSIMD_FLOAT32_C(-6790.90), EASYSIMD_FLOAT32_C( 2460.00),
                         EASYSIMD_FLOAT32_C( 1940.60), EASYSIMD_FLOAT32_C( 1137.45)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -73.70), EASYSIMD_FLOAT32_C(  -39.20),
                         EASYSIMD_FLOAT32_C(   40.90), EASYSIMD_FLOAT32_C(    0.60),
                         EASYSIMD_FLOAT32_C(  -64.50), EASYSIMD_FLOAT32_C(   35.70),
                         EASYSIMD_FLOAT32_C(  -58.10), EASYSIMD_FLOAT32_C(  -23.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   98.30), EASYSIMD_FLOAT32_C(   77.60),
                         EASYSIMD_FLOAT32_C(   33.80), EASYSIMD_FLOAT32_C(   94.20),
                         EASYSIMD_FLOAT32_C(    8.60), EASYSIMD_FLOAT32_C(  -96.70),
                         EASYSIMD_FLOAT32_C(  -22.70), EASYSIMD_FLOAT32_C(  -38.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -52.20), EASYSIMD_FLOAT32_C(  -35.80),
                         EASYSIMD_FLOAT32_C(   76.20), EASYSIMD_FLOAT32_C(  -32.30),
                         EASYSIMD_FLOAT32_C(  -84.60), EASYSIMD_FLOAT32_C(   76.00),
                         EASYSIMD_FLOAT32_C(  -84.30), EASYSIMD_FLOAT32_C(   87.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( 7296.91), EASYSIMD_FLOAT32_C( 3077.72),
                         EASYSIMD_FLOAT32_C(-1458.62), EASYSIMD_FLOAT32_C(  -24.22),
                         EASYSIMD_FLOAT32_C(  639.30), EASYSIMD_FLOAT32_C( 3376.19),
                         EASYSIMD_FLOAT32_C(-1234.57), EASYSIMD_FLOAT32_C( -992.00)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -10.70), EASYSIMD_FLOAT32_C(   52.50),
                         EASYSIMD_FLOAT32_C(   95.50), EASYSIMD_FLOAT32_C(  -35.90),
                         EASYSIMD_FLOAT32_C(  -55.60), EASYSIMD_FLOAT32_C(    1.10),
                         EASYSIMD_FLOAT32_C(  -20.80), EASYSIMD_FLOAT32_C(  -55.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -20.50), EASYSIMD_FLOAT32_C(   25.80),
                         EASYSIMD_FLOAT32_C(   85.10), EASYSIMD_FLOAT32_C(  -30.10),
                         EASYSIMD_FLOAT32_C(   98.50), EASYSIMD_FLOAT32_C(  -42.90),
                         EASYSIMD_FLOAT32_C(   14.30), EASYSIMD_FLOAT32_C(   52.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   81.70), EASYSIMD_FLOAT32_C(   21.40),
                         EASYSIMD_FLOAT32_C(   41.10), EASYSIMD_FLOAT32_C(   65.30),
                         EASYSIMD_FLOAT32_C(  -66.60), EASYSIMD_FLOAT32_C(    6.20),
                         EASYSIMD_FLOAT32_C(   29.60), EASYSIMD_FLOAT32_C(   47.20)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C( -301.05), EASYSIMD_FLOAT32_C(-1375.90),
                         EASYSIMD_FLOAT32_C(-8168.15), EASYSIMD_FLOAT32_C(-1145.89),
                         EASYSIMD_FLOAT32_C( 5543.20), EASYSIMD_FLOAT32_C(   40.99),
                         EASYSIMD_FLOAT32_C(  267.84), EASYSIMD_FLOAT32_C( 2872.10)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -94.70), EASYSIMD_FLOAT32_C(   14.30),
                         EASYSIMD_FLOAT32_C(   36.30), EASYSIMD_FLOAT32_C(  -95.40),
                         EASYSIMD_FLOAT32_C(  -85.70), EASYSIMD_FLOAT32_C(   15.60),
                         EASYSIMD_FLOAT32_C(  -45.20), EASYSIMD_FLOAT32_C(  -87.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -75.50), EASYSIMD_FLOAT32_C(   52.00),
                         EASYSIMD_FLOAT32_C(   88.60), EASYSIMD_FLOAT32_C(  -12.10),
                         EASYSIMD_FLOAT32_C(  -27.40), EASYSIMD_FLOAT32_C(   41.00),
                         EASYSIMD_FLOAT32_C(  -70.80), EASYSIMD_FLOAT32_C(   22.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -14.10), EASYSIMD_FLOAT32_C(  -90.60),
                         EASYSIMD_FLOAT32_C(   84.80), EASYSIMD_FLOAT32_C(  -47.50),
                         EASYSIMD_FLOAT32_C(  -49.90), EASYSIMD_FLOAT32_C(   72.50),
                         EASYSIMD_FLOAT32_C(   90.90), EASYSIMD_FLOAT32_C(  -74.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-7135.75), EASYSIMD_FLOAT32_C( -653.00),
                         EASYSIMD_FLOAT32_C(-3300.98), EASYSIMD_FLOAT32_C(-1106.84),
                         EASYSIMD_FLOAT32_C(-2298.28), EASYSIMD_FLOAT32_C( -712.10),
                         EASYSIMD_FLOAT32_C(-3291.06), EASYSIMD_FLOAT32_C( 2052.10)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -59.20), EASYSIMD_FLOAT32_C(  -79.60),
                         EASYSIMD_FLOAT32_C(   47.00), EASYSIMD_FLOAT32_C(  -96.90),
                         EASYSIMD_FLOAT32_C(  -44.60), EASYSIMD_FLOAT32_C(   50.20),
                         EASYSIMD_FLOAT32_C(   10.60), EASYSIMD_FLOAT32_C(  -70.80)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -68.90), EASYSIMD_FLOAT32_C(   37.70),
                         EASYSIMD_FLOAT32_C(   58.60), EASYSIMD_FLOAT32_C(  -25.80),
                         EASYSIMD_FLOAT32_C(   57.80), EASYSIMD_FLOAT32_C(  -89.20),
                         EASYSIMD_FLOAT32_C(   27.50), EASYSIMD_FLOAT32_C(   46.60)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   61.30), EASYSIMD_FLOAT32_C(  -66.60),
                         EASYSIMD_FLOAT32_C(   75.60), EASYSIMD_FLOAT32_C(   -6.00),
                         EASYSIMD_FLOAT32_C(  -95.90), EASYSIMD_FLOAT32_C(   11.80),
                         EASYSIMD_FLOAT32_C(   59.10), EASYSIMD_FLOAT32_C(   34.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-4140.18), EASYSIMD_FLOAT32_C( 3067.52),
                         EASYSIMD_FLOAT32_C(-2829.80), EASYSIMD_FLOAT32_C(-2494.02),
                         EASYSIMD_FLOAT32_C( 2673.78), EASYSIMD_FLOAT32_C( 4466.04),
                         EASYSIMD_FLOAT32_C( -350.60), EASYSIMD_FLOAT32_C( 3264.38)) },
    { easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -53.00), EASYSIMD_FLOAT32_C(  -46.10),
                         EASYSIMD_FLOAT32_C(   53.90), EASYSIMD_FLOAT32_C(   19.20),
                         EASYSIMD_FLOAT32_C(  -73.10), EASYSIMD_FLOAT32_C(   23.40),
                         EASYSIMD_FLOAT32_C(   67.90), EASYSIMD_FLOAT32_C(  -74.50)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -80.10), EASYSIMD_FLOAT32_C(   56.30),
                         EASYSIMD_FLOAT32_C(  -45.20), EASYSIMD_FLOAT32_C(   32.20),
                         EASYSIMD_FLOAT32_C(  -17.90), EASYSIMD_FLOAT32_C(  -44.50),
                         EASYSIMD_FLOAT32_C(   62.10), EASYSIMD_FLOAT32_C(    8.00)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   30.70), EASYSIMD_FLOAT32_C(   11.70),
                         EASYSIMD_FLOAT32_C(  -61.10), EASYSIMD_FLOAT32_C(   76.30),
                         EASYSIMD_FLOAT32_C(   57.20), EASYSIMD_FLOAT32_C(   36.40),
                         EASYSIMD_FLOAT32_C(   67.50), EASYSIMD_FLOAT32_C(    2.90)),
      easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(-4276.00), EASYSIMD_FLOAT32_C( 2583.73),
                         EASYSIMD_FLOAT32_C( 2497.38), EASYSIMD_FLOAT32_C( -694.54),
                         EASYSIMD_FLOAT32_C(-1365.69), EASYSIMD_FLOAT32_C( 1004.90),
                         EASYSIMD_FLOAT32_C(-4284.09), EASYSIMD_FLOAT32_C(  593.10)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 r = easysimd_mm256_fnmsub_ps(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m256_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fnmsub_sd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128d a;
    easysimd__m128d b;
    easysimd__m128d c;
    easysimd__m128d r;
  } test_vec[8] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -60.10), EASYSIMD_FLOAT64_C(  -84.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -22.80), EASYSIMD_FLOAT64_C(   63.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   81.10), EASYSIMD_FLOAT64_C(  -77.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -60.10), EASYSIMD_FLOAT64_C( 5386.00)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   37.60), EASYSIMD_FLOAT64_C(   78.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -10.00), EASYSIMD_FLOAT64_C(   53.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -88.70), EASYSIMD_FLOAT64_C(   54.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   37.60), EASYSIMD_FLOAT64_C(-4283.94)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -38.20), EASYSIMD_FLOAT64_C(   72.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -86.10), EASYSIMD_FLOAT64_C(   25.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -52.90), EASYSIMD_FLOAT64_C(  -86.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -38.20), EASYSIMD_FLOAT64_C(-1752.25)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    7.50), EASYSIMD_FLOAT64_C(   35.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    5.30), EASYSIMD_FLOAT64_C(   97.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -49.40), EASYSIMD_FLOAT64_C(  -58.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    7.50), EASYSIMD_FLOAT64_C(-3336.40)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   41.40), EASYSIMD_FLOAT64_C(  -46.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -70.70), EASYSIMD_FLOAT64_C(  -78.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   77.70), EASYSIMD_FLOAT64_C(  -33.20)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   41.40), EASYSIMD_FLOAT64_C(-3629.69)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -63.30), EASYSIMD_FLOAT64_C(  -78.40)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   58.80), EASYSIMD_FLOAT64_C(   11.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -18.00), EASYSIMD_FLOAT64_C(  -49.30)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -63.30), EASYSIMD_FLOAT64_C(  950.90)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -80.50), EASYSIMD_FLOAT64_C(   28.50)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -48.00), EASYSIMD_FLOAT64_C(   38.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -73.40), EASYSIMD_FLOAT64_C(  -29.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -80.50), EASYSIMD_FLOAT64_C(-1056.85)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   42.80), EASYSIMD_FLOAT64_C(  -10.90)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    8.60), EASYSIMD_FLOAT64_C(  -39.60)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -25.50), EASYSIMD_FLOAT64_C(   42.10)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   42.80), EASYSIMD_FLOAT64_C( -473.74)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r = easysimd_mm_fnmsub_sd(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm_fnmsub_ss(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m128 a;
    easysimd__m128 b;
    easysimd__m128 c;
    easysimd__m128 r;
  } test_vec[8] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   17.30), EASYSIMD_FLOAT32_C(   17.40), EASYSIMD_FLOAT32_C(   41.70), EASYSIMD_FLOAT32_C(   37.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   61.00), EASYSIMD_FLOAT32_C(   15.10), EASYSIMD_FLOAT32_C(    2.00), EASYSIMD_FLOAT32_C(   43.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   28.00), EASYSIMD_FLOAT32_C(   83.70), EASYSIMD_FLOAT32_C(   43.30), EASYSIMD_FLOAT32_C(  -38.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   17.30), EASYSIMD_FLOAT32_C(   17.40), EASYSIMD_FLOAT32_C(   41.70), EASYSIMD_FLOAT32_C(-1563.90)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -37.00), EASYSIMD_FLOAT32_C(  -28.20), EASYSIMD_FLOAT32_C(   12.60), EASYSIMD_FLOAT32_C(  -73.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    5.90), EASYSIMD_FLOAT32_C(   68.10), EASYSIMD_FLOAT32_C(   57.10), EASYSIMD_FLOAT32_C(   23.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   17.40), EASYSIMD_FLOAT32_C(   89.40), EASYSIMD_FLOAT32_C(   38.60), EASYSIMD_FLOAT32_C(  -36.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -37.00), EASYSIMD_FLOAT32_C(  -28.20), EASYSIMD_FLOAT32_C(   12.60), EASYSIMD_FLOAT32_C( 1785.80)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   35.60), EASYSIMD_FLOAT32_C(  -64.00), EASYSIMD_FLOAT32_C(   95.10), EASYSIMD_FLOAT32_C(  -83.40)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   90.40), EASYSIMD_FLOAT32_C(   58.10), EASYSIMD_FLOAT32_C(   -8.40), EASYSIMD_FLOAT32_C(  -87.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   26.50), EASYSIMD_FLOAT32_C(  -91.50), EASYSIMD_FLOAT32_C(   38.20), EASYSIMD_FLOAT32_C(   39.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   35.60), EASYSIMD_FLOAT32_C(  -64.00), EASYSIMD_FLOAT32_C(   95.10), EASYSIMD_FLOAT32_C(-7370.06)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   26.00), EASYSIMD_FLOAT32_C(   35.10), EASYSIMD_FLOAT32_C(   90.70), EASYSIMD_FLOAT32_C(  -77.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -18.70), EASYSIMD_FLOAT32_C(   97.20), EASYSIMD_FLOAT32_C(  -13.90), EASYSIMD_FLOAT32_C(    3.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -66.00), EASYSIMD_FLOAT32_C(  -38.90), EASYSIMD_FLOAT32_C(   92.90), EASYSIMD_FLOAT32_C(   44.20)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   26.00), EASYSIMD_FLOAT32_C(   35.10), EASYSIMD_FLOAT32_C(   90.70), EASYSIMD_FLOAT32_C(  233.72)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   63.90), EASYSIMD_FLOAT32_C(  -84.10), EASYSIMD_FLOAT32_C(   20.70), EASYSIMD_FLOAT32_C(  -87.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    2.30), EASYSIMD_FLOAT32_C(  -39.50), EASYSIMD_FLOAT32_C(  -17.30), EASYSIMD_FLOAT32_C(  -98.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -59.10), EASYSIMD_FLOAT32_C(  -12.50), EASYSIMD_FLOAT32_C(   12.60), EASYSIMD_FLOAT32_C(   34.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   63.90), EASYSIMD_FLOAT32_C(  -84.10), EASYSIMD_FLOAT32_C(   20.70), EASYSIMD_FLOAT32_C(-8613.10)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -37.30), EASYSIMD_FLOAT32_C(  -17.50), EASYSIMD_FLOAT32_C(  -37.30), EASYSIMD_FLOAT32_C(   -7.90)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   85.10), EASYSIMD_FLOAT32_C(  -93.00), EASYSIMD_FLOAT32_C(   -6.70), EASYSIMD_FLOAT32_C(   16.00)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   17.50), EASYSIMD_FLOAT32_C(  -83.60), EASYSIMD_FLOAT32_C(   98.60), EASYSIMD_FLOAT32_C(  -20.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -37.30), EASYSIMD_FLOAT32_C(  -17.50), EASYSIMD_FLOAT32_C(  -37.30), EASYSIMD_FLOAT32_C(  146.90)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -98.40), EASYSIMD_FLOAT32_C(   46.60), EASYSIMD_FLOAT32_C(  -57.20), EASYSIMD_FLOAT32_C(  -62.70)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   23.90), EASYSIMD_FLOAT32_C(   59.10), EASYSIMD_FLOAT32_C(   62.20), EASYSIMD_FLOAT32_C(   48.60)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -80.80), EASYSIMD_FLOAT32_C(  -51.00), EASYSIMD_FLOAT32_C(   63.40), EASYSIMD_FLOAT32_C(   30.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -98.40), EASYSIMD_FLOAT32_C(   46.60), EASYSIMD_FLOAT32_C(  -57.20), EASYSIMD_FLOAT32_C( 3017.12)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -52.30), EASYSIMD_FLOAT32_C(   90.80), EASYSIMD_FLOAT32_C(   10.20), EASYSIMD_FLOAT32_C(   40.80)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   25.10), EASYSIMD_FLOAT32_C(   -1.00), EASYSIMD_FLOAT32_C(   38.80), EASYSIMD_FLOAT32_C(    1.50)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -21.30), EASYSIMD_FLOAT32_C(  -30.30), EASYSIMD_FLOAT32_C(   80.90), EASYSIMD_FLOAT32_C(  -98.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -52.30), EASYSIMD_FLOAT32_C(   90.80), EASYSIMD_FLOAT32_C(   10.20), EASYSIMD_FLOAT32_C(   36.90)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r = easysimd_mm_fnmsub_ss(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    easysimd_assert_m128_close(r, test_vec[i].r, 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fmadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmadd_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmadd_ss)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmaddsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fmaddsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmaddsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fmaddsub_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmsub_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmsub_ss)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmsubadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fmsubadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fmsubadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fmsubadd_ps)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fnmadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fnmadd_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fnmadd_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fnmadd_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fnmadd_ss)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fnmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fnmsub_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fnmsub_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fnmsub_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fnmsub_ss)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/test-x86-footer.h>
