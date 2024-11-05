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

#define EASYSIMD_TEST_X86_AVX512_INSN setr4

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/setr4.h>

static int
test_easysimd_mm512_setr4_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    int32_t d; int32_t c; int32_t b; int32_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    { INT32_C(  440568275),
      INT32_C(-1307171366),
      INT32_C( -667071334),
      INT32_C(-1006059139),
      easysimd_mm512_set_epi32(INT32_C(-1006059139), INT32_C( -667071334), INT32_C(-1307171366), INT32_C(  440568275),
                            INT32_C(-1006059139), INT32_C( -667071334), INT32_C(-1307171366), INT32_C(  440568275),
                            INT32_C(-1006059139), INT32_C( -667071334), INT32_C(-1307171366), INT32_C(  440568275),
                            INT32_C(-1006059139), INT32_C( -667071334), INT32_C(-1307171366), INT32_C(  440568275)) },
    { INT32_C(  985235756),
      INT32_C(-2117652171),
      INT32_C( -492848785),
      INT32_C(  765678538),
      easysimd_mm512_set_epi32(INT32_C(  765678538), INT32_C( -492848785), INT32_C(-2117652171), INT32_C(  985235756),
                            INT32_C(  765678538), INT32_C( -492848785), INT32_C(-2117652171), INT32_C(  985235756),
                            INT32_C(  765678538), INT32_C( -492848785), INT32_C(-2117652171), INT32_C(  985235756),
                            INT32_C(  765678538), INT32_C( -492848785), INT32_C(-2117652171), INT32_C(  985235756)) },
    { INT32_C( 1812566322),
      INT32_C( -457041277),
      INT32_C(-1069434801),
      INT32_C( -605856203),
      easysimd_mm512_set_epi32(INT32_C( -605856203), INT32_C(-1069434801), INT32_C( -457041277), INT32_C( 1812566322),
                            INT32_C( -605856203), INT32_C(-1069434801), INT32_C( -457041277), INT32_C( 1812566322),
                            INT32_C( -605856203), INT32_C(-1069434801), INT32_C( -457041277), INT32_C( 1812566322),
                            INT32_C( -605856203), INT32_C(-1069434801), INT32_C( -457041277), INT32_C( 1812566322)) },
    { INT32_C( 1968671665),
      INT32_C(  838296696),
      INT32_C( -693015358),
      INT32_C(-1386069498),
      easysimd_mm512_set_epi32(INT32_C(-1386069498), INT32_C( -693015358), INT32_C(  838296696), INT32_C( 1968671665),
                            INT32_C(-1386069498), INT32_C( -693015358), INT32_C(  838296696), INT32_C( 1968671665),
                            INT32_C(-1386069498), INT32_C( -693015358), INT32_C(  838296696), INT32_C( 1968671665),
                            INT32_C(-1386069498), INT32_C( -693015358), INT32_C(  838296696), INT32_C( 1968671665)) },
    { INT32_C(  717585874),
      INT32_C( -870190090),
      INT32_C(   62628055),
      INT32_C(-1058408989),
      easysimd_mm512_set_epi32(INT32_C(-1058408989), INT32_C(   62628055), INT32_C( -870190090), INT32_C(  717585874),
                            INT32_C(-1058408989), INT32_C(   62628055), INT32_C( -870190090), INT32_C(  717585874),
                            INT32_C(-1058408989), INT32_C(   62628055), INT32_C( -870190090), INT32_C(  717585874),
                            INT32_C(-1058408989), INT32_C(   62628055), INT32_C( -870190090), INT32_C(  717585874)) },
    { INT32_C( -646678116),
      INT32_C( -636471021),
      INT32_C( 2050242002),
      INT32_C( 1467573389),
      easysimd_mm512_set_epi32(INT32_C( 1467573389), INT32_C( 2050242002), INT32_C( -636471021), INT32_C( -646678116),
                            INT32_C( 1467573389), INT32_C( 2050242002), INT32_C( -636471021), INT32_C( -646678116),
                            INT32_C( 1467573389), INT32_C( 2050242002), INT32_C( -636471021), INT32_C( -646678116),
                            INT32_C( 1467573389), INT32_C( 2050242002), INT32_C( -636471021), INT32_C( -646678116)) },
    { INT32_C( -468604998),
      INT32_C(  416458537),
      INT32_C(-1356493538),
      INT32_C( -338084785),
      easysimd_mm512_set_epi32(INT32_C( -338084785), INT32_C(-1356493538), INT32_C(  416458537), INT32_C( -468604998),
                            INT32_C( -338084785), INT32_C(-1356493538), INT32_C(  416458537), INT32_C( -468604998),
                            INT32_C( -338084785), INT32_C(-1356493538), INT32_C(  416458537), INT32_C( -468604998),
                            INT32_C( -338084785), INT32_C(-1356493538), INT32_C(  416458537), INT32_C( -468604998)) },
    { INT32_C( 1519812884),
      INT32_C(  743581731),
      INT32_C(-1035717687),
      INT32_C(  -38963525),
      easysimd_mm512_set_epi32(INT32_C(  -38963525), INT32_C(-1035717687), INT32_C(  743581731), INT32_C( 1519812884),
                            INT32_C(  -38963525), INT32_C(-1035717687), INT32_C(  743581731), INT32_C( 1519812884),
                            INT32_C(  -38963525), INT32_C(-1035717687), INT32_C(  743581731), INT32_C( 1519812884),
                            INT32_C(  -38963525), INT32_C(-1035717687), INT32_C(  743581731), INT32_C( 1519812884)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int32_t d = test_vec[i].d;
    int32_t c = test_vec[i].c;
    int32_t b = test_vec[i].b;
    int32_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_setr4_epi32(d, c, b, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_setr4_epi32");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_setr4_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    int64_t d; int64_t c; int64_t b; int64_t a;
    easysimd__m512i r;
  } test_vec[8] = {
    { INT64_C( 6563849718269597141),
      INT64_C(-6183679436467555899),
      INT64_C( -626758305238464386),
      INT64_C( 8994159492887548356),
      easysimd_mm512_set_epi64(INT64_C( 8994159492887548356), INT64_C( -626758305238464386),
                            INT64_C(-6183679436467555899), INT64_C( 6563849718269597141),
                            INT64_C( 8994159492887548356), INT64_C( -626758305238464386),
                            INT64_C(-6183679436467555899), INT64_C( 6563849718269597141)) },
    { INT64_C( 6729148419260484779),
      INT64_C( -277511807975612216),
      INT64_C(-8071294684814160544),
      INT64_C( 9097495128638227239),
      easysimd_mm512_set_epi64(INT64_C( 9097495128638227239), INT64_C(-8071294684814160544),
                            INT64_C( -277511807975612216), INT64_C( 6729148419260484779),
                            INT64_C( 9097495128638227239), INT64_C(-8071294684814160544),
                            INT64_C( -277511807975612216), INT64_C( 6729148419260484779)) },
    { INT64_C( 2451446111308764542),
      INT64_C( 7443262200234995807),
      INT64_C( 1452118817457897022),
      INT64_C( 8577124855339817739),
      easysimd_mm512_set_epi64(INT64_C( 8577124855339817739), INT64_C( 1452118817457897022),
                            INT64_C( 7443262200234995807), INT64_C( 2451446111308764542),
                            INT64_C( 8577124855339817739), INT64_C( 1452118817457897022),
                            INT64_C( 7443262200234995807), INT64_C( 2451446111308764542)) },
    { INT64_C( 5794476453905478874),
      INT64_C(-1405809235656433875),
      INT64_C(-9152840578969258696),
      INT64_C( 8562326329950659697),
      easysimd_mm512_set_epi64(INT64_C( 8562326329950659697), INT64_C(-9152840578969258696),
                            INT64_C(-1405809235656433875), INT64_C( 5794476453905478874),
                            INT64_C( 8562326329950659697), INT64_C(-9152840578969258696),
                            INT64_C(-1405809235656433875), INT64_C( 5794476453905478874)) },
    { INT64_C(-8764167661207563767),
      INT64_C( -157881503650322899),
      INT64_C(-4202918664443291804),
      INT64_C( 2806446076990238010),
      easysimd_mm512_set_epi64(INT64_C( 2806446076990238010), INT64_C(-4202918664443291804),
                            INT64_C( -157881503650322899), INT64_C(-8764167661207563767),
                            INT64_C( 2806446076990238010), INT64_C(-4202918664443291804),
                            INT64_C( -157881503650322899), INT64_C(-8764167661207563767)) },
    { INT64_C(-5837281652074857748),
      INT64_C(-7080037588592146058),
      INT64_C(-4482514275105483583),
      INT64_C( 7870122127635681284),
      easysimd_mm512_set_epi64(INT64_C( 7870122127635681284), INT64_C(-4482514275105483583),
                            INT64_C(-7080037588592146058), INT64_C(-5837281652074857748),
                            INT64_C( 7870122127635681284), INT64_C(-4482514275105483583),
                            INT64_C(-7080037588592146058), INT64_C(-5837281652074857748)) },
    { INT64_C(-2741649954653767454),
      INT64_C( 7022257894354348987),
      INT64_C(-7607333645615092101),
      INT64_C( 3821399499306603551),
      easysimd_mm512_set_epi64(INT64_C( 3821399499306603551), INT64_C(-7607333645615092101),
                            INT64_C( 7022257894354348987), INT64_C(-2741649954653767454),
                            INT64_C( 3821399499306603551), INT64_C(-7607333645615092101),
                            INT64_C( 7022257894354348987), INT64_C(-2741649954653767454)) },
    { INT64_C( 6134432460743068033),
      INT64_C( 1716871541978724160),
      INT64_C(-7436535278984624040),
      INT64_C( 1873233539406121615),
      easysimd_mm512_set_epi64(INT64_C( 1873233539406121615), INT64_C(-7436535278984624040),
                            INT64_C( 1716871541978724160), INT64_C( 6134432460743068033),
                            INT64_C( 1873233539406121615), INT64_C(-7436535278984624040),
                            INT64_C( 1716871541978724160), INT64_C( 6134432460743068033)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    int64_t d = test_vec[i].d;
    int64_t c = test_vec[i].c;
    int64_t b = test_vec[i].b;
    int64_t a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_setr4_epi64(d, c, b, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_setr4_epi64");
    easysimd_assert_m512i_i64(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_setr4_ps(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd_float32 d; easysimd_float32 c; easysimd_float32 b; easysimd_float32 a;
    easysimd__m512 r;
  } test_vec[8] = {
    { EASYSIMD_FLOAT32_C(   -92.68),
      EASYSIMD_FLOAT32_C(   845.12),
      EASYSIMD_FLOAT32_C(  -953.73),
      EASYSIMD_FLOAT32_C(   237.00),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   237.00), EASYSIMD_FLOAT32_C(  -953.73), EASYSIMD_FLOAT32_C(   845.12), EASYSIMD_FLOAT32_C(   -92.68),
                         EASYSIMD_FLOAT32_C(   237.00), EASYSIMD_FLOAT32_C(  -953.73), EASYSIMD_FLOAT32_C(   845.12), EASYSIMD_FLOAT32_C(   -92.68),
                         EASYSIMD_FLOAT32_C(   237.00), EASYSIMD_FLOAT32_C(  -953.73), EASYSIMD_FLOAT32_C(   845.12), EASYSIMD_FLOAT32_C(   -92.68),
                         EASYSIMD_FLOAT32_C(   237.00), EASYSIMD_FLOAT32_C(  -953.73), EASYSIMD_FLOAT32_C(   845.12), EASYSIMD_FLOAT32_C(   -92.68)) },
    { EASYSIMD_FLOAT32_C(  -555.84),
      EASYSIMD_FLOAT32_C(  -722.05),
      EASYSIMD_FLOAT32_C(  -788.55),
      EASYSIMD_FLOAT32_C(   545.68),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   545.68), EASYSIMD_FLOAT32_C(  -788.55), EASYSIMD_FLOAT32_C(  -722.05), EASYSIMD_FLOAT32_C(  -555.84),
                         EASYSIMD_FLOAT32_C(   545.68), EASYSIMD_FLOAT32_C(  -788.55), EASYSIMD_FLOAT32_C(  -722.05), EASYSIMD_FLOAT32_C(  -555.84),
                         EASYSIMD_FLOAT32_C(   545.68), EASYSIMD_FLOAT32_C(  -788.55), EASYSIMD_FLOAT32_C(  -722.05), EASYSIMD_FLOAT32_C(  -555.84),
                         EASYSIMD_FLOAT32_C(   545.68), EASYSIMD_FLOAT32_C(  -788.55), EASYSIMD_FLOAT32_C(  -722.05), EASYSIMD_FLOAT32_C(  -555.84)) },
    { EASYSIMD_FLOAT32_C(   823.18),
      EASYSIMD_FLOAT32_C(  -207.95),
      EASYSIMD_FLOAT32_C(  -413.77),
      EASYSIMD_FLOAT32_C(   808.21),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   808.21), EASYSIMD_FLOAT32_C(  -413.77), EASYSIMD_FLOAT32_C(  -207.95), EASYSIMD_FLOAT32_C(   823.18),
                         EASYSIMD_FLOAT32_C(   808.21), EASYSIMD_FLOAT32_C(  -413.77), EASYSIMD_FLOAT32_C(  -207.95), EASYSIMD_FLOAT32_C(   823.18),
                         EASYSIMD_FLOAT32_C(   808.21), EASYSIMD_FLOAT32_C(  -413.77), EASYSIMD_FLOAT32_C(  -207.95), EASYSIMD_FLOAT32_C(   823.18),
                         EASYSIMD_FLOAT32_C(   808.21), EASYSIMD_FLOAT32_C(  -413.77), EASYSIMD_FLOAT32_C(  -207.95), EASYSIMD_FLOAT32_C(   823.18)) },
    { EASYSIMD_FLOAT32_C(  -179.14),
      EASYSIMD_FLOAT32_C(    28.27),
      EASYSIMD_FLOAT32_C(  -190.88),
      EASYSIMD_FLOAT32_C(  -337.32),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -337.32), EASYSIMD_FLOAT32_C(  -190.88), EASYSIMD_FLOAT32_C(    28.27), EASYSIMD_FLOAT32_C(  -179.14),
                         EASYSIMD_FLOAT32_C(  -337.32), EASYSIMD_FLOAT32_C(  -190.88), EASYSIMD_FLOAT32_C(    28.27), EASYSIMD_FLOAT32_C(  -179.14),
                         EASYSIMD_FLOAT32_C(  -337.32), EASYSIMD_FLOAT32_C(  -190.88), EASYSIMD_FLOAT32_C(    28.27), EASYSIMD_FLOAT32_C(  -179.14),
                         EASYSIMD_FLOAT32_C(  -337.32), EASYSIMD_FLOAT32_C(  -190.88), EASYSIMD_FLOAT32_C(    28.27), EASYSIMD_FLOAT32_C(  -179.14)) },
    { EASYSIMD_FLOAT32_C(  -691.46),
      EASYSIMD_FLOAT32_C(  -801.82),
      EASYSIMD_FLOAT32_C(  -579.89),
      EASYSIMD_FLOAT32_C(  -420.42),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -420.42), EASYSIMD_FLOAT32_C(  -579.89), EASYSIMD_FLOAT32_C(  -801.82), EASYSIMD_FLOAT32_C(  -691.46),
                         EASYSIMD_FLOAT32_C(  -420.42), EASYSIMD_FLOAT32_C(  -579.89), EASYSIMD_FLOAT32_C(  -801.82), EASYSIMD_FLOAT32_C(  -691.46),
                         EASYSIMD_FLOAT32_C(  -420.42), EASYSIMD_FLOAT32_C(  -579.89), EASYSIMD_FLOAT32_C(  -801.82), EASYSIMD_FLOAT32_C(  -691.46),
                         EASYSIMD_FLOAT32_C(  -420.42), EASYSIMD_FLOAT32_C(  -579.89), EASYSIMD_FLOAT32_C(  -801.82), EASYSIMD_FLOAT32_C(  -691.46)) },
    { EASYSIMD_FLOAT32_C(   490.22),
      EASYSIMD_FLOAT32_C(   560.02),
      EASYSIMD_FLOAT32_C(  -244.24),
      EASYSIMD_FLOAT32_C(   184.70),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   184.70), EASYSIMD_FLOAT32_C(  -244.24), EASYSIMD_FLOAT32_C(   560.02), EASYSIMD_FLOAT32_C(   490.22),
                         EASYSIMD_FLOAT32_C(   184.70), EASYSIMD_FLOAT32_C(  -244.24), EASYSIMD_FLOAT32_C(   560.02), EASYSIMD_FLOAT32_C(   490.22),
                         EASYSIMD_FLOAT32_C(   184.70), EASYSIMD_FLOAT32_C(  -244.24), EASYSIMD_FLOAT32_C(   560.02), EASYSIMD_FLOAT32_C(   490.22),
                         EASYSIMD_FLOAT32_C(   184.70), EASYSIMD_FLOAT32_C(  -244.24), EASYSIMD_FLOAT32_C(   560.02), EASYSIMD_FLOAT32_C(   490.22)) },
    { EASYSIMD_FLOAT32_C(   353.38),
      EASYSIMD_FLOAT32_C(   199.20),
      EASYSIMD_FLOAT32_C(   132.74),
      EASYSIMD_FLOAT32_C(   599.57),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   599.57), EASYSIMD_FLOAT32_C(   132.74), EASYSIMD_FLOAT32_C(   199.20), EASYSIMD_FLOAT32_C(   353.38),
                         EASYSIMD_FLOAT32_C(   599.57), EASYSIMD_FLOAT32_C(   132.74), EASYSIMD_FLOAT32_C(   199.20), EASYSIMD_FLOAT32_C(   353.38),
                         EASYSIMD_FLOAT32_C(   599.57), EASYSIMD_FLOAT32_C(   132.74), EASYSIMD_FLOAT32_C(   199.20), EASYSIMD_FLOAT32_C(   353.38),
                         EASYSIMD_FLOAT32_C(   599.57), EASYSIMD_FLOAT32_C(   132.74), EASYSIMD_FLOAT32_C(   199.20), EASYSIMD_FLOAT32_C(   353.38)) },
    { EASYSIMD_FLOAT32_C(  -109.85),
      EASYSIMD_FLOAT32_C(    62.56),
      EASYSIMD_FLOAT32_C(   250.77),
      EASYSIMD_FLOAT32_C(  -873.95),
      easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -873.95), EASYSIMD_FLOAT32_C(   250.77), EASYSIMD_FLOAT32_C(    62.56), EASYSIMD_FLOAT32_C(  -109.85),
                         EASYSIMD_FLOAT32_C(  -873.95), EASYSIMD_FLOAT32_C(   250.77), EASYSIMD_FLOAT32_C(    62.56), EASYSIMD_FLOAT32_C(  -109.85),
                         EASYSIMD_FLOAT32_C(  -873.95), EASYSIMD_FLOAT32_C(   250.77), EASYSIMD_FLOAT32_C(    62.56), EASYSIMD_FLOAT32_C(  -109.85),
                         EASYSIMD_FLOAT32_C(  -873.95), EASYSIMD_FLOAT32_C(   250.77), EASYSIMD_FLOAT32_C(    62.56), EASYSIMD_FLOAT32_C(  -109.85)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd_float32 d = test_vec[i].d;
    easysimd_float32 c = test_vec[i].c;
    easysimd_float32 b = test_vec[i].b;
    easysimd_float32 a = test_vec[i].a;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_setr4_ps(d, c, b, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_setr4_ps");
    easysimd_assert_m512_close(r, test_vec[i].r, 1);
  }

  return 0;
}

static int
test_easysimd_mm512_setr4_pd(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd_float64  d; easysimd_float64  c; easysimd_float64  b; easysimd_float64  a;
    easysimd__m512d r;
  } test_vec[8] = {
   {  EASYSIMD_FLOAT64_C( -159.85),
      EASYSIMD_FLOAT64_C(  360.42),
      EASYSIMD_FLOAT64_C( -560.02),
      EASYSIMD_FLOAT64_C( -340.11),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -340.11), EASYSIMD_FLOAT64_C( -560.02),
                         EASYSIMD_FLOAT64_C(  360.42), EASYSIMD_FLOAT64_C( -159.85),
                         EASYSIMD_FLOAT64_C( -340.11), EASYSIMD_FLOAT64_C( -560.02),
                         EASYSIMD_FLOAT64_C(  360.42), EASYSIMD_FLOAT64_C( -159.85)) },
   {  EASYSIMD_FLOAT64_C(   76.83),
      EASYSIMD_FLOAT64_C( -871.20),
      EASYSIMD_FLOAT64_C(  277.42),
      EASYSIMD_FLOAT64_C(  632.86),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  632.86), EASYSIMD_FLOAT64_C(  277.42),
                         EASYSIMD_FLOAT64_C( -871.20), EASYSIMD_FLOAT64_C(   76.83),
                         EASYSIMD_FLOAT64_C(  632.86), EASYSIMD_FLOAT64_C(  277.42),
                         EASYSIMD_FLOAT64_C( -871.20), EASYSIMD_FLOAT64_C(   76.83)) },
   {  EASYSIMD_FLOAT64_C(  908.32),
      EASYSIMD_FLOAT64_C( -754.84),
      EASYSIMD_FLOAT64_C( -232.66),
      EASYSIMD_FLOAT64_C(  453.94),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  453.94), EASYSIMD_FLOAT64_C( -232.66),
                         EASYSIMD_FLOAT64_C( -754.84), EASYSIMD_FLOAT64_C(  908.32),
                         EASYSIMD_FLOAT64_C(  453.94), EASYSIMD_FLOAT64_C( -232.66),
                         EASYSIMD_FLOAT64_C( -754.84), EASYSIMD_FLOAT64_C(  908.32)) },
   {  EASYSIMD_FLOAT64_C(  389.27),
      EASYSIMD_FLOAT64_C(  400.56),
      EASYSIMD_FLOAT64_C(  223.12),
      EASYSIMD_FLOAT64_C( -299.15),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -299.15), EASYSIMD_FLOAT64_C(  223.12),
                         EASYSIMD_FLOAT64_C(  400.56), EASYSIMD_FLOAT64_C(  389.27),
                         EASYSIMD_FLOAT64_C( -299.15), EASYSIMD_FLOAT64_C(  223.12),
                         EASYSIMD_FLOAT64_C(  400.56), EASYSIMD_FLOAT64_C(  389.27)) },
   {  EASYSIMD_FLOAT64_C(  642.96),
      EASYSIMD_FLOAT64_C(  603.97),
      EASYSIMD_FLOAT64_C( -782.74),
      EASYSIMD_FLOAT64_C(  593.11),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  593.11), EASYSIMD_FLOAT64_C( -782.74),
                         EASYSIMD_FLOAT64_C(  603.97), EASYSIMD_FLOAT64_C(  642.96),
                         EASYSIMD_FLOAT64_C(  593.11), EASYSIMD_FLOAT64_C( -782.74),
                         EASYSIMD_FLOAT64_C(  603.97), EASYSIMD_FLOAT64_C(  642.96)) },
   {  EASYSIMD_FLOAT64_C(  918.13),
      EASYSIMD_FLOAT64_C(  886.70),
      EASYSIMD_FLOAT64_C(  337.10),
      EASYSIMD_FLOAT64_C( -359.87),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C( -359.87), EASYSIMD_FLOAT64_C(  337.10),
                         EASYSIMD_FLOAT64_C(  886.70), EASYSIMD_FLOAT64_C(  918.13),
                         EASYSIMD_FLOAT64_C( -359.87), EASYSIMD_FLOAT64_C(  337.10),
                         EASYSIMD_FLOAT64_C(  886.70), EASYSIMD_FLOAT64_C(  918.13)) },
   {  EASYSIMD_FLOAT64_C(  794.16),
      EASYSIMD_FLOAT64_C( -191.83),
      EASYSIMD_FLOAT64_C( -298.69),
      EASYSIMD_FLOAT64_C(  612.50),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  612.50), EASYSIMD_FLOAT64_C( -298.69),
                         EASYSIMD_FLOAT64_C( -191.83), EASYSIMD_FLOAT64_C(  794.16),
                         EASYSIMD_FLOAT64_C(  612.50), EASYSIMD_FLOAT64_C( -298.69),
                         EASYSIMD_FLOAT64_C( -191.83), EASYSIMD_FLOAT64_C(  794.16)) },
   {  EASYSIMD_FLOAT64_C(  850.90),
      EASYSIMD_FLOAT64_C( -669.22),
      EASYSIMD_FLOAT64_C(  -90.20),
      EASYSIMD_FLOAT64_C(  431.18),
      easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  431.18), EASYSIMD_FLOAT64_C(  -90.20),
                         EASYSIMD_FLOAT64_C( -669.22), EASYSIMD_FLOAT64_C(  850.90),
                         EASYSIMD_FLOAT64_C(  431.18), EASYSIMD_FLOAT64_C(  -90.20),
                         EASYSIMD_FLOAT64_C( -669.22), EASYSIMD_FLOAT64_C(  850.90)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd_float64 d = test_vec[i].d;
    easysimd_float64 c = test_vec[i].c;
    easysimd_float64 b = test_vec[i].b;
    easysimd_float64 a = test_vec[i].a;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_setr4_pd(d, c, b, a);
    }
    EASYSIMD_TEST_PERF_END("_mm512_setr4_pd");
    easysimd_assert_m512d_close(r, test_vec[i].r, 1);
  }

  return 0;
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_setr4_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_setr4_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_setr4_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_setr4_pd )
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
