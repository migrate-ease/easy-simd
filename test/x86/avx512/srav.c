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

#define EASYSIMD_TEST_X86_AVX512_INSN srav

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/srav.h>

static int
test_easysimd_mm512_srav_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t b[32];
    const int16_t r[32];
  } test_vec[] = {
    { { -INT16_C( 26126),  INT16_C(  9315),  INT16_C( 20615), -INT16_C( 25676),  INT16_C( 25129), -INT16_C( 13781), -INT16_C( 22935), -INT16_C( 22311),
        -INT16_C(   968),  INT16_C( 28073), -INT16_C(  1215),  INT16_C(  1561), -INT16_C(  5797),  INT16_C( 15623), -INT16_C( 24120), -INT16_C( 17815),
        -INT16_C( 13254), -INT16_C( 15649), -INT16_C( 27875),  INT16_C( 18013), -INT16_C( 30475),  INT16_C( 24080), -INT16_C(  5841),  INT16_C( 26374),
        -INT16_C( 20250),  INT16_C( 10196), -INT16_C(  4693),  INT16_C(  1581),  INT16_C( 13782), -INT16_C( 24765), -INT16_C( 21290),  INT16_C(  4185) },
      {  INT16_C(     3),  INT16_C(     9),  INT16_C(     2),  INT16_C(    12),  INT16_C(    14),  INT16_C(    14),  INT16_C(     2),  INT16_C(    11),
         INT16_C(     2),  INT16_C(     8),  INT16_C(     0),  INT16_C(    14),  INT16_C(    12),  INT16_C(     1),  INT16_C(    13),  INT16_C(    14),
         INT16_C(    15),  INT16_C(    14),  INT16_C(     5),  INT16_C(    14),  INT16_C(    12),  INT16_C(     2),  INT16_C(     1),  INT16_C(    12),
         INT16_C(    11),  INT16_C(     4),  INT16_C(    12),  INT16_C(     1),  INT16_C(    11),  INT16_C(     0),  INT16_C(     5),  INT16_C(    11) },
      { -INT16_C(  3266),  INT16_C(    18),  INT16_C(  5153), -INT16_C(     7),  INT16_C(     1), -INT16_C(     1), -INT16_C(  5734), -INT16_C(    11),
        -INT16_C(   242),  INT16_C(   109), -INT16_C(  1215),  INT16_C(     0), -INT16_C(     2),  INT16_C(  7811), -INT16_C(     3), -INT16_C(     2),
        -INT16_C(     1), -INT16_C(     1), -INT16_C(   872),  INT16_C(     1), -INT16_C(     8),  INT16_C(  6020), -INT16_C(  2921),  INT16_C(     6),
        -INT16_C(    10),  INT16_C(   637), -INT16_C(     2),  INT16_C(   790),  INT16_C(     6), -INT16_C( 24765), -INT16_C(   666),  INT16_C(     2) } },
    { {  INT16_C( 31316), -INT16_C( 32362),  INT16_C( 15066),  INT16_C(  9574),  INT16_C( 13054), -INT16_C(  6075),  INT16_C(  8268), -INT16_C( 30040),
        -INT16_C(  5667), -INT16_C( 22056),  INT16_C(  3505),  INT16_C( 17338), -INT16_C(  6456),  INT16_C( 25163),  INT16_C( 16956), -INT16_C( 28651),
        -INT16_C( 21572), -INT16_C( 27119),  INT16_C( 30693), -INT16_C(  6981),  INT16_C(   169), -INT16_C(  2356),  INT16_C( 29984), -INT16_C(   640),
         INT16_C( 22622),  INT16_C(  4263),  INT16_C( 24933),  INT16_C( 11603), -INT16_C( 24761), -INT16_C( 31601), -INT16_C( 23327), -INT16_C( 25324) },
      {  INT16_C(     2),  INT16_C(     3),  INT16_C(    14),  INT16_C(     4),  INT16_C(    14),  INT16_C(     1),  INT16_C(    11),  INT16_C(    11),
         INT16_C(    11),  INT16_C(     7),  INT16_C(     1),  INT16_C(     5),  INT16_C(     3),  INT16_C(     9),  INT16_C(    15),  INT16_C(     2),
         INT16_C(     6),  INT16_C(    11),  INT16_C(     7),  INT16_C(     4),  INT16_C(     3),  INT16_C(    11),  INT16_C(     6),  INT16_C(     1),
         INT16_C(     3),  INT16_C(     3),  INT16_C(     3),  INT16_C(     1),  INT16_C(     7),  INT16_C(     4),  INT16_C(    14),  INT16_C(     8) },
      {  INT16_C(  7829), -INT16_C(  4046),  INT16_C(     0),  INT16_C(   598),  INT16_C(     0), -INT16_C(  3038),  INT16_C(     4), -INT16_C(    15),
        -INT16_C(     3), -INT16_C(   173),  INT16_C(  1752),  INT16_C(   541), -INT16_C(   807),  INT16_C(    49),  INT16_C(     0), -INT16_C(  7163),
        -INT16_C(   338), -INT16_C(    14),  INT16_C(   239), -INT16_C(   437),  INT16_C(    21), -INT16_C(     2),  INT16_C(   468), -INT16_C(   320),
         INT16_C(  2827),  INT16_C(   532),  INT16_C(  3116),  INT16_C(  5801), -INT16_C(   194), -INT16_C(  1976), -INT16_C(     2), -INT16_C(    99) } },
    { { -INT16_C( 13225), -INT16_C( 19388),  INT16_C( 17733), -INT16_C( 23806),  INT16_C( 24707),  INT16_C( 32347), -INT16_C( 12852), -INT16_C(  4722),
         INT16_C(  6151),  INT16_C( 24100), -INT16_C( 17847), -INT16_C( 20111),  INT16_C(  8241), -INT16_C( 24589),  INT16_C( 24585),  INT16_C( 24618),
         INT16_C( 28205),  INT16_C( 29204),  INT16_C(  5812),  INT16_C( 14101),  INT16_C( 28790),  INT16_C( 17078),  INT16_C( 17469),  INT16_C( 17711),
         INT16_C( 21597), -INT16_C( 22877),  INT16_C(  5134),  INT16_C( 16215),  INT16_C( 19252),  INT16_C( 15839),  INT16_C(  2475), -INT16_C( 10083) },
      {  INT16_C(    11),  INT16_C(     2),  INT16_C(     5),  INT16_C(     3),  INT16_C(     1),  INT16_C(     0),  INT16_C(    11),  INT16_C(    11),
         INT16_C(    15),  INT16_C(     1),  INT16_C(    11),  INT16_C(     3),  INT16_C(     3),  INT16_C(    10),  INT16_C(     1),  INT16_C(    11),
         INT16_C(    13),  INT16_C(     9),  INT16_C(     4),  INT16_C(     0),  INT16_C(     5),  INT16_C(    11),  INT16_C(     6),  INT16_C(     0),
         INT16_C(    13),  INT16_C(     6),  INT16_C(     7),  INT16_C(     8),  INT16_C(     1),  INT16_C(    13),  INT16_C(    12),  INT16_C(     0) },
      { -INT16_C(     7), -INT16_C(  4847),  INT16_C(   554), -INT16_C(  2976),  INT16_C( 12353),  INT16_C( 32347), -INT16_C(     7), -INT16_C(     3),
         INT16_C(     0),  INT16_C( 12050), -INT16_C(     9), -INT16_C(  2514),  INT16_C(  1030), -INT16_C(    25),  INT16_C( 12292),  INT16_C(    12),
         INT16_C(     3),  INT16_C(    57),  INT16_C(   363),  INT16_C( 14101),  INT16_C(   899),  INT16_C(     8),  INT16_C(   272),  INT16_C( 17711),
         INT16_C(     2), -INT16_C(   358),  INT16_C(    40),  INT16_C(    63),  INT16_C(  9626),  INT16_C(     1),  INT16_C(     0), -INT16_C( 10083) } },
    { {  INT16_C( 27799), -INT16_C( 14184),  INT16_C( 27564),  INT16_C(  1738), -INT16_C(  9792), -INT16_C( 14659),  INT16_C( 11834), -INT16_C( 27951),
        -INT16_C(  4351), -INT16_C( 29452), -INT16_C( 27296),  INT16_C(   538),  INT16_C( 22706), -INT16_C(  5410),  INT16_C( 27933), -INT16_C( 19219),
        -INT16_C( 31271), -INT16_C( 31364),  INT16_C( 18161), -INT16_C( 20085),  INT16_C( 18463),  INT16_C( 23160),  INT16_C( 18807),  INT16_C( 30956),
        -INT16_C(  8135), -INT16_C( 26364),  INT16_C(  7797),  INT16_C( 10139),  INT16_C( 31094), -INT16_C( 27887), -INT16_C(    26), -INT16_C( 16569) },
      {  INT16_C(    12),  INT16_C(     7),  INT16_C(    13),  INT16_C(     2),  INT16_C(     9),  INT16_C(     9),  INT16_C(     7),  INT16_C(     2),
         INT16_C(     0),  INT16_C(    12),  INT16_C(     5),  INT16_C(    10),  INT16_C(    15),  INT16_C(    11),  INT16_C(     7),  INT16_C(     8),
         INT16_C(    11),  INT16_C(     4),  INT16_C(     1),  INT16_C(    10),  INT16_C(    15),  INT16_C(    10),  INT16_C(     3),  INT16_C(    11),
         INT16_C(     8),  INT16_C(     7),  INT16_C(     6),  INT16_C(    10),  INT16_C(     4),  INT16_C(     6),  INT16_C(    13),  INT16_C(     0) },
      {  INT16_C(     6), -INT16_C(   111),  INT16_C(     3),  INT16_C(   434), -INT16_C(    20), -INT16_C(    29),  INT16_C(    92), -INT16_C(  6988),
        -INT16_C(  4351), -INT16_C(     8), -INT16_C(   853),  INT16_C(     0),  INT16_C(     0), -INT16_C(     3),  INT16_C(   218), -INT16_C(    76),
        -INT16_C(    16), -INT16_C(  1961),  INT16_C(  9080), -INT16_C(    20),  INT16_C(     0),  INT16_C(    22),  INT16_C(  2350),  INT16_C(    15),
        -INT16_C(    32), -INT16_C(   206),  INT16_C(   121),  INT16_C(     9),  INT16_C(  1943), -INT16_C(   436), -INT16_C(     1), -INT16_C( 16569) } },
    { { -INT16_C(  8822),  INT16_C(  5454), -INT16_C( 15621), -INT16_C( 18248), -INT16_C(  4933),  INT16_C(  9054),  INT16_C(  9511),  INT16_C( 28636),
         INT16_C( 22950),  INT16_C( 32225), -INT16_C(  2877),  INT16_C( 11043),  INT16_C( 32571), -INT16_C(   112),  INT16_C( 30543), -INT16_C(  9726),
         INT16_C( 20564),  INT16_C( 20719), -INT16_C( 22765), -INT16_C( 12792),  INT16_C( 26259), -INT16_C( 17423), -INT16_C( 12917),  INT16_C( 12842),
         INT16_C(  2855), -INT16_C(  5457), -INT16_C( 11265),  INT16_C( 14870), -INT16_C( 22958), -INT16_C( 24263),  INT16_C( 15389),  INT16_C( 29307) },
      {  INT16_C(     6),  INT16_C(     9),  INT16_C(    12),  INT16_C(    10),  INT16_C(     5),  INT16_C(    11),  INT16_C(     8),  INT16_C(     5),
         INT16_C(     9),  INT16_C(     9),  INT16_C(     5),  INT16_C(    12),  INT16_C(     0),  INT16_C(     1),  INT16_C(    14),  INT16_C(    13),
         INT16_C(     4),  INT16_C(     5),  INT16_C(    13),  INT16_C(     4),  INT16_C(     6),  INT16_C(     6),  INT16_C(    15),  INT16_C(     8),
         INT16_C(    15),  INT16_C(     0),  INT16_C(    14),  INT16_C(     4),  INT16_C(     2),  INT16_C(     3),  INT16_C(    14),  INT16_C(     5) },
      { -INT16_C(   138),  INT16_C(    10), -INT16_C(     4), -INT16_C(    18), -INT16_C(   155),  INT16_C(     4),  INT16_C(    37),  INT16_C(   894),
         INT16_C(    44),  INT16_C(    62), -INT16_C(    90),  INT16_C(     2),  INT16_C( 32571), -INT16_C(    56),  INT16_C(     1), -INT16_C(     2),
         INT16_C(  1285),  INT16_C(   647), -INT16_C(     3), -INT16_C(   800),  INT16_C(   410), -INT16_C(   273), -INT16_C(     1),  INT16_C(    50),
         INT16_C(     0), -INT16_C(  5457), -INT16_C(     1),  INT16_C(   929), -INT16_C(  5740), -INT16_C(  3033),  INT16_C(     0),  INT16_C(   915) } },
    { {  INT16_C( 29751),  INT16_C( 20144), -INT16_C( 19886), -INT16_C( 28779), -INT16_C( 26348),  INT16_C(   505), -INT16_C( 18804),  INT16_C(  7300),
        -INT16_C( 25679),  INT16_C(    30), -INT16_C(  7551),  INT16_C( 28489), -INT16_C( 21749),  INT16_C(  5282), -INT16_C( 22890), -INT16_C( 12696),
         INT16_C(  6171),  INT16_C( 27932), -INT16_C( 20022), -INT16_C(  8451), -INT16_C(  2485), -INT16_C( 10272),  INT16_C( 25772),  INT16_C( 24051),
         INT16_C(  4607), -INT16_C( 32675), -INT16_C( 22796), -INT16_C(    17), -INT16_C( 28079), -INT16_C(  6124),  INT16_C( 31800),  INT16_C( 21430) },
      {  INT16_C(    13),  INT16_C(     5),  INT16_C(    11),  INT16_C(    12),  INT16_C(     1),  INT16_C(     6),  INT16_C(     9),  INT16_C(     7),
         INT16_C(     1),  INT16_C(     9),  INT16_C(    14),  INT16_C(     1),  INT16_C(    11),  INT16_C(    11),  INT16_C(    11),  INT16_C(    12),
         INT16_C(    12),  INT16_C(     0),  INT16_C(     5),  INT16_C(     4),  INT16_C(     7),  INT16_C(    15),  INT16_C(     5),  INT16_C(    11),
         INT16_C(     7),  INT16_C(     3),  INT16_C(    15),  INT16_C(    14),  INT16_C(     4),  INT16_C(    13),  INT16_C(    10),  INT16_C(     7) },
      {  INT16_C(     3),  INT16_C(   629), -INT16_C(    10), -INT16_C(     8), -INT16_C( 13174),  INT16_C(     7), -INT16_C(    37),  INT16_C(    57),
        -INT16_C( 12840),  INT16_C(     0), -INT16_C(     1),  INT16_C( 14244), -INT16_C(    11),  INT16_C(     2), -INT16_C(    12), -INT16_C(     4),
         INT16_C(     1),  INT16_C( 27932), -INT16_C(   626), -INT16_C(   529), -INT16_C(    20), -INT16_C(     1),  INT16_C(   805),  INT16_C(    11),
         INT16_C(    35), -INT16_C(  4085), -INT16_C(     1), -INT16_C(     1), -INT16_C(  1755), -INT16_C(     1),  INT16_C(    31),  INT16_C(   167) } },
    { { -INT16_C( 17539),  INT16_C(  2427),  INT16_C( 20248), -INT16_C( 28343), -INT16_C(  5688), -INT16_C(  9334),  INT16_C(   838), -INT16_C( 17000),
        -INT16_C(  3204), -INT16_C(  7180),  INT16_C( 16109), -INT16_C( 26420),  INT16_C( 28289),  INT16_C( 30066),  INT16_C(  3357), -INT16_C( 25878),
         INT16_C( 26057), -INT16_C(  7773), -INT16_C(  4940),  INT16_C( 32114), -INT16_C(   811),  INT16_C(  7000), -INT16_C(  4096),  INT16_C( 31960),
        -INT16_C( 13085), -INT16_C( 12193),  INT16_C( 11018), -INT16_C( 29591), -INT16_C(  9319), -INT16_C( 18943), -INT16_C(  5144), -INT16_C( 20144) },
      {  INT16_C(    15),  INT16_C(     0),  INT16_C(     0),  INT16_C(    11),  INT16_C(    13),  INT16_C(     0),  INT16_C(    10),  INT16_C(    10),
         INT16_C(    13),  INT16_C(     7),  INT16_C(    14),  INT16_C(    10),  INT16_C(     0),  INT16_C(    10),  INT16_C(    10),  INT16_C(     4),
         INT16_C(    14),  INT16_C(     7),  INT16_C(    13),  INT16_C(    15),  INT16_C(    15),  INT16_C(     7),  INT16_C(     7),  INT16_C(     1),
         INT16_C(    10),  INT16_C(     5),  INT16_C(    10),  INT16_C(     4),  INT16_C(     5),  INT16_C(    10),  INT16_C(     5),  INT16_C(     9) },
      { -INT16_C(     1),  INT16_C(  2427),  INT16_C( 20248), -INT16_C(    14), -INT16_C(     1), -INT16_C(  9334),  INT16_C(     0), -INT16_C(    17),
        -INT16_C(     1), -INT16_C(    57),  INT16_C(     0), -INT16_C(    26),  INT16_C( 28289),  INT16_C(    29),  INT16_C(     3), -INT16_C(  1618),
         INT16_C(     1), -INT16_C(    61), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),  INT16_C(    54), -INT16_C(    32),  INT16_C( 15980),
        -INT16_C(    13), -INT16_C(   382),  INT16_C(    10), -INT16_C(  1850), -INT16_C(   292), -INT16_C(    19), -INT16_C(   161), -INT16_C(    40) } },
    { {  INT16_C( 15426),  INT16_C( 13584),  INT16_C( 16396), -INT16_C( 18902),  INT16_C(  8000), -INT16_C(  6357),  INT16_C( 20114), -INT16_C(  7934),
        -INT16_C( 25616),  INT16_C( 31032),  INT16_C( 12351), -INT16_C(  4156), -INT16_C( 17535),  INT16_C( 31381), -INT16_C( 31730),  INT16_C( 20495),
         INT16_C(  8128), -INT16_C( 13179), -INT16_C( 20640), -INT16_C( 24446), -INT16_C( 20785),  INT16_C( 24967), -INT16_C( 30212), -INT16_C(  5054),
         INT16_C( 31268),  INT16_C( 25701),  INT16_C( 10922),  INT16_C( 11091), -INT16_C(  5915), -INT16_C(  3163), -INT16_C( 19348),  INT16_C( 11331) },
      {  INT16_C(    12),  INT16_C(     3),  INT16_C(     7),  INT16_C(     4),  INT16_C(     5),  INT16_C(     2),  INT16_C(    14),  INT16_C(     0),
         INT16_C(     7),  INT16_C(     0),  INT16_C(    11),  INT16_C(     8),  INT16_C(    13),  INT16_C(     1),  INT16_C(    11),  INT16_C(     6),
         INT16_C(     3),  INT16_C(    15),  INT16_C(     7),  INT16_C(    13),  INT16_C(    14),  INT16_C(    10),  INT16_C(     1),  INT16_C(     3),
         INT16_C(     2),  INT16_C(     2),  INT16_C(     8),  INT16_C(     8),  INT16_C(     2),  INT16_C(    15),  INT16_C(    14),  INT16_C(     6) },
      {  INT16_C(     3),  INT16_C(  1698),  INT16_C(   128), -INT16_C(  1182),  INT16_C(   250), -INT16_C(  1590),  INT16_C(     1), -INT16_C(  7934),
        -INT16_C(   201),  INT16_C( 31032),  INT16_C(     6), -INT16_C(    17), -INT16_C(     3),  INT16_C( 15690), -INT16_C(    16),  INT16_C(   320),
         INT16_C(  1016), -INT16_C(     1), -INT16_C(   162), -INT16_C(     3), -INT16_C(     2),  INT16_C(    24), -INT16_C( 15106), -INT16_C(   632),
         INT16_C(  7817),  INT16_C(  6425),  INT16_C(    42),  INT16_C(    43), -INT16_C(  1479), -INT16_C(     1), -INT16_C(     2),  INT16_C(   177) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi16(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_srav_epi16(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_srav_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
}

static int
test_easysimd_mm512_srav_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1598104928), -INT32_C(    85428437), -INT32_C(  1628440452), -INT32_C(  1771996618),  INT32_C(  1713347371), -INT32_C(   351316457), -INT32_C(  1688458930), -INT32_C(  1243778028),
         INT32_C(  2052430671), -INT32_C(  1904984814),  INT32_C(  1814913845),  INT32_C(   134385373),  INT32_C(  1080959273), -INT32_C(   970162824), -INT32_C(  1151235930),  INT32_C(  1265647356) },
      { -INT32_C(  1598104928), -INT32_C(    85428437), -INT32_C(  1628440452), -INT32_C(  1771996618),  INT32_C(  1713347371), -INT32_C(   351316457), -INT32_C(  1688458930), -INT32_C(  1243778028),
         INT32_C(  2052430671), -INT32_C(  1904984814),  INT32_C(  1814913845),  INT32_C(   134385373),  INT32_C(  1080959273), -INT32_C(   970162824), -INT32_C(  1151235930),  INT32_C(  1265647356) },
      { -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),
         INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0) } },
    { { -INT32_C(   859073743),  INT32_C(   100928889),  INT32_C(  2143991539),  INT32_C(   332721644),  INT32_C(  1068066992),  INT32_C(    86135811), -INT32_C(  1918020144),  INT32_C(  1448072229),
        -INT32_C(   534635673),  INT32_C(   468133416), -INT32_C(   929386020),  INT32_C(  1323003550), -INT32_C(   628259625), -INT32_C(  1444958247),  INT32_C(   691440644), -INT32_C(   931166624) },
      { -INT32_C(   859073743),  INT32_C(   100928889),  INT32_C(  2143991539),  INT32_C(   332721644),  INT32_C(  1068066992),  INT32_C(    86135811), -INT32_C(  1918020144),  INT32_C(  1448072229),
        -INT32_C(   534635673),  INT32_C(   468133416), -INT32_C(   929386020),  INT32_C(  1323003550), -INT32_C(   628259625), -INT32_C(  1444958247),  INT32_C(   691440644), -INT32_C(   931166624) },
      { -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),
        -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1) } },
    { { -INT32_C(  1960359134),  INT32_C(  2100644517), -INT32_C(  1321025831), -INT32_C(   514471053), -INT32_C(  1766285582),  INT32_C(  1430518839), -INT32_C(   948020777), -INT32_C(  1016114015),
         INT32_C(  1162843808), -INT32_C(  1161657120), -INT32_C(  1251277758), -INT32_C(  1047019313), -INT32_C(  1789440162),  INT32_C(   602577484), -INT32_C(  1779799565),  INT32_C(  1566071485) },
      { -INT32_C(  1960359134),  INT32_C(  2100644517), -INT32_C(  1321025831), -INT32_C(   514471053), -INT32_C(  1766285582),  INT32_C(  1430518839), -INT32_C(   948020777), -INT32_C(  1016114015),
         INT32_C(  1162843808), -INT32_C(  1161657120), -INT32_C(  1251277758), -INT32_C(  1047019313), -INT32_C(  1789440162),  INT32_C(   602577484), -INT32_C(  1779799565),  INT32_C(  1566071485) },
      { -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),
         INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0) } },
    { { -INT32_C(  1256581483),  INT32_C(   557655840),  INT32_C(  1138228311), -INT32_C(  1137073484),  INT32_C(   565019927),  INT32_C(  1965587275), -INT32_C(   983700805),  INT32_C(  1514337476),
         INT32_C(   705649674), -INT32_C(   615822205), -INT32_C(    48356536), -INT32_C(   575055931),  INT32_C(   687761116), -INT32_C(  1667422495), -INT32_C(   731776240), -INT32_C(   114383889) },
      { -INT32_C(  1256581483),  INT32_C(   557655840),  INT32_C(  1138228311), -INT32_C(  1137073484),  INT32_C(   565019927),  INT32_C(  1965587275), -INT32_C(   983700805),  INT32_C(  1514337476),
         INT32_C(   705649674), -INT32_C(   615822205), -INT32_C(    48356536), -INT32_C(   575055931),  INT32_C(   687761116), -INT32_C(  1667422495), -INT32_C(   731776240), -INT32_C(   114383889) },
      { -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),
         INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1) } },
    { {  INT32_C(   796926788),  INT32_C(  1152157045), -INT32_C(  1259962280), -INT32_C(   973260663),  INT32_C(    22615650), -INT32_C(   384739943),  INT32_C(   883523983), -INT32_C(  1920804535),
         INT32_C(  1924924157), -INT32_C(   524916601),  INT32_C(  1922342120),  INT32_C(   909611476),  INT32_C(  1094226087),  INT32_C(  1948928485),  INT32_C(   329831626), -INT32_C(  1432343891) },
      {  INT32_C(   796926788),  INT32_C(  1152157045), -INT32_C(  1259962280), -INT32_C(   973260663),  INT32_C(    22615650), -INT32_C(   384739943),  INT32_C(   883523983), -INT32_C(  1920804535),
         INT32_C(  1924924157), -INT32_C(   524916601),  INT32_C(  1922342120),  INT32_C(   909611476),  INT32_C(  1094226087),  INT32_C(  1948928485),  INT32_C(   329831626), -INT32_C(  1432343891) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),
         INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1) } },
    { {  INT32_C(   609804091),  INT32_C(   444506472), -INT32_C(   947577847),  INT32_C(    23810613), -INT32_C(    20325438),  INT32_C(   584866186), -INT32_C(  1462957885),  INT32_C(  1751431213),
         INT32_C(     9223576),  INT32_C(  1746537054),  INT32_C(  1529847846), -INT32_C(  1268999182),  INT32_C(    11740790),  INT32_C(  1109626751), -INT32_C(   991235945),  INT32_C(  1093422761) },
      {  INT32_C(   609804091),  INT32_C(   444506472), -INT32_C(   947577847),  INT32_C(    23810613), -INT32_C(    20325438),  INT32_C(   584866186), -INT32_C(  1462957885),  INT32_C(  1751431213),
         INT32_C(     9223576),  INT32_C(  1746537054),  INT32_C(  1529847846), -INT32_C(  1268999182),  INT32_C(    11740790),  INT32_C(  1109626751), -INT32_C(   991235945),  INT32_C(  1093422761) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0) } },
    { {  INT32_C(   145392730), -INT32_C(  1749910643),  INT32_C(  1228753809),  INT32_C(   415823494), -INT32_C(   997290558), -INT32_C(  1662156325),  INT32_C(  1377691059),  INT32_C(  1209363437),
         INT32_C(  2102444016), -INT32_C(   871103685), -INT32_C(   736800434),  INT32_C(   183295304),  INT32_C(  1188002667), -INT32_C(  1344094980), -INT32_C(  1912536927),  INT32_C(  1540757099) },
      {  INT32_C(   145392730), -INT32_C(  1749910643),  INT32_C(  1228753809),  INT32_C(   415823494), -INT32_C(   997290558), -INT32_C(  1662156325),  INT32_C(  1377691059),  INT32_C(  1209363437),
         INT32_C(  2102444016), -INT32_C(   871103685), -INT32_C(   736800434),  INT32_C(   183295304),  INT32_C(  1188002667), -INT32_C(  1344094980), -INT32_C(  1912536927),  INT32_C(  1540757099) },
      {  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_srav_epi32(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_srav_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_srav_epi32(a, b);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

static int
test_easysimd_mm512_srav_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t b[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 2428071496759468640), -INT64_C( 3435435949124692070),  INT64_C( 7799547264520681091), -INT64_C( 4598476069222446358),
        -INT64_C( 6945993161153067184), -INT64_C( 3477715365854025033), -INT64_C( 3100486543016567645),  INT64_C( 1406807276878943589) },
      {  INT64_C( 2428071496759468640), -INT64_C( 3435435949124692070),  INT64_C( 7799547264520681091), -INT64_C( 4598476069222446358),
        -INT64_C( 6945993161153067184), -INT64_C( 3477715365854025033), -INT64_C( 3100486543016567645),  INT64_C( 1406807276878943589) },
      {  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0), -INT64_C(                   1),
        -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0) } },
    { {  INT64_C( 2404432475286458593),  INT64_C( 3841043641847563487), -INT64_C( 8312295205772826939),  INT64_C( 7792304371607500588),
         INT64_C( 2404255077641748388),  INT64_C( 6517517442817750373),  INT64_C( 2548754779460938821),  INT64_C( 4137587144372175221) },
      {  INT64_C( 2404432475286458593),  INT64_C( 3841043641847563487), -INT64_C( 8312295205772826939),  INT64_C( 7792304371607500588),
         INT64_C( 2404255077641748388),  INT64_C( 6517517442817750373),  INT64_C( 2548754779460938821),  INT64_C( 4137587144372175221) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { { -INT64_C( 4031469660817441990), -INT64_C( 4329465478841371029),  INT64_C( 2582883339178310279),  INT64_C( 2303091915902197630),
        -INT64_C( 1157094683137069964),  INT64_C( 8573372576272828450), -INT64_C( 5327576695269623242),  INT64_C( 3046924524326190148) },
      { -INT64_C( 4031469660817441990), -INT64_C( 4329465478841371029),  INT64_C( 2582883339178310279),  INT64_C( 2303091915902197630),
        -INT64_C( 1157094683137069964),  INT64_C( 8573372576272828450), -INT64_C( 5327576695269623242),  INT64_C( 3046924524326190148) },
      { -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1),  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0) } },
    { { -INT64_C( 4607406850882974681),  INT64_C( 5907535700271908070),  INT64_C(  348337312164908564), -INT64_C( 2126028078111818677),
         INT64_C(  270856013705898273),  INT64_C( 4847982268351997834), -INT64_C( 5260008187895095028), -INT64_C( 7364693290268187602) },
      { -INT64_C( 4607406850882974681),  INT64_C( 5907535700271908070),  INT64_C(  348337312164908564), -INT64_C( 2126028078111818677),
         INT64_C(  270856013705898273),  INT64_C( 4847982268351997834), -INT64_C( 5260008187895095028), -INT64_C( 7364693290268187602) },
      { -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),
         INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1) } },
    { {  INT64_C( 3777339205567884907),  INT64_C( 8469260244416938588),  INT64_C( 6717752645020316502), -INT64_C( 2919365633855274497),
        -INT64_C( 5965639682118155517),  INT64_C( 3878214865514857484),  INT64_C( 8246283826497860258),  INT64_C( 6535662419612140144) },
      {  INT64_C( 3777339205567884907),  INT64_C( 8469260244416938588),  INT64_C( 6717752645020316502), -INT64_C( 2919365633855274497),
        -INT64_C( 5965639682118155517),  INT64_C( 3878214865514857484),  INT64_C( 8246283826497860258),  INT64_C( 6535662419612140144) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),
        -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { {  INT64_C( 5971046963670832228), -INT64_C( 4995479922181974186),  INT64_C( 5431642410717789343),  INT64_C( 4732413347045792627),
         INT64_C( 3756552797703074364), -INT64_C(  252135205688582069), -INT64_C( 3800207572127889416),  INT64_C( 7020012467085131086) },
      {  INT64_C( 5971046963670832228), -INT64_C( 4995479922181974186),  INT64_C( 5431642410717789343),  INT64_C( 4732413347045792627),
         INT64_C( 3756552797703074364), -INT64_C(  252135205688582069), -INT64_C( 3800207572127889416),  INT64_C( 7020012467085131086) },
      {  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0) } },
    { {  INT64_C( 4475009623626417190),  INT64_C( 4510306579502448303),  INT64_C( 3768774669375626660),  INT64_C( 5094382759632799917),
        -INT64_C( 8885487902670362650), -INT64_C( 9128208360234088155), -INT64_C(   17490676429098252),  INT64_C( 5966323292858059365) },
      {  INT64_C( 4475009623626417190),  INT64_C( 4510306579502448303),  INT64_C( 3768774669375626660),  INT64_C( 5094382759632799917),
        -INT64_C( 8885487902670362650), -INT64_C( 9128208360234088155), -INT64_C(   17490676429098252),  INT64_C( 5966323292858059365) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi64(test_vec[i].b);
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_srav_epi64(a, b);
    }
    EASYSIMD_TEST_PERF_END("_mm512_srav_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_srav_epi64(a, b);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;

#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srav_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srav_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srav_epi64)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
