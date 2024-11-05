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

#define EASYSIMD_TEST_X86_AVX512_INSN srai

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/srai.h>

static int
test_easysimd_mm512_srai_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t r0[32];
    const int16_t r3[32];
    const int16_t r7[32];
    const int16_t r13[32];
    const int16_t r24[32];
  } test_vec[] = {
    { { -INT16_C(  2725),  INT16_C(  6711),  INT16_C(  7327),  INT16_C( 11963),  INT16_C( 28148),  INT16_C(  5058),  INT16_C( 21695), -INT16_C( 19668),
        -INT16_C( 11147),  INT16_C( 27930), -INT16_C(  5129), -INT16_C( 26938), -INT16_C( 23608),  INT16_C( 22277),  INT16_C( 10373), -INT16_C(  8091),
        -INT16_C( 25571), -INT16_C( 17158), -INT16_C( 19015), -INT16_C( 21013), -INT16_C( 21214), -INT16_C(  7488), -INT16_C(  5119),  INT16_C( 30357),
        -INT16_C( 20543), -INT16_C( 18205), -INT16_C( 21861),  INT16_C( 25422),  INT16_C( 21325), -INT16_C( 11590),  INT16_C(  8315), -INT16_C( 26446) },
      { -INT16_C(  2725),  INT16_C(  6711),  INT16_C(  7327),  INT16_C( 11963),  INT16_C( 28148),  INT16_C(  5058),  INT16_C( 21695), -INT16_C( 19668),
        -INT16_C( 11147),  INT16_C( 27930), -INT16_C(  5129), -INT16_C( 26938), -INT16_C( 23608),  INT16_C( 22277),  INT16_C( 10373), -INT16_C(  8091),
        -INT16_C( 25571), -INT16_C( 17158), -INT16_C( 19015), -INT16_C( 21013), -INT16_C( 21214), -INT16_C(  7488), -INT16_C(  5119),  INT16_C( 30357),
        -INT16_C( 20543), -INT16_C( 18205), -INT16_C( 21861),  INT16_C( 25422),  INT16_C( 21325), -INT16_C( 11590),  INT16_C(  8315), -INT16_C( 26446) },
      { -INT16_C(   341),  INT16_C(   838),  INT16_C(   915),  INT16_C(  1495),  INT16_C(  3518),  INT16_C(   632),  INT16_C(  2711), -INT16_C(  2459),
        -INT16_C(  1394),  INT16_C(  3491), -INT16_C(   642), -INT16_C(  3368), -INT16_C(  2951),  INT16_C(  2784),  INT16_C(  1296), -INT16_C(  1012),
        -INT16_C(  3197), -INT16_C(  2145), -INT16_C(  2377), -INT16_C(  2627), -INT16_C(  2652), -INT16_C(   936), -INT16_C(   640),  INT16_C(  3794),
        -INT16_C(  2568), -INT16_C(  2276), -INT16_C(  2733),  INT16_C(  3177),  INT16_C(  2665), -INT16_C(  1449),  INT16_C(  1039), -INT16_C(  3306) },
      { -INT16_C(    22),  INT16_C(    52),  INT16_C(    57),  INT16_C(    93),  INT16_C(   219),  INT16_C(    39),  INT16_C(   169), -INT16_C(   154),
        -INT16_C(    88),  INT16_C(   218), -INT16_C(    41), -INT16_C(   211), -INT16_C(   185),  INT16_C(   174),  INT16_C(    81), -INT16_C(    64),
        -INT16_C(   200), -INT16_C(   135), -INT16_C(   149), -INT16_C(   165), -INT16_C(   166), -INT16_C(    59), -INT16_C(    40),  INT16_C(   237),
        -INT16_C(   161), -INT16_C(   143), -INT16_C(   171),  INT16_C(   198),  INT16_C(   166), -INT16_C(    91),  INT16_C(    64), -INT16_C(   207) },
      { -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1),  INT16_C(     3),  INT16_C(     0),  INT16_C(     2), -INT16_C(     3),
        -INT16_C(     2),  INT16_C(     3), -INT16_C(     1), -INT16_C(     4), -INT16_C(     3),  INT16_C(     2),  INT16_C(     1), -INT16_C(     1),
        -INT16_C(     4), -INT16_C(     3), -INT16_C(     3), -INT16_C(     3), -INT16_C(     3), -INT16_C(     1), -INT16_C(     1),  INT16_C(     3),
        -INT16_C(     3), -INT16_C(     3), -INT16_C(     3),  INT16_C(     3),  INT16_C(     2), -INT16_C(     2),  INT16_C(     1), -INT16_C(     4) },
      { -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),
        -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),
        -INT16_C(     1), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),
        -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1) } },
    { { -INT16_C( 21316),  INT16_C( 30036),  INT16_C( 16225), -INT16_C( 31710), -INT16_C(  7444), -INT16_C(  4762), -INT16_C(  1073), -INT16_C( 28572),
         INT16_C( 18347),  INT16_C( 17992), -INT16_C( 26895),  INT16_C( 16041),  INT16_C( 25833),  INT16_C( 25616), -INT16_C( 15740),  INT16_C( 16636),
         INT16_C( 20590), -INT16_C( 12106), -INT16_C( 10096),  INT16_C( 31828), -INT16_C( 17733), -INT16_C( 30102), -INT16_C( 12619),  INT16_C( 24602),
         INT16_C( 25109),  INT16_C(  1958),  INT16_C( 20728), -INT16_C(  7867),  INT16_C( 22196),  INT16_C( 14405),  INT16_C( 16664), -INT16_C( 30856) },
      { -INT16_C( 21316),  INT16_C( 30036),  INT16_C( 16225), -INT16_C( 31710), -INT16_C(  7444), -INT16_C(  4762), -INT16_C(  1073), -INT16_C( 28572),
         INT16_C( 18347),  INT16_C( 17992), -INT16_C( 26895),  INT16_C( 16041),  INT16_C( 25833),  INT16_C( 25616), -INT16_C( 15740),  INT16_C( 16636),
         INT16_C( 20590), -INT16_C( 12106), -INT16_C( 10096),  INT16_C( 31828), -INT16_C( 17733), -INT16_C( 30102), -INT16_C( 12619),  INT16_C( 24602),
         INT16_C( 25109),  INT16_C(  1958),  INT16_C( 20728), -INT16_C(  7867),  INT16_C( 22196),  INT16_C( 14405),  INT16_C( 16664), -INT16_C( 30856) },
      { -INT16_C(  2665),  INT16_C(  3754),  INT16_C(  2028), -INT16_C(  3964), -INT16_C(   931), -INT16_C(   596), -INT16_C(   135), -INT16_C(  3572),
         INT16_C(  2293),  INT16_C(  2249), -INT16_C(  3362),  INT16_C(  2005),  INT16_C(  3229),  INT16_C(  3202), -INT16_C(  1968),  INT16_C(  2079),
         INT16_C(  2573), -INT16_C(  1514), -INT16_C(  1262),  INT16_C(  3978), -INT16_C(  2217), -INT16_C(  3763), -INT16_C(  1578),  INT16_C(  3075),
         INT16_C(  3138),  INT16_C(   244),  INT16_C(  2591), -INT16_C(   984),  INT16_C(  2774),  INT16_C(  1800),  INT16_C(  2083), -INT16_C(  3857) },
      { -INT16_C(   167),  INT16_C(   234),  INT16_C(   126), -INT16_C(   248), -INT16_C(    59), -INT16_C(    38), -INT16_C(     9), -INT16_C(   224),
         INT16_C(   143),  INT16_C(   140), -INT16_C(   211),  INT16_C(   125),  INT16_C(   201),  INT16_C(   200), -INT16_C(   123),  INT16_C(   129),
         INT16_C(   160), -INT16_C(    95), -INT16_C(    79),  INT16_C(   248), -INT16_C(   139), -INT16_C(   236), -INT16_C(    99),  INT16_C(   192),
         INT16_C(   196),  INT16_C(    15),  INT16_C(   161), -INT16_C(    62),  INT16_C(   173),  INT16_C(   112),  INT16_C(   130), -INT16_C(   242) },
      { -INT16_C(     3),  INT16_C(     3),  INT16_C(     1), -INT16_C(     4), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1), -INT16_C(     4),
         INT16_C(     2),  INT16_C(     2), -INT16_C(     4),  INT16_C(     1),  INT16_C(     3),  INT16_C(     3), -INT16_C(     2),  INT16_C(     2),
         INT16_C(     2), -INT16_C(     2), -INT16_C(     2),  INT16_C(     3), -INT16_C(     3), -INT16_C(     4), -INT16_C(     2),  INT16_C(     3),
         INT16_C(     3),  INT16_C(     0),  INT16_C(     2), -INT16_C(     1),  INT16_C(     2),  INT16_C(     1),  INT16_C(     2), -INT16_C(     4) },
      { -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),
         INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),
         INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1) } },
    { {  INT16_C( 11921),  INT16_C(  8535), -INT16_C( 21753), -INT16_C( 15714),  INT16_C(  2149),  INT16_C(  6732),  INT16_C( 26326), -INT16_C(  5253),
         INT16_C(  8648), -INT16_C( 16142),  INT16_C( 14449),  INT16_C(  9633), -INT16_C(  6514), -INT16_C( 22947), -INT16_C( 10713), -INT16_C( 18387),
        -INT16_C( 31740),  INT16_C(  3034),  INT16_C( 30767), -INT16_C( 27443),  INT16_C(  6528),  INT16_C( 22191),  INT16_C( 10879),  INT16_C( 18241),
         INT16_C( 13387), -INT16_C( 17145), -INT16_C( 22420), -INT16_C(  1310),  INT16_C( 16526), -INT16_C( 19040), -INT16_C( 12778),  INT16_C(  6766) },
      {  INT16_C( 11921),  INT16_C(  8535), -INT16_C( 21753), -INT16_C( 15714),  INT16_C(  2149),  INT16_C(  6732),  INT16_C( 26326), -INT16_C(  5253),
         INT16_C(  8648), -INT16_C( 16142),  INT16_C( 14449),  INT16_C(  9633), -INT16_C(  6514), -INT16_C( 22947), -INT16_C( 10713), -INT16_C( 18387),
        -INT16_C( 31740),  INT16_C(  3034),  INT16_C( 30767), -INT16_C( 27443),  INT16_C(  6528),  INT16_C( 22191),  INT16_C( 10879),  INT16_C( 18241),
         INT16_C( 13387), -INT16_C( 17145), -INT16_C( 22420), -INT16_C(  1310),  INT16_C( 16526), -INT16_C( 19040), -INT16_C( 12778),  INT16_C(  6766) },
      {  INT16_C(  1490),  INT16_C(  1066), -INT16_C(  2720), -INT16_C(  1965),  INT16_C(   268),  INT16_C(   841),  INT16_C(  3290), -INT16_C(   657),
         INT16_C(  1081), -INT16_C(  2018),  INT16_C(  1806),  INT16_C(  1204), -INT16_C(   815), -INT16_C(  2869), -INT16_C(  1340), -INT16_C(  2299),
        -INT16_C(  3968),  INT16_C(   379),  INT16_C(  3845), -INT16_C(  3431),  INT16_C(   816),  INT16_C(  2773),  INT16_C(  1359),  INT16_C(  2280),
         INT16_C(  1673), -INT16_C(  2144), -INT16_C(  2803), -INT16_C(   164),  INT16_C(  2065), -INT16_C(  2380), -INT16_C(  1598),  INT16_C(   845) },
      {  INT16_C(    93),  INT16_C(    66), -INT16_C(   170), -INT16_C(   123),  INT16_C(    16),  INT16_C(    52),  INT16_C(   205), -INT16_C(    42),
         INT16_C(    67), -INT16_C(   127),  INT16_C(   112),  INT16_C(    75), -INT16_C(    51), -INT16_C(   180), -INT16_C(    84), -INT16_C(   144),
        -INT16_C(   248),  INT16_C(    23),  INT16_C(   240), -INT16_C(   215),  INT16_C(    51),  INT16_C(   173),  INT16_C(    84),  INT16_C(   142),
         INT16_C(   104), -INT16_C(   134), -INT16_C(   176), -INT16_C(    11),  INT16_C(   129), -INT16_C(   149), -INT16_C(   100),  INT16_C(    52) },
      {  INT16_C(     1),  INT16_C(     1), -INT16_C(     3), -INT16_C(     2),  INT16_C(     0),  INT16_C(     0),  INT16_C(     3), -INT16_C(     1),
         INT16_C(     1), -INT16_C(     2),  INT16_C(     1),  INT16_C(     1), -INT16_C(     1), -INT16_C(     3), -INT16_C(     2), -INT16_C(     3),
        -INT16_C(     4),  INT16_C(     0),  INT16_C(     3), -INT16_C(     4),  INT16_C(     0),  INT16_C(     2),  INT16_C(     1),  INT16_C(     2),
         INT16_C(     1), -INT16_C(     3), -INT16_C(     3), -INT16_C(     1),  INT16_C(     2), -INT16_C(     3), -INT16_C(     2),  INT16_C(     0) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),
         INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),
        -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0) } },
    { {  INT16_C( 18514), -INT16_C( 32218), -INT16_C(  3136),  INT16_C( 16406), -INT16_C( 15091), -INT16_C( 29546), -INT16_C( 10257),  INT16_C( 15316),
        -INT16_C(  9461),  INT16_C( 30712), -INT16_C(  9596),  INT16_C(  4721),  INT16_C(  4634),  INT16_C( 12488),  INT16_C( 14048),  INT16_C( 12875),
         INT16_C( 29054),  INT16_C( 16052), -INT16_C( 13468),  INT16_C( 29054),  INT16_C(  5264), -INT16_C( 32514), -INT16_C( 11541), -INT16_C(  2117),
        -INT16_C( 19539),  INT16_C( 12654), -INT16_C(  8051), -INT16_C( 22460),  INT16_C(  3314), -INT16_C( 11560),  INT16_C(  9026), -INT16_C( 16380) },
      {  INT16_C( 18514), -INT16_C( 32218), -INT16_C(  3136),  INT16_C( 16406), -INT16_C( 15091), -INT16_C( 29546), -INT16_C( 10257),  INT16_C( 15316),
        -INT16_C(  9461),  INT16_C( 30712), -INT16_C(  9596),  INT16_C(  4721),  INT16_C(  4634),  INT16_C( 12488),  INT16_C( 14048),  INT16_C( 12875),
         INT16_C( 29054),  INT16_C( 16052), -INT16_C( 13468),  INT16_C( 29054),  INT16_C(  5264), -INT16_C( 32514), -INT16_C( 11541), -INT16_C(  2117),
        -INT16_C( 19539),  INT16_C( 12654), -INT16_C(  8051), -INT16_C( 22460),  INT16_C(  3314), -INT16_C( 11560),  INT16_C(  9026), -INT16_C( 16380) },
      {  INT16_C(  2314), -INT16_C(  4028), -INT16_C(   392),  INT16_C(  2050), -INT16_C(  1887), -INT16_C(  3694), -INT16_C(  1283),  INT16_C(  1914),
        -INT16_C(  1183),  INT16_C(  3839), -INT16_C(  1200),  INT16_C(   590),  INT16_C(   579),  INT16_C(  1561),  INT16_C(  1756),  INT16_C(  1609),
         INT16_C(  3631),  INT16_C(  2006), -INT16_C(  1684),  INT16_C(  3631),  INT16_C(   658), -INT16_C(  4065), -INT16_C(  1443), -INT16_C(   265),
        -INT16_C(  2443),  INT16_C(  1581), -INT16_C(  1007), -INT16_C(  2808),  INT16_C(   414), -INT16_C(  1445),  INT16_C(  1128), -INT16_C(  2048) },
      {  INT16_C(   144), -INT16_C(   252), -INT16_C(    25),  INT16_C(   128), -INT16_C(   118), -INT16_C(   231), -INT16_C(    81),  INT16_C(   119),
        -INT16_C(    74),  INT16_C(   239), -INT16_C(    75),  INT16_C(    36),  INT16_C(    36),  INT16_C(    97),  INT16_C(   109),  INT16_C(   100),
         INT16_C(   226),  INT16_C(   125), -INT16_C(   106),  INT16_C(   226),  INT16_C(    41), -INT16_C(   255), -INT16_C(    91), -INT16_C(    17),
        -INT16_C(   153),  INT16_C(    98), -INT16_C(    63), -INT16_C(   176),  INT16_C(    25), -INT16_C(    91),  INT16_C(    70), -INT16_C(   128) },
      {  INT16_C(     2), -INT16_C(     4), -INT16_C(     1),  INT16_C(     2), -INT16_C(     2), -INT16_C(     4), -INT16_C(     2),  INT16_C(     1),
        -INT16_C(     2),  INT16_C(     3), -INT16_C(     2),  INT16_C(     0),  INT16_C(     0),  INT16_C(     1),  INT16_C(     1),  INT16_C(     1),
         INT16_C(     3),  INT16_C(     1), -INT16_C(     2),  INT16_C(     3),  INT16_C(     0), -INT16_C(     4), -INT16_C(     2), -INT16_C(     1),
        -INT16_C(     3),  INT16_C(     1), -INT16_C(     1), -INT16_C(     3),  INT16_C(     0), -INT16_C(     2),  INT16_C(     1), -INT16_C(     2) },
      {  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),
        -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),
        -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1) } },
    { { -INT16_C( 18028), -INT16_C(  1538),  INT16_C( 31876),  INT16_C(  5226),  INT16_C( 26768),  INT16_C( 31636),  INT16_C( 20282), -INT16_C(  6030),
        -INT16_C(  7934), -INT16_C( 28647),  INT16_C( 24001), -INT16_C( 19656),  INT16_C(  4201), -INT16_C( 21627), -INT16_C( 30412), -INT16_C( 14229),
         INT16_C( 26946), -INT16_C( 14655),  INT16_C( 11493),  INT16_C( 30171),  INT16_C( 28564), -INT16_C( 12303),  INT16_C( 25535), -INT16_C( 15945),
        -INT16_C( 12220),  INT16_C(  1361), -INT16_C( 30418), -INT16_C( 26696),  INT16_C( 15770), -INT16_C( 12733), -INT16_C( 20793),  INT16_C(  2454) },
      { -INT16_C( 18028), -INT16_C(  1538),  INT16_C( 31876),  INT16_C(  5226),  INT16_C( 26768),  INT16_C( 31636),  INT16_C( 20282), -INT16_C(  6030),
        -INT16_C(  7934), -INT16_C( 28647),  INT16_C( 24001), -INT16_C( 19656),  INT16_C(  4201), -INT16_C( 21627), -INT16_C( 30412), -INT16_C( 14229),
         INT16_C( 26946), -INT16_C( 14655),  INT16_C( 11493),  INT16_C( 30171),  INT16_C( 28564), -INT16_C( 12303),  INT16_C( 25535), -INT16_C( 15945),
        -INT16_C( 12220),  INT16_C(  1361), -INT16_C( 30418), -INT16_C( 26696),  INT16_C( 15770), -INT16_C( 12733), -INT16_C( 20793),  INT16_C(  2454) },
      { -INT16_C(  2254), -INT16_C(   193),  INT16_C(  3984),  INT16_C(   653),  INT16_C(  3346),  INT16_C(  3954),  INT16_C(  2535), -INT16_C(   754),
        -INT16_C(   992), -INT16_C(  3581),  INT16_C(  3000), -INT16_C(  2457),  INT16_C(   525), -INT16_C(  2704), -INT16_C(  3802), -INT16_C(  1779),
         INT16_C(  3368), -INT16_C(  1832),  INT16_C(  1436),  INT16_C(  3771),  INT16_C(  3570), -INT16_C(  1538),  INT16_C(  3191), -INT16_C(  1994),
        -INT16_C(  1528),  INT16_C(   170), -INT16_C(  3803), -INT16_C(  3337),  INT16_C(  1971), -INT16_C(  1592), -INT16_C(  2600),  INT16_C(   306) },
      { -INT16_C(   141), -INT16_C(    13),  INT16_C(   249),  INT16_C(    40),  INT16_C(   209),  INT16_C(   247),  INT16_C(   158), -INT16_C(    48),
        -INT16_C(    62), -INT16_C(   224),  INT16_C(   187), -INT16_C(   154),  INT16_C(    32), -INT16_C(   169), -INT16_C(   238), -INT16_C(   112),
         INT16_C(   210), -INT16_C(   115),  INT16_C(    89),  INT16_C(   235),  INT16_C(   223), -INT16_C(    97),  INT16_C(   199), -INT16_C(   125),
        -INT16_C(    96),  INT16_C(    10), -INT16_C(   238), -INT16_C(   209),  INT16_C(   123), -INT16_C(   100), -INT16_C(   163),  INT16_C(    19) },
      { -INT16_C(     3), -INT16_C(     1),  INT16_C(     3),  INT16_C(     0),  INT16_C(     3),  INT16_C(     3),  INT16_C(     2), -INT16_C(     1),
        -INT16_C(     1), -INT16_C(     4),  INT16_C(     2), -INT16_C(     3),  INT16_C(     0), -INT16_C(     3), -INT16_C(     4), -INT16_C(     2),
         INT16_C(     3), -INT16_C(     2),  INT16_C(     1),  INT16_C(     3),  INT16_C(     3), -INT16_C(     2),  INT16_C(     3), -INT16_C(     2),
        -INT16_C(     2),  INT16_C(     0), -INT16_C(     4), -INT16_C(     4),  INT16_C(     1), -INT16_C(     2), -INT16_C(     3),  INT16_C(     0) },
      { -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),
        -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),
         INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),
        -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0) } },
    { {  INT16_C( 22552), -INT16_C(   560), -INT16_C( 21628),  INT16_C(  6259),  INT16_C( 25626), -INT16_C(  9753), -INT16_C( 24889),  INT16_C(  3227),
        -INT16_C(  5009), -INT16_C( 25327), -INT16_C( 13706),  INT16_C(  4148),  INT16_C( 30471), -INT16_C( 12578),  INT16_C( 29734),  INT16_C( 16088),
        -INT16_C( 22324),  INT16_C( 20539), -INT16_C( 20909),  INT16_C( 28009),  INT16_C( 20498), -INT16_C(  9657), -INT16_C(  7441),  INT16_C( 24294),
        -INT16_C(  2098),  INT16_C( 17659),  INT16_C( 12225), -INT16_C( 13996),  INT16_C( 12967), -INT16_C( 12905),  INT16_C( 28583),  INT16_C( 29451) },
      {  INT16_C( 22552), -INT16_C(   560), -INT16_C( 21628),  INT16_C(  6259),  INT16_C( 25626), -INT16_C(  9753), -INT16_C( 24889),  INT16_C(  3227),
        -INT16_C(  5009), -INT16_C( 25327), -INT16_C( 13706),  INT16_C(  4148),  INT16_C( 30471), -INT16_C( 12578),  INT16_C( 29734),  INT16_C( 16088),
        -INT16_C( 22324),  INT16_C( 20539), -INT16_C( 20909),  INT16_C( 28009),  INT16_C( 20498), -INT16_C(  9657), -INT16_C(  7441),  INT16_C( 24294),
        -INT16_C(  2098),  INT16_C( 17659),  INT16_C( 12225), -INT16_C( 13996),  INT16_C( 12967), -INT16_C( 12905),  INT16_C( 28583),  INT16_C( 29451) },
      {  INT16_C(  2819), -INT16_C(    70), -INT16_C(  2704),  INT16_C(   782),  INT16_C(  3203), -INT16_C(  1220), -INT16_C(  3112),  INT16_C(   403),
        -INT16_C(   627), -INT16_C(  3166), -INT16_C(  1714),  INT16_C(   518),  INT16_C(  3808), -INT16_C(  1573),  INT16_C(  3716),  INT16_C(  2011),
        -INT16_C(  2791),  INT16_C(  2567), -INT16_C(  2614),  INT16_C(  3501),  INT16_C(  2562), -INT16_C(  1208), -INT16_C(   931),  INT16_C(  3036),
        -INT16_C(   263),  INT16_C(  2207),  INT16_C(  1528), -INT16_C(  1750),  INT16_C(  1620), -INT16_C(  1614),  INT16_C(  3572),  INT16_C(  3681) },
      {  INT16_C(   176), -INT16_C(     5), -INT16_C(   169),  INT16_C(    48),  INT16_C(   200), -INT16_C(    77), -INT16_C(   195),  INT16_C(    25),
        -INT16_C(    40), -INT16_C(   198), -INT16_C(   108),  INT16_C(    32),  INT16_C(   238), -INT16_C(    99),  INT16_C(   232),  INT16_C(   125),
        -INT16_C(   175),  INT16_C(   160), -INT16_C(   164),  INT16_C(   218),  INT16_C(   160), -INT16_C(    76), -INT16_C(    59),  INT16_C(   189),
        -INT16_C(    17),  INT16_C(   137),  INT16_C(    95), -INT16_C(   110),  INT16_C(   101), -INT16_C(   101),  INT16_C(   223),  INT16_C(   230) },
      {  INT16_C(     2), -INT16_C(     1), -INT16_C(     3),  INT16_C(     0),  INT16_C(     3), -INT16_C(     2), -INT16_C(     4),  INT16_C(     0),
        -INT16_C(     1), -INT16_C(     4), -INT16_C(     2),  INT16_C(     0),  INT16_C(     3), -INT16_C(     2),  INT16_C(     3),  INT16_C(     1),
        -INT16_C(     3),  INT16_C(     2), -INT16_C(     3),  INT16_C(     3),  INT16_C(     2), -INT16_C(     2), -INT16_C(     1),  INT16_C(     2),
        -INT16_C(     1),  INT16_C(     2),  INT16_C(     1), -INT16_C(     2),  INT16_C(     1), -INT16_C(     2),  INT16_C(     3),  INT16_C(     3) },
      {  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),
        -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),
        -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),
        -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0) } },
    { {  INT16_C( 17943),  INT16_C( 27332),  INT16_C( 11765),  INT16_C(  2008),  INT16_C(  8061),  INT16_C( 27873), -INT16_C( 14591), -INT16_C( 12342),
        -INT16_C( 14913), -INT16_C( 32748),  INT16_C( 26869), -INT16_C( 25527), -INT16_C(  7781),  INT16_C( 17001),  INT16_C( 29776),  INT16_C( 26805),
         INT16_C( 31162), -INT16_C( 20526), -INT16_C( 21850),  INT16_C(  9399), -INT16_C( 26423), -INT16_C( 13680),  INT16_C( 23392),  INT16_C(  8090),
        -INT16_C( 20960),  INT16_C(  5535), -INT16_C(  5866), -INT16_C( 20047),  INT16_C(  6858),  INT16_C(  6899), -INT16_C( 22130),  INT16_C( 18818) },
      {  INT16_C( 17943),  INT16_C( 27332),  INT16_C( 11765),  INT16_C(  2008),  INT16_C(  8061),  INT16_C( 27873), -INT16_C( 14591), -INT16_C( 12342),
        -INT16_C( 14913), -INT16_C( 32748),  INT16_C( 26869), -INT16_C( 25527), -INT16_C(  7781),  INT16_C( 17001),  INT16_C( 29776),  INT16_C( 26805),
         INT16_C( 31162), -INT16_C( 20526), -INT16_C( 21850),  INT16_C(  9399), -INT16_C( 26423), -INT16_C( 13680),  INT16_C( 23392),  INT16_C(  8090),
        -INT16_C( 20960),  INT16_C(  5535), -INT16_C(  5866), -INT16_C( 20047),  INT16_C(  6858),  INT16_C(  6899), -INT16_C( 22130),  INT16_C( 18818) },
      {  INT16_C(  2242),  INT16_C(  3416),  INT16_C(  1470),  INT16_C(   251),  INT16_C(  1007),  INT16_C(  3484), -INT16_C(  1824), -INT16_C(  1543),
        -INT16_C(  1865), -INT16_C(  4094),  INT16_C(  3358), -INT16_C(  3191), -INT16_C(   973),  INT16_C(  2125),  INT16_C(  3722),  INT16_C(  3350),
         INT16_C(  3895), -INT16_C(  2566), -INT16_C(  2732),  INT16_C(  1174), -INT16_C(  3303), -INT16_C(  1710),  INT16_C(  2924),  INT16_C(  1011),
        -INT16_C(  2620),  INT16_C(   691), -INT16_C(   734), -INT16_C(  2506),  INT16_C(   857),  INT16_C(   862), -INT16_C(  2767),  INT16_C(  2352) },
      {  INT16_C(   140),  INT16_C(   213),  INT16_C(    91),  INT16_C(    15),  INT16_C(    62),  INT16_C(   217), -INT16_C(   114), -INT16_C(    97),
        -INT16_C(   117), -INT16_C(   256),  INT16_C(   209), -INT16_C(   200), -INT16_C(    61),  INT16_C(   132),  INT16_C(   232),  INT16_C(   209),
         INT16_C(   243), -INT16_C(   161), -INT16_C(   171),  INT16_C(    73), -INT16_C(   207), -INT16_C(   107),  INT16_C(   182),  INT16_C(    63),
        -INT16_C(   164),  INT16_C(    43), -INT16_C(    46), -INT16_C(   157),  INT16_C(    53),  INT16_C(    53), -INT16_C(   173),  INT16_C(   147) },
      {  INT16_C(     2),  INT16_C(     3),  INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     3), -INT16_C(     2), -INT16_C(     2),
        -INT16_C(     2), -INT16_C(     4),  INT16_C(     3), -INT16_C(     4), -INT16_C(     1),  INT16_C(     2),  INT16_C(     3),  INT16_C(     3),
         INT16_C(     3), -INT16_C(     3), -INT16_C(     3),  INT16_C(     1), -INT16_C(     4), -INT16_C(     2),  INT16_C(     2),  INT16_C(     0),
        -INT16_C(     3),  INT16_C(     0), -INT16_C(     1), -INT16_C(     3),  INT16_C(     0),  INT16_C(     0), -INT16_C(     3),  INT16_C(     2) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),
        -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),
        -INT16_C(     1),  INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0) } },
    { {  INT16_C( 21794), -INT16_C( 13832), -INT16_C( 20481), -INT16_C( 13843),  INT16_C( 32072), -INT16_C( 22381),  INT16_C( 11736), -INT16_C(  1593),
         INT16_C( 26331), -INT16_C(  3570), -INT16_C( 16305),  INT16_C(  6563), -INT16_C( 26662),  INT16_C( 26932), -INT16_C( 18880),  INT16_C( 25266),
        -INT16_C( 22005),  INT16_C(  2859),  INT16_C(  6234), -INT16_C( 23852),  INT16_C( 26518),  INT16_C( 28234),  INT16_C(  4501),  INT16_C( 28775),
         INT16_C( 30327), -INT16_C( 14494),  INT16_C(  1590),  INT16_C(  4320),  INT16_C(  5277), -INT16_C(  8839),  INT16_C( 11211), -INT16_C( 10689) },
      {  INT16_C( 21794), -INT16_C( 13832), -INT16_C( 20481), -INT16_C( 13843),  INT16_C( 32072), -INT16_C( 22381),  INT16_C( 11736), -INT16_C(  1593),
         INT16_C( 26331), -INT16_C(  3570), -INT16_C( 16305),  INT16_C(  6563), -INT16_C( 26662),  INT16_C( 26932), -INT16_C( 18880),  INT16_C( 25266),
        -INT16_C( 22005),  INT16_C(  2859),  INT16_C(  6234), -INT16_C( 23852),  INT16_C( 26518),  INT16_C( 28234),  INT16_C(  4501),  INT16_C( 28775),
         INT16_C( 30327), -INT16_C( 14494),  INT16_C(  1590),  INT16_C(  4320),  INT16_C(  5277), -INT16_C(  8839),  INT16_C( 11211), -INT16_C( 10689) },
      {  INT16_C(  2724), -INT16_C(  1729), -INT16_C(  2561), -INT16_C(  1731),  INT16_C(  4009), -INT16_C(  2798),  INT16_C(  1467), -INT16_C(   200),
         INT16_C(  3291), -INT16_C(   447), -INT16_C(  2039),  INT16_C(   820), -INT16_C(  3333),  INT16_C(  3366), -INT16_C(  2360),  INT16_C(  3158),
        -INT16_C(  2751),  INT16_C(   357),  INT16_C(   779), -INT16_C(  2982),  INT16_C(  3314),  INT16_C(  3529),  INT16_C(   562),  INT16_C(  3596),
         INT16_C(  3790), -INT16_C(  1812),  INT16_C(   198),  INT16_C(   540),  INT16_C(   659), -INT16_C(  1105),  INT16_C(  1401), -INT16_C(  1337) },
      {  INT16_C(   170), -INT16_C(   109), -INT16_C(   161), -INT16_C(   109),  INT16_C(   250), -INT16_C(   175),  INT16_C(    91), -INT16_C(    13),
         INT16_C(   205), -INT16_C(    28), -INT16_C(   128),  INT16_C(    51), -INT16_C(   209),  INT16_C(   210), -INT16_C(   148),  INT16_C(   197),
        -INT16_C(   172),  INT16_C(    22),  INT16_C(    48), -INT16_C(   187),  INT16_C(   207),  INT16_C(   220),  INT16_C(    35),  INT16_C(   224),
         INT16_C(   236), -INT16_C(   114),  INT16_C(    12),  INT16_C(    33),  INT16_C(    41), -INT16_C(    70),  INT16_C(    87), -INT16_C(    84) },
      {  INT16_C(     2), -INT16_C(     2), -INT16_C(     3), -INT16_C(     2),  INT16_C(     3), -INT16_C(     3),  INT16_C(     1), -INT16_C(     1),
         INT16_C(     3), -INT16_C(     1), -INT16_C(     2),  INT16_C(     0), -INT16_C(     4),  INT16_C(     3), -INT16_C(     3),  INT16_C(     3),
        -INT16_C(     3),  INT16_C(     0),  INT16_C(     0), -INT16_C(     3),  INT16_C(     3),  INT16_C(     3),  INT16_C(     0),  INT16_C(     3),
         INT16_C(     3), -INT16_C(     2),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     2),  INT16_C(     1), -INT16_C(     2) },
      {  INT16_C(     0), -INT16_C(     1), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),
         INT16_C(     0), -INT16_C(     1), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),
        -INT16_C(     1),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C(     1),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(     1),  INT16_C(     0), -INT16_C(     1) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    easysimd__m512i r0 = easysimd_mm512_srai_epi16(a, 0);
    easysimd__m512i r3;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r3 = easysimd_mm512_srai_epi16(a, 3);
    }
    EASYSIMD_TEST_PERF_END("_mm512_srai_epi16");
    easysimd__m512i r7 = easysimd_mm512_srai_epi16(a, 7);
    easysimd__m512i r13 = easysimd_mm512_srai_epi16(a, 13);
    easysimd__m512i r24 = easysimd_mm512_srai_epi16(a, 24);
    easysimd_test_x86_assert_equal_i16x32(r0, easysimd_mm512_loadu_epi16(test_vec[i].r0));
    easysimd_test_x86_assert_equal_i16x32(r3, easysimd_mm512_loadu_epi16(test_vec[i].r3));
    easysimd_test_x86_assert_equal_i16x32(r7, easysimd_mm512_loadu_epi16(test_vec[i].r7));
    easysimd_test_x86_assert_equal_i16x32(r13, easysimd_mm512_loadu_epi16(test_vec[i].r13));
    easysimd_test_x86_assert_equal_i16x32(r24, easysimd_mm512_loadu_epi16(test_vec[i].r24));
  }

  return 0;
}

static int
test_easysimd_mm512_srai_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t a[16];
    const int32_t r0[16];
    const int32_t r7[16];
    const int32_t r31[16];
    const int32_t r32[16];
  } test_vec[] = {
    { { -INT32_C(  1063070456), -INT32_C(  1817801635),  INT32_C(  1051256723), -INT32_C(  1073112139), -INT32_C(   900778477),  INT32_C(  1663137146), -INT32_C(  1674899410), -INT32_C(  1773488243),
        -INT32_C(   380179316),  INT32_C(   209583224), -INT32_C(  1723193884), -INT32_C(   732343359),  INT32_C(    60729481),  INT32_C(  1281802014),  INT32_C(  1575523024), -INT32_C(   705482167) },
      { -INT32_C(  1063070456), -INT32_C(  1817801635),  INT32_C(  1051256723), -INT32_C(  1073112139), -INT32_C(   900778477),  INT32_C(  1663137146), -INT32_C(  1674899410), -INT32_C(  1773488243),
        -INT32_C(   380179316),  INT32_C(   209583224), -INT32_C(  1723193884), -INT32_C(   732343359),  INT32_C(    60729481),  INT32_C(  1281802014),  INT32_C(  1575523024), -INT32_C(   705482167) },
      { -INT32_C(     8305238), -INT32_C(    14201576),  INT32_C(     8212943), -INT32_C(     8383689), -INT32_C(     7037332),  INT32_C(    12993258), -INT32_C(    13085152), -INT32_C(    13855377),
        -INT32_C(     2970151),  INT32_C(     1637368), -INT32_C(    13462453), -INT32_C(     5721433),  INT32_C(      474449),  INT32_C(    10014078),  INT32_C(    12308773), -INT32_C(     5511580) },
      { -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),
        -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1) },
      { -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),
        -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1) } },
    { { -INT32_C(  1749071585),  INT32_C(   715340870),  INT32_C(   583265889), -INT32_C(   873063359), -INT32_C(   489777980),  INT32_C(   607008084),  INT32_C(   276895687),  INT32_C(  1759868233),
         INT32_C(    67151038),  INT32_C(  1110352864), -INT32_C(   748359279), -INT32_C(   761373939),  INT32_C(  1135897839),  INT32_C(  1751638945),  INT32_C(  1131997690),  INT32_C(   481058398) },
      { -INT32_C(  1749071585),  INT32_C(   715340870),  INT32_C(   583265889), -INT32_C(   873063359), -INT32_C(   489777980),  INT32_C(   607008084),  INT32_C(   276895687),  INT32_C(  1759868233),
         INT32_C(    67151038),  INT32_C(  1110352864), -INT32_C(   748359279), -INT32_C(   761373939),  INT32_C(  1135897839),  INT32_C(  1751638945),  INT32_C(  1131997690),  INT32_C(   481058398) },
      { -INT32_C(    13664622),  INT32_C(     5588600),  INT32_C(     4556764), -INT32_C(     6820808), -INT32_C(     3826391),  INT32_C(     4742250),  INT32_C(     2163247),  INT32_C(    13748970),
         INT32_C(      524617),  INT32_C(     8674631), -INT32_C(     5846557), -INT32_C(     5948234),  INT32_C(     8874201),  INT32_C(    13684679),  INT32_C(     8843731),  INT32_C(     3758268) },
      { -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      { -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(   484332542), -INT32_C(   517648561),  INT32_C(  1320454465), -INT32_C(   736079132),  INT32_C(  1612174782), -INT32_C(  1295482952), -INT32_C(   957005464), -INT32_C(  1578983009),
        -INT32_C(  1652292787), -INT32_C(  1803638445),  INT32_C(   400699955),  INT32_C(  1122698116), -INT32_C(  1868430376), -INT32_C(   364745854),  INT32_C(  1269839788),  INT32_C(   653038297) },
      { -INT32_C(   484332542), -INT32_C(   517648561),  INT32_C(  1320454465), -INT32_C(   736079132),  INT32_C(  1612174782), -INT32_C(  1295482952), -INT32_C(   957005464), -INT32_C(  1578983009),
        -INT32_C(  1652292787), -INT32_C(  1803638445),  INT32_C(   400699955),  INT32_C(  1122698116), -INT32_C(  1868430376), -INT32_C(   364745854),  INT32_C(  1269839788),  INT32_C(   653038297) },
      { -INT32_C(     3783848), -INT32_C(     4044130),  INT32_C(    10316050), -INT32_C(     5750619),  INT32_C(    12595115), -INT32_C(    10120961), -INT32_C(     7476606), -INT32_C(    12335805),
        -INT32_C(    12908538), -INT32_C(    14090926),  INT32_C(     3130468),  INT32_C(     8771079), -INT32_C(    14597113), -INT32_C(     2849577),  INT32_C(     9920623),  INT32_C(     5101861) },
      { -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),
        -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0) },
      { -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),
        -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(   373067370),  INT32_C(  1300054298), -INT32_C(   144351373),  INT32_C(   976900194), -INT32_C(   708125613), -INT32_C(   205583289),  INT32_C(   490630980), -INT32_C(  1740428542),
        -INT32_C(  1233057892), -INT32_C(  1140523448), -INT32_C(  1061983907),  INT32_C(   234548665),  INT32_C(   300074442),  INT32_C(   352690897),  INT32_C(   322061073),  INT32_C(   179009134) },
      { -INT32_C(   373067370),  INT32_C(  1300054298), -INT32_C(   144351373),  INT32_C(   976900194), -INT32_C(   708125613), -INT32_C(   205583289),  INT32_C(   490630980), -INT32_C(  1740428542),
        -INT32_C(  1233057892), -INT32_C(  1140523448), -INT32_C(  1061983907),  INT32_C(   234548665),  INT32_C(   300074442),  INT32_C(   352690897),  INT32_C(   322061073),  INT32_C(   179009134) },
      { -INT32_C(     2914589),  INT32_C(    10156674), -INT32_C(     1127746),  INT32_C(     7632032), -INT32_C(     5532232), -INT32_C(     1606120),  INT32_C(     3833054), -INT32_C(    13597098),
        -INT32_C(     9633265), -INT32_C(     8910340), -INT32_C(     8296750),  INT32_C(     1832411),  INT32_C(     2344331),  INT32_C(     2755397),  INT32_C(     2516102),  INT32_C(     1398508) },
      { -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),
        -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) },
      { -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),
        -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(   977195907), -INT32_C(  2004761302), -INT32_C(   414698194), -INT32_C(   319536606), -INT32_C(   637610233), -INT32_C(  1964113031), -INT32_C(  1247927994),  INT32_C(   348080535),
        -INT32_C(  1596358538),  INT32_C(  1932024645), -INT32_C(  1302630256), -INT32_C(  1163964493), -INT32_C(  1617715930), -INT32_C(   433421664),  INT32_C(   983287971), -INT32_C(  2024908015) },
      { -INT32_C(   977195907), -INT32_C(  2004761302), -INT32_C(   414698194), -INT32_C(   319536606), -INT32_C(   637610233), -INT32_C(  1964113031), -INT32_C(  1247927994),  INT32_C(   348080535),
        -INT32_C(  1596358538),  INT32_C(  1932024645), -INT32_C(  1302630256), -INT32_C(  1163964493), -INT32_C(  1617715930), -INT32_C(   433421664),  INT32_C(   983287971), -INT32_C(  2024908015) },
      { -INT32_C(     7634344), -INT32_C(    15662198), -INT32_C(     3239830), -INT32_C(     2496380), -INT32_C(     4981330), -INT32_C(    15344634), -INT32_C(     9749438),  INT32_C(     2719379),
        -INT32_C(    12471552),  INT32_C(    15093942), -INT32_C(    10176799), -INT32_C(     9093473), -INT32_C(    12638406), -INT32_C(     3386107),  INT32_C(     7681937), -INT32_C(    15819594) },
      { -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),
        -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1) },
      { -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),
        -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1) } },
    { {  INT32_C(   556279771),  INT32_C(   311709826),  INT32_C(  1959129025),  INT32_C(  1697539135), -INT32_C(  1593458175), -INT32_C(   410570940),  INT32_C(   136389623),  INT32_C(  1502637950),
         INT32_C(   427473046), -INT32_C(   919924984),  INT32_C(  1027469566),  INT32_C(  1436773460),  INT32_C(  1928767534), -INT32_C(   832995625),  INT32_C(   534149793),  INT32_C(  2138597097) },
      {  INT32_C(   556279771),  INT32_C(   311709826),  INT32_C(  1959129025),  INT32_C(  1697539135), -INT32_C(  1593458175), -INT32_C(   410570940),  INT32_C(   136389623),  INT32_C(  1502637950),
         INT32_C(   427473046), -INT32_C(   919924984),  INT32_C(  1027469566),  INT32_C(  1436773460),  INT32_C(  1928767534), -INT32_C(   832995625),  INT32_C(   534149793),  INT32_C(  2138597097) },
      {  INT32_C(     4345935),  INT32_C(     2435233),  INT32_C(    15305695),  INT32_C(    13262024), -INT32_C(    12448892), -INT32_C(     3207586),  INT32_C(     1065543),  INT32_C(    11739358),
         INT32_C(     3339633), -INT32_C(     7186914),  INT32_C(     8027105),  INT32_C(    11224792),  INT32_C(    15068496), -INT32_C(     6507779),  INT32_C(     4173045),  INT32_C(    16707789) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0) } },
    { {  INT32_C(   664335134),  INT32_C(    15778818),  INT32_C(   155070132), -INT32_C(   933305958),  INT32_C(  1614435721),  INT32_C(  1949209555), -INT32_C(   158137331), -INT32_C(  1988818069),
         INT32_C(    11538174), -INT32_C(  2046713390),  INT32_C(  1770995663), -INT32_C(  1456345568),  INT32_C(   369716035),  INT32_C(   193607678), -INT32_C(  1509876421),  INT32_C(   657487400) },
      {  INT32_C(   664335134),  INT32_C(    15778818),  INT32_C(   155070132), -INT32_C(   933305958),  INT32_C(  1614435721),  INT32_C(  1949209555), -INT32_C(   158137331), -INT32_C(  1988818069),
         INT32_C(    11538174), -INT32_C(  2046713390),  INT32_C(  1770995663), -INT32_C(  1456345568),  INT32_C(   369716035),  INT32_C(   193607678), -INT32_C(  1509876421),  INT32_C(   657487400) },
      {  INT32_C(     5190118),  INT32_C(      123272),  INT32_C(     1211485), -INT32_C(     7291453),  INT32_C(    12612779),  INT32_C(    15228199), -INT32_C(     1235448), -INT32_C(    15537642),
         INT32_C(       90141), -INT32_C(    15989949),  INT32_C(    13835903), -INT32_C(    11377700),  INT32_C(     2888406),  INT32_C(     1512559), -INT32_C(    11795910),  INT32_C(     5136620) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),
         INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),
         INT32_C(           0), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0) } },
    { {  INT32_C(  1445453956),  INT32_C(  1356671105), -INT32_C(  2017891225), -INT32_C(  1657738662),  INT32_C(  1404254549), -INT32_C(  1403110032), -INT32_C(  2091753638),  INT32_C(  1521124054),
        -INT32_C(   458108573),  INT32_C(  1630899962),  INT32_C(  1441394426),  INT32_C(   787618265), -INT32_C(  1014847917),  INT32_C(  1047519459),  INT32_C(   381796928), -INT32_C(  1485804732) },
      {  INT32_C(  1445453956),  INT32_C(  1356671105), -INT32_C(  2017891225), -INT32_C(  1657738662),  INT32_C(  1404254549), -INT32_C(  1403110032), -INT32_C(  2091753638),  INT32_C(  1521124054),
        -INT32_C(   458108573),  INT32_C(  1630899962),  INT32_C(  1441394426),  INT32_C(   787618265), -INT32_C(  1014847917),  INT32_C(  1047519459),  INT32_C(   381796928), -INT32_C(  1485804732) },
      {  INT32_C(    11292609),  INT32_C(    10598993), -INT32_C(    15764776), -INT32_C(    12951084),  INT32_C(    10970738), -INT32_C(    10961798), -INT32_C(    16341826),  INT32_C(    11883781),
        -INT32_C(     3578974),  INT32_C(    12741405),  INT32_C(    11260893),  INT32_C(     6153267), -INT32_C(     7928500),  INT32_C(     8183745),  INT32_C(     2982788), -INT32_C(    11607850) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),
        -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0), -INT32_C(           1), -INT32_C(           1),  INT32_C(           0),
        -INT32_C(           1),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1),  INT32_C(           0),  INT32_C(           0), -INT32_C(           1) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i r0 = easysimd_mm512_srai_epi32(a, 0);
    easysimd__m512i r7;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r7 = easysimd_mm512_srai_epi32(a, 7);
    }
    EASYSIMD_TEST_PERF_END("_mm512_srai_epi32");
    easysimd__m512i r31 = easysimd_mm512_srai_epi32(a, 31);
    easysimd__m512i r32 = easysimd_mm512_srai_epi32(a, 32);
    easysimd_test_x86_assert_equal_i32x16(r0, easysimd_mm512_loadu_epi32(test_vec[i].r0));
    easysimd_test_x86_assert_equal_i32x16(r7, easysimd_mm512_loadu_epi32(test_vec[i].r7));
    easysimd_test_x86_assert_equal_i32x16(r31, easysimd_mm512_loadu_epi32(test_vec[i].r31));
    easysimd_test_x86_assert_equal_i32x16(r32, easysimd_mm512_loadu_epi32(test_vec[i].r32));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i r0 = easysimd_mm512_srai_epi32(a, 0);
    easysimd__m512i r7 = easysimd_mm512_srai_epi32(a, 7);
    easysimd__m512i r31 = easysimd_mm512_srai_epi32(a, 31);
    easysimd__m512i r32 = easysimd_mm512_srai_epi32(a, 32);

    easysimd_test_x86_write_i32x16(2, a,   EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, r0,  EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r7,  EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r31, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r32, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

static int
test_easysimd_mm512_srai_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t r0[8];
    const int64_t r7[8];
    const int64_t r63[8];
    const int64_t r64[8];
  } test_vec[] = {
    { {  INT64_C( 1837740447273516598), -INT64_C( 9032200989554600294), -INT64_C( 8282513756413395191), -INT64_C( 4719699255966020669),
         INT64_C( 3282126400805706560),  INT64_C( 7897855124379940363), -INT64_C( 6305742178178937775),  INT64_C( 8580072210332054987) },
      {  INT64_C( 1837740447273516598), -INT64_C( 9032200989554600294), -INT64_C( 8282513756413395191), -INT64_C( 4719699255966020669),
         INT64_C( 3282126400805706560),  INT64_C( 7897855124379940363), -INT64_C( 6305742178178937775),  INT64_C( 8580072210332054987) },
      {  INT64_C(   14357347244324348), -INT64_C(   70564070230895315), -INT64_C(   64707138721979650), -INT64_C(   36872650437234537),
         INT64_C(   25641612506294582),  INT64_C(   61701993159218284), -INT64_C(   49263610767022952),  INT64_C(   67031814143219179) },
      {  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1),
         INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0) },
      {  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1),
         INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0) } },
    { { -INT64_C( 8726438429205068251),  INT64_C( 3566351589455040429),  INT64_C( 4328977872487313541), -INT64_C( 3656462580906451565),
         INT64_C( 5740517919667071913),  INT64_C(  471715233121978489), -INT64_C( 2415472791425798962),  INT64_C( 6203122091742005921) },
      { -INT64_C( 8726438429205068251),  INT64_C( 3566351589455040429),  INT64_C( 4328977872487313541), -INT64_C( 3656462580906451565),
         INT64_C( 5740517919667071913),  INT64_C(  471715233121978489), -INT64_C( 2415472791425798962),  INT64_C( 6203122091742005921) },
      { -INT64_C(   68175300228164596),  INT64_C(   27862121792617503),  INT64_C(   33820139628807137), -INT64_C(   28566113913331653),
         INT64_C(   44847796247398999),  INT64_C(    3685275258765456), -INT64_C(   18870881183014055),  INT64_C(   48461891341734421) },
      { -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),
         INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0) },
      { -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),
         INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0) } },
    { {  INT64_C( 9009209182271478294),  INT64_C( 1437882233537390444), -INT64_C( 8486808195181746930), -INT64_C( 2172449384441847338),
        -INT64_C( 2016028822941903997),  INT64_C( 5503335240549194759),  INT64_C( 6026060930237247788), -INT64_C( 2419364992586910785) },
      {  INT64_C( 9009209182271478294),  INT64_C( 1437882233537390444), -INT64_C( 8486808195181746930), -INT64_C( 2172449384441847338),
        -INT64_C( 2016028822941903997),  INT64_C( 5503335240549194759),  INT64_C( 6026060930237247788), -INT64_C( 2419364992586910785) },
      {  INT64_C(   70384446736495924),  INT64_C(   11233454949510862), -INT64_C(   66303189024857398), -INT64_C(   16972260815951933),
        -INT64_C(   15750225179233625),  INT64_C(   42994806566790584),  INT64_C(   47078601017478498), -INT64_C(   18901289004585241) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1),
        -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1),
        -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(                   1) } },
    { { -INT64_C( 5953921819581479676),  INT64_C( 2653000579432460615),  INT64_C( 1503033862022330122),  INT64_C( 7387604757111616076),
        -INT64_C( 7033424873059587056), -INT64_C( 4427049740556607885), -INT64_C( 1236760766203705236), -INT64_C( 7953605980353825322) },
      { -INT64_C( 5953921819581479676),  INT64_C( 2653000579432460615),  INT64_C( 1503033862022330122),  INT64_C( 7387604757111616076),
        -INT64_C( 7033424873059587056), -INT64_C( 4427049740556607885), -INT64_C( 1236760766203705236), -INT64_C( 7953605980353825322) },
      { -INT64_C(   46515014215480310),  INT64_C(   20726567026816098),  INT64_C(   11742452047049454),  INT64_C(   57715662164934500),
        -INT64_C(   54948631820778024), -INT64_C(   34586326098098500), -INT64_C(    9662193485966448), -INT64_C(   62137546721514261) },
      { -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1) },
      { -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1) } },
    { {  INT64_C( 3963075052748374281), -INT64_C( 8898215555868056823), -INT64_C( 4068687864839877384),  INT64_C( 5090369658712760210),
        -INT64_C( 6910027506940122335), -INT64_C( 1988650483687769143),  INT64_C( 6658862086640020657),  INT64_C( 3146066750033848871) },
      {  INT64_C( 3963075052748374281), -INT64_C( 8898215555868056823), -INT64_C( 4068687864839877384),  INT64_C( 5090369658712760210),
        -INT64_C( 6910027506940122335), -INT64_C( 1988650483687769143),  INT64_C( 6658862086640020657),  INT64_C( 3146066750033848871) },
      {  INT64_C(   30961523849596674), -INT64_C(   69517309030219194), -INT64_C(   31786623944061543),  INT64_C(   39768512958693439),
        -INT64_C(   53984589897969706), -INT64_C(   15536331903810697),  INT64_C(   52022360051875161),  INT64_C(   24578646484639444) },
      {  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0) },
      {  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0) } },
    { { -INT64_C( 4596439126838849282),  INT64_C( 3293548034677446630),  INT64_C( 7123067588821595316),  INT64_C( 6546104502209366434),
        -INT64_C(  542923368237569856), -INT64_C( 2997568721288291907),  INT64_C( 1175772031716963763),  INT64_C( 5128919961250371447) },
      { -INT64_C( 4596439126838849282),  INT64_C( 3293548034677446630),  INT64_C( 7123067588821595316),  INT64_C( 6546104502209366434),
        -INT64_C(  542923368237569856), -INT64_C( 2997568721288291907),  INT64_C( 1175772031716963763),  INT64_C( 5128919961250371447) },
      { -INT64_C(   35909680678428511),  INT64_C(   25730844020917551),  INT64_C(   55648965537668713),  INT64_C(   51141441423510675),
        -INT64_C(    4241588814356015), -INT64_C(   23418505635064781),  INT64_C(    9185718997788779),  INT64_C(   40069687197268526) },
      { -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0) },
      { -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0) } },
    { {  INT64_C(  110467123909631357), -INT64_C( 8825188497809059185),  INT64_C( 9175456455136172758),  INT64_C( 2642347062813922013),
        -INT64_C( 5917101956589704803), -INT64_C( 7717792359406232435), -INT64_C(  412311234950965057), -INT64_C( 5257379608157841089) },
      {  INT64_C(  110467123909631357), -INT64_C( 8825188497809059185),  INT64_C( 9175456455136172758),  INT64_C( 2642347062813922013),
        -INT64_C( 5917101956589704803), -INT64_C( 7717792359406232435), -INT64_C(  412311234950965057), -INT64_C( 5257379608157841089) },
      {  INT64_C(     863024405543994), -INT64_C(   68946785139133275),  INT64_C(   71683253555751349),  INT64_C(   20643336428233765),
        -INT64_C(   46227359035857069), -INT64_C(   60295252807861191), -INT64_C(    3221181523054415), -INT64_C(   41073278188733134) },
      {  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1) },
      {  INT64_C(                   0), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1), -INT64_C(                   1) } },
    { { -INT64_C( 8683367818141843790), -INT64_C( 9157846910555988842),  INT64_C( 3259074434271002658),  INT64_C( 6122677375635593256),
        -INT64_C( 6819113894319648548),  INT64_C( 3441274639298675584), -INT64_C( 4631485372941071971), -INT64_C( 4019469350883701439) },
      { -INT64_C( 8683367818141843790), -INT64_C( 9157846910555988842),  INT64_C( 3259074434271002658),  INT64_C( 6122677375635593256),
        -INT64_C( 6819113894319648548),  INT64_C( 3441274639298675584), -INT64_C( 4631485372941071971), -INT64_C( 4019469350883701439) },
      { -INT64_C(   67838811079233155), -INT64_C(   71545678988718663),  INT64_C(   25461519017742208),  INT64_C(   47833416997153072),
        -INT64_C(   53274327299372255),  INT64_C(   26884958119520903), -INT64_C(   36183479476102125), -INT64_C(   31402104303778918) },
      { -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1),  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1) },
      { -INT64_C(                   1), -INT64_C(                   1),  INT64_C(                   0),  INT64_C(                   0),
        -INT64_C(                   1),  INT64_C(                   0), -INT64_C(                   1), -INT64_C(                   1) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    easysimd__m512i r0 = easysimd_mm512_srai_epi64(a, 0);
    easysimd__m512i r7;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r7 = easysimd_mm512_srai_epi64(a, 7);
    }
    EASYSIMD_TEST_PERF_END("_mm512_srai_epi64");
    easysimd__m512i r63 = easysimd_mm512_srai_epi64(a, 63);
    easysimd__m512i r64 = easysimd_mm512_srai_epi64(a, 64);
    easysimd_test_x86_assert_equal_i64x8(r0, easysimd_mm512_loadu_epi64(test_vec[i].r0));
    easysimd_test_x86_assert_equal_i64x8(r7, easysimd_mm512_loadu_epi64(test_vec[i].r7));
    easysimd_test_x86_assert_equal_i64x8(r63, easysimd_mm512_loadu_epi64(test_vec[i].r63));
    easysimd_test_x86_assert_equal_i64x8(r64, easysimd_mm512_loadu_epi64(test_vec[i].r64));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i r0 = easysimd_mm512_srai_epi64(a, 0);
    easysimd__m512i r7 = easysimd_mm512_srai_epi64(a, 7);
    easysimd__m512i r63 = easysimd_mm512_srai_epi64(a, 63);
    easysimd__m512i r64 = easysimd_mm512_srai_epi64(a, 64);

    easysimd_test_x86_write_i64x8(2, a,   EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, r0,  EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r7,  EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r63, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r64, EASYSIMD_TEST_VEC_POS_LAST);
  }

  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srai_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srai_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_srai_epi64)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
