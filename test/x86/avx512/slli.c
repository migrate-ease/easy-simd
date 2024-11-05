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
 *   2020      Christopher Moore <moore@free.fr>
 */

#define EASYSIMD_TEST_X86_AVX512_INSN slli

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/slli.h>

static int
test_easysimd_mm512_slli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    const int16_t a[32];
    const int16_t r0[32];
    const int16_t r3[32];
    const int16_t r7[32];
    const int16_t r13[32];
    const int16_t r24[32];
  } test_vec[] = {
    { { -INT16_C(  4513), -INT16_C( 32064), -INT16_C( 20539),  INT16_C( 16953), -INT16_C( 19443),  INT16_C(  8904),  INT16_C( 17111), -INT16_C( 18058),
         INT16_C(  9034), -INT16_C( 18739), -INT16_C( 25271),  INT16_C(  4847), -INT16_C( 27918), -INT16_C(  9400), -INT16_C(  4204), -INT16_C(  3107),
        -INT16_C( 24867), -INT16_C( 23691), -INT16_C( 20915),  INT16_C( 23269), -INT16_C( 21149),  INT16_C( 14973), -INT16_C(  3088),  INT16_C( 15091),
        -INT16_C( 16362),  INT16_C( 24561), -INT16_C(  8099),  INT16_C( 20594), -INT16_C( 17806),  INT16_C(  1579),  INT16_C(  2218), -INT16_C( 30727) },
      { -INT16_C(  4513), -INT16_C( 32064), -INT16_C( 20539),  INT16_C( 16953), -INT16_C( 19443),  INT16_C(  8904),  INT16_C( 17111), -INT16_C( 18058),
         INT16_C(  9034), -INT16_C( 18739), -INT16_C( 25271),  INT16_C(  4847), -INT16_C( 27918), -INT16_C(  9400), -INT16_C(  4204), -INT16_C(  3107),
        -INT16_C( 24867), -INT16_C( 23691), -INT16_C( 20915),  INT16_C( 23269), -INT16_C( 21149),  INT16_C( 14973), -INT16_C(  3088),  INT16_C( 15091),
        -INT16_C( 16362),  INT16_C( 24561), -INT16_C(  8099),  INT16_C( 20594), -INT16_C( 17806),  INT16_C(  1579),  INT16_C(  2218), -INT16_C( 30727) },
      {  INT16_C( 29432),  INT16_C(  5632),  INT16_C( 32296),  INT16_C(  4552), -INT16_C( 24472),  INT16_C(  5696),  INT16_C(  5816), -INT16_C( 13392),
         INT16_C(  6736), -INT16_C( 18840), -INT16_C(  5560), -INT16_C( 26760), -INT16_C( 26736), -INT16_C(  9664),  INT16_C( 31904), -INT16_C( 24856),
        -INT16_C(  2328),  INT16_C(  7080),  INT16_C( 29288), -INT16_C( 10456),  INT16_C( 27416), -INT16_C( 11288), -INT16_C( 24704), -INT16_C( 10344),
         INT16_C(   176), -INT16_C(   120),  INT16_C(   744), -INT16_C( 31856), -INT16_C( 11376),  INT16_C( 12632),  INT16_C( 17744),  INT16_C( 16328) },
      {  INT16_C( 12160),  INT16_C( 24576), -INT16_C(  7552),  INT16_C(  7296),  INT16_C(  1664),  INT16_C( 25600),  INT16_C( 27520), -INT16_C( 17664),
        -INT16_C( 23296),  INT16_C( 26240), -INT16_C( 23424),  INT16_C( 30592),  INT16_C( 30976), -INT16_C( 23552), -INT16_C( 13824), -INT16_C(  4480),
         INT16_C( 28288), -INT16_C( 17792),  INT16_C(  9856),  INT16_C( 29312), -INT16_C( 20096),  INT16_C( 16000), -INT16_C(  2048),  INT16_C( 31104),
         INT16_C(  2816), -INT16_C(  1920),  INT16_C( 11904),  INT16_C( 14592),  INT16_C( 14592),  INT16_C(  5504),  INT16_C( 21760), -INT16_C(   896) },
      { -INT16_C(  8192),  INT16_C(     0), -INT16_C( 24576),  INT16_C(  8192), -INT16_C( 24576),  INT16_C(     0), -INT16_C(  8192), -INT16_C( 16384),
         INT16_C( 16384), -INT16_C( 24576),  INT16_C(  8192), -INT16_C(  8192),  INT16_C( 16384),  INT16_C(     0),       INT16_MIN, -INT16_C( 24576),
        -INT16_C( 24576), -INT16_C( 24576), -INT16_C( 24576), -INT16_C( 24576),  INT16_C( 24576), -INT16_C( 24576),  INT16_C(     0),  INT16_C( 24576),
        -INT16_C( 16384),  INT16_C(  8192), -INT16_C( 24576),  INT16_C( 16384),  INT16_C( 16384),  INT16_C( 24576),  INT16_C( 16384),  INT16_C(  8192) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { {  INT16_C( 28582), -INT16_C(  3030),  INT16_C(  3869), -INT16_C( 32690), -INT16_C( 13379), -INT16_C( 21062), -INT16_C( 20801), -INT16_C( 10777),
        -INT16_C( 10130), -INT16_C( 13259), -INT16_C( 22600),  INT16_C( 11036),  INT16_C( 18273),  INT16_C(  2865),  INT16_C( 11087), -INT16_C(  2413),
        -INT16_C( 16998), -INT16_C( 18454),  INT16_C( 14541), -INT16_C( 30152), -INT16_C(  3580), -INT16_C( 15561),  INT16_C(  7840),  INT16_C(  3992),
        -INT16_C( 12809), -INT16_C( 20517), -INT16_C(  2188), -INT16_C( 10534),  INT16_C(  3134), -INT16_C( 29215),  INT16_C( 29751), -INT16_C( 11901) },
      {  INT16_C( 28582), -INT16_C(  3030),  INT16_C(  3869), -INT16_C( 32690), -INT16_C( 13379), -INT16_C( 21062), -INT16_C( 20801), -INT16_C( 10777),
        -INT16_C( 10130), -INT16_C( 13259), -INT16_C( 22600),  INT16_C( 11036),  INT16_C( 18273),  INT16_C(  2865),  INT16_C( 11087), -INT16_C(  2413),
        -INT16_C( 16998), -INT16_C( 18454),  INT16_C( 14541), -INT16_C( 30152), -INT16_C(  3580), -INT16_C( 15561),  INT16_C(  7840),  INT16_C(  3992),
        -INT16_C( 12809), -INT16_C( 20517), -INT16_C(  2188), -INT16_C( 10534),  INT16_C(  3134), -INT16_C( 29215),  INT16_C( 29751), -INT16_C( 11901) },
      {  INT16_C( 32048), -INT16_C( 24240),  INT16_C( 30952),  INT16_C(   624),  INT16_C( 24040),  INT16_C( 28112),  INT16_C( 30200), -INT16_C( 20680),
        -INT16_C( 15504),  INT16_C( 25000),  INT16_C( 15808),  INT16_C( 22752),  INT16_C( 15112),  INT16_C( 22920),  INT16_C( 23160), -INT16_C( 19304),
        -INT16_C(  4912), -INT16_C( 16560), -INT16_C( 14744),  INT16_C( 20928), -INT16_C( 28640),  INT16_C(  6584), -INT16_C(  2816),  INT16_C( 31936),
         INT16_C( 28600),  INT16_C( 32472), -INT16_C( 17504), -INT16_C( 18736),  INT16_C( 25072),  INT16_C( 28424), -INT16_C( 24136), -INT16_C( 29672) },
      { -INT16_C( 11520),  INT16_C(  5376), -INT16_C( 29056),  INT16_C(  9984), -INT16_C(  8576), -INT16_C(  8960),  INT16_C( 24448), -INT16_C(  3200),
         INT16_C( 14080),  INT16_C(  6784), -INT16_C(  9216), -INT16_C( 29184), -INT16_C( 20352), -INT16_C( 26496), -INT16_C( 22656),  INT16_C( 18816),
        -INT16_C( 13056), -INT16_C(  2816),  INT16_C( 26240),  INT16_C(  7168),  INT16_C(   512), -INT16_C( 25728),  INT16_C( 20480), -INT16_C( 13312),
        -INT16_C(  1152), -INT16_C(  4736), -INT16_C( 17920),  INT16_C( 27904),  INT16_C(  7936), -INT16_C(  3968),  INT16_C(  7040), -INT16_C( 16000) },
      { -INT16_C( 16384),  INT16_C( 16384), -INT16_C( 24576), -INT16_C( 16384), -INT16_C( 24576),  INT16_C( 16384), -INT16_C(  8192), -INT16_C(  8192),
        -INT16_C( 16384), -INT16_C( 24576),  INT16_C(     0),       INT16_MIN,  INT16_C(  8192),  INT16_C(  8192), -INT16_C(  8192),  INT16_C( 24576),
         INT16_C( 16384),  INT16_C( 16384), -INT16_C( 24576),  INT16_C(     0),       INT16_MIN, -INT16_C(  8192),  INT16_C(     0),  INT16_C(     0),
        -INT16_C(  8192),  INT16_C( 24576),       INT16_MIN,  INT16_C( 16384), -INT16_C( 16384),  INT16_C(  8192), -INT16_C(  8192),  INT16_C( 24576) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { {  INT16_C( 27954), -INT16_C(   120), -INT16_C( 16218), -INT16_C( 21879), -INT16_C( 16205),  INT16_C( 21357),  INT16_C(  1502), -INT16_C( 10910),
         INT16_C( 15827),  INT16_C( 18309),  INT16_C( 24372),  INT16_C( 29213), -INT16_C(   149), -INT16_C( 24064), -INT16_C( 31885), -INT16_C( 23181),
        -INT16_C(   783), -INT16_C( 26716),  INT16_C( 11708),  INT16_C( 28481), -INT16_C( 20755), -INT16_C( 13117),  INT16_C(  9651), -INT16_C( 31071),
         INT16_C(  9827), -INT16_C( 26674), -INT16_C(  5242), -INT16_C(  3830),  INT16_C(  2794),  INT16_C( 24212),  INT16_C(  1933),  INT16_C( 32259) },
      {  INT16_C( 27954), -INT16_C(   120), -INT16_C( 16218), -INT16_C( 21879), -INT16_C( 16205),  INT16_C( 21357),  INT16_C(  1502), -INT16_C( 10910),
         INT16_C( 15827),  INT16_C( 18309),  INT16_C( 24372),  INT16_C( 29213), -INT16_C(   149), -INT16_C( 24064), -INT16_C( 31885), -INT16_C( 23181),
        -INT16_C(   783), -INT16_C( 26716),  INT16_C( 11708),  INT16_C( 28481), -INT16_C( 20755), -INT16_C( 13117),  INT16_C(  9651), -INT16_C( 31071),
         INT16_C(  9827), -INT16_C( 26674), -INT16_C(  5242), -INT16_C(  3830),  INT16_C(  2794),  INT16_C( 24212),  INT16_C(  1933),  INT16_C( 32259) },
      {  INT16_C( 27024), -INT16_C(   960),  INT16_C(  1328),  INT16_C( 21576),  INT16_C(  1432), -INT16_C( 25752),  INT16_C( 12016), -INT16_C( 21744),
        -INT16_C(  4456),  INT16_C( 15400), -INT16_C(  1632), -INT16_C( 28440), -INT16_C(  1192),  INT16_C(  4096),  INT16_C(  7064),  INT16_C( 11160),
        -INT16_C(  6264), -INT16_C( 17120),  INT16_C( 28128),  INT16_C( 31240),  INT16_C( 30568),  INT16_C( 26136),  INT16_C( 11672),  INT16_C( 13576),
         INT16_C( 13080), -INT16_C( 16784),  INT16_C( 23600), -INT16_C( 30640),  INT16_C( 22352), -INT16_C(  2912),  INT16_C( 15464), -INT16_C(  4072) },
      { -INT16_C( 26368), -INT16_C( 15360),  INT16_C( 21248),  INT16_C( 17536),  INT16_C( 22912), -INT16_C( 18816), -INT16_C(  4352), -INT16_C( 20224),
        -INT16_C(  5760), -INT16_C( 15744), -INT16_C( 26112),  INT16_C(  3712), -INT16_C( 19072),  INT16_C(     0), -INT16_C( 18048), -INT16_C( 18048),
         INT16_C( 30848), -INT16_C( 11776), -INT16_C(  8704), -INT16_C( 24448),  INT16_C( 30336),  INT16_C( 24960), -INT16_C(  9856),  INT16_C( 20608),
         INT16_C( 12672), -INT16_C(  6400), -INT16_C( 15616), -INT16_C( 31488),  INT16_C( 29952),  INT16_C( 18944), -INT16_C( 14720),  INT16_C(   384) },
      {  INT16_C( 16384),  INT16_C(     0), -INT16_C( 16384),  INT16_C(  8192),  INT16_C( 24576), -INT16_C( 24576), -INT16_C( 16384),  INT16_C( 16384),
         INT16_C( 24576), -INT16_C( 24576),       INT16_MIN, -INT16_C( 24576),  INT16_C( 24576),  INT16_C(     0),  INT16_C( 24576),  INT16_C( 24576),
         INT16_C(  8192),       INT16_MIN,       INT16_MIN,  INT16_C(  8192), -INT16_C( 24576),  INT16_C( 24576),  INT16_C( 24576),  INT16_C(  8192),
         INT16_C( 24576), -INT16_C( 16384), -INT16_C( 16384),  INT16_C( 16384),  INT16_C( 16384),       INT16_MIN, -INT16_C( 24576),  INT16_C( 24576) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { -INT16_C( 22525), -INT16_C( 16363),  INT16_C( 22229), -INT16_C( 15569), -INT16_C(  3580), -INT16_C( 18289),  INT16_C( 12312),  INT16_C( 31550),
         INT16_C(  3159), -INT16_C(  8942),  INT16_C(  7416), -INT16_C(  7474),  INT16_C( 25126), -INT16_C( 19392),  INT16_C( 17514),  INT16_C( 27954),
         INT16_C( 18668), -INT16_C( 16083),  INT16_C( 23966), -INT16_C( 23676),  INT16_C(  4943),  INT16_C( 26459), -INT16_C( 26300), -INT16_C( 25630),
        -INT16_C(  2650), -INT16_C( 24968),  INT16_C( 17937),  INT16_C( 14464), -INT16_C( 15959),  INT16_C(  5100),  INT16_C(  7685), -INT16_C(  3712) },
      { -INT16_C( 22525), -INT16_C( 16363),  INT16_C( 22229), -INT16_C( 15569), -INT16_C(  3580), -INT16_C( 18289),  INT16_C( 12312),  INT16_C( 31550),
         INT16_C(  3159), -INT16_C(  8942),  INT16_C(  7416), -INT16_C(  7474),  INT16_C( 25126), -INT16_C( 19392),  INT16_C( 17514),  INT16_C( 27954),
         INT16_C( 18668), -INT16_C( 16083),  INT16_C( 23966), -INT16_C( 23676),  INT16_C(  4943),  INT16_C( 26459), -INT16_C( 26300), -INT16_C( 25630),
        -INT16_C(  2650), -INT16_C( 24968),  INT16_C( 17937),  INT16_C( 14464), -INT16_C( 15959),  INT16_C(  5100),  INT16_C(  7685), -INT16_C(  3712) },
      {  INT16_C( 16408),  INT16_C(   168), -INT16_C( 18776),  INT16_C(  6520), -INT16_C( 28640), -INT16_C( 15240), -INT16_C( 32576), -INT16_C(  9744),
         INT16_C( 25272), -INT16_C(  6000), -INT16_C(  6208),  INT16_C(  5744),  INT16_C(  4400), -INT16_C( 24064),  INT16_C(  9040),  INT16_C( 27024),
         INT16_C( 18272),  INT16_C(  2408), -INT16_C(  4880),  INT16_C(  7200), -INT16_C( 25992),  INT16_C( 15064), -INT16_C( 13792), -INT16_C(  8432),
        -INT16_C( 21200), -INT16_C(  3136),  INT16_C( 12424), -INT16_C( 15360),  INT16_C(  3400), -INT16_C( 24736), -INT16_C(  4056), -INT16_C( 29696) },
      {  INT16_C(   384),  INT16_C(  2688),  INT16_C( 27264), -INT16_C( 26752),  INT16_C(   512),  INT16_C( 18304),  INT16_C(  3072), -INT16_C( 24832),
         INT16_C( 11136), -INT16_C( 30464),  INT16_C( 31744),  INT16_C( 26368),  INT16_C(  4864),  INT16_C(  8192),  INT16_C( 13568), -INT16_C( 26368),
         INT16_C( 30208), -INT16_C( 27008), -INT16_C( 12544), -INT16_C( 15872), -INT16_C( 22656), -INT16_C( 21120), -INT16_C( 24064), -INT16_C(  3840),
        -INT16_C( 11520),  INT16_C( 15360),  INT16_C(  2176),  INT16_C( 16384), -INT16_C( 11136), -INT16_C(  2560),  INT16_C(   640), -INT16_C( 16384) },
      {  INT16_C( 24576), -INT16_C( 24576), -INT16_C( 24576), -INT16_C(  8192),       INT16_MIN, -INT16_C(  8192),  INT16_C(     0), -INT16_C( 16384),
        -INT16_C(  8192),  INT16_C( 16384),  INT16_C(     0), -INT16_C( 16384), -INT16_C( 16384),  INT16_C(     0),  INT16_C( 16384),  INT16_C( 16384),
              INT16_MIN, -INT16_C( 24576), -INT16_C( 16384),       INT16_MIN, -INT16_C(  8192),  INT16_C( 24576),       INT16_MIN,  INT16_C( 16384),
        -INT16_C( 16384),  INT16_C(     0),  INT16_C(  8192),  INT16_C(     0),  INT16_C(  8192),       INT16_MIN, -INT16_C( 24576),  INT16_C(     0) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { -INT16_C( 20890),  INT16_C(  1458),  INT16_C( 14091),  INT16_C( 23208),  INT16_C(   842), -INT16_C( 28990), -INT16_C( 23396),  INT16_C( 16937),
        -INT16_C( 24167), -INT16_C( 21536),  INT16_C( 25064), -INT16_C( 28189), -INT16_C( 12510),  INT16_C( 10148),  INT16_C(  9453),  INT16_C( 21528),
        -INT16_C( 13614), -INT16_C(  8871),  INT16_C(   257),  INT16_C( 19512), -INT16_C(  1532), -INT16_C( 24358),  INT16_C(  1182),  INT16_C( 14563),
        -INT16_C( 15451), -INT16_C( 29213), -INT16_C( 14812),  INT16_C( 17950), -INT16_C( 15723), -INT16_C( 32147), -INT16_C( 31257), -INT16_C( 17962) },
      { -INT16_C( 20890),  INT16_C(  1458),  INT16_C( 14091),  INT16_C( 23208),  INT16_C(   842), -INT16_C( 28990), -INT16_C( 23396),  INT16_C( 16937),
        -INT16_C( 24167), -INT16_C( 21536),  INT16_C( 25064), -INT16_C( 28189), -INT16_C( 12510),  INT16_C( 10148),  INT16_C(  9453),  INT16_C( 21528),
        -INT16_C( 13614), -INT16_C(  8871),  INT16_C(   257),  INT16_C( 19512), -INT16_C(  1532), -INT16_C( 24358),  INT16_C(  1182),  INT16_C( 14563),
        -INT16_C( 15451), -INT16_C( 29213), -INT16_C( 14812),  INT16_C( 17950), -INT16_C( 15723), -INT16_C( 32147), -INT16_C( 31257), -INT16_C( 17962) },
      {  INT16_C( 29488),  INT16_C( 11664), -INT16_C( 18344), -INT16_C( 10944),  INT16_C(  6736),  INT16_C( 30224),  INT16_C(  9440),  INT16_C(  4424),
         INT16_C(  3272),  INT16_C( 24320),  INT16_C(  3904), -INT16_C( 28904),  INT16_C( 30992),  INT16_C( 15648),  INT16_C( 10088), -INT16_C( 24384),
         INT16_C( 22160), -INT16_C(  5432),  INT16_C(  2056),  INT16_C( 25024), -INT16_C( 12256),  INT16_C(  1744),  INT16_C(  9456), -INT16_C( 14568),
         INT16_C(  7464),  INT16_C( 28440),  INT16_C( 12576),  INT16_C( 12528),  INT16_C(  5288),  INT16_C(  4968),  INT16_C( 12088), -INT16_C( 12624) },
      {  INT16_C( 13056), -INT16_C(  9984), -INT16_C( 31360),  INT16_C( 21504), -INT16_C( 23296),  INT16_C( 24832),  INT16_C( 19968),  INT16_C(  5248),
        -INT16_C( 13184), -INT16_C(  4096), -INT16_C(  3072), -INT16_C(  3712), -INT16_C( 28416), -INT16_C( 11776),  INT16_C( 30336),  INT16_C(  3072),
         INT16_C( 26880), -INT16_C( 21376), -INT16_C( 32640),  INT16_C(  7168),  INT16_C(   512),  INT16_C( 27904),  INT16_C( 20224),  INT16_C( 29056),
        -INT16_C( 11648), -INT16_C(  3712),  INT16_C(  4608),  INT16_C(  3840),  INT16_C( 19072),  INT16_C( 13952), -INT16_C(  3200), -INT16_C(  5376) },
      { -INT16_C( 16384),  INT16_C( 16384),  INT16_C( 24576),  INT16_C(     0),  INT16_C( 16384),  INT16_C( 16384),       INT16_MIN,  INT16_C(  8192),
         INT16_C(  8192),  INT16_C(     0),  INT16_C(     0),  INT16_C( 24576),  INT16_C( 16384),       INT16_MIN, -INT16_C( 24576),  INT16_C(     0),
         INT16_C( 16384),  INT16_C(  8192),  INT16_C(  8192),  INT16_C(     0),       INT16_MIN,  INT16_C( 16384), -INT16_C( 16384),  INT16_C( 24576),
        -INT16_C( 24576),  INT16_C( 24576),       INT16_MIN, -INT16_C( 16384), -INT16_C( 24576), -INT16_C( 24576), -INT16_C(  8192), -INT16_C( 16384) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { {  INT16_C( 12112),  INT16_C( 20887), -INT16_C( 12496),  INT16_C( 13469),  INT16_C( 30921),  INT16_C( 26581), -INT16_C( 18308),  INT16_C(  8607),
        -INT16_C( 32133), -INT16_C( 24401), -INT16_C( 12984), -INT16_C(  8730),  INT16_C( 21648),  INT16_C( 30560),  INT16_C( 14041),  INT16_C( 10544),
        -INT16_C( 14490), -INT16_C( 27013),  INT16_C(  6294),  INT16_C( 24523), -INT16_C( 24432),  INT16_C(  3271),  INT16_C( 26200), -INT16_C( 11474),
        -INT16_C(  8727),  INT16_C( 12659),  INT16_C( 23210),  INT16_C( 14863),  INT16_C( 28590), -INT16_C( 30799), -INT16_C(  7515),  INT16_C(  2993) },
      {  INT16_C( 12112),  INT16_C( 20887), -INT16_C( 12496),  INT16_C( 13469),  INT16_C( 30921),  INT16_C( 26581), -INT16_C( 18308),  INT16_C(  8607),
        -INT16_C( 32133), -INT16_C( 24401), -INT16_C( 12984), -INT16_C(  8730),  INT16_C( 21648),  INT16_C( 30560),  INT16_C( 14041),  INT16_C( 10544),
        -INT16_C( 14490), -INT16_C( 27013),  INT16_C(  6294),  INT16_C( 24523), -INT16_C( 24432),  INT16_C(  3271),  INT16_C( 26200), -INT16_C( 11474),
        -INT16_C(  8727),  INT16_C( 12659),  INT16_C( 23210),  INT16_C( 14863),  INT16_C( 28590), -INT16_C( 30799), -INT16_C(  7515),  INT16_C(  2993) },
      {  INT16_C( 31360), -INT16_C( 29512),  INT16_C( 31104), -INT16_C( 23320), -INT16_C( 14776),  INT16_C( 16040), -INT16_C( 15392),  INT16_C(  3320),
         INT16_C(  5080),  INT16_C(  1400),  INT16_C( 27200), -INT16_C(  4304), -INT16_C( 23424), -INT16_C( 17664), -INT16_C( 18744),  INT16_C( 18816),
         INT16_C( 15152), -INT16_C( 19496), -INT16_C( 15184), -INT16_C(   424),  INT16_C(  1152),  INT16_C( 26168),  INT16_C( 12992), -INT16_C( 26256),
        -INT16_C(  4280), -INT16_C( 29800), -INT16_C( 10928), -INT16_C( 12168),  INT16_C( 32112),  INT16_C( 15752),  INT16_C(  5416),  INT16_C( 23944) },
      { -INT16_C( 22528), -INT16_C( 13440), -INT16_C( 26624),  INT16_C( 20096),  INT16_C( 25728), -INT16_C(  5504),  INT16_C( 15872), -INT16_C( 12416),
         INT16_C( 15744),  INT16_C( 22400), -INT16_C( 23552), -INT16_C(  3328),  INT16_C( 18432), -INT16_C( 20480),  INT16_C( 27776), -INT16_C( 26624),
        -INT16_C( 19712),  INT16_C( 15744),  INT16_C( 19200), -INT16_C(  6784),  INT16_C( 18432),  INT16_C( 25472),  INT16_C( 11264), -INT16_C( 26880),
        -INT16_C(  2944), -INT16_C( 18048),  INT16_C( 21760),  INT16_C(  1920), -INT16_C( 10496), -INT16_C( 10112),  INT16_C( 21120), -INT16_C( 10112) },
      {  INT16_C(     0), -INT16_C(  8192),  INT16_C(     0), -INT16_C( 24576),  INT16_C(  8192), -INT16_C( 24576),       INT16_MIN, -INT16_C(  8192),
         INT16_C( 24576), -INT16_C(  8192),  INT16_C(     0), -INT16_C( 16384),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8192),  INT16_C(     0),
        -INT16_C( 16384),  INT16_C( 24576), -INT16_C( 16384),  INT16_C( 24576),  INT16_C(     0), -INT16_C(  8192),  INT16_C(     0), -INT16_C( 16384),
         INT16_C(  8192),  INT16_C( 24576),  INT16_C( 16384), -INT16_C(  8192), -INT16_C( 16384),  INT16_C(  8192), -INT16_C( 24576),  INT16_C(  8192) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { {  INT16_C( 11433),  INT16_C( 16546),  INT16_C( 27972), -INT16_C( 10849),  INT16_C( 26125),  INT16_C( 26081),  INT16_C(  4045), -INT16_C( 18888),
        -INT16_C( 21268), -INT16_C( 26649), -INT16_C(  2554), -INT16_C( 19247), -INT16_C( 31899),  INT16_C(  2875), -INT16_C(  5019),  INT16_C(  3606),
        -INT16_C( 18408),  INT16_C( 23886), -INT16_C(  4571),  INT16_C( 12850),  INT16_C(  4948),  INT16_C(  8599), -INT16_C( 12253),  INT16_C(  4055),
        -INT16_C( 16516), -INT16_C( 32090),  INT16_C( 30901),  INT16_C(  6966),  INT16_C( 29179),  INT16_C( 24614),  INT16_C( 15454),  INT16_C( 30318) },
      {  INT16_C( 11433),  INT16_C( 16546),  INT16_C( 27972), -INT16_C( 10849),  INT16_C( 26125),  INT16_C( 26081),  INT16_C(  4045), -INT16_C( 18888),
        -INT16_C( 21268), -INT16_C( 26649), -INT16_C(  2554), -INT16_C( 19247), -INT16_C( 31899),  INT16_C(  2875), -INT16_C(  5019),  INT16_C(  3606),
        -INT16_C( 18408),  INT16_C( 23886), -INT16_C(  4571),  INT16_C( 12850),  INT16_C(  4948),  INT16_C(  8599), -INT16_C( 12253),  INT16_C(  4055),
        -INT16_C( 16516), -INT16_C( 32090),  INT16_C( 30901),  INT16_C(  6966),  INT16_C( 29179),  INT16_C( 24614),  INT16_C( 15454),  INT16_C( 30318) },
      {  INT16_C( 25928),  INT16_C(  1296),  INT16_C( 27168), -INT16_C( 21256),  INT16_C( 12392),  INT16_C( 12040),  INT16_C( 32360), -INT16_C( 20032),
         INT16_C( 26464), -INT16_C( 16584), -INT16_C( 20432), -INT16_C( 22904),  INT16_C(  6952),  INT16_C( 23000),  INT16_C( 25384),  INT16_C( 28848),
        -INT16_C( 16192), -INT16_C(  5520),  INT16_C( 28968), -INT16_C( 28272), -INT16_C( 25952),  INT16_C(  3256), -INT16_C( 32488),  INT16_C( 32440),
        -INT16_C(  1056),  INT16_C(  5424), -INT16_C( 14936), -INT16_C(  9808), -INT16_C( 28712),  INT16_C(   304), -INT16_C(  7440), -INT16_C( 19600) },
      {  INT16_C( 21632),  INT16_C( 20736), -INT16_C( 24064), -INT16_C( 12416),  INT16_C(  1664), -INT16_C(  3968), -INT16_C(  6528),  INT16_C(  7168),
         INT16_C( 30208), -INT16_C(  3200),  INT16_C(   768),  INT16_C( 26752), -INT16_C( 19840), -INT16_C( 25216),  INT16_C( 12928),  INT16_C(  2816),
         INT16_C(  3072), -INT16_C( 22784),  INT16_C(  4736),  INT16_C(  6400), -INT16_C( 22016), -INT16_C( 13440),  INT16_C(  4480), -INT16_C(  5248),
        -INT16_C( 16896),  INT16_C( 21248),  INT16_C( 23168), -INT16_C( 25856), -INT16_C(   640),  INT16_C(  4864),  INT16_C( 12032),  INT16_C( 14080) },
      {  INT16_C(  8192),  INT16_C( 16384),       INT16_MIN, -INT16_C(  8192), -INT16_C( 24576),  INT16_C(  8192), -INT16_C( 24576),  INT16_C(     0),
              INT16_MIN, -INT16_C(  8192), -INT16_C( 16384),  INT16_C(  8192), -INT16_C( 24576),  INT16_C( 24576), -INT16_C( 24576), -INT16_C( 16384),
         INT16_C(     0), -INT16_C( 16384), -INT16_C( 24576),  INT16_C( 16384),       INT16_MIN, -INT16_C(  8192),  INT16_C( 24576), -INT16_C(  8192),
              INT16_MIN, -INT16_C( 16384), -INT16_C( 24576), -INT16_C( 16384),  INT16_C( 24576), -INT16_C( 16384), -INT16_C( 16384), -INT16_C( 16384) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { { -INT16_C( 16907),  INT16_C(  6867),  INT16_C(  1451), -INT16_C(   179), -INT16_C(  7143),  INT16_C( 15393), -INT16_C(  1868),  INT16_C( 12363),
        -INT16_C(  3401),  INT16_C( 28082), -INT16_C(  6038),  INT16_C( 25992), -INT16_C( 20902), -INT16_C( 18235),  INT16_C( 13290), -INT16_C(  8402),
         INT16_C(   752), -INT16_C( 25606),  INT16_C( 18183),  INT16_C(  8347), -INT16_C( 17365), -INT16_C(  8100), -INT16_C( 22348),  INT16_C( 27664),
        -INT16_C( 15462),  INT16_C(  1241),  INT16_C( 25003),  INT16_C(  1385),  INT16_C( 11791), -INT16_C(  1603), -INT16_C(  5023),  INT16_C( 21209) },
      { -INT16_C( 16907),  INT16_C(  6867),  INT16_C(  1451), -INT16_C(   179), -INT16_C(  7143),  INT16_C( 15393), -INT16_C(  1868),  INT16_C( 12363),
        -INT16_C(  3401),  INT16_C( 28082), -INT16_C(  6038),  INT16_C( 25992), -INT16_C( 20902), -INT16_C( 18235),  INT16_C( 13290), -INT16_C(  8402),
         INT16_C(   752), -INT16_C( 25606),  INT16_C( 18183),  INT16_C(  8347), -INT16_C( 17365), -INT16_C(  8100), -INT16_C( 22348),  INT16_C( 27664),
        -INT16_C( 15462),  INT16_C(  1241),  INT16_C( 25003),  INT16_C(  1385),  INT16_C( 11791), -INT16_C(  1603), -INT16_C(  5023),  INT16_C( 21209) },
      { -INT16_C(  4184), -INT16_C( 10600),  INT16_C( 11608), -INT16_C(  1432),  INT16_C(  8392), -INT16_C(  7928), -INT16_C( 14944), -INT16_C( 32168),
        -INT16_C( 27208),  INT16_C( 28048),  INT16_C( 17232),  INT16_C( 11328),  INT16_C( 29392), -INT16_C( 14808), -INT16_C( 24752), -INT16_C(  1680),
         INT16_C(  6016), -INT16_C(  8240),  INT16_C( 14392),  INT16_C(  1240), -INT16_C(  7848),  INT16_C(   736),  INT16_C( 17824),  INT16_C( 24704),
         INT16_C(  7376),  INT16_C(  9928),  INT16_C(  3416),  INT16_C( 11080),  INT16_C( 28792), -INT16_C( 12824),  INT16_C( 25352), -INT16_C( 26936) },
      { -INT16_C(  1408),  INT16_C( 27008), -INT16_C( 10880), -INT16_C( 22912),  INT16_C(  3200),  INT16_C(  4224),  INT16_C( 23040),  INT16_C(  9600),
         INT16_C( 23424), -INT16_C(  9984),  INT16_C( 13568), -INT16_C( 15360),  INT16_C( 11520),  INT16_C( 25216), -INT16_C(  2816), -INT16_C( 26880),
         INT16_C( 30720), -INT16_C(   768), -INT16_C( 31872),  INT16_C( 19840),  INT16_C(  5504),  INT16_C( 11776),  INT16_C( 23040),  INT16_C(  2048),
        -INT16_C( 13056),  INT16_C( 27776), -INT16_C( 10880), -INT16_C( 19328),  INT16_C(  1920), -INT16_C(  8576),  INT16_C( 12416),  INT16_C( 27776) },
      { -INT16_C( 24576),  INT16_C( 24576),  INT16_C( 24576), -INT16_C( 24576),  INT16_C(  8192),  INT16_C(  8192),       INT16_MIN,  INT16_C( 24576),
        -INT16_C(  8192),  INT16_C( 16384),  INT16_C( 16384),  INT16_C(     0),  INT16_C( 16384), -INT16_C( 24576),  INT16_C( 16384), -INT16_C( 16384),
         INT16_C(     0),  INT16_C( 16384), -INT16_C(  8192),  INT16_C( 24576),  INT16_C( 24576),       INT16_MIN,       INT16_MIN,  INT16_C(     0),
         INT16_C( 16384),  INT16_C(  8192),  INT16_C( 24576),  INT16_C(  8192), -INT16_C(  8192), -INT16_C( 24576),  INT16_C(  8192),  INT16_C(  8192) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } }
  };

  easysimd__m512i a;
  easysimd__m512i r0;
  easysimd__m512i r3;
  easysimd__m512i r7;
  easysimd__m512i r13;

  a = easysimd_mm512_loadu_epi16(test_vec[0].a);
  r0 = easysimd_mm512_slli_epi16(a, 0);
  easysimd_test_x86_assert_equal_i16x32(r0, easysimd_mm512_loadu_epi16(test_vec[0].r0));
  r3 = easysimd_mm512_slli_epi16(a, 3);
  easysimd_test_x86_assert_equal_i16x32(r3, easysimd_mm512_loadu_epi16(test_vec[0].r3));
  r7 = easysimd_mm512_slli_epi16(a, 7);
  easysimd_test_x86_assert_equal_i16x32(r7, easysimd_mm512_loadu_epi16(test_vec[0].r7));
  EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
    r13 = easysimd_mm512_slli_epi16(a, 13);
  } EASYSIMD_TEST_PERF_END("easysimd_mm512_slli_epi16");
  easysimd_test_x86_assert_equal_i16x32(r13, easysimd_mm512_loadu_epi16(test_vec[0].r13));

  a = easysimd_mm512_loadu_epi16(test_vec[1].a);
  r0 = easysimd_mm512_slli_epi16(a, 0);
  easysimd_test_x86_assert_equal_i16x32(r0, easysimd_mm512_loadu_epi16(test_vec[1].r0));
  r3 = easysimd_mm512_slli_epi16(a, 3);
  easysimd_test_x86_assert_equal_i16x32(r3, easysimd_mm512_loadu_epi16(test_vec[1].r3));
  r7 = easysimd_mm512_slli_epi16(a, 7);
  easysimd_test_x86_assert_equal_i16x32(r7, easysimd_mm512_loadu_epi16(test_vec[1].r7));
  EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
    r13 = easysimd_mm512_slli_epi16(a, 13);
  } EASYSIMD_TEST_PERF_END("easysimd_mm512_slli_epi16");
  easysimd_test_x86_assert_equal_i16x32(r13, easysimd_mm512_loadu_epi16(test_vec[1].r13));

  return 0;
}

static int
test_easysimd_mm512_slli_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi32(INT32_C( -687706949), INT32_C( 1593775683), INT32_C(  332932989), INT32_C(  583872054),
                            INT32_C( 1838832857), INT32_C(  847835558), INT32_C(-1396128258), INT32_C( -183977070),
                            INT32_C( -902383138), INT32_C( -512492201), INT32_C(-1812249336), INT32_C( -562835271),
                            INT32_C(-1029714159), INT32_C( 1476158556), INT32_C(  877549641), INT32_C( 1218378177)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
    { easysimd_mm512_set_epi32(INT32_C(  241549121), INT32_C( 1732816264), INT32_C(  875489890), INT32_C(   72071518),
                            INT32_C(-1641761300), INT32_C(  313288882), INT32_C(-1735158939), INT32_C( 1219761116),
                            INT32_C(  877921588), INT32_C( 2045964482), INT32_C( -360092415), INT32_C(-1302958505),
                            INT32_C(-1122092800), INT32_C( -177019481), INT32_C(  875636041), INT32_C( -150268654)),
      easysimd_mm512_set_epi32(INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0),
                            INT32_C(          0), INT32_C(          0), INT32_C(          0), INT32_C(          0)) },
    { easysimd_mm512_set_epi32(INT32_C(  -52534216), INT32_C( -794188551), INT32_C( -186449823), INT32_C( 1580979103),
                            INT32_C( -972993456), INT32_C( -666426563), INT32_C( -645023430), INT32_C(-1043227266),
                            INT32_C( 1237525980), INT32_C(  349749966), INT32_C( -978999744), INT32_C( -487689408),
                            INT32_C(  898649460), INT32_C(-1217796896), INT32_C( 1277301360), INT32_C( 1454357892)),
      easysimd_mm512_set_epi32(INT32_C( 1681391616), INT32_C(-1527185408), INT32_C(    6356992), INT32_C( -744554496),
                            INT32_C( 1280311296), INT32_C(  591200256), INT32_C(-1187381248), INT32_C(-1652686848),
                            INT32_C(  635174912), INT32_C(-1026686976), INT32_C(-1505755136), INT32_C( 1933574144),
                            INT32_C( 1299447808), INT32_C( -455081984), INT32_C(  309329920), INT32_C(-1115422720)) },
    { easysimd_mm512_set_epi32(INT32_C(-1898779532), INT32_C( 1141724513), INT32_C( -782352739), INT32_C(  973072470),
                            INT32_C( 1112939167), INT32_C(  368903984), INT32_C( 1631675339), INT32_C(  -86505288),
                            INT32_C( 2011287771), INT32_C( 1938765310), INT32_C( 2078191935), INT32_C( -616151900),
                            INT32_C( -454977425), INT32_C(  544105809), INT32_C(-1307935124), INT32_C(-1400984309)),
      easysimd_mm512_set_epi32(INT32_C( 1269944320), INT32_C(  894796800), INT32_C( 2029679616), INT32_C(   -6203392),
                            INT32_C( 1483373568), INT32_C( -199442432), INT32_C(   93268992), INT32_C( 1612898304),
                            INT32_C(-2025624576), INT32_C( 1020786688), INT32_C( 2059729920), INT32_C(  420646912),
                            INT32_C(-2040415232), INT32_C(-1181400064), INT32_C(  704229376), INT32_C(  -88855552)) },
    { easysimd_mm512_set_epi32(INT32_C( -955538666), INT32_C( 1399393330), INT32_C( 1832782688), INT32_C(-1931362608),
                            INT32_C(-1247233529), INT32_C( -537843102), INT32_C( -120831887), INT32_C( 1329473476),
                            INT32_C( 1569899726), INT32_C(  920247722), INT32_C(  275348332), INT32_C( 1640312018),
                            INT32_C( -873496512), INT32_C(  957396290), INT32_C(  390504842), INT32_C( 1781792417)),
      easysimd_mm512_set_epi32(INT32_C( -779419648), INT32_C(-2045181952), INT32_C(   95420416), INT32_C( -546832384),
                            INT32_C( 1560510464), INT32_C(-1791950848), INT32_C(  540573696), INT32_C(  333578240),
                            INT32_C( 1650917376), INT32_C( -288030720), INT32_C(-1112145920), INT32_C(-1771503616),
                            INT32_C(-1071644672), INT32_C( 1520500736), INT32_C( 1355087872), INT32_C(  -11501568)) },
    { easysimd_mm512_set_epi32(INT32_C(  188085108), INT32_C(  489074602), INT32_C( 1720231560), INT32_C(  106164094),
                            INT32_C( 1250223633), INT32_C( -962071158), INT32_C(   38255424), INT32_C(  801121683),
                            INT32_C(-1580720854), INT32_C(  609844423), INT32_C(   44983522), INT32_C(  481953328),
                            INT32_C( -181212371), INT32_C(  912186226), INT32_C(  -42587351), INT32_C(  680089879)),
      easysimd_mm512_set_epi32(INT32_C( 1946157056), INT32_C(-1442840576), INT32_C(-2013265920), INT32_C( 2113929216),
                            INT32_C(  285212672), INT32_C(-1979711488), INT32_C( 1073741824), INT32_C(-1828716544),
                            INT32_C(  704643072), INT32_C( -956301312), INT32_C( -503316480), INT32_C(  805306368),
                            INT32_C(  754974720), INT32_C( 1912602624), INT32_C(  687865856), INT32_C(  385875968)) },
    { easysimd_mm512_set_epi32(INT32_C(-1878529143), INT32_C(  968369206), INT32_C(-2025408372), INT32_C( -521427427),
                            INT32_C(  750337953), INT32_C( 1599422728), INT32_C( 1832999614), INT32_C( -922516627),
                            INT32_C( 1054703043), INT32_C( -229764941), INT32_C(-1888970968), INT32_C( -770679003),
                            INT32_C(  957667650), INT32_C(-1367078699), INT32_C(  400185050), INT32_C(  619858989)),
      easysimd_mm512_set_epi32(INT32_C(-1878529143), INT32_C(  968369206), INT32_C(-2025408372), INT32_C( -521427427),
                            INT32_C(  750337953), INT32_C( 1599422728), INT32_C( 1832999614), INT32_C( -922516627),
                            INT32_C( 1054703043), INT32_C( -229764941), INT32_C(-1888970968), INT32_C( -770679003),
                            INT32_C(  957667650), INT32_C(-1367078699), INT32_C(  400185050), INT32_C(  619858989)) },
    { easysimd_mm512_set_epi32(INT32_C( -939632719), INT32_C( 1727963384), INT32_C( 1880331239), INT32_C(  699090974),
                            INT32_C( 1068401563), INT32_C(-1558361689), INT32_C(-1814494206), INT32_C( 1865180366),
                            INT32_C(-1767733366), INT32_C(-1147256695), INT32_C(-1631901793), INT32_C( -198157319),
                            INT32_C(  285018015), INT32_C(  583696937), INT32_C( 1785762602), INT32_C(-1724046997)),
      easysimd_mm512_set_epi32(INT32_C( -331350016), INT32_C( 1040187392), INT32_C( -104857600), INT32_C(  125829120),
                            INT32_C( -423624704), INT32_C( 1774190592), INT32_C(    8388608), INT32_C(  864026624),
                            INT32_C( -494927872), INT32_C(  574619648), INT32_C( -406847488), INT32_C(  -29360128),
                            INT32_C( -406847488), INT32_C(-1975517184), INT32_C( -897581056), INT32_C( 1522532352)) },
  };

  easysimd__m512i a;
  easysimd__m512i r;

#if (!defined(__clang__))
  a = test_vec[0].a;
  r = easysimd_mm512_slli_epi32(a, 0xac);
  easysimd_assert_m512i_i32(r, ==, test_vec[0].r);

  a = test_vec[1].a;
  r = easysimd_mm512_slli_epi32(a, 0x8017);
  easysimd_assert_m512i_i32(r, ==, test_vec[1].r);
#endif
  a = test_vec[2].a;
  r = easysimd_mm512_slli_epi32(a, 0x10);
  easysimd_assert_m512i_i32(r, ==, test_vec[2].r);

  a = test_vec[3].a;
  r = easysimd_mm512_slli_epi32(a, 0xa);
  easysimd_assert_m512i_i32(r, ==, test_vec[3].r);

  a = test_vec[4].a;
  r = easysimd_mm512_slli_epi32(a, 0xf);
  easysimd_assert_m512i_i32(r, ==, test_vec[4].r);

  a = test_vec[5].a;
  r = easysimd_mm512_slli_epi32(a, 0x18);
  easysimd_assert_m512i_i32(r, ==, test_vec[5].r);

  a = test_vec[6].a;
  r = easysimd_mm512_slli_epi32(a, 0);
  easysimd_assert_m512i_i32(r, ==, test_vec[6].r);

  a = test_vec[7].a;
  EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
    r = easysimd_mm512_slli_epi32(a, 0x16);
  } EASYSIMD_TEST_PERF_END("easysimd_mm512_slli_epi32");
  easysimd_assert_m512i_i32(r, ==, test_vec[7].r);

  return 0;
}

static int
test_easysimd_mm512_slli_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_mm512_set_epi64(INT64_C(-2953678853593164221), INT64_C( 1429936300098399798),
                            INT64_C( 7897726984473080230), INT64_C(-5996325205020460142),
                            INT64_C(-3875706062389379753), INT64_C(-7783551626585583431),
                            INT64_C(-4422588635656985508), INT64_C( 3769047009929918913)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { easysimd_mm512_set_epi64(INT64_C( 1037445576805363080), INT64_C( 3760200445600708958),
                            INT64_C(-7051311091025155918), INT64_C(-7452450895147297828),
                            INT64_C( 3770644510958350530), INT64_C(-1546585142970651049),
                            INT64_C(-4819351874959120985), INT64_C( 3760828163438613778)),
      easysimd_mm512_set_epi64(INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0),
                            INT64_C(                   0), INT64_C(                   0)) },
    { easysimd_mm512_set_epi64(INT64_C( -225632736140221191), INT64_C( -800795890549009505),
                            INT64_C(-4178975069113474243), INT64_C(-2770354533752005250),
                            INT64_C( 5315133612400100046), INT64_C(-4204771879465094336),
                            INT64_C( 3859670044345230560), INT64_C( 5485967569790680452)),
      easysimd_mm512_set_epi64(INT64_C( 7221751429524422656), INT64_C(   27406683787427840),
                            INT64_C( 5499132945064853504), INT64_C(-5099550522009059328),
                            INT64_C( 2728078395493449728), INT64_C(-6466919551140364288),
                            INT64_C( 5581287503658221568), INT64_C( 1328657202873106432)) },
    { easysimd_mm512_set_epi64(INT64_C(  854012069371251830), INT64_C(-7338075353641633319),
                            INT64_C(-3664756911608965568), INT64_C( 6190577389993756354),
                            INT64_C(-5695521678932466387), INT64_C( 8986269833406294113),
                            INT64_C(-6594347992267195055), INT64_C(  897886006004895547)),
      easysimd_mm512_set_epi64(INT64_C( 1455181649128980480), INT64_C(-8005319861464989696),
                            INT64_C(-8117456853358608384), INT64_C(-7918726723707863040),
                            INT64_C(-4137484245553643520), INT64_C( -675113333593997312),
                            INT64_C( 7635083510067232768), INT64_C(-2504867807980683264)) },
    { easysimd_mm512_set_epi64(INT64_C( 6876450537877586373), INT64_C(-2498836913726354503),
                            INT64_C(-5925650014767999746), INT64_C(-5091981247482556140),
                            INT64_C( 2276397305581596841), INT64_C( 4333846664358463853),
                            INT64_C( -724672155607878887), INT64_C( 3417746373838389455)),
      easysimd_mm512_set_epi64(INT64_C( -934756407423533056), INT64_C(-3157335600089006080),
                            INT64_C(-2570438283414732800), INT64_C( 7513218039291052032),
                            INT64_C( 7188488275143688192), INT64_C( 7216371890024087552),
                            INT64_C(-9165841190443024384), INT64_C( 1225889494272573440)) },
    { easysimd_mm512_set_epi64(INT64_C(-8036497785869311574), INT64_C( 3581702479948115598),
                            INT64_C(  748249211564829520), INT64_C( -816680525172154454),
                            INT64_C(-4839891842343135042), INT64_C(-6001583230129728210),
                            INT64_C(-4279294013059977744), INT64_C(-1555144075545091790)),
      easysimd_mm512_set_epi64(INT64_C(-4397647938138931200), INT64_C( 2043406626093793280),
                            INT64_C(-5590422890961960960), INT64_C(-8604150727591329792),
                            INT64_C( 8423472379845410816), INT64_C(-8411233715916636160),
                            INT64_C(-8742757912167841792), INT64_C(-6541394346116120576)) },
    { easysimd_mm512_set_epi64(INT64_C(-6276545081940248579), INT64_C(-9016855820360504888),
                            INT64_C( 2589347389053699338), INT64_C(-6212989007002338187),
                            INT64_C( 5925964847698460032), INT64_C( 8758478916256841908),
                            INT64_C( 5134329058456078862), INT64_C(-4414137185393506410)),
      easysimd_mm512_set_epi64(INT64_C(-9176583453456465920), INT64_C( 7465982649455083520),
                            INT64_C(-4954907897243893760), INT64_C( 7673069422566703104),
                            INT64_C(-6790719338690117632), INT64_C( 3134572139001151488),
                            INT64_C( 1398716822424911872), INT64_C(-7065366029995606016)) },
    { easysimd_mm512_set_epi64(INT64_C(-4035691796628594440), INT64_C( 8075961177851250718),
                            INT64_C( 4588749774816889255), INT64_C(-7793193271686306610),
                            INT64_C(-7592356991870287735), INT64_C(-7008964827121951751),
                            INT64_C( 1224143053779534377), INT64_C( 7669791976580784491)),
      easysimd_mm512_set_epi64(INT64_C(-1415889878515712000), INT64_C( -447427762668437504),
                            INT64_C(-1807976093613817856), INT64_C(   43851930488799232),
                            INT64_C(-2112496568954257408), INT64_C(-1730213388945981440),
                            INT64_C(-1744948453022105600), INT64_C(-3844298059735367680)) },
  };

  easysimd__m512i a;
  easysimd__m512i r;

#if (!defined(__clang__))
  a = test_vec[0].a;
  r = easysimd_mm512_slli_epi64(a, 0xac);
  easysimd_assert_m512i_i64(r, ==, test_vec[0].r);

  a = test_vec[1].a;
  r = easysimd_mm512_slli_epi64(a, 0x8017);
  easysimd_assert_m512i_i64(r, ==, test_vec[1].r);
#endif
  a = test_vec[2].a;
  r = easysimd_mm512_slli_epi64(a, 0x10);
  easysimd_assert_m512i_i64(r, ==, test_vec[2].r);

  a = test_vec[3].a;
  r = easysimd_mm512_slli_epi64(a, 0x2a);
  easysimd_assert_m512i_i64(r, ==, test_vec[3].r);

  a = test_vec[4].a;
  r = easysimd_mm512_slli_epi64(a, 0x2a);
  easysimd_assert_m512i_i64(r, ==, test_vec[4].r);

  a = test_vec[5].a;
  r = easysimd_mm512_slli_epi64(a, 0x26);
  easysimd_assert_m512i_i64(r, ==, test_vec[5].r);

  a = test_vec[6].a;
  r = easysimd_mm512_slli_epi64(a, 0x18);
  easysimd_assert_m512i_i64(r, ==, test_vec[6].r);

  a = test_vec[7].a;
  r = easysimd_mm512_slli_epi64(a, 0x16);
  easysimd_assert_m512i_i64(r, ==, test_vec[7].r);

  return 0;
}

static int
test_easysimd_mm512_mask_slli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[32];
    const easysimd__mmask32 k;
    const int16_t a[32];
    const uint32_t imm8;
    const int16_t r[32];
  } test_vec[] = {
    { {  INT16_C( 31610), -INT16_C( 15077), -INT16_C( 32756),  INT16_C( 12058),  INT16_C( 21521), -INT16_C( 28036),  INT16_C( 32528),  INT16_C( 30729),
         INT16_C( 29766),  INT16_C(  7505),  INT16_C(   671), -INT16_C(  7920), -INT16_C( 22543),  INT16_C( 31116), -INT16_C( 10240),  INT16_C( 31467),
         INT16_C(  1619),  INT16_C( 24639),  INT16_C( 23174), -INT16_C( 26737),  INT16_C(  2990), -INT16_C( 16855),  INT16_C( 12939), -INT16_C( 11978),
        -INT16_C( 30809),  INT16_C( 18158), -INT16_C(   375),  INT16_C( 31272), -INT16_C( 19290), -INT16_C( 22796), -INT16_C(  8308), -INT16_C(  8160) },
      UINT32_C(1799380965),
      { -INT16_C( 12359),  INT16_C( 26626),  INT16_C( 11226),  INT16_C( 25894),  INT16_C( 23902),  INT16_C(  1335),  INT16_C(  9700),  INT16_C( 28235),
         INT16_C( 29476), -INT16_C( 13592), -INT16_C(  9177), -INT16_C( 19344), -INT16_C( 28485), -INT16_C( 24428), -INT16_C( 11025), -INT16_C( 22261),
         INT16_C(  3491),  INT16_C( 32017),  INT16_C( 14137), -INT16_C( 26653),  INT16_C(  6804),  INT16_C( 31132), -INT16_C(  6337),  INT16_C( 25575),
        -INT16_C( 12453), -INT16_C( 32211), -INT16_C( 25172),  INT16_C( 26422), -INT16_C( 13779),  INT16_C(  7432),  INT16_C(  5022),  INT16_C( 16838) },
      UINT32_C(         9),
      {  INT16_C( 29184), -INT16_C( 15077), -INT16_C( 19456),  INT16_C( 12058),  INT16_C( 21521),  INT16_C( 28160), -INT16_C( 14336), -INT16_C( 27136),
         INT16_C( 18432), -INT16_C( 12288),  INT16_C( 19968), -INT16_C(  8192),  INT16_C( 30208),  INT16_C( 31116), -INT16_C(  8704),  INT16_C( 31467),
         INT16_C(  1619),  INT16_C( 24639),  INT16_C( 23174), -INT16_C( 26737),  INT16_C(  2990), -INT16_C( 16855),  INT16_C( 32256), -INT16_C( 11978),
        -INT16_C( 18944),  INT16_C( 23040), -INT16_C(   375),  INT16_C( 27648), -INT16_C( 19290),  INT16_C(  4096),  INT16_C( 15360), -INT16_C(  8160) } },
    { { -INT16_C( 16425),  INT16_C(  3674), -INT16_C(  3678), -INT16_C( 17245),  INT16_C(  7309),  INT16_C( 29947),  INT16_C( 24323), -INT16_C( 11569),
         INT16_C( 21132),  INT16_C( 10878), -INT16_C(  6520),  INT16_C( 21335),  INT16_C( 29934),  INT16_C(   497),  INT16_C( 13114),  INT16_C(  4386),
         INT16_C( 31986), -INT16_C( 27616), -INT16_C( 15507), -INT16_C(  1456),  INT16_C( 19423), -INT16_C(  7569),  INT16_C( 16042),  INT16_C( 14260),
         INT16_C( 13200),  INT16_C(  6497), -INT16_C( 18407),  INT16_C(  1900),  INT16_C( 23853),  INT16_C( 26376),  INT16_C( 11152), -INT16_C( 32135) },
      UINT32_C( 353802663),
      {  INT16_C( 26204),  INT16_C( 15119),  INT16_C( 32434),  INT16_C( 23581), -INT16_C( 11843),  INT16_C( 19859), -INT16_C(  3068),  INT16_C(  7526),
        -INT16_C( 11603), -INT16_C(  9692),  INT16_C( 11568), -INT16_C( 16319), -INT16_C( 17832), -INT16_C(   189),  INT16_C( 22867), -INT16_C( 20716),
         INT16_C(  9408),  INT16_C( 29418),  INT16_C(  1954),  INT16_C( 24526),  INT16_C( 25305), -INT16_C(  8787),  INT16_C(  4950),  INT16_C(  1019),
         INT16_C(  8166),  INT16_C(  5853),  INT16_C(  8012), -INT16_C( 23338),  INT16_C(  6617),  INT16_C( 11684), -INT16_C( 18317),  INT16_C( 13276) },
      UINT32_C(        13),
      {        INT16_MIN, -INT16_C(  8192),  INT16_C( 16384), -INT16_C( 17245),  INT16_C(  7309),  INT16_C( 24576),  INT16_C( 24323), -INT16_C( 16384),
        -INT16_C( 24576),  INT16_C( 10878), -INT16_C(  6520),  INT16_C(  8192),  INT16_C(     0),  INT16_C(   497),  INT16_C( 13114),        INT16_MIN,
         INT16_C( 31986),  INT16_C( 16384),  INT16_C( 16384), -INT16_C(  1456),  INT16_C(  8192), -INT16_C(  7569),  INT16_C( 16042),  INT16_C( 14260),
        -INT16_C( 16384),  INT16_C(  6497),        INT16_MIN,  INT16_C(  1900),  INT16_C(  8192),  INT16_C( 26376),  INT16_C( 11152), -INT16_C( 32135) } },
    { { -INT16_C( 23097), -INT16_C( 12673), -INT16_C(  8589), -INT16_C( 10841), -INT16_C( 31349), -INT16_C( 24788),  INT16_C( 12160), -INT16_C( 24699),
        -INT16_C( 25843),  INT16_C( 11500), -INT16_C( 28559), -INT16_C( 29947),  INT16_C( 12852), -INT16_C(  4610),  INT16_C( 12559), -INT16_C( 10551),
         INT16_C( 18646),  INT16_C( 18852),  INT16_C( 19495), -INT16_C( 19937),  INT16_C( 19409),  INT16_C( 20817), -INT16_C( 10630), -INT16_C( 30736),
        -INT16_C(  9103), -INT16_C(  7245), -INT16_C( 18067), -INT16_C( 24210),  INT16_C( 27883), -INT16_C(  1394),  INT16_C( 22685),  INT16_C( 29648) },
      UINT32_C(3351016864),
      { -INT16_C(  9279), -INT16_C( 28038), -INT16_C( 13530), -INT16_C( 24093), -INT16_C( 11358),  INT16_C(  4904), -INT16_C(  9040),  INT16_C(  7670),
         INT16_C( 25749), -INT16_C( 32578),  INT16_C( 19920),  INT16_C( 28027),  INT16_C( 19365),  INT16_C( 17888), -INT16_C( 25152), -INT16_C( 32499),
        -INT16_C( 30856), -INT16_C( 24813), -INT16_C(  2478), -INT16_C(  3008),  INT16_C( 26826),  INT16_C( 31240), -INT16_C(   444), -INT16_C(  9833),
         INT16_C( 21859),  INT16_C( 13146), -INT16_C( 10846),  INT16_C( 18337), -INT16_C( 32480), -INT16_C(  7795), -INT16_C( 26082), -INT16_C( 26782) },
      UINT32_C(        12),
      { -INT16_C( 23097), -INT16_C( 12673), -INT16_C(  8589), -INT16_C( 10841), -INT16_C( 31349),        INT16_MIN,  INT16_C( 12160),  INT16_C( 24576),
         INT16_C( 20480),  INT16_C( 11500),  INT16_C(     0), -INT16_C( 29947),  INT16_C( 20480),  INT16_C(     0),  INT16_C(     0), -INT16_C( 10551),
         INT16_C( 18646),  INT16_C( 18852),  INT16_C(  8192),  INT16_C(     0), -INT16_C( 24576),        INT16_MIN, -INT16_C( 10630),  INT16_C( 28672),
         INT16_C( 12288), -INT16_C( 24576),  INT16_C(  8192), -INT16_C( 24210),  INT16_C( 27883), -INT16_C(  1394), -INT16_C(  8192),  INT16_C(  8192) } },
    { {  INT16_C( 13942),  INT16_C( 27763),  INT16_C( 26742), -INT16_C(  8650), -INT16_C( 20368),  INT16_C( 28195), -INT16_C(   953), -INT16_C( 25135),
         INT16_C(  1366),  INT16_C( 11071), -INT16_C( 30810),  INT16_C( 10060),  INT16_C( 11540), -INT16_C( 20922), -INT16_C(  8817),  INT16_C(  1487),
         INT16_C( 16915), -INT16_C( 30350), -INT16_C( 22358),  INT16_C(  6759), -INT16_C( 30119), -INT16_C( 24439),  INT16_C( 23175), -INT16_C(  8899),
         INT16_C( 32095),  INT16_C(  1289),  INT16_C( 21764),  INT16_C(  6189),  INT16_C( 29570),  INT16_C(  4550), -INT16_C( 27312),  INT16_C( 25367) },
      UINT32_C(2196539863),
      {  INT16_C( 21297), -INT16_C( 30052),  INT16_C(  9694),  INT16_C( 25899),  INT16_C( 26752), -INT16_C(  8382),  INT16_C( 19429), -INT16_C(  5659),
         INT16_C(  4768),  INT16_C(  8705), -INT16_C( 14459), -INT16_C( 10956),  INT16_C( 19292),  INT16_C( 13368),  INT16_C(  9428),  INT16_C(  1462),
         INT16_C( 21111),  INT16_C( 21904), -INT16_C( 17544), -INT16_C(  1862), -INT16_C(   733),  INT16_C(  2519), -INT16_C( 17336), -INT16_C(  5646),
        -INT16_C(  2866),  INT16_C( 21259),  INT16_C( 16315),  INT16_C(  6184),  INT16_C( 24714),  INT16_C( 24140),  INT16_C(   644), -INT16_C(   924) },
      UINT32_C(         7),
      { -INT16_C( 26496),  INT16_C( 19968), -INT16_C(  4352), -INT16_C(  8650),  INT16_C( 16384),  INT16_C( 28195), -INT16_C(  3456), -INT16_C(  3456),
         INT16_C( 20480),  INT16_C( 11071), -INT16_C( 30810), -INT16_C( 26112),  INT16_C( 11540), -INT16_C( 20922), -INT16_C(  8817), -INT16_C(  9472),
         INT16_C( 16915), -INT16_C( 30350), -INT16_C( 17408),  INT16_C( 23808), -INT16_C( 30119), -INT16_C(  5248),  INT16_C(  9216), -INT16_C(  1792),
         INT16_C( 32095), -INT16_C( 31360),  INT16_C( 21764),  INT16_C(  6189),  INT16_C( 29570),  INT16_C(  4550), -INT16_C( 27312),  INT16_C( 12800) } },
    { {  INT16_C( 20980), -INT16_C( 20532), -INT16_C( 15348),  INT16_C(  2514), -INT16_C(  9316),  INT16_C( 22609),  INT16_C( 15054), -INT16_C( 15833),
         INT16_C( 31302), -INT16_C( 31363), -INT16_C( 27229),  INT16_C(   784),  INT16_C( 28385), -INT16_C(  7288), -INT16_C( 31534), -INT16_C( 14792),
         INT16_C(  1237), -INT16_C(  7819),  INT16_C( 18633),  INT16_C( 26090),  INT16_C( 15395), -INT16_C(  3651), -INT16_C(  7050), -INT16_C( 17229),
         INT16_C( 12639),  INT16_C(   578),  INT16_C( 21190), -INT16_C( 22523), -INT16_C( 29248), -INT16_C( 27765), -INT16_C( 15599), -INT16_C(  6311) },
      UINT32_C(2445856712),
      { -INT16_C( 19689),  INT16_C( 15094), -INT16_C( 19473),  INT16_C( 25900), -INT16_C(  8296), -INT16_C(  2270),  INT16_C( 25616), -INT16_C( 10247),
        -INT16_C(   330),  INT16_C( 30335),  INT16_C(  2700), -INT16_C( 25335),  INT16_C( 25550), -INT16_C( 27004),  INT16_C( 19762),  INT16_C( 18727),
         INT16_C(  7424), -INT16_C(  4221), -INT16_C( 20528),  INT16_C( 26708),  INT16_C( 30351), -INT16_C( 24737),  INT16_C( 22746), -INT16_C( 28554),
        -INT16_C(  2729), -INT16_C(  7417),  INT16_C(  4096), -INT16_C( 12672),  INT16_C(  1395), -INT16_C( 23196), -INT16_C( 29870),  INT16_C( 21230) },
      UINT32_C(        11),
      {  INT16_C( 20980), -INT16_C( 20532), -INT16_C( 15348),  INT16_C( 24576), -INT16_C(  9316),  INT16_C( 22609),        INT16_MIN, -INT16_C( 14336),
        -INT16_C( 20480), -INT16_C(  2048),  INT16_C( 24576),  INT16_C( 18432),  INT16_C( 28385), -INT16_C(  7288), -INT16_C( 28672),  INT16_C( 14336),
         INT16_C(  1237), -INT16_C(  7819),  INT16_C( 18633), -INT16_C( 24576),  INT16_C( 15395), -INT16_C(  3651), -INT16_C( 12288), -INT16_C( 20480),
        -INT16_C( 18432),  INT16_C(   578),  INT16_C( 21190), -INT16_C( 22523), -INT16_C( 26624), -INT16_C( 27765), -INT16_C( 15599),  INT16_C( 28672) } },
    { {  INT16_C( 16754),  INT16_C(  8568), -INT16_C(  7787),  INT16_C(  3248),  INT16_C( 20544), -INT16_C( 26138),  INT16_C( 30662), -INT16_C( 17168),
        -INT16_C( 11394), -INT16_C( 28996), -INT16_C( 30125),  INT16_C( 22530), -INT16_C( 22546),  INT16_C( 31146), -INT16_C(   874),  INT16_C(  2081),
        -INT16_C( 26307), -INT16_C( 11479), -INT16_C(  9606), -INT16_C( 17441), -INT16_C( 15062), -INT16_C(  4012),  INT16_C( 17468), -INT16_C( 17748),
         INT16_C( 26647),  INT16_C( 27209),  INT16_C( 19442), -INT16_C(  7997),  INT16_C( 28146), -INT16_C( 30631),  INT16_C( 31338), -INT16_C( 22640) },
      UINT32_C(2390407700),
      {  INT16_C( 22932), -INT16_C( 16823), -INT16_C( 25313),  INT16_C( 23470),  INT16_C( 23521), -INT16_C(  2026),  INT16_C( 24515), -INT16_C( 18845),
         INT16_C(  9898), -INT16_C( 25450), -INT16_C(  3949), -INT16_C(   731), -INT16_C( 19094),  INT16_C( 32421),  INT16_C(  8047),  INT16_C(   781),
         INT16_C( 22137), -INT16_C( 26431),  INT16_C( 28916), -INT16_C( 10765),  INT16_C(  2507), -INT16_C( 28978),  INT16_C( 12648),  INT16_C(  4676),
        -INT16_C(  9385), -INT16_C(  5457), -INT16_C( 11061),  INT16_C( 13800), -INT16_C( 29303), -INT16_C(  1612), -INT16_C( 15956),  INT16_C(  9724) },
      UINT32_C(        11),
      {  INT16_C( 16754),  INT16_C(  8568), -INT16_C(  2048),  INT16_C(  3248),  INT16_C(  2048), -INT16_C( 26138),  INT16_C( 30662), -INT16_C( 17168),
        -INT16_C( 11394), -INT16_C( 20480), -INT16_C( 30125),  INT16_C( 10240),  INT16_C( 20480),  INT16_C( 10240), -INT16_C(   874),  INT16_C( 26624),
        -INT16_C( 26307),  INT16_C(  2048), -INT16_C(  9606), -INT16_C( 26624),  INT16_C( 22528),  INT16_C( 28672),  INT16_C( 16384), -INT16_C( 17748),
         INT16_C( 26647),  INT16_C( 30720),  INT16_C( 22528),  INT16_C( 16384),  INT16_C( 28146), -INT16_C( 30631),  INT16_C( 31338), -INT16_C(  8192) } },
    { { -INT16_C( 16962),  INT16_C( 11787), -INT16_C(  7759), -INT16_C( 17671), -INT16_C( 30801), -INT16_C(  8157),  INT16_C( 13772), -INT16_C( 22729),
         INT16_C(  8676), -INT16_C( 18318), -INT16_C( 22775), -INT16_C( 27070),  INT16_C( 15195),  INT16_C(  7235),  INT16_C( 26679), -INT16_C(  2764),
         INT16_C( 16166), -INT16_C( 10461),  INT16_C(  7200), -INT16_C( 12399), -INT16_C( 19292),  INT16_C( 28847), -INT16_C(  6422), -INT16_C( 12777),
        -INT16_C( 30456),  INT16_C(  4487), -INT16_C( 14032), -INT16_C( 29528), -INT16_C(  5372),  INT16_C( 15272), -INT16_C(  9133),  INT16_C( 31025) },
      UINT32_C(1011897372),
      { -INT16_C(  7567),  INT16_C(  5388), -INT16_C( 17514), -INT16_C( 32635), -INT16_C( 25438), -INT16_C( 21937), -INT16_C( 10715),  INT16_C( 21947),
         INT16_C( 25503), -INT16_C( 23583), -INT16_C( 30130), -INT16_C( 23842),  INT16_C(  3942), -INT16_C( 32229),  INT16_C( 27748), -INT16_C( 10817),
        -INT16_C( 13490), -INT16_C(  6934),  INT16_C( 28550),  INT16_C( 10341), -INT16_C( 19445),  INT16_C( 12498), -INT16_C( 29046),  INT16_C( 10629),
         INT16_C( 26609),  INT16_C( 16588), -INT16_C( 21775),  INT16_C( 22498), -INT16_C(   582),  INT16_C(  7898), -INT16_C( 26263), -INT16_C( 18445) },
      UINT32_C(        16),
      { -INT16_C( 16962),  INT16_C( 11787),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  8157),  INT16_C( 13772), -INT16_C( 22729),
         INT16_C(  8676), -INT16_C( 18318),  INT16_C(     0), -INT16_C( 27070),  INT16_C(     0),  INT16_C(  7235),  INT16_C(     0), -INT16_C(  2764),
         INT16_C( 16166), -INT16_C( 10461),  INT16_C(  7200), -INT16_C( 12399),  INT16_C(     0),  INT16_C( 28847),  INT16_C(     0), -INT16_C( 12777),
        -INT16_C( 30456),  INT16_C(  4487),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  9133),  INT16_C( 31025) } },
    { { -INT16_C( 25379),  INT16_C( 19690),  INT16_C(  4865), -INT16_C( 19113), -INT16_C( 30747),  INT16_C( 29503),  INT16_C( 26636),  INT16_C( 29541),
        -INT16_C( 23244), -INT16_C(  8604), -INT16_C( 17273), -INT16_C( 31592), -INT16_C( 18794),  INT16_C( 12270), -INT16_C( 23127), -INT16_C( 31085),
         INT16_C( 32065),  INT16_C( 17106),  INT16_C( 10640),  INT16_C( 30455),  INT16_C( 14000), -INT16_C( 16919),  INT16_C( 20126), -INT16_C( 11728),
        -INT16_C( 27149),  INT16_C( 31409),  INT16_C( 18769), -INT16_C(  6145), -INT16_C(  4864), -INT16_C( 22250), -INT16_C( 22126), -INT16_C( 11216) },
      UINT32_C(3071672870),
      {  INT16_C(  3628), -INT16_C(  9171),  INT16_C(  5700), -INT16_C(  7271), -INT16_C( 13723),  INT16_C( 22709),  INT16_C( 26207), -INT16_C( 20269),
        -INT16_C( 11600), -INT16_C( 20329), -INT16_C( 21057),  INT16_C( 20825), -INT16_C( 30378),  INT16_C( 31781),  INT16_C( 15500), -INT16_C( 18381),
         INT16_C( 24650), -INT16_C( 29036),  INT16_C( 11895), -INT16_C(  9103),  INT16_C( 10232),  INT16_C( 22324),  INT16_C(  1933),  INT16_C( 15623),
        -INT16_C( 24871), -INT16_C( 26387),  INT16_C( 18251), -INT16_C( 24086),  INT16_C(  4048),  INT16_C( 23581),  INT16_C( 20811), -INT16_C( 27372) },
      UINT32_C(         3),
      { -INT16_C( 25379), -INT16_C(  7832), -INT16_C( 19936), -INT16_C( 19113), -INT16_C( 30747), -INT16_C( 14936),  INT16_C( 26636),  INT16_C( 29541),
        -INT16_C( 23244), -INT16_C( 31560), -INT16_C( 17273), -INT16_C( 31592), -INT16_C( 18794),  INT16_C( 12270), -INT16_C( 23127), -INT16_C( 31085),
         INT16_C( 32065),  INT16_C( 29856),  INT16_C( 29624),  INT16_C( 30455),  INT16_C( 16320), -INT16_C( 16919),  INT16_C( 20126), -INT16_C( 11728),
        -INT16_C(  2360), -INT16_C( 14488),  INT16_C( 14936), -INT16_C(  6145),  INT16_C( 32384), -INT16_C(  7960), -INT16_C( 22126), -INT16_C( 22368) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi16(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_slli_epi16(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_slli_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i16x32();
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m512i r = easysimd_mm512_mask_slli_epi16(src, k, a, imm8);

    easysimd_test_x86_write_i16x32(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_mask_slli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const uint32_t imm8;
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(   487225068), -INT32_C(  1598699536),  INT32_C(  1669364414),  INT32_C(  1656521918), -INT32_C(  1489183252),  INT32_C(  1249922223), -INT32_C(   800872464), -INT32_C(  1156990258),
         INT32_C(   567809073), -INT32_C(  1715368485), -INT32_C(   956546808), -INT32_C(  1322731067),  INT32_C(  1163420822), -INT32_C(  1601185360),  INT32_C(  1316017024), -INT32_C(  1106675059) },
      UINT16_C(57998),
      { -INT32_C(  1586533920), -INT32_C(     1870078),  INT32_C(  1706600509), -INT32_C(  1295364518),  INT32_C(   596343443), -INT32_C(  1946809573),  INT32_C(  1661305690),  INT32_C(   574985026),
        -INT32_C(     3951108),  INT32_C(  1795073580), -INT32_C(  1446005169), -INT32_C(  1755604732),  INT32_C(   800778004),  INT32_C(  1287303410),  INT32_C(  1991295028),  INT32_C(  1352201555) },
      UINT32_C(         1),
      {  INT32_C(   487225068), -INT32_C(     3740156), -INT32_C(   881766278),  INT32_C(  1704238260), -INT32_C(  1489183252),  INT32_C(  1249922223), -INT32_C(   800872464),  INT32_C(  1149970052),
         INT32_C(   567809073), -INT32_C(   704820136), -INT32_C(   956546808), -INT32_C(  1322731067),  INT32_C(  1163420822), -INT32_C(  1720360476), -INT32_C(   312377240), -INT32_C(  1590564186) } },
    { {  INT32_C(    47664987),  INT32_C(    55656781), -INT32_C(  1442317808),  INT32_C(  1019125333),  INT32_C(   154135897),  INT32_C(  1732082599),  INT32_C(   565949227), -INT32_C(  1479865525),
        -INT32_C(  1482055078), -INT32_C(   190121245),  INT32_C(  1235137012), -INT32_C(  1467589553), -INT32_C(   256723639),  INT32_C(  1549332272), -INT32_C(   293792861),  INT32_C(  2023049246) },
      UINT16_C(16107),
      { -INT32_C(   919024097),  INT32_C(  1618619586), -INT32_C(    54736522),  INT32_C(   615581042),  INT32_C(  1309925878), -INT32_C(  1168001475),  INT32_C(   973307813),  INT32_C(   377089783),
         INT32_C(  2145366460),  INT32_C(  1423923678), -INT32_C(  1806656478), -INT32_C(  1749548639),  INT32_C(   551931106),  INT32_C(   668616322), -INT32_C(  1134371387), -INT32_C(  1999447093) },
      UINT32_C(        21),
      { -INT32_C(  1008730112), -INT32_C(  1740636160), -INT32_C(  1442317808), -INT32_C(  1371537408),  INT32_C(   154135897), -INT32_C(   945815552), -INT32_C(   190840832), -INT32_C(   555745280),
        -INT32_C(  1482055078),  INT32_C(  1002438656), -INT32_C(  2076180480),  INT32_C(   874512384), -INT32_C(  1673527296), -INT32_C(   801112064), -INT32_C(   293792861),  INT32_C(  2023049246) } },
    { {  INT32_C(   174720945), -INT32_C(  2110931226),  INT32_C(   254066958),  INT32_C(  1190312826), -INT32_C(   406318431), -INT32_C(   894570260),  INT32_C(   764766546), -INT32_C(   306635460),
         INT32_C(   200811556), -INT32_C(   242407966),  INT32_C(  1610658278),  INT32_C(   245822061), -INT32_C(   235573500), -INT32_C(  1313103265),  INT32_C(  1222529036), -INT32_C(  1841981586) },
      UINT16_C(11707),
      {  INT32_C(   726834845), -INT32_C(  1881392753),  INT32_C(  1082280345),  INT32_C(  1303348823),  INT32_C(   854527607), -INT32_C(  1652294721),  INT32_C(  2033512771),  INT32_C(   564588675),
         INT32_C(   491583886), -INT32_C(   877909966), -INT32_C(   922014094),  INT32_C(   739687093), -INT32_C(  2023815480),  INT32_C(  1160045057),  INT32_C(  1455315411), -INT32_C(   680041399) },
      UINT32_C(        23),
      {  INT32_C(  1317011456), -INT32_C(   947912704),  INT32_C(   254066958),  INT32_C(   729808896),  INT32_C(   998244352), -INT32_C(   545259520),  INT32_C(   764766546),  INT32_C(  1098907648),
        -INT32_C(   956301312), -INT32_C(   242407966),  INT32_C(   956301312),  INT32_C(  1518338048), -INT32_C(   235573500),  INT32_C(     8388608),  INT32_C(  1222529036), -INT32_C(  1841981586) } },
    { { -INT32_C(   326110013), -INT32_C(   815899744),  INT32_C(   562308966),  INT32_C(  1156165694), -INT32_C(   247107313), -INT32_C(   289109355), -INT32_C(  1388897464),  INT32_C(  1443499666),
        -INT32_C(  1555916286),  INT32_C(  1534238965),  INT32_C(    92075719), -INT32_C(  1236703578),  INT32_C(  1822920663),  INT32_C(  1650158617),  INT32_C(   420450951), -INT32_C(  1553000031) },
      UINT16_C(45491),
      { -INT32_C(  1202608058), -INT32_C(  2136073980),  INT32_C(  1759925534), -INT32_C(  1292387061), -INT32_C(  2061627350), -INT32_C(  2129156750),  INT32_C(   781957311), -INT32_C(  1562358180),
        -INT32_C(    77909513),  INT32_C(  1769670986),  INT32_C(  1775329886),  INT32_C(  1226623007),  INT32_C(  1271806680), -INT32_C(  1613961504), -INT32_C(    87201890), -INT32_C(  1382175306) },
      UINT32_C(        26),
      {  INT32_C(   402653184),  INT32_C(   268435456),  INT32_C(   562308966),  INT32_C(  1156165694), -INT32_C(  1476395008), -INT32_C(   939524096), -INT32_C(  1388897464),  INT32_C(  1879048192),
        -INT32_C(   603979776),  INT32_C(  1534238965),  INT32_C(    92075719), -INT32_C(  1236703578),  INT32_C(  1610612736),              INT32_MIN,  INT32_C(   420450951), -INT32_C(   671088640) } },
    { {  INT32_C(    19507448), -INT32_C(  2057268701),  INT32_C(   732285283),  INT32_C(   537194213), -INT32_C(  1560260675), -INT32_C(  2092851429),  INT32_C(   439958636), -INT32_C(   772151591),
        -INT32_C(  1311628658),  INT32_C(   406270645), -INT32_C(   515580676), -INT32_C(  2029958966), -INT32_C(  1305869929),  INT32_C(   221605024), -INT32_C(  2128122200), -INT32_C(   497934252) },
      UINT16_C( 9282),
      { -INT32_C(   883427436),  INT32_C(  1420251920),  INT32_C(   916222260),  INT32_C(   590820345),  INT32_C(   462412005),  INT32_C(   210319333),  INT32_C(   170712760),  INT32_C(  1412394688),
         INT32_C(  1981777510),  INT32_C(   231392985),  INT32_C(   826500664),  INT32_C(  2119465881),  INT32_C(   949609555), -INT32_C(   733732069), -INT32_C(  1042386943),  INT32_C(  1142296030) },
      UINT32_C(        35),
      {  INT32_C(    19507448),  INT32_C(           0),  INT32_C(   732285283),  INT32_C(   537194213), -INT32_C(  1560260675), -INT32_C(  2092851429),  INT32_C(           0), -INT32_C(   772151591),
        -INT32_C(  1311628658),  INT32_C(   406270645),  INT32_C(           0), -INT32_C(  2029958966), -INT32_C(  1305869929),  INT32_C(           0), -INT32_C(  2128122200), -INT32_C(   497934252) } },
    { { -INT32_C(    59983307), -INT32_C(   365659516),  INT32_C(   948135357), -INT32_C(  1651834183), -INT32_C(  1095122021),  INT32_C(  2009042183),  INT32_C(  2035712363), -INT32_C(   871589225),
        -INT32_C(   641173163), -INT32_C(  1329267469),  INT32_C(   468272993), -INT32_C(   440896439),  INT32_C(  1067675960),  INT32_C(  1790403582),  INT32_C(  2078477796), -INT32_C(    62394457) },
      UINT16_C( 4200),
      { -INT32_C(  1710466090), -INT32_C(   169775604),  INT32_C(  1097411465), -INT32_C(  1280073456), -INT32_C(  1743343135), -INT32_C(    22676965), -INT32_C(  1108456331),  INT32_C(   533550665),
        -INT32_C(  1095116366), -INT32_C(   793535929), -INT32_C(   720233275), -INT32_C(  1618360898), -INT32_C(  1841848458),  INT32_C(   261151898),  INT32_C(  1909227048), -INT32_C(  2020566571) },
      UINT32_C(         0),
      { -INT32_C(    59983307), -INT32_C(   365659516),  INT32_C(   948135357), -INT32_C(  1280073456), -INT32_C(  1095122021), -INT32_C(    22676965), -INT32_C(  1108456331), -INT32_C(   871589225),
        -INT32_C(   641173163), -INT32_C(  1329267469),  INT32_C(   468272993), -INT32_C(   440896439), -INT32_C(  1841848458),  INT32_C(  1790403582),  INT32_C(  2078477796), -INT32_C(    62394457) } },
    { { -INT32_C(   457554615),  INT32_C(   346655736),  INT32_C(  1657962397), -INT32_C(  1478987512), -INT32_C(  2076087640),  INT32_C(  2041401851), -INT32_C(  1219617251), -INT32_C(   131410513),
         INT32_C(   316466202),  INT32_C(   203916911),  INT32_C(   225376517),  INT32_C(   330581867), -INT32_C(  1399261519),  INT32_C(  1680229703),  INT32_C(   303789155),  INT32_C(  1678460234) },
      UINT16_C(59178),
      { -INT32_C(  1636984457),  INT32_C(   345469606),  INT32_C(   878379647), -INT32_C(  1372975850), -INT32_C(   537693767), -INT32_C(   246130987),  INT32_C(  1949736553),  INT32_C(  2036031490),
        -INT32_C(  1609053702), -INT32_C(  1145786565), -INT32_C(   940633935), -INT32_C(   696968931),  INT32_C(  1605789834),  INT32_C(   676334271), -INT32_C(  1432582232), -INT32_C(   517670937) },
      UINT32_C(        33),
      { -INT32_C(   457554615),  INT32_C(           0),  INT32_C(  1657962397),  INT32_C(           0), -INT32_C(  2076087640),  INT32_C(           0), -INT32_C(  1219617251), -INT32_C(   131410513),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   330581867), -INT32_C(  1399261519),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { { -INT32_C(   352550597),  INT32_C(  1184675638), -INT32_C(  1084005466),  INT32_C(  1095317977), -INT32_C(   117397265), -INT32_C(  2136921863), -INT32_C(  1134080828), -INT32_C(  1417918096),
         INT32_C(     9861322), -INT32_C(   716819921),  INT32_C(  1855236501), -INT32_C(   760160797),  INT32_C(  2144120966), -INT32_C(  1644204583),  INT32_C(   693724857),  INT32_C(  2060834479) },
      UINT16_C(27470),
      { -INT32_C(  1046643078), -INT32_C(   395693485),  INT32_C(  1371884961), -INT32_C(   335459552), -INT32_C(   916858166), -INT32_C(   785378440),  INT32_C(   296214332), -INT32_C(   730007975),
        -INT32_C(   946529932), -INT32_C(   307232948),  INT32_C(  1832809805), -INT32_C(  1957085248), -INT32_C(  1839943142),  INT32_C(    40076485), -INT32_C(  1122825372),  INT32_C(  1972473857) },
      UINT32_C(        10),
      { -INT32_C(   352550597), -INT32_C(  1463202816),  INT32_C(   355894272),  INT32_C(    86802432), -INT32_C(   117397265), -INT32_C(  2136921863), -INT32_C(  1619202048), -INT32_C(  1417918096),
         INT32_C(  1415958528), -INT32_C(  1073926144),  INT32_C(  1855236501),  INT32_C(  1694433280),  INT32_C(  2144120966), -INT32_C(  1911352320),  INT32_C(  1278054400),  INT32_C(  2060834479) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_slli_epi32(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_slli_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m512i r = easysimd_mm512_mask_slli_epi32(src, k, a, imm8);

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_mask_slli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[8];
    const easysimd__mmask8 k;
    const int64_t a[8];
    const uint32_t imm8;
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 3994081184337448464), -INT64_C( 8005827586047219994),  INT64_C( 8944896926035028382),  INT64_C( 5933978777617984166),
        -INT64_C( 7715874874127361272), -INT64_C( 4511302651941712742),  INT64_C( 9197643297923084747),  INT64_C(  546644188527543297) },
      UINT8_C(236),
      {  INT64_C( 3686810542176648156), -INT64_C( 7757424641805189397),  INT64_C( 7504665093568626828), -INT64_C( 2620409988620260955),
         INT64_C( 6126186778638171177), -INT64_C(  104643326348469611), -INT64_C( 8266558426002031941),  INT64_C( 5083408515796807345) },
      UINT32_C(        42),
      {  INT64_C( 3994081184337448464), -INT64_C( 8005827586047219994), -INT64_C( 5043415856143400960), -INT64_C( 7086814235899985920),
        -INT64_C( 7715874874127361272), -INT64_C( 8544924984005361664), -INT64_C( 8364613873143119872),  INT64_C( 1061376166478217216) } },
    { {  INT64_C( 8440837630799320823), -INT64_C( 6721088459174896701), -INT64_C( 8214257993155700728),  INT64_C( 3774281728805024090),
        -INT64_C( 8027695145845331193),  INT64_C( 7730789828208262656),  INT64_C( 4922360634181145568), -INT64_C(  946068270383629715) },
      UINT8_C(207),
      {  INT64_C( 2991845543138497696),  INT64_C( 5324090895237983418), -INT64_C( 1378531453268876364),  INT64_C(  481543653854513404),
         INT64_C( 3437109037618645726), -INT64_C( 1370197952887177156), -INT64_C( 6587293284377481033),  INT64_C( 1382378814537231555) },
      UINT32_C(        60),
      {  INT64_C(                   0), -INT64_C( 6917529027641081856),  INT64_C( 4611686018427387904), -INT64_C( 4611686018427387904),
        -INT64_C( 8027695145845331193),  INT64_C( 7730789828208262656),  INT64_C( 8070450532247928832),  INT64_C( 3458764513820540928) } },
    { { -INT64_C( 8068725051501666886), -INT64_C( 1936910998987818669), -INT64_C( 4030666294477076718),  INT64_C( 2478406312220506550),
        -INT64_C( 3778816435830368338),  INT64_C( 2109180010281719467), -INT64_C( 4955232801161293033), -INT64_C(  559281664339789389) },
      UINT8_C( 95),
      { -INT64_C( 2681995265260849623),  INT64_C( 4629141234322385110),  INT64_C( 2782099691612254574),  INT64_C( 2165220925227710372),
         INT64_C( 6754276323788252254),  INT64_C( 1502164496490471978), -INT64_C( 4429685927228727949), -INT64_C( 4871915826574330288) },
      UINT32_C(        39),
      {  INT64_C( 7816864118219997184),  INT64_C( 8292370461640097792),  INT64_C( 2510958002886934528),  INT64_C( 4061070386446467072),
         INT64_C( 2121810051491430400),  INT64_C( 2109180010281719467),  INT64_C( 9083401457859821568), -INT64_C(  559281664339789389) } },
    { { -INT64_C(  159240775997677922),  INT64_C( 6340425308157731271), -INT64_C( 2694217023399567948), -INT64_C( 2053934702792154715),
         INT64_C( 8064906420681925780), -INT64_C(  342284257945087826), -INT64_C( 8065132283771177901), -INT64_C( 3103491141091177530) },
      UINT8_C(147),
      {  INT64_C( 4642274091506320241),  INT64_C( 5182943416149300388), -INT64_C( 9046109632573856309), -INT64_C( 2362534453076393869),
        -INT64_C( 4590845754597346982),  INT64_C( 2321966616861659742), -INT64_C( 6762641719519384467), -INT64_C( 2732704762615699259) },
      UINT32_C(        27),
      { -INT64_C( 8795004387450683392), -INT64_C( 7146022227849248768), -INT64_C( 2694217023399567948), -INT64_C( 2053934702792154715),
        -INT64_C( 2281940129742323712), -INT64_C(  342284257945087826), -INT64_C( 8065132283771177901), -INT64_C(  458752233673064448) } },
    { {  INT64_C(   79315675294327985), -INT64_C( 5631339818507695324), -INT64_C( 8408076865379467963),  INT64_C( 5201628532447085767),
         INT64_C( 2736250142897420489),  INT64_C(  801195179897868072),  INT64_C( 5194591564757073639), -INT64_C( 3510069899600125591) },
      UINT8_C(242),
      {  INT64_C( 1266102763758274637),  INT64_C( 2160849440562610346), -INT64_C( 4699815597576386697),  INT64_C( 6709680717514983281),
        -INT64_C( 3157487151805427365),  INT64_C( 6743549894602589274), -INT64_C( 8374303757269650579),  INT64_C( 8913982793553775686) },
      UINT32_C(        30),
      {  INT64_C(   79315675294327985), -INT64_C( 8830010574365523968), -INT64_C( 8408076865379467963),  INT64_C( 5201628532447085767),
        -INT64_C( 8686946329353519104), -INT64_C( 7045247102175150080), -INT64_C( 9030195698275975168), -INT64_C( 3031517309849042944) } },
    { { -INT64_C( 3877347736519993220),  INT64_C( 8151529888494430413),  INT64_C( 9118971149828172487),  INT64_C( 1479460620945253582),
         INT64_C( 5278529355850094513), -INT64_C( 4147219325767585085), -INT64_C( 2037709415426163129),  INT64_C( 2317961203982620286) },
      UINT8_C(165),
      { -INT64_C( 3750537683133229291),  INT64_C( 7262777218572153958),  INT64_C( 1420424700247999646),  INT64_C(  978587304316882146),
        -INT64_C( 8872319835149847237), -INT64_C( 2814800093217576668), -INT64_C( 6365900996386723626),  INT64_C(  619011635580119699) },
      UINT32_C(        53),
      { -INT64_C( 2116691824864133120),  INT64_C( 8151529888494430413), -INT64_C( 7800234554605699072),  INT64_C( 1479460620945253582),
         INT64_C( 5278529355850094513),  INT64_C( 2630102182384369664), -INT64_C( 2037709415426163129),  INT64_C( 5935744308874313728) } },
    { { -INT64_C( 6545836944239223596),  INT64_C(  735134171673899914), -INT64_C( 7556347011837999416),  INT64_C( 1736257796122717197),
         INT64_C( 2605971860052842122), -INT64_C( 8456155076921128200), -INT64_C(  540647545880611718), -INT64_C( 2646911167516056344) },
      UINT8_C( 94),
      { -INT64_C( 7665376654617190047),  INT64_C( 6037887699197049500),  INT64_C( 1449937185749774843), -INT64_C( 1077132650509612186),
        -INT64_C( 4772910564427430494), -INT64_C(  530061373153141903),  INT64_C( 9209384160407315081),  INT64_C( 1736765980692853913) },
      UINT32_C(        25),
      { -INT64_C( 6545836944239223596), -INT64_C( 4710573391755608064),  INT64_C( 6948629143361683456),  INT64_C( 2259906061072859136),
         INT64_C( 4782031204280958976), -INT64_C( 8456155076921128200),  INT64_C( 3707130743294525440), -INT64_C( 2646911167516056344) } },
    { {  INT64_C( 2619720636165080754),  INT64_C( 3642315125293165356),  INT64_C( 5202161234392860233),  INT64_C( 2770792890823957240),
         INT64_C( 2254451780518728731), -INT64_C( 3559902517957334250), -INT64_C( 6462216121345448971), -INT64_C(   32283978931564814) },
      UINT8_C( 33),
      { -INT64_C( 2801463895526202211),  INT64_C(   70417247333418326),  INT64_C( 7069898611016663955), -INT64_C( 5792959450507213779),
         INT64_C( 4157341304633208938), -INT64_C( 3496701803293175972), -INT64_C( 6344983109842622917), -INT64_C( 1630940237764937572) },
      UINT32_C(        55),
      {  INT64_C( 5656521131977342976),  INT64_C( 3642315125293165356),  INT64_C( 5202161234392860233),  INT64_C( 2770792890823957240),
         INT64_C( 2254451780518728731), -INT64_C( 5908722711110090752), -INT64_C( 6462216121345448971), -INT64_C(   32283978931564814) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi64(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_slli_epi64(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_mask_slli_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i64x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m512i r = easysimd_mm512_mask_slli_epi64(src, k, a, imm8);

    easysimd_test_x86_write_i64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_maskz_slli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask32 k;
    const int16_t a[32];
    const uint32_t imm8;
    const int16_t r[32];
  } test_vec[] = {
    { UINT32_C(4242409693),
      {  INT16_C( 31900),  INT16_C( 31584), -INT16_C( 20027), -INT16_C( 31412), -INT16_C( 17108),  INT16_C( 12579),  INT16_C( 13665), -INT16_C(  1319),
        -INT16_C( 26161), -INT16_C( 30516), -INT16_C(  9896),  INT16_C( 32587), -INT16_C( 25440),  INT16_C( 32347),  INT16_C( 14756),  INT16_C( 16506),
        -INT16_C(  9291),  INT16_C( 31420),  INT16_C(  2188), -INT16_C( 18177),  INT16_C(  8902),  INT16_C( 10218), -INT16_C( 15528),  INT16_C( 10018),
        -INT16_C(  4516), -INT16_C( 19024), -INT16_C(  1081),  INT16_C( 26676), -INT16_C( 28777),  INT16_C( 15590),  INT16_C( 24776),  INT16_C( 32124) },
      UINT32_C(         2),
      { -INT16_C(  3472),  INT16_C(     0), -INT16_C( 14572),  INT16_C(  5424), -INT16_C(  2896),  INT16_C(     0), -INT16_C( 10876), -INT16_C(  5276),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(   724),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C(  5392),  INT16_C(  8752), -INT16_C(  7172), -INT16_C( 29928),  INT16_C(     0),  INT16_C(  3424), -INT16_C( 25464),
         INT16_C(     0),  INT16_C(     0), -INT16_C(  4324), -INT16_C( 24368),  INT16_C( 15964), -INT16_C(  3176), -INT16_C( 31968), -INT16_C(  2576) } },
    { UINT32_C(1103623992),
      { -INT16_C( 32521),  INT16_C(  6407),  INT16_C( 11882),  INT16_C( 11633), -INT16_C( 26288),  INT16_C( 16009),  INT16_C( 15945),  INT16_C( 17414),
         INT16_C( 28275),  INT16_C(   731),  INT16_C(  5972), -INT16_C( 19253),  INT16_C( 18580), -INT16_C( 13072), -INT16_C( 18624),  INT16_C( 14093),
         INT16_C(  5175), -INT16_C( 24240), -INT16_C( 15805), -INT16_C( 27698),  INT16_C( 22619), -INT16_C( 23342), -INT16_C( 10090),  INT16_C(  2536),
        -INT16_C( 15546), -INT16_C( 26100), -INT16_C( 10277),  INT16_C( 28494),  INT16_C( 15903),  INT16_C( 24379),  INT16_C( 18934),  INT16_C( 11670) },
      UINT32_C(         5),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 20960),  INT16_C( 10752), -INT16_C( 12000),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 12704),  INT16_C( 23392), -INT16_C(  5504),  INT16_C(     0),  INT16_C(  4736), -INT16_C( 25088), -INT16_C(  6144), -INT16_C(  7776),
        -INT16_C( 31008),  INT16_C( 10752),  INT16_C( 18528),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4800),  INT16_C( 15616),
         INT16_C( 26816),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 16064),  INT16_C(     0) } },
    { UINT32_C(2845888487),
      {  INT16_C( 13469), -INT16_C(  2812), -INT16_C( 22522), -INT16_C(  8564), -INT16_C( 27248),  INT16_C( 21284), -INT16_C( 16735),  INT16_C( 30766),
        -INT16_C( 25332),  INT16_C( 19352), -INT16_C(  2087),  INT16_C(  8769),  INT16_C( 28302),  INT16_C( 30079),  INT16_C(  8253), -INT16_C(  9442),
         INT16_C(  8788),  INT16_C( 23248),  INT16_C( 23754),  INT16_C( 23096),  INT16_C( 23794), -INT16_C( 27731), -INT16_C(  9190),  INT16_C(  9740),
        -INT16_C( 23431),  INT16_C( 21105), -INT16_C( 19813),  INT16_C( 10612), -INT16_C(  3039),  INT16_C( 24222), -INT16_C( 17388),  INT16_C( 26681) },
      UINT32_C(        19),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(1722335754),
      {  INT16_C(   762),  INT16_C( 22104), -INT16_C(  4944), -INT16_C( 29584), -INT16_C( 26888), -INT16_C( 25595),  INT16_C( 22536), -INT16_C( 17865),
         INT16_C( 25036), -INT16_C( 16165),  INT16_C( 15103), -INT16_C( 17196),  INT16_C( 15475),  INT16_C( 32154),  INT16_C( 17406), -INT16_C(  1820),
         INT16_C( 15429), -INT16_C(  2738), -INT16_C( 16856),  INT16_C(  8321), -INT16_C( 30891),  INT16_C( 23996), -INT16_C(  2849), -INT16_C( 21737),
        -INT16_C(  3243),  INT16_C( 21612),  INT16_C( 16429), -INT16_C( 24560), -INT16_C( 21635),  INT16_C( 31518),  INT16_C(   750),  INT16_C( 13172) },
      UINT32_C(         3),
      {  INT16_C(     0), -INT16_C( 19776),  INT16_C(     0),  INT16_C( 25472),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(  1752),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8176), -INT16_C( 14560),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1032),  INT16_C(     0), -INT16_C(  4640),  INT16_C(     0),  INT16_C( 22712),
         INT16_C(     0), -INT16_C( 23712),  INT16_C(   360),  INT16_C(     0),  INT16_C(     0), -INT16_C( 10000),  INT16_C(  6000),  INT16_C(     0) } },
    { UINT32_C(2171021762),
      { -INT16_C( 30806),  INT16_C( 12758),  INT16_C( 13124),  INT16_C( 14352), -INT16_C( 17334),  INT16_C( 15757), -INT16_C(  7896),  INT16_C( 26730),
         INT16_C(  3058), -INT16_C( 25115),  INT16_C( 24873),  INT16_C( 11147), -INT16_C( 16683), -INT16_C( 26775), -INT16_C( 12057), -INT16_C( 28136),
        -INT16_C(  4520), -INT16_C( 25405), -INT16_C( 11231),  INT16_C( 27860),  INT16_C( 24976), -INT16_C( 18263),  INT16_C(  5186),  INT16_C( 13344),
         INT16_C(  1567),  INT16_C( 18641),  INT16_C( 23655),  INT16_C( 15475), -INT16_C(  9189),  INT16_C(   723), -INT16_C(  4947),  INT16_C(  1428) },
      UINT32_C(         9),
      {  INT16_C(     0), -INT16_C( 21504),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20480), -INT16_C( 11264),
        -INT16_C(  7168),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5632),  INT16_C(     0), -INT16_C( 11776),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 20480), -INT16_C( 31232),  INT16_C( 16896),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20992), -INT16_C( 31744),  INT16_C(     0),
         INT16_C( 15872),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10240) } },
    { UINT32_C( 754753880),
      {  INT16_C( 26741), -INT16_C( 10564),  INT16_C( 29713),  INT16_C(  9496),  INT16_C( 19860), -INT16_C( 26044), -INT16_C( 29666),  INT16_C( 31489),
         INT16_C( 15871), -INT16_C(  9066), -INT16_C( 26607), -INT16_C(   631), -INT16_C( 29139), -INT16_C( 31273), -INT16_C( 11473), -INT16_C( 23375),
         INT16_C( 27963),  INT16_C( 19834), -INT16_C( 27935),  INT16_C( 30066), -INT16_C( 18465), -INT16_C(   496),  INT16_C(  4419),  INT16_C( 17273),
         INT16_C(  3919),  INT16_C( 24607), -INT16_C( 22361), -INT16_C( 11171),  INT16_C( 13366),  INT16_C( 25945),  INT16_C(  2568),  INT16_C( 17161) },
      UINT32_C(         2),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 27552),  INT16_C( 13904),  INT16_C(     0),  INT16_C( 12408),  INT16_C(     0),
        -INT16_C(  2052),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5980),  INT16_C(     0), -INT16_C( 27964),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 19332), -INT16_C( 10808), -INT16_C(  8324), -INT16_C(  1984),  INT16_C( 17676),  INT16_C(  3556),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 23908),  INT16_C( 20852),  INT16_C(     0), -INT16_C( 27292),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C( 358125699),
      { -INT16_C( 12797), -INT16_C( 17675), -INT16_C(  3106), -INT16_C(  4099),  INT16_C( 16492),  INT16_C( 31550), -INT16_C( 24993),  INT16_C(  1826),
        -INT16_C(  2053),  INT16_C( 12349), -INT16_C( 23984),  INT16_C( 23352),  INT16_C( 31659),  INT16_C( 11986),  INT16_C( 11020),  INT16_C(  3908),
         INT16_C( 14841), -INT16_C( 10295), -INT16_C( 14804), -INT16_C( 26426),  INT16_C(  1287),  INT16_C( 26131),  INT16_C( 13731), -INT16_C( 24722),
        -INT16_C( 21716),  INT16_C( 32207),  INT16_C(  1870), -INT16_C(  1576), -INT16_C( 21886), -INT16_C( 29144),  INT16_C( 27861), -INT16_C( 12643) },
      UINT32_C(         0),
      { -INT16_C( 12797), -INT16_C( 17675),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  1826),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31659),  INT16_C(     0),  INT16_C(     0),  INT16_C(  3908),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 26426),  INT16_C(  1287),  INT16_C(     0),  INT16_C( 13731),  INT16_C(     0),
        -INT16_C( 21716),  INT16_C(     0),  INT16_C(  1870),  INT16_C(     0), -INT16_C( 21886),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C( 768714086),
      {  INT16_C( 26988),  INT16_C( 28980), -INT16_C( 25988), -INT16_C( 20204), -INT16_C( 19704), -INT16_C( 19234),  INT16_C( 23426), -INT16_C( 30462),
        -INT16_C(  1229), -INT16_C(  8948), -INT16_C( 26077), -INT16_C( 28749), -INT16_C( 32456), -INT16_C( 25036),  INT16_C(  1319), -INT16_C( 27701),
        -INT16_C(   146), -INT16_C(  5628),  INT16_C(  6298), -INT16_C( 23908),  INT16_C( 31436),  INT16_C( 20054),  INT16_C( 22741),  INT16_C(  2264),
        -INT16_C(  7084),  INT16_C( 30693), -INT16_C( 26498), -INT16_C( 18937),  INT16_C( 15130),  INT16_C( 16725),  INT16_C(  8257), -INT16_C( 20524) },
      UINT32_C(         0),
      {  INT16_C(     0),  INT16_C( 28980), -INT16_C( 25988),  INT16_C(     0),  INT16_C(     0), -INT16_C( 19234),  INT16_C( 23426),  INT16_C(     0),
        -INT16_C(  1229),  INT16_C(     0), -INT16_C( 26077),  INT16_C(     0),  INT16_C(     0), -INT16_C( 25036),  INT16_C(     0), -INT16_C( 27701),
        -INT16_C(   146),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31436),  INT16_C(     0),  INT16_C( 22741),  INT16_C(  2264),
        -INT16_C(  7084),  INT16_C(     0), -INT16_C( 26498), -INT16_C( 18937),  INT16_C(     0),  INT16_C( 16725),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_slli_epi16(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_slli_epi16");
    easysimd_test_x86_assert_equal_i16x32(r, easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    // uint32_t imm8 = easysimd_test_x86_random_mmask32();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m512i r = easysimd_mm512_maskz_slli_epi16(k, a, imm8);

    easysimd_test_codegen_write_u32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_maskz_slli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int32_t a[16];
    const uint32_t imm8;
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(19064),
      {  INT32_C(  1797124935),  INT32_C(   993560555),  INT32_C(   434934181), -INT32_C(  1618757544),  INT32_C(  1996373682),  INT32_C(  2048309867), -INT32_C(  1460845680),  INT32_C(   921855470),
         INT32_C(  1151406168),  INT32_C(  1015011735), -INT32_C(  1017811861), -INT32_C(   698165212), -INT32_C(  2108858345), -INT32_C(  1426300134), -INT32_C(  1789728346), -INT32_C(  1496627890) },
      UINT32_C(        14),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   300548096), -INT32_C(  1884520448), -INT32_C(  1365590016),  INT32_C(  1357119488),  INT32_C(           0),
         INT32_C(           0), -INT32_C(   161103872),  INT32_C(           0), -INT32_C(  1240924160),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1167491072),  INT32_C(           0) } },
    { UINT16_C(60012),
      {  INT32_C(   694830828),  INT32_C(  1954469553), -INT32_C(   774416390), -INT32_C(   266455186), -INT32_C(    51609262), -INT32_C(  1135618777), -INT32_C(   913861597), -INT32_C(   709561111),
        -INT32_C(   335667654), -INT32_C(   295731724),  INT32_C(  1119827668), -INT32_C(  1070408338), -INT32_C(  2034491810),  INT32_C(   373427187), -INT32_C(  1998599521), -INT32_C(   128019522) },
      UINT32_C(        21),
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(    12582912), -INT32_C(   306184192),  INT32_C(           0), -INT32_C(  1528823808), -INT32_C(  2074083328),  INT32_C(           0),
         INT32_C(           0), -INT32_C(  1098907648),  INT32_C(           0), -INT32_C(  1379926016),  INT32_C(           0),  INT32_C(  2120220672),  INT32_C(  1407188992),  INT32_C(  2009071616) } },
    { UINT16_C(58204),
      { -INT32_C(  1807558235), -INT32_C(   279742035), -INT32_C(  1490996761),  INT32_C(   358825871),  INT32_C(  1213755186), -INT32_C(  1758977522), -INT32_C(   654984224),  INT32_C(   314265965),
        -INT32_C(   660144853),  INT32_C(  1573452406), -INT32_C(  1174083285),  INT32_C(  1540384809), -INT32_C(   442226473),  INT32_C(   712821834), -INT32_C(   201166457), -INT32_C(   301482557) },
      UINT32_C(         4),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1913855600),  INT32_C(  1446246640), -INT32_C(  2054753504),  INT32_C(           0), -INT32_C(  1889812992),  INT32_C(           0),
        -INT32_C(  1972383056), -INT32_C(   594565280),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1479752544),  INT32_C(  1076303984), -INT32_C(   528753616) } },
    { UINT16_C(50861),
      { -INT32_C(  1903253711), -INT32_C(  1919715118), -INT32_C(    60949599), -INT32_C(  1214216750), -INT32_C(   114004785), -INT32_C(   386161164),  INT32_C(  1458550888), -INT32_C(  1558340238),
         INT32_C(   389196612), -INT32_C(   995834589), -INT32_C(  1832844864),  INT32_C(  1430872454), -INT32_C(  1035043378), -INT32_C(  1951774429),  INT32_C(  1944230401),  INT32_C(  2065170231) },
      UINT32_C(         9),
      {  INT32_C(   491676160),  INT32_C(           0), -INT32_C(  1141423616),  INT32_C(  1091281920),  INT32_C(           0), -INT32_C(   146020352),  INT32_C(           0),  INT32_C(   993715200),
         INT32_C(           0),  INT32_C(  1233798656), -INT32_C(  2113699840),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   986447360),  INT32_C(   805203456) } },
    { UINT16_C(37449),
      { -INT32_C(  1841951026),  INT32_C(  1616066510),  INT32_C(   313111997),  INT32_C(  1163929475), -INT32_C(   705713846),  INT32_C(   515344810),  INT32_C(  1010153152),  INT32_C(   802061921),
         INT32_C(  1522599308), -INT32_C(    88468164),  INT32_C(  1292657610), -INT32_C(   728601206), -INT32_C(  1045790442), -INT32_C(   874553077),  INT32_C(  2030507032),  INT32_C(   514381458) },
      UINT32_C(         7),
      {  INT32_C(   453469952),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1340882560),  INT32_C(           0),  INT32_C(           0),  INT32_C(   450584576),  INT32_C(           0),
         INT32_C(           0),  INT32_C(  1560976896),  INT32_C(           0),  INT32_C(           0), -INT32_C(   717190400),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1416317184) } },
    { UINT16_C(30825),
      {  INT32_C(   288521751), -INT32_C(  1793157816), -INT32_C(   198735072),  INT32_C(  1654565281),  INT32_C(  2134966452), -INT32_C(  1853401832),  INT32_C(    87645671),  INT32_C(  1333633592),
         INT32_C(  1768009505),  INT32_C(  1711243077), -INT32_C(  1420220918), -INT32_C(  2096170801),  INT32_C(   268652536), -INT32_C(  1935570267),  INT32_C(   529717991), -INT32_C(  1653731460) },
      UINT32_C(        16),
      {  INT32_C(  2115436544),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1449066496),  INT32_C(           0),  INT32_C(  1427636224),  INT32_C(  1575419904),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   120651776),  INT32_C(  1341652992), -INT32_C(  1968898048), -INT32_C(   622395392),  INT32_C(           0) } },
    { UINT16_C( 1999),
      {  INT32_C(  1762020868),  INT32_C(    79834200),  INT32_C(  2131933948), -INT32_C(  1031642702), -INT32_C(  1822225402),  INT32_C(   304430579),  INT32_C(  1468085434), -INT32_C(   144813837),
        -INT32_C(   144677729), -INT32_C(  1912921199), -INT32_C(  1878258211),  INT32_C(  1968344943), -INT32_C(  1895254373), -INT32_C(  1381945869),  INT32_C(  1426399842),  INT32_C(   307061619) },
      UINT32_C(         5),
      {  INT32_C(   550092928), -INT32_C(  1740272896), -INT32_C(   497590400),  INT32_C(  1347171904),  INT32_C(           0),  INT32_C(           0), -INT32_C(   265906368), -INT32_C(   339075488),
        -INT32_C(   334720032), -INT32_C(  1083936224),  INT32_C(    25279392),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C( 2477),
      { -INT32_C(   435826599),  INT32_C(  1056051886), -INT32_C(   141524607), -INT32_C(  1392556515), -INT32_C(   414372550), -INT32_C(   454266737), -INT32_C(   181317405),  INT32_C(  1912528664),
         INT32_C(  1565983663), -INT32_C(  1751430890), -INT32_C(   393335605), -INT32_C(  1399550350),  INT32_C(  1251271099),  INT32_C(   875462737), -INT32_C(   399941424), -INT32_C(   296146881) },
      UINT32_C(        20),
      {  INT32_C(    93323264),  INT32_C(           0),  INT32_C(   403701760),  INT32_C(  1641021440),  INT32_C(           0),  INT32_C(   149946368),  INT32_C(           0), -INT32_C(   243269632),
         INT32_C(   988807168),  INT32_C(           0),  INT32_C(           0), -INT32_C(   417333248),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_slli_epi32(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_slli_epi32");
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m512i r = easysimd_mm512_maskz_slli_epi32(k, a, imm8);

    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm512_maskz_slli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[8];
    const uint32_t imm8;
    const int64_t r[8];
  } test_vec[] = {
    { UINT8_C(204),
      {  INT64_C( 9164915761926729643), -INT64_C( 2583429065325256256),  INT64_C( 4923988685455190467),  INT64_C( 5381061170551305080),
        -INT64_C( 4206259775351180339),  INT64_C( 1337231881559113406),  INT64_C( 2601217654719784758), -INT64_C( 6845836661716847339) },
      UINT32_C(        49),
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 3204874084827529216),  INT64_C( 7417428586279206912),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C( 5650891632443129856), -INT64_C( 1573445119812567040) } },
    { UINT8_C( 43),
      { -INT64_C( 2747117495280636991),  INT64_C( 5597654660386533369), -INT64_C( 8844854714510189646),  INT64_C( 8476230471602672503),
         INT64_C( 7715712287857125897),  INT64_C( 5293033132649593574),  INT64_C( 6701732168916542204), -INT64_C( 2264618095851914904) },
      UINT32_C(         9),
      { -INT64_C( 4571607981760216576),  INT64_C( 6753854692924584448),  INT64_C(                   0),  INT64_C( 4845144138823691776),
         INT64_C(                   0), -INT64_C( 1638414918712177664),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(105),
      {  INT64_C( 8135759861559333147), -INT64_C( 2467111916010373331), -INT64_C( 2682850120815171190), -INT64_C( 7587739109375894601),
         INT64_C(  427188685004722688), -INT64_C( 7506822120196341971),  INT64_C( 6265886552591827987),  INT64_C( 3716403401234269313) },
      UINT32_C(        69),
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 88),
      { -INT64_C( 6440936913837360207),  INT64_C( 7205515595579632168), -INT64_C( 7821122027711259076), -INT64_C( 8308569723840976697),
         INT64_C( 1344545396154414850),  INT64_C( 1592035502226978675), -INT64_C(  923031023807667799),  INT64_C( 2147371614856877156) },
      UINT32_C(        62),
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 4611686018427387904),
                             INT64_MIN,  INT64_C(                   0),  INT64_C( 4611686018427387904),  INT64_C(                   0) } },
    { UINT8_C(247),
      { -INT64_C( 4647939447474577377), -INT64_C( 5232773267646162457), -INT64_C( 3650492402554925807), -INT64_C( 5774597067665280488),
        -INT64_C( 5290333389265134557), -INT64_C( 1984653862581674162), -INT64_C( 2814330027936126812),  INT64_C( 7074479087657270767) },
      UINT32_C(        49),
      {  INT64_C( 4629137466983448576), -INT64_C( 5490450895718055936),  INT64_C( 1883067594194288640),  INT64_C(                   0),
         INT64_C( 4054928514493710336), -INT64_C( 4711891110136381440), -INT64_C( 3366440721459445760),  INT64_C( 8925571511494901760) } },
    { UINT8_C(164),
      {  INT64_C( 8082809845240628280),  INT64_C(  649411025690951334),  INT64_C( 6484468352441304143), -INT64_C( 2898867220720239116),
        -INT64_C( 1344659531004354773),  INT64_C( 6465492351296528078), -INT64_C( 6141335707087978033),  INT64_C(  813904546577037524) },
      UINT32_C(        51),
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 2483735194494828544),  INT64_C(                   0),
         INT64_C(                   0), -INT64_C(  689050742987685888),  INT64_C(                   0), -INT64_C( 4134304457926115328) } },
    { UINT8_C(239),
      {  INT64_C( 3181102799257483249),  INT64_C( 2393952961785507645),  INT64_C( 7417864655277191027), -INT64_C( 3730413312753020928),
        -INT64_C( 4435982388424280033),  INT64_C( 2527350282186827880),  INT64_C( 6332990539173776099), -INT64_C( 2105753045779162594) },
      UINT32_C(        48),
      {  INT64_C( 4031003141473304576),  INT64_C( 3403876893361963008), -INT64_C( 1192609476323049472), -INT64_C( 3458764513820540928),
         INT64_C(                   0), -INT64_C( 6311794877759750144), -INT64_C( 3827215258334789632), -INT64_C( 2729744324139941888) } },
    { UINT8_C(218),
      { -INT64_C( 3008549563841423066),  INT64_C( 8019130906458124140), -INT64_C( 7356670711664173014), -INT64_C( 5587760369424291116),
         INT64_C( 8112538753941669795),  INT64_C( 3960976399265587062),  INT64_C( 7162741628996924631), -INT64_C( 3957695237861810987) },
      UINT32_C(        57),
      {  INT64_C(                   0), -INT64_C( 2882303761517117440),  INT64_C(                   0), -INT64_C( 6341068275337658368),
         INT64_C( 5044031582654955520),  INT64_C(                   0), -INT64_C( 5908722711110090752), -INT64_C( 6196953087261802496) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_slli_epi64(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm512_maskz_slli_epi64");
    easysimd_test_x86_assert_equal_i64x8(r, easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m512i r = easysimd_mm512_maskz_slli_epi64(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm256_mask_slli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[16];
    const easysimd__mmask16 k;
    const int16_t a[16];
    const uint32_t imm8;
    const int16_t r[16];
  } test_vec[] = {
    { {  INT16_C( 26949),  INT16_C( 25323),  INT16_C(  3614), -INT16_C( 23799), -INT16_C( 14467), -INT16_C(  8639),  INT16_C( 20406),  INT16_C( 21722),
        -INT16_C( 28638),  INT16_C(  5486),  INT16_C( 16128), -INT16_C( 29728), -INT16_C( 27963),  INT16_C( 25175), -INT16_C(   870), -INT16_C(  7993) },
      UINT16_C(45925),
      { -INT16_C( 31678),  INT16_C( 19393),  INT16_C( 16167),  INT16_C( 26643), -INT16_C( 14051), -INT16_C(  2120), -INT16_C(  9698), -INT16_C( 29560),
        -INT16_C( 30481), -INT16_C( 12341), -INT16_C( 28397),  INT16_C( 27234), -INT16_C(   781), -INT16_C( 17817), -INT16_C( 13092),  INT16_C(  8045) },
      UINT32_C(         0),
      { -INT16_C( 31678),  INT16_C( 25323),  INT16_C( 16167), -INT16_C( 23799), -INT16_C( 14467), -INT16_C(  2120), -INT16_C(  9698),  INT16_C( 21722),
        -INT16_C( 30481), -INT16_C( 12341),  INT16_C( 16128), -INT16_C( 29728), -INT16_C(   781), -INT16_C( 17817), -INT16_C(   870),  INT16_C(  8045) } },
    { {  INT16_C( 27183),  INT16_C( 28279), -INT16_C(  8067),  INT16_C( 18315), -INT16_C( 32104),  INT16_C( 29285), -INT16_C(  3830), -INT16_C( 27807),
         INT16_C( 12477),  INT16_C( 20134),  INT16_C(  4498), -INT16_C( 28863), -INT16_C(  1160),  INT16_C( 17515), -INT16_C( 30103), -INT16_C( 26475) },
      UINT16_C( 3317),
      {  INT16_C( 29190), -INT16_C( 28180), -INT16_C( 31559),  INT16_C(  7699),  INT16_C(  7926),  INT16_C( 22288), -INT16_C( 12879),  INT16_C( 22408),
         INT16_C(  6683),  INT16_C( 23656), -INT16_C(  8023),  INT16_C(  5463), -INT16_C( 16347), -INT16_C( 17761), -INT16_C( 27560),  INT16_C( 24262) },
      UINT32_C(        12),
      {  INT16_C( 24576),  INT16_C( 28279), -INT16_C( 28672),  INT16_C( 18315),  INT16_C( 24576),  INT16_C(     0),  INT16_C(  4096),        INT16_MIN,
         INT16_C( 12477),  INT16_C( 20134), -INT16_C( 28672),  INT16_C( 28672), -INT16_C(  1160),  INT16_C( 17515), -INT16_C( 30103), -INT16_C( 26475) } },
    { { -INT16_C(  4173),  INT16_C( 14272), -INT16_C(  8445),  INT16_C(  8494), -INT16_C( 31249), -INT16_C( 17198),  INT16_C( 10509),  INT16_C( 10455),
         INT16_C( 13202),  INT16_C( 29393), -INT16_C(  6518),  INT16_C( 19351),  INT16_C( 20870),  INT16_C(  6819),  INT16_C(   536), -INT16_C( 13535) },
      UINT16_C(58097),
      { -INT16_C(  3070),  INT16_C( 12481), -INT16_C( 20459), -INT16_C(  6218), -INT16_C( 15508),  INT16_C( 17169), -INT16_C( 23573), -INT16_C( 17034),
         INT16_C(    21), -INT16_C( 21085),  INT16_C( 10571), -INT16_C(  4098),  INT16_C(  5700),  INT16_C( 26097), -INT16_C(  7455), -INT16_C(  7097) },
      UINT32_C(         0),
      { -INT16_C(  3070),  INT16_C( 14272), -INT16_C(  8445),  INT16_C(  8494), -INT16_C( 15508),  INT16_C( 17169), -INT16_C( 23573), -INT16_C( 17034),
         INT16_C( 13202), -INT16_C( 21085), -INT16_C(  6518),  INT16_C( 19351),  INT16_C( 20870),  INT16_C( 26097), -INT16_C(  7455), -INT16_C(  7097) } },
    { {  INT16_C(  5128), -INT16_C( 18196), -INT16_C( 11062), -INT16_C( 29148),  INT16_C( 26597), -INT16_C( 30599),  INT16_C( 14045), -INT16_C(  8547),
         INT16_C( 19162),  INT16_C(   809),  INT16_C(  6217),  INT16_C( 24391), -INT16_C( 21239), -INT16_C(  5055),  INT16_C(  9716), -INT16_C(   573) },
      UINT16_C(44857),
      {  INT16_C(  1205), -INT16_C(  9597),  INT16_C( 26770),  INT16_C(  2881),  INT16_C(  8176), -INT16_C( 29118),  INT16_C(  7421),  INT16_C(  9944),
         INT16_C(  8479),  INT16_C( 26431),  INT16_C( 18561), -INT16_C( 15852),  INT16_C(  2100), -INT16_C(  2073),  INT16_C(  8197), -INT16_C( 17497) },
      UINT32_C(         0),
      {  INT16_C(  1205), -INT16_C( 18196), -INT16_C( 11062),  INT16_C(  2881),  INT16_C(  8176), -INT16_C( 29118),  INT16_C( 14045), -INT16_C(  8547),
         INT16_C(  8479),  INT16_C( 26431),  INT16_C( 18561), -INT16_C( 15852), -INT16_C( 21239), -INT16_C(  2073),  INT16_C(  9716), -INT16_C( 17497) } },
    { { -INT16_C( 27350), -INT16_C( 27722), -INT16_C( 15658), -INT16_C(  2685),  INT16_C(  4356),  INT16_C(  8434),  INT16_C(  6634),  INT16_C(  2879),
        -INT16_C( 22952), -INT16_C( 24436),  INT16_C( 20154), -INT16_C( 15403), -INT16_C( 13259),  INT16_C( 22216), -INT16_C( 31885), -INT16_C( 24966) },
      UINT16_C(12568),
      { -INT16_C(  4303), -INT16_C( 19213), -INT16_C(  2076), -INT16_C( 10298), -INT16_C( 20457),  INT16_C( 22256),  INT16_C( 18619),  INT16_C( 18685),
        -INT16_C( 18456), -INT16_C( 17002), -INT16_C( 13190),  INT16_C( 17290), -INT16_C(   734), -INT16_C( 25402), -INT16_C(  8293), -INT16_C( 13107) },
      UINT32_C(        13),
      { -INT16_C( 27350), -INT16_C( 27722), -INT16_C( 15658), -INT16_C( 16384), -INT16_C(  8192),  INT16_C(  8434),  INT16_C(  6634),  INT16_C(  2879),
         INT16_C(     0), -INT16_C( 24436),  INT16_C( 20154), -INT16_C( 15403),  INT16_C( 16384), -INT16_C( 16384), -INT16_C( 31885), -INT16_C( 24966) } },
    { { -INT16_C( 32320), -INT16_C( 18510), -INT16_C( 30393), -INT16_C(  2098),  INT16_C(  9593), -INT16_C( 15950), -INT16_C(  1502), -INT16_C(  9814),
         INT16_C( 26513),  INT16_C( 23892), -INT16_C( 26639), -INT16_C(  4225),  INT16_C(  7005),  INT16_C( 15498),  INT16_C( 22505), -INT16_C( 22262) },
      UINT16_C(48600),
      {  INT16_C(  8033),  INT16_C( 12102), -INT16_C( 16362), -INT16_C( 14252),  INT16_C( 30337),  INT16_C( 11203),  INT16_C( 21584), -INT16_C( 23405),
        -INT16_C( 31567),  INT16_C( 12347), -INT16_C( 26509), -INT16_C(   437),  INT16_C( 13525), -INT16_C(  8363),  INT16_C( 11742),  INT16_C( 16284) },
      UINT32_C(        13),
      { -INT16_C( 32320), -INT16_C( 18510), -INT16_C( 30393),        INT16_MIN,  INT16_C(  8192), -INT16_C( 15950),  INT16_C(     0),  INT16_C( 24576),
         INT16_C(  8192),  INT16_C( 23892),  INT16_C( 24576),  INT16_C( 24576), -INT16_C( 24576), -INT16_C( 24576),  INT16_C( 22505),        INT16_MIN } },
    { {  INT16_C( 28387), -INT16_C( 23710),  INT16_C( 10947),  INT16_C( 14628),  INT16_C( 20717),  INT16_C( 16777),  INT16_C( 11747),  INT16_C( 26610),
         INT16_C(  8808),  INT16_C(   475), -INT16_C(  9874), -INT16_C( 23850), -INT16_C( 19154),  INT16_C( 23424), -INT16_C( 16558),  INT16_C( 13735) },
      UINT16_C( 2350),
      { -INT16_C(  3624), -INT16_C(   973),  INT16_C(  8490), -INT16_C( 19380),  INT16_C( 12130),  INT16_C( 21985),  INT16_C( 19095),  INT16_C( 29303),
        -INT16_C(  6837),  INT16_C(  8523),  INT16_C( 31112),  INT16_C(  2262),  INT16_C( 10452),  INT16_C( 31688), -INT16_C(  2467),  INT16_C( 13700) },
      UINT32_C(        14),
      {  INT16_C( 28387), -INT16_C( 16384),        INT16_MIN,  INT16_C(     0),  INT16_C( 20717),  INT16_C( 16384),  INT16_C( 11747),  INT16_C( 26610),
        -INT16_C( 16384),  INT16_C(   475), -INT16_C(  9874),        INT16_MIN, -INT16_C( 19154),  INT16_C( 23424), -INT16_C( 16558),  INT16_C( 13735) } },
    { {  INT16_C( 12983), -INT16_C( 10223), -INT16_C( 14978), -INT16_C( 20933), -INT16_C( 28505), -INT16_C(  3771), -INT16_C( 18681), -INT16_C(  4804),
         INT16_C( 23810),  INT16_C( 31605),  INT16_C( 32051),  INT16_C( 23631), -INT16_C( 13755),  INT16_C( 15289), -INT16_C(  4274),  INT16_C(  1314) },
      UINT16_C(13345),
      { -INT16_C( 24610),  INT16_C(  6649), -INT16_C( 24499), -INT16_C( 27991), -INT16_C( 20335), -INT16_C( 12983),  INT16_C( 19357),  INT16_C(  4650),
         INT16_C( 24262),  INT16_C(  5520), -INT16_C( 10822),  INT16_C( 29663),  INT16_C( 11537),  INT16_C( 13154), -INT16_C( 31949),  INT16_C(  4455) },
      UINT32_C(         1),
      {  INT16_C( 16316), -INT16_C( 10223), -INT16_C( 14978), -INT16_C( 20933), -INT16_C( 28505), -INT16_C( 25966), -INT16_C( 18681), -INT16_C(  4804),
         INT16_C( 23810),  INT16_C( 31605), -INT16_C( 21644),  INT16_C( 23631),  INT16_C( 23074),  INT16_C( 26308), -INT16_C(  4274),  INT16_C(  1314) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_slli_epi16(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_slli_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    // uint32_t imm8 = easysimd_test_x86_random_mmask16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m256i r = easysimd_mm256_mask_slli_epi16(src, k, a, imm8);

    easysimd_test_x86_write_i16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm256_mask_slli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const int32_t a[8];
    const uint32_t imm8;
    const int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(   968029356),  INT32_C(   857084784), -INT32_C(   866843682), -INT32_C(  1801620612),  INT32_C(   276276693),  INT32_C(   954808613), -INT32_C(  1613317021), -INT32_C(   522504140) },
      UINT8_C( 44),
      { -INT32_C(  1583605363),  INT32_C(   864014128), -INT32_C(  1733342427), -INT32_C(  1905376279), -INT32_C(   189563205),  INT32_C(   592964455), -INT32_C(    94898494),  INT32_C(  1596340178) },
      UINT32_C(         6),
      {  INT32_C(   968029356),  INT32_C(   857084784),  INT32_C(   735234368), -INT32_C(  1684997568),  INT32_C(   276276693), -INT32_C(   704980544), -INT32_C(  1613317021), -INT32_C(   522504140) } },
    { { -INT32_C(  1837039422), -INT32_C(   877153153), -INT32_C(  1498132637),  INT32_C(   996229821), -INT32_C(   526231819), -INT32_C(  1532836435),  INT32_C(  1417059356), -INT32_C(  2052729406) },
      UINT8_C(214),
      { -INT32_C(   631957722),  INT32_C(   490545358),  INT32_C(   383509460), -INT32_C(  1693772219),  INT32_C(  2118708409),  INT32_C(   731638926),  INT32_C(   955117410), -INT32_C(  1173458284) },
      UINT32_C(         5),
      { -INT32_C(  1837039422), -INT32_C(  1482417728), -INT32_C(   612599168),  INT32_C(   996229821), -INT32_C(   920807648), -INT32_C(  1532836435),  INT32_C(   498986048),  INT32_C(  1104040576) } },
    { { -INT32_C(  2074635165), -INT32_C(  1269271087),  INT32_C(  1727622992),  INT32_C(  1713345658), -INT32_C(   906649891),  INT32_C(   673980473), -INT32_C(  2135137267), -INT32_C(   720800142) },
      UINT8_C( 10),
      { -INT32_C(   690267807),  INT32_C(   556175282),  INT32_C(   496733321),  INT32_C(  1257898668),  INT32_C(   394511607),  INT32_C(  1411689456), -INT32_C(   574184345),  INT32_C(   266836910) },
      UINT32_C(         0),
      { -INT32_C(  2074635165),  INT32_C(   556175282),  INT32_C(  1727622992),  INT32_C(  1257898668), -INT32_C(   906649891),  INT32_C(   673980473), -INT32_C(  2135137267), -INT32_C(   720800142) } },
    { {  INT32_C(  1369957826), -INT32_C(  1747269621),  INT32_C(  1698953315), -INT32_C(  1235448334), -INT32_C(  1146719472),  INT32_C(  1008925336),  INT32_C(  1558904768), -INT32_C(  1471022618) },
      UINT8_C(222),
      { -INT32_C(  1058407688), -INT32_C(   870088236),  INT32_C(  1388284356),  INT32_C(  1499624933),  INT32_C(   384900636),  INT32_C(  1087843648),  INT32_C(   287716120),  INT32_C(  2096156548) },
      UINT32_C(         1),
      {  INT32_C(  1369957826), -INT32_C(  1740176472), -INT32_C(  1518398584), -INT32_C(  1295717430),  INT32_C(   769801272),  INT32_C(  1008925336),  INT32_C(   575432240), -INT32_C(   102654200) } },
    { {  INT32_C(  1520254169), -INT32_C(   400594337), -INT32_C(  1630637783), -INT32_C(   239458349),  INT32_C(  1160958232), -INT32_C(   614632792),  INT32_C(  1734307736),  INT32_C(   942725983) },
      UINT8_C( 23),
      {  INT32_C(   947295182),  INT32_C(   593584050),  INT32_C(  1425473325), -INT32_C(  1955796806), -INT32_C(  1942769382),  INT32_C(  2116292111),  INT32_C(  1222478957), -INT32_C(  1973480004) },
      UINT32_C(        17),
      {  INT32_C(   664535040), -INT32_C(  1083965440), -INT32_C(    27656192), -INT32_C(   239458349),  INT32_C(  1647575040), -INT32_C(   614632792),  INT32_C(  1734307736),  INT32_C(   942725983) } },
    { {  INT32_C(   878363349),  INT32_C(   593591587),  INT32_C(  1558033780), -INT32_C(   747214815), -INT32_C(  1444805989), -INT32_C(  1307156442),  INT32_C(  1383030333), -INT32_C(  1828980291) },
      UINT8_C(188),
      { -INT32_C(   740309163), -INT32_C(   565771736),  INT32_C(  1207935967),  INT32_C(   467849753),  INT32_C(   339839924), -INT32_C(    11406175),  INT32_C(  1555866723), -INT32_C(   199733345) },
      UINT32_C(         6),
      {  INT32_C(   878363349),  INT32_C(   593591587), -INT32_C(     1509440), -INT32_C(   122386880),  INT32_C(   274918656), -INT32_C(   729995200),  INT32_C(  1383030333),  INT32_C(   101967808) } },
    { { -INT32_C(    96548872), -INT32_C(  1311106034), -INT32_C(   288743140), -INT32_C(  1885084412),  INT32_C(   456242983), -INT32_C(  1384239095),  INT32_C(  1011669741), -INT32_C(   346930957) },
      UINT8_C(  8),
      { -INT32_C(  1391008368), -INT32_C(   506869569),  INT32_C(  2028320658),  INT32_C(   312439898), -INT32_C(   702825819),  INT32_C(   314820664),  INT32_C(  1409679124), -INT32_C(   514002863) },
      UINT32_C(        35),
      { -INT32_C(    96548872), -INT32_C(  1311106034), -INT32_C(   288743140),  INT32_C(           0),  INT32_C(   456242983), -INT32_C(  1384239095),  INT32_C(  1011669741), -INT32_C(   346930957) } },
    { {  INT32_C(   999657075),  INT32_C(   265123415), -INT32_C(   798407333), -INT32_C(  1636467740),  INT32_C(  1591102358),  INT32_C(   225634574), -INT32_C(   547436562), -INT32_C(  1766506461) },
      UINT8_C(206),
      { -INT32_C(  1054420662), -INT32_C(   467913313),  INT32_C(   449440926), -INT32_C(  1380948126),  INT32_C(   666636094),  INT32_C(  1209387393),  INT32_C(  1751970856), -INT32_C(   197786967) },
      UINT32_C(         5),
      {  INT32_C(   999657075), -INT32_C(  2088324128),  INT32_C(  1497207744), -INT32_C(  1240667072),  INT32_C(  1591102358),  INT32_C(   225634574),  INT32_C(   228492544), -INT32_C(  2034215648) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    // printf("\ni = %ld, imm8 = %6d\n", i, imm8);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_slli_epi32(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_slli_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m256i r = easysimd_mm256_mask_slli_epi32(src, k, a, imm8);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

/*      256 64     */
static int
test_easysimd_mm256_mask_slli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const easysimd__mmask8 k;
    const int64_t a[4];
    const uint32_t imm8;
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 4812187132717087731),  INT64_C(  831167788357331278),  INT64_C( 1150457494602327688),  INT64_C( 7769099101622368994) },
      UINT8_C( 29),
      { -INT64_C( 2612066126723421664), -INT64_C( 7833278805651298852),  INT64_C( 3652596563153404303),  INT64_C( 1306632070607020450) },
      UINT32_C(        33),
      { -INT64_C( 1899441246477221888),  INT64_C(  831167788357331278), -INT64_C( 7123424545721024512),  INT64_C(  672236203721883648) } },
    { {  INT64_C( 3090469092615947507),  INT64_C( 3550292194107041052), -INT64_C( 2017441369657109571),  INT64_C( 1827310424787715141) },
      UINT8_C(110),
      { -INT64_C( 6289316446544453619), -INT64_C( 4153222987323511436), -INT64_C( 4750331433354413518), -INT64_C( 2910171114864546235) },
      UINT32_C(        58),
      {  INT64_C( 3090469092615947507), -INT64_C( 3458764513820540928), -INT64_C( 4035225266123964416),  INT64_C( 1441151880758558720) } },
    { { -INT64_C( 7593128168557284785),  INT64_C( 6079313701839126114), -INT64_C( 5959444011326493544),  INT64_C( 7336169276878322354) },
      UINT8_C(193),
      {  INT64_C( 8006697758829023728), -INT64_C( 3210399509367928374),  INT64_C( 4533929398882170988),  INT64_C( 6577712226752568375) },
      UINT32_C(         3),
      {  INT64_C( 8713349849503534976),  INT64_C( 6079313701839126114), -INT64_C( 5959444011326493544),  INT64_C( 7336169276878322354) } },
    { {  INT64_C( 8971146490537514547),  INT64_C( 4881611494449699060),  INT64_C( 9151672762426193929),  INT64_C(  188666090676193177) },
      UINT8_C( 92),
      { -INT64_C( 8144747599466875110),  INT64_C(   85458481038576174), -INT64_C( 9114893146304218973), -INT64_C( 8218977564076485298) },
      UINT32_C(        21),
      {  INT64_C( 8971146490537514547),  INT64_C( 4881611494449699060), -INT64_C( 6972385178532446208),  INT64_C( 7958575338638802944) } },
    { { -INT64_C( 3046176578519133569),  INT64_C(  461301262065786407), -INT64_C( 6523885378686988605),  INT64_C( 8804157919128455889) },
      UINT8_C( 56),
      { -INT64_C( 7177445062802799250), -INT64_C( 2587123860572067979), -INT64_C( 4611314704803151953), -INT64_C( 4519303595187034774) },
      UINT32_C(         1),
      { -INT64_C( 3046176578519133569),  INT64_C(  461301262065786407), -INT64_C( 6523885378686988605), -INT64_C( 9038607190374069548) } },
    { { -INT64_C( 8568215519470486262), -INT64_C( 4350772437720798145), -INT64_C(  839013340302160224), -INT64_C( 4727644143247463436) },
      UINT8_C(  6),
      { -INT64_C( 2075895030275940260), -INT64_C( 8315004187747868921),  INT64_C( 6039697536357010596), -INT64_C( 4925032132280583528) },
      UINT32_C(         4),
      { -INT64_C( 8568215519470486262), -INT64_C( 3912858487999041424),  INT64_C( 4401440213164411456), -INT64_C( 4727644143247463436) } },
    { { -INT64_C(  286399835343639538),  INT64_C( 2933610942933107175),  INT64_C( 6884706840103392163), -INT64_C( 2281784792037118869) },
      UINT8_C( 37),
      { -INT64_C( 1848165910661898255), -INT64_C( 5358252068626601031),  INT64_C( 5659469069202134715),  INT64_C(   59673861936953910) },
      UINT32_C(        65),
      {  INT64_C(                   0),  INT64_C( 2933610942933107175),  INT64_C(                   0), -INT64_C( 2281784792037118869) } },
    { {  INT64_C( 3930986941700731606), -INT64_C( 4066171910628012755), -INT64_C(  732055685165371096),  INT64_C( 3474532998098224727) },
      UINT8_C(119),
      {  INT64_C( 2365929750536521213), -INT64_C( 1078868022634959407),  INT64_C(  472634415679055872),  INT64_C( 2135428740679788326) },
      UINT32_C(         1),
      {  INT64_C( 4731859501073042426), -INT64_C( 2157736045269918814),  INT64_C(  945268831358111744),  INT64_C( 3474532998098224727) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_slli_epi64(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_mask_slli_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m256i r = easysimd_mm256_mask_slli_epi64(src, k, a, imm8);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm256_maskz_slli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int16_t a[16];
    const uint32_t imm8;
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C(28480),
      { -INT16_C(  1351),  INT16_C( 19847), -INT16_C(  1694),  INT16_C(  5421),  INT16_C( 13070),  INT16_C( 27384), -INT16_C( 15406), -INT16_C( 11529),
         INT16_C(  5063), -INT16_C(  3123),  INT16_C( 22491),  INT16_C( 26694),  INT16_C( 14252), -INT16_C( 28074), -INT16_C( 26976),  INT16_C( 22786) },
      UINT32_C(         3),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  7824),  INT16_C(     0),
        -INT16_C( 25032), -INT16_C( 24984), -INT16_C( 16680),  INT16_C( 16944),  INT16_C(     0), -INT16_C( 27984), -INT16_C( 19200),  INT16_C(     0) } },
    { UINT16_C(42633),
      { -INT16_C( 32014),  INT16_C(  2004),  INT16_C(  1936), -INT16_C(  1536), -INT16_C( 15399), -INT16_C( 21519),  INT16_C(  1419),  INT16_C( 32376),
        -INT16_C( 12064),  INT16_C( 18629), -INT16_C(   900),  INT16_C(  3998),  INT16_C( 13468), -INT16_C(  2543), -INT16_C( 25916), -INT16_C( 18532) },
      UINT32_C(        11),
      { -INT16_C( 28672),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 16384),
         INT16_C(     0),  INT16_C( 10240), -INT16_C(  8192),  INT16_C(     0),  INT16_C(     0), -INT16_C( 30720),  INT16_C(     0), -INT16_C(  8192) } },
    { UINT16_C(48752),
      {  INT16_C( 30637), -INT16_C( 22594), -INT16_C( 32176), -INT16_C(  1127), -INT16_C( 25075), -INT16_C( 29836),  INT16_C( 17534), -INT16_C( 14512),
         INT16_C( 19904), -INT16_C( 12443), -INT16_C( 25879), -INT16_C(  8224),  INT16_C( 31326),  INT16_C(  5500), -INT16_C(  4969),  INT16_C( 17620) },
      UINT32_C(         6),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31936), -INT16_C(  8960),  INT16_C(  8064),  INT16_C(     0),
         INT16_C(     0), -INT16_C(  9920), -INT16_C( 17856), -INT16_C(  2048), -INT16_C( 26752),  INT16_C( 24320),  INT16_C(     0),  INT16_C( 13568) } },
    { UINT16_C(60306),
      {  INT16_C(  5300), -INT16_C( 20348),  INT16_C(  8737), -INT16_C( 21212),  INT16_C( 26785),  INT16_C( 26877),  INT16_C( 18984), -INT16_C(  1843),
         INT16_C( 26420),  INT16_C(  5080),  INT16_C( 21446), -INT16_C(  9329),  INT16_C( 31978),  INT16_C( 11951),  INT16_C( 17120), -INT16_C( 27623) },
      UINT32_C(         6),
      {  INT16_C(     0),  INT16_C(  8448),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10304),  INT16_C(     0),  INT16_C(     0),  INT16_C( 13120),
        -INT16_C( 13056), -INT16_C(  2560),  INT16_C(     0), -INT16_C(  7232),  INT16_C(     0), -INT16_C( 21568), -INT16_C( 18432),  INT16_C(  1600) } },
    { UINT16_C(17566),
      { -INT16_C( 16264),  INT16_C(  9576), -INT16_C( 12191), -INT16_C( 14046),  INT16_C( 28153), -INT16_C(  3689), -INT16_C(   351), -INT16_C( 19255),
         INT16_C(  7364), -INT16_C( 24508), -INT16_C( 16378),  INT16_C( 13391), -INT16_C( 28256),  INT16_C( 13390), -INT16_C(  4888),  INT16_C( 24697) },
      UINT32_C(        10),
      {  INT16_C(     0), -INT16_C( 24576), -INT16_C( 31744), -INT16_C( 30720), -INT16_C(  7168),  INT16_C(     0),  INT16_C(     0),  INT16_C(  9216),
         INT16_C(     0),  INT16_C(     0),  INT16_C(  6144),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 24576),  INT16_C(     0) } },
    { UINT16_C(34273),
      { -INT16_C( 19954), -INT16_C( 10329),  INT16_C(  5291), -INT16_C( 25490),  INT16_C( 28085),  INT16_C( 27237), -INT16_C( 32207), -INT16_C( 11858),
         INT16_C( 28296), -INT16_C( 17119), -INT16_C( 19954),  INT16_C( 16907), -INT16_C(  2150), -INT16_C(  1349), -INT16_C( 25181), -INT16_C( 20097) },
      UINT32_C(        10),
      {  INT16_C( 14336),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 27648), -INT16_C( 15360), -INT16_C( 18432),
         INT16_C(  8192),  INT16_C(     0),  INT16_C( 14336),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  1024) } },
    { UINT16_C(35111),
      {  INT16_C( 15354), -INT16_C( 26889),  INT16_C( 25841),  INT16_C( 23547),  INT16_C( 32150),  INT16_C( 26377),  INT16_C( 30470), -INT16_C( 15480),
         INT16_C( 15237), -INT16_C( 14386), -INT16_C( 14891), -INT16_C( 12157),  INT16_C(  8296),  INT16_C(  6735),  INT16_C( 30319),  INT16_C( 27043) },
      UINT32_C(         8),
      { -INT16_C(  1536), -INT16_C(  2304), -INT16_C(  3840),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2304),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 31488),  INT16_C(     0),  INT16_C(     0), -INT16_C( 32000),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 23808) } },
    { UINT16_C(65434),
      { -INT16_C(    93), -INT16_C(   262),  INT16_C( 30869), -INT16_C(  1017),  INT16_C( 32382),  INT16_C( 16773), -INT16_C( 16381), -INT16_C( 13809),
        -INT16_C( 11115),  INT16_C( 25933),  INT16_C( 27964),  INT16_C( 22197),  INT16_C( 11228),  INT16_C( 17913), -INT16_C( 27427), -INT16_C( 32700) },
      UINT32_C(        10),
      {  INT16_C(     0), -INT16_C(  6144),  INT16_C(     0),  INT16_C(  7168), -INT16_C(  2048),  INT16_C(     0),  INT16_C(     0),  INT16_C( 15360),
         INT16_C( 21504),  INT16_C( 13312), -INT16_C(  4096), -INT16_C( 11264),  INT16_C( 28672), -INT16_C(  7168),  INT16_C( 29696),  INT16_C(  4096) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_slli_epi16(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_slli_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    // uint32_t imm8 = easysimd_test_x86_random_mmask16();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m256i r = easysimd_mm256_maskz_slli_epi16(k, a, imm8);

    easysimd_test_codegen_write_u16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm256_maskz_slli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[8];
    const uint32_t imm8;
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(214),
      {  INT32_C(   727059997),  INT32_C(   955575126),  INT32_C(   859882472), -INT32_C(   978119702), -INT32_C(   889112471),  INT32_C(   877243695),  INT32_C(   752303167), -INT32_C(  1878866830) },
      UINT32_C(        31),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),              INT32_MIN,  INT32_C(           0),              INT32_MIN,  INT32_C(           0) } },
    { UINT8_C( 88),
      { -INT32_C(  1354552901), -INT32_C(  1569575839),  INT32_C(   313875551),  INT32_C(   592059938),  INT32_C(   852523753), -INT32_C(  2056318802),  INT32_C(  1045094716),  INT32_C(   748035185) },
      UINT32_C(        10),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   678987776),  INT32_C(  1105961984),  INT32_C(           0),  INT32_C(   730132480),  INT32_C(           0) } },
    { UINT8_C(218),
      {  INT32_C(  1309059036), -INT32_C(    22321847), -INT32_C(  1038299784), -INT32_C(  1053030928), -INT32_C(  1479415240), -INT32_C(  1165488784), -INT32_C(   506004917),  INT32_C(    45816870) },
      UINT32_C(        17),
      {  INT32_C(           0), -INT32_C(   896401408),  INT32_C(           0),  INT32_C(   199229440), -INT32_C(   730857472),  INT32_C(           0), -INT32_C(   191496192),  INT32_C(   944504832) } },
    { UINT8_C(193),
      { -INT32_C(    64549807),  INT32_C(   650747658), -INT32_C(  1641301663),  INT32_C(  1267229562), -INT32_C(  1822885878), -INT32_C(  1987206222),  INT32_C(  1084601221),  INT32_C(   117598390) },
      UINT32_C(         4),
      { -INT32_C(  1032796912),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   173750352),  INT32_C(  1881574240) } },
    { UINT8_C( 41),
      { -INT32_C(   859275773), -INT32_C(   813356636),  INT32_C(  1345519815), -INT32_C(  1538769589),  INT32_C(  1564998608),  INT32_C(   689032324),  INT32_C(   261211917), -INT32_C(   784857650) },
      UINT32_C(         8),
      { -INT32_C(   931265792),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1211976448),  INT32_C(           0),  INT32_C(   298615808),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(  0),
      {  INT32_C(   573122461),  INT32_C(   690156022), -INT32_C(  1989710271),  INT32_C(  1501837074), -INT32_C(  1507456875), -INT32_C(  1033030863),  INT32_C(  2059877186), -INT32_C(  1451603444) },
      UINT32_C(         7),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(163),
      { -INT32_C(   292263733), -INT32_C(   128133743), -INT32_C(   483365281), -INT32_C(   303249977), -INT32_C(   668983957), -INT32_C(  1491901728),  INT32_C(  1458708444), -INT32_C(  1812307000) },
      UINT32_C(        24),
      { -INT32_C(   889192448), -INT32_C(  1862270976),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   536870912),  INT32_C(           0), -INT32_C(   939524096) } },
    { UINT8_C(142),
      { -INT32_C(   580625023), -INT32_C(  2058632363), -INT32_C(  1840639066),  INT32_C(   565228800), -INT32_C(  1601990515),  INT32_C(   700407863), -INT32_C(  1332901962), -INT32_C(  1673570277) },
      UINT32_C(        29),
      {  INT32_C(           0), -INT32_C(  1610612736), -INT32_C(  1073741824),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1610612736) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    // printf("\ni = %ld, imm8 = %6d\n", i, imm8);
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_slli_epi32(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_slli_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m256i r = easysimd_mm256_maskz_slli_epi32(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm256_maskz_slli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[4];
    const uint32_t imm8;
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C( 46),
      { -INT64_C( 3664917945119470360),  INT64_C( 5981604723986789692),  INT64_C( 1686317717696267767), -INT64_C( 5251893362825298080) },
      UINT32_C(        19),
      {  INT64_C(                   0),  INT64_C( 7957790446252457984),  INT64_C(  593610789445173248),  INT64_C( 3926985527470850048) } },
    { UINT8_C( 80),
      { -INT64_C( 1029594874113417827), -INT64_C( 7933980428373884131), -INT64_C( 7864706605666604173), -INT64_C(  121972806335813151) },
      UINT32_C(        19),
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 54),
      { -INT64_C( 9009318411110731073), -INT64_C(  979871158443127145),  INT64_C( 1308795945934481574),  INT64_C( 5137062945439386309) },
      UINT32_C(        73),
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(165),
      { -INT64_C( 5439587093953259076), -INT64_C( 5116890911434809041), -INT64_C( 4225990491390347353), -INT64_C( 1925713073389641129) },
      UINT32_C(        58),
      { -INT64_C( 1152921504606846976),  INT64_C(                   0), -INT64_C( 7205759403792793600),  INT64_C(                   0) } },
    { UINT8_C( 99),
      { -INT64_C( 1802206871523183975),  INT64_C(  634924576610832603), -INT64_C( 1818992876029421436),  INT64_C( 7705583991820446533) },
      UINT32_C(        44),
      {  INT64_C( 8874782870384672768), -INT64_C( 2518163102583750656),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(  6),
      { -INT64_C( 4083875655579741051),  INT64_C( 3506334767634963457), -INT64_C( 8031298930529319201),  INT64_C( 5447589486807178404) },
      UINT32_C(        44),
      {  INT64_C(                   0), -INT64_C( 4773798012826681344),  INT64_C( 3777939545224380416),  INT64_C(                   0) } },
    { UINT8_C(230),
      { -INT64_C( 8431565270202497074), -INT64_C( 1958950313863224864), -INT64_C( 2814290662650938358), -INT64_C( 3336008422292080844) },
      UINT32_C(        62),
      {  INT64_C(                   0),  INT64_C(                   0),                      INT64_MIN,  INT64_C(                   0) } },
    { UINT8_C(195),
      { -INT64_C( 8444051389498034702), -INT64_C( 4464758532776664771),  INT64_C(   17055003820922515),  INT64_C(  206185987139649381) },
      UINT32_C(        57),
      { -INT64_C( 2017612633061982208),  INT64_C( 8791026472627208192),  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i a = easysimd_mm256_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_slli_epi64(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm256_maskz_slli_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m256i r = easysimd_mm256_maskz_slli_epi64(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int 
test_easysimd_mm_mask_slli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int16_t src[8];
    const easysimd__mmask8 k;
    const int16_t a[8];
    const uint32_t imm8;
    const int16_t r[8];
  } test_vec[] = {
    { { -INT16_C( 19196),  INT16_C( 18965), -INT16_C(  3204), -INT16_C( 15087),  INT16_C( 12208),  INT16_C( 12060), -INT16_C( 16995), -INT16_C( 23750) },
      UINT8_C(250),
      {  INT16_C( 28646),  INT16_C( 30033), -INT16_C( 21637),  INT16_C( 23966), -INT16_C( 30726),  INT16_C(  1651),  INT16_C( 30498),  INT16_C( 14267) },
      UINT32_C(         9),
      { -INT16_C( 19196), -INT16_C( 24064), -INT16_C(  3204),  INT16_C( 15360), -INT16_C(  3072), -INT16_C(  6656),  INT16_C( 17408),  INT16_C( 30208) } },
    { {  INT16_C( 10808), -INT16_C(   557),  INT16_C(   730),  INT16_C(  2585), -INT16_C( 10593),  INT16_C( 16964), -INT16_C( 22832),  INT16_C( 16425) },
      UINT8_C(247),
      { -INT16_C( 23621),  INT16_C(  6204), -INT16_C( 15203), -INT16_C( 23668),  INT16_C(   998),  INT16_C(  7519), -INT16_C( 26683), -INT16_C( 26297) },
      UINT32_C(         6),
      { -INT16_C(  4416),  INT16_C(  3840),  INT16_C( 10048),  INT16_C(  2585), -INT16_C(  1664),  INT16_C( 22464), -INT16_C(  3776),  INT16_C( 20928) } },
    { { -INT16_C( 25822),  INT16_C( 11437), -INT16_C( 31685),  INT16_C( 32112),  INT16_C(  5716), -INT16_C( 27482),  INT16_C( 17678), -INT16_C( 20144) },
      UINT8_C(129),
      {  INT16_C( 17742), -INT16_C(  3340), -INT16_C(  2005),  INT16_C( 18513), -INT16_C(  5955),  INT16_C( 22160), -INT16_C( 19844),  INT16_C( 10738) },
      UINT32_C(        13),
      { -INT16_C( 16384),  INT16_C( 11437), -INT16_C( 31685),  INT16_C( 32112),  INT16_C(  5716), -INT16_C( 27482),  INT16_C( 17678),  INT16_C( 16384) } },
    { { -INT16_C( 21203), -INT16_C( 21938),  INT16_C( 25858), -INT16_C( 27055), -INT16_C( 27021),  INT16_C(  9446),  INT16_C( 20247),  INT16_C( 23922) },
      UINT8_C( 67),
      {  INT16_C( 15240), -INT16_C( 11851), -INT16_C( 25095),  INT16_C( 20321),  INT16_C(  4889),  INT16_C( 17217),  INT16_C( 28401),  INT16_C( 16368) },
      UINT32_C(         8),
      { -INT16_C( 30720), -INT16_C( 19200),  INT16_C( 25858), -INT16_C( 27055), -INT16_C( 27021),  INT16_C(  9446), -INT16_C(  3840),  INT16_C( 23922) } },
    { { -INT16_C( 23310), -INT16_C( 30358),  INT16_C(    23),  INT16_C( 15215), -INT16_C( 16873),  INT16_C( 29870),  INT16_C(  4610),  INT16_C( 15869) },
      UINT8_C(200),
      {  INT16_C( 25910), -INT16_C( 31185),  INT16_C( 17023), -INT16_C( 15673),  INT16_C( 13875),  INT16_C( 29362), -INT16_C( 23217), -INT16_C( 18153) },
      UINT32_C(         7),
      { -INT16_C( 23310), -INT16_C( 30358),  INT16_C(    23),  INT16_C( 25472), -INT16_C( 16873),  INT16_C( 29870), -INT16_C( 22656), -INT16_C( 29824) } },
    { { -INT16_C( 18130),  INT16_C( 27293),  INT16_C( 23760),  INT16_C( 17688),  INT16_C( 10846), -INT16_C( 25790),  INT16_C(  4338),  INT16_C( 22738) },
      UINT8_C( 63),
      { -INT16_C( 32297), -INT16_C( 26337),  INT16_C( 21940),  INT16_C(  9803), -INT16_C(  3932),  INT16_C( 23869),  INT16_C( 27678), -INT16_C( 17386) },
      UINT32_C(         9),
      { -INT16_C( 20992),  INT16_C( 15872),  INT16_C( 26624), -INT16_C( 27136),  INT16_C( 18432),  INT16_C( 31232),  INT16_C(  4338),  INT16_C( 22738) } },
    { {  INT16_C(  6375),  INT16_C( 11502),  INT16_C(  6262),  INT16_C(  4462),  INT16_C( 32267),  INT16_C( 25571),  INT16_C( 15293),  INT16_C( 15930) },
      UINT8_C( 91),
      { -INT16_C( 20238),  INT16_C(  6174),  INT16_C(  3925), -INT16_C( 19882), -INT16_C( 15827), -INT16_C(  5687), -INT16_C( 20328), -INT16_C( 31231) },
      UINT32_C(         4),
      {  INT16_C(  3872), -INT16_C( 32288),  INT16_C(  6262),  INT16_C(  9568),  INT16_C(  8912),  INT16_C( 25571),  INT16_C(  2432),  INT16_C( 15930) } },
    { { -INT16_C( 24969), -INT16_C( 30390), -INT16_C( 14167),  INT16_C(  3180), -INT16_C( 22395), -INT16_C( 15546),  INT16_C(  6403), -INT16_C( 19531) },
      UINT8_C( 56),
      {  INT16_C( 18184), -INT16_C( 17629), -INT16_C(  6796),  INT16_C( 24196),  INT16_C( 13437),  INT16_C(   863), -INT16_C( 10480),  INT16_C( 23202) },
      UINT32_C(         3),
      { -INT16_C( 24969), -INT16_C( 30390), -INT16_C( 14167), -INT16_C(  3040), -INT16_C( 23576),  INT16_C(  6904),  INT16_C(  6403), -INT16_C( 19531) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi16(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_slli_epi16(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_slli_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_mask_slli_epi16(src, k, a, imm8);

    easysimd_test_x86_write_i16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_mask_slli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const int32_t a[4];
    const uint32_t imm8;
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(   507309944), -INT32_C(  1825175957),  INT32_C(  1772847558),  INT32_C(   549284870) },
      UINT8_C(154),
      { -INT32_C(  1320524597),  INT32_C(  1443400285), -INT32_C(   854249124),  INT32_C(  1633075792) },
      UINT32_C(        10),
      { -INT32_C(   507309944),  INT32_C(   573142016),  INT32_C(  1772847558),  INT32_C(  1527332864) } },
    { {  INT32_C(  1837878327),  INT32_C(   183390815),  INT32_C(  2037839035),  INT32_C(  2017726484) },
      UINT8_C( 91),
      { -INT32_C(    18491659),  INT32_C(  1093615916),  INT32_C(  1694728207), -INT32_C(  1382250271) },
      UINT32_C(         1),
      { -INT32_C(    36983318), -INT32_C(  2107735464),  INT32_C(  2037839035),  INT32_C(  1530466754) } },
    { { -INT32_C(   144372727), -INT32_C(  1930747626), -INT32_C(   560070758),  INT32_C(  1305802871) },
      UINT8_C(221),
      {  INT32_C(    18840018), -INT32_C(  1115607366),  INT32_C(   793076371),  INT32_C(   456677647) },
      UINT32_C(         7),
      { -INT32_C(  1883444992), -INT32_C(  1930747626), -INT32_C(  1565439616), -INT32_C(  1674803328) } },
    { {  INT32_C(   450834735),  INT32_C(  1528396222),  INT32_C(   743673944),  INT32_C(  1476276446) },
      UINT8_C( 79),
      { -INT32_C(  2122509569),  INT32_C(   367202511),  INT32_C(  2003759935),  INT32_C(  1084632078) },
      UINT32_C(        13),
      { -INT32_C(  1570775040),  INT32_C(  1645862912), -INT32_C(   563617792), -INT32_C(   981352448) } },
    { { -INT32_C(   645529920), -INT32_C(  1402282663),  INT32_C(   131876873), -INT32_C(  1324929889) },
      UINT8_C(169),
      {  INT32_C(  1807384968),  INT32_C(    73333142),  INT32_C(   389049713), -INT32_C(  1411892051) },
      UINT32_C(        19),
      {  INT32_C(   205520896), -INT32_C(  1402282663),  INT32_C(   131876873), -INT32_C(   446169088) } },
    { {  INT32_C(   463406257), -INT32_C(  1939625552), -INT32_C(  1212677200),  INT32_C(   893346484) },
      UINT8_C( 28),
      {  INT32_C(   152423339),  INT32_C(     7767759),  INT32_C(  1983652766), -INT32_C(   769130802) },
      UINT32_C(        27),
      {  INT32_C(   463406257), -INT32_C(  1939625552), -INT32_C(   268435456),  INT32_C(  1879048192) } },
    { { -INT32_C(  1505263037), -INT32_C(   945162481), -INT32_C(   299278673),  INT32_C(  1603880339) },
      UINT8_C( 91),
      {  INT32_C(   434187939),  INT32_C(  1799192366),  INT32_C(   493423605),  INT32_C(  1616906206) },
      UINT32_C(        17),
      {  INT32_C(  1564868608), -INT32_C(    27525120), -INT32_C(   299278673),  INT32_C(   263979008) } },
    { { -INT32_C(  1337561338),  INT32_C(  1611593015),  INT32_C(  2108072419),  INT32_C(   790626560) },
      UINT8_C(226),
      {  INT32_C(  1986157881),  INT32_C(   847402952), -INT32_C(   734437260),  INT32_C(   819693760) },
      UINT32_C(        37),
      { -INT32_C(  1337561338),  INT32_C(           0),  INT32_C(  2108072419),  INT32_C(   790626560) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_slli_epi32(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_slli_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m128i r = easysimd_mm_mask_slli_epi32(src, k, a, imm8);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_mask_slli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const easysimd__mmask8 k;
    const int64_t a[2];
    const uint32_t imm8;
    const int64_t r[2];
  } test_vec[] = {
    { { -INT64_C( 8140957326133637546),  INT64_C( 8523997435335188792) },
      UINT8_C( 99),
      {  INT64_C( 7023058543914710142), -INT64_C( 4985274864623350966) },
      UINT32_C(        35),
      { -INT64_C( 7604429194604314624), -INT64_C( 3204211835230224384) } },
    { {  INT64_C( 4054270708601138076),  INT64_C( 3446078603480712201) },
      UINT8_C(188),
      { -INT64_C( 8392475388342257659),  INT64_C( 6473985136047780715) },
      UINT32_C(        15),
      {  INT64_C( 4054270708601138076),  INT64_C( 3446078603480712201) } },
    { { -INT64_C( 2917950698424232839), -INT64_C( 6663495750153552721) },
      UINT8_C( 40),
      { -INT64_C( 7927316896517931774), -INT64_C( 7489361366021035329) },
      UINT32_C(        44),
      { -INT64_C( 2917950698424232839), -INT64_C( 6663495750153552721) } },
    { { -INT64_C( 9220331082734443315),  INT64_C( 6185459449113802062) },
      UINT8_C( 66),
      { -INT64_C(  610372790313902752), -INT64_C( 8648833629563647204) },
      UINT32_C(        22),
      { -INT64_C( 9220331082734443315), -INT64_C( 1672214237407084544) } },
    { { -INT64_C( 4231584009826188473), -INT64_C( 3283068929149247687) },
      UINT8_C(248),
      {  INT64_C( 6872380843944116429), -INT64_C( 3745073451534515515) },
      UINT32_C(        72),
      { -INT64_C( 4231584009826188473), -INT64_C( 3283068929149247687) } },
    { {  INT64_C( 5007307162503778392), -INT64_C( 8875064877795265896) },
      UINT8_C(236),
      {  INT64_C( 7388800445403462697),  INT64_C( 8099763375077659657) },
      UINT32_C(        67),
      {  INT64_C( 5007307162503778392), -INT64_C( 8875064877795265896) } },
    { { -INT64_C( 9063898820886608378), -INT64_C( 5001004125975889255) },
      UINT8_C( 98),
      {  INT64_C( 3873015400890144288), -INT64_C( 3011561179469799482) },
      UINT32_C(        71),
      { -INT64_C( 9063898820886608378),  INT64_C(                   0) } },
    { {  INT64_C( 6100384509706035640),  INT64_C( 3968396545429202778) },
      UINT8_C( 63),
      { -INT64_C( 4531541063299157059), -INT64_C( 2029349910586118935) },
      UINT32_C(        32),
      {  INT64_C( 9004159916839010304),  INT64_C( 2200902222139621376) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_slli_epi64(src, k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_mask_slli_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m128i r = easysimd_mm_mask_slli_epi64(src, k, a, imm8);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_maskz_slli_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int16_t a[8];
    const uint32_t imm8;
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C(120),
      {  INT16_C( 29569),  INT16_C(  6286),  INT16_C(  3844),  INT16_C( 17522), -INT16_C( 25101), -INT16_C(  8617), -INT16_C( 21592), -INT16_C( 30899) },
      UINT32_C(         4),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 18208), -INT16_C(  8400), -INT16_C(  6800), -INT16_C( 17792),  INT16_C(     0) } },
    { UINT8_C(175),
      {  INT16_C( 26406),  INT16_C( 28983), -INT16_C( 17512), -INT16_C(  6111), -INT16_C( 16431),  INT16_C(  8032), -INT16_C( 11456),  INT16_C( 22701) },
      UINT32_C(         4),
      {  INT16_C( 29280),  INT16_C(  4976), -INT16_C( 18048), -INT16_C( 32240),  INT16_C(     0), -INT16_C(  2560),  INT16_C(     0), -INT16_C( 30000) } },
    { UINT8_C(188),
      { -INT16_C( 20709),  INT16_C( 29288),  INT16_C(  4494), -INT16_C(  9443), -INT16_C(  6504), -INT16_C( 21366), -INT16_C(  3828),  INT16_C( 32227) },
      UINT32_C(         0),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(  4494), -INT16_C(  9443), -INT16_C(  6504), -INT16_C( 21366),  INT16_C(     0),  INT16_C( 32227) } },
    { UINT8_C(158),
      {  INT16_C( 28529), -INT16_C( 11938), -INT16_C( 24946),  INT16_C( 15268),  INT16_C( 31735), -INT16_C( 15625), -INT16_C( 22634),  INT16_C(  2090) },
      UINT32_C(         0),
      {  INT16_C(     0), -INT16_C( 11938), -INT16_C( 24946),  INT16_C( 15268),  INT16_C( 31735),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2090) } },
    { UINT8_C( 59),
      { -INT16_C( 11504), -INT16_C( 26101),  INT16_C(  6015),  INT16_C( 25227),  INT16_C(  5269),  INT16_C( 13057),  INT16_C( 28805),  INT16_C( 22161) },
      UINT32_C(         4),
      {  INT16_C( 12544), -INT16_C( 24400),  INT16_C(     0),  INT16_C( 10416),  INT16_C( 18768),  INT16_C( 12304),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 48),
      {  INT16_C( 10042),  INT16_C( 12917),  INT16_C(  3049),  INT16_C(  5081),  INT16_C(  3603),  INT16_C( 14415),  INT16_C(  8734), -INT16_C( 18364) },
      UINT32_C(        12),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 12288), -INT16_C(  4096),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 91),
      { -INT16_C(  4092),  INT16_C(  1367), -INT16_C(  9180), -INT16_C( 19082),  INT16_C( 30002),  INT16_C( 11493),  INT16_C(  3247), -INT16_C(  7775) },
      UINT32_C(         1),
      { -INT16_C(  8184),  INT16_C(  2734),  INT16_C(     0),  INT16_C( 27372), -INT16_C(  5532),  INT16_C(     0),  INT16_C(  6494),  INT16_C(     0) } },
    { UINT8_C(172),
      { -INT16_C( 16631),  INT16_C( 22728), -INT16_C(  6409),  INT16_C( 15226),  INT16_C(  7326), -INT16_C(  7785), -INT16_C( 30943),  INT16_C(  9784) },
      UINT32_C(         6),
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 16960), -INT16_C(  8576),  INT16_C(     0),  INT16_C( 26048),  INT16_C(     0), -INT16_C( 29184) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_slli_epi16(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_slli_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 20.0);
    easysimd__m128i r = easysimd_mm_maskz_slli_epi16(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_maskz_slli_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t a[4];
    const uint32_t imm8;
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(250),
      {  INT32_C(   238959600),  INT32_C(  1210507164),  INT32_C(   342273596),  INT32_C(  1773419623) },
      UINT32_C(         0),
      {  INT32_C(           0),  INT32_C(  1210507164),  INT32_C(           0),  INT32_C(  1773419623) } },
    { UINT8_C(233),
      { -INT32_C(   465085932), -INT32_C(  1906059361),  INT32_C(  1515956615), -INT32_C(   493532253) },
      UINT32_C(        11),
      {  INT32_C(   986750976),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1436739584) } },
    { UINT8_C( 49),
      { -INT32_C(    25526079), -INT32_C(  1777147857),  INT32_C(   486524700), -INT32_C(   143256400) },
      UINT32_C(        26),
      {  INT32_C(    67108864),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 21),
      {  INT32_C(  1453611983), -INT32_C(  1464795388), -INT32_C(  1534441910), -INT32_C(   249214089) },
      UINT32_C(        26),
      {  INT32_C(  1006632960),  INT32_C(           0),  INT32_C(   671088640),  INT32_C(           0) } },
    { UINT8_C( 84),
      { -INT32_C(   286630703), -INT32_C(   754259677), -INT32_C(   154435075),  INT32_C(   978426518) },
      UINT32_C(        38),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 86),
      { -INT32_C(  2063687367),  INT32_C(  1579714791),  INT32_C(   491802067), -INT32_C(  1954995807) },
      UINT32_C(         3),
      {  INT32_C(           0), -INT32_C(   247183560), -INT32_C(   360550760),  INT32_C(           0) } },
    { UINT8_C(156),
      {  INT32_C(  1919883893),  INT32_C(   828914331), -INT32_C(   982730028),  INT32_C(   241607952) },
      UINT32_C(        23),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1778384896), -INT32_C(  2013265920) } },
    { UINT8_C( 77),
      {  INT32_C(  1772835478),  INT32_C(  1099365279),  INT32_C(   751632157),  INT32_C(   189153691) },
      UINT32_C(        12),
      { -INT32_C(  1255579648),  INT32_C(           0), -INT32_C(   806236160),  INT32_C(  1679405056) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_slli_epi32(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_slli_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 40.0);
    easysimd__m128i r = easysimd_mm_maskz_slli_epi32(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

static int
test_easysimd_mm_maskz_slli_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int64_t a[2];
    const uint32_t imm8;
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(149),
      {  INT64_C( 7845450076042948756),  INT64_C( 5865007161669753358) },
      UINT32_C(        18),
      { -INT64_C( 8278787748860526592),  INT64_C(                   0) } },
    { UINT8_C(  8),
      {  INT64_C( 4834915191332182032),  INT64_C( 5281126629442067204) },
      UINT32_C(        29),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(144),
      { -INT64_C( 5514534856178193939),  INT64_C( 7072645811204840479) },
      UINT32_C(        39),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(232),
      { -INT64_C(  229143392353046156), -INT64_C( 6218851319314769089) },
      UINT32_C(        65),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(118),
      {  INT64_C( 7477715368101389848),  INT64_C(  111250606656187971) },
      UINT32_C(        63),
      {  INT64_C(                   0),                      INT64_MIN } },
    { UINT8_C(215),
      {  INT64_C( 5671187114588669932),  INT64_C(  441933157377730381) },
      UINT32_C(        23),
      {  INT64_C( 5842833669193990144), -INT64_C( 1243561173073264640) } },
    { UINT8_C(164),
      { -INT64_C( 3327361898662654668),  INT64_C(  670964007713266962) },
      UINT32_C(        62),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(241),
      { -INT64_C( 2095520705113186781), -INT64_C( 4784479001405551289) },
      UINT32_C(        60),
      {  INT64_C( 3458764513820540928),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    uint32_t imm8 = test_vec[i].imm8;
    const easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_slli_epi64(k, a, imm8);
    }
    EASYSIMD_TEST_PERF_END("_mm_maskz_slli_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);

  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    uint32_t imm8 = (uint32_t)easysimd_test_codegen_random_f32(0.0, 75.0);
    easysimd__m128i r = easysimd_mm_maskz_slli_epi64(k, a, imm8);

    easysimd_test_codegen_write_u8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_u32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_slli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_slli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_slli_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_slli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_slli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_slli_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_slli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_slli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_slli_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_slli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_slli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_slli_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_slli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_slli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_slli_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_slli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_slli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_slli_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_slli_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_slli_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_slli_epi64)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
