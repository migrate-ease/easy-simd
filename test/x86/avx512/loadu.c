#define EASYSIMD_TEST_X86_AVX512_INSN loadu

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/loadu.h>
#include <easysimd/x86/avx512/set.h>

static int
test_easysimd_mm_mask_loadu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__m128i src;
    easysimd__mmask16 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[] = {
    { easysimd_mm_set_epi8( INT8_C( 108), -INT8_C(  29), -INT8_C( 103), -INT8_C(  38),  INT8_C( 111),  INT8_C(  42), -INT8_C(   5), -INT8_C(  48),
         INT8_C( 107),  INT8_C(  94), -INT8_C(   1), -INT8_C(  23),  INT8_C(  98),  INT8_C(  29),  INT8_C(  74),  INT8_C( 111)),
      UINT16_C( 1701),
      easysimd_mm_set_epi8( INT8_C( 125), -INT8_C(  14), -INT8_C(  55),  INT8_C(  96), -INT8_C(  88),  INT8_C(  90),  INT8_C(  79), -INT8_C(  45),
        -INT8_C(  48), -INT8_C(  22), -INT8_C(   3),  INT8_C(  33),  INT8_C(  17), -INT8_C(  14), -INT8_C( 108),  INT8_C( 115)),
      easysimd_mm_set_epi8( INT8_C( 108), -INT8_C(  29), -INT8_C( 103), -INT8_C(  38),  INT8_C( 111),  INT8_C(  90),  INT8_C(  79), -INT8_C(  48),
        -INT8_C(  48),  INT8_C(  94), -INT8_C(   3), -INT8_C(  23),  INT8_C(  98), -INT8_C(  14),  INT8_C(  74),  INT8_C( 115)) },
    { easysimd_mm_set_epi8( INT8_C(  63), -INT8_C(  90), -INT8_C(  80), -INT8_C(  53), -INT8_C(  96),  INT8_C(  11),  INT8_C(  95), -INT8_C(  67),
         INT8_C( 113), -INT8_C( 123),  INT8_C(  77),  INT8_C(  71), -INT8_C( 118),  INT8_C( 125), -INT8_C(  36),  INT8_C(  44)),
      UINT16_C(39237),
      easysimd_mm_set_epi8(-INT8_C(  28), -INT8_C(  32), -INT8_C( 123),  INT8_C( 103),  INT8_C(   4),  INT8_C(  89), -INT8_C(  22),  INT8_C(  17),
        -INT8_C( 112), -INT8_C( 119),  INT8_C( 105),  INT8_C(  54),  INT8_C(  58), -INT8_C( 106),  INT8_C( 102),  INT8_C(  80)),
      easysimd_mm_set_epi8(-INT8_C(  28), -INT8_C(  90), -INT8_C(  80),  INT8_C( 103),  INT8_C(   4),  INT8_C(  11),  INT8_C(  95),  INT8_C(  17),
         INT8_C( 113), -INT8_C( 119),  INT8_C(  77),  INT8_C(  71), -INT8_C( 118), -INT8_C( 106), -INT8_C(  36),  INT8_C(  80)) },
    { easysimd_mm_set_epi8( INT8_C(  79), -INT8_C(  49), -INT8_C( 103), -INT8_C(   2),  INT8_C(  54),  INT8_C(  84), -INT8_C(  65), -INT8_C( 113),
        -INT8_C(  92), -INT8_C(  12), -INT8_C(  17), -INT8_C( 103), -INT8_C( 108),  INT8_C(  50),  INT8_C(  39),  INT8_C(  15)),
      UINT16_C(26111),
      easysimd_mm_set_epi8( INT8_C( 123), -INT8_C(  21), -INT8_C(  76),  INT8_C(  73), -INT8_C(  61), -INT8_C(  92),  INT8_C( 100), -INT8_C(  29),
         INT8_C(  31), -INT8_C(   3), -INT8_C(  33), -INT8_C(  59),  INT8_C(  19), -INT8_C(  50),  INT8_C(  53), -INT8_C( 119)),
      easysimd_mm_set_epi8( INT8_C(  79), -INT8_C(  21), -INT8_C(  76), -INT8_C(   2),  INT8_C(  54), -INT8_C(  92), -INT8_C(  65), -INT8_C(  29),
         INT8_C(  31), -INT8_C(   3), -INT8_C(  33), -INT8_C(  59),  INT8_C(  19), -INT8_C(  50),  INT8_C(  53), -INT8_C( 119)) },
    { easysimd_mm_set_epi8(-INT8_C(  45),  INT8_C(  99),  INT8_C(  21),  INT8_C(  73), -INT8_C(   2),  INT8_C(  22), -INT8_C(   6),  INT8_C(  47),
         INT8_C( 124), -INT8_C(   4), -INT8_C(   7),  INT8_C(  40),  INT8_C(  60),  INT8_C( 106), -INT8_C( 124),  INT8_C(  72)),
      UINT16_C(12619),
      easysimd_mm_set_epi8( INT8_C( 117),  INT8_C(  39), -INT8_C(  48),  INT8_C(  11), -INT8_C(  93), -INT8_C( 120), -INT8_C( 112), -INT8_C(  72),
        -INT8_C(  44),  INT8_C(  71), -INT8_C(  12),  INT8_C(  47), -INT8_C(  29),  INT8_C(  17),  INT8_C(  16), -INT8_C(  26)),
      easysimd_mm_set_epi8(-INT8_C(  45),  INT8_C(  99), -INT8_C(  48),  INT8_C(  11), -INT8_C(   2),  INT8_C(  22), -INT8_C(   6), -INT8_C(  72),
         INT8_C( 124),  INT8_C(  71), -INT8_C(   7),  INT8_C(  40), -INT8_C(  29),  INT8_C( 106),  INT8_C(  16), -INT8_C(  26)) },
    { easysimd_mm_set_epi8( INT8_C(   6),  INT8_C(  50),  INT8_C(  66),  INT8_C(  32),  INT8_C(   0), -INT8_C(   9),  INT8_C(  77), -INT8_C(  99),
        -INT8_C(  31),  INT8_C(   3), -INT8_C(  98), -INT8_C(  53),  INT8_C(   9),  INT8_C( 111),  INT8_C(  79),  INT8_C(  13)),
      UINT16_C(17234),
      easysimd_mm_set_epi8(-INT8_C(  79),  INT8_C(   8), -INT8_C(  69),  INT8_C(  66), -INT8_C(  71), -INT8_C(  82), -INT8_C(  52), -INT8_C( 110),
        -INT8_C(  34), -INT8_C(  63), -INT8_C(  17),  INT8_C(  86),  INT8_C(  48),  INT8_C(  55), -INT8_C( 126), -INT8_C(  23)),
      easysimd_mm_set_epi8( INT8_C(   6),  INT8_C(   8),  INT8_C(  66),  INT8_C(  32),  INT8_C(   0), -INT8_C(   9), -INT8_C(  52), -INT8_C( 110),
        -INT8_C(  31), -INT8_C(  63), -INT8_C(  98),  INT8_C(  86),  INT8_C(   9),  INT8_C( 111), -INT8_C( 126),  INT8_C(  13)) },
    { easysimd_mm_set_epi8( INT8_C(  36),  INT8_C(  98),  INT8_C(  65),  INT8_C(  59),  INT8_C(  31), -INT8_C(  18),  INT8_C(  53), -INT8_C(  19),
        -INT8_C(  84),  INT8_C(  21), -INT8_C(  20), -INT8_C(  75), -INT8_C(  56),  INT8_C(  79), -INT8_C(  44), -INT8_C(  60)),
      UINT16_C(39363),
      easysimd_mm_set_epi8( INT8_C(  36), -INT8_C(  79),  INT8_C(  37), -INT8_C(  43), -INT8_C(  35),  INT8_C(  97),  INT8_C(  36), -INT8_C(  43),
        -INT8_C(  91), -INT8_C(  30),  INT8_C(  27), -INT8_C(   9),  INT8_C(  21), -INT8_C( 119),  INT8_C(  25),  INT8_C(  84)),
      easysimd_mm_set_epi8( INT8_C(  36),  INT8_C(  98),  INT8_C(  65), -INT8_C(  43), -INT8_C(  35), -INT8_C(  18),  INT8_C(  53), -INT8_C(  43),
        -INT8_C(  91), -INT8_C(  30), -INT8_C(  20), -INT8_C(  75), -INT8_C(  56),  INT8_C(  79),  INT8_C(  25),  INT8_C(  84)) },
    { easysimd_mm_set_epi8(-INT8_C(  21),  INT8_C(  24),  INT8_C(   6), -INT8_C( 106),      INT8_MAX,  INT8_C(  67),  INT8_C( 114),  INT8_C(  29),
         INT8_C(   2),  INT8_C(  55), -INT8_C(   2),  INT8_C(  19),  INT8_C(   2),  INT8_C(  17),  INT8_C( 103), -INT8_C(  19)),
      UINT16_C(41247),
      easysimd_mm_set_epi8( INT8_C(  17), -INT8_C( 120),  INT8_C(  47),  INT8_C(   0),  INT8_C(  33),  INT8_C(  66), -INT8_C(  37),  INT8_C( 111),
         INT8_C(  28),  INT8_C(   6), -INT8_C( 110), -INT8_C(  69), -INT8_C(  30), -INT8_C(  67),  INT8_C(  22),  INT8_C(   0)),
      easysimd_mm_set_epi8( INT8_C(  17),  INT8_C(  24),  INT8_C(  47), -INT8_C( 106),      INT8_MAX,  INT8_C(  67),  INT8_C( 114),  INT8_C( 111),
         INT8_C(   2),  INT8_C(  55), -INT8_C(   2), -INT8_C(  69), -INT8_C(  30), -INT8_C(  67),  INT8_C(  22),  INT8_C(   0)) },
    { easysimd_mm_set_epi8( INT8_C(  94),  INT8_C( 101),  INT8_C(   5),  INT8_C(  93), -INT8_C(  61), -INT8_C(  26),  INT8_C( 114), -INT8_C(  85),
        -INT8_C(  32), -INT8_C(  36),  INT8_C(  44), -INT8_C(  99),  INT8_C( 105),  INT8_C(  15), -INT8_C( 101),  INT8_C(  50)),
      UINT16_C( 8731),
      easysimd_mm_set_epi8( INT8_C(  66),  INT8_C( 104), -INT8_C( 105),  INT8_C(  51), -INT8_C(  52),  INT8_C( 101),  INT8_C(  34),  INT8_C(  68),
         INT8_C(  53),  INT8_C(  34),  INT8_C(  35), -INT8_C(  13),  INT8_C(  71), -INT8_C(  76), -INT8_C(  41),  INT8_C(  64)),
      easysimd_mm_set_epi8( INT8_C(  94),  INT8_C( 101), -INT8_C( 105),  INT8_C(  93), -INT8_C(  61), -INT8_C(  26),  INT8_C(  34), -INT8_C(  85),
        -INT8_C(  32), -INT8_C(  36),  INT8_C(  44), -INT8_C(  13),  INT8_C(  71),  INT8_C(  15), -INT8_C(  41),  INT8_C(  64)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_loadu_epi8(test_vec[i].src, test_vec[i].k, &(test_vec[i].a));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_loadu_epi8");
    easysimd_test_x86_assert_equal_i8x16(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i8x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i r = easysimd_mm_mask_loadu_epi8(src, k, &a);
    src = easysimd_mm_set_epi8(src.i8[0], src.i8[1], src.i8[2], src.i8[3], src.i8[4], src.i8[5], src.i8[6], src.i8[7], src.i8[8], src.i8[9], src.i8[10], src.i8[11], src.i8[12], src.i8[13], src.i8[14], src.i8[15]);
    a = easysimd_mm_set_epi8(a.i8[0], a.i8[1], a.i8[2], a.i8[3], a.i8[4], a.i8[5], a.i8[6], a.i8[7], a.i8[8], a.i8[9], a.i8[10], a.i8[11],a.i8[12], a.i8[13], a.i8[14], a.i8[15]);
    r = easysimd_mm_set_epi8(r.i8[0], r.i8[1], r.i8[2], r.i8[3], r.i8[4], r.i8[5], r.i8[6], r.i8[7], r.i8[8], r.i8[9], r.i8[10], r.i8[11],r.i8[12], r.i8[13], r.i8[14], r.i8[15]);

    easysimd_test_x86_write_i8x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_loadu_epi8 (EASYSIMD_MUNIT_TEST_ARGS)
{
#if 1
  static const struct {
    easysimd__mmask16 k;
    const int8_t a[16];
    const int8_t r[16];
  } test_vec[] = {
    { UINT16_C(51498),
      {  INT8_C( 115), -INT8_C( 115), -INT8_C(  55),  INT8_C( 111),  INT8_C( 121),      INT8_MAX, -INT8_C(   7), -INT8_C(  53),
        -INT8_C( 104), -INT8_C(  77), -INT8_C( 106), -INT8_C( 101), -INT8_C(  71), -INT8_C(  19),  INT8_C(  84), -INT8_C( 115) },
      {  INT8_C(   0), -INT8_C( 115),  INT8_C(   0),  INT8_C( 111),  INT8_C(   0),      INT8_MAX,  INT8_C(   0),  INT8_C(   0),
        -INT8_C( 104),  INT8_C(   0),  INT8_C(   0), -INT8_C( 101),  INT8_C(   0),  INT8_C(   0),  INT8_C(  84), -INT8_C( 115) } },
    { UINT16_C(59327),
      { -INT8_C(  88), -INT8_C( 104), -INT8_C(  17), -INT8_C(  93), -INT8_C( 124), -INT8_C(  81),  INT8_C(  42), -INT8_C(  62),
        -INT8_C(  92),  INT8_C(  62),  INT8_C( 108), -INT8_C(  50),  INT8_C(   7), -INT8_C(  33),  INT8_C(  91), -INT8_C(  47) },
      { -INT8_C(  88), -INT8_C( 104), -INT8_C(  17), -INT8_C(  93), -INT8_C( 124), -INT8_C(  81),  INT8_C(   0), -INT8_C(  62),
        -INT8_C(  92),  INT8_C(  62),  INT8_C( 108),  INT8_C(   0),  INT8_C(   0), -INT8_C(  33),  INT8_C(  91), -INT8_C(  47) } },
    { UINT16_C(54350),
      {  INT8_C(  80),  INT8_C(  72), -INT8_C(  97), -INT8_C(  24), -INT8_C(   5),  INT8_C(  53), -INT8_C( 124), -INT8_C(  76),
         INT8_C(  35), -INT8_C(  40),  INT8_C(  66), -INT8_C(  30), -INT8_C(  64), -INT8_C(  22),  INT8_C( 123), -INT8_C(  81) },
      {  INT8_C(   0),  INT8_C(  72), -INT8_C(  97), -INT8_C(  24),  INT8_C(   0),  INT8_C(   0), -INT8_C( 124),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  66),  INT8_C(   0), -INT8_C(  64),  INT8_C(   0),  INT8_C( 123), -INT8_C(  81) } },
    { UINT16_C(65421),
      {  INT8_C(  94), -INT8_C(  72), -INT8_C(  63),  INT8_C(   2), -INT8_C(  10),  INT8_C(  45), -INT8_C(  48), -INT8_C(   3),
         INT8_C(  12),  INT8_C(  43), -INT8_C(  50),  INT8_C(  91),  INT8_C(   0),  INT8_C(  30), -INT8_C(  93), -INT8_C(  97) },
      {  INT8_C(  94),  INT8_C(   0), -INT8_C(  63),  INT8_C(   2),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(   3),
         INT8_C(  12),  INT8_C(  43), -INT8_C(  50),  INT8_C(  91),  INT8_C(   0),  INT8_C(  30), -INT8_C(  93), -INT8_C(  97) } },
    { UINT16_C(40455),
      { -INT8_C(  43), -INT8_C( 117),  INT8_C(  82), -INT8_C(   8),  INT8_C(  99), -INT8_C( 108), -INT8_C(  38),  INT8_C(  35),
             INT8_MAX,  INT8_C(  85), -INT8_C(  46),  INT8_C(  12),  INT8_C(  85),  INT8_C(  48), -INT8_C(  60),  INT8_C(  22) },
      { -INT8_C(  43), -INT8_C( 117),  INT8_C(  82),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  85), -INT8_C(  46),  INT8_C(  12),  INT8_C(  85),  INT8_C(   0),  INT8_C(   0),  INT8_C(  22) } },
    { UINT16_C(47666),
      {  INT8_C(  68),  INT8_C(   2), -INT8_C(  72),  INT8_C(  80),  INT8_C(  46), -INT8_C( 122), -INT8_C(  85),  INT8_C(  46),
        -INT8_C(  91),  INT8_C(  78), -INT8_C(  51), -INT8_C(  84), -INT8_C(  20), -INT8_C(  94),  INT8_C(  55),  INT8_C(  63) },
      {  INT8_C(   0),  INT8_C(   2),  INT8_C(   0),  INT8_C(   0),  INT8_C(  46), -INT8_C( 122),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  78),  INT8_C(   0), -INT8_C(  84), -INT8_C(  20), -INT8_C(  94),  INT8_C(   0),  INT8_C(  63) } },
    { UINT16_C(39578),
      { -INT8_C(  45),  INT8_C( 117), -INT8_C(  66),  INT8_C(  82), -INT8_C(  54), -INT8_C( 112),  INT8_C(  95),  INT8_C(  31),
        -INT8_C(  63),  INT8_C(  35),  INT8_C(  54), -INT8_C(  13), -INT8_C(  34),  INT8_C( 122), -INT8_C(  10), -INT8_C( 106) },
      {  INT8_C(   0),  INT8_C( 117),  INT8_C(   0),  INT8_C(  82), -INT8_C(  54),  INT8_C(   0),  INT8_C(   0),  INT8_C(  31),
         INT8_C(   0),  INT8_C(  35),  INT8_C(   0), -INT8_C(  13), -INT8_C(  34),  INT8_C(   0),  INT8_C(   0), -INT8_C( 106) } },
    { UINT16_C( 9418),
      {  INT8_C(  28),  INT8_C( 118),  INT8_C(  82), -INT8_C(  63), -INT8_C(  60),  INT8_C(  31),  INT8_C( 109), -INT8_C(  79),
        -INT8_C(  62), -INT8_C(  92), -INT8_C(  16),  INT8_C(  92),  INT8_C(  63), -INT8_C(  61), -INT8_C(  47), -INT8_C(   3) },
      {  INT8_C(   0),  INT8_C( 118),  INT8_C(   0), -INT8_C(  63),  INT8_C(   0),  INT8_C(   0),  INT8_C( 109), -INT8_C(  79),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  16),  INT8_C(   0),  INT8_C(   0), -INT8_C(  61),  INT8_C(   0),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi8(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_loadu_epi8(k, &a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_loadu_epi8");
    easysimd_assert_equal_vi8(sizeof(r), (const int8_t*)&r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m128i a = easysimd_test_x86_random_i8x16();
    easysimd__m128i r = easysimd_mm_maskz_loadu_epi8(k, &a.i8[0]);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_loadu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__m128i src;
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[] = {
    { easysimd_mm_set_epi16( INT16_C( 19300), -INT16_C(  4854),  INT16_C( 17106), -INT16_C( 21283), -INT16_C( 13233),  INT16_C(  6630), -INT16_C(  9106),  INT16_C(  1280)),
      UINT8_C(196),
      easysimd_mm_set_epi16(-INT16_C( 17686),  INT16_C( 19892), -INT16_C(  5878),  INT16_C( 19538), -INT16_C( 10368), -INT16_C(  4684),  INT16_C( 15288), -INT16_C( 28136)),
      easysimd_mm_set_epi16(-INT16_C( 17686),  INT16_C( 19892),  INT16_C( 17106), -INT16_C( 21283), -INT16_C( 13233), -INT16_C(  4684), -INT16_C(  9106),  INT16_C(  1280)) },
    { easysimd_mm_set_epi16( INT16_C( 28912),  INT16_C(  5976),  INT16_C( 11468), -INT16_C(  3010), -INT16_C( 15694),  INT16_C( 27669), -INT16_C( 10848), -INT16_C( 14661)),
      UINT8_C(169),
      easysimd_mm_set_epi16(-INT16_C( 32184), -INT16_C(  6468), -INT16_C( 29652), -INT16_C( 11713),  INT16_C( 30952),  INT16_C( 13611), -INT16_C( 27042), -INT16_C( 21592)),
      easysimd_mm_set_epi16(-INT16_C( 32184),  INT16_C(  5976), -INT16_C( 29652), -INT16_C(  3010),  INT16_C( 30952),  INT16_C( 27669), -INT16_C( 10848), -INT16_C( 21592)) },
    { easysimd_mm_set_epi16(-INT16_C(  2329), -INT16_C( 29874),  INT16_C( 15899),  INT16_C( 23846), -INT16_C( 15567),  INT16_C( 23247), -INT16_C(  3176),  INT16_C(  7558)),
      UINT8_C( 54),
      easysimd_mm_set_epi16(-INT16_C( 21645), -INT16_C( 31346), -INT16_C(  4861),  INT16_C( 17926),  INT16_C( 18362), -INT16_C(  9611),  INT16_C( 31330), -INT16_C( 29627)),
      easysimd_mm_set_epi16(-INT16_C(  2329), -INT16_C( 29874), -INT16_C(  4861),  INT16_C( 17926), -INT16_C( 15567), -INT16_C(  9611),  INT16_C( 31330),  INT16_C(  7558)) },
    { easysimd_mm_set_epi16( INT16_C(  2330),  INT16_C( 15043), -INT16_C(  7357), -INT16_C(  9128), -INT16_C(  2658),  INT16_C( 15768),  INT16_C( 30586),  INT16_C( 26141)),
      UINT8_C(198),
      easysimd_mm_set_epi16( INT16_C(  9000), -INT16_C( 16707),  INT16_C(  2834),  INT16_C( 18821), -INT16_C( 31652), -INT16_C( 32195),  INT16_C( 22075), -INT16_C( 31876)),
      easysimd_mm_set_epi16( INT16_C(  9000), -INT16_C( 16707), -INT16_C(  7357), -INT16_C(  9128), -INT16_C(  2658), -INT16_C( 32195),  INT16_C( 22075),  INT16_C( 26141)) },
    { easysimd_mm_set_epi16( INT16_C( 18677), -INT16_C(  6452),  INT16_C( 11997), -INT16_C( 19724),  INT16_C(  6862), -INT16_C( 20162),  INT16_C( 30395), -INT16_C( 24520)),
      UINT8_C(106),
      easysimd_mm_set_epi16( INT16_C(  3124),  INT16_C(  9324), -INT16_C(  1023),  INT16_C( 17214),  INT16_C( 17464),  INT16_C( 11258), -INT16_C( 19545), -INT16_C( 25040)),
      easysimd_mm_set_epi16( INT16_C( 18677),  INT16_C(  9324), -INT16_C(  1023), -INT16_C( 19724),  INT16_C( 17464), -INT16_C( 20162), -INT16_C( 19545), -INT16_C( 24520)) },
    { easysimd_mm_set_epi16(-INT16_C(  9838),  INT16_C( 16297),  INT16_C( 10487), -INT16_C( 19391),  INT16_C( 11141),  INT16_C( 25721), -INT16_C( 28342), -INT16_C( 21792)),
      UINT8_C(221),
      easysimd_mm_set_epi16( INT16_C( 32289), -INT16_C( 32557),  INT16_C( 16756), -INT16_C( 24804),  INT16_C(  2211),  INT16_C(  7109),  INT16_C( 26071), -INT16_C( 29639)),
      easysimd_mm_set_epi16( INT16_C( 32289), -INT16_C( 32557),  INT16_C( 10487), -INT16_C( 24804),  INT16_C(  2211),  INT16_C(  7109), -INT16_C( 28342), -INT16_C( 29639)) },
    { easysimd_mm_set_epi16( INT16_C( 25397),  INT16_C( 28202),  INT16_C( 22676), -INT16_C( 26599), -INT16_C(  5264),  INT16_C(  8759),  INT16_C( 12023), -INT16_C( 19766)),
      UINT8_C(250),
      easysimd_mm_set_epi16( INT16_C( 22663),  INT16_C( 23718), -INT16_C( 16930), -INT16_C( 31428),  INT16_C(  2628), -INT16_C( 14229),  INT16_C( 10431), -INT16_C( 14324)),
      easysimd_mm_set_epi16( INT16_C( 22663),  INT16_C( 23718), -INT16_C( 16930), -INT16_C( 31428),  INT16_C(  2628),  INT16_C(  8759),  INT16_C( 10431), -INT16_C( 19766)) },
    { easysimd_mm_set_epi16( INT16_C( 10104),  INT16_C( 19227),  INT16_C( 32488), -INT16_C(  6640), -INT16_C( 16754),  INT16_C( 31526),  INT16_C( 30096), -INT16_C( 18861)),
      UINT8_C( 20),
      easysimd_mm_set_epi16( INT16_C( 14103),  INT16_C(  6273), -INT16_C( 15168), -INT16_C(  1688),  INT16_C(  6460), -INT16_C( 30060),  INT16_C(   127),  INT16_C( 20280)),
      easysimd_mm_set_epi16( INT16_C( 10104),  INT16_C( 19227),  INT16_C( 32488), -INT16_C(  1688), -INT16_C( 16754), -INT16_C( 30060),  INT16_C( 30096), -INT16_C( 18861)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_loadu_epi16(test_vec[i].src, test_vec[i].k, &(test_vec[i].a));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_loadu_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i16x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_mask_loadu_epi16(src, k, &a);
    src = easysimd_mm_set_epi16(src.i16[0], src.i16[1], src.i16[2], src.i16[3], src.i16[4], src.i16[5], src.i16[6], src.i16[7]);
    a = easysimd_mm_set_epi16(a.i16[0], a.i16[1], a.i16[2], a.i16[3], a.i16[4], a.i16[5], a.i16[6], a.i16[7]);
    r = easysimd_mm_set_epi16(r.i16[0], r.i16[1], r.i16[2], r.i16[3], r.i16[4], r.i16[5], r.i16[6], r.i16[7]);

    easysimd_test_x86_write_i16x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_loadu_epi16 (EASYSIMD_MUNIT_TEST_ARGS)
{
#if 1
  static const struct {
    easysimd__mmask8 k;
    const int16_t a[8];
    const int16_t r[8];
  } test_vec[] = {
    { UINT8_C(147),
      {  INT16_C( 10837),  INT16_C(  7522), -INT16_C(   368),  INT16_C( 27507),  INT16_C(  5129),  INT16_C( 29762),  INT16_C( 16976), -INT16_C(  2018) },
      {  INT16_C( 10837),  INT16_C(  7522),  INT16_C(     0),  INT16_C(     0),  INT16_C(  5129),  INT16_C(     0),  INT16_C(     0), -INT16_C(  2018) } },
    { UINT8_C( 37),
      {  INT16_C( 24027),  INT16_C(  3308), -INT16_C(  6688), -INT16_C( 16842), -INT16_C( 15866), -INT16_C( 29808),  INT16_C(  9394), -INT16_C(  8992) },
      {  INT16_C( 24027),  INT16_C(     0), -INT16_C(  6688),  INT16_C(     0),  INT16_C(     0), -INT16_C( 29808),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(134),
      {  INT16_C( 28157),  INT16_C( 28804), -INT16_C( 29224),  INT16_C(  6789), -INT16_C( 11006),  INT16_C(  8284), -INT16_C( 32307),  INT16_C( 11004) },
      {  INT16_C(     0),  INT16_C( 28804), -INT16_C( 29224),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 11004) } },
    { UINT8_C(109),
      {  INT16_C(  2824),  INT16_C( 15955),  INT16_C( 22985),  INT16_C( 22784), -INT16_C( 19740), -INT16_C( 15235),  INT16_C(   911), -INT16_C(   831) },
      {  INT16_C(  2824),  INT16_C(     0),  INT16_C( 22985),  INT16_C( 22784),  INT16_C(     0), -INT16_C( 15235),  INT16_C(   911),  INT16_C(     0) } },
    { UINT8_C(136),
      { -INT16_C( 11215), -INT16_C( 18923),  INT16_C(  6126),  INT16_C( 19084),  INT16_C( 22840),  INT16_C( 13516),  INT16_C( 14724), -INT16_C( 28868) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 19084),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28868) } },
    { UINT8_C(140),
      {  INT16_C( 22650),  INT16_C( 31461), -INT16_C( 13903),  INT16_C( 12076), -INT16_C( 17523),  INT16_C( 20018), -INT16_C( 17737), -INT16_C( 29824) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 13903),  INT16_C( 12076),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 29824) } },
    { UINT8_C(208),
      {  INT16_C( 31030), -INT16_C( 15641),  INT16_C(  8132), -INT16_C( 28644), -INT16_C( 24493), -INT16_C( 28727),  INT16_C( 22063), -INT16_C( 30967) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 24493),  INT16_C(     0),  INT16_C( 22063), -INT16_C( 30967) } },
    { UINT8_C( 59),
      {  INT16_C( 14467), -INT16_C( 20475), -INT16_C( 28057), -INT16_C( 26005),  INT16_C(  9185),  INT16_C( 24916),  INT16_C(  9390),  INT16_C( 10391) },
      {  INT16_C( 14467), -INT16_C( 20475),  INT16_C(     0), -INT16_C( 26005),  INT16_C(  9185),  INT16_C( 24916),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi16(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_loadu_epi16(k, &a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_loadu_epi16");
    easysimd_test_x86_assert_equal_i16x8(r, easysimd_mm_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i r = easysimd_mm_maskz_loadu_epi16(k, &a.i8[0]);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_loadu_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__m128i src;
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[] = {
    { easysimd_mm_set_epi32( INT32_C(  1650606634),  INT32_C(  1310437891),  INT32_C(     3414810),  INT32_C(   610110632)),
      UINT8_C(146),
      easysimd_mm_set_epi32( INT32_C(  1002459311),  INT32_C(   404330751), -INT32_C(  1730920545),  INT32_C(  1814455010)),
      easysimd_mm_set_epi32( INT32_C(  1650606634),  INT32_C(  1310437891), -INT32_C(  1730920545),  INT32_C(   610110632)) },
    { easysimd_mm_set_epi32( INT32_C(  1581537148), -INT32_C(  1292559760), -INT32_C(   892644072), -INT32_C(   933894995)),
      UINT8_C(186),
      easysimd_mm_set_epi32( INT32_C(  1711874178),  INT32_C(  1541783819),  INT32_C(   917107102), -INT32_C(  1436956054)),
      easysimd_mm_set_epi32( INT32_C(  1711874178), -INT32_C(  1292559760),  INT32_C(   917107102), -INT32_C(   933894995)) },
    { easysimd_mm_set_epi32(-INT32_C(  1672441295), -INT32_C(  1774981826), -INT32_C(  1220916835), -INT32_C(   897592971)),
      UINT8_C( 92),
      easysimd_mm_set_epi32(-INT32_C(   609380086),  INT32_C(   926089495), -INT32_C(  1314573841), -INT32_C(  1694808407)),
      easysimd_mm_set_epi32(-INT32_C(   609380086),  INT32_C(   926089495), -INT32_C(  1220916835), -INT32_C(   897592971)) },
    { easysimd_mm_set_epi32( INT32_C(   793865350), -INT32_C(   170903965), -INT32_C(  1616891982),  INT32_C(  1652061976)),
      UINT8_C(178),
      easysimd_mm_set_epi32(-INT32_C(   247966087),  INT32_C(   528141446),  INT32_C(  1888834415),  INT32_C(  2124597836)),
      easysimd_mm_set_epi32( INT32_C(   793865350), -INT32_C(   170903965),  INT32_C(  1888834415),  INT32_C(  1652061976)) },
    { easysimd_mm_set_epi32(-INT32_C(   198105177),  INT32_C(  2146915858),  INT32_C(  1902199354), -INT32_C(    22832434)),
      UINT8_C(241),
      easysimd_mm_set_epi32(-INT32_C(  1636811260), -INT32_C(  1588736949), -INT32_C(   743649017),  INT32_C(   643855059)),
      easysimd_mm_set_epi32(-INT32_C(   198105177),  INT32_C(  2146915858),  INT32_C(  1902199354),  INT32_C(   643855059)) },
    { easysimd_mm_set_epi32(-INT32_C(  1185300250),  INT32_C(  1758245953),  INT32_C(   624249295),  INT32_C(   601386721)),
      UINT8_C( 50),
      easysimd_mm_set_epi32( INT32_C(  1434057447), -INT32_C(  1738770598),  INT32_C(  1490423180), -INT32_C(  1975918407)),
      easysimd_mm_set_epi32(-INT32_C(  1185300250),  INT32_C(  1758245953),  INT32_C(  1490423180),  INT32_C(   601386721)) },
    { easysimd_mm_set_epi32(-INT32_C(   670489314), -INT32_C(   697514730), -INT32_C(  2040706607),  INT32_C(   472161491)),
      UINT8_C(  5),
      easysimd_mm_set_epi32( INT32_C(  1444212154),  INT32_C(  1100431687), -INT32_C(  1331041736),  INT32_C(  1334993474)),
      easysimd_mm_set_epi32(-INT32_C(   670489314),  INT32_C(  1100431687), -INT32_C(  2040706607),  INT32_C(  1334993474)) },
    { easysimd_mm_set_epi32(-INT32_C(  1824175536),  INT32_C(  1064453914),  INT32_C(  1416736398),  INT32_C(  1529360657)),
      UINT8_C(173),
      easysimd_mm_set_epi32(-INT32_C(  1892950434),  INT32_C(   445139501), -INT32_C(   855075444), -INT32_C(  1041898793)),
      easysimd_mm_set_epi32(-INT32_C(  1892950434),  INT32_C(   445139501),  INT32_C(  1416736398), -INT32_C(  1041898793)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_loadu_epi32(test_vec[i].src, test_vec[i].k, &(test_vec[i].a));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_loadu_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_loadu_epi32(src, k, &a);
    src = easysimd_mm_set_epi32(src.i32[0], src.i32[1], src.i32[2], src.i32[3]);
    a = easysimd_mm_set_epi32(a.i32[0], a.i32[1], a.i32[2], a.i32[3]);
    r = easysimd_mm_set_epi32(r.i32[0], r.i32[1], r.i32[2], r.i32[3]);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_loadu_epi32 (EASYSIMD_MUNIT_TEST_ARGS)
{
#if 1
  static const struct {
    easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C( 12),
      {  INT32_C(  1982590042),  INT32_C(  1159102332),  INT32_C(   412828942),  INT32_C(    77322188) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   412828942),  INT32_C(    77322188) } },
    { UINT8_C(220),
      { -INT32_C(  1217500085),  INT32_C(  1524256518),  INT32_C(  1216317616), -INT32_C(  1667069008) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1216317616), -INT32_C(  1667069008) } },
    { UINT8_C(182),
      {  INT32_C(   775231512), -INT32_C(   109886370), -INT32_C(   137347236), -INT32_C(  1354519485) },
      {  INT32_C(           0), -INT32_C(   109886370), -INT32_C(   137347236),  INT32_C(           0) } },
    { UINT8_C( 27),
      { -INT32_C(   731204102), -INT32_C(  1889789168),  INT32_C(    85593443), -INT32_C(  1038233431) },
      { -INT32_C(   731204102), -INT32_C(  1889789168),  INT32_C(           0), -INT32_C(  1038233431) } },
    { UINT8_C(  6),
      { -INT32_C(  1102438325), -INT32_C(   352475623), -INT32_C(   527023971), -INT32_C(  1512394256) },
      {  INT32_C(           0), -INT32_C(   352475623), -INT32_C(   527023971),  INT32_C(           0) } },
    { UINT8_C( 28),
      {  INT32_C(   188200366),  INT32_C(  1578670660), -INT32_C(  1137720929), -INT32_C(  1559808637) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1137720929), -INT32_C(  1559808637) } },
    { UINT8_C(127),
      { -INT32_C(  1037713979),  INT32_C(  1023591079),  INT32_C(  2112877219),  INT32_C(  1294732184) },
      { -INT32_C(  1037713979),  INT32_C(  1023591079),  INT32_C(  2112877219),  INT32_C(  1294732184) } },
    { UINT8_C( 66),
      {  INT32_C(  1339855415),  INT32_C(   521174000),  INT32_C(  1045730359), -INT32_C(   201075913) },
      {  INT32_C(           0),  INT32_C(   521174000),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_loadu_epi32(k, &a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_loadu_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_loadu_epi32(k, &a.i8[0]);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_loadu_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__m128i src;
    easysimd__mmask8 k;
    easysimd__m128i a;
    easysimd__m128i r;
  } test_vec[] = {
    { easysimd_mm_set_epi64x( INT64_C( 3088381185144852196),  INT64_C( 4597738677796242950)),
      UINT8_C(  5),
      easysimd_mm_set_epi64x( INT64_C( 5821796188421922534),  INT64_C( 4504829930811550657)),
      easysimd_mm_set_epi64x( INT64_C( 3088381185144852196),  INT64_C( 4504829930811550657)) },
    { easysimd_mm_set_epi64x( INT64_C( 7871687053127175958),  INT64_C( 6674482320487448650)),
      UINT8_C(197),
      easysimd_mm_set_epi64x( INT64_C( 2596887743211493523), -INT64_C( 1867877531051768626)),
      easysimd_mm_set_epi64x( INT64_C( 7871687053127175958), -INT64_C( 1867877531051768626)) },
    { easysimd_mm_set_epi64x( INT64_C( 2890552681491930238),  INT64_C( 3883175344163964231)),
      UINT8_C( 18),
      easysimd_mm_set_epi64x(-INT64_C( 1209769374179319569),  INT64_C( 4710836029095564758)),
      easysimd_mm_set_epi64x(-INT64_C( 1209769374179319569),  INT64_C( 3883175344163964231)) },
    { easysimd_mm_set_epi64x( INT64_C( 1498436705963483499),  INT64_C(  272901978421789563)),
      UINT8_C(217),
      easysimd_mm_set_epi64x( INT64_C( 9073762015554258024),  INT64_C( 7013656311594410277)),
      easysimd_mm_set_epi64x( INT64_C( 1498436705963483499),  INT64_C( 7013656311594410277)) },
    { easysimd_mm_set_epi64x( INT64_C( 8008089850297155268),  INT64_C( 2433464424861022159)),
      UINT8_C( 84),
      easysimd_mm_set_epi64x(-INT64_C( 2032419013175754622),  INT64_C( 7948843630978258276)),
      easysimd_mm_set_epi64x( INT64_C( 8008089850297155268),  INT64_C( 2433464424861022159)) },
    { easysimd_mm_set_epi64x(-INT64_C( 4311596993451245981),  INT64_C( 7562956548849476810)),
      UINT8_C(246),
      easysimd_mm_set_epi64x( INT64_C( 2640879599057515894), -INT64_C( 5027830301517108288)),
      easysimd_mm_set_epi64x( INT64_C( 2640879599057515894),  INT64_C( 7562956548849476810)) },
    { easysimd_mm_set_epi64x( INT64_C( 6215019808294661160),  INT64_C(  834703929627301677)),
      UINT8_C(218),
      easysimd_mm_set_epi64x(-INT64_C( 4653167399934410252),  INT64_C( 8891669474635418976)),
      easysimd_mm_set_epi64x(-INT64_C( 4653167399934410252),  INT64_C(  834703929627301677)) },
    { easysimd_mm_set_epi64x( INT64_C( 3894361150073509122), -INT64_C( 6968028277536678226)),
      UINT8_C(154),
      easysimd_mm_set_epi64x(-INT64_C( 6186413115250073385), -INT64_C( 3719590067612670497)),
      easysimd_mm_set_epi64x(-INT64_C( 6186413115250073385), -INT64_C( 6968028277536678226)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_loadu_epi64(test_vec[i].src, test_vec[i].k, &(test_vec[i].a));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_loadu_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, test_vec[i].r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_mask_loadu_epi64(src, k, &a);
    src = easysimd_mm_set_epi64x(src.i64[0], src.i64[1]);
    a = easysimd_mm_set_epi64x(a.i64[0], a.i64[1]);
    r = easysimd_mm_set_epi64x(r.i64[0], r.i64[1]);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_loadu_epi64 (EASYSIMD_MUNIT_TEST_ARGS)
{
#if 1
  static const struct {
    easysimd__mmask8 k;
    const int64_t a[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C(248),
      { -INT64_C( 3982766631003907131),  INT64_C( 3548297329189802715) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(241),
      { -INT64_C( 7623851716958739827), -INT64_C(  456725727502833183) },
      { -INT64_C( 7623851716958739827),  INT64_C(                   0) } },
    { UINT8_C( 29),
      {  INT64_C( 7889275867383976560),  INT64_C( 8010341664407297118) },
      {  INT64_C( 7889275867383976560),  INT64_C(                   0) } },
    { UINT8_C(223),
      { -INT64_C( 5736632339543182904), -INT64_C( 8259267757387628217) },
      { -INT64_C( 5736632339543182904), -INT64_C( 8259267757387628217) } },
    { UINT8_C(170),
      {  INT64_C( 1739620449760913547), -INT64_C( 7859507159949938181) },
      {  INT64_C(                   0), -INT64_C( 7859507159949938181) } },
    { UINT8_C(237),
      { -INT64_C( 7202213988630668825),  INT64_C( 9040261938399659913) },
      { -INT64_C( 7202213988630668825),  INT64_C(                   0) } },
    { UINT8_C(119),
      { -INT64_C( 5326683953215171971),  INT64_C( 6902908301367964919) },
      { -INT64_C( 5326683953215171971),  INT64_C( 6902908301367964919) } },
    { UINT8_C( 95),
      {  INT64_C( 5830896547100420119), -INT64_C( 3504295574318464917) },
      {  INT64_C( 5830896547100420119), -INT64_C( 3504295574318464917) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_epi64(test_vec[i].a);
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_loadu_epi64(k, &a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_loadu_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_loadu_epi64(k, &a.i8[0]);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_loadu_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__m128 src;
    easysimd__mmask8 k;
    easysimd__m128 a;
    easysimd__m128 r;
  } test_vec[] = {
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   945.48), EASYSIMD_FLOAT32_C(  -245.65), EASYSIMD_FLOAT32_C(  -390.51), EASYSIMD_FLOAT32_C(   701.56)),
      UINT8_C( 55),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -625.52), EASYSIMD_FLOAT32_C(  -804.81), EASYSIMD_FLOAT32_C(  -737.48), EASYSIMD_FLOAT32_C(   348.28)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   945.48), EASYSIMD_FLOAT32_C(  -804.81), EASYSIMD_FLOAT32_C(  -737.48), EASYSIMD_FLOAT32_C(   348.28)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -458.45), EASYSIMD_FLOAT32_C(   654.46), EASYSIMD_FLOAT32_C(  -216.92), EASYSIMD_FLOAT32_C(  -691.92)),
      UINT8_C(214),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -84.24), EASYSIMD_FLOAT32_C(  -381.72), EASYSIMD_FLOAT32_C(  -416.15), EASYSIMD_FLOAT32_C(  -676.30)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -458.45), EASYSIMD_FLOAT32_C(  -381.72), EASYSIMD_FLOAT32_C(  -416.15), EASYSIMD_FLOAT32_C(  -691.92)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -44.84), EASYSIMD_FLOAT32_C(    -3.64), EASYSIMD_FLOAT32_C(  -464.37), EASYSIMD_FLOAT32_C(    68.35)),
      UINT8_C(184),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   654.62), EASYSIMD_FLOAT32_C(  -710.49), EASYSIMD_FLOAT32_C(  -820.63), EASYSIMD_FLOAT32_C(   213.87)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   654.62), EASYSIMD_FLOAT32_C(    -3.64), EASYSIMD_FLOAT32_C(  -464.37), EASYSIMD_FLOAT32_C(    68.35)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -521.06), EASYSIMD_FLOAT32_C(   750.40), EASYSIMD_FLOAT32_C(  -813.42), EASYSIMD_FLOAT32_C(  -590.54)),
      UINT8_C( 35),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -233.69), EASYSIMD_FLOAT32_C(   833.62), EASYSIMD_FLOAT32_C(   233.29), EASYSIMD_FLOAT32_C(  -640.10)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -521.06), EASYSIMD_FLOAT32_C(   750.40), EASYSIMD_FLOAT32_C(   233.29), EASYSIMD_FLOAT32_C(  -640.10)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -43.95), EASYSIMD_FLOAT32_C(   -38.50), EASYSIMD_FLOAT32_C(  -903.85), EASYSIMD_FLOAT32_C(  -418.42)),
      UINT8_C(172),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   890.33), EASYSIMD_FLOAT32_C(   -54.22), EASYSIMD_FLOAT32_C(  -389.49), EASYSIMD_FLOAT32_C(   744.57)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   890.33), EASYSIMD_FLOAT32_C(   -54.22), EASYSIMD_FLOAT32_C(  -903.85), EASYSIMD_FLOAT32_C(  -418.42)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   849.97), EASYSIMD_FLOAT32_C(  -491.38), EASYSIMD_FLOAT32_C(   529.63), EASYSIMD_FLOAT32_C(   -65.78)),
      UINT8_C(145),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -207.76), EASYSIMD_FLOAT32_C(   553.13), EASYSIMD_FLOAT32_C(  -153.66), EASYSIMD_FLOAT32_C(    44.24)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   849.97), EASYSIMD_FLOAT32_C(  -491.38), EASYSIMD_FLOAT32_C(   529.63), EASYSIMD_FLOAT32_C(    44.24)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   714.83), EASYSIMD_FLOAT32_C(    81.75), EASYSIMD_FLOAT32_C(   732.50), EASYSIMD_FLOAT32_C(  -939.79)),
      UINT8_C( 67),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   156.47), EASYSIMD_FLOAT32_C(  -379.10), EASYSIMD_FLOAT32_C(   465.23), EASYSIMD_FLOAT32_C(   268.33)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   714.83), EASYSIMD_FLOAT32_C(    81.75), EASYSIMD_FLOAT32_C(   465.23), EASYSIMD_FLOAT32_C(   268.33)) },
    { easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -408.57), EASYSIMD_FLOAT32_C(    -9.91), EASYSIMD_FLOAT32_C(   854.20), EASYSIMD_FLOAT32_C(   825.13)),
      UINT8_C(142),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   490.47), EASYSIMD_FLOAT32_C(   391.83), EASYSIMD_FLOAT32_C(   552.93), EASYSIMD_FLOAT32_C(    86.24)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   490.47), EASYSIMD_FLOAT32_C(   391.83), EASYSIMD_FLOAT32_C(   552.93), EASYSIMD_FLOAT32_C(   825.13)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_loadu_ps(test_vec[i].src, test_vec[i].k, &(test_vec[i].a));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_loadu_ps");
    easysimd_test_x86_assert_equal_f32x4(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_loadu_ps(src, k, &a);
    src = easysimd_mm_set_ps(src.f32[0], src.f32[1], src.f32[2], src.f32[3]);
    a = easysimd_mm_set_ps(a.f32[0], a.f32[1], a.f32[2], a.f32[3]);
    r = easysimd_mm_set_ps(r.f32[0], r.f32[1], r.f32[2], r.f32[3]);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_loadu_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    easysimd__m128 a;
    easysimd__m128 r;
  } test_vec[] = {
    { UINT8_C( 18),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   459.46), EASYSIMD_FLOAT32_C(  -212.38), EASYSIMD_FLOAT32_C(   362.35), EASYSIMD_FLOAT32_C(  -890.99)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   362.35), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT8_C( 82),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -361.11), EASYSIMD_FLOAT32_C(   817.90), EASYSIMD_FLOAT32_C(  -722.69), EASYSIMD_FLOAT32_C(  -175.74)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -722.69), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT8_C( 17),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -669.35), EASYSIMD_FLOAT32_C(   829.13), EASYSIMD_FLOAT32_C(  -557.48), EASYSIMD_FLOAT32_C(   930.82)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   930.82)) },
    { UINT8_C(175),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -53.75), EASYSIMD_FLOAT32_C(   778.14), EASYSIMD_FLOAT32_C(   886.18), EASYSIMD_FLOAT32_C(  -875.18)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -53.75), EASYSIMD_FLOAT32_C(   778.14), EASYSIMD_FLOAT32_C(   886.18), EASYSIMD_FLOAT32_C(  -875.18)) },
    { UINT8_C(228),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   972.16), EASYSIMD_FLOAT32_C(   838.44), EASYSIMD_FLOAT32_C(   588.89), EASYSIMD_FLOAT32_C(  -205.95)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   838.44), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00)) },
    { UINT8_C(179),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -69.68), EASYSIMD_FLOAT32_C(   219.39), EASYSIMD_FLOAT32_C(  -248.74), EASYSIMD_FLOAT32_C(  -599.67)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -248.74), EASYSIMD_FLOAT32_C(  -599.67)) },
    { UINT8_C(147),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    38.19), EASYSIMD_FLOAT32_C(  -879.76), EASYSIMD_FLOAT32_C(    39.33), EASYSIMD_FLOAT32_C(  -749.44)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    39.33), EASYSIMD_FLOAT32_C(  -749.44)) },
    { UINT8_C(110),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -598.31), EASYSIMD_FLOAT32_C(  -223.89), EASYSIMD_FLOAT32_C(   862.45), EASYSIMD_FLOAT32_C(  -416.21)),
      easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -598.31), EASYSIMD_FLOAT32_C(  -223.89), EASYSIMD_FLOAT32_C(   862.45), EASYSIMD_FLOAT32_C(     0.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_loadu_ps(test_vec[i].k, &(test_vec[i].a));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_loadu_ps");
    easysimd_test_x86_assert_equal_f32x4(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_loadu_ps( k, &a);
    a = easysimd_mm_set_ps(a.f32[0], a.f32[1], a.f32[2], a.f32[3]);
    r = easysimd_mm_set_ps(r.f32[0], r.f32[1], r.f32[2], r.f32[3]);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_loadu_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__m128d src;
    easysimd__mmask8 k;
    easysimd__m128d a;
    easysimd__m128d r;
  } test_vec[] = {
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -882.90), EASYSIMD_FLOAT64_C(   176.45)),
      UINT8_C(228),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   282.41), EASYSIMD_FLOAT64_C(  -866.21)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -882.90), EASYSIMD_FLOAT64_C(   176.45)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -358.32), EASYSIMD_FLOAT64_C(   404.17)),
      UINT8_C(150),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -729.54), EASYSIMD_FLOAT64_C(   817.65)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -729.54), EASYSIMD_FLOAT64_C(   404.17)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -282.33), EASYSIMD_FLOAT64_C(   789.77)),
      UINT8_C(142),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   189.29), EASYSIMD_FLOAT64_C(  -388.75)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   189.29), EASYSIMD_FLOAT64_C(   789.77)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -597.36), EASYSIMD_FLOAT64_C(   270.34)),
      UINT8_C( 45),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -26.80), EASYSIMD_FLOAT64_C(   957.25)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -597.36), EASYSIMD_FLOAT64_C(   957.25)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   444.91), EASYSIMD_FLOAT64_C(  -450.18)),
      UINT8_C(150),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -714.10), EASYSIMD_FLOAT64_C(  -899.37)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -714.10), EASYSIMD_FLOAT64_C(  -450.18)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   817.41), EASYSIMD_FLOAT64_C(     8.54)),
      UINT8_C(187),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -467.20), EASYSIMD_FLOAT64_C(  -637.38)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -467.20), EASYSIMD_FLOAT64_C(  -637.38)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   539.06), EASYSIMD_FLOAT64_C(  -447.38)),
      UINT8_C( 11),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   672.85), EASYSIMD_FLOAT64_C(   762.33)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   672.85), EASYSIMD_FLOAT64_C(   762.33)) },
    { easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   166.50), EASYSIMD_FLOAT64_C(   932.30)),
      UINT8_C(206),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -15.85), EASYSIMD_FLOAT64_C(    23.84)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -15.85), EASYSIMD_FLOAT64_C(   932.30)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_loadu_pd(test_vec[i].src, test_vec[i].k, &(test_vec[i].a));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_loadu_pd");
    easysimd_test_x86_assert_equal_f64x2(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_loadu_pd(src, k, &a);
    src = easysimd_mm_set_pd(src.f64[0], src.f64[1]);
    a = easysimd_mm_set_pd(a.f64[0], a.f64[1]);
    r = easysimd_mm_set_pd(r.f64[0], r.f64[1]);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_loadu_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    easysimd__m128d a;
    easysimd__m128d r;
  } test_vec[] = {
    { UINT8_C( 62),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   268.42), EASYSIMD_FLOAT64_C(   129.13)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   268.42), EASYSIMD_FLOAT64_C(     0.00)) },
    { UINT8_C(254),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -261.74), EASYSIMD_FLOAT64_C(  -222.70)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -261.74), EASYSIMD_FLOAT64_C(     0.00)) },
    { UINT8_C( 54),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(  -667.49)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(     0.00)) },
    { UINT8_C(194),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   220.24), EASYSIMD_FLOAT64_C(    48.39)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   220.24), EASYSIMD_FLOAT64_C(     0.00)) },
    { UINT8_C(187),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   609.99), EASYSIMD_FLOAT64_C(  -718.73)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   609.99), EASYSIMD_FLOAT64_C(  -718.73)) },
    { UINT8_C(186),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -693.14), EASYSIMD_FLOAT64_C(  -361.02)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -693.14), EASYSIMD_FLOAT64_C(     0.00)) },
    { UINT8_C(172),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -102.78), EASYSIMD_FLOAT64_C(  -516.94)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00)) },
    { UINT8_C( 59),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   705.46), EASYSIMD_FLOAT64_C(   701.00)),
      easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   705.46), EASYSIMD_FLOAT64_C(   701.00)) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_loadu_pd(test_vec[i].k, &(test_vec[i].a));
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_loadu_pd");
    easysimd_test_x86_assert_equal_f64x2(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_loadu_pd(k, &a);
    a = easysimd_mm_set_pd(a.f64[0], a.f64[1]);
    r = easysimd_mm_set_pd(r.f64[0], r.f64[1]);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_loadu_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    EASYSIMD_ALIGN_LIKE_16(easysimd__m128i) const int8_t a[16];
    const int8_t r[16];
  } test_vec[] = {
    { {  INT8_C(  98),  INT8_C( 124), -INT8_C(  57), -INT8_C(  74),  INT8_C( 104),  INT8_C(  59),  INT8_C(  69), -INT8_C(  25),
         INT8_C(  20), -INT8_C(  48), -INT8_C(  57), -INT8_C(  59),  INT8_C(  21),  INT8_C( 119), -INT8_C(  97),  INT8_C(  28) },
      {  INT8_C(  98),  INT8_C( 124), -INT8_C(  57), -INT8_C(  74),  INT8_C( 104),  INT8_C(  59),  INT8_C(  69), -INT8_C(  25),
         INT8_C(  20), -INT8_C(  48), -INT8_C(  57), -INT8_C(  59),  INT8_C(  21),  INT8_C( 119), -INT8_C(  97),  INT8_C(  28) } },
    { {  INT8_C(  70), -INT8_C(  33),  INT8_C(  29),  INT8_C(  10),  INT8_C(  39),  INT8_C(  31), -INT8_C(  91),  INT8_C(  90),
         INT8_C( 111), -INT8_C(  46), -INT8_C(  65),  INT8_C( 124), -INT8_C(  81),  INT8_C(  57), -INT8_C(  34),  INT8_C(  17) },
      {  INT8_C(  70), -INT8_C(  33),  INT8_C(  29),  INT8_C(  10),  INT8_C(  39),  INT8_C(  31), -INT8_C(  91),  INT8_C(  90),
         INT8_C( 111), -INT8_C(  46), -INT8_C(  65),  INT8_C( 124), -INT8_C(  81),  INT8_C(  57), -INT8_C(  34),  INT8_C(  17) } },
    { { -INT8_C(  75), -INT8_C(  91), -INT8_C(  56),  INT8_C(  30), -INT8_C(  31),  INT8_C(  13),  INT8_C(   5), -INT8_C(  11),
        -INT8_C(  35), -INT8_C(  52), -INT8_C(  70), -INT8_C(  14),  INT8_C(  67),  INT8_C(  89),  INT8_C(  15), -INT8_C( 118) },
      { -INT8_C(  75), -INT8_C(  91), -INT8_C(  56),  INT8_C(  30), -INT8_C(  31),  INT8_C(  13),  INT8_C(   5), -INT8_C(  11),
        -INT8_C(  35), -INT8_C(  52), -INT8_C(  70), -INT8_C(  14),  INT8_C(  67),  INT8_C(  89),  INT8_C(  15), -INT8_C( 118) } },
    { {  INT8_C(  56),  INT8_C(  44), -INT8_C( 108),  INT8_C(  96),  INT8_C(  75),  INT8_C(  57), -INT8_C(  70), -INT8_C(  70),
         INT8_C(  12),  INT8_C( 121),  INT8_C(  55), -INT8_C(  69), -INT8_C(  78),  INT8_C(  21), -INT8_C(  51),  INT8_C( 104) },
      {  INT8_C(  56),  INT8_C(  44), -INT8_C( 108),  INT8_C(  96),  INT8_C(  75),  INT8_C(  57), -INT8_C(  70), -INT8_C(  70),
         INT8_C(  12),  INT8_C( 121),  INT8_C(  55), -INT8_C(  69), -INT8_C(  78),  INT8_C(  21), -INT8_C(  51),  INT8_C( 104) } },
    { { -INT8_C(  69), -INT8_C( 107), -INT8_C( 122), -INT8_C( 100), -INT8_C(  94), -INT8_C( 117), -INT8_C( 111),      INT8_MAX,
         INT8_C(  87),  INT8_C(  75),  INT8_C( 114), -INT8_C( 102), -INT8_C(  91), -INT8_C( 127),  INT8_C(  36), -INT8_C(  35) },
      { -INT8_C(  69), -INT8_C( 107), -INT8_C( 122), -INT8_C( 100), -INT8_C(  94), -INT8_C( 117), -INT8_C( 111),      INT8_MAX,
         INT8_C(  87),  INT8_C(  75),  INT8_C( 114), -INT8_C( 102), -INT8_C(  91), -INT8_C( 127),  INT8_C(  36), -INT8_C(  35) } },
    { { -INT8_C(  83), -INT8_C(  72),  INT8_C(  61), -INT8_C(   8), -INT8_C(  14), -INT8_C(   8), -INT8_C(  78), -INT8_C(   2),
         INT8_C( 113), -INT8_C(  23), -INT8_C(  71),  INT8_C(  36), -INT8_C(   1), -INT8_C( 122), -INT8_C( 116), -INT8_C(  70) },
      { -INT8_C(  83), -INT8_C(  72),  INT8_C(  61), -INT8_C(   8), -INT8_C(  14), -INT8_C(   8), -INT8_C(  78), -INT8_C(   2),
         INT8_C( 113), -INT8_C(  23), -INT8_C(  71),  INT8_C(  36), -INT8_C(   1), -INT8_C( 122), -INT8_C( 116), -INT8_C(  70) } },
    { {  INT8_C(  27),  INT8_C(  18),  INT8_C(  86), -INT8_C(  67), -INT8_C(  99), -INT8_C(  25),  INT8_C(  61), -INT8_C(  12),
         INT8_C(  50), -INT8_C(  81), -INT8_C( 114), -INT8_C(  41),  INT8_C(  48), -INT8_C(  77), -INT8_C(  75), -INT8_C(  35) },
      {  INT8_C(  27),  INT8_C(  18),  INT8_C(  86), -INT8_C(  67), -INT8_C(  99), -INT8_C(  25),  INT8_C(  61), -INT8_C(  12),
         INT8_C(  50), -INT8_C(  81), -INT8_C( 114), -INT8_C(  41),  INT8_C(  48), -INT8_C(  77), -INT8_C(  75), -INT8_C(  35) } },
    { {  INT8_C( 107), -INT8_C(  14), -INT8_C(  43),  INT8_C(  93), -INT8_C(  22), -INT8_C( 121),  INT8_C(  91),  INT8_C(  92),
         INT8_C( 113),  INT8_C(  21),      INT8_MIN,  INT8_C( 112), -INT8_C( 101),  INT8_C(  12),  INT8_C(  42), -INT8_C(  73) },
      {  INT8_C( 107), -INT8_C(  14), -INT8_C(  43),  INT8_C(  93), -INT8_C(  22), -INT8_C( 121),  INT8_C(  91),  INT8_C(  92),
         INT8_C( 113),  INT8_C(  21),      INT8_MIN,  INT8_C( 112), -INT8_C( 101),  INT8_C(  12),  INT8_C(  42), -INT8_C(  73) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    void const * pr = (void const *)&(test_vec[i].r);
    easysimd__m128i vec_r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      vec_r = easysimd_mm_loadu_epi8(pr);
    }
    EASYSIMD_TEST_PERF_END("_mm_loadu_epi8 on easysimd");
    easysimd_test_x86_assert_equal_i8x16(easysimd_mm_load_si128(EASYSIMD_ALIGN_CAST(easysimd__m128i const *, test_vec[i].a)), vec_r);
  }

  return 0;
}

static int
test_easysimd_mm_loadu_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
  static const struct {
    EASYSIMD_ALIGN_LIKE_16(easysimd__m128i) const int16_t a[8];
    const int16_t r[8];
  } test_vec[] = {
    { { -INT16_C( 32738), -INT16_C( 17548), -INT16_C( 20121), -INT16_C( 26193),  INT16_C( 15712), -INT16_C( 28559),  INT16_C(  9968),  INT16_C( 23661) },
      { -INT16_C( 32738), -INT16_C( 17548), -INT16_C( 20121), -INT16_C( 26193),  INT16_C( 15712), -INT16_C( 28559),  INT16_C(  9968),  INT16_C( 23661) } },
    { {  INT16_C( 16920),  INT16_C(   953),  INT16_C(  5578),  INT16_C( 15199), -INT16_C(  8406), -INT16_C( 14933), -INT16_C( 10773),  INT16_C(  2428) },
      {  INT16_C( 16920),  INT16_C(   953),  INT16_C(  5578),  INT16_C( 15199), -INT16_C(  8406), -INT16_C( 14933), -INT16_C( 10773),  INT16_C(  2428) } },
    { { -INT16_C(  3755), -INT16_C( 17212),  INT16_C( 29602),  INT16_C(   853), -INT16_C( 14672), -INT16_C( 24173),  INT16_C(   492),  INT16_C(  1533) },
      { -INT16_C(  3755), -INT16_C( 17212),  INT16_C( 29602),  INT16_C(   853), -INT16_C( 14672), -INT16_C( 24173),  INT16_C(   492),  INT16_C(  1533) } },
    { { -INT16_C( 18877),  INT16_C(  3336),  INT16_C( 26571), -INT16_C(  2744), -INT16_C(  3258),  INT16_C( 12731),  INT16_C( 14280),  INT16_C(  7482) },
      { -INT16_C( 18877),  INT16_C(  3336),  INT16_C( 26571), -INT16_C(  2744), -INT16_C(  3258),  INT16_C( 12731),  INT16_C( 14280),  INT16_C(  7482) } },
    { { -INT16_C(   472), -INT16_C( 13351),  INT16_C( 12145),  INT16_C(  8654),  INT16_C( 25077), -INT16_C(  7486), -INT16_C( 16542), -INT16_C( 22809) },
      { -INT16_C(   472), -INT16_C( 13351),  INT16_C( 12145),  INT16_C(  8654),  INT16_C( 25077), -INT16_C(  7486), -INT16_C( 16542), -INT16_C( 22809) } },
    { { -INT16_C(  4234),  INT16_C( 16819), -INT16_C(   938), -INT16_C( 25545), -INT16_C(  3345), -INT16_C( 18227),  INT16_C(  1833),  INT16_C( 21205) },
      { -INT16_C(  4234),  INT16_C( 16819), -INT16_C(   938), -INT16_C( 25545), -INT16_C(  3345), -INT16_C( 18227),  INT16_C(  1833),  INT16_C( 21205) } },
    { { -INT16_C( 20731),  INT16_C( 30237), -INT16_C(  5154), -INT16_C( 11369),  INT16_C( 23116), -INT16_C( 20555), -INT16_C( 25575), -INT16_C( 28843) },
      { -INT16_C( 20731),  INT16_C( 30237), -INT16_C(  5154), -INT16_C( 11369),  INT16_C( 23116), -INT16_C( 20555), -INT16_C( 25575), -INT16_C( 28843) } },
    { {  INT16_C(  2187), -INT16_C(  7727),  INT16_C(  2052), -INT16_C(  2947),  INT16_C( 19194),  INT16_C(  9132), -INT16_C( 32431),  INT16_C( 22133) },
      {  INT16_C(  2187), -INT16_C(  7727),  INT16_C(  2052), -INT16_C(  2947),  INT16_C( 19194),  INT16_C(  9132), -INT16_C( 32431),  INT16_C( 22133) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    void const * pr = (void const *)&(test_vec[i].r);
    easysimd__m128i vec_r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      vec_r = easysimd_mm_loadu_epi16(pr);
    }
    EASYSIMD_TEST_PERF_END("_mm_loadu_epi16 on easysimd");
    easysimd_test_x86_assert_equal_i16x8(easysimd_mm_load_si128(EASYSIMD_ALIGN_CAST(easysimd__m128i const *, test_vec[i].a)), vec_r);
  }

  return 0;
}

static int
test_easysimd_mm_loadu_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_16(easysimd__m128i) const int32_t a[4];
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(   248287792), -INT32_C(   891132803), -INT32_C(   679897154), -INT32_C(  1083716044) },
      {  INT32_C(   248287792), -INT32_C(   891132803), -INT32_C(   679897154), -INT32_C(  1083716044) } },
    { { -INT32_C(   610191146),  INT32_C(   986652224), -INT32_C(  1168278679),  INT32_C(   756143100) },
      { -INT32_C(   610191146),  INT32_C(   986652224), -INT32_C(  1168278679),  INT32_C(   756143100) } },
    { { -INT32_C(   482615963), -INT32_C(     5431999), -INT32_C(   371775819), -INT32_C(   894943500) },
      { -INT32_C(   482615963), -INT32_C(     5431999), -INT32_C(   371775819), -INT32_C(   894943500) } },
    { { -INT32_C(  1230681738), -INT32_C(   772770712), -INT32_C(   326414865), -INT32_C(  2045141984) },
      { -INT32_C(  1230681738), -INT32_C(   772770712), -INT32_C(   326414865), -INT32_C(  2045141984) } },
    { { -INT32_C(  1150724998),  INT32_C(   666572402),  INT32_C(   806392380),  INT32_C(  1190836432) },
      { -INT32_C(  1150724998),  INT32_C(   666572402),  INT32_C(   806392380),  INT32_C(  1190836432) } },
    { {  INT32_C(  1794940930),  INT32_C(    71035924),  INT32_C(  1525728825), -INT32_C(   572520093) },
      {  INT32_C(  1794940930),  INT32_C(    71035924),  INT32_C(  1525728825), -INT32_C(   572520093) } },
    { { -INT32_C(   812103331), -INT32_C(  1678355617), -INT32_C(  1244985627), -INT32_C(  1040464449) },
      { -INT32_C(   812103331), -INT32_C(  1678355617), -INT32_C(  1244985627), -INT32_C(  1040464449) } },
    { {  INT32_C(  2049701733),  INT32_C(   494823139), -INT32_C(  1887998420), -INT32_C(   731097225) },
      {  INT32_C(  2049701733),  INT32_C(   494823139), -INT32_C(  1887998420), -INT32_C(   731097225) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    void const * pr = (void const *)&(test_vec[i].r);
    easysimd__m128i vec_r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      vec_r = easysimd_mm_loadu_epi32(pr);
    }
    EASYSIMD_TEST_PERF_END("_mm_loadu_epi32 on easysimd");
    easysimd_test_x86_assert_equal_i32x4(easysimd_mm_load_si128(EASYSIMD_ALIGN_CAST(easysimd__m128i const *, test_vec[i].a)), vec_r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = a;

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_loadu_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_16(easysimd__m128i) const int64_t a[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 4511087683801712032), -INT64_C( 8060898892722248287) },
      {  INT64_C( 4511087683801712032), -INT64_C( 8060898892722248287) } },
    { { -INT64_C( 2384787176194159386),  INT64_C( 6143431839469952758) },
      { -INT64_C( 2384787176194159386),  INT64_C( 6143431839469952758) } },
    { {  INT64_C( 2411376884971791839),  INT64_C( 4543466100033153363) },
      {  INT64_C( 2411376884971791839),  INT64_C( 4543466100033153363) } },
    { { -INT64_C( 7674432146617329682), -INT64_C( 6460338043923272626) },
      { -INT64_C( 7674432146617329682), -INT64_C( 6460338043923272626) } },
    { { -INT64_C( 1312143318173438935), -INT64_C( 2642072646704280642) },
      { -INT64_C( 1312143318173438935), -INT64_C( 2642072646704280642) } },
    { { -INT64_C( 7150315094646497649),  INT64_C( 3770910417545578470) },
      { -INT64_C( 7150315094646497649),  INT64_C( 3770910417545578470) } },
    { {  INT64_C( 4983981236450898595), -INT64_C( 7152365960020912652) },
      {  INT64_C( 4983981236450898595), -INT64_C( 7152365960020912652) } },
    { {  INT64_C( 1871967141139003407),  INT64_C( 3861302942246541911) },
      {  INT64_C( 1871967141139003407),  INT64_C( 3861302942246541911) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    void const * pr = (void const *)&(test_vec[i].r);
    easysimd__m128i vec_r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      vec_r = easysimd_mm_loadu_epi64(pr);
    }
    EASYSIMD_TEST_PERF_END("_mm_loadu_epi64 on easysimd");
    easysimd_test_x86_assert_equal_i64x2(easysimd_mm_load_si128(EASYSIMD_ALIGN_CAST(easysimd__m128i const *, test_vec[i].a)), vec_r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = a;

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_loadu_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_32(easysimd__m256i) const int8_t a[32];
    const int8_t r[32];
  } test_vec[] = {
    { {  INT8_C(  29), -INT8_C(  94),  INT8_C(  76),  INT8_C(  20),  INT8_C(  54), -INT8_C(  63),      INT8_MAX,  INT8_C(  69),
        -INT8_C(   2),  INT8_C( 122),  INT8_C(  91),  INT8_C(  14),  INT8_C( 107),  INT8_C( 122), -INT8_C(  75),  INT8_C( 113),
         INT8_C(  74), -INT8_C(  31), -INT8_C(  17), -INT8_C(  25),  INT8_C(  39),  INT8_C(  64), -INT8_C(  71),  INT8_C( 123),
             INT8_MAX, -INT8_C(  97), -INT8_C( 125), -INT8_C(  43), -INT8_C(  70), -INT8_C( 126),  INT8_C( 107), -INT8_C(  41) },
      {  INT8_C(  29), -INT8_C(  94),  INT8_C(  76),  INT8_C(  20),  INT8_C(  54), -INT8_C(  63),      INT8_MAX,  INT8_C(  69),
        -INT8_C(   2),  INT8_C( 122),  INT8_C(  91),  INT8_C(  14),  INT8_C( 107),  INT8_C( 122), -INT8_C(  75),  INT8_C( 113),
         INT8_C(  74), -INT8_C(  31), -INT8_C(  17), -INT8_C(  25),  INT8_C(  39),  INT8_C(  64), -INT8_C(  71),  INT8_C( 123),
             INT8_MAX, -INT8_C(  97), -INT8_C( 125), -INT8_C(  43), -INT8_C(  70), -INT8_C( 126),  INT8_C( 107), -INT8_C(  41) } },
    { {  INT8_C(  37), -INT8_C(  73), -INT8_C(  21),  INT8_C(  91),  INT8_C( 121),  INT8_C( 107), -INT8_C(  96),  INT8_C( 119),
        -INT8_C(  27), -INT8_C(   5), -INT8_C( 123),  INT8_C(  80),  INT8_C( 118),  INT8_C(  59), -INT8_C(  63), -INT8_C(  64),
         INT8_C(  28), -INT8_C(  80), -INT8_C(  89),  INT8_C(  67), -INT8_C(  16),  INT8_C(  96), -INT8_C(  65),  INT8_C( 112),
        -INT8_C(   1),  INT8_C(  66),  INT8_C(  69), -INT8_C(  71), -INT8_C(  59), -INT8_C(  80), -INT8_C( 111), -INT8_C(  22) },
      {  INT8_C(  37), -INT8_C(  73), -INT8_C(  21),  INT8_C(  91),  INT8_C( 121),  INT8_C( 107), -INT8_C(  96),  INT8_C( 119),
        -INT8_C(  27), -INT8_C(   5), -INT8_C( 123),  INT8_C(  80),  INT8_C( 118),  INT8_C(  59), -INT8_C(  63), -INT8_C(  64),
         INT8_C(  28), -INT8_C(  80), -INT8_C(  89),  INT8_C(  67), -INT8_C(  16),  INT8_C(  96), -INT8_C(  65),  INT8_C( 112),
        -INT8_C(   1),  INT8_C(  66),  INT8_C(  69), -INT8_C(  71), -INT8_C(  59), -INT8_C(  80), -INT8_C( 111), -INT8_C(  22) } },
    { {  INT8_C( 103),  INT8_C( 124),  INT8_C(  69), -INT8_C(  32), -INT8_C(  25), -INT8_C(  27),  INT8_C(  87), -INT8_C(  52),
        -INT8_C(  31), -INT8_C(  35),  INT8_C(  28),  INT8_C(  87),  INT8_C(  24), -INT8_C(  34),  INT8_C(  23),  INT8_C(  52),
        -INT8_C( 114), -INT8_C(  66),  INT8_C( 120),      INT8_MAX,  INT8_C(  30),  INT8_C(  55), -INT8_C(  17),  INT8_C(  29),
         INT8_C( 121),  INT8_C(  52), -INT8_C(  42),  INT8_C(  62), -INT8_C(  28),  INT8_C( 103),  INT8_C(  40),  INT8_C(  75) },
      {  INT8_C( 103),  INT8_C( 124),  INT8_C(  69), -INT8_C(  32), -INT8_C(  25), -INT8_C(  27),  INT8_C(  87), -INT8_C(  52),
        -INT8_C(  31), -INT8_C(  35),  INT8_C(  28),  INT8_C(  87),  INT8_C(  24), -INT8_C(  34),  INT8_C(  23),  INT8_C(  52),
        -INT8_C( 114), -INT8_C(  66),  INT8_C( 120),      INT8_MAX,  INT8_C(  30),  INT8_C(  55), -INT8_C(  17),  INT8_C(  29),
         INT8_C( 121),  INT8_C(  52), -INT8_C(  42),  INT8_C(  62), -INT8_C(  28),  INT8_C( 103),  INT8_C(  40),  INT8_C(  75) } },
    { { -INT8_C(  28),  INT8_C( 109),  INT8_C(  44), -INT8_C(  53),  INT8_C(  83), -INT8_C( 125), -INT8_C( 104),  INT8_C(  52),
         INT8_C(  96), -INT8_C(  76), -INT8_C( 117),  INT8_C( 120), -INT8_C( 110), -INT8_C(  94), -INT8_C(  83),  INT8_C(  33),
         INT8_C(  96),  INT8_C(  37), -INT8_C(  96),  INT8_C( 126),  INT8_C(  92), -INT8_C( 113), -INT8_C( 101), -INT8_C(  43),
        -INT8_C(  61),  INT8_C( 113),  INT8_C(  20), -INT8_C(  89), -INT8_C(  39),  INT8_C(  60), -INT8_C(  14), -INT8_C(  67) },
      { -INT8_C(  28),  INT8_C( 109),  INT8_C(  44), -INT8_C(  53),  INT8_C(  83), -INT8_C( 125), -INT8_C( 104),  INT8_C(  52),
         INT8_C(  96), -INT8_C(  76), -INT8_C( 117),  INT8_C( 120), -INT8_C( 110), -INT8_C(  94), -INT8_C(  83),  INT8_C(  33),
         INT8_C(  96),  INT8_C(  37), -INT8_C(  96),  INT8_C( 126),  INT8_C(  92), -INT8_C( 113), -INT8_C( 101), -INT8_C(  43),
        -INT8_C(  61),  INT8_C( 113),  INT8_C(  20), -INT8_C(  89), -INT8_C(  39),  INT8_C(  60), -INT8_C(  14), -INT8_C(  67) } },
    { { -INT8_C(  86),  INT8_C(  30), -INT8_C( 120), -INT8_C(   3), -INT8_C(  94),  INT8_C(  32),  INT8_C(  49),  INT8_C(   2),
        -INT8_C(  43), -INT8_C(  68),  INT8_C( 123),  INT8_C( 103),  INT8_C(  94),  INT8_C(  40), -INT8_C( 120), -INT8_C(  66),
         INT8_C(  77),  INT8_C(  40),  INT8_C(  60), -INT8_C(  87), -INT8_C(  73), -INT8_C(  41),  INT8_C( 126),  INT8_C( 122),
         INT8_C(  72), -INT8_C( 110),  INT8_C(  33),  INT8_C(  33), -INT8_C(  49),  INT8_C(  20), -INT8_C(  34),  INT8_C( 121) },
      { -INT8_C(  86),  INT8_C(  30), -INT8_C( 120), -INT8_C(   3), -INT8_C(  94),  INT8_C(  32),  INT8_C(  49),  INT8_C(   2),
        -INT8_C(  43), -INT8_C(  68),  INT8_C( 123),  INT8_C( 103),  INT8_C(  94),  INT8_C(  40), -INT8_C( 120), -INT8_C(  66),
         INT8_C(  77),  INT8_C(  40),  INT8_C(  60), -INT8_C(  87), -INT8_C(  73), -INT8_C(  41),  INT8_C( 126),  INT8_C( 122),
         INT8_C(  72), -INT8_C( 110),  INT8_C(  33),  INT8_C(  33), -INT8_C(  49),  INT8_C(  20), -INT8_C(  34),  INT8_C( 121) } },
    { {  INT8_C(  50),  INT8_C( 103),  INT8_C( 118), -INT8_C(  44), -INT8_C( 121), -INT8_C(  89), -INT8_C(  41),  INT8_C(  92),
         INT8_C(  99),  INT8_C(  82), -INT8_C(  60), -INT8_C(  63),  INT8_C( 122),  INT8_C(  76),      INT8_MAX, -INT8_C(  57),
         INT8_C( 117), -INT8_C(  69),  INT8_C( 112),  INT8_C(  44), -INT8_C( 110), -INT8_C(  18), -INT8_C(  89), -INT8_C(  38),
        -INT8_C( 127), -INT8_C(  56), -INT8_C(   4),  INT8_C(  80), -INT8_C(  36), -INT8_C(  38), -INT8_C(  55),  INT8_C(  15) },
      {  INT8_C(  50),  INT8_C( 103),  INT8_C( 118), -INT8_C(  44), -INT8_C( 121), -INT8_C(  89), -INT8_C(  41),  INT8_C(  92),
         INT8_C(  99),  INT8_C(  82), -INT8_C(  60), -INT8_C(  63),  INT8_C( 122),  INT8_C(  76),      INT8_MAX, -INT8_C(  57),
         INT8_C( 117), -INT8_C(  69),  INT8_C( 112),  INT8_C(  44), -INT8_C( 110), -INT8_C(  18), -INT8_C(  89), -INT8_C(  38),
        -INT8_C( 127), -INT8_C(  56), -INT8_C(   4),  INT8_C(  80), -INT8_C(  36), -INT8_C(  38), -INT8_C(  55),  INT8_C(  15) } },
    { {  INT8_C(  65),  INT8_C(  63), -INT8_C(  29), -INT8_C(  55), -INT8_C(  26), -INT8_C(  70),  INT8_C(  37),  INT8_C(  73),
         INT8_C(  12), -INT8_C(  23),  INT8_C(  10), -INT8_C( 122),  INT8_C(  54), -INT8_C( 119),  INT8_C(  77), -INT8_C(  85),
         INT8_C(  68), -INT8_C(  67), -INT8_C(  41), -INT8_C(  42), -INT8_C(  84),  INT8_C( 126), -INT8_C(  80),  INT8_C(  45),
         INT8_C(  71), -INT8_C(  84),  INT8_C( 125),  INT8_C(  35), -INT8_C( 121),  INT8_C(  70),  INT8_C(  50), -INT8_C(  56) },
      {  INT8_C(  65),  INT8_C(  63), -INT8_C(  29), -INT8_C(  55), -INT8_C(  26), -INT8_C(  70),  INT8_C(  37),  INT8_C(  73),
         INT8_C(  12), -INT8_C(  23),  INT8_C(  10), -INT8_C( 122),  INT8_C(  54), -INT8_C( 119),  INT8_C(  77), -INT8_C(  85),
         INT8_C(  68), -INT8_C(  67), -INT8_C(  41), -INT8_C(  42), -INT8_C(  84),  INT8_C( 126), -INT8_C(  80),  INT8_C(  45),
         INT8_C(  71), -INT8_C(  84),  INT8_C( 125),  INT8_C(  35), -INT8_C( 121),  INT8_C(  70),  INT8_C(  50), -INT8_C(  56) } },
    { { -INT8_C( 123),  INT8_C(  22), -INT8_C( 111),  INT8_C( 107), -INT8_C(  48), -INT8_C(  73), -INT8_C(  76), -INT8_C(  35),
        -INT8_C(  96), -INT8_C(  66),  INT8_C(  99), -INT8_C(  42),  INT8_C(  71), -INT8_C(  79), -INT8_C( 127), -INT8_C( 117),
         INT8_C( 110),  INT8_C(  89),  INT8_C(  97),  INT8_C(  26), -INT8_C(  41),  INT8_C(  17),  INT8_C(  71),  INT8_C(  30),
        -INT8_C(  66), -INT8_C(  60),  INT8_C(  66),  INT8_C(  69),  INT8_C(  10),  INT8_C( 116),  INT8_C(  13), -INT8_C( 113) },
      { -INT8_C( 123),  INT8_C(  22), -INT8_C( 111),  INT8_C( 107), -INT8_C(  48), -INT8_C(  73), -INT8_C(  76), -INT8_C(  35),
        -INT8_C(  96), -INT8_C(  66),  INT8_C(  99), -INT8_C(  42),  INT8_C(  71), -INT8_C(  79), -INT8_C( 127), -INT8_C( 117),
         INT8_C( 110),  INT8_C(  89),  INT8_C(  97),  INT8_C(  26), -INT8_C(  41),  INT8_C(  17),  INT8_C(  71),  INT8_C(  30),
        -INT8_C(  66), -INT8_C(  60),  INT8_C(  66),  INT8_C(  69),  INT8_C(  10),  INT8_C( 116),  INT8_C(  13), -INT8_C( 113) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_loadu_epi8(test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_loadu_epi8");
    easysimd_test_x86_assert_equal_i8x32(easysimd_mm256_load_si256(EASYSIMD_ALIGN_CAST(easysimd__m256i const *, test_vec[i].r)), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = a;

    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_loadu_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_32(easysimd__m256i) const int16_t a[16];
    const int16_t r[16];
  } test_vec[] = {
    { { -INT16_C( 24694),  INT16_C( 23546), -INT16_C( 20906), -INT16_C(  2504), -INT16_C( 25748), -INT16_C( 19507),  INT16_C( 20044), -INT16_C( 17602),
        -INT16_C( 24665),  INT16_C( 32725),  INT16_C(  7601),  INT16_C( 28573), -INT16_C(  8223), -INT16_C(  4940), -INT16_C( 16044), -INT16_C(  8581) },
      { -INT16_C( 24694),  INT16_C( 23546), -INT16_C( 20906), -INT16_C(  2504), -INT16_C( 25748), -INT16_C( 19507),  INT16_C( 20044), -INT16_C( 17602),
        -INT16_C( 24665),  INT16_C( 32725),  INT16_C(  7601),  INT16_C( 28573), -INT16_C(  8223), -INT16_C(  4940), -INT16_C( 16044), -INT16_C(  8581) } },
    { {  INT16_C( 30304), -INT16_C( 18887),  INT16_C( 28964), -INT16_C( 28243),  INT16_C( 31245),  INT16_C( 22852), -INT16_C( 31800),  INT16_C( 28692),
        -INT16_C(  5598), -INT16_C( 11281), -INT16_C( 29689), -INT16_C(  6078), -INT16_C(  2452), -INT16_C( 16172),  INT16_C( 20664),  INT16_C(  6302) },
      {  INT16_C( 30304), -INT16_C( 18887),  INT16_C( 28964), -INT16_C( 28243),  INT16_C( 31245),  INT16_C( 22852), -INT16_C( 31800),  INT16_C( 28692),
        -INT16_C(  5598), -INT16_C( 11281), -INT16_C( 29689), -INT16_C(  6078), -INT16_C(  2452), -INT16_C( 16172),  INT16_C( 20664),  INT16_C(  6302) } },
    { { -INT16_C( 10042), -INT16_C(  5425),  INT16_C( 31817),  INT16_C( 22139), -INT16_C( 16138), -INT16_C( 16720), -INT16_C( 15293),  INT16_C( 25902),
         INT16_C(  7598), -INT16_C( 19143),  INT16_C( 31658),  INT16_C(  5790),  INT16_C( 29298),  INT16_C( 10966),  INT16_C( 29890), -INT16_C( 30654) },
      { -INT16_C( 10042), -INT16_C(  5425),  INT16_C( 31817),  INT16_C( 22139), -INT16_C( 16138), -INT16_C( 16720), -INT16_C( 15293),  INT16_C( 25902),
         INT16_C(  7598), -INT16_C( 19143),  INT16_C( 31658),  INT16_C(  5790),  INT16_C( 29298),  INT16_C( 10966),  INT16_C( 29890), -INT16_C( 30654) } },
    { {  INT16_C(  4428), -INT16_C( 27021), -INT16_C(  4467), -INT16_C( 31764), -INT16_C( 25426), -INT16_C(  3774),  INT16_C( 28769),  INT16_C(  3927),
        -INT16_C( 28530),  INT16_C( 14533),  INT16_C( 25355),  INT16_C( 32078),  INT16_C(  9429), -INT16_C( 26457), -INT16_C(  5480), -INT16_C(  6880) },
      {  INT16_C(  4428), -INT16_C( 27021), -INT16_C(  4467), -INT16_C( 31764), -INT16_C( 25426), -INT16_C(  3774),  INT16_C( 28769),  INT16_C(  3927),
        -INT16_C( 28530),  INT16_C( 14533),  INT16_C( 25355),  INT16_C( 32078),  INT16_C(  9429), -INT16_C( 26457), -INT16_C(  5480), -INT16_C(  6880) } },
    { { -INT16_C( 27653), -INT16_C( 30341),  INT16_C( 26498),  INT16_C( 12300),  INT16_C( 19972),  INT16_C( 25890),  INT16_C( 31167),  INT16_C( 19828),
         INT16_C( 14601),  INT16_C(  5253), -INT16_C( 11364),  INT16_C( 29330),  INT16_C( 14839), -INT16_C( 28918),  INT16_C( 10787),  INT16_C(  8052) },
      { -INT16_C( 27653), -INT16_C( 30341),  INT16_C( 26498),  INT16_C( 12300),  INT16_C( 19972),  INT16_C( 25890),  INT16_C( 31167),  INT16_C( 19828),
         INT16_C( 14601),  INT16_C(  5253), -INT16_C( 11364),  INT16_C( 29330),  INT16_C( 14839), -INT16_C( 28918),  INT16_C( 10787),  INT16_C(  8052) } },
    { { -INT16_C(  4162),  INT16_C( 16552), -INT16_C( 19369),  INT16_C( 23408), -INT16_C( 28157), -INT16_C( 15680),  INT16_C( 13323),  INT16_C(  5135),
        -INT16_C( 27538),  INT16_C(  2601), -INT16_C( 17561),  INT16_C( 24188), -INT16_C( 30988),  INT16_C(  6381),  INT16_C( 25265),  INT16_C( 28471) },
      { -INT16_C(  4162),  INT16_C( 16552), -INT16_C( 19369),  INT16_C( 23408), -INT16_C( 28157), -INT16_C( 15680),  INT16_C( 13323),  INT16_C(  5135),
        -INT16_C( 27538),  INT16_C(  2601), -INT16_C( 17561),  INT16_C( 24188), -INT16_C( 30988),  INT16_C(  6381),  INT16_C( 25265),  INT16_C( 28471) } },
    { { -INT16_C(  8367), -INT16_C( 22353),  INT16_C(  8083), -INT16_C( 27133), -INT16_C( 15438), -INT16_C( 17064),  INT16_C( 26616),  INT16_C( 26322),
        -INT16_C(  1029),  INT16_C( 25200), -INT16_C(  4682), -INT16_C( 21824), -INT16_C( 20877),  INT16_C(  9410), -INT16_C(  1776),  INT16_C( 24979) },
      { -INT16_C(  8367), -INT16_C( 22353),  INT16_C(  8083), -INT16_C( 27133), -INT16_C( 15438), -INT16_C( 17064),  INT16_C( 26616),  INT16_C( 26322),
        -INT16_C(  1029),  INT16_C( 25200), -INT16_C(  4682), -INT16_C( 21824), -INT16_C( 20877),  INT16_C(  9410), -INT16_C(  1776),  INT16_C( 24979) } },
    { {  INT16_C( 17112),  INT16_C( 27658),  INT16_C(  3426),  INT16_C(  5122),  INT16_C( 23505), -INT16_C( 13871), -INT16_C( 23614), -INT16_C( 16849),
        -INT16_C( 24674),  INT16_C( 21536), -INT16_C(  7796),  INT16_C(   255), -INT16_C( 15985), -INT16_C( 24796), -INT16_C( 18245), -INT16_C( 27904) },
      {  INT16_C( 17112),  INT16_C( 27658),  INT16_C(  3426),  INT16_C(  5122),  INT16_C( 23505), -INT16_C( 13871), -INT16_C( 23614), -INT16_C( 16849),
        -INT16_C( 24674),  INT16_C( 21536), -INT16_C(  7796),  INT16_C(   255), -INT16_C( 15985), -INT16_C( 24796), -INT16_C( 18245), -INT16_C( 27904) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_loadu_epi16(test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_loadu_epi16");
    easysimd_test_x86_assert_equal_i16x16(easysimd_mm256_load_si256(EASYSIMD_ALIGN_CAST(easysimd__m256i const *, test_vec[i].r)), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = a;

    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_loadu_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_32(easysimd__m256i) const int32_t a[8];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(  1560218362), -INT32_C(   378535400),  INT32_C(   531776093), -INT32_C(  2065833499),  INT32_C(   232324736),  INT32_C(  1846400991),  INT32_C(  1410150809), -INT32_C(   454619671) },
      {  INT32_C(  1560218362), -INT32_C(   378535400),  INT32_C(   531776093), -INT32_C(  2065833499),  INT32_C(   232324736),  INT32_C(  1846400991),  INT32_C(  1410150809), -INT32_C(   454619671) } },
    { {  INT32_C(   809559832),  INT32_C(  1176089065), -INT32_C(   664417293),  INT32_C(   744244140),  INT32_C(   540620097), -INT32_C(  1517402612),  INT32_C(  1643748216), -INT32_C(  1069162072) },
      {  INT32_C(   809559832),  INT32_C(  1176089065), -INT32_C(   664417293),  INT32_C(   744244140),  INT32_C(   540620097), -INT32_C(  1517402612),  INT32_C(  1643748216), -INT32_C(  1069162072) } },
    { { -INT32_C(  1309636920),  INT32_C(   720832823), -INT32_C(  2147328812), -INT32_C(   525508705), -INT32_C(  1610553708), -INT32_C(  1522102739), -INT32_C(   771342551), -INT32_C(   393065440) },
      { -INT32_C(  1309636920),  INT32_C(   720832823), -INT32_C(  2147328812), -INT32_C(   525508705), -INT32_C(  1610553708), -INT32_C(  1522102739), -INT32_C(   771342551), -INT32_C(   393065440) } },
    { {  INT32_C(   161055698),  INT32_C(  1630769292), -INT32_C(  1931397651),  INT32_C(   678268564), -INT32_C(  1563857547),  INT32_C(   625414140),  INT32_C(  1878478158),  INT32_C(  1800899225) },
      {  INT32_C(   161055698),  INT32_C(  1630769292), -INT32_C(  1931397651),  INT32_C(   678268564), -INT32_C(  1563857547),  INT32_C(   625414140),  INT32_C(  1878478158),  INT32_C(  1800899225) } },
    { { -INT32_C(  1720389363),  INT32_C(  1861920641),  INT32_C(  1912331485), -INT32_C(   543528854), -INT32_C(   780049451), -INT32_C(  1057503118), -INT32_C(  1355813354), -INT32_C(  2061793416) },
      { -INT32_C(  1720389363),  INT32_C(  1861920641),  INT32_C(  1912331485), -INT32_C(   543528854), -INT32_C(   780049451), -INT32_C(  1057503118), -INT32_C(  1355813354), -INT32_C(  2061793416) } },
    { { -INT32_C(   115372168),  INT32_C(   342366519),  INT32_C(  1619354613), -INT32_C(  1606475829), -INT32_C(   193805950), -INT32_C(  1615500919), -INT32_C(   800070569), -INT32_C(   480941461) },
      { -INT32_C(   115372168),  INT32_C(   342366519),  INT32_C(  1619354613), -INT32_C(  1606475829), -INT32_C(   193805950), -INT32_C(  1615500919), -INT32_C(   800070569), -INT32_C(   480941461) } },
    { {  INT32_C(   819819769), -INT32_C(  2092677746),  INT32_C(  1944308392),  INT32_C(  1813193705),  INT32_C(  1835042276),  INT32_C(  1175262702),  INT32_C(  1695964410), -INT32_C(  1085707322) },
      {  INT32_C(   819819769), -INT32_C(  2092677746),  INT32_C(  1944308392),  INT32_C(  1813193705),  INT32_C(  1835042276),  INT32_C(  1175262702),  INT32_C(  1695964410), -INT32_C(  1085707322) } },
    { {  INT32_C(  1861232352),  INT32_C(   334574699), -INT32_C(   393816578), -INT32_C(   598435336),  INT32_C(   222934047), -INT32_C(  1001171254),  INT32_C(  2015979954), -INT32_C(  1254591787) },
      {  INT32_C(  1861232352),  INT32_C(   334574699), -INT32_C(   393816578), -INT32_C(   598435336),  INT32_C(   222934047), -INT32_C(  1001171254),  INT32_C(  2015979954), -INT32_C(  1254591787) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_loadu_epi32(test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_loadu_epi32");
    easysimd_test_x86_assert_equal_i32x8(easysimd_mm256_load_si256(EASYSIMD_ALIGN_CAST(easysimd__m256i const *, test_vec[i].r)), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = a;

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_loadu_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_32(easysimd__m256i) const int64_t a[4];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 2730480485383468799),  INT64_C( 3961809025040497319), -INT64_C( 7760876929369986550),  INT64_C( 2630957754019333904) },
      { -INT64_C( 2730480485383468799),  INT64_C( 3961809025040497319), -INT64_C( 7760876929369986550),  INT64_C( 2630957754019333904) } },
    { {  INT64_C( 2406427958756382740),  INT64_C( 1484904791614610964), -INT64_C( 2360563711534695397), -INT64_C( 4781223386344087970) },
      {  INT64_C( 2406427958756382740),  INT64_C( 1484904791614610964), -INT64_C( 2360563711534695397), -INT64_C( 4781223386344087970) } },
    { {  INT64_C( 5864722717878051783),  INT64_C( 7398096197995911564), -INT64_C( 1370136013387598003), -INT64_C( 7183219597633509398) },
      {  INT64_C( 5864722717878051783),  INT64_C( 7398096197995911564), -INT64_C( 1370136013387598003), -INT64_C( 7183219597633509398) } },
    { { -INT64_C( 3205083785718752777), -INT64_C( 2680742885939594470),  INT64_C( 1263244472435006221),  INT64_C(   39394444786922777) },
      { -INT64_C( 3205083785718752777), -INT64_C( 2680742885939594470),  INT64_C( 1263244472435006221),  INT64_C(   39394444786922777) } },
    { { -INT64_C( 4203624633801086578),  INT64_C( 3916713484056400884),  INT64_C( 6704269143766553041),  INT64_C( 3699662719747403598) },
      { -INT64_C( 4203624633801086578),  INT64_C( 3916713484056400884),  INT64_C( 6704269143766553041),  INT64_C( 3699662719747403598) } },
    { {  INT64_C( 4084959771584138049), -INT64_C( 6690919879014753339), -INT64_C( 7060055079283591580),  INT64_C( 7578517111345336660) },
      {  INT64_C( 4084959771584138049), -INT64_C( 6690919879014753339), -INT64_C( 7060055079283591580),  INT64_C( 7578517111345336660) } },
    { { -INT64_C( 6100963634310728488), -INT64_C( 8766892043372261664), -INT64_C( 1539611403198992203), -INT64_C( 4821648871914235772) },
      { -INT64_C( 6100963634310728488), -INT64_C( 8766892043372261664), -INT64_C( 1539611403198992203), -INT64_C( 4821648871914235772) } },
    { {  INT64_C( 5896672549719927620), -INT64_C( 2096610149610066370), -INT64_C( 3618209875233467063),  INT64_C(   55047857207160097) },
      {  INT64_C( 5896672549719927620), -INT64_C( 2096610149610066370), -INT64_C( 3618209875233467063),  INT64_C(   55047857207160097) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_loadu_epi64(test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_loadu_epi64");
    easysimd_test_x86_assert_equal_i64x4(easysimd_mm256_load_si256(EASYSIMD_ALIGN_CAST(easysimd__m256i const *, test_vec[i].r)), r);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = a;

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_loadu_epi8(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int8_t src[32];
    const uint32_t k;
    const int8_t a[32];
    const int8_t r[32];
  } test_vec[] = {
    { { -INT8_C(  66),  INT8_C(  86),  INT8_C(  10), -INT8_C(  76), -INT8_C(  23), -INT8_C(  56), -INT8_C( 122),  INT8_C(  46),
        -INT8_C(  72), -INT8_C(  53),  INT8_C(  42),  INT8_C(  23),  INT8_C(  44),  INT8_C(  52), -INT8_C(  82),  INT8_C(  34),
         INT8_C(  15), -INT8_C( 101), -INT8_C(  35), -INT8_C(   6),  INT8_C(  13),  INT8_C(  58), -INT8_C( 123),  INT8_C(  87),
        -INT8_C( 104), -INT8_C(  84),  INT8_C( 103), -INT8_C( 103), -INT8_C(  16),  INT8_C(  98),  INT8_C(  79), -INT8_C(  82) },
      UINT32_C(2707577272),
      {  INT8_C(  33), -INT8_C(  24), -INT8_C(  48), -INT8_C(  38), -INT8_C(  76), -INT8_C(   6), -INT8_C(  15), -INT8_C(  32),
         INT8_C(  46), -INT8_C(  97),  INT8_C(   2),  INT8_C(  61),  INT8_C(  59), -INT8_C(  32),  INT8_C(  55),  INT8_C(  72),
         INT8_C(  26), -INT8_C(  68), -INT8_C(  97), -INT8_C(  78),  INT8_C( 104),  INT8_C(   6),  INT8_C(  75),  INT8_C(  88),
         INT8_C( 104), -INT8_C( 101),  INT8_C(   6),  INT8_C(  32), -INT8_C(  12),  INT8_C( 104), -INT8_C(  62),  INT8_C(  22) },
      { -INT8_C(  66),  INT8_C(  86),  INT8_C(  10), -INT8_C(  38), -INT8_C(  76), -INT8_C(   6), -INT8_C( 122), -INT8_C(  32),
         INT8_C(  46), -INT8_C(  53),  INT8_C(  42),  INT8_C(  61),  INT8_C(  59),  INT8_C(  52),  INT8_C(  55),  INT8_C(  34),
         INT8_C(  15), -INT8_C(  68), -INT8_C(  35), -INT8_C(   6),  INT8_C(  13),  INT8_C(   6),  INT8_C(  75),  INT8_C(  87),
         INT8_C( 104), -INT8_C(  84),  INT8_C( 103), -INT8_C( 103), -INT8_C(  16),  INT8_C( 104),  INT8_C(  79),  INT8_C(  22) } },
    { {  INT8_C(  81), -INT8_C( 110), -INT8_C(  16),  INT8_C(   5), -INT8_C( 116), -INT8_C(  31), -INT8_C(  27), -INT8_C(  70),
             INT8_MIN, -INT8_C(  24), -INT8_C(   9), -INT8_C(  69), -INT8_C(  56),  INT8_C(  46),  INT8_C(   3), -INT8_C(  30),
        -INT8_C(  22), -INT8_C(  94), -INT8_C( 108),  INT8_C(  82), -INT8_C(  87), -INT8_C(  32), -INT8_C(  85),  INT8_C(  17),
         INT8_C( 123), -INT8_C(  79),  INT8_C(  50),  INT8_C( 111),  INT8_C(  26), -INT8_C(  12), -INT8_C( 123),  INT8_C( 107) },
      UINT32_C( 309360006),
      {  INT8_C(  86),  INT8_C(  85), -INT8_C(  52), -INT8_C(  41),  INT8_C(  61), -INT8_C(  61), -INT8_C( 110),  INT8_C(   5),
        -INT8_C(  15), -INT8_C( 106), -INT8_C(  25), -INT8_C(  37),  INT8_C(  56),  INT8_C( 124),  INT8_C(  45), -INT8_C(  31),
         INT8_C(  92), -INT8_C(  40), -INT8_C(  13), -INT8_C(  41), -INT8_C( 118),  INT8_C(  37),  INT8_C(  70), -INT8_C(  92),
         INT8_C(  25), -INT8_C(  52),  INT8_C(  15), -INT8_C(  97),  INT8_C(  65),      INT8_MAX, -INT8_C(  79), -INT8_C( 104) },
      {  INT8_C(  81),  INT8_C(  85), -INT8_C(  52),  INT8_C(   5), -INT8_C( 116), -INT8_C(  31), -INT8_C(  27),  INT8_C(   5),
        -INT8_C(  15), -INT8_C(  24), -INT8_C(  25), -INT8_C(  69),  INT8_C(  56),  INT8_C( 124),  INT8_C(  45), -INT8_C(  30),
        -INT8_C(  22), -INT8_C(  94), -INT8_C( 108),  INT8_C(  82), -INT8_C( 118),  INT8_C(  37),  INT8_C(  70),  INT8_C(  17),
         INT8_C( 123), -INT8_C(  52),  INT8_C(  50),  INT8_C( 111),  INT8_C(  65), -INT8_C(  12), -INT8_C( 123),  INT8_C( 107) } },
    { { -INT8_C(  44),  INT8_C( 125),  INT8_C( 111),  INT8_C(  18),  INT8_C(  64),  INT8_C(   1),  INT8_C(  23),  INT8_C(  49),
        -INT8_C( 105), -INT8_C(   1),  INT8_C(  12), -INT8_C(  48),  INT8_C( 123),  INT8_C(  57), -INT8_C(  79), -INT8_C(  41),
         INT8_C(  18), -INT8_C(  92), -INT8_C(  82), -INT8_C( 100), -INT8_C(  55), -INT8_C(  12),  INT8_C(  64), -INT8_C(  30),
        -INT8_C(  64),  INT8_C(  79), -INT8_C( 127),  INT8_C(   2), -INT8_C(  50),  INT8_C(  50), -INT8_C( 102), -INT8_C(  94) },
      UINT32_C(4021553583),
      {  INT8_C(  10), -INT8_C(  52),  INT8_C(  32), -INT8_C(  94), -INT8_C(  53),  INT8_C(  44),  INT8_C( 114),  INT8_C(  70),
         INT8_C( 102),  INT8_C(  35),  INT8_C(  29),  INT8_C( 120), -INT8_C(  56), -INT8_C(  53),  INT8_C(  20), -INT8_C( 111),
        -INT8_C(  65),  INT8_C(  84),  INT8_C( 116),      INT8_MIN, -INT8_C(  93), -INT8_C(  11), -INT8_C( 126),  INT8_C( 113),
         INT8_C(  40),  INT8_C(  28),  INT8_C(  19), -INT8_C(  41),  INT8_C(  37), -INT8_C(  56), -INT8_C(  57),  INT8_C(  47) },
      {  INT8_C(  10), -INT8_C(  52),  INT8_C(  32), -INT8_C(  94),  INT8_C(  64),  INT8_C(  44),  INT8_C(  23),  INT8_C(  70),
         INT8_C( 102), -INT8_C(   1),  INT8_C(  12),  INT8_C( 120),  INT8_C( 123),  INT8_C(  57), -INT8_C(  79), -INT8_C(  41),
         INT8_C(  18), -INT8_C(  92),  INT8_C( 116), -INT8_C( 100), -INT8_C(  93), -INT8_C(  11),  INT8_C(  64),  INT8_C( 113),
         INT8_C(  40),  INT8_C(  28),  INT8_C(  19), -INT8_C(  41), -INT8_C(  50), -INT8_C(  56), -INT8_C(  57),  INT8_C(  47) } },
    { { -INT8_C( 108), -INT8_C(  25), -INT8_C(  47),  INT8_C(  95),  INT8_C(  20),  INT8_C(  67), -INT8_C(  91),  INT8_C( 122),
         INT8_C( 103), -INT8_C(  62), -INT8_C(  14),  INT8_C(  47), -INT8_C( 115),  INT8_C(   6), -INT8_C(  64),  INT8_C(  76),
         INT8_C(  90),  INT8_C(  52), -INT8_C(  52), -INT8_C(   3),  INT8_C(  42),  INT8_C(  78),  INT8_C( 110),  INT8_C(  82),
         INT8_C( 106), -INT8_C( 127),  INT8_C(  41), -INT8_C( 113),  INT8_C(  73), -INT8_C(  16), -INT8_C(  65), -INT8_C(  35) },
      UINT32_C(3963392216),
      { -INT8_C(  44), -INT8_C(  31),  INT8_C( 102),  INT8_C(  59), -INT8_C(  93),  INT8_C(  88),  INT8_C( 106),  INT8_C(  48),
         INT8_C(  94),  INT8_C(  42),  INT8_C( 125), -INT8_C(  72),  INT8_C(  95),  INT8_C(  73), -INT8_C(  75), -INT8_C( 119),
        -INT8_C( 104),  INT8_C(  35), -INT8_C(  37),  INT8_C(   2), -INT8_C(  92),  INT8_C(   4), -INT8_C( 110), -INT8_C(  18),
        -INT8_C(  11),  INT8_C(  81), -INT8_C(  53), -INT8_C(  51), -INT8_C(  31),  INT8_C(   8), -INT8_C(  71), -INT8_C(  75) },
      { -INT8_C( 108), -INT8_C(  25), -INT8_C(  47),  INT8_C(  59), -INT8_C(  93),  INT8_C(  67),  INT8_C( 106),  INT8_C(  48),
         INT8_C( 103), -INT8_C(  62), -INT8_C(  14),  INT8_C(  47),  INT8_C(  95),  INT8_C(   6), -INT8_C(  64), -INT8_C( 119),
         INT8_C(  90),  INT8_C(  52), -INT8_C(  37),  INT8_C(   2), -INT8_C(  92),  INT8_C(   4),  INT8_C( 110),  INT8_C(  82),
         INT8_C( 106), -INT8_C( 127), -INT8_C(  53), -INT8_C(  51),  INT8_C(  73),  INT8_C(   8), -INT8_C(  71), -INT8_C(  75) } },
    { { -INT8_C(  23),  INT8_C(  31), -INT8_C(  16), -INT8_C( 115),  INT8_C( 119),  INT8_C(  90), -INT8_C(  67), -INT8_C(  43),
        -INT8_C( 123),  INT8_C(  58), -INT8_C( 115), -INT8_C(  28), -INT8_C( 124),  INT8_C(  66),  INT8_C( 109),  INT8_C(  28),
         INT8_C( 101),  INT8_C(  72),  INT8_C(  30),  INT8_C(   9),  INT8_C(  76), -INT8_C(  80), -INT8_C(   9),  INT8_C(  65),
         INT8_C(   1), -INT8_C(  61),  INT8_C(  14), -INT8_C(  29), -INT8_C(  53), -INT8_C(  57), -INT8_C( 104), -INT8_C(  76) },
      UINT32_C(1564576230),
      { -INT8_C(  29), -INT8_C(   1),  INT8_C(  50),  INT8_C( 104),  INT8_C(  57), -INT8_C(  65),  INT8_C(  76), -INT8_C(  67),
         INT8_C(   1), -INT8_C(  71), -INT8_C(  39),  INT8_C( 102),  INT8_C(   1), -INT8_C(   8),  INT8_C( 112),  INT8_C(  78),
        -INT8_C(  88),  INT8_C( 103), -INT8_C( 113), -INT8_C(  86),  INT8_C(  42), -INT8_C(  98), -INT8_C( 115), -INT8_C(  11),
         INT8_C( 101),  INT8_C(  37), -INT8_C(  86),  INT8_C(  76), -INT8_C(  82), -INT8_C(  21), -INT8_C(  87), -INT8_C( 110) },
      { -INT8_C(  23), -INT8_C(   1),  INT8_C(  50), -INT8_C( 115),  INT8_C( 119), -INT8_C(  65),  INT8_C(  76), -INT8_C(  67),
         INT8_C(   1),  INT8_C(  58), -INT8_C( 115),  INT8_C( 102), -INT8_C( 124),  INT8_C(  66),  INT8_C( 109),  INT8_C(  78),
        -INT8_C(  88),  INT8_C(  72),  INT8_C(  30),  INT8_C(   9),  INT8_C(  76), -INT8_C(  80), -INT8_C( 115),  INT8_C(  65),
         INT8_C( 101), -INT8_C(  61), -INT8_C(  86),  INT8_C(  76), -INT8_C(  82), -INT8_C(  57), -INT8_C(  87), -INT8_C(  76) } },
    { { -INT8_C(  22), -INT8_C(  36), -INT8_C(   6),  INT8_C(  36), -INT8_C( 101),  INT8_C(  71), -INT8_C(  31), -INT8_C(  99),
         INT8_C(   0), -INT8_C(  69),  INT8_C(   3),  INT8_C(   2), -INT8_C(  77),  INT8_C( 115),  INT8_C(  80),  INT8_C(  91),
        -INT8_C(  37), -INT8_C(  33),  INT8_C(   5),  INT8_C(   5),  INT8_C( 125), -INT8_C( 110), -INT8_C(   5), -INT8_C(  29),
        -INT8_C(  72), -INT8_C(  91),  INT8_C(  47),  INT8_C( 102), -INT8_C( 112), -INT8_C(  40), -INT8_C(   8),  INT8_C( 123) },
      UINT32_C(1352659892),
      {  INT8_C(  58),      INT8_MIN, -INT8_C(  19),  INT8_C(  58),  INT8_C(  59), -INT8_C(  16),  INT8_C(  60), -INT8_C(  18),
         INT8_C( 100), -INT8_C( 116),  INT8_C(  74),  INT8_C(  63),  INT8_C( 108),  INT8_C(  79),  INT8_C(  68), -INT8_C(  23),
        -INT8_C(  30),  INT8_C(  63), -INT8_C(  52), -INT8_C( 102), -INT8_C(  28), -INT8_C(   5),  INT8_C(   0),  INT8_C( 117),
        -INT8_C(  44), -INT8_C(   7), -INT8_C(  16), -INT8_C( 120), -INT8_C(  20), -INT8_C( 113), -INT8_C(  40),  INT8_C(  38) },
      { -INT8_C(  22), -INT8_C(  36), -INT8_C(  19),  INT8_C(  36),  INT8_C(  59), -INT8_C(  16), -INT8_C(  31), -INT8_C(  18),
         INT8_C( 100), -INT8_C( 116),  INT8_C(   3),  INT8_C(   2),  INT8_C( 108),  INT8_C(  79),  INT8_C(  68), -INT8_C(  23),
        -INT8_C(  30),  INT8_C(  63), -INT8_C(  52), -INT8_C( 102), -INT8_C(  28), -INT8_C( 110), -INT8_C(   5),  INT8_C( 117),
        -INT8_C(  72), -INT8_C(  91),  INT8_C(  47),  INT8_C( 102), -INT8_C(  20), -INT8_C(  40), -INT8_C(  40),  INT8_C( 123) } },
    { {  INT8_C(  15), -INT8_C(  59),  INT8_C(  96),  INT8_C(  75), -INT8_C(  74), -INT8_C(  99),  INT8_C(  57),  INT8_C(  26),
         INT8_C(  41), -INT8_C( 125),  INT8_C(  89), -INT8_C( 107), -INT8_C(  45), -INT8_C(  99),      INT8_MAX, -INT8_C(  75),
        -INT8_C(  35),  INT8_C(  75),  INT8_C(  79), -INT8_C(  63),  INT8_C(  71),  INT8_C(  79),  INT8_C(  54),  INT8_C(  27),
         INT8_C(  72),  INT8_C(  38), -INT8_C(  93),  INT8_C(  52), -INT8_C(  75),  INT8_C( 124),  INT8_C(  90), -INT8_C(  59) },
      UINT32_C(4145068865),
      {  INT8_C(  88),  INT8_C(  73),  INT8_C(  17), -INT8_C( 127), -INT8_C(  51),  INT8_C( 106),  INT8_C(  23), -INT8_C(  96),
         INT8_C(   8), -INT8_C( 106),  INT8_C(  85), -INT8_C(  27), -INT8_C(  31), -INT8_C(  92), -INT8_C(  90),  INT8_C(  40),
        -INT8_C(  13), -INT8_C(  35),  INT8_C(  67),  INT8_C(  60),  INT8_C(   3), -INT8_C(  25),  INT8_C( 112), -INT8_C(  71),
         INT8_C(  99), -INT8_C(  53),  INT8_C( 126), -INT8_C(  92), -INT8_C( 122), -INT8_C( 114), -INT8_C( 100), -INT8_C(  34) },
      {  INT8_C(  88), -INT8_C(  59),  INT8_C(  96),  INT8_C(  75), -INT8_C(  74), -INT8_C(  99),  INT8_C(  23),  INT8_C(  26),
         INT8_C(   8), -INT8_C( 106),  INT8_C(  89), -INT8_C(  27), -INT8_C(  31), -INT8_C(  92),      INT8_MAX,  INT8_C(  40),
        -INT8_C(  35),  INT8_C(  75),  INT8_C(  79), -INT8_C(  63),  INT8_C(   3),  INT8_C(  79),  INT8_C(  54),  INT8_C(  27),
         INT8_C(  99), -INT8_C(  53),  INT8_C( 126),  INT8_C(  52), -INT8_C( 122), -INT8_C( 114), -INT8_C( 100), -INT8_C(  34) } },
    { { -INT8_C(  41), -INT8_C(  83),  INT8_C(  95), -INT8_C(  92),  INT8_C(  24),  INT8_C( 118),  INT8_C(  68),  INT8_C(  32),
         INT8_C(  12), -INT8_C( 103),  INT8_C(   5), -INT8_C(  18),  INT8_C(  61), -INT8_C(  85),  INT8_C(  22),  INT8_C(  49),
        -INT8_C( 120),  INT8_C(  90),  INT8_C( 109), -INT8_C( 116),  INT8_C(  65), -INT8_C(  35),  INT8_C(  69), -INT8_C(  92),
        -INT8_C(  88), -INT8_C(  61),  INT8_C(  72),  INT8_C(  46),  INT8_C(  81), -INT8_C(  28),  INT8_C(  12),  INT8_C(  40) },
      UINT32_C(2865589394),
      { -INT8_C(  30),  INT8_C(  17), -INT8_C(  54), -INT8_C(  17), -INT8_C(  85), -INT8_C(  49), -INT8_C(  35), -INT8_C(  24),
         INT8_C( 122), -INT8_C(  13),  INT8_C(  25),  INT8_C(   3),  INT8_C(  77), -INT8_C( 122), -INT8_C( 113), -INT8_C( 114),
         INT8_C( 100), -INT8_C(  44),  INT8_C(  50),  INT8_C(  12), -INT8_C( 105),  INT8_C( 123),  INT8_C(  59), -INT8_C(  24),
         INT8_C(  95),  INT8_C(  71),  INT8_C(  16), -INT8_C(  15), -INT8_C(  77), -INT8_C(  35), -INT8_C( 101), -INT8_C( 106) },
      { -INT8_C(  41),  INT8_C(  17),  INT8_C(  95), -INT8_C(  92), -INT8_C(  85),  INT8_C( 118),  INT8_C(  68), -INT8_C(  24),
         INT8_C(  12), -INT8_C( 103),  INT8_C(  25),  INT8_C(   3),  INT8_C(  61), -INT8_C( 122), -INT8_C( 113),  INT8_C(  49),
         INT8_C( 100),  INT8_C(  90),  INT8_C(  50),  INT8_C(  12),  INT8_C(  65), -INT8_C(  35),  INT8_C(  59), -INT8_C(  24),
        -INT8_C(  88),  INT8_C(  71),  INT8_C(  72), -INT8_C(  15),  INT8_C(  81), -INT8_C(  35),  INT8_C(  12), -INT8_C( 106) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi8(test_vec[i].src);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_loadu_epi8(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_loadu_epi8");
    easysimd_test_x86_assert_equal_i8x32(r, easysimd_mm256_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i8x32();
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = easysimd_mm256_mask_loadu_epi8(src, k, &a);

    easysimd_test_x86_write_i8x32(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_loadu_epi8()
{
#if 1
  static const struct {
    easysimd__mmask32 k;
    const int8_t a[32];
    const int8_t r[32];
  } test_vec[] = {
    { UINT32_C(3034227991),
      { -INT8_C(  23), -INT8_C(   1),  INT8_C(  50), -INT8_C( 100), -INT8_C(   6),  INT8_C(  19), -INT8_C(   4), -INT8_C(  16),
         INT8_C(  60), -INT8_C( 111), -INT8_C(  20), -INT8_C( 118), -INT8_C(   4),  INT8_C(  79), -INT8_C( 124), -INT8_C(  65),
        -INT8_C(  90), -INT8_C( 105),  INT8_C(  21), -INT8_C( 113), -INT8_C(  77), -INT8_C(  39),  INT8_C(   5),  INT8_C(  71),
        -INT8_C(  83),  INT8_C(  85), -INT8_C(  26), -INT8_C(  60), -INT8_C(   6), -INT8_C(  64),  INT8_C( 120), -INT8_C(  29) },
      { -INT8_C(  23), -INT8_C(   1),  INT8_C(  50),  INT8_C(   0), -INT8_C(   6),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  60),  INT8_C(   0), -INT8_C(  20),  INT8_C(   0),  INT8_C(   0),  INT8_C(  79),  INT8_C(   0), -INT8_C(  65),
         INT8_C(   0), -INT8_C( 105),  INT8_C(   0), -INT8_C( 113), -INT8_C(  77),  INT8_C(   0),  INT8_C(   5),  INT8_C(  71),
         INT8_C(   0),  INT8_C(   0), -INT8_C(  26),  INT8_C(   0), -INT8_C(   6), -INT8_C(  64),  INT8_C(   0), -INT8_C(  29) } },
    { UINT32_C(3112151743),
      { -INT8_C(  67),  INT8_C( 124), -INT8_C(  87), -INT8_C(   7),  INT8_C(  13), -INT8_C( 106), -INT8_C( 125),  INT8_C(   9),
        -INT8_C(  27),  INT8_C(   8), -INT8_C(  55), -INT8_C( 117), -INT8_C(  97), -INT8_C(  34),  INT8_C(  26),  INT8_C(  82),
        -INT8_C(  73),  INT8_C(  32), -INT8_C( 103),  INT8_C( 100),  INT8_C( 117),      INT8_MAX,  INT8_C(  40),  INT8_C( 111),
         INT8_C(  63), -INT8_C(  96),  INT8_C(  82), -INT8_C(   2),  INT8_C(  75), -INT8_C(  47), -INT8_C(  73),  INT8_C(   8) },
      { -INT8_C(  67),  INT8_C( 124), -INT8_C(  87), -INT8_C(   7),  INT8_C(  13), -INT8_C( 106),  INT8_C(   0),  INT8_C(   9),
         INT8_C(   0),  INT8_C(   8),  INT8_C(   0), -INT8_C( 117),  INT8_C(   0), -INT8_C(  34),  INT8_C(   0),  INT8_C(  82),
        -INT8_C(  73),  INT8_C(  32), -INT8_C( 103),  INT8_C( 100),  INT8_C( 117),      INT8_MAX,  INT8_C(  40),  INT8_C(   0),
         INT8_C(  63),  INT8_C(   0),  INT8_C(   0), -INT8_C(   2),  INT8_C(  75), -INT8_C(  47),  INT8_C(   0),  INT8_C(   8) } },
    { UINT32_C(1526882381),
      { -INT8_C(  10), -INT8_C( 123),  INT8_C( 100), -INT8_C(  36), -INT8_C( 115),  INT8_C(  45),  INT8_C( 103),  INT8_C(  44),
         INT8_C(  11), -INT8_C( 126),  INT8_C( 126), -INT8_C(  61), -INT8_C(  94),  INT8_C(  23),  INT8_C(  39),  INT8_C(  23),
        -INT8_C( 106),  INT8_C(  80), -INT8_C( 122), -INT8_C(  43), -INT8_C(  16), -INT8_C(  40), -INT8_C(  45),  INT8_C(  59),
        -INT8_C(  87), -INT8_C( 118),  INT8_C(  68), -INT8_C(   9), -INT8_C(  21),  INT8_C(  70),  INT8_C(  82), -INT8_C(  31) },
      { -INT8_C(  10),  INT8_C(   0),  INT8_C( 100), -INT8_C(  36),  INT8_C(   0),  INT8_C(   0),  INT8_C( 103),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  23),  INT8_C(  39),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  80),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  87), -INT8_C( 118),  INT8_C(   0), -INT8_C(   9), -INT8_C(  21),  INT8_C(   0),  INT8_C(  82),  INT8_C(   0) } },
    { UINT32_C(1505605323),
      { -INT8_C(  28),  INT8_C(  37), -INT8_C( 123), -INT8_C(  17), -INT8_C(  89),  INT8_C(   4), -INT8_C(  78),  INT8_C(  73),
         INT8_C(  27), -INT8_C(  38),  INT8_C(  96), -INT8_C(  78),  INT8_C(  42), -INT8_C(  26), -INT8_C( 121),  INT8_C(  26),
        -INT8_C(  66),  INT8_C(  91),  INT8_C(  86),  INT8_C( 103), -INT8_C(  27), -INT8_C( 102),  INT8_C(  94), -INT8_C(  48),
        -INT8_C(  32), -INT8_C(  80), -INT8_C(  78), -INT8_C(  85),  INT8_C( 103),  INT8_C( 111),  INT8_C(   4),  INT8_C(  75) },
      { -INT8_C(  28),  INT8_C(  37),  INT8_C(   0), -INT8_C(  17),  INT8_C(   0),  INT8_C(   0), -INT8_C(  78),  INT8_C(  73),
         INT8_C(   0), -INT8_C(  38),  INT8_C(  96),  INT8_C(   0),  INT8_C(  42), -INT8_C(  26),  INT8_C(   0),  INT8_C(  26),
        -INT8_C(  66),  INT8_C(   0),  INT8_C(  86),  INT8_C( 103), -INT8_C(  27), -INT8_C( 102),  INT8_C(   0), -INT8_C(  48),
        -INT8_C(  32),  INT8_C(   0),  INT8_C(   0), -INT8_C(  85),  INT8_C( 103),  INT8_C(   0),  INT8_C(   4),  INT8_C(   0) } },
    { UINT32_C( 993692308),
      { -INT8_C( 114), -INT8_C(  19), -INT8_C( 124), -INT8_C(  87), -INT8_C(  57), -INT8_C(  28),  INT8_C(  91), -INT8_C(  15),
        -INT8_C(  54), -INT8_C(  29),  INT8_C(  11), -INT8_C( 120),  INT8_C(  62),  INT8_C(  97), -INT8_C(  16),  INT8_C(  35),
        -INT8_C(   5),  INT8_C(  78), -INT8_C(  12), -INT8_C(  37), -INT8_C(   1), -INT8_C(  90), -INT8_C( 121),  INT8_C( 102),
         INT8_C(  21), -INT8_C( 117), -INT8_C(  79), -INT8_C(  86),  INT8_C(  21), -INT8_C(  21), -INT8_C(  27), -INT8_C(  93) },
      {  INT8_C(   0),  INT8_C(   0), -INT8_C( 124),  INT8_C(   0), -INT8_C(  57),  INT8_C(   0),  INT8_C(   0), -INT8_C(  15),
         INT8_C(   0), -INT8_C(  29),  INT8_C(   0), -INT8_C( 120),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  35),
         INT8_C(   0),  INT8_C(  78),  INT8_C(   0), -INT8_C(  37), -INT8_C(   1), -INT8_C(  90),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  21), -INT8_C( 117),  INT8_C(   0), -INT8_C(  86),  INT8_C(  21), -INT8_C(  21),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C(2672650968),
      {  INT8_C(  78), -INT8_C(  88), -INT8_C( 112),  INT8_C(  25), -INT8_C( 117), -INT8_C( 100), -INT8_C(  95), -INT8_C(  55),
        -INT8_C(   3), -INT8_C( 111), -INT8_C(  19), -INT8_C(   7), -INT8_C(  32), -INT8_C(  31), -INT8_C(  44), -INT8_C(  33),
        -INT8_C( 121),  INT8_C(  91),  INT8_C(  69), -INT8_C( 100), -INT8_C(  25), -INT8_C(  10),  INT8_C(  70), -INT8_C(   4),
        -INT8_C(  31),  INT8_C(  44), -INT8_C(  96), -INT8_C(  70), -INT8_C( 106), -INT8_C(  19),  INT8_C(  89), -INT8_C(  28) },
      {  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  25), -INT8_C( 117),  INT8_C(   0), -INT8_C(  95), -INT8_C(  55),
         INT8_C(   0), -INT8_C( 111),  INT8_C(   0), -INT8_C(   7),  INT8_C(   0), -INT8_C(  31), -INT8_C(  44),  INT8_C(   0),
        -INT8_C( 121),  INT8_C(   0),  INT8_C(  69), -INT8_C( 100),  INT8_C(   0),  INT8_C(   0),  INT8_C(  70),  INT8_C(   0),
        -INT8_C(  31),  INT8_C(  44), -INT8_C(  96), -INT8_C(  70), -INT8_C( 106),  INT8_C(   0),  INT8_C(   0), -INT8_C(  28) } },
    { UINT32_C( 570288789),
      { -INT8_C( 122), -INT8_C(  97), -INT8_C(  22), -INT8_C( 125),  INT8_C(  48), -INT8_C(  41),  INT8_C( 124),  INT8_C(  16),
        -INT8_C(  72),  INT8_C(  81), -INT8_C(  17),  INT8_C(  63), -INT8_C(  84),  INT8_C(  52), -INT8_C(  36), -INT8_C( 109),
         INT8_C(  42),  INT8_C(  34), -INT8_C( 112),  INT8_C(  12),  INT8_C(  78),  INT8_C(  48), -INT8_C(  58), -INT8_C(  28),
         INT8_C(  29),  INT8_C(  31), -INT8_C(  55), -INT8_C(  78),  INT8_C(   9), -INT8_C(  58), -INT8_C(  45), -INT8_C( 113) },
      { -INT8_C( 122),  INT8_C(   0), -INT8_C(  22),  INT8_C(   0),  INT8_C(  48),  INT8_C(   0),  INT8_C(   0),  INT8_C(  16),
         INT8_C(   0),  INT8_C(  81),  INT8_C(   0),  INT8_C(  63),  INT8_C(   0),  INT8_C(  52), -INT8_C(  36), -INT8_C( 109),
         INT8_C(  42),  INT8_C(   0), -INT8_C( 112),  INT8_C(  12),  INT8_C(  78),  INT8_C(  48), -INT8_C(  58), -INT8_C(  28),
         INT8_C(  29),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  58),  INT8_C(   0),  INT8_C(   0) } },
    { UINT32_C(2517876325),
      { -INT8_C( 107), -INT8_C( 113), -INT8_C(  90),  INT8_C(  78), -INT8_C(  32), -INT8_C( 106), -INT8_C( 115), -INT8_C( 115),
        -INT8_C(  54),  INT8_C( 105),  INT8_C(  32), -INT8_C(  11), -INT8_C( 116), -INT8_C(  80),  INT8_C(   1), -INT8_C(  38),
        -INT8_C(  32), -INT8_C(  57), -INT8_C(  65), -INT8_C(   3), -INT8_C(  26), -INT8_C( 120), -INT8_C(  80), -INT8_C(  16),
         INT8_C(  78), -INT8_C( 125),      INT8_MAX, -INT8_C(  76),  INT8_C(  65), -INT8_C( 110),  INT8_C(  74), -INT8_C(  41) },
      { -INT8_C( 107),  INT8_C(   0), -INT8_C(  90),  INT8_C(   0),  INT8_C(   0), -INT8_C( 106), -INT8_C( 115),  INT8_C(   0),
         INT8_C(   0),  INT8_C( 105),  INT8_C(  32), -INT8_C(  11), -INT8_C( 116), -INT8_C(  80),  INT8_C(   0), -INT8_C(  38),
        -INT8_C(  32), -INT8_C(  57),  INT8_C(   0),  INT8_C(   0), -INT8_C(  26),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0), -INT8_C( 125),      INT8_MAX,  INT8_C(   0),  INT8_C(  65),  INT8_C(   0),  INT8_C(   0), -INT8_C(  41) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_loadu_epi8(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_loadu_epi8");
    easysimd_assert_equal_vi8(sizeof(r), (const int8_t*)&r, test_vec[i].r);
  }
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m256i a = easysimd_test_x86_random_i8x32();
    easysimd__m256i r = easysimd_mm256_maskz_loadu_epi8(k, &a.i8[0]);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

  return 0;
}

static int
test_easysimd_mm256_mask_loadu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int16_t src[16];
    const uint16_t k;
    const int16_t a[16];
    const int16_t r[16];
  } test_vec[] = {
    { {  INT16_C( 26095), -INT16_C( 25979),  INT16_C( 25140), -INT16_C( 20606), -INT16_C( 25515), -INT16_C( 23630),  INT16_C( 16674), -INT16_C( 31183),
         INT16_C( 25621), -INT16_C( 21357), -INT16_C( 12577),  INT16_C( 16020), -INT16_C( 23531), -INT16_C( 14032), -INT16_C( 13438),  INT16_C( 29023) },
      UINT16_C(58417),
      {  INT16_C( 25867), -INT16_C( 29370), -INT16_C( 25836), -INT16_C( 14807),  INT16_C( 19518),  INT16_C( 28679),  INT16_C(  7378),  INT16_C( 26068),
        -INT16_C( 19512),  INT16_C( 23603),  INT16_C( 18929),  INT16_C(  8449), -INT16_C( 31982),  INT16_C( 29165),  INT16_C(  7924), -INT16_C(   171) },
      {  INT16_C( 25867), -INT16_C( 25979),  INT16_C( 25140), -INT16_C( 20606),  INT16_C( 19518),  INT16_C( 28679),  INT16_C( 16674), -INT16_C( 31183),
         INT16_C( 25621), -INT16_C( 21357),  INT16_C( 18929),  INT16_C( 16020), -INT16_C( 23531),  INT16_C( 29165),  INT16_C(  7924), -INT16_C(   171) } },
    { { -INT16_C( 25725), -INT16_C( 26484), -INT16_C( 18890),  INT16_C( 30046),  INT16_C( 26114), -INT16_C( 11035), -INT16_C( 18046),  INT16_C( 19258),
         INT16_C( 28012),  INT16_C( 23975), -INT16_C( 22346), -INT16_C( 14209),  INT16_C( 27691),  INT16_C(  7993), -INT16_C( 29046),  INT16_C(  3358) },
      UINT16_C(43817),
      {  INT16_C( 24741),  INT16_C(  1121),  INT16_C( 25557), -INT16_C( 17814), -INT16_C(  5065),  INT16_C( 29043), -INT16_C(  8393), -INT16_C(  8225),
        -INT16_C( 27332), -INT16_C( 17529), -INT16_C( 19618), -INT16_C( 26841), -INT16_C( 20014), -INT16_C(  3802),  INT16_C( 20415),  INT16_C( 25756) },
      {  INT16_C( 24741), -INT16_C( 26484), -INT16_C( 18890), -INT16_C( 17814),  INT16_C( 26114),  INT16_C( 29043), -INT16_C( 18046),  INT16_C( 19258),
        -INT16_C( 27332), -INT16_C( 17529), -INT16_C( 22346), -INT16_C( 26841),  INT16_C( 27691), -INT16_C(  3802), -INT16_C( 29046),  INT16_C( 25756) } },
    { { -INT16_C(   593), -INT16_C( 31640), -INT16_C( 11680), -INT16_C( 26818), -INT16_C( 20033), -INT16_C(  2551), -INT16_C(  6000), -INT16_C( 12843),
         INT16_C( 23933), -INT16_C(  9336), -INT16_C( 20464), -INT16_C(  7565), -INT16_C( 26271),  INT16_C(  8403),  INT16_C( 28648), -INT16_C( 26491) },
      UINT16_C(60780),
      { -INT16_C( 13284),  INT16_C( 23488),  INT16_C( 32612),  INT16_C( 27916), -INT16_C( 25227),  INT16_C( 19285), -INT16_C( 11670), -INT16_C(  3416),
        -INT16_C( 18258),  INT16_C(  8610),  INT16_C(  1178),  INT16_C( 28346), -INT16_C( 24028), -INT16_C( 22051),  INT16_C( 19002),  INT16_C( 22423) },
      { -INT16_C(   593), -INT16_C( 31640),  INT16_C( 32612),  INT16_C( 27916), -INT16_C( 20033),  INT16_C( 19285), -INT16_C( 11670), -INT16_C( 12843),
        -INT16_C( 18258), -INT16_C(  9336),  INT16_C(  1178),  INT16_C( 28346), -INT16_C( 26271), -INT16_C( 22051),  INT16_C( 19002),  INT16_C( 22423) } },
    { {  INT16_C( 22294),  INT16_C( 31410), -INT16_C( 16682),  INT16_C( 19431),  INT16_C( 15451), -INT16_C( 14954),  INT16_C( 15887), -INT16_C( 16968),
         INT16_C( 23286), -INT16_C( 28194), -INT16_C( 26530), -INT16_C( 31745), -INT16_C(  9158),  INT16_C( 29996), -INT16_C( 15578),  INT16_C( 15820) },
      UINT16_C(32282),
      { -INT16_C(  3913), -INT16_C( 24772), -INT16_C( 26564), -INT16_C( 11557), -INT16_C(  5539),  INT16_C(  5393),  INT16_C(  1959), -INT16_C( 31376),
        -INT16_C( 12648), -INT16_C( 26851),  INT16_C( 22609),  INT16_C( 32372), -INT16_C( 25907), -INT16_C( 26303),  INT16_C( 23767), -INT16_C( 28905) },
      {  INT16_C( 22294), -INT16_C( 24772), -INT16_C( 16682), -INT16_C( 11557), -INT16_C(  5539), -INT16_C( 14954),  INT16_C( 15887), -INT16_C( 16968),
         INT16_C( 23286), -INT16_C( 26851),  INT16_C( 22609),  INT16_C( 32372), -INT16_C( 25907), -INT16_C( 26303),  INT16_C( 23767),  INT16_C( 15820) } },
    { {  INT16_C( 21324), -INT16_C( 30674),  INT16_C(  2539),  INT16_C( 18779),  INT16_C( 27892), -INT16_C( 25762), -INT16_C( 12685),  INT16_C(  3105),
         INT16_C( 16029), -INT16_C(  4445),  INT16_C(  6038),  INT16_C( 25452), -INT16_C( 20814), -INT16_C( 30212),  INT16_C(  4874),  INT16_C( 22040) },
      UINT16_C(18023),
      {  INT16_C( 21215),  INT16_C( 14928),  INT16_C( 17563), -INT16_C(  1370),  INT16_C(  6623),  INT16_C(   200),  INT16_C( 25893), -INT16_C( 14017),
        -INT16_C( 10924), -INT16_C( 16160), -INT16_C( 28103),  INT16_C( 13678),  INT16_C( 30748),  INT16_C( 13385), -INT16_C( 20273), -INT16_C( 20869) },
      {  INT16_C( 21215),  INT16_C( 14928),  INT16_C( 17563),  INT16_C( 18779),  INT16_C( 27892),  INT16_C(   200),  INT16_C( 25893),  INT16_C(  3105),
         INT16_C( 16029), -INT16_C( 16160), -INT16_C( 28103),  INT16_C( 25452), -INT16_C( 20814), -INT16_C( 30212), -INT16_C( 20273),  INT16_C( 22040) } },
    { { -INT16_C( 13566), -INT16_C( 24856), -INT16_C( 29169), -INT16_C(  4456),  INT16_C( 24743), -INT16_C( 12817),  INT16_C( 11974),  INT16_C(  6806),
         INT16_C( 30211),  INT16_C( 15578),  INT16_C( 18697),  INT16_C(  9586), -INT16_C( 17471), -INT16_C( 28583), -INT16_C( 11157),  INT16_C( 27966) },
      UINT16_C( 9887),
      { -INT16_C( 20981), -INT16_C( 23628),  INT16_C( 23709), -INT16_C( 29692), -INT16_C( 13783), -INT16_C( 16454), -INT16_C( 16924), -INT16_C( 16843),
         INT16_C( 16122),  INT16_C( 27655), -INT16_C( 13981), -INT16_C( 17113), -INT16_C( 28071), -INT16_C( 26479),  INT16_C( 12799),  INT16_C(  3006) },
      { -INT16_C( 20981), -INT16_C( 23628),  INT16_C( 23709), -INT16_C( 29692), -INT16_C( 13783), -INT16_C( 12817),  INT16_C( 11974), -INT16_C( 16843),
         INT16_C( 30211),  INT16_C( 27655), -INT16_C( 13981),  INT16_C(  9586), -INT16_C( 17471), -INT16_C( 26479), -INT16_C( 11157),  INT16_C( 27966) } },
    { {  INT16_C( 29663),  INT16_C( 31918), -INT16_C( 19761), -INT16_C(  2040), -INT16_C( 15748),  INT16_C( 24759), -INT16_C(  4992),  INT16_C( 31263),
         INT16_C(  9771), -INT16_C( 28954),  INT16_C(  3567),  INT16_C( 18763), -INT16_C(  8801), -INT16_C( 24863), -INT16_C( 24818), -INT16_C(  4695) },
      UINT16_C(22546),
      { -INT16_C(  7830),  INT16_C( 29194), -INT16_C( 30759), -INT16_C( 28619), -INT16_C( 18969),  INT16_C(  1661), -INT16_C( 22481),  INT16_C(  5421),
         INT16_C(  7222), -INT16_C( 32222), -INT16_C( 16027),  INT16_C( 18015),  INT16_C( 27999),  INT16_C(  2534), -INT16_C(  1958), -INT16_C( 15263) },
      {  INT16_C( 29663),  INT16_C( 29194), -INT16_C( 19761), -INT16_C(  2040), -INT16_C( 18969),  INT16_C( 24759), -INT16_C(  4992),  INT16_C( 31263),
         INT16_C(  9771), -INT16_C( 28954),  INT16_C(  3567),  INT16_C( 18015),  INT16_C( 27999), -INT16_C( 24863), -INT16_C(  1958), -INT16_C(  4695) } },
    { {  INT16_C( 27610), -INT16_C( 19657),  INT16_C( 27890), -INT16_C(  9660), -INT16_C( 16095),  INT16_C( 20704),  INT16_C(  3433), -INT16_C( 24731),
        -INT16_C( 30934), -INT16_C( 28895), -INT16_C( 32696), -INT16_C( 22570), -INT16_C( 17171),  INT16_C( 18608),  INT16_C(  4532), -INT16_C( 29172) },
      UINT16_C(17277),
      {  INT16_C( 28482), -INT16_C( 31057), -INT16_C( 12215),  INT16_C( 10823), -INT16_C( 20448), -INT16_C( 31433),  INT16_C( 24911),  INT16_C( 28940),
         INT16_C( 21745), -INT16_C( 14351), -INT16_C(  8196), -INT16_C( 21373),  INT16_C( 14119),  INT16_C( 13246),  INT16_C( 15302),  INT16_C(  2167) },
      {  INT16_C( 28482), -INT16_C( 19657), -INT16_C( 12215),  INT16_C( 10823), -INT16_C( 20448), -INT16_C( 31433),  INT16_C( 24911), -INT16_C( 24731),
         INT16_C( 21745), -INT16_C( 14351), -INT16_C( 32696), -INT16_C( 22570), -INT16_C( 17171),  INT16_C( 18608),  INT16_C( 15302), -INT16_C( 29172) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi16(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_loadu_epi16(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_loadu_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i16x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_mask_loadu_epi16(src, k, &a);

    easysimd_test_x86_write_i16x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_loadu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint16_t k;
    const int16_t a[16];
    const int16_t r[16];
  } test_vec[] = {
    { UINT16_C(19194),
      { -INT16_C( 26377), -INT16_C( 25459), -INT16_C( 28956), -INT16_C( 16327),  INT16_C(  4362), -INT16_C( 12246),  INT16_C(   882),  INT16_C( 14010),
        -INT16_C( 22675),  INT16_C( 24299),  INT16_C(  7491), -INT16_C( 24012), -INT16_C( 31956), -INT16_C( 18088),  INT16_C( 21154), -INT16_C( 26108) },
      {  INT16_C(     0), -INT16_C( 25459),  INT16_C(     0), -INT16_C( 16327),  INT16_C(  4362), -INT16_C( 12246),  INT16_C(   882),  INT16_C( 14010),
         INT16_C(     0),  INT16_C( 24299),  INT16_C(     0), -INT16_C( 24012),  INT16_C(     0),  INT16_C(     0),  INT16_C( 21154),  INT16_C(     0) } },
    { UINT16_C(37355),
      { -INT16_C( 12490),  INT16_C( 28703),  INT16_C( 10639), -INT16_C( 18047), -INT16_C(  3078), -INT16_C( 19268),  INT16_C( 10537),  INT16_C(  5211),
        -INT16_C( 24953), -INT16_C( 17615),  INT16_C( 24129), -INT16_C( 26306), -INT16_C(  7913),  INT16_C(  7147), -INT16_C( 10629), -INT16_C( 20051) },
      { -INT16_C( 12490),  INT16_C( 28703),  INT16_C(     0), -INT16_C( 18047),  INT16_C(     0), -INT16_C( 19268),  INT16_C( 10537),  INT16_C(  5211),
        -INT16_C( 24953),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  7913),  INT16_C(     0),  INT16_C(     0), -INT16_C( 20051) } },
    { UINT16_C(52389),
      {  INT16_C( 13345), -INT16_C( 23818), -INT16_C(  3859), -INT16_C( 22123), -INT16_C( 16476),  INT16_C(   210),  INT16_C( 23251),  INT16_C(  1438),
        -INT16_C(  8427),  INT16_C( 21603),  INT16_C( 31352),  INT16_C( 25653), -INT16_C( 20330),  INT16_C( 17210), -INT16_C(  8095), -INT16_C( 31985) },
      {  INT16_C( 13345),  INT16_C(     0), -INT16_C(  3859),  INT16_C(     0),  INT16_C(     0),  INT16_C(   210),  INT16_C(     0),  INT16_C(  1438),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 31352),  INT16_C( 25653),  INT16_C(     0),  INT16_C(     0), -INT16_C(  8095), -INT16_C( 31985) } },
    { UINT16_C( 1300),
      {  INT16_C(   549), -INT16_C( 17419), -INT16_C( 25941),  INT16_C( 32378),  INT16_C( 19866),  INT16_C( 14552), -INT16_C(  4782), -INT16_C( 19176),
        -INT16_C( 28607),  INT16_C( 30256), -INT16_C( 14604),  INT16_C( 12070), -INT16_C( 30711),  INT16_C(  6159),  INT16_C(  8971),  INT16_C( 12318) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 25941),  INT16_C(     0),  INT16_C( 19866),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 28607),  INT16_C(     0), -INT16_C( 14604),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C( 4901),
      { -INT16_C( 11797),  INT16_C( 26029),  INT16_C( 18255),  INT16_C( 10163),  INT16_C(  1408), -INT16_C( 26604),  INT16_C( 22203), -INT16_C(  5336),
         INT16_C(  7628), -INT16_C(  3151), -INT16_C( 17844),  INT16_C( 23419), -INT16_C( 31022), -INT16_C(  3970), -INT16_C( 23370), -INT16_C( 24060) },
      { -INT16_C( 11797),  INT16_C(     0),  INT16_C( 18255),  INT16_C(     0),  INT16_C(     0), -INT16_C( 26604),  INT16_C(     0),  INT16_C(     0),
         INT16_C(  7628), -INT16_C(  3151),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31022),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(45429),
      { -INT16_C( 15353), -INT16_C( 17671),  INT16_C( 31211), -INT16_C(    64),  INT16_C( 31505),  INT16_C( 14677),  INT16_C(  8806),  INT16_C(  5974),
        -INT16_C( 24043), -INT16_C( 28463), -INT16_C( 23555),  INT16_C( 31766), -INT16_C( 13164), -INT16_C( 26592), -INT16_C( 27282),  INT16_C( 30281) },
      { -INT16_C( 15353),  INT16_C(     0),  INT16_C( 31211),  INT16_C(     0),  INT16_C( 31505),  INT16_C( 14677),  INT16_C(  8806),  INT16_C(     0),
        -INT16_C( 24043),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 13164), -INT16_C( 26592),  INT16_C(     0),  INT16_C( 30281) } },
    { UINT16_C(16985),
      {  INT16_C( 17456), -INT16_C(  3909), -INT16_C( 13245), -INT16_C( 26261), -INT16_C( 12026),  INT16_C( 23739), -INT16_C( 12056), -INT16_C( 17921),
        -INT16_C(   928),  INT16_C( 30301), -INT16_C(  3720), -INT16_C( 26558), -INT16_C( 20087), -INT16_C( 11731), -INT16_C( 31193),  INT16_C( 22293) },
      {  INT16_C( 17456),  INT16_C(     0),  INT16_C(     0), -INT16_C( 26261), -INT16_C( 12026),  INT16_C(     0), -INT16_C( 12056),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 30301),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31193),  INT16_C(     0) } },
    { UINT16_C(53450),
      {  INT16_C(  3656), -INT16_C( 19555), -INT16_C( 23641),  INT16_C( 25221),  INT16_C( 28159), -INT16_C(   462), -INT16_C( 28121), -INT16_C( 31493),
         INT16_C( 29448),  INT16_C( 19061), -INT16_C(   500),  INT16_C( 14843),  INT16_C(  8912), -INT16_C(  6720), -INT16_C( 30086), -INT16_C( 15690) },
      {  INT16_C(     0), -INT16_C( 19555),  INT16_C(     0),  INT16_C( 25221),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28121), -INT16_C( 31493),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8912),  INT16_C(     0), -INT16_C( 30086), -INT16_C( 15690) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_loadu_epi16(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_loadu_epi16");
    easysimd_test_x86_assert_equal_i16x16(r, easysimd_mm256_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i r = easysimd_mm256_maskz_loadu_epi16(k, &a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_loadu_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int32_t src[8];
    const uint8_t k;
    const int32_t a[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(   192010582),  INT32_C(   387896823), -INT32_C(   727886459), -INT32_C(  1471829577), -INT32_C(    93374466),  INT32_C(  1034351126), -INT32_C(   277846999),  INT32_C(  1257760671) },
      UINT8_C( 14),
      {  INT32_C(  1510293125), -INT32_C(  1310778276),  INT32_C(  1667806393),  INT32_C(   811667705),  INT32_C(  1900436351),  INT32_C(  1721467649), -INT32_C(   620393740),  INT32_C(   132730754) },
      { -INT32_C(   192010582), -INT32_C(  1310778276),  INT32_C(  1667806393),  INT32_C(   811667705), -INT32_C(    93374466),  INT32_C(  1034351126), -INT32_C(   277846999),  INT32_C(  1257760671) } },
    { { -INT32_C(   379392371), -INT32_C(   996458229), -INT32_C(   282655755), -INT32_C(  1826650092), -INT32_C(   452630813), -INT32_C(   582246167), -INT32_C(  1397141462),  INT32_C(   766812832) },
      UINT8_C(145),
      {  INT32_C(  1469847318), -INT32_C(  1253220174),  INT32_C(   281623688), -INT32_C(  1040949925),  INT32_C(    44816738),  INT32_C(  1949075492), -INT32_C(   468395711), -INT32_C(  1552596339) },
      {  INT32_C(  1469847318), -INT32_C(   996458229), -INT32_C(   282655755), -INT32_C(  1826650092),  INT32_C(    44816738), -INT32_C(   582246167), -INT32_C(  1397141462), -INT32_C(  1552596339) } },
    { {  INT32_C(   200937817), -INT32_C(   104839311), -INT32_C(   552957309),  INT32_C(  1235287783), -INT32_C(    78951465),  INT32_C(   359626708), -INT32_C(   570850224),  INT32_C(   528510662) },
      UINT8_C(127),
      { -INT32_C(  1024382341),  INT32_C(  1950804714), -INT32_C(   228907532),  INT32_C(   298427589),  INT32_C(  1743111407), -INT32_C(  1195902412),  INT32_C(  1652463092), -INT32_C(  1864196843) },
      { -INT32_C(  1024382341),  INT32_C(  1950804714), -INT32_C(   228907532),  INT32_C(   298427589),  INT32_C(  1743111407), -INT32_C(  1195902412),  INT32_C(  1652463092),  INT32_C(   528510662) } },
    { { -INT32_C(  1319906361), -INT32_C(  1306093123), -INT32_C(  2086370882),  INT32_C(   362049062),  INT32_C(  1719433522),  INT32_C(  1746809972), -INT32_C(   557146935),  INT32_C(     7318585) },
      UINT8_C(127),
      {  INT32_C(  1530770114),  INT32_C(  1494872024),  INT32_C(    25140371), -INT32_C(  1439394511),  INT32_C(  1159633425),  INT32_C(  1427015353), -INT32_C(    40899247),  INT32_C(   511545180) },
      {  INT32_C(  1530770114),  INT32_C(  1494872024),  INT32_C(    25140371), -INT32_C(  1439394511),  INT32_C(  1159633425),  INT32_C(  1427015353), -INT32_C(    40899247),  INT32_C(     7318585) } },
    { {  INT32_C(   427407937),  INT32_C(  1014207145),  INT32_C(  1597960750), -INT32_C(  1727368569), -INT32_C(   975296500),  INT32_C(     1830319),  INT32_C(   922594010),  INT32_C(  2069133881) },
      UINT8_C( 52),
      {  INT32_C(  1608357069), -INT32_C(    91416057), -INT32_C(   897454760),  INT32_C(   534125303), -INT32_C(   422667015),  INT32_C(  1640025783),  INT32_C(  1184560844),  INT32_C(   393942346) },
      {  INT32_C(   427407937),  INT32_C(  1014207145), -INT32_C(   897454760), -INT32_C(  1727368569), -INT32_C(   422667015),  INT32_C(  1640025783),  INT32_C(   922594010),  INT32_C(  2069133881) } },
    { { -INT32_C(  1317644118), -INT32_C(   894762126), -INT32_C(   409719312),  INT32_C(  1074228039), -INT32_C(  1104751353),  INT32_C(  1881138852),  INT32_C(   649509340),  INT32_C(  2034053583) },
      UINT8_C(138),
      { -INT32_C(  1208210765),  INT32_C(    61327318), -INT32_C(   968192165),  INT32_C(  1808632726), -INT32_C(  1743811663),  INT32_C(  1685356458),  INT32_C(  1731435318), -INT32_C(  1930318632) },
      { -INT32_C(  1317644118),  INT32_C(    61327318), -INT32_C(   409719312),  INT32_C(  1808632726), -INT32_C(  1104751353),  INT32_C(  1881138852),  INT32_C(   649509340), -INT32_C(  1930318632) } },
    { { -INT32_C(  1404834090),  INT32_C(   279964341),  INT32_C(   265812601),  INT32_C(   914072709), -INT32_C(   623998416),  INT32_C(  1077822218), -INT32_C(  1230540322), -INT32_C(   196962019) },
      UINT8_C(135),
      {  INT32_C(  1883021445),  INT32_C(  1256803408), -INT32_C(   925894365), -INT32_C(    17300108),  INT32_C(   386454228), -INT32_C(  2097854447), -INT32_C(  1985958928),  INT32_C(  1930466542) },
      {  INT32_C(  1883021445),  INT32_C(  1256803408), -INT32_C(   925894365),  INT32_C(   914072709), -INT32_C(   623998416),  INT32_C(  1077822218), -INT32_C(  1230540322),  INT32_C(  1930466542) } },
    { { -INT32_C(  2065478604), -INT32_C(  1127297640),  INT32_C(   981769670),  INT32_C(  2000190627),  INT32_C(  1603158350),  INT32_C(  2044887945),  INT32_C(   469926446),  INT32_C(  1250955798) },
      UINT8_C( 94),
      {  INT32_C(  1089982323),  INT32_C(   990294941), -INT32_C(  1277280201), -INT32_C(  1174317703),  INT32_C(  1715692003), -INT32_C(   980107965), -INT32_C(   757354049), -INT32_C(  1271913151) },
      { -INT32_C(  2065478604),  INT32_C(   990294941), -INT32_C(  1277280201), -INT32_C(  1174317703),  INT32_C(  1715692003),  INT32_C(  2044887945), -INT32_C(   757354049),  INT32_C(  1250955798) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_loadu_epi32(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_loadu_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_mask_loadu_epi32(src, k, &a);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_loadu_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const int32_t a[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(152),
      { -INT32_C(   163613357),  INT32_C(  1760928250),  INT32_C(  1703933139),  INT32_C(  1651315695), -INT32_C(  2039564152), -INT32_C(   698963789), -INT32_C(   229622680),  INT32_C(  1166742258) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1651315695), -INT32_C(  2039564152),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1166742258) } },
    { UINT8_C(135),
      {  INT32_C(  1820474314),  INT32_C(   608168496), -INT32_C(  1944869511), -INT32_C(   904628718), -INT32_C(  1937859868),  INT32_C(   754210032), -INT32_C(  1239488604),  INT32_C(  1010656114) },
      {  INT32_C(  1820474314),  INT32_C(   608168496), -INT32_C(  1944869511),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1010656114) } },
    { UINT8_C(158),
      { -INT32_C(  1446008641), -INT32_C(  1927089176),  INT32_C(  2107616775),  INT32_C(  1549888194),  INT32_C(  1011740392), -INT32_C(   908035614), -INT32_C(    79980904), -INT32_C(  1818593069) },
      {  INT32_C(           0), -INT32_C(  1927089176),  INT32_C(  2107616775),  INT32_C(  1549888194),  INT32_C(  1011740392),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1818593069) } },
    { UINT8_C( 32),
      {  INT32_C(  1544043625),  INT32_C(   224630111), -INT32_C(  1613701067),  INT32_C(   814165058), -INT32_C(   216874119),  INT32_C(   965467299), -INT32_C(  1895004649), -INT32_C(  1984913632) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   965467299),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(220),
      {  INT32_C(  1312548536),  INT32_C(   713246793),  INT32_C(  1164714520),  INT32_C(  1824431273), -INT32_C(  1978682962), -INT32_C(  1012774852), -INT32_C(   186437292), -INT32_C(  1697616415) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(  1164714520),  INT32_C(  1824431273), -INT32_C(  1978682962),  INT32_C(           0), -INT32_C(   186437292), -INT32_C(  1697616415) } },
    { UINT8_C( 83),
      {  INT32_C(  1402791947), -INT32_C(  1922251157), -INT32_C(   835276494),  INT32_C(   561881711), -INT32_C(   111343695), -INT32_C(   615702359),  INT32_C(  1908162820),  INT32_C(   482629137) },
      {  INT32_C(  1402791947), -INT32_C(  1922251157),  INT32_C(           0),  INT32_C(           0), -INT32_C(   111343695),  INT32_C(           0),  INT32_C(  1908162820),  INT32_C(           0) } },
    { UINT8_C( 62),
      {  INT32_C(   665415776), -INT32_C(  1923533092),  INT32_C(   251406444), -INT32_C(  1396761179), -INT32_C(  1688815493),  INT32_C(  1201615110),  INT32_C(  1146687725),  INT32_C(   897742292) },
      {  INT32_C(           0), -INT32_C(  1923533092),  INT32_C(   251406444), -INT32_C(  1396761179), -INT32_C(  1688815493),  INT32_C(  1201615110),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(229),
      {  INT32_C(  1656839212), -INT32_C(   573616459),  INT32_C(  1753406795),  INT32_C(  1440952221), -INT32_C(  1235517563),  INT32_C(   799253278),  INT32_C(  1896081404), -INT32_C(  1772668822) },
      {  INT32_C(  1656839212),  INT32_C(           0),  INT32_C(  1753406795),  INT32_C(           0),  INT32_C(           0),  INT32_C(   799253278),  INT32_C(  1896081404), -INT32_C(  1772668822) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_loadu_epi32(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_loadu_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_loadu_epi32(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_loadu_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const int64_t src[4];
    const uint8_t k;
    const int64_t a[4];
    const int64_t r[4];
  } test_vec[] = {
    { {  INT64_C( 1282958385057441780), -INT64_C( 2058208798333162692),  INT64_C( 3400179714282336807),  INT64_C( 2631282163697388174) },
      UINT8_C( 89),
      {  INT64_C( 3364265352538338937), -INT64_C( 8649706428611795703), -INT64_C( 7366158843468423078), -INT64_C( 7876253580174715610) },
      {  INT64_C( 3364265352538338937), -INT64_C( 2058208798333162692),  INT64_C( 3400179714282336807), -INT64_C( 7876253580174715610) } },
    { {  INT64_C( 3698782581337613475), -INT64_C( 1335042875716369891), -INT64_C( 4935124316357103783), -INT64_C( 2653530333062673536) },
      UINT8_C(168),
      { -INT64_C( 2952385164148080333), -INT64_C( 2280414118459803385),  INT64_C( 1289771967673877372), -INT64_C( 2328601061540396989) },
      {  INT64_C( 3698782581337613475), -INT64_C( 1335042875716369891), -INT64_C( 4935124316357103783), -INT64_C( 2328601061540396989) } },
    { { -INT64_C( 8190972921707396826), -INT64_C( 7990256460970934582),  INT64_C(  289038250674786603), -INT64_C( 7500258714041234594) },
      UINT8_C( 60),
      {  INT64_C( 2616116834793034675),  INT64_C( 3959732410261342687),  INT64_C( 3531099075853673667), -INT64_C( 6341848976884364538) },
      { -INT64_C( 8190972921707396826), -INT64_C( 7990256460970934582),  INT64_C( 3531099075853673667), -INT64_C( 6341848976884364538) } },
    { { -INT64_C( 1164328866036334772), -INT64_C( 6854573342646202109), -INT64_C( 2835786918715533683), -INT64_C( 3681245742781348019) },
      UINT8_C( 86),
      {  INT64_C( 7390782289441357333), -INT64_C( 1206152050627102197), -INT64_C( 6722297151407147483), -INT64_C(  783363685931092173) },
      { -INT64_C( 1164328866036334772), -INT64_C( 1206152050627102197), -INT64_C( 6722297151407147483), -INT64_C( 3681245742781348019) } },
    { { -INT64_C( 1862214412312213908),  INT64_C( 8311637540656081743), -INT64_C( 1760466876827493996),  INT64_C(  963156386486793095) },
      UINT8_C( 87),
      {  INT64_C( 7153432515686092769),  INT64_C( 2316579768489914036),  INT64_C( 6419805018397021440), -INT64_C( 4786473336467456149) },
      {  INT64_C( 7153432515686092769),  INT64_C( 2316579768489914036),  INT64_C( 6419805018397021440),  INT64_C(  963156386486793095) } },
    { { -INT64_C( 7911986859973818921),  INT64_C(  778641195313456296), -INT64_C( 4008570510012341665),  INT64_C( 3948135557351364657) },
      UINT8_C( 93),
      {  INT64_C( 2026563999742231934),  INT64_C( 4510972523842031345),  INT64_C( 6174545029526953932),  INT64_C(  458781710699838823) },
      {  INT64_C( 2026563999742231934),  INT64_C(  778641195313456296),  INT64_C( 6174545029526953932),  INT64_C(  458781710699838823) } },
    { {  INT64_C( 6288605516801087729),  INT64_C( 7585800774812396651), -INT64_C( 6115721972251369011), -INT64_C( 5628645484837218987) },
      UINT8_C(160),
      { -INT64_C( 8329304015044683168), -INT64_C( 7499739443567693583), -INT64_C( 6096036791618652551), -INT64_C( 8729632877694991567) },
      {  INT64_C( 6288605516801087729),  INT64_C( 7585800774812396651), -INT64_C( 6115721972251369011), -INT64_C( 5628645484837218987) } },
    { {  INT64_C( 3432847092413161719), -INT64_C( 2834671414256153508),  INT64_C( 4355067751650919663), -INT64_C( 1135381022578910062) },
      UINT8_C( 48),
      {  INT64_C( 8650181969515055298),  INT64_C( 5482480291766738668),  INT64_C( 2844185118880816023),  INT64_C( 3569684824578863113) },
      {  INT64_C( 3432847092413161719), -INT64_C( 2834671414256153508),  INT64_C( 4355067751650919663), -INT64_C( 1135381022578910062) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_loadu_epi64(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_loadu_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_mask_loadu_epi64(src, k, &a);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_loadu_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const int64_t a[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(148),
      { -INT64_C( 6507094774212003816),  INT64_C( 9033825023374269098), -INT64_C( 6948268385224682253),  INT64_C( 8330427144748991438) },
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 6948268385224682253),  INT64_C(                   0) } },
    { UINT8_C(151),
      { -INT64_C( 6433950018102633755),  INT64_C(  130641694703259462), -INT64_C(  264207178679097667), -INT64_C( 2551348601182331807) },
      { -INT64_C( 6433950018102633755),  INT64_C(  130641694703259462), -INT64_C(  264207178679097667),  INT64_C(                   0) } },
    { UINT8_C(163),
      {  INT64_C( 3332899706839754998),  INT64_C( 6149979676809633402), -INT64_C( 7886003925448531754),  INT64_C( 2093440621745901472) },
      {  INT64_C( 3332899706839754998),  INT64_C( 6149979676809633402),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 77),
      {  INT64_C( 8542514133280233532),  INT64_C( 7073952708639926583),  INT64_C( 8714793935523305735), -INT64_C( 6600379953372505421) },
      {  INT64_C( 8542514133280233532),  INT64_C(                   0),  INT64_C( 8714793935523305735), -INT64_C( 6600379953372505421) } },
    { UINT8_C(213),
      {  INT64_C( 7828562661778241106),  INT64_C( 9064146418384435465),  INT64_C( 7478190949246835689), -INT64_C( 7489360534340444034) },
      {  INT64_C( 7828562661778241106),  INT64_C(                   0),  INT64_C( 7478190949246835689),  INT64_C(                   0) } },
    { UINT8_C( 41),
      { -INT64_C( 6634970298033895874), -INT64_C( 7781580824028664121), -INT64_C( 4825316877655861589), -INT64_C( 7822299187671513898) },
      { -INT64_C( 6634970298033895874),  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 7822299187671513898) } },
    { UINT8_C(162),
      {  INT64_C( 1044536461779301854), -INT64_C( 5789348072315775049), -INT64_C( 9180506219867262277),  INT64_C( 3498004240903037261) },
      {  INT64_C(                   0), -INT64_C( 5789348072315775049),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(221),
      { -INT64_C(  180342555253282856),  INT64_C( 6244035433262906577), -INT64_C( 2060095554283476527), -INT64_C( 5319763954833730841) },
      { -INT64_C(  180342555253282856),  INT64_C(                   0), -INT64_C( 2060095554283476527), -INT64_C( 5319763954833730841) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_loadu_epi64(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_loadu_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_loadu_epi64(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_loadu_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   758.89), EASYSIMD_FLOAT32_C(   893.95), EASYSIMD_FLOAT32_C(   103.27), EASYSIMD_FLOAT32_C(   -29.32),
        EASYSIMD_FLOAT32_C(  -712.94), EASYSIMD_FLOAT32_C(  -386.07), EASYSIMD_FLOAT32_C(   935.48), EASYSIMD_FLOAT32_C(  -891.48) },
      UINT8_C( 85),
      { EASYSIMD_FLOAT32_C(   754.91), EASYSIMD_FLOAT32_C(   855.49), EASYSIMD_FLOAT32_C(  -868.92), EASYSIMD_FLOAT32_C(   -81.61),
        EASYSIMD_FLOAT32_C(   479.56), EASYSIMD_FLOAT32_C(   464.78), EASYSIMD_FLOAT32_C(   494.57), EASYSIMD_FLOAT32_C(  -181.79) },
      { EASYSIMD_FLOAT32_C(   754.91), EASYSIMD_FLOAT32_C(   893.95), EASYSIMD_FLOAT32_C(  -868.92), EASYSIMD_FLOAT32_C(   -29.32),
        EASYSIMD_FLOAT32_C(   479.56), EASYSIMD_FLOAT32_C(  -386.07), EASYSIMD_FLOAT32_C(   494.57), EASYSIMD_FLOAT32_C(  -891.48) } },
    { { EASYSIMD_FLOAT32_C(   213.57), EASYSIMD_FLOAT32_C(   932.15), EASYSIMD_FLOAT32_C(  -775.18), EASYSIMD_FLOAT32_C(   423.06),
        EASYSIMD_FLOAT32_C(  -828.82), EASYSIMD_FLOAT32_C(  -717.48), EASYSIMD_FLOAT32_C(  -147.09), EASYSIMD_FLOAT32_C(   292.26) },
      UINT8_C( 19),
      { EASYSIMD_FLOAT32_C(  -836.54), EASYSIMD_FLOAT32_C(   773.35), EASYSIMD_FLOAT32_C(  -294.72), EASYSIMD_FLOAT32_C(   -91.53),
        EASYSIMD_FLOAT32_C(  -234.74), EASYSIMD_FLOAT32_C(  -535.83), EASYSIMD_FLOAT32_C(  -197.58), EASYSIMD_FLOAT32_C(   868.53) },
      { EASYSIMD_FLOAT32_C(  -836.54), EASYSIMD_FLOAT32_C(   773.35), EASYSIMD_FLOAT32_C(  -775.18), EASYSIMD_FLOAT32_C(   423.06),
        EASYSIMD_FLOAT32_C(  -234.74), EASYSIMD_FLOAT32_C(  -717.48), EASYSIMD_FLOAT32_C(  -147.09), EASYSIMD_FLOAT32_C(   292.26) } },
    { { EASYSIMD_FLOAT32_C(   434.84), EASYSIMD_FLOAT32_C(    89.48), EASYSIMD_FLOAT32_C(  -517.54), EASYSIMD_FLOAT32_C(   370.33),
        EASYSIMD_FLOAT32_C(   198.00), EASYSIMD_FLOAT32_C(  -976.91), EASYSIMD_FLOAT32_C(   125.24), EASYSIMD_FLOAT32_C(    53.49) },
      UINT8_C(124),
      { EASYSIMD_FLOAT32_C(  -956.37), EASYSIMD_FLOAT32_C(  -466.94), EASYSIMD_FLOAT32_C(   618.95), EASYSIMD_FLOAT32_C(   538.20),
        EASYSIMD_FLOAT32_C(   351.27), EASYSIMD_FLOAT32_C(  -167.48), EASYSIMD_FLOAT32_C(   470.35), EASYSIMD_FLOAT32_C(   576.08) },
      { EASYSIMD_FLOAT32_C(   434.84), EASYSIMD_FLOAT32_C(    89.48), EASYSIMD_FLOAT32_C(   618.95), EASYSIMD_FLOAT32_C(   538.20),
        EASYSIMD_FLOAT32_C(   351.27), EASYSIMD_FLOAT32_C(  -167.48), EASYSIMD_FLOAT32_C(   470.35), EASYSIMD_FLOAT32_C(    53.49) } },
    { { EASYSIMD_FLOAT32_C(  -744.42), EASYSIMD_FLOAT32_C(   641.53), EASYSIMD_FLOAT32_C(   858.60), EASYSIMD_FLOAT32_C(   108.49),
        EASYSIMD_FLOAT32_C(   -66.20), EASYSIMD_FLOAT32_C(   335.77), EASYSIMD_FLOAT32_C(   271.95), EASYSIMD_FLOAT32_C(  -292.85) },
      UINT8_C( 93),
      { EASYSIMD_FLOAT32_C(  -819.58), EASYSIMD_FLOAT32_C(   472.40), EASYSIMD_FLOAT32_C(  -494.78), EASYSIMD_FLOAT32_C(   -17.16),
        EASYSIMD_FLOAT32_C(   340.93), EASYSIMD_FLOAT32_C(   940.06), EASYSIMD_FLOAT32_C(  -927.68), EASYSIMD_FLOAT32_C(   823.40) },
      { EASYSIMD_FLOAT32_C(  -819.58), EASYSIMD_FLOAT32_C(   641.53), EASYSIMD_FLOAT32_C(  -494.78), EASYSIMD_FLOAT32_C(   -17.16),
        EASYSIMD_FLOAT32_C(   340.93), EASYSIMD_FLOAT32_C(   335.77), EASYSIMD_FLOAT32_C(  -927.68), EASYSIMD_FLOAT32_C(  -292.85) } },
    { { EASYSIMD_FLOAT32_C(   310.39), EASYSIMD_FLOAT32_C(   270.32), EASYSIMD_FLOAT32_C(   846.49), EASYSIMD_FLOAT32_C(  -564.37),
        EASYSIMD_FLOAT32_C(  -676.19), EASYSIMD_FLOAT32_C(  -999.34), EASYSIMD_FLOAT32_C(  -520.74), EASYSIMD_FLOAT32_C(  -143.14) },
      UINT8_C( 51),
      { EASYSIMD_FLOAT32_C(  -982.54), EASYSIMD_FLOAT32_C(  -791.87), EASYSIMD_FLOAT32_C(  -547.88), EASYSIMD_FLOAT32_C(   487.81),
        EASYSIMD_FLOAT32_C(   784.21), EASYSIMD_FLOAT32_C(  -292.30), EASYSIMD_FLOAT32_C(   129.35), EASYSIMD_FLOAT32_C(   642.81) },
      { EASYSIMD_FLOAT32_C(  -982.54), EASYSIMD_FLOAT32_C(  -791.87), EASYSIMD_FLOAT32_C(   846.49), EASYSIMD_FLOAT32_C(  -564.37),
        EASYSIMD_FLOAT32_C(   784.21), EASYSIMD_FLOAT32_C(  -292.30), EASYSIMD_FLOAT32_C(  -520.74), EASYSIMD_FLOAT32_C(  -143.14) } },
    { { EASYSIMD_FLOAT32_C(   816.20), EASYSIMD_FLOAT32_C(  -936.86), EASYSIMD_FLOAT32_C(   -21.41), EASYSIMD_FLOAT32_C(    88.15),
        EASYSIMD_FLOAT32_C(  -229.71), EASYSIMD_FLOAT32_C(    19.64), EASYSIMD_FLOAT32_C(   268.56), EASYSIMD_FLOAT32_C(  -757.31) },
      UINT8_C( 40),
      { EASYSIMD_FLOAT32_C(  -748.60), EASYSIMD_FLOAT32_C(   583.63), EASYSIMD_FLOAT32_C(   464.92), EASYSIMD_FLOAT32_C(  -676.28),
        EASYSIMD_FLOAT32_C(   407.02), EASYSIMD_FLOAT32_C(  -224.69), EASYSIMD_FLOAT32_C(   594.03), EASYSIMD_FLOAT32_C(   253.51) },
      { EASYSIMD_FLOAT32_C(   816.20), EASYSIMD_FLOAT32_C(  -936.86), EASYSIMD_FLOAT32_C(   -21.41), EASYSIMD_FLOAT32_C(  -676.28),
        EASYSIMD_FLOAT32_C(  -229.71), EASYSIMD_FLOAT32_C(  -224.69), EASYSIMD_FLOAT32_C(   268.56), EASYSIMD_FLOAT32_C(  -757.31) } },
    { { EASYSIMD_FLOAT32_C(   210.94), EASYSIMD_FLOAT32_C(   917.84), EASYSIMD_FLOAT32_C(   254.17), EASYSIMD_FLOAT32_C(   690.21),
        EASYSIMD_FLOAT32_C(  -225.30), EASYSIMD_FLOAT32_C(  -126.23), EASYSIMD_FLOAT32_C(   707.67), EASYSIMD_FLOAT32_C(   -17.17) },
      UINT8_C(208),
      { EASYSIMD_FLOAT32_C(   195.49), EASYSIMD_FLOAT32_C(  -232.96), EASYSIMD_FLOAT32_C(  -966.40), EASYSIMD_FLOAT32_C(  -675.17),
        EASYSIMD_FLOAT32_C(  -590.14), EASYSIMD_FLOAT32_C(   849.79), EASYSIMD_FLOAT32_C(  -612.02), EASYSIMD_FLOAT32_C(   388.44) },
      { EASYSIMD_FLOAT32_C(   210.94), EASYSIMD_FLOAT32_C(   917.84), EASYSIMD_FLOAT32_C(   254.17), EASYSIMD_FLOAT32_C(   690.21),
        EASYSIMD_FLOAT32_C(  -590.14), EASYSIMD_FLOAT32_C(  -126.23), EASYSIMD_FLOAT32_C(  -612.02), EASYSIMD_FLOAT32_C(   388.44) } },
    { { EASYSIMD_FLOAT32_C(   -62.06), EASYSIMD_FLOAT32_C(   158.27), EASYSIMD_FLOAT32_C(  -591.92), EASYSIMD_FLOAT32_C(  -793.50),
        EASYSIMD_FLOAT32_C(   400.96), EASYSIMD_FLOAT32_C(   932.94), EASYSIMD_FLOAT32_C(  -542.09), EASYSIMD_FLOAT32_C(   -15.41) },
      UINT8_C(232),
      { EASYSIMD_FLOAT32_C(  -218.38), EASYSIMD_FLOAT32_C(  -608.39), EASYSIMD_FLOAT32_C(  -826.82), EASYSIMD_FLOAT32_C(  -624.35),
        EASYSIMD_FLOAT32_C(   645.12), EASYSIMD_FLOAT32_C(   384.12), EASYSIMD_FLOAT32_C(  -706.50), EASYSIMD_FLOAT32_C(  -100.71) },
      { EASYSIMD_FLOAT32_C(   -62.06), EASYSIMD_FLOAT32_C(   158.27), EASYSIMD_FLOAT32_C(  -591.92), EASYSIMD_FLOAT32_C(  -624.35),
        EASYSIMD_FLOAT32_C(   400.96), EASYSIMD_FLOAT32_C(   384.12), EASYSIMD_FLOAT32_C(  -706.50), EASYSIMD_FLOAT32_C(  -100.71) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_loadu_ps(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_loadu_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_loadu_ps(src, k, &a);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_loadu_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const easysimd_float32 a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C( 95),
      { EASYSIMD_FLOAT32_C(  -741.86), EASYSIMD_FLOAT32_C(   705.91), EASYSIMD_FLOAT32_C(   372.13), EASYSIMD_FLOAT32_C(   312.47),
        EASYSIMD_FLOAT32_C(  -319.15), EASYSIMD_FLOAT32_C(   957.05), EASYSIMD_FLOAT32_C(  -486.31), EASYSIMD_FLOAT32_C(   945.97) },
      { EASYSIMD_FLOAT32_C(  -741.86), EASYSIMD_FLOAT32_C(   705.91), EASYSIMD_FLOAT32_C(   372.13), EASYSIMD_FLOAT32_C(   312.47),
        EASYSIMD_FLOAT32_C(  -319.15), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -486.31), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(163),
      { EASYSIMD_FLOAT32_C(     1.21), EASYSIMD_FLOAT32_C(  -796.49), EASYSIMD_FLOAT32_C(   -53.69), EASYSIMD_FLOAT32_C(   408.58),
        EASYSIMD_FLOAT32_C(   983.44), EASYSIMD_FLOAT32_C(   798.69), EASYSIMD_FLOAT32_C(   709.52), EASYSIMD_FLOAT32_C(  -634.73) },
      { EASYSIMD_FLOAT32_C(     1.21), EASYSIMD_FLOAT32_C(  -796.49), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   798.69), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -634.73) } },
    { UINT8_C(201),
      { EASYSIMD_FLOAT32_C(  -407.02), EASYSIMD_FLOAT32_C(  -419.52), EASYSIMD_FLOAT32_C(  -136.27), EASYSIMD_FLOAT32_C(  -273.01),
        EASYSIMD_FLOAT32_C(  -625.61), EASYSIMD_FLOAT32_C(  -175.04), EASYSIMD_FLOAT32_C(   571.06), EASYSIMD_FLOAT32_C(   -72.87) },
      { EASYSIMD_FLOAT32_C(  -407.02), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -273.01),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   571.06), EASYSIMD_FLOAT32_C(   -72.87) } },
    { UINT8_C(240),
      { EASYSIMD_FLOAT32_C(  -459.75), EASYSIMD_FLOAT32_C(  -381.02), EASYSIMD_FLOAT32_C(  -722.71), EASYSIMD_FLOAT32_C(  -943.66),
        EASYSIMD_FLOAT32_C(  -122.88), EASYSIMD_FLOAT32_C(   983.20), EASYSIMD_FLOAT32_C(   428.46), EASYSIMD_FLOAT32_C(  -810.41) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -122.88), EASYSIMD_FLOAT32_C(   983.20), EASYSIMD_FLOAT32_C(   428.46), EASYSIMD_FLOAT32_C(  -810.41) } },
    { UINT8_C(148),
      { EASYSIMD_FLOAT32_C(   385.51), EASYSIMD_FLOAT32_C(  -296.72), EASYSIMD_FLOAT32_C(  -389.98), EASYSIMD_FLOAT32_C(  -556.09),
        EASYSIMD_FLOAT32_C(   704.49), EASYSIMD_FLOAT32_C(  -186.47), EASYSIMD_FLOAT32_C(   390.22), EASYSIMD_FLOAT32_C(   113.07) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -389.98), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   704.49), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   113.07) } },
    { UINT8_C( 60),
      { EASYSIMD_FLOAT32_C(   188.91), EASYSIMD_FLOAT32_C(  -177.40), EASYSIMD_FLOAT32_C(   162.24), EASYSIMD_FLOAT32_C(  -549.96),
        EASYSIMD_FLOAT32_C(   415.58), EASYSIMD_FLOAT32_C(   742.72), EASYSIMD_FLOAT32_C(   313.77), EASYSIMD_FLOAT32_C(  -857.44) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   162.24), EASYSIMD_FLOAT32_C(  -549.96),
        EASYSIMD_FLOAT32_C(   415.58), EASYSIMD_FLOAT32_C(   742.72), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(204),
      { EASYSIMD_FLOAT32_C(  -861.27), EASYSIMD_FLOAT32_C(   713.62), EASYSIMD_FLOAT32_C(    44.24), EASYSIMD_FLOAT32_C(  -954.76),
        EASYSIMD_FLOAT32_C(  -746.13), EASYSIMD_FLOAT32_C(   663.22), EASYSIMD_FLOAT32_C(  -677.47), EASYSIMD_FLOAT32_C(  -689.80) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    44.24), EASYSIMD_FLOAT32_C(  -954.76),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -677.47), EASYSIMD_FLOAT32_C(  -689.80) } },
    { UINT8_C(154),
      { EASYSIMD_FLOAT32_C(  -694.26), EASYSIMD_FLOAT32_C(   738.67), EASYSIMD_FLOAT32_C(  -270.08), EASYSIMD_FLOAT32_C(   -30.21),
        EASYSIMD_FLOAT32_C(   124.18), EASYSIMD_FLOAT32_C(   433.21), EASYSIMD_FLOAT32_C(   579.81), EASYSIMD_FLOAT32_C(   568.09) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   738.67), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -30.21),
        EASYSIMD_FLOAT32_C(   124.18), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   568.09) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_loadu_ps(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_loadu_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_loadu_ps(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_loadu_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float64 src[4];
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   294.47), EASYSIMD_FLOAT64_C(  -657.32), EASYSIMD_FLOAT64_C(  -923.80), EASYSIMD_FLOAT64_C(    -5.44) },
      UINT8_C( 38),
      { EASYSIMD_FLOAT64_C(   867.66), EASYSIMD_FLOAT64_C(  -569.11), EASYSIMD_FLOAT64_C(  -505.69), EASYSIMD_FLOAT64_C(  -859.48) },
      { EASYSIMD_FLOAT64_C(   294.47), EASYSIMD_FLOAT64_C(  -569.11), EASYSIMD_FLOAT64_C(  -505.69), EASYSIMD_FLOAT64_C(    -5.44) } },
    { { EASYSIMD_FLOAT64_C(   -75.37), EASYSIMD_FLOAT64_C(  -589.90), EASYSIMD_FLOAT64_C(   584.15), EASYSIMD_FLOAT64_C(   654.21) },
      UINT8_C(232),
      { EASYSIMD_FLOAT64_C(   949.58), EASYSIMD_FLOAT64_C(  -407.82), EASYSIMD_FLOAT64_C(  -644.48), EASYSIMD_FLOAT64_C(   212.99) },
      { EASYSIMD_FLOAT64_C(   -75.37), EASYSIMD_FLOAT64_C(  -589.90), EASYSIMD_FLOAT64_C(   584.15), EASYSIMD_FLOAT64_C(   212.99) } },
    { { EASYSIMD_FLOAT64_C(  -886.74), EASYSIMD_FLOAT64_C(  -832.13), EASYSIMD_FLOAT64_C(   -73.34), EASYSIMD_FLOAT64_C(   211.61) },
      UINT8_C(101),
      { EASYSIMD_FLOAT64_C(  -467.07), EASYSIMD_FLOAT64_C(  -494.51), EASYSIMD_FLOAT64_C(  -905.82), EASYSIMD_FLOAT64_C(  -316.43) },
      { EASYSIMD_FLOAT64_C(  -467.07), EASYSIMD_FLOAT64_C(  -832.13), EASYSIMD_FLOAT64_C(  -905.82), EASYSIMD_FLOAT64_C(   211.61) } },
    { { EASYSIMD_FLOAT64_C(   822.89), EASYSIMD_FLOAT64_C(   -66.11), EASYSIMD_FLOAT64_C(  -239.08), EASYSIMD_FLOAT64_C(   764.98) },
      UINT8_C(224),
      { EASYSIMD_FLOAT64_C(   103.60), EASYSIMD_FLOAT64_C(   841.18), EASYSIMD_FLOAT64_C(   222.92), EASYSIMD_FLOAT64_C(   537.64) },
      { EASYSIMD_FLOAT64_C(   822.89), EASYSIMD_FLOAT64_C(   -66.11), EASYSIMD_FLOAT64_C(  -239.08), EASYSIMD_FLOAT64_C(   764.98) } },
    { { EASYSIMD_FLOAT64_C(   708.84), EASYSIMD_FLOAT64_C(   653.81), EASYSIMD_FLOAT64_C(  -968.05), EASYSIMD_FLOAT64_C(   849.36) },
      UINT8_C(144),
      { EASYSIMD_FLOAT64_C(  -557.95), EASYSIMD_FLOAT64_C(   433.52), EASYSIMD_FLOAT64_C(  -767.36), EASYSIMD_FLOAT64_C(  -325.08) },
      { EASYSIMD_FLOAT64_C(   708.84), EASYSIMD_FLOAT64_C(   653.81), EASYSIMD_FLOAT64_C(  -968.05), EASYSIMD_FLOAT64_C(   849.36) } },
    { { EASYSIMD_FLOAT64_C(   383.09), EASYSIMD_FLOAT64_C(  -175.18), EASYSIMD_FLOAT64_C(    30.44), EASYSIMD_FLOAT64_C(  -403.92) },
      UINT8_C( 76),
      { EASYSIMD_FLOAT64_C(   198.31), EASYSIMD_FLOAT64_C(   522.74), EASYSIMD_FLOAT64_C(  -850.30), EASYSIMD_FLOAT64_C(   116.27) },
      { EASYSIMD_FLOAT64_C(   383.09), EASYSIMD_FLOAT64_C(  -175.18), EASYSIMD_FLOAT64_C(  -850.30), EASYSIMD_FLOAT64_C(   116.27) } },
    { { EASYSIMD_FLOAT64_C(  -944.33), EASYSIMD_FLOAT64_C(  -344.81), EASYSIMD_FLOAT64_C(   210.46), EASYSIMD_FLOAT64_C(  -260.76) },
      UINT8_C(168),
      { EASYSIMD_FLOAT64_C(  -855.65), EASYSIMD_FLOAT64_C(   500.17), EASYSIMD_FLOAT64_C(  -756.94), EASYSIMD_FLOAT64_C(  -627.29) },
      { EASYSIMD_FLOAT64_C(  -944.33), EASYSIMD_FLOAT64_C(  -344.81), EASYSIMD_FLOAT64_C(   210.46), EASYSIMD_FLOAT64_C(  -627.29) } },
    { { EASYSIMD_FLOAT64_C(  -396.23), EASYSIMD_FLOAT64_C(  -915.76), EASYSIMD_FLOAT64_C(   595.63), EASYSIMD_FLOAT64_C(  -858.59) },
      UINT8_C(195),
      { EASYSIMD_FLOAT64_C(   249.44), EASYSIMD_FLOAT64_C(  -826.64), EASYSIMD_FLOAT64_C(   642.44), EASYSIMD_FLOAT64_C(   827.87) },
      { EASYSIMD_FLOAT64_C(   249.44), EASYSIMD_FLOAT64_C(  -826.64), EASYSIMD_FLOAT64_C(   595.63), EASYSIMD_FLOAT64_C(  -858.59) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_loadu_pd(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_loadu_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_loadu_pd(src, k, &a);

    easysimd_test_x86_write_f64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_loadu_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint8_t k;
    const easysimd_float64 a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(123),
      { EASYSIMD_FLOAT64_C(  -319.23), EASYSIMD_FLOAT64_C(   718.78), EASYSIMD_FLOAT64_C(   147.17), EASYSIMD_FLOAT64_C(   724.67) },
      { EASYSIMD_FLOAT64_C(  -319.23), EASYSIMD_FLOAT64_C(   718.78), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   724.67) } },
    { UINT8_C(136),
      { EASYSIMD_FLOAT64_C(  -122.27), EASYSIMD_FLOAT64_C(  -354.82), EASYSIMD_FLOAT64_C(   453.56), EASYSIMD_FLOAT64_C(    90.65) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    90.65) } },
    { UINT8_C(128),
      { EASYSIMD_FLOAT64_C(   104.38), EASYSIMD_FLOAT64_C(    48.96), EASYSIMD_FLOAT64_C(  -170.80), EASYSIMD_FLOAT64_C(   650.29) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(120),
      { EASYSIMD_FLOAT64_C(   483.17), EASYSIMD_FLOAT64_C(   608.29), EASYSIMD_FLOAT64_C(   433.76), EASYSIMD_FLOAT64_C(   145.28) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   145.28) } },
    { UINT8_C( 78),
      { EASYSIMD_FLOAT64_C(  -613.84), EASYSIMD_FLOAT64_C(  -642.64), EASYSIMD_FLOAT64_C(   657.53), EASYSIMD_FLOAT64_C(    82.63) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -642.64), EASYSIMD_FLOAT64_C(   657.53), EASYSIMD_FLOAT64_C(    82.63) } },
    { UINT8_C(248),
      { EASYSIMD_FLOAT64_C(   760.35), EASYSIMD_FLOAT64_C(   940.48), EASYSIMD_FLOAT64_C(  -766.24), EASYSIMD_FLOAT64_C(   538.05) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   538.05) } },
    { UINT8_C(102),
      { EASYSIMD_FLOAT64_C(  -546.99), EASYSIMD_FLOAT64_C(  -781.17), EASYSIMD_FLOAT64_C(   -44.56), EASYSIMD_FLOAT64_C(   600.18) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -781.17), EASYSIMD_FLOAT64_C(   -44.56), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(233),
      { EASYSIMD_FLOAT64_C(   967.69), EASYSIMD_FLOAT64_C(  -522.09), EASYSIMD_FLOAT64_C(  -411.32), EASYSIMD_FLOAT64_C(   421.25) },
      { EASYSIMD_FLOAT64_C(   967.69), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   421.25) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_loadu_pd(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_loadu_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_loadu_pd(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_loadu_epi8 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_64(easysimd__m512i) const int8_t a[64];
    const int8_t r[64];
  } test_vec[] = {
    { {  INT8_C( 115),  INT8_C(   0), -INT8_C(  90), -INT8_C(  57),  INT8_C(  50),  INT8_C(  15), -INT8_C( 121),  INT8_C(  47),
        -INT8_C(   4), -INT8_C(  87), -INT8_C(  26), -INT8_C( 116),  INT8_C(  32), -INT8_C(  59),  INT8_C(  96),  INT8_C(  45),
         INT8_C(  33), -INT8_C(  50), -INT8_C(  83), -INT8_C(  99),  INT8_C(  12),  INT8_C(  64), -INT8_C(  87),  INT8_C( 109),
        -INT8_C(  66), -INT8_C(  55),  INT8_C(  19), -INT8_C(  76), -INT8_C(  80),  INT8_C(  43),  INT8_C(  80),  INT8_C(  36),
         INT8_C(  43), -INT8_C(   9), -INT8_C(  21),  INT8_C(  94),  INT8_C(   6),  INT8_C( 114), -INT8_C( 115),  INT8_C(   2),
         INT8_C(  27),  INT8_C( 115), -INT8_C( 114),  INT8_C(  59),  INT8_C(  56), -INT8_C(  17),  INT8_C( 104),  INT8_C(  89),
        -INT8_C(  67),  INT8_C(  21), -INT8_C(  10), -INT8_C(  55),  INT8_C(  85), -INT8_C(  96),  INT8_C(  55),  INT8_C(  20),
         INT8_C( 105),  INT8_C(  74), -INT8_C(  56),  INT8_C(  26),  INT8_C( 117),  INT8_C(  24),  INT8_C(  62), -INT8_C(  95) },
      {  INT8_C( 115),  INT8_C(   0), -INT8_C(  90), -INT8_C(  57),  INT8_C(  50),  INT8_C(  15), -INT8_C( 121),  INT8_C(  47),
        -INT8_C(   4), -INT8_C(  87), -INT8_C(  26), -INT8_C( 116),  INT8_C(  32), -INT8_C(  59),  INT8_C(  96),  INT8_C(  45),
         INT8_C(  33), -INT8_C(  50), -INT8_C(  83), -INT8_C(  99),  INT8_C(  12),  INT8_C(  64), -INT8_C(  87),  INT8_C( 109),
        -INT8_C(  66), -INT8_C(  55),  INT8_C(  19), -INT8_C(  76), -INT8_C(  80),  INT8_C(  43),  INT8_C(  80),  INT8_C(  36),
         INT8_C(  43), -INT8_C(   9), -INT8_C(  21),  INT8_C(  94),  INT8_C(   6),  INT8_C( 114), -INT8_C( 115),  INT8_C(   2),
         INT8_C(  27),  INT8_C( 115), -INT8_C( 114),  INT8_C(  59),  INT8_C(  56), -INT8_C(  17),  INT8_C( 104),  INT8_C(  89),
        -INT8_C(  67),  INT8_C(  21), -INT8_C(  10), -INT8_C(  55),  INT8_C(  85), -INT8_C(  96),  INT8_C(  55),  INT8_C(  20),
         INT8_C( 105),  INT8_C(  74), -INT8_C(  56),  INT8_C(  26),  INT8_C( 117),  INT8_C(  24),  INT8_C(  62), -INT8_C(  95) } },
    { {  INT8_C(  15),  INT8_C(  41), -INT8_C(   1),  INT8_C(  22), -INT8_C( 101), -INT8_C( 116),  INT8_C(  24), -INT8_C(  74),
        -INT8_C(   1), -INT8_C(  89), -INT8_C(  15),  INT8_C(  55), -INT8_C( 106),  INT8_C(  90), -INT8_C( 112),  INT8_C(  83),
         INT8_C( 111), -INT8_C( 121),  INT8_C(  29), -INT8_C(  59),  INT8_C(  39),  INT8_C(  84), -INT8_C(  39), -INT8_C( 112),
        -INT8_C(  98), -INT8_C(  95), -INT8_C(  86),  INT8_C(  20), -INT8_C(  71), -INT8_C(  24), -INT8_C(  75), -INT8_C(  55),
         INT8_C(  17), -INT8_C(  76), -INT8_C(  33), -INT8_C(  84),  INT8_C(  64), -INT8_C(   9),  INT8_C(  98),  INT8_C(  63),
        -INT8_C(  98),  INT8_C(  84),  INT8_C( 118),  INT8_C(  52), -INT8_C(  82),  INT8_C(   6), -INT8_C( 120),  INT8_C(  29),
        -INT8_C( 115), -INT8_C(  91), -INT8_C(  30), -INT8_C(  76), -INT8_C(   7), -INT8_C(  69),  INT8_C(  69), -INT8_C( 105),
         INT8_C(  92), -INT8_C(  17), -INT8_C(  85),  INT8_C(  22), -INT8_C(  40),  INT8_C(  96), -INT8_C(  33), -INT8_C(  23) },
      {  INT8_C(  15),  INT8_C(  41), -INT8_C(   1),  INT8_C(  22), -INT8_C( 101), -INT8_C( 116),  INT8_C(  24), -INT8_C(  74),
        -INT8_C(   1), -INT8_C(  89), -INT8_C(  15),  INT8_C(  55), -INT8_C( 106),  INT8_C(  90), -INT8_C( 112),  INT8_C(  83),
         INT8_C( 111), -INT8_C( 121),  INT8_C(  29), -INT8_C(  59),  INT8_C(  39),  INT8_C(  84), -INT8_C(  39), -INT8_C( 112),
        -INT8_C(  98), -INT8_C(  95), -INT8_C(  86),  INT8_C(  20), -INT8_C(  71), -INT8_C(  24), -INT8_C(  75), -INT8_C(  55),
         INT8_C(  17), -INT8_C(  76), -INT8_C(  33), -INT8_C(  84),  INT8_C(  64), -INT8_C(   9),  INT8_C(  98),  INT8_C(  63),
        -INT8_C(  98),  INT8_C(  84),  INT8_C( 118),  INT8_C(  52), -INT8_C(  82),  INT8_C(   6), -INT8_C( 120),  INT8_C(  29),
        -INT8_C( 115), -INT8_C(  91), -INT8_C(  30), -INT8_C(  76), -INT8_C(   7), -INT8_C(  69),  INT8_C(  69), -INT8_C( 105),
         INT8_C(  92), -INT8_C(  17), -INT8_C(  85),  INT8_C(  22), -INT8_C(  40),  INT8_C(  96), -INT8_C(  33), -INT8_C(  23) } },
    { {  INT8_C(  20), -INT8_C(  66), -INT8_C( 106),  INT8_C(  84), -INT8_C(  75), -INT8_C(   8), -INT8_C( 109),  INT8_C(  84),
         INT8_C(  76),  INT8_C(   9), -INT8_C( 120), -INT8_C(   6),  INT8_C(  16),  INT8_C(  16),  INT8_C(  24), -INT8_C(  99),
        -INT8_C(  75), -INT8_C(   6),  INT8_C(  82), -INT8_C(  82), -INT8_C(  74), -INT8_C( 105),  INT8_C(  70),  INT8_C(  18),
        -INT8_C( 122), -INT8_C(  15),  INT8_C(  40),  INT8_C(  94),  INT8_C(  82),  INT8_C(   7),  INT8_C(  72),  INT8_C( 102),
        -INT8_C(  59), -INT8_C(  34), -INT8_C(  69),  INT8_C( 123), -INT8_C(  42),  INT8_C(  78), -INT8_C(  49),  INT8_C(  35),
         INT8_C(  88),  INT8_C(  87),  INT8_C(  29),  INT8_C( 104),  INT8_C( 104),  INT8_C(  53),  INT8_C(   5),  INT8_C(  29),
         INT8_C(  48),  INT8_C(  87), -INT8_C(  52), -INT8_C(  26), -INT8_C(  18),  INT8_C(  18), -INT8_C(   8),  INT8_C( 117),
         INT8_C(   3),  INT8_C(  33), -INT8_C(  45),  INT8_C(  85),  INT8_C(  40),  INT8_C(  27), -INT8_C(  68), -INT8_C(  18) },
      {  INT8_C(  20), -INT8_C(  66), -INT8_C( 106),  INT8_C(  84), -INT8_C(  75), -INT8_C(   8), -INT8_C( 109),  INT8_C(  84),
         INT8_C(  76),  INT8_C(   9), -INT8_C( 120), -INT8_C(   6),  INT8_C(  16),  INT8_C(  16),  INT8_C(  24), -INT8_C(  99),
        -INT8_C(  75), -INT8_C(   6),  INT8_C(  82), -INT8_C(  82), -INT8_C(  74), -INT8_C( 105),  INT8_C(  70),  INT8_C(  18),
        -INT8_C( 122), -INT8_C(  15),  INT8_C(  40),  INT8_C(  94),  INT8_C(  82),  INT8_C(   7),  INT8_C(  72),  INT8_C( 102),
        -INT8_C(  59), -INT8_C(  34), -INT8_C(  69),  INT8_C( 123), -INT8_C(  42),  INT8_C(  78), -INT8_C(  49),  INT8_C(  35),
         INT8_C(  88),  INT8_C(  87),  INT8_C(  29),  INT8_C( 104),  INT8_C( 104),  INT8_C(  53),  INT8_C(   5),  INT8_C(  29),
         INT8_C(  48),  INT8_C(  87), -INT8_C(  52), -INT8_C(  26), -INT8_C(  18),  INT8_C(  18), -INT8_C(   8),  INT8_C( 117),
         INT8_C(   3),  INT8_C(  33), -INT8_C(  45),  INT8_C(  85),  INT8_C(  40),  INT8_C(  27), -INT8_C(  68), -INT8_C(  18) } },
    { { -INT8_C(   7),  INT8_C( 119),  INT8_C( 105), -INT8_C(  48), -INT8_C(  59),  INT8_C(  56), -INT8_C(  13),  INT8_C(  29),
        -INT8_C( 113),  INT8_C(  16), -INT8_C( 123), -INT8_C(   9),  INT8_C(  70), -INT8_C( 117),  INT8_C(  21),  INT8_C( 118),
        -INT8_C(  30), -INT8_C(  31),  INT8_C(  92), -INT8_C(  47), -INT8_C(  13),  INT8_C(  84),  INT8_C(  70), -INT8_C(  10),
         INT8_C( 117),  INT8_C(  25),  INT8_C(  76), -INT8_C(  98),  INT8_C(  53),  INT8_C(   8), -INT8_C( 116),  INT8_C(  46),
             INT8_MAX, -INT8_C(  11), -INT8_C(   2),  INT8_C(  68),  INT8_C(  45), -INT8_C(  15),  INT8_C(  98), -INT8_C(  68),
         INT8_C(   2), -INT8_C(  25), -INT8_C(  76),  INT8_C(  72),  INT8_C( 114), -INT8_C(  55), -INT8_C(  66),  INT8_C(  85),
        -INT8_C(  86),  INT8_C(  26),  INT8_C(  38), -INT8_C(  99),  INT8_C( 110),  INT8_C( 108), -INT8_C( 109), -INT8_C(  28),
        -INT8_C( 123), -INT8_C(  33), -INT8_C( 126), -INT8_C(  70), -INT8_C(  25),  INT8_C(  14), -INT8_C(  23),  INT8_C( 102) },
      { -INT8_C(   7),  INT8_C( 119),  INT8_C( 105), -INT8_C(  48), -INT8_C(  59),  INT8_C(  56), -INT8_C(  13),  INT8_C(  29),
        -INT8_C( 113),  INT8_C(  16), -INT8_C( 123), -INT8_C(   9),  INT8_C(  70), -INT8_C( 117),  INT8_C(  21),  INT8_C( 118),
        -INT8_C(  30), -INT8_C(  31),  INT8_C(  92), -INT8_C(  47), -INT8_C(  13),  INT8_C(  84),  INT8_C(  70), -INT8_C(  10),
         INT8_C( 117),  INT8_C(  25),  INT8_C(  76), -INT8_C(  98),  INT8_C(  53),  INT8_C(   8), -INT8_C( 116),  INT8_C(  46),
             INT8_MAX, -INT8_C(  11), -INT8_C(   2),  INT8_C(  68),  INT8_C(  45), -INT8_C(  15),  INT8_C(  98), -INT8_C(  68),
         INT8_C(   2), -INT8_C(  25), -INT8_C(  76),  INT8_C(  72),  INT8_C( 114), -INT8_C(  55), -INT8_C(  66),  INT8_C(  85),
        -INT8_C(  86),  INT8_C(  26),  INT8_C(  38), -INT8_C(  99),  INT8_C( 110),  INT8_C( 108), -INT8_C( 109), -INT8_C(  28),
        -INT8_C( 123), -INT8_C(  33), -INT8_C( 126), -INT8_C(  70), -INT8_C(  25),  INT8_C(  14), -INT8_C(  23),  INT8_C( 102) } },
    { {  INT8_C(   3), -INT8_C(  25), -INT8_C(  85),  INT8_C(  48), -INT8_C(  39),  INT8_C(  13), -INT8_C(  20), -INT8_C(  37),
        -INT8_C(  12), -INT8_C(  96),  INT8_C(  35),  INT8_C( 103),  INT8_C( 105), -INT8_C(  31), -INT8_C(  68),  INT8_C(  19),
        -INT8_C(   5), -INT8_C(  30), -INT8_C(  80),  INT8_C( 105),  INT8_C(  78),  INT8_C(  68),  INT8_C(  77), -INT8_C(  45),
         INT8_C(  35), -INT8_C(  49), -INT8_C( 114),  INT8_C(  11), -INT8_C(  35),  INT8_C( 119),  INT8_C( 113), -INT8_C(  32),
         INT8_C(  94),  INT8_C(  28),  INT8_C(  16),  INT8_C(  55),  INT8_C(  41), -INT8_C(   3),  INT8_C(  18),  INT8_C(  30),
        -INT8_C(  99),  INT8_C(  53), -INT8_C( 123),  INT8_C(   7),  INT8_C(  22),  INT8_C(  65),  INT8_C(  26),  INT8_C(  17),
         INT8_C(  35), -INT8_C(  53),  INT8_C( 123),  INT8_C( 113),  INT8_C(  15), -INT8_C(  56),  INT8_C(  68),  INT8_C(  50),
        -INT8_C( 104), -INT8_C(  46),  INT8_C(  61),  INT8_C( 117),  INT8_C(  73), -INT8_C(  81),  INT8_C(  86), -INT8_C(  88) },
      {  INT8_C(   3), -INT8_C(  25), -INT8_C(  85),  INT8_C(  48), -INT8_C(  39),  INT8_C(  13), -INT8_C(  20), -INT8_C(  37),
        -INT8_C(  12), -INT8_C(  96),  INT8_C(  35),  INT8_C( 103),  INT8_C( 105), -INT8_C(  31), -INT8_C(  68),  INT8_C(  19),
        -INT8_C(   5), -INT8_C(  30), -INT8_C(  80),  INT8_C( 105),  INT8_C(  78),  INT8_C(  68),  INT8_C(  77), -INT8_C(  45),
         INT8_C(  35), -INT8_C(  49), -INT8_C( 114),  INT8_C(  11), -INT8_C(  35),  INT8_C( 119),  INT8_C( 113), -INT8_C(  32),
         INT8_C(  94),  INT8_C(  28),  INT8_C(  16),  INT8_C(  55),  INT8_C(  41), -INT8_C(   3),  INT8_C(  18),  INT8_C(  30),
        -INT8_C(  99),  INT8_C(  53), -INT8_C( 123),  INT8_C(   7),  INT8_C(  22),  INT8_C(  65),  INT8_C(  26),  INT8_C(  17),
         INT8_C(  35), -INT8_C(  53),  INT8_C( 123),  INT8_C( 113),  INT8_C(  15), -INT8_C(  56),  INT8_C(  68),  INT8_C(  50),
        -INT8_C( 104), -INT8_C(  46),  INT8_C(  61),  INT8_C( 117),  INT8_C(  73), -INT8_C(  81),  INT8_C(  86), -INT8_C(  88) } },
    { { -INT8_C(  53),  INT8_C( 102), -INT8_C(  33), -INT8_C(  11),  INT8_C(  99), -INT8_C(  14),  INT8_C(  19),  INT8_C(   1),
         INT8_C(  39), -INT8_C( 104),  INT8_C(   8),  INT8_C(  62), -INT8_C(  39),  INT8_C(  34),  INT8_C(  79), -INT8_C(   4),
        -INT8_C(  19), -INT8_C(  54),  INT8_C( 109), -INT8_C(   4), -INT8_C( 109), -INT8_C(  79),  INT8_C(  47),  INT8_C(  43),
        -INT8_C( 124),  INT8_C( 108), -INT8_C(  96), -INT8_C(  51),  INT8_C(  27), -INT8_C(  10),  INT8_C( 117), -INT8_C(  25),
         INT8_C(  93),  INT8_C(  85), -INT8_C(  36), -INT8_C(  64),  INT8_C(  71), -INT8_C(  17), -INT8_C(  63),  INT8_C( 110),
        -INT8_C( 121), -INT8_C(  55), -INT8_C(  84),  INT8_C(  96), -INT8_C(  20), -INT8_C(   4),  INT8_C(  92), -INT8_C(  39),
        -INT8_C(  58), -INT8_C(  55), -INT8_C(  42),  INT8_C(  89),  INT8_C( 122),  INT8_C(   5), -INT8_C( 124), -INT8_C(   2),
         INT8_C( 113),  INT8_C(  37), -INT8_C(  52), -INT8_C( 115),  INT8_C(  27),  INT8_C(  65),  INT8_C( 116),  INT8_C( 120) },
      { -INT8_C(  53),  INT8_C( 102), -INT8_C(  33), -INT8_C(  11),  INT8_C(  99), -INT8_C(  14),  INT8_C(  19),  INT8_C(   1),
         INT8_C(  39), -INT8_C( 104),  INT8_C(   8),  INT8_C(  62), -INT8_C(  39),  INT8_C(  34),  INT8_C(  79), -INT8_C(   4),
        -INT8_C(  19), -INT8_C(  54),  INT8_C( 109), -INT8_C(   4), -INT8_C( 109), -INT8_C(  79),  INT8_C(  47),  INT8_C(  43),
        -INT8_C( 124),  INT8_C( 108), -INT8_C(  96), -INT8_C(  51),  INT8_C(  27), -INT8_C(  10),  INT8_C( 117), -INT8_C(  25),
         INT8_C(  93),  INT8_C(  85), -INT8_C(  36), -INT8_C(  64),  INT8_C(  71), -INT8_C(  17), -INT8_C(  63),  INT8_C( 110),
        -INT8_C( 121), -INT8_C(  55), -INT8_C(  84),  INT8_C(  96), -INT8_C(  20), -INT8_C(   4),  INT8_C(  92), -INT8_C(  39),
        -INT8_C(  58), -INT8_C(  55), -INT8_C(  42),  INT8_C(  89),  INT8_C( 122),  INT8_C(   5), -INT8_C( 124), -INT8_C(   2),
         INT8_C( 113),  INT8_C(  37), -INT8_C(  52), -INT8_C( 115),  INT8_C(  27),  INT8_C(  65),  INT8_C( 116),  INT8_C( 120) } },
    { { -INT8_C( 106),  INT8_C(  80),  INT8_C(  57), -INT8_C(  35),  INT8_C(  63), -INT8_C(   6),  INT8_C(  76), -INT8_C(  58),
        -INT8_C(  60), -INT8_C(   8),  INT8_C(  38), -INT8_C(  80), -INT8_C(  12), -INT8_C( 126), -INT8_C( 119), -INT8_C(  69),
         INT8_C(  75),  INT8_C(  95),  INT8_C(  20), -INT8_C(  59),  INT8_C( 100), -INT8_C( 103), -INT8_C(  60), -INT8_C(  42),
        -INT8_C(  66), -INT8_C( 112),  INT8_C(  99), -INT8_C(  39), -INT8_C(  47), -INT8_C(  41),  INT8_C(  82),  INT8_C( 104),
         INT8_C(  39), -INT8_C( 117),  INT8_C(  69),  INT8_C( 102), -INT8_C( 123), -INT8_C( 111),  INT8_C(  44),  INT8_C(  73),
        -INT8_C( 118),  INT8_C(  82), -INT8_C(   7),  INT8_C( 126), -INT8_C(  44), -INT8_C( 125),  INT8_C(  57),  INT8_C(  31),
        -INT8_C(  30),  INT8_C(  78), -INT8_C(  28),  INT8_C(  71), -INT8_C(  25), -INT8_C(  88),  INT8_C(  29), -INT8_C(  91),
         INT8_C(  56),      INT8_MIN,  INT8_C( 126),  INT8_C(  10),  INT8_C(  87), -INT8_C(  48),  INT8_C( 114),  INT8_C( 126) },
      { -INT8_C( 106),  INT8_C(  80),  INT8_C(  57), -INT8_C(  35),  INT8_C(  63), -INT8_C(   6),  INT8_C(  76), -INT8_C(  58),
        -INT8_C(  60), -INT8_C(   8),  INT8_C(  38), -INT8_C(  80), -INT8_C(  12), -INT8_C( 126), -INT8_C( 119), -INT8_C(  69),
         INT8_C(  75),  INT8_C(  95),  INT8_C(  20), -INT8_C(  59),  INT8_C( 100), -INT8_C( 103), -INT8_C(  60), -INT8_C(  42),
        -INT8_C(  66), -INT8_C( 112),  INT8_C(  99), -INT8_C(  39), -INT8_C(  47), -INT8_C(  41),  INT8_C(  82),  INT8_C( 104),
         INT8_C(  39), -INT8_C( 117),  INT8_C(  69),  INT8_C( 102), -INT8_C( 123), -INT8_C( 111),  INT8_C(  44),  INT8_C(  73),
        -INT8_C( 118),  INT8_C(  82), -INT8_C(   7),  INT8_C( 126), -INT8_C(  44), -INT8_C( 125),  INT8_C(  57),  INT8_C(  31),
        -INT8_C(  30),  INT8_C(  78), -INT8_C(  28),  INT8_C(  71), -INT8_C(  25), -INT8_C(  88),  INT8_C(  29), -INT8_C(  91),
         INT8_C(  56),      INT8_MIN,  INT8_C( 126),  INT8_C(  10),  INT8_C(  87), -INT8_C(  48),  INT8_C( 114),  INT8_C( 126) } },
    { {  INT8_C(  91), -INT8_C(  73), -INT8_C(  28), -INT8_C(  31),  INT8_C(  73),  INT8_C(  16),  INT8_C(  42), -INT8_C(  45),
         INT8_C(  98),  INT8_C(  36),  INT8_C(  81),  INT8_C(  54), -INT8_C(  89), -INT8_C( 117),  INT8_C(  85), -INT8_C( 119),
        -INT8_C(  39),  INT8_C(  57), -INT8_C(  48), -INT8_C(  64), -INT8_C(  30), -INT8_C(  19),  INT8_C( 101),  INT8_C(  26),
         INT8_C( 109), -INT8_C(  29),  INT8_C(  36), -INT8_C(  60), -INT8_C(  76), -INT8_C( 106),  INT8_C(  66),  INT8_C(  15),
         INT8_C(  78),  INT8_C(  38), -INT8_C(  16), -INT8_C( 105),  INT8_C(  54),  INT8_C(  27),  INT8_C( 106), -INT8_C( 104),
         INT8_C(  63), -INT8_C(  69), -INT8_C(  50), -INT8_C(  26),  INT8_C(  70),  INT8_C(  35),  INT8_C( 111),  INT8_C(  31),
         INT8_C(  93),  INT8_C(  64), -INT8_C(  33),  INT8_C(  63),  INT8_C(  45),  INT8_C(  68),  INT8_C(  89), -INT8_C( 101),
         INT8_C(  40),  INT8_C( 126),  INT8_C(  95), -INT8_C(  36),  INT8_C(  20), -INT8_C(  94), -INT8_C(  21),  INT8_C(  98) },
      {  INT8_C(  91), -INT8_C(  73), -INT8_C(  28), -INT8_C(  31),  INT8_C(  73),  INT8_C(  16),  INT8_C(  42), -INT8_C(  45),
         INT8_C(  98),  INT8_C(  36),  INT8_C(  81),  INT8_C(  54), -INT8_C(  89), -INT8_C( 117),  INT8_C(  85), -INT8_C( 119),
        -INT8_C(  39),  INT8_C(  57), -INT8_C(  48), -INT8_C(  64), -INT8_C(  30), -INT8_C(  19),  INT8_C( 101),  INT8_C(  26),
         INT8_C( 109), -INT8_C(  29),  INT8_C(  36), -INT8_C(  60), -INT8_C(  76), -INT8_C( 106),  INT8_C(  66),  INT8_C(  15),
         INT8_C(  78),  INT8_C(  38), -INT8_C(  16), -INT8_C( 105),  INT8_C(  54),  INT8_C(  27),  INT8_C( 106), -INT8_C( 104),
         INT8_C(  63), -INT8_C(  69), -INT8_C(  50), -INT8_C(  26),  INT8_C(  70),  INT8_C(  35),  INT8_C( 111),  INT8_C(  31),
         INT8_C(  93),  INT8_C(  64), -INT8_C(  33),  INT8_C(  63),  INT8_C(  45),  INT8_C(  68),  INT8_C(  89), -INT8_C( 101),
         INT8_C(  40),  INT8_C( 126),  INT8_C(  95), -INT8_C(  36),  INT8_C(  20), -INT8_C(  94), -INT8_C(  21),  INT8_C(  98) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_test_x86_assert_equal_i8x64(easysimd_mm512_load_si512(test_vec[i].a), easysimd_mm512_loadu_epi8(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i r = a;

    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_loadu_epi16 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_64(easysimd__m512i) const int16_t a[32];
    const int16_t r[32];
  } test_vec[] = {
    { {  INT16_C(  1628), -INT16_C( 19656), -INT16_C( 13922),  INT16_C(   835), -INT16_C(  2787), -INT16_C( 10395),  INT16_C(  6399),  INT16_C( 11765),
         INT16_C(  4017),  INT16_C( 12521),  INT16_C( 21092), -INT16_C( 30322),  INT16_C(  5175),  INT16_C( 12717), -INT16_C( 28003), -INT16_C(  1686),
        -INT16_C( 23911),  INT16_C( 14252), -INT16_C(  4245), -INT16_C( 30662), -INT16_C( 24604), -INT16_C(  7329),  INT16_C( 21687),  INT16_C( 26897),
        -INT16_C(  1436), -INT16_C( 14183),  INT16_C( 10060), -INT16_C( 31919), -INT16_C(   452), -INT16_C(  9803),  INT16_C(  8081),  INT16_C( 10962) },
      {  INT16_C(  1628), -INT16_C( 19656), -INT16_C( 13922),  INT16_C(   835), -INT16_C(  2787), -INT16_C( 10395),  INT16_C(  6399),  INT16_C( 11765),
         INT16_C(  4017),  INT16_C( 12521),  INT16_C( 21092), -INT16_C( 30322),  INT16_C(  5175),  INT16_C( 12717), -INT16_C( 28003), -INT16_C(  1686),
        -INT16_C( 23911),  INT16_C( 14252), -INT16_C(  4245), -INT16_C( 30662), -INT16_C( 24604), -INT16_C(  7329),  INT16_C( 21687),  INT16_C( 26897),
        -INT16_C(  1436), -INT16_C( 14183),  INT16_C( 10060), -INT16_C( 31919), -INT16_C(   452), -INT16_C(  9803),  INT16_C(  8081),  INT16_C( 10962) } },
    { {  INT16_C( 32450),  INT16_C( 11617), -INT16_C( 25746),  INT16_C( 21174),  INT16_C(  5434), -INT16_C(  3786),  INT16_C( 18282), -INT16_C( 12710),
        -INT16_C(  3263), -INT16_C( 29034), -INT16_C(  6373),  INT16_C( 22289), -INT16_C( 14618),  INT16_C( 30512),  INT16_C(   742), -INT16_C( 22367),
         INT16_C(   640), -INT16_C(  4395), -INT16_C( 29795), -INT16_C( 10431),  INT16_C( 30625),  INT16_C(  3016),  INT16_C(  9150), -INT16_C(    39),
         INT16_C( 28438),  INT16_C( 12685), -INT16_C( 24746),  INT16_C( 15496), -INT16_C( 18331),  INT16_C( 19379),  INT16_C( 21690),  INT16_C( 15347) },
      {  INT16_C( 32450),  INT16_C( 11617), -INT16_C( 25746),  INT16_C( 21174),  INT16_C(  5434), -INT16_C(  3786),  INT16_C( 18282), -INT16_C( 12710),
        -INT16_C(  3263), -INT16_C( 29034), -INT16_C(  6373),  INT16_C( 22289), -INT16_C( 14618),  INT16_C( 30512),  INT16_C(   742), -INT16_C( 22367),
         INT16_C(   640), -INT16_C(  4395), -INT16_C( 29795), -INT16_C( 10431),  INT16_C( 30625),  INT16_C(  3016),  INT16_C(  9150), -INT16_C(    39),
         INT16_C( 28438),  INT16_C( 12685), -INT16_C( 24746),  INT16_C( 15496), -INT16_C( 18331),  INT16_C( 19379),  INT16_C( 21690),  INT16_C( 15347) } },
    { { -INT16_C( 13994), -INT16_C(  3287),  INT16_C( 27220), -INT16_C(  2614), -INT16_C( 27679), -INT16_C( 24832), -INT16_C(  9802), -INT16_C( 13153),
         INT16_C( 11336), -INT16_C( 24578), -INT16_C( 31029),  INT16_C( 12763), -INT16_C( 28865), -INT16_C(  1668),  INT16_C( 28899),  INT16_C( 14900),
         INT16_C( 24121), -INT16_C( 29395), -INT16_C(  1848), -INT16_C( 21885), -INT16_C( 31861),  INT16_C( 16713), -INT16_C(  6051), -INT16_C( 23283),
         INT16_C(  2837), -INT16_C(  8124),  INT16_C(  8338), -INT16_C( 12015), -INT16_C( 29009), -INT16_C( 27958), -INT16_C(     2),  INT16_C( 14284) },
      { -INT16_C( 13994), -INT16_C(  3287),  INT16_C( 27220), -INT16_C(  2614), -INT16_C( 27679), -INT16_C( 24832), -INT16_C(  9802), -INT16_C( 13153),
         INT16_C( 11336), -INT16_C( 24578), -INT16_C( 31029),  INT16_C( 12763), -INT16_C( 28865), -INT16_C(  1668),  INT16_C( 28899),  INT16_C( 14900),
         INT16_C( 24121), -INT16_C( 29395), -INT16_C(  1848), -INT16_C( 21885), -INT16_C( 31861),  INT16_C( 16713), -INT16_C(  6051), -INT16_C( 23283),
         INT16_C(  2837), -INT16_C(  8124),  INT16_C(  8338), -INT16_C( 12015), -INT16_C( 29009), -INT16_C( 27958), -INT16_C(     2),  INT16_C( 14284) } },
    { { -INT16_C(  1443),  INT16_C(  9668),  INT16_C( 18418),  INT16_C( 32207),  INT16_C(  6603),  INT16_C( 10430), -INT16_C( 13567),  INT16_C(  5837),
         INT16_C(  4823),  INT16_C( 27127),  INT16_C(  2098), -INT16_C(  7878),  INT16_C(  1174), -INT16_C( 27533),  INT16_C( 16387),  INT16_C( 24779),
        -INT16_C( 28614),  INT16_C( 11398),  INT16_C( 21975), -INT16_C( 23895),  INT16_C( 26478),  INT16_C( 28874), -INT16_C( 26574),  INT16_C(  2438),
         INT16_C( 32170), -INT16_C(  9102), -INT16_C( 21370),  INT16_C(  7357),  INT16_C( 12465), -INT16_C( 19279),  INT16_C( 31856), -INT16_C( 21995) },
      { -INT16_C(  1443),  INT16_C(  9668),  INT16_C( 18418),  INT16_C( 32207),  INT16_C(  6603),  INT16_C( 10430), -INT16_C( 13567),  INT16_C(  5837),
         INT16_C(  4823),  INT16_C( 27127),  INT16_C(  2098), -INT16_C(  7878),  INT16_C(  1174), -INT16_C( 27533),  INT16_C( 16387),  INT16_C( 24779),
        -INT16_C( 28614),  INT16_C( 11398),  INT16_C( 21975), -INT16_C( 23895),  INT16_C( 26478),  INT16_C( 28874), -INT16_C( 26574),  INT16_C(  2438),
         INT16_C( 32170), -INT16_C(  9102), -INT16_C( 21370),  INT16_C(  7357),  INT16_C( 12465), -INT16_C( 19279),  INT16_C( 31856), -INT16_C( 21995) } },
    { { -INT16_C( 25844), -INT16_C(  6954),  INT16_C( 32752),  INT16_C( 24454),  INT16_C( 20966),  INT16_C(  6607),  INT16_C( 21993), -INT16_C( 27870),
        -INT16_C( 27181),  INT16_C( 22895),  INT16_C( 11329), -INT16_C(  3467),  INT16_C(  9820), -INT16_C( 12889), -INT16_C( 17245), -INT16_C( 20617),
         INT16_C( 20055),  INT16_C( 18323),  INT16_C(  6861), -INT16_C( 19290),  INT16_C( 30059),  INT16_C( 21709), -INT16_C(  4149), -INT16_C( 24857),
         INT16_C( 22148), -INT16_C( 14601),  INT16_C( 27778), -INT16_C(  8520),  INT16_C( 24467),  INT16_C( 13995),  INT16_C(  8987),  INT16_C( 29413) },
      { -INT16_C( 25844), -INT16_C(  6954),  INT16_C( 32752),  INT16_C( 24454),  INT16_C( 20966),  INT16_C(  6607),  INT16_C( 21993), -INT16_C( 27870),
        -INT16_C( 27181),  INT16_C( 22895),  INT16_C( 11329), -INT16_C(  3467),  INT16_C(  9820), -INT16_C( 12889), -INT16_C( 17245), -INT16_C( 20617),
         INT16_C( 20055),  INT16_C( 18323),  INT16_C(  6861), -INT16_C( 19290),  INT16_C( 30059),  INT16_C( 21709), -INT16_C(  4149), -INT16_C( 24857),
         INT16_C( 22148), -INT16_C( 14601),  INT16_C( 27778), -INT16_C(  8520),  INT16_C( 24467),  INT16_C( 13995),  INT16_C(  8987),  INT16_C( 29413) } },
    { {  INT16_C( 31089),  INT16_C( 16058),  INT16_C( 24723), -INT16_C(   270), -INT16_C( 16426), -INT16_C( 24238),  INT16_C( 14767),  INT16_C( 13119),
         INT16_C( 13967),  INT16_C(  4601), -INT16_C( 19806),  INT16_C( 13807), -INT16_C( 25839),  INT16_C( 11627),  INT16_C( 20926),  INT16_C( 12191),
         INT16_C( 22986),  INT16_C( 23917),  INT16_C( 24762), -INT16_C( 28581), -INT16_C( 21217), -INT16_C( 12751),  INT16_C( 28902),  INT16_C( 29954),
        -INT16_C(  1114),  INT16_C( 18566),  INT16_C( 30125), -INT16_C( 16514), -INT16_C(  5872), -INT16_C( 12564), -INT16_C( 29894),  INT16_C(  1277) },
      {  INT16_C( 31089),  INT16_C( 16058),  INT16_C( 24723), -INT16_C(   270), -INT16_C( 16426), -INT16_C( 24238),  INT16_C( 14767),  INT16_C( 13119),
         INT16_C( 13967),  INT16_C(  4601), -INT16_C( 19806),  INT16_C( 13807), -INT16_C( 25839),  INT16_C( 11627),  INT16_C( 20926),  INT16_C( 12191),
         INT16_C( 22986),  INT16_C( 23917),  INT16_C( 24762), -INT16_C( 28581), -INT16_C( 21217), -INT16_C( 12751),  INT16_C( 28902),  INT16_C( 29954),
        -INT16_C(  1114),  INT16_C( 18566),  INT16_C( 30125), -INT16_C( 16514), -INT16_C(  5872), -INT16_C( 12564), -INT16_C( 29894),  INT16_C(  1277) } },
    { {  INT16_C( 27621), -INT16_C( 24735), -INT16_C( 17205), -INT16_C(  5585),  INT16_C( 24681),  INT16_C( 20409), -INT16_C( 17456),  INT16_C( 30404),
         INT16_C( 19126),  INT16_C( 25790),  INT16_C( 15552), -INT16_C( 12253),  INT16_C(  3878),  INT16_C( 24735), -INT16_C( 25446),  INT16_C( 32613),
        -INT16_C( 14841), -INT16_C( 11746),  INT16_C( 19843), -INT16_C(  4931),  INT16_C( 30381),  INT16_C( 32060),  INT16_C(    49), -INT16_C(  6157),
        -INT16_C( 19893),  INT16_C(  2891),  INT16_C( 28398),  INT16_C(  5339),  INT16_C( 31357),  INT16_C(  6261), -INT16_C(  9705),  INT16_C(  7831) },
      {  INT16_C( 27621), -INT16_C( 24735), -INT16_C( 17205), -INT16_C(  5585),  INT16_C( 24681),  INT16_C( 20409), -INT16_C( 17456),  INT16_C( 30404),
         INT16_C( 19126),  INT16_C( 25790),  INT16_C( 15552), -INT16_C( 12253),  INT16_C(  3878),  INT16_C( 24735), -INT16_C( 25446),  INT16_C( 32613),
        -INT16_C( 14841), -INT16_C( 11746),  INT16_C( 19843), -INT16_C(  4931),  INT16_C( 30381),  INT16_C( 32060),  INT16_C(    49), -INT16_C(  6157),
        -INT16_C( 19893),  INT16_C(  2891),  INT16_C( 28398),  INT16_C(  5339),  INT16_C( 31357),  INT16_C(  6261), -INT16_C(  9705),  INT16_C(  7831) } },
    { { -INT16_C( 18784),  INT16_C(  9201), -INT16_C( 20989), -INT16_C( 20208),  INT16_C( 19492),  INT16_C( 21806),  INT16_C(  8780), -INT16_C( 26820),
        -INT16_C( 30508), -INT16_C( 15710),  INT16_C( 32502),  INT16_C( 29911),  INT16_C( 19704),  INT16_C(  3980),  INT16_C(  8998), -INT16_C( 14802),
         INT16_C(  8153), -INT16_C(  8726), -INT16_C(  1331), -INT16_C(  3698), -INT16_C( 17338), -INT16_C( 28090), -INT16_C( 32034), -INT16_C( 19926),
        -INT16_C( 13302),  INT16_C(   373),  INT16_C( 19530),  INT16_C( 17269),  INT16_C(   408), -INT16_C( 16814), -INT16_C( 32732), -INT16_C(   380) },
      { -INT16_C( 18784),  INT16_C(  9201), -INT16_C( 20989), -INT16_C( 20208),  INT16_C( 19492),  INT16_C( 21806),  INT16_C(  8780), -INT16_C( 26820),
        -INT16_C( 30508), -INT16_C( 15710),  INT16_C( 32502),  INT16_C( 29911),  INT16_C( 19704),  INT16_C(  3980),  INT16_C(  8998), -INT16_C( 14802),
         INT16_C(  8153), -INT16_C(  8726), -INT16_C(  1331), -INT16_C(  3698), -INT16_C( 17338), -INT16_C( 28090), -INT16_C( 32034), -INT16_C( 19926),
        -INT16_C( 13302),  INT16_C(   373),  INT16_C( 19530),  INT16_C( 17269),  INT16_C(   408), -INT16_C( 16814), -INT16_C( 32732), -INT16_C(   380) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_test_x86_assert_equal_i16x32(easysimd_mm512_load_si512(test_vec[i].a), easysimd_mm512_loadu_epi16(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i r = a;

    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_loadu_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_64(easysimd__m512i) const int32_t a[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1085279312),  INT32_C(  1689654203), -INT32_C(  1704027444),  INT32_C(  1992004399),  INT32_C(  1598136076), -INT32_C(  1107314712),  INT32_C(  1634510512),  INT32_C(  2144272078),
        -INT32_C(  1405215247), -INT32_C(    65931984), -INT32_C(  1097433201),  INT32_C(   523522579), -INT32_C(   629246223), -INT32_C(   560497363), -INT32_C(   230751453), -INT32_C(   210694911) },
      { -INT32_C(  1085279312),  INT32_C(  1689654203), -INT32_C(  1704027444),  INT32_C(  1992004399),  INT32_C(  1598136076), -INT32_C(  1107314712),  INT32_C(  1634510512),  INT32_C(  2144272078),
        -INT32_C(  1405215247), -INT32_C(    65931984), -INT32_C(  1097433201),  INT32_C(   523522579), -INT32_C(   629246223), -INT32_C(   560497363), -INT32_C(   230751453), -INT32_C(   210694911) } },
    { {  INT32_C(  1537191723),  INT32_C(   878227620),  INT32_C(  1139994160),  INT32_C(   845293376), -INT32_C(   905125475), -INT32_C(  2102877346), -INT32_C(  1468733529),  INT32_C(   547087861),
         INT32_C(   964377492),  INT32_C(   460182507),  INT32_C(    39739330),  INT32_C(   590659974),  INT32_C(    15614114), -INT32_C(  1954375964),  INT32_C(  1932785278),  INT32_C(  1888735195) },
      {  INT32_C(  1537191723),  INT32_C(   878227620),  INT32_C(  1139994160),  INT32_C(   845293376), -INT32_C(   905125475), -INT32_C(  2102877346), -INT32_C(  1468733529),  INT32_C(   547087861),
         INT32_C(   964377492),  INT32_C(   460182507),  INT32_C(    39739330),  INT32_C(   590659974),  INT32_C(    15614114), -INT32_C(  1954375964),  INT32_C(  1932785278),  INT32_C(  1888735195) } },
    { { -INT32_C(   173470198), -INT32_C(  1542383902), -INT32_C(    56201355), -INT32_C(   769664208), -INT32_C(     2945765),  INT32_C(   579491236),  INT32_C(   664125004), -INT32_C(  1751701363),
         INT32_C(   411844662), -INT32_C(   860054186),  INT32_C(  1036542733),  INT32_C(  1494279998), -INT32_C(  1722162187), -INT32_C(  2068061384),  INT32_C(   783044769), -INT32_C(  1362803848) },
      { -INT32_C(   173470198), -INT32_C(  1542383902), -INT32_C(    56201355), -INT32_C(   769664208), -INT32_C(     2945765),  INT32_C(   579491236),  INT32_C(   664125004), -INT32_C(  1751701363),
         INT32_C(   411844662), -INT32_C(   860054186),  INT32_C(  1036542733),  INT32_C(  1494279998), -INT32_C(  1722162187), -INT32_C(  2068061384),  INT32_C(   783044769), -INT32_C(  1362803848) } },
    { { -INT32_C(   624471420), -INT32_C(    56196113),  INT32_C(   607809254),  INT32_C(  1266567766),  INT32_C(  1709496109),  INT32_C(  1558880186),  INT32_C(  1737135855),  INT32_C(  1561678041),
        -INT32_C(  1858544478),  INT32_C(  1183768160), -INT32_C(  1553217459),  INT32_C(  1072621842),  INT32_C(  2057622208),  INT32_C(  1624673905), -INT32_C(    20487900),  INT32_C(  1398529201) },
      { -INT32_C(   624471420), -INT32_C(    56196113),  INT32_C(   607809254),  INT32_C(  1266567766),  INT32_C(  1709496109),  INT32_C(  1558880186),  INT32_C(  1737135855),  INT32_C(  1561678041),
        -INT32_C(  1858544478),  INT32_C(  1183768160), -INT32_C(  1553217459),  INT32_C(  1072621842),  INT32_C(  2057622208),  INT32_C(  1624673905), -INT32_C(    20487900),  INT32_C(  1398529201) } },
    { {  INT32_C(   434410425), -INT32_C(  1084263822),  INT32_C(  1281542714),  INT32_C(  1938510003), -INT32_C(  1813106654), -INT32_C(   470563650), -INT32_C(   689849819),  INT32_C(  1328102550),
         INT32_C(  1114115792), -INT32_C(  1157511040),  INT32_C(  1174889362), -INT32_C(   709258317), -INT32_C(  2123847741), -INT32_C(  1855693972), -INT32_C(  1419229931),  INT32_C(  1392218498) },
      {  INT32_C(   434410425), -INT32_C(  1084263822),  INT32_C(  1281542714),  INT32_C(  1938510003), -INT32_C(  1813106654), -INT32_C(   470563650), -INT32_C(   689849819),  INT32_C(  1328102550),
         INT32_C(  1114115792), -INT32_C(  1157511040),  INT32_C(  1174889362), -INT32_C(   709258317), -INT32_C(  2123847741), -INT32_C(  1855693972), -INT32_C(  1419229931),  INT32_C(  1392218498) } },
    { {  INT32_C(   546595743), -INT32_C(  1092905685), -INT32_C(  1425743112),  INT32_C(   947961205), -INT32_C(   776279963),  INT32_C(  1482825283), -INT32_C(   435959196), -INT32_C(    80150948),
        -INT32_C(  1927558046),  INT32_C(  1498150497),  INT32_C(  1308905433),  INT32_C(  1921483789), -INT32_C(  1354546836), -INT32_C(  1022909089), -INT32_C(   861336976),  INT32_C(  1808261385) },
      {  INT32_C(   546595743), -INT32_C(  1092905685), -INT32_C(  1425743112),  INT32_C(   947961205), -INT32_C(   776279963),  INT32_C(  1482825283), -INT32_C(   435959196), -INT32_C(    80150948),
        -INT32_C(  1927558046),  INT32_C(  1498150497),  INT32_C(  1308905433),  INT32_C(  1921483789), -INT32_C(  1354546836), -INT32_C(  1022909089), -INT32_C(   861336976),  INT32_C(  1808261385) } },
    { {  INT32_C(   251192237), -INT32_C(  1301855015), -INT32_C(  1610519661),  INT32_C(  1527941359),  INT32_C(   671765961),  INT32_C(  1810633211),  INT32_C(   624399644),  INT32_C(   613482103),
        -INT32_C(  1154250527),  INT32_C(  1617795788), -INT32_C(   184521210), -INT32_C(  1085205514),  INT32_C(  1676172136), -INT32_C(  1982933907), -INT32_C(   525466263), -INT32_C(   452641276) },
      {  INT32_C(   251192237), -INT32_C(  1301855015), -INT32_C(  1610519661),  INT32_C(  1527941359),  INT32_C(   671765961),  INT32_C(  1810633211),  INT32_C(   624399644),  INT32_C(   613482103),
        -INT32_C(  1154250527),  INT32_C(  1617795788), -INT32_C(   184521210), -INT32_C(  1085205514),  INT32_C(  1676172136), -INT32_C(  1982933907), -INT32_C(   525466263), -INT32_C(   452641276) } },
    { { -INT32_C(  1818216250), -INT32_C(   655159598),  INT32_C(  1942942588),  INT32_C(  1865555718), -INT32_C(   405661062),  INT32_C(  1483776494), -INT32_C(  1439162714),  INT32_C(   596655452),
         INT32_C(  1219899509), -INT32_C(  1155487426), -INT32_C(  1557205348), -INT32_C(  2012061683),  INT32_C(  1768940667),  INT32_C(   750903429),  INT32_C(  1540815614), -INT32_C(  1384225225) },
      { -INT32_C(  1818216250), -INT32_C(   655159598),  INT32_C(  1942942588),  INT32_C(  1865555718), -INT32_C(   405661062),  INT32_C(  1483776494), -INT32_C(  1439162714),  INT32_C(   596655452),
         INT32_C(  1219899509), -INT32_C(  1155487426), -INT32_C(  1557205348), -INT32_C(  2012061683),  INT32_C(  1768940667),  INT32_C(   750903429),  INT32_C(  1540815614), -INT32_C(  1384225225) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_test_x86_assert_equal_i32x16(easysimd_mm512_load_si512(test_vec[i].a), easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = a;

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_loadu_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_64(easysimd__m512i) const int64_t a[8];
    const int64_t r[8];
  } test_vec[] = {
    { { -INT64_C( 2031689546876671122), -INT64_C( 4181824216786536295),  INT64_C( 3378378289711491617),  INT64_C( 1413316256384642707),
        -INT64_C( 4698950795030485050),  INT64_C( 7475600748512024817), -INT64_C( 2293462173326056235),  INT64_C( 2603685720003003242) },
      { -INT64_C( 2031689546876671122), -INT64_C( 4181824216786536295),  INT64_C( 3378378289711491617),  INT64_C( 1413316256384642707),
        -INT64_C( 4698950795030485050),  INT64_C( 7475600748512024817), -INT64_C( 2293462173326056235),  INT64_C( 2603685720003003242) } },
    { {  INT64_C( 4049249843808980558),  INT64_C( 7392641815426491883),  INT64_C( 1801878204460544724), -INT64_C(   48330471621752111),
         INT64_C( 2024547467117354649),  INT64_C( 1044804576756910729),  INT64_C( 4782031642370761366),  INT64_C( 7668159918304822970) },
      {  INT64_C( 4049249843808980558),  INT64_C( 7392641815426491883),  INT64_C( 1801878204460544724), -INT64_C(   48330471621752111),
         INT64_C( 2024547467117354649),  INT64_C( 1044804576756910729),  INT64_C( 4782031642370761366),  INT64_C( 7668159918304822970) } },
    { {  INT64_C(  680433322035960868),  INT64_C( 4032026382637907372),  INT64_C( 1024807869850854276), -INT64_C( 7738621839182026145),
        -INT64_C(  916101787114937152), -INT64_C( 7858554787118552041), -INT64_C( 6533667226337645326),  INT64_C( 8173594282907061610) },
      {  INT64_C(  680433322035960868),  INT64_C( 4032026382637907372),  INT64_C( 1024807869850854276), -INT64_C( 7738621839182026145),
        -INT64_C(  916101787114937152), -INT64_C( 7858554787118552041), -INT64_C( 6533667226337645326),  INT64_C( 8173594282907061610) } },
    { { -INT64_C( 3994697604197623979),  INT64_C( 3028796336221808999), -INT64_C( 4986958888383311650),  INT64_C( 7327921812528210064),
        -INT64_C( 4048013273381903271), -INT64_C( 6603326236083268358),  INT64_C( 2296716578005830869), -INT64_C( 3555290135981427917) },
      { -INT64_C( 3994697604197623979),  INT64_C( 3028796336221808999), -INT64_C( 4986958888383311650),  INT64_C( 7327921812528210064),
        -INT64_C( 4048013273381903271), -INT64_C( 6603326236083268358),  INT64_C( 2296716578005830869), -INT64_C( 3555290135981427917) } },
    { {  INT64_C( 3935770298369485431), -INT64_C(  762861917337756674), -INT64_C(  558453203728190831),  INT64_C( 2931813335080607596),
         INT64_C( 3458938454811838351), -INT64_C( 4643389136534410887),  INT64_C( 1174628764682791568), -INT64_C(  245550163283572547) },
      {  INT64_C( 3935770298369485431), -INT64_C(  762861917337756674), -INT64_C(  558453203728190831),  INT64_C( 2931813335080607596),
         INT64_C( 3458938454811838351), -INT64_C( 4643389136534410887),  INT64_C( 1174628764682791568), -INT64_C(  245550163283572547) } },
    { { -INT64_C( 5869378661672118744),  INT64_C( 7934735468561203248), -INT64_C( 2939425477300585343), -INT64_C( 1152397282285115752),
        -INT64_C( 4659583426481174413), -INT64_C(  675194194085700267),  INT64_C(  947814707075179574), -INT64_C( 4886946240843846537) },
      { -INT64_C( 5869378661672118744),  INT64_C( 7934735468561203248), -INT64_C( 2939425477300585343), -INT64_C( 1152397282285115752),
        -INT64_C( 4659583426481174413), -INT64_C(  675194194085700267),  INT64_C(  947814707075179574), -INT64_C( 4886946240843846537) } },
    { { -INT64_C( 6617272956007253540), -INT64_C( 5205464620909246634), -INT64_C( 6530450158184309283),  INT64_C( 7034361509239288218),
         INT64_C( 1332492355739845515), -INT64_C( 6091786111122778819),  INT64_C( 4994593874853592189), -INT64_C( 3335539744629574450) },
      { -INT64_C( 6617272956007253540), -INT64_C( 5205464620909246634), -INT64_C( 6530450158184309283),  INT64_C( 7034361509239288218),
         INT64_C( 1332492355739845515), -INT64_C( 6091786111122778819),  INT64_C( 4994593874853592189), -INT64_C( 3335539744629574450) } },
    { { -INT64_C( 5361209383270579765), -INT64_C( 7640663431528024195),  INT64_C( 2185812967214347366),  INT64_C( 1286946775314366149),
         INT64_C( 3158766812587919016), -INT64_C( 7397886743846434135), -INT64_C( 1382324539653187999),  INT64_C( 1284884244920222333) },
      { -INT64_C( 5361209383270579765), -INT64_C( 7640663431528024195),  INT64_C( 2185812967214347366),  INT64_C( 1286946775314366149),
         INT64_C( 3158766812587919016), -INT64_C( 7397886743846434135), -INT64_C( 1382324539653187999),  INT64_C( 1284884244920222333) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_test_x86_assert_equal_i64x8(easysimd_mm512_load_si512(test_vec[i].a), easysimd_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = a;

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_loadu_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const float a[16];
    const float r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(    -5.44), EASYSIMD_FLOAT32_C(    95.29), EASYSIMD_FLOAT32_C(    15.56), EASYSIMD_FLOAT32_C(   -23.83),
        EASYSIMD_FLOAT32_C(   -48.46), EASYSIMD_FLOAT32_C(    53.99), EASYSIMD_FLOAT32_C(    86.26), EASYSIMD_FLOAT32_C(    13.50),
        EASYSIMD_FLOAT32_C(   -29.49), EASYSIMD_FLOAT32_C(   -86.28), EASYSIMD_FLOAT32_C(    33.58), EASYSIMD_FLOAT32_C(   -45.35),
        EASYSIMD_FLOAT32_C(    81.12), EASYSIMD_FLOAT32_C(   -58.19), EASYSIMD_FLOAT32_C(    43.52), EASYSIMD_FLOAT32_C(   -12.09) },
      { EASYSIMD_FLOAT32_C(    -5.44), EASYSIMD_FLOAT32_C(    95.29), EASYSIMD_FLOAT32_C(    15.56), EASYSIMD_FLOAT32_C(   -23.83),
        EASYSIMD_FLOAT32_C(   -48.46), EASYSIMD_FLOAT32_C(    53.99), EASYSIMD_FLOAT32_C(    86.26), EASYSIMD_FLOAT32_C(    13.50),
        EASYSIMD_FLOAT32_C(   -29.49), EASYSIMD_FLOAT32_C(   -86.28), EASYSIMD_FLOAT32_C(    33.58), EASYSIMD_FLOAT32_C(   -45.35),
        EASYSIMD_FLOAT32_C(    81.12), EASYSIMD_FLOAT32_C(   -58.19), EASYSIMD_FLOAT32_C(    43.52), EASYSIMD_FLOAT32_C(   -12.09) } },
    { { EASYSIMD_FLOAT32_C(    59.25), EASYSIMD_FLOAT32_C(    98.23), EASYSIMD_FLOAT32_C(    38.09), EASYSIMD_FLOAT32_C(    23.01),
        EASYSIMD_FLOAT32_C(    90.49), EASYSIMD_FLOAT32_C(    33.17), EASYSIMD_FLOAT32_C(   -24.27), EASYSIMD_FLOAT32_C(   -65.07),
        EASYSIMD_FLOAT32_C(    32.10), EASYSIMD_FLOAT32_C(   -17.33), EASYSIMD_FLOAT32_C(    85.73), EASYSIMD_FLOAT32_C(   -82.76),
        EASYSIMD_FLOAT32_C(   -35.98), EASYSIMD_FLOAT32_C(    36.36), EASYSIMD_FLOAT32_C(   -31.27), EASYSIMD_FLOAT32_C(    58.57) },
      { EASYSIMD_FLOAT32_C(    59.25), EASYSIMD_FLOAT32_C(    98.23), EASYSIMD_FLOAT32_C(    38.09), EASYSIMD_FLOAT32_C(    23.01),
        EASYSIMD_FLOAT32_C(    90.49), EASYSIMD_FLOAT32_C(    33.17), EASYSIMD_FLOAT32_C(   -24.27), EASYSIMD_FLOAT32_C(   -65.07),
        EASYSIMD_FLOAT32_C(    32.10), EASYSIMD_FLOAT32_C(   -17.33), EASYSIMD_FLOAT32_C(    85.73), EASYSIMD_FLOAT32_C(   -82.76),
        EASYSIMD_FLOAT32_C(   -35.98), EASYSIMD_FLOAT32_C(    36.36), EASYSIMD_FLOAT32_C(   -31.27), EASYSIMD_FLOAT32_C(    58.57) } },
    { { EASYSIMD_FLOAT32_C(    31.65), EASYSIMD_FLOAT32_C(    84.29), EASYSIMD_FLOAT32_C(   -65.26), EASYSIMD_FLOAT32_C(    83.19),
        EASYSIMD_FLOAT32_C(    38.27), EASYSIMD_FLOAT32_C(   -78.99), EASYSIMD_FLOAT32_C(    -3.31), EASYSIMD_FLOAT32_C(   -91.22),
        EASYSIMD_FLOAT32_C(   -65.28), EASYSIMD_FLOAT32_C(   -69.73), EASYSIMD_FLOAT32_C(   -36.56), EASYSIMD_FLOAT32_C(   -84.16),
        EASYSIMD_FLOAT32_C(   -27.92), EASYSIMD_FLOAT32_C(   -93.04), EASYSIMD_FLOAT32_C(     3.75), EASYSIMD_FLOAT32_C(   -68.66) },
      { EASYSIMD_FLOAT32_C(    31.65), EASYSIMD_FLOAT32_C(    84.29), EASYSIMD_FLOAT32_C(   -65.26), EASYSIMD_FLOAT32_C(    83.19),
        EASYSIMD_FLOAT32_C(    38.27), EASYSIMD_FLOAT32_C(   -78.99), EASYSIMD_FLOAT32_C(    -3.31), EASYSIMD_FLOAT32_C(   -91.22),
        EASYSIMD_FLOAT32_C(   -65.28), EASYSIMD_FLOAT32_C(   -69.73), EASYSIMD_FLOAT32_C(   -36.56), EASYSIMD_FLOAT32_C(   -84.16),
        EASYSIMD_FLOAT32_C(   -27.92), EASYSIMD_FLOAT32_C(   -93.04), EASYSIMD_FLOAT32_C(     3.75), EASYSIMD_FLOAT32_C(   -68.66) } },
    { { EASYSIMD_FLOAT32_C(   -94.82), EASYSIMD_FLOAT32_C(   -58.15), EASYSIMD_FLOAT32_C(    54.35), EASYSIMD_FLOAT32_C(    95.68),
        EASYSIMD_FLOAT32_C(    75.02), EASYSIMD_FLOAT32_C(   -69.92), EASYSIMD_FLOAT32_C(   -69.39), EASYSIMD_FLOAT32_C(     7.12),
        EASYSIMD_FLOAT32_C(    12.75), EASYSIMD_FLOAT32_C(   -83.66), EASYSIMD_FLOAT32_C(    24.36), EASYSIMD_FLOAT32_C(    76.77),
        EASYSIMD_FLOAT32_C(    52.70), EASYSIMD_FLOAT32_C(    93.09), EASYSIMD_FLOAT32_C(    35.34), EASYSIMD_FLOAT32_C(   -15.65) },
      { EASYSIMD_FLOAT32_C(   -94.82), EASYSIMD_FLOAT32_C(   -58.15), EASYSIMD_FLOAT32_C(    54.35), EASYSIMD_FLOAT32_C(    95.68),
        EASYSIMD_FLOAT32_C(    75.02), EASYSIMD_FLOAT32_C(   -69.92), EASYSIMD_FLOAT32_C(   -69.39), EASYSIMD_FLOAT32_C(     7.12),
        EASYSIMD_FLOAT32_C(    12.75), EASYSIMD_FLOAT32_C(   -83.66), EASYSIMD_FLOAT32_C(    24.36), EASYSIMD_FLOAT32_C(    76.77),
        EASYSIMD_FLOAT32_C(    52.70), EASYSIMD_FLOAT32_C(    93.09), EASYSIMD_FLOAT32_C(    35.34), EASYSIMD_FLOAT32_C(   -15.65) } },
    { { EASYSIMD_FLOAT32_C(    77.38), EASYSIMD_FLOAT32_C(    70.08), EASYSIMD_FLOAT32_C(   -32.46), EASYSIMD_FLOAT32_C(    15.66),
        EASYSIMD_FLOAT32_C(    91.09), EASYSIMD_FLOAT32_C(    64.24), EASYSIMD_FLOAT32_C(    24.44), EASYSIMD_FLOAT32_C(   -74.19),
        EASYSIMD_FLOAT32_C(    94.51), EASYSIMD_FLOAT32_C(    87.88), EASYSIMD_FLOAT32_C(   -58.34), EASYSIMD_FLOAT32_C(   -33.41),
        EASYSIMD_FLOAT32_C(    94.84), EASYSIMD_FLOAT32_C(    45.41), EASYSIMD_FLOAT32_C(    -2.07), EASYSIMD_FLOAT32_C(   -99.98) },
      { EASYSIMD_FLOAT32_C(    77.38), EASYSIMD_FLOAT32_C(    70.08), EASYSIMD_FLOAT32_C(   -32.46), EASYSIMD_FLOAT32_C(    15.66),
        EASYSIMD_FLOAT32_C(    91.09), EASYSIMD_FLOAT32_C(    64.24), EASYSIMD_FLOAT32_C(    24.44), EASYSIMD_FLOAT32_C(   -74.19),
        EASYSIMD_FLOAT32_C(    94.51), EASYSIMD_FLOAT32_C(    87.88), EASYSIMD_FLOAT32_C(   -58.34), EASYSIMD_FLOAT32_C(   -33.41),
        EASYSIMD_FLOAT32_C(    94.84), EASYSIMD_FLOAT32_C(    45.41), EASYSIMD_FLOAT32_C(    -2.07), EASYSIMD_FLOAT32_C(   -99.98) } },
    { { EASYSIMD_FLOAT32_C(    87.26), EASYSIMD_FLOAT32_C(   -47.72), EASYSIMD_FLOAT32_C(    95.70), EASYSIMD_FLOAT32_C(    62.28),
        EASYSIMD_FLOAT32_C(   -17.64), EASYSIMD_FLOAT32_C(   -73.69), EASYSIMD_FLOAT32_C(   -30.60), EASYSIMD_FLOAT32_C(    95.11),
        EASYSIMD_FLOAT32_C(   -57.36), EASYSIMD_FLOAT32_C(    93.76), EASYSIMD_FLOAT32_C(    71.88), EASYSIMD_FLOAT32_C(    95.34),
        EASYSIMD_FLOAT32_C(    86.86), EASYSIMD_FLOAT32_C(     7.22), EASYSIMD_FLOAT32_C(   -20.31), EASYSIMD_FLOAT32_C(    64.24) },
      { EASYSIMD_FLOAT32_C(    87.26), EASYSIMD_FLOAT32_C(   -47.72), EASYSIMD_FLOAT32_C(    95.70), EASYSIMD_FLOAT32_C(    62.28),
        EASYSIMD_FLOAT32_C(   -17.64), EASYSIMD_FLOAT32_C(   -73.69), EASYSIMD_FLOAT32_C(   -30.60), EASYSIMD_FLOAT32_C(    95.11),
        EASYSIMD_FLOAT32_C(   -57.36), EASYSIMD_FLOAT32_C(    93.76), EASYSIMD_FLOAT32_C(    71.88), EASYSIMD_FLOAT32_C(    95.34),
        EASYSIMD_FLOAT32_C(    86.86), EASYSIMD_FLOAT32_C(     7.22), EASYSIMD_FLOAT32_C(   -20.31), EASYSIMD_FLOAT32_C(    64.24) } },
    { { EASYSIMD_FLOAT32_C(   -22.70), EASYSIMD_FLOAT32_C(    47.23), EASYSIMD_FLOAT32_C(   -20.10), EASYSIMD_FLOAT32_C(   -31.61),
        EASYSIMD_FLOAT32_C(    11.47), EASYSIMD_FLOAT32_C(   -95.66), EASYSIMD_FLOAT32_C(    -5.80), EASYSIMD_FLOAT32_C(     5.98),
        EASYSIMD_FLOAT32_C(    92.22), EASYSIMD_FLOAT32_C(    35.86), EASYSIMD_FLOAT32_C(    72.57), EASYSIMD_FLOAT32_C(    87.05),
        EASYSIMD_FLOAT32_C(   -18.73), EASYSIMD_FLOAT32_C(   -29.51), EASYSIMD_FLOAT32_C(    87.07), EASYSIMD_FLOAT32_C(   -31.48) },
      { EASYSIMD_FLOAT32_C(   -22.70), EASYSIMD_FLOAT32_C(    47.23), EASYSIMD_FLOAT32_C(   -20.10), EASYSIMD_FLOAT32_C(   -31.61),
        EASYSIMD_FLOAT32_C(    11.47), EASYSIMD_FLOAT32_C(   -95.66), EASYSIMD_FLOAT32_C(    -5.80), EASYSIMD_FLOAT32_C(     5.98),
        EASYSIMD_FLOAT32_C(    92.22), EASYSIMD_FLOAT32_C(    35.86), EASYSIMD_FLOAT32_C(    72.57), EASYSIMD_FLOAT32_C(    87.05),
        EASYSIMD_FLOAT32_C(   -18.73), EASYSIMD_FLOAT32_C(   -29.51), EASYSIMD_FLOAT32_C(    87.07), EASYSIMD_FLOAT32_C(   -31.48) } },
    { { EASYSIMD_FLOAT32_C(    22.77), EASYSIMD_FLOAT32_C(    82.77), EASYSIMD_FLOAT32_C(   -69.20), EASYSIMD_FLOAT32_C(   -94.87),
        EASYSIMD_FLOAT32_C(   -90.93), EASYSIMD_FLOAT32_C(     0.20), EASYSIMD_FLOAT32_C(   -99.76), EASYSIMD_FLOAT32_C(   -48.28),
        EASYSIMD_FLOAT32_C(    -6.04), EASYSIMD_FLOAT32_C(    72.11), EASYSIMD_FLOAT32_C(   -52.94), EASYSIMD_FLOAT32_C(   -19.18),
        EASYSIMD_FLOAT32_C(   -20.67), EASYSIMD_FLOAT32_C(    26.75), EASYSIMD_FLOAT32_C(   -54.95), EASYSIMD_FLOAT32_C(    56.63) },
      { EASYSIMD_FLOAT32_C(    22.77), EASYSIMD_FLOAT32_C(    82.77), EASYSIMD_FLOAT32_C(   -69.20), EASYSIMD_FLOAT32_C(   -94.87),
        EASYSIMD_FLOAT32_C(   -90.93), EASYSIMD_FLOAT32_C(     0.20), EASYSIMD_FLOAT32_C(   -99.76), EASYSIMD_FLOAT32_C(   -48.28),
        EASYSIMD_FLOAT32_C(    -6.04), EASYSIMD_FLOAT32_C(    72.11), EASYSIMD_FLOAT32_C(   -52.94), EASYSIMD_FLOAT32_C(   -19.18),
        EASYSIMD_FLOAT32_C(   -20.67), EASYSIMD_FLOAT32_C(    26.75), EASYSIMD_FLOAT32_C(   -54.95), EASYSIMD_FLOAT32_C(    56.63) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    void const * pa = (void *)test_vec[i].r;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_loadu_ps(pa);
    }
    EASYSIMD_TEST_PERF_END("_mm512_loadu_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_load_ps((void *)test_vec[i].r), 2);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(-100.0, 100.0);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_loadu_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const double a[8];
    const double r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(    49.26), EASYSIMD_FLOAT64_C(    24.53), EASYSIMD_FLOAT64_C(   -29.29), EASYSIMD_FLOAT64_C(    19.12),
        EASYSIMD_FLOAT64_C(   -44.61), EASYSIMD_FLOAT64_C(   -21.37), EASYSIMD_FLOAT64_C(    89.17), EASYSIMD_FLOAT64_C(    74.97) },
      { EASYSIMD_FLOAT64_C(    49.26), EASYSIMD_FLOAT64_C(    24.53), EASYSIMD_FLOAT64_C(   -29.29), EASYSIMD_FLOAT64_C(    19.12),
        EASYSIMD_FLOAT64_C(   -44.61), EASYSIMD_FLOAT64_C(   -21.37), EASYSIMD_FLOAT64_C(    89.17), EASYSIMD_FLOAT64_C(    74.97) } },
    { { EASYSIMD_FLOAT64_C(   -46.12), EASYSIMD_FLOAT64_C(    73.96), EASYSIMD_FLOAT64_C(    31.37), EASYSIMD_FLOAT64_C(   -77.50),
        EASYSIMD_FLOAT64_C(    97.43), EASYSIMD_FLOAT64_C(    22.13), EASYSIMD_FLOAT64_C(   -96.75), EASYSIMD_FLOAT64_C(    16.57) },
      { EASYSIMD_FLOAT64_C(   -46.12), EASYSIMD_FLOAT64_C(    73.96), EASYSIMD_FLOAT64_C(    31.37), EASYSIMD_FLOAT64_C(   -77.50),
        EASYSIMD_FLOAT64_C(    97.43), EASYSIMD_FLOAT64_C(    22.13), EASYSIMD_FLOAT64_C(   -96.75), EASYSIMD_FLOAT64_C(    16.57) } },
    { { EASYSIMD_FLOAT64_C(    -0.37), EASYSIMD_FLOAT64_C(   -87.28), EASYSIMD_FLOAT64_C(    -6.48), EASYSIMD_FLOAT64_C(   -84.24),
        EASYSIMD_FLOAT64_C(   -13.21), EASYSIMD_FLOAT64_C(    67.54), EASYSIMD_FLOAT64_C(    86.68), EASYSIMD_FLOAT64_C(   -50.10) },
      { EASYSIMD_FLOAT64_C(    -0.37), EASYSIMD_FLOAT64_C(   -87.28), EASYSIMD_FLOAT64_C(    -6.48), EASYSIMD_FLOAT64_C(   -84.24),
        EASYSIMD_FLOAT64_C(   -13.21), EASYSIMD_FLOAT64_C(    67.54), EASYSIMD_FLOAT64_C(    86.68), EASYSIMD_FLOAT64_C(   -50.10) } },
    { { EASYSIMD_FLOAT64_C(    83.91), EASYSIMD_FLOAT64_C(   -23.33), EASYSIMD_FLOAT64_C(    67.47), EASYSIMD_FLOAT64_C(    19.78),
        EASYSIMD_FLOAT64_C(    91.23), EASYSIMD_FLOAT64_C(    72.34), EASYSIMD_FLOAT64_C(    27.16), EASYSIMD_FLOAT64_C(    40.48) },
      { EASYSIMD_FLOAT64_C(    83.91), EASYSIMD_FLOAT64_C(   -23.33), EASYSIMD_FLOAT64_C(    67.47), EASYSIMD_FLOAT64_C(    19.78),
        EASYSIMD_FLOAT64_C(    91.23), EASYSIMD_FLOAT64_C(    72.34), EASYSIMD_FLOAT64_C(    27.16), EASYSIMD_FLOAT64_C(    40.48) } },
    { { EASYSIMD_FLOAT64_C(    -3.13), EASYSIMD_FLOAT64_C(    97.86), EASYSIMD_FLOAT64_C(   -40.39), EASYSIMD_FLOAT64_C(    52.26),
        EASYSIMD_FLOAT64_C(   -23.51), EASYSIMD_FLOAT64_C(   -51.23), EASYSIMD_FLOAT64_C(    27.24), EASYSIMD_FLOAT64_C(    30.38) },
      { EASYSIMD_FLOAT64_C(    -3.13), EASYSIMD_FLOAT64_C(    97.86), EASYSIMD_FLOAT64_C(   -40.39), EASYSIMD_FLOAT64_C(    52.26),
        EASYSIMD_FLOAT64_C(   -23.51), EASYSIMD_FLOAT64_C(   -51.23), EASYSIMD_FLOAT64_C(    27.24), EASYSIMD_FLOAT64_C(    30.38) } },
    { { EASYSIMD_FLOAT64_C(   -77.26), EASYSIMD_FLOAT64_C(   -41.39), EASYSIMD_FLOAT64_C(    52.88), EASYSIMD_FLOAT64_C(   -79.84),
        EASYSIMD_FLOAT64_C(    80.74), EASYSIMD_FLOAT64_C(    56.14), EASYSIMD_FLOAT64_C(    36.73), EASYSIMD_FLOAT64_C(   -19.63) },
      { EASYSIMD_FLOAT64_C(   -77.26), EASYSIMD_FLOAT64_C(   -41.39), EASYSIMD_FLOAT64_C(    52.88), EASYSIMD_FLOAT64_C(   -79.84),
        EASYSIMD_FLOAT64_C(    80.74), EASYSIMD_FLOAT64_C(    56.14), EASYSIMD_FLOAT64_C(    36.73), EASYSIMD_FLOAT64_C(   -19.63) } },
    { { EASYSIMD_FLOAT64_C(    68.85), EASYSIMD_FLOAT64_C(   -69.75), EASYSIMD_FLOAT64_C(    -3.87), EASYSIMD_FLOAT64_C(   -44.36),
        EASYSIMD_FLOAT64_C(    97.79), EASYSIMD_FLOAT64_C(   -17.19), EASYSIMD_FLOAT64_C(     5.54), EASYSIMD_FLOAT64_C(    81.69) },
      { EASYSIMD_FLOAT64_C(    68.85), EASYSIMD_FLOAT64_C(   -69.75), EASYSIMD_FLOAT64_C(    -3.87), EASYSIMD_FLOAT64_C(   -44.36),
        EASYSIMD_FLOAT64_C(    97.79), EASYSIMD_FLOAT64_C(   -17.19), EASYSIMD_FLOAT64_C(     5.54), EASYSIMD_FLOAT64_C(    81.69) } },
    { { EASYSIMD_FLOAT64_C(    59.48), EASYSIMD_FLOAT64_C(   -26.99), EASYSIMD_FLOAT64_C(     1.47), EASYSIMD_FLOAT64_C(    50.71),
        EASYSIMD_FLOAT64_C(   -54.65), EASYSIMD_FLOAT64_C(   -71.37), EASYSIMD_FLOAT64_C(    -8.81), EASYSIMD_FLOAT64_C(    42.22) },
      { EASYSIMD_FLOAT64_C(    59.48), EASYSIMD_FLOAT64_C(   -26.99), EASYSIMD_FLOAT64_C(     1.47), EASYSIMD_FLOAT64_C(    50.71),
        EASYSIMD_FLOAT64_C(   -54.65), EASYSIMD_FLOAT64_C(   -71.37), EASYSIMD_FLOAT64_C(    -8.81), EASYSIMD_FLOAT64_C(    42.22) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    void const * pa = (void *)test_vec[i].r;
    easysimd__m512d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_loadu_pd(pa);
    }
    EASYSIMD_TEST_PERF_END("_mm512_loadu_pd");
    easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_load_pd((void *)test_vec[i].r), 2);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(-100.0, 100.0);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_loadu_si512(EASYSIMD_MUNIT_TEST_ARGS) {
  const struct {
    easysimd__m512i a;
    easysimd__m512i r;
  } test_vec[8] = {
    { easysimd_x_mm512_set_epu32(UINT32_C(2465927924), UINT32_C(3593197775), UINT32_C( 612910812), UINT32_C(3812769805),
                              UINT32_C(4149829677), UINT32_C(3483799324), UINT32_C(1459962882), UINT32_C(4149819515),
                              UINT32_C(2650201844), UINT32_C( 758753621), UINT32_C(1440172455), UINT32_C(1093653043),
                              UINT32_C(4135659774), UINT32_C(2249292246), UINT32_C(3926208727), UINT32_C( 363227362)),
      easysimd_x_mm512_set_epu32(UINT32_C(2465927924), UINT32_C(3593197775), UINT32_C( 612910812), UINT32_C(3812769805),
                              UINT32_C(4149829677), UINT32_C(3483799324), UINT32_C(1459962882), UINT32_C(4149819515),
                              UINT32_C(2650201844), UINT32_C( 758753621), UINT32_C(1440172455), UINT32_C(1093653043),
                              UINT32_C(4135659774), UINT32_C(2249292246), UINT32_C(3926208727), UINT32_C( 363227362)) },
    { easysimd_x_mm512_set_epu32(UINT32_C( 468967701), UINT32_C(1464888328), UINT32_C(2623912787), UINT32_C(3584306317),
                              UINT32_C(3441172772), UINT32_C(1957813224), UINT32_C(3956090282), UINT32_C(2819645236),
                              UINT32_C(2119397630), UINT32_C(3325357179), UINT32_C( 910080153), UINT32_C(3698201489),
                              UINT32_C(3945376801), UINT32_C(2699586726), UINT32_C(1169343086), UINT32_C(2983594096)),
      easysimd_x_mm512_set_epu32(UINT32_C( 468967701), UINT32_C(1464888328), UINT32_C(2623912787), UINT32_C(3584306317),
                              UINT32_C(3441172772), UINT32_C(1957813224), UINT32_C(3956090282), UINT32_C(2819645236),
                              UINT32_C(2119397630), UINT32_C(3325357179), UINT32_C( 910080153), UINT32_C(3698201489),
                              UINT32_C(3945376801), UINT32_C(2699586726), UINT32_C(1169343086), UINT32_C(2983594096)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(3220925730), UINT32_C(4163700514), UINT32_C( 208162340), UINT32_C(  72282893),
                              UINT32_C(2784701415), UINT32_C(2960668076), UINT32_C(2280551509), UINT32_C( 511971347),
                              UINT32_C(3142311802), UINT32_C(3582165504), UINT32_C(3533175269), UINT32_C(3138584679),
                              UINT32_C(3117232701), UINT32_C(1582887517), UINT32_C(2957127939), UINT32_C(3388466484)),
      easysimd_x_mm512_set_epu32(UINT32_C(3220925730), UINT32_C(4163700514), UINT32_C( 208162340), UINT32_C(  72282893),
                              UINT32_C(2784701415), UINT32_C(2960668076), UINT32_C(2280551509), UINT32_C( 511971347),
                              UINT32_C(3142311802), UINT32_C(3582165504), UINT32_C(3533175269), UINT32_C(3138584679),
                              UINT32_C(3117232701), UINT32_C(1582887517), UINT32_C(2957127939), UINT32_C(3388466484)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(2382371522), UINT32_C(  66180421), UINT32_C(3915007092), UINT32_C(3548556152),
                              UINT32_C(3063171483), UINT32_C( 175336822), UINT32_C(2621074902), UINT32_C(2785523281),
                              UINT32_C(3351907467), UINT32_C(3611626580), UINT32_C(3274777282), UINT32_C(2819588991),
                              UINT32_C(4142757399), UINT32_C(3841212820), UINT32_C(1375549108), UINT32_C(3217099434)),
      easysimd_x_mm512_set_epu32(UINT32_C(2382371522), UINT32_C(  66180421), UINT32_C(3915007092), UINT32_C(3548556152),
                              UINT32_C(3063171483), UINT32_C( 175336822), UINT32_C(2621074902), UINT32_C(2785523281),
                              UINT32_C(3351907467), UINT32_C(3611626580), UINT32_C(3274777282), UINT32_C(2819588991),
                              UINT32_C(4142757399), UINT32_C(3841212820), UINT32_C(1375549108), UINT32_C(3217099434)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(1625945136), UINT32_C(  82950125), UINT32_C(3598722192), UINT32_C(2456005821),
                              UINT32_C(3054050921), UINT32_C(3350002014), UINT32_C(1546778759), UINT32_C(3175686900),
                              UINT32_C(3418645543), UINT32_C(1247476579), UINT32_C(2559569107), UINT32_C(3884223622),
                              UINT32_C(2206347705), UINT32_C(1195297710), UINT32_C(4206427691), UINT32_C(2187435296)),
      easysimd_x_mm512_set_epu32(UINT32_C(1625945136), UINT32_C(  82950125), UINT32_C(3598722192), UINT32_C(2456005821),
                              UINT32_C(3054050921), UINT32_C(3350002014), UINT32_C(1546778759), UINT32_C(3175686900),
                              UINT32_C(3418645543), UINT32_C(1247476579), UINT32_C(2559569107), UINT32_C(3884223622),
                              UINT32_C(2206347705), UINT32_C(1195297710), UINT32_C(4206427691), UINT32_C(2187435296)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(3055114510), UINT32_C( 314498376), UINT32_C( 259740532), UINT32_C(2845634146),
                              UINT32_C(3528445754), UINT32_C(1438308061), UINT32_C(1618483487), UINT32_C(4280155704),
                              UINT32_C(4191548278), UINT32_C( 955760205), UINT32_C(3071952989), UINT32_C(3353486020),
                              UINT32_C(3091053226), UINT32_C(2241572393), UINT32_C(3491849165), UINT32_C(2750648051)),
      easysimd_x_mm512_set_epu32(UINT32_C(3055114510), UINT32_C( 314498376), UINT32_C( 259740532), UINT32_C(2845634146),
                              UINT32_C(3528445754), UINT32_C(1438308061), UINT32_C(1618483487), UINT32_C(4280155704),
                              UINT32_C(4191548278), UINT32_C( 955760205), UINT32_C(3071952989), UINT32_C(3353486020),
                              UINT32_C(3091053226), UINT32_C(2241572393), UINT32_C(3491849165), UINT32_C(2750648051)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(2791699552), UINT32_C(1697626027), UINT32_C(3068022880), UINT32_C( 492436222),
                              UINT32_C(2413088982), UINT32_C(1530446668), UINT32_C(1370127960), UINT32_C(2402932897),
                              UINT32_C(4061542194), UINT32_C( 154485056), UINT32_C(3577835063), UINT32_C(3500138573),
                              UINT32_C(  48074834), UINT32_C(1773313389), UINT32_C(3571862316), UINT32_C(1059958902)),
      easysimd_x_mm512_set_epu32(UINT32_C(2791699552), UINT32_C(1697626027), UINT32_C(3068022880), UINT32_C( 492436222),
                              UINT32_C(2413088982), UINT32_C(1530446668), UINT32_C(1370127960), UINT32_C(2402932897),
                              UINT32_C(4061542194), UINT32_C( 154485056), UINT32_C(3577835063), UINT32_C(3500138573),
                              UINT32_C(  48074834), UINT32_C(1773313389), UINT32_C(3571862316), UINT32_C(1059958902)) },
    { easysimd_x_mm512_set_epu32(UINT32_C(1166001194), UINT32_C( 115042765), UINT32_C( 557502548), UINT32_C(2408114255),
                              UINT32_C( 870354895), UINT32_C( 955362708), UINT32_C(1149136654), UINT32_C(1920883489),
                              UINT32_C(3238897491), UINT32_C(1952390233), UINT32_C( 223001918), UINT32_C( 310736118),
                              UINT32_C(2747509005), UINT32_C( 134376306), UINT32_C(1234549716), UINT32_C( 594304164)),
      easysimd_x_mm512_set_epu32(UINT32_C(1166001194), UINT32_C( 115042765), UINT32_C( 557502548), UINT32_C(2408114255),
                              UINT32_C( 870354895), UINT32_C( 955362708), UINT32_C(1149136654), UINT32_C(1920883489),
                              UINT32_C(3238897491), UINT32_C(1952390233), UINT32_C( 223001918), UINT32_C( 310736118),
                              UINT32_C(2747509005), UINT32_C( 134376306), UINT32_C(1234549716), UINT32_C( 594304164)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = test_vec[i].a;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_loadu_si512(&a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_loadu_si512");
    easysimd_assert_m512i_i32(r, ==, test_vec[i].r);
  }

  return 0;
}

static int
test_easysimd_mm512_mask_loadu_epi8()
{
#if 1
  static const struct {
    const int8_t src[64];
    easysimd__mmask64 k;
    const int8_t a[64];
    const int8_t r[64];
  } test_vec[] = {
    { { -INT8_C(  65), -INT8_C( 111),  INT8_C(  80), -INT8_C(  80),  INT8_C(  73),  INT8_C(  80), -INT8_C( 114), -INT8_C(  16),
        -INT8_C(   6),  INT8_C( 109),  INT8_C(  67),  INT8_C( 118),  INT8_C(  84),  INT8_C(  33),  INT8_C(  13), -INT8_C(  20),
         INT8_C(  60),  INT8_C(  45),  INT8_C(  71), -INT8_C(   7),  INT8_C(  71), -INT8_C(  76), -INT8_C(  83),  INT8_C(  67),
         INT8_C(  87),  INT8_C( 112),  INT8_C(  61),  INT8_C(  70), -INT8_C(  14), -INT8_C(  90), -INT8_C( 125), -INT8_C(  79),
         INT8_C(  56), -INT8_C(  45),  INT8_C(  98), -INT8_C( 127),  INT8_C(  35), -INT8_C(  16),  INT8_C( 113),  INT8_C(  29),
         INT8_C(  94), -INT8_C(  76), -INT8_C( 109), -INT8_C(  78), -INT8_C(  43), -INT8_C(  96), -INT8_C(  98),  INT8_C(  18),
        -INT8_C(  50), -INT8_C(  26),  INT8_C(  11),  INT8_C(  21), -INT8_C( 102), -INT8_C(  72),  INT8_C(  88), -INT8_C(  15),
         INT8_C(  40), -INT8_C( 107),  INT8_C(  56),  INT8_C(  26),  INT8_C(  60), -INT8_C(  69), -INT8_C(  52),  INT8_C( 116) },
      UINT64_C( 8993237787926146702),
      {  INT8_C(  26),  INT8_C(  97),  INT8_C(  46), -INT8_C(  16),  INT8_C(   1), -INT8_C(  51),  INT8_C(   2), -INT8_C(  49),
        -INT8_C(  77),  INT8_C(  13), -INT8_C(  28),  INT8_C(  77), -INT8_C(  59),  INT8_C(  61),  INT8_C(  62), -INT8_C(  18),
        -INT8_C(  46),  INT8_C( 118),  INT8_C(   8),  INT8_C(  14),  INT8_C(  49), -INT8_C(  44), -INT8_C( 126), -INT8_C(  65),
         INT8_C(   2),  INT8_C( 119),  INT8_C( 112),  INT8_C(  33), -INT8_C(  35),  INT8_C(  62), -INT8_C(  99), -INT8_C(   8),
        -INT8_C(  97), -INT8_C(  52), -INT8_C(  24), -INT8_C(  95), -INT8_C( 103), -INT8_C(  22),  INT8_C( 112),  INT8_C(  76),
        -INT8_C(   9),  INT8_C(  85), -INT8_C( 103), -INT8_C(  68), -INT8_C( 110), -INT8_C(  41), -INT8_C(  86),  INT8_C( 100),
         INT8_C(  78), -INT8_C(  77),  INT8_C( 115),      INT8_MAX, -INT8_C( 121), -INT8_C(  11),  INT8_C(  63), -INT8_C( 118),
         INT8_C( 109), -INT8_C(  81), -INT8_C(  85),  INT8_C(  74), -INT8_C(  18),  INT8_C(  72),  INT8_C(  66), -INT8_C( 115) },
      { -INT8_C(  65),  INT8_C(  97),  INT8_C(  46), -INT8_C(  16),  INT8_C(  73),  INT8_C(  80), -INT8_C( 114), -INT8_C(  49),
        -INT8_C(   6),  INT8_C(  13), -INT8_C(  28),  INT8_C(  77),  INT8_C(  84),  INT8_C(  61),  INT8_C(  13), -INT8_C(  20),
        -INT8_C(  46),  INT8_C(  45),  INT8_C(   8), -INT8_C(   7),  INT8_C(  49), -INT8_C(  44), -INT8_C( 126), -INT8_C(  65),
         INT8_C(   2),  INT8_C( 112),  INT8_C(  61),  INT8_C(  70), -INT8_C(  35),  INT8_C(  62), -INT8_C( 125), -INT8_C(   8),
         INT8_C(  56), -INT8_C(  52), -INT8_C(  24), -INT8_C(  95), -INT8_C( 103), -INT8_C(  16),  INT8_C( 113),  INT8_C(  29),
         INT8_C(  94),  INT8_C(  85), -INT8_C( 103), -INT8_C(  78), -INT8_C(  43), -INT8_C(  41), -INT8_C(  86),  INT8_C(  18),
        -INT8_C(  50), -INT8_C(  77),  INT8_C( 115),      INT8_MAX, -INT8_C( 102), -INT8_C(  72),  INT8_C(  63), -INT8_C( 118),
         INT8_C(  40), -INT8_C( 107), -INT8_C(  85),  INT8_C(  74), -INT8_C(  18),  INT8_C(  72),  INT8_C(  66),  INT8_C( 116) } },
    { {  INT8_C(  20),  INT8_C(  42),  INT8_C(  46), -INT8_C(  83),  INT8_C(  20), -INT8_C(  97), -INT8_C(   7),  INT8_C(  11),
        -INT8_C(  12), -INT8_C( 110), -INT8_C(  56), -INT8_C( 122),  INT8_C( 106),  INT8_C( 114), -INT8_C(  22), -INT8_C(  72),
         INT8_C(  37),  INT8_C(  93),  INT8_C(  55), -INT8_C(  83),  INT8_C(  83),  INT8_C( 118),  INT8_C(  55), -INT8_C(  64),
         INT8_C(  38), -INT8_C(  30),  INT8_C(  10),  INT8_C(  20),  INT8_C(  42),  INT8_C(  77), -INT8_C(  95),  INT8_C(  63),
         INT8_C( 119), -INT8_C(  48), -INT8_C(  20), -INT8_C( 116),  INT8_C( 111), -INT8_C(  26), -INT8_C( 105),  INT8_C(  99),
         INT8_C( 120),  INT8_C(  95), -INT8_C(  23), -INT8_C(  30), -INT8_C(  46), -INT8_C(  45), -INT8_C( 102), -INT8_C(   9),
         INT8_C(  49), -INT8_C(  46), -INT8_C(  92), -INT8_C( 124),  INT8_C(  72), -INT8_C(  37),  INT8_C(  68),  INT8_C( 110),
        -INT8_C(  67),  INT8_C(  78), -INT8_C( 126), -INT8_C(  24), -INT8_C( 101),  INT8_C(  36),  INT8_C(  39),  INT8_C(  19) },
      UINT64_C( 8270358210458620916),
      { -INT8_C( 106), -INT8_C(  81),  INT8_C(  84),  INT8_C( 104), -INT8_C( 126), -INT8_C(  17),  INT8_C(  95), -INT8_C(  77),
        -INT8_C(  63),  INT8_C(   4),  INT8_C(  55),  INT8_C(   9), -INT8_C(  33),  INT8_C( 123),  INT8_C( 120), -INT8_C(  99),
        -INT8_C(  54), -INT8_C(   6), -INT8_C( 123),  INT8_C( 101),  INT8_C(  30), -INT8_C(  84),  INT8_C( 120),  INT8_C(  18),
        -INT8_C(  65),  INT8_C(  23),  INT8_C( 117), -INT8_C(  71),  INT8_C(  78),  INT8_C(  59),  INT8_C(  43), -INT8_C(  28),
        -INT8_C(  22),      INT8_MAX,  INT8_C(  76),  INT8_C( 109),  INT8_C( 110), -INT8_C(  85),  INT8_C(  32),  INT8_C(  47),
        -INT8_C(  81),  INT8_C(  88),  INT8_C(  57), -INT8_C( 113), -INT8_C(  45), -INT8_C(  79),  INT8_C(  44), -INT8_C(  99),
        -INT8_C(  85), -INT8_C(  79),  INT8_C(   3), -INT8_C(  54),  INT8_C(  93),  INT8_C( 123), -INT8_C(  36),  INT8_C(  28),
        -INT8_C( 109),  INT8_C(  82), -INT8_C(  43), -INT8_C(  31), -INT8_C( 115),  INT8_C(   0), -INT8_C(  59),  INT8_C( 120) },
      {  INT8_C(  20),  INT8_C(  42),  INT8_C(  84), -INT8_C(  83), -INT8_C( 126), -INT8_C(  17),  INT8_C(  95), -INT8_C(  77),
        -INT8_C(  63),  INT8_C(   4), -INT8_C(  56), -INT8_C( 122), -INT8_C(  33),  INT8_C( 114), -INT8_C(  22), -INT8_C(  72),
        -INT8_C(  54), -INT8_C(   6), -INT8_C( 123),  INT8_C( 101),  INT8_C(  30),  INT8_C( 118),  INT8_C(  55),  INT8_C(  18),
        -INT8_C(  65),  INT8_C(  23),  INT8_C(  10),  INT8_C(  20),  INT8_C(  42),  INT8_C(  59),  INT8_C(  43),  INT8_C(  63),
        -INT8_C(  22), -INT8_C(  48), -INT8_C(  20),  INT8_C( 109),  INT8_C( 110), -INT8_C(  85),  INT8_C(  32),  INT8_C(  47),
         INT8_C( 120),  INT8_C(  88),  INT8_C(  57), -INT8_C(  30), -INT8_C(  45), -INT8_C(  79), -INT8_C( 102), -INT8_C(   9),
         INT8_C(  49), -INT8_C(  79),  INT8_C(   3), -INT8_C( 124),  INT8_C(  72), -INT8_C(  37), -INT8_C(  36),  INT8_C(  28),
        -INT8_C(  67),  INT8_C(  82), -INT8_C( 126), -INT8_C(  24), -INT8_C( 115),  INT8_C(   0), -INT8_C(  59),  INT8_C(  19) } },
    { {      INT8_MIN,  INT8_C(  17), -INT8_C(  27), -INT8_C(  18), -INT8_C(  68),  INT8_C(   5),  INT8_C(  30),  INT8_C( 108),
         INT8_C(  93),  INT8_C(  87), -INT8_C(   5),  INT8_C(  49),  INT8_C(   8),  INT8_C(  39), -INT8_C(  50), -INT8_C(  77),
        -INT8_C(  40), -INT8_C(  47),  INT8_C( 125),  INT8_C(  53),  INT8_C(  77),  INT8_C(  90),  INT8_C(  81), -INT8_C(  32),
        -INT8_C(  84),  INT8_C(  39), -INT8_C(  63),  INT8_C(  57),  INT8_C(  39), -INT8_C( 122), -INT8_C(  79), -INT8_C(  89),
        -INT8_C( 105), -INT8_C( 106), -INT8_C( 106),  INT8_C(  83), -INT8_C( 100), -INT8_C(  76), -INT8_C(  65), -INT8_C(   7),
         INT8_C(  11), -INT8_C(  70),  INT8_C(  42),  INT8_C(  19), -INT8_C(  31), -INT8_C(   7), -INT8_C(  58), -INT8_C(  71),
        -INT8_C(  54),  INT8_C(  68), -INT8_C(  18),  INT8_C(  23), -INT8_C(  98),  INT8_C(  64), -INT8_C(   9),  INT8_C(  74),
         INT8_C( 103), -INT8_C(  72), -INT8_C( 125), -INT8_C( 114),  INT8_C(  62),  INT8_C(  53),  INT8_C(  54), -INT8_C(  43) },
      UINT64_C(10043564282197167307),
      { -INT8_C(  93), -INT8_C( 117), -INT8_C(  98), -INT8_C( 124), -INT8_C( 124),  INT8_C( 100),  INT8_C(  62),  INT8_C(  79),
        -INT8_C(  88),  INT8_C(  44),  INT8_C( 102),  INT8_C(  70),  INT8_C( 108),  INT8_C(  94), -INT8_C( 112), -INT8_C(  45),
         INT8_C(  22),  INT8_C(  20),  INT8_C(  98),  INT8_C(  85),  INT8_C(  73), -INT8_C( 104),  INT8_C(  42),  INT8_C(  20),
         INT8_C( 100),  INT8_C(  83),  INT8_C( 124), -INT8_C(  28),  INT8_C(  60), -INT8_C(  35),  INT8_C( 111), -INT8_C(  33),
         INT8_C( 104),  INT8_C(  13),  INT8_C(  99), -INT8_C(  19),  INT8_C( 113), -INT8_C(  95),  INT8_C(  60),  INT8_C(  26),
        -INT8_C(  50), -INT8_C(  94),  INT8_C(  96),  INT8_C(  58),  INT8_C(   0), -INT8_C(  15),  INT8_C(  14),  INT8_C(  23),
         INT8_C(   5),  INT8_C( 112),  INT8_C( 108),  INT8_C(  78),  INT8_C(   8), -INT8_C( 106),  INT8_C(  98),  INT8_C( 108),
        -INT8_C(  22), -INT8_C(  34),  INT8_C(  80),  INT8_C(  38), -INT8_C(  69), -INT8_C(  65),  INT8_C(   5),  INT8_C(  36) },
      { -INT8_C(  93), -INT8_C( 117), -INT8_C(  27), -INT8_C( 124), -INT8_C(  68),  INT8_C(   5),  INT8_C(  62),  INT8_C(  79),
         INT8_C(  93),  INT8_C(  87),  INT8_C( 102),  INT8_C(  70),  INT8_C(   8),  INT8_C(  39), -INT8_C( 112), -INT8_C(  45),
         INT8_C(  22), -INT8_C(  47),  INT8_C( 125),  INT8_C(  85),  INT8_C(  77), -INT8_C( 104),  INT8_C(  81), -INT8_C(  32),
         INT8_C( 100),  INT8_C(  83),  INT8_C( 124),  INT8_C(  57),  INT8_C(  39), -INT8_C(  35),  INT8_C( 111), -INT8_C(  89),
        -INT8_C( 105), -INT8_C( 106), -INT8_C( 106),  INT8_C(  83), -INT8_C( 100), -INT8_C(  76), -INT8_C(  65),  INT8_C(  26),
         INT8_C(  11), -INT8_C(  70),  INT8_C(  42),  INT8_C(  58), -INT8_C(  31), -INT8_C(  15),  INT8_C(  14),  INT8_C(  23),
         INT8_C(   5),  INT8_C(  68), -INT8_C(  18),  INT8_C(  23), -INT8_C(  98), -INT8_C( 106),  INT8_C(  98),  INT8_C(  74),
        -INT8_C(  22), -INT8_C(  34), -INT8_C( 125),  INT8_C(  38),  INT8_C(  62),  INT8_C(  53),  INT8_C(  54),  INT8_C(  36) } },
    { { -INT8_C(  52),  INT8_C( 104),  INT8_C(  17),  INT8_C(  61),  INT8_C(  10),  INT8_C(  77),  INT8_C(  87), -INT8_C(  40),
        -INT8_C(  17), -INT8_C(  72),  INT8_C(  18), -INT8_C(  16), -INT8_C(  87),  INT8_C(  32),  INT8_C(   7), -INT8_C(  82),
        -INT8_C( 112),  INT8_C( 115), -INT8_C(   4), -INT8_C( 104),  INT8_C(   9),  INT8_C(  94),  INT8_C(   4), -INT8_C(  13),
         INT8_C(  61),  INT8_C(  84),  INT8_C(  25), -INT8_C(   8),  INT8_C(  19),  INT8_C(  30),  INT8_C(  28), -INT8_C(  33),
        -INT8_C( 121),  INT8_C(  45),  INT8_C(  29), -INT8_C( 111),  INT8_C( 122),  INT8_C( 116),  INT8_C( 105),  INT8_C( 106),
         INT8_C(  44),  INT8_C( 123),  INT8_C(  90), -INT8_C(  43), -INT8_C( 100),  INT8_C(  97), -INT8_C( 125),  INT8_C(  44),
        -INT8_C(  44),      INT8_MAX, -INT8_C(  59), -INT8_C(  35), -INT8_C(  34), -INT8_C(  55), -INT8_C(  47),  INT8_C(  27),
         INT8_C(  30), -INT8_C(  22),  INT8_C(  19),  INT8_C(  49),  INT8_C(   9),  INT8_C(  48),  INT8_C(  17), -INT8_C( 112) },
      UINT64_C(14934651746329374301),
      {  INT8_C(   5), -INT8_C( 100), -INT8_C(  92), -INT8_C(  95), -INT8_C(   3),  INT8_C(  40), -INT8_C(  50), -INT8_C(  47),
        -INT8_C(  89), -INT8_C( 109), -INT8_C(  82), -INT8_C( 123),  INT8_C(  92),      INT8_MAX, -INT8_C(  96),  INT8_C( 122),
         INT8_C( 106), -INT8_C(  76), -INT8_C(  84),  INT8_C( 115), -INT8_C(  28), -INT8_C(  67),  INT8_C(   3),  INT8_C(  65),
        -INT8_C(  21),  INT8_C(  36),  INT8_C(  25), -INT8_C( 115), -INT8_C(  82),  INT8_C(  91),  INT8_C(  92), -INT8_C(  77),
        -INT8_C(   9),  INT8_C(   1),  INT8_C(  85), -INT8_C(  12),  INT8_C(  41),  INT8_C(  35), -INT8_C(  59), -INT8_C(  48),
        -INT8_C(  74),  INT8_C( 116),  INT8_C(  86),  INT8_C(  18), -INT8_C(  13), -INT8_C(  10), -INT8_C( 115),  INT8_C(  93),
        -INT8_C(  86),  INT8_C(  57), -INT8_C(  48), -INT8_C( 114), -INT8_C(  10), -INT8_C(  45), -INT8_C(  48), -INT8_C(  31),
        -INT8_C(   9), -INT8_C(  23),  INT8_C( 110), -INT8_C(  91),  INT8_C(  69), -INT8_C(  53),  INT8_C(  89),  INT8_C(  60) },
      {  INT8_C(   5),  INT8_C( 104), -INT8_C(  92), -INT8_C(  95), -INT8_C(   3),  INT8_C(  77), -INT8_C(  50), -INT8_C(  40),
        -INT8_C(  17), -INT8_C( 109), -INT8_C(  82), -INT8_C( 123), -INT8_C(  87),      INT8_MAX,  INT8_C(   7), -INT8_C(  82),
         INT8_C( 106),  INT8_C( 115), -INT8_C(   4), -INT8_C( 104),  INT8_C(   9), -INT8_C(  67),  INT8_C(   4), -INT8_C(  13),
         INT8_C(  61),  INT8_C(  84),  INT8_C(  25), -INT8_C( 115), -INT8_C(  82),  INT8_C(  30),  INT8_C(  92), -INT8_C(  77),
        -INT8_C( 121),  INT8_C(   1),  INT8_C(  29), -INT8_C( 111),  INT8_C( 122),  INT8_C(  35),  INT8_C( 105), -INT8_C(  48),
         INT8_C(  44),  INT8_C( 116),  INT8_C(  90),  INT8_C(  18), -INT8_C( 100),  INT8_C(  97), -INT8_C( 125),  INT8_C(  93),
        -INT8_C(  44),  INT8_C(  57), -INT8_C(  59), -INT8_C(  35), -INT8_C(  34), -INT8_C(  55), -INT8_C(  48),  INT8_C(  27),
        -INT8_C(   9), -INT8_C(  23),  INT8_C( 110), -INT8_C(  91),  INT8_C(   9),  INT8_C(  48),  INT8_C(  89),  INT8_C(  60) } },
    { { -INT8_C(  52), -INT8_C(  82),  INT8_C(  49), -INT8_C(  11), -INT8_C(  47), -INT8_C(  10), -INT8_C(  59), -INT8_C( 121),
         INT8_C( 106),  INT8_C(  27), -INT8_C( 103),  INT8_C(  94),  INT8_C(  18),  INT8_C(  38), -INT8_C(  69), -INT8_C(  68),
         INT8_C(  95), -INT8_C( 116),  INT8_C(  75),  INT8_C(  85),  INT8_C(  95),  INT8_C(  27),  INT8_C(  54),  INT8_C(  87),
         INT8_C(   4), -INT8_C(  91), -INT8_C(   4),  INT8_C(  73),  INT8_C( 112),  INT8_C(  85), -INT8_C( 122),  INT8_C(  60),
         INT8_C(   3), -INT8_C(  73),  INT8_C(  49), -INT8_C(  44), -INT8_C(  83), -INT8_C(  10),  INT8_C(  91),  INT8_C(  24),
         INT8_C(  18), -INT8_C(  11),  INT8_C( 118),  INT8_C(  36),  INT8_C(  27),  INT8_C(  49), -INT8_C(  32),  INT8_C( 123),
        -INT8_C(  67),  INT8_C(  43), -INT8_C(  48),  INT8_C(  29),  INT8_C(  70),  INT8_C(   7),  INT8_C( 116),  INT8_C(  75),
        -INT8_C(  84),  INT8_C( 112), -INT8_C( 108),  INT8_C(  28), -INT8_C(  58),  INT8_C(  26),  INT8_C(  88), -INT8_C(  55) },
      UINT64_C(10491128182980118993),
      { -INT8_C(  18),  INT8_C(  13), -INT8_C(  75),  INT8_C(  10),  INT8_C(  62), -INT8_C( 106), -INT8_C( 123), -INT8_C(   4),
        -INT8_C(  63),  INT8_C(  85),  INT8_C(  25),  INT8_C(   8),  INT8_C(  92), -INT8_C( 115),  INT8_C(  83),  INT8_C(   8),
        -INT8_C(   3), -INT8_C(  25),  INT8_C(  36), -INT8_C(  61),  INT8_C(   2),  INT8_C( 124), -INT8_C( 115), -INT8_C(  45),
         INT8_C(   5),  INT8_C(  43),  INT8_C(  82), -INT8_C( 123),  INT8_C(  36), -INT8_C(  23),  INT8_C(  22),  INT8_C(  19),
        -INT8_C(  10), -INT8_C(  52),  INT8_C(  29),  INT8_C(  53),  INT8_C(  98), -INT8_C(  94),  INT8_C(  49),  INT8_C(  35),
        -INT8_C(   9),  INT8_C(  74),  INT8_C(  43),  INT8_C(  84), -INT8_C(  41),  INT8_C( 126),  INT8_C(  92), -INT8_C(  44),
         INT8_C( 102), -INT8_C( 127), -INT8_C( 104),  INT8_C( 104), -INT8_C(   3),  INT8_C(  37),  INT8_C(  59),  INT8_C(   3),
         INT8_C(  80), -INT8_C( 114), -INT8_C( 120),  INT8_C( 116),  INT8_C( 119), -INT8_C(  98), -INT8_C( 121),  INT8_C( 110) },
      { -INT8_C(  18), -INT8_C(  82),  INT8_C(  49), -INT8_C(  11),  INT8_C(  62), -INT8_C(  10), -INT8_C( 123), -INT8_C(   4),
        -INT8_C(  63),  INT8_C(  27), -INT8_C( 103),  INT8_C(   8),  INT8_C(  18),  INT8_C(  38), -INT8_C(  69),  INT8_C(   8),
         INT8_C(  95), -INT8_C(  25),  INT8_C(  36), -INT8_C(  61),  INT8_C(   2),  INT8_C(  27),  INT8_C(  54), -INT8_C(  45),
         INT8_C(   5),  INT8_C(  43),  INT8_C(  82), -INT8_C( 123),  INT8_C(  36), -INT8_C(  23),  INT8_C(  22),  INT8_C(  60),
        -INT8_C(  10), -INT8_C(  52),  INT8_C(  29),  INT8_C(  53),  INT8_C(  98), -INT8_C(  94),  INT8_C(  49),  INT8_C(  24),
        -INT8_C(   9), -INT8_C(  11),  INT8_C( 118),  INT8_C(  84), -INT8_C(  41),  INT8_C( 126),  INT8_C(  92), -INT8_C(  44),
         INT8_C( 102), -INT8_C( 127), -INT8_C( 104),  INT8_C(  29), -INT8_C(   3),  INT8_C(   7),  INT8_C( 116),  INT8_C(   3),
         INT8_C(  80),  INT8_C( 112), -INT8_C( 108),  INT8_C(  28),  INT8_C( 119),  INT8_C(  26),  INT8_C(  88),  INT8_C( 110) } },
    { {  INT8_C( 106), -INT8_C(  92), -INT8_C(  93), -INT8_C(  52),  INT8_C(  70), -INT8_C(  44), -INT8_C(  16),  INT8_C(  62),
         INT8_C(  30),  INT8_C(  27), -INT8_C( 110), -INT8_C(  11), -INT8_C( 102), -INT8_C(  18), -INT8_C(  55),  INT8_C(   0),
         INT8_C( 111),  INT8_C(  97),  INT8_C( 104),  INT8_C( 109), -INT8_C( 122), -INT8_C(  93),  INT8_C( 112), -INT8_C(  42),
         INT8_C(  49), -INT8_C(   8),  INT8_C(  75), -INT8_C(  87), -INT8_C( 106), -INT8_C(  46),  INT8_C(  23),  INT8_C(   1),
         INT8_C( 119), -INT8_C(  70), -INT8_C(  51), -INT8_C(  67), -INT8_C( 114), -INT8_C(  67), -INT8_C(   5), -INT8_C(  84),
        -INT8_C(  39), -INT8_C( 115), -INT8_C(  95),  INT8_C( 115),  INT8_C( 124),  INT8_C( 106),  INT8_C( 115), -INT8_C(  21),
        -INT8_C(  52), -INT8_C(  37),  INT8_C(  88),  INT8_C(  82),  INT8_C( 126), -INT8_C(  56),  INT8_C(  41), -INT8_C(  80),
        -INT8_C(  64),  INT8_C( 116),  INT8_C(  89),  INT8_C(  87),  INT8_C(  70),  INT8_C( 112),  INT8_C(  88), -INT8_C(  67) },
      UINT64_C(13575105897226249514),
      {  INT8_C(   4),  INT8_C(   5),  INT8_C(  47),      INT8_MIN,  INT8_C( 111), -INT8_C(  94),  INT8_C( 107),  INT8_C(  59),
         INT8_C( 125), -INT8_C(  60), -INT8_C( 114), -INT8_C(   5), -INT8_C( 116), -INT8_C(  73), -INT8_C(  85),  INT8_C(  77),
         INT8_C(  43),  INT8_C(   4), -INT8_C(  92),  INT8_C( 113),  INT8_C( 116), -INT8_C(   4),  INT8_C(  47), -INT8_C(  98),
         INT8_C(  33), -INT8_C(  86),  INT8_C(  86),  INT8_C(   4),  INT8_C(  32), -INT8_C(  70), -INT8_C(  64),  INT8_C(  36),
        -INT8_C(  65), -INT8_C(  17), -INT8_C(  92),  INT8_C(  47), -INT8_C( 111),  INT8_C(  16),  INT8_C( 106),  INT8_C(  14),
        -INT8_C(  44), -INT8_C(   8),  INT8_C(  10),  INT8_C(  96), -INT8_C(  81), -INT8_C(  75), -INT8_C(  83), -INT8_C(  38),
        -INT8_C(  70),  INT8_C(  81),  INT8_C(  76),  INT8_C(  46),  INT8_C(  77),  INT8_C( 123), -INT8_C(  51),  INT8_C( 111),
         INT8_C(  37),  INT8_C(  35),  INT8_C( 115),  INT8_C(  69), -INT8_C(  34),  INT8_C(  52),  INT8_C( 106), -INT8_C(  99) },
      {  INT8_C( 106),  INT8_C(   5), -INT8_C(  93),      INT8_MIN,  INT8_C(  70), -INT8_C(  94), -INT8_C(  16),  INT8_C(  62),
         INT8_C( 125),  INT8_C(  27), -INT8_C( 114), -INT8_C(  11), -INT8_C( 102), -INT8_C(  73), -INT8_C(  55),  INT8_C(   0),
         INT8_C(  43),  INT8_C(   4),  INT8_C( 104),  INT8_C( 113),  INT8_C( 116), -INT8_C(   4),  INT8_C(  47), -INT8_C(  42),
         INT8_C(  49), -INT8_C(   8),  INT8_C(  75),  INT8_C(   4),  INT8_C(  32), -INT8_C(  70),  INT8_C(  23),  INT8_C(  36),
        -INT8_C(  65), -INT8_C(  17), -INT8_C(  51), -INT8_C(  67), -INT8_C( 114),  INT8_C(  16),  INT8_C( 106),  INT8_C(  14),
        -INT8_C(  39), -INT8_C(   8),  INT8_C(  10),  INT8_C( 115), -INT8_C(  81), -INT8_C(  75), -INT8_C(  83), -INT8_C(  21),
        -INT8_C(  52), -INT8_C(  37),  INT8_C(  76),  INT8_C(  82),  INT8_C( 126),  INT8_C( 123), -INT8_C(  51), -INT8_C(  80),
        -INT8_C(  64),  INT8_C( 116),  INT8_C( 115),  INT8_C(  69), -INT8_C(  34),  INT8_C(  52),  INT8_C(  88), -INT8_C(  99) } },
    { {  INT8_C(  35),  INT8_C(  14), -INT8_C(  52), -INT8_C(  75),  INT8_C(  30),  INT8_C(  55), -INT8_C(  61), -INT8_C(  14),
         INT8_C(  47), -INT8_C(  51),  INT8_C(  83), -INT8_C(  33), -INT8_C( 125),  INT8_C(   0), -INT8_C(  71),  INT8_C(  61),
         INT8_C(  82),  INT8_C(   5),  INT8_C( 107), -INT8_C(  97),      INT8_MIN,  INT8_C(  56),  INT8_C(  14), -INT8_C(  91),
         INT8_C(  92), -INT8_C( 126), -INT8_C(  21),  INT8_C(  58), -INT8_C(  74),  INT8_C(  85), -INT8_C(  41), -INT8_C(  39),
         INT8_C(  99), -INT8_C(  92), -INT8_C( 114), -INT8_C( 126), -INT8_C(  37),  INT8_C(  82),  INT8_C( 116),  INT8_C(  10),
         INT8_C(  31), -INT8_C(  57), -INT8_C(  23), -INT8_C(  94), -INT8_C(  56), -INT8_C(  93), -INT8_C(  33),  INT8_C(  26),
        -INT8_C(  88),  INT8_C(  75), -INT8_C(  71),  INT8_C(  41), -INT8_C( 125), -INT8_C(  56), -INT8_C(  50), -INT8_C(  33),
         INT8_C(  74), -INT8_C(  71),  INT8_C(  25),  INT8_C(   0),  INT8_C(  14), -INT8_C(  15), -INT8_C(  39),  INT8_C( 114) },
      UINT64_C(15670953003357333653),
      {  INT8_C(  48),  INT8_C( 100),  INT8_C( 124), -INT8_C(   8),  INT8_C(   7),  INT8_C(  91),  INT8_C(  18), -INT8_C(  81),
        -INT8_C(  90), -INT8_C(  53), -INT8_C(  40),  INT8_C(  42), -INT8_C( 109), -INT8_C(  89),  INT8_C(   9), -INT8_C(  35),
         INT8_C(  96),  INT8_C(  35), -INT8_C(  35),  INT8_C( 111),  INT8_C(  20), -INT8_C(  73), -INT8_C(  31), -INT8_C(  87),
         INT8_C(  31), -INT8_C(  43),  INT8_C(  25), -INT8_C(  39),  INT8_C(  61), -INT8_C( 109), -INT8_C(  78),  INT8_C( 109),
        -INT8_C(   9),  INT8_C(  46),  INT8_C( 101), -INT8_C(   2), -INT8_C( 118),  INT8_C( 119), -INT8_C(  82),  INT8_C(  48),
         INT8_C(  67), -INT8_C( 122),  INT8_C(  90), -INT8_C(  42),  INT8_C(  45),  INT8_C( 100), -INT8_C(  76), -INT8_C( 114),
        -INT8_C( 121), -INT8_C( 111), -INT8_C(   3), -INT8_C( 101),  INT8_C(  72), -INT8_C(  34),  INT8_C(  68),  INT8_C( 103),
        -INT8_C(  77),  INT8_C(  93),  INT8_C(  64), -INT8_C(  16), -INT8_C(  16), -INT8_C(  13),  INT8_C(  94), -INT8_C(  24) },
      {  INT8_C(  48),  INT8_C(  14),  INT8_C( 124), -INT8_C(  75),  INT8_C(   7),  INT8_C(  55), -INT8_C(  61), -INT8_C(  81),
         INT8_C(  47), -INT8_C(  51),  INT8_C(  83),  INT8_C(  42), -INT8_C( 125), -INT8_C(  89),  INT8_C(   9),  INT8_C(  61),
         INT8_C(  82),  INT8_C(   5), -INT8_C(  35), -INT8_C(  97),  INT8_C(  20), -INT8_C(  73), -INT8_C(  31), -INT8_C(  87),
         INT8_C(  92), -INT8_C( 126), -INT8_C(  21),  INT8_C(  58),  INT8_C(  61), -INT8_C( 109), -INT8_C(  78), -INT8_C(  39),
         INT8_C(  99),  INT8_C(  46), -INT8_C( 114), -INT8_C(   2), -INT8_C( 118),  INT8_C( 119),  INT8_C( 116),  INT8_C(  48),
         INT8_C(  31), -INT8_C(  57), -INT8_C(  23), -INT8_C(  42), -INT8_C(  56),  INT8_C( 100), -INT8_C(  76),  INT8_C(  26),
        -INT8_C(  88), -INT8_C( 111), -INT8_C(  71), -INT8_C( 101),  INT8_C(  72), -INT8_C(  34),  INT8_C(  68), -INT8_C(  33),
        -INT8_C(  77), -INT8_C(  71),  INT8_C(  25), -INT8_C(  16), -INT8_C(  16), -INT8_C(  15),  INT8_C(  94), -INT8_C(  24) } },
    { {  INT8_C(  33), -INT8_C(  61), -INT8_C(  26), -INT8_C(  85),  INT8_C(  59), -INT8_C( 108), -INT8_C(  36),  INT8_C( 126),
         INT8_C(  27),  INT8_C(  54),  INT8_C(  84),  INT8_C(  72), -INT8_C( 102),  INT8_C(   8), -INT8_C(  42),  INT8_C(  33),
        -INT8_C( 102), -INT8_C(  45), -INT8_C(  68), -INT8_C(  30), -INT8_C(  79),  INT8_C(   0),  INT8_C(  74),  INT8_C( 100),
         INT8_C(  93), -INT8_C( 118),  INT8_C(  85),  INT8_C(  78),  INT8_C( 125), -INT8_C(  77),  INT8_C(  54), -INT8_C(  97),
         INT8_C( 118),  INT8_C(  28),  INT8_C(  74), -INT8_C(  79), -INT8_C(  79),  INT8_C(  38),  INT8_C(  47), -INT8_C(  52),
         INT8_C(  93), -INT8_C( 124),  INT8_C(  20), -INT8_C(   9), -INT8_C( 116), -INT8_C(  21),  INT8_C(  25),  INT8_C(  38),
        -INT8_C(  66), -INT8_C(  43),  INT8_C(   9),  INT8_C( 112), -INT8_C(  42),  INT8_C(  83), -INT8_C(  44),  INT8_C(  51),
        -INT8_C(  35),  INT8_C(  41), -INT8_C( 127),  INT8_C(  91), -INT8_C(  36), -INT8_C(  73), -INT8_C(   6),  INT8_C(  83) },
      UINT64_C(14434375917096944852),
      { -INT8_C(  72),  INT8_C( 101), -INT8_C(  65),  INT8_C(  68),  INT8_C(  80), -INT8_C(  40),  INT8_C( 107),  INT8_C(  15),
        -INT8_C(  82),  INT8_C( 116),      INT8_MAX, -INT8_C( 124), -INT8_C(  57),  INT8_C(  83), -INT8_C(  73), -INT8_C(  92),
         INT8_C( 125),  INT8_C(  57), -INT8_C(   1),  INT8_C(  89), -INT8_C(  16), -INT8_C(   7), -INT8_C(  84), -INT8_C(  60),
         INT8_C(  62), -INT8_C(  79),  INT8_C(  73), -INT8_C(  87), -INT8_C(  27), -INT8_C( 102),  INT8_C( 113), -INT8_C(  99),
         INT8_C(   0),  INT8_C(  48), -INT8_C(  31),  INT8_C(  80),  INT8_C(   9),  INT8_C(  76),  INT8_C(  95), -INT8_C(  73),
        -INT8_C(  64), -INT8_C(  34),  INT8_C(  59), -INT8_C( 121),  INT8_C(  50), -INT8_C(  14),  INT8_C(  44), -INT8_C(  81),
         INT8_C(  43),  INT8_C(  43),  INT8_C(   8),  INT8_C(  28),  INT8_C(  37), -INT8_C(  75), -INT8_C(  32),  INT8_C(  99),
         INT8_C( 102),  INT8_C(  42),  INT8_C(  12),  INT8_C(  75), -INT8_C(  60),  INT8_C( 125), -INT8_C(  24), -INT8_C(  60) },
      {  INT8_C(  33), -INT8_C(  61), -INT8_C(  65), -INT8_C(  85),  INT8_C(  80), -INT8_C( 108),  INT8_C( 107),  INT8_C(  15),
         INT8_C(  27),  INT8_C(  54),      INT8_MAX,  INT8_C(  72), -INT8_C( 102),  INT8_C(   8), -INT8_C(  73),  INT8_C(  33),
        -INT8_C( 102), -INT8_C(  45), -INT8_C(   1), -INT8_C(  30), -INT8_C(  79),  INT8_C(   0),  INT8_C(  74),  INT8_C( 100),
         INT8_C(  62), -INT8_C( 118),  INT8_C(  73),  INT8_C(  78),  INT8_C( 125), -INT8_C(  77),  INT8_C(  54), -INT8_C(  99),
         INT8_C(   0),  INT8_C(  48),  INT8_C(  74),  INT8_C(  80), -INT8_C(  79),  INT8_C(  76),  INT8_C(  95), -INT8_C(  52),
         INT8_C(  93), -INT8_C( 124),  INT8_C(  59), -INT8_C(   9),  INT8_C(  50), -INT8_C(  14),  INT8_C(  25),  INT8_C(  38),
         INT8_C(  43), -INT8_C(  43),  INT8_C(   9),  INT8_C( 112),  INT8_C(  37),  INT8_C(  83), -INT8_C(  32),  INT8_C(  51),
        -INT8_C(  35),  INT8_C(  41), -INT8_C( 127),  INT8_C(  75), -INT8_C(  36), -INT8_C(  73), -INT8_C(  24), -INT8_C(  60) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi8(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_loadu_epi8(src, k, (void*)&a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_loadu_epi8");
    easysimd_assert_equal_vi8(sizeof(r), (const int8_t*)&r, test_vec[i].r);
  }
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i8x64();
    easysimd__mmask64 k = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i r = easysimd_mm512_mask_loadu_epi8(src, k, &a.i8[0]);

    easysimd_test_x86_write_i8x64(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask64(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

  return 0;
}

static int
test_easysimd_mm512_maskz_loadu_epi8()
{
#if 1
  static const struct {
    easysimd__mmask64 k;
    const int8_t a[64];
    const int8_t r[64];
  } test_vec[] = {
    { UINT64_C( 3061090896087144618),
      {  INT8_C(  57), -INT8_C( 108), -INT8_C(  70), -INT8_C(  37),  INT8_C( 107), -INT8_C( 123),  INT8_C(  73),  INT8_C(   9),
        -INT8_C(  24),  INT8_C(  77),  INT8_C(  61),  INT8_C(   8),  INT8_C(  72),  INT8_C(  25),  INT8_C( 104), -INT8_C( 127),
         INT8_C(  63),  INT8_C(  84),  INT8_C(  18), -INT8_C(  34),  INT8_C(  33), -INT8_C(  50), -INT8_C(  82), -INT8_C(  53),
        -INT8_C(  90),  INT8_C(  34),  INT8_C( 105), -INT8_C( 103),  INT8_C(  79), -INT8_C(  28), -INT8_C(  61), -INT8_C( 120),
         INT8_C( 121),  INT8_C( 125),  INT8_C( 100), -INT8_C(  28),  INT8_C(   2), -INT8_C(  83), -INT8_C(  19), -INT8_C(  22),
        -INT8_C(   6),  INT8_C(  43), -INT8_C(  14),  INT8_C(  67),  INT8_C(  68),  INT8_C(  91), -INT8_C(  60), -INT8_C( 124),
        -INT8_C(  81), -INT8_C(  42),  INT8_C(  98), -INT8_C(  48), -INT8_C(  91),  INT8_C(  16), -INT8_C( 101),  INT8_C(  75),
         INT8_C(  51),  INT8_C(   4), -INT8_C(  27), -INT8_C( 126), -INT8_C(  24), -INT8_C(  88),  INT8_C(  11),  INT8_C(  97) },
      {  INT8_C(   0), -INT8_C( 108),  INT8_C(   0), -INT8_C(  37),  INT8_C(   0), -INT8_C( 123),  INT8_C(   0),  INT8_C(   9),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   8),  INT8_C(  72),  INT8_C(   0),  INT8_C( 104), -INT8_C( 127),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  18),  INT8_C(   0),  INT8_C(  33), -INT8_C(  50), -INT8_C(  82),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  34),  INT8_C( 105), -INT8_C( 103),  INT8_C(  79),  INT8_C(   0),  INT8_C(   0), -INT8_C( 120),
         INT8_C( 121),  INT8_C( 125),  INT8_C(   0),  INT8_C(   0),  INT8_C(   2), -INT8_C(  83), -INT8_C(  19), -INT8_C(  22),
        -INT8_C(   6),  INT8_C(   0), -INT8_C(  14),  INT8_C(  67),  INT8_C(   0),  INT8_C(  91),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  81), -INT8_C(  42),  INT8_C(   0), -INT8_C(  48), -INT8_C(  91),  INT8_C(  16), -INT8_C( 101),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   4),  INT8_C(   0), -INT8_C( 126),  INT8_C(   0), -INT8_C(  88),  INT8_C(   0),  INT8_C(   0) } },
    { UINT64_C( 1590671289419656998),
      {  INT8_C(  94),  INT8_C(   5),  INT8_C(  89), -INT8_C(  94),  INT8_C(  96),  INT8_C(  29),  INT8_C(  38),  INT8_C(  15),
        -INT8_C(  12), -INT8_C( 119), -INT8_C(  33), -INT8_C( 103), -INT8_C( 103),  INT8_C( 122), -INT8_C(  28), -INT8_C(  52),
         INT8_C( 126), -INT8_C(  55),  INT8_C(  79),  INT8_C( 103),  INT8_C( 114),  INT8_C(  90), -INT8_C(  56), -INT8_C( 104),
        -INT8_C(  55),  INT8_C(  14), -INT8_C(  64), -INT8_C(  27),  INT8_C(  65), -INT8_C(  45), -INT8_C(   5), -INT8_C(  97),
        -INT8_C(  39),  INT8_C(  85),  INT8_C(  65),  INT8_C(  57),  INT8_C( 114),  INT8_C( 104),  INT8_C(  73),  INT8_C( 102),
        -INT8_C(  15),  INT8_C(  40), -INT8_C(   1), -INT8_C( 118), -INT8_C(  93), -INT8_C(  28),  INT8_C(  87),  INT8_C(  33),
        -INT8_C(  83), -INT8_C(  90), -INT8_C( 120),  INT8_C(  31),  INT8_C(   0),  INT8_C(  81), -INT8_C(  73), -INT8_C(  55),
         INT8_C(  95),  INT8_C( 120), -INT8_C(  82), -INT8_C(  96),  INT8_C(  75), -INT8_C(  87),  INT8_C(  63),  INT8_C(  36) },
      {  INT8_C(   0),  INT8_C(   5),  INT8_C(  89),  INT8_C(   0),  INT8_C(   0),  INT8_C(  29),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  12), -INT8_C( 119), -INT8_C(  33), -INT8_C( 103),  INT8_C(   0),  INT8_C( 122), -INT8_C(  28),  INT8_C(   0),
         INT8_C( 126),  INT8_C(   0),  INT8_C(  79),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  56),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  27),  INT8_C(   0), -INT8_C(  45),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  65),  INT8_C(  57),  INT8_C( 114),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  15),  INT8_C(  40),  INT8_C(   0),  INT8_C(   0), -INT8_C(  93), -INT8_C(  28),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  83), -INT8_C(  90),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C( 120), -INT8_C(  82),  INT8_C(   0),  INT8_C(  75),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0) } },
    { UINT64_C(15697199642999423230),
      { -INT8_C(  49), -INT8_C(  41),  INT8_C( 100),  INT8_C( 114), -INT8_C(  69), -INT8_C(  69), -INT8_C( 108),  INT8_C( 104),
         INT8_C(  97),  INT8_C(  28), -INT8_C( 120),  INT8_C(  97),  INT8_C( 109),  INT8_C(  63),  INT8_C(  42), -INT8_C(  52),
        -INT8_C(  73), -INT8_C(  40),  INT8_C( 108),  INT8_C(   3), -INT8_C( 127), -INT8_C(  85),  INT8_C(  39),      INT8_MIN,
         INT8_C(  44), -INT8_C( 123), -INT8_C(  15),  INT8_C(  20),  INT8_C(  44), -INT8_C(  56), -INT8_C(  18), -INT8_C(   4),
        -INT8_C(  97),  INT8_C(  82),  INT8_C( 110),  INT8_C(  90),  INT8_C(  13),  INT8_C(   2), -INT8_C(  61),  INT8_C( 110),
         INT8_C(  31),  INT8_C(  75), -INT8_C(  49), -INT8_C( 116), -INT8_C( 118), -INT8_C(   7),  INT8_C(  89),  INT8_C(  66),
        -INT8_C(  47), -INT8_C(  59),  INT8_C(  69),  INT8_C(  82),  INT8_C( 113),  INT8_C( 108), -INT8_C(  46), -INT8_C(  99),
        -INT8_C(  14), -INT8_C(  61), -INT8_C(  79),  INT8_C(  30), -INT8_C( 116), -INT8_C(  97),  INT8_C(  26),  INT8_C(  43) },
      {  INT8_C(   0), -INT8_C(  41),  INT8_C( 100),  INT8_C( 114), -INT8_C(  69), -INT8_C(  69), -INT8_C( 108),  INT8_C( 104),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  52),
         INT8_C(   0), -INT8_C(  40),  INT8_C( 108),  INT8_C(   3), -INT8_C( 127),  INT8_C(   0),  INT8_C(  39),  INT8_C(   0),
         INT8_C(  44),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  44), -INT8_C(  56), -INT8_C(  18),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  90),  INT8_C(   0),  INT8_C(   2), -INT8_C(  61),  INT8_C( 110),
         INT8_C(  31),  INT8_C(  75), -INT8_C(  49),  INT8_C(   0),  INT8_C(   0), -INT8_C(   7),  INT8_C(   0),  INT8_C(  66),
        -INT8_C(  47), -INT8_C(  59),  INT8_C(  69),  INT8_C(   0),  INT8_C( 113),  INT8_C(   0), -INT8_C(  46), -INT8_C(  99),
        -INT8_C(  14),  INT8_C(   0),  INT8_C(   0),  INT8_C(  30), -INT8_C( 116),  INT8_C(   0),  INT8_C(  26),  INT8_C(  43) } },
    { UINT64_C(12280271149552011761),
      { -INT8_C( 108),  INT8_C(  59),  INT8_C(  55),  INT8_C(  30),  INT8_C(  52), -INT8_C( 112),  INT8_C(  96),  INT8_C(   5),
         INT8_C(  85), -INT8_C(  91),  INT8_C(  88), -INT8_C(  58),  INT8_C(  18),  INT8_C(  42),  INT8_C(  99),  INT8_C(   4),
        -INT8_C(  18),  INT8_C(  21),  INT8_C(  34),  INT8_C( 122), -INT8_C(  76),  INT8_C(  61), -INT8_C(  91), -INT8_C(  90),
        -INT8_C(  58),  INT8_C(  43), -INT8_C(  92),  INT8_C(  81),  INT8_C( 116),  INT8_C(  17), -INT8_C(   4),  INT8_C(   8),
         INT8_C(  76),  INT8_C(  51),  INT8_C(  39), -INT8_C( 127), -INT8_C(  61), -INT8_C( 121), -INT8_C( 122),  INT8_C(  24),
         INT8_C(  45), -INT8_C(  34), -INT8_C(  33),  INT8_C(  63),  INT8_C(   9),  INT8_C(  66),  INT8_C(  67), -INT8_C(   9),
         INT8_C(  87),  INT8_C( 101),  INT8_C( 113),  INT8_C(  12), -INT8_C(  94),  INT8_C(  22), -INT8_C(  78),  INT8_C( 104),
         INT8_C(  66),  INT8_C(  86), -INT8_C(  70), -INT8_C(  74),  INT8_C( 103), -INT8_C(  74), -INT8_C(  65), -INT8_C(  76) },
      { -INT8_C( 108),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  52), -INT8_C( 112),  INT8_C(  96),  INT8_C(   5),
         INT8_C(  85),  INT8_C(   0),  INT8_C(   0), -INT8_C(  58),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   4),
         INT8_C(   0),  INT8_C(  21),  INT8_C(  34),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  90),
         INT8_C(   0),  INT8_C(  43), -INT8_C(  92),  INT8_C(  81),  INT8_C( 116),  INT8_C(  17), -INT8_C(   4),  INT8_C(   8),
         INT8_C(  76),  INT8_C(  51),  INT8_C(   0), -INT8_C( 127),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  24),
         INT8_C(  45),  INT8_C(   0),  INT8_C(   0),  INT8_C(  63),  INT8_C(   0),  INT8_C(   0),  INT8_C(  67),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C( 113),  INT8_C(  12),  INT8_C(   0),  INT8_C(  22), -INT8_C(  78),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  86),  INT8_C(   0), -INT8_C(  74),  INT8_C(   0), -INT8_C(  74),  INT8_C(   0), -INT8_C(  76) } },
    { UINT64_C(11152244656991233769),
      { -INT8_C( 102), -INT8_C(  93), -INT8_C(  39), -INT8_C(  93), -INT8_C(  26),  INT8_C(  28), -INT8_C( 102),  INT8_C(  61),
        -INT8_C( 126),  INT8_C(  11),  INT8_C(  73),  INT8_C(  36),  INT8_C(  33), -INT8_C(   5), -INT8_C( 115),  INT8_C(  99),
         INT8_C(  82),  INT8_C(  71),  INT8_C(  26), -INT8_C(  71), -INT8_C(   3), -INT8_C(  39),  INT8_C( 109), -INT8_C(  26),
        -INT8_C(  65), -INT8_C(  94), -INT8_C( 110),  INT8_C(  44),  INT8_C(  94),  INT8_C(  86), -INT8_C(  57), -INT8_C(   8),
        -INT8_C(   6), -INT8_C(  96), -INT8_C( 101), -INT8_C(  32), -INT8_C(  67),  INT8_C(  53),  INT8_C(  29),  INT8_C(  63),
         INT8_C(  64),  INT8_C( 103),  INT8_C(  99),  INT8_C(  97),  INT8_C(  98), -INT8_C(  16), -INT8_C(  59), -INT8_C(  76),
         INT8_C(  55), -INT8_C(  33),  INT8_C( 110),  INT8_C(  52), -INT8_C(  72), -INT8_C(  37),  INT8_C(  26),  INT8_C( 119),
         INT8_C( 126), -INT8_C(  84), -INT8_C(  93), -INT8_C(  36),  INT8_C(   3),  INT8_C( 106), -INT8_C(  44), -INT8_C(   3) },
      { -INT8_C( 102),  INT8_C(   0),  INT8_C(   0), -INT8_C(  93),  INT8_C(   0),  INT8_C(  28), -INT8_C( 102),  INT8_C(  61),
         INT8_C(   0),  INT8_C(  11),  INT8_C(  73),  INT8_C(   0),  INT8_C(   0), -INT8_C(   5), -INT8_C( 115),  INT8_C(  99),
         INT8_C(  82),  INT8_C(   0),  INT8_C(  26),  INT8_C(   0), -INT8_C(   3), -INT8_C(  39),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0), -INT8_C( 110),  INT8_C(  44),  INT8_C(   0),  INT8_C(  86),  INT8_C(   0), -INT8_C(   8),
        -INT8_C(   6),  INT8_C(   0), -INT8_C( 101), -INT8_C(  32),  INT8_C(   0),  INT8_C(  53),  INT8_C(  29),  INT8_C(   0),
         INT8_C(  64),  INT8_C( 103),  INT8_C(   0),  INT8_C(  97),  INT8_C(  98), -INT8_C(  16),  INT8_C(   0), -INT8_C(  76),
         INT8_C(   0),  INT8_C(   0),  INT8_C( 110),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  26),  INT8_C( 119),
         INT8_C(   0), -INT8_C(  84),  INT8_C(   0), -INT8_C(  36),  INT8_C(   3),  INT8_C(   0),  INT8_C(   0), -INT8_C(   3) } },
    { UINT64_C(16431377351136079627),
      {  INT8_C(  97),  INT8_C( 106),  INT8_C(  69), -INT8_C(  60),  INT8_C(  91),  INT8_C(  10),  INT8_C( 120), -INT8_C( 110),
        -INT8_C(  23), -INT8_C(  26), -INT8_C(  57), -INT8_C(  95), -INT8_C(  62), -INT8_C(  31),  INT8_C(  24),  INT8_C(  64),
        -INT8_C( 114), -INT8_C(  68),  INT8_C(  28), -INT8_C( 111),  INT8_C(  38), -INT8_C(  16), -INT8_C( 114),  INT8_C(  49),
         INT8_C(  95),  INT8_C( 107), -INT8_C(   7),  INT8_C(   3),  INT8_C( 101),  INT8_C(   0), -INT8_C(  25), -INT8_C(  57),
         INT8_C( 107),  INT8_C(  44), -INT8_C( 117), -INT8_C(  58),  INT8_C(  55),  INT8_C(   3),  INT8_C(  88),  INT8_C(  32),
        -INT8_C(  22),  INT8_C(  31), -INT8_C(  62), -INT8_C(  84),  INT8_C(   1), -INT8_C(  38), -INT8_C(  20), -INT8_C( 113),
        -INT8_C( 106),  INT8_C(   8),  INT8_C(  32), -INT8_C(  67), -INT8_C(   8), -INT8_C(  82), -INT8_C(  18),  INT8_C(  87),
         INT8_C(  25), -INT8_C(  24),  INT8_C(  90),  INT8_C( 126), -INT8_C(  24),  INT8_C(  65),  INT8_C(  69),  INT8_C(  83) },
      {  INT8_C(  97),  INT8_C( 106),  INT8_C(   0), -INT8_C(  60),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
        -INT8_C(  23), -INT8_C(  26), -INT8_C(  57), -INT8_C(  95),  INT8_C(   0), -INT8_C(  31),  INT8_C(  24),  INT8_C(   0),
        -INT8_C( 114),  INT8_C(   0),  INT8_C(  28), -INT8_C( 111),  INT8_C(  38),  INT8_C(   0), -INT8_C( 114),  INT8_C(  49),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   3),  INT8_C(   0),  INT8_C(   0), -INT8_C(  25), -INT8_C(  57),
         INT8_C(   0),  INT8_C(   0), -INT8_C( 117),  INT8_C(   0),  INT8_C(   0),  INT8_C(   3),  INT8_C(   0),  INT8_C(  32),
         INT8_C(   0),  INT8_C(  31),  INT8_C(   0), -INT8_C(  84),  INT8_C(   1), -INT8_C(  38), -INT8_C(  20), -INT8_C( 113),
        -INT8_C( 106),  INT8_C(   8),  INT8_C(  32),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  90),  INT8_C(   0),  INT8_C(   0),  INT8_C(  65),  INT8_C(  69),  INT8_C(  83) } },
    { UINT64_C(13746519695230095469),
      { -INT8_C( 111), -INT8_C( 121),  INT8_C( 106), -INT8_C( 110),  INT8_C(  97),  INT8_C(  86),  INT8_C(  33), -INT8_C(   8),
         INT8_C(  94),  INT8_C(  65), -INT8_C(  75),  INT8_C(  86), -INT8_C(  17), -INT8_C(  93), -INT8_C(  83),  INT8_C(   8),
        -INT8_C( 117),  INT8_C(   7), -INT8_C( 121),  INT8_C( 116),  INT8_C(  72), -INT8_C(  52), -INT8_C(  57), -INT8_C(  75),
        -INT8_C(  99), -INT8_C(  31),  INT8_C(  90),  INT8_C( 113),  INT8_C(  83),  INT8_C(  31),  INT8_C(  47), -INT8_C(  28),
        -INT8_C(  90), -INT8_C( 103),  INT8_C( 119),  INT8_C(   7), -INT8_C(  17), -INT8_C( 104), -INT8_C(   1),  INT8_C(  77),
        -INT8_C(  38), -INT8_C(  76), -INT8_C(  93), -INT8_C(  55),  INT8_C(  88),  INT8_C(  80), -INT8_C(  46), -INT8_C(  29),
         INT8_C(  87),  INT8_C(  89),  INT8_C(  87), -INT8_C(  97),  INT8_C(  37),  INT8_C(  31),  INT8_C(  84), -INT8_C(  62),
         INT8_C(   0), -INT8_C(  82),  INT8_C(  51),  INT8_C(  83), -INT8_C(  51),  INT8_C(  98),  INT8_C(  55),  INT8_C( 115) },
      { -INT8_C( 111),  INT8_C(   0),  INT8_C( 106), -INT8_C( 110),  INT8_C(   0),  INT8_C(  86),  INT8_C(  33),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C(  17),  INT8_C(   0), -INT8_C(  83),  INT8_C(   8),
        -INT8_C( 117),  INT8_C(   0),  INT8_C(   0),  INT8_C( 116),  INT8_C(  72),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),
         INT8_C(   0),  INT8_C(   0),  INT8_C(  90),  INT8_C(   0),  INT8_C(   0),  INT8_C(  31),  INT8_C(   0), -INT8_C(  28),
         INT8_C(   0),  INT8_C(   0),  INT8_C( 119),  INT8_C(   0), -INT8_C(  17),  INT8_C(   0), -INT8_C(   1),  INT8_C(  77),
         INT8_C(   0), -INT8_C(  76),  INT8_C(   0),  INT8_C(   0),  INT8_C(  88),  INT8_C(  80), -INT8_C(  46),  INT8_C(   0),
         INT8_C(  87),  INT8_C(   0),  INT8_C(  87),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  84), -INT8_C(  62),
         INT8_C(   0), -INT8_C(  82),  INT8_C(  51),  INT8_C(  83), -INT8_C(  51),  INT8_C(  98),  INT8_C(   0),  INT8_C( 115) } },
    { UINT64_C( 2393516176265948923),
      {  INT8_C(  47), -INT8_C(  38), -INT8_C(  22), -INT8_C( 121),  INT8_C(  42), -INT8_C(  68),  INT8_C( 106), -INT8_C( 127),
         INT8_C(  21), -INT8_C(  62),  INT8_C(  32),  INT8_C(  59), -INT8_C(  31),  INT8_C( 117), -INT8_C(   3), -INT8_C(  31),
         INT8_C(  35),  INT8_C(  49),  INT8_C(  52), -INT8_C(  15), -INT8_C( 109),  INT8_C( 107),  INT8_C( 100), -INT8_C( 113),
         INT8_C(  26), -INT8_C(  33),  INT8_C( 121),  INT8_C(  97),  INT8_C(  90), -INT8_C(  79), -INT8_C( 126), -INT8_C( 119),
        -INT8_C( 117),  INT8_C( 108),  INT8_C(  16), -INT8_C(  74),  INT8_C(  41),  INT8_C( 122),  INT8_C(  55),  INT8_C(  62),
         INT8_C(  60),  INT8_C(  88),  INT8_C( 121),  INT8_C(  29), -INT8_C(  51),  INT8_C( 119), -INT8_C(   2), -INT8_C(  16),
        -INT8_C(  88),  INT8_C(  50), -INT8_C(  31),  INT8_C(  59), -INT8_C(  98),  INT8_C(  70), -INT8_C(  54), -INT8_C(  72),
         INT8_C(  37),  INT8_C(  68),  INT8_C(  25),      INT8_MAX, -INT8_C(  11), -INT8_C( 101),  INT8_C(   8),      INT8_MIN },
      {  INT8_C(  47), -INT8_C(  38),  INT8_C(   0), -INT8_C( 121),  INT8_C(  42), -INT8_C(  68),  INT8_C( 106), -INT8_C( 127),
         INT8_C(   0), -INT8_C(  62),  INT8_C(  32),  INT8_C(  59),  INT8_C(   0),  INT8_C( 117),  INT8_C(   0), -INT8_C(  31),
         INT8_C(  35),  INT8_C(  49),  INT8_C(   0), -INT8_C(  15), -INT8_C( 109),  INT8_C( 107),  INT8_C( 100),  INT8_C(   0),
         INT8_C(   0), -INT8_C(  33),  INT8_C(   0),  INT8_C(  97),  INT8_C(   0), -INT8_C(  79), -INT8_C( 126), -INT8_C( 119),
        -INT8_C( 117),  INT8_C( 108),  INT8_C(  16),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(  55),  INT8_C(   0),
         INT8_C(   0),  INT8_C(  88),  INT8_C(   0),  INT8_C(  29), -INT8_C(  51),  INT8_C( 119), -INT8_C(   2),  INT8_C(   0),
        -INT8_C(  88),  INT8_C(  50), -INT8_C(  31),  INT8_C(   0), -INT8_C(  98),  INT8_C(  70),  INT8_C(   0),  INT8_C(   0),
         INT8_C(  37),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0),  INT8_C(   0), -INT8_C( 101),  INT8_C(   0),  INT8_C(   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_epi8(test_vec[i].a);
    easysimd__mmask64 k = test_vec[i].k;
    easysimd__m512i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_loadu_epi8(k, &a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_loadu_epi8");
    easysimd_assert_equal_vi8(sizeof(r), (const int8_t*)&r, test_vec[i].r);
  }
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask64 k = easysimd_test_x86_random_mmask64();
    easysimd__m512i a = easysimd_test_x86_random_i8x64();
    easysimd__m512i r = easysimd_mm512_maskz_loadu_epi8(k, &a.i8[0]);

    easysimd_test_x86_write_mmask64(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i8x64(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i8x64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

  return 0;
}

static int
test_easysimd_mm512_maskz_loadu_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd__mmask32 k;
    const int16_t a[32];
    const int16_t r[32];
  } test_vec[] = {
    { UINT32_C(3411558677),
      {  INT16_C(  3253),  INT16_C( 31391),  INT16_C( 19905), -INT16_C(  1526), -INT16_C( 18566), -INT16_C(  6785),  INT16_C( 27973), -INT16_C( 31463),
        -INT16_C( 32634), -INT16_C(  8129),  INT16_C(  7371),  INT16_C( 30372),  INT16_C(  4860),  INT16_C(  4468), -INT16_C( 13229),  INT16_C(  2268),
         INT16_C( 31704), -INT16_C( 25982), -INT16_C( 29239),  INT16_C( 17300),  INT16_C(  4932), -INT16_C( 30424),  INT16_C( 16768),  INT16_C(  1550),
         INT16_C( 20162), -INT16_C( 29209), -INT16_C( 29846),  INT16_C( 26116),  INT16_C( 30877), -INT16_C(  3976),  INT16_C( 21573),  INT16_C(  7672) },
      {  INT16_C(  3253),  INT16_C(     0),  INT16_C( 19905),  INT16_C(     0), -INT16_C( 18566),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 32634),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 13229),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 17300),  INT16_C(  4932),  INT16_C(     0),  INT16_C( 16768),  INT16_C(     0),
         INT16_C( 20162), -INT16_C( 29209),  INT16_C(     0),  INT16_C( 26116),  INT16_C(     0),  INT16_C(     0),  INT16_C( 21573),  INT16_C(  7672) } },
    { UINT32_C(2578938832),
      {  INT16_C( 19208),  INT16_C( 19676),  INT16_C(  1118), -INT16_C(  8235), -INT16_C(  7355),  INT16_C(  2021), -INT16_C( 13263), -INT16_C( 25451),
        -INT16_C( 26281), -INT16_C(  3070),  INT16_C( 31249),  INT16_C( 22244), -INT16_C(  8753), -INT16_C( 24716),  INT16_C( 11096),  INT16_C( 24632),
         INT16_C(  5239), -INT16_C( 10836), -INT16_C( 32488),  INT16_C( 23988), -INT16_C( 26012), -INT16_C( 27035), -INT16_C(  1434), -INT16_C( 16846),
         INT16_C( 13459), -INT16_C( 23374), -INT16_C( 26705),  INT16_C( 32507),  INT16_C( 28532), -INT16_C( 13283),  INT16_C( 21914),  INT16_C(  4396) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  7355),  INT16_C(     0), -INT16_C( 13263), -INT16_C( 25451),
        -INT16_C( 26281), -INT16_C(  3070),  INT16_C(     0),  INT16_C( 22244), -INT16_C(  8753), -INT16_C( 24716),  INT16_C( 11096),  INT16_C(     0),
         INT16_C(  5239), -INT16_C( 10836), -INT16_C( 32488),  INT16_C(     0), -INT16_C( 26012), -INT16_C( 27035),  INT16_C(     0), -INT16_C( 16846),
         INT16_C( 13459),  INT16_C(     0),  INT16_C(     0),  INT16_C( 32507),  INT16_C( 28532),  INT16_C(     0),  INT16_C(     0),  INT16_C(  4396) } },
    { UINT32_C(2179455081),
      { -INT16_C( 25767), -INT16_C( 16930),  INT16_C( 17205), -INT16_C( 25517), -INT16_C( 31427), -INT16_C( 12198),  INT16_C(  3258),  INT16_C( 26997),
         INT16_C( 28835),  INT16_C(  6119),  INT16_C(  1247),  INT16_C( 31203),  INT16_C(  3929), -INT16_C( 15733),  INT16_C( 29415),  INT16_C( 16451),
         INT16_C(  8461),  INT16_C( 17406),  INT16_C( 20837), -INT16_C( 23841),  INT16_C( 14807), -INT16_C( 28301), -INT16_C(  6075), -INT16_C(  5638),
        -INT16_C(  7848),  INT16_C( 14080), -INT16_C(  6939),  INT16_C( 16048),  INT16_C( 15347), -INT16_C(  9472),  INT16_C( 17325), -INT16_C( 17637) },
      { -INT16_C( 25767),  INT16_C(     0),  INT16_C(     0), -INT16_C( 25517),  INT16_C(     0), -INT16_C( 12198),  INT16_C(  3258),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31203),  INT16_C(  3929),  INT16_C(     0),  INT16_C( 29415),  INT16_C( 16451),
         INT16_C(  8461),  INT16_C( 17406),  INT16_C( 20837),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28301), -INT16_C(  6075), -INT16_C(  5638),
        -INT16_C(  7848),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 17637) } },
    { UINT32_C(3388873060),
      { -INT16_C(  8853),  INT16_C( 17004), -INT16_C(  8426),  INT16_C( 23507), -INT16_C( 12857),  INT16_C(  8004),  INT16_C( 17838), -INT16_C( 27818),
         INT16_C(  1577),  INT16_C(  7377), -INT16_C( 11966), -INT16_C(  4105),  INT16_C(  4884),  INT16_C( 30890), -INT16_C( 22484), -INT16_C( 26814),
        -INT16_C( 20859), -INT16_C( 25639), -INT16_C( 21363),  INT16_C( 21751),  INT16_C( 15225),  INT16_C( 10099), -INT16_C( 13952), -INT16_C( 22086),
        -INT16_C( 29745),  INT16_C(  4550), -INT16_C( 17060),  INT16_C( 28673), -INT16_C( 21552), -INT16_C(   535),  INT16_C( 11092), -INT16_C(  9836) },
      {  INT16_C(     0),  INT16_C(     0), -INT16_C(  8426),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8004),  INT16_C( 17838),  INT16_C(     0),
         INT16_C(  1577),  INT16_C(     0),  INT16_C(     0), -INT16_C(  4105),  INT16_C(  4884),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 25639), -INT16_C( 21363),  INT16_C( 21751),  INT16_C( 15225),  INT16_C( 10099), -INT16_C( 13952), -INT16_C( 22086),
        -INT16_C( 29745),  INT16_C(     0),  INT16_C(     0),  INT16_C( 28673),  INT16_C(     0),  INT16_C(     0),  INT16_C( 11092), -INT16_C(  9836) } },
    { UINT32_C(1718972121),
      {  INT16_C( 27674), -INT16_C( 27462),  INT16_C( 11687),  INT16_C( 10427),  INT16_C( 30454), -INT16_C( 14895), -INT16_C( 26879),  INT16_C( 24279),
        -INT16_C( 10155),  INT16_C(  9678), -INT16_C( 18557), -INT16_C( 10462), -INT16_C( 18462), -INT16_C( 17487),  INT16_C(  9765),  INT16_C( 16161),
        -INT16_C(  9326),  INT16_C( 14803), -INT16_C( 28920), -INT16_C(   415),  INT16_C( 13061),  INT16_C(  1732), -INT16_C( 25654),  INT16_C(  8036),
         INT16_C( 13171), -INT16_C(  2491),  INT16_C( 26602), -INT16_C( 12850),  INT16_C( 32542),  INT16_C( 17288), -INT16_C( 21851),  INT16_C( 14211) },
      {  INT16_C( 27674),  INT16_C(     0),  INT16_C(     0),  INT16_C( 10427),  INT16_C( 30454),  INT16_C(     0), -INT16_C( 26879),  INT16_C( 24279),
         INT16_C(     0),  INT16_C(  9678), -INT16_C( 18557), -INT16_C( 10462),  INT16_C(     0), -INT16_C( 17487),  INT16_C(  9765),  INT16_C(     0),
        -INT16_C(  9326),  INT16_C(     0), -INT16_C( 28920),  INT16_C(     0),  INT16_C( 13061),  INT16_C(  1732), -INT16_C( 25654),  INT16_C(     0),
         INT16_C(     0), -INT16_C(  2491),  INT16_C( 26602),  INT16_C(     0),  INT16_C(     0),  INT16_C( 17288), -INT16_C( 21851),  INT16_C(     0) } },
    { UINT32_C(2389726853),
      { -INT16_C( 11547), -INT16_C(  5492),  INT16_C( 20485), -INT16_C( 12303),  INT16_C( 21995),  INT16_C( 24303),  INT16_C( 13448),  INT16_C( 29525),
         INT16_C(  9115), -INT16_C( 17856), -INT16_C( 14174),  INT16_C( 18429), -INT16_C( 32654), -INT16_C(  1922), -INT16_C(  4393), -INT16_C( 17274),
         INT16_C(  4800), -INT16_C( 14937), -INT16_C( 26525),  INT16_C( 20117), -INT16_C( 31507),  INT16_C( 30381),  INT16_C(   696),  INT16_C( 21481),
         INT16_C( 10533), -INT16_C( 14579),  INT16_C(  3057),  INT16_C( 25614), -INT16_C( 29557),  INT16_C( 25180), -INT16_C(  7558),  INT16_C( 15135) },
      { -INT16_C( 11547),  INT16_C(     0),  INT16_C( 20485),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 29525),
         INT16_C(     0), -INT16_C( 17856), -INT16_C( 14174),  INT16_C(     0), -INT16_C( 32654),  INT16_C(     0), -INT16_C(  4393),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31507),  INT16_C( 30381),  INT16_C(   696),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 14579),  INT16_C(  3057),  INT16_C( 25614),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 15135) } },
    { UINT32_C(1459668724),
      { -INT16_C( 27298),  INT16_C( 19366),  INT16_C( 21273), -INT16_C( 11839), -INT16_C( 21931),  INT16_C( 31269),  INT16_C( 13011), -INT16_C( 15039),
         INT16_C( 20285), -INT16_C( 14039), -INT16_C( 31269),  INT16_C( 21803),  INT16_C( 19047),  INT16_C( 23440), -INT16_C( 28400),  INT16_C( 28339),
         INT16_C( 22822),  INT16_C( 16570),  INT16_C( 31660),  INT16_C(   273),  INT16_C( 13862), -INT16_C(  1669), -INT16_C( 17303), -INT16_C( 22850),
        -INT16_C(  6389), -INT16_C(  6545), -INT16_C( 25748), -INT16_C( 11461), -INT16_C( 13083), -INT16_C(  2513), -INT16_C(  7587), -INT16_C( 31900) },
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 21273),  INT16_C(     0), -INT16_C( 21931),  INT16_C( 31269),  INT16_C( 13011), -INT16_C( 15039),
         INT16_C(     0), -INT16_C( 14039), -INT16_C( 31269),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28400),  INT16_C( 28339),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
        -INT16_C(  6389), -INT16_C(  6545), -INT16_C( 25748),  INT16_C(     0), -INT16_C( 13083),  INT16_C(     0), -INT16_C(  7587),  INT16_C(     0) } },
    { UINT32_C(3888324155),
      { -INT16_C( 10854), -INT16_C( 16152),  INT16_C( 25355),  INT16_C( 29881),  INT16_C( 30751),  INT16_C( 10779), -INT16_C( 30113), -INT16_C( 13296),
         INT16_C( 19237),  INT16_C(  2975), -INT16_C( 12777),  INT16_C( 29697),  INT16_C( 26032), -INT16_C(  5128), -INT16_C( 17532),  INT16_C(  7890),
        -INT16_C( 17776), -INT16_C( 25378), -INT16_C( 26851),  INT16_C( 15376),  INT16_C( 11023),  INT16_C( 28518),  INT16_C( 30390), -INT16_C(  9413),
        -INT16_C(  9534), -INT16_C(  9754), -INT16_C(  6231),  INT16_C( 22862),  INT16_C( 17997), -INT16_C( 11963),  INT16_C(  5889), -INT16_C( 27921) },
      { -INT16_C( 10854), -INT16_C( 16152),  INT16_C(     0),  INT16_C( 29881),  INT16_C( 30751),  INT16_C( 10779),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(  2975), -INT16_C( 12777),  INT16_C( 29697),  INT16_C( 26032),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
        -INT16_C( 17776), -INT16_C( 25378),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 30390), -INT16_C(  9413),
        -INT16_C(  9534), -INT16_C(  9754), -INT16_C(  6231),  INT16_C(     0),  INT16_C(     0), -INT16_C( 11963),  INT16_C(  5889), -INT16_C( 27921) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_loadu_epi16(k, &a);
    } EASYSIMD_TEST_PERF_END("_mm512_maskz_loadu_epi16");
    easysimd_assert_m512i_i16(r, ==, easysimd_mm512_loadu_si512(test_vec[i].r));
  }
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i r = easysimd_mm512_maskz_loadu_epi16(k, &a);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

  return 0;
}

static int
test_easysimd_mm512_maskz_loadu_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C( 5427),
      {  INT32_C(  1592239447),  INT32_C(   568703076), -INT32_C(  1136349295), -INT32_C(  1342249786), -INT32_C(   121141073), -INT32_C(  1249871380),  INT32_C(  1794886885),  INT32_C(  1803496980),
         INT32_C(   936011731), -INT32_C(  1336365281),  INT32_C(   627874911),  INT32_C(   802450304), -INT32_C(   550986765), -INT32_C(    90920939), -INT32_C(  2006675596), -INT32_C(  1846287171) },
      {  INT32_C(  1592239447),  INT32_C(   568703076),  INT32_C(           0),  INT32_C(           0), -INT32_C(   121141073), -INT32_C(  1249871380),  INT32_C(           0),  INT32_C(           0),
         INT32_C(   936011731),  INT32_C(           0),  INT32_C(   627874911),  INT32_C(           0), -INT32_C(   550986765),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(48459),
      {  INT32_C(   560818888), -INT32_C(  2017604582), -INT32_C(   957202703), -INT32_C(  1788746387),  INT32_C(  1480423108), -INT32_C(   706236047),  INT32_C(   767140921), -INT32_C(    18217931),
        -INT32_C(  1994434706),  INT32_C(   353426467), -INT32_C(  1998912998), -INT32_C(  1424147225),  INT32_C(   604199859),  INT32_C(  1190783500), -INT32_C(   999050354),  INT32_C(   650272183) },
      {  INT32_C(   560818888), -INT32_C(  2017604582),  INT32_C(           0), -INT32_C(  1788746387),  INT32_C(           0),  INT32_C(           0),  INT32_C(   767140921),  INT32_C(           0),
        -INT32_C(  1994434706),  INT32_C(           0), -INT32_C(  1998912998), -INT32_C(  1424147225),  INT32_C(   604199859),  INT32_C(  1190783500),  INT32_C(           0),  INT32_C(   650272183) } },
    { UINT16_C(57781),
      { -INT32_C(  1078077265), -INT32_C(   926820115),  INT32_C(  2097522784),  INT32_C(  1457043539), -INT32_C(   717167140), -INT32_C(  1635201493),  INT32_C(  1425752210),  INT32_C(   355840102),
         INT32_C(  1993667465),  INT32_C(   725587403), -INT32_C(  1868020675), -INT32_C(   655982084), -INT32_C(  1850923418), -INT32_C(  2026948876), -INT32_C(   589616522),  INT32_C(  1693520347) },
      { -INT32_C(  1078077265),  INT32_C(           0),  INT32_C(  2097522784),  INT32_C(           0), -INT32_C(   717167140), -INT32_C(  1635201493),  INT32_C(           0),  INT32_C(   355840102),
         INT32_C(  1993667465),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2026948876), -INT32_C(   589616522),  INT32_C(  1693520347) } },
    { UINT16_C(50436),
      {  INT32_C(   425381850), -INT32_C(  1570924550),  INT32_C(   220420391), -INT32_C(   550270671),  INT32_C(  1226123034), -INT32_C(  1972073554),  INT32_C(  1486573415), -INT32_C(  1927438413),
         INT32_C(  1755805550),  INT32_C(   889848846), -INT32_C(  1891488162), -INT32_C(   764512841),  INT32_C(  1260094364),  INT32_C(  1976930062), -INT32_C(  1848807202),  INT32_C(  2099243535) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   220420391),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(  1755805550),  INT32_C(           0), -INT32_C(  1891488162),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1848807202),  INT32_C(  2099243535) } },
    { UINT16_C(50785),
      { -INT32_C(   271945755), -INT32_C(   434296668),  INT32_C(   643552696), -INT32_C(  1028982617),  INT32_C(   408074307), -INT32_C(    74960850), -INT32_C(   504981566), -INT32_C(   106477804),
         INT32_C(  1541960119),  INT32_C(  1380058778), -INT32_C(  2106024485), -INT32_C(   666623339),  INT32_C(   166762203), -INT32_C(  1996195641),  INT32_C(   594209295), -INT32_C(   384036558) },
      { -INT32_C(   271945755),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(    74960850), -INT32_C(   504981566),  INT32_C(           0),
         INT32_C(           0),  INT32_C(  1380058778), -INT32_C(  2106024485),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   594209295), -INT32_C(   384036558) } },
    { UINT16_C( 1410),
      { -INT32_C(  2046092220), -INT32_C(   417012114), -INT32_C(  1408648856), -INT32_C(  2109479790), -INT32_C(   218428691), -INT32_C(    35911021),  INT32_C(  1242435118),  INT32_C(  1011847415),
         INT32_C(   465722029), -INT32_C(  1476204992),  INT32_C(   844434592), -INT32_C(   558589712),  INT32_C(   902868898), -INT32_C(   399332166), -INT32_C(  1305329477),  INT32_C(  2112783056) },
      {  INT32_C(           0), -INT32_C(   417012114),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1011847415),
         INT32_C(   465722029),  INT32_C(           0),  INT32_C(   844434592),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(45532),
      { -INT32_C(  1684595303),  INT32_C(   447166405),  INT32_C(   515020905),  INT32_C(  1187861622), -INT32_C(  1141733495), -INT32_C(  1560564369),  INT32_C(  1311034464), -INT32_C(   486604727),
        -INT32_C(   478243043),  INT32_C(   956114382),  INT32_C(   861319357), -INT32_C(  1904663804),  INT32_C(   441019563),  INT32_C(  2042381593),  INT32_C(  1523114001), -INT32_C(    29505567) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   515020905),  INT32_C(  1187861622), -INT32_C(  1141733495),  INT32_C(           0),  INT32_C(  1311034464), -INT32_C(   486604727),
        -INT32_C(   478243043),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   441019563),  INT32_C(  2042381593),  INT32_C(           0), -INT32_C(    29505567) } },
    { UINT16_C(47966),
      { -INT32_C(   538956319), -INT32_C(  1148215707),  INT32_C(  1272878033),  INT32_C(  1807190305),  INT32_C(  1622200484),  INT32_C(   306299210),  INT32_C(  1490625051),  INT32_C(    34813984),
        -INT32_C(   891161499),  INT32_C(  1669689490),  INT32_C(   632185603), -INT32_C(  1852807699), -INT32_C(  2131673034),  INT32_C(   479343105),  INT32_C(  1970629460),  INT32_C(   142051491) },
      {  INT32_C(           0), -INT32_C(  1148215707),  INT32_C(  1272878033),  INT32_C(  1807190305),  INT32_C(  1622200484),  INT32_C(           0),  INT32_C(  1490625051),  INT32_C(           0),
        -INT32_C(   891161499),  INT32_C(  1669689490),  INT32_C(           0), -INT32_C(  1852807699), -INT32_C(  2131673034),  INT32_C(   479343105),  INT32_C(           0),  INT32_C(   142051491) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_loadu_epi32(k, &a);
    } EASYSIMD_TEST_PERF_END("_mm512_maskz_loadu_epi32");
    easysimd_assert_m512i_i32(r, ==, easysimd_mm512_loadu_si512(test_vec[i].r));
  }
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_maskz_loadu_epi32(k, &a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

  return 0;
}

static int
test_easysimd_mm512_mask_loadu_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(   153273140),  INT32_C(   660505800),  INT32_C(  1707591292), -INT32_C(   336553590),  INT32_C(   767526665),  INT32_C(   434848387), -INT32_C(  1324012280),  INT32_C(   625854449),
         INT32_C(  1395552395),  INT32_C(  1887079668), -INT32_C(   573226669), -INT32_C(   473315878), -INT32_C(   804222899), -INT32_C(   739640374),  INT32_C(   444923688),  INT32_C(  1363137222) },
      UINT16_C(28226),
      {  INT32_C(   519714468),  INT32_C(  2086620583), -INT32_C(   196986581),  INT32_C(   746360604),  INT32_C(  1227376479), -INT32_C(  1639428070), -INT32_C(  1435496854),  INT32_C(    68727392),
        -INT32_C(  1876749591), -INT32_C(  1962114464), -INT32_C(   662745412),  INT32_C(  1023802333),  INT32_C(  1552297282), -INT32_C(   386150787),  INT32_C(  1016228828),  INT32_C(   121743902) },
      {  INT32_C(   153273140),  INT32_C(  2086620583),  INT32_C(  1707591292), -INT32_C(   336553590),  INT32_C(   767526665),  INT32_C(   434848387), -INT32_C(  1435496854),  INT32_C(   625854449),
         INT32_C(  1395552395), -INT32_C(  1962114464), -INT32_C(   662745412),  INT32_C(  1023802333), -INT32_C(   804222899), -INT32_C(   386150787),  INT32_C(  1016228828),  INT32_C(  1363137222) } },
    { {  INT32_C(   479683772), -INT32_C(  1566071834), -INT32_C(   814012686),  INT32_C(  1678540834),  INT32_C(   717329069),  INT32_C(  1024638048),  INT32_C(  1165599783),  INT32_C(   189577806),
         INT32_C(    86500126),  INT32_C(  2041040775),  INT32_C(   390603509),  INT32_C(  1333548450),  INT32_C(  1215970791),  INT32_C(   545623289),  INT32_C(  2137456177), -INT32_C(   678776135) },
      UINT16_C(45718),
      { -INT32_C(  2071912996), -INT32_C(   559516010),  INT32_C(   171133326),  INT32_C(   306650008), -INT32_C(   392216477), -INT32_C(   957951904), -INT32_C(   629563569),  INT32_C(  1401687927),
        -INT32_C(  1026093780),  INT32_C(   295730819),  INT32_C(  1595659463),  INT32_C(  1383228143),  INT32_C(    54137250),  INT32_C(   818487521),  INT32_C(   923419328),  INT32_C(  2106234449) },
      {  INT32_C(   479683772), -INT32_C(   559516010),  INT32_C(   171133326),  INT32_C(  1678540834), -INT32_C(   392216477),  INT32_C(  1024638048),  INT32_C(  1165599783),  INT32_C(  1401687927),
         INT32_C(    86500126),  INT32_C(   295730819),  INT32_C(   390603509),  INT32_C(  1333548450),  INT32_C(    54137250),  INT32_C(   818487521),  INT32_C(  2137456177),  INT32_C(  2106234449) } },
    { {  INT32_C(   658465443), -INT32_C(  1489444640), -INT32_C(  1559866188),  INT32_C(  1509259446),  INT32_C(  1784426377),  INT32_C(   261825871), -INT32_C(  1186552472), -INT32_C(   549990084),
         INT32_C(   319190579),  INT32_C(   179978070),  INT32_C(  1236123795), -INT32_C(  1029529031),  INT32_C(   556662481), -INT32_C(  1942960092), -INT32_C(  1455065235),  INT32_C(  2072607816) },
      UINT16_C(36851),
      {  INT32_C(  1221478798),  INT32_C(    17326420),  INT32_C(  1302610347),  INT32_C(   827094276), -INT32_C(   939954026),  INT32_C(  1094608636), -INT32_C(  1715567088), -INT32_C(  1893158655),
         INT32_C(  1322776570),  INT32_C(    55631704),  INT32_C(   626127905),  INT32_C(     5676393),  INT32_C(   164056845), -INT32_C(   985004619), -INT32_C(  1939994485), -INT32_C(  1273264199) },
      {  INT32_C(  1221478798),  INT32_C(    17326420), -INT32_C(  1559866188),  INT32_C(  1509259446), -INT32_C(   939954026),  INT32_C(  1094608636), -INT32_C(  1715567088), -INT32_C(  1893158655),
         INT32_C(  1322776570),  INT32_C(    55631704),  INT32_C(   626127905),  INT32_C(     5676393),  INT32_C(   556662481), -INT32_C(  1942960092), -INT32_C(  1455065235), -INT32_C(  1273264199) } },
    { { -INT32_C(   704449922), -INT32_C(   220572975), -INT32_C(  1340658874), -INT32_C(   709857848),  INT32_C(  1927182268),  INT32_C(   121055356), -INT32_C(   359426511), -INT32_C(  1684099555),
         INT32_C(  1903272352),  INT32_C(   979651571),  INT32_C(  1055554422), -INT32_C(  1525441815), -INT32_C(  1927810799),  INT32_C(  1268010778),  INT32_C(    37038053),  INT32_C(  1973277909) },
      UINT16_C( 3701),
      {  INT32_C(  1247439078), -INT32_C(  1933127518),  INT32_C(   572960527),  INT32_C(  1813264212),  INT32_C(  1488662212), -INT32_C(  1367367559),  INT32_C(  1065505954), -INT32_C(  1337067575),
         INT32_C(    49981280), -INT32_C(  2020622216), -INT32_C(   995510929), -INT32_C(  1322205715),  INT32_C(  1678437355),  INT32_C(   756255115), -INT32_C(  1486055970), -INT32_C(   313017971) },
      {  INT32_C(  1247439078), -INT32_C(   220572975),  INT32_C(   572960527), -INT32_C(   709857848),  INT32_C(  1488662212), -INT32_C(  1367367559),  INT32_C(  1065505954), -INT32_C(  1684099555),
         INT32_C(  1903272352), -INT32_C(  2020622216), -INT32_C(   995510929), -INT32_C(  1322205715), -INT32_C(  1927810799),  INT32_C(  1268010778),  INT32_C(    37038053),  INT32_C(  1973277909) } },
    { { -INT32_C(   638627231), -INT32_C(  2107605486),  INT32_C(   558238004), -INT32_C(  1294829881), -INT32_C(   333980575),  INT32_C(  1142499942),  INT32_C(  1290503615), -INT32_C(  1623637186),
        -INT32_C(  1485297259), -INT32_C(   618014553), -INT32_C(  1443074078),  INT32_C(  1180422117),  INT32_C(   288518827),  INT32_C(  1549093788),  INT32_C(   262685136),  INT32_C(   430891652) },
      UINT16_C( 9995),
      { -INT32_C(   352341312), -INT32_C(  1973821042), -INT32_C(   430358646), -INT32_C(  1202190971),  INT32_C(  1812198678), -INT32_C(   106048431), -INT32_C(  1847905821),  INT32_C(   179889738),
         INT32_C(   670349465),  INT32_C(   615599769),  INT32_C(   302648205),  INT32_C(   650797584), -INT32_C(  1466773929), -INT32_C(  2052964446), -INT32_C(  1172931216), -INT32_C(    37367964) },
      { -INT32_C(   352341312), -INT32_C(  1973821042),  INT32_C(   558238004), -INT32_C(  1202190971), -INT32_C(   333980575),  INT32_C(  1142499942),  INT32_C(  1290503615), -INT32_C(  1623637186),
         INT32_C(   670349465),  INT32_C(   615599769),  INT32_C(   302648205),  INT32_C(  1180422117),  INT32_C(   288518827), -INT32_C(  2052964446),  INT32_C(   262685136),  INT32_C(   430891652) } },
    { {  INT32_C(   539277703), -INT32_C(  1807428345), -INT32_C(   257470752),  INT32_C(   135754161), -INT32_C(   508450497),  INT32_C(  1499878377),  INT32_C(   873758160), -INT32_C(   751707828),
        -INT32_C(  1695328878),  INT32_C(   187578411),  INT32_C(   939316614), -INT32_C(  2042621113), -INT32_C(  1519849028),  INT32_C(   352308804), -INT32_C(  1756818613),  INT32_C(  2120907500) },
      UINT16_C(24272),
      {  INT32_C(  1201077016),  INT32_C(    35396614), -INT32_C(  1810537644),  INT32_C(  1384501994),  INT32_C(  1981860215),  INT32_C(   646540509),  INT32_C(  1856075011),  INT32_C(   214725108),
         INT32_C(  1918067308), -INT32_C(   764055426), -INT32_C(  1117353261), -INT32_C(   737154212), -INT32_C(  1840631628), -INT32_C(  1615276900),  INT32_C(  1024285001),  INT32_C(   910875082) },
      {  INT32_C(   539277703), -INT32_C(  1807428345), -INT32_C(   257470752),  INT32_C(   135754161),  INT32_C(  1981860215),  INT32_C(  1499878377),  INT32_C(  1856075011),  INT32_C(   214725108),
        -INT32_C(  1695328878), -INT32_C(   764055426), -INT32_C(  1117353261), -INT32_C(   737154212), -INT32_C(  1840631628),  INT32_C(   352308804),  INT32_C(  1024285001),  INT32_C(  2120907500) } },
    { { -INT32_C(  1163289285), -INT32_C(   510910963),  INT32_C(    94303144), -INT32_C(  1814450466),  INT32_C(  2049254366),  INT32_C(  1092214263),  INT32_C(    25044791),  INT32_C(  1010354176),
         INT32_C(  1945559398), -INT32_C(  1487633665),  INT32_C(  1420620661),  INT32_C(  2145879457), -INT32_C(  1594291031),  INT32_C(   551621353),  INT32_C(   975331385), -INT32_C(  1904846296) },
      UINT16_C(27707),
      {  INT32_C(  1458452994), -INT32_C(  1907792671), -INT32_C(  1626084680),  INT32_C(  1655422057),  INT32_C(  1047893085), -INT32_C(   677466443),  INT32_C(  1580320744),  INT32_C(  1472883797),
        -INT32_C(  2018592346), -INT32_C(   719980771),  INT32_C(  1265904098),  INT32_C(  1118707685),  INT32_C(  1753293747), -INT32_C(  1170268207),  INT32_C(  1008234726), -INT32_C(  2104237092) },
      {  INT32_C(  1458452994), -INT32_C(  1907792671),  INT32_C(    94303144),  INT32_C(  1655422057),  INT32_C(  1047893085), -INT32_C(   677466443),  INT32_C(    25044791),  INT32_C(  1010354176),
         INT32_C(  1945559398), -INT32_C(  1487633665),  INT32_C(  1265904098),  INT32_C(  1118707685), -INT32_C(  1594291031), -INT32_C(  1170268207),  INT32_C(  1008234726), -INT32_C(  1904846296) } },
    { { -INT32_C(  1190510180),  INT32_C(   462298937),  INT32_C(   778437192), -INT32_C(   730852319),  INT32_C(   155054391), -INT32_C(   138183663), -INT32_C(   919348243),  INT32_C(  1514981310),
         INT32_C(  1091786248), -INT32_C(  1101225611), -INT32_C(   991116381),  INT32_C(   261709015),  INT32_C(  1595463246),  INT32_C(  1062656850),  INT32_C(  1963559606),  INT32_C(  1506760017) },
      UINT16_C(58283),
      { -INT32_C(   142335845), -INT32_C(   893769506), -INT32_C(  2061004052), -INT32_C(  1185188447),  INT32_C(   714387156), -INT32_C(   155956499), -INT32_C(  1890908737), -INT32_C(    93129121),
        -INT32_C(   168692201),  INT32_C(   197176350), -INT32_C(   544151746),  INT32_C(   815328348), -INT32_C(  2024133478),  INT32_C(   914231158),  INT32_C(  1942341908), -INT32_C(   680708160) },
      { -INT32_C(   142335845), -INT32_C(   893769506),  INT32_C(   778437192), -INT32_C(  1185188447),  INT32_C(   155054391), -INT32_C(   155956499), -INT32_C(   919348243), -INT32_C(    93129121),
        -INT32_C(   168692201),  INT32_C(   197176350), -INT32_C(   991116381),  INT32_C(   261709015),  INT32_C(  1595463246),  INT32_C(   914231158),  INT32_C(  1942341908), -INT32_C(   680708160) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_si512(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i r;

    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_loadu_epi32(src, k, &a);
    } EASYSIMD_TEST_PERF_END("_mm512_mask_loadu_epi32");
    easysimd_assert_m512i_i32(r, ==, easysimd_mm512_loadu_si512(test_vec[i].r));
  }
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_mask_loadu_epi32(src, k, &a);

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif

  return 0;
}

static int
test_easysimd_mm512_mask_loadu_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const easysimd_float32 src[16];
    const uint16_t k;
    const easysimd_float32 a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -133.94), EASYSIMD_FLOAT32_C(  -562.03), EASYSIMD_FLOAT32_C(   826.76), EASYSIMD_FLOAT32_C(  -338.95),
        EASYSIMD_FLOAT32_C(  -929.48), EASYSIMD_FLOAT32_C(   619.85), EASYSIMD_FLOAT32_C(   493.91), EASYSIMD_FLOAT32_C(   240.62),
        EASYSIMD_FLOAT32_C(   126.08), EASYSIMD_FLOAT32_C(   238.85), EASYSIMD_FLOAT32_C(   719.97), EASYSIMD_FLOAT32_C(     8.44),
        EASYSIMD_FLOAT32_C(   258.67), EASYSIMD_FLOAT32_C(   393.61), EASYSIMD_FLOAT32_C(  -893.92), EASYSIMD_FLOAT32_C(  -877.97) },
      UINT16_C(31706),
      { EASYSIMD_FLOAT32_C(  -926.18), EASYSIMD_FLOAT32_C(  -239.95), EASYSIMD_FLOAT32_C(   763.95), EASYSIMD_FLOAT32_C(   507.62),
        EASYSIMD_FLOAT32_C(  -952.53), EASYSIMD_FLOAT32_C(   454.05), EASYSIMD_FLOAT32_C(  -586.82), EASYSIMD_FLOAT32_C(   -40.88),
        EASYSIMD_FLOAT32_C(   -34.77), EASYSIMD_FLOAT32_C(  -826.06), EASYSIMD_FLOAT32_C(   498.39), EASYSIMD_FLOAT32_C(   769.87),
        EASYSIMD_FLOAT32_C(  -101.24), EASYSIMD_FLOAT32_C(  -635.55), EASYSIMD_FLOAT32_C(  -792.16), EASYSIMD_FLOAT32_C(  -274.48) },
      { EASYSIMD_FLOAT32_C(  -133.94), EASYSIMD_FLOAT32_C(  -239.95), EASYSIMD_FLOAT32_C(   826.76), EASYSIMD_FLOAT32_C(   507.62),
        EASYSIMD_FLOAT32_C(  -952.53), EASYSIMD_FLOAT32_C(   619.85), EASYSIMD_FLOAT32_C(  -586.82), EASYSIMD_FLOAT32_C(   -40.88),
        EASYSIMD_FLOAT32_C(   -34.77), EASYSIMD_FLOAT32_C(  -826.06), EASYSIMD_FLOAT32_C(   719.97), EASYSIMD_FLOAT32_C(   769.87),
        EASYSIMD_FLOAT32_C(  -101.24), EASYSIMD_FLOAT32_C(  -635.55), EASYSIMD_FLOAT32_C(  -792.16), EASYSIMD_FLOAT32_C(  -877.97) } },
    { { EASYSIMD_FLOAT32_C(    25.50), EASYSIMD_FLOAT32_C(  -721.65), EASYSIMD_FLOAT32_C(  -654.64), EASYSIMD_FLOAT32_C(  -480.59),
        EASYSIMD_FLOAT32_C(   518.97), EASYSIMD_FLOAT32_C(   471.45), EASYSIMD_FLOAT32_C(   758.26), EASYSIMD_FLOAT32_C(   238.94),
        EASYSIMD_FLOAT32_C(  -520.11), EASYSIMD_FLOAT32_C(    16.93), EASYSIMD_FLOAT32_C(  -367.45), EASYSIMD_FLOAT32_C(  -414.04),
        EASYSIMD_FLOAT32_C(   138.96), EASYSIMD_FLOAT32_C(   -54.22), EASYSIMD_FLOAT32_C(  -329.44), EASYSIMD_FLOAT32_C(   212.78) },
      UINT16_C(10020),
      { EASYSIMD_FLOAT32_C(  -279.60), EASYSIMD_FLOAT32_C(   753.31), EASYSIMD_FLOAT32_C(   888.57), EASYSIMD_FLOAT32_C(   133.58),
        EASYSIMD_FLOAT32_C(  -287.57), EASYSIMD_FLOAT32_C(  -146.21), EASYSIMD_FLOAT32_C(   307.51), EASYSIMD_FLOAT32_C(  -789.18),
        EASYSIMD_FLOAT32_C(  -376.33), EASYSIMD_FLOAT32_C(  -793.73), EASYSIMD_FLOAT32_C(  -424.73), EASYSIMD_FLOAT32_C(  -168.50),
        EASYSIMD_FLOAT32_C(   -68.21), EASYSIMD_FLOAT32_C(   600.77), EASYSIMD_FLOAT32_C(   109.86), EASYSIMD_FLOAT32_C(   277.15) },
      { EASYSIMD_FLOAT32_C(    25.50), EASYSIMD_FLOAT32_C(  -721.65), EASYSIMD_FLOAT32_C(   888.57), EASYSIMD_FLOAT32_C(  -480.59),
        EASYSIMD_FLOAT32_C(   518.97), EASYSIMD_FLOAT32_C(  -146.21), EASYSIMD_FLOAT32_C(   758.26), EASYSIMD_FLOAT32_C(   238.94),
        EASYSIMD_FLOAT32_C(  -376.33), EASYSIMD_FLOAT32_C(  -793.73), EASYSIMD_FLOAT32_C(  -424.73), EASYSIMD_FLOAT32_C(  -414.04),
        EASYSIMD_FLOAT32_C(   138.96), EASYSIMD_FLOAT32_C(   600.77), EASYSIMD_FLOAT32_C(  -329.44), EASYSIMD_FLOAT32_C(   212.78) } },
    { { EASYSIMD_FLOAT32_C(  -879.82), EASYSIMD_FLOAT32_C(  -371.17), EASYSIMD_FLOAT32_C(  -251.40), EASYSIMD_FLOAT32_C(   878.44),
        EASYSIMD_FLOAT32_C(   867.77), EASYSIMD_FLOAT32_C(   228.49), EASYSIMD_FLOAT32_C(  -104.64), EASYSIMD_FLOAT32_C(  -499.68),
        EASYSIMD_FLOAT32_C(   814.45), EASYSIMD_FLOAT32_C(  -965.67), EASYSIMD_FLOAT32_C(   446.10), EASYSIMD_FLOAT32_C(  -514.98),
        EASYSIMD_FLOAT32_C(   247.11), EASYSIMD_FLOAT32_C(   151.93), EASYSIMD_FLOAT32_C(   -80.47), EASYSIMD_FLOAT32_C(   967.51) },
      UINT16_C(58070),
      { EASYSIMD_FLOAT32_C(   101.09), EASYSIMD_FLOAT32_C(   617.66), EASYSIMD_FLOAT32_C(   661.89), EASYSIMD_FLOAT32_C(  -591.40),
        EASYSIMD_FLOAT32_C(   828.48), EASYSIMD_FLOAT32_C(  -714.44), EASYSIMD_FLOAT32_C(  -385.13), EASYSIMD_FLOAT32_C(  -596.25),
        EASYSIMD_FLOAT32_C(   117.06), EASYSIMD_FLOAT32_C(   546.66), EASYSIMD_FLOAT32_C(  -995.48), EASYSIMD_FLOAT32_C(  -773.08),
        EASYSIMD_FLOAT32_C(  -176.19), EASYSIMD_FLOAT32_C(  -875.30), EASYSIMD_FLOAT32_C(  -144.25), EASYSIMD_FLOAT32_C(   572.41) },
      { EASYSIMD_FLOAT32_C(  -879.82), EASYSIMD_FLOAT32_C(   617.66), EASYSIMD_FLOAT32_C(   661.89), EASYSIMD_FLOAT32_C(   878.44),
        EASYSIMD_FLOAT32_C(   828.48), EASYSIMD_FLOAT32_C(   228.49), EASYSIMD_FLOAT32_C(  -385.13), EASYSIMD_FLOAT32_C(  -596.25),
        EASYSIMD_FLOAT32_C(   814.45), EASYSIMD_FLOAT32_C(   546.66), EASYSIMD_FLOAT32_C(   446.10), EASYSIMD_FLOAT32_C(  -514.98),
        EASYSIMD_FLOAT32_C(   247.11), EASYSIMD_FLOAT32_C(  -875.30), EASYSIMD_FLOAT32_C(  -144.25), EASYSIMD_FLOAT32_C(   572.41) } },
    { { EASYSIMD_FLOAT32_C(  -996.86), EASYSIMD_FLOAT32_C(  -276.49), EASYSIMD_FLOAT32_C(  -199.10), EASYSIMD_FLOAT32_C(  -101.50),
        EASYSIMD_FLOAT32_C(   223.83), EASYSIMD_FLOAT32_C(  -384.65), EASYSIMD_FLOAT32_C(   -67.17), EASYSIMD_FLOAT32_C(  -330.07),
        EASYSIMD_FLOAT32_C(   100.37), EASYSIMD_FLOAT32_C(  -820.07), EASYSIMD_FLOAT32_C(   821.86), EASYSIMD_FLOAT32_C(  -980.10),
        EASYSIMD_FLOAT32_C(  -852.56), EASYSIMD_FLOAT32_C(  -272.91), EASYSIMD_FLOAT32_C(  -172.00), EASYSIMD_FLOAT32_C(   248.53) },
      UINT16_C( 8869),
      { EASYSIMD_FLOAT32_C(   657.13), EASYSIMD_FLOAT32_C(  -826.77), EASYSIMD_FLOAT32_C(  -224.55), EASYSIMD_FLOAT32_C(  -728.00),
        EASYSIMD_FLOAT32_C(  -423.02), EASYSIMD_FLOAT32_C(   892.52), EASYSIMD_FLOAT32_C(   818.66), EASYSIMD_FLOAT32_C(  -418.50),
        EASYSIMD_FLOAT32_C(  -880.56), EASYSIMD_FLOAT32_C(  -357.53), EASYSIMD_FLOAT32_C(  -293.80), EASYSIMD_FLOAT32_C(   -24.82),
        EASYSIMD_FLOAT32_C(  -785.11), EASYSIMD_FLOAT32_C(  -290.66), EASYSIMD_FLOAT32_C(   698.70), EASYSIMD_FLOAT32_C(    15.79) },
      { EASYSIMD_FLOAT32_C(   657.13), EASYSIMD_FLOAT32_C(  -276.49), EASYSIMD_FLOAT32_C(  -224.55), EASYSIMD_FLOAT32_C(  -101.50),
        EASYSIMD_FLOAT32_C(   223.83), EASYSIMD_FLOAT32_C(   892.52), EASYSIMD_FLOAT32_C(   -67.17), EASYSIMD_FLOAT32_C(  -418.50),
        EASYSIMD_FLOAT32_C(   100.37), EASYSIMD_FLOAT32_C(  -357.53), EASYSIMD_FLOAT32_C(   821.86), EASYSIMD_FLOAT32_C(  -980.10),
        EASYSIMD_FLOAT32_C(  -852.56), EASYSIMD_FLOAT32_C(  -290.66), EASYSIMD_FLOAT32_C(  -172.00), EASYSIMD_FLOAT32_C(   248.53) } },
    { { EASYSIMD_FLOAT32_C(   607.84), EASYSIMD_FLOAT32_C(   -77.47), EASYSIMD_FLOAT32_C(   631.14), EASYSIMD_FLOAT32_C(  -459.34),
        EASYSIMD_FLOAT32_C(   592.46), EASYSIMD_FLOAT32_C(  -268.49), EASYSIMD_FLOAT32_C(  -279.41), EASYSIMD_FLOAT32_C(   414.32),
        EASYSIMD_FLOAT32_C(  -248.58), EASYSIMD_FLOAT32_C(  -131.96), EASYSIMD_FLOAT32_C(  -858.59), EASYSIMD_FLOAT32_C(   579.42),
        EASYSIMD_FLOAT32_C(  -883.43), EASYSIMD_FLOAT32_C(  -513.83), EASYSIMD_FLOAT32_C(  -930.69), EASYSIMD_FLOAT32_C(   773.70) },
      UINT16_C(10416),
      { EASYSIMD_FLOAT32_C(  -954.30), EASYSIMD_FLOAT32_C(   236.38), EASYSIMD_FLOAT32_C(  -262.72), EASYSIMD_FLOAT32_C(   864.37),
        EASYSIMD_FLOAT32_C(   817.88), EASYSIMD_FLOAT32_C(  -143.28), EASYSIMD_FLOAT32_C(  -493.16), EASYSIMD_FLOAT32_C(  -475.92),
        EASYSIMD_FLOAT32_C(   831.90), EASYSIMD_FLOAT32_C(  -278.27), EASYSIMD_FLOAT32_C(   233.42), EASYSIMD_FLOAT32_C(   530.60),
        EASYSIMD_FLOAT32_C(   737.51), EASYSIMD_FLOAT32_C(  -158.75), EASYSIMD_FLOAT32_C(  -546.87), EASYSIMD_FLOAT32_C(   368.66) },
      { EASYSIMD_FLOAT32_C(   607.84), EASYSIMD_FLOAT32_C(   -77.47), EASYSIMD_FLOAT32_C(   631.14), EASYSIMD_FLOAT32_C(  -459.34),
        EASYSIMD_FLOAT32_C(   817.88), EASYSIMD_FLOAT32_C(  -143.28), EASYSIMD_FLOAT32_C(  -279.41), EASYSIMD_FLOAT32_C(  -475.92),
        EASYSIMD_FLOAT32_C(  -248.58), EASYSIMD_FLOAT32_C(  -131.96), EASYSIMD_FLOAT32_C(  -858.59), EASYSIMD_FLOAT32_C(   530.60),
        EASYSIMD_FLOAT32_C(  -883.43), EASYSIMD_FLOAT32_C(  -158.75), EASYSIMD_FLOAT32_C(  -930.69), EASYSIMD_FLOAT32_C(   773.70) } },
    { { EASYSIMD_FLOAT32_C(   381.92), EASYSIMD_FLOAT32_C(  -954.41), EASYSIMD_FLOAT32_C(  -899.83), EASYSIMD_FLOAT32_C(  -897.49),
        EASYSIMD_FLOAT32_C(   459.91), EASYSIMD_FLOAT32_C(  -148.41), EASYSIMD_FLOAT32_C(   -29.45), EASYSIMD_FLOAT32_C(   601.32),
        EASYSIMD_FLOAT32_C(  -569.00), EASYSIMD_FLOAT32_C(    87.12), EASYSIMD_FLOAT32_C(  -912.52), EASYSIMD_FLOAT32_C(  -499.69),
        EASYSIMD_FLOAT32_C(  -139.18), EASYSIMD_FLOAT32_C(  -253.12), EASYSIMD_FLOAT32_C(   345.08), EASYSIMD_FLOAT32_C(   -93.48) },
      UINT16_C(22290),
      { EASYSIMD_FLOAT32_C(  -229.11), EASYSIMD_FLOAT32_C(   801.15), EASYSIMD_FLOAT32_C(   -60.92), EASYSIMD_FLOAT32_C(   277.73),
        EASYSIMD_FLOAT32_C(  -674.77), EASYSIMD_FLOAT32_C(  -229.02), EASYSIMD_FLOAT32_C(   999.45), EASYSIMD_FLOAT32_C(   558.65),
        EASYSIMD_FLOAT32_C(  -698.42), EASYSIMD_FLOAT32_C(   736.97), EASYSIMD_FLOAT32_C(  -600.10), EASYSIMD_FLOAT32_C(  -245.29),
        EASYSIMD_FLOAT32_C(   105.62), EASYSIMD_FLOAT32_C(   781.82), EASYSIMD_FLOAT32_C(  -199.70), EASYSIMD_FLOAT32_C(   205.80) },
      { EASYSIMD_FLOAT32_C(   381.92), EASYSIMD_FLOAT32_C(   801.15), EASYSIMD_FLOAT32_C(  -899.83), EASYSIMD_FLOAT32_C(  -897.49),
        EASYSIMD_FLOAT32_C(  -674.77), EASYSIMD_FLOAT32_C(  -148.41), EASYSIMD_FLOAT32_C(   -29.45), EASYSIMD_FLOAT32_C(   601.32),
        EASYSIMD_FLOAT32_C(  -698.42), EASYSIMD_FLOAT32_C(   736.97), EASYSIMD_FLOAT32_C(  -600.10), EASYSIMD_FLOAT32_C(  -499.69),
        EASYSIMD_FLOAT32_C(   105.62), EASYSIMD_FLOAT32_C(  -253.12), EASYSIMD_FLOAT32_C(  -199.70), EASYSIMD_FLOAT32_C(   -93.48) } },
    { { EASYSIMD_FLOAT32_C(   884.33), EASYSIMD_FLOAT32_C(  -739.79), EASYSIMD_FLOAT32_C(  -942.62), EASYSIMD_FLOAT32_C(  -145.12),
        EASYSIMD_FLOAT32_C(   861.53), EASYSIMD_FLOAT32_C(  -511.62), EASYSIMD_FLOAT32_C(   941.99), EASYSIMD_FLOAT32_C(   949.01),
        EASYSIMD_FLOAT32_C(   -11.30), EASYSIMD_FLOAT32_C(  -197.19), EASYSIMD_FLOAT32_C(  -304.11), EASYSIMD_FLOAT32_C(  -666.22),
        EASYSIMD_FLOAT32_C(   709.33), EASYSIMD_FLOAT32_C(  -320.84), EASYSIMD_FLOAT32_C(  -583.86), EASYSIMD_FLOAT32_C(  -519.78) },
      UINT16_C(56775),
      { EASYSIMD_FLOAT32_C(   757.95), EASYSIMD_FLOAT32_C(  -194.47), EASYSIMD_FLOAT32_C(  -873.80), EASYSIMD_FLOAT32_C(   757.40),
        EASYSIMD_FLOAT32_C(  -635.83), EASYSIMD_FLOAT32_C(  -572.22), EASYSIMD_FLOAT32_C(   494.37), EASYSIMD_FLOAT32_C(  -235.92),
        EASYSIMD_FLOAT32_C(   182.50), EASYSIMD_FLOAT32_C(  -400.01), EASYSIMD_FLOAT32_C(  -454.11), EASYSIMD_FLOAT32_C(   982.80),
        EASYSIMD_FLOAT32_C(   805.79), EASYSIMD_FLOAT32_C(  -569.78), EASYSIMD_FLOAT32_C(  -756.99), EASYSIMD_FLOAT32_C(   863.17) },
      { EASYSIMD_FLOAT32_C(   757.95), EASYSIMD_FLOAT32_C(  -194.47), EASYSIMD_FLOAT32_C(  -873.80), EASYSIMD_FLOAT32_C(  -145.12),
        EASYSIMD_FLOAT32_C(   861.53), EASYSIMD_FLOAT32_C(  -511.62), EASYSIMD_FLOAT32_C(   494.37), EASYSIMD_FLOAT32_C(  -235.92),
        EASYSIMD_FLOAT32_C(   182.50), EASYSIMD_FLOAT32_C(  -197.19), EASYSIMD_FLOAT32_C(  -454.11), EASYSIMD_FLOAT32_C(   982.80),
        EASYSIMD_FLOAT32_C(   805.79), EASYSIMD_FLOAT32_C(  -320.84), EASYSIMD_FLOAT32_C(  -756.99), EASYSIMD_FLOAT32_C(   863.17) } },
    { { EASYSIMD_FLOAT32_C(   285.10), EASYSIMD_FLOAT32_C(  -895.46), EASYSIMD_FLOAT32_C(  -648.45), EASYSIMD_FLOAT32_C(   227.09),
        EASYSIMD_FLOAT32_C(  -946.45), EASYSIMD_FLOAT32_C(   340.25), EASYSIMD_FLOAT32_C(  -970.10), EASYSIMD_FLOAT32_C(  -250.56),
        EASYSIMD_FLOAT32_C(   674.03), EASYSIMD_FLOAT32_C(   739.23), EASYSIMD_FLOAT32_C(   428.59), EASYSIMD_FLOAT32_C(  -909.84),
        EASYSIMD_FLOAT32_C(  -780.55), EASYSIMD_FLOAT32_C(   908.89), EASYSIMD_FLOAT32_C(   445.38), EASYSIMD_FLOAT32_C(   977.40) },
      UINT16_C(36947),
      { EASYSIMD_FLOAT32_C(   734.80), EASYSIMD_FLOAT32_C(    78.60), EASYSIMD_FLOAT32_C(   999.36), EASYSIMD_FLOAT32_C(   229.17),
        EASYSIMD_FLOAT32_C(   842.67), EASYSIMD_FLOAT32_C(   181.85), EASYSIMD_FLOAT32_C(   829.16), EASYSIMD_FLOAT32_C(  -611.44),
        EASYSIMD_FLOAT32_C(   164.65), EASYSIMD_FLOAT32_C(   634.95), EASYSIMD_FLOAT32_C(  -181.22), EASYSIMD_FLOAT32_C(   407.66),
        EASYSIMD_FLOAT32_C(   498.12), EASYSIMD_FLOAT32_C(  -896.12), EASYSIMD_FLOAT32_C(   512.20), EASYSIMD_FLOAT32_C(   849.67) },
      { EASYSIMD_FLOAT32_C(   734.80), EASYSIMD_FLOAT32_C(    78.60), EASYSIMD_FLOAT32_C(  -648.45), EASYSIMD_FLOAT32_C(   227.09),
        EASYSIMD_FLOAT32_C(   842.67), EASYSIMD_FLOAT32_C(   340.25), EASYSIMD_FLOAT32_C(   829.16), EASYSIMD_FLOAT32_C(  -250.56),
        EASYSIMD_FLOAT32_C(   674.03), EASYSIMD_FLOAT32_C(   739.23), EASYSIMD_FLOAT32_C(   428.59), EASYSIMD_FLOAT32_C(  -909.84),
        EASYSIMD_FLOAT32_C(   498.12), EASYSIMD_FLOAT32_C(   908.89), EASYSIMD_FLOAT32_C(   445.38), EASYSIMD_FLOAT32_C(   849.67) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512 src = easysimd_mm512_loadu_ps(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_loadu_ps(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_loadu_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 src = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_mask_loadu_ps(src, k, &a);

    easysimd_test_x86_write_f32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_loadu_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    const uint16_t k;
    const easysimd_float32 a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { UINT16_C(48349),
      { EASYSIMD_FLOAT32_C(   237.27), EASYSIMD_FLOAT32_C(   675.98), EASYSIMD_FLOAT32_C(  -533.02), EASYSIMD_FLOAT32_C(    -3.77),
        EASYSIMD_FLOAT32_C(  -299.42), EASYSIMD_FLOAT32_C(   801.95), EASYSIMD_FLOAT32_C(   -99.68), EASYSIMD_FLOAT32_C(  -885.62),
        EASYSIMD_FLOAT32_C(  -874.45), EASYSIMD_FLOAT32_C(  -747.05), EASYSIMD_FLOAT32_C(   718.54), EASYSIMD_FLOAT32_C(   277.50),
        EASYSIMD_FLOAT32_C(  -285.52), EASYSIMD_FLOAT32_C(  -639.12), EASYSIMD_FLOAT32_C(    36.81), EASYSIMD_FLOAT32_C(   592.75) },
      { EASYSIMD_FLOAT32_C(   237.27), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -533.02), EASYSIMD_FLOAT32_C(    -3.77),
        EASYSIMD_FLOAT32_C(  -299.42), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -99.68), EASYSIMD_FLOAT32_C(  -885.62),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   718.54), EASYSIMD_FLOAT32_C(   277.50),
        EASYSIMD_FLOAT32_C(  -285.52), EASYSIMD_FLOAT32_C(  -639.12), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   592.75) } },
    { UINT16_C(27547),
      { EASYSIMD_FLOAT32_C(   126.42), EASYSIMD_FLOAT32_C(  -708.09), EASYSIMD_FLOAT32_C(    54.33), EASYSIMD_FLOAT32_C(  -930.45),
        EASYSIMD_FLOAT32_C(   861.77), EASYSIMD_FLOAT32_C(   667.92), EASYSIMD_FLOAT32_C(   822.30), EASYSIMD_FLOAT32_C(   792.74),
        EASYSIMD_FLOAT32_C(  -279.98), EASYSIMD_FLOAT32_C(   267.36), EASYSIMD_FLOAT32_C(   930.24), EASYSIMD_FLOAT32_C(   551.11),
        EASYSIMD_FLOAT32_C(   877.93), EASYSIMD_FLOAT32_C(   167.50), EASYSIMD_FLOAT32_C(   227.09), EASYSIMD_FLOAT32_C(  -655.09) },
      { EASYSIMD_FLOAT32_C(   126.42), EASYSIMD_FLOAT32_C(  -708.09), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -930.45),
        EASYSIMD_FLOAT32_C(   861.77), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   792.74),
        EASYSIMD_FLOAT32_C(  -279.98), EASYSIMD_FLOAT32_C(   267.36), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   551.11),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   167.50), EASYSIMD_FLOAT32_C(   227.09), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(29820),
      { EASYSIMD_FLOAT32_C(  -853.14), EASYSIMD_FLOAT32_C(    64.05), EASYSIMD_FLOAT32_C(  -957.95), EASYSIMD_FLOAT32_C(  -727.59),
        EASYSIMD_FLOAT32_C(   317.01), EASYSIMD_FLOAT32_C(   760.59), EASYSIMD_FLOAT32_C(   549.91), EASYSIMD_FLOAT32_C(  -968.52),
        EASYSIMD_FLOAT32_C(  -878.53), EASYSIMD_FLOAT32_C(  -413.27), EASYSIMD_FLOAT32_C(   624.23), EASYSIMD_FLOAT32_C(  -275.95),
        EASYSIMD_FLOAT32_C(   386.63), EASYSIMD_FLOAT32_C(  -249.35), EASYSIMD_FLOAT32_C(    15.96), EASYSIMD_FLOAT32_C(  -559.03) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -957.95), EASYSIMD_FLOAT32_C(  -727.59),
        EASYSIMD_FLOAT32_C(   317.01), EASYSIMD_FLOAT32_C(   760.59), EASYSIMD_FLOAT32_C(   549.91), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   624.23), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   386.63), EASYSIMD_FLOAT32_C(  -249.35), EASYSIMD_FLOAT32_C(    15.96), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C( 3091),
      { EASYSIMD_FLOAT32_C(  -891.11), EASYSIMD_FLOAT32_C(  -357.50), EASYSIMD_FLOAT32_C(  -329.52), EASYSIMD_FLOAT32_C(  -171.09),
        EASYSIMD_FLOAT32_C(   909.86), EASYSIMD_FLOAT32_C(  -399.28), EASYSIMD_FLOAT32_C(  -619.99), EASYSIMD_FLOAT32_C(   787.79),
        EASYSIMD_FLOAT32_C(   768.22), EASYSIMD_FLOAT32_C(   607.10), EASYSIMD_FLOAT32_C(  -867.29), EASYSIMD_FLOAT32_C(   931.96),
        EASYSIMD_FLOAT32_C(   534.77), EASYSIMD_FLOAT32_C(  -720.43), EASYSIMD_FLOAT32_C(    -3.99), EASYSIMD_FLOAT32_C(   576.82) },
      { EASYSIMD_FLOAT32_C(  -891.11), EASYSIMD_FLOAT32_C(  -357.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   909.86), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -867.29), EASYSIMD_FLOAT32_C(   931.96),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(41564),
      { EASYSIMD_FLOAT32_C(   337.41), EASYSIMD_FLOAT32_C(  -898.10), EASYSIMD_FLOAT32_C(  -655.50), EASYSIMD_FLOAT32_C(   458.88),
        EASYSIMD_FLOAT32_C(  -311.38), EASYSIMD_FLOAT32_C(   968.73), EASYSIMD_FLOAT32_C(  -817.07), EASYSIMD_FLOAT32_C(  -924.74),
        EASYSIMD_FLOAT32_C(  -280.62), EASYSIMD_FLOAT32_C(   198.90), EASYSIMD_FLOAT32_C(  -483.78), EASYSIMD_FLOAT32_C(   539.58),
        EASYSIMD_FLOAT32_C(  -923.36), EASYSIMD_FLOAT32_C(  -374.89), EASYSIMD_FLOAT32_C(  -817.92), EASYSIMD_FLOAT32_C(  -252.89) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -655.50), EASYSIMD_FLOAT32_C(   458.88),
        EASYSIMD_FLOAT32_C(  -311.38), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -817.07), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   198.90), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -374.89), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -252.89) } },
    { UINT16_C(26412),
      { EASYSIMD_FLOAT32_C(   347.83), EASYSIMD_FLOAT32_C(   834.03), EASYSIMD_FLOAT32_C(   879.73), EASYSIMD_FLOAT32_C(   116.06),
        EASYSIMD_FLOAT32_C(   441.13), EASYSIMD_FLOAT32_C(  -987.56), EASYSIMD_FLOAT32_C(    48.02), EASYSIMD_FLOAT32_C(   -24.10),
        EASYSIMD_FLOAT32_C(  -707.99), EASYSIMD_FLOAT32_C(  -955.97), EASYSIMD_FLOAT32_C(  -447.28), EASYSIMD_FLOAT32_C(  -156.01),
        EASYSIMD_FLOAT32_C(  -642.95), EASYSIMD_FLOAT32_C(   890.12), EASYSIMD_FLOAT32_C(   -54.11), EASYSIMD_FLOAT32_C(  -298.44) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   879.73), EASYSIMD_FLOAT32_C(   116.06),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -987.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -707.99), EASYSIMD_FLOAT32_C(  -955.97), EASYSIMD_FLOAT32_C(  -447.28), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   890.12), EASYSIMD_FLOAT32_C(   -54.11), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(12786),
      { EASYSIMD_FLOAT32_C(  -329.71), EASYSIMD_FLOAT32_C(   531.94), EASYSIMD_FLOAT32_C(   709.77), EASYSIMD_FLOAT32_C(   389.68),
        EASYSIMD_FLOAT32_C(  -269.17), EASYSIMD_FLOAT32_C(  -774.01), EASYSIMD_FLOAT32_C(   -70.74), EASYSIMD_FLOAT32_C(  -192.53),
        EASYSIMD_FLOAT32_C(  -148.90), EASYSIMD_FLOAT32_C(   111.34), EASYSIMD_FLOAT32_C(   554.58), EASYSIMD_FLOAT32_C(  -694.88),
        EASYSIMD_FLOAT32_C(   203.28), EASYSIMD_FLOAT32_C(   -97.58), EASYSIMD_FLOAT32_C(  -860.85), EASYSIMD_FLOAT32_C(    83.01) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   531.94), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -269.17), EASYSIMD_FLOAT32_C(  -774.01), EASYSIMD_FLOAT32_C(   -70.74), EASYSIMD_FLOAT32_C(  -192.53),
        EASYSIMD_FLOAT32_C(  -148.90), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   203.28), EASYSIMD_FLOAT32_C(   -97.58), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C( 3909),
      { EASYSIMD_FLOAT32_C(    95.45), EASYSIMD_FLOAT32_C(    66.49), EASYSIMD_FLOAT32_C(  -443.82), EASYSIMD_FLOAT32_C(   387.47),
        EASYSIMD_FLOAT32_C(   110.53), EASYSIMD_FLOAT32_C(   108.90), EASYSIMD_FLOAT32_C(  -768.54), EASYSIMD_FLOAT32_C(   467.58),
        EASYSIMD_FLOAT32_C(    -0.98), EASYSIMD_FLOAT32_C(   177.35), EASYSIMD_FLOAT32_C(  -830.86), EASYSIMD_FLOAT32_C(  -651.98),
        EASYSIMD_FLOAT32_C(  -188.14), EASYSIMD_FLOAT32_C(  -160.57), EASYSIMD_FLOAT32_C(   879.96), EASYSIMD_FLOAT32_C(  -478.37) },
      { EASYSIMD_FLOAT32_C(    95.45), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -443.82), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -768.54), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    -0.98), EASYSIMD_FLOAT32_C(   177.35), EASYSIMD_FLOAT32_C(  -830.86), EASYSIMD_FLOAT32_C(  -651.98),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_loadu_ps(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_loadu_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_maskz_loadu_ps(k, &a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_loadu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_loadu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_loadu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_loadu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_loadu_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_loadu_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_loadu_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_loadu_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_loadu_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_loadu_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_loadu_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_loadu_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_loadu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_loadu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_loadu_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_loadu_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_loadu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_loadu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_loadu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_loadu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_loadu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_loadu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_loadu_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_loadu_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_loadu_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_loadu_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_loadu_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_loadu_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_loadu_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_loadu_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_loadu_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_loadu_pd)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_loadu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_loadu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_loadu_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_loadu_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_loadu_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_loadu_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_loadu_si512)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_loadu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_loadu_epi8)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_loadu_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_loadu_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_loadu_epi32)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_loadu_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_loadu_ps)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
