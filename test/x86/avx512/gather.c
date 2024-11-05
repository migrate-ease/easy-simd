#define EASYSIMD_TEST_X86_AVX512_INSN gather

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/gather.h>


static easysimd_float32 f32gather_buffer[4096];
static int32_t i32gather_buffer[4096];

static int
test_easysimd_mm512_i64gather_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t vindex[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { {  INT64_C(                  42),  INT64_C(                 100),  INT64_C(                 221),  INT64_C(                 205),
         INT64_C(                 119),  INT64_C(                  89),  INT64_C(                 152),  INT64_C(                   6) },
      { EASYSIMD_FLOAT32_C(    42.00), EASYSIMD_FLOAT32_C(   100.00), EASYSIMD_FLOAT32_C(   221.00), EASYSIMD_FLOAT32_C(   205.00),
        EASYSIMD_FLOAT32_C(   119.00), EASYSIMD_FLOAT32_C(    89.00), EASYSIMD_FLOAT32_C(   152.00), EASYSIMD_FLOAT32_C(     6.00) } },
    { {  INT64_C(                   7),  INT64_C(                  66),  INT64_C(                 122),  INT64_C(                  88),
         INT64_C(                 205),  INT64_C(                 225),  INT64_C(                 153),  INT64_C(                  12) },
      { EASYSIMD_FLOAT32_C(     7.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(   122.00), EASYSIMD_FLOAT32_C(    88.00),
        EASYSIMD_FLOAT32_C(   205.00), EASYSIMD_FLOAT32_C(   225.00), EASYSIMD_FLOAT32_C(   153.00), EASYSIMD_FLOAT32_C(    12.00) } },
    { {  INT64_C(                 224),  INT64_C(                 125),  INT64_C(                  98),  INT64_C(                  77),
         INT64_C(                 185),  INT64_C(                  66),  INT64_C(                  12),  INT64_C(                   7) },
      { EASYSIMD_FLOAT32_C(   224.00), EASYSIMD_FLOAT32_C(   125.00), EASYSIMD_FLOAT32_C(    98.00), EASYSIMD_FLOAT32_C(    77.00),
        EASYSIMD_FLOAT32_C(   185.00), EASYSIMD_FLOAT32_C(    66.00), EASYSIMD_FLOAT32_C(    12.00), EASYSIMD_FLOAT32_C(     7.00) } },
    { {  INT64_C(                  71),  INT64_C(                 198),  INT64_C(                 199),  INT64_C(                   2),
         INT64_C(                   6),  INT64_C(                  16),  INT64_C(                 201),  INT64_C(                 220) },
      { EASYSIMD_FLOAT32_C(    71.00), EASYSIMD_FLOAT32_C(   198.00), EASYSIMD_FLOAT32_C(   199.00), EASYSIMD_FLOAT32_C(     2.00),
        EASYSIMD_FLOAT32_C(     6.00), EASYSIMD_FLOAT32_C(    16.00), EASYSIMD_FLOAT32_C(   201.00), EASYSIMD_FLOAT32_C(   220.00) } }
  };
  for (size_t i = 0 ; i < (sizeof(f32gather_buffer) / sizeof(f32gather_buffer[0])) ; i++) { f32gather_buffer[i] = HEDLEY_STATIC_CAST(easysimd_float32, i); }

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i vindex = easysimd_mm512_loadu_epi64(test_vec[i].vindex);
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_i64gather_ps(vindex, f32gather_buffer, 4);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_i64gather_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (size_t i = 0 ; i < (sizeof(f32gather_buffer) / sizeof(f32gather_buffer[0])) ; i++) { f32gather_buffer[i] = HEDLEY_STATIC_CAST(easysimd_float32, i); }

  easysimd__m512i vindex;
  easysimd__m256 r;

  int64_t a[8] = {42, 100, 221, 205, 119, 89, 152, 6};
  vindex = easysimd_mm512_loadu_epi64(a);
  r = easysimd_mm512_i64gather_ps(vindex, f32gather_buffer, 4);

  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  int64_t b[8] = {7, 66, 122, 88, 205, 225, 153, 12};
  vindex = easysimd_mm512_loadu_epi64(b);
  r = easysimd_mm512_i64gather_ps(vindex, f32gather_buffer, 4);

  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  int64_t c[8] = {224, 125, 98, 77, 185, 66, 12, 7};
  vindex = easysimd_mm512_loadu_epi64(c);
  r = easysimd_mm512_i64gather_ps(vindex, f32gather_buffer, 4);

  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  int64_t d[8] = {71, 198, 199, 2, 6, 16, 201, 220};
  vindex = easysimd_mm512_loadu_epi64(d);
  r = easysimd_mm512_i64gather_ps(vindex, f32gather_buffer, 4);

  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm512_mask_i64gather_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    const int64_t vindex[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -200.36), EASYSIMD_FLOAT32_C(  -872.15), EASYSIMD_FLOAT32_C(   482.46), EASYSIMD_FLOAT32_C(  -226.01),
        EASYSIMD_FLOAT32_C(   343.51), EASYSIMD_FLOAT32_C(  -119.29), EASYSIMD_FLOAT32_C(  -336.70), EASYSIMD_FLOAT32_C(   995.84) },
      UINT8_C(130),
      {  INT64_C(                  42),  INT64_C(                 100),  INT64_C(                 221),  INT64_C(                 205),
         INT64_C(                 119),  INT64_C(                  89),  INT64_C(                 152),  INT64_C(                   6) },
      { EASYSIMD_FLOAT32_C(  -200.36), EASYSIMD_FLOAT32_C(   100.00), EASYSIMD_FLOAT32_C(   482.46), EASYSIMD_FLOAT32_C(  -226.01),
        EASYSIMD_FLOAT32_C(   343.51), EASYSIMD_FLOAT32_C(  -119.29), EASYSIMD_FLOAT32_C(  -336.70), EASYSIMD_FLOAT32_C(     6.00) } },
    { { EASYSIMD_FLOAT32_C(   225.98), EASYSIMD_FLOAT32_C(   355.66), EASYSIMD_FLOAT32_C(   867.81), EASYSIMD_FLOAT32_C(   582.77),
        EASYSIMD_FLOAT32_C(  -139.41), EASYSIMD_FLOAT32_C(   766.79), EASYSIMD_FLOAT32_C(   670.98), EASYSIMD_FLOAT32_C(   656.23) },
      UINT8_C( 93),
      {  INT64_C(                   7),  INT64_C(                  66),  INT64_C(                 122),  INT64_C(                  88),
         INT64_C(                 205),  INT64_C(                 225),  INT64_C(                 153),  INT64_C(                  12) },
      { EASYSIMD_FLOAT32_C(     7.00), EASYSIMD_FLOAT32_C(   355.66), EASYSIMD_FLOAT32_C(   122.00), EASYSIMD_FLOAT32_C(    88.00),
        EASYSIMD_FLOAT32_C(   205.00), EASYSIMD_FLOAT32_C(   766.79), EASYSIMD_FLOAT32_C(   153.00), EASYSIMD_FLOAT32_C(   656.23) } },
    { { EASYSIMD_FLOAT32_C(  -385.42), EASYSIMD_FLOAT32_C(  -633.40), EASYSIMD_FLOAT32_C(   820.84), EASYSIMD_FLOAT32_C(    62.87),
        EASYSIMD_FLOAT32_C(   175.53), EASYSIMD_FLOAT32_C(    29.55), EASYSIMD_FLOAT32_C(  -237.55), EASYSIMD_FLOAT32_C(  -411.07) },
      UINT8_C(130),
      {  INT64_C(                 224),  INT64_C(                 125),  INT64_C(                  98),  INT64_C(                  77),
         INT64_C(                 185),  INT64_C(                  66),  INT64_C(                  12),  INT64_C(                   7) },
      { EASYSIMD_FLOAT32_C(  -385.42), EASYSIMD_FLOAT32_C(   125.00), EASYSIMD_FLOAT32_C(   820.84), EASYSIMD_FLOAT32_C(    62.87),
        EASYSIMD_FLOAT32_C(   175.53), EASYSIMD_FLOAT32_C(    29.55), EASYSIMD_FLOAT32_C(  -237.55), EASYSIMD_FLOAT32_C(     7.00) } },
    { { EASYSIMD_FLOAT32_C(  -388.73), EASYSIMD_FLOAT32_C(   578.40), EASYSIMD_FLOAT32_C(  -526.96), EASYSIMD_FLOAT32_C(  -577.18),
        EASYSIMD_FLOAT32_C(  -621.95), EASYSIMD_FLOAT32_C(  -399.12), EASYSIMD_FLOAT32_C(   905.29), EASYSIMD_FLOAT32_C(   152.03) },
      UINT8_C(195),
      {  INT64_C(                  71),  INT64_C(                 198),  INT64_C(                 199),  INT64_C(                   2),
         INT64_C(                   6),  INT64_C(                  16),  INT64_C(                 201),  INT64_C(                 220) },
      { EASYSIMD_FLOAT32_C(    71.00), EASYSIMD_FLOAT32_C(   198.00), EASYSIMD_FLOAT32_C(  -526.96), EASYSIMD_FLOAT32_C(  -577.18),
        EASYSIMD_FLOAT32_C(  -621.95), EASYSIMD_FLOAT32_C(  -399.12), EASYSIMD_FLOAT32_C(   201.00), EASYSIMD_FLOAT32_C(   220.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(f32gather_buffer) / sizeof(f32gather_buffer[0])) ; i++) { f32gather_buffer[i] = HEDLEY_STATIC_CAST(easysimd_float32, i); }

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__m512i vindex = easysimd_mm512_loadu_epi64(test_vec[i].vindex);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_i64gather_ps (src, k, vindex, f32gather_buffer, 4);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_i64gather_ps ");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (size_t i = 0 ; i < (sizeof(f32gather_buffer) / sizeof(f32gather_buffer[0])) ; i++) { f32gather_buffer[i] = HEDLEY_STATIC_CAST(easysimd_float32, i); }

  easysimd__m256 src;
  easysimd__mmask8 k;
  easysimd__m512i vindex;
  easysimd__m256 r;

  src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  k = easysimd_test_x86_random_mmask8();
  int64_t a[8] = {42, 100, 221, 205, 119, 89, 152, 6};
  vindex = easysimd_mm512_loadu_epi64(a);
  r = easysimd_mm512_mask_i64gather_ps(src, k, vindex, f32gather_buffer, 4);

  easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  k = easysimd_test_x86_random_mmask8();
  int64_t b[8] = {7, 66, 122, 88, 205, 225, 153, 12};
  vindex = easysimd_mm512_loadu_epi64(b);
  r = easysimd_mm512_mask_i64gather_ps(src, k, vindex, f32gather_buffer, 4);

  easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  k = easysimd_test_x86_random_mmask8();
  int64_t c[8] = {224, 125, 98, 77, 185, 66, 12, 7};
  vindex = easysimd_mm512_loadu_epi64(c);
  r = easysimd_mm512_mask_i64gather_ps(src, k, vindex, f32gather_buffer, 4);

  easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  k = easysimd_test_x86_random_mmask8();
  int64_t d[8] = {71, 198, 199, 2, 6, 16, 201, 220};
  vindex = easysimd_mm512_loadu_epi64(d);
  r = easysimd_mm512_mask_i64gather_ps(src, k, vindex, f32gather_buffer, 4);

  easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm512_i64gather_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t vindex[8];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT64_C(                  42),  INT64_C(                 100),  INT64_C(                 221),  INT64_C(                 205),
         INT64_C(                 119),  INT64_C(                  89),  INT64_C(                 152),  INT64_C(                   6) },
      {  INT32_C(          42),  INT32_C(         100),  INT32_C(         221),  INT32_C(         205),  INT32_C(         119),  INT32_C(          89),  INT32_C(         152),  INT32_C(           6) } },
    { {  INT64_C(                   7),  INT64_C(                  66),  INT64_C(                 122),  INT64_C(                  88),
         INT64_C(                 205),  INT64_C(                 225),  INT64_C(                 153),  INT64_C(                  12) },
      {  INT32_C(           7),  INT32_C(          66),  INT32_C(         122),  INT32_C(          88),  INT32_C(         205),  INT32_C(         225),  INT32_C(         153),  INT32_C(          12) } },
    { {  INT64_C(                 224),  INT64_C(                 125),  INT64_C(                  98),  INT64_C(                  77),
         INT64_C(                 185),  INT64_C(                  66),  INT64_C(                  12),  INT64_C(                   7) },
      {  INT32_C(         224),  INT32_C(         125),  INT32_C(          98),  INT32_C(          77),  INT32_C(         185),  INT32_C(          66),  INT32_C(          12),  INT32_C(           7) } },
    { {  INT64_C(                  71),  INT64_C(                 198),  INT64_C(                 199),  INT64_C(                   2),
         INT64_C(                   6),  INT64_C(                  16),  INT64_C(                 201),  INT64_C(                 220) },
      {  INT32_C(          71),  INT32_C(         198),  INT32_C(         199),  INT32_C(           2),  INT32_C(           6),  INT32_C(          16),  INT32_C(         201),  INT32_C(         220) } }
  };
  for (size_t i = 0 ; i < (sizeof(i32gather_buffer) / sizeof(i32gather_buffer[0])) ; i++) { i32gather_buffer[i] = HEDLEY_STATIC_CAST(int32_t, i); }

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i vindex = easysimd_mm512_loadu_epi64(test_vec[i].vindex);
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_i64gather_epi32(vindex, i32gather_buffer, 4);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_i64gather_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (size_t i = 0 ; i < (sizeof(i32gather_buffer) / sizeof(i32gather_buffer[0])) ; i++) { i32gather_buffer[i] = HEDLEY_STATIC_CAST(int32_t, i); }

  easysimd__m512i vindex;
  easysimd__m256i r;

  int64_t a[8] = {42, 100, 221, 205, 119, 89, 152, 6};
  vindex = easysimd_mm512_loadu_epi64(a);
  r = easysimd_mm512_i64gather_epi32(vindex, i32gather_buffer, 4);

  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  int64_t b[8] = {7, 66, 122, 88, 205, 225, 153, 12};
  vindex = easysimd_mm512_loadu_epi64(b);
  r = easysimd_mm512_i64gather_epi32(vindex, i32gather_buffer, 4);

  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  int64_t c[8] = {224, 125, 98, 77, 185, 66, 12, 7};
  vindex = easysimd_mm512_loadu_epi64(c);
  r = easysimd_mm512_i64gather_epi32(vindex, i32gather_buffer, 4);

  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  int64_t d[8] = {71, 198, 199, 2, 6, 16, 201, 220};
  vindex = easysimd_mm512_loadu_epi64(d);
  r = easysimd_mm512_i64gather_epi32(vindex, i32gather_buffer, 4);

  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm512_mask_i64gather_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const uint8_t k;
    const int64_t vindex[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1525534312),  INT32_C(  1080289262),  INT32_C(  1185748847),  INT32_C(   880547759),  INT32_C(  1623508818), -INT32_C(  1256193100), -INT32_C(  1976941865),  INT32_C(   535337607) },
      UINT8_C(204),
      {  INT64_C(                  42),  INT64_C(                 100),  INT64_C(                 221),  INT64_C(                 205),
         INT64_C(                 119),  INT64_C(                  89),  INT64_C(                 152),  INT64_C(                   6) },
      { -INT32_C(  1525534312),  INT32_C(  1080289262),  INT32_C(         221),  INT32_C(         205),  INT32_C(  1623508818), -INT32_C(  1256193100),  INT32_C(         152),  INT32_C(           6) } },
    { { -INT32_C(   507853318),  INT32_C(  1062271784), -INT32_C(  1074882648), -INT32_C(   619634157), -INT32_C(   376475162), -INT32_C(   725531503),  INT32_C(   156978031),  INT32_C(   768965427) },
      UINT8_C( 64),
      {  INT64_C(                   7),  INT64_C(                  66),  INT64_C(                 122),  INT64_C(                  88),
         INT64_C(                 205),  INT64_C(                 225),  INT64_C(                 153),  INT64_C(                  12) },
      { -INT32_C(   507853318),  INT32_C(  1062271784), -INT32_C(  1074882648), -INT32_C(   619634157), -INT32_C(   376475162), -INT32_C(   725531503),  INT32_C(         153),  INT32_C(   768965427) } },
    { { -INT32_C(  1956114544), -INT32_C(   164386977), -INT32_C(  1224019051),  INT32_C(  1990124804), -INT32_C(  1190688908), -INT32_C(  1826038968), -INT32_C(  1295634121), -INT32_C(  1745685497) },
      UINT8_C(  3),
      {  INT64_C(                 224),  INT64_C(                 125),  INT64_C(                  98),  INT64_C(                  77),
         INT64_C(                 185),  INT64_C(                  66),  INT64_C(                  12),  INT64_C(                   7) },
      {  INT32_C(         224),  INT32_C(         125), -INT32_C(  1224019051),  INT32_C(  1990124804), -INT32_C(  1190688908), -INT32_C(  1826038968), -INT32_C(  1295634121), -INT32_C(  1745685497) } },
    { {  INT32_C(    23208538),  INT32_C(  1217812821),  INT32_C(  1213025635),  INT32_C(  1941750763), -INT32_C(  1497664054), -INT32_C(   807579747),  INT32_C(   165056277), -INT32_C(   619942527) },
      UINT8_C(143),
      {  INT64_C(                  71),  INT64_C(                 198),  INT64_C(                 199),  INT64_C(                   2),
         INT64_C(                   6),  INT64_C(                  16),  INT64_C(                 201),  INT64_C(                 220) },
      {  INT32_C(          71),  INT32_C(         198),  INT32_C(         199),  INT32_C(           2), -INT32_C(  1497664054), -INT32_C(   807579747),  INT32_C(   165056277),  INT32_C(         220) } }
  };

  for (size_t i = 0 ; i < (sizeof(i32gather_buffer) / sizeof(i32gather_buffer[0])) ; i++) { i32gather_buffer[i] = HEDLEY_STATIC_CAST(int32_t, i); }

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m512i vindex = easysimd_mm512_loadu_epi64(test_vec[i].vindex);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_i64gather_epi32(src, k, vindex, i32gather_buffer, 4);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_i64gather_epi32 ");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (size_t i = 0 ; i < (sizeof(i32gather_buffer) / sizeof(i32gather_buffer[0])) ; i++) { i32gather_buffer[i] = HEDLEY_STATIC_CAST(int32_t, i); }

  easysimd__m256i src;
  easysimd__mmask8 k;
  easysimd__m512i vindex;
  easysimd__m256i r;

  src = easysimd_test_x86_random_i32x8();
  k = easysimd_test_x86_random_mmask8();
  int64_t a[8] = {42, 100, 221, 205, 119, 89, 152, 6};
  vindex = easysimd_mm512_loadu_epi64(a);
  r = easysimd_mm512_mask_i64gather_epi32(src, k, vindex, i32gather_buffer, 4);

  easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  src = easysimd_test_x86_random_i32x8();
  k = easysimd_test_x86_random_mmask8();
  int64_t b[8] = {7, 66, 122, 88, 205, 225, 153, 12};
  vindex = easysimd_mm512_loadu_epi64(b);
  r = easysimd_mm512_mask_i64gather_epi32(src, k, vindex, i32gather_buffer, 4);

  easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  src = easysimd_test_x86_random_i32x8();
  k = easysimd_test_x86_random_mmask8();
  int64_t c[8] = {224, 125, 98, 77, 185, 66, 12, 7};
  vindex = easysimd_mm512_loadu_epi64(c);
  r = easysimd_mm512_mask_i64gather_epi32(src, k, vindex, i32gather_buffer, 4);

  easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  src = easysimd_test_x86_random_i32x8();
  k = easysimd_test_x86_random_mmask8();
  int64_t d[8] = {71, 198, 199, 2, 6, 16, 201, 220};
  vindex = easysimd_mm512_loadu_epi64(d);
  r = easysimd_mm512_mask_i64gather_epi32(src, k, vindex, i32gather_buffer, 4);

  easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm512_i32gather_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t vindex[16];
    int32_t base_addr[16];
    int32_t r[16];
  } test_vec[8] = {
    { {  INT32_C(           5),  INT32_C(           7),  INT32_C(           3),  INT32_C(          11),  INT32_C(           5),  INT32_C(           1),  INT32_C(           8),  INT32_C(          11),
         INT32_C(          11),  INT32_C(           7),  INT32_C(          12),  INT32_C(           3),  INT32_C(          10),  INT32_C(           4),  INT32_C(           2),  INT32_C(          15) },
      { -INT32_C(  2039632097), -INT32_C(    58618768),  INT32_C(   411252437), -INT32_C(  1074661259), -INT32_C(  2002073996),  INT32_C(  1347700078),  INT32_C(  1869521904),  INT32_C(   369708023),
         INT32_C(  1805481723), -INT32_C(   664330749), -INT32_C(   890181036),  INT32_C(  1451877090),  INT32_C(   383661224),  INT32_C(  1768371065), -INT32_C(   824650282),  INT32_C(   467984928) },
      { -INT32_C(   704872052), -INT32_C(  2093558276), -INT32_C(  2121502586), -INT32_C(   235375336), -INT32_C(   704872052),  INT32_C(  1887858095),  INT32_C(   411252437), -INT32_C(   235375336),
        -INT32_C(   235375336), -INT32_C(  2093558276), -INT32_C(  1074661259), -INT32_C(  2121502586), -INT32_C(   126543741), -INT32_C(    58618768), -INT32_C(  1938782611), -INT32_C(  1429834561) } },
    { {  INT32_C(           8),  INT32_C(           0),  INT32_C(           7),  INT32_C(           6),  INT32_C(          12),  INT32_C(           2),  INT32_C(           1),  INT32_C(           9),
         INT32_C(          12),  INT32_C(           1),  INT32_C(           5),  INT32_C(           7),  INT32_C(           2),  INT32_C(          10),  INT32_C(          15),  INT32_C(          14) },
      { -INT32_C(  1631115027),  INT32_C(  1311213609), -INT32_C(   814126841), -INT32_C(    32353348), -INT32_C(  1134408719), -INT32_C(  1282949149), -INT32_C(  1331923742), -INT32_C(  1934952546),
         INT32_C(  1227518239),  INT32_C(    26694137),  INT32_C(  2010124731),  INT32_C(  1433854564),  INT32_C(   286382126),  INT32_C(  2042927511), -INT32_C(  1406574579),  INT32_C(  2000213079) },
      { -INT32_C(   814126841), -INT32_C(  1631115027),  INT32_C(  2036926286),  INT32_C(  1762086439), -INT32_C(    32353348), -INT32_C(  2010538297),  INT32_C(   698271528), -INT32_C(  1127253655),
        -INT32_C(    32353348),  INT32_C(   698271528),  INT32_C(   122562440),  INT32_C(  2036926286), -INT32_C(  2010538297),  INT32_C(  1404882809),  INT32_C(  1649144318),  INT32_C(  1274150418) } },
    { {  INT32_C(           5),  INT32_C(           4),  INT32_C(           8),  INT32_C(           2),  INT32_C(           5),  INT32_C(          12),  INT32_C(           7),  INT32_C(           3),
         INT32_C(           0),  INT32_C(          13),  INT32_C(           8),  INT32_C(           6),  INT32_C(           7),  INT32_C(           3),  INT32_C(          13),  INT32_C(           2) },
      { -INT32_C(  1582837148), -INT32_C(   230233382), -INT32_C(  1677936620),  INT32_C(   341190476), -INT32_C(   660021628), -INT32_C(   360573268), -INT32_C(  1133782262), -INT32_C(  1097960102),
         INT32_C(   559887687),  INT32_C(   873702688), -INT32_C(  1429270690), -INT32_C(  1145166281), -INT32_C(  1315740156), -INT32_C(  2036656516),  INT32_C(  1262618353), -INT32_C(  1039543941) },
      {  INT32_C(   351422186), -INT32_C(   230233382), -INT32_C(  1677936620), -INT32_C(   354770521),  INT32_C(   351422186),  INT32_C(   341190476), -INT32_C(    55044878),  INT32_C(  1189796513),
        -INT32_C(  1582837148), -INT32_C(  2079042009), -INT32_C(  1677936620), -INT32_C(  1206586810), -INT32_C(    55044878),  INT32_C(  1189796513), -INT32_C(  2079042009), -INT32_C(   354770521) } },
    { {  INT32_C(           6),  INT32_C(          14),  INT32_C(           5),  INT32_C(           0),  INT32_C(          11),  INT32_C(           1),  INT32_C(           7),  INT32_C(           2),
         INT32_C(           0),  INT32_C(          13),  INT32_C(           4),  INT32_C(           6),  INT32_C(           2),  INT32_C(           1),  INT32_C(           9),  INT32_C(          13) },
      { -INT32_C(  2111008059),  INT32_C(  1900345196), -INT32_C(  1899609128), -INT32_C(  1115721670),  INT32_C(  1910581536),  INT32_C(  2053740561), -INT32_C(  1422556371),  INT32_C(  1182353537),
         INT32_C(  1204331995),  INT32_C(  1958218908), -INT32_C(  1979548080),  INT32_C(   239567598), -INT32_C(  1132517205), -INT32_C(   298391360), -INT32_C(   325489813), -INT32_C(  1909321037) },
      {  INT32_C(  1138258244),  INT32_C(   690011519), -INT32_C(   663665417), -INT32_C(  2111008059),  INT32_C(  2138061454),  INT32_C(  1820470418), -INT32_C(   968632207), -INT32_C(   143883732),
        -INT32_C(  2111008059),  INT32_C(   549289840),  INT32_C(  1900345196),  INT32_C(  1138258244), -INT32_C(   143883732),  INT32_C(  1820470418),  INT32_C(   982435395),  INT32_C(   549289840) } },
    { {  INT32_C(           7),  INT32_C(           7),  INT32_C(          13),  INT32_C(          13),  INT32_C(           1),  INT32_C(           2),  INT32_C(           5),  INT32_C(          11),
         INT32_C(           7),  INT32_C(          11),  INT32_C(          10),  INT32_C(          15),  INT32_C(           7),  INT32_C(          13),  INT32_C(          12),  INT32_C(          13) },
      { -INT32_C(  1969086322), -INT32_C(   376244321), -INT32_C(  1920952786),  INT32_C(  1010700196),  INT32_C(  2135228754),  INT32_C(   536947908), -INT32_C(  1960342563),  INT32_C(   992973229),
         INT32_C(  1355141809), -INT32_C(   147236663), -INT32_C(  1819952658),  INT32_C(   449823688), -INT32_C(  1986456379),  INT32_C(   514497089), -INT32_C(   760622811),  INT32_C(   319740258) },
      { -INT32_C(  2137641239), -INT32_C(  2137641239),  INT32_C(  1379679759),  INT32_C(  1379679759), -INT32_C(  1618304480), -INT32_C(   140539230),  INT32_C(   787059447),  INT32_C(  1041212557),
        -INT32_C(  2137641239),  INT32_C(  1041212557),  INT32_C(   262442368),  INT32_C(  1157714492), -INT32_C(  2137641239),  INT32_C(  1379679759),  INT32_C(  1010700196),  INT32_C(  1379679759) } },
    { {  INT32_C(          11),  INT32_C(          11),  INT32_C(           5),  INT32_C(           4),  INT32_C(           0),  INT32_C(          10),  INT32_C(          14),  INT32_C(           2),
         INT32_C(           4),  INT32_C(           4),  INT32_C(          15),  INT32_C(           2),  INT32_C(          15),  INT32_C(          13),  INT32_C(          14),  INT32_C(          13) },
      {  INT32_C(  1844494457), -INT32_C(  1380484418),  INT32_C(  1666195376),  INT32_C(   765881790),  INT32_C(  1510455949),  INT32_C(  1571189183),  INT32_C(  2070794814),  INT32_C(  1828494323),
         INT32_C(   433777755),  INT32_C(   398889318),  INT32_C(  1853495216),  INT32_C(   312221828), -INT32_C(  1720933414),  INT32_C(  1408635413), -INT32_C(   657562395),  INT32_C(  1531300607) },
      { -INT32_C(  1502757277), -INT32_C(  1502757277), -INT32_C(  1330792582), -INT32_C(  1380484418),  INT32_C(  1844494457),  INT32_C(  1841193808), -INT32_C(  1165152858),  INT32_C(  2059300336),
        -INT32_C(  1380484418), -INT32_C(  1380484418),  INT32_C(   129666349),  INT32_C(  2059300336),  INT32_C(   129666349), -INT32_C(  1926388115), -INT32_C(  1165152858), -INT32_C(  1926388115) } },
    { {  INT32_C(           6),  INT32_C(           0),  INT32_C(           2),  INT32_C(          14),  INT32_C(          15),  INT32_C(           7),  INT32_C(          12),  INT32_C(          15),
         INT32_C(           9),  INT32_C(           6),  INT32_C(           4),  INT32_C(           5),  INT32_C(          12),  INT32_C(           5),  INT32_C(           4),  INT32_C(          15) },
      { -INT32_C(   149783728),  INT32_C(  1458547858), -INT32_C(   320225770), -INT32_C(  2100288394),  INT32_C(  1459018145),  INT32_C(   497596248), -INT32_C(   319777124), -INT32_C(   859534981),
        -INT32_C(  1748773371), -INT32_C(  1712475517), -INT32_C(   410659215), -INT32_C(  1687595270), -INT32_C(  2014158801), -INT32_C(  1281058281), -INT32_C(  1398827727),  INT32_C(   595092510) },
      { -INT32_C(  1105832209), -INT32_C(   149783728), -INT32_C(  1399654638), -INT32_C(   643726640), -INT32_C(   153509502), -INT32_C(   373418410), -INT32_C(  2100288394), -INT32_C(   153509502),
         INT32_C(  1995237822), -INT32_C(  1105832209),  INT32_C(  1458547858),  INT32_C(   374796204), -INT32_C(  2100288394),  INT32_C(   374796204),  INT32_C(  1458547858), -INT32_C(   153509502) } },
    { {  INT32_C(          10),  INT32_C(          14),  INT32_C(          14),  INT32_C(           2),  INT32_C(          15),  INT32_C(           1),  INT32_C(           2),  INT32_C(          13),
         INT32_C(           2),  INT32_C(           7),  INT32_C(           9),  INT32_C(           5),  INT32_C(           9),  INT32_C(           3),  INT32_C(          14),  INT32_C(           2) },
      { -INT32_C(  1055536646), -INT32_C(    16280330), -INT32_C(   818231974),  INT32_C(  1918097304),  INT32_C(  1911357021), -INT32_C(  2102969981), -INT32_C(   966910460), -INT32_C(   189115655),
         INT32_C(   297127707), -INT32_C(  1106199452),  INT32_C(   445467522), -INT32_C(  2138251230),  INT32_C(  1659992543), -INT32_C(  1226532686), -INT32_C(  1350810954), -INT32_C(  1331480940) },
      { -INT32_C(   677851334), -INT32_C(    27430317), -INT32_C(    27430317), -INT32_C(  1795768043), -INT32_C(   318874254), -INT32_C(   155118135), -INT32_C(  1795768043),  INT32_C(  1567773655),
        -INT32_C(  1795768043),  INT32_C(   986012415), -INT32_C(  1731249467),  INT32_C(  1526663060), -INT32_C(  1731249467),  INT32_C(   127203009), -INT32_C(    27430317), -INT32_C(  1795768043) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i vindex = easysimd_mm512_loadu_si512(test_vec[i].vindex);
    easysimd__m512i base_addr = easysimd_mm512_loadu_si512(test_vec[i].base_addr);
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_i32gather_epi32(vindex, (void *)&base_addr, 1);
    } EASYSIMD_TEST_PERF_END("_mm512_i32gather_epi32");
    easysimd_assert_m512i_i32(r, ==, ret);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i vindex = easysimd_test_x86_random_i32x16();
    for(size_t i = 0; i < 16; i++) {
      *(((int32_t *)&vindex) + i) = *(((int32_t *)&vindex) + i) & 15;
    }
    easysimd__m512i base_addr = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_i32gather_epi32(vindex, (void *)&base_addr, 1);

    easysimd_test_x86_write_i32x16(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, base_addr, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_i32gather_ps(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t vindex[16];
    int32_t base_addr[16];
    int32_t r[16];
  } test_vec[8] = {
    { {  INT32_C(           2),  INT32_C(           1),  INT32_C(           2),  INT32_C(          13),  INT32_C(          12),  INT32_C(           7),  INT32_C(           5),  INT32_C(          12),
         INT32_C(           1),  INT32_C(           2),  INT32_C(          13),  INT32_C(          11),  INT32_C(           4),  INT32_C(           7),  INT32_C(           3),  INT32_C(           2) },
      {  INT32_C(  1106336809),  INT32_C(  1100386140),  INT32_C(  1116046623),  INT32_C(  1092700078), -INT32_C(  1037489930),  INT32_C(  1119778243), -INT32_C(  1057656668),  INT32_C(  1116826501),
         INT32_C(  1118501601), -INT32_C(  1038334034), -INT32_C(  1037083607), -INT32_C(  1039387853),  INT32_C(  1112806523), -INT32_C(  1045157642), -INT32_C(  1053630136), -INT32_C(  1030081741) },
      { -INT32_C(  1889779215),  INT32_C(  1547825500), -INT32_C(  1889779215), -INT32_C(   163503801),  INT32_C(  1092700078), -INT32_C(  2054873279),  INT32_C(   524392079),  INT32_C(  1092700078),
         INT32_C(  1547825500), -INT32_C(  1889779215), -INT32_C(   163503801),  INT32_C(   558345794),  INT32_C(  1100386140), -INT32_C(  2054873279), -INT32_C(  1768989631), -INT32_C(  1889779215) } },
    { {  INT32_C(           2),  INT32_C(           4),  INT32_C(           7),  INT32_C(          13),  INT32_C(           9),  INT32_C(           4),  INT32_C(           2),  INT32_C(           5),
         INT32_C(          10),  INT32_C(           7),  INT32_C(           8),  INT32_C(           0),  INT32_C(           9),  INT32_C(          12),  INT32_C(           5),  INT32_C(          13) },
      {  INT32_C(  1112497193), -INT32_C(  1029301862), -INT32_C(  1038276362),  INT32_C(  1107086541), -INT32_C(  1033062318), -INT32_C(  1033374269), -INT32_C(  1049393889), -INT32_C(  1029531238),
         INT32_C(  1098603561), -INT32_C(  1070889697), -INT32_C(  1036630098),  INT32_C(  1116421489), -INT32_C(  1042415616), -INT32_C(  1045524644),  INT32_C(  1102357463),  INT32_C(  1117302292) },
      {  INT32_C(   429539919), -INT32_C(  1029301862),  INT32_C(   489223874),  INT32_C(  1380056268), -INT32_C(   842916568), -INT32_C(  1029301862),  INT32_C(   429539919), -INT32_C(   155015655),
        -INT32_C(   858930659),  INT32_C(   489223874), -INT32_C(  1038276362),  INT32_C(  1112497193), -INT32_C(   842916568),  INT32_C(  1107086541), -INT32_C(   155015655),  INT32_C(  1380056268) } },
    { {  INT32_C(           3),  INT32_C(          11),  INT32_C(           9),  INT32_C(           8),  INT32_C(          15),  INT32_C(          14),  INT32_C(           6),  INT32_C(           6),
         INT32_C(           4),  INT32_C(           3),  INT32_C(           2),  INT32_C(           8),  INT32_C(           2),  INT32_C(          10),  INT32_C(           6),  INT32_C(           5) },
      { -INT32_C(  1030997934),  INT32_C(  1118204068), -INT32_C(  1030566707),  INT32_C(  1106950226), -INT32_C(  1028920443),  INT32_C(  1108586004), -INT32_C(  1053881795), -INT32_C(  1053734994),
        -INT32_C(  1031186678), -INT32_C(  1046394962), -INT32_C(  1027498312),  INT32_C(  1114408223), -INT32_C(  1041429955),  INT32_C(  1091242557),  INT32_C(  1108677755),  INT32_C(  1101717832) },
      { -INT32_C(  1502567230), -INT32_C(    88583486),  INT32_C(  1388483276), -INT32_C(  1030566707), -INT32_C(  1410628287), -INT32_C(   343588358), -INT32_C(   858963290), -INT32_C(   858963290),
         INT32_C(  1118204068), -INT32_C(  1502567230),  INT32_C(  1889845900), -INT32_C(  1030566707),  INT32_C(  1889845900), -INT32_C(  1202535790), -INT32_C(   858963290), -INT32_C(   851270032) } },
    { {  INT32_C(          10),  INT32_C(          13),  INT32_C(          14),  INT32_C(           7),  INT32_C(           3),  INT32_C(           0),  INT32_C(          13),  INT32_C(           2),
         INT32_C(          15),  INT32_C(           1),  INT32_C(          12),  INT32_C(           3),  INT32_C(           7),  INT32_C(           1),  INT32_C(           2),  INT32_C(          11) },
      {  INT32_C(  1098330931),  INT32_C(  1114701824),  INT32_C(  1110898115), -INT32_C(  1038625014),  INT32_C(  1103904113), -INT32_C(  1034582753), -INT32_C(  1028987290),  INT32_C(  1102425620),
        -INT32_C(  1038748221), -INT32_C(  1033581363), -INT32_C(  1030140723), -INT32_C(  1029937562),  INT32_C(  1114450166), -INT32_C(  1037395558),  INT32_C(  1089889894),  INT32_C(  1119174001) },
      { -INT32_C(   687193546),  INT32_C(  1908545495),  INT32_C(  1030865431),  INT32_C(   922075970),  INT32_C(  1895825473),  INT32_C(  1098330931),  INT32_C(  1908545495),  INT32_C(       16759),
        -INT32_C(   868388414),  INT32_C(     4290355), -INT32_C(  1038625014),  INT32_C(  1895825473),  INT32_C(   922075970),  INT32_C(     4290355),  INT32_C(       16759),  INT32_C(   399968834) } },
    { {  INT32_C(           5),  INT32_C(          12),  INT32_C(          12),  INT32_C(           4),  INT32_C(           4),  INT32_C(           1),  INT32_C(           0),  INT32_C(          12),
         INT32_C(          10),  INT32_C(          11),  INT32_C(          14),  INT32_C(          11),  INT32_C(          11),  INT32_C(           8),  INT32_C(          14),  INT32_C(          15) },
      { -INT32_C(  1027988521),  INT32_C(  1106142822), -INT32_C(  1034653532), -INT32_C(  1027756524), -INT32_C(  1067114824), -INT32_C(  1034278666), -INT32_C(  1035927552),  INT32_C(  1111369974),
         INT32_C(  1112961188),  INT32_C(  1107990938), -INT32_C(  1077936128), -INT32_C(  1046567977), -INT32_C(  1030727926), -INT32_C(  1039332803),  INT32_C(  1118485873), -INT32_C(  1044926956) },
      { -INT32_C(  1539183002), -INT32_C(  1027756524), -INT32_C(  1027756524),  INT32_C(  1106142822),  INT32_C(  1106142822),  INT32_C(  1724037667), -INT32_C(  1027988521), -INT32_C(  1027756524),
        -INT32_C(  1374371244), -INT32_C(  1112664894),  INT32_C(   515424957), -INT32_C(  1112664894), -INT32_C(  1112664894), -INT32_C(  1034653532),  INT32_C(   515424957),  INT32_C(  1696512194) } },
    { {  INT32_C(           3),  INT32_C(           1),  INT32_C(          12),  INT32_C(           6),  INT32_C(           6),  INT32_C(          12),  INT32_C(           6),  INT32_C(           9),
         INT32_C(          11),  INT32_C(           3),  INT32_C(           7),  INT32_C(          10),  INT32_C(           8),  INT32_C(           1),  INT32_C(           1),  INT32_C(          11) },
      {  INT32_C(  1084898673),  INT32_C(  1116535521),  INT32_C(  1117664051),  INT32_C(  1092626678),  INT32_C(  1110082847), -INT32_C(  1028404019),  INT32_C(  1115767439),  INT32_C(  1119440077),
         INT32_C(  1112725258), -INT32_C(  1040145449),  INT32_C(  1116363817), -INT32_C(  1039306588),  INT32_C(  1111293952), -INT32_C(  1046022717), -INT32_C(  1044219167), -INT32_C(  1035398021) },
      { -INT32_C(  1929715392), -INT32_C(   515855811),  INT32_C(  1092626678),  INT32_C(   858997388),  INT32_C(   858997388),  INT32_C(  1092626678),  INT32_C(   858997388), -INT32_C(   163406285),
         INT32_C(   539555394), -INT32_C(  1929715392), -INT32_C(  1640811710),  INT32_C(   687227550),  INT32_C(  1117664051), -INT32_C(   515855811), -INT32_C(   515855811),  INT32_C(   539555394) } },
    { {  INT32_C(           7),  INT32_C(          11),  INT32_C(           0),  INT32_C(          11),  INT32_C(          14),  INT32_C(           5),  INT32_C(          11),  INT32_C(          10),
         INT32_C(           3),  INT32_C(           2),  INT32_C(           6),  INT32_C(           1),  INT32_C(           4),  INT32_C(           6),  INT32_C(           2),  INT32_C(          12) },
      { -INT32_C(  1038771814),  INT32_C(  1092773478),  INT32_C(  1065520988),  INT32_C(  1108295025),  INT32_C(  1101964247), -INT32_C(  1045891645),  INT32_C(  1100181668),  INT32_C(  1117872456),
         INT32_C(  1117449093),  INT32_C(  1119568527), -INT32_C(  1031444890),  INT32_C(  1091483730), -INT32_C(  1030628311),  INT32_C(  1118432133), -INT32_C(  1033830400),  INT32_C(  1120117719) },
      { -INT32_C(  2104533951),  INT32_C(   255684927), -INT32_C(  1038771814),  INT32_C(   255684927), -INT32_C(  1546173937),  INT32_C(  1547772518),  INT32_C(   255684927),  INT32_C(  1030832002),
         INT32_C(   577136322),  INT32_C(  1718010389), -INT32_C(  1889779422),  INT32_C(  1723995545),  INT32_C(  1092773478), -INT32_C(  1889779422),  INT32_C(  1718010389),  INT32_C(  1108295025) } },
    { {  INT32_C(          13),  INT32_C(          14),  INT32_C(           7),  INT32_C(          11),  INT32_C(           4),  INT32_C(          12),  INT32_C(          11),  INT32_C(           1),
         INT32_C(          10),  INT32_C(           9),  INT32_C(          14),  INT32_C(           2),  INT32_C(          11),  INT32_C(           5),  INT32_C(          12),  INT32_C(           7) },
      {  INT32_C(  1113354404),  INT32_C(  1083933983),  INT32_C(  1119778243),  INT32_C(  1115825111), -INT32_C(  1031324303),  INT32_C(  1107464028), -INT32_C(  1057992212),  INT32_C(  1117890806),
        -INT32_C(  1036963021),  INT32_C(  1114882703),  INT32_C(  1112945459),  INT32_C(  1119531827),  INT32_C(  1099311350), -INT32_C(  1028799857), -INT32_C(  1036970885), -INT32_C(  1038349763) },
      {  INT32_C(  1900184099),  INT32_C(  1030832770), -INT32_C(  1099578560), -INT32_C(  2111580350),  INT32_C(  1083933983),  INT32_C(  1115825111), -INT32_C(  2111580350),  INT32_C(   524442736),
         INT32_C(   601309886), -INT32_C(   683491723),  INT32_C(  1030832770), -INT32_C(  2061548964), -INT32_C(  2111580350), -INT32_C(  1019176059),  INT32_C(  1115825111), -INT32_C(  1099578560) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i vindex = easysimd_mm512_loadu_si512(test_vec[i].vindex);
    easysimd__m512i base_addr0 = easysimd_mm512_loadu_si512(test_vec[i].base_addr);
    easysimd__m512i r0 = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512 base_addr, r, ret;
    easysimd_memcpy(&base_addr, &base_addr0, sizeof(base_addr));
    easysimd_memcpy(&r, &r0, sizeof(r));
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_i32gather_ps(vindex, (void *)&base_addr, 1);
    } EASYSIMD_TEST_PERF_END("_mm512_i32gather_ps");
    easysimd_assert_m512_close(r, ret, 1);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i vindex = easysimd_test_x86_random_i32x16();
    for(size_t i = 0; i < 16; i++) {
      *(((int32_t *)&vindex) + i) = *(((int32_t *)&vindex) + i) & 15;
    }
    easysimd__m512 base_addr = easysimd_test_x86_random_f32x16(-100.0, 100.0);
    easysimd__m512 r = easysimd_mm512_i32gather_ps(vindex, (void *)&base_addr, 1);
    easysimd__m512i base_addr0, r0;
    easysimd_memcpy(&base_addr0, &base_addr, sizeof(base_addr));
    easysimd_memcpy(&r0, &r, sizeof(r));

    easysimd_test_x86_write_i32x16(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, base_addr0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r0, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_i64gather_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t vindex[8];
    int64_t base_addr[8];
    int64_t r[8];
  } test_vec[8] = {
    { {  INT64_C(                  13),  INT64_C(                  10),  INT64_C(                   5),  INT64_C(                   6),
         INT64_C(                  14),  INT64_C(                  11),  INT64_C(                  13),  INT64_C(                   9) },
      { -INT64_C( 4411085110211796296), -INT64_C( 3670385933369749875), -INT64_C( 2333113410766214212), -INT64_C( 1105934734112654624),
        -INT64_C( 5707948438893398640), -INT64_C( 7541156786345234406), -INT64_C( 1626625157668606760),  INT64_C( 6787931577489678347) },
      { -INT64_C( 5270737223433777109),  INT64_C( 7763305327011867879),  INT64_C( 8119119102569793709),  INT64_C( 3130191852625314504),
         INT64_C( 2141139003858799888), -INT64_C( 7751894744662576980), -INT64_C( 5270737223433777109), -INT64_C( 4842196245593397298) } },
    { {  INT64_C(                   9),  INT64_C(                  15),  INT64_C(                  10),  INT64_C(                  11),
         INT64_C(                  12),  INT64_C(                   9),  INT64_C(                  11),  INT64_C(                   2) },
      { -INT64_C( 4437668402884793773), -INT64_C(  896249140066770933),  INT64_C( 1426819476519860671),  INT64_C( 7257227619085822903),
         INT64_C( 1180011296214796676),  INT64_C( 1475341557864886878), -INT64_C( 2844593671864841039), -INT64_C( 6477795695711926108) },
      { -INT64_C( 4615186991630773728), -INT64_C( 3669095485106700301),  INT64_C( 4161312380013762578), -INT64_C( 7838022623649716264),
         INT64_C( 1770822575074566695), -INT64_C( 4615186991630773728), -INT64_C( 7838022623649716264),  INT64_C( 2309152995488132883) } },
    { {  INT64_C(                   9),  INT64_C(                   6),  INT64_C(                   6),  INT64_C(                  15),
         INT64_C(                   6),  INT64_C(                   5),  INT64_C(                   8),  INT64_C(                  13) },
      { -INT64_C( 7488989188763257665), -INT64_C( 7720311273750220909),  INT64_C(  387791943867925207),  INT64_C( 8217573654358814570),
         INT64_C( 7475518895773650980),  INT64_C( 3706794944670952208), -INT64_C( 7823462663708970320), -INT64_C( 1580810994684651543) },
      { -INT64_C( 2912461227430204241), -INT64_C( 1023182788895729647), -INT64_C( 1023182788895729647),  INT64_C( 7041017261641095060),
        -INT64_C( 1023182788895729647), -INT64_C( 3680376925373066812), -INT64_C( 7720311273750220909), -INT64_C( 5195744933627765775) } },
    { {  INT64_C(                   4),  INT64_C(                  10),  INT64_C(                  14),  INT64_C(                  13),
         INT64_C(                  15),  INT64_C(                   5),  INT64_C(                  12),  INT64_C(                   9) },
      {  INT64_C( 4173425240038146323),  INT64_C( 4566638971939883653), -INT64_C( 3982072126954255692),  INT64_C(  682285406736736934),
         INT64_C( 1789289781925746295),  INT64_C( 7823494410955637343), -INT64_C(  706007065918276884), -INT64_C( 7375254470137140059) },
      {  INT64_C( 3165619892895022116),  INT64_C( 5382997135995382766), -INT64_C( 2990501305074303137),  INT64_C( 9194916996779565045),
        -INT64_C( 4839540446264118209), -INT64_C(  924383019786441988), -INT64_C( 7284490259253955085), -INT64_C( 5458538713398382966) } },
    { {  INT64_C(                  11),  INT64_C(                  11),  INT64_C(                   5),  INT64_C(                   5),
         INT64_C(                  12),  INT64_C(                  10),  INT64_C(                  14),  INT64_C(                   9) },
      {  INT64_C( 3749417915355798241), -INT64_C( 3719397229121043702),  INT64_C( 2301993855217637057), -INT64_C( 2777812804593796036),
         INT64_C( 6684028296850094036),  INT64_C( 4120198240773631347),  INT64_C( 2111390946938097400),  INT64_C(  195996945068586471) },
      {  INT64_C( 6995992154729082147),  INT64_C( 6995992154729082147), -INT64_C( 1359054632991324005), -INT64_C( 1359054632991324005),
        -INT64_C(   44729499683517459),  INT64_C( 1639816460818523051),  INT64_C( 5996260746349104226), -INT64_C( 4482099725777786109) } },
    { {  INT64_C(                  10),  INT64_C(                  15),  INT64_C(                   0),  INT64_C(                   3),
         INT64_C(                   2),  INT64_C(                   9),  INT64_C(                  10),  INT64_C(                  10) },
      { -INT64_C( 5348972283147019051), -INT64_C(  133612296366107891), -INT64_C( 5099843346671779224),  INT64_C( 8885086006030898572),
        -INT64_C( 7311404269890377018), -INT64_C( 4949235734580886785),  INT64_C( 1243841794161691914),  INT64_C( 8662695537361127192) },
      {  INT64_C( 4785354040296016249),  INT64_C( 4158932485402683646), -INT64_C( 5348972283147019051),  INT64_C( 8742346375995460717),
         INT64_C( 5984639335982198172),  INT64_C( 7565525450949753171),  INT64_C( 4785354040296016249),  INT64_C( 4785354040296016249) } },
    { {  INT64_C(                  14),  INT64_C(                   5),  INT64_C(                  12),  INT64_C(                  14),
         INT64_C(                  14),  INT64_C(                   1),  INT64_C(                   4),  INT64_C(                  10) },
      {  INT64_C( 8008384531470611680),  INT64_C( 3627205769983710996), -INT64_C( 7819970613511563556),  INT64_C( 4359880659215965859),
         INT64_C( 2096836885402061404), -INT64_C( 9048423629754975795), -INT64_C( 7377828622583327776),  INT64_C(  731339806654350733) },
      { -INT64_C( 2150271295066197418),  INT64_C( 7932844827013489535), -INT64_C( 5501614391049163666), -INT64_C( 2150271295066197418),
        -INT64_C( 2150271295066197418),  INT64_C( 1472434632834615796),  INT64_C( 1666427607402643277),  INT64_C( 6258932928905877280) } },
    { {  INT64_C(                   7),  INT64_C(                   6),  INT64_C(                  10),  INT64_C(                   7),
         INT64_C(                  10),  INT64_C(                   9),  INT64_C(                   7),  INT64_C(                  14) },
      { -INT64_C( 9096701871876386469), -INT64_C( 5854753496144271914), -INT64_C(  574097855045050314),  INT64_C( 2709090868042092957),
        -INT64_C( 6752316634726021392),  INT64_C( 1476459255605240344),  INT64_C( 4058306881460385360), -INT64_C( 3944776308126500527) },
      { -INT64_C( 4630625042459928959), -INT64_C( 4848390152330509886), -INT64_C( 2290451221943218472), -INT64_C( 4630625042459928959),
        -INT64_C( 2290451221943218472),  INT64_C( 3940297541241722917), -INT64_C( 4630625042459928959),  INT64_C( 7280882135067963071) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i vindex = easysimd_mm512_loadu_si512(test_vec[i].vindex);
    easysimd__m512i base_addr = easysimd_mm512_loadu_si512(test_vec[i].base_addr);
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_i64gather_epi64(vindex, (void *)&base_addr, 1);
    } EASYSIMD_TEST_PERF_END("_mm512_i64gather_epi64");
    easysimd_assert_m512i_i64(r, ==, ret);
   }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i vindex = easysimd_test_x86_random_i64x8();
    for(size_t i = 0; i < 8; i++) {
      *(((int64_t *)&vindex) + i) = *(((int64_t *)&vindex) + i) & 15;
    }
    easysimd__m512i base_addr = easysimd_test_x86_random_i64x8();
    easysimd__m512i r = easysimd_mm512_i64gather_epi64(vindex, (void *)&base_addr, 1);

    easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, base_addr, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_i64gather_pd(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t vindex[8];
    int64_t base_addr[8];
    int64_t r[8];
  } test_vec[8] = {
    { {  INT64_C(                  10),  INT64_C(                   4),  INT64_C(                   4),  INT64_C(                  14),
         INT64_C(                  15),  INT64_C(                   5),  INT64_C(                   0),  INT64_C(                  15) },
      {  INT64_C( 4624836529339309752),  INT64_C( 4636727439730451415),  INT64_C( 4635582540262680822), -INT64_C( 4587854235836738437),
        -INT64_C( 4592989043099382579),  INT64_C( 4630954387958115860), -INT64_C( 4597829005323922309),  INT64_C( 4634614266342796165) },
      {  INT64_C( 2951617356633881968),  INT64_C( 4427218578595297361),  INT64_C( 4427218578595297361), -INT64_C( 1890791267555196840),
         INT64_C( 6117509589834987072),  INT64_C(  737869762951917240),  INT64_C( 4624836529339309752),  INT64_C( 6117509589834987072) } },
    { {  INT64_C(                   9),  INT64_C(                  10),  INT64_C(                   6),  INT64_C(                  14),
         INT64_C(                   2),  INT64_C(                  14),  INT64_C(                   4),  INT64_C(                   0) },
      {  INT64_C( 4635097699615296717),  INT64_C( 4630153591649374044), -INT64_C( 4590240439951803023), -INT64_C( 4593677249417440133),
        -INT64_C( 4591246712993543619), -INT64_C( 4602560599682428436), -INT64_C( 4587002070344746926),  INT64_C( 4631749554767323464) },
      {  INT64_C( 8160594663753237135),  INT64_C( 4427390559218890178), -INT64_C( 7194230188746719149),  INT64_C( 3504881374004854849),
        -INT64_C( 8116541702450393908),  INT64_C( 3504881374004854849), -INT64_C(  737869762556384052),  INT64_C( 4635097699615296717) } },
    { {  INT64_C(                  12),  INT64_C(                  11),  INT64_C(                   8),  INT64_C(                   7),
         INT64_C(                   9),  INT64_C(                   4),  INT64_C(                  10),  INT64_C(                   2) },
      {  INT64_C( 4620569368692376207),  INT64_C( 4626770262429311959),  INT64_C( 4634898556069273928),  INT64_C( 4629829895426156790),
         INT64_C( 4635246177665511588), -INT64_C( 4596759400412421816),  INT64_C( 4634567119284197130),  INT64_C( 4629855228174060749) },
      {  INT64_C( 1475739528348407562),  INT64_C( 8854437783001303613),  INT64_C( 4626770262429311959),  INT64_C( 3861566464492558144),
         INT64_C( 5206220092068425891),  INT64_C( 4427218578594303836), -INT64_C( 2213448617941123728), -INT64_C( 6640768621241161483) } },
    { {  INT64_C(                   8),  INT64_C(                   9),  INT64_C(                   0),  INT64_C(                  11),
         INT64_C(                   2),  INT64_C(                   9),  INT64_C(                   3),  INT64_C(                  13) },
      {  INT64_C( 4635930161858918482), -INT64_C( 4590438879810384036), -INT64_C( 4588335558046913659),  INT64_C( 4634259607872140739),
         INT64_C( 4631795998138480722), -INT64_C( 4601248926290956780), -INT64_C( 4589460754266314506),  INT64_C( 4636610627615116493) },
      { -INT64_C( 4590438879810384036), -INT64_C( 8808957874501467505),  INT64_C( 4635930161858918482),  INT64_C( 5902958797545810165),
        -INT64_C( 8116541689748028130), -INT64_C( 8808957874501467505), -INT64_C( 4427218477289182331),  INT64_C( 2213609288855735164) } },
    { {  INT64_C(                   6),  INT64_C(                  11),  INT64_C(                  15),  INT64_C(                  12),
         INT64_C(                   4),  INT64_C(                  14),  INT64_C(                   0),  INT64_C(                   0) },
      {  INT64_C( 4631331564426908140),  INT64_C( 4636039233412393861),  INT64_C( 4634320828679575306),  INT64_C( 4629435830458761871),
         INT64_C( 4631953624125438689),  INT64_C( 4627096773402296320), -INT64_C( 4617180409972779909),  INT64_C( 4621537642612260864) },
      { -INT64_C( 8854437155380576187), -INT64_C( 6640827854088757576),  INT64_C( 5794511424559974976),  INT64_C( 8116567392480822558),
        -INT64_C( 5165088340075754619),  INT64_C( 7655398790589464662),  INT64_C( 4631331564426908140),  INT64_C( 4631331564426908140) } },
    { {  INT64_C(                   9),  INT64_C(                   2),  INT64_C(                  11),  INT64_C(                   4),
         INT64_C(                   5),  INT64_C(                   8),  INT64_C(                   2),  INT64_C(                  15) },
      { -INT64_C( 4612541702356588298),  INT64_C( 4635543837453383107),  INT64_C( 4626114425733576131), -INT64_C( 4593277554950511002),
        -INT64_C( 4588487554534337413), -INT64_C( 4587204028640536822),  INT64_C( 4604840546993784750), -INT64_C( 4588886545313824768) },
      { -INT64_C( 4377405643198551819), -INT64_C(  737534820782796964),  INT64_C( 2951479811878588252),  INT64_C( 6640827871646250434),
        -INT64_C( 8116567392412238603),  INT64_C( 4635543837453383107), -INT64_C(  737534820782796964),  INT64_C( 3693672270384186176) } },
    { {  INT64_C(                   7),  INT64_C(                   7),  INT64_C(                  12),  INT64_C(                   2),
         INT64_C(                   5),  INT64_C(                   7),  INT64_C(                   8),  INT64_C(                  13) },
      { -INT64_C( 4587000662969863373), -INT64_C( 4598338475031768596), -INT64_C( 4611438320447882527),  INT64_C( 4635324286971548795),
         INT64_C( 4631810071887316255), -INT64_C( 4589470605890499379),  INT64_C( 4628937619749984010), -INT64_C( 4591584482965596406) },
      {  INT64_C( 3416971109278543040),  INT64_C( 3416971109278543040), -INT64_C( 5902958101565314171),  INT64_C( 5903304694477042483),
        -INT64_C( 8854437155359926349),  INT64_C( 3416971109278543040), -INT64_C( 4598338475031768596),  INT64_C( 5165088340646571883) } },
    { {  INT64_C(                  12),  INT64_C(                   6),  INT64_C(                   7),  INT64_C(                   1),
         INT64_C(                  13),  INT64_C(                   0),  INT64_C(                  12),  INT64_C(                   3) },
      {  INT64_C( 4616639978017495450),  INT64_C( 4625866727754070753), -INT64_C( 4594487897350366822), -INT64_C( 4587649462791181435),
         INT64_C( 4611888680410619576),  INT64_C( 4632129545985882849), -INT64_C( 4600742271332877599),  INT64_C( 4631914217628699197) },
      { -INT64_C( 7378697626688790201),  INT64_C( 7009762748009627665),  INT64_C( 3630261587630809408), -INT64_C( 2215751665261635175),
        -INT64_C( 7378697629472902559),  INT64_C( 4616639978017495450), -INT64_C( 7378697626688790201),  INT64_C( 1475739493206694297) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i vindex = easysimd_mm512_loadu_si512(test_vec[i].vindex);
    easysimd__m512i base_addr0 = easysimd_mm512_loadu_si512(test_vec[i].base_addr);
    easysimd__m512i r0 = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512d base_addr, r, ret;
    easysimd_memcpy(&base_addr, &base_addr0, sizeof(base_addr));
    easysimd_memcpy(&r, &r0, sizeof(r));
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      ret = easysimd_mm512_i64gather_pd(vindex, (void *)&base_addr, 1);
    } EASYSIMD_TEST_PERF_END("_mm512_i64gather_pd");
    easysimd_assert_m512d_close(r, ret, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i vindex = easysimd_test_x86_random_i64x8();
    for(size_t i = 0; i < 8; i++) {
      *(((int64_t *)&vindex) + i) = *(((int64_t *)&vindex) + i) & 15;
    }
    easysimd__m512d base_addr = easysimd_test_x86_random_f64x8(-100.0, 100.0);
    easysimd__m512d r = easysimd_mm512_i64gather_pd(vindex, (void *)&base_addr, 1);
    easysimd__m512i base_addr0, r0;
    easysimd_memcpy(&base_addr0, &base_addr, sizeof(base_addr));
    easysimd_memcpy(&r0, &r, sizeof(r));

    easysimd_test_x86_write_i64x8(2, vindex, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, base_addr0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r0, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_i64gather_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_i64gather_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_i64gather_epi32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_i64gather_epi32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_i32gather_epi32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_i32gather_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_i64gather_epi64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_i64gather_pd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>