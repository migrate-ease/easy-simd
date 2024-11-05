
#include <stdlib.h>
#include <string.h>
#include <easysimd/easysimd.h>

int main()
{
  struct {
    int32_t a[4];
    int32_t b[4];
    int32_t r[4];
  } test_vec = {
      {  INT32_C(  1587156417),  INT32_C(  1768270179), -INT32_C(  1942404587),  INT32_C(   346970517) },
      {  INT32_C(  2141391970),  INT32_C(  1584534422),  INT32_C(  1144809083), -INT32_C(   446909148) },
      { -INT32_C(   566418909), -INT32_C(   942162695), -INT32_C(   797595504), -INT32_C(    99938631) }
  };

    __m128i a = _mm_loadu_epi32(test_vec.a);
    __m128i b = _mm_loadu_epi32(test_vec.b);
    __m128i r = _mm_loadu_epi32(test_vec.r);
    __m128i sum = _mm_add_epi32(a, b);

    if (memcmp(&sum, &r, sizeof(__m128i)) != 0) {
        fprintf(stderr, "Example test failed.\n");
    } else {
        fprintf(stderr, "Example test OK.\n");
    }

    return 0;
}