#if !defined(EASYSIMD_ARM_SVE_EXTEND_H)
#define EASYSIMD_ARM_SVE_EXTEND_H

#include "types.h"

#if defined(EASYSIMD_ARM_SVE_NATIVE)

/*
    * Set one element be true according to index and return it.
    * The index counts from 1.
*/
#define svSetOneElementTrue(bitwidth, index) ({         \
    svbool_t pg1 = svwhilelt_b##bitwidth(1, (index)),   \
             pg2 = svwhilele_b##bitwidth(1, (index));   \
    sveor_b_z(svptrue_b##bitwidth(), pg1, pg2);})

/*
    * Return one element in a vector according to index.
    * The index counts from 1.
*/
#define svGetElementFromVector(type, bitwidth, sv, index) ({    \
    svbool_t pg = svSetOneElementTrue(bitwidth, index);         \
    (type)svaddv(pg, (sv));})

#define EASYSIMD_MASK_TO_B64(mask, index) ({    \
    svcmpne_n_u64(svptrue_b64(),    \
                svand_u64_x(svptrue_b64(), svdup_n_u64((mask) >> ((index) * 2)), mask64.sve_u64), 0);  \
})

#define EASYSIMD_MASK_TO_B32(mask, index) ({    \
    svcmpne_n_u32(svptrue_b32(),    \
                svand_u32_x(svptrue_b32(), svdup_n_u32(mask >> ((index) * 4)), mask32.sve_u32), 0);  \
})

#define EASYSIMD_MASK_TO_B16(mask, index) ({    \
    svcmpne_n_u16(svptrue_b16(),    \
                svand_u16_x(svptrue_b16(), svdup_n_u16(mask >> ((index) * 8)), mask16.sve_u16), 0);  \
})


#define EASYSIMD_MASK_TO_B8(mask, index) ({    \
    svcmpne_n_u8(svptrue_b8(),    \
               svand_u8_x(svptrue_b8(),    \
                          svsel_u8(svwhilele_b8(1, 8), \
                                   svdup_n_u8(mask >> ((index) * 16)),    \
                                   svdup_n_u8(mask >> ((index) * 16 + 8))),    \
                          svsel_u8(svwhilele_b8(1, 8), \
                                   mask8l.sve_u8,   \
                                   mask8h.sve_u8)),   \
               0);  \
})

#define EASYSIMD_B64_TO_MASK(_k, _pg, _index)  ({  \
          (_k) |= svaddv_u64(_pg, mask64.sve_u64) << (_index * 2);  \
        })

#define EASYSIMD_B32_TO_MASK(_k, _pg, _index)  ({  \
          (_k) |= svaddv_u32(_pg, mask32.sve_u32) << (_index * 4);  \
        })

#define EASYSIMD_B16_TO_MASK(_k, _pg, _index)  ({  \
          (_k) |= svaddv_u16(_pg, mask16.sve_u16) << (_index * 8);  \
        })

#define EASYSIMD_B8_TO_MASK(_k, _pg, _index)  ({  \
          (_k) |= svaddv_u8(_pg, mask8l.sve_u8) << (_index * 16);  \
          (_k) |= svaddv_u8(_pg, mask8h.sve_u8) << (_index * 16 + 8);  \
        })

#endif
#endif