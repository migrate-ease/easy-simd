#if !defined (EASYSIMD_ARM_SVE_DIV_H)
#define EASYSIMD_ARM_SVE_DIV_H

  #define easysimd_svdiv_f32_z(pg, op1, op2)   svdiv_f32_z((pg), (op1), (op2))
  #define easysimd_svdiv_f64_z(pg, op1, op2)   svdiv_f64_z((pg), (op1), (op2))
  #define easysimd_svdiv_s32_z(pg, op1, op2)   svdiv_s32_z((pg), (op1), (op2))
  #define easysimd_svdiv_s64_z(pg, op1, op2)   svdiv_s64_z((pg), (op1), (op2))
  #define easysimd_svdiv_u32_z(pg, op1, op2)   svdiv_u32_z((pg), (op1), (op2))
  #define easysimd_svdiv_u64_z(pg, op1, op2)   svdiv_u64_z((pg), (op1), (op2))
  
#endif