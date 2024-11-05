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
 *   2021      Evan Nemerson <evan@nemerson.com>
 */

#if !defined(EASYSIMD_ARM_SVE_REINTERPRET_H)
#define EASYSIMD_ARM_SVE_REINTERPRET_H

#include "types.h"

HEDLEY_DIAGNOSTIC_PUSH
EASYSIMD_DISABLE_UNWANTED_DIAGNOSTICS

#if defined(EASYSIMD_ARM_SVE_NATIVE)
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8_s16(   easysimd_svint16_t op) { return   svreinterpret_s8_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8_s32(   easysimd_svint32_t op) { return   svreinterpret_s8_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8_s64(   easysimd_svint64_t op) { return   svreinterpret_s8_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t    easysimd_svreinterpret_s8_u8(   easysimd_svuint8_t op) { return    svreinterpret_s8_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8_u16(  easysimd_svuint16_t op) { return   svreinterpret_s8_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8_u32(  easysimd_svuint32_t op) { return   svreinterpret_s8_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8_u64(  easysimd_svuint64_t op) { return   svreinterpret_s8_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8_f16( easysimd_svfloat16_t op) { return   svreinterpret_s8_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8_f32( easysimd_svfloat32_t op) { return   svreinterpret_s8_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8_f64( easysimd_svfloat64_t op) { return   svreinterpret_s8_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t   easysimd_svreinterpret_s16_s8(    easysimd_svint8_t op) { return   svreinterpret_s16_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16_s32(   easysimd_svint32_t op) { return  svreinterpret_s16_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16_s64(   easysimd_svint64_t op) { return  svreinterpret_s16_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t   easysimd_svreinterpret_s16_u8(   easysimd_svuint8_t op) { return   svreinterpret_s16_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16_u16(  easysimd_svuint16_t op) { return  svreinterpret_s16_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16_u32(  easysimd_svuint32_t op) { return  svreinterpret_s16_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16_u64(  easysimd_svuint64_t op) { return  svreinterpret_s16_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16_f16( easysimd_svfloat16_t op) { return  svreinterpret_s16_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16_f32( easysimd_svfloat32_t op) { return  svreinterpret_s16_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16_f64( easysimd_svfloat64_t op) { return  svreinterpret_s16_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t   easysimd_svreinterpret_s32_s8(    easysimd_svint8_t op) { return   svreinterpret_s32_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32_s16(   easysimd_svint16_t op) { return  svreinterpret_s32_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32_s64(   easysimd_svint64_t op) { return  svreinterpret_s32_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t   easysimd_svreinterpret_s32_u8(   easysimd_svuint8_t op) { return   svreinterpret_s32_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32_u16(  easysimd_svuint16_t op) { return  svreinterpret_s32_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32_u32(  easysimd_svuint32_t op) { return  svreinterpret_s32_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32_u64(  easysimd_svuint64_t op) { return  svreinterpret_s32_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32_f16( easysimd_svfloat16_t op) { return  svreinterpret_s32_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32_f32( easysimd_svfloat32_t op) { return  svreinterpret_s32_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32_f64( easysimd_svfloat64_t op) { return  svreinterpret_s32_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t   easysimd_svreinterpret_s64_s8(    easysimd_svint8_t op) { return   svreinterpret_s64_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64_s16(   easysimd_svint16_t op) { return  svreinterpret_s64_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64_s32(   easysimd_svint32_t op) { return  svreinterpret_s64_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t   easysimd_svreinterpret_s64_u8(   easysimd_svuint8_t op) { return   svreinterpret_s64_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64_u16(  easysimd_svuint16_t op) { return  svreinterpret_s64_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64_u32(  easysimd_svuint32_t op) { return  svreinterpret_s64_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64_u64(  easysimd_svuint64_t op) { return  svreinterpret_s64_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64_f16( easysimd_svfloat16_t op) { return  svreinterpret_s64_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64_f32( easysimd_svfloat32_t op) { return  svreinterpret_s64_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64_f64( easysimd_svfloat64_t op) { return  svreinterpret_s64_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t    easysimd_svreinterpret_u8_s8(    easysimd_svint8_t op) { return    svreinterpret_u8_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8_s16(   easysimd_svint16_t op) { return   svreinterpret_u8_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8_s32(   easysimd_svint32_t op) { return   svreinterpret_u8_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8_s64(   easysimd_svint64_t op) { return   svreinterpret_u8_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8_u16(  easysimd_svuint16_t op) { return   svreinterpret_u8_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8_u32(  easysimd_svuint32_t op) { return   svreinterpret_u8_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8_u64(  easysimd_svuint64_t op) { return   svreinterpret_u8_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8_f16( easysimd_svfloat16_t op) { return   svreinterpret_u8_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8_f32( easysimd_svfloat32_t op) { return   svreinterpret_u8_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8_f64( easysimd_svfloat64_t op) { return   svreinterpret_u8_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t   easysimd_svreinterpret_u16_s8(    easysimd_svint8_t op) { return   svreinterpret_u16_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16_s16(   easysimd_svint16_t op) { return  svreinterpret_u16_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16_s32(   easysimd_svint32_t op) { return  svreinterpret_u16_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16_s64(   easysimd_svint64_t op) { return  svreinterpret_u16_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t   easysimd_svreinterpret_u16_u8(   easysimd_svuint8_t op) { return   svreinterpret_u16_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16_u32(  easysimd_svuint32_t op) { return  svreinterpret_u16_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16_u64(  easysimd_svuint64_t op) { return  svreinterpret_u16_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16_f16( easysimd_svfloat16_t op) { return  svreinterpret_u16_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16_f32( easysimd_svfloat32_t op) { return  svreinterpret_u16_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16_f64( easysimd_svfloat64_t op) { return  svreinterpret_u16_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t   easysimd_svreinterpret_u32_s8(    easysimd_svint8_t op) { return   svreinterpret_u32_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32_s16(   easysimd_svint16_t op) { return  svreinterpret_u32_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32_s32(   easysimd_svint32_t op) { return  svreinterpret_u32_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32_s64(   easysimd_svint64_t op) { return  svreinterpret_u32_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t   easysimd_svreinterpret_u32_u8(   easysimd_svuint8_t op) { return   svreinterpret_u32_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32_u16(  easysimd_svuint16_t op) { return  svreinterpret_u32_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32_u64(  easysimd_svuint64_t op) { return  svreinterpret_u32_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32_f16( easysimd_svfloat16_t op) { return  svreinterpret_u32_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32_f32( easysimd_svfloat32_t op) { return  svreinterpret_u32_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32_f64( easysimd_svfloat64_t op) { return  svreinterpret_u32_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t   easysimd_svreinterpret_u64_s8(    easysimd_svint8_t op) { return   svreinterpret_u64_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64_s16(   easysimd_svint16_t op) { return  svreinterpret_u64_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64_s32(   easysimd_svint32_t op) { return  svreinterpret_u64_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64_s64(   easysimd_svint64_t op) { return  svreinterpret_u64_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t   easysimd_svreinterpret_u64_u8(   easysimd_svuint8_t op) { return   svreinterpret_u64_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64_u16(  easysimd_svuint16_t op) { return  svreinterpret_u64_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64_u32(  easysimd_svuint32_t op) { return  svreinterpret_u64_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64_f16( easysimd_svfloat16_t op) { return  svreinterpret_u64_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64_f32( easysimd_svfloat32_t op) { return  svreinterpret_u64_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64_f64( easysimd_svfloat64_t op) { return  svreinterpret_u64_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t   easysimd_svreinterpret_f16_s8(    easysimd_svint8_t op) { return   svreinterpret_f16_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16_s16(   easysimd_svint16_t op) { return  svreinterpret_f16_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16_s32(   easysimd_svint32_t op) { return  svreinterpret_f16_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16_s64(   easysimd_svint64_t op) { return  svreinterpret_f16_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t   easysimd_svreinterpret_f16_u8(   easysimd_svuint8_t op) { return   svreinterpret_f16_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16_u16(  easysimd_svuint16_t op) { return  svreinterpret_f16_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16_u32(  easysimd_svuint32_t op) { return  svreinterpret_f16_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16_u64(  easysimd_svuint64_t op) { return  svreinterpret_f16_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16_f32( easysimd_svfloat32_t op) { return  svreinterpret_f16_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16_f64( easysimd_svfloat64_t op) { return  svreinterpret_f16_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t   easysimd_svreinterpret_f32_s8(    easysimd_svint8_t op) { return   svreinterpret_f32_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32_s16(   easysimd_svint16_t op) { return  svreinterpret_f32_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32_s32(   easysimd_svint32_t op) { return  svreinterpret_f32_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32_s64(   easysimd_svint64_t op) { return  svreinterpret_f32_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t   easysimd_svreinterpret_f32_u8(   easysimd_svuint8_t op) { return   svreinterpret_f32_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32_u16(  easysimd_svuint16_t op) { return  svreinterpret_f32_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32_u32(  easysimd_svuint32_t op) { return  svreinterpret_f32_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32_u64(  easysimd_svuint64_t op) { return  svreinterpret_f32_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32_f16( easysimd_svfloat16_t op) { return  svreinterpret_f32_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32_f64( easysimd_svfloat64_t op) { return  svreinterpret_f32_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t   easysimd_svreinterpret_f64_s8(    easysimd_svint8_t op) { return   svreinterpret_f64_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64_s16(   easysimd_svint16_t op) { return  svreinterpret_f64_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64_s32(   easysimd_svint32_t op) { return  svreinterpret_f64_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64_s64(   easysimd_svint64_t op) { return  svreinterpret_f64_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t   easysimd_svreinterpret_f64_u8(   easysimd_svuint8_t op) { return   svreinterpret_f64_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64_u16(  easysimd_svuint16_t op) { return  svreinterpret_f64_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64_u32(  easysimd_svuint32_t op) { return  svreinterpret_f64_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64_u64(  easysimd_svuint64_t op) { return  svreinterpret_f64_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64_f16( easysimd_svfloat16_t op) { return  svreinterpret_f64_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64_f32( easysimd_svfloat32_t op) { return  svreinterpret_f64_f32(op); }
#else
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s8_s16,     easysimd_svint8_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s8_s32,     easysimd_svint8_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s8_s64,     easysimd_svint8_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(   easysimd_svreinterpret_s8_u8,     easysimd_svint8_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s8_u16,     easysimd_svint8_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s8_u32,     easysimd_svint8_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s8_u64,     easysimd_svint8_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s8_f16,     easysimd_svint8_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s8_f32,     easysimd_svint8_t,  easysimd_svfloat32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s8_f64,     easysimd_svint8_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s16_s8,    easysimd_svint16_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s16_s32,    easysimd_svint16_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s16_s64,    easysimd_svint16_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s16_u8,    easysimd_svint16_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s16_u16,    easysimd_svint16_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s16_u32,    easysimd_svint16_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s16_u64,    easysimd_svint16_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s16_f16,    easysimd_svint16_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s16_f32,    easysimd_svint16_t,  easysimd_svfloat32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s16_f64,    easysimd_svint16_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s32_s8,    easysimd_svint32_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s32_s16,    easysimd_svint32_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s32_s64,    easysimd_svint32_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s32_u8,    easysimd_svint32_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s32_u16,    easysimd_svint32_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s32_u32,    easysimd_svint32_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s32_u64,    easysimd_svint32_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s32_f16,    easysimd_svint32_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s32_f32,    easysimd_svint32_t,  easysimd_svfloat32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s32_f64,    easysimd_svint32_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s64_s8,    easysimd_svint64_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s64_s16,    easysimd_svint64_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s64_s32,    easysimd_svint64_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_s64_u8,    easysimd_svint64_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s64_u16,    easysimd_svint64_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s64_u32,    easysimd_svint64_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s64_u64,    easysimd_svint64_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s64_f16,    easysimd_svint64_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s64_f32,    easysimd_svint64_t,  easysimd_svfloat32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_s64_f64,    easysimd_svint64_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(   easysimd_svreinterpret_u8_s8,    easysimd_svuint8_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u8_s16,    easysimd_svuint8_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u8_s32,    easysimd_svuint8_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u8_s64,    easysimd_svuint8_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u8_u16,    easysimd_svuint8_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u8_u32,    easysimd_svuint8_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u8_u64,    easysimd_svuint8_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u8_f16,    easysimd_svuint8_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u8_f32,    easysimd_svuint8_t,  easysimd_svfloat32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u8_f64,    easysimd_svuint8_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u16_s8,   easysimd_svuint16_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u16_s16,   easysimd_svuint16_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u16_s32,   easysimd_svuint16_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u16_s64,   easysimd_svuint16_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u16_u8,   easysimd_svuint16_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u16_u32,   easysimd_svuint16_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u16_u64,   easysimd_svuint16_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u16_f16,   easysimd_svuint16_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u16_f32,   easysimd_svuint16_t,  easysimd_svfloat32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u16_f64,   easysimd_svuint16_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u32_s8,   easysimd_svuint32_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u32_s16,   easysimd_svuint32_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u32_s32,   easysimd_svuint32_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u32_s64,   easysimd_svuint32_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u32_u8,   easysimd_svuint32_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u32_u16,   easysimd_svuint32_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u32_u64,   easysimd_svuint32_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u32_f16,   easysimd_svuint32_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u32_f32,   easysimd_svuint32_t,  easysimd_svfloat32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u32_f64,   easysimd_svuint32_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u64_s8,   easysimd_svuint64_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u64_s16,   easysimd_svuint64_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u64_s32,   easysimd_svuint64_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u64_s64,   easysimd_svuint64_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_u64_u8,   easysimd_svuint64_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u64_u16,   easysimd_svuint64_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u64_u32,   easysimd_svuint64_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u64_f16,   easysimd_svuint64_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u64_f32,   easysimd_svuint64_t,  easysimd_svfloat32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_u64_f64,   easysimd_svuint64_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_f16_s8,  easysimd_svfloat16_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f16_s16,  easysimd_svfloat16_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f16_s32,  easysimd_svfloat16_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f16_s64,  easysimd_svfloat16_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_f16_u8,  easysimd_svfloat16_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f16_u16,  easysimd_svfloat16_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f16_u32,  easysimd_svfloat16_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f16_u64,  easysimd_svfloat16_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f16_f32,  easysimd_svfloat16_t,  easysimd_svfloat32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f16_f64,  easysimd_svfloat16_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_f32_s8,  easysimd_svfloat32_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f32_s16,  easysimd_svfloat32_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f32_s32,  easysimd_svfloat32_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f32_s64,  easysimd_svfloat32_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_f32_u8,  easysimd_svfloat32_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f32_u16,  easysimd_svfloat32_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f32_u32,  easysimd_svfloat32_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f32_u64,  easysimd_svfloat32_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f32_f16,  easysimd_svfloat32_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f32_f64,  easysimd_svfloat32_t,  easysimd_svfloat64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_f64_s8,  easysimd_svfloat64_t,     easysimd_svint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f64_s16,  easysimd_svfloat64_t,    easysimd_svint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f64_s32,  easysimd_svfloat64_t,    easysimd_svint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f64_s64,  easysimd_svfloat64_t,    easysimd_svint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  easysimd_svreinterpret_f64_u8,  easysimd_svfloat64_t,    easysimd_svuint8_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f64_u16,  easysimd_svfloat64_t,   easysimd_svuint16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f64_u32,  easysimd_svfloat64_t,   easysimd_svuint32_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f64_u64,  easysimd_svfloat64_t,   easysimd_svuint64_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f64_f16,  easysimd_svfloat64_t,  easysimd_svfloat16_t)
  EASYSIMD_DEFINE_CONVERSION_FUNCTION_( easysimd_svreinterpret_f64_f32,  easysimd_svfloat64_t,  easysimd_svfloat32_t)
#endif

#if defined(__cplusplus)
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8(   easysimd_svint16_t op) { return   easysimd_svreinterpret_s8_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8(   easysimd_svint32_t op) { return   easysimd_svreinterpret_s8_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8(   easysimd_svint64_t op) { return   easysimd_svreinterpret_s8_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8(   easysimd_svuint8_t op) { return    easysimd_svreinterpret_s8_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8(  easysimd_svuint16_t op) { return   easysimd_svreinterpret_s8_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8(  easysimd_svuint32_t op) { return   easysimd_svreinterpret_s8_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8(  easysimd_svuint64_t op) { return   easysimd_svreinterpret_s8_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8( easysimd_svfloat16_t op) { return   easysimd_svreinterpret_s8_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8( easysimd_svfloat32_t op) { return   easysimd_svreinterpret_s8_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES     easysimd_svint8_t   easysimd_svreinterpret_s8( easysimd_svfloat64_t op) { return   easysimd_svreinterpret_s8_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16(    easysimd_svint8_t op) { return   easysimd_svreinterpret_s16_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16(   easysimd_svint32_t op) { return  easysimd_svreinterpret_s16_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16(   easysimd_svint64_t op) { return  easysimd_svreinterpret_s16_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16(   easysimd_svuint8_t op) { return   easysimd_svreinterpret_s16_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16(  easysimd_svuint16_t op) { return  easysimd_svreinterpret_s16_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16(  easysimd_svuint32_t op) { return  easysimd_svreinterpret_s16_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16(  easysimd_svuint64_t op) { return  easysimd_svreinterpret_s16_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16( easysimd_svfloat16_t op) { return  easysimd_svreinterpret_s16_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16( easysimd_svfloat32_t op) { return  easysimd_svreinterpret_s16_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint16_t  easysimd_svreinterpret_s16( easysimd_svfloat64_t op) { return  easysimd_svreinterpret_s16_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32(    easysimd_svint8_t op) { return   easysimd_svreinterpret_s32_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32(   easysimd_svint16_t op) { return  easysimd_svreinterpret_s32_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32(   easysimd_svint64_t op) { return  easysimd_svreinterpret_s32_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32(   easysimd_svuint8_t op) { return   easysimd_svreinterpret_s32_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32(  easysimd_svuint16_t op) { return  easysimd_svreinterpret_s32_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32(  easysimd_svuint32_t op) { return  easysimd_svreinterpret_s32_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32(  easysimd_svuint64_t op) { return  easysimd_svreinterpret_s32_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32( easysimd_svfloat16_t op) { return  easysimd_svreinterpret_s32_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32( easysimd_svfloat32_t op) { return  easysimd_svreinterpret_s32_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint32_t  easysimd_svreinterpret_s32( easysimd_svfloat64_t op) { return  easysimd_svreinterpret_s32_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64(    easysimd_svint8_t op) { return   easysimd_svreinterpret_s64_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64(   easysimd_svint16_t op) { return  easysimd_svreinterpret_s64_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64(   easysimd_svint32_t op) { return  easysimd_svreinterpret_s64_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64(   easysimd_svuint8_t op) { return   easysimd_svreinterpret_s64_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64(  easysimd_svuint16_t op) { return  easysimd_svreinterpret_s64_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64(  easysimd_svuint32_t op) { return  easysimd_svreinterpret_s64_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64(  easysimd_svuint64_t op) { return  easysimd_svreinterpret_s64_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64( easysimd_svfloat16_t op) { return  easysimd_svreinterpret_s64_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64( easysimd_svfloat32_t op) { return  easysimd_svreinterpret_s64_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svint64_t  easysimd_svreinterpret_s64( easysimd_svfloat64_t op) { return  easysimd_svreinterpret_s64_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8(    easysimd_svint8_t op) { return    easysimd_svreinterpret_u8_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8(   easysimd_svint16_t op) { return   easysimd_svreinterpret_u8_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8(   easysimd_svint32_t op) { return   easysimd_svreinterpret_u8_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8(   easysimd_svint64_t op) { return   easysimd_svreinterpret_u8_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8(  easysimd_svuint16_t op) { return   easysimd_svreinterpret_u8_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8(  easysimd_svuint32_t op) { return   easysimd_svreinterpret_u8_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8(  easysimd_svuint64_t op) { return   easysimd_svreinterpret_u8_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8( easysimd_svfloat16_t op) { return   easysimd_svreinterpret_u8_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8( easysimd_svfloat32_t op) { return   easysimd_svreinterpret_u8_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES    easysimd_svuint8_t   easysimd_svreinterpret_u8( easysimd_svfloat64_t op) { return   easysimd_svreinterpret_u8_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16(    easysimd_svint8_t op) { return   easysimd_svreinterpret_u16_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16(   easysimd_svint16_t op) { return  easysimd_svreinterpret_u16_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16(   easysimd_svint32_t op) { return  easysimd_svreinterpret_u16_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16(   easysimd_svint64_t op) { return  easysimd_svreinterpret_u16_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16(   easysimd_svuint8_t op) { return   easysimd_svreinterpret_u16_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16(  easysimd_svuint32_t op) { return  easysimd_svreinterpret_u16_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16(  easysimd_svuint64_t op) { return  easysimd_svreinterpret_u16_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16( easysimd_svfloat16_t op) { return  easysimd_svreinterpret_u16_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16( easysimd_svfloat32_t op) { return  easysimd_svreinterpret_u16_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint16_t  easysimd_svreinterpret_u16( easysimd_svfloat64_t op) { return  easysimd_svreinterpret_u16_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32(    easysimd_svint8_t op) { return   easysimd_svreinterpret_u32_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32(   easysimd_svint16_t op) { return  easysimd_svreinterpret_u32_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32(   easysimd_svint32_t op) { return  easysimd_svreinterpret_u32_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32(   easysimd_svint64_t op) { return  easysimd_svreinterpret_u32_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32(   easysimd_svuint8_t op) { return   easysimd_svreinterpret_u32_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32(  easysimd_svuint16_t op) { return  easysimd_svreinterpret_u32_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32(  easysimd_svuint64_t op) { return  easysimd_svreinterpret_u32_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32( easysimd_svfloat16_t op) { return  easysimd_svreinterpret_u32_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32( easysimd_svfloat32_t op) { return  easysimd_svreinterpret_u32_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint32_t  easysimd_svreinterpret_u32( easysimd_svfloat64_t op) { return  easysimd_svreinterpret_u32_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64(    easysimd_svint8_t op) { return   easysimd_svreinterpret_u64_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64(   easysimd_svint16_t op) { return  easysimd_svreinterpret_u64_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64(   easysimd_svint32_t op) { return  easysimd_svreinterpret_u64_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64(   easysimd_svint64_t op) { return  easysimd_svreinterpret_u64_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64(   easysimd_svuint8_t op) { return   easysimd_svreinterpret_u64_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64(  easysimd_svuint16_t op) { return  easysimd_svreinterpret_u64_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64(  easysimd_svuint32_t op) { return  easysimd_svreinterpret_u64_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64( easysimd_svfloat16_t op) { return  easysimd_svreinterpret_u64_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64( easysimd_svfloat32_t op) { return  easysimd_svreinterpret_u64_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES   easysimd_svuint64_t  easysimd_svreinterpret_u64( easysimd_svfloat64_t op) { return  easysimd_svreinterpret_u64_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16(    easysimd_svint8_t op) { return   easysimd_svreinterpret_f16_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16(   easysimd_svint16_t op) { return  easysimd_svreinterpret_f16_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16(   easysimd_svint32_t op) { return  easysimd_svreinterpret_f16_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16(   easysimd_svint64_t op) { return  easysimd_svreinterpret_f16_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16(   easysimd_svuint8_t op) { return   easysimd_svreinterpret_f16_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16(  easysimd_svuint16_t op) { return  easysimd_svreinterpret_f16_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16(  easysimd_svuint32_t op) { return  easysimd_svreinterpret_f16_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16(  easysimd_svuint64_t op) { return  easysimd_svreinterpret_f16_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16( easysimd_svfloat32_t op) { return  easysimd_svreinterpret_f16_f32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat16_t  easysimd_svreinterpret_f16( easysimd_svfloat64_t op) { return  easysimd_svreinterpret_f16_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32(    easysimd_svint8_t op) { return   easysimd_svreinterpret_f32_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32(   easysimd_svint16_t op) { return  easysimd_svreinterpret_f32_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32(   easysimd_svint32_t op) { return  easysimd_svreinterpret_f32_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32(   easysimd_svint64_t op) { return  easysimd_svreinterpret_f32_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32(   easysimd_svuint8_t op) { return   easysimd_svreinterpret_f32_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32(  easysimd_svuint16_t op) { return  easysimd_svreinterpret_f32_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32(  easysimd_svuint32_t op) { return  easysimd_svreinterpret_f32_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32(  easysimd_svuint64_t op) { return  easysimd_svreinterpret_f32_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32( easysimd_svfloat16_t op) { return  easysimd_svreinterpret_f32_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat32_t  easysimd_svreinterpret_f32( easysimd_svfloat64_t op) { return  easysimd_svreinterpret_f32_f64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64(    easysimd_svint8_t op) { return   easysimd_svreinterpret_f64_s8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64(   easysimd_svint16_t op) { return  easysimd_svreinterpret_f64_s16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64(   easysimd_svint32_t op) { return  easysimd_svreinterpret_f64_s32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64(   easysimd_svint64_t op) { return  easysimd_svreinterpret_f64_s64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64(   easysimd_svuint8_t op) { return   easysimd_svreinterpret_f64_u8(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64(  easysimd_svuint16_t op) { return  easysimd_svreinterpret_f64_u16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64(  easysimd_svuint32_t op) { return  easysimd_svreinterpret_f64_u32(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64(  easysimd_svuint64_t op) { return  easysimd_svreinterpret_f64_u64(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64( easysimd_svfloat16_t op) { return  easysimd_svreinterpret_f64_f16(op); }
  EASYSIMD_FUNCTION_ATTRIBUTES  easysimd_svfloat64_t  easysimd_svreinterpret_f64( easysimd_svfloat32_t op) { return  easysimd_svreinterpret_f64_f32(op); }

  #if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,  svfloat32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_s8,     svint8_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,  svfloat32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s16,    svint16_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,  svfloat32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s32,    svint32_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,  svfloat32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_s64,    svint64_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,  svfloat32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_(  svreinterpret_u8,    svuint8_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,  svfloat32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u16,   svuint16_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,  svfloat32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u32,   svuint32_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,  svfloat32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_u64,   svuint64_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,  svfloat32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f16,  svfloat16_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f32,  svfloat32_t,  svfloat64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,     svint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,    svint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,    svint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,    svint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,    svuint8_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,   svuint16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,   svuint32_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,   svuint64_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,  svfloat16_t)
    EASYSIMD_DEFINE_CONVERSION_FUNCTION_( svreinterpret_f64,  svfloat64_t,  svfloat32_t)
  #endif /* defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES) */
#elif defined(EASYSIMD_GENERIC_)
  #define easysimd_svreinterpret_f64(op) \
    (_Generic((op), \
        easysimd_svint16_t: easysimd_svreinterpret_s8_s16, \
        easysimd_svint32_t: easysimd_svreinterpret_s8_s32, \
        easysimd_svint64_t: easysimd_svreinterpret_s8_s64, \
        easysimd_svuint8_t: easysimd_svreinterpret_s8_u8, \
       easysimd_svuint16_t: easysimd_svreinterpret_s8_u16, \
       easysimd_svuint32_t: easysimd_svreinterpret_s8_u32, \
       easysimd_svuint64_t: easysimd_svreinterpret_s8_u64, \
      easysimd_svfloat16_t: easysimd_svreinterpret_s8_f16, \
      easysimd_svfloat32_t: easysimd_svreinterpret_s8_f32, \
      easysimd_svfloat64_t: easysimd_svreinterpret_s8_f64)(op))
  #define easysimd_svreinterpret_s8(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_s16_s8, \
        easysimd_svint32_t: easysimd_svreinterpret_s16_s32, \
        easysimd_svint64_t: easysimd_svreinterpret_s16_s64, \
        easysimd_svuint8_t: easysimd_svreinterpret_s16_u8, \
       easysimd_svuint16_t: easysimd_svreinterpret_s16_u16, \
       easysimd_svuint32_t: easysimd_svreinterpret_s16_u32, \
       easysimd_svuint64_t: easysimd_svreinterpret_s16_u64, \
      easysimd_svfloat16_t: easysimd_svreinterpret_s16_f16, \
      easysimd_svfloat32_t: easysimd_svreinterpret_s16_f32, \
      easysimd_svfloat64_t: easysimd_svreinterpret_s16_f64)(op))
  #define easysimd_svreinterpret_s16(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_s32_s8, \
        easysimd_svint16_t: easysimd_svreinterpret_s32_s16, \
        easysimd_svint64_t: easysimd_svreinterpret_s32_s64, \
        easysimd_svuint8_t: easysimd_svreinterpret_s32_u8, \
       easysimd_svuint16_t: easysimd_svreinterpret_s32_u16, \
       easysimd_svuint32_t: easysimd_svreinterpret_s32_u32, \
       easysimd_svuint64_t: easysimd_svreinterpret_s32_u64, \
      easysimd_svfloat16_t: easysimd_svreinterpret_s32_f16, \
      easysimd_svfloat32_t: easysimd_svreinterpret_s32_f32, \
      easysimd_svfloat64_t: easysimd_svreinterpret_s32_f64)(op))
  #define easysimd_svreinterpret_s32(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_s64_s8, \
        easysimd_svint16_t: easysimd_svreinterpret_s64_s16, \
        easysimd_svint32_t: easysimd_svreinterpret_s64_s32, \
        easysimd_svuint8_t: easysimd_svreinterpret_s64_u8, \
       easysimd_svuint16_t: easysimd_svreinterpret_s64_u16, \
       easysimd_svuint32_t: easysimd_svreinterpret_s64_u32, \
       easysimd_svuint64_t: easysimd_svreinterpret_s64_u64, \
      easysimd_svfloat16_t: easysimd_svreinterpret_s64_f16, \
      easysimd_svfloat32_t: easysimd_svreinterpret_s64_f32, \
      easysimd_svfloat64_t: easysimd_svreinterpret_s64_f64)(op))
  #define easysimd_svreinterpret_s64(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_u8_s8, \
        easysimd_svint16_t: easysimd_svreinterpret_u8_s16, \
        easysimd_svint32_t: easysimd_svreinterpret_u8_s32, \
        easysimd_svint64_t: easysimd_svreinterpret_u8_s64, \
       easysimd_svuint16_t: easysimd_svreinterpret_u8_u16, \
       easysimd_svuint32_t: easysimd_svreinterpret_u8_u32, \
       easysimd_svuint64_t: easysimd_svreinterpret_u8_u64, \
      easysimd_svfloat16_t: easysimd_svreinterpret_u8_f16, \
      easysimd_svfloat32_t: easysimd_svreinterpret_u8_f32, \
      easysimd_svfloat64_t: easysimd_svreinterpret_u8_f64)(op))
  #define easysimd_svreinterpret_u8(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_u16_s8, \
        easysimd_svint16_t: easysimd_svreinterpret_u16_s16, \
        easysimd_svint32_t: easysimd_svreinterpret_u16_s32, \
        easysimd_svint64_t: easysimd_svreinterpret_u16_s64, \
        easysimd_svuint8_t: easysimd_svreinterpret_u16_u8, \
       easysimd_svuint32_t: easysimd_svreinterpret_u16_u32, \
       easysimd_svuint64_t: easysimd_svreinterpret_u16_u64, \
      easysimd_svfloat16_t: easysimd_svreinterpret_u16_f16, \
      easysimd_svfloat32_t: easysimd_svreinterpret_u16_f32, \
      easysimd_svfloat64_t: easysimd_svreinterpret_u16_f64)(op))
  #define easysimd_svreinterpret_u16(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_u32_s8, \
        easysimd_svint16_t: easysimd_svreinterpret_u32_s16, \
        easysimd_svint32_t: easysimd_svreinterpret_u32_s32, \
        easysimd_svint64_t: easysimd_svreinterpret_u32_s64, \
        easysimd_svuint8_t: easysimd_svreinterpret_u32_u8, \
       easysimd_svuint16_t: easysimd_svreinterpret_u32_u16, \
       easysimd_svuint64_t: easysimd_svreinterpret_u32_u64, \
      easysimd_svfloat16_t: easysimd_svreinterpret_u32_f16, \
      easysimd_svfloat32_t: easysimd_svreinterpret_u32_f32, \
      easysimd_svfloat64_t: easysimd_svreinterpret_u32_f64)(op))
  #define easysimd_svreinterpret_u32(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_u64_s8, \
        easysimd_svint16_t: easysimd_svreinterpret_u64_s16, \
        easysimd_svint32_t: easysimd_svreinterpret_u64_s32, \
        easysimd_svint64_t: easysimd_svreinterpret_u64_s64, \
        easysimd_svuint8_t: easysimd_svreinterpret_u64_u8, \
       easysimd_svuint16_t: easysimd_svreinterpret_u64_u16, \
       easysimd_svuint32_t: easysimd_svreinterpret_u64_u32, \
      easysimd_svfloat16_t: easysimd_svreinterpret_u64_f16, \
      easysimd_svfloat32_t: easysimd_svreinterpret_u64_f32, \
      easysimd_svfloat64_t: easysimd_svreinterpret_u64_f64)(op))
  #define easysimd_svreinterpret_u64(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_f16_s8, \
        easysimd_svint16_t: easysimd_svreinterpret_f16_s16, \
        easysimd_svint32_t: easysimd_svreinterpret_f16_s32, \
        easysimd_svint64_t: easysimd_svreinterpret_f16_s64, \
        easysimd_svuint8_t: easysimd_svreinterpret_f16_u8, \
       easysimd_svuint16_t: easysimd_svreinterpret_f16_u16, \
       easysimd_svuint32_t: easysimd_svreinterpret_f16_u32, \
       easysimd_svuint64_t: easysimd_svreinterpret_f16_u64, \
      easysimd_svfloat32_t: easysimd_svreinterpret_f16_f32, \
      easysimd_svfloat64_t: easysimd_svreinterpret_f16_f64)(op))
  #define easysimd_svreinterpret_f16(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_f32_s8, \
        easysimd_svint16_t: easysimd_svreinterpret_f32_s16, \
        easysimd_svint32_t: easysimd_svreinterpret_f32_s32, \
        easysimd_svint64_t: easysimd_svreinterpret_f32_s64, \
        easysimd_svuint8_t: easysimd_svreinterpret_f32_u8, \
       easysimd_svuint16_t: easysimd_svreinterpret_f32_u16, \
       easysimd_svuint32_t: easysimd_svreinterpret_f32_u32, \
       easysimd_svuint64_t: easysimd_svreinterpret_f32_u64, \
      easysimd_svfloat16_t: easysimd_svreinterpret_f32_f16, \
      easysimd_svfloat64_t: easysimd_svreinterpret_f32_f64)(op))
  #define easysimd_svreinterpret_f32(op) \
    (_Generic((op), \
         easysimd_svint8_t: easysimd_svreinterpret_f64_s8, \
        easysimd_svint16_t: easysimd_svreinterpret_f64_s16, \
        easysimd_svint32_t: easysimd_svreinterpret_f64_s32, \
        easysimd_svint64_t: easysimd_svreinterpret_f64_s64, \
        easysimd_svuint8_t: easysimd_svreinterpret_f64_u8, \
       easysimd_svuint16_t: easysimd_svreinterpret_f64_u16, \
       easysimd_svuint32_t: easysimd_svreinterpret_f64_u32, \
       easysimd_svuint64_t: easysimd_svreinterpret_f64_u64, \
      easysimd_svfloat16_t: easysimd_svreinterpret_f64_f16, \
      easysimd_svfloat32_t: easysimd_svreinterpret_f64_f32)(op))
  #if defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES)
  #define svreinterpret_f64(op) \
    (_Generic((op), \
              svint16_t: svreinterpret_s8_s16, \
              svint32_t: svreinterpret_s8_s32, \
              svint64_t: svreinterpret_s8_s64, \
              svuint8_t: svreinterpret_s8_u8, \
             svuint16_t: svreinterpret_s8_u16, \
             svuint32_t: svreinterpret_s8_u32, \
             svuint64_t: svreinterpret_s8_u64, \
            svfloat16_t: svreinterpret_s8_f16, \
            svfloat32_t: svreinterpret_s8_f32, \
            svfloat64_t: svreinterpret_s8_f64)(op))
  #define svreinterpret_s8(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_s16_s8, \
              svint32_t: svreinterpret_s16_s32, \
              svint64_t: svreinterpret_s16_s64, \
              svuint8_t: svreinterpret_s16_u8, \
             svuint16_t: svreinterpret_s16_u16, \
             svuint32_t: svreinterpret_s16_u32, \
             svuint64_t: svreinterpret_s16_u64, \
            svfloat16_t: svreinterpret_s16_f16, \
            svfloat32_t: svreinterpret_s16_f32, \
            svfloat64_t: svreinterpret_s16_f64)(op))
  #define svreinterpret_s16(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_s32_s8, \
              svint16_t: svreinterpret_s32_s16, \
              svint64_t: svreinterpret_s32_s64, \
              svuint8_t: svreinterpret_s32_u8, \
             svuint16_t: svreinterpret_s32_u16, \
             svuint32_t: svreinterpret_s32_u32, \
             svuint64_t: svreinterpret_s32_u64, \
            svfloat16_t: svreinterpret_s32_f16, \
            svfloat32_t: svreinterpret_s32_f32, \
            svfloat64_t: svreinterpret_s32_f64)(op))
  #define svreinterpret_s32(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_s64_s8, \
              svint16_t: svreinterpret_s64_s16, \
              svint32_t: svreinterpret_s64_s32, \
              svuint8_t: svreinterpret_s64_u8, \
             svuint16_t: svreinterpret_s64_u16, \
             svuint32_t: svreinterpret_s64_u32, \
             svuint64_t: svreinterpret_s64_u64, \
            svfloat16_t: svreinterpret_s64_f16, \
            svfloat32_t: svreinterpret_s64_f32, \
            svfloat64_t: svreinterpret_s64_f64)(op))
  #define svreinterpret_s64(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_u8_s8, \
              svint16_t: svreinterpret_u8_s16, \
              svint32_t: svreinterpret_u8_s32, \
              svint64_t: svreinterpret_u8_s64, \
             svuint16_t: svreinterpret_u8_u16, \
             svuint32_t: svreinterpret_u8_u32, \
             svuint64_t: svreinterpret_u8_u64, \
            svfloat16_t: svreinterpret_u8_f16, \
            svfloat32_t: svreinterpret_u8_f32, \
            svfloat64_t: svreinterpret_u8_f64)(op))
  #define svreinterpret_u8(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_u16_s8, \
              svint16_t: svreinterpret_u16_s16, \
              svint32_t: svreinterpret_u16_s32, \
              svint64_t: svreinterpret_u16_s64, \
              svuint8_t: svreinterpret_u16_u8, \
             svuint32_t: svreinterpret_u16_u32, \
             svuint64_t: svreinterpret_u16_u64, \
            svfloat16_t: svreinterpret_u16_f16, \
            svfloat32_t: svreinterpret_u16_f32, \
            svfloat64_t: svreinterpret_u16_f64)(op))
  #define svreinterpret_u16(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_u32_s8, \
              svint16_t: svreinterpret_u32_s16, \
              svint32_t: svreinterpret_u32_s32, \
              svint64_t: svreinterpret_u32_s64, \
              svuint8_t: svreinterpret_u32_u8, \
             svuint16_t: svreinterpret_u32_u16, \
             svuint64_t: svreinterpret_u32_u64, \
            svfloat16_t: svreinterpret_u32_f16, \
            svfloat32_t: svreinterpret_u32_f32, \
            svfloat64_t: svreinterpret_u32_f64)(op))
  #define svreinterpret_u32(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_u64_s8, \
              svint16_t: svreinterpret_u64_s16, \
              svint32_t: svreinterpret_u64_s32, \
              svint64_t: svreinterpret_u64_s64, \
              svuint8_t: svreinterpret_u64_u8, \
             svuint16_t: svreinterpret_u64_u16, \
             svuint32_t: svreinterpret_u64_u32, \
            svfloat16_t: svreinterpret_u64_f16, \
            svfloat32_t: svreinterpret_u64_f32, \
            svfloat64_t: svreinterpret_u64_f64)(op))
  #define svreinterpret_u64(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_f16_s8, \
              svint16_t: svreinterpret_f16_s16, \
              svint32_t: svreinterpret_f16_s32, \
              svint64_t: svreinterpret_f16_s64, \
              svuint8_t: svreinterpret_f16_u8, \
             svuint16_t: svreinterpret_f16_u16, \
             svuint32_t: svreinterpret_f16_u32, \
             svuint64_t: svreinterpret_f16_u64, \
            svfloat32_t: svreinterpret_f16_f32, \
            svfloat64_t: svreinterpret_f16_f64)(op))
  #define svreinterpret_f16(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_f32_s8, \
              svint16_t: svreinterpret_f32_s16, \
              svint32_t: svreinterpret_f32_s32, \
              svint64_t: svreinterpret_f32_s64, \
              svuint8_t: svreinterpret_f32_u8, \
             svuint16_t: svreinterpret_f32_u16, \
             svuint32_t: svreinterpret_f32_u32, \
             svuint64_t: svreinterpret_f32_u64, \
            svfloat16_t: svreinterpret_f32_f16, \
            svfloat64_t: svreinterpret_f32_f64)(op))
  #define svreinterpret_f32(op) \
    (_Generic((op), \
               svint8_t: svreinterpret_f64_s8, \
              svint16_t: svreinterpret_f64_s16, \
              svint32_t: svreinterpret_f64_s32, \
              svint64_t: svreinterpret_f64_s64, \
              svuint8_t: svreinterpret_f64_u8, \
             svuint16_t: svreinterpret_f64_u16, \
             svuint32_t: svreinterpret_f64_u32, \
             svuint64_t: svreinterpret_f64_u64, \
            svfloat16_t: svreinterpret_f64_f16, \
            svfloat32_t: svreinterpret_f64_f32)(op))
  #endif /* defined(EASYSIMD_ARM_SVE_ENABLE_NATIVE_ALIASES) */
#endif


#if defined (NATIVE_ARM_SVE_NATIVE)
  #define easysimd_svreinterpret_f32_s32(op)    svreinterpret_f32_s32((op))
#endif
HEDLEY_DIAGNOSTIC_POP

#endif /* EASYSIMD_ARM_SVE_REINTERPRET_H */
