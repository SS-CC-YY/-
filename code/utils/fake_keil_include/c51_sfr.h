/* c51_sfr.h */
#ifndef C51_SFR_H
#define C51_SFR_H

/* 定义 8051 特殊寄存器宏 */
#define __sfr volatile unsigned char
#define __sbit volatile unsigned char

/* 示例寄存器定义 - 根据您的硬件手册补充 */
#define P0ASF   (* (__sfr *) 0x9D)
#define P1ASF   (* (__sfr *) 0x9D)  // 根据实际地址修改
#define P2ASF   (* (__sfr *) 0x9D)
// 添加其他需要的寄存器...

#endif