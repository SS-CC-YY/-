
#ifndef _INTRINS_H_
#define _INTRINS_H_

/* 空操作 */
#define _nop_() do {} while(0)

/* 循环移位 */
static inline unsigned char _crol_(unsigned char a, unsigned char b) {
    return (a << b) | (a >> (8 - b));
}

static inline unsigned char _cror_(unsigned char a, unsigned char b) {
    return (a >> b) | (a << (8 - b));
}

#endif
