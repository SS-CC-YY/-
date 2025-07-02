
#ifndef _ABSACC_H_
#define _ABSACC_H_

/* 绝对地址访问宏 */
#define CBYTE(addr) (*((volatile unsigned char *)(addr)))
#define DBYTE(addr) (*((volatile unsigned char *)(addr)))
#define XBYTE(addr) (*((volatile unsigned char *)(addr)))

#endif
