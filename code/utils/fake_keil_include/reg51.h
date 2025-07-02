
#ifndef _REG51_H_
#define _REG51_H_

/* 使用标准C语法定义8051寄存器 */
#define P0   (*((volatile unsigned char *)0x80))
#define P1   (*((volatile unsigned char *)0x90))
#define P2   (*((volatile unsigned char *)0xA0))
#define P3   (*((volatile unsigned char *)0xB0))
#define PSW  (*((volatile unsigned char *)0xD0))
#define ACC  (*((volatile unsigned char *)0xE0))
#define B    (*((volatile unsigned char *)0xF0))
#define SP   (*((volatile unsigned char *)0x81))
#define DPL  (*((volatile unsigned char *)0x82))
#define DPH  (*((volatile unsigned char *)0x83))
#define PCON (*((volatile unsigned char *)0x87))
#define TCON (*((volatile unsigned char *)0x88))
#define TMOD (*((volatile unsigned char *)0x89))
#define TL0  (*((volatile unsigned char *)0x8A))
#define TL1  (*((volatile unsigned char *)0x8B))
#define TH0  (*((volatile unsigned char *)0x8C))
#define TH1  (*((volatile unsigned char *)0x8D))
#define IE   (*((volatile unsigned char *)0xA8))
#define IP   (*((volatile unsigned char *)0xB8))
#define SCON (*((volatile unsigned char *)0x98))
#define SBUF (*((volatile unsigned char *)0x99))

#endif
