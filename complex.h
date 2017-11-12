#ifndef _COMPLEX_H_
#define _COMPLEX_H_

/*******************************************************************************
*this is tricky, to make sure dcomplex is the same size of an array of two
*doubles by using no memory alignment
*******************************************************************************/
struct _dcomplex { double re, im; }__attribute__((packed, aligned(1)));
typedef struct _dcomplex dcomplex;

#endif
