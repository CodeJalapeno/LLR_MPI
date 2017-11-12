#ifndef _MAT_H_
#define _MAT_H_
#include "complex.h"

extern void
dscal_(int *, double *, double *, int *);

extern double
ddot_(int *, double *, int *, double *, int *);

extern double
dnrm2_(int *, double *, int*);

extern void
daxpy_(int *, double *, double *, int *, double *, int *);

extern void
zgemm_(char *, char *, int*, int*, int *, dcomplex *, dcomplex *, int *,
  dcomplex *, int *, dcomplex *, dcomplex *, int *);

extern void
dgesdd_(char *, int *, int *, double *, int *, double *, double *,
  int *, double *, int *, double *, int *, int *, int *);

extern void
zgesdd_(char *, int *, int *, dcomplex *, int *, double *, dcomplex *, int *,
  dcomplex *, int *, dcomplex *, int *, double *, int *, int *);

inline void
axpy(double a, double *x, double *y, int size);

inline void
caxpy(double a, dcomplex *x, dcomplex *y, int size);

inline void
sub(double *a, double *b, double *c, int size);

inline void
csub(dcomplex *a, dcomplex *b, dcomplex *c, int size);

inline void
add(double *a, double *b, double *c, int size);

inline void
cadd(dcomplex *a, dcomplex *b, dcomplex *c, int size);

inline void
sadd(double s, double *a, double *b, double *c, int size);

inline void
csadd(double s, dcomplex *a, dcomplex *b, dcomplex *c, int size);

inline double
nrm2(double *x, int size);

inline double
sqnrm2(double *x, int size);

inline double
cnrm2(dcomplex *x, int size);

inline double
csqnrm2(dcomplex *x, int size);

/*
void
t_svd(double *x, double *y, int Nx, int Ny, double coef);
*/
void
ct_svd(dcomplex *x, dcomplex *y, int Nx, int Ny, double coef);

#endif
