#ifndef _LLR_H_
#define _LLR_H_

#include "complex.h"
/*
void
denoise(double *in_array, double *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int mat_iter, double mu);

void
denoise_mpi(double *in_array, double *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int mat_iter, double mu);

void
cdenoise(dcomplex *in_array, dcomplex *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int max_iter, double mu);
*/
void
cdenoise_mpi(dcomplex *in_array, dcomplex *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int max_iter, double mu);

void
cdenoise_mpi_acc(dcomplex *in_array, dcomplex *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int max_iter, double mu);

#endif
