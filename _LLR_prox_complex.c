/*A python wrapper for LLR proximal operator*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mat.h"
#include "LLR.h"

inline void
csplit(dcomplex *in_array, dcomplex *v, int nx, int ny, int nc, int nt, int nb)
{
  int lblk = nb*nb*nc*nt;
  int l2d = nc*nt;
  int i, j, m, n;

  for(i=0; i<nx; i++){
    for(j=0; j<ny; j++){
      for(m=0; m<nb; m++){
        for(n=0; n<nb; n++){
          memcpy((void*)(v+i*ny*lblk+j*lblk+m*nb*nc*nt+n*nc*nt),
            (void*)(in_array+((i+m)%nx)*ny*l2d+((j+n)%ny)*l2d),
            sizeof(dcomplex)*l2d);
        }
      }
    }
  }
}

inline void
csum(dcomplex *v, dcomplex *out_array, int nx, int ny, int nc, int nt, int nb)
{
  int lblk = nb*nb*nc*nt;
  int l2d = nc*nt;
  int i, j, m, n;

  for(i=0; i<nx; i++){
    for(j=0; j<ny; j++){
      for(m=0; m<nb; m++){
        for(n=0; n<nb; n++){
          cadd(out_array+((i+m)%nx)*ny*l2d+((j+n)%ny)*l2d,
            v+i*ny*lblk+j*lblk+m*nb*l2d+n*l2d,
            out_array+((i+m)%nx)*ny*l2d+((j+n)%ny)*l2d, l2d);
        }
      }
    }
  }
}

/*******************************************************************************
*    u = u + mu*(v - z)
*******************************************************************************/
inline void
cblk_grad_update(dcomplex *u, dcomplex *v, dcomplex *z, int size, double mu)
{
  int i;
  for(i=0; i<size; i++){
    u[i].re += mu*(v[i].re - z[i].re);
    u[i].im += mu*(v[i].im - z[i].im);
  }
}

inline void
cblk_prox_update(dcomplex *u, int nx, int ny, double coef)
{
  ct_svd(u, u, nx, ny, coef);
}

inline void
cblk_update(dcomplex *u, dcomplex *v, dcomplex *z, int nx, int ny, int size,
  double coef, double mu)
{
  cblk_grad_update(u, v, z, size, mu);
  cblk_prox_update(u, nx, ny, coef);
}

inline void
cupdate(dcomplex *v_array, dcomplex *u_array, dcomplex *out_array, dcomplex *z_array,
  int nx, int ny, int nc, int nt, int nb, double coef, double mu)
{
  int i;
  int blk_nx = nb*nb*nc;
  int blk_ny = nt;
  int lblk = blk_nx*blk_ny;
  int size = nx*ny;

  csplit(out_array, z_array, nx, ny, nc, nt, nb);
  for(i=0; i<size; i++){
    cblk_update(u_array+i*lblk, v_array+i*lblk, z_array+i*lblk, blk_nx, blk_ny,
      lblk, coef, mu);
  }
  memset((void*)out_array, 0, sizeof(dcomplex)*nx*ny*nc*nt);
  csum(u_array, out_array, nx, ny, nc, nt, nb);
}

/*******************************************************************************
*
*  out_array:an all zero array
*
*******************************************************************************/
void
cdenoise(dcomplex *in_array, dcomplex *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int max_iter, double mu)
{
  int size = nx*ny*nc*nt;
  int i;
  //int lblk = nb*nb*nc*nt;

  dcomplex *v = (dcomplex*)malloc(sizeof(dcomplex)*nb*nb*size);
  dcomplex *u = (dcomplex*)malloc(sizeof(dcomplex)*nb*nb*size);
  dcomplex *z = (dcomplex*)malloc(sizeof(dcomplex)*nb*nb*size);
  dcomplex *c = (dcomplex*)malloc(sizeof(dcomplex)*size);

  csplit(in_array, v, nx, ny, nc, nt, nb);

  memset((void*)out_array, 0, sizeof(dcomplex)*size);
  memset((void*)u, 0, sizeof(dcomplex)*nb*nb*size);
  memset((void*)z, 0, sizeof(dcomplex)*nb*nb*size);

  for(i=0; i<max_iter; i++){
    cupdate(v, u, out_array, z, nx, ny, nc, nt, nb, coef, mu);
    csub(in_array, out_array, c, size);
    printf("Iteration: %d, Objective: %lf\n",i+1,cnrm2(c,size));
  }

  csub(in_array, out_array, out_array, size);
  free(v);
  free(u);
  free(z);
  free(c);

}
