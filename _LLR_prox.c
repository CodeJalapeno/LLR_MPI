/*A python wrapper for LLR proximal operator*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mat.h"
#include "LLR.h"

inline void
split(double *in_array, double *v, int nx, int ny, int nc, int nt, int nb)
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
            sizeof(double)*l2d);
        }
      }
    }
  }
}

inline void
sum(double *v, double *out_array, int nx, int ny, int nc, int nt, int nb)
{
  int lblk = nb*nb*nc*nt;
  int l2d = nc*nt;
  int i, j, m, n;

  for(i=0; i<nx; i++){
    for(j=0; j<ny; j++){
      for(m=0; m<nb; m++){
        for(n=0; n<nb; n++){
          add(out_array+((i+m)%nx)*ny*l2d+((j+n)%ny)*l2d,
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
blk_grad_update(double *u, double *v, double *z, int size, double mu)
{
  int i;
  for(i=0; i<size; i++){
    u[i] += mu*(v[i] - z[i]);
  }
}

inline void
blk_prox_update(double *u, int nx, int ny, double coef)
{
  t_svd(u, u, nx, ny, coef);
}

inline void
blk_update(double *u, double *v, double *z, int nx, int ny, int size,
  double coef, double mu)
{
  blk_grad_update(u, v, z, size, mu);
  blk_prox_update(u, nx, ny, coef);
}

inline void
update(double *v_array, double *u_array, double *out_array, double *z_array,
  int nx, int ny, int nc, int nt, int nb, double coef, double mu)
{
  int i;
  int blk_nx = nb*nb*nc;
  int blk_ny = nt;
  int lblk = blk_nx*blk_ny;
  int size = nx*ny;

  split(out_array, z_array, nx, ny, nc, nt, nb);
  for(i=0; i<size; i++){
    blk_update(u_array+i*lblk, v_array+i*lblk, z_array+i*lblk, blk_nx, blk_ny,
      lblk, coef, mu);
  }
  memset((void*)out_array, 0, sizeof(double)*nx*ny*nc*nt);
  sum(u_array, out_array, nx, ny, nc, nt, nb);
}

/*******************************************************************************
*
*  out_array:an all zero array
*
*******************************************************************************/
void
denoise(double *in_array, double *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int max_iter, double mu)
{
  int size = nx*ny*nc*nt;
  int i;
  //int lblk = nb*nb*nc*nt;

  double *v = (double*)malloc(sizeof(double)*nb*nb*size);
  double *u = (double*)malloc(sizeof(double)*nb*nb*size);
  double *z = (double*)malloc(sizeof(double)*nb*nb*size);
  double *c = (double*)malloc(sizeof(double)*size);

  split(in_array, v, nx, ny, nc, nt, nb);
  memset((void*)u, 0, sizeof(double)*nb*nb*size);
  memset((void*)z, 0, sizeof(double)*nb*nb*size);

  for(i=0; i<max_iter; i++){
    update(v, u, out_array, z, nx, ny, nc, nt, nb, coef, mu);
    sub(in_array, out_array, c, size);
    printf("Iteration: %d, Objective: %lf\n",i+1,nrm2(c,size));
  }

  sub(in_array, out_array, out_array, size);
  free(v);
  free(u);
  free(z);
  free(c);

}
