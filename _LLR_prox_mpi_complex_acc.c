#include "LLR.h"
#include "mat.h"
#include "param.h"
#include "_LLR_prox_mpi_complex.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

inline void
cgrad_update_submatrix_acc(dcomplex *u, dcomplex *u0, dcomplex *grad, int size,
  double mu)
{
  csadd(-mu, grad, u0, u, size);
}

inline void
cblk_update_submatrix_acc(dcomplex *u, dcomplex *u0, dcomplex *grad, int nx,
  int ny, int size, double coef, double mu)
{
  cgrad_update_submatrix_acc(u, u0, grad, size, mu);
  cblk_prox_update_submatrix(u, nx, ny, coef);
}

inline void
cfista_combine_mpi_acc(dcomplex *__restrict__ u_array,
  dcomplex *__restrict__ u1_array, dcomplex *__restrict__ u0_array, param *p)
{
  const unsigned size = 2*p->usize;
  unsigned i;

  for(i=0; i<size; i++)
  {
    ((double*)u_array)[i] = ((double*)u1_array)[i] + (1-p->s)*
    (((double*)u1_array)[i] - ((double*)u0_array)[i]);
  }

  p->s = 2/(1 + sqrt(1+4/(p->s*p->s)));
}

/* at the beginning out_array = in_array - sum E*/
inline void
cupdate_submatrix_acc(dcomplex *u_array, dcomplex *u1_array, dcomplex *u0_array,
  dcomplex *out_array, dcomplex *grad_array, param *p)
{
  int i;

  csplit_submatrix(out_array, grad_array, p);

  for(i=0; i<p->nblk; i++){
    cblk_update_submatrix_acc(u1_array+i*p->blk_size, u_array+i*p->blk_size,
      grad_array+i*p->blk_size, p->blk_nx, p->blk_ny, p->blk_size, p->coef,
      p->mu);
  }

  cfista_combine_mpi_acc(u_array, u1_array, u0_array, p);
}

inline void
cupdate_mpi_acc(dcomplex *out_array, dcomplex *in_array, dcomplex *u_array,
  dcomplex *u1_array, dcomplex *u0_array, dcomplex *grad_array,
  dcomplex *buffer, param *p)
{
  //double local_value, global_value, local_tvalue, global_tvalue;
  cupdate_submatrix_acc(u_array, u1_array, u0_array, out_array, grad_array, p);

  memset((void*)out_array, 0, p->rsize*sizeof(dcomplex));
  csum_submatrix(u_array, out_array, p);


  MPI_Sendrecv(out_array, p->buf_size*2, MPI_DOUBLE, p->left_neighbor, 1, buffer,
    p->buf_size*2, MPI_DOUBLE, p->right_neighbor, 1, MPI_COMM_WORLD,
    &(p->status));

  cadd(out_array+p->chunk_size, buffer, out_array+p->chunk_size, p->buf_size);

  MPI_Sendrecv(out_array+p->chunk_size, p->buf_size*2, MPI_DOUBLE,
    p->right_neighbor, 1, out_array, p->buf_size*2, MPI_DOUBLE, p->left_neighbor,
    1, MPI_COMM_WORLD, &(p->status));

  csub(out_array, in_array, out_array, p->rsize);

}

inline double
cgetobj_mpi_acc(dcomplex *out_array, param *p)
{
  double local_value, global_value=0;
  local_value = csqnrm2(out_array, p->chunk_size);
  MPI_Reduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM, 0,
    MPI_COMM_WORLD);
  return global_value;
}

void
cdenoise_mpi_acc(dcomplex *in_array, dcomplex *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int max_iter, double mu)
{
  int i;
  double f_value;
  dcomplex *ichunk, *ochunk, *pchunk, *buffer, *u, *u1, *u0, *tmp, *grad;
  param p;
  getparam(&p, nx, ny, nc, nt, nb, coef, max_iter, mu);

  ichunk = (dcomplex*)malloc(p.rsize*sizeof(dcomplex));
  ochunk = (dcomplex*)malloc(p.rsize*sizeof(dcomplex));
  //tochunk = (dcomplex*)malloc(p.rsize*sizeof(dcomplex));
  pchunk = (dcomplex*)malloc(p.rsize*sizeof(dcomplex));
  buffer = (dcomplex*)malloc(p.buf_size*sizeof(dcomplex));
  u = (dcomplex*)malloc(p.usize*sizeof(dcomplex));
  u1 = (dcomplex*)malloc(p.usize*sizeof(dcomplex));
  u0 = (dcomplex*)malloc(p.usize*sizeof(dcomplex));
  grad = (dcomplex*)malloc(p.nb*p.nb*p.chunk_size*sizeof(dcomplex));

  cextract_submatrix_mpi(in_array, ichunk, &p);
  memset((void*)ochunk, 0, p.rsize*sizeof(dcomplex));
  memset((void*)u, 0, p.nb*p.nb*p.chunk_size*sizeof(dcomplex));
  memset((void*)u1, 0, p.nb*p.nb*p.chunk_size*sizeof(dcomplex));
  memset((void*)u0, 0, p.nb*p.nb*p.chunk_size*sizeof(dcomplex));
  csub(ochunk, ichunk, ochunk, p.rsize);

  for(i=0; i<p.max_iter; i++){
    /*swap u0 and u*/
    tmp = u1;
    u1 = u0;
    u0 = tmp;
    /*proximal gradient update*/
    cupdate_mpi_acc(ochunk, ichunk, u, u1, u0, grad, buffer, &p);
    /*gather all parts of output*/
    //cget_primal_mpi(ichunk, ochunk, pchunk, &p);
    //cgather_submatrix_mpi(ochunk, out_array, &p);
    f_value = cgetobj_mpi_acc(ochunk, &p);
    if(p.procid == 0){
      printf("Iteration: %d, Objective: %lf\n",i+1,sqrt(f_value));
    }
  }
  cgather_submatrix_mpi(ochunk, out_array, &p);
  MPI_Barrier(MPI_COMM_WORLD);
  free(ichunk);
  free(ochunk);
  free(pchunk);
  free(buffer);
  free(u);
  free(u1);
  free(u0);
  free(grad);
}
