#include "LLR.h"
#include "mat.h"
#include "param.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*local gather operations, try to work on contiguous memory*/
inline void
csplit_submatrix(dcomplex *chunk, dcomplex *v, param *p)
{
  int i, j, m, n;

  for(i = 0; i < p->nx; i++){
    for(j = 0; j < p->ny; j++){
      for(m = 0; m < p->nb; m++){
        for(n = 0; n < p->nb; n++){
          memcpy((void*)(v + i * p->ny * p->blk_size + j * p->blk_size + m * p->nb * p->nc *
            p->nt + n * p->nc * p->nt), (void*)(chunk + (i + m) * p->ny * p->nc * p->nt +
            ((j + n) % p->ny) * p->nc * p->nt), p->nc * p->nt * sizeof(dcomplex));
        }
      }
    }
  }
}

/*******************************************************************************
*                        sum_submatrix
*******************************************************************************/
inline void
csum_submatrix(dcomplex *v, dcomplex *chunk, param *p)
{
  int i, j, m, n;

  for(i = 0; i < p->nx; i++){
    for(j = 0; j < p->ny; j++){
      for(m = 0; m < p->nb; m++){
        for(n = 0; n < p->nb; n++){
          caxpy(1, v + i * p->ny * p->blk_size + j * p->blk_size + m * p->nb * p->nc * p->nt + n *
            p->nc * p->nt, chunk + (i + m) * p->ny * p->nc * p->nt + ((j + n) % p->ny) * p->nc * p->nt,
            p->nc * p->nt);
        }
      }
    }
  }
}

/*******************************************************************************
*    u = u + mu*(v - z)
*******************************************************************************/
inline void
cgrad_update_submatrix(dcomplex *u, dcomplex *grad, int size, double mu)
{
  caxpy(-mu, grad, u, size);
}

inline void
cblk_prox_update_submatrix(dcomplex *u, int nx, int ny, double coef)
{
  ct_svd(u, u, nx, ny, coef);
}

inline void
cblk_update_submatrix(dcomplex *u, dcomplex *grad, int nx, int ny, int size,
  double coef, double mu)
{
  cgrad_update_submatrix(u, grad, size, mu);
  cblk_prox_update_submatrix(u, nx, ny, coef);
}

inline void
cupdate_submatrix(dcomplex *in_array, dcomplex *u_array, dcomplex *out_array,
  dcomplex *grad_array, param *p)
{
  int i;

  csub(out_array, in_array, out_array, p->rsize);
  csplit_submatrix(out_array, grad_array, p);

  for(i=0; i<p->nblk; i++){
    cblk_update_submatrix(u_array+i*p->blk_size, grad_array+i*p->blk_size,
      p->blk_nx, p->blk_ny, p->blk_size, p->coef, p->mu);
  }

  memset((void*)out_array, 0, p->rsize*sizeof(dcomplex));
  csum_submatrix(u_array, out_array, p);

}

inline void
cupdate_mpi(dcomplex *in_array, dcomplex *u_array, dcomplex *out_array,
  dcomplex *grad_array, dcomplex *buffer, param *p)
{
  cupdate_submatrix(in_array, u_array, out_array, grad_array, p);

  MPI_Sendrecv(out_array, p->buf_size*2, MPI_DOUBLE, p->left_neighbor, 1, buffer,
    p->buf_size*2, MPI_DOUBLE, p->right_neighbor, 1, MPI_COMM_WORLD,
    &(p->status));

  cadd(out_array+p->chunk_size, buffer, out_array+p->chunk_size, p->buf_size);

  MPI_Sendrecv(out_array+p->chunk_size, p->buf_size*2, MPI_DOUBLE,
    p->right_neighbor, 1, out_array, p->buf_size*2, MPI_DOUBLE, p->left_neighbor,
    1, MPI_COMM_WORLD, &(p->status));

}

inline void
cextract_submatrix_mpi(dcomplex *in_array, dcomplex *chunk, param *p)
{

  MPI_Scatter(in_array+p->offset_size, p->base_size*2, MPI_DOUBLE, chunk
    + p->offset_size, p->base_size*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  if(p->procid == 0){
    memcpy(chunk, in_array, p->offset_size*sizeof(dcomplex));
  }

  MPI_Sendrecv(chunk, p->buf_size*2, MPI_DOUBLE, p->left_neighbor, 1,
    chunk+p->chunk_size, p->buf_size*2, MPI_DOUBLE, p->right_neighbor,
    1, MPI_COMM_WORLD, &(p->status));
  MPI_Barrier(MPI_COMM_WORLD);

}

inline void
cget_primal_mpi(dcomplex *ichunk, dcomplex *ochunk, dcomplex *pchunk, param *p)
{
  csub(ichunk, ochunk, pchunk, p->chunk_size);
}

inline void
cgather_submatrix_mpi(dcomplex *ochunk, dcomplex *out_array, param *p)
{
  MPI_Gather(ochunk+p->offset_size, p->base_size*2, MPI_DOUBLE,
    out_array+p->offset_size, p->base_size*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(p->procid == 0){
    memcpy(ochunk, out_array, p->offset_size*sizeof(dcomplex));
  }
}

void
cdenoise_mpi(dcomplex *in_array, dcomplex *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int max_iter, double mu)
{
  int i;
  dcomplex *ichunk, *ochunk, *pchunk, *buffer, *u, *grad;
  param p;
  getparam(&p, nx, ny, nc, nt, nb, coef, max_iter, mu);

  printf("size of complex: %lu\n", sizeof(dcomplex));

  ichunk = (dcomplex*)malloc(p.rsize * sizeof(dcomplex));
  ochunk = (dcomplex*)malloc(p.rsize * sizeof(dcomplex));
  pchunk = (dcomplex*)malloc(p.rsize * sizeof(dcomplex));
  buffer = (dcomplex*)malloc(p.buf_size * sizeof(dcomplex));
  u = (dcomplex*)malloc(p.nb * p.nb * p.chunk_size * sizeof(dcomplex));
  grad = (dcomplex*)malloc(p.nb * p.nb * p.chunk_size * sizeof(dcomplex));

  cextract_submatrix_mpi(in_array, ichunk, &p);
  memset((void*)ochunk, 0, p.rsize * sizeof(dcomplex));
  memset((void*)u, 0, p.nb * p.nb * p.chunk_size * sizeof(dcomplex));

  for(i=0; i<p.max_iter; i++){
    /*proximal gradient update*/
    cupdate_mpi(ichunk, u, ochunk, grad, buffer, &p);
    /*gather all parts of output*/
    cget_primal_mpi(ichunk, ochunk, pchunk, &p);
    cgather_submatrix_mpi(pchunk, out_array, &p);
    if(p.procid == 0){
      printf("Iteration: %d, Objective: %lf\n", i + 1, cnrm2(out_array, p.full_size));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  free(ichunk);
  free(ochunk);
  free(pchunk);
  free(buffer);
  free(u);
  free(grad);
}
