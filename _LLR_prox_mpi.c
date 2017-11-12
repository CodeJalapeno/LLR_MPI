#include "LLR.h"
#include "mat.h"
#include "param.h"
#include "_LLR_prox_mpi_complex.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*local gather operations, try to work on contiguous memory*/
inline void
split_submatrix(double *chunk, double *v, param *p)
{
  int i, j, m, n;

  for(i=0; i<p->nx; i++){
    for(j=0; j<p->ny; j++){
      for(m=0; m<p->nb; m++){
        for(n=0; n<p->nb; n++){
          memcpy((void*)(v+i*p->ny*p->blk_size+j*p->blk_size+m*p->nb*p->nc*
            p->nt+n*p->nc*p->nt), (void*)(chunk+(i+m)*p->ny*p->nc*p->nt+
            ((j+n)%p->ny)*p->nc*p->nt), p->nc*p->nt*sizeof(double));
        }
      }
    }
  }
}

/*******************************************************************************
*                        sum_submatrix
*******************************************************************************/
inline void
sum_submatrix(double *v, double *chunk, param *p)
{
  int i, j, m, n;

  for(i=0; i<p->nx; i++){
    for(j=0; j<p->ny; j++){
      for(m=0; m<p->nb; m++){
        for(n=0; n<p->nb; n++){
          add(chunk+(i+m)*p->ny*p->nc*p->nt+((j+n)%p->ny)*p->nc*p->nt,
            v+i*p->ny*p->blk_size+j*p->blk_size+m*p->nb*p->nc*p->nt+n*p->nc*
            p->nt,chunk+(i+m)*p->ny*p->nc*p->nt+((j+n)%p->ny)*p->nc*p->nt,
            p->nc*p->nt);
        }
      }
    }
  }
}

/*******************************************************************************
*    u = u + mu*(v - z)
*******************************************************************************/
inline void
grad_update_submatrix(double *u, double *grad, int size, double mu)
{
  axpy(-mu, grad, u, size);
}

inline void
blk_prox_update_submatrix(double *u, int nx, int ny, double coef)
{
  t_svd(u, u, nx, ny, coef);
}

inline void
blk_update_submatrix(double *u, double *grad, int nx, int ny, int size,
  double coef, double mu)
{
  grad_update_submatrix(u, grad, size, mu);
  blk_prox_update_submatrix(u, nx, ny, coef);
}

inline void
update_submatrix(double *in_array, double *u_array, double *out_array,
  double *grad_array, param *p)
{
  int i;

  sub(out_array, in_array, out_array, p->rsize);
  split_submatrix(out_array, grad_array, p);

  for(i=0; i<p->nblk; i++){
    blk_update_submatrix(u_array+i*p->blk_size, grad_array+i*p->blk_size,
      p->blk_nx, p->blk_ny, p->blk_size, p->coef, p->mu);
  }

  memset((void*)out_array, 0, p->rsize*sizeof(double));
  sum_submatrix(u_array, out_array, p);

}

inline void
update_mpi(double *in_array, double *u_array, double *out_array,
  double *grad_array, double *buffer, param *p)
{
  update_submatrix(in_array, u_array, out_array, grad_array, p);

  MPI_Sendrecv(out_array, p->buf_size, MPI_DOUBLE, p->left_neighbor, 1, buffer,
    p->buf_size, MPI_DOUBLE, p->right_neighbor, 1, MPI_COMM_WORLD,
    &(p->status));

  add(out_array+p->chunk_size, buffer, out_array+p->chunk_size, p->buf_size);

  MPI_Sendrecv(out_array+p->chunk_size, p->buf_size, MPI_DOUBLE,
    p->right_neighbor, 1, out_array, p->buf_size, MPI_DOUBLE, p->left_neighbor,
    1, MPI_COMM_WORLD, &(p->status));

}

inline void
extract_submatrix_mpi(double *in_array, double *chunk, param *p)
{

  MPI_Scatter(in_array+p->offset_size, p->base_size, MPI_DOUBLE, chunk
    + p->offset_size, p->base_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  if(p->procid == 0){
    memcpy(chunk, in_array, p->offset_size*sizeof(double));
  }

  MPI_Sendrecv(chunk, p->buf_size, MPI_DOUBLE, p->left_neighbor, 1,
    chunk+p->chunk_size, p->buf_size, MPI_DOUBLE, p->right_neighbor,
    1, MPI_COMM_WORLD, &(p->status));
  MPI_Barrier(MPI_COMM_WORLD);

}

inline void
get_primal_mpi(double *ichunk, double *ochunk, double *pchunk, param *p)
{
  sub(ichunk, ochunk, pchunk, p->chunk_size);
}

inline void
gather_submatrix_mpi(double *ochunk, double *out_array, param *p)
{
  MPI_Gather(ochunk+p->offset_size, p->base_size, MPI_DOUBLE,
    out_array+p->offset_size, p->base_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(p->procid == 0){
    memcpy(ochunk, out_array, p->offset_size*sizeof(double));
  }
}

void
denoise_mpi(double *in_array, double *out_array, int nx, int ny, int nc, int nt,
  int nb, double coef, int max_iter, double mu)
{
  int i;
  double *ichunk, *ochunk, *pchunk, *buffer, *u, *grad;
  param p;
  getparam(&p, nx, ny, nc, nt, nb, coef, max_iter, mu);

  ichunk = (double*)malloc(p.rsize*sizeof(double));
  ochunk = (double*)malloc(p.rsize*sizeof(double));
  pchunk = (double*)malloc(p.rsize*sizeof(double));
  buffer = (double*)malloc(p.buf_size*sizeof(double));
  u = (double*)malloc(p.nb*p.nb*p.chunk_size*sizeof(double));
  grad = (double*)malloc(p.nb*p.nb*p.chunk_size*sizeof(double));

  extract_submatrix_mpi(in_array, ichunk, &p);
  memset((void*)ochunk, 0, p.rsize*sizeof(double));
  memset((void*)u, 0, p.nb*p.nb*p.chunk_size*sizeof(double));

  for(i=0; i<p.max_iter; i++){
    /*proximal gradient update*/
    update_mpi(ichunk, u, ochunk, grad, buffer, &p);
    /*gather all parts of output*/
    get_primal_mpi(ichunk, ochunk, pchunk, &p);
    gather_submatrix_mpi(pchunk, out_array, &p);
    if(p.procid == 0){
      printf("Iteration: %d, Objective: %lf\n",i+1,nrm2(out_array,p.full_size));
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
