#include "param.h"
#include <stdio.h>

void
compute_param(param *p, int nx, int ny, int nc, int nt, int nb, double coef,
  int max_iter, double mu)
{
  /* rows of a chunk */
  p->nx = nx;
  /* columns of a chunk */
  p->ny = ny;
  p->nc = nc;
  p->nt = nt;
  p->nb = nb;
  p->nx2d = p->nx;
  p->ny2d = ny*nc*nt;
  p->rnx = p->nx+nb-1;
  p->rsize = p->rnx*p->ny2d;
  p->buf_size = (nb-1)*ny*nc*nt;
  p->chunk_size = p->nx2d*p->ny2d;
  p->x_blk_size = ny*nb*nb*nc*nt;
  p->usize = p->nx*p->x_blk_size;
  p->blk_nx = nb*nb*nc;
  p->blk_ny = nt;
  p->blk_size = p->blk_nx*p->blk_ny;
  p->nblk = p->nx*p->ny;
  p->max_iter = max_iter;
  p->coef = coef;
  p->mu = mu;
}

void
getparam(param *p, int nx, int ny, int nc, int nt, int nb, double coef,
  int max_iter, double mu)
{
  int procid;
  MPI_Comm_size(MPI_COMM_WORLD, &(p->nprocs));
  MPI_Comm_rank(MPI_COMM_WORLD, &(procid));

  if(procid == 0) {
    p->full_size = nx*ny*nc*nt;
    p->nbase = nx/p->nprocs;
    compute_param(p, p->nbase, ny, nc, nt, nb, coef, max_iter, mu);
  }
  MPI_Bcast(p, sizeof(param), MPI_BYTE, 0, MPI_COMM_WORLD);
  p->offset = 0;
  if(procid == 0) {
    compute_param(p, nx/p->nprocs+nx%p->nprocs, ny, nc, nt, nb, coef, max_iter,
      mu);
    p->offset = nx%p->nprocs;
  }
  p->procid = procid;
  p->base_size = p->nbase*p->ny2d;
  p->offset_size = p->offset*p->ny2d;
  p->left_neighbor = (p->procid==0)?p->nprocs-1:p->procid-1;
  p->right_neighbor = (p->procid==(p->nprocs-1))?0:p->procid+1;
  p->s = 1;
}
