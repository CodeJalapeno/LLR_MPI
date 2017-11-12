#ifndef _H_PARAM_
#define _H_PARAM_
#include <mpi.h>

typedef struct{
  int nx, ny, nc, nt, nb, nx2d, ny2d, full_size, nbase, offset, base_size,
    offset_size, rnx, rsize, buf_size, chunk_size, blk_nx, blk_ny, x_blk_size,
    usize, blk_size, nblk, nprocs, procid, left_neighbor, right_neighbor,
    max_iter;
  double coef, mu, s/*acceleration parameter*/;
  MPI_Status status;
} param;

void
getparam(param *p, int nx, int ny, int nc, int nt, int nb, double coef,
  int max_iter, double mu);

#endif
