#ifndef _LLR_PROX_MPI_COMPLEX_H_
#define _LLR_PROX_MPI_COMPLEX_H_

inline void
csplit_submatrix(dcomplex *chunk, dcomplex *v, param *p);

inline void
csum_submatrix(dcomplex *v, dcomplex *chunk, param *p);

inline void
cblk_prox_update_submatrix(dcomplex *u, int nx, int ny, double coef);

inline void
cextract_submatrix_mpi(dcomplex *in_array, dcomplex *chunk, param *p);

inline void
cget_primal_mpi(dcomplex *ichunk, dcomplex *ochunk, dcomplex *pchunk, param *p);

inline void
cgather_submatrix_mpi(dcomplex *ochunk, dcomplex *out_array, param *p);

#endif
