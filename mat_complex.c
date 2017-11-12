#include "mat.h"
//#include "clapack.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define min(a,b) ((a<b)?(a):(b))

/*
extern void
dgesdd_(char *jobz, int *m, int *n, dcomplex *a, int *lda, dcomplex *s, dcomplex *u,
  int *ldu, dcomplex *vt, int *ldvt, dcomplex *work, int *lwork, int *iwork,
  int *info);
*/

inline double
cnrm2(dcomplex *x, int size)
{
  //extern dcomplex cblas_dnrm2( const int N, const dcomplex *X, const int incX);
  return nrm2((double*)x, size*2);
}

inline double
csqnrm2(dcomplex *x, int size)
{
  return sqnrm2((double*)x, 2*size);
}

/*******************************************************************************
*          For further optimization
*******************************************************************************/

inline void
caxpy(double a, dcomplex *__restrict__ x, dcomplex *__restrict__ y, int size)
{
  axpy(a, (double*)x, (double*)y, 2*size);
}

inline void
csub(dcomplex *a, dcomplex *b, dcomplex *c, int size)
{
  sub((double*)a, (double*)b, (double*)c, 2*size);
}

inline void
cadd(dcomplex *a, dcomplex *b, dcomplex *c, int size)
{
  add((double*)a, (double*)b, (double*)c, 2*size);
}

inline void
csadd(double s, dcomplex *__restrict__ a, dcomplex *__restrict__ b, dcomplex *__restrict__ c, int size)
{
  sadd(s, (double*)a, (double*)b, (double*)c, 2*size);
}

/*truncted SVD of row-major matrix x*/
inline void
_ct_svd(dcomplex *x, dcomplex *y, int Nx, int Ny, double coef, dcomplex *uwork,
  dcomplex *vwork, double *swork, double *rwork, int *iwork)
{
  int i;
  double a;
  dcomplex alpha={1.0,0}, beta={0,0};
  int nsv = min(Nx,Ny);
  int n = Ny*2, incx = 1;
  int lda = Ny;

  // Workspace and status variables:
  dcomplex workSize;
  dcomplex *work = &workSize;
  int lwork = -1;
  int info = 0;

  zgesdd_("A",&Ny,&Nx,x,&lda,swork,uwork,&Ny,vwork,&Nx,work,&lwork,rwork,iwork,&info);
  if(info){
    exit(0);
  }

  // Optimal workspace size is returned in work[0].
  lwork = (int)workSize.re;
  work = malloc(lwork*sizeof(dcomplex));

  // Call dgesdd_ to do the actual computation:
  zgesdd_("A",&Ny,&Nx,x,&lda,swork,uwork,&Ny,vwork,&Nx,work,&lwork,rwork,iwork,&info);
  if (info){
    exit(0);
  }
  // Cleanup workspace:
  free(work);

  for(i=0; i<nsv; i++){
    a = min(coef,swork[i]);
    dscal_(&n, &a, (double*)(uwork+i*Ny), &incx);
  }

  //printf("Call BLAS\n");
  zgemm_("N","N",&Ny,&Nx,&nsv,&alpha,uwork,&Ny,vwork,&Nx,&beta,y,&Ny);

}

inline void
ct_svd(dcomplex *x, dcomplex *y, int Nx, int Ny, double coef)
{
  int nsv = min(Nx,Ny);

  dcomplex *u = (dcomplex*)malloc(Ny*Ny*sizeof(dcomplex));
  dcomplex *vt = (dcomplex*)malloc(Nx*Nx*sizeof(dcomplex));
  double *s = (double*)malloc(nsv*sizeof(double));
  double *rwork = (double*)malloc((5*nsv*nsv+7*nsv)*sizeof(double));
  int *iwork = malloc(8*nsv*sizeof(int));

  _ct_svd(x, y, Nx, Ny, coef, u, vt, s, rwork, iwork);

  free(rwork);
  free(iwork);
  free(u);
  free(s);
  free(vt);
}
