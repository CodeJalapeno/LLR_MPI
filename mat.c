#include "mat.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define min(a,b) ((a<b)?(a):(b))

inline double
nrm2(double *x, int size)
{
  //extern double cblas_dnrm2( const int N, const double *X, const int incX);
  int inca=1;
  return dnrm2_(&size, x, &inca);
}

inline double
sqnrm2(double *x, int size)
{
  int incx=1;
  return ddot_(&size, x, &incx, x, &incx);
}

/*******************************************************************************
*          For further optimization
*******************************************************************************/

inline void
axpy(double a, double *x, double *y, int size)
{
  int inca = 1;
  daxpy_(&size, &a, x, &inca, y, &inca);
}

inline void
sub(double *a, double *b, double *c, int size)
{
  int i;
  for(i=0; i< size; i++){
    c[i] = a[i] - b[i];
  }
}

inline void
add(double *a, double *b, double *c, int size)
{
  int i;
  for(i=0; i< size; i++){
    c[i] = a[i] + b[i];
  }
}

inline void
sadd(double s, double *__restrict__ a, double *__restrict__ b, double *__restrict__ c, int size)
{
  int i;
  for(i=0; i<size; i++)
  {
    c[i] = s*a[i] + b[i];
  }
}


/*truncted SVD of row-major matrix x*/
/*
void
t_svd(double *x, double *y, int Nx, int Ny, double coef)
{
  int i;
  int nsv = min(Nx,Ny);
  int lda = Ny;

  double *u = (double*)malloc(Ny*Ny*sizeof(double));
  double *vt = (double*)malloc(Nx*Nx*sizeof(double));
  double *s = (double*)malloc(nsv*sizeof(double));

  // Workspace and status variables:
  double workSize;
  double *work = &workSize;
  int lwork = -1;
  int *iwork = malloc(8*nsv*sizeof(int));
  int info = 0;

  dgesdd_("A",&Ny,&Nx,x,&lda,s,u,&Ny,vt,&Nx,work,&lwork,iwork,&info);
  if(info){
    exit(0);
  }

  // Optimal workspace size is returned in work[0].
  lwork = workSize;
  work = malloc(lwork*sizeof(double));

  // Call dgesdd_ to do the actual computation:
  dgesdd_("A",&Ny,&Nx,x,&lda,s,u,&Ny,vt,&Nx,work,&lwork,iwork,&info);
  if (info){
    exit(0);
  }
  // Cleanup workspace:
  free(work);
  free(iwork);

  for(i=0; i<nsv; i++){
    cblas_dscal(Ny, min(coef,s[i]), u+i*Ny, 1);
  }

  //printf("Call BLAS\n");
  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,Ny,Nx,nsv,1,u,Ny,vt,Nx,0,y,Ny);

  free(u);
  free(s);
  free(vt);
}
*/
