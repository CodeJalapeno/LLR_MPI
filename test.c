#include "LLR.h"
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int
main(int argc, char *argv[])
{
  size_t i=0;
  int procid, nprocs, nx, ny, nc, nt, nb, max_iter;
  double coef, mu;
  double start_time, end_time;
  double cmp = 0;
  dcomplex *in_array, *out_array, *serial_out_array;
  FILE *mat;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  if(procid == 0) {


    mat = fopen("./mat.in", "r");
    fscanf(mat, "%d", &nx);
    fscanf(mat, "%d", &ny);
    fscanf(mat, "%d", &nc);
    fscanf(mat, "%d", &nt);
    fscanf(mat, "%d", &nb);
    fscanf(mat, "%d", &max_iter);
    fscanf(mat, "%lf", &coef);
    fscanf(mat, "%lf", &mu);

    in_array = (dcomplex*)malloc(nx*ny*nc*nt*sizeof(dcomplex));
    out_array = (dcomplex*)malloc(nx*ny*nc*nt*sizeof(dcomplex));
    for(i=0; i<nx*ny*nc*nt; i++){
      fscanf(mat, "%lf,%lf", &(in_array[i].re), &(in_array[i].im));
    }

    fclose(mat);
  }
  else{
    in_array = NULL;
    out_array = NULL;
  }
  if(procid == 0){
    printf("MPI denoising\n");
  }
  if(procid == 0){
    start_time = get_wall_time();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  cdenoise_mpi_acc(in_array, out_array, nx, ny, nc, nt, nb, coef, max_iter, mu);
  if(procid == 0){  
    end_time = get_wall_time();
    printf("Running time of MPI denoising: %f\n", end_time-start_time);
  }
/*
  if(procid == 0){
    printf("Serial denoising\n");
    serial_out_array = (dcomplex*)malloc(nx*ny*nc*nt*sizeof(dcomplex));

    start_time = get_wall_time();
    cdenoise(in_array, serial_out_array, nx, ny, nc, nt, nb, coef, max_iter, mu);
    end_time = get_wall_time();
    printf("Running time of serial denoising: %f\n", end_time-start_time);

    for(i=0; i<2*nx*ny*nc*nt; i++){
      cmp += (((double*)out_array)[i] - ((double*)serial_out_array)[i])*
        (((double*)out_array)[i] - ((double*)serial_out_array)[i]);
    }
    printf("Comparision of version: %lf\n", cmp);
*/
    free(in_array);
    free(out_array);
 //   free(serial_out_array);
 // }

  MPI_Finalize();
  return 0;
}
