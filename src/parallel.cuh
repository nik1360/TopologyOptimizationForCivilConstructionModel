#pragma once
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "sparse_struc.cuh"

#define THREADS 32

__global__ void sparse2full_kernel(double *full, int*row, int *col, double* val, int ncol, int nnz, int alpha);
__global__ void create_identity_kernel(double *mat, int ncol);
__global__ void sparse2full_diff_kernel(int *row1, int *col1, double *val1, int nnz1, int *row2, int *col2, double *val2, int nnz2, double *res, int nrow, int ncol, double omega);
__host__ void checkCudaError(int linea);

void freqresp_parallel(struct sys *sys, double omega, double *val, cublasHandle_t cublas_handle, cusolverDnHandle_t cusolverH);
void sys_inf_norm_parallel(struct sys *sys, double *omega_peak, double *hinf_nrm, cublasHandle_t cublas_handle, cusolverDnHandle_t cusolverH);

void gsens2_parallel(double *dprime, struct sparse *dunoprime_h, struct sparse *dzeroprime_h, struct sparse *dunoprime_d, struct sparse *dzeroprime_d, struct sparse *matB_d, struct sparse *matC_d, struct sparse *invu_d,
	double *invu_full_d, double* diff_d, double *matB_full_d, double *matC_full_d, double fpeak, cublasHandle_t cublas_handle);


void init_pointers_gsens2(struct sparse *dunoprime_h, struct sparse *dzeroprime_h, struct sparse *matB_h, struct sparse *matC_h, struct sparse *invu_h,
	struct sparse *dunoprime_d, struct sparse *dzeroprime_d, struct sparse *matB_d, struct sparse *matC_d, struct sparse *invu_d,
	double **invu_full_d, double **diff_d, double **matB_full_d, double **matC_full_d);
void free_pointers_gsens2(struct sparse *dunoprime_d, struct sparse *dzeroprime_d, struct sparse *matB_d, struct sparse *matC_d, struct sparse *invu_d,
	double *invu_full_d, double* diff_d, double *matB_full_d, double *matC_full_d);

void invforsens2_parallel(struct sparse *invu, struct sys *g, double h_inf_peak_frq, cusolverDnHandle_t cusolverH);