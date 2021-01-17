#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "constants.cuh"
#include "matrix_functions.cuh"
#include "sys.cuh"
#include "parallel.cuh"


void invforsens2_parallel(struct sparse *invu, struct sys *g, double h_inf_peak_frq, cusolverDnHandle_t cusolverH) {
	double *diff_d, *invu_full_d, *invu_full_h;

	struct sparse matA_d;
	struct sparse matE_d;

	int size_x, size_y;

	//inversion parameters
	int lda, ldb;
	double *inv_full_d;
	int *ipiv_d = NULL, *info_d = NULL;
	int lwork = 0;
	double *d_work = NULL;


	matA_d.nrow = g->a.nrow;
	matA_d.ncol = g->a.ncol;
	matA_d.nnz = g->a.nnz;

	matE_d.nrow = g->e.nrow;
	matE_d.ncol = g->e.ncol;
	matE_d.nnz = g->e.nnz;

	lda = matE_d.nrow;
	ldb = matE_d.nrow;


	
	//-------------allocating space on the device------------------

	cudaMalloc((void**)&matA_d.row, matA_d.nnz * sizeof(int));
	cudaMalloc((void**)&matA_d.col, matA_d.nnz * sizeof(int));
	cudaMalloc((void**)&matA_d.val, matA_d.nnz * sizeof(double));

	cudaMalloc((void**)&matE_d.row, matE_d.nnz * sizeof(int));
	cudaMalloc((void**)&matE_d.col, matE_d.nnz * sizeof(int));
	cudaMalloc((void**)&matE_d.val, matE_d.nnz * sizeof(double));

	cudaMalloc((void**)&diff_d, matE_d.nrow*matE_d.ncol * sizeof(double));
	cudaMalloc((void**)&invu_full_d, matE_d.nrow*matE_d.ncol * sizeof(double));
	
	cudaMalloc((void**)&ipiv_d, matE_d.nrow * sizeof(int));
	cudaMalloc((void**)&info_d, sizeof(int));

	invu_full_h = (double*)malloc(matE_d.nrow*matE_d.ncol * sizeof(double));

	//--------------copying sparce matrices from host to device------------

	cudaMemcpy(matA_d.row, g->a.row, matA_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matA_d.col, g->a.col, matA_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matA_d.val, g->a.val, matA_d.nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(matE_d.row, g->e.row, matE_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matE_d.col, g->e.col, matE_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matE_d.val, g->e.val, matE_d.nnz * sizeof(double), cudaMemcpyHostToDevice);

	checkCudaError(__LINE__);

	//-----------------------computing diff=omega*E-matA and creating the sparse version of diff-----------------------
	if (matE_d.nrow % THREADS == 0)
		size_x = matE_d.nrow / THREADS;
	else
		size_x = (matE_d.nrow / THREADS) + 1;

	size_y = size_x;
	dim3 threads1(THREADS, THREADS, 1);
	dim3 blocks1(size_x, size_y, 1);

	cudaMemset(diff_d, 0, matE_d.nrow*matE_d.ncol * sizeof(double));
	sparse2full_diff_kernel << <blocks1, threads1 >> > (matE_d.row, matE_d.col, matE_d.val, matE_d.nnz, matA_d.row, matA_d.col, matA_d.val, matA_d.nnz, diff_d, matE_d.nrow, matE_d.ncol, h_inf_peak_frq);
	checkCudaError(__LINE__);
	//------------------------------INVERSION of (omega*E - A) ---------------------------

	//allocations

	//set inv matrix as identity
	if (matE_d.nrow % THREADS == 0)
		size_x = matE_d.nrow / THREADS;
	else
		size_x = (matE_d.nrow / THREADS) + 1;
	dim3 threads2(THREADS, 1, 1);
	dim3 blocks2(size_x, 1, 1);

	cudaMemset(invu_full_d, 0, matE_d.nrow * matE_d.ncol * sizeof(double));
	checkCudaError(__LINE__);
	create_identity_kernel << <blocks2, threads2 >> > (invu_full_d, matE_d.ncol);
	checkCudaError(__LINE__);
	cudaDeviceSynchronize();
	//query working space of getrf
	cusolverDnDgetrf_bufferSize(cusolverH, matE_d.nrow, matE_d.nrow, diff_d, lda, &lwork);
	checkCudaError(__LINE__);
	cudaMalloc((void**)&d_work, sizeof(double) * lwork);
	checkCudaError(__LINE__);
	//LU factorization of the matrix to invert
	cusolverDnDgetrf(cusolverH, matE_d.nrow, matE_d.nrow, diff_d, lda, d_work, ipiv_d, info_d);
	checkCudaError(__LINE__);
	cudaFree(d_work);
	
	//computation of the inverse  
	cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, matE_d.nrow, matE_d.nrow, diff_d, lda, ipiv_d, invu_full_d, ldb, info_d);
	checkCudaError(__LINE__);



	//----------------copying results on the host-------------------------
	cudaMemcpy(invu_full_h, invu_full_d, matE_d.nrow*matE_d.ncol * sizeof(double), cudaMemcpyDeviceToHost);
	checkCudaError(__LINE__);
	dense_to_sparse(invu_full_h, matE_d.nrow, matE_d.ncol, invu);
	checkCudaError(__LINE__);

	
	//------------------free------------------------------
	
	cudaFree(matA_d.row);
	cudaFree(matA_d.col);
	cudaFree(matA_d.val);

	cudaFree(matE_d.row);
	cudaFree(matE_d.col);
	cudaFree(matE_d.val);
	
	cudaFree(diff_d);
	cudaFree(invu_full_d);
	
	free(invu_full_h);

}