#include "parallel.cuh"
#include "sparse_struc.cuh"
#include "sys.cuh"

void gsens2_parallel(double *dprime, struct sparse *dunoprime_h, struct sparse *dzeroprime_h,struct sparse *dunoprime_d, struct sparse *dzeroprime_d, struct sparse *matB_d, struct sparse *matC_d, struct sparse *invu_d,
	double *invu_full_d, double* diff_d, double *matB_full_d, double *matC_full_d, double fpeak, cublasHandle_t cublas_handle) {

	cublasStatus_t status;

	double *prod1_d, *prod2_d, *prod3_d;
	int prod1_nrow, prod1_ncol, prod2_nrow, prod2_ncol, prod3_nrow, prod3_ncol;
	double alpha = 1, beta = 0;
	int incx = 1, incy = 1;
	int size_x, size_y;


	dunoprime_d->nnz = dunoprime_h->nnz;
	dzeroprime_d->nnz = dzeroprime_h->nnz;

	prod1_nrow = matC_d->nrow;
	prod1_ncol = invu_d->ncol;
	prod2_nrow = matC_d->nrow;
	prod2_ncol = dzeroprime_d->ncol;
	prod3_nrow = matC_d->nrow;
	prod3_ncol = invu_d->ncol;


	cudaMalloc((void**)&dunoprime_d->row, dunoprime_d->nnz * sizeof(int));
	cudaMalloc((void**)&dunoprime_d->col, dunoprime_d->nnz * sizeof(int));
	cudaMalloc((void**)&dunoprime_d->val, dunoprime_d->nnz * sizeof(double));

	cudaMalloc((void**)&dzeroprime_d->row, dzeroprime_d->nnz * sizeof(int));
	cudaMalloc((void**)&dzeroprime_d->col, dzeroprime_d->nnz * sizeof(int));
	cudaMalloc((void**)&dzeroprime_d->val, dzeroprime_d->nnz * sizeof(double));

	cudaMalloc((void**)&prod1_d, prod1_nrow*prod1_ncol * sizeof(double));
	cudaMalloc((void**)&prod2_d, prod2_nrow*prod2_ncol * sizeof(double));
	cudaMalloc((void**)&prod3_d, prod3_nrow*prod3_ncol * sizeof(double));


	//-----------------------copying from host to device---------------------------

	cudaMemcpy(dunoprime_d->row, dunoprime_h->row, dunoprime_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dunoprime_d->col, dunoprime_h->col, dunoprime_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dunoprime_d->val, dunoprime_h->val, dunoprime_d->nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(dzeroprime_d->row, dzeroprime_h->row, dzeroprime_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dzeroprime_d->col, dzeroprime_h->col, dzeroprime_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dzeroprime_d->val, dzeroprime_h->val, dzeroprime_d->nnz * sizeof(double), cudaMemcpyHostToDevice);

	//-----------------computing diff_d=fpeak*DUNOPRIME-DZEROPRIME------------------------
	if (dzeroprime_d->nrow % THREADS == 0)
		size_x = dzeroprime_d->nrow / THREADS;
	else
		size_x = (dzeroprime_d->nrow / THREADS) + 1;

	size_y = size_x;
	dim3 threads1(THREADS, THREADS, 1);
	dim3 blocks1(size_x, size_y, 1);

	cudaMemset(diff_d, 0, dzeroprime_d->nrow*dzeroprime_d->ncol * sizeof(double));
	sparse2full_diff_kernel << <blocks1, threads1 >> > (dunoprime_d->row, dunoprime_d->col, dunoprime_d->val, dunoprime_d->nnz, dzeroprime_d->row, dzeroprime_d->col, dzeroprime_d->val, dzeroprime_d->nnz, diff_d, dzeroprime_d->nrow, dzeroprime_d->ncol, fpeak);
	cudaDeviceSynchronize();
	
	//-------------------computing prod1=-C*INVU----------------------------------
	status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, matC_d->nrow, invu_d->ncol, matC_d->ncol, &alpha, matC_full_d, matC_d->nrow, invu_full_d, invu_d->nrow, &beta, prod1_d, prod1_nrow);
	//-------------------computing prod2=prod1*diff_d----------------------------------
	status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, prod1_nrow, dzeroprime_d->ncol, prod1_ncol, &alpha, prod1_d, prod1_nrow, diff_d, dzeroprime_d->nrow, &beta, prod2_d, prod2_nrow);
	//-------------------computing prod3=prod2*invu----------------------------------
	status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, prod2_nrow, invu_d->ncol, prod2_ncol, &alpha, prod2_d, prod2_nrow, invu_full_d, invu_d->nrow, &beta, prod3_d, prod3_nrow);
	//-------------------computing prod4=prod3*B----------------------------------
	status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, prod3_nrow, matB_d->ncol, prod3_ncol, &alpha, prod3_d, prod3_nrow, matB_full_d, matB_d->nrow, &beta, prod3_d, prod3_nrow);
	cublasDdot(cublas_handle, prod3_ncol, prod3_d, incx, matB_full_d, incy, dprime);

	//------------------------------free---------------------------------

	cudaFree(dunoprime_d->row);
	cudaFree(dunoprime_d->col);
	cudaFree(dunoprime_d->val);

	cudaFree(dzeroprime_d->row);
	cudaFree(dzeroprime_d->col);
	cudaFree(dzeroprime_d->val);

	cudaFree(prod1_d);
	cudaFree(prod2_d);
	cudaFree(prod3_d);
}