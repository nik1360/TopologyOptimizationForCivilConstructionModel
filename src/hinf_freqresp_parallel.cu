#include "parallel.cuh"
#include "sparse_struc.cuh"
#include "sys.cuh"




void sys_inf_norm_parallel(struct sys *sys, double *hinf_freq_peak, double *hinf_nrm, cublasHandle_t cublas_handle, cusolverDnHandle_t cusolverH) {
	
	struct sparse matA_d;
	struct sparse matE_d;
	struct sparse matC_d;
	struct sparse matB_d;

	double *diff_d, *prod1_d, *prod2_d, *matB_full_d, *matC_full_d;
	double alpha = 1, beta = 0;
	int size_x, size_y;
	int inv_nrow, inv_ncol;

	int incx = 1, incy = 1;

	//inversion parameters
	int lda, ldb;
	double *inv_full_d;
	int *ipiv_d = NULL, *info_d = NULL;
	int lwork = 0;
	double *d_work = NULL;
	
	double omega, omega_peak, omegastep, omegamax, nrm_max, current_nrm;

	matE_d.nrow = sys->e.nrow;
	matE_d.ncol = sys->e.ncol;
	matE_d.nnz = sys->e.nnz;

	matA_d.nrow = sys->a.nrow;
	matA_d.ncol = sys->a.ncol;
	matA_d.nnz = sys->a.nnz;

	matC_d.nrow = sys->c.nrow;
	matC_d.ncol = sys->c.ncol;
	matC_d.nnz = sys->c.nnz;

	matB_d.nrow = sys->b.nrow;
	matB_d.ncol = sys->b.ncol;
	matB_d.nnz = sys->b.nnz;

	inv_nrow = matE_d.nrow;
	inv_ncol = matE_d.ncol;

	lda = matE_d.nrow;
	ldb = matE_d.nrow;


	//------------------------------allocating device pointers------------------------------------------

	cudaMalloc((void**)&matA_d.row, matA_d.nnz * sizeof(int));
	
	cudaMalloc((void**)&matA_d.col, matA_d.nnz * sizeof(int));
	cudaMalloc((void**)&matA_d.val, matA_d.nnz * sizeof(double));

	cudaMalloc((void**)&matE_d.row, matE_d.nnz * sizeof(int));
	cudaMalloc((void**)&matE_d.col, matE_d.nnz * sizeof(int));
	cudaMalloc((void**)&matE_d.val, matE_d.nnz * sizeof(double));
	
	cudaMalloc((void**)&matC_d.row, matC_d.nnz * sizeof(int));
	cudaMalloc((void**)&matC_d.col, matC_d.nnz * sizeof(int));
	cudaMalloc((void**)&matC_d.val, matC_d.nnz * sizeof(double));

	cudaMalloc((void**)&matB_d.row, matB_d.nnz * sizeof(int));
	cudaMalloc((void**)&matB_d.col, matB_d.nnz * sizeof(int));
	cudaMalloc((void**)&matB_d.val, matB_d.nnz * sizeof(double));

	cudaMalloc((void**)&diff_d, matE_d.nrow*matE_d.ncol * sizeof(double));

	cudaMalloc((void**)&inv_full_d, inv_nrow * inv_ncol * sizeof(double));
	cudaMalloc((void**)&ipiv_d, matE_d.nrow * sizeof(int));
	cudaMalloc((void**)&info_d, sizeof(int));

	cudaMalloc((void**)&matB_full_d, matB_d.nrow*matB_d.ncol * sizeof(double));
	cudaMalloc((void**)&matC_full_d, matC_d.nrow*matC_d.ncol * sizeof(double));

	cudaMalloc((void**)&prod1_d, matC_d.nrow*matE_d.ncol * sizeof(double));
	cudaMalloc((void**)&prod2_d, inv_nrow*matB_d.ncol * sizeof(double));

	
	//----------------------------copy matrices to the device---------------------------------------------

	cudaMemcpy(matA_d.row, sys->a.row, matA_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matA_d.col, sys->a.col, matA_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matA_d.val, sys->a.val, matA_d.nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(matB_d.row, sys->b.row, matB_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matB_d.col, sys->b.col, matB_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matB_d.val, sys->b.val, matB_d.nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(matC_d.row, sys->c.row, matC_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matC_d.col, sys->c.col, matC_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matC_d.val, sys->c.val, matC_d.nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(matE_d.row, sys->e.row, matE_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matE_d.col, sys->e.col, matE_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matE_d.val, sys->e.val, matE_d.nnz * sizeof(double), cudaMemcpyHostToDevice);

	omega = 0;
	omega_peak = 0;
	omegamax = 0.1;
	omegastep = 0.01;
	nrm_max = 0;
	current_nrm = 0;

	//converting B,C into dense format

	cudaMemset(matB_full_d, 0, matB_d.nrow*matB_d.ncol * sizeof(double));
	cudaMemset(matC_full_d, 0, matC_d.nrow*matC_d.ncol * sizeof(double));


	if (matB_d.nnz % THREADS == 0)
		size_x = matB_d.nnz / THREADS;
	else
		size_x = (matB_d.nnz / THREADS) + 1;
	dim3 threads3(THREADS, 1, 1);
	dim3 blocks3(size_x, 1, 1);

	sparse2full_kernel << <blocks3, threads3 >> > (matB_full_d, matB_d.row, matB_d.col, matB_d.val, matB_d.ncol, matB_d.nnz, 1);

	if (matC_d.nnz % THREADS == 0)
		size_x = matC_d.nnz / THREADS;
	else
		size_x = (matC_d.nnz / THREADS) + 1;
	dim3 threads4(THREADS, 1, 1);
	dim3 blocks4(size_x, 1, 1);

	sparse2full_kernel << <blocks4, threads4 >> > (matC_full_d, matC_d.row, matC_d.col, matC_d.val, matC_d.ncol, matC_d.nnz, 1);
	cudaDeviceSynchronize();

	while (omega < omegamax) {
		//-----------------------computing diff=omega*E-matA and creating the sparse version of diff-----------------------
		if (matE_d.nrow % THREADS == 0)
			size_x = matE_d.nrow / THREADS;
		else
			size_x = (matE_d.nrow / THREADS) + 1;

		size_y = size_x;
		dim3 threads1(THREADS, THREADS, 1);
		dim3 blocks1(size_x, size_y, 1);

		cudaMemset(diff_d, 0, matE_d.nrow*matE_d.ncol * sizeof(double));
		sparse2full_diff_kernel << <blocks1, threads1 >> > (matE_d.row, matE_d.col, matE_d.val, matE_d.nnz, matA_d.row, matA_d.col, matA_d.val, matA_d.nnz, diff_d, matE_d.nrow, matE_d.ncol, omega);
		cudaDeviceSynchronize();

		//------------------------------INVERSION of (omega*E - A) ---------------------------

		//set inv matrix as identity
		if (matE_d.nrow % THREADS == 0)
			size_x = matE_d.nrow / THREADS;
		else
			size_x = (matE_d.nrow / THREADS) + 1;
		dim3 threads2(THREADS, 1, 1);
		dim3 blocks2(size_x, 1, 1);

		cudaMemset(inv_full_d, 0, matE_d.nrow*matE_d.ncol * sizeof(double));
		create_identity_kernel << <blocks2, threads2 >> > (inv_full_d, matE_d.ncol);
		cudaDeviceSynchronize();

		//query working space of getrf
		cusolverDnDgetrf_bufferSize(cusolverH, matE_d.nrow, matE_d.nrow, diff_d, lda, &lwork);
		cudaMalloc((void**)&d_work, sizeof(double) * lwork);
		//LU factorization of the matrix to invert
		cusolverDnDgetrf(cusolverH, matE_d.nrow, matE_d.nrow, diff_d, lda, d_work, ipiv_d, info_d);
		cudaFree(d_work);

		//computation of the inverse  
		cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, matE_d.nrow, matE_d.nrow, diff_d, lda, ipiv_d, inv_full_d, ldb, info_d);

		//----------------------------------computing C*inv*B----------------------------------------------
		
		//computing prod1=C*inv
		cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, matC_d.nrow, inv_ncol, matC_d.ncol, &alpha, matC_full_d, matC_d.nrow, inv_full_d, inv_nrow, &beta, prod1_d, matC_d.nrow);
		//computing current_nrm=prod1*B
		cublasDdot(cublas_handle, matB_d.nrow, prod1_d, incx, matB_full_d, incy, &current_nrm);

		if (fabs(current_nrm) > nrm_max) {
			nrm_max = fabs(current_nrm);
			omega_peak = omega;
		}

		omega += omegastep;
	}

	(*hinf_freq_peak) = omega_peak;
	(*hinf_nrm) = nrm_max;


	//------------------------------------free----------------------------

	cudaFree(matA_d.row);
	cudaFree(matA_d.col);
	cudaFree(matA_d.val);

	cudaFree(matB_d.row);
	cudaFree(matB_d.col);
	cudaFree(matB_d.val);

	cudaFree(matC_d.row);
	cudaFree(matC_d.col);
	cudaFree(matC_d.val);

	cudaFree(matE_d.row);
	cudaFree(matE_d.col);
	cudaFree(matE_d.val);

	cudaFree(matB_full_d);
	cudaFree(matC_full_d);

	cudaFree(diff_d);

	cudaFree(inv_full_d);
	cudaFree(ipiv_d);
	cudaFree(info_d);

	
}

void freqresp_parallel(struct sys *sys, double omega, double *val, cublasHandle_t cublas_handle, cusolverDnHandle_t cusolverH) {	//NOT USED!

	struct sparse matA_d;
	struct sparse matE_d;
	struct sparse matC_d;
	struct sparse matB_d;

	double *diff_d, *prod1_d, *prod2_d, *matB_full_d, *matC_full_d;
	double alpha = 1, beta = 0;
	int size_x, size_y;
	int inv_nrow, inv_ncol;

	//inversion parameters
	int lda, ldb;
	double *inv_full_d;
	int *ipiv_d = NULL, *info_d = NULL;
	int lwork = 0;
	double *d_work = NULL;

	int incx = 1, incy = 1;

	matE_d.nrow = sys->e.nrow;
	matE_d.ncol = sys->e.ncol;
	matE_d.nnz = sys->e.nnz;

	matA_d.nrow = sys->a.nrow;
	matA_d.ncol = sys->a.ncol;
	matA_d.nnz = sys->a.nnz;

	matC_d.nrow = sys->c.nrow;
	matC_d.ncol = sys->c.ncol;
	matC_d.nnz = sys->c.nnz;

	matB_d.nrow = sys->b.nrow;
	matB_d.ncol = sys->b.ncol;
	matB_d.nnz = sys->b.nnz;

	inv_nrow = matE_d.nrow;
	inv_ncol = matE_d.ncol;

	lda = matE_d.nrow;
	ldb = matE_d.nrow;


	//------------------------------allocating device pointers------------------------------------------

	cudaMalloc((void**)&matA_d.row, matA_d.nnz * sizeof(int));
	cudaMalloc((void**)&matA_d.col, matA_d.nnz * sizeof(int));
	cudaMalloc((void**)&matA_d.val, matA_d.nnz * sizeof(double));

	cudaMalloc((void**)&matE_d.row, matE_d.nnz * sizeof(int));
	cudaMalloc((void**)&matE_d.col, matE_d.nnz * sizeof(int));
	cudaMalloc((void**)&matE_d.val, matE_d.nnz * sizeof(double));

	cudaMalloc((void**)&matC_d.row, matC_d.nnz * sizeof(int));
	cudaMalloc((void**)&matC_d.col, matC_d.nnz * sizeof(int));
	cudaMalloc((void**)&matC_d.val, matC_d.nnz * sizeof(double));

	cudaMalloc((void**)&matB_d.row, matB_d.nnz * sizeof(int));
	cudaMalloc((void**)&matB_d.col, matB_d.nnz * sizeof(int));
	cudaMalloc((void**)&matB_d.val, matB_d.nnz * sizeof(double));

	cudaMalloc((void**)&diff_d, matE_d.nrow*matE_d.ncol * sizeof(double));

	cudaMalloc((void**)&inv_full_d, inv_nrow * inv_ncol * sizeof(double));
	cudaMalloc((void**)&ipiv_d, matE_d.nrow * sizeof(int));
	cudaMalloc((void**)&info_d, sizeof(int));

	cudaMalloc((void**)&matB_full_d, matB_d.nrow*matB_d.ncol * sizeof(double));
	cudaMalloc((void**)&matC_full_d, matC_d.nrow*matC_d.ncol * sizeof(double));

	cudaMalloc((void**)&prod1_d, matC_d.nrow*matE_d.ncol * sizeof(double));
	cudaMalloc((void**)&prod2_d, inv_nrow*matB_d.ncol * sizeof(double));


	//----------------------------copy matrices to the device---------------------------------------------

	cudaMemcpy(matA_d.row, sys->a.row, matA_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matA_d.col, sys->a.col, matA_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matA_d.val, sys->a.val, matA_d.nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(matB_d.row, sys->b.row, matB_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matB_d.col, sys->b.col, matB_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matB_d.val, sys->b.val, matB_d.nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(matC_d.row, sys->c.row, matC_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matC_d.col, sys->c.col, matC_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matC_d.val, sys->c.val, matC_d.nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(matE_d.row, sys->e.row, matE_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matE_d.col, sys->e.col, matE_d.nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matE_d.val, sys->e.val, matE_d.nnz * sizeof(double), cudaMemcpyHostToDevice);


	//-----------------------computing diff=omega*E-matA and creating the sparse version of diff-----------------------
	if (matE_d.nrow % THREADS == 0)
		size_x = matE_d.nrow / THREADS;
	else
		size_x = (matE_d.nrow / THREADS) + 1;

	size_y = size_x;
	dim3 threads1(THREADS, THREADS, 1);
	dim3 blocks1(size_x, size_y, 1);

	cudaMemset(diff_d, 0, matE_d.nrow*matE_d.ncol * sizeof(double));
	sparse2full_diff_kernel << <blocks1, threads1 >> > (matE_d.row, matE_d.col, matE_d.val, matE_d.nnz, matA_d.row, matA_d.col, matA_d.val, matA_d.nnz, diff_d, matE_d.nrow, matE_d.ncol, omega);
	cudaDeviceSynchronize();

	//------------------------------INVERSION of (omega*E - A) ---------------------------

	//allocations

	//set inv matrix as identity
	if (matE_d.nrow % THREADS == 0)
		size_x = matE_d.nrow / THREADS;
	else
		size_x = (matE_d.nrow / THREADS) + 1;
	dim3 threads2(THREADS, 1, 1);
	dim3 blocks2(size_x, 1, 1);

	cudaMemset(inv_full_d, 0, matE_d.nrow*matE_d.ncol * sizeof(double));
	create_identity_kernel << <blocks2, threads2 >> > (inv_full_d, matE_d.ncol);
	cudaDeviceSynchronize();
	//query working space of getrf
	cusolverDnDgetrf_bufferSize(cusolverH, matE_d.nrow, matE_d.nrow, diff_d, lda, &lwork);
	cudaMalloc((void**)&d_work, sizeof(double) * lwork);
	//LU factorization of the matrix to invert
	cusolverDnDgetrf(cusolverH, matE_d.nrow, matE_d.nrow, diff_d, lda, d_work, ipiv_d, info_d);
	cudaFree(d_work);

	//computation of the inverse  
	cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, matE_d.nrow, matE_d.nrow, diff_d, lda, ipiv_d, inv_full_d, ldb, info_d);

	//----------------------------------computing C*inv*B----------------------------------------------
	//converting B,C into dense format

	cudaMemset(matB_full_d, 0, matB_d.nrow*matB_d.ncol * sizeof(double));
	cudaMemset(matC_full_d, 0, matC_d.nrow*matC_d.ncol * sizeof(double));

	if (matB_d.nnz % THREADS == 0)
		size_x = matB_d.nnz / THREADS;
	else
		size_x = (matB_d.nnz / THREADS) + 1;
	dim3 threads3(THREADS, 1, 1);
	dim3 blocks3(size_x, 1, 1);

	sparse2full_kernel << <blocks3, threads3 >> > (matB_full_d, matB_d.row, matB_d.col, matB_d.val, matB_d.ncol, matB_d.nnz, 1);

	if (matC_d.nnz % THREADS == 0)
		size_x = matC_d.nnz / THREADS;
	else
		size_x = (matC_d.nnz / THREADS) + 1;
	dim3 threads4(THREADS, 1, 1);
	dim3 blocks4(size_x, 1, 1);

	sparse2full_kernel << <blocks4, threads4 >> > (matC_full_d, matC_d.row, matC_d.col, matC_d.val, matC_d.ncol, matC_d.nnz, 1);
	cudaDeviceSynchronize();

	//computing prod1=C*inv
	cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, matC_d.nrow, inv_ncol, matC_d.ncol, &alpha, matC_full_d, matC_d.nrow, inv_full_d, inv_nrow, &beta, prod1_d, matC_d.nrow);
	//computing  val=prod1*B
	cublasDdot(cublas_handle, matB_d.nrow, prod1_d, incx, matB_full_d, incy, val);


	//------------------------------------free----------------------------

	cudaFree(matA_d.row);
	cudaFree(matA_d.col);
	cudaFree(matA_d.val);

	cudaFree(matB_d.row);
	cudaFree(matB_d.col);
	cudaFree(matB_d.val);

	cudaFree(matC_d.row);
	cudaFree(matC_d.col);
	cudaFree(matC_d.val);

	cudaFree(matE_d.row);
	cudaFree(matE_d.col);
	cudaFree(matE_d.val);

	cudaFree(matB_full_d);
	cudaFree(matC_full_d);

	cudaFree(diff_d);

	cudaFree(inv_full_d);
	cudaFree(ipiv_d);
	cudaFree(info_d);
}