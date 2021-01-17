#include "parallel.cuh"

__host__ void checkCudaError(int linea) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Linea %d CudaError:%s\n", linea, cudaGetErrorString(err));
	}

}
__global__ void sparse2full_diff_kernel(int *row1, int *col1, double *val1, int nnz1, int *row2, int *col2, double *val2, int nnz2, double *res, int nrow, int ncol, double omega) {

	int i = blockIdx.x*blockDim.x + threadIdx.x;	//col
	int j = blockIdx.y*blockDim.y + threadIdx.y;	//row
	int sparse_index;

	if ((i < nrow) && (j < ncol)) {
		if (omega != 0.0) {
			for (sparse_index = 0;sparse_index < nnz1; sparse_index++) {
				if ((row1[sparse_index] == (j + 1)) && (col1[sparse_index] == (i + 1))) {
					res[(row1[sparse_index] - 1)*ncol + (col1[sparse_index] - 1)] = omega*val1[sparse_index];
					break;
				}
			}
			for (sparse_index = 0;sparse_index < nnz2; sparse_index++) {
				if ((row2[sparse_index] == (j + 1)) && (col2[sparse_index] == (i + 1))) {
					res[(row2[sparse_index] - 1)*ncol + (col2[sparse_index] - 1)] -= val2[sparse_index];
					break;
				}
			}
		}
		else {
			for (sparse_index = 0;sparse_index < nnz2; sparse_index++) {
				if ((row2[sparse_index] == (j + 1)) && (col2[sparse_index] == (i + 1))) {
					res[(row2[sparse_index] - 1)*ncol + (col2[sparse_index] - 1)] = -val2[sparse_index];
					break;
				}
			}
		}
	}
}
__global__ void create_identity_kernel(double *mat, int ncol) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<ncol) {
		mat[(i*ncol) + i] = 1.0;
	}
}
__global__ void sparse2full_kernel(double *full, int*row, int *col, double* val, int ncol, int nnz, int alpha) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<nnz) {
		full[((row[i] - 1)*ncol) + (col[i] - 1)] = alpha*val[i];
	}
}
