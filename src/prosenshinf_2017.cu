#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "main_functions.cuh"
#include "constants.cuh"
#include "mesh.cuh"
#include "matrix_functions.cuh"
#include "material.cuh"
#include "sys.cuh"

#include "parallel.cuh" 

void prosenshinf_2017(double *xphys, struct sparse *invu, int us, int vs, double h_inf_peak_frq, double *df0dx,
	double *me, double *ke, int *edofmat, struct mesh *mesh, struct material *material, int ntutto, int *freedofs, int *fixeddofs, struct sparse *bstato, struct sparse *cstato) {

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	int *dofvec;
	double *diff_full;
	double a;
	double b;
	double pi = 3.141592653589793, val1, val2;
	int  dunoprime_index = 0, dzeroprime_index = 0;
	int dunoprime_new_index = 0, dzeroprime_new_index = 0;
	int nfreedofs, *indici;
	int numel;
	int found_row, found_col, count_row, count_col;
	double dprime;

	struct sparse dunoprime;
	struct sparse dzeroprime;
	struct sparse dunoprime_new;
	struct sparse dzeroprime_new;

	//---------------------------------pointers used in the device------------------------------
	struct sparse dunoprime_d;		
	struct sparse dzeroprime_d;
	struct sparse matB_d;
	struct sparse matC_d;
	struct sparse invu_d;
	double *invu_full_d, *diff_d, *matB_full_d, *matC_full_d;

	

	//----------------------------------------------------------------------------------------

	numel = mesh->nelx*mesh->nely;

	dunoprime.nrow = 2 * ntutto - 2 * mesh->fixed_count;
	dunoprime.ncol = 2 * ntutto - 2 * mesh->fixed_count;
	dzeroprime.nrow = 2 * ntutto - 2 * mesh->fixed_count;
	dzeroprime.ncol = 2 * ntutto - 2 * mesh->fixed_count;
	dunoprime_new.nrow = 2 * ntutto - 2 * mesh->fixed_count;
	dunoprime_new.ncol = 2 * ntutto - 2 * mesh->fixed_count;
	dzeroprime_new.nrow = 2 * ntutto - 2 * mesh->fixed_count;
	dzeroprime_new.ncol = 2 * ntutto - 2 * mesh->fixed_count;

	dunoprime.nnz = 2 * MEROW*MECOL;		//why 32 in matlab?
	dzeroprime.nnz = 2 * KEROW*KECOL;		//why 16 in matlab?

	diff_full = (double*)malloc(dzeroprime.nrow*dzeroprime.ncol * sizeof(double));

	dunoprime.val = (double*)malloc(dunoprime.nnz * sizeof(double));
	dunoprime.row = (int*)malloc(dunoprime.nnz * sizeof(int));
	dunoprime.col = (int*)malloc(dunoprime.nnz * sizeof(int));

	dzeroprime.val= (double*)malloc(dzeroprime.nnz * sizeof(double));
	dzeroprime.row = (int*)malloc(dzeroprime.nnz * sizeof(int));
	dzeroprime.col = (int*)malloc(dzeroprime.nnz * sizeof(int));

	dunoprime_new.val = (double*)malloc(dunoprime.nnz * sizeof(double));
	dunoprime_new.row = (int*)malloc(dunoprime.nnz * sizeof(int));
	dunoprime_new.col = (int*)malloc(dunoprime.nnz * sizeof(int));

	dzeroprime_new.val = (double*)malloc(dzeroprime.nnz * sizeof(double));
	dzeroprime_new.row = (int*)malloc(dzeroprime.nnz * sizeof(int));
	dzeroprime_new.col = (int*)malloc(dzeroprime.nnz * sizeof(int));

	init_pointers_gsens2(&dunoprime_new, &dzeroprime_new, bstato, cstato, invu, &dunoprime_d, &dzeroprime_d, &matB_d, &matC_d, &invu_d, &invu_full_d, &diff_d, &matB_full_d, &matC_full_d);


	dofvec = (int*)malloc(EDOFMAT_COL * sizeof(int));
	nfreedofs = (2 * (mesh->nelx + 1)*(mesh->nely + 1)) - mesh->fixed_count;
	indici = (int*)malloc(2 * nfreedofs * sizeof(int));

	//INDICI = union(freedofs,NTUTTO+freedofs);
	for (int j = 0; j < nfreedofs;j++) {
		indici[j] = freedofs[j];
	}
	for (int j = nfreedofs; j < 2 * nfreedofs; j++) {
		indici[j] = freedofs[j - nfreedofs] + ntutto;
	}

	for (int i = 0; i < numel;i++) {

		dunoprime_index = 0;
		dzeroprime_index = 0;
		dunoprime_new_index = 0;
		dzeroprime_new_index = 0;

		for (int j = 0; j < EDOFMAT_COL; j++) {
			dofvec[j] = edofmat[i*EDOFMAT_COL + j];
		}

		//DUNOPRIME(NTUTTO+edofMat(ii,:),NTUTTO+edofMat(ii,:)) = (sin(pi / 2 * xPhys(ii) ^ 2) + pi * xPhys(ii) ^ 2 * cos(pi / 2 * xPhys(ii) ^ 2))*(Rho0 - Rhomin)*ME;
		a = (sin((pi / 2) * pow(xphys[i], 2)) + (pi*pow(xphys[i], 2)*cos((pi / 2)*pow(xphys[i], 2)))) * (material->rho0 - material->rhomin);
		for (int j = 0; j < MEROW; j++) {
			for (int k = 0; k < MECOL;k++) {
				val1 = a * me[j*MECOL + k];
				if (fabs(val1)>pow(10, -9)) {
					dunoprime.val[dunoprime_index] = val1;
					dunoprime.row[dunoprime_index] = dofvec[j] + ntutto;
					dunoprime.col[dunoprime_index] = dofvec[k] + ntutto;
					dunoprime_index++;
				}
			}
		}
		// DZEROPRIME(NTUTTO+edofMat(ii,:),edofMat(ii,:)) = -penal*xPhys(ii)^(penal-1)*(E0-Emin)*KE;
		// DZEROPRIME(NTUTTO+edofMat(ii,:),NTUTTO+edofMat(ii,:)) = ...
		//-ALPHA * (sin(pi / 2 * xPhys(ii) ^ 2) + pi * xPhys(ii) ^ 2 * cos(pi / 2 * xPhys(ii) ^ 2))*(Rho0 - Rhomin)*ME - BETA * (penal*xPhys(ii) ^ (penal - 1)*(E0 - Emin)*KE);
		b = -(mesh->penal)*pow(xphys[i], (mesh->penal - 1))*(material->e0 - material->emin);
		for (int j = 0; j < KEROW; j++) {
			for (int k = 0; k < KECOL; k++) {
				val1 = b * ke[j*KECOL + k];
				if (fabs(val1)>pow(10, -9)) {
					dzeroprime.val[dzeroprime_index] = val1;
					dzeroprime.row[dzeroprime_index] = dofvec[j] + ntutto;
					dzeroprime.col[dzeroprime_index] = dofvec[k];
					dzeroprime_index++;
				}
				val2 = -mesh->alpha*a * me[j*MECOL + k] + mesh->beta*b*ke[j*MECOL + k];
				if (fabs(val2)>pow(10, -9)) {
					dzeroprime.val[dzeroprime_index] = val2;
					dzeroprime.row[dzeroprime_index] = dofvec[j] + ntutto;
					dzeroprime.col[dzeroprime_index] = dofvec[k] + ntutto;
					dzeroprime_index++;
				}
			}
		}

		
		// DZEROPRIME = DZEROPRIME(INDICI,INDICI);
		for (int j = 0; j < dzeroprime_index; j++) {
			found_row = 0;
			for (int k = 0; k < 2 * nfreedofs; k++) {
				if (dzeroprime.row[j] == indici[k]) {
					found_row = 1;
					break;
				}
			}
			found_col = 0;
			for (int k = 0; k < 2 * nfreedofs; k++) {
				if (dzeroprime.col[j] == indici[k]) {
					found_col = 1;
					break;
				}
			}
			if ((found_row == 1) && (found_col == 1)) {
				count_row = 0;
				count_col = 0;
				for (int k = 0; k < mesh->fixed_count; k++) {
					if (dzeroprime.row[j] > fixeddofs[k]) {
						count_row++;
					}
					if (dzeroprime.row[j] > (fixeddofs[k] + ntutto)) {
						count_row++;
					}
				}
				for (int k = 0; k < mesh->fixed_count; k++) {
					if (dzeroprime.col[j] > fixeddofs[k]) {
						count_col++;
					}
					if (dzeroprime.col[j] > (fixeddofs[k] + ntutto)) {
						count_col++;
					}
				}
				dzeroprime_new.val[dzeroprime_new_index] = dzeroprime.val[j];
				dzeroprime_new.row[dzeroprime_new_index] = dzeroprime.row[j] - count_row;
				dzeroprime_new.col[dzeroprime_new_index] = dzeroprime.col[j] - count_col;
				//printf("(%d,%d) %f \n", dzeroprime_new_row[dzeroprime_new_index], dzeroprime_new_col[dzeroprime_new_index], dzeroprime_new[dzeroprime_new_index]);
				dzeroprime_new_index++;
			}
		}
		dzeroprime_new.nnz = dzeroprime_new_index;


		//DUNOPRIME = DUNOPRIME(INDICI,INDICI);
		for (int j = 0; j < dunoprime_index; j++) {
			found_row = 0;
			for (int k = 0; k < 2 * nfreedofs; k++) {
				if (dunoprime.row[j] == indici[k]) {
					found_row = 1;
					break;
				}
			}
			found_col = 0;
			for (int k = 0; k < 2 * nfreedofs; k++) {
				if (dunoprime.col[j] == indici[k]) {
					found_col = 1;
					break;
				}
			}
			if ((found_row == 1) && (found_col == 1)) {
				count_row = 0;
				count_col = 0;
				for (int k = 0; k < mesh->fixed_count; k++) {
					if (dunoprime.row[j] > fixeddofs[k]) {
						count_row++;
					}
					if (dunoprime.row[j] > (fixeddofs[k] + ntutto)) {
						count_row++;
					}
				}
				for (int k = 0; k < mesh->fixed_count; k++) {
					if (dunoprime.col[j] > fixeddofs[k]) {
						count_col++;
					}
					if (dunoprime.col[j] > (fixeddofs[k] + ntutto)) {
						count_col++;
					}
				}
				dunoprime_new.val[dunoprime_new_index] = dunoprime.val[j];
				dunoprime_new.row[dunoprime_new_index] = dunoprime.row[j] - count_row;
				dunoprime_new.col[dunoprime_new_index] = dunoprime.col[j] - count_col;
				dunoprime_new_index++;
			}
		}

		dunoprime_new.nnz = dunoprime_new_index;

		gsens2_parallel(&dprime, &dunoprime_new, &dzeroprime_new, &dunoprime_d, &dzeroprime_d, &matB_d, &matC_d, &invu_d, invu_full_d, diff_d, matB_full_d, matC_full_d, h_inf_peak_frq, cublas_handle);
		
		df0dx[i] = us * dprime*vs;
		//printf("%f\n", (*df0dx)[i]);

	}

	cublasDestroy(cublas_handle);
	
	free_pointers_gsens2(&dunoprime_d, &dzeroprime_d, &matB_d, &matC_d, &invu_d, invu_full_d, diff_d, matB_full_d, matC_full_d);

	free(dunoprime.val);
	free(dunoprime.row);
	free(dunoprime.col);
	free(dzeroprime.val);
	free(dzeroprime.row);
	free(dzeroprime.col);
	free(dunoprime_new.val);
	free(dunoprime_new.row);
	free(dunoprime_new.col);
	free(dzeroprime_new.val);
	free(dzeroprime_new.row);
	free(dzeroprime_new.col);
	free(dofvec);
	free(indici);
	free(diff_full);
}

void init_pointers_gsens2(struct sparse *dunoprime_h, struct sparse *dzeroprime_h, struct sparse *matB_h, struct sparse *matC_h, struct sparse *invu_h,
	struct sparse *dunoprime_d, struct sparse *dzeroprime_d, struct sparse *matB_d, struct sparse *matC_d, struct sparse *invu_d,
	double **invu_full_d, double **diff_d, double **matB_full_d, double **matC_full_d) {

	int size_x;

	dunoprime_d->nrow = dunoprime_h->nrow;
	dunoprime_d->ncol = dunoprime_h->ncol;

	dzeroprime_d->nrow = dzeroprime_h->nrow;
	dzeroprime_d->ncol = dzeroprime_h->ncol;

	matB_d->nrow = matB_h->nrow;
	matB_d->ncol = matB_h->ncol;
	matB_d->nnz = matB_h->nnz;

	matC_d->nrow = matC_h->nrow;
	matC_d->ncol = matC_h->ncol;
	matC_d->nnz = matC_h->nnz;

	invu_d->nrow = invu_h->nrow;
	invu_d->ncol = invu_h->ncol;
	invu_d->nnz = invu_h->nnz;

	cudaMalloc((void**)diff_d, dzeroprime_d->nrow * dzeroprime_d->ncol * sizeof(double));

	cudaMalloc((void**)&matB_d->col, matB_d->nnz * sizeof(int));
	cudaMalloc((void**)&matB_d->val, matB_d->nnz * sizeof(double));
	cudaMalloc((void**)&matB_d->row, matB_d->nnz * sizeof(int));

	cudaMalloc((void**)&matC_d->row, matC_d->nnz * sizeof(int));
	cudaMalloc((void**)&matC_d->col, matC_d->nnz * sizeof(int));
	cudaMalloc((void**)&matC_d->val, matC_d->nnz * sizeof(double));

	cudaMalloc((void**)matC_full_d, matC_d->nrow*matC_d->ncol * sizeof(double));
	cudaMalloc((void**)matB_full_d, matB_d->nrow*matB_d->ncol * sizeof(double));

	cudaMalloc((void**)&invu_d->row, invu_d->nnz * sizeof(int));
	cudaMalloc((void**)&invu_d->col, invu_d->nnz * sizeof(int));
	cudaMalloc((void**)&invu_d->val, invu_d->nnz * sizeof(double));

	cudaMalloc((void**)invu_full_d, invu_d->nrow*invu_d->ncol * sizeof(double));

	//---------------------------------------------------------------------------------

	cudaMemcpy(matB_d->row, matB_h->row, matB_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matB_d->col, matB_h->col, matB_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matB_d->val, matB_h->val, matB_d->nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(matC_d->row, matC_h->row, matC_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matC_d->col, matC_h->col, matC_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matC_d->val, matC_h->val, matC_d->nnz * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(invu_d->row, invu_h->row, invu_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(invu_d->col, invu_h->col, invu_d->nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(invu_d->val, invu_h->val, invu_d->nnz * sizeof(double), cudaMemcpyHostToDevice);

	//---------------------converting C sparse in -C dense-----------------

	cudaMemset((*matC_full_d), 0, matC_d->nrow*matC_d->ncol * sizeof(double));
	checkCudaError(__LINE__);

	if (matC_d->nnz % THREADS == 0)
		size_x = matC_d->nnz / THREADS;
	else
		size_x = (matC_d->nnz / THREADS) + 1;
	dim3 threads2(THREADS, 1, 1);
	dim3 blocks2(size_x, 1, 1);

	sparse2full_kernel << <blocks2, threads2 >> > ((*matC_full_d), matC_d->row, matC_d->col, matC_d->val, matC_d->ncol, matC_d->nnz, -1);

	//---------------------converting B sparse in B dense-----------------

	cudaMemset((*matB_full_d), 0, matB_d->nrow*matB_d->ncol * sizeof(double));

	if (matB_d->nnz % THREADS == 0)
		size_x = matB_d->nnz / THREADS;
	else
		size_x = (matB_d->nnz / THREADS) + 1;
	dim3 threads3(THREADS, 1, 1);
	dim3 blocks3(size_x, 1, 1);

	sparse2full_kernel << <blocks3, threads3 >> > ((*matB_full_d), matB_d->row, matB_d->col, matB_d->val, matB_d->ncol, matB_d->nnz, 1);

	//--------------------converting INVU sparse in INVU dense-----------------
	cudaMemset((*invu_full_d), 0, invu_d->nrow*invu_d->ncol * sizeof(double));
	checkCudaError(__LINE__);

	if (invu_d->nnz % THREADS == 0)
		size_x = invu_d->nnz / THREADS;
	else
		size_x = (invu_d->nnz / THREADS) + 1;
	dim3 threads4(THREADS, 1, 1);
	dim3 blocks4(size_x, 1, 1);

	sparse2full_kernel << <blocks4, threads4 >> > ((*invu_full_d), invu_d->row, invu_d->col, invu_d->val, invu_d->ncol, invu_d->nnz, 1);
	cudaDeviceSynchronize();

}

void free_pointers_gsens2(struct sparse *dunoprime_d, struct sparse *dzeroprime_d, struct sparse *matB_d, struct sparse *matC_d, struct sparse *invu_d,
	double *invu_full_d, double* diff_d, double *matB_full_d, double *matC_full_d) {

	cudaFree(matB_d->row);
	cudaFree(matB_d->col);
	cudaFree(matB_d->val);

	cudaFree(matC_d->row);
	cudaFree(matC_d->col);
	cudaFree(matC_d->val);

	cudaFree(invu_d->row);
	cudaFree(invu_d->col);
	cudaFree(invu_d->val);

	cudaFree(matB_full_d);
	cudaFree(matC_full_d);
	cudaFree(diff_d);
	cudaFree(invu_full_d);
}
