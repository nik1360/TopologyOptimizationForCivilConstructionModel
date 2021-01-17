#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "constants.cuh"
#include "matrix_functions.cuh"
#include "sys.cuh"


void invforsens2(struct sparse *invu, struct sys *g, double h_inf_peak_frq) {
	double *e_a_diff_full, *inv;

	struct sparse *e_tmp;
	struct sparse *diff;

	e_tmp = (struct sparse*)malloc(sizeof(struct sparse));
	diff = (struct sparse*)malloc(sizeof(struct sparse));
	
	inv = (double*)malloc(g->e.nrow*g->e.ncol * sizeof(double));
	
	e_tmp->nnz = g->e.nnz;
	e_tmp->nrow = g->e.nrow;
	e_tmp->ncol = g->e.ncol;

	e_tmp->row = (int*)malloc(e_tmp->nnz * sizeof(int));
	e_tmp->col = (int*)malloc(e_tmp->nnz * sizeof(int));
	e_tmp->val = (double*)malloc(e_tmp->nnz * sizeof(double));

	for (int j = 0; j < g->e.nnz; j++) {				//sE
		e_tmp->row[j] = g->e.row[j];
		e_tmp->col[j] = g->e.col[j];
		e_tmp->val[j] = h_inf_peak_frq * g->e.val[j];
	}

	
	sparse_diff(e_tmp, &g->a, diff);	//sE - A
	free(e_tmp->row);
	free(e_tmp->col);
	free(e_tmp->val);

	sparse_to_dense(diff, &e_a_diff_full);
	
	matrix_inverse(e_a_diff_full, inv, g->e.nrow); //inv(sE-A)
	dense_to_sparse(inv, g->e.nrow, g->e.ncol, invu);

	
	free(e_a_diff_full);
	free(inv);

	free(diff->row);
	free(diff->col);
	free(diff->val);
	free(e_tmp);
	free(diff);
}