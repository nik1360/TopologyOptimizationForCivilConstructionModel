#pragma once

struct sparse {
	double *val;
	int *row;
	int *col;
	int nrow;
	int ncol;
	int nnz;

	int *csr_row; 
};