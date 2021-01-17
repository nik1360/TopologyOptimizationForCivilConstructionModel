#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "sparse_struc.cuh"

void matrix_sum(int *mat1, int *mat2, int*sum, int nrow, int ncol) {
	int mat_index;
	for (int row_index = 0; row_index < nrow; row_index++) {
		for (int col_index = 0; col_index < ncol; col_index++) {
			mat_index = row_index *ncol + col_index;
			sum[mat_index] = mat1[mat_index] + mat2[mat_index];
		}
	}
}

int max_value(int*mat, int nelem) {
	int max = mat[0];
	for (int i = 1;i < nelem;i++) {
		if (mat[i] > max) 
			max = mat[i];
	}
	return max;
}

void linearize_matrix(double *mat, double *lin_mat,int nrow, int ncol) {
	int lin_index=0, mat_index;
	for (int col_index = 0;col_index <ncol;col_index++) {
		for (int row_index = 0;row_index < nrow ;row_index++) {
			mat_index = row_index*ncol + col_index;
			lin_mat[lin_index] = mat[mat_index];
			lin_index++;
		}
	}
}

void prepare_sparse(int *row, int*col, double *val, struct sparse *s, int size) {
	double  *val2;
	int *row2, *col2, index=0, nonzero_count = 0, count, found;
	
	row2 = (int*)malloc(size * sizeof(int));  
	col2 = (int*)malloc(size * sizeof(int));  
	val2 = (double*)malloc(size * sizeof(double));

	count = 0;
	row2[count] = row[0];
	col2[count] = col[0];
	val2[count] = val[0];
	count++;

	
	for (int i = 1;i < size;i++) {
		found = 0;
		for (int j = 0;j < count;j++) {
			if ((row[i] == row2[j]) && (col[i] == col2[j])) {
				found = 1;
				index = j;
			}
		}
		if (found == 1) {
			val2[index] += val[i];
		}
		else {
			row2[count] = row[i];
			col2[count] = col[i];
			val2[count] = val[i];
			count++;
		}
	}

	nonzero_count = 0;
	for (int i = 0; i < count; i++) {
		if (fabs(val2[i]) >pow(10, -9)) {
			nonzero_count++;
		}
	}

	s->row = (int*)malloc(nonzero_count * sizeof(int));
	s->col = (int*)malloc(nonzero_count * sizeof(int));
	s->val = (double*)malloc(nonzero_count * sizeof(double));

	nonzero_count = 0;
	for (int i = 0; i < count; i++) {
		if (fabs(val2[i]) >pow(10, -9)) {
			s->row[nonzero_count] = row2[i];
			s->col[nonzero_count] = col2[i];
			s->val[nonzero_count] = val2[i];
			nonzero_count++;
		}
	}

	s->nnz = nonzero_count;


	free(row2);
	free(col2);
	free(val2);
	
}

void sparse_matrix_product(struct sparse *a, struct sparse *b, struct sparse *p) {

	struct sparse tmp;
	int count = 0, found = 0;
	int prod_index, tmp_index, tmp_dim ;

	if (a->ncol == b->nrow) {
		

		p->nrow = a->nrow;
		p->ncol = b->ncol;
		p->nnz = 0;

		for (int a_index = 0; a_index < a->nnz; a_index++) {
			for (int b_index = 0; b_index < b->nnz; b_index++) {
				if ((a->col[a_index] == b->row[b_index])) {
					count++;
				}
			}
		}
		
		
		tmp_dim = count;

		tmp.row = (int*)malloc(tmp_dim * sizeof(int));
		tmp.col = (int*)malloc(tmp_dim * sizeof(int));
		tmp.val = (double*)malloc(tmp_dim * sizeof(double));

		tmp_index = 0;
		for (int a_index = 0; a_index < a->nnz; a_index++) {
			for (int b_index = 0; b_index < b->nnz; b_index++) {
				if ((a->col[a_index] == b->row[b_index])) {
					if (count == 0) {
						tmp.row[tmp_index] = a->row[a_index];
						tmp.col[tmp_index] = b->col[b_index];
						tmp.val[tmp_index] = a->val[a_index] * b->val[b_index];
						tmp_index++;
					}
					else {
						found = 0;
						for (int p_index = 0; p_index < tmp_dim; p_index++) {
							if ((tmp.row[p_index] == a->row[a_index]) && tmp.col[p_index] == b->col[b_index]) {
								found = 1;
								tmp.val[p_index] += a->val[a_index] * b->val[b_index];

							}
						}
						if (found == 0) {
							tmp.row[tmp_index] = a->row[a_index];
							tmp.col[tmp_index] = b->col[b_index];
							tmp.val[tmp_index] = a->val[a_index] * b->val[b_index];
							tmp_index++;
						}
					}
				}
			}
		}
		
		for (int i = 0; i < tmp_dim; i++) {
			if (fabs(tmp.val[i])>pow(10,-9)) {
				p->nnz++;
			}
		}

		p->row = (int*)malloc(p->nnz * sizeof(int));
		p->col = (int*)malloc(p->nnz * sizeof(int));
		p->val = (double*)malloc(p->nnz * sizeof(double));

		prod_index = 0;
		for (int i = 0; i < count; i++) {
			if (fabs(tmp.val[i])>pow(10, -9)) {
				p->row[prod_index] = tmp.row[i];
				p->col[prod_index] = tmp.col[i];
				p->val[prod_index] = tmp.val[i];
				prod_index++;
			}
		}

		free(tmp.row);
		free(tmp.col);
		free(tmp.val);
	}
	else {
		printf("Matrici non compatibili\n");
	}
}

void sparse_to_dense(struct sparse *sparse_m, double **dense_m) {
	(*dense_m) = (double*)malloc(sparse_m->nrow*sparse_m->ncol * sizeof(double));
	int index;

	for (int i = 0;i < sparse_m->nrow*sparse_m->ncol;i++) {
		(*dense_m)[i] = 0;
	}

	for (int i = 0;i < sparse_m->nnz;i++) {
		index = (sparse_m->row[i] - 1)*sparse_m->ncol + (sparse_m->col[i] - 1);  //row_i*ncol + col_i
		(*dense_m)[index] = sparse_m->val[i];
	}
}

void dense_to_sparse(double *dense_m, int nrow, int ncol, struct sparse *sparse_m) {
	
	int s_index = 0,d_index;
	
	for (int i = 0;i < nrow;i++) {
		for (int j = 0;j < ncol;j++) {
			if (fabs(dense_m[i*ncol + j]) > pow(10, -10)) {
				s_index++;
			}
		}
	}

	sparse_m->nrow = nrow;
	sparse_m->ncol = ncol;
	sparse_m->nnz = s_index;
	sparse_m->row = (int*)malloc(sparse_m->nnz * sizeof(int));
	sparse_m->col = (int*)malloc(sparse_m->nnz * sizeof(int));
	sparse_m->val = (double*)malloc(sparse_m->nnz * sizeof(double));

	s_index = 0;
	for (int i = 0;i < nrow;i++) {
		for (int j = 0;j < ncol;j++) {
			d_index = i*ncol + j;
			if (fabs(dense_m[d_index]) > pow(10, -10)) {
				sparse_m->row[s_index] = i + 1;
				sparse_m->col[s_index] = j + 1;
				sparse_m->val[s_index] = dense_m[d_index];
				s_index++;
			}
		}
	}
}

void matrix_inverse(double *Min, double *Mout, int actualsize) {
	/* This function calculates the inverse of a square matrix
	*
	* matrix_inverse(double *Min, double *Mout, int actualsize)
	*
	* Min : Pointer to Input square Double Matrix
	* Mout : Pointer to Output (empty) memory space with size of Min
	* actualsize : The number of rows/columns
	*
	* Notes:
	*  - the matrix must be invertible
	*  - there's no pivoting of rows or columns, hence,
	*        accuracy might not be adequate for your needs.
	*
	*/

	/* Loop variables */
	int i, j, k;
	/* Sum variables */
	double sum, x;

	/*  Copy the input matrix to output matrix */
	for (i = 0; i<actualsize*actualsize; i++) { Mout[i] = Min[i]; }

	/* Add small value to diagonal if diagonal is zero */
	for (i = 0; i<actualsize; i++)
	{
		j = i*actualsize + i;
		if ((Mout[j]<1e-12) && (Mout[j]>-1e-12)) { Mout[j] = 1e-12; }
	}

	/* Matrix size must be larger than one */
	if (actualsize <= 1) return;

	for (i = 1; i < actualsize; i++) {
		Mout[i] /= Mout[0]; /* normalize row 0 */
	}

	for (i = 1; i < actualsize; i++) {
		for (j = i; j < actualsize; j++) { /* do a column of L */
			sum = 0.0;
			for (k = 0; k < i; k++) {
				sum += Mout[j*actualsize + k] * Mout[k*actualsize + i];
			}
			Mout[j*actualsize + i] -= sum;
		}
		if (i == actualsize - 1) continue;
		for (j = i + 1; j < actualsize; j++) {  /* do a row of U */
			sum = 0.0;
			for (k = 0; k < i; k++) {
				sum += Mout[i*actualsize + k] * Mout[k*actualsize + j];
			}
			Mout[i*actualsize + j] = (Mout[i*actualsize + j] - sum) / Mout[i*actualsize + i];
		}
	}
	for (i = 0; i < actualsize; i++)  /* invert L */ {
		for (j = i; j < actualsize; j++) {
			x = 1.0;
			if (i != j) {
				x = 0.0;
				for (k = i; k < j; k++) {
					x -= Mout[j*actualsize + k] * Mout[k*actualsize + i];
				}
			}
			Mout[j*actualsize + i] = x / Mout[j*actualsize + j];
		}
	}
	for (i = 0; i < actualsize; i++) /* invert U */ {
		for (j = i; j < actualsize; j++) {
			if (i == j) continue;
			sum = 0.0;
			for (k = i; k < j; k++) {
				sum += Mout[k*actualsize + j] * ((i == k) ? 1.0 : Mout[i*actualsize + k]);
			}
			Mout[i*actualsize + j] = -sum;
		}
	}
	for (i = 0; i < actualsize; i++) /* final inversion */ {
		for (j = 0; j < actualsize; j++) {
			sum = 0.0;
			for (k = ((i>j) ? i : j); k < actualsize; k++) {
				sum += ((j == k) ? 1.0 : Mout[j*actualsize + k])*Mout[k*actualsize + i];
			}
			Mout[j*actualsize + i] = sum;
		}
	}
}

void matrix_diff(double *a, double *b, double **diff, int a_nrow, int a_ncol, int b_nrow, int b_ncol) {
	
	int index;
	if ((a_nrow==b_nrow)&&(a_ncol==b_ncol)) {
		(*diff) = (double*)malloc(a_nrow*a_ncol * sizeof(double));
		for (int i = 0;i < a_nrow;i++) {
			for (int j = 0;j < a_ncol;j++) {
				index = i*a_ncol + j;
				(*diff)[index] = a[index] - b[index];
			}
		}
	}
	else
		printf("ERROR!\n");
}

void sparse_diff(struct sparse *s1, struct sparse *s2, struct sparse *res) {
	int found, count = 0;
	for (int i = 0; i < s1->nnz; i++) {
		found = 0;
		for (int j = 0; j < s2->nnz; j++) {
			if ((s1->row[i] == s2->row[j]) && (s1->col[i] == s2->col[j])) {
				found = 1;
				if ((s1->val[i] - s2->val[j]) != 0)
				{
					count++;
				}
				break;
			}
		}
		if (found == 0) {
			count++;
		}
	}
	for (int i = 0; i < s2->nnz; i++) {
		found = 0;
		for (int j = 0; j < s1->nnz; j++) {
			if ((s2->row[i] == s1->row[j]) && (s2->col[i] == s1->col[j])) {
				found = 1;
			}
		}
		if (found == 0) {
			count++;
		}
	}

	res->nrow = s1->nrow;
	res->ncol = s1->ncol;
	res->nnz = count;
	res->row = (int*)malloc(res->nnz * sizeof(int));
	res->col = (int*)malloc(res->nnz * sizeof(int));
	res->val = (double*)malloc(res->nnz * sizeof(double));

	count = 0;
	for (int i = 0; i < s1->nnz; i++) {
		found = 0;
		for (int j = 0; j < s2->nnz; j++) {
			if ((s1->row[i] == s2->row[j]) && (s1->col[i] == s2->col[j])) {
				found = 1;
				if (fabs(s1->val[i] - s2->val[j]) > pow(10, -9))
				{
					res->row[count] = s1->row[i];
					res->col[count] = s1->col[i];
					res->val[count] = (s1->val[i] - s2->val[j]);
					count++;
				}
				break;
			}
		}
		if (found == 0) {
			res->row[count] = s1->row[i];
			res->col[count] = s1->col[i];
			res->val[count] = s1->val[i];
			count++;
		}
	}
	for (int i = 0; i < s2->nnz; i++) {
		found = 0;
		for (int j = 0; j < s1->nnz; j++) {
			if ((s2->row[i] == s1->row[j]) && (s2->col[i] == s1->col[j])) {
				found = 1;
			}
		}
		if (found == 0) {
			res->row[count] = s2->row[i];
			res->col[count] = s2->col[i];
			res->val[count] = -s2->val[i];
			count++;
		}
	}
}