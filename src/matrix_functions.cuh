#pragma once
#include "mesh.cuh"
#include "material.cuh"
#include "sparse_struc.cuh"

void matrix_sum(int *mat1, int *mat2, int*sum, int nrow, int ncol);
int max_value(int*mat, int nelem);
void linearize_matrix(double *mat, double *lin_mat, int nrow, int ncol);
void prepare_sparse(int *row, int*col, double *val,struct sparse *s, int size);
void sparse_matrix_product(struct sparse *a, struct sparse *b, struct sparse *p);
void sparse_to_dense(struct sparse *sparse_m, double **dense_m);
void dense_to_sparse(double *dense_m, int nrow, int ncol, struct sparse *sparse_m);
void matrix_inverse(double *Min, double *Mout, int actualsize);
void matrix_diff(double *a, double *b, double **diff, int a_nrow, int a_ncol, int b_nrow, int b_ncol);
void sparse_diff(struct sparse *s1, struct sparse *s2, struct sparse *res);