#pragma once
#include "mesh.cuh"
#include "material.cuh"

void mesh_material_print(struct mesh *me, struct material *ma);
void print_mat_int(int *mat, int nrow, int ncol);
void print_mat_double(double *mat, int nrow, int ncol);
void print_vec_double(double *vec, int nelem);
void print_vec_int(int *vec, int nelem);