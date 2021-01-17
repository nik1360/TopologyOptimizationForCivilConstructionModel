#include <stdio.h>
#include <stdlib.h>
#include "mesh.cuh"
#include "material.cuh"

void print_mesh_details(struct mesh *m);
void print_material_details(struct material *mat);


void mesh_material_print(struct mesh *me, struct material *ma) {
	printf("MESH PROPERTIES: \n");
	print_mesh_details(me);
	printf("MATERIAL PROPERTIES: \n");
	print_material_details(ma);
}

void print_mesh_details(struct mesh *m) {
	printf("lx:%f ly:%f vmax:%f\n", m->lx, m->ly, m->vmax);
	printf("nelx:%d nely:%d volfrac:%f\n", m->nelx, m->nely, m->volfrac);
	printf("ax:%f by:%f area:%f\n", m->ax, m->by, m->area);
	printf("penal:%f prho:%f rmin:%f ft:%d\n", m->penal, m->prho, m->rmin, m->ft);
	printf("alpha:%f beta:%f ninp:%d nout:%d\n", m->alpha, m->beta, m->ninp, m->nout);
}

void print_material_details(struct material *mat) {
	printf("e0:%f emin:%f rho0:%f rhomin:%f nu:%f\n", mat->e0, mat->emin, mat->rho0, mat->rhomin, mat->nu);
}

void print_mat_int(int *mat, int nrow, int ncol) {
	for (int row_index = 0; row_index < nrow; row_index++) {
		for (int col_index = 0;col_index < ncol; col_index++) {
			printf("%d ", mat[row_index*ncol + col_index]);
		}
		printf("\n");
	}
}

void print_mat_double(double *mat, int nrow, int ncol) {
	for (int row_index = 0; row_index < nrow; row_index++) {
		for (int col_index = 0;col_index < ncol; col_index++) {
			printf("%f ", mat[row_index*ncol + col_index]);
		}
		printf("\n");
	}
}

void print_vec_double(double *vec, int nelem) {
	for (int i = 0;i < nelem;i++) {
		printf("%.15f\n ",vec[i]);
	}
	printf("\n");
}

void print_vec_int(int *vec, int nelem) {
	for (int i = 0;i < nelem;i++) {
		printf("%d\n ", vec[i]);
	}
	printf("\n");
}
