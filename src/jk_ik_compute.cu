#include <stdlib.h>

#include "mesh.cuh"
#include "constants.cuh"

//-------------------------------PROTOTIPES---------------------------
void ik_compute(int* edofmat, int *ik, struct mesh *mesh);
void jk_compute(int* edofmat, int *jk, struct mesh *mesh);


//------------------------------BODIES--------------------------------
void jk_ik_compute(int **ik,int **jk, int* edofmat, struct mesh *mesh) {
	
	(*ik) = (int*)malloc(mesh->nelx*mesh->nely * 64 * sizeof(int));
	(*jk) = (int*)malloc(mesh->nelx*mesh->nely * 64 * sizeof(int));

	ik_compute(edofmat, (*ik), mesh);
	jk_compute(edofmat, (*jk), mesh);
	
}


void ik_compute(int* edofmat, int *ik, struct mesh *mesh) {		//iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);

	int e_index, tmp_index = 0, ik_index = 0;
	int *tmp ;
	tmp = (int*)malloc(mesh->nely*mesh->nelx * sizeof(int));
	
	for (int e_row_index = 0; e_row_index < mesh->nely*mesh->nelx;e_row_index++) {
		for (int e_col_index = 0;e_col_index < EDOFMAT_COL;e_col_index++) {
			e_index = e_row_index * EDOFMAT_COL + e_col_index;
			tmp[e_col_index] = edofmat[e_index];
		}
		for (int i = 0;i < 8;i++) {
			for (tmp_index = 0;tmp_index <EDOFMAT_COL;tmp_index++) {
				ik[ik_index] = tmp[tmp_index];
				ik_index++;
			}
		}
	}
	free(tmp);
}

void jk_compute(int* edofmat, int *jk, struct mesh *mesh) {	//jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
	int e_index, jk_index = 0;

	for (int e_row_index = 0; e_row_index < mesh->nely*mesh->nelx; e_row_index++) {
		for (int e_col_index = 0; e_col_index < EDOFMAT_COL; e_col_index++) {
			e_index = e_row_index * EDOFMAT_COL + e_col_index;
			for (int i = 0; i < EDOFMAT_COL;i++) {
				jk[jk_index] = edofmat[e_index];
				jk_index++;
			}
		}
	}
}