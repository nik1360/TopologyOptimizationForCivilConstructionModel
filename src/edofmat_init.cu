#include <stdlib.h>

#include "constants.cuh"
#include "mesh.cuh"
#include "matrix_functions.cuh"

//----------------------------------------------PROTOTIPES--------------------------------------
void create_noderns(int *noderns, struct mesh *mesh);
void create_edofvec(int *edofvec, int *noderns, struct mesh *mesh);
void create_repmat1(int *repmat, int *edofvec, struct mesh *mesh);
void create_repmat2(int *repmat, struct mesh *mesh);

//----------------------------------------------BODIES------------------------------------------
void edofmat_init(int **edofmat, struct mesh *mesh) {
	int *noderns,*edofvec,*repmat1,*repmat2;
	noderns = (int*)malloc((1 + mesh->nely)*(1 + mesh->nelx) * sizeof(int));
	edofvec = (int*)malloc((mesh->nely)*(mesh->nelx) * sizeof(int));
	repmat1 = (int*)malloc((mesh->nely)*(mesh->nelx) * EDOFMAT_COL * sizeof(int));
	repmat2 = (int*)malloc((mesh->nely)*(mesh->nelx) * EDOFMAT_COL * sizeof(int));
	(*edofmat) = (int*)malloc(mesh->nelx * mesh->nely * EDOFMAT_COL * sizeof(int));
	
	create_noderns(noderns, mesh);
	create_edofvec(edofvec, noderns, mesh);
	create_repmat1(repmat1, edofvec, mesh);
	create_repmat2(repmat2, mesh);

	matrix_sum(repmat1, repmat2, (*edofmat), mesh->nelx*mesh->nely, 8);	//edofMat = repmat(edofVec, 1, 8) + repmat([0 1 2 * nely + [2 3 0 1] - 2 - 1], nelx*nely, 1);

	free(noderns);
	free(edofvec);
	free(repmat1);
	free(repmat2);
}


void create_noderns(int *noderns, struct mesh *mesh) {		//nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
	int count=1;
	for (int col_index = 0;col_index < (1+mesh->nelx);col_index++) {
		for (int row_index = 0;row_index <(1+mesh->nely);row_index++) {
			noderns[col_index + (1 + mesh->nelx)*row_index] = count;
			count++;
		}
	}
}

void create_edofvec(int *edofvec, int *noderns, struct mesh *mesh) {	//edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
	int value;
	int edofvec_index=0;
	for (int col_index = 0;col_index <  mesh->nelx; col_index++) {
		for (int row_index = 0;row_index <mesh->nely;row_index++) {
			value = (noderns[col_index + (1 + mesh->nelx)*row_index]*2) + 1;
			edofvec[edofvec_index]=value;
			edofvec_index++;
		}
	}
}

void create_repmat1(int *repmat, int *edofvec, struct mesh *mesh) {		//repmat(edofVec,1,8)
	int mat_index,edofvec_index=0;
	for (int col_index = 0;col_index < 8; col_index++) {
		for (int row_index = 0;row_index <mesh->nely*mesh->nelx;row_index++) {
			mat_index = col_index + (row_index * 8);
			repmat[mat_index] = edofvec[edofvec_index];
			edofvec_index++;
		}
		edofvec_index = 0;
	}
}

void create_repmat2(int *repmat, struct mesh *mesh) {	//repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1)
	int mat_index, vec_index = 0;
	int *vec;
	
	vec = (int*)malloc(8 * sizeof(int));
	
	vec[0] = 0;
	vec[1] = 1;
	vec[2] = 2 * mesh->nely + 2;
	vec[3] = 2 * mesh->nely + 3;
	vec[4] = 2 * mesh->nely + 0;
	vec[5] = 2 * mesh->nely + 1;
	vec[6] = -2;
	vec[7] = -1;

	for (int row_index = 0; row_index < mesh->nelx*mesh->nely; row_index++) {
		for (int col_index = 0; col_index < 8; col_index++) {
			mat_index = row_index * 8 + col_index;
			repmat[mat_index] = vec[vec_index];
			vec_index++;
		}
		vec_index = 0;
	}

	free(vec);
}