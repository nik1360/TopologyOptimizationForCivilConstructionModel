#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "mesh.cuh"
#include "matrix_functions.cuh"
#include "sparse_struc.cuh"
#include "minmax.cuh"

//--------------------------------------------PROTOTIPES--------------------------
void ih_jh_sh_compute(int *ih, int *jh, double *sh, struct mesh *mesh);
void hs_compute(struct sparse *h, struct mesh *mesh, double *hs);

//--------------------------------------------BODIES------------------------------
void prepare_filter(int **ih, int **jh, double **sh, struct mesh *mesh, double **hs, struct sparse *h) {
	int size =(int)mesh->nelx*mesh->nely * (int)pow((2 * (mesh->rmin - 1) + 1), 2);

	(*ih) = (int*)malloc(size * sizeof(int));
	(*jh) = (int*)malloc(size * sizeof(int));
	(*sh) = (double*)malloc(size * sizeof(double));
	
	ih_jh_sh_compute((*ih), (*jh), (*sh), mesh);
	prepare_sparse((*ih), (*jh), (*sh), h, size);
	
	(*hs) = (double*)malloc((max_value(h->row,h->nnz)* sizeof(double)));
	hs_compute(h, mesh, (*hs));
}

void ih_jh_sh_compute(int *ih, int *jh, double *sh, struct mesh *mesh) {
	int size, e1, e2, k, i2_max, i2_min, j2_max, j2_min;
	
	
	size = (int) mesh->nelx*mesh->nely * (int)pow((2 * (mesh->rmin - 1) + 1), 2);
	
	for (int i = 0; i < size; i++) 
		ih[i] = 1;
	for (int i = 0; i < size; i++) 
		jh[i] = 1;
	for (int i = 0; i < size; i++) 
		sh[i] = 0;
	
	k = 0;
	for (int i1 = 0; i1 < mesh->nelx; i1++) {
		for (int j1 = 0; j1 < mesh->nely; j1++) {
			e1 = i1*mesh->nely + (j1 + 1);

			i2_min = (int)maxVal((i1 + 1) - (mesh->rmin - 1), 1);
			i2_max = (int)minVal((i1 + 1) + (mesh->rmin - 1), mesh->nelx);
			j2_min = (int)maxVal((j1 + 1) - (mesh->rmin - 1), 1);
			j2_max = (int)minVal((j1 + 1) + (mesh->rmin - 1), mesh->nely);

			for (int i2 = (i2_min - 1);i2 < i2_max;i2++) {
				for (int j2 = (j2_min - 1);j2 < j2_max; j2++) {
					e2 = i2*mesh->nely + (j2 + 1);
					ih[k] = e1;
					jh[k] = e2;
					sh[k] = maxVal(0, mesh->rmin - sqrt(pow(i1 - i2, 2) + pow(j1 - j2, 2)));
					k++;
				}
			}
		}
	}
}

void hs_compute(struct sparse *h, struct mesh *mesh, double *hs) {
	int k=0, size = h->nnz;
	double tmp_sum = 0;
	
	tmp_sum = h->val[0];
	for (int i = 1; i <= size;i++) {
		if (h->row[i] == h->row[i-1])
			tmp_sum += h->val[i];
		else {
			hs[k] = tmp_sum;
			tmp_sum = h->val[i];
			k++;
		}
	}

}

