#include <stdlib.h>

#include "mesh.cuh"
#include "constants.cuh"


//------------------------------------------PROTOTIPES----------------------------------------------
void define_freedofs(int *fixeddofs, int *alldofs, int *freedofs, struct mesh *mesh);

//-----------------------------------------BODIES-------------------------------------------
void define_loads_support(int **fixeddofs,int **alldofs, int **freedofs, struct mesh *mesh) {
	
	int index;
	
	mesh->fixed_count = 0;
	for (int i = 1; i <= 2*(mesh->nely + 1); i += 2) {
		mesh->fixed_count++;
	}
	mesh->fixed_count++;

	(*fixeddofs) = (int*)malloc((mesh->fixed_count) * sizeof(int));
	(*alldofs) = (int*)malloc(2 * (mesh->nelx + 1)*(mesh->nely + 1) * sizeof(int));
	(*freedofs) = (int*)malloc(((2 * (mesh->nelx + 1)*(mesh->nely + 1)) - mesh->fixed_count) * sizeof(int));
	
	//define fixeddofs
	//fixeddofs = union([1:2:2*(nely+1)],[2*(nelx+1)*(nely+1)]);
	index = 0;
	for (int i = 1; i <= 2 * (mesh->nely + 1); i += 2) {
		(*fixeddofs)[index] = i;
		index++;
	}

	(*fixeddofs)[mesh->fixed_count-1] = 2 * (mesh->nelx + 1)*(mesh->nely + 1);
	
	//define alldofs
	//alldofs = [1:2*(nely+1)*(nelx+1)];
	for (int i = 0; i < (mesh->nely + 1)*(mesh->nelx + 1) * 2; i++) {
		(*alldofs)[i] = i + 1;
	}

	define_freedofs((*fixeddofs), (*alldofs), (*freedofs), mesh);
}


void define_freedofs(int *fixeddofs, int *alldofs, int *freedofs, struct mesh *mesh) {
	int free_flag,free_index=0;
	
	//freedofs = setdiff(alldofs,fixeddofs);
	for (int all_index = 0; all_index < (mesh->nely + 1)*(mesh->nelx + 1) * 2; all_index++) {
		free_flag = 1;
		for (int fixed_index = 0; fixed_index < mesh->fixed_count;fixed_index++) {
			if (alldofs[all_index]==fixeddofs[fixed_index]) {
				free_flag = 0;
			}
		}
		if (free_flag == 1) {
			freedofs[free_index] = alldofs[all_index];
			free_index++;
		}
	}
}