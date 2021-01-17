
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "constants.cuh"
#include "main_functions.cuh"
#include "mesh.cuh"
#include "material.cuh"
#include "sparse_struc.cuh"




int main(int argc, char **argv) {
	struct mesh mesh;
	struct material material;
	double *ke, *me; //FINITE ELEMENT ANALYSIS
	int *edofmat, *ik, *jk; //FINITE ELEMENT ANALYSIS
	struct sparse f; //LOADS AND SUPPORT
	int *fixeddofs, *alldofs, *freedofs; //LOADS AND SUPPORT
	struct sparse h;
	int *ih, *jh; //FILTER
	double *sh, *hs; //FILTER
	
	clock_t start_mma, start1, start2, start3, start4, start5, start6, stop_mma, stop1, stop2,stop3,stop4,stop5, stop6 ;
	double tempo_tot,t1,t2,t3,t4,t5,t6;

	
	
	
	//MATERIAL AND MESH PROPERTIES
	start1 = clock();
	mesh_material_init(&mesh, &material);
	stop1 = clock();
	
	//PREPARE FINITE ELEMENT ANALYSIS
	start2 = clock();
	stiffness_mass_init(&ke, &me, &mesh, &material);
	stop2 = clock();
	start3 = clock();
	edofmat_init(&edofmat,&mesh);
	stop3 = clock();
	start4 = clock();
	jk_ik_compute(&ik,&jk,edofmat, &mesh);
	stop4 = clock();
	//DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
	f.row = (int*)malloc((2 * (mesh.nely + 1)*(mesh.nelx + 1)) * sizeof(int));
	f.col = (int*)malloc((2 * (mesh.nely + 1)*(mesh.nelx + 1)) * sizeof(int));
	f.val = (double*)malloc((2 * (mesh.nely + 1)*(mesh.nelx + 1)) * sizeof(double));

	f.nnz=1;
	f.row[0] = 2;	
	f.col[0] = 1;
	f.val[0] = -1;
	
	start5 = clock();
	define_loads_support(&fixeddofs,&alldofs,&freedofs, &mesh);	
	stop5 = clock();

	//PREPARE FILTER
	start6 = clock();
	prepare_filter(&ih, &jh, &sh, &mesh, &hs, &h);
	stop6 = clock();

	//MMA
	start_mma = clock();
	mma(&h,hs,&mesh,&material,ik,jk,me,ke,freedofs,fixeddofs,&f, edofmat);
	stop_mma = clock();
	
	free(ke);
	free(me);
	free(edofmat);
	free(ik);
	free(jk);
	free(fixeddofs);
	free(alldofs);
	free(freedofs);
	free(ih);
	free(jh);
	free(sh);
	free(hs);

	free(h.val);
	free(h.row);
	free(h.col);

	free(f.val);
	free(f.row);
	free(f.col);

	
	tempo_tot = ((double)stop_mma - start_mma) / CLOCKS_PER_SEC;
	t1 = ((double)stop1 - start1) / CLOCKS_PER_SEC;
	t2 = ((double)stop2 - start2) / CLOCKS_PER_SEC;
	t3 = ((double)stop3 - start3) / CLOCKS_PER_SEC;
	t4 = ((double)stop4 - start4) / CLOCKS_PER_SEC;
	t5 = ((double)stop5 - start5) / CLOCKS_PER_SEC;
	t6 = ((double)stop6 - start6) / CLOCKS_PER_SEC;
	printf("MESH MATERIAL INTI: %.10lf\n", t1);
	printf("STIFFNESS MASS INIT: %.10lf\n", t2);
	printf("EDOFMAT INIT: %.10lf\n", t3);
	printf("JK IK COMPUTE: %.10lf\n", t4);
	printf("DEFINE LOAD SUPPORT: %.10lf\n", t5);
	printf("PREPARE FILTER: %.10lf\n", t6);
	printf("TOTAL: %.10lf\n", tempo_tot);
	
	return 0;
}

