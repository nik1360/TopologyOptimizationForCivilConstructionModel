#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "constants.cuh"
#include "mesh.cuh"
#include "material.cuh"
#include "sys.cuh"
#include "matrix_functions.cuh"
#include "mma.cuh"
#include "pgm_utility.cuh"

#include "parallel.cuh"





void mma(struct sparse *h, double *hs, struct mesh *mesh, struct material *material, int *ik, int *jk, double *me, double *ke, int *freedofs,
	int*fixeddofs, struct sparse *f, int *edofmat) {
	
	struct subsolv_pointers ssp;
	struct mmasub_pointers msp;
	struct kktcheck_pointers kktp;
	struct mma_param mma_p;
	double *xphys;
	
	struct sys g;
	struct sys gec8;
	struct sys g2;
	double kktnorm,residumax;
	int kktcount;



	
	
	int outit;

	char filename[10];

	cublasHandle_t cublas_handle;
	cusolverDnHandle_t cusolverH = NULL;
	cublasCreate(&cublas_handle);
	cusolverDnCreate(&cusolverH);

	
	
	gec8_sys_init(&gec8);
	mma_init(&mma_p, mesh);

	
	xphys = (double*)malloc(mesh->nelx*mesh->nely * sizeof(double));
	subsolv_malloc(mma_p.m,mma_p.n,&ssp);
	mmasub_malloc(mma_p.m, mma_p.n, &msp);
	kktcheck_malloc(mma_p.m, mma_p.n, &kktp);

	//creating the image with the initial conditions
	sprintf(filename, "img%d.pgm", 0);
	write_image(filename, mma_p.n, mesh, mma_p.xval);

	if (mma_p.outeriter < 0.5) {
		hinf_compute_mma(xphys,&mma_p, h, hs, mesh, material, ik, jk, me, ke, freedofs, fixeddofs, f , &g, &gec8, &g2, edofmat, cublas_handle, cusolverH);
	}

	printf("INTIAL HINF NRM: %f\n", (mma_p.f0val / 100));

	kktnorm = mma_p.kkttol + 10;
	mma_p.f0val = mma_p.f0valsoglia + 50;
	
	outit = 0;

	kktcount = 0;

	while ((kktnorm>mma_p.kkttol)&&(outit<mma_p.maxoutit)&&(kktcount<3)) {
		
		outit++;
		mma_p.outeriter++;

		mmasub(&mma_p, mma_p.f0val, mma_p.df0dx, mma_p.fval, mma_p.dfdx, mesh, &msp, &ssp);
		

		for (int i = 0; i < mma_p.n; i++) {
			mma_p.xold2[i] = mma_p.xold1[i];	//xold2 = xold1;
			mma_p.xold1[i] = mma_p.xval[i];	//xold1 = xval;
			mma_p.xval[i] = mma_p.xmma[i];	//xval  = xmma;
		}

		
		hinf_compute_mma(xphys, &mma_p, h, hs, mesh, material, ik, jk, me, ke, freedofs, fixeddofs, f, &g, &gec8, &g2, edofmat,cublas_handle,cusolverH); //[f0val,df0dx,fval,dfdx] = HINF_COMPUTE_MMA(xval);
		kktcheck(&mma_p, &kktp, mma_p.df0dx, mma_p.fval, mma_p.dfdx, &kktnorm, &residumax);
		
		if (kktnorm < mma_p.kkttol) {
			kktcount++;
		}
		else {
			kktcount = 0;
		}
		//printf("%f %f %f %f\n", kktnorm, residumax, mma_p.fval[0], mma_p.f0val);
		
		//every iterations a new image is created
		if (outit % 1==0 ){
			sprintf(filename, "img%d.pgm", outit);
			write_image(filename, mma_p.n, mesh, xphys);
		}

		if (outit % (int)(mma_p.maxoutit/100) == 0) {
			printf("%d/100 completed\n", (int)((outit/mma_p.maxoutit)*100));
		}
	
	}
	
	printf("FINAL HINF NRM: %f\n", (mma_p.f0val / 100));

	cublasDestroy(cublas_handle);
	cusolverDnDestroy(cusolverH);

	subsolv_free(&ssp);
	mmasub_free(&msp);
	kktcheck_free(&kktp);
	mma_param_free(&mma_p);
	free_sys(&gec8);
	free(xphys);

}





	




