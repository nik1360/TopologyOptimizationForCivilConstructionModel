
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "constants.cuh"
#include "mesh.cuh"
#include "material.cuh"
#include "sys.cuh"
#include "mma.cuh"
#include "matrix_functions.cuh"
#include "hinf_functions.cuh"

#include "parallel.cuh"


void hinf_compute_mma(double *xphys, struct mma_param *mma_p, struct sparse *h, double *hs, struct mesh *mesh, struct material *material, int *ik, int *jk, double *me, double *ke, 
	int *freedofs, int *fixeddofs, struct sparse *f, struct sys *g, struct sys *gec8, struct sys *g2, int *edofmat, cublasHandle_t cublas_handle, cusolverDnHandle_t cusolverH) {

	//-----DECLARATIONS-----------
	struct sparse k;
	struct sparse m;
	struct sparse s;
	int x_mat_index = 0,ntutto;
	int ndof, nstate;
	double *sk, *sm, *x_mat;
	double nrm, h_inf_peak_frq;
	
	//G system parameters
	struct sparse a;
	struct sparse b;
	struct sparse c;
	struct sparse e;
	double  g_d;
	

	double scale ;
	double valuess,ss;
	int us, vs;

	struct sparse invu;


	//-------END DECLARATIONS---------
	
	x_mat = (double*)malloc(mesh->nelx*mesh->nely  * sizeof(double));

	//reshape x as a nely x nelx
	for (int col_index = 0; col_index < mesh->nelx; col_index++) {
		for (int row_index = 0; row_index < mesh->nely; row_index++) {
			x_mat[row_index*mesh->nelx + col_index] = mma_p->xval[x_mat_index];
			x_mat_index++;
		}
	}

	if (mesh->ft == 0 || mesh->ft == 1)
		for (int i = 0; i < mesh->nelx*mesh->nely; i++)
			xphys[i] = x_mat[i];
	else
		if (mesh->ft == 2) {
			compute_xphys(h, hs, x_mat, xphys, mesh);
		}

	//xPhys(passive==2) = 1;

	compute_sk(&sk, ke, xphys, mesh, material);	//sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),64*nelx*nely,1);
	prepare_sparse(ik, jk, sk, &k, KEROW*KECOL*mesh->nelx*mesh->nely);		
	ntutto = max_value(k.row, k.nnz); //NTUTTO = size(K,1); 
	
	compute_sm(&sm, me, xphys, mesh, material); //sM = reshape(ME(:)*(Rhomin+xPhys(:).*sin(pi/2*xPhys(:).^2))'*(Rho0-Rhomin),64*nelx*nely,1);
	prepare_sparse(ik, jk, sm, &m, MEROW*MECOL*mesh->nelx*mesh->nely);
	
	modify_m_k(&k, mesh, freedofs, fixeddofs);	//K=(K+K')/2 , K=K(freedofs,freedofs)
	modify_m_k(&m, mesh, freedofs, fixeddofs); //M=(M+M')/2 , M=M(freedofs,freedofs)

	compute_s(&k, &m, &s, mesh); //S = ALPHA*M+BETA*K;

	//-------- G AND GEC8 SYSTEM INIT------------
	
	ndof = (2 * (mesh->nelx + 1)*(mesh->nely + 1)) - mesh->fixed_count; //NDOF = length(freedofs);
	nstate = 2 * ndof; //NSTATE = 2 * NDOF;

	g_matrix_init(ndof, nstate, &a, &b, &c, &g_d, &e, &k, &m, &s, freedofs, f);

	g_sys_init(g, ndof, nstate, &a,&b,&c,g_d,&e );
	
	if (ISPETTRO==0) {
		//sys_inf_norm(g, &nrm, &h_inf_peak_frq); //[nrm hInfPeakFreq] = norm(G,Inf);
		sys_inf_norm_parallel(g, &h_inf_peak_frq, &nrm, cublas_handle, cusolverH);
	}
	else {
		sys_multiplication(g, gec8, g2);	//G2=G*GEC8
		sys_inf_norm_parallel(g, &h_inf_peak_frq, &nrm, cublas_handle, cusolverH);
	}
	
	scale = pow(10, 2); //SCALE = 10^2;
	mma_p->f0val = scale * nrm;
	
	/*if (ISPETTRO==0) {
		freqresp_parallel(g, h_inf_peak_frq, &valuess, cublas_handle, cusolverH); //may I use valuess=nrm?
	}
	else {
		freqresp_parallel(g2, h_inf_peak_frq, &valuess, cublas_handle, cusolverH); //where G2=G*GEC8
	}*/
	valuess = nrm;

	ss = valuess; us = 1; vs = 1; //[Us,Ss,Vs] = svdsim(valuess);	SVD of a 1x1 matrix?
	invforsens2_parallel(&invu, g, h_inf_peak_frq,cusolverH);

	//-------COMPUTE SENSITIVITIES----
	prosenshinf_2017(xphys, &invu, us, vs, h_inf_peak_frq, mma_p->df0dx, me, ke, edofmat, mesh, material, ntutto,freedofs,fixeddofs, &b, &c );

	for (int i = 0;i < mesh->nelx*mesh->nely;i++) {	//df0dx = SCALE*df0dx;
		mma_p->df0dx[i] = scale*mma_p->df0dx[i];
	}
	filt_mod_sensitivities(mesh, x_mat, mma_p->df0dx,h, hs);
	volume_constraint(&mma_p->fval, &mma_p->dfdx, x_mat, mesh, h, hs);
	
	
	free(x_mat);
	free(sk);
	free(sm);
	free_sparse(&a, &b, &c, &e, &k, &m, &s, &invu);
	free_sys(g);
	if (ISPETTRO != 0) {
		free_sys(g2);
	}
	
	
	
}


