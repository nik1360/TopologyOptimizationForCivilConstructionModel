#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "constants.cuh"
#include "mesh.cuh"
#include "matrix_functions.cuh"
#include "material.cuh"
#include "sys.cuh"



void compute_xphys(struct sparse *h ,double *hs, double*x, double *xphys, struct mesh *mesh) {
	
	double  *x_lin, *xphys_tmp;

	x_lin = (double*)malloc(mesh->nelx*mesh->nely * sizeof(double));
	xphys_tmp = (double*)malloc(mesh->nelx*mesh->nely * sizeof(double));

	for (int i = 0; i < mesh->nelx*mesh->nely; i++) {
		xphys_tmp[i] = 0;
		xphys[i] = 0;
	}

	linearize_matrix(x, x_lin, mesh->nely, mesh->nelx);
	
	for (int i = 0; i < h->nnz; i++) {		//tmp2=H*tmp1
		xphys_tmp[h->row[i] - 1] += h->val[i] * x_lin[h->col[i] - 1];
	}

	for (int i = 0; i < mesh->nelx*mesh->nely; i++) {
		xphys[i] = xphys_tmp[i] / hs[i];
	}
	

	free(xphys_tmp);
	free(x_lin);

}
void compute_sk(double **sk, double *ke, double *xphys, struct mesh *mesh, struct material *material) {
	int k;
	double *xphys_lin, *tmp, *ke_lin;

	(*sk) = (double*)malloc(mesh->nelx*mesh->nely * 64 * sizeof(double));
	ke_lin = (double*)malloc(KEROW*KECOL * sizeof(double)); //ke linearizzato
	xphys_lin = (double*)malloc(mesh->nelx*mesh->nely * sizeof(double));	//xphys linearizzato
	tmp = (double*)malloc(mesh->nelx*mesh->nely * sizeof(double));

	linearize_matrix(ke, ke_lin, KEROW, KECOL);

	if (mesh->ft == 0 || mesh->ft == 1) {
		linearize_matrix(xphys, xphys_lin, mesh->nely, mesh->nelx);	
		for (int i = 0; i < mesh->nelx*mesh->nely; i++) {
			tmp[i] = pow(xphys_lin[i], mesh->penal)*(material->e0 - material->emin) + material->emin;	
		}
	}
	else {
		for (int i = 0; i < mesh->nelx*mesh->nely; i++) {
			tmp[i] = pow(xphys[i], mesh->penal)*(material->e0 - material->emin) + material->emin;	
		}
	}
	
	k = 0;
	for (int tmp_index = 0; tmp_index < mesh->nelx*mesh->nely; tmp_index++) {
		for (int i = 0; i < 64; i++) {
			(*sk)[k] = ke_lin[i] * tmp[tmp_index];
			k++;
		}
	}


	free(tmp);
	free(ke_lin);
	free(xphys_lin);

}
void compute_sm(double **sm, double *me, double *xphys, struct mesh *mesh, struct material *material) {
	int k = 0;
	double *xphys_lin, *tmp, *me_lin, pi = 3.141592653589;

	(*sm) = (double*)malloc(mesh->nelx*mesh->nely * 64 * sizeof(double));
	me_lin = (double*)malloc(MEROW*MECOL * sizeof(double)); //me linearizzato
	xphys_lin = (double*)malloc(mesh->nelx*mesh->nely * sizeof(double));	//xphys linearizzato
	tmp = (double*)malloc(mesh->nelx*mesh->nely * sizeof(double));
	
	linearize_matrix(me, me_lin, MEROW, MECOL);

	if ((mesh->ft == 0) || (mesh->ft == 1)) {
		linearize_matrix(xphys, xphys_lin, mesh->nely, mesh->nelx);
		for (int i = 0; i < mesh->nelx*mesh->nely; i++) {
			tmp[i] = (xphys_lin[i] * sin((pi / 2)*pow(xphys_lin[i], 2))) + material->rhomin;
		}
	}
	else {
		for (int i = 0; i < mesh->nelx*mesh->nely; i++) {
			tmp[i] = (xphys[i] * sin((pi / 2)*pow(xphys[i], 2))) + material->rhomin;
		}
	}
	

	for (int tmp_index = 0; tmp_index < mesh->nelx*mesh->nely; tmp_index++) {
		for (int i = 0; i < 64; i++) {
			(*sm)[k] = (me_lin[i] * tmp[tmp_index])*(material->rho0 - material->rhomin);
			k++;
		}
	}

	free(tmp);
	free(me_lin);
	free(xphys_lin);

}
void modify_m_k(struct sparse *mat, struct mesh *mesh, int *freedofs, int *fixeddofs) {

	int *ik_tmp, *jk_tmp, index = 0, found, count, c;
	int n_freedofs = (2 * (mesh->nelx + 1)*(mesh->nely + 1)) - mesh->fixed_count;
	double *vk_tmp, value;

	ik_tmp = (int*)malloc(mat->nnz * sizeof(int));
	jk_tmp = (int*)malloc(mat->nnz * sizeof(int));
	vk_tmp = (double*)malloc(mat->nnz * sizeof(double));

	for (int i = 0; i < mat->nnz; i++) {		//MAT=(MAT+MAT')/2
		found = 0;
		for (int j = 0;j<mat->nnz; j++) {
			if ((mat->row[i] == mat->col[j]) && (mat->col[i] == mat->row[j])) {
				value = (mat->val[i] + mat->val[j]);
				if (fabs(value) > pow(10, -9)) {
					found = 1;
					ik_tmp[index] = mat->row[i];
					jk_tmp[index] = mat->col[i];
					vk_tmp[index] = value / 2;
					index++;
					break;
				}
			}
		}
		if (found == 0) {
			ik_tmp[index] = mat->row[i];
			jk_tmp[index] = mat->col[i];
			vk_tmp[index] = mat->val[i] / 2;
			index++;
		}
	}

	count = 0;				//MAT=MAT(freedofs,freedofs)
	for (int i = 0; i < n_freedofs;i++) {
		for (int j = 0; j < mat->nnz;j++) {
			if (ik_tmp[j] == freedofs[i]) {
				for (int k = 0; k < n_freedofs;k++) {
					if (jk_tmp[j] == freedofs[k]) {
						count++;
						break;
					}
				}
			}
		}
	}

	free(mat->row);
	free(mat->col);
	free(mat->val);
	mat->row = (int*)malloc(count * sizeof(int));
	mat->col = (int*)malloc(count * sizeof(int));
	mat->val = (double*)malloc(count * sizeof(double));

	count = 0;
	for (int i = 0; i < n_freedofs; i++) {
		for (int j = 0; j < mat->nnz; j++) {
			if (ik_tmp[j] == freedofs[i]) {
				for (int k = 0; k < n_freedofs; k++) {
					if (jk_tmp[j] == freedofs[k]) {
						mat->row[count] = ik_tmp[j];
						mat->col[count] = jk_tmp[j];
						mat->val[count] = vk_tmp[j];
						count++;
						//printf("(%d,%d) %f\n", ik_tmp[j], jk_tmp[j], vk_tmp[j]);
						break;
					}
				}
			}
		}
	}

	mat->nnz = count;
	for (int i = 0; i < mat->nnz; i++) {		//correct indexes
		c = 0;
		for (int j = 0; j < mesh->fixed_count;j++) {
			if (mat->row[i] > fixeddofs[j]) {
				c++;
			}
		}
		mat->row[i] -= c;
		c = 0;
		for (int j = 0; j < mesh->fixed_count; j++) {
			if (mat->col[i] > fixeddofs[j]) {
				c++;
			}
		}
		mat->col[i] -= c;
		//printf("(%d,%d) %f\n", (*ik)[i], (*jk)[i], (*vk)[i]);

	}

	free(ik_tmp);
	free(jk_tmp);
	free(vk_tmp);

}
void compute_s(struct sparse *k, struct sparse *m, struct sparse *s, struct mesh *mesh) {
	int found, count = 0;
	for (int i = 0; i < k->nnz;i++) {
		found = 0;
		for (int j = 0; j < m->nnz; j++) {
			if ((k->row[i] == m->row[j]) && (k->col[i] == m->col[j])) {
				found = 1;
				if ((k->val[i] + m->val[j]) != 0)
				{
					count++;
				}
				break;
			}
		}
		if (found == 0) {
			count++;
		}
	}
	for (int i = 0; i < m->nnz; i++) {
		found = 0;
		for (int j = 0; j < k->nnz; j++) {
			if ((m->row[i] == k->row[j]) && (m->col[i] == k->col[j])) {
				found = 1;
			}
		}
		if (found == 0) {
			count++;
		}
	}

	s->nnz = count;
	s->row = (int*)malloc(s->nnz * sizeof(int));
	s->col = (int*)malloc(s->nnz * sizeof(int));
	s->val = (double*)malloc(s->nnz * sizeof(double));

	count = 0;
	for (int i = 0; i < k->nnz; i++) {
		found = 0;
		for (int j = 0; j < m->nnz; j++) {
			if ((k->row[i] == m->row[j]) && (k->col[i] == m->col[j])) {
				found = 1;
				if (fabs(k->val[i] + m->val[j]) > pow(10, -9))
				{
					s->row[count] = k->row[i];
					s->col[count] = k->col[i];
					s->val[count] = (mesh->beta*k->val[i]) + (mesh->alpha*m->val[j]);
					count++;
				}
				break;
			}
		}
		if (found == 0) {
			s->row[count] = k->row[i];
			s->col[count] = k->col[i];
			s->val[count] = (mesh->beta*k->val[i]);
			count++;
		}
	}
	for (int i = 0; i < m->nnz; i++) {
		found = 0;
		for (int j = 0; j < k->nnz; j++) {
			if ((m->row[i] == k->row[j]) && (m->col[i] == k->col[j])) {
				found = 1;
			}
		}
		if (found == 0) {
			s->row[count] = m->row[i];
			s->col[count] = m->col[i];
			s->val[count] = mesh->alpha*m->val[i];
			count++;
		}
	}

	/*for (int i = 0; i < (*s_nnz); i++) {
	printf("(%d,%d) %f\n", (*is)[i], (*js)[i], (*vs)[i]);
	}*/

}
void g_matrix_init(int ndof, int nstate, struct sparse *a, struct sparse *b, struct sparse *c, double *dstato, struct sparse *e, struct sparse *k, struct sparse *m, struct sparse *s, int *freedofs, struct sparse *f) {

	int found;

	//-------estato init-----------
	e->val = (double*)malloc((int)(pow(ndof, 2) + ndof) * sizeof(double));
	e->row = (int*)malloc((int)(pow(ndof, 2) + ndof) * sizeof(int));
	e->col = (int*)malloc((int)(pow(ndof, 2) + ndof) * sizeof(int));
	e->nnz = 0;
	e->nrow = nstate;
	e->ncol = nstate;

	for (int i = 0; i < ndof;i++) {		//ESTATO(1:NDOF,1:NDOF)=speye(NDOF);
		e->row[i] = i + 1;
		e->col[i] = i + 1;
		e->val[i] = 1;
		e->nnz++;
	}

	for (int i = 0; i < m->nnz; i++) {	//ESTATO(NDOF+1:NSTATE,NDOF+1:NSTATE) = M;
		e->row[i + ndof] = m->row[i] + ndof;
		e->col[i + ndof] = m->col[i] + ndof;
		e->val[i + ndof] = m->val[i];
		e->nnz++;
	}

	/*for (int i = 0;i < e->nnz;i++) {
		printf("(%d,%d) %f\n", e->row[i], e->col[i], e->val[i]);
	}*/
	//-------astato init----------

	a->val = (double*)malloc((int)(2 * pow(ndof, 2) + ndof) * sizeof(double));
	a->row = (int*)malloc((int)(2 * pow(ndof, 2) + ndof) * sizeof(int));
	a->col = (int*)malloc((int)(2 * pow(ndof, 2) + ndof) * sizeof(int));
	a->nnz = 0;
	a->nrow = nstate;
	a->ncol = nstate;

	for (int i = 0; i < ndof; i++) {		//ASTATO(1:NDOF,NDOF+1:NSTATE) = speye(NDOF);
		a->row[i] = i + 1;
		a->col[i] = i + ndof + 1;
		a->val[i] = 1;
		a->nnz++;
	}

	for (int i = 0; i < k->nnz; i++) {		//ASTATO(NDOF+1:NSTATE,:) = [-K -S];
		a->row[i + ndof] = k->row[i] + ndof;
		a->col[i + ndof] = k->col[i];
		a->val[i + ndof] = -k->val[i];
		a->nnz++;
	}
	for (int i = 0; i < s->nnz; i++) {
		a->row[i + ndof + k->nnz] = k->row[i] + ndof;
		a->col[i + ndof + k->nnz] = k->col[i] + ndof;
		a->val[i + ndof + k->nnz] = -s->val[i];
		a->nnz++;
	}

	/*for (int i = 0; i < (*a_nnz); i++) {
	printf("(%d,%d) %f\n", (*astato_row)[i], (*astato_col)[i], (*astato)[i]);
	}*/

	//-------bstato init----------
	b->val = (double*)malloc(ndof * sizeof(double));
	b->row = (int*)malloc(ndof * sizeof(int));
	b->col = (int*)malloc(ndof * sizeof(int));
	b->nnz = 0;
	b->nrow = nstate;
	b->ncol = NINP;

	for (int i = 0; i < f->nnz; i++) {		//BSTATO(NDOF+1:NSTATE,1) = F(freedofs);
		found = 0;
		for (int j = 0; j < ndof;j++) {
			if (f->row[i] == freedofs[j]) {
				found = 1;
				b->row[b->nnz] = b->nnz + ndof + 1;
				b->col[b->nnz] = f->col[i];
				b->val[b->nnz] = f->val[i];
				b->nnz++;
				break;
			}
		}
	}
	/*for (int i = 0; i < (*b_nnz); i++) {
	printf("(%d,%d) %f\n", (*bstato_row)[i], (*bstato_col)[i], (*bstato)[i]);
	}*/

	//-----cstato init ------
	c->val = (double*)malloc(nstate * sizeof(double));
	c->row = (int*)malloc(nstate * sizeof(int));
	c->col = (int*)malloc(nstate * sizeof(int));
	c->nnz = 0;
	c->nrow = NOUT;
	c->ncol = nstate;

	for (int i = 0; i < f->nnz; i++) {		//BSTATO(NDOF+1:NSTATE,1) = F(freedofs);
		found = 0;
		for (int j = 0; j < ndof;j++) {
			if (f->row[i] == freedofs[j]) {
				found = 1;
				c->row[c->nnz] = f->col[i];
				c->col[c->nnz] = c->nnz + 1;
				c->val[c->nnz] = f->val[i];
				c->nnz++;
				break;
			}
		}
	}

	/*for (int i = 0; i < (*c_nnz); i++) {
	printf("(%d,%d) %f\n", (*cstato_row)[i], (*cstato_col)[i], (*cstato)[i]);
	}*/

	//-------dstato init-----

	(*dstato) = 0;

}
void filt_mod_sensitivities(struct mesh *mesh, double *x, double *df0dx,struct sparse *h, double *hs) {
	int numel;
	double *x_lin, *tmp1, *tmp2;


	numel = mesh->nelx*mesh->nely;
	tmp1 = (double*)malloc(numel * sizeof(double));

	if (mesh->ft == 1) {
		x_lin = (double*)malloc(numel * sizeof(double));
		tmp2 = (double*)malloc(numel * sizeof(double));

		linearize_matrix(x, x_lin, mesh->nely, mesh->nelx);	//x_lin=x(:)
		for (int i = 0; i < numel; i++) {
			tmp1[i] = x_lin[i] * df0dx[i];	//tmp1=x(:).*df0dx(:)
			tmp2[i] = 0;
		}

		for (int i = 0; i < h->nnz; i++) {		//tmp2=H*tmp1
			tmp2[h->row[i] - 1] += h->val[i] * tmp1[h->col[i] - 1];
		}

		for (int i = 0; i < numel; i++) {	//tmp3./Hs./max(1e-3,x(:));
			df0dx[i] = (tmp2[i] / hs[i]) / max((pow(10, -3)), x_lin[i]);
		}
		free(x_lin);
		free(tmp2);

	}
	else if (mesh->ft == 2) {
		for (int i = 0; i < numel; i++) {		//tmp1=df0dx(:)./Hs
			tmp1[i] = df0dx[i] / hs[i];
			df0dx[i] = 0;
		}
		for (int i = 0; i < h->nnz; i++) {
			df0dx[h->row[i] - 1] += h->val[i] * tmp1[h->col[i] - 1];	//H*tmp1
		}
	}


	free(tmp1);

}
void volume_constraint(double **fval, double **dfdx, double *x, struct mesh *mesh, struct sparse *h, double *hs) {

	double epsi, epsi2, sum;
	double *x_lin, *tmp, *tmp2, *dfdx2;
	int numel;

	numel = mesh->nelx*mesh->nely;
	x_lin = (double*)malloc(numel * sizeof(double));


	epsi = 0.01;

	linearize_matrix(x, x_lin, mesh->nely, mesh->nelx);
	if (IVOLUME == 0) {
		tmp = (double*)malloc(numel * sizeof(double));

		sum = 0;
		for (int i = 0; i < numel; i++) {
			sum += x_lin[i];
		}
		(*fval)[0] = ((mesh->area*sum) / (mesh->vmax*VOLFRAC)) - 1 - epsi; // fval = area*sum(x(:))/(VMAX*volfrac)-1-EPSI; 
		for (int i = 0; i < numel; i++) {	// dfdx = area/(VMAX*volfrac)*ones(length(x(:)),1);
			(*dfdx)[i] = mesh->area / (mesh->vmax*VOLFRAC);
		}
		if (mesh->ft == 2) {

			for (int i = 0; i < numel; i++) {		//tmp=dfdx(:) ./ HS
				tmp[i] = (*dfdx)[i] / hs[i];
				(*dfdx)[i] = 0;
			}

			for (int i = 0; i < h->nnz; i++) {	//dfdx=H*tmp
				(*dfdx)[h->row[i] - 1] += h->val[i] * tmp[h->col[i] - 1];
			}

		}
		free(tmp);
	}
	else {
		
		dfdx2 = (double*)malloc(numel * sizeof(double));
		tmp = (double*)malloc(numel * sizeof(double));
		tmp2 = (double*)malloc(numel * sizeof(double));

		epsi2 = 0.05;
		sum = 0;
		for (int i = 0; i < numel; i++) {
			sum += x_lin[i];
		}
		//fval = [fval1; fval2];
		(*fval)[0] = ((mesh->area*sum) / (mesh->vmax * VOLFRAC)) - 1 - epsi; // fval1 = area*sum(x(:))/(VMAX*volfrac)-1-EPSI; 
		(*fval)[1] = ((-mesh->area*sum) / (mesh->vmax * VOLFRAC)) + 1 - epsi2; // fval2 = area*sum(x(:))/(VMAX*volfrac)+1-EPSI2; 

		for (int i = 0; i < numel; i++) {	// dfdx = area/(VMAX*volfrac)*ones(length(x(:)),1);  dfdx2=-area/(VMAX*volfrac)*ones(length(x(:)),1);
			(*dfdx)[i] = mesh->area / (mesh->vmax*VOLFRAC);
			dfdx2[i] = -(mesh->area) / (mesh->vmax*VOLFRAC);
		}

		if (mesh->ft == 2) {
			for (int i = 0; i < numel; i++) {	//tmp=dfdx(:)./Hs 	tmp2=dfdx2(:)./Hs	
				tmp[i] = (*dfdx)[i] / hs[i];
				tmp2[i] = dfdx2[i] / hs[i];
				(*dfdx)[i] = 0;
				dfdx2[i] = 0;
			}

			for (int i = 0; i < h->nnz; i++) {
				(*dfdx)[h->row[i] - 1] += h->val[i] * tmp[h->col[i] - 1];	//H*tmp
				dfdx2[h->row[i] - 1] += h->val[i] * tmp2[h->col[i] - 1];
			}

		}
		for (int i = numel; i < 2 * numel; i++) {
			(*dfdx)[i] = dfdx2[i - numel];
		}

		free(tmp);
		free(dfdx2);
		free(tmp2);

	}
	free(x_lin);
}

void free_sparse(struct sparse *a, struct sparse *b, struct sparse *c, struct sparse *e, struct sparse *k, struct sparse *m, struct sparse *s, struct sparse *invu) {
	free(a->row);
	free(a->col);
	free(a->val);
	free(b->row);
	free(b->col);
	free(b->val);
	free(c->row);
	free(c->col);
	free(c->val);
	free(e->row);
	free(e->col);
	free(e->val);
	free(k->row);
	free(k->col);
	free(k->val);
	free(m->row);
	free(m->col);
	free(m->val);
	free(s->row);
	free(s->col);
	free(s->val);
	free(invu->row);
	free(invu->col);
	free(invu->val);
}

