#include <stdio.h>
#include <math.h>
#include "matrix_functions.cuh"
#include "mma.cuh"
#include "sparse_struc.cuh"
void kktcheck(struct mma_param *mma_p, struct kktcheck_pointers *kktp, double *df0dx, double *fval,double *dfdx, double*residunorm, double *residumax) {

	int residu1size, residu2size, residusize;
	int r1_index = 0, r2_index = 0;

	residu1size = mma_p->n + (2 * mma_p->m);
	residu2size = (2 * mma_p->n) + (4 * mma_p->m);
	residusize = residu1size + residu2size;

	for (int i = 0;i < mma_p->n; i++) {
		kktp->rex[i] = df0dx[i] + (dfdx[i] * mma_p->lam[0]) - mma_p->xsi[i] + mma_p->eta[i]; //rex   = df0dx + dfdx'*lam - xsi + eta;
		kktp->rexsi[i] = mma_p->xsi[i] * (mma_p->xmma[i] - mma_p->xmin[i]); //rexsi = xsi.*(x - xmin);
		kktp->reeta[i] = mma_p->eta[i] * (mma_p->xmax[i] - mma_p->xmma[i]); //reeta = eta.*(xmax-x);
		
	}

	for (int i = 0;i < mma_p->m; i++) {
		kktp->rey[i] = mma_p->c[i] + (mma_p->d[i] * mma_p->ymma[i]) - mma_p->mu[i] - mma_p->lam[i];	//rey   = c + d.*y - mu - lam;
		kktp->rez[i] = mma_p->a0 - mma_p->zet[i] - (mma_p->a[i] * mma_p->lam[i]);	//rez   = a0 - zet - a'*lam;
		kktp->relam[i] = fval[i] - (mma_p->a[i] * mma_p->zmma[i]) - mma_p->ymma[i] + mma_p->s[i]; //relam = fval - a*z - y + s;
		kktp->remu[i] = mma_p->mu[i] * mma_p->ymma[i]; //remu  = mu.*y;
		kktp->rezet[i] = mma_p->zet[i] * mma_p->zmma[i]; //rezet = zet*z;
		kktp->res[i] = mma_p->lam[i] * mma_p->s[i];
	}

	//residu1 = [rex' rey' rez]';
	for (int i = 0;i <mma_p->n; i++) {
		kktp->residu1[r1_index] = kktp->rex[i];
		r1_index++;
	}
	kktp->residu1[r1_index] = kktp->rey[0];
	r1_index++;
	kktp->residu1[r1_index] = kktp->rez[0];
	
	//residu2 = [relam' rexsi' reeta' remu' rezet res']';
	kktp->residu2[r2_index] = kktp->relam[0];
	r2_index++;
	for (int i = 0;i <mma_p->n; i++) {
		kktp->residu2[r2_index] = kktp->rexsi[i];
		r2_index++;
	}
	for (int i = 0;i <mma_p->n; i++) {
		kktp->residu2[r2_index] = kktp->reeta[i];
		r2_index++;
	}
	kktp->residu2[r2_index] = kktp->remu[0];
	r2_index++;
	kktp->residu2[r2_index] = kktp->rezet[0];
	r2_index++;
	kktp->residu2[r2_index] = kktp->res[0];
	r2_index++;

	//residu = [residu1' residu2']';
	//residunorm = sqrt(residu'*residu);
	(*residunorm) = 0;
	for (int i = 0; i < residu1size; i++) {
		kktp->residu[i] = kktp->residu1[i];
		(*residunorm) += kktp->residu[i] * kktp->residu[i];
	}
	for (int i = 0; i < residu2size; i++) {
		kktp->residu[i + residu1size] = kktp->residu2[i];
		(*residunorm) += kktp->residu[i + residu1size] * kktp->residu[i + residu1size];
	}
	(*residunorm) = sqrt((*residunorm));

	//residumax = max(abs(residu));
	(*residumax) = fabs(kktp->residu[0]);
	for (int i = 0;i < residusize;i++) {
		if (fabs(kktp->residu[i]) > (*residumax)) {
			(*residumax) = fabs(kktp->residu[i]);
		}
	}
	

}