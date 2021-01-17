

#include <stdlib.h>
#include <stdio.h>

#include "constants.cuh"
#include "mesh.cuh"
#include "mma.cuh"




void mma_init(struct mma_param *mma_p, struct mesh *mesh) {
	
	mma_p->m = 1 + IVOLUME;
	mma_p->n = mesh->nelx*mesh->nely;
	mma_p->epsimin = 0.0000001;
	mma_p->outeriter = 0;
	mma_p->maxoutit = 500;
	mma_p->kkttol = 0.2/*0.000001*/;
	mma_p->f0valsoglia = 5;
	mma_p->a0 = 1;
	
	mma_p->x = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->xval = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->xold1 = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->xold2 = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->xmin = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->xmax = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->low = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->upp = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->a = (int*)malloc(mma_p->m * sizeof(int));
	mma_p->c = (int*)malloc(mma_p->m * sizeof(int));
	mma_p->d = (int*)malloc(mma_p->m * sizeof(int));
	
	mma_p->xmma = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->ymma = (double*)malloc(mma_p->m * sizeof(double));
	mma_p->zmma = (double*)malloc(mma_p->m * sizeof(double));
	mma_p->lam = (double*)malloc(mma_p->m * sizeof(double));
	mma_p->xsi = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->eta = (double*)malloc(mma_p->n * sizeof(double));
	mma_p->mu = (double*)malloc(mma_p->m * sizeof(double));
	mma_p->zet = (double*)malloc(mma_p->m * sizeof(double));
	mma_p->s = (double*)malloc(mma_p->m * sizeof(double));

	mma_p->df0dx= (double*)malloc(mma_p->n * sizeof(double));
	mma_p->dfdx= (double*)malloc(mma_p->n * sizeof(double));
	mma_p->fval= (double*)malloc(mma_p->m * sizeof(double));


	for (int i = 0; i < mma_p->n; i++) {
		mma_p->xval[i] = mesh->volfrac;
		mma_p->xval[i] = mesh->volfrac;
		mma_p->xold1[i] = mesh->volfrac;
		mma_p->xold2[i] = mesh->volfrac;
		mma_p->xmin[i] = 0.000001;
		mma_p->xmax[i] = 1;
		mma_p->low[i] = 0.000001;
		mma_p->upp[i] = 1;

	}
	for (int i = 0; i < mma_p->m; i++) {
		mma_p->a[i] = 0;
		mma_p->c[i] = 10000;
		mma_p->d[i] = 0;
	}
}
void mma_param_free(struct mma_param *mma_p) {
	
	free(mma_p->x);
	free(mma_p->xval);
	free(mma_p->xold1);
	free(mma_p->xold2);
	free(mma_p->xmin);
	free(mma_p->xmax);
	free(mma_p->low);
	free(mma_p->upp);
	free(mma_p->a);
	free(mma_p->c);
	free(mma_p->d);
	free(mma_p->xmma);
	free(mma_p->ymma);
	free(mma_p->zmma);
	free(mma_p->lam);
	free(mma_p->xsi);
	free(mma_p->eta);
	free(mma_p->mu);
	free(mma_p->zet);
	free(mma_p->s);
	free(mma_p->dfdx);
	free(mma_p->df0dx);
	free(mma_p->fval);
}

void mmasub_malloc(int m, int n, struct mmasub_pointers *p) {
	p->eeen = (int*)malloc(n * sizeof(int));
	p->zeron = (int*)malloc(n * sizeof(int));
	p->eeem = (int*)malloc(m * sizeof(int));
	p->factor = (double*)malloc(n * sizeof(double));
	p->lowmin = (double*)malloc(n * sizeof(double));
	p->lowmax = (double*)malloc(n * sizeof(double));
	p->uppmin = (double*)malloc(n * sizeof(double));
	p->uppmax = (double*)malloc(n * sizeof(double));
	p->zzz = (double*)malloc(n * sizeof(double));
	p->zzz1 = (double*)malloc(n * sizeof(double));
	p->zzz2 = (double*)malloc(n * sizeof(double));
	p->alfa = (double*)malloc(n * sizeof(double));
	p->beta = (double*)malloc(n * sizeof(double));
	p->xmami = (double*)malloc(n * sizeof(double));
	p->xmamieps = (double*)malloc(n * sizeof(double));
	p->xmamiinv = (double*)malloc(n * sizeof(double));
	p->ux1 = (double*)malloc(n * sizeof(double));
	p->ux2 = (double*)malloc(n * sizeof(double));
	p->xl1 = (double*)malloc(n * sizeof(double));
	p->xl2 = (double*)malloc(n * sizeof(double));
	p->uxinv = (double*)malloc(n * sizeof(double));
	p->xlinv = (double*)malloc(n * sizeof(double));
	p->p0 = (double*)malloc(n * sizeof(double));
	p->q0 = (double*)malloc(n * sizeof(double));
	p->pq0 = (double*)malloc(n * sizeof(double));
	p->p = (double*)malloc(m*n * sizeof(double));
	p->q = (double*)malloc(m*n * sizeof(double));
	p->pq = (double*)malloc(m*n * sizeof(double));
}
void mmasub_free(struct mmasub_pointers *p) {
	free(p->eeen);
	free(p->zeron);
	free(p->eeem);
	free(p->factor);
	free(p->lowmin);
	free(p->lowmax);
	free(p->uppmin);
	free(p->uppmax);
	free(p->zzz);
	free(p->zzz1);
	free(p->zzz2);
	free(p->alfa);
	free(p->beta);
	free(p->xmami);
	free(p->xmamieps);
	free(p->xmamiinv);
	free(p->ux1);
	free(p->ux2);
	free(p->xl1);
	free(p->xl2);
	free(p->p0);
	free(p->q0);
	free(p->pq0);
	free(p->uxinv);
	free(p->xlinv);
	free(p->p);
	free(p->q);
	free(p->pq);
}

void subsolv_malloc(int m, int n, struct subsolv_pointers *p) {
	int residu1size, residu2size, residusize;

	residu1size = n + (2 * m);
	residu2size = (2 * n) + (4 * m);
	residusize = residu1size + residu2size;
	
	p->een = (double*)malloc(n * sizeof(double));
	p->eem = (double*)malloc(m * sizeof(double));
	p->epsvecn = (double*)malloc(n * sizeof(double));
	p->epsvecm = (double*)malloc(m * sizeof(double));
	
	p->x = (double*)malloc(n * sizeof(double));
	p->y = (double*)malloc(m * sizeof(double));
	p->z = (double*)malloc(m * sizeof(double));
	p->lam = (double*)malloc(m * sizeof(double));
	p->xsi = (double*)malloc(n * sizeof(double));
	p->eta = (double*)malloc(n * sizeof(double));
	p->mu = (double*)malloc(m * sizeof(double));
	p->zet = (double*)malloc(m * sizeof(double));
	p->s = (double*)malloc(m * sizeof(double));

	p->ux1 = (double*)malloc(n * sizeof(double));
	p->ux2 = (double*)malloc(n * sizeof(double));
	p->ux3 = (double*)malloc(n * sizeof(double));
	p->xl1 = (double*)malloc(n * sizeof(double));
	p->xl2 = (double*)malloc(n * sizeof(double));
	p->xl3 = (double*)malloc(n * sizeof(double));
	p->uxinv1 = (double*)malloc(n * sizeof(double));
	p->xlinv1 = (double*)malloc(n * sizeof(double));
	p->uxinv2 = (double*)malloc(n * sizeof(double));
	p->xlinv2 = (double*)malloc(n * sizeof(double));

	p->plam = (double*)malloc(n * sizeof(double));
	p->qlam = (double*)malloc(n * sizeof(double));

	p->gg = (double*)malloc(n * sizeof(double));
	p->dpsidx = (double*)malloc(n * sizeof(double));

	p->rex = (double*)malloc(n * sizeof(double));
	p->rey = (double*)malloc(m * sizeof(double));
	p->rez = (double*)malloc(m * sizeof(double));
	p->relam = (double*)malloc(m * sizeof(double));
	p->rexsi = (double*)malloc(n * sizeof(double));
	p->reeta = (double*)malloc(n * sizeof(double));
	p->remu = (double*)malloc(m * sizeof(double));
	p->rezet = (double*)malloc(m * sizeof(double));
	p->res = (double*)malloc(m * sizeof(double));
	
	p->residu1 = (double*)malloc(residu1size * sizeof(double));
	p->residu2 = (double*)malloc(residu2size * sizeof(double));
	p->residu = (double*)malloc(residusize * sizeof(double));

	p->delx = (double*)malloc(n * sizeof(double));
	p->dely = (double*)malloc(m * sizeof(double));
	p->delz = (double*)malloc(m * sizeof(double));

	p->dellam = (double*)malloc(m * sizeof(double));

	p->diagx = (double*)malloc(n * sizeof(double));
	p->diagxinv = (double*)malloc(n * sizeof(double));
	p->diagy = (double*)malloc(m * sizeof(double));
	p->diagyinv = (double*)malloc(m * sizeof(double));
	p->diaglam = (double*)malloc(m * sizeof(double));
	p->diaglamyi = (double*)malloc(m * sizeof(double));

	p->blam = (double*)malloc(m * sizeof(double));
	p->bb = (double*)malloc(2 * m * sizeof(double));
	p->alam = (double*)malloc(m * sizeof(double));
	p->aa = (double*)malloc(4 * m * sizeof(double));
	p->aa_inv = (double*)malloc(4 * m * sizeof(double));

	p->solut = (double*)malloc(2 * m * sizeof(double));

	p->dlam = (double*)malloc(m * sizeof(double));
	p->dz = (double*)malloc(m * sizeof(double));
	p->dy = (double*)malloc(m * sizeof(double));
	p->dx = (double*)malloc(n * sizeof(double));

	p->dxsi = (double*)malloc(n * sizeof(double));
	p->deta = (double*)malloc(n * sizeof(double));

	p->dmu = (double*)malloc(m * sizeof(double));
	p->dzet = (double*)malloc(m * sizeof(double));
	p->ds = (double*)malloc(m * sizeof(double));

	p->xx = (double*)malloc((6 * m + 2 * n) * sizeof(double));
	p->dxx = (double*)malloc((6 * m + 2 * n) * sizeof(double));
	p->stepxx = (double*)malloc((6 * m + 2 * n) * sizeof(double));

	p->stepalfa = (double*)malloc(n * sizeof(double));
	p->stepbeta = (double*)malloc(n * sizeof(double));

	p->xold = (double*)malloc(n * sizeof(double));
	p->yold = (double*)malloc(m * sizeof(double));
	p->zold = (double*)malloc(m * sizeof(double));
	p->lamold = (double*)malloc(m * sizeof(double));
	p->xsiold = (double*)malloc(n * sizeof(double));
	p->etaold = (double*)malloc(n * sizeof(double));
	p->muold = (double*)malloc(m * sizeof(double));
	p->zetold = (double*)malloc(m * sizeof(double));
	p->sold = (double*)malloc(m * sizeof(double));

}
void subsolv_free(struct subsolv_pointers *p) {
	free(p->een);
	free(p->eem);
	free(p->epsvecm);
	free(p->epsvecn);
	free(p->x);
	free(p->y);
	free(p->z);
	free(p->lam);
	free(p->xsi);
	free(p->eta);
	free(p->mu);
	free(p->zet);
	free(p->s);
	free(p->ux1);
	free(p->ux2);
	free(p->ux3);
	free(p->xl1);
	free(p->xl2);
	free(p->xl3);
	free(p->uxinv1);
	free(p->xlinv1);
	free(p->uxinv2);
	free(p->xlinv2);
	free(p->plam);
	free(p->qlam);
	free(p->dpsidx);
	free(p->rex);
	free(p->rey);
	free(p->rez);
	free(p->relam);
	free(p->rexsi);
	free(p->reeta);
	free(p->remu);
	free(p->rezet);
	free(p->res);
	free(p->residu1);
	free(p->residu2);
	free(p->residu);
	free(p->gg);
	free(p->delx);
	free(p->dely);
	free(p->delz);
	free(p->dellam);
	free(p->diagx);
	free(p->diagxinv);
	free(p->diagy);
	free(p->diagyinv);
	free(p->diaglam);
	free(p->diaglamyi);
	free(p->blam);
	free(p->bb);
	free(p->alam);
	free(p->aa);
	free(p->aa_inv);
	free(p->solut);
	free(p->dlam);
	free(p->dz);
	free(p->dx);
	free(p->dy);
	free(p->dxsi);
	free(p->deta);
	free(p->dmu);
	free(p->dzet);
	free(p->ds);
	free(p->xx);
	free(p->dxx);
	free(p->stepxx);
	free(p->stepalfa);
	free(p->stepbeta);
	free(p->xold);
	free(p->yold);
	free(p->zold);
	free(p->lamold);
	free(p->xsiold);
	free(p->etaold);
	free(p->muold);
	free(p->zetold);
	free(p->sold);


}

void kktcheck_malloc(int m, int n, struct kktcheck_pointers *p) {
	int residu1size, residu2size, residusize;

	residu1size = n + (2 * m);
	residu2size = (2 * n) + (4 * m);
	residusize = residu1size + residu2size;

	p->rex = (double*)malloc(n * sizeof(double));
	p->rey = (double*)malloc(m * sizeof(double));
	p->rez = (double*)malloc(m * sizeof(double));
	p->relam = (double*)malloc(m * sizeof(double));
	p->rexsi = (double*)malloc(n * sizeof(double));
	p->reeta = (double*)malloc(n * sizeof(double));
	p->remu = (double*)malloc(m * sizeof(double));
	p->rezet = (double*)malloc(m * sizeof(double));
	p->res = (double*)malloc(m * sizeof(double));

	p->residu1 = (double*)malloc(residu1size * sizeof(double));
	p->residu2 = (double*)malloc(residu2size * sizeof(double));
	p->residu = (double*)malloc(residusize * sizeof(double));
}

void kktcheck_free(struct kktcheck_pointers *p) {
	free(p->rex);
	free(p->rey);
	free(p->rez);
	free(p->relam);
	free(p->rexsi);
	free(p->reeta);
	free(p->remu);
	free(p->rezet);
	free(p->res);
	free(p->residu1);
	free(p->residu2);
	free(p->residu);
}