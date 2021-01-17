#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "constants.cuh"
#include "matrix_functions.cuh"
#include "mesh.cuh"
#include "material.cuh"
#include "print.cuh"
#include "sys.cuh"
#include "mma.cuh"
#include "minmax.cuh"

void subsolv(struct mma_param *mma_p, double epsimin, struct mmasub_pointers *msp, struct subsolv_pointers *ssp);


void mmasub(struct mma_param *mma_p, double f0val, double *df0dx, double *fval, double *dfdx, struct mesh *mesh, struct mmasub_pointers *msp, struct subsolv_pointers *ssp)
{

	double epsimin, raa0, move, albefa, asyinit, asyincr, asydecr;
	double sum1, sum2;

	epsimin = pow(10, -7);
	raa0 = 0.00001;
	move = 1.0;
	albefa = 0.1;
	asyinit = 0.5;
	asyincr = 1.2;
	asydecr = 0.7;

	for (int i = 0; i < mma_p->n; i++) {	//eeen = ones(n,1);   zeron = zeros(n,1);
		msp->eeen[i] = 1;
		msp->zeron[i] = 0;
	}
	for (int i = 0; i < mma_p->m; i++) {	//eeem = ones(m,1);
		msp->eeem[i] = 1;
	}

	if (mma_p->outeriter < 2.5) {
		for (int i = 0; i <mma_p->n; i++) {
			mma_p->low[i] = mma_p->xval[i] - (asyinit * (mma_p->xmax[i] - mma_p->xmin[i]));
			mma_p->upp[i] = mma_p->xval[i] + (asyinit * (mma_p->xmax[i] - mma_p->xmin[i]));
		}
	}
	else {
		for (int i = 0; i < mma_p->n; i++) {	//zzz = (xval-xold1).*(xold1-xold2);
			msp->zzz[i] = (mma_p->xval[i] - mma_p->xold1[i])*(mma_p->xold1[i] - mma_p->xold2[i]);
			msp->factor[i] = msp->eeen[i];
		}
		for (int i = 0; i <mma_p->n; i++) {	//factor(find(zzz > 0)) = asyincr;   factor(find(zzz < 0)) = asydecr;
			if (msp->zzz[i]>0.0) {
				msp->factor[i] = asyincr;
			}
			if (msp->zzz[i]<0.0) {
				msp->factor[i] = asydecr;
			}
		}
		for (int i = 0; i < mma_p->n; i++) {
			mma_p->low[i] = mma_p->xval[i] - (msp->factor[i] * (mma_p->xold1[i] - mma_p->low[i]));	//low = xval - factor.*(xold1 - low);
			mma_p->upp[i] = mma_p->xval[i] + (msp->factor[i] * (mma_p->upp[i] - mma_p->xold1[i]));	//upp = xval + factor.*(upp - xold1);

			msp->lowmin[i] = mma_p->xval[i] - 10 * (mma_p->xmax[i] - mma_p->xmin[i]);	 //lowmin = xval - 10*(xmax-xmin);
			msp->lowmax[i] = mma_p->xval[i] - 0.01*(mma_p->xmax[i] - mma_p->xmin[i]);	//lowmax = xval - 0.01*(xmax-xmin);
			msp->uppmin[i] = mma_p->xval[i] + 0.01*(mma_p->xmax[i] - mma_p->xmin[i]); //uppmin = xval + 0.01*(xmax-xmin);
			msp->uppmax[i] = mma_p->xval[i] + 10 * (mma_p->xmax[i] - mma_p->xmin[i]); //uppmax = xval + 10*(xmax-xmin);
		}

		for (int i = 0; i < mma_p->n; i++) {	// low = max(low,lowmin); upp = min(upp,uppmax);
			mma_p->low[i] = maxVal(mma_p->low[i], msp->lowmin[i]);
			mma_p->upp[i] = minVal(mma_p->upp[i], msp->uppmax[i]);
		}
		for (int i = 0; i < mma_p->n; i++) { //low = min(low,lowmax); upp = max(upp,uppmin);
			mma_p->low[i] = minVal(mma_p->low[i], msp->lowmax[i]);
			mma_p->upp[i] = maxVal(mma_p->upp[i], msp->uppmin[i]);
		}
	}

	//--calculations of the bounds alfa and beta
	for (int i = 0; i < mma_p->n; i++) {	//zzz1 = low + albefa*(xval-low);   zzz2 = xval - move*(xmax-xmin);	
		msp->zzz1[i] = mma_p->low[i] + (albefa*(mma_p->xval[i] - mma_p->low[i]));
		msp->zzz2[i] = mma_p->xval[i] - (move*(mma_p->xmax[i] - mma_p->xmin[i]));
		msp->zzz[i] = maxVal(msp->zzz1[i], msp->zzz2[i]); //zzz  = max(zzz1,zzz2);
		msp->alfa[i] = maxVal(msp->zzz[i], mma_p->xmin[i]); //alfa = max(zzz,xmin);
	}

	for (int i = 0; i < mma_p->n; i++) {	//zzz1 = upp - albefa*(upp-xval);  zzz2 = xval + move*(xmax-xmin);
		msp->zzz1[i] = mma_p->upp[i] - (albefa*(mma_p->upp[i] - mma_p->xval[i]));
		msp->zzz2[i] = mma_p->xval[i] + (move*(mma_p->xmax[i] - mma_p->xmin[i]));
		msp->zzz[i] = minVal(msp->zzz1[i], msp->zzz2[i]);	//zzz  = min(zzz1,zzz2);
		msp->beta[i] = minVal(msp->zzz[i], mma_p->xmax[i]);	//beta = min(zzz,xmax);
	}

	//--Calculations of p0, q0, P, Q and b
	for (int i = 0; i <mma_p->n; i++) {
		msp->xmami[i] = mma_p->xmax[i] - mma_p->xmin[i];	//xmami = xmax-xmin;
		msp->xmamieps[i] = 0.00001*msp->eeen[i];	//xmamieps = 0.00001*eeen;
		msp->xmami[i] = maxVal(msp->xmami[i], msp->xmamieps[i]); //xmami = max(xmami,xmamieps);
		msp->xmamiinv[i] = msp->eeen[i] / msp->xmami[i]; //xmamiinv = eeen./xmami;
		msp->ux1[i] = mma_p->upp[i] - mma_p->xval[i]; //ux1 = upp-xval;
		msp->ux2[i] = msp->ux1[i] * msp->ux1[i]; //ux2 = ux1.*ux1;
		msp->xl1[i] = mma_p->xval[i] - mma_p->low[i]; //xl1 = xval-low;
		msp->xl2[i] = msp->xl1[i] * msp->xl1[i]; //xl2 = xl1.*xl1;
		msp->uxinv[i] = msp->eeen[i] / msp->ux1[i]; //uxinv = eeen./ux1;
		msp->xlinv[i] = msp->eeen[i] / msp->xl1[i]; //xlinv = eeen./xl1;


		msp->p0[i] = msp->zeron[i];	//p0 = zeron;
		msp->q0[i] = msp->zeron[i];	//q0 = zeron;
		msp->p0[i] = maxVal((double)df0dx[i], 0.);	//p0 = max(df0dx,0);
		msp->q0[i] = maxVal(-df0dx[i], 0.);	//q0 = max(-df0dx,0);
		msp->pq0[i] = (0.001*(msp->p0[i] + msp->q0[i])) + (raa0*msp->xmamiinv[i]); //pq0 = 0.001*(p0 + q0) + raa0*xmamiinv;

		msp->p0[i] = msp->p0[i] + msp->pq0[i];	// p0 = p0 + pq0;
		msp->q0[i] = msp->q0[i] + msp->pq0[i]; //q0 = q0 + pq0;
		msp->p0[i] = msp->p0[i] * msp->ux2[i]; //p0 = p0.*ux2;
		msp->q0[i] = msp->q0[i] * msp->xl2[i]; //q0 = q0.*xl2;

		msp->p[i] = maxVal(dfdx[i], 0.);	//P = max(dfdx,0);
		msp->q[i] = maxVal(-dfdx[i], 0.); //Q = max(-dfdx,0);

	}

	sum1 = 0;
	sum2 = 0;

	for (int i = 0; i < mma_p->n; i++) {
		msp->pq[i] = (0.001*(msp->p[i] + msp->q[i])) + (raa0*msp->eeem[0] * msp->xmamiinv[i]); //PQ = 0.001*(P + Q) + raa0*eeem*xmamiinv';
		msp->p[i] = msp->p[i] + msp->pq[i]; //P = P + PQ;
		msp->q[i] = msp->q[i] + msp->pq[i]; //Q = Q + PQ;
		msp->p[i] = msp->p[i] * msp->ux2[i]; //P = P * spdiags(ux2,0,n,n);
		msp->q[i] = msp->q[i] * msp->xl2[i]; //Q = Q * spdiags(xl2,0,n,n);

		sum1 += msp->p[i] * msp->uxinv[i];	//P*uxinv
		sum2 += msp->q[i] * msp->xlinv[i];	//Q*xlinv
	}
	msp->b = sum1 + sum2 - fval[0];	//b = P*uxinv + Q*xlinv - fval ;
	subsolv(mma_p, epsimin, msp, ssp);


}

void subsolv(struct mma_param *mma_p, double epsimin, struct mmasub_pointers *msp, struct subsolv_pointers *ssp) {


	int itera, ittt, itto, residu1size, residu2size, residusize, residu2_index = 0;;
	double epsi;
	double gvec = 0;
	double residunorm, residumax, resinew;
	double tmp_val = 0, tmp_val2 = 0, *tmp_vec;
	double stmxx;
	double stmalfa, stmbeta;
	double stmalbe, stmalbexx, stminv, steg;

	tmp_vec = (double*)malloc(mma_p->n * sizeof(double));

	residu1size = mma_p->n + (2 * mma_p->m);
	residu2size = (2 * mma_p->n) + (4 * mma_p->m);
	residusize = residu1size + residu2size;


	epsi = 1;
	ssp->z[0] = 1; //z = 1;
	ssp->zet[0] = 1; //zet=1;

	for (int i = 0; i < mma_p->n; i++) {
		ssp->een[i] = 1;	//een = ones(n,1);
		ssp->epsvecn[i] = epsi * ssp->een[i];	//epsvecn = epsi*een;
		ssp->x[i] = 0.5*(msp->alfa[i] + msp->beta[i]); //x = 0.5*(alfa + beta);
		ssp->xsi[i] = ssp->een[i] / (ssp->x[i] - msp->alfa[i]); //xsi = een./(x-alfa);
		ssp->xsi[i] = maxVal(ssp->xsi[i], ssp->een[i]); //xsi = max(xsi,een);
		ssp->eta[i] = ssp->een[i] / (msp->beta[i] - ssp->x[i]); //eta = een. / (beta - x);
		ssp->eta[i] = maxVal(ssp->eta[i], ssp->een[i]); //eta = max(eta, een);

		ssp->gg[i] = 0;	//gg is used later
		tmp_vec[i] = 0; //tmp_vec is used later
	}

	for (int i = 0; i < mma_p->m; i++) {
		ssp->eem[i] = 1;	//eem = ones(m,1);
		ssp->epsvecm[i] = epsi * ssp->eem[i];	//epsvecm = epsi*eem;
		ssp->y[i] = ssp->eem[i]; //y = eem;
		ssp->lam[i] = ssp->eem[i];	//lam = eem;
		ssp->mu[i] = maxVal(ssp->eem[i], 0.5*mma_p->c[i]); //mu  = max(eem,0.5*c);
		ssp->s[i] = ssp->eem[i];
	}

	itera = 0;
	while (epsi > epsimin) {

		gvec = 0;
		for (int i = 0; i < mma_p->n; i++) {
			ssp->epsvecn[i] = epsi * ssp->een[i];	//epsvecn = epsi*een;
			ssp->ux1[i] = mma_p->upp[i] - ssp->x[i];	//ux1 = upp-x;
			ssp->xl1[i] = ssp->x[i] - mma_p->low[i];	//xl1 = x-low;
			ssp->ux2[i] = ssp->ux1[i] * ssp->ux1[i]; //ux2 = ux1.*ux1;
			ssp->xl2[i] = ssp->xl1[i] * ssp->xl1[i]; //xl2 = xl1.*xl1;
			ssp->uxinv1[i] = ssp->een[i] / ssp->ux1[i]; //uxinv1 = een./ux1;
			ssp->xlinv1[i] = ssp->een[i] / ssp->xl1[i]; //xlinv1 = een./xl1;
			ssp->plam[i] = msp->p0[i] + (msp->p[i] * ssp->lam[0]);	//plam = p0 + P'*lam ;
			ssp->qlam[i] = msp->q0[i] + (msp->q[i] * ssp->lam[0]);	//qlam = q0 + Q'*lam ;
			gvec += msp->p[i] * ssp->uxinv1[i]; // gvec = P*uxinv1 + Q*xlinv1;
			gvec += msp->q[i] * ssp->xlinv1[i];
			ssp->dpsidx[i] = (ssp->plam[i] / ssp->ux2[i]) - (ssp->qlam[i] / ssp->xl2[i]); //dpsidx = plam./ux2 - qlam./xl2 ;
			ssp->rex[i] = ssp->dpsidx[i] - ssp->xsi[i] + ssp->eta[i];	//rex = dpsidx - xsi + eta;
			ssp->rexsi[i] = ssp->xsi[i] * (ssp->x[i] - msp->alfa[i]) - ssp->epsvecn[i]; // rexsi = xsi.*(x-alfa) - epsvecn;
			ssp->reeta[i] = ssp->eta[i] * (msp->beta[i] - ssp->x[i]) - ssp->epsvecn[i]; //reeta = eta.*(beta-x) - epsvecn;

			ssp->residu1[i] = ssp->rex[i];

		}
		for (int i = 0; i < mma_p->m; i++) {
			ssp->epsvecm[i] = epsi * ssp->eem[i];
			ssp->rey[i] = mma_p->c[i] + (mma_p->d[i] * ssp->y[i]) - ssp->mu[i] - ssp->lam[i]; // rey = c + d.*y - mu - lam;
			ssp->rez[i] = mma_p->a0 - ssp->zet[i] - (mma_p->a[i] * ssp->lam[i]); //rez = a0 - zet - a'*lam;
			ssp->relam[i] = gvec - (mma_p->a[i] * ssp->z[i]) - ssp->y[i] + ssp->s[i] - msp->b; //relam = gvec - a*z - y + s - b;
			ssp->remu[i] = (ssp->mu[i] * ssp->y[i]) - ssp->epsvecm[i]; //reeta = eta.*(beta - x) - epsvecn;
			ssp->rezet[i] = (ssp->zet[i] * ssp->z[i]) - epsi; //rezet = zet*z - epsi;
			ssp->res[i] = (ssp->lam[i] * ssp->s[i]) - ssp->epsvecm[i]; //res = lam.*s - epsvecm;

			ssp->residu1[i + mma_p->n] = ssp->rey[i];
			ssp->residu1[i + mma_p->n + mma_p->m] = ssp->rez[i];

		}

		//residu2 = [relam' rexsi' reeta' remu' rezet res']';

		residu2_index = 0;
		ssp->residu2[residu2_index] = ssp->relam[0];
		residu2_index++;
		for (int i = 0; i <mma_p->n; i++) {
			ssp->residu2[residu2_index] = ssp->rexsi[i];
			residu2_index++;
		}
		for (int i = 0; i < mma_p->n; i++) {
			ssp->residu2[residu2_index] = ssp->reeta[i];
			residu2_index++;
		}
		ssp->residu2[residu2_index] = ssp->remu[0];
		residu2_index++;
		ssp->residu2[residu2_index] = ssp->rezet[0];
		residu2_index++;
		ssp->residu2[residu2_index] = ssp->res[0];
		residu2_index++;


		//residu = [residu1' residu2']';
		//residunorm = sqrt(residu'*residu);
		//residumax = max(abs(residu));

		residumax = fabs(ssp->residu1[0]);
		residunorm = 0;
		for (int i = 0; i < residu1size; i++) {
			ssp->residu[i] = ssp->residu1[i];
			residunorm += ssp->residu[i] * ssp->residu[i];
			if (ssp->residu[i] > residumax)
				residumax = fabs(ssp->residu[i]);
		}
		for (int i = 0; i < residu2size; i++) {
			ssp->residu[i + residu1size] = ssp->residu2[i];
			residunorm += ssp->residu[i + residu1size] * ssp->residu[i + residu1size];
			if (fabs(ssp->residu[i + residu1size]) > residumax)
				residumax = fabs(ssp->residu[i + residu1size]);
		}

		residunorm = sqrt(residunorm);

		ittt = 0;
		while ((residumax>0.9*epsi) && (ittt<200)) {
			tmp_val = 0;
			tmp_val2 = 0;
			ittt = ittt + 1;
			itera = itera + 1;
			gvec = 0;
			for (int i = 0; i < mma_p->n; i++) {
				ssp->gg[i] = 0;	//gg is used later
				tmp_vec[i] = 0; //tmp_vec is used later
			}
			for (int i = 0; i < mma_p->n; i++) {
				ssp->epsvecn[i] = epsi * ssp->een[i];	//epsvecn = epsi*een;
				ssp->ux1[i] = mma_p->upp[i] - ssp->x[i];	//ux1 = upp-x;
				ssp->xl1[i] = ssp->x[i] - mma_p->low[i];	//xl1 = x-low;
				ssp->ux2[i] = ssp->ux1[i] * ssp->ux1[i]; //ux2 = ux1.*ux1;
				ssp->xl2[i] = ssp->xl1[i] * ssp->xl1[i]; //xl2 = xl1.*xl1;
				ssp->ux3[i] = ssp->ux1[i] * ssp->ux2[i]; //ux3 = ux1.*ux2;
				ssp->xl3[i] = ssp->xl1[i] * ssp->xl2[i]; //xl3 = xl1.*xl2;
				ssp->uxinv1[i] = ssp->een[i] / ssp->ux1[i]; //uxinv1 = een./ux1;
				ssp->xlinv1[i] = ssp->een[i] / ssp->xl1[i]; //xlinv1 = een./xl1;
				ssp->uxinv2[i] = ssp->een[i] / ssp->ux2[i]; //uxinv1 = een./ux1;
				ssp->xlinv2[i] = ssp->een[i] / ssp->xl2[i]; //xlinv1 = een./xl1;
				ssp->plam[i] = msp->p0[i] + (msp->p[i] * ssp->lam[0]);	//plam = p0 + P'*lam ;
				ssp->qlam[i] = msp->q0[i] + (msp->q[i] * ssp->lam[0]);	//qlam = q0 + Q'*lam ;
				gvec += msp->p[i] * ssp->uxinv1[i]; // gvec = P*uxinv1 + Q*xlinv1;
				gvec += msp->q[i] * ssp->xlinv1[i];
				ssp->gg[i] += msp->p[i] * ssp->uxinv2[i]; // GG = P * spdiags(uxinv2, 0, n, n) - Q * spdiags(xlinv2, 0, n, n);
				ssp->gg[i] -= msp->q[i] * ssp->xlinv2[i];

				ssp->dpsidx[i] = (ssp->plam[i] / ssp->ux2[i]) - (ssp->qlam[i] / ssp->xl2[i]); //dpsidx = plam./ux2 - qlam./xl2 ;
				ssp->delx[i] = ssp->dpsidx[i] - (ssp->epsvecn[i] / (ssp->x[i] - msp->alfa[i])) + (ssp->epsvecn[i] / (msp->beta[i] - ssp->x[i]));	//delx = dpsidx - epsvecn./(x-alfa) + epsvecn./(beta-x);
				ssp->diagx[i] = (ssp->plam[i] / ssp->ux3[i]) + (ssp->qlam[i] / ssp->xl3[i]); //diagx = plam./ux3 + qlam./xl3;
				ssp->diagx[i] = (2 * ssp->diagx[i]) + (ssp->xsi[i] / (ssp->x[i] - msp->alfa[i])) + (ssp->eta[i] / (msp->beta[i] - ssp->x[i])); //diagx = 2 * diagx + xsi. / (x - alfa) + eta. / (beta - x);

				ssp->diagxinv[i] = ssp->een[i] / ssp->diagx[i]; //diagxinv = een./diagx;

				tmp_val += ssp->gg[i] * (ssp->delx[i] / ssp->diagx[i]); //GG*(delx./diagx)
																		//GG*spdiags(diagxinv,0,n,n)*GG'
				tmp_vec[i] += ssp->gg[i] * ssp->diagxinv[i];
				tmp_val2 += tmp_vec[i] * ssp->gg[i];
			}
			for (int i = 0; i <mma_p->m; i++) {
				ssp->epsvecm[i] = epsi * ssp->eem[i];	//epsvecm = epsi*eem;
				ssp->dely[i] = mma_p->c[i] + (mma_p->d[i] * ssp->y[i]) - ssp->lam[i] - (ssp->epsvecm[i] / ssp->y[i]);
				ssp->delz[i] = mma_p->a0 - (mma_p->a[i] * ssp->lam[i]) - (epsi / ssp->z[0]);
				ssp->dellam[i] = gvec - (mma_p->a[i] * ssp->z[0]) - ssp->y[i] - msp->b + (ssp->epsvecm[i] / ssp->lam[i]);
				ssp->diagy[i] = mma_p->d[i] + (ssp->mu[i] / ssp->y[i]); //diagy = d + mu. / y;
				ssp->diagyinv[i] = ssp->eem[i] / ssp->diagy[i]; //diagyinv = eem. / diagy;
				ssp->diaglam[i] = ssp->s[i] / ssp->lam[i]; //diaglam = s./lam;
				ssp->diaglamyi[i] = ssp->diaglam[i] + ssp->diagyinv[i]; //diaglamyi = diaglam+diagyinv;
				ssp->blam[i] = ssp->dellam[i] + (ssp->dely[i] / ssp->diagy[i]) - tmp_val;

				ssp->bb[i] = ssp->blam[i];	// bb = [blam' delz]';
				ssp->bb[i + mma_p->m] = ssp->delz[i];

				ssp->alam[i] = ssp->diaglamyi[i] + tmp_val2; //Alam = spdiags(diaglamyi,0,m,m) + GG*spdiags(diagxinv,0,n,n)*GG';

				ssp->aa[i] = ssp->alam[i];	//AA = [Alam     a;a'    -zet/z ];
				ssp->aa[i + mma_p->m] = mma_p->a[i];
				ssp->aa[i + (2 * mma_p->m)] = mma_p->a[i];
				ssp->aa[i + (3 * mma_p->m)] = -ssp->zet[i] / ssp->z[i];

				matrix_inverse(ssp->aa, ssp->aa_inv, 2 * mma_p->m);	// solut = AA\bb;
				for (int j = 0; j < 2 * mma_p->m; j++) {
					ssp->solut[j] = 0;
				}
				for (int j = 0; j < 2 * mma_p->m; j++) {
					for (int k = 0; k < 2 * mma_p->m; k++) {
						ssp->solut[j] += ssp->aa_inv[j * 2 * mma_p->m + k] * ssp->bb[k];
					}
				}
				ssp->dlam[i] = ssp->solut[i];	//dlam = solut(1:m);
				ssp->dz[i] = ssp->solut[mma_p->m];	//dz = solut(m+1);
				ssp->dy[i] = (-ssp->dely[i] / ssp->diagy[i]) + (ssp->dlam[i] / ssp->diagy[i]); // dy = -dely./diagy + dlam./diagy;

				ssp->dmu[i] = -ssp->mu[i] + (ssp->epsvecm[i] / ssp->y[i]) - ((ssp->mu[i] * ssp->dy[i]) / ssp->y[i]);
				ssp->dzet[i] = -ssp->zet[i] + (epsi / ssp->z[i]) - ((ssp->zet[i] * ssp->dz[i]) / ssp->z[i]);
				ssp->ds[i] = -ssp->s[i] + (ssp->epsvecm[i] / ssp->lam[i]) - ((ssp->s[i] * ssp->dlam[i]) / ssp->lam[i]);

				ssp->yold[i] = ssp->y[i];	//yold   =   y;
				ssp->zold[i] = ssp->z[i];	//zold   =   z;
				ssp->lamold[i] = ssp->lam[i]; //lamold =  lam;
				ssp->muold[i] = ssp->mu[i];//muold  =  mu;
				ssp->zetold[i] = ssp->zet[i];	//zetold =  zet;
				ssp->sold[i] = ssp->s[i];	//sold   =   s;
			}

			ssp->dx[0] = (-ssp->delx[0] / ssp->diagx[0]) - ((ssp->dlam[0] * ssp->gg[0]) / ssp->diagx[0]);
			stmalfa = -1.01*ssp->dx[0] / (ssp->x[0] - msp->alfa[0]);
			stmbeta = 1.01*ssp->dx[0] / (msp->beta[0] - ssp->x[0]);

			for (int i = 0; i < mma_p->n; i++) {
				ssp->dx[i] = (-ssp->delx[i] / ssp->diagx[i]) - ((ssp->dlam[0] * ssp->gg[i]) / ssp->diagx[i]);//dx = -delx./diagx - (GG'*dlam)./diagx;
				ssp->dxsi[i] = -ssp->xsi[i] + (ssp->epsvecn[i] / (ssp->x[i] - msp->alfa[i])) - ((ssp->xsi[i] * ssp->dx[i]) / (ssp->x[i] - msp->alfa[i])); //dxsi = -xsi + epsvecn./(x-alfa) - (xsi.*dx)./(x-alfa);
				ssp->deta[i] = -ssp->eta[i] + (ssp->epsvecn[i] / (msp->beta[i] - ssp->x[i])) + ((ssp->eta[i] * ssp->dx[i]) / (msp->beta[i] - ssp->x[i])); //deta = -eta + epsvecn./(beta-x) + (eta.*dx)./(beta-x);

				ssp->stepalfa[i] = -1.01*ssp->dx[i] / (ssp->x[i] - msp->alfa[i]); //stepalfa = -1.01*dx./(x-alfa);
				if (ssp->stepalfa[i] > stmalfa)	//stmalfa = max(stepalfa);
					stmalfa = ssp->stepalfa[i];
				ssp->stepbeta[i] = 1.01*ssp->dx[i] / (msp->beta[i] - ssp->x[i]); //stepbeta = 1.01*dx./(beta-x);
				if (ssp->stepbeta[i] > stmbeta)	//stmbeta = max(stepbeta);
					stmbeta = ssp->stepbeta[i];

				ssp->xold[i] = ssp->x[i]; //xold   =   x;
				ssp->xsiold[i] = ssp->xsi[i]; //xsiold = xsi;
				ssp->etaold[i] = ssp->eta[i];  //etaold =  eta;
			}

			//xx = [y'  z  lam'  xsi'  eta'  mu'  zet  s']';
			ssp->xx[0] = ssp->y[0];
			ssp->xx[mma_p->m] = ssp->z[0];
			ssp->xx[2 * mma_p->m] = ssp->lam[0];
			for (int i = 3; i < mma_p->n + 3; i++) {
				ssp->xx[i] = ssp->xsi[i - 3];
				ssp->xx[i + mma_p->n] = ssp->eta[i - 3];
			}
			ssp->xx[(2 * mma_p->n) + 3 * mma_p->m] = ssp->mu[0];
			ssp->xx[(2 * mma_p->n) + 4 * mma_p->m] = ssp->zet[0];
			ssp->xx[(2 * mma_p->n) + 5 * mma_p->m] = ssp->s[0];

			//dxx = [dy' dz dlam' dxsi' deta' dmu' dzet ds']';
			ssp->dxx[0] = ssp->dy[0];
			ssp->dxx[mma_p->m] = ssp->dz[0];
			ssp->dxx[2 * mma_p->m] = ssp->dlam[0];
			for (int i = 3; i < mma_p->n + 3; i++) {
				ssp->dxx[i] = ssp->dxsi[i - 3];
				ssp->dxx[i + mma_p->n] = ssp->deta[i - 3];
			}
			ssp->dxx[(2 * mma_p->n) + 3 * mma_p->m] = ssp->dmu[0];
			ssp->dxx[(2 * mma_p->n) + 4 * mma_p->m] = ssp->dzet[0];
			ssp->dxx[(2 * mma_p->n) + 5 * mma_p->m] = ssp->ds[0];

			stmxx = -1.01*(ssp->dxx[0] / ssp->xx[0]);
			for (int i = 0; i < (6 * mma_p->m + 2 * mma_p->n); i++) {
				ssp->stepxx[i] = -1.01*(ssp->dxx[i] / ssp->xx[i]); //stepxx = -1.01*dxx./xx;
				if (ssp->stepxx[i] > stmxx)	//stmxx  = max(stepxx);
					stmxx = ssp->stepxx[i];
			}



			stmalbe = maxVal(stmalfa, stmbeta);
			stmalbexx = maxVal(stmalbe, stmxx);
			stminv = maxVal(stmalbexx, 1);
			steg = 1 / stminv;

			itto = 0;
			resinew = 2 * residunorm;

			while ((resinew >residunorm) && (itto<50)) {
				itto = itto + 1;
				gvec = 0;

				for (int i = 0; i < mma_p->m; i++) {
					ssp->y[i] = ssp->yold[i] + steg * ssp->dy[i];	// y   =   yold + steg*dy;
					ssp->z[i] = ssp->zold[i] + steg * ssp->dz[i];	//z   =   zold + steg*dz;
					ssp->lam[i] = ssp->lamold[i] + steg * ssp->dlam[i];	//lam = lamold + steg*dlam;
					ssp->mu[i] = ssp->muold[i] + steg * ssp->dmu[i];	//mu  = muold  + steg*dmu;
					ssp->zet[i] = ssp->zetold[i] + steg * ssp->dzet[i];	//zet = zetold + steg*dzet;
					ssp->s[i] = ssp->sold[i] + steg * ssp->ds[i];	//s   =   sold + steg*ds;

					ssp->rey[i] = mma_p->c[i] + (mma_p->d[i] * ssp->y[i]) - ssp->mu[i] - ssp->lam[i]; // rey = c + d.*y - mu - lam;
					ssp->rez[i] = mma_p->a0 - ssp->zet[i] - (mma_p->a[i] * ssp->lam[i]); //rez = a0 - zet - a'*lam;

				}

				for (int i = 0; i < mma_p->n; i++) {
					ssp->x[i] = ssp->xold[i] + steg * ssp->dx[i];	//x   =   xold + steg*dx;
					ssp->xsi[i] = ssp->xsiold[i] + steg * ssp->dxsi[i];	//xsi = xsiold + steg * dxsi;
					ssp->eta[i] = ssp->etaold[i] + steg * ssp->deta[i];	// eta = etaold + steg*deta;
					ssp->ux1[i] = mma_p->upp[i] - ssp->x[i];	//ux1 = upp-x;
					ssp->xl1[i] = ssp->x[i] - mma_p->low[i];	//xl1 = x-low;
					ssp->ux2[i] = ssp->ux1[i] * ssp->ux1[i]; //ux2 = ux1.*ux1;
					ssp->xl2[i] = ssp->xl1[i] * ssp->xl1[i]; //xl2 = xl1.*xl1;
					ssp->uxinv1[i] = ssp->een[i] / ssp->ux1[i]; //uxinv1 = een./ux1;
					ssp->xlinv1[i] = ssp->een[i] / ssp->xl1[i]; //xlinv1 = een./xl1;
					ssp->plam[i] = msp->p0[i] + (msp->p[i] * ssp->lam[0]);	//plam = p0 + P'*lam ;
					ssp->qlam[i] = msp->q0[i] + (msp->q[i] * ssp->lam[0]);	//qlam = q0 + Q'*lam ;
					gvec += msp->p[i] * ssp->uxinv1[i]; // gvec = P*uxinv1 + Q*xlinv1;
					gvec += msp->q[i] * ssp->xlinv1[i];
					ssp->dpsidx[i] = (ssp->plam[i] / ssp->ux2[i]) - (ssp->qlam[i] / ssp->xl2[i]); //dpsidx = plam./ux2 - qlam./xl2 ;
					ssp->rex[i] = ssp->dpsidx[i] - ssp->xsi[i] + ssp->eta[i];	//rex = dpsidx - xsi + eta;
					ssp->rexsi[i] = ssp->xsi[i] * (ssp->x[i] - msp->alfa[i]) - ssp->epsvecn[i]; // rexsi = xsi.*(x-alfa) - epsvecn;
					ssp->reeta[i] = ssp->eta[i] * (msp->beta[i] - ssp->x[i]) - ssp->epsvecn[i]; //reeta = eta.*(beta-x) - epsvecn;

					ssp->residu1[i] = ssp->rex[i];

				}

				for (int i = 0; i < mma_p->m; i++) {
					ssp->relam[i] = gvec - (mma_p->a[i] * ssp->z[i]) - ssp->y[i] + ssp->s[i] - msp->b; //relam = gvec - a*z - y + s - b;
					ssp->remu[i] = (ssp->mu[i] * ssp->y[i]) - ssp->epsvecm[i]; //reeta = eta.*(beta - x) - epsvecn;
					ssp->rezet[i] = (ssp->zet[i] * ssp->z[i]) - epsi; //rezet = zet*z - epsi;
					ssp->res[i] = ssp->lam[i] * ssp->s[i] - ssp->epsvecm[i]; //res = lam.*s - epsvecm;

					ssp->residu1[i + mma_p->n] = ssp->rey[i];
					ssp->residu1[i + mma_p->n + mma_p->m] = ssp->rez[i];

				}


				//residu2 = [relam' rexsi' reeta' remu' rezet res']';
				residu2_index = 0;
				ssp->residu2[residu2_index] = ssp->relam[0];
				residu2_index++;
				for (int i = 0; i < mma_p->n; i++) {
					ssp->residu2[residu2_index] = ssp->rexsi[i];
					residu2_index++;
				}
				for (int i = 0; i < mma_p->n; i++) {
					ssp->residu2[residu2_index] = ssp->reeta[i];
					residu2_index++;
				}
				ssp->residu2[residu2_index] = ssp->remu[0];
				residu2_index++;
				ssp->residu2[residu2_index] = ssp->rezet[0];
				residu2_index++;
				ssp->residu2[residu2_index] = ssp->res[0];
				residu2_index++;

				resinew = 0;
				for (int i = 0; i < residu1size; i++) {
					ssp->residu[i] = ssp->residu1[i];
					resinew += ssp->residu[i] * ssp->residu[i];
				}
				for (int i = 0; i < residu2size; i++) {
					ssp->residu[i + residu1size] = ssp->residu2[i];
					resinew += ssp->residu[i + residu1size] * ssp->residu[i + residu1size];
				}


				resinew = sqrt(resinew);
				steg = steg / 2;
			}

			residunorm = resinew;

			//residumax = max(abs(residu));


			residumax = fabs(ssp->residu[0]);

			for (int i = 0; i < residusize; i++) {

				if (fabs(ssp->residu[i]) > residumax)
					residumax = fabs(ssp->residu[i]);
			}
			steg = 2 * steg;
		}

		epsi = 0.1*epsi;
	}

	for (int i = 0; i <mma_p->n; i++) {
		mma_p->xmma[i] = ssp->x[i];
		mma_p->xsi[i] = ssp->xsi[i];
		mma_p->eta[i] = ssp->eta[i];
	}
	for (int i = 0; i < mma_p->m; i++) {
		mma_p->ymma[i] = ssp->y[i];
		mma_p->zmma[i] = ssp->z[i];
		mma_p->lam[i] = ssp->lam[i];
		mma_p->mu[i] = ssp->mu[i];
		mma_p->zet[i] = ssp->zet[i];
		mma_p->s[i] = ssp->s[i];
	}

	free(tmp_vec);
}