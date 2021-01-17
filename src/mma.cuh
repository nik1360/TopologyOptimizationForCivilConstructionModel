#pragma once
#include "sparse_struc.cuh"
#include "sys.cuh"
#include "mesh.cuh"
#include "parallel.cuh"


//-----------------------------------------------------STRUCTURES-------------------------------------------------------------------------
struct mma_param {
	int m, n, *a, *c, *d;
	double epsimin, outeriter, maxoutit, kkttol, f0valsoglia, a0;
	double *x, *xval, *xold1, *xold2, *xmin, *xmax, *low, *upp;
	double *xmma, *ymma, *zmma, *lam, *xsi, *eta, *mu, *zet, *s;
	double *dfdx, *fval, *df0dx, f0val;
};

struct subsolv_pointers
{
	double *lam, *xsi, *eta, *mu, *zet, *s;
	double *een, *eem, *epsvecn, *epsvecm, *x, *y, *z;
	double *ux1, *ux2, *ux3, *xl1, *xl2, *xl3, *uxinv1, *xlinv1, *uxinv2, *xlinv2;
	double *plam, *qlam;
	double *gg, *dpsidx;
	double *rex, *rey, *rez, *relam, *rexsi, *reeta, *remu, *rezet, *res;
	double *residu1, *residu2, *residu;
	double *delx, *dely, *delz;
	double *dellam;
	double *diagx, *diagxinv, *diagy, *diagyinv, *diaglam, *diaglamyi;
	double *blam, *bb, *alam, *aa, *aa_inv, *solut;
	double *dlam, *dz, *dx, *dy;
	double *dxsi, *deta;
	double *dmu, *dzet, *ds;
	double *xx, *dxx, *stepxx;
	double *stepalfa, *stepbeta;
	double *xold, *yold, *zold, *lamold, *xsiold, *etaold, *muold, *zetold, *sold;

};

struct mmasub_pointers {
	int*eeen, *eeem, *zeron;
	double *zzz, *factor, *zzz1, *zzz2, *alfa, *beta;
	double *lowmin, *lowmax, *uppmin, *uppmax;
	double *xmami, *xmamieps, *xmamiinv, *ux1, *ux2, *xl1, *xl2, *uxinv, *xlinv, *p0, *q0, *pq0;
	double *p, *q, *pq;
	double b;

};

struct kktcheck_pointers {
	double *rex, *rey, *rez, *relam, *rexsi, *reeta, *remu, *rezet, *res;
	double *residu1, *residu2, *residu;
};


//--------------------------------------------------PROTOTIPES----------------------------------------------------------------------------

void mma_init(struct mma_param *mma_p, struct mesh *mesh);
void hinf_compute_mma(double *xphys, struct mma_param *mma_p, struct sparse *h, double *hs, struct mesh *mesh, struct material *material, int *ik, int *jk, double *me, double *ke,
	int *freedofs, int *fixeddofs, struct sparse *f, struct sys *g, struct sys *gec8, struct sys *g2, int *edofmat, cublasHandle_t cublas_handle, cusolverDnHandle_t cusolverH);
void mmasub(struct mma_param *mma_p, double f0val, double *df0dx, double *fval, double *dfdx, struct mesh *mesh, struct mmasub_pointers *msp, struct subsolv_pointers *ssp);
void kktcheck(struct mma_param *mma_p, struct kktcheck_pointers *kktp, double *df0dx, double *fval, double *dfdx, double*residunorm, double *residumax);

void mmasub_malloc(int m, int n, struct mmasub_pointers *p);
void mmasub_free(struct mmasub_pointers *p);
void subsolv_malloc(int m, int n, struct subsolv_pointers *p);
void subsolv_free(struct subsolv_pointers *p);
void mma_param_free(struct mma_param *mma_p);
void kktcheck_malloc(int m, int n, struct kktcheck_pointers *p);
void kktcheck_free(struct kktcheck_pointers *p);
