#include <math.h>

#include "mesh.cuh"
#include "material.cuh"
#include "constants.cuh"

void material_init(struct material *mat);
void mesh_init(struct mesh *m);

void mesh_material_init(struct mesh *me, struct material *ma) {
	mesh_init(me);
	material_init(ma);
}

void mesh_init(struct mesh *m) {
	m->lx = LX;
	m->ly = LY;
	m->vmax = LX*LY;
	m->nelx = NELX;
	m->nely = NELY;
	m->volfrac = VOLFRAC;
	m->ax = (double)LX / (double)NELX;
	m->by = (double)LY / (double)NELY;
	m->area = m->ax*m->by;
	m->penal = PENAL;
	m->prho = PRHO;
	m->rmin = RMIN;
	m->ft = FT;

	m->alpha = ALPHA;
	m->beta = BETA;
	m->ninp = NINP;
	m->nout = NOUT;

	m->fixed_count = 0;
}

void material_init(struct material *mat) {
	mat->e0 = E0;
	mat->emin = 0.000001;
	mat->rho0 = RHO0;
	mat->rhomin = 0.000001;
	mat->nu = NU;

}

