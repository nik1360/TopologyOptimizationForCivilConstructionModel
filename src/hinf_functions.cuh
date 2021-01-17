#pragma once
#include "mesh.cuh"
#include "material.cuh"
#include "sparse_struc.cuh"

void compute_xphys(struct sparse *h, double *hs, double*x, double *xphys, struct mesh *mesh);
void compute_sk(double **sk, double *ke, double *xphys, struct mesh *mesh, struct material *material);
void compute_sm(double **sm, double *me, double *xphys, struct mesh *mesh, struct material *material);
void modify_m_k(struct sparse *mat, struct mesh *mesh, int *freedofs, int *fixeddofs);
void compute_s(struct sparse *k, struct sparse *m, struct sparse *s, struct mesh *mesh);
void g_matrix_init(int ndof, int nstate, struct sparse *a, struct sparse *b, struct sparse *c, double *dstato, struct sparse *e, struct sparse *k, struct sparse *m, struct sparse *s, int *freedofs, struct sparse *f);

void invforsens2(struct sparse *invu, struct sys *g, double h_inf_peak_frq);
void prosenshinf_2017(double *xphys, struct sparse *invu, int us, int vs, double h_inf_peak_frq, double *df0dx, double *me, double *ke, int *edofmat, struct mesh *mesh, struct material *material, int ntutto, int *freedofs, int *fixeddofs, struct sparse *bstato, struct sparse *cstato);
void filt_mod_sensitivities(struct mesh *mesh, double *x, double *df0dx, struct sparse *h, double *hs);
void volume_constraint(double **fval, double **dfdx, double *x, struct mesh *mesh, struct sparse *h, double *hs);

void free_sparse(struct sparse *a, struct sparse *b, struct sparse *c, struct sparse *e, struct sparse *k, struct sparse *m, struct sparse *s, struct sparse *invu);