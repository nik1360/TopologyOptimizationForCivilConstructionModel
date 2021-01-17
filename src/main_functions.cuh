#pragma once
#include "mesh.cuh"
#include "material.cuh"
#include "sparse_struc.cuh"

void mesh_material_init(struct mesh *me, struct material *ma);
void stiffness_mass_init(double **ke, double **me, struct mesh *mesh, struct material *material);
void edofmat_init(int **edofmat,struct mesh *mesh);
void jk_ik_compute(int **ik,int **jk, int* edofmat, struct mesh *mesh);
void define_loads_support(int **fixeddofs, int **alldofs, int **freedofs, struct mesh *mesh);
void prepare_filter(int **ih, int **jh, double **sh, struct mesh *mesh, double **hs, struct sparse *h);

void mma(struct sparse *h ,double *hs, struct mesh *mesh, struct material *material, int *ik, int *jk, double *me, double *ke, int *freedofs, int*fixeddofs, struct sparse *f, int *edofmat);