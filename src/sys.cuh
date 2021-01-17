#pragma once
#include "mesh.cuh"
#include "material.cuh"
#include "sparse_struc.cuh"





struct sys {
	
	struct sparse a;
	struct sparse b;
	struct sparse c;
	double d;
	struct sparse e;
	
};

void g_sys_init(struct sys *sys, int ndof, int nstate, struct sparse *a, struct sparse *b, struct sparse *c, double dstato, struct sparse *e);
void gec8_sys_init(struct sys *sys);
void sys_equal(struct sys *sys1, struct sys *sys2);
void sys_multiplication(struct sys *sys1, struct sys *sys2, struct sys *sys);
void sys_inf_norm(struct sys *sys, double *norm, double *h_inf_peak_freq);
void freqresp(struct sys *sys, double omega, double *value);
void free_sys(struct sys *sys);
