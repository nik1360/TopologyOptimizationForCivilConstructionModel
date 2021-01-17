#include <stdlib.h>
#include <math.h>

#include "constants.cuh"
#include "mesh.cuh"
#include "material.cuh"

//------------------------------------------------------PROTOTIPES--------------------------------
void stiffness_init(double *ke, struct mesh *mesh, struct material *material);
void mass_init(double *me, struct mesh *mesh, struct material *material);
void init_stiff_t_array(double *t, struct mesh *mesh, struct material *material);
void populate_stiff_mat(double *t, double *ke, struct mesh *mesh);
void init_mass_t_array(double *t, struct mesh *mesh, struct material *material);
void populate_mass_mat(double *t, double *ke);

//------------------------------------------------------BODIES-------------------------------------
void stiffness_mass_init(double **ke, double **me, struct mesh *mesh, struct material *material) {
	(*ke) = (double*)malloc(KEROW*KECOL * sizeof(double));
	(*me) = (double*)malloc(MEROW*MECOL * sizeof(double));

	stiffness_init((*ke), mesh, material);
	mass_init((*me), mesh, material);
}

void stiffness_init(double *ke, struct mesh *mesh, struct material *material) {
	double *t;
	t = (double*)malloc(31 * sizeof(double));
	init_stiff_t_array(t, mesh, material);
	populate_stiff_mat(t, ke, mesh);
	free(t);

}

void mass_init(double *me, struct mesh *mesh, struct material *material) {
	double *t;
	t = (double*)malloc(3 * sizeof(double));
	init_mass_t_array(t, mesh, material);
	populate_mass_mat(t, me);

	free(t);
	
}

void init_stiff_t_array(double *t, struct mesh *mesh, struct material *material) {
	t[0] = material->nu + 1.0; //t2
	t[1] = 1.0 / t[0];	//t3
	t[2] = material->nu * 2.0; //t4
	t[3] = t[2] - 1; //t5
	t[4] = 1.0 / t[3];//t6
	t[5] = 1.0 / mesh->by; //t7
	t[6] = 1.0 / mesh->ax; //t8
	t[7] = material->nu - 1.0; //t9
	t[8] = mesh->by*t[1] * t[4] * t[6] * t[7] * (1.0 / 3.0); //t10
	t[9] = mesh->ax *t[1] * t[5] * (1.0 / (2.4 * pow(10, 1))); //t11
	t[10] = mesh->ax *t[1] * t[5] * (1.0 / (1.2 * pow(10, 1))); //t12
	t[11] = t[4] * (1.0 / (1.2 * pow(10, 1))); //t13
	t[12] = t[1] * (1.0 / (4.8 * pow(10, 1))); //t14
	t[13] = t[1] * (7.0 / (4.8 * pow(10, 1))); //t15
	t[14] = t[11] + t[13]; //t16
	t[15] = mesh->by*t[1] * t[6] * (1.0 / (1.2 * pow(10, 1))); //t17
	t[16] = t[11] + t[12]; //t18
	t[17] = mesh->ax*t[1] * t[4] * t[5] * t[7] * (1.0 / 6.0); //t19
	t[18] = mesh->ax*t[1] * t[4] * t[5] * t[7] * (1.0 / 3.0); //t20
	t[19] = -t[8] + t[9]; //t21
	t[20] = t[8] + t[10]; //t22
	t[21] = mesh->by*t[1] * t[4] * t[6] * t[7] * (1.0 / 6.0); //t23
	t[22] = -t[10] + t[21]; //t24
	t[23] = -t[11] - t[13]; //t25
	t[24] = -t[11] - t[12]; //t26
	t[25] = -t[15] + t[17]; //t27
	t[26] = t[15] + t[18]; //t28
	t[27] = mesh->by*t[1] * t[6] * (1.0 / (2.4*pow(10, 1))); //t29
	t[28] = -t[18] + t[27]; //t30
	t[29] = -t[9] - t[21]; //t31
	t[30] = -t[17] - t[27]; //t32

}

void populate_stiff_mat(double *t, double *ke, struct mesh *mesh ) {
	int row_index=0,col_index=0;
	ke[row_index + col_index] = t[20]; col_index += 8;
	ke[row_index + col_index] = t[24]; col_index += 8;
	ke[row_index + col_index] = t[19]; col_index += 8;
	ke[row_index + col_index] = t[23]; col_index += 8;
	ke[row_index + col_index] = t[29]; col_index += 8;
	ke[row_index + col_index] = t[16]; col_index += 8;
	ke[row_index + col_index] = t[22]; col_index += 8;
	ke[row_index + col_index] = t[14]; col_index = 0; row_index++;
	
	ke[row_index + col_index] = t[1]*(-1.0/(4.8*pow(10,1)))-t[4]*(1.0/(1.2*pow(10,1))); col_index += 8;
	ke[row_index + col_index] = t[26]; col_index += 8;
	ke[row_index + col_index] = t[14]; col_index += 8;
	ke[row_index + col_index] = t[25]; col_index += 8;
	ke[row_index + col_index] = t[16]; col_index += 8;
	ke[row_index + col_index] = t[30]; col_index += 8;
	ke[row_index + col_index] = t[23]; col_index += 8;
	ke[row_index + col_index] = t[28]; col_index = 0; row_index++;
	
	ke[row_index + col_index] = t[19]; col_index += 8;
	ke[row_index + col_index] = t[14]; col_index += 8;
	ke[row_index + col_index] = t[20]; col_index += 8;
	ke[row_index + col_index] = t[16]; col_index += 8;
	ke[row_index + col_index] = t[22]; col_index += 8;
	ke[row_index + col_index] = t[23]; col_index += 8;
	ke[row_index + col_index] = t[29]; col_index += 8;
	ke[row_index + col_index] = t[24]; col_index = 0; row_index++;
	
	ke[row_index + col_index] = t[1] * (-7.0 / (4.8*pow(10, 1))) - t[4] * (1.0 / (1.2*pow(10, 1))); col_index += 8;
	ke[row_index + col_index] = t[25]; col_index += 8;
	ke[row_index + col_index] = t[16]; col_index += 8;
	ke[row_index + col_index] = t[26]; col_index += 8;
	ke[row_index + col_index] = t[14]; col_index += 8;
	ke[row_index + col_index] = t[28]; col_index += 8;
	ke[row_index + col_index] = t[24]; col_index += 8;
	ke[row_index + col_index] = t[30]; col_index = 0; row_index++;
	
	ke[row_index + col_index] = -t[9] - mesh->by*t[1]*t[4]*t[6]*t[7]*(1.0/6.0); col_index += 8;
	ke[row_index + col_index] = t[16]; col_index += 8;
	ke[row_index + col_index] = t[22]; col_index += 8;
	ke[row_index + col_index] = t[14]; col_index += 8;
	ke[row_index + col_index] = t[20]; col_index += 8;
	ke[row_index + col_index] = t[24]; col_index += 8;
	ke[row_index + col_index] = t[19]; col_index += 8;
	ke[row_index + col_index] = t[23]; col_index = 0; row_index++;
	
	ke[row_index + col_index] = t[16]; col_index += 8;
	ke[row_index + col_index] = - t[17] - mesh->by * t[1] * t[6] * (1.0/(2.4*pow(10,1))); col_index += 8;
	ke[row_index + col_index] = t[23]; col_index += 8;
	ke[row_index + col_index] = t[28]; col_index += 8;
	ke[row_index + col_index] = t[24]; col_index += 8;
	ke[row_index + col_index] = t[26]; col_index += 8;
	ke[row_index + col_index] = t[14]; col_index += 8;
	ke[row_index + col_index] = t[25]; col_index = 0; row_index++;
	
	ke[row_index + col_index] = t[22]; col_index += 8;
	ke[row_index + col_index] = t[23]; col_index += 8;
	ke[row_index + col_index] = t[29]; col_index += 8;
	ke[row_index + col_index] = t[24]; col_index += 8;
	ke[row_index + col_index] = t[19]; col_index += 8;
	ke[row_index + col_index] = t[14]; col_index += 8;
	ke[row_index + col_index] = t[20]; col_index += 8;
	ke[row_index + col_index] = t[16]; col_index = 0; row_index++;

	ke[row_index + col_index] = t[14]; col_index += 8;
	ke[row_index + col_index] = t[28]; col_index += 8;
	ke[row_index + col_index] = t[24]; col_index += 8;
	ke[row_index + col_index] = t[30]; col_index += 8;
	ke[row_index + col_index] = t[23]; col_index += 8;
	ke[row_index + col_index] = t[25]; col_index += 8;
	ke[row_index + col_index] = t[16]; col_index += 8;
	ke[row_index + col_index] = t[26]; col_index += 8;
}

void init_mass_t_array(double *t, struct mesh *mesh, struct material *material) {
	t[0] = mesh->ax*mesh->by*(2.0 / 9.0);
	t[1] = mesh->ax*mesh->by*(4.0 / 9.0);
	t[2] = mesh->ax*mesh->by*(1.0 / 9.0);

}

void populate_mass_mat(double *t, double *me) {
	int row_index = 0, col_index = 0;
	me[row_index + col_index] = t[1]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[2]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index = 0; row_index++;

	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[1]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[2]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index = 0; row_index++;

	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[1]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[2]; col_index += 8;
	me[row_index + col_index] = 0; col_index = 0; row_index++;

	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[1]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[2]; col_index = 0; row_index++;

	me[row_index + col_index] = t[2]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[1]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index = 0; row_index++;

	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[2]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[1]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index = 0; row_index++;

	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[2]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[1]; col_index += 8;
	me[row_index + col_index] = 0; col_index = 0; row_index++;

	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[2]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[0]; col_index += 8;
	me[row_index + col_index] = 0; col_index += 8;
	me[row_index + col_index] = t[1]; col_index += 8;
}



