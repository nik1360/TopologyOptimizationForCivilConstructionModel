#pragma once

struct mesh {
	double lx;	//mesh x lenght
	double ly; // mesh y lenght
	double vmax; //mesh max volume
	int nelx; //number of elements on x
	int nely; // number of elements on y
	double volfrac;
	double ax; //element x size
	double by; //element y size
	double area;//element area
	double penal; //Young penalization
	double prho; //mass penalization
	double rmin; //filter radius
	int ft; //1->sensitivity filtering, 2-> density filtering

	double alpha;
	double beta;
	int ninp;
	int nout;

	int fixed_count;
};
