#pragma once

struct material {
	double e0; //material young module
	double emin; // void area young module
	double rho0;//material density(?)
	double rhomin;//void area density(?)
	double nu; //poisson coeff
};
