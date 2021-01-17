#include "minmax.cuh"

double minVal(double a, double b) {
	if (a < b)
		return a;
	else
		return b;
}

double maxVal(double a, double b) {
	if (a > b)
		return a;
	else
		return b;
}