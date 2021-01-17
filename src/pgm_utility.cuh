#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

struct image{
	int col;
	int row;
	int max;
	int *data;
};


int write_image(char *filename, int n, struct mesh *mesh, double *xphys);
