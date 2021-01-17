#pragma once

//MESH 
#define LX 24
#define LY 8
#define NELX 24
#define NELY 8
#define PENAL 3.0
#define PRHO 1.0
#define VOLFRAC 0.45
#define RMIN 2.0
#define FT 2	//not working if ft=1, why?
#define ALPHA 0.15
#define BETA 0.15
#define NINP 1
#define NOUT 1

//MATERIAL
#define E0 10
#define RHO0 0.01
#define NU 0.3

//STTIFFNES MATRIX
#define KEROW 8
#define KECOL 8

//MASS MATRIX
#define MEROW 8
#define MECOL 8

#define ISPETTRO 0
#define IVOLUME 0 //MUST BE 0!

//OTHERS
#define EDOFMAT_COL 8

