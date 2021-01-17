#include "pgm_utility.cuh"
#include "mesh.cuh"


int write_image(char *filename, int n, struct mesh *mesh, double *xphys){
	
	struct image img;
	int img_index;
	FILE *fout;
	int i, npixel;

	img.row = mesh->nely;
	img.col = mesh->nelx;
	img.max = 255;
	img.data = (int*)malloc(n * sizeof(int));
	img_index = 0;

	for (int col_index = 0;col_index < mesh->nelx;col_index++) {
		for (int row_index = 0;row_index < mesh->nely;row_index++) {

			if ((int)((1.0f - xphys[img_index]) * img.max) > img.max) {
				img.data[row_index*mesh->nelx + col_index] = img.max;
			}
			else {
				img.data[row_index*mesh->nelx + col_index] = (int)((1.0f - xphys[img_index]) * img.max);
			}
			img_index++;

		}
	}
	
	if((fout=fopen(filename,"w"))!=NULL){
		fprintf(fout,"P2\n");
		fprintf(fout,"#Creato con C\n");
		fprintf(fout,"%d %d\n",img.col,img.row);
		fprintf(fout,"%d\n",img.max);
		npixel=img.col*img.row;
		for(i=0;i<npixel;i++){
			fprintf(fout,"%d\n",img.data[i]);
		}
		fclose(fout);
		free(img.data);
	}else{
		printf("\nImpossibile creare il file %s\n",filename);
		free(img.data);
		return -1;
	}
	
	return 0;
}



