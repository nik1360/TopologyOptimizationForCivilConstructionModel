
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "main_functions.cuh"
#include "constants.cuh"
#include "mesh.cuh"
#include "matrix_functions.cuh"
#include "material.cuh"
#include "sys.cuh"
#include "parallel.cuh"


void g_sys_init(struct sys *sys, int ndof, int nstate, struct sparse *a,struct sparse *b, struct sparse *c,  double dstato, struct sparse *e) {

	sys->a.val = (double*)malloc((int)(2 * pow(ndof, 2) + ndof) * sizeof(double));
	sys->a.row = (int*)malloc((int)(2 * pow(ndof, 2) + ndof) * sizeof(int));
	sys->a.col = (int*)malloc((int)(2 * pow(ndof, 2) + ndof) * sizeof(int));
	sys->a.csr_row = (int*)malloc(sizeof(int));  
	sys->a.nnz = a->nnz;
	sys->a.nrow = a->nrow;
	sys->a.ncol = a->ncol;
	for (int i = 0;i < a->nnz;i++) {
		sys->a.val[i] = a->val[i];
		sys->a.row[i] = a->row[i];
		sys->a.col[i] = a->col[i];
		//printf("(%d,%d) %f\n", sys->a_row[i],sys->a_col[i],sys->a[i]);
	}

	sys->b.val = (double*)malloc(ndof * sizeof(double));
	sys->b.row = (int*)malloc(ndof * sizeof(int));
	sys->b.col = (int*)malloc(ndof * sizeof(int));
	sys->b.csr_row = (int*)malloc(sizeof(int));
	sys->b.nnz = b->nnz;
	sys->b.nrow = b->nrow;
	sys->b.ncol = b->ncol;
	for (int i = 0;i < b->nnz;i++) {
		sys->b.val[i] = b->val[i];
		sys->b.row[i] = b->row[i];
		sys->b.col[i] = b->col[i];
		//printf("(%d,%d) %f\n", sys->b_row[i], sys->b_col[i], sys->b[i]);
	}

	sys->c.val = (double*)malloc(nstate * sizeof(double));
	sys->c.row = (int*)malloc(nstate * sizeof(int));
	sys->c.col = (int*)malloc(nstate * sizeof(int));
	sys->c.csr_row = (int*)malloc(sizeof(int));
	sys->c.nnz = c->nnz;
	sys->c.nrow = c->nrow;
	sys->c.ncol = c->ncol;
	for (int i = 0;i < c->nnz;i++) {
		sys->c.val[i] = c->val[i];
		sys->c.row[i] = c->row[i];
		sys->c.col[i] = c->col[i];
		//printf("(%d,%d) %f\n", sys->c_row[i], sys->c_col[i], sys->c[i]);
	}

	sys->d = 0;

	sys->e.val = (double*)malloc((int)(pow(ndof, 2) + ndof) * sizeof(double));
	sys->e.row = (int*)malloc((int)(pow(ndof, 2) + ndof) * sizeof(int));
	sys->e.col = (int*)malloc((int)(pow(ndof, 2) + ndof) * sizeof(int));
	sys->e.csr_row = (int*)malloc(sizeof(int));
	sys->e.nnz = e->nnz;
	sys->e.nrow = e->nrow;
	sys->e.ncol = e->ncol;
	for (int i = 0;i < e->nnz;i++) {
		sys->e.val[i] = e->val[i];
		sys->e.row[i] = e->row[i];
		sys->e.col[i] = e->col[i];
		//printf("(%d,%d) %f\n", sys->e_row[i], sys->e_col[i], sys->e[i]);
	}

}

void gec8_sys_init(struct sys *sys) {
	sys->a.nrow = 2;
	sys->a.ncol = 2;
	sys->a.nnz = 3;
	sys->a.val = (double*)malloc(sys->a.nnz * sizeof(double));
	sys->a.row = (int*)malloc(sys->a.nnz * sizeof(int));
	sys->a.col = (int*)malloc(sys->a.nnz * sizeof(int));
	sys->a.csr_row = (int*)malloc(sizeof(int));
	sys->a.row[0] = 1; sys->a.col[0] = 1; sys->a.val[0] = -2.1;
	sys->a.row[1] = 2; sys->a.col[1] = 1; sys->a.val[1] = 1;
	sys->a.row[2] = 1; sys->a.col[2] = 2; sys->a.val[2] = -0.2;

	sys->b.nrow = 2;
	sys->b.ncol = 1;
	sys->b.nnz = 1;
	sys->b.val = (double*)malloc(sys->b.nnz * sizeof(double));
	sys->b.row = (int*)malloc(sys->b.nnz * sizeof(int));
	sys->b.col = (int*)malloc(sys->b.nnz * sizeof(int));
	sys->b.csr_row = (int*)malloc(sizeof(int));
	sys->b.row[0] = 1; sys->b.col[0] = 1; sys->b.val[0] = 1;


	sys->c.nrow = 1;
	sys->c.ncol = 2;
	sys->c.nnz = 1;
	sys->c.val = (double*)malloc(sys->c.nnz * sizeof(double));
	sys->c.row = (int*)malloc(sys->c.nnz * sizeof(int));
	sys->c.col = (int*)malloc(sys->c.nnz * sizeof(int));
	sys->c.csr_row = (int*)malloc(sizeof(int));
	sys->c.row[0] = 1; sys->c.col[0] = 1; sys->c.val[0] = 20;

	sys->d = 0;

	sys->e.nrow = 2;
	sys->e.ncol = 2;
	sys->e.nnz = 2;
	sys->e.val = (double*)malloc(sys->e.nnz * sizeof(double));
	sys->e.row = (int*)malloc(sys->e.nnz * sizeof(int));
	sys->e.col = (int*)malloc(sys->e.nnz * sizeof(int));
	sys->e.csr_row = (int*)malloc(sizeof(int));
	sys->e.row[0] = 1; sys->e.col[0] = 1; sys->e.val[0] = 1;
	sys->e.row[1] = 2; sys->e.col[1] = 2; sys->e.val[1] = 1;

}

void sys_multiplication(struct sys *sys1, struct sys *sys2, struct sys *sys_p) {
	
	struct sparse tmp;
	int count ;

	sparse_matrix_product(&sys1->b, &sys2->c,&tmp);	//B1*C2
	

	//------building sys_p A Matrix-----------
	sys_p->a.nrow = sys1->a.nrow + sys2->a.nrow;
	sys_p->a.ncol = sys1->a.ncol + sys2->a.ncol;
	sys_p->a.nnz = sys1->a.nnz + sys2->a.nnz + tmp.nnz;
	sys_p->a.val = (double*)malloc(sys_p->a.nnz * sizeof(double));
	sys_p->a.row = (int*)malloc(sys_p->a.nnz * sizeof(int));
	sys_p->a.col = (int*)malloc(sys_p->a.nnz * sizeof(int));

	count = 0;
	for (int i = 0;i < sys1->a.nnz;i++) {
		sys_p->a.val[count] = sys1->a.val[i];
		sys_p->a.row[count] = sys1->a.row[i];
		sys_p->a.col[count] = sys1->a.col[i];
		count++;
	}
	for (int i = 0;i < tmp.nnz;i++) {
		sys_p->a.val[count] =tmp.val[i];
		sys_p->a.row[count] = tmp.row[i];
		sys_p->a.col[count] = tmp.col[i]+sys1->a.ncol;
		count++;
	}
	for (int i = 0;i < sys2->a.nnz;i++) {
		sys_p->a.val[count] = sys2->a.val[i];
		sys_p->a.row[count] = sys2->a.row[i]+sys1->a.nrow;
		sys_p->a.col[count] = sys2->a.col[i] + sys1->a.ncol;
		count++;
	}
	free(tmp.row);
	free(tmp.col);
	free(tmp.val);

	//------building sys_p B Matrix-----------
	sys_p->b.nrow = sys1->b.nrow + sys2->b.nrow;
	sys_p->b.ncol = sys1->b.ncol ;
	sys_p->b.nnz = sys2->b.nnz;
	sys_p->b.val = (double*)malloc(sys_p->b.nnz * sizeof(double));
	sys_p->b.row = (int*)malloc(sys_p->b.nnz * sizeof(int));
	sys_p->b.col = (int*)malloc(sys_p->b.nnz * sizeof(int));

	count = 0;
	for (int i = 0;i < sys2->b.nnz;i++) {
		sys_p->b.val[count] = sys2->b.val[i];
		sys_p->b.row[count] = sys2->b.row[i] + sys1->b.nrow;
		sys_p->b.col[count] = sys2->b.col[i];
		count++;
	}

	//------building sys_p C Matrix-----------
	sys_p->c.nrow = sys1->c.nrow;
	sys_p->c.ncol = sys1->c.ncol + sys2->c.ncol;
	sys_p->c.nnz = sys1->c.nnz;
	sys_p->c.val = (double*)malloc(sys_p->c.nnz * sizeof(double));
	sys_p->c.row = (int*)malloc(sys_p->c.nnz * sizeof(int));
	sys_p->c.col = (int*)malloc(sys_p->c.nnz * sizeof(int));

	count = 0;
	for (int i = 0;i < sys1->c.nnz;i++) {
		sys_p->c.val[count] = sys1->c.val[i];
		sys_p->c.row[count] = sys1->c.row[i];
		sys_p->c.col[count] = sys1->c.col[i];
		count++;
	}

	//------building sys_p D Matrix-----------
	sys_p->d = sys1->d*sys2->d;

	//------building sys_p E Matrix-----------
	sys_p->e.nrow = sys1->e.nrow+sys2->e.nrow;
	sys_p->e.ncol = sys1->e.ncol + sys2->e.ncol;
	sys_p->e.nnz = sys1->e.nnz+sys2->e.nnz;
	sys_p->e.val = (double*)malloc(sys_p->e.nnz * sizeof(double));
	sys_p->e.row = (int*)malloc(sys_p->e.nnz * sizeof(int));
	sys_p->e.col = (int*)malloc(sys_p->e.nnz * sizeof(int));

	count = 0;
	for (int i = 0;i < sys1->e.nnz;i++) {
		sys_p->e.val[count] = sys1->e.val[i];
		sys_p->e.row[count] = sys1->e.row[i];
		sys_p->e.col[count] = sys1->e.col[i];
		count++;
	}
	for (int i = 0;i < sys2->e.nnz;i++) {
		sys_p->e.val[count] = sys2->e.val[i];
		sys_p->e.row[count] = sys2->e.row[i]+sys1->e.nrow;
		sys_p->e.col[count] = sys2->e.col[i]+sys1->e.ncol;
		count++;
	}

}

void sys_inf_norm(struct sys *sys, double *norm, double *h_inf_peak_freq) {
	
	double omega=0,omega_peak=0, current_norm ,max_norm=0;
	struct sparse *tmp1;
	struct sparse *tmp2;
	struct sparse *e_tmp;
	struct sparse *ea_diff;
	
	double *ea_diff_full,*inv;
	
	tmp1 = (struct sparse*)malloc(sizeof(struct sparse));
	tmp2 = (struct sparse*)malloc(sizeof(struct sparse));
	e_tmp = (struct sparse*)malloc(sizeof(struct sparse));
	ea_diff = (struct sparse*)malloc(sizeof(struct sparse));

	inv = (double*)malloc(sys->e.nrow*sys->e.ncol * sizeof(double));
	

	e_tmp->nnz = sys->e.nnz;
	e_tmp->nrow = sys->e.nrow;
	e_tmp->ncol = sys->e.ncol;

	e_tmp->row = (int*)malloc(e_tmp->nnz * sizeof(int));
	e_tmp->col = (int*)malloc(e_tmp->nnz * sizeof(int));
	e_tmp->val = (double*)malloc(e_tmp->nnz * sizeof(double));

	omega = 0;
	while(omega<0.1) {		//0< omega< 1  step=0.2
		
		for (int j = 0; j < sys->e.nnz; j++) {				//sE
			e_tmp->row[j] = sys->e.row[j];
			e_tmp->col[j] = sys->e.col[j];
			e_tmp->val[j] = omega*sys->e.val[j];
		}

		sparse_diff(e_tmp, &sys->a, ea_diff);  //sE-A
		sparse_to_dense(ea_diff, &ea_diff_full);
		free(ea_diff->row);
		free(ea_diff->col);
		free(ea_diff->val);

		
		matrix_inverse(ea_diff_full, inv, sys->e.nrow); //inv(sE-A)
		free(ea_diff_full);

		dense_to_sparse(inv, sys->e.nrow, sys->e.ncol, ea_diff);
		sparse_matrix_product(&sys->c,ea_diff,tmp1); //C*inv(sE-A)
		free(ea_diff->row);
		free(ea_diff->col);
		free(ea_diff->val);

		sparse_matrix_product(tmp1, &sys->b,tmp2);//C*inv(sE-A)*B
		free(tmp1->row);
		free(tmp1->col);
		free(tmp1->val);


		if (tmp2->nnz==0)
			current_norm = 0;
		else 
			current_norm = tmp2->val[0];
		
		if (fabs(current_norm) > max_norm) {
			max_norm = fabs(current_norm);
			omega_peak = omega;
		}
		
		//printf("%f %f \n", current_norm, omega);
		omega=omega+0.02;
		
		free(tmp2->row);
		free(tmp2->col);
		free(tmp2->val);
		
		
	}
	(*norm) = max_norm;
	(*h_inf_peak_freq) = omega_peak;
	
	//printf("max: %f %f", (*norm), (*h_inf_peak_freq));
	
	free(e_tmp->row);
	free(e_tmp->col);
	free(e_tmp->val);
	free(inv);
	free(tmp1);
	free(tmp2);
	free(e_tmp);
	free(ea_diff);

	
}

void freqresp(struct sys *sys,double omega, double *value) {

	struct sparse *tmp1;
	struct sparse *tmp2;
	struct sparse *e_tmp;
	struct sparse *ea_diff;

	double *ea_diff_full, *inv;

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	tmp1 = (struct sparse*)malloc(sizeof(struct sparse));
	tmp2 = (struct sparse*)malloc(sizeof(struct sparse));
	e_tmp = (struct sparse*)malloc(sizeof(struct sparse));
	ea_diff = (struct sparse*)malloc(sizeof(struct sparse));

	inv = (double*)malloc(sys->e.nrow*sys->e.ncol * sizeof(double));

	e_tmp->nnz = sys->e.nnz;
	e_tmp->nrow = sys->e.nrow;
	e_tmp->ncol = sys->e.ncol;

	e_tmp->row = (int*)malloc(e_tmp->nnz * sizeof(int));
	e_tmp->col = (int*)malloc(e_tmp->nnz * sizeof(int));
	e_tmp->val = (double*)malloc(e_tmp->nnz * sizeof(double));
	
	for (int j = 0; j < sys->e.nnz; j++) {				//sE
		e_tmp->row[j] = sys->e.row[j];
		e_tmp->col[j] = sys->e.col[j];
		e_tmp->val[j] = omega * sys->e.val[j];
	}



	sparse_diff(e_tmp, &sys->a, ea_diff);  //sE-A

	free(e_tmp->row);
	free(e_tmp->col);
	free(e_tmp->val);

	sparse_to_dense(ea_diff, &ea_diff_full);

	matrix_inverse(ea_diff_full, inv, sys->e.nrow); //inv(sE-A)

	free(ea_diff->row);
	free(ea_diff->col);
	free(ea_diff->val);

	dense_to_sparse(inv, sys->e.nrow, sys->e.ncol, ea_diff);

	sparse_matrix_product(&sys->c, ea_diff, tmp1); //C*inv(sE-A)
	free(ea_diff->row);
	free(ea_diff->col);
	free(ea_diff->val);

	sparse_matrix_product(tmp1, &sys->b, tmp2);//C*inv(sE-A)*B
	free(tmp1->row);
	free(tmp1->col);
	free(tmp1->val);


	if (tmp2->nnz == 0)
		(*value) = 0;

	else
		(*value) = tmp2->val[0];
	free(tmp2->row);
	free(tmp2->col);
	free(tmp2->val);
	
	
	free(inv);
	free(ea_diff_full);

	free(tmp1);
	free(tmp2);
	free(ea_diff);
	free(e_tmp);

	cublasDestroy(cublas_handle);

}

void sys_equal(struct sys *sys1, struct sys *sys2) {
	//SYS1=SYS2

	sys1->a.nrow = sys2->a.nrow;
	sys1->a.ncol = sys2->a.ncol;
	sys1->a.nnz = sys2->a.nnz;
	
	sys1->a.row = (int*)malloc(sys1->a.nnz * sizeof(int));
	sys1->a.col = (int*)malloc(sys1->a.nnz * sizeof(int));
	sys1->a.val = (double*)malloc(sys1->a.nnz * sizeof(double));

	for (int i = 0; i < sys1->a.nnz; i++) {
		sys1->a.row[i] = sys2->a.row[i];
		sys1->a.col[i] = sys2->a.col[i];
		sys1->a.val[i] = sys2->a.val[i];
	}

	sys1->b.nrow = sys2->b.nrow;
	sys1->b.ncol = sys2->b.ncol;
	sys1->b.nnz = sys2->b.nnz;

	sys1->b.row = (int*)malloc(sys1->b.nnz * sizeof(int));
	sys1->b.col = (int*)malloc(sys1->b.nnz * sizeof(int));
	sys1->b.val = (double*)malloc(sys1->b.nnz * sizeof(double));

	for (int i = 0; i < sys1->b.nnz; i++) {
		sys1->b.row[i] = sys2->b.row[i];
		sys1->b.col[i] = sys2->b.col[i];
		sys1->b.val[i] = sys2->b.val[i];
	}

	sys1->c.nrow = sys2->c.nrow;
	sys1->c.ncol = sys2->c.ncol;
	sys1->c.nnz = sys2->c.nnz;

	sys1->c.row = (int*)malloc(sys1->c.nnz * sizeof(int));
	sys1->c.col = (int*)malloc(sys1->c.nnz * sizeof(int));
	sys1->c.val = (double*)malloc(sys1->c.nnz * sizeof(double));

	for (int i = 0; i < sys1->c.nnz; i++) {
		sys1->c.row[i] = sys2->c.row[i];
		sys1->c.col[i] = sys2->c.col[i];
		sys1->c.val[i] = sys2->c.val[i];
	}

	sys1->d = sys2->d;

	sys1->e.nrow = sys2->e.nrow;
	sys1->e.ncol = sys2->e.ncol;
	sys1->e.nnz = sys2->e.nnz;

	sys1->e.row = (int*)malloc(sys1->e.nnz * sizeof(int));
	sys1->e.col = (int*)malloc(sys1->e.nnz * sizeof(int));
	sys1->e.val = (double*)malloc(sys1->e.nnz * sizeof(double));

	for (int i = 0; i < sys1->e.nnz; i++) {
		sys1->e.row[i] = sys2->e.row[i];
		sys1->e.col[i] = sys2->e.col[i];
		sys1->e.val[i] = sys2->e.val[i];
	}

	
}

void free_sys(struct sys *sys) {
	free(sys->a.row);
	free(sys->a.col);
	free(sys->a.val);
	free(sys->a.csr_row);
	free(sys->b.row);
	free(sys->b.col);
	free(sys->b.val);
	free(sys->b.csr_row);
	free(sys->c.row);
	free(sys->c.col);
	free(sys->c.val);
	free(sys->c.csr_row);
	free(sys->e.row);
	free(sys->e.col);
	free(sys->e.val);
	free(sys->e.csr_row);
}