/*
 * Cell(i, j) = cell[i + j * n]
 */
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include"mt.h"

void h_DBG(int *A, int *B, int dim) {
	int i;
	double res = 0.0;
	for(i = 0; i < dim; i += 1) {
		res += (A[i] - B[i]) * (A[i] - B[i]);
	}
	printf("res %.1f\n", res);

}
void init_configuration(double *h_x, double *h_y, double h_L, int h_Np) {
	int i;

	for(i = 0; i < h_Np; i += 1) {
		h_x[i] = h_L * genrand_real2();
		h_y[i] = h_L * genrand_real2();
	}
}

void h_check_active(double *h_x, double *h_y, double h_L, int h_Np, int *h_active) {
	int i, j;
	double dx, dy, dr_square;
	double diameter_square = 1.0;

	for(i = 0; i < h_Np; i += 1) {
		h_active[i] = 0;
	}
	for(i = 0; i < h_Np; i += 1) {
		for(j = 0; j < i; j += 1) {
			dx = h_x[i] - h_x[j];
			if(dx > 0.5 * h_L) {
				dx -= h_L;
			} else if(dx < -0.5 * h_L) {
				dx += h_L;
			}
			dy = h_y[i] - h_y[j];
			if(dy > 0.5 * h_L) {
				dy -= h_L;
			} else if(dy < -0.5 * h_L) {
				dy += h_L;
			}

			dr_square = dx * dx + dy * dy;
			if(dr_square < diameter_square) {
				h_active[i] = 1;
				h_active[j] = 1;
			}
		}
	}
}

void h_check_active_with_list(double *h_x, double *h_y, double h_L, int h_Np, int *h_active, int *h_cell_list, int cell_per_axis, int N_per_cell) {
	int i, j, k;
	int x_c, y_c;
	int cell_id, N_in_cell;
	int pair_id;
	double dx, dy, dr_square;
	double diameter_square = 1.0;

	for(i = 0; i < h_Np; i += 1) {
		x_c = (int)(h_x[i] * (double)cell_per_axis / h_L);
		y_c = (int)(h_y[i] * (double)cell_per_axis / h_L);
		cell_id = x_c + y_c * cell_per_axis;
		N_in_cell = h_cell_list[cell_id * N_per_cell];
		for(j = 1; j <= N_in_cell; j += 1) {
			pair_id = h_cell_list[cell_id * N_per_cell + j];
			if(i == pair_id) {continue;}
			dx = h_x[i] - h_x[pair_id];
			if(dx < -0.5 * h_L) {
				dx += h_L;
			} else if(dx > 0.5 * h_L) {
				dx -= h_L;
			}
			dy = h_y[i] - h_y[pair_id];
			if(dy < -0.5 * h_L) {
				dy += h_L;
			} else if(dy > 0.5 * h_L) {
				dy -= h_L;
			}
			dr_square = dx * dx + dy * dy;
			if(diameter_square > dr_square) {
				h_active[i] = 1;
			}
		}
	}
}

void h_make_list(double *h_x, double *h_y, double h_L, int h_Np, int *h_cell_list, int cell_per_axis, int N_per_cell) {
	int i, j, k;
	int x_cell, y_cell;
	int x_next, y_next;
	int cell_id, N_in_cell;

	for(i = 0; i < cell_per_axis * cell_per_axis * N_per_cell; i += 1) {
		h_cell_list[i] = 0;
	}

	for(k = 0; k < h_Np; k += 1) {
		x_cell = (int)(h_x[k] * (double)cell_per_axis / h_L);
		y_cell = (int)(h_y[k] * (double)cell_per_axis / h_L);
		for(i = -1; i <= 1; i += 1) {
			x_next = x_cell + i;
			if(x_next < 0) {
				x_next += cell_per_axis;
			} else if(x_next >= cell_per_axis) {
				x_next -= cell_per_axis;
			}
			for(j = -1; j <= 1; j += 1) {
				y_next = y_cell + j;
				if(y_next < 0) {
					y_next += cell_per_axis;
				} else if (y_next >= cell_per_axis){
					y_next -= cell_per_axis;
				}
				cell_id = x_next + y_next * cell_per_axis;
				h_cell_list[cell_id * N_per_cell] += 1;
				N_in_cell = h_cell_list[cell_id * N_per_cell];
				if(N_in_cell >= N_per_cell) {printf(">>>cell list overrun<<<\n");}
				h_cell_list[cell_id * N_per_cell + N_in_cell] = k;
			}
		}
	}
}

int main(void) {
	//utility variables
	int i, j;
	clock_t start, end;
	int cell_per_axis;
	int N_per_cell;
	FILE *file;

	//variables in host
	double *h_x;
	double *h_y;
	double h_L;
	int *h_active;
	int *h_check_result;
	int h_Np;
	int *h_cell_list;
	int *h_active_DBG;

	//initialize
	//init_genrand(19970303);
	init_genrand((int)time(NULL));

	//--set variable
	printf("set variable");
	h_Np = 18000;
	h_L = 140.0;
	cell_per_axis = (int)(h_L / 11.0) + 1;//renew list every 5 steps
	N_per_cell = (h_Np * 11) / (cell_per_axis * cell_per_axis);
	printf("--end\n");
	printf("cell per axis:%d N_per_cell:%d\n", cell_per_axis, N_per_cell);

	//--allocate memory
	printf("alloc mem");
	h_x = (double *)calloc(h_Np, sizeof(double));
	if(h_x == NULL) {printf(">>>no mem h_x<<<\n");return 0;}
	h_y = (double *)calloc(h_Np, sizeof(double));
	if(h_y == NULL) {printf(">>>no mem h_y<<<\n");return 0;}
	h_active = (int *)calloc(h_Np, sizeof(int));
	if(h_active == NULL) {printf(">>>no mem h_active<<<\n");return 0;}
	h_active_DBG = (int *)calloc(h_Np, sizeof(int));
	if(h_active_DBG == NULL) {printf(">>>no mem h_active_DBG<<<\n");return 0;}
	h_cell_list = (int *)calloc(N_per_cell * cell_per_axis * cell_per_axis, sizeof(int));
	if(h_cell_list == NULL) {printf(">>>no mem h_cell_list<<<\n");return 0;}
	printf("--end\n");
	//--place particles
	printf("init conifguration");
	init_configuration(h_x, h_y, h_L, h_Np);
	printf("--end\n");

	//--make first acriveness array
	//----without list as reference
	start = clock();
	printf("check activeness without list");
	h_check_active(h_x, h_y, h_L, h_Np, h_active_DBG);
	printf("--end\n");
	end = clock();
	printf("straighforward:%d [ms]\n", (int)((end - start)*1000 /CLOCKS_PER_SEC ));
	//----make cell list
	start = clock();
	printf("make cell list");
	h_make_list(h_x, h_y, h_L, h_Np, h_cell_list, cell_per_axis, N_per_cell);
	printf("--end\n");
	//----with list
	printf("check activeness with list");
	h_check_active_with_list(h_x, h_y, h_L, h_Np, h_active, h_cell_list, cell_per_axis, N_per_cell);
	printf("--end\n");
	end = clock();
	printf("cell_list host:%d [ms]\n", (int)((end - start)*1000 /CLOCKS_PER_SEC ));

	h_DBG(h_active, h_active_DBG, h_Np);

	//move particles

	//finalize
	//--free memory
	free(h_x);
	free(h_y);
	free(h_active);
	free(h_active_DBG);
	free(h_cell_list);
 
	return 0;
}
