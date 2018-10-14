#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#define N 5

int A[N][N],B[N][N],C[N][N];

typedef struct{
	int n;				/* The number of processors in a row (column). */
	int size;   		/* Number of processors. (Size = N*N*/
	int row;			/* This processor's row number.*/
	int col;			/* This processor's column number.*/
	int MyRank;   		/* This processor's unique identifier.*/
	MPI_Comm comm;     	/* Communicator for all processors in the grid.*/
	MPI_Comm row_comm; 	/* All processors in this processor's row   .  */
	MPI_Comm col_comm; 	/* All processors in this processor's column.  */
}grid_info;

void populate_arrays()
{
	for(int i=0;i<N;++i)
		for(int j=0;j<N;++j)
		{
			A[i][j] = i*N + j + 1;
			B[i][j] = i*N + j + 1;
			C[i][j] = 0;
		}
}

void print_matrix(int a[][N])
{
	for(int i=0;i<N;++i)
	{
		for(int j=0;j<N;++j)
			printf("%d ",a[i][j] );
		printf("\n");
	}
}


int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

int main(int argc, char const *argv[])
{
	int rank,world_size,namelen;
	char name[30];

	MPI_Init(NULL,NULL);

	MPI_Comm_size(MPI_COMM_WORLD,&world_size);

	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	MPI_Get_processor_name(name,&namelen);

	//Creation of the Grid of processes
	grid_info *grid = (grid_info*)malloc(sizeof(grid_info));
	grid->MyRank = rank;

	int dims[2] = {N,1};
	int periods[2] = {1,1};
	MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,0,&(grid->comm));

	
	int coordinates[2];
	MPI_Cart_coords(grid->comm, grid->MyRank, 2, coordinates);

	grid->row = coordinates[0];
	grid->col = coordinates[1];

	if(rank == 0)
	{
		populate_arrays();
		printf("Operand matrix A\n");
		print_matrix(A);
		printf("Operand matrix B\n");
		print_matrix(B);
	}

	MPI_Comm sub_grid_comm;

	int remain_dims[2];
	remain_dims[0] = 1,remain_dims[1] = 0;
	MPI_Cart_sub(grid->comm,remain_dims,&(grid->row_comm));
	remain_dims[0] = 0,remain_dims[1] = 1;
	MPI_Cart_sub(grid->comm,remain_dims,&(grid->col_comm));

	int recv_dataA,recv_dataB;
	MPI_Scatter(A,1,MPI_INT,&recv_dataA,1,MPI_INT,0,grid->comm);
	MPI_Scatter(B,1,MPI_INT,&recv_dataB,1,MPI_INT,0,grid->comm);

	//printf("(%d %d) %d \n",grid->row,grid->col, recv_dataA);

	//Skewing
	MPI_Status status;
	if(grid->row != 0)
	{
		//printf("Rank %d (%d,%d)%d %d\n",rank,grid->row,grid->col,N*grid->row + mod(grid->col - grid->row,N),N*grid->row + mod(grid->col + grid->row,N) );
		MPI_Sendrecv_replace(&recv_dataA,1,MPI_INT,N*grid->row + mod(grid->col - grid->row,N)
			,0,N*grid->row +mod(grid->col + grid->row,N),0,grid->comm,&status);
	}
	if(grid->col != 0)
	{
		//printf("Rank %d (%d,%d)%d %d\n",rank,grid->row,grid->col,mod(grid->row - grid->col,N)*N + grid->col,mod(grid->row + grid->col,N)*N + grid->col);
		
		MPI_Sendrecv_replace(&recv_dataB,1,MPI_INT,mod(grid->row - grid->col,N)*N + grid->col,
			0,mod(grid->row + grid->col,N)*N + grid->col,0,grid->comm,&status);	
	}

	int sum = recv_dataA*recv_dataB;
	
	int num_steps = N - 1;
	while(num_steps--)
	{
		//Multiply and add
		
		//Rotation
		MPI_Sendrecv_replace(&recv_dataA,1,MPI_INT,N*grid->row + mod(grid->col - 1,N)
			,0,N*grid->row +mod(grid->col + 1,N),0,grid->comm,&status);
		
		MPI_Sendrecv_replace(&recv_dataB,1,MPI_INT,mod(grid->row - 1,N)*N + grid->col,
			0,mod(grid->row + 1,N)*N + grid->col,0,grid->comm,&status);	

		sum += recv_dataA*recv_dataB;
	}
	
	MPI_Gather(&sum,1,MPI_INT,C,1,MPI_INT,0,grid->comm);
	if(rank == 0)
	{
		printf("Resultant matrix C\n");
		for(int i=0;i<N;++i)
		{
			for(int j=0;j<N;++j)
				printf("%d ",C[i][j] );
			printf("\n");
		}
	}
	MPI_Finalize();

	return 0;
}
