#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>

int main()
{
	MPI_Init(NULL, NULL);
	int world_size, rank, i;
	float *A, *B, buff;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (rank == 0)
	{
		A = (float *)malloc(sizeof(float) * world_size);
		B = (float *)malloc(sizeof(float) * world_size);
		printf("The old array: \n");
		for (i = 0; i < world_size; i++)
		{
			A[i] = (i + 1)*(i + 1);
			printf("%f ", A[i]);
		}

	}
	int *displs = (int *)malloc(sizeof(int) * world_size);
	int *sendcounts = (int *)malloc(sizeof(int) * world_size);
	for (i = 0; i < world_size; i++)
	{
		sendcounts[i] = 1;
		displs[i] = i;
	}

	//Scattering the array non-uniformly across the processes
	MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT, &buff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	float root = sqrt(buff);

	//Calculating root of every element
	MPI_Gatherv(&root, 1, MPI_FLOAT, B, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//Printing the new array
	if (rank == 0)
	{
		printf("\nThe new array:\n");
		for(i = 0; i < world_size; i++)
			printf("%f ", B[i]);
		printf("\n");
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
