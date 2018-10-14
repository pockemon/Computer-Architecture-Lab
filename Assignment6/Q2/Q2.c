#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10000000
// Initialize vectors
void fill_arrays(double X[], double Y[])
{
	int i;
	for(i = 0; i < N; i++)
	{
		X[i] = i;
		Y[i] = 2 * i;
	}	
}

// DAXPY computation
void compute_DAXPY(double X[], double Y[], int n)
{
	int i;
	double a = 1.0;
	for(i = 0; i < n; i++)
		X[i] = a * X[i] + Y[i];
}

int main()
{	
	// Initialize MPI
	MPI_Init(NULL, NULL);

	// Get rank and size  
	int rank, world_size, size_per_process,namelen;
	char name[30];
	double *X, *Y, *sub_X, *sub_Y, start, end, parallel_time, serial_time;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Get_processor_name(name,&namelen);

	// Process 0 performs serial execution of DAXPY and initializes vectors for parallel execution
	if(rank == 0)
	{
		
		X = (double*) malloc(sizeof(double) * (N));
		Y = (double*) malloc(sizeof(double) * (N));
		fill_arrays(X, Y);
		
		// Serial section
		start = MPI_Wtime();
		compute_DAXPY(X, Y, N);
		end = MPI_Wtime();
		serial_time = end - start;

		// For parallel section
		fill_arrays(X, Y);
		start = MPI_Wtime();
	}

	// Setup variables for scatter-gather
	size_per_process = (N) / world_size;
	sub_X = (double*)malloc(sizeof(double) * size_per_process);
	sub_Y = (double*)malloc(sizeof(double) * size_per_process);

	// Scatter X and Y
	MPI_Scatter(X, size_per_process, MPI_DOUBLE, sub_X, size_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(Y, size_per_process, MPI_DOUBLE, sub_Y, size_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Compute DAXPY with the subset of X and Y
	compute_DAXPY(sub_X, sub_Y, size_per_process);

	// Gather results back in X
	MPI_Gather(sub_X, size_per_process, MPI_DOUBLE, X, size_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Process 0 prints results
	if(rank == 0)
	{
		end = MPI_Wtime();
		parallel_time = end - start;
		printf("Time taken by %d processes : %lf seconds\n", world_size, parallel_time);
		printf("Speedup for %d processes : %lf\n", world_size, parallel_time / serial_time);
		
		free(X); 
		free(Y);
	}

	MPI_Finalize();
	return 0;
}
